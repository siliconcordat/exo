"""Integration tests for L5 MdeClient, L3 WarpModeldBridge, and Exo CLI."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socketserver
import tempfile
import threading
import unittest
from contextlib import redirect_stdout

from exo.bridge import (
    build_warp_adapter_bundle,
    build_warp_bundles_for_route,
    scale_to_rational,
    translate_weft_certificate_to_warp,
    translate_weft_checkpoint_to_warp,
)
from exo.cli import main as cli_main
from exo.client import (
    InMemoryJsonRpcTransport,
    MDE_META_KEY,
    MdeClient,
    MdeProtocolError,
    SPLICE_META_KEY,
    StdioJsonRpcTransport,
    WarpLoadError,
    WarpModeldBridge,
    WarpUnixBridge,
)
from exo.gate import ActionEnvelope, ApprovalGate
from exo.ledger import AppendOnlyLedger, canonical_json_bytes
from exo.policy import corpus_repo_policy, siliconcordat_repo_policy
from exo.router import RouteRequest, ZoneRouter
from exo.sim import SIMMemory, SIMRuleRecord, record_baseline_probe_receipt


class TestMdeClient(unittest.TestCase):
    def test_read_before_write_and_exact_byte_approval_with_vendor_meta(self) -> None:
        def mock_server(req: dict) -> dict:
            name = req["params"]["name"]
            args = req["params"]["arguments"]
            if name == "splice_locate":
                payload = {
                    "path": args["path"],
                    "zone": "code",
                    "file_read_token": "tok-abc",
                }
            elif name == "splice_trace":
                payload = {
                    "path": args["path"],
                    "symbol": args.get("symbol"),
                    "file_read_token": "tok-trace",
                }
            elif name == "splice_patch":
                payload = {"packet_id": "pkt-001", "status": "spliced", "edits_count": len(args["edits"])}
            elif name == "splice_verify":
                payload = {"bundle_id": "bnd-001", "packet_id": args["packet_id"], "eval_count": args.get("eval_count")}
            elif name == "splice_project":
                payload = {
                    "packet_id": args["packet_id"],
                    "bundle_id": args["bundle_id"],
                    "approve_diff_sha256": args.get("approve_diff_sha256"),
                    "status": "projected",
                }
            else:
                payload = {"ok": True}
            return {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }

        ledger = AppendOnlyLedger()
        gate = ApprovalGate(signing_key=b"client-key", ledger=ledger)
        transport = InMemoryJsonRpcTransport(mock_server)
        client = MdeClient(transport, gate=gate, ledger=ledger, actor_id="exo-test")

        # 1. Calling splice_patch before splice_locate fails closed client-side
        with self.assertRaises(MdeProtocolError) as ctx:
            client.splice_patch(
                capability_id="cap.math",
                file_path="src/app/calc.py",
                edits=[{"byte_range": [0, 5], "new_bytes": "x = 2"}],
                seam_contract={"effects": []},
            )
        self.assertIn("ERR_PRECONDITION_READ_REQUIRED", str(ctx.exception))
        self.assertEqual(len(transport.sent_requests), 0)

        # 2. After splice_locate (structuredContent + file_read_token), routine patch succeeds
        loc = client.splice_locate("src/app/calc.py", capability_id="cap.math")
        self.assertEqual(loc["file_read_token"], "tok-abc")
        self.assertEqual(transport.sent_requests[-1]["params"]["arguments"]["path"], "src/app/calc.py")

        tr = client.splice_trace("src/app/calc.py", symbol="calc")
        self.assertEqual(tr["symbol"], "calc")

        res = client.splice_patch(
            capability_id="cap.math",
            file_path="src/app/calc.py",
            edits=[{"byte_range": [0, 5], "new_bytes": "x = 2"}],
            seam_contract={"effects": []},
        )
        self.assertEqual(res["packet_id"], "pkt-001")
        patch_req = transport.sent_requests[-1]
        self.assertEqual(SPLICE_META_KEY, "gg.siliconcordat/splice")
        self.assertEqual(MDE_META_KEY, "gg.siliconcordat/mde")
        self.assertIn(SPLICE_META_KEY, patch_req["params"]["_meta"])
        self.assertIn("edits", patch_req["params"]["arguments"])

        ver = client.splice_verify("pkt-001", eval_count=2)
        self.assertEqual(ver["bundle_id"], "bnd-001")
        proj = client.splice_project("pkt-001", "bnd-001", approve_diff_sha256="deadbeef")
        self.assertEqual(proj["status"], "projected")
        self.assertEqual(proj["approve_diff_sha256"], "deadbeef")

        # 3. High-risk protected path requires exact-byte HumanApproval
        client.splice_locate("src/exo/gate/rules.py", capability_id="cap.gate")
        edits = [{"byte_range": [0, 1], "new_bytes": "#"}]
        seam = {"effects": ["gate"]}
        with self.assertRaises(MdeProtocolError) as ctx2:
            client.splice_patch(
                capability_id="cap.gate",
                file_path="src/exo/gate/rules.py",
                edits=edits,
                seam_contract=seam,
            )
        self.assertIn("require_human_approval", str(ctx2.exception))

        # Sign exact challenge and retry -> succeeds
        payload = canonical_json_bytes({
            "capability_id": "cap.gate",
            "file_path": "src/exo/gate/rules.py",
            "edits": edits,
            "seam_contract": seam,
            "allow_oracle_write": False,
        })
        challenge = gate.evaluate(
            ActionEnvelope(
                action_kind="splice_patch",
                target="src/exo/gate/rules.py",
                paths=("src/exo/gate/rules.py",),
                payload_bytes=payload,
                read_tokens=("tok-abc",),
                metadata={"capability_id": "cap.gate", "allow_oracle_write": False},
            )
        ).challenge
        assert challenge is not None
        approval = gate.sign_approval(challenge, approver_id="owner")
        res2 = client.splice_patch(
            capability_id="cap.gate",
            file_path="src/exo/gate/rules.py",
            edits=edits,
            seam_contract=seam,
            approval=approval,
        )
        self.assertEqual(res2["status"], "spliced")

    def test_mcp_is_error_true_raises_protocol_error(self) -> None:
        def refusing_server(req: dict) -> dict:
            err = {"code": "ERR_SYNTAX_ERROR_NODE", "message": "Refused splice inside ERROR subtree"}
            return {
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err)}],
                    "structuredContent": err,
                    "isError": True,
                },
            }

        client = MdeClient(InMemoryJsonRpcTransport(refusing_server), actor_id="exo-test")
        with self.assertRaises(MdeProtocolError) as ctx:
            client.splice_locate("src/broken.py")
        self.assertIn("isError=true", str(ctx.exception))
        self.assertIn("ERR_SYNTAX_ERROR_NODE", str(ctx.exception))

    def test_stdio_jsonrpc_transport_against_live_subprocess(self) -> None:
        import sys

        server_script = (
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "  req = json.loads(line)\n"
            "  name = req['params']['name']\n"
            "  payload = {'path': 'src/app.py', 'zone': 'code', 'file_read_token': 'stdio-tok'} if name == 'splice_locate' else {'packet_id': 'pkt-stdio', 'status': 'spliced'}\n"
            "  res = {'content': [{'type': 'text', 'text': json.dumps(payload)}], 'structuredContent': payload, 'isError': False}\n"
            "  sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': res}) + '\\n')\n"
            "  sys.stdout.flush()\n"
        )
        with StdioJsonRpcTransport([sys.executable, "-c", server_script]) as transport:
            client = MdeClient(transport, actor_id="exo-stdio-test")
            loc = client.splice_locate("src/app.py", capability_id="cap.stdio")
            self.assertEqual(loc["file_read_token"], "stdio-tok")
            patched = client.splice_patch(
                capability_id="cap.stdio",
                file_path="src/app.py",
                edits=[{"byte_range": [0, 1], "new_bytes": "a"}],
                seam_contract={"effects": []},
            )
            self.assertEqual(patched["packet_id"], "pkt-stdio")


class TestWarpModeldBridge(unittest.TestCase):
    def _start_mock_warp(self, mode: str) -> tuple[ThreadingHTTPServer, list[dict]]:
        events: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                events.append({"path": self.path, "body": body})
                if self.path == "/rollback":
                    resp = {"status": "rolled_back", "flushed": True}
                elif self.path == "/lora-adapters":
                    if mode == "ok":
                        resp = {
                            "status": "mounted",
                            "checkpoint_sha256": body["expected_checkpoint_sha256"],
                            "corpus_sha256": body["expected_corpus_sha256"],
                            "applied_scales": body["lora"],
                            "receipt_sha256": "rcpt-sha-001",
                        }
                    else:
                        # Simulate forged/drifted checkpoint SHA returned by daemon
                        resp = {
                            "status": "mounted",
                            "checkpoint_sha256": "wrong-checkpoint-sha",
                            "corpus_sha256": body["expected_corpus_sha256"],
                            "applied_scales": body["lora"],
                            "receipt_sha256": "rcpt-bad",
                        }
                else:
                    resp = {"error": "unknown"}
                raw = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, events

    def test_warp_bridge_happy_mount_and_fail_closed_rollback(self) -> None:
        router = ZoneRouter()
        decision = router.route(
            RouteRequest(
                zone="prose",
                active_domains=("corpus", "weft-eido"),
                repo_policy=siliconcordat_repo_policy(),
            )
        )

        # Happy path
        srv_ok, events_ok = self._start_mock_warp("ok")
        try:
            ledger = AppendOnlyLedger()
            bridge = WarpModeldBridge(f"http://127.0.0.1:{srv_ok.server_port}", ledger=ledger)
            receipt = bridge.apply_route(
                decision,
                expected_checkpoint_sha="ckpt-good",
                expected_corpus_sha="corpus-good",
            )
            self.assertEqual(receipt.status, "mounted")
            self.assertEqual(receipt.receipt_sha256, "rcpt-sha-001")
            self.assertIn(MDE_META_KEY, events_ok[0]["body"]["_meta"])
            self.assertEqual(ledger.verify_chain(), [])
        finally:
            srv_ok.shutdown()
            srv_ok.server_close()

        # Drifted checkpoint SHA -> automatic rollback with KV + recurrent flush
        srv_bad, events_bad = self._start_mock_warp("bad_ckpt")
        try:
            ledger_bad = AppendOnlyLedger()
            bridge_bad = WarpModeldBridge(f"http://127.0.0.1:{srv_bad.server_port}", ledger=ledger_bad)
            with self.assertRaises(WarpLoadError) as ctx:
                bridge_bad.apply_route(
                    decision,
                    expected_checkpoint_sha="ckpt-good",
                    expected_corpus_sha="corpus-good",
                )
            self.assertIn("checkpoint_sha_mismatch", str(ctx.exception))
            self.assertEqual(len(events_bad), 2)
            self.assertEqual(events_bad[1]["path"], "/rollback")
            self.assertTrue(events_bad[1]["body"]["flush_kv"])
            self.assertTrue(events_bad[1]["body"]["flush_recurrent_state"])
            self.assertEqual(ledger_bad.verify_chain(), [])
        finally:
            srv_bad.shutdown()
            srv_bad.server_close()


class TestExoCli(unittest.TestCase):
    def test_cli_verify_ledger_route_and_check_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "mem.jsonl"
            AppendOnlyLedger(lp).append("init", "exo", {"v": 1})

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["verify-ledger", str(lp)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(buf.getvalue())["ok"])

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = cli_main(["route-zone", "--zone", "html_report", "--domains", "corpus,weft-eido", "--preset", "corpus"])
            self.assertEqual(rc2, 2)
            self.assertTrue(json.loads(buf2.getvalue())["escalate_human"])

            buf3 = io.StringIO()
            with redirect_stdout(buf3):
                rc3 = cli_main([
                    "check-policy",
                    "--preset",
                    "weft-eido",
                    "--message",
                    "Fix layout\n\nSigned-off-by: Dev <dev@example.invalid>",
                    "--merge-method",
                    "linear",
                    "src/cell-layout.js",
                ])
            self.assertEqual(rc3, 0)
            self.assertTrue(json.loads(buf3.getvalue())["ok"])

            sim_path = Path(tmp) / "sim.jsonl"
            sim_ledger_path = Path(tmp) / "sim-ledger.jsonl"
            probes_path = Path(tmp) / "probes.json"
            sim_ledger = AppendOnlyLedger(sim_ledger_path)
            sim = SIMMemory(sim_path, ledger=sim_ledger)
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.eos",
                    capability_id="cap.tool",
                    name="Old EOS fix",
                    problem="Old model emitted early EOS",
                    rule_kind="model_workaround",
                    adapter_id="corpus",
                    base_checkpoint_sha="old",
                    corpus_sha="c1",
                )
            )
            eos_probe_entry = record_baseline_probe_receipt(
                sim_ledger,
                rule_id="rule.eos",
                checkpoint_sha="new",
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha="c1",
            )
            probes_path.write_text(
                json.dumps([
                    {
                        "rule_id": "rule.eos",
                        "passed_natively": True,
                        "observed_corpus_sha": "c1",
                        "evidence_receipt_sha256": eos_probe_entry.entry_sha256,
                    }
                ]),
                encoding="utf-8",
            )
            buf4 = io.StringIO()
            with redirect_stdout(buf4):
                rc4 = cli_main([
                    "plan-upgrade",
                    str(sim_path),
                    "--old-sha",
                    "old",
                    "--new-sha",
                    "new",
                    "--corpus-sha",
                    "c1",
                    "--probes",
                    str(probes_path),
                    "--ledger",
                    str(sim_ledger_path),
                ])
            self.assertEqual(rc4, 0)
            plan_out = json.loads(buf4.getvalue())
            self.assertEqual(plan_out["actions"][0]["action"], "retire_candidate")

            # Without --ledger, plan-upgrade must fail closed rather than retire on caller's word
            buf5 = io.StringIO()
            with redirect_stdout(buf5):
                rc5 = cli_main([
                    "plan-upgrade",
                    str(sim_path),
                    "--old-sha",
                    "old",
                    "--new-sha",
                    "new",
                    "--corpus-sha",
                    "c1",
                    "--probes",
                    str(probes_path),
                ])
            self.assertEqual(rc5, 1)
            self.assertFalse(json.loads(buf5.getvalue())["ok"])


class TestWarpUnixBridgeAndSchemaAdapter(unittest.TestCase):
    def test_schema_bridge_translates_weft_to_warp_deterministically(self) -> None:
        weft_cert = {
            "radix_scale_exp2": 24,
            "frobenius_sq_scaled": 16777216,
            "spectral_norm_sq_scaled": 4194304,
            "leakage_energy_scaled": 1024,
            "max_safety_dot_sq_scaled": 0,
            "projector_epoch": 1,
            "factor_digest": "a" * 64,
            "quant_dtype": "f16",
        }
        weft_manifest = {
            "schema": "weft-binding/v0.1",
            "adapter_name": "corpus_plain_writing",
            "adapter_class": "class1_rederivable",
            "layout_sha256": "1" * 64,
            "checkpoint_fingerprint": "2" * 64,
            "n_tensors": 2,
            "config_hashes": {"config.json": "3" * 64},
            "tokenizer_hashes": {
                "tokenizer.json": "4" * 64,
                "chat_template.jinja": "5" * 64,
            },
            "plane1_hashes": {"principles/writing.md": "6" * 64},
            "derivation_recipe": {"compiler": "weft/0.1.0"},
            "runtime_flags": {"quantization_mode": "bf16"},
            "runtime_flags_sha256": "7" * 64,
            "factor_digest": "a" * 64,
            "certificate_sha256": "b" * 64,
        }
        warp_cert = translate_weft_certificate_to_warp(
            weft_cert,
            adapter_id="mde:corpus",
            adapter_sha256="c" * 64,
            base_weights_sha256="2" * 64,
        )
        self.assertEqual(warp_cert["radix_scale"], 1 << 24)
        self.assertEqual(len(warp_cert["cert_sha256"]), 64)

        ckpt = translate_weft_checkpoint_to_warp(weft_manifest, model_id="Synthetic-W0")
        self.assertEqual(ckpt["weights_sha256"], "2" * 64)
        self.assertEqual(ckpt["chat_template_sha256"], "5" * 64)
        self.assertEqual(scale_to_rational(0.75), (3, 4))

        decision = ZoneRouter().route(
            RouteRequest(
                zone="prose",
                active_domains=("corpus", "formal-verification"),
                repo_policy=siliconcordat_repo_policy(),
            )
        )
        bundles = build_warp_bundles_for_route(
            decision,
            {
                "corpus": {
                    "adapter_id": "mde:corpus",
                    "llama_adapter_id": 0,
                    "weft_manifest": weft_manifest,
                    "weft_certificate": weft_cert,
                    "gguf_sha256": "c" * 64,
                },
            },
            model_id="Synthetic-W0",
            source_lineage={
                "corpus_tree_sha256": "d" * 64,
                "domain_commits": {"corpus": "e" * 40},
                "contested_heads": [],
            },
            runtime_flags_sha256="f" * 64,
        )
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0]["domain"], "corpus")
        self.assertEqual(bundles[0]["scale_num"], 1)
        self.assertEqual(bundles[0]["scale_den"], 1)

    def test_warp_unix_bridge_rpc_lifecycle_and_error_codes(self) -> None:
        class MockHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline()
                req = json.loads(raw.decode("utf-8"))
                method = req["method"]
                params = req.get("params", {})
                if method == "wmcp_slot_mount":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req["id"],
                        "result": {
                            "slot_manifest_digest": "m" * 64,
                            "active_adapters": [b["adapter_id"] for b in params["bundles"]],
                            "slots_flushed": [0, 1],
                            "stage2_elapsed_ns": 12000,
                            "canaries_passed": 5,
                        },
                    }
                elif method == "warp_infer" and "lora" in params:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req["id"],
                        "error": {
                            "code": -32000,
                            "message": "raw lora override forbidden",
                            "data": {"code": "ERR_UNVERIFIED_LORA_OVERRIDE", "stage": "stage4_proxy"},
                        },
                    }
                elif method == "warp_kill_switch":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req["id"],
                        "result": {
                            "reason": params["reason"],
                            "kill_switch_engaged": True,
                            "slots_flushed": [0, 1],
                            "elapsed_ms": 0.4,
                        },
                    }
                else:
                    resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"ok": True}}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()

        with tempfile.TemporaryDirectory() as tmp:
            sock_path = Path(tmp) / "warp-test.sock"
            srv = socketserver.ThreadingUnixStreamServer(str(sock_path), MockHandler)
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                ledger = AppendOnlyLedger()
                bridge = WarpUnixBridge(sock_path, ledger=ledger)
                mounted = bridge.mount_stack([{"adapter_id": "mde:corpus"}])
                self.assertEqual(mounted["active_adapters"], ["mde:corpus"])

                with self.assertRaises(WarpLoadError) as ctx:
                    bridge.infer(
                        [{"role": "user", "content": "override"}],
                        extra_params={"lora": [{"id": 0, "scale": 1.0}]},
                    )
                self.assertEqual(ctx.exception.code, "ERR_UNVERIFIED_LORA_OVERRIDE")

                ks = bridge.kill_switch(reason="test_emergency")
                self.assertTrue(ks["kill_switch_engaged"])
                self.assertEqual(ledger.verify_chain(), [])
                self.assertEqual(len(ledger.entries()), 2)
            finally:
                srv.shutdown()
                srv.server_close()
                thread.join(timeout=2.0)

    def test_schema_bridge_rejects_bool_float_and_str_integer_coercion(self) -> None:
        base_cert = {
            "radix_scale_exp2": 24,
            "frobenius_sq_scaled": 16777216,
            "spectral_norm_sq_scaled": 4194304,
            "leakage_energy_scaled": 1024,
            "max_safety_dot_sq_scaled": 0,
        }
        for field_name, bad_val in (
            ("radix_scale_exp2", True),
            ("radix_scale_exp2", 24.0),
            ("radix_scale_exp2", "24"),
            ("frobenius_sq_scaled", 16777216.9),
            ("frobenius_sq_scaled", False),
            ("spectral_norm_sq_scaled", 1.5),
            ("leakage_energy_scaled", True),
            ("max_safety_dot_sq_scaled", "0"),
        ):
            mutated = dict(base_cert)
            mutated[field_name] = bad_val
            with self.assertRaises(TypeError, msg=f"{field_name}={bad_val!r} should fail"):
                translate_weft_certificate_to_warp(
                    mutated,
                    adapter_id="mde:corpus",
                    adapter_sha256="a" * 64,
                    base_weights_sha256="b" * 64,
                )

        # Negative metric also rejected
        neg_cert = dict(base_cert, leakage_energy_scaled=-1)
        with self.assertRaises(ValueError):
            translate_weft_certificate_to_warp(
                neg_cert,
                adapter_id="mde:corpus",
                adapter_sha256="a" * 64,
                base_weights_sha256="b" * 64,
            )

        # Strict denominator, non-bool scale, and strict llama_adapter_id
        with self.assertRaises(TypeError):
            scale_to_rational(1.0, denominator=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            scale_to_rational(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            scale_to_rational(float("nan"))

        manifest = {
            "layout_sha256": "1" * 64,
            "checkpoint_fingerprint": "2" * 64,
            "config_hashes": {"config.json": "3" * 64},
            "tokenizer_hashes": {"tokenizer.json": "4" * 64, "chat_template.jinja": "5" * 64},
        }
        with self.assertRaises(TypeError):
            build_warp_adapter_bundle(
                adapter_id="mde:corpus",
                domain="corpus",
                llama_adapter_id=True,  # type: ignore[arg-type]
                weft_manifest=manifest,
                weft_certificate=base_cert,
                gguf_sha256="c" * 64,
                model_id="Synthetic-W0",
                source_lineage={"corpus_tree_sha256": "d" * 64, "domain_commits": {"corpus": "e" * 40}},
                runtime_flags_sha256="f" * 64,
            )

    def test_stdio_transport_strips_ambient_parent_credentials_by_default(self) -> None:
        import sys

        probe_script = (
            "import json, os, sys\n"
            "for line in sys.stdin:\n"
            "  req = json.loads(line)\n"
            "  seen = {\n"
            "    'has_aws': 'AWS_SECRET_ACCESS_KEY' in os.environ,\n"
            "    'has_gh': 'GH_TOKEN' in os.environ,\n"
            "    'has_path': 'PATH' in os.environ,\n"
            "  }\n"
            "  res = {'content': [], 'structuredContent': seen, 'isError': False}\n"
            "  sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': res}) + '\\n')\n"
            "  sys.stdout.flush()\n"
        )
        prev_aws = os.environ.get("AWS_SECRET_ACCESS_KEY")
        prev_gh = os.environ.get("GH_TOKEN")
        os.environ["AWS_SECRET_ACCESS_KEY"] = "synthetic-aws-secret-for-test"
        os.environ["GH_TOKEN"] = "synthetic-gh-token-for-test"
        try:
            with StdioJsonRpcTransport([sys.executable, "-c", probe_script]) as transport:
                client = MdeClient(transport, actor_id="exo-env-test")
                res = client.splice_locate("src/app.py")
                self.assertFalse(res["has_aws"])
                self.assertFalse(res["has_gh"])
                self.assertTrue(res["has_path"])
        finally:
            if prev_aws is None:
                os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
            else:
                os.environ["AWS_SECRET_ACCESS_KEY"] = prev_aws
            if prev_gh is None:
                os.environ.pop("GH_TOKEN", None)
            else:
                os.environ["GH_TOKEN"] = prev_gh


if __name__ == "__main__":
    unittest.main()
