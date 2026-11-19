"""End-to-end 4-layer vertical stack integration tests (L5 mde -> L4 exo -> L2 weft -> L3 warp).

Discovers sibling binaries and packages strictly through environment variables
(`MDE_BIN` or `PATH`, `WARP_SRC`, `WEFT_SRC`). When any sibling env var is unset,
the suite skips cleanly so standalone zero-dependency `exo` tests stay green.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

from exo.policy import foundation0_repo_policy, weft_eido_repo_policy
from exo.router import RouteRequest, ZoneRouter

_DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "vertical_splice_demo.py"
_SPEC = importlib.util.spec_from_file_location("_exo_vertical_splice_demo", str(_DEMO_PATH))
assert _SPEC is not None and _SPEC.loader is not None
_DEMO = importlib.util.module_from_spec(_SPEC)
sys.modules["_exo_vertical_splice_demo"] = _DEMO
_SPEC.loader.exec_module(_DEMO)

_SIBLING_ENV, _MISSING_SIBLINGS = _DEMO.discover_siblings()


@unittest.skipUnless(
    _SIBLING_ENV is not None,
    f"4-layer vertical stack skipped; set sibling env vars ({', '.join(_MISSING_SIBLINGS)})",
)
class TestVerticalFourLayerStack(unittest.TestCase):
    def test_unbroken_l5_l4_l2_l3_chain_and_cross_generation_upgrade(self) -> None:
        assert _SIBLING_ENV is not None
        report = _DEMO.run_vertical_stack_demo(_SIBLING_ENV)
        self.assertTrue(report["ok"])

        # Step 1 (L5 mde): real Rust CST syntactic zone tagging + read token
        s1 = report["step1_l5_mde"]
        self.assertTrue(s1["file_read_token"].startswith("sha256:"))
        self.assertEqual(s1["trace_symbol"], "compute_scale")
        self.assertEqual(s1["symbol_zones"]["verify_add_zero"], "formal_proof")
        self.assertEqual(s1["symbol_zones"]["render_badge"], "ui_module")
        self.assertEqual(s1["symbol_zones"]["compute_scale"], "code")
        self.assertGreater(s1["zone_bytes"].get("prose", 0), 0)
        self.assertEqual(
            set(s1["discovered_zones"]),
            {"prose", "formal_proof", "ui_module", "code"},
        )

        # Step 2 (L4 exo): zone routing, exact-byte approval gates, live mde verify/project, SIM + ledger
        s2 = report["step2_l4_exo"]
        self.assertEqual(
            s2["routed_scales"]["prose"],
            {"corpus": 1.0, "weft-eido": 0.0, "formal-verification": 0.0},
        )
        self.assertEqual(
            s2["routed_scales"]["formal_proof"],
            {"corpus": 0.0, "weft-eido": 0.0, "formal-verification": 1.0},
        )
        self.assertEqual(
            s2["routed_scales"]["ui_module"],
            {"corpus": 0.0, "weft-eido": 1.0, "formal-verification": 0.0},
        )
        self.assertTrue(s2["code_contested_escalated"])
        self.assertEqual(s2["routed_scales"]["code"], {"corpus": 0.5})
        self.assertTrue(s2["unapproved_high_risk_blocked"])
        self.assertTrue(s2["l5_diff_approval_blocked"])
        self.assertEqual(s2["evidence_grade"], "canonical")
        self.assertTrue(s2["oracle_tamper_free"])
        self.assertTrue(s2["secret_scan_passed"])
        self.assertTrue(len(s2["projected_commit"]) >= 7)
        self.assertEqual(s2["sim_attempts_recorded"], 1)
        self.assertEqual(s2["exo_ledger_errors"], [])
        self.assertGreaterEqual(s2["exo_ledger_entries"], 10)

        # Step 3 (L2 weft): 3 deterministic GGUF adapters + 0-byte re-derivation
        s3 = report["step3_l2_weft"]
        self.assertEqual(len(s3["w_old_weights_sha256"]), 64)
        for domain in ("corpus", "weft-eido", "formal-verification"):
            adapter = s3["adapters"][domain]
            self.assertTrue(adapter["rederived_ok"])
            self.assertEqual(len(adapter["gguf_sha256"]), 64)
            self.assertGreater(adapter["gguf_bytes"], 64)

        # Step 4 (L3 warp): Unix socket mount, <1ms Stage 2, shadow canaries, zone inference, O(1) rollback
        s4 = report["step4_l3_warp"]
        self.assertEqual(s4["checkpoint_model_id"], "Synthetic-W_old-F32")
        self.assertEqual(
            [a["adapter_id"] for a in s4["zone_mounts"]["prose"]["active_adapters"]],
            ["mde:corpus"],
        )
        self.assertIn("[ACTIVE_LORA:0]", s4["zone_mounts"]["prose"]["reply_text"])
        self.assertEqual(
            [a["adapter_id"] for a in s4["zone_mounts"]["formal_proof"]["active_adapters"]],
            ["mde:formal-verification"],
        )
        self.assertIn("[ACTIVE_LORA:2]", s4["zone_mounts"]["formal_proof"]["reply_text"])
        self.assertEqual(
            [a["adapter_id"] for a in s4["zone_mounts"]["ui_module"]["active_adapters"]],
            ["mde:weft-eido"],
        )
        self.assertIn("[ACTIVE_LORA:1]", s4["zone_mounts"]["ui_module"]["reply_text"])
        for z_info in s4["zone_mounts"].values():
            self.assertLess(z_info["stage2_elapsed_ns"], 5_000_000)
            self.assertGreaterEqual(z_info["canaries_passed"], 5)
            self.assertEqual(z_info["slots_flushed"], [0, 1])

        self.assertEqual(s4["raw_lora_override_code"], "ERR_UNVERIFIED_LORA_OVERRIDE")
        self.assertTrue(s4["kill_switch"]["engaged"])
        self.assertEqual(s4["kill_switch"]["slots_flushed"], [0, 1])
        self.assertIn("[BASE_W0]", s4["kill_switch"]["post_rollback_text"])
        self.assertNotIn("RESIDUAL_RECURRENT_LEAK", s4["kill_switch"]["post_rollback_text"])
        self.assertEqual(s4["warp_old_ledger_receipts_verified"], 4)  # 3 mounts + 1 kill_switch

        # Step 5 (Cross-generation upgrade): W_old -> W_new rejection, planner, recompile, mount
        s5 = report["step5_cross_generation_upgrade"]
        self.assertNotEqual(s5["w_old_sha256"], s5["w_new_sha256"])
        self.assertNotEqual(s5["recompiled_w_new_gguf_sha256"], s3["adapters"]["corpus"]["gguf_sha256"])
        self.assertEqual(s5["stale_bundle_reject_code"], "ERR_CHECKPOINT_MISMATCH")
        self.assertEqual(
            s5["planner_actions"],
            {
                "rule.corpus.plain_prose": "recompile",
                "rule.old.eos_guard": "retire_candidate",
                "rule.report.density": "escalate_human",
            },
        )
        self.assertTrue(s5["recompiled_w_new_rederived_ok"])
        self.assertEqual([a["adapter_id"] for a in s5["w_new_active_adapters"]], ["mde:corpus"])
        self.assertIn("[ACTIVE_LORA:0]", s5["w_new_infer_text"])
        self.assertEqual(s5["warp_new_ledger_receipts_verified"], 2)  # 1 stale reject + 1 fresh mount

    def test_live_l5_zone_manifest_verifies_and_compiles_router(self) -> None:
        if _SIBLING_ENV is None:
            self.skipTest("sibling env unavailable")
        raw_manifest = os.environ.get("MDE_MANIFEST")
        if raw_manifest:
            manifest_path = Path(raw_manifest)
        else:
            # Standard Cargo target layout: <mde_root>/target/{debug,release}/mde
            manifest_path = _SIBLING_ENV.mde_bin.parents[2] / "schemas" / "zone-manifest.v0.1.json"
        if not manifest_path.is_file():
            self.skipTest("mde schemas/zone-manifest.v0.1.json not found alongside MDE_BIN")

        router = ZoneRouter.from_zone_manifest_file(manifest_path)
        self.assertIsNotNone(router.manifest_id)
        self.assertTrue((router.manifest_id or "").startswith("sha256:"))

        fp = router.route(
            RouteRequest(
                zone="formal_proof",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=foundation0_repo_policy(),
            )
        )
        self.assertEqual(fp.l1_bank_tag, 3)
        self.assertEqual(fp.scale_map(), {"corpus": 0.0, "weft-eido": 0.0, "formal-verification": 1.0})
        self.assertTrue(fp.constraints.enforce_data_eido_skip)

        ui = router.route(
            RouteRequest(
                zone="ui_module",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=weft_eido_repo_policy(),
            )
        )
        self.assertEqual(ui.l1_bank_tag, 2)
        self.assertEqual(ui.scale_map(), {"corpus": 0.2, "weft-eido": 1.0, "formal-verification": 0.0})


if __name__ == "__main__":
    unittest.main()
