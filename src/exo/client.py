"""L5 siliconcordat/mde MCP client and L3 warp-modeld HTTP/Unix-socket bridges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import subprocess
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
import urllib.error
import urllib.request

from exo.gate import ActionEnvelope, ApprovalGate, HumanApproval, normalize_repo_path
from exo.ledger import AppendOnlyLedger, canonical_json_bytes
from exo.router import RouteDecision

SPLICE_META_KEY = "gg.siliconcordat/splice"
MDE_META_KEY = "gg.siliconcordat/mde"


class MdeProtocolError(Exception):
    """Raised when L5 MCP protocol preconditions or JSON-RPC calls fail."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class WarpLoadError(Exception):
    """Raised when L3 warp-modeld rejects, mismatches, or rolls back an adapter mount."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class JsonRpcTransport(Protocol):
    """Minimal pluggable JSON-RPC 2.0 request transport."""

    def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        ...


class InMemoryJsonRpcTransport:
    """Deterministic in-memory JSON-RPC transport for offline tests."""

    def __init__(self, handler: Callable[[Mapping[str, Any]], dict[str, Any]]) -> None:
        self._handler = handler
        self.sent_requests: list[dict[str, Any]] = []

    def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        copied = json.loads(canonical_json_bytes(request).decode("ascii"))
        self.sent_requests.append(copied)
        return self._handler(copied)


class StdioJsonRpcTransport:
    """Line-delimited JSON-RPC 2.0 stdio transport for live L5 siliconcordat/mde servers."""

    def __init__(self, command: Sequence[str], *, cwd: Optional[str] = None) -> None:
        if not command:
            raise ValueError("command must be non-empty")
        self._proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
        )

    def send(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise MdeProtocolError(f"stdio MCP process exited with code {self._proc.returncode}")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise MdeProtocolError("stdio MCP process pipes are unavailable")
        line = canonical_json_bytes(request).decode("ascii") + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        reply = self._proc.stdout.readline()
        if not reply:
            raise MdeProtocolError("stdio MCP process closed stdout without a reply")
        parsed = json.loads(reply)
        if not isinstance(parsed, dict):
            raise MdeProtocolError("stdio MCP server returned non-object JSON-RPC payload")
        return parsed

    def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3.0)
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def __enter__(self) -> "StdioJsonRpcTransport":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class MdeClient:
    """Client for the 5 standardized L5 siliconcordat/mde structural broker verbs.

    Enforces session read-before-write locally before sending splice_patch,
    integrates ApprovalGate for exact-byte human approvals, and keeps all
    protocol extensions strictly inside vendor-namespaced _meta (Hard Veto 4).
    """

    def __init__(
        self,
        transport: JsonRpcTransport,
        *,
        gate: Optional[ApprovalGate] = None,
        ledger: Optional[AppendOnlyLedger] = None,
        actor_id: str = "exo-client",
    ) -> None:
        self.transport = transport
        self.gate = gate or ApprovalGate(ledger=ledger)
        self.ledger = ledger
        self.actor_id = actor_id
        self._req_id = 0
        self._read_tokens: dict[str, str] = {}

    def _call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        splice_meta: Optional[Mapping[str, Any]] = None,
        mde_meta: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        self._req_id += 1
        meta: dict[str, Any] = {}
        if splice_meta:
            meta[SPLICE_META_KEY] = dict(splice_meta)
        if mde_meta:
            meta[MDE_META_KEY] = dict(mde_meta)
        params: dict[str, Any] = {"name": tool_name, "arguments": dict(arguments)}
        if meta:
            params["_meta"] = meta
        request = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": "tools/call",
            "params": params,
        }
        response = self.transport.send(request)
        if "error" in response and response["error"] is not None:
            err = response["error"]
            raise MdeProtocolError(f"{tool_name} failed: {err}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise MdeProtocolError(f"{tool_name} returned non-object result")
        if result.get("isError") is True:
            detail = result.get("structuredContent") or result.get("content") or result
            code = detail.get("code") if isinstance(detail, dict) else None
            raise MdeProtocolError(
                f"{tool_name} refused (isError=true): {detail}",
                code=str(code) if code is not None else None,
                detail=detail,
            )
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        return result

    def _cache_read_token(self, payload: Mapping[str, Any], fallback_path: str) -> None:
        token = payload.get("file_read_token") or payload.get("read_token")
        path = payload.get("path") or fallback_path
        if isinstance(token, str) and token:
            try:
                norm = normalize_repo_path(str(path))
            except ValueError:
                norm = str(path)
            self._read_tokens[norm] = token

    def splice_locate(
        self,
        path: str,
        *,
        capability_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Locate CST span and syntactic zone tag; cache issued session read_token."""
        args: dict[str, Any] = {"path": path}
        if capability_id is not None:
            args["capability_id"] = capability_id
        result = self._call_tool(
            "splice_locate",
            args,
            splice_meta={"actor_id": self.actor_id, "capability_id": capability_id},
        )
        self._cache_read_token(result, path)
        return result

    def splice_trace(self, path: str, *, symbol: Optional[str] = None) -> dict[str, Any]:
        """Trace static callers, capability peers, and unproven dynamic sites."""
        args: dict[str, Any] = {"path": path}
        if symbol is not None:
            args["symbol"] = symbol
        result = self._call_tool(
            "splice_trace",
            args,
            splice_meta={"actor_id": self.actor_id, "path": path},
        )
        self._cache_read_token(result, path)
        return result

    def splice_patch(
        self,
        *,
        capability_id: str,
        file_path: str,
        edits: Sequence[Mapping[str, Any]],
        seam_contract: Optional[Mapping[str, Any]] = None,
        commit_subject: Optional[str] = None,
        summary_lines: Optional[Sequence[str]] = None,
        allow_oracle_write: bool = False,
        approval: Optional[HumanApproval] = None,
        read_tokens: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Apply CST splice edits after local read-token and ApprovalGate verification."""
        norm_path = normalize_repo_path(file_path)
        tokens = tuple(read_tokens) if read_tokens is not None else (
            (self._read_tokens[norm_path],) if norm_path in self._read_tokens else ()
        )
        if not tokens:
            raise MdeProtocolError(
                f"ERR_PRECONDITION_READ_REQUIRED: call splice_locate/splice_trace on {norm_path} first"
            )

        payload_bytes = canonical_json_bytes({
            "capability_id": capability_id,
            "file_path": norm_path,
            "edits": list(edits),
            "seam_contract": dict(seam_contract or {}),
            "allow_oracle_write": bool(allow_oracle_write),
        })
        envelope = ActionEnvelope(
            action_kind="splice_patch",
            target=norm_path,
            paths=(norm_path,),
            payload_bytes=payload_bytes,
            read_tokens=tokens,
            metadata={"capability_id": capability_id, "allow_oracle_write": bool(allow_oracle_write)},
        )
        verdict = self.gate.evaluate(envelope, approval=approval)
        if verdict.decision != "allow":
            raise MdeProtocolError(
                f"ApprovalGate blocked splice_patch ({verdict.decision}): {', '.join(verdict.reasons)}"
            )

        args: dict[str, Any] = {
            "capability_id": capability_id,
            "read_tokens": list(tokens),
            "edits": list(edits),
            "allow_oracle_write": bool(allow_oracle_write),
        }
        if seam_contract is not None and {
            "exported_symbols_touched",
            "calls_added",
            "unresolved_dynamic_sites",
            "declared_effects",
        }.issubset(seam_contract.keys()):
            args["seam_contract"] = dict(seam_contract)
        if commit_subject is not None:
            args["commit_subject"] = commit_subject
        if summary_lines is not None:
            args["summary_lines"] = list(summary_lines)

        return self._call_tool(
            "splice_patch",
            args,
            splice_meta={
                "actor_id": self.actor_id,
                "capability_id": capability_id,
                "approval_sha256": approval.challenge_sha256 if approval else None,
            },
        )

    def splice_verify(self, packet_id: str, *, eval_count: Optional[int] = None) -> dict[str, Any]:
        args: dict[str, Any] = {"packet_id": packet_id}
        if eval_count is not None:
            args["eval_count"] = int(eval_count)
        return self._call_tool(
            "splice_verify",
            args,
            splice_meta={"actor_id": self.actor_id},
        )

    def splice_project(
        self,
        packet_id: str,
        bundle_id: str,
        *,
        approve_diff_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"packet_id": packet_id, "bundle_id": bundle_id}
        if approve_diff_sha256 is not None:
            args["approve_diff_sha256"] = approve_diff_sha256
        return self._call_tool(
            "splice_project",
            args,
            splice_meta={"actor_id": self.actor_id},
        )


@dataclass(frozen=True)
class MountReceipt:
    """Verified adapter mount receipt returned by L3 warp-modeld."""

    status: str
    checkpoint_sha256: str
    corpus_sha256: str
    applied_scales: tuple[dict[str, Any], ...]
    receipt_sha256: str


class WarpModeldBridge:
    """Fail-closed HTTP bridge from Layer 4 Exo to Layer 3 warp-modeld."""

    def __init__(
        self,
        base_url: str,
        *,
        ledger: Optional[AppendOnlyLedger] = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ledger = ledger
        self.timeout_sec = float(timeout_sec)

    def _post_json(self, endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = canonical_json_bytes(body)
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise WarpLoadError(f"non-object response from {endpoint}")
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise WarpLoadError(f"warp-modeld request to {endpoint} failed: {exc}") from exc

    def rollback(self, *, reason: str) -> dict[str, Any]:
        """Trigger O(1) adapter disable plus explicit KV and recurrent-state flush."""
        payload = {
            "action": "rollback",
            "reason": reason,
            "flush_kv": True,
            "flush_recurrent_state": True,
        }
        receipt = self._post_json("/rollback", payload)
        if self.ledger is not None:
            self.ledger.append(
                "warp_rollback",
                "exo-warp-bridge",
                {"reason": reason, "receipt": receipt},
            )
        return receipt

    def apply_route(
        self,
        decision: RouteDecision,
        *,
        expected_checkpoint_sha: str,
        expected_corpus_sha: str,
    ) -> MountReceipt:
        """Apply routed multi-MDE scales on warp-modeld, failing closed with rollback on drift."""
        request_body = {
            "zone": decision.zone,
            "expected_checkpoint_sha256": expected_checkpoint_sha,
            "expected_corpus_sha256": expected_corpus_sha,
            "lora": list(decision.lora_scales),
            "_meta": {
                MDE_META_KEY: {
                    "claim_vocabulary": decision.constraints.claim_vocabulary,
                    "enforce_data_eido_skip": decision.constraints.enforce_data_eido_skip,
                }
            },
        }
        raw = self._post_json("/lora-adapters", request_body)
        status = str(raw.get("status", ""))
        ckpt = str(raw.get("checkpoint_sha256", ""))
        corpus = str(raw.get("corpus_sha256", ""))
        applied = raw.get("applied_scales", [])
        receipt_sha = str(raw.get("receipt_sha256", ""))

        mismatch_reasons: list[str] = []
        if status != "mounted":
            mismatch_reasons.append(f"unexpected_status:{status}")
        if ckpt != expected_checkpoint_sha:
            mismatch_reasons.append(
                f"checkpoint_sha_mismatch:expected={expected_checkpoint_sha},got={ckpt}"
            )
        if corpus != expected_corpus_sha:
            mismatch_reasons.append(
                f"corpus_sha_mismatch:expected={expected_corpus_sha},got={corpus}"
            )
        expected_map = decision.scale_map()
        got_map = {
            str(item["id"]): float(item["scale"])
            for item in applied
            if isinstance(item, dict) and "id" in item and "scale" in item
        }
        if got_map != expected_map:
            mismatch_reasons.append(f"scale_mismatch:expected={expected_map},got={got_map}")
        if not receipt_sha:
            mismatch_reasons.append("missing_receipt_sha256")

        if mismatch_reasons:
            reason_str = "; ".join(mismatch_reasons)
            self.rollback(reason=reason_str)
            raise WarpLoadError(f"warp-modeld mount verification failed (rolled back): {reason_str}")

        receipt = MountReceipt(
            status=status,
            checkpoint_sha256=ckpt,
            corpus_sha256=corpus,
            applied_scales=tuple(applied),
            receipt_sha256=receipt_sha,
        )
        if self.ledger is not None:
            self.ledger.append(
                "warp_mount_verified",
                "exo-warp-bridge",
                {
                    "zone": decision.zone,
                    "checkpoint_sha256": ckpt,
                    "corpus_sha256": corpus,
                    "applied_scales": list(applied),
                    "receipt_sha256": receipt_sha,
                },
            )
        return receipt


class WarpUnixBridge:
    """Pure-stdlib Unix domain socket JSON-RPC 2.0 client for live L3 warp-modeld.

    Speaks the newline-delimited JSON-RPC 2.0 protocol implemented by
    siliconcordat/warp's WarpUnixRpcServer (`wmcp_checkpoint_identify`,
    `wmcp_slot_status`, `wmcp_slot_mount`, `wmcp_slot_unmount`,
    `warp_kill_switch`, `warp_reset_kill_switch`, `warp_update_lineage`,
    `warp_infer`).
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        ledger: Optional[AppendOnlyLedger] = None,
        timeout_sec: float = 10.0,
        actor_id: str = "exo-warp-unix-bridge",
    ) -> None:
        self.socket_path = str(socket_path)
        self.ledger = ledger
        self.timeout_sec = float(timeout_sec)
        self.actor_id = actor_id
        self._req_id = 0

    def _rpc(self, method: str, params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        self._req_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": dict(params or {}),
        }
        wire = canonical_json_bytes(request) + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_sec)
                sock.connect(self.socket_path)
                sock.sendall(wire)
                chunks: list[bytes] = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
        except OSError as exc:
            raise WarpLoadError(f"warp-modeld unix socket call '{method}' failed: {exc}") from exc

        raw = b"".join(chunks).splitlines()
        if not raw:
            raise WarpLoadError(f"warp-modeld closed socket without reply for '{method}'")
        try:
            response = json.loads(raw[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WarpLoadError(f"warp-modeld returned invalid JSON for '{method}': {exc}") from exc

        if not isinstance(response, dict):
            raise WarpLoadError(f"warp-modeld returned non-object JSON-RPC payload for '{method}'")
        if "error" in response and response["error"] is not None:
            err = response["error"]
            data = err.get("data") if isinstance(err, dict) else None
            code = data.get("code") if isinstance(data, dict) else None
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise WarpLoadError(
                f"warp-modeld '{method}' rejected ({code or 'RPC_ERROR'}): {msg}",
                code=str(code) if code is not None else None,
                data=dict(data) if isinstance(data, dict) else None,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise WarpLoadError(f"warp-modeld returned non-object result for '{method}'")
        return result

    def identify_checkpoint(self) -> dict[str, Any]:
        return self._rpc("wmcp_checkpoint_identify")

    def slot_status(self) -> dict[str, Any]:
        return self._rpc("wmcp_slot_status")

    def mount_stack(
        self,
        bundles: Sequence[Mapping[str, Any]],
        *,
        composed_certificate: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"bundles": [dict(b) for b in bundles]}
        if composed_certificate is not None:
            params["composed_certificate"] = dict(composed_certificate)
        result = self._rpc("wmcp_slot_mount", params)
        if self.ledger is not None:
            self.ledger.append(
                "warp_unix_mount",
                self.actor_id,
                {
                    "slot_manifest_digest": result.get("slot_manifest_digest"),
                    "active_adapters": result.get("active_adapters"),
                    "slots_flushed": result.get("slots_flushed"),
                    "stage2_elapsed_ns": result.get("stage2_elapsed_ns"),
                    "canaries_passed": result.get("canaries_passed"),
                },
            )
        return result

    def unmount(self, *, reason: str = "exo_unmount") -> dict[str, Any]:
        result = self._rpc("wmcp_slot_unmount", {"reason": reason})
        if self.ledger is not None:
            self.ledger.append(
                "warp_unix_unmount",
                self.actor_id,
                {"reason": reason, "slots_flushed": result.get("slots_flushed")},
            )
        return result

    def kill_switch(self, *, reason: str = "exo_kill_switch") -> dict[str, Any]:
        result = self._rpc("warp_kill_switch", {"reason": reason})
        if self.ledger is not None:
            self.ledger.append(
                "warp_unix_kill_switch",
                self.actor_id,
                {
                    "reason": reason,
                    "kill_switch_engaged": result.get("kill_switch_engaged"),
                    "slots_flushed": result.get("slots_flushed"),
                    "elapsed_ms": result.get("elapsed_ms"),
                },
            )
        return result

    def reset_kill_switch(self, confirmation: str) -> dict[str, Any]:
        result = self._rpc("warp_reset_kill_switch", {"confirmation": confirmation})
        if self.ledger is not None:
            self.ledger.append("warp_unix_kill_switch_reset", self.actor_id, {"status": result})
        return result

    def update_lineage(self, lineage: Mapping[str, Any]) -> dict[str, Any]:
        result = self._rpc("warp_update_lineage", {"lineage": dict(lineage)})
        if self.ledger is not None:
            self.ledger.append("warp_unix_lineage_update", self.actor_id, {"status": result.get("status")})
        return result

    def infer(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 256,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "messages": [dict(m) for m in messages],
            "temperature": float(temperature),
            "seed": int(seed),
            "max_tokens": int(max_tokens),
        }
        if extra_params:
            params.update(extra_params)
        return self._rpc("warp_infer", params)
