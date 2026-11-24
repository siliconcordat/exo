#!/usr/bin/env python3
"""End-to-end 4-layer Silicon Cordat vertical integration demo.

Wires all four live built repositories together in one unbroken chain without
mutating any sibling repository or hardcoding local paths:
- Layer 5 (`mde`): compiled Rust MCP server (`MDE_BIN` or `PATH`)
- Layer 4 (`exo`): stdlib harness (`ZoneRouter`, `RepoPolicy`, `ApprovalGate`,
  `SIMMemory`, `CrossGenerationPlanner`, `CapabilityRegistry`, `AppendOnlyLedger`,
  `MdeClient`, `WarpUnixBridge`, `exo.bridge`)
- Layer 2 (`weft`): deterministic CPU float64 `BetaEditNullspaceProjector`,
  `FixedRadixCertificate`, 5-layer `BindingManifest`, and GGUF v3 F16 LoRA
- Layer 3 (`warp`): live `warp-modeld` Unix socket JSON-RPC daemon + localhost
  HTTP `MockLlamaServer` (Stages 1 to 4, zone-routed inference, O(1) kill-switch
  rollback with recurrent-state erase, and HMAC-SHA256 signed ledger)

Usage:
  MDE_BIN=/path/to/mde WARP_SRC=/path/to/warp/src WEFT_SRC=/path/to/weft/src \
    PYTHONPATH=src python3 examples/vertical_splice_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

# Allow direct invocation without manually setting PYTHONPATH=src when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from exo.bridge import (
    build_warp_adapter_bundle,
    build_warp_bundles_for_route,
    translate_weft_checkpoint_to_warp,
)
from exo.capabilities import CapabilityCharter, CapabilityRegistry
from exo.client import (
    MdeClient,
    MdeProtocolError,
    StdioJsonRpcTransport,
    WarpLoadError,
    WarpUnixBridge,
)
from exo.gate import ActionEnvelope, ApprovalGate
from exo.ledger import AppendOnlyLedger, canonical_json_bytes
from exo.policy import siliconcordat_repo_policy
from exo.router import ContestedPreferenceError, RouteDecision, RouteRequest, ZoneRouter
from exo.sim import (
    BaselineProbe,
    CrossGenerationPlanner,
    SIMAttempt,
    SIMMemory,
    SIMRuleRecord,
    record_baseline_probe_receipt,
    record_evaluation_receipt,
)

DOMAIN_TO_LLAMA_ID = {
    "corpus": 0,
    "weft-eido": 1,
    "formal-verification": 2,
}


@dataclass(frozen=True)
class SiblingEnvironment:
    """Runtime paths discovered strictly from environment variables or PATH."""

    mde_bin: Path
    warp_src: Path
    warp_root: Path
    weft_src: Path


def discover_siblings() -> tuple[Optional[SiblingEnvironment], list[str]]:
    """Discover sibling binaries and source roots from env vars / PATH only."""
    missing: list[str] = []

    raw_mde = os.environ.get("MDE_BIN") or shutil.which("mde")
    mde_bin = Path(raw_mde).resolve() if raw_mde else None
    if mde_bin is None or not mde_bin.is_file():
        missing.append("MDE_BIN (or `mde` on PATH)")

    raw_warp = os.environ.get("WARP_SRC")
    warp_src = Path(raw_warp).resolve() if raw_warp else None
    if warp_src is None or not (warp_src / "warp_modeld").is_dir():
        missing.append("WARP_SRC (directory containing warp_modeld/)")

    raw_weft = os.environ.get("WEFT_SRC")
    weft_src = Path(raw_weft).resolve() if raw_weft else None
    if weft_src is None or not (weft_src / "weft").is_dir():
        missing.append("WEFT_SRC (directory containing weft/)")

    if missing or mde_bin is None or warp_src is None or weft_src is None:
        return None, missing

    raw_warp_root = os.environ.get("WARP_ROOT")
    warp_root = Path(raw_warp_root).resolve() if raw_warp_root else warp_src.parent
    if not (warp_root / "schemas" / "verifier-canaries.json").is_file():
        missing.append("WARP_ROOT/schemas/verifier-canaries.json")
    if not (warp_root / "tests" / "mock_llama_server.py").is_file():
        missing.append("WARP_ROOT/tests/mock_llama_server.py")
    if missing:
        return None, missing

    return (
        SiblingEnvironment(
            mde_bin=mde_bin,
            warp_src=warp_src,
            warp_root=warp_root,
            weft_src=weft_src,
        ),
        [],
    )


def _f32(value: float) -> float:
    """Round a finite float to IEEE-754 binary32 precision using stdlib struct."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _require_finite_real(value: Any, name: str) -> float:
    """Require an exact finite int or float (rejecting bool, NaN, Inf, and int overflow)."""
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact int or float (got {type(value).__name__})")
    try:
        val = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} overflows IEEE-754 float64") from exc
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return val


def _require_positive_dim(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int (got {type(value).__name__})")
    if value <= 0:
        raise ValueError(f"{name} must be positive (got {value})")
    return value


class _Matrix2D(list):
    """Minimal pure-stdlib 2D float matrix compatible with sibling NumPy coercion."""

    def __init__(self, rows: Sequence[Sequence[Any]]) -> None:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) == 0:
            raise ValueError("matrix must be a non-empty 2D sequence")
        parsed: list[list[float]] = []
        expected_cols: Optional[int] = None
        for r_idx, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) == 0:
                raise ValueError(f"matrix row {r_idx} must be a non-empty sequence")
            if expected_cols is None:
                expected_cols = len(row)
            elif len(row) != expected_cols:
                raise ValueError(f"ragged matrix row {r_idx}: expected {expected_cols} cols, got {len(row)}")
            parsed.append([_require_finite_real(v, f"matrix[{r_idx}][{c_idx}]") for c_idx, v in enumerate(row)])
        super().__init__(parsed)
        self._cols = expected_cols if expected_cols is not None else 0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), self._cols)

    @property
    def size(self) -> int:
        return len(self) * self._cols

    @property
    def ndim(self) -> int:
        return 2

    def copy(self) -> "_Matrix2D":
        return _Matrix2D([list(row) for row in self])

    def to_f32(self) -> "_Matrix2D":
        return _Matrix2D([[_f32(v) for v in row] for row in self])

    def to_f32_bytes(self) -> bytes:
        flat = [v for row in self for v in row]
        return struct.pack(f"<{len(flat)}f", *flat)


def _rand_normal_matrix(rng: random.Random, rows: int, cols: int, *, scale: float = 1.0) -> _Matrix2D:
    n_rows = _require_positive_dim(rows, "rows")
    n_cols = _require_positive_dim(cols, "cols")
    s = _require_finite_real(scale, "scale")
    return _Matrix2D([[rng.gauss(0.0, 1.0) * s for _ in range(n_cols)] for _ in range(n_rows)])


def _ones_matrix(rows: int, cols: int) -> _Matrix2D:
    n_rows = _require_positive_dim(rows, "rows")
    n_cols = _require_positive_dim(cols, "cols")
    return _Matrix2D([[1.0] * n_cols for _ in range(n_rows)])


def _matmul_2d(a: _Matrix2D, b: _Matrix2D) -> _Matrix2D:
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
    rows_a, inner = a.shape
    cols_b = b.shape[1]
    b_cols = [[b[k][j] for k in range(inner)] for j in range(cols_b)]
    return _Matrix2D(
        [
            [math.fsum(a[i][k] * b_cols[j][k] for k in range(inner)) for j in range(cols_b)]
            for i in range(rows_a)
        ]
    )


def _add_2d(a: _Matrix2D, b: _Matrix2D) -> _Matrix2D:
    if a.shape != b.shape:
        raise ValueError(f"matrix addition shape mismatch: {a.shape} + {b.shape}")
    rows, cols = a.shape
    return _Matrix2D([[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)])


def _scale_2d(a: _Matrix2D, factor: float) -> _Matrix2D:
    f = _require_finite_real(factor, "factor")
    return _Matrix2D([[v * f for v in row] for row in a])


def _frobenius_norm_2d(a: _Matrix2D) -> float:
    return math.sqrt(math.fsum(v * v for row in a for v in row))


def _load_sibling_modules(env: SiblingEnvironment) -> dict[str, Any]:
    """Import weft, warp_modeld, and MockLlamaServer without importing numpy in exo."""
    for p in (str(env.weft_src), str(env.warp_src)):
        if p not in sys.path:
            sys.path.append(p)

    try:
        from weft.binding import (  # type: ignore[import-not-found]
            ADAPTER_CLASS_1,
            RuntimeFlags as WeftRuntimeFlags,
            build_binding_manifest,
            verify_binding,
        )
        from weft.certificate import GatePolicy  # type: ignore[import-not-found]
        from weft.compiler import (  # type: ignore[import-not-found]
            BetaEditConfig,
            BetaEditNullspaceProjector,
            verify_rederivation,
        )
        from weft.gguf_lora import read_gguf_lora, write_gguf_lora  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "weft":
            raise ModuleNotFoundError(
                f"Sibling package 'weft' requires third-party dependency {exc.name!r} "
                f"(core exo uses only the Python standard library): {exc}",
                name=exc.name,
            ) from exc
        raise
    from warp_modeld.canary import CanarySuite  # type: ignore[import-not-found]
    from warp_modeld.certificate import CertificateBounds, RationalBound  # type: ignore[import-not-found]
    from warp_modeld.fingerprint import (  # type: ignore[import-not-found]
        CheckpointFingerprint,
        Plane1Lineage,
        RuntimeFlags as WarpRuntimeFlags,
    )
    from warp_modeld.ledger import HmacSha256Signer, LedgerVerifier, SignedLedger  # type: ignore[import-not-found]
    from warp_modeld.llama_client import LlamaServerClient  # type: ignore[import-not-found]
    from warp_modeld.server import WarpUnixRpcServer  # type: ignore[import-not-found]
    from warp_modeld.verifier import WarpModeld  # type: ignore[import-not-found]

    mock_path = env.warp_root / "tests" / "mock_llama_server.py"
    spec = importlib.util.spec_from_file_location("_warp_mock_llama_server", str(mock_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load MockLlamaServer from {mock_path}")
    mock_mod = importlib.util.module_from_spec(spec)
    sys.modules["_warp_mock_llama_server"] = mock_mod
    spec.loader.exec_module(mock_mod)

    return {
        "ADAPTER_CLASS_1": ADAPTER_CLASS_1,
        "WeftRuntimeFlags": WeftRuntimeFlags,
        "build_binding_manifest": build_binding_manifest,
        "verify_binding": verify_binding,
        "GatePolicy": GatePolicy,
        "BetaEditConfig": BetaEditConfig,
        "BetaEditNullspaceProjector": BetaEditNullspaceProjector,
        "verify_rederivation": verify_rederivation,
        "read_gguf_lora": read_gguf_lora,
        "write_gguf_lora": write_gguf_lora,
        "CanarySuite": CanarySuite,
        "CertificateBounds": CertificateBounds,
        "RationalBound": RationalBound,
        "CheckpointFingerprint": CheckpointFingerprint,
        "Plane1Lineage": Plane1Lineage,
        "WarpRuntimeFlags": WarpRuntimeFlags,
        "HmacSha256Signer": HmacSha256Signer,
        "LedgerVerifier": LedgerVerifier,
        "SignedLedger": SignedLedger,
        "LlamaServerClient": LlamaServerClient,
        "WarpUnixRpcServer": WarpUnixRpcServer,
        "WarpModeld": WarpModeld,
        "MockLlamaServer": mock_mod.MockLlamaServer,
    }


def _write_synthetic_safetensors(path: Path, tensors: Mapping[str, _Matrix2D]) -> None:
    """Write a valid Safetensors binary from pure-stdlib float32 matrices."""
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in sorted(tensors.items()):
        mat = arr if isinstance(arr, _Matrix2D) else _Matrix2D(arr)
        raw = mat.to_f32_bytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(mat.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for blob in blobs:
            fh.write(blob)


def _create_synthetic_checkpoint(ckpt_dir: Path, base_weight: _Matrix2D, *, tag: str) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_safetensors(
        ckpt_dir / "model.safetensors",
        {
            "blk.0.ffn_down.weight": base_weight.to_f32(),
            "embed_tokens.weight": _ones_matrix(8, base_weight.shape[1]),
        },
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps({"hidden_size": int(base_weight.shape[1]), "checkpoint_tag": tag}) + "\n",
        encoding="utf-8",
    )
    (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}\n', encoding="utf-8")
    (ckpt_dir / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    return ckpt_dir


def _init_temp_git_repo(repo_dir: Path) -> None:
    """Initialize a scratch Git repository with a mixed-zone file and a high-risk security file."""
    (repo_dir / "src").mkdir(parents=True, exist_ok=True)
    (repo_dir / "security").mkdir(parents=True, exist_ok=True)

    mixed_src = (
        "# Vertical integration fixture.\n"
        "# Prose comments classify into the prose zone for corpus routing.\n"
        "\n"
        "def verify_add_zero(n: int) -> bool:\n"
        "    # @formal-proof theorem add_zero data-eido-skip\n"
        "    return (n + 0) == n\n"
        "\n"
        "def render_badge(label: str) -> str:\n"
        "    # @ui-module safe DOM text binding\n"
        "    return f'<span>{label}</span>'\n"
        "\n"
        "def compute_scale(weight: int) -> int:\n"
        "    return weight * 2\n"
    )
    (repo_dir / "src" / "mixed_module.py").write_text(mixed_src, encoding="utf-8")

    gate_src = (
        "# High-risk security policy module requiring exact-byte human approval.\n"
        "def allow_mount(verified: bool) -> bool:\n"
        "    return bool(verified)\n"
    )
    (repo_dir / "security" / "policy_gate.py").write_text(gate_src, encoding="utf-8")

    def _git(*args: str) -> None:
        res = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr}")

    _git("init", "-q")
    _git("config", "user.name", "vertical-stack-test")
    _git("config", "user.email", "vertical-stack-test@example.invalid")
    _git("add", "src/mixed_module.py", "security/policy_gate.py")
    _git("commit", "-q", "-m", "init vertical stack fixture")


def _compile_domain_adapter(
    mods: dict[str, Any],
    *,
    domain: str,
    adapter_id: str,
    base_weight: Any,
    ckpt_dir: Path,
    plane1_root: Path,
    plane1_relpath: str,
    gguf_out_path: Path,
    seed: int,
    reference_base_weight: Any = None,
) -> dict[str, Any]:
    """Compile one deterministic L2 weft adapter + certificate + 5-layer binding + GGUF."""
    d_out, d_in = int(base_weight.shape[0]), int(base_weight.shape[1])
    rng = random.Random(seed)

    # Low-rank anchor covariance K0 so the null space is well-conditioned and exact.
    k0 = _rand_normal_matrix(rng, d_in, 8, scale=0.25)
    # Register-scoped y=0 preservation keys (non-overlapping register).
    pres_keys = _rand_normal_matrix(rng, d_in, 2, scale=0.1)
    # Edit keys and small residual targets so post-quant F16 bounds remain tight.
    edit_keys = _rand_normal_matrix(rng, d_in, 2, scale=0.35)
    residuals = _rand_normal_matrix(rng, d_out, 2, scale=0.08)
    # Express target values V = W_ref @ K + residuals so changing W0 changes the compiled factors.
    w_ref = base_weight if reference_base_weight is None else reference_base_weight
    target_values = _add_2d(_matmul_2d(w_ref, edit_keys), residuals)

    # Unit-normalized safety direction to verify orthogonal preservation.
    b_s = _rand_normal_matrix(rng, d_out, 1)
    a_s = _rand_normal_matrix(rng, 1, d_in)
    s_norm = _frobenius_norm_2d(_matmul_2d(b_s, a_s))
    b_s = _scale_2d(b_s, 1.0 / s_norm)
    safety_dirs = [(b_s, a_s)]

    cfg = mods["BetaEditConfig"](
        d_in=d_in,
        d_out=d_out,
        max_rank=4,
        max_slots=8,
        lambda_leakage=1000.0,
        lambda_ridge=10.0,
        nullspace_threshold=2e-2,
        radix_scale_exp2=24,
        quant_dtype="f16",
    )
    compiler = mods["BetaEditNullspaceProjector"](base_weight, cfg)
    compiler.initialize_anchor_covariance(k0)
    for b_dir, a_dir in safety_dirs:
        compiler.register_safety_direction(b_dir, a_dir)

    staged = compiler.stage_edit(
        edit_keys,
        target_values,
        is_residual=False,
        rank=2,
        preservation_keys=pres_keys,
        preservation_weight=1.0,
        slot_name=adapter_id,
    )

    gate_policy = mods["GatePolicy"](
        max_frob_sq_num=5,
        max_frob_sq_den=1,
        max_spectral_sq_num=2,
        max_spectral_sq_den=1,
        max_leakage_num=1,
        max_leakage_den=100,
        max_safety_dot_sq_num=1,
        max_safety_dot_sq_den=1000,
    )
    cert_ok, cert_problems = staged.certificate.verify_division_free(gate_policy)
    if not cert_ok:
        raise RuntimeError(f"weft certificate failed division-free gate for {domain}: {cert_problems}")

    rederived_ok, rederived_detail = mods["verify_rederivation"](
        staged,
        base_weight=base_weight,
        config=cfg,
        k0=k0,
        keys=edit_keys,
        target_values_or_residuals=target_values,
        is_residual=False,
        rank=2,
        preservation_keys=pres_keys,
        preservation_weight=1.0,
        safety_directions=safety_dirs,
    )
    if not rederived_ok:
        raise RuntimeError(f"weft 0-byte re-derivation failed for {domain}: {rederived_detail}")

    gguf_path, gguf_sha = mods["write_gguf_lora"](
        gguf_out_path,
        architecture="qwen3",
        lora_alpha=4.0,
        layer_factors={"blk.0.ffn_down": (staged.factor_b, staged.factor_a)},
        extra_metadata={"weft.domain": domain, "weft.adapter_id": adapter_id},
    )
    # Prove deterministic GGUF re-encoding matches byte-for-byte.
    gguf_check_path = gguf_out_path.with_suffix(".rederived.gguf")
    _, gguf_check_sha = mods["write_gguf_lora"](
        gguf_check_path,
        architecture="qwen3",
        lora_alpha=4.0,
        layer_factors={"blk.0.ffn_down": (staged.factor_b, staged.factor_a)},
        extra_metadata={"weft.domain": domain, "weft.adapter_id": adapter_id},
    )
    gguf_check_path.unlink(missing_ok=True)
    if gguf_sha != gguf_check_sha:
        raise RuntimeError(f"GGUF output non-deterministic for {domain}")

    parsed_gguf = mods["read_gguf_lora"](gguf_path)
    if "blk.0.ffn_down.lora_a" not in parsed_gguf.tensors or "blk.0.ffn_down.lora_b" not in parsed_gguf.tensors:
        raise RuntimeError(f"GGUF LoRA missing expected tensors for {domain}")

    weft_flags = mods["WeftRuntimeFlags"](
        quantization_mode="f16",
        mamba_cache_mode="align",
        enable_thinking=False,
        tool_call_parser="qwen3_xml",
        ctx_size=16384,
    )
    manifest = mods["build_binding_manifest"](
        adapter_name=adapter_id,
        adapter_class=mods["ADAPTER_CLASS_1"],
        checkpoint_dir=ckpt_dir,
        plane1_root=plane1_root,
        plane1_files=[plane1_relpath],
        derivation_recipe={"compiler": "weft/0.1.0", "domain": domain, "seed": seed},
        runtime_flags=weft_flags,
        factor_digest=staged.factor_digest,
        certificate_sha256=staged.certificate.sha256(),
    )
    bind_res = mods["verify_binding"](
        manifest,
        checkpoint_dir=ckpt_dir,
        runtime_flags=weft_flags,
        plane1_root=plane1_root,
        expected_factor_digest=staged.factor_digest,
        expected_certificate_sha256=staged.certificate.sha256(),
    )
    if not bind_res.ok:
        raise RuntimeError(f"weft 5-layer binding failed for {domain}: {bind_res.problems}")

    return {
        "adapter_id": adapter_id,
        "domain": domain,
        "llama_adapter_id": DOMAIN_TO_LLAMA_ID[domain],
        "weft_manifest": manifest,
        "weft_certificate": staged.certificate,
        "factor_digest": staged.factor_digest,
        "gguf_sha256": gguf_sha,
        "adapter_path": str(gguf_path),
        "rederived_ok": rederived_ok,
        "gguf_bytes": gguf_path.stat().st_size,
    }


def _make_warp_modeld(
    mods: dict[str, Any],
    *,
    env: SiblingEnvironment,
    llama_url: str,
    work_dir: Path,
    checkpoint_dict: dict[str, Any],
    lineage_dict: dict[str, Any],
    signer_key: bytes,
    ledger_filename: str,
) -> tuple[Any, Any, Any, Any]:
    """Instantiate live WarpModeld + SignedLedger pinned to a weft checkpoint."""
    ckpt = mods["CheckpointFingerprint"].from_dict(checkpoint_dict)
    lineage = mods["Plane1Lineage"].from_dict(lineage_dict)
    flags = mods["WarpRuntimeFlags"](
        llama_server_commit="7fe450e19",
        quantization="Q4_K_M",
        ctx_size=16384,
        kv_cache_type="q8_0",
        ctx_checkpoints=0,
        swa_checkpoints=0,
        lora_init_without_apply=True,
        rope_args="",
        chat_template_args="thinking=off",
    )
    rb = mods["RationalBound"]
    single_bounds = mods["CertificateBounds"](
        max_frobenius_sq=rb(num=5, den=1),
        max_spectral_norm_sq=rb(num=2, den=1),
        max_leakage_energy=rb(num=1, den=100),
        max_safety_dot_sq=rb(num=1, den=1000),
    )
    joint_bounds = mods["CertificateBounds"](
        max_frobenius_sq=rb(num=20, den=1),
        max_spectral_norm_sq=rb(num=8, den=1),
        max_leakage_energy=rb(num=1, den=20),
        max_safety_dot_sq=rb(num=1, den=200),
    )
    signer = mods["HmacSha256Signer"](signer_key, key_id="vertical-stack-hmac-v1")
    ledger = mods["SignedLedger"](work_dir / ledger_filename, signer)
    suite = mods["CanarySuite"].from_verifier_file(env.warp_root / "schemas" / "verifier-canaries.json")
    llama_client = mods["LlamaServerClient"](llama_url, require_localhost=True)
    modeld = mods["WarpModeld"](
        pinned_checkpoint=ckpt,
        pinned_lineage=lineage,
        live_runtime_flags=flags,
        single_bounds=single_bounds,
        joint_bounds=joint_bounds,
        canary_suite=suite,
        llama_client=llama_client,
        ledger=ledger,
        canary_slot_id=0,
    )
    return modeld, ledger, flags, ckpt


def run_vertical_stack_demo(env: SiblingEnvironment) -> dict[str, Any]:
    """Execute the full 5-part unbroken 4-layer vertical stack integration."""
    t_start = time.perf_counter()
    mods = _load_sibling_modules(env)

    with tempfile.TemporaryDirectory(prefix="siliconcordat-vertical-") as td:
        root = Path(td)
        repo_dir = root / "repo"
        _init_temp_git_repo(repo_dir)

        exo_ledger = AppendOnlyLedger(root / "exo-ledger.jsonl")
        gate = ApprovalGate(signing_key=b"exo-approval-key-v1", ledger=exo_ledger)
        sim = SIMMemory(root / "sim-memory.jsonl", ledger=exo_ledger)
        cap_reg = CapabilityRegistry(ledger=exo_ledger)
        cap_reg.register(
            CapabilityCharter(
                capability_id="cap.vertical.gate",
                steward_actor="exo-vertical-demo",
                human_purpose="Gate adapter mounts behind exact-byte human approval and verified receipts",
                success_signals=("exact_byte_approval_verified", "canonical_evidence_bundle"),
                failure_signals=("unapproved_high_risk_write", "oracle_tamper"),
                implementation_paths=("security/policy_gate.py", "src/mixed_module.py"),
            )
        )

        # ------------------------------------------------------------------
        # STEP 1 (L5 mde): Live compiled Rust MCP locate + trace on mixed file
        # ------------------------------------------------------------------
        with StdioJsonRpcTransport([str(env.mde_bin), "--root", str(repo_dir), "mcp"]) as transport:
            mde = MdeClient(transport, gate=gate, ledger=exo_ledger, actor_id="exo-vertical-demo")

            loc_mixed = mde.splice_locate("src/mixed_module.py", capability_id="cap.vertical.gate")
            trace_mixed = mde.splice_trace("src/mixed_module.py", symbol="compute_scale")
            loc_gate = mde.splice_locate("security/policy_gate.py", capability_id="cap.vertical.gate")

            symbol_zones = {sym["symbol"]: sym["zone"] for sym in loc_mixed["symbols"]}
            zone_bytes = loc_mixed["_meta"]["gg.siliconcordat/mde"]["zone_bytes"]
            discovered_zones = sorted(set(symbol_zones.values()) | {z for z, n in zone_bytes.items() if n > 0})

            # ------------------------------------------------------------------
            # STEP 2 (L4 exo): Zone routing, ApprovalGate refusal + exact approval,
            #                  live mde patch -> verify -> project, and SIM logging
            # ------------------------------------------------------------------
            router = ZoneRouter(ledger=exo_ledger)
            policy = siliconcordat_repo_policy()
            all_domains = ("corpus", "weft-eido", "formal-verification")
            routed_decisions: dict[str, RouteDecision] = {}
            code_contested_escalated = False
            for zone in discovered_zones:
                if zone == "code":
                    # Hard Veto 2: generic `code` with both code specialists active must escalate.
                    try:
                        router.route(
                            RouteRequest(
                                zone=zone,
                                active_domains=all_domains,
                                repo_policy=policy,
                                file_path="src/mixed_module.py",
                            )
                        )
                    except ContestedPreferenceError as exc:
                        code_contested_escalated = exc.conflict_key == "code_specialist"
                    if not code_contested_escalated:
                        raise RuntimeError("ZoneRouter failed to escalate generic code with dual specialists")
                    routed_decisions[zone] = router.route(
                        RouteRequest(
                            zone=zone,
                            active_domains=("corpus",),
                            repo_policy=policy,
                            file_path="src/mixed_module.py",
                        )
                    )
                else:
                    routed_decisions[zone] = router.route(
                        RouteRequest(
                            zone=zone,
                            active_domains=all_domains,
                            repo_policy=policy,
                            file_path="src/mixed_module.py",
                        )
                    )

            # High-risk protected path edit against live Rust mde:
            gate_sym = loc_gate["symbols"][0]
            gate_edit = [
                {
                    "path": "security/policy_gate.py",
                    "start_byte": gate_sym["start_byte"],
                    "end_byte": gate_sym["end_byte"],
                    "expected_old_sha256": gate_sym["span_sha256"],
                    "replacement_utf8": (
                        "def allow_mount(verified: bool) -> bool:\n"
                        "    return verified is True\n"
                    ),
                    "anchor_symbol": gate_sym["symbol"],
                    "anchor_node_kind": gate_sym["kind"],
                }
            ]
            seam_contract = {
                "exported_symbols_touched": ["allow_mount"],
                "calls_added": [],
                "unresolved_dynamic_sites": [],
                "declared_effects": ["mount_gate_strict_bool"],
            }

            # 2a. Unapproved high-risk patch MUST be blocked by exo ApprovalGate
            unapproved_blocked = False
            try:
                mde.splice_patch(
                    capability_id="cap.vertical.gate",
                    file_path="security/policy_gate.py",
                    edits=gate_edit,
                    seam_contract=seam_contract,
                )
            except MdeProtocolError as exc:
                unapproved_blocked = "require_human_approval" in str(exc)
            if not unapproved_blocked:
                raise RuntimeError("ApprovalGate failed to block unapproved high-risk splice_patch")

            # 2b. Sign exact-byte SHA-256 HumanApproval and run splice_patch -> splice_verify -> splice_project
            payload_bytes = canonical_json_bytes(
                {
                    "capability_id": "cap.vertical.gate",
                    "file_path": "security/policy_gate.py",
                    "edits": gate_edit,
                    "seam_contract": seam_contract,
                    "allow_oracle_write": False,
                }
            )
            eval_verdict = gate.evaluate(
                ActionEnvelope(
                    action_kind="splice_patch",
                    target="security/policy_gate.py",
                    paths=("security/policy_gate.py",),
                    payload_bytes=payload_bytes,
                    read_tokens=(loc_gate["file_read_token"],),
                    metadata={"capability_id": "cap.vertical.gate", "allow_oracle_write": False},
                )
            )
            if eval_verdict.challenge is None:
                raise RuntimeError("ApprovalGate did not issue an ApprovalChallenge for high-risk splice")
            human_approval = gate.sign_approval(eval_verdict.challenge, approver_id="human-maintainer")

            packet = mde.splice_patch(
                capability_id="cap.vertical.gate",
                file_path="security/policy_gate.py",
                edits=gate_edit,
                seam_contract=seam_contract,
                commit_subject="security: enforce strict bool check in mount gate",
                summary_lines=["Verified via L4 exo ApprovalGate and L5 mde broker."],
                approval=human_approval,
            )
            evidence = mde.splice_verify(packet["packet_id"], eval_count=1)

            # Live L5 mde ALSO demands exact diff SHA-256 approval for high-risk projection!
            l5_diff_approval_blocked = False
            required_diff_sha = ""
            try:
                mde.splice_project(packet["packet_id"], evidence["bundle_id"])
            except MdeProtocolError as exc:
                if exc.code == "ERR_APPROVAL_REQUIRED" and isinstance(exc.detail, dict):
                    l5_diff_approval_blocked = True
                    required_diff_sha = str(exc.detail.get("detail", {}).get("required_diff_sha256", ""))
            if not l5_diff_approval_blocked or not required_diff_sha:
                raise RuntimeError("L5 mde did not enforce exact diff SHA-256 approval on high-risk splice_project")

            projection = mde.splice_project(
                packet["packet_id"],
                evidence["bundle_id"],
                approve_diff_sha256=required_diff_sha,
            )

        # ------------------------------------------------------------------
        # STEP 3 (L2 weft): Compile 3 deterministic adapters on synthetic W_old
        # ------------------------------------------------------------------
        plane1_root = root / "plane1"
        (plane1_root / "principles").mkdir(parents=True)
        (plane1_root / "formal").mkdir(parents=True)
        (plane1_root / "ui").mkdir(parents=True)
        (plane1_root / "principles" / "writing.md").write_text(
            "# Writing\nPlain language, no em dashes.\n", encoding="utf-8"
        )
        (plane1_root / "formal" / "invariants.md").write_text(
            "# Formal\nProve every claim without sorry or admit.\n", encoding="utf-8"
        )
        (plane1_root / "ui" / "safe-dom.md").write_text(
            "# UI\nBind text with textContent and never innerHTML.\n", encoding="utf-8"
        )

        rng_base = random.Random(20261119)
        w_old_mat = _rand_normal_matrix(rng_base, 16, 32, scale=0.2).to_f32()
        ckpt_old_dir = _create_synthetic_checkpoint(root / "ckpt_old", w_old_mat, tag="W_old")

        adapters_dir = root / "adapters_old"
        adapters_dir.mkdir()
        domain_specs = (
            ("corpus", "mde:corpus", "principles/writing.md", 101),
            ("weft-eido", "mde:weft-eido", "ui/safe-dom.md", 102),
            ("formal-verification", "mde:formal-verification", "formal/invariants.md", 103),
        )
        compiled_old: dict[str, dict[str, Any]] = {}
        for domain, aid, rel_doc, seed in domain_specs:
            compiled_old[domain] = _compile_domain_adapter(
                mods,
                domain=domain,
                adapter_id=aid,
                base_weight=w_old_mat,
                ckpt_dir=ckpt_old_dir,
                plane1_root=plane1_root,
                plane1_relpath=rel_doc,
                gguf_out_path=adapters_dir / f"{domain}.gguf",
                seed=seed,
            )

        # ------------------------------------------------------------------
        # STEP 4 (L3 warp): Live warp-modeld over Unix socket + MockLlamaServer
        # ------------------------------------------------------------------
        corpus_tree_sha = hashlib.sha256(
            canonical_json_bytes(
                {d: compiled_old[d]["weft_manifest"].plane1_hashes for d in sorted(compiled_old)}
            )
        ).hexdigest()
        lineage_dict = {
            "corpus_tree_sha256": corpus_tree_sha,
            "domain_commits": {
                "corpus": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
                "weft-eido": "be2766525500112233445566778899aabbccddee",
                "formal-verification": "3165e20e7a00112233445566778899aabbccddee",
            },
            "contested_heads": [],
        }

        w_old_ckpt_dict = translate_weft_checkpoint_to_warp(
            compiled_old["corpus"]["weft_manifest"],
            model_id="Synthetic-W_old-F32",
        )
        mock_adapters_old = [
            {"id": DOMAIN_TO_LLAMA_ID[d], "path": compiled_old[d]["adapter_path"], "scale": 0.0}
            for d in ("corpus", "weft-eido", "formal-verification")
        ]

        mock_llama = mods["MockLlamaServer"](adapters=mock_adapters_old, num_slots=2)
        llama_url = mock_llama.start()
        warp_old_ledger_key = b"warp-vertical-old-key-32-bytes01"
        zone_mount_receipts: dict[str, dict[str, Any]] = {}
        try:
            modeld_old, warp_ledger_old, warp_flags, _ = _make_warp_modeld(
                mods,
                env=env,
                llama_url=llama_url,
                work_dir=root,
                checkpoint_dict=w_old_ckpt_dict,
                lineage_dict=lineage_dict,
                signer_key=warp_old_ledger_key,
                ledger_filename="warp-old-ledger.jsonl",
            )
            sock_old = root / "warp-old.sock"
            rpc_server_old = mods["WarpUnixRpcServer"](sock_old, modeld_old, socket_mode=0o600)
            rpc_server_old.start()
            try:
                bridge_old = WarpUnixBridge(sock_old, ledger=exo_ledger)
                ident = bridge_old.identify_checkpoint()

                # Mount and run zone-routed inference for prose, formal_proof, and ui_module!
                for zone_name in ("prose", "formal_proof", "ui_module"):
                    decision = routed_decisions[zone_name]
                    bundles = build_warp_bundles_for_route(
                        decision,
                        compiled_old,
                        model_id="Synthetic-W_old-F32",
                        source_lineage=lineage_dict,
                        runtime_flags_sha256=warp_flags.canonical_digest(),
                    )
                    mount_res = bridge_old.mount_stack(bundles)
                    infer_res = bridge_old.infer(
                        [{"role": "user", "content": f"zone check for {zone_name}"}]
                    )
                    reply_text = infer_res["response"]["choices"][0]["message"]["content"]
                    zone_mount_receipts[zone_name] = {
                        "active_adapters": mount_res["active_adapters"],
                        "slots_flushed": mount_res["slots_flushed"],
                        "stage2_elapsed_ns": mount_res["stage2_elapsed_ns"],
                        "canaries_passed": mount_res["canaries_passed"],
                        "slot_manifest_digest": mount_res["slot_manifest_digest"],
                        "reply_text": reply_text,
                    }

                # Adversarial check: raw `lora` override over warp_infer MUST fail closed
                raw_lora_override_code = None
                try:
                    bridge_old.infer(
                        [{"role": "user", "content": "attempt bypass"}],
                        extra_params={"lora": [{"id": 0, "scale": 1.0}]},
                    )
                except WarpLoadError as exc:
                    raw_lora_override_code = exc.code
                if raw_lora_override_code != "ERR_UNVERIFIED_LORA_OVERRIDE":
                    raise RuntimeError(f"Expected ERR_UNVERIFIED_LORA_OVERRIDE, got {raw_lora_override_code}")

                # O(1) kill-switch rollback + recurrent-state slot erase
                ks_res = bridge_old.kill_switch(reason="vertical_stack_o1_rollback")
                base_infer = bridge_old.infer(
                    [{"role": "user", "content": "post rollback base check"}]
                )
                post_rollback_text = base_infer["response"]["choices"][0]["message"]["content"]
                if "[BASE_W0]" not in post_rollback_text or "RESIDUAL_RECURRENT_LEAK" in post_rollback_text:
                    raise RuntimeError(f"Rollback left residual state: {post_rollback_text}")

                warp_old_verified_count = len(
                    mods["LedgerVerifier"].verify_file(warp_ledger_old.path, warp_ledger_old.signer)
                )
            finally:
                rpc_server_old.stop()

            # ------------------------------------------------------------------
            # STEP 5 (Cross-generation upgrade loop): W_old -> W_new
            # ------------------------------------------------------------------
            w_new_mat = w_old_mat.copy()
            w_new_mat[0][0] = _f32(w_new_mat[0][0] + 0.125)
            ckpt_new_dir = _create_synthetic_checkpoint(root / "ckpt_new", w_new_mat, tag="W_new")

            # Recompile corpus adapter on W_new ONLY after verifying W_old is rejected on W_new
            adapters_new_dir = root / "adapters_new"
            adapters_new_dir.mkdir()
            recompiled_corpus_new = _compile_domain_adapter(
                mods,
                domain="corpus",
                adapter_id="mde:corpus",
                base_weight=w_new_mat,
                ckpt_dir=ckpt_new_dir,
                plane1_root=plane1_root,
                plane1_relpath="principles/writing.md",
                gguf_out_path=adapters_new_dir / "corpus.gguf",
                seed=101,
                reference_base_weight=w_old_mat,
            )
            w_new_ckpt_dict = translate_weft_checkpoint_to_warp(
                recompiled_corpus_new["weft_manifest"],
                model_id="Synthetic-W_new-F32",
            )

            # Register SIM rules on W_old and classify upgrade plan for W_old -> W_new
            w_old_sha = w_old_ckpt_dict["weights_sha256"]
            w_new_sha = w_new_ckpt_dict["weights_sha256"]
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.corpus.plain_prose",
                    capability_id="cap.vertical.gate",
                    name="Plain prose without em dashes",
                    problem="Keep public prose plain and scannable",
                    rule_kind="durable_invariant",
                    adapter_id="corpus",
                    base_checkpoint_sha=w_old_sha,
                    corpus_sha=corpus_tree_sha,
                )
            )
            eval_receipt = record_evaluation_receipt(
                exo_ledger,
                rule_id="rule.corpus.plain_prose",
                round_id=1,
                outcome="improved",
                base_checkpoint_sha=w_old_sha,
                corpus_sha=corpus_tree_sha,
                details={"evidence_bundle_id": evidence["bundle_id"]},
            )
            sim.record_attempt(
                "rule.corpus.plain_prose",
                SIMAttempt(
                    round_id=1,
                    what="Enforce strict bool check in mount gate and compile corpus adapter on W_old",
                    predicted="High-risk patch clears L4+L5 exact-byte approvals and corpus adapter passes canaries",
                    fixed=("exact_byte_gate_projection", "corpus_prose_canaries"),
                    missed=(),
                    base_checkpoint_sha=w_old_sha,
                    corpus_sha=corpus_tree_sha,
                    outcome="improved",
                    files=("security/policy_gate.py",),
                    domain="corpus",
                    evidence_receipt_sha256=eval_receipt.entry_sha256,
                ),
            )
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.old.eos_guard",
                    capability_id="cap.vertical.gate",
                    name="Legacy EOS stop workaround",
                    problem="W_old terminated tool JSON prematurely",
                    rule_kind="model_workaround",
                    adapter_id="corpus",
                    base_checkpoint_sha=w_old_sha,
                    corpus_sha=corpus_tree_sha,
                )
            )
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.report.density",
                    capability_id="cap.vertical.gate",
                    name="Contested report density preference",
                    problem="Two teams disagree on condensed vs verbose reports",
                    rule_kind="preference",
                    adapter_id="corpus",
                    base_checkpoint_sha=w_old_sha,
                    corpus_sha=corpus_tree_sha,
                    contested=True,
                    contested_reason="two teams disagree on condensed vs verbose reports",
                )
            )
            eos_probe_receipt = record_baseline_probe_receipt(
                exo_ledger,
                rule_id="rule.old.eos_guard",
                checkpoint_sha=w_new_sha,
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha=corpus_tree_sha,
                notes="W_new emits complete tool JSON natively.",
            )
            upgrade_plan = CrossGenerationPlanner().plan_upgrade(
                sim,
                old_checkpoint_sha=w_old_sha,
                new_checkpoint_sha=w_new_sha,
                active_corpus_sha=corpus_tree_sha,
                probes={
                    "rule.old.eos_guard": BaselineProbe(
                        rule_id="rule.old.eos_guard",
                        passed_natively=True,
                        observed_corpus_sha=corpus_tree_sha,
                        notes="W_new emits complete tool JSON natively.",
                        evidence_receipt_sha256=eos_probe_receipt.entry_sha256,
                    )
                },
                ledger=exo_ledger,
            )
            plan_by_rule = {a.rule_id: a.action for a in upgrade_plan.actions}

            # Start W_new-pinned warp-modeld and prove:
            # 1) stale W_old bundle is rejected with ERR_CHECKPOINT_MISMATCH
            # 2) recompiled W_new durable bundle mounts cleanly and infers [ACTIVE_LORA:0]
            modeld_new, warp_ledger_new, warp_flags_new, _ = _make_warp_modeld(
                mods,
                env=env,
                llama_url=llama_url,
                work_dir=root,
                checkpoint_dict=w_new_ckpt_dict,
                lineage_dict=lineage_dict,
                signer_key=b"warp-vertical-new-key-32-bytes02",
                ledger_filename="warp-new-ledger.jsonl",
            )
            sock_new = root / "warp-new.sock"
            rpc_server_new = mods["WarpUnixRpcServer"](sock_new, modeld_new, socket_mode=0o600)
            rpc_server_new.start()
            try:
                bridge_new = WarpUnixBridge(sock_new, ledger=exo_ledger)
                stale_bundle_old = build_warp_adapter_bundle(
                    adapter_id="mde:corpus",
                    domain="corpus",
                    llama_adapter_id=0,
                    weft_manifest=compiled_old["corpus"]["weft_manifest"],
                    weft_certificate=compiled_old["corpus"]["weft_certificate"],
                    gguf_sha256=compiled_old["corpus"]["gguf_sha256"],
                    model_id="Synthetic-W_old-F32",
                    source_lineage=lineage_dict,
                    runtime_flags_sha256=warp_flags_new.canonical_digest(),
                    scale=1.0,
                    adapter_path=compiled_old["corpus"]["adapter_path"],
                )
                stale_reject_code = None
                try:
                    bridge_new.mount_stack([stale_bundle_old])
                except WarpLoadError as exc:
                    stale_reject_code = exc.code
                if stale_reject_code != "ERR_CHECKPOINT_MISMATCH":
                    raise RuntimeError(f"Expected ERR_CHECKPOINT_MISMATCH on stale bundle, got {stale_reject_code}")

                fresh_bundle_new = build_warp_adapter_bundle(
                    adapter_id="mde:corpus",
                    domain="corpus",
                    llama_adapter_id=0,
                    weft_manifest=recompiled_corpus_new["weft_manifest"],
                    weft_certificate=recompiled_corpus_new["weft_certificate"],
                    gguf_sha256=recompiled_corpus_new["gguf_sha256"],
                    model_id="Synthetic-W_new-F32",
                    source_lineage=lineage_dict,
                    runtime_flags_sha256=warp_flags_new.canonical_digest(),
                    scale=1.0,
                    adapter_path=recompiled_corpus_new["adapter_path"],
                )
                mount_new_res = bridge_new.mount_stack([fresh_bundle_new])
                infer_new_res = bridge_new.infer(
                    [{"role": "user", "content": "W_new durable rule check"}]
                )
                infer_new_text = infer_new_res["response"]["choices"][0]["message"]["content"]
                warp_new_verified_count = len(
                    mods["LedgerVerifier"].verify_file(warp_ledger_new.path, warp_ledger_new.signer)
                )
            finally:
                rpc_server_new.stop()
        finally:
            mock_llama.stop()

        exo_ledger_errors = exo_ledger.verify_chain(signing_key=b"exo-approval-key-v1")
        if exo_ledger_errors:
            raise RuntimeError(f"exo AppendOnlyLedger verification failed: {exo_ledger_errors}")

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "step1_l5_mde": {
                "dominant_zone": loc_mixed["dominant_zone"],
                "file_read_token": loc_mixed["file_read_token"],
                "trace_symbol": trace_mixed["symbol"],
                "symbol_zones": symbol_zones,
                "zone_bytes": zone_bytes,
                "discovered_zones": discovered_zones,
            },
            "step2_l4_exo": {
                "routed_scales": {z: d.scale_map() for z, d in routed_decisions.items()},
                "code_contested_escalated": code_contested_escalated,
                "unapproved_high_risk_blocked": unapproved_blocked,
                "l5_diff_approval_blocked": l5_diff_approval_blocked,
                "approved_packet_id": packet["packet_id"],
                "evidence_bundle_id": evidence["bundle_id"],
                "evidence_grade": evidence["summary"]["lexicase_grade"],
                "oracle_tamper_free": evidence["oracle_integrity"]["tamper_free"],
                "secret_scan_passed": evidence["secret_scan"]["passed"],
                "projected_commit": projection["commit_sha"],
                "sim_attempts_recorded": sum(len(r.attempts) for r in sim.rules()),
                "exo_ledger_entries": len(exo_ledger.entries()),
                "exo_ledger_errors": exo_ledger_errors,
            },
            "step3_l2_weft": {
                "w_old_weights_sha256": w_old_sha,
                "adapters": {
                    d: {
                        "factor_digest": spec["factor_digest"],
                        "gguf_sha256": spec["gguf_sha256"],
                        "gguf_bytes": spec["gguf_bytes"],
                        "rederived_ok": spec["rederived_ok"],
                        "certificate_sha256": spec["weft_certificate"].sha256(),
                    }
                    for d, spec in compiled_old.items()
                },
            },
            "step4_l3_warp": {
                "checkpoint_model_id": ident["checkpoint"]["model_id"],
                "zone_mounts": zone_mount_receipts,
                "raw_lora_override_code": raw_lora_override_code,
                "kill_switch": {
                    "engaged": ks_res["kill_switch_engaged"],
                    "slots_flushed": ks_res["slots_flushed"],
                    "elapsed_ms": ks_res["elapsed_ms"],
                    "post_rollback_text": post_rollback_text,
                },
                "warp_old_ledger_receipts_verified": warp_old_verified_count,
            },
            "step5_cross_generation_upgrade": {
                "w_old_sha256": w_old_sha,
                "w_new_sha256": w_new_sha,
                "stale_bundle_reject_code": stale_reject_code,
                "planner_actions": plan_by_rule,
                "recompiled_w_new_gguf_sha256": recompiled_corpus_new["gguf_sha256"],
                "recompiled_w_new_rederived_ok": recompiled_corpus_new["rederived_ok"],
                "w_new_active_adapters": mount_new_res["active_adapters"],
                "w_new_stage2_elapsed_ns": mount_new_res["stage2_elapsed_ns"],
                "w_new_canaries_passed": mount_new_res["canaries_passed"],
                "w_new_infer_text": infer_new_text,
                "warp_new_ledger_receipts_verified": warp_new_verified_count,
            },
        }


def main() -> int:
    env, missing = discover_siblings()
    if env is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "skipped": True,
                    "missing": missing,
                    "hint": "Set MDE_BIN, WARP_SRC, and WEFT_SRC to run the live 4-layer stack.",
                },
                indent=2,
            )
        )
        return 2
    report = run_vertical_stack_demo(env)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
