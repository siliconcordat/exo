# Four-Layer Vertical Integration

This document records the live end-to-end integration run across all four built Siliconcordat repositories: L5 `mde`, L4 `exo`, L2 `weft`, and L3 `warp`.

## Verification boundary

The vertical integration suite runs real compiled code from all four layers in one unbroken chain:

- L5 `mde`: compiled Rust binary invoked over stdio JSON-RPC 2.0 (`mde --root <dir> mcp`).
- L4 `exo`: zero-dependency Python standard-library harness (`MdeClient`, `ZoneRouter`, `RepoPolicy`, `ApprovalGate`, `SIMMemory`, `CrossGenerationPlanner`, `CapabilityRegistry`, `AppendOnlyLedger`, `WarpUnixBridge`, and `exo.bridge`).
- L2 `weft`: deterministic CPU float64 `BetaEditNullspaceProjector`, division-free `FixedRadixCertificate`, five-layer `BindingManifest`, and GGUF v3 F16 LoRA writer running on synthetic Safetensors checkpoints (`16 x 32` float32 matrices).
- L3 `warp`: live `warp-modeld` daemon (`WarpUnixRpcServer` over a `0o600` Unix domain socket with HMAC-SHA256 `SignedLedger` and verifier-owned `schemas/verifier-canaries.json`) talking to `MockLlamaServer` over localhost HTTP.

This suite does not run live 8B or 27B open-weights checkpoints on GPU hardware. Those runs belong to hardware evaluation after multi-layer schema alignment.

## How to run

Core `exo` stays zero-dependency. Sibling binaries and source trees are discovered only through environment variables (`MDE_BIN` or `PATH`, `WARP_SRC`, `WEFT_SRC`, and optional `WARP_ROOT`).

Standalone offline mode (env vars unset, vertical test skips cleanly):

```bash
PYTHONPATH=src python3 -B -W error -m unittest discover -s tests -v
```

Full live four-layer mode (all tests run with zero skips):

```bash
MDE_BIN=/path/to/mde/target/debug/mde \
WARP_SRC=/path/to/warp/src \
WEFT_SRC=/path/to/weft/src \
PYTHONPATH=src python3 -B -W error -m unittest discover -s tests -v
```

Standalone JSON demo report:

```bash
MDE_BIN=/path/to/mde/target/debug/mde \
WARP_SRC=/path/to/warp/src \
WEFT_SRC=/path/to/weft/src \
PYTHONPATH=src python3 -B -W error examples/vertical_splice_demo.py
```

## Measured results

Measured locally on macOS with Python 3.14 and compiled Rust `mde`:

- Standalone offline `unittest`: 32 tests discovered, 31 passed, 1 skipped in `1.594s` under `-B -W error`.
- Live four-layer `unittest`: 32 tests discovered, 32 passed, 0 skipped in `3.595s` under `-B -W error`.
- Direct CLI run of `examples/vertical_splice_demo.py`: completed all five stages in `1509.46 ms`.

| Stage | Layer | Verified behavior | Measured output |
| --- | --- | --- | --- |
| 1. CST locate and trace | L5 `mde` | Live Rust MCP parsed a mixed Python file, tagged four zones, and issued a session `file_read_token`. | Zones `code` (`60` bytes), `formal_proof` (`159` bytes), `prose` (`96` bytes), `ui_module` (`144` bytes). Symbol `compute_scale` traced cleanly. |
| 2. Routing and approval gates | L4 `exo` + L5 `mde` | `ZoneRouter` mapped `prose -> corpus 1.0`, `formal_proof -> formal-verification 1.0`, `ui_module -> weft-eido 1.0`, and raised `ContestedPreferenceError` on generic `code` when both specialists were active. Both L4 `ApprovalGate` and L5 `mde` (`ERR_APPROVAL_REQUIRED`) blocked an unapproved write to `security/policy_gate.py` until exact-byte SHA-256 approvals were supplied. | `evidence_grade = canonical`, `oracle_tamper_free = true`, `secret_scan_passed = true`, `exo` ledger verified 17 entries with 0 errors. |
| 3. Adapter compilation | L2 `weft` | Compiled three domain adapters (`corpus`, `weft-eido`, `formal-verification`) on synthetic Safetensors `W_old`, checked post-quantization F16 certificates, wrote deterministic GGUF v3 F16 binaries, and verified five-layer bindings. | GGUF sizes `672`, `672`, and `704` bytes. All three passed 0-byte re-derivation (`rederived_ok = true`). |
| 4. Load gate, inference, and rollback | L3 `warp` | `WarpUnixBridge` mounted translated `weft` bundles over a `0o600` Unix socket. Stage 1 hashed on-disk `.gguf` binaries, Stage 2 checked integer certificates, Stage 3 ran five verifier-owned shadow canaries, and Stage 4 flushed slots `[0, 1]` before zone-routed inference. Raw `lora` override failed closed (`ERR_UNVERIFIED_LORA_OVERRIDE`). `warp_kill_switch` rolled back to clean base `W0`. | Stage 2 timings `11.3 us` to `21.0 us`. Active replies tagged `[ACTIVE_LORA:0]`, `[ACTIVE_LORA:2]`, `[ACTIVE_LORA:1]`. Kill switch took `2.41 ms`, returned `[BASE_W0]` with zero residual leak, and `LedgerVerifier` verified 4 of 4 HMAC receipts. |
| 5. Checkpoint upgrade `W_old -> W_new` | L3 + L4 + L2 | Perturbed base weights (`W_old` `98fc7c99...` to `W_new` `0a4978dc...`). `warp-modeld` rejected the stale `W_old` bundle with `ERR_CHECKPOINT_MISMATCH`. `CrossGenerationPlanner` classified three SIM rules into `recompile`, `retire_candidate`, and `escalate_human`. `weft` recompiled `corpus` against `W_new` with a new GGUF digest (`c20b7471...` vs `7c7a9fc9...`) and mounted cleanly. | Recompiled `W_new` adapter passed 0-byte re-derivation, Stage 2 check (`16.2 us`), 5 shadow canaries, `[ACTIVE_LORA:0]` inference, and 2 of 2 HMAC ledger receipts. |

## Cross-layer schema mismatches found

Testing the four live repositories together surfaced four schema gaps. None of the sibling repositories were modified here. Each gap is bridged cleanly inside `src/exo/bridge.py` and `src/exo/client.py` and reported back to sibling owners for root alignment.

1. `FixedRadixCertificate` shape between L2 `weft` and L3 `warp`. `weft` stores `radix_scale_exp2` (exponent `24`), `projector_epoch`, `factor_digest`, and `quant_dtype`, and computes `.sha256()` externally. `warp` expects `radix_scale` (integer `1 << 24`), `adapter_id`, `adapter_sha256`, `base_weights_sha256`, and embedded `cert_sha256`. Both verify the same four scaled integer bounds via division-free cross-multiplication. `exo.bridge.translate_weft_certificate_to_warp` expands `radix_scale_exp2` and stamps `cert_sha256`.
2. Five-layer binding and runtime-flag schemas between L2 `weft` and L3 `warp`. `weft` `BindingManifest` stores nested maps (`config_hashes`, `tokenizer_hashes`, `plane1_hashes`, `checkpoint_fingerprint`) and `RuntimeFlags` with `quantization_mode`, `mamba_cache_mode`, `enable_thinking`, `tool_call_parser`, `rope_scaling`, and `ctx_size`. `warp` `AdapterBundle` expects flat checkpoint digests (`config_sha256`, `tokenizer_sha256`, `chat_template_sha256`, `weights_sha256`, `safetensors_provenance_sha256`), `Plane1Lineage`, rational scales (`scale_num`, `scale_den`), and `RuntimeFlags` with `llama_server_commit`, `quantization`, `kv_cache_type`, `ctx_checkpoints`, `swa_checkpoints`, `lora_init_without_apply`, `rope_args`, and `chat_template_args`. `exo.bridge.build_warp_adapter_bundle` performs this deterministic translation.
3. Transport mismatch between earlier `exo` HTTP client and live `warp-modeld`. Earlier `WarpModeldBridge` posted JSON to HTTP `/lora-adapters` and `/rollback`, whereas live `warp-modeld` serves newline-delimited JSON-RPC 2.0 over a `0o600` Unix domain socket (`wmcp_slot_mount`, `warp_infer`, `warp_kill_switch`) and forbids raw `lora` overrides inside inference calls. `src/exo/client.py` now provides pure-stdlib `WarpUnixBridge` alongside `WarpModeldBridge`.
4. High-risk path defaults between L5 `mde` and L4 `exo`. `mde` flagged `security/`, `migrations/`, `infra/`, and `.github/` as high-risk prefixes while `exo` `DEFAULT_PROTECTED_PATH_PATTERNS` omitted those four prefixes. `src/exo/gate.py` now includes all four so L4 and L5 reject unapproved edits consistently.
