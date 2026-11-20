# exo

Exo is the Layer 4 agent harness in the Siliconcordat stack. It sits between human authority and machine execution so coding and research agents can act across repositories without bypassing approval gates, losing why a rule exists, or colliding across domain adapters.

Exo is written for people and teams running multi-agent workflows over self-hosted open-weights models and frontier tools. The core kernel uses only the Python 3.10+ standard library so every gate is inspectable, deterministic, and runnable offline with zero external runtime dependencies.

## Position in the stack

| Layer | Repository | Role |
| --- | --- | --- |
| L5 Improved MCP | [siliconcordat/mde](https://github.com/siliconcordat/mde) | Five-verb structural code broker, syntactic zone tags, and W-MCP metadata |
| L4 Harness | **siliconcordat/exo** | Single append-only memory, exact-byte approval gates, causal memory, and multi-MDE routing |
| L3 OS | [siliconcordat/warp](https://github.com/siliconcordat/warp) | Load-time weight verification, process sandboxing, and rollback control |
| L2 Model | [siliconcordat/weft](https://github.com/siliconcordat/weft) | Composable parametric adapters compiled from plain-text specifications |
| L1 Hardware | [siliconcordat/loom](https://github.com/siliconcordat/loom) | Hardware theory and verification physics research |

## Modules

- `src/exo/ledger.py`: Content-addressed append-only JSONL memory (`AppendOnlyLedger`) with `prev_sha256` hash chaining, optional HMAC-SHA256 verification, and pre-JSON raw-string secret redaction (`redact_secrets_in_value`).
- `src/exo/gate.py`: Exact-byte human approval gate (`ApprovalGate`), monotonic human stop and takeover fence (`ContinuationToken`, `raise_human_stop`, `clear_human_stop`), authoritative allowlist environment scrubber (`scrub_tool_env`), wrapper-aware `argv` unwrapper, Git octal/non-ASCII path normalizer, and risk-class classifier (`routine` versus `high_risk`). Any one-byte mutation invalidates approval.
- `src/exo/capabilities.py`: Durable capability charters (`CapabilityRegistry`) surviving file moves and rewrites, with digest-pinned peer responses.
- `src/exo/sim.py`: Append-only Self-Improvement Memory (`SIMMemory`) with pre-JSON secret redaction and `CrossGenerationPlanner` classifying rules across base model upgrades into `recompile`, `retire_candidate`, or `escalate_human`.
- `src/exo/policy.py`: Declarative per-repository Layer 2 policy manifests (`RepoPolicy`) enforcing merge strategy, commit trailers (DCO versus none), CI status, and locked-checker separation strictly outside model weights.
- `src/exo/router.py`: Dynamic zone-scoped multi-MDE router (`ZoneRouter`) supporting native L5 `zone-manifest/v0.1` SHA-256 verification and compilation (`ZoneRouter.from_zone_manifest`), mapping syntactic zones across `corpus`, `weft-eido`, and `formal-verification`, enforcing `data-eido-skip`, and raising `ContestedPreferenceError` under Hard Veto 2.
- `src/exo/client.py`: L5 `mde` JSON-RPC 2.0 client (`MdeClient`, in-memory and scrubbed-env stdio subprocess transports) enforcing session `read_tokens` before `splice_patch`, plus L3 `warp-modeld` HTTP (`WarpModeldBridge`) and Unix domain socket JSON-RPC 2.0 (`WarpUnixBridge`) bridges verifying load receipts and triggering `O(1)` rollback with KV and recurrent-state flush on drift.
- `src/exo/bridge.py`: Pure-stdlib cross-layer schema translator with strict integer type checks converting L2 `weft` certificates, manifests, and L4 `RouteDecision` scale vectors into L3 `warp-modeld` `AdapterBundle` payloads without adding runtime dependencies.
- `src/exo/cli.py`: Command-line utility (`python3 -m exo`) for `verify-ledger`, `route-zone`, `check-policy`, and `plan-upgrade`.

## Run tests and CLI

Requires Python 3.10 or newer with no third-party packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Example CLI commands:

```bash
PYTHONPATH=src python3 -m exo route-zone --zone formal_proof --domains corpus,weft-eido,formal-verification --preset proven-commons --formal-claims
PYTHONPATH=src python3 -m exo check-policy --preset weft-eido --message "Fix layout" --merge-method linear src/cell-layout.js
```

## Status and verification boundary

The core stdlib kernel and offline test suite (38 tests) pass with zero external dependencies under both standard and `-O` optimized execution.

The live vertical stack tests (`tests/test_vertical_stack.py`, 2 tests) and `examples/vertical_splice_demo.py` pass end-to-end against compiled Rust `siliconcordat/mde` (including native `zone-manifest/v0.1` SHA-256 verification), `siliconcordat/weft` on synthetic Safetensors checkpoints, and `siliconcordat/warp` `warp-modeld` over Unix domain socket against `MockLlamaServer`. Live runs on 8B or 27B open-weights GPU checkpoints belong to Phase 3 in [PLAN.md](PLAN.md) and are not claimed here.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): Trust firewall, core module contracts, interference matrix, and Hard Vetoes compliance.
- [docs/contextual-compute.md](docs/contextual-compute.md): Proposed note on steering, the open parameter set, and a two-sided record.
- [docs/VERTICAL-INTEGRATION.md](docs/VERTICAL-INTEGRATION.md): Four-layer live vertical integration commands, measured results, and cross-layer schema findings.
- [LICENSE](LICENSE): Apache License 2.0 for code. Documentation is licensed under CC BY 4.0.
