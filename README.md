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

- `src/exo/ledger.py`: Content-addressed append-only JSONL memory (`AppendOnlyLedger`) with `prev_sha256` hash chaining and optional HMAC-SHA256 verification.
- `src/exo/gate.py`: Exact-byte human approval gate (`ApprovalGate`), wrapper-aware `argv` unwrapper, Git octal/non-ASCII path normalizer, and risk-class classifier (`routine` versus `high_risk`). Any one-byte mutation invalidates approval.
- `src/exo/capabilities.py`: Durable capability charters (`CapabilityRegistry`) surviving file moves and rewrites, with digest-pinned peer responses.
- `src/exo/sim.py`: Append-only Self-Improvement Memory (`SIMMemory`) and `CrossGenerationPlanner` classifying rules across base model upgrades into `recompile`, `retire_candidate`, or `escalate_human`.
- `src/exo/policy.py`: Declarative per-repository Layer 2 policy manifests (`RepoPolicy`) enforcing merge strategy, commit trailers (DCO versus none), CI status, and locked-checker separation strictly outside model weights.
- `src/exo/router.py`: Dynamic zone-scoped multi-MDE router (`ZoneRouter`) mapping syntactic zones (`prose`, `code`, `formal_proof`, `ui_module`, `condensed_json`, `research_page`, `html_report`) across `corpus`, `weft-eido`, and `formal-verification`, enforcing `data-eido-skip` and raising `ContestedPreferenceError` under Hard Veto 2.
- `src/exo/client.py`: L5 `mde` JSON-RPC 2.0 client (`MdeClient`, in-memory and stdio subprocess transports) enforcing session `read_tokens` before `splice_patch`, plus L3 `warp-modeld` HTTP bridge (`WarpModeldBridge`) verifying load receipts and triggering `O(1)` rollback with KV and recurrent-state flush on drift.
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

Step 2 kernel and unit/integration tests are implemented and pass locally against in-memory, stdio subprocess, and localhost HTTP mocks. Live end-to-end runs against the Rust `siliconcordat/mde` binary, live `siliconcordat/warp` `warp-modeld`, and real multi-MDE open-weights checkpoints belong to Phases 2 and 3 in [PLAN.md](PLAN.md) and are not claimed here.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): Trust firewall, core module contracts, interference matrix, and Hard Vetoes compliance.
- [LICENSE](LICENSE): Apache License 2.0 for code. Documentation is licensed under CC BY 4.0.
