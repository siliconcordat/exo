# exo

Exo is the Layer 4 agent harness in the Siliconcordat stack. It sits between human authority and machine execution so coding and research agents can act across repositories without bypassing approval gates, losing why a rule exists, or colliding across domain adapters.

Exo is written for people and teams running multi-agent workflows over self-hosted open-weights models and frontier tools. It uses only the Python standard library in its core kernel so every gate is inspectable, deterministic, and easy to audit.

## Position in the stack

| Layer | Repository | Role |
| --- | --- | --- |
| L5 Improved MCP | [siliconcordat/mde](https://github.com/siliconcordat/mde) | Five-verb structural code broker, syntactic zone tags, and W-MCP metadata |
| L4 Harness | **siliconcordat/exo** | Single append-only memory, exact-byte approval gates, causal memory, and multi-MDE routing |
| L3 OS | [siliconcordat/warp](https://github.com/siliconcordat/warp) | Load-time weight verification, process sandboxing, and rollback control |
| L2 Model | [siliconcordat/weft](https://github.com/siliconcordat/weft) | Composable parametric adapters compiled from plain-text specifications |
| L1 Hardware | [siliconcordat/loom](https://github.com/siliconcordat/loom) | Hardware theory and verification physics research |

## What Exo enforces

- **Exact-byte human approvals**: Every high-risk action binds human sign-off to the exact SHA-256 of its canonical payload. A one-byte change voids approval. Raw shell text is rejected at the gate instead of parsed with fragile regexes.
- **Causal self-improvement memory and durable capabilities**: Every harness rule and domain adapter tracks its stable capability ID, what problem it solved, what it predicted, what broke, and which base model checkpoint it targeted. When a base model upgrades, Exo classifies each rule as recompile, retire candidate, or escalate to human.
- **Dynamic zone-scoped multi-MDE routing**: Exo reads syntactic zone tags from L5 (`prose`, `code`, `formal_proof`, `ui_module`, `condensed_json`) and computes exact per-request adapter scales across `corpus`, `weft-eido`, and `formal-verification`. Contested preferences never resolve silently.
- **Per-repository Layer 2 policy manifests**: Git merge strategy, commit trailer rules (DCO sign-off versus no trailers), licenses, and protected oracle paths live in declarative per-repo manifests, never inside model weights.

## Repository status

Status: Step 1 foundation (architecture and initial plan). The runnable Python stdlib kernel (`src/exo/`) and automated test suite (`tests/`) land in Step 2.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): Trust firewall, core module contracts, interference matrix, and Hard Vetoes compliance.
- [PLAN.md](PLAN.md): Phased implementation roadmap, falsifiable exit gates, and scope boundaries.
- [LICENSE](LICENSE): Apache License 2.0 for code. Documentation is licensed under CC BY 4.0.
