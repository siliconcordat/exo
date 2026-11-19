# Exo Initial Implementation Plan

This plan defines the staged build of `siliconcordat/exo`, the Layer 4 agent harness in the Siliconcordat stack. Every stage has concrete deliverables and falsifiable verification checks.

## Scope and boundaries

Exo owns agent memory, exact-byte human approval gates, causal self-improvement memory, durable capability charters, per-repository Layer 2 policy enforcement, and dynamic zone-scoped multi-MDE routing.

Exo does not own:

- CST parsing or structural patching (owned by L5 `siliconcordat/mde`).
- OS sandboxes, `mkosi` images, or GPU inference daemons (owned by L3 `siliconcordat/warp`).
- Weight compilation math or `.gpatch` binary encoding (owned by L2 `siliconcordat/weft`).
- Hardware roofline or DMA verification theory (owned by L1 `siliconcordat/loom`).

## Phase 0: Foundation (Step 1)

Deliverables:

- Clean public repository with Apache-2.0 `LICENSE`, `README.md`, `ARCHITECTURE.md`, and `PLAN.md`.
- Issue tracking the full L4 initial plan on `siliconcordat/exo`.

Exit gate:

- Zero leaked private content, secrets, or local machine paths.
- Prose lint and duplicate-reference checks exit 0 on every markdown file and issue body.

## Phase 1: Stdlib Harness Kernel and Adversarial Suite (Step 2)

Build the zero-dependency Python 3.10+ package (`pyproject.toml`, `src/exo/`, `tests/`) with six modules:

1. `src/exo/ledger.py`: Content-addressed append-only JSONL memory with `prev_sha256` hash chaining and chain verification.
2. `src/exo/gate.py`: Exact-byte human approval gate, structured argv wrapper analyzer, Git octal/non-ASCII path normalizer, and risk-class classifier (`routine` versus `high_risk`).
3. `src/exo/capabilities.py` and `src/exo/sim.py`: Continuous-synthesis durable capability registry, append-only SIM causal attempt log, and Cross-Generation Recompile/Retire Planner (`recompile`, `retire_candidate`, `escalate_human`).
4. `src/exo/policy.py` and `src/exo/router.py`: Declarative per-repository Layer 2 policy loader/validator (merge strategy, trailers, license, protected paths) and deterministic zone-scoped multi-MDE router covering `corpus`, `weft-eido`, and `formal-verification`.
5. `src/exo/client.py`: L5 `mde` JSON-RPC 2.0 MCP client (read-before-write enforcement + vendor `_meta`) and L3 `warp-modeld` HTTP/IPC bridge (receipt verification + `O(1)` rollback trigger).

Falsifiable exit gates (automated tests):

- **Shell-bypass immunity:** Multi-line pipe continuations (`curl ... |\n  ssh host`), option-value prefixes (`sudo --user`, `env -iu`, `nice --adjustment`, `timeout --sig`, `watch --int`), and Git-quoted non-ASCII or imported paths fail closed or classify `high_risk` with 0 misses.
- **Exact-byte lock:** Mutating any single byte of payload, path, or target after human approval fails verification immediately.
- **All 7 cross-MDE interference hazards tested:** Router deterministically activates the right specialist zone (`formal_proof`, `ui_module`, `condensed_json`, `prose`), enforces `data-eido-skip` on formal statements, protects exact-math vs float64 boundaries, and raises `ContestedPreferenceError` on unresolved conflicts (Hard Veto 2).
- **SIM cross-generation planner:** Correctly separates durable invariants (`recompile`), obsolete workarounds (`retire_candidate`), and contested rules (`escalate_human`) across `W_old -> W_new` upgrades.
- **Cross-org policy enforcement:** Correctly enforces squash/no-trailer (`corpus`), DCO/linear (`weft-eido`), and merge-commit/locked-checker (`Foundation-0`/`Proven-Commons`) rules from manifests without weight edits.
- **Transport contracts:** Mocked L5 JSON-RPC server and L3 `warp-modeld` HTTP server verify read-token ordering, scale routing, and rollback triggers end to end.

## Phase 2: Live L5 and L3 Integration

- Connect `exo` client directly to the live Rust `siliconcordat/mde` stdio MCP binary and live `siliconcordat/warp` `warp-modeld` daemon on local loopback.
- Verify end-to-end splice packet creation, approval challenge signing, `warp-modeld` scale application, and rollback flush on real tool traces.

## Phase 3: Multi-MDE Canary Runs on Three-Machine Topology

- Run `exo` across the 3-machine setup: clean macOS hub (27B read-only serving via `llama-server`), 8 GiB RTX 3080 Linux lab (`MDE#4` running 1.7B to 8B multi-MDE experiments), and lightweight Linux client.
- Log every live failure and upgrade transition in SIM causal memory and verify zero silent regressions across base model swaps.
