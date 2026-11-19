# Exo Architecture

Exo is the Layer 4 harness of the Siliconcordat stack. It coordinates agent memory, human approvals, repository policy, and runtime adapter selection while keeping deterministic enforcement strictly outside model weights.

## 1. Three-tier trust firewall

Exo separates authority, deterministic enforcement, and model habits into three strict tiers:

1. **Layer 1: Human Authority.** Approval of high-risk actions, contested preference rulings, spend above threshold, external public posts, credential custody, and live production deploys. Layer 1 never compiles into model weights.
2. **Layer 2: Deterministic External Enforcement.** Exo's exact-byte approval gate, per-repository policy manifest, L5 `mde` structural broker locks, and L3 `warp-modeld` load-time verification. Layer 2 fails closed outside the model.
3. **Layer 3: Revocable Parametric Priors.** L2 `weft` domain adapters (`corpus`, `weft-eido`, `formal-verification`) running on self-hosted open weights. Adapters shape defaults and fluency. They carry zero authority and never override Layer 1 or Layer 2.

## 2. Core components

### 2.1 Single continuous append-only signed memory

Session context is ephemeral, so Exo writes every harness event (decisions, approval challenges, human signatures, tool receipts, causal attempts, and adapter scale selections) to an append-only JSONL ledger. Each entry pins `prev_sha256` and `entry_sha256` over canonical sorted JSON so tampering or missing history fails verification immediately.

### 2.2 Exact-byte human approval gate (`src/exo/gate.py`)

Earlier shell-text gates broke in three concrete ways:

1. Multi-line pipelines ending in a trailing pipe (`curl ... |\n  ssh host`) split the command across lines and bypassed stdin inspection.
2. Command prefixes and wrappers whose options take a separate value argument (`sudo --user bob`, `env -iu VAR`, `nice --adjustment 5`, `timeout --sig KILL 5`, `watch --int 5`, `stdbuf --output L`) fooled token skippers into reading the option value as the executable name.
3. Preflight diff filters missed imported source paths and Git-quoted non-ASCII filenames (such as octal-escaped UTF-8 paths).

Exo eliminates this whole error class mechanically:

- **Structured calls only.** The gate evaluates typed action envelopes (`tool`, `argv`, `paths`, `payload_bytes`, `target`). Raw shell strings containing pipes, redirections, subshells, command substitution, or unparsed newlines fail closed as `high_risk` and require explicit structured decomposition or human sign-off.
- **Wrapper-aware argv inspection.** Known privilege and process wrappers parse option-value pairs explicitly, and any unknown wrapper flag or nested shell (`sh -c`, `bash -c`, `eval`, `ssh`) escalates immediately to `high_risk`.
- **Canonical path normalization.** Git C-style octal-escaped paths and non-ASCII filenames are decoded before matching protected paths, imported module roots, test oracles, schemas, or auth surfaces.
- **Exact-byte lock.** Approval challenges compute `challenge_sha256 = SHA-256(canonical_action_envelope)`. A human approval token binds strictly to that digest. Changing a single byte of diff, text, target, or argument invalidates the approval.

Risk classes:

- `routine`: Read-only queries, bounded edits inside non-protected code or docs paths backed by valid L5 read tokens.
- `high_risk`: Auth or credential changes, schema or public API changes, protected test/oracle edits, file deletions, force-pushes, external posts, production deploys, spend above limit, or raw shell execution.

### 2.3 Causal self-improvement memory and durable capabilities (`src/exo/sim.py`, `src/exo/capabilities.py`)

Policies and prompt patches rot when the underlying base model changes. Drawing on continuous capability stewardship and Self-Improvement Memory (SIM, arXiv:2611.18856v1), Exo binds every rule and adapter to two durable records:

- **Capability Charter (`capabilities.py`):** Assigns a stable `capability_id` independent of file paths, naming human purpose, success signals, failure signals, scope, and affected peer capabilities. File moves or rewrites update links without losing purpose.
- **SIM Causal Record (`sim.py`):** Append-only JSONL history per problem or invariant: `problem -> attempts[{what, predicted, fixed, missed, base_checkpoint_sha, corpus_sha, outcome, rule_kind}]`. Predicted effects are logged before evaluation, and failed or superseded attempts remain on record.
- **Cross-Generation Recompile/Retire Planner:** When the base checkpoint upgrades (`W_old -> W_new`), the planner inspects every active rule and adapter alongside fresh baseline probe results on `W_new` and classifies it into one of three deterministic outcomes:
  - `recompile`: Durable specification invariant or policy still required on `W_new`.
  - `retire_candidate`: Workaround introduced strictly for a bug on `W_old` that `W_new` now passes natively with zero adapter assistance.
  - `escalate_human`: Contested rule, conflicting peer evidence, or unexplained regression across checkpoints (enforcing Hard Veto 2).

### 2.4 Dynamic zone-scoped multi-MDE router and repo policy (`src/exo/router.py`, `src/exo/policy.py`)

Running three domain adapters simultaneously (`corpus` base, `weft-eido` design, `formal-verification` proofs) creates seven verified cross-domain interference hazards:

1. `corpus` bans stacking negative-absence statements, while formal verification requires explicit `What is not proved` and `What you must trust` sections.
2. `corpus` bans technical jargon and long hashes in prose, while formal verification requires exact Lean terms and 40-character commit hashes in verdict records.
3. Claim-tier vocabularies collide (`prepared/deployed` versus `PROVEN/VERIFIED/KNOWN/CONJECTURE` versus `in-progress/verified`).
4. `corpus` bans telegraphic fragments, while `weft-eido` `condensed.json` Level 2 (`Terse`) and Level 3 (`Caveman`) require dropped grammar.
5. `corpus` expects dark static HTML with `assets/` and `?v=` cache-busters, while `weft-eido` uses dual light/dark OKLCH palettes and root-level zero-build ES modules.
6. `weft-eido` uses IEEE-754 floats and `illustrative: true` toy canvases, while formal verification bans float literals and `/` division inside exact checkers and binds research plots to `certificate.json`.
7. Dynamic prose condensing over pinned formal specs would break spec hashes or strip hedges unless `data-eido-skip` protects formal blocks.

Exo resolves these hazards deterministically in Layer 2 without letting weight sums resolve conflicts silently:

- **Syntactic zone routing (`router.py`):** Consumes L5 `mde` zone tags (`prose`, `code`, `formal_proof`, `ui_module`, `condensed_json`) plus active task domains and outputs an exact per-request `weft` LoRA scale vector (`[{"id": ..., "scale": ...}]`).
  - `prose`: `corpus = 1.0`, `weft-eido = 0.0`, `formal-verification = 0.0`.
  - `formal_proof`: `formal-verification = 1.0`, `corpus = 0.0` (yielding prose rules inside Lean, JSON verdicts, and mandatory limitation blocks), `weft-eido = 0.0`, with `data-eido-skip` enforced.
  - `ui_module` / `condensed_json`: `weft-eido = 1.0`, `corpus = 0.0` (so Terse/Caveman slots and OKLCH modules stay intact), `formal-verification = 0.0` unless the file also holds certified research plots, in which case `certificate_binding` and `illustrative` checks activate.
  - **Contested overlap (Hard Veto 2):** When two active rules conflict on the same output zone without an explicit Plane 1 precedence flag (for example, static dark HTML report versus reactive `weft-eido` page), the router raises a structured `ContestedPreferenceError` and escalates to human authority instead of averaging scales.
- **Per-repository Layer 2 policy manifest (`policy.py`):** Git merge strategy (`squash` versus `merge`), commit trailers (`dco` `Signed-off-by:` versus `none`), license headers, CI status, and protected paths differ across organizations. Every repository loads a declarative `repo-policy.json` enforced by Exo before commit or PR actions. None of these mechanics enter model weights.

### 2.5 L5 `mde` MCP client and L3 `warp-modeld` bridge (`src/exo/client.py`)

- **L5 `mde` client:** Speaks JSON-RPC 2.0 over stdio or pluggable transport to invoke the five structural verbs (`splice_locate`, `splice_trace`, `splice_patch`, `splice_verify`, `splice_project`), enforcing that `splice_patch` never runs without valid session `read_tokens` and carrying vendor-namespaced `_meta` (`gg.siliconcordat/splice`, `gg.siliconcordat/mde`).
- **L3 `warp-modeld` bridge:** Submits desired adapter scale vectors to L3 `warp-modeld`, verifies signed load receipts and active checkpoint digests before dispatching inference, and invokes `O(1)` adapter disable plus KV and recurrent-state flush whenever a gate fails.

## 3. Compliance with the Five Hard Vetoes

1. **No opaque deltas:** Exo treats every L2 adapter as a revocable cache pinned to Plane 1 source digests and base checkpoint SHAs.
2. **No automatic conflict resolution:** Contested style or cross-generation rule conflicts raise explicit escalation records instead of silent averaging.
3. **Load-time enforcement and kill switch:** Exo delegates open-weights loading to L3 `warp-modeld` and drops adapter scales to `0.0` immediately on receipt mismatch.
4. **No premature protocol standardization:** All splice and weight metadata stays inside vendor-namespaced `_meta` fields over standard JSON-RPC 2.0.
5. **Plane 1 holds human truth:** Append-only JSONL memory, capability charters, SIM records, and repo policy manifests remain plain human-readable files under version control.
