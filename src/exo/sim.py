"""SIM causal self-improvement memory and cross-generation recompile/retire planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from exo.ledger import AppendOnlyLedger, canonical_json_bytes, utc_now_iso

VALID_RULE_KINDS = frozenset({"durable_invariant", "model_workaround", "preference"})
VALID_OUTCOMES = frozenset({"pending", "improved", "regressed", "neutral", "failed"})


@dataclass(frozen=True)
class SIMAttempt:
    """Single causal attempt recorded before and after evaluation (arXiv:2611.18856v1)."""

    round_id: int
    what: str
    predicted: str
    fixed: tuple[str, ...]
    missed: tuple[str, ...]
    base_checkpoint_sha: str
    corpus_sha: str
    outcome: str = "pending"
    files: tuple[str, ...] = ()
    domain: str = "general"
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "what": self.what,
            "predicted": self.predicted,
            "fixed": list(self.fixed),
            "missed": list(self.missed),
            "base_checkpoint_sha": self.base_checkpoint_sha,
            "corpus_sha": self.corpus_sha,
            "outcome": self.outcome,
            "files": list(self.files),
            "domain": self.domain,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SIMAttempt":
        return cls(
            round_id=int(raw["round_id"]),
            what=str(raw["what"]),
            predicted=str(raw["predicted"]),
            fixed=tuple(raw.get("fixed", ())),
            missed=tuple(raw.get("missed", ())),
            base_checkpoint_sha=str(raw["base_checkpoint_sha"]),
            corpus_sha=str(raw["corpus_sha"]),
            outcome=str(raw.get("outcome", "pending")),
            files=tuple(raw.get("files", ())),
            domain=str(raw.get("domain", "general")),
            timestamp_utc=str(raw.get("timestamp_utc", "")),
        )


@dataclass
class SIMRuleRecord:
    """Causal problem-and-attempts record bound to a durable capability_id."""

    rule_id: str
    capability_id: str
    name: str
    problem: str
    rule_kind: str  # "durable_invariant" | "model_workaround" | "preference"
    adapter_id: str  # "corpus" | "weft-eido" | "formal-verification" | "harness"
    base_checkpoint_sha: str
    corpus_sha: str
    contested: bool = False
    contested_reason: Optional[str] = None
    attempts: list[SIMAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "capability_id": self.capability_id,
            "name": self.name,
            "problem": self.problem,
            "rule_kind": self.rule_kind,
            "adapter_id": self.adapter_id,
            "base_checkpoint_sha": self.base_checkpoint_sha,
            "corpus_sha": self.corpus_sha,
            "contested": self.contested,
            "contested_reason": self.contested_reason,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SIMRuleRecord":
        return cls(
            rule_id=str(raw["rule_id"]),
            capability_id=str(raw["capability_id"]),
            name=str(raw["name"]),
            problem=str(raw["problem"]),
            rule_kind=str(raw["rule_kind"]),
            adapter_id=str(raw["adapter_id"]),
            base_checkpoint_sha=str(raw["base_checkpoint_sha"]),
            corpus_sha=str(raw["corpus_sha"]),
            contested=bool(raw.get("contested", False)),
            contested_reason=raw.get("contested_reason"),
            attempts=[SIMAttempt.from_dict(a) for a in raw.get("attempts", ())],
        )


class SIMMemory:
    """Append-only JSONL Self-Improvement Memory keyed by rule_id and capability_id."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        ledger: Optional[AppendOnlyLedger] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.ledger = ledger
        self._rules: dict[str, SIMRuleRecord] = {}
        if self.path is not None and self.path.exists():
            self._replay_disk()

    def _replay_disk(self) -> None:
        assert self.path is not None
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("op")
            if kind == "register_rule":
                rec = SIMRuleRecord.from_dict(event["record"])
                self._rules[rec.rule_id] = rec
            elif kind == "append_attempt":
                rid = event["rule_id"]
                self._rules[rid].attempts.append(SIMAttempt.from_dict(event["attempt"]))
            elif kind == "mark_contested":
                rid = event["rule_id"]
                self._rules[rid].contested = bool(event["contested"])
                self._rules[rid].contested_reason = event.get("reason")
            else:
                raise ValueError(f"unknown SIM journal op at line {line_no}: {kind!r}")

    def _write_event(self, event: Mapping[str, Any]) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json_bytes(event).decode("ascii") + "\n")

    def register_rule(self, record: SIMRuleRecord) -> SIMRuleRecord:
        if record.rule_kind not in VALID_RULE_KINDS:
            raise ValueError(f"invalid rule_kind: {record.rule_kind!r}")
        if not record.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        self._rules[record.rule_id] = record
        self._write_event({"op": "register_rule", "record": record.to_dict()})
        if self.ledger is not None:
            self.ledger.append(
                "sim_rule_registered",
                "exo-sim",
                {
                    "rule_id": record.rule_id,
                    "capability_id": record.capability_id,
                    "rule_kind": record.rule_kind,
                    "adapter_id": record.adapter_id,
                    "base_checkpoint_sha": record.base_checkpoint_sha,
                    "corpus_sha": record.corpus_sha,
                },
            )
        return record

    def record_attempt(self, rule_id: str, attempt: SIMAttempt) -> SIMAttempt:
        """Append an attempt without ever deleting earlier failed or superseded attempts."""
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        if attempt.outcome not in VALID_OUTCOMES:
            raise ValueError(f"invalid attempt outcome: {attempt.outcome!r}")
        if not attempt.predicted.strip():
            raise ValueError("attempt must log predicted effect before evaluation")
        self._rules[rule_id].attempts.append(attempt)
        self._write_event({"op": "append_attempt", "rule_id": rule_id, "attempt": attempt.to_dict()})
        if self.ledger is not None:
            self.ledger.append(
                "sim_attempt_recorded",
                "exo-sim",
                {
                    "rule_id": rule_id,
                    "round_id": attempt.round_id,
                    "outcome": attempt.outcome,
                    "base_checkpoint_sha": attempt.base_checkpoint_sha,
                    "corpus_sha": attempt.corpus_sha,
                },
            )
        return attempt

    def mark_contested(self, rule_id: str, reason: str, *, contested: bool = True) -> None:
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        self._rules[rule_id].contested = contested
        self._rules[rule_id].contested_reason = reason
        self._write_event({
            "op": "mark_contested",
            "rule_id": rule_id,
            "contested": contested,
            "reason": reason,
        })
        if self.ledger is not None:
            self.ledger.append(
                "sim_rule_contested",
                "exo-sim",
                {"rule_id": rule_id, "contested": contested, "reason": reason},
            )

    def rules(self) -> list[SIMRuleRecord]:
        return list(self._rules.values())

    def get(self, rule_id: str) -> SIMRuleRecord:
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        return self._rules[rule_id]


@dataclass(frozen=True)
class BaselineProbe:
    """Fresh baseline evaluation of a rule's problem on the upgraded checkpoint W_new."""

    rule_id: str
    passed_natively: bool
    regression_detected: bool = False
    peer_conflict: bool = False
    observed_corpus_sha: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class UpgradeAction:
    """Deterministic classification of one rule across a checkpoint upgrade."""

    rule_id: str
    capability_id: str
    adapter_id: str
    action: str  # "recompile" | "retire_candidate" | "escalate_human"
    reason: str
    target_checkpoint_sha: str


@dataclass(frozen=True)
class UpgradePlan:
    """Complete Cross-Generation Recompile/Retire Plan for W_old -> W_new."""

    old_checkpoint_sha: str
    new_checkpoint_sha: str
    active_corpus_sha: str
    actions: tuple[UpgradeAction, ...]

    def by_action(self, action: str) -> list[UpgradeAction]:
        return [a for a in self.actions if a.action == action]


class CrossGenerationPlanner:
    """Classifies active MDE and harness rules on checkpoint upgrade (W_old -> W_new).

    Enforces Hard Veto 2: contested rules, peer conflicts, preference drift, or
    unexpected regressions always yield ``escalate_human`` instead of silent
    automated resolution.
    """

    def plan_upgrade(
        self,
        sim: SIMMemory,
        *,
        old_checkpoint_sha: str,
        new_checkpoint_sha: str,
        active_corpus_sha: str,
        probes: Mapping[str, BaselineProbe] = {},
    ) -> UpgradePlan:
        actions: list[UpgradeAction] = []
        for rec in sorted(sim.rules(), key=lambda r: r.rule_id):
            probe = probes.get(rec.rule_id)
            action, reason = self._classify(rec, probe, active_corpus_sha)
            actions.append(
                UpgradeAction(
                    rule_id=rec.rule_id,
                    capability_id=rec.capability_id,
                    adapter_id=rec.adapter_id,
                    action=action,
                    reason=reason,
                    target_checkpoint_sha=new_checkpoint_sha,
                )
            )
        return UpgradePlan(
            old_checkpoint_sha=old_checkpoint_sha,
            new_checkpoint_sha=new_checkpoint_sha,
            active_corpus_sha=active_corpus_sha,
            actions=tuple(actions),
        )

    @staticmethod
    def _classify(
        rec: SIMRuleRecord,
        probe: Optional[BaselineProbe],
        active_corpus_sha: str,
    ) -> tuple[str, str]:
        # Hard Veto 2: never silently resolve contested rules or peer conflicts.
        if rec.contested:
            return "escalate_human", f"contested_rule:{rec.contested_reason or 'unspecified'}"
        if probe is not None and probe.peer_conflict:
            return "escalate_human", "peer_steward_conflict_on_new_checkpoint"
        if probe is not None and probe.regression_detected:
            return "escalate_human", "unexpected_regression_on_new_checkpoint"

        # Detect historical regression oscillation in SIM memory
        regressed_rounds = sum(1 for a in rec.attempts if a.outcome == "regressed")
        if regressed_rounds >= 2 and rec.rule_kind == "preference":
            return "escalate_human", "repeated_preference_regressions"

        if rec.rule_kind == "durable_invariant":
            if rec.corpus_sha != active_corpus_sha:
                return "recompile", "durable_invariant_corpus_updated"
            return "recompile", "durable_invariant_repin_to_new_checkpoint"

        if rec.rule_kind == "model_workaround":
            if probe is not None and probe.passed_natively:
                return "retire_candidate", "new_checkpoint_passes_natively_without_workaround"
            if probe is None:
                return "recompile", "workaround_unprobed_on_new_checkpoint_keep_pinned"
            return "recompile", "workaround_still_required_on_new_checkpoint"

        # rule_kind == "preference" (uncontested)
        if rec.corpus_sha != active_corpus_sha:
            return "escalate_human", "preference_corpus_sha_drifted_needs_human_confirmation"
        return "recompile", "authorized_preference_repin_to_new_checkpoint"
