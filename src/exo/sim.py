"""SIM causal self-improvement memory and cross-generation recompile/retire planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

from exo.ledger import (
    AppendOnlyLedger,
    LedgerEntry,
    canonical_json_bytes,
    redact_secrets_in_text,
    redact_secrets_in_value,
    utc_now_iso,
)

VALID_RULE_KINDS = frozenset({"durable_invariant", "model_workaround", "preference"})
VALID_OUTCOMES = frozenset({"pending", "improved", "regressed", "neutral", "failed"})
RESOLVED_OUTCOMES = frozenset({"improved", "regressed", "neutral", "failed"})
EVALUATION_RECEIPT_EVENT_TYPE = "sim_evaluation_receipt"
BASELINE_PROBE_RECEIPT_EVENT_TYPE = "sim_baseline_probe_receipt"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTERED_METADATA_FIELDS = (
    "capability_id",
    "rule_kind",
    "adapter_id",
    "base_checkpoint_sha",
    "corpus_sha",
)


class SIMEvidenceError(ValueError):
    """Raised when a SIM attempt outcome or baseline probe lacks verified ledger evidence."""


def _require_exact_int(value: Any, name: str, *, min_value: int = 1) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int (got {type(value).__name__})")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value} (got {value})")
    return value


def _require_exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool (got {type(value).__name__})")
    return value


def _require_non_empty_str(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256_hex(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
        raise SIMEvidenceError(f"{name} must be a 64-character lowercase hex SHA-256 digest")
    return value


def record_evaluation_receipt(
    ledger: AppendOnlyLedger,
    *,
    rule_id: str,
    round_id: int,
    outcome: str,
    base_checkpoint_sha: str,
    corpus_sha: str,
    actor_id: str = "exo-evaluator",
    details: Optional[Mapping[str, Any]] = None,
    signing_key: Optional[bytes] = None,
) -> LedgerEntry:
    """Append a verifiable SIM evaluation receipt to the ledger and return its entry."""
    _require_non_empty_str(rule_id, "rule_id")
    _require_exact_int(round_id, "round_id", min_value=1)
    if outcome not in RESOLVED_OUTCOMES:
        raise ValueError(f"evaluation receipt requires a resolved outcome (got {outcome!r})")
    _require_non_empty_str(base_checkpoint_sha, "base_checkpoint_sha")
    _require_non_empty_str(corpus_sha, "corpus_sha")
    payload: dict[str, Any] = {
        "rule_id": rule_id,
        "round_id": round_id,
        "outcome": outcome,
        "base_checkpoint_sha": base_checkpoint_sha,
        "corpus_sha": corpus_sha,
    }
    if details is not None:
        payload["details"] = dict(details)
    return ledger.append(
        EVALUATION_RECEIPT_EVENT_TYPE,
        actor_id,
        payload,
        signing_key=signing_key,
    )


def record_baseline_probe_receipt(
    ledger: AppendOnlyLedger,
    *,
    rule_id: str,
    checkpoint_sha: str,
    passed_natively: bool,
    bare_model: bool = True,
    regression_detected: bool = False,
    peer_conflict: bool = False,
    observed_corpus_sha: Optional[str] = None,
    actor_id: str = "exo-probe",
    notes: str = "",
    signing_key: Optional[bytes] = None,
) -> LedgerEntry:
    """Append a verifiable bare-model baseline probe receipt on W_new to the ledger."""
    _require_non_empty_str(rule_id, "rule_id")
    _require_non_empty_str(checkpoint_sha, "checkpoint_sha")
    _require_exact_bool(passed_natively, "passed_natively")
    _require_exact_bool(bare_model, "bare_model")
    _require_exact_bool(regression_detected, "regression_detected")
    _require_exact_bool(peer_conflict, "peer_conflict")
    if observed_corpus_sha is not None:
        _require_non_empty_str(observed_corpus_sha, "observed_corpus_sha")
    payload: dict[str, Any] = {
        "rule_id": rule_id,
        "checkpoint_sha": checkpoint_sha,
        "bare_model": bare_model,
        "passed_natively": passed_natively,
        "regression_detected": regression_detected,
        "peer_conflict": peer_conflict,
        "observed_corpus_sha": observed_corpus_sha,
        "notes": str(notes),
    }
    return ledger.append(
        BASELINE_PROBE_RECEIPT_EVENT_TYPE,
        actor_id,
        payload,
        signing_key=signing_key,
    )


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
    evidence_receipt_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _require_exact_int(self.round_id, "round_id", min_value=1)
        if self.evidence_receipt_sha256 is not None:
            _require_sha256_hex(self.evidence_receipt_sha256, "evidence_receipt_sha256")
            if self.outcome == "pending":
                raise SIMEvidenceError("pending attempt must not carry evidence_receipt_sha256")

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
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SIMAttempt":
        round_id = _require_exact_int(raw["round_id"], "round_id", min_value=1)
        return cls(
            round_id=round_id,
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
            evidence_receipt_sha256=raw.get("evidence_receipt_sha256"),
        )


def verify_attempt_evidence(
    ledger: Optional[AppendOnlyLedger],
    *,
    rule_id: str,
    attempt: SIMAttempt,
    signing_key: Optional[bytes] = None,
    require_signatures: bool = False,
) -> Optional[LedgerEntry]:
    """Verify that a non-pending SIMAttempt resolves to a matching ledger evaluation receipt
    and that a pending attempt does not downgrade an already-resolved ledger evaluation."""
    if attempt.outcome not in VALID_OUTCOMES:
        raise ValueError(f"invalid attempt outcome: {attempt.outcome!r}")
    if attempt.outcome == "pending":
        if ledger is not None:
            try:
                eval_entries = ledger.verified_entries(
                    event_type=EVALUATION_RECEIPT_EVENT_TYPE,
                    signing_key=signing_key,
                    require_signatures=require_signatures,
                )
            except ValueError as exc:
                raise SIMEvidenceError(str(exc)) from exc
            for entry in eval_entries:
                p = entry.payload
                if (
                    p.get("rule_id") == rule_id
                    and type(p.get("round_id")) is int
                    and p.get("round_id") == attempt.round_id
                    and p.get("base_checkpoint_sha") == attempt.base_checkpoint_sha
                    and p.get("corpus_sha") == attempt.corpus_sha
                    and p.get("outcome") in RESOLVED_OUTCOMES
                ):
                    raise SIMEvidenceError(
                        f"pending attempt for rule {rule_id!r} round {attempt.round_id} conflicts with "
                        f"resolved evaluation receipt {entry.entry_sha256}"
                    )
        return None
    if attempt.evidence_receipt_sha256 is None:
        raise SIMEvidenceError(
            f"attempt outcome {attempt.outcome!r} for rule {rule_id!r} requires evidence_receipt_sha256"
        )
    if ledger is None:
        raise SIMEvidenceError(
            f"attempt outcome {attempt.outcome!r} for rule {rule_id!r} requires a ledger to verify evidence"
        )
    try:
        entry = ledger.resolve_verified_entry(
            attempt.evidence_receipt_sha256,
            expected_event_type=EVALUATION_RECEIPT_EVENT_TYPE,
            signing_key=signing_key,
            require_signatures=require_signatures,
        )
    except ValueError as exc:
        raise SIMEvidenceError(str(exc)) from exc

    payload = entry.payload
    if payload.get("rule_id") != rule_id:
        raise SIMEvidenceError(
            f"evaluation receipt rule_id mismatch: expected {rule_id!r}, got {payload.get('rule_id')!r}"
        )
    receipt_round = payload.get("round_id")
    if type(receipt_round) is not int or receipt_round != attempt.round_id:
        raise SIMEvidenceError(
            f"evaluation receipt round_id mismatch: expected {attempt.round_id!r}, got {receipt_round!r}"
        )
    if payload.get("outcome") != attempt.outcome:
        raise SIMEvidenceError(
            f"evaluation receipt outcome mismatch: expected {attempt.outcome!r}, got {payload.get('outcome')!r}"
        )
    if payload.get("base_checkpoint_sha") != attempt.base_checkpoint_sha:
        raise SIMEvidenceError("evaluation receipt base_checkpoint_sha mismatch")
    if payload.get("corpus_sha") != attempt.corpus_sha:
        raise SIMEvidenceError("evaluation receipt corpus_sha mismatch")
    return entry


def _verify_rule_state_against_ledger(
    ledger: AppendOnlyLedger,
    record: "SIMRuleRecord",
    *,
    signing_key: Optional[bytes] = None,
    require_signatures: bool = False,
) -> None:
    """Cross-check registered metadata, contested state, and resolved receipt completeness against ledger."""
    try:
        entries = ledger.verified_entries(
            signing_key=signing_key,
            require_signatures=require_signatures,
        )
    except ValueError as exc:
        raise SIMEvidenceError(str(exc)) from exc

    latest_reg: Optional[LedgerEntry] = None
    has_contested_state = False
    expected_contested: Any = None
    resolved_eval_entries: list[LedgerEntry] = []

    for entry in entries:
        payload = entry.payload
        if payload.get("rule_id") != record.rule_id:
            continue
        if entry.event_type == "sim_rule_registered":
            latest_reg = entry
            if "contested" in payload:
                expected_contested = payload["contested"]
                has_contested_state = True
        elif entry.event_type == "sim_rule_contested":
            expected_contested = payload.get("contested")
            has_contested_state = True
        elif entry.event_type == EVALUATION_RECEIPT_EVENT_TYPE and payload.get("outcome") in RESOLVED_OUTCOMES:
            resolved_eval_entries.append(entry)

    if latest_reg is not None:
        for field_name in _REGISTERED_METADATA_FIELDS:
            ledger_val = latest_reg.payload.get(field_name)
            journal_val = getattr(record, field_name)
            if ledger_val != journal_val:
                raise SIMEvidenceError(
                    f"{field_name} mismatch for rule {record.rule_id!r}: journal has {journal_val!r}, "
                    f"ledger registration recorded {ledger_val!r}"
                )

    if has_contested_state and (type(expected_contested) is not bool or expected_contested is not record.contested):
        raise SIMEvidenceError(
            f"contested state mismatch for rule {record.rule_id!r}: journal has {record.contested!r}, "
            f"ledger recorded {expected_contested!r}"
        )

    recorded_receipts = {
        att.evidence_receipt_sha256
        for att in record.attempts
        if att.outcome in RESOLVED_OUTCOMES and att.evidence_receipt_sha256 is not None
    }
    for eval_entry in resolved_eval_entries:
        if eval_entry.entry_sha256 not in recorded_receipts:
            raise SIMEvidenceError(
                f"missing resolved attempt for rule {record.rule_id!r} ledger evaluation receipt "
                f"{eval_entry.entry_sha256}"
            )


_LEDGER_RULE_STATE_EVENTS = frozenset({"sim_rule_registered", "sim_rule_contested"})


def _verify_ledger_rule_completeness(
    ledger: AppendOnlyLedger,
    known_rule_ids: set[str],
    *,
    signing_key: Optional[bytes] = None,
    require_signatures: bool = False,
) -> None:
    """Require every rule referenced by ledger registration, contest, or resolved receipt to exist in SIM."""
    try:
        entries = ledger.verified_entries(
            signing_key=signing_key,
            require_signatures=require_signatures,
        )
    except ValueError as exc:
        raise SIMEvidenceError(str(exc)) from exc

    for entry in entries:
        payload = entry.payload
        is_rule_state = entry.event_type in _LEDGER_RULE_STATE_EVENTS
        is_resolved_eval = (
            entry.event_type == EVALUATION_RECEIPT_EVENT_TYPE
            and payload.get("outcome") in RESOLVED_OUTCOMES
        )
        if not (is_rule_state or is_resolved_eval):
            continue
        rid = payload.get("rule_id")
        if type(rid) is not str or rid not in known_rule_ids:
            raise SIMEvidenceError(
                f"ledger {entry.event_type} entry {entry.entry_sha256} references rule_id "
                f"{rid!r} omitted from SIM journal"
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
            contested=_require_exact_bool(raw.get("contested", False), "contested"),
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
        signing_key: Optional[bytes] = None,
        require_signatures: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.ledger = ledger
        self.signing_key = signing_key
        self.require_signatures = require_signatures
        self._rules: dict[str, SIMRuleRecord] = {}
        if self.path is not None and self.path.exists():
            self._replay_disk()

    def _replay_disk(self) -> None:
        if self.path is None:
            raise RuntimeError("cannot load SIM journal from disk when path is None")
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SIMEvidenceError(f"cannot read SIM journal {self.path}: {exc}") from exc
        for line_no, line in enumerate(raw_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise SIMEvidenceError(f"malformed SIM journal JSON at line {line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise SIMEvidenceError(f"non-object SIM journal record at line {line_no}")
            kind = event.get("op")
            if kind == "register_rule":
                try:
                    rec = SIMRuleRecord.from_dict(event["record"])
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    raise SIMEvidenceError(f"invalid register_rule record at line {line_no}: {exc}") from exc
                if rec.rule_kind not in VALID_RULE_KINDS:
                    raise SIMEvidenceError(f"invalid rule_kind at line {line_no}: {rec.rule_kind!r}")
                if self.ledger is not None:
                    for att in rec.attempts:
                        verify_attempt_evidence(
                            self.ledger,
                            rule_id=rec.rule_id,
                            attempt=att,
                            signing_key=self.signing_key,
                            require_signatures=self.require_signatures,
                        )
                self._rules[rec.rule_id] = rec
            elif kind == "append_attempt":
                rid = event.get("rule_id")
                if type(rid) is not str or rid not in self._rules:
                    raise SIMEvidenceError(f"append_attempt references unknown rule_id at line {line_no}: {rid!r}")
                try:
                    att = SIMAttempt.from_dict(event["attempt"])
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    raise SIMEvidenceError(f"invalid append_attempt at line {line_no}: {exc}") from exc
                if self.ledger is not None:
                    verify_attempt_evidence(
                        self.ledger,
                        rule_id=rid,
                        attempt=att,
                        signing_key=self.signing_key,
                        require_signatures=self.require_signatures,
                    )
                self._rules[rid].attempts.append(att)
            elif kind == "mark_contested":
                rid = event.get("rule_id")
                if type(rid) is not str or rid not in self._rules or type(event.get("contested")) is not bool:
                    raise SIMEvidenceError(f"invalid mark_contested event at line {line_no}")
                self._rules[rid].contested = event["contested"]
                self._rules[rid].contested_reason = event.get("reason")
            else:
                raise ValueError(f"unknown SIM journal op at line {line_no}: {kind!r}")
        if self.ledger is not None:
            for rec in self._rules.values():
                _verify_rule_state_against_ledger(
                    self.ledger,
                    rec,
                    signing_key=self.signing_key,
                    require_signatures=self.require_signatures,
                )
            _verify_ledger_rule_completeness(
                self.ledger,
                set(self._rules.keys()),
                signing_key=self.signing_key,
                require_signatures=self.require_signatures,
            )

    def _write_event(self, event: Mapping[str, Any]) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            safe_event = redact_secrets_in_value(event)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json_bytes(safe_event).decode("ascii") + "\n")

    def register_rule(self, record: SIMRuleRecord) -> SIMRuleRecord:
        if record.rule_kind not in VALID_RULE_KINDS:
            raise ValueError(f"invalid rule_kind: {record.rule_kind!r}")
        if not record.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        safe_record = SIMRuleRecord.from_dict(redact_secrets_in_value(record.to_dict()))
        for att in safe_record.attempts:
            verify_attempt_evidence(
                self.ledger,
                rule_id=safe_record.rule_id,
                attempt=att,
                signing_key=self.signing_key,
                require_signatures=self.require_signatures,
            )
        self._rules[safe_record.rule_id] = safe_record
        self._write_event({"op": "register_rule", "record": safe_record.to_dict()})
        if self.ledger is not None:
            self.ledger.append(
                "sim_rule_registered",
                "exo-sim",
                {
                    "rule_id": safe_record.rule_id,
                    "capability_id": safe_record.capability_id,
                    "rule_kind": safe_record.rule_kind,
                    "adapter_id": safe_record.adapter_id,
                    "base_checkpoint_sha": safe_record.base_checkpoint_sha,
                    "corpus_sha": safe_record.corpus_sha,
                    "contested": safe_record.contested,
                },
                signing_key=self.signing_key,
            )
        return safe_record

    def record_attempt(self, rule_id: str, attempt: SIMAttempt) -> SIMAttempt:
        """Append an attempt without ever deleting earlier failed or superseded attempts."""
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        if not attempt.predicted.strip():
            raise ValueError("attempt must log predicted effect before evaluation")
        safe_attempt = SIMAttempt.from_dict(redact_secrets_in_value(attempt.to_dict()))
        verify_attempt_evidence(
            self.ledger,
            rule_id=rule_id,
            attempt=safe_attempt,
            signing_key=self.signing_key,
            require_signatures=self.require_signatures,
        )
        self._rules[rule_id].attempts.append(safe_attempt)
        self._write_event({"op": "append_attempt", "rule_id": rule_id, "attempt": safe_attempt.to_dict()})
        if self.ledger is not None:
            self.ledger.append(
                "sim_attempt_recorded",
                "exo-sim",
                {
                    "rule_id": rule_id,
                    "round_id": safe_attempt.round_id,
                    "outcome": safe_attempt.outcome,
                    "base_checkpoint_sha": safe_attempt.base_checkpoint_sha,
                    "corpus_sha": safe_attempt.corpus_sha,
                    "evidence_receipt_sha256": safe_attempt.evidence_receipt_sha256,
                },
                signing_key=self.signing_key,
            )
        return safe_attempt

    def mark_contested(self, rule_id: str, reason: str, *, contested: bool = True) -> None:
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        _require_exact_bool(contested, "contested")
        safe_reason = redact_secrets_in_text(reason)
        self._rules[rule_id].contested = contested
        self._rules[rule_id].contested_reason = safe_reason
        self._write_event({
            "op": "mark_contested",
            "rule_id": rule_id,
            "contested": contested,
            "reason": safe_reason,
        })
        if self.ledger is not None:
            self.ledger.append(
                "sim_rule_contested",
                "exo-sim",
                {"rule_id": rule_id, "contested": contested, "reason": safe_reason},
                signing_key=self.signing_key,
            )

    def rules(self) -> list[SIMRuleRecord]:
        return list(self._rules.values())

    def get(self, rule_id: str) -> SIMRuleRecord:
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule_id: {rule_id}")
        return self._rules[rule_id]


@dataclass(frozen=True)
class BaselineProbe:
    """Fresh bare-model baseline evaluation of a rule's problem on the upgraded checkpoint W_new."""

    rule_id: str
    passed_natively: bool
    regression_detected: bool = False
    peer_conflict: bool = False
    observed_corpus_sha: Optional[str] = None
    notes: str = ""
    evidence_receipt_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _require_exact_bool(self.passed_natively, "passed_natively")
        _require_exact_bool(self.regression_detected, "regression_detected")
        _require_exact_bool(self.peer_conflict, "peer_conflict")
        if self.evidence_receipt_sha256 is not None:
            _require_sha256_hex(self.evidence_receipt_sha256, "evidence_receipt_sha256")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BaselineProbe":
        if not isinstance(raw, Mapping):
            raise TypeError(f"baseline probe entry must be a mapping (got {type(raw).__name__})")
        obs_sha = raw.get("observed_corpus_sha")
        if obs_sha is not None:
            _require_non_empty_str(obs_sha, "observed_corpus_sha")
        return cls(
            rule_id=_require_non_empty_str(raw["rule_id"], "rule_id"),
            passed_natively=_require_exact_bool(raw["passed_natively"], "passed_natively"),
            regression_detected=_require_exact_bool(
                raw.get("regression_detected", False), "regression_detected"
            ),
            peer_conflict=_require_exact_bool(raw.get("peer_conflict", False), "peer_conflict"),
            observed_corpus_sha=obs_sha,
            notes=str(raw.get("notes", "")),
            evidence_receipt_sha256=raw.get("evidence_receipt_sha256"),
        )


def verify_baseline_probe_evidence(
    ledger: Optional[AppendOnlyLedger],
    *,
    rule_id: str,
    probe: BaselineProbe,
    new_checkpoint_sha: str,
    expected_corpus_sha: Optional[str] = None,
    signing_key: Optional[bytes] = None,
    require_signatures: bool = False,
) -> LedgerEntry:
    """Verify that a BaselineProbe resolves to a matching bare-model ledger probe receipt on W_new."""
    if probe.rule_id != rule_id:
        raise SIMEvidenceError(
            f"baseline probe rule_id mismatch: expected {rule_id!r}, got {probe.rule_id!r}"
        )
    if probe.evidence_receipt_sha256 is None:
        raise SIMEvidenceError(
            f"baseline probe for rule {rule_id!r} requires evidence_receipt_sha256"
        )
    if ledger is None:
        raise SIMEvidenceError(
            f"baseline probe for rule {rule_id!r} requires a ledger to verify evidence"
        )
    try:
        entry = ledger.resolve_verified_entry(
            probe.evidence_receipt_sha256,
            expected_event_type=BASELINE_PROBE_RECEIPT_EVENT_TYPE,
            signing_key=signing_key,
            require_signatures=require_signatures,
        )
    except ValueError as exc:
        raise SIMEvidenceError(str(exc)) from exc

    payload = entry.payload
    if payload.get("rule_id") != rule_id:
        raise SIMEvidenceError(
            f"baseline probe receipt rule_id mismatch: expected {rule_id!r}, got {payload.get('rule_id')!r}"
        )
    if payload.get("checkpoint_sha") != new_checkpoint_sha:
        raise SIMEvidenceError("baseline probe receipt checkpoint_sha mismatch")
    if payload.get("bare_model") is not True:
        raise SIMEvidenceError("baseline probe receipt must record bare_model=True")
    if payload.get("passed_natively") is not probe.passed_natively:
        raise SIMEvidenceError("baseline probe receipt passed_natively mismatch")
    if payload.get("regression_detected") is not probe.regression_detected:
        raise SIMEvidenceError("baseline probe receipt regression_detected mismatch")
    if payload.get("peer_conflict") is not probe.peer_conflict:
        raise SIMEvidenceError("baseline probe receipt peer_conflict mismatch")
    rec_corpus = payload.get("observed_corpus_sha")
    if probe.observed_corpus_sha != rec_corpus:
        raise SIMEvidenceError("baseline probe receipt observed_corpus_sha mismatch")
    if expected_corpus_sha is not None and rec_corpus != expected_corpus_sha:
        raise SIMEvidenceError(
            f"baseline probe receipt observed_corpus_sha {rec_corpus!r} does not match expected {expected_corpus_sha!r}"
        )
    return entry


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
    automated resolution. Every non-pending attempt outcome and every supplied
    baseline probe must resolve to a chain-verified ledger receipt.
    """

    def plan_upgrade(
        self,
        sim: SIMMemory,
        *,
        old_checkpoint_sha: str,
        new_checkpoint_sha: str,
        active_corpus_sha: str,
        probes: Mapping[str, BaselineProbe] = {},
        ledger: Optional[AppendOnlyLedger] = None,
        signing_key: Optional[bytes] = None,
        require_signatures: bool = False,
    ) -> UpgradePlan:
        active_ledger = ledger if ledger is not None else sim.ledger
        effective_key = signing_key if signing_key is not None else sim.signing_key
        effective_req_sig = require_signatures or sim.require_signatures

        rules_sorted = sorted(sim.rules(), key=lambda r: r.rule_id)
        known_rule_ids = {rec.rule_id for rec in rules_sorted}
        if active_ledger is not None:
            _verify_ledger_rule_completeness(
                active_ledger,
                known_rule_ids,
                signing_key=effective_key,
                require_signatures=effective_req_sig,
            )
        verified_probes: dict[str, LedgerEntry] = {}
        for probe_rule_id, probe in sorted(probes.items()):
            if probe_rule_id not in known_rule_ids:
                raise SIMEvidenceError(f"baseline probe supplied for unknown rule_id {probe_rule_id!r}")
            verified_probes[probe_rule_id] = verify_baseline_probe_evidence(
                active_ledger,
                rule_id=probe_rule_id,
                probe=probe,
                new_checkpoint_sha=new_checkpoint_sha,
                expected_corpus_sha=active_corpus_sha,
                signing_key=effective_key,
                require_signatures=effective_req_sig,
            )

        actions: list[UpgradeAction] = []
        for rec in rules_sorted:
            for att in rec.attempts:
                verify_attempt_evidence(
                    active_ledger,
                    rule_id=rec.rule_id,
                    attempt=att,
                    signing_key=effective_key,
                    require_signatures=effective_req_sig,
                )
            if active_ledger is not None:
                _verify_rule_state_against_ledger(
                    active_ledger,
                    rec,
                    signing_key=effective_key,
                    require_signatures=effective_req_sig,
                )
            probe = probes.get(rec.rule_id)
            probe_receipt = verified_probes.get(rec.rule_id)
            action, reason = self._classify(rec, probe, active_corpus_sha, probe_receipt=probe_receipt)
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
        *,
        probe_receipt: Optional[LedgerEntry] = None,
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
                if probe_receipt is None:
                    raise SIMEvidenceError(
                        f"cannot emit retire_candidate for rule {rec.rule_id!r} without verified probe receipt"
                    )
                return "retire_candidate", "new_checkpoint_passes_natively_without_workaround"
            if probe is None:
                return "recompile", "workaround_unprobed_on_new_checkpoint_keep_pinned"
            return "recompile", "workaround_still_required_on_new_checkpoint"

        # rule_kind == "preference" (uncontested)
        if rec.corpus_sha != active_corpus_sha:
            return "escalate_human", "preference_corpus_sha_drifted_needs_human_confirmation"
        return "recompile", "authorized_preference_repin_to_new_checkpoint"
