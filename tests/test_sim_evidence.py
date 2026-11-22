"""Fail-closed SIM ledger evidence binding tests (Closes siliconcordat/exo#8).

Zero bare ``assert`` statements; runs cleanly under ``-B``, ``-O -B``, ``-B -W error``,
and isolated ``-I`` mode.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from exo.cli import main as cli_main
from exo.ledger import AppendOnlyLedger, LedgerEntry
from exo.sim import (
    BASELINE_PROBE_RECEIPT_EVENT_TYPE,
    EVALUATION_RECEIPT_EVENT_TYPE,
    BaselineProbe,
    CrossGenerationPlanner,
    SIMAttempt,
    SIMEvidenceError,
    SIMMemory,
    SIMRuleRecord,
    record_baseline_probe_receipt,
    record_evaluation_receipt,
    verify_attempt_evidence,
    verify_baseline_probe_evidence,
)

H_OLD = "a" * 64
H_NEW = "b" * 64
C_CUR = "c" * 64
C_ALT = "d" * 64
UNKNOWN_DIGEST = "f" * 64


def _make_rule(rule_id: str, rule_kind: str, *, corpus_sha: str = C_CUR) -> SIMRuleRecord:
    return SIMRuleRecord(
        rule_id=rule_id,
        capability_id=f"cap.{rule_id}",
        name=rule_id,
        problem=f"problem for {rule_id}",
        rule_kind=rule_kind,
        adapter_id="corpus",
        base_checkpoint_sha=H_OLD,
        corpus_sha=corpus_sha,
    )


def _make_attempt(
    round_id: int,
    outcome: str,
    *,
    base_checkpoint_sha: str = H_OLD,
    corpus_sha: str = C_CUR,
    evidence_receipt_sha256: str | None = None,
    predicted: str = "expected change",
) -> SIMAttempt:
    return SIMAttempt(
        round_id=round_id,
        what="test attempt",
        predicted=predicted,
        fixed=(),
        missed=(),
        base_checkpoint_sha=base_checkpoint_sha,
        corpus_sha=corpus_sha,
        outcome=outcome,
        evidence_receipt_sha256=evidence_receipt_sha256,
    )


class TestSIMLedgerEvidenceBinding(unittest.TestCase):
    def test_issue8_reproducer_fails_closed_for_attempts_and_probes(self) -> None:
        """Exact Issue #8 reproducer: caller-declared outcomes and unbacked probes raise."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_no_ledger = SIMMemory(Path(tmp) / "sim.jsonl")
            sim_no_ledger.register_rule(_make_rule("pref", "preference"))
            sim_no_ledger.register_rule(_make_rule("wk", "model_workaround"))

            # Pending is allowed without a ledger or receipt digest
            pending = sim_no_ledger.record_attempt("pref", _make_attempt(1, "pending"))
            self.assertEqual(pending.outcome, "pending")
            self.assertIsNone(pending.evidence_receipt_sha256)

            # Every non-pending outcome without a resolvable ledger receipt MUST raise
            for outcome in ("neutral", "improved", "regressed", "failed"):
                with self.assertRaises(SIMEvidenceError, msg=f"outcome={outcome} must raise"):
                    sim_no_ledger.record_attempt("pref", _make_attempt(2, outcome))

            # Planner MUST NEVER return retire_candidate for a probe with no receipt
            planner = CrossGenerationPlanner()
            for passed_flag in (True, False):
                with self.assertRaises(SIMEvidenceError):
                    planner.plan_upgrade(
                        sim_no_ledger,
                        old_checkpoint_sha=H_OLD,
                        new_checkpoint_sha=H_NEW,
                        active_corpus_sha=C_CUR,
                        probes={"wk": BaselineProbe("wk", passed_natively=passed_flag)},
                    )

            # Unprobed workaround with only pending attempts stays safely pinned (recompile)
            plan_unprobed = planner.plan_upgrade(
                sim_no_ledger,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
            )
            by_id = {a.rule_id: a for a in plan_unprobed.actions}
            self.assertEqual(by_id["wk"].action, "recompile")
            self.assertEqual(by_id["wk"].reason, "workaround_unprobed_on_new_checkpoint_keep_pinned")
            self.assertEqual(by_id["pref"].action, "recompile")
            self.assertEqual(by_id["pref"].reason, "authorized_preference_repin_to_new_checkpoint")
            self.assertEqual(len(plan_unprobed.by_action("recompile")), 2)
            self.assertEqual(plan_unprobed.by_action("retire_candidate"), [])

    def test_verified_attempt_and_probe_receipts_drive_planner_and_boundaries(self) -> None:
        """Verified receipts enable retire_candidate, 1-vs-2 regression threshold, and escalations."""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = AppendOnlyLedger(Path(tmp) / "ledger.jsonl")
            sim = SIMMemory(Path(tmp) / "sim.jsonl", ledger=ledger)
            sim.register_rule(_make_rule("pref.one", "preference"))
            sim.register_rule(_make_rule("pref.two", "preference"))
            sim.register_rule(_make_rule("wk.retire", "model_workaround"))
            sim.register_rule(_make_rule("wk.keep", "model_workaround"))
            sim.register_rule(_make_rule("wk.reg", "model_workaround"))
            sim.register_rule(_make_rule("wk.conflict", "model_workaround"))
            sim.register_rule(_make_rule("inv.drifted", "durable_invariant", corpus_sha=C_ALT))

            # 1 regressed attempt on preference stays below threshold (>= 2) -> recompile
            r1 = record_evaluation_receipt(
                ledger,
                rule_id="pref.one",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
                details={"score_delta": -1},
            )
            self.assertEqual(r1.event_type, EVALUATION_RECEIPT_EVENT_TYPE)
            self.assertEqual(r1.payload["details"], {"score_delta": -1})
            sim.record_attempt(
                "pref.one",
                _make_attempt(1, "regressed", evidence_receipt_sha256=r1.entry_sha256),
            )

            # 2 regressed attempts on preference cross >= 2 threshold -> escalate_human
            for rnd in (1, 2):
                rec_entry = record_evaluation_receipt(
                    ledger,
                    rule_id="pref.two",
                    round_id=rnd,
                    outcome="regressed",
                    base_checkpoint_sha=H_OLD,
                    corpus_sha=C_CUR,
                )
                sim.record_attempt(
                    "pref.two",
                    _make_attempt(rnd, "regressed", evidence_receipt_sha256=rec_entry.entry_sha256),
                )

            p_retire = record_baseline_probe_receipt(
                ledger,
                rule_id="wk.retire",
                checkpoint_sha=H_NEW,
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha=C_CUR,
                notes="bare W_new passed",
            )
            self.assertEqual(p_retire.event_type, BASELINE_PROBE_RECEIPT_EVENT_TYPE)
            p_keep = record_baseline_probe_receipt(
                ledger,
                rule_id="wk.keep",
                checkpoint_sha=H_NEW,
                passed_natively=False,
                bare_model=True,
                observed_corpus_sha=C_CUR,
            )
            p_reg = record_baseline_probe_receipt(
                ledger,
                rule_id="wk.reg",
                checkpoint_sha=H_NEW,
                passed_natively=False,
                bare_model=True,
                regression_detected=True,
                observed_corpus_sha=C_CUR,
            )
            p_conf = record_baseline_probe_receipt(
                ledger,
                rule_id="wk.conflict",
                checkpoint_sha=H_NEW,
                passed_natively=False,
                bare_model=True,
                peer_conflict=True,
                observed_corpus_sha=C_CUR,
            )

            plan = CrossGenerationPlanner().plan_upgrade(
                sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "wk.retire": BaselineProbe(
                        "wk.retire",
                        passed_natively=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=p_retire.entry_sha256,
                    ),
                    "wk.keep": BaselineProbe(
                        "wk.keep",
                        passed_natively=False,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=p_keep.entry_sha256,
                    ),
                    "wk.reg": BaselineProbe(
                        "wk.reg",
                        passed_natively=False,
                        regression_detected=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=p_reg.entry_sha256,
                    ),
                    "wk.conflict": BaselineProbe(
                        "wk.conflict",
                        passed_natively=False,
                        peer_conflict=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=p_conf.entry_sha256,
                    ),
                },
            )
            by_id = {a.rule_id: a for a in plan.actions}
            self.assertEqual(by_id["pref.one"].action, "recompile")
            self.assertEqual(by_id["pref.two"].action, "escalate_human")
            self.assertEqual(by_id["pref.two"].reason, "repeated_preference_regressions")
            self.assertEqual(by_id["wk.retire"].action, "retire_candidate")
            self.assertEqual(by_id["wk.keep"].action, "recompile")
            self.assertEqual(by_id["wk.keep"].reason, "workaround_still_required_on_new_checkpoint")
            self.assertEqual(by_id["wk.reg"].action, "escalate_human")
            self.assertEqual(by_id["wk.reg"].reason, "unexpected_regression_on_new_checkpoint")
            self.assertEqual(by_id["wk.conflict"].action, "escalate_human")
            self.assertEqual(by_id["wk.conflict"].reason, "peer_steward_conflict_on_new_checkpoint")
            self.assertEqual(by_id["inv.drifted"].action, "recompile")
            self.assertEqual(by_id["inv.drifted"].reason, "durable_invariant_corpus_updated")
            self.assertEqual(ledger.verify_chain(), [])

            # Mark contested persists and reloads cleanly
            sim.mark_contested("pref.one", "steward dispute")
            reloaded = SIMMemory(Path(tmp) / "sim.jsonl", ledger=ledger)
            self.assertTrue(reloaded.get("pref.one").contested)
            self.assertEqual(reloaded.get("pref.one").contested_reason, "steward dispute")

    def test_attempt_receipt_rejects_missing_forged_wrong_type_and_mismatched_fields(self) -> None:
        ledger = AppendOnlyLedger()
        sim = SIMMemory(ledger=ledger)
        sim.register_rule(_make_rule("r1", "preference"))
        sim.register_rule(_make_rule("r2", "preference"))

        # Wrong event type (sim_rule_registered, baseline_probe, AND lookalike payload with wrong event_type!)
        reg_digest = ledger.entries("sim_rule_registered")[0].entry_sha256
        probe_entry = record_baseline_probe_receipt(
            ledger, rule_id="r1", checkpoint_sha=H_NEW, passed_natively=True, bare_model=True
        )
        lookalike_eval = ledger.append(
            "unrelated_note",
            "spoof",
            {
                "rule_id": "r1",
                "round_id": 1,
                "outcome": "improved",
                "base_checkpoint_sha": H_OLD,
                "corpus_sha": C_CUR,
            },
        )
        with self.assertRaises(ValueError) as wrong_ev_ctx:
            ledger.resolve_verified_entry(lookalike_eval.entry_sha256, expected_event_type=EVALUATION_RECEIPT_EVENT_TYPE)
        self.assertIn("event_type", str(wrong_ev_ctx.exception))
        for bad_digest in (UNKNOWN_DIGEST, reg_digest, probe_entry.entry_sha256, lookalike_eval.entry_sha256):
            with self.assertRaises(SIMEvidenceError):
                sim.record_attempt("r1", _make_attempt(1, "improved", evidence_receipt_sha256=bad_digest))

        # Missing evidence_receipt_sha256 even when a real ledger is attached MUST cite evidence_receipt_sha256
        with self.assertRaises(SIMEvidenceError) as missing_att_ctx:
            sim.record_attempt("r1", _make_attempt(1, "improved", evidence_receipt_sha256=None))
        self.assertIn("requires evidence_receipt_sha256", str(missing_att_ctx.exception))

        # Bogus outcome rejected specifically as invalid attempt outcome (even if a forged ledger entry claims it)
        bogus_entry = ledger.append(
            EVALUATION_RECEIPT_EVENT_TYPE,
            "rogue",
            {
                "rule_id": "r1",
                "round_id": 1,
                "outcome": "bogus",
                "base_checkpoint_sha": H_OLD,
                "corpus_sha": C_CUR,
            },
        )
        with self.assertRaises(ValueError) as bogus_ctx:
            sim.record_attempt("r1", _make_attempt(1, "bogus", evidence_receipt_sha256=bogus_entry.entry_sha256))
        self.assertNotIsInstance(bogus_ctx.exception, SIMEvidenceError)
        self.assertIn("invalid attempt outcome", str(bogus_ctx.exception))

        # Valid receipt for (r1, round_id=1, outcome="regressed", H_OLD, C_CUR)
        eval_entry = record_evaluation_receipt(
            ledger,
            rule_id="r1",
            round_id=1,
            outcome="regressed",
            base_checkpoint_sha=H_OLD,
            corpus_sha=C_CUR,
        )

        # Replay on wrong rule_id
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt("r2", _make_attempt(1, "regressed", evidence_receipt_sha256=eval_entry.entry_sha256))

        # Replay with mismatched round_id
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt("r1", _make_attempt(2, "regressed", evidence_receipt_sha256=eval_entry.entry_sha256))

        # Replay downgrading outcome ("regressed" -> "neutral")
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt("r1", _make_attempt(1, "neutral", evidence_receipt_sha256=eval_entry.entry_sha256))

        # Replay with mismatched base_checkpoint_sha
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt(
                "r1",
                _make_attempt(1, "regressed", base_checkpoint_sha=H_NEW, evidence_receipt_sha256=eval_entry.entry_sha256),
            )

        # Replay with mismatched corpus_sha
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt(
                "r1",
                _make_attempt(1, "regressed", corpus_sha=C_ALT, evidence_receipt_sha256=eval_entry.entry_sha256),
            )

        # Receipt with bool round_id injected directly into ledger payload MUST fail
        bad_round_entry = ledger.append(
            EVALUATION_RECEIPT_EVENT_TYPE,
            "rogue",
            {
                "rule_id": "r1",
                "round_id": True,
                "outcome": "improved",
                "base_checkpoint_sha": H_OLD,
                "corpus_sha": C_CUR,
            },
        )
        with self.assertRaises(SIMEvidenceError):
            sim.record_attempt("r1", _make_attempt(1, "improved", evidence_receipt_sha256=bad_round_entry.entry_sha256))

        # Valid matching attempt succeeds; ledger records evidence_receipt_sha256
        saved = sim.record_attempt("r1", _make_attempt(1, "regressed", evidence_receipt_sha256=eval_entry.entry_sha256))
        self.assertEqual(saved.evidence_receipt_sha256, eval_entry.entry_sha256)
        recorded_events = ledger.entries("sim_attempt_recorded")
        self.assertEqual(recorded_events[-1].payload["evidence_receipt_sha256"], eval_entry.entry_sha256)

        # Smuggling an unbacked non-pending attempt inside register_rule MUST fail
        smuggled = SIMRuleRecord(
            rule_id="r3",
            capability_id="cap.r3",
            name="r3",
            problem="p",
            rule_kind="preference",
            adapter_id="corpus",
            base_checkpoint_sha=H_OLD,
            corpus_sha=C_CUR,
            attempts=[_make_attempt(1, "improved")],
        )
        with self.assertRaises(SIMEvidenceError):
            sim.register_rule(smuggled)

        # Registering a rule with a verified attempt succeeds (and reloads with ledger)
        ledger_s2 = AppendOnlyLedger()
        ev_r3 = record_evaluation_receipt(
            ledger_s2, rule_id="r3", round_id=1, outcome="improved", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR
        )
        valid_with_att = SIMRuleRecord(
            rule_id="r3",
            capability_id="cap.r3",
            name="r3",
            problem="p",
            rule_kind="preference",
            adapter_id="corpus",
            base_checkpoint_sha=H_OLD,
            corpus_sha=C_CUR,
            attempts=[_make_attempt(1, "improved", evidence_receipt_sha256=ev_r3.entry_sha256)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "sim.jsonl"
            s2 = SIMMemory(sp, ledger=ledger_s2)
            s2.register_rule(valid_with_att)
            s2_reloaded = SIMMemory(sp, ledger=ledger_s2)
            self.assertEqual(len(s2_reloaded.get("r3").attempts), 1)

            # Tampering with register_rule's embedded attempt on disk fails reload
            txt = sp.read_text(encoding="utf-8").replace('"improved"', '"neutral"')
            sp.write_text(txt, encoding="utf-8")
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp, ledger=ledger_s2)

    def test_baseline_probe_rejects_forged_adapter_assisted_and_mismatched_fields(self) -> None:
        ledger = AppendOnlyLedger()
        sim = SIMMemory(ledger=ledger)
        sim.register_rule(_make_rule("wk1", "model_workaround"))
        sim.register_rule(_make_rule("wk2", "model_workaround"))
        planner = CrossGenerationPlanner()

        # Non-bare-model probe (adapter still mounted) MUST fail even if passed_natively=True
        assisted = record_baseline_probe_receipt(
            ledger,
            rule_id="wk1",
            checkpoint_sha=H_NEW,
            passed_natively=True,
            bare_model=False,
            observed_corpus_sha=C_CUR,
        )
        with self.assertRaises(SIMEvidenceError):
            planner.plan_upgrade(
                sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "wk1": BaselineProbe(
                        "wk1",
                        passed_natively=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=assisted.entry_sha256,
                    )
                },
            )

        # Receipt with bare_model=1 (int, not bool) in raw ledger payload MUST fail
        non_bool_bare = ledger.append(
            BASELINE_PROBE_RECEIPT_EVENT_TYPE,
            "rogue",
            {
                "rule_id": "wk1",
                "checkpoint_sha": H_NEW,
                "bare_model": 1,
                "passed_natively": True,
                "regression_detected": False,
                "peer_conflict": False,
                "observed_corpus_sha": C_CUR,
            },
        )
        with self.assertRaises(SIMEvidenceError):
            planner.plan_upgrade(
                sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "wk1": BaselineProbe(
                        "wk1",
                        passed_natively=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=non_bool_bare.entry_sha256,
                    )
                },
            )

        # Valid bare-model receipt for wk1 on H_NEW with observed_corpus_sha=C_CUR
        valid = record_baseline_probe_receipt(
            ledger,
            rule_id="wk1",
            checkpoint_sha=H_NEW,
            passed_natively=True,
            bare_model=True,
            regression_detected=False,
            peer_conflict=False,
            observed_corpus_sha=C_CUR,
        )
        wrong_event = record_evaluation_receipt(
            ledger, rule_id="wk1", round_id=1, outcome="improved", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR
        )
        sim.record_attempt("wk1", _make_attempt(1, "improved", evidence_receipt_sha256=wrong_event.entry_sha256))

        # Wrong event type, unknown digest, dict-key vs probe.rule_id mismatch, cross-rule replay
        bad_probes: list[tuple[str, dict[str, BaselineProbe], str, str]] = [
            (
                "wrong_event",
                {"wk1": BaselineProbe("wk1", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=wrong_event.entry_sha256)},
                H_NEW,
                "event_type",
            ),
            (
                "unknown_digest",
                {"wk1": BaselineProbe("wk1", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=UNKNOWN_DIGEST)},
                H_NEW,
                "ledger receipt digest not found",
            ),
            (
                "probe_key_mismatch",
                {"wk1": BaselineProbe("wk2", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=valid.entry_sha256)},
                H_NEW,
                "baseline probe rule_id mismatch",
            ),
            (
                "cross_rule_receipt",
                {"wk2": BaselineProbe("wk2", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=valid.entry_sha256)},
                H_NEW,
                "baseline probe receipt rule_id mismatch",
            ),
            (
                "wrong_checkpoint",
                {"wk1": BaselineProbe("wk1", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=valid.entry_sha256)},
                H_OLD,
                "checkpoint_sha mismatch",
            ),
            (
                "passed_flip",
                {"wk1": BaselineProbe("wk1", False, observed_corpus_sha=C_CUR, evidence_receipt_sha256=valid.entry_sha256)},
                H_NEW,
                "passed_natively mismatch",
            ),
            (
                "regression_flip",
                {
                    "wk1": BaselineProbe(
                        "wk1",
                        True,
                        regression_detected=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=valid.entry_sha256,
                    )
                },
                H_NEW,
                "regression_detected mismatch",
            ),
            (
                "conflict_flip",
                {
                    "wk1": BaselineProbe(
                        "wk1",
                        True,
                        peer_conflict=True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=valid.entry_sha256,
                    )
                },
                H_NEW,
                "peer_conflict mismatch",
            ),
            (
                "corpus_mismatch",
                {
                    "wk1": BaselineProbe(
                        "wk1", True, observed_corpus_sha=C_ALT, evidence_receipt_sha256=valid.entry_sha256
                    )
                },
                H_NEW,
                "observed_corpus_sha mismatch",
            ),
            (
                "corpus_none_wildcard_blocked",
                {
                    "wk1": BaselineProbe(
                        "wk1", True, observed_corpus_sha=None, evidence_receipt_sha256=valid.entry_sha256
                    )
                },
                H_NEW,
                "observed_corpus_sha mismatch",
            ),
        ]
        for label, probe_map, target_ckpt, expected_msg in bad_probes:
            with self.assertRaises(SIMEvidenceError, msg=f"case {label} must raise") as bp_ctx:
                planner.plan_upgrade(
                    sim,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=target_ckpt,
                    active_corpus_sha=C_CUR,
                    probes=probe_map,
                )
            self.assertIn(expected_msg, str(bp_ctx.exception))

        # Missing evidence_receipt_sha256 on BaselineProbe even with a real ledger MUST cite evidence_receipt_sha256
        with self.assertRaises(SIMEvidenceError) as missing_probe_ctx:
            planner.plan_upgrade(
                sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={"wk1": BaselineProbe("wk1", True, evidence_receipt_sha256=None)},
            )
        self.assertIn("requires evidence_receipt_sha256", str(missing_probe_ctx.exception))

        # Lookalike probe payload stored under a wrong ledger event_type MUST be rejected
        lookalike_probe = ledger.append(
            "unrelated_probe_note",
            "spoof",
            {
                "rule_id": "wk1",
                "checkpoint_sha": H_NEW,
                "bare_model": True,
                "passed_natively": True,
                "regression_detected": False,
                "peer_conflict": False,
                "observed_corpus_sha": C_CUR,
            },
        )
        with self.assertRaises(SIMEvidenceError):
            planner.plan_upgrade(
                sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "wk1": BaselineProbe(
                        "wk1", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=lookalike_probe.entry_sha256
                    )
                },
            )

        # No ledger provided with a probe digest MUST fail
        with self.assertRaises(SIMEvidenceError):
            verify_baseline_probe_evidence(
                None,
                rule_id="wk1",
                probe=BaselineProbe("wk1", True, evidence_receipt_sha256=valid.entry_sha256),
                new_checkpoint_sha=H_NEW,
            )

        # Defense-in-depth: direct _classify without verified probe_receipt cannot emit retire_candidate
        with self.assertRaises(SIMEvidenceError):
            CrossGenerationPlanner._classify(
                sim.get("wk1"),
                BaselineProbe("wk1", passed_natively=True),
                C_CUR,
                probe_receipt=None,
            )

    def test_disk_tampering_signatures_and_deleted_ledger_fail_closed(self) -> None:
        key = b"sim-evidence-hmac-key-32-bytes!!"
        wrong_key = b"wrong-evidence-hmac-key-32bytes!"
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            sim_path = Path(tmp) / "sim.jsonl"
            ledger = AppendOnlyLedger(ledger_path)
            sim = SIMMemory(sim_path, ledger=ledger, signing_key=key, require_signatures=True)
            sim.register_rule(_make_rule("pref.signed", "preference"))
            sim.register_rule(_make_rule("wk.signed", "model_workaround"))

            # Unsigned receipt MUST be rejected when require_signatures=True
            unsigned_eval = record_evaluation_receipt(
                ledger,
                rule_id="pref.signed",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
                signing_key=None,
            )
            with self.assertRaises(SIMEvidenceError):
                sim.record_attempt(
                    "pref.signed",
                    _make_attempt(1, "regressed", evidence_receipt_sha256=unsigned_eval.entry_sha256),
                )

            # Build a clean signed ledger from scratch
            signed_ledger_path = Path(tmp) / "signed-ledger.jsonl"
            signed_sim_path = Path(tmp) / "signed-sim.jsonl"
            s_ledger = AppendOnlyLedger(signed_ledger_path)
            s_sim = SIMMemory(signed_sim_path, ledger=s_ledger, signing_key=key, require_signatures=True)
            s_sim.register_rule(_make_rule("pref.signed", "preference"))
            s_sim.register_rule(_make_rule("wk.signed", "model_workaround"))
            ev = record_evaluation_receipt(
                s_ledger,
                rule_id="pref.signed",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
                signing_key=key,
            )
            s_sim.record_attempt("pref.signed", _make_attempt(1, "regressed", evidence_receipt_sha256=ev.entry_sha256))
            pr = record_baseline_probe_receipt(
                s_ledger,
                rule_id="wk.signed",
                checkpoint_sha=H_NEW,
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha=C_CUR,
                signing_key=key,
            )
            planner = CrossGenerationPlanner()
            ok_plan = planner.plan_upgrade(
                s_sim,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "wk.signed": BaselineProbe(
                        "wk.signed",
                        True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=pr.entry_sha256,
                    )
                },
            )
            self.assertEqual({a.rule_id: a.action for a in ok_plan.actions}["wk.signed"], "retire_candidate")

            # Wrong signing_key MUST fail
            with self.assertRaises(SIMEvidenceError):
                planner.plan_upgrade(
                    s_sim,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    signing_key=wrong_key,
                    require_signatures=True,
                )

            # Replaying sim.jsonl without a ledger cannot bypass attempt evidence check in plan_upgrade
            reloaded_no_ledger = SIMMemory(signed_sim_path)
            with self.assertRaises(SIMEvidenceError):
                planner.plan_upgrade(
                    reloaded_no_ledger,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                )

            # Tampering sim.jsonl outcome on disk ("regressed" -> "neutral") fails BOTH reload & plan_upgrade
            tampered_sim_path = Path(tmp) / "tampered-sim.jsonl"
            tampered_sim_path.write_text(
                signed_sim_path.read_text(encoding="utf-8").replace('"regressed"', '"neutral"'),
                encoding="utf-8",
            )
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(tampered_sim_path, ledger=s_ledger, signing_key=key, require_signatures=True)
            tampered_mem = SIMMemory(tampered_sim_path)
            with self.assertRaises(SIMEvidenceError):
                planner.plan_upgrade(
                    tampered_mem,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=s_ledger,
                    signing_key=key,
                    require_signatures=True,
                )

            # Tampering ledger.jsonl on disk (1-byte payload mutation) fails verification
            lines = signed_ledger_path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines):
                obj = json.loads(line)
                if obj["event_type"] == EVALUATION_RECEIPT_EVENT_TYPE:
                    obj["payload"]["outcome"] = "neutral"
                    lines[idx] = json.dumps(obj, sort_keys=True, separators=(",", ":"))
                    break
            signed_ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError):
                verify_attempt_evidence(
                    s_ledger,
                    rule_id="pref.signed",
                    attempt=_make_attempt(1, "regressed", evidence_receipt_sha256=ev.entry_sha256),
                    signing_key=key,
                    require_signatures=True,
                )

            # Deleted ledger file on disk MUST fail even though in-memory cache held entries
            signed_ledger_path.unlink()
            with self.assertRaises(ValueError):
                s_ledger.resolve_verified_entry(ev.entry_sha256)

            # Ledger seq=True (bool) rejected by LedgerEntry.from_dict
            with self.assertRaises(ValueError):
                LedgerEntry.from_dict(
                    {
                        "seq": True,
                        "timestamp_utc": "2026-11-22T00:00:00Z",
                        "event_type": "note",
                        "actor_id": "a",
                        "payload": {},
                        "prev_sha256": "0" * 64,
                        "entry_sha256": "1" * 64,
                    }
                )
            with self.assertRaises(ValueError):
                LedgerEntry.from_dict(
                    {
                        "seq": 0,
                        "timestamp_utc": "2026-11-22T00:00:00Z",
                        "event_type": "note",
                        "actor_id": "a",
                        "payload": {},
                        "prev_sha256": "0" * 64,
                        "entry_sha256": "1" * 64,
                    }
                )

    def test_strict_type_validation_and_malformed_inputs_rejected(self) -> None:
        ledger = AppendOnlyLedger()
        sim = SIMMemory(ledger=ledger)
        sim.register_rule(_make_rule("r1", "preference"))

        # bool-as-int and non-positive round_id rejected on SIMAttempt and from_dict
        for bad_round in (True, False, 1.0, "1"):
            with self.assertRaises(TypeError):
                _make_attempt(bad_round, "pending")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                SIMAttempt.from_dict(
                    {
                        "round_id": bad_round,
                        "what": "w",
                        "predicted": "p",
                        "base_checkpoint_sha": H_OLD,
                        "corpus_sha": C_CUR,
                    }
                )
        for bad_int_round in (0, -1):
            with self.assertRaises(ValueError):
                _make_attempt(bad_int_round, "pending")

        # Malformed evidence_receipt_sha256 rejected on SIMAttempt, BaselineProbe, resolve_verified_entry
        for bad_sha in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 123, True):
            with self.assertRaises(SIMEvidenceError):
                _make_attempt(1, "pending", evidence_receipt_sha256=bad_sha)  # type: ignore[arg-type]
            with self.assertRaises(SIMEvidenceError):
                BaselineProbe("r1", True, evidence_receipt_sha256=bad_sha)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                ledger.resolve_verified_entry(bad_sha)  # type: ignore[arg-type]

        # Non-bool flags rejected on BaselineProbe & BaselineProbe.from_dict
        for bad_bool in (1, 0, "true", "false", None):
            with self.assertRaises(TypeError):
                BaselineProbe("r1", bad_bool)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                BaselineProbe("r1", True, regression_detected=bad_bool)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                BaselineProbe("r1", True, peer_conflict=bad_bool)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                BaselineProbe.from_dict({"rule_id": "r1", "passed_natively": bad_bool})
        with self.assertRaises(TypeError):
            BaselineProbe.from_dict("not-a-mapping")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BaselineProbe.from_dict({"rule_id": " ", "passed_natively": True})
        with self.assertRaises(ValueError):
            BaselineProbe.from_dict({"rule_id": "r1", "passed_natively": True, "observed_corpus_sha": ""})

        # record_evaluation_receipt validation
        with self.assertRaises(ValueError):
            record_evaluation_receipt(
                ledger, rule_id="r1", round_id=1, outcome="pending", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR
            )
        with self.assertRaises(ValueError):
            record_evaluation_receipt(
                ledger, rule_id="r1", round_id=1, outcome="bogus", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR
            )
        with self.assertRaises(ValueError):
            record_evaluation_receipt(
                ledger, rule_id="", round_id=1, outcome="improved", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR
            )
        with self.assertRaises(ValueError):
            record_evaluation_receipt(
                ledger, rule_id="r1", round_id=1, outcome="improved", base_checkpoint_sha=" ", corpus_sha=C_CUR
            )
        with self.assertRaises(ValueError):
            record_evaluation_receipt(
                ledger, rule_id="r1", round_id=1, outcome="improved", base_checkpoint_sha=H_OLD, corpus_sha=""
            )
        with self.assertRaises(TypeError):
            record_evaluation_receipt(
                ledger, rule_id="r1", round_id=True, outcome="improved", base_checkpoint_sha=H_OLD, corpus_sha=C_CUR  # type: ignore[arg-type]
            )

        # record_baseline_probe_receipt validation
        with self.assertRaises(ValueError):
            record_baseline_probe_receipt(ledger, rule_id="", checkpoint_sha=H_NEW, passed_natively=True)
        with self.assertRaises(ValueError):
            record_baseline_probe_receipt(ledger, rule_id="r1", checkpoint_sha="", passed_natively=True)
        with self.assertRaises(TypeError):
            record_baseline_probe_receipt(ledger, rule_id="r1", checkpoint_sha=H_NEW, passed_natively=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            record_baseline_probe_receipt(ledger, rule_id="r1", checkpoint_sha=H_NEW, passed_natively=True, bare_model=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            record_baseline_probe_receipt(
                ledger, rule_id="r1", checkpoint_sha=H_NEW, passed_natively=True, observed_corpus_sha="  "
            )

        # SIMMemory error guards
        with self.assertRaises(ValueError):
            sim.register_rule(_make_rule("badkind", "unknown_kind"))
        with self.assertRaises(ValueError):
            sim.register_rule(
                SIMRuleRecord("r_empty_cap", "  ", "n", "p", "preference", "corpus", H_OLD, C_CUR)
            )
        with self.assertRaises(KeyError) as rec_missing_ctx:
            sim.record_attempt("missing", _make_attempt(1, "pending"))
        self.assertIn("unknown rule_id", str(rec_missing_ctx.exception))
        with self.assertRaises(ValueError):
            sim.record_attempt("r1", _make_attempt(1, "pending", predicted="   "))
        with self.assertRaises(ValueError) as bad_out_ctx:
            sim.record_attempt("r1", _make_attempt(1, "bogus"))
        self.assertIn("invalid attempt outcome", str(bad_out_ctx.exception))
        with self.assertRaises(KeyError) as contest_missing_ctx:
            sim.mark_contested("missing", "why")
        self.assertIn("unknown rule_id", str(contest_missing_ctx.exception))
        with self.assertRaises(KeyError) as get_missing_ctx:
            sim.get("missing")
        self.assertIn("unknown rule_id", str(get_missing_ctx.exception))
        with self.assertRaises(RuntimeError):
            sim._replay_disk()

    def test_cli_plan_upgrade_rejects_unbacked_non_bool_and_non_list_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.jsonl"
            ledger_path = Path(tmp) / "ledger.jsonl"
            probes_path = Path(tmp) / "probes.json"
            ledger = AppendOnlyLedger(ledger_path)
            sim = SIMMemory(sim_path, ledger=ledger)
            sim.register_rule(_make_rule("wk.cli", "model_workaround"))

            # Non-list probes JSON (including empty dict {}) MUST be rejected
            probes_path.write_text(json.dumps({}), encoding="utf-8")
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                rc1 = cli_main([
                    "plan-upgrade",
                    str(sim_path),
                    "--old-sha",
                    H_OLD,
                    "--new-sha",
                    H_NEW,
                    "--corpus-sha",
                    C_CUR,
                    "--probes",
                    str(probes_path),
                    "--ledger",
                    str(ledger_path),
                ])
            self.assertEqual(rc1, 1)
            out1 = json.loads(buf1.getvalue())
            self.assertFalse(out1["ok"])
            self.assertIn("JSON array", out1["error"])

            # String "false" for passed_natively MUST be rejected (not bool("false") == True!)
            pr = record_baseline_probe_receipt(
                ledger,
                rule_id="wk.cli",
                checkpoint_sha=H_NEW,
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha=C_CUR,
            )
            probes_path.write_text(
                json.dumps([
                    {
                        "rule_id": "wk.cli",
                        "passed_natively": "false",
                        "observed_corpus_sha": C_CUR,
                        "evidence_receipt_sha256": pr.entry_sha256,
                    }
                ]),
                encoding="utf-8",
            )
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = cli_main([
                    "plan-upgrade",
                    str(sim_path),
                    "--old-sha",
                    H_OLD,
                    "--new-sha",
                    H_NEW,
                    "--corpus-sha",
                    C_CUR,
                    "--probes",
                    str(probes_path),
                    "--ledger",
                    str(ledger_path),
                ])
            self.assertEqual(rc2, 1)
            self.assertFalse(json.loads(buf2.getvalue())["ok"])

            # Valid receipt passes cleanly through CLI
            probes_path.write_text(
                json.dumps([
                    {
                        "rule_id": "wk.cli",
                        "passed_natively": True,
                        "observed_corpus_sha": C_CUR,
                        "evidence_receipt_sha256": pr.entry_sha256,
                    }
                ]),
                encoding="utf-8",
            )
            buf3 = io.StringIO()
            with redirect_stdout(buf3):
                rc3 = cli_main([
                    "plan-upgrade",
                    str(sim_path),
                    "--old-sha",
                    H_OLD,
                    "--new-sha",
                    H_NEW,
                    "--corpus-sha",
                    C_CUR,
                    "--probes",
                    str(probes_path),
                    "--ledger",
                    str(ledger_path),
                ])
            self.assertEqual(rc3, 0)
            out3 = json.loads(buf3.getvalue())
            self.assertEqual(out3["actions"][0]["action"], "retire_candidate")

    def test_unknown_sim_journal_op_and_blank_lines_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "sim.jsonl"
            sp.write_text("\n{\"op\":\"bogus_op\"}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                SIMMemory(sp)

    def test_r2_exo8_bug01_to_bug05_and_rule_kind_cross_check(self) -> None:
        """Closes all 5 dual-family Zone-B defects (EXO8-BUG-01..05) + rule_kind ledger cross-check."""
        planner = CrossGenerationPlanner()

        # EXO8-BUG-01: wrong-corpus, None-corpus wildcard, and missing-corpus probes MUST fail closed
        ledger1 = AppendOnlyLedger()
        sim1 = SIMMemory(ledger=ledger1)
        sim1.register_rule(_make_rule("r_patch", "model_workaround", corpus_sha=C_CUR))
        sim1.register_rule(_make_rule("r_inv", "durable_invariant", corpus_sha=C_CUR))

        wrong_corpus_receipt = record_baseline_probe_receipt(
            ledger1,
            rule_id="r_patch",
            checkpoint_sha=H_NEW,
            passed_natively=True,
            bare_model=True,
            observed_corpus_sha=C_ALT,
        )
        missing_corpus_receipt = record_baseline_probe_receipt(
            ledger1,
            rule_id="r_patch",
            checkpoint_sha=H_NEW,
            passed_natively=True,
            bare_model=True,
            observed_corpus_sha=None,
        )
        # (a) Direct verify_baseline_probe_evidence forbids observed_corpus_sha=None wildcard against non-null receipt
        with self.assertRaises(SIMEvidenceError) as wild_ctx:
            verify_baseline_probe_evidence(
                ledger1,
                rule_id="r_patch",
                probe=BaselineProbe("r_patch", True, observed_corpus_sha=None, evidence_receipt_sha256=wrong_corpus_receipt.entry_sha256),
                new_checkpoint_sha=H_NEW,
            )
        self.assertIn("observed_corpus_sha mismatch", str(wild_ctx.exception))
        # (b) plan_upgrade rejects wrong-corpus probe matching its own receipt (C_ALT != active C_CUR)
        with self.assertRaises(SIMEvidenceError) as wrong_c_ctx:
            planner.plan_upgrade(
                sim1,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "r_patch": BaselineProbe(
                        "r_patch",
                        True,
                        observed_corpus_sha=C_ALT,
                        evidence_receipt_sha256=wrong_corpus_receipt.entry_sha256,
                    )
                },
            )
        self.assertIn("does not match expected", str(wrong_c_ctx.exception))
        # (c) plan_upgrade rejects missing-corpus probe (None != active C_CUR)
        with self.assertRaises(SIMEvidenceError):
            planner.plan_upgrade(
                sim1,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "r_patch": BaselineProbe(
                        "r_patch",
                        True,
                        observed_corpus_sha=None,
                        evidence_receipt_sha256=missing_corpus_receipt.entry_sha256,
                    )
                },
            )

        # EXO8-BUG-05: forged probe on durable_invariant OR unknown rule_id MUST fail up front
        with self.assertRaises(SIMEvidenceError):
            planner.plan_upgrade(
                sim1,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={"r_inv": BaselineProbe("r_inv", True, observed_corpus_sha=C_CUR, evidence_receipt_sha256=UNKNOWN_DIGEST)},
            )
        valid_patch_receipt = record_baseline_probe_receipt(
            ledger1,
            rule_id="r_patch",
            checkpoint_sha=H_NEW,
            passed_natively=True,
            bare_model=True,
            observed_corpus_sha=C_CUR,
        )
        with self.assertRaises(SIMEvidenceError) as unk_probe_ctx:
            planner.plan_upgrade(
                sim1,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
                probes={
                    "ghost_rule": BaselineProbe(
                        "ghost_rule",
                        True,
                        observed_corpus_sha=C_CUR,
                        evidence_receipt_sha256=valid_patch_receipt.entry_sha256,
                    )
                },
            )
        self.assertIn("unknown rule_id", str(unk_probe_ctx.exception))

        # EXO8-BUG-02: regressed -> pending downgrade MUST fail closed (both with & without digest)
        with tempfile.TemporaryDirectory() as tmp:
            lp2 = Path(tmp) / "ledger2.jsonl"
            sp2 = Path(tmp) / "sim2.jsonl"
            ledger2 = AppendOnlyLedger(lp2)
            # Brand-new empty file-backed ledger accepts a genuine pending attempt
            sim_empty = SIMMemory(Path(tmp) / "sim_empty.jsonl", ledger=ledger2)
            sim_empty.register_rule(_make_rule("r_pending_ok", "preference"))
            sim_empty.record_attempt("r_pending_ok", _make_attempt(1, "pending"))

            # Pending attempt carrying a valid 64-hex digest is rejected immediately at construction
            with self.assertRaises(SIMEvidenceError) as pend_sha_ctx:
                _make_attempt(1, "pending", evidence_receipt_sha256="1" * 64)
            self.assertIn("pending attempt must not carry evidence_receipt_sha256", str(pend_sha_ctx.exception))

            sim2 = SIMMemory(sp2, ledger=ledger2)
            sim2.register_rule(_make_rule("r_deg", "preference"))
            ev_deg = record_evaluation_receipt(
                ledger2,
                rule_id="r_deg",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
            )
            sim2.record_attempt("r_deg", _make_attempt(1, "regressed", evidence_receipt_sha256=ev_deg.entry_sha256))

            # (a) Tamper disk: "regressed" -> "pending" keeping evidence_receipt_sha256
            sp2_keep_sha = Path(tmp) / "sim2_keep_sha.jsonl"
            sp2_keep_sha.write_text(
                sp2.read_text(encoding="utf-8").replace('"regressed"', '"pending"'),
                encoding="utf-8",
            )
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp2_keep_sha, ledger=ledger2)

            # (b) Tamper disk: "regressed" -> "pending" AND strip evidence_receipt_sha256 -> null
            sp2_strip = Path(tmp) / "sim2_strip.jsonl"
            stripped_lines = []
            for ln in sp2.read_text(encoding="utf-8").splitlines():
                obj = json.loads(ln)
                if obj.get("op") == "append_attempt":
                    obj["attempt"]["outcome"] = "pending"
                    obj["attempt"]["evidence_receipt_sha256"] = None
                stripped_lines.append(json.dumps(obj, sort_keys=True, separators=(",", ":")))
            sp2_strip.write_text("\n".join(stripped_lines) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as conflict_ctx:
                SIMMemory(sp2_strip, ledger=ledger2)
            self.assertIn("conflicts with resolved evaluation receipt", str(conflict_ctx.exception))
            # Even loaded without ledger, plan_upgrade with ledger rejects the downgraded pending attempt
            mem_strip = SIMMemory(sp2_strip)
            with self.assertRaises(SIMEvidenceError):
                planner.plan_upgrade(
                    mem_strip,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger2,
                )
            # Direct record_attempt("r_deg", pending round 1) against resolved ledger receipt also fails
            with self.assertRaises(SIMEvidenceError):
                sim2.record_attempt("r_deg", _make_attempt(1, "pending"))

            # Item 6 bonus: tampering rule_kind on disk ("durable_invariant" -> "model_workaround") fails closed
            lp_rk = Path(tmp) / "ledger_rk.jsonl"
            ledger_rk = AppendOnlyLedger(lp_rk)
            sp_rk = Path(tmp) / "sim_rk.jsonl"
            sim_rk = SIMMemory(sp_rk, ledger=ledger_rk)
            sim_rk.register_rule(_make_rule("r_durable", "durable_invariant"))
            sp_rk_bad = Path(tmp) / "sim_rk_bad.jsonl"
            sp_rk_bad.write_text(
                sp_rk.read_text(encoding="utf-8").replace('"durable_invariant"', '"model_workaround"'),
                encoding="utf-8",
            )
            with self.assertRaises(SIMEvidenceError) as rk_ctx:
                SIMMemory(sp_rk_bad, ledger=ledger_rk)
            self.assertIn("rule_kind mismatch", str(rk_ctx.exception))
            mem_rk_bad = SIMMemory(sp_rk_bad)
            with self.assertRaises(SIMEvidenceError):
                planner.plan_upgrade(
                    mem_rk_bad,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger_rk,
                )

            # Deleting populated ledger2.jsonl causes verified_entries / pending check to fail closed
            lp2.unlink()
            with self.assertRaises(SIMEvidenceError):
                verify_attempt_evidence(ledger2, rule_id="r_deg", attempt=_make_attempt(99, "pending"))

        # EXO8-BUG-03: malformed & non-dict sim.jsonl records fail cleanly with SIMEvidenceError (never AttributeError)
        with tempfile.TemporaryDirectory() as tmp:
            bad_journals: list[tuple[str, bytes]] = [
                ("non_dict_list", b"[]\n"),
                ("bad_json", b"{not json\n"),
                ("bad_utf8", b"\xff\xfe\n"),
                ("reg_non_dict", b'{"op":"register_rule","record":[]}\n'),
                ("reg_missing_fields", b'{"op":"register_rule","record":{}}\n'),
                (
                    "reg_bad_kind",
                    (
                        json.dumps(
                            {
                                "op": "register_rule",
                                "record": {
                                    "rule_id": "x",
                                    "capability_id": "c",
                                    "name": "n",
                                    "problem": "p",
                                    "rule_kind": "bogus_kind",
                                    "adapter_id": "corpus",
                                    "base_checkpoint_sha": H_OLD,
                                    "corpus_sha": C_CUR,
                                },
                            }
                        )
                        + "\n"
                    ).encode("utf-8"),
                ),
                (
                    "att_unknown_rule",
                    (
                        json.dumps({"op": "append_attempt", "rule_id": "missing", "attempt": _make_attempt(1, "pending").to_dict()})
                        + "\n"
                    ).encode("utf-8"),
                ),
                (
                    "att_bad_body",
                    (
                        json.dumps({"op": "register_rule", "record": _make_rule("r1", "preference").to_dict()})
                        + "\n"
                        + '{"op":"append_attempt","rule_id":"r1","attempt":{"round_id":"bad"}}\n'
                    ).encode("utf-8"),
                ),
                (
                    "contested_non_bool",
                    (
                        json.dumps({"op": "register_rule", "record": _make_rule("r1", "preference").to_dict()})
                        + "\n"
                        + '{"op":"mark_contested","rule_id":"r1","contested":"yes"}\n'
                    ).encode("utf-8"),
                ),
            ]
            for label, payload_bytes in bad_journals:
                jp = Path(tmp) / f"bad_{label}.jsonl"
                jp.write_bytes(payload_bytes)
                with self.assertRaises(SIMEvidenceError, msg=f"bad journal {label} must raise SIMEvidenceError"):
                    SIMMemory(jp)

            # CLI plan-upgrade on non-dict sim.jsonl ("[]\n") returns clean JSON error exit 1
            buf_cli = io.StringIO()
            with redirect_stdout(buf_cli):
                rc_cli = cli_main([
                    "plan-upgrade",
                    str(Path(tmp) / "bad_non_dict_list.jsonl"),
                    "--old-sha",
                    H_OLD,
                    "--new-sha",
                    H_NEW,
                    "--corpus-sha",
                    C_CUR,
                ])
            self.assertEqual(rc_cli, 1)
            self.assertFalse(json.loads(buf_cli.getvalue())["ok"])

        # EXO8-BUG-04: malformed ledger entries & non-ASCII digests raise ValueError (never KeyError/TypeError)
        valid_raw = {
            "seq": 1,
            "timestamp_utc": "2026-11-22T00:00:00Z",
            "event_type": "note",
            "actor_id": "a",
            "payload": {},
            "prev_sha256": "0" * 64,
            "entry_sha256": "1" * 64,
        }
        bad_ledger_dicts: list[tuple[str, Any]] = [
            ("non_mapping_int", 123),
            ("non_mapping_list", []),
            ("empty_dict", {}),
            ("payload_int", dict(valid_raw, payload=5)),
            ("non_str_event_type", dict(valid_raw, event_type=123)),
            ("non_ascii_entry_sha", dict(valid_raw, entry_sha256="\u2603" * 64)),
            ("bad_prev_sha", dict(valid_raw, prev_sha256="short")),
            ("non_str_sig_kind", dict(valid_raw, signature_kind=123)),
            ("non_ascii_sig", dict(valid_raw, signature="\u2603")),
        ]
        for label, bad_raw in bad_ledger_dicts:
            with self.assertRaises(ValueError, msg=f"LedgerEntry.from_dict({label}) must raise ValueError"):
                LedgerEntry.from_dict(bad_raw)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            lp_empty_obj = Path(tmp) / "empty_obj.jsonl"
            lp_empty_obj.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                AppendOnlyLedger(lp_empty_obj)

            lp_bad_utf8 = Path(tmp) / "bad_utf8.jsonl"
            lp_bad_utf8.write_bytes(b"\xff\xfe\n")
            with self.assertRaises(ValueError):
                AppendOnlyLedger(lp_bad_utf8)

            # Mutating an already-opened ledger file on disk to "{}\n" fails cleanly inside resolve_verified_entry
            lp_live = Path(tmp) / "live_mutated.jsonl"
            live_ledger = AppendOnlyLedger(lp_live)
            e_live = live_ledger.append("note", "a", {"ok": True})
            lp_live.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                live_ledger.resolve_verified_entry(e_live.entry_sha256)

    def test_r3_journal_completeness_metadata_contested_and_duplicate_probes(self) -> None:
        """Closes EXO8-R2-J1..J4 (receipt completeness, all registered metadata, contested state) & CLI dup probes."""
        planner = CrossGenerationPlanner()

        with tempfile.TemporaryDirectory() as tmp:
            # --- J1 & J2: Ledger-to-journal resolved receipt completeness ---
            lp_j12 = Path(tmp) / "ledger_j12.jsonl"
            sp_j12 = Path(tmp) / "sim_j12.jsonl"
            ledger_j12 = AppendOnlyLedger(lp_j12)
            sim_j12 = SIMMemory(sp_j12, ledger=ledger_j12)
            sim_j12.register_rule(_make_rule("pref.reg2", "preference", corpus_sha=C_CUR))
            ev1 = record_evaluation_receipt(
                ledger_j12,
                rule_id="pref.reg2",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
            )
            sim_j12.record_attempt("pref.reg2", _make_attempt(1, "regressed", evidence_receipt_sha256=ev1.entry_sha256))
            ev2 = record_evaluation_receipt(
                ledger_j12,
                rule_id="pref.reg2",
                round_id=2,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
            )
            sim_j12.record_attempt("pref.reg2", _make_attempt(2, "regressed", evidence_receipt_sha256=ev2.entry_sha256))

            # Clean journal reloads & escalates on 2 regressions
            clean_j12 = SIMMemory(sp_j12, ledger=ledger_j12)
            plan_clean = planner.plan_upgrade(
                clean_j12,
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
            )
            self.assertEqual(plan_clean.actions[0].action, "escalate_human")
            self.assertEqual(plan_clean.actions[0].reason, "repeated_preference_regressions")

            orig_lines = sp_j12.read_text(encoding="utf-8").splitlines()

            # J1: Relabel round 2 regressed attempt to round_id=99, outcome="pending", evidence_receipt_sha256=None
            j1_lines: list[str] = []
            for ln in orig_lines:
                obj = json.loads(ln)
                if obj.get("op") == "append_attempt" and obj["attempt"]["round_id"] == 2:
                    obj["attempt"]["round_id"] = 99
                    obj["attempt"]["outcome"] = "pending"
                    obj["attempt"]["evidence_receipt_sha256"] = None
                j1_lines.append(json.dumps(obj, sort_keys=True, separators=(",", ":")))
            sp_j1 = Path(tmp) / "sim_j1_relabeled.jsonl"
            sp_j1.write_text("\n".join(j1_lines) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as j1_reload_ctx:
                SIMMemory(sp_j1, ledger=ledger_j12)
            self.assertIn("missing resolved attempt", str(j1_reload_ctx.exception))
            mem_j1 = SIMMemory(sp_j1)
            with self.assertRaises(SIMEvidenceError) as j1_plan_ctx:
                planner.plan_upgrade(
                    mem_j1,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger_j12,
                )
            self.assertIn("missing resolved attempt", str(j1_plan_ctx.exception))

            # J2: Delete ALL append_attempt lines from sim.jsonl completely while 2 regressed receipts remain in ledger
            j2_lines = [ln for ln in orig_lines if json.loads(ln).get("op") != "append_attempt"]
            sp_j2 = Path(tmp) / "sim_j2_deleted.jsonl"
            sp_j2.write_text("\n".join(j2_lines) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as j2_reload_ctx:
                SIMMemory(sp_j2, ledger=ledger_j12)
            self.assertIn("missing resolved attempt", str(j2_reload_ctx.exception))
            mem_j2 = SIMMemory(sp_j2)
            with self.assertRaises(SIMEvidenceError) as j2_plan_ctx:
                planner.plan_upgrade(
                    mem_j2,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger_j12,
                )
            self.assertIn("missing resolved attempt", str(j2_plan_ctx.exception))

            # Duplicate round-1 attempt while dropping round-2 receipt MUST still fail completeness
            j2_dup_lines = [orig_lines[0], orig_lines[1], orig_lines[1]]
            sp_j2_dup = Path(tmp) / "sim_j2_dup.jsonl"
            sp_j2_dup.write_text("\n".join(j2_dup_lines) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp_j2_dup, ledger=ledger_j12)

            # --- J3: Cross-check ALL 5 registered metadata fields against sim_rule_registered ---
            lp_j3 = Path(tmp) / "ledger_j3.jsonl"
            sp_j3 = Path(tmp) / "sim_j3.jsonl"
            ledger_j3 = AppendOnlyLedger(lp_j3)
            sim_j3 = SIMMemory(sp_j3, ledger=ledger_j3)
            sim_j3.register_rule(_make_rule("pref.meta", "preference", corpus_sha=C_CUR))

            # Clean preference with corpus drift C_CUR != C_ALT escalates to human
            plan_drift = planner.plan_upgrade(
                SIMMemory(sp_j3, ledger=ledger_j3),
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_ALT,
            )
            self.assertEqual(plan_drift.actions[0].action, "escalate_human")
            self.assertEqual(plan_drift.actions[0].reason, "preference_corpus_sha_drifted_needs_human_confirmation")

            reg_obj = json.loads(sp_j3.read_text(encoding="utf-8").splitlines()[0])
            tamper_cases: tuple[tuple[str, str], ...] = (
                ("corpus_sha", C_ALT),
                ("capability_id", "cap.forged"),
                ("rule_kind", "durable_invariant"),
                ("adapter_id", "weft-eido"),
                ("base_checkpoint_sha", H_NEW),
            )
            for field_name, tampered_val in tamper_cases:
                mutated = json.loads(json.dumps(reg_obj))
                mutated["record"][field_name] = tampered_val
                sp_mut = Path(tmp) / f"sim_j3_{field_name}.jsonl"
                sp_mut.write_text(json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                with self.assertRaises(SIMEvidenceError, msg=f"reload must reject tampered {field_name}") as r_ctx:
                    SIMMemory(sp_mut, ledger=ledger_j3)
                self.assertIn(f"{field_name} mismatch", str(r_ctx.exception))
                mem_mut = SIMMemory(sp_mut)
                with self.assertRaises(SIMEvidenceError, msg=f"plan_upgrade must reject tampered {field_name}") as p_ctx:
                    planner.plan_upgrade(
                        mem_mut,
                        old_checkpoint_sha=H_OLD,
                        new_checkpoint_sha=H_NEW,
                        active_corpus_sha=C_ALT,
                        ledger=ledger_j3,
                    )
                self.assertIn(f"{field_name} mismatch", str(p_ctx.exception))

            # --- J4: Contested state cross-check against sim_rule_registered & sim_rule_contested ---
            lp_j4 = Path(tmp) / "ledger_j4.jsonl"
            sp_j4 = Path(tmp) / "sim_j4.jsonl"
            ledger_j4 = AppendOnlyLedger(lp_j4)
            sim_j4 = SIMMemory(sp_j4, ledger=ledger_j4)
            sim_j4.register_rule(_make_rule("pref.veto", "preference", corpus_sha=C_CUR))
            sim_j4.mark_contested("pref.veto", "hard veto 2 dispute")

            # Clean contested rule escalates
            plan_veto = planner.plan_upgrade(
                SIMMemory(sp_j4, ledger=ledger_j4),
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
            )
            self.assertEqual(plan_veto.actions[0].action, "escalate_human")
            self.assertEqual(plan_veto.actions[0].reason, "contested_rule:hard veto 2 dispute")

            # Deleting mark_contested from sim.jsonl MUST fail closed on BOTH reload & plan_upgrade
            j4_lines = sp_j4.read_text(encoding="utf-8").splitlines()
            sp_j4_deleted = Path(tmp) / "sim_j4_deleted.jsonl"
            sp_j4_deleted.write_text(j4_lines[0] + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as j4_r_ctx:
                SIMMemory(sp_j4_deleted, ledger=ledger_j4)
            self.assertIn("contested state mismatch", str(j4_r_ctx.exception))
            mem_j4_del = SIMMemory(sp_j4_deleted)
            with self.assertRaises(SIMEvidenceError) as j4_p_ctx:
                planner.plan_upgrade(
                    mem_j4_del,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger_j4,
                )
            self.assertIn("contested state mismatch", str(j4_p_ctx.exception))

            # Legitimate resolution via mark_contested(..., contested=False) updates ledger & clears cleanly
            sim_j4.mark_contested("pref.veto", "resolved by human steward", contested=False)
            resolved_plan = planner.plan_upgrade(
                SIMMemory(sp_j4, ledger=ledger_j4),
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
            )
            self.assertEqual(resolved_plan.actions[0].action, "recompile")

            # Rule initially registered with contested=True cannot be flipped to contested=False in sim.jsonl
            lp_j4b = Path(tmp) / "ledger_j4b.jsonl"
            sp_j4b = Path(tmp) / "sim_j4b.jsonl"
            ledger_j4b = AppendOnlyLedger(lp_j4b)
            sim_j4b = SIMMemory(sp_j4b, ledger=ledger_j4b)
            init_contested = SIMRuleRecord(
                rule_id="pref.init_veto",
                capability_id="cap.init_veto",
                name="init veto",
                problem="contested at birth",
                rule_kind="preference",
                adapter_id="corpus",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
                contested=True,
                contested_reason="born contested",
            )
            sim_j4b.register_rule(init_contested)
            sp_j4b_tampered = Path(tmp) / "sim_j4b_tampered.jsonl"
            sp_j4b_tampered.write_text(
                sp_j4b.read_text(encoding="utf-8").replace('"contested":true', '"contested":false'),
                encoding="utf-8",
            )
            with self.assertRaises(SIMEvidenceError) as j4b_ctx:
                SIMMemory(sp_j4b_tampered, ledger=ledger_j4b)
            self.assertIn("contested state mismatch", str(j4b_ctx.exception))

            # Non-bool contested rejected on mark_contested, SIMRuleRecord.from_dict, & malformed ledger entry
            with self.assertRaises(TypeError):
                sim_j4b.mark_contested("pref.init_veto", "bad type", contested=1)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                SIMRuleRecord.from_dict(dict(init_contested.to_dict(), contested="yes"))
            ledger_j4b.append(
                "sim_rule_contested",
                "rogue",
                {"rule_id": "pref.init_veto", "contested": "yes", "reason": "bad"},
            )
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp_j4b, ledger=ledger_j4b)

            # --- Bonus CLI Guard: duplicate rule_id in --probes JSON array fails closed with exit 1 ---
            lp_dup = Path(tmp) / "ledger_dup.jsonl"
            sp_dup = Path(tmp) / "sim_dup.jsonl"
            probes_dup = Path(tmp) / "probes_dup.json"
            ledger_dup = AppendOnlyLedger(lp_dup)
            sim_dup = SIMMemory(sp_dup, ledger=ledger_dup)
            sim_dup.register_rule(_make_rule("wk.dup", "model_workaround"))
            pr_dup = record_baseline_probe_receipt(
                ledger_dup,
                rule_id="wk.dup",
                checkpoint_sha=H_NEW,
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha=C_CUR,
            )
            probes_dup.write_text(
                json.dumps([
                    {
                        "rule_id": "wk.dup",
                        "passed_natively": False,
                        "observed_corpus_sha": C_CUR,
                        "evidence_receipt_sha256": UNKNOWN_DIGEST,
                    },
                    {
                        "rule_id": "wk.dup",
                        "passed_natively": True,
                        "observed_corpus_sha": C_CUR,
                        "evidence_receipt_sha256": pr_dup.entry_sha256,
                    },
                ]),
                encoding="utf-8",
            )
            buf_dup = io.StringIO()
            with redirect_stdout(buf_dup):
                rc_dup = cli_main([
                    "plan-upgrade",
                    str(sp_dup),
                    "--old-sha",
                    H_OLD,
                    "--new-sha",
                    H_NEW,
                    "--corpus-sha",
                    C_CUR,
                    "--probes",
                    str(probes_dup),
                    "--ledger",
                    str(lp_dup),
                ])
            self.assertEqual(rc_dup, 1)
            out_dup = json.loads(buf_dup.getvalue())
            self.assertFalse(out_dup["ok"])
            self.assertIn("duplicate baseline probe", out_dup["error"])

    def test_whole_rule_deletion_rename_orphan_and_5000_digit_journal_fail_closed(self) -> None:
        """EXO8-R3-J5 (whole-rule deletion / rename-to-ghost) & EXO8-R3-J6 (5000-digit int JSON)."""
        planner = CrossGenerationPlanner()
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "ledger_j5.jsonl"
            sp = Path(tmp) / "sim_j5.jsonl"
            ledger = AppendOnlyLedger(lp)
            sim = SIMMemory(sp, ledger=ledger)
            sim.register_rule(_make_rule("pref.reg_j5", "preference"))
            e1 = record_evaluation_receipt(
                ledger,
                rule_id="pref.reg_j5",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
            )
            sim.record_attempt(
                "pref.reg_j5",
                _make_attempt(1, "regressed", evidence_receipt_sha256=e1.entry_sha256),
            )
            sim.mark_contested("pref.reg_j5", "j5 dispute")

            # Benign non-SIM ledger event carrying outcome/rule_id keys must NOT break completeness
            ledger.append("audit_note", "exo-audit", {"rule_id": "unrelated.note", "outcome": "verified"})
            clean_plan = planner.plan_upgrade(
                SIMMemory(sp, ledger=ledger),
                old_checkpoint_sha=H_OLD,
                new_checkpoint_sha=H_NEW,
                active_corpus_sha=C_CUR,
            )
            self.assertEqual(clean_plan.actions[0].action, "escalate_human")

            # J5a: Whole-rule deletion (empty sim.jsonl) fails closed on BOTH reload & plan_upgrade
            sp_empty = Path(tmp) / "sim_j5_empty.jsonl"
            sp_empty.write_text("", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as r_empty:
                SIMMemory(sp_empty, ledger=ledger)
            self.assertIn("omitted from SIM journal", str(r_empty.exception))
            mem_empty = SIMMemory(sp_empty)
            with self.assertRaises(SIMEvidenceError) as p_empty:
                planner.plan_upgrade(
                    mem_empty,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger,
                )
            self.assertIn("omitted from SIM journal", str(p_empty.exception))

            # J5b: Renaming rule_id in sim.jsonl (pref.reg_j5 -> pref.ghost) fails closed on BOTH reload & plan_upgrade
            reg_line = json.loads(sp.read_text(encoding="utf-8").splitlines()[0])
            reg_line["record"]["rule_id"] = "pref.ghost"
            sp_ghost = Path(tmp) / "sim_j5_ghost.jsonl"
            sp_ghost.write_text(json.dumps(reg_line, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as r_ghost:
                SIMMemory(sp_ghost, ledger=ledger)
            self.assertIn("omitted from SIM journal", str(r_ghost.exception))
            mem_ghost = SIMMemory(sp_ghost)
            with self.assertRaises(SIMEvidenceError) as p_ghost:
                planner.plan_upgrade(
                    mem_ghost,
                    old_checkpoint_sha=H_OLD,
                    new_checkpoint_sha=H_NEW,
                    active_corpus_sha=C_CUR,
                    ledger=ledger,
                )
            self.assertIn("omitted from SIM journal", str(p_ghost.exception))

            # J5c: Orphaned resolved sim_evaluation_receipt & orphaned sim_rule_contested with no registration
            lp_eval_only = Path(tmp) / "ledger_eval_only.jsonl"
            led_eval = AppendOnlyLedger(lp_eval_only)
            record_evaluation_receipt(
                led_eval,
                rule_id="pref.orphan_eval",
                round_id=1,
                outcome="regressed",
                base_checkpoint_sha=H_OLD,
                corpus_sha=C_CUR,
            )
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp_empty, ledger=led_eval)

            lp_cont_only = Path(tmp) / "ledger_cont_only.jsonl"
            led_cont = AppendOnlyLedger(lp_cont_only)
            led_cont.append("sim_rule_contested", "exo-sim", {"rule_id": "pref.orphan_cont", "contested": True})
            with self.assertRaises(SIMEvidenceError):
                SIMMemory(sp_empty, ledger=led_cont)

            # J6: 5000-digit integer JSON line in sim.jsonl raises SIMEvidenceError (never raw 4300-digit ValueError)
            sp_big5000 = Path(tmp) / "sim_big5000.jsonl"
            sp_big5000.write_text('{"op":"register_rule","x":' + ("9" * 5000) + "}\n", encoding="utf-8")
            with self.assertRaises(SIMEvidenceError) as big_ctx:
                SIMMemory(sp_big5000)
            self.assertIn("malformed SIM journal JSON", str(big_ctx.exception))


if __name__ == "__main__":
    unittest.main()
