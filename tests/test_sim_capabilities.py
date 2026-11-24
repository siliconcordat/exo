"""Tests for durable capabilities, SIM causal memory, and cross-generation upgrade planner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from exo.capabilities import AdaptationProposal, CapabilityCharter, CapabilityRegistry, PeerResponse
from exo.ledger import AppendOnlyLedger
from exo.sim import (
    BaselineProbe,
    CrossGenerationPlanner,
    SIMAttempt,
    SIMMemory,
    SIMRuleRecord,
    record_baseline_probe_receipt,
    record_evaluation_receipt,
)


class TestCapabilitiesAndSIM(unittest.TestCase):
    def test_capability_survives_relocation_and_rejects_stale_peer_response(self) -> None:
        ledger = AppendOnlyLedger()
        reg = CapabilityRegistry(ledger=ledger)
        reg.register(
            CapabilityCharter(
                capability_id="cap.approval-gate",
                steward_actor="exo-steward",
                human_purpose="Bind high-risk actions to exact-byte human sign-off",
                success_signals=("1-byte tamper rejected",),
                failure_signals=("shell bypass allowed",),
                implementation_paths=("src/gate_v1.py",),
                peers=("cap.mde-broker",),
            )
        )
        reg.register(
            CapabilityCharter(
                capability_id="cap.mde-broker",
                steward_actor="mde-steward",
                human_purpose="Provide read-before-write CST splicing",
                success_signals=("read_token checked",),
                failure_signals=("unread patch applied",),
                implementation_paths=("src/broker.rs",),
            )
        )

        # Relocate implementation without changing capability_id or purpose
        moved = reg.relocate("cap.approval-gate", ("src/exo/gate.py",), actor="exo-steward")
        self.assertEqual(moved.capability_id, "cap.approval-gate")
        self.assertEqual(moved.version, 2)
        self.assertEqual(moved.implementation_paths, ("src/exo/gate.py",))
        self.assertIn("exact-byte", moved.human_purpose)

        prop = reg.propose(
            AdaptationProposal(
                proposal_id="prop-1",
                proposer_actor="exo-steward",
                affected_capabilities=("cap.approval-gate", "cap.mde-broker"),
                candidate_digest="cand-sha-v1",
                rationale="Require read_token on splice_patch envelopes",
            )
        )
        reg.respond(
            PeerResponse(
                actor="mde-steward",
                proposal_id="prop-1",
                proposal_digest=prop.digest(),
                candidate_digest="cand-sha-v1",
                decision="support",
                comment="Verified against broker contract",
            )
        )

        check_v1 = reg.check_proposal("prop-1", live_candidate_digest="cand-sha-v1")
        self.assertTrue(check_v1["advisory_ready"])
        self.assertFalse(check_v1["grants_merge_authority"])

        # Candidate bytes mutate -> earlier peer response becomes stale immediately
        check_v2 = reg.check_proposal("prop-1", live_candidate_digest="cand-sha-v2")
        self.assertFalse(check_v2["advisory_ready"])
        self.assertIn("mde-steward", check_v2["stale_responses_from"])
        self.assertEqual(ledger.verify_chain(), [])

    def test_sim_persistence_and_cross_generation_upgrade_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim.jsonl"
            ledger = AppendOnlyLedger(Path(tmp) / "ledger.jsonl")
            sim = SIMMemory(sim_path, ledger=ledger)

            # 1. Durable invariant: must always recompile on W_new
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.clean-writing",
                    capability_id="cap.prose",
                    name="Plain language without em dashes",
                    problem="Model inserts em dashes and meta-frames in PR reviews",
                    rule_kind="durable_invariant",
                    adapter_id="corpus",
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-1",
                )
            )
            clean_eval = record_evaluation_receipt(
                ledger,
                rule_id="rule.clean-writing",
                round_id=1,
                outcome="improved",
                base_checkpoint_sha="ckpt-old",
                corpus_sha="corpus-sha-1",
            )
            sim.record_attempt(
                "rule.clean-writing",
                SIMAttempt(
                    round_id=1,
                    what="Add prose canary examples",
                    predicted="Eliminate em dashes on held-out PRs",
                    fixed=("em_dash_rate",),
                    missed=("outcome_forecasts",),
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-1",
                    outcome="improved",
                    evidence_receipt_sha256=clean_eval.entry_sha256,
                ),
            )

            # 2. Old-model workaround: W_new fixes natively -> retire_candidate
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.xml-eos-fix",
                    capability_id="cap.tool-format",
                    name="Suppress premature EOS before XML tool block",
                    problem="Old 4B checkpoint emitted premature EOS before tool call",
                    rule_kind="model_workaround",
                    adapter_id="corpus",
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-1",
                )
            )

            # 3. Contested preference rule -> escalate_human (Hard Veto 2)
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.html-theme",
                    capability_id="cap.html-delivery",
                    name="Static dark HTML vs reactive OKLCH page",
                    problem="Two teams disagree on default report surface",
                    rule_kind="preference",
                    adapter_id="weft-eido",
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-1",
                    contested=True,
                    contested_reason="corpus static dark HTML conflicts with weft-eido reactive page",
                )
            )

            # 4. Workaround still failing on W_new -> recompile
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.lean-omega-guard",
                    capability_id="cap.fv",
                    name="Avoid nonlinear products in Lean omega",
                    problem="Model hands nonlinear integer products directly to omega",
                    rule_kind="model_workaround",
                    adapter_id="formal-verification",
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-1",
                )
            )

            # 5. Uncontested preference whose corpus_sha drifted -> escalate_human
            sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.tone-preference",
                    capability_id="cap.tone",
                    name="Custom summary cadence",
                    problem="User tone preference",
                    rule_kind="preference",
                    adapter_id="corpus",
                    base_checkpoint_sha="ckpt-old",
                    corpus_sha="corpus-sha-old",
                )
            )

            # Reload from append-only disk journal with backing ledger
            reloaded = SIMMemory(sim_path, ledger=ledger)
            self.assertEqual(len(reloaded.get("rule.clean-writing").attempts), 1)
            self.assertEqual(
                reloaded.get("rule.clean-writing").attempts[0].evidence_receipt_sha256,
                clean_eval.entry_sha256,
            )

            xml_probe = record_baseline_probe_receipt(
                ledger,
                rule_id="rule.xml-eos-fix",
                checkpoint_sha="ckpt-new",
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha="corpus-sha-1",
            )
            omega_probe = record_baseline_probe_receipt(
                ledger,
                rule_id="rule.lean-omega-guard",
                checkpoint_sha="ckpt-new",
                passed_natively=False,
                bare_model=True,
                observed_corpus_sha="corpus-sha-1",
            )
            prose_probe = record_baseline_probe_receipt(
                ledger,
                rule_id="rule.clean-writing",
                checkpoint_sha="ckpt-new",
                passed_natively=True,
                bare_model=True,
                observed_corpus_sha="corpus-sha-1",
            )

            planner = CrossGenerationPlanner()
            plan = planner.plan_upgrade(
                reloaded,
                old_checkpoint_sha="ckpt-old",
                new_checkpoint_sha="ckpt-new",
                active_corpus_sha="corpus-sha-1",
                probes={
                    "rule.xml-eos-fix": BaselineProbe(
                        "rule.xml-eos-fix",
                        passed_natively=True,
                        observed_corpus_sha="corpus-sha-1",
                        evidence_receipt_sha256=xml_probe.entry_sha256,
                    ),
                    "rule.lean-omega-guard": BaselineProbe(
                        "rule.lean-omega-guard",
                        passed_natively=False,
                        observed_corpus_sha="corpus-sha-1",
                        evidence_receipt_sha256=omega_probe.entry_sha256,
                    ),
                    "rule.clean-writing": BaselineProbe(
                        "rule.clean-writing",
                        passed_natively=True,
                        observed_corpus_sha="corpus-sha-1",
                        evidence_receipt_sha256=prose_probe.entry_sha256,
                    ),
                },
                ledger=ledger,
            )

            by_id = {a.rule_id: a for a in plan.actions}
            self.assertEqual(by_id["rule.clean-writing"].action, "recompile")
            self.assertEqual(by_id["rule.xml-eos-fix"].action, "retire_candidate")
            self.assertEqual(by_id["rule.html-theme"].action, "escalate_human")
            self.assertEqual(by_id["rule.lean-omega-guard"].action, "recompile")
            self.assertEqual(by_id["rule.tone-preference"].action, "escalate_human")
            self.assertEqual(ledger.verify_chain(), [])

    def test_sim_memory_pre_json_redacts_secrets_in_journal_and_memory(self) -> None:
        gh_tok = "ghp_" + "Z9y8X7w6V5u4T3s2R1q0P9o8"
        pem_blob = "-----BEGIN " + "EC PRIVATE KEY-----\nSECRET_BODY_LINE\n-----END " + "EC PRIVATE KEY-----"
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim-redacted.jsonl"
            ledger = AppendOnlyLedger(Path(tmp) / "ledger.jsonl")
            sim = SIMMemory(sim_path, ledger=ledger)
            rec = sim.register_rule(
                SIMRuleRecord(
                    rule_id="rule.secret-test",
                    capability_id="cap.audit",
                    name="Redact raw secret",
                    problem=f"Observed leak {gh_tok} and\n{pem_blob}",
                    rule_kind="durable_invariant",
                    adapter_id="corpus",
                    base_checkpoint_sha="a" * 64,
                    corpus_sha="b" * 64,
                )
            )
            secret_eval = record_evaluation_receipt(
                ledger,
                rule_id="rule.secret-test",
                round_id=1,
                outcome="improved",
                base_checkpoint_sha="a" * 64,
                corpus_sha="b" * 64,
            )
            att = sim.record_attempt(
                "rule.secret-test",
                SIMAttempt(
                    round_id=1,
                    what=f"Saw api_key=myTopSecretVal12345 in trace",
                    predicted="Secret is redacted before JSON write",
                    fixed=("redacted",),
                    missed=(),
                    base_checkpoint_sha="a" * 64,
                    corpus_sha="b" * 64,
                    outcome="improved",
                    evidence_receipt_sha256=secret_eval.entry_sha256,
                ),
            )
            raw_journal = sim_path.read_text(encoding="utf-8")
            for leaked in (gh_tok, "SECRET_BODY_LINE", "myTopSecretVal12345"):
                self.assertNotIn(leaked, raw_journal)
                self.assertNotIn(leaked, rec.problem)
                self.assertNotIn(leaked, att.what)
            self.assertEqual(ledger.verify_chain(), [])


if __name__ == "__main__":
    unittest.main()
