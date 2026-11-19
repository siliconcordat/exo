"""Tests for durable capabilities, SIM causal memory, and cross-generation upgrade planner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from exo.capabilities import AdaptationProposal, CapabilityCharter, CapabilityRegistry, PeerResponse
from exo.ledger import AppendOnlyLedger
from exo.sim import BaselineProbe, CrossGenerationPlanner, SIMAttempt, SIMMemory, SIMRuleRecord


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

            # Reload from append-only disk journal
            reloaded = SIMMemory(sim_path)
            self.assertEqual(len(reloaded.get("rule.clean-writing").attempts), 1)

            planner = CrossGenerationPlanner()
            plan = planner.plan_upgrade(
                reloaded,
                old_checkpoint_sha="ckpt-old",
                new_checkpoint_sha="ckpt-new",
                active_corpus_sha="corpus-sha-1",
                probes={
                    "rule.xml-eos-fix": BaselineProbe("rule.xml-eos-fix", passed_natively=True),
                    "rule.lean-omega-guard": BaselineProbe("rule.lean-omega-guard", passed_natively=False),
                    "rule.clean-writing": BaselineProbe("rule.clean-writing", passed_natively=True),
                },
            )

            by_id = {a.rule_id: a for a in plan.actions}
            self.assertEqual(by_id["rule.clean-writing"].action, "recompile")
            self.assertEqual(by_id["rule.xml-eos-fix"].action, "retire_candidate")
            self.assertEqual(by_id["rule.html-theme"].action, "escalate_human")
            self.assertEqual(by_id["rule.lean-omega-guard"].action, "recompile")
            self.assertEqual(by_id["rule.tone-preference"].action, "escalate_human")
            self.assertEqual(ledger.verify_chain(), [])


if __name__ == "__main__":
    unittest.main()
