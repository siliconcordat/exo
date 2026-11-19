"""Tests covering all 7 cross-MDE interference hazards and cross-org RepoPolicy rules."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from exo.ledger import AppendOnlyLedger
from exo.policy import (
    RepoPolicy,
    corpus_repo_policy,
    foundation0_repo_policy,
    proven_commons_repo_policy,
    validate_change_against_policy,
    weft_eido_repo_policy,
)
from exo.router import (
    ContestedPreferenceError,
    RouteRequest,
    ZoneManifestError,
    ZoneRouter,
    compute_zone_manifest_id,
)


class TestZoneRouterAndPolicy(unittest.TestCase):
    def test_hazard_1_and_2_formal_proof_overrides_corpus_prose_rules(self) -> None:
        ledger = AppendOnlyLedger()
        router = ZoneRouter(ledger=ledger)
        policy = proven_commons_repo_policy()
        decision = router.route(
            RouteRequest(
                zone="formal_proof",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=policy,
                file_path="TarPilot/Archive.lean",
                contains_formal_claims=True,
            )
        )
        scales = decision.scale_map()
        self.assertEqual(scales["formal-verification"], 1.0)
        self.assertEqual(scales["corpus"], 0.0)
        self.assertEqual(scales["weft-eido"], 0.0)
        # Hazard 1: mandatory 'What is not proved' disclaimers allowed
        self.assertTrue(decision.constraints.allow_negative_disclaimers)
        # Hazard 2: Lean identifiers and 40-char verdict SHAs allowed
        self.assertTrue(decision.constraints.allow_exact_formal_notation)
        # Hazard 3: claim vocabulary locked to repo policy
        self.assertEqual(decision.constraints.claim_vocabulary, "proven_commons")
        # Hazard 6 & 7: exact math + data-eido-skip
        self.assertTrue(decision.constraints.forbid_float_in_exact_math)
        self.assertTrue(decision.constraints.enforce_data_eido_skip)
        self.assertEqual(ledger.verify_chain(), [])

    def test_prose_zone_activates_corpus_only(self) -> None:
        router = ZoneRouter()
        decision = router.route(
            RouteRequest(
                zone="prose",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=corpus_repo_policy(),
            )
        )
        self.assertEqual(decision.scale_map(), {"corpus": 1.0, "weft-eido": 0.0, "formal-verification": 0.0})
        self.assertFalse(decision.constraints.allow_negative_disclaimers)
        self.assertFalse(decision.constraints.allow_telegraphic_condense)
        self.assertEqual(decision.constraints.claim_vocabulary, "corpus")

    def test_hazard_4_condensed_json_allows_terse_and_caveman_slots(self) -> None:
        router = ZoneRouter()
        decision = router.route(
            RouteRequest(
                zone="condensed_json",
                active_domains=("corpus", "weft-eido"),
                repo_policy=weft_eido_repo_policy(),
                file_path="site/condensed.json",
                contains_formal_claims=False,
            )
        )
        self.assertEqual(decision.scale_map(), {"corpus": 0.0, "weft-eido": 1.0})
        self.assertTrue(decision.constraints.allow_telegraphic_condense)

    def test_hazard_5_hard_veto_2_escalates_contested_html_delivery(self) -> None:
        router = ZoneRouter()
        # No explicit Plane 1 precedence -> must raise ContestedPreferenceError (never average 0.5/0.5)
        with self.assertRaises(ContestedPreferenceError) as ctx:
            router.route(
                RouteRequest(
                    zone="html_report",
                    active_domains=("corpus", "weft-eido"),
                    repo_policy=corpus_repo_policy(),
                )
            )
        self.assertEqual(ctx.exception.conflict_key, "html_delivery")

        # Explicit Plane 1 precedence resolves cleanly without silent weight guessing
        resolved_policy = RepoPolicy(
            **{**corpus_repo_policy().to_dict(), "precedence_overrides": {"html_delivery": "weft-eido"}}
        )
        decision = router.route(
            RouteRequest(
                zone="html_report",
                active_domains=("corpus", "weft-eido"),
                repo_policy=resolved_policy,
            )
        )
        self.assertEqual(decision.scale_map(), {"corpus": 0.0, "weft-eido": 1.0})

    def test_hazard_6_and_7_research_page_and_condense_formal_protection(self) -> None:
        router = ZoneRouter()
        f0_policy = foundation0_repo_policy()

        # Hazard 7: condensing formal claims without data-eido-skip fails closed
        with self.assertRaises(ContestedPreferenceError) as ctx:
            router.route(
                RouteRequest(
                    zone="condensed_json",
                    active_domains=("weft-eido", "formal-verification"),
                    repo_policy=f0_policy,
                    contains_formal_claims=True,
                    data_eido_skip_marked=False,
                )
            )
        self.assertEqual(ctx.exception.conflict_key, "condense_formal_claims")

        # Marked with data-eido-skip succeeds with skip enforced
        ok_condense = router.route(
            RouteRequest(
                zone="condensed_json",
                active_domains=("weft-eido", "formal-verification"),
                repo_policy=f0_policy,
                contains_formal_claims=True,
                data_eido_skip_marked=True,
            )
        )
        self.assertTrue(ok_condense.constraints.enforce_data_eido_skip)

        # Hazard 6: research_page composes weft-eido + formal-verification with certificate & illustrative bounds
        research = router.route(
            RouteRequest(
                zone="research_page",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=f0_policy,
                contains_formal_claims=True,
                contains_certified_plot=True,
                contains_illustrative_toy=True,
            )
        )
        self.assertEqual(research.scale_map(), {"corpus": 0.0, "weft-eido": 1.0, "formal-verification": 1.0})
        self.assertTrue(research.constraints.forbid_float_in_exact_math)
        self.assertTrue(research.constraints.enforce_data_eido_skip)
        self.assertTrue(research.constraints.require_certificate_binding)
        self.assertTrue(research.constraints.require_illustrative_tag)
        self.assertEqual(research.constraints.claim_vocabulary, "foundation0")

    def test_generic_code_zone_with_both_specialists_escalates_hard_veto_2(self) -> None:
        router = ZoneRouter()
        with self.assertRaises(ContestedPreferenceError) as ctx:
            router.route(
                RouteRequest(
                    zone="code",
                    active_domains=("corpus", "weft-eido", "formal-verification"),
                    repo_policy=foundation0_repo_policy(),
                )
            )
        self.assertEqual(ctx.exception.conflict_key, "code_specialist")

    def test_cross_org_repo_policy_enforcement_outside_weights(self) -> None:
        # 1. corpus: squash, no trailers, no CI workflows
        corpus_pol = corpus_repo_policy()
        v_corpus = validate_change_against_policy(
            corpus_pol,
            commit_message="Add rule\n\nCo-authored-by: bot <bot@example.invalid>",
            changed_paths=(".github/workflows/ci.yml",),
            merge_method="merge",
        )
        self.assertTrue(any("forbidden_trailer:Co-authored-by" in v for v in v_corpus), v_corpus)
        self.assertTrue(any("merge_strategy_mismatch" in v for v in v_corpus), v_corpus)
        self.assertTrue(any("ci_workflows_disabled" in v for v in v_corpus), v_corpus)

        # 2. weft-eido: DCO Signed-off-by required + linear history
        eido_pol = weft_eido_repo_policy()
        v_eido_bad = validate_change_against_policy(
            eido_pol,
            commit_message="Update simplex pad",
            changed_paths=("src/simplex-pad.js",),
            merge_method="squash",
        )
        self.assertTrue(any("missing_required_trailer:Signed-off-by" in v for v in v_eido_bad), v_eido_bad)
        self.assertTrue(any("merge_strategy_mismatch" in v for v in v_eido_bad), v_eido_bad)

        v_eido_ok = validate_change_against_policy(
            eido_pol,
            commit_message="Update simplex pad\n\nSigned-off-by: Dev <dev@example.invalid>",
            changed_paths=("src/simplex-pad.js",),
            merge_method="linear",
            author_id="alice",
            reviewer_id="bob",
        )
        self.assertEqual(v_eido_ok, [])

        # 3. Foundation-0: merge commit + locked checker never mixed with results
        f0_pol = foundation0_repo_policy()
        v_f0 = validate_change_against_policy(
            f0_pol,
            commit_message="Add new bound and adjust verifier",
            changed_paths=("verify.py", "results/bound-1/certificate.json"),
            merge_method="merge",
            author_id="alice",
            reviewer_id="alice",
        )
        self.assertTrue(any("locked_checker_and_result_mixed" in v for v in v_f0), v_f0)
        self.assertTrue(any("self_review_forbidden" in v for v in v_f0), v_f0)

        # 4. JSON round-trip from disk
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "repo-policy.json"
            p.write_text(json.dumps(f0_pol.to_dict()), encoding="utf-8")
            loaded = RepoPolicy.load(p)
            self.assertEqual(loaded, f0_pol)

    def test_zone_router_from_l5_zone_manifest_verifies_digest_and_compiles_rules(self) -> None:
        body = {
            "schema": "zone-manifest/v0.1",
            "mde_version": "0.1.0",
            "precedence_order": ["formal_proof", "condensed_json", "ui_module", "code", "prose"],
            "zones": [
                {
                    "zone": "formal_proof",
                    "l1_bank_tag": 3,
                    "protection_flags": {
                        "protect_axioms": True,
                        "protect_sha_digests": True,
                        "protect_exact_keys": True,
                        "respect_data_eido_skip": True,
                        "preserve_sibling_comments": True,
                        "enforce_prose_guard": False,
                        "prose_rewrite_allowed": False,
                    },
                    "default_l2_scales": {"corpus": 0.0, "formal-verification": 1.0, "weft-eido": 0.0},
                },
                {
                    "zone": "condensed_json",
                    "l1_bank_tag": 4,
                    "protection_flags": {
                        "protect_axioms": False,
                        "protect_sha_digests": True,
                        "protect_exact_keys": True,
                        "respect_data_eido_skip": True,
                        "preserve_sibling_comments": False,
                        "enforce_prose_guard": False,
                        "prose_rewrite_allowed": False,
                    },
                    "default_l2_scales": {"corpus": 0.0, "formal-verification": 0.0, "weft-eido": 0.0},
                },
                {
                    "zone": "ui_module",
                    "l1_bank_tag": 2,
                    "protection_flags": {
                        "protect_axioms": False,
                        "protect_sha_digests": True,
                        "protect_exact_keys": False,
                        "respect_data_eido_skip": True,
                        "preserve_sibling_comments": True,
                        "enforce_prose_guard": False,
                        "prose_rewrite_allowed": False,
                    },
                    "default_l2_scales": {"corpus": 0.2, "formal-verification": 0.0, "weft-eido": 1.0},
                },
                {
                    "zone": "code",
                    "l1_bank_tag": 1,
                    "protection_flags": {
                        "protect_axioms": False,
                        "protect_sha_digests": True,
                        "protect_exact_keys": False,
                        "respect_data_eido_skip": True,
                        "preserve_sibling_comments": True,
                        "enforce_prose_guard": False,
                        "prose_rewrite_allowed": False,
                    },
                    "default_l2_scales": {"corpus": 0.4, "formal-verification": 0.5, "weft-eido": 0.0},
                },
                {
                    "zone": "prose",
                    "l1_bank_tag": 0,
                    "protection_flags": {
                        "protect_axioms": False,
                        "protect_sha_digests": True,
                        "protect_exact_keys": False,
                        "respect_data_eido_skip": False,
                        "preserve_sibling_comments": True,
                        "enforce_prose_guard": True,
                        "prose_rewrite_allowed": True,
                    },
                    "default_l2_scales": {"corpus": 1.0, "formal-verification": 0.0, "weft-eido": 0.0},
                },
            ],
        }
        manifest = dict(body)
        manifest["manifest_id"] = compute_zone_manifest_id(body)

        ledger = AppendOnlyLedger()
        router = ZoneRouter.from_zone_manifest(manifest, ledger=ledger)
        self.assertEqual(router.manifest_id, manifest["manifest_id"])

        # Compiled ui_module uses exact manifest scales (corpus=0.2, weft-eido=1.0) and l1_bank_tag=2
        ui_dec = router.route(
            RouteRequest(
                zone="ui_module",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=weft_eido_repo_policy(),
            )
        )
        self.assertEqual(ui_dec.scale_map(), {"corpus": 0.2, "weft-eido": 1.0, "formal-verification": 0.0})
        self.assertEqual(ui_dec.l1_bank_tag, 2)
        self.assertEqual(ui_dec.manifest_id, manifest["manifest_id"])

        # Compiled formal_proof enforces data-eido-skip and exact math with l1_bank_tag=3
        fp_dec = router.route(
            RouteRequest(
                zone="formal_proof",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=foundation0_repo_policy(),
            )
        )
        self.assertEqual(fp_dec.scale_map(), {"corpus": 0.0, "weft-eido": 0.0, "formal-verification": 1.0})
        self.assertEqual(fp_dec.l1_bank_tag, 3)
        self.assertTrue(fp_dec.constraints.enforce_data_eido_skip)
        self.assertTrue(fp_dec.constraints.forbid_float_in_exact_math)

        # Fallback zones (research_page) still preserve manifest_id and forbid_float_in_exact_math
        rp_dec = router.route(
            RouteRequest(
                zone="research_page",
                active_domains=("corpus", "weft-eido", "formal-verification"),
                repo_policy=foundation0_repo_policy(),
                contains_formal_claims=True,
                contains_certified_plot=True,
                contains_illustrative_toy=True,
            )
        )
        self.assertEqual(rp_dec.manifest_id, manifest["manifest_id"])
        self.assertTrue(rp_dec.constraints.forbid_float_in_exact_math)

        # Hard Veto 2 still escalates on generic code with dual specialists
        with self.assertRaises(ContestedPreferenceError):
            router.route(
                RouteRequest(
                    zone="code",
                    active_domains=("corpus", "weft-eido", "formal-verification"),
                    repo_policy=foundation0_repo_policy(),
                )
            )

        # Any 1-byte tamper in the manifest fails closed on manifest_id SHA-256 verification
        tampered = json.loads(json.dumps(manifest))
        tampered["zones"][0]["default_l2_scales"]["corpus"] = 0.9
        with self.assertRaises(ZoneManifestError):
            ZoneRouter.from_zone_manifest(tampered)

        self.assertEqual(ledger.verify_chain(), [])


if __name__ == "__main__":
    unittest.main()
