"""Command-line entry point for Layer 4 Exo harness verification utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from exo.ledger import AppendOnlyLedger
from exo.policy import (
    RepoPolicy,
    corpus_repo_policy,
    foundation0_repo_policy,
    proven_commons_repo_policy,
    siliconcordat_repo_policy,
    validate_change_against_policy,
    weft_eido_repo_policy,
)
from exo.router import ContestedPreferenceError, RouteRequest, ZoneRouter
from exo.sim import BaselineProbe, CrossGenerationPlanner, SIMEvidenceError, SIMMemory

PRESETS = {
    "corpus": corpus_repo_policy,
    "weft-eido": weft_eido_repo_policy,
    "foundation0": foundation0_repo_policy,
    "proven-commons": proven_commons_repo_policy,
    "siliconcordat": siliconcordat_repo_policy,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="exo", description="Layer 4 Exo harness CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_led = sub.add_parser("verify-ledger", help="Verify an append-only JSONL ledger file.")
    p_led.add_argument("ledger_path", type=Path)

    p_route = sub.add_parser("route-zone", help="Compute deterministic multi-MDE scale vector.")
    p_route.add_argument("--zone", required=True)
    p_route.add_argument("--domains", required=True, help="Comma-separated active domains.")
    p_route.add_argument("--preset", default="siliconcordat", choices=sorted(PRESETS))
    p_route.add_argument("--policy-file", type=Path, default=None)
    p_route.add_argument("--formal-claims", action="store_true")
    p_route.add_argument("--data-eido-skip", action="store_true")

    p_pol = sub.add_parser("check-policy", help="Check commit message and paths against RepoPolicy.")
    p_pol.add_argument("--preset", default="siliconcordat", choices=sorted(PRESETS))
    p_pol.add_argument("--policy-file", type=Path, default=None)
    p_pol.add_argument("--message", required=True)
    p_pol.add_argument("--merge-method", default=None)
    p_pol.add_argument("paths", nargs="*")

    p_up = sub.add_parser("plan-upgrade", help="Classify SIM rules across a base checkpoint upgrade.")
    p_up.add_argument("sim_path", type=Path)
    p_up.add_argument("--old-sha", required=True)
    p_up.add_argument("--new-sha", required=True)
    p_up.add_argument("--corpus-sha", required=True)
    p_up.add_argument("--probes", type=Path, default=None)
    p_up.add_argument("--ledger", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "verify-ledger":
        ledger = AppendOnlyLedger(args.ledger_path)
        errs = ledger.verify_chain()
        print(json.dumps({"ok": not errs, "entries": len(ledger.entries()), "errors": errs}, indent=2))
        return 0 if not errs else 1

    if args.cmd == "route-zone":
        policy = RepoPolicy.load(args.policy_file) if args.policy_file else PRESETS[args.preset]()
        domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
        router = ZoneRouter()
        try:
            decision = router.route(
                RouteRequest(
                    zone=args.zone,
                    active_domains=domains,
                    repo_policy=policy,
                    contains_formal_claims=args.formal_claims,
                    data_eido_skip_marked=args.data_eido_skip,
                )
            )
        except ContestedPreferenceError as exc:
            print(json.dumps({"escalate_human": True, "error": str(exc)}, indent=2))
            return 2
        print(
            json.dumps(
                {
                    "zone": decision.zone,
                    "lora_scales": list(decision.lora_scales),
                    "constraints": decision.constraints.__dict__,
                    "rationale": decision.rationale,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "check-policy":
        policy = RepoPolicy.load(args.policy_file) if args.policy_file else PRESETS[args.preset]()
        violations = validate_change_against_policy(
            policy,
            commit_message=args.message,
            changed_paths=args.paths,
            merge_method=args.merge_method,
        )
        print(json.dumps({"ok": not violations, "violations": violations}, indent=2))
        return 0 if not violations else 1

    if args.cmd == "plan-upgrade":
        try:
            ledger = AppendOnlyLedger(args.ledger) if args.ledger is not None else None
            sim = SIMMemory(args.sim_path, ledger=ledger)
            probes: dict[str, BaselineProbe] = {}
            if args.probes is not None:
                raw_probes = json.loads(args.probes.read_text(encoding="utf-8"))
                if not isinstance(raw_probes, list):
                    raise ValueError("probes file must contain a JSON array")
                for item in raw_probes:
                    p = BaselineProbe.from_dict(item)
                    if p.rule_id in probes:
                        raise ValueError(f"duplicate baseline probe for rule_id {p.rule_id!r}")
                    probes[p.rule_id] = p
            plan = CrossGenerationPlanner().plan_upgrade(
                sim,
                old_checkpoint_sha=args.old_sha,
                new_checkpoint_sha=args.new_sha,
                active_corpus_sha=args.corpus_sha,
                probes=probes,
                ledger=ledger,
            )
        except (SIMEvidenceError, ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        print(
            json.dumps(
                {
                    "old_checkpoint_sha": plan.old_checkpoint_sha,
                    "new_checkpoint_sha": plan.new_checkpoint_sha,
                    "active_corpus_sha": plan.active_corpus_sha,
                    "actions": [a.__dict__ for a in plan.actions],
                },
                indent=2,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
