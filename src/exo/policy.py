"""Declarative Layer 2 per-repository policy manifests (never stored in model weights)."""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from exo.gate import normalize_repo_path

VALID_MERGE_STRATEGIES = frozenset({"squash", "merge", "linear"})
VALID_TRAILER_POLICIES = frozenset({"none", "dco", "allow_any"})
VALID_CLAIM_VOCABULARIES = frozenset({"corpus", "foundation0", "proven_commons", "f0_ledger"})

DCO_RE = re.compile(r"(?m)^Signed-off-by:\s+.+\s+<[^<>\s]+@[^<>\s]+>\s*$")
CO_AUTHORED_RE = re.compile(r"(?im)^Co-authored-by:")


@dataclass(frozen=True)
class RepoPolicy:
    """Declarative Layer 2 repository policy loaded from repo-policy.json."""

    repo_id: str
    merge_strategy: str  # "squash" | "merge" | "linear"
    commit_trailers: str  # "none" | "dco" | "allow_any"
    forbid_co_authored_by: bool
    license_code: str
    license_docs: str
    ci_enabled: bool = True
    require_non_author_review: bool = True
    protected_paths: tuple[str, ...] = ()
    locked_checker_paths: tuple[str, ...] = ()
    result_paths: tuple[str, ...] = ()
    claim_vocabulary: str = "corpus"
    precedence_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.merge_strategy not in VALID_MERGE_STRATEGIES:
            raise ValueError(f"invalid merge_strategy: {self.merge_strategy!r}")
        if self.commit_trailers not in VALID_TRAILER_POLICIES:
            raise ValueError(f"invalid commit_trailers: {self.commit_trailers!r}")
        if self.claim_vocabulary not in VALID_CLAIM_VOCABULARIES:
            raise ValueError(f"invalid claim_vocabulary: {self.claim_vocabulary!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "merge_strategy": self.merge_strategy,
            "commit_trailers": self.commit_trailers,
            "forbid_co_authored_by": self.forbid_co_authored_by,
            "license_code": self.license_code,
            "license_docs": self.license_docs,
            "ci_enabled": self.ci_enabled,
            "require_non_author_review": self.require_non_author_review,
            "protected_paths": list(self.protected_paths),
            "locked_checker_paths": list(self.locked_checker_paths),
            "result_paths": list(self.result_paths),
            "claim_vocabulary": self.claim_vocabulary,
            "precedence_overrides": dict(self.precedence_overrides),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepoPolicy":
        return cls(
            repo_id=str(raw["repo_id"]),
            merge_strategy=str(raw["merge_strategy"]),
            commit_trailers=str(raw["commit_trailers"]),
            forbid_co_authored_by=bool(raw.get("forbid_co_authored_by", True)),
            license_code=str(raw["license_code"]),
            license_docs=str(raw.get("license_docs", "CC-BY-4.0")),
            ci_enabled=bool(raw.get("ci_enabled", True)),
            require_non_author_review=bool(raw.get("require_non_author_review", True)),
            protected_paths=tuple(raw.get("protected_paths", ())),
            locked_checker_paths=tuple(raw.get("locked_checker_paths", ())),
            result_paths=tuple(raw.get("result_paths", ())),
            claim_vocabulary=str(raw.get("claim_vocabulary", "corpus")),
            precedence_overrides=dict(raw.get("precedence_overrides", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "RepoPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("repo policy file must contain a JSON object")
        return cls.from_dict(data)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(path, pat) or path == pat or path.startswith(pat.rstrip("/") + "/"):
            return True
    return False


def validate_change_against_policy(
    policy: RepoPolicy,
    *,
    commit_message: str,
    changed_paths: Sequence[str],
    merge_method: Optional[str] = None,
    author_id: Optional[str] = None,
    reviewer_id: Optional[str] = None,
) -> list[str]:
    """Deterministically enforce repository policy outside model weights.

    Returns a list of policy violation strings (empty means pass).
    """
    violations: list[str] = []

    if policy.forbid_co_authored_by and CO_AUTHORED_RE.search(commit_message):
        violations.append("forbidden_trailer:Co-authored-by")

    if policy.commit_trailers == "dco":
        if not DCO_RE.search(commit_message):
            violations.append("missing_required_trailer:Signed-off-by (DCO)")
    elif policy.commit_trailers == "none":
        if DCO_RE.search(commit_message):
            violations.append("unexpected_trailer:Signed-off-by disallowed when commit_trailers=none")

    if merge_method is not None and merge_method != policy.merge_strategy:
        violations.append(
            f"merge_strategy_mismatch:expected {policy.merge_strategy}, got {merge_method}"
        )

    if policy.require_non_author_review and author_id and reviewer_id:
        if author_id == reviewer_id:
            violations.append(f"self_review_forbidden:{author_id}")

    norm_paths: list[str] = []
    for raw in changed_paths:
        try:
            norm_paths.append(normalize_repo_path(raw))
        except ValueError as exc:
            violations.append(f"invalid_path:{exc}")

    if not policy.ci_enabled:
        for p in norm_paths:
            if p.startswith(".github/workflows/"):
                violations.append(f"ci_workflows_disabled:{p}")

    # Foundation-0 REPO-STANDARD rule: locked-checker edits never merge with result edits
    touched_checkers = [p for p in norm_paths if _matches_any(p, policy.locked_checker_paths)]
    touched_results = [p for p in norm_paths if _matches_any(p, policy.result_paths)]
    if touched_checkers and touched_results:
        violations.append(
            "locked_checker_and_result_mixed:"
            f"checkers={touched_checkers},results={touched_results}"
        )

    return violations


# Built-in reference presets matching audited organization conventions.

def corpus_repo_policy(repo_id: str = "aska-digital/corpus") -> RepoPolicy:
    return RepoPolicy(
        repo_id=repo_id,
        merge_strategy="squash",
        commit_trailers="none",
        forbid_co_authored_by=True,
        license_code="Apache-2.0",
        license_docs="CC-BY-4.0",
        ci_enabled=False,
        require_non_author_review=False,
        protected_paths=("principles/security.md", "sops/governance.md"),
        claim_vocabulary="corpus",
    )


def weft_eido_repo_policy(repo_id: str = "Moissanite-Labs/weft-eido") -> RepoPolicy:
    return RepoPolicy(
        repo_id=repo_id,
        merge_strategy="linear",
        commit_trailers="dco",
        forbid_co_authored_by=True,
        license_code="Apache-2.0",
        license_docs="CC-BY-4.0",
        ci_enabled=True,
        require_non_author_review=True,
        protected_paths=(".github/scripts/check-dco.sh", "test/*"),
        claim_vocabulary="corpus",
    )


def foundation0_repo_policy(repo_id: str = "Foundation-0/f0") -> RepoPolicy:
    return RepoPolicy(
        repo_id=repo_id,
        merge_strategy="merge",
        commit_trailers="none",
        forbid_co_authored_by=True,
        license_code="MIT",
        license_docs="CC-BY-4.0",
        ci_enabled=True,
        require_non_author_review=True,
        protected_paths=("SHA256SUMS", "scripts/gate.py", "verify.py", "verify_independent.py"),
        locked_checker_paths=("verify.py", "verify_independent.py", "scripts/gate.py", "SHA256SUMS"),
        result_paths=("results/*", "certificate.json", "docs/data/ledger.json", "tracks/*/results.tsv"),
        claim_vocabulary="foundation0",
    )


def proven_commons_repo_policy(repo_id: str = "Proven-Commons/tar") -> RepoPolicy:
    return RepoPolicy(
        repo_id=repo_id,
        merge_strategy="merge",
        commit_trailers="none",
        forbid_co_authored_by=True,
        license_code="Apache-2.0",
        license_docs="CC-BY-4.0",
        ci_enabled=True,
        require_non_author_review=True,
        protected_paths=("bar.toml", "SPEC.md", "comparator.json", "*Spec.lean", "tests/run_tests.py"),
        locked_checker_paths=("bar.toml", "comparator.json", "*Spec.lean"),
        result_paths=("*Proofs.lean", "tests/TEST-RESULTS.md"),
        claim_vocabulary="proven_commons",
    )


def siliconcordat_repo_policy(repo_id: str = "siliconcordat/exo") -> RepoPolicy:
    return RepoPolicy(
        repo_id=repo_id,
        merge_strategy="squash",
        commit_trailers="none",
        forbid_co_authored_by=True,
        license_code="Apache-2.0",
        license_docs="CC-BY-4.0",
        ci_enabled=True,
        require_non_author_review=True,
        protected_paths=("src/exo/gate.py", "src/exo/policy.py", "tests/*"),
        claim_vocabulary="corpus",
    )
