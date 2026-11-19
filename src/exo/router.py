"""Dynamic zone-scoped multi-MDE router enforcing precedence and Hard Veto 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from exo.ledger import AppendOnlyLedger
from exo.policy import RepoPolicy

ZONE_MANIFEST_SCHEMA = "zone-manifest/v0.1"
_MANIFEST_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

KNOWN_DOMAINS = ("corpus", "weft-eido", "formal-verification")
VALID_ZONES = frozenset({
    "prose",
    "code",
    "formal_proof",
    "ui_module",
    "condensed_json",
    "research_page",
    "html_report",
})


class ZoneManifestError(ValueError):
    """Raised when an L5 mde zone-manifest/v0.1 document fails integrity or schema checks."""


@dataclass(frozen=True)
class ZoneManifestRule:
    """Verified zone routing and protection rule compiled from L5 mde zone-manifest/v0.1."""

    zone: str
    l1_bank_tag: int
    default_l2_scales: Mapping[str, float]
    protection_flags: Mapping[str, bool]


def compute_zone_manifest_id(manifest: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 manifest_id matching siliconcordat/mde zone_manifest.rs."""
    body = {k: v for k, v in manifest.items() if k != "manifest_id"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def verify_zone_manifest(manifest: Mapping[str, Any]) -> tuple[str, tuple[str, ...], dict[str, ZoneManifestRule]]:
    """Verify an L5 mde zone-manifest/v0.1 mapping and return (manifest_id, precedence, rules)."""
    if not isinstance(manifest, Mapping):
        raise ZoneManifestError("zone manifest must be a JSON object")
    schema = manifest.get("schema")
    if schema != ZONE_MANIFEST_SCHEMA:
        raise ZoneManifestError(f"unsupported zone manifest schema: {schema!r} (expected {ZONE_MANIFEST_SCHEMA!r})")

    stored_id = manifest.get("manifest_id")
    if not isinstance(stored_id, str) or not _MANIFEST_ID_RE.match(stored_id):
        raise ZoneManifestError(f"invalid or missing manifest_id: {stored_id!r}")
    computed_id = compute_zone_manifest_id(manifest)
    if not hmac.compare_digest(stored_id, computed_id):
        raise ZoneManifestError(
            f"zone manifest_id mismatch: stored {stored_id}, computed {computed_id}"
        )

    raw_prec = manifest.get("precedence_order")
    if not isinstance(raw_prec, list) or not raw_prec or not all(isinstance(z, str) and z for z in raw_prec):
        raise ZoneManifestError("precedence_order must be a non-empty list of zone names")
    if len(set(raw_prec)) != len(raw_prec):
        raise ZoneManifestError("precedence_order contains duplicate zone names")

    raw_zones = manifest.get("zones")
    if not isinstance(raw_zones, list) or not raw_zones:
        raise ZoneManifestError("zones must be a non-empty list of zone entries")

    rules: dict[str, ZoneManifestRule] = {}
    for entry in raw_zones:
        if not isinstance(entry, Mapping):
            raise ZoneManifestError("each zone entry must be an object")
        zone_name = entry.get("zone")
        if not isinstance(zone_name, str) or zone_name not in VALID_ZONES:
            raise ZoneManifestError(f"unsupported zone name in manifest: {zone_name!r}")
        if zone_name in rules:
            raise ZoneManifestError(f"duplicate zone entry in manifest: {zone_name!r}")

        bank_tag = entry.get("l1_bank_tag")
        if type(bank_tag) is not int or bank_tag < 0:
            raise ZoneManifestError(f"zone {zone_name!r} l1_bank_tag must be a non-negative int")

        raw_scales = entry.get("default_l2_scales")
        if not isinstance(raw_scales, Mapping):
            raise ZoneManifestError(f"zone {zone_name!r} default_l2_scales must be a mapping")
        clean_scales: dict[str, float] = {}
        for dom, val in raw_scales.items():
            if dom not in KNOWN_DOMAINS:
                raise ZoneManifestError(f"zone {zone_name!r} references unknown domain {dom!r}")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ZoneManifestError(f"zone {zone_name!r} scale for {dom!r} must be numeric")
            val_f = float(val)
            if not math.isfinite(val_f) or val_f < 0.0:
                raise ZoneManifestError(f"zone {zone_name!r} scale for {dom!r} must be finite and non-negative")
            clean_scales[dom] = val_f

        raw_flags = entry.get("protection_flags")
        if not isinstance(raw_flags, Mapping):
            raise ZoneManifestError(f"zone {zone_name!r} protection_flags must be a mapping")
        clean_flags: dict[str, bool] = {}
        for fk, fv in raw_flags.items():
            if type(fv) is not bool:
                raise ZoneManifestError(f"zone {zone_name!r} protection flag {fk!r} must be bool")
            clean_flags[str(fk)] = fv

        rules[zone_name] = ZoneManifestRule(
            zone=zone_name,
            l1_bank_tag=bank_tag,
            default_l2_scales=clean_scales,
            protection_flags=clean_flags,
        )

    missing_prec = [z for z in raw_prec if z not in rules]
    if missing_prec:
        raise ZoneManifestError(f"precedence_order references undefined zone(s): {missing_prec}")

    return stored_id, tuple(raw_prec), rules


class ContestedPreferenceError(Exception):
    """Raised when two active MDEs conflict with no Plane 1 human precedence (Hard Veto 2)."""

    def __init__(
        self,
        *,
        conflict_key: str,
        conflicting_domains: tuple[str, ...],
        zone: str,
        detail: str,
    ) -> None:
        self.conflict_key = conflict_key
        self.conflicting_domains = conflicting_domains
        self.zone = zone
        self.detail = detail
        super().__init__(
            f"contested preference '{conflict_key}' in zone '{zone}' "
            f"between {', '.join(conflicting_domains)}: {detail}"
        )


@dataclass(frozen=True)
class RouteRequest:
    """Inputs from L5 syntactic zone tagging and active Layer 4 task state."""

    zone: str
    active_domains: tuple[str, ...]
    repo_policy: RepoPolicy
    file_path: Optional[str] = None
    contains_formal_claims: bool = False
    contains_certified_plot: bool = False
    contains_illustrative_toy: bool = False
    data_eido_skip_marked: bool = False


@dataclass(frozen=True)
class ZoneConstraints:
    """Deterministic Layer 2 constraints paired with the routed LoRA scale vector."""

    claim_vocabulary: str
    allow_negative_disclaimers: bool
    allow_exact_formal_notation: bool
    allow_telegraphic_condense: bool
    forbid_float_in_exact_math: bool
    enforce_data_eido_skip: bool
    require_certificate_binding: bool
    require_illustrative_tag: bool


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic multi-MDE scale vector and constraints for one forward pass."""

    zone: str
    lora_scales: tuple[dict[str, Any], ...]
    constraints: ZoneConstraints
    rationale: str
    manifest_id: Optional[str] = None
    l1_bank_tag: Optional[int] = None

    def scale_map(self) -> dict[str, float]:
        return {entry["id"]: float(entry["scale"]) for entry in self.lora_scales}


class ZoneRouter:
    """Computes per-zone LoRA scales across corpus, weft-eido, and formal-verification.

    Mitigates all seven verified cross-MDE interference hazards:
    1. corpus negative-absence ban vs mandatory formal 'What is not proved' blocks.
    2. corpus jargon/7-char SHA prose rule vs Lean identifiers and 40-char verdict SHAs.
    3. Claim-tier vocabulary drift across corpus / Foundation-0 / Proven-Commons / f0 Ledger.
    4. corpus anti-fragment rule vs weft-eido condensed.json Terse/Caveman levels.
    5. corpus static dark HTML vs weft-eido dual light/dark OKLCH ES modules (Hard Veto 2).
    6. weft-eido float64/illustrative toys vs formal exact-math & certificate-bound plots.
    7. Dynamic prose condensing over pinned formal specs (enforces data-eido-skip).
    """

    def __init__(
        self,
        ledger: Optional[AppendOnlyLedger] = None,
        *,
        manifest_id: Optional[str] = None,
        precedence_order: Optional[tuple[str, ...]] = None,
        manifest_rules: Optional[Mapping[str, ZoneManifestRule]] = None,
    ) -> None:
        self.ledger = ledger
        self.manifest_id = manifest_id
        self.precedence_order = precedence_order
        self._manifest_rules: Optional[dict[str, ZoneManifestRule]] = (
            dict(manifest_rules) if manifest_rules is not None else None
        )

    @classmethod
    def from_zone_manifest(
        cls,
        manifest_dict: Mapping[str, Any],
        *,
        ledger: Optional[AppendOnlyLedger] = None,
    ) -> "ZoneRouter":
        """Verify an L5 mde zone-manifest/v0.1 dict and compile routing rules directly from it."""
        manifest_id, precedence, rules = verify_zone_manifest(manifest_dict)
        if ledger is not None:
            ledger.append(
                "zone_manifest_loaded",
                "exo-router",
                {
                    "manifest_id": manifest_id,
                    "precedence_order": list(precedence),
                    "zones": sorted(rules.keys()),
                },
            )
        return cls(
            ledger=ledger,
            manifest_id=manifest_id,
            precedence_order=precedence,
            manifest_rules=rules,
        )

    @classmethod
    def from_zone_manifest_file(
        cls,
        path: str | Path,
        *,
        ledger: Optional[AppendOnlyLedger] = None,
    ) -> "ZoneRouter":
        """Load and verify a canonical L5 mde zone-manifest/v0.1 JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_zone_manifest(raw, ledger=ledger)

    def zone_rule(self, zone: str) -> Optional[ZoneManifestRule]:
        if self._manifest_rules is None:
            return None
        return self._manifest_rules.get(zone)

    def _route_from_manifest(
        self,
        req: RouteRequest,
        rule: ZoneManifestRule,
        active: set[str],
    ) -> RouteDecision:
        scales: dict[str, float] = {d: 0.0 for d in KNOWN_DOMAINS}
        flags = rule.protection_flags
        vocab = req.repo_policy.claim_vocabulary

        if req.zone == "code" and "formal-verification" in active and "weft-eido" in active:
            winner = req.repo_policy.precedence_overrides.get("code_specialist")
            if winner not in ("formal-verification", "weft-eido"):
                raise ContestedPreferenceError(
                    conflict_key="code_specialist",
                    conflicting_domains=("formal-verification", "weft-eido"),
                    zone=req.zone,
                    detail=(
                        "generic code zone has both formal-verification (exact int/Fraction) "
                        "and weft-eido (float64 JS) active; tag a specific zone "
                        "(formal_proof or ui_module) or set precedence_overrides['code_specialist']"
                    ),
                )
            if "corpus" in active:
                scales["corpus"] = float(rule.default_l2_scales.get("corpus", 0.0))
            scales[winner] = 1.0
            rationale = f"manifest {self.manifest_id} code zone: Plane 1 override '{winner}'"
        elif req.zone == "condensed_json" and req.contains_formal_claims and not req.data_eido_skip_marked:
            raise ContestedPreferenceError(
                conflict_key="condense_formal_claims",
                conflicting_domains=("weft-eido", "formal-verification"),
                zone=req.zone,
                detail=(
                    "formal claims cannot be dynamically condensed without explicit "
                    "data-eido-skip protection on pinned theorem/claim blocks"
                ),
            )
        else:
            for dom in KNOWN_DOMAINS:
                if dom in active:
                    scales[dom] = float(rule.default_l2_scales.get(dom, 0.0))
            rationale = (
                f"manifest {self.manifest_id} zone '{req.zone}' "
                f"(l1_bank_tag={rule.l1_bank_tag}): compiled from L5 zone-manifest/v0.1"
            )

        protect_axioms = bool(flags.get("protect_axioms", False))
        protect_exact_keys = bool(flags.get("protect_exact_keys", False))
        respect_skip = bool(flags.get("respect_data_eido_skip", False))
        formal_active = scales.get("formal-verification", 0.0) > 0.0

        ordered_scales = tuple(
            {"id": dom, "scale": float(scales[dom])}
            for dom in KNOWN_DOMAINS
            if dom in active
        )
        constraints = ZoneConstraints(
            claim_vocabulary=vocab,
            allow_negative_disclaimers=protect_axioms,
            allow_exact_formal_notation=protect_axioms or protect_exact_keys or formal_active,
            allow_telegraphic_condense=(req.zone == "condensed_json"),
            forbid_float_in_exact_math=protect_axioms or (req.zone == "code" and formal_active),
            enforce_data_eido_skip=respect_skip and (protect_axioms or req.contains_formal_claims),
            require_certificate_binding=req.contains_certified_plot,
            require_illustrative_tag=req.contains_illustrative_toy,
        )
        decision = RouteDecision(
            zone=req.zone,
            lora_scales=ordered_scales,
            constraints=constraints,
            rationale=rationale,
            manifest_id=self.manifest_id,
            l1_bank_tag=rule.l1_bank_tag,
        )
        if self.ledger is not None:
            self.ledger.append(
                "mde_route_selected",
                "exo-router",
                {
                    "zone": req.zone,
                    "repo_id": req.repo_policy.repo_id,
                    "file_path": req.file_path,
                    "lora_scales": list(ordered_scales),
                    "enforce_data_eido_skip": constraints.enforce_data_eido_skip,
                    "claim_vocabulary": constraints.claim_vocabulary,
                    "manifest_id": self.manifest_id,
                    "l1_bank_tag": rule.l1_bank_tag,
                },
            )
        return decision

    def route(self, req: RouteRequest) -> RouteDecision:
        if req.zone not in VALID_ZONES:
            raise ValueError(f"unknown syntactic zone: {req.zone!r}")
        unknown = [d for d in req.active_domains if d not in KNOWN_DOMAINS]
        if unknown:
            raise ValueError(f"unknown MDE domain(s): {unknown}")

        active = set(req.active_domains)
        if self._manifest_rules is not None and req.zone in self._manifest_rules:
            return self._route_from_manifest(req, self._manifest_rules[req.zone], active)
        scales: dict[str, float] = {d: 0.0 for d in KNOWN_DOMAINS}
        vocab = req.repo_policy.claim_vocabulary

        allow_negative_disclaimers = False
        allow_exact_formal_notation = False
        allow_telegraphic_condense = False
        forbid_float_in_exact_math = False
        enforce_data_eido_skip = False
        require_certificate_binding = False
        require_illustrative_tag = False
        rationale = ""

        if req.zone == "prose":
            if "corpus" in active:
                scales["corpus"] = 1.0
            rationale = "prose zone: corpus plain-language base active; specialist adapters muted"

        elif req.zone == "code":
            if "formal-verification" in active and "weft-eido" in active:
                winner = req.repo_policy.precedence_overrides.get("code_specialist")
                if winner not in ("formal-verification", "weft-eido"):
                    raise ContestedPreferenceError(
                        conflict_key="code_specialist",
                        conflicting_domains=("formal-verification", "weft-eido"),
                        zone=req.zone,
                        detail=(
                            "generic code zone has both formal-verification (exact int/Fraction) "
                            "and weft-eido (float64 JS) active; tag a specific zone "
                            "(formal_proof or ui_module) or set precedence_overrides['code_specialist']"
                        ),
                    )
                if "corpus" in active:
                    scales["corpus"] = 0.5
                scales[winner] = 1.0
                if winner == "formal-verification":
                    forbid_float_in_exact_math = True
                    allow_exact_formal_notation = True
                rationale = f"code zone: resolved by explicit Plane 1 precedence '{winner}'"
            else:
                if "corpus" in active:
                    scales["corpus"] = 0.5  # epistemic/review discipline without prose-only overrides
                if "formal-verification" in active:
                    scales["formal-verification"] = 1.0
                    forbid_float_in_exact_math = True
                    allow_exact_formal_notation = True
                    rationale = "code zone with formal verification: exact-math and dual-oracle habits active"
                elif "weft-eido" in active:
                    scales["weft-eido"] = 1.0
                    rationale = "code zone with weft-eido: zero-dependency ES module habits active"
                else:
                    rationale = "general code zone: corpus epistemic defaults active"

        elif req.zone == "formal_proof":
            # Hazards 1, 2, 6, 7: formal syntax, mandatory disclaimers, exact math, no condensing
            if "formal-verification" in active:
                scales["formal-verification"] = 1.0
            scales["corpus"] = 0.0
            scales["weft-eido"] = 0.0
            allow_negative_disclaimers = True
            allow_exact_formal_notation = True
            forbid_float_in_exact_math = True
            enforce_data_eido_skip = True
            rationale = (
                "formal_proof zone: formal-verification overrides corpus prose rules "
                "to preserve Lean syntax, 40-char SHAs, and mandatory limitation blocks"
            )

        elif req.zone == "ui_module":
            # Hazards 5 & 6: Eido float64 Canvas/WebGPU geometry allowed; corpus HTML/prose muted
            if "weft-eido" in active:
                scales["weft-eido"] = 1.0
            scales["corpus"] = 0.0
            scales["formal-verification"] = 0.0
            require_illustrative_tag = req.contains_illustrative_toy
            require_certificate_binding = req.contains_certified_plot
            if req.contains_formal_claims:
                enforce_data_eido_skip = True
            rationale = "ui_module zone: weft-eido active; float64 rendering permitted outside exact checkers"

        elif req.zone == "condensed_json":
            # Hazard 4 & Hazard 7: allow Terse/Caveman slots, but NEVER condense unskipped formal claims
            if req.contains_formal_claims and not req.data_eido_skip_marked:
                raise ContestedPreferenceError(
                    conflict_key="condense_formal_claims",
                    conflicting_domains=("weft-eido", "formal-verification"),
                    zone=req.zone,
                    detail=(
                        "formal claims cannot be dynamically condensed without explicit "
                        "data-eido-skip protection on pinned theorem/claim blocks"
                    ),
                )
            if "weft-eido" in active:
                scales["weft-eido"] = 1.0
            scales["corpus"] = 0.0
            scales["formal-verification"] = 0.0
            allow_telegraphic_condense = True
            enforce_data_eido_skip = req.contains_formal_claims
            rationale = (
                "condensed_json zone: weft-eido active; corpus anti-fragment rule muted "
                "for Terse/Caveman slots"
            )

        elif req.zone == "html_report":
            # Hazard 5 (Hard Veto 2): corpus dark static HTML vs weft-eido dual light/dark OKLCH
            if "corpus" in active and "weft-eido" in active:
                winner = req.repo_policy.precedence_overrides.get("html_delivery")
                if winner not in ("corpus", "weft-eido"):
                    raise ContestedPreferenceError(
                        conflict_key="html_delivery",
                        conflicting_domains=("corpus", "weft-eido"),
                        zone=req.zone,
                        detail=(
                            "corpus static dark-mode HTML conflicts with weft-eido reactive OKLCH modules; "
                            "set repo_policy.precedence_overrides['html_delivery'] explicitly"
                        ),
                    )
                scales[winner] = 1.0
                rationale = f"html_report zone: resolved by explicit Plane 1 precedence '{winner}'"
            elif "weft-eido" in active:
                scales["weft-eido"] = 1.0
                rationale = "html_report zone: weft-eido sole active design adapter"
            elif "corpus" in active:
                scales["corpus"] = 1.0
                rationale = "html_report zone: corpus static HTML defaults active"

        elif req.zone == "research_page":
            # Composition of all 3 MDEs on Foundation-0 / incentre-lattices style research pages
            if "formal-verification" in active and "weft-eido" in active:
                scales["formal-verification"] = 1.0
                scales["weft-eido"] = 1.0
                scales["corpus"] = 0.0  # specialist prose + UI govern; claim vocab stays locked
                allow_negative_disclaimers = True
                allow_exact_formal_notation = True
                forbid_float_in_exact_math = True
                enforce_data_eido_skip = True
                require_certificate_binding = req.contains_certified_plot
                require_illustrative_tag = req.contains_illustrative_toy
                rationale = (
                    "research_page zone: formal-verification + weft-eido composed with "
                    "data-eido-skip on formal claims and strict certificate/illustrative separation"
                )
            elif "formal-verification" in active:
                scales["formal-verification"] = 1.0
                allow_negative_disclaimers = True
                allow_exact_formal_notation = True
                forbid_float_in_exact_math = True
                require_certificate_binding = req.contains_certified_plot
                rationale = "research_page zone: formal-verification active"
            elif "weft-eido" in active:
                scales["weft-eido"] = 1.0
                require_illustrative_tag = req.contains_illustrative_toy
                rationale = "research_page zone: weft-eido active"
            elif "corpus" in active:
                scales["corpus"] = 1.0
                rationale = "research_page zone: corpus base active"

        ordered_scales = tuple(
            {"id": dom, "scale": float(scales[dom])}
            for dom in KNOWN_DOMAINS
            if dom in active
        )
        constraints = ZoneConstraints(
            claim_vocabulary=vocab,
            allow_negative_disclaimers=allow_negative_disclaimers,
            allow_exact_formal_notation=allow_exact_formal_notation,
            allow_telegraphic_condense=allow_telegraphic_condense,
            forbid_float_in_exact_math=forbid_float_in_exact_math,
            enforce_data_eido_skip=enforce_data_eido_skip,
            require_certificate_binding=require_certificate_binding,
            require_illustrative_tag=require_illustrative_tag,
        )
        decision = RouteDecision(
            zone=req.zone,
            lora_scales=ordered_scales,
            constraints=constraints,
            rationale=rationale,
            manifest_id=self.manifest_id,
        )
        if self.ledger is not None:
            ledger_payload: dict[str, Any] = {
                "zone": req.zone,
                "repo_id": req.repo_policy.repo_id,
                "file_path": req.file_path,
                "lora_scales": list(ordered_scales),
                "enforce_data_eido_skip": constraints.enforce_data_eido_skip,
                "claim_vocabulary": constraints.claim_vocabulary,
            }
            if self.manifest_id is not None:
                ledger_payload["manifest_id"] = self.manifest_id
            self.ledger.append(
                "mde_route_selected",
                "exo-router",
                ledger_payload,
            )
        return decision
