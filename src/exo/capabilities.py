"""Durable capability identity and peer-adaptation tracking (continuous synthesis)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from exo.ledger import AppendOnlyLedger, canonical_json_bytes, sha256_hex, utc_now_iso


@dataclass(frozen=True)
class CapabilityCharter:
    """Durable software capability identity that survives path moves and rewrites."""

    capability_id: str
    steward_actor: str
    human_purpose: str
    success_signals: tuple[str, ...]
    failure_signals: tuple[str, ...]
    implementation_paths: tuple[str, ...]
    peers: tuple[str, ...] = ()
    version: int = 1
    created_at_utc: str = field(default_factory=utc_now_iso)

    def digest(self) -> str:
        return sha256_hex(
            canonical_json_bytes({
                "capability_id": self.capability_id,
                "steward_actor": self.steward_actor,
                "human_purpose": self.human_purpose,
                "success_signals": list(self.success_signals),
                "failure_signals": list(self.failure_signals),
                "implementation_paths": list(self.implementation_paths),
                "peers": list(self.peers),
                "version": self.version,
            })
        )


@dataclass(frozen=True)
class AdaptationProposal:
    """Proposed contract change affecting one or more capabilities."""

    proposal_id: str
    proposer_actor: str
    affected_capabilities: tuple[str, ...]
    candidate_digest: str
    rationale: str
    created_at_utc: str = field(default_factory=utc_now_iso)

    def digest(self) -> str:
        return sha256_hex(
            canonical_json_bytes({
                "proposal_id": self.proposal_id,
                "proposer_actor": self.proposer_actor,
                "affected_capabilities": sorted(self.affected_capabilities),
                "candidate_digest": self.candidate_digest,
                "rationale": self.rationale,
            })
        )


@dataclass(frozen=True)
class PeerResponse:
    """Cooperative peer steward verdict pinned to exact proposal and candidate digests."""

    actor: str
    proposal_id: str
    proposal_digest: str
    candidate_digest: str
    decision: str  # "support" | "revise" | "decline"
    comment: str
    recorded_at_utc: str = field(default_factory=utc_now_iso)


class CapabilityRegistry:
    """Tracks durable capabilities and digest-pinned peer responses."""

    def __init__(self, ledger: Optional[AppendOnlyLedger] = None) -> None:
        self._charters: dict[str, CapabilityCharter] = {}
        self._proposals: dict[str, AdaptationProposal] = {}
        self._responses: dict[str, list[PeerResponse]] = {}
        self.ledger = ledger

    def register(self, charter: CapabilityCharter) -> CapabilityCharter:
        if not charter.capability_id.strip():
            raise ValueError("capability_id must be non-empty")
        if not charter.human_purpose.strip():
            raise ValueError("human_purpose must be non-empty")
        self._charters[charter.capability_id] = charter
        if self.ledger is not None:
            self.ledger.append(
                "capability_registered",
                charter.steward_actor,
                {
                    "capability_id": charter.capability_id,
                    "version": charter.version,
                    "digest": charter.digest(),
                    "implementation_paths": list(charter.implementation_paths),
                },
            )
        return charter

    def get(self, capability_id: str) -> CapabilityCharter:
        if capability_id not in self._charters:
            raise KeyError(f"unknown capability_id: {capability_id}")
        return self._charters[capability_id]

    def relocate(
        self,
        capability_id: str,
        new_paths: Sequence[str],
        *,
        actor: str,
    ) -> CapabilityCharter:
        """Update implementation paths while keeping capability_id and human_purpose fixed."""
        prev = self.get(capability_id)
        updated = CapabilityCharter(
            capability_id=prev.capability_id,
            steward_actor=prev.steward_actor,
            human_purpose=prev.human_purpose,
            success_signals=prev.success_signals,
            failure_signals=prev.failure_signals,
            implementation_paths=tuple(new_paths),
            peers=prev.peers,
            version=prev.version + 1,
        )
        self._charters[capability_id] = updated
        if self.ledger is not None:
            self.ledger.append(
                "capability_relocated",
                actor,
                {
                    "capability_id": capability_id,
                    "old_paths": list(prev.implementation_paths),
                    "new_paths": list(updated.implementation_paths),
                    "version": updated.version,
                },
            )
        return updated

    def propose(self, proposal: AdaptationProposal) -> AdaptationProposal:
        for cid in proposal.affected_capabilities:
            if cid not in self._charters:
                raise KeyError(f"unknown affected capability_id: {cid}")
        self._proposals[proposal.proposal_id] = proposal
        self._responses.setdefault(proposal.proposal_id, [])
        if self.ledger is not None:
            self.ledger.append(
                "adaptation_proposed",
                proposal.proposer_actor,
                {
                    "proposal_id": proposal.proposal_id,
                    "proposal_digest": proposal.digest(),
                    "candidate_digest": proposal.candidate_digest,
                    "affected_capabilities": list(proposal.affected_capabilities),
                },
            )
        return proposal

    def respond(self, response: PeerResponse) -> PeerResponse:
        if response.decision not in ("support", "revise", "decline"):
            raise ValueError(f"invalid peer decision: {response.decision!r}")
        if response.proposal_id not in self._proposals:
            raise KeyError(f"unknown proposal_id: {response.proposal_id}")
        self._responses[response.proposal_id].append(response)
        if self.ledger is not None:
            self.ledger.append(
                "adaptation_response",
                response.actor,
                {
                    "proposal_id": response.proposal_id,
                    "proposal_digest": response.proposal_digest,
                    "candidate_digest": response.candidate_digest,
                    "decision": response.decision,
                },
            )
        return response

    def required_peers(self, proposal_id: str) -> set[str]:
        proposal = self._proposals[proposal_id]
        actors: set[str] = set()
        for cid in proposal.affected_capabilities:
            charter = self._charters[cid]
            if charter.steward_actor != proposal.proposer_actor:
                actors.add(charter.steward_actor)
            for peer_cid in charter.peers:
                if peer_cid in self._charters:
                    peer_actor = self._charters[peer_cid].steward_actor
                    if peer_actor != proposal.proposer_actor:
                        actors.add(peer_actor)
        return actors

    def check_proposal(self, proposal_id: str, live_candidate_digest: str) -> dict[str, Any]:
        """Check whether all required peers support the exact live candidate bytes.

        Returns an advisory readiness report; peer support is cooperative
        evidence and never substitutes for human merge or deploy authority.
        """
        proposal = self._proposals[proposal_id]
        expected_p_digest = proposal.digest()
        required = self.required_peers(proposal_id)
        latest_valid: dict[str, PeerResponse] = {}
        stale_actors: list[str] = []

        for resp in self._responses.get(proposal_id, []):
            if resp.proposal_digest != expected_p_digest or resp.candidate_digest != live_candidate_digest:
                stale_actors.append(resp.actor)
                continue
            latest_valid[resp.actor] = resp

        missing = sorted(required - set(latest_valid))
        non_supporting = sorted(
            actor for actor, resp in latest_valid.items() if resp.decision != "support"
        )
        candidate_matches_proposal = proposal.candidate_digest == live_candidate_digest
        advisory_ready = (
            candidate_matches_proposal
            and bool(required)
            and not missing
            and not non_supporting
        )
        return {
            "proposal_id": proposal_id,
            "advisory_ready": advisory_ready,
            "candidate_current": candidate_matches_proposal,
            "required_peers": sorted(required),
            "missing_peers": missing,
            "non_supporting_peers": non_supporting,
            "stale_responses_from": sorted(set(stale_actors)),
            "grants_merge_authority": False,
        }
