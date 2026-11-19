"""Append-only hash-chained ledger for Layer 4 Exo harness memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping, Optional

GENESIS_HASH = "0" * 64


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_bytes(obj: Any) -> bytes:
    """Encode a JSON value deterministically, rejecting NaN and Infinity."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Compute lowercase hexadecimal SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hmac_sha256_hex(key: bytes, data: bytes) -> str:
    """Compute lowercase hexadecimal HMAC-SHA256 over raw bytes."""
    return hmac.new(key, data, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class LedgerEntry:
    """Single immutable record in the Exo append-only memory chain."""

    seq: int
    timestamp_utc: str
    event_type: str
    actor_id: str
    payload: dict[str, Any]
    prev_sha256: str
    entry_sha256: str
    signature_kind: str = "unsigned"
    signature: Optional[str] = None

    def body_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "payload": self.payload,
            "prev_sha256": self.prev_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.body_dict()
        data["entry_sha256"] = self.entry_sha256
        data["signature_kind"] = self.signature_kind
        data["signature"] = self.signature
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LedgerEntry":
        return cls(
            seq=int(raw["seq"]),
            timestamp_utc=str(raw["timestamp_utc"]),
            event_type=str(raw["event_type"]),
            actor_id=str(raw["actor_id"]),
            payload=dict(raw["payload"]),
            prev_sha256=str(raw["prev_sha256"]),
            entry_sha256=str(raw["entry_sha256"]),
            signature_kind=str(raw.get("signature_kind", "unsigned")),
            signature=raw.get("signature"),
        )


class AppendOnlyLedger:
    """Content-addressed append-only JSONL ledger (file-backed or in-memory)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[LedgerEntry] = []
        if self.path is not None and self.path.exists():
            self._entries = self._load_from_disk()

    def _load_from_disk(self) -> list[LedgerEntry]:
        assert self.path is not None
        loaded: list[LedgerEntry] = []
        raw_text = self.path.read_text(encoding="utf-8")
        for line_no, line in enumerate(raw_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"non-object ledger record at line {line_no}")
            loaded.append(LedgerEntry.from_dict(obj))
        return loaded

    @property
    def head_sha256(self) -> str:
        if not self._entries:
            return GENESIS_HASH
        return self._entries[-1].entry_sha256

    def append(
        self,
        event_type: str,
        actor_id: str,
        payload: Mapping[str, Any],
        *,
        signing_key: Optional[bytes] = None,
        timestamp_utc: Optional[str] = None,
    ) -> LedgerEntry:
        """Append one content-addressed entry and persist it if file-backed."""
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type must be a non-empty string")
        if not actor_id or not isinstance(actor_id, str):
            raise ValueError("actor_id must be a non-empty string")
        seq = len(self._entries) + 1
        ts = timestamp_utc or utc_now_iso()
        prev = self.head_sha256
        body = {
            "seq": seq,
            "timestamp_utc": ts,
            "event_type": event_type,
            "actor_id": actor_id,
            "payload": dict(payload),
            "prev_sha256": prev,
        }
        body_bytes = canonical_json_bytes(body)
        entry_hash = sha256_hex(body_bytes)
        if signing_key is not None:
            sig_kind = "hmac-sha256"
            sig_val: Optional[str] = hmac_sha256_hex(signing_key, entry_hash.encode("ascii"))
        else:
            sig_kind = "unsigned"
            sig_val = None
        entry = LedgerEntry(
            seq=seq,
            timestamp_utc=ts,
            event_type=event_type,
            actor_id=actor_id,
            payload=dict(payload),
            prev_sha256=prev,
            entry_sha256=entry_hash,
            signature_kind=sig_kind,
            signature=sig_val,
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = canonical_json_bytes(entry.to_dict()).decode("ascii") + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        self._entries.append(entry)
        return entry

    def entries(self, event_type: Optional[str] = None) -> list[LedgerEntry]:
        """Return all entries, optionally filtered by event_type."""
        if event_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.event_type == event_type]

    def verify_chain(
        self,
        *,
        signing_key: Optional[bytes] = None,
        require_signatures: bool = False,
    ) -> list[str]:
        """Verify full hash-chain integrity (reloading from disk when file-backed)."""
        errors: list[str] = []
        if self.path is not None:
            if not self.path.exists():
                return []
            try:
                records = self._load_from_disk()
            except ValueError as exc:
                return [str(exc)]
        else:
            records = list(self._entries)

        expected_prev = GENESIS_HASH
        for idx, entry in enumerate(records, 1):
            if entry.seq != idx:
                errors.append(f"entry {idx}: seq mismatch (got {entry.seq})")
            if entry.prev_sha256 != expected_prev:
                errors.append(
                    f"entry {idx}: prev_sha256 mismatch "
                    f"(expected {expected_prev[:12]}, got {entry.prev_sha256[:12]})"
                )
            recomputed = sha256_hex(canonical_json_bytes(entry.body_dict()))
            if not hmac.compare_digest(recomputed, entry.entry_sha256):
                errors.append(
                    f"entry {idx}: entry_sha256 tampered "
                    f"(expected {recomputed[:12]}, got {entry.entry_sha256[:12]})"
                )
            if require_signatures and entry.signature_kind != "hmac-sha256":
                errors.append(f"entry {idx}: missing required signature")
            if entry.signature_kind == "hmac-sha256":
                if signing_key is None:
                    errors.append(f"entry {idx}: signed entry cannot be verified without signing_key")
                else:
                    expected_sig = hmac_sha256_hex(signing_key, entry.entry_sha256.encode("ascii"))
                    if not entry.signature or not hmac.compare_digest(expected_sig, entry.signature):
                        errors.append(f"entry {idx}: invalid hmac-sha256 signature")
            expected_prev = entry.entry_sha256
        return errors
