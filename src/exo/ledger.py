"""Append-only hash-chained ledger for Layer 4 Exo harness memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

GENESIS_HASH = "0" * 64
REDACTED_PLACEHOLDER = "[REDACTED]"
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# Pre-JSON raw-string secret patterns (applied BEFORE json.dumps escaping so
# multiline PEM private keys and quoted secrets cannot slip past JSON escaping).
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
        REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}\b"),
        REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
        r"\1" + REDACTED_PLACEHOLDER,
    ),
    (
        re.compile(
            r'(?i)(\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|refresh[_-]?token|'
            r'client[_-]?secret|private[_-]?key|password|passwd)\s*[:=]\s*)'
            r'(?!sha256:)(?!\[REDACTED\])["\']?([^\s"\',;}{]{8,})["\']?'
        ),
        r"\1" + REDACTED_PLACEHOLDER,
    ),
)


def redact_secrets_in_text(text: str) -> str:
    """Redact raw private keys, tokens, and credentials on unescaped text."""
    out = text
    for pattern, repl in _SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_secrets_in_value(value: Any) -> Any:
    """Recursively redact secrets on raw Python strings BEFORE JSON serialization."""
    if isinstance(value, str):
        return redact_secrets_in_text(value)
    if isinstance(value, Mapping):
        return {
            (redact_secrets_in_text(k) if isinstance(k, str) else k): redact_secrets_in_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets_in_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets_in_value(item) for item in value]
    return value


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
        if not isinstance(raw, Mapping):
            raise ValueError(f"ledger entry must be a mapping (got {type(raw).__name__})")
        required = (
            "seq",
            "timestamp_utc",
            "event_type",
            "actor_id",
            "payload",
            "prev_sha256",
            "entry_sha256",
        )
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"ledger entry missing required field(s): {', '.join(missing)}")
        raw_seq = raw["seq"]
        if type(raw_seq) is not int or raw_seq < 1:
            raise ValueError(f"ledger entry seq must be a positive int (got {raw_seq!r})")
        raw_payload = raw["payload"]
        if not isinstance(raw_payload, dict):
            raise ValueError(f"ledger entry payload must be a dict (got {type(raw_payload).__name__})")
        for key in ("timestamp_utc", "event_type", "actor_id", "prev_sha256", "entry_sha256"):
            if type(raw[key]) is not str:
                raise ValueError(f"ledger entry {key} must be a string (got {type(raw[key]).__name__})")
        if (
            _SHA256_HEX_RE.fullmatch(raw["entry_sha256"]) is None
            or _SHA256_HEX_RE.fullmatch(raw["prev_sha256"]) is None
        ):
            raise ValueError("ledger entry digests must be 64-character lowercase hex SHA-256 strings")
        sig_kind = raw.get("signature_kind", "unsigned")
        sig = raw.get("signature")
        if type(sig_kind) is not str or (sig is not None and (type(sig) is not str or not sig.isascii())):
            raise ValueError("ledger entry signature metadata must be ASCII strings")
        return cls(
            seq=raw_seq,
            timestamp_utc=raw["timestamp_utc"],
            event_type=raw["event_type"],
            actor_id=raw["actor_id"],
            payload=dict(raw_payload),
            prev_sha256=raw["prev_sha256"],
            entry_sha256=raw["entry_sha256"],
            signature_kind=sig_kind,
            signature=sig,
        )


class AppendOnlyLedger:
    """Content-addressed append-only JSONL ledger (file-backed or in-memory)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[LedgerEntry] = []
        if self.path is not None and self.path.exists():
            self._entries = self._load_from_disk()

    def _load_from_disk(self) -> list[LedgerEntry]:
        if self.path is None:
            raise RuntimeError("cannot load ledger from disk when path is None")
        loaded: list[LedgerEntry] = []
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read ledger file {self.path}: {exc}") from exc
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
        safe_actor = redact_secrets_in_text(actor_id)
        safe_payload = redact_secrets_in_value(dict(payload))
        body = {
            "seq": seq,
            "timestamp_utc": ts,
            "event_type": event_type,
            "actor_id": safe_actor,
            "payload": safe_payload,
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
            actor_id=safe_actor,
            payload=safe_payload,
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

    def verified_entries(
        self,
        *,
        event_type: Optional[str] = None,
        signing_key: Optional[bytes] = None,
        require_signatures: bool = False,
        require_file: bool = False,
    ) -> list[LedgerEntry]:
        """Verify full hash-chain integrity and return verified entries."""
        errors = self.verify_chain(
            signing_key=signing_key,
            require_signatures=require_signatures,
        )
        if errors:
            raise ValueError(f"ledger verification failed: {errors[0]}")
        if self.path is not None:
            if not self.path.exists():
                if require_file or self._entries:
                    raise ValueError(f"ledger file does not exist: {self.path}")
                records: list[LedgerEntry] = []
            else:
                records = self._load_from_disk()
        else:
            records = list(self._entries)
        if event_type is None:
            return records
        return [e for e in records if e.event_type == event_type]

    def resolve_verified_entry(
        self,
        entry_sha256: str,
        *,
        expected_event_type: Optional[str] = None,
        signing_key: Optional[bytes] = None,
        require_signatures: bool = False,
    ) -> LedgerEntry:
        """Verify full chain integrity and resolve one entry by exact entry_sha256."""
        if type(entry_sha256) is not str or _SHA256_HEX_RE.fullmatch(entry_sha256) is None:
            raise ValueError("entry_sha256 must be a 64-character lowercase hex SHA-256 digest")
        records = self.verified_entries(
            signing_key=signing_key,
            require_signatures=require_signatures,
            require_file=True,
        )
        for entry in records:
            if hmac.compare_digest(entry.entry_sha256, entry_sha256):
                if expected_event_type is not None and entry.event_type != expected_event_type:
                    raise ValueError(
                        f"ledger receipt {entry_sha256} has event_type {entry.event_type!r}, "
                        f"expected {expected_event_type!r}"
                    )
                return entry
        raise ValueError(f"ledger receipt digest not found: {entry_sha256}")
