"""Tests for AppendOnlyLedger hash-chaining and tamper detection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from exo.ledger import AppendOnlyLedger


class TestAppendOnlyLedger(unittest.TestCase):
    def test_append_and_verify_chain_with_signatures(self) -> None:
        key = b"test-signing-key-bytes"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            ledger = AppendOnlyLedger(path)
            e1 = ledger.append("decision", "human-a", {"choice": "option_1"}, signing_key=key)
            e2 = ledger.append("tool_call", "exo-client", {"tool": "splice_locate"}, signing_key=key)

            self.assertEqual(e1.seq, 1)
            self.assertEqual(e2.seq, 2)
            self.assertEqual(e2.prev_sha256, e1.entry_sha256)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

            reloaded = AppendOnlyLedger(path)
            self.assertEqual(len(reloaded.entries()), 2)
            self.assertEqual(reloaded.verify_chain(signing_key=key, require_signatures=True), [])

    def test_one_byte_tamper_fails_chain_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.jsonl"
            ledger = AppendOnlyLedger(path)
            ledger.append("approval", "human-a", {"approved": True})
            ledger.append("deploy", "exo-ship", {"target": "staging"})

            lines = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["payload"]["approved"] = False
            lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            errors = ledger.verify_chain()
            self.assertTrue(any("entry_sha256 tampered" in err for err in errors), errors)

    def test_wrong_signing_key_rejected(self) -> None:
        ledger = AppendOnlyLedger()
        ledger.append("approval", "human-a", {"id": 1}, signing_key=b"key-one")
        errors = ledger.verify_chain(signing_key=b"key-two", require_signatures=True)
        self.assertTrue(any("invalid hmac-sha256 signature" in err for err in errors), errors)

    def test_unsigned_entry_fails_when_signatures_required(self) -> None:
        ledger = AppendOnlyLedger()
        ledger.append("note", "agent", {"msg": "plain"})
        self.assertEqual(ledger.verify_chain(), [])
        errors = ledger.verify_chain(signing_key=b"key", require_signatures=True)
        self.assertTrue(any("missing required signature" in err for err in errors), errors)


if __name__ == "__main__":
    unittest.main()
