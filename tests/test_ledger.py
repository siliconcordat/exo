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

    def test_pre_json_secret_redaction_strips_multiline_pem_and_tokens_before_disk_write(self) -> None:
        pem_secret = (
            "-----BEGIN " + "PRIVATE KEY-----\n"
            "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSk\n"
            "\"quoted-line-inside-pem\"\n"
            "-----END " + "PRIVATE KEY-----"
        )
        gh_tok = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
        aws_key = "AKIA" + "1234567890ABCDEF"
        api_sk = "sk-proj-" + "z9Y8x7W6v5U4t3S2r1Q0"
        bearer = "Bearer " + "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        normal_digest = "sha256:" + "a" * 64

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "redacted.jsonl"
            ledger = AppendOnlyLedger(ledger_path)
            entry = ledger.append(
                "audit_note",
                "agent",
                {
                    "pem": f"leaked:\n{pem_secret}\nend",
                    "tokens": [gh_tok, aws_key, api_sk, bearer],
                    "inline": "password=superSecretValue999",
                    "file_read_token": normal_digest,
                },
                signing_key=b"redact-key",
            )
            disk_text = ledger_path.read_text(encoding="utf-8")
            for forbidden in (
                "MIIEvgIBADAN",
                gh_tok,
                aws_key,
                api_sk,
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                "superSecretValue999",
            ):
                self.assertNotIn(forbidden, disk_text)
                self.assertNotIn(forbidden, json.dumps(entry.payload))
            # Normal SHA-256 read tokens stay intact and chain verifies cleanly
            self.assertEqual(entry.payload["file_read_token"], normal_digest)
            self.assertEqual(ledger.verify_chain(signing_key=b"redact-key", require_signatures=True), [])


if __name__ == "__main__":
    unittest.main()
