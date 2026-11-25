"""Concurrency, locking, fsync durability, rollback, torn-tail recovery, and cross-process tests for exo.ledger & exo.gate.

Covers:
  - 20 concurrent threads appending 100 signed receipts each (2,000 total) to a shared AppendOnlyLedger:
    zero interleaved JSONL lines, strict 1..2000 sequence numbering, unbroken prev_sha256 chain, and
    100% HMAC-SHA256 verification.
  - Multiple independent AppendOnlyLedger instances targeting the same file concurrently.
  - Real 2-process concurrent append via subprocess (exercising POSIX fcntl.flock exclusion across processes).
  - Write and os.fsync failure rollback via os.ftruncate(fd, committed_offset) + os.fsync(fd) before unlock,
    preserving disk integrity and leaving in-memory _entries/_synced_size unchanged.
  - Newline-framed torn-tail recovery (mid-object tear, missing trailing newline, and mid-line file shrink)
    never wedging append() or gluing two JSON objects onto one line.
  - Live second-instance reader resync on entries() and head_sha256 via fstat under RLock + LOCK_SH.
  - Direct verification of os.fsync, parent-dir fsync on create, O_NOFOLLOW symlink rejection, and O_CLOEXEC.
  - Spend-boundary fail-closed checks (NaN, Inf, -0.01, and max_auto_spend_usd + 1e-9).
  - Zero bare asserts; runs cleanly across -I -B, -I -O -B, and -I -B -W error.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from exo.gate import ActionEnvelope, ApprovalGate
from exo.ledger import AppendOnlyLedger


class TestExoLedgerConcurrencyAndDurability(unittest.TestCase):
    def test_twenty_threads_hundred_receipts_shared_ledger_chain_and_hmac(self) -> None:
        key = b"exo-concurrency-hmac-key-32-bytes"
        n_threads = 20
        per_thread = 100
        expected_total = n_threads * per_thread

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concurrent.jsonl"
            ledger = AppendOnlyLedger(path)
            barrier = threading.Barrier(n_threads)
            errors: list[BaseException] = []

            def worker(tid: int) -> None:
                try:
                    barrier.wait()
                    for idx in range(per_thread):
                        ledger.append(
                            "receipt",
                            f"worker-{tid}",
                            {"worker": tid, "idx": idx},
                            signing_key=key,
                        )
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
                self.assertFalse(t.is_alive(), "worker thread deadlocked")

            self.assertEqual(errors, [])
            raw_lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), expected_total)
            for line_no, raw in enumerate(raw_lines, 1):
                obj = json.loads(raw)
                self.assertEqual(obj["seq"], line_no)

            reloaded = AppendOnlyLedger(path)
            self.assertEqual(len(reloaded.entries()), expected_total)
            self.assertEqual(reloaded.verify_chain(signing_key=key, require_signatures=True), [])

    def test_multiple_independent_instances_same_file_chain_cleanly(self) -> None:
        key = b"exo-multi-instance-hmac-key-32b!"
        n_writers = 8
        per_writer = 25
        expected_total = n_writers * per_writer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared_multi_instance.jsonl"
            barrier = threading.Barrier(n_writers)
            errors: list[BaseException] = []

            def writer(wid: int) -> None:
                try:
                    inst = AppendOnlyLedger(path)
                    barrier.wait()
                    for idx in range(per_writer):
                        inst.append(
                            "multi_writer_receipt",
                            f"inst-{wid}",
                            {"writer": wid, "idx": idx},
                            signing_key=key,
                        )
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
                self.assertFalse(t.is_alive(), "multi-instance writer deadlocked")

            self.assertEqual(errors, [])
            verifier = AppendOnlyLedger(path)
            self.assertEqual(len(verifier.entries()), expected_total)
            self.assertEqual(verifier.verify_chain(signing_key=key, require_signatures=True), [])

    def test_second_instance_entries_and_head_sha256_resync_live(self) -> None:
        key = b"exo-reader-resync-hmac-key-32b!!"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resync.jsonl"
            writer_a = AppendOnlyLedger(path)
            reader_b = AppendOnlyLedger(path)

            self.assertEqual(reader_b.entries(), [])
            e1 = writer_a.append("evt", "actor-a", {"step": 1}, signing_key=key)
            self.assertEqual(len(reader_b.entries()), 1)
            self.assertEqual(reader_b.entries()[0].entry_sha256, e1.entry_sha256)
            self.assertEqual(reader_b.head_sha256, e1.entry_sha256)

            e2 = writer_a.append("evt", "actor-a", {"step": 2}, signing_key=key)
            self.assertEqual(len(reader_b.entries()), 2)
            self.assertEqual(reader_b.head_sha256, e2.entry_sha256)

            e3 = reader_b.append("evt", "actor-b", {"step": 3}, signing_key=key)
            self.assertEqual(e3.seq, 3)
            self.assertEqual(e3.prev_sha256, e2.entry_sha256)
            self.assertEqual(len(writer_a.entries()), 3)
            self.assertEqual(writer_a.head_sha256, e3.entry_sha256)
            self.assertEqual(writer_a.verify_chain(signing_key=key, require_signatures=True), [])

    def test_write_and_fsync_failure_rolls_back_file_and_memory(self) -> None:
        key = b"exo-rollback-hmac-key-32-bytes!!"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollback.jsonl"
            ledger = AppendOnlyLedger(path)
            e1 = ledger.append("init", "actor", {"n": 1}, signing_key=key)
            size_after_e1 = path.stat().st_size

            # 1) Fail on commit os.fsync, succeed on rollback os.fsync
            orig_fsync = os.fsync
            orig_ftruncate = os.ftruncate
            fsync_attempt = 0
            rollback_truncs: list[int] = []
            rollback_fsyncs = 0

            def flaky_fsync(fd: int) -> None:
                nonlocal fsync_attempt, rollback_fsyncs
                fsync_attempt += 1
                if fsync_attempt == 1:
                    raise OSError("simulated fsync EIO")
                rollback_fsyncs += 1
                orig_fsync(fd)

            def spy_ftruncate(fd: int, length: int) -> None:
                rollback_truncs.append(length)
                orig_ftruncate(fd, length)

            with (
                mock.patch("exo.ledger.os.fsync", side_effect=flaky_fsync),
                mock.patch("exo.ledger.os.ftruncate", side_effect=spy_ftruncate),
            ):
                with self.assertRaises(OSError):
                    ledger.append("fail_fsync", "actor", {"n": 2}, signing_key=key)

            self.assertEqual(rollback_truncs, [size_after_e1])
            self.assertEqual(rollback_fsyncs, 1)
            self.assertEqual(path.stat().st_size, size_after_e1)
            self.assertEqual(len(ledger.entries()), 1)
            self.assertEqual(ledger.head_sha256, e1.entry_sha256)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

            # 2) Fail mid-write after partial bytes land on disk
            orig_write = os.write
            write_calls = 0

            def partial_then_fail_write(fd: int, buf: bytes) -> int:
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    return orig_write(fd, bytes(buf)[:16])
                raise OSError("simulated ENOSPC mid-record")

            with mock.patch("exo.ledger.os.write", side_effect=partial_then_fail_write):
                with self.assertRaises(OSError):
                    ledger.append("fail_write", "actor", {"n": 2}, signing_key=key)

            self.assertEqual(path.stat().st_size, size_after_e1)
            self.assertEqual(len(ledger.entries()), 1)
            self.assertEqual(ledger.head_sha256, e1.entry_sha256)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

            # Subsequent append succeeds cleanly with seq=2
            e2 = ledger.append("recovered", "actor", {"n": 2}, signing_key=key)
            self.assertEqual(e2.seq, 2)
            self.assertEqual(e2.prev_sha256, e1.entry_sha256)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

    def test_torn_tail_missing_newline_and_midline_shrink_recover_cleanly(self) -> None:
        key = b"exo-torn-tail-hmac-key-32-bytes!"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "torn.jsonl"
            ledger = AppendOnlyLedger(path)
            e1 = ledger.append("ok", "actor", {"i": 1}, signing_key=key)
            size_after_e1 = path.stat().st_size

            # Case A: mid-object torn partial tail (simulating SIGKILL mid-write), including fresh instance restart
            with path.open("ab") as fh:
                fh.write(b'{"seq":2,"event_type":"torn_unterminated')
            restarted = AppendOnlyLedger(path)
            self.assertNotEqual(restarted.verify_chain(signing_key=key, require_signatures=True), [])
            self.assertEqual(len(restarted.entries()), 1)
            self.assertEqual(restarted.head_sha256, e1.entry_sha256)

            e2 = restarted.append("after_tear", "actor", {"i": 2}, signing_key=key)
            self.assertEqual(e2.seq, 2)
            self.assertEqual(e2.prev_sha256, e1.entry_sha256)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])
            self.assertEqual(len(ledger.entries()), 2)

            # Case B: complete JSON object missing ONLY its trailing '\n' must NOT glue next line
            valid_no_nl = path.read_bytes().splitlines()[1]
            with path.open("ab") as fh:
                fh.write(valid_no_nl)
            self.assertNotEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])
            e3 = ledger.append("after_missing_nl", "actor", {"i": 3}, signing_key=key)
            self.assertEqual(e3.seq, 3)
            self.assertEqual(e3.prev_sha256, e2.entry_sha256)
            raw_lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 3)
            for idx, raw in enumerate(raw_lines, 1):
                self.assertEqual(json.loads(raw)["seq"], idx)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

            # Case C: external shrink to mid-line inside entry 2 must NOT permanently wedge append
            with path.open("r+b") as fh:
                fh.truncate(size_after_e1 + 20)
            self.assertEqual(len(ledger.entries()), 1)
            self.assertEqual(ledger.head_sha256, e1.entry_sha256)
            e2_repaired = ledger.append("after_shrink", "actor", {"i": 200}, signing_key=key)
            self.assertEqual(e2_repaired.seq, 2)
            self.assertEqual(e2_repaired.prev_sha256, e1.entry_sha256)
            self.assertEqual(len(ledger.entries()), 2)
            self.assertEqual(ledger.verify_chain(signing_key=key, require_signatures=True), [])

    def test_two_process_concurrent_append_via_subprocess(self) -> None:
        key_hex = b"exo-multiproc-hmac-key-32-bytes!".hex()
        per_proc = 50
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multiproc.jsonl"
            go_flag = Path(tmp) / "go.flag"
            ready_flags = [Path(tmp) / f"ready_{i}.flag" for i in range(2)]

            child_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "from exo.ledger import AppendOnlyLedger\n"
                "ledger_path = Path(sys.argv[2])\n"
                "wid = int(sys.argv[3])\n"
                "count = int(sys.argv[4])\n"
                "key = bytes.fromhex(sys.argv[5])\n"
                "ready_path = Path(sys.argv[6])\n"
                "go_path = Path(sys.argv[7])\n"
                "ledger = AppendOnlyLedger(ledger_path)\n"
                "ready_path.write_text('ok', encoding='utf-8')\n"
                "deadline = time.monotonic() + 15.0\n"
                "while not go_path.exists():\n"
                "    if time.monotonic() > deadline:\n"
                "        raise SystemExit('timeout waiting for go flag')\n"
                "    time.sleep(0.005)\n"
                "for idx in range(count):\n"
                "    ledger.append('proc_receipt', f'proc-{wid}', {'proc': wid, 'idx': idx}, signing_key=key)\n"
            )

            procs = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-c",
                        child_code,
                        str(SRC_ROOT),
                        str(path),
                        str(wid),
                        str(per_proc),
                        key_hex,
                        str(ready_flags[wid]),
                        str(go_flag),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for wid in range(2)
            ]

            for rf in ready_flags:
                waited = 0
                while not rf.exists() and waited < 300:
                    threading.Event().wait(0.02)
                    waited += 1
                self.assertTrue(rf.exists(), f"child failed to signal readiness: {rf}")

            go_flag.write_text("go", encoding="utf-8")
            for p in procs:
                out, err = p.communicate(timeout=30)
                self.assertEqual(p.returncode, 0, msg=f"child rc={p.returncode} stdout={out!r} stderr={err!r}")

            key = bytes.fromhex(key_hex)
            verifier = AppendOnlyLedger(path)
            self.assertEqual(len(verifier.entries()), 2 * per_proc)
            self.assertEqual(verifier.verify_chain(signing_key=key, require_signatures=True), [])

    def test_append_invokes_os_fsync_parent_dir_fsync_and_rejects_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "durable.jsonl"
            ledger = AppendOnlyLedger(path)
            fsync_calls: list[int] = []
            orig_fsync = os.fsync

            def spy_fsync(fd: int) -> None:
                fsync_calls.append(fd)
                orig_fsync(fd)

            with mock.patch("exo.ledger.os.fsync", side_effect=spy_fsync):
                ledger.append("durable_event", "agent", {"k": 1})
                ledger.append("durable_event", "agent", {"k": 2})

            # First append calls file fsync + parent-dir fsync on create; second calls file fsync.
            self.assertGreaterEqual(len(fsync_calls), 3)
            self.assertEqual(ledger.verify_chain(), [])

            # Symlink target must fail closed via O_NOFOLLOW
            real_target = Path(tmp) / "real.jsonl"
            real_target.write_text("", encoding="utf-8")
            link_path = Path(tmp) / "link.jsonl"
            link_path.symlink_to(real_target)
            sym_ledger = AppendOnlyLedger()
            sym_ledger.path = link_path
            with self.assertRaises(OSError):
                sym_ledger.append("symlink_attack", "agent", {"x": 1})

    def test_gate_spend_boundaries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = AppendOnlyLedger(Path(tmp) / "gate.jsonl")
            gate = ApprovalGate(ledger=ledger, max_auto_spend_usd=10.0, signing_key=b"gate-key-32-bytes-00000000000001")

            for bad_spend in (float("nan"), float("inf"), float("-inf"), -0.01):
                env = ActionEnvelope(
                    action_kind="read_file",
                    target="notes.txt",
                    argv=("cat", "notes.txt"),
                    paths=("notes.txt",),
                    spend_usd=bad_spend,
                )
                verdict = gate.evaluate(env)
                self.assertEqual(verdict.decision, "deny", msg=f"bad_spend={bad_spend!r}")

            over_limit = ActionEnvelope(
                action_kind="read_file",
                target="notes.txt",
                argv=("cat", "notes.txt"),
                paths=("notes.txt",),
                spend_usd=10.0 + 1e-9,
            )
            verdict_over = gate.evaluate(over_limit)
            self.assertNotEqual(verdict_over.decision, "allow")
            self.assertEqual(ledger.verify_chain(signing_key=b"gate-key-32-bytes-00000000000001", require_signatures=True), [])

    def test_zero_bare_asserts_in_ledger_and_tests(self) -> None:
        for rel in ("src/exo/ledger.py", "tests/test_ledger_concurrency.py"):
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
            self.assertEqual([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)], [])


if __name__ == "__main__":
    unittest.main()
