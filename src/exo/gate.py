"""Exact-byte human approval gate immune to shell-text and path-quoting bypasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hmac
import math
import posixpath
import re
from typing import Any, Mapping, Optional, Sequence

from exo.ledger import AppendOnlyLedger, canonical_json_bytes, hmac_sha256_hex, sha256_hex, utc_now_iso

# Wrapper flags that execute an embedded shell command string and must always escalate.
WRAPPER_SHELL_EXEC_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"-S", "--split-string"}),
    "flock": frozenset({"-c", "--command"}),
    "script": frozenset({"-c", "--command"}),
}

HIGH_RISK_ACTION_KINDS = frozenset({
    "delete_file",
    "external_post",
    "production_deploy",
    "spend",
    "credential_change",
    "force_push",
    "model_change",
    "policy_change",
    "shell_exec",
})

HIGH_RISK_EXECUTABLES = frozenset({
    "sh",
    "bash",
    "zsh",
    "dash",
    "fish",
    "csh",
    "ksh",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "rm",
    "rmdir",
    "unlink",
    "dd",
    "mkfs",
    "chmod",
    "chown",
    "curl",
    "wget",
    "nc",
    "ncat",
    "socat",
    "eval",
    "sudo",
    "su",
    "doas",
    "pkexec",
    "systemctl",
    "launchctl",
    "mount",
    "umount",
    "docker",
    "podman",
    "kubectl",
    "vercel",
    "wrangler",
})

# Wrapper commands whose flags may consume a following argument token.
# Maps wrapper name -> (short_value_flags, long_value_flags, positional_prefix_count)
WRAPPER_SPECS: dict[str, tuple[frozenset[str], frozenset[str], int]] = {
    "sudo": (
        frozenset({"u", "g", "C", "h", "p", "r", "t", "T", "U", "D", "R"}),
        frozenset({
            "--user",
            "--group",
            "--close-from",
            "--host",
            "--prompt",
            "--role",
            "--type",
            "--command-timeout",
            "--other-user",
            "--chdir",
            "--chroot",
        }),
        0,
    ),
    "env": (
        frozenset({"u", "C", "S"}),
        frozenset({"--unset", "--chdir", "--split-string"}),
        0,
    ),
    "nice": (
        frozenset({"n"}),
        frozenset({"--adjustment", "--adj"}),
        0,
    ),
    "timeout": (
        frozenset({"k", "s"}),
        frozenset({"--kill-after", "--signal", "--sig"}),
        1,  # positional DURATION token before inner command
    ),
    "stdbuf": (
        frozenset({"i", "o", "e"}),
        frozenset({"--input", "--output", "--error"}),
        0,
    ),
    "ionice": (
        frozenset({"c", "n", "p", "P", "u"}),
        frozenset({"--class", "--classdata", "--pid", "--pgid", "--uid"}),
        0,
    ),
    "watch": (
        frozenset({"n", "d"}),
        frozenset({"--interval", "--int", "--differences"}),
        0,
    ),
    "exec": (
        frozenset({"a"}),
        frozenset(),
        0,
    ),
    "flock": (
        frozenset({"w", "E", "c"}),
        frozenset({"--timeout", "--wait", "--conflict-exit-code", "--command"}),
        1,  # lockfile path
    ),
    "chroot": (
        frozenset(),
        frozenset({"--userspec", "--groups"}),
        1,  # newroot path
    ),
    "unshare": (
        frozenset({"R", "w", "S", "G"}),
        frozenset({"--root", "--wd", "--setuid", "--setgid", "--propagation"}),
        0,
    ),
    "nsenter": (
        frozenset({"t", "m", "u", "i", "n", "p", "C", "U", "S", "G", "r", "w"}),
        frozenset({"--target", "--mount", "--uts", "--ipc", "--net", "--pid", "--cgroup", "--user", "--root", "--wd"}),
        0,
    ),
    "systemd-run": (
        frozenset({"p", "E", "u", "G", "t"}),
        frozenset({"--property", "--setenv", "--unit", "--uid", "--gid", "--slice", "--working-directory"}),
        0,
    ),
    "xvfb-run": (
        frozenset({"n", "f", "e", "s", "p", "w"}),
        frozenset({"--server-num", "--auth-file", "--error-file", "--server-args", "--wait"}),
        0,
    ),
    "script": (
        frozenset({"c", "t", "O", "I", "B", "T"}),
        frozenset({"--command", "--timing", "--log-out", "--log-in", "--log-io", "--log-timing"}),
        0,
    ),
    "xargs": (
        frozenset({"a", "d", "E", "I", "L", "n", "P", "s"}),
        frozenset({"--arg-file", "--delimiter", "--eof", "--replace", "--max-lines", "--max-args", "--max-procs", "--max-chars"}),
        0,
    ),
}

# Known boolean (non-value) flags on wrappers so unknown flags can fail closed.
WRAPPER_BOOL_FLAGS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "sudo": (frozenset({"A", "b", "E", "H", "i", "k", "K", "n", "P", "S", "v"}), frozenset({"--non-interactive", "--preserve-env", "--set-home", "--login", "--stdin"})),
    "env": (frozenset({"i", "0", "v"}), frozenset({"--ignore-environment", "--null", "--debug"})),
    "nice": (frozenset(), frozenset()),
    "timeout": (frozenset({"v"}), frozenset({"--foreground", "--preserve-status", "--verbose"})),
    "stdbuf": (frozenset(), frozenset()),
    "ionice": (frozenset({"t"}), frozenset({"--ignore"})),
    "watch": (frozenset({"b", "c", "e", "g", "p", "q", "r", "t", "w", "x"}), frozenset({"--beep", "--color", "--errexit", "--chgexit", "--precise", "--no-title", "--exec"})),
    "exec": (frozenset({"c", "l"}), frozenset()),
    "flock": (frozenset({"s", "x", "u", "n", "o", "F"}), frozenset({"--shared", "--exclusive", "--unlock", "--nonblock", "--close", "--no-fork"})),
    "chroot": (frozenset(), frozenset({"--skip-chdir"})),
    "unshare": (frozenset({"m", "u", "i", "n", "p", "U", "C", "f", "r"}), frozenset({"--mount", "--uts", "--ipc", "--net", "--pid", "--user", "--cgroup", "--fork", "--map-root-user"})),
    "nsenter": (frozenset({"F", "a"}), frozenset({"--no-fork", "--all"})),
    "systemd-run": (frozenset({"P", "q", "G"}), frozenset({"--user", "--system", "--scope", "--pipe", "--pty", "--wait", "--collect", "--quiet"})),
    "xvfb-run": (frozenset({"a", "d"}), frozenset({"--auto-servernum", "--auto-display"})),
    "script": (frozenset({"a", "e", "f", "q", "r"}), frozenset({"--append", "--return", "--flush", "--quiet"})),
    "xargs": (frozenset({"0", "o", "p", "r", "t", "x"}), frozenset({"--null", "--open-tty", "--interactive", "--no-run-if-empty", "--verbose", "--exit"})),
}

DEFAULT_PROTECTED_PATH_PATTERNS = (
    r"(^|/)auth(/|$|\.)",
    r"(^|/)credentials?(/|$|\.)",
    r"(^|/)secrets?(/|$|\.)",
    r"(^|/)\.env(\.|$)",
    r"(^|/)schemas?(/|$|\.)",
    r"\.schema\.json$",
    r"(^|/)tests?(/|$)",
    r"(^|/)oracles?(/|$)",
    r"(^|/)anchors?(/|$)",
    r"(^|/)SHA256SUMS$",
    r"(^|/)bar\.toml$",
    r"(^|/)repo-policy\.json$",
    r"(^|/)gate(/|$)",
    r"(^|/)exo_gate(/|$)",
    r"(^|/)exo/__init__\.py$",
    r"(^|/)exo/store\.py$",
    r"(^|/)catalog\.toml$",
    r"(^|/)public_api(/|$|\.)",
)

SHELL_META_RE = re.compile(r"(\||;|&&|\|\||`|\$\(|[\r\n]|[<>])")
MULTILINE_PIPE_RE = re.compile(r"\|\s*[\r\n]+\s*\S+")


def decode_git_path(raw_path: str) -> str:
    """Decode Git C-style quoted paths (including octal UTF-8 escapes) cleanly.

    Fixes the exo#1162 hazard where non-ASCII paths such as
    ``"plugins/exo-gate/exo_gate/\\303\\251.py"`` bypassed literal path checks.
    """
    if len(raw_path) >= 2 and raw_path[0] == '"' and raw_path[-1] == '"':
        inner = raw_path[1:-1]
        out = bytearray()
        i = 0
        n = len(inner)
        while i < n:
            ch = inner[i]
            if ch != "\\":
                out.extend(ch.encode("utf-8"))
                i += 1
                continue
            i += 1
            if i >= n:
                raise ValueError(f"truncated escape sequence in git path: {raw_path!r}")
            esc = inner[i]
            if esc in "01234567":
                oct_digits = inner[i : i + 3]
                if len(oct_digits) != 3 or any(c not in "01234567" for c in oct_digits):
                    raise ValueError(f"invalid octal escape in git path: {raw_path!r}")
                out.append(int(oct_digits, 8))
                i += 3
            elif esc == '"':
                out.append(ord('"'))
                i += 1
            elif esc == "\\":
                out.append(ord("\\"))
                i += 1
            elif esc == "n":
                out.append(ord("\n"))
                i += 1
            elif esc == "r":
                out.append(ord("\r"))
                i += 1
            elif esc == "t":
                out.append(ord("\t"))
                i += 1
            else:
                out.extend(esc.encode("utf-8"))
                i += 1
        return out.decode("utf-8")
    return raw_path


def normalize_repo_path(raw_path: str) -> str:
    """Decode and normalize a repository-relative path, rejecting traversal escapes."""
    decoded = decode_git_path(raw_path).strip()
    if not decoded or "\x00" in decoded:
        raise ValueError("path must be non-empty without NUL bytes")
    if decoded.startswith("/") or decoded.startswith("\\"):
        raise ValueError(f"absolute path not allowed: {raw_path!r}")
    unified = decoded.replace("\\", "/")
    parts = [p for p in unified.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"path escapes repository root: {raw_path!r}")
    norm = posixpath.normpath("/".join(parts))
    if norm in (".", "..") or norm.startswith("../"):
        raise ValueError(f"path escapes repository root: {raw_path!r}")
    return norm


def unwrap_command_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Inspect structured argv through prefix/wrapper layers.

    Returns (unwrapped_executables, risk_reasons). Fails closed on any unknown
    wrapper flag or unparseable wrapper structure (fixing exo#1141).
    """
    if not argv:
        return [], []
    reasons: list[str] = []
    executables: list[str] = []
    idx = 0
    n = len(argv)

    while idx < n:
        prog = posixpath.basename(argv[idx])
        executables.append(prog)

        # Special-case `uv run ...` wrapper
        if prog == "uv" and idx + 1 < n and argv[idx + 1] == "run":
            idx += 2
            while idx < n and argv[idx].startswith("-"):
                tok = argv[idx]
                if tok == "--":
                    idx += 1
                    break
                if "=" in tok:
                    idx += 1
                elif tok in ("--with", "-w", "--python", "-p", "--directory", "--project"):
                    if idx + 1 >= n:
                        reasons.append("truncated_wrapper_option:uv")
                        return executables, reasons
                    idx += 2
                elif tok in ("--isolated", "--no-project", "--frozen", "--locked", "--offline", "-q", "-v"):
                    idx += 1
                else:
                    reasons.append(f"unknown_wrapper_option:uv:{tok}")
                    return executables, reasons
            continue

        if prog not in WRAPPER_SPECS:
            # Check for `git` config-injection, exec options, and destructive subcommands
            if prog == "git":
                g_idx = idx + 1
                while g_idx < n and argv[g_idx].startswith("-"):
                    gtok = argv[g_idx]
                    if gtok == "--":
                        g_idx += 1
                        break
                    if gtok in ("-c", "--config-env"):
                        reasons.append(f"git_config_injection:{gtok}")
                        g_idx += 2
                    elif gtok.startswith("--config-env=") or (gtok.startswith("-c") and len(gtok) > 2):
                        reasons.append(f"git_config_injection:{gtok.split('=', 1)[0]}")
                        g_idx += 1
                    elif gtok in ("--exec-path", "--upload-pack", "--receive-pack"):
                        reasons.append(f"git_exec_option:{gtok}")
                        g_idx += 2
                    elif gtok.startswith(("--exec-path=", "--upload-pack=", "--receive-pack=")):
                        reasons.append(f"git_exec_option:{gtok.split('=', 1)[0]}")
                        g_idx += 1
                    elif gtok in ("-C", "--git-dir", "--work-tree", "--namespace"):
                        g_idx += 2
                    else:
                        g_idx += 1
                if g_idx < n:
                    sub = argv[g_idx]
                    rest = argv[g_idx + 1 :]
                    for a in rest:
                        if a in ("--exec", "--receive-pack", "--upload-pack") or a.startswith(
                            ("--exec=", "--receive-pack=", "--upload-pack=")
                        ):
                            reasons.append(f"git_exec_option:{a.split('=', 1)[0]}")
                    if sub == "push":
                        for a in rest:
                            if a in ("-f", "--force", "--force-with-lease", "--force-if-includes"):
                                reasons.append("git_force_push")
                            elif a in ("-d", "--delete"):
                                reasons.append("git_destructive:push_delete")
                            elif not a.startswith("-") and a.startswith("+"):
                                reasons.append(f"git_force_push_refspec:{a}")
                            elif not a.startswith("-") and a.startswith(":"):
                                reasons.append(f"git_delete_refspec:{a}")
                    elif sub in ("reset", "clean") and any(a in ("--hard", "-fd", "-fdx", "-f") for a in rest):
                        reasons.append(f"git_destructive:{sub}")
            # Check for `python` / `python3` inline code execution even after flags (`python3 -u -c ...`)
            if prog in ("python", "python3"):
                p_idx = idx + 1
                while p_idx < n and argv[p_idx].startswith("-"):
                    ptok = argv[p_idx]
                    if ptok == "--":
                        break
                    if ptok == "-c" or (ptok.startswith("-") and not ptok.startswith("--") and "c" in ptok[1:]):
                        reasons.append("inline_code_execution:python")
                        break
                    if ptok in ("-m", "-W", "-X"):
                        p_idx += 2
                    else:
                        p_idx += 1
            if prog in HIGH_RISK_EXECUTABLES:
                reasons.append(f"high_risk_executable:{prog}")
            break

        if prog in HIGH_RISK_EXECUTABLES:
            reasons.append(f"high_risk_executable:{prog}")

        short_val, long_val, pos_count = WRAPPER_SPECS[prog]
        short_bool, long_bool = WRAPPER_BOOL_FLAGS[prog]
        shell_exec_flags = WRAPPER_SHELL_EXEC_FLAGS.get(prog, frozenset())
        idx += 1
        while idx < n:
            tok = argv[idx]
            if tok == "--":
                idx += 1
                break
            if prog == "env" and not tok.startswith("-") and "=" in tok:
                idx += 1
                continue
            if not tok.startswith("-") or tok == "-":
                break
            if tok.startswith("--"):
                flag_name = tok.split("=", 1)[0]
                if flag_name in shell_exec_flags:
                    reasons.append(f"wrapper_shell_exec_flag:{prog}:{flag_name}")
                if "=" in tok:
                    if flag_name in long_val:
                        idx += 1
                        continue
                    reasons.append(f"unknown_wrapper_option:{prog}:{tok}")
                    return executables, reasons
                if tok in long_val:
                    if idx + 1 >= n:
                        reasons.append(f"truncated_wrapper_option:{prog}:{tok}")
                        return executables, reasons
                    idx += 2
                    continue
                if tok in long_bool:
                    idx += 1
                    continue
                reasons.append(f"unknown_wrapper_option:{prog}:{tok}")
                return executables, reasons
            # Short option cluster, e.g., `-iu VAR` or `-Ep prompt` or `-uVAR`
            cluster = tok[1:]
            consumed_next = False
            pos_in_cluster = 0
            while pos_in_cluster < len(cluster):
                ch = cluster[pos_in_cluster]
                if f"-{ch}" in shell_exec_flags:
                    reasons.append(f"wrapper_shell_exec_flag:{prog}:-{ch}")
                if ch in short_val:
                    if pos_in_cluster + 1 < len(cluster):
                        # Remainder of token is the attached option value
                        break
                    if idx + 1 >= n:
                        reasons.append(f"truncated_wrapper_option:{prog}:-{ch}")
                        return executables, reasons
                    consumed_next = True
                    break
                if ch in short_bool:
                    pos_in_cluster += 1
                    continue
                reasons.append(f"unknown_wrapper_option:{prog}:-{ch}")
                return executables, reasons
            idx += 2 if consumed_next else 1

        # Skip required positional wrapper arguments (e.g., timeout DURATION, flock LOCKFILE)
        if idx + pos_count > n:
            reasons.append(f"truncated_wrapper_positional:{prog}")
            return executables, reasons
        idx += pos_count

    return executables, reasons


@dataclass(frozen=True)
class ActionEnvelope:
    """Typed action request evaluated by the Layer 4 approval gate."""

    action_kind: str
    target: str
    paths: tuple[str, ...] = ()
    argv: tuple[str, ...] = ()
    raw_shell: Optional[str] = None
    payload_bytes: bytes = b""
    read_tokens: tuple[str, ...] = ()
    spend_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_paths(self) -> tuple[str, ...]:
        return tuple(normalize_repo_path(p) for p in self.paths)

    def payload_sha256(self) -> str:
        return sha256_hex(self.payload_bytes)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "action_kind": self.action_kind,
            "target": self.target,
            "paths": list(self.normalized_paths()),
            "argv": list(self.argv),
            "raw_shell": self.raw_shell,
            "payload_sha256": self.payload_sha256(),
            "spend_usd": round(float(self.spend_usd), 6),
            "metadata": self.metadata,
        }

    def challenge_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.canonical_dict()))


@dataclass(frozen=True)
class ApprovalChallenge:
    """Exact-byte challenge presented to a human approver."""

    challenge_sha256: str
    payload_sha256: str
    risk_class: str
    reasons: tuple[str, ...]
    created_at_utc: str
    summary: str


@dataclass(frozen=True)
class HumanApproval:
    """Human sign-off strictly bound to one challenge_sha256 and payload_sha256."""

    approver_id: str
    challenge_sha256: str
    payload_sha256: str
    approved_at_utc: str
    expires_at_utc: Optional[str] = None
    signature: Optional[str] = None


@dataclass(frozen=True)
class GateVerdict:
    """Result of evaluating an ActionEnvelope at the Layer 4 gate."""

    decision: str  # "allow" | "require_human_approval" | "deny"
    risk_class: str  # "routine" | "high_risk"
    reasons: tuple[str, ...]
    challenge: Optional[ApprovalChallenge] = None
    unwrapped_executables: tuple[str, ...] = ()


class ApprovalGate:
    """Deterministic exact-byte human approval gate."""

    def __init__(
        self,
        *,
        protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATH_PATTERNS,
        max_auto_spend_usd: float = 0.0,
        signing_key: Optional[bytes] = None,
        ledger: Optional[AppendOnlyLedger] = None,
    ) -> None:
        self._protected_res = tuple(re.compile(p, re.IGNORECASE) for p in protected_patterns)
        self.max_auto_spend_usd = float(max_auto_spend_usd)
        self.signing_key = signing_key
        self.ledger = ledger

    def is_protected_path(self, raw_path: str) -> bool:
        norm = normalize_repo_path(raw_path)
        return any(rx.search(norm) for rx in self._protected_res)

    def evaluate(
        self,
        envelope: ActionEnvelope,
        *,
        approval: Optional[HumanApproval] = None,
        now_utc: Optional[str] = None,
    ) -> GateVerdict:
        """Evaluate an ActionEnvelope, returning allow, require_human_approval, or deny."""
        reasons: list[str] = []

        # 1. Normalize paths fail-closed (catches path traversal & decodes Git octal paths)
        try:
            norm_paths = envelope.normalized_paths()
        except ValueError as exc:
            verdict = GateVerdict(
                decision="deny",
                risk_class="high_risk",
                reasons=(f"invalid_path:{exc}",),
            )
            self._record("gate_deny", envelope, verdict)
            return verdict

        # 2. Enforce L5 read-before-write precondition on splice_patch
        if envelope.action_kind == "splice_patch" and not envelope.read_tokens:
            verdict = GateVerdict(
                decision="deny",
                risk_class="high_risk",
                reasons=("ERR_PRECONDITION_READ_REQUIRED",),
            )
            self._record("gate_deny", envelope, verdict)
            return verdict

        # 3. Raw shell strings NEVER pass silently (fixes exo#1127 multi-line pipe bypasses)
        if envelope.raw_shell is not None and envelope.raw_shell.strip():
            reasons.append("raw_shell_forbidden_without_approval")
            if MULTILINE_PIPE_RE.search(envelope.raw_shell):
                reasons.append("multiline_pipe_continuation_detected")
            elif SHELL_META_RE.search(envelope.raw_shell):
                reasons.append("shell_metacharacters_detected")

        # 4. Wrapper-aware structured argv analysis (fixes exo#1141 option-value bypasses)
        unwrapped_execs, argv_reasons = unwrap_command_argv(envelope.argv)
        reasons.extend(argv_reasons)

        # 5. Action kind & spend checks (reject negative or non-finite spend fail-closed)
        if not math.isfinite(envelope.spend_usd) or envelope.spend_usd < 0:
            verdict = GateVerdict(
                decision="deny",
                risk_class="high_risk",
                reasons=(f"invalid_spend_usd:{envelope.spend_usd}",),
            )
            self._record("gate_deny", envelope, verdict)
            return verdict
        if envelope.action_kind in HIGH_RISK_ACTION_KINDS:
            reasons.append(f"high_risk_action:{envelope.action_kind}")
        if envelope.spend_usd > self.max_auto_spend_usd:
            reasons.append(f"spend_exceeds_limit:{envelope.spend_usd:.2f}>{self.max_auto_spend_usd:.2f}")

        # 6. Protected path checks (fixes exo#1162 imported & Git-quoted non-ASCII paths)
        for p in norm_paths:
            if any(rx.search(p) for rx in self._protected_res):
                reasons.append(f"protected_path:{p}")

        if not reasons:
            verdict = GateVerdict(
                decision="allow",
                risk_class="routine",
                reasons=(),
                unwrapped_executables=tuple(unwrapped_execs),
            )
            self._record("gate_allow", envelope, verdict)
            return verdict

        challenge = ApprovalChallenge(
            challenge_sha256=envelope.challenge_sha256(),
            payload_sha256=envelope.payload_sha256(),
            risk_class="high_risk",
            reasons=tuple(reasons),
            created_at_utc=now_utc or utc_now_iso(),
            summary=f"{envelope.action_kind} on {envelope.target} ({len(norm_paths)} path(s))",
        )

        if approval is not None:
            ok, err = self.verify_approval(envelope, approval, now_utc=now_utc)
            if ok:
                verdict = GateVerdict(
                    decision="allow",
                    risk_class="high_risk",
                    reasons=tuple(reasons),
                    challenge=challenge,
                    unwrapped_executables=tuple(unwrapped_execs),
                )
                self._record("gate_approved", envelope, verdict, approver=approval.approver_id)
                return verdict
            reasons.append(f"approval_rejected:{err}")
            verdict = GateVerdict(
                decision="deny",
                risk_class="high_risk",
                reasons=tuple(reasons),
                challenge=challenge,
                unwrapped_executables=tuple(unwrapped_execs),
            )
            self._record("gate_approval_rejected", envelope, verdict, approver=approval.approver_id)
            return verdict

        verdict = GateVerdict(
            decision="require_human_approval",
            risk_class="high_risk",
            reasons=tuple(reasons),
            challenge=challenge,
            unwrapped_executables=tuple(unwrapped_execs),
        )
        self._record("gate_challenge", envelope, verdict)
        return verdict

    def sign_approval(
        self,
        challenge: ApprovalChallenge,
        approver_id: str,
        *,
        approved_at_utc: Optional[str] = None,
        expires_at_utc: Optional[str] = None,
    ) -> HumanApproval:
        """Create a HumanApproval strictly pinned to challenge_sha256 and payload_sha256."""
        if not approver_id.strip():
            raise ValueError("approver_id must be non-empty")
        approved_ts = approved_at_utc or utc_now_iso()
        sig: Optional[str] = None
        if self.signing_key is not None:
            msg = canonical_json_bytes({
                "approver_id": approver_id,
                "challenge_sha256": challenge.challenge_sha256,
                "payload_sha256": challenge.payload_sha256,
                "approved_at_utc": approved_ts,
                "expires_at_utc": expires_at_utc,
            })
            sig = hmac_sha256_hex(self.signing_key, msg)
        return HumanApproval(
            approver_id=approver_id,
            challenge_sha256=challenge.challenge_sha256,
            payload_sha256=challenge.payload_sha256,
            approved_at_utc=approved_ts,
            expires_at_utc=expires_at_utc,
            signature=sig,
        )

    def verify_approval(
        self,
        envelope: ActionEnvelope,
        approval: HumanApproval,
        *,
        now_utc: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Verify that a HumanApproval matches the exact live bytes of envelope."""
        if not approval.approver_id or not approval.approver_id.strip():
            return False, "empty_approver"
        live_challenge = envelope.challenge_sha256()
        live_payload = envelope.payload_sha256()
        if not hmac.compare_digest(live_challenge, approval.challenge_sha256):
            return False, "challenge_sha256_mismatch"
        if not hmac.compare_digest(live_payload, approval.payload_sha256):
            return False, "payload_sha256_mismatch"
        if approval.expires_at_utc is not None:
            current = _parse_utc(now_utc or utc_now_iso())
            expiry = _parse_utc(approval.expires_at_utc)
            if current > expiry:
                return False, "approval_expired"
        if self.signing_key is not None:
            msg = canonical_json_bytes({
                "approver_id": approval.approver_id,
                "challenge_sha256": approval.challenge_sha256,
                "payload_sha256": approval.payload_sha256,
                "approved_at_utc": approval.approved_at_utc,
                "expires_at_utc": approval.expires_at_utc,
            })
            expected_sig = hmac_sha256_hex(self.signing_key, msg)
            if not approval.signature or not hmac.compare_digest(expected_sig, approval.signature):
                return False, "invalid_signature"
        return True, None

    def _record(
        self,
        event_type: str,
        envelope: ActionEnvelope,
        verdict: GateVerdict,
        *,
        approver: Optional[str] = None,
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            event_type,
            approver or "exo-gate",
            {
                "action_kind": envelope.action_kind,
                "target": envelope.target,
                "decision": verdict.decision,
                "risk_class": verdict.risk_class,
                "reasons": list(verdict.reasons),
                "challenge_sha256": verdict.challenge.challenge_sha256 if verdict.challenge else None,
            },
            signing_key=self.signing_key,
        )


def _parse_utc(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)
