"""Zero-dependency pure-stdlib vertical-stack helper tests (Closes siliconcordat/exo#5).

Zero bare ``assert`` statements; runs cleanly under ``-B``, ``-O -B``, ``-B -W error``,
and isolated ``-I`` mode with ``numpy`` explicitly blocked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEMO_PATH = _REPO_ROOT / "examples" / "vertical_splice_demo.py"
_SPEC = importlib.util.spec_from_file_location("_exo_vertical_splice_demo_stdlib", str(_DEMO_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load vertical_splice_demo from {_DEMO_PATH}")
_DEMO = importlib.util.module_from_spec(_SPEC)
sys.modules["_exo_vertical_splice_demo_stdlib"] = _DEMO
_SPEC.loader.exec_module(_DEMO)


class TestVerticalStackZeroNumpy(unittest.TestCase):
    def test_repo_contains_zero_numpy_imports(self) -> None:
        """AST audit across src/, examples/, and tests/ confirms exo never imports numpy."""
        offenders: list[str] = []
        for sub in ("src", "examples", "tests"):
            for py_file in sorted((_REPO_ROOT / sub).rglob("*.py")):
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "numpy" or alias.name.startswith("numpy."):
                                offenders.append(f"{py_file.relative_to(_REPO_ROOT)}:{node.lineno}")
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        if mod == "numpy" or mod.startswith("numpy."):
                            offenders.append(f"{py_file.relative_to(_REPO_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], f"Unexpected numpy import(s) in exo: {offenders}")

    def test_exo_and_vertical_demo_import_with_numpy_blocked_in_fresh_subprocess(self) -> None:
        """Issue #5 reproducer: fresh Python process with numpy blocked imports exo & demo cleanly."""
        probe_code = (
            "import importlib.abc, importlib.util, pathlib, sys\n"
            "class _BlockNumpy(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path, target=None):\n"
            "        if fullname == 'numpy' or fullname.startswith('numpy.'):\n"
            "            raise ModuleNotFoundError(\"No module named 'numpy'\", name='numpy')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _BlockNumpy())\n"
            f"sys.path.insert(0, {str(_SRC_DIR)!r})\n"
            "import exo, exo.bridge, exo.capabilities, exo.cli, exo.client, exo.gate, exo.ledger, exo.policy, exo.router, exo.sim\n"
            f"spec = importlib.util.spec_from_file_location('_demo', {str(_DEMO_PATH)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['_demo'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "if 'numpy' in sys.modules:\n"
            "    raise SystemExit('numpy unexpectedly imported')\n"
            "print('ZERO_NUMPY_IMPORT_OK')\n"
        )
        res = subprocess.run(
            [sys.executable, "-I", "-B", "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=f"stderr={res.stderr}\nstdout={res.stdout}")
        self.assertIn("ZERO_NUMPY_IMPORT_OK", res.stdout)

    def test_pure_stdlib_matrix_helpers_and_validation_guards(self) -> None:
        """Pure-stdlib _Matrix2D, linear algebra, and strict boundary guards work without numpy."""
        m = _DEMO._Matrix2D([[1, 2.5], [3.0, -4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m.size, 4)
        self.assertEqual(m.ndim, 2)
        m_copy = m.copy()
        m_copy[0][0] = 99.0
        self.assertEqual(m[0][0], 1.0)
        self.assertEqual(m_copy[0][0], 99.0)

        # Float32 quantization & packed bytes match struct '<f'
        m_f32 = _DEMO._Matrix2D([[0.2, 1.5]]).to_f32()
        self.assertEqual(m_f32[0][0], struct.unpack("<f", struct.pack("<f", 0.2))[0])
        self.assertEqual(m_f32[0][1], 1.5)
        raw_bytes = m_f32.to_f32_bytes()
        self.assertEqual(len(raw_bytes), 8)
        self.assertEqual(struct.unpack("<2f", raw_bytes), (m_f32[0][0], 1.5))

        # Deterministic RNG, ones, matmul, add, scale, Frobenius norm
        r1 = _DEMO._rand_normal_matrix(random.Random(42), 3, 4, scale=0.25)
        r2 = _DEMO._rand_normal_matrix(random.Random(42), 3, 4, scale=0.25)
        self.assertEqual(r1, r2)
        self.assertEqual(r1.shape, (3, 4))

        ones = _DEMO._ones_matrix(2, 3)
        self.assertEqual(ones, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        a = _DEMO._Matrix2D([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = _DEMO._Matrix2D([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        prod = _DEMO._matmul_2d(a, b)
        self.assertEqual(prod, [[58.0, 64.0], [139.0, 154.0]])

        added = _DEMO._add_2d(prod, _DEMO._Matrix2D([[2.0, -4.0], [1.0, 6.0]]))
        self.assertEqual(added, [[60.0, 60.0], [140.0, 160.0]])
        scaled = _DEMO._scale_2d(_DEMO._Matrix2D([[3.0, 4.0]]), 0.5)
        self.assertEqual(scaled, [[1.5, 2.0]])
        self.assertAlmostEqual(_DEMO._frobenius_norm_2d(_DEMO._Matrix2D([[3.0, 4.0]])), 5.0, places=12)

        # Validation guards (TypeError & ValueError)
        for bad_num in (True, False, "1.0", None):
            with self.assertRaises(TypeError):
                _DEMO._Matrix2D([[bad_num]])
            with self.assertRaises(TypeError):
                _DEMO._scale_2d(a, bad_num)  # type: ignore[arg-type]
        for bad_finite in (float("nan"), float("inf"), float("-inf"), 10**400):
            with self.assertRaises(ValueError):
                _DEMO._Matrix2D([[bad_finite]])
            with self.assertRaises(ValueError):
                _DEMO._scale_2d(a, bad_finite)
        for bad_dim_type in (True, 2.0, "2"):
            with self.assertRaises(TypeError):
                _DEMO._ones_matrix(bad_dim_type, 2)  # type: ignore[arg-type]
        for bad_dim_val in (0, -1):
            with self.assertRaises(ValueError) as dim_ctx1:
                _DEMO._ones_matrix(bad_dim_val, 2)
            self.assertIn("must be positive", str(dim_ctx1.exception))
            with self.assertRaises(ValueError) as dim_ctx2:
                _DEMO._rand_normal_matrix(random.Random(1), 2, bad_dim_val)
            self.assertIn("must be positive", str(dim_ctx2.exception))
        for bad_rows in ([], "ab", b"ab", 123, [[]], ["ab"], [[1.0], [1.0, 2.0]]):
            with self.assertRaises(ValueError):
                _DEMO._Matrix2D(bad_rows)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _DEMO._matmul_2d(_DEMO._Matrix2D([[1.0, 2.0]]), _DEMO._Matrix2D([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            _DEMO._add_2d(_DEMO._Matrix2D([[1.0, 2.0]]), _DEMO._Matrix2D([[1.0], [2.0]]))

    def test_synthetic_safetensors_checkpoint_and_compile_adapter_without_numpy(self) -> None:
        """Safetensors writer, synthetic checkpoint, and _compile_domain_adapter run with numpy blocked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng_base = random.Random(20261119)
            w_old = _DEMO._rand_normal_matrix(rng_base, 16, 32, scale=0.2).to_f32()
            ckpt_dir = _DEMO._create_synthetic_checkpoint(root / "ckpt_old", w_old, tag="W_old")

            # Verify valid Safetensors binary framing & exact float32 payload
            st_bytes = (ckpt_dir / "model.safetensors").read_bytes()
            hdr_len = struct.unpack("<Q", st_bytes[:8])[0]
            header = json.loads(st_bytes[8 : 8 + hdr_len].decode("utf-8"))
            self.assertEqual(sorted(header.keys()), ["blk.0.ffn_down.weight", "embed_tokens.weight"])
            self.assertEqual(header["blk.0.ffn_down.weight"]["dtype"], "F32")
            self.assertEqual(header["blk.0.ffn_down.weight"]["shape"], [16, 32])
            self.assertEqual(header["embed_tokens.weight"]["shape"], [8, 32])
            data_section = st_bytes[8 + hdr_len :]
            s0, e0 = header["blk.0.ffn_down.weight"]["data_offsets"]
            self.assertEqual(e0 - s0, 16 * 32 * 4)
            unpacked_first = struct.unpack("<f", data_section[s0 : s0 + 4])[0]
            self.assertEqual(unpacked_first, w_old[0][0])
            s1, e1 = header["embed_tokens.weight"]["data_offsets"]
            self.assertEqual(e1 - s1, 8 * 32 * 4)
            self.assertEqual(struct.unpack("<f", data_section[s1 : s1 + 4])[0], 1.0)
            cfg_json = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg_json, {"hidden_size": 32, "checkpoint_tag": "W_old"})

            # Exercise _compile_domain_adapter with pure-stdlib stub weft modules and numpy blocked
            plane1_root = root / "plane1"
            (plane1_root / "principles").mkdir(parents=True)
            (plane1_root / "principles" / "writing.md").write_text("# Writing\n", encoding="utf-8")

            @dataclass(frozen=True)
            class _FakeCert:
                factor_digest: str
                def verify_division_free(self, policy: Any) -> tuple[bool, list[str]]:
                    return True, []
                def sha256(self) -> str:
                    return hashlib.sha256(self.factor_digest.encode("ascii")).hexdigest()

            @dataclass(frozen=True)
            class _FakePatch:
                factor_b: Any
                factor_a: Any
                factor_digest: str
                certificate: _FakeCert

            class _FakeProjector:
                def __init__(self, base_weight: Any, cfg: Any) -> None:
                    self.base_weight = base_weight
                    self.cfg = cfg
                    self.k0 = None
                    self.safety: list[tuple[Any, Any]] = []
                def initialize_anchor_covariance(self, k0: Any) -> None:
                    self.k0 = k0
                def register_safety_direction(self, b_dir: Any, a_dir: Any) -> None:
                    self.safety.append((b_dir, a_dir))
                def stage_edit(self, edit_keys: Any, target_values: Any, **kwargs: Any) -> _FakePatch:
                    # Confirm pure-stdlib _Matrix2D carries .shape, .size, .ndim for weft compatibility
                    if edit_keys.shape != (32, 2) or target_values.shape != (16, 2):
                        raise AssertionError("unexpected edit/target shapes")
                    pres = kwargs["preservation_keys"]
                    if pres.size != 64:
                        raise AssertionError("unexpected preservation_keys size")
                    digest = hashlib.sha256(self.base_weight.to_f32_bytes()).hexdigest()
                    return _FakePatch(
                        factor_b=target_values,
                        factor_a=edit_keys,
                        factor_digest=digest,
                        certificate=_FakeCert(digest),
                    )

            def _fake_write_gguf(out_path: Path, **kwargs: Any) -> tuple[Path, str]:
                payload = json.dumps(
                    {"arch": kwargs["architecture"], "meta": kwargs["extra_metadata"]},
                    sort_keys=True,
                ).encode("utf-8")
                out_path.write_bytes(payload)
                return out_path, hashlib.sha256(payload).hexdigest()

            @dataclass(frozen=True)
            class _FakeParsedGguf:
                tensors: dict[str, Any]

            @dataclass(frozen=True)
            class _FakeBindRes:
                ok: bool
                problems: list[str]

            fake_mods: dict[str, Any] = {
                "ADAPTER_CLASS_1": "class1_rederivable",
                "BetaEditConfig": lambda **kw: kw,
                "BetaEditNullspaceProjector": _FakeProjector,
                "GatePolicy": lambda **kw: kw,
                "verify_rederivation": lambda staged, **kw: (True, {}),
                "write_gguf_lora": _fake_write_gguf,
                "read_gguf_lora": lambda p: _FakeParsedGguf(
                    tensors={"blk.0.ffn_down.lora_a": True, "blk.0.ffn_down.lora_b": True}
                ),
                "WeftRuntimeFlags": lambda **kw: kw,
                "build_binding_manifest": lambda **kw: kw,
                "verify_binding": lambda manifest, **kw: _FakeBindRes(True, []),
            }
            with mock.patch.dict(sys.modules, {"numpy": None}):
                compiled = _DEMO._compile_domain_adapter(
                    fake_mods,
                    domain="corpus",
                    adapter_id="mde:corpus",
                    base_weight=w_old,
                    ckpt_dir=ckpt_dir,
                    plane1_root=plane1_root,
                    plane1_relpath="principles/writing.md",
                    gguf_out_path=root / "corpus.gguf",
                    seed=101,
                )
            self.assertEqual(compiled["domain"], "corpus")
            self.assertTrue(compiled["rederived_ok"])
            self.assertEqual(len(compiled["gguf_sha256"]), 64)

    def test_load_sibling_modules_attributes_missing_weft_dependency_honestly(self) -> None:
        """If sibling weft itself raises ModuleNotFoundError('numpy'), error blames weft, not exo."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weft_pkg = root / "weft_src" / "weft"
            weft_pkg.mkdir(parents=True)
            (weft_pkg / "__init__.py").write_text("", encoding="utf-8")
            (weft_pkg / "binding.py").write_text(
                "raise ModuleNotFoundError(\"No module named 'numpy'\", name='numpy')\n",
                encoding="utf-8",
            )
            warp_src = root / "warp_src"
            warp_src.mkdir()
            env = _DEMO.SiblingEnvironment(
                mde_bin=root / "mde",
                warp_src=warp_src,
                warp_root=root,
                weft_src=root / "weft_src",
            )
            orig_path = list(sys.path)
            clean_path = [
                p
                for p in orig_path
                if not (Path(p or ".") / "weft").exists() and not (Path(p or ".") / "weft.py").exists()
            ]
            saved_weft = {k: v for k, v in sys.modules.items() if k == "weft" or k.startswith("weft.")}
            try:
                for k in list(saved_weft):
                    sys.modules.pop(k, None)
                sys.path[:] = [str(root / "weft_src"), *clean_path]
                with self.assertRaises(ModuleNotFoundError) as ctx:
                    _DEMO._load_sibling_modules(env)
                self.assertEqual(ctx.exception.name, "numpy")
                self.assertIn("Sibling package 'weft' requires third-party dependency 'numpy'", str(ctx.exception))

                # Missing weft package itself re-raises directly without misattribution
                for k in [k for k in list(sys.modules) if k == "weft" or k.startswith("weft.")]:
                    sys.modules.pop(k, None)
                sys.path[:] = list(clean_path)
                missing_env = _DEMO.SiblingEnvironment(
                    mde_bin=root / "mde",
                    warp_src=warp_src,
                    warp_root=root,
                    weft_src=root / "no_weft_here",
                )
                with self.assertRaises(ModuleNotFoundError) as miss_ctx:
                    _DEMO._load_sibling_modules(missing_env)
                self.assertEqual(miss_ctx.exception.name, "weft")
                self.assertNotIn("third-party dependency", str(miss_ctx.exception))
            finally:
                for k in [k for k in list(sys.modules) if k == "weft" or k.startswith("weft.")]:
                    sys.modules.pop(k, None)
                sys.modules.update(saved_weft)
                sys.path[:] = orig_path


if __name__ == "__main__":
    unittest.main()
