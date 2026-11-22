"""Fail-closed exact IEEE-754 binary32 rational scale tests (Closes siliconcordat/exo#7).

Zero bare ``assert`` statements; runs cleanly under ``-B``, ``-O -B``, ``-B -W error``,
and isolated ``-I`` mode.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
import unittest

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from exo.bridge import (
    build_warp_adapter_bundle,
    build_warp_bundles_for_route,
    scale_to_rational,
    translate_weft_certificate_to_warp,
)
from exo.policy import siliconcordat_repo_policy
from exo.router import RouteDecision, RouteRequest, ZoneRouter

FLT32_MAX = (2**24 - 1) * (2**104)
FLT32_MIN_SUBNORMAL = 2.0**-149


class _EvilInt(int):
    @property
    def numerator(self) -> int:
        return -5


class _EvilFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        return (-1, 2)


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


class TestBridgeExactFloat32Scale(unittest.TestCase):
    def _assert_f32_scale_invariants(
        self,
        scale: int | float,
        num: int,
        den: int,
    ) -> None:
        self.assertIs(type(num), int)
        self.assertIs(type(den), int)
        self.assertGreater(num, 0)
        self.assertGreater(den, 0)
        self.assertEqual(math.gcd(num, den), 1)
        # Dyadic denominator (exact power of two)
        self.assertEqual(den & (den - 1), 0)
        certified = Fraction(num, den)
        # Never exceeds the requested route scale
        self.assertLessEqual(certified, Fraction(scale))
        # Exact IEEE-754 binary32 round-trip (zero upward overshoot in llama-server)
        served_f64 = num / den
        served_f32 = _f32(served_f64)
        self.assertEqual(Fraction(served_f32), certified)
        # JSON number round-trip through float -> float32 stays exact
        json_roundtrip = _f32(json.loads(json.dumps(served_f64)))
        self.assertEqual(Fraction(json_roundtrip), certified)

    def test_issue7_exact_cases_no_overshoot_and_no_deadband(self) -> None:
        """Issue #7 reproducer: 0.2004, 0.2006, 0.0005, and manifest scales are exact f32 & <= scale."""
        cases: tuple[int | float, ...] = (
            0.2004,
            0.2006,
            0.0005,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.75,
            1.0,
            1,
            2,
            1.5,
            (1 << 24) + 3,  # integer whose nearest f32 rounds up; must floor down
            FLT32_MAX,
        )
        for s in cases:
            num, den = scale_to_rational(s)
            self._assert_f32_scale_invariants(s, num, den)

        # 0.2004 and 0.2006 stay distinct and neither rounds up to 201/1000
        n_04, d_04 = scale_to_rational(0.2004)
        n_06, d_06 = scale_to_rational(0.2006)
        self.assertLess(Fraction(n_04, d_04), Fraction(n_06, d_06))
        self.assertLess(Fraction(n_06, d_06), Fraction(201, 1000))

        # Exact dyadic values preserve exact equality (0 error)
        self.assertEqual(scale_to_rational(1.0), (1, 1))
        self.assertEqual(scale_to_rational(0.5), (1, 2))
        self.assertEqual(scale_to_rational(0.75), (3, 4))
        self.assertEqual(scale_to_rational(0.25), (1, 4))

        # Exhaustive check across all 1000 thousandths k/1000 (k=1..1000)
        for k in range(1, 1001):
            s = k / 1000.0
            num, den = scale_to_rational(s)
            self._assert_f32_scale_invariants(s, num, den)
            # Within 1 binary32 ULP below s
            bits = struct.unpack("<I", struct.pack("<f", num / den))[0]
            next_up = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
            self.assertGreater(Fraction(next_up), Fraction(s))

    def test_subnormals_overflow_and_underflow(self) -> None:
        # Smallest positive IEEE-754 binary32 subnormal 2^-149 is exact
        n_sub, d_sub = scale_to_rational(FLT32_MIN_SUBNORMAL)
        self.assertEqual((n_sub, d_sub), (1, 1 << 149))
        self._assert_f32_scale_invariants(FLT32_MIN_SUBNORMAL, n_sub, d_sub)

        # Larger subnormal (2^-140) is exact
        n_sub2, d_sub2 = scale_to_rational(2.0**-140)
        self.assertEqual((n_sub2, d_sub2), (1, 1 << 140))
        self._assert_f32_scale_invariants(2.0**-140, n_sub2, d_sub2)

        # Strictly below smallest positive binary32 subnormal:
        # 0.75 * 2^-149 rounds UP to 2^-149 in round-to-nearest-even, so floor MUST reject (underflow to 0)
        for tiny in (0.75 * FLT32_MIN_SUBNORMAL, 0.5 * FLT32_MIN_SUBNORMAL, 1e-50):
            with self.assertRaises(ValueError, msg=f"tiny={tiny!r} must raise ValueError"):
                scale_to_rational(tiny)

        # Exact FLT32_MAX (both int and float) is accepted with zero saturation
        self.assertEqual(scale_to_rational(FLT32_MAX), (FLT32_MAX, 1))
        self.assertEqual(scale_to_rational(float(FLT32_MAX)), (FLT32_MAX, 1))

        # Overflow strictly above IEEE-754 binary32 max MUST reject with ValueError (never saturate)
        for huge in (
            FLT32_MAX + 1,
            math.nextafter(float(FLT32_MAX), math.inf),
            1e39,
            1e308,
            1 << 128,
            10**400,
        ):
            with self.assertRaises(ValueError, msg=f"huge={huge!r} must raise ValueError"):
                scale_to_rational(huge)

    def test_explicit_power_of_two_denominator_floor_and_rejection(self) -> None:
        # Power-of-two grid (1024): floors down and reduces to coprime pair
        self.assertEqual(scale_to_rational(0.2006, denominator=1024), (205, 1024))
        self.assertEqual(scale_to_rational(0.5, denominator=1024), (1, 2))
        self.assertEqual(scale_to_rational(1.0, denominator=1024), (1, 1))
        for n in range(1, 1025):
            s = n / 1024.0
            num, den = scale_to_rational(s, denominator=1024)
            self._assert_f32_scale_invariants(s, num, den)
            self.assertEqual(Fraction(num, den), Fraction(n, 1024))

        # Non-power-of-two denominators (like 1000) MUST be rejected
        for bad_den in (3, 10, 100, 1000, 1023, 1025):
            with self.assertRaises(ValueError):
                scale_to_rational(0.5, denominator=bad_den)

        # Non-positive or non-int denominators MUST be rejected
        for bad_type_den in (True, False, 1024.0, "1024"):
            with self.assertRaises(TypeError):
                scale_to_rational(0.5, denominator=bad_type_den)  # type: ignore[arg-type]
        for non_pos_den in (0, -1024):
            with self.assertRaises(ValueError):
                scale_to_rational(0.5, denominator=non_pos_den)

        # Scale flooring to zero on coarse power-of-two denominator MUST reject
        with self.assertRaises(ValueError):
            scale_to_rational(0.0005, denominator=1024)

        # Rational with >24-bit significand on huge power-of-two denominator is NOT exact in f32 -> MUST reject
        with self.assertRaises(ValueError):
            scale_to_rational(0.2, denominator=1 << 30)

        # Rational overflowing float32 on explicit denominator=1 -> MUST reject
        with self.assertRaises(ValueError):
            scale_to_rational(1 << 128, denominator=1)

    def test_invalid_scale_types_and_non_positive_values_rejected(self) -> None:
        # Reject bool, hostile int/float subclasses, Fraction, Decimal, and non-numerics
        for bad_type in (
            True,
            False,
            _EvilInt(1),
            _EvilFloat(0.5),
            Fraction(1, 2),
            Decimal("0.5"),
            "0.5",
            None,
            [0.5],
            object(),
        ):
            with self.assertRaises(TypeError, msg=f"bad_type={bad_type!r} must raise TypeError"):
                scale_to_rational(bad_type)  # type: ignore[arg-type]
            with self.assertRaises(TypeError, msg=f"bad_type={bad_type!r} (den=1024) must raise TypeError"):
                scale_to_rational(bad_type, denominator=1024)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            scale_to_rational(0.5, denominator=_EvilInt(1024))

        for bad_val in (0, -1, 0.0, -0.0, -0.5):
            with self.assertRaises(ValueError, msg=f"bad_val={bad_val!r} must raise ValueError") as ctx:
                scale_to_rational(bad_val)
            self.assertIn("finite positive scale", str(ctx.exception))
            with self.assertRaises(ValueError, msg=f"bad_val={bad_val!r} (den=1024) must raise ValueError") as ctx2:
                scale_to_rational(bad_val, denominator=1024)
            self.assertIn("finite positive scale", str(ctx2.exception))

        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"non_finite={non_finite!r} must raise ValueError") as nf_ctx:
                scale_to_rational(non_finite)
            self.assertIn("finite scale", str(nf_ctx.exception))

    def test_bridge_dispatch_and_certificate_guards_end_to_end(self) -> None:
        weft_cert = {
            "radix_scale_exp2": 24,
            "frobenius_sq_scaled": 16777216,
            "spectral_norm_sq_scaled": 4194304,
            "leakage_energy_scaled": 1024,
            "max_safety_dot_sq_scaled": 0,
            "projector_epoch": 1,
            "factor_digest": "a" * 64,
            "quant_dtype": "f16",
        }
        weft_manifest = {
            "schema": "weft-binding/v0.1",
            "adapter_name": "corpus_plain_writing",
            "adapter_class": "class1_rederivable",
            "layout_sha256": "1" * 64,
            "checkpoint_fingerprint": "2" * 64,
            "n_tensors": 2,
            "config_hashes": {"config.json": "3" * 64},
            "tokenizer_hashes": {
                "tokenizer.json": "4" * 64,
                "chat_template.jinja": "5" * 64,
            },
            "plane1_hashes": {"principles/writing.md": "6" * 64},
            "derivation_recipe": {"compiler": "weft/0.1.0"},
            "runtime_flags": {"quantization_mode": "bf16"},
            "runtime_flags_sha256": "7" * 64,
            "factor_digest": "a" * 64,
            "certificate_sha256": "b" * 64,
        }

        # End-to-end route -> bundles dispatch across non-dyadic manifest scale (0.2) and 1.0
        decision = ZoneRouter().route(
            RouteRequest(
                zone="prose",
                active_domains=("corpus", "weft-eido"),
                repo_policy=siliconcordat_repo_policy(),
            )
        )
        compiled = {
            d: {
                "adapter_id": f"mde:{d}",
                "llama_adapter_id": idx,
                "weft_manifest": weft_manifest,
                "weft_certificate": weft_cert,
                "gguf_sha256": "c" * 64,
            }
            for idx, d in enumerate(("corpus", "weft-eido"))
        }
        bundles = build_warp_bundles_for_route(
            decision,
            compiled,
            model_id="Synthetic-W0",
            source_lineage={
                "corpus_tree_sha256": "d" * 64,
                "domain_commits": {"corpus": "e" * 40, "weft-eido": "f" * 40},
                "contested_heads": [],
            },
            runtime_flags_sha256="9" * 64,
        )
        scale_by_domain = {e["id"]: float(e["scale"]) for e in decision.lora_scales}
        self.assertGreaterEqual(len(bundles), 1)
        for bundle in bundles:
            req_scale = scale_by_domain[bundle["domain"]]
            self._assert_f32_scale_invariants(req_scale, bundle["scale_num"], bundle["scale_den"])

        # Explicit power-of-two scale_denominator works end-to-end through build_warp_adapter_bundle
        b_1024 = build_warp_adapter_bundle(
            adapter_id="mde:corpus",
            domain="corpus",
            llama_adapter_id=0,
            weft_manifest=weft_manifest,
            weft_certificate=weft_cert,
            gguf_sha256="c" * 64,
            model_id="Synthetic-W0",
            source_lineage={"corpus_tree_sha256": "d" * 64, "domain_commits": {"corpus": "e" * 40}},
            runtime_flags_sha256="9" * 64,
            scale=0.2006,
            scale_denominator=1024,
        )
        self.assertEqual((b_1024["scale_num"], b_1024["scale_den"]), (205, 1024))

        # Cover bridge.py helper guards so mutation kill on bridge.py reaches 100%
        with self.assertRaises(TypeError):
            translate_weft_certificate_to_warp(
                "not-a-mapping",
                adapter_id="a",
                adapter_sha256="b" * 64,
                base_weights_sha256="c" * 64,
            )
        for bad_exp in (7, 49):
            with self.assertRaises(ValueError):
                translate_weft_certificate_to_warp(
                    dict(weft_cert, radix_scale_exp2=bad_exp),
                    adapter_id="a",
                    adapter_sha256="b" * 64,
                    base_weights_sha256="c" * 64,
                )
        for edge_exp in (8, 48):
            c_ok = translate_weft_certificate_to_warp(
                dict(weft_cert, radix_scale_exp2=edge_exp),
                adapter_id="a",
                adapter_sha256="b" * 64,
                base_weights_sha256="c" * 64,
            )
            self.assertEqual(c_ok["radix_scale"], 1 << edge_exp)
        with self.assertRaises(KeyError):
            no_radix = dict(weft_cert)
            no_radix.pop("radix_scale_exp2")
            translate_weft_certificate_to_warp(
                no_radix,
                adapter_id="a",
                adapter_sha256="b" * 64,
                base_weights_sha256="c" * 64,
            )
        with self.assertRaises(KeyError) as no_comp_ctx:
            build_warp_bundles_for_route(
                decision,
                {},
                model_id="Synthetic-W0",
                source_lineage={"corpus_tree_sha256": "d" * 64, "domain_commits": {}},
                runtime_flags_sha256="9" * 64,
            )
        self.assertIn("no compiled adapter", str(no_comp_ctx.exception))
        # radix_scale direct path (allow_zero=False) rejects 0 and -1
        for bad_rs in (0, -1):
            no_exp_cert = dict(weft_cert)
            no_exp_cert.pop("radix_scale_exp2")
            no_exp_cert["radix_scale"] = bad_rs
            with self.assertRaises(ValueError):
                translate_weft_certificate_to_warp(
                    no_exp_cert,
                    adapter_id="a",
                    adapter_sha256="b" * 64,
                    base_weights_sha256="c" * 64,
                )
        with self.assertRaises(ValueError):
            translate_weft_certificate_to_warp(
                dict(weft_cert, frobenius_sq_scaled=-1),
                adapter_id="a",
                adapter_sha256="b" * 64,
                base_weights_sha256="c" * 64,
            )
        # Zero allowed on frobenius_sq_scaled (allow_zero=True boundary)
        c_zero = translate_weft_certificate_to_warp(
            dict(weft_cert, frobenius_sq_scaled=0, spectral_norm_sq_scaled=0),
            adapter_id="a",
            adapter_sha256="b" * 64,
            base_weights_sha256="c" * 64,
        )
        self.assertEqual(c_zero["frobenius_sq_scaled"], 0)

    def test_route_bundles_strict_scale_validation_no_float_coercion(self) -> None:
        """D2 regression: build_warp_bundles_for_route rejects bool/subclass/Fraction/Decimal/non-finite/negative/overflow."""
        weft_cert = {
            "radix_scale_exp2": 24,
            "frobenius_sq_scaled": 16777216,
            "spectral_norm_sq_scaled": 4194304,
            "leakage_energy_scaled": 1024,
            "max_safety_dot_sq_scaled": 0,
            "projector_epoch": 1,
            "factor_digest": "a" * 64,
            "quant_dtype": "f16",
        }
        weft_manifest = {
            "schema": "weft-binding/v0.1",
            "adapter_name": "corpus_plain_writing",
            "adapter_class": "class1_rederivable",
            "layout_sha256": "1" * 64,
            "checkpoint_fingerprint": "2" * 64,
            "n_tensors": 2,
            "config_hashes": {"config.json": "3" * 64},
            "tokenizer_hashes": {
                "tokenizer.json": "4" * 64,
                "chat_template.jinja": "5" * 64,
            },
            "plane1_hashes": {"principles/writing.md": "6" * 64},
            "derivation_recipe": {"compiler": "weft/0.1.0"},
            "runtime_flags": {"quantization_mode": "bf16"},
            "runtime_flags_sha256": "7" * 64,
            "factor_digest": "a" * 64,
            "certificate_sha256": "b" * 64,
        }
        base_decision = ZoneRouter().route(
            RouteRequest(
                zone="prose",
                active_domains=("corpus",),
                repo_policy=siliconcordat_repo_policy(),
            )
        )
        compiled = {
            "corpus": {
                "adapter_id": "mde:corpus",
                "llama_adapter_id": 0,
                "weft_manifest": weft_manifest,
                "weft_certificate": weft_cert,
                "gguf_sha256": "c" * 64,
            }
        }
        lineage = {
            "corpus_tree_sha256": "d" * 64,
            "domain_commits": {"corpus": "e" * 40},
            "contested_heads": [],
        }

        def _dispatch(entries: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
            crafted = RouteDecision(
                zone="prose",
                lora_scales=entries,
                constraints=base_decision.constraints,
                rationale="r2-strict-scale-test",
                manifest_id=base_decision.manifest_id,
                l1_bank_tag=base_decision.l1_bank_tag,
            )
            return build_warp_bundles_for_route(
                crafted,
                compiled,
                model_id="Synthetic-W0",
                source_lineage=lineage,
                runtime_flags_sha256="9" * 64,
            )

        # Valid exact int/float scales dispatch cleanly; exact zeros (0, 0.0, -0.0) are skipped
        bundles_int = _dispatch(
            (
                {"id": "corpus", "scale": 1},
                {"id": "inactive-int", "scale": 0},
                {"id": "inactive-float", "scale": 0.0},
                {"id": "inactive-negzero", "scale": -0.0},
            )
        )
        self.assertEqual(len(bundles_int), 1)
        self.assertEqual((bundles_int[0]["scale_num"], bundles_int[0]["scale_den"]), (1, 1))

        bundles_f = _dispatch(({"id": "corpus", "scale": 0.2},))
        self.assertEqual(len(bundles_f), 1)
        self.assertEqual((bundles_f[0]["scale_num"], bundles_f[0]["scale_den"]), (3355443, 16777216))

        # Hostile types, bool, Fraction (including tiny 1/10**400), Decimal, str -> TypeError (no float() coercion)
        for bad_type in (
            True,
            False,
            _EvilInt(1),
            _EvilFloat(0.5),
            Fraction(1, 3),
            Fraction(1, 10**400),
            Decimal("0.2"),
            "0.5",
            None,
        ):
            with self.assertRaises(TypeError, msg=f"route scale={bad_type!r} must raise TypeError"):
                _dispatch(({"id": "corpus", "scale": bad_type},))

        # Non-finite floats, negative scales, and binary32 overflow (including 10**400) -> ValueError (NOT OverflowError)
        for bad_val in (
            float("nan"),
            float("inf"),
            float("-inf"),
            -1,
            -0.5,
            FLT32_MAX + 1,
            10**400,
        ):
            with self.assertRaises(ValueError, msg=f"route scale={bad_val!r} must raise ValueError"):
                _dispatch(({"id": "corpus", "scale": bad_val},))

        # Non-finite and negative scales rejected BEFORE compiled_adapters lookup (ValueError, not KeyError)
        for bad_missing in (float("nan"), float("inf"), float("-inf"), -1, -0.5):
            with self.assertRaises(ValueError, msg=f"missing-domain scale={bad_missing!r} must raise ValueError"):
                _dispatch(({"id": "missing-domain", "scale": bad_missing},))


if __name__ == "__main__":
    unittest.main()
