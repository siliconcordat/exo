"""Pure-stdlib cross-layer schema adapters bridging L2 weft, L4 exo, and L3 warp.

Keeps exo's zero-dependency stdlib-only core intact while translating between
real sibling schemas without modifying sibling repositories:
- L2 weft FixedRadixCertificate (`radix_scale_exp2`, `factor_digest`) ->
  L3 warp FixedRadixCertificate (`radix_scale`, `adapter_id`, `adapter_sha256`,
  `base_weights_sha256`, `cert_sha256`).
- L2 weft BindingManifest (`checkpoint_fingerprint`, `config_hashes`,
  `tokenizer_hashes`) -> L3 warp CheckpointFingerprint & AdapterBundle dicts.
- L4 exo RouteDecision float scales -> L3 warp strict rational (`scale_num`,
  `scale_den`).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fractions import Fraction
import hashlib
import math
import struct
from typing import Any, Mapping, Optional

from exo.ledger import canonical_json_bytes
from exo.router import RouteDecision


_FLT32_MAX = 340282346638528859811704183484516925440  # (2**24 - 1) * 2**104


def _require_strict_int(value: Any, name: str, *, allow_zero: bool = True) -> int:
    """Enforce exact Python int (rejecting bool, float, str, and negative values)."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int (got {type(value).__name__})")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative (got {value})")
    elif value <= 0:
        raise ValueError(f"{name} must be strictly positive (got {value})")
    return value


def _require_strict_scale(scale: Any, *, allow_zero: bool = False) -> int | float:
    """Enforce exact built-in int or float (rejecting subclasses, bool, Fraction, Decimal)."""
    if type(scale) not in (int, float):
        raise TypeError(f"scale must be an exact int or float (got {type(scale).__name__})")
    if type(scale) is float and not math.isfinite(scale):
        raise ValueError(f"warp AdapterBundle requires finite scale; got {scale}")
    if allow_zero:
        if scale < 0:
            raise ValueError(f"warp AdapterBundle requires non-negative scale; got {scale}")
    elif scale <= 0:
        raise ValueError(f"warp AdapterBundle requires finite positive scale; got {scale}")
    return scale


def _as_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Expected mapping or dataclass instance, got {type(obj).__name__}")


def _sha256_canonical(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _is_exact_f32_rational(num: int, den: int) -> bool:
    """Return True iff num/den is finite and represented identically as IEEE-754 binary32."""
    try:
        packed = struct.pack("<f", num / den)
    except OverflowError:
        return False
    unpacked = struct.unpack("<f", packed)[0]
    return math.isfinite(unpacked) and Fraction(unpacked) == Fraction(num, den)


def _floor_to_exact_f32(exact_value: Fraction) -> float:
    """Return the largest positive finite IEEE-754 binary32 value <= exact_value.

    Valid domain is ``2^-149 <= exact_value <= FLT32_MAX`` (``(2^24 - 1) * 2^104``):
    values strictly above ``FLT32_MAX`` overflow binary32 and raise ``ValueError``
    (never saturating to ``FLT32_MAX``), and values strictly below ``2^-149``
    underflow binary32 to zero and raise ``ValueError``.
    """
    if exact_value > _FLT32_MAX:
        raise ValueError(f"scale {exact_value} overflows IEEE-754 float32")
    packed = struct.pack("<f", float(exact_value))
    f32_val = struct.unpack("<f", packed)[0]
    bits = struct.unpack("<I", packed)[0]
    if Fraction(f32_val) > exact_value:
        bits -= 1
        f32_val = struct.unpack("<f", struct.pack("<I", bits))[0]
    if bits == 0:
        raise ValueError(f"scale {exact_value} underflows IEEE-754 float32 to zero")
    return f32_val


def scale_to_rational(scale: float, *, denominator: Optional[int] = None) -> tuple[int, int]:
    """Convert a positive LoRA scale into a coprime, float32-exact (scale_num, scale_den).

    Inputs must be exact built-in ``int`` or ``float`` (subclasses, ``bool``,
    ``Fraction``, and ``Decimal`` raise ``TypeError``), finite, strictly positive,
    and within IEEE-754 binary32 representable bounds: ``>= 2^-149`` (smaller positive
    values underflow to zero and raise ``ValueError``) and ``<= FLT32_MAX``
    (``(2^24 - 1) * 2^104``; larger values overflow binary32 and raise ``ValueError``
    rather than saturating).

    Guarantees:
      - ``scale_num > 0``, ``scale_den > 0``, and ``math.gcd(scale_num, scale_den) == 1``
      - ``scale_den`` is a power of two (dyadic rational)
      - ``Fraction(scale_num, scale_den) <= Fraction(scale)`` (never above requested scale)
      - ``float32(scale_num / scale_den) == Fraction(scale_num, scale_den)`` exactly
        (zero upward IEEE-754 binary32 overshoot when applied by ``llama-server``).

    When ``denominator`` is omitted (default), returns the exact reduced dyadic rational
    of the largest IEEE-754 binary32 value ``<= scale`` within ``[2^-149, FLT32_MAX]``.
    When ``denominator`` is supplied, it must be an exact positive power-of-two ``int``;
    ``scale`` is floored to that grid and reduced to lowest terms, rejecting values that
    floor to zero or are not exact in binary32.
    """
    scale = _require_strict_scale(scale, allow_zero=False)
    exact_scale = Fraction(scale)

    if denominator is None:
        f32_val = _floor_to_exact_f32(exact_scale)
        return f32_val.as_integer_ratio()

    den = _require_strict_int(denominator, "denominator", allow_zero=False)
    if den & (den - 1) != 0:
        raise ValueError(
            f"denominator must be a power of two for exact float32 representation (got {den})"
        )
    num = (exact_scale.numerator * den) // exact_scale.denominator
    if num <= 0:
        raise ValueError(f"scale {scale} floors to non-positive numerator at denominator {den}")
    g = math.gcd(num, den)
    num //= g
    den //= g
    if not _is_exact_f32_rational(num, den):
        raise ValueError(
            f"rational scale {num}/{den} is not exactly representable as IEEE-754 float32"
        )
    return num, den


def translate_weft_certificate_to_warp(
    weft_certificate: Any,
    *,
    adapter_id: str,
    adapter_sha256: str,
    base_weights_sha256: str,
) -> dict[str, Any]:
    """Translate an L2 weft FixedRadixCertificate into an L3 warp FixedRadixCertificate dict.

    Preserves the four exact scaled integer witness metrics without silent
    coercion, expands weft's `radix_scale_exp2` exponent into warp's explicit
    `radix_scale = 1 << exp2`, and computes warp's embedded canonical
    `cert_sha256`.
    """
    cert = _as_mapping(weft_certificate)
    if "radix_scale_exp2" in cert:
        exp2 = _require_strict_int(cert["radix_scale_exp2"], "radix_scale_exp2", allow_zero=False)
        if not (8 <= exp2 <= 48):
            raise ValueError(f"weft radix_scale_exp2 out of range [8, 48]: {exp2}")
        radix_scale = 1 << exp2
    elif "radix_scale" in cert:
        radix_scale = _require_strict_int(cert["radix_scale"], "radix_scale", allow_zero=False)
    else:
        raise KeyError("weft certificate missing both 'radix_scale_exp2' and 'radix_scale'")

    body: dict[str, Any] = {
        "adapter_id": str(adapter_id),
        "adapter_sha256": str(adapter_sha256),
        "base_weights_sha256": str(base_weights_sha256),
        "radix_scale": radix_scale,
        "frobenius_sq_scaled": _require_strict_int(cert["frobenius_sq_scaled"], "frobenius_sq_scaled"),
        "spectral_norm_sq_scaled": _require_strict_int(
            cert["spectral_norm_sq_scaled"], "spectral_norm_sq_scaled"
        ),
        "leakage_energy_scaled": _require_strict_int(cert["leakage_energy_scaled"], "leakage_energy_scaled"),
        "max_safety_dot_sq_scaled": _require_strict_int(
            cert["max_safety_dot_sq_scaled"], "max_safety_dot_sq_scaled"
        ),
    }
    out = dict(body)
    out["cert_sha256"] = _sha256_canonical(body)
    return out


def translate_weft_checkpoint_to_warp(
    weft_manifest: Any,
    *,
    model_id: str,
) -> dict[str, Any]:
    """Translate L2 weft BindingManifest checkpoint fields into L3 warp CheckpointFingerprint dict."""
    manifest = _as_mapping(weft_manifest)
    config_hashes = dict(manifest.get("config_hashes") or {})
    tokenizer_hashes = dict(manifest.get("tokenizer_hashes") or {})

    config_sha = config_hashes.get("config.json") or _sha256_canonical(config_hashes)
    tokenizer_sha = tokenizer_hashes.get("tokenizer.json") or _sha256_canonical(tokenizer_hashes)
    chat_template_sha = (
        tokenizer_hashes.get("chat_template.jinja")
        or tokenizer_hashes.get("chat_template.json")
        or _sha256_canonical({"chat_template": "absent", "tokenizer": tokenizer_hashes})
    )
    weights_sha = str(manifest["checkpoint_fingerprint"])
    return {
        "model_id": str(model_id),
        "layout_sha256": str(manifest["layout_sha256"]),
        "weights_sha256": weights_sha,
        "config_sha256": str(config_sha),
        "tokenizer_sha256": str(tokenizer_sha),
        "chat_template_sha256": str(chat_template_sha),
        "safetensors_provenance_sha256": weights_sha,
    }


def build_warp_adapter_bundle(
    *,
    adapter_id: str,
    domain: str,
    llama_adapter_id: int,
    weft_manifest: Any,
    weft_certificate: Any,
    gguf_sha256: str,
    model_id: str,
    source_lineage: Mapping[str, Any],
    runtime_flags_sha256: str,
    scale: float = 1.0,
    scale_denominator: Optional[int] = None,
    adapter_path: Optional[str] = None,
    zone_manifest_id: Optional[str] = None,
) -> dict[str, Any]:
    """Construct a complete L3 warp-modeld AdapterBundle dict from L2 weft + L4 exo outputs."""
    manifest = _as_mapping(weft_manifest)
    checkpoint = translate_weft_checkpoint_to_warp(manifest, model_id=model_id)
    cert_dict = translate_weft_certificate_to_warp(
        weft_certificate,
        adapter_id=adapter_id,
        adapter_sha256=gguf_sha256,
        base_weights_sha256=checkpoint["weights_sha256"],
    )
    llama_id = _require_strict_int(llama_adapter_id, "llama_adapter_id", allow_zero=True)
    scale_num, scale_den = scale_to_rational(scale, denominator=scale_denominator)
    derivation_digest = _sha256_canonical(manifest)

    out: dict[str, Any] = {
        "adapter_id": str(adapter_id),
        "domain": str(domain),
        "llama_adapter_id": llama_id,
        "adapter_sha256": str(gguf_sha256),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "scale_num": scale_num,
        "scale_den": scale_den,
        "checkpoint": checkpoint,
        "source_lineage": {
            "corpus_tree_sha256": str(source_lineage["corpus_tree_sha256"]),
            "domain_commits": {
                str(domain): str(source_lineage["domain_commits"][domain]),
            },
            "contested_heads": list(source_lineage.get("contested_heads", ())),
        },
        "runtime_flags_sha256": str(runtime_flags_sha256),
        "derivation_digest": derivation_digest,
        "certificate": cert_dict,
    }
    if zone_manifest_id is not None:
        out["zone_manifest_id"] = str(zone_manifest_id)
    return out


def build_warp_bundles_for_route(
    decision: RouteDecision,
    compiled_adapters: Mapping[str, Mapping[str, Any]],
    *,
    model_id: str,
    source_lineage: Mapping[str, Any],
    runtime_flags_sha256: str,
    scale_denominator: Optional[int] = None,
    zone_manifest_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build warp-modeld AdapterBundle dicts for every non-zero domain in an exo RouteDecision.

    `compiled_adapters` maps domain name (e.g. "corpus", "formal-verification",
    "weft-eido") to a dict with keys:
      - `adapter_id`: str (e.g. "mde:corpus")
      - `llama_adapter_id`: int
      - `weft_manifest`: BindingManifest or dict
      - `weft_certificate`: FixedRadixCertificate or dict
      - `gguf_sha256`: 64-hex SHA-256 of the compiled GGUF file
      - `adapter_path`: optional path string to the GGUF file on disk
    """
    effective_zm = zone_manifest_id if zone_manifest_id is not None else decision.manifest_id
    bundles: list[dict[str, Any]] = []
    for entry in decision.lora_scales:
        domain = str(entry["id"])
        raw_scale = _require_strict_scale(entry["scale"], allow_zero=True)
        if raw_scale == 0:
            continue
        if domain not in compiled_adapters:
            raise KeyError(f"RouteDecision activated domain '{domain}' with no compiled adapter")
        spec = compiled_adapters[domain]
        bundles.append(
            build_warp_adapter_bundle(
                adapter_id=str(spec["adapter_id"]),
                domain=domain,
                llama_adapter_id=spec["llama_adapter_id"],
                weft_manifest=spec["weft_manifest"],
                weft_certificate=spec["weft_certificate"],
                gguf_sha256=str(spec["gguf_sha256"]),
                model_id=model_id,
                source_lineage=source_lineage,
                runtime_flags_sha256=runtime_flags_sha256,
                scale=raw_scale,
                scale_denominator=scale_denominator,
                adapter_path=spec.get("adapter_path"),
                zone_manifest_id=effective_zm,
            )
        )
    return bundles
