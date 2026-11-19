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
import hashlib
from typing import Any, Mapping, Optional

from exo.ledger import canonical_json_bytes
from exo.router import RouteDecision


def _as_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Expected mapping or dataclass instance, got {type(obj).__name__}")


def _sha256_canonical(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def scale_to_rational(scale: float, *, denominator: int = 1000) -> tuple[int, int]:
    """Convert a positive float LoRA scale into warp's strict (scale_num, scale_den)."""
    if denominator <= 0:
        raise ValueError("denominator must be strictly positive")
    if scale <= 0.0:
        raise ValueError(f"warp AdapterBundle requires positive scale; got {scale}")
    num = int(round(float(scale) * denominator))
    if num <= 0:
        raise ValueError(f"scale {scale} rounds to non-positive numerator at denominator {denominator}")
    return num, int(denominator)


def translate_weft_certificate_to_warp(
    weft_certificate: Any,
    *,
    adapter_id: str,
    adapter_sha256: str,
    base_weights_sha256: str,
) -> dict[str, Any]:
    """Translate an L2 weft FixedRadixCertificate into an L3 warp FixedRadixCertificate dict.

    Preserves the four exact scaled integer witness metrics and expands weft's
    `radix_scale_exp2` exponent into warp's explicit `radix_scale = 1 << exp2`,
    then computes warp's embedded canonical `cert_sha256`.
    """
    cert = _as_mapping(weft_certificate)
    if "radix_scale_exp2" in cert:
        exp2 = int(cert["radix_scale_exp2"])
        if not (8 <= exp2 <= 48):
            raise ValueError(f"weft radix_scale_exp2 out of range [8, 48]: {exp2}")
        radix_scale = 1 << exp2
    elif "radix_scale" in cert:
        radix_scale = int(cert["radix_scale"])
        if radix_scale <= 0:
            raise ValueError(f"radix_scale must be positive: {radix_scale}")
    else:
        raise KeyError("weft certificate missing both 'radix_scale_exp2' and 'radix_scale'")

    body: dict[str, Any] = {
        "adapter_id": str(adapter_id),
        "adapter_sha256": str(adapter_sha256),
        "base_weights_sha256": str(base_weights_sha256),
        "radix_scale": radix_scale,
        "frobenius_sq_scaled": int(cert["frobenius_sq_scaled"]),
        "spectral_norm_sq_scaled": int(cert["spectral_norm_sq_scaled"]),
        "leakage_energy_scaled": int(cert["leakage_energy_scaled"]),
        "max_safety_dot_sq_scaled": int(cert["max_safety_dot_sq_scaled"]),
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
    scale_denominator: int = 1000,
    adapter_path: Optional[str] = None,
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
    scale_num, scale_den = scale_to_rational(scale, denominator=scale_denominator)
    derivation_digest = _sha256_canonical(manifest)

    return {
        "adapter_id": str(adapter_id),
        "domain": str(domain),
        "llama_adapter_id": int(llama_adapter_id),
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


def build_warp_bundles_for_route(
    decision: RouteDecision,
    compiled_adapters: Mapping[str, Mapping[str, Any]],
    *,
    model_id: str,
    source_lineage: Mapping[str, Any],
    runtime_flags_sha256: str,
    scale_denominator: int = 1000,
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
    bundles: list[dict[str, Any]] = []
    for entry in decision.lora_scales:
        domain = str(entry["id"])
        scale = float(entry["scale"])
        if scale <= 0.0:
            continue
        if domain not in compiled_adapters:
            raise KeyError(f"RouteDecision activated domain '{domain}' with no compiled adapter")
        spec = compiled_adapters[domain]
        bundles.append(
            build_warp_adapter_bundle(
                adapter_id=str(spec["adapter_id"]),
                domain=domain,
                llama_adapter_id=int(spec["llama_adapter_id"]),
                weft_manifest=spec["weft_manifest"],
                weft_certificate=spec["weft_certificate"],
                gguf_sha256=str(spec["gguf_sha256"]),
                model_id=model_id,
                source_lineage=source_lineage,
                runtime_flags_sha256=runtime_flags_sha256,
                scale=scale,
                scale_denominator=scale_denominator,
                adapter_path=spec.get("adapter_path"),
            )
        )
    return bundles
