"""Versioned Phase 3c final missing-KV policy artifact helpers.

This module serializes only fitted Phase 3c final-policy parameters and
lightweight metadata. It intentionally does not load, modify, or insert runtime
``past_key_values`` and does not replace the legacy runtime K/V restoration
artifact schema.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from our_kv_restoration.missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_CONFIDENCE_COMPUTE_DTYPE,
    CALM_CONFIDENCE_TYPE,
    CALM_HYBRID_POLICY_NAME,
    CALM_HYBRID_SOURCE_POPULATION_MODE_MAP,
    CALM_POLICY_NAME,
    CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME,
    CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP,
    CALM_THRESHOLD,
    CALM_THRESHOLD_COMPARATOR,
    CALM_USE_ADAPT_THRESHOLD,
    OFFICIAL_FREE_CALM_CONFIDENCE_COMPUTE_DTYPE,
    OFFICIAL_FREE_CALM_CONFIDENCE_TYPE,
    OFFICIAL_FREE_CALM_POLICY_NAME,
    OFFICIAL_FREE_CALM_THRESHOLD_COMPARATOR,
    OFFICIAL_FREE_CALM_USE_ADAPT_THRESHOLD,
    calm_hybrid_calibration_policy_payload,
    calm_hybrid_calibration_policy_sha256,
    calm_policy_payload,
    calm_policy_sha256,
)
from our_kv_restoration.missing_kv_dump_provenance import canonical_json_sha256


ARTIFACT_TYPE = "phase3c_final_missing_kv_policy"
LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = LEGACY_SCHEMA_VERSION
LATEST_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (LEGACY_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
METHOD_NAME = "would_exit_aware_hidden_kv_residual_restoration"

HIDDEN_METHOD = "diagonal_affine_raw"
HIDDEN_FIT_SCOPE = "would_exit_layer_pair"
K_CORRECTION = "head_channel_affine"
K_FIT_SCOPE = "layer_pair_head_channel"
V_CORRECTION = "headwise_procrustes"
V_FIT_SCOPE = "gap_bin_head"
EXIT_CONFIDENCE_LEGACY_SEMANTICS_VERSION = 1
EXIT_CONFIDENCE_SEMANTICS_VERSION = 2
EXIT_CONFIDENCE_TYPE = "softmax_top1_top2_margin"
EXIT_CONFIDENCE_THRESHOLD_COMPARATOR = "strict_gt"
EXIT_CONFIDENCE_HIDDEN_POSITION = "raw_hidden_before_source_block_self_attention_layer_norm"
EXIT_CONFIDENCE_PROJECTION_PATH = "decoder_final_layer_norm_then_tied_embedding_scaled_lm_head"
EXIT_CONFIDENCE_TIE_SCALE = "d_model^-0.5_when_tied"
EXIT_CONFIDENCE_RUNTIME_FRAMEWORK = "FREE_fixed_shallow_layer"
EXIT_CONFIDENCE_COMPUTE_DTYPE = "float32"
FIXED_LAYER_MEMBERSHIP_SEMANTICS_VERSION = 1
FIXED_LAYER_POLICY_SEMANTICS_VERSION = 1
FIXED_LAYER_POLICY_TYPE = "phase3c_fixed_layer_restoration_policy_v1"
FIXED_LAYER_MEMBERSHIP_LAYER = "fixed_source_layer"
FIXED_LAYER_EARLIER_CONFIDENCE_POLICY = "ignored_for_fixed_layer_membership"
SOURCE_LAYER_MODE_FIXED = "fixed_layer"
SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING = "candidate_first_crossing"
# Official FREE CALM-style production early-exit path: a contiguous
# candidate-layer range (exit_min_layer..num_decoder_layers-1) with a
# runtime-config-driven threshold, never the frozen historical
# (4,6,8,10)/0.9 policy above. A parallel, separately validated mode --
# SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING and its exact-equality-to-the-
# frozen-tuple validation are never modified or relaxed by this addition.
SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM = "official_free_calm_first_crossing"
CANDIDATE_FIRST_CROSSING_SEMANTICS_VERSION = 1
SAME_LAYER_PROJECTION_POLICY = "target_native_projection_no_learned_map"
SAME_LAYER_PROJECTION_VALIDATION_SCHEMA_VERSION = 1
SAME_LAYER_PROJECTION_ATOL = 1e-5
SAME_LAYER_PROJECTION_RTOL = 1e-4
STABLE_SAMPLE_CLUSTER_SPLIT_VERSION = 1

_THRESHOLD_PREFIX = "threshold:"
_PAIR_PATTERN = re.compile(r"^s:(-?\d+)->t:(-?\d+)$")
_GAP_PREFIX = "gap:"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TOP_LEVEL_KEYS = {
    "artifact_type",
    "schema_version",
    "method_name",
    "model_spec",
    "layer_index_semantics",
    "fit_config",
    "policy_config",
    "parameters_by_threshold",
    "provenance",
}
_THRESHOLD_PAYLOAD_KEYS = {"hidden_by_layer_pair", "k_by_layer_pair", "v_by_gap_bin"}
_AFFINE_PARAM_KEYS = {"scale", "bias"}
_V_PARAM_KEYS = {"rotations", "biases"}
_FORBIDDEN_KEYS = {
    "fit_cache",
    "FitCache",
    "examples",
    "records",
    "train_examples",
    "eval_examples",
    "model",
    "tokenizer",
    "dataloader",
}


def threshold_to_key(threshold: Any) -> str:
    """Return the canonical threshold key used in Phase 3c artifacts."""

    if isinstance(threshold, str) and threshold.startswith(_THRESHOLD_PREFIX):
        threshold = threshold_key_to_float(threshold)
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError("threshold must be finite, got {!r}".format(threshold))
    return _THRESHOLD_PREFIX + repr(value)


def threshold_key_to_float(key: str) -> float:
    """Decode a canonical threshold key."""

    if not isinstance(key, str) or not key.startswith(_THRESHOLD_PREFIX):
        raise ValueError("threshold key must start with {!r}: {!r}".format(_THRESHOLD_PREFIX, key))
    value = float(key[len(_THRESHOLD_PREFIX) :])
    if not math.isfinite(value):
        raise ValueError("threshold key must decode to a finite value: {!r}".format(key))
    return value


def layer_pair_to_key(source_layer: int, target_layer: int) -> str:
    """Encode a source/target decoder layer pair deterministically."""

    return "s:{}->t:{}".format(int(source_layer), int(target_layer))


def layer_pair_key_to_tuple(key: str) -> Tuple[int, int]:
    """Decode a canonical layer-pair key."""

    match = _PAIR_PATTERN.match(str(key))
    if not match:
        raise ValueError("invalid layer-pair key {!r}; expected s:<source>->t:<target>".format(key))
    return int(match.group(1)), int(match.group(2))


def gap_bin_to_key(gap_bin: Any) -> str:
    """Encode a gap-bin label deterministically."""

    if isinstance(gap_bin, str) and gap_bin.startswith(_GAP_PREFIX):
        return gap_bin
    text = str(gap_bin)
    if not text:
        raise ValueError("gap-bin key cannot be empty")
    return _GAP_PREFIX + text


def gap_bin_key_to_label(key: str) -> str:
    """Decode a canonical gap-bin key."""

    if not isinstance(key, str) or not key.startswith(_GAP_PREFIX):
        raise ValueError("gap-bin key must start with {!r}: {!r}".format(_GAP_PREFIX, key))
    label = key[len(_GAP_PREFIX) :]
    if not label:
        raise ValueError("gap-bin label cannot be empty")
    return label


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_layer_index_semantics() -> Dict[str, Any]:
    return {
        "version": 1,
        "decoder_layer_indexing": "zero-based decoder block index",
        "description": "Offline Phase 3c layer indices come from the all-layer hidden/KV diagnostic dumps.",
        "source_raw_hidden_position": (
            "source_layer identifies the decoder block whose raw hidden state is used as h_s. "
            "In models/deploying_t5.py, T5LayerSelfAttention stores "
            "_last_raw_hidden_states_for_hidden_dump before that block's self-attention LayerNorm."
        ),
        "target_raw_hidden_position": (
            "target_layer identifies the decoder block whose raw hidden state is used as h_t in the "
            "offline diagnostic, captured before that block's self-attention LayerNorm when raw hidden "
            "dumping is enabled."
        ),
        "target_kv_projection_layer": (
            "K/V projection for target_layer uses model.decoder.block[target_layer].layer[0].layer_norm "
            "followed by that block's SelfAttention.k and SelfAttention.v projections."
        ),
        "kv_dump_position": (
            "The all-layer K/V calibration dump stores the per-token self-attention present_key_value "
            "slice produced by the same zero-based decoder block layer_idx."
        ),
        "raw_hidden_capture_before_self_attention_layer_norm": True,
        "requires_raw_hidden_state_dump": True,
        "raw_hidden_unavailable_note": (
            "If raw_hidden_state was not included in the all-layer hidden dump, the Phase 2/3 hidden-based "
            "diagnostic cannot build the final-policy source/target examples from this convention."
        ),
        "source_target_pair_key": "s:<source_layer>->t:<target_layer>",
        "gap_bin_key": "gap:<gap_bin_label>",
        "threshold_key": "threshold:<repr(float_threshold)>",
        "runtime_indexing_note": (
            "For FREE shallow-deep runtime reconciliation, when confidence is checked before decoder "
            "block i executes, legacy source-K/V restoration uses the last exact K/V layer i-1, while "
            "Phase 3c hidden-based restoration uses the raw hidden entering block i as "
            "phase3c_source_hidden_layer=i. Runtime target layer i is handled by same-layer exact "
            "projection through that block's self-attention LayerNorm and K/V projections; target "
            "layers deeper than i use the stored Phase 3c source-target artifact maps."
        ),
    }


def default_policy_config() -> Dict[str, Any]:
    return {
        "final_hidden_policy": {
            "method": HIDDEN_METHOD,
            "fit_scope": HIDDEN_FIT_SCOPE,
            "condition": "would_exit_tokens",
        },
        "final_k_correction_policy": {
            "method": K_CORRECTION,
            "fit_scope": K_FIT_SCOPE,
        },
        "final_v_correction_policy": {
            "method": V_CORRECTION,
            "fit_scope": V_FIT_SCOPE,
        },
        "exact_threshold_match_required": True,
        "nearest_threshold_fallback": False,
        "short_gap_v_identity_fallback": False,
    }


def default_exit_confidence_semantics() -> Dict[str, Any]:
    """Return the runtime-equivalent FREE confidence semantics for new artifacts."""

    return {
        "version": EXIT_CONFIDENCE_SEMANTICS_VERSION,
        "confidence_type": EXIT_CONFIDENCE_TYPE,
        "threshold_comparator": EXIT_CONFIDENCE_THRESHOLD_COMPARATOR,
        "hidden_position": EXIT_CONFIDENCE_HIDDEN_POSITION,
        "projection_path": EXIT_CONFIDENCE_PROJECTION_PATH,
        "tie_word_embedding_scale": EXIT_CONFIDENCE_TIE_SCALE,
        "runtime_framework": EXIT_CONFIDENCE_RUNTIME_FRAMEWORK,
        "confidence_compute_dtype": EXIT_CONFIDENCE_COMPUTE_DTYPE,
    }


def default_fixed_layer_membership_semantics() -> Dict[str, Any]:
    """Return the fixed-layer would-exit population semantics for new artifacts."""

    return {
        "version": FIXED_LAYER_MEMBERSHIP_SEMANTICS_VERSION,
        "source_layer_mode": SOURCE_LAYER_MODE_FIXED,
        "membership_layer": FIXED_LAYER_MEMBERSHIP_LAYER,
        "earlier_layer_confidence_policy": FIXED_LAYER_EARLIER_CONFIDENCE_POLICY,
    }


def default_candidate_first_crossing_semantics(
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
    source_layer_mode: str = SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
    policy_name: str = CALM_POLICY_NAME,
) -> Dict[str, Any]:
    """Return the frozen Stage-2 CALM candidate-first-crossing metadata.

    ``source_layer_mode``/``policy_name`` default to the frozen historical
    (4,6,8,10)/0.9 policy identity -- every existing caller is unaffected.
    The official FREE CALM path passes ``source_layer_mode=
    SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM`` and
    ``policy_name=OFFICIAL_FREE_CALM_POLICY_NAME`` explicitly instead.
    """

    layers = [int(layer) for layer in candidate_layers]
    return {
        "version": CANDIDATE_FIRST_CROSSING_SEMANTICS_VERSION,
        "source_layer_mode": str(source_layer_mode),
        "policy_name": str(policy_name),
        "candidate_exit_layers": layers,
        "candidate_evaluation_order": layers,
        "policy": calm_policy_payload(candidate_layers=layers, threshold=float(threshold), policy_name=policy_name),
        "policy_sha256": calm_policy_sha256(candidate_layers=layers, threshold=float(threshold), policy_name=policy_name),
        "confidence_type": CALM_CONFIDENCE_TYPE,
        "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
        "threshold": float(threshold),
        "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
        "adaptive_threshold": CALM_USE_ADAPT_THRESHOLD,
        "trace_source": "authenticated_calm_candidate_confidence_trace",
        "same_layer_projection_policy": SAME_LAYER_PROJECTION_POLICY,
    }


def default_official_free_calm_semantics(
    *,
    candidate_layers: Sequence[int],
    threshold: float,
) -> Dict[str, Any]:
    """Official FREE CALM counterpart of
    ``default_candidate_first_crossing_semantics``: same schema, but
    ``source_layer_mode``/``policy_name`` identify the official contiguous
    policy instead of the frozen historical (4,6,8,10)/0.9 one, and
    ``candidate_layers``/``threshold`` are required (never silently defaulted
    to the historical policy's values)."""

    return default_candidate_first_crossing_semantics(
        candidate_layers=candidate_layers,
        threshold=threshold,
        source_layer_mode=SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
        policy_name=OFFICIAL_FREE_CALM_POLICY_NAME,
    )


def _normalized_gap_bin_definitions(fit_config: Mapping[str, Any]) -> List[Dict[str, int | str]]:
    gap_bins = _require_mapping(fit_config, "fit_config").get("gap_bins")
    if not isinstance(gap_bins, (list, tuple)) or not gap_bins:
        raise ValueError("fit_config.gap_bins must be a non-empty list")
    result: List[Dict[str, int | str]] = []
    for idx, item in enumerate(gap_bins):
        item = _require_mapping(item, "fit_config.gap_bins[{}]".format(idx))
        label = str(_require_required_mapping_field(item, "label", "fit_config.gap_bins[{}]".format(idx)))
        start = int(_require_required_mapping_field(item, "start", "fit_config.gap_bins[{}]".format(idx)))
        end = int(_require_required_mapping_field(item, "end", "fit_config.gap_bins[{}]".format(idx)))
        result.append({"label": label, "start": start, "end": end})
    return result


def fixed_layer_policy_semantics_payload(
    fit_config: Mapping[str, Any],
    policy_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the canonical fixed-layer Phase 3c policy identity payload."""

    fit_config = _require_mapping(fit_config, "fit_config")
    policy_config = _require_mapping(policy_config or default_policy_config(), "policy_config")
    thresholds = [float(item) for item in fit_config.get("thresholds", [])]
    if not thresholds:
        raise ValueError("fixed-layer policy identity requires fit_config.thresholds")
    membership = fit_config.get("fixed_layer_membership_semantics") or default_fixed_layer_membership_semantics()
    exit_semantics = fit_config.get("exit_confidence_semantics") or default_exit_confidence_semantics()
    return {
        "policy_schema_version": FIXED_LAYER_POLICY_SEMANTICS_VERSION,
        "policy_type": FIXED_LAYER_POLICY_TYPE,
        "source_layer_mode": SOURCE_LAYER_MODE_FIXED,
        "fixed_source_layer": int(_require_required_mapping_field(fit_config, "fixed_source_layer", "fit_config")),
        "thresholds": thresholds,
        "threshold": thresholds[0] if len(thresholds) == 1 else None,
        "threshold_comparator": EXIT_CONFIDENCE_THRESHOLD_COMPARATOR,
        "confidence_type": EXIT_CONFIDENCE_TYPE,
        "confidence_compute_dtype": EXIT_CONFIDENCE_COMPUTE_DTYPE,
        "adaptive_threshold": False,
        "exit_confidence_semantics": dict(exit_semantics),
        "fixed_layer_membership_semantics": dict(membership),
        "hidden_restoration_method": str(fit_config.get("hidden_method", HIDDEN_METHOD)),
        "hidden_fit_scope": str(fit_config.get("hidden_fit_scope", HIDDEN_FIT_SCOPE)),
        "k_correction_method": str(fit_config.get("k_correction", K_CORRECTION)),
        "k_fit_scope": str(fit_config.get("k_fit_scope", K_FIT_SCOPE)),
        "v_correction_method": str(fit_config.get("v_correction", V_CORRECTION)),
        "v_fit_scope": str(fit_config.get("v_fit_scope", V_FIT_SCOPE)),
        "policy_config": {
            "final_hidden_policy": dict(_require_mapping(policy_config.get("final_hidden_policy"), "policy_config.final_hidden_policy")),
            "final_k_correction_policy": dict(_require_mapping(policy_config.get("final_k_correction_policy"), "policy_config.final_k_correction_policy")),
            "final_v_correction_policy": dict(_require_mapping(policy_config.get("final_v_correction_policy"), "policy_config.final_v_correction_policy")),
            "exact_threshold_match_required": bool(policy_config.get("exact_threshold_match_required")),
            "nearest_threshold_fallback": bool(policy_config.get("nearest_threshold_fallback")),
            "short_gap_v_identity_fallback": bool(policy_config.get("short_gap_v_identity_fallback")),
        },
        "gap_bins": _normalized_gap_bin_definitions(fit_config),
        "exact_threshold_match_required": bool(fit_config.get("exact_threshold_match_required")),
        "nearest_threshold_fallback": not bool(fit_config.get("no_nearest_threshold_fallback")),
        "same_layer_target_policy": SAME_LAYER_PROJECTION_POLICY,
        "decoder_layer_indexing_semantics_version": 1,
        "artifact_parameter_identity_included": False,
        "artifact_parameter_identity_note": (
            "policy_sha256 identifies policy semantics; artifact_file_sha256 identifies fitted parameter tensors"
        ),
    }


def fixed_layer_policy_sha256(
    fit_config: Mapping[str, Any],
    policy_config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the canonical SHA-256 for fixed-layer Phase 3c policy semantics."""

    return canonical_json_sha256(fixed_layer_policy_semantics_payload(fit_config, policy_config))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _type_name(value: Any) -> str:
    return "{}.{}".format(type(value).__module__, type(value).__qualname__)


def _require_mapping(value: Any, field_path: str) -> Mapping[Any, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError("{} must be a mapping, got {}".format(field_path, _type_name(value)))
    return value


def _metadata_json_safe(value: Any, field_path: str) -> None:
    """Recursively reject non-primitive metadata payloads.

    Parameter tensors are only legal under ``parameters_by_threshold`` and are
    validated separately. This keeps artifacts free of FitCache/model/tokenizer
    objects, Path instances, callables, raw records, and other accidental state.
    """

    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("{} must be a finite metadata float, got {}".format(field_path, value))
        return
    if isinstance(value, torch.Tensor):
        raise ValueError("{} has unsupported metadata type torch.Tensor".format(field_path))
    if isinstance(value, Path):
        raise ValueError("{} has unsupported metadata type pathlib.Path".format(field_path))
    if callable(value):
        raise ValueError("{} has unsupported metadata type callable {}".format(field_path, _type_name(value)))
    if isinstance(value, (set, frozenset)):
        raise ValueError("{} has unsupported metadata type {}".format(field_path, _type_name(value)))
    if isinstance(value, MappingABC):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} has non-string metadata key {!r} of type {}".format(field_path, key, _type_name(key)))
            _metadata_json_safe(item, "{}.{}".format(field_path, key))
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _metadata_json_safe(item, "{}[{}]".format(field_path, idx))
        return
    raise ValueError("{} has unsupported metadata type {}".format(field_path, _type_name(value)))


def _require_required_mapping_field(mapping: Mapping[Any, Any], field: str, parent_path: str) -> Any:
    if field not in mapping:
        raise ValueError("{}.{} is required".format(parent_path, field))
    return mapping[field]


def _require_positive_int(mapping: Mapping[Any, Any], field: str, parent_path: str) -> int:
    value = _require_required_mapping_field(mapping, field, parent_path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{}.{} must be a positive integer".format(parent_path, field))
    return int(value)


def _require_nonnegative_int(mapping: Mapping[Any, Any], field: str, parent_path: str) -> int:
    value = _require_required_mapping_field(mapping, field, parent_path)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{}.{} must be a non-negative integer".format(parent_path, field))
    return int(value)


def _require_positive_float(mapping: Mapping[Any, Any], field: str, parent_path: str) -> float:
    value = _require_required_mapping_field(mapping, field, parent_path)
    if isinstance(value, bool):
        raise ValueError("{}.{} must be a finite positive float".format(parent_path, field))
    try:
        value = float(value)
    except Exception as exc:
        raise ValueError("{}.{} must be a finite positive float".format(parent_path, field)) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("{}.{} must be a finite positive float".format(parent_path, field))
    return value


def _require_exact_value(mapping: Mapping[Any, Any], field: str, expected: Any, parent_path: str) -> None:
    value = _require_required_mapping_field(mapping, field, parent_path)
    if value != expected:
        raise ValueError("{}.{} must be {!r}, got {!r}".format(parent_path, field, expected, value))


def _sorted_key_repr(keys: Iterable[Any]) -> list:
    return sorted([str(key) for key in keys])


def _require_exact_keys(mapping: Mapping[Any, Any], allowed: set, field_path: str) -> None:
    actual = set(mapping.keys())
    missing = allowed - actual
    if missing:
        raise ValueError("{} missing required fields: {}".format(field_path, _sorted_key_repr(missing)))
    extra = actual - allowed
    if extra:
        raise ValueError("{} contains unexpected fields: {}".format(field_path, _sorted_key_repr(extra)))


def _normalized_path(path: Any) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _ensure_tensor_cpu_float32(value: Any, field_path: str) -> torch.Tensor:
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
        value = torch.stack([item.detach().cpu() for item in value], dim=0)
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a torch.Tensor".format(field_path))
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError("{} contains NaN or Inf".format(field_path))
    return tensor


def _canonical_layer_pair_key(key: Any) -> str:
    if isinstance(key, str):
        source, target = layer_pair_key_to_tuple(key)
        return layer_pair_to_key(source, target)
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return layer_pair_to_key(int(key[0]), int(key[1]))
    raise ValueError("layer-pair map key must be canonical string or (source,target), got {!r}".format(key))


def _canonical_gap_bin_key(key: Any) -> str:
    return gap_bin_to_key(key)


def _insert_unique(mapping: Dict[str, Any], key: str, value: Any, message: str) -> None:
    if key in mapping:
        raise ValueError(message.format(key))
    mapping[key] = value


def _canonical_params(parameters_by_threshold: Mapping[Any, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    _require_mapping(parameters_by_threshold, "parameters_by_threshold")
    canonical: Dict[str, Dict[str, Any]] = {}
    for threshold, payload in parameters_by_threshold.items():
        threshold_key = threshold_to_key(threshold)
        if threshold_key in canonical:
            raise ValueError("duplicate canonical threshold key: {}".format(threshold_key))
        payload = _require_mapping(payload, "{} payload".format(threshold_key))
        _require_exact_keys(payload, _THRESHOLD_PAYLOAD_KEYS, "parameters_by_threshold.{}".format(threshold_key))
        threshold_payload = {
            "hidden_by_layer_pair": {},
            "k_by_layer_pair": {},
            "v_by_gap_bin": {},
        }
        hidden_group = _require_mapping(payload["hidden_by_layer_pair"], "{}.hidden_by_layer_pair".format(threshold_key))
        k_group = _require_mapping(payload["k_by_layer_pair"], "{}.k_by_layer_pair".format(threshold_key))
        v_group = _require_mapping(payload["v_by_gap_bin"], "{}.v_by_gap_bin".format(threshold_key))
        for raw_key, params in hidden_group.items():
            pair_key = _canonical_layer_pair_key(raw_key)
            param_path = "parameters_by_threshold.{}.hidden_by_layer_pair.{}".format(threshold_key, pair_key)
            params = _require_mapping(params, param_path)
            _require_exact_keys(params, _AFFINE_PARAM_KEYS, param_path)
            _insert_unique(threshold_payload["hidden_by_layer_pair"], pair_key, {
                "scale": _ensure_tensor_cpu_float32(
                    params["scale"],
                    "{}.hidden.{}.scale".format(threshold_key, pair_key),
                ),
                "bias": _ensure_tensor_cpu_float32(
                    params["bias"],
                    "{}.hidden.{}.bias".format(threshold_key, pair_key),
                ),
            }, "duplicate canonical hidden layer-pair key: {}")
        for raw_key, params in k_group.items():
            pair_key = _canonical_layer_pair_key(raw_key)
            param_path = "parameters_by_threshold.{}.k_by_layer_pair.{}".format(threshold_key, pair_key)
            params = _require_mapping(params, param_path)
            _require_exact_keys(params, _AFFINE_PARAM_KEYS, param_path)
            _insert_unique(threshold_payload["k_by_layer_pair"], pair_key, {
                "scale": _ensure_tensor_cpu_float32(
                    params["scale"],
                    "{}.k.{}.scale".format(threshold_key, pair_key),
                ),
                "bias": _ensure_tensor_cpu_float32(
                    params["bias"],
                    "{}.k.{}.bias".format(threshold_key, pair_key),
                ),
            }, "duplicate canonical K layer-pair key: {}")
        for raw_key, params in v_group.items():
            gap_key = _canonical_gap_bin_key(raw_key)
            param_path = "parameters_by_threshold.{}.v_by_gap_bin.{}".format(threshold_key, gap_key)
            params = _require_mapping(params, param_path)
            _require_exact_keys(params, _V_PARAM_KEYS, param_path)
            _insert_unique(threshold_payload["v_by_gap_bin"], gap_key, {
                "rotations": _ensure_tensor_cpu_float32(
                    params["rotations"],
                    "{}.v.{}.rotations".format(threshold_key, gap_key),
                ),
                "biases": _ensure_tensor_cpu_float32(
                    params["biases"],
                    "{}.v.{}.biases".format(threshold_key, gap_key),
                ),
            }, "duplicate canonical V gap-bin key: {}")
        canonical[threshold_key] = threshold_payload
    return dict(sorted(canonical.items(), key=lambda item: threshold_key_to_float(item[0])))


def build_phase3c_policy_artifact(
    *,
    parameters_by_threshold: Mapping[Any, Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    fit_config: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
    layer_index_semantics: Optional[Mapping[str, Any]] = None,
    policy_config: Optional[Mapping[str, Any]] = None,
    schema_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and validate a Phase 3c final-policy artifact.

    ``parameters_by_threshold`` must contain only the final-policy maps:
    would-exit layer-pair hidden diagonal affine, layer-pair head/channel K
    affine, and gap-bin head-wise V Procrustes.
    """

    model_spec = _require_mapping(model_spec, "model_spec")
    fit_config = _require_mapping(fit_config, "fit_config")
    provenance = _require_mapping(provenance or {}, "provenance")
    layer_index_semantics = _require_mapping(layer_index_semantics or default_layer_index_semantics(), "layer_index_semantics")
    policy_config = _require_mapping(policy_config or default_policy_config(), "policy_config")
    fit_config = dict(fit_config)
    fit_config.setdefault("source_layer_mode", "first_threshold")
    fit_config.setdefault("exit_confidence_semantics", default_exit_confidence_semantics())
    source_layer_mode = fit_config.get("source_layer_mode")
    if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        fit_config.setdefault("fixed_layer_membership_semantics", default_fixed_layer_membership_semantics())
        fixed_policy_payload = fixed_layer_policy_semantics_payload(fit_config, policy_config)
        fixed_policy_payload["policy_sha256"] = canonical_json_sha256(fixed_policy_payload)
        fit_config["fixed_layer_policy_semantics"] = fixed_policy_payload
        fit_config["policy_sha256"] = fixed_policy_payload["policy_sha256"]
    elif source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        thresholds = fit_config.get("thresholds") or [CALM_THRESHOLD]
        if len(thresholds) != 1:
            raise ValueError("candidate_first_crossing artifacts require exactly one frozen threshold")
        candidate_layers = fit_config.get("candidate_exit_layers") or CALM_CANDIDATE_LAYERS
        fit_config.setdefault(
            "candidate_first_crossing_semantics",
            default_candidate_first_crossing_semantics(
                candidate_layers=candidate_layers,
                threshold=float(thresholds[0]),
            ),
        )
        schema_version = LATEST_SCHEMA_VERSION if schema_version is None else schema_version
    elif source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
        thresholds = fit_config.get("thresholds") or []
        if len(thresholds) != 1:
            raise ValueError("official_free_calm artifacts require exactly one runtime threshold")
        candidate_layers = fit_config.get("candidate_exit_layers")
        if not candidate_layers:
            raise ValueError("official_free_calm artifacts require candidate_exit_layers")
        fit_config.setdefault(
            "official_free_calm_semantics",
            default_official_free_calm_semantics(
                candidate_layers=candidate_layers,
                threshold=float(thresholds[0]),
            ),
        )
        schema_version = LATEST_SCHEMA_VERSION if schema_version is None else schema_version
    if schema_version is None:
        schema_version = SCHEMA_VERSION
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": int(schema_version),
        "method_name": METHOD_NAME,
        "model_spec": dict(model_spec),
        "layer_index_semantics": dict(layer_index_semantics),
        "fit_config": fit_config,
        "policy_config": dict(policy_config),
        "parameters_by_threshold": _canonical_params(parameters_by_threshold),
        "provenance": dict(provenance),
    }
    artifact["provenance"].setdefault("creation_timestamp_utc", utc_now_iso())
    validate_phase3c_policy_artifact(artifact)
    return artifact


def _iter_tensors(value: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, MappingABC):
        for key, item in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            yield from _iter_tensors(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = "{}[{}]".format(prefix, idx)
            yield from _iter_tensors(item, child_prefix)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, MappingABC):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def _validate_tensor(tensor: Any, field_path: str, shape: Tuple[int, ...]) -> None:
    _require(isinstance(tensor, torch.Tensor), "{} must be a torch.Tensor".format(field_path))
    _require(tensor.device.type == "cpu", "{} must be on CPU".format(field_path))
    _require(tensor.dtype == torch.float32, "{} must be float32".format(field_path))
    _require(tuple(tensor.shape) == tuple(shape), "{} shape must be {}, got {}".format(field_path, shape, tuple(tensor.shape)))
    _require(torch.isfinite(tensor).all().item(), "{} must contain only finite values".format(field_path))


def _as_float_tensor(value: Any, field_path: str, *, validate_finite: bool = True) -> torch.Tensor:
    """Coerce an artifact/runtime value to a detached float32 tensor.

    ``validate_finite`` defaults to True, so every existing caller keeps the
    finite check unchanged. It may be set False ONLY by a caller that has
    already finite-validated this exact, unmodified tensor -- on CUDA the
    check is a ``.item()`` device synchronization, so repeating it on
    immutable data is pure stall. Type and dtype coercion are never skipped.
    """

    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a torch.Tensor".format(field_path))
    tensor = value.detach().to(dtype=torch.float32)
    if validate_finite and not torch.isfinite(tensor).all().item():
        raise ValueError("{} must contain only finite values".format(field_path))
    return tensor


def _require_shape(tensor: torch.Tensor, shape: Tuple[int, ...], field_path: str) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError("{} shape must be {}, got {}".format(field_path, shape, tuple(tensor.shape)))


def _require_trailing_shape(tensor: torch.Tensor, shape: Tuple[int, ...], field_path: str) -> None:
    if tensor.ndim < len(shape) or tuple(tensor.shape[-len(shape) :]) != tuple(shape):
        raise ValueError("{} trailing shape must be {}, got {}".format(field_path, shape, tuple(tensor.shape)))


def _threshold_key_set_from_fit_config(fit_config: Mapping[Any, Any]) -> set:
    thresholds = _require_required_mapping_field(fit_config, "thresholds", "fit_config")
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise ValueError("fit_config.thresholds must be a non-empty list")
    keys = set()
    for idx, threshold in enumerate(thresholds):
        if isinstance(threshold, bool):
            raise ValueError("fit_config.thresholds[{}] must be a finite float".format(idx))
        try:
            key = threshold_to_key(threshold)
        except Exception as exc:
            raise ValueError("fit_config.thresholds[{}] must be a finite float: {}".format(idx, exc)) from exc
        if key in keys:
            raise ValueError("fit_config.thresholds contains duplicate canonical threshold: {}".format(key))
        keys.add(key)
    return keys


def _configured_gap_bin_keys(fit_config: Mapping[Any, Any]) -> Tuple[list, set]:
    gap_bins = _require_required_mapping_field(fit_config, "gap_bins", "fit_config")
    if not isinstance(gap_bins, (list, tuple)) or not gap_bins:
        raise ValueError("fit_config.gap_bins must be a non-empty list")
    keys = set()
    ordered_labels = []
    previous_end: Optional[int] = None
    for idx, gap_bin in enumerate(gap_bins):
        path = "fit_config.gap_bins[{}]".format(idx)
        gap_bin = _require_mapping(gap_bin, path)
        _require_exact_keys(gap_bin, {"label", "start", "end"}, path)
        label = gap_bin["label"]
        if not isinstance(label, str) or not label:
            raise ValueError("{}.label must be a non-empty string".format(path))
        start = gap_bin["start"]
        end = gap_bin["end"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("{}.start must be an integer".format(path))
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("{}.end must be an integer".format(path))
        if start < 1:
            raise ValueError("{}.start must be >= 1".format(path))
        if end < start:
            raise ValueError("{}.end must be >= start".format(path))
        if previous_end is not None and start <= previous_end:
            raise ValueError("fit_config.gap_bins must be strictly increasing and non-overlapping")
        previous_end = end
        key = gap_bin_to_key(label)
        if key in keys:
            raise ValueError("fit_config.gap_bins contains duplicate canonical gap bin: {}".format(key))
        keys.add(key)
        ordered_labels.append(gap_bin_key_to_label(key))
    return ordered_labels, keys


def _normalized_gap_bins(gap_bins: Iterable[Any]) -> list:
    normalized = []
    for idx, gap_bin in enumerate(gap_bins):
        if isinstance(gap_bin, MappingABC):
            label = gap_bin.get("label")
            start = gap_bin.get("start")
            end = gap_bin.get("end")
        elif isinstance(gap_bin, (list, tuple)) and len(gap_bin) == 3:
            label, start, end = gap_bin
        else:
            raise ValueError("gap_bins[{}] must be a mapping or (label,start,end) tuple".format(idx))
        key = gap_bin_to_key(label)
        label = gap_bin_key_to_label(key)
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("gap_bins[{}].start must be an integer".format(idx))
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("gap_bins[{}].end must be an integer".format(idx))
        if start < 1 or end < start:
            raise ValueError("gap_bins[{}] has invalid range {}-{}".format(idx, start, end))
        normalized.append({"label": label, "key": key, "start": int(start), "end": int(end)})
    return normalized


def required_phase3c_runtime_coverage(
    *,
    thresholds: Iterable[Any],
    fixed_source_layer: int,
    decoder_layer_count: int,
    gap_bins: Iterable[Any],
) -> Dict[str, Any]:
    """Return fixed-layer runtime coverage required by Phase 3c Task C1.

    Same-layer target ``s -> s`` is intentionally excluded because runtime uses
    target-native exact projection for that layer.  Fitted maps are required
    only for deeper target layers.
    """

    source_layer = int(fixed_source_layer)
    layer_count = int(decoder_layer_count)
    if source_layer < 0 or source_layer >= layer_count:
        raise ValueError("fixed_source_layer must be within decoder depth")
    target_layers = list(range(source_layer + 1, layer_count))
    reachable_gaps = [target - source_layer for target in target_layers]
    normalized_bins = _normalized_gap_bins(gap_bins)
    required_gap_bins = [
        item
        for item in normalized_bins
        if reachable_gaps and item["start"] <= max(reachable_gaps) and item["end"] >= min(reachable_gaps)
    ]
    uncovered_reachable_gaps = [
        gap
        for gap in reachable_gaps
        if not any(item["start"] <= gap <= item["end"] for item in normalized_bins)
    ]
    pair_keys = [layer_pair_to_key(source_layer, target) for target in target_layers]
    threshold_payload = {}
    for threshold in thresholds:
        threshold_key = threshold_to_key(threshold)
        threshold_payload[threshold_key] = {
            "required_layer_pairs": list(pair_keys),
            "required_hidden_layer_pairs": list(pair_keys),
            "required_k_layer_pairs": list(pair_keys),
            "required_v_gap_bins": [item["label"] for item in required_gap_bins],
            "required_v_gap_bin_keys": [item["key"] for item in required_gap_bins],
            "required_deeper_target_layers": list(target_layers),
            "required_layer_pair_count": len(pair_keys),
        }
    return {
        "source_layer_mode": "fixed_layer",
        "fixed_source_layer": source_layer,
        "decoder_layer_count": layer_count,
        "same_layer_exact_projection_target": source_layer,
        "required_deeper_target_layers": list(target_layers),
        "required_layer_pairs": list(pair_keys),
        "required_layer_pair_count": len(pair_keys),
        "required_v_gap_bins": [item["label"] for item in required_gap_bins],
        "required_v_gap_bin_keys": [item["key"] for item in required_gap_bins],
        "reachable_gaps": list(reachable_gaps),
        "uncovered_reachable_gaps": uncovered_reachable_gaps,
        "thresholds": threshold_payload,
    }


def required_phase3c_candidate_first_crossing_coverage(
    *,
    thresholds: Iterable[Any],
    candidate_source_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    decoder_layer_count: int,
    gap_bins: Iterable[Any],
) -> Dict[str, Any]:
    """Return complete CALM multi-source runtime coverage requirements."""

    layer_count = int(decoder_layer_count)
    if layer_count <= 0:
        raise ValueError("decoder_layer_count must be positive")
    sources = [int(layer) for layer in candidate_source_layers]
    if not sources:
        raise ValueError("candidate_source_layers cannot be empty")
    if sources != sorted(sources):
        raise ValueError("candidate_source_layers must be sorted")
    if len(set(sources)) != len(sources):
        raise ValueError("candidate_source_layers must not contain duplicates")
    if any(layer < 0 or layer >= layer_count for layer in sources):
        raise ValueError("candidate_source_layers must be within decoder depth")

    normalized_bins = _normalized_gap_bins(gap_bins)
    by_source: Dict[str, Any] = {}
    all_pair_keys: List[str] = []
    reachable_gaps = set()
    for source_layer in sources:
        target_layers = list(range(source_layer + 1, layer_count))
        pair_keys = [layer_pair_to_key(source_layer, target) for target in target_layers]
        gaps = [target - source_layer for target in target_layers]
        reachable_gaps.update(gaps)
        all_pair_keys.extend(pair_keys)
        by_source[str(source_layer)] = {
            "source_layer": source_layer,
            "same_layer_exact_projection_target": source_layer,
            "required_deeper_target_layers": target_layers,
            "required_layer_pairs": pair_keys,
            "required_layer_pair_count": len(pair_keys),
            "reachable_gaps": gaps,
        }

    required_gap_bins = [
        item
        for item in normalized_bins
        if any(item["start"] <= gap <= item["end"] for gap in reachable_gaps)
    ]
    uncovered_reachable_gaps = [
        gap
        for gap in sorted(reachable_gaps)
        if not any(item["start"] <= gap <= item["end"] for item in normalized_bins)
    ]
    threshold_payload = {}
    for threshold in thresholds:
        threshold_key = threshold_to_key(threshold)
        threshold_payload[threshold_key] = {
            "candidate_source_layers": list(sources),
            "same_layer_exact_projection_targets": list(sources),
            "required_deeper_target_layers": sorted(
                {
                    target
                    for payload in by_source.values()
                    for target in payload["required_deeper_target_layers"]
                }
            ),
            "required_layer_pairs": list(all_pair_keys),
            "required_hidden_layer_pairs": list(all_pair_keys),
            "required_k_layer_pairs": list(all_pair_keys),
            "required_v_gap_bins": [item["label"] for item in required_gap_bins],
            "required_v_gap_bin_keys": [item["key"] for item in required_gap_bins],
            "required_layer_pair_count": len(all_pair_keys),
            "required_layer_pairs_by_source": {
                source: list(payload["required_layer_pairs"])
                for source, payload in sorted(by_source.items(), key=lambda item: int(item[0]))
            },
        }

    return {
        "source_layer_mode": SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
        "candidate_source_layers": list(sources),
        "decoder_layer_count": layer_count,
        "same_layer_exact_projection_targets": list(sources),
        "required_deeper_target_layers": sorted(
            {
                target
                for payload in by_source.values()
                for target in payload["required_deeper_target_layers"]
            }
        ),
        "required_layer_pairs": list(all_pair_keys),
        "required_layer_pair_count": len(all_pair_keys),
        "required_layer_pairs_by_source": by_source,
        "required_v_gap_bins": [item["label"] for item in required_gap_bins],
        "required_v_gap_bin_keys": [item["key"] for item in required_gap_bins],
        "reachable_gaps": sorted(reachable_gaps),
        "uncovered_reachable_gaps": uncovered_reachable_gaps,
        "thresholds": threshold_payload,
    }


def _runtime_coverage_validation_from_parts(
    model_spec: Mapping[Any, Any],
    fit_config: Mapping[Any, Any],
    params_by_threshold: Mapping[Any, Any],
) -> Dict[str, Any]:
    source_layer_mode = fit_config.get("source_layer_mode")
    decoder_layer_count = int(model_spec.get("decoder_layer_count", 0) or 0)
    fixed_source_layer = fit_config.get("fixed_source_layer")
    base = {
        "status": "not_applicable",
        "source_layer_mode": source_layer_mode,
        "fixed_source_layer": fixed_source_layer,
        "candidate_source_layers": [],
        "decoder_layer_count": decoder_layer_count,
        "same_layer_exact_projection_target": fixed_source_layer,
        "same_layer_exact_projection_targets": [],
        "required_deeper_target_layers": [],
        "required_layer_pair_count": 0,
        "hidden_layer_pair_count": 0,
        "k_layer_pair_count": 0,
        "missing_hidden_layer_pairs": [],
        "missing_k_layer_pairs": [],
        "extra_hidden_layer_pairs": [],
        "extra_k_layer_pairs": [],
        "required_v_gap_bins": [],
        "uncovered_reachable_gaps": [],
        "present_v_gap_bins": [],
        "missing_v_gap_bins": [],
        "extra_v_gap_bins": [],
        "errors": [],
        "thresholds": {},
    }
    if source_layer_mode not in {
        SOURCE_LAYER_MODE_FIXED,
        SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
        SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
    }:
        return base

    errors = []
    thresholds = _require_required_mapping_field(fit_config, "thresholds", "fit_config")
    if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        try:
            fixed_source_layer_int = int(fixed_source_layer)
        except Exception:
            base["status"] = "failed"
            base["errors"] = ["fixed_source_layer_invalid"]
            return base
        coverage = required_phase3c_runtime_coverage(
            thresholds=thresholds,
            fixed_source_layer=fixed_source_layer_int,
            decoder_layer_count=decoder_layer_count,
            gap_bins=fit_config.get("gap_bins") or [],
        )
        base.update(
            {
                "fixed_source_layer": fixed_source_layer_int,
                "same_layer_exact_projection_target": fixed_source_layer_int,
                "same_layer_exact_projection_targets": [fixed_source_layer_int],
            }
        )
    else:
        candidate_layers = fit_config.get("candidate_exit_layers") or fit_config.get("candidate_source_layers")
        if candidate_layers is None:
            semantics = fit_config.get("candidate_first_crossing_semantics")
            candidate_layers = semantics.get("candidate_exit_layers") if isinstance(semantics, MappingABC) else None
        try:
            coverage = required_phase3c_candidate_first_crossing_coverage(
                thresholds=thresholds,
                candidate_source_layers=candidate_layers or CALM_CANDIDATE_LAYERS,
                decoder_layer_count=decoder_layer_count,
                gap_bins=fit_config.get("gap_bins") or [],
            )
        except Exception as exc:
            base["status"] = "failed"
            base["errors"] = ["candidate_first_crossing_runtime_coverage_invalid:{}".format(str(exc))]
            return base
        base.update(
            {
                "candidate_source_layers": coverage["candidate_source_layers"],
                "same_layer_exact_projection_target": None,
                "same_layer_exact_projection_targets": coverage["same_layer_exact_projection_targets"],
                "required_layer_pairs_by_source": coverage["required_layer_pairs_by_source"],
            }
        )
    base.update(
        {
            "status": "ok",
            "required_deeper_target_layers": coverage["required_deeper_target_layers"],
            "required_layer_pair_count": coverage["required_layer_pair_count"],
            "required_v_gap_bins": coverage["required_v_gap_bins"],
            "uncovered_reachable_gaps": coverage["uncovered_reachable_gaps"],
        }
    )
    if coverage.get("uncovered_reachable_gaps"):
        errors.append("uncovered_runtime_reachable_gaps")
    aggregate_missing_hidden = set()
    aggregate_missing_k = set()
    aggregate_extra_hidden = set()
    aggregate_extra_k = set()
    aggregate_missing_v = set()
    aggregate_extra_v = set()
    present_v_labels_all = set()
    hidden_counts = []
    k_counts = []
    for threshold_key, expected in coverage["thresholds"].items():
        payload = params_by_threshold.get(threshold_key) if isinstance(params_by_threshold, MappingABC) else None
        if not isinstance(payload, MappingABC):
            errors.append("missing_threshold_runtime_coverage:{}".format(threshold_key))
            present_hidden = set()
            present_k = set()
            present_v = set()
        else:
            present_hidden = set(_require_mapping(payload.get("hidden_by_layer_pair"), "{}.hidden_by_layer_pair".format(threshold_key)).keys())
            present_k = set(_require_mapping(payload.get("k_by_layer_pair"), "{}.k_by_layer_pair".format(threshold_key)).keys())
            present_v = set(_require_mapping(payload.get("v_by_gap_bin"), "{}.v_by_gap_bin".format(threshold_key)).keys())
        required_pairs = set(expected["required_layer_pairs"])
        required_v_keys = set(expected["required_v_gap_bin_keys"])
        missing_hidden = sorted(required_pairs - present_hidden)
        missing_k = sorted(required_pairs - present_k)
        extra_hidden = sorted(present_hidden - required_pairs)
        extra_k = sorted(present_k - required_pairs)
        missing_v_keys = sorted(required_v_keys - present_v)
        extra_v_keys = sorted(present_v - required_v_keys)
        missing_v_labels = [gap_bin_key_to_label(item) for item in missing_v_keys]
        extra_v_labels = [gap_bin_key_to_label(item) for item in extra_v_keys]
        present_v_labels = [gap_bin_key_to_label(item) for item in sorted(present_v)]
        present_v_labels_all.update(present_v_labels)
        hidden_counts.append(len(present_hidden))
        k_counts.append(len(present_k))
        aggregate_missing_hidden.update(missing_hidden)
        aggregate_missing_k.update(missing_k)
        aggregate_extra_hidden.update(extra_hidden)
        aggregate_extra_k.update(extra_k)
        aggregate_missing_v.update(missing_v_labels)
        aggregate_extra_v.update(extra_v_labels)
        if missing_hidden:
            errors.append("missing_runtime_hidden_layer_pairs:{}".format(threshold_key))
        if missing_k:
            errors.append("missing_runtime_k_layer_pairs:{}".format(threshold_key))
        if extra_hidden:
            errors.append("extra_runtime_hidden_layer_pairs:{}".format(threshold_key))
        if extra_k:
            errors.append("extra_runtime_k_layer_pairs:{}".format(threshold_key))
        if missing_v_labels:
            errors.append("missing_runtime_v_gap_bins:{}".format(threshold_key))
        if extra_v_labels:
            errors.append("extra_runtime_v_gap_bins:{}".format(threshold_key))
        base["thresholds"][threshold_key] = {
            "status": "ok"
            if not (missing_hidden or missing_k or extra_hidden or extra_k or missing_v_labels or extra_v_labels)
            else "failed",
            "required_layer_pair_count": len(expected["required_layer_pairs"]),
            "required_deeper_target_layers": expected["required_deeper_target_layers"],
            "expected_runtime_layer_pairs": expected["required_layer_pairs"],
            "present_hidden_layer_pairs": sorted(present_hidden),
            "present_k_layer_pairs": sorted(present_k),
            "missing_hidden_layer_pairs": missing_hidden,
            "missing_k_layer_pairs": missing_k,
            "extra_hidden_layer_pairs": extra_hidden,
            "extra_k_layer_pairs": extra_k,
            "required_v_gap_bins": expected["required_v_gap_bins"],
            "present_v_gap_bins": present_v_labels,
            "missing_v_gap_bins": missing_v_labels,
            "extra_v_gap_bins": extra_v_labels,
            "hidden_layer_pair_count": len(present_hidden),
            "k_layer_pair_count": len(present_k),
        }
    base.update(
        {
            "status": "ok" if not errors else "failed",
            "errors": errors,
            "hidden_layer_pair_count": hidden_counts[0] if len(set(hidden_counts)) == 1 and hidden_counts else sum(hidden_counts),
            "k_layer_pair_count": k_counts[0] if len(set(k_counts)) == 1 and k_counts else sum(k_counts),
            "missing_hidden_layer_pairs": sorted(aggregate_missing_hidden),
            "missing_k_layer_pairs": sorted(aggregate_missing_k),
            "extra_hidden_layer_pairs": sorted(aggregate_extra_hidden),
            "extra_k_layer_pairs": sorted(aggregate_extra_k),
            "present_v_gap_bins": sorted(present_v_labels_all),
            "missing_v_gap_bins": sorted(aggregate_missing_v),
            "extra_v_gap_bins": sorted(aggregate_extra_v),
        }
    )
    return base


def phase3c_runtime_coverage_validation(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fixed-layer runtime map coverage diagnostics for an artifact."""

    artifact = _require_mapping(artifact, "artifact")
    return _runtime_coverage_validation_from_parts(
        _require_mapping(artifact.get("model_spec"), "model_spec"),
        _require_mapping(artifact.get("fit_config"), "fit_config"),
        _require_mapping(artifact.get("parameters_by_threshold"), "parameters_by_threshold"),
    )


def phase3c_fixed_source_subset_coverage_validation(
    artifact: Mapping[str, Any],
    *,
    threshold: Any,
    fixed_source_layer: int,
    decoder_layer_count: int,
) -> Dict[str, Any]:
    """Validate only one fixed-source runtime slice of a Phase 3c artifact.

    This is deliberately narrower than :func:`phase3c_runtime_coverage_validation`:
    additional layer-pair maps and V gap bins are permitted.  It is used by
    the explicit development-only Native FREE connection that consumes the
    source-6 slice of an otherwise candidate-first-crossing artifact without
    reinterpreting or mutating the artifact itself.
    """

    artifact = _require_mapping(artifact, "artifact")
    model_spec = _require_mapping(artifact.get("model_spec"), "model_spec")
    fit_config = _require_mapping(artifact.get("fit_config"), "fit_config")
    params_by_threshold = _require_mapping(
        artifact.get("parameters_by_threshold"), "parameters_by_threshold"
    )
    source_layer = int(fixed_source_layer)
    expected_layer_count = int(decoder_layer_count)
    artifact_layer_count = int(model_spec.get("decoder_layer_count", 0) or 0)
    threshold_key = threshold_to_key(threshold)
    target_layers = list(range(source_layer + 1, expected_layer_count))
    required_pairs = [layer_pair_to_key(source_layer, target) for target in target_layers]
    errors: List[str] = []

    if artifact_layer_count != expected_layer_count:
        errors.append(
            "decoder_layer_count_mismatch:expected={}:actual={}".format(
                expected_layer_count, artifact_layer_count
            )
        )
    for field in ("d_model", "num_heads", "d_kv"):
        try:
            if int(model_spec.get(field, 0) or 0) <= 0:
                errors.append("model_spec_{}_invalid".format(field))
        except Exception:
            errors.append("model_spec_{}_invalid".format(field))

    source_mode = fit_config.get("source_layer_mode")
    candidate_layers: List[int] = []
    if source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        raw_layers = fit_config.get("candidate_exit_layers") or fit_config.get(
            "candidate_source_layers"
        )
        if raw_layers is None:
            semantics = fit_config.get("candidate_first_crossing_semantics")
            raw_layers = (
                semantics.get("candidate_exit_layers")
                if isinstance(semantics, MappingABC)
                else None
            )
        candidate_layers = [int(item) for item in (raw_layers or [])]
        if source_layer not in candidate_layers:
            errors.append("fixed_source_layer_not_in_candidate_source_layers")
    elif source_mode == SOURCE_LAYER_MODE_FIXED:
        try:
            artifact_fixed = int(fit_config.get("fixed_source_layer"))
        except Exception:
            artifact_fixed = None
        if artifact_fixed != source_layer:
            errors.append("fixed_source_layer_mismatch")
    else:
        errors.append("unsupported_source_layer_mode:{}".format(source_mode))

    payload = params_by_threshold.get(threshold_key)
    if not isinstance(payload, MappingABC):
        errors.append("missing_artifact_threshold:{}".format(threshold_key))
        present_hidden = set()
        present_k = set()
        present_v = set()
    else:
        present_hidden = set(
            _require_mapping(
                payload.get("hidden_by_layer_pair"),
                "{}.hidden_by_layer_pair".format(threshold_key),
            ).keys()
        )
        present_k = set(
            _require_mapping(
                payload.get("k_by_layer_pair"),
                "{}.k_by_layer_pair".format(threshold_key),
            ).keys()
        )
        present_v = set(
            _require_mapping(
                payload.get("v_by_gap_bin"),
                "{}.v_by_gap_bin".format(threshold_key),
            ).keys()
        )

    missing_hidden = sorted(set(required_pairs) - present_hidden)
    missing_k = sorted(set(required_pairs) - present_k)
    if missing_hidden:
        errors.append("missing_source_subset_hidden_layer_pairs")
    if missing_k:
        errors.append("missing_source_subset_k_layer_pairs")

    normalized_bins = _normalized_gap_bins(fit_config.get("gap_bins") or [])
    required_v_keys = set()
    uncovered_gaps = []
    for gap in range(1, expected_layer_count - source_layer):
        matching = [item for item in normalized_bins if item["start"] <= gap <= item["end"]]
        if not matching:
            uncovered_gaps.append(gap)
            continue
        required_v_keys.add(matching[0]["key"])
    missing_v_keys = sorted(required_v_keys - present_v)
    if uncovered_gaps:
        errors.append("uncovered_source_subset_reachable_gaps")
    if missing_v_keys:
        errors.append("missing_source_subset_v_gap_bins")

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "source_layer_mode": source_mode,
        "candidate_source_layers": candidate_layers,
        "runtime_source_mode": SOURCE_LAYER_MODE_FIXED,
        "runtime_fixed_source_layer": source_layer,
        "decoder_layer_count": artifact_layer_count,
        "threshold": float(threshold),
        "threshold_key": threshold_key,
        "same_layer_exact_projection_target": source_layer,
        "required_deeper_target_layers": target_layers,
        "required_layer_pairs": required_pairs,
        "missing_hidden_layer_pairs": missing_hidden,
        "missing_k_layer_pairs": missing_k,
        "required_v_gap_bins": [gap_bin_key_to_label(item) for item in sorted(required_v_keys)],
        "missing_v_gap_bins": [gap_bin_key_to_label(item) for item in missing_v_keys],
        "uncovered_reachable_gaps": uncovered_gaps,
        "extra_layer_pairs_allowed": True,
        "extra_v_gap_bins_allowed": True,
    }


def _v_gap_key_sets_by_threshold_from_fit_config(fit_config: Mapping[Any, Any], declared_threshold_keys: set) -> Dict[str, set]:
    if "exported_v_gap_bins" in fit_config:
        raise ValueError("fit_config.exported_v_gap_bins is unsupported; use fit_config.exported_v_gap_bins_by_threshold")
    ordered_gap_labels, configured_gap_keys = _configured_gap_bin_keys(fit_config)
    configured_order = {gap_bin_to_key(label): idx for idx, label in enumerate(ordered_gap_labels)}
    exported = _require_required_mapping_field(fit_config, "exported_v_gap_bins_by_threshold", "fit_config")
    exported = _require_mapping(exported, "fit_config.exported_v_gap_bins_by_threshold")
    exported_keys = set()
    result: Dict[str, set] = {}
    for raw_threshold_key, labels in exported.items():
        threshold_key = threshold_to_key(raw_threshold_key)
        if threshold_key != raw_threshold_key:
            raise ValueError(
                "fit_config.exported_v_gap_bins_by_threshold has noncanonical threshold key {!r}; expected {!r}".format(
                    raw_threshold_key, threshold_key
                )
            )
        if threshold_key in exported_keys:
            raise ValueError("fit_config.exported_v_gap_bins_by_threshold contains duplicate threshold key: {}".format(threshold_key))
        exported_keys.add(threshold_key)
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError("fit_config.exported_v_gap_bins_by_threshold.{} must be a non-empty list".format(threshold_key))
        gap_keys = set()
        previous_order: Optional[int] = None
        for idx, label in enumerate(labels):
            gap_key = gap_bin_to_key(label)
            if gap_key not in configured_gap_keys:
                raise ValueError(
                    "fit_config.exported_v_gap_bins_by_threshold.{}[{}] gap label {!r} is not present in fit_config.gap_bins".format(
                        threshold_key, idx, label
                    )
                )
            if gap_key in gap_keys:
                raise ValueError(
                    "fit_config.exported_v_gap_bins_by_threshold.{} contains duplicate canonical gap bin: {}".format(
                        threshold_key, gap_key
                    )
                )
            current_order = configured_order[gap_key]
            if previous_order is not None and current_order <= previous_order:
                raise ValueError(
                    "fit_config.exported_v_gap_bins_by_threshold.{} must follow fit_config.gap_bins order".format(
                        threshold_key
                    )
                )
            previous_order = current_order
            gap_keys.add(gap_key)
        result[threshold_key] = gap_keys
    if exported_keys != declared_threshold_keys:
        missing = sorted(declared_threshold_keys - exported_keys, key=threshold_key_to_float)
        extra = sorted(exported_keys - declared_threshold_keys, key=threshold_key_to_float)
        raise ValueError(
            "fit_config.exported_v_gap_bins_by_threshold keys do not match fit_config.thresholds: missing {}, extra {}".format(
                missing, extra
            )
        )
    return result


def validate_exit_confidence_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Return True only for the supported FREE softmax-margin confidence metadata.

    Missing metadata is accepted for old plumbing-only artifacts and returns
    False. Malformed structure raises ``ValueError``; structured but unsupported
    semantics return False so runtime can mark the artifact unverified.
    """

    if "exit_confidence_semantics" not in fit_config or fit_config.get("exit_confidence_semantics") is None:
        return False
    semantics = _require_mapping(fit_config["exit_confidence_semantics"], "fit_config.exit_confidence_semantics")
    _metadata_json_safe(semantics, "fit_config.exit_confidence_semantics")
    legacy_required = {
        "version",
        "confidence_type",
        "threshold_comparator",
        "hidden_position",
        "projection_path",
        "tie_word_embedding_scale",
        "runtime_framework",
    }
    current_required = legacy_required | {"confidence_compute_dtype"}
    version = semantics.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("fit_config.exit_confidence_semantics.version must be an integer")
    if version == EXIT_CONFIDENCE_LEGACY_SEMANTICS_VERSION:
        _require_exact_keys(semantics, legacy_required, "fit_config.exit_confidence_semantics")
        return False
    if version != EXIT_CONFIDENCE_SEMANTICS_VERSION:
        return False
    _require_exact_keys(semantics, current_required, "fit_config.exit_confidence_semantics")
    return bool(
        semantics.get("confidence_type") == EXIT_CONFIDENCE_TYPE
        and semantics.get("threshold_comparator") == EXIT_CONFIDENCE_THRESHOLD_COMPARATOR
        and semantics.get("hidden_position") == EXIT_CONFIDENCE_HIDDEN_POSITION
        and semantics.get("projection_path") == EXIT_CONFIDENCE_PROJECTION_PATH
        and semantics.get("tie_word_embedding_scale") == EXIT_CONFIDENCE_TIE_SCALE
        and semantics.get("runtime_framework") == EXIT_CONFIDENCE_RUNTIME_FRAMEWORK
        and semantics.get("confidence_compute_dtype") == EXIT_CONFIDENCE_COMPUTE_DTYPE
    )


def validate_fixed_layer_membership_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Return True only for explicit fixed-source-layer population metadata."""

    if "fixed_layer_membership_semantics" not in fit_config or fit_config.get("fixed_layer_membership_semantics") is None:
        return False
    semantics = _require_mapping(
        fit_config["fixed_layer_membership_semantics"],
        "fit_config.fixed_layer_membership_semantics",
    )
    _metadata_json_safe(semantics, "fit_config.fixed_layer_membership_semantics")
    required = {
        "version",
        "source_layer_mode",
        "membership_layer",
        "earlier_layer_confidence_policy",
    }
    _require_exact_keys(semantics, required, "fit_config.fixed_layer_membership_semantics")
    return bool(
        semantics.get("version") == FIXED_LAYER_MEMBERSHIP_SEMANTICS_VERSION
        and semantics.get("source_layer_mode") == SOURCE_LAYER_MODE_FIXED
        and semantics.get("membership_layer") == FIXED_LAYER_MEMBERSHIP_LAYER
        and semantics.get("earlier_layer_confidence_policy") == FIXED_LAYER_EARLIER_CONFIDENCE_POLICY
    )


def validate_fixed_layer_policy_identity(
    artifact: Mapping[str, Any],
    *,
    require_authoritative: bool = False,
) -> Dict[str, Any]:
    """Validate stored fixed-layer policy semantics and SHA without requiring tensors."""

    errors: List[str] = []
    if not isinstance(artifact, MappingABC):
        return {"status": "failed", "errors": ["artifact_not_mapping"], "policy_sha256": None}
    fit_config = artifact.get("fit_config")
    policy_config = artifact.get("policy_config")
    if not isinstance(fit_config, MappingABC):
        return {"status": "failed", "errors": ["fit_config_missing"], "policy_sha256": None}
    if fit_config.get("source_layer_mode") != SOURCE_LAYER_MODE_FIXED:
        return {
            "status": "failed",
            "errors": ["source_layer_mode_not_fixed_layer"],
            "policy_sha256": None,
            "source_layer_mode": fit_config.get("source_layer_mode"),
        }
    stored_payload = fit_config.get("fixed_layer_policy_semantics")
    stored_sha = fit_config.get("policy_sha256")
    if not isinstance(stored_payload, MappingABC):
        errors.append("fixed_layer_policy_semantics_missing")
        stored_payload = {}
    elif not stored_payload:
        errors.append("fixed_layer_policy_semantics_malformed")
    if stored_sha in (None, ""):
        errors.append("fixed_layer_policy_sha256_missing")
    try:
        expected_payload = fixed_layer_policy_semantics_payload(fit_config, policy_config)
        expected_sha = canonical_json_sha256(expected_payload)
    except Exception as exc:
        return {
            "status": "failed",
            "errors": sorted(set(errors + ["fixed_layer_policy_recompute_failed"])),
            "error_message": str(exc),
            "policy_sha256": None if stored_sha in (None, "") else str(stored_sha),
        }
    if stored_payload:
        payload_without_sha = dict(stored_payload)
        embedded_payload_sha = payload_without_sha.pop("policy_sha256", None)
        if payload_without_sha != expected_payload:
            errors.append("fixed_layer_policy_payload_mismatch")
        if embedded_payload_sha in (None, ""):
            errors.append("fixed_layer_policy_embedded_sha256_missing")
        elif str(embedded_payload_sha) != expected_sha:
            errors.append("fixed_layer_policy_embedded_sha256_mismatch")
    if stored_sha not in (None, "") and str(stored_sha) != expected_sha:
        errors.append("fixed_layer_policy_sha256_mismatch")
    if require_authoritative and errors:
        errors.append("fixed_layer_policy_identity_not_authoritative")
    return {
        "status": "ok" if not errors else "failed",
        "errors": sorted(set(errors)),
        "source_layer_mode": SOURCE_LAYER_MODE_FIXED,
        "policy_type": FIXED_LAYER_POLICY_TYPE,
        "policy_schema_version": FIXED_LAYER_POLICY_SEMANTICS_VERSION,
        "policy_sha256": expected_sha if not errors else (None if stored_sha in (None, "") else str(stored_sha)),
        "recomputed_policy_sha256": expected_sha,
        "stored_policy_sha256": None if stored_sha in (None, "") else str(stored_sha),
        "require_authoritative": bool(require_authoritative),
    }


def artifact_policy_identity_validation(
    artifact: Mapping[str, Any],
    *,
    require_authoritative: bool = False,
) -> Dict[str, Any]:
    """Return the authoritative policy identity validation for supported artifacts."""

    if not isinstance(artifact, MappingABC):
        return {"status": "failed", "errors": ["artifact_not_mapping"], "policy_sha256": None}
    fit_config = artifact.get("fit_config")
    if not isinstance(fit_config, MappingABC):
        return {"status": "failed", "errors": ["fit_config_missing"], "policy_sha256": None}
    source_layer_mode = fit_config.get("source_layer_mode")
    if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        return validate_fixed_layer_policy_identity(artifact, require_authoritative=require_authoritative)
    if source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        errors: List[str] = []
        try:
            valid = validate_candidate_first_crossing_semantics(fit_config)
        except Exception as exc:
            return {
                "status": "failed",
                "errors": ["candidate_first_crossing_policy_validation_exception"],
                "error_message": str(exc),
                "source_layer_mode": source_layer_mode,
                "policy_sha256": None,
            }
        semantics = fit_config.get("candidate_first_crossing_semantics") or {}
        policy_sha = semantics.get("policy_sha256") if isinstance(semantics, MappingABC) else None
        if not valid:
            errors.append("candidate_first_crossing_semantics_invalid")
        if policy_sha in (None, ""):
            errors.append("candidate_first_crossing_policy_sha256_missing")
        if require_authoritative and errors:
            errors.append("candidate_first_crossing_policy_identity_not_authoritative")
        return {
            "status": "ok" if not errors else "failed",
            "errors": sorted(set(errors)),
            "source_layer_mode": source_layer_mode,
            "policy_sha256": None if policy_sha in (None, "") else str(policy_sha),
            "require_authoritative": bool(require_authoritative),
        }
    if source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
        # Official FREE CALM counterpart of the candidate-first-crossing
        # branch above: the policy identity is the canonical policy_sha256
        # already carried inside the (structurally validated)
        # official_free_calm_semantics payload -- read generically, never a
        # hard-coded experiment-specific SHA.
        errors = []
        try:
            valid = validate_official_free_calm_semantics(fit_config)
        except Exception as exc:
            return {
                "status": "failed",
                "errors": ["official_free_calm_policy_validation_exception"],
                "error_message": str(exc),
                "source_layer_mode": source_layer_mode,
                "policy_sha256": None,
            }
        semantics = fit_config.get("official_free_calm_semantics") or {}
        policy_sha = semantics.get("policy_sha256") if isinstance(semantics, MappingABC) else None
        if not valid:
            errors.append("official_free_calm_semantics_invalid")
        if policy_sha in (None, ""):
            errors.append("official_free_calm_policy_sha256_missing")
        if require_authoritative and errors:
            errors.append("official_free_calm_policy_identity_not_authoritative")
        return {
            "status": "ok" if not errors else "failed",
            "errors": sorted(set(errors)),
            "source_layer_mode": source_layer_mode,
            "policy_sha256": None if policy_sha in (None, "") else str(policy_sha),
            "require_authoritative": bool(require_authoritative),
        }
    return {
        "status": "failed",
        "errors": ["artifact_policy_identity_source_layer_mode_unsupported"],
        "source_layer_mode": source_layer_mode,
        "policy_sha256": None,
        "require_authoritative": bool(require_authoritative),
    }


def artifact_policy_sha256(
    artifact: Mapping[str, Any],
    *,
    require_authoritative: bool = False,
) -> Optional[str]:
    """Resolve the canonical artifact policy SHA for fixed-layer or CALM artifacts."""

    validation = artifact_policy_identity_validation(
        artifact,
        require_authoritative=bool(require_authoritative),
    )
    if validation.get("status") == "ok":
        return None if validation.get("policy_sha256") in (None, "") else str(validation.get("policy_sha256"))
    if require_authoritative:
        raise ValueError(
            "artifact_policy_identity_validation_failed: {}".format(
                ",".join(validation.get("errors") or ["unknown"])
            )
        )
    return None


def validate_candidate_first_crossing_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Return True only for the frozen CALM candidate-first-crossing metadata."""

    if fit_config.get("source_layer_mode") != SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        return False
    semantics = _require_mapping(
        fit_config.get("candidate_first_crossing_semantics"),
        "fit_config.candidate_first_crossing_semantics",
    )
    _metadata_json_safe(semantics, "fit_config.candidate_first_crossing_semantics")
    required = {
        "version",
        "source_layer_mode",
        "policy_name",
        "candidate_exit_layers",
        "candidate_evaluation_order",
        "policy",
        "policy_sha256",
        "confidence_type",
        "confidence_compute_dtype",
        "threshold",
        "threshold_comparator",
        "adaptive_threshold",
        "trace_source",
        "same_layer_projection_policy",
    }
    _require_exact_keys(semantics, required, "fit_config.candidate_first_crossing_semantics")
    candidate_layers = [int(item) for item in semantics.get("candidate_exit_layers")]
    threshold = float(semantics.get("threshold"))
    expected = default_candidate_first_crossing_semantics(
        candidate_layers=CALM_CANDIDATE_LAYERS,
        threshold=CALM_THRESHOLD,
    )
    return bool(
        semantics.get("version") == CANDIDATE_FIRST_CROSSING_SEMANTICS_VERSION
        and semantics.get("source_layer_mode") == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
        and semantics.get("policy_name") == CALM_POLICY_NAME
        and candidate_layers == list(CALM_CANDIDATE_LAYERS)
        and semantics.get("candidate_evaluation_order") == list(CALM_CANDIDATE_LAYERS)
        and abs(threshold - float(CALM_THRESHOLD)) <= 1e-12
        and semantics.get("threshold_comparator") == CALM_THRESHOLD_COMPARATOR
        and semantics.get("confidence_type") == CALM_CONFIDENCE_TYPE
        and semantics.get("confidence_compute_dtype") == CALM_CONFIDENCE_COMPUTE_DTYPE
        and semantics.get("adaptive_threshold") is CALM_USE_ADAPT_THRESHOLD
        and semantics.get("same_layer_projection_policy") == SAME_LAYER_PROJECTION_POLICY
        and semantics.get("policy") == expected["policy"]
        and semantics.get("policy_sha256") == expected["policy_sha256"]
    )


def validate_official_free_calm_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Official FREE CALM counterpart of
    ``validate_candidate_first_crossing_semantics``: same required-key/schema
    shape, but validates STRUCTURAL properties (contiguous ascending
    candidate range, a real finite threshold, the original FREE strict-gt
    comparator and confidence semantics) instead of exact equality to the
    frozen historical (4,6,8,10)/0.9 tuple. Never touches, weakens, or is
    consulted by ``validate_candidate_first_crossing_semantics`` above."""

    if fit_config.get("source_layer_mode") != SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
        return False
    semantics = _require_mapping(
        fit_config.get("official_free_calm_semantics"),
        "fit_config.official_free_calm_semantics",
    )
    _metadata_json_safe(semantics, "fit_config.official_free_calm_semantics")
    required = {
        "version",
        "source_layer_mode",
        "policy_name",
        "candidate_exit_layers",
        "candidate_evaluation_order",
        "policy",
        "policy_sha256",
        "confidence_type",
        "confidence_compute_dtype",
        "threshold",
        "threshold_comparator",
        "adaptive_threshold",
        "trace_source",
        "same_layer_projection_policy",
    }
    _require_exact_keys(semantics, required, "fit_config.official_free_calm_semantics")

    candidate_layers = semantics.get("candidate_exit_layers")
    if not isinstance(candidate_layers, list) or not candidate_layers:
        return False
    try:
        candidate_layers = [int(item) for item in candidate_layers]
    except Exception:
        return False
    if candidate_layers != list(range(candidate_layers[0], candidate_layers[0] + len(candidate_layers))):
        return False
    if any(layer < 0 for layer in candidate_layers):
        return False
    try:
        threshold = float(semantics.get("threshold"))
    except Exception:
        return False
    if threshold != threshold or threshold in (float("inf"), float("-inf")):
        return False
    expected = default_official_free_calm_semantics(candidate_layers=candidate_layers, threshold=threshold)
    return bool(
        semantics.get("version") == CANDIDATE_FIRST_CROSSING_SEMANTICS_VERSION
        and semantics.get("source_layer_mode") == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM
        and semantics.get("policy_name") == OFFICIAL_FREE_CALM_POLICY_NAME
        and semantics.get("candidate_evaluation_order") == candidate_layers
        and semantics.get("threshold_comparator") == OFFICIAL_FREE_CALM_THRESHOLD_COMPARATOR
        and semantics.get("confidence_type") == OFFICIAL_FREE_CALM_CONFIDENCE_TYPE
        and semantics.get("confidence_compute_dtype") == OFFICIAL_FREE_CALM_CONFIDENCE_COMPUTE_DTYPE
        and semantics.get("adaptive_threshold") is OFFICIAL_FREE_CALM_USE_ADAPT_THRESHOLD
        and semantics.get("same_layer_projection_policy") == SAME_LAYER_PROJECTION_POLICY
        and semantics.get("policy") == expected["policy"]
        and semantics.get("policy_sha256") == expected["policy_sha256"]
    )


def validate_calm_hybrid_policy_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Validate the frozen source-specific hybrid calibration policy, when present.

    Only called when ``fit_config`` claims to be a hybrid artifact (i.e.
    ``calm_hybrid_policy_semantics`` is present) -- non-hybrid
    ``candidate_first_crossing`` artifacts (fixed-layer, single-mode
    multi-source, or the source-6 comparison output) never set this key and
    are therefore unaffected. Fails closed (raises ``ValueError`` via the
    ``_require*`` helpers, or returns ``False``) when the map is missing,
    incomplete, differs from the frozen map, the hybrid SHA is missing or
    mismatched, or the embedded CALM runtime policy identity does not match
    the artifact's own (already-validated) ``candidate_first_crossing_semantics``.
    """

    semantics = _require_mapping(
        fit_config.get("calm_hybrid_policy_semantics"),
        "fit_config.calm_hybrid_policy_semantics",
    )
    _metadata_json_safe(semantics, "fit_config.calm_hybrid_policy_semantics")
    required = {
        "hybrid_policy_schema_version",
        "hybrid_policy_name",
        "candidate_source_layers",
        "source_population_mode_map",
        "first_crossing_only_semantics",
        "counterfactual_reachable_semantics",
        "calm_runtime_policy_sha256",
    }
    _require_exact_keys(semantics, required, "fit_config.calm_hybrid_policy_semantics")
    candidate_layers = [int(item) for item in semantics.get("candidate_source_layers", [])]
    expected = calm_hybrid_calibration_policy_payload(candidate_layers=CALM_CANDIDATE_LAYERS, threshold=CALM_THRESHOLD)
    if semantics != expected or candidate_layers != list(CALM_CANDIDATE_LAYERS):
        return False
    if semantics.get("hybrid_policy_name") != CALM_HYBRID_POLICY_NAME:
        return False
    stored_sha = fit_config.get("calm_hybrid_policy_sha256")
    expected_sha = calm_hybrid_calibration_policy_sha256(candidate_layers=CALM_CANDIDATE_LAYERS, threshold=CALM_THRESHOLD)
    if stored_sha != expected_sha:
        return False
    # Internal consistency: the hybrid payload's own embedded CALM runtime
    # policy identity must match the artifact's already-validated
    # candidate_first_crossing_semantics.policy_sha256 -- the frozen CALM
    # runtime policy identity must not silently differ between the two.
    runtime_semantics = fit_config.get("candidate_first_crossing_semantics") or {}
    if semantics.get("calm_runtime_policy_sha256") != runtime_semantics.get("policy_sha256"):
        return False
    return True


def require_calm_hybrid_policy_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that ``artifact`` explicitly declares the frozen hybrid policy.

    Raises ``ValueError`` if ``fit_config.calm_hybrid_policy_semantics`` is
    entirely absent (i.e. this is not a hybrid artifact at all) or fails
    ``validate_calm_hybrid_policy_semantics``. Intended for the dedicated
    hybrid fitting/export/load/replay path, which must never silently accept
    a non-hybrid artifact as if it were the hybrid one.
    """

    artifact = _require_mapping(artifact, "artifact")
    fit_config = _require_mapping(artifact.get("fit_config"), "artifact.fit_config")
    if fit_config.get("calm_hybrid_policy_semantics") is None:
        raise ValueError("calm_hybrid_policy_semantics_missing: artifact does not declare the frozen hybrid policy")
    _require(
        validate_calm_hybrid_policy_semantics(fit_config),
        "calm_hybrid_policy_semantics mismatch",
    )


def validate_calm_source4_reachable_ablation_policy_semantics(fit_config: Mapping[str, Any]) -> bool:
    """Validate the source-4-reachable fitting ablation policy, when present.

    Mirrors ``validate_calm_hybrid_policy_semantics`` exactly, but against the
    distinct ablation policy name/source map
    (``CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME`` /
    ``CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP``) and the
    distinct ``fit_config.calm_source4_reachable_ablation_policy_semantics`` /
    ``calm_source4_reachable_ablation_policy_sha256`` keys -- never the
    canonical hybrid ones, so a canonical artifact and an ablation artifact
    can never be confused for one another. Only called when ``fit_config``
    claims to be an ablation artifact (i.e.
    ``calm_source4_reachable_ablation_policy_semantics`` is present) --
    non-ablation ``candidate_first_crossing`` artifacts (fixed-layer,
    single-mode multi-source, the source-6 comparison output, or the
    canonical hybrid artifact) never set this key and are therefore
    unaffected. Fails closed (raises ``ValueError`` via the ``_require*``
    helpers, or returns ``False``) when the map is missing, incomplete,
    differs from the frozen ablation map, the ablation SHA is missing or
    mismatched, or the embedded CALM runtime policy identity does not match
    the artifact's own (already-validated)
    ``candidate_first_crossing_semantics``.
    """

    semantics = _require_mapping(
        fit_config.get("calm_source4_reachable_ablation_policy_semantics"),
        "fit_config.calm_source4_reachable_ablation_policy_semantics",
    )
    _metadata_json_safe(semantics, "fit_config.calm_source4_reachable_ablation_policy_semantics")
    required = {
        "hybrid_policy_schema_version",
        "hybrid_policy_name",
        "candidate_source_layers",
        "source_population_mode_map",
        "first_crossing_only_semantics",
        "counterfactual_reachable_semantics",
        "calm_runtime_policy_sha256",
    }
    _require_exact_keys(semantics, required, "fit_config.calm_source4_reachable_ablation_policy_semantics")
    candidate_layers = [int(item) for item in semantics.get("candidate_source_layers", [])]
    expected = calm_hybrid_calibration_policy_payload(
        candidate_layers=CALM_CANDIDATE_LAYERS,
        threshold=CALM_THRESHOLD,
        source_population_mode_map=CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP,
        policy_name=CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME,
    )
    if semantics != expected or candidate_layers != list(CALM_CANDIDATE_LAYERS):
        return False
    if semantics.get("hybrid_policy_name") != CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME:
        return False
    stored_sha = fit_config.get("calm_source4_reachable_ablation_policy_sha256")
    expected_sha = calm_hybrid_calibration_policy_sha256(
        candidate_layers=CALM_CANDIDATE_LAYERS,
        threshold=CALM_THRESHOLD,
        source_population_mode_map=CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP,
        policy_name=CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME,
    )
    if stored_sha != expected_sha:
        return False
    runtime_semantics = fit_config.get("candidate_first_crossing_semantics") or {}
    if semantics.get("calm_runtime_policy_sha256") != runtime_semantics.get("policy_sha256"):
        return False
    return True


def require_calm_source4_reachable_ablation_policy_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that ``artifact`` explicitly declares the source-4-reachable
    fitting ablation policy.

    Raises ``ValueError`` if
    ``fit_config.calm_source4_reachable_ablation_policy_semantics`` is
    entirely absent (i.e. this is not an ablation artifact at all -- this
    includes the canonical hybrid artifact, which only ever sets
    ``calm_hybrid_policy_semantics``) or fails
    ``validate_calm_source4_reachable_ablation_policy_semantics``. Intended
    for the dedicated ablation fitting/export/load/replay path, which must
    never silently accept the canonical hybrid artifact (or any other
    non-ablation artifact) as if it were the ablation one.
    """

    artifact = _require_mapping(artifact, "artifact")
    fit_config = _require_mapping(artifact.get("fit_config"), "artifact.fit_config")
    if fit_config.get("calm_source4_reachable_ablation_policy_semantics") is None:
        raise ValueError(
            "calm_source4_reachable_ablation_policy_semantics_missing: "
            "artifact does not declare the source-4-reachable ablation policy"
        )
    _require(
        validate_calm_source4_reachable_ablation_policy_semantics(fit_config),
        "calm_source4_reachable_ablation_policy_semantics mismatch",
    )


def _candidate_semantics_for_same_layer_validation(fit_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the fit_config's OWN declared-mode candidate semantics mapping
    -- never a fallback borrowed from the other mode.

    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING artifacts carry
    ``candidate_first_crossing_semantics`` (the frozen historical
    (4,6,8,10)/0.9 contract); SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM artifacts
    carry ``official_free_calm_semantics`` instead (a genuinely different,
    non-frozen contiguous range/threshold -- see
    ``validate_official_free_calm_semantics``). Both shapes carry the same
    ``candidate_exit_layers`` field, which is all the same-layer validator
    below actually needs, so this resolver is the ONLY mode-aware part of
    that validator; the validation logic itself is unchanged."""

    if fit_config.get("source_layer_mode") == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
        return _require_mapping(
            fit_config.get("official_free_calm_semantics"),
            "fit_config.official_free_calm_semantics",
        )
    return _require_mapping(
        fit_config.get("candidate_first_crossing_semantics"),
        "fit_config.candidate_first_crossing_semantics",
    )


def validate_same_layer_projection_validation_payload(
    payload: Mapping[str, Any],
    fit_config: Mapping[str, Any],
) -> bool:
    """Validate compact same-layer projection evidence for schema-v2 CALM
    artifacts -- both SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING (historical
    (4,6,8,10)/0.9) and SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM (official
    contiguous exit_min_layer..num_decoder_layers-1) artifacts, each reading
    its own mode's candidate semantics via
    ``_candidate_semantics_for_same_layer_validation`` -- never the other
    mode's."""

    payload = _require_mapping(payload, "provenance.same_layer_projection_validation")
    semantics = _candidate_semantics_for_same_layer_validation(fit_config)
    expected_sources = [int(item) for item in semantics.get("candidate_exit_layers", CALM_CANDIDATE_LAYERS)]
    _require_exact_value(
        payload,
        "validation_schema_version",
        SAME_LAYER_PROJECTION_VALIDATION_SCHEMA_VERSION,
        "provenance.same_layer_projection_validation",
    )
    _require_exact_value(
        payload,
        "projection_policy",
        SAME_LAYER_PROJECTION_POLICY,
        "provenance.same_layer_projection_validation",
    )
    _require_exact_value(payload, "status", "ok", "provenance.same_layer_projection_validation")
    _require_exact_value(
        payload,
        "candidate_source_layers",
        expected_sources,
        "provenance.same_layer_projection_validation",
    )
    _require_exact_value(
        payload,
        "same_layer_projection_atol",
        SAME_LAYER_PROJECTION_ATOL,
        "provenance.same_layer_projection_validation",
    )
    _require_exact_value(
        payload,
        "same_layer_projection_rtol",
        SAME_LAYER_PROJECTION_RTOL,
        "provenance.same_layer_projection_validation",
    )
    record_count = _require_nonnegative_int(payload, "record_count", "provenance.same_layer_projection_validation")
    _require(record_count > 0, "same_layer_projection_validation record_count must be positive")
    _require_exact_value(payload, "key_failed_record_count", 0, "provenance.same_layer_projection_validation")
    _require_exact_value(payload, "value_failed_record_count", 0, "provenance.same_layer_projection_validation")
    per_source = _require_mapping(
        payload.get("per_source_record_count"),
        "provenance.same_layer_projection_validation.per_source_record_count",
    )
    _require(
        set(per_source.keys()) == {str(item) for item in expected_sources},
        "same_layer_projection_validation per_source_record_count source mismatch",
    )
    _require(
        sum(int(value) for value in per_source.values()) == record_count,
        "same_layer_projection_validation per-source counts do not sum to record_count",
    )
    for field in ("logical_token_population_sha256", "same_layer_record_identity_sha256"):
        digest = _require_required_mapping_field(payload, field, "provenance.same_layer_projection_validation")
        _require(
            isinstance(digest, str) and _SHA256_PATTERN.match(digest),
            "same_layer_projection_validation.{} must be lowercase 64-character SHA-256 hex".format(field),
        )
    for field in ("key_max_abs", "value_max_abs"):
        value = _require_required_mapping_field(payload, field, "provenance.same_layer_projection_validation")
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            "same_layer_projection_validation.{} must be finite".format(field),
        )
    return True


def validate_phase3c_policy_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate a Phase 3c artifact and raise ``ValueError`` on failure."""

    _require(isinstance(artifact, MappingABC), "artifact must be a mapping")
    _require_exact_keys(artifact, _ARTIFACT_TOP_LEVEL_KEYS, "artifact")
    _require(artifact.get("artifact_type") == ARTIFACT_TYPE, "unsupported artifact_type: {!r}".format(artifact.get("artifact_type")))
    schema_version = artifact.get("schema_version")
    _require(
        schema_version in SUPPORTED_SCHEMA_VERSIONS,
        "unsupported schema_version: {!r}".format(schema_version),
    )
    _require(artifact.get("method_name") == METHOD_NAME, "unsupported method_name: {!r}".format(artifact.get("method_name")))
    forbidden = sorted(set(_walk_keys(artifact)) & _FORBIDDEN_KEYS)
    _require(not forbidden, "artifact contains forbidden raw/runtime fields: {}".format(forbidden))

    model_spec = _require_mapping(artifact["model_spec"], "model_spec")
    layer_index_semantics = _require_mapping(artifact["layer_index_semantics"], "layer_index_semantics")
    fit_config = _require_mapping(artifact["fit_config"], "fit_config")
    policy = _require_mapping(artifact["policy_config"], "policy_config")
    params_by_threshold = _require_mapping(artifact["parameters_by_threshold"], "parameters_by_threshold")
    provenance = _require_mapping(artifact["provenance"], "provenance")

    for field_name, field_value in (
        ("model_spec", model_spec),
        ("layer_index_semantics", layer_index_semantics),
        ("fit_config", fit_config),
        ("policy_config", policy),
        ("provenance", provenance),
    ):
        _metadata_json_safe(field_value, field_name)

    hidden_policy = _require_mapping(
        _require_required_mapping_field(policy, "final_hidden_policy", "policy_config"),
        "policy_config.final_hidden_policy",
    )
    k_policy = _require_mapping(
        _require_required_mapping_field(policy, "final_k_correction_policy", "policy_config"),
        "policy_config.final_k_correction_policy",
    )
    v_policy = _require_mapping(
        _require_required_mapping_field(policy, "final_v_correction_policy", "policy_config"),
        "policy_config.final_v_correction_policy",
    )
    _require_exact_value(policy, "exact_threshold_match_required", True, "policy_config")
    _require_exact_value(policy, "nearest_threshold_fallback", False, "policy_config")
    _require_exact_value(policy, "short_gap_v_identity_fallback", False, "policy_config")
    _require_exact_value(hidden_policy, "method", HIDDEN_METHOD, "policy_config.final_hidden_policy")
    _require_exact_value(hidden_policy, "fit_scope", HIDDEN_FIT_SCOPE, "policy_config.final_hidden_policy")
    _require_exact_value(hidden_policy, "condition", "would_exit_tokens", "policy_config.final_hidden_policy")
    _require_exact_value(k_policy, "method", K_CORRECTION, "policy_config.final_k_correction_policy")
    _require_exact_value(k_policy, "fit_scope", K_FIT_SCOPE, "policy_config.final_k_correction_policy")
    _require_exact_value(v_policy, "method", V_CORRECTION, "policy_config.final_v_correction_policy")
    _require_exact_value(v_policy, "fit_scope", V_FIT_SCOPE, "policy_config.final_v_correction_policy")

    _require_exact_value(fit_config, "hidden_method", HIDDEN_METHOD, "fit_config")
    _require_exact_value(fit_config, "hidden_fit_scope", HIDDEN_FIT_SCOPE, "fit_config")
    _require_exact_value(fit_config, "k_correction", K_CORRECTION, "fit_config")
    _require_exact_value(fit_config, "k_fit_scope", K_FIT_SCOPE, "fit_config")
    _require_exact_value(fit_config, "v_correction", V_CORRECTION, "fit_config")
    _require_exact_value(fit_config, "v_fit_scope", V_FIT_SCOPE, "fit_config")
    _require_exact_value(fit_config, "exact_threshold_match_required", True, "fit_config")
    _require_exact_value(fit_config, "no_nearest_threshold_fallback", True, "fit_config")
    _ = validate_exit_confidence_semantics(fit_config)
    source_layer_mode = _require_required_mapping_field(fit_config, "source_layer_mode", "fit_config")
    _require(
        source_layer_mode
        in {
            "first_threshold",
            "all_threshold",
            "all_layers",
            SOURCE_LAYER_MODE_FIXED,
            SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
        },
        "fit_config.source_layer_mode is unsupported: {!r}".format(source_layer_mode),
    )
    if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        _ = validate_fixed_layer_membership_semantics(fit_config)
        fixed_policy_has_stored_identity = (
            "policy_sha256" in fit_config
            or "fixed_layer_policy_semantics" in fit_config
        )
        if fixed_policy_has_stored_identity:
            fixed_policy_validation = validate_fixed_layer_policy_identity(
                artifact,
                require_authoritative=True,
            )
            _require(
                fixed_policy_validation.get("status") == "ok",
                "fixed_layer_policy_identity_validation_failed: {}".format(
                    ",".join(fixed_policy_validation.get("errors") or ["unknown"])
                ),
            )
    elif source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        _require(
            int(schema_version) == LATEST_SCHEMA_VERSION,
            "candidate_first_crossing artifacts require schema_version {}".format(LATEST_SCHEMA_VERSION),
        )
        _require(
            validate_candidate_first_crossing_semantics(fit_config),
            "candidate_first_crossing_semantics mismatch",
        )
        dump_binding = _require_mapping(
            provenance.get("dump_run_binding"),
            "provenance.dump_run_binding",
        )
        _require(
            dump_binding.get("binding_schema_version") == 4,
            "candidate_first_crossing artifacts require dump_run_binding schema v4",
        )
        same_layer_validation = _require_mapping(
            provenance.get("same_layer_projection_validation"),
            "provenance.same_layer_projection_validation",
        )
        _require(
            validate_same_layer_projection_validation_payload(same_layer_validation, fit_config),
            "same_layer_projection_validation invalid",
        )
        if fit_config.get("calm_hybrid_policy_semantics") is not None:
            _require(
                validate_calm_hybrid_policy_semantics(fit_config),
                "calm_hybrid_policy_semantics mismatch",
            )
        if fit_config.get("calm_source4_reachable_ablation_policy_semantics") is not None:
            _require(
                validate_calm_source4_reachable_ablation_policy_semantics(fit_config),
                "calm_source4_reachable_ablation_policy_semantics mismatch",
            )
        _require(
            fit_config.get("calm_hybrid_policy_semantics") is None
            or fit_config.get("calm_source4_reachable_ablation_policy_semantics") is None,
            "calm_hybrid_policy_semantics and calm_source4_reachable_ablation_policy_semantics "
            "are mutually exclusive",
        )
    elif source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
        _require(
            int(schema_version) == LATEST_SCHEMA_VERSION,
            "official_free_calm artifacts require schema_version {}".format(LATEST_SCHEMA_VERSION),
        )
        _require(
            validate_official_free_calm_semantics(fit_config),
            "official_free_calm_semantics mismatch",
        )
        dump_binding = _require_mapping(
            provenance.get("dump_run_binding"),
            "provenance.dump_run_binding",
        )
        _require(
            dump_binding.get("binding_schema_version") == 4,
            "official_free_calm artifacts require dump_run_binding schema v4",
        )
        same_layer_validation = _require_mapping(
            provenance.get("same_layer_projection_validation"),
            "provenance.same_layer_projection_validation",
        )
        _require(
            validate_same_layer_projection_validation_payload(same_layer_validation, fit_config),
            "same_layer_projection_validation invalid",
        )
    elif int(schema_version) == LATEST_SCHEMA_VERSION:
        raise ValueError(
            "schema_version 2 requires fixed_layer, candidate_first_crossing, or "
            "official_free_calm source_layer_mode"
        )

    d_model = _require_positive_int(model_spec, "d_model", "model_spec")
    num_heads = _require_positive_int(model_spec, "num_heads", "model_spec")
    d_kv = _require_positive_int(model_spec, "d_kv", "model_spec")
    decoder_layer_count = _require_positive_int(model_spec, "decoder_layer_count", "model_spec")
    _require_required_mapping_field(model_spec, "serialized_tensor_dtype", "model_spec")
    _require(model_spec.get("serialized_tensor_dtype") == "float32", "model_spec.serialized_tensor_dtype must be float32")
    _ = _require_positive_float(fit_config, "eps", "fit_config")
    exit_min_layer = _require_nonnegative_int(fit_config, "exit_min_layer", "fit_config")
    _require(exit_min_layer < decoder_layer_count, "fit_config.exit_min_layer must be less than model_spec.decoder_layer_count")
    _ = _require_nonnegative_int(fit_config, "max_eval_records_per_threshold", "fit_config")
    eval_stratify_by = _require_required_mapping_field(fit_config, "eval_stratify_by", "fit_config")
    _require(eval_stratify_by in {"none", "gap_bin"}, "fit_config.eval_stratify_by must be 'none' or 'gap_bin'")

    _require(bool(params_by_threshold), "parameters_by_threshold must be a non-empty mapping")
    declared_threshold_keys = _threshold_key_set_from_fit_config(fit_config)
    stored_threshold_keys = set()
    declared_v_gap_keys_by_threshold = _v_gap_key_sets_by_threshold_from_fit_config(fit_config, declared_threshold_keys)
    runtime_coverage = _runtime_coverage_validation_from_parts(model_spec, fit_config, params_by_threshold)
    if runtime_coverage.get("status") == "failed":
        raise ValueError(
            "runtime_coverage_validation failed: {}".format(
                json.dumps(runtime_coverage, sort_keys=True)
            )
        )
    for threshold_key, payload in params_by_threshold.items():
        canonical_threshold_key = threshold_to_key(threshold_key)
        _require(
            canonical_threshold_key == threshold_key,
            "noncanonical threshold key {!r}; expected {!r}".format(threshold_key, canonical_threshold_key),
        )
        _require(threshold_key not in stored_threshold_keys, "duplicate threshold key: {}".format(threshold_key))
        stored_threshold_keys.add(threshold_key)
        threshold = threshold_key_to_float(threshold_key)
        _require(math.isfinite(threshold), "threshold key must decode to a finite value: {}".format(threshold_key))
        payload = _require_mapping(payload, "{} payload".format(threshold_key))
        _require_exact_keys(payload, _THRESHOLD_PAYLOAD_KEYS, "parameters_by_threshold.{}".format(threshold_key))
        for group in _THRESHOLD_PAYLOAD_KEYS:
            _require_mapping(payload[group], "{}.{}".format(threshold_key, group))
            _require(bool(payload[group]), "{}.{} must be a non-empty mapping".format(threshold_key, group))

        hidden_keys = set(payload["hidden_by_layer_pair"].keys())
        k_keys = set(payload["k_by_layer_pair"].keys())
        if hidden_keys != k_keys:
            missing_k = sorted(hidden_keys - k_keys)
            extra_k = sorted(k_keys - hidden_keys)
            raise ValueError(
                "{} hidden/K layer-pair coverage mismatch: missing K pairs {}, extra K pairs {}".format(
                    threshold_key, missing_k, extra_k
                )
            )
        v_keys = set(payload["v_by_gap_bin"].keys())
        declared_v_gap_keys = declared_v_gap_keys_by_threshold.get(threshold_key, set())
        if v_keys != declared_v_gap_keys:
            missing_v = sorted(declared_v_gap_keys - v_keys)
            extra_v = sorted(v_keys - declared_v_gap_keys)
            raise ValueError(
                "{} V gap-bin coverage mismatch: missing bins {}, extra bins {}".format(
                    threshold_key, missing_v, extra_v
                )
            )

        for pair_key, params in payload["hidden_by_layer_pair"].items():
            source, target = layer_pair_key_to_tuple(pair_key)
            _require(pair_key == layer_pair_to_key(source, target), "noncanonical hidden layer-pair key: {}".format(pair_key))
            _require(0 <= source < target < decoder_layer_count, "{} hidden pair out of bounds".format(pair_key))
            params = _require_mapping(params, "{}.hidden.{}".format(threshold_key, pair_key))
            _require_exact_keys(
                params,
                _AFFINE_PARAM_KEYS,
                "parameters_by_threshold.{}.hidden_by_layer_pair.{}".format(threshold_key, pair_key),
            )
            _validate_tensor(params.get("scale"), "{}.hidden.{}.scale".format(threshold_key, pair_key), (d_model,))
            _validate_tensor(params.get("bias"), "{}.hidden.{}.bias".format(threshold_key, pair_key), (d_model,))

        for pair_key, params in payload["k_by_layer_pair"].items():
            source, target = layer_pair_key_to_tuple(pair_key)
            _require(pair_key == layer_pair_to_key(source, target), "noncanonical K layer-pair key: {}".format(pair_key))
            _require(0 <= source < target < decoder_layer_count, "{} K pair out of bounds".format(pair_key))
            params = _require_mapping(params, "{}.k.{}".format(threshold_key, pair_key))
            _require_exact_keys(
                params,
                _AFFINE_PARAM_KEYS,
                "parameters_by_threshold.{}.k_by_layer_pair.{}".format(threshold_key, pair_key),
            )
            _validate_tensor(params.get("scale"), "{}.k.{}.scale".format(threshold_key, pair_key), (num_heads, d_kv))
            _validate_tensor(params.get("bias"), "{}.k.{}.bias".format(threshold_key, pair_key), (num_heads, d_kv))

        for gap_key, params in payload["v_by_gap_bin"].items():
            label = gap_bin_key_to_label(gap_key)
            _require(gap_key == gap_bin_to_key(label), "noncanonical V gap-bin key: {}".format(gap_key))
            params = _require_mapping(params, "{}.v.{}".format(threshold_key, gap_key))
            _require_exact_keys(
                params,
                _V_PARAM_KEYS,
                "parameters_by_threshold.{}.v_by_gap_bin.{}".format(threshold_key, gap_key),
            )
            _validate_tensor(params.get("rotations"), "{}.v.{}.rotations".format(threshold_key, gap_key), (num_heads, d_kv, d_kv))
            _validate_tensor(params.get("biases"), "{}.v.{}.biases".format(threshold_key, gap_key), (num_heads, d_kv))

    if stored_threshold_keys != declared_threshold_keys:
        missing = sorted(declared_threshold_keys - stored_threshold_keys, key=threshold_key_to_float)
        extra = sorted(stored_threshold_keys - declared_threshold_keys, key=threshold_key_to_float)
        raise ValueError(
            "parameters_by_threshold keys do not match fit_config.thresholds: missing {}, extra {}".format(missing, extra)
        )


def get_threshold_policy_parameters(artifact: Mapping[str, Any], threshold: Any) -> Mapping[str, Any]:
    """Return exact threshold payload; no nearest-threshold fallback is allowed."""

    params_by_threshold = _require_mapping(artifact.get("parameters_by_threshold"), "parameters_by_threshold")
    threshold_key = threshold_to_key(threshold)
    if threshold_key not in params_by_threshold:
        raise ValueError("missing_artifact_threshold: {}".format(threshold_key))
    return _require_mapping(params_by_threshold[threshold_key], "parameters_by_threshold.{}".format(threshold_key))


def get_hidden_layer_pair_parameters(
    artifact: Mapping[str, Any],
    threshold: Any,
    source_layer: int,
    target_layer: int,
) -> Mapping[str, torch.Tensor]:
    """Return hidden diagonal-affine parameters for an exact source/target pair."""

    threshold_key = threshold_to_key(threshold)
    payload = get_threshold_policy_parameters(artifact, threshold_key)
    pair_key = layer_pair_to_key(source_layer, target_layer)
    hidden_maps = _require_mapping(payload.get("hidden_by_layer_pair"), "{}.hidden_by_layer_pair".format(threshold_key))
    if pair_key not in hidden_maps:
        raise ValueError("missing_hidden_layer_pair: {} {}".format(threshold_key, pair_key))
    return _require_mapping(hidden_maps[pair_key], "{}.hidden_by_layer_pair.{}".format(threshold_key, pair_key))


def get_k_layer_pair_parameters(
    artifact: Mapping[str, Any],
    threshold: Any,
    source_layer: int,
    target_layer: int,
) -> Mapping[str, torch.Tensor]:
    """Return K head/channel affine parameters for an exact source/target pair."""

    threshold_key = threshold_to_key(threshold)
    payload = get_threshold_policy_parameters(artifact, threshold_key)
    pair_key = layer_pair_to_key(source_layer, target_layer)
    k_maps = _require_mapping(payload.get("k_by_layer_pair"), "{}.k_by_layer_pair".format(threshold_key))
    if pair_key not in k_maps:
        raise ValueError("missing_k_layer_pair: {} {}".format(threshold_key, pair_key))
    return _require_mapping(k_maps[pair_key], "{}.k_by_layer_pair.{}".format(threshold_key, pair_key))


def get_v_gap_bin_parameters(
    artifact: Mapping[str, Any],
    threshold: Any,
    gap_bin: Any,
) -> Mapping[str, torch.Tensor]:
    """Return V head-wise Procrustes parameters for an exact threshold/gap bin."""

    threshold_key = threshold_to_key(threshold)
    payload = get_threshold_policy_parameters(artifact, threshold_key)
    gap_key = gap_bin_to_key(gap_bin)
    v_maps = _require_mapping(payload.get("v_by_gap_bin"), "{}.v_by_gap_bin".format(threshold_key))
    if gap_key not in v_maps:
        raise ValueError("missing_v_gap_bin: {} {}".format(threshold_key, gap_key))
    return _require_mapping(v_maps[gap_key], "{}.v_by_gap_bin.{}".format(threshold_key, gap_key))


def apply_hidden_diagonal_affine(
    source_hidden: torch.Tensor,
    params: Mapping[str, Any],
    *,
    validate_input_finite: bool = True,
    validate_parameter_finite: bool = True,
    validate_output_finite: bool = True,
) -> torch.Tensor:
    """Apply ``h_t_hat = h_s * scale + bias`` from loaded artifact parameters.

    All three validation flags default to True, so offline diagnostics,
    fitting validation and every existing caller are completely unchanged. The
    runtime Task C2 fast path sets them False only for data it has ALREADY
    finite-validated and that cannot have changed since: the source hidden
    checked once per transaction, and the immutable fitted parameters
    validated once when they entered the device cache. Shape, dtype and device
    validation are never skipped.

    ``validate_output_finite=False`` is opt-in ONLY for the batched Task C2
    transaction, which defers this decision to one aggregate finite check over
    the final candidate K/V it is about to commit. That is sound because this
    output is consumed exclusively by LayerNorm -> Linear -> the K/V
    corrections, and a nonfinite element cannot be turned back into a finite
    one by that chain: NaN propagates through every op, and Inf can only
    become NaN (``Inf*0``, ``Inf-Inf``) or stay Inf, never a finite value.
    Skipping it here therefore moves WHEN the failure is detected, not
    WHETHER. See models/deploying_t5.py
    ``_try_task_c2_fixed_source6_batched_insertion``.
    """

    params = _require_mapping(params, "hidden_params")
    source = _as_float_tensor(source_hidden, "source_hidden", validate_finite=validate_input_finite)
    scale = _as_float_tensor(
        params.get("scale"), "hidden_params.scale", validate_finite=validate_parameter_finite
    ).to(device=source.device)
    bias = _as_float_tensor(
        params.get("bias"), "hidden_params.bias", validate_finite=validate_parameter_finite
    ).to(device=source.device)
    _require_shape(scale, (source.shape[-1],), "hidden_params.scale")
    _require_shape(bias, (source.shape[-1],), "hidden_params.bias")
    result = source * scale + bias
    if validate_output_finite and not torch.isfinite(result).all().item():
        raise ValueError("hidden diagonal affine output contains NaN or Inf")
    return result


def apply_k_head_channel_affine(
    k_regenerated: torch.Tensor,
    params: Mapping[str, Any],
    *,
    validate_input_finite: bool = True,
    validate_parameter_finite: bool = True,
    validate_output_finite: bool = True,
) -> torch.Tensor:
    """Apply Phase 3c K head/channel affine correction to ``[B,H,T,D]`` tensors.

    ``k_regenerated`` is newly computed per target layer, so every caller
    except the batched Task C2 transaction keeps ``validate_input_finite``
    True; only the immutable cached fitted parameters skip revalidation.

    ``validate_input_finite``/``validate_output_finite`` are set False
    together ONLY by that batched transaction, which validates the final
    candidate K it commits instead. ``result = tensor * scale + bias`` is
    elementwise with finite scale/bias, so a nonfinite input element yields a
    nonfinite output element at the same index (``Inf*0`` is NaN, never 0) --
    deferring detection cannot lose it.
    """

    params = _require_mapping(params, "k_params")
    tensor = _as_float_tensor(k_regenerated, "k_regenerated", validate_finite=validate_input_finite)
    if tensor.ndim != 4:
        raise ValueError("k_regenerated must be a 4D [B,H,T,D] tensor, got {}".format(tuple(tensor.shape)))
    _batch, heads, _seq_len, dim = tensor.shape
    scale = _as_float_tensor(
        params.get("scale"), "k_params.scale", validate_finite=validate_parameter_finite
    ).to(device=tensor.device)
    bias = _as_float_tensor(
        params.get("bias"), "k_params.bias", validate_finite=validate_parameter_finite
    ).to(device=tensor.device)
    _require_shape(scale, (heads, dim), "k_params.scale")
    _require_shape(bias, (heads, dim), "k_params.bias")
    result = tensor * scale.view(1, heads, 1, dim) + bias.view(1, heads, 1, dim)
    if validate_output_finite and not torch.isfinite(result).all().item():
        raise ValueError("K head/channel affine output contains NaN or Inf")
    return result


def apply_v_headwise_procrustes(
    v_regenerated: torch.Tensor,
    params: Mapping[str, Any],
    *,
    validate_input_finite: bool = True,
    validate_parameter_finite: bool = True,
    validate_output_finite: bool = True,
) -> torch.Tensor:
    """Apply Phase 3b/3c head-wise Procrustes orientation: ``x @ R + b`` per head.

    ``v_regenerated`` is newly computed per target layer, so every caller
    except the batched Task C2 transaction keeps ``validate_input_finite``
    True; only the immutable cached fitted parameters skip revalidation.

    ``validate_input_finite``/``validate_output_finite`` are set False
    together ONLY by that batched transaction, which validates the final
    candidate V it commits instead. The einsum contracts only the last input
    dim, so a nonfinite ``tensor[b,h,t,d]`` contributes ``value * R[h,d,e]``
    to every output ``[b,h,t,e]``: finite ``R`` gives nonfinite, and a zero
    ``R`` entry gives NaN (IEEE ``Inf*0``/``NaN*0``). The nonfinite value
    therefore always survives into the output row, so deferring detection to
    the final candidate cannot lose it.
    """

    params = _require_mapping(params, "v_params")
    tensor = _as_float_tensor(v_regenerated, "v_regenerated", validate_finite=validate_input_finite)
    if tensor.ndim != 4:
        raise ValueError("v_regenerated must be a 4D [B,H,T,D] tensor, got {}".format(tuple(tensor.shape)))
    _batch, heads, _seq_len, dim = tensor.shape
    rotations = _as_float_tensor(
        params.get("rotations"), "v_params.rotations", validate_finite=validate_parameter_finite
    ).to(device=tensor.device)
    biases = _as_float_tensor(
        params.get("biases"), "v_params.biases", validate_finite=validate_parameter_finite
    ).to(device=tensor.device)
    _require_shape(rotations, (heads, dim, dim), "v_params.rotations")
    _require_shape(biases, (heads, dim), "v_params.biases")
    result = torch.einsum("bhtd,hde->bhte", tensor, rotations) + biases.view(1, heads, 1, dim)
    if validate_output_finite and not torch.isfinite(result).all().item():
        raise ValueError("V headwise Procrustes output contains NaN or Inf")
    return result


def validate_selected_eval_identity(identity: Mapping[str, Any], expected_threshold_keys: Optional[Iterable[str]] = None) -> None:
    """Validate replay-selection identity without storing full example ids."""

    identity = _require_mapping(identity, "selected_eval_identity")
    keys = set()
    for raw_key, payload in identity.items():
        threshold_key = threshold_to_key(raw_key)
        if threshold_key != raw_key:
            raise ValueError(
                "selected_eval_identity has noncanonical threshold key {!r}; expected {!r}".format(raw_key, threshold_key)
            )
        if threshold_key in keys:
            raise ValueError("selected_eval_identity contains duplicate threshold key: {}".format(threshold_key))
        keys.add(threshold_key)
        payload = _require_mapping(payload, "selected_eval_identity.{}".format(threshold_key))
        count = _require_required_mapping_field(payload, "count", "selected_eval_identity.{}".format(threshold_key))
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("selected_eval_identity.{}.count must be a non-negative integer".format(threshold_key))
        digest = _require_required_mapping_field(payload, "sha256", "selected_eval_identity.{}".format(threshold_key))
        if not isinstance(digest, str) or not _SHA256_PATTERN.match(digest):
            raise ValueError("selected_eval_identity.{}.sha256 must be lowercase 64-character SHA-256 hex".format(threshold_key))
    if expected_threshold_keys is not None:
        expected = {threshold_to_key(key) for key in expected_threshold_keys}
        if keys != expected:
            missing = sorted(expected - keys, key=threshold_key_to_float)
            extra = sorted(keys - expected, key=threshold_key_to_float)
            raise ValueError("selected_eval_identity threshold mismatch: missing {}, extra {}".format(missing, extra))


def assert_selected_eval_identity_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    expected_threshold_keys: Optional[Iterable[str]] = None,
) -> None:
    """Raise when reconstructed replay eval ids differ from artifact provenance."""

    validate_selected_eval_identity(expected, expected_threshold_keys)
    validate_selected_eval_identity(actual, expected_threshold_keys)
    if dict(expected) != dict(actual):
        raise ValueError("selected_eval_identity_mismatch: expected {}, actual {}".format(expected, actual))


def summarize_phase3c_policy_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe metadata summary without tensor values."""

    validate_phase3c_policy_artifact(artifact)
    threshold_summaries: Dict[str, Any] = {}
    parameter_count = 0
    tensor_bytes = 0
    tensor_shapes = {}
    tensor_dtypes = {}
    for field_path, tensor in _iter_tensors(artifact.get("parameters_by_threshold", {}), "parameters_by_threshold"):
        parameter_count += int(tensor.numel())
        tensor_bytes += int(tensor.numel() * tensor.element_size())
        tensor_shapes[field_path] = list(tensor.shape)
        tensor_dtypes[field_path] = str(tensor.dtype).replace("torch.", "")
    for threshold_key, payload in artifact["parameters_by_threshold"].items():
        threshold_summaries[threshold_key] = {
            "threshold": threshold_key_to_float(threshold_key),
            "hidden_layer_pair_map_count": len(payload["hidden_by_layer_pair"]),
            "k_layer_pair_map_count": len(payload["k_by_layer_pair"]),
            "v_gap_bin_map_count": len(payload["v_by_gap_bin"]),
        }
    runtime_coverage = phase3c_runtime_coverage_validation(artifact)
    policy_identity_validation = artifact_policy_identity_validation(
        artifact,
        require_authoritative=False,
    )
    return {
        "validation_status": "ok",
        "artifact_type": artifact["artifact_type"],
        "schema_version": artifact["schema_version"],
        "method_name": artifact["method_name"],
        "model_spec": artifact["model_spec"],
        "policy_config": artifact["policy_config"],
        "fit_config": artifact["fit_config"],
        "layer_index_semantics": artifact["layer_index_semantics"],
        "thresholds": list(artifact["parameters_by_threshold"].keys()),
        "threshold_summaries": threshold_summaries,
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "finite_check_status": "ok",
        "approx_parameter_count": parameter_count,
        "approx_serialized_tensor_bytes": tensor_bytes,
        "runtime_coverage_validation": runtime_coverage,
        "artifact_policy_identity_validation": policy_identity_validation,
        "policy_sha256": policy_identity_validation.get("policy_sha256"),
        "provenance": artifact["provenance"],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        raise TypeError("JSON summary must not contain tensor values")
    return value


def save_phase3c_policy_artifact(
    artifact: Mapping[str, Any],
    path: Any,
    *,
    summary_path: Optional[Any] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate and save a Phase 3c artifact, optionally with JSON summary."""

    validate_phase3c_policy_artifact(artifact)
    path = Path(path)
    if summary_path is not None and _normalized_path(path) == _normalized_path(summary_path):
        raise ValueError("artifact path and summary path must be different")
    if path.exists() and not overwrite:
        raise FileExistsError("artifact already exists: {}".format(path))
    if summary_path is not None:
        summary_path = Path(summary_path)
        if summary_path.exists() and not overwrite:
            raise FileExistsError("artifact summary already exists: {}".format(summary_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), path)
    summary = summarize_phase3c_policy_artifact(artifact)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_phase3c_policy_artifact(path: Any) -> Dict[str, Any]:
    """Load a Phase 3c artifact from disk and validate it."""

    path = Path(path)
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    validate_phase3c_policy_artifact(artifact)
    return artifact


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "LATEST_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "METHOD_NAME",
    "HIDDEN_METHOD",
    "HIDDEN_FIT_SCOPE",
    "K_CORRECTION",
    "K_FIT_SCOPE",
    "V_CORRECTION",
    "V_FIT_SCOPE",
    "EXIT_CONFIDENCE_LEGACY_SEMANTICS_VERSION",
    "EXIT_CONFIDENCE_SEMANTICS_VERSION",
    "EXIT_CONFIDENCE_TYPE",
    "EXIT_CONFIDENCE_THRESHOLD_COMPARATOR",
    "EXIT_CONFIDENCE_COMPUTE_DTYPE",
    "SOURCE_LAYER_MODE_FIXED",
    "SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING",
    "SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM",
    "FIXED_LAYER_POLICY_TYPE",
    "FIXED_LAYER_POLICY_SEMANTICS_VERSION",
    "SAME_LAYER_PROJECTION_POLICY",
    "SAME_LAYER_PROJECTION_VALIDATION_SCHEMA_VERSION",
    "SAME_LAYER_PROJECTION_ATOL",
    "SAME_LAYER_PROJECTION_RTOL",
    "build_phase3c_policy_artifact",
    "artifact_policy_identity_validation",
    "artifact_policy_sha256",
    "apply_hidden_diagonal_affine",
    "apply_k_head_channel_affine",
    "apply_v_headwise_procrustes",
    "assert_selected_eval_identity_matches",
    "default_layer_index_semantics",
    "default_exit_confidence_semantics",
    "default_candidate_first_crossing_semantics",
    "default_official_free_calm_semantics",
    "default_fixed_layer_membership_semantics",
    "default_policy_config",
    "fixed_layer_policy_semantics_payload",
    "fixed_layer_policy_sha256",
    "gap_bin_key_to_label",
    "gap_bin_to_key",
    "get_hidden_layer_pair_parameters",
    "get_k_layer_pair_parameters",
    "get_threshold_policy_parameters",
    "get_v_gap_bin_parameters",
    "layer_pair_key_to_tuple",
    "layer_pair_to_key",
    "load_phase3c_policy_artifact",
    "phase3c_runtime_coverage_validation",
    "required_phase3c_candidate_first_crossing_coverage",
    "required_phase3c_runtime_coverage",
    "save_phase3c_policy_artifact",
    "summarize_phase3c_policy_artifact",
    "threshold_key_to_float",
    "threshold_to_key",
    "validate_exit_confidence_semantics",
    "validate_fixed_layer_policy_identity",
    "validate_candidate_first_crossing_semantics",
    "validate_official_free_calm_semantics",
    "validate_fixed_layer_membership_semantics",
    "validate_selected_eval_identity",
    "validate_same_layer_projection_validation_payload",
    "validate_phase3c_policy_artifact",
]
