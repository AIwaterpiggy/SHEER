"""Counterfactual CALM candidate-confidence trace provenance helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from our_kv_restoration.missing_kv_dump_provenance import (
    PACKED_GENERATION_STORAGE_FORMAT,
    ProvenanceValidationError,
    canonical_json_sha256,
    canonical_json_text,
    logical_population_keys_for_packed_record,
)

try:  # Optional for import-only tests.
    import torch
except Exception:  # pragma: no cover - torch is present in normal runs.
    torch = None

try:
    from util.skip_conf import compute_exit_lm_logits, softmax_top1_top2_margin
except Exception:  # pragma: no cover - model import path covers this in normal runs.
    compute_exit_lm_logits = None
    softmax_top1_top2_margin = None


CALM_TRACE_SCHEMA_VERSION = 1
CALM_TRACE_RECORD_TYPE = "calm_candidate_confidence_trace_token"
CALM_POLICY_NAME = "candidate_restricted_first_crossing_v1"
CALM_CANDIDATE_LAYERS = (4, 6, 8, 10)
CALM_CANDIDATE_LAYERS_PAPER_ONE_BASED = tuple(layer + 1 for layer in CALM_CANDIDATE_LAYERS)
CALM_CONFIDENCE_TYPE = "softmax_top1_top2_probability_margin"
CALM_CONFIDENCE_COMPUTE_DTYPE = "float32"
CALM_THRESHOLD = 0.9
CALM_THRESHOLD_COMPARATOR = "strict_gt"
CALM_USE_ADAPT_THRESHOLD = False
CALM_FALLBACK_RULE = "full_depth"
CALM_TRACE_SEMANTICS = "counterfactual_full_depth_observation_only"
CALM_FULL_DEPTH_REFERENCE_GENERATION = True
CALM_USE_EARLY_EXIT = False
CALM_USE_SHALLOW_DEEP = False
CALM_STATIC_EXIT_LAYER = None
CALM_RUNTIME_RESTORATION_ENABLED = False
CALM_RUNTIME_RESTORATION_CALM_ENABLED = False

# Consumer-side (Phase 3c fitting) calibration-population modes. These are not
# a change to the frozen CALM runtime policy above: they only control which
# authenticated trace tokens contribute learned-map *fitting* examples for a
# given source layer. Only these two values are supported.
CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY = "first_crossing_only"
CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE = "counterfactual_reachable"
CALM_CALIBRATION_POPULATION_MODES = (
    CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
    CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
)
CALM_CALIBRATION_POPULATION_SEMANTICS_VERSION = 1

# Frozen source-specific hybrid fitting policy: an engineering rule for
# constructing one complete multi-source [4, 6, 8, 10] artifact, not a change
# to the frozen CALM runtime policy above and not a claim that reachable
# fitting is more accurate. Sources 4/6 have enough authoritative actual
# first-crossing support; sources 8/10 do not, so they fall back to
# counterfactual_reachable fitting. This map is frozen -- no arbitrary
# user-defined source map is exposed in the paper path.
CALM_HYBRID_POLICY_NAME = "first_crossing_4_6_reachable_8_10_v1"
CALM_HYBRID_POLICY_SEMANTICS_VERSION = 1
CALM_HYBRID_FIRST_CROSSING_SOURCE_LAYERS = (4, 6)
CALM_HYBRID_REACHABLE_SOURCE_LAYERS = (8, 10)
CALM_HYBRID_SOURCE_POPULATION_MODE_MAP = {
    4: CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
    6: CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
    8: CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
    10: CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
}

# Source-4-reachable fitting-population ablation: a distinct, separately
# fitted/exported/loaded/replayed policy used only to test whether source-4
# underperformance under the frozen hybrid policy above is caused by sparse
# actual-first-crossing training support. Source 4 is the earliest candidate
# layer, so under the frozen ordered CALM policy every aligned token is
# reachable there. This is a fitting-population-only ablation -- it does not
# change the frozen CALM runtime policy above, does not change or replace the
# canonical hybrid policy/map/name above, and is not itself a new frozen
# production policy.
CALM_SOURCE4_REACHABLE_ABLATION_POLICY_NAME = "reachable_4_first_crossing_6_reachable_8_10_ablation_v1"
CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP = {
    4: CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
    6: CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
    8: CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
    10: CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
}

# Official FREE CALM-style production early-exit path (models/deploying_t5.py
# ``use_early_exit`` decoder loop): the authoritative candidate set is every
# decoder layer from ``exit_min_layer`` through ``num_decoder_layers - 1``,
# evaluated in order, using the model's own runtime ``exit_conf_threshold`` --
# never the frozen historical (4, 6, 8, 10)/0.9 research policy above. This is
# a distinct, separately named policy: it never redefines
# ``CALM_CANDIDATE_LAYERS``/``CALM_THRESHOLD``/``CALM_POLICY_NAME`` and the
# historical policy stays fully in force wherever it is already used.
OFFICIAL_FREE_CALM_POLICY_NAME = "official_free_calm_contiguous_first_crossing_v1"
OFFICIAL_FREE_CALM_CONFIDENCE_TYPE = CALM_CONFIDENCE_TYPE
OFFICIAL_FREE_CALM_CONFIDENCE_COMPUTE_DTYPE = CALM_CONFIDENCE_COMPUTE_DTYPE
OFFICIAL_FREE_CALM_THRESHOLD_COMPARATOR = CALM_THRESHOLD_COMPARATOR
OFFICIAL_FREE_CALM_USE_ADAPT_THRESHOLD = False


def official_free_calm_candidate_layers(*, exit_min_layer: int, num_decoder_layers: int) -> Tuple[int, ...]:
    """Derive the official FREE CALM candidate range from the model's own
    ``exit_min_layer``/decoder depth -- never a separately hard-coded layer
    list. For T5-large (exit_min_layer=4, num_decoder_layers=24) this is
    ``(4, 5, ..., 23)``."""

    exit_min_layer = int(exit_min_layer)
    num_decoder_layers = int(num_decoder_layers)
    if exit_min_layer < 0:
        raise ProvenanceValidationError("official_free_calm_exit_min_layer_negative")
    if num_decoder_layers <= exit_min_layer:
        raise ProvenanceValidationError("official_free_calm_num_decoder_layers_not_greater_than_exit_min_layer")
    return tuple(range(exit_min_layer, num_decoder_layers))


CALM_TRACE_EXECUTION_INVARIANTS = {
    "trace_semantics": CALM_TRACE_SEMANTICS,
    "full_depth_reference_generation": CALM_FULL_DEPTH_REFERENCE_GENERATION,
    "use_early_exit": CALM_USE_EARLY_EXIT,
    "use_shallow_deep": CALM_USE_SHALLOW_DEEP,
    "static_exit_layer": CALM_STATIC_EXIT_LAYER,
    "runtime_restoration_enabled": CALM_RUNTIME_RESTORATION_ENABLED,
    "runtime_restoration_calm_enabled": CALM_RUNTIME_RESTORATION_CALM_ENABLED,
    "adaptive_threshold": CALM_USE_ADAPT_THRESHOLD,
}


def parse_candidate_layers(value: Any) -> Tuple[int, ...]:
    if value is None or value == "":
        return CALM_CANDIDATE_LAYERS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    try:
        layers = tuple(int(part) for part in parts)
    except Exception as exc:
        raise ProvenanceValidationError("calm_candidate_layers_invalid") from exc
    if not layers:
        raise ProvenanceValidationError("calm_candidate_layers_empty")
    if list(layers) != sorted(layers):
        raise ProvenanceValidationError("calm_candidate_layers_not_sorted")
    if len(set(layers)) != len(layers):
        raise ProvenanceValidationError("calm_candidate_layers_duplicate")
    if any(layer < 0 for layer in layers):
        raise ProvenanceValidationError("calm_candidate_layers_negative")
    return layers


def calm_policy_payload(
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
    policy_name: str = CALM_POLICY_NAME,
) -> Dict[str, Any]:
    layers = [int(layer) for layer in candidate_layers]
    return {
        "policy_schema_version": 1,
        "policy_name": str(policy_name),
        "candidate_exit_layers": layers,
        "candidate_exit_layers_paper_one_based": [layer + 1 for layer in layers],
        "candidate_evaluation_order": layers,
        "confidence_type": CALM_CONFIDENCE_TYPE,
        "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
        "threshold": float(threshold),
        "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
        "exit_rule": "first_candidate_strictly_greater_than_threshold",
        "fallback_rule": CALM_FALLBACK_RULE,
        "adaptive_threshold": CALM_USE_ADAPT_THRESHOLD,
    }


def calm_policy_sha256(
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
    policy_name: str = CALM_POLICY_NAME,
) -> str:
    return canonical_json_sha256(
        calm_policy_payload(candidate_layers=candidate_layers, threshold=threshold, policy_name=policy_name)
    )


def compute_calm_candidate_logits_and_confidence(
    hidden_states: Any,
    *,
    final_layer_norm: Any,
    dropout: Any,
    lm_head: Any,
    config: Any,
) -> Tuple[Any, float]:
    """Compute FREE exit logits and float32 margin for a candidate source hidden state."""

    if torch is None or compute_exit_lm_logits is None or softmax_top1_top2_margin is None:
        raise RuntimeError("torch_or_confidence_helpers_unavailable")
    with torch.no_grad():
        confidence_hidden = dropout(final_layer_norm(hidden_states))
        lm_logits = compute_exit_lm_logits(confidence_hidden, lm_head, config)
        confidence = softmax_top1_top2_margin(lm_logits)
    return lm_logits, float(confidence.detach().float().view(-1)[-1].item())


def compute_calm_candidate_confidence(
    hidden_states: Any,
    *,
    final_layer_norm: Any,
    dropout: Any,
    lm_head: Any,
    config: Any,
) -> float:
    """Compute the existing FREE softmax margin on a candidate source hidden state."""

    _lm_logits, confidence = compute_calm_candidate_logits_and_confidence(
        hidden_states,
        final_layer_norm=final_layer_norm,
        dropout=dropout,
        lm_head=lm_head,
        config=config,
    )
    return confidence


def _finite_float(value: Any) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ProvenanceValidationError("calm_confidence_nonfinite")
    return number


def _candidate_pass(confidence: float, threshold: float) -> bool:
    return float(confidence) > float(threshold)


def calm_reachable_source_layers(
    first_crossing_candidate_layer: Optional[int],
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
) -> Tuple[int, ...]:
    """Sources reachable under the frozen ordered CALM candidate policy.

    A token is eligible for source ``s`` when it would reach ``s`` under the
    frozen ordered policy, i.e. every candidate strictly before ``s`` failed.
    Compact equivalent (candidate order ``[4, 6, 8, 10]``): the reachable set
    is every candidate up to and including the actual first crossing, or every
    candidate when the token fell back to full depth (no first crossing).
    """

    layers = parse_candidate_layers(candidate_layers)
    if first_crossing_candidate_layer is None:
        return tuple(layers)
    layer = int(first_crossing_candidate_layer)
    if layer not in layers:
        raise ProvenanceValidationError("calm_first_crossing_candidate_layer_unsupported")
    return tuple(layers[: layers.index(layer) + 1])


def validate_calm_candidate_evaluations_reachability(
    row: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
) -> Dict[str, Any]:
    """Fail-closed structural validation of one CALM trace row's authoritative
    ``candidate_evaluations`` array.

    Reachability is derived purely from the authoritative ``candidate_evaluations``
    array already recorded by the CALM trace producer; hidden tensors are never
    consulted. Raises ``ProvenanceValidationError`` (fail closed, no silent
    repair) for any malformed or internally inconsistent row.
    """

    layers = parse_candidate_layers(candidate_layers)
    if tuple(layers) != tuple(CALM_CANDIDATE_LAYERS):
        raise ProvenanceValidationError("calm_candidate_layers_not_frozen")
    evaluations = row.get("candidate_evaluations")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise ProvenanceValidationError("calm_candidate_evaluations_missing")
    if len(evaluations) != len(layers):
        raise ProvenanceValidationError("calm_candidate_evaluations_count_mismatch")
    seen_layers: List[int] = []
    candidate_pass_by_layer: Dict[int, bool] = {}
    confidence_by_layer: Dict[int, float] = {}
    recomputed_first: Optional[int] = None
    for index, item in enumerate(evaluations):
        if not isinstance(item, Mapping):
            raise ProvenanceValidationError("calm_candidate_evaluation_not_mapping")
        try:
            layer = int(item.get("candidate_layer"))
        except Exception as exc:
            raise ProvenanceValidationError("calm_candidate_evaluation_layer_invalid") from exc
        if layer != layers[index]:
            raise ProvenanceValidationError("calm_candidate_evaluation_order_mismatch")
        if layer in seen_layers:
            raise ProvenanceValidationError("calm_candidate_evaluation_duplicate")
        seen_layers.append(layer)
        confidence = _finite_float(item.get("confidence"))
        expected_pass = _candidate_pass(confidence, threshold)
        if "candidate_pass" not in item:
            raise ProvenanceValidationError("calm_candidate_evaluation_pass_missing")
        candidate_pass_value = item.get("candidate_pass")
        if type(candidate_pass_value) is not bool:
            raise ProvenanceValidationError("calm_candidate_evaluation_pass_not_bool")
        if candidate_pass_value != expected_pass:
            raise ProvenanceValidationError("calm_candidate_evaluation_pass_mismatch")
        if "threshold" in item:
            try:
                item_threshold_matches = abs(float(item.get("threshold")) - float(threshold)) <= 1e-12
            except Exception as exc:
                raise ProvenanceValidationError("calm_candidate_evaluation_threshold_invalid") from exc
            if not item_threshold_matches:
                raise ProvenanceValidationError("calm_candidate_evaluation_threshold_mismatch")
        if "threshold_comparator" in item and item.get("threshold_comparator") != CALM_THRESHOLD_COMPARATOR:
            raise ProvenanceValidationError("calm_candidate_evaluation_threshold_comparator_mismatch")
        if "candidate_order_index" in item:
            try:
                order_index_matches = int(item.get("candidate_order_index")) == index
            except Exception as exc:
                raise ProvenanceValidationError("calm_candidate_evaluation_order_index_invalid") from exc
            if not order_index_matches:
                raise ProvenanceValidationError("calm_candidate_evaluation_order_index_mismatch")
        if expected_pass and recomputed_first is None:
            recomputed_first = layer
        candidate_pass_by_layer[layer] = expected_pass
        confidence_by_layer[layer] = confidence
    if seen_layers != list(layers):
        raise ProvenanceValidationError("calm_candidate_evaluation_layers_mismatch")
    declared_first = row.get("first_crossing_candidate_layer")
    declared_first = None if declared_first is None else int(declared_first)
    if declared_first != recomputed_first:
        raise ProvenanceValidationError("calm_first_crossing_mismatch")
    if "full_depth_fallback" not in row:
        raise ProvenanceValidationError("calm_full_depth_fallback_missing")
    full_depth_fallback_value = row.get("full_depth_fallback")
    if type(full_depth_fallback_value) is not bool:
        raise ProvenanceValidationError("calm_full_depth_fallback_not_bool")
    if full_depth_fallback_value != (recomputed_first is None):
        raise ProvenanceValidationError("calm_full_depth_fallback_mismatch")
    reachable = calm_reachable_source_layers(recomputed_first, candidate_layers=layers)
    return {
        "first_crossing_candidate_layer": recomputed_first,
        "full_depth_fallback": recomputed_first is None,
        "reachable_source_layers": reachable,
        "candidate_pass_by_layer": candidate_pass_by_layer,
        "confidence_by_layer": confidence_by_layer,
    }


def calm_calibration_population_semantics_payload(
    calibration_population_mode: str,
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    comparison_source_layer: Optional[int] = None,
    evaluation_population_mode: str = CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
) -> Dict[str, Any]:
    """Explicit, JSON-safe semantics payload for a calibration-population mode.

    This documents a *consumer-side fitting* decision. It never changes the
    frozen CALM runtime policy or its ``policy_sha256``.
    """

    if calibration_population_mode not in CALM_CALIBRATION_POPULATION_MODES:
        raise ProvenanceValidationError("calm_calibration_population_mode_unsupported")
    layers = parse_candidate_layers(candidate_layers)
    if tuple(layers) != tuple(CALM_CANDIDATE_LAYERS):
        raise ProvenanceValidationError("calm_candidate_layers_not_frozen")
    return {
        "calibration_population_semantics_version": CALM_CALIBRATION_POPULATION_SEMANTICS_VERSION,
        "calibration_population_mode": str(calibration_population_mode),
        "candidate_order": [int(item) for item in layers],
        "reachability_rule": (
            "token is eligible for source s iff every candidate strictly before s failed "
            "(reachable set = candidates up to and including the actual first crossing, "
            "or every candidate on full-depth fallback)"
        ),
        "source_candidate_pass_required": calibration_population_mode == CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
        "evaluation_population_mode": str(evaluation_population_mode),
        "comparison_source_layer": None if comparison_source_layer is None else int(comparison_source_layer),
    }


def calm_hybrid_source_population_mode_map_json(
    source_population_mode_map: Mapping[int, str] = CALM_HYBRID_SOURCE_POPULATION_MODE_MAP,
) -> Dict[str, str]:
    """Canonical JSON-safe (string-keyed) form of the frozen hybrid source map."""

    return {str(int(layer)): str(mode) for layer, mode in sorted(source_population_mode_map.items())}


def calm_hybrid_calibration_policy_payload(
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
    source_population_mode_map: Mapping[int, str] = CALM_HYBRID_SOURCE_POPULATION_MODE_MAP,
    policy_name: str = CALM_HYBRID_POLICY_NAME,
) -> Dict[str, Any]:
    """Canonical payload for a source-specific hybrid fitting policy.

    ``policy_name``/``source_population_mode_map`` default to the frozen
    canonical hybrid policy (``CALM_HYBRID_POLICY_NAME`` /
    ``CALM_HYBRID_SOURCE_POPULATION_MODE_MAP``) for backward compatibility --
    every existing call site keeps producing the exact same payload/SHA. A
    distinct source map (e.g. the source-4-reachable ablation,
    ``CALM_SOURCE4_REACHABLE_ABLATION_SOURCE_POPULATION_MODE_MAP``) must
    always be paired with its own distinct ``policy_name`` -- never generate
    an ablation hash while silently retaining the canonical policy name.

    Binds: schema version, policy name, candidate source layers, the
    source-to-population-mode map, the two reused primitive-mode semantics
    payloads (first-crossing and reachability), and the frozen CALM runtime
    policy identity. This is a consumer-side fitting-population decision only
    -- it never changes the frozen CALM runtime policy or its ``policy_sha256``.
    """

    layers = parse_candidate_layers(candidate_layers)
    if tuple(layers) != tuple(CALM_CANDIDATE_LAYERS):
        raise ProvenanceValidationError("calm_candidate_layers_not_frozen")
    mode_map = {int(layer): str(mode) for layer, mode in source_population_mode_map.items()}
    if set(mode_map.keys()) != set(int(item) for item in layers):
        raise ProvenanceValidationError("calm_hybrid_source_population_mode_map_layers_mismatch")
    for mode in mode_map.values():
        if mode not in CALM_CALIBRATION_POPULATION_MODES:
            raise ProvenanceValidationError("calm_calibration_population_mode_unsupported")
    return {
        "hybrid_policy_schema_version": CALM_HYBRID_POLICY_SEMANTICS_VERSION,
        "hybrid_policy_name": str(policy_name),
        "candidate_source_layers": [int(item) for item in layers],
        "source_population_mode_map": calm_hybrid_source_population_mode_map_json(mode_map),
        "first_crossing_only_semantics": calm_calibration_population_semantics_payload(
            CALM_CALIBRATION_POPULATION_FIRST_CROSSING_ONLY,
            candidate_layers=layers,
        ),
        "counterfactual_reachable_semantics": calm_calibration_population_semantics_payload(
            CALM_CALIBRATION_POPULATION_COUNTERFACTUAL_REACHABLE,
            candidate_layers=layers,
        ),
        "calm_runtime_policy_sha256": calm_policy_sha256(candidate_layers=layers, threshold=threshold),
    }


def calm_hybrid_calibration_policy_sha256(
    *,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
    source_population_mode_map: Mapping[int, str] = CALM_HYBRID_SOURCE_POPULATION_MODE_MAP,
    policy_name: str = CALM_HYBRID_POLICY_NAME,
) -> str:
    return canonical_json_sha256(
        calm_hybrid_calibration_policy_payload(
            candidate_layers=candidate_layers,
            threshold=threshold,
            source_population_mode_map=source_population_mode_map,
            policy_name=policy_name,
        )
    )


def make_calm_trace_row_uid(row: Mapping[str, Any]) -> str:
    payload = {
        "trace_schema_version": row.get("trace_schema_version"),
        "record_type": row.get("record_type"),
        "trace_semantics": row.get("trace_semantics"),
        "full_depth_reference_generation": row.get("full_depth_reference_generation"),
        "use_early_exit": row.get("use_early_exit"),
        "use_shallow_deep": row.get("use_shallow_deep"),
        "static_exit_layer": row.get("static_exit_layer"),
        "runtime_restoration_enabled": row.get("runtime_restoration_enabled"),
        "runtime_restoration_calm_enabled": row.get("runtime_restoration_calm_enabled"),
        "adaptive_threshold": row.get("adaptive_threshold"),
        "policy_name": row.get("policy_name"),
        "policy_sha256": row.get("policy_sha256"),
        "stable_sample_id": row.get("stable_sample_id"),
        "generation_index": row.get("generation_index"),
        "decoder_position": row.get("decoder_position"),
        "candidate_evaluations": row.get("candidate_evaluations"),
        "first_crossing_candidate_layer": row.get("first_crossing_candidate_layer"),
        "full_depth_fallback": row.get("full_depth_fallback"),
    }
    return canonical_json_sha256(payload)


def build_calm_trace_row(
    *,
    sample_context: Mapping[str, Any],
    generation_index: int,
    decoder_position: int,
    candidate_confidences: Mapping[int, Any],
    model_num_decoder_layers: int,
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
) -> Dict[str, Any]:
    layers = parse_candidate_layers(candidate_layers)
    policy = calm_policy_payload(candidate_layers=layers, threshold=threshold)
    evaluations = []
    first_crossing = None
    for layer in layers:
        confidence = _finite_float(candidate_confidences[layer])
        passed = _candidate_pass(confidence, threshold)
        if passed and first_crossing is None:
            first_crossing = int(layer)
        evaluations.append(
            {
                "candidate_layer": int(layer),
                "candidate_layer_paper_one_based": int(layer) + 1,
                "candidate_order_index": len(evaluations),
                "confidence": confidence,
                "threshold": float(threshold),
                "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
                "candidate_pass": bool(passed),
                "confidence_type": CALM_CONFIDENCE_TYPE,
                "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
                "phase3c_source_hidden_layer": int(layer),
                "last_exact_kv_layer": int(layer) - 1,
                "first_missing_target_layer": int(layer),
                "missing_target_layer_end_exclusive": int(model_num_decoder_layers),
                "same_layer_target_map_required": False,
            }
        )
    row = {
        "trace_schema_version": CALM_TRACE_SCHEMA_VERSION,
        "record_type": CALM_TRACE_RECORD_TYPE,
        **CALM_TRACE_EXECUTION_INVARIANTS,
        "policy_name": CALM_POLICY_NAME,
        "policy_sha256": canonical_json_sha256(policy),
        "policy": policy,
        "stable_sample_id": sample_context.get("stable_sample_id"),
        "selected_order": sample_context.get("selected_order"),
        "raw_dataset_index": sample_context.get("raw_dataset_index"),
        "dataset_provided_id": sample_context.get("dataset_provided_id"),
        "generation_index": int(generation_index),
        "decoder_position": int(decoder_position),
        "token_index": int(decoder_position),
        "model_num_decoder_layers": int(model_num_decoder_layers),
        "candidate_exit_layers": list(layers),
        "candidate_exit_layers_paper_one_based": [layer + 1 for layer in layers],
        "confidence_type": CALM_CONFIDENCE_TYPE,
        "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
        "threshold": float(threshold),
        "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
        "candidate_evaluations": evaluations,
        "first_crossing_candidate_layer": first_crossing,
        "first_crossing_candidate_layer_paper_one_based": None if first_crossing is None else first_crossing + 1,
        "full_depth_fallback": first_crossing is None,
        "phase3c_source_hidden_layer": first_crossing,
        "last_exact_kv_layer": None if first_crossing is None else first_crossing - 1,
        "first_missing_target_layer": first_crossing,
    }
    row["calm_trace_row_uid"] = make_calm_trace_row_uid(row)
    return row


def _successful_packed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("dump_succeeded") is True
        and row.get("storage_format") == PACKED_GENERATION_STORAGE_FORMAT
        and row.get("file_path")
    ]


def token_population_keys_from_packed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, int, int]]:
    keys = {
        (str(stable), int(generation), int(position))
        for row in _successful_packed_rows(rows)
        for stable, generation, position, _layer in logical_population_keys_for_packed_record(row)
    }
    return sorted(keys, key=lambda item: canonical_json_text(item))


def token_population_sha256(keys: Iterable[Tuple[Any, int, int]]) -> str:
    normalized = sorted(
        [[str(stable), int(generation), int(position)] for stable, generation, position in keys],
        key=lambda item: canonical_json_text(item),
    )
    return canonical_json_sha256(normalized)


def trace_rows_semantic_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = sorted(
        [dict(row) for row in rows],
        key=lambda row: canonical_json_text(
            [
                str(row.get("stable_sample_id")),
                int(row.get("generation_index", -1)),
                int(row.get("decoder_position", -1)),
            ]
        ),
    )
    return canonical_json_sha256(normalized)


def _binding_by_generation(binding_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, Mapping[str, Any]], List[str]]:
    errors: List[str] = []
    rows_by_generation: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in binding_rows:
        try:
            rows_by_generation[int(row["generation_index"])].append(row)
        except Exception:
            errors.append("calm_trace_binding_missing_generation_index")
    out: Dict[int, Mapping[str, Any]] = {}
    for generation_index, rows in rows_by_generation.items():
        if len(rows) != 1:
            errors.append("calm_trace_binding_duplicate_generation:{}".format(generation_index))
            continue
        out[generation_index] = rows[0]
    return out, errors


def _identity_matches_trace(row: Mapping[str, Any], binding: Mapping[str, Any]) -> List[str]:
    errors = []
    if str(row.get("stable_sample_id")) != str(binding.get("stable_sample_id")):
        errors.append("calm_trace_stable_sample_id_mismatch")
    for field in ("selected_order", "raw_dataset_index"):
        try:
            if int(row.get(field)) != int(binding.get(field)):
                errors.append("calm_trace_{}_mismatch".format(field))
        except Exception:
            errors.append("calm_trace_{}_mismatch".format(field))
    row_dataset_id = row.get("dataset_provided_id")
    binding_dataset_id = binding.get("dataset_provided_id")
    if (None if row_dataset_id is None else str(row_dataset_id)) != (
        None if binding_dataset_id is None else str(binding_dataset_id)
    ):
        errors.append("calm_trace_dataset_provided_id_mismatch")
    return errors


def _validate_execution_invariants(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if row.get("trace_semantics") != CALM_TRACE_SEMANTICS:
        errors.append("calm_trace_semantics_mismatch")
    if row.get("full_depth_reference_generation") is not True:
        errors.append("calm_trace_full_depth_reference_mismatch")
    if row.get("use_early_exit") is not False:
        errors.append("calm_trace_use_early_exit_mismatch")
    if row.get("use_shallow_deep") is not False:
        errors.append("calm_trace_use_shallow_deep_mismatch")
    if row.get("static_exit_layer") is not None:
        errors.append("calm_trace_static_exit_layer_mismatch")
    if row.get("runtime_restoration_enabled") is not False:
        errors.append("calm_trace_runtime_restoration_enabled")
    if row.get("runtime_restoration_calm_enabled") is not False:
        errors.append("calm_trace_runtime_restoration_calm_enabled")
    if row.get("adaptive_threshold") is not False:
        errors.append("calm_trace_adaptive_threshold_mismatch")
    return errors


def validate_calm_trace_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    population_rows: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
    hidden_rows: Sequence[Mapping[str, Any]],
    kv_rows: Sequence[Mapping[str, Any]],
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
) -> Dict[str, Any]:
    errors: List[str] = []
    layers = parse_candidate_layers(candidate_layers)
    expected_policy = calm_policy_payload(candidate_layers=layers, threshold=threshold)
    expected_policy_sha = canonical_json_sha256(expected_policy)
    hidden_tokens = token_population_keys_from_packed_rows(hidden_rows)
    kv_tokens = token_population_keys_from_packed_rows(kv_rows)
    hidden_token_set = set(hidden_tokens)
    kv_token_set = set(kv_tokens)
    if hidden_token_set != kv_token_set:
        errors.append("calm_trace_hidden_kv_token_population_mismatch")
    expected_token_set = hidden_token_set & kv_token_set
    hidden_layer_keys = {
        (str(stable), int(generation), int(position), int(layer))
        for row in _successful_packed_rows(hidden_rows)
        for stable, generation, position, layer in logical_population_keys_for_packed_record(row)
    }
    bindings_by_generation, binding_errors = _binding_by_generation(binding_rows)
    errors.extend(binding_errors)
    population_stable_ids = {str(row.get("stable_sample_id")) for row in population_rows}

    token_counts = Counter()
    per_candidate_evaluation_counts = Counter()
    per_candidate_pass_counts = Counter()
    per_candidate_first_crossing_counts = Counter()
    per_candidate_confidence_values: Dict[str, List[float]] = defaultdict(list)
    first_crossing_count = 0
    fallback_count = 0
    nonfinite_count = 0
    out_of_range_count = 0
    missing_candidate_hidden_identity_count = 0
    row_uid_mismatch_count = 0
    trace_token_set = set()

    for row in trace_rows:
        try:
            generation_index = int(row.get("generation_index"))
            decoder_position = int(row.get("decoder_position"))
        except Exception:
            errors.append("calm_trace_row_missing_token_identity")
            continue
        stable_id = str(row.get("stable_sample_id"))
        token_identity = (stable_id, generation_index, decoder_position)
        trace_token_set.add(token_identity)
        token_counts[token_identity] += 1
        if stable_id not in population_stable_ids:
            errors.append("calm_trace_unknown_stable_sample_id")
        binding = bindings_by_generation.get(generation_index)
        if binding is None:
            errors.append("calm_trace_unknown_generation_index:{}".format(generation_index))
        else:
            errors.extend(_identity_matches_trace(row, binding))
        if row.get("trace_schema_version") != CALM_TRACE_SCHEMA_VERSION:
            errors.append("calm_trace_schema_mismatch")
        if row.get("record_type") != CALM_TRACE_RECORD_TYPE:
            errors.append("calm_trace_record_type_mismatch")
        errors.extend(_validate_execution_invariants(row))
        if row.get("policy_name") != CALM_POLICY_NAME:
            errors.append("calm_trace_policy_name_mismatch")
        if row.get("policy_sha256") != expected_policy_sha:
            errors.append("calm_trace_policy_sha256_mismatch")
        if row.get("policy") != expected_policy:
            errors.append("calm_trace_policy_payload_mismatch")
        if list(row.get("candidate_exit_layers") or []) != list(layers):
            errors.append("calm_trace_candidate_layers_mismatch")
        if str(row.get("confidence_type")) != CALM_CONFIDENCE_TYPE:
            errors.append("calm_trace_confidence_type_mismatch")
        if str(row.get("confidence_compute_dtype")) != CALM_CONFIDENCE_COMPUTE_DTYPE:
            errors.append("calm_trace_confidence_dtype_mismatch")
        if str(row.get("threshold_comparator")) != CALM_THRESHOLD_COMPARATOR:
            errors.append("calm_trace_threshold_comparator_mismatch")
        if bool(row.get("adaptive_threshold")) != CALM_USE_ADAPT_THRESHOLD:
            errors.append("calm_trace_adaptive_threshold_mismatch")
        try:
            if abs(float(row.get("threshold")) - float(threshold)) > 1e-12:
                errors.append("calm_trace_threshold_mismatch")
        except Exception:
            errors.append("calm_trace_threshold_mismatch")
        evaluations = row.get("candidate_evaluations")
        if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
            errors.append("calm_trace_candidate_evaluations_missing")
            evaluations = []
        if [item.get("candidate_layer") for item in evaluations if isinstance(item, Mapping)] != list(layers):
            errors.append("calm_trace_candidate_evaluation_layers_mismatch")
        recomputed_first = None
        for item in evaluations:
            if not isinstance(item, Mapping):
                errors.append("calm_trace_candidate_evaluation_not_mapping")
                continue
            layer = int(item.get("candidate_layer", -1))
            per_candidate_evaluation_counts[str(layer)] += 1
            if (stable_id, generation_index, decoder_position, layer) not in hidden_layer_keys:
                missing_candidate_hidden_identity_count += 1
                errors.append("calm_trace_missing_candidate_hidden_identity:{}:{}".format(generation_index, layer))
            try:
                confidence = _finite_float(item.get("confidence"))
            except ProvenanceValidationError:
                confidence = 0.0
                nonfinite_count += 1
                errors.append("calm_trace_nonfinite_confidence")
            if confidence < 0.0 or confidence > 1.0:
                out_of_range_count += 1
                errors.append("calm_trace_confidence_out_of_range")
            else:
                per_candidate_confidence_values[str(layer)].append(confidence)
            passed = _candidate_pass(confidence, threshold)
            if bool(item.get("candidate_pass")) != passed:
                errors.append("calm_trace_candidate_pass_mismatch")
            if passed:
                per_candidate_pass_counts[str(layer)] += 1
                if recomputed_first is None:
                    recomputed_first = layer
        first_value = row.get("first_crossing_candidate_layer")
        normalized_first = None if first_value is None else int(first_value)
        if normalized_first != recomputed_first:
            errors.append("calm_trace_first_crossing_mismatch")
        if bool(row.get("full_depth_fallback")) != (recomputed_first is None):
            errors.append("calm_trace_fallback_mismatch")
        if recomputed_first is None:
            fallback_count += 1
        else:
            first_crossing_count += 1
            per_candidate_first_crossing_counts[str(recomputed_first)] += 1
        try:
            row_uid_matches = make_calm_trace_row_uid(row) == row.get("calm_trace_row_uid")
        except Exception:
            row_uid_matches = False
        if not row_uid_matches:
            row_uid_mismatch_count += 1
            errors.append("calm_trace_row_uid_mismatch")

    duplicate_tokens = sorted([key for key, count in token_counts.items() if count > 1], key=lambda item: canonical_json_text(item))
    if duplicate_tokens:
        errors.append("calm_trace_duplicate_token_identity")
    missing_tokens = sorted(expected_token_set - trace_token_set, key=lambda item: canonical_json_text(item))
    unexpected_tokens = sorted(trace_token_set - expected_token_set, key=lambda item: canonical_json_text(item))
    if missing_tokens:
        errors.append("calm_trace_missing_token_rows")
    if unexpected_tokens:
        errors.append("calm_trace_unexpected_token_rows")

    try:
        rows_sha = trace_rows_semantic_sha256(trace_rows)
    except Exception:
        rows_sha = None
        errors.append("calm_trace_rows_semantic_sha256_failed")

    per_candidate_confidence_min = {}
    per_candidate_confidence_max = {}
    per_candidate_confidence_sum = {}
    per_candidate_confidence_mean = {}
    for layer in layers:
        key = str(layer)
        values = per_candidate_confidence_values.get(key, [])
        if values:
            total = float(sum(values))
            per_candidate_confidence_min[key] = float(min(values))
            per_candidate_confidence_max[key] = float(max(values))
            per_candidate_confidence_sum[key] = total
            per_candidate_confidence_mean[key] = float(total / len(values))
        else:
            per_candidate_confidence_min[key] = None
            per_candidate_confidence_max[key] = None
            per_candidate_confidence_sum[key] = 0.0
            per_candidate_confidence_mean[key] = None

    return {
        "trace_schema_version": CALM_TRACE_SCHEMA_VERSION,
        "status": "ok" if not errors else "failed",
        "errors": sorted(set(errors)),
        "policy": expected_policy,
        "policy_sha256": expected_policy_sha,
        "trace_row_count": len(trace_rows),
        "unique_token_count": len(trace_token_set),
        "expected_token_count": len(expected_token_set),
        "unique_generation_count": len({item[1] for item in trace_token_set}),
        "unique_sample_count": len({item[0] for item in trace_token_set}),
        "first_crossing_count": first_crossing_count,
        "full_depth_fallback_count": fallback_count,
        "per_candidate_evaluation_counts": dict(sorted(per_candidate_evaluation_counts.items())),
        "per_candidate_pass_counts": dict(sorted(per_candidate_pass_counts.items())),
        "per_candidate_first_crossing_counts": dict(sorted(per_candidate_first_crossing_counts.items())),
        "per_candidate_confidence_min": dict(sorted(per_candidate_confidence_min.items())),
        "per_candidate_confidence_max": dict(sorted(per_candidate_confidence_max.items())),
        "per_candidate_confidence_sum": dict(sorted(per_candidate_confidence_sum.items())),
        "per_candidate_confidence_mean": dict(sorted(per_candidate_confidence_mean.items())),
        "duplicate_token_identity_count": len(duplicate_tokens),
        "missing_token_count": len(missing_tokens),
        "unexpected_token_count": len(unexpected_tokens),
        "missing_candidate_hidden_identity_count": missing_candidate_hidden_identity_count,
        "nonfinite_confidence_count": nonfinite_count,
        "out_of_range_confidence_count": out_of_range_count,
        "trace_row_uid_mismatch_count": row_uid_mismatch_count,
        "token_population_sha256": token_population_sha256(trace_token_set),
        "expected_token_population_sha256": token_population_sha256(expected_token_set),
        "trace_rows_semantic_sha256": rows_sha,
        "missing_token_preview": [list(item) for item in missing_tokens[:32]],
        "unexpected_token_preview": [list(item) for item in unexpected_tokens[:32]],
    }


def validate_calm_trace_sidecar(
    trace_rows: Sequence[Mapping[str, Any]],
    supplied_summary: Optional[Mapping[str, Any]],
    *,
    population_rows: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
    hidden_rows: Sequence[Mapping[str, Any]],
    kv_rows: Sequence[Mapping[str, Any]],
    candidate_layers: Sequence[int] = CALM_CANDIDATE_LAYERS,
    threshold: float = CALM_THRESHOLD,
) -> Dict[str, Any]:
    recomputed = validate_calm_trace_rows(
        trace_rows,
        population_rows=population_rows,
        binding_rows=binding_rows,
        hidden_rows=hidden_rows,
        kv_rows=kv_rows,
        candidate_layers=candidate_layers,
        threshold=threshold,
    )
    errors: List[str] = []
    if not isinstance(supplied_summary, Mapping):
        errors.append("calm_trace_summary_missing")
    else:
        for key in (
            "status",
            "policy_sha256",
            "trace_row_count",
            "unique_token_count",
            "expected_token_count",
            "per_candidate_confidence_min",
            "per_candidate_confidence_max",
            "token_population_sha256",
            "expected_token_population_sha256",
            "trace_rows_semantic_sha256",
        ):
            if supplied_summary.get(key) != recomputed.get(key):
                errors.append("calm_trace_summary_{}_mismatch".format(key))
    return {
        **recomputed,
        "status": "ok" if not errors and recomputed.get("status") == "ok" else "failed",
        "errors": sorted(set(list(recomputed.get("errors", [])) + errors)),
        "recomputed": recomputed,
        "supplied_status": supplied_summary.get("status") if isinstance(supplied_summary, Mapping) else None,
    }
