"""F2a frozen-prefix / frozen-exit-schedule component replay helpers.

The helpers in this module are deliberately pure or narrowly tensor-local.
They do not run generation.  Runtime callers provide the exact reference
trajectory state, and these helpers validate the frozen event identity, build
candidate K/V tensors for the requested component, and compute compact
downstream logit metrics.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from our_kv_restoration.missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_CONFIDENCE_COMPUTE_DTYPE,
    CALM_POLICY_NAME,
    CALM_THRESHOLD,
    CALM_THRESHOLD_COMPARATOR,
    calm_policy_sha256,
)
from our_kv_restoration.missing_kv_dump_provenance import canonical_json_sha256, sha256_file
from our_kv_restoration.phase3c_policy_artifact import (
    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
    SOURCE_LAYER_MODE_FIXED,
    apply_hidden_diagonal_affine,
    apply_k_head_channel_affine,
    apply_v_headwise_procrustes,
    artifact_policy_identity_validation,
    artifact_policy_sha256,
    get_hidden_layer_pair_parameters,
    get_k_layer_pair_parameters,
    get_v_gap_bin_parameters,
    load_phase3c_policy_artifact,
    phase3c_runtime_coverage_validation,
    threshold_to_key,
    validate_candidate_first_crossing_semantics,
    validate_phase3c_policy_artifact,
)


F2A_LEGACY_SCHEDULE_SCHEMA_VERSION = 1
F2A_LEGACY_RECORD_SCHEMA_VERSION = 1
F2A_LEGACY_SCHEDULE_SCHEMA_V2 = 2
F2A_LEGACY_RECORD_SCHEMA_V2 = 2
F2A_SCHEDULE_SCHEMA_VERSION = 3
F2A_RECORD_SCHEMA_VERSION = 3
F2A_PRODUCER_PROTOCOL_VERSION = 3
F2A_EVALUATION_PROTOCOL_NAME = "f2a_frozen_prefix_frozen_exit_schedule_v3"
F2A_LEGACY_EVALUATION_PROTOCOL_NAME = "f2a_frozen_prefix_frozen_exit_schedule_v1"
F2A_EVENT_RECORD_TYPE = "missing_kv_f2a_frozen_event"
F2A_COMPONENT_RECORD_TYPE = "missing_kv_f2a_component_replay"
F2A_REFERENCE_TRAJECTORY_SCHEMA_VERSION = 1
F2A_REFERENCE_TRAJECTORY_RECORD_TYPE = "missing_kv_f2a_reference_generation_trajectory"
F2A_REFERENCE_TRAJECTORY_SUMMARY_SCHEMA_VERSION = 1
F2A_REFERENCE_TRAJECTORY_SUMMARY_RECORD_TYPE = "missing_kv_f2a_reference_generation_trajectory_summary"
F2A_EXACT_SHADOW_METHOD = "exact_reference_cache_shadow"
F2A_SUMMARY_SCHEMA_VERSION = 1
F2A_STATISTICS_EVALUATION_MODE = "frozen_schedule_replay"
F2A_STATISTICS_EVALUATION_SEMANTICS = "frozen_schedule"

F2A_METHOD_EXIT_HIDDEN_TARGET_PROJECTION = "exit_hidden_target_projection"
F2A_METHOD_EXIT_CONDITIONED_HIDDEN_RESTORATION = "exit_conditioned_hidden_restoration"
F2A_METHOD_FINAL_KV_RESTORATION = "final_kv_restoration"
F2A_REQUIRED_METHODS = (
    F2A_METHOD_EXIT_HIDDEN_TARGET_PROJECTION,
    F2A_METHOD_EXIT_CONDITIONED_HIDDEN_RESTORATION,
    F2A_METHOD_FINAL_KV_RESTORATION,
)
F2A_SUPPORTED_SOURCE_LAYER_MODES = (
    SOURCE_LAYER_MODE_FIXED,
    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
)
F2A_REFERENCE_TRAJECTORY_TERMINAL_REASONS = (
    "eos",
    "max_length",
    "stopping_criteria",
    "terminal_no_followup",
    "generation_failure",
    "unknown_generation_end",
)
F2A_FIXED_REFERENCE_DECISION_TYPES = (
    "full_depth_reference_forward",
    "fixed_shallow_skip",
    "fixed_exact_catchup_flush",
)
F2A_CALM_REFERENCE_DECISION_TYPES = (
    "full_depth_reference_forward",
    "first_crossing",
    "full_depth_fallback",
)

F2A_LOGIT_METRICS = (
    "logit_mse",
    "logit_relative_l2",
    "logit_cosine",
    "reference_to_candidate_kl",
    "candidate_to_reference_kl",
    "jensen_shannon_divergence",
    "top1_agreement",
    "reference_top1_token_id",
    "candidate_top1_token_id",
    "reference_top1_probability",
    "candidate_reference_token_probability",
    "top1_probability_delta",
    "top1_margin_delta",
)

F2A_PARITY_TOLERANCES = {
    "float32": {
        "logit_mse": 1.0e-10,
        "logit_relative_l2": 1.0e-5,
        "one_minus_logit_cosine": 1.0e-6,
        "reference_to_candidate_kl": 1.0e-7,
        "candidate_to_reference_kl": 1.0e-7,
        "jensen_shannon_divergence": 1.0e-7,
    },
    "float16": {
        "logit_mse": 1.0e-5,
        "logit_relative_l2": 2.0e-3,
        "one_minus_logit_cosine": 1.0e-4,
        "reference_to_candidate_kl": 1.0e-4,
        "candidate_to_reference_kl": 1.0e-4,
        "jensen_shannon_divergence": 1.0e-4,
    },
    "bfloat16": {
        "logit_mse": 1.0e-5,
        "logit_relative_l2": 2.0e-3,
        "one_minus_logit_cosine": 1.0e-4,
        "reference_to_candidate_kl": 1.0e-4,
        "candidate_to_reference_kl": 1.0e-4,
        "jensen_shannon_divergence": 1.0e-4,
    },
}

# Separate from F2A_PARITY_TOLERANCES above: this is not a parity-acceptance
# tolerance. KL/Jensen-Shannon divergence are mathematically nonnegative, but
# a float64 sum-reduction over (near-)identical distributions can still land
# a hair below zero from ordinary roundoff. This bounds how negative a raw
# divergence may be before it is treated as roundoff (and clamped to 0.0)
# versus treated as an implementation/numerical-invariant failure (fail
# closed). It must stay many orders of magnitude below the frozen 1e-7
# forward/reverse-KL and JS parity tolerances above.
F2A_NEGATIVE_ROUNDOFF_TOLERANCE = 1.0e-12

F2A_EVENT_REQUIRED_FIELDS = (
    "stable_sample_id",
    "generation_index",
    "decoder_position",
    "current_decoder_input_position",
    "predicted_token_position",
    "current_decoder_input_token_id",
    "prefix_token_sha256",
    "source_layer_mode",
    "source_layer",
    "last_exact_kv_layer",
    "first_missing_target_layer",
    "target_layers",
    "pending_token_positions",
    "restore_relative_indices",
    "cache_positions",
    "reference_schedule_identity",
    "reference_run_identity",
    "artifact_file_sha256",
    "policy_sha256",
    "decoder_layer_count",
    "threshold",
    "threshold_comparator",
)

_SEMANTIC_EVENT_FIELDS = (
    "schema_version",
    "record_type",
    "producer_protocol",
    "stable_sample_id",
    "generation_index",
    "decoder_position",
    "event_timing",
    "restored_token_position",
    "current_decoder_input_position",
    "predicted_token_position",
    "current_decoder_input_token_id",
    "prefix_token_sha256",
    "source_layer_mode",
    "source_layer",
    "last_exact_kv_layer",
    "first_missing_target_layer",
    "target_layers",
    "pending_token_positions",
    "restore_relative_indices",
    "cache_positions",
    "reference_schedule_identity",
    "reference_run_identity",
    "artifact_file_sha256",
    "policy_sha256",
    "candidate_policy_sha256",
    "policy_name",
    "candidate_layers",
    "threshold",
    "threshold_comparator",
    "confidence_compute_dtype",
    "adaptive_threshold",
    "source_confidence",
    "candidate_evaluations",
    "first_crossing_decision_sha256",
    "followup_decoder_input_token_id",
    "followup_decoder_input_position",
    "followup_predicted_token_position",
    "followup_prefix_token_sha256",
    "followup_reference_decision_layer",
    "followup_reference_decision_type",
    "decoder_layer_count",
)


class F2AFrozenScheduleError(ValueError):
    """Raised when F2a frozen-schedule inputs are structurally invalid."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _as_int(value: Any, field: str, errors: Optional[List[str]] = None) -> Optional[int]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value)
    except Exception:
        if errors is not None:
            errors.append("{}_invalid".format(field))
            return None
        raise F2AFrozenScheduleError("{}_invalid".format(field))


def _as_int_list(value: Any, field: str, errors: Optional[List[str]] = None) -> Optional[List[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        if errors is not None:
            errors.append("{}_invalid".format(field))
            return None
        raise F2AFrozenScheduleError("{}_invalid".format(field))
    result: List[int] = []
    for item in value:
        parsed = _as_int(item, field, errors)
        if parsed is None:
            return None
        result.append(parsed)
    return result


def prefix_token_sha256(token_ids: Sequence[int]) -> str:
    """Return a canonical SHA-256 digest for generated prefix token IDs."""

    return canonical_json_sha256([int(item) for item in token_ids])


def generated_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Return a canonical SHA-256 digest for newly generated token IDs."""

    return canonical_json_sha256([int(item) for item in token_ids])


def reference_decision_trace_sha256(decision_trace: Sequence[Mapping[str, Any]]) -> str:
    """Return the semantic digest for the reference execution decision trace."""

    return canonical_json_sha256([_json_safe(dict(row)) for row in decision_trace])


_REFERENCE_TRAJECTORY_SEMANTIC_FIELDS = (
    "schema_version",
    "record_type",
    "producer_protocol",
    "stable_sample_id",
    "generation_index",
    "selected_order",
    "raw_dataset_index",
    "dataset_provided_id",
    "generated_token_ids",
    "generated_token_count",
    "generated_token_ids_sha256",
    "reference_decision_trace",
    "reference_decision_trace_sha256",
    "terminal_reason",
    "generation_completed",
    "decoder_layer_count",
)


def _reference_trajectory_row_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        field: _json_safe(row[field])
        for field in _REFERENCE_TRAJECTORY_SEMANTIC_FIELDS
        if field in row
    }
    return dict(sorted(payload.items()))


def make_reference_generation_trajectory_row_uid(row: Mapping[str, Any]) -> str:
    """Build a path-independent semantic UID for one completed reference generation."""

    return canonical_json_sha256(_reference_trajectory_row_payload(row))


def build_reference_generation_trajectory_row(
    *,
    stable_sample_id: Any,
    generation_index: Any,
    selected_order: Any = None,
    raw_dataset_index: Any = None,
    dataset_provided_id: Any = None,
    generated_token_ids: Sequence[int],
    reference_decision_trace: Sequence[Mapping[str, Any]],
    terminal_reason: str,
    generation_completed: bool = True,
    decoder_layer_count: Any = None,
) -> Dict[str, Any]:
    """Build one authoritative F2a reference generation trajectory row.

    ``generated_token_ids`` are the ordered newly generated decoder output
    tokens, excluding the initial decoder-start token.  Candidate component
    replay outputs are never included here.
    """

    tokens = [int(item) for item in generated_token_ids]
    decisions = [_json_safe(dict(item)) for item in reference_decision_trace]
    row: Dict[str, Any] = {
        "schema_version": F2A_REFERENCE_TRAJECTORY_SCHEMA_VERSION,
        "record_type": F2A_REFERENCE_TRAJECTORY_RECORD_TYPE,
        "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
        "stable_sample_id": stable_sample_id,
        "generation_index": int(generation_index),
        "selected_order": selected_order,
        "raw_dataset_index": raw_dataset_index,
        "dataset_provided_id": dataset_provided_id,
        "generated_token_ids": tokens,
        "generated_token_count": len(tokens),
        "generated_token_ids_sha256": generated_token_ids_sha256(tokens),
        "reference_decision_trace": decisions,
        "reference_decision_trace_sha256": reference_decision_trace_sha256(decisions),
        "terminal_reason": str(terminal_reason),
        "generation_completed": bool(generation_completed),
    }
    if decoder_layer_count is not None:
        row["decoder_layer_count"] = int(decoder_layer_count)
    row["reference_generation_trajectory_row_uid"] = make_reference_generation_trajectory_row_uid(row)
    return row


def reference_generation_trajectory_population_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    population = []
    for row in rows:
        payload = _reference_trajectory_row_payload(row)
        payload["reference_generation_trajectory_row_uid"] = row.get(
            "reference_generation_trajectory_row_uid"
        ) or make_reference_generation_trajectory_row_uid(row)
        population.append(payload)
    population.sort(key=lambda item: str(item.get("reference_generation_trajectory_row_uid")))
    return canonical_json_sha256(population)


def ordered_reference_generation_trajectory_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return the order-sensitive digest for reference generation trajectory rows."""

    ordered = []
    for index, row in enumerate(rows):
        ordered.append(
            {
                "row_order": int(index),
                "selected_order": row.get("selected_order"),
                "generation_index": row.get("generation_index"),
                "stable_sample_id": row.get("stable_sample_id"),
                "reference_generation_trajectory_row_uid": row.get(
                    "reference_generation_trajectory_row_uid"
                )
                or make_reference_generation_trajectory_row_uid(row),
                "generated_token_ids_sha256": row.get("generated_token_ids_sha256"),
                "reference_decision_trace_sha256": row.get("reference_decision_trace_sha256"),
            }
        )
    return canonical_json_sha256(ordered)


def _decision_passes_threshold(evaluation: Mapping[str, Any], threshold: float) -> bool:
    try:
        return float(evaluation.get("confidence")) > float(threshold)
    except Exception:
        return False


def _float32_value(value: Any) -> float:
    return float(torch.tensor(float(value), dtype=torch.float32).item())


def validate_calm_candidate_evaluation_prefix(
    *,
    evaluations: Any,
    selected_source_layer: Any,
    fallback: bool,
    error_prefix: str,
    errors: List[str],
    context_suffix: str = "",
) -> Dict[str, Any]:
    """Validate the frozen CALM first-crossing evaluation prefix.

    The authoritative CALM policy evaluates candidates in order and stops at
    the first strict ``confidence > 0.9`` crossing.  A fallback evaluates every
    authoritative candidate and observes no crossing.
    """

    result: Dict[str, Any] = {
        "status": "failed",
        "observed_layers": [],
        "passing_layers": [],
        "selected_confidence": None,
    }
    if not isinstance(evaluations, list) or not evaluations:
        errors.append("{}_candidate_evaluations_missing{}".format(error_prefix, context_suffix))
        return result

    candidate_layers = [int(item) for item in CALM_CANDIDATE_LAYERS]
    try:
        selected = None if fallback else int(selected_source_layer)
    except Exception:
        errors.append("{}_selected_source_invalid{}".format(error_prefix, context_suffix))
        return result
    if not fallback and selected not in candidate_layers:
        errors.append("{}_selected_source_not_candidate{}".format(error_prefix, context_suffix))
        return result

    expected_layers = candidate_layers if fallback else candidate_layers[: candidate_layers.index(selected) + 1]
    observed_layers: List[int] = []
    passing_layers: List[int] = []
    selected_confidence: Optional[float] = None
    local_error_count = len(errors)
    for eval_index, evaluation in enumerate(evaluations):
        eval_suffix = "{}:eval{}".format(context_suffix, eval_index)
        if not isinstance(evaluation, Mapping):
            errors.append("{}_candidate_evaluation_invalid{}".format(error_prefix, eval_suffix))
            continue
        for field in ("candidate_layer", "confidence", "threshold", "threshold_comparator", "candidate_pass"):
            if field not in evaluation:
                errors.append("{}_candidate_evaluation_required_field_missing:{}{}".format(error_prefix, field, eval_suffix))
        try:
            layer = int(evaluation.get("candidate_layer"))
        except Exception:
            errors.append("{}_candidate_evaluation_layer_invalid{}".format(error_prefix, eval_suffix))
            continue
        observed_layers.append(layer)
        if layer not in candidate_layers:
            errors.append("{}_candidate_evaluation_layer_unknown{}".format(error_prefix, eval_suffix))
        try:
            confidence = _float32_value(evaluation.get("confidence"))
            if not math.isfinite(confidence):
                raise ValueError
        except Exception:
            errors.append("{}_candidate_evaluation_confidence_invalid{}".format(error_prefix, eval_suffix))
            continue
        try:
            threshold = float(evaluation.get("threshold"))
        except Exception:
            errors.append("{}_candidate_evaluation_threshold_invalid{}".format(error_prefix, eval_suffix))
            threshold = None
        if threshold != float(CALM_THRESHOLD):
            errors.append("{}_candidate_evaluation_threshold_mismatch{}".format(error_prefix, eval_suffix))
        comparator = evaluation.get("threshold_comparator")
        if comparator != CALM_THRESHOLD_COMPARATOR:
            errors.append("{}_candidate_evaluation_comparator_mismatch{}".format(error_prefix, eval_suffix))
        if "candidate_pass" in evaluation and not isinstance(evaluation.get("candidate_pass"), bool):
            errors.append("{}_candidate_evaluation_pass_type_invalid{}".format(error_prefix, eval_suffix))
        actual_pass = confidence > float(CALM_THRESHOLD)
        if bool(evaluation.get("candidate_pass")) != actual_pass:
            errors.append("{}_candidate_pass_mismatch{}".format(error_prefix, eval_suffix))
        if actual_pass:
            passing_layers.append(layer)
        if not fallback and layer == selected:
            selected_confidence = confidence

    if observed_layers != expected_layers:
        errors.append("{}_candidate_evaluation_prefix_mismatch{}".format(error_prefix, context_suffix))
    if len(set(observed_layers)) != len(observed_layers):
        errors.append("{}_candidate_evaluation_duplicate_layer{}".format(error_prefix, context_suffix))
    if fallback:
        if passing_layers:
            errors.append("{}_fallback_has_passing_candidate{}".format(error_prefix, context_suffix))
    else:
        if not passing_layers:
            errors.append("{}_first_crossing_without_pass{}".format(error_prefix, context_suffix))
        elif passing_layers[0] != selected:
            errors.append("{}_first_crossing_not_earliest{}".format(error_prefix, context_suffix))
        if observed_layers and observed_layers[-1] != selected:
            errors.append("{}_selected_source_not_final_evaluation{}".format(error_prefix, context_suffix))
        if selected_confidence is None:
            errors.append("{}_selected_source_not_evaluated{}".format(error_prefix, context_suffix))
        elif selected_confidence <= float(CALM_THRESHOLD):
            errors.append("{}_selected_source_not_crossing{}".format(error_prefix, context_suffix))

    result.update(
        {
            "status": "ok" if len(errors) == local_error_count else "failed",
            "observed_layers": observed_layers,
            "passing_layers": passing_layers,
            "selected_confidence": selected_confidence,
        }
    )
    return result


def _validate_calm_candidate_evaluations(
    *,
    evaluations: Any,
    selected_source_layer: Any,
    fallback: bool,
    row_index: int,
    decision_index: int,
    errors: List[str],
) -> None:
    validate_calm_candidate_evaluation_prefix(
        evaluations=evaluations,
        selected_source_layer=selected_source_layer,
        fallback=fallback,
        error_prefix="reference_trajectory_calm",
        context_suffix=":row{}:decision{}".format(row_index, decision_index),
        errors=errors,
    )


def validate_reference_generation_trajectory_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    summary: Optional[Mapping[str, Any]] = None,
    generation_binding_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    expected_generation_count: Any = None,
    expected_generated_token_count: Any = None,
    source_layer_mode: Optional[str] = None,
    expected_generation_binding_sha256: Optional[str] = None,
    expected_dataset_population_sha256: Optional[str] = None,
    expected_decoding_configuration_sha256: Optional[str] = None,
    expected_decoder_layer_count: Any = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    materialized = [dict(row) for row in rows]
    seen_generation_keys = set()
    seen_uids = set()
    token_sum = 0
    for index, row in enumerate(materialized):
        if int(row.get("schema_version", -1) or -1) != F2A_REFERENCE_TRAJECTORY_SCHEMA_VERSION:
            errors.append("reference_trajectory_schema_version_mismatch:row{}".format(index))
        if row.get("record_type") != F2A_REFERENCE_TRAJECTORY_RECORD_TYPE:
            errors.append("reference_trajectory_record_type_mismatch:row{}".format(index))
        if row.get("producer_protocol") != F2A_EVALUATION_PROTOCOL_NAME:
            errors.append("reference_trajectory_protocol_mismatch:row{}".format(index))
        for field in ("stable_sample_id", "generation_index", "generated_token_ids", "reference_decision_trace", "terminal_reason"):
            if row.get(field) in (None, ""):
                errors.append("reference_trajectory_required_field_missing:{}:row{}".format(field, index))
        if not isinstance(row.get("generation_completed"), bool):
            errors.append("reference_trajectory_generation_completed_type_invalid:row{}".format(index))
        row_decoder_layer_count = None
        if row.get("decoder_layer_count") not in (None, ""):
            try:
                row_decoder_layer_count = int(row.get("decoder_layer_count"))
            except Exception:
                errors.append("reference_trajectory_decoder_layer_count_invalid:row{}".format(index))
        if expected_decoder_layer_count not in (None, ""):
            try:
                expected_layers = int(expected_decoder_layer_count)
                if row_decoder_layer_count is None:
                    row_decoder_layer_count = expected_layers
                elif row_decoder_layer_count != expected_layers:
                    errors.append("reference_trajectory_decoder_layer_count_mismatch:row{}".format(index))
            except Exception:
                errors.append("reference_trajectory_expected_decoder_layer_count_invalid")
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list):
            errors.append("reference_trajectory_generated_token_ids_invalid:row{}".format(index))
            tokens = []
        parsed_tokens = []
        for token in tokens:
            try:
                if isinstance(token, bool):
                    raise ValueError
                parsed_tokens.append(int(token))
            except Exception:
                errors.append("reference_trajectory_generated_token_id_invalid:row{}".format(index))
                break
        declared_count = row.get("generated_token_count")
        try:
            if int(declared_count) != len(parsed_tokens):
                errors.append("reference_trajectory_generated_token_count_mismatch:row{}".format(index))
        except Exception:
            errors.append("reference_trajectory_generated_token_count_invalid:row{}".format(index))
        if not parsed_tokens:
            errors.append("reference_trajectory_generated_token_ids_empty:row{}".format(index))
        else:
            token_sum += len(parsed_tokens)
        declared_token_sha = row.get("generated_token_ids_sha256")
        if declared_token_sha != generated_token_ids_sha256(parsed_tokens):
            errors.append("reference_trajectory_generated_token_sha256_mismatch:row{}".format(index))
        decisions = row.get("reference_decision_trace")
        if not isinstance(decisions, list):
            errors.append("reference_trajectory_decision_trace_invalid:row{}".format(index))
            decisions = []
        if row.get("reference_decision_trace_sha256") != reference_decision_trace_sha256(decisions):
            errors.append("reference_trajectory_decision_trace_sha256_mismatch:row{}".format(index))
        if len(decisions) != len(parsed_tokens):
            errors.append("reference_trajectory_decision_trace_count_mismatch:row{}".format(index))
        row_source_modes = set()
        terminal_no_followup_indices = []
        for decision_index, decision in enumerate(decisions):
            if not isinstance(decision, Mapping):
                errors.append("reference_trajectory_decision_invalid:row{}:decision{}".format(index, decision_index))
                continue
            required_decision_fields = (
                "generated_token_offset",
                "generated_token_id",
                "decoder_input_position",
                "predicted_token_position",
                "reference_decision_type",
                "reference_decision_layer",
                "source_layer_mode",
                "first_crossing_source_layer",
                "fallback_trigger_candidate_layer",
                "full_depth_fallback",
                "exact_catchup_flush_occurred",
                "pending_token_count",
            )
            for field in required_decision_fields:
                if field not in decision:
                    errors.append(
                        "reference_trajectory_decision_required_field_missing:{}:row{}:decision{}".format(
                            field, index, decision_index
                        )
                    )
            try:
                offset = int(decision.get("generated_token_offset"))
            except Exception:
                offset = None
                errors.append("reference_trajectory_decision_offset_invalid:row{}:decision{}".format(index, decision_index))
            if offset != decision_index:
                errors.append("reference_trajectory_decision_offset_sequence_mismatch:row{}:decision{}".format(index, decision_index))
            try:
                decision_token = int(decision.get("generated_token_id"))
                if decision_index < len(parsed_tokens) and decision_token != parsed_tokens[decision_index]:
                    errors.append("reference_trajectory_decision_token_mismatch:row{}:decision{}".format(index, decision_index))
            except Exception:
                errors.append("reference_trajectory_decision_token_invalid:row{}:decision{}".format(index, decision_index))
            try:
                decoder_input_position = int(decision.get("decoder_input_position"))
                predicted_token_position = int(decision.get("predicted_token_position"))
                if predicted_token_position <= decoder_input_position:
                    errors.append("reference_trajectory_decision_position_order_invalid:row{}:decision{}".format(index, decision_index))
            except Exception:
                errors.append("reference_trajectory_decision_position_invalid:row{}:decision{}".format(index, decision_index))
            try:
                pending_count = int(decision.get("pending_token_count"))
                if isinstance(decision.get("pending_token_count"), bool) or pending_count < 0:
                    raise ValueError
            except Exception:
                pending_count = None
                errors.append("reference_trajectory_decision_pending_token_count_invalid:row{}:decision{}".format(index, decision_index))
            mode = decision.get("source_layer_mode")
            row_source_modes.add(mode)
            if mode not in F2A_SUPPORTED_SOURCE_LAYER_MODES:
                errors.append("reference_trajectory_decision_source_mode_invalid:row{}:decision{}".format(index, decision_index))
            if source_layer_mode not in (None, "") and mode != source_layer_mode:
                errors.append("reference_trajectory_decision_source_mode_mismatch:row{}:decision{}".format(index, decision_index))
            decision_type = decision.get("reference_decision_type")
            if bool(decision.get("terminal_no_followup")):
                terminal_no_followup_indices.append(decision_index)

            if mode == SOURCE_LAYER_MODE_FIXED:
                if decision_type not in F2A_FIXED_REFERENCE_DECISION_TYPES:
                    errors.append("reference_trajectory_fixed_decision_type_invalid:row{}:decision{}".format(index, decision_index))
                if decision.get("first_crossing_source_layer") is not None:
                    errors.append("reference_trajectory_fixed_first_crossing_not_null:row{}:decision{}".format(index, decision_index))
                if decision.get("fallback_trigger_candidate_layer") is not None:
                    errors.append("reference_trajectory_fixed_fallback_trigger_not_null:row{}:decision{}".format(index, decision_index))
                if decision.get("full_depth_fallback") is not False:
                    errors.append("reference_trajectory_fixed_full_depth_fallback_invalid:row{}:decision{}".format(index, decision_index))
                if decision_type == "fixed_exact_catchup_flush":
                    if decision.get("exact_catchup_flush_occurred") is not True:
                        errors.append("reference_trajectory_fixed_flush_flag_invalid:row{}:decision{}".format(index, decision_index))
                    if pending_count is not None and pending_count <= 0:
                        errors.append("reference_trajectory_fixed_flush_pending_nonpositive:row{}:decision{}".format(index, decision_index))
                    try:
                        if int(decision.get("reference_decision_layer")) != 6:
                            errors.append("reference_trajectory_fixed_flush_layer_mismatch:row{}:decision{}".format(index, decision_index))
                    except Exception:
                        errors.append("reference_trajectory_fixed_flush_layer_invalid:row{}:decision{}".format(index, decision_index))
                elif decision_type == "fixed_shallow_skip":
                    if decision.get("exact_catchup_flush_occurred") is not False:
                        errors.append("reference_trajectory_fixed_skip_flush_flag_invalid:row{}:decision{}".format(index, decision_index))
                    if pending_count is not None and pending_count <= 0:
                        errors.append("reference_trajectory_fixed_skip_pending_nonpositive:row{}:decision{}".format(index, decision_index))
            elif mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
                if decision_type not in F2A_CALM_REFERENCE_DECISION_TYPES:
                    errors.append("reference_trajectory_calm_decision_type_invalid:row{}:decision{}".format(index, decision_index))
                if decision_type == "first_crossing":
                    first_crossing_source = decision.get("first_crossing_source_layer")
                    if decision.get("fallback_trigger_candidate_layer") is not None:
                        errors.append("reference_trajectory_calm_first_crossing_fallback_trigger_not_null:row{}:decision{}".format(index, decision_index))
                    if first_crossing_source in (None, ""):
                        errors.append("reference_trajectory_calm_first_crossing_source_missing:row{}:decision{}".format(index, decision_index))
                    try:
                        if int(decision.get("reference_decision_layer")) != int(first_crossing_source):
                            errors.append("reference_trajectory_calm_decision_layer_mismatch:row{}:decision{}".format(index, decision_index))
                    except Exception:
                        errors.append("reference_trajectory_calm_decision_layer_invalid:row{}:decision{}".format(index, decision_index))
                    if decision.get("full_depth_fallback") is not False:
                        errors.append("reference_trajectory_calm_first_crossing_fallback_invalid:row{}:decision{}".format(index, decision_index))
                    if decision.get("exact_catchup_flush_occurred") is not True:
                        errors.append("reference_trajectory_calm_first_crossing_flush_flag_invalid:row{}:decision{}".format(index, decision_index))
                    if pending_count is not None and pending_count <= 0:
                        errors.append("reference_trajectory_calm_first_crossing_pending_nonpositive:row{}:decision{}".format(index, decision_index))
                    _validate_calm_candidate_evaluations(
                        evaluations=decision.get("candidate_evaluations"),
                        selected_source_layer=first_crossing_source,
                        fallback=False,
                        row_index=index,
                        decision_index=decision_index,
                        errors=errors,
                    )
                elif decision_type == "full_depth_fallback":
                    if decision.get("first_crossing_source_layer") is not None:
                        errors.append("reference_trajectory_calm_fallback_source_not_null:row{}:decision{}".format(index, decision_index))
                    fallback_trigger = decision.get("fallback_trigger_candidate_layer")
                    if fallback_trigger in (None, ""):
                        errors.append("reference_trajectory_calm_fallback_trigger_missing:row{}:decision{}".format(index, decision_index))
                    else:
                        try:
                            if int(fallback_trigger) != int(CALM_CANDIDATE_LAYERS[-1]):
                                errors.append("reference_trajectory_calm_fallback_trigger_mismatch:row{}:decision{}".format(index, decision_index))
                        except Exception:
                            errors.append("reference_trajectory_calm_fallback_trigger_invalid:row{}:decision{}".format(index, decision_index))
                    if row_decoder_layer_count is None:
                        errors.append("reference_trajectory_calm_fallback_decoder_layer_count_missing:row{}:decision{}".format(index, decision_index))
                    else:
                        try:
                            if int(decision.get("reference_decision_layer")) != int(row_decoder_layer_count) - 1:
                                errors.append("reference_trajectory_calm_fallback_decision_layer_mismatch:row{}:decision{}".format(index, decision_index))
                        except Exception:
                            errors.append("reference_trajectory_calm_fallback_decision_layer_invalid:row{}:decision{}".format(index, decision_index))
                    if decision.get("full_depth_fallback") is not True:
                        errors.append("reference_trajectory_calm_fallback_flag_invalid:row{}:decision{}".format(index, decision_index))
                    if decision.get("exact_catchup_flush_occurred") is not False:
                        errors.append("reference_trajectory_calm_fallback_flush_flag_invalid:row{}:decision{}".format(index, decision_index))
                    _validate_calm_candidate_evaluations(
                        evaluations=decision.get("candidate_evaluations"),
                        selected_source_layer=None,
                        fallback=True,
                        row_index=index,
                        decision_index=decision_index,
                        errors=errors,
                    )
        expected_offsets = list(range(len(parsed_tokens)))
        observed_offsets: List[int] = []
        for decision in decisions:
            if isinstance(decision, Mapping):
                try:
                    observed_offsets.append(int(decision.get("generated_token_offset")))
                except Exception:
                    pass
        if observed_offsets != expected_offsets:
            errors.append("reference_trajectory_decision_offsets_not_contiguous:row{}".format(index))
        if len(row_source_modes - {None}) > 1:
            errors.append("reference_trajectory_mixed_source_modes:row{}".format(index))
        terminal_reason = row.get("terminal_reason")
        if terminal_reason not in F2A_REFERENCE_TRAJECTORY_TERMINAL_REASONS:
            errors.append("reference_trajectory_terminal_reason_invalid:row{}".format(index))
        generation_completed = row.get("generation_completed")
        if terminal_reason == "generation_failure" and generation_completed is not False:
            errors.append("reference_trajectory_generation_failure_marked_completed:row{}".format(index))
        if terminal_reason in (
            "eos",
            "max_length",
            "stopping_criteria",
            "terminal_no_followup",
            "unknown_generation_end",
        ) and generation_completed is not True:
            errors.append("reference_trajectory_success_marked_incomplete:row{}".format(index))
        if generation_completed is False:
            errors.append("reference_trajectory_generation_incomplete:row{}".format(index))
        if terminal_no_followup_indices:
            if terminal_no_followup_indices != [len(decisions) - 1]:
                errors.append("reference_trajectory_terminal_no_followup_not_final:row{}".format(index))
            if terminal_reason != "terminal_no_followup":
                errors.append("reference_trajectory_terminal_no_followup_reason_mismatch:row{}".format(index))
        if terminal_reason == "terminal_no_followup":
            if not decisions or not bool(decisions[-1].get("terminal_no_followup")):
                errors.append("reference_trajectory_terminal_reason_missing_final_marker:row{}".format(index))
        uid = row.get("reference_generation_trajectory_row_uid")
        recomputed_uid = make_reference_generation_trajectory_row_uid(row)
        if uid in (None, ""):
            errors.append("reference_trajectory_row_uid_missing:row{}".format(index))
        elif uid != recomputed_uid:
            errors.append("reference_trajectory_row_uid_mismatch:row{}".format(index))
        if uid in seen_uids:
            errors.append("reference_trajectory_duplicate_row_uid:{}".format(uid))
        seen_uids.add(uid)
        generation_key = (row.get("stable_sample_id"), row.get("generation_index"))
        if generation_key in seen_generation_keys:
            errors.append("reference_trajectory_duplicate_generation_identity:{}:{}".format(*generation_key))
        seen_generation_keys.add(generation_key)
    if expected_generation_count is not None:
        try:
            if len(materialized) != int(expected_generation_count):
                errors.append("reference_trajectory_generation_count_mismatch")
        except Exception:
            errors.append("reference_trajectory_expected_generation_count_invalid")
    if expected_generated_token_count is not None:
        try:
            if token_sum != int(expected_generated_token_count):
                errors.append("reference_trajectory_generated_token_sum_mismatch")
        except Exception:
            errors.append("reference_trajectory_expected_token_count_invalid")
    binding_rows = list(generation_binding_rows or [])
    if binding_rows:
        binding_keys = {
            (row.get("stable_sample_id"), row.get("generation_index"))
            for row in binding_rows
        }
        trajectory_keys = {
            (row.get("stable_sample_id"), row.get("generation_index"))
            for row in materialized
        }
        if binding_keys != trajectory_keys:
            errors.append("reference_trajectory_generation_binding_population_mismatch")
    summary_payload = dict(summary or {})
    if summary_payload:
        if int(summary_payload.get("schema_version", -1) or -1) != F2A_REFERENCE_TRAJECTORY_SUMMARY_SCHEMA_VERSION:
            errors.append("reference_trajectory_summary_schema_version_mismatch")
        if summary_payload.get("record_type") != F2A_REFERENCE_TRAJECTORY_SUMMARY_RECORD_TYPE:
            errors.append("reference_trajectory_summary_record_type_mismatch")
        if summary_payload.get("producer_protocol") != F2A_EVALUATION_PROTOCOL_NAME:
            errors.append("reference_trajectory_summary_protocol_mismatch")
        if summary_payload.get("status") not in (None, "ok"):
            errors.append("reference_trajectory_summary_status_not_ok")
        if summary_payload.get("expected_generation_count") not in (None, len(materialized)):
            errors.append("reference_trajectory_summary_expected_generation_count_mismatch")
        if summary_payload.get("written_generation_count") not in (None, len(materialized)):
            errors.append("reference_trajectory_summary_written_generation_count_mismatch")
        if summary_payload.get("complete_population_recording") is not True:
            errors.append("reference_trajectory_summary_incomplete_population")
        declared_population_sha = summary_payload.get("generation_trajectory_population_sha256")
        actual_population_sha = reference_generation_trajectory_population_sha256(materialized)
        if declared_population_sha not in (None, actual_population_sha):
            errors.append("reference_trajectory_summary_population_sha256_mismatch")
        declared_ordered_sha = summary_payload.get("ordered_generation_trajectory_sha256")
        actual_ordered_sha = ordered_reference_generation_trajectory_sha256(materialized)
        if declared_ordered_sha not in (None, actual_ordered_sha):
            errors.append("reference_trajectory_summary_ordered_sha256_mismatch")
        if summary_payload.get("total_generated_token_count") not in (None, token_sum):
            errors.append("reference_trajectory_summary_token_count_mismatch")
        if expected_generation_binding_sha256 is not None and summary_payload.get("generation_binding_sha256") != expected_generation_binding_sha256:
            errors.append("reference_trajectory_summary_generation_binding_sha256_mismatch")
        if expected_dataset_population_sha256 is not None and summary_payload.get("dataset_population_sha256") != expected_dataset_population_sha256:
            errors.append("reference_trajectory_summary_dataset_population_sha256_mismatch")
        if expected_decoding_configuration_sha256 is not None and summary_payload.get("decoding_configuration_sha256") != expected_decoding_configuration_sha256:
            errors.append("reference_trajectory_summary_decoding_configuration_sha256_mismatch")
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "written_generation_count": len(materialized),
        "total_generated_token_count": token_sum,
        "generation_trajectory_population_sha256": reference_generation_trajectory_population_sha256(materialized),
        "ordered_generation_trajectory_sha256": ordered_reference_generation_trajectory_sha256(materialized),
    }


def validate_prefix_digest(event: Mapping[str, Any]) -> List[str]:
    token_ids = event.get("reference_decoder_prefix_token_ids", event.get("prefix_token_ids"))
    if token_ids in (None, ""):
        return ["f2a_prefix_token_ids_missing"]
    try:
        if len(token_ids) == 0:
            return ["f2a_prefix_token_ids_empty"]
        actual = prefix_token_sha256(token_ids)
    except Exception:
        return ["f2a_prefix_token_ids_invalid"]
    declared = event.get("prefix_token_sha256")
    if declared != actual:
        return ["f2a_prefix_digest_mismatch"]
    return []


def first_crossing_decision_payload(
    *,
    candidate_evaluations: Sequence[Mapping[str, Any]],
    selected_source_layer: int,
    threshold: float = CALM_THRESHOLD,
    threshold_comparator: str = CALM_THRESHOLD_COMPARATOR,
    candidate_policy_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "candidate_evaluations": _json_safe(list(candidate_evaluations)),
        "candidate_policy_sha256": candidate_policy_sha256,
        "selected_source_layer": int(selected_source_layer),
        "threshold": float(threshold),
        "threshold_comparator": str(threshold_comparator),
    }


def first_crossing_decision_sha256(
    *,
    candidate_evaluations: Sequence[Mapping[str, Any]],
    selected_source_layer: int,
    threshold: float = CALM_THRESHOLD,
    threshold_comparator: str = CALM_THRESHOLD_COMPARATOR,
    candidate_policy_sha256: Optional[str] = None,
) -> str:
    return canonical_json_sha256(
        first_crossing_decision_payload(
            candidate_evaluations=candidate_evaluations,
            selected_source_layer=selected_source_layer,
            threshold=threshold,
            threshold_comparator=threshold_comparator,
            candidate_policy_sha256=candidate_policy_sha256,
        )
    )


def _semantic_event_payload(event: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "schema_version": int(event.get("schema_version", F2A_SCHEDULE_SCHEMA_VERSION)),
        "record_type": event.get("record_type", F2A_EVENT_RECORD_TYPE),
        "producer_protocol": event.get("producer_protocol", F2A_EVALUATION_PROTOCOL_NAME),
    }
    for field in _SEMANTIC_EVENT_FIELDS:
        if field in payload:
            continue
        if field in event:
            payload[field] = _json_safe(event[field])
    return dict(sorted(payload.items()))


def make_f2a_event_uid(event: Mapping[str, Any]) -> str:
    """Build the deterministic semantic UID for a frozen event."""

    return canonical_json_sha256(_semantic_event_payload(event))


def schedule_semantic_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = []
    for row in rows:
        semantic = dict(_semantic_event_payload(row))
        semantic["frozen_event_uid"] = row.get("frozen_event_uid") or make_f2a_event_uid(row)
        payload.append(semantic)
    payload.sort(key=lambda item: str(item.get("frozen_event_uid")))
    return canonical_json_sha256(payload)


def reference_trajectory_sha256(
    *,
    schedule_rows: Iterable[Mapping[str, Any]],
    reference_generation_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    generation_binding_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    source_layer_mode: Optional[str],
    decoding_configuration_sha256: Optional[str],
    reference_generation_count: Any,
    reference_generated_token_count: Any,
    terminal_exclusion_accounting: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a run-level F2a reference trajectory identity.

    This digest is deliberately order-sensitive for schedule events and
    generation bindings, and deliberately path-independent.  It is not an event
    UID and must not fall back to the first schedule row's identity.
    """

    rows = list(schedule_rows)
    ordered_events = []
    for index, row in enumerate(rows):
        semantic = dict(_semantic_event_payload(row))
        semantic["event_order"] = int(index)
        semantic["frozen_event_uid"] = row.get("frozen_event_uid") or make_f2a_event_uid(row)
        ordered_events.append(semantic)
    ordered_bindings = []
    for index, row in enumerate(generation_binding_rows or []):
        ordered_bindings.append(
            {
                "binding_order": int(index),
                "stable_sample_id": row.get("stable_sample_id"),
                "selected_order": row.get("selected_order"),
                "generation_index": row.get("generation_index"),
                "raw_dataset_index": row.get("raw_dataset_index"),
                "dataset_provided_id": row.get("dataset_provided_id"),
            }
        )
    ordered_trajectory_rows = []
    ordered_generated_token_digests = []
    ordered_decision_trace_digests = []
    for index, row in enumerate(reference_generation_rows or []):
        semantic = _reference_trajectory_row_payload(row)
        semantic["generation_order"] = int(index)
        semantic["reference_generation_trajectory_row_uid"] = row.get(
            "reference_generation_trajectory_row_uid"
        ) or make_reference_generation_trajectory_row_uid(row)
        ordered_trajectory_rows.append(semantic)
        ordered_generated_token_digests.append(
            {
                "generation_order": int(index),
                "stable_sample_id": row.get("stable_sample_id"),
                "generation_index": row.get("generation_index"),
                "generated_token_ids_sha256": row.get("generated_token_ids_sha256"),
            }
        )
        ordered_decision_trace_digests.append(
            {
                "generation_order": int(index),
                "stable_sample_id": row.get("stable_sample_id"),
                "generation_index": row.get("generation_index"),
                "reference_decision_trace_sha256": row.get("reference_decision_trace_sha256"),
            }
        )
    if ordered_trajectory_rows:
        generated_token_population_identity = canonical_json_sha256(
            {
                "reference_generation_count": int(reference_generation_count or 0),
                "reference_generated_token_count": int(reference_generated_token_count or 0),
                "ordered_generated_token_digests": ordered_generated_token_digests,
            }
        )
    else:
        # Compatibility for non-paper engineering callers.  Paper-strict
        # finalization requires reference_generation_rows and never accepts
        # this counts-only fallback.
        generated_token_population_identity = canonical_json_sha256(
            {
                "reference_generation_count": int(reference_generation_count or 0),
                "reference_generated_token_count": int(reference_generated_token_count or 0),
                "generation_binding_count": len(ordered_bindings),
            }
        )
    payload = {
        "protocol": F2A_EVALUATION_PROTOCOL_NAME,
        "source_layer_mode": source_layer_mode,
        "decoding_configuration_sha256": decoding_configuration_sha256,
        "reference_generation_count": int(reference_generation_count or 0),
        "reference_generated_token_count": int(reference_generated_token_count or 0),
        "ordered_generation_bindings": ordered_bindings,
        "ordered_reference_generation_trajectory_rows": ordered_trajectory_rows,
        "ordered_generated_token_sequence_digests": ordered_generated_token_digests,
        "ordered_reference_decision_trace_digests": ordered_decision_trace_digests,
        "ordered_frozen_event_semantics": ordered_events,
        "terminal_exclusion_accounting": _json_safe(dict(terminal_exclusion_accounting or {})),
        "source_decision_schedule_identity": canonical_json_sha256(
            [
                {
                    "event_order": int(index),
                    "stable_sample_id": row.get("stable_sample_id"),
                    "generation_index": row.get("generation_index"),
                    "decoder_position": row.get("decoder_position"),
                    "source_layer_mode": row.get("source_layer_mode"),
                    "source_layer": row.get("source_layer"),
                    "candidate_policy_sha256": row.get("candidate_policy_sha256"),
                    "target_layers": row.get("target_layers"),
                    "pending_token_positions": row.get("pending_token_positions"),
                    "cache_positions": row.get("cache_positions"),
                    "event_timing": row.get("event_timing"),
                }
                for index, row in enumerate(rows)
            ]
        ),
        "generated_token_population_identity": generated_token_population_identity,
    }
    return canonical_json_sha256(payload)


def validate_f2a_event(event: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    try:
        schema_version = int(event.get("schema_version", -1))
    except Exception:
        schema_version = -1
    if schema_version in (F2A_LEGACY_SCHEDULE_SCHEMA_VERSION, F2A_LEGACY_SCHEDULE_SCHEMA_V2):
        errors.append("f2a_legacy_schema_not_primary")
    elif schema_version != F2A_SCHEDULE_SCHEMA_VERSION:
        errors.append("f2a_schedule_schema_version_mismatch")
    if event.get("record_type") != F2A_EVENT_RECORD_TYPE:
        errors.append("f2a_record_type_mismatch")
    if event.get("producer_protocol") != F2A_EVALUATION_PROTOCOL_NAME:
        errors.append("f2a_producer_protocol_mismatch")
    for field in F2A_EVENT_REQUIRED_FIELDS:
        if event.get(field) in (None, ""):
            errors.append("f2a_required_field_missing:{}".format(field))
    errors.extend(validate_prefix_digest(event))
    source_mode = event.get("source_layer_mode")
    if source_mode not in F2A_SUPPORTED_SOURCE_LAYER_MODES:
        errors.append("f2a_source_layer_mode_unsupported")
    decoder_layer_count = _as_int(event.get("decoder_layer_count"), "decoder_layer_count", errors)
    current_input_position = _as_int(event.get("current_decoder_input_position"), "current_decoder_input_position", errors)
    predicted_position = _as_int(event.get("predicted_token_position"), "predicted_token_position", errors)
    source_layer = _as_int(event.get("source_layer"), "source_layer", errors)
    last_exact = _as_int(event.get("last_exact_kv_layer"), "last_exact_kv_layer", errors)
    first_missing = _as_int(event.get("first_missing_target_layer"), "first_missing_target_layer", errors)
    target_layers = _as_int_list(event.get("target_layers"), "target_layers", errors)
    pending = _as_int_list(event.get("pending_token_positions"), "pending_token_positions", errors)
    restore_indices = _as_int_list(event.get("restore_relative_indices"), "restore_relative_indices", errors)
    cache_positions = _as_int_list(event.get("cache_positions"), "cache_positions", errors)
    if event.get("full_depth_fallback") is True or source_mode == "full_depth_fallback":
        errors.append("f2a_fallback_event_not_replayable")
    if decoder_layer_count is not None and decoder_layer_count <= 0:
        errors.append("f2a_decoder_layer_count_invalid")
    if source_layer is not None:
        if source_mode == SOURCE_LAYER_MODE_FIXED and source_layer != 6:
            errors.append("f2a_fixed_source_layer_mismatch")
        if source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
            expected_candidate_policy_sha = calm_policy_sha256()
            if source_layer not in tuple(int(item) for item in CALM_CANDIDATE_LAYERS):
                errors.append("f2a_calm_source_layer_mismatch")
            if event.get("candidate_policy_sha256") in (None, ""):
                errors.append("f2a_calm_candidate_policy_sha_missing")
            elif event.get("candidate_policy_sha256") != expected_candidate_policy_sha:
                errors.append("f2a_calm_candidate_policy_sha_mismatch")
            if event.get("policy_name") != CALM_POLICY_NAME:
                errors.append("f2a_calm_policy_name_mismatch")
            if event.get("candidate_layers") not in (list(CALM_CANDIDATE_LAYERS), tuple(CALM_CANDIDATE_LAYERS)):
                errors.append("f2a_calm_candidate_layers_mismatch")
            try:
                threshold = float(event.get("threshold"))
            except Exception:
                threshold = None
            if threshold != float(CALM_THRESHOLD):
                errors.append("f2a_calm_threshold_mismatch")
            if event.get("threshold_comparator") != CALM_THRESHOLD_COMPARATOR:
                errors.append("f2a_calm_threshold_comparator_mismatch")
            if event.get("confidence_compute_dtype") != CALM_CONFIDENCE_COMPUTE_DTYPE:
                errors.append("f2a_calm_confidence_dtype_mismatch")
            if bool(event.get("adaptive_threshold", False)):
                errors.append("f2a_calm_adaptive_threshold_enabled")
            if event.get("source_confidence") in (None, ""):
                errors.append("f2a_calm_source_confidence_missing")
            evaluations = event.get("candidate_evaluations")
            if source_layer is not None:
                prefix_validation = validate_calm_candidate_evaluation_prefix(
                    evaluations=evaluations,
                    selected_source_layer=source_layer,
                    fallback=False,
                    error_prefix="f2a_calm",
                    errors=errors,
                )
                try:
                    selected_confidence = prefix_validation.get("selected_confidence")
                    if selected_confidence is None or _float32_value(event.get("source_confidence")) != float(selected_confidence):
                        errors.append("f2a_calm_source_confidence_mismatch")
                except Exception:
                    errors.append("f2a_calm_source_confidence_mismatch")
                if isinstance(evaluations, list):
                    try:
                        expected_decision_sha = first_crossing_decision_sha256(
                            candidate_evaluations=evaluations,
                            selected_source_layer=source_layer,
                            threshold=float(CALM_THRESHOLD),
                            threshold_comparator=CALM_THRESHOLD_COMPARATOR,
                            candidate_policy_sha256=event.get("candidate_policy_sha256"),
                        )
                    except Exception:
                        expected_decision_sha = None
                        errors.append("f2a_calm_first_crossing_decision_sha_noncanonical")
                    if expected_decision_sha is not None and event.get("first_crossing_decision_sha256") != expected_decision_sha:
                        errors.append("f2a_calm_first_crossing_decision_sha_mismatch")
            if event.get("first_crossing_decision_sha256") in (None, ""):
                errors.append("f2a_calm_first_crossing_decision_sha_missing")
            if event.get("event_timing") != "deferred_followup":
                errors.append("f2a_calm_event_timing_mismatch")
            for field in (
                "restored_token_position",
                "followup_decoder_input_token_id",
                "followup_decoder_input_position",
                "followup_predicted_token_position",
                "followup_prefix_token_sha256",
                "followup_reference_decision_layer",
                "followup_reference_decision_type",
            ):
                if event.get(field) in (None, ""):
                    errors.append("f2a_calm_followup_field_missing:{}".format(field))
            try:
                if int(event.get("followup_decoder_input_position")) != int(event.get("restored_token_position")) + 1:
                    errors.append("f2a_calm_followup_position_mismatch")
                if int(event.get("followup_predicted_token_position")) != int(event.get("followup_decoder_input_position")) + 1:
                    errors.append("f2a_calm_followup_predicted_position_mismatch")
            except Exception:
                errors.append("f2a_calm_followup_position_invalid")
        elif source_mode == SOURCE_LAYER_MODE_FIXED and event.get("event_timing") not in (None, "", "same_flush"):
            errors.append("f2a_fixed_event_timing_mismatch")
    if current_input_position is not None and predicted_position is not None:
        if predicted_position != current_input_position + 1:
            errors.append("f2a_predicted_position_mismatch")
    if source_layer is not None and last_exact is not None and last_exact != source_layer - 1:
        errors.append("f2a_last_exact_kv_layer_mismatch")
    if source_layer is not None and first_missing is not None and first_missing != source_layer:
        errors.append("f2a_first_missing_target_layer_mismatch")
    if source_layer is not None and decoder_layer_count is not None and target_layers is not None:
        expected_targets = list(range(source_layer, decoder_layer_count))
        if target_layers != expected_targets:
            errors.append("f2a_target_layer_incomplete")
        if any(layer < 0 or layer >= decoder_layer_count for layer in target_layers):
            errors.append("f2a_target_layer_out_of_range")
    if pending is not None and cache_positions is not None and len(pending) != len(cache_positions):
        errors.append("f2a_pending_cache_position_mismatch")
    if pending is not None and restore_indices is not None and len(pending) != len(restore_indices):
        errors.append("f2a_pending_restore_index_mismatch")
    if pending is not None and restore_indices is not None:
        if restore_indices != list(range(len(restore_indices))):
            errors.append("f2a_restore_relative_indices_noncanonical")
        if len(set(pending)) != len(pending):
            errors.append("f2a_duplicate_pending_token_position")
    if cache_positions is not None:
        if len(set(cache_positions)) != len(cache_positions):
            errors.append("f2a_duplicate_cache_position")
        if any(position < 0 for position in cache_positions):
            errors.append("f2a_cache_position_out_of_range")
        if any(cache_positions[idx] >= cache_positions[idx + 1] for idx in range(len(cache_positions) - 1)):
            errors.append("f2a_cache_positions_not_strictly_increasing")
    if pending is not None and len(pending) == 0:
        errors.append("f2a_pending_token_positions_empty")
    uid = event.get("frozen_event_uid")
    try:
        expected_uid = make_f2a_event_uid(event)
    except Exception:
        expected_uid = None
        errors.append("f2a_event_uid_noncanonical")
    if uid not in (None, "") and expected_uid is not None and uid != expected_uid:
        errors.append("f2a_event_uid_mismatch")
    return errors


def validate_f2a_schedule_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rows_list = list(rows)
    errors: List[str] = []
    seen = set()
    source_counts: Counter = Counter()
    for index, row in enumerate(rows_list):
        row_errors = validate_f2a_event(row)
        errors.extend("{}:row{}".format(error, index) for error in row_errors)
        try:
            uid = row.get("frozen_event_uid") or make_f2a_event_uid(row)
        except Exception:
            uid = None
            errors.append("f2a_event_uid_noncanonical:row{}".format(index))
        if uid in seen:
            errors.append("f2a_duplicate_event_uid:{}".format(uid))
        if uid is not None:
            seen.add(uid)
        if row.get("source_layer") not in (None, ""):
            source_counts[str(int(row["source_layer"]))] += 1
    try:
        semantic_sha = schedule_semantic_sha256(rows_list)
    except Exception:
        semantic_sha = None
        errors.append("f2a_schedule_semantic_sha256_noncanonical")
    if summary is not None:
        expected = summary.get("expected_record_count")
        written = summary.get("written_record_count", summary.get("row_count"))
        skipped = summary.get("skipped_record_count")
        complete = summary.get("complete_population_recording")
        if expected is None:
            errors.append("f2a_expected_record_count_missing")
        elif int(expected) != len(rows_list):
            errors.append("f2a_expected_record_count_mismatch")
        if written is None:
            errors.append("f2a_written_record_count_missing")
        elif int(written) != len(rows_list):
            errors.append("f2a_written_record_count_mismatch")
        if skipped is None or int(skipped) != 0:
            errors.append("f2a_skipped_record_count_nonzero")
        if complete is not True:
            errors.append("f2a_incomplete_population_recording")
        if semantic_sha is not None and summary.get("schedule_semantic_sha256") not in (None, semantic_sha):
            errors.append("f2a_schedule_semantic_sha256_mismatch")
    return {
        "status": "ok" if not errors else "failed",
        "schema_version": F2A_SUMMARY_SCHEMA_VERSION,
        "event_count": len(rows_list),
        "unique_event_uid_count": len(seen),
        "source_layer_counts": dict(sorted(source_counts.items())),
        "schedule_semantic_sha256": semantic_sha,
        "errors": errors,
    }


def assert_f2a_schedule_valid(rows: Iterable[Mapping[str, Any]], *, summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    validation = validate_f2a_schedule_rows(rows, summary=summary)
    if validation["status"] != "ok":
        raise F2AFrozenScheduleError("f2a_schedule_validation_failed: {}".format(validation["errors"]))
    return validation


def select_artifact_gap_bin(artifact: Mapping[str, Any], source_layer: int, target_layer: int) -> str:
    gap = int(target_layer) - int(source_layer)
    for item in artifact.get("fit_config", {}).get("gap_bins", []):
        if int(item.get("start")) <= gap <= int(item.get("end")):
            return str(item.get("label"))
    raise F2AFrozenScheduleError("f2a_gap_bin_missing:{}".format(gap))


def _project_to_kv(
    hidden: torch.Tensor,
    target_layer_norm: torch.nn.Module,
    key_projection: torch.nn.Module,
    value_projection: torch.nn.Module,
    *,
    num_heads: int,
    d_kv: int,
    output_device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    param = next(key_projection.parameters(), None)
    if param is None:
        param = next(value_projection.parameters(), None)
    module_device = param.device if param is not None else hidden.device
    module_dtype = param.dtype if param is not None else hidden.dtype
    normed = target_layer_norm(hidden.to(device=module_device, dtype=module_dtype))
    key_flat = key_projection(normed)
    value_flat = value_projection(normed)
    expected = int(num_heads) * int(d_kv)
    if key_flat.shape[-1] != expected or value_flat.shape[-1] != expected:
        raise F2AFrozenScheduleError("f2a_projection_output_dim_mismatch")
    batch, seq_len, _ = key_flat.shape
    key = key_flat.view(batch, seq_len, int(num_heads), int(d_kv)).transpose(1, 2).contiguous()
    value = value_flat.view(batch, seq_len, int(num_heads), int(d_kv)).transpose(1, 2).contiguous()
    output_device = output_device or module_device
    output_dtype = output_dtype or module_dtype
    key = key.to(device=output_device, dtype=output_dtype)
    value = value.to(device=output_device, dtype=output_dtype)
    if not torch.isfinite(key).all().item() or not torch.isfinite(value).all().item():
        raise F2AFrozenScheduleError("f2a_projected_kv_nonfinite")
    return key, value


def restore_f2a_method_from_hidden(
    *,
    method: str,
    artifact: Mapping[str, Any],
    source_hidden: torch.Tensor,
    source_layer: int,
    target_layer: int,
    target_layer_norm: torch.nn.Module,
    key_projection: torch.nn.Module,
    value_projection: torch.nn.Module,
    threshold: Any = CALM_THRESHOLD,
    output_device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Restore one target layer's current-token K/V for an F2a component."""

    key, value, metadata = restore_f2a_method_block_from_hidden(
        method=method,
        artifact=artifact,
        source_hidden=source_hidden,
        source_layer=source_layer,
        target_layer=target_layer,
        target_layer_norm=target_layer_norm,
        key_projection=key_projection,
        value_projection=value_projection,
        threshold=threshold,
        output_device=output_device,
        output_dtype=output_dtype,
    )
    if int(key.shape[2]) != 1:
        raise F2AFrozenScheduleError("f2a_source_hidden_shape_invalid")
    return key, value, metadata


def restore_f2a_method_block_from_hidden(
    *,
    method: str,
    artifact: Mapping[str, Any],
    source_hidden: torch.Tensor,
    source_layer: int,
    target_layer: int,
    target_layer_norm: torch.nn.Module,
    key_projection: torch.nn.Module,
    value_projection: torch.nn.Module,
    threshold: Any = CALM_THRESHOLD,
    output_device: Optional[torch.device] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Restore one target layer's K/V block for all pending F2a tokens."""

    if method not in F2A_REQUIRED_METHODS:
        raise F2AFrozenScheduleError("f2a_method_unsupported:{}".format(method))
    validate_phase3c_policy_artifact(artifact)
    model_spec = artifact.get("model_spec") or {}
    num_heads = int(model_spec.get("num_heads"))
    d_kv = int(model_spec.get("d_kv"))
    source_layer = int(source_layer)
    target_layer = int(target_layer)
    if target_layer < source_layer:
        raise F2AFrozenScheduleError("f2a_target_before_source")
    if source_hidden.ndim != 3 or int(source_hidden.shape[0]) != 1 or int(source_hidden.shape[1]) < 1:
        raise F2AFrozenScheduleError("f2a_source_hidden_shape_invalid")
    if not torch.isfinite(source_hidden).all().item():
        raise F2AFrozenScheduleError("f2a_source_hidden_nonfinite")

    hidden_hat = source_hidden
    hidden_affine = False
    k_correction = False
    v_correction = False
    gap_bin = None
    if target_layer > source_layer and method in {
        F2A_METHOD_EXIT_CONDITIONED_HIDDEN_RESTORATION,
        F2A_METHOD_FINAL_KV_RESTORATION,
    }:
        hidden_params = get_hidden_layer_pair_parameters(artifact, threshold, source_layer, target_layer)
        hidden_hat = apply_hidden_diagonal_affine(source_hidden, hidden_params)
        hidden_affine = True
    key, value = _project_to_kv(
        hidden_hat,
        target_layer_norm,
        key_projection,
        value_projection,
        num_heads=num_heads,
        d_kv=d_kv,
        output_device=output_device,
        output_dtype=output_dtype,
    )
    if target_layer > source_layer and method == F2A_METHOD_FINAL_KV_RESTORATION:
        k_params = get_k_layer_pair_parameters(artifact, threshold, source_layer, target_layer)
        gap_bin = select_artifact_gap_bin(artifact, source_layer, target_layer)
        v_params = get_v_gap_bin_parameters(artifact, threshold, gap_bin)
        key = apply_k_head_channel_affine(key, k_params).to(device=key.device, dtype=key.dtype)
        value = apply_v_headwise_procrustes(value, v_params).to(device=value.device, dtype=value.dtype)
        k_correction = True
        v_correction = True
    metadata = {
        "method": method,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "gap": target_layer - source_layer,
        "gap_bin": gap_bin,
        "same_layer_target_native_projection": target_layer == source_layer,
        "source_hidden_semantics": "raw_h_s",
        "hidden_affine_applied": hidden_affine,
        "k_correction_applied": k_correction,
        "v_correction_applied": v_correction,
        "restored_key_shape": list(key.shape),
        "restored_value_shape": list(value.shape),
        "pending_token_count": int(key.shape[2]),
    }
    return key, value, metadata


def _normalize_nonnegative_divergence(
    raw_value: float,
    *,
    metric_name: str,
    negative_roundoff_tolerance: float,
) -> Tuple[float, bool]:
    """Normalize a mathematically-nonnegative divergence (KL or JS) that was
    computed in float64.

    KL divergence and Jensen-Shannon divergence are mathematically
    nonnegative, but a float64 ``sum(p * (log_p - log_q))`` reduction over
    (near-)identical distributions can still land a hair below zero from
    ordinary floating-point roundoff. A raw value inside
    ``[-negative_roundoff_tolerance, 0)`` is that roundoff and is normalized
    to ``0.0``; a raw value more negative than that is not roundoff, it is
    evidence of an implementation or numerical-invariant bug, and must fail
    closed rather than being silently concealed by an unconditional
    ``max(raw, 0.0)``. ``negative_roundoff_tolerance`` is not a parity
    acceptance tolerance and is deliberately kept many orders of magnitude
    below the frozen exact-shadow parity tolerances.

    Returns ``(normalized_value, was_clamped)``.
    """

    if not math.isfinite(raw_value):
        raise F2AFrozenScheduleError(
            "f2a_divergence_nonfinite:{}:{}".format(metric_name, raw_value)
        )
    if raw_value >= 0.0:
        return raw_value, False
    if raw_value >= -negative_roundoff_tolerance:
        return 0.0, True
    raise F2AFrozenScheduleError(
        "f2a_divergence_negative_invariant_violation:{}:{}:tolerance={}".format(
            metric_name, raw_value, negative_roundoff_tolerance
        )
    )


def _f2a_probability_distribution_metrics(
    reference_logits_f32: torch.Tensor, candidate_logits_f32: torch.Tensor
) -> Dict[str, Any]:
    """Shared float64 log-softmax/softmax/KL/JS computation.

    Both ``f2a_logit_metrics()`` (public, clamped values) and the optional
    exact-shadow failure diagnostics (raw pre-clamp values) are derived from
    this single computation so the function does not mix inconsistent
    probability precisions and there is one source of truth for the
    distribution math.
    """

    reference_f64 = reference_logits_f32.reshape(-1).to(dtype=torch.float64)
    candidate_f64 = candidate_logits_f32.reshape(-1).to(dtype=torch.float64)
    ref_logp = torch.log_softmax(reference_f64, dim=-1)
    cand_logp = torch.log_softmax(candidate_f64, dim=-1)
    ref_p = ref_logp.exp()
    cand_p = cand_logp.exp()
    m = 0.5 * (ref_p + cand_p)
    m_log = torch.log(torch.clamp(m, min=torch.finfo(torch.float64).tiny))
    raw_reference_to_candidate_kl = float(torch.sum(ref_p * (ref_logp - cand_logp)).item())
    raw_candidate_to_reference_kl = float(torch.sum(cand_p * (cand_logp - ref_logp)).item())
    raw_jensen_shannon_divergence = float(
        (0.5 * torch.sum(ref_p * (ref_logp - m_log)) + 0.5 * torch.sum(cand_p * (cand_logp - m_log))).item()
    )
    return {
        "ref_p": ref_p,
        "cand_p": cand_p,
        "raw_reference_to_candidate_kl": raw_reference_to_candidate_kl,
        "raw_candidate_to_reference_kl": raw_candidate_to_reference_kl,
        "raw_jensen_shannon_divergence": raw_jensen_shannon_divergence,
    }


def f2a_logit_metrics(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> Dict[str, Any]:
    """Compute compact downstream next-token distribution metrics.

    Direct logit-space metrics (MSE, relative L2, cosine similarity, and
    top-1 token identity/agreement, which is derived directly from the
    logits via argmax rather than from softmax probabilities) stay on the
    existing float32 input-normalization contract.

    Probability-distribution metrics (log-softmax, softmax, forward/reverse
    KL, Jensen-Shannon divergence, and every probability-derived value) are
    computed in float64, all from the same float64 probability tensors, so
    near-identical logits do not trigger the float32 softmax/log-softmax
    reduction cancellation that produced tiny negative KL/JS values (and one
    reverse-KL landing within the frozen 1e-7 tolerance's own margin) on
    real exact-shadow GPU smokes despite the direct logit metrics already
    showing near machine-precision agreement. This does not change how the
    logits themselves were produced (position bias, cache replay).
    """

    reference = reference_logits.detach().to(dtype=torch.float32)
    candidate = candidate_logits.detach().to(dtype=torch.float32)
    if reference.shape != candidate.shape:
        raise F2AFrozenScheduleError("f2a_logit_shape_mismatch")
    if reference.numel() == 0:
        raise F2AFrozenScheduleError("f2a_logit_empty")
    if not torch.isfinite(reference).all().item() or not torch.isfinite(candidate).all().item():
        raise F2AFrozenScheduleError("f2a_logit_nonfinite")
    diff = reference - candidate
    ref_norm = torch.linalg.vector_norm(reference)
    cand_norm = torch.linalg.vector_norm(candidate)
    relative = torch.linalg.vector_norm(diff) / torch.clamp(ref_norm, min=torch.finfo(torch.float32).eps)
    cosine = torch.nn.functional.cosine_similarity(reference.reshape(1, -1), candidate.reshape(1, -1), dim=-1)[0]

    ref_logit_flat = reference.reshape(-1)
    cand_logit_flat = candidate.reshape(-1)
    ref_top1 = int(torch.argmax(ref_logit_flat).item())
    cand_top1 = int(torch.argmax(cand_logit_flat).item())
    top1_agreement = 1.0 if ref_top1 == cand_top1 else 0.0

    distribution = _f2a_probability_distribution_metrics(reference, candidate)
    ref_p = distribution["ref_p"]
    cand_p = distribution["cand_p"]
    ref_to_cand_kl, _ = _normalize_nonnegative_divergence(
        distribution["raw_reference_to_candidate_kl"],
        metric_name="reference_to_candidate_kl",
        negative_roundoff_tolerance=F2A_NEGATIVE_ROUNDOFF_TOLERANCE,
    )
    cand_to_ref_kl, _ = _normalize_nonnegative_divergence(
        distribution["raw_candidate_to_reference_kl"],
        metric_name="candidate_to_reference_kl",
        negative_roundoff_tolerance=F2A_NEGATIVE_ROUNDOFF_TOLERANCE,
    )
    js, _ = _normalize_nonnegative_divergence(
        distribution["raw_jensen_shannon_divergence"],
        metric_name="jensen_shannon_divergence",
        negative_roundoff_tolerance=F2A_NEGATIVE_ROUNDOFF_TOLERANCE,
    )

    ref_top_values, _ = torch.topk(ref_p, k=min(2, ref_p.numel()))
    cand_top_values, _ = torch.topk(cand_p, k=min(2, cand_p.numel()))
    ref_top1_probability = float(ref_p[ref_top1].item())
    cand_top1_probability = float(cand_p[cand_top1].item())
    ref_margin = float(ref_top_values[0].item() - (ref_top_values[1].item() if ref_top_values.numel() > 1 else 0.0))
    cand_margin = float(cand_top_values[0].item() - (cand_top_values[1].item() if cand_top_values.numel() > 1 else 0.0))
    metrics = {
        "logit_mse": float(torch.mean(diff * diff).item()),
        "logit_relative_l2": float(relative.item()),
        "logit_cosine": float(cosine.item()),
        "reference_to_candidate_kl": ref_to_cand_kl,
        "candidate_to_reference_kl": cand_to_ref_kl,
        "jensen_shannon_divergence": js,
        "top1_agreement": top1_agreement,
        "reference_top1_token_id": ref_top1,
        "candidate_top1_token_id": cand_top1,
        "reference_top1_probability": ref_top1_probability,
        "candidate_reference_token_probability": float(cand_p[ref_top1].item()),
        "top1_probability_delta": float(cand_top1_probability - ref_top1_probability),
        "top1_margin_delta": float(cand_margin - ref_margin),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise F2AFrozenScheduleError("f2a_metric_nonfinite")
    return metrics


def validate_exact_shadow_parity(
    metrics: Mapping[str, Any],
    *,
    dtype: Any = torch.float32,
) -> Dict[str, Any]:
    dtype_name = str(dtype).replace("torch.", "")
    if dtype_name not in F2A_PARITY_TOLERANCES:
        dtype_name = "float32"
    tolerances = F2A_PARITY_TOLERANCES[dtype_name]
    errors: List[str] = []
    try:
        if float(metrics.get("logit_mse")) > tolerances["logit_mse"]:
            errors.append("f2a_exact_shadow_logit_mse_tolerance_failure")
        if float(metrics.get("logit_relative_l2")) > tolerances["logit_relative_l2"]:
            errors.append("f2a_exact_shadow_relative_l2_tolerance_failure")
        if (1.0 - float(metrics.get("logit_cosine"))) > tolerances["one_minus_logit_cosine"]:
            errors.append("f2a_exact_shadow_cosine_tolerance_failure")
        if float(metrics.get("reference_to_candidate_kl")) > tolerances["reference_to_candidate_kl"]:
            errors.append("f2a_exact_shadow_ref_to_shadow_kl_tolerance_failure")
        if float(metrics.get("candidate_to_reference_kl")) > tolerances["candidate_to_reference_kl"]:
            errors.append("f2a_exact_shadow_shadow_to_ref_kl_tolerance_failure")
        if float(metrics.get("jensen_shannon_divergence")) > tolerances["jensen_shannon_divergence"]:
            errors.append("f2a_exact_shadow_js_tolerance_failure")
        if float(metrics.get("top1_agreement")) != 1.0:
            errors.append("f2a_exact_shadow_top1_mismatch")
    except Exception as exc:
        errors.append("f2a_exact_shadow_metric_malformed:{}".format(exc))
    return {
        "status": "ok" if not errors else "failed",
        "dtype": dtype_name,
        "tolerances": dict(tolerances),
        "metrics": dict(metrics),
        "errors": errors,
    }


def validate_f2a_reference_position_bias(
    reference_position_bias: Any,
    *,
    expected_key_length: int,
    expected_num_heads: Optional[int] = None,
    expected_dtype: Any = None,
    expected_device: Any = None,
) -> torch.Tensor:
    """Validate the exact reference position-bias slice used to seed fixed-layer
    F2a exact-shadow/candidate replay, and return a defensive clone.

    Fails closed (raises ``F2AFrozenScheduleError``) rather than allowing a
    caller to fall back to a silently synthesized zero position bias. The
    expected shape is ``(batch-or-1, num_heads, 1, expected_key_length)``: a
    single current-query row (fixed-layer shadow replay is always a
    single-query forward) covering every key position the reference exact
    catch-up attended over at the fixed source layer.
    """

    if reference_position_bias is None:
        raise F2AFrozenScheduleError("f2a_reference_position_bias_unavailable")
    if not isinstance(reference_position_bias, torch.Tensor):
        raise F2AFrozenScheduleError("f2a_reference_position_bias_unavailable:not_a_tensor")
    if reference_position_bias.dim() != 4:
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_shape_mismatch:rank={}".format(reference_position_bias.dim())
        )
    query_length = int(reference_position_bias.shape[2])
    if query_length != 1:
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_shape_mismatch:query_length={}".format(query_length)
        )
    if expected_num_heads is not None and int(reference_position_bias.shape[1]) != int(expected_num_heads):
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_shape_mismatch:num_heads={},expected={}".format(
                reference_position_bias.shape[1], expected_num_heads
            )
        )
    key_length = int(reference_position_bias.shape[3])
    if key_length != int(expected_key_length):
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_key_length_mismatch:got={},expected={}".format(
                key_length, expected_key_length
            )
        )
    if expected_dtype is not None and reference_position_bias.dtype != expected_dtype:
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_shape_mismatch:dtype={},expected={}".format(
                reference_position_bias.dtype, expected_dtype
            )
        )
    if expected_device is not None and torch.device(reference_position_bias.device) != torch.device(expected_device):
        raise F2AFrozenScheduleError(
            "f2a_reference_position_bias_shape_mismatch:device={},expected={}".format(
                reference_position_bias.device, expected_device
            )
        )
    if not torch.isfinite(reference_position_bias).all().item():
        raise F2AFrozenScheduleError("f2a_reference_position_bias_nonfinite")
    return reference_position_bias.detach().clone()


def build_f2a_exact_shadow_parity_diagnostics(
    *,
    parity: Mapping[str, Any],
    reference_logits: torch.Tensor,
    shadow_logits: torch.Tensor,
    reference_position_bias: Optional[torch.Tensor] = None,
    shadow_position_bias: Optional[torch.Tensor] = None,
    reference_position_bias_source: Optional[str] = None,
    shadow_cache_sequence_length: Optional[int] = None,
    current_query_length: Optional[int] = None,
    expected_key_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Compact, JSON-safe exact-shadow parity failure diagnostics.

    Never includes full-vocabulary logits; only shapes/dtypes and scalar
    summary statistics, so it is safe to always attach to a fail-closed
    parity error without weakening or replacing that error.
    """

    reference = reference_logits.detach().to(dtype=torch.float32)
    shadow = shadow_logits.detach().to(dtype=torch.float32)
    diff = (reference - shadow).abs()
    metrics = dict(parity.get("metrics") or {})
    diagnostics: Dict[str, Any] = {
        "metrics": metrics,
        "tolerances": dict(parity.get("tolerances") or {}),
        "errors": list(parity.get("errors") or []),
        "reference_logit_shape": list(reference_logits.shape),
        "reference_logit_dtype": str(reference_logits.dtype).replace("torch.", ""),
        "shadow_logit_shape": list(shadow_logits.shape),
        "shadow_logit_dtype": str(shadow_logits.dtype).replace("torch.", ""),
        "maximum_absolute_logit_difference": float(diff.max().item()) if diff.numel() else None,
        "mean_absolute_logit_difference": float(diff.mean().item()) if diff.numel() else None,
        "reference_top1_token_id": metrics.get("reference_top1_token_id"),
        "shadow_top1_token_id": metrics.get("candidate_top1_token_id"),
        "reference_top1_probability": metrics.get("reference_top1_probability"),
        "shadow_probability_of_reference_top1": metrics.get("candidate_reference_token_probability"),
        "reference_position_bias_shape": (
            list(reference_position_bias.shape) if isinstance(reference_position_bias, torch.Tensor) else None
        ),
        "shadow_position_bias_shape": (
            list(shadow_position_bias.shape) if isinstance(shadow_position_bias, torch.Tensor) else None
        ),
        "reference_position_bias_source": reference_position_bias_source,
        "shadow_cache_sequence_length": shadow_cache_sequence_length,
        "current_query_length": current_query_length,
        "expected_key_length": expected_key_length,
    }

    # Optional nested diagnostic block only: recomputed independently from the
    # same (already-finite, already-validated) logits so it can never raise
    # or replace the primary fail-closed parity error above. Not part of the
    # component-record schema; this whole function is only invoked on the
    # exact-shadow failure path and its output is attached as a kv_trace
    # sidecar, never as a required record field.
    try:
        distribution = _f2a_probability_distribution_metrics(reference, shadow)
        clamped_fields = []
        for field_name, raw_key in (
            ("reference_to_candidate_kl", "raw_reference_to_candidate_kl"),
            ("candidate_to_reference_kl", "raw_candidate_to_reference_kl"),
            ("jensen_shannon_divergence", "raw_jensen_shannon_divergence"),
        ):
            raw_value = distribution[raw_key]
            if math.isfinite(raw_value) and -F2A_NEGATIVE_ROUNDOFF_TOLERANCE <= raw_value < 0.0:
                clamped_fields.append(field_name)
        diagnostics["logit_distribution_numerics"] = {
            "compute_dtype": "float64",
            "negative_roundoff_tolerance": F2A_NEGATIVE_ROUNDOFF_TOLERANCE,
            "raw_reference_to_candidate_kl": distribution["raw_reference_to_candidate_kl"],
            "raw_candidate_to_reference_kl": distribution["raw_candidate_to_reference_kl"],
            "raw_jensen_shannon_divergence": distribution["raw_jensen_shannon_divergence"],
            "clamped_fields": clamped_fields,
        }
    except Exception:
        pass

    return diagnostics


def cache_tensor_digest(cache: Any) -> str:
    """Digest cache tensor content/shape/dtype without relying on paths."""

    payload = []
    for layer_idx, state in enumerate(cache or []):
        state_payload = []
        for tensor_idx, tensor in enumerate(state or ()):
            if isinstance(tensor, torch.Tensor):
                tensor_cpu = tensor.detach().cpu().contiguous()
                state_payload.append(
                    {
                        "tensor_idx": tensor_idx,
                        "shape": list(tensor_cpu.shape),
                        "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
                        "sha256": canonical_json_sha256(
                            {
                                "shape": list(tensor_cpu.shape),
                                "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
                                "bytes": tensor_cpu.numpy().tobytes().hex(),
                            }
                        ),
                    }
                )
            else:
                state_payload.append({"tensor_idx": tensor_idx, "value": None if tensor is None else str(type(tensor))})
        payload.append({"layer_idx": layer_idx, "state": state_payload})
    return canonical_json_sha256(payload)


def clone_cache(cache: Any) -> List[Tuple[Any, ...]]:
    cloned = []
    for state in cache or []:
        cloned.append(tuple(item.detach().clone() if isinstance(item, torch.Tensor) else item for item in (state or ())))
    return cloned


def cache_storage_ptrs(cache: Any) -> set:
    ptrs = set()
    for state in cache or []:
        for tensor in state or ():
            if isinstance(tensor, torch.Tensor):
                ptrs.add(int(tensor.untyped_storage().data_ptr()))
    return ptrs


def caches_share_writable_storage(left: Any, right: Any) -> bool:
    return bool(cache_storage_ptrs(left) & cache_storage_ptrs(right))


def _cache_tensor_clone(tensor: Any) -> Any:
    return tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor


def cache_noninterference_snapshot(cache: Sequence[Sequence[Any]], event: Mapping[str, Any]) -> Dict[str, Any]:
    """Capture only the cache slices F2a must prove it did not mutate."""

    positions = {int(item) for item in event.get("cache_positions", [])}
    target_layers = {int(item) for item in event.get("target_layers", [])}
    snapshot: Dict[str, Any] = {"layers": {}}
    for layer_idx, state in enumerate(cache or []):
        if state is None:
            continue
        layer_payload: Dict[str, Any] = {}
        for tensor_idx, tensor in enumerate(state or ()):
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor_payload: Dict[str, Any] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            if tensor_idx in (0, 1) and layer_idx in target_layers:
                keep = [pos for pos in range(int(tensor.shape[2])) if pos not in positions]
                tensor_payload["untouched_positions"] = keep
                if keep:
                    indices = torch.tensor(keep, device=tensor.device, dtype=torch.long)
                    tensor_payload["untouched_slice"] = tensor.index_select(2, indices).detach().clone()
                else:
                    tensor_payload["untouched_slice"] = None
            elif tensor_idx >= 2:
                tensor_payload["full_tensor"] = tensor.detach().clone()
            elif layer_idx not in target_layers:
                tensor_payload["full_tensor"] = tensor.detach().clone()
            layer_payload[str(tensor_idx)] = tensor_payload
        snapshot["layers"][str(layer_idx)] = layer_payload
    return snapshot


def validate_cache_noninterference(
    before: Mapping[str, Any],
    after_cache: Sequence[Sequence[Any]],
    *,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    for layer_key, layer_payload in before.get("layers", {}).items():
        layer_idx = int(layer_key)
        if layer_idx >= len(after_cache or []):
            errors.append("f2a_cache_layer_missing_after:{}".format(layer_idx))
            continue
        state = after_cache[layer_idx] or ()
        for tensor_key, tensor_payload in layer_payload.items():
            tensor_idx = int(tensor_key)
            if tensor_idx >= len(state) or not isinstance(state[tensor_idx], torch.Tensor):
                errors.append("f2a_cache_tensor_missing_after:{}:{}".format(layer_idx, tensor_idx))
                continue
            tensor = state[tensor_idx]
            if list(tensor.shape) != tensor_payload.get("shape"):
                errors.append("f2a_cache_tensor_shape_changed:{}:{}".format(layer_idx, tensor_idx))
            if str(tensor.dtype) != tensor_payload.get("dtype"):
                errors.append("f2a_cache_tensor_dtype_changed:{}:{}".format(layer_idx, tensor_idx))
            if "full_tensor" in tensor_payload and not torch.equal(tensor, tensor_payload["full_tensor"].to(device=tensor.device)):
                errors.append("f2a_cache_tensor_mutated:{}:{}".format(layer_idx, tensor_idx))
            if "untouched_slice" in tensor_payload and tensor_payload["untouched_slice"] is not None:
                positions = tensor_payload.get("untouched_positions") or []
                indices = torch.tensor(positions, device=tensor.device, dtype=torch.long)
                current = tensor.index_select(2, indices)
                expected = tensor_payload["untouched_slice"].to(device=tensor.device, dtype=tensor.dtype)
                if not torch.equal(current, expected):
                    errors.append("f2a_cache_untouched_positions_mutated:{}:{}".format(layer_idx, tensor_idx))
    return {"status": "ok" if not errors else "failed", "errors": errors}


def patch_candidate_cache_for_full_event(
    reference_cache: Sequence[Sequence[Any]],
    *,
    event: Mapping[str, Any],
    restored_by_target_layer: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[Any, ...]]:
    """Clone a cache and replace all frozen missing K/V positions for one event."""

    positions = [int(item) for item in event.get("cache_positions", [])]
    restore_indices = [int(item) for item in event.get("restore_relative_indices", [])]
    pending = [int(item) for item in event.get("pending_token_positions", [])]
    if not positions or not restore_indices or len(positions) != len(restore_indices) or len(pending) != len(positions):
        raise F2AFrozenScheduleError("f2a_full_event_position_identity_mismatch")
    if len(set(positions)) != len(positions):
        raise F2AFrozenScheduleError("f2a_duplicate_cache_position")
    if len(set(restore_indices)) != len(restore_indices):
        raise F2AFrozenScheduleError("f2a_duplicate_restore_relative_index")
    if restore_indices != list(range(len(restore_indices))):
        raise F2AFrozenScheduleError("f2a_restore_relative_indices_noncanonical")
    if len(set(pending)) != len(pending):
        raise F2AFrozenScheduleError("f2a_duplicate_pending_token_position")
    if any(positions[idx] >= positions[idx + 1] for idx in range(len(positions) - 1)):
        raise F2AFrozenScheduleError("f2a_cache_positions_not_strictly_increasing")
    candidate = clone_cache(reference_cache)
    if caches_share_writable_storage(reference_cache, candidate):
        raise F2AFrozenScheduleError("f2a_candidate_cache_aliases_reference")
    target_layers = [int(layer) for layer in event.get("target_layers", [])]
    if not target_layers:
        raise F2AFrozenScheduleError("f2a_target_layers_missing")
    for target_layer in target_layers:
        if target_layer not in restored_by_target_layer:
            raise F2AFrozenScheduleError("f2a_restored_target_layer_missing:{}".format(target_layer))
        if target_layer >= len(candidate):
            raise F2AFrozenScheduleError("f2a_candidate_cache_layer_missing:{}".format(target_layer))
        key, value = restored_by_target_layer[target_layer]
        state = list(candidate[target_layer])
        if len(state) < 2 or not isinstance(state[0], torch.Tensor) or not isinstance(state[1], torch.Tensor):
            raise F2AFrozenScheduleError("f2a_reference_cache_self_attention_missing:{}".format(target_layer))
        expected_count = len(positions)
        if (
            key.shape[:2] != state[0].shape[:2]
            or key.shape[3:] != state[0].shape[3:]
            or int(key.shape[2]) != expected_count
        ):
            raise F2AFrozenScheduleError("f2a_restored_key_shape_mismatch:{}".format(target_layer))
        if (
            value.shape[:2] != state[1].shape[:2]
            or value.shape[3:] != state[1].shape[3:]
            or int(value.shape[2]) != expected_count
        ):
            raise F2AFrozenScheduleError("f2a_restored_value_shape_mismatch:{}".format(target_layer))
        if any(pos < 0 or pos >= int(state[0].shape[2]) or pos >= int(state[1].shape[2]) for pos in positions):
            raise F2AFrozenScheduleError("f2a_cache_position_out_of_range:{}".format(target_layer))
        if not torch.isfinite(key).all().item() or not torch.isfinite(value).all().item():
            raise F2AFrozenScheduleError("f2a_restored_kv_nonfinite:{}".format(target_layer))
        key = key.to(device=state[0].device, dtype=state[0].dtype)
        value = value.to(device=state[1].device, dtype=state[1].dtype)
        if positions == list(range(positions[0], positions[0] + len(positions))):
            start = positions[0]
            state[0][:, :, start : start + len(positions), :].copy_(key)
            state[1][:, :, start : start + len(positions), :].copy_(value)
        else:
            for source_block_idx, cache_position in enumerate(positions):
                state[0][:, :, cache_position : cache_position + 1, :].copy_(key[:, :, source_block_idx : source_block_idx + 1, :])
                state[1][:, :, cache_position : cache_position + 1, :].copy_(value[:, :, source_block_idx : source_block_idx + 1, :])
        candidate[target_layer] = tuple(state)
    if caches_share_writable_storage(reference_cache, candidate):
        raise F2AFrozenScheduleError("f2a_candidate_cache_aliases_reference_after_patch")
    return candidate


def patch_candidate_cache_for_event(
    reference_cache: Sequence[Sequence[Any]],
    *,
    event: Mapping[str, Any],
    restored_by_target_layer: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
    cache_position: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
    """Compatibility wrapper around the full-event patch helper."""

    positions = [int(item) for item in event.get("cache_positions", [])]
    if cache_position is None:
        return patch_candidate_cache_for_full_event(
            reference_cache,
            event=event,
            restored_by_target_layer=restored_by_target_layer,
        )
    event_copy = dict(event)
    event_copy["cache_positions"] = [int(cache_position)]
    event_copy["restore_relative_indices"] = [0]
    event_copy["pending_token_positions"] = [int(event.get("pending_token_positions", [cache_position])[0])]
    restored_single = {}
    for target_layer, tensors in restored_by_target_layer.items():
        key, value = tensors
        restored_single[int(target_layer)] = (key[:, :, :1, :], value[:, :, :1, :])
    return patch_candidate_cache_for_full_event(
        reference_cache,
        event=event_copy,
        restored_by_target_layer=restored_single,
    )


def build_f2a_event(
    *,
    stable_sample_id: str,
    generation_index: int,
    decoder_position: int,
    current_decoder_input_token_id: int,
    current_decoder_input_position: Optional[int] = None,
    prefix_token_ids: Sequence[int],
    source_layer_mode: str,
    source_layer: int,
    decoder_layer_count: int,
    pending_token_positions: Sequence[int],
    restore_relative_indices: Sequence[int],
    cache_positions: Sequence[int],
    reference_schedule_identity: str,
    reference_run_identity: str,
    artifact_file_sha256: Optional[str] = None,
    policy_sha256: Optional[str] = None,
    predicted_token_position: Optional[int] = None,
    candidate_policy_sha256: Optional[str] = None,
    policy_name: Optional[str] = None,
    candidate_layers: Optional[Sequence[int]] = None,
    source_confidence: Optional[float] = None,
    candidate_evaluations: Optional[Sequence[Mapping[str, Any]]] = None,
    event_timing: Optional[str] = None,
    restored_token_position: Optional[int] = None,
    followup_decoder_input_token_id: Optional[int] = None,
    followup_decoder_input_position: Optional[int] = None,
    followup_predicted_token_position: Optional[int] = None,
    followup_prefix_token_ids: Optional[Sequence[int]] = None,
    followup_reference_decision_layer: Optional[int] = None,
    followup_reference_decision_type: Optional[str] = None,
    threshold: float = CALM_THRESHOLD,
    threshold_comparator: str = CALM_THRESHOLD_COMPARATOR,
    confidence_compute_dtype: Optional[str] = None,
    adaptive_threshold: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source_layer = int(source_layer)
    current_position = int(current_decoder_input_position if current_decoder_input_position is not None else decoder_position)
    predicted_position = int(predicted_token_position if predicted_token_position is not None else current_position + 1)
    event = {
        "schema_version": F2A_SCHEDULE_SCHEMA_VERSION,
        "record_type": F2A_EVENT_RECORD_TYPE,
        "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
        "stable_sample_id": str(stable_sample_id),
        "generation_index": int(generation_index),
        "decoder_position": int(decoder_position),
        "event_timing": event_timing or (
            "deferred_followup" if source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING else "same_flush"
        ),
        "restored_token_position": None if restored_token_position is None else int(restored_token_position),
        "current_decoder_input_position": current_position,
        "predicted_token_position": predicted_position,
        "current_decoder_input_token_id": int(current_decoder_input_token_id),
        "reference_decoder_prefix_token_ids": [int(item) for item in prefix_token_ids],
        "prefix_token_ids": [int(item) for item in prefix_token_ids],
        "prefix_token_sha256": prefix_token_sha256(prefix_token_ids),
        "source_layer_mode": str(source_layer_mode),
        "source_layer": source_layer,
        "last_exact_kv_layer": source_layer - 1,
        "first_missing_target_layer": source_layer,
        "target_layers": list(range(source_layer, int(decoder_layer_count))),
        "pending_token_positions": [int(item) for item in pending_token_positions],
        "restore_relative_indices": [int(item) for item in restore_relative_indices],
        "cache_positions": [int(item) for item in cache_positions],
        "reference_schedule_identity": str(reference_schedule_identity),
        "reference_run_identity": str(reference_run_identity),
        "artifact_file_sha256": artifact_file_sha256,
        "policy_sha256": policy_sha256,
        "candidate_policy_sha256": candidate_policy_sha256,
        "policy_name": policy_name,
        "candidate_layers": [int(item) for item in candidate_layers] if candidate_layers is not None else None,
        "source_confidence": float(source_confidence) if source_confidence is not None else None,
        "candidate_evaluations": _json_safe(list(candidate_evaluations)) if candidate_evaluations is not None else None,
        "threshold": float(threshold),
        "threshold_comparator": str(threshold_comparator),
        "confidence_compute_dtype": confidence_compute_dtype,
        "adaptive_threshold": bool(adaptive_threshold),
        "followup_decoder_input_token_id": None if followup_decoder_input_token_id is None else int(followup_decoder_input_token_id),
        "followup_decoder_input_position": None if followup_decoder_input_position is None else int(followup_decoder_input_position),
        "followup_predicted_token_position": None if followup_predicted_token_position is None else int(followup_predicted_token_position),
        "followup_prefix_token_sha256": (
            prefix_token_sha256(followup_prefix_token_ids) if followup_prefix_token_ids is not None else None
        ),
        "followup_reference_decision_layer": (
            None if followup_reference_decision_layer is None else int(followup_reference_decision_layer)
        ),
        "followup_reference_decision_type": followup_reference_decision_type,
        "decoder_layer_count": int(decoder_layer_count),
    }
    if followup_prefix_token_ids is not None:
        event["followup_reference_decoder_prefix_token_ids"] = [int(item) for item in followup_prefix_token_ids]
    if candidate_evaluations is not None:
        event["first_crossing_decision_sha256"] = first_crossing_decision_sha256(
            candidate_evaluations=event["candidate_evaluations"],
            selected_source_layer=source_layer,
            threshold=float(threshold),
            threshold_comparator=str(threshold_comparator),
            candidate_policy_sha256=candidate_policy_sha256,
        )
    if extra:
        event.update(dict(extra))
    event["frozen_event_uid"] = make_f2a_event_uid(event)
    return event


def build_f2a_component_record(
    event: Mapping[str, Any],
    *,
    method: str,
    metrics: Mapping[str, Any],
    status: str = "ok",
    artifact_file_sha256: Optional[str] = None,
    policy_sha256: Optional[str] = None,
    replay_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if method not in F2A_REQUIRED_METHODS:
        raise F2AFrozenScheduleError("f2a_method_unsupported:{}".format(method))
    record = {
        "schema_version": F2A_RECORD_SCHEMA_VERSION,
        "record_type": F2A_COMPONENT_RECORD_TYPE,
        "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
        "frozen_event_uid": event.get("frozen_event_uid") or make_f2a_event_uid(event),
        "method": method,
        "status": str(status),
        "stable_sample_id": event.get("stable_sample_id"),
        "generation_index": event.get("generation_index"),
        "decoder_position": event.get("decoder_position"),
        "event_timing": event.get("event_timing"),
        "restored_token_position": event.get("restored_token_position"),
        "current_decoder_input_position": event.get("current_decoder_input_position"),
        "predicted_token_position": event.get("predicted_token_position"),
        "current_decoder_input_token_id": event.get("current_decoder_input_token_id"),
        "followup_decoder_input_token_id": event.get("followup_decoder_input_token_id"),
        "followup_decoder_input_position": event.get("followup_decoder_input_position"),
        "followup_predicted_token_position": event.get("followup_predicted_token_position"),
        "followup_prefix_token_sha256": event.get("followup_prefix_token_sha256"),
        "followup_reference_decision_layer": event.get("followup_reference_decision_layer"),
        "followup_reference_decision_type": event.get("followup_reference_decision_type"),
        "source_layer_mode": event.get("source_layer_mode"),
        "source_layer": event.get("source_layer"),
        "target_layer_scope": "all_missing_targets",
        "target_layers": list(event.get("target_layers") or []),
        "last_exact_kv_layer": event.get("last_exact_kv_layer"),
        "first_missing_target_layer": event.get("first_missing_target_layer"),
        "last_missing_target_layer": (list(event.get("target_layers") or [])[-1] if event.get("target_layers") else None),
        "target_layer_count": len(event.get("target_layers") or []),
        "pending_token_count": len(event.get("pending_token_positions") or []),
        "patched_cache_positions": list(event.get("cache_positions") or []),
        "cache_position_identity": canonical_json_sha256(event.get("cache_positions") or []),
        "prefix_sha256": event.get("prefix_token_sha256"),
        "prefix_identity": event.get("prefix_token_sha256"),
        "schedule_identity": event.get("reference_schedule_identity"),
        "frozen_schedule_sha256": event.get("reference_schedule_identity"),
        "reference_schedule_identity": event.get("reference_schedule_identity"),
        "reference_run_identity": event.get("reference_run_identity"),
        "artifact_file_sha256": artifact_file_sha256,
        "policy_sha256": policy_sha256,
        "candidate_policy_sha256": event.get("candidate_policy_sha256"),
        "evaluation_identity": event.get("reference_schedule_identity"),
    }
    for metric in F2A_LOGIT_METRICS:
        if metric in metrics:
            record[metric] = metrics[metric]
    if replay_diagnostics:
        record["replay_diagnostics"] = dict(replay_diagnostics)
    return record


def validate_f2a_record_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    schedule_rows: Iterable[Mapping[str, Any]],
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    schedule = list(schedule_rows)
    schedule_uids = {row.get("frozen_event_uid") or make_f2a_event_uid(row) for row in schedule}
    schedule_by_uid = {str(row.get("frozen_event_uid") or make_f2a_event_uid(row)): dict(row) for row in schedule}
    records_list = list(records)
    errors: List[str] = []
    method_counts: Counter = Counter()
    event_methods: Dict[str, set] = defaultdict(set)
    for index, record in enumerate(records_list):
        try:
            record_schema_version = int(record.get("schema_version", -1))
        except Exception:
            record_schema_version = -1
        if record_schema_version in (F2A_LEGACY_RECORD_SCHEMA_VERSION, F2A_LEGACY_RECORD_SCHEMA_V2):
            errors.append("f2a_legacy_component_schema_not_primary:row{}".format(index))
        elif record_schema_version != F2A_RECORD_SCHEMA_VERSION:
            errors.append("f2a_component_schema_version_mismatch:row{}".format(index))
        if record.get("record_type") != F2A_COMPONENT_RECORD_TYPE:
            errors.append("f2a_component_record_type_mismatch:row{}".format(index))
        method = record.get("method")
        if method not in F2A_REQUIRED_METHODS:
            errors.append("f2a_component_method_unsupported:row{}".format(index))
        else:
            method_counts[str(method)] += 1
        uid = record.get("frozen_event_uid")
        if uid not in schedule_uids:
            errors.append("f2a_component_event_uid_unknown:row{}".format(index))
        event = schedule_by_uid.get(str(uid))
        if event is not None:
            compare_fields = (
                "stable_sample_id",
                "generation_index",
                "decoder_position",
                "event_timing",
                "restored_token_position",
                "current_decoder_input_position",
                "predicted_token_position",
                "current_decoder_input_token_id",
                "followup_decoder_input_position",
                "followup_predicted_token_position",
                "followup_prefix_token_sha256",
                "followup_reference_decision_layer",
                "followup_reference_decision_type",
                "source_layer_mode",
                "source_layer",
                "target_layers",
                "artifact_file_sha256",
                "policy_sha256",
                "reference_schedule_identity",
                "reference_run_identity",
            )
            for field in compare_fields:
                if record.get(field) != event.get(field):
                    errors.append("f2a_component_event_field_mismatch:{}:row{}".format(field, index))
            if event.get("source_layer_mode") == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
                if record.get("candidate_policy_sha256") != event.get("candidate_policy_sha256"):
                    errors.append("f2a_component_event_field_mismatch:candidate_policy_sha256:row{}".format(index))
        if uid and method in event_methods.get(str(uid), set()):
            errors.append("f2a_duplicate_component_event_method:{}:{}".format(uid, method))
        if uid:
            event_methods[str(uid)].add(str(method))
        if record.get("target_layer_scope") != "all_missing_targets":
            errors.append("f2a_component_target_layer_scope_invalid:row{}".format(index))
        if record.get("target_layer") not in (None, ""):
            errors.append("f2a_component_legacy_target_layer_present:row{}".format(index))
        if record.get("status") == "ok":
            diagnostics = record.get("replay_diagnostics")
            if not isinstance(diagnostics, Mapping):
                errors.append("f2a_component_replay_diagnostics_missing:row{}".format(index))
            else:
                parity = diagnostics.get("exact_shadow_parity")
                if not isinstance(parity, Mapping) or parity.get("status") != "ok":
                    errors.append("f2a_component_exact_shadow_parity_failed:row{}".format(index))
                elif not isinstance(parity.get("metrics"), Mapping):
                    errors.append("f2a_component_exact_shadow_parity_metrics_missing:row{}".format(index))
                noninterference = diagnostics.get("reference_cache_noninterference")
                if not isinstance(noninterference, Mapping) or noninterference.get("status") != "ok":
                    errors.append("f2a_component_reference_cache_noninterference_failed:row{}".format(index))
                if diagnostics.get("candidate_output_used_for_reference_continuation") is not False:
                    errors.append("f2a_component_candidate_output_used_for_continuation:row{}".format(index))
                if diagnostics.get("candidate_cache_aliases_reference") is not False:
                    errors.append("f2a_component_candidate_cache_aliases_reference:row{}".format(index))
                if record.get("source_layer_mode") == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
                    if diagnostics.get("followup_query_frozen") is not True:
                        errors.append("f2a_component_followup_query_not_frozen:row{}".format(index))
                    if diagnostics.get("followup_reference_schedule_frozen") is not True:
                        errors.append("f2a_component_followup_schedule_not_frozen:row{}".format(index))
                    if diagnostics.get("candidate_schedule_recomputed") is not False:
                        errors.append("f2a_component_candidate_schedule_recomputed:row{}".format(index))
            for metric in F2A_LOGIT_METRICS:
                if metric not in record:
                    errors.append("f2a_component_metric_missing:{}:row{}".format(metric, index))
                    continue
                try:
                    if not math.isfinite(float(record[metric])):
                        errors.append("f2a_component_metric_nonfinite:{}:row{}".format(metric, index))
                except Exception:
                    errors.append("f2a_component_metric_malformed:{}:row{}".format(metric, index))
    for uid in schedule_uids:
        if event_methods.get(uid, set()) != set(F2A_REQUIRED_METHODS):
            errors.append("f2a_component_method_coverage_mismatch:{}".format(uid))
    if summary is not None:
        expected = summary.get("expected_record_count")
        written = summary.get("written_record_count", summary.get("row_count"))
        skipped = summary.get("skipped_record_count")
        complete = summary.get("complete_population_recording")
        if expected is None or int(expected) != len(records_list):
            errors.append("f2a_component_expected_record_count_mismatch")
        if written is None or int(written) != len(records_list):
            errors.append("f2a_component_written_record_count_mismatch")
        if skipped is None or int(skipped) != 0:
            errors.append("f2a_component_skipped_record_count_nonzero")
        if complete is not True:
            errors.append("f2a_component_incomplete_population_recording")
    return {
        "status": "ok" if not errors else "failed",
        "record_count": len(records_list),
        "method_counts": dict(sorted(method_counts.items())),
        "errors": errors,
    }


def validate_f2a_artifact_for_source_mode(
    artifact: Mapping[str, Any],
    *,
    source_layer_mode: str,
    decoder_layer_count: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        validate_phase3c_policy_artifact(artifact)
    except Exception as exc:
        errors.append("f2a_artifact_validation_failed:{}".format(exc))
        return {"status": "failed", "errors": errors}
    fit = artifact.get("fit_config") or {}
    model_spec = artifact.get("model_spec") or {}
    if fit.get("source_layer_mode") != source_layer_mode:
        errors.append("f2a_artifact_source_mode_mismatch")
    if int(model_spec.get("decoder_layer_count", -1)) != int(decoder_layer_count):
        errors.append("f2a_artifact_decoder_layer_count_mismatch")
    if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        if int(fit.get("fixed_source_layer", -1)) != 6:
            errors.append("f2a_fixed_artifact_source_layer_mismatch")
    elif source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
        try:
            semantics_valid = validate_candidate_first_crossing_semantics(fit)
            if not semantics_valid:
                errors.append("f2a_calm_artifact_policy_invalid")
        except Exception as exc:
            errors.append("f2a_calm_artifact_policy_invalid:{}".format(exc))
        coverage = phase3c_runtime_coverage_validation(artifact)
        if coverage.get("status") != "ok":
            errors.append("f2a_calm_artifact_runtime_coverage_invalid")
    else:
        errors.append("f2a_artifact_source_mode_unsupported")
    try:
        policy_sha = artifact_policy_sha256(artifact, require_authoritative=True)
    except Exception as exc:
        policy_sha = None
        errors.append("f2a_artifact_policy_sha_unavailable:{}".format(exc))
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "source_layer_mode": source_layer_mode,
        "policy_sha256": policy_sha,
        "decoder_layer_count": int(model_spec.get("decoder_layer_count", -1)),
    }


F2A_ARTIFACT_PREFLIGHT_SCHEMA_VERSION = 1
F2A_ARTIFACT_PREFLIGHT_FAILURE_STAGE = "artifact_preflight"


def f2a_artifact_preflight(
    artifact_path: Optional[str],
    *,
    requested_source_layer_mode: str,
    decoder_layer_count: Optional[int] = None,
    expected_fixed_source_layer: Optional[int] = None,
    expected_threshold: Optional[float] = None,
    expected_threshold_comparator: Optional[str] = None,
    expected_adaptive_threshold: Optional[bool] = None,
) -> Dict[str, Any]:
    """CPU-only, offline F2a artifact preflight shared by the outer runner, the
    fixed-layer migration tool, and tests.

    This is an earlier diagnostic gate only.  It reuses the same authoritative
    validation helpers (``validate_f2a_artifact_for_source_mode``,
    ``phase3c_runtime_coverage_validation``, ``artifact_policy_identity_validation``)
    that the model-construction hard gate in ``models/deploying_t5.py`` uses, so
    it cannot silently diverge from runtime behavior; it never loads a model,
    tokenizer, or dataset.

    The ``expected_*`` arguments are optional and apply only when
    ``requested_source_layer_mode`` is ``fixed_layer``. When supplied, they
    bind the artifact to the *actual* runtime policy the caller is about to
    execute (fixed source layer, threshold, threshold comparator, adaptive
    threshold), so an internally self-consistent authoritative fixed-layer
    artifact that simply encodes a *different* policy than what will run is
    still rejected here rather than only at model-construction time. Callers
    (the outer bash runner) should pass the same values used to build the
    generated runtime command so a future change to the runtime policy cannot
    leave this check silently comparing an unrelated hardcoded value.
    """

    errors: List[str] = []
    result: Dict[str, Any] = {
        "preflight_schema_version": F2A_ARTIFACT_PREFLIGHT_SCHEMA_VERSION,
        "status": "failed",
        "artifact_path": str(artifact_path) if artifact_path else None,
        "requested_source_layer_mode": requested_source_layer_mode,
        "errors": errors,
    }
    if not artifact_path:
        errors.append("f2a_artifact_preflight_artifact_path_missing")
        return result
    path = Path(artifact_path)
    if not path.is_file():
        errors.append("f2a_artifact_preflight_file_missing")
        return result
    result["artifact_file_sha256"] = sha256_file(path)
    try:
        artifact = load_phase3c_policy_artifact(path)
    except Exception as exc:
        errors.append("f2a_artifact_preflight_artifact_load_failed:{}".format(exc))
        return result

    fit_config = artifact.get("fit_config") or {}
    model_spec = artifact.get("model_spec") or {}
    artifact_decoder_layer_count = model_spec.get("decoder_layer_count")
    result.update(
        {
            "artifact_type": artifact.get("artifact_type"),
            "schema_version": artifact.get("schema_version"),
            "artifact_source_layer_mode": fit_config.get("source_layer_mode"),
            "fixed_source_layer": fit_config.get("fixed_source_layer"),
            "thresholds": [float(item) for item in (fit_config.get("thresholds") or [])],
            "decoder_layer_count": artifact_decoder_layer_count,
        }
    )
    if decoder_layer_count is not None and int(decoder_layer_count) != int(artifact_decoder_layer_count or -1):
        errors.append("f2a_artifact_preflight_decoder_layer_count_mismatch")
    effective_decoder_layer_count = (
        int(decoder_layer_count) if decoder_layer_count is not None else int(artifact_decoder_layer_count or 0)
    )

    source_mode_validation = validate_f2a_artifact_for_source_mode(
        artifact,
        source_layer_mode=requested_source_layer_mode,
        decoder_layer_count=effective_decoder_layer_count,
    )
    result["f2a_source_mode_validation"] = source_mode_validation
    if source_mode_validation.get("status") != "ok":
        errors.extend(source_mode_validation.get("errors") or [])

    coverage = phase3c_runtime_coverage_validation(artifact)
    result["runtime_coverage_validation"] = coverage
    if coverage.get("status") != "ok":
        errors.append("f2a_artifact_preflight_runtime_coverage_invalid")

    identity_validation = artifact_policy_identity_validation(artifact, require_authoritative=True)
    result["policy_identity_validation"] = identity_validation
    if identity_validation.get("status") != "ok":
        errors.extend(identity_validation.get("errors") or [])
    result["policy_sha256"] = identity_validation.get("policy_sha256")

    if requested_source_layer_mode == SOURCE_LAYER_MODE_FIXED:
        fixed_policy_errors = _fixed_runtime_policy_binding_errors(
            fit_config,
            artifact.get("parameters_by_threshold") or {},
            expected_fixed_source_layer=expected_fixed_source_layer,
            expected_threshold=expected_threshold,
            expected_threshold_comparator=expected_threshold_comparator,
            expected_adaptive_threshold=expected_adaptive_threshold,
        )
        result["fixed_runtime_policy_binding_errors"] = fixed_policy_errors
        errors.extend(fixed_policy_errors)

    result["errors"] = sorted(set(errors))
    result["status"] = "ok" if not result["errors"] else "failed"
    return result


def _fixed_runtime_policy_binding_errors(
    fit_config: Mapping[str, Any],
    parameters_by_threshold: Mapping[str, Any],
    *,
    expected_fixed_source_layer: Optional[int],
    expected_threshold: Optional[float],
    expected_threshold_comparator: Optional[str],
    expected_adaptive_threshold: Optional[bool],
) -> List[str]:
    """Bind a fixed-layer artifact to the actual runtime policy about to run.

    Uses ``threshold_to_key`` for canonical threshold comparison instead of
    comparing float/string representations by hand.
    """

    errors: List[str] = []
    if expected_fixed_source_layer is not None:
        try:
            actual_layer = int(fit_config.get("fixed_source_layer"))
        except (TypeError, ValueError):
            actual_layer = None
        if actual_layer != int(expected_fixed_source_layer):
            errors.append("f2a_artifact_preflight_fixed_source_layer_policy_mismatch")

    if expected_threshold is not None:
        expected_key = threshold_to_key(expected_threshold)
        try:
            actual_keys = {threshold_to_key(item) for item in (fit_config.get("thresholds") or [])}
        except Exception:
            actual_keys = set()
        if actual_keys != {expected_key}:
            errors.append("f2a_artifact_preflight_threshold_policy_mismatch")
        param_keys = set(parameters_by_threshold.keys())
        if param_keys != {expected_key}:
            errors.append("f2a_artifact_preflight_parameters_threshold_key_mismatch")

    semantics = fit_config.get("fixed_layer_policy_semantics")
    semantics = semantics if isinstance(semantics, MappingABC) else {}
    if expected_threshold_comparator is not None:
        if semantics.get("threshold_comparator") != expected_threshold_comparator:
            errors.append("f2a_artifact_preflight_threshold_comparator_policy_mismatch")
    if expected_adaptive_threshold is not None:
        if bool(semantics.get("adaptive_threshold")) != bool(expected_adaptive_threshold):
            errors.append("f2a_artifact_preflight_adaptive_threshold_policy_mismatch")

    return errors


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def paper_statistics_data_file_attestation(
    path: Path,
    *,
    schema: str,
    expected_record_count: int,
    written_record_count: int,
    skipped_record_count: int = 0,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path)
    display_path = path
    if relative_to is not None:
        try:
            display_path = path.resolve().relative_to(Path(relative_to).resolve())
        except Exception:
            display_path = path
    row_count = sum(1 for _line in path.open("r", encoding="utf-8"))
    complete = int(expected_record_count) == int(written_record_count) and int(skipped_record_count) == 0
    status = "ok" if complete and row_count == int(written_record_count) else "failed"
    return {
        "path": str(display_path).replace("\\", "/"),
        "sha256": sha256_file(path),
        "row_count": row_count,
        "schema": str(schema),
        "expected_record_count": int(expected_record_count),
        "written_record_count": int(written_record_count),
        "skipped_record_count": int(skipped_record_count),
        "complete_population_recording": bool(complete),
        "status": status,
    }


__all__ = [
    "F2A_COMPONENT_RECORD_TYPE",
    "F2A_EVALUATION_PROTOCOL_NAME",
    "F2A_EVENT_RECORD_TYPE",
    "F2A_EXACT_SHADOW_METHOD",
    "F2A_LEGACY_EVALUATION_PROTOCOL_NAME",
    "F2A_LEGACY_RECORD_SCHEMA_VERSION",
    "F2A_LEGACY_SCHEDULE_SCHEMA_VERSION",
    "F2A_LOGIT_METRICS",
    "F2A_NEGATIVE_ROUNDOFF_TOLERANCE",
    "F2A_METHOD_EXIT_CONDITIONED_HIDDEN_RESTORATION",
    "F2A_METHOD_EXIT_HIDDEN_TARGET_PROJECTION",
    "F2A_METHOD_FINAL_KV_RESTORATION",
    "F2A_RECORD_SCHEMA_VERSION",
    "F2A_PARITY_TOLERANCES",
    "F2A_REQUIRED_METHODS",
    "F2A_SCHEDULE_SCHEMA_VERSION",
    "F2A_STATISTICS_EVALUATION_MODE",
    "F2A_ARTIFACT_PREFLIGHT_SCHEMA_VERSION",
    "F2A_ARTIFACT_PREFLIGHT_FAILURE_STAGE",
    "F2AFrozenScheduleError",
    "assert_f2a_schedule_valid",
    "build_f2a_component_record",
    "build_f2a_event",
    "build_f2a_exact_shadow_parity_diagnostics",
    "cache_tensor_digest",
    "cache_noninterference_snapshot",
    "caches_share_writable_storage",
    "clone_cache",
    "f2a_artifact_preflight",
    "f2a_logit_metrics",
    "make_f2a_event_uid",
    "paper_statistics_data_file_attestation",
    "patch_candidate_cache_for_event",
    "patch_candidate_cache_for_full_event",
    "prefix_token_sha256",
    "restore_f2a_method_block_from_hidden",
    "restore_f2a_method_from_hidden",
    "schedule_semantic_sha256",
    "validate_f2a_artifact_for_source_mode",
    "validate_f2a_event",
    "validate_f2a_record_rows",
    "validate_f2a_schedule_rows",
    "validate_cache_noninterference",
    "validate_exact_shadow_parity",
    "validate_f2a_reference_position_bias",
    "write_csv",
    "write_json",
    "write_jsonl",
]
