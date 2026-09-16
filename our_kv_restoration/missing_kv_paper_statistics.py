"""Shared paper-facing statistics helpers for missing-K/V experiments.

This module is deliberately CPU-only and side-effect free except for the small
serialization helpers at the bottom.  It consumes normalized per-row records
from existing evaluators and computes paired stable-sample cluster bootstrap
summaries.  It does not run models, load tensors, or recompute task metrics.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from our_kv_restoration.missing_kv_dump_provenance import canonical_json_sha256, canonical_json_text, sha256_file
from our_kv_restoration.missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_THRESHOLD,
    CALM_THRESHOLD_COMPARATOR,
)


PAPER_STATISTICS_SCHEMA_VERSION = 1
PAPER_STATISTICS_EVALUATOR_NAME = "missing_kv_paper_statistics"
PAPER_STATISTICS_EVALUATOR_VERSION = 1
PAPER_STATISTICS_INPUT_BINDING_SCHEMA_VERSION = 1
PAPER_STATISTICS_INPUT_BINDING_TYPE = "missing_kv_paper_statistics_input_binding"
DEFAULT_BOOTSTRAP_SEED = 20260721
DEFAULT_BOOTSTRAP_REPLICATES = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_INTERVAL_METHOD = "percentile_paired_cluster_bootstrap"
DEFAULT_MIN_PAIRED_SAMPLES = 30
DEFAULT_MULTIPLICITY_ADJUSTMENT = "none"

EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY = "frozen_schedule_replay"
EVALUATION_MODES = {"offline_vector", "generation_quality", EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY}
EVALUATION_SEMANTICS = {"offline_matched_rows", "free_running", "frozen_schedule"}
SIGN_CONVENTIONS = {"lower_is_better", "higher_is_better"}
PROVENANCE_MODES = {"paper_strict", "synthetic_relaxed"}

PATH_IDENTITY_KEYS = {"path", "file_path", "resolved_path", "output_dir", "log_path"}
PROVENANCE_ALIAS_KEYS = {
    "repository_commit": ("repository_commit", "git_commit"),
}
COMMON_REQUIRED_IDENTITY_KEYS = (
    "repository_commit",
    "model_checkpoint_identity_sha256",
    "tokenizer_identity_sha256",
    "dataset_population_sha256",
    "generation_sample_binding_sha256",
    "decoder_layer_count",
)
OFFLINE_REQUIRED_IDENTITY_KEYS = ("dump_run_binding_sha256",)
GENERATION_REQUIRED_IDENTITY_KEYS = ("decoding_configuration_sha256",)
FROZEN_SCHEDULE_REPLAY_REQUIRED_IDENTITY_KEYS = (
    "reference_trajectory_sha256",
    "frozen_schedule_sha256",
    "frozen_event_population_sha256",
    "decoding_configuration_sha256",
)
ARTIFACT_REQUIRED_IDENTITY_KEYS = ("artifact_file_sha256",)
POLICY_REQUIRED_IDENTITY_KEYS = ("policy_sha256",)
CALM_REQUIRED_IDENTITY_KEYS = ("candidate_policy_sha256",)
COMPARABLE_IDENTITY_KEYS = tuple(
    dict.fromkeys(
        COMMON_REQUIRED_IDENTITY_KEYS
        + OFFLINE_REQUIRED_IDENTITY_KEYS
        + GENERATION_REQUIRED_IDENTITY_KEYS
        + FROZEN_SCHEDULE_REPLAY_REQUIRED_IDENTITY_KEYS
        + (
            "dataset_split",
            "effective_population_sha256",
        )
    )
)
METHOD_EXECUTION_IDENTITY_KEYS = (
    "source_layer_mode",
    "candidate_policy_sha256",
    "threshold",
    "threshold_comparator",
    "artifact_file_sha256",
    "policy_sha256",
    "runtime_method",
    "restoration_method",
)
SUPPORT_FIELDS = (
    "hidden_train_count",
    "hidden_train_used_count",
    "k_train_count",
    "k_train_used_count",
    "v_train_count",
    "v_train_used_count",
    "training_support_count",
    "training_used_count",
    "evaluation_support_count",
    "artifact_support_count",
)
METRIC_DIRECTION_REGISTRY = {
    "hidden_mse": "lower_is_better",
    "hidden_relative_l2": "lower_is_better",
    "k_mse": "lower_is_better",
    "k_relative_l2": "lower_is_better",
    "v_mse": "lower_is_better",
    "v_relative_l2": "lower_is_better",
    "kv_mse_sum": "lower_is_better",
    "hidden_cosine": "higher_is_better",
    "k_cosine": "higher_is_better",
    "v_cosine": "higher_is_better",
    "correct": "higher_is_better",
    "sequence_match_reference": "higher_is_better",
    "sequence_agreement": "higher_is_better",
    "rouge1": "higher_is_better",
    "rouge2": "higher_is_better",
    "rougeL": "higher_is_better",
    "rougeLsum": "higher_is_better",
    "logit_mse": "lower_is_better",
    "logit_relative_l2": "lower_is_better",
    "reference_to_candidate_kl": "lower_is_better",
    "candidate_to_reference_kl": "lower_is_better",
    "jensen_shannon_divergence": "lower_is_better",
    "logit_cosine": "higher_is_better",
    "top1_agreement": "higher_is_better",
    "reference_top1_probability": "higher_is_better",
    "candidate_reference_token_probability": "higher_is_better",
    "top1_probability_delta": "higher_is_better",
    "top1_margin_delta": "higher_is_better",
}

VECTOR_METRIC_ALIASES = {
    "hidden_mse": ("hidden_mse", "hidden_regen_mse", "hidden_regen_mse_mean"),
    "hidden_cosine": ("hidden_cosine", "hidden_regen_cosine", "hidden_regen_cosine_mean"),
    "hidden_relative_l2": ("hidden_relative_l2", "hidden_regen_relative_l2", "hidden_regen_relative_l2_mean"),
    "k_mse": ("k_mse", "k_regen_mse", "k_regen_mse_mean", "k_corrected_mse", "k_corrected_mse_mean"),
    "k_cosine": ("k_cosine", "k_regen_cosine", "k_regen_cosine_mean", "k_corrected_cosine", "k_corrected_cosine_mean"),
    "k_relative_l2": (
        "k_relative_l2",
        "k_regen_relative_l2",
        "k_regen_relative_l2_mean",
        "k_corrected_relative_l2",
        "k_corrected_relative_l2_mean",
    ),
    "v_mse": ("v_mse", "v_regen_mse", "v_regen_mse_mean", "v_corrected_mse", "v_corrected_mse_mean"),
    "v_cosine": ("v_cosine", "v_regen_cosine", "v_regen_cosine_mean", "v_corrected_cosine", "v_corrected_cosine_mean"),
    "v_relative_l2": (
        "v_relative_l2",
        "v_regen_relative_l2",
        "v_regen_relative_l2_mean",
        "v_corrected_relative_l2",
        "v_corrected_relative_l2_mean",
    ),
    "kv_mse_sum": ("kv_mse_sum", "kv_mse_sum_regen_mean", "kv_mse_sum_corrected_mean"),
    "logit_mse": ("logit_mse",),
    "logit_relative_l2": ("logit_relative_l2",),
    "logit_cosine": ("logit_cosine",),
    "reference_to_candidate_kl": ("reference_to_candidate_kl",),
    "candidate_to_reference_kl": ("candidate_to_reference_kl",),
    "jensen_shannon_divergence": ("jensen_shannon_divergence",),
    "top1_agreement": ("top1_agreement",),
    "reference_top1_probability": ("reference_top1_probability",),
    "candidate_reference_token_probability": ("candidate_reference_token_probability",),
    "top1_probability_delta": ("top1_probability_delta",),
    "top1_margin_delta": ("top1_margin_delta",),
}

CONFIDENCE_BINS = (
    ("confidence_(0.90,0.925)", 0.90, 0.925, False, False),
    ("confidence_[0.925,0.95)", 0.925, 0.95, True, False),
    ("confidence_[0.95,0.975)", 0.95, 0.975, True, False),
    ("confidence_[0.975,1.0]", 0.975, 1.0, True, True),
)


class PaperStatisticsError(ValueError):
    """Raised when an input cannot produce paper statistics."""


def minimum_equal_keys(evaluation_mode: str) -> List[str]:
    """Return the non-removable paper-strict equality keys for a comparison."""

    keys = list(COMMON_REQUIRED_IDENTITY_KEYS)
    if evaluation_mode == "offline_vector":
        keys.extend(OFFLINE_REQUIRED_IDENTITY_KEYS)
    elif evaluation_mode == "generation_quality":
        keys.extend(GENERATION_REQUIRED_IDENTITY_KEYS)
    elif evaluation_mode == EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY:
        keys.extend(FROZEN_SCHEDULE_REPLAY_REQUIRED_IDENTITY_KEYS)
    return list(dict.fromkeys(keys))


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    return value_f if math.isfinite(value_f) else None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_metric_value(value: Any) -> Tuple[str, Optional[float]]:
    if value is None:
        return "missing", None
    if isinstance(value, str) and value.strip() == "":
        return "missing", None
    try:
        parsed = float(value)
    except Exception:
        return "malformed", None
    if not math.isfinite(parsed):
        return "nonfinite", None
    return "finite", parsed


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise PaperStatisticsError("jsonl_row_not_mapping:{}:{}".format(path, line_no))
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_text(_json_safe(payload)))
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def jsonl_row_count(path: Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def paper_statistics_data_file_binding(
    path: Path,
    *,
    schema: str,
    row_count: Optional[int] = None,
    expected_record_count: Optional[int] = None,
    written_record_count: Optional[int] = None,
    skipped_record_count: Optional[int] = 0,
    complete_population_recording: Optional[bool] = None,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    data_path = Path(path)
    observed_row_count = jsonl_row_count(data_path) if row_count is None else int(row_count)
    if written_record_count is None:
        written_record_count = observed_row_count
    if expected_record_count is None:
        expected_record_count = written_record_count
    if complete_population_recording is None:
        complete_population_recording = int(skipped_record_count or 0) == 0 and int(written_record_count) == int(expected_record_count)
    binding_path = data_path
    if relative_to is not None:
        try:
            binding_path = data_path.resolve().relative_to(Path(relative_to).resolve())
        except Exception:
            binding_path = data_path
    return {
        "path": str(binding_path),
        "sha256": sha256_file(data_path),
        "row_count": observed_row_count,
        "schema": str(schema),
        "expected_record_count": int(expected_record_count),
        "written_record_count": int(written_record_count),
        "skipped_record_count": int(skipped_record_count or 0),
        "complete_population_recording": bool(complete_population_recording),
    }


def paper_statistics_input_eligibility(
    required_conditions: Mapping[str, Any],
    *,
    extra_failure_reasons: Optional[Iterable[str]] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a fail-closed paper-statistics input eligibility decision.

    ``required_conditions`` maps a failure-reason label to the boolean condition
    that must be true for a paper-strict input binding.  This keeps producer
    execution status separate from whether the emitted file is eligible for
    provenance-mode ``paper_strict`` consumption.
    """

    reasons: List[str] = []
    condition_results: Dict[str, bool] = {}
    for reason, value in sorted(required_conditions.items(), key=lambda item: str(item[0])):
        passed = bool(value is True)
        condition_results[str(reason)] = passed
        if not passed:
            reasons.append(str(reason))
    for reason in extra_failure_reasons or []:
        if reason not in (None, ""):
            reasons.append(str(reason))
    reasons = sorted(set(reasons))
    payload: Dict[str, Any] = {
        "status": "ok" if not reasons else "failed",
        "paper_input_eligible": not reasons,
        "paper_input_ineligibility_reasons": reasons,
        "required_condition_results": condition_results,
    }
    if diagnostics:
        payload.update({str(key): _json_safe(value) for key, value in diagnostics.items()})
    return payload


def build_paper_statistics_input_binding(
    *,
    producer_name: str,
    evaluation_mode: str,
    evaluation_semantics: str,
    data_files: Mapping[str, Mapping[str, Any]],
    common_population_identity: Mapping[str, Any],
    method_identities: Mapping[str, Mapping[str, Any]],
    status: str = "ok",
    eligibility: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(eligibility, Mapping):
        status = str(eligibility.get("status", status))
    payload = {
        "schema_version": PAPER_STATISTICS_INPUT_BINDING_SCHEMA_VERSION,
        "sidecar_type": PAPER_STATISTICS_INPUT_BINDING_TYPE,
        "status": str(status),
        "producer_name": str(producer_name),
        "evaluation_mode": str(evaluation_mode),
        "evaluation_semantics": str(evaluation_semantics),
        "data_files": {
            str(name): _json_safe(dict(payload))
            for name, payload in sorted(data_files.items(), key=lambda item: str(item[0]))
        },
        "common_population_identity": _canonical_identity(common_population_identity),
        "method_identities": {
            str(name): _canonical_identity(identity)
            for name, identity in sorted(method_identities.items(), key=lambda item: str(item[0]))
        },
    }
    if isinstance(eligibility, Mapping):
        for key, value in sorted(eligibility.items(), key=lambda item: str(item[0])):
            if key == "status":
                continue
            payload[str(key)] = _json_safe(value)
    return payload


def missing_required_population_identity_keys(
    identity: Mapping[str, Any],
    *,
    evaluation_mode: str,
) -> List[str]:
    """Return mandatory paper population identity keys absent from ``identity``."""

    return [key for key in minimum_equal_keys(evaluation_mode) if identity.get(key) in (None, "")]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return value


def stable_sample_identity(row: Mapping[str, Any], *, allow_sample_index: bool = False) -> str:
    value = row.get("stable_sample_id")
    if value not in (None, ""):
        return str(value)
    if allow_sample_index and row.get("sample_index") not in (None, ""):
        return "sample_index:{}".format(int(row["sample_index"]))
    raise PaperStatisticsError("stable_sample_id_missing")


def _row_position(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("decoder_position", "token_index", "generation_token_index", "position"):
        value = _to_int(row.get(key))
        if value is not None:
            return value
    return None


def offline_pair_identity(row: Mapping[str, Any], *, allow_sample_index: bool = False) -> Tuple[Any, ...]:
    stable = stable_sample_identity(row, allow_sample_index=allow_sample_index)
    frozen_event_uid = row.get("frozen_event_uid")
    if frozen_event_uid not in (None, ""):
        return (stable, "f2a_frozen_event", str(frozen_event_uid))
    generation = _to_int(row.get("generation_index"))
    position = _row_position(row)
    source = _to_int(row.get("source_layer"))
    target = _to_int(row.get("target_layer"))
    if generation is None or position is None or source is None or target is None:
        raise PaperStatisticsError("offline_identity_field_missing")
    threshold = row.get("threshold", row.get("threshold_key", row.get("policy_threshold", "none")))
    eval_identity = row.get("evaluation_identity", row.get("eval_group", row.get("eval_scope", "none")))
    return (stable, generation, position, source, target, str(threshold), str(eval_identity))


def generation_pair_identity(
    row: Mapping[str, Any],
    *,
    semantics: str,
    allow_sample_index: bool = False,
) -> Tuple[Any, ...]:
    stable = stable_sample_identity(row, allow_sample_index=allow_sample_index)
    if semantics == "free_running":
        return (stable,)
    if semantics != "frozen_schedule":
        raise PaperStatisticsError("invalid_generation_semantics:{}".format(semantics))
    generation = _to_int(row.get("generation_index"))
    position = _row_position(row)
    source = _to_int(row.get("source_layer", row.get("selected_source_layer")))
    target = _to_int(row.get("target_layer"))
    schedule = row.get("target_schedule", row.get("target_schedule_identity", row.get("schedule_identity")))
    prefix = row.get("prefix_identity", row.get("prefix_hash", row.get("prefix_sha256", "none")))
    policy = row.get("policy_identity", row.get("schedule_policy_identity", "none"))
    if position is None or source is None or target is None or schedule in (None, ""):
        raise PaperStatisticsError("frozen_schedule_identity_field_missing")
    return (stable, generation if generation is not None else "none", position, str(prefix), source, target, str(schedule), str(policy))


def _metric_value(row: Mapping[str, Any], canonical_metric: str) -> Optional[float]:
    state, value = _metric_parse(row, canonical_metric)
    return value if state == "finite" else None


def _metric_parse(row: Mapping[str, Any], canonical_metric: str) -> Tuple[str, Optional[float]]:
    for key in VECTOR_METRIC_ALIASES.get(canonical_metric, (canonical_metric,)):
        if key not in row:
            continue
        state, value = parse_metric_value(row.get(key))
        if state != "missing":
            return state, value
    if canonical_metric == "kv_mse_sum":
        k_state, k = _metric_parse(row, "k_mse")
        v_state, v = _metric_parse(row, "v_mse")
        if k_state == "finite" and v_state == "finite":
            return "finite", float(k) + float(v)
        if k_state in {"nonfinite", "malformed"}:
            return k_state, None
        if v_state in {"nonfinite", "malformed"}:
            return v_state, None
    return "missing", None


def _generation_metrics(row: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    correctness_value = None
    for key in ("is_correct", "correct", "task_correct"):
        if key in row:
            correctness_value = _coerce_bool(row.get(key))
            break
    if correctness_value is not None:
        metrics["correct"] = 1.0 if correctness_value else 0.0
    pred = row.get("prediction_token_sha256")
    ref = row.get("reference_token_sha256")
    if pred and ref:
        metrics["sequence_match_reference"] = 1.0 if pred == ref else 0.0
    for key in ("output_length", "prediction_length", "generated_length", "gen_len", "rouge1", "rouge2", "rougeL", "rougeLsum"):
        value = _to_float(row.get(key))
        if value is not None:
            metrics[key if key not in {"prediction_length", "generated_length"} else "output_length"] = value
    for key, value in row.items():
        if str(key).startswith("eval_rouge"):
            value_f = _to_float(value)
            if value_f is not None:
                metrics[str(key).replace("eval_", "")] = value_f
    return metrics


def _base_attrs(row: Mapping[str, Any]) -> Dict[str, Any]:
    source = _to_int(row.get("source_layer"))
    target = _to_int(row.get("target_layer"))
    gap = _to_int(row.get("layer_gap"))
    if gap is None and source is not None and target is not None:
        gap = target - source
    return {
        "source_layer": source,
        "target_layer": target,
        "layer_gap": gap,
        "gap_bin": row.get("gap_bin"),
        "source_confidence": _to_float(row.get("source_confidence", row.get("confidence"))),
        "frozen_event_uid": row.get("frozen_event_uid"),
        "prefix_sha256": row.get("prefix_sha256", row.get("prefix_token_sha256")),
        "frozen_schedule_sha256": row.get("frozen_schedule_sha256"),
    }


def _observation(
    *,
    method: str,
    row: Mapping[str, Any],
    metrics: Mapping[str, float],
    identity: Tuple[Any, ...],
    sample_id: str,
    source_schema: str,
    metric_states: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    states = {
        metric: _metric_parse(row, metric)[0]
        for metric in VECTOR_METRIC_ALIASES
    }
    if metric_states is not None:
        states.update({str(metric): str(state) for metric, state in metric_states.items()})
    return {
        "method": str(method),
        "identity": identity,
        "sample_id": str(sample_id),
        "metrics": dict(metrics),
        "metric_states": states,
        "attrs": _base_attrs(row),
        "row_status": str(row.get("status", "ok")),
        "support": {
            key: _to_float(row.get(key))
            for key in SUPPORT_FIELDS
            if row.get(key) not in (None, "")
        },
        "diagnostics": {
            "prediction_token_sha256": row.get("prediction_token_sha256"),
            "reference_token_sha256": row.get("reference_token_sha256"),
            "token_agreement_sha256": row.get("token_agreement_sha256"),
        },
        "source_schema": str(source_schema),
    }


def normalize_vector_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_schema: str,
    allow_sample_index: bool = False,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for row in rows:
        identity = offline_pair_identity(row, allow_sample_index=allow_sample_index)
        sample = stable_sample_identity(row, allow_sample_index=allow_sample_index)
        if row.get("method") is not None:
            metrics = {
                metric: value
                for metric in VECTOR_METRIC_ALIASES
                for value in [_metric_value(row, metric)]
                if value is not None
            }
            observations.append(
                _observation(
                    method=str(row["method"]),
                    row=row,
                    metrics=metrics,
                    identity=identity,
                    sample_id=sample,
                    source_schema=source_schema,
                )
            )
            continue
        variant = row.get("variant_name")
        if variant is not None:
            def _parse_variant(keys: Sequence[str]) -> Tuple[str, Optional[float]]:
                for key in keys:
                    if key in row:
                        state, value = parse_metric_value(row.get(key))
                        if state != "missing":
                            return state, value
                return "missing", None

            regen_states_values = {
                "hidden_mse": _parse_variant(("hidden_regen_mse", "hidden_regen_mse_mean")),
                "hidden_relative_l2": _parse_variant(("hidden_regen_relative_l2", "hidden_regen_relative_l2_mean")),
                "k_mse": _parse_variant(("k_regen_mse", "k_regen_mse_mean")),
                "k_relative_l2": _parse_variant(("k_regen_relative_l2", "k_regen_relative_l2_mean")),
                "v_mse": _parse_variant(("v_regen_mse", "v_regen_mse_mean")),
                "v_relative_l2": _parse_variant(("v_regen_relative_l2", "v_regen_relative_l2_mean")),
            }
            corrected_states_values = {
                "hidden_mse": regen_states_values["hidden_mse"],
                "hidden_relative_l2": regen_states_values["hidden_relative_l2"],
                "k_mse": _parse_variant(("k_corrected_mse", "k_corrected_mse_mean")),
                "k_relative_l2": _parse_variant(("k_corrected_relative_l2", "k_corrected_relative_l2_mean")),
                "v_mse": _parse_variant(("v_corrected_mse", "v_corrected_mse_mean")),
                "v_relative_l2": _parse_variant(("v_corrected_relative_l2", "v_corrected_relative_l2_mean")),
            }

            def _payload_and_states(states_values: Mapping[str, Tuple[str, Optional[float]]]) -> Tuple[Dict[str, float], Dict[str, str]]:
                payload = {
                    metric: float(value)
                    for metric, (state, value) in states_values.items()
                    if state == "finite" and value is not None
                }
                states = {metric: state for metric, (state, _value) in states_values.items()}
                k_state, k = states_values["k_mse"]
                v_state, v = states_values["v_mse"]
                if k_state == "finite" and v_state == "finite" and k is not None and v is not None:
                    payload["kv_mse_sum"] = float(k) + float(v)
                    states["kv_mse_sum"] = "finite"
                elif k_state in {"nonfinite", "malformed"}:
                    states["kv_mse_sum"] = k_state
                elif v_state in {"nonfinite", "malformed"}:
                    states["kv_mse_sum"] = v_state
                else:
                    states["kv_mse_sum"] = "missing"
                return payload, states

            regen, regen_states = _payload_and_states(regen_states_values)
            corrected, corrected_states = _payload_and_states(corrected_states_values)
            observations.append(
                _observation(
                    method="{}_regen".format(variant),
                    row=row,
                    metrics={k: v for k, v in regen.items() if v is not None},
                    identity=identity,
                    sample_id=sample,
                    source_schema=source_schema,
                    metric_states=regen_states,
                )
            )
            observations.append(
                _observation(
                    method="{}_corrected".format(variant),
                    row=row,
                    metrics={k: v for k, v in corrected.items() if v is not None},
                    identity=identity,
                    sample_id=sample,
                    source_schema=source_schema,
                    metric_states=corrected_states,
                )
            )
            continue
        raise PaperStatisticsError("vector_method_identity_missing")
    return observations


def normalize_generation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    method: str,
    semantics: str,
    allow_sample_index: bool = False,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for row in rows:
        identity = generation_pair_identity(row, semantics=semantics, allow_sample_index=allow_sample_index)
        sample = stable_sample_identity(row, allow_sample_index=allow_sample_index)
        metrics = _generation_metrics(row)
        if row.get("prediction_token_sha256"):
            metrics["prediction_digest_present"] = 1.0
        observations.append(
            _observation(
                method=method,
                row=row,
                metrics=metrics,
                identity=identity,
                sample_id=sample,
                source_schema="generation_quality",
            )
        )
    return observations


def _canonical_identity(spec_identity: Mapping[str, Any]) -> Dict[str, Any]:
    identity: Dict[str, Any] = {}
    for key, value in spec_identity.items():
        if key in PATH_IDENTITY_KEYS or value in (None, ""):
            continue
        identity[str(key)] = value
    for canonical, aliases in PROVENANCE_ALIAS_KEYS.items():
        for alias in aliases:
            if alias in identity:
                identity[canonical] = identity.pop(alias)
                break
    return dict(sorted(identity.items()))


def _data_file_path_from_spec(spec: Mapping[str, Any]) -> Optional[Path]:
    for key in ("data_file", "input_file", "path", "predictions", "records_path"):
        value = spec.get(key)
        if value not in (None, ""):
            return Path(value)
    return None


def _resolve_binding_path(path_text: Any, *, sidecar_path: Path) -> Optional[Path]:
    if path_text in (None, ""):
        return None
    path = Path(str(path_text))
    if not path.is_absolute():
        path = sidecar_path.parent / path
    return path


def _same_binding_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except Exception:
        return left.absolute() == right.absolute()


def _select_bound_data_file(
    binding: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    sidecar_path: Path,
    data_file_path: Optional[Path],
    label: str,
    errors: List[str],
) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    data_files = binding.get("data_files")
    if not isinstance(data_files, Mapping) or not data_files:
        errors.append("paper_statistics_data_files_missing:{}".format(label))
        return None, None
    requested_key = spec.get("data_file_key") or spec.get("binding_data_file_key")
    if requested_key not in (None, ""):
        key = str(requested_key)
        entry = data_files.get(key)
        if not isinstance(entry, Mapping):
            errors.append("paper_statistics_data_file_key_missing:{}:{}".format(label, key))
            return None, None
        return key, entry
    if len(data_files) > 1:
        errors.append("paper_statistics_data_file_key_required:{}".format(label))
        return None, None
    if data_file_path is not None:
        matches: List[Tuple[str, Mapping[str, Any]]] = []
        for key, entry in data_files.items():
            if not isinstance(entry, Mapping):
                continue
            declared = _resolve_binding_path(entry.get("path"), sidecar_path=sidecar_path)
            if declared is not None and _same_binding_path(declared, data_file_path):
                matches.append((str(key), entry))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            errors.append("paper_statistics_data_file_path_ambiguous:{}".format(label))
            return None, None
        if len(data_files) == 1:
            key, entry = next(iter(data_files.items()))
            if isinstance(entry, Mapping):
                return str(key), entry
    if len(data_files) == 1:
        key, entry = next(iter(data_files.items()))
        if isinstance(entry, Mapping):
            return str(key), entry
    errors.append("paper_statistics_data_file_unresolved:{}".format(label))
    return None, None


def _paper_statistics_binding_identity(
    binding: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    sidecar_path: Path,
    data_file_path: Optional[Path],
    provenance_mode: str,
    label: str,
    errors: List[str],
    warnings: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        schema_version = int(binding.get("schema_version", -1))
    except Exception:
        schema_version = -1
    if schema_version != PAPER_STATISTICS_INPUT_BINDING_SCHEMA_VERSION:
        errors.append("unsupported_paper_statistics_input_binding_schema:{}".format(label))
    if binding.get("sidecar_type") != PAPER_STATISTICS_INPUT_BINDING_TYPE:
        errors.append("paper_statistics_input_binding_type_mismatch:{}".format(label))
    if binding.get("status") != "ok":
        errors.append("paper_statistics_input_binding_status_not_ok:{}".format(label))
    key, data_entry = _select_bound_data_file(
        binding,
        spec=spec,
        sidecar_path=sidecar_path,
        data_file_path=data_file_path,
        label=label,
        errors=errors,
    )
    actual_data_sha = None
    declared_data_sha = None
    actual_data_row_count = None
    if data_entry is not None:
        declared_path = _resolve_binding_path(data_entry.get("path"), sidecar_path=sidecar_path)
        if data_file_path is None:
            data_file_path = declared_path
        elif declared_path is not None and not _same_binding_path(declared_path, data_file_path):
            errors.append("paper_statistics_data_file_path_mismatch:{}:{}".format(label, key))
        declared_data_sha = data_entry.get("sha256")
        if data_file_path is None:
            errors.append("paper_statistics_data_file_path_missing:{}".format(label))
        elif not data_file_path.exists():
            errors.append("input_data_file_missing:{}:{}".format(label, data_file_path))
        elif declared_data_sha in (None, ""):
            if provenance_mode == "paper_strict":
                errors.append("sidecar_data_file_sha256_missing:{}".format(label))
        else:
            actual_data_sha = sha256_file(data_file_path)
            if str(actual_data_sha) != str(declared_data_sha):
                errors.append("input_data_file_sha256_mismatch:{}".format(label))
            if data_file_path.suffix.lower() == ".jsonl":
                try:
                    actual_data_row_count = jsonl_row_count(data_file_path)
                except Exception as exc:
                    errors.append("paper_statistics_jsonl_row_count_failed:{}:{}".format(label, exc))
        declared_row_count = _to_int(data_entry.get("row_count"))
        expected_record_count = _to_int(data_entry.get("expected_record_count"))
        written_record_count = _to_int(data_entry.get("written_record_count"))
        skipped_record_count = _to_int(data_entry.get("skipped_record_count"))
        complete_population = _coerce_bool(data_entry.get("complete_population_recording"))
        completeness_errors = []
        completeness_warnings = []
        if actual_data_row_count is not None and declared_row_count is not None and actual_data_row_count != declared_row_count:
            completeness_errors.append("paper_statistics_declared_row_count_mismatch")
        if actual_data_row_count is not None and written_record_count is not None and actual_data_row_count != written_record_count:
            completeness_errors.append("paper_statistics_actual_written_record_count_mismatch")
        if declared_row_count is not None and written_record_count is not None and declared_row_count != written_record_count:
            completeness_errors.append("paper_statistics_row_count_written_record_count_mismatch")
        if provenance_mode == "paper_strict":
            if complete_population is not True:
                completeness_errors.append("paper_statistics_incomplete_population_recording")
            if skipped_record_count is None or int(skipped_record_count) != 0:
                completeness_errors.append("paper_statistics_skipped_record_count_nonzero")
            if expected_record_count is None:
                completeness_errors.append("paper_statistics_expected_record_count_missing")
            if written_record_count is None:
                completeness_errors.append("paper_statistics_written_record_count_missing")
            if expected_record_count is not None and written_record_count is not None and expected_record_count != written_record_count:
                completeness_errors.append("paper_statistics_expected_written_record_count_mismatch")
        else:
            if complete_population is False or (skipped_record_count is not None and int(skipped_record_count) != 0):
                completeness_warnings.append("paper_statistics_incomplete_population_recording")
        for item in completeness_errors:
            errors.append("{}:{}".format(item, label))
        for item in completeness_warnings:
            warnings.append("{}:{}".format(item, label))
    common = binding.get("common_population_identity")
    if not isinstance(common, Mapping):
        errors.append("paper_statistics_common_population_identity_missing:{}".format(label))
        common = {}
    method_name = spec.get("name") or spec.get("method_name")
    method_identity: Mapping[str, Any] = {}
    if method_name not in (None, ""):
        method_identities = binding.get("method_identities")
        if not isinstance(method_identities, Mapping):
            errors.append("paper_statistics_method_identities_missing:{}".format(label))
        else:
            candidate = method_identities.get(str(method_name))
            if not isinstance(candidate, Mapping):
                errors.append("paper_statistics_method_identity_missing:{}:{}".format(label, method_name))
            else:
                method_identity = candidate
    identity = _canonical_identity({**dict(common), **dict(method_identity)})
    diagnostics = {
        "label": label,
        "paper_statistics_input_binding": True,
        "binding_data_file_key": key,
        "input_data_file_path": str(data_file_path) if data_file_path is not None else None,
        "input_data_file_sha256": actual_data_sha,
        "sidecar_declared_data_file_sha256": declared_data_sha,
        "actual_data_file_row_count": actual_data_row_count,
        "declared_row_count": data_entry.get("row_count") if data_entry is not None else None,
        "expected_record_count": data_entry.get("expected_record_count") if data_entry is not None else None,
        "written_record_count": data_entry.get("written_record_count") if data_entry is not None else None,
        "skipped_record_count": data_entry.get("skipped_record_count") if data_entry is not None else None,
        "complete_population_recording": data_entry.get("complete_population_recording") if data_entry is not None else None,
        "bound_method_identities": {
            str(name): _canonical_identity({**dict(common), **dict(identity_payload)})
            for name, identity_payload in (binding.get("method_identities") or {}).items()
            if isinstance(identity_payload, Mapping)
        }
        if isinstance(binding.get("method_identities"), Mapping)
        else {},
    }
    return identity or None, diagnostics


def _validate_identity_sidecar_binding(
    spec: Mapping[str, Any],
    *,
    data_file_path: Optional[Path],
    provenance_mode: str,
    label: str,
) -> Tuple[List[str], List[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    warnings: List[str] = []
    sidecar_path_raw = spec.get("identity_sidecar") or spec.get("authoritative_sidecar") or spec.get("result_manifest")
    inline_identity = _canonical_identity(spec.get("identity") or {}) if isinstance(spec.get("identity"), Mapping) else {}
    if sidecar_path_raw in (None, ""):
        if provenance_mode == "paper_strict":
            errors.append("paper_strict_identity_sidecar_missing:{}".format(label))
        else:
            warnings.append("cryptographic_file_binding_not_validated:{}".format(label))
            return errors, warnings, inline_identity or None, None
        return errors, warnings, inline_identity or None, None
    sidecar_path = Path(sidecar_path_raw)
    if not sidecar_path.exists():
        errors.append("identity_sidecar_missing:{}:{}".format(label, sidecar_path))
        return errors, warnings, inline_identity or None, None
    actual_sidecar_sha = sha256_file(sidecar_path)
    expected_sidecar_sha = spec.get("identity_sidecar_sha256") or spec.get("authoritative_sidecar_sha256") or spec.get("result_manifest_sha256")
    if expected_sidecar_sha in (None, "") and provenance_mode == "paper_strict":
        errors.append("paper_strict_identity_sidecar_sha256_missing:{}".format(label))
    elif expected_sidecar_sha not in (None, "") and str(expected_sidecar_sha) != str(actual_sidecar_sha):
        errors.append("identity_sidecar_sha256_mismatch:{}".format(label))
    try:
        sidecar_payload = read_json(sidecar_path)
    except Exception as exc:
        errors.append("identity_sidecar_invalid:{}:{}".format(label, exc))
        return errors, warnings, inline_identity or None, None
    if not isinstance(sidecar_payload, Mapping):
        errors.append("identity_sidecar_not_mapping:{}".format(label))
        return errors, warnings, inline_identity or None, None
    binding = sidecar_payload.get("paper_statistics_input_binding")
    binding_diagnostics: Optional[Dict[str, Any]] = None
    if isinstance(binding, Mapping):
        sidecar_identity, binding_diagnostics = _paper_statistics_binding_identity(
            binding,
            spec=spec,
            sidecar_path=sidecar_path,
            data_file_path=data_file_path,
            provenance_mode=provenance_mode,
            label=label,
            errors=errors,
            warnings=warnings,
        )
        if inline_identity and sidecar_identity and inline_identity != sidecar_identity:
            errors.append("inline_identity_sidecar_mismatch:{}".format(label))
        diagnostics = {
            "label": label,
            "identity_sidecar_path": str(sidecar_path),
            "identity_sidecar_sha256": actual_sidecar_sha,
            **dict(binding_diagnostics or {}),
        }
        return errors, warnings, sidecar_identity or inline_identity or None, diagnostics
    sidecar_identity_raw = sidecar_payload.get("identity") or sidecar_payload.get("input_identity") or sidecar_payload.get("method_identity")
    if not isinstance(sidecar_identity_raw, Mapping):
        errors.append("identity_sidecar_identity_missing:{}".format(label))
        sidecar_identity: Dict[str, Any] = {}
    else:
        sidecar_identity = _canonical_identity(sidecar_identity_raw)
    if inline_identity and sidecar_identity and inline_identity != sidecar_identity:
        errors.append("inline_identity_sidecar_mismatch:{}".format(label))
    if data_file_path is None and sidecar_payload.get("data_file_path") not in (None, ""):
        data_file_path = Path(str(sidecar_payload["data_file_path"]))
    expected_data_sha = (
        sidecar_payload.get("data_file_sha256")
        or sidecar_payload.get("input_file_sha256")
        or sidecar_payload.get("records_sha256")
        or sidecar_payload.get("predictions_sha256")
    )
    if data_file_path is not None:
        if not data_file_path.exists():
            errors.append("input_data_file_missing:{}:{}".format(label, data_file_path))
        elif expected_data_sha in (None, ""):
            if provenance_mode == "paper_strict":
                errors.append("sidecar_data_file_sha256_missing:{}".format(label))
            else:
                warnings.append("data_file_sha256_not_declared:{}".format(label))
        else:
            actual_data_sha = sha256_file(data_file_path)
            if str(actual_data_sha) != str(expected_data_sha):
                errors.append("input_data_file_sha256_mismatch:{}".format(label))
    diagnostics = {
        "label": label,
        "input_data_file_path": str(data_file_path) if data_file_path is not None else None,
        "input_data_file_sha256": sha256_file(data_file_path) if data_file_path is not None and data_file_path.exists() else None,
        "identity_sidecar_path": str(sidecar_path),
        "identity_sidecar_sha256": actual_sidecar_sha,
        "sidecar_declared_data_file_sha256": expected_data_sha,
    }
    return errors, warnings, sidecar_identity or inline_identity or None, diagnostics


def _required_identity_keys(evaluation_mode: str, identity: Mapping[str, Any], spec: Mapping[str, Any]) -> List[str]:
    required = list(COMMON_REQUIRED_IDENTITY_KEYS)
    required.append("source_layer_mode")
    if evaluation_mode == "offline_vector":
        required.extend(OFFLINE_REQUIRED_IDENTITY_KEYS)
    if evaluation_mode == "generation_quality":
        required.extend(GENERATION_REQUIRED_IDENTITY_KEYS)
    if evaluation_mode == EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY:
        required.extend(FROZEN_SCHEDULE_REPLAY_REQUIRED_IDENTITY_KEYS)
    if bool(spec.get("uses_artifact", False)) or _coerce_bool(identity.get("uses_artifact")) is True:
        required.extend(ARTIFACT_REQUIRED_IDENTITY_KEYS)
        required.extend(POLICY_REQUIRED_IDENTITY_KEYS)
    if str(identity.get("source_layer_mode")) == "candidate_first_crossing":
        required.extend(CALM_REQUIRED_IDENTITY_KEYS)
        required.extend(("threshold", "threshold_comparator"))
    return list(dict.fromkeys(required))


def validate_method_identities(
    method_specs: Sequence[Mapping[str, Any]],
    *,
    observed_methods: Optional[Iterable[str]] = None,
    comparison_methods: Optional[Iterable[str]] = None,
    evaluation_mode: str = "offline_vector",
    provenance_mode: str = "paper_strict",
) -> Tuple[List[str], Dict[str, Any], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    identity_by_method: Dict[str, Dict[str, Any]] = {}
    seen_names: Counter = Counter()
    for spec in method_specs:
        name = str(spec.get("name", ""))
        if not name:
            errors.append("method_name_missing")
            continue
        seen_names[name] += 1
        if seen_names[name] > 1:
            errors.append("duplicate_method_name:{}".format(name))
        sidecar_errors, sidecar_warnings, sidecar_identity, _sidecar_diagnostics = _validate_identity_sidecar_binding(
            spec,
            data_file_path=_data_file_path_from_spec(spec),
            provenance_mode=provenance_mode,
            label="method:{}".format(name),
        )
        errors.extend(sidecar_errors)
        warnings.extend(sidecar_warnings)
        raw_identity = sidecar_identity if sidecar_identity is not None else spec.get("identity")
        if not isinstance(raw_identity, Mapping) or not raw_identity:
            errors.append("method_identity_missing:{}".format(name))
            raw_identity = {}
        identity = _canonical_identity(raw_identity)
        identity_by_method[name] = identity
        missing_required = [
            key
            for key in _required_identity_keys(evaluation_mode, identity, spec)
            if identity.get(key) in (None, "")
        ]
        if str(identity.get("source_layer_mode")) == "candidate_first_crossing":
            threshold = _to_float(identity.get("threshold"))
            comparator = identity.get("threshold_comparator")
            if threshold is not None and abs(float(threshold) - float(CALM_THRESHOLD)) > 1e-12:
                missing_required.append("threshold_frozen_value")
            if comparator not in (None, "") and str(comparator) != CALM_THRESHOLD_COMPARATOR:
                missing_required.append("threshold_comparator_frozen_value")
        if missing_required:
            items = ["missing_required_identity:{}:{}".format(name, key) for key in missing_required]
            if provenance_mode == "paper_strict":
                errors.extend(items)
            else:
                warnings.extend(items)
    observed = set(str(method) for method in (observed_methods or []))
    specified = set(identity_by_method)
    requested = set(str(method) for method in (comparison_methods or []))
    for method in sorted(requested - specified):
        errors.append("comparison_method_identity_missing:{}".format(method))
    for method in sorted(observed - specified):
        errors.append("observed_method_identity_missing:{}".format(method))
    for method in sorted(specified - observed):
        errors.append("identity_spec_without_observed_method:{}".format(method))
    if provenance_mode not in PROVENANCE_MODES:
        errors.append("invalid_provenance_mode:{}".format(provenance_mode))
    return errors, identity_by_method, warnings


def validate_comparison_identity_contract(
    comparison: Mapping[str, Any],
    identity_by_method: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_mode: str = "offline_vector",
    provenance_mode: str = "paper_strict",
) -> List[str]:
    errors: List[str] = []
    name = str(comparison.get("name") or "{}_vs_{}".format(comparison.get("method_a"), comparison.get("method_b")))
    method_a = str(comparison.get("method_a"))
    method_b = str(comparison.get("method_b"))
    contract = comparison.get("identity_contract")
    if not isinstance(contract, Mapping):
        return ["identity_contract_missing:{}".format(name)]
    required_equal = [str(key) for key in contract.get("required_equal_keys", [])]
    allowed_different = [str(key) for key in contract.get("allowed_different_keys", [])]
    if provenance_mode == "paper_strict":
        minimum = minimum_equal_keys(evaluation_mode)
        minimum_allowed = sorted(set(minimum) & set(allowed_different))
        for key in minimum_allowed:
            errors.append("identity_contract_minimum_key_allowed_different:{}:{}".format(name, key))
        required_equal = list(dict.fromkeys(required_equal + minimum))
    overlap = sorted(set(required_equal) & set(allowed_different))
    for key in overlap:
        errors.append("identity_contract_key_conflict:{}:{}".format(name, key))
    required_present = contract.get("required_present_by_method", {})
    if not isinstance(required_present, Mapping):
        errors.append("identity_contract_required_present_invalid:{}".format(name))
        required_present = {}
    for method in (method_a, method_b):
        if method not in identity_by_method:
            errors.append("identity_contract_unknown_method:{}:{}".format(name, method))
    for method in required_present:
        if str(method) not in {method_a, method_b}:
            errors.append("identity_contract_unknown_method:{}:{}".format(name, method))
    if method_a not in identity_by_method or method_b not in identity_by_method:
        return errors
    identity_a = identity_by_method[method_a]
    identity_b = identity_by_method[method_b]
    for key in required_equal:
        a_value = identity_a.get(key)
        b_value = identity_b.get(key)
        if a_value in (None, "") or b_value in (None, ""):
            if provenance_mode == "paper_strict":
                errors.append("identity_contract_required_equal_missing:{}:{}".format(name, key))
        elif a_value != b_value:
            errors.append("identity_contract_required_equal_mismatch:{}:{}".format(name, key))
    for method, keys in required_present.items():
        method_name = str(method)
        identity = identity_by_method.get(method_name, {})
        for key in keys or []:
            key_s = str(key)
            if identity.get(key_s) in (None, "") and provenance_mode == "paper_strict":
                errors.append("identity_contract_required_present_missing:{}:{}:{}".format(name, method_name, key_s))
    return errors


def _duplicate_keys(observations: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Tuple[Any, ...]]]:
    counts = Counter((str(obs["method"]), tuple(obs["identity"])) for obs in observations)
    return sorted([key for key, count in counts.items() if count > 1], key=lambda item: (item[0], repr(item[1])))


def _pairs_for_methods(
    observations: Sequence[Mapping[str, Any]],
    method_a: str,
    method_b: str,
    *,
    require_exact_identity_match: bool = False,
) -> Tuple[List[Tuple[Mapping[str, Any], Mapping[str, Any]]], Dict[str, Any], List[str]]:
    errors: List[str] = []
    duplicates = _duplicate_keys(observations)
    if duplicates:
        errors.append("duplicate_identity_count:{}".format(len(duplicates)))
    by_method: Dict[str, Dict[Tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
    for obs in observations:
        by_method[str(obs["method"])][tuple(obs["identity"])] = obs
    ids_a = set(by_method.get(method_a, {}))
    ids_b = set(by_method.get(method_b, {}))
    paired_ids = sorted(ids_a & ids_b, key=repr)
    samples_a = {str(obs["sample_id"]) for obs in by_method.get(method_a, {}).values()}
    samples_b = {str(obs["sample_id"]) for obs in by_method.get(method_b, {}).values()}
    paired_samples = {
        str(by_method[method_a][identity]["sample_id"])
        for identity in paired_ids
        if identity in by_method.get(method_a, {})
    }
    coverage = {
        "method_a_identity_count": len(ids_a),
        "method_b_identity_count": len(ids_b),
        "paired_identity_count": len(paired_ids),
        "unmatched_method_a_identity_count": len(ids_a - ids_b),
        "unmatched_method_b_identity_count": len(ids_b - ids_a),
        "method_a_sample_count": len(samples_a),
        "method_b_sample_count": len(samples_b),
        "paired_sample_count": len(paired_samples),
        "unmatched_method_a_sample_count": len(samples_a - samples_b),
        "unmatched_method_b_sample_count": len(samples_b - samples_a),
        "duplicate_identity_count": len(duplicates),
    }
    if require_exact_identity_match and (ids_a != ids_b or samples_a != samples_b):
        errors.append("frozen_or_paper_identity_population_mismatch")
    return [(by_method[method_a][identity], by_method[method_b][identity]) for identity in paired_ids], coverage, errors


def _pair_value(
    obs_a: Mapping[str, Any],
    obs_b: Mapping[str, Any],
    metric: str,
    sign_convention: str,
) -> Optional[Dict[str, Any]]:
    a_state, a = parse_metric_value((obs_a.get("metrics") or {}).get(metric))
    b_state, b = parse_metric_value((obs_b.get("metrics") or {}).get(metric))
    if a_state != "finite" or b_state != "finite":
        return None
    improvement = a - b if sign_convention == "lower_is_better" else b - a
    attrs_a = dict(obs_a.get("attrs") or {})
    attrs_b = dict(obs_b.get("attrs") or {})
    attrs = dict(attrs_a)
    for key, value in attrs_b.items():
        attrs.setdefault(key, value)
    return {
        "sample_id": str(obs_a["sample_id"]),
        "identity": tuple(obs_a["identity"]),
        "method_a_value": a,
        "method_b_value": b,
        "paired_improvement": improvement,
        "attrs": attrs,
        "attrs_a": attrs_a,
        "attrs_b": attrs_b,
        "diagnostics_a": dict(obs_a.get("diagnostics") or {}),
        "diagnostics_b": dict(obs_b.get("diagnostics") or {}),
        "support_a": dict(obs_a.get("support") or {}),
        "support_b": dict(obs_b.get("support") or {}),
    }


def aggregate_pairs_by_sample(pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["sample_id"])].append(pair)
    sample_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "row_count": len(rows),
                "method_a_mean": sum(float(row["method_a_value"]) for row in rows) / len(rows),
                "method_b_mean": sum(float(row["method_b_value"]) for row in rows) / len(rows),
                "paired_improvement_mean": sum(float(row["paired_improvement"]) for row in rows) / len(rows),
            }
        )
    return sample_rows


def paired_cluster_bootstrap(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    if not sample_rows:
        return {"status": "invalid", "errors": ["empty_sample_rows"]}
    improvements = [float(row["paired_improvement_mean"]) for row in sample_rows]
    point = sum(improvements) / len(improvements)
    if int(replicates) <= 0:
        return {
            "status": "disabled",
            "paired_difference": point,
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
        }
    rng = random.Random(int(seed))
    boot: List[float] = []
    for _idx in range(int(replicates)):
        sampled = [improvements[rng.randrange(len(improvements))] for _ in improvements]
        boot.append(sum(sampled) / len(sampled))
    boot.sort()
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower_idx = max(0, min(len(boot) - 1, int(math.floor(alpha * (len(boot) - 1)))))
    upper_idx = max(0, min(len(boot) - 1, int(math.ceil((1.0 - alpha) * (len(boot) - 1)))))
    return {
        "status": "ok",
        "paired_difference": point,
        "ci_lower": boot[lower_idx],
        "ci_upper": boot[upper_idx],
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def _confidence_bin(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    for label, start, end, include_start, include_end in CONFIDENCE_BINS:
        lower_ok = confidence >= start if include_start else confidence > start
        upper_ok = confidence <= end if include_end else confidence < end
        if lower_ok and upper_ok:
            return label
    return None


def strata_for_attrs(attrs: Mapping[str, Any]) -> List[str]:
    strata = ["all"]
    source = _to_int(attrs.get("source_layer"))
    if source in CALM_CANDIDATE_LAYERS:
        strata.append("source_{}".format(source))
    gap = _to_int(attrs.get("layer_gap"))
    gap_bin = attrs.get("gap_bin")
    if gap == 0 or gap_bin == "same_layer":
        strata.extend(["same_layer", "all_gaps"])
    elif gap is not None:
        strata.append("all_gaps")
        if gap_bin:
            strata.append("gap_{}".format(gap_bin))
        if 1 <= gap <= 4:
            strata.append("short_gap")
        elif 5 <= gap <= 12:
            strata.append("medium_gap")
        elif gap >= 13:
            strata.append("long_gap")
    confidence = _to_float(attrs.get("source_confidence"))
    conf_label = _confidence_bin(confidence)
    if confidence is not None:
        strata.append("all_confidence")
    if conf_label is not None:
        strata.append(conf_label)
    if source in CALM_CANDIDATE_LAYERS and gap_bin:
        strata.append("source_{}__gap_{}".format(source, gap_bin))
    if source in CALM_CANDIDATE_LAYERS and conf_label is not None:
        strata.append("source_{}__{}".format(source, conf_label))
    if gap_bin and conf_label is not None:
        strata.append("gap_{}__{}".format(gap_bin, conf_label))
    return strata


def strata_for_pair(pair: Mapping[str, Any], *, anchor: Optional[str] = None) -> List[str]:
    if anchor == "method_a":
        attrs = pair.get("attrs_a") or {}
    elif anchor == "method_b":
        attrs = pair.get("attrs_b") or {}
    else:
        attrs = pair.get("attrs") or {}
    return strata_for_attrs(attrs)


def _comparison_result(
    *,
    name: str,
    method_a: str,
    method_b: str,
    metric: str,
    sign_convention: str,
    pairs: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    support_threshold: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    stratum: str = "all",
    free_running_stratum_anchor: Optional[str] = None,
    generation_outcome_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    sample_rows = aggregate_pairs_by_sample(pairs)
    bootstrap = paired_cluster_bootstrap(sample_rows, replicates=bootstrap_replicates, seed=bootstrap_seed)
    paired_sample_count = len(sample_rows)
    support_status = "ok" if paired_sample_count >= int(support_threshold) else "low_support"
    method_a_point = None
    method_b_point = None
    relative = None
    if sample_rows:
        method_a_point = sum(float(row["method_a_mean"]) for row in sample_rows) / len(sample_rows)
        method_b_point = sum(float(row["method_b_mean"]) for row in sample_rows) / len(sample_rows)
        denom = abs(method_a_point)
        if denom > 1e-12:
            relative = float(bootstrap["paired_difference"]) / denom
    return {
        "comparison_name": name,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "stratum": stratum,
        "direction_of_improvement": "positive_means_method_b_better",
        "sign_convention": sign_convention,
        "sample_aggregation_rule": "mean_within_stable_sample_before_bootstrap",
        "paired_sample_count": paired_sample_count,
        "eligible_pair_count": len(pairs),
        "method_a_point_estimate": method_a_point,
        "method_b_point_estimate": method_b_point,
        "paired_difference": bootstrap.get("paired_difference"),
        "relative_difference": relative,
        "ci_lower": bootstrap.get("ci_lower"),
        "ci_upper": bootstrap.get("ci_upper"),
        "bootstrap_replicates": bootstrap.get("bootstrap_replicates"),
        "bootstrap_seed": bootstrap.get("bootstrap_seed"),
        "support_status": support_status,
        "support_diagnostics": _support_summary(pairs),
        "generation_outcome_diagnostics": dict(generation_outcome_diagnostics or {}),
        "free_running_stratum_anchor": free_running_stratum_anchor,
        "claim_validity": False,
        "coverage": dict(coverage),
    }


def _attrs_for_pair(pair: Mapping[str, Any], *, anchor: Optional[str] = None) -> Mapping[str, Any]:
    if anchor == "method_a":
        return pair.get("attrs_a") or {}
    if anchor == "method_b":
        return pair.get("attrs_b") or {}
    return pair.get("attrs") or {}


def _coverage_distribution(pairs: Sequence[Mapping[str, Any]], *, anchor: Optional[str] = None) -> Dict[str, Any]:
    source_counts = Counter()
    gap_counts = Counter()
    conf_counts = Counter()
    for pair in pairs:
        attrs = _attrs_for_pair(pair, anchor=anchor)
        source = attrs.get("source_layer")
        if source is not None:
            source_counts[str(source)] += 1
        gap = attrs.get("gap_bin")
        if gap is not None:
            gap_counts[str(gap)] += 1
        conf = _confidence_bin(_to_float(attrs.get("source_confidence")))
        if conf is not None:
            conf_counts[conf] += 1
    return {
        "source_layer_distribution": dict(sorted(source_counts.items())),
        "gap_distribution": dict(sorted(gap_counts.items())),
        "confidence_bin_distribution": dict(sorted(conf_counts.items())),
    }


def _support_summary(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values_by_key: Dict[str, List[float]] = defaultdict(list)
    missing = Counter()
    for pair in pairs:
        for side in ("support_a", "support_b"):
            support = pair.get(side) or {}
            for key in SUPPORT_FIELDS:
                if key in support:
                    parsed = _to_float(support.get(key))
                    if parsed is not None:
                        values_by_key[key].append(parsed)
                    else:
                        missing[key] += 1
                else:
                    missing[key] += 1
    summary: Dict[str, Any] = {}
    for key in SUPPORT_FIELDS:
        values = sorted(values_by_key.get(key, []))
        if values:
            mid = len(values) // 2
            median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
            summary[key] = {
                "min": values[0],
                "median": median,
                "max": values[-1],
                "missing_count": int(missing.get(key, 0)),
            }
        elif missing.get(key, 0):
            summary[key] = {"min": None, "median": None, "max": None, "missing_count": int(missing[key])}
    return summary


def _generation_outcome_summary(pairs: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    payload = {
        "correctness_status": "not_available",
        "paired_samples_with_explicit_correctness": 0,
        "paired_samples_with_sequence_digests": 0,
        "both_correct_count": 0,
        "both_wrong_count": 0,
        "method_a_only_correct_count": 0,
        "method_b_only_correct_count": 0,
        "regression_count": 0,
        "rescue_count": 0,
        "regression_rate": None,
        "rescue_rate": None,
        "rate_denominator": "paired_samples_with_correctness",
        "net_correctness_change": None,
        "sequence_agreement_count": 0,
        "sequence_disagreement_count": 0,
        "sequence_agreement_rate": None,
        "token_agreement_status": "not_available",
    }
    errors: List[str] = []
    by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_sample[str(pair["sample_id"])].append(pair)
    correctness_pairs: List[Tuple[int, int]] = []
    sequence_pairs: List[bool] = []
    for sample_id in sorted(by_sample):
        sample_pairs = by_sample[sample_id]
        a_values = {int(pair["method_a_value"]) for pair in sample_pairs if pair.get("method_a_value") in (0.0, 1.0)}
        b_values = {int(pair["method_b_value"]) for pair in sample_pairs if pair.get("method_b_value") in (0.0, 1.0)}
        if len(a_values) > 1 or len(b_values) > 1:
            errors.append("sample_correctness_conflict:{}".format(sample_id))
        elif len(a_values) == 1 and len(b_values) == 1:
            correctness_pairs.append((next(iter(a_values)), next(iter(b_values))))
        pred_a_values = {
            str((pair.get("diagnostics_a") or {}).get("prediction_token_sha256"))
            for pair in sample_pairs
            if (pair.get("diagnostics_a") or {}).get("prediction_token_sha256") not in (None, "")
        }
        pred_b_values = {
            str((pair.get("diagnostics_b") or {}).get("prediction_token_sha256"))
            for pair in sample_pairs
            if (pair.get("diagnostics_b") or {}).get("prediction_token_sha256") not in (None, "")
        }
        if len(pred_a_values) > 1 or len(pred_b_values) > 1:
            errors.append("sample_prediction_digest_conflict:{}".format(sample_id))
        elif len(pred_a_values) == 1 and len(pred_b_values) == 1:
            sequence_pairs.append(next(iter(pred_a_values)) == next(iter(pred_b_values)))
    if correctness_pairs:
        payload["correctness_status"] = "available"
        denom = len(correctness_pairs)
        both_correct = sum(1 for a, b in correctness_pairs if a == 1 and b == 1)
        both_wrong = sum(1 for a, b in correctness_pairs if a == 0 and b == 0)
        a_only = sum(1 for a, b in correctness_pairs if a == 1 and b == 0)
        b_only = sum(1 for a, b in correctness_pairs if a == 0 and b == 1)
        payload.update(
            {
                "both_correct_count": both_correct,
                "both_wrong_count": both_wrong,
                "method_a_only_correct_count": a_only,
                "method_b_only_correct_count": b_only,
                "regression_count": a_only,
                "rescue_count": b_only,
                "paired_samples_with_explicit_correctness": denom,
                "regression_rate": a_only / denom,
                "rescue_rate": b_only / denom,
                "net_correctness_change": b_only - a_only,
            }
        )
    if sequence_pairs:
        agreement = sum(1 for item in sequence_pairs if item)
        payload.update(
            {
                "sequence_agreement_count": agreement,
                "sequence_disagreement_count": len(sequence_pairs) - agreement,
                "sequence_agreement_rate": agreement / len(sequence_pairs),
                "paired_samples_with_sequence_digests": len(sequence_pairs),
            }
        )
    return payload, errors


def evaluate_comparison(
    observations: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    *,
    identity_by_method: Optional[Mapping[str, Mapping[str, Any]]] = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    support_threshold: int = DEFAULT_MIN_PAIRED_SAMPLES,
    evaluation_mode: str = "offline_vector",
    evaluation_semantics: str = "offline_matched_rows",
    provenance_mode: str = "paper_strict",
    free_running_stratum_anchor: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    method_a = str(comparison.get("method_a"))
    method_b = str(comparison.get("method_b"))
    metric = str(comparison.get("metric"))
    requested_sign = comparison.get("sign_convention", comparison.get("metric_direction"))
    registry_sign = METRIC_DIRECTION_REGISTRY.get(metric)
    sign = str(requested_sign or registry_sign or "")
    name = str(comparison.get("name") or "{}_vs_{}_{}".format(method_a, method_b, metric))
    errors: List[str] = []
    if not sign:
        errors.append("metric_direction_missing:{}".format(metric))
        sign = "lower_is_better"
    elif sign not in SIGN_CONVENTIONS:
        errors.append("invalid_sign_convention:{}".format(sign))
    elif registry_sign is not None and requested_sign is not None and str(requested_sign) != registry_sign:
        errors.append("metric_direction_conflict:{}:{}:{}".format(metric, requested_sign, registry_sign))
    if identity_by_method is not None:
        errors.extend(
            validate_comparison_identity_contract(
                comparison,
                identity_by_method,
                evaluation_mode=evaluation_mode,
                provenance_mode=provenance_mode,
            )
        )
    require_exact = provenance_mode == "paper_strict"
    raw_pairs, coverage, pair_errors = _pairs_for_methods(
        observations,
        method_a,
        method_b,
        require_exact_identity_match=require_exact,
    )
    errors.extend(pair_errors)
    usable_pairs: List[Dict[str, Any]] = []
    missing_metric_count = 0
    malformed_metric_count = 0
    nonfinite_metric_count = 0
    failed_row_count_by_method = Counter()
    for obs_a, obs_b in raw_pairs:
        failed = False
        for obs in (obs_a, obs_b):
            if obs.get("row_status") not in (None, "ok"):
                failed_row_count_by_method[str(obs.get("method"))] += 1
                failed = True
        if failed:
            continue
        pair_invalid = False
        for obs in (obs_a, obs_b):
            state = (obs.get("metric_states") or {}).get(metric)
            if state is None:
                state, _value = parse_metric_value((obs.get("metrics") or {}).get(metric))
            if state == "missing":
                missing_metric_count += 1
                pair_invalid = True
            elif state == "malformed":
                malformed_metric_count += 1
                pair_invalid = True
            elif state == "nonfinite":
                nonfinite_metric_count += 1
                pair_invalid = True
        if pair_invalid:
            continue
        pair = _pair_value(obs_a, obs_b, metric, sign)
        if pair is None:
            continue
        if not _is_number(pair["paired_improvement"]):
            nonfinite_metric_count += 1
            continue
        usable_pairs.append(pair)
    anchor_for_distribution = None
    if evaluation_semantics == "free_running" and free_running_stratum_anchor in {"method_a", "method_b"}:
        anchor_for_distribution = str(free_running_stratum_anchor)
    coverage = {
        **dict(coverage),
        **_coverage_distribution(usable_pairs, anchor=anchor_for_distribution),
        "missing_metric_count": missing_metric_count,
        "malformed_metric_count": malformed_metric_count,
        "nonfinite_metric_count": nonfinite_metric_count,
        "failed_row_count_by_method": dict(sorted(failed_row_count_by_method.items())),
        "failed_expected_method_row_count": sum(failed_row_count_by_method.values()),
        "eligible_token_target_layer_count": len(usable_pairs),
        "eligible_pair_count": len(usable_pairs),
        "paired_sample_count": len({pair["sample_id"] for pair in usable_pairs}),
    }
    if failed_row_count_by_method:
        errors.append("failed_expected_method_row_count:{}".format(sum(failed_row_count_by_method.values())))
    if malformed_metric_count:
        errors.append("malformed_metric_count:{}".format(malformed_metric_count))
    if nonfinite_metric_count:
        errors.append("nonfinite_metric_count:{}".format(nonfinite_metric_count))
    if provenance_mode == "paper_strict" and missing_metric_count:
        errors.append("missing_metric_count:{}".format(missing_metric_count))
    if not usable_pairs:
        errors.append("empty_comparison")
    generation_outcome: Dict[str, Any] = {}
    if metric == "correct":
        generation_outcome, outcome_errors = _generation_outcome_summary(usable_pairs)
        errors.extend(outcome_errors)
    result = _comparison_result(
        name=name,
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        sign_convention=sign,
        pairs=usable_pairs,
        coverage=coverage,
        support_threshold=support_threshold,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        free_running_stratum_anchor=(free_running_stratum_anchor or "disabled") if evaluation_semantics == "free_running" else None,
        generation_outcome_diagnostics=generation_outcome,
    )
    stratum_pairs: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    if evaluation_semantics == "free_running" and free_running_stratum_anchor not in {"method_a", "method_b"}:
        for pair in usable_pairs:
            stratum_pairs["all"].append(pair)
    else:
        for pair in usable_pairs:
            for stratum in strata_for_pair(pair, anchor=anchor_for_distribution):
                stratum_pairs[stratum].append(pair)
    stratified = [
        _comparison_result(
            name=name,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            sign_convention=sign,
            pairs=stratum_pairs[stratum],
            coverage={"stratum_pair_count": len(stratum_pairs[stratum])},
            support_threshold=support_threshold,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            stratum=stratum,
            free_running_stratum_anchor=(free_running_stratum_anchor or "disabled") if evaluation_semantics == "free_running" else None,
            generation_outcome_diagnostics=generation_outcome if metric == "correct" and stratum == "all" else {},
        )
        for stratum in sorted(stratum_pairs)
    ]
    return result, stratified, errors


def _load_vector_observations(manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    observations: List[Dict[str, Any]] = []
    diagnostics = {
        "raw_input_row_count": 0,
        "normalized_observation_count": 0,
        "method_status_counts": {},
        "failed_row_count_by_method": {},
        "skipped_row_count_by_status": {},
        "input_bindings": [],
        "method_identity_specs": [],
    }
    status_counts: Counter = Counter()
    failed_by_method: Counter = Counter()
    allow_sample_index = bool(manifest.get("allow_sample_index_identity", False))
    provenance_mode = str(manifest.get("provenance_mode", "paper_strict"))
    is_f2a_replay = str(manifest.get("evaluation_mode")) == EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY
    for record_spec in manifest.get("records", []):
        path = Path(record_spec["path"])
        if path.suffix.lower() != ".jsonl":
            errors.append("aggregate_or_non_jsonl_input_rejected:{}".format(path))
            continue
        sidecar_errors, sidecar_warnings, _identity, binding = _validate_identity_sidecar_binding(
            record_spec,
            data_file_path=path,
            provenance_mode=provenance_mode,
            label="records:{}".format(path.name),
        )
        errors.extend(sidecar_errors)
        warnings.extend(sidecar_warnings)
        if binding is not None:
            diagnostics["input_bindings"].append(binding)
        try:
            rows = read_jsonl(path)
            diagnostics["raw_input_row_count"] += len(rows)
            if is_f2a_replay:
                for row_index, row in enumerate(rows):
                    if int(row.get("schema_version", -1)) != 3:
                        errors.append("f2a_primary_schema_version_invalid:{}:row{}".format(path, row_index))
                    if row.get("target_layer_scope") != "all_missing_targets":
                        errors.append("f2a_primary_target_layer_scope_invalid:{}:row{}".format(path, row_index))
                    if row.get("target_layer") not in (None, ""):
                        errors.append("f2a_tokenwise_marginal_record_rejected:{}:row{}".format(path, row_index))
                    if row.get("pending_token_count") in (None, "") or int(row.get("pending_token_count")) < 1:
                        errors.append("f2a_primary_pending_token_count_invalid:{}:row{}".format(path, row_index))
            normalized = normalize_vector_rows(
                rows,
                source_schema=str(record_spec.get("schema", "auto")),
                allow_sample_index=allow_sample_index,
            )
            for row in rows:
                method = str(row.get("method", row.get("variant_name", "unknown")))
                status = str(row.get("status", "ok"))
                status_counts["{}:{}".format(method, status)] += 1
                if status not in ("ok", "None"):
                    failed_by_method[method] += 1
            observations.extend(normalized)
            bound_methods = (binding or {}).get("bound_method_identities") if isinstance(binding, Mapping) else None
            if isinstance(bound_methods, Mapping):
                observed_here = sorted({str(obs.get("method")) for obs in normalized})
                for method_name in observed_here:
                    identity = bound_methods.get(method_name)
                    if not isinstance(identity, Mapping):
                        continue
                    diagnostics["method_identity_specs"].append(
                        {
                            "name": method_name,
                            "identity": dict(identity),
                            "result_manifest": (binding or {}).get("identity_sidecar_path"),
                            "result_manifest_sha256": (binding or {}).get("identity_sidecar_sha256"),
                            "path": str(path),
                        }
                    )
        except Exception as exc:
            errors.append("vector_input_invalid:{}:{}".format(path, exc))
    diagnostics["normalized_observation_count"] = len(observations)
    diagnostics["method_status_counts"] = dict(sorted(status_counts.items()))
    diagnostics["failed_row_count_by_method"] = dict(sorted(failed_by_method.items()))
    return observations, errors, warnings, diagnostics


def _load_generation_observations(manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Any], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    observations: List[Dict[str, Any]] = []
    descriptive: Dict[str, Any] = {}
    diagnostics = {"raw_input_row_count": 0, "normalized_observation_count": 0, "input_bindings": []}
    semantics = str(manifest.get("evaluation_semantics"))
    allow_sample_index = bool(manifest.get("allow_sample_index_identity", False))
    provenance_mode = str(manifest.get("provenance_mode", "paper_strict"))
    for method in manifest.get("methods", []):
        name = str(method.get("name", ""))
        if not name:
            errors.append("method_name_missing")
            continue
        predictions = method.get("predictions")
        if predictions:
            sidecar_errors, sidecar_warnings, _identity, binding = _validate_identity_sidecar_binding(
                method,
                data_file_path=Path(predictions),
                provenance_mode=provenance_mode,
                label="predictions:{}:{}".format(name, Path(predictions).name),
            )
            errors.extend(sidecar_errors)
            warnings.extend(sidecar_warnings)
            if binding is not None:
                diagnostics["input_bindings"].append(binding)
            try:
                rows = read_jsonl(Path(predictions))
                diagnostics["raw_input_row_count"] += len(rows)
                observations.extend(
                    normalize_generation_rows(
                        rows,
                        method=name,
                        semantics=semantics,
                        allow_sample_index=allow_sample_index,
                    )
                )
            except Exception as exc:
                errors.append("generation_input_invalid:{}:{}".format(predictions, exc))
        metrics_path = method.get("metrics")
        if metrics_path:
            try:
                payload = read_json(Path(metrics_path))
                descriptive[name] = {
                    key: value
                    for key, value in payload.items()
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                }
            except Exception as exc:
                errors.append("generation_metrics_invalid:{}:{}".format(metrics_path, exc))
    diagnostics["normalized_observation_count"] = len(observations)
    return observations, errors, warnings, descriptive, diagnostics


def validate_input_manifest(manifest: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(manifest.get("schema_version", -1)) != PAPER_STATISTICS_SCHEMA_VERSION:
        errors.append("unsupported_input_manifest_schema")
    if manifest.get("evaluation_mode") not in EVALUATION_MODES:
        errors.append("invalid_evaluation_mode")
    if manifest.get("evaluation_semantics") not in EVALUATION_SEMANTICS:
        errors.append("invalid_evaluation_semantics")
    if not isinstance(manifest.get("comparisons"), list) or not manifest.get("comparisons"):
        errors.append("comparisons_missing")
    if manifest.get("evaluation_mode") == "offline_vector" and manifest.get("evaluation_semantics") != "offline_matched_rows":
        errors.append("offline_vector_requires_offline_matched_rows")
    if (
        manifest.get("evaluation_mode") == EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY
        and manifest.get("evaluation_semantics") != "frozen_schedule"
    ):
        errors.append("frozen_schedule_replay_requires_frozen_schedule")
    if manifest.get("provenance_mode") not in PROVENANCE_MODES:
        errors.append("invalid_or_missing_provenance_mode")
    return errors


def evaluate_manifest(
    manifest: Mapping[str, Any],
    *,
    bootstrap_replicates: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    bootstrap_replicates = DEFAULT_BOOTSTRAP_REPLICATES if bootstrap_replicates is None else int(bootstrap_replicates)
    bootstrap_seed = DEFAULT_BOOTSTRAP_SEED if bootstrap_seed is None else int(bootstrap_seed)
    errors = validate_input_manifest(manifest)
    warnings: List[str] = []
    observations: List[Dict[str, Any]] = []
    descriptive: Dict[str, Any] = {}
    load_diagnostics: Dict[str, Any] = {}
    provenance_mode = str(manifest.get("provenance_mode", "paper_strict"))
    if provenance_mode not in PROVENANCE_MODES:
        errors.append("invalid_provenance_mode:{}".format(provenance_mode))
    method_specs = manifest.get("methods") or []
    if manifest.get("evaluation_mode") in {"offline_vector", EVALUATION_MODE_FROZEN_SCHEDULE_REPLAY}:
        loaded, load_errors, load_warnings, load_diagnostics = _load_vector_observations(manifest)
        observations.extend(loaded)
        errors.extend(load_errors)
        warnings.extend(load_warnings)
        method_specs = (
            manifest.get("method_identities")
            or load_diagnostics.get("method_identity_specs")
            or manifest.get("methods")
            or []
        )
    elif manifest.get("evaluation_mode") == "generation_quality":
        loaded, load_errors, load_warnings, descriptive, load_diagnostics = _load_generation_observations(manifest)
        observations.extend(loaded)
        errors.extend(load_errors)
        warnings.extend(load_warnings)
    observed_methods = sorted({str(obs.get("method")) for obs in observations})
    comparison_methods = []
    for comparison in manifest.get("comparisons", []):
        comparison_methods.extend([comparison.get("method_a"), comparison.get("method_b")])
    provenance_errors, identity_by_method, provenance_warnings = validate_method_identities(
        method_specs,
        observed_methods=observed_methods,
        comparison_methods=[item for item in comparison_methods if item is not None],
        evaluation_mode=str(manifest.get("evaluation_mode")),
        provenance_mode=provenance_mode,
    )
    errors.extend(provenance_errors)
    warnings.extend(provenance_warnings)
    if provenance_mode == "synthetic_relaxed":
        warnings.append("synthetic_relaxed_provenance_mode_claim_invalid")
    comparisons: List[Dict[str, Any]] = []
    stratified_results: List[Dict[str, Any]] = []
    for comparison in manifest.get("comparisons", []):
        result, strata, comparison_errors = evaluate_comparison(
            observations,
            comparison,
            identity_by_method=identity_by_method,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            support_threshold=int(manifest.get("min_paired_samples", DEFAULT_MIN_PAIRED_SAMPLES)),
            evaluation_mode=str(manifest.get("evaluation_mode")),
            evaluation_semantics=str(manifest.get("evaluation_semantics")),
            provenance_mode=provenance_mode,
            free_running_stratum_anchor=manifest.get("free_running_stratum_anchor"),
        )
        if comparison_errors:
            result["status"] = "invalid"
            result["errors"] = comparison_errors
        else:
            result["status"] = "ok" if result["paired_sample_count"] else "invalid"
            result["errors"] = []
        comparisons.append(result)
        stratified_results.extend(strata)
        errors.extend("comparison:{}:{}".format(result["comparison_name"], error) for error in comparison_errors)
    duplicate_count = len(_duplicate_keys(observations))
    if duplicate_count:
        errors.append("duplicate_identity_count:{}".format(duplicate_count))
    sample_count = len({str(obs.get("sample_id")) for obs in observations})
    result = {
        "schema_version": PAPER_STATISTICS_SCHEMA_VERSION,
        "evaluator_name": PAPER_STATISTICS_EVALUATOR_NAME,
        "evaluator_version": PAPER_STATISTICS_EVALUATOR_VERSION,
        "status": "invalid" if errors else "ok",
        "paper_claim_valid": False,
        "real_population_evaluated": False if provenance_mode == "synthetic_relaxed" else bool(manifest.get("real_population_evaluated", False)),
        "evaluation_semantics": manifest.get("evaluation_semantics"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "protocol": {
            "primary_inference_unit": "stable_sample_id_dialogue_cluster",
            "comparison_type": "paired",
            "bootstrap_unit": "stable_sample_id",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "interval_method": DEFAULT_INTERVAL_METHOD,
            "interval_sidedness": "two_sided",
            "multiplicity_adjustment": DEFAULT_MULTIPLICITY_ADJUSTMENT,
            "minimum_paired_samples": int(manifest.get("min_paired_samples", DEFAULT_MIN_PAIRED_SAMPLES)),
            "positive_improvement_convention": "method_b_better",
            "provenance_mode": provenance_mode,
            "free_running_stratum_policy": (
                "disabled"
                if manifest.get("evaluation_semantics") == "free_running" and not manifest.get("free_running_stratum_anchor")
                else manifest.get("free_running_stratum_anchor")
            ),
        },
        "input_identity": {
            "manifest_sha256": canonical_json_sha256(_json_safe(manifest)),
            "method_identities": identity_by_method,
            "descriptive_generation_metrics": descriptive,
        },
        "population_validation": {
            "status": "failed" if errors else "ok",
            "observation_count": len(observations),
            "unique_sample_count": sample_count,
            "duplicate_identity_count": duplicate_count,
            **load_diagnostics,
            "errors": list(errors),
        },
        "comparisons": comparisons,
        "stratified_results": stratified_results,
        "coverage": {
            "raw_input_row_count": load_diagnostics.get("raw_input_row_count", len(observations)),
            "normalized_observation_count": len(observations),
            "unique_sample_count": sample_count,
            "method_counts": dict(sorted(Counter(str(obs.get("method")) for obs in observations).items())),
            **{key: value for key, value in load_diagnostics.items() if key not in {"raw_input_row_count", "normalized_observation_count"}},
        },
        "warnings": warnings,
        "errors": list(errors),
    }
    return result


def validate_result_schema(result: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version",
        "evaluator_name",
        "evaluator_version",
        "status",
        "paper_claim_valid",
        "real_population_evaluated",
        "evaluation_semantics",
        "protocol",
        "input_identity",
        "population_validation",
        "comparisons",
        "stratified_results",
        "coverage",
        "warnings",
        "errors",
    }
    missing = sorted(required - set(result))
    errors.extend("missing_top_level_field:{}".format(field) for field in missing)
    if result.get("schema_version") != PAPER_STATISTICS_SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    if result.get("evaluation_semantics") not in EVALUATION_SEMANTICS:
        errors.append("invalid_evaluation_semantics")
    for comparison in result.get("comparisons", []):
        if comparison.get("sign_convention") not in SIGN_CONVENTIONS:
            errors.append("invalid_sign_convention")
    return errors


COMPARISON_CSV_FIELDS = [
    "comparison_name",
    "method_a",
    "method_b",
    "metric",
    "stratum",
    "paired_sample_count",
    "eligible_pair_count",
    "method_a_point_estimate",
    "method_b_point_estimate",
    "paired_difference",
    "relative_difference",
    "ci_lower",
    "ci_upper",
    "bootstrap_replicates",
    "bootstrap_seed",
    "support_status",
    "claim_validity",
    "status",
]


def write_result_outputs(
    result: Mapping[str, Any],
    *,
    output_json: Path,
    output_comparison_csv: Path,
    output_stratified_csv: Path,
    output_markdown: Path,
) -> None:
    write_json(output_json, result)
    write_csv(output_comparison_csv, result.get("comparisons", []), COMPARISON_CSV_FIELDS)
    write_csv(output_stratified_csv, result.get("stratified_results", []), COMPARISON_CSV_FIELDS)
    Path(output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(output_markdown).write_text(render_markdown_summary(result), encoding="utf-8", newline="\n")


def render_markdown_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# Missing-KV Paper Statistics Summary",
        "",
        "- status: `{}`".format(result.get("status")),
        "- paper_claim_valid: `{}`".format(str(result.get("paper_claim_valid")).lower()),
        "- real_population_evaluated: `{}`".format(str(result.get("real_population_evaluated")).lower()),
        "- evaluation_semantics: `{}`".format(result.get("evaluation_semantics")),
        "- bootstrap: `{}` replicates, seed `{}`".format(
            result.get("protocol", {}).get("bootstrap_replicates"),
            result.get("protocol", {}).get("bootstrap_seed"),
        ),
        "",
        "Synthetic/CPU validation does not establish a paper claim.",
        "",
        "## Comparisons",
        "",
    ]
    for comparison in result.get("comparisons", []):
        lines.extend(
            [
                "### {}".format(comparison.get("comparison_name")),
                "",
                "- metric: `{}`".format(comparison.get("metric")),
                "- paired samples: `{}`".format(comparison.get("paired_sample_count")),
                "- support: `{}`".format(comparison.get("support_status")),
                "- point estimate A: `{}`".format(comparison.get("method_a_point_estimate")),
                "- point estimate B: `{}`".format(comparison.get("method_b_point_estimate")),
                "- paired difference: `{}`".format(comparison.get("paired_difference")),
                "- 95% CI: `[{}, {}]`".format(comparison.get("ci_lower"), comparison.get("ci_upper")),
                "- status: `{}`".format(comparison.get("status", "ok")),
                "",
            ]
        )
    if result.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend("- `{}`".format(error) for error in result["errors"])
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_INTERVAL_METHOD",
    "DEFAULT_MIN_PAIRED_SAMPLES",
    "PAPER_STATISTICS_INPUT_BINDING_SCHEMA_VERSION",
    "PAPER_STATISTICS_INPUT_BINDING_TYPE",
    "PAPER_STATISTICS_SCHEMA_VERSION",
    "PaperStatisticsError",
    "aggregate_pairs_by_sample",
    "build_paper_statistics_input_binding",
    "evaluate_comparison",
    "evaluate_manifest",
    "generation_pair_identity",
    "jsonl_row_count",
    "minimum_equal_keys",
    "missing_required_population_identity_keys",
    "normalize_generation_rows",
    "normalize_vector_rows",
    "offline_pair_identity",
    "paper_statistics_data_file_binding",
    "paper_statistics_input_eligibility",
    "paired_cluster_bootstrap",
    "render_markdown_summary",
    "stable_sample_identity",
    "strata_for_pair",
    "validate_input_manifest",
    "validate_method_identities",
    "validate_result_schema",
    "write_result_outputs",
]
