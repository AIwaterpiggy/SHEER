"""CPU-only helpers for Phase 3c F2b free-running generation comparisons."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .missing_kv_dump_provenance import (
    canonical_json_sha256,
    model_checkpoint_identity,
    sha256_file,
    validate_effective_population_sidecar,
    validate_generation_binding_sidecar,
    validate_tokenizer_identity_payload,
)
from .missing_kv_paper_statistics import (
    build_paper_statistics_input_binding,
    paper_statistics_data_file_binding,
)
from .missing_kv_review_bundle import create_review_bundle_archive
from . import missing_kv_paper_population as _paper_population


F2B_SCHEMA_VERSION = 1
F2B_POPULATION_IDENTITY_SCHEMA_VERSION = 1
F2B_ARM_IDENTITY_SCHEMA_VERSION = 1
F2B_SPLIT_EVIDENCE_SCHEMA_VERSION = 1
F2B_RUNTIME_PROVENANCE_SCHEMA_VERSION = 1
F2B_METHODS = (
    "full_reference",
    "direct_shallow_kv_reuse",
    "calm_exit_hidden_projection",
    "phase3c_kv_final",
)
F2B_RUNTIME_METHODS = F2B_METHODS + ("exact_catchup",)
APPROXIMATION_METHODS = (
    "direct_shallow_kv_reuse",
    "calm_exit_hidden_projection",
    "phase3c_kv_final",
)
F2B_COMPARISONS = (
    ("full_vs_direct", "full_reference", "direct_shallow_kv_reuse"),
    ("full_vs_calm_style", "full_reference", "calm_exit_hidden_projection"),
    ("full_vs_phase3c", "full_reference", "phase3c_kv_final"),
    ("direct_vs_phase3c", "direct_shallow_kv_reuse", "phase3c_kv_final"),
    ("calm_style_vs_phase3c", "calm_exit_hidden_projection", "phase3c_kv_final"),
)
LARGE_ARTIFACT_SUFFIXES = {".pt", ".bin", ".safetensors", ".ckpt", ".pth", ".npy", ".npz"}
ACCEPTED_PHASE3C_ARTIFACT_SHA256 = _paper_population.ACCEPTED_PHASE3C_ARTIFACT_SHA256
ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256 = _paper_population.ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256
ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256 = _paper_population.ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256
F2B_CANDIDATE_POLICY_NAME = "candidate_restricted_first_crossing_v1"
F2B_CANDIDATE_LAYERS = (4, 6, 8, 10)
F2B_CANDIDATE_THRESHOLD = 0.9
F2B_CANDIDATE_THRESHOLD_COMPARATOR = "strict_gt"
F2B_CANDIDATE_CONFIDENCE_DTYPE = "float32"
F2B_CANDIDATE_ADAPTIVE_THRESHOLD = False
F2B_CANDIDATE_FALLBACK_POLICY = "full_depth"
F2B_EXPECTED_DATASET_POPULATION_COUNT = _paper_population.EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT
F2B_EXPECTED_ARTIFACT_FITTING_COUNT = _paper_population.EXPECTED_ARTIFACT_FITTING_COUNT
F2B_EXPECTED_HELDOUT_COUNT = _paper_population.EXPECTED_ARTIFACT_HELDOUT_COUNT

_PLACEHOLDER_STRINGS = {"", "unknown", "placeholder", "todo", "tbd", "none", "null", "<unset>", "unset"}
_SPECIAL_TOKEN_FIELDS = (
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
)
_COMMON_ARM_IDENTITY_FIELDS = (
    "repository_commit",
    "checkpoint_inventory_schema_version",
    "checkpoint_inventory_file_count",
    "checkpoint_inventory_total_bytes",
    "checkpoint_inventory_sha256",
    "checkpoint_config_sha256",
    "tokenizer_identity_sha256",
    "tokenizer_config_identity_sha256",
    "special_token_ids",
    "accepted_artifact_file_sha256",
    "split_evidence_sha256",
    "split_evidence_schema_version",
    "split_evidence_type",
    "source_manifest_sha256",
    "source_manifest_schema_version",
    "source_manifest_identity",
    "source_manifest_accepted_artifact_binding",
    "source_manifest_hybrid_fitting_policy_binding",
    "source_manifest_runtime_policy_binding",
    "source_manifest_immutable_binding_type",
    "source_manifest_immutable_binding_valid",
    "source_manifest_immutable_binding_reference_sha256",
    "source_manifest_expected_sha256",
    "source_manifest_actual_sha256",
    "stable_sample_id_algorithm_identity",
    "artifact_fitting_stable_sample_count",
    "artifact_fitting_stable_sample_set_sha256",
    "artifact_fitting_stable_sample_ordered_sha256",
    "intersection_count",
    "union_count",
    "dataset_name",
    "dataset_config_name",
    "dataset_split",
    "text_column",
    "summary_column",
    "population_mode",
    "population_capped",
    "full_heldout_population_count",
    "heldout_stable_sample_set_sha256",
    "heldout_stable_sample_ordered_sha256",
    "selected_population_count",
    "selected_stable_sample_set_sha256",
    "selected_stable_sample_ordered_sha256",
    "selection_source_type",
    "selection_source_file_sha256",
    "selection_source_sha256",
    "selection_source_schema_version",
    "selection_source_run_identity",
    "selection_source_artifact_identity",
    "selection_source_accepted_artifact_binding",
    "generation_sample_binding_sha256",
    "decoding_configuration_sha256",
    "decoder_layer_count",
)
_APPROXIMATION_POLICY_FIELDS = (
    "candidate_policy_name",
    "candidate_layers",
    "threshold",
    "threshold_comparator",
    "confidence_dtype",
    "adaptive_threshold",
    "fallback_policy",
    "candidate_policy_sha256",
)


def _is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_STRINGS
    return False


def _append_invalid_if_placeholder(failures: List[str], *, method: str, field: str, value: Any) -> None:
    if _is_placeholder_value(value):
        failures.append("{}_{}_missing_or_placeholder".format(method, field))


def checkpoint_inventory_identity_from_model_root(model_root: os.PathLike[str] | str) -> Dict[str, Any]:
    payload = model_checkpoint_identity(model_root)
    inventory_rows = []
    for row in payload.get("included_file_inventory") or []:
        inventory_rows.append(
            {
                "relative_path": row.get("relative_path"),
                "byte_size": int(row.get("byte_size", 0)),
                "file_sha256": row.get("sha256") or row.get("file_sha256"),
            }
        )
    config_path = Path(model_root) / "config.json"
    total_bytes = sum(int(row["byte_size"]) for row in inventory_rows)
    ordered_rows = sorted(inventory_rows, key=lambda item: str(item["relative_path"]))
    return {
        "checkpoint_inventory_schema_version": payload.get("manifest_schema_version"),
        "checkpoint_inventory_file_count": len(ordered_rows),
        "checkpoint_inventory_total_bytes": total_bytes,
        "checkpoint_inventory_sha256": canonical_json_sha256(ordered_rows),
        "checkpoint_config_sha256": sha256_file(config_path),
        "checkpoint_inventory_files": ordered_rows,
    }


def build_decoding_configuration_identity(config: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(config.get("configuration"), Mapping):
        configuration = dict(config.get("configuration") or {})
        field_sources = dict(config.get("field_sources") or {})
    else:
        configuration = dict(config)
        supplied_sources = dict(config.get("field_sources") or {}) if isinstance(config.get("field_sources"), Mapping) else {}
        field_sources = {
            key: supplied_sources.get(key, "caller_supplied.{}".format(key))
            for key in configuration
            if key != "field_sources"
        }
        configuration.pop("field_sources", None)
    payload = {
        "schema_version": 1,
        "identity_type": "f2b_decoding_configuration",
        "configuration": configuration,
        "field_sources": field_sources,
    }
    payload["decoder_layer_count"] = configuration.get("decoder_layer_count")
    payload["decoding_configuration_sha256"] = canonical_json_sha256(payload)
    return payload


def build_f2b_candidate_policy_identity(
    *,
    candidate_layers: Sequence[Any] = F2B_CANDIDATE_LAYERS,
    threshold: Any = F2B_CANDIDATE_THRESHOLD,
    threshold_comparator: Any = F2B_CANDIDATE_THRESHOLD_COMPARATOR,
    confidence_dtype: Any = F2B_CANDIDATE_CONFIDENCE_DTYPE,
    adaptive_threshold: Any = F2B_CANDIDATE_ADAPTIVE_THRESHOLD,
    fallback_policy: Any = F2B_CANDIDATE_FALLBACK_POLICY,
) -> Dict[str, Any]:
    policy = {
        "candidate_policy_name": F2B_CANDIDATE_POLICY_NAME,
        "candidate_layers": [int(layer) for layer in candidate_layers],
        "threshold": float(threshold),
        "threshold_comparator": str(threshold_comparator),
        "confidence_dtype": str(confidence_dtype),
        "adaptive_threshold": bool(adaptive_threshold),
        "fallback_policy": str(fallback_policy),
    }
    policy["candidate_policy_sha256"] = canonical_json_sha256(policy)
    return policy


def build_observed_runtime_calm_candidate_policy_identity() -> Dict[str, Any]:
    from . import missing_kv_calm_trace as calm_trace

    payload = calm_trace.calm_policy_payload(
        candidate_layers=calm_trace.CALM_CANDIDATE_LAYERS,
        threshold=calm_trace.CALM_THRESHOLD,
    )
    policy = {
        "candidate_policy_name": payload.get("policy_name"),
        "candidate_layers": [int(layer) for layer in payload.get("candidate_evaluation_order") or payload.get("candidate_exit_layers") or []],
        "threshold": float(payload.get("threshold")),
        "threshold_comparator": str(payload.get("threshold_comparator")),
        "confidence_dtype": str(payload.get("confidence_compute_dtype")),
        "adaptive_threshold": bool(payload.get("adaptive_threshold")),
        "fallback_policy": str(payload.get("fallback_rule")),
    }
    policy["candidate_policy_sha256"] = canonical_json_sha256(policy)
    return policy


def validate_f2b_checkpoint_inventory_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    files = payload.get("checkpoint_inventory_files")
    if not isinstance(files, list):
        errors.append("checkpoint_inventory_files_missing")
        files = []
    normalized_rows = []
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            errors.append("checkpoint_inventory_file_row_invalid:{}".format(index))
            continue
        normalized = {
            "relative_path": row.get("relative_path"),
            "byte_size": row.get("byte_size"),
            "file_sha256": row.get("file_sha256"),
        }
        for key, value in normalized.items():
            if _is_placeholder_value(value):
                errors.append("checkpoint_inventory_file_{}_missing:{}".format(key, index))
        try:
            normalized["byte_size"] = int(normalized["byte_size"])
        except Exception:
            errors.append("checkpoint_inventory_file_byte_size_invalid:{}".format(index))
        normalized_rows.append(normalized)
    normalized_rows = sorted(normalized_rows, key=lambda item: str(item.get("relative_path")))
    embedded = payload.get("checkpoint_inventory_sha256")
    recomputed = canonical_json_sha256(normalized_rows)
    if embedded != recomputed:
        errors.append("checkpoint_inventory_sha256_mismatch")
    if payload.get("checkpoint_inventory_file_count") != len(normalized_rows):
        errors.append("checkpoint_inventory_file_count_mismatch")
    total_bytes = sum(int(row.get("byte_size") or 0) for row in normalized_rows)
    if payload.get("checkpoint_inventory_total_bytes") != total_bytes:
        errors.append("checkpoint_inventory_total_bytes_mismatch")
    if _is_placeholder_value(payload.get("checkpoint_config_sha256")):
        errors.append("checkpoint_config_sha256_missing")
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "recomputed_checkpoint_inventory_sha256": recomputed,
    }


def validate_f2b_decoding_configuration_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        errors.append("decoding_configuration_missing")
        configuration = {}
    field_sources = payload.get("field_sources")
    if not isinstance(field_sources, Mapping):
        errors.append("decoding_configuration_field_sources_missing")
        field_sources = {}
    required = (
        "num_beams",
        "do_sample",
        "max_source_length",
        "max_target_length",
        "max_new_tokens",
        "effective_generation_max_length",
        "decoder_start_token_id",
        "eos_token_id",
        "pad_token_id",
        "early_stopping",
        "length_penalty",
        "repetition_penalty",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "generation_library_name",
        "generation_library_version",
        "transformers_version",
        "torch_version",
        "decoder_layer_count",
    )
    for key in required:
        if key not in configuration:
            errors.append("decoding_configuration_{}_missing".format(key))
        if key not in field_sources:
            errors.append("decoding_configuration_{}_source_missing".format(key))
    embedded = payload.get("decoding_configuration_sha256")
    recomputed = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "decoding_configuration_sha256"}
    )
    if embedded != recomputed:
        errors.append("decoding_configuration_sha256_mismatch")
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "recomputed_decoding_configuration_sha256": recomputed,
    }


def validate_f2b_candidate_policy_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    for field in _APPROXIMATION_POLICY_FIELDS:
        if field not in payload:
            errors.append("candidate_policy_{}_missing".format(field))
    try:
        expected = build_f2b_candidate_policy_identity(
            candidate_layers=payload.get("candidate_layers") or [],
            threshold=payload.get("threshold"),
            threshold_comparator=payload.get("threshold_comparator"),
            confidence_dtype=payload.get("confidence_dtype"),
            adaptive_threshold=payload.get("adaptive_threshold"),
            fallback_policy=payload.get("fallback_policy"),
        )
        if payload.get("candidate_policy_sha256") != expected.get("candidate_policy_sha256"):
            errors.append("candidate_policy_sha256_mismatch")
    except Exception as exc:
        expected = {}
        errors.append("candidate_policy_invalid:{}".format(type(exc).__name__))
    frozen = build_f2b_candidate_policy_identity()
    for field in _APPROXIMATION_POLICY_FIELDS:
        if payload.get(field) != frozen.get(field):
            errors.append("candidate_policy_{}_not_frozen".format(field))
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "recomputed_candidate_policy_sha256": expected.get("candidate_policy_sha256"),
    }


def build_f2b_method_identity(
    method: str,
    *,
    artifact_file_sha256: Optional[str] = None,
    runtime_policy_sha256: Optional[str] = None,
    hybrid_fitting_policy_sha256: Optional[str] = None,
    repository_commit: Optional[str] = None,
    decoder_layer_count: Optional[int] = None,
    candidate_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    restoration_method = method
    if method == "full_reference":
        restoration_method = "none"
    elif method == "calm_exit_hidden_projection":
        restoration_method = "exit_hidden_target_projection"
    identity: Dict[str, Any] = {
        "schema_version": F2B_ARM_IDENTITY_SCHEMA_VERSION,
        "identity_type": "f2b_runtime_method_identity",
        "repository_commit": repository_commit,
        "runtime_method": method,
        "restoration_method": restoration_method,
        "source_layer_mode": "full_depth" if method == "full_reference" else "candidate_first_crossing",
        "speed_claim_valid": False,
        "calm_early_exit_enabled": method != "full_reference",
        "kv_restoration_enabled": method != "full_reference",
        "candidate_policy": None,
        "candidate_policy_sha256": None,
        "artifact_file_sha256": None,
        "runtime_policy_sha256": None,
        "hybrid_fitting_policy_sha256": None,
        "decoder_layer_count": decoder_layer_count,
    }
    if method != "full_reference":
        candidate_policy = dict(candidate_policy or build_f2b_candidate_policy_identity())
        identity.update(candidate_policy)
        identity["candidate_policy"] = {
            key: candidate_policy[key]
            for key in _APPROXIMATION_POLICY_FIELDS
            if key != "candidate_policy_sha256"
        }
    if method == "phase3c_kv_final":
        identity["artifact_file_sha256"] = artifact_file_sha256
        identity["runtime_policy_sha256"] = runtime_policy_sha256
        identity["hybrid_fitting_policy_sha256"] = hybrid_fitting_policy_sha256
    identity["runtime_method_identity_sha256"] = canonical_json_sha256(
        {key: value for key, value in identity.items() if key != "runtime_method_identity_sha256"}
    )
    return identity


def validate_f2b_runtime_method_identity_payload(payload: Mapping[str, Any], *, method: str) -> Dict[str, Any]:
    errors: List[str] = []
    if payload.get("runtime_method") != method:
        errors.append("runtime_method_mismatch")
    embedded = payload.get("runtime_method_identity_sha256")
    recomputed = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "runtime_method_identity_sha256"}
    )
    if embedded != recomputed:
        errors.append("runtime_method_identity_sha256_mismatch")
    for field in (
        "repository_commit",
        "runtime_method",
        "restoration_method",
        "calm_early_exit_enabled",
        "kv_restoration_enabled",
        "candidate_policy_sha256",
        "decoder_layer_count",
    ):
        if field not in payload:
            errors.append("runtime_method_{}_missing".format(field))
    if method == "full_reference":
        if payload.get("candidate_policy") is not None or payload.get("candidate_policy_sha256") is not None:
            errors.append("full_reference_candidate_policy_not_null")
        if payload.get("calm_early_exit_enabled") is not False or payload.get("kv_restoration_enabled") is not False:
            errors.append("full_reference_restoration_not_disabled")
    else:
        if _is_placeholder_value(payload.get("candidate_policy_sha256")):
            errors.append("runtime_method_candidate_policy_sha256_missing")
        if payload.get("calm_early_exit_enabled") is not True or payload.get("kv_restoration_enabled") is not True:
            errors.append("approximation_restoration_not_enabled")
    if method == "phase3c_kv_final":
        if payload.get("artifact_file_sha256") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            errors.append("phase3c_kv_final_artifact_sha256_not_accepted")
        if payload.get("runtime_policy_sha256") != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
            errors.append("phase3c_kv_final_runtime_policy_sha256_not_accepted")
        if payload.get("hybrid_fitting_policy_sha256") != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
            errors.append("phase3c_kv_final_hybrid_fitting_policy_sha256_not_accepted")
    elif method in ("exact_catchup", "direct_shallow_kv_reuse", "calm_exit_hidden_projection"):
        for field in ("artifact_file_sha256", "runtime_policy_sha256", "hybrid_fitting_policy_sha256"):
            if payload.get(field) is not None:
                errors.append("{}_must_be_null".format(field))
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "recomputed_runtime_method_identity_sha256": recomputed,
    }


def emit_f2b_runtime_provenance_sidecars(
    *,
    model_root: os.PathLike[str] | str,
    checkpoint_inventory_output: Optional[os.PathLike[str] | str],
    decoding_configuration_identity_output: Optional[os.PathLike[str] | str],
    candidate_policy_identity_output: Optional[os.PathLike[str] | str],
    runtime_method_identity_output: Optional[os.PathLike[str] | str],
    runtime_method: str,
    restoration_method: Optional[str],
    calm_early_exit_enabled: bool,
    kv_restoration_enabled: bool,
    decoding_configuration: Mapping[str, Any],
    repository_commit: Optional[str] = None,
    artifact_path: Optional[os.PathLike[str] | str] = None,
    artifact_file_sha256: Optional[str] = None,
    runtime_policy_sha256: Optional[str] = None,
    hybrid_fitting_policy_sha256: Optional[str] = None,
    candidate_layers: Sequence[Any] = F2B_CANDIDATE_LAYERS,
    threshold: Any = F2B_CANDIDATE_THRESHOLD,
    threshold_comparator: Any = F2B_CANDIDATE_THRESHOLD_COMPARATOR,
    confidence_dtype: Any = F2B_CANDIDATE_CONFIDENCE_DTYPE,
    adaptive_threshold: Any = F2B_CANDIDATE_ADAPTIVE_THRESHOLD,
    fallback_policy: Any = F2B_CANDIDATE_FALLBACK_POLICY,
) -> Dict[str, Any]:
    if repository_commit is None:
        try:
            repository_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            repository_commit = None
    checkpoint_payload = checkpoint_inventory_identity_from_model_root(model_root)
    config = dict(decoding_configuration)
    decoder_layer_count = config.get("decoder_layer_count")
    decoding_payload = build_decoding_configuration_identity(config)

    method = str(runtime_method)
    if method not in F2B_RUNTIME_METHODS:
        raise ValueError("unsupported_f2b_runtime_method:{}".format(method))
    candidate_payload: Optional[Dict[str, Any]] = None
    if method == "full_reference":
        if calm_early_exit_enabled or kv_restoration_enabled:
            raise ValueError("full_reference_restoration_must_be_disabled")
        null_policy_payload = {
            "schema_version": F2B_RUNTIME_PROVENANCE_SCHEMA_VERSION,
            "identity_type": "f2b_null_candidate_policy",
            "candidate_policy": None,
            "candidate_policy_sha256": None,
        }
        if candidate_policy_identity_output:
            write_json(Path(candidate_policy_identity_output), null_policy_payload)
    else:
        if not calm_early_exit_enabled or not kv_restoration_enabled:
            raise ValueError("approximation_restoration_must_be_enabled")
        supplied_candidate_payload = build_f2b_candidate_policy_identity(
            candidate_layers=candidate_layers,
            threshold=threshold,
            threshold_comparator=threshold_comparator,
            confidence_dtype=confidence_dtype,
            adaptive_threshold=adaptive_threshold,
            fallback_policy=fallback_policy,
        )
        candidate_payload = build_observed_runtime_calm_candidate_policy_identity()
        for key in _APPROXIMATION_POLICY_FIELDS:
            if candidate_payload.get(key) != supplied_candidate_payload.get(key):
                raise ValueError("f2b_candidate_policy_runtime_cli_{}_mismatch".format(key))
        frozen_policy = build_f2b_candidate_policy_identity()
        for key in _APPROXIMATION_POLICY_FIELDS:
            if candidate_payload.get(key) != frozen_policy.get(key):
                raise ValueError("f2b_candidate_policy_{}_mismatch".format(key))
        if candidate_policy_identity_output:
            write_json(Path(candidate_policy_identity_output), candidate_payload)

    if method == "phase3c_kv_final":
        if artifact_file_sha256 is None and artifact_path:
            artifact_file_sha256 = sha256_file(artifact_path)
        if artifact_file_sha256 != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            raise ValueError("phase3c_kv_final_artifact_sha256_not_accepted")
        if runtime_policy_sha256 != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
            raise ValueError("phase3c_kv_final_runtime_policy_sha256_not_accepted")
        if hybrid_fitting_policy_sha256 != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
            raise ValueError("phase3c_kv_final_hybrid_fitting_policy_sha256_not_accepted")
    else:
        artifact_file_sha256 = None
        runtime_policy_sha256 = None
        hybrid_fitting_policy_sha256 = None

    runtime_payload = build_f2b_method_identity(
        method,
        artifact_file_sha256=artifact_file_sha256,
        runtime_policy_sha256=runtime_policy_sha256,
        hybrid_fitting_policy_sha256=hybrid_fitting_policy_sha256,
        repository_commit=repository_commit,
        decoder_layer_count=None if decoder_layer_count is None else int(decoder_layer_count),
        candidate_policy=candidate_payload,
    )
    runtime_payload["calm_early_exit_enabled"] = bool(calm_early_exit_enabled)
    runtime_payload["kv_restoration_enabled"] = bool(kv_restoration_enabled)
    runtime_payload["restoration_method"] = restoration_method if restoration_method is not None else runtime_payload["restoration_method"]
    runtime_payload["runtime_method_identity_sha256"] = canonical_json_sha256(
        {key: value for key, value in runtime_payload.items() if key != "runtime_method_identity_sha256"}
    )

    if checkpoint_inventory_output:
        write_json(Path(checkpoint_inventory_output), checkpoint_payload)
    if decoding_configuration_identity_output:
        write_json(Path(decoding_configuration_identity_output), decoding_payload)
    if runtime_method_identity_output:
        write_json(Path(runtime_method_identity_output), runtime_payload)
    return {
        "checkpoint_inventory": checkpoint_payload,
        "decoding_configuration_identity": decoding_payload,
        "candidate_policy_identity": candidate_payload,
        "runtime_method_identity": runtime_payload,
    }


def _json_or_failure(path: Path, *, method: str, label: str, failures: List[str]) -> Dict[str, Any]:
    if not path.is_file():
        failures.append("{}_{}_missing".format(method, label))
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        failures.append("{}_{}_parse_failed:{}".format(method, label, type(exc).__name__))
        return {}


def _jsonl_or_failure(path: Path, *, method: str, label: str, failures: List[str]) -> List[Dict[str, Any]]:
    if not path.is_file():
        failures.append("{}_{}_missing".format(method, label))
        return []
    try:
        return read_jsonl(path)
    except Exception as exc:
        failures.append("{}_{}_parse_failed:{}".format(method, label, type(exc).__name__))
        return []


def _first_existing_payload(paths: Sequence[Path], *, method: str, label: str, failures: List[str]) -> Dict[str, Any]:
    for path in paths:
        if path.is_file():
            return _json_or_failure(path, method=method, label=label, failures=failures)
    failures.append("{}_{}_missing".format(method, label))
    return {}


def _special_token_ids_from_tokenizer_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("special_token_ids"), Mapping):
        source = payload.get("special_token_ids") or {}
    else:
        semantic = payload.get("semantic_content") if isinstance(payload.get("semantic_content"), Mapping) else {}
        source = semantic.get("special_token_ids") if isinstance(semantic.get("special_token_ids"), Mapping) else {}
    return {field: source.get(field) for field in _SPECIAL_TOKEN_FIELDS}


def _object_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _resolve_effective_field(
    field: str,
    *,
    explicit_generate_kwargs: Optional[Mapping[str, Any]] = None,
    trainer_overrides: Optional[Mapping[str, Any]] = None,
    generation_config: Any = None,
    model_config: Any = None,
    aliases: Sequence[str] = (),
) -> Tuple[Any, str]:
    names = (field, *aliases)
    for source_name, source in (
        ("explicit_generate_kwargs", explicit_generate_kwargs),
        ("trainer_generation_override", trainer_overrides),
        ("model_generation_config", generation_config),
        ("model_config", model_config),
    ):
        for name in names:
            value = _object_value(source, name)
            if value is not None:
                return value, "{}.{}".format(source_name, name)
    return None, "missing"


def _maybe_int_value(value: Any) -> Any:
    if value is None:
        return None
    return int(value)


def _maybe_float_value(value: Any) -> Any:
    if value is None:
        return None
    return float(value)


def resolve_effective_decoding_configuration(
    *,
    data_args: Any = None,
    training_args: Any = None,
    model_config: Any = None,
    generation_config: Any = None,
    generate_kwargs: Optional[Mapping[str, Any]] = None,
    generation_library_name: str = "transformers",
    generation_library_version: Optional[str] = None,
    transformers_version: Optional[str] = None,
    torch_version: Optional[str] = None,
) -> Dict[str, Any]:
    trainer_overrides: Dict[str, Any] = {}
    if training_args is not None:
        trainer_overrides["num_beams"] = _object_value(training_args, "generation_num_beams")
        trainer_overrides["max_length"] = _object_value(training_args, "generation_max_length")
        trainer_overrides["max_new_tokens"] = _object_value(training_args, "generation_max_new_tokens")
        trainer_overrides["do_sample"] = _object_value(training_args, "do_sample")
    if data_args is not None:
        if _object_value(data_args, "num_beams") is not None:
            trainer_overrides["num_beams"] = _object_value(data_args, "num_beams")
        trainer_overrides["max_source_length"] = _object_value(data_args, "max_source_length")
        trainer_overrides["max_target_length"] = _object_value(data_args, "val_max_target_length") or _object_value(data_args, "max_target_length")
    fields: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for field, aliases, caster in (
        ("num_beams", (), _maybe_int_value),
        ("do_sample", (), bool),
        ("max_source_length", (), _maybe_int_value),
        ("max_target_length", (), _maybe_int_value),
        ("max_new_tokens", (), _maybe_int_value),
        ("effective_generation_max_length", ("max_length",), _maybe_int_value),
        ("decoder_start_token_id", (), _maybe_int_value),
        ("eos_token_id", (), _maybe_int_value),
        ("pad_token_id", (), _maybe_int_value),
        ("early_stopping", (), bool),
        ("length_penalty", (), _maybe_float_value),
        ("repetition_penalty", (), _maybe_float_value),
        ("forced_bos_token_id", (), _maybe_int_value),
        ("forced_eos_token_id", (), _maybe_int_value),
        ("decoder_layer_count", ("num_decoder_layers", "num_layers"), _maybe_int_value),
    ):
        value, source = _resolve_effective_field(
            field,
            explicit_generate_kwargs=generate_kwargs,
            trainer_overrides=trainer_overrides,
            generation_config=generation_config,
            model_config=model_config,
            aliases=aliases,
        )
        fields[field] = caster(value) if value is not None else None
        sources[field] = source
    fields["generation_library_name"] = generation_library_name
    fields["generation_library_version"] = generation_library_version
    fields["transformers_version"] = transformers_version
    fields["torch_version"] = torch_version
    for key in ("generation_library_name", "generation_library_version", "transformers_version", "torch_version"):
        sources[key] = "runtime_library"
    return {"configuration": fields, "field_sources": sources}


def extract_f2b_arm_identity(
    arm_dir: Path,
    *,
    method: str,
    artifact_file_sha256: Optional[str] = None,
    runtime_policy_sha256: Optional[str] = None,
    hybrid_fitting_policy_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    arm_dir = Path(arm_dir)
    failures: List[str] = []
    provenance_dir = arm_dir / "missing_kv_provenance"
    population_rows = _jsonl_or_failure(provenance_dir / "effective_eval_population.jsonl", method=method, label="effective_population", failures=failures)
    population_summary = _json_or_failure(provenance_dir / "effective_eval_population_summary.json", method=method, label="effective_population_summary", failures=failures)
    generation_rows = _jsonl_or_failure(provenance_dir / "generation_sample_binding.jsonl", method=method, label="generation_binding", failures=failures)
    generation_summary = _json_or_failure(provenance_dir / "generation_sample_binding_summary.json", method=method, label="generation_binding_summary", failures=failures)
    tokenizer_payload = _json_or_failure(provenance_dir / "tokenizer_identity.json", method=method, label="tokenizer_identity", failures=failures)
    checkpoint_payload = _json_or_failure(provenance_dir / "checkpoint_inventory.json", method=method, label="checkpoint_inventory", failures=failures)
    decoding_payload = _json_or_failure(provenance_dir / "decoding_configuration_identity.json", method=method, label="decoding_configuration_identity", failures=failures)
    runtime_method_payload = _json_or_failure(provenance_dir / "runtime_method_identity.json", method=method, label="runtime_method_identity", failures=failures)
    candidate_policy_payload: Dict[str, Any] = {}
    if method != "full_reference":
        candidate_policy_payload = _json_or_failure(
            provenance_dir / "candidate_policy_identity.json",
            method=method,
            label="candidate_policy_identity",
            failures=failures,
        )
    elif (provenance_dir / "candidate_policy_identity.json").is_file():
        candidate_policy_payload = _json_or_failure(
            provenance_dir / "candidate_policy_identity.json",
            method=method,
            label="candidate_policy_identity",
            failures=failures,
        )
    eval_results = _first_existing_payload(
        (
            arm_dir / "outputs" / "eval_results.json",
            arm_dir / "full_depth" / "eval_results.json",
            arm_dir / "calm_restore_bench" / "eval_results.json",
        ),
        method=method,
        label="eval_results",
        failures=failures,
    )

    effective_validation = validate_effective_population_sidecar(population_rows, population_summary)
    if effective_validation.get("status") != "ok":
        failures.append("{}_effective_population_sidecar_invalid".format(method))
    generation_validation = validate_generation_binding_sidecar(population_rows, generation_rows, generation_summary)
    if generation_validation.get("status") != "ok":
        failures.append("{}_generation_binding_sidecar_invalid".format(method))
    tokenizer_validation = validate_tokenizer_identity_payload(tokenizer_payload)
    if tokenizer_validation.get("status") != "ok":
        failures.append("{}_tokenizer_identity_invalid".format(method))
    checkpoint_validation = validate_f2b_checkpoint_inventory_payload(checkpoint_payload)
    if checkpoint_validation.get("status") != "ok":
        failures.append("{}_checkpoint_inventory_invalid".format(method))
        failures.extend("{}_{}".format(method, error) for error in checkpoint_validation.get("errors") or [])
    decoding_validation = validate_f2b_decoding_configuration_payload(decoding_payload)
    if decoding_validation.get("status") != "ok":
        failures.append("{}_decoding_configuration_identity_invalid".format(method))
        failures.extend("{}_{}".format(method, error) for error in decoding_validation.get("errors") or [])
    runtime_method_validation = validate_f2b_runtime_method_identity_payload(runtime_method_payload, method=method)
    if runtime_method_validation.get("status") != "ok":
        failures.append("{}_runtime_method_identity_invalid".format(method))
        failures.extend("{}_{}".format(method, error) for error in runtime_method_validation.get("errors") or [])
    candidate_policy_validation: Dict[str, Any] = {"status": "skipped_full_reference", "errors": []}
    if method != "full_reference":
        candidate_policy_validation = validate_f2b_candidate_policy_payload(candidate_policy_payload)
        if candidate_policy_validation.get("status") != "ok":
            failures.append("{}_candidate_policy_identity_invalid".format(method))
            failures.extend("{}_{}".format(method, error) for error in candidate_policy_validation.get("errors") or [])
        if runtime_method_payload.get("candidate_policy_sha256") != candidate_policy_payload.get("candidate_policy_sha256"):
            failures.append("{}_runtime_candidate_policy_sha256_mismatch".format(method))

    for index, row in enumerate(population_rows):
        for key in ("stable_sample_id", "selected_order", "raw_dataset_index", "dataset_provided_id"):
            if key not in row:
                failures.append("{}_effective_population_{}_missing:{}".format(method, key, index))
    for index, row in enumerate(generation_rows):
        for key in ("stable_sample_id", "selected_order", "raw_dataset_index", "dataset_provided_id"):
            if key not in row:
                failures.append("{}_generation_binding_{}_missing:{}".format(method, key, index))

    dataset_metadata = population_summary.get("dataset_metadata") if isinstance(population_summary.get("dataset_metadata"), Mapping) else {}
    checkpoint_inventory_sha = checkpoint_payload.get("checkpoint_inventory_sha256")
    checkpoint_config_sha = checkpoint_payload.get("checkpoint_config_sha256") or eval_results.get("checkpoint_config_sha256")
    tokenizer_config_sha = (
        tokenizer_payload.get("tokenizer_config_identity_sha256")
        or tokenizer_payload.get("tokenizer_asset_inventory_sha256")
        or eval_results.get("tokenizer_config_identity_sha256")
    )
    special_token_ids = _special_token_ids_from_tokenizer_payload(tokenizer_payload)
    decoding_configuration = (
        decoding_payload.get("configuration")
        if isinstance(decoding_payload.get("configuration"), Mapping)
        else {}
    )
    selected_population_count = (
        population_summary["selected_population_count"]
        if "selected_population_count" in population_summary
        else len(population_rows)
    )
    common_identity = {
        "repository_commit": runtime_method_payload.get("repository_commit") or eval_results.get("repository_commit") or generation_summary.get("repository_commit"),
        "checkpoint_inventory_sha256": checkpoint_inventory_sha,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "checkpoint_inventory_schema_version": checkpoint_payload.get("checkpoint_inventory_schema_version"),
        "checkpoint_inventory_file_count": checkpoint_payload.get("checkpoint_inventory_file_count"),
        "checkpoint_inventory_total_bytes": checkpoint_payload.get("checkpoint_inventory_total_bytes"),
        "tokenizer_identity_sha256": tokenizer_payload.get("tokenizer_identity_sha256"),
        "tokenizer_config_identity_sha256": tokenizer_config_sha,
        "special_token_ids": special_token_ids,
        "accepted_artifact_file_sha256": population_summary.get("accepted_artifact_file_sha256"),
        "split_evidence_sha256": population_summary.get("split_evidence_sha256"),
        "split_evidence_schema_version": population_summary.get("split_evidence_schema_version"),
        "split_evidence_type": population_summary.get("split_evidence_type"),
        "source_manifest_sha256": population_summary.get("source_manifest_sha256"),
        "source_manifest_schema_version": population_summary.get("source_manifest_schema_version"),
        "source_manifest_identity": population_summary.get("source_manifest_identity"),
        "source_manifest_accepted_artifact_binding": population_summary.get("source_manifest_accepted_artifact_binding"),
        "source_manifest_hybrid_fitting_policy_binding": population_summary.get("source_manifest_hybrid_fitting_policy_binding"),
        "source_manifest_runtime_policy_binding": population_summary.get("source_manifest_runtime_policy_binding"),
        "source_manifest_immutable_binding_type": population_summary.get("source_manifest_immutable_binding_type"),
        "source_manifest_immutable_binding_valid": population_summary.get("source_manifest_immutable_binding_valid"),
        "source_manifest_immutable_binding_reference_sha256": population_summary.get("source_manifest_immutable_binding_reference_sha256"),
        "source_manifest_expected_sha256": population_summary.get("source_manifest_expected_sha256"),
        "source_manifest_actual_sha256": population_summary.get("source_manifest_actual_sha256"),
        "stable_sample_id_algorithm_identity": population_summary.get("stable_sample_id_algorithm_identity"),
        "artifact_fitting_stable_sample_count": population_summary.get("artifact_fitting_stable_sample_count"),
        "artifact_fitting_stable_sample_set_sha256": population_summary.get("artifact_fitting_stable_sample_set_sha256"),
        "artifact_fitting_stable_sample_ordered_sha256": population_summary.get("artifact_fitting_stable_sample_ordered_sha256"),
        "intersection_count": population_summary.get("intersection_count"),
        "union_count": population_summary.get("union_count"),
        "dataset_name": population_summary.get("dataset_name") or dataset_metadata.get("dataset_name") or (population_rows[0].get("dataset_name") if population_rows else None),
        "dataset_config_name": population_summary.get("dataset_config_name") or dataset_metadata.get("dataset_config_name") or (population_rows[0].get("dataset_config_name") if population_rows else None),
        "dataset_split": population_summary.get("dataset_split") or dataset_metadata.get("split") or (population_rows[0].get("split") if population_rows else None),
        "text_column": population_summary.get("text_column") or dataset_metadata.get("text_column") or (population_rows[0].get("text_column") if population_rows else None),
        "summary_column": population_summary.get("summary_column") or dataset_metadata.get("summary_column") or (population_rows[0].get("summary_column") if population_rows else None),
        "population_mode": population_summary.get("population_mode"),
        "population_capped": population_summary.get("population_capped"),
        "full_heldout_population_count": population_summary.get("full_heldout_population_count"),
        "heldout_stable_sample_set_sha256": population_summary.get("heldout_stable_sample_set_sha256"),
        "heldout_stable_sample_ordered_sha256": population_summary.get("heldout_stable_sample_ordered_sha256"),
        "selected_population_count": selected_population_count,
        "selected_stable_sample_set_sha256": population_summary.get("selected_stable_sample_set_sha256"),
        "selected_stable_sample_ordered_sha256": population_summary.get("selected_stable_sample_ordered_sha256"),
        "selection_source_type": population_summary.get("selection_source_type"),
        "selection_source_file_sha256": population_summary.get("selection_source_file_sha256"),
        "selection_source_sha256": population_summary.get("selection_source_sha256"),
        "selection_source_schema_version": population_summary.get("selection_source_schema_version"),
        "selection_source_run_identity": population_summary.get("selection_source_run_identity"),
        "selection_source_artifact_identity": population_summary.get("selection_source_artifact_identity"),
        "selection_source_accepted_artifact_binding": population_summary.get("selection_source_accepted_artifact_binding"),
        "generation_sample_binding_sha256": generation_summary.get("generation_sample_binding_sha256"),
        "decoding_configuration_sha256": decoding_payload.get("decoding_configuration_sha256") or eval_results.get("decoding_configuration_sha256"),
        "decoder_layer_count": (
            eval_results.get("decoder_layer_count")
            or decoding_configuration.get("decoder_layer_count")
            or runtime_method_payload.get("decoder_layer_count")
        ),
    }
    nullable_common_fields = {"dataset_config_name"}
    if common_identity.get("selection_source_type") == "artifact_heldout_complement":
        nullable_common_fields.add("selection_source_run_identity")
    for field in _COMMON_ARM_IDENTITY_FIELDS:
        if field in nullable_common_fields:
            continue
        _append_invalid_if_placeholder(failures, method=method, field=field, value=common_identity.get(field))
    if common_identity.get("population_mode") != "artifact_heldout":
        failures.append("{}_population_mode_not_artifact_heldout".format(method))
    if common_identity.get("population_capped") is False and common_identity.get("selection_source_type") != "artifact_heldout_complement":
        failures.append("{}_full_selection_source_type_mismatch".format(method))
    if common_identity.get("population_capped") is True and common_identity.get("selection_source_type") == "artifact_heldout_complement":
        failures.append("{}_smoke_selection_source_type_mismatch".format(method))
    if common_identity.get("full_heldout_population_count") not in (F2B_EXPECTED_HELDOUT_COUNT, str(F2B_EXPECTED_HELDOUT_COUNT)):
        failures.append("{}_full_heldout_population_count_mismatch".format(method))
    try:
        selected_ids_from_rows = [str(row.get("stable_sample_id")) for row in population_rows]
        if common_identity.get("selected_stable_sample_set_sha256") != stable_sample_ids_sha256(selected_ids_from_rows, sort_ids=True):
            failures.append("{}_selected_stable_sample_set_sha256_mismatch".format(method))
        if common_identity.get("selected_stable_sample_ordered_sha256") != stable_sample_ids_sha256(selected_ids_from_rows, sort_ids=False):
            failures.append("{}_selected_stable_sample_ordered_sha256_mismatch".format(method))
    except Exception as exc:
        failures.append("{}_selected_stable_sample_identity_recompute_failed:{}".format(method, type(exc).__name__))
    if common_identity.get("accepted_artifact_file_sha256") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
        failures.append("{}_common_identity_accepted_artifact_file_sha256_mismatch".format(method))
    if common_identity.get("split_evidence_schema_version") != _paper_population.ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION:
        failures.append("{}_split_evidence_schema_version_mismatch".format(method))
    if common_identity.get("split_evidence_type") != _paper_population.ARTIFACT_SPLIT_EVIDENCE_TYPE:
        failures.append("{}_split_evidence_type_mismatch".format(method))
    if common_identity.get("stable_sample_id_algorithm_identity") != "missing_kv_dump_provenance.stable_sample_id:v1":
        failures.append("{}_stable_sample_id_algorithm_identity_mismatch".format(method))
    if common_identity.get("dataset_name") != "knkarthick/samsum":
        failures.append("{}_dataset_name_mismatch".format(method))
    if common_identity.get("dataset_config_name") is not None:
        failures.append("{}_dataset_config_name_mismatch".format(method))
    if common_identity.get("dataset_split") != "validation":
        failures.append("{}_dataset_split_mismatch".format(method))
    if common_identity.get("text_column") != "dialogue":
        failures.append("{}_text_column_mismatch".format(method))
    if common_identity.get("summary_column") != "summary":
        failures.append("{}_summary_column_mismatch".format(method))
    if common_identity.get("artifact_fitting_stable_sample_count") not in (F2B_EXPECTED_ARTIFACT_FITTING_COUNT, str(F2B_EXPECTED_ARTIFACT_FITTING_COUNT)):
        failures.append("{}_artifact_fitting_stable_sample_count_mismatch".format(method))
    if common_identity.get("intersection_count") not in (0, "0"):
        failures.append("{}_intersection_count_mismatch".format(method))
    if common_identity.get("union_count") not in (F2B_EXPECTED_DATASET_POPULATION_COUNT, str(F2B_EXPECTED_DATASET_POPULATION_COUNT)):
        failures.append("{}_union_count_mismatch".format(method))
    for key in (
        "source_manifest_accepted_artifact_binding",
        "source_manifest_hybrid_fitting_policy_binding",
        "source_manifest_runtime_policy_binding",
        "source_manifest_immutable_binding_valid",
        "selection_source_accepted_artifact_binding",
    ):
        if common_identity.get(key) is not True:
            failures.append("{}_{}_not_true".format(method, key))
    if common_identity.get("source_manifest_expected_sha256") != common_identity.get("source_manifest_actual_sha256"):
        failures.append("{}_source_manifest_expected_actual_sha256_mismatch".format(method))
    if common_identity.get("source_manifest_actual_sha256") != common_identity.get("source_manifest_sha256"):
        failures.append("{}_source_manifest_actual_sha256_mismatch".format(method))
    if common_identity.get("source_manifest_immutable_binding_type") not in _paper_population.SOURCE_MANIFEST_IMMUTABLE_BINDING_TYPES:
        failures.append("{}_source_manifest_immutable_binding_type_mismatch".format(method))
    if common_identity.get("selection_source_type") == "artifact_heldout_complement":
        if common_identity.get("selection_source_sha256") != common_identity.get("split_evidence_sha256"):
            failures.append("{}_full_selection_source_sha256_mismatch".format(method))
        if common_identity.get("selection_source_run_identity") is not None:
            failures.append("{}_full_selection_source_run_identity_not_null".format(method))
        if common_identity.get("selection_source_artifact_identity") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("{}_full_selection_source_artifact_identity_mismatch".format(method))
    else:
        if common_identity.get("selection_source_artifact_identity") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("{}_selection_source_artifact_identity_mismatch".format(method))

    method_identity = dict(runtime_method_payload)
    if method != "full_reference":
        for field in _APPROXIMATION_POLICY_FIELDS:
            method_identity[field] = candidate_policy_payload.get(field)
        method_identity["candidate_policy"] = {
            key: candidate_policy_payload.get(key)
            for key in _APPROXIMATION_POLICY_FIELDS
            if key != "candidate_policy_sha256"
        }
        method_identity["candidate_policy_sha256"] = candidate_policy_payload.get("candidate_policy_sha256")
    if method == "phase3c_kv_final":
        if method_identity.get("artifact_file_sha256") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("{}_accepted_artifact_sha256_mismatch".format(method))
        if method_identity.get("runtime_policy_sha256") != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
            failures.append("{}_runtime_policy_sha256_mismatch".format(method))
        if method_identity.get("hybrid_fitting_policy_sha256") != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
            failures.append("{}_hybrid_fitting_policy_sha256_mismatch".format(method))
        if artifact_file_sha256 is not None and method_identity.get("artifact_file_sha256") != artifact_file_sha256:
            failures.append("{}_preflight_artifact_sha256_mismatch".format(method))
        if runtime_policy_sha256 is not None and method_identity.get("runtime_policy_sha256") != runtime_policy_sha256:
            failures.append("{}_preflight_runtime_policy_sha256_mismatch".format(method))
        if hybrid_fitting_policy_sha256 is not None and method_identity.get("hybrid_fitting_policy_sha256") != hybrid_fitting_policy_sha256:
            failures.append("{}_preflight_hybrid_fitting_policy_sha256_mismatch".format(method))
    elif method != "full_reference":
        if not _is_placeholder_value(method_identity.get("artifact_file_sha256")):
            failures.append("{}_unexpected_artifact_identity".format(method))

    return {
        "schema_version": F2B_ARM_IDENTITY_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "method": method,
        "failures": failures,
        "common_identity": common_identity,
        "method_identity": method_identity,
        "effective_population_validation": effective_validation,
        "generation_binding_validation": generation_validation,
        "tokenizer_identity_validation": tokenizer_validation,
        "checkpoint_inventory_validation": checkpoint_validation,
        "decoding_configuration_validation": decoding_validation,
        "candidate_policy_validation": candidate_policy_validation,
        "runtime_method_identity_validation": runtime_method_validation,
        "checkpoint_identity_valid": not any("checkpoint" in failure for failure in failures),
        "tokenizer_identity_valid": tokenizer_validation.get("status") == "ok"
        and not any("tokenizer" in failure for failure in failures),
        "decoding_identity_valid": not any("decoding_configuration" in failure for failure in failures),
        "generation_binding_valid": generation_validation.get("status") == "ok",
        "heldout_population_valid": not any("heldout" in failure or "population_mode" in failure for failure in failures),
    }


def validate_f2b_cross_arm_identities(arm_identities: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    per_field: Dict[str, Any] = {}
    for method in F2B_METHODS:
        identity = arm_identities.get(method)
        if not isinstance(identity, Mapping):
            failures.append("{}_arm_identity_missing".format(method))
            continue
        if identity.get("status") != "ok":
            failures.append("{}_arm_identity_invalid".format(method))
    common_by_method = {
        method: (arm_identities.get(method, {}).get("common_identity") if isinstance(arm_identities.get(method), Mapping) else None)
        for method in F2B_METHODS
    }
    reference = common_by_method.get("full_reference") if isinstance(common_by_method.get("full_reference"), Mapping) else {}
    for field in _COMMON_ARM_IDENTITY_FIELDS:
        values = {
            method: (common_by_method.get(method) or {}).get(field) if isinstance(common_by_method.get(method), Mapping) else None
            for method in F2B_METHODS
        }
        per_field[field] = values
        for method, value in values.items():
            source_type = (common_by_method.get(method) or {}).get("selection_source_type") if isinstance(common_by_method.get(method), Mapping) else None
            placeholder_allowed = field == "dataset_config_name" or (
                field == "selection_source_run_identity"
                and source_type == "artifact_heldout_complement"
            )
            if not placeholder_allowed and _is_placeholder_value(value):
                failures.append("{}_common_identity_{}_missing_or_placeholder".format(method, field))
        for method in F2B_METHODS[1:]:
            if values.get(method) != reference.get(field):
                failures.append("{}_common_identity_{}_mismatch".format(method, field))
    for method, common_identity in common_by_method.items():
        if not isinstance(common_identity, Mapping):
            continue
        if common_identity.get("selection_source_artifact_identity") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("{}_selection_source_artifact_identity_mismatch".format(method))
        for key in (
            "source_manifest_accepted_artifact_binding",
            "source_manifest_hybrid_fitting_policy_binding",
            "source_manifest_runtime_policy_binding",
            "source_manifest_immutable_binding_valid",
            "selection_source_accepted_artifact_binding",
        ):
            if common_identity.get(key) is not True:
                failures.append("{}_{}_not_true".format(method, key))

    approx_reference_method = APPROXIMATION_METHODS[0]
    approx_ref_identity = arm_identities.get(approx_reference_method, {}).get("method_identity") if isinstance(arm_identities.get(approx_reference_method), Mapping) else {}
    approx_per_field: Dict[str, Any] = {}
    for method in APPROXIMATION_METHODS:
        method_identity = arm_identities.get(method, {}).get("method_identity") if isinstance(arm_identities.get(method), Mapping) else {}
        for field in _APPROXIMATION_POLICY_FIELDS:
            value = method_identity.get(field) if isinstance(method_identity, Mapping) else None
            approx_per_field.setdefault(field, {})[method] = value
            if _is_placeholder_value(value):
                failures.append("{}_candidate_policy_{}_missing_or_placeholder".format(method, field))
            if method != approx_reference_method and value != approx_ref_identity.get(field):
                failures.append("{}_candidate_policy_{}_mismatch".format(method, field))
    full_method_identity = arm_identities.get("full_reference", {}).get("method_identity") if isinstance(arm_identities.get("full_reference"), Mapping) else {}
    if isinstance(full_method_identity, Mapping):
        if full_method_identity.get("candidate_policy") is not None or full_method_identity.get("candidate_policy_sha256") is not None:
            failures.append("full_reference_candidate_policy_not_null")
        if full_method_identity.get("calm_early_exit_enabled") is not False or full_method_identity.get("kv_restoration_enabled") is not False:
            failures.append("full_reference_restoration_not_disabled")
    phase3c_identity = arm_identities.get("phase3c_kv_final", {}).get("method_identity") if isinstance(arm_identities.get("phase3c_kv_final"), Mapping) else {}
    if isinstance(phase3c_identity, Mapping):
        if phase3c_identity.get("artifact_file_sha256") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("phase3c_kv_final_artifact_sha256_not_accepted")
    for method in ("direct_shallow_kv_reuse", "calm_exit_hidden_projection"):
        identity = arm_identities.get(method, {}).get("method_identity") if isinstance(arm_identities.get(method), Mapping) else {}
        if isinstance(identity, Mapping) and identity.get("artifact_file_sha256") is not None:
            failures.append("{}_artifact_identity_not_null".format(method))
    return {
        "schema_version": F2B_ARM_IDENTITY_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "cross_arm_common_identity_valid": not any("common_identity" in failure for failure in failures),
        "approximation_arm_policy_identity_valid": not any("candidate_policy" in failure for failure in failures),
        "per_field_comparison": per_field,
        "approximation_policy_field_comparison": approx_per_field,
        "failure_reasons": failures,
        "warnings": warnings,
        "common_fields": list(_COMMON_ARM_IDENTITY_FIELDS),
        "approximation_only_fields": list(_APPROXIMATION_POLICY_FIELDS),
        "intentional_differences": [
            "arm name",
            "restoration method",
            "artifact usage",
            "generated token trajectory",
            "candidate event schedule",
            "candidate source-layer schedule",
            "prediction token SHA",
            "prediction text",
            "generation length",
        ],
    }


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("{}:{} must contain a JSON object".format(path, line_no))
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _digest_values(values: Iterable[Any]) -> str:
    return stable_sample_ids_sha256([str(value) for value in values], sort_ids=False)


def _stable_sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("stable_sample_id")
    if value in (None, ""):
        raise ValueError("missing_stable_sample_id")
    return str(value)


def _reference_key(row: Mapping[str, Any]) -> Any:
    return row.get("reference_token_sha256") or row.get("reference_text_sha256") or row.get("reference_text")


def _prediction_hash(row: Mapping[str, Any]) -> Any:
    return row.get("prediction_token_sha256") or canonical_json_sha256(row.get("prediction_text", ""))


def _generated_length(row: Mapping[str, Any]) -> Optional[int]:
    value = row.get("generated_length")
    if value is None:
        return None
    return int(value)


def index_prediction_rows(rows: Sequence[Mapping[str, Any]], *, method: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for row in rows:
        try:
            stable = _stable_sample_id(row)
        except ValueError:
            failures.append("{}_prediction_missing_stable_sample_id".format(method))
            continue
        if stable in indexed:
            failures.append("{}_prediction_duplicate_stable_sample_id:{}".format(method, stable))
            continue
        indexed[stable] = dict(row)
    return indexed, failures


def validate_cross_arm_population(predictions_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    failures: List[str] = []
    indexed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ordered_ids: Dict[str, List[str]] = {}
    reference_method = F2B_METHODS[0]
    for method in F2B_METHODS:
        rows = list(predictions_by_method.get(method) or [])
        method_index, method_failures = index_prediction_rows(rows, method=method)
        failures.extend(method_failures)
        indexed[method] = method_index
        ordered_ids[method] = [str(row.get("stable_sample_id")) for row in rows if row.get("stable_sample_id") not in (None, "")]
    reference_ids = list(ordered_ids.get(reference_method) or [])
    reference_set = set(reference_ids)
    for method in F2B_METHODS[1:]:
        method_ids = ordered_ids.get(method) or []
        method_set = set(method_ids)
        for stable in sorted(reference_set - method_set):
            failures.append("{}_missing_sample:{}".format(method, stable))
        for stable in sorted(method_set - reference_set):
            failures.append("{}_extra_sample:{}".format(method, stable))
        if method_ids != reference_ids:
            failures.append("{}_ordered_stable_sample_id_mismatch".format(method))
        for stable in sorted(reference_set & method_set):
            if _reference_key(indexed[method][stable]) != _reference_key(indexed[reference_method][stable]):
                failures.append("{}_reference_mismatch:{}".format(method, stable))
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "sample_count": len(reference_ids),
        "ordered_stable_sample_sha256": _digest_values(reference_ids),
        "stable_sample_set_sha256": _digest_values(sorted(reference_set)),
        "method_sample_counts": {method: len(ordered_ids.get(method) or []) for method in F2B_METHODS},
    }


def pair_prediction_rows(predictions_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    validation = validate_cross_arm_population(predictions_by_method)
    if validation["status"] != "ok":
        return [], validation
    indexed = {
        method: index_prediction_rows(list(predictions_by_method.get(method) or []), method=method)[0]
        for method in F2B_METHODS
    }
    ordered_ids = [
        str(row["stable_sample_id"])
        for row in list(predictions_by_method.get("full_reference") or [])
        if row.get("stable_sample_id") not in (None, "")
    ]
    paired: List[Dict[str, Any]] = []
    for stable in ordered_ids:
        full_row = indexed["full_reference"][stable]
        row = {
            "stable_sample_id": stable,
            "selected_order": full_row.get("selected_order"),
            "raw_dataset_index": full_row.get("raw_dataset_index"),
            "dataset_provided_id": full_row.get("dataset_provided_id"),
            "reference_text": full_row.get("reference_text"),
            "reference_token_sha256": full_row.get("reference_token_sha256"),
        }
        for method in F2B_METHODS:
            pred = indexed[method][stable]
            prefix = method
            row["{}_prediction_text".format(prefix)] = pred.get("prediction_text")
            row["{}_prediction_token_sha256".format(prefix)] = _prediction_hash(pred)
            row["{}_generated_length".format(prefix)] = _generated_length(pred)
        paired.append(row)
    return paired, validation


def write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_quality_summary(paired_rows: Sequence[Mapping[str, Any]], arm_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    comparisons = []
    for name, method_a, method_b in F2B_COMPARISONS:
        agreements = 0
        lengths_a = []
        lengths_b = []
        for row in paired_rows:
            pred_a = row.get("{}_prediction_token_sha256".format(method_a))
            pred_b = row.get("{}_prediction_token_sha256".format(method_b))
            if pred_a == pred_b:
                agreements += 1
            len_a = row.get("{}_generated_length".format(method_a))
            len_b = row.get("{}_generated_length".format(method_b))
            if len_a is not None:
                lengths_a.append(float(len_a))
            if len_b is not None:
                lengths_b.append(float(len_b))
        count = len(paired_rows)
        comparisons.append(
            {
                "name": name,
                "method_a": method_a,
                "method_b": method_b,
                "paired_sample_count": count,
                "sequence_agreement_count": agreements,
                "sequence_agreement_rate": (agreements / count) if count else None,
                "sequence_disagreement_count": count - agreements,
                "mean_generated_length_a": (sum(lengths_a) / len(lengths_a)) if lengths_a else None,
                "mean_generated_length_b": (sum(lengths_b) / len(lengths_b)) if lengths_b else None,
            }
        )
    return {
        "schema_version": F2B_SCHEMA_VERSION,
        "status": "ok",
        "paired_sample_count": len(paired_rows),
        "arm_metrics": {method: dict(arm_metrics.get(method) or {}) for method in F2B_METHODS},
        "comparisons": comparisons,
        "speed_claim_valid": False,
    }


def validate_taskc1_accounting(accounting: Mapping[str, Any], *, method: str) -> Dict[str, Any]:
    failures: List[str] = []
    counters = accounting.get("counters") if isinstance(accounting, Mapping) else None
    if not isinstance(counters, Mapping):
        counters = {}

    def count(key: str) -> int:
        try:
            return int(counters.get(key, accounting.get(key, 0)))  # type: ignore[arg-type]
        except Exception:
            return 0

    if method == "full_reference":
        for key in (
            "exact_catchup_required_token_layer_units",
            "exact_catchup_executed_token_layer_units",
            "restoration_requested_token_layer_units",
            "restoration_succeeded_token_layer_units",
            "restoration_overwritten_token_layer_units",
            "restoration_failed_token_layer_units",
            "fallback_token_layer_units",
            "exact_catchup_avoided_token_layer_units",
        ):
            if count(key) != 0:
                failures.append("full_reference_{}_nonzero".format(key))
    else:
        required = count("exact_catchup_required_token_layer_units")
        executed = count("exact_catchup_executed_token_layer_units")
        requested = count("restoration_requested_token_layer_units")
        succeeded = count("restoration_succeeded_token_layer_units")
        overwritten = count("restoration_overwritten_token_layer_units")
        if required <= 0:
            failures.append("{}_exact_catchup_required_zero".format(method))
        if executed != required:
            failures.append("{}_executed_required_mismatch".format(method))
        if requested != required:
            failures.append("{}_requested_required_mismatch".format(method))
        if succeeded != required:
            failures.append("{}_succeeded_required_mismatch".format(method))
        if overwritten != required:
            failures.append("{}_overwritten_required_mismatch".format(method))
        if count("restoration_failed_token_layer_units") != 0:
            failures.append("{}_restoration_failed_nonzero".format(method))
        if count("fallback_token_layer_units") != 0:
            failures.append("{}_fallback_nonzero".format(method))
        if count("fallback_event_count") != 0:
            failures.append("{}_fallback_event_count_nonzero".format(method))
        if count("exact_catchup_avoided_token_layer_units") != 0:
            failures.append("{}_exact_catchup_avoided_nonzero".format(method))
        if accounting.get("generation_quality_run_valid") is not True:
            failures.append("{}_generation_quality_run_invalid".format(method))
    if bool(accounting.get("speed_claim_valid", False)):
        failures.append("{}_speed_claim_valid_true".format(method))
    validation = accounting.get("validation")
    if isinstance(validation, Mapping) and validation.get("status") not in (None, "ok"):
        failures.append("{}_accounting_validation_not_ok".format(method))
    return {
        "status": "ok" if not failures else "failed",
        "method": method,
        "failures": failures,
        "speed_claim_valid": bool(accounting.get("speed_claim_valid", False)),
        "expected_speed_claim_valid": False,
    }


def simulate_independent_free_running(
    generators: Mapping[str, Callable[[Sequence[str]], str]],
    *,
    initial_token: str = "<s>",
    max_steps: int,
) -> Dict[str, List[str]]:
    prefixes = {method: [initial_token] for method in generators}
    for _ in range(int(max_steps)):
        for method, generator in generators.items():
            prefixes[method].append(str(generator(tuple(prefixes[method]))))
    return prefixes


def build_paper_statistics_manifest(
    *,
    predictions_by_method: Mapping[str, Path],
    result_manifest_by_method: Mapping[str, Optional[Path]],
    real_population_evaluated: bool,
) -> Dict[str, Any]:
    methods = []
    for method in F2B_METHODS:
        spec: Dict[str, Any] = {"name": method, "predictions": str(predictions_by_method[method])}
        sidecar = result_manifest_by_method.get(method)
        if sidecar is not None:
            spec["result_manifest"] = str(sidecar)
            if Path(sidecar).is_file():
                spec["result_manifest_sha256"] = sha256_file(sidecar)
        methods.append(spec)
    comparisons = [
        {
            "name": name,
            "method_a": method_a,
            "method_b": method_b,
            "metric": "sequence_agreement",
            "sign_convention": "higher_is_better",
        }
        for name, method_a, method_b in F2B_COMPARISONS
    ]
    return {
        "schema_version": F2B_SCHEMA_VERSION,
        "evaluation_mode": "generation_quality",
        "evaluation_semantics": "free_running",
        "provenance_mode": "paper_strict",
        "real_population_evaluated": bool(real_population_evaluated),
        "free_running_stratum_anchor": "method_a",
        "methods": methods,
        "comparisons": comparisons,
    }


def build_result_manifest(
    *,
    output_path: Path,
    run_dir: Path,
    method: str,
    predictions_path: Path,
    common_population_identity: Mapping[str, Any],
    method_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    binding = build_paper_statistics_input_binding(
        producer_name="phase3c_f2b_free_running",
        evaluation_mode="generation_quality",
        evaluation_semantics="free_running",
        data_files={
            "eval_predictions": paper_statistics_data_file_binding(
                predictions_path,
                schema="eval_predictions",
                relative_to=run_dir,
            )
        },
        common_population_identity=common_population_identity,
        method_identities={method: dict(method_identity)},
    )
    payload = {"status": "ok", "paper_statistics_input_binding": binding}
    write_json(output_path, payload)
    return payload


def copy_review_bundle(
    run_dir: Path,
    bundle_dir: Path,
    *,
    max_file_bytes: int = 5_000_000,
    required_relative_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    bundle_dir = Path(bundle_dir)
    specs: List[Dict[str, Any]] = []
    excluded_preview: List[Dict[str, Any]] = []
    allowed_suffixes = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md"}
    required_paths = {
        str(Path(value)).replace("\\", "/") for value in (required_relative_paths or ())
    }
    observed_relative_paths: set[str] = set()
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        rel_text = str(rel).replace("\\", "/")
        if bundle_dir in path.parents or path == bundle_dir:
            continue
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if suffix in LARGE_ARTIFACT_SUFFIXES or suffix not in allowed_suffixes or size > max_file_bytes:
            excluded_preview.append({"path": str(rel), "size_bytes": size, "reason": "not_bounded_review_artifact"})
            continue
        specs.append(
            {
                "label": str(rel).replace("\\", "__").replace("/", "__"),
                "member_path": rel_text,
                "path": path,
                "required": rel_text in required_paths,
            }
        )
        observed_relative_paths.add(rel_text)
    for rel_text in sorted(required_paths - observed_relative_paths):
        specs.append(
            {
                "label": rel_text.replace("/", "__"),
                "member_path": rel_text,
                "path": run_dir / Path(rel_text),
                "required": True,
            }
        )
    result = create_review_bundle_archive(
        bundle_dir=bundle_dir,
        archive_path=bundle_dir.with_suffix(".tar.gz"),
        files=specs,
        bundle_type="f2b_free_running_review_bundle",
        max_file_bytes=max_file_bytes,
        allowed_suffixes=allowed_suffixes,
    )
    result["included_files_preview"] = [row.get("relative_path") for row in (result.get("members") or [])[:100]]
    result["excluded_files_preview"] = excluded_preview[:100] + list(result.get("excluded_files") or [])[:100]
    return result


build_f2b_dataset_population_records = _paper_population.build_dataset_population_records
build_f2b_smoke_subset = _paper_population.build_artifact_heldout_smoke_subset
build_stable_sample_identities = _paper_population.build_stable_sample_identities
load_f2b_split_evidence = _paper_population.load_artifact_split_evidence
load_f2b_smoke_selection_evidence = _paper_population.load_smoke_selection_evidence
resolve_artifact_heldout_population = _paper_population.resolve_artifact_heldout_population
resolve_f2b_population_from_split_evidence = _paper_population.resolve_population_from_split_evidence
stable_sample_ids_sha256 = _paper_population.stable_sample_ids_sha256
stable_sample_ids_to_lf_bytes = _paper_population.stable_sample_ids_to_lf_bytes


__all__ = [
    "ACCEPTED_PHASE3C_ARTIFACT_SHA256",
    "ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256",
    "ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256",
    "APPROXIMATION_METHODS",
    "F2B_ARM_IDENTITY_SCHEMA_VERSION",
    "F2B_CANDIDATE_LAYERS",
    "F2B_CANDIDATE_POLICY_NAME",
    "F2B_COMPARISONS",
    "F2B_EXPECTED_ARTIFACT_FITTING_COUNT",
    "F2B_EXPECTED_DATASET_POPULATION_COUNT",
    "F2B_EXPECTED_HELDOUT_COUNT",
    "F2B_METHODS",
    "F2B_POPULATION_IDENTITY_SCHEMA_VERSION",
    "build_decoding_configuration_identity",
    "build_f2b_dataset_population_records",
    "build_f2b_candidate_policy_identity",
    "build_f2b_method_identity",
    "build_observed_runtime_calm_candidate_policy_identity",
    "build_paper_statistics_manifest",
    "build_quality_summary",
    "build_result_manifest",
    "build_f2b_smoke_subset",
    "build_stable_sample_identities",
    "checkpoint_inventory_identity_from_model_root",
    "copy_review_bundle",
    "emit_f2b_runtime_provenance_sidecars",
    "extract_f2b_arm_identity",
    "load_f2b_split_evidence",
    "load_f2b_smoke_selection_evidence",
    "pair_prediction_rows",
    "read_json",
    "read_jsonl",
    "resolve_artifact_heldout_population",
    "resolve_effective_decoding_configuration",
    "resolve_f2b_population_from_split_evidence",
    "simulate_independent_free_running",
    "stable_sample_ids_sha256",
    "stable_sample_ids_to_lf_bytes",
    "validate_f2b_cross_arm_identities",
    "validate_f2b_candidate_policy_payload",
    "validate_f2b_checkpoint_inventory_payload",
    "validate_f2b_decoding_configuration_payload",
    "validate_f2b_runtime_method_identity_payload",
    "validate_cross_arm_population",
    "validate_taskc1_accounting",
    "write_json",
    "write_jsonl",
    "write_paired_csv",
]
