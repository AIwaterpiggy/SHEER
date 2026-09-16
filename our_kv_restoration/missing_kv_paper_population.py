"""Shared paper-population helpers for missing-KV experiments.

The helpers in this module are deliberately CPU-only.  They resolve the
accepted Phase 3c artifact fitting stable-sample population against the
authoritative SAMSum validation population and derive the artifact-held-out
complement used by paper-facing motivation, F2a, and F2b infrastructure.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .missing_kv_dump_provenance import canonical_json_sha256, sha256_file, stable_sample_id, text_sha256
from .missing_kv_calm_trace import CALM_TRACE_SCHEMA_VERSION


ACCEPTED_PHASE3C_ARTIFACT_SHA256 = "c8ddfa41ed1aa7c80df5a82e0d0065b757c76fec6dfedfd179c838d1db0c9ddb"
ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256 = "d30d98a172f0ce684f2ba5f6e416403e4efefe8f209d294d4ab11e8f1a0d4061"
ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256 = "f91570e1d8e6530f22292ebf3584df3500bf379d3c147b16db058079dc10aaa1"

PAPER_POPULATION_IDENTITY_SCHEMA_VERSION = 1
ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION = 1
ARTIFACT_SPLIT_EVIDENCE_TYPE = "accepted_phase3c_artifact_split_evidence_v1"
EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT = 818
EXPECTED_ARTIFACT_FITTING_COUNT = 409
EXPECTED_ARTIFACT_HELDOUT_COUNT = 409
STABLE_SAMPLE_ID_ALGORITHM_IDENTITY = "missing_kv_dump_provenance.stable_sample_id:v1"
EXPECTED_DATASET_NAME = "knkarthick/samsum"
EXPECTED_DATASET_CONFIG_NAME = None
EXPECTED_DATASET_SPLIT = "validation"
EXPECTED_TEXT_COLUMN = "dialogue"
EXPECTED_SUMMARY_COLUMN = "summary"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
SMOKE_SELECTION_SCHEMA_VERSION = 1
SOURCE_MANIFEST_EVIDENCE_TYPES = {
    "accepted_phase3c_artifact_fitting_source_manifest_v1",
    "accepted_phase3c_artifact_fitting_source_manifest_jsonl_v1",
}
SOURCE_MANIFEST_IMMUTABLE_BINDING_SCHEMA_VERSION = 1
SOURCE_MANIFEST_IMMUTABLE_BINDING_TYPES = {
    "artifact_internal_provenance",
    "accepted_fitting_run_manifest",
    "canonical_approved_source_manifest",
}
SOURCE_MANIFEST_IMMUTABLE_BINDING_EVIDENCE_TYPES = {
    "phase3c_artifact_internal_source_manifest_binding_v1": "artifact_internal_provenance",
    "accepted_phase3c_fitting_run_manifest_v1": "accepted_fitting_run_manifest",
    "accepted_phase3c_source_manifest_binding_v1": "canonical_approved_source_manifest",
}

_PLACEHOLDER_STRINGS = {"", "unknown", "placeholder", "todo", "tbd", "none", "null", "<unset>", "unset"}


def is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_STRINGS
    return False


def as_stable_sample_id(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("stable_sample_id_missing")
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError("stable_sample_id_contains_newline")
    return text


def stable_sample_ids_to_lf_bytes(stable_sample_ids: Sequence[Any], *, expected_count: Optional[int] = None) -> bytes:
    ids = [as_stable_sample_id(value) for value in stable_sample_ids]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError("duplicate_stable_sample_id:{}".format(duplicates[0]))
    if expected_count is not None and len(ids) != int(expected_count):
        raise ValueError("stable_sample_id_count_mismatch:{}:{}".format(len(ids), int(expected_count)))
    return ("".join("{}\n".format(item) for item in ids)).encode("utf-8")


def stable_sample_ids_sha256(stable_sample_ids: Sequence[Any], *, sort_ids: bool, expected_count: Optional[int] = None) -> str:
    ids = [as_stable_sample_id(value) for value in stable_sample_ids]
    if sort_ids:
        ids = sorted(ids)
    return hashlib.sha256(stable_sample_ids_to_lf_bytes(ids, expected_count=expected_count)).hexdigest()


def build_stable_sample_identities(
    stable_sample_ids: Sequence[Any],
    *,
    expected_count: Optional[int] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    ids = [as_stable_sample_id(value) for value in stable_sample_ids]
    return {
        "{}stable_sample_count".format(prefix): len(ids),
        "{}stable_sample_set_sha256".format(prefix): stable_sample_ids_sha256(ids, sort_ids=True, expected_count=expected_count),
        "{}stable_sample_ordered_sha256".format(prefix): stable_sample_ids_sha256(ids, sort_ids=False, expected_count=expected_count),
        "{}stable_sample_serialization_encoding".format(prefix): "utf-8",
        "{}stable_sample_serialization_newline".format(prefix): "lf",
        "{}stable_sample_serialization_final_newline".format(prefix): True,
        "{}stable_sample_set_sort".format(prefix): "lexicographic",
    }


def row_stable_sample_ids(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return [as_stable_sample_id(row.get("stable_sample_id")) for row in rows]


def duplicate_count(values: Sequence[str]) -> int:
    return sum(max(0, count - 1) for count in Counter(values).values())


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return payload


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _list_from_payload(payload: Mapping[str, Any], keys: Sequence[str]) -> Optional[List[Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return list(value)
    return None


def _resolve_manifest_path(base_path: Path, value: Any) -> Optional[Path]:
    if is_placeholder_value(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_path.parent / path
    return path


def _metadata_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _canonical_dataset_config_name(value: Any) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _explicit_value(payload: Mapping[str, Any], *keys: str) -> tuple[bool, Any]:
    for key in keys:
        if key in payload:
            return True, payload.get(key)
    return False, None


def _as_exact_schema_version(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected: int,
    missing_failure: str,
    mismatch_failure: str,
    failures: List[str],
) -> Optional[int]:
    if key not in payload:
        failures.append(missing_failure)
        return None
    value = payload.get(key)
    if isinstance(value, bool) or value != expected:
        failures.append(mismatch_failure)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    return int(value)


def _source_binding_reference_sha(payload: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "source_manifest_sha256",
        "expected_source_manifest_sha256",
        "source_manifest_expected_sha256",
        "artifact_fitting_source_manifest_sha256",
    ):
        value = payload.get(key)
        if not is_placeholder_value(value):
            return str(value)
    return None


def _validate_source_manifest_immutable_binding(
    split_payload: Mapping[str, Any],
    split_path: Path,
    *,
    source_manifest_sha256: Optional[str],
    expected_artifact_sha256: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    binding_type = split_payload.get("source_manifest_immutable_binding_type")
    reference_value = (
        split_payload.get("source_manifest_immutable_binding_reference_path")
        or split_payload.get("source_manifest_immutable_binding_reference")
    )
    expected_reference_sha = split_payload.get("source_manifest_immutable_binding_reference_sha256")
    expected_source_sha = split_payload.get("source_manifest_expected_sha256")

    if is_placeholder_value(binding_type) or is_placeholder_value(reference_value):
        failures.append("source_manifest_immutable_binding_missing")
    elif str(binding_type) not in SOURCE_MANIFEST_IMMUTABLE_BINDING_TYPES:
        failures.append("source_manifest_immutable_binding_type_mismatch")

    if is_placeholder_value(expected_source_sha):
        failures.append("source_manifest_expected_sha256_missing")
    elif source_manifest_sha256 is not None and str(expected_source_sha) != str(source_manifest_sha256):
        failures.append("source_manifest_expected_sha256_mismatch")

    reference_path = _resolve_manifest_path(split_path, reference_value)
    reference_sha = None
    reference_payload: Dict[str, Any] = {}
    if reference_path is not None:
        if reference_path.is_file():
            reference_sha = sha256_file(reference_path)
            if not is_placeholder_value(expected_reference_sha) and str(expected_reference_sha) != reference_sha:
                failures.append("source_manifest_immutable_binding_reference_sha256_mismatch")
            try:
                reference_payload = read_json(reference_path)
            except Exception as exc:
                failures.append("source_manifest_immutable_binding_reference_parse_failed:{}".format(type(exc).__name__))
        else:
            failures.append("source_manifest_immutable_binding_reference_missing")

    if reference_payload:
        _as_exact_schema_version(
            reference_payload,
            key="schema_version",
            expected=SOURCE_MANIFEST_IMMUTABLE_BINDING_SCHEMA_VERSION,
            missing_failure="source_manifest_immutable_binding_schema_version_missing",
            mismatch_failure="source_manifest_immutable_binding_schema_version_mismatch",
            failures=failures,
        )
        reference_evidence_type = reference_payload.get("evidence_type")
        expected_binding_type = SOURCE_MANIFEST_IMMUTABLE_BINDING_EVIDENCE_TYPES.get(str(reference_evidence_type))
        if expected_binding_type is None:
            failures.append("source_manifest_immutable_binding_evidence_type_mismatch")
        elif not is_placeholder_value(binding_type) and expected_binding_type != str(binding_type):
            failures.append("source_manifest_immutable_binding_type_mismatch")
        reference_artifact_sha = (
            reference_payload.get("accepted_artifact_file_sha256")
            or reference_payload.get("artifact_file_sha256")
        )
        if reference_artifact_sha != expected_artifact_sha256:
            failures.append("source_manifest_immutable_binding_artifact_sha256_mismatch")
        reference_source_sha = _source_binding_reference_sha(reference_payload)
        if reference_source_sha is None:
            failures.append("source_manifest_immutable_binding_source_sha256_missing")
        elif source_manifest_sha256 is not None and reference_source_sha != source_manifest_sha256:
            failures.append("source_manifest_immutable_binding_source_sha256_mismatch")

    return {
        "source_manifest_immutable_binding_type": None if is_placeholder_value(binding_type) else str(binding_type),
        "source_manifest_immutable_binding_valid": not failures,
        "source_manifest_immutable_binding_reference": None if reference_path is None else str(reference_path),
        "source_manifest_immutable_binding_reference_sha256": reference_sha or expected_reference_sha,
        "source_manifest_expected_sha256": None if is_placeholder_value(expected_source_sha) else str(expected_source_sha),
        "source_manifest_actual_sha256": source_manifest_sha256,
        "failures": failures,
    }


def _source_manifest_payload(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        metadata: Dict[str, Any] = {}
        records: List[Dict[str, Any]] = []
        for row in rows:
            role = str(row.get("record_type") or row.get("manifest_record_type") or "").lower()
            if role in {"source_manifest_metadata", "metadata"}:
                metadata.update(row)
            else:
                records.append(row)
        return {
            "schema_version": metadata.get("schema_version") or metadata.get("manifest_schema_version"),
            "evidence_type": metadata.get("evidence_type"),
            "metadata": metadata,
            "records": records,
        }
    payload = read_json(path)
    records = payload.get("records")
    if records is None:
        records = payload.get("artifact_fitting_records")
    if records is not None and not isinstance(records, list):
        raise ValueError("source_manifest_records_not_list")
    return {
        "schema_version": payload.get("schema_version") or payload.get("manifest_schema_version"),
        "evidence_type": payload.get("evidence_type") or payload.get("manifest_type"),
        "metadata": payload,
        "records": list(records or []),
        "artifact_fitting_stable_sample_ids": _list_from_payload(
            payload,
            ("artifact_fitting_stable_sample_ids", "fitting_stable_sample_ids", "train_stable_sample_ids"),
        ),
    }


def _source_manifest_row_is_fitting(row: Mapping[str, Any], *, default_all_rows_are_fitting: bool) -> bool:
    for key in ("population_role", "split_role", "role", "artifact_population", "fit_split", "split_name"):
        if key not in row:
            continue
        value = str(row.get(key)).strip().lower()
        if value in {"artifact_fitting", "fitting", "fit", "train", "training", "calibration_train"}:
            return True
        if value in {"heldout", "held_out", "eval", "evaluation", "validation_heldout"}:
            return False
    for key in ("is_artifact_fitting", "is_fitting", "is_train"):
        if key in row:
            return bool(row.get(key))
    return bool(default_all_rows_are_fitting)


def load_source_manifest_fitting_ids(
    path: Path | str,
    *,
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    expected_fitting_count: int = EXPECTED_ARTIFACT_FITTING_COUNT,
    immutable_binding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    target = Path(path)
    failures: List[str] = []
    if not target.is_file():
        return {
            "status": "failed",
            "failures": ["source_manifest_missing"],
            "source_manifest_path": str(target),
        }
    try:
        payload = _source_manifest_payload(target)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": ["source_manifest_parse_failed:{}".format(type(exc).__name__)],
            "source_manifest_path": str(target),
            "source_manifest_sha256": sha256_file(target) if target.is_file() else None,
        }

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    evidence_type = str(payload.get("evidence_type") or "")
    if "schema_version" not in payload or payload.get("schema_version") is None:
        schema_version = None
        failures.append("source_manifest_schema_version_missing")
    elif isinstance(payload.get("schema_version"), bool) or payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        schema_version = payload.get("schema_version") if isinstance(payload.get("schema_version"), int) and not isinstance(payload.get("schema_version"), bool) else None
        failures.append("source_manifest_schema_version_mismatch")
    else:
        schema_version = SOURCE_MANIFEST_SCHEMA_VERSION
    if is_placeholder_value(evidence_type) or evidence_type not in SOURCE_MANIFEST_EVIDENCE_TYPES:
        failures.append("source_manifest_schema_unrecognized")

    artifact_sha = _metadata_value(metadata, "accepted_artifact_file_sha256", "artifact_file_sha256")
    runtime_policy_sha = _metadata_value(metadata, "runtime_policy_sha256", "accepted_runtime_policy_sha256")
    hybrid_policy_sha = _metadata_value(
        metadata,
        "hybrid_fitting_policy_sha256",
        "accepted_hybrid_fitting_policy_sha256",
        "calm_hybrid_policy_sha256",
    )
    artifact_bound = artifact_sha == expected_artifact_sha256
    hybrid_bound = hybrid_policy_sha == ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256
    runtime_bound = runtime_policy_sha == ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256
    if not artifact_bound and not hybrid_bound:
        failures.append("source_manifest_artifact_or_policy_binding_missing")
    if runtime_policy_sha is None:
        failures.append("source_manifest_runtime_policy_sha256_missing")
    elif runtime_policy_sha != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
        failures.append("source_manifest_runtime_policy_sha256_mismatch")
    if hybrid_policy_sha is not None and hybrid_policy_sha != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
        failures.append("source_manifest_hybrid_fitting_policy_sha256_mismatch")

    algorithm = _metadata_value(metadata, "stable_sample_id_algorithm_identity", "stable_sample_id_algorithm")
    if algorithm != STABLE_SAMPLE_ID_ALGORITHM_IDENTITY:
        failures.append("source_manifest_stable_sample_id_algorithm_mismatch")

    dataset_name_present, dataset_name = _explicit_value(metadata, "dataset_name")
    dataset_config_present, dataset_config_value = _explicit_value(metadata, "dataset_config_name", "dataset_config")
    dataset_config_name = _canonical_dataset_config_name(dataset_config_value)
    dataset_split_present, dataset_split = _explicit_value(metadata, "dataset_split", "split")
    text_column_present, text_column = _explicit_value(metadata, "text_column")
    summary_column_present, summary_column = _explicit_value(metadata, "summary_column")
    if not dataset_name_present:
        failures.append("source_manifest_dataset_name_missing")
    elif dataset_name != EXPECTED_DATASET_NAME:
        failures.append("source_manifest_dataset_name_mismatch")
    if not dataset_config_present:
        failures.append("source_manifest_dataset_config_name_missing")
    elif dataset_config_name != EXPECTED_DATASET_CONFIG_NAME:
        failures.append("source_manifest_dataset_config_name_mismatch")
    if not dataset_split_present:
        failures.append("source_manifest_dataset_split_missing")
    elif dataset_split != EXPECTED_DATASET_SPLIT:
        failures.append("source_manifest_dataset_split_mismatch")
    if not text_column_present:
        failures.append("source_manifest_text_column_missing")
    elif text_column != EXPECTED_TEXT_COLUMN:
        failures.append("source_manifest_text_column_mismatch")
    if not summary_column_present:
        failures.append("source_manifest_summary_column_missing")
    elif summary_column != EXPECTED_SUMMARY_COLUMN:
        failures.append("source_manifest_summary_column_mismatch")

    fitting_values = payload.get("artifact_fitting_stable_sample_ids")
    if fitting_values is not None:
        raw_ids = list(fitting_values)
    else:
        records = payload.get("records")
        if not isinstance(records, list):
            records = []
        default_all_rows_are_fitting = str(metadata.get("population_role") or metadata.get("manifest_population_role") or "").strip().lower() in {
            "artifact_fitting",
            "fitting",
            "train",
            "training",
        }
        raw_ids = []
        for index, row in enumerate(records):
            if not isinstance(row, Mapping):
                failures.append("source_manifest_record_not_mapping:{}".format(index))
                continue
            if _source_manifest_row_is_fitting(row, default_all_rows_are_fitting=default_all_rows_are_fitting):
                raw_ids.append(row.get("stable_sample_id"))
    if not raw_ids:
        failures.append("source_manifest_fitting_stable_sample_ids_missing")
        fitting_ids: List[str] = []
    else:
        try:
            fitting_ids = [as_stable_sample_id(value) for value in raw_ids]
        except Exception as exc:
            failures.append("source_manifest_fitting_stable_sample_ids_invalid:{}".format(type(exc).__name__))
            fitting_ids = []

    duplicates = sorted(item for item, count in Counter(fitting_ids).items() if count > 1)
    if duplicates:
        failures.append("source_manifest_fitting_duplicate")
    if len(fitting_ids) != int(expected_fitting_count):
        failures.append("source_manifest_fitting_count_mismatch:{}:{}".format(len(fitting_ids), int(expected_fitting_count)))

    binding_fields = dict(immutable_binding or {})
    identity: Dict[str, Any] = {}
    if fitting_ids and not duplicates:
        try:
            fitting_identity = build_stable_sample_identities(
                fitting_ids,
                expected_count=expected_fitting_count,
                prefix="artifact_fitting_",
            )
            identity = {
                "source_manifest_schema_version": schema_version,
                "source_manifest_evidence_type": evidence_type,
                "source_manifest_sha256": sha256_file(target),
                "source_manifest_accepted_artifact_binding": bool(artifact_bound),
                "source_manifest_hybrid_fitting_policy_binding": bool(hybrid_bound),
                "source_manifest_runtime_policy_binding": bool(runtime_bound),
                "source_manifest_immutable_binding_type": binding_fields.get("source_manifest_immutable_binding_type"),
                "source_manifest_immutable_binding_valid": bool(binding_fields.get("source_manifest_immutable_binding_valid")),
                "source_manifest_immutable_binding_reference": binding_fields.get("source_manifest_immutable_binding_reference"),
                "source_manifest_immutable_binding_reference_sha256": binding_fields.get("source_manifest_immutable_binding_reference_sha256"),
                "source_manifest_expected_sha256": binding_fields.get("source_manifest_expected_sha256"),
                "source_manifest_actual_sha256": binding_fields.get("source_manifest_actual_sha256") or sha256_file(target),
                "accepted_artifact_file_sha256": artifact_sha,
                "runtime_policy_sha256": runtime_policy_sha,
                "hybrid_fitting_policy_sha256": hybrid_policy_sha,
                "stable_sample_id_algorithm_identity": algorithm,
                "dataset_name": dataset_name,
                "dataset_config_name": dataset_config_name,
                "dataset_split": dataset_split,
                "text_column": text_column,
                "summary_column": summary_column,
                **fitting_identity,
            }
            identity["source_manifest_identity"] = canonical_json_sha256(identity)
        except Exception as exc:
            failures.append("source_manifest_identity_failed:{}".format(type(exc).__name__))
            identity = {}

    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "source_manifest_path": str(target),
        "source_manifest_sha256": sha256_file(target),
        "source_manifest_schema_version": schema_version,
        "source_manifest_evidence_type": evidence_type,
        "artifact_fitting_stable_sample_ids": fitting_ids,
        "source_manifest_identity_payload": identity,
        "source_manifest_identity": identity.get("source_manifest_identity"),
        "source_manifest_accepted_artifact_binding": bool(artifact_bound),
        "source_manifest_hybrid_fitting_policy_binding": bool(hybrid_bound),
        "source_manifest_runtime_policy_binding": bool(runtime_bound),
        "source_manifest_immutable_binding_type": binding_fields.get("source_manifest_immutable_binding_type"),
        "source_manifest_immutable_binding_valid": bool(binding_fields.get("source_manifest_immutable_binding_valid")),
        "source_manifest_immutable_binding_reference": binding_fields.get("source_manifest_immutable_binding_reference"),
        "source_manifest_immutable_binding_reference_sha256": binding_fields.get("source_manifest_immutable_binding_reference_sha256"),
        "source_manifest_expected_sha256": binding_fields.get("source_manifest_expected_sha256"),
        "source_manifest_actual_sha256": binding_fields.get("source_manifest_actual_sha256") or sha256_file(target),
    }


def load_artifact_split_evidence(
    path: Path | str,
    *,
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    failure_prefix: str = "artifact",
) -> Dict[str, Any]:
    target = Path(path)
    failures: List[str] = []
    if not str(path):
        return {
            "schema_version": ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "failures": ["{}_split_evidence_path_missing".format(failure_prefix)],
            "split_evidence_path": str(path),
        }
    if not target.is_file():
        return {
            "schema_version": ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "failures": ["{}_split_evidence_missing".format(failure_prefix)],
            "split_evidence_path": str(target),
        }
    try:
        payload = read_json(target)
    except Exception as exc:
        return {
            "schema_version": ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION,
            "status": "failed",
            "failures": ["{}_split_evidence_parse_failed:{}".format(failure_prefix, type(exc).__name__)],
            "split_evidence_path": str(target),
            "split_evidence_sha256": sha256_file(target) if target.is_file() else None,
        }

    split_schema_version = _as_exact_schema_version(
        payload,
        key="schema_version",
        expected=ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION,
        missing_failure="artifact_split_evidence_schema_version_missing",
        mismatch_failure="artifact_split_evidence_schema_version_mismatch",
        failures=failures,
    )
    split_evidence_type = payload.get("evidence_type")
    if is_placeholder_value(split_evidence_type):
        failures.append("artifact_split_evidence_type_missing")
    elif split_evidence_type != ARTIFACT_SPLIT_EVIDENCE_TYPE:
        failures.append("artifact_split_evidence_type_mismatch")

    artifact_sha = payload.get("accepted_artifact_file_sha256") or payload.get("artifact_file_sha256")
    if artifact_sha != expected_artifact_sha256:
        failures.append("accepted_artifact_file_sha256_mismatch")
    runtime_policy_sha = payload.get("runtime_policy_sha256") or payload.get("accepted_runtime_policy_sha256")
    if runtime_policy_sha != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
        failures.append("runtime_policy_sha256_mismatch")
    hybrid_policy_sha = payload.get("hybrid_fitting_policy_sha256") or payload.get("accepted_hybrid_fitting_policy_sha256")
    if hybrid_policy_sha != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
        failures.append("hybrid_fitting_policy_sha256_mismatch")
    algorithm = payload.get("stable_sample_id_algorithm_identity") or payload.get("stable_sample_id_algorithm")
    if algorithm != STABLE_SAMPLE_ID_ALGORITHM_IDENTITY:
        failures.append("stable_sample_id_algorithm_identity_mismatch")

    if payload.get("dataset_name") != EXPECTED_DATASET_NAME:
        failures.append("dataset_name_mismatch")
    if _canonical_dataset_config_name(payload.get("dataset_config_name")) != EXPECTED_DATASET_CONFIG_NAME:
        failures.append("dataset_config_name_mismatch")
    if payload.get("dataset_split") != EXPECTED_DATASET_SPLIT:
        failures.append("dataset_split_mismatch")
    text_column = payload.get("text_column", EXPECTED_TEXT_COLUMN)
    summary_column = payload.get("summary_column", EXPECTED_SUMMARY_COLUMN)
    if text_column != EXPECTED_TEXT_COLUMN:
        failures.append("text_column_mismatch")
    if summary_column != EXPECTED_SUMMARY_COLUMN:
        failures.append("summary_column_mismatch")

    wrapper_fitting_ids: Optional[List[str]] = None
    fitting_values = _list_from_payload(
        payload,
        ("artifact_fitting_stable_sample_ids", "fitting_stable_sample_ids", "train_stable_sample_ids"),
    )
    if fitting_values is not None:
        try:
            wrapper_fitting_ids = [as_stable_sample_id(value) for value in fitting_values]
        except Exception as exc:
            failures.append("artifact_fitting_stable_sample_ids_invalid:{}".format(type(exc).__name__))

    supplied_heldout_ids: Optional[List[str]] = None
    heldout_values = _list_from_payload(payload, ("heldout_stable_sample_ids", "artifact_heldout_stable_sample_ids"))
    if heldout_values is not None:
        try:
            supplied_heldout_ids = [as_stable_sample_id(value) for value in heldout_values]
        except Exception as exc:
            failures.append("artifact_heldout_stable_sample_ids_invalid:{}".format(type(exc).__name__))

    source_manifest_path = _resolve_manifest_path(target, payload.get("source_manifest_path"))
    source_manifest_sha256 = payload.get("source_manifest_sha256")
    computed_source_manifest_sha256 = None
    immutable_binding_result: Dict[str, Any] = {}
    source_manifest_result: Dict[str, Any] = {}
    fitting_ids: List[str] = []
    if source_manifest_path is not None:
        if source_manifest_path.is_file():
            computed_source_manifest_sha256 = sha256_file(source_manifest_path)
            if source_manifest_sha256 and computed_source_manifest_sha256 != source_manifest_sha256:
                failures.append("source_manifest_sha256_mismatch")
            immutable_binding_result = _validate_source_manifest_immutable_binding(
                payload,
                target,
                source_manifest_sha256=computed_source_manifest_sha256,
                expected_artifact_sha256=expected_artifact_sha256,
            )
            failures.extend(immutable_binding_result.get("failures") or [])
            source_manifest_result = load_source_manifest_fitting_ids(
                source_manifest_path,
                expected_artifact_sha256=expected_artifact_sha256,
                expected_fitting_count=EXPECTED_ARTIFACT_FITTING_COUNT,
                immutable_binding=immutable_binding_result,
            )
            failures.extend(source_manifest_result.get("failures") or [])
            fitting_ids = [str(item) for item in source_manifest_result.get("artifact_fitting_stable_sample_ids") or []]
        else:
            failures.append("source_manifest_missing")
    else:
        failures.append("source_manifest_file_required")

    if wrapper_fitting_ids is not None and fitting_ids:
        if set(wrapper_fitting_ids) != set(fitting_ids) or wrapper_fitting_ids != fitting_ids:
            failures.append("source_manifest_fitting_id_mismatch")

    split_evidence_sha256 = sha256_file(target)
    source_identity_payload = source_manifest_result.get("source_manifest_identity_payload") or {}
    split_source = {
        "source": str(split_evidence_type or "artifact_split_evidence"),
        "split_evidence_schema_version": split_schema_version,
        "split_evidence_type": split_evidence_type,
        "split_evidence_path": str(target),
        "split_evidence_sha256": split_evidence_sha256,
        "accepted_artifact_file_sha256": artifact_sha,
        "runtime_policy_sha256": runtime_policy_sha,
        "hybrid_fitting_policy_sha256": hybrid_policy_sha,
        "source_manifest_path": None if source_manifest_path is None else str(source_manifest_path),
        "source_manifest_sha256": computed_source_manifest_sha256 or source_manifest_sha256,
        "source_manifest_schema_version": source_manifest_result.get("source_manifest_schema_version"),
        "source_manifest_identity": source_manifest_result.get("source_manifest_identity"),
        "source_manifest_accepted_artifact_binding": source_manifest_result.get("source_manifest_accepted_artifact_binding"),
        "source_manifest_hybrid_fitting_policy_binding": source_manifest_result.get("source_manifest_hybrid_fitting_policy_binding"),
        "source_manifest_runtime_policy_binding": source_manifest_result.get("source_manifest_runtime_policy_binding"),
        "source_manifest_immutable_binding_type": immutable_binding_result.get("source_manifest_immutable_binding_type"),
        "source_manifest_immutable_binding_valid": immutable_binding_result.get("source_manifest_immutable_binding_valid"),
        "source_manifest_immutable_binding_reference": immutable_binding_result.get("source_manifest_immutable_binding_reference"),
        "source_manifest_immutable_binding_reference_sha256": immutable_binding_result.get("source_manifest_immutable_binding_reference_sha256"),
        "source_manifest_expected_sha256": immutable_binding_result.get("source_manifest_expected_sha256"),
        "source_manifest_actual_sha256": immutable_binding_result.get("source_manifest_actual_sha256"),
        "stable_sample_id_algorithm_identity": algorithm,
        "stable_sample_id_algorithm": algorithm,
        "dataset_name": payload.get("dataset_name"),
        "dataset_config_name": _canonical_dataset_config_name(payload.get("dataset_config_name")),
        "dataset_split": payload.get("dataset_split"),
        "text_column": text_column,
        "summary_column": summary_column,
        "artifact_fitting_stable_sample_count": source_identity_payload.get("artifact_fitting_stable_sample_count"),
        "artifact_fitting_stable_sample_set_sha256": source_identity_payload.get("artifact_fitting_stable_sample_set_sha256"),
        "artifact_fitting_stable_sample_ordered_sha256": source_identity_payload.get("artifact_fitting_stable_sample_ordered_sha256"),
    }
    return {
        "schema_version": split_schema_version,
        "split_evidence_schema_version": split_schema_version,
        "split_evidence_type": split_evidence_type,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "split_evidence_path": str(target),
        "split_evidence_sha256": split_evidence_sha256,
        "split_source": split_source,
        "artifact_fitting_stable_sample_ids": fitting_ids,
        "supplied_heldout_stable_sample_ids": supplied_heldout_ids,
        "dataset_name": payload.get("dataset_name"),
        "dataset_config_name": payload.get("dataset_config_name"),
        "dataset_split": payload.get("dataset_split"),
    }


def build_dataset_population_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_name: Optional[str] = "knkarthick/samsum",
    dataset_config_name: Optional[str] = None,
    dataset_split: str = "validation",
    text_column: str = "dialogue",
    summary_column: str = "summary",
    allow_precomputed_stable_sample_id: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    dataset_config_name = _canonical_dataset_config_name(dataset_config_name)
    for raw_index, row in enumerate(rows):
        source_text = row.get(text_column)
        reference_text = row.get(summary_column)
        dataset_provided_id = None
        for key in ("id", "sample_id", "dialogue_id", "guid", "dataset_provided_id"):
            if key in row:
                dataset_provided_id = row.get(key)
                break
        if source_text is not None and reference_text is not None and source_text != "" and reference_text != "":
            source_digest = text_sha256(source_text)
            reference_digest = text_sha256(reference_text)
            record = {
                "stable_sample_id": stable_sample_id(
                    dataset_name=dataset_name,
                    dataset_config_name=dataset_config_name,
                    split=dataset_split,
                    raw_dataset_index=int(row.get("raw_dataset_index", raw_index)),
                    dataset_provided_id=dataset_provided_id,
                    source_text_sha256=source_digest,
                    reference_text_sha256=reference_digest,
                ),
                "raw_dataset_index": int(row.get("raw_dataset_index", raw_index)),
                "dataset_provided_id": None if dataset_provided_id is None else str(dataset_provided_id),
                "source_text_sha256": source_digest,
                "reference_text_sha256": reference_digest,
                "dataset_name": dataset_name,
                "dataset_config_name": dataset_config_name,
                "split": dataset_split,
                "text_column": text_column,
                "summary_column": summary_column,
                "stable_sample_id_algorithm_identity": STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
            }
            for key in (
                "selected_order",
                "tokenized_input_sha256",
                "tokenized_label_sha256",
                "input_token_count",
                "label_token_count",
            ):
                if key in row:
                    record[key] = row[key]
            records.append(record)
            continue
        if "stable_sample_id" in row and allow_precomputed_stable_sample_id:
            records.append(
                {
                    "stable_sample_id": as_stable_sample_id(row.get("stable_sample_id")),
                    "raw_dataset_index": int(row.get("raw_dataset_index", raw_index)),
                    "dataset_provided_id": None
                    if dataset_provided_id is None
                    else str(dataset_provided_id),
                    "dataset_name": row.get("dataset_name", dataset_name),
                    "dataset_config_name": _canonical_dataset_config_name(row.get("dataset_config_name", dataset_config_name)),
                    "split": row.get("split", dataset_split),
                    "text_column": row.get("text_column", text_column),
                    "summary_column": row.get("summary_column", summary_column),
                    "stable_sample_id_algorithm_identity": STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
                }
            )
            continue
        raise ValueError("dataset_row_text_or_summary_missing:{}".format(raw_index))
    return records


def resolve_artifact_heldout_population(
    dataset_rows: Sequence[Mapping[str, Any]],
    fitting_stable_sample_ids: Sequence[Any],
    *,
    split_source: Optional[Mapping[str, Any]],
    expected_dataset_count: int = EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
    expected_fitting_count: int = EXPECTED_ARTIFACT_FITTING_COUNT,
    expected_heldout_count: int = EXPECTED_ARTIFACT_HELDOUT_COUNT,
) -> Dict[str, Any]:
    failures: List[str] = []
    if not isinstance(split_source, Mapping) or is_placeholder_value(split_source.get("source")):
        failures.append("authoritative_split_source_required")
    split_source = dict(split_source or {})
    if split_source.get("accepted_artifact_file_sha256") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
        failures.append("split_source_accepted_artifact_sha256_mismatch")
    if split_source.get("runtime_policy_sha256") != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
        failures.append("split_source_runtime_policy_sha256_mismatch")
    if split_source.get("hybrid_fitting_policy_sha256") != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
        failures.append("split_source_hybrid_fitting_policy_sha256_mismatch")
    if split_source.get("stable_sample_id_algorithm_identity") != STABLE_SAMPLE_ID_ALGORITHM_IDENTITY:
        failures.append("split_source_stable_sample_id_algorithm_mismatch")
    if split_source.get("dataset_name") not in (None, EXPECTED_DATASET_NAME):
        failures.append("split_source_dataset_name_mismatch")
    if _canonical_dataset_config_name(split_source.get("dataset_config_name")) != EXPECTED_DATASET_CONFIG_NAME:
        failures.append("split_source_dataset_config_name_mismatch")
    if split_source.get("dataset_split") not in (None, EXPECTED_DATASET_SPLIT):
        failures.append("split_source_dataset_split_mismatch")
    if split_source.get("text_column") not in (None, EXPECTED_TEXT_COLUMN):
        failures.append("split_source_text_column_mismatch")
    if split_source.get("summary_column") not in (None, EXPECTED_SUMMARY_COLUMN):
        failures.append("split_source_summary_column_mismatch")
    try:
        dataset_ids = row_stable_sample_ids(dataset_rows)
    except Exception as exc:
        failures.append("dataset_stable_sample_id_invalid:{}".format(type(exc).__name__))
        dataset_ids = []
    try:
        fitting_ids = [as_stable_sample_id(value) for value in fitting_stable_sample_ids]
    except Exception as exc:
        failures.append("fitting_stable_sample_id_invalid:{}".format(type(exc).__name__))
        fitting_ids = []

    duplicate_dataset_count = duplicate_count(dataset_ids)
    duplicate_fitting_count = duplicate_count(fitting_ids)
    if duplicate_dataset_count:
        failures.append("duplicate_dataset_stable_id")
    if duplicate_fitting_count:
        failures.append("duplicate_fitting_id")
    if len(dataset_ids) != int(expected_dataset_count):
        failures.append("dataset_population_count_mismatch:{}:{}".format(len(dataset_ids), int(expected_dataset_count)))
    if len(fitting_ids) != int(expected_fitting_count):
        failures.append("fitting_count_mismatch:{}:{}".format(len(fitting_ids), int(expected_fitting_count)))
    for index, row in enumerate(dataset_rows):
        if row.get("dataset_name") != EXPECTED_DATASET_NAME:
            failures.append("dataset_row_dataset_name_mismatch:{}".format(index))
            break
        if _canonical_dataset_config_name(row.get("dataset_config_name")) != EXPECTED_DATASET_CONFIG_NAME:
            failures.append("dataset_row_dataset_config_name_mismatch:{}".format(index))
            break
        if row.get("split") != EXPECTED_DATASET_SPLIT:
            failures.append("dataset_row_split_mismatch:{}".format(index))
            break
        if row.get("text_column") not in (None, EXPECTED_TEXT_COLUMN):
            failures.append("dataset_row_text_column_mismatch:{}".format(index))
            break
        if row.get("summary_column") not in (None, EXPECTED_SUMMARY_COLUMN):
            failures.append("dataset_row_summary_column_mismatch:{}".format(index))
            break
        if row.get("stable_sample_id_algorithm_identity") not in (None, STABLE_SAMPLE_ID_ALGORITHM_IDENTITY):
            failures.append("dataset_row_stable_sample_id_algorithm_mismatch:{}".format(index))
            break

    dataset_set = set(dataset_ids)
    fitting_set = set(fitting_ids)
    fitting_missing = sorted(fitting_set - dataset_set)
    if fitting_missing:
        failures.append("fitting_id_missing_from_dataset")
    heldout_rows = [dict(row) for row in dataset_rows if str(row.get("stable_sample_id")) not in fitting_set]
    heldout_ids = row_stable_sample_ids(heldout_rows) if dataset_ids else []
    duplicate_heldout_count = duplicate_count(heldout_ids)
    if duplicate_heldout_count:
        failures.append("duplicate_heldout_id")
    if len(heldout_ids) != int(expected_heldout_count):
        failures.append("heldout_count_mismatch:{}:{}".format(len(heldout_ids), int(expected_heldout_count)))
    fitting_heldout_intersection = fitting_set & set(heldout_ids)
    union = fitting_set | set(heldout_ids)
    if fitting_heldout_intersection:
        failures.append("fitting_heldout_overlap")
    if len(union) != int(expected_dataset_count):
        failures.append("fitting_heldout_union_count_mismatch:{}:{}".format(len(union), int(expected_dataset_count)))
    if sorted(union) != sorted(dataset_set):
        failures.append("fitting_heldout_union_dataset_mismatch")

    identity: Dict[str, Any] = {}
    if not duplicate_heldout_count and len(heldout_ids) == int(expected_heldout_count):
        identity = build_stable_sample_identities(heldout_ids, expected_count=expected_heldout_count, prefix="heldout_")
    fitting_identity: Dict[str, Any] = {}
    if not duplicate_fitting_count and len(fitting_ids) == int(expected_fitting_count):
        fitting_identity = build_stable_sample_identities(
            fitting_ids,
            expected_count=expected_fitting_count,
            prefix="artifact_fitting_",
        )

    return {
        "schema_version": PAPER_POPULATION_IDENTITY_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "failure_stage": None if not failures else "heldout_population_resolution",
        "failures": failures,
        "population_mode": "artifact_heldout",
        "split_source": split_source,
        "dataset_population_count": len(dataset_ids),
        "fitting_count": len(fitting_ids),
        "heldout_count": len(heldout_ids),
        "intersection_count": len(fitting_heldout_intersection),
        "union_count": len(union),
        "duplicate_dataset_stable_id_count": duplicate_dataset_count,
        "duplicate_fitting_id_count": duplicate_fitting_count,
        "duplicate_heldout_id_count": duplicate_heldout_count,
        "fitting_id_missing_from_dataset_count": len(fitting_missing),
        "heldout_id_missing_from_dataset_count": len(set(heldout_ids) - dataset_set),
        "heldout_identity": identity,
        "heldout_records": heldout_rows,
        "dataset_only_intersection_count": len(dataset_set & fitting_set & set(heldout_ids)),
        **fitting_identity,
        **identity,
    }


def _stable_sample_id_from_trace_row(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("stable_sample_id", "sample_stable_sample_id"):
        value = row.get(key)
        if value not in (None, ""):
            return as_stable_sample_id(value)
    for key in ("sample_identity", "generation_sample_identity"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            value = nested.get("stable_sample_id")
            if value not in (None, ""):
                return as_stable_sample_id(value)
    return None


def _trace_row_has_candidate_first_crossing(row: Mapping[str, Any]) -> bool:
    event_type = str(row.get("event_type") or row.get("record_type") or "")
    transaction_status = row.get("transaction_status")
    runtime_path = row.get("runtime_path")
    if event_type == "kv_calm_phase3c_taskc1_transaction":
        if transaction_status != "committed":
            return False
        if runtime_path not in (None, "", "candidate_first_crossing"):
            return False
        source_layer = row.get("selected_source_layer", row.get("source_layer"))
        try:
            return int(source_layer) in (4, 6, 8, 10)
        except Exception:
            return False
    if row.get("runtime_path") == "candidate_first_crossing" and transaction_status == "committed":
        return True
    first_crossing = row.get("first_crossing_candidate_layer")
    if first_crossing is None:
        first_crossing = row.get("actual_first_crossing_candidate_layer")
    if first_crossing is not None:
        try:
            return int(first_crossing) in (4, 6, 8, 10)
        except Exception:
            return False
    evaluations = row.get("candidate_evaluations")
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            if bool(item.get("candidate_pass")) or bool(item.get("passed")):
                try:
                    return int(item.get("candidate_layer")) in (4, 6, 8, 10)
                except Exception:
                    return False
    return False


_TRACE_ARTIFACT_IDENTITY_ALIASES = (
    "accepted_artifact_file_sha256",
    "artifact_file_sha256",
    "phase3c_artifact_sha256",
    "runtime_restoration_artifact_sha256",
    "kv_runtime_restoration_artifact_sha256",
)
_TRACE_RUN_IDENTITY_ALIASES = (
    "run_context_identity_sha256",
    "run_identity_sha256",
    "run_id",
)
_TRACE_SCHEMA_VERSION_ALIASES = (
    "trace_schema_version",
    "schema_version",
    "calm_trace_schema_version",
)


def _row_alias_string(
    row: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    conflict_failure: str,
    index: int,
    failures: List[str],
) -> Optional[str]:
    values = [str(row.get(alias)) for alias in aliases if alias in row and not is_placeholder_value(row.get(alias))]
    unique = sorted(set(values))
    if len(unique) > 1:
        failures.append("{}:{}".format(conflict_failure, index))
        return None
    return unique[0] if unique else None


def _row_schema_version(row: Mapping[str, Any], index: int, failures: List[str]) -> Optional[int]:
    value = None
    for alias in _TRACE_SCHEMA_VERSION_ALIASES:
        if alias in row and not is_placeholder_value(row.get(alias)):
            value = row.get(alias)
            break
    if value is None:
        return None
    if isinstance(value, bool):
        failures.append("smoke_selection_trace_schema_version_invalid:{}".format(index))
        return None
    try:
        return int(value)
    except Exception:
        failures.append("smoke_selection_trace_schema_version_invalid:{}".format(index))
        return None


def derive_smoke_candidate_ids_from_trace(
    path: Path | str,
    *,
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
) -> Dict[str, Any]:
    target = Path(path)
    failures: List[str] = []
    try:
        rows = read_jsonl(target)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": ["smoke_selection_trace_parse_failed:{}".format(type(exc).__name__)],
            "selection_source_sha256": sha256_file(target) if target.is_file() else None,
        }
    if not rows:
        failures.append("smoke_selection_trace_empty")
    artifact_values: set[str] = set()
    schema_versions: List[int] = []
    run_identity_values: set[str] = set()
    row_identity_cache: Dict[int, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        artifact_value = _row_alias_string(
            row,
            _TRACE_ARTIFACT_IDENTITY_ALIASES,
            conflict_failure="smoke_selection_trace_artifact_identity_conflict",
            index=index,
            failures=failures,
        )
        if artifact_value is not None:
            artifact_values.add(artifact_value)
        run_value = _row_alias_string(
            row,
            _TRACE_RUN_IDENTITY_ALIASES,
            conflict_failure="smoke_selection_trace_run_identity_conflict",
            index=index,
            failures=failures,
        )
        if run_value is not None:
            run_identity_values.add(run_value)
        schema_version = _row_schema_version(row, index, failures)
        if schema_version is not None:
            schema_versions.append(schema_version)
        row_identity_cache[index] = {
            "artifact": artifact_value,
            "run": run_value,
            "schema": schema_version,
        }
    if not artifact_values:
        failures.append("smoke_selection_trace_artifact_identity_missing")
    elif len(artifact_values) > 1:
        failures.append("smoke_selection_trace_artifact_identity_multiple")
    elif next(iter(artifact_values)) != expected_artifact_sha256:
        failures.append("smoke_selection_trace_artifact_sha256_mismatch")
    unique_schema_versions = sorted(set(schema_versions))
    if not unique_schema_versions:
        failures.append("smoke_selection_trace_schema_version_missing")
    elif len(unique_schema_versions) > 1:
        failures.append("smoke_selection_trace_schema_version_multiple")
    elif unique_schema_versions[0] != CALM_TRACE_SCHEMA_VERSION:
        failures.append("smoke_selection_trace_schema_version_mismatch")
    sorted_run_identity_values = sorted(run_identity_values)
    if not sorted_run_identity_values:
        failures.append("smoke_selection_trace_run_identity_missing")
    elif len(sorted_run_identity_values) > 1:
        failures.append("smoke_selection_trace_run_identity_multiple")
    seen: set[str] = set()
    eligible_ids: List[str] = []
    candidate_event_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append("smoke_selection_trace_row_not_mapping:{}".format(index))
            continue
        if not _trace_row_has_candidate_first_crossing(row):
            continue
        identity = row_identity_cache.get(index) or {}
        if identity.get("artifact") is None:
            failures.append("smoke_selection_trace_candidate_artifact_identity_missing:{}".format(index))
            continue
        if identity.get("run") is None:
            failures.append("smoke_selection_trace_candidate_run_identity_missing:{}".format(index))
            continue
        if identity.get("schema") is None:
            failures.append("smoke_selection_trace_candidate_schema_version_missing:{}".format(index))
            continue
        if identity.get("artifact") != expected_artifact_sha256:
            failures.append("smoke_selection_trace_candidate_artifact_identity_mismatch:{}".format(index))
            continue
        if identity.get("schema") != CALM_TRACE_SCHEMA_VERSION:
            failures.append("smoke_selection_trace_candidate_schema_version_mismatch:{}".format(index))
            continue
        candidate_event_count += 1
        try:
            stable = _stable_sample_id_from_trace_row(row)
        except Exception:
            failures.append("smoke_selection_trace_stable_sample_id_invalid:{}".format(index))
            continue
        if stable is None:
            failures.append("smoke_selection_trace_stable_sample_id_missing:{}".format(index))
            continue
        if stable not in seen:
            seen.add(stable)
            eligible_ids.append(stable)
    if candidate_event_count == 0:
        failures.append("smoke_selection_candidate_events_missing")
    if not eligible_ids:
        failures.append("smoke_selection_candidate_stable_sample_ids_missing")
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "eligible_candidate_stable_sample_ids": eligible_ids,
        "eligible_candidate_sample_count": len(eligible_ids),
        "candidate_first_crossing_event_count": candidate_event_count,
        "selection_source_sha256": sha256_file(target),
        "selection_source_schema_version": unique_schema_versions[0] if len(unique_schema_versions) == 1 else None,
        "selection_source_run_identity": sorted_run_identity_values[0] if len(sorted_run_identity_values) == 1 else None,
        "selection_source_artifact_identity": sorted(artifact_values)[0] if len(artifact_values) == 1 else None,
    }


def load_smoke_selection_evidence(
    path: Path | str,
    *,
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
) -> Dict[str, Any]:
    target = Path(path)
    failures: List[str] = []
    if not str(path):
        return {"status": "failed", "failures": ["smoke_selection_source_required"]}
    if not target.is_file():
        return {"status": "failed", "failures": ["smoke_selection_source_missing"], "selection_source_path": str(target)}
    try:
        payload = read_json(target)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": ["smoke_selection_source_parse_failed:{}".format(type(exc).__name__)],
            "selection_source_path": str(target),
            "selection_source_sha256": sha256_file(target) if target.is_file() else None,
        }
    artifact_sha = payload.get("accepted_artifact_file_sha256") or payload.get("artifact_file_sha256")
    if artifact_sha != expected_artifact_sha256:
        failures.append("smoke_selection_source_artifact_sha256_mismatch")
    source_path = _resolve_manifest_path(target, payload.get("selection_source_path") or payload.get("source_trace_path"))
    expected_source_sha = payload.get("selection_source_sha256") or payload.get("source_trace_sha256")
    computed_source_sha = None
    trace_result: Dict[str, Any] = {}
    if source_path is not None:
        if source_path.is_file():
            computed_source_sha = sha256_file(source_path)
            if expected_source_sha and expected_source_sha != computed_source_sha:
                failures.append("smoke_selection_source_sha256_mismatch")
            trace_result = derive_smoke_candidate_ids_from_trace(
                source_path,
                expected_artifact_sha256=expected_artifact_sha256,
            )
            failures.extend(trace_result.get("failures") or [])
        else:
            failures.append("smoke_selection_source_file_missing")
    else:
        failures.append("smoke_selection_source_file_required")

    candidate_ids = _list_from_payload(
        payload,
        ("selected_stable_sample_ids", "candidate_stable_sample_ids", "stable_sample_ids"),
    )
    if candidate_ids is None and isinstance(payload.get("records"), list):
        candidate_ids = [
            row.get("stable_sample_id") if isinstance(row, Mapping) else row
            for row in payload.get("records") or []
        ]
    if candidate_ids is None:
        declared_ids = None
        selected_ids: List[str] = []
    else:
        try:
            selected_ids = [as_stable_sample_id(value) for value in candidate_ids]
            declared_ids = list(selected_ids)
        except Exception as exc:
            failures.append("smoke_selection_stable_sample_ids_invalid:{}".format(type(exc).__name__))
            selected_ids = []
            declared_ids = None
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "selection_source_type": payload.get("selection_source_type") or payload.get("evidence_type") or "trace_linked_stable_sample_ids",
        "selection_source_path": str(target),
        "selection_source_file_sha256": sha256_file(target),
        "selection_source_sha256": computed_source_sha or expected_source_sha,
        "selection_source_schema_version": trace_result.get("selection_source_schema_version") or payload.get("selection_source_schema_version"),
        "selection_source_run_identity": trace_result.get("selection_source_run_identity") or payload.get("selection_source_run_identity"),
        "selection_source_artifact_identity": trace_result.get("selection_source_artifact_identity"),
        "selection_source_accepted_artifact_binding": trace_result.get("selection_source_artifact_identity") == expected_artifact_sha256,
        "eligible_candidate_stable_sample_ids": trace_result.get("eligible_candidate_stable_sample_ids") or [],
        "eligible_candidate_sample_count": trace_result.get("eligible_candidate_sample_count", 0),
        "candidate_first_crossing_event_count": trace_result.get("candidate_first_crossing_event_count", 0),
        "declared_selected_stable_sample_ids": declared_ids,
        "selected_stable_sample_ids": selected_ids,
        "selection_rule": payload.get("selection_rule") or "candidate_first_crossing_trace_first_n_by_heldout_order",
    }


def build_artifact_heldout_smoke_subset(
    heldout_records: Sequence[Mapping[str, Any]],
    *,
    max_samples: int,
    selection_source_identity: Optional[Mapping[str, Any]] = None,
    selected_stable_sample_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    heldout_ids = row_stable_sample_ids(heldout_records)
    heldout_by_id = {stable: dict(row) for stable, row in zip(heldout_ids, heldout_records)}
    if len(heldout_by_id) != len(heldout_ids):
        failures.append("heldout_duplicate_stable_sample_id")
    if selected_stable_sample_ids is None:
        selected_ids = heldout_ids[: int(max_samples)]
        selection_rule = "first_n_from_artifact_heldout_order"
    else:
        selected_ids = [as_stable_sample_id(value) for value in selected_stable_sample_ids]
        selection_rule = "explicit_pre_quality_stable_sample_ids"
    if len(selected_ids) > int(max_samples):
        failures.append("smoke_subset_exceeds_max_samples")
    missing = [stable for stable in selected_ids if stable not in heldout_by_id]
    if missing:
        failures.append("smoke_subset_id_not_in_heldout")
    try:
        identity = build_stable_sample_identities(selected_ids, expected_count=len(selected_ids), prefix="selected_")
    except Exception as exc:
        failures.append("smoke_subset_identity_failed:{}".format(type(exc).__name__))
        identity = {}
    return {
        "schema_version": PAPER_POPULATION_IDENTITY_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "population_mode": "artifact_heldout_smoke_subset",
        "selection_rule": selection_rule,
        "selection_source_identity": dict(selection_source_identity or {}),
        "full_heldout_population_count": len(heldout_ids),
        "selected_population_count": len(selected_ids),
        "population_capped": True,
        "selected_stable_sample_ids": selected_ids,
        "selected_records": [heldout_by_id[stable] for stable in selected_ids if stable in heldout_by_id],
        **identity,
    }


def resolve_population_from_split_evidence(
    *,
    dataset_records: Sequence[Mapping[str, Any]],
    split_evidence_path: Path | str,
    mode: str,
    smoke_max_eval_samples: int,
    smoke_selection_evidence_path: Optional[Path | str] = None,
    expected_dataset_count: int = EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
    expected_fitting_count: int = EXPECTED_ARTIFACT_FITTING_COUNT,
    expected_heldout_count: int = EXPECTED_ARTIFACT_HELDOUT_COUNT,
    failure_prefix: str = "artifact",
) -> Dict[str, Any]:
    failures: List[str] = []
    split = load_artifact_split_evidence(split_evidence_path, failure_prefix=failure_prefix)
    failures.extend(split.get("failures") or [])
    population = resolve_artifact_heldout_population(
        dataset_records,
        split.get("artifact_fitting_stable_sample_ids") or [],
        split_source=split.get("split_source"),
        expected_dataset_count=expected_dataset_count,
        expected_fitting_count=expected_fitting_count,
        expected_heldout_count=expected_heldout_count,
    )
    failures.extend(population.get("failures") or [])
    heldout_records = list(population.get("heldout_records") or [])
    heldout_ids = row_stable_sample_ids(heldout_records) if heldout_records else []

    supplied_heldout_ids = split.get("supplied_heldout_stable_sample_ids")
    if supplied_heldout_ids is not None:
        supplied_ids = [str(item) for item in supplied_heldout_ids]
        if len(supplied_ids) != len(set(supplied_ids)):
            failures.append("supplied_heldout_duplicate_stable_sample_id")
        if set(supplied_ids) != set(heldout_ids):
            failures.append("supplied_heldout_set_mismatch")
        if supplied_ids != heldout_ids:
            failures.append("supplied_heldout_order_mismatch")

    selected_ids: List[str]
    selection_identity: Dict[str, Any]
    population_capped = mode != "full"
    if mode == "full":
        selected_ids = heldout_ids
        selection_identity = {
            "selection_rule": "all_artifact_heldout_samples",
            "selection_source_type": "artifact_heldout_complement",
            "selection_source_path": str(split_evidence_path),
            "selection_source_file_sha256": split.get("split_evidence_sha256"),
            "selection_source_sha256": split.get("split_evidence_sha256"),
            "selection_source_schema_version": split.get("schema_version"),
            "selection_source_run_identity": None,
            "selection_source_artifact_identity": split.get("split_source", {}).get("accepted_artifact_file_sha256"),
            "selection_source_accepted_artifact_binding": split.get("status") == "ok",
            "eligible_candidate_sample_count": None,
            "candidate_first_crossing_event_count": None,
        }
    else:
        selection = load_smoke_selection_evidence(str(smoke_selection_evidence_path or ""))
        failures.extend(selection.get("failures") or [])
        eligible_candidate_ids = [str(item) for item in selection.get("eligible_candidate_stable_sample_ids") or []]
        eligible_candidate_set = set(eligible_candidate_ids)
        heldout_set = set(heldout_ids)
        eligible_heldout_ids = [stable for stable in heldout_ids if stable in eligible_candidate_set]
        declared_ids = selection.get("declared_selected_stable_sample_ids")
        if declared_ids is not None:
            declared_ids = [str(item) for item in declared_ids]
            missing_from_heldout = [stable for stable in declared_ids if stable not in heldout_set]
            if missing_from_heldout:
                failures.append("smoke_selection_id_not_in_heldout")
            missing_from_trace = [stable for stable in declared_ids if stable not in eligible_candidate_set]
            if missing_from_trace:
                failures.append("smoke_selection_id_not_in_candidate_trace")
        selected_ids = eligible_heldout_ids[: int(smoke_max_eval_samples)]
        if declared_ids is not None and declared_ids != selected_ids:
            failures.append("smoke_selection_declared_ids_mismatch")
        if len(selected_ids) < int(smoke_max_eval_samples):
            failures.append("smoke_selection_candidate_count_insufficient:{}:{}".format(len(selected_ids), int(smoke_max_eval_samples)))
        selection_identity = {
            key: selection.get(key)
            for key in (
                "selection_rule",
                "selection_source_type",
                "selection_source_path",
                "selection_source_file_sha256",
                "selection_source_sha256",
                "selection_source_schema_version",
                "selection_source_run_identity",
                "selection_source_artifact_identity",
                "selection_source_accepted_artifact_binding",
                "eligible_candidate_sample_count",
                "candidate_first_crossing_event_count",
            )
        }
        selection_identity["selection_rule"] = "candidate_first_crossing_trace_first_n_by_heldout_order"

    heldout_by_id = {stable: dict(row) for stable, row in zip(heldout_ids, heldout_records)}
    selected_records = [heldout_by_id[stable] for stable in selected_ids if stable in heldout_by_id]
    try:
        heldout_identity = build_stable_sample_identities(heldout_ids, expected_count=expected_heldout_count, prefix="heldout_")
    except Exception as exc:
        failures.append("heldout_identity_failed:{}".format(type(exc).__name__))
        heldout_identity = {}
    try:
        selected_identity = build_stable_sample_identities(selected_ids, expected_count=len(selected_ids), prefix="selected_")
    except Exception as exc:
        failures.append("selected_identity_failed:{}".format(type(exc).__name__))
        selected_identity = {}
    fitting_ids = [str(item) for item in split.get("artifact_fitting_stable_sample_ids") or []]
    try:
        fitting_identity = build_stable_sample_identities(
            fitting_ids,
            expected_count=expected_fitting_count,
            prefix="artifact_fitting_",
        )
    except Exception as exc:
        failures.append("artifact_fitting_identity_failed:{}".format(type(exc).__name__))
        fitting_identity = {}
    return {
        "schema_version": PAPER_POPULATION_IDENTITY_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "failure_stage": None if not failures else "heldout_population_resolution",
        "failures": failures,
        "population_mode": "artifact_heldout",
        "population_capped": population_capped,
        "dataset_population_count": population.get("dataset_population_count"),
        "fitting_count": population.get("fitting_count"),
        "heldout_count": population.get("heldout_count"),
        "full_heldout_population_count": population.get("heldout_count"),
        "selected_population_count": len(selected_ids),
        "intersection_count": population.get("intersection_count"),
        "union_count": population.get("union_count"),
        "duplicate_dataset_stable_id_count": population.get("duplicate_dataset_stable_id_count"),
        "duplicate_fitting_id_count": population.get("duplicate_fitting_id_count"),
        "duplicate_heldout_id_count": population.get("duplicate_heldout_id_count"),
        "fitting_id_missing_from_dataset_count": population.get("fitting_id_missing_from_dataset_count"),
        "heldout_id_missing_from_dataset_count": population.get("heldout_id_missing_from_dataset_count"),
        "accepted_artifact_file_sha256": ACCEPTED_PHASE3C_ARTIFACT_SHA256,
        "split_source": split.get("split_source"),
        "split_evidence_path": split.get("split_evidence_path"),
        "split_evidence_sha256": split.get("split_evidence_sha256"),
        "split_evidence_schema_version": split.get("split_evidence_schema_version"),
        "split_evidence_type": split.get("split_evidence_type"),
        "source_manifest_sha256": (split.get("split_source") or {}).get("source_manifest_sha256"),
        "source_manifest_schema_version": (split.get("split_source") or {}).get("source_manifest_schema_version"),
        "source_manifest_identity": (split.get("split_source") or {}).get("source_manifest_identity"),
        "source_manifest_accepted_artifact_binding": (split.get("split_source") or {}).get("source_manifest_accepted_artifact_binding"),
        "source_manifest_hybrid_fitting_policy_binding": (split.get("split_source") or {}).get("source_manifest_hybrid_fitting_policy_binding"),
        "source_manifest_runtime_policy_binding": (split.get("split_source") or {}).get("source_manifest_runtime_policy_binding"),
        "source_manifest_immutable_binding_type": (split.get("split_source") or {}).get("source_manifest_immutable_binding_type"),
        "source_manifest_immutable_binding_valid": (split.get("split_source") or {}).get("source_manifest_immutable_binding_valid"),
        "source_manifest_immutable_binding_reference": (split.get("split_source") or {}).get("source_manifest_immutable_binding_reference"),
        "source_manifest_immutable_binding_reference_sha256": (split.get("split_source") or {}).get("source_manifest_immutable_binding_reference_sha256"),
        "source_manifest_expected_sha256": (split.get("split_source") or {}).get("source_manifest_expected_sha256"),
        "source_manifest_actual_sha256": (split.get("split_source") or {}).get("source_manifest_actual_sha256"),
        "stable_sample_id_algorithm_identity": (split.get("split_source") or {}).get("stable_sample_id_algorithm_identity"),
        "dataset_name": (split.get("split_source") or {}).get("dataset_name"),
        "dataset_config_name": (split.get("split_source") or {}).get("dataset_config_name"),
        "dataset_split": (split.get("split_source") or {}).get("dataset_split"),
        "text_column": (split.get("split_source") or {}).get("text_column"),
        "summary_column": (split.get("split_source") or {}).get("summary_column"),
        "selected_stable_sample_ids": selected_ids,
        **selection_identity,
        **fitting_identity,
        **heldout_identity,
        **selected_identity,
        "selected_records": selected_records,
    }


def public_population_summary(population: Mapping[str, Any]) -> Dict[str, Any]:
    hidden_keys = {
        "artifact_fitting_stable_sample_ids",
        "heldout_records",
        "selected_records",
        "selected_stable_sample_ids",
        "supplied_heldout_stable_sample_ids",
    }
    return {key: value for key, value in population.items() if key not in hidden_keys}


def write_population_outputs(
    population: Mapping[str, Any],
    *,
    population_jsonl: Path,
    summary_json: Path,
    stable_sample_ids_txt: Path,
    selected_ids_jsonl: Optional[Path] = None,
) -> None:
    selected_records = list(population.get("selected_records") or [])
    write_jsonl(population_jsonl, selected_records)
    write_json(summary_json, public_population_summary(population))
    ids = [as_stable_sample_id(row.get("stable_sample_id")) for row in selected_records]
    stable_sample_ids_txt.parent.mkdir(parents=True, exist_ok=True)
    stable_sample_ids_txt.write_text("".join("{}\n".format(stable) for stable in ids), encoding="utf-8")
    if selected_ids_jsonl is not None:
        write_jsonl(selected_ids_jsonl, [{"stable_sample_id": stable} for stable in ids])


__all__ = [
    "ACCEPTED_PHASE3C_ARTIFACT_SHA256",
    "ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256",
    "ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256",
    "ARTIFACT_SPLIT_EVIDENCE_SCHEMA_VERSION",
    "ARTIFACT_SPLIT_EVIDENCE_TYPE",
    "EXPECTED_DATASET_CONFIG_NAME",
    "EXPECTED_DATASET_NAME",
    "EXPECTED_DATASET_SPLIT",
    "EXPECTED_ARTIFACT_FITTING_COUNT",
    "EXPECTED_ARTIFACT_HELDOUT_COUNT",
    "EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT",
    "EXPECTED_SUMMARY_COLUMN",
    "EXPECTED_TEXT_COLUMN",
    "PAPER_POPULATION_IDENTITY_SCHEMA_VERSION",
    "SMOKE_SELECTION_SCHEMA_VERSION",
    "SOURCE_MANIFEST_IMMUTABLE_BINDING_SCHEMA_VERSION",
    "SOURCE_MANIFEST_IMMUTABLE_BINDING_TYPES",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "STABLE_SAMPLE_ID_ALGORITHM_IDENTITY",
    "as_stable_sample_id",
    "build_artifact_heldout_smoke_subset",
    "build_dataset_population_records",
    "build_stable_sample_identities",
    "derive_smoke_candidate_ids_from_trace",
    "duplicate_count",
    "is_placeholder_value",
    "load_artifact_split_evidence",
    "load_source_manifest_fitting_ids",
    "load_smoke_selection_evidence",
    "public_population_summary",
    "read_json",
    "read_jsonl",
    "resolve_artifact_heldout_population",
    "resolve_population_from_split_evidence",
    "row_stable_sample_ids",
    "stable_sample_ids_sha256",
    "stable_sample_ids_to_lf_bytes",
    "write_json",
    "write_jsonl",
    "write_population_outputs",
]
