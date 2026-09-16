"""CPU-only validation for reconstructed Phase 3c fitting provenance.

This module keeps three identities deliberately separate:

* the accepted read-only provenance-audit archive,
* the historical producer split digest, and
* the current UTF-8/LF stable-sample population identity.

The resulting corrective manifest is a draft for central review.  It can
never authorize a paper run or approve itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .missing_kv_dump_provenance import (
    canonical_json_sha256,
    sha256_file,
)
from .missing_kv_paper_population import (
    ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256,
    ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256,
    EXPECTED_ARTIFACT_FITTING_COUNT,
    EXPECTED_ARTIFACT_HELDOUT_COUNT,
    EXPECTED_DATASET_CONFIG_NAME,
    EXPECTED_DATASET_NAME,
    EXPECTED_DATASET_SPLIT,
    EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
    EXPECTED_SUMMARY_COLUMN,
    EXPECTED_TEXT_COLUMN,
    STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
    build_dataset_population_records,
    resolve_artifact_heldout_population,
    stable_sample_ids_sha256,
)
from .phase3c_fitting_evidence_roles import (
    ACCEPTED_ARTIFACT_IDENTITY_SIDECAR_SHA256,
    ACCEPTED_AUDIT_ROLE_SPECS,
    ACCEPTED_CHILD_FULL_EXIT_CODE_SHA256,
    ACCEPTED_FULL_RUN_EXIT_CODE_SHA256,
    ACCEPTED_FULL_RUN_FILE_MANIFEST_SHA256,
    ACCEPTED_GIT_IDENTITY_SIDECAR_SHA256,
    ACCEPTED_OUTER_ACCEPTANCE_EXIT_CODE_SHA256,
    ACCEPTED_OUTER_FULL_ACCEPTANCE_SHA256,
    REQUIRED_EVIDENCE_ROLES as EXACT_REQUIRED_EVIDENCE_ROLES,
    ROLE_BINDING_SCHEMA_VERSION,
    SUPPORTING_AUDIT_ROLE_SPECS,
    canonical_resolved_role_specs_from_records,
    extract_artifact_provenance_identities,
    resolve_exact_roles_from_verified_audit,
    validate_exact_role_evidence_chain,
)


ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256 = (
    "a242fd8dee159a775382eb5c4bfd7b086e04cbf73128a199ad885f944c86fec6"
)
ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256 = (
    "610329004da0ce619b6ee0398bb6af4102a1ca4969902e9a0daf6ef97e63e525"
)

EXPECTED_HISTORICAL_TRAIN_DIGEST = (
    "f607b32389e78bc7c61829c56c8982ae014530d173c9da98b0fcf8b2a7ddb385"
)
EXPECTED_HISTORICAL_EVAL_DIGEST = (
    "cabeebee18f45bfea659e25609fc8accb15fdcb66c25e062d4d7d7396254440d"
)
EXPECTED_HISTORICAL_ALIGNED_DIGEST = (
    "f05dcd2960e50fb7070e80defc84590ff4e882b7a93e19e91a356ca0292db98d"
)

EXPECTED_CANONICAL_FITTING_SET_SHA256 = (
    "4281213b5248fcf1889e3599ad2841b108630b2c31102f0e79aa29ee905b32d8"
)
EXPECTED_CANONICAL_HELDOUT_SET_SHA256 = (
    "1025ae3ae1f8fa4fd53d5dc65a2941a8630bab6838fe68ea190905800b53dc77"
)
EXPECTED_CANONICAL_UNION_SET_SHA256 = (
    "223b4bd0b41134a7d47738e5a56b821ef76ac490145f2006fb96aa29b050f985"
)

# Centrally approved root-of-trust identities (Conversation 1 approval).
# The corrective manifest bytes are never rewritten to carry these values;
# approval is represented purely as an external SHA-256 pin checked by
# paper_facing_corrective_binding_preflight() against the immutable,
# still-pending-review manifest file.
APPROVED_CORRECTIVE_MANIFEST_SHA256 = (
    "afc3d54e7c8d1eecb1cf6191b7addc834dfd0e9f4d5dd4cf868e858b66fb835b"
)
APPROVED_CORRECTIVE_EVIDENCE_CHAIN_SHA256 = (
    "780c1d0c95e11e6e41d06a329458321c28e1a8fad6ac17704de0601a390b86c3"
)
APPROVED_HELDOUT_DATASET_ORDER_SHA256 = (
    "cc4787e33d9d14f8f1e1a175c93b02d78d4c3fe87da4934c16b4f07354ea0ef7"
)
# Review-provenance metadata only. Not a mandatory preflight dependency:
# the resume review bundle is not re-fetched or re-verified by the
# approved-root preflight below.
APPROVED_RESUME_REVIEW_BUNDLE_SHA256 = (
    "843fff2f5d8bcb1765f70b9bb8e5eee8fbedb4b8dc01b1599c8a87699b4216f6"
)

CENTRAL_APPROVAL_CONTRACT: Dict[str, Any] = {
    "approval_status": "approved_established",
    "corrective_manifest_sha256": APPROVED_CORRECTIVE_MANIFEST_SHA256,
    "corrective_evidence_chain_sha256": APPROVED_CORRECTIVE_EVIDENCE_CHAIN_SHA256,
    "accepted_audit_bundle_sha256": ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256,
    "resume_review_bundle_sha256": APPROVED_RESUME_REVIEW_BUNDLE_SHA256,
    "accepted_artifact_sha256": ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    "fitting_set_sha256": EXPECTED_CANONICAL_FITTING_SET_SHA256,
    "heldout_set_sha256": EXPECTED_CANONICAL_HELDOUT_SET_SHA256,
    "union_set_sha256": EXPECTED_CANONICAL_UNION_SET_SHA256,
    "heldout_dataset_order_sha256": APPROVED_HELDOUT_DATASET_ORDER_SHA256,
    "fitting_count": EXPECTED_ARTIFACT_FITTING_COUNT,
    "heldout_count": EXPECTED_ARTIFACT_HELDOUT_COUNT,
    "intersection_count": 0,
    "union_count": EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
}

HISTORICAL_DIGEST_ALGORITHM_IDENTITY = (
    "evaluate_phase2_hidden_based_kv_regeneration_from_dumps.identity_digest:"
    "format_identity_v1"
)
HISTORICAL_DIGEST_SERIALIZATION = {
    "encoding": "utf-8",
    "identity_format": "pipe_join_each_identity_iterable",
    "identity_sort": "python_default_sorted",
    "record_separator": "lf",
    "final_newline": False,
}
CANONICAL_SET_ALGORITHM_IDENTITY = "stable_sample_ids_sha256:utf8_lf_final_newline:lexicographic_v1"

CORRECTIVE_BINDING_SCHEMA_VERSION = 2
CORRECTIVE_BINDING_EVIDENCE_TYPE = (
    "accepted_phase3c_corrective_fitting_run_binding_manifest_v2"
)
CORRECTIVE_BINDING_PROVENANCE_CONSTRUCTION = (
    "reconstructed_from_original_run_evidence"
)
CORRECTIVE_BINDING_APPROVAL_STATUS = "pending_central_review"

EXPECTED_SPLIT_ALGORITHM = "stable_sample_cluster_split_v1"
EXPECTED_SPLIT_UNIT = "stable_sample_id"
EXPECTED_SPLIT_MODE = "shuffle"
EXPECTED_SPLIT_RATIO = 0.5
EXPECTED_SPLIT_SEED = 0

EXPECTED_OUTER_ACCEPTANCE_FAILURES = (
    "replay_run_config:max_train_records_per_group_not_zero",
    "replay_run_config:max_eval_records_per_threshold_not_zero",
)

LEGACY_REQUIRED_EVIDENCE_ROLES = (
    "external_population_comparison",
    "original_fitting_command",
    "fit_summary",
    "artifact_summary",
    "artifact_identity",
    "child_final_status",
    "outer_full_acceptance",
    "current_artifact_preflight",
    "producer_git_identity",
    "return_code_sidecar",
    "file_manifest",
)
REQUIRED_EVIDENCE_ROLES = EXACT_REQUIRED_EVIDENCE_ROLES

ROLE_RELATIONS = {
    "external_population_comparison": "input_population",
    "original_fitting_command": "fitting_invocation",
    "fit_summary": "fitting_result",
    "artifact_summary": "artifact_output",
    "artifact_identity": "artifact_bytes_identity",
    "child_final_status": "successful_child_completion",
    "outer_full_acceptance": "outer_acceptance_state",
    "current_artifact_preflight": "current_artifact_identity",
    "producer_git_identity": "producer_identity",
    "return_code_sidecar": "successful_child_completion",
    "child_full_exit_code": "outer_wrapper_child_fitting_completion",
    "outer_acceptance_exit_code": "outer_acceptance_execution_code",
    "file_manifest": "producer_identity",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ROLE_ORDER = {role: index for index, role in enumerate(REQUIRED_EVIDENCE_ROLES)}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _normalized_local_artifact_path(value: Any) -> Optional[str]:
    if not isinstance(value, (str, Path)) or str(value) in ("", "."):
        return None
    try:
        return Path(value).resolve(strict=False).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def _bounded(value: Any, limit: int = 160) -> str:
    return str(value)[:limit]


def _normalized_archive_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    if normalized in ("", ".", "..") or normalized.startswith(("/", "//")):
        return None
    if len(normalized) >= 2 and normalized[1] == ":":
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return "/".join(path.parts)


def _json_load(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl_load(path: Path) -> List[Any]:
    rows: List[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    "jsonl_parse_failed:{}:{}".format(line_no, type(exc).__name__)
                ) from exc
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def historical_format_identity(identity: Any) -> str:
    """Exact compatibility copy of the Phase 2 producer formatter."""

    return "|".join(str(part) for part in identity)


def historical_identity_digest(identities: Iterable[Any]) -> str:
    """Reproduce the historical producer digest without changing semantics."""

    formatted = [
        historical_format_identity(item)
        for item in sorted(identities)
    ]
    return hashlib.sha256("\n".join(formatted).encode("utf-8")).hexdigest()


def _extract_archive(
    archive_path: Path,
    *,
    destination: Path,
) -> Dict[str, Any]:
    failures: List[str] = []
    observed: Dict[str, Dict[str, Any]] = {}
    seen_paths: set[str] = set()
    roots: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                normalized = _normalized_archive_path(member.name)
                if normalized is None:
                    failures.append("audit_archive_member_path_unsafe")
                    continue
                if normalized in seen_paths:
                    failures.append(
                        "audit_archive_duplicate_member_path:{}".format(normalized)
                    )
                    continue
                seen_paths.add(normalized)
                parts = PurePosixPath(normalized).parts
                if parts:
                    roots.add(parts[0])
                if member.isdir():
                    (destination / Path(*parts)).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    failures.append(
                        "audit_archive_member_not_regular_file:{}".format(normalized)
                    )
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    failures.append(
                        "audit_archive_member_extract_failed:{}".format(normalized)
                    )
                    continue
                target = destination / Path(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                try:
                    with stream, target.open("xb") as output:
                        while True:
                            chunk = stream.read(1024 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            digest.update(chunk)
                            output.write(chunk)
                except FileExistsError:
                    failures.append(
                        "audit_archive_duplicate_extraction_target:{}".format(
                            normalized
                        )
                    )
                    continue
                observed[normalized] = {
                    "relative_path": normalized,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
                if size != int(member.size):
                    failures.append(
                        "audit_archive_member_size_changed:{}".format(normalized)
                    )
    except Exception as exc:
        failures.append("audit_archive_open_failed:{}".format(type(exc).__name__))
    if len(roots) != 1:
        failures.append("audit_archive_single_root_required")
    return {
        "failures": failures,
        "members_by_path": observed,
        "archive_root_name": next(iter(roots)) if len(roots) == 1 else None,
    }


def _verify_internal_audit_manifest(
    extracted_root: Path,
    members_by_path: Mapping[str, Mapping[str, Any]],
    archive_root_name: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    manifest_rel = "{}/audit_output_file_manifest.jsonl".format(
        archive_root_name
    )
    manifest_member = members_by_path.get(manifest_rel)
    if not isinstance(manifest_member, Mapping):
        return {
            "status": "failed",
            "failures": ["audit_internal_manifest_missing"],
            "manifest_relative_path": manifest_rel,
        }
    manifest_path = extracted_root / Path(*PurePosixPath(manifest_rel).parts)
    try:
        rows = _jsonl_load(manifest_path)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": [
                "audit_internal_manifest_parse_failed:{}".format(
                    type(exc).__name__
                )
            ],
            "manifest_relative_path": manifest_rel,
            "manifest_sha256": manifest_member.get("sha256"),
        }
    expected: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append("audit_internal_manifest_row_not_mapping:{}".format(index))
            continue
        relative = _normalized_archive_path(row.get("relative_path"))
        if relative is None:
            failures.append(
                "audit_internal_manifest_member_path_unsafe:{}".format(index)
            )
            continue
        archive_relative = "{}/{}".format(archive_root_name, relative)
        if archive_relative in expected:
            failures.append(
                "audit_internal_manifest_duplicate_member_path:{}".format(
                    archive_relative
                )
            )
            continue
        expected[archive_relative] = row
    for relative, row in sorted(expected.items()):
        actual = members_by_path.get(relative)
        if actual is None:
            failures.append(
                "audit_internal_manifest_member_missing:{}".format(relative)
            )
            continue
        if row.get("size_bytes") != actual.get("size_bytes"):
            failures.append(
                "audit_internal_manifest_member_size_mismatch:{}".format(
                    relative
                )
            )
        if row.get("sha256") != actual.get("sha256"):
            failures.append(
                "audit_internal_manifest_member_sha256_mismatch:{}".format(
                    relative
                )
            )
    allowed = set(expected) | {manifest_rel}
    for relative in sorted(set(members_by_path) - allowed):
        failures.append(
            "audit_archive_unexpected_file_not_in_manifest:{}".format(relative)
        )
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "manifest_relative_path": manifest_rel,
        "manifest_sha256": manifest_member.get("sha256"),
        "manifest_row_count": len(rows),
    }


def _verify_audit_bundle_archive(
    archive_path: Path | str,
    *,
    expected_sha256: str,
    extract_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Internal verifier with injectable root for frozen test fixtures."""

    archive_path = Path(archive_path)
    failures: List[str] = []
    if not archive_path.is_file():
        return {
            "schema_version": 1,
            "status": "failed",
            "failures": ["accepted_audit_bundle_missing"],
            "archive_path": str(archive_path),
            "expected_archive_sha256": expected_sha256,
        }
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_sha256:
        failures.append("accepted_audit_bundle_sha256_mismatch")
        return {
            "schema_version": 1,
            "status": "failed",
            "failures": failures,
            "archive_path": str(archive_path),
            "archive_sha256": actual_sha,
            "expected_archive_sha256": expected_sha256,
        }

    temporary: Optional[tempfile.TemporaryDirectory[str]] = None
    if extract_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="phase3c_fitting_audit_verify_"
        )
        destination = Path(temporary.name)
    else:
        destination = Path(extract_dir)
        if destination.exists():
            return {
                "schema_version": 1,
                "status": "failed",
                "failures": ["audit_extract_directory_must_not_exist"],
                "archive_path": str(archive_path),
                "archive_sha256": actual_sha,
                "expected_archive_sha256": expected_sha256,
            }
        destination.mkdir(parents=True)

    extracted = _extract_archive(archive_path, destination=destination)
    failures.extend(extracted["failures"])
    internal: Dict[str, Any] = {}
    audit_summary: Dict[str, Any] = {}
    root_name = extracted.get("archive_root_name")
    if root_name:
        internal = _verify_internal_audit_manifest(
            destination,
            extracted["members_by_path"],
            root_name,
        )
        failures.extend(internal.get("failures") or [])
        audit_summary_path = destination / str(root_name) / "audit_summary.json"
        if audit_summary_path.is_file():
            try:
                loaded_summary = _json_load(audit_summary_path)
                if isinstance(loaded_summary, Mapping):
                    audit_summary = dict(loaded_summary)
                else:
                    failures.append("audit_summary_not_mapping")
            except Exception as exc:
                failures.append(
                    "audit_summary_parse_failed:{}".format(type(exc).__name__)
                )
    result = {
        "schema_version": 1,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "archive_path": str(archive_path),
        "archive_sha256": actual_sha,
        "expected_archive_sha256": expected_sha256,
        "archive_root_name": root_name,
        "archive_member_count": len(extracted["members_by_path"]),
        "members": sorted(
            extracted["members_by_path"].values(),
            key=lambda row: str(row["relative_path"]),
        ),
        "internal_manifest_sha256": internal.get("manifest_sha256"),
        "internal_manifest_row_count": internal.get("manifest_row_count"),
        "audit_schema_version": audit_summary.get("audit_schema_version")
        or audit_summary.get("schema_version"),
        "audit_evidence_type": audit_summary.get("evidence_type"),
        "audit_original_final_status": audit_summary.get("final_status"),
        "extracted_root": (
            str(destination / str(root_name))
            if extract_dir is not None and root_name
            else None
        ),
    }
    if temporary is not None:
        temporary.cleanup()
    return result


def verify_accepted_fitting_audit_bundle(
    archive_path: Path | str,
    *,
    extract_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify only the centrally accepted audit root."""

    return _verify_audit_bundle_archive(
        archive_path,
        expected_sha256=ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256,
        extract_dir=extract_dir,
    )


def _chain_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    fields = (
        "evidence_role",
        "absolute_historical_path",
        "snapshot_relative_path",
        "file_size",
        "file_sha256",
        "snapshot_sha256",
        "parse_status",
        "schema_or_evidence_type",
        "original_run_identity",
        "artifact_identity",
        "artifact_path",
        "producer_commit",
        "relation_supported",
        "details",
    )
    return {field: record.get(field) for field in fields}


def compute_multi_file_evidence_chain_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    ordered = sorted(
        (_chain_record(record) for record in records),
        key=lambda record: (
            _ROLE_ORDER.get(str(record.get("evidence_role")), 999),
            str(record.get("snapshot_relative_path") or ""),
            str(record.get("absolute_historical_path") or ""),
        ),
    )
    return canonical_json_sha256(ordered)


def validate_multi_file_evidence_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    audit_members: Optional[Sequence[Mapping[str, Any]]] = None,
    live_original_check: bool = False,
    require_live_original: bool = False,
) -> Dict[str, Any]:
    if any(
        isinstance(record, Mapping)
        and record.get("role_binding_schema_version")
        == ROLE_BINDING_SCHEMA_VERSION
        for record in records
    ):
        try:
            canonical_specs = canonical_resolved_role_specs_from_records(
                records,
                audit_members=audit_members,
            )
        except ValueError as exc:
            return {
                "status": "failed",
                "paper_role_binding_valid": False,
                "failures": list(
                    getattr(exc, "failures", (str(exc),))
                ),
                "resolved_role_specs": {},
                "evidence_files": list(records),
            }
        record_specs: Dict[str, Dict[str, Any]] = {}
        expected_artifact_sha256: Optional[str] = None
        for role, canonical_spec in canonical_specs.items():
            spec = dict(
                ACCEPTED_AUDIT_ROLE_SPECS.get(role)
                or SUPPORTING_AUDIT_ROLE_SPECS[role]
            )
            spec.update(canonical_spec)
            if role == "replay_summary":
                spec["required_for_paper"] = True
            record_specs[role] = spec
            record = next(
                row
                for row in records
                if isinstance(row, Mapping)
                and row.get("evidence_role") == role
            )
            if role == "artifact_identity":
                fields = record.get("normalized_fields")
                if isinstance(fields, Mapping):
                    value = fields.get("artifact_sha256")
                    if _is_sha256(value):
                        expected_artifact_sha256 = str(value)
        return validate_exact_role_evidence_chain(
            records,
            role_specs=record_specs,
            expected_artifact_sha256=(
                expected_artifact_sha256 or ACCEPTED_PHASE3C_ARTIFACT_SHA256
            ),
            audit_members=audit_members,
            live_original_check=live_original_check,
            require_live_original=require_live_original,
        )

    failures: List[str] = []
    by_role: Dict[str, List[Mapping[str, Any]]] = {}
    audit_by_path = {
        str(row.get("relative_path")): row
        for row in (audit_members or [])
        if isinstance(row, Mapping)
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append("evidence_role_record_not_mapping:{}".format(index))
            continue
        role = record.get("evidence_role")
        if role not in LEGACY_REQUIRED_EVIDENCE_ROLES:
            failures.append(
                "evidence_role_unsupported:{}".format(_bounded(role))
            )
            continue
        by_role.setdefault(str(role), []).append(record)
        if record.get("parse_status") != "ok":
            failures.append("{}_parse_status_not_ok".format(role))
        if record.get("relation_supported") != ROLE_RELATIONS[role]:
            failures.append("{}_relation_mismatch".format(role))
        if not isinstance(record.get("absolute_historical_path"), str):
            failures.append("{}_historical_path_missing".format(role))
        snapshot_relative = record.get("snapshot_relative_path")
        if (
            not isinstance(snapshot_relative, str)
            or _normalized_archive_path(snapshot_relative) is None
        ):
            failures.append("{}_snapshot_relative_path_invalid".format(role))
        if not _is_sha256(record.get("snapshot_sha256")):
            failures.append("{}_snapshot_sha256_invalid".format(role))
        if not isinstance(record.get("schema_or_evidence_type"), str) or not str(
            record.get("schema_or_evidence_type")
        ).strip():
            failures.append("{}_schema_or_evidence_type_missing".format(role))
        if not isinstance(record.get("original_run_identity"), str) or not str(
            record.get("original_run_identity")
        ).strip():
            failures.append("{}_original_run_identity_missing".format(role))
        if record.get("artifact_identity") != ACCEPTED_PHASE3C_ARTIFACT_SHA256:
            failures.append("{}_artifact_identity_mismatch".format(role))
        if not isinstance(record.get("artifact_path"), str) or not str(
            record.get("artifact_path")
        ).strip():
            failures.append("{}_artifact_path_missing".format(role))
        if not isinstance(record.get("producer_commit"), str) or not (
            _COMMIT_RE.fullmatch(str(record.get("producer_commit")))
        ):
            failures.append("{}_producer_commit_invalid".format(role))
        if not isinstance(record.get("details"), Mapping):
            failures.append("{}_details_missing".format(role))
        if not isinstance(record.get("file_size"), int) or isinstance(
            record.get("file_size"), bool
        ) or int(record.get("file_size", -1)) < 0:
            failures.append("{}_file_size_invalid".format(role))
        if not _is_sha256(record.get("file_sha256")):
            failures.append("{}_file_sha256_invalid".format(role))
        if (
            role == "external_population_comparison"
            and record.get("file_sha256")
            != ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256
        ):
            failures.append(
                "external_population_comparison_file_sha256_mismatch"
            )
        if (
            role == "artifact_identity"
            and record.get("file_sha256")
            != ACCEPTED_PHASE3C_ARTIFACT_SHA256
        ):
            failures.append("artifact_identity_file_sha256_mismatch")
        if audit_by_path:
            actual = audit_by_path.get(str(snapshot_relative))
            if actual is None:
                failures.append("{}_snapshot_missing_from_audit".format(role))
            elif record.get("snapshot_sha256") != actual.get("sha256"):
                failures.append("{}_snapshot_sha256_mismatch".format(role))
        if live_original_check:
            live_path = Path(str(record.get("absolute_historical_path") or ""))
            if live_path.is_file():
                if sha256_file(live_path) != record.get("file_sha256"):
                    failures.append("{}_live_original_sha256_mismatch".format(role))
            elif require_live_original:
                failures.append("{}_live_original_missing".format(role))

    for role in LEGACY_REQUIRED_EVIDENCE_ROLES:
        candidates = by_role.get(role) or []
        if not candidates:
            failures.append("required_evidence_role_missing:{}".format(role))
        elif len(candidates) > 1:
            failures.append("required_evidence_role_ambiguous:{}".format(role))

    child_records = by_role.get("child_final_status") or []
    if len(child_records) == 1:
        child_details = child_records[0].get("details")
        child_details = (
            child_details if isinstance(child_details, Mapping) else {}
        )
        if child_details.get("execution_status") in (None, ""):
            failures.append("child_execution_status_missing")
        if not isinstance(child_details.get("exit_code"), int) or isinstance(
            child_details.get("exit_code"), bool
        ):
            failures.append("child_exit_code_missing_or_invalid")
        for field in ("replay_matched_count", "replay_expected_count"):
            if not isinstance(child_details.get(field), int) or isinstance(
                child_details.get(field), bool
            ):
                failures.append("child_{}_missing_or_invalid".format(field))

    outer_records = by_role.get("outer_full_acceptance") or []
    if len(outer_records) == 1:
        outer_details = outer_records[0].get("details")
        outer_details = (
            outer_details if isinstance(outer_details, Mapping) else {}
        )
        if outer_details.get("acceptance_status") in (None, ""):
            failures.append("outer_acceptance_status_missing")
        if not isinstance(outer_details.get("exit_code"), int) or isinstance(
            outer_details.get("exit_code"), bool
        ):
            failures.append("outer_acceptance_exit_code_missing_or_invalid")
        outer_failures = outer_details.get("failures")
        if not isinstance(outer_failures, list):
            failures.append("outer_acceptance_failures_missing")
        else:
            for required in EXPECTED_OUTER_ACCEPTANCE_FAILURES:
                if required not in outer_failures:
                    failures.append(
                        "outer_acceptance_required_warning_missing:{}".format(
                            required
                        )
                    )

    run_identities = {
        str(record.get("original_run_identity"))
        for record in records
        if record.get("original_run_identity") not in (None, "")
    }
    artifact_identities = {
        str(record.get("artifact_identity"))
        for record in records
        if record.get("artifact_identity") not in (None, "")
    }
    producer_commits = {
        str(record.get("producer_commit"))
        for record in records
        if record.get("producer_commit") not in (None, "")
    }
    artifact_paths = {
        str(record.get("artifact_path"))
        for record in records
        if record.get("artifact_path") not in (None, "")
    }
    if len(run_identities) != 1:
        failures.append("evidence_chain_original_run_identity_not_unique")
    if artifact_identities != {ACCEPTED_PHASE3C_ARTIFACT_SHA256}:
        failures.append("evidence_chain_artifact_identity_mismatch")
    if len(producer_commits) != 1:
        failures.append("evidence_chain_producer_commit_not_unique")
    elif not _COMMIT_RE.fullmatch(next(iter(producer_commits))):
        failures.append("evidence_chain_producer_commit_invalid")
    if len(artifact_paths) > 1:
        failures.append("evidence_chain_artifact_path_relocation_unbound")

    distinct_files = {
        (record.get("absolute_historical_path"), record.get("file_sha256"))
        for record in records
        if isinstance(record, Mapping)
    }
    if len(distinct_files) < 3:
        failures.append("evidence_chain_requires_three_independent_files")

    chain_sha = compute_multi_file_evidence_chain_sha256(records)
    return {
        "schema_version": 1,
        "status": "ok" if not failures else "failed",
        "paper_role_binding_valid": False,
        "role_binding_diagnostic_only": True,
        "failures": failures,
        "required_evidence_roles": list(LEGACY_REQUIRED_EVIDENCE_ROLES),
        "evidence_files": sorted(
            (_chain_record(record) for record in records),
            key=lambda record: _ROLE_ORDER.get(
                str(record.get("evidence_role")), 999
            ),
        ),
        "multi_file_evidence_chain_sha256": chain_sha,
        "original_run_identity": (
            next(iter(run_identities)) if len(run_identities) == 1 else None
        ),
        "artifact_identity": (
            next(iter(artifact_identities))
            if len(artifact_identities) == 1
            else None
        ),
        "producer_commit": (
            next(iter(producer_commits))
            if len(producer_commits) == 1
            else None
        ),
        "artifact_path": (
            next(iter(artifact_paths)) if len(artifact_paths) == 1 else None
        ),
        "live_original_check_performed": bool(live_original_check),
    }


def load_explicit_evidence_role_inventory(
    path: Path | str,
) -> List[Dict[str, Any]]:
    payload = _json_load(Path(path))
    if isinstance(payload, Mapping):
        records = payload.get("evidence_files") or payload.get("evidence_roles")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("evidence_role_inventory_records_missing")
    return [dict(record) for record in records if isinstance(record, Mapping)]


def find_role_inventory_in_verified_audit(
    audit_verification: Mapping[str, Any],
) -> Optional[Path]:
    extracted_root = audit_verification.get("extracted_root")
    if not extracted_root:
        return None
    root = Path(str(extracted_root))
    candidates: List[Path] = []
    for pattern in (
        "*evidence_role_inventory*.json",
        "*corrective*role*.json",
        "*multi_file*evidence*.json",
    ):
        candidates.extend(root.rglob(pattern))
    explicit: List[Path] = []
    for candidate in sorted(set(candidates)):
        try:
            payload = _json_load(candidate)
        except Exception:
            continue
        rows = (
            payload.get("evidence_files") or payload.get("evidence_roles")
            if isinstance(payload, Mapping)
            else payload
        )
        if (
            isinstance(rows, list)
            and rows
            and all(
                isinstance(row, Mapping) and row.get("evidence_role")
                for row in rows
            )
        ):
            explicit.append(candidate)
    return explicit[0] if len(explicit) == 1 else None


def derive_evidence_role_inventory_from_verified_audit(
    audit_verification: Mapping[str, Any],
    *,
    live_original_check: bool = False,
    require_live_original: bool = False,
) -> Dict[str, Any]:
    """Resolve legacy audit inventory rows into the narrow corrective roles.

    Explicit role metadata wins.  This fallback is for the accepted audit
    producer, which recorded broad ``evidence_roles`` plus exact original
    paths and SHA values.  Exact sidecar basenames and semantic flags are
    both required; filename similarity alone is insufficient.
    """

    if audit_verification.get("archive_sha256") == (
        ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256
    ):
        return resolve_exact_roles_from_verified_audit(
            audit_verification,
            live_original_check=live_original_check,
            require_live_original=require_live_original,
        )

    failures: List[str] = []
    extracted_root = audit_verification.get("extracted_root")
    if not extracted_root:
        return {
            "status": "failed",
            "failures": ["audit_extracted_root_required_for_role_resolution"],
            "evidence_files": [],
        }
    root = Path(str(extracted_root))
    explicit = find_role_inventory_in_verified_audit(audit_verification)
    if explicit is not None:
        try:
            records = load_explicit_evidence_role_inventory(explicit)
        except Exception as exc:
            return {
                "status": "failed",
                "failures": [
                    "explicit_evidence_role_inventory_parse_failed:{}".format(
                        type(exc).__name__
                    )
                ],
                "evidence_files": [],
            }
        return {
            "status": "ok",
            "failures": [],
            "resolution_method": "explicit_audit_role_metadata",
            "paper_role_binding_valid": False,
            "role_inventory_path": str(explicit),
            "evidence_files": records,
        }

    inventory_path = root / "evidence_file_inventory.jsonl"
    candidate_path = root / "candidate_original_manifests.jsonl"
    audit_summary_path = root / "audit_summary.json"
    if not inventory_path.is_file():
        failures.append("audit_evidence_file_inventory_missing")
        inventory_rows: List[Any] = []
    else:
        try:
            inventory_rows = _jsonl_load(inventory_path)
        except Exception as exc:
            failures.append(
                "audit_evidence_file_inventory_parse_failed:{}".format(
                    type(exc).__name__
                )
            )
            inventory_rows = []
    try:
        candidate_rows = (
            _jsonl_load(candidate_path) if candidate_path.is_file() else []
        )
    except Exception as exc:
        failures.append(
            "audit_candidate_original_manifests_parse_failed:{}".format(
                type(exc).__name__
            )
        )
        candidate_rows = []
    try:
        audit_summary = (
            _json_load(audit_summary_path)
            if audit_summary_path.is_file()
            else {}
        )
    except Exception:
        audit_summary = {}

    candidate_by_path = {
        str(row.get("physical_path")): row
        for row in candidate_rows
        if isinstance(row, Mapping) and row.get("physical_path")
    }
    original_run_directory = audit_summary.get("original_run_directory")
    run_identity = str(original_run_directory or "")
    artifact_path = audit_summary.get("expected_artifact_path")
    artifact_identity = audit_summary.get("artifact_sha_before")
    producer_commits: set[str] = set()
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            continue
        for excerpt in row.get("excerpts") or []:
            text = excerpt.get("text") if isinstance(excerpt, Mapping) else ""
            for match in re.findall(r"\b[0-9a-fA-F]{40}\b", str(text)):
                producer_commits.add(match.lower())
    producer_commit = (
        next(iter(producer_commits)) if len(producer_commits) == 1 else None
    )

    def snapshot_for(historical_path: str) -> Optional[Path]:
        prefix = hashlib.sha256(historical_path.encode("utf-8")).hexdigest()[:12]
        candidate = (
            root
            / "evidence_snapshots"
            / "{}_{}".format(prefix, Path(historical_path).name)
        )
        return candidate if candidate.is_file() else None

    def parsed_snapshot(path: Optional[Path]) -> Any:
        if path is None:
            return None
        try:
            if path.suffix.lower() == ".json":
                return _json_load(path)
            if path.suffix.lower() == ".jsonl":
                return _jsonl_load(path)
        except Exception:
            return None
        return None

    def recursive_values(value: Any, names: set[str]) -> List[Any]:
        found: List[Any] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in names:
                    found.append(item)
                found.extend(recursive_values(item, names))
        elif isinstance(value, list):
            for item in value:
                found.extend(recursive_values(item, names))
        return found

    def score(role: str, row: Mapping[str, Any]) -> int:
        path_text = str(row.get("absolute_path") or "")
        name = Path(path_text).name.lower()
        relative = str(row.get("relative_path") or "").replace("\\", "/")
        metadata = candidate_by_path.get(path_text) or {}
        broad = set(row.get("evidence_roles") or []) | set(
            metadata.get("supports_relation") or []
        )
        in_original = bool(row.get("in_original_run"))
        value = 0
        if role == "external_population_comparison":
            return (
                100
                if row.get("sha256")
                == ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256
                else 0
            )
        if role == "original_fitting_command":
            if name == "command.txt" and in_original:
                value += 80
            if "fitting_invocation" in broad:
                value += 20
        elif role == "fit_summary":
            if name == "phase3c_final_policy_summary.json" and in_original:
                value += 100
        elif role == "artifact_summary":
            if (
                name == "calm_hybrid_multisource_policy_artifact_summary.json"
                and in_original
            ):
                value += 100
        elif role == "artifact_identity":
            if (
                row.get("sha256") == ACCEPTED_PHASE3C_ARTIFACT_SHA256
                and in_original
            ):
                value += 100
        elif role == "child_final_status":
            if name == "final_status.json" and in_original:
                value += 40
                if "calm_hybrid_multisource_artifact_" in relative:
                    value += 60
        elif role == "outer_full_acceptance":
            if name == "final_status.json" and in_original:
                value += 40
                if str(Path(relative).parent).replace("\\", "/") in ("", "."):
                    value += 60
        elif role == "producer_git_identity":
            if name in {
                "initial_repository_state.json",
                "repository_run_manifest.json",
                "run_start_manifest.json",
            } and in_original:
                value += 100
        elif role == "return_code_sidecar":
            if name in {
                "return_codes.json",
                "top_level_return_code.txt",
                "child_return_code.txt",
            } and in_original:
                value += 100
        elif role == "file_manifest":
            if name in {"file_manifest.tsv", "file_manifest.jsonl"} and in_original:
                value += 100
        return value

    records: List[Dict[str, Any]] = []
    for role in LEGACY_REQUIRED_EVIDENCE_ROLES:
        if role == "current_artifact_preflight":
            generated = root / "artifact_metadata_summary.json"
            if generated.is_file():
                records.append(
                    {
                        "evidence_role": role,
                        "absolute_historical_path": str(generated),
                        "snapshot_relative_path": (
                            "{}/artifact_metadata_summary.json".format(
                                audit_verification.get("archive_root_name")
                            )
                        ),
                        "file_size": generated.stat().st_size,
                        "file_sha256": sha256_file(generated),
                        "snapshot_sha256": sha256_file(generated),
                        "parse_status": "ok",
                        "schema_or_evidence_type": (
                            "phase3c_fitting_provenance_artifact_metadata_summary"
                        ),
                        "original_run_identity": run_identity,
                        "artifact_identity": artifact_identity,
                        "artifact_path": artifact_path,
                        "producer_commit": producer_commit,
                        "relation_supported": ROLE_RELATIONS[role],
                        "details": {
                            "file_sha256": sha256_file(generated),
                        },
                    }
                )
                continue
        candidates = [
            (score(role, row), row)
            for row in inventory_rows
            if isinstance(row, Mapping) and score(role, row) > 0
        ]
        if not candidates:
            failures.append("required_evidence_role_unresolved:{}".format(role))
            continue
        best_score = max(value for value, _ in candidates)
        best = [row for value, row in candidates if value == best_score]
        if len(best) != 1:
            failures.append("required_evidence_role_ambiguous:{}".format(role))
            continue
        row = best[0]
        historical_path = str(row.get("absolute_path"))
        metadata = candidate_by_path.get(historical_path) or {}
        snapshot = snapshot_for(historical_path)
        parsed = parsed_snapshot(snapshot)
        statuses = recursive_values(
            parsed,
            {
                "status",
                "final_status",
                "execution_status",
                "acceptance_status",
            },
        )
        exit_codes = recursive_values(
            parsed,
            {
                "exit_code",
                "return_code",
                "overall_return_code",
                "child_return_code",
            },
        )
        replay_matched = recursive_values(
            parsed,
            {"replay_matched_count", "matched_count"},
        )
        replay_expected = recursive_values(
            parsed,
            {"replay_expected_count", "expected_count"},
        )
        failure_lists = recursive_values(
            parsed,
            {"failures", "errors", "failure_reasons"},
        )
        producer_branches = recursive_values(
            parsed,
            {
                "git_branch",
                "branch",
                "starting_git_branch",
                "repository_branch",
            },
        )
        details = {
            "file_sha256": row.get("sha256"),
            "historical_path": historical_path,
            "execution_status": statuses[0] if statuses else None,
            "acceptance_status": statuses[0] if statuses else None,
            "exit_code": exit_codes[0] if exit_codes else None,
            "replay_matched_count": (
                replay_matched[0] if replay_matched else None
            ),
            "replay_expected_count": (
                replay_expected[0] if replay_expected else None
            ),
            "failures": next(
                (value for value in failure_lists if isinstance(value, list)),
                [],
            ),
            "original_run_directory": original_run_directory,
            "producer_branch": (
                audit_summary.get("repository_branch")
                or (producer_branches[0] if producer_branches else None)
            ),
        }
        archive_root = str(audit_verification.get("archive_root_name"))
        if snapshot is not None:
            snapshot_relative = "{}/{}".format(
                archive_root,
                snapshot.relative_to(root).as_posix(),
            )
            snapshot_sha = sha256_file(snapshot)
            parse_status = (
                "ok"
                if parsed is not None or metadata
                else "unparsed"
            )
        else:
            snapshot_relative = "{}/evidence_file_inventory.jsonl".format(
                archive_root
            )
            snapshot_sha = (
                sha256_file(inventory_path) if inventory_path.is_file() else None
            )
            parse_status = (
                "ok" if metadata or row.get("evidence_roles") else "unknown"
            )
        records.append(
            {
                "evidence_role": role,
                "absolute_historical_path": historical_path,
                "snapshot_relative_path": snapshot_relative,
                "file_size": row.get("size_bytes"),
                "file_sha256": row.get("sha256"),
                "snapshot_sha256": snapshot_sha,
                "parse_status": parse_status,
                "schema_or_evidence_type": Path(historical_path).name,
                "original_run_identity": run_identity,
                "artifact_identity": artifact_identity,
                "artifact_path": artifact_path,
                "producer_commit": producer_commit,
                "relation_supported": ROLE_RELATIONS[role],
                "details": details,
            }
        )
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "resolution_method": (
            "accepted_audit_inventory_semantics_and_exact_sidecar_names"
        ),
        "paper_role_binding_valid": False,
        "role_inventory_path": str(inventory_path),
        "evidence_files": records,
    }


def parse_external_population_comparison(
    path: Path | str,
    *,
    expected_sha256: str = ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256,
    expected_fitting_count: int = EXPECTED_ARTIFACT_FITTING_COUNT,
    expected_heldout_count: int = EXPECTED_ARTIFACT_HELDOUT_COUNT,
    expected_union_count: int = EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
    expected_historical_digests: Optional[Mapping[str, str]] = None,
    expected_canonical_sets: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Parse the historical comparison summary's frozen split."""

    target = Path(path)
    failures: List[str] = []
    if not target.is_file():
        return {
            "status": "failed",
            "failures": ["external_population_evidence_missing"],
            "source_path": str(target),
        }
    actual_sha = sha256_file(target)
    if actual_sha != expected_sha256:
        failures.append("external_population_evidence_sha256_mismatch")
    try:
        payload = _json_load(target)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": failures
            + [
                "external_population_evidence_parse_failed:{}".format(
                    type(exc).__name__
                )
            ],
            "source_path": str(target),
            "source_sha256": actual_sha,
        }
    split = payload.get("frozen_split")
    if not isinstance(split, Mapping):
        failures.append("external_population_frozen_split_missing")
        split = {}
    raw_fitting_ids = split.get("train_stable_sample_ids")
    raw_heldout_ids = split.get("eval_stable_sample_ids")
    if not isinstance(raw_fitting_ids, list):
        failures.append("external_population_fitting_ids_invalid_type")
        raw_fitting_ids = []
    if not isinstance(raw_heldout_ids, list):
        failures.append("external_population_heldout_ids_invalid_type")
        raw_heldout_ids = []
    if any(
        not isinstance(value, str) or not value
        for value in raw_fitting_ids
    ):
        failures.append("external_population_fitting_id_invalid")
    if any(
        not isinstance(value, str) or not value
        for value in raw_heldout_ids
    ):
        failures.append("external_population_heldout_id_invalid")
    fitting_ids = [str(value) for value in raw_fitting_ids]
    heldout_ids = [str(value) for value in raw_heldout_ids]
    duplicate_fitting = sum(
        max(0, count - 1) for count in Counter(fitting_ids).values()
    )
    duplicate_heldout = sum(
        max(0, count - 1) for count in Counter(heldout_ids).values()
    )
    fitting_set = set(fitting_ids)
    heldout_set = set(heldout_ids)
    intersection = fitting_set & heldout_set
    union = fitting_set | heldout_set
    if len(fitting_ids) != expected_fitting_count:
        failures.append("external_population_fitting_count_mismatch")
    if len(heldout_ids) != expected_heldout_count:
        failures.append("external_population_heldout_count_mismatch")
    if duplicate_fitting:
        failures.append("external_population_fitting_duplicates")
    if duplicate_heldout:
        failures.append("external_population_heldout_duplicates")
    if intersection:
        failures.append("external_population_intersection_nonzero")
    if len(union) != expected_union_count:
        failures.append("external_population_union_count_mismatch")
    for field, actual in (
        ("train_stable_sample_id_count", len(fitting_ids)),
        ("eval_stable_sample_id_count", len(heldout_ids)),
        ("aligned_stable_sample_id_count", len(union)),
        ("train_eval_overlap_count", len(intersection)),
    ):
        declared = split.get(field)
        if (
            not isinstance(declared, int)
            or isinstance(declared, bool)
            or declared != actual
        ):
            failures.append(
                "external_population_declared_{}_mismatch".format(field)
            )

    historical = {
        "train": historical_identity_digest(fitting_ids),
        "eval": historical_identity_digest(heldout_ids),
        "aligned": historical_identity_digest(sorted(union)),
    }
    stored_historical = {
        "train": split.get("train_stable_sample_id_sha256"),
        "eval": split.get("eval_stable_sample_id_sha256"),
        "aligned": split.get("aligned_stable_sample_id_sha256"),
    }
    expected_historical = dict(
        expected_historical_digests
        or {
            "train": EXPECTED_HISTORICAL_TRAIN_DIGEST,
            "eval": EXPECTED_HISTORICAL_EVAL_DIGEST,
            "aligned": EXPECTED_HISTORICAL_ALIGNED_DIGEST,
        }
    )
    for role in ("train", "eval", "aligned"):
        if stored_historical[role] != historical[role]:
            failures.append(
                "external_population_historical_{}_digest_mismatch".format(role)
            )
        if historical[role] != expected_historical[role]:
            failures.append(
                "external_population_historical_{}_digest_not_frozen".format(
                    role
                )
            )

    canonical = {
        "fitting": (
            stable_sample_ids_sha256(fitting_ids, sort_ids=True)
            if not duplicate_fitting
            else None
        ),
        "heldout": (
            stable_sample_ids_sha256(heldout_ids, sort_ids=True)
            if not duplicate_heldout
            else None
        ),
        "union": stable_sample_ids_sha256(sorted(union), sort_ids=True),
    }
    expected_sets = dict(
        expected_canonical_sets
        or {
            "fitting": EXPECTED_CANONICAL_FITTING_SET_SHA256,
            "heldout": EXPECTED_CANONICAL_HELDOUT_SET_SHA256,
            "union": EXPECTED_CANONICAL_UNION_SET_SHA256,
        }
    )
    for role in ("fitting", "heldout", "union"):
        if canonical[role] != expected_sets[role]:
            failures.append(
                "external_population_canonical_{}_set_sha256_mismatch".format(
                    role
                )
            )

    for field, expected in (
        ("split_algorithm", EXPECTED_SPLIT_ALGORITHM),
        ("split_unit", EXPECTED_SPLIT_UNIT),
        ("split_mode", EXPECTED_SPLIT_MODE),
        ("split_ratio", EXPECTED_SPLIT_RATIO),
        ("split_seed", EXPECTED_SPLIT_SEED),
    ):
        if split.get(field) != expected:
            failures.append(
                "external_population_{}_mismatch".format(field)
            )
    provenance = payload.get("provenance") or {}
    return {
        "schema_version": 1,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "source_path": str(target),
        "source_sha256": actual_sha,
        "source_size_bytes": target.stat().st_size,
        "artifact_fitting_stable_sample_ids": fitting_ids,
        "artifact_heldout_stable_sample_ids": heldout_ids,
        "artifact_fitting_count": len(fitting_ids),
        "artifact_heldout_count": len(heldout_ids),
        "duplicate_fitting_count": duplicate_fitting,
        "duplicate_heldout_count": duplicate_heldout,
        "intersection_count": len(intersection),
        "union_count": len(union),
        "historical_train_digest": historical["train"],
        "historical_eval_digest": historical["eval"],
        "historical_aligned_digest": historical["aligned"],
        "stored_historical_train_digest": stored_historical["train"],
        "stored_historical_eval_digest": stored_historical["eval"],
        "stored_historical_aligned_digest": stored_historical["aligned"],
        "canonical_fitting_set_sha256": canonical["fitting"],
        "canonical_heldout_set_sha256": canonical["heldout"],
        "canonical_union_set_sha256": canonical["union"],
        "historical_digest_algorithm_identity": (
            HISTORICAL_DIGEST_ALGORITHM_IDENTITY
        ),
        "historical_digest_serialization": dict(
            HISTORICAL_DIGEST_SERIALIZATION
        ),
        "historical_digest_is_canonical_lf_set_sha": False,
        "historical_and_canonical_algorithms_distinct": True,
        "split_algorithm": split.get("split_algorithm"),
        "split_unit": split.get("split_unit"),
        "split_mode": split.get("split_mode"),
        "split_ratio": split.get("split_ratio"),
        "split_seed": split.get("split_seed"),
        "dataset_name": provenance.get("dataset_name"),
        "dataset_config_name": provenance.get("dataset_config_name"),
        "dataset_split": provenance.get("dataset_split"),
        "original_run_identifier": provenance.get("run_identity")
        or provenance.get("dump_run_binding_sha256"),
        "original_producer_commit": provenance.get("git_commit")
        or provenance.get("starting_git_commit"),
    }


def _normalized_split_commitment(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "train_count": payload.get("train_stable_sample_id_count"),
        "eval_count": payload.get("eval_stable_sample_id_count"),
        "aligned_count": payload.get("aligned_stable_sample_id_count"),
        "train_digest": payload.get("train_stable_sample_id_sha256"),
        "eval_digest": payload.get("eval_stable_sample_id_sha256"),
        "aligned_digest": payload.get("aligned_stable_sample_id_sha256"),
        "overlap_count": payload.get("train_eval_overlap_count"),
        "split_algorithm": payload.get("split_algorithm"),
        "split_unit": payload.get("split_unit"),
        "split_mode": payload.get("split_mode"),
        "split_ratio": payload.get("split_ratio"),
        "split_seed": payload.get("split_seed"),
    }


def verify_accepted_artifact_split_commitment(
    artifact_path: Path | str,
    *,
    _loader: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    _expected_artifact_sha256: Optional[str] = None,
    _expected_commitment: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Read-only accepted-artifact split verification.

    Underscored overrides exist only for small CPU fixtures.  The CLI does not
    expose them.
    """

    target = Path(artifact_path)
    failures: List[str] = []
    expected_sha = (
        _expected_artifact_sha256 or ACCEPTED_PHASE3C_ARTIFACT_SHA256
    )
    if not target.is_file():
        return {
            "status": "failed",
            "failures": ["accepted_artifact_missing"],
            "artifact_path": str(target),
        }
    before = sha256_file(target)
    if before != expected_sha:
        return {
            "schema_version": 1,
            "status": "failed",
            "failures": ["accepted_artifact_sha256_mismatch"],
            "artifact_path": str(target),
            "artifact_size": target.stat().st_size,
            "accepted_artifact_sha256": expected_sha,
            "artifact_file_sha256_before_load": before,
            "artifact_file_sha256_after_load": None,
            "artifact_load_preserved_bytes": None,
            "accepted_artifact_resaved": False,
        }
    if _loader is None:
        from .phase3c_policy_artifact import load_phase3c_policy_artifact

        _loader = load_phase3c_policy_artifact
    try:
        artifact = _loader(target)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": failures
            + ["accepted_artifact_load_failed:{}".format(type(exc).__name__)],
            "artifact_path": str(target),
            "artifact_file_sha256_before_load": before,
        }
    after = sha256_file(target)
    if after != before:
        failures.append("accepted_artifact_bytes_changed_during_load")
    if not isinstance(artifact, Mapping):
        failures.append("accepted_artifact_payload_not_mapping")
        artifact = {}
    fit_config = artifact.get("fit_config")
    provenance = artifact.get("provenance")
    fit_config = fit_config if isinstance(fit_config, Mapping) else {}
    provenance = provenance if isinstance(provenance, Mapping) else {}
    artifact_provenance_identities = extract_artifact_provenance_identities(
        artifact
    )
    if (
        _expected_artifact_sha256 is None
        and artifact_provenance_identities.get("status") != "ok"
    ):
        failures.extend(
            artifact_provenance_identities.get("failures") or []
        )
    candidates: List[tuple[str, Mapping[str, Any]]] = []
    fit_split = fit_config.get("split_identity")
    if isinstance(fit_split, Mapping):
        candidates.append(("$.fit_config.split_identity", fit_split))
    calm_diag = provenance.get("calm_example_diagnostics")
    if isinstance(calm_diag, Mapping):
        diag_split = calm_diag.get("calm_hybrid_frozen_split_identity")
        if isinstance(diag_split, Mapping):
            candidates.append(
                (
                    "$.provenance.calm_example_diagnostics."
                    "calm_hybrid_frozen_split_identity",
                    diag_split,
                )
            )
    if not candidates:
        failures.append("artifact_internal_split_commitment_missing")
    normalized = [
        (field_path, _normalized_split_commitment(payload))
        for field_path, payload in candidates
    ]
    unique_payloads = {
        canonical_json_sha256(payload) for _, payload in normalized
    }
    if len(unique_payloads) > 1:
        failures.append("artifact_internal_split_commitment_conflict")
    commitment = normalized[0][1] if normalized else {}
    expected = dict(
        _expected_commitment
        or {
            "train_count": EXPECTED_ARTIFACT_FITTING_COUNT,
            "eval_count": EXPECTED_ARTIFACT_HELDOUT_COUNT,
            "aligned_count": EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
            "train_digest": EXPECTED_HISTORICAL_TRAIN_DIGEST,
            "eval_digest": EXPECTED_HISTORICAL_EVAL_DIGEST,
            "aligned_digest": EXPECTED_HISTORICAL_ALIGNED_DIGEST,
            "overlap_count": 0,
            "split_algorithm": EXPECTED_SPLIT_ALGORITHM,
            "split_unit": EXPECTED_SPLIT_UNIT,
            "split_mode": EXPECTED_SPLIT_MODE,
            "split_ratio": EXPECTED_SPLIT_RATIO,
            "split_seed": EXPECTED_SPLIT_SEED,
        }
    )
    for field, expected_value in expected.items():
        if commitment.get(field) != expected_value:
            failures.append(
                "artifact_internal_split_{}_mismatch".format(field)
            )

    runtime_semantics = fit_config.get("candidate_first_crossing_semantics")
    runtime_semantics = (
        runtime_semantics if isinstance(runtime_semantics, Mapping) else {}
    )
    hybrid_semantics = fit_config.get("calm_hybrid_policy_semantics")
    hybrid_semantics = (
        hybrid_semantics if isinstance(hybrid_semantics, Mapping) else {}
    )
    runtime_sha = runtime_semantics.get("policy_sha256") or hybrid_semantics.get(
        "calm_runtime_policy_sha256"
    )
    hybrid_sha = fit_config.get("calm_hybrid_policy_sha256")
    if runtime_sha != ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256:
        failures.append("artifact_runtime_policy_sha256_mismatch")
    if hybrid_sha != ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256:
        failures.append("artifact_hybrid_fitting_policy_sha256_mismatch")

    return {
        "schema_version": 1,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "artifact_path": str(target),
        "artifact_size": target.stat().st_size,
        "accepted_artifact_sha256": expected_sha,
        "artifact_file_sha256_before_load": before,
        "artifact_file_sha256_after_load": after,
        "artifact_load_preserved_bytes": before == after,
        "accepted_artifact_resaved": False,
        "artifact_internal_split_field_paths": [
            field_path for field_path, _ in candidates
        ],
        "artifact_internal_split_payloads": [
            {"field_path": field_path, "payload": dict(payload)}
            for field_path, payload in candidates
        ],
        "artifact_metadata_schema_version": artifact.get("schema_version"),
        "artifact_train_count": commitment.get("train_count"),
        "artifact_eval_count": commitment.get("eval_count"),
        "artifact_aligned_count": commitment.get("aligned_count"),
        "artifact_historical_train_digest": commitment.get("train_digest"),
        "artifact_historical_eval_digest": commitment.get("eval_digest"),
        "artifact_historical_aligned_digest": commitment.get("aligned_digest"),
        "artifact_train_eval_overlap_count": commitment.get("overlap_count"),
        "split_algorithm": commitment.get("split_algorithm"),
        "split_unit": commitment.get("split_unit"),
        "split_mode": commitment.get("split_mode"),
        "split_ratio": commitment.get("split_ratio"),
        "split_seed": commitment.get("split_seed"),
        "runtime_policy_sha256": runtime_sha,
        "hybrid_fitting_policy_sha256": hybrid_sha,
        "original_run_identifier": artifact_provenance_identities.get(
            "artifact_original_run_identity"
        ),
        "original_producer_commit": artifact_provenance_identities.get(
            "artifact_original_producer_commit"
        )
        or provenance.get("git_commit")
        or provenance.get("repository_commit"),
        "dump_run_binding": artifact_provenance_identities.get(
            "dump_run_binding"
        )
        or provenance.get("dump_run_binding"),
        "dump_run_binding_identity": artifact_provenance_identities.get(
            "dump_run_binding_identity"
        ),
        "checkpoint_identity": artifact_provenance_identities.get(
            "checkpoint_identity"
        )
        or provenance.get("model_checkpoint_identity_sha256"),
        "tokenizer_identity": artifact_provenance_identities.get(
            "tokenizer_identity"
        )
        or provenance.get("tokenizer_identity_sha256"),
        "artifact_original_run_identity": artifact_provenance_identities.get(
            "artifact_original_run_identity"
        ),
        "artifact_original_run_identity_present": (
            artifact_provenance_identities.get(
                "artifact_original_run_identity_present"
            )
        ),
        "artifact_original_run_identity_status": (
            artifact_provenance_identities.get(
                "artifact_original_run_identity_status"
            )
        ),
        "artifact_original_producer_commit": (
            artifact_provenance_identities.get(
                "artifact_original_producer_commit"
            )
        ),
        "artifact_identity_field_provenance": (
            artifact_provenance_identities.get(
                "normalized_field_provenance"
            )
        ),
    }


def verify_historical_digest_binding(
    population: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    _expected_historical_digests: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    expected = dict(
        _expected_historical_digests
        or {
            "train": EXPECTED_HISTORICAL_TRAIN_DIGEST,
            "eval": EXPECTED_HISTORICAL_EVAL_DIGEST,
            "aligned": EXPECTED_HISTORICAL_ALIGNED_DIGEST,
        }
    )
    pairs = (
        (
            "train",
            population.get("historical_train_digest"),
            artifact.get("artifact_historical_train_digest"),
            expected["train"],
        ),
        (
            "eval",
            population.get("historical_eval_digest"),
            artifact.get("artifact_historical_eval_digest"),
            expected["eval"],
        ),
        (
            "aligned",
            population.get("historical_aligned_digest"),
            artifact.get("artifact_historical_aligned_digest"),
            expected["aligned"],
        ),
    )
    result: Dict[str, Any] = {}
    for role, recomputed, stored, frozen in pairs:
        match = recomputed == stored == frozen
        result["historical_{}_digest_match".format(role)] = match
        if not match:
            failures.append("historical_{}_digest_binding_mismatch".format(role))
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        **result,
    }


def resolve_authoritative_samsum_population_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
    fitting_stable_sample_ids: Sequence[Any],
    *,
    _allow_precomputed_stable_sample_id: bool = False,
    _expected_dataset_count: int = EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT,
    _expected_fitting_count: int = EXPECTED_ARTIFACT_FITTING_COUNT,
    _expected_heldout_count: int = EXPECTED_ARTIFACT_HELDOUT_COUNT,
    _expected_canonical_sets: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve fitting and held-out populations in authoritative dataset order."""

    records = build_dataset_population_records(
        dataset_rows,
        dataset_name=EXPECTED_DATASET_NAME,
        dataset_config_name=EXPECTED_DATASET_CONFIG_NAME,
        dataset_split=EXPECTED_DATASET_SPLIT,
        text_column=EXPECTED_TEXT_COLUMN,
        summary_column=EXPECTED_SUMMARY_COLUMN,
        allow_precomputed_stable_sample_id=(
            _allow_precomputed_stable_sample_id
        ),
    )
    split_source = {
        "source": "corrective_fitting_provenance_external_population_evidence",
        "accepted_artifact_file_sha256": ACCEPTED_PHASE3C_ARTIFACT_SHA256,
        "runtime_policy_sha256": ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256,
        "hybrid_fitting_policy_sha256": (
            ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256
        ),
        "stable_sample_id_algorithm_identity": (
            STABLE_SAMPLE_ID_ALGORITHM_IDENTITY
        ),
        "dataset_name": EXPECTED_DATASET_NAME,
        "dataset_config_name": EXPECTED_DATASET_CONFIG_NAME,
        "dataset_split": EXPECTED_DATASET_SPLIT,
        "text_column": EXPECTED_TEXT_COLUMN,
        "summary_column": EXPECTED_SUMMARY_COLUMN,
    }
    resolved = resolve_artifact_heldout_population(
        records,
        fitting_stable_sample_ids,
        split_source=split_source,
        expected_dataset_count=_expected_dataset_count,
        expected_fitting_count=_expected_fitting_count,
        expected_heldout_count=_expected_heldout_count,
    )
    failures = list(resolved.get("failures") or [])
    fitting_ids = [str(value) for value in fitting_stable_sample_ids]
    heldout_records = list(resolved.get("heldout_records") or [])
    heldout_ids = [str(row.get("stable_sample_id")) for row in heldout_records]
    union_ids = sorted(set(fitting_ids) | set(heldout_ids))

    def canonical_sha(
        values: Sequence[str],
        *,
        role: str,
        sort_ids: bool,
    ) -> Optional[str]:
        try:
            return stable_sample_ids_sha256(values, sort_ids=sort_ids)
        except Exception as exc:
            failures.append(
                "authoritative_population_{}_identity_invalid:{}".format(
                    role,
                    type(exc).__name__,
                )
            )
            return None

    canonical = {
        "fitting": canonical_sha(
            fitting_ids,
            role="fitting",
            sort_ids=True,
        ),
        "heldout": canonical_sha(
            heldout_ids,
            role="heldout",
            sort_ids=True,
        ),
        "union": canonical_sha(
            union_ids,
            role="union",
            sort_ids=True,
        ),
    }
    expected_sets = dict(
        _expected_canonical_sets
        or {
            "fitting": EXPECTED_CANONICAL_FITTING_SET_SHA256,
            "heldout": EXPECTED_CANONICAL_HELDOUT_SET_SHA256,
            "union": EXPECTED_CANONICAL_UNION_SET_SHA256,
        }
    )
    for role in ("fitting", "heldout", "union"):
        if canonical[role] != expected_sets[role]:
            failures.append(
                "authoritative_population_canonical_{}_set_sha256_mismatch".format(
                    role
                )
            )
    heldout_ordered_sha = canonical_sha(
        heldout_ids,
        role="heldout_dataset_order",
        sort_ids=False,
    )
    return {
        **resolved,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "artifact_fitting_stable_sample_ids": fitting_ids,
        "artifact_heldout_stable_sample_ids": heldout_ids,
        "canonical_fitting_set_sha256": canonical["fitting"],
        "canonical_heldout_set_sha256": canonical["heldout"],
        "canonical_union_set_sha256": canonical["union"],
        "heldout_stable_sample_set_sha256": canonical["heldout"],
        "heldout_stable_sample_dataset_order_sha256": heldout_ordered_sha,
        "canonical_heldout_dataset_order_sha256": heldout_ordered_sha,
        "stable_sample_id_algorithm_identity": (
            STABLE_SAMPLE_ID_ALGORITHM_IDENTITY
        ),
        "dataset_name": EXPECTED_DATASET_NAME,
        "dataset_config_name": EXPECTED_DATASET_CONFIG_NAME,
        "dataset_split": EXPECTED_DATASET_SPLIT,
        "text_column": EXPECTED_TEXT_COLUMN,
        "summary_column": EXPECTED_SUMMARY_COLUMN,
    }


def load_and_resolve_authoritative_samsum_population(
    fitting_stable_sample_ids: Sequence[Any],
) -> Dict[str, Any]:
    """Paper-facing dataset resolver.  No synthetic adapter is accepted."""

    from datasets import load_dataset

    dataset = load_dataset(
        EXPECTED_DATASET_NAME,
        EXPECTED_DATASET_CONFIG_NAME,
        split=EXPECTED_DATASET_SPLIT,
    )
    return resolve_authoritative_samsum_population_rows(
        [dict(row) for row in dataset],
        fitting_stable_sample_ids,
    )


def _role_details(
    evidence_chain: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    for record in evidence_chain.get("evidence_files") or []:
        if record.get("evidence_role") == role:
            details = record.get("details")
            return details if isinstance(details, Mapping) else {}
    return {}


def _role_record(
    evidence_chain: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    for record in evidence_chain.get("evidence_files") or []:
        if record.get("evidence_role") == role:
            return record
    return {}


def _artifact_internal_run_identity_observation(
    artifact_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    failures: List[str] = []
    identity = artifact_verification.get("artifact_original_run_identity")
    identity_present = identity not in (None, "")
    if identity_present and (
        not isinstance(identity, str) or not identity.strip()
    ):
        failures.append("artifact_internal_original_run_identity_invalid")
        identity = None
        identity_present = False

    declared_present = artifact_verification.get(
        "artifact_original_run_identity_present"
    )
    if declared_present is not None and declared_present is not identity_present:
        failures.append(
            "artifact_internal_original_run_identity_presence_mismatch"
        )
    declared_status = artifact_verification.get(
        "artifact_original_run_identity_status"
    )
    expected_status = (
        "found_and_verified" if identity_present else "not_found"
    )
    if (
        declared_status is not None
        and declared_status != expected_status
    ):
        failures.append(
            "artifact_internal_original_run_identity_status_mismatch"
        )

    field_provenance = artifact_verification.get(
        "artifact_identity_field_provenance"
    )
    field_provenance = (
        dict(field_provenance)
        if isinstance(field_provenance, Mapping)
        else {}
    )
    identity_field_path = field_provenance.get(
        "artifact_original_run_identity"
    )
    if identity_present and identity_field_path in (None, ""):
        failures.append(
            "artifact_internal_original_run_identity_field_path_missing"
        )
    if not identity_present and identity_field_path not in (None, ""):
        failures.append(
            "artifact_internal_original_run_identity_absence_provenance_mismatch"
        )

    observation = {
        "accepted_artifact_sha256": artifact_verification.get(
            "accepted_artifact_sha256"
        ),
        "artifact_internal_original_run_identity_present": identity_present,
        "artifact_internal_original_run_identity": identity,
        "artifact_internal_original_run_identity_status": expected_status,
        "artifact_internal_original_run_identity_field_path": (
            identity_field_path
        ),
    }
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "observation": observation,
        "observation_sha256": canonical_json_sha256(observation),
    }


def _load_artifact_run_identity_observation_read_only(
    artifact_path: Path | str,
    *,
    expected_artifact_sha256: str,
    _loader: Optional[Callable[[Path], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    target = Path(artifact_path)
    failures: List[str] = []
    if not target.is_file():
        return {
            "status": "failed",
            "failures": ["accepted_artifact_file_missing"],
            "artifact_path": str(target),
            "artifact_file_sha256_before_load": None,
            "artifact_file_sha256_after_load": None,
            "artifact_load_preserved_bytes": None,
            "artifact_observation": {},
            "artifact_observation_sha256": None,
        }
    before = sha256_file(target)
    if before != expected_artifact_sha256:
        return {
            "status": "failed",
            "failures": ["accepted_artifact_sha256_before_load_mismatch"],
            "artifact_path": str(target),
            "artifact_file_sha256_before_load": before,
            "artifact_file_sha256_after_load": None,
            "artifact_load_preserved_bytes": None,
            "artifact_observation": {},
            "artifact_observation_sha256": None,
        }

    if _loader is None:
        from .phase3c_policy_artifact import load_phase3c_policy_artifact

        _loader = load_phase3c_policy_artifact
    artifact: Mapping[str, Any] = {}
    try:
        loaded = _loader(target)
        if not isinstance(loaded, Mapping):
            failures.append("accepted_artifact_payload_not_mapping")
        else:
            artifact = loaded
    except Exception as exc:
        failures.append(
            "accepted_artifact_read_only_load_failed:{}".format(
                type(exc).__name__
            )
        )

    after = sha256_file(target) if target.is_file() else None
    if after != before:
        failures.append("accepted_artifact_bytes_changed_during_load")
    if after != expected_artifact_sha256:
        failures.append("accepted_artifact_sha256_after_load_mismatch")

    identities: Mapping[str, Any] = {}
    if artifact:
        identities = extract_artifact_provenance_identities(artifact)
        if identities.get("status") != "ok":
            failures.extend(identities.get("failures") or [])
    artifact_verification = {
        "accepted_artifact_sha256": expected_artifact_sha256,
        "artifact_original_run_identity": identities.get(
            "artifact_original_run_identity"
        ),
        "artifact_original_run_identity_present": identities.get(
            "artifact_original_run_identity_present"
        ),
        "artifact_original_run_identity_status": identities.get(
            "artifact_original_run_identity_status"
        ),
        "artifact_identity_field_provenance": identities.get(
            "normalized_field_provenance"
        ),
    }
    observation = _artifact_internal_run_identity_observation(
        artifact_verification
    )
    if observation.get("status") != "ok":
        failures.extend(observation.get("failures") or [])
    return {
        "status": "ok" if not failures else "failed",
        "failures": sorted(set(failures)),
        "artifact_path": str(target),
        "artifact_file_sha256_before_load": before,
        "artifact_file_sha256_after_load": after,
        "artifact_load_preserved_bytes": before == after,
        "artifact_observation": dict(
            observation.get("observation") or {}
        ),
        "artifact_observation_sha256": observation.get(
            "observation_sha256"
        ),
    }


def _validate_external_artifact_verification_authority(
    verification: Mapping[str, Any],
    *,
    expected_artifact_path: Optional[str],
    expected_artifact_sha256: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    if verification.get("status") != "ok":
        failures.append("external_artifact_verification_status_not_ok")
    if _normalized_local_artifact_path(
        verification.get("artifact_path")
    ) != _normalized_local_artifact_path(expected_artifact_path):
        failures.append("external_artifact_verification_path_mismatch")
    if verification.get(
        "accepted_artifact_sha256"
    ) != expected_artifact_sha256:
        failures.append("external_artifact_verification_sha256_mismatch")
    before = verification.get("artifact_file_sha256_before_load")
    after = verification.get("artifact_file_sha256_after_load")
    if (
        before != expected_artifact_sha256
        or after != expected_artifact_sha256
        or before != after
    ):
        failures.append(
            "external_artifact_verification_before_after_sha256_mismatch"
        )
    if verification.get("artifact_load_preserved_bytes") is not True:
        failures.append(
            "external_artifact_verification_load_preserved_bytes_not_true"
        )
    observation = _artifact_internal_run_identity_observation(verification)
    if observation.get("status") != "ok":
        failures.extend(observation.get("failures") or [])
    return {
        "status": "ok" if not failures else "failed",
        "failures": sorted(set(failures)),
        "artifact_path": verification.get("artifact_path"),
        "artifact_file_sha256_before_load": before,
        "artifact_file_sha256_after_load": after,
        "artifact_load_preserved_bytes": before == after,
        "artifact_observation": dict(
            observation.get("observation") or {}
        ),
        "artifact_observation_sha256": observation.get(
            "observation_sha256"
        ),
    }


def _corrective_child_run_identity_binding(
    evidence_chain: Mapping[str, Any],
    artifact_verification: Mapping[str, Any],
    *,
    allow_legacy_role_fixture: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    chain_child_run = evidence_chain.get("original_child_run_identifier")
    if allow_legacy_role_fixture and chain_child_run in (None, ""):
        chain_child_run = evidence_chain.get("original_run_identity")
    child_run_directory = evidence_chain.get(
        "original_child_run_directory"
    )
    child_run_directory_identity = evidence_chain.get(
        "child_run_directory_identity"
    )

    artifact_observation = _artifact_internal_run_identity_observation(
        artifact_verification
    )
    if artifact_observation.get("status") != "ok":
        failures.extend(artifact_observation.get("failures") or [])
    observed = dict(artifact_observation.get("observation") or {})
    artifact_internal_identity = observed.get(
        "artifact_internal_original_run_identity"
    )

    if not allow_legacy_role_fixture:
        if evidence_chain.get("status") != "ok":
            failures.append("exact_evidence_chain_status_not_ok")
        if evidence_chain.get("paper_role_binding_valid") is not True:
            failures.append("exact_evidence_chain_paper_role_binding_not_valid")
        if (
            not isinstance(chain_child_run, str)
            or not chain_child_run.strip()
        ):
            failures.append(
                "exact_evidence_chain_original_child_run_identifier_missing"
            )
        if (
            not isinstance(child_run_directory, str)
            or not child_run_directory.strip()
        ):
            failures.append(
                "exact_evidence_chain_original_child_run_directory_missing"
            )
        if not _is_sha256(child_run_directory_identity):
            failures.append(
                "exact_evidence_chain_child_run_directory_identity_invalid"
            )
        chain_artifact_path = evidence_chain.get(
            "current_artifact_path"
        ) or evidence_chain.get("artifact_path")
        if artifact_verification.get("artifact_path") != chain_artifact_path:
            failures.append(
                "artifact_verification_path_evidence_chain_mismatch"
            )
        if artifact_verification.get(
            "accepted_artifact_sha256"
        ) != evidence_chain.get("artifact_bytes_sha256"):
            failures.append(
                "artifact_verification_sha256_evidence_chain_mismatch"
            )

    if (
        artifact_internal_identity not in (None, "")
        and chain_child_run not in (None, "")
        and artifact_internal_identity != chain_child_run
    ):
        failures.append(
            "artifact_internal_original_run_identity_chain_mismatch"
        )

    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "artifact_observation": observed,
        "artifact_observation_sha256": artifact_observation.get(
            "observation_sha256"
        ),
        "corrective_original_child_run_identifier": chain_child_run,
        "corrective_original_child_run_identifier_source": (
            "exact_multi_file_evidence_chain"
            if not allow_legacy_role_fixture
            else "legacy_multi_file_evidence_chain_fixture"
        ),
        "original_child_run_directory": child_run_directory,
        "child_run_directory_identity": child_run_directory_identity,
    }


def build_corrective_binding_manifest_draft(
    *,
    audit_verification: Mapping[str, Any],
    evidence_chain: Mapping[str, Any],
    artifact_verification: Mapping[str, Any],
    external_population: Mapping[str, Any],
    authoritative_population: Mapping[str, Any],
    creation_timestamp_utc: Optional[str] = None,
    _test_expectations: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    failures = []
    for name, payload in (
        ("audit", audit_verification),
        ("evidence_chain", evidence_chain),
        ("artifact", artifact_verification),
        ("external_population", external_population),
        ("authoritative_population", authoritative_population),
    ):
        if payload.get("status") != "ok":
            failures.append("{}_validation_not_ok".format(name))
    allow_legacy_role_fixture = bool(
        (_test_expectations or {}).get("allow_legacy_role_fixture")
    )
    if (
        evidence_chain.get("paper_role_binding_valid") is not True
        and not allow_legacy_role_fixture
    ):
        failures.append("exact_paper_role_binding_not_ok")
    canonical_role_specs = dict(
        evidence_chain.get("resolved_role_specs") or {}
    )
    if not allow_legacy_role_fixture:
        try:
            record_role_specs = canonical_resolved_role_specs_from_records(
                evidence_chain.get("evidence_files") or [],
                role_contracts=(
                    (_test_expectations or {}).get("accepted_role_specs")
                ),
            )
        except ValueError as exc:
            failures.extend(
                getattr(exc, "failures", (str(exc),))
            )
            record_role_specs = {}
        if canonical_role_specs != record_role_specs:
            failures.append(
                "evidence_chain_resolved_role_specs_record_mismatch"
            )
        canonical_role_specs = record_role_specs
    digest_binding = verify_historical_digest_binding(
        external_population,
        artifact_verification,
        _expected_historical_digests=(
            (_test_expectations or {}).get("historical")
        ),
    )
    if digest_binding.get("status") != "ok":
        failures.append("historical_digest_binding_not_ok")
    if failures:
        raise ValueError(
            "corrective_binding_inputs_invalid:{}".format(",".join(failures))
        )
    timestamp = creation_timestamp_utc or datetime.now(timezone.utc).isoformat()
    child = _role_details(evidence_chain, "child_final_status")
    outer = _role_details(evidence_chain, "outer_full_acceptance")
    producer = _role_details(evidence_chain, "producer_git_identity")
    child_exit = _role_details(evidence_chain, "child_full_exit_code")
    outer_exit = _role_details(
        evidence_chain, "outer_acceptance_exit_code"
    )
    manifest_required_roles = (
        LEGACY_REQUIRED_EVIDENCE_ROLES
        if allow_legacy_role_fixture
        else REQUIRED_EVIDENCE_ROLES
    )
    manifest_evidence_roles = tuple(manifest_required_roles)
    if (
        not allow_legacy_role_fixture
        and _role_record(evidence_chain, "replay_summary")
    ):
        manifest_evidence_roles += ("replay_summary",)
    role_records = {
        role: _role_record(evidence_chain, role)
        for role in manifest_evidence_roles
    }
    corrective_run_binding = _corrective_child_run_identity_binding(
        evidence_chain,
        artifact_verification,
        allow_legacy_role_fixture=allow_legacy_role_fixture,
    )
    if corrective_run_binding.get("status") != "ok":
        raise ValueError(
            "corrective_binding_run_identity_invalid:{}".format(
                ",".join(corrective_run_binding.get("failures") or [])
            )
        )
    artifact_run_observation = dict(
        corrective_run_binding.get("artifact_observation") or {}
    )
    corrective_child_run_identifier = corrective_run_binding.get(
        "corrective_original_child_run_identifier"
    )
    population_role = role_records.get(
        "external_population_comparison", {}
    )
    replay_role = role_records.get("replay_summary", {})
    identity_domains = evidence_chain.get("identity_domains")
    identity_domains = (
        dict(identity_domains)
        if isinstance(identity_domains, Mapping)
        else {
            "population_source_run_identifier": None,
            "original_child_run_identifier": evidence_chain.get(
                "original_run_identity"
            ),
            "original_outer_run_identifier": None,
            "current_audit_run_identifier": None,
        }
    )
    if identity_domains.get("current_audit_run_identifier") in (None, ""):
        identity_domains["current_audit_run_identifier"] = (
            "accepted_audit_bundle_sha256:{}".format(
                audit_verification.get("archive_sha256")
            )
        )
    current_audit_run_identifier = identity_domains.get(
        "current_audit_run_identifier"
    )
    child_execution_status = child.get(
        "original_child_execution_status",
        child.get("execution_status"),
    )
    child_exit_code = child.get(
        "original_child_exit_code",
        child.get("exit_code"),
    )
    outer_acceptance_status = outer.get(
        "original_outer_acceptance_status",
        outer.get("acceptance_status"),
    )
    outer_exit_code = (
        outer.get(
            "original_outer_acceptance_exit_code",
            outer.get("exit_code"),
        )
        if allow_legacy_role_fixture
        else evidence_chain.get(
            "original_outer_acceptance_exit_code",
            outer_exit.get("exit_code"),
        )
    )
    outer_failures = outer.get(
        "original_outer_acceptance_failures",
        outer.get("failures"),
    )
    replay_matched_count = evidence_chain.get(
        "replay_matched_count",
        child.get("replay_matched_count"),
    )
    replay_expected_count = evidence_chain.get(
        "replay_expected_count",
        child.get("replay_expected_count"),
    )
    reconstructed_population = {
        "evidence_type": "reconstructed_fitting_population_manifest_v1",
        "artifact_fitting_stable_sample_ids": list(
            authoritative_population.get("artifact_fitting_stable_sample_ids")
            or []
        ),
        "artifact_heldout_stable_sample_ids": list(
            authoritative_population.get("artifact_heldout_stable_sample_ids")
            or []
        ),
        "artifact_fitting_count": authoritative_population.get("fitting_count"),
        "artifact_heldout_count": authoritative_population.get("heldout_count"),
        "historical_train_digest": external_population.get(
            "historical_train_digest"
        ),
        "historical_eval_digest": external_population.get(
            "historical_eval_digest"
        ),
        "historical_aligned_digest": external_population.get(
            "historical_aligned_digest"
        ),
        "canonical_fitting_set_sha256": authoritative_population.get(
            "canonical_fitting_set_sha256"
        ),
        "canonical_heldout_set_sha256": authoritative_population.get(
            "canonical_heldout_set_sha256"
        ),
        "canonical_union_set_sha256": authoritative_population.get(
            "canonical_union_set_sha256"
        ),
        "canonical_heldout_dataset_order_sha256": (
            authoritative_population.get(
                "canonical_heldout_dataset_order_sha256"
            )
        ),
        "historical_serialization_identity": (
            HISTORICAL_DIGEST_ALGORITHM_IDENTITY
        ),
        "canonical_serialization_identity": (
            CANONICAL_SET_ALGORITHM_IDENTITY
        ),
        "source_evidence_file_sha256": external_population.get("source_sha256"),
        "reconstruction_source_roles": [
            "external_population_comparison",
            "current_artifact_preflight",
            "artifact_identity",
        ],
    }
    manifest = {
        "schema_version": CORRECTIVE_BINDING_SCHEMA_VERSION,
        "evidence_type": CORRECTIVE_BINDING_EVIDENCE_TYPE,
        "provenance_construction": (
            CORRECTIVE_BINDING_PROVENANCE_CONSTRUCTION
        ),
        "approval_status": CORRECTIVE_BINDING_APPROVAL_STATUS,
        "central_acceptance": False,
        "creation_timestamp_utc": timestamp,
        "source_audit_bundle_path": audit_verification.get("archive_path"),
        "source_audit_bundle_sha256": audit_verification.get("archive_sha256"),
        "source_audit_bundle_member_count": audit_verification.get(
            "archive_member_count"
        ),
        "source_audit_bundle_internal_manifest_sha256": (
            audit_verification.get("internal_manifest_sha256")
        ),
        "source_audit_bundle_verification_sha256": canonical_json_sha256(
            {
                "archive_sha256": audit_verification.get("archive_sha256"),
                "internal_manifest_sha256": audit_verification.get(
                    "internal_manifest_sha256"
                ),
                "member_count": audit_verification.get(
                    "archive_member_count"
                ),
            }
        ),
        "multi_file_evidence_chain_sha256": evidence_chain.get(
            "multi_file_evidence_chain_sha256"
        ),
        "paper_role_binding_valid": evidence_chain.get(
            "paper_role_binding_valid"
        )
        is True,
        "identity_domains": identity_domains,
        "accepted_audit_role_specs": canonical_role_specs,
        "accepted_audit_role_specs_sha256": canonical_json_sha256(
            canonical_role_specs
        ),
        "role_specific_observations": dict(
            evidence_chain.get("role_specific_observations") or {}
        ),
        "normalized_field_provenance": dict(
            evidence_chain.get("normalized_field_provenance") or {}
        ),
        "outer_acceptance_normalized_field_provenance": dict(
            evidence_chain.get(
                "outer_acceptance_normalized_field_provenance"
            )
            or {}
        ),
        "child_outer_relation_evidence": dict(
            evidence_chain.get("child_outer_relation_evidence") or {}
        ),
        "required_evidence_roles": list(manifest_evidence_roles),
        "evidence_files": list(evidence_chain.get("evidence_files") or []),
        "accepted_artifact_path": artifact_verification.get("artifact_path"),
        "historical_artifact_path": evidence_chain.get(
            "historical_artifact_path"
        ),
        "current_artifact_path": evidence_chain.get(
            "current_artifact_path"
        )
        or artifact_verification.get("artifact_path"),
        "artifact_relocation_binding": evidence_chain.get(
            "artifact_relocation_binding"
        ),
        "accepted_artifact_size": artifact_verification.get("artifact_size"),
        "accepted_artifact_sha256": artifact_verification.get(
            "accepted_artifact_sha256"
        ),
        "artifact_sha256_before_load": artifact_verification.get(
            "artifact_file_sha256_before_load"
        ),
        "artifact_sha256_after_load": artifact_verification.get(
            "artifact_file_sha256_after_load"
        ),
        "artifact_load_preserved_bytes": artifact_verification.get(
            "artifact_load_preserved_bytes"
        ),
        "artifact_internal_split_field_paths": artifact_verification.get(
            "artifact_internal_split_field_paths"
        ),
        "artifact_internal_split_payloads": artifact_verification.get(
            "artifact_internal_split_payloads"
        ),
        "artifact_metadata_schema_version": artifact_verification.get(
            "artifact_metadata_schema_version"
        ),
        "artifact_internal_run_identity_observation": (
            artifact_run_observation
        ),
        "artifact_internal_run_identity_observation_sha256": (
            corrective_run_binding.get("artifact_observation_sha256")
        ),
        "artifact_internal_original_run_identity_present": (
            artifact_run_observation.get(
                "artifact_internal_original_run_identity_present"
            )
        ),
        "artifact_internal_original_run_identity": (
            artifact_run_observation.get(
                "artifact_internal_original_run_identity"
            )
        ),
        "artifact_internal_original_run_identity_status": (
            artifact_run_observation.get(
                "artifact_internal_original_run_identity_status"
            )
        ),
        "corrective_original_child_run_identifier": (
            corrective_child_run_identifier
        ),
        "corrective_original_child_run_identifier_source": (
            corrective_run_binding.get(
                "corrective_original_child_run_identifier_source"
            )
        ),
        "artifact_train_count": artifact_verification.get(
            "artifact_train_count"
        ),
        "artifact_eval_count": artifact_verification.get("artifact_eval_count"),
        "artifact_aligned_count": artifact_verification.get(
            "artifact_aligned_count"
        ),
        "artifact_historical_train_digest": artifact_verification.get(
            "artifact_historical_train_digest"
        ),
        "artifact_historical_eval_digest": artifact_verification.get(
            "artifact_historical_eval_digest"
        ),
        "artifact_historical_aligned_digest": artifact_verification.get(
            "artifact_historical_aligned_digest"
        ),
        "external_population_evidence_historical_path": (
            evidence_chain.get("population_evidence_historical_path")
            or population_role.get("absolute_historical_path")
            or external_population.get("source_path")
        ),
        "external_population_evidence_snapshot_path": population_role.get(
            "snapshot_relative_path"
        ),
        "external_population_evidence_snapshot_present": population_role.get(
            "snapshot_present"
        ),
        "external_population_evidence_binding_source": population_role.get(
            "binding_source"
        ),
        "external_population_evidence_live_original_verified": (
            population_role.get("live_original_verified")
        ),
        "external_population_evidence_sha256": external_population.get(
            "source_sha256"
        ),
        "artifact_fitting_count": authoritative_population.get("fitting_count"),
        "artifact_heldout_count": authoritative_population.get("heldout_count"),
        "duplicate_fitting_count": authoritative_population.get(
            "duplicate_fitting_id_count"
        ),
        "duplicate_heldout_count": authoritative_population.get(
            "duplicate_heldout_id_count"
        ),
        "intersection_count": authoritative_population.get(
            "intersection_count"
        ),
        "union_count": authoritative_population.get("union_count"),
        "historical_digest_algorithm_identity": (
            HISTORICAL_DIGEST_ALGORITHM_IDENTITY
        ),
        "historical_digest_serialization": dict(
            HISTORICAL_DIGEST_SERIALIZATION
        ),
        "recomputed_historical_train_digest": external_population.get(
            "historical_train_digest"
        ),
        "recomputed_historical_eval_digest": external_population.get(
            "historical_eval_digest"
        ),
        "recomputed_historical_aligned_digest": external_population.get(
            "historical_aligned_digest"
        ),
        "historical_train_digest_match": digest_binding.get(
            "historical_train_digest_match"
        ),
        "historical_eval_digest_match": digest_binding.get(
            "historical_eval_digest_match"
        ),
        "historical_aligned_digest_match": digest_binding.get(
            "historical_aligned_digest_match"
        ),
        "historical_digest_is_canonical_lf_set_sha": False,
        "historical_and_canonical_algorithms_distinct": True,
        "canonical_fitting_set_sha256": authoritative_population.get(
            "canonical_fitting_set_sha256"
        ),
        "canonical_heldout_set_sha256": authoritative_population.get(
            "canonical_heldout_set_sha256"
        ),
        "canonical_union_set_sha256": authoritative_population.get(
            "canonical_union_set_sha256"
        ),
        "canonical_heldout_dataset_order_sha256": (
            authoritative_population.get(
                "canonical_heldout_dataset_order_sha256"
            )
        ),
        "split_algorithm": artifact_verification.get("split_algorithm"),
        "split_unit": artifact_verification.get("split_unit"),
        "split_mode": artifact_verification.get("split_mode"),
        "split_ratio": artifact_verification.get("split_ratio"),
        "split_seed": artifact_verification.get("split_seed"),
        "stable_sample_id_algorithm_identity": (
            STABLE_SAMPLE_ID_ALGORITHM_IDENTITY
        ),
        "dataset_name": authoritative_population.get("dataset_name"),
        "dataset_config_name": authoritative_population.get(
            "dataset_config_name"
        ),
        "dataset_split": authoritative_population.get("dataset_split"),
        "text_column": authoritative_population.get("text_column"),
        "summary_column": authoritative_population.get("summary_column"),
        "runtime_policy_sha256": artifact_verification.get(
            "runtime_policy_sha256"
        ),
        "hybrid_fitting_policy_sha256": artifact_verification.get(
            "hybrid_fitting_policy_sha256"
        ),
        "dump_run_binding": artifact_verification.get("dump_run_binding"),
        "dump_run_binding_identity": artifact_verification.get(
            "dump_run_binding_identity"
        ),
        "checkpoint_identity": artifact_verification.get(
            "checkpoint_identity"
        ),
        "tokenizer_identity": artifact_verification.get("tokenizer_identity"),
        "population_source_run_identifier": evidence_chain.get(
            "population_source_run_identifier"
        ),
        "population_explicit_run_identifier_status": evidence_chain.get(
            "population_explicit_run_identifier_status"
        ),
        "population_explicit_run_identifier": evidence_chain.get(
            "population_explicit_run_identifier"
        ),
        "population_evidence_historical_path": evidence_chain.get(
            "population_evidence_historical_path"
        ),
        "population_evidence_historical_directory": evidence_chain.get(
            "population_evidence_historical_directory"
        ),
        "population_directory_derived_run_identity": evidence_chain.get(
            "population_directory_derived_run_identity"
        ),
        "population_directory_derived_run_identity_status": (
            evidence_chain.get(
                "population_directory_derived_run_identity_status"
            )
        ),
        "population_evidence_directory": evidence_chain.get(
            "population_evidence_directory"
        ),
        "population_path_root": evidence_chain.get(
            "population_path_root"
        ),
        "original_child_run_identifier": corrective_child_run_identifier,
        "original_child_run_directory": corrective_run_binding.get(
            "original_child_run_directory"
        ),
        "child_run_directory_identity": corrective_run_binding.get(
            "child_run_directory_identity"
        ),
        "original_outer_run_identifier": evidence_chain.get(
            "original_outer_run_identifier"
        ),
        "original_outer_run_identity_status": evidence_chain.get(
            "original_outer_run_identity_status"
        ),
        "original_outer_run_directory": producer.get(
            "original_outer_run_directory"
        )
        or outer.get("original_outer_run_directory")
        or evidence_chain.get("original_outer_run_directory")
        or producer.get("original_run_directory"),
        "outer_run_directory_identity": evidence_chain.get(
            "outer_run_directory_identity"
        ),
        "current_audit_run_identifier": evidence_chain.get(
            "current_audit_run_identifier"
        )
        or current_audit_run_identifier,
        "original_run_directory": evidence_chain.get(
            "original_child_run_directory"
        )
        or producer.get("original_run_directory"),
        "original_run_identifier": corrective_child_run_identifier,
        "original_producer_commit": evidence_chain.get("producer_commit"),
        "original_producer_branch": evidence_chain.get("producer_branch")
        or producer.get("producer_branch"),
        "original_producer_tracked_status": evidence_chain.get(
            "producer_tracked_status"
        ),
        "original_producer_tracked_worktree_clean": evidence_chain.get(
            "producer_tracked_worktree_clean"
        ),
        "original_command_sha256": role_records[
            "original_fitting_command"
        ].get("file_sha256"),
        "selected_fitting_command": evidence_chain.get(
            "selected_fitting_command"
        ),
        "selected_fitting_command_sha256": evidence_chain.get(
            "selected_fitting_command_sha256"
        ),
        "selected_fitting_command_field_provenance": evidence_chain.get(
            "selected_fitting_command_field_provenance"
        ),
        "selected_dump_run_manifest": evidence_chain.get(
            "selected_dump_run_manifest"
        ),
        "selected_kv_manifest": evidence_chain.get("selected_kv_manifest"),
        "selected_hidden_manifest": evidence_chain.get(
            "selected_hidden_manifest"
        ),
        "d2_command_binding": {
            **dict(evidence_chain.get("d2_command_binding") or {}),
            "artifact_dump_run_binding_identity": (
                artifact_verification.get("dump_run_binding_identity")
            ),
        },
        "fit_summary_sha256": role_records["fit_summary"].get("file_sha256"),
        "artifact_summary_sha256": role_records["artifact_summary"].get(
            "file_sha256"
        ),
        "artifact_identity_sha256": role_records["artifact_identity"].get(
            "file_sha256"
        ),
        "artifact_identity_sidecar_sha256": role_records[
            "artifact_identity"
        ].get("evidence_file_sha256")
        or role_records["artifact_identity"].get("file_sha256"),
        "artifact_bytes_sha256": evidence_chain.get("artifact_bytes_sha256")
        or artifact_verification.get("accepted_artifact_sha256"),
        "child_final_status_sha256": role_records["child_final_status"].get(
            "file_sha256"
        ),
        "outer_full_acceptance_sha256": role_records[
            "outer_full_acceptance"
        ].get("file_sha256"),
        "original_child_exit_code_sidecar_path": role_records.get(
            "child_full_exit_code", {}
        ).get("absolute_historical_path"),
        "original_child_exit_code_sidecar_sha256": role_records.get(
            "child_full_exit_code", {}
        ).get("file_sha256"),
        "original_child_exit_code_sidecar_value": child_exit.get(
            "exit_code"
        ),
        "original_full_run_exit_code_sidecar_path": role_records.get(
            "child_full_exit_code", {}
        ).get("absolute_historical_path"),
        "original_full_run_exit_code_sidecar_sha256": role_records.get(
            "child_full_exit_code", {}
        ).get("file_sha256"),
        "original_full_run_exit_code": child_exit.get("exit_code"),
        "original_outer_acceptance_exit_code_sidecar_path": role_records.get(
            "outer_acceptance_exit_code", {}
        ).get("absolute_historical_path"),
        "original_outer_acceptance_exit_code_sidecar_sha256": role_records.get(
            "outer_acceptance_exit_code", {}
        ).get("file_sha256"),
        "original_outer_acceptance_exit_code_sidecar_value": outer_exit.get(
            "exit_code"
        ),
        "original_file_manifest_sha256": role_records["file_manifest"].get(
            "file_sha256"
        ),
        "original_full_run_file_manifest_path": evidence_chain.get(
            "original_full_run_file_manifest_path"
        ),
        "original_full_run_file_manifest_sha256": (
            evidence_chain.get("original_full_run_file_manifest_sha256")
            or role_records["file_manifest"].get("file_sha256")
        ),
        "original_full_run_file_manifest_entry_count": evidence_chain.get(
            "original_full_run_file_manifest_entry_count"
        ),
        "original_full_run_file_manifest_physical_parent": (
            evidence_chain.get(
                "original_full_run_file_manifest_physical_parent"
            )
        ),
        "original_full_run_file_manifest_logical_resolution_root": (
            evidence_chain.get(
                "original_full_run_file_manifest_logical_resolution_root"
            )
        ),
        "outer_acceptance_artifact_path_field": evidence_chain.get(
            "outer_acceptance_artifact_path_field"
        ),
        "nested_child_artifact_path_field": evidence_chain.get(
            "nested_child_artifact_path_field"
        ),
        "nested_artifact_sha_present": evidence_chain.get(
            "nested_artifact_sha_present"
        ),
        "producer_git_identity_sha256": role_records[
            "producer_git_identity"
        ].get("file_sha256"),
        "original_child_execution_status": child_execution_status,
        "original_child_exit_code": child_exit_code,
        "replay_matched_count": replay_matched_count,
        "replay_expected_count": replay_expected_count,
        "replay_mismatch_count": evidence_chain.get(
            "replay_mismatch_count"
        ),
        "replay_summary_supporting_role_path": replay_role.get(
            "absolute_historical_path"
        ),
        "replay_summary_supporting_role_sha256": replay_role.get(
            "evidence_file_sha256"
        ),
        "replay_summary_supporting_role_snapshot_present": replay_role.get(
            "snapshot_present"
        ),
        "replay_summary_supporting_role_snapshot_path": replay_role.get(
            "snapshot_relative_path"
        ),
        "replay_summary_supporting_role_binding_source": replay_role.get(
            "binding_source"
        ),
        "replay_summary_supporting_role_live_original_verified": (
            replay_role.get("live_original_verified")
        ),
        "replay_summary_status": evidence_chain.get("replay_summary_status"),
        "replay_summary_comparison_status": evidence_chain.get(
            "replay_summary_comparison_status"
        ),
        "outer_acceptance_replay_summary_exists": evidence_chain.get(
            "outer_acceptance_replay_summary_exists"
        ),
        "outer_acceptance_replay_summary_path": evidence_chain.get(
            "outer_acceptance_replay_summary_path"
        ),
        "replay_field_provenance": dict(
            evidence_chain.get("replay_field_provenance") or {}
        ),
        "external_population_role_field_provenance": dict(
            population_role.get("normalized_field_provenance") or {}
        ),
        "external_population_role_identity": {
            key: (population_role.get("normalized_fields") or {}).get(key)
            for key in (
                "train_count",
                "eval_count",
                "aligned_count",
                "historical_train_digest",
                "historical_eval_digest",
                "historical_aligned_digest",
                "canonical_fitting_set_sha256",
                "canonical_heldout_set_sha256",
                "canonical_union_set_sha256",
                "explicit_run_identifier_status",
                "explicit_run_identifier",
                "population_evidence_historical_path",
                "population_evidence_historical_directory",
                "population_evidence_directory",
                "directory_derived_run_identity",
                "directory_derived_run_identity_status",
            )
        },
        "original_outer_acceptance_status": outer_acceptance_status,
        "original_outer_acceptance_exit_code": outer_exit_code,
        "outer_acceptance_json_exit_code_present": evidence_chain.get(
            "outer_acceptance_json_exit_code_present"
        ),
        "outer_acceptance_json_exit_code": evidence_chain.get(
            "outer_acceptance_json_exit_code"
        ),
        "original_outer_acceptance_exit_code_source_role": (
            evidence_chain.get(
                "original_outer_acceptance_exit_code_source_role"
            )
        ),
        "original_outer_acceptance_exit_code_source_path": (
            evidence_chain.get(
                "original_outer_acceptance_exit_code_source_path"
            )
        ),
        "original_outer_acceptance_failures": list(
            outer_failures or []
        ),
        "accepted_artifact_resaved": False,
        "artifact_refit_performed": False,
        "original_files_modified": False,
        "reconstructed_fitting_population_manifest": reconstructed_population,
        "unresolved_fields": [],
        "audit_warnings": [
            "pending_central_review",
            "original_outer_acceptance_failed",
            *list(outer_failures or []),
        ],
        "central_approval_valid": False,
        "paper_candidate_valid": False,
        "gpu_authorized": False,
    }
    return manifest


def _manifest_failure(
    failures: List[str],
    condition: bool,
    reason: str,
) -> None:
    if condition:
        failures.append(reason)


def validate_corrective_binding_manifest_draft(
    payload: Mapping[str, Any],
    *,
    accepted_artifact_path: Optional[Path | str] = None,
    _artifact_loader: Optional[
        Callable[[Path], Mapping[str, Any]]
    ] = None,
    _test_expectations: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    expectations = dict(_test_expectations or {})
    allow_legacy_role_fixture = bool(
        expectations.get("allow_legacy_role_fixture")
    )
    expected_historical_values = dict(
        expectations.get("historical")
        or {
            "train": EXPECTED_HISTORICAL_TRAIN_DIGEST,
            "eval": EXPECTED_HISTORICAL_EVAL_DIGEST,
            "aligned": EXPECTED_HISTORICAL_ALIGNED_DIGEST,
        }
    )
    expected_canonical_values = dict(
        expectations.get("canonical")
        or {
            "fitting": EXPECTED_CANONICAL_FITTING_SET_SHA256,
            "heldout": EXPECTED_CANONICAL_HELDOUT_SET_SHA256,
            "union": EXPECTED_CANONICAL_UNION_SET_SHA256,
        }
    )
    expected_fitting_count = int(
        expectations.get("fitting_count", EXPECTED_ARTIFACT_FITTING_COUNT)
    )
    expected_heldout_count = int(
        expectations.get("heldout_count", EXPECTED_ARTIFACT_HELDOUT_COUNT)
    )
    expected_union_count = int(
        expectations.get(
            "union_count", EXPECTED_SAMSUM_VALIDATION_POPULATION_COUNT
        )
    )
    expected_artifact_sha256 = str(
        expectations.get(
            "accepted_artifact_sha256",
            ACCEPTED_PHASE3C_ARTIFACT_SHA256,
        )
    )
    expected_audit_bundle_sha256 = str(
        expectations.get(
            "audit_bundle_sha256",
            ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256,
        )
    )
    expected_external_evidence_sha256 = str(
        expectations.get(
            "external_population_evidence_sha256",
            ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256,
        )
    )
    expected_role_specs = expectations.get("accepted_role_specs")
    expected_role_specs = (
        expected_role_specs
        if isinstance(expected_role_specs, Mapping)
        else ACCEPTED_AUDIT_ROLE_SPECS
    )
    failures: List[str] = []
    _manifest_failure(
        failures,
        payload.get("schema_version") != CORRECTIVE_BINDING_SCHEMA_VERSION,
        "corrective_manifest_schema_version_mismatch",
    )
    _manifest_failure(
        failures,
        payload.get("evidence_type") != CORRECTIVE_BINDING_EVIDENCE_TYPE,
        "corrective_manifest_evidence_type_mismatch",
    )
    _manifest_failure(
        failures,
        payload.get("provenance_construction")
        != CORRECTIVE_BINDING_PROVENANCE_CONSTRUCTION,
        "corrective_manifest_provenance_construction_mismatch",
    )
    _manifest_failure(
        failures,
        payload.get("approval_status")
        != CORRECTIVE_BINDING_APPROVAL_STATUS,
        "corrective_manifest_approval_status_not_pending",
    )
    _manifest_failure(
        failures,
        payload.get("central_acceptance") is not False,
        "corrective_manifest_central_acceptance_must_be_false",
    )
    for field in (
        "central_approval_valid",
        "paper_candidate_valid",
        "gpu_authorized",
    ):
        _manifest_failure(
            failures,
            payload.get(field) is not False,
            "corrective_manifest_{}_must_be_false".format(field),
        )
    for field, reason in (
        ("accepted_artifact_resaved", "accepted_artifact_resaved_must_be_false"),
        ("artifact_refit_performed", "artifact_refit_performed_must_be_false"),
        ("original_files_modified", "original_files_modified_must_be_false"),
    ):
        _manifest_failure(failures, payload.get(field) is not False, reason)
    _manifest_failure(
        failures,
        payload.get("source_audit_bundle_sha256")
        != expected_audit_bundle_sha256,
        "source_audit_bundle_sha256_mismatch",
    )
    internal_manifest_sha = payload.get(
        "source_audit_bundle_internal_manifest_sha256"
    )
    member_count = payload.get("source_audit_bundle_member_count")
    if not _is_sha256(internal_manifest_sha):
        failures.append("source_audit_bundle_internal_manifest_sha256_invalid")
    if (
        not isinstance(member_count, int)
        or isinstance(member_count, bool)
        or member_count <= 0
    ):
        failures.append("source_audit_bundle_member_count_invalid")
    expected_audit_verification_sha = canonical_json_sha256(
        {
            "archive_sha256": payload.get("source_audit_bundle_sha256"),
            "internal_manifest_sha256": internal_manifest_sha,
            "member_count": member_count,
        }
    )
    if payload.get(
        "source_audit_bundle_verification_sha256"
    ) != expected_audit_verification_sha:
        failures.append("source_audit_bundle_verification_sha256_mismatch")
    _manifest_failure(
        failures,
        payload.get("accepted_artifact_sha256")
        != expected_artifact_sha256,
        "accepted_artifact_sha256_mismatch",
    )
    before = payload.get("artifact_sha256_before_load")
    after = payload.get("artifact_sha256_after_load")
    _manifest_failure(
        failures,
        before != expected_artifact_sha256
        or after != expected_artifact_sha256
        or before != after,
        "artifact_before_after_sha256_mismatch",
    )
    _manifest_failure(
        failures,
        payload.get("artifact_load_preserved_bytes") is not True,
        "artifact_load_preserved_bytes_not_true",
    )
    _manifest_failure(
        failures,
        payload.get("external_population_evidence_sha256")
        != expected_external_evidence_sha256,
        "external_population_evidence_sha256_mismatch",
    )

    records = payload.get("evidence_files")
    if not isinstance(records, list):
        failures.append("corrective_manifest_evidence_files_missing")
        records = []
    chain = (
        validate_multi_file_evidence_chain(records)
        if allow_legacy_role_fixture
        else validate_exact_role_evidence_chain(
            records,
            role_specs=expected_role_specs,
            expected_artifact_sha256=expected_artifact_sha256,
        )
    )
    if chain.get("status") != "ok":
        failures.extend(chain.get("failures") or [])
    if (
        chain.get("paper_role_binding_valid") is not True
        and not allow_legacy_role_fixture
    ):
        failures.append("corrective_manifest_exact_role_binding_required")
    if (
        payload.get("paper_role_binding_valid") is not True
        and not allow_legacy_role_fixture
    ):
        failures.append("corrective_manifest_paper_role_binding_not_valid")
    expected_manifest_roles = (
        LEGACY_REQUIRED_EVIDENCE_ROLES
        if allow_legacy_role_fixture
        else REQUIRED_EVIDENCE_ROLES
    )
    if (
        not allow_legacy_role_fixture
        and any(
            isinstance(record, Mapping)
            and record.get("evidence_role") == "replay_summary"
            for record in records
        )
    ):
        expected_manifest_roles += ("replay_summary",)
    if payload.get("required_evidence_roles") != list(
        expected_manifest_roles
    ):
        failures.append("corrective_manifest_required_evidence_roles_mismatch")
    if payload.get("multi_file_evidence_chain_sha256") != chain.get(
        "multi_file_evidence_chain_sha256"
    ):
        failures.append("multi_file_evidence_chain_sha256_mismatch")
    role_records = {
        str(record.get("evidence_role")): record
        for record in records
        if isinstance(record, Mapping)
    }
    for field, role in (
        ("original_command_sha256", "original_fitting_command"),
        ("fit_summary_sha256", "fit_summary"),
        ("artifact_summary_sha256", "artifact_summary"),
        ("artifact_identity_sha256", "artifact_identity"),
        ("child_final_status_sha256", "child_final_status"),
        ("outer_full_acceptance_sha256", "outer_full_acceptance"),
        (
            "original_child_exit_code_sidecar_sha256",
            "child_full_exit_code",
        ),
        (
            "original_full_run_exit_code_sidecar_sha256",
            "child_full_exit_code",
        ),
        (
            "original_outer_acceptance_exit_code_sidecar_sha256",
            "outer_acceptance_exit_code",
        ),
        ("original_file_manifest_sha256", "file_manifest"),
        ("original_full_run_file_manifest_sha256", "file_manifest"),
        ("producer_git_identity_sha256", "producer_git_identity"),
        (
            "replay_summary_supporting_role_sha256",
            "replay_summary",
        ),
    ):
        if payload.get(field) != (role_records.get(role) or {}).get(
            "file_sha256"
        ):
            failures.append("{}_role_binding_mismatch".format(field))
    chain_child_run = chain.get("original_child_run_identifier")
    if allow_legacy_role_fixture and chain_child_run in (None, ""):
        chain_child_run = chain.get("original_run_identity")
    if not allow_legacy_role_fixture:
        if (
            not isinstance(chain_child_run, str)
            or not chain_child_run.strip()
        ):
            failures.append(
                "exact_evidence_chain_original_child_run_identifier_missing"
            )
        if (
            not isinstance(chain.get("original_child_run_directory"), str)
            or not chain.get("original_child_run_directory")
        ):
            failures.append(
                "exact_evidence_chain_original_child_run_directory_missing"
            )
        if not _is_sha256(chain.get("child_run_directory_identity")):
            failures.append(
                "exact_evidence_chain_child_run_directory_identity_invalid"
            )

    trusted_chain_artifact_path = chain.get(
        "current_artifact_path"
    ) or chain.get("artifact_path")
    supplied_artifact_path = (
        str(accepted_artifact_path)
        if accepted_artifact_path is not None
        else None
    )
    if supplied_artifact_path is not None and (
        _normalized_local_artifact_path(supplied_artifact_path)
        != _normalized_local_artifact_path(trusted_chain_artifact_path)
    ):
        failures.append(
            "validator_artifact_path_evidence_chain_mismatch"
        )

    external_artifact_verification = expectations.get(
        "independent_artifact_verification"
    )
    if isinstance(external_artifact_verification, Mapping):
        independent_artifact = (
            _validate_external_artifact_verification_authority(
                external_artifact_verification,
                expected_artifact_path=trusted_chain_artifact_path,
                expected_artifact_sha256=expected_artifact_sha256,
            )
        )
    elif (
        trusted_chain_artifact_path in (None, "")
        or (
            supplied_artifact_path is not None
            and _normalized_local_artifact_path(supplied_artifact_path)
            != _normalized_local_artifact_path(
                trusted_chain_artifact_path
            )
        )
    ):
        independent_artifact = {
            "status": "failed",
            "failures": ["accepted_artifact_path_missing_or_untrusted"],
            "artifact_path": supplied_artifact_path,
            "artifact_file_sha256_before_load": None,
            "artifact_file_sha256_after_load": None,
            "artifact_load_preserved_bytes": None,
            "artifact_observation": {},
            "artifact_observation_sha256": None,
        }
    else:
        independent_artifact = (
            _load_artifact_run_identity_observation_read_only(
                supplied_artifact_path or trusted_chain_artifact_path,
                expected_artifact_sha256=expected_artifact_sha256,
                _loader=_artifact_loader,
            )
        )
    if independent_artifact.get("status") != "ok":
        failures.extend(independent_artifact.get("failures") or [])
    if independent_artifact.get(
        "artifact_path"
    ) is None or _normalized_local_artifact_path(
        independent_artifact.get("artifact_path")
    ) != _normalized_local_artifact_path(trusted_chain_artifact_path):
        failures.append(
            "independent_artifact_verification_path_chain_mismatch"
        )
    if independent_artifact.get(
        "artifact_file_sha256_before_load"
    ) != expected_artifact_sha256:
        failures.append(
            "independent_artifact_sha256_before_load_mismatch"
        )
    if independent_artifact.get(
        "artifact_file_sha256_after_load"
    ) != expected_artifact_sha256:
        failures.append(
            "independent_artifact_sha256_after_load_mismatch"
        )
    if independent_artifact.get(
        "artifact_load_preserved_bytes"
    ) is not True:
        failures.append(
            "independent_artifact_load_preserved_bytes_not_true"
        )

    artifact_observation = payload.get(
        "artifact_internal_run_identity_observation"
    )
    if not isinstance(artifact_observation, Mapping):
        failures.append(
            "artifact_internal_run_identity_observation_missing"
        )
        artifact_observation = {}
    else:
        artifact_observation = dict(artifact_observation)
    expected_artifact_observation = dict(
        independent_artifact.get("artifact_observation") or {}
    )
    expected_observation_sha = independent_artifact.get(
        "artifact_observation_sha256"
    )
    if payload.get(
        "artifact_internal_run_identity_observation_sha256"
    ) != expected_observation_sha:
        failures.append(
            "artifact_internal_run_identity_observation_sha256_mismatch"
        )
    if artifact_observation != expected_artifact_observation:
        failures.append(
            "artifact_internal_run_identity_observation_independent_mismatch"
        )
    if expected_observation_sha != canonical_json_sha256(
        expected_artifact_observation
    ):
        failures.append(
            "independent_artifact_observation_sha256_mismatch"
        )
    observed_artifact_run = expected_artifact_observation.get(
        "artifact_internal_original_run_identity"
    )
    observed_artifact_run_present = expected_artifact_observation.get(
        "artifact_internal_original_run_identity_present"
    )
    expected_artifact_run_status = expected_artifact_observation.get(
        "artifact_internal_original_run_identity_status"
    )
    for field, expected in (
        (
            "artifact_internal_original_run_identity_present",
            observed_artifact_run_present,
        ),
        (
            "artifact_internal_original_run_identity",
            observed_artifact_run,
        ),
        (
            "artifact_internal_original_run_identity_status",
            expected_artifact_run_status,
        ),
    ):
        if payload.get(field) != expected:
            failures.append("{}_observation_mismatch".format(field))
    if (
        observed_artifact_run_present
        and observed_artifact_run != chain_child_run
    ):
        failures.append(
            "artifact_internal_original_run_identity_chain_mismatch"
        )
    if payload.get(
        "corrective_original_child_run_identifier"
    ) != chain_child_run:
        failures.append(
            "corrective_original_child_run_identifier_chain_mismatch"
        )
    expected_corrective_source = (
        "legacy_multi_file_evidence_chain_fixture"
        if allow_legacy_role_fixture
        else "exact_multi_file_evidence_chain"
    )
    if payload.get(
        "corrective_original_child_run_identifier_source"
    ) != expected_corrective_source:
        failures.append(
            "corrective_original_child_run_identifier_source_mismatch"
        )
    if payload.get("original_run_identifier") != chain_child_run:
        failures.append("original_run_identifier_chain_mismatch")
    if payload.get("original_child_run_identifier") != chain_child_run:
        failures.append("original_child_run_identifier_chain_mismatch")
    if payload.get("original_child_run_directory") != chain.get(
        "original_child_run_directory"
    ):
        failures.append("original_child_run_directory_chain_mismatch")
    if payload.get("child_run_directory_identity") != chain.get(
        "child_run_directory_identity"
    ):
        failures.append("child_run_directory_identity_chain_mismatch")
    if payload.get("original_producer_commit") != chain.get(
        "producer_commit"
    ):
        failures.append("original_producer_commit_chain_mismatch")
    if payload.get("accepted_artifact_path") != chain.get("artifact_path"):
        failures.append("accepted_artifact_path_chain_mismatch")
    if not allow_legacy_role_fixture:
        if payload.get("original_producer_branch") != chain.get(
            "producer_branch"
        ):
            failures.append("original_producer_branch_chain_mismatch")
        if payload.get("original_producer_tracked_status") != chain.get(
            "producer_tracked_status"
        ):
            failures.append(
                "original_producer_tracked_status_chain_mismatch"
            )
        if payload.get(
            "original_producer_tracked_worktree_clean"
        ) != chain.get("producer_tracked_worktree_clean"):
            failures.append(
                "original_producer_tracked_worktree_clean_chain_mismatch"
            )
        if payload.get("artifact_identity_sidecar_sha256") != chain.get(
            "artifact_identity_sidecar_sha256"
        ):
            failures.append("artifact_identity_sidecar_sha256_chain_mismatch")
        if payload.get("artifact_bytes_sha256") != (
            expected_artifact_sha256
        ):
            failures.append("artifact_bytes_sha256_mismatch")
        if payload.get("artifact_identity_sidecar_sha256") == payload.get(
            "artifact_bytes_sha256"
        ):
            failures.append(
                "artifact_identity_sidecar_and_bytes_sha256_conflated"
            )
        expected_identity_domains = dict(chain.get("identity_domains") or {})
        if expected_identity_domains.get(
            "current_audit_run_identifier"
        ) in (None, ""):
            expected_identity_domains["current_audit_run_identifier"] = (
                "accepted_audit_bundle_sha256:{}".format(
                    payload.get("source_audit_bundle_sha256")
                )
            )
        if payload.get("identity_domains") != expected_identity_domains:
            failures.append("identity_domains_chain_mismatch")
        independently_resolved_role_specs = dict(
            chain.get("resolved_role_specs") or {}
        )
        if payload.get(
            "accepted_audit_role_specs"
        ) != independently_resolved_role_specs:
            failures.append("accepted_audit_role_specs_chain_mismatch")
        if payload.get(
            "accepted_audit_role_specs_sha256"
        ) != canonical_json_sha256(independently_resolved_role_specs):
            failures.append(
                "accepted_audit_role_specs_sha256_chain_mismatch"
            )
        if payload.get("role_specific_observations") != chain.get(
            "role_specific_observations"
        ):
            failures.append("role_specific_observations_chain_mismatch")
        if payload.get("normalized_field_provenance") != chain.get(
            "normalized_field_provenance"
        ):
            failures.append("normalized_field_provenance_chain_mismatch")
        if payload.get("historical_artifact_path") != chain.get(
            "historical_artifact_path"
        ):
            failures.append("historical_artifact_path_chain_mismatch")
        if payload.get("current_artifact_path") != chain.get(
            "current_artifact_path"
        ):
            failures.append("current_artifact_path_chain_mismatch")
        if payload.get("artifact_relocation_binding") != chain.get(
            "artifact_relocation_binding"
        ):
            failures.append("artifact_relocation_binding_chain_mismatch")
        if payload.get("replay_field_provenance") != chain.get(
            "replay_field_provenance"
        ):
            failures.append("replay_field_provenance_chain_mismatch")
        if payload.get("replay_mismatch_count") != chain.get(
            "replay_mismatch_count"
        ):
            failures.append("replay_mismatch_count_chain_mismatch")
        for field in (
            "replay_summary_status",
            "replay_summary_comparison_status",
            "outer_acceptance_replay_summary_exists",
            "outer_acceptance_replay_summary_path",
            "outer_acceptance_json_exit_code_present",
            "outer_acceptance_json_exit_code",
            "original_outer_acceptance_exit_code",
            "original_outer_acceptance_exit_code_source_role",
            "original_outer_acceptance_exit_code_source_path",
            "population_explicit_run_identifier_status",
            "population_explicit_run_identifier",
            "population_evidence_historical_path",
            "population_evidence_historical_directory",
            "population_directory_derived_run_identity",
            "population_directory_derived_run_identity_status",
            "population_evidence_directory",
            "population_path_root",
        ):
            if payload.get(field) != chain.get(field):
                failures.append("{}_chain_mismatch".format(field))
        replay_role = role_records.get("replay_summary") or {}
        for field, record_field in (
            (
                "replay_summary_supporting_role_path",
                "absolute_historical_path",
            ),
            (
                "replay_summary_supporting_role_snapshot_present",
                "snapshot_present",
            ),
            (
                "replay_summary_supporting_role_snapshot_path",
                "snapshot_relative_path",
            ),
            (
                "replay_summary_supporting_role_binding_source",
                "binding_source",
            ),
            (
                "replay_summary_supporting_role_live_original_verified",
                "live_original_verified",
            ),
        ):
            if payload.get(field) != replay_role.get(record_field):
                failures.append("{}_role_binding_mismatch".format(field))
        population_role = role_records.get(
            "external_population_comparison"
        ) or {}
        for field, record_field in (
            (
                "external_population_evidence_historical_path",
                "absolute_historical_path",
            ),
            (
                "external_population_evidence_snapshot_path",
                "snapshot_relative_path",
            ),
            (
                "external_population_evidence_snapshot_present",
                "snapshot_present",
            ),
            (
                "external_population_evidence_binding_source",
                "binding_source",
            ),
            (
                "external_population_evidence_live_original_verified",
                "live_original_verified",
            ),
        ):
            if payload.get(field) != population_role.get(record_field):
                failures.append("{}_role_binding_mismatch".format(field))
        if payload.get(
            "external_population_role_field_provenance"
        ) != dict(population_role.get("normalized_field_provenance") or {}):
            failures.append(
                "external_population_role_field_provenance_mismatch"
            )
        expected_population_role_identity = {
            key: (population_role.get("normalized_fields") or {}).get(key)
            for key in (
                "train_count",
                "eval_count",
                "aligned_count",
                "historical_train_digest",
                "historical_eval_digest",
                "historical_aligned_digest",
                "canonical_fitting_set_sha256",
                "canonical_heldout_set_sha256",
                "canonical_union_set_sha256",
                "explicit_run_identifier_status",
                "explicit_run_identifier",
                "population_evidence_historical_path",
                "population_evidence_historical_directory",
                "population_evidence_directory",
                "directory_derived_run_identity",
                "directory_derived_run_identity_status",
            )
        }
        if payload.get(
            "external_population_role_identity"
        ) != expected_population_role_identity:
            failures.append(
                "external_population_role_identity_mismatch"
            )
        for field in (
            "selected_fitting_command",
            "selected_fitting_command_sha256",
            "selected_fitting_command_field_provenance",
            "selected_dump_run_manifest",
            "selected_kv_manifest",
            "selected_hidden_manifest",
            "outer_acceptance_normalized_field_provenance",
            "child_outer_relation_evidence",
            "child_run_directory_identity",
            "outer_run_directory_identity",
            "original_full_run_file_manifest_path",
            "original_full_run_file_manifest_sha256",
            "original_full_run_file_manifest_entry_count",
            "original_full_run_file_manifest_physical_parent",
            (
                "original_full_run_file_manifest_"
                "logical_resolution_root"
            ),
            "outer_acceptance_artifact_path_field",
            "nested_child_artifact_path_field",
            "nested_artifact_sha_present",
        ):
            if payload.get(field) != chain.get(field):
                failures.append("{}_chain_mismatch".format(field))
        for field, role, normalized_field in (
            (
                "original_child_exit_code_sidecar_path",
                "child_full_exit_code",
                None,
            ),
            (
                "original_full_run_exit_code_sidecar_path",
                "child_full_exit_code",
                None,
            ),
            (
                "original_outer_acceptance_exit_code_sidecar_path",
                "outer_acceptance_exit_code",
                None,
            ),
            (
                "original_child_exit_code_sidecar_value",
                "child_full_exit_code",
                "exit_code",
            ),
            (
                "original_full_run_exit_code",
                "child_full_exit_code",
                "exit_code",
            ),
            (
                "original_outer_acceptance_exit_code_sidecar_value",
                "outer_acceptance_exit_code",
                "exit_code",
            ),
        ):
            role_record = role_records.get(role) or {}
            if normalized_field is None:
                expected = role_record.get("absolute_historical_path")
            else:
                normalized = role_record.get("normalized_fields")
                normalized = (
                    normalized if isinstance(normalized, Mapping) else {}
                )
                expected = normalized.get(normalized_field)
            if payload.get(field) != expected:
                failures.append("{}_role_binding_mismatch".format(field))
        expected_d2_binding = {
            **dict(chain.get("d2_command_binding") or {}),
            "artifact_dump_run_binding_identity": payload.get(
                "dump_run_binding_identity"
            ),
        }
        if payload.get("d2_command_binding") != expected_d2_binding:
            failures.append("d2_command_binding_chain_mismatch")
        for field in (
            "population_source_run_identifier",
            "original_outer_run_directory",
        ):
            if payload.get(field) != chain.get(field):
                failures.append("{}_chain_mismatch".format(field))
        if payload.get("original_outer_run_identifier") != chain.get(
            "original_outer_run_identifier"
        ):
            failures.append(
                "original_outer_run_identifier_chain_mismatch"
            )
        if payload.get("original_outer_run_identity_status") != chain.get(
            "original_outer_run_identity_status"
        ):
            failures.append(
                "original_outer_run_identity_status_chain_mismatch"
            )

    timestamp = payload.get("creation_timestamp_utc")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(timestamp).replace("Z", "+00:00")
        )
        if (
            parsed_timestamp.tzinfo is None
            or parsed_timestamp.utcoffset()
            != timezone.utc.utcoffset(parsed_timestamp)
        ):
            raise ValueError("utc_timezone_required")
    except Exception:
        failures.append("creation_timestamp_utc_invalid")
    artifact_size = payload.get("accepted_artifact_size")
    if (
        not isinstance(artifact_size, int)
        or isinstance(artifact_size, bool)
        or artifact_size <= 0
    ):
        failures.append("accepted_artifact_size_invalid")
    split_paths = payload.get("artifact_internal_split_field_paths")
    if not isinstance(split_paths, list) or not split_paths:
        failures.append("artifact_internal_split_field_paths_missing")
    dump_binding = payload.get("dump_run_binding")
    if not isinstance(dump_binding, Mapping) or not dump_binding:
        failures.append("dump_run_binding_missing")
    for field in (
        "checkpoint_identity",
        "tokenizer_identity",
        "original_run_directory",
        "original_producer_branch",
    ):
        if payload.get(field) in (None, ""):
            failures.append("{}_missing".format(field))
    if not allow_legacy_role_fixture:
        for field in (
            "original_child_run_identifier",
            "original_child_run_directory",
            "corrective_original_child_run_identifier",
            "corrective_original_child_run_identifier_source",
            "artifact_internal_original_run_identity_present",
            "artifact_internal_original_run_identity_status",
            "artifact_internal_run_identity_observation",
            "artifact_internal_run_identity_observation_sha256",
            "original_outer_run_directory",
            "current_audit_run_identifier",
            "dump_run_binding_identity",
            "child_run_directory_identity",
            "outer_run_directory_identity",
            "selected_fitting_command",
            "selected_fitting_command_sha256",
            "selected_fitting_command_field_provenance",
            "selected_dump_run_manifest",
            "selected_kv_manifest",
            "selected_hidden_manifest",
            "d2_command_binding",
            "outer_acceptance_normalized_field_provenance",
            "child_outer_relation_evidence",
            "original_child_exit_code_sidecar_path",
            "original_child_exit_code_sidecar_sha256",
            "original_full_run_exit_code_sidecar_path",
            "original_full_run_exit_code_sidecar_sha256",
            "original_outer_acceptance_exit_code_sidecar_path",
            "original_outer_acceptance_exit_code_sidecar_sha256",
            "original_outer_acceptance_exit_code_source_role",
            "original_outer_acceptance_exit_code_source_path",
            "population_evidence_historical_path",
            "population_evidence_historical_directory",
            "original_full_run_file_manifest_path",
            "original_full_run_file_manifest_sha256",
            "original_full_run_file_manifest_entry_count",
            "outer_acceptance_artifact_path_field",
            "nested_child_artifact_path_field",
        ):
            if payload.get(field) in (None, ""):
                failures.append("{}_missing".format(field))
        explicit_population_status = payload.get(
            "population_explicit_run_identifier_status"
        )
        explicit_population_id = payload.get(
            "population_explicit_run_identifier"
        )
        if explicit_population_status == "not_found":
            if explicit_population_id is not None:
                failures.append(
                    "population_explicit_run_identifier_absence_mismatch"
                )
        elif explicit_population_status == "found_and_verified":
            if (
                not isinstance(explicit_population_id, str)
                or not explicit_population_id
            ):
                failures.append(
                    "population_explicit_run_identifier_missing"
                )
        else:
            failures.append(
                "population_explicit_run_identifier_status_invalid"
            )
        if payload.get(
            "population_directory_derived_run_identity_status"
        ) != "candidate_pending_central_review":
            failures.append(
                "population_directory_derived_run_identity_status_invalid"
            )
        if payload.get("population_directory_derived_run_identity") in (
            None,
            "",
        ):
            failures.append("population_directory_derived_run_identity_missing")
        if payload.get("population_evidence_directory") != payload.get(
            "population_evidence_historical_directory"
        ):
            failures.append("population_evidence_directory_alias_mismatch")
        historical_population_directory = payload.get(
            "population_evidence_historical_directory"
        )
        expected_population_path_root = (
            str(
                PurePosixPath(
                    str(historical_population_directory)
                ).parent
            )
            if historical_population_directory
            else None
        )
        if payload.get(
            "population_path_root"
        ) != expected_population_path_root:
            failures.append("population_path_root_relation_mismatch")
        if payload.get("population_evidence_historical_path") != payload.get(
            "external_population_evidence_historical_path"
        ):
            failures.append(
                "population_evidence_historical_path_role_mismatch"
            )
        outer_run_identifier = payload.get("original_outer_run_identifier")
        outer_identity_status = payload.get(
            "original_outer_run_identity_status"
        )
        if outer_run_identifier in (None, ""):
            if outer_identity_status != (
                "directory_derived_only_pending_central_review"
            ):
                failures.append(
                    "original_outer_run_identity_status_invalid"
                )
        elif outer_identity_status != "explicit_evidence_identity":
            failures.append("original_outer_run_identity_status_invalid")
        if payload.get("original_child_run_identifier") == payload.get(
            "original_child_run_directory"
        ):
            failures.append(
                "original_child_run_identifier_directory_conflated"
            )
        if payload.get("original_child_exit_code_sidecar_value") != 0:
            failures.append(
                "original_child_exit_code_sidecar_value_mismatch"
            )
        if payload.get("original_full_run_exit_code") != 0:
            failures.append("original_full_run_exit_code_mismatch")
        if (
            payload.get(
                "original_outer_acceptance_exit_code_sidecar_value"
            )
            != 1
        ):
            failures.append(
                "original_outer_acceptance_exit_code_sidecar_value_mismatch"
            )

    for role, expected in expected_historical_values.items():
        if payload.get(
            "recomputed_historical_{}_digest".format(role)
        ) != expected:
            failures.append("historical_{}_digest_mismatch".format(role))
        if payload.get(
            "artifact_historical_{}_digest".format(role)
        ) != expected:
            failures.append(
                "artifact_historical_{}_digest_mismatch".format(role)
            )
        if payload.get(
            "historical_{}_digest_match".format(role)
        ) is not True:
            failures.append(
                "historical_{}_digest_match_not_true".format(role)
            )
    if payload.get("historical_digest_algorithm_identity") != (
        HISTORICAL_DIGEST_ALGORITHM_IDENTITY
    ):
        failures.append("historical_digest_algorithm_identity_mismatch")
    if payload.get("historical_digest_serialization") != (
        HISTORICAL_DIGEST_SERIALIZATION
    ):
        failures.append("historical_digest_serialization_mismatch")
    if payload.get("historical_digest_is_canonical_lf_set_sha") is not False:
        failures.append("historical_digest_canonical_equivalence_forbidden")
    if payload.get("historical_and_canonical_algorithms_distinct") is not True:
        failures.append("historical_canonical_algorithm_distinction_missing")

    for field, expected in (
        ("canonical_fitting_set_sha256", expected_canonical_values["fitting"]),
        ("canonical_heldout_set_sha256", expected_canonical_values["heldout"]),
        ("canonical_union_set_sha256", expected_canonical_values["union"]),
    ):
        if payload.get(field) != expected:
            failures.append("{}_mismatch".format(field))
    for field, expected in (
        ("artifact_fitting_count", expected_fitting_count),
        ("artifact_heldout_count", expected_heldout_count),
        ("duplicate_fitting_count", 0),
        ("duplicate_heldout_count", 0),
        ("intersection_count", 0),
        ("union_count", expected_union_count),
        ("artifact_train_count", expected_fitting_count),
        ("artifact_eval_count", expected_heldout_count),
        ("artifact_aligned_count", expected_union_count),
    ):
        if payload.get(field) != expected:
            failures.append("{}_mismatch".format(field))
    for field, expected in (
        ("split_algorithm", EXPECTED_SPLIT_ALGORITHM),
        ("split_unit", EXPECTED_SPLIT_UNIT),
        ("split_mode", EXPECTED_SPLIT_MODE),
        ("split_ratio", EXPECTED_SPLIT_RATIO),
        ("split_seed", EXPECTED_SPLIT_SEED),
        ("runtime_policy_sha256", ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256),
        (
            "hybrid_fitting_policy_sha256",
            ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256,
        ),
        (
            "stable_sample_id_algorithm_identity",
            STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
        ),
        ("dataset_name", EXPECTED_DATASET_NAME),
        ("dataset_config_name", EXPECTED_DATASET_CONFIG_NAME),
        ("dataset_split", EXPECTED_DATASET_SPLIT),
        ("text_column", EXPECTED_TEXT_COLUMN),
        ("summary_column", EXPECTED_SUMMARY_COLUMN),
    ):
        if payload.get(field) != expected:
            failures.append("{}_mismatch".format(field))

    reconstructed = payload.get("reconstructed_fitting_population_manifest")
    if not isinstance(reconstructed, Mapping):
        failures.append("reconstructed_fitting_population_manifest_missing")
        reconstructed = {}
    fitting_ids = list(
        reconstructed.get("artifact_fitting_stable_sample_ids") or []
    )
    heldout_ids = list(
        reconstructed.get("artifact_heldout_stable_sample_ids") or []
    )
    try:
        if len(fitting_ids) != expected_fitting_count:
            failures.append("reconstructed_fitting_count_mismatch")
        if len(heldout_ids) != expected_heldout_count:
            failures.append("reconstructed_heldout_count_mismatch")
        duplicate_fitting = sum(
            max(0, count - 1)
            for count in Counter(str(value) for value in fitting_ids).values()
        )
        duplicate_heldout = sum(
            max(0, count - 1)
            for count in Counter(str(value) for value in heldout_ids).values()
        )
        if duplicate_fitting:
            failures.append("reconstructed_fitting_duplicates")
        if duplicate_heldout:
            failures.append("reconstructed_heldout_duplicates")
        fitting_set = {str(value) for value in fitting_ids}
        heldout_set = {str(value) for value in heldout_ids}
        if fitting_set & heldout_set:
            failures.append("reconstructed_fitting_heldout_intersection_nonzero")
        if len(fitting_set | heldout_set) != expected_union_count:
            failures.append("reconstructed_union_count_mismatch")
        reconstructed_historical = {
            "train": historical_identity_digest(fitting_ids),
            "eval": historical_identity_digest(heldout_ids),
            "aligned": historical_identity_digest(
                sorted(fitting_set | heldout_set)
            ),
        }
        for role, digest in reconstructed_historical.items():
            if digest != payload.get(
                "recomputed_historical_{}_digest".format(role)
            ):
                failures.append(
                    "reconstructed_historical_{}_digest_mismatch".format(role)
                )
        if stable_sample_ids_sha256(fitting_ids, sort_ids=True) != payload.get(
            "canonical_fitting_set_sha256"
        ):
            failures.append("reconstructed_fitting_set_sha256_mismatch")
        if stable_sample_ids_sha256(heldout_ids, sort_ids=True) != payload.get(
            "canonical_heldout_set_sha256"
        ):
            failures.append("reconstructed_heldout_set_sha256_mismatch")
        if stable_sample_ids_sha256(
            heldout_ids, sort_ids=False
        ) != payload.get("canonical_heldout_dataset_order_sha256"):
            failures.append(
                "canonical_heldout_dataset_order_sha256_mismatch"
            )
        union_ids = sorted(set(fitting_ids) | set(heldout_ids))
        if stable_sample_ids_sha256(
            union_ids, sort_ids=True
        ) != payload.get("canonical_union_set_sha256"):
            failures.append("reconstructed_union_set_sha256_mismatch")
    except Exception as exc:
        failures.append(
            "reconstructed_population_identity_invalid:{}".format(
                type(exc).__name__
            )
        )

    if payload.get("original_child_execution_status") != "complete":
        failures.append("original_child_execution_status_mismatch")
    if payload.get("original_child_exit_code") != 0:
        failures.append("original_child_exit_code_mismatch")
    if payload.get("replay_matched_count") != 1427:
        failures.append("replay_matched_count_mismatch")
    if payload.get("replay_expected_count") != 1427:
        failures.append("replay_expected_count_mismatch")
    if payload.get("replay_matched_count") != payload.get(
        "replay_expected_count"
    ):
        failures.append("replay_matched_expected_count_mismatch")
    if (
        not allow_legacy_role_fixture
        and payload.get("replay_mismatch_count") != 0
    ):
        failures.append("replay_mismatch_count_not_zero")
    if (
        not allow_legacy_role_fixture
        and payload.get("replay_summary_supporting_role_path") is not None
    ):
        if payload.get("replay_summary_status") != "ok":
            failures.append("replay_summary_status_not_ok")
        if payload.get("replay_summary_comparison_status") != "matched":
            failures.append("replay_summary_comparison_status_not_matched")
        if payload.get("outer_acceptance_replay_summary_exists") is not True:
            failures.append(
                "outer_acceptance_replay_summary_exists_not_true"
            )
        if payload.get("outer_acceptance_replay_summary_path") != payload.get(
            "replay_summary_supporting_role_path"
        ):
            failures.append(
                "outer_acceptance_replay_summary_path_mismatch"
            )
    if payload.get("original_outer_acceptance_status") != "failed":
        failures.append("original_outer_acceptance_status_mismatch")
    if payload.get("original_outer_acceptance_exit_code") != 1:
        failures.append("original_outer_acceptance_exit_code_mismatch")
    if not allow_legacy_role_fixture:
        json_exit_present = payload.get(
            "outer_acceptance_json_exit_code_present"
        )
        json_exit_code = payload.get("outer_acceptance_json_exit_code")
        if json_exit_present is False:
            if json_exit_code is not None:
                failures.append(
                    "outer_acceptance_json_exit_code_absence_mismatch"
                )
        elif json_exit_present is True:
            if json_exit_code != payload.get(
                "original_outer_acceptance_exit_code"
            ):
                failures.append(
                    "outer_acceptance_json_exit_code_sidecar_mismatch"
                )
        else:
            failures.append(
                "outer_acceptance_json_exit_code_presence_invalid"
            )
        if payload.get(
            "original_outer_acceptance_exit_code_source_role"
        ) != "outer_acceptance_exit_code":
            failures.append(
                "outer_acceptance_exit_code_source_role_mismatch"
            )
    actual_outer_failures = payload.get("original_outer_acceptance_failures")
    if actual_outer_failures != list(EXPECTED_OUTER_ACCEPTANCE_FAILURES):
        failures.append("original_outer_acceptance_failures_mismatch")
    unresolved = payload.get("unresolved_fields")
    if unresolved not in ([], ()):
        failures.append("binding_affecting_unresolved_fields_present")
    audit_warnings = payload.get("audit_warnings")
    required_audit_warnings = [
        "pending_central_review",
        "original_outer_acceptance_failed",
        *EXPECTED_OUTER_ACCEPTANCE_FAILURES,
    ]
    if not isinstance(audit_warnings, list) or any(
        warning not in audit_warnings
        for warning in required_audit_warnings
    ):
        failures.append("corrective_manifest_audit_warnings_incomplete")

    return {
        "schema_version": 1,
        "status": "ok" if not failures else "failed",
        "corrective_binding_draft_valid": not failures,
        "failures": failures,
        "independent_artifact_validation_status": (
            independent_artifact.get("status")
        ),
        "independent_artifact_path": independent_artifact.get(
            "artifact_path"
        ),
        "independent_artifact_sha256_before_load": (
            independent_artifact.get(
                "artifact_file_sha256_before_load"
            )
        ),
        "independent_artifact_sha256_after_load": (
            independent_artifact.get(
                "artifact_file_sha256_after_load"
            )
        ),
        "independent_artifact_load_preserved_bytes": (
            independent_artifact.get("artifact_load_preserved_bytes")
        ),
        "central_approval_valid": False,
        "paper_candidate_valid": False,
        "gpu_authorized": False,
    }


def write_corrective_binding_manifest_draft(
    output_dir: Path | str,
    payload: Mapping[str, Any],
    *,
    accepted_artifact_path: Optional[Path | str] = None,
    _artifact_loader: Optional[
        Callable[[Path], Mapping[str, Any]]
    ] = None,
    _test_expectations: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    validation = validate_corrective_binding_manifest_draft(
        payload,
        accepted_artifact_path=accepted_artifact_path,
        _artifact_loader=_artifact_loader,
        _test_expectations=_test_expectations,
    )
    if validation.get("status") != "ok":
        raise ValueError(
            "corrective_binding_manifest_invalid:{}".format(
                ",".join(validation.get("failures") or [])
            )
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "corrective_binding_manifest_draft.json"
    _write_json(manifest_path, payload)
    digest = sha256_file(manifest_path)
    digest_path = output_dir / "corrective_binding_manifest_draft.json.sha256"
    digest_path.write_text(
        "{}  {}\n".format(digest, manifest_path.name),
        encoding="ascii",
    )
    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "sha256_path": str(digest_path),
        "central_approval_valid": False,
        "paper_candidate_valid": False,
        "gpu_authorized": False,
    }


def _validator_expectations_from_approval_contract(
    approval_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map CENTRAL_APPROVAL_CONTRACT-shaped identities onto the partial
    override shape accepted by validate_corrective_binding_manifest_draft's
    `_test_expectations`.  Every field left out here (population evidence
    SHA, historical digests, role specs, ...) falls back to that function's
    own existing module-level defaults, which this preflight does not
    duplicate or second-guess."""

    return {
        "accepted_artifact_sha256": approval_contract["accepted_artifact_sha256"],
        "audit_bundle_sha256": approval_contract["accepted_audit_bundle_sha256"],
        "canonical": {
            "fitting": approval_contract["fitting_set_sha256"],
            "heldout": approval_contract["heldout_set_sha256"],
            "union": approval_contract["union_set_sha256"],
        },
        "fitting_count": approval_contract["fitting_count"],
        "heldout_count": approval_contract["heldout_count"],
        "union_count": approval_contract["union_count"],
    }


def _paper_facing_corrective_binding_preflight_impl(
    manifest_path: Path | str,
    *,
    approval_contract: Mapping[str, Any],
    validator_expectations: Optional[Mapping[str, Any]] = None,
    accepted_artifact_path: Optional[Path | str] = None,
    _artifact_loader: Optional[
        Callable[[Path], Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """Shared implementation. Not exported: production code must go through
    paper_facing_corrective_binding_preflight(), which always supplies
    CENTRAL_APPROVAL_CONTRACT and never accepts caller-controlled overrides.
    This helper exists so small synthetic fixtures can exercise the same
    logic against a self-consistent stand-in contract in tests, without
    ever letting a caller redirect the production approval identities."""

    def _fail(
        stage: str,
        extra_failures: Sequence[str],
        *,
        manifest_sha256: Optional[str],
        corrective_binding_draft_valid: bool = False,
    ) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "failed",
            "failure_stage": stage,
            "failures": sorted(set(extra_failures)),
            "corrective_binding_draft_valid": corrective_binding_draft_valid,
            "manifest_sha256": manifest_sha256,
            "central_root_of_trust_configured": True,
            "central_root_of_trust_validated": False,
            "approved_corrective_root_preflight": "FAIL",
            "central_approval_valid": False,
            "paper_candidate_valid": False,
            "gpu_authorized": False,
        }

    manifest_path = Path(manifest_path)
    try:
        manifest_sha256 = sha256_file(manifest_path)
    except OSError as exc:
        return _fail(
            "corrective_manifest_unreadable",
            [
                "corrective_manifest_unreadable:{}".format(
                    type(exc).__name__
                )
            ],
            manifest_sha256=None,
        )

    failures: List[str] = []
    expected_manifest_sha256 = str(
        approval_contract["corrective_manifest_sha256"]
    )
    if manifest_sha256 != expected_manifest_sha256:
        failures.append("approved_corrective_manifest_sha256_mismatch")

    try:
        payload = _json_load(manifest_path)
    except Exception as exc:
        return _fail(
            "corrective_manifest_parse_failed",
            failures + [
                "corrective_manifest_parse_failed:{}".format(
                    type(exc).__name__
                )
            ],
            manifest_sha256=manifest_sha256,
        )
    if not isinstance(payload, Mapping):
        failures.append("corrective_manifest_not_json_object")
        payload = {}

    validation = validate_corrective_binding_manifest_draft(
        payload,
        accepted_artifact_path=accepted_artifact_path,
        _artifact_loader=_artifact_loader,
        _test_expectations=validator_expectations,
    )
    if validation.get("status") != "ok":
        failures.extend(validation.get("failures") or [])

    expected_chain_sha256 = str(
        approval_contract["corrective_evidence_chain_sha256"]
    )
    if payload.get("multi_file_evidence_chain_sha256") != expected_chain_sha256:
        failures.append("approved_corrective_evidence_chain_sha256_mismatch")

    expected_dataset_order_sha256 = str(
        approval_contract["heldout_dataset_order_sha256"]
    )
    if payload.get("canonical_heldout_dataset_order_sha256") != (
        expected_dataset_order_sha256
    ):
        failures.append("approved_heldout_dataset_order_sha256_mismatch")

    failures = sorted(set(failures))
    status_ok = not failures
    return {
        "schema_version": 1,
        "status": "ok" if status_ok else "failed",
        "failure_stage": (
            None if status_ok else "approved_corrective_root_preflight"
        ),
        "failures": failures,
        "corrective_binding_draft_valid": validation.get("status") == "ok",
        "manifest_sha256": manifest_sha256,
        "central_root_of_trust_configured": True,
        "central_root_of_trust_validated": status_ok,
        "approved_corrective_root_preflight": (
            "PASS" if status_ok else "FAIL"
        ),
        "central_approval_contract": dict(approval_contract),
        "central_approval_valid": status_ok,
        "paper_candidate_valid": False,
        "gpu_authorized": False,
    }


def paper_facing_corrective_binding_preflight(
    manifest_path: Path | str,
    *,
    accepted_artifact_path: Optional[Path | str] = None,
    caller_supplied_approved_sha256: Optional[str] = None,
    _artifact_loader: Optional[
        Callable[[Path], Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """Pin the immutable accepted corrective manifest to the centrally
    approved root-of-trust identities in CENTRAL_APPROVAL_CONTRACT.

    This is the production entrypoint. It accepts no override for any
    approved SHA, population count, or canonical population identity: it
    always validates against CENTRAL_APPROVAL_CONTRACT, never a
    caller-supplied or test-only substitute. (Model-free unit tests that
    need a small synthetic fixture must call
    _paper_facing_corrective_binding_preflight_impl directly with their own
    self-consistent stand-in contract; that helper is intentionally not
    exported.)

    Approval is represented purely as an external SHA-256 match against
    the manifest's own file bytes plus its embedded evidence-chain and
    held-out dataset-order identities. The manifest bytes themselves
    (including its pending-review approval_status and false gate fields)
    are read-only and are never rewritten. A caller-supplied approval SHA
    is always rejected, regardless of whether the manifest itself would
    otherwise validate.

    On success this reports only APPROVED_CORRECTIVE_ROOT_PREFLIGHT=PASS.
    It never implies GPU authorization, F2b readiness, or paper-result
    acceptance. A failed or unreadable manifest never reports
    central_root_of_trust_validated=True.
    """

    result = _paper_facing_corrective_binding_preflight_impl(
        manifest_path,
        approval_contract=CENTRAL_APPROVAL_CONTRACT,
        validator_expectations=_validator_expectations_from_approval_contract(
            CENTRAL_APPROVAL_CONTRACT
        ),
        accepted_artifact_path=accepted_artifact_path,
        _artifact_loader=_artifact_loader,
    )
    if caller_supplied_approved_sha256 is not None:
        result = dict(result)
        result["failures"] = sorted(
            set(result.get("failures") or [])
            | {"caller_supplied_central_approval_sha_forbidden"}
        )
        result["status"] = "failed"
        result["failure_stage"] = "caller_supplied_central_approval_sha_forbidden"
        result["approved_corrective_root_preflight"] = "FAIL"
        result["central_root_of_trust_validated"] = False
        result["central_approval_valid"] = False
    return result


__all__ = [
    "ACCEPTED_ARTIFACT_IDENTITY_SIDECAR_SHA256",
    "ACCEPTED_AUDIT_ROLE_SPECS",
    "ACCEPTED_CHILD_FULL_EXIT_CODE_SHA256",
    "ACCEPTED_FULL_RUN_EXIT_CODE_SHA256",
    "ACCEPTED_FULL_RUN_FILE_MANIFEST_SHA256",
    "ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256",
    "ACCEPTED_FITTING_PROVENANCE_AUDIT_BUNDLE_SHA256",
    "ACCEPTED_GIT_IDENTITY_SIDECAR_SHA256",
    "ACCEPTED_OUTER_ACCEPTANCE_EXIT_CODE_SHA256",
    "ACCEPTED_OUTER_FULL_ACCEPTANCE_SHA256",
    "APPROVED_CORRECTIVE_MANIFEST_SHA256",
    "APPROVED_CORRECTIVE_EVIDENCE_CHAIN_SHA256",
    "APPROVED_HELDOUT_DATASET_ORDER_SHA256",
    "APPROVED_RESUME_REVIEW_BUNDLE_SHA256",
    "CANONICAL_SET_ALGORITHM_IDENTITY",
    "CENTRAL_APPROVAL_CONTRACT",
    "CORRECTIVE_BINDING_APPROVAL_STATUS",
    "CORRECTIVE_BINDING_EVIDENCE_TYPE",
    "CORRECTIVE_BINDING_PROVENANCE_CONSTRUCTION",
    "CORRECTIVE_BINDING_SCHEMA_VERSION",
    "EXPECTED_CANONICAL_FITTING_SET_SHA256",
    "EXPECTED_CANONICAL_HELDOUT_SET_SHA256",
    "EXPECTED_CANONICAL_UNION_SET_SHA256",
    "EXPECTED_HISTORICAL_ALIGNED_DIGEST",
    "EXPECTED_HISTORICAL_EVAL_DIGEST",
    "EXPECTED_HISTORICAL_TRAIN_DIGEST",
    "EXPECTED_OUTER_ACCEPTANCE_FAILURES",
    "HISTORICAL_DIGEST_ALGORITHM_IDENTITY",
    "HISTORICAL_DIGEST_SERIALIZATION",
    "LEGACY_REQUIRED_EVIDENCE_ROLES",
    "REQUIRED_EVIDENCE_ROLES",
    "ROLE_RELATIONS",
    "build_corrective_binding_manifest_draft",
    "compute_multi_file_evidence_chain_sha256",
    "derive_evidence_role_inventory_from_verified_audit",
    "extract_artifact_provenance_identities",
    "find_role_inventory_in_verified_audit",
    "historical_format_identity",
    "historical_identity_digest",
    "load_and_resolve_authoritative_samsum_population",
    "load_explicit_evidence_role_inventory",
    "paper_facing_corrective_binding_preflight",
    "parse_external_population_comparison",
    "resolve_authoritative_samsum_population_rows",
    "validate_corrective_binding_manifest_draft",
    "validate_multi_file_evidence_chain",
    "verify_accepted_artifact_split_commitment",
    "verify_accepted_fitting_audit_bundle",
    "verify_historical_digest_binding",
    "write_corrective_binding_manifest_draft",
]
