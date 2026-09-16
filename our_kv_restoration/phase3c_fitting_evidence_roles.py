"""Exact evidence-role binding for the accepted Phase 3c fitting audit.

The accepted chain spans several runs.  This module binds each evidence
role to its own file, parser, and identity domain instead of copying global
audit-summary values into every role.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .missing_kv_dump_provenance import canonical_json_sha256, sha256_file
from .missing_kv_paper_population import (
    ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256,
    ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256,
    stable_sample_ids_sha256,
)


ACCEPTED_ARTIFACT_IDENTITY_SIDECAR_SHA256 = (
    "d51da367540552260db050a00065f33303b6d3c1fd69eb351e2790aba10396de"
)
ACCEPTED_OUTER_FULL_ACCEPTANCE_SHA256 = (
    "05169bad37f3fc5ff2a61458633f431faad35423aca6cc2b084c3373c5384ac6"
)
ACCEPTED_GIT_IDENTITY_SIDECAR_SHA256 = (
    "dd3e6ffa14b0cb1a57223b869e954219785744ddf8fce87fc35ce1b04ffd48b0"
)
ACCEPTED_FULL_RUN_EXIT_CODE_SHA256 = (
    "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
)
# Compatibility alias for the original role name. The physical sidecar belongs
# to the outer wrapper even though it reports completion of the child fit.
ACCEPTED_CHILD_FULL_EXIT_CODE_SHA256 = ACCEPTED_FULL_RUN_EXIT_CODE_SHA256
ACCEPTED_OUTER_ACCEPTANCE_EXIT_CODE_SHA256 = (
    "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
)
ACCEPTED_FULL_RUN_FILE_MANIFEST_SHA256 = (
    "4245459d79e068c4d2c24ef4c0c3879e3049b59da67b65c364d7539b08009f51"
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

EXPECTED_OUTER_ACCEPTANCE_FAILURES = (
    "replay_run_config:max_train_records_per_group_not_zero",
    "replay_run_config:max_eval_records_per_threshold_not_zero",
)

ROLE_BINDING_SCHEMA_VERSION = 2
EXACT_ROLE_BINDING_METHOD = "exact_frozen_role_spec"

IDENTITY_DOMAIN_POPULATION = "population_source_run"
IDENTITY_DOMAIN_CHILD = "original_child_fitting_run"
IDENTITY_DOMAIN_OUTER = "original_outer_acceptance_run"
IDENTITY_DOMAIN_AUDIT = "current_audit_preflight_run"

PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER = "child_run_under_outer"
PATH_AUTHORITY_OUTER_RUN_ROOT = "outer_run_root"
PATH_AUTHORITY_POPULATION_RUN_PARENT = "population_run_parent"
PATH_AUTHORITY_AUDIT_ARCHIVE_MEMBER = "audit_archive_member"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PATHISH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")


def _spec(
    *,
    basename: Optional[str],
    identity_domain: str,
    path_authority_domain: str,
    path_relation: str,
    parser: str,
    relation: str,
    expected_sha256: Optional[str] = None,
    expected_relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "expected_sha256": expected_sha256,
        "expected_basename": basename,
        "expected_relative_path": expected_relative_path,
        "identity_domain": identity_domain,
        "path_authority_domain": path_authority_domain,
        "path_relation": path_relation,
        "parser": parser,
        "relation_supported": relation,
    }


ACCEPTED_AUDIT_ROLE_SPECS: Dict[str, Dict[str, Any]] = {
    "external_population_comparison": _spec(
        basename="calm_calibration_population_comparison_summary.json",
        identity_domain=IDENTITY_DOMAIN_POPULATION,
        path_authority_domain=PATH_AUTHORITY_POPULATION_RUN_PARENT,
        path_relation="population_run_file_under_population_parent",
        parser="external_population_comparison_v1",
        relation="input_population",
        expected_sha256=ACCEPTED_EXTERNAL_POPULATION_EVIDENCE_SHA256,
    ),
    "original_fitting_command": _spec(
        basename="command.txt",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER,
        path_relation="child_run_command_for_child_fit",
        parser="original_fitting_command_v1",
        relation="fitting_invocation",
    ),
    "fit_summary": _spec(
        basename="phase3c_final_policy_summary.json",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER,
        path_relation="child_run_file_under_outer",
        parser="phase3c_fit_summary_v1",
        relation="fitting_result",
    ),
    "artifact_summary": _spec(
        basename="calm_hybrid_multisource_policy_artifact_summary.json",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER,
        path_relation="child_run_file_under_outer",
        parser="phase3c_artifact_summary_v1",
        relation="artifact_output",
    ),
    "artifact_identity": _spec(
        basename="artifact_identity.txt",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_sidecar_describing_child_artifact",
        parser="artifact_identity_sidecar_v1",
        relation="artifact_identity_sidecar",
        expected_sha256=ACCEPTED_ARTIFACT_IDENTITY_SIDECAR_SHA256,
    ),
    "child_final_status": _spec(
        basename="final_status.json",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER,
        path_relation="child_run_file_under_outer",
        parser="child_final_status_v1",
        relation="successful_child_completion",
    ),
    "outer_full_acceptance": _spec(
        basename="full_acceptance.json",
        identity_domain=IDENTITY_DOMAIN_OUTER,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_file",
        parser="outer_full_acceptance_v1",
        relation="outer_acceptance_state",
        expected_sha256=ACCEPTED_OUTER_FULL_ACCEPTANCE_SHA256,
    ),
    "current_artifact_preflight": _spec(
        basename="artifact_metadata_summary.json",
        identity_domain=IDENTITY_DOMAIN_AUDIT,
        path_authority_domain=PATH_AUTHORITY_AUDIT_ARCHIVE_MEMBER,
        path_relation="audit_archive_root_member",
        parser="current_artifact_preflight_v1",
        relation="current_artifact_identity",
    ),
    "producer_git_identity": _spec(
        basename="git_identity.txt",
        identity_domain=IDENTITY_DOMAIN_OUTER,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_file",
        parser="producer_git_identity_v1",
        relation="producer_identity",
        expected_sha256=ACCEPTED_GIT_IDENTITY_SIDECAR_SHA256,
    ),
    "child_full_exit_code": _spec(
        basename="full_exit_code.txt",
        identity_domain=IDENTITY_DOMAIN_OUTER,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_file",
        parser="child_full_exit_code_v1",
        relation="outer_wrapper_child_fitting_completion",
        expected_sha256=ACCEPTED_FULL_RUN_EXIT_CODE_SHA256,
    ),
    "outer_acceptance_exit_code": _spec(
        basename="full_acceptance_exit_code.txt",
        identity_domain=IDENTITY_DOMAIN_OUTER,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_file",
        parser="outer_acceptance_exit_code_v1",
        relation="outer_acceptance_execution_code",
        expected_sha256=ACCEPTED_OUTER_ACCEPTANCE_EXIT_CODE_SHA256,
    ),
    "file_manifest": _spec(
        basename="full_run_file_manifest.txt",
        identity_domain=IDENTITY_DOMAIN_OUTER,
        path_authority_domain=PATH_AUTHORITY_OUTER_RUN_ROOT,
        path_relation="outer_root_file",
        parser="original_file_manifest_v1",
        relation="original_file_inventory",
        expected_sha256=ACCEPTED_FULL_RUN_FILE_MANIFEST_SHA256,
    ),
}
ACCEPTED_AUDIT_ROLE_SPECS["external_population_comparison"].update(
    {
        "require_exact_population_lists": True,
        "expected_train_count": 409,
        "expected_eval_count": 409,
        "expected_aligned_count": 818,
        "expected_train_digest": EXPECTED_HISTORICAL_TRAIN_DIGEST,
        "expected_eval_digest": EXPECTED_HISTORICAL_EVAL_DIGEST,
        "expected_aligned_digest": EXPECTED_HISTORICAL_ALIGNED_DIGEST,
        "expected_canonical_fitting_set_sha256": (
            EXPECTED_CANONICAL_FITTING_SET_SHA256
        ),
        "expected_canonical_heldout_set_sha256": (
            EXPECTED_CANONICAL_HELDOUT_SET_SHA256
        ),
        "expected_canonical_union_set_sha256": (
            EXPECTED_CANONICAL_UNION_SET_SHA256
        ),
    }
)

REQUIRED_EVIDENCE_ROLES = tuple(ACCEPTED_AUDIT_ROLE_SPECS)
SUPPORTING_AUDIT_ROLE_SPECS: Dict[str, Dict[str, Any]] = {
    "replay_summary": _spec(
        basename="phase3c_artifact_replay_summary.json",
        identity_domain=IDENTITY_DOMAIN_CHILD,
        path_authority_domain=PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER,
        path_relation="child_run_file_under_outer",
        parser="artifact_replay_summary_v1",
        relation="replay_population_validation",
    ),
}
SUPPORTING_AUDIT_ROLE_SPECS["replay_summary"][
    "required_for_paper"
] = False
_ROLE_ORDER = {
    role: index
    for index, role in enumerate(
        REQUIRED_EVIDENCE_ROLES + tuple(SUPPORTING_AUDIT_ROLE_SPECS)
    )
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("json_root_not_mapping")
    return payload


def _value_at(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _unique_observed(
    payload: Mapping[str, Any],
    candidates: Sequence[Tuple[str, Sequence[str]]],
    *,
    label: str,
    failures: List[str],
) -> Tuple[Any, Optional[str]]:
    observed: List[Tuple[Any, str]] = []
    for field_path, parts in candidates:
        value = _value_at(payload, parts)
        if value not in (None, ""):
            observed.append((value, field_path))
    unique = {json.dumps(value, sort_keys=True, default=str) for value, _ in observed}
    if len(unique) > 1:
        failures.append("{}_conflict".format(label))
        return None, None
    return observed[0] if observed else (None, None)


def _replay_counts(
    payload: Mapping[str, Any],
    *,
    failures: List[str],
    label: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    containers = (
        ("$", ()),
        ("$.replay_validation", ("replay_validation",)),
        ("$.artifact_replay_validation", ("artifact_replay_validation",)),
        ("$.replay_summary", ("replay_summary",)),
        ("$.artifact_replay_summary", ("artifact_replay_summary",)),
        ("$.validation", ("validation",)),
        ("$.comparison", ("comparison",)),
        ("$.reference_comparison", ("reference_comparison",)),
    )
    values: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for field, aliases in (
        (
            "replay_matched_count",
            (
                "replay_matched_count",
                "matched_count",
                "matched_record_count",
                "matching_record_count",
            ),
        ),
        (
            "replay_expected_count",
            (
                "replay_expected_count",
                "expected_count",
                "expected_record_count",
                "total_record_count",
            ),
        ),
        (
            "replay_mismatch_count",
            (
                "replay_mismatch_count",
                "mismatch_count",
                "mismatched_count",
                "mismatched_record_count",
            ),
        ),
    ):
        observed: List[Tuple[Any, str]] = []
        for container_path, container_parts in containers:
            container = _value_at(payload, container_parts)
            if container_parts == ():
                container = payload
            if not isinstance(container, Mapping):
                continue
            for alias in aliases:
                value = container.get(alias)
                if value is not None:
                    observed.append(
                        (
                            value,
                            "{}.{}".format(container_path, alias).replace(
                                "$.", "$."
                            ),
                        )
                    )
        unique = {value for value, _ in observed}
        if len(unique) > 1:
            failures.append("{}_{}_conflict".format(label, field))
        elif observed:
            values[field] = observed[0][0]
            provenance[field] = observed[0][1]
    return values, provenance


def _outer_acceptance_replay_counts(
    payload: Mapping[str, Any],
    *,
    failures: List[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    comparison_status_counts = payload.get("comparison_status_counts")
    comparison_status_counts = (
        comparison_status_counts
        if isinstance(comparison_status_counts, Mapping)
        else {}
    )
    actual_schema_present = (
        "comparison_row_count" in payload
        or "comparison_status_counts" in payload
    )
    if not actual_schema_present:
        values, provenance = _replay_counts(
            payload,
            failures=failures,
            label="outer_full_acceptance",
        )
        expected = values.get("replay_expected_count")
        matched = values.get("replay_matched_count")
        if (
            "replay_mismatch_count" not in values
            and isinstance(expected, int)
            and not isinstance(expected, bool)
            and isinstance(matched, int)
            and not isinstance(matched, bool)
        ):
            values["replay_mismatch_count"] = expected - matched
            provenance["replay_mismatch_count"] = (
                "derived:legacy-replay-expected-minus-matched"
            )
        return values, provenance

    expected = payload.get("comparison_row_count")
    matched = comparison_status_counts.get("matched")
    for field, value in (
        ("comparison_row_count", expected),
        ("comparison_status_counts_matched", matched),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            failures.append("outer_full_acceptance_{}_invalid".format(field))

    mismatch = (
        expected - matched
        if isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(matched, int)
        and not isinstance(matched, bool)
        else None
    )
    if isinstance(mismatch, int) and mismatch < 0:
        failures.append("outer_full_acceptance_replay_matched_exceeds_expected")

    explicit_mismatch, explicit_mismatch_path = _unique_observed(
        payload,
        (
            ("$.comparison_mismatch_count", ("comparison_mismatch_count",)),
            (
                "$.comparison_status_counts.mismatch",
                ("comparison_status_counts", "mismatch"),
            ),
            (
                "$.comparison_status_counts.mismatched",
                ("comparison_status_counts", "mismatched"),
            ),
        ),
        label="outer_full_acceptance_replay_mismatch_count",
        failures=failures,
    )
    if explicit_mismatch is not None:
        if (
            not isinstance(explicit_mismatch, int)
            or isinstance(explicit_mismatch, bool)
            or explicit_mismatch < 0
        ):
            failures.append(
                "outer_full_acceptance_replay_mismatch_count_invalid"
            )
        elif mismatch is not None and explicit_mismatch != mismatch:
            failures.append(
                "outer_full_acceptance_replay_mismatch_count_conflict"
            )

    values = {
        "replay_expected_count": expected,
        "replay_matched_count": matched,
        "replay_mismatch_count": mismatch,
    }
    provenance = {
        "replay_expected_count": "$.comparison_row_count",
        "replay_matched_count": "$.comparison_status_counts.matched",
        "replay_mismatch_count": (
            explicit_mismatch_path
            if explicit_mismatch_path is not None
            else "derived:$.comparison_row_count-"
            "$.comparison_status_counts.matched"
        ),
    }
    return values, provenance


def _parse_kv_text(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:=|:)\s*(.*?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace("-", "_")
        value = match.group(2).strip().strip("\"'")
        if key in values and values[key] != value:
            raise ValueError("duplicate_key_conflict:{}".format(key))
        values[key] = value
    return values


def _base_result(
    *,
    parser: str,
    failures: List[str],
    normalized: Mapping[str, Any],
    provenance: Mapping[str, str],
    observed: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "parser": parser,
        "normalized_fields": dict(normalized),
        "normalized_field_provenance": dict(provenance),
    }
    result.update(dict(observed or {}))
    return result


def _parse_external_population(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="external_population_comparison_v1",
            failures=["external_population_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    split = payload.get("frozen_split")
    provenance_payload = payload.get("provenance")
    split = split if isinstance(split, Mapping) else {}
    provenance_payload = (
        provenance_payload if isinstance(provenance_payload, Mapping) else {}
    )
    run_id, run_id_path = _unique_observed(
        payload,
        (
            ("$.provenance.run_identity", ("provenance", "run_identity")),
            ("$.provenance.run_id", ("provenance", "run_id")),
            ("$.run_identity", ("run_identity",)),
        ),
        label="external_population_explicit_run_identifier",
        failures=failures,
    )
    commit = provenance_payload.get("git_commit")
    run_id_valid = (
        isinstance(run_id, str)
        and bool(run_id)
        and not _path_is_run_identifier(run_id)
    )
    if run_id is not None and not run_id_valid:
        failures.append("external_population_explicit_run_identifier_invalid")
    explicit_run_identifier_status = (
        "found_and_verified"
        if run_id_valid
        else "not_found" if run_id is None else "invalid"
    )
    raw_train_ids = split.get("train_stable_sample_ids")
    raw_eval_ids = split.get("eval_stable_sample_ids")
    population_lists_present = (
        raw_train_ids is not None or raw_eval_ids is not None
    )
    train_ids: Optional[List[str]] = None
    eval_ids: Optional[List[str]] = None
    duplicate_train_count: Optional[int] = None
    duplicate_eval_count: Optional[int] = None
    overlap_count: Optional[int] = None
    union_count: Optional[int] = None
    historical_train_digest: Optional[str] = None
    historical_eval_digest: Optional[str] = None
    historical_aligned_digest: Optional[str] = None
    canonical_fitting_sha: Optional[str] = None
    canonical_heldout_sha: Optional[str] = None
    canonical_union_sha: Optional[str] = None
    if population_lists_present:
        if not isinstance(raw_train_ids, list):
            failures.append("external_population_fitting_ids_invalid_type")
        if not isinstance(raw_eval_ids, list):
            failures.append("external_population_heldout_ids_invalid_type")
        if isinstance(raw_train_ids, list) and isinstance(raw_eval_ids, list):
            if any(
                not isinstance(value, str) or not value
                for value in raw_train_ids
            ):
                failures.append("external_population_fitting_id_invalid")
            if any(
                not isinstance(value, str) or not value
                for value in raw_eval_ids
            ):
                failures.append("external_population_heldout_id_invalid")
            train_ids = [str(value) for value in raw_train_ids]
            eval_ids = [str(value) for value in raw_eval_ids]
            duplicate_train_count = sum(
                max(0, count - 1)
                for count in Counter(train_ids).values()
            )
            duplicate_eval_count = sum(
                max(0, count - 1)
                for count in Counter(eval_ids).values()
            )
            train_set = set(train_ids)
            eval_set = set(eval_ids)
            overlap = train_set & eval_set
            union = train_set | eval_set
            overlap_count = len(overlap)
            union_count = len(union)
            if duplicate_train_count:
                failures.append("external_population_fitting_duplicates")
            if duplicate_eval_count:
                failures.append("external_population_heldout_duplicates")
            if overlap_count:
                failures.append("external_population_intersection_nonzero")
            historical_train_digest = _historical_identity_digest(train_ids)
            historical_eval_digest = _historical_identity_digest(eval_ids)
            historical_aligned_digest = _historical_identity_digest(
                sorted(union)
            )
            canonical_fitting_sha = (
                stable_sample_ids_sha256(train_ids, sort_ids=True)
                if not duplicate_train_count
                else None
            )
            canonical_heldout_sha = (
                stable_sample_ids_sha256(eval_ids, sort_ids=True)
                if not duplicate_eval_count
                else None
            )
            canonical_union_sha = stable_sample_ids_sha256(
                sorted(union), sort_ids=True
            )
            for label, stored, recomputed in (
                (
                    "train",
                    split.get("train_stable_sample_id_sha256"),
                    historical_train_digest,
                ),
                (
                    "eval",
                    split.get("eval_stable_sample_id_sha256"),
                    historical_eval_digest,
                ),
                (
                    "aligned",
                    split.get("aligned_stable_sample_id_sha256"),
                    historical_aligned_digest,
                ),
            ):
                if stored != recomputed:
                    failures.append(
                        "external_population_historical_{}_digest_mismatch".format(
                            label
                        )
                    )
            for field, actual in (
                ("train_stable_sample_id_count", len(train_ids)),
                ("eval_stable_sample_id_count", len(eval_ids)),
                ("aligned_stable_sample_id_count", union_count),
            ):
                declared = split.get(field)
                if (
                    not isinstance(declared, int)
                    or isinstance(declared, bool)
                    or declared != actual
                ):
                    failures.append(
                        "external_population_declared_{}_mismatch".format(
                            field
                        )
                    )
            if (
                "train_eval_overlap_count" in split
                and split.get("train_eval_overlap_count") != overlap_count
            ):
                failures.append(
                    "external_population_declared_"
                    "train_eval_overlap_count_mismatch"
                )
    normalized = {
        "population_source_run_identifier": run_id,
        "explicit_run_identifier_status": explicit_run_identifier_status,
        "explicit_run_identifier": run_id,
        "train_count": split.get("train_stable_sample_id_count"),
        "eval_count": split.get("eval_stable_sample_id_count"),
        "aligned_count": split.get("aligned_stable_sample_id_count"),
        "train_digest": split.get("train_stable_sample_id_sha256"),
        "eval_digest": split.get("eval_stable_sample_id_sha256"),
        "aligned_digest": split.get("aligned_stable_sample_id_sha256"),
        "split_algorithm": split.get("split_algorithm"),
        "split_unit": split.get("split_unit"),
        "split_mode": split.get("split_mode"),
        "split_ratio": split.get("split_ratio"),
        "split_seed": split.get("split_seed"),
        "population_lists_present": population_lists_present,
        "train_stable_sample_ids": train_ids,
        "eval_stable_sample_ids": eval_ids,
        "duplicate_train_count": duplicate_train_count,
        "duplicate_eval_count": duplicate_eval_count,
        "overlap_count": overlap_count,
        "union_count": union_count,
        "historical_train_digest": historical_train_digest,
        "historical_eval_digest": historical_eval_digest,
        "historical_aligned_digest": historical_aligned_digest,
        "canonical_fitting_set_sha256": canonical_fitting_sha,
        "canonical_heldout_set_sha256": canonical_heldout_sha,
        "canonical_union_set_sha256": canonical_union_sha,
        "historical_digest_is_canonical_lf_set_sha": False,
    }
    return _base_result(
        parser="external_population_comparison_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "population_source_run_identifier": run_id_path,
            "explicit_run_identifier": run_id_path,
            "explicit_run_identifier_status": (
                run_id_path or "derived:explicit-run-identifier-absence"
            ),
            "train_count": "$.frozen_split.train_stable_sample_id_count",
            "eval_count": "$.frozen_split.eval_stable_sample_id_count",
            "aligned_count": "$.frozen_split.aligned_stable_sample_id_count",
            "train_digest": "$.frozen_split.train_stable_sample_id_sha256",
            "eval_digest": "$.frozen_split.eval_stable_sample_id_sha256",
            "aligned_digest": "$.frozen_split.aligned_stable_sample_id_sha256",
            "train_stable_sample_ids": (
                "$.frozen_split.train_stable_sample_ids"
            ),
            "eval_stable_sample_ids": (
                "$.frozen_split.eval_stable_sample_ids"
            ),
            "historical_train_digest": (
                "$.frozen_split.train_stable_sample_ids"
            ),
            "historical_eval_digest": (
                "$.frozen_split.eval_stable_sample_ids"
            ),
            "historical_aligned_digest": (
                "$.frozen_split.train_stable_sample_ids"
                "+$.frozen_split.eval_stable_sample_ids"
            ),
            "canonical_fitting_set_sha256": (
                "$.frozen_split.train_stable_sample_ids"
            ),
            "canonical_heldout_set_sha256": (
                "$.frozen_split.eval_stable_sample_ids"
            ),
            "canonical_union_set_sha256": (
                "$.frozen_split.train_stable_sample_ids"
                "+$.frozen_split.eval_stable_sample_ids"
            ),
        },
        observed={
            "observed_population_source_run_identity": run_id,
            "observed_producer_commit": commit,
        },
    )


def _historical_identity_digest(identities: Sequence[str]) -> str:
    formatted = [
        "|".join(str(part) for part in identity)
        for identity in sorted(identities)
    ]
    return hashlib.sha256(
        "\n".join(formatted).encode("utf-8")
    ).hexdigest()


def _command_option(
    tokens: Sequence[str],
    names: Sequence[str],
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    for index, token in enumerate(tokens):
        for name in names:
            if token == name and index + 1 < len(tokens):
                return tokens[index + 1], index, name
            prefix = name + "="
            if token.startswith(prefix):
                return token[len(prefix) :], index, name
    return None, None, None


def _typed_command_value(
    value: Optional[str],
    value_type: str,
) -> Any:
    if value is None:
        return None
    try:
        if value_type == "int":
            return int(value)
        if value_type == "float":
            return float(value)
    except (TypeError, ValueError):
        return value
    return value


def _command_blocks(text: str) -> List[Dict[str, Any]]:
    logical: List[Tuple[Optional[str], str, int, int]] = []
    label: Optional[str] = None
    parts: List[str] = []
    start_line: Optional[int] = None
    last_line: Optional[int] = None
    continuation_pending = False

    def flush() -> None:
        nonlocal parts, start_line, last_line, continuation_pending
        if parts and start_line is not None and last_line is not None:
            logical.append(
                (label, " ".join(parts).strip(), start_line, last_line)
            )
        parts = []
        start_line = None
        last_line = None
        continuation_pending = False

    for line_number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            flush()
            label = None
            continue
        label_match = re.match(
            r"^([^:]{1,120}(?:command|invocation))\s*:\s*(.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if label_match:
            flush()
            label = label_match.group(1).strip()
            stripped = label_match.group(2).strip()
            if not stripped:
                continue
        if parts and not continuation_pending:
            flush()
        if start_line is None:
            start_line = line_number
        last_line = line_number
        continuation_pending = stripped.endswith("\\")
        if continuation_pending:
            parts.append(stripped[:-1].rstrip())
        else:
            parts.append(stripped)
    flush()

    blocks: List[Dict[str, Any]] = []
    for index, (block_label, raw_command, line_start, line_end) in enumerate(
        logical
    ):
        try:
            argv = shlex.split(raw_command)
            tokenization_error = None
        except Exception as exc:
            argv = []
            tokenization_error = type(exc).__name__
        invoked_script = next(
            (
                token
                for token in argv
                if token.endswith(".py") or ".py" in Path(token).name
            ),
            None,
        )
        label_lower = str(block_label or "").lower()
        if invoked_script and (
            "evaluate_phase3c_final_missing_cache_policy_from_dumps.py"
            in invoked_script
        ):
            classification = (
                "phase3c_fitting_diagnostic"
                if any(
                    marker in label_lower
                    for marker in ("print", "capability", "diagnostic")
                )
                else "phase3c_fitting"
            )
        elif invoked_script and (
            "evaluate_phase3c_final_policy_artifact_replay_from_dumps.py"
            in invoked_script
        ):
            classification = "artifact_replay"
        else:
            classification = "other"

        option_specs = {
            "input_manifest": (
                ("--input-manifest", "--dump-manifest"),
                "str",
            ),
            "dump_run_manifest": (("--dump-run-manifest",), "str"),
            "kv_manifest": (("--kv-manifest",), "str"),
            "hidden_manifest": (("--hidden-manifest",), "str"),
            "split_mode": (("--split-mode",), "str"),
            "split_ratio": (("--split-ratio",), "float"),
            "split_seed": (("--split-seed",), "int"),
            "output_artifact_path": (
                (
                    "--output-artifact",
                    "--artifact-output",
                    "--policy-artifact-output",
                ),
                "str",
            ),
            "output_directory": (("--output-dir",), "str"),
            "replay_artifact_path": (
                (
                    "--policy-artifact",
                    "--artifact",
                    "--input-artifact",
                ),
                "str",
            ),
        }
        block: Dict[str, Any] = {
            "command_block_index": index,
            "label": block_label,
            "line_start": line_start,
            "line_end": line_end,
            "raw_command": raw_command,
            "argv": argv,
            "invoked_script": invoked_script,
            "classification": classification,
            "tokenization_error": tokenization_error,
            "field_provenance": {},
        }
        for field, (names, value_type) in option_specs.items():
            value, token_index, option_name = _command_option(argv, names)
            block[field] = _typed_command_value(value, value_type)
            if value is not None:
                block["field_provenance"][field] = {
                    "command_block_index": index,
                    "token_index": token_index,
                    "option_name": option_name,
                    "line_start": line_start,
                    "line_end": line_end,
                }
        blocks.append(block)
    return blocks


def _parse_command(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        blocks = _command_blocks(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _base_result(
            parser="original_fitting_command_v1",
            failures=[
                "fitting_command_parse_failed:{}".format(type(exc).__name__)
            ],
            normalized={},
            provenance={},
        )
    if not blocks:
        failures.append("fitting_command_blocks_missing")
    if any(block.get("tokenization_error") for block in blocks):
        failures.append("fitting_command_block_tokenization_failed")
    fitting_blocks = [
        block
        for block in blocks
        if block.get("classification")
        in ("phase3c_fitting", "phase3c_fitting_diagnostic")
    ]
    if not fitting_blocks:
        failures.append("fitting_command_invocation_missing")
    if fitting_blocks and not any(
        block.get("output_artifact_path") or block.get("output_directory")
        for block in fitting_blocks
    ):
        failures.append("fitting_command_output_relation_missing")
    normalized = {
        "parsed_command_blocks": blocks,
        "parsed_command_block_count": len(blocks),
        "fitting_command_block_count": len(fitting_blocks),
    }
    return _base_result(
        parser="original_fitting_command_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "parsed_command_blocks": "$command.blocks",
            "parsed_command_block_count": "$command.blocks",
            "fitting_command_block_count": "$command.blocks",
        },
    )


def _semantic_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(PurePosixPath(value.replace("\\", "/")))


def _population_historical_directory_binding(
    absolute_historical_path: Any,
) -> Dict[str, Any]:
    normalized_path = _semantic_path(absolute_historical_path)
    failures: List[str] = []
    if (
        normalized_path is None
        or _PATHISH_RE.match(normalized_path) is None
    ):
        failures.append("external_population_historical_path_invalid")
        normalized_directory = None
    else:
        normalized_directory = _semantic_path(
            str(PurePosixPath(normalized_path).parent)
        )
        if normalized_directory in (None, "", "."):
            failures.append("external_population_historical_directory_invalid")
            normalized_directory = None
    population_path_root = (
        _semantic_path(
            str(PurePosixPath(normalized_directory).parent)
        )
        if normalized_directory is not None
        else None
    )
    if population_path_root in (None, "", "."):
        failures.append("external_population_path_root_invalid")
        population_path_root = None
    directory_identity = (
        canonical_json_sha256(
            {
                "normalized_population_evidence_directory": (
                    normalized_directory
                )
            }
        )
        if normalized_directory is not None
        else None
    )
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "population_evidence_historical_path": normalized_path,
        "population_evidence_historical_directory": normalized_directory,
        "population_evidence_directory": normalized_directory,
        "population_path_root": population_path_root,
        "directory_derived_run_identity": directory_identity,
        "directory_derived_run_identity_status": (
            "candidate_pending_central_review"
        ),
    }


def select_fitting_command_for_artifact(
    command_fields: Mapping[str, Any],
    accepted_artifact_path: Any,
) -> Dict[str, Any]:
    target = _semantic_path(accepted_artifact_path)
    blocks = command_fields.get("parsed_command_blocks")
    blocks = blocks if isinstance(blocks, list) else []
    matches: List[Mapping[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        if block.get("classification") != "phase3c_fitting":
            continue
        output_artifact = _semantic_path(block.get("output_artifact_path"))
        output_directory = _semantic_path(block.get("output_directory"))
        if output_artifact == target or (
            output_directory is not None
            and target is not None
            and _semantic_path(str(PurePosixPath(target).parent))
            == output_directory
        ):
            matches.append(block)
    if len(matches) != 1:
        return {
            "status": "failed",
            "failures": [
                "fitting_command_for_accepted_artifact_{}".format(
                    "missing" if not matches else "ambiguous"
                )
            ],
            "matching_fitting_command_count": len(matches),
        }
    selected = dict(matches[0])
    selected_index = selected.get("command_block_index")
    field_provenance = {
        "selected_fitting_invocation": {
            "command_block_index": selected_index,
            "token_index": (
                selected.get("argv", []).index(selected.get("invoked_script"))
                if selected.get("invoked_script") in selected.get("argv", [])
                else None
            ),
            "option_name": None,
            "line_start": selected.get("line_start"),
            "line_end": selected.get("line_end"),
        }
    }
    selected_field_map = {
        "selected_input_manifest": "input_manifest",
        "selected_dump_run_manifest": "dump_run_manifest",
        "selected_kv_manifest": "kv_manifest",
        "selected_hidden_manifest": "hidden_manifest",
        "selected_split_mode": "split_mode",
        "selected_split_ratio": "split_ratio",
        "selected_split_seed": "split_seed",
        "selected_output_artifact_path": "output_artifact_path",
        "selected_output_directory": "output_directory",
    }
    source_provenance = selected.get("field_provenance")
    source_provenance = (
        source_provenance if isinstance(source_provenance, Mapping) else {}
    )
    result: Dict[str, Any] = {
        "status": "ok",
        "failures": [],
        "matching_fitting_command_count": 1,
        "selected_fitting_command_index": selected_index,
        "selected_fitting_command": selected,
        "selected_fitting_command_sha256": canonical_json_sha256(
            {"argv": selected.get("argv") or []}
        ),
        "selected_fitting_invocation": selected.get("invoked_script"),
    }
    for target_field, source_field in selected_field_map.items():
        result[target_field] = selected.get(source_field)
        if source_field in source_provenance:
            field_provenance[target_field] = dict(
                source_provenance[source_field]
            )
    result["selected_fitting_command_field_provenance"] = field_provenance
    return result


def _parse_fit_summary(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="phase3c_fit_summary_v1",
            failures=["fit_summary_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    run_id, run_path = _unique_observed(
        payload,
        (
            ("$.run_identity", ("run_identity",)),
            ("$.provenance.run_identity", ("provenance", "run_identity")),
            (
                "$.policy_artifact_summary.provenance.run_identifier",
                (
                    "policy_artifact_summary",
                    "provenance",
                    "run_identifier",
                ),
            ),
        ),
        label="fit_summary_run_identity",
        failures=failures,
    )
    commit, commit_path = _unique_observed(
        payload,
        (
            ("$.producer_commit", ("producer_commit",)),
            ("$.provenance.git_commit", ("provenance", "git_commit")),
            (
                "$.policy_artifact_summary.provenance.git_commit",
                (
                    "policy_artifact_summary",
                    "provenance",
                    "git_commit",
                ),
            ),
        ),
        label="fit_summary_producer_commit",
        failures=failures,
    )
    artifact_path, artifact_field = _unique_observed(
        payload,
        (
            ("$.output_artifact", ("output_artifact",)),
            ("$.artifact_path", ("artifact_path",)),
            ("$.paths.policy_artifact", ("paths", "policy_artifact")),
        ),
        label="fit_summary_artifact_path",
        failures=failures,
    )
    split = payload.get("split_identity")
    if not isinstance(split, Mapping):
        split = _value_at(
            payload,
            ("policy_artifact_summary", "fit_config", "split_identity"),
        )
    split = split if isinstance(split, Mapping) else {}
    replay_values, replay_provenance = _replay_counts(
        payload,
        failures=failures,
        label="fit_summary",
    )
    if not run_id:
        failures.append("fit_summary_run_identity_missing")
    if not artifact_path:
        failures.append("fit_summary_artifact_path_missing")
    normalized = {
        "status": payload.get("status"),
        "child_run_identifier": run_id,
        "artifact_path": artifact_path,
        "split_identity": dict(split),
        **replay_values,
    }
    return _base_result(
        parser="phase3c_fit_summary_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "child_run_identifier": run_path or "",
            "artifact_path": artifact_field or "",
            "split_identity": "$.split_identity",
            **replay_provenance,
        },
        observed={
            "observed_original_run_identity": run_id,
            "observed_producer_commit": commit,
            "observed_artifact_path": artifact_path,
            "observed_producer_commit_field_path": commit_path,
        },
    )


def _parse_artifact_summary(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="phase3c_artifact_summary_v1",
            failures=["artifact_summary_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    provenance_payload = payload.get("provenance")
    provenance_payload = (
        provenance_payload if isinstance(provenance_payload, Mapping) else {}
    )
    fit_config = payload.get("fit_config")
    fit_config = fit_config if isinstance(fit_config, Mapping) else {}
    artifact_path = payload.get("artifact_path")
    artifact_output_directory = provenance_payload.get("output_dir")
    artifact_sha = payload.get("artifact_file_sha256")
    run_id = payload.get("run_identity") or provenance_payload.get(
        "run_identifier"
    )
    if not artifact_path and not artifact_output_directory:
        failures.append("artifact_summary_artifact_output_relation_missing")
    if artifact_sha is not None and not _is_sha256(artifact_sha):
        failures.append("artifact_summary_artifact_sha256_invalid")
    runtime_semantics = fit_config.get("candidate_first_crossing_semantics")
    runtime_semantics = (
        runtime_semantics if isinstance(runtime_semantics, Mapping) else {}
    )
    hybrid_semantics = fit_config.get("calm_hybrid_policy_semantics")
    hybrid_semantics = (
        hybrid_semantics if isinstance(hybrid_semantics, Mapping) else {}
    )
    runtime_policy_sha = payload.get("runtime_policy_sha256") or (
        runtime_semantics.get("policy_sha256")
        or hybrid_semantics.get("calm_runtime_policy_sha256")
    )
    hybrid_policy_sha = payload.get("hybrid_fitting_policy_sha256") or (
        fit_config.get("calm_hybrid_policy_sha256")
    )
    split_identity = fit_config.get("split_identity")
    split_identity = (
        dict(split_identity)
        if isinstance(split_identity, Mapping)
        else {}
    )
    replay_values, replay_provenance = _replay_counts(
        payload,
        failures=failures,
        label="artifact_summary",
    )
    normalized = {
        "status": payload.get("status") or payload.get("validation_status"),
        "artifact_type": payload.get("artifact_type"),
        "artifact_schema_version": payload.get("schema_version"),
        "child_run_identifier": run_id,
        "artifact_path": artifact_path,
        "artifact_output_directory": artifact_output_directory,
        "artifact_sha256": artifact_sha,
        "runtime_policy_sha256": runtime_policy_sha,
        "hybrid_fitting_policy_sha256": hybrid_policy_sha,
        "split_identity": split_identity,
        **replay_values,
    }
    return _base_result(
        parser="phase3c_artifact_summary_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "child_run_identifier": "$.run_identity",
            "artifact_path": "$.artifact_path",
            "artifact_output_directory": "$.provenance.output_dir",
            "artifact_sha256": "$.artifact_file_sha256",
            "runtime_policy_sha256": "$.runtime_policy_sha256",
            "hybrid_fitting_policy_sha256": "$.hybrid_fitting_policy_sha256",
            "split_identity": "$.fit_config.split_identity",
            **replay_provenance,
        },
        observed={
            "observed_original_run_identity": run_id,
            "observed_producer_commit": payload.get("producer_commit")
            or provenance_payload.get("git_commit"),
            "observed_artifact_path": artifact_path,
            "observed_artifact_sha256": artifact_sha,
        },
    )


def _parse_artifact_identity(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    if not (
        path.name == "artifact_identity.txt"
        or path.name.endswith("_artifact_identity.txt")
    ):
        failures.append("artifact_identity_sidecar_basename_mismatch")
        return _base_result(
            parser="artifact_identity_sidecar_v1",
            failures=failures,
            normalized={},
            provenance={},
        )
    try:
        values = _parse_kv_text(path)
    except Exception as exc:
        return _base_result(
            parser="artifact_identity_sidecar_v1",
            failures=["artifact_identity_sidecar_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    path_values = {
        values[key]
        for key in (
            "artifact_path",
            "output_artifact",
            "artifact_file",
        )
        if values.get(key)
    }
    sha_values = {
        values[key].lower()
        for key in (
            "artifact_sha256",
            "artifact_file_sha256",
            "sha256",
        )
        if values.get(key)
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^\s*([0-9a-fA-F]{64})\s+\*?(.*?)\s*$",
            line,
        )
        if match and match.group(2):
            sha_values.add(match.group(1).lower())
            path_values.add(match.group(2))
    if len(path_values) != 1:
        failures.append(
            "artifact_identity_artifact_path_missing"
            if not path_values
            else "artifact_identity_artifact_path_ambiguous"
        )
    if len(sha_values) != 1:
        failures.append(
            "artifact_identity_artifact_sha256_missing"
            if not sha_values
            else "artifact_identity_artifact_sha256_ambiguous"
        )
    artifact_path = next(iter(path_values)) if len(path_values) == 1 else None
    artifact_sha = next(iter(sha_values)) if len(sha_values) == 1 else None
    if artifact_sha is not None and not _is_sha256(artifact_sha):
        failures.append("artifact_identity_artifact_sha256_invalid")
    artifact_size: Optional[int] = None
    if values.get("artifact_size") is not None:
        try:
            artifact_size = int(values["artifact_size"])
            if artifact_size < 0:
                raise ValueError
        except Exception:
            failures.append("artifact_identity_artifact_size_invalid")
    normalized = {
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha,
        "artifact_size": artifact_size,
        "child_run_identifier": values.get("run_identity"),
    }
    return _base_result(
        parser="artifact_identity_sidecar_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "artifact_path": "$text.artifact_path",
            "artifact_sha256": "$text.artifact_sha256",
            "artifact_size": "$text.artifact_size",
            "child_run_identifier": "$text.run_identity",
        },
        observed={
            "observed_original_run_identity": values.get("run_identity"),
            "observed_artifact_path": artifact_path,
            "observed_artifact_sha256": artifact_sha,
        },
    )


def _parse_child_status(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="child_final_status_v1",
            failures=["child_final_status_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    raw_status = payload.get("status")
    failure_stage = payload.get("failure_stage")
    exit_code = payload.get("exit_code")
    normalized_status = (
        "complete"
        if raw_status == "ok" and failure_stage == "complete" and exit_code == 0
        else None
    )
    if raw_status != "ok":
        failures.append("child_status_not_ok")
    if failure_stage != "complete":
        failures.append("child_failure_stage_not_complete")
    if exit_code != 0 or isinstance(exit_code, bool):
        failures.append("child_exit_code_not_zero")
    replay_values, replay_provenance = _replay_counts(
        payload,
        failures=failures,
        label="child_final_status",
    )
    normalized = {
        "raw_status": raw_status,
        "raw_failure_stage": failure_stage,
        "original_child_execution_status": normalized_status,
        "original_child_exit_code": exit_code,
        "child_run_identifier": payload.get("run_identity"),
        **replay_values,
    }
    return _base_result(
        parser="child_final_status_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "raw_status": "$.status",
            "raw_failure_stage": "$.failure_stage",
            "original_child_execution_status": "$.failure_stage",
            "original_child_exit_code": "$.exit_code",
            "child_run_identifier": "$.run_identity",
            **replay_provenance,
        },
        observed={
            "observed_original_run_identity": payload.get("run_identity"),
            "observed_producer_commit": payload.get("producer_commit"),
        },
    )


def _parse_outer_acceptance(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    if not (
        path.name == "full_acceptance.json"
        or path.name.endswith("_full_acceptance.json")
    ):
        return _base_result(
            parser="outer_full_acceptance_v1",
            failures=["outer_full_acceptance_basename_mismatch"],
            normalized={},
            provenance={},
        )
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="outer_full_acceptance_v1",
            failures=["outer_full_acceptance_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    outer_comparison_schema_present = (
        "comparison_row_count" in payload
        or "comparison_status_counts" in payload
    )
    replay_values, replay_provenance = _outer_acceptance_replay_counts(
        payload,
        failures=failures,
    )
    json_exit_code_present = "exit_code" in payload
    json_exit_code = payload.get("exit_code")
    if json_exit_code_present and (
        not isinstance(json_exit_code, int)
        or isinstance(json_exit_code, bool)
    ):
        failures.append("outer_acceptance_json_exit_code_invalid")
    final_status = payload.get("final_status")
    if not isinstance(final_status, Mapping):
        failures.append("outer_acceptance_nested_final_status_missing")
        final_status = {}
    nested_status = final_status.get("status")
    nested_failure_stage = final_status.get("failure_stage")
    nested_exit_code = final_status.get("exit_code")
    if final_status:
        if "status" in final_status and nested_status in (None, ""):
            failures.append("outer_acceptance_nested_child_status_invalid")
        if (
            "failure_stage" in final_status
            and nested_failure_stage in (None, "")
        ):
            failures.append("outer_acceptance_nested_child_failure_stage_invalid")
        if "exit_code" in final_status and (
            not isinstance(nested_exit_code, int)
            or isinstance(nested_exit_code, bool)
        ):
            failures.append("outer_acceptance_nested_child_exit_code_invalid")

    top_artifact_path = payload.get("artifact_path")
    top_artifact_sha = payload.get("artifact_sha256")
    policy_artifact_path = final_status.get("policy_artifact_path")
    artifact_path_alias = final_status.get("artifact_path")
    if (
        policy_artifact_path not in (None, "")
        and artifact_path_alias not in (None, "")
        and _semantic_path(policy_artifact_path)
        != _semantic_path(artifact_path_alias)
    ):
        failures.append("outer_acceptance_nested_artifact_path_alias_conflict")
    if policy_artifact_path not in (None, ""):
        nested_artifact_path = policy_artifact_path
        nested_artifact_path_field = "$.final_status.policy_artifact_path"
    else:
        nested_artifact_path = artifact_path_alias
        nested_artifact_path_field = (
            "$.final_status.artifact_path"
            if artifact_path_alias not in (None, "")
            else None
        )
    nested_artifact_sha_present = "artifact_sha256" in final_status
    nested_artifact_sha = final_status.get("artifact_sha256")
    if not top_artifact_path or not nested_artifact_path:
        failures.append("outer_acceptance_artifact_path_missing")
    elif _semantic_path(top_artifact_path) != _semantic_path(
        nested_artifact_path
    ):
        failures.append("outer_acceptance_artifact_path_conflict")
    if not _is_sha256(top_artifact_sha):
        failures.append("outer_acceptance_artifact_sha256_invalid")
    if nested_artifact_sha_present and not _is_sha256(nested_artifact_sha):
        failures.append("outer_acceptance_nested_artifact_sha256_invalid")
    elif (
        nested_artifact_sha_present
        and top_artifact_sha != nested_artifact_sha
    ):
        failures.append("outer_acceptance_artifact_sha256_conflict")

    starting_commit = final_status.get("starting_git_commit")
    ending_commit = final_status.get("ending_git_commit")
    commits_present = (
        "starting_git_commit" in final_status
        or "ending_git_commit" in final_status
    )
    if commits_present:
        if not _is_commit(starting_commit) or not _is_commit(ending_commit):
            failures.append("outer_acceptance_git_commit_incomplete_or_invalid")
        elif starting_commit != ending_commit:
            failures.append("outer_acceptance_git_commit_changed")

    policy_artifact_exists = final_status.get("policy_artifact_exists")
    if (
        "policy_artifact_exists" in final_status
        and not isinstance(policy_artifact_exists, bool)
    ):
        failures.append("outer_acceptance_policy_artifact_exists_invalid")
    elif policy_artifact_exists is False:
        failures.append("outer_acceptance_policy_artifact_not_found")
    policy_artifact_summary_path = final_status.get(
        "policy_artifact_summary_path"
    )
    policy_artifact_summary_exists = final_status.get(
        "policy_artifact_summary_exists"
    )
    if (
        "policy_artifact_summary_exists" in final_status
        and not isinstance(policy_artifact_summary_exists, bool)
    ):
        failures.append(
            "outer_acceptance_policy_artifact_summary_exists_invalid"
        )
    replay_summary_exists = final_status.get("replay_summary_exists")
    replay_summary_path = final_status.get("replay_summary_path")
    if outer_comparison_schema_present and not isinstance(
        replay_summary_exists, bool
    ):
        failures.append("outer_acceptance_replay_summary_exists_invalid")
    if outer_comparison_schema_present and (
        not isinstance(replay_summary_path, str) or not replay_summary_path
    ):
        failures.append("outer_acceptance_replay_summary_path_invalid")

    normalized = {
        "original_outer_acceptance_status": payload.get("status"),
        "original_outer_acceptance_exit_code": json_exit_code,
        "outer_acceptance_json_exit_code_present": json_exit_code_present,
        "outer_acceptance_json_exit_code": json_exit_code,
        "original_outer_acceptance_failures": payload.get("failures"),
        "outer_run_identifier": payload.get("run_identity"),
        "nested_child_status": nested_status,
        "nested_child_failure_stage": nested_failure_stage,
        "nested_child_exit_code": nested_exit_code,
        "top_level_artifact_path": top_artifact_path,
        "top_level_artifact_sha256": top_artifact_sha,
        "nested_artifact_path": nested_artifact_path,
        "outer_acceptance_artifact_path_field": "$.artifact_path",
        "nested_child_artifact_path_field": nested_artifact_path_field,
        "nested_artifact_sha_present": nested_artifact_sha_present,
        "nested_artifact_sha256": nested_artifact_sha,
        "nested_policy_artifact_exists": policy_artifact_exists,
        "nested_policy_artifact_summary_path": policy_artifact_summary_path,
        "nested_policy_artifact_summary_exists": (
            policy_artifact_summary_exists
        ),
        "replay_summary_exists": replay_summary_exists,
        "replay_summary_path": replay_summary_path,
        "starting_git_commit": starting_commit,
        "ending_git_commit": ending_commit,
        "original_child_run_identifier": None,
        **replay_values,
    }
    provenance = {
        "original_outer_acceptance_status": "$.status",
        "outer_acceptance_json_exit_code_present": (
            "derived:key-presence:$.exit_code"
        ),
        "original_outer_acceptance_failures": "$.failures",
        "top_level_artifact_path": "$.artifact_path",
        "top_level_artifact_sha256": "$.artifact_sha256",
        "outer_acceptance_artifact_path_field": "$.artifact_path",
        "nested_child_artifact_path_field": nested_artifact_path_field,
        **replay_provenance,
    }
    if json_exit_code_present:
        provenance["original_outer_acceptance_exit_code"] = "$.exit_code"
        provenance["outer_acceptance_json_exit_code"] = "$.exit_code"
    for field, source_path in (
        ("nested_child_status", "$.final_status.status"),
        ("nested_child_failure_stage", "$.final_status.failure_stage"),
        ("nested_child_exit_code", "$.final_status.exit_code"),
        ("starting_git_commit", "$.final_status.starting_git_commit"),
        ("ending_git_commit", "$.final_status.ending_git_commit"),
        (
            "nested_policy_artifact_exists",
            "$.final_status.policy_artifact_exists",
        ),
        (
            "nested_policy_artifact_summary_path",
            "$.final_status.policy_artifact_summary_path",
        ),
        (
            "nested_policy_artifact_summary_exists",
            "$.final_status.policy_artifact_summary_exists",
        ),
        ("replay_summary_exists", "$.final_status.replay_summary_exists"),
        ("replay_summary_path", "$.final_status.replay_summary_path"),
    ):
        key = source_path.rsplit(".", 1)[-1]
        if key in final_status:
            provenance[field] = source_path
    if nested_artifact_path_field is not None:
        provenance["nested_artifact_path"] = nested_artifact_path_field
    if nested_artifact_sha_present:
        provenance["nested_artifact_sha256"] = (
            "$.final_status.artifact_sha256"
        )
    if "run_identity" in payload:
        provenance["outer_run_identifier"] = "$.run_identity"

    return _base_result(
        parser="outer_full_acceptance_v1",
        failures=failures,
        normalized=normalized,
        provenance=provenance,
        observed={
            "observed_outer_run_identity": payload.get("run_identity"),
            "observed_producer_commit": (
                starting_commit if starting_commit == ending_commit else None
            ),
            "observed_artifact_path": top_artifact_path,
            "observed_artifact_sha256": top_artifact_sha,
        },
    )


def _parse_replay_summary(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="artifact_replay_summary_v1",
            failures=[
                "artifact_replay_summary_parse_failed:{}".format(
                    type(exc).__name__
                )
            ],
            normalized={},
            provenance={},
        )
    status = payload.get("status")
    comparison_status = payload.get("comparison_status")
    if status != "ok":
        failures.append("artifact_replay_summary_status_not_ok")
    if comparison_status != "matched":
        failures.append("artifact_replay_summary_comparison_status_not_matched")
    run_id = payload.get("run_identity")
    overall = payload.get("overall")
    first_overall = (
        overall[0]
        if isinstance(overall, list)
        and overall
        and isinstance(overall[0], Mapping)
        else {}
    )
    normalized = {
        "child_run_identifier": run_id,
        "replay_summary_status": status,
        "replay_summary_comparison_status": comparison_status,
        "metric_row_count_diagnostic": first_overall.get("num_records"),
        "metric_ok_row_count_diagnostic": _value_at(
            payload, ("row_status_counts", "ok")
        ),
    }
    return _base_result(
        parser="artifact_replay_summary_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "child_run_identifier": "$.run_identity",
            "replay_summary_status": "$.status",
            "replay_summary_comparison_status": "$.comparison_status",
            "metric_row_count_diagnostic": "$.overall[0].num_records",
            "metric_ok_row_count_diagnostic": "$.row_status_counts.ok",
        },
        observed={
            "observed_original_run_identity": run_id,
            "observed_producer_commit": payload.get("producer_commit"),
        },
    )


def _parse_current_preflight(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        payload = _load_json(path)
    except Exception as exc:
        return _base_result(
            parser="current_artifact_preflight_v1",
            failures=["artifact_preflight_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    artifact_sha, artifact_sha_path = _unique_observed(
        payload,
        (
            ("$.artifact_file_sha256", ("artifact_file_sha256",)),
            ("$.sha_before", ("sha_before",)),
            ("$.expected_sha", ("expected_sha",)),
        ),
        label="artifact_preflight_artifact_sha256",
        failures=failures,
    )
    if not _is_sha256(artifact_sha):
        failures.append("artifact_preflight_artifact_sha256_invalid")
    sha_after = payload.get("sha_after")
    if sha_after not in (None, artifact_sha):
        failures.append("artifact_preflight_after_sha256_mismatch")
    unchanged = payload.get("unchanged")
    status = payload.get("status")
    if status is None and artifact_sha and sha_after == artifact_sha and unchanged is True:
        status = "ok"
    normalized = {
        "status": status,
        "current_audit_run_identifier": payload.get("audit_run_identity"),
        "artifact_path": payload.get("artifact_path"),
        "artifact_sha256": artifact_sha,
        "artifact_type": payload.get("artifact_type"),
        "artifact_schema_version": payload.get("schema_version"),
    }
    return _base_result(
        parser="current_artifact_preflight_v1",
        failures=failures,
        normalized=normalized,
        provenance={
            "current_audit_run_identifier": "$.audit_run_identity",
            "artifact_path": "$.artifact_path",
            "artifact_sha256": artifact_sha_path or "",
        },
        observed={
            "observed_audit_run_identity": payload.get("audit_run_identity"),
            "observed_artifact_path": payload.get("artifact_path"),
            "observed_artifact_sha256": artifact_sha,
        },
    )


def _parse_git_identity(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    if not (
        path.name == "git_identity.txt"
        or path.name.endswith("_git_identity.txt")
    ):
        return _base_result(
            parser="producer_git_identity_v1",
            failures=["producer_git_identity_basename_mismatch"],
            normalized={},
            provenance={},
        )
    try:
        values = _parse_kv_text(path)
    except Exception as exc:
        return _base_result(
            parser="producer_git_identity_v1",
            failures=["producer_git_identity_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    branch, branch_key = _unique_kv_alias(
        values,
        ("branch", "git_branch"),
        label="producer_git_identity_branch",
        failures=failures,
    )
    commit, commit_key = _unique_kv_alias(
        values,
        ("head", "commit", "git_commit"),
        label="producer_git_identity_commit",
        failures=failures,
    )
    if not isinstance(branch, str) or not branch:
        failures.append("producer_git_identity_branch_missing")
    if not _is_commit(commit):
        failures.append("producer_git_identity_commit_invalid")
    tracked_status_present = "tracked_status" in values
    tracked_status = values.get("tracked_status")
    declared_clean = values.get("tracked_worktree_clean")
    if declared_clean is None:
        declared_clean_bool = None
    elif declared_clean.lower() == "true":
        declared_clean_bool = True
    elif declared_clean.lower() == "false":
        declared_clean_bool = False
    else:
        declared_clean_bool = None
        failures.append("producer_git_identity_tracked_clean_invalid")
    tracked_status_clean = (
        tracked_status == "" if tracked_status_present else None
    )
    if (
        tracked_status_clean is not None
        and declared_clean_bool is not None
        and tracked_status_clean != declared_clean_bool
    ):
        failures.append("producer_git_identity_tracked_status_conflict")
    tracked_worktree_clean = (
        tracked_status_clean
        if tracked_status_clean is not None
        else declared_clean_bool
    )
    normalized = {
        "producer_branch": branch,
        "producer_commit": commit,
        "producer_tracked_status": tracked_status,
        "tracked_worktree_clean": tracked_worktree_clean,
        "repository_path": values.get("repository_path"),
        "outer_run_identifier": values.get("run_identity"),
    }
    provenance = {
        "producer_branch": "$text.{}".format(branch_key or "branch"),
        "producer_commit": "$text.{}".format(commit_key or "head"),
        "repository_path": "$text.repository_path",
        "outer_run_identifier": "$text.run_identity",
    }
    if tracked_status_present:
        provenance["producer_tracked_status"] = "$text.tracked_status"
        provenance["tracked_worktree_clean"] = "$text.tracked_status"
    elif declared_clean is not None:
        provenance["tracked_worktree_clean"] = "$text.tracked_worktree_clean"
    return _base_result(
        parser="producer_git_identity_v1",
        failures=failures,
        normalized=normalized,
        provenance=provenance,
        observed={
            "observed_outer_run_identity": values.get("run_identity"),
            "observed_producer_commit": commit,
        },
    )


def _unique_kv_alias(
    values: Mapping[str, str],
    aliases: Sequence[str],
    *,
    label: str,
    failures: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    observed = [
        (values[key], key)
        for key in aliases
        if key in values and values[key] != ""
    ]
    if len({value for value, _ in observed}) > 1:
        failures.append("{}_conflict".format(label))
        return None, None
    return observed[0] if observed else (None, None)


def _parse_exact_exit_code(
    path: Path,
    *,
    parser: str,
    expected_basename: str,
    expected_value: int,
    failure_prefix: str,
) -> Dict[str, Any]:
    failures: List[str] = []
    if not (
        path.name == expected_basename
        or path.name.endswith("_" + expected_basename)
    ):
        failures.append("{}_basename_mismatch".format(failure_prefix))
    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return _base_result(
            parser=parser,
            failures=[
                "{}_parse_failed:{}".format(
                    failure_prefix, type(exc).__name__
                )
            ],
            normalized={},
            provenance={},
        )
    if not re.fullmatch(r"-?\d+", raw_text):
        exit_code = None
        failures.append("{}_integer_missing".format(failure_prefix))
    else:
        exit_code = int(raw_text)
        if exit_code != expected_value:
            failures.append(
                "{}_value_not_{}".format(
                    failure_prefix,
                    "zero" if expected_value == 0 else "one",
                )
            )
    return _base_result(
        parser=parser,
        failures=failures,
        normalized={"exit_code": exit_code},
        provenance={"exit_code": "$text.integer"},
    )


def _parse_child_full_exit_code(path: Path) -> Dict[str, Any]:
    return _parse_exact_exit_code(
        path,
        parser="child_full_exit_code_v1",
        expected_basename="full_exit_code.txt",
        expected_value=0,
        failure_prefix="child_full_exit_code",
    )


def _parse_outer_acceptance_exit_code(path: Path) -> Dict[str, Any]:
    return _parse_exact_exit_code(
        path,
        parser="outer_acceptance_exit_code_v1",
        expected_basename="full_acceptance_exit_code.txt",
        expected_value=1,
        failure_prefix="outer_acceptance_exit_code",
    )


def _parse_file_manifest(path: Path) -> Dict[str, Any]:
    failures: List[str] = []
    rows: List[Dict[str, Any]] = []
    if not (
        path.name == "full_run_file_manifest.txt"
        or path.name.endswith("_full_run_file_manifest.txt")
    ):
        return _base_result(
            parser="original_file_manifest_v1",
            failures=["full_run_file_manifest_basename_mismatch"],
            normalized={},
            provenance={},
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("empty")
        seen_paths = set()
        for line in lines:
            if not line.strip():
                raise ValueError("blank_row")
            values = line.split("\t")
            if len(values) not in (2, 3):
                raise ValueError("columns")
            relative_path = values[0]
            if (
                not relative_path
                or _PATHISH_RE.match(relative_path)
                or "\\" in relative_path
            ):
                raise ValueError("absolute_or_non_posix_path")
            normalized_path = PurePosixPath(relative_path)
            if (
                normalized_path.is_absolute()
                or ".." in normalized_path.parts
                or "." in normalized_path.parts
            ):
                raise ValueError("path_traversal")
            normalized_relative = str(normalized_path)
            if normalized_relative in seen_paths:
                raise ValueError("duplicate_relative_path")
            seen_paths.add(normalized_relative)
            size_match = re.fullmatch(r"(0|[1-9][0-9]*) bytes", values[1])
            if size_match is None:
                raise ValueError("size_serialization")
            row = {
                "relative_path": normalized_relative,
                "size_bytes": int(size_match.group(1)),
            }
            if len(values) == 3:
                row["sha256"] = values[2]
            if row.get("sha256") is not None and not _is_sha256(row["sha256"]):
                raise ValueError("sha256")
            rows.append(row)
    except Exception as exc:
        return _base_result(
            parser="original_file_manifest_v1",
            failures=["file_manifest_parse_failed:{}".format(type(exc).__name__)],
            normalized={},
            provenance={},
        )
    artifact_rows = [
        row
        for row in rows
        if str(row.get("relative_path", "")).endswith(
            "calm_hybrid_multisource_policy_artifact.pt"
        )
    ]
    if len(artifact_rows) != 1:
        failures.append("file_manifest_artifact_entry_not_unique")
    artifact_row = artifact_rows[0] if len(artifact_rows) == 1 else {}
    normalized = {
        "manifest_serialization": "relative_path_tab_integer_bytes_v1",
        "manifest_entry_count": len(rows),
        "artifact_relative_path": artifact_row.get("relative_path"),
        "artifact_size_bytes": artifact_row.get("size_bytes"),
        "artifact_sha256": artifact_row.get("sha256"),
        "artifact_sha_present": artifact_row.get("sha256") is not None,
        "entries": rows,
    }
    provenance = {
        "manifest_serialization": "$text.rows",
        "manifest_entry_count": "$text.rows",
        "artifact_relative_path": "$text.rows[artifact].relative_path",
        "artifact_size_bytes": "$text.rows[artifact].size_bytes",
        "artifact_sha_present": "$text.rows[artifact]",
    }
    if artifact_row.get("sha256") is not None:
        provenance["artifact_sha256"] = "$text.rows[artifact].sha256"
    return _base_result(
        parser="original_file_manifest_v1",
        failures=failures,
        normalized=normalized,
        provenance=provenance,
        observed={
            "observed_artifact_path": artifact_row.get("relative_path"),
            "observed_artifact_sha256": artifact_row.get("sha256"),
        },
    )


_PARSERS = {
    "external_population_comparison": _parse_external_population,
    "original_fitting_command": _parse_command,
    "fit_summary": _parse_fit_summary,
    "artifact_summary": _parse_artifact_summary,
    "artifact_identity": _parse_artifact_identity,
    "child_final_status": _parse_child_status,
    "outer_full_acceptance": _parse_outer_acceptance,
    "current_artifact_preflight": _parse_current_preflight,
    "producer_git_identity": _parse_git_identity,
    "child_full_exit_code": _parse_child_full_exit_code,
    "outer_acceptance_exit_code": _parse_outer_acceptance_exit_code,
    "file_manifest": _parse_file_manifest,
    "replay_summary": _parse_replay_summary,
}


def parse_evidence_role_file(role: str, path: Path | str) -> Dict[str, Any]:
    parser = _PARSERS.get(role)
    if parser is None:
        return {
            "status": "failed",
            "failures": ["unsupported_evidence_role:{}".format(role)],
            "normalized_fields": {},
            "normalized_field_provenance": {},
        }
    return parser(Path(path))


def _merged_specs(
    role_specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    base_specs = {
        **ACCEPTED_AUDIT_ROLE_SPECS,
        **SUPPORTING_AUDIT_ROLE_SPECS,
    }
    for role in base_specs:
        base = dict(base_specs[role])
        override = role_specs.get(role)
        if isinstance(override, Mapping):
            base.update(dict(override))
        merged[role] = base
    return merged


class _CanonicalResolvedRoleSpecsError(ValueError):
    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(sorted(set(failures)))
        super().__init__(
            "canonical_resolved_role_specs_invalid:{}".format(
                ",".join(self.failures)
            )
        )


def _canonical_role_relative_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith(("/", "//")):
        return None
    if len(normalized) >= 2 and normalized[1] == ":":
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        return None
    return "/".join(path.parts)


def _canonical_absolute_historical_path(value: Any) -> Optional[str]:
    normalized = _semantic_path(value)
    if (
        normalized is None
        or _PATHISH_RE.match(normalized) is None
        or any(
            part in (".", "..")
            for part in PurePosixPath(normalized).parts
        )
    ):
        return None
    return normalized


def _bound_role_path_context(
    by_role: Mapping[str, Mapping[str, Any]],
    *,
    audit_members: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    failures: List[str] = []

    def historical_parent(role: str) -> Optional[str]:
        record = by_role.get(role)
        historical = _canonical_absolute_historical_path(
            record.get("absolute_historical_path")
            if isinstance(record, Mapping)
            else None
        )
        if historical is None:
            failures.append(
                "canonical_role_{}_absolute_historical_path_invalid".format(
                    role
                )
            )
            return None
        parent = _semantic_path(str(PurePosixPath(historical).parent))
        if parent in (None, "", "."):
            failures.append(
                "canonical_role_{}_historical_parent_invalid".format(role)
            )
            return None
        return parent

    child_run_directory = historical_parent("child_final_status")
    outer_run_directory = historical_parent("outer_full_acceptance")
    population_evidence_directory = historical_parent(
        "external_population_comparison"
    )
    population_path_root = (
        _semantic_path(
            str(PurePosixPath(population_evidence_directory).parent)
        )
        if population_evidence_directory
        else None
    )
    if population_path_root in (None, "", "."):
        failures.append("canonical_population_path_root_invalid")
        population_path_root = None
    if child_run_directory and outer_run_directory:
        try:
            relative_child = PurePosixPath(
                child_run_directory
            ).relative_to(PurePosixPath(outer_run_directory))
            if not relative_child.parts:
                raise ValueError("child_equals_outer")
        except ValueError:
            failures.append(
                "canonical_child_run_directory_not_nested_under_outer"
            )

    audit_roots = set()
    for member in audit_members or []:
        if not isinstance(member, Mapping):
            failures.append("canonical_audit_member_not_mapping")
            continue
        relative = _canonical_role_relative_path(
            member.get("relative_path")
        )
        if relative is None:
            failures.append("canonical_audit_member_relative_path_invalid")
            continue
        parts = PurePosixPath(relative).parts
        if len(parts) < 2:
            failures.append("canonical_audit_member_archive_root_missing")
            continue
        audit_roots.add(parts[0])
    if len(audit_roots) > 1:
        failures.append("canonical_audit_archive_root_ambiguous")

    preflight = by_role.get("current_artifact_preflight")
    preflight_snapshot = _canonical_role_relative_path(
        preflight.get("snapshot_relative_path")
        if isinstance(preflight, Mapping)
        else None
    )
    snapshot_root = None
    if preflight_snapshot is not None:
        snapshot_parts = PurePosixPath(preflight_snapshot).parts
        if len(snapshot_parts) >= 2:
            snapshot_root = snapshot_parts[0]
    audit_archive_root_name = (
        next(iter(audit_roots))
        if len(audit_roots) == 1
        else snapshot_root
    )
    if (
        audit_archive_root_name is not None
        and snapshot_root is not None
        and audit_archive_root_name != snapshot_root
    ):
        failures.append("canonical_audit_archive_root_mismatch")

    return {
        "status": "ok" if not failures else "failed",
        "failures": sorted(set(failures)),
        "child_run_directory": child_run_directory,
        "outer_run_directory": outer_run_directory,
        "population_evidence_directory": population_evidence_directory,
        "population_path_root": population_path_root,
        "audit_archive_root_name": audit_archive_root_name,
    }


def derive_expected_relative_path_from_bound_record(
    record: Mapping[str, Any],
    *,
    child_run_directory: Optional[str],
    outer_run_directory: Optional[str],
    population_evidence_directory: Optional[str],
    population_path_root: Optional[str],
    audit_archive_root_name: Optional[str] = None,
) -> str:
    """Derive a role path from historical identity, not its copied claim."""

    role = record.get("evidence_role")
    path_authority = record.get("path_authority_domain")
    historical = _canonical_absolute_historical_path(
        record.get("absolute_historical_path")
    )
    if historical is None:
        raise ValueError("absolute_historical_path_invalid")
    basename = record.get("expected_basename")
    if (
        not isinstance(basename, str)
        or not basename
        or PurePosixPath(historical).name != basename
    ):
        raise ValueError("historical_path_basename_mismatch")

    def relative_to(root: Optional[str]) -> str:
        if root is None:
            raise ValueError("verified_resolution_root_missing")
        try:
            relative = PurePosixPath(historical).relative_to(
                PurePosixPath(root)
            )
        except ValueError as exc:
            raise ValueError("historical_path_outside_verified_root") from exc
        normalized = _canonical_role_relative_path(str(relative))
        if normalized is None:
            raise ValueError("derived_relative_path_invalid")
        return normalized

    if path_authority == PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER:
        relative_to(child_run_directory)
        derived = relative_to(outer_run_directory)
    elif path_authority == PATH_AUTHORITY_OUTER_RUN_ROOT:
        derived = relative_to(outer_run_directory)
        if len(PurePosixPath(derived).parts) != 1:
            raise ValueError("outer_role_not_in_outer_root")
    elif path_authority == PATH_AUTHORITY_POPULATION_RUN_PARENT:
        if population_evidence_directory is None:
            raise ValueError("population_evidence_directory_missing")
        if population_path_root is None:
            raise ValueError("population_path_root_missing")
        if (
            _semantic_path(
                str(PurePosixPath(population_evidence_directory).parent)
            )
            != population_path_root
        ):
            raise ValueError("population_path_root_relation_invalid")
        if (
            _semantic_path(str(PurePosixPath(historical).parent))
            != population_evidence_directory
        ):
            raise ValueError("population_file_parent_mismatch")
        derived = relative_to(population_path_root)
        parts = PurePosixPath(derived).parts
        if (
            len(parts) != 2
            or parts[0]
            != PurePosixPath(population_evidence_directory).name
        ):
            raise ValueError("population_run_prefix_invalid")
    elif path_authority == PATH_AUTHORITY_AUDIT_ARCHIVE_MEMBER:
        if role != "current_artifact_preflight":
            raise ValueError("unsupported_audit_role")
        snapshot = _canonical_role_relative_path(
            record.get("snapshot_relative_path")
        )
        if record.get("snapshot_present") is not True or snapshot is None:
            raise ValueError("audit_role_snapshot_required")
        parts = PurePosixPath(snapshot).parts
        if audit_archive_root_name is None:
            if len(parts) != 1:
                raise ValueError("audit_archive_root_missing")
            derived = parts[0]
        else:
            if (
                len(parts) != 2
                or parts[0] != audit_archive_root_name
            ):
                raise ValueError("audit_role_archive_member_relation_invalid")
            derived = parts[1]
    else:
        raise ValueError("unsupported_path_authority_domain")

    if PurePosixPath(derived).name != basename:
        raise ValueError("derived_relative_path_basename_mismatch")
    return derived


def _validate_snapshot_transport_relation(
    record: Mapping[str, Any],
    *,
    independently_derived_relative_path: str,
    audit_archive_root_name: Optional[str],
) -> None:
    snapshot_present = record.get("snapshot_present")
    snapshot = _canonical_role_relative_path(
        record.get("snapshot_relative_path")
    )
    if snapshot_present is not True:
        if snapshot is not None:
            raise ValueError("snapshotless_role_has_snapshot_path")
        return
    if snapshot is None:
        raise ValueError("snapshot_relative_path_invalid")

    if audit_archive_root_name is None:
        if snapshot != independently_derived_relative_path:
            raise ValueError("direct_snapshot_relative_path_mismatch")
        return

    parts = PurePosixPath(snapshot).parts
    if not parts or parts[0] != audit_archive_root_name:
        raise ValueError("snapshot_archive_root_mismatch")
    member_parts = parts[1:]
    role = record.get("evidence_role")
    basename = record.get("expected_basename")
    if role == "current_artifact_preflight":
        if member_parts != (basename,):
            raise ValueError("audit_preflight_snapshot_member_mismatch")
        return
    historical = _canonical_absolute_historical_path(
        record.get("absolute_historical_path")
    )
    if historical is None:
        raise ValueError("snapshot_historical_path_invalid")
    expected_snapshot_name = "{}_{}".format(
        hashlib.sha256(historical.encode("utf-8")).hexdigest()[:12],
        basename,
    )
    if member_parts != ("evidence_snapshots", expected_snapshot_name):
        raise ValueError("snapshot_member_naming_mismatch")


def canonical_resolved_role_specs_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    role_contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    audit_members: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Reconstruct role specs from independently bound path evidence."""

    contracts = _merged_specs(role_contracts or {})
    failures: List[str] = []
    by_role: Dict[str, Mapping[str, Any]] = {}
    required_fields = (
        "expected_evidence_file_sha256",
        "expected_basename",
        "expected_relative_path",
        "identity_domain",
        "path_authority_domain",
        "path_relation",
        "parser",
        "relation_supported",
    )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append(
                "canonical_role_record_not_mapping:{}".format(index)
            )
            continue
        role = record.get("evidence_role")
        if not isinstance(role, str) or role not in contracts:
            failures.append(
                "canonical_role_unsupported:{}".format(role)
            )
            continue
        if role in by_role:
            failures.append("canonical_role_duplicate:{}".format(role))
            continue
        by_role[role] = record
        for field in required_fields:
            if field not in record:
                failures.append(
                    "canonical_role_{}_{}_missing".format(role, field)
                )

        expected_sha = record.get("expected_evidence_file_sha256")
        if not _is_sha256(expected_sha):
            failures.append(
                "canonical_role_{}_expected_sha256_invalid".format(role)
            )
        if expected_sha != record.get("evidence_file_sha256"):
            failures.append(
                "canonical_role_{}_expected_sha256_file_mismatch".format(
                    role
                )
            )

        basename = record.get("expected_basename")
        if (
            not isinstance(basename, str)
            or not basename
            or "/" in basename
            or "\\" in basename
            or PurePosixPath(basename).name != basename
        ):
            failures.append(
                "canonical_role_{}_expected_basename_invalid".format(role)
            )

        relative = _canonical_role_relative_path(
            record.get("expected_relative_path")
        )
        if relative is None:
            failures.append(
                "canonical_role_{}_expected_relative_path_invalid".format(
                    role
                )
            )
        elif isinstance(basename, str) and PurePosixPath(relative).name != (
            basename
        ):
            failures.append(
                "canonical_role_{}_relative_path_basename_mismatch".format(
                    role
                )
            )

        historical = _semantic_path(
            record.get("absolute_historical_path")
        )
        if (
            historical is not None
            and isinstance(basename, str)
            and PurePosixPath(historical).name != basename
        ):
            failures.append(
                "canonical_role_{}_historical_path_basename_mismatch".format(
                    role
                )
            )

        contract = contracts[role]
        for field in (
            "identity_domain",
            "path_authority_domain",
            "path_relation",
            "parser",
            "relation_supported",
        ):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                failures.append(
                    "canonical_role_{}_{}_invalid".format(role, field)
                )
            elif value != contract.get(field):
                failures.append(
                    "canonical_role_{}_{}_contract_mismatch".format(
                        role, field
                    )
                )

    for role in REQUIRED_EVIDENCE_ROLES:
        if role not in by_role:
            failures.append("canonical_required_role_missing:{}".format(role))
    replay_required = (
        (contracts.get("external_population_comparison") or {}).get(
            "require_exact_population_lists"
        )
        is True
        or (contracts.get("replay_summary") or {}).get("required_for_paper")
        is True
    )
    if replay_required and "replay_summary" not in by_role:
        failures.append("canonical_required_role_missing:replay_summary")

    path_context = _bound_role_path_context(
        by_role,
        audit_members=audit_members,
    )
    failures.extend(path_context.get("failures") or [])
    derived_relative_paths: Dict[str, str] = {}
    for role, record in by_role.items():
        try:
            derived = derive_expected_relative_path_from_bound_record(
                record,
                child_run_directory=path_context.get(
                    "child_run_directory"
                ),
                outer_run_directory=path_context.get(
                    "outer_run_directory"
                ),
                population_evidence_directory=path_context.get(
                    "population_evidence_directory"
                ),
                population_path_root=path_context.get(
                    "population_path_root"
                ),
                audit_archive_root_name=path_context.get(
                    "audit_archive_root_name"
                ),
            )
            claimed = _canonical_role_relative_path(
                record.get("expected_relative_path")
            )
            if claimed != derived:
                failures.append(
                    "canonical_role_{}_expected_relative_path_"
                    "authority_mismatch".format(role)
                )
            _validate_snapshot_transport_relation(
                record,
                independently_derived_relative_path=derived,
                audit_archive_root_name=path_context.get(
                    "audit_archive_root_name"
                ),
            )
            derived_relative_paths[role] = derived
        except ValueError as exc:
            failures.append(
                "canonical_role_{}_path_authority_invalid:{}".format(
                    role, str(exc)
                )
            )
    if failures:
        raise _CanonicalResolvedRoleSpecsError(failures)

    ordered_roles = sorted(
        by_role,
        key=lambda role: (_ROLE_ORDER.get(role, 999), role),
    )
    return {
        role: {
            "expected_sha256": by_role[role].get(
                "expected_evidence_file_sha256"
            ),
            "expected_basename": by_role[role].get("expected_basename"),
            "expected_relative_path": derived_relative_paths[role],
            "identity_domain": by_role[role].get("identity_domain"),
            "path_authority_domain": by_role[role].get(
                "path_authority_domain"
            ),
            "path_relation": by_role[role].get("path_relation"),
            "parser": by_role[role].get("parser"),
            "relation_supported": by_role[role].get(
                "relation_supported"
            ),
        }
        for role in ordered_roles
    }


def _record_from_file(
    role: str,
    path: Path,
    *,
    spec: Mapping[str, Any],
    expected_relative_path: str,
    historical_path: str,
    role_binding_method: str = EXACT_ROLE_BINDING_METHOD,
    original_inventory_sha256: Optional[str] = None,
    snapshot_relative_path: Optional[str] = None,
    snapshot_sha256: Optional[str] = None,
    archive_member_sha256: Optional[str] = None,
    snapshot_present: bool = True,
    binding_source: Optional[str] = None,
    live_original_verified: bool = False,
    live_original_sha256: Optional[str] = None,
    live_original_size_bytes: Optional[int] = None,
    original_inventory_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    digest = sha256_file(path)
    parsed = parse_evidence_role_file(role, path)
    normalized = parsed.get("normalized_fields")
    normalized = normalized if isinstance(normalized, Mapping) else {}
    record = {
        "role_binding_schema_version": ROLE_BINDING_SCHEMA_VERSION,
        "role_binding_method": role_binding_method,
        "evidence_role": role,
        "identity_domain": spec.get("identity_domain"),
        "path_authority_domain": spec.get("path_authority_domain"),
        "path_relation": spec.get("path_relation"),
        "parser": spec.get("parser"),
        "relation_supported": spec.get("relation_supported"),
        "expected_basename": spec.get("expected_basename"),
        "expected_relative_path": expected_relative_path,
        "expected_evidence_file_sha256": spec.get("expected_sha256"),
        "absolute_historical_path": historical_path,
        "snapshot_present": bool(snapshot_present),
        "snapshot_relative_path": (
            (snapshot_relative_path or expected_relative_path)
            if snapshot_present
            else None
        ),
        "binding_source": (
            binding_source
            or (
                "accepted_audit_inventory_plus_snapshot"
                if snapshot_present
                else "accepted_audit_inventory_plus_live_original"
            )
        ),
        "live_original_verified": bool(live_original_verified),
        "live_original_sha256": live_original_sha256,
        "live_original_size_bytes": live_original_size_bytes,
        "file_size": path.stat().st_size,
        "evidence_file_sha256": digest,
        "file_sha256": digest,
        "original_inventory_sha256": original_inventory_sha256 or digest,
        "original_inventory_size_bytes": (
            original_inventory_size_bytes
            if original_inventory_size_bytes is not None
            else path.stat().st_size
        ),
        "snapshot_sha256": (
            snapshot_sha256 or digest if snapshot_present else None
        ),
        "archive_member_sha256": (
            archive_member_sha256 or digest if snapshot_present else None
        ),
        "parse_status": parsed.get("status"),
        "parse_failures": list(parsed.get("failures") or []),
        "schema_or_evidence_type": spec.get("parser"),
        "normalized_fields": dict(normalized),
        "normalized_field_provenance": dict(
            parsed.get("normalized_field_provenance") or {}
        ),
        "observed_original_run_identity": parsed.get(
            "observed_original_run_identity"
        ),
        "observed_outer_run_identity": parsed.get("observed_outer_run_identity"),
        "observed_population_source_run_identity": parsed.get(
            "observed_population_source_run_identity"
        ),
        "observed_audit_run_identity": parsed.get("observed_audit_run_identity"),
        "observed_producer_commit": parsed.get("observed_producer_commit"),
        "observed_artifact_path": parsed.get("observed_artifact_path"),
        "observed_artifact_sha256": parsed.get("observed_artifact_sha256"),
        "details": dict(normalized),
    }
    if role == "external_population_comparison":
        historical_binding = _population_historical_directory_binding(
            historical_path
        )
        binding_fields = {
            key: value
            for key, value in historical_binding.items()
            if key not in ("status", "failures")
        }
        record["normalized_fields"].update(binding_fields)
        record["details"].update(binding_fields)
        record["normalized_field_provenance"].update(
            {
                "population_evidence_historical_path": (
                    "role_record.absolute_historical_path"
                ),
                "population_evidence_historical_directory": (
                    "derived:absolute_historical_path.parent"
                ),
                "population_evidence_directory": (
                    "derived:absolute_historical_path.parent"
                ),
                "population_path_root": (
                    "derived:absolute_historical_path.parent.parent"
                ),
                "directory_derived_run_identity": (
                    "derived:absolute_historical_path.parent"
                ),
                "directory_derived_run_identity_status": (
                    "policy:directory-derived-candidate-pending-central-review"
                ),
            }
        )
        historical_failures = list(historical_binding.get("failures") or [])
        if historical_failures:
            record["parse_status"] = "failed"
            record["parse_failures"].extend(historical_failures)
    record["original_run_identity"] = (
        record["observed_original_run_identity"]
        or record["observed_outer_run_identity"]
        or record["observed_population_source_run_identity"]
        or record["observed_audit_run_identity"]
    )
    record["producer_commit"] = record["observed_producer_commit"]
    record["artifact_path"] = record["observed_artifact_path"]
    return record


def _bind_selected_fitting_command(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_role = {
        str(record.get("evidence_role")): record
        for record in records
        if isinstance(record, Mapping)
    }
    command = by_role.get("original_fitting_command")
    artifact = by_role.get("artifact_identity")
    if command is None or artifact is None:
        return {
            "status": "failed",
            "failures": ["fitting_command_or_artifact_identity_missing"],
        }
    command_fields = command.get("normalized_fields")
    command_fields = (
        dict(command_fields)
        if isinstance(command_fields, Mapping)
        else {}
    )
    artifact_fields = artifact.get("normalized_fields")
    artifact_fields = (
        artifact_fields if isinstance(artifact_fields, Mapping) else {}
    )
    selected = select_fitting_command_for_artifact(
        command_fields,
        artifact_fields.get("artifact_path"),
    )
    command_fields.update(
        {
            key: value
            for key, value in selected.items()
            if key not in ("status", "failures")
        }
    )
    command["normalized_fields"] = command_fields
    command["details"] = dict(command_fields)
    provenance = command.get("normalized_field_provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    provenance.update(
        dict(
            selected.get("selected_fitting_command_field_provenance") or {}
        )
    )
    command["normalized_field_provenance"] = provenance
    command["observed_artifact_path"] = selected.get(
        "selected_output_artifact_path"
    )
    command["artifact_path"] = command["observed_artifact_path"]
    if selected.get("status") != "ok":
        parse_failures = list(command.get("parse_failures") or [])
        parse_failures.extend(selected.get("failures") or [])
        command["parse_failures"] = sorted(set(parse_failures))
        command["parse_status"] = "failed"
    return selected


def build_exact_role_records_from_directory(
    root: Path | str,
    *,
    role_specs: Mapping[str, Mapping[str, Any]],
    expected_artifact_sha256: str,
    historical_root: str,
    historical_paths: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    root = Path(root)
    specs = _merged_specs(role_specs)
    failures: List[str] = []
    records: List[Dict[str, Any]] = []
    for role in REQUIRED_EVIDENCE_ROLES:
        spec = specs[role]
        relative = spec.get("expected_relative_path")
        if not isinstance(relative, str) or not relative:
            basename = spec.get("expected_basename")
            relative = str(basename or "")
        target = root / Path(relative)
        if not target.is_file():
            failures.append("required_evidence_role_missing:{}".format(role))
            continue
        record = _record_from_file(
            role,
            target,
            spec=spec,
            expected_relative_path=relative.replace("\\", "/"),
            historical_path=(
                str((historical_paths or {}).get(role))
                if (historical_paths or {}).get(role)
                else "{}/{}".format(
                    historical_root.rstrip("/"),
                    relative.replace("\\", "/"),
                )
            ),
        )
        if record["evidence_file_sha256"] != spec.get("expected_sha256"):
            failures.append("{}_expected_sha256_mismatch".format(role))
        records.append(record)
    by_role = {
        str(record.get("evidence_role")): record for record in records
    }
    path_context = _bound_role_path_context(by_role)
    failures.extend(path_context.get("failures") or [])
    if path_context.get("status") == "ok":
        for record in records:
            try:
                derived_relative = (
                    derive_expected_relative_path_from_bound_record(
                        record,
                        child_run_directory=path_context.get(
                            "child_run_directory"
                        ),
                        outer_run_directory=path_context.get(
                            "outer_run_directory"
                        ),
                        population_evidence_directory=path_context.get(
                            "population_evidence_directory"
                        ),
                        population_path_root=path_context.get(
                            "population_path_root"
                        ),
                        audit_archive_root_name=path_context.get(
                            "audit_archive_root_name"
                        ),
                    )
                )
            except ValueError as exc:
                failures.append(
                    "canonical_role_{}_path_authority_invalid:{}".format(
                        record.get("evidence_role"), str(exc)
                    )
                )
                continue
            record["expected_relative_path"] = derived_relative
            if record.get("snapshot_present") is True:
                record["snapshot_relative_path"] = derived_relative
    command_selection = _bind_selected_fitting_command(records)
    if command_selection.get("status") != "ok":
        failures.extend(command_selection.get("failures") or [])
    validation = validate_exact_role_evidence_chain(
        records,
        role_specs=specs,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    failures.extend(validation.get("failures") or [])
    return {
        "status": "ok" if not failures else "failed",
        "failures": sorted(set(failures)),
        "resolution_method": EXACT_ROLE_BINDING_METHOD,
        "paper_role_binding_valid": not failures,
        "evidence_files": records,
        "selected_fitting_command": command_selection,
        "validation": validation,
    }


def _role_digest_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_role": record.get("evidence_role"),
        "identity_domain": record.get("identity_domain"),
        "path_authority_domain": record.get("path_authority_domain"),
        "path_relation": record.get("path_relation"),
        "expected_evidence_file_sha256": record.get(
            "expected_evidence_file_sha256"
        ),
        "expected_basename": record.get("expected_basename"),
        "expected_relative_path": record.get("expected_relative_path"),
        "parser": record.get("parser"),
        "relation_supported": record.get("relation_supported"),
        "evidence_file_sha256": record.get("evidence_file_sha256"),
        "snapshot_present": record.get("snapshot_present"),
        "binding_source": record.get("binding_source"),
        "live_original_verified": record.get("live_original_verified"),
        "live_original_sha256": record.get("live_original_sha256"),
        "live_original_size_bytes": record.get(
            "live_original_size_bytes"
        ),
        "original_inventory_size_bytes": record.get(
            "original_inventory_size_bytes"
        ),
        "snapshot_sha256": record.get("snapshot_sha256"),
        "archive_member_sha256": record.get("archive_member_sha256"),
        "normalized_fields": record.get("normalized_fields"),
        "normalized_field_provenance": record.get(
            "normalized_field_provenance"
        ),
        "observed_original_run_identity": record.get(
            "observed_original_run_identity"
        ),
        "observed_outer_run_identity": record.get("observed_outer_run_identity"),
        "observed_population_source_run_identity": record.get(
            "observed_population_source_run_identity"
        ),
        "observed_audit_run_identity": record.get("observed_audit_run_identity"),
        "observed_producer_commit": record.get("observed_producer_commit"),
        "observed_artifact_path": record.get("observed_artifact_path"),
        "observed_artifact_sha256": record.get("observed_artifact_sha256"),
    }


def normalized_role_observations_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    ordered = sorted(
        (_role_digest_record(record) for record in records),
        key=lambda row: (
            _ROLE_ORDER.get(str(row.get("evidence_role")), 999),
            str(row.get("evidence_file_sha256") or ""),
        ),
    )
    return canonical_json_sha256(ordered)


def _path_is_run_identifier(value: Any) -> bool:
    return isinstance(value, str) and (
        _PATHISH_RE.match(value) is not None
        or "\\" in value
        or ("/" in value and value.count("/") > 0)
    )


def _single_role(
    by_role: Mapping[str, List[Mapping[str, Any]]],
    role: str,
) -> Mapping[str, Any]:
    rows = by_role.get(role) or []
    return rows[0] if len(rows) == 1 else {}


def validate_exact_role_evidence_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    role_specs: Mapping[str, Mapping[str, Any]],
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    audit_members: Optional[Sequence[Mapping[str, Any]]] = None,
    live_original_check: bool = False,
    require_live_original: bool = False,
) -> Dict[str, Any]:
    specs = _merged_specs(role_specs)
    failures: List[str] = []
    try:
        canonical_role_specs = canonical_resolved_role_specs_from_records(
            records,
            role_contracts=specs,
            audit_members=audit_members,
        )
    except _CanonicalResolvedRoleSpecsError as exc:
        canonical_role_specs = {}
        failures.extend(exc.failures)
    by_role: Dict[str, List[Mapping[str, Any]]] = {}
    audit_by_path = {
        str(row.get("relative_path")): row
        for row in (audit_members or [])
        if isinstance(row, Mapping)
    }
    historical_paths: List[Any] = []
    snapshot_paths: List[Any] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            failures.append("evidence_role_record_not_mapping:{}".format(index))
            continue
        role = str(record.get("evidence_role") or "")
        if role not in specs:
            failures.append("evidence_role_unsupported:{}".format(role))
            continue
        by_role.setdefault(role, []).append(record)
        spec = specs[role]
        if record.get("role_binding_method") != EXACT_ROLE_BINDING_METHOD:
            failures.append("{}_exact_role_binding_required".format(role))
        if record.get("role_binding_schema_version") != ROLE_BINDING_SCHEMA_VERSION:
            failures.append("{}_role_binding_schema_version_mismatch".format(role))
        if record.get("identity_domain") != spec.get("identity_domain"):
            failures.append("{}_identity_domain_mismatch".format(role))
        if record.get("path_authority_domain") != spec.get(
            "path_authority_domain"
        ):
            failures.append(
                "{}_path_authority_domain_mismatch".format(role)
            )
        if record.get("path_relation") != spec.get("path_relation"):
            failures.append("{}_path_relation_mismatch".format(role))
        if record.get("parser") != spec.get("parser"):
            failures.append("{}_parser_mismatch".format(role))
        if record.get("relation_supported") != spec.get("relation_supported"):
            failures.append("{}_relation_mismatch".format(role))
        if record.get("parse_status") != "ok":
            failures.append("{}_parse_status_not_ok".format(role))
        evidence_sha = record.get("evidence_file_sha256")
        original_sha = record.get("original_inventory_sha256")
        snapshot_sha = record.get("snapshot_sha256")
        archive_sha = record.get("archive_member_sha256")
        snapshot_present = record.get("snapshot_present")
        if snapshot_present is None:
            snapshot_present = record.get("snapshot_relative_path") not in (
                None,
                "",
            )
        binding_source = record.get("binding_source")
        inventory_size = record.get("original_inventory_size_bytes")
        if not _is_sha256(evidence_sha):
            failures.append("{}_evidence_file_sha256_invalid".format(role))
        expected_sha = spec.get("expected_sha256")
        if expected_sha is not None and evidence_sha != expected_sha:
            failures.append("{}_expected_sha256_mismatch".format(role))
        if record.get("expected_evidence_file_sha256") != evidence_sha:
            failures.append(
                "{}_recorded_expected_sha256_mismatch".format(role)
            )
        expected_basename = spec.get("expected_basename")
        historical = record.get("absolute_historical_path")
        snapshot_path = record.get("snapshot_relative_path")
        if (
            not isinstance(inventory_size, int)
            or isinstance(inventory_size, bool)
            or inventory_size < 0
        ):
            failures.append("{}_inventory_size_invalid".format(role))
        if record.get("file_size") != inventory_size:
            failures.append("{}_inventory_file_size_mismatch".format(role))
        if snapshot_present is True:
            if binding_source not in (
                None,
                "accepted_audit_inventory_plus_snapshot",
                "accepted_audit_inventory_plus_snapshot_and_live_original",
            ):
                failures.append("{}_snapshot_binding_source_invalid".format(role))
            if original_sha != snapshot_sha or original_sha != evidence_sha:
                failures.append(
                    "{}_original_snapshot_sha256_mismatch".format(role)
                )
            if snapshot_sha != archive_sha:
                failures.append(
                    "{}_snapshot_archive_sha256_mismatch".format(role)
                )
            if not isinstance(snapshot_path, str) or not snapshot_path:
                failures.append("{}_snapshot_relative_path_invalid".format(role))
        else:
            if snapshot_path is not None:
                failures.append(
                    "{}_snapshot_absent_relative_path_not_null".format(role)
                )
            if snapshot_sha is not None or archive_sha is not None:
                failures.append(
                    "{}_snapshot_absent_digest_not_null".format(role)
                )
            if binding_source != (
                "accepted_audit_inventory_plus_live_original"
            ):
                failures.append(
                    "{}_snapshotless_binding_source_invalid".format(role)
                )
            if original_sha != evidence_sha:
                failures.append(
                    "{}_inventory_live_original_sha256_mismatch".format(role)
                )
            if record.get("live_original_verified") is not True:
                failures.append(
                    "{}_snapshotless_live_original_not_verified".format(role)
                )
            if record.get("live_original_sha256") != evidence_sha:
                failures.append(
                    "{}_live_original_sha256_mismatch".format(role)
                )
            if record.get("live_original_size_bytes") != inventory_size:
                failures.append(
                    "{}_live_original_size_mismatch".format(role)
                )
        if expected_basename is not None:
            if Path(str(historical or "")).name != expected_basename:
                failures.append("{}_historical_basename_mismatch".format(role))
            if snapshot_present is True:
                snapshot_name = Path(str(snapshot_path or "")).name
                if not (
                    snapshot_name == expected_basename
                    or snapshot_name.endswith("_" + expected_basename)
                ):
                    failures.append(
                        "{}_snapshot_basename_mismatch".format(role)
                    )
        if audit_by_path and snapshot_present is True:
            actual = audit_by_path.get(str(snapshot_path))
            if actual is None:
                failures.append("{}_snapshot_missing_from_audit".format(role))
            elif actual.get("sha256") != archive_sha:
                failures.append("{}_archive_member_sha256_mismatch".format(role))
        live_path = Path(str(historical or ""))
        should_check_live = (
            snapshot_present is False
            or live_original_check
            or require_live_original
        )
        if should_check_live:
            if not live_path.is_file():
                failures.append("{}_live_original_missing".format(role))
            else:
                live_sha = sha256_file(live_path)
                live_size = live_path.stat().st_size
                if live_sha != evidence_sha:
                    failures.append(
                        "{}_live_original_sha256_mismatch".format(role)
                    )
                if live_size != inventory_size:
                    failures.append(
                        "{}_live_original_size_mismatch".format(role)
                    )
        historical_paths.append(historical)
        if snapshot_path not in (None, ""):
            snapshot_paths.append(snapshot_path)

        for field in (
            "observed_original_run_identity",
            "observed_outer_run_identity",
            "observed_population_source_run_identity",
            "observed_audit_run_identity",
        ):
            value = record.get(field)
            if value not in (None, "") and _path_is_run_identifier(value):
                failures.append("{}_run_identifier_is_directory_path".format(role))

    for role in REQUIRED_EVIDENCE_ROLES:
        rows = by_role.get(role) or []
        if not rows:
            failures.append("required_evidence_role_missing:{}".format(role))
        elif len(rows) > 1:
            failures.append("required_evidence_role_ambiguous:{}".format(role))
    if len(set(historical_paths)) != len(historical_paths) or len(
        set(snapshot_paths)
    ) != len(snapshot_paths):
        failures.append("required_role_physical_file_reused")

    population = _single_role(by_role, "external_population_comparison")
    population_fields = population.get("normalized_fields")
    population_fields = (
        population_fields
        if isinstance(population_fields, Mapping)
        else {}
    )
    population_historical_binding = _population_historical_directory_binding(
        population.get("absolute_historical_path")
    )
    failures.extend(population_historical_binding.get("failures") or [])
    for field in (
        "population_evidence_historical_path",
        "population_evidence_historical_directory",
        "population_evidence_directory",
        "population_path_root",
        "directory_derived_run_identity",
        "directory_derived_run_identity_status",
    ):
        if population_fields.get(field) != population_historical_binding.get(
            field
        ):
            failures.append(
                "external_population_{}_mismatch".format(field)
            )
    population_field_provenance = population.get(
        "normalized_field_provenance"
    )
    population_field_provenance = (
        population_field_provenance
        if isinstance(population_field_provenance, Mapping)
        else {}
    )
    for field, expected_source in (
        (
            "population_evidence_historical_path",
            "role_record.absolute_historical_path",
        ),
        (
            "population_evidence_historical_directory",
            "derived:absolute_historical_path.parent",
        ),
        (
            "population_evidence_directory",
            "derived:absolute_historical_path.parent",
        ),
        (
            "population_path_root",
            "derived:absolute_historical_path.parent.parent",
        ),
        (
            "directory_derived_run_identity",
            "derived:absolute_historical_path.parent",
        ),
    ):
        if population_field_provenance.get(field) != expected_source:
            failures.append(
                "external_population_{}_provenance_mismatch".format(field)
            )
    population_spec = specs.get("external_population_comparison") or {}
    if population_spec.get("require_exact_population_lists") is True:
        if population_fields.get("population_lists_present") is not True:
            failures.append("external_population_stable_sample_lists_missing")
        for field, expected_field in (
            ("train_count", "expected_train_count"),
            ("eval_count", "expected_eval_count"),
            ("aligned_count", "expected_aligned_count"),
            ("historical_train_digest", "expected_train_digest"),
            ("historical_eval_digest", "expected_eval_digest"),
            ("historical_aligned_digest", "expected_aligned_digest"),
            (
                "canonical_fitting_set_sha256",
                "expected_canonical_fitting_set_sha256",
            ),
            (
                "canonical_heldout_set_sha256",
                "expected_canonical_heldout_set_sha256",
            ),
            (
                "canonical_union_set_sha256",
                "expected_canonical_union_set_sha256",
            ),
        ):
            if population_fields.get(field) != population_spec.get(
                expected_field
            ):
                failures.append(
                    "external_population_{}_not_frozen".format(field)
                )
        if population_fields.get("duplicate_train_count") != 0:
            failures.append("external_population_fitting_duplicates")
        if population_fields.get("duplicate_eval_count") != 0:
            failures.append("external_population_heldout_duplicates")
        if population_fields.get("overlap_count") != 0:
            failures.append("external_population_intersection_nonzero")
        if population_fields.get("union_count") != population_spec.get(
            "expected_aligned_count"
        ):
            failures.append("external_population_union_count_mismatch")

    child = _single_role(by_role, "child_final_status")
    child_fields = child.get("normalized_fields")
    child_fields = child_fields if isinstance(child_fields, Mapping) else {}
    if child_fields.get("raw_status") != "ok":
        failures.append("child_status_not_ok")
    if child_fields.get("raw_failure_stage") != "complete":
        failures.append("child_failure_stage_not_complete")
    if child_fields.get("original_child_execution_status") != "complete":
        failures.append("child_execution_status_not_complete")
    if child_fields.get("original_child_exit_code") != 0:
        failures.append("child_exit_code_not_zero")

    outer = _single_role(by_role, "outer_full_acceptance")
    outer_fields = outer.get("normalized_fields")
    outer_fields = outer_fields if isinstance(outer_fields, Mapping) else {}
    if outer_fields.get("original_outer_acceptance_status") != "failed":
        failures.append("outer_acceptance_status_not_failed")
    if outer_fields.get("original_outer_acceptance_failures") != list(
        EXPECTED_OUTER_ACCEPTANCE_FAILURES
    ):
        failures.append("outer_acceptance_failures_mismatch")
    if (
        outer_fields.get("nested_child_status") is not None
        and outer_fields.get("nested_child_status")
        != child_fields.get("raw_status")
    ):
        failures.append("outer_nested_child_status_mismatch")
    if (
        outer_fields.get("nested_child_failure_stage") is not None
        and outer_fields.get("nested_child_failure_stage")
        != child_fields.get("raw_failure_stage")
    ):
        failures.append("outer_nested_child_failure_stage_mismatch")
    if (
        outer_fields.get("nested_child_exit_code") is not None
        and outer_fields.get("nested_child_exit_code")
        != child_fields.get("original_child_exit_code")
    ):
        failures.append("outer_nested_child_exit_code_mismatch")
    if (
        outer_fields.get("starting_git_commit") is not None
        or outer_fields.get("ending_git_commit") is not None
    ) and outer_fields.get("starting_git_commit") != outer_fields.get(
        "ending_git_commit"
    ):
        failures.append("outer_acceptance_git_commit_changed")

    child_exit = _single_role(by_role, "child_full_exit_code")
    child_exit_fields = child_exit.get("normalized_fields")
    child_exit_fields = (
        child_exit_fields
        if isinstance(child_exit_fields, Mapping)
        else {}
    )
    outer_exit = _single_role(by_role, "outer_acceptance_exit_code")
    outer_exit_fields = outer_exit.get("normalized_fields")
    outer_exit_fields = (
        outer_exit_fields
        if isinstance(outer_exit_fields, Mapping)
        else {}
    )
    if child_exit_fields.get("exit_code") != 0 or (
        child_exit_fields.get("exit_code")
        != child_fields.get("original_child_exit_code")
    ):
        failures.append("child_full_exit_code_sidecar_mismatch")
    authoritative_outer_exit_code = outer_exit_fields.get("exit_code")
    if authoritative_outer_exit_code != 1:
        failures.append("outer_acceptance_exit_code_sidecar_mismatch")
    json_exit_code_present = outer_fields.get(
        "outer_acceptance_json_exit_code_present"
    )
    json_exit_code = outer_fields.get("outer_acceptance_json_exit_code")
    if json_exit_code_present is not False and json_exit_code_present is not True:
        failures.append("outer_acceptance_json_exit_code_presence_invalid")
    elif json_exit_code_present is True and (
        json_exit_code != authoritative_outer_exit_code
    ):
        failures.append("outer_acceptance_exit_code_sidecar_mismatch")
    elif json_exit_code_present is False and json_exit_code is not None:
        failures.append("outer_acceptance_json_exit_code_absence_mismatch")

    replay_summary_required = (
        (specs.get("replay_summary") or {}).get("required_for_paper") is True
        or population_spec.get("require_exact_population_lists") is True
    )
    replay_summary_rows = by_role.get("replay_summary") or []
    if replay_summary_required:
        if not replay_summary_rows:
            failures.append("replay_summary_supporting_role_missing")
        elif len(replay_summary_rows) > 1:
            failures.append("replay_summary_supporting_role_ambiguous")

    outer_field_paths = outer.get("normalized_field_provenance")
    outer_field_paths = (
        outer_field_paths if isinstance(outer_field_paths, Mapping) else {}
    )
    replay_values = {
        field: outer_fields.get(field)
        for field in (
            "replay_matched_count",
            "replay_expected_count",
            "replay_mismatch_count",
        )
    }
    replay_provenance: Dict[str, Any] = {}
    for field, value in replay_values.items():
        source_path = outer_field_paths.get(field)
        if value is None or not source_path:
            failures.append("{}_source_missing".format(field))
            continue
        replay_provenance[field] = {
            "source_role": "outer_full_acceptance",
            "source_snapshot_path": outer.get("snapshot_relative_path"),
            "source_field_path": source_path,
            "source_file_sha256": outer.get("evidence_file_sha256"),
        }
    if replay_values.get("replay_matched_count") != 1427:
        failures.append("replay_matched_count_mismatch")
    if replay_values.get("replay_expected_count") != 1427:
        failures.append("replay_expected_count_mismatch")
    if replay_values.get("replay_matched_count") != replay_values.get(
        "replay_expected_count"
    ):
        failures.append("replay_matched_expected_count_mismatch")
    if replay_values.get("replay_mismatch_count") != 0:
        failures.append("replay_mismatch_count_not_zero")
    if replay_summary_required:
        replay_summary = _single_role(by_role, "replay_summary")
        replay_summary_fields = replay_summary.get("normalized_fields")
        replay_summary_fields = (
            replay_summary_fields
            if isinstance(replay_summary_fields, Mapping)
            else {}
        )
        if replay_summary_fields.get("replay_summary_status") != "ok":
            failures.append("replay_summary_status_not_ok")
        if (
            replay_summary_fields.get("replay_summary_comparison_status")
            != "matched"
        ):
            failures.append("replay_summary_comparison_status_not_matched")
        if outer_fields.get("replay_summary_exists") is not True:
            failures.append("outer_acceptance_replay_summary_exists_not_true")
        if _semantic_path(outer_fields.get("replay_summary_path")) != (
            _semantic_path(replay_summary.get("absolute_historical_path"))
        ):
            failures.append("outer_acceptance_replay_summary_path_mismatch")

    sidecar = _single_role(by_role, "artifact_identity")
    sidecar_fields = sidecar.get("normalized_fields")
    sidecar_fields = (
        sidecar_fields if isinstance(sidecar_fields, Mapping) else {}
    )
    sidecar_artifact_sha = sidecar_fields.get("artifact_sha256")
    sidecar_artifact_path = sidecar_fields.get("artifact_path")
    if not sidecar_artifact_path:
        failures.append("artifact_identity_artifact_path_missing")
    if sidecar_artifact_sha != expected_artifact_sha256:
        failures.append("artifact_identity_embedded_artifact_sha256_mismatch")
    if sidecar.get("evidence_file_sha256") == expected_artifact_sha256:
        failures.append("artifact_sidecar_sha_and_artifact_bytes_sha_conflated")

    summary = _single_role(by_role, "artifact_summary")
    summary_fields = summary.get("normalized_fields")
    summary_fields = (
        summary_fields if isinstance(summary_fields, Mapping) else {}
    )
    preflight = _single_role(by_role, "current_artifact_preflight")
    preflight_fields = preflight.get("normalized_fields")
    preflight_fields = (
        preflight_fields if isinstance(preflight_fields, Mapping) else {}
    )
    manifest = _single_role(by_role, "file_manifest")
    manifest_fields = manifest.get("normalized_fields")
    manifest_fields = (
        manifest_fields if isinstance(manifest_fields, Mapping) else {}
    )
    summary_artifact_sha = summary_fields.get("artifact_sha256")
    if (
        summary_artifact_sha is not None
        and summary_artifact_sha != expected_artifact_sha256
    ):
        failures.append("artifact_summary_artifact_sha256_mismatch")
    if preflight_fields.get("artifact_sha256") != expected_artifact_sha256:
        failures.append("artifact_preflight_artifact_sha256_mismatch")
    manifest_artifact_sha = manifest_fields.get("artifact_sha256")
    if (
        manifest_artifact_sha is not None
        and manifest_artifact_sha != expected_artifact_sha256
    ):
        failures.append("file_manifest_artifact_sha256_mismatch")
    historical_artifact_path = (
        summary_fields.get("artifact_path") or sidecar_artifact_path
    )
    if (
        summary_fields.get("artifact_path")
        and summary_fields.get("artifact_path") != sidecar_artifact_path
    ):
        failures.append("artifact_summary_sidecar_path_mismatch")
    summary_output_directory = summary_fields.get(
        "artifact_output_directory"
    )
    if (
        summary_output_directory
        and sidecar_artifact_path
        and str(Path(str(sidecar_artifact_path)).parent)
        != str(Path(str(summary_output_directory)))
    ):
        failures.append("artifact_summary_output_directory_mismatch")
    if summary_fields.get("runtime_policy_sha256") != (
        ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256
    ):
        failures.append("artifact_summary_runtime_policy_sha256_mismatch")
    if summary_fields.get("hybrid_fitting_policy_sha256") != (
        ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256
    ):
        failures.append("artifact_summary_hybrid_policy_sha256_mismatch")
    if outer_fields.get("top_level_artifact_sha256") != expected_artifact_sha256:
        failures.append("top_level_artifact_sha256_mismatch")
    if (
        outer_fields.get("nested_artifact_sha_present") is True
        and outer_fields.get("nested_artifact_sha256")
        != expected_artifact_sha256
    ):
        failures.append("nested_artifact_sha256_mismatch")
    for field in (
        "top_level_artifact_path",
        "nested_artifact_path",
    ):
        if _semantic_path(outer_fields.get(field)) != _semantic_path(
            sidecar_artifact_path
        ):
            failures.append("{}_mismatch".format(field))
    command = _single_role(by_role, "original_fitting_command")
    command_fields = command.get("normalized_fields")
    command_fields = (
        command_fields if isinstance(command_fields, Mapping) else {}
    )
    recomputed_command_selection = select_fitting_command_for_artifact(
        command_fields,
        historical_artifact_path,
    )
    if recomputed_command_selection.get("status") != "ok":
        failures.extend(recomputed_command_selection.get("failures") or [])
    for field in (
        "matching_fitting_command_count",
        "selected_fitting_command_index",
        "selected_fitting_command",
        "selected_fitting_command_sha256",
        "selected_fitting_invocation",
        "selected_input_manifest",
        "selected_dump_run_manifest",
        "selected_kv_manifest",
        "selected_hidden_manifest",
        "selected_split_mode",
        "selected_split_ratio",
        "selected_split_seed",
        "selected_output_artifact_path",
        "selected_output_directory",
        "selected_fitting_command_field_provenance",
    ):
        if command_fields.get(field) != recomputed_command_selection.get(
            field
        ):
            failures.append(
                "{}_recomputed_selection_mismatch".format(field)
            )
    if command_fields.get("matching_fitting_command_count") != 1:
        failures.append("fitting_command_for_accepted_artifact_not_unique")
    selected_command = command_fields.get("selected_fitting_command")
    if not isinstance(selected_command, Mapping):
        failures.append("selected_fitting_command_missing")
        selected_command = {}
    elif selected_command.get("classification") != "phase3c_fitting":
        failures.append("selected_fitting_command_not_producer")
    selected_command_sha = command_fields.get(
        "selected_fitting_command_sha256"
    )
    if selected_command_sha != canonical_json_sha256(
        {"argv": selected_command.get("argv") or []}
    ):
        failures.append("selected_fitting_command_sha256_mismatch")
    if _semantic_path(
        command_fields.get("selected_output_artifact_path")
    ) != _semantic_path(historical_artifact_path):
        failures.append("selected_fitting_command_artifact_path_mismatch")
    selected_output_directory = _semantic_path(
        command_fields.get("selected_output_directory")
    )
    if selected_output_directory is not None and _semantic_path(
        str(PurePosixPath(str(historical_artifact_path)).parent)
    ) != selected_output_directory:
        failures.append("selected_fitting_command_output_directory_mismatch")
    for field in (
        "selected_dump_run_manifest",
        "selected_kv_manifest",
        "selected_hidden_manifest",
    ):
        if command_fields.get(field) in (None, ""):
            failures.append("{}_missing".format(field))

    fit_summary = _single_role(by_role, "fit_summary")
    fit_summary_fields = fit_summary.get("normalized_fields")
    fit_summary_fields = (
        fit_summary_fields
        if isinstance(fit_summary_fields, Mapping)
        else {}
    )
    fit_split = fit_summary_fields.get("split_identity")
    fit_split = fit_split if isinstance(fit_split, Mapping) else {}
    artifact_split = summary_fields.get("split_identity")
    artifact_split = (
        artifact_split if isinstance(artifact_split, Mapping) else {}
    )
    selected_split_contract = {
        "split_mode": command_fields.get("selected_split_mode"),
        "split_ratio": command_fields.get("selected_split_ratio"),
        "split_seed": command_fields.get("selected_split_seed"),
    }
    expected_split_contract = {
        "split_mode": "shuffle",
        "split_ratio": 0.5,
        "split_seed": 0,
    }
    for field, expected in expected_split_contract.items():
        if selected_split_contract.get(field) != expected:
            failures.append(
                "selected_fitting_command_{}_mismatch".format(field)
            )
        if fit_split.get(field) != expected:
            failures.append("fit_summary_{}_mismatch".format(field))
        if artifact_split.get(field) != expected:
            failures.append("artifact_summary_{}_mismatch".format(field))
    current_artifact_path = preflight_fields.get("artifact_path")
    relocation_valid = bool(
        historical_artifact_path
        and current_artifact_path
        and sidecar_artifact_sha == preflight_fields.get("artifact_sha256")
        == expected_artifact_sha256
    )
    if historical_artifact_path != current_artifact_path and not relocation_valid:
        failures.append("artifact_path_relocation_unbound")

    domain_values: Dict[str, List[str]] = {
        IDENTITY_DOMAIN_POPULATION: [],
        IDENTITY_DOMAIN_CHILD: [],
        IDENTITY_DOMAIN_OUTER: [],
        IDENTITY_DOMAIN_AUDIT: [],
    }
    child_commits: List[str] = []
    for record in records:
        domain = record.get("identity_domain")
        field = {
            IDENTITY_DOMAIN_POPULATION: "observed_population_source_run_identity",
            IDENTITY_DOMAIN_CHILD: "observed_original_run_identity",
            IDENTITY_DOMAIN_OUTER: "observed_outer_run_identity",
            IDENTITY_DOMAIN_AUDIT: "observed_audit_run_identity",
        }.get(str(domain))
        value = record.get(field) if field else None
        if isinstance(value, str) and value:
            domain_values[str(domain)].append(value)
        if domain == IDENTITY_DOMAIN_CHILD:
            commit = record.get("observed_producer_commit")
            if isinstance(commit, str) and commit:
                child_commits.append(commit)
    domain_failures = {
        IDENTITY_DOMAIN_POPULATION: "population_source_run_identity_mismatch",
        IDENTITY_DOMAIN_CHILD: "original_child_fitting_run_identity_mismatch",
        IDENTITY_DOMAIN_OUTER: "original_outer_acceptance_run_identity_mismatch",
        IDENTITY_DOMAIN_AUDIT: "current_audit_preflight_run_identity_mismatch",
    }
    domain_identity: Dict[str, Optional[str]] = {}
    for domain, values in domain_values.items():
        unique = set(values)
        if len(unique) > 1:
            failures.append(domain_failures[domain])
        domain_identity[domain] = next(iter(unique)) if len(unique) == 1 else None
    if len(set(child_commits)) > 1:
        failures.append("original_child_fitting_producer_commit_mismatch")
    child_run = domain_identity[IDENTITY_DOMAIN_CHILD]
    if not child_run:
        failures.append("child_outer_parent_relation_missing_or_mismatch")
    child_status_historical_path = _single_role(
        by_role, "child_final_status"
    ).get("absolute_historical_path")
    outer_acceptance_historical_path = outer.get(
        "absolute_historical_path"
    )
    child_run_directory = (
        _semantic_path(str(PurePosixPath(str(child_status_historical_path)).parent))
        if child_status_historical_path
        else None
    )
    outer_run_directory = (
        _semantic_path(
            str(PurePosixPath(str(outer_acceptance_historical_path)).parent)
        )
        if outer_acceptance_historical_path
        else None
    )
    child_nested_under_outer = False
    if child_run_directory and outer_run_directory:
        try:
            PurePosixPath(child_run_directory).relative_to(
                PurePosixPath(outer_run_directory)
            )
            child_nested_under_outer = (
                child_run_directory != outer_run_directory
            )
        except ValueError:
            child_nested_under_outer = False
    if not child_nested_under_outer:
        failures.append("child_run_directory_not_nested_under_outer")
    child_exit_historical_path = child_exit.get("absolute_historical_path")
    outer_exit_historical_path = outer_exit.get("absolute_historical_path")
    child_exit_parent = (
        _semantic_path(
            str(PurePosixPath(str(child_exit_historical_path)).parent)
        )
        if child_exit_historical_path
        else None
    )
    outer_exit_parent = (
        _semantic_path(
            str(PurePosixPath(str(outer_exit_historical_path)).parent)
        )
        if outer_exit_historical_path
        else None
    )
    if child_exit_parent != outer_run_directory:
        failures.append("full_run_exit_code_run_directory_mismatch")
    if outer_exit_parent != outer_run_directory:
        failures.append(
            "outer_acceptance_exit_code_run_directory_mismatch"
        )
    manifest_historical_path = manifest.get("absolute_historical_path")
    manifest_parent = (
        _semantic_path(
            str(PurePosixPath(str(manifest_historical_path)).parent)
        )
        if manifest_historical_path
        else None
    )
    if manifest_parent != outer_run_directory:
        failures.append("full_run_file_manifest_run_directory_mismatch")
    manifest_artifact_relative_path = manifest_fields.get(
        "artifact_relative_path"
    )
    manifest_logical_root_candidates: List[Tuple[str, str]] = []
    if (
        child_run_directory
        and manifest_artifact_relative_path
        and _semantic_path(
            str(
                PurePosixPath(child_run_directory)
                / str(manifest_artifact_relative_path)
            )
        )
        == _semantic_path(historical_artifact_path)
    ):
        manifest_logical_root_candidates.append(
            ("child_final_status_parent", child_run_directory)
        )

    def logical_root_from_artifact(
        source: str,
        artifact_path: Any,
    ) -> None:
        if not artifact_path or not manifest_artifact_relative_path:
            return
        artifact = PurePosixPath(str(_semantic_path(artifact_path)))
        relative = PurePosixPath(str(manifest_artifact_relative_path))
        if len(artifact.parts) < len(relative.parts):
            return
        if artifact.parts[-len(relative.parts) :] != relative.parts:
            return
        root_parts = artifact.parts[: -len(relative.parts)]
        if not root_parts:
            return
        manifest_logical_root_candidates.append(
            (source, str(PurePosixPath(*root_parts)))
        )

    for source, artifact_path in (
        ("artifact_identity", sidecar_artifact_path),
        ("artifact_summary", summary_fields.get("artifact_path")),
        (
            "outer_acceptance_top_level",
            outer_fields.get("top_level_artifact_path"),
        ),
        (
            "outer_acceptance_nested",
            outer_fields.get("nested_artifact_path"),
        ),
    ):
        logical_root_from_artifact(source, artifact_path)
    if _semantic_path(current_artifact_path) == _semantic_path(
        historical_artifact_path
    ):
        logical_root_from_artifact(
            "current_artifact_preflight", current_artifact_path
        )
    artifact_root_support = [
        row
        for row in manifest_logical_root_candidates
        if row[0] != "child_final_status_parent"
    ]
    unique_manifest_logical_roots = {
        root for _, root in manifest_logical_root_candidates
    }
    if not artifact_root_support:
        failures.append(
            "full_run_file_manifest_logical_root_candidate_missing"
        )
    if len(unique_manifest_logical_roots) != 1:
        failures.append(
            "full_run_file_manifest_logical_root_candidate_{}".format(
                "missing"
                if not unique_manifest_logical_roots
                else "ambiguous"
            )
        )
        manifest_logical_root = None
    else:
        manifest_logical_root = next(
            iter(unique_manifest_logical_roots)
        )
    resolved_manifest_artifact_path = (
        _semantic_path(
            str(
                PurePosixPath(str(manifest_logical_root))
                / str(manifest_artifact_relative_path)
            )
        )
        if manifest_logical_root and manifest_artifact_relative_path
        else None
    )
    if resolved_manifest_artifact_path != _semantic_path(
        historical_artifact_path
    ):
        failures.append("full_run_file_manifest_artifact_path_mismatch")
    manifest_artifact_size = manifest_fields.get("artifact_size_bytes")
    sidecar_artifact_size = sidecar_fields.get("artifact_size")
    if (
        sidecar_artifact_size is not None
        and manifest_artifact_size != sidecar_artifact_size
    ):
        failures.append("full_run_file_manifest_artifact_size_mismatch")
    artifact_inside_child = False
    if child_run_directory and historical_artifact_path:
        try:
            PurePosixPath(
                str(_semantic_path(historical_artifact_path))
            ).relative_to(PurePosixPath(child_run_directory))
            artifact_inside_child = True
        except ValueError:
            artifact_inside_child = False
    if not artifact_inside_child:
        failures.append("accepted_artifact_not_within_child_run_directory")

    producer_git = _single_role(by_role, "producer_git_identity")
    producer_git_fields = producer_git.get("normalized_fields")
    producer_git_fields = (
        producer_git_fields
        if isinstance(producer_git_fields, Mapping)
        else {}
    )
    producer_git_commit = producer_git_fields.get("producer_commit")
    producer_git_branch = producer_git_fields.get("producer_branch")
    producer_tracked_status = producer_git_fields.get(
        "producer_tracked_status"
    )
    producer_tracked_worktree_clean = producer_git_fields.get(
        "tracked_worktree_clean"
    )
    if producer_tracked_worktree_clean is not True:
        failures.append("producer_git_tracked_worktree_not_clean")
    child_commit = (
        next(iter(set(child_commits))) if len(set(child_commits)) == 1 else None
    )
    outer_starting_commit = outer_fields.get("starting_git_commit")
    outer_ending_commit = outer_fields.get("ending_git_commit")
    if (
        child_commit is None
        or producer_git_commit != child_commit
        or (
            outer_starting_commit is not None
            and outer_starting_commit != child_commit
        )
        or (
            outer_ending_commit is not None
            and outer_ending_commit != child_commit
        )
    ):
        failures.append("child_outer_repository_commit_mismatch")

    child_outer_relation_evidence = {
        "child_status_historical_path": child_status_historical_path,
        "outer_acceptance_historical_path": outer_acceptance_historical_path,
        "child_run_directory": child_run_directory,
        "outer_run_directory": outer_run_directory,
        "child_nested_under_outer": child_nested_under_outer,
        "historical_artifact_path": historical_artifact_path,
        "artifact_inside_child_run_directory": artifact_inside_child,
        "full_run_exit_code_historical_path": child_exit_historical_path,
        "full_run_exit_code_parent_directory": child_exit_parent,
        "child_exit_code_historical_path": child_exit_historical_path,
        "child_exit_code_parent_directory": child_exit_parent,
        "outer_exit_code_historical_path": outer_exit_historical_path,
        "outer_exit_code_parent_directory": outer_exit_parent,
        "full_run_file_manifest_historical_path": manifest_historical_path,
        "full_run_file_manifest_parent_directory": manifest_parent,
        "manifest_physical_parent": manifest_parent,
        "manifest_logical_resolution_root": manifest_logical_root,
        "manifest_logical_root_candidates": [
            {"source": source, "root": root}
            for source, root in manifest_logical_root_candidates
        ],
        "full_run_file_manifest_artifact_relative_path": (
            manifest_artifact_relative_path
        ),
        "full_run_file_manifest_resolved_artifact_path": (
            resolved_manifest_artifact_path
        ),
        "artifact_path_matches_outer_acceptance": (
            _semantic_path(historical_artifact_path)
            == _semantic_path(outer_fields.get("top_level_artifact_path"))
            == _semantic_path(outer_fields.get("nested_artifact_path"))
        ),
        "artifact_sha_matches_outer_acceptance": (
            sidecar_artifact_sha
            == outer_fields.get("top_level_artifact_sha256")
            and (
                outer_fields.get("nested_artifact_sha_present") is not True
                or sidecar_artifact_sha
                == outer_fields.get("nested_artifact_sha256")
            )
        ),
        "child_commit": child_commit,
        "outer_starting_commit": outer_starting_commit,
        "outer_ending_commit": outer_ending_commit,
        "producer_git_commit": producer_git_commit,
        "producer_git_branch": producer_git_branch,
        "producer_tracked_status": producer_tracked_status,
        "producer_tracked_worktree_clean": (
            producer_tracked_worktree_clean
        ),
        "supporting_field_provenance": {
            "outer_nested_child_status": (
                outer.get("normalized_field_provenance") or {}
            ).get("nested_child_status"),
            "outer_nested_child_failure_stage": (
                outer.get("normalized_field_provenance") or {}
            ).get("nested_child_failure_stage"),
            "outer_nested_child_exit_code": (
                outer.get("normalized_field_provenance") or {}
            ).get("nested_child_exit_code"),
            "outer_artifact_path": (
                outer.get("normalized_field_provenance") or {}
            ).get("top_level_artifact_path"),
            "outer_nested_artifact_path": (
                outer.get("normalized_field_provenance") or {}
            ).get("nested_artifact_path"),
            "outer_artifact_sha256": (
                outer.get("normalized_field_provenance") or {}
            ).get("top_level_artifact_sha256"),
            "full_run_file_manifest_artifact_path": (
                manifest.get("normalized_field_provenance") or {}
            ).get("artifact_relative_path"),
            "child_status_file": child_status_historical_path,
            "outer_acceptance_file": outer_acceptance_historical_path,
        },
    }

    paper_valid = not failures
    return {
        "schema_version": ROLE_BINDING_SCHEMA_VERSION,
        "status": "ok" if paper_valid else "failed",
        "failures": sorted(set(failures)),
        "paper_role_binding_valid": paper_valid,
        "required_evidence_roles": list(REQUIRED_EVIDENCE_ROLES),
        "evidence_files": sorted(
            (dict(record) for record in records),
            key=lambda row: _ROLE_ORDER.get(
                str(row.get("evidence_role")), 999
            ),
        ),
        "multi_file_evidence_chain_sha256": normalized_role_observations_sha256(
            records
        ),
        "normalized_role_observations_sha256": (
            normalized_role_observations_sha256(records)
        ),
        "identity_domains": {
            "population_source_run_identifier": domain_identity[
                IDENTITY_DOMAIN_POPULATION
            ],
            "original_child_run_identifier": child_run,
            "original_outer_run_identifier": domain_identity[
                IDENTITY_DOMAIN_OUTER
            ],
            "current_audit_run_identifier": domain_identity[
                IDENTITY_DOMAIN_AUDIT
            ],
        },
        "population_source_run_identifier": domain_identity[
            IDENTITY_DOMAIN_POPULATION
        ],
        "population_explicit_run_identifier_status": population_fields.get(
            "explicit_run_identifier_status"
        ),
        "population_explicit_run_identifier": population_fields.get(
            "explicit_run_identifier"
        ),
        "population_evidence_historical_path": (
            population_historical_binding.get(
                "population_evidence_historical_path"
            )
        ),
        "population_evidence_historical_directory": (
            population_historical_binding.get(
                "population_evidence_historical_directory"
            )
        ),
        "population_directory_derived_run_identity": (
            population_historical_binding.get(
                "directory_derived_run_identity"
            )
        ),
        "population_directory_derived_run_identity_status": (
            population_historical_binding.get(
                "directory_derived_run_identity_status"
            )
        ),
        "population_evidence_directory": population_historical_binding.get(
            "population_evidence_directory"
        ),
        "population_path_root": population_historical_binding.get(
            "population_path_root"
        ),
        "original_child_run_identifier": child_run,
        "original_child_run_directory": child_run_directory,
        "child_run_directory_identity": (
            canonical_json_sha256(
                {"normalized_run_directory": child_run_directory}
            )
            if child_run_directory
            else None
        ),
        "original_outer_run_identifier": domain_identity[
            IDENTITY_DOMAIN_OUTER
        ],
        "original_outer_run_identity_status": (
            "explicit_evidence_identity"
            if domain_identity[IDENTITY_DOMAIN_OUTER] not in (None, "")
            else "directory_derived_only_pending_central_review"
        ),
        "original_outer_run_directory": outer_run_directory,
        "outer_run_directory_identity": (
            canonical_json_sha256(
                {"normalized_run_directory": outer_run_directory}
            )
            if outer_run_directory
            else None
        ),
        "current_audit_run_identifier": domain_identity[IDENTITY_DOMAIN_AUDIT],
        "original_run_identity": child_run,
        "producer_commit": (
            next(iter(set(child_commits)))
            if len(set(child_commits)) == 1
            else None
        ),
        "producer_branch": producer_git_branch,
        "producer_tracked_status": producer_tracked_status,
        "producer_tracked_worktree_clean": (
            producer_tracked_worktree_clean
        ),
        "artifact_identity_sidecar_sha256": sidecar.get(
            "evidence_file_sha256"
        ),
        "artifact_bytes_sha256": sidecar_artifact_sha,
        "historical_artifact_path": historical_artifact_path,
        "current_artifact_path": current_artifact_path,
        "artifact_path": current_artifact_path,
        "artifact_relocation_binding": {
            "historical_artifact_path": historical_artifact_path,
            "current_artifact_path": current_artifact_path,
            "artifact_bytes_sha256": sidecar_artifact_sha,
            "relocation_binding_valid": relocation_valid,
        },
        "replay_matched_count": replay_values.get("replay_matched_count"),
        "replay_expected_count": replay_values.get("replay_expected_count"),
        "replay_mismatch_count": replay_values.get("replay_mismatch_count"),
        "replay_field_provenance": replay_provenance,
        "replay_summary_status": (
            replay_summary_fields.get("replay_summary_status")
            if replay_summary_required
            else None
        ),
        "replay_summary_comparison_status": (
            replay_summary_fields.get("replay_summary_comparison_status")
            if replay_summary_required
            else None
        ),
        "replay_summary_historical_path": (
            replay_summary.get("absolute_historical_path")
            if replay_summary_required
            else None
        ),
        "outer_acceptance_replay_summary_path": outer_fields.get(
            "replay_summary_path"
        ),
        "outer_acceptance_replay_summary_exists": outer_fields.get(
            "replay_summary_exists"
        ),
        "outer_acceptance_json_exit_code_present": (
            json_exit_code_present
        ),
        "outer_acceptance_json_exit_code": json_exit_code,
        "original_outer_acceptance_exit_code": (
            authoritative_outer_exit_code
        ),
        "original_outer_acceptance_exit_code_source_role": (
            "outer_acceptance_exit_code"
        ),
        "original_outer_acceptance_exit_code_source_path": (
            outer_exit.get("absolute_historical_path")
        ),
        "selected_fitting_command": command_fields.get(
            "selected_fitting_command"
        ),
        "selected_fitting_command_sha256": command_fields.get(
            "selected_fitting_command_sha256"
        ),
        "selected_fitting_command_field_provenance": command_fields.get(
            "selected_fitting_command_field_provenance"
        ),
        "selected_dump_run_manifest": command_fields.get(
            "selected_dump_run_manifest"
        ),
        "selected_kv_manifest": command_fields.get("selected_kv_manifest"),
        "selected_hidden_manifest": command_fields.get(
            "selected_hidden_manifest"
        ),
        "d2_command_binding": {
            "selected_dump_run_manifest": command_fields.get(
                "selected_dump_run_manifest"
            ),
            "selected_kv_manifest": command_fields.get(
                "selected_kv_manifest"
            ),
            "selected_hidden_manifest": command_fields.get(
                "selected_hidden_manifest"
            ),
            "artifact_identity_relation": (
                "command_paths_observed_artifact_identity_pending_"
                "exact_server_integration"
            ),
        },
        "original_full_run_file_manifest_path": manifest_historical_path,
        "original_full_run_file_manifest_sha256": manifest.get(
            "evidence_file_sha256"
        ),
        "original_full_run_file_manifest_entry_count": manifest_fields.get(
            "manifest_entry_count"
        ),
        "original_full_run_file_manifest_artifact_relative_path": (
            manifest_artifact_relative_path
        ),
        "original_full_run_file_manifest_artifact_size_bytes": (
            manifest_artifact_size
        ),
        "original_full_run_file_manifest_physical_parent": manifest_parent,
        "original_full_run_file_manifest_logical_resolution_root": (
            manifest_logical_root
        ),
        "outer_acceptance_artifact_path_field": outer_fields.get(
            "outer_acceptance_artifact_path_field"
        ),
        "nested_child_artifact_path_field": outer_fields.get(
            "nested_child_artifact_path_field"
        ),
        "nested_artifact_sha_present": outer_fields.get(
            "nested_artifact_sha_present"
        ),
        "child_outer_relation_evidence": child_outer_relation_evidence,
        "outer_acceptance_normalized_field_provenance": dict(
            outer.get("normalized_field_provenance") or {}
        ),
        "normalized_field_provenance": {
            str(record.get("evidence_role")): dict(
                record.get("normalized_field_provenance") or {}
            )
            for record in records
        },
        "role_specific_observations": {
            str(record.get("evidence_role")): _role_digest_record(record)
            for record in records
        },
        "resolved_role_specs": canonical_role_specs,
    }


def extract_artifact_provenance_identities(
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    failures: List[str] = []
    provenance = artifact.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    dump_binding = provenance.get("dump_run_binding")
    dump_binding = dump_binding if isinstance(dump_binding, Mapping) else {}

    def resolve(
        label: str,
        top_key: str,
        nested_key: str,
    ) -> Tuple[Any, Optional[str]]:
        top = provenance.get(top_key)
        nested = dump_binding.get(nested_key)
        if top not in (None, "") and nested not in (None, "") and top != nested:
            failures.append("artifact_{}_conflict".format(label))
            return None, None
        if nested not in (None, ""):
            return nested, "$.provenance.dump_run_binding.{}".format(nested_key)
        if top not in (None, ""):
            return top, "$.provenance.{}".format(top_key)
        failures.append("artifact_{}_missing".format(label))
        return None, None

    checkpoint, checkpoint_path = resolve(
        "checkpoint_identity",
        "model_checkpoint_identity_sha256",
        "model_checkpoint_identity_sha256",
    )
    tokenizer, tokenizer_path = resolve(
        "tokenizer_identity",
        "tokenizer_identity_sha256",
        "tokenizer_identity_sha256",
    )
    commit_candidates = [
        (
            provenance.get("repository_commit"),
            "$.provenance.repository_commit",
        ),
        (provenance.get("git_commit"), "$.provenance.git_commit"),
        (
            provenance.get("starting_git_commit"),
            "$.provenance.starting_git_commit",
        ),
        (
            dump_binding.get("repository_commit"),
            "$.provenance.dump_run_binding.repository_commit",
        ),
    ]
    present_commits = [
        (value, path)
        for value, path in commit_candidates
        if value not in (None, "")
    ]
    unique_commits = {value for value, _ in present_commits}
    if len(unique_commits) > 1:
        failures.append("artifact_producer_commit_conflict")
        commit, commit_path = None, None
    elif present_commits:
        commit, commit_path = present_commits[0]
    else:
        failures.append("artifact_producer_commit_missing")
        commit, commit_path = None, None
    binding_identity = provenance.get("dump_run_binding_sha256")
    binding_path = "$.provenance.dump_run_binding_sha256"
    if not _is_sha256(binding_identity):
        failures.append("artifact_dump_run_binding_identity_missing_or_invalid")
        binding_identity = None
        binding_path = None
    run_candidates = [
        (
            provenance.get("run_identity"),
            "$.provenance.run_identity",
        ),
        (
            provenance.get("original_run_identity"),
            "$.provenance.original_run_identity",
        ),
    ]
    present_run_candidates = [
        (value, path)
        for value, path in run_candidates
        if value not in (None, "")
    ]
    invalid_run_candidates = [
        (value, path)
        for value, path in present_run_candidates
        if not isinstance(value, str) or not value.strip()
    ]
    unique_run_identities = {
        value
        for value, _ in present_run_candidates
        if isinstance(value, str)
    }
    if invalid_run_candidates:
        failures.append("artifact_original_run_identity_invalid")
        run_identity = None
        run_path = None
        run_identity_present = True
        run_identity_status = "invalid"
    elif len(unique_run_identities) > 1:
        failures.append("artifact_original_run_identity_alias_conflict")
        run_identity = None
        run_path = None
        run_identity_present = True
        run_identity_status = "conflicting"
    elif present_run_candidates:
        run_identity, run_path = present_run_candidates[0]
        run_identity_present = True
        run_identity_status = "found_and_verified"
    else:
        run_identity = None
        run_path = None
        run_identity_present = False
        run_identity_status = "not_found"
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "dump_run_binding": dict(dump_binding),
        "dump_run_binding_identity": binding_identity,
        "checkpoint_identity": checkpoint,
        "tokenizer_identity": tokenizer,
        "artifact_original_run_identity": run_identity,
        "artifact_original_run_identity_present": run_identity_present,
        "artifact_original_run_identity_status": run_identity_status,
        "artifact_original_producer_commit": commit,
        "normalized_field_provenance": {
            "dump_run_binding": "$.provenance.dump_run_binding",
            "dump_run_binding_identity": binding_path,
            "checkpoint_identity": checkpoint_path,
            "tokenizer_identity": tokenizer_path,
            "artifact_original_run_identity": run_path,
            "artifact_original_producer_commit": commit_path,
        },
    }


def _jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("jsonl_row_not_mapping:{}".format(line_no))
        rows.append(payload)
    return rows


def resolve_exact_roles_from_verified_audit(
    audit_verification: Mapping[str, Any],
    *,
    role_specs: Mapping[str, Mapping[str, Any]] = ACCEPTED_AUDIT_ROLE_SPECS,
    expected_artifact_sha256: str = ACCEPTED_PHASE3C_ARTIFACT_SHA256,
    live_original_check: bool = False,
    require_live_original: bool = False,
) -> Dict[str, Any]:
    """Resolve the accepted root by exact SHA/name semantics.

    SHAs not centrally published as module constants are read from the
    accepted archive's original inventory only after the archive root and
    snapshot bytes have been verified.
    """

    failures: List[str] = []
    root_value = audit_verification.get("extracted_root")
    archive_root_name = audit_verification.get("archive_root_name")
    if not root_value or not archive_root_name:
        return {
            "status": "failed",
            "failures": ["audit_extracted_root_required_for_exact_role_resolution"],
            "paper_role_binding_valid": False,
            "evidence_files": [],
        }
    root = Path(str(root_value))
    inventory_path = root / "evidence_file_inventory.jsonl"
    if not inventory_path.is_file():
        return {
            "status": "failed",
            "failures": ["audit_evidence_file_inventory_missing"],
            "paper_role_binding_valid": False,
            "evidence_files": [],
        }
    try:
        inventory = _jsonl(inventory_path)
    except Exception as exc:
        return {
            "status": "failed",
            "failures": [
                "audit_evidence_file_inventory_parse_failed:{}".format(
                    type(exc).__name__
                )
            ],
            "paper_role_binding_valid": False,
            "evidence_files": [],
        }
    specs = _merged_specs(role_specs)
    records: List[Dict[str, Any]] = []
    resolved_specs: Dict[str, Dict[str, Any]] = {}
    historical_artifact_path: Optional[str] = None
    child_run_directory: Optional[str] = None
    outer_run_directory: Optional[str] = None
    resolution_order = (
        "artifact_identity",
        "outer_full_acceptance",
        "external_population_comparison",
        "original_fitting_command",
        "fit_summary",
        "artifact_summary",
        "child_final_status",
        "current_artifact_preflight",
        "producer_git_identity",
        "child_full_exit_code",
        "outer_acceptance_exit_code",
        "file_manifest",
    )

    def snapshot_for_row(row: Mapping[str, Any]) -> Optional[Path]:
        historical = str(row.get("absolute_path") or "")
        if not historical:
            return None
        prefix = hashlib.sha256(historical.encode("utf-8")).hexdigest()[:12]
        candidate = root / "evidence_snapshots" / "{}_{}".format(
            prefix, Path(historical).name
        )
        return candidate if candidate.is_file() else None

    def evidence_path_for_candidate(
        row: Mapping[str, Any],
    ) -> Optional[Path]:
        snapshot = snapshot_for_row(row)
        if snapshot is not None:
            return snapshot
        historical = row.get("absolute_path")
        if not isinstance(historical, str) or not historical:
            return None
        live = Path(historical)
        return live if live.is_file() else None

    def inventory_size(
        role: str,
        row: Mapping[str, Any],
    ) -> Optional[int]:
        observed = [
            row[key]
            for key in ("size_bytes", "file_size")
            if key in row and row[key] is not None
        ]
        unique = {
            value
            for value in observed
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if (
            not observed
            or len(unique) != 1
            or len(unique) != len(set(observed))
            or next(iter(unique), -1) < 0
        ):
            failures.append("{}_inventory_size_invalid".format(role))
            return None
        return next(iter(unique))

    def record_from_inventory_row(
        role: str,
        row: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        historical = str(row.get("absolute_path") or "")
        row_sha = row.get("sha256")
        size = inventory_size(role, row)
        if not historical:
            failures.append("{}_inventory_absolute_path_missing".format(role))
            return None
        if not _is_sha256(row_sha):
            failures.append("{}_inventory_sha256_invalid".format(role))
            return None
        if size is None:
            return None
        expected_sha = spec.get("expected_sha256")
        if expected_sha is not None and row_sha != expected_sha:
            failures.append("{}_inventory_expected_sha256_mismatch".format(role))
            return None

        generated = bool(row.get("_generated_current_preflight"))
        snapshot = (
            root / str(spec.get("expected_basename"))
            if generated
            else snapshot_for_row(row)
        )
        snapshot_present = snapshot is not None and snapshot.is_file()
        live = Path(historical)
        must_check_live = (
            not snapshot_present
            or live_original_check
            or require_live_original
        )
        live_verified = False
        live_sha: Optional[str] = None
        live_size: Optional[int] = None
        if must_check_live:
            if not live.is_file():
                failures.append("{}_live_original_missing".format(role))
                return None
            live_sha = sha256_file(live)
            live_size = live.stat().st_size
            if live_sha != row_sha:
                failures.append(
                    "{}_inventory_live_original_sha256_mismatch".format(role)
                )
                return None
            if live_size != size:
                failures.append(
                    "{}_inventory_live_original_size_mismatch".format(role)
                )
                return None
            live_verified = True

        if snapshot_present:
            evidence_path = snapshot
            snapshot_sha = sha256_file(snapshot)
            if snapshot_sha != row_sha:
                failures.append(
                    "{}_inventory_snapshot_sha256_mismatch".format(role)
                )
            if snapshot.stat().st_size != size:
                failures.append(
                    "{}_inventory_snapshot_size_mismatch".format(role)
                )
            if generated:
                snapshot_relative = "{}/{}".format(
                    archive_root_name,
                    spec.get("expected_basename"),
                )
            else:
                prefix = hashlib.sha256(
                    historical.encode("utf-8")
                ).hexdigest()[:12]
                snapshot_relative = (
                    "{}/evidence_snapshots/{}_{}".format(
                        archive_root_name,
                        prefix,
                        Path(historical).name,
                    )
                )
            members = [
                member
                for member in audit_verification.get("members") or []
                if member.get("relative_path") == snapshot_relative
            ]
            if len(members) != 1:
                failures.append(
                    "{}_snapshot_archive_member_not_unique".format(role)
                )
                archive_sha = None
            else:
                archive_sha = members[0].get("sha256")
            binding_source = (
                "accepted_audit_inventory_plus_snapshot_and_live_original"
                if live_verified
                else "accepted_audit_inventory_plus_snapshot"
            )
        else:
            # A snapshotless role is authenticated only by the accepted
            # inventory and bytes at that exact historical path.
            evidence_path = live
            snapshot_relative = None
            snapshot_sha = None
            archive_sha = None
            binding_source = "accepted_audit_inventory_plus_live_original"

        return _record_from_file(
            role,
            evidence_path,
            spec=spec,
            expected_relative_path=str(
                row.get("relative_path")
                or spec.get("expected_basename")
                or ""
            ),
            historical_path=historical,
            original_inventory_sha256=str(row_sha),
            snapshot_relative_path=snapshot_relative,
            snapshot_sha256=snapshot_sha,
            archive_member_sha256=archive_sha,
            snapshot_present=snapshot_present,
            binding_source=binding_source,
            live_original_verified=live_verified,
            live_original_sha256=live_sha,
            live_original_size_bytes=live_size,
            original_inventory_size_bytes=size,
        )

    for role in resolution_order:
        spec = dict(specs[role])
        basename = spec.get("expected_basename")
        expected_sha = spec.get("expected_sha256")
        candidates: List[Mapping[str, Any]] = []
        for row in inventory:
            historical = str(row.get("absolute_path") or "")
            row_sha = row.get("sha256")
            if expected_sha is not None and row_sha != expected_sha:
                continue
            if basename is not None and Path(historical).name != basename:
                continue
            if role != "external_population_comparison" and not row.get(
                "in_original_run", False
            ):
                continue
            candidates.append(row)
        if role == "child_full_exit_code" and outer_run_directory:
            candidates = [
                row
                for row in candidates
                if _semantic_path(
                    str(
                        PurePosixPath(
                            str(row.get("absolute_path") or "")
                        ).parent
                    )
                )
                == _semantic_path(outer_run_directory)
            ]
        if role == "outer_acceptance_exit_code" and outer_run_directory:
            candidates = [
                row
                for row in candidates
                if _semantic_path(
                    str(
                        PurePosixPath(
                            str(row.get("absolute_path") or "")
                        ).parent
                    )
                )
                == _semantic_path(outer_run_directory)
            ]
        if role == "current_artifact_preflight":
            generated = root / str(basename)
            if generated.is_file():
                candidates = [
                    {
                        "absolute_path": str(generated),
                        "relative_path": str(basename),
                        "sha256": sha256_file(generated),
                        "size_bytes": generated.stat().st_size,
                        "_generated_current_preflight": True,
                    }
                ]
        if len(candidates) > 1 and role == "original_fitting_command":
            semantically_bound = []
            for row in candidates:
                evidence_candidate = evidence_path_for_candidate(row)
                if evidence_candidate is None:
                    continue
                parsed = parse_evidence_role_file(role, evidence_candidate)
                selection = select_fitting_command_for_artifact(
                    parsed.get("normalized_fields") or {},
                    historical_artifact_path,
                )
                if (
                    parsed.get("status") == "ok"
                    and selection.get("status") == "ok"
                ):
                    semantically_bound.append(row)
            candidates = semantically_bound
        if len(candidates) > 1 and role == "child_final_status":
            artifact_parent = (
                Path(historical_artifact_path).parent
                if historical_artifact_path
                else None
            )
            ranked: List[Tuple[int, Mapping[str, Any]]] = []
            for row in candidates:
                parent = Path(str(row.get("absolute_path") or "")).parent
                try:
                    if artifact_parent is not None:
                        artifact_parent.relative_to(parent)
                    ranked.append((len(parent.parts), row))
                except Exception:
                    continue
            if ranked:
                best_depth = max(depth for depth, _ in ranked)
                candidates = [
                    row for depth, row in ranked if depth == best_depth
                ]
        if len(candidates) > 1 and role == "file_manifest":
            semantically_bound = []
            for row in candidates:
                evidence_candidate = evidence_path_for_candidate(row)
                if evidence_candidate is None:
                    continue
                parsed = parse_evidence_role_file(role, evidence_candidate)
                fields = parsed.get("normalized_fields")
                fields = fields if isinstance(fields, Mapping) else {}
                if (
                    parsed.get("status") == "ok"
                    and fields.get("artifact_sha256")
                    in (None, expected_artifact_sha256)
                ):
                    semantically_bound.append(row)
            candidates = semantically_bound
        if len(candidates) != 1:
            failures.append(
                "required_evidence_role_{}_{}".format(
                    "missing" if not candidates else "ambiguous",
                    role,
                )
            )
            continue
        row = candidates[0]
        historical = str(row.get("absolute_path"))
        row_sha = row.get("sha256")
        if not _is_sha256(row_sha):
            failures.append("{}_inventory_sha256_invalid".format(role))
            continue
        if expected_sha is None:
            spec["expected_sha256"] = row_sha
        resolved_specs[role] = spec
        record = record_from_inventory_row(role, row, spec)
        if record is None:
            continue
        records.append(record)
        if role == "artifact_identity":
            historical_artifact_path = record.get(
                "observed_artifact_path"
            )
        elif role == "outer_full_acceptance":
            outer_run_directory = _semantic_path(
                str(PurePosixPath(historical).parent)
            )
        elif role == "child_final_status":
            child_run_directory = _semantic_path(
                str(PurePosixPath(historical).parent)
            )

    replay_role_required = (
        (specs.get("external_population_comparison") or {}).get(
            "require_exact_population_lists"
        )
        is True
        or (specs.get("replay_summary") or {}).get("required_for_paper")
        is True
    )
    if replay_role_required:
        replay_summary_names = {
            "phase3c_artifact_replay_summary.json",
            "artifact_replay_summary.json",
        }
        replay_comparison_names = {
            "phase3c_artifact_replay_comparison.json",
        }
        replay_candidates = [
            row
            for row in inventory
            if row.get("in_original_run", False)
            and Path(str(row.get("absolute_path") or "")).name
            in replay_summary_names | replay_comparison_names
            and (
                child_run_directory is not None
                and _semantic_path(
                    str(row.get("absolute_path") or "")
                ).startswith(
                    child_run_directory.rstrip("/") + "/replay/"
                )
            )
        ]
        candidate_pool = [
            row
            for row in replay_candidates
            if Path(str(row.get("absolute_path") or "")).name
            in replay_summary_names
        ]
        # A named replay summary is authoritative when present. The exact
        # comparison JSON is a compatibility source only when no summary
        # candidate exists; this avoids selecting an arbitrary first file.
        if not candidate_pool:
            candidate_pool = [
                row
                for row in replay_candidates
                if Path(str(row.get("absolute_path") or "")).name
                in replay_comparison_names
            ]
        if len(candidate_pool) != 1:
            failures.append(
                "replay_summary_supporting_role_{}".format(
                    "missing" if not candidate_pool else "ambiguous"
                )
            )
        else:
            row = candidate_pool[0]
            historical = str(row.get("absolute_path"))
            replay_spec = dict(
                SUPPORTING_AUDIT_ROLE_SPECS["replay_summary"]
            )
            replay_spec.update(
                {
                    "expected_sha256": str(row.get("sha256")),
                    "expected_basename": Path(historical).name,
                    "expected_relative_path": str(
                        row.get("relative_path")
                        or Path(historical).name
                    ),
                    "required_for_paper": True,
                }
            )
            resolved_specs["replay_summary"] = replay_spec
            replay_record = record_from_inventory_row(
                "replay_summary", row, replay_spec
            )
            if replay_record is not None:
                records.append(replay_record)
    command_selection = _bind_selected_fitting_command(records)
    if command_selection.get("status") != "ok":
        failures.extend(command_selection.get("failures") or [])
    validation = validate_exact_role_evidence_chain(
        records,
        role_specs=resolved_specs or specs,
        expected_artifact_sha256=expected_artifact_sha256,
        audit_members=audit_verification.get("members") or [],
        live_original_check=live_original_check,
        require_live_original=require_live_original,
    )
    failures.extend(validation.get("failures") or [])
    return {
        "status": "ok" if not failures else "failed",
        "failures": sorted(set(failures)),
        "resolution_method": EXACT_ROLE_BINDING_METHOD,
        "paper_role_binding_valid": not failures,
        "resolved_role_specs": dict(
            validation.get("resolved_role_specs") or {}
        ),
        "evidence_files": records,
        "selected_fitting_command": command_selection,
        "validation": validation,
    }


__all__ = [
    "ACCEPTED_ARTIFACT_IDENTITY_SIDECAR_SHA256",
    "ACCEPTED_AUDIT_ROLE_SPECS",
    "ACCEPTED_CHILD_FULL_EXIT_CODE_SHA256",
    "ACCEPTED_FULL_RUN_EXIT_CODE_SHA256",
    "ACCEPTED_FULL_RUN_FILE_MANIFEST_SHA256",
    "ACCEPTED_GIT_IDENTITY_SIDECAR_SHA256",
    "ACCEPTED_OUTER_ACCEPTANCE_EXIT_CODE_SHA256",
    "ACCEPTED_OUTER_FULL_ACCEPTANCE_SHA256",
    "EXACT_ROLE_BINDING_METHOD",
    "IDENTITY_DOMAIN_AUDIT",
    "IDENTITY_DOMAIN_CHILD",
    "IDENTITY_DOMAIN_OUTER",
    "IDENTITY_DOMAIN_POPULATION",
    "PATH_AUTHORITY_AUDIT_ARCHIVE_MEMBER",
    "PATH_AUTHORITY_CHILD_RUN_UNDER_OUTER",
    "PATH_AUTHORITY_OUTER_RUN_ROOT",
    "PATH_AUTHORITY_POPULATION_RUN_PARENT",
    "REQUIRED_EVIDENCE_ROLES",
    "ROLE_BINDING_SCHEMA_VERSION",
    "build_exact_role_records_from_directory",
    "canonical_resolved_role_specs_from_records",
    "derive_expected_relative_path_from_bound_record",
    "extract_artifact_provenance_identities",
    "normalized_role_observations_sha256",
    "parse_evidence_role_file",
    "resolve_exact_roles_from_verified_audit",
    "select_fitting_command_for_artifact",
    "validate_exact_role_evidence_chain",
]
