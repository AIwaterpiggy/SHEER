"""Native FREE fixed-source-layer-6 CNN/DailyMail train-calibration population contract.

Represents and validates the centrally approved CNN/DailyMail train-split
calibration population contract that routes the existing Native FREE
fixed-source-layer-6 exact-cache calibration collector at a deterministic,
externally SHA-approved train population instead of the SAMSum legacy
validation population.

This module is dataset-protocol/population-contract validation only. It
never touches collector tensor logic, Phase 3c fitting mathematics, or the
artifact tensor schema. The contract's bytes are never self-authorizing: a
contract is only trusted because the caller separately supplies an expected
SHA-256 approved out of band, so every entry point here requires that
expected SHA-256 alongside the contract path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .missing_kv_dump_provenance import canonical_json_sha256, sha256_file
from .missing_kv_paper_population import (
    STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
    is_placeholder_value,
    stable_sample_ids_sha256,
)

CONTRACT_SCHEMA_VERSION = 1
CONTRACT_EVIDENCE_TYPE = "native_free_cnndm_source6_population_contract_v1"

REQUIRED_DATASET_NAME = "cnn_dailymail"
REQUIRED_DATASET_CONFIG_NAME = "3.0.0"
REQUIRED_SPLIT = "train"
REQUIRED_TEXT_COLUMN = "article"
REQUIRED_SUMMARY_COLUMN = "highlights"
REQUIRED_SOURCE_PREFIX = "summarize: "
REQUIRED_MAX_SOURCE_LENGTH = 512
REQUIRED_MAX_TARGET_LENGTH = 128

# Frozen before observing any model output. No ROUGE, confidence, generation
# quality, Phase 3c quality, validation quality, or held-out MSE may ever
# enter this ranking.
SELECTION_ALGORITHM = "cnndm_train_stable_sample_seeded_sha256_order_v1"
SELECTION_SEED = 0
CANDIDATE_BUDGETS = (512, 1024, 2048, 4096)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class PopulationContractError(ValueError):
    """Raised for any CNN/DailyMail population-contract validation failure."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PopulationContractError(reason)


def _require_sha256_hex(value: Any, field_name: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256_HEX_RE.fullmatch(value)), "{}_invalid".format(field_name))
    return value


def compute_ranking_key(stable_sample_id: str, *, algorithm: str = SELECTION_ALGORITHM, seed: int = SELECTION_SEED) -> str:
    """Deterministic sha256 ranking key for the frozen train ordering.

    Domain-separated by algorithm identity + seed so this ranking can never
    be confused with any other stable-sample ordering scheme, and never
    depends on model output.
    """
    return canonical_json_sha256(
        {
            "algorithm": str(algorithm),
            "seed": int(seed),
            "stable_sample_id": str(stable_sample_id),
        }
    )


def order_stable_sample_ids(
    stable_sample_ids: Sequence[str], *, algorithm: str = SELECTION_ALGORITHM, seed: int = SELECTION_SEED
) -> List[str]:
    """Deterministic total order: ranking hash, then stable_sample_id tie-break.

    A candidate budget is defined as the first N ids of this one fixed
    order, so nesting (candidate_512 subset candidate_1024 subset
    candidate_2048 subset candidate_4096) holds automatically by
    construction -- callers never re-derive a separate order per budget.
    """
    ids = [str(value) for value in stable_sample_ids]
    keyed = [(compute_ranking_key(value, algorithm=algorithm, seed=seed), value) for value in ids]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [value for _key, value in keyed]


def select_candidate_budget(ordered_stable_sample_ids: Sequence[str], candidate_budget: int) -> List[str]:
    _require(int(candidate_budget) in CANDIDATE_BUDGETS, "candidate_budget_unsupported:{}".format(candidate_budget))
    ordered = [str(value) for value in ordered_stable_sample_ids]
    _require(len(ordered) >= int(candidate_budget), "insufficient_candidate_population_for_budget")
    return ordered[: int(candidate_budget)]


def load_population_contract(path: "os.PathLike[str] | str", *, expected_contract_sha256: str) -> Dict[str, Any]:
    """Load and externally SHA-validate a CNN/DailyMail population contract file.

    The SHA check runs before the payload is parsed for anything else --
    the contract is not self-authorizing, so an internal "approved: true"
    field (if one were present) is never trusted on its own.
    """
    target = Path(path)
    _require(target.is_file(), "population_contract_file_missing:{}".format(path))
    _require_sha256_hex(expected_contract_sha256, "expected_contract_sha256")
    actual_sha256 = sha256_file(target)
    _require(actual_sha256 == expected_contract_sha256, "population_contract_sha256_mismatch")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PopulationContractError("population_contract_parse_failed:{}".format(type(exc).__name__)) from exc
    _require(isinstance(payload, Mapping), "population_contract_not_object")
    payload = dict(payload)
    payload["contract_file_sha256"] = actual_sha256
    return payload


def validate_population_contract_schema(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the frozen dataset protocol, deterministic-ordering identity,
    population binding, and asset binding of a loaded contract payload.

    Returns a plain dict copy of the payload on success; raises
    PopulationContractError with a specific reason on the first mismatch.
    """
    _require(isinstance(payload, Mapping), "population_contract_not_object")

    _require(payload.get("schema_version") == CONTRACT_SCHEMA_VERSION, "population_contract_schema_version_invalid")
    _require(payload.get("evidence_type") == CONTRACT_EVIDENCE_TYPE, "population_contract_evidence_type_invalid")

    _require(payload.get("dataset_name") == REQUIRED_DATASET_NAME, "population_contract_dataset_name_invalid")
    _require(
        payload.get("dataset_config_name") == REQUIRED_DATASET_CONFIG_NAME,
        "population_contract_dataset_config_name_invalid",
    )
    _require(payload.get("split") == REQUIRED_SPLIT, "population_contract_split_invalid")
    _require(payload.get("text_column") == REQUIRED_TEXT_COLUMN, "population_contract_text_column_invalid")
    _require(payload.get("summary_column") == REQUIRED_SUMMARY_COLUMN, "population_contract_summary_column_invalid")
    _require(payload.get("source_prefix") == REQUIRED_SOURCE_PREFIX, "population_contract_source_prefix_invalid")

    max_source_length = payload.get("max_source_length")
    _require(
        isinstance(max_source_length, int)
        and not isinstance(max_source_length, bool)
        and int(max_source_length) == REQUIRED_MAX_SOURCE_LENGTH,
        "population_contract_max_source_length_invalid",
    )
    max_target_length = payload.get("max_target_length")
    _require(
        isinstance(max_target_length, int)
        and not isinstance(max_target_length, bool)
        and int(max_target_length) == REQUIRED_MAX_TARGET_LENGTH,
        "population_contract_max_target_length_invalid",
    )

    dataset_fingerprint = payload.get("dataset_fingerprint")
    _require(
        isinstance(dataset_fingerprint, str) and not is_placeholder_value(dataset_fingerprint),
        "population_contract_dataset_fingerprint_invalid",
    )

    _require(
        payload.get("stable_sample_id_algorithm_identity") == STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
        "population_contract_stable_sample_id_algorithm_identity_invalid",
    )

    _require(payload.get("selection_algorithm") == SELECTION_ALGORITHM, "population_contract_selection_algorithm_invalid")
    selection_seed = payload.get("selection_seed")
    _require(
        isinstance(selection_seed, int) and not isinstance(selection_seed, bool) and int(selection_seed) == SELECTION_SEED,
        "population_contract_selection_seed_invalid",
    )
    candidate_budgets = payload.get("candidate_budgets")
    _require(
        isinstance(candidate_budgets, list)
        and not isinstance(candidate_budgets, (str, bytes))
        and [int(item) for item in candidate_budgets] == list(CANDIDATE_BUDGETS),
        "population_contract_candidate_budgets_invalid",
    )

    candidate_budget = payload.get("candidate_budget")
    _require(
        isinstance(candidate_budget, int)
        and not isinstance(candidate_budget, bool)
        and int(candidate_budget) in CANDIDATE_BUDGETS,
        "population_contract_candidate_budget_invalid",
    )
    selected_count = payload.get("selected_count")
    _require(
        isinstance(selected_count, int)
        and not isinstance(selected_count, bool)
        and int(selected_count) == int(candidate_budget),
        "population_contract_selected_count_invalid",
    )
    _require_sha256_hex(
        payload.get("selected_stable_sample_set_sha256"), "population_contract_selected_stable_sample_set_sha256"
    )
    _require_sha256_hex(
        payload.get("selected_stable_sample_ordered_sha256"), "population_contract_selected_stable_sample_ordered_sha256"
    )

    _require_sha256_hex(
        payload.get("model_checkpoint_identity_sha256"), "population_contract_model_checkpoint_identity_sha256"
    )
    _require_sha256_hex(payload.get("tokenizer_identity_sha256"), "population_contract_tokenizer_identity_sha256")

    return dict(payload)


def load_and_validate_population_contract(
    path: "os.PathLike[str] | str", *, expected_contract_sha256: str
) -> Dict[str, Any]:
    """Load, externally SHA-validate, and schema-validate a contract in one call."""
    payload = load_population_contract(path, expected_contract_sha256=expected_contract_sha256)
    return validate_population_contract_schema(payload)


def validate_selected_stable_sample_ids(
    selected_stable_sample_ids: Sequence[str], contract: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate a caller-supplied selected stable-ID population against a
    validated contract's own count / set-SHA / ordered-SHA binding."""

    ids = [str(value) for value in selected_stable_sample_ids]
    seen = set()
    duplicates = sorted({value for value in ids if value in seen or seen.add(value)})
    _require(not duplicates, "selected_stable_sample_ids_duplicate:{}".format(duplicates[0] if duplicates else ""))
    expected_count = int(contract["selected_count"])
    _require(
        len(ids) == expected_count, "selected_stable_sample_ids_count_mismatch:{}:{}".format(len(ids), expected_count)
    )
    actual_set_sha256 = stable_sample_ids_sha256(ids, sort_ids=True, expected_count=expected_count)
    _require(
        actual_set_sha256 == contract["selected_stable_sample_set_sha256"],
        "selected_stable_sample_ids_set_sha256_mismatch",
    )
    actual_ordered_sha256 = stable_sample_ids_sha256(ids, sort_ids=False, expected_count=expected_count)
    _require(
        actual_ordered_sha256 == contract["selected_stable_sample_ordered_sha256"],
        "selected_stable_sample_ids_ordered_sha256_mismatch",
    )
    return {
        "selected_count": expected_count,
        "selected_stable_sample_set_sha256": actual_set_sha256,
        "selected_stable_sample_ordered_sha256": actual_ordered_sha256,
    }


def validate_runtime_dataset_protocol(
    contract: Mapping[str, Any],
    *,
    dataset_name: str,
    dataset_config_name: str,
    effective_split: str,
    text_column: str,
    summary_column: str,
    source_prefix: str,
    max_source_length: int,
    max_target_length: int,
    raw_split_dataset_fingerprint: Any,
) -> None:
    """Cross-check the ACTUAL runtime dataset protocol against a previously
    schema-validated contract's own frozen fields (and the frozen
    constants), plus the ACTUAL raw full train split fingerprint against the
    contract's recorded ``dataset_fingerprint``.

    ``validate_population_contract_schema`` only checks that the contract's
    own bytes are internally well-formed; it never compares them against
    what the runtime is actually about to execute. This is that runtime-side
    binding. ``raw_split_dataset_fingerprint`` must be the fingerprint of
    the raw full train split BEFORE candidate preselection -- the exact
    snapshot the frozen stable-hash population was derived from -- not the
    selected subset's fingerprint. Fails closed (raises
    PopulationContractError) on the first mismatch or on either fingerprint
    being missing.
    """
    _require(isinstance(contract, Mapping), "population_contract_not_object")

    def _bind(actual: Any, required: Any, contract_field: str, field_label: str) -> None:
        _require(actual == required, "runtime_{}_invalid".format(field_label))
        _require(contract.get(contract_field) == required, "population_contract_{}_invalid".format(field_label))

    _bind(dataset_name, REQUIRED_DATASET_NAME, "dataset_name", "dataset_name")
    _bind(dataset_config_name, REQUIRED_DATASET_CONFIG_NAME, "dataset_config_name", "dataset_config_name")
    _bind(effective_split, REQUIRED_SPLIT, "split", "split")
    _bind(text_column, REQUIRED_TEXT_COLUMN, "text_column", "text_column")
    _bind(summary_column, REQUIRED_SUMMARY_COLUMN, "summary_column", "summary_column")
    _bind(source_prefix, REQUIRED_SOURCE_PREFIX, "source_prefix", "source_prefix")

    normalized_max_source_length = (
        int(max_source_length)
        if isinstance(max_source_length, int) and not isinstance(max_source_length, bool)
        else None
    )
    _bind(normalized_max_source_length, REQUIRED_MAX_SOURCE_LENGTH, "max_source_length", "max_source_length")
    normalized_max_target_length = (
        int(max_target_length)
        if isinstance(max_target_length, int) and not isinstance(max_target_length, bool)
        else None
    )
    _bind(normalized_max_target_length, REQUIRED_MAX_TARGET_LENGTH, "max_target_length", "max_target_length")

    contract_fingerprint = contract.get("dataset_fingerprint")
    _require(
        isinstance(raw_split_dataset_fingerprint, str) and bool(raw_split_dataset_fingerprint),
        "runtime_raw_split_dataset_fingerprint_missing",
    )
    _require(
        isinstance(contract_fingerprint, str) and not is_placeholder_value(contract_fingerprint),
        "population_contract_dataset_fingerprint_invalid",
    )
    _require(
        raw_split_dataset_fingerprint == contract_fingerprint,
        "runtime_raw_split_dataset_fingerprint_mismatch",
    )


def validate_model_checkpoint_identity(actual_sha256: str, contract: Mapping[str, Any]) -> None:
    _require_sha256_hex(actual_sha256, "model_checkpoint_identity_sha256")
    _require(
        actual_sha256 == contract["model_checkpoint_identity_sha256"],
        "population_contract_model_checkpoint_identity_mismatch",
    )


def validate_tokenizer_identity(actual_sha256: str, contract: Mapping[str, Any]) -> None:
    _require_sha256_hex(actual_sha256, "tokenizer_identity_sha256")
    _require(
        actual_sha256 == contract["tokenizer_identity_sha256"],
        "population_contract_tokenizer_identity_mismatch",
    )
