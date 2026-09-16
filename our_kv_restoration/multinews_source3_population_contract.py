"""Official LongT5 FREE Multi-News train-calibration population contract.

Structural counterpart of ``cnndm_source6_population_contract`` for the
Multi-News / LongT5 route. It is a SEPARATE evidence type with its own
deterministic-ordering algorithm identity, so a CNN/DailyMail contract can
never be replayed as a Multi-News one (or vice versa) even if every other
field happened to line up.

This module previously targeted a Native FREE fixed-source-layer-6 Multi-
News protocol (``native_free_multinews_source6_population_contract_v1``).
That scientific identity is superseded: the official FREE-released LongT5-
base x Multi-News checkpoint actually executes shallow_deep_kd_dyna / FREE
at its own native configured shallow depth (currently 3, not 6), so this
module's evidence type and required ``fixed_source_layer`` were updated to
match. There is no accepted production source6 Multi-News artifact/contract
requiring backward compatibility. The contract's own ``fixed_source_layer``
field is EVIDENCE BINDING only -- the value the centrally approved official
checkpoint protocol says the collector must have been run with -- never
runtime source-layer selection; the actual runtime authority stays
``config.shallow_exit_layer`` (see ``run_summarization.py`` /
``ExactCacheCalibrationCollector._from_multinews_population_contract``).

Three deliberate differences from the CNN module, all scientific rather than
stylistic:

* The candidate schedule is NOT frozen here. CNN's ``(512, 1024, 2048,
  4096)`` is a CNN-specific centrally approved production value; Multi-News
  documents are far longer and no central approval exists yet. The contract
  therefore CARRIES its own ``candidate_budgets`` and this module only
  validates its shape (non-empty, positive ints, unique, strictly
  increasing, and ``candidate_budget`` a member of it). Nothing here picks a
  production budget.
* The selection seed is likewise contract-carried and validated as a real
  integer rather than pinned to CNN's ``0``. Silently inheriting CNN's seed
  would be an unapproved scientific choice, so the ordering helpers require
  the seed to be passed explicitly -- there is no module-level default.
* The approved fixed source layer is contract-carried (``fixed_source_layer``)
  rather than a module-frozen ``6`` like the legacy Native FREE T5/CNN route.
  Schema validation still requires it to equal this contract version's one
  approved value (currently 3) -- it is not an arbitrary caller-supplied
  layer -- but that approved value is bound here as evidence, not selected
  by any dataset-name or model-class rule anywhere in the runtime.

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

from .missing_kv_dump_provenance import sha256_file
from .missing_kv_paper_population import (
    STABLE_SAMPLE_ID_ALGORITHM_IDENTITY,
    is_placeholder_value,
    stable_sample_ids_sha256,
)
from .cnndm_source6_population_contract import compute_ranking_key as _compute_ranking_key

CONTRACT_SCHEMA_VERSION = 1
CONTRACT_EVIDENCE_TYPE = "native_free_multinews_source3_population_contract_v1"

# The one approved fixed_source_layer value for this contract version --
# evidence binding to the official FREE-released LongT5-base x Multi-News
# checkpoint protocol (shallow_exit_layer=3), never a rule that derives 3
# from the dataset name or model class. A future contract SCHEMA VERSION
# would carry its own value if the approved checkpoint protocol ever
# changes; this module never silently reinterprets a different layer.
REQUIRED_FIXED_SOURCE_LAYER = 3

REQUIRED_DATASET_NAME = "multi_news"
# The actual runtime identity: scripts/run_sum_multinews.sh passes
# --dataset_name multi_news and NO --dataset_config_name, so
# data_args.dataset_config_name is None and run_summarization.py calls
# load_dataset("multi_news", None, ...). That None also flows into
# stable_sample_id, so the contract must bind the same semantic value rather
# than an invented "default" string.
REQUIRED_DATASET_CONFIG_NAME = None
REQUIRED_SPLIT = "train"
REQUIRED_TEXT_COLUMN = "document"
REQUIRED_SUMMARY_COLUMN = "summary"
REQUIRED_SOURCE_PREFIX = "summarize: "
REQUIRED_MAX_SOURCE_LENGTH = 2048
REQUIRED_MAX_TARGET_LENGTH = 512

# Dataset-specific ordering identity. Deliberately NOT the CNN string: the
# ranking is domain-separated by this identity, so the same stable_sample_id
# ranks differently under the two datasets' schemes.
SELECTION_ALGORITHM = "multinews_train_stable_sample_seeded_sha256_order_v1"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class PopulationContractError(ValueError):
    """Raised for any Multi-News population-contract validation failure."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PopulationContractError(reason)


def _require_sha256_hex(value: Any, field_name: str) -> str:
    _require(isinstance(value, str) and bool(_SHA256_HEX_RE.fullmatch(value)), "{}_invalid".format(field_name))
    return value


def _require_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compute_ranking_key(stable_sample_id: str, *, seed: int, algorithm: str = SELECTION_ALGORITHM) -> str:
    """Deterministic sha256 ranking key for the frozen Multi-News train ordering.

    Domain-separated by algorithm identity + seed, and computed from nothing
    but ``(algorithm, seed, stable_sample_id)`` -- no model output, ROUGE,
    confidence, restoration quality, or validation result can ever enter it.

    ``seed`` is required: there is no approved Multi-News default, and
    defaulting to CNN's seed would be an unapproved scientific choice. The
    key derivation itself is byte-identical to the accepted CNN helper (same
    canonical-JSON SHA-256 over the same three fields), so only the
    domain-separating algorithm identity and the seed differ.
    """

    _require(_require_int(seed), "selection_seed_invalid")
    return _compute_ranking_key(stable_sample_id, algorithm=str(algorithm), seed=int(seed))


def order_stable_sample_ids(
    stable_sample_ids: Sequence[str], *, seed: int, algorithm: str = SELECTION_ALGORITHM
) -> List[str]:
    """Deterministic total order: ranking hash, then stable_sample_id tie-break.

    A candidate budget is the first N ids of this ONE fixed order, so a
    larger pre-registered candidate always contains a smaller one as an exact
    prefix. Callers must never re-derive a separate order per budget.
    """

    ids = [str(value) for value in stable_sample_ids]
    keyed = [(compute_ranking_key(value, seed=seed, algorithm=algorithm), value) for value in ids]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [value for _key, value in keyed]


def validate_candidate_schedule(candidate_budgets: Any, candidate_budget: Any) -> List[int]:
    """Validate a contract-declared Multi-News candidate schedule.

    Shape only -- this never selects or approves a production budget. The
    schedule must be a non-empty list of unique, strictly increasing positive
    integers, and ``candidate_budget`` must be one of its members.
    """

    _require(
        isinstance(candidate_budgets, (list, tuple)) and not isinstance(candidate_budgets, (str, bytes)),
        "population_contract_candidate_budgets_invalid",
    )
    budgets = list(candidate_budgets)
    _require(len(budgets) > 0, "population_contract_candidate_budgets_empty")
    _require(all(_require_int(item) for item in budgets), "population_contract_candidate_budgets_invalid")
    values = [int(item) for item in budgets]
    _require(all(item > 0 for item in values), "population_contract_candidate_budgets_non_positive")
    _require(len(set(values)) == len(values), "population_contract_candidate_budgets_duplicate")
    _require(values == sorted(values), "population_contract_candidate_budgets_not_strictly_increasing")
    _require(_require_int(candidate_budget), "population_contract_candidate_budget_invalid")
    _require(int(candidate_budget) in values, "population_contract_candidate_budget_not_in_schedule")
    return values


def select_candidate_budget(
    ordered_stable_sample_ids: Sequence[str], candidate_budget: int, *, candidate_budgets: Sequence[int]
) -> List[str]:
    """First ``candidate_budget`` ids of the single deterministic full order."""

    values = validate_candidate_schedule(candidate_budgets, candidate_budget)
    del values
    ordered = [str(value) for value in ordered_stable_sample_ids]
    _require(len(ordered) >= int(candidate_budget), "insufficient_candidate_population_for_budget")
    return ordered[: int(candidate_budget)]


def load_population_contract(path: "os.PathLike[str] | str", *, expected_contract_sha256: str) -> Dict[str, Any]:
    """Load and externally SHA-validate a Multi-News population contract file.

    The SHA check runs before the payload is parsed for anything else -- the
    contract is not self-authorizing.
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
    """Validate the frozen Multi-News dataset protocol, deterministic-ordering
    identity, population binding, and asset binding of a loaded contract."""

    _require(isinstance(payload, Mapping), "population_contract_not_object")

    _require(payload.get("schema_version") == CONTRACT_SCHEMA_VERSION, "population_contract_schema_version_invalid")
    _require(payload.get("evidence_type") == CONTRACT_EVIDENCE_TYPE, "population_contract_evidence_type_invalid")

    _require(payload.get("dataset_name") == REQUIRED_DATASET_NAME, "population_contract_dataset_name_invalid")
    # Explicitly the semantic None the runtime actually uses -- `in payload`
    # so a missing key is still rejected rather than defaulting to None.
    _require("dataset_config_name" in payload, "population_contract_dataset_config_name_invalid")
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
        _require_int(max_source_length) and int(max_source_length) == REQUIRED_MAX_SOURCE_LENGTH,
        "population_contract_max_source_length_invalid",
    )
    max_target_length = payload.get("max_target_length")
    _require(
        _require_int(max_target_length) and int(max_target_length) == REQUIRED_MAX_TARGET_LENGTH,
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

    _require(
        payload.get("selection_algorithm") == SELECTION_ALGORITHM,
        "population_contract_selection_algorithm_invalid",
    )
    # Contract-carried, not module-frozen: any integer is structurally
    # acceptable, but it must genuinely be an integer and it is what the
    # deterministic ordering is later reproduced with.
    _require(_require_int(payload.get("selection_seed")), "population_contract_selection_seed_invalid")

    # Evidence binding to the official checkpoint's actual configured
    # shallow_exit_layer, NOT runtime source-layer selection -- see the
    # module docstring. Must equal this contract version's one approved
    # value; a syntactically valid but wrong integer (e.g. the legacy T5/CNN
    # source6 value) still fails closed here.
    fixed_source_layer = payload.get("fixed_source_layer")
    _require(
        _require_int(fixed_source_layer) and int(fixed_source_layer) == REQUIRED_FIXED_SOURCE_LAYER,
        "population_contract_fixed_source_layer_invalid",
    )

    candidate_budget = payload.get("candidate_budget")
    validate_candidate_schedule(payload.get("candidate_budgets"), candidate_budget)

    selected_count = payload.get("selected_count")
    _require(
        _require_int(selected_count) and int(selected_count) == int(candidate_budget),
        "population_contract_selected_count_invalid",
    )
    _require_sha256_hex(
        payload.get("selected_stable_sample_set_sha256"), "population_contract_selected_stable_sample_set_sha256"
    )
    _require_sha256_hex(
        payload.get("selected_stable_sample_ordered_sha256"),
        "population_contract_selected_stable_sample_ordered_sha256",
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
    dataset_config_name: Any,
    effective_split: str,
    text_column: str,
    summary_column: str,
    source_prefix: str,
    max_source_length: int,
    max_target_length: int,
    raw_split_dataset_fingerprint: Any,
) -> None:
    """Cross-check the ACTUAL runtime Multi-News dataset protocol against a
    previously schema-validated contract, plus the ACTUAL raw FULL train split
    fingerprint against the contract's recorded ``dataset_fingerprint``.

    ``raw_split_dataset_fingerprint`` must be the fingerprint of the raw full
    train split BEFORE candidate preselection -- the exact snapshot the frozen
    stable-hash population was derived from -- never the selected subset's
    fingerprint. Fails closed on the first mismatch.
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

    normalized_max_source_length = int(max_source_length) if _require_int(max_source_length) else None
    _bind(normalized_max_source_length, REQUIRED_MAX_SOURCE_LENGTH, "max_source_length", "max_source_length")
    normalized_max_target_length = int(max_target_length) if _require_int(max_target_length) else None
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
