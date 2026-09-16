#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import inspect
import json
import logging
import os
import sys
from collections import Counter
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from filelock import FileLock

import datasets
import evaluate
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from sum_lib import (
    ModelArguments,
    DataTrainingArguments,
    SumTrainer,
    adjust_training_args,
)
from models import (
    T5ForConditionalGeneration,
    DeployT5ForConditionalGeneration,
    LongT5ForConditionalGeneration,
    DeployLongT5ForConditionalGeneration,
)
from util import (
    AdditionalArguments,
    additional_args,
    update_autoconfig,
)
from our_kv_restoration.missing_kv_dump_provenance import (
    effective_token_count,
    make_effective_population_summary,
    model_checkpoint_identity,
    preprocessing_identity,
    sha256_file,
    stable_sample_id as make_stable_sample_id,
    text_sha256,
    token_sequence_sha256,
    tokenizer_identity,
    write_json_file,
    write_jsonl,
)
from our_kv_restoration import multinews_source3_population_contract as multinews_population_contract
from our_kv_restoration.multinews_source3_population_contract import (
    validate_runtime_dataset_protocol as validate_multinews_runtime_dataset_protocol,
)
from our_kv_restoration.cnndm_source6_population_contract import (
    CANDIDATE_BUDGETS as CNN_CANDIDATE_BUDGETS,
    load_and_validate_population_contract,
    order_stable_sample_ids as order_cnn_stable_sample_ids,
    select_candidate_budget as select_cnn_candidate_budget,
    validate_model_checkpoint_identity as validate_cnn_model_checkpoint_identity,
    validate_runtime_dataset_protocol as validate_cnn_runtime_dataset_protocol,
    validate_selected_stable_sample_ids as validate_cnn_selected_stable_sample_ids,
    validate_tokenizer_identity as validate_cnn_tokenizer_identity,
)
from our_kv_restoration.f2b_free_running import (
    ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256,
    ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256,
    build_decoding_configuration_identity,
    checkpoint_inventory_identity_from_model_root,
    emit_f2b_runtime_provenance_sidecars,
    resolve_effective_decoding_configuration,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

# Official FREE CALM train-calibration route: the EXPLICIT approved raw
# dataset identities, mapped to their route. This is a paper-facing
# authority gate, so it is an exact allow-list rather than a delegation to
# the generic normalize_dataset_name_for_dispatch() -- that helper
# deliberately maps ANY "*/samsum" name to "samsum" for ordinary dataset
# dispatch, which would let an unapproved mirror (e.g. "foo/samsum") enter
# this route. The generic helper's behavior is intentionally left unchanged
# for its other callers. Case is normalized the same way it is.
OFFICIAL_FREE_CALM_TRAIN_CALIBRATION_DATASET_ROUTES = {
    "cnn_dailymail": "cnn_dailymail",
    "samsum": "samsum",
    "knkarthick/samsum": "samsum",
    "multi_news": "multi_news",
}
# The frozen deterministic train ORDER is domain-separated by its algorithm
# identity (the string is hashed into every ranking key -- see
# cnndm_source6_population_contract.compute_ranking_key), so it also defines
# WHICH rows a candidate budget selects and is reported as the selection
# authority in paper-facing provenance. Reusing the CNN identity for SAMSum
# would therefore both mislabel the SAMSum population and violate that
# domain separation, exactly as the Multi-News contract module already
# refuses to do. Only the identity differs: the seed, the ranking-key
# derivation, the candidate schedule and the selection mechanics are the
# accepted shared ones, reused unchanged.
OFFICIAL_FREE_CALM_SAMSUM_SELECTION_ALGORITHM = "samsum_train_stable_sample_seeded_sha256_order_v1"

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "knkarthick/samsum": ("dialogue", "summary"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def _load_selected_stable_sample_ids(path: str):
    target = Path(path)
    if not target.is_file():
        raise ValueError("selected stable-sample ID file is missing: {}".format(path))
    if target.suffix.lower() == ".jsonl":
        rows = []
        with target.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    value = payload.get("stable_sample_id")
                else:
                    value = payload
                if value in (None, ""):
                    raise ValueError("selected stable-sample ID missing at line {}".format(line_no))
                rows.append(str(value))
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("stable_sample_ids") or payload.get("selected_stable_sample_ids")
        if not isinstance(payload, list):
            raise ValueError("selected stable-sample ID file must contain a JSON array or JSONL rows")
        rows = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                value = item.get("stable_sample_id")
            else:
                value = item
            if value in (None, ""):
                raise ValueError("selected stable-sample ID missing at index {}".format(index))
            rows.append(str(value))
    duplicates = [item for item, count in Counter(rows).items() if count > 1]
    if duplicates:
        raise ValueError("duplicate selected stable-sample ID: {}".format(duplicates[0]))
    return rows


def _load_json_object(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return payload


def resolve_effective_eval_split(data_args, training_args, additional_args):
    """Resolve the effective --do_eval split.

    Default is always "validation" -- byte/semantically unchanged. Only an
    explicit, fully authorized approved train-calibration population-contract
    opt-in may ever route --do_eval at "train" instead, and only when every
    required companion setting is present. update_autoconfig() already
    fail-closed validates the argument COMBINATION whenever
    kv_early_exit_exact_cache_calibration_dataset_split is supplied at all
    -- calibration enabled, fixed_layer, the route's approved
    fixed_source_layer, provenance enabled, contract path+sha set for the
    Native FREE fixed-layer routes, or calibration enabled for the Official
    FREE CALM (official_free_calm_first_crossing) route -- so by the time this runs,
    dataset_split=="train" implies that combination is already
    self-consistent; this function only adds the do_eval-specific and
    selected-ID-file requirements update_autoconfig() cannot see, and those
    apply identically to both routes.
    """

    requested_split = additional_args.kv_early_exit_exact_cache_calibration_dataset_split
    if not requested_split:
        return "validation"
    if requested_split != "train":
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_dataset_split only supports 'train': {!r}".format(
                requested_split
            )
        )
    if not training_args.do_eval:
        raise ValueError("kv_early_exit_exact_cache_calibration_dataset_split=train requires --do_eval")
    if not additional_args.missing_kv_selected_stable_sample_ids_file:
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_dataset_split=train requires "
            "missing_kv_selected_stable_sample_ids_file (the selected stable-ID file is the sole, "
            "authoritative cap on the selected stable-ID train population)"
        )
    if data_args.max_eval_samples is not None:
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_dataset_split=train cannot be combined with "
            "max_eval_samples; the selected stable-ID file is the authoritative cap"
        )
    return "train"


def validate_cnn_population_contract_before_model_construction(
    additional_args, *, model_root, tokenizer_identity_payload
):
    """CNN/DailyMail train-calibration route only: load + externally SHA-
    validate the population contract, validate its dataset protocol and
    selected-ID binding, and validate the model checkpoint + tokenizer
    identity against it -- all before model_cls.from_pretrained() is ever
    called. Fails closed on any mismatch; never falls back to a different
    checkpoint."""

    contract = load_and_validate_population_contract(
        additional_args.kv_early_exit_exact_cache_calibration_population_contract,
        expected_contract_sha256=additional_args.kv_early_exit_exact_cache_calibration_population_contract_sha256,
    )
    selected_ids = _load_selected_stable_sample_ids(additional_args.missing_kv_selected_stable_sample_ids_file)
    validate_cnn_selected_stable_sample_ids(selected_ids, contract)

    checkpoint_identity_payload = model_checkpoint_identity(model_root)
    validate_cnn_model_checkpoint_identity(
        checkpoint_identity_payload["model_checkpoint_identity_sha256"], contract
    )
    if tokenizer_identity_payload is None:
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_dataset_split=train requires "
            "missing_kv_provenance_enabled=True (tokenizer identity must be computed before it can be "
            "validated against the population contract)"
        )
    validate_cnn_tokenizer_identity(tokenizer_identity_payload["tokenizer_identity_sha256"], contract)
    return contract, selected_ids


def validate_multinews_population_contract_before_model_construction(
    additional_args, *, model_root, tokenizer_identity_payload
):
    """Multi-News train-calibration counterpart of
    validate_cnn_population_contract_before_model_construction: same order and
    same fail-closed semantics, using the Multi-News contract validators.
    Never falls back to a different checkpoint or tokenizer."""

    contract = multinews_population_contract.load_and_validate_population_contract(
        additional_args.kv_early_exit_exact_cache_calibration_population_contract,
        expected_contract_sha256=additional_args.kv_early_exit_exact_cache_calibration_population_contract_sha256,
    )
    selected_ids = _load_selected_stable_sample_ids(additional_args.missing_kv_selected_stable_sample_ids_file)
    multinews_population_contract.validate_selected_stable_sample_ids(selected_ids, contract)

    checkpoint_identity_payload = model_checkpoint_identity(model_root)
    multinews_population_contract.validate_model_checkpoint_identity(
        checkpoint_identity_payload["model_checkpoint_identity_sha256"], contract
    )
    if tokenizer_identity_payload is None:
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_dataset_split=train requires "
            "missing_kv_provenance_enabled=True (tokenizer identity must be computed before it can be "
            "validated against the population contract)"
        )
    multinews_population_contract.validate_tokenizer_identity(
        tokenizer_identity_payload["tokenizer_identity_sha256"], contract
    )
    return contract, selected_ids


def official_free_calm_train_calibration_dataset_route(dataset_name):
    """Official FREE CALM train-calibration dataset route, or None when this
    dataset has no approved route.

    Deliberately an EXACT allow-list of approved raw identities
    (``cnn_dailymail``, ``samsum``, ``knkarthick/samsum``, ``multi_news``)
    rather than a
    delegation to normalize_dataset_name_for_dispatch(): that generic helper
    resolves any ``*/samsum`` name to ``samsum``, which is right for ordinary
    dataset dispatch but far too permissive for a paper-facing population
    authority gate -- an unapproved mirror such as ``foo/samsum`` must never
    reach this route. Every unapproved identity returns None and fails closed
    at the caller."""

    if dataset_name is None:
        return None
    return OFFICIAL_FREE_CALM_TRAIN_CALIBRATION_DATASET_ROUTES.get(str(dataset_name).lower())


def validate_official_free_calm_train_population_before_model_construction(
    additional_args, *, dataset_name
):
    """Official FREE CALM train-calibration route only: the authoritative
    fitting population is missing_kv_selected_stable_sample_ids_file itself
    (see ExactCacheCalibrationCollector._from_official_free_calm_config) --
    unlike the Native FREE fixed-source-6 route, no population contract file
    is loaded here, and no checkpoint/tokenizer identity is cross-checked
    against one, because no such Official contract artifact exists. Still
    fails closed before model construction if the dataset has no approved
    Official CALM train-calibration route (currently cnn_dailymail, samsum
    and multi_news), or -- on the CNN/SAMSum routes -- if the selected
    population size is not one of the frozen candidate budgets that
    _preselect_cnn_train_rows() will independently re-verify (after model
    construction) as the exact frozen first-N deterministic order over the
    actual full raw train split. The multi_news route has no module-frozen
    schedule/seed by design (see inline comment below)."""

    dataset_route = official_free_calm_train_calibration_dataset_route(dataset_name)
    if dataset_route is None:
        raise ValueError(
            "kv_early_exit_exact_cache_calibration_source_layer_mode=official_free_calm_first_crossing "
            "train-calibration route is only supported for {}: {!r}".format(
                sorted(OFFICIAL_FREE_CALM_TRAIN_CALIBRATION_DATASET_ROUTES), dataset_name
            )
        )
    selected_ids = _load_selected_stable_sample_ids(additional_args.missing_kv_selected_stable_sample_ids_file)
    if dataset_route == "multi_news":
        # Multi-News deliberately has NO module-frozen candidate schedule
        # (the Native contract module keeps its seed/schedule in the contract
        # file, and the Official Multi-News calibration budget/seed will be
        # frozen in the experiment contract before GPU collection) -- so the
        # CNN (512/1024/2048/4096) schedule must not be silently inherited as
        # a Multi-News authority. The explicitly frozen selected-stable-ID
        # file is the sole population source here; it must simply be
        # non-empty (emptiness would silently disable calibration).
        if not selected_ids:
            raise ValueError(
                "official_free_calm_first_crossing multi_news train-calibration selected "
                "population must not be empty"
            )
    elif len(selected_ids) not in CNN_CANDIDATE_BUDGETS:
        raise ValueError(
            "official_free_calm_first_crossing train-calibration selected population size must be one "
            "of {}: got {}".format(CNN_CANDIDATE_BUDGETS, len(selected_ids))
        )
    return None, selected_ids


def validate_train_population_contract_before_model_construction(
    additional_args, *, dataset_name, model_root, tokenizer_identity_payload
):
    """Route the train-calibration pre-construction gate by source-layer
    mode, then by dataset.

    official_free_calm_first_crossing -> Official FREE CALM route: no
    Native population contract is loaded or consulted; approved datasets are
    cnn_dailymail, samsum, and multi_news (exact allow-list).
    cnn_dailymail (fixed_layer)       -> existing CNN behavior, unchanged.
    multi_news    (fixed_layer)       -> Multi-News behavior, unchanged.
    anything else                     -> fail closed; no dataset is ever
    silently routed through another dataset's (or mode's) contract.
    """

    if (
        getattr(additional_args, "kv_early_exit_exact_cache_calibration_source_layer_mode", None)
        == "official_free_calm_first_crossing"
    ):
        return validate_official_free_calm_train_population_before_model_construction(
            additional_args, dataset_name=dataset_name
        )
    if dataset_name == "cnn_dailymail":
        return validate_cnn_population_contract_before_model_construction(
            additional_args, model_root=model_root, tokenizer_identity_payload=tokenizer_identity_payload
        )
    if dataset_name == "multi_news":
        return validate_multinews_population_contract_before_model_construction(
            additional_args, model_root=model_root, tokenizer_identity_payload=tokenizer_identity_payload
        )
    raise ValueError(
        "kv_early_exit_exact_cache_calibration_dataset_split=train is only supported for "
        "cnn_dailymail and multi_news: {!r}".format(dataset_name)
    )


def _preselect_cnn_train_rows(
    raw_dataset,
    requested_stable_ids,
    *,
    dataset_name,
    dataset_config_name,
    text_column,
    summary_column,
    provenance_dataset_id_column,
    candidate_budget=None,
    expected_order_fn=None,
    dataset_label="CNN/DailyMail",
    selection_algorithm_label="cnndm_train_stable_sample_seeded_sha256_order_v1",
):
    """CNN/DailyMail train-calibration route only: select + reorder the raw
    train split down to exactly the requested stable-ID population BEFORE
    any tokenization. The full train split can be roughly 287k rows;
    tokenizing all of them merely to discover a few thousand centrally
    selected rows would be needlessly expensive, so stable_sample_id is
    instead computed here from cheap text hashing only.

    A temporary column carries the true original raw dataset index through
    ``Dataset.select()`` (which would otherwise renumber rows 0..N-1 within
    the selected subset) so downstream preprocessing can still record the
    real raw_dataset_index rather than a post-selection position. Returns
    ``(selected_dataset, raw_index_column)``; the caller is responsible for
    removing ``raw_index_column`` once preprocessing has consumed it.

    When ``candidate_budget`` is supplied, this reuses this same full-train
    stable-ID scan (never a second scan of the dataset) to also verify that
    ``requested_stable_ids`` is EXACTLY the frozen first-``candidate_budget``
    ids of the approved ``cnndm_train_stable_sample_seeded_sha256_order_v1``
    deterministic order over the actual full train split -- exact ORDERED
    equality, not merely the same set in a different order. Matching the
    contract's own set/order SHA is not sufficient proof the ids were
    actually derived by the frozen ranking rule; this closes that gap.

    ``expected_order_fn``/``dataset_label``/``selection_algorithm_label``
    exist only so the Multi-News route can reuse this one implementation of
    the delicate raw-index-preserving selection. Their defaults reproduce the
    accepted CNN behavior exactly; ``_preselect_multinews_train_rows`` below
    is the only caller that overrides them.
    """

    raw_index_column = "__missing_kv_raw_dataset_index"
    indexed_dataset = raw_dataset.add_column(raw_index_column, list(range(len(raw_dataset))))
    stable_id_column = "__missing_kv_stable_sample_id"

    def _compute_stable_id(examples):
        stable_ids = []
        for i in range(len(examples[text_column])):
            dataset_provided_id = (
                examples[provenance_dataset_id_column][i] if provenance_dataset_id_column is not None else None
            )
            stable_ids.append(
                make_stable_sample_id(
                    dataset_name=dataset_name,
                    dataset_config_name=dataset_config_name,
                    split="train",
                    raw_dataset_index=int(examples[raw_index_column][i]),
                    dataset_provided_id=dataset_provided_id,
                    source_text_sha256=text_sha256(examples[text_column][i]),
                    reference_text_sha256=text_sha256(examples[summary_column][i]),
                )
            )
        examples[stable_id_column] = stable_ids
        return examples

    indexed_dataset = indexed_dataset.map(
        _compute_stable_id,
        batched=True,
        desc="Computing stable_sample_id for {} train row preselection".format(dataset_label),
    )
    all_train_stable_ids = [str(value) for value in indexed_dataset[stable_id_column]]
    stable_to_index = {}
    duplicate_stable_ids = set()
    for row_index, stable_id in enumerate(all_train_stable_ids):
        if stable_id in stable_to_index:
            duplicate_stable_ids.add(stable_id)
        stable_to_index[stable_id] = row_index
    if duplicate_stable_ids:
        raise ValueError(
            "duplicate stable_sample_id in {} train population: {}".format(
                dataset_label, sorted(duplicate_stable_ids)[0]
            )
        )
    if candidate_budget is not None:
        # Reuses all_train_stable_ids from the scan above -- no second pass
        # over the dataset. Exact ordered-list equality: the same set in a
        # different order must fail.
        if expected_order_fn is None:
            expected_selected_ids = select_cnn_candidate_budget(
                order_cnn_stable_sample_ids(all_train_stable_ids), int(candidate_budget)
            )
        else:
            expected_selected_ids = expected_order_fn(all_train_stable_ids, int(candidate_budget))
        if list(requested_stable_ids) != expected_selected_ids:
            raise ValueError(
                "requested {} train stable-ID population does not match the frozen "
                "{} first-{} order over the actual "
                "full train split".format(dataset_label, selection_algorithm_label, int(candidate_budget))
            )
    missing_requested = [stable for stable in requested_stable_ids if stable not in stable_to_index]
    if missing_requested:
        raise ValueError(
            "selected stable_sample_id not found in {} train population: {}".format(
                dataset_label, missing_requested[0]
            )
        )
    selected_dataset = indexed_dataset.select([stable_to_index[stable] for stable in requested_stable_ids])
    actual_ids = [str(value) for value in selected_dataset[stable_id_column]]
    if actual_ids != list(requested_stable_ids):
        raise ValueError(
            "{} train preselection order does not match the requested stable-ID file".format(dataset_label)
        )
    selected_dataset = selected_dataset.remove_columns([stable_id_column])
    return selected_dataset, raw_index_column


def _preselect_multinews_train_rows(
    raw_dataset,
    requested_stable_ids,
    *,
    dataset_name,
    dataset_config_name,
    text_column,
    summary_column,
    provenance_dataset_id_column,
    candidate_budget,
    candidate_budgets,
    selection_seed,
):
    """Multi-News train-calibration route: same raw-index-preserving
    selection as the CNN route (one shared implementation), but the frozen
    first-N verification uses the Multi-News deterministic ordering identity
    and the CONTRACT-declared seed and candidate schedule -- Multi-News has no
    module-frozen production schedule. Also fails closed on any selected row
    whose document/summary is empty, so the approved population can never
    silently shrink once preprocess_function() later drops such rows."""

    def _expected_order(all_train_stable_ids, budget):
        ordered = multinews_population_contract.order_stable_sample_ids(
            all_train_stable_ids, seed=int(selection_seed)
        )
        return multinews_population_contract.select_candidate_budget(
            ordered, int(budget), candidate_budgets=candidate_budgets
        )

    selected_dataset, raw_index_column = _preselect_cnn_train_rows(
        raw_dataset,
        requested_stable_ids,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        text_column=text_column,
        summary_column=summary_column,
        provenance_dataset_id_column=provenance_dataset_id_column,
        candidate_budget=candidate_budget,
        expected_order_fn=_expected_order,
        dataset_label="Multi-News",
        selection_algorithm_label=multinews_population_contract.SELECTION_ALGORITHM,
    )

    _validate_selected_multinews_rows_nonempty(
        selected_dataset,
        raw_index_column,
        text_column=text_column,
        summary_column=summary_column,
    )
    if len(selected_dataset) != len(requested_stable_ids):
        raise ValueError(
            "Multi-News train calibration selected population size does not match "
            "the requested stable-ID population: {} vs {}".format(
                len(selected_dataset), len(requested_stable_ids)
            )
        )

    return selected_dataset, raw_index_column


def _validate_selected_multinews_rows_nonempty(
    selected_dataset, raw_index_column, *, text_column, summary_column
):
    """A row can be correctly selected by the population authority yet still
    vanish silently inside preprocess_function()'s
    `if examples[text_column][i] and examples[summary_column][i]:` guard if
    its document/summary is empty/falsey. Fail closed here -- after exact
    selection but strictly before any tokenization -- so the externally
    approved population can never shrink without an explicit error, and never
    gets a replacement sample. Extracted VERBATIM from the accepted Native
    `_preselect_multinews_train_rows()` protection (same error strings, same
    true raw_dataset_index) so the Official CALM multi_news route shares the
    single implementation rather than duplicating it."""

    selected_texts = selected_dataset[text_column]
    selected_summaries = selected_dataset[summary_column]
    selected_raw_indices = selected_dataset[raw_index_column]
    for source_text, reference_text, raw_index in zip(selected_texts, selected_summaries, selected_raw_indices):
        if not source_text:
            raise ValueError(
                "selected Multi-News train calibration row has empty source text "
                "at raw_dataset_index={}".format(int(raw_index))
            )
        if not reference_text:
            raise ValueError(
                "selected Multi-News train calibration row has empty summary text "
                "at raw_dataset_index={}".format(int(raw_index))
            )


def _preselect_official_free_calm_train_rows(
    raw_dataset,
    requested_stable_ids,
    *,
    dataset_name,
    dataset_config_name,
    text_column,
    summary_column,
    provenance_dataset_id_column,
    dataset_route=None,
):
    """Official FREE CALM train-calibration route: reuses
    _preselect_cnn_train_rows()'s raw-index-preserving selection and the
    shared deterministic stable-hash ordering machinery unchanged -- no
    second deterministic-order checker, no per-dataset selection
    implementation. Unlike the Native fixed-source-6 route, there is no
    population contract file to read a pre-approved candidate_budget from.
    On the CNN/SAMSum routes the budget is simply the size of the externally
    selected population itself, and _preselect_cnn_train_rows() already
    independently re-verifies both that this size is one of the frozen
    candidate budgets (512/1024/2048/4096) and that the requested ids are
    the EXACT first-N of that budget's frozen order over the actual full raw
    train split -- same set wrong order, unsupported size, missing id, and
    duplicate id all fail closed exactly as for the Native route. The
    multi_news route skips only the frozen first-N re-derivation (no frozen
    Official Multi-News seed/schedule exists yet -- see its branch below).

    ``dataset_route`` selects only the ORDERING IDENTITY (the domain string
    hashed into every ranking key), never the mechanics: cnn_dailymail keeps
    the accepted CNN identity byte-for-byte, samsum uses its own identity so
    the SAMSum population is not defined -- or labeled -- by a CNN algorithm.
    When omitted it is derived from ``dataset_name`` through the same
    approved allow-list the pre-construction gate uses, so this paper-facing
    helper fails closed on an unapproved dataset even if it were ever called
    directly rather than through that gate. The accepted historical CNN call
    (``dataset_name="cnn_dailymail"`` with no route) is unchanged: it derives
    "cnn_dailymail" and takes the identical default path.
    """

    if dataset_route is None:
        dataset_route = official_free_calm_train_calibration_dataset_route(dataset_name)
    if dataset_route is None:
        raise ValueError(
            "official_free_calm_first_crossing train preselection has no approved route for "
            "dataset {!r}".format(dataset_name)
        )

    candidate_budget = len(requested_stable_ids)
    order_kwargs = {}
    if dataset_route == "samsum":
        def _expected_order(all_train_stable_ids, budget):
            return select_cnn_candidate_budget(
                order_cnn_stable_sample_ids(
                    all_train_stable_ids,
                    algorithm=OFFICIAL_FREE_CALM_SAMSUM_SELECTION_ALGORITHM,
                ),
                int(budget),
            )

        order_kwargs = {
            "expected_order_fn": _expected_order,
            "dataset_label": "SAMSum",
            "selection_algorithm_label": OFFICIAL_FREE_CALM_SAMSUM_SELECTION_ALGORITHM,
        }
    elif dataset_route == "multi_news":
        # Official CALM Multi-News: the SAME raw-index-preserving selection
        # machinery, labeled with the Multi-News ordering identity -- never
        # the CNN or SAMSum algorithm. The Multi-News deterministic seeded
        # SHA order (multinews_source3_population_contract.order_stable_
        # sample_ids) takes its seed from a contract, and the Official
        # Multi-News calibration budget/seed are deliberately NOT frozen as
        # module constants in this patch (they will be frozen in the
        # experiment contract before GPU collection), so no frozen first-N
        # re-derivation is possible yet: candidate_budget=None skips exactly
        # that one re-verification. The explicitly frozen selected-stable-ID
        # file remains the sole population authority, and the shared
        # machinery still fail-closes on duplicate ids, missing ids, and any
        # deviation from the file's exact selected order, while preserving
        # true raw_dataset_index through selection.
        candidate_budget = None
        order_kwargs = {
            "dataset_label": "Multi-News",
            "selection_algorithm_label": multinews_population_contract.SELECTION_ALGORITHM,
        }
    elif dataset_route != "cnn_dailymail":
        raise ValueError(
            "official_free_calm_first_crossing train preselection route unsupported: {!r}".format(
                dataset_route
            )
        )

    selected_dataset, raw_index_column = _preselect_cnn_train_rows(
        raw_dataset,
        requested_stable_ids,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        text_column=text_column,
        summary_column=summary_column,
        provenance_dataset_id_column=provenance_dataset_id_column,
        candidate_budget=candidate_budget,
        **order_kwargs,
    )
    if dataset_route == "multi_news":
        # Same shared non-empty protection the Native Multi-News route runs:
        # a selected row whose document/summary would be silently dropped by
        # preprocess_function() must fail closed, never shrink the approved
        # population or admit a replacement row.
        _validate_selected_multinews_rows_nonempty(
            selected_dataset,
            raw_index_column,
            text_column=text_column,
            summary_column=summary_column,
        )
    return selected_dataset, raw_index_column


def _f2b_runtime_method_name(additional_args):
    if not bool(additional_args.kv_runtime_restoration_enabled):
        return "full_reference"
    method = str(additional_args.kv_runtime_restoration_method)
    if method == "direct_shallow_kv_reuse":
        return "direct_shallow_kv_reuse"
    if method == "exit_hidden_target_projection":
        return "calm_exit_hidden_projection"
    if method == "phase3c_kv_final":
        return "phase3c_kv_final"
    return method


def _optional_config_int(config, *names):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _maybe_emit_f2b_runtime_sidecars(
    *,
    model_root,
    model,
    data_args,
    training_args,
    additional_args,
):
    output_fields = (
        additional_args.missing_kv_checkpoint_inventory_output,
        additional_args.missing_kv_decoding_configuration_identity_output,
        additional_args.missing_kv_candidate_policy_identity_output,
        additional_args.missing_kv_runtime_method_identity_output,
    )
    if not any(output_fields):
        return
    runtime_method = _f2b_runtime_method_name(additional_args)
    artifact_path = additional_args.kv_runtime_restoration_artifact if runtime_method == "phase3c_kv_final" else None
    artifact_sha = sha256_file(artifact_path) if artifact_path else None
    decoding_configuration = resolve_effective_decoding_configuration(
        data_args=data_args,
        training_args=training_args,
        model_config=model.config,
        generation_config=getattr(model, "generation_config", None),
        generate_kwargs=None,
        generation_library_name="transformers",
        generation_library_version=getattr(transformers, "__version__", None),
        transformers_version=getattr(transformers, "__version__", None),
        torch_version=getattr(torch, "__version__", None),
    )
    native_free_fixed_runtime = bool(additional_args.use_shallow_deep) and not bool(
        additional_args.use_early_exit
    )
    if native_free_fixed_runtime:
        # Reuse the existing production checkpoint and decoding identity
        # builders without attaching the unrelated multi-source F2b policy
        # identity to a Native FREE fixed-source run.
        if additional_args.missing_kv_checkpoint_inventory_output:
            write_json_file(
                additional_args.missing_kv_checkpoint_inventory_output,
                checkpoint_inventory_identity_from_model_root(model_root),
            )
        if additional_args.missing_kv_decoding_configuration_identity_output:
            write_json_file(
                additional_args.missing_kv_decoding_configuration_identity_output,
                build_decoding_configuration_identity(decoding_configuration),
            )
        return
    emit_f2b_runtime_provenance_sidecars(
        model_root=model_root,
        checkpoint_inventory_output=additional_args.missing_kv_checkpoint_inventory_output,
        decoding_configuration_identity_output=additional_args.missing_kv_decoding_configuration_identity_output,
        candidate_policy_identity_output=additional_args.missing_kv_candidate_policy_identity_output,
        runtime_method_identity_output=additional_args.missing_kv_runtime_method_identity_output,
        runtime_method=runtime_method,
        restoration_method="none"
        if runtime_method == "full_reference"
        else str(additional_args.kv_runtime_restoration_method),
        calm_early_exit_enabled=bool(additional_args.use_early_exit),
        kv_restoration_enabled=bool(additional_args.kv_runtime_restoration_enabled),
        decoding_configuration=decoding_configuration,
        artifact_path=artifact_path,
        artifact_file_sha256=artifact_sha,
        runtime_policy_sha256=ACCEPTED_PHASE3C_RUNTIME_POLICY_SHA256 if runtime_method == "phase3c_kv_final" else None,
        hybrid_fitting_policy_sha256=ACCEPTED_PHASE3C_HYBRID_FITTING_POLICY_SHA256 if runtime_method == "phase3c_kv_final" else None,
        threshold=additional_args.kv_runtime_restoration_threshold
        if additional_args.kv_runtime_restoration_threshold is not None
        else additional_args.exit_conf_threshold,
        threshold_comparator="strict_gt",
        adaptive_threshold=bool(additional_args.use_adapt_threshold),
    )


def _dataset_auth_kwargs(use_auth_token):
    """Auth kwargs for datasets.load_dataset across installed API versions.

    No-auth (the ordinary paper-experiment path): return {} so NO
    authentication keyword is passed at all. Modern `datasets` (>=3) removed
    `use_auth_token` from load_dataset's signature, so the historical
    `use_auth_token=None` kwarg silently fell through into **config_kwargs
    and mutated the builder-config hash -- which is exactly how an existing
    offline Multi-News cache (config 'default') stopped resolving
    ('default-31a8eb8e26a3d2e8'). Never passing the keyword restores the
    plain cache identity without touching dataset semantics
    (dataset_name/dataset_config_name are forwarded unchanged by callers).

    Explicit auth opt-in: pick the keyword the INSTALLED load_dataset
    actually declares -- `token=True` on the modern API, `use_auth_token=True`
    on the legacy one -- preserving private-dataset backward compatibility.
    Inspected at call time on this module's `load_dataset` global (a plain
    signature check, not a compatibility framework).
    """

    if not use_auth_token:
        return {}
    if "token" in inspect.signature(load_dataset).parameters:
        return {"token": True}
    return {"use_auth_token": True}


def normalize_dataset_name_for_dispatch(name):
    if name is None:
        return None
    lower = name.lower()
    if lower in {"samsum", "knkarthick/samsum"} or lower.endswith("/samsum"):
        return "samsum"
    return lower


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main(model_args, data_args, training_args, additional_args, model_cls, trainer_cls, jupyter=False):
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        # dataset_name/dataset_config_name are forwarded UNCHANGED (for the
        # authoritative Multi-News runtime that is dataset_config_name=None,
        # never the string "default" -- see
        # our_kv_restoration/multinews_source3_population_contract.py); only
        # the authentication keyword is version-adapted by
        # _dataset_auth_kwargs(), and it is omitted entirely without auth.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            **_dataset_auth_kwargs(model_args.use_auth_token),
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **_dataset_auth_kwargs(model_args.use_auth_token),
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if additional_args.missing_kv_provenance_enabled:
        provenance_dir = os.path.join(training_args.output_dir, "missing_kv_provenance")
        if not additional_args.missing_kv_effective_population_output:
            additional_args.missing_kv_effective_population_output = os.path.join(
                provenance_dir, "effective_eval_population.jsonl"
            )
        if not additional_args.missing_kv_effective_population_summary_output:
            additional_args.missing_kv_effective_population_summary_output = os.path.join(
                provenance_dir, "effective_eval_population_summary.json"
            )
        if not additional_args.missing_kv_generation_binding_output:
            additional_args.missing_kv_generation_binding_output = os.path.join(
                provenance_dir, "generation_sample_binding.jsonl"
            )
        if not additional_args.missing_kv_generation_binding_summary_output:
            additional_args.missing_kv_generation_binding_summary_output = os.path.join(
                provenance_dir, "generation_sample_binding_summary.json"
            )
        if not additional_args.missing_kv_tokenizer_identity_output:
            additional_args.missing_kv_tokenizer_identity_output = os.path.join(
                provenance_dir, "tokenizer_identity.json"
            )
    if additional_args.kv_calm_counterfactual_trace_enabled:
        if int(training_args.per_device_eval_batch_size) != 1:
            raise ValueError("kv_calm_counterfactual_trace_enabled requires per_device_eval_batch_size == 1")
        if training_args.local_rank not in (-1, 0):
            raise ValueError("kv_calm_counterfactual_trace_enabled supports single-process evaluation only")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if not additional_args.use_lora or training_args.do_train:
        config_name = model_args.config_name if model_args.config_name else model_args.model_name_or_path
        tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        model_name = model_args.model_name_or_path
    else:
        lora_config = LoraConfig.from_pretrained(model_args.model_name_or_path)
        config_name = tokenizer_name = model_name = lora_config.base_model_name_or_path

    config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config = update_autoconfig(
        config,
        additional_args,
        max_answer_length=data_args.max_target_length,
        # Validation-only route marker (see update_autoconfig's fixed_layer
        # branch): picks which fixed-layer contract's checks apply, never
        # used to compute/assign a source layer.
        dataset_name=data_args.dataset_name,
    )
    effective_eval_split = resolve_effective_eval_split(data_args, training_args, additional_args)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer_identity_payload = None
    if additional_args.missing_kv_provenance_enabled:
        tokenizer_root = tokenizer_name if tokenizer_name and Path(tokenizer_name).is_dir() else None
        tokenizer_identity_payload = tokenizer_identity(
            tokenizer,
            requested_identifier=tokenizer_name,
            tokenizer_root=tokenizer_root,
        )
        write_json_file(
            additional_args.missing_kv_tokenizer_identity_output,
            tokenizer_identity_payload,
        )

    train_population_contract = None
    cnn_population_contract = None
    if effective_eval_split == "train":
        # Fail closed before model construction if the train-calibration
        # population contract, checkpoint identity, or tokenizer identity do
        # not match. The collector re-resolves and re-validates the same
        # contract independently when it attaches (see
        # ExactCacheCalibrationCollector.from_config's population-contract
        # branch); the contract returned here is kept only to cross-check the
        # actual runtime dataset protocol and frozen selection order once the
        # raw dataset/columns/prefix are resolved below.
        train_population_contract, _selected_ids_at_construction = (
            validate_train_population_contract_before_model_construction(
                additional_args,
                dataset_name=data_args.dataset_name,
                model_root=model_name,
                tokenizer_identity_payload=tokenizer_identity_payload,
            )
        )
        # Preserved name for the unchanged CNN route.
        if data_args.dataset_name == "cnn_dailymail":
            cnn_population_contract = train_population_contract
        # Explicit internal marker so the collector picks the correct
        # contract validator instead of guessing the dataset from the
        # checkpoint path or the model class. The contract independently
        # re-validates the real dataset identity, so this only selects a
        # validator. Set before model construction, hence before the
        # collector is built inside the decoder stack.
        setattr(
            config,
            "kv_early_exit_exact_cache_calibration_population_contract_dataset",
            data_args.dataset_name,
        )

    model = model_cls.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
        
    if additional_args.use_lora:
        if training_args.do_train:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, r=additional_args.lora_rank, 
                lora_alpha=additional_args.lora_alpha, lora_dropout=additional_args.lora_dropout,
                target_modules=additional_args.lora_target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, config=lora_config)
            model = model.merge_and_unload()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if effective_eval_split not in raw_datasets:
            raise ValueError("--do_eval requires a {} dataset".format(effective_eval_split))
        column_names = raw_datasets[effective_eval_split].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dispatch_dataset_name = normalize_dataset_name_for_dispatch(data_args.dataset_name)
    dataset_columns = summarization_name_mapping.get(
        data_args.dataset_name,
        summarization_name_mapping.get(dispatch_dataset_name, None),
    )
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    provenance_dataset_id_column = None
    for candidate_id_column in ("id", "sample_id", "dialogue_id", "guid"):
        if candidate_id_column in column_names:
            provenance_dataset_id_column = candidate_id_column
            break

    def preprocess_function(examples, indices=None, provenance_split=None, raw_index_column=None):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        provenance_entries = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                source_text = examples[text_column][i]
                reference_text = examples[summary_column][i]
                inputs.append(source_text)
                targets.append(reference_text)
                if provenance_split is not None:
                    # A CNN/DailyMail train-calibration preselected dataset
                    # carries the TRUE original raw dataset index in
                    # raw_index_column -- Dataset.select() renumbers rows
                    # 0..N-1 within the selected subset, so `indices` alone
                    # would silently record the wrong (post-selection)
                    # position instead of the real raw_dataset_index.
                    if raw_index_column is not None and raw_index_column in examples:
                        raw_index = int(examples[raw_index_column][i])
                    else:
                        raw_index = int(indices[i]) if indices is not None else i
                    dataset_provided_id = (
                        examples[provenance_dataset_id_column][i]
                        if provenance_dataset_id_column is not None
                        else None
                    )
                    source_digest = text_sha256(source_text)
                    reference_digest = text_sha256(reference_text)
                    provenance_entries.append(
                        {
                            "manifest_schema_version": 1,
                            "stable_sample_id": make_stable_sample_id(
                                dataset_name=data_args.dataset_name,
                                dataset_config_name=data_args.dataset_config_name,
                                split=provenance_split,
                                raw_dataset_index=raw_index,
                                dataset_provided_id=dataset_provided_id,
                                source_text_sha256=source_digest,
                                reference_text_sha256=reference_digest,
                            ),
                            "raw_dataset_index": raw_index,
                            "dataset_provided_id": None
                            if dataset_provided_id is None
                            else str(dataset_provided_id),
                            "source_text_sha256": source_digest,
                            "reference_text_sha256": reference_digest,
                        }
                    )

        inputs = [prefix + inp for inp in inputs]
        max_target_length = data_args.max_target_length
        # Encoder-Decoder Language Models

        padding = "max_length" if data_args.pad_to_max_length else False

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        if provenance_split is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            for entry, input_ids, label_ids in zip(provenance_entries, model_inputs["input_ids"], labels["input_ids"]):
                entry["tokenized_input_sha256"] = token_sequence_sha256(input_ids)
                entry["tokenized_label_sha256"] = token_sequence_sha256(label_ids)
                entry["input_token_count"] = effective_token_count(input_ids, pad_token_id)
                entry["label_token_count"] = effective_token_count(label_ids, pad_token_id)
            for key in [
                "manifest_schema_version",
                "stable_sample_id",
                "raw_dataset_index",
                "dataset_provided_id",
                "source_text_sha256",
                "reference_text_sha256",
                "tokenized_input_sha256",
                "tokenized_label_sha256",
                "input_token_count",
                "label_token_count",
            ]:
                model_inputs["missing_kv_{}".format(key)] = [entry[key] for entry in provenance_entries]
        return model_inputs

    def preprocess_eval_split_with_missing_kv_provenance(examples, indices):
        return preprocess_function(
            examples,
            indices=indices,
            provenance_split=effective_eval_split,
            raw_index_column=eval_raw_index_column,
        )
        
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    eval_population_records = None
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_raw_index_column = None
        eval_dataset = raw_datasets[effective_eval_split]
        raw_eval_row_count = len(eval_dataset)
        raw_split_dataset_fingerprint = getattr(eval_dataset, "_fingerprint", None)
        # raw_validation_dataset_fingerprint is validation-only by name: for
        # the train-calibration route it stays None rather than being
        # populated with a train fingerprint under a validation-labeled
        # field. raw_split_dataset_fingerprint (above) is the split-agnostic
        # equivalent used regardless of effective_eval_split.
        raw_validation_dataset_fingerprint = raw_split_dataset_fingerprint if effective_eval_split == "validation" else None
        selected_stable_sample_ids_file = additional_args.missing_kv_selected_stable_sample_ids_file
        if selected_stable_sample_ids_file and not additional_args.missing_kv_provenance_enabled:
            raise ValueError("missing_kv_selected_stable_sample_ids_file requires missing_kv_provenance_enabled")
        if selected_stable_sample_ids_file and data_args.max_eval_samples is not None:
            raise ValueError(
                "missing_kv_selected_stable_sample_ids_file cannot be combined with max_eval_samples; "
                "the selected stable-ID file is the authoritative cap"
            )
        if effective_eval_split == "train":
            # Train-calibration route: cross-check the ACTUAL runtime dataset
            # protocol (name/config/split/columns/prefix/lengths) and the
            # ACTUAL raw FULL train split fingerprint against the population
            # contract validated before model construction -- now that
            # text_column/summary_column/prefix and the raw split fingerprint
            # are all resolved, but still strictly before any tokenization/
            # generation/collection. Then select + reorder to the requested
            # population before any tokenization, instead of the
            # max_eval_samples cap below (which resolve_effective_eval_split()
            # already guarantees is unset on this route). candidate_budget is
            # forwarded so the requested stable-ID file is validated as the
            # exact frozen first-N stable-hash order over the actual full
            # train split, not merely a matching set/order SHA.
            requested_train_stable_ids = _load_selected_stable_sample_ids(selected_stable_sample_ids_file)
            if additional_args.kv_early_exit_exact_cache_calibration_source_layer_mode == "official_free_calm_first_crossing":
                # Official FREE CALM train-calibration route (cnn_dailymail,
                # samsum or multi_news): there is no Native population
                # contract to
                # cross-check the runtime dataset protocol against
                # (train_population_contract is None on this route -- see
                # validate_official_free_calm_train_population_before_model_
                # construction), so validate_cnn_runtime_dataset_protocol is
                # never called here. _preselect_official_free_calm_train_rows
                # reuses _preselect_cnn_train_rows() directly; the normalized
                # route only selects the frozen ordering identity. The
                # pre-construction validator above already rejected any
                # dataset without an approved Official CALM route.
                eval_dataset, eval_raw_index_column = _preselect_official_free_calm_train_rows(
                    eval_dataset,
                    requested_train_stable_ids,
                    dataset_name=data_args.dataset_name,
                    dataset_config_name=data_args.dataset_config_name,
                    text_column=text_column,
                    summary_column=summary_column,
                    provenance_dataset_id_column=provenance_dataset_id_column,
                    dataset_route=official_free_calm_train_calibration_dataset_route(
                        data_args.dataset_name
                    ),
                )
            elif data_args.dataset_name == "cnn_dailymail":
                validate_cnn_runtime_dataset_protocol(
                    train_population_contract,
                    dataset_name=data_args.dataset_name,
                    dataset_config_name=data_args.dataset_config_name,
                    effective_split=effective_eval_split,
                    text_column=text_column,
                    summary_column=summary_column,
                    source_prefix=prefix,
                    max_source_length=data_args.max_source_length,
                    max_target_length=data_args.val_max_target_length,
                    raw_split_dataset_fingerprint=raw_split_dataset_fingerprint,
                )
                eval_dataset, eval_raw_index_column = _preselect_cnn_train_rows(
                    eval_dataset,
                    requested_train_stable_ids,
                    dataset_name=data_args.dataset_name,
                    dataset_config_name=data_args.dataset_config_name,
                    text_column=text_column,
                    summary_column=summary_column,
                    provenance_dataset_id_column=provenance_dataset_id_column,
                    candidate_budget=int(train_population_contract["candidate_budget"]),
                )
            elif data_args.dataset_name == "multi_news":
                validate_multinews_runtime_dataset_protocol(
                    train_population_contract,
                    dataset_name=data_args.dataset_name,
                    dataset_config_name=data_args.dataset_config_name,
                    effective_split=effective_eval_split,
                    text_column=text_column,
                    summary_column=summary_column,
                    source_prefix=prefix,
                    max_source_length=data_args.max_source_length,
                    max_target_length=data_args.val_max_target_length,
                    raw_split_dataset_fingerprint=raw_split_dataset_fingerprint,
                )
                eval_dataset, eval_raw_index_column = _preselect_multinews_train_rows(
                    eval_dataset,
                    requested_train_stable_ids,
                    dataset_name=data_args.dataset_name,
                    dataset_config_name=data_args.dataset_config_name,
                    text_column=text_column,
                    summary_column=summary_column,
                    provenance_dataset_id_column=provenance_dataset_id_column,
                    candidate_budget=int(train_population_contract["candidate_budget"]),
                    candidate_budgets=list(train_population_contract["candidate_budgets"]),
                    selection_seed=int(train_population_contract["selection_seed"]),
                )
            else:
                raise ValueError(
                    "kv_early_exit_exact_cache_calibration_dataset_split=train is only supported for "
                    "cnn_dailymail and multi_news: {!r}".format(data_args.dataset_name)
                )
        elif data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        post_cap_eval_row_count = len(eval_dataset)
        post_cap_dataset_fingerprint = getattr(eval_dataset, "_fingerprint", None)
        missing_kv_preprocessing_identity_sha256 = None
        missing_kv_preprocessing_identity_payload = None
        with training_args.main_process_first(desc="{} dataset map pre-processing".format(effective_eval_split)):
            if additional_args.missing_kv_provenance_enabled:
                training_args.remove_unused_columns = False
                missing_kv_preprocessing_identity_payload = {
                    "raw_validation_dataset_fingerprint": raw_validation_dataset_fingerprint,
                    "raw_split_dataset_fingerprint": raw_split_dataset_fingerprint,
                    "post_cap_dataset_fingerprint": post_cap_dataset_fingerprint,
                    "dataset_name": data_args.dataset_name,
                    "dataset_config_name": data_args.dataset_config_name,
                    "split": effective_eval_split,
                    "text_column": text_column,
                    "summary_column": summary_column,
                    "source_prefix": prefix,
                    "max_source_length": int(data_args.max_source_length),
                    "max_target_length": int(data_args.val_max_target_length),
                    "padding": "max_length" if data_args.pad_to_max_length else False,
                    "ignore_pad_token_for_loss": bool(data_args.ignore_pad_token_for_loss),
                    "tokenizer_identity_sha256": None
                    if tokenizer_identity_payload is None
                    else tokenizer_identity_payload.get("tokenizer_identity_sha256"),
                }
                missing_kv_preprocessing_identity_sha256 = preprocessing_identity(
                    missing_kv_preprocessing_identity_payload
                )
                map_function = preprocess_eval_split_with_missing_kv_provenance
                map_kwargs = {"with_indices": True}
                map_kwargs["new_fingerprint"] = "missing_kv_preproc_{}".format(
                    missing_kv_preprocessing_identity_sha256[:32]
                )
            else:
                map_function = preprocess_function
                map_kwargs = {}
            map_remove_columns = column_names if eval_raw_index_column is None else column_names + [eval_raw_index_column]
            eval_dataset = eval_dataset.map(
                map_function,
                batched=True,
                num_proc=None if additional_args.missing_kv_provenance_enabled else data_args.preprocessing_num_workers,
                remove_columns=map_remove_columns,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on {} dataset".format(effective_eval_split),
                **map_kwargs,
            )
        post_preprocessing_dataset_fingerprint = getattr(eval_dataset, "_fingerprint", None)
        if additional_args.missing_kv_provenance_enabled:
            # The CNN/DailyMail train-calibration route already selected and
            # reordered to the exact requested population before
            # tokenization above; redoing the same reselection post-
            # tokenization here would be redundant (this block's own
            # existing SAMSum/validation behavior is otherwise preserved
            # byte-for-byte).
            if selected_stable_sample_ids_file and effective_eval_split != "train":
                requested_stable_ids = _load_selected_stable_sample_ids(selected_stable_sample_ids_file)
                stable_to_index = {}
                duplicate_stable_ids = set()
                for row_index, row in enumerate(eval_dataset):
                    stable_id = str(row["missing_kv_stable_sample_id"])
                    if stable_id in stable_to_index:
                        duplicate_stable_ids.add(stable_id)
                    stable_to_index[stable_id] = row_index
                if duplicate_stable_ids:
                    raise ValueError(
                        "duplicate stable_sample_id in validation population: {}".format(
                            sorted(duplicate_stable_ids)[0]
                        )
                    )
                missing_requested = [stable for stable in requested_stable_ids if stable not in stable_to_index]
                if missing_requested:
                    raise ValueError("selected stable_sample_id not found in validation population: {}".format(missing_requested[0]))
                eval_dataset = eval_dataset.select([stable_to_index[stable] for stable in requested_stable_ids])
                post_preprocessing_dataset_fingerprint = getattr(eval_dataset, "_fingerprint", None)
            selected_orders = list(range(len(eval_dataset)))
            eval_dataset = eval_dataset.add_column("missing_kv_selected_order", selected_orders)
            provenance_columns = [
                "missing_kv_manifest_schema_version",
                "missing_kv_stable_sample_id",
                "missing_kv_raw_dataset_index",
                "missing_kv_dataset_provided_id",
                "missing_kv_source_text_sha256",
                "missing_kv_reference_text_sha256",
                "missing_kv_tokenized_input_sha256",
                "missing_kv_tokenized_label_sha256",
                "missing_kv_input_token_count",
                "missing_kv_label_token_count",
            ]
            eval_population_records = []
            for selected_order, row in enumerate(eval_dataset):
                record = {
                    "manifest_schema_version": int(row["missing_kv_manifest_schema_version"]),
                    "stable_sample_id": row["missing_kv_stable_sample_id"],
                    "raw_dataset_index": int(row["missing_kv_raw_dataset_index"]),
                    "dataset_provided_id": row.get("missing_kv_dataset_provided_id"),
                    "selected_order": int(selected_order),
                    "source_text_sha256": row["missing_kv_source_text_sha256"],
                    "reference_text_sha256": row["missing_kv_reference_text_sha256"],
                    "tokenized_input_sha256": row["missing_kv_tokenized_input_sha256"],
                    "tokenized_label_sha256": row["missing_kv_tokenized_label_sha256"],
                    "input_token_count": int(row["missing_kv_input_token_count"]),
                    "label_token_count": int(row["missing_kv_label_token_count"]),
                    "dataset_name": data_args.dataset_name,
                    "dataset_config_name": data_args.dataset_config_name,
                    "split": effective_eval_split,
                    "text_column": text_column,
                    "summary_column": summary_column,
                    "source_prefix": prefix,
                    "max_source_length": int(data_args.max_source_length),
                    "max_target_length": int(data_args.val_max_target_length),
                    "padding": "max_length" if data_args.pad_to_max_length else False,
                    "ignore_pad_token_for_loss": bool(data_args.ignore_pad_token_for_loss),
                }
                eval_population_records.append(record)
            population_summary = make_effective_population_summary(
                eval_population_records,
                raw_split_row_count=raw_eval_row_count,
                post_cap_row_count=post_cap_eval_row_count,
                dataset_fingerprint=post_preprocessing_dataset_fingerprint,
                raw_validation_dataset_fingerprint=raw_validation_dataset_fingerprint,
                post_cap_dataset_fingerprint=post_cap_dataset_fingerprint,
                post_preprocessing_dataset_fingerprint=post_preprocessing_dataset_fingerprint,
                dataset_metadata={
                    "dataset_name": data_args.dataset_name,
                    "dataset_config_name": data_args.dataset_config_name,
                    "split": effective_eval_split,
                    "text_column": text_column,
                    "summary_column": summary_column,
                    "source_prefix": prefix,
                    "max_source_length": data_args.max_source_length,
                    "max_target_length": data_args.val_max_target_length,
                    "padding": "max_length" if data_args.pad_to_max_length else False,
                    "ignore_pad_token_for_loss": bool(data_args.ignore_pad_token_for_loss),
                    "tokenizer_identity_sha256": None
                    if tokenizer_identity_payload is None
                    else tokenizer_identity_payload.get("tokenizer_identity_sha256"),
                    "preprocessing_identity_sha256": missing_kv_preprocessing_identity_sha256,
                    "preprocessing_identity_payload": missing_kv_preprocessing_identity_payload,
                },
            )
            population_summary["preprocessing_identity_sha256"] = missing_kv_preprocessing_identity_sha256
            f2b_population_identity_file = additional_args.missing_kv_f2b_population_identity_file
            if f2b_population_identity_file:
                f2b_population_identity = _load_json_object(f2b_population_identity_file)
                if f2b_population_identity.get("status") != "ok":
                    raise ValueError("F2b population identity status is not ok")
                if f2b_population_identity.get("schema_version") != 1:
                    raise ValueError("F2b population identity schema version mismatch")
                if f2b_population_identity.get("population_mode") != "artifact_heldout":
                    raise ValueError("F2b population identity mode mismatch")
                selected_stable_ids = [record["stable_sample_id"] for record in eval_population_records]
                expected_selected_ids = f2b_population_identity.get("selected_stable_sample_ids")
                if expected_selected_ids is not None and [str(item) for item in expected_selected_ids] != selected_stable_ids:
                    raise ValueError("F2b selected stable-sample IDs do not match effective evaluation population")
                selected_count = f2b_population_identity.get("selected_population_count")
                if selected_count is not None and int(selected_count) != len(eval_population_records):
                    raise ValueError("F2b selected population count does not match effective evaluation population")
                for key in (
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
                    "source_manifest_immutable_binding_reference",
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
                    "full_heldout_population_count",
                    "selected_population_count",
                    "population_capped",
                    "selection_rule",
                    "selection_source_identity",
                    "selection_source_type",
                    "selection_source_file_sha256",
                    "selection_source_sha256",
                    "selection_source_schema_version",
                    "selection_source_run_identity",
                    "selection_source_artifact_identity",
                    "selection_source_accepted_artifact_binding",
                    "eligible_candidate_sample_count",
                    "candidate_first_crossing_event_count",
                    "heldout_stable_sample_set_sha256",
                    "heldout_stable_sample_ordered_sha256",
                    "selected_stable_sample_set_sha256",
                    "selected_stable_sample_ordered_sha256",
                    "stable_sample_serialization_encoding",
                    "stable_sample_serialization_newline",
                    "stable_sample_serialization_final_newline",
                    "stable_sample_set_sort",
                ):
                    if key in f2b_population_identity:
                        population_summary[key] = f2b_population_identity[key]
            if population_summary["status"] != "ok":
                raise ValueError("missing-KV effective eval population provenance is invalid: {}".format(population_summary))
            write_jsonl(additional_args.missing_kv_effective_population_output, eval_population_records)
            write_json_file(additional_args.missing_kv_effective_population_summary_output, population_summary)
            eval_dataset = eval_dataset.remove_columns([name for name in provenance_columns if name in eval_dataset.column_names])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        except:
            decoded_preds = tokenizer.batch_decode(np.where(preds != -100, preds, tokenizer.pad_token_id), 
                                                   skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
        try:
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except:
            decoded_labels = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), 
                                                    skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )    
    # adjust training arguments
    training_args = adjust_training_args(training_args, additional_args)
    if additional_args.missing_kv_provenance_enabled:
        _maybe_emit_f2b_runtime_sidecars(
            model_root=model_name,
            model=model,
            data_args=data_args,
            training_args=training_args,
            additional_args=additional_args,
        )

    # Initialize our Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )
    if additional_args.missing_kv_provenance_enabled and eval_population_records is not None:
        if hasattr(trainer, "set_missing_kv_effective_population_records"):
            trainer.set_missing_kv_effective_population_records(eval_population_records)
        else:
            trainer.missing_kv_effective_population_records = eval_population_records

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if not additional_args.use_lora: trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if additional_args.use_lora:
            model.save_pretrained(training_args.output_dir)  # save adapter_config.json
            model.base_model.save_pretrained(training_args.output_dir)  # save config.json

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if not jupyter:
        return results
    else:
        return trainer


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()
    
    dispatch_dataset_name = normalize_dataset_name_for_dispatch(data_args.dataset_name)
    if 't5' in model_args.model_name_or_path:
        if dispatch_dataset_name in ["cnn_dailymail", "xsum", "samsum"]:
            model_cls = T5ForConditionalGeneration if not additional_args.deploy_scenario \
                else DeployT5ForConditionalGeneration
        elif dispatch_dataset_name in ["multi_news", "big_patent"]:
            model_cls = LongT5ForConditionalGeneration if not additional_args.deploy_scenario \
                else DeployLongT5ForConditionalGeneration
    else:
        raise NotImplemented

    trainer_cls = SumTrainer

    main(model_args, data_args, training_args, additional_args, model_cls, trainer_cls)
