"""Bounded collector for early-exit-token / exact-cache calibration events.

The collector writes the existing ``packed_generation_v1`` hidden/K/V
representation. It is deliberately limited to committed first-crossing events
from the artifact-free ``exact_catchup`` runtime path.
"""

from __future__ import annotations

import json
import os
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from .missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_POLICY_NAME,
    CALM_THRESHOLD,
    OFFICIAL_FREE_CALM_POLICY_NAME,
    calm_policy_sha256,
    official_free_calm_candidate_layers,
)
from .phase3c_policy_artifact import (
    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
    SOURCE_LAYER_MODE_FIXED,
    SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
)
from .missing_kv_dump_provenance import (
    HIDDEN_LOGICAL_RECORD_TYPE,
    KV_LOGICAL_RECORD_TYPE,
    PACKED_GENERATION_STORAGE_FORMAT,
    append_jsonl,
    canonical_json_sha256,
    populate_packed_record_identities,
    sha256_file,
    write_json_file,
)
from .missing_kv_exact_catchup_overhead import GAP_BIN_LABELS, gap_bin_for_gap
from .missing_kv_paper_population import (
    load_source_manifest_fitting_ids,
    stable_sample_ids_sha256,
)


TRAJECTORY_TYPE = "early_exit_token_exact_cache_free_running"
TRAJECTORY_NAME = "early-exit-token / exact-cache free-running calibration"
SELECTED_TOKEN_SOURCE = "early_exit_source_layer_logits"
COLLECTOR_SCHEMA_VERSION = 1
EVENT_RECORD_TYPE = "early_exit_exact_cache_calibration_target"
SHARD_RECORD_TYPE = "early_exit_exact_cache_calibration_shard"
SUMMARY_EVIDENCE_TYPE = "early_exit_exact_cache_calibration_summary"
_COLLECTOR_CONSTRUCTION_TOKEN = object()

# Native FREE fixed-source-layer collection mode. Distinct from the
# CALM candidate_first_crossing constants above: this trajectory has a single
# frozen source layer and no multi-candidate evaluation chain.
FIXED_LAYER_TRAJECTORY_TYPE = "native_free_fixed_source_layer_exact_cache_free_running"
# The historical Native FREE source6 trajectory name, preserved BYTE-FOR-BYTE.
# It is written straight into calibration_summary.json's "trajectory_name",
# so it is externally observable metadata of the already-accepted source6
# path; adding the LongT5 Multi-News source3 route must not change it. Other
# fixed source layers are named by _fixed_layer_trajectory_name() below.
FIXED_LAYER_TRAJECTORY_NAME = "Native FREE fixed-source-layer-6 / exact-cache free-running calibration"
FIXED_LAYER_RUNTIME_FRAMEWORK = "official_FREE_shallow_deep"
FIXED_LAYER_THRESHOLD_COMPARATOR = "strict_gt"
FIXED_LAYER_REQUIRED_SOURCE_LAYER = 6
FIXED_LAYER_REQUIRED_THRESHOLD = 0.9
FIXED_LAYER_SELECTED_TOKEN_SOURCE = "fixed_shallow_exit_layer_logits"


def _fixed_layer_trajectory_name(source_layer: Any) -> str:
    """Human-readable fixed-layer trajectory name for one source layer.

    Source 6 returns the historical constant unchanged, so the already-
    accepted Native FREE T5/CNN source6 summary metadata stays byte-identical
    to what it emitted before the LongT5 source3 route existed. Any other
    structurally valid fixed source layer (currently the official LongT5
    Multi-News source3 route) is named accurately for that layer.

    Naming only -- this never authorizes a source layer. Production
    authorization stays exactly where it already is: the runtime /
    calibration / contract triple match enforced at construction.
    """

    source_layer = int(source_layer)
    if source_layer == FIXED_LAYER_REQUIRED_SOURCE_LAYER:
        return FIXED_LAYER_TRAJECTORY_NAME
    return "Native FREE fixed-source-layer-{} / exact-cache free-running calibration".format(source_layer)


# Provenance: distinguishes a fixed-layer exit resolved by the official
# synchronized parallel flush (parallel_gen_token, during free-running
# generation) from one resolved only after generation already terminated
# (calibration-only terminal exact finalization). Both remain the same
# higher-level "Native FREE fixed-source-layer exact-cache calibration"
# trajectory and the same artifact source mode -- this is bookkeeping about
# *when/how* the deep tensors were observed, never a second restoration
# method or a second artifact schema.
EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH = "official_synchronized_flush"
EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION = "calibration_only_terminal_exact_finalization"
_EVENT_ORIGIN_PROVENANCE = {
    EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH: {
        "target_observation_mode": "exact_cache_observed_during_free_running",
        "generated_sequence_affected": False,
        "selected_token_affected": False,
        "confidence_trajectory_affected": False,
        "future_runtime_cache_consumed": True,
    },
    EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION: {
        "target_observation_mode": "exact_deep_recomputation_after_generation_end",
        "generated_sequence_affected": False,
        "selected_token_affected": False,
        "confidence_trajectory_affected": False,
        "future_runtime_cache_consumed": False,
    },
}

# Collection provenance only (never a restoration method): which runtime
# method(s) actually produced the fixed-layer summary's successful exact
# target events. Never describes terminal finalization as an official
# runtime flush. RUNTIME_METHOD_NORMAL_ONLY intentionally preserves the
# pre-existing "official_free_synchronized_exact_path" value (distinct from
# the EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH event-row field spelling) so
# any existing reader of a normal-only fixed-layer summary sees the same
# runtime_method value as before terminal finalization was introduced.
RUNTIME_METHOD_NORMAL_ONLY = "official_free_synchronized_exact_path"
RUNTIME_METHOD_TERMINAL_ONLY = EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION
RUNTIME_METHOD_MIXED = "mixed_official_and_calibration_only_exact_target_collection"
RUNTIME_METHOD_NO_SUCCESSFUL_EVENTS = "no_successful_exact_target_events"


def _require_central_approval_contract(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("calibration_central_approval_contract_missing")
    if value.get("approval_status") != "approved_established":
        raise ValueError("calibration_central_approval_status_invalid")
    for field in (
        "corrective_manifest_sha256",
        "accepted_artifact_sha256",
        "fitting_set_sha256",
    ):
        field_value = value.get(field)
        if not isinstance(field_value, str) or len(field_value) != 64:
            raise ValueError("calibration_central_{}_invalid".format(field))
    count = value.get("fitting_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("calibration_central_fitting_count_invalid")
    return value


def _load_selected_stable_sample_ids(path: Path | str) -> List[str]:
    target = Path(path)
    if not target.is_file():
        raise ValueError("calibration_selected_population_file_missing")
    try:
        if target.suffix.lower() == ".jsonl":
            values: List[Any] = []
            with target.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get("stable_sample_id") if isinstance(row, Mapping) else row
                    if value in (None, ""):
                        raise ValueError("selected_stable_sample_id_missing:line{}".format(line_no))
                    values.append(value)
        else:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                payload = payload.get("stable_sample_ids") or payload.get("selected_stable_sample_ids")
            if not isinstance(payload, list):
                raise ValueError("selected_population_not_list")
            values = [
                item.get("stable_sample_id") if isinstance(item, Mapping) else item
                for item in payload
            ]
        ids = [str(value) for value in values if value not in (None, "")]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "calibration_selected_population_parse_failed:{}".format(type(exc).__name__)
        ) from exc
    if len(ids) != len(values):
        raise ValueError("calibration_selected_population_id_missing")
    if len(ids) != len(set(ids)):
        raise ValueError("calibration_selected_population_duplicate")
    return ids


def _validate_approved_fitting_ids(
    stable_sample_ids: Sequence[Any],
    approval_contract: Mapping[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    if not isinstance(approval_contract, Mapping):
        raise ValueError("calibration_central_approval_contract_missing")
    count = approval_contract.get("fitting_count")
    expected_sha = approval_contract.get("fitting_set_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("calibration_central_fitting_count_invalid")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("calibration_central_fitting_set_sha256_invalid")
    ids = [str(value) for value in stable_sample_ids]
    if len(ids) != count:
        raise ValueError("calibration_{}_count_mismatch".format(source))
    if len(set(ids)) != count:
        raise ValueError("calibration_{}_duplicate_ids".format(source))
    actual_sha = stable_sample_ids_sha256(ids, sort_ids=True, expected_count=count)
    if actual_sha != expected_sha:
        raise ValueError("calibration_{}_fitting_set_sha256_mismatch".format(source))
    return {
        "fitting_count": count,
        "fitting_unique_count": len(set(ids)),
        "fitting_set_sha256": actual_sha,
    }


def _approved_corrective_fitting_ids(
    manifest_path: Path | str,
    approval_contract: Mapping[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    from .missing_kv_fitting_provenance import paper_facing_corrective_binding_preflight

    preflight = paper_facing_corrective_binding_preflight(manifest_path)
    if preflight.get("status") != "ok":
        raise ValueError(
            "calibration_approved_corrective_root_preflight_failed:{}".format(
                ",".join(str(item) for item in preflight.get("failures", []))
            )
        )
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        reconstructed = payload["reconstructed_fitting_population_manifest"]
        ids = list(reconstructed["artifact_fitting_stable_sample_ids"])
    except Exception as exc:
        raise ValueError(
            "calibration_approved_corrective_fitting_population_missing:{}".format(
                type(exc).__name__
            )
        ) from exc
    identity = _validate_approved_fitting_ids(
        ids,
        approval_contract,
        source="approved_corrective_manifest",
    )
    return [str(value) for value in ids], {
        **identity,
        "approved_corrective_manifest_sha256": preflight.get("manifest_sha256"),
        "approved_corrective_root_preflight": preflight.get(
            "approved_corrective_root_preflight"
        ),
        "central_root_of_trust_validated": preflight.get(
            "central_root_of_trust_validated"
        ),
    }


def _validate_fixed_layer_runtime_config(
    config: Any, fixed_source_layer: int, *, contract_dataset: Optional[str] = None
) -> None:
    """Reject any runtime configuration that is not one of the two frozen
    fixed-source-layer exact-cache free-running contracts this collector
    supports:

    - the Native FREE fixed-source-layer-6 contract (``contract_dataset`` is
      ``None`` or ``"cnn_dailymail"`` -- the legacy SAMSum/CNN routes):
      ``fixed_source_layer`` and ``config.shallow_exit_layer`` must both be
      exactly 6, unchanged from before this function gained the
      ``contract_dataset`` parameter.
    - the official LongT5 Multi-News FREE contract (``contract_dataset ==
      "multi_news"``): ``fixed_source_layer`` must equal the model's own
      ``config.shallow_exit_layer``, whatever the externally approved
      Multi-News contract/checkpoint protocol says that is. This is a
      consistency check, not source-layer selection -- the actual value
      still comes from ``--shallow_exit_layer`` on the command line and is
      independently re-bound by the Multi-News population contract's own
      ``fixed_source_layer`` field (see ``_from_multinews_population_
      contract``).

    Collection must be usable without turning runtime restoration on, and
    must never be mistaken for the CALM candidate_first_crossing
    exact_catchup path."""

    shallow_exit_layer = getattr(config, "shallow_exit_layer", None)
    mismatches = []
    if contract_dataset == "multi_news":
        if shallow_exit_layer is None or isinstance(shallow_exit_layer, bool):
            mismatches.append("shallow_exit_layer_required")
        elif int(shallow_exit_layer) != int(fixed_source_layer):
            mismatches.append("fixed_source_layer_must_equal_shallow_exit_layer")
    else:
        if int(fixed_source_layer) != FIXED_LAYER_REQUIRED_SOURCE_LAYER:
            mismatches.append("fixed_source_layer_not_6")
        if shallow_exit_layer is None or int(shallow_exit_layer) != FIXED_LAYER_REQUIRED_SOURCE_LAYER:
            mismatches.append("shallow_exit_layer_must_be_6")
    if not bool(getattr(config, "use_shallow_deep", False)):
        mismatches.append("use_shallow_deep_must_be_true")
    if bool(getattr(config, "use_early_exit", False)):
        mismatches.append("use_early_exit_must_be_false")
    threshold = getattr(config, "shallow2deep_conf_threshold", None)
    if threshold is None or float(threshold) != float(FIXED_LAYER_REQUIRED_THRESHOLD):
        mismatches.append("shallow2deep_conf_threshold_must_be_0_9")
    if getattr(config, "shallow2deep_conf_type", None) != "softmax":
        mismatches.append("shallow2deep_conf_type_must_be_softmax")
    if bool(getattr(config, "use_adapt_threshold", False)):
        mismatches.append("use_adapt_threshold_must_be_false")
    if not bool(getattr(config, "parallel_gen_token", False)):
        mismatches.append("parallel_gen_token_must_be_true")
    if not bool(getattr(config, "parallel_causal_mask", False)):
        mismatches.append("parallel_causal_mask_must_be_true")
    if bool(getattr(config, "copy_skipped_hidden_states", False)):
        mismatches.append("copy_skipped_hidden_states_must_be_false")
    if getattr(config, "static_exit_layer", None) is not None:
        mismatches.append("static_exit_layer_must_be_unset")
    if getattr(config, "rollback_conf_threshold", None) is not None:
        mismatches.append("rollback_conf_threshold_must_be_unset")
    if int(getattr(config, "smoke_force_flush_after_skips", None) or 0) != 0:
        mismatches.append("smoke_force_flush_after_skips_must_be_disabled")
    if bool(getattr(config, "kv_runtime_restoration_enabled", False)):
        mismatches.append("kv_runtime_restoration_enabled_must_be_false")
    if bool(getattr(config, "kv_runtime_restoration_calm_enabled", False)):
        mismatches.append("kv_runtime_restoration_calm_enabled_must_be_false")
    if getattr(config, "kv_runtime_restoration_artifact", None):
        mismatches.append("kv_runtime_restoration_artifact_must_be_empty")
    if bool(
        getattr(
            config,
            "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime",
            False,
        )
    ):
        mismatches.append("preliminary_artifact_override_must_be_false")
    if mismatches:
        raise ValueError(
            "calibration_fixed_layer_runtime_config_invalid:{}".format(";".join(mismatches))
        )


def _cpu_tensor(value: torch.Tensor, *, name: str, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("{}_not_tensor".format(name))
    tensor = value.detach().cpu().contiguous()
    if tensor.ndim != dimensions:
        raise ValueError("{}_shape_invalid:{}".format(name, list(tensor.shape)))
    if not torch.isfinite(tensor).all().item():
        raise ValueError("{}_nonfinite".format(name))
    return tensor


def _event_identity(payload: Mapping[str, Any], *, trajectory_type: str = TRAJECTORY_TYPE) -> str:
    return canonical_json_sha256(
        {
            "trajectory_type": trajectory_type,
            "stable_sample_id": payload.get("stable_sample_id"),
            "selected_order": payload.get("selected_order"),
            "generation_index": payload.get("generation_index"),
            "decoder_position": payload.get("decoder_position"),
            "selected_token_id": payload.get("selected_token_id"),
            "source_layer": payload.get("source_layer"),
        }
    )


class ExactCacheCalibrationCollector:
    """Collect committed exact-catchup events with bounded CPU buffering."""

    def __init__(
        self,
        *,
        output_dir: Path | str,
        approved_fitting_ids: Sequence[str],
        model_num_decoder_layers: int,
        model_d_model: int,
        model_num_heads: int,
        model_d_kv: int,
        max_events_per_flush: int = 8,
        population_identity: Optional[Mapping[str, Any]] = None,
        source_layer_mode: str = SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
        fixed_source_layer: Optional[int] = None,
        official_candidate_layers: Optional[Sequence[int]] = None,
        official_threshold: Optional[float] = None,
        _construction_token: Any = None,
    ) -> None:
        if _construction_token is not _COLLECTOR_CONSTRUCTION_TOKEN:
            raise ValueError(
                "calibration_direct_construction_forbidden_use_from_config"
            )
        ids = [str(value) for value in approved_fitting_ids]
        if not ids:
            raise ValueError("calibration_fitting_population_empty")
        if len(ids) != len(set(ids)):
            raise ValueError("calibration_fitting_population_duplicate")
        if int(max_events_per_flush) <= 0:
            raise ValueError("calibration_max_events_per_flush_must_be_positive")
        if source_layer_mode not in (
            SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            SOURCE_LAYER_MODE_FIXED,
            SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
        ):
            raise ValueError("calibration_source_layer_mode_invalid")
        if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
            if isinstance(fixed_source_layer, bool) or not isinstance(fixed_source_layer, int):
                raise ValueError("calibration_fixed_source_layer_required")
            # Structural validity only (a real target range must exist) --
            # NOT production authorization. Which fixed_source_layer values
            # are actually approved for which route (Native FREE source6,
            # official LongT5 Multi-News source3, ...) is decided by
            # from_config()/_validate_fixed_layer_runtime_config and the
            # dataset-specific contract branches below, never here. Direct
            # construction stays forbidden outside approved constructors
            # (see the _construction_token check above).
            if int(fixed_source_layer) <= 0 or int(fixed_source_layer) >= int(model_num_decoder_layers):
                raise ValueError("calibration_fixed_source_layer_out_of_range")
        official_layers_tuple = None
        if source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
            # The official FREE CALM candidate range/threshold are supplied
            # by the caller (derived from the model's own exit_min_layer /
            # decoder depth / exit_conf_threshold) -- never the frozen
            # historical CALM_CANDIDATE_LAYERS/CALM_THRESHOLD import used by
            # SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING below.
            official_layers_tuple = tuple(int(value) for value in (official_candidate_layers or ()))
            if not official_layers_tuple:
                raise ValueError("calibration_official_candidate_layers_required")
            if list(official_layers_tuple) != list(
                range(official_layers_tuple[0], official_layers_tuple[0] + len(official_layers_tuple))
            ):
                raise ValueError("calibration_official_candidate_layers_not_contiguous")
            if any(layer < 0 for layer in official_layers_tuple):
                raise ValueError("calibration_official_candidate_layers_negative")
            if official_threshold is None or not math.isfinite(float(official_threshold)):
                raise ValueError("calibration_official_threshold_required")
        self.source_layer_mode = source_layer_mode
        self.fixed_source_layer = int(fixed_source_layer) if fixed_source_layer is not None else None
        self.official_candidate_layers = official_layers_tuple
        self.official_threshold = float(official_threshold) if official_threshold is not None else None
        self.output_dir = Path(output_dir)
        self.approved_fitting_ids = tuple(ids)
        self._approved_order = {stable_id: index for index, stable_id in enumerate(ids)}
        self.model_num_decoder_layers = int(model_num_decoder_layers)
        self.model_d_model = int(model_d_model)
        self.model_num_heads = int(model_num_heads)
        self.model_d_kv = int(model_d_kv)
        self.max_events_per_flush = int(max_events_per_flush)
        self.population_identity = dict(population_identity or {})
        self._buffer: List[Dict[str, Any]] = []
        self._pending: Optional[Dict[str, Any]] = None
        self._committed_event_ids = set()
        self._source_target_keys = set()
        self._shard_rows: List[Dict[str, Any]] = []
        self._source_counts = Counter()
        self._source_target_counts = Counter()
        self._gap_counts = Counter()
        self._contributing_stable_samples = set()
        self._generated_token_count = 0
        self._first_crossing_token_count = 0
        self._no_crossing_token_count = 0
        self._required_units = 0
        self._executed_units = 0
        self._written_units = 0
        self._closed = False
        self._tail_pending_fixed_layer_exit_count = 0
        self._tail_pending_loss_reasons: List[Dict[str, Any]] = []
        self._total_actual_fixed_layer_exit_count = 0
        self._normal_synchronized_flush_event_count = 0
        self._calibration_only_tail_finalized_event_count = 0
        self._tail_finalized_stable_samples = set()
        self._normal_synchronized_flush_stable_samples = set()
        self._tail_finalization_failed_event_count = 0
        self._remaining_unfinalized_tail_event_count = 0
        self._tail_finalization_failure_reasons: List[Dict[str, Any]] = []
        self._source_target_counts_by_origin = {
            EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH: Counter(),
            EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION: Counter(),
        }
        self._gap_counts_by_origin = {
            EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH: Counter(),
            EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION: Counter(),
        }

    @classmethod
    def _from_test_population(
        cls,
        *,
        output_dir: Path | str,
        approved_fitting_ids: Sequence[str],
        model_num_decoder_layers: int,
        model_d_model: int,
        model_num_heads: int,
        model_d_kv: int,
        max_events_per_flush: int = 8,
        population_identity: Optional[Mapping[str, Any]] = None,
        source_layer_mode: str = SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
        fixed_source_layer: Optional[int] = None,
        official_candidate_layers: Optional[Sequence[int]] = None,
        official_threshold: Optional[float] = None,
    ) -> "ExactCacheCalibrationCollector":
        return cls(
            output_dir=output_dir,
            approved_fitting_ids=approved_fitting_ids,
            model_num_decoder_layers=model_num_decoder_layers,
            model_d_model=model_d_model,
            model_num_heads=model_num_heads,
            model_d_kv=model_d_kv,
            max_events_per_flush=max_events_per_flush,
            population_identity=population_identity,
            source_layer_mode=source_layer_mode,
            fixed_source_layer=fixed_source_layer,
            official_candidate_layers=official_candidate_layers,
            official_threshold=official_threshold,
            _construction_token=_COLLECTOR_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def from_config(cls, config: Any) -> "ExactCacheCalibrationCollector":
        from . import missing_kv_fitting_provenance as fitting_provenance

        source_layer_mode = (
            getattr(config, "kv_early_exit_exact_cache_calibration_source_layer_mode", None)
            or SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
        )
        if source_layer_mode not in (
            SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            SOURCE_LAYER_MODE_FIXED,
            SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
        ):
            raise ValueError("calibration_source_layer_mode_invalid")
        fixed_source_layer = None
        if source_layer_mode == SOURCE_LAYER_MODE_FIXED:
            fixed_source_layer = getattr(
                config, "kv_early_exit_exact_cache_calibration_fixed_source_layer", None
            )
            if isinstance(fixed_source_layer, bool) or not isinstance(fixed_source_layer, int):
                raise ValueError("calibration_fixed_source_layer_required")
            # Validation-only route marker (never used to choose a source
            # layer): which of the two frozen fixed-layer runtime contracts
            # applies. Read here, before any dataset-contract dispatch below,
            # solely so _validate_fixed_layer_runtime_config can pick the
            # right one.
            fixed_layer_contract_dataset = getattr(
                config, "kv_early_exit_exact_cache_calibration_population_contract_dataset", None
            )
            _validate_fixed_layer_runtime_config(
                config, fixed_source_layer, contract_dataset=fixed_layer_contract_dataset
            )
        if source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
            return cls._from_official_free_calm_config(config)

        population_contract_path = getattr(
            config, "kv_early_exit_exact_cache_calibration_population_contract", None
        )
        if population_contract_path:
            # Train-calibration population branch. This is a separate
            # population-binding root of trust from the SAMSum legacy
            # corrective/source-manifest branch below -- it never requires
            # (or reads) the historical SAMSum corrective manifest, and it
            # does not touch tensor extraction/staging/commit at all.
            #
            # The dataset route is taken from an EXPLICIT internal marker
            # that run_summarization.py sets from data_args.dataset_name
            # before model construction. It is never guessed from checkpoint
            # path text or model class, and the contract itself
            # independently re-validates the real dataset identity, so this
            # marker only chooses which validator runs.
            contract_dataset = getattr(
                config, "kv_early_exit_exact_cache_calibration_population_contract_dataset", None
            )
            if contract_dataset == "cnn_dailymail":
                return cls._from_cnn_dailymail_population_contract(
                    config,
                    source_layer_mode=source_layer_mode,
                    fixed_source_layer=fixed_source_layer,
                    population_contract_path=population_contract_path,
                )
            if contract_dataset == "multi_news":
                return cls._from_multinews_population_contract(
                    config,
                    source_layer_mode=source_layer_mode,
                    fixed_source_layer=fixed_source_layer,
                    population_contract_path=population_contract_path,
                )
            raise ValueError(
                "calibration_population_contract_dataset_unsupported:{}".format(contract_dataset)
            )

        source_manifest = getattr(config, "kv_early_exit_exact_cache_calibration_fitting_source_manifest", None)
        if not source_manifest:
            raise ValueError("calibration_fitting_population_binding_missing")
        corrective_manifest = getattr(
            config,
            "kv_early_exit_exact_cache_calibration_approved_corrective_manifest",
            None,
        )
        if not corrective_manifest:
            raise ValueError("calibration_approved_corrective_manifest_missing")
        selected_ids_file = getattr(config, "missing_kv_selected_stable_sample_ids_file", None)
        if not selected_ids_file:
            raise ValueError("calibration_selected_population_binding_missing")
        approval_contract = _require_central_approval_contract(
            getattr(fitting_provenance, "CENTRAL_APPROVAL_CONTRACT", None)
        )
        corrective_ids, corrective_identity = _approved_corrective_fitting_ids(
            corrective_manifest,
            approval_contract,
        )
        expected_count = approval_contract.get("fitting_count")
        expected_artifact_sha = approval_contract.get("accepted_artifact_sha256")
        population = load_source_manifest_fitting_ids(
            source_manifest,
            expected_fitting_count=expected_count,
            expected_artifact_sha256=expected_artifact_sha,
        )
        if population.get("status") != "ok":
            raise ValueError(
                "calibration_fitting_population_binding_failed:{}".format(
                    ",".join(str(item) for item in population.get("failures", []))
                )
            )
        source_ids = list(population["artifact_fitting_stable_sample_ids"])
        source_identity = _validate_approved_fitting_ids(
            source_ids,
            approval_contract,
            source="source_manifest",
        )
        selected_ids = _load_selected_stable_sample_ids(selected_ids_file)
        selected_identity = _validate_approved_fitting_ids(
            selected_ids,
            approval_contract,
            source="selected_population",
        )
        approved_set = set(corrective_ids)
        if set(source_ids) != approved_set:
            raise ValueError("calibration_source_manifest_not_bound_to_approved_corrective_population")
        if set(selected_ids) != approved_set:
            raise ValueError("calibration_selected_population_mismatch")
        return cls(
            output_dir=getattr(config, "kv_early_exit_exact_cache_calibration_output_dir"),
            approved_fitting_ids=selected_ids,
            population_identity={
                "population_role": "fitting",
                "central_approval_contract": {
                    "fitting_count": expected_count,
                    "fitting_set_sha256": approval_contract.get("fitting_set_sha256"),
                    "corrective_manifest_sha256": approval_contract.get(
                        "corrective_manifest_sha256"
                    ),
                },
                "approved_corrective_binding": corrective_identity,
                "source_manifest_binding": {
                    **source_identity,
                    "source_manifest_sha256": population.get("source_manifest_sha256"),
                    "source_manifest_identity": population.get("source_manifest_identity"),
                },
                "selected_population_binding": selected_identity,
            },
            model_num_decoder_layers=int(getattr(config, "num_layers")),
            model_d_model=int(getattr(config, "d_model")),
            model_num_heads=int(getattr(config, "num_heads")),
            model_d_kv=int(getattr(config, "d_kv")),
            max_events_per_flush=int(
                getattr(config, "kv_early_exit_exact_cache_calibration_max_events_per_flush", 8)
            ),
            source_layer_mode=source_layer_mode,
            fixed_source_layer=fixed_source_layer,
            _construction_token=_COLLECTOR_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def _from_official_free_calm_config(cls, config: Any) -> "ExactCacheCalibrationCollector":
        """Official FREE CALM-style production early-exit calibration
        collection. The candidate range/threshold are DERIVED from the same
        config fields that already govern the actual runtime decoder loop
        (``exit_min_layer``, ``num_layers``, ``exit_conf_threshold``) --
        never a separately hard-coded list -- so the collector can never
        silently drift from what the model itself actually evaluated.

        Population binding here is deliberately minimal: the caller-supplied
        stable-ID file is the sole population source, with no cross-check
        against the historical SAMSum CENTRAL_APPROVAL_CONTRACT (that
        contract is specific to the frozen fitting-409 population and has no
        meaning for a new official FREE CALM experiment population). Central
        approval/provenance for an official FREE CALM population, if ever
        needed, is a follow-up decision for whoever runs that experiment --
        this method does not invent a new provenance/approval framework for
        it.
        """

        exit_min_layer = getattr(config, "exit_min_layer", None)
        if exit_min_layer is None or isinstance(exit_min_layer, bool):
            raise ValueError("calibration_official_exit_min_layer_missing")
        num_decoder_layers = getattr(config, "num_layers", None)
        if num_decoder_layers is None or isinstance(num_decoder_layers, bool):
            raise ValueError("calibration_official_num_decoder_layers_missing")
        threshold = getattr(config, "exit_conf_threshold", None)
        if threshold is None:
            raise ValueError("calibration_official_threshold_missing")
        candidate_layers = official_free_calm_candidate_layers(
            exit_min_layer=int(exit_min_layer), num_decoder_layers=int(num_decoder_layers)
        )

        selected_ids_file = getattr(config, "missing_kv_selected_stable_sample_ids_file", None)
        if not selected_ids_file:
            raise ValueError("calibration_selected_population_binding_missing")
        selected_ids = _load_selected_stable_sample_ids(selected_ids_file)

        return cls(
            output_dir=getattr(config, "kv_early_exit_exact_cache_calibration_output_dir"),
            approved_fitting_ids=selected_ids,
            population_identity={
                "population_role": "fitting",
                "population_source": "official_free_calm_selected_stable_sample_ids_file",
                "selected_population_count": len(selected_ids),
            },
            model_num_decoder_layers=int(getattr(config, "num_layers")),
            model_d_model=int(getattr(config, "d_model")),
            model_num_heads=int(getattr(config, "num_heads")),
            model_d_kv=int(getattr(config, "d_kv")),
            max_events_per_flush=int(
                getattr(config, "kv_early_exit_exact_cache_calibration_max_events_per_flush", 8)
            ),
            source_layer_mode=SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
            official_candidate_layers=candidate_layers,
            official_threshold=float(threshold),
            _construction_token=_COLLECTOR_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def _from_cnn_dailymail_population_contract(
        cls,
        config: Any,
        *,
        source_layer_mode: str,
        fixed_source_layer: Optional[int],
        population_contract_path: str,
    ) -> "ExactCacheCalibrationCollector":
        from .cnndm_source6_population_contract import (
            load_and_validate_population_contract,
            validate_selected_stable_sample_ids,
        )

        if source_layer_mode != SOURCE_LAYER_MODE_FIXED or fixed_source_layer != 6:
            raise ValueError("calibration_cnn_population_contract_requires_fixed_source_layer_6")
        population_contract_sha256 = getattr(
            config, "kv_early_exit_exact_cache_calibration_population_contract_sha256", None
        )
        if not population_contract_sha256:
            raise ValueError("calibration_cnn_population_contract_sha256_missing")
        selected_ids_file = getattr(config, "missing_kv_selected_stable_sample_ids_file", None)
        if not selected_ids_file:
            raise ValueError("calibration_selected_population_binding_missing")

        contract = load_and_validate_population_contract(
            population_contract_path, expected_contract_sha256=population_contract_sha256
        )
        selected_ids = _load_selected_stable_sample_ids(selected_ids_file)
        selected_identity = validate_selected_stable_sample_ids(selected_ids, contract)

        return cls(
            output_dir=getattr(config, "kv_early_exit_exact_cache_calibration_output_dir"),
            approved_fitting_ids=selected_ids,
            population_identity={
                "population_role": "fitting",
                "dataset_protocol": {
                    "dataset_name": contract["dataset_name"],
                    "dataset_config_name": contract["dataset_config_name"],
                    "split": contract["split"],
                    "text_column": contract["text_column"],
                    "summary_column": contract["summary_column"],
                    "source_prefix": contract["source_prefix"],
                    "max_source_length": contract["max_source_length"],
                    "max_target_length": contract["max_target_length"],
                    "dataset_fingerprint": contract["dataset_fingerprint"],
                },
                "population_contract_binding": {
                    "evidence_type": contract["evidence_type"],
                    "contract_file_sha256": contract["contract_file_sha256"],
                    "selection_algorithm": contract["selection_algorithm"],
                    "selection_seed": contract["selection_seed"],
                    "candidate_budget": contract["candidate_budget"],
                    "stable_sample_id_algorithm_identity": contract["stable_sample_id_algorithm_identity"],
                },
                "selected_population_binding": selected_identity,
                # Model/tokenizer identity is verified against this same
                # contract before model construction in run_summarization.py
                # (fail-closed, so collection can never begin against a
                # mismatched checkpoint/tokenizer); echoed here only for
                # audit, not re-checked against a live model/tokenizer.
                "expected_model_checkpoint_identity_sha256": contract["model_checkpoint_identity_sha256"],
                "expected_tokenizer_identity_sha256": contract["tokenizer_identity_sha256"],
            },
            model_num_decoder_layers=int(getattr(config, "num_layers")),
            model_d_model=int(getattr(config, "d_model")),
            model_num_heads=int(getattr(config, "num_heads")),
            model_d_kv=int(getattr(config, "d_kv")),
            max_events_per_flush=int(
                getattr(config, "kv_early_exit_exact_cache_calibration_max_events_per_flush", 8)
            ),
            source_layer_mode=source_layer_mode,
            fixed_source_layer=fixed_source_layer,
            _construction_token=_COLLECTOR_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def _from_multinews_population_contract(
        cls,
        config: Any,
        *,
        source_layer_mode: str,
        fixed_source_layer: Optional[int],
        population_contract_path: str,
    ) -> "ExactCacheCalibrationCollector":
        """Multi-News counterpart of _from_cnn_dailymail_population_contract.

        Mirrors that accepted path exactly -- selected-ID file required,
        externally approved contract SHA required, selected count/set/order
        validated, population identity recorded, model shape still taken from
        ``config``, source mode fixed_layer -- and differs in which contract
        validator runs, in also recording the contract-declared
        ``candidate_budgets`` schedule (Multi-News has no module-frozen
        schedule), and in the approved fixed source layer itself: the
        contract's own ``fixed_source_layer`` field is the source of truth
        (currently 3 for the official LongT5 checkpoint protocol, never a
        second hard-coded 6). No tensor staging/extraction/commit logic is
        touched.
        """

        from .multinews_source3_population_contract import (
            load_and_validate_population_contract,
            validate_selected_stable_sample_ids,
        )

        if source_layer_mode != SOURCE_LAYER_MODE_FIXED or fixed_source_layer is None:
            raise ValueError("calibration_multinews_population_contract_requires_fixed_layer_mode")
        population_contract_sha256 = getattr(
            config, "kv_early_exit_exact_cache_calibration_population_contract_sha256", None
        )
        if not population_contract_sha256:
            raise ValueError("calibration_multinews_population_contract_sha256_missing")
        selected_ids_file = getattr(config, "missing_kv_selected_stable_sample_ids_file", None)
        if not selected_ids_file:
            raise ValueError("calibration_selected_population_binding_missing")

        contract = load_and_validate_population_contract(
            population_contract_path, expected_contract_sha256=population_contract_sha256
        )
        # The contract's own approved fixed_source_layer must equal the
        # actual runtime value -- _validate_fixed_layer_runtime_config (in
        # from_config(), above) already enforced fixed_source_layer ==
        # config.shallow_exit_layer for this route, so this closes the third
        # leg: contract.fixed_source_layer == config.fixed_source_layer ==
        # config.shallow_exit_layer.
        if int(contract["fixed_source_layer"]) != int(fixed_source_layer):
            raise ValueError("calibration_multinews_population_contract_fixed_source_layer_mismatch")
        selected_ids = _load_selected_stable_sample_ids(selected_ids_file)
        selected_identity = validate_selected_stable_sample_ids(selected_ids, contract)

        return cls(
            output_dir=getattr(config, "kv_early_exit_exact_cache_calibration_output_dir"),
            approved_fitting_ids=selected_ids,
            population_identity={
                # Fitting/calibration population -- never a held-out
                # evaluation population.
                "population_role": "fitting",
                "dataset_protocol": {
                    "dataset_name": contract["dataset_name"],
                    "dataset_config_name": contract["dataset_config_name"],
                    "split": contract["split"],
                    "text_column": contract["text_column"],
                    "summary_column": contract["summary_column"],
                    "source_prefix": contract["source_prefix"],
                    "max_source_length": contract["max_source_length"],
                    "max_target_length": contract["max_target_length"],
                    "dataset_fingerprint": contract["dataset_fingerprint"],
                },
                "population_contract_binding": {
                    "evidence_type": contract["evidence_type"],
                    "contract_file_sha256": contract["contract_file_sha256"],
                    "selection_algorithm": contract["selection_algorithm"],
                    "selection_seed": contract["selection_seed"],
                    # Multi-News carries its own schedule; recording it makes
                    # the pre-registered candidate set auditable.
                    "candidate_budgets": list(contract["candidate_budgets"]),
                    "candidate_budget": contract["candidate_budget"],
                    "stable_sample_id_algorithm_identity": contract["stable_sample_id_algorithm_identity"],
                    # Evidence binding, not runtime selection: the contract's
                    # own approved source layer, already cross-checked above
                    # against the actual runtime fixed_source_layer/
                    # shallow_exit_layer.
                    "fixed_source_layer": contract["fixed_source_layer"],
                },
                "selected_population_binding": selected_identity,
                # Model/tokenizer identity is verified against this same
                # contract before model construction in run_summarization.py
                # (fail-closed, so collection can never begin against a
                # mismatched checkpoint/tokenizer); echoed here only for
                # audit, not re-checked against a live model/tokenizer.
                "expected_model_checkpoint_identity_sha256": contract["model_checkpoint_identity_sha256"],
                "expected_tokenizer_identity_sha256": contract["tokenizer_identity_sha256"],
            },
            model_num_decoder_layers=int(getattr(config, "num_layers")),
            model_d_model=int(getattr(config, "d_model")),
            model_num_heads=int(getattr(config, "num_heads")),
            model_d_kv=int(getattr(config, "d_kv")),
            max_events_per_flush=int(
                getattr(config, "kv_early_exit_exact_cache_calibration_max_events_per_flush", 8)
            ),
            source_layer_mode=source_layer_mode,
            fixed_source_layer=fixed_source_layer,
            _construction_token=_COLLECTOR_CONSTRUCTION_TOKEN,
        )

    @property
    def buffered_event_count(self) -> int:
        return len(self._buffer)

    @property
    def pending_event(self) -> Optional[Mapping[str, Any]]:
        return self._pending

    def reset_run(self) -> None:
        if self._pending is not None:
            raise ValueError("calibration_pending_event_at_run_reset")

    def _validate_and_extract_hidden_and_kv(
        self,
        *,
        source_layer: int,
        hidden_by_layer: Sequence[torch.Tensor],
        complete_cache: Sequence[Sequence[torch.Tensor]],
        target_records: Sequence[Mapping[str, Any]],
    ):
        """Shared by stage_first_crossing and stage_fixed_layer_exit: neither
        the target-layer/tensor-population checks nor the tensor extraction
        depend on which source-layer policy produced the event."""

        if len(hidden_by_layer) != self.model_num_decoder_layers:
            raise ValueError("calibration_hidden_layer_population_incomplete")
        if len(complete_cache) != self.model_num_decoder_layers:
            raise ValueError("calibration_cache_layer_population_incomplete")
        expected_targets = list(range(source_layer, self.model_num_decoder_layers))
        observed_targets = [int(record.get("target_layer")) for record in target_records]
        if observed_targets != expected_targets:
            raise ValueError("calibration_committed_target_layer_population_incomplete")
        if any(record.get("final_status") != "exact_cache_committed" for record in target_records):
            raise ValueError("calibration_target_not_exact_cache_committed")

        hidden = torch.stack(
            [_cpu_tensor(value, name="calibration_hidden", dimensions=3).squeeze(0).squeeze(0) for value in hidden_by_layer],
            dim=0,
        ).contiguous()
        if list(hidden.shape) != [self.model_num_decoder_layers, self.model_d_model]:
            raise ValueError("calibration_hidden_shape_mismatch:{}".format(list(hidden.shape)))
        cache_position = int(target_records[0].get("cache_position"))
        if cache_position < 0 or any(int(record.get("cache_position")) != cache_position for record in target_records):
            raise ValueError("calibration_cache_position_invalid")
        keys = []
        values = []
        for layer_idx, state in enumerate(complete_cache):
            if not isinstance(state, (list, tuple)) or len(state) < 2:
                raise ValueError("calibration_cache_state_missing:layer{}".format(layer_idx))
            key = _cpu_tensor(state[0], name="calibration_exact_target_key", dimensions=4)
            value = _cpu_tensor(state[1], name="calibration_exact_target_value", dimensions=4)
            if tuple(key.shape) != tuple(value.shape):
                raise ValueError("calibration_exact_key_value_shape_mismatch")
            if cache_position >= int(key.shape[2]):
                raise ValueError("calibration_cache_position_out_of_range")
            keys.append(key[0, :, cache_position, :].contiguous())
            values.append(value[0, :, cache_position, :].contiguous())
        key_tensor = torch.stack(keys, dim=0).contiguous()
        value_tensor = torch.stack(values, dim=0).contiguous()
        expected_kv_shape = [self.model_num_decoder_layers, self.model_num_heads, self.model_d_kv]
        if list(key_tensor.shape) != expected_kv_shape or list(value_tensor.shape) != expected_kv_shape:
            raise ValueError("calibration_exact_kv_shape_mismatch")
        return expected_targets, hidden, cache_position, key_tensor, value_tensor

    def stage_first_crossing(
        self,
        *,
        sample_context: Mapping[str, Any],
        generation_index: int,
        decoder_position: int,
        source_layer: int,
        confidence: float,
        threshold: float,
        candidate_evaluations: Sequence[Mapping[str, Any]],
        hidden_by_layer: Sequence[torch.Tensor],
        complete_cache: Sequence[Sequence[torch.Tensor]],
        target_records: Sequence[Mapping[str, Any]],
        source_selected_token_id: int,
    ) -> None:
        """Historical CALM candidate_first_crossing mode: source layer and
        threshold are validated against the frozen (4,6,8,10)/0.9 tuple,
        exactly as before. Unchanged behavior -- delegates to the shared
        implementation with those frozen historical values."""

        self._stage_candidate_first_crossing_impl(
            expected_mode=SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            expected_candidate_layers=CALM_CANDIDATE_LAYERS,
            expected_threshold=CALM_THRESHOLD,
            mode_error="calibration_first_crossing_stage_requires_candidate_first_crossing_mode",
            sample_context=sample_context,
            generation_index=generation_index,
            decoder_position=decoder_position,
            source_layer=source_layer,
            confidence=confidence,
            threshold=threshold,
            candidate_evaluations=candidate_evaluations,
            hidden_by_layer=hidden_by_layer,
            complete_cache=complete_cache,
            target_records=target_records,
            source_selected_token_id=source_selected_token_id,
        )

    def stage_official_free_calm_first_crossing(
        self,
        *,
        sample_context: Mapping[str, Any],
        generation_index: int,
        decoder_position: int,
        source_layer: int,
        confidence: float,
        threshold: float,
        candidate_evaluations: Sequence[Mapping[str, Any]],
        hidden_by_layer: Sequence[torch.Tensor],
        complete_cache: Sequence[Sequence[torch.Tensor]],
        target_records: Sequence[Mapping[str, Any]],
        source_selected_token_id: int,
    ) -> None:
        """Official FREE CALM-style production early-exit mode counterpart of
        ``stage_first_crossing``: the same shape of fail-closed validation
        (every candidate strictly before the crossing layer must have
        failed, the crossing layer itself must pass, confidence/threshold/
        comparator consistency, exact target-layer population, etc.), but
        checked against THIS collector's own ``official_candidate_layers``/
        ``official_threshold`` (the contiguous exit_min_layer..num_decoder_
        layers-1 range and the model's own runtime exit_conf_threshold) --
        never the frozen historical (4,6,8,10)/0.9 tuple ``stage_first_
        crossing`` above continues to use unchanged."""

        if self.official_candidate_layers is None or self.official_threshold is None:
            raise ValueError("calibration_official_candidate_layers_or_threshold_missing")
        self._stage_candidate_first_crossing_impl(
            expected_mode=SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
            expected_candidate_layers=self.official_candidate_layers,
            expected_threshold=self.official_threshold,
            mode_error="calibration_official_first_crossing_stage_requires_official_free_calm_mode",
            sample_context=sample_context,
            generation_index=generation_index,
            decoder_position=decoder_position,
            source_layer=source_layer,
            confidence=confidence,
            threshold=threshold,
            candidate_evaluations=candidate_evaluations,
            hidden_by_layer=hidden_by_layer,
            complete_cache=complete_cache,
            target_records=target_records,
            source_selected_token_id=source_selected_token_id,
        )

    def _stage_candidate_first_crossing_impl(
        self,
        *,
        expected_mode: str,
        expected_candidate_layers: Sequence[int],
        expected_threshold: float,
        mode_error: str,
        sample_context: Mapping[str, Any],
        generation_index: int,
        decoder_position: int,
        source_layer: int,
        confidence: float,
        threshold: float,
        candidate_evaluations: Sequence[Mapping[str, Any]],
        hidden_by_layer: Sequence[torch.Tensor],
        complete_cache: Sequence[Sequence[torch.Tensor]],
        target_records: Sequence[Mapping[str, Any]],
        source_selected_token_id: int,
    ) -> None:
        """Shared by ``stage_first_crossing`` (historical, frozen values) and
        ``stage_official_free_calm_first_crossing`` (official, caller-bound
        values): the invariant SHAPE -- all-earlier-candidates-fail,
        crossing-candidate-passes, confidence/threshold/comparator
        consistency, exact target-layer population -- was already generic
        over the candidate layer set/threshold; only the frozen import was
        hard-coded. This factors that shape out once instead of duplicating
        it, without changing either caller's externally observable
        behavior."""

        if self.source_layer_mode != expected_mode:
            raise ValueError(mode_error)
        if self._closed:
            raise ValueError("calibration_collector_closed")
        if self._pending is not None:
            raise ValueError("calibration_pending_event_duplicate")
        stable_id = str(sample_context.get("stable_sample_id") or "")
        if stable_id not in self._approved_order:
            raise ValueError("calibration_stable_sample_id_not_in_fitting_population")
        selected_order = sample_context.get("selected_order")
        if isinstance(selected_order, bool) or int(selected_order) != self._approved_order[stable_id]:
            raise ValueError("calibration_selected_order_mismatch")
        source_layer = int(source_layer)
        expected_candidate_layers = [int(value) for value in expected_candidate_layers]
        if source_layer not in expected_candidate_layers:
            raise ValueError("calibration_source_layer_not_frozen_candidate")
        if float(threshold) != float(expected_threshold):
            raise ValueError("calibration_threshold_not_frozen")
        if not math.isfinite(float(confidence)) or not float(confidence) > float(threshold):
            raise ValueError("calibration_first_crossing_confidence_invalid")
        if int(generation_index) < 0 or int(decoder_position) < 0:
            raise ValueError("calibration_generation_or_decoder_position_invalid")
        if isinstance(source_selected_token_id, bool) or int(source_selected_token_id) < 0:
            raise ValueError("calibration_selected_token_id_invalid")
        expected_targets, hidden, cache_position, key_tensor, value_tensor = (
            self._validate_and_extract_hidden_and_kv(
                source_layer=source_layer,
                hidden_by_layer=hidden_by_layer,
                complete_cache=complete_cache,
                target_records=target_records,
            )
        )

        candidate_rows = [dict(row) for row in candidate_evaluations]
        expected_candidate_chain = [layer for layer in expected_candidate_layers if layer <= source_layer]
        observed_candidate_layers = [int(row.get("candidate_layer")) for row in candidate_rows]
        if observed_candidate_layers != expected_candidate_chain:
            raise ValueError("calibration_first_crossing_decision_missing")
        if any(row.get("candidate_pass") is not False for row in candidate_rows[:-1]):
            raise ValueError("calibration_candidate_before_first_crossing_passed")
        if candidate_rows[-1].get("candidate_pass") is not True:
            raise ValueError("calibration_first_crossing_decision_not_pass")
        for row in candidate_rows:
            row_confidence = float(row.get("confidence"))
            if not math.isfinite(row_confidence):
                raise ValueError("calibration_candidate_confidence_nonfinite")
            if float(row.get("threshold", expected_threshold)) != float(expected_threshold):
                raise ValueError("calibration_candidate_threshold_not_frozen")
            if row.get("threshold_comparator", "strict_gt") != "strict_gt":
                raise ValueError("calibration_candidate_comparator_not_frozen")
        event = {
            "stable_sample_id": stable_id,
            "selected_order": int(selected_order),
            "raw_dataset_index": sample_context.get("raw_dataset_index"),
            "dataset_provided_id": sample_context.get("dataset_provided_id"),
            "generation_index": int(generation_index),
            "decoder_position": int(decoder_position),
            "selected_token_id": int(source_selected_token_id),
            "source_layer": source_layer,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "threshold_comparator": "strict_gt",
            "candidate_evaluations": candidate_rows,
            "target_layers": expected_targets,
            "cache_position": cache_position,
            "hidden": hidden,
            "key": key_tensor,
            "value": value_tensor,
        }
        event["event_identity"] = _event_identity(event)
        if event["event_identity"] in self._committed_event_ids:
            raise ValueError("calibration_duplicate_event_identity")
        self._pending = event

    def stage_fixed_layer_exit(
        self,
        *,
        sample_context: Mapping[str, Any],
        generation_index: int,
        decoder_position: int,
        confidence: float,
        threshold: float,
        hidden_by_layer: Sequence[torch.Tensor],
        complete_cache: Sequence[Sequence[torch.Tensor]],
        target_records: Sequence[Mapping[str, Any]],
        source_selected_token_id: int,
        event_origin: str = EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH,
    ) -> None:
        """Stage one Native FREE fixed-source-layer exact-cache event.

        Unlike stage_first_crossing, there is no multi-candidate evaluation
        chain to validate: the Native FREE framework evaluates confidence at
        exactly one frozen layer, so a single confidence/threshold check is
        the whole decision.

        ``event_origin`` distinguishes the official synchronized parallel
        flush (the default, for backward compatibility with every existing
        caller) from calibration-only terminal exact finalization. It never
        changes what tensors are validated or how -- only the provenance
        recorded alongside the event.
        """

        if self.source_layer_mode != SOURCE_LAYER_MODE_FIXED:
            raise ValueError("calibration_fixed_layer_stage_requires_fixed_layer_mode")
        if event_origin not in _EVENT_ORIGIN_PROVENANCE:
            raise ValueError("calibration_event_origin_invalid")
        if self._closed:
            raise ValueError("calibration_collector_closed")
        if self._pending is not None:
            raise ValueError("calibration_pending_event_duplicate")
        stable_id = str(sample_context.get("stable_sample_id") or "")
        if stable_id not in self._approved_order:
            raise ValueError("calibration_stable_sample_id_not_in_fitting_population")
        selected_order = sample_context.get("selected_order")
        if isinstance(selected_order, bool) or int(selected_order) != self._approved_order[stable_id]:
            raise ValueError("calibration_selected_order_mismatch")
        source_layer = int(self.fixed_source_layer)
        if float(threshold) != float(FIXED_LAYER_REQUIRED_THRESHOLD):
            raise ValueError("calibration_threshold_not_frozen")
        if not math.isfinite(float(confidence)) or not float(confidence) > float(threshold):
            raise ValueError("calibration_fixed_layer_exit_confidence_invalid")
        if int(generation_index) < 0 or int(decoder_position) < 0:
            raise ValueError("calibration_generation_or_decoder_position_invalid")
        if isinstance(source_selected_token_id, bool) or int(source_selected_token_id) < 0:
            raise ValueError("calibration_selected_token_id_invalid")
        expected_targets, hidden, cache_position, key_tensor, value_tensor = (
            self._validate_and_extract_hidden_and_kv(
                source_layer=source_layer,
                hidden_by_layer=hidden_by_layer,
                complete_cache=complete_cache,
                target_records=target_records,
            )
        )

        event = {
            "stable_sample_id": stable_id,
            "selected_order": int(selected_order),
            "raw_dataset_index": sample_context.get("raw_dataset_index"),
            "dataset_provided_id": sample_context.get("dataset_provided_id"),
            "generation_index": int(generation_index),
            "decoder_position": int(decoder_position),
            "selected_token_id": int(source_selected_token_id),
            "source_layer": source_layer,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "threshold_comparator": FIXED_LAYER_THRESHOLD_COMPARATOR,
            "candidate_evaluations": [],
            "target_layers": expected_targets,
            "cache_position": cache_position,
            "hidden": hidden,
            "key": key_tensor,
            "value": value_tensor,
            "event_origin": event_origin,
            **_EVENT_ORIGIN_PROVENANCE[event_origin],
        }
        event["event_identity"] = _event_identity(event, trajectory_type=FIXED_LAYER_TRAJECTORY_TYPE)
        if event["event_identity"] in self._committed_event_ids:
            raise ValueError("calibration_duplicate_event_identity")
        self._pending = event

    def stage_no_crossing(self) -> None:
        if self._pending is not None:
            raise ValueError("calibration_pending_event_before_no_crossing")
        self._pending = {"policy_no_crossing": True}

    def abort_pending(self) -> None:
        self._pending = None

    def record_fixed_layer_exit_observed(self) -> None:
        """Record that one actual fixed-source-layer exit occurred (the
        skip_mask=True decision itself), independent of whether it is later
        resolved by a normal synchronized flush, a calibration-only terminal
        finalization, or neither. Called exactly once per actual exit, at
        exit time -- never inferred later from committed-event counts, so
        the conservation equation (normal + terminal + failed +
        remaining_unfinalized == total_actual) holds even when some exits
        never resolve."""

        if self.source_layer_mode != SOURCE_LAYER_MODE_FIXED:
            return
        self._total_actual_fixed_layer_exit_count += 1

    def record_tail_finalization_failure(
        self,
        count: int,
        *,
        stage: str,
        failure_type: str,
        reason: str,
    ) -> None:
        """Record fixed-layer exits whose calibration-only terminal exact
        finalization was attempted but failed (e.g. a deep-layer exception,
        or a post-computation invariant violation). Distinct from
        record_tail_pending_loss: this always means finalization was
        actually attempted for these events and did not complete -- it does
        not fabricate a partially completed event, and it never marks a
        successfully finalized event as failed."""

        if self.source_layer_mode != SOURCE_LAYER_MODE_FIXED:
            return
        count = int(count)
        if count <= 0:
            return
        self._tail_finalization_failed_event_count += count
        self._tail_finalization_failure_reasons.append(
            {"stage": str(stage), "failure_type": str(failure_type), "reason": str(reason), "count": count}
        )

    def record_tail_finalization_unresolved(self, count: int, *, reason: str) -> None:
        """Record fixed-layer exits that calibration-only terminal exact
        finalization never even attempted (e.g. a precondition failed before
        any deep computation began, or finalization stopped early after an
        earlier event in the same batch failed). Distinct from
        record_tail_finalization_failure: no deep computation was attempted
        for these specific events."""

        if self.source_layer_mode != SOURCE_LAYER_MODE_FIXED:
            return
        count = int(count)
        if count <= 0:
            return
        self._remaining_unfinalized_tail_event_count += count
        self._tail_finalization_failure_reasons.append(
            {"stage": "not_attempted", "failure_type": "unresolved", "reason": str(reason), "count": count}
        )

    def record_tail_pending_loss(self, pending_count: int, *, reason: str) -> None:
        """Record actual exits that never reached a synchronized flush before
        generation ended or restarted (no official exact tail-flush path
        exists for the fixed-source-layer trajectory). This never fabricates
        a fitting event for them; it only makes the loss explicit and
        nonzero so summary()/status reflects an invalid/incomplete
        collection rather than silently dropping them."""

        if self.source_layer_mode != SOURCE_LAYER_MODE_FIXED:
            return
        count = int(pending_count)
        if count <= 0:
            return
        self._tail_pending_fixed_layer_exit_count += count
        self._tail_pending_loss_reasons.append({"reason": str(reason), "count": count})

    def commit_selected_token(self, selected_token_id: int) -> None:
        pending = self._pending
        if pending is None:
            return
        if pending.get("policy_no_crossing"):
            self._pending = None
            self._generated_token_count += 1
            self._no_crossing_token_count += 1
            return
        if int(selected_token_id) != int(pending["selected_token_id"]):
            raise ValueError("calibration_selected_token_changed_after_exact_catchup")
        self._pending = None
        self._generated_token_count += 1
        event_id = pending["event_identity"]
        if event_id in self._committed_event_ids:
            raise ValueError("calibration_duplicate_event_identity")
        source_layer = int(pending["source_layer"])
        target_layers = list(pending["target_layers"])
        for target_layer in target_layers:
            key = (event_id, source_layer, int(target_layer))
            if key in self._source_target_keys:
                raise ValueError("calibration_duplicate_source_target_record")
        self._committed_event_ids.add(event_id)
        for target_layer in target_layers:
            self._source_target_keys.add((event_id, source_layer, int(target_layer)))
        self._contributing_stable_samples.add(str(pending["stable_sample_id"]))
        self._first_crossing_token_count += 1
        units = len(target_layers)
        self._required_units += units
        self._executed_units += units
        self._written_units += units
        self._source_counts[source_layer] += 1
        event_origin = pending.get("event_origin", EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH)
        for target_layer in target_layers:
            if target_layer > source_layer:
                self._source_target_counts[(source_layer, target_layer)] += 1
                gap_bin = gap_bin_for_gap(target_layer - source_layer)
                self._gap_counts[gap_bin] += 1
                if event_origin in self._source_target_counts_by_origin:
                    self._source_target_counts_by_origin[event_origin][(source_layer, target_layer)] += 1
                    self._gap_counts_by_origin[event_origin][gap_bin] += 1
        if event_origin == EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION:
            self._calibration_only_tail_finalized_event_count += 1
            self._tail_finalized_stable_samples.add(str(pending["stable_sample_id"]))
        else:
            self._normal_synchronized_flush_event_count += 1
            # Candidate-first-crossing events (stage_first_crossing) never
            # carry event_origin, so they also fall into this branch via the
            # pending.get(..., EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH)
            # default above. This stable-sample set is Native FREE
            # fixed-source-layer-6 provenance only, so it must not accept
            # that default -- only a genuine, explicit
            # official_synchronized_flush event on a fixed-layer collector.
            if (
                self.source_layer_mode == SOURCE_LAYER_MODE_FIXED
                and event_origin == EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH
            ):
                self._normal_synchronized_flush_stable_samples.add(str(pending["stable_sample_id"]))
        self._buffer.append(pending)
        if len(self._buffer) >= self.max_events_per_flush:
            self.flush()

    @property
    def _active_trajectory_type(self) -> str:
        return FIXED_LAYER_TRAJECTORY_TYPE if self.source_layer_mode == SOURCE_LAYER_MODE_FIXED else TRAJECTORY_TYPE

    def _common_manifest(self, event: Mapping[str, Any], *, record_type: str, relative_path: str) -> Dict[str, Any]:
        is_fixed = self.source_layer_mode == SOURCE_LAYER_MODE_FIXED
        return {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": record_type,
            "file_path": relative_path,
            "stable_sample_id": event["stable_sample_id"],
            "selected_order": event["selected_order"],
            "raw_dataset_index": event.get("raw_dataset_index"),
            "dataset_provided_id": event.get("dataset_provided_id"),
            "missing_kv_sample_context_status": "ok",
            "generation_index": event["generation_index"],
            "token_count": 1,
            "layer_count": self.model_num_decoder_layers,
            "decoder_position_start": event["decoder_position"],
            "decoder_position_end_exclusive": event["decoder_position"] + 1,
            "decoder_positions_contiguous": True,
            "layer_start": 0,
            "layer_end_exclusive": self.model_num_decoder_layers,
            "layers_contiguous": True,
            "model_num_decoder_layers": self.model_num_decoder_layers,
            "full_depth_execution": False,
            "use_shallow_deep": is_fixed,
            "use_early_exit": not is_fixed,
            "static_exit_layer": None,
            "trajectory_type": self._active_trajectory_type,
            "calibration_event_identity": event["event_identity"],
            "dump_succeeded": True,
        }

    @staticmethod
    def _save_payload(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("calibration_shard_already_exists:{}".format(path.name))
        temp = path.with_name(path.name + ".tmp")
        try:
            with temp.open("wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    def _write_event(self, event: Mapping[str, Any]) -> None:
        stem = "g{:06d}_p{:06d}_{}".format(
            int(event["generation_index"]), int(event["decoder_position"]), event["event_identity"][:12]
        )
        hidden_relative = "hidden_generation_{}.pt".format(stem)
        kv_relative = "kv_generation_{}.pt".format(stem)
        hidden_path = self.output_dir / hidden_relative
        kv_path = self.output_dir / kv_relative
        hidden_tensor = event["hidden"].unsqueeze(0).contiguous()
        key_tensor = event["key"].unsqueeze(0).contiguous()
        value_tensor = event["value"].unsqueeze(0).contiguous()

        hidden_row = self._common_manifest(event, record_type="packed_generation_hidden", relative_path=hidden_relative)
        hidden_row.update(
            {
                "include_raw_hidden": True,
                "include_normed_hidden": False,
                "raw_hidden_shape": list(hidden_tensor.shape),
                "normed_hidden_shape": None,
                "dtype": str(hidden_tensor.dtype).replace("torch.", ""),
                "device_before_dump": "collector_detached_to_cpu",
                "model_d_model": self.model_d_model,
            }
        )
        hidden_row = populate_packed_record_identities(hidden_row, logical_record_type=HIDDEN_LOGICAL_RECORD_TYPE)
        hidden_payload = {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": "packed_generation_hidden",
            "raw_hidden": hidden_tensor,
            "metadata": dict(hidden_row),
        }
        kv_row = self._common_manifest(event, record_type="packed_generation_kv", relative_path=kv_relative)
        kv_row.update(
            {
                "key_shape": list(key_tensor.shape),
                "value_shape": list(value_tensor.shape),
                "dtype": str(key_tensor.dtype).replace("torch.", ""),
                "device_before_dump": "collector_detached_to_cpu",
                "model_num_heads": self.model_num_heads,
                "model_d_kv": self.model_d_kv,
            }
        )
        kv_row = populate_packed_record_identities(kv_row, logical_record_type=KV_LOGICAL_RECORD_TYPE)
        kv_payload = {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": "packed_generation_kv",
            "key": key_tensor,
            "value": value_tensor,
            "metadata": dict(kv_row),
        }
        self._save_payload(hidden_path, hidden_payload)
        try:
            self._save_payload(kv_path, kv_payload)
        except Exception:
            hidden_path.unlink(missing_ok=True)
            raise
        append_jsonl(self.output_dir / "all_layer_hidden_manifest.jsonl", hidden_row)
        append_jsonl(self.output_dir / "all_layer_kv_manifest.jsonl", kv_row)

        if self.source_layer_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
            # Fixed-layer events rely on calibration_event_manifest.jsonl alone
            # (it already carries confidence/threshold/source_layer/exact
            # commit status); a CALM-branded trace row would misdescribe a
            # single-layer decision as a multi-candidate one.
            trace_row = {
                "trace_schema_version": COLLECTOR_SCHEMA_VERSION,
                "record_type": "early_exit_exact_cache_actual_first_crossing_trace",
                "trace_semantics": TRAJECTORY_TYPE,
                "actual_first_crossing_only": True,
                "policy_name": CALM_POLICY_NAME,
                "policy_sha256": calm_policy_sha256(),
                "stable_sample_id": event["stable_sample_id"],
                "selected_order": event["selected_order"],
                "raw_dataset_index": event.get("raw_dataset_index"),
                "dataset_provided_id": event.get("dataset_provided_id"),
                "generation_index": event["generation_index"],
                "decoder_position": event["decoder_position"],
                "token_index": event["decoder_position"],
                "selected_token_id": event["selected_token_id"],
                "model_num_decoder_layers": self.model_num_decoder_layers,
                "candidate_exit_layers": list(CALM_CANDIDATE_LAYERS),
                "threshold": CALM_THRESHOLD,
                "threshold_comparator": "strict_gt",
                "confidence_compute_dtype": "float32",
                "adaptive_threshold": False,
                "candidate_evaluations": event["candidate_evaluations"],
                "first_crossing_candidate_layer": event["source_layer"],
                "full_depth_fallback": False,
                "trajectory_type": TRAJECTORY_TYPE,
                "selected_token_source": SELECTED_TOKEN_SOURCE,
                "exact_final_logits_used_for_token_selection": False,
            }
            trace_row["trace_row_uid"] = canonical_json_sha256(trace_row)
            append_jsonl(self.output_dir / "calm_first_crossing_trace.jsonl", trace_row)
        elif self.source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
            # Official FREE CALM counterpart of the historical CALM-branded
            # trace row above: same shape, but the official contiguous
            # candidate range / model-config-driven threshold / official
            # policy name -- written to its own file so it is never confused
            # with (or aggregated together with) the frozen historical
            # (4,6,8,10)/0.9 trace rows above.
            official_layers = list(self.official_candidate_layers or ())
            official_threshold = float(self.official_threshold)
            official_trace_row = {
                "trace_schema_version": COLLECTOR_SCHEMA_VERSION,
                "record_type": "official_free_calm_exact_cache_actual_first_crossing_trace",
                "trace_semantics": TRAJECTORY_TYPE,
                "actual_first_crossing_only": True,
                "policy_name": OFFICIAL_FREE_CALM_POLICY_NAME,
                "policy_sha256": calm_policy_sha256(
                    candidate_layers=official_layers,
                    threshold=official_threshold,
                    policy_name=OFFICIAL_FREE_CALM_POLICY_NAME,
                ),
                "stable_sample_id": event["stable_sample_id"],
                "selected_order": event["selected_order"],
                "raw_dataset_index": event.get("raw_dataset_index"),
                "dataset_provided_id": event.get("dataset_provided_id"),
                "generation_index": event["generation_index"],
                "decoder_position": event["decoder_position"],
                "token_index": event["decoder_position"],
                "selected_token_id": event["selected_token_id"],
                "model_num_decoder_layers": self.model_num_decoder_layers,
                "candidate_exit_layers": official_layers,
                "threshold": official_threshold,
                "threshold_comparator": "strict_gt",
                "confidence_compute_dtype": "float32",
                "adaptive_threshold": False,
                "candidate_evaluations": event["candidate_evaluations"],
                "first_crossing_candidate_layer": event["source_layer"],
                "full_depth_fallback": False,
                "trajectory_type": TRAJECTORY_TYPE,
                "selected_token_source": SELECTED_TOKEN_SOURCE,
                "exact_final_logits_used_for_token_selection": False,
            }
            official_trace_row["trace_row_uid"] = canonical_json_sha256(official_trace_row)
            append_jsonl(self.output_dir / "official_free_calm_first_crossing_trace.jsonl", official_trace_row)

        is_fixed = self.source_layer_mode == SOURCE_LAYER_MODE_FIXED
        for target_layer in event["target_layers"]:
            target_layer = int(target_layer)
            append_jsonl(
                self.output_dir / "calibration_event_manifest.jsonl",
                {
                    "collector_schema_version": COLLECTOR_SCHEMA_VERSION,
                    "record_type": EVENT_RECORD_TYPE,
                    "event_identity": event["event_identity"],
                    "stable_sample_id": event["stable_sample_id"],
                    "selected_order": event["selected_order"],
                    "generation_index": event["generation_index"],
                    "decoder_position": event["decoder_position"],
                    "selected_token_id": event["selected_token_id"],
                    "source_layer": event["source_layer"],
                    "confidence": event["confidence"],
                    "threshold": event["threshold"],
                    "threshold_comparator": "strict_gt",
                    "candidate_evaluations": event["candidate_evaluations"],
                    # fixed_layer_exit_decision is the semantic field for
                    # fixed-source-layer-6 events; first_crossing_decision is
                    # retained, always True, as a backward-compatibility
                    # alias -- it never carries fixed-mode-specific meaning.
                    "first_crossing_decision": True,
                    "fixed_layer_exit_decision": True if is_fixed else None,
                    "target_layer": target_layer,
                    "cache_position": event["cache_position"],
                    "source_hidden": {"file_path": hidden_relative, "layer_offset": event["source_layer"]},
                    "exact_target_hidden": {"file_path": hidden_relative, "layer_offset": target_layer},
                    "exact_target_key": {"file_path": kv_relative, "layer_offset": target_layer},
                    "exact_target_value": {"file_path": kv_relative, "layer_offset": target_layer},
                    "trajectory_type": self._active_trajectory_type,
                    "selected_token_source": (
                        FIXED_LAYER_SELECTED_TOKEN_SOURCE
                        if self.source_layer_mode == SOURCE_LAYER_MODE_FIXED
                        else SELECTED_TOKEN_SOURCE
                    ),
                    "exact_final_logits_used_for_token_selection": False,
                    "artifact_used": False,
                    "learned_restoration_used": False,
                    "approximate_overwrite_used": False,
                    "exact_cache_commit_status": "committed",
                    "event_origin": event.get("event_origin", EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH),
                    "target_observation_mode": event.get("target_observation_mode"),
                    "generated_sequence_affected": event.get("generated_sequence_affected"),
                    "selected_token_affected": event.get("selected_token_affected"),
                    "confidence_trajectory_affected": event.get("confidence_trajectory_affected"),
                    "future_runtime_cache_consumed": event.get("future_runtime_cache_consumed"),
                },
            )
        shard_row = {
            "collector_schema_version": COLLECTOR_SCHEMA_VERSION,
            "record_type": SHARD_RECORD_TYPE,
            "event_identity": event["event_identity"],
            "relative_shard_paths": [hidden_relative, kv_relative],
            "file_sizes": {hidden_relative: hidden_path.stat().st_size, kv_relative: kv_path.stat().st_size},
            "file_sha256": {hidden_relative: sha256_file(hidden_path), kv_relative: sha256_file(kv_path)},
            "record_count": len(event["target_layers"]),
            "stable_sample_id_count": 1,
            "event_count": 1,
            "source_layer_counts": {str(event["source_layer"]): 1},
            "source_target_pair_counts": {
                "{}->{}".format(event["source_layer"], target): 1
                for target in event["target_layers"]
            },
            "target_gap_bin_counts": dict(
                Counter(
                    gap_bin_for_gap(target - int(event["source_layer"]))
                    for target in event["target_layers"]
                )
            ),
            "tensor_shape_summary": {
                "raw_hidden": list(hidden_tensor.shape),
                "key": list(key_tensor.shape),
                "value": list(value_tensor.shape),
            },
            "tensor_dtype_summary": {
                "raw_hidden": str(hidden_tensor.dtype).replace("torch.", ""),
                "key": str(key_tensor.dtype).replace("torch.", ""),
                "value": str(value_tensor.dtype).replace("torch.", ""),
            },
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
        }
        append_jsonl(self.output_dir / "calibration_shard_manifest.jsonl", shard_row)
        self._shard_rows.append(shard_row)

    def flush(self) -> None:
        events = self._buffer
        self._buffer = []
        for event in events:
            self._write_event(event)
        if events:
            self._write_summary()

    def summary(self) -> Dict[str, Any]:
        relation_ok = (
            self._first_crossing_token_count + self._no_crossing_token_count == self._generated_token_count
        )
        unit_ok = self._required_units == self._executed_units == self._written_units
        tail_pending_ok = self._tail_pending_fixed_layer_exit_count == 0
        tail_finalization_ok = (
            self._tail_finalization_failed_event_count == 0
            and self._remaining_unfinalized_tail_event_count == 0
        )
        is_fixed = self.source_layer_mode == SOURCE_LAYER_MODE_FIXED
        if is_fixed:
            source_layers = (self.fixed_source_layer,)
        elif self.source_layer_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
            source_layers = tuple(self.official_candidate_layers or ())
        else:
            source_layers = tuple(CALM_CANDIDATE_LAYERS)
        event_count_conservation_ok = (
            self._normal_synchronized_flush_event_count
            + self._calibration_only_tail_finalized_event_count
            + self._tail_finalization_failed_event_count
            + self._remaining_unfinalized_tail_event_count
            == self._total_actual_fixed_layer_exit_count
        ) if is_fixed else True
        if is_fixed:
            has_normal = self._normal_synchronized_flush_event_count > 0
            has_terminal = self._calibration_only_tail_finalized_event_count > 0
            if has_normal and has_terminal:
                runtime_method = RUNTIME_METHOD_MIXED
            elif has_terminal:
                runtime_method = RUNTIME_METHOD_TERMINAL_ONLY
            elif has_normal:
                runtime_method = RUNTIME_METHOD_NORMAL_ONLY
            else:
                runtime_method = RUNTIME_METHOD_NO_SUCCESSFUL_EVENTS
        else:
            runtime_method = "exact_catchup"
        return {
            "collector_schema_version": COLLECTOR_SCHEMA_VERSION,
            "evidence_type": SUMMARY_EVIDENCE_TYPE,
            "status": (
                "ok"
                if relation_ok
                and unit_ok
                and tail_pending_ok
                and tail_finalization_ok
                and event_count_conservation_ok
                and self._pending is None
                else "invalid"
            ),
            "trajectory_name": (
                _fixed_layer_trajectory_name(self.fixed_source_layer) if is_fixed else TRAJECTORY_NAME
            ),
            "trajectory_type": self._active_trajectory_type,
            "runtime_method": runtime_method,
            "source_layer_mode": self.source_layer_mode,
            "fixed_source_layer": self.fixed_source_layer if is_fixed else None,
            # These frozen-contract fields are only meaningful (and only
            # ever constructed) for source_layer_mode == fixed_layer; the
            # collector's own from_config()/_validate_fixed_layer_runtime_config
            # already guarantees every one of them at construction time.
            "threshold": FIXED_LAYER_REQUIRED_THRESHOLD if is_fixed else None,
            "threshold_comparator": FIXED_LAYER_THRESHOLD_COMPARATOR if is_fixed else None,
            "adaptive_threshold": False if is_fixed else None,
            "use_shallow_deep": True if is_fixed else None,
            "use_early_exit": False if is_fixed else None,
            "population_role": "fitting",
            "population_identity": self.population_identity,
            "total_generated_token_count": self._generated_token_count,
            "first_crossing_token_count": self._first_crossing_token_count,
            "fixed_layer_exit_token_count": self._first_crossing_token_count if is_fixed else None,
            "policy_no_crossing_full_depth_token_count": self._no_crossing_token_count,
            "tail_pending_fixed_layer_exit_count": self._tail_pending_fixed_layer_exit_count,
            "tail_pending_loss_reasons": list(self._tail_pending_loss_reasons),
            "contributing_stable_sample_count": len(self._contributing_stable_samples),
            # Provenance-split accounting (Native FREE fixed-source-layer-6
            # trajectory only): every actual shallow-layer exit is counted
            # exactly once, in exactly one of these four buckets, regardless
            # of whether it was ever staged into the collector's tensor
            # population -- see event_count_conservation_check.
            "normal_synchronized_flush_event_count": self._normal_synchronized_flush_event_count,
            "calibration_only_tail_finalized_event_count": self._calibration_only_tail_finalized_event_count,
            "normal_synchronized_flush_unique_stable_sample_count": len(
                self._normal_synchronized_flush_stable_samples
            ),
            "tail_finalized_unique_stable_sample_count": len(self._tail_finalized_stable_samples),
            "total_actual_fixed_layer_exit_count": self._total_actual_fixed_layer_exit_count,
            "total_successfully_finalized_exit_count": (
                self._normal_synchronized_flush_event_count + self._calibration_only_tail_finalized_event_count
            ),
            "tail_finalization_failed_event_count": self._tail_finalization_failed_event_count,
            "remaining_unfinalized_tail_event_count": self._remaining_unfinalized_tail_event_count,
            "tail_finalization_failure_reasons": list(self._tail_finalization_failure_reasons),
            "event_count_conservation_check": event_count_conservation_ok,
            "source_layer_event_counts": {str(layer): self._source_counts[layer] for layer in source_layers},
            "observed_strict_deeper_source_target_pair_counts": {
                "{}->{}".format(source, target): count
                for (source, target), count in sorted(self._source_target_counts.items())
            },
            "observed_strict_deeper_gap_bin_counts": dict(sorted(self._gap_counts.items())),
            "normal_event_source_target_pair_counts": {
                "{}->{}".format(source, target): count
                for (source, target), count in sorted(
                    self._source_target_counts_by_origin[EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH].items()
                )
            },
            "terminal_event_source_target_pair_counts": {
                "{}->{}".format(source, target): count
                for (source, target), count in sorted(
                    self._source_target_counts_by_origin[
                        EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION
                    ].items()
                )
            },
            "normal_event_gap_bin_counts": dict(
                sorted(self._gap_counts_by_origin[EVENT_ORIGIN_OFFICIAL_SYNCHRONIZED_FLUSH].items())
            ),
            "terminal_event_gap_bin_counts": dict(
                sorted(
                    self._gap_counts_by_origin[
                        EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION
                    ].items()
                )
            ),
            "combined_fitting_support": dict(sorted(self._gap_counts.items())),
            "zero_support_source_layers": [
                int(layer) for layer in source_layers if self._source_counts[layer] == 0
            ],
            "zero_support_strict_deeper_source_target_pairs": [
                "{}->{}".format(source, target)
                for source in source_layers
                for target in range(source + 1, self.model_num_decoder_layers)
                if self._source_target_counts[(source, target)] == 0
            ],
            "zero_support_strict_deeper_gap_bins": [
                label
                for label in GAP_BIN_LABELS
                if label != "same_layer" and self._gap_counts[label] == 0
            ],
            "zero_support_entries_fabricated": False,
            "fitter_compatibility_status": "NOT_RUN",
            "fitter_compatibility": None,
            "exact_catchup_required_token_layer_units": self._required_units,
            "exact_catchup_executed_token_layer_units": self._executed_units,
            "exact_cache_written_token_layer_units": self._written_units,
            "approximation_requested_token_layer_units": 0,
            "approximation_succeeded_token_layer_units": 0,
            "approximate_overwrite_token_layer_units": 0,
            "restoration_requested_token_layer_units": 0,
            "restoration_succeeded_token_layer_units": 0,
            "restoration_overwritten_token_layer_units": 0,
            "catchup_failure": 0,
            "cache_write_failure": 0,
            "restoration_error_failure": 0,
            "restoration_error_fallback": 0,
            "unresolved_transaction": 0,
            "task_c1_exact_overwrite": False,
            "restoration_policy_mode": "exact_catchup_no_approximation",
            "artifact_used": False,
            "learned_restoration_used": False,
            "calm_projection_used": False,
            "approximate_overwrite_used": False,
            "speed_claim_valid": False,
            "task_c2_implemented": False,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "shard_count": len(self._shard_rows),
            "buffered_event_count": len(self._buffer),
            "validation": {
                "token_population_relation_valid": relation_ok,
                "exact_token_layer_unit_relation_valid": unit_ok,
            },
        }

    def _write_summary(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_file(self.output_dir / "calibration_summary.json", self.summary())

    def close(self) -> Dict[str, Any]:
        if self._closed:
            return self.summary()
        if self._pending is not None:
            raise ValueError("calibration_unresolved_pending_event")
        self.flush()
        self._closed = True
        self._write_summary()
        return self.summary()


def existing_fitter_compatibility_probe(output_dir: Path | str) -> Dict[str, Any]:
    """Exercise the existing fitter's read-only input-construction path."""

    root = Path(output_dir)
    hidden_manifest = root / "all_layer_hidden_manifest.jsonl"
    kv_manifest = root / "all_layer_kv_manifest.jsonl"
    trace_path = root / "calm_first_crossing_trace.jsonl"
    try:
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import evaluate_phase2_hidden_based_kv_regeneration_from_dumps as p2
        import evaluate_phase3c_final_missing_cache_policy_from_dumps as p3
    except Exception as exc:
        return {
            "status": "FITTER_COMPATIBILITY_BLOCKED",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "blocking_reasons": [
                "existing_fitter_input_api_unavailable:{}:{}".format(
                    type(exc).__name__, str(exc)[:240]
                )
            ],
        }

    try:
        hidden, hidden_diagnostics = p2.load_hidden_records(hidden_manifest, root)
        kv, kv_diagnostics = p2.load_kv_records(kv_manifest, root)
        loader_errors = []
        for label, diagnostics in (
            ("hidden", hidden_diagnostics),
            ("kv", kv_diagnostics),
        ):
            for field in ("manifest_expand_errors", "load_errors"):
                if int(diagnostics.get(field, 0)):
                    loader_errors.append(
                        "{}_{}:{}".format(label, field, diagnostics.get(field))
                    )
        if loader_errors:
            raise ValueError("collector_fitter_loader_errors:" + ",".join(loader_errors))
        hidden_keys = set(hidden)
        kv_keys = set(kv)
        if not hidden_keys or hidden_keys != kv_keys:
            raise ValueError("collector_fitter_logical_population_mismatch")
        layer_records, join_diagnostics = p2.build_layer_records(
            kv,
            hidden,
            require_paper_identity=True,
        )
        if not layer_records:
            raise ValueError("collector_fitter_layer_records_empty")
        trace_rows = p2.load_jsonl(trace_path)
        if not trace_rows:
            raise ValueError("collector_fitter_trace_rows_empty")
    except Exception as exc:
        return {
            "status": "INVALID_TRACE",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "invalid_reasons": [
                "existing_fitter_load_or_identity_validation_failed:{}:{}".format(
                    type(exc).__name__, str(exc)[:240]
                )
            ],
        }

    decoder_layer_counts = {
        int(record.get("metadata", {}).get("model_num_decoder_layers"))
        for record in kv.values()
        if record.get("metadata", {}).get("model_num_decoder_layers") is not None
    }
    if len(decoder_layer_counts) != 1:
        return {
            "status": "INVALID_TRACE",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "invalid_reasons": ["collector_fitter_decoder_layer_count_invalid"],
        }
    decoder_layer_count = next(iter(decoder_layer_counts))
    try:
        alignment_validation = p3.validate_calm_trace_dump_alignment(
            layer_records,
            trace_rows,
            decoder_layer_count,
        )
    except Exception as exc:
        return {
            "status": "INVALID_TRACE",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "invalid_reasons": [
                "existing_fitter_trace_identity_validation_failed:{}:{}".format(
                    type(exc).__name__, str(exc)[:240]
                )
            ],
        }
    identity_failure_fields = (
        "duplicate_dump_token_count",
        "duplicate_trace_token_count",
        "missing_target_token_count",
        "missing_trace_token_count",
        "unexpected_trace_token_count",
    )
    if (
        any(int(alignment_validation.get(field, 0)) for field in identity_failure_fields)
        or alignment_validation.get("dump_token_count")
        != alignment_validation.get("trace_token_count")
        or alignment_validation.get("dump_token_population_sha256")
        != alignment_validation.get("trace_token_population_sha256")
    ):
        return {
            "status": "INVALID_TRACE",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "invalid_reasons": ["existing_fitter_trace_population_identity_mismatch"],
            "alignment_validation": p3.sanitized_alignment_validation(
                alignment_validation
            ),
        }
    semantic_assumption_failures = sorted(
        {
            error
            for row in trace_rows
            for error in p3._trace_row_semantics_errors(row)
        }
    )
    try:
        examples, example_diagnostics, same_layer_records = (
            p3.build_calm_first_crossing_examples(
                layer_records,
                trace_rows,
                [CALM_THRESHOLD],
                p2.parse_gap_bins("1-2,3-4,5-8,9-12,13-16,17-999"),
                decoder_layer_count,
            )
        )
    except Exception as exc:
        return {
            "status": "FITTER_COMPATIBILITY_BLOCKED",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "hidden_logical_record_count": len(hidden),
            "kv_logical_record_count": len(kv),
            "logical_population_match": True,
            "tensor_loader_compatibility": "COMPATIBLE",
            "blocking_reasons": [
                "existing_fitter_input_construction_rejected:{}:{}".format(
                    type(exc).__name__, str(exc)[:240]
                )
            ],
            "incompatible_semantic_assumptions": semantic_assumption_failures,
            "hidden_loader_diagnostics": hidden_diagnostics,
            "kv_loader_diagnostics": kv_diagnostics,
            "join_diagnostics": join_diagnostics,
            "alignment_validation": p3.sanitized_alignment_validation(
                alignment_validation
            ),
        }

    invalid_examples = []
    learned_pairs = Counter()
    required_tensor_fields = (
        "source_hidden",
        "target_hidden",
        "last_exact_key",
        "last_exact_value",
        "target_key",
        "target_value",
    )
    for index, example in enumerate(examples):
        source_layer = example.get("source_layer")
        target_layer = example.get("target_layer")
        if (
            isinstance(source_layer, bool)
            or isinstance(target_layer, bool)
            or not isinstance(source_layer, int)
            or not isinstance(target_layer, int)
            or target_layer <= source_layer
        ):
            invalid_examples.append("example_{}_strict_deeper_pair_invalid".format(index))
            continue
        tensors = {field: example.get(field) for field in required_tensor_fields}
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            invalid_examples.append("example_{}_required_tensor_missing".format(index))
            continue
        if (
            tensors["source_hidden"].ndim != 1
            or tensors["target_hidden"].shape != tensors["source_hidden"].shape
            or tensors["last_exact_key"].ndim != 4
            or tensors["last_exact_value"].shape != tensors["last_exact_key"].shape
            or tensors["target_key"].shape != tensors["last_exact_key"].shape
            or tensors["target_value"].shape != tensors["last_exact_key"].shape
            or any(not torch.isfinite(value).all().item() for value in tensors.values())
        ):
            invalid_examples.append("example_{}_required_tensor_shape_invalid".format(index))
            continue
        learned_pairs[(source_layer, target_layer)] += 1
    if invalid_examples or not examples:
        return {
            "status": "INVALID_TRACE",
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "invalid_reasons": invalid_examples[:32]
            or ["existing_fitter_examples_empty"],
        }
    return {
        "status": "COMPATIBLE",
        "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
        "hidden_logical_record_count": len(hidden),
        "kv_logical_record_count": len(kv),
        "logical_population_match": True,
        "tensor_loader_compatibility": "COMPATIBLE",
        "layer_record_token_count": len(layer_records),
        "learned_strict_deeper_example_count": len(examples),
        "same_layer_projection_record_count": len(same_layer_records),
        "learned_strict_deeper_pair_counts": {
            "{}->{}".format(source, target): count
            for (source, target), count in sorted(learned_pairs.items())
        },
        "hidden_loader_diagnostics": hidden_diagnostics,
        "kv_loader_diagnostics": kv_diagnostics,
        "join_diagnostics": join_diagnostics,
        "example_diagnostics": example_diagnostics,
        "alignment_validation": p3.sanitized_alignment_validation(
            alignment_validation
        ),
        "parameter_estimation_executed": False,
        "artifact_serialization_executed": False,
    }


__all__ = [
    "ExactCacheCalibrationCollector",
    "TRAJECTORY_NAME",
    "TRAJECTORY_TYPE",
    "existing_fitter_compatibility_probe",
]
