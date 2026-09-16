# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
A subclass of `Trainer` specific to Question-Answering tasks
"""
from typing import Dict, List, Mapping, Optional, Union, Any, Tuple

import hashlib
import json
import math
import os
import time
import copy
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import Seq2SeqTrainer
try:
    from transformers.utils import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available():
        return False
try:
    from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
except ImportError:
    def is_deepspeed_zero3_enabled():
        return False

    def deepspeed_init(*args, **kwargs):
        raise ImportError("transformers.deepspeed is unavailable in this Transformers version")
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    EvalLoopOutput, 
    has_length,
    EvalPrediction, 
    denumpify_detensorize,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    find_batch_size, 
    nested_concat, 
    nested_numpify, 
    nested_truncate,
    IterableDatasetShard,
)
from models.deploying_t5 import DeployT5ForConditionalGeneration
from models.deploying_longt5 import DeployLongT5ForConditionalGeneration
from our_kv_restoration.missing_kv_dump_provenance import (
    append_jsonl,
    read_jsonl,
    validate_generation_bindings,
    write_json_file,
)
from our_kv_restoration.missing_kv_exact_catchup_overhead import (
    flatten_runtime_path_stratified_rows,
    write_aggregate_csv_atomic,
    write_events_jsonl_atomic,
    write_json_atomic,
)
from our_kv_restoration.missing_kv_runtime_accounting import GenerationWallTimer


class SumTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_kv_effective_population_records = None
        self._missing_kv_population_records_by_order = {}
        self._missing_kv_trainer_generation_index = 0
        self._missing_kv_per_sample_generation_index = 0
        self._generation_wall_timer = None

    def set_missing_kv_effective_population_records(self, records: List[Dict[str, Any]]) -> None:
        self.missing_kv_effective_population_records = list(records)
        self._missing_kv_population_records_by_order = {
            int(record["selected_order"]): dict(record) for record in self.missing_kv_effective_population_records
        }

    def _missing_kv_decoder(self):
        model = self.model
        if getattr(getattr(model, "config", None), "use_lora", False) and hasattr(model, "base_model"):
            model = model.base_model
        return getattr(model, "decoder", None)

    def _begin_missing_kv_evaluation(self) -> None:
        decoder = self._missing_kv_decoder()
        if decoder is not None and hasattr(decoder, "begin_missing_kv_evaluation"):
            decoder.begin_missing_kv_evaluation()
        elif decoder is not None and hasattr(decoder, "_reset_runtime_restoration_counters"):
            decoder._reset_runtime_restoration_counters()
        # One-time Phase-3c inference setup (device-local fitted parameters +
        # validated stacked bank), performed HERE -- after checkpoint loading
        # and final device placement, before the generation wall timer is
        # reset and before the first timed generate() -- so first-sample
        # timing never pays model/runtime setup. Idempotent; no restoration
        # arithmetic is performed.
        if decoder is not None and hasattr(decoder, "prepare_phase3c_runtime_for_inference"):
            decoder.prepare_phase3c_runtime_for_inference()

    def _end_missing_kv_evaluation(self) -> None:
        decoder = self._missing_kv_decoder()
        if decoder is not None and hasattr(decoder, "end_missing_kv_evaluation"):
            decoder.end_missing_kv_evaluation()

    def _missing_kv_accounting_summary(self) -> Optional[Dict[str, Any]]:
        decoder = self._missing_kv_decoder()
        if decoder is not None and hasattr(decoder, "missing_kv_runtime_accounting_summary"):
            return decoder.missing_kv_runtime_accounting_summary()
        return None

    def _missing_kv_accounting_metrics(self, metric_key_prefix: str) -> Dict[str, Any]:
        decoder = self._missing_kv_decoder()
        if decoder is not None and hasattr(decoder, "missing_kv_runtime_accounting_metrics"):
            return decoder.missing_kv_runtime_accounting_metrics(metric_key_prefix=metric_key_prefix)
        return {}

    def _missing_kv_per_sample_accounting_snapshot(self) -> Dict[str, Any]:
        """Return a synchronization-free snapshot of existing counters."""

        decoder = self._missing_kv_decoder()
        if decoder is None or not hasattr(decoder, "_missing_kv_accounting_obj"):
            return {"counters": {}, "runtime_counters": {}, "generated_token_count": 0}
        accounting = decoder._missing_kv_accounting_obj()
        counters = dict(getattr(accounting, "counters", {}) or {})
        return {
            "counters": counters,
            "generation_count": int(getattr(accounting, "generation_count", 0) or 0),
            "calm_first_crossing_token_count": int(
                getattr(accounting, "calm_first_crossing_token_count", 0) or 0
            ),
            "calm_full_depth_fallback_token_count": int(
                getattr(accounting, "calm_full_depth_fallback_token_count", 0) or 0
            ),
            "calm_source_layer_counts": dict(
                getattr(accounting, "calm_source_layer_counts", {}) or {}
            ),
            "calm_transaction_failure_token_count": int(
                getattr(accounting, "calm_transaction_failure_token_count", 0) or 0
            ),
            "runtime_counters": dict(
                getattr(decoder, "_kv_runtime_restoration_counters", {}) or {}
            ),
        }

    @staticmethod
    def _missing_kv_nonnegative_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> int:
        value = int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        if value < 0:
            raise ValueError("negative per-sample missing-KV accounting delta: {}={}".format(key, value))
        return value

    def _build_missing_kv_per_sample_accounting_row(
        self,
        *,
        context: Optional[Mapping[str, Any]],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        status: str,
        generated_token_count_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not context or context.get("stable_sample_id") in (None, ""):
            raise ValueError("stable_sample_id is required for per-sample missing-KV accounting")
        before_counters = dict(before.get("counters") or {})
        after_counters = dict(after.get("counters") or {})
        delta = lambda key: self._missing_kv_nonnegative_delta(before_counters, after_counters, key)
        generated = delta("generated_token_count")
        if "generated_token_count" not in after_counters:
            if generated_token_count_override is None:
                generated = 0
            else:
                generated = int(generated_token_count_override)
                if generated < 0:
                    raise ValueError("generated_token_count_override must be non-negative")
        nominal_skipped_units = delta("nominal_skipped_token_layer_units")

        decoder = self._missing_kv_decoder()
        decoder_layer_count = len(getattr(decoder, "block", ()) or ())
        # Only the CALM Task C1 runtime restoration path (candidate-
        # first-crossing / official_free_calm source modes) selects a
        # variable source layer per token, so only it lacks a single fixed
        # skipped-layers-per-exit denominator. use_shallow_deep == False
        # alone is NOT a reliable CALM signal: production Full WCE
        # (use_early_exit=False) and Original FREE State Copying (which
        # does not populate calm_first_crossing_token_count /
        # calm_source_layer_counts at all) both also run with
        # use_shallow_deep == False. Reuse the decoder's own existing
        # discriminator instead of re-deriving mode from counter values;
        # default False (i.e. the original fixed-formula behavior below)
        # when the decoder does not expose it, so pre-existing decoder
        # stubs/tests keep their current behavior unchanged.
        calm_taskc1_runtime = bool(getattr(decoder, "_is_calm_taskc1_runtime_enabled", lambda: False)())
        if not calm_taskc1_runtime:
            source_layer = int(getattr(decoder, "shallow_exit_layer", 0) or 0)
            skipped_layers_per_exit = decoder_layer_count - source_layer
            if nominal_skipped_units:
                if skipped_layers_per_exit <= 0 or nominal_skipped_units % skipped_layers_per_exit != 0:
                    raise ValueError(
                        "fixed-source nominal skip accounting invariant failed: units={} skipped_layers={}".format(
                            nominal_skipped_units, skipped_layers_per_exit
                        )
                    )
                early_exit_count = nominal_skipped_units // skipped_layers_per_exit
            else:
                early_exit_count = 0
        else:
            # Variable-source CALM Task C1 runtime accounting: derive early_exit_count and
            # the expected nominal-skip units from the existing per-source-
            # layer counters (already populated by record_calm_first_crossing
            # for every CALM run), instead of assuming one fixed source.
            before_source_counts = dict(before.get("calm_source_layer_counts") or {})
            after_source_counts = dict(after.get("calm_source_layer_counts") or {})
            expected_nominal_skipped_units = 0
            early_exit_count = 0
            for key in set(before_source_counts) | set(after_source_counts):
                source_layer = int(key)
                if source_layer < 0 or source_layer >= decoder_layer_count:
                    raise ValueError(
                        "invalid CALM source layer in per-sample accounting: {}".format(source_layer)
                    )
                source_delta = self._missing_kv_nonnegative_delta(before_source_counts, after_source_counts, key)
                early_exit_count += source_delta
                expected_nominal_skipped_units += source_delta * (decoder_layer_count - source_layer)
            calm_first_crossing_delta = self._missing_kv_nonnegative_delta(
                before, after, "calm_first_crossing_token_count"
            )
            if early_exit_count != calm_first_crossing_delta:
                raise ValueError(
                    "CALM per-sample source-layer count mismatch: source_sum={} first_crossing={}".format(
                        early_exit_count, calm_first_crossing_delta
                    )
                )
            if expected_nominal_skipped_units != nominal_skipped_units:
                raise ValueError(
                    "variable-source CALM nominal skip accounting invariant failed: expected={} actual={}".format(
                        expected_nominal_skipped_units, nominal_skipped_units
                    )
                )
        if early_exit_count > generated:
            raise ValueError(
                "fixed-source early-exit count exceeds generated token count: {}>{}".format(
                    early_exit_count, generated
                )
            )
        no_crossing_count = generated - early_exit_count
        if calm_taskc1_runtime:
            calm_full_depth_fallback_delta = self._missing_kv_nonnegative_delta(
                before, after, "calm_full_depth_fallback_token_count"
            )
            if no_crossing_count != calm_full_depth_fallback_delta:
                raise ValueError(
                    "CALM per-sample no-crossing/full-depth-fallback mismatch: derived={} counter={}".format(
                        no_crossing_count, calm_full_depth_fallback_delta
                    )
                )

        runtime = dict(after.get("runtime_counters") or {})
        missing_map_count = sum(
            int(runtime.get(key, 0) or 0)
            for key in (
                "missing_map_count",
                "missing_threshold_count",
                "missing_hidden_pair_count",
                "missing_k_pair_count",
                "missing_v_gap_count",
                "calm_missing_map_count",
            )
        )
        nonfinite_count = int(runtime.get("nan_or_inf_count", 0) or 0) + int(
            runtime.get("calm_nan_or_inf_count", 0) or 0
        )
        transaction_failure_count = self._missing_kv_nonnegative_delta(
            before,
            after,
            "calm_transaction_failure_token_count",
        ) + int(runtime.get("unexpected_runtime_error_count", 0) or 0)

        return {
            "schema_version": 1,
            "record_type": "missing_kv_per_sample_accounting",
            "stable_sample_id": str(context.get("stable_sample_id")),
            "selected_order": int(context.get("selected_order")),
            "raw_dataset_index": int(context.get("raw_dataset_index")),
            "generation_index": int(self._missing_kv_per_sample_generation_index),
            "status": str(status),
            "generated_token_count": generated,
            "early_exit_token_count": early_exit_count,
            "no_crossing_full_depth_token_count": no_crossing_count,
            "exact_catchup_required_token_layer_units": delta(
                "exact_catchup_required_token_layer_units"
            ),
            "exact_catchup_executed_token_layer_units": delta(
                "exact_catchup_executed_token_layer_units"
            ),
            "restoration_requested_count": delta(
                "restoration_requested_token_layer_units"
            ),
            "restoration_succeeded_count": delta(
                "restoration_succeeded_token_layer_units"
            ),
            "restoration_failed_count": delta(
                "restoration_failed_token_layer_units"
            ),
            "fallback_count": delta("fallback_token_layer_units"),
            "missing_map_count": missing_map_count,
            "nonfinite_count": nonfinite_count,
            "transaction_failure_count": transaction_failure_count,
        }

    def _prepare_missing_kv_per_sample_accounting_output(self) -> None:
        config = getattr(self.model, "config", None)
        path = getattr(config, "missing_kv_per_sample_accounting_output", None)
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8"):
            pass
        self._missing_kv_per_sample_generation_index = 0

    def _write_missing_kv_per_sample_accounting_row(
        self,
        *,
        context: Optional[Mapping[str, Any]],
        before: Mapping[str, Any],
        status: str,
        generated_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        config = getattr(self.model, "config", None)
        path = getattr(config, "missing_kv_per_sample_accounting_output", None)
        if not path:
            return
        after = self._missing_kv_per_sample_accounting_snapshot()
        generated_count_override = None
        if generated_tokens is not None:
            config = getattr(self.model, "config", None)
            values = generated_tokens[0].detach().cpu().reshape(-1).tolist()
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(config, "eos_token_id", None)
            generated_count_override = len(
                self._effective_generated_token_ids(
                    values,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=eos_token_id,
                    decoder_start_token_id=getattr(config, "decoder_start_token_id", None),
                )
            )
        row = self._build_missing_kv_per_sample_accounting_row(
            context=context,
            before=before,
            after=after,
            status=status,
            generated_token_count_override=generated_count_override,
        )
        append_jsonl(path, row)
        self._missing_kv_per_sample_generation_index += 1

    def _generation_wall_timer_obj(self) -> GenerationWallTimer:
        config = getattr(self.model, "config", None)
        enabled = bool(getattr(config, "kv_generation_timing_enabled", False))
        existing = getattr(self, "_generation_wall_timer", None)
        if existing is None or existing.enabled != enabled:
            existing = GenerationWallTimer(enabled=enabled)
            self._generation_wall_timer = existing
        return existing

    def _reset_generation_wall_timer(self) -> None:
        self._generation_wall_timer_obj().reset()

    def _generation_timing_summary(self) -> Dict[str, Any]:
        return self._generation_wall_timer_obj().to_summary()

    def _maybe_write_pure_recovery_cost_sidecar(self) -> None:
        """Persist the PURE_MISSING_KV_RECOVERY_COMPUTE_COST shadow summary.
        Finalizes the dedicated pure-recovery component timer exactly once so
        every deferred CUDA event pair (and each event's on_resolved per-event
        timing) is resolved before aggregation."""

        config = getattr(self.model, "config", None)
        if not bool(getattr(config, "kv_pure_recovery_cost_enabled", False)):
            return
        decoder = self._missing_kv_decoder()
        if decoder is None or not hasattr(decoder, "pure_recovery_cost_summary"):
            return
        timer = decoder.__dict__.get("pure_recovery_component_timer")
        if timer is not None:
            timer.finalize()
        summary = decoder.pure_recovery_cost_summary()
        output_path = getattr(config, "kv_pure_recovery_cost_output", None)
        if output_path:
            write_json_atomic(output_path, summary)

    def _maybe_write_generation_timing_sidecar(self) -> None:
        """Persist the existing GenerationWallTimer summary independently of
        kv_exact_catchup_overhead_enabled -- kv_generation_timing_output is
        documented as independent of that flag (see AdditionalArguments),
        so a caller wanting only generation timing must not be forced to
        also enable the much heavier exact-catchup overhead profiler."""

        config = getattr(self.model, "config", None)
        generation_timing_output = getattr(config, "kv_generation_timing_output", None)
        if not generation_timing_output:
            return
        write_json_atomic(generation_timing_output, self._generation_timing_summary())

    def _maybe_write_exact_catchup_overhead_sidecars(self) -> None:
        config = getattr(self.model, "config", None)
        if not bool(getattr(config, "kv_exact_catchup_overhead_enabled", False)):
            return
        decoder = self._missing_kv_decoder()
        if decoder is None or not hasattr(decoder, "exact_catchup_overhead_summary"):
            return
        generation_timing_summary = self._generation_timing_summary()
        aggregate = decoder.exact_catchup_overhead_summary(generation_timing_summary=generation_timing_summary)
        events = decoder.exact_catchup_overhead_events()
        event_output = getattr(config, "kv_exact_catchup_event_output", None)
        if event_output:
            write_events_jsonl_atomic(event_output, events)
        summary_output = getattr(config, "kv_exact_catchup_summary_output", None)
        if summary_output:
            write_json_atomic(summary_output, aggregate)
        csv_dir = getattr(config, "kv_exact_catchup_csv_dir", None)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_overall.csv"),
                {"overall": aggregate["overall"]},
                key_column="scope",
                table="overall",
            )
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_by_runtime_path.csv"),
                aggregate["by_runtime_path"],
                key_column="runtime_path",
                table="overall",
            )
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_by_source.csv"),
                aggregate["by_source_layer"],
                key_column="source_layer",
                table="overall",
            )
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_by_gap.csv"),
                aggregate["by_gap"],
                key_column="gap_bin",
                table="by_gap",
            )
            # Runtime-path-stratified breakdowns (Major 2): candidate and
            # fixed-source timing populations are never placed in the same
            # row without a runtime_path-prefixed key.
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_by_runtime_path_and_source.csv"),
                flatten_runtime_path_stratified_rows(aggregate.get("by_runtime_path_and_source_layer", {})),
                key_column="runtime_path_and_source_layer",
                table="overall",
            )
            write_aggregate_csv_atomic(
                os.path.join(csv_dir, "exact_catchup_by_runtime_path_and_gap.csv"),
                flatten_runtime_path_stratified_rows(aggregate.get("by_runtime_path_and_gap", {})),
                key_column="runtime_path_and_gap",
                table="by_gap",
            )
        require_complete = bool(getattr(config, "kv_exact_catchup_require_complete_events", False))
        if require_complete and not aggregate.get("run_valid"):
            raise ValueError(
                "kv_exact_catchup_require_complete_events is set but the exact-catchup overhead "
                "run is invalid: {}".format(aggregate.get("cross_validation", {}).get("errors"))
            )

    @staticmethod
    def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _maybe_write_missing_kv_sidecars(self, summary: Optional[Dict[str, Any]]) -> None:
        if summary is None:
            return
        config = getattr(self.model, "config", None)
        accounting_output = getattr(config, "kv_runtime_accounting_output", None)
        if accounting_output:
            self._write_json_file(accounting_output, summary)
        timing_output = getattr(config, "kv_runtime_component_timing_output", None)
        if timing_output:
            self._write_json_file(timing_output, summary.get("timing", {}))
        f2a_summary_output = getattr(config, "kv_f2a_summary_output", None)
        decoder = self._missing_kv_decoder()
        if f2a_summary_output and decoder is not None and hasattr(decoder, "f2a_summary"):
            self._write_json_file(f2a_summary_output, decoder.f2a_summary())

    def _missing_kv_provenance_enabled(self) -> bool:
        config = getattr(self.model, "config", None)
        return bool(getattr(config, "missing_kv_provenance_enabled", False))

    def _prepare_missing_kv_generation_binding_output(self) -> None:
        if not self._missing_kv_provenance_enabled():
            return
        config = getattr(self.model, "config", None)
        path = getattr(config, "missing_kv_generation_binding_output", None)
        if not path:
            raise ValueError("missing_kv_generation_binding_output is required when provenance is enabled")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8"):
            pass
        self._missing_kv_trainer_generation_index = 0

    def _validate_missing_kv_provenance_eval_context(self) -> None:
        if not self._missing_kv_provenance_enabled():
            return
        if int(getattr(self.args, "eval_batch_size", 0) or 0) != 1:
            raise ValueError("missing-KV provenance requires eval batch size 1")
        if int(getattr(self.args, "world_size", 1) or 1) != 1:
            raise ValueError("missing-KV provenance currently supports only single-process evaluation")
        if not self._missing_kv_population_records_by_order:
            raise ValueError("missing-KV effective population records are required when provenance is enabled")

    def _maybe_write_generation_binding_summary(self) -> None:
        if not self._missing_kv_provenance_enabled():
            return
        config = getattr(self.model, "config", None)
        binding_path = getattr(config, "missing_kv_generation_binding_output", None)
        summary_path = getattr(config, "missing_kv_generation_binding_summary_output", None)
        if not binding_path or not summary_path:
            return
        rows = read_jsonl(binding_path) if os.path.exists(binding_path) else []
        summary = validate_generation_bindings(self.missing_kv_effective_population_records or [], rows)
        write_json_file(summary_path, summary)

    def _decoder_from_model(self, model):
        if getattr(getattr(model, "config", None), "use_lora", False) and hasattr(model, "base_model"):
            model = model.base_model
        return getattr(model, "decoder", None)

    def _extract_missing_kv_sample_context(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Optional[Dict[str, Any]]:
        if not self._missing_kv_provenance_enabled():
            return None
        if "missing_kv_selected_order" not in inputs:
            raise ValueError("missing_kv_selected_order missing from eval batch")
        selected_order = inputs.pop("missing_kv_selected_order")
        if isinstance(selected_order, torch.Tensor):
            flat = selected_order.detach().cpu().reshape(-1).tolist()
        elif isinstance(selected_order, (list, tuple)):
            flat = list(selected_order)
        else:
            flat = [selected_order]
        if len(flat) != 1:
            raise ValueError("missing-KV provenance requires exactly one selected_order per generation batch")
        order = int(flat[0])
        if order not in self._missing_kv_population_records_by_order:
            raise ValueError("selected_order not found in effective population: {}".format(order))
        return dict(self._missing_kv_population_records_by_order[order])

    def _set_missing_kv_sample_context(self, model, context: Optional[Dict[str, Any]]) -> bool:
        if context is None:
            return False
        decoder = self._decoder_from_model(model)
        if decoder is not None and hasattr(decoder, "set_missing_kv_generation_sample_context"):
            decoder.set_missing_kv_generation_sample_context(context)
            return False
        if not self._can_record_full_reference_binding_without_decoder_context(model):
            raise ValueError("model decoder does not support missing-KV generation sample context")
        return True

    @staticmethod
    def _can_record_full_reference_binding_without_decoder_context(model) -> bool:
        if isinstance(model, (DeployT5ForConditionalGeneration, DeployLongT5ForConditionalGeneration)):
            return False
        config = getattr(model, "config", None)
        if config is None or getattr(config, "static_exit_layer", None) is not None:
            return False
        context_dependent_flags = (
            "use_early_exit",
            "use_shallow_deep",
            "kv_runtime_restoration_enabled",
            "kv_f2a_frozen_schedule_enabled",
            "kv_trace_enabled",
            "kv_calm_counterfactual_trace_enabled",
            "kv_exact_catchup_dump_enabled",
            "kv_source_dump_enabled",
            "kv_adjacent_anchor_dump_enabled",
            "kv_all_layer_calib_dump_enabled",
            "kv_all_layer_hidden_dump_enabled",
            "kv_attention_diag_dump_enabled",
            "kv_full_attention_diag_dump_enabled",
            "kv_restoration_dryrun_enabled",
        )
        return not any(bool(getattr(config, name, False)) for name in context_dependent_flags)

    def _record_full_reference_generation_binding(self, context: Dict[str, Any]) -> None:
        config = getattr(self.model, "config", None)
        output_path = getattr(config, "missing_kv_generation_binding_output", None)
        if not output_path:
            raise ValueError("missing_kv_generation_binding_output is required when provenance is enabled")
        generation_index = int(getattr(self, "_missing_kv_trainer_generation_index", 0))
        append_jsonl(
            output_path,
            {
                "manifest_schema_version": 1,
                "stable_sample_id": context.get("stable_sample_id"),
                "selected_order": int(context.get("selected_order")),
                "generation_index": generation_index,
                "raw_dataset_index": int(context.get("raw_dataset_index")),
                "dataset_provided_id": context.get("dataset_provided_id"),
            },
        )
        self._missing_kv_trainer_generation_index = generation_index + 1

    def _clear_missing_kv_sample_context(self, model) -> None:
        decoder = self._decoder_from_model(model)
        if decoder is not None and hasattr(decoder, "clear_missing_kv_generation_sample_context"):
            decoder.clear_missing_kv_generation_sample_context()

    def _finalize_missing_kv_generation_dump(self, model) -> None:
        decoder = self._decoder_from_model(model)
        if decoder is not None and hasattr(decoder, "finalize_missing_kv_generation_dump"):
            decoder.finalize_missing_kv_generation_dump()

    def _abort_missing_kv_generation_dump(self, model) -> None:
        decoder = self._decoder_from_model(model)
        if decoder is not None and hasattr(decoder, "abort_missing_kv_generation_dump"):
            decoder.abort_missing_kv_generation_dump(reason="generation_exception")

    @staticmethod
    def _effective_token_ids(token_ids, pad_token_id=None) -> List[int]:
        return SumTrainer._effective_label_token_ids(token_ids, pad_token_id)

    @staticmethod
    def _effective_label_token_ids(token_ids, pad_token_id=None) -> List[int]:
        values: List[int] = []
        for token_id in list(token_ids):
            value = int(token_id)
            if value == -100:
                continue
            if pad_token_id is not None and value == int(pad_token_id):
                break
            values.append(value)
        return values

    @staticmethod
    def _special_token_id(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _effective_generated_token_ids(
        token_ids,
        *,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
    ) -> List[int]:
        values = [int(token_id) for token_id in list(token_ids) if int(token_id) != -100]
        pad_id = SumTrainer._special_token_id(pad_token_id)
        eos_id = SumTrainer._special_token_id(eos_token_id)
        decoder_start_id = SumTrainer._special_token_id(decoder_start_token_id)

        while values and pad_id is not None and values[-1] == pad_id:
            values.pop()

        leading_start_ids = set()
        if decoder_start_id is not None:
            leading_start_ids.add(decoder_start_id)
        elif pad_id is not None:
            leading_start_ids.add(pad_id)
        if values and values[0] in leading_start_ids:
            values = values[1:]

        if eos_id is not None:
            for index, value in enumerate(values):
                if value == eos_id:
                    values = values[:index]
                    break
        return values

    @staticmethod
    def _token_digest_for_effective_ids(token_ids) -> str:
        values = [int(value) for value in list(token_ids)]
        text = ",".join(str(value) for value in values)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _token_digest(self, token_ids, pad_token_id=None, *, effective_token_ids=None) -> str:
        if effective_token_ids is None:
            effective_token_ids = self._effective_token_ids(token_ids, pad_token_id)
        return self._token_digest_for_effective_ids(effective_token_ids)

    def _eval_prediction_context_for_index(self, sample_index: int, eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        config = getattr(self.model, "config", None)
        provenance_enabled = bool(getattr(config, "missing_kv_provenance_enabled", False))
        selected_order = None
        if eval_dataset is not None:
            try:
                row = eval_dataset[int(sample_index)]
                if isinstance(row, dict):
                    selected_order = row.get("missing_kv_selected_order")
            except Exception:
                selected_order = None
        if provenance_enabled and selected_order is None:
            raise ValueError("missing_kv_selected_order missing from eval prediction row")
        if selected_order is None:
            selected_order = sample_index
        population_by_order = getattr(self, "_missing_kv_population_records_by_order", {}) or {}
        population = population_by_order.get(int(selected_order))
        if provenance_enabled and not population:
            raise ValueError("selected_order not found in effective population: {}".format(selected_order))
        population_records = getattr(self, "missing_kv_effective_population_records", None)
        if not provenance_enabled and not population and population_records is not None:
            try:
                population = population_records[int(sample_index)]
            except Exception:
                population = None
        if not population:
            return {}
        return {
            key: population.get(key)
            for key in (
                "stable_sample_id",
                "selected_order",
                "raw_dataset_index",
                "dataset_provided_id",
                "reference_text_sha256",
                "tokenized_label_sha256",
                "generation_sample_binding_sha256",
            )
            if population.get(key) is not None
        }

    def _maybe_write_eval_predictions(self, all_preds, all_labels, metric_key_prefix: str, eval_dataset: Optional[Dataset] = None) -> None:
        if metric_key_prefix != "eval":
            return
        config = getattr(self.model, "config", None)
        if not bool(getattr(config, "save_eval_predictions", False)):
            return
        if all_preds is None or all_labels is None or self.tokenizer is None:
            return
        output_path = getattr(config, "eval_predictions_output", None)
        if not output_path:
            output_path = os.path.join(self.args.output_dir, "eval_predictions.jsonl")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        preds = np.asarray(all_preds)
        labels = np.asarray(all_labels)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(config, "eos_token_id", None)
        decoder_start_token_id = getattr(config, "decoder_start_token_id", None)
        decode_labels = labels
        if pad_token_id is not None:
            decode_labels = np.where(labels != -100, labels, pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(decode_labels, skip_special_tokens=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for sample_index, (pred_ids, label_ids, pred_text, label_text) in enumerate(
                zip(preds, labels, decoded_preds, decoded_labels)
            ):
                effective_prediction_ids = self._effective_generated_token_ids(
                    pred_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    decoder_start_token_id=decoder_start_token_id,
                )
                row = {
                    "sample_index": sample_index,
                    "prediction_text": pred_text.strip(),
                    "reference_text": label_text.strip(),
                    "prediction_token_sha256": self._token_digest(
                        pred_ids,
                        effective_token_ids=effective_prediction_ids,
                    ),
                    "reference_token_sha256": self._token_digest(label_ids, pad_token_id),
                    "generated_length": len(effective_prediction_ids),
                }
                row.update(self._eval_prediction_context_for_index(sample_index, eval_dataset))
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self._validate_missing_kv_provenance_eval_context()
        self._prepare_missing_kv_generation_binding_output()
        self._prepare_missing_kv_per_sample_accounting_output()
        self._begin_missing_kv_evaluation()
        self._reset_generation_wall_timer()
        try:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
                start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            # average block layers
            if self.model.decoder.use_shallow_deep:
                total, deep = self.model.decoder.block_op[0], self.model.decoder.block_op[self.model.decoder.shallow_exit_layer]
                shallow = total - deep

                # self.model.rollback_num: we should consider redundant operations due to rollback
                block_op_metric = {'{}_block_avg'.format(metric_key_prefix): (deep * len(self.model.decoder.block_op) \
                    + (shallow + self.model.rollback_num) * self.model.decoder.shallow_exit_layer) / (total + 1e-10)}
                # 'block_num' contains the number of token for [shallow, deep, parallel, rollback]
                block_op_metric['{}_block_num'.format(metric_key_prefix)] = str([shallow, deep, self.model.decoder.parallel_tokens_shallow, self.model.decoder.parallel_tokens_deep, self.model.rollback_num])
            else:
                block_op_metric = {'{}_block_avg'.format(metric_key_prefix): sum(self.model.decoder.block_op) / (self.model.decoder.block_op[0] + 1e-10),}
            output.metrics.update(block_op_metric)

            # deploy time
            if self.model.deploy_time is not None:
                deploy_time = {}
                for k, v in self.model.deploy_time.items():
                    if type(v) != list: deploy_time[k] = str(v).split('.')[0]
                    else: deploy_time[k] = str([str(_v).split('.')[0] for _v in v])
                output.metrics.update(deploy_time)

            output.metrics.update(self._missing_kv_accounting_metrics(metric_key_prefix))
            self._maybe_write_missing_kv_sidecars(self._missing_kv_accounting_summary())
            self._maybe_write_pure_recovery_cost_sidecar()
            self._maybe_write_generation_timing_sidecar()
            self._maybe_write_exact_catchup_overhead_sidecars()
            self._maybe_write_generation_binding_summary()

            self.log(output.metrics)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

            return output.metrics
        finally:
            self._end_missing_kv_evaluation()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        self._maybe_write_eval_predictions(all_preds, all_labels, metric_key_prefix, eval_dataset)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        inputs = dict(inputs)
        sample_context = self._extract_missing_kv_sample_context(inputs)
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # TODO (Joao): the following line is needed to keep a consistent result on SQUAD. Ideally, we should not block
        # users from preparing a dataset with `decoder_input_ids`.
        inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        # generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        
        gen_model = self.model.base_model if self.model.config.use_lora else self.model
        trainer_binding_required = self._set_missing_kv_sample_context(gen_model, sample_context)
        accounting_before = self._missing_kv_per_sample_accounting_snapshot()
        try:
            with self._generation_wall_timer_obj().time_generation_call(device=inputs["input_ids"].device):
                generated_tokens = gen_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs)  # Decoder input shape: (batch_size, 1)
        except Exception:
            self._abort_missing_kv_generation_dump(gen_model)
            self._write_missing_kv_per_sample_accounting_row(
                context=sample_context,
                before=accounting_before,
                status="failed",
            )
            raise
        else:
            self._finalize_missing_kv_generation_dump(gen_model)
            if trainer_binding_required:
                self._record_full_reference_generation_binding(sample_context)
            self._write_missing_kv_per_sample_accounting_row(
                context=sample_context,
                before=accounting_before,
                status="ok",
                generated_tokens=generated_tokens,
            )
        finally:
            self._clear_missing_kv_sample_context(gen_model)
        
        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        if isinstance(self.model, DeployT5ForConditionalGeneration) or isinstance(self.model, DeployLongT5ForConditionalGeneration):
            loss = None
        else:
            with torch.no_grad():
                if has_labels:
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None
            
        return (loss, generated_tokens, labels)
