"""
T5: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L19
"""
from typing import Optional, Tuple, Union, List, Callable

import os
import copy
import math
import time
import datetime
import warnings
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions, 
    Seq2SeqLMOutput
)
from transformers.models.longt5.modeling_longt5 import (
    LongT5LayerNorm,
    LongT5LayerLocalSelfAttention,
    LongT5LayerTransientGlobalSelfAttention,
    LongT5LayerFF,
    LongT5Block,
    LongT5Stack,
    LongT5ForConditionalGeneration,
    _get_local_attention_mask,
)
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput
except ImportError:
    from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

    GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
    GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.utils import logging

from .deploying_t5 import DeployT5LayerSelfAttention
from .deploying_t5 import DeployT5LayerCrossAttention
from .deploying_t5 import DeployT5Stack
from our_kv_restoration import RuntimeKVRestorationManager
from our_kv_restoration.early_exit_exact_cache_calibration import (
    EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION,
    SOURCE_LAYER_MODE_FIXED,
    SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
    ExactCacheCalibrationCollector,
)
from our_kv_restoration.kv_trace import KVTraceRecorder, infer_decoder_position, safe_cache_seq_len
from our_kv_restoration.missing_kv_calm_trace import (
    CALM_THRESHOLD,
    compute_calm_candidate_logits_and_confidence,
)
from our_kv_restoration.runtime_kv_restoration import EXACT_CATCHUP_METHOD, PHASE3C_RUNTIME_MODE
from our_kv_restoration.missing_kv_dump_provenance import append_jsonl as append_provenance_jsonl
from our_kv_restoration.position_bookkeeping import make_skipped_token_metadata
from util import (
    get_skip_mask,
    BetaMixture1D,
)

logger = logging.get_logger(__name__)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]

class DeployLongT5Block(LongT5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.is_decoder = config.is_decoder
        if config.is_decoder:
            attention_layer = DeployT5LayerSelfAttention
        elif config.encoder_attention_type == "local":
            attention_layer = LongT5LayerLocalSelfAttention
        elif config.encoder_attention_type == "transient-global":
            attention_layer = LongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {config.encoder_attention_type}."
            )
            
        self.layer = nn.ModuleList()
        self.layer.append(attention_layer(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(DeployT5LayerCrossAttention(config))

        self.layer.append(LongT5LayerFF(config))

    def gen_cross_attn_key_value(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        r""" 
        In Shallow-Deep framework, if all previous tokens, including <start> token, have skipped Deep decoder,
        generate cross-attn key_values only ONCE because they are shared for all sequence.
        
        return (None, None) + cross_attn_past_key_value: Tuple[torch.Tensor] (length of 2)
        """

        # if all previous tokens, including <start> token, have skipped Deep decoder
        assert self.is_decoder and encoder_hidden_states is not None
        cross_attn_past_key_value = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            gen_cross_attn_key_value=True,
        )
        self.key_value_gen_time = self.layer[1].key_value_gen_time

        past_key_value = [None, None,] + cross_attn_past_key_value
        return past_key_value

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        skip_mask=False,
        parallel_mask=False,
        stack_hidden_states=None,
        layer_idx=None,
        kv_importance_tracker=None,
    ):
        """``layer_idx``/``kv_importance_tracker`` are behavior-neutral
        interface-compatibility parameters only. They exist so this block can
        be driven through the same calling convention the shared
        ``DeployT5LayerSelfAttention`` already accepts (the T5 fixed-layer
        calibration finalization primitive passes both). Their defaults are
        ``None``, which is exactly the previous behavior; no H2O importance
        tracking is ported or initialized here."""

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skip_mask=skip_mask,
            stack_hidden_states=stack_hidden_states,
            layer_idx=layer_idx,
            kv_importance_tracker=kv_importance_tracker,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None
            
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=skip_mask,
                parallel_mask=parallel_mask,
            )
            hidden_states = cross_attention_outputs[0]

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        # Apply Feed Forward layer
        if not skip_mask:
            hidden_states = self.layer[-1](hidden_states)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder:
            self.ffn_time = datetime.datetime.now() - start
            self.key_value_gen_time = (self.layer[0].key_value_gen_time, self.layer[1].key_value_gen_time)
            self.attn_time = (self.layer[0].attn_ffn_time, self.layer[1].attn_ffn_time)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class DeployLongT5Stack(LongT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        self.block = nn.ModuleList(
            [DeployLongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.device_map = None
        self.gradient_checkpointing = False

        # Early-Exit framework
        self.use_early_exit = config.use_early_exit
        self.exit_min_layer = config.exit_min_layer
            
        # Shallow-Deep framework
        self.use_shallow_deep = config.use_shallow_deep
        self.shallow_exit_layer = config.shallow_exit_layer
        if self.is_decoder and config.use_shallow_deep:
            assert config.shallow_exit_layer > 0 and config.shallow_exit_layer < len(self.block)
        
        # Synchronized Parallel Decoding
        self.block_op = [0] * config.num_layers  # to calculate the average number of forward block layers
        self.parallel_tokens_shallow = 0  # how much tokens are used in parallel decoding as stack_hidden_states
        self.parallel_tokens_deep = 0  # how much tokens are used in parallel decoding with skip_mask = False
        self.stack_hidden_states = ()  # store hidden_states that do not forward Deep decoder

        # Native FREE fixed-source-layer exact-cache calibration (observer
        # only). These pending tuples stay index-aligned with
        # stack_hidden_states; any drift fails closed. Mirrors the narrow
        # DeployT5Stack fixed-layer state -- deliberately WITHOUT porting
        # H2O/F2a/Task-C2 state.
        self.stack_hidden_metadata = ()
        self.stack_fixed_layer_calibration_records = ()
        self._missing_kv_generation_sample_context = None
        # -1 so the FIRST real generation reset increments it to 0, matching
        # the accepted DeployT5Stack runtime numbering exactly.
        self._generation_index = -1
        # Fixed-layer Phase-3c Task C1 (exact-catch-up-then-overwrite) state,
        # ported minimally from DeployT5Stack: the SAME
        # RuntimeKVRestorationManager, plan/apply/accounting implementations
        # are reused as unbound aliases below -- never a second restoration
        # implementation. stack_phase3c_source_hidden_records stays
        # index-aligned with stack_hidden_states (one entry per pending
        # shallow exit; None when restoration is disabled). The runtime
        # source layer is resolved from config.shallow_exit_layer by the
        # manager itself -- nothing LongT5-specific is hard-coded here.
        self.stack_phase3c_source_hidden_records = ()
        self.kv_runtime_restorer = None
        if self.is_decoder and getattr(config, "kv_runtime_restoration_enabled", False):
            self.kv_runtime_restorer = RuntimeKVRestorationManager.from_path(
                getattr(config, "kv_runtime_restoration_artifact", None),
                getattr(config, "kv_runtime_restoration_method", "source_procrustes"),
                threshold=getattr(config, "kv_runtime_restoration_threshold", None),
                model_config=config,
            )
        self._reset_runtime_restoration_generation_state()
        # Behavior-neutral compatibility state for the aliased DeployT5Stack
        # Official CALM Task C1 transaction (which records through kv_trace
        # and passes kv_importance into block calls). Same disabled-by-default
        # KVTraceRecorder construction as DeployT5Stack; kv_importance stays
        # None -- H2O importance tracking is NOT ported, and the shared
        # DeployT5LayerSelfAttention treats None as "no tracking" exactly as
        # the existing LongT5 layer calls (which omit it) already do.
        self.kv_trace = KVTraceRecorder(
            enabled=self.is_decoder and getattr(config, "kv_trace_enabled", False),
            max_records=getattr(config, "kv_trace_max_records", 100000),
        )
        self.kv_importance = None
        self.exact_cache_calibration_collector = None
        if self.is_decoder and bool(
            getattr(config, "kv_early_exit_exact_cache_calibration_enabled", False)
        ):
            collector = ExactCacheCalibrationCollector.from_config(config)
            # Patch A implemented ONLY Native FREE fixed-source-layer
            # exact-cache calibration; Patch 1 additionally connects the
            # existing Official FREE CALM first-crossing collector (whose
            # events are staged by the aliased DeployT5Stack Task C1
            # transaction below, never by the fixed-layer plumbing). Any
            # OTHER mode -- including the historical T5-only
            # candidate_first_crossing (4,6,8,10) policy -- still has event
            # semantics this model-side plumbing does not implement, so it
            # must keep failing loudly at construction rather than silently
            # running through it. The shared collector is not modified.
            collector_mode = getattr(collector, "source_layer_mode", None)
            if collector_mode not in (
                SOURCE_LAYER_MODE_FIXED,
                SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
            ):
                raise NotImplementedError(
                    "longt5_exact_cache_calibration_supports_only_fixed_or_official_free_calm_source_layer_mode:"
                    "got {}".format(collector_mode)
                )
            self.exact_cache_calibration_collector = collector

        # Adaptive Threshold Estimator
        self.bmm_model = BetaMixture1D()
        self.bmm_threshold = None
        self.stack_conf, self.stack_pred = (), ()
        self.stack_conf_all, self.stack_ident_all = (), ()
        
        if self.is_decoder:
            self._reset_time_measure()
        else: self.deploy_time = None
        
    def _reset_time_measure(self):
        self.deploy_time = {'time_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_ffn': datetime.timedelta(),
                            'time_confidence': datetime.timedelta(),
                            'time_exit_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_exit_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_exit_ffn': datetime.timedelta(),
                            'time_parallel_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_parallel_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_parallel_ffn': datetime.timedelta(),
                            'time_others': datetime.timedelta(),}

    # ------------------------------------------------------------------
    # Native FREE fixed-source-layer exact-cache calibration (observer only)
    #
    # Every method below mirrors the corresponding narrow DeployT5Stack
    # method one-for-one. The collector, its tensor schema, its fitting
    # semantics and the Phase-3c math are reused unchanged; nothing here
    # runs when kv_early_exit_exact_cache_calibration_enabled is False.
    # ------------------------------------------------------------------

    def set_missing_kv_generation_sample_context(self, context):
        """Mirrors DeployT5Stack.set_missing_kv_generation_sample_context.
        SumTrainer already feature-detects this method for both model
        families, so no trainer change is needed."""

        self._missing_kv_generation_sample_context = dict(context or {})

    def clear_missing_kv_generation_sample_context(self):
        self._missing_kv_generation_sample_context = None

    def _missing_kv_provenance_enabled(self):
        return bool(getattr(self.config, "missing_kv_provenance_enabled", False))

    def _missing_kv_sample_context_fields(self):
        if not self._missing_kv_provenance_enabled():
            return {}
        context = self._missing_kv_generation_sample_context
        if not context:
            return {
                "stable_sample_id": None,
                "selected_order": None,
                "raw_dataset_index": None,
                "dataset_provided_id": None,
                "missing_kv_sample_context_status": "missing",
            }
        return {
            "stable_sample_id": context.get("stable_sample_id"),
            "selected_order": context.get("selected_order"),
            "raw_dataset_index": context.get("raw_dataset_index"),
            "dataset_provided_id": context.get("dataset_provided_id"),
            "missing_kv_sample_context_status": "ok",
        }

    def _record_missing_kv_generation_binding(self):
        """Mirrors DeployT5Stack._record_missing_kv_generation_binding.

        SumTrainer feature-detects set_missing_kv_generation_sample_context()
        and therefore expects the deployment decoder to write its own
        generation binding. Same row fields and the same existing append
        helper as T5 -- no new provenance format."""

        if not (self.is_decoder and self._missing_kv_provenance_enabled()):
            return
        output_path = getattr(self.config, "missing_kv_generation_binding_output", None)
        if not output_path:
            raise ValueError("missing_kv_generation_binding_output is required when provenance is enabled")
        context = self._missing_kv_generation_sample_context
        if not context:
            raise ValueError("missing-KV provenance sample context is unavailable at generation reset")
        row = {
            "manifest_schema_version": 1,
            "stable_sample_id": context.get("stable_sample_id"),
            "selected_order": int(context.get("selected_order")),
            "generation_index": int(self._generation_index),
            "raw_dataset_index": int(context.get("raw_dataset_index")),
            "dataset_provided_id": context.get("dataset_provided_id"),
        }
        append_provenance_jsonl(output_path, row)

    def abort_missing_kv_generation_dump(self, reason="generation_failed"):
        """Mirrors the DeployT5Stack method SumTrainer calls on a generation
        exception, narrowed to Patch A scope: only the existing collector's
        pending event is cleared, so a failed generation cannot leave a stale
        pending event behind. No packed-generation/Task-C1/trace/H2O/F2a/
        runtime-restoration state is ported."""

        del reason
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is not None:
            collector.abort_pending()

    def _exact_cache_calibration_enabled(self):
        return getattr(self, "exact_cache_calibration_collector", None) is not None

    def _fixed_layer_exact_cache_calibration_active(self):
        collector = getattr(self, "exact_cache_calibration_collector", None)
        return collector is not None and getattr(collector, "source_layer_mode", None) == SOURCE_LAYER_MODE_FIXED

    def _infer_decoder_position_from_source_present_kv(self, present_key_value_states, source_layer):
        """Mirrors DeployT5Stack._infer_decoder_position_from_source_present_kv.

        infer_decoder_position(past_key_values) has nothing to read at the
        very first decoder position (no prior cache length yet). The last
        already-exact layer's OWN present K/V, produced during this same
        forward call, already carries that position."""

        if source_layer is None:
            return None
        try:
            source_layer = int(source_layer)
        except (TypeError, ValueError):
            return None
        if source_layer < 0:
            return None
        try:
            if present_key_value_states is None or len(present_key_value_states) <= source_layer:
                return None
            source_state = present_key_value_states[source_layer]
            if source_state is None or len(source_state) < 1:
                return None
            key_tensor = source_state[0]
            shape = getattr(key_tensor, "shape", None)
            if shape is None or len(shape) < 3:
                return None
            seq_len = int(shape[2])
            if seq_len <= 0:
                return None
            return seq_len - 1
        except (TypeError, IndexError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Fixed-layer Phase-3c Task C1 (exact-catch-up-then-overwrite): the
    # production DeployT5Stack implementations are reused VERBATIM as
    # unbound aliases -- one restoration implementation, never a LongT5
    # rewrite. They resolve module-level names (PHASE3C_RUNTIME_MODE,
    # safe_cache_seq_len, MissingKVRuntimeAccounting, ...) in
    # deploying_t5's own namespace, and every `self.*` attribute they touch
    # (config, block, kv_runtime_restorer, stack_phase3c_source_hidden_
    # records, _kv_runtime_restoration_counters, lazily-created accounting/
    # component-timer objects) exists identically on this stack. The decoder
    # blocks' layer[0] is the SAME DeployT5LayerSelfAttention class, so the
    # same-layer native LayerNorm/W_K/W_V projection and the learned
    # strict-deeper restore paths inside restore_from_hidden() operate on
    # identical module interfaces. Source layer and target range are
    # resolved dynamically (config.shallow_exit_layer / len(self.block)) --
    # no LongT5-specific restoration math anywhere.
    # ------------------------------------------------------------------
    _runtime_restoration_enabled = DeployT5Stack._runtime_restoration_enabled
    _phase3c_runtime_restoration_enabled = DeployT5Stack._phase3c_runtime_restoration_enabled
    _f2a_enabled = DeployT5Stack._f2a_enabled
    _json_safe = DeployT5Stack._json_safe
    _make_phase3c_source_hidden_record = DeployT5Stack._make_phase3c_source_hidden_record
    _maybe_make_phase3c_source_hidden_record_for_skip = (
        DeployT5Stack._maybe_make_phase3c_source_hidden_record_for_skip
    )
    _phase3c_source_hidden_record_trace = DeployT5Stack._phase3c_source_hidden_record_trace
    _runtime_restoration_plan = DeployT5Stack._runtime_restoration_plan
    _stage_runtime_restored_kv_slices = DeployT5Stack._stage_runtime_restored_kv_slices
    _commit_runtime_restored_kv_slices = DeployT5Stack._commit_runtime_restored_kv_slices
    _load_runtime_source_kv_from_record = DeployT5Stack._load_runtime_source_kv_from_record
    _apply_runtime_restoration_to_present_kv = DeployT5Stack._apply_runtime_restoration_to_present_kv
    _missing_kv_accounting_obj = DeployT5Stack._missing_kv_accounting_obj
    _missing_kv_component_timer_obj = DeployT5Stack._missing_kv_component_timer_obj
    _reset_runtime_restoration_generation_state = DeployT5Stack._reset_runtime_restoration_generation_state
    # Paper-facing accounting export + evaluation/generation lifecycle: the
    # SAME production surface SumTrainer feature-detects on DeployT5Stack
    # (begin/end_missing_kv_evaluation, missing_kv_runtime_accounting_
    # summary/metrics), reused verbatim. Every `self.*` dependency is either
    # already present here (config, accounting/timer objects via the aliased
    # lazy getters, _reset_runtime_restoration_generation_state) or resolved
    # through getattr defaults (_missing_kv_evaluation_active, the F2a step
    # context -- inert because _f2a_enabled() is never true on LongT5).
    _missing_kv_exact_catchup_overhead_recorder_obj = (
        DeployT5Stack._missing_kv_exact_catchup_overhead_recorder_obj
    )
    _configure_missing_kv_accounting_policy = DeployT5Stack._configure_missing_kv_accounting_policy
    missing_kv_runtime_accounting_summary = DeployT5Stack.missing_kv_runtime_accounting_summary
    missing_kv_runtime_accounting_metrics = DeployT5Stack.missing_kv_runtime_accounting_metrics
    reset_missing_kv_evaluation_aggregates = DeployT5Stack.reset_missing_kv_evaluation_aggregates
    begin_missing_kv_evaluation = DeployT5Stack.begin_missing_kv_evaluation
    end_missing_kv_evaluation = DeployT5Stack.end_missing_kv_evaluation
    _begin_missing_kv_generation = DeployT5Stack._begin_missing_kv_generation
    _snapshot_f2a_generation_step_context = DeployT5Stack._snapshot_f2a_generation_step_context
    _restore_f2a_generation_step_context = DeployT5Stack._restore_f2a_generation_step_context
    _reset_runtime_restoration_counters = DeployT5Stack._reset_runtime_restoration_counters
    _is_calm_taskc1_runtime_enabled = DeployT5Stack._is_calm_taskc1_runtime_enabled
    # ------------------------------------------------------------------
    # Official FREE CALM first-crossing Task C1 (Patch 1): the SAME
    # production DeployT5Stack exact-reference transaction, reused VERBATIM
    # as unbound aliases -- one arbitrary-source Exact implementation, never
    # a LongT5 rewrite. The transaction resolves module-level names
    # (EXACT_CATCHUP_METHOD, resolve_exact_catchup_event_decoder_position,
    # ExactCatchupOverheadEvent, infer_decoder_position, ...) in
    # deploying_t5's own namespace; every `self.*` attribute it touches
    # exists identically on this stack (config, block, kv_runtime_restorer,
    # kv_trace, kv_importance, block_op, lm_logits, _generation_index, the
    # aliased accounting/timer/overhead-recorder lazy getters and the LongT5
    # mirror of _missing_kv_sample_context_fields). The decoder blocks'
    # layer[0] is the SAME DeployT5LayerSelfAttention class and
    # DeployLongT5Block.forward already accepts the layer_idx/
    # kv_importance_tracker calling convention, so each target block s..N-1
    # is executed exactly once by the transaction itself -- the collector
    # only observes that call's own hidden/K/V (no duplicate replay).
    # Patch 2 connects the COMPLETE Official CALM four-arm
    # PURE_MISSING_KV_RECOVERY_COMPUTE_COST helper chain below as unbound
    # aliases of the SAME accepted DeployT5Stack implementation (Exact
    # timed in place on the live replay, then Ours Phase3c stacked, then
    # the State-Stacked fairness control, then conventional
    # State-Sequential). Nothing is re-implemented: source layer and
    # target range resolve dynamically from the real first crossing and
    # len(self.block) (candidates 4..11, targets s..11 on the official
    # 12-layer Multi-News checkpoint; source 11 is native-only), the
    # decoder blocks' layer[0] is the SAME DeployT5LayerSelfAttention
    # module surface the T5 helpers project through, and the shadow
    # Phase3c artifact binds through the SAME measurement-only
    # kv_pure_recovery_cost_phase3c_artifact(+_sha256) fields with the
    # SHA verified before deserialization.
    # ------------------------------------------------------------------
    _run_calm_phase3c_exact_reference_and_stage_cache = (
        DeployT5Stack._run_calm_phase3c_exact_reference_and_stage_cache
    )
    _calm_taskc1_authoritative_restoration_cache_positions = (
        DeployT5Stack._calm_taskc1_authoritative_restoration_cache_positions
    )
    _calm_taskc1_first_token_context_confirmed = (
        DeployT5Stack._calm_taskc1_first_token_context_confirmed
    )
    _calm_taskc1_evidence_prefix = DeployT5Stack._calm_taskc1_evidence_prefix
    _call_calm_phase3c_transaction_test_hook = (
        DeployT5Stack._call_calm_phase3c_transaction_test_hook
    )
    _calm_phase3c_candidate_layers = DeployT5Stack._calm_phase3c_candidate_layers
    _exact_catchup_overhead_enabled = DeployT5Stack._exact_catchup_overhead_enabled
    _exact_catchup_overhead_sample_context_fields = (
        DeployT5Stack._exact_catchup_overhead_sample_context_fields
    )
    _calm_pure_recovery_exact_target_time_block = (
        DeployT5Stack._calm_pure_recovery_exact_target_time_block
    )
    _calm_pure_recovery_begin_event = DeployT5Stack._calm_pure_recovery_begin_event
    _calm_pure_recovery_measure_shadows = DeployT5Stack._calm_pure_recovery_measure_shadows
    _CALM_PURE_RECOVERY_STATE_KEY = DeployT5Stack._CALM_PURE_RECOVERY_STATE_KEY
    _CALM_PURE_RECOVERY_STATE_STACKED_KEY = (
        DeployT5Stack._CALM_PURE_RECOVERY_STATE_STACKED_KEY
    )
    _CALM_PURE_RECOVERY_EXACT_KEY = DeployT5Stack._CALM_PURE_RECOVERY_EXACT_KEY
    _CALM_PURE_RECOVERY_OURS_KEY = DeployT5Stack._CALM_PURE_RECOVERY_OURS_KEY
    # Patch 2: the exact T5 Official CALM gate replaces the temporary
    # Patch-1a NotImplementedError scope guard -- the measurement is now
    # fully connected, so the same config predicate that governs T5
    # governs LongT5 (False on every Native FREE / disabled / non-Exact
    # configuration, exactly as before).
    _calm_pure_recovery_cost_enabled = DeployT5Stack._calm_pure_recovery_cost_enabled
    _calm_pure_recovery_events_obj = DeployT5Stack._calm_pure_recovery_events_obj
    _CalmPureRecoveryShadowConfigView = DeployT5Stack._CalmPureRecoveryShadowConfigView
    _calm_pure_recovery_phase3c_restorer_obj = (
        DeployT5Stack._calm_pure_recovery_phase3c_restorer_obj
    )
    _calm_pure_recovery_state_restorer_obj = (
        DeployT5Stack._calm_pure_recovery_state_restorer_obj
    )
    _calm_pure_recovery_learned_target_modules = (
        DeployT5Stack._calm_pure_recovery_learned_target_modules
    )
    _calm_pure_recovery_prepare_source = DeployT5Stack._calm_pure_recovery_prepare_source
    _calm_pure_recovery_run_ours_restoration = (
        DeployT5Stack._calm_pure_recovery_run_ours_restoration
    )
    _calm_pure_recovery_run_state_stacked_restoration = (
        DeployT5Stack._calm_pure_recovery_run_state_stacked_restoration
    )
    _calm_pure_recovery_run_state_restoration = (
        DeployT5Stack._calm_pure_recovery_run_state_restoration
    )
    _calm_pure_recovery_event_invalid_reasons = (
        DeployT5Stack._calm_pure_recovery_event_invalid_reasons
    )
    _calm_pure_recovery_cost_summary = DeployT5Stack._calm_pure_recovery_cost_summary
    # ------------------------------------------------------------------
    # Native FREE PURE_MISSING_KV_RECOVERY_COMPUTE_COST shadow measurement:
    # the SAME accepted DeployT5Stack implementation, reused verbatim as
    # unbound aliases. The aliased _calm_pure_recovery_cost_enabled gate is
    # False on every Native FREE configuration (use_early_exit=False), so
    # pure_recovery_cost_summary() dispatches to the Native FREE summary
    # there and to the Official CALM four-arm summary only on the Official
    # CALM + Exact live arm (Patch 2). Every self.*
    # dependency exists
    # identically on this stack: config flags, shallow_exit_layer,
    # stack_hidden_states, block[j].gen_cross_attn_key_value, the shared
    # DeployT5LayerSelfAttention modules (same-layer native + stacked
    # strictly-deeper learned restoration), get_extended_attention_mask,
    # compute_bias, the aliased begin/end_missing_kv_evaluation lifecycle
    # (which resets _pure_recovery_events and the dedicated timer), and the
    # LongT5 mirror of _missing_kv_sample_context_fields. Source layer and
    # target range resolve dynamically (shallow_exit_layer=3, targets 3..11
    # on the official Multi-News checkpoint) -- nothing is source-6-bound.
    # ------------------------------------------------------------------
    _PURE_RECOVERY_FREE_KEY = DeployT5Stack._PURE_RECOVERY_FREE_KEY
    _PURE_RECOVERY_OURS_KEY = DeployT5Stack._PURE_RECOVERY_OURS_KEY
    _pure_recovery_cost_enabled = DeployT5Stack._pure_recovery_cost_enabled
    _pure_recovery_timer_obj = DeployT5Stack._pure_recovery_timer_obj
    _pure_recovery_events_obj = DeployT5Stack._pure_recovery_events_obj
    _pure_recovery_shadow_restorer_obj = DeployT5Stack._pure_recovery_shadow_restorer_obj
    _pure_recovery_prepare_shadow = DeployT5Stack._pure_recovery_prepare_shadow
    _pure_recovery_prepare_exact_replay_inputs = (
        DeployT5Stack._pure_recovery_prepare_exact_replay_inputs
    )
    _pure_recovery_run_pending_only_exact_replay = (
        DeployT5Stack._pure_recovery_run_pending_only_exact_replay
    )
    _pure_recovery_run_ours_restoration = DeployT5Stack._pure_recovery_run_ours_restoration
    _measure_pure_missing_kv_recovery = DeployT5Stack._measure_pure_missing_kv_recovery
    _pure_recovery_validate_pending_parity = DeployT5Stack._pure_recovery_validate_pending_parity
    _pure_recovery_event_invalid_reasons = DeployT5Stack._pure_recovery_event_invalid_reasons
    pure_recovery_cost_summary = DeployT5Stack.pure_recovery_cost_summary
    _native_pure_recovery_cost_summary = DeployT5Stack._native_pure_recovery_cost_summary

    def _stage_fixed_layer_exact_cache_calibration_event(
        self,
        *,
        hidden_by_layer,
        key_value_by_layer,
        confidence,
        decoder_position,
        source_selected_token_id,
        event_origin=None,
    ):
        """Mirrors DeployT5Stack._stage_fixed_layer_exact_cache_calibration_event.

        Stages one event from data the existing FREE synchronized exact path
        already computed; performs no deep computation of its own.
        ``hidden_by_layer``/``key_value_by_layer`` must each span every
        decoder layer 0..len(self.block)-1."""

        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is None:
            return
        fixed_source_layer = int(collector.fixed_source_layer)
        complete_cache = [
            (key.detach().cpu().contiguous(), value.detach().cpu().contiguous())
            for key, value in key_value_by_layer
        ]
        target_records = [
            {
                "target_layer": layer,
                "cache_position": 0,
                "final_status": "exact_cache_committed",
            }
            for layer in range(fixed_source_layer, len(self.block))
        ]
        stage_kwargs = dict(
            sample_context=self._missing_kv_sample_context_fields(),
            generation_index=self._generation_index,
            decoder_position=int(decoder_position),
            confidence=float(confidence),
            threshold=float(getattr(self.config, "shallow2deep_conf_threshold")),
            hidden_by_layer=hidden_by_layer,
            complete_cache=complete_cache,
            target_records=target_records,
            source_selected_token_id=source_selected_token_id,
        )
        if event_origin is not None:
            stage_kwargs["event_origin"] = event_origin
        collector.stage_fixed_layer_exit(**stage_kwargs)

    def _commit_exact_cache_calibration_selected_token(self, selected_token):
        """Mirrors DeployT5Stack._commit_exact_cache_calibration_selected_token.

        The shallow exit predicted a token; the generation loop then does its
        own argmax/EOS-padding. Before that token is appended to input_ids,
        verify it is the SAME token the pending record stored, and mark the
        record committed. A mismatch fails closed rather than collecting an
        unverified event."""

        actual_token_id = None
        if self._fixed_layer_exact_cache_calibration_active():
            records = self.stack_fixed_layer_calibration_records
            if records and records[-1] is not None and not records[-1].get("selected_token_committed"):
                actual_token_id = int(selected_token.detach().cpu().reshape(-1)[0].item())
                if actual_token_id != int(records[-1]["shallow_selected_token_id"]):
                    raise ValueError("calibration_selected_token_changed_before_exact_catchup")
                records[-1]["selected_token_committed"] = True
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is None:
            return
        if actual_token_id is None:
            actual_token_id = int(selected_token.detach().cpu().reshape(-1)[0].item())
        if collector.pending_event is not None:
            collector.commit_selected_token(actual_token_id)

    def _execute_deep_target_layer(
        self,
        j,
        hidden_states,
        *,
        extended_attention_mask,
        position_bias,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        past_key_values,
        use_cache,
        output_attentions,
    ):
        """Mirrors DeployT5Stack._execute_deep_target_layer: one deep decoder
        block executed against already-shallow-computed hidden states, with
        the attention mask and relative-position bias built on first use and
        reused for later layers of the same batch.

        Used ONLY by calibration-only terminal exact finalization. It is not
        called by parallel_gen_token(), whose existing inline FREE logic
        remains the authoritative exact path and is untouched."""

        past_key_value = past_key_values[j]
        if past_key_value is None:
            past_key_value = self.block[j].gen_cross_attn_key_value(
                hidden_states,  # dummy
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[j],
                cross_attn_layer_head_mask=cross_attn_head_mask[j],
                past_key_value=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        if extended_attention_mask is None or position_bias is None:
            real_seq_length = hidden_states.shape[1]
            if past_key_value[0] is not None:
                real_seq_length += past_key_value[0].shape[2]
            key_length = real_seq_length

            if self.config.parallel_causal_mask and extended_attention_mask is None:
                attention_mask = torch.ones(hidden_states.shape[0], real_seq_length, device=hidden_states.device)
                extended_attention_mask = self.get_extended_attention_mask(
                    attention_mask, torch.Size([hidden_states.shape[0], hidden_states.shape[1]])
                )

            if position_bias is None:
                position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                    real_seq_length, key_length, device=hidden_states.device
                )
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1):, :]
                if extended_attention_mask is not None:
                    position_bias = position_bias + extended_attention_mask

        layer_outputs = self.block[j](
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            layer_head_mask=head_mask[j],
            cross_attn_layer_head_mask=cross_attn_head_mask[j],
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skip_mask=False,
            parallel_mask=True,
            stack_hidden_states=None,
            layer_idx=j,
            kv_importance_tracker=None,
        )
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

        return (
            hidden_states,
            present_key_value_state,
            extended_attention_mask,
            position_bias,
            encoder_decoder_position_bias,
            past_key_value,
        )

    def finalize_pending_fixed_layer_calibration_exits(
        self,
        *,
        encoder_hidden_states,
        encoder_attention_mask,
        head_mask,
        cross_attn_head_mask,
        past_key_values,
    ):
        """Mirrors DeployT5Stack.finalize_pending_fixed_layer_calibration_exits.

        CALIBRATION-ONLY terminal exact finalization for shallow exits that
        were still pending when generation ended and therefore never reached
        a normal synchronized flush. It runs only after termination is final,
        only over already-generated tokens: it never generates a token, never
        touches the LM head, never changes confidence/early-exit decisions,
        and is never counted as normal FREE runtime work. It exists purely so
        the fitting collector can obtain exact targets for terminal exits."""

        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is None:
            return
        if not self._fixed_layer_exact_cache_calibration_active():
            return
        pending_count = len(self.stack_hidden_states)
        if pending_count == 0:
            return

        source_layer = int(collector.fixed_source_layer)
        records = self.stack_fixed_layer_calibration_records
        metadata = self.stack_hidden_metadata

        # Tail input invariants -- fail closed, never fabricate. None of these
        # were even attempted, so a violation makes the batch
        # "remaining_unfinalized", not "failed".
        invariant_violation = None
        if len(records) != pending_count or len(metadata) != pending_count:
            invariant_violation = "pending_record_count_mismatch"
        elif any(record is None for record in records):
            invariant_violation = "pending_record_missing"
        elif any(not record.get("selected_token_committed") for record in records):
            invariant_violation = "selected_token_not_committed"
        if invariant_violation is None:
            self_attn_past_len_at_start = None
            if past_key_values is not None:
                try:
                    self_attn_past_len_at_start = safe_cache_seq_len(past_key_values[source_layer])
                except (TypeError, IndexError):
                    self_attn_past_len_at_start = None
            base_position = int(self_attn_past_len_at_start or 0)
            for k, record in enumerate(records):
                if record.get("decoder_position") is None or int(record["decoder_position"]) != base_position + k:
                    invariant_violation = "pending_position_invalid_or_unordered"
                    break
        if invariant_violation is not None:
            collector.record_tail_finalization_unresolved(pending_count, reason=invariant_violation)
            return

        extended_attention_mask = None
        position_bias = None
        encoder_decoder_position_bias = None
        present_key_value_states = []
        current_layer = source_layer
        try:
            pending_hidden_by_layer = [
                list(record["hidden_prefix_0_to_source_layer"]) for record in records
            ]
            hidden_states = torch.cat(self.stack_hidden_states, dim=1)
            if int(hidden_states.shape[1]) != pending_count:
                raise ValueError("calibration_pending_hidden_width_mismatch")

            head_mask = self.get_head_mask(head_mask, len(self.block))
            cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, len(self.block))
            encoder_extended_attention_mask = None
            if encoder_hidden_states is not None:
                if encoder_attention_mask is None:
                    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                    encoder_attention_mask = torch.ones(
                        (encoder_batch_size, encoder_sequence_length), device=encoder_hidden_states.device
                    )
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

            for j in range(source_layer, len(self.block)):
                current_layer = j
                for k in range(pending_count):
                    pending_slice = hidden_states[:, k : k + 1, :].detach().cpu().contiguous()
                    if j == source_layer:
                        stored_source_hidden = pending_hidden_by_layer[k][j]
                        if not torch.equal(stored_source_hidden, pending_slice):
                            raise ValueError(
                                "calibration_fixed_layer_source_hidden_mismatch_at_tail_finalization"
                            )
                    else:
                        if len(pending_hidden_by_layer[k]) != int(j):
                            raise ValueError("calibration_prefix_hidden_capture_order_invalid")
                        pending_hidden_by_layer[k].append(pending_slice)

                (
                    hidden_states,
                    present_key_value_state,
                    extended_attention_mask,
                    position_bias,
                    encoder_decoder_position_bias,
                    _past_key_value,
                ) = self._execute_deep_target_layer(
                    j,
                    hidden_states,
                    extended_attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_extended_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    head_mask=head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                )
                if not torch.isfinite(hidden_states).all():
                    raise ValueError("calibration_tail_finalization_nonfinite_hidden")
                if (
                    present_key_value_state is None
                    or not torch.isfinite(present_key_value_state[0]).all()
                    or not torch.isfinite(present_key_value_state[1]).all()
                ):
                    raise ValueError("calibration_tail_finalization_nonfinite_or_missing_kv")
                present_key_value_states.append(present_key_value_state)
        except Exception as exc:
            collector.record_tail_finalization_failure(
                pending_count,
                stage="layer_{}".format(current_layer),
                failure_type=type(exc).__name__,
                reason=str(exc),
            )
            return

        base_position = int(safe_cache_seq_len(past_key_values[source_layer]) or 0) if past_key_values is not None else 0
        finalized_count = 0
        try:
            for k, record in enumerate(records):
                derived_position = base_position + k
                if derived_position < 0:
                    raise ValueError("calibration_fixed_layer_pending_position_invalid")
                target_kv_by_layer = []
                # The collector requires full 0..len(self.block)-1 coverage
                # (only layers >= source_layer are used downstream). Layers
                # below source_layer were never touched here -- each pending
                # token already passed through them normally at its own exit
                # time, so their exact K/V already live in the untouched
                # input past_key_values at this same absolute position.
                for layer in range(source_layer):
                    past_state = past_key_values[layer]
                    if past_state is None or derived_position >= int(past_state[0].shape[2]):
                        raise ValueError("calibration_fixed_layer_pending_position_out_of_range")
                    target_kv_by_layer.append(
                        (
                            past_state[0][:, :, derived_position : derived_position + 1, :],
                            past_state[1][:, :, derived_position : derived_position + 1, :],
                        )
                    )
                for state in present_key_value_states:
                    if derived_position >= int(state[0].shape[2]):
                        raise ValueError("calibration_fixed_layer_pending_position_out_of_range")
                    target_kv_by_layer.append(
                        (
                            state[0][:, :, derived_position : derived_position + 1, :],
                            state[1][:, :, derived_position : derived_position + 1, :],
                        )
                    )
                self._stage_fixed_layer_exact_cache_calibration_event(
                    hidden_by_layer=pending_hidden_by_layer[k],
                    key_value_by_layer=target_kv_by_layer,
                    confidence=record["confidence"],
                    decoder_position=record["decoder_position"],
                    source_selected_token_id=record["shallow_selected_token_id"],
                    event_origin=EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION,
                )
                self.exact_cache_calibration_collector.commit_selected_token(
                    record["shallow_selected_token_id"]
                )
                finalized_count += 1
        except Exception as exc:
            remaining = pending_count - finalized_count - 1
            collector.record_tail_finalization_failure(
                1,
                stage="staging_event_{}".format(finalized_count),
                failure_type=type(exc).__name__,
                reason=str(exc),
            )
            if remaining > 0:
                collector.record_tail_finalization_unresolved(remaining, reason="prior_staging_event_failed")
            # Remove ONLY what actually finalized; unresolved evidence stays.
            self.stack_hidden_states = self.stack_hidden_states[finalized_count:]
            self.stack_hidden_metadata = self.stack_hidden_metadata[finalized_count:]
            self.stack_fixed_layer_calibration_records = self.stack_fixed_layer_calibration_records[finalized_count:]
            self.stack_phase3c_source_hidden_records = self.stack_phase3c_source_hidden_records[finalized_count:]
            return

        # Every pending exit finalized: clear the pending stack exactly like a
        # normal synchronized flush does.
        self.stack_hidden_states = ()
        self.stack_hidden_metadata = ()
        self.stack_fixed_layer_calibration_records = ()
        self.stack_phase3c_source_hidden_records = ()

    def parallel_gen_token(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_extended_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        present_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        layer_idx=None,
    ):
        r""" 
        if pask_key_values is not defined, it implies that all previous tokens have skipped Deep decoder.
            Because all sequences share key_value of cross-attn layer,
            we need to generate key_value of cross-attn layer only once for <start> token.
        else:
            key_values of cross-attn are already stored in 'past_key_values'.

        Then, generate the next token in a non-autoregressive manner.
        if copy_skipped_hidden_states is True,
            copy previous skipped hidden_states for Deep decoder blocks.
        else:
            attention calculate for stack_hidden_states as well.
            thus, we can utilize them in RollBack policy.
        """
        
        # Calibration observer state. Captured from the exact computation this
        # method already performs -- never a second deep replay.
        _fixed_layer_calibration_active_here = self._fixed_layer_exact_cache_calibration_active()
        pending_skipped_tokens = len(self.stack_hidden_states)
        pending_fixed_layer_calibration_records = self.stack_fixed_layer_calibration_records
        # Task C1: snapshot the pending Phase-3c source-hidden records and
        # build the SAME restoration plan DeployT5Stack builds, before the
        # exact deep flush below runs. The plan and every restoration event
        # stay gated on the runtime-restoration predicate so the
        # restoration-disabled FREE cache path is byte-identical; the exact-
        # catch-up ACCOUNTING below is recorded unconditionally, exactly like
        # DeployT5Stack -- the baseline FREE flush genuinely performs this
        # catch-up whether or not restoration then overwrites it.
        pending_phase3c_source_hidden_records = self.stack_phase3c_source_hidden_records
        runtime_restoration_plan = None
        if self._runtime_restoration_enabled():
            if isinstance(pending_phase3c_source_hidden_records, tuple) and isinstance(
                self.stack_hidden_states, tuple
            ):
                assert len(pending_phase3c_source_hidden_records) == pending_skipped_tokens
            runtime_restoration_plan = self._runtime_restoration_plan(
                pending_skipped_tokens,
                self.stack_hidden_metadata,
                (),
                layer_idx,
                phase3c_source_hidden_records=pending_phase3c_source_hidden_records,
            )
        # Record only the REQUIRED units (and flush count) here, before any
        # target-layer call has actually run; EXECUTED units are recorded per
        # target layer inside the for-j loop below, only after that layer's
        # forward call actually returns (T5-parity). Exact-catch-up units are
        # claimed ONLY for the Synchronized Exact path
        # (copy_skipped_hidden_states=False), where the pending skipped
        # tokens are genuinely replayed through the missing deep blocks.
        # State Copying (copy_skipped_hidden_states=True) generates deep K/V
        # from the COPIED shallow hidden states -- no exact replay happens,
        # so it must never report exact-catch-up work. (Runtime restoration
        # with State Copying is impossible: _runtime_restoration_plan raises
        # NotImplementedError for that combination.)
        _exact_parallel_catchup = not bool(self.config.copy_skipped_hidden_states)
        _taskc1_accounting = self._missing_kv_accounting_obj()
        if _exact_parallel_catchup:
            _taskc1_accounting.record_exact_catchup_required(
                pending_skipped_tokens, len(self.block) - int(layer_idx)
            )
        _taskc1_accounting.record_restoration_flush(
            bool(runtime_restoration_plan.get("enabled")) if runtime_restoration_plan else False,
            len(runtime_restoration_plan.get("restore_relative_indices") or [])
            if runtime_restoration_plan
            else 0,
        )
        self_attn_past_len_at_start = None
        if past_key_values is not None and layer_idx is not None:
            try:
                self_attn_past_len_at_start = safe_cache_seq_len(past_key_values[layer_idx])
            except (TypeError, IndexError):
                self_attn_past_len_at_start = None
        # Each pending event's hidden-by-layer list starts as its OWN stored
        # raw prefix for layers 0..source_layer (captured at its own exit
        # time, never from this flush's current no-crossing token) and gains
        # one entry per deeper target layer as the for-j loop below actually
        # executes.
        pending_fixed_layer_calibration_hidden_by_layer = (
            [
                list(record["hidden_prefix_0_to_source_layer"]) if record is not None else None
                for record in pending_fixed_layer_calibration_records
            ]
            if _fixed_layer_calibration_active_here
            else None
        )
        if _fixed_layer_calibration_active_here:
            if len(pending_fixed_layer_calibration_records) != pending_skipped_tokens:
                raise ValueError("calibration_fixed_layer_pending_record_count_mismatch")
            for record in pending_fixed_layer_calibration_records:
                if record is None or not record.get("selected_token_committed"):
                    raise ValueError("calibration_fixed_layer_pending_record_missing_or_unverified")

        if not self.config.copy_skipped_hidden_states:
            hidden_states = torch.cat(self.stack_hidden_states + (hidden_states,), dim=1)
            # reset and re-calculate based on the length of hidden_states
            extended_attention_mask, position_bias = None, None
        else:
            self.stack_hidden_states = torch.cat(self.stack_hidden_states, dim=1)
            extended_attention_mask = attention_mask

        for j in range(layer_idx, len(self.block)):

            if _fixed_layer_calibration_active_here:
                # hidden_states here spans every pending exit position
                # (columns 0..N-1, in generation order) followed by the
                # current no-crossing token (last column). Each pending
                # event's pre-block-j hidden must come from ITS OWN column --
                # never from the current token's column (-1).
                for _k in range(pending_skipped_tokens):
                    _pending_slice = hidden_states[:, _k : _k + 1, :].detach().cpu().contiguous()
                    if j == layer_idx:
                        _stored_source_hidden = pending_fixed_layer_calibration_hidden_by_layer[_k][j]
                        if not torch.equal(_stored_source_hidden, _pending_slice):
                            raise ValueError("calibration_fixed_layer_source_hidden_mismatch_at_flush")
                    else:
                        if len(pending_fixed_layer_calibration_hidden_by_layer[_k]) != int(j):
                            raise ValueError("calibration_prefix_hidden_capture_order_invalid")
                        pending_fixed_layer_calibration_hidden_by_layer[_k].append(_pending_slice)

            past_key_value = past_key_values[j]
            layer_input_seq_len = int(hidden_states.shape[1])
            if past_key_value is None:
                # if pask_key_values is not defined, it implies that all previous tokens have skipped Deep decoder
                # need to generate key_value of cross-attn layer only once for <start> token
                past_key_value = self.block[j].gen_cross_attn_key_value(
                    hidden_states,  # dummy
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=head_mask[j],
                    cross_attn_layer_head_mask=cross_attn_head_mask[j],
                    past_key_value=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                self.deploy_time['time_parallel_key_value_gen'][1] += self.block[j].key_value_gen_time
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
            if extended_attention_mask is None or position_bias is None:
                real_seq_length = hidden_states.shape[1]
                if past_key_value[0] is not None: real_seq_length += past_key_value[0].shape[2]
                key_length = real_seq_length
                
                if self.config.parallel_causal_mask and extended_attention_mask is None:
                    attention_mask = torch.ones(hidden_states.shape[0], real_seq_length, device=hidden_states.device)
                    extended_attention_mask = self.get_extended_attention_mask(attention_mask, torch.Size([hidden_states.shape[0], hidden_states.shape[1]]))
                
                if position_bias is None:      
                    position_bias = self.block[0].layer[0].SelfAttention.compute_bias(real_seq_length, key_length, device=hidden_states.device)

                    # if key and values are already calculated
                    # we want only the last query position bias
                    if past_key_value is not None:
                        position_bias = position_bias[:, :, -hidden_states.size(1):, :]

                    if extended_attention_mask is not None:
                        position_bias = position_bias + extended_attention_mask  # (batch_size, n_heads, seq_length, key_length)

            if self.config.use_synchronize: torch.cuda.synchronize()
            self.deploy_time['time_others'] += (datetime.datetime.now() - start)

            layer_outputs = self.block[j](
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[j],
                cross_attn_layer_head_mask=cross_attn_head_mask[j],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=False,
                parallel_mask=True,
                stack_hidden_states=self.stack_hidden_states if self.config.copy_skipped_hidden_states else None,
            )
            for idx, t in enumerate(self.block[j].key_value_gen_time): self.deploy_time['time_parallel_key_value_gen'][idx] += t
            for idx, t in enumerate(self.block[j].attn_time): self.deploy_time['time_parallel_attn'][idx] += t
            self.deploy_time['time_parallel_ffn'] += self.block[j].ffn_time
            
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # Task C1: the exact deep computation for this target layer has
            # just run -- its executed units are recorded only now, exactly
            # like DeployT5Stack, and only on the Synchronized Exact path
            # (State Copying performs no exact replay -- see
            # _exact_parallel_catchup above). The reused apply overwrites
            # ONLY the self-attention K/V slices at the restored pending
            # positions in a clone -- present_key_value_state[2:] (the
            # cross-attention tail) is reattached structurally untouched.
            # Restoration-disabled runs never enter that block.
            if _exact_parallel_catchup:
                self._missing_kv_accounting_obj().record_exact_catchup_executed(pending_skipped_tokens)
            if use_cache and self._runtime_restoration_enabled():
                present_key_value_state, _runtime_restoration_event = (
                    self._apply_runtime_restoration_to_present_kv(
                        j,
                        present_key_value_state,
                        past_key_value,
                        pending_skipped_tokens,
                        (),
                        runtime_restoration_plan,
                        layer_input_seq_len,
                        pending_source_hidden_states=None,
                        phase3c_source_hidden_records=pending_phase3c_source_hidden_records,
                    )
                )
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + [present_key_value_state,]

            if self.config.use_synchronize: torch.cuda.synchronize()
            self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        if _fixed_layer_calibration_active_here:
            # present_key_value_states now spans every layer, each holding one
            # position per pending exit (columns self_attn_past_len_at_start
            # .. +N-1, in generation order) followed by the current
            # no-crossing token's own position. Commit exactly one fitting
            # event per pending exit -- never reduced to a single event, and
            # never using the current token's own (-1) position. The K/V used
            # here is the exact self-attention K/V the FREE flush above just
            # produced; cross-attention entries are left untouched.
            _base_position = int(self_attn_past_len_at_start or 0)
            for _k, _record in enumerate(pending_fixed_layer_calibration_records):
                _derived_position = _base_position + _k
                if int(_record["decoder_position"]) != _derived_position:
                    raise ValueError("calibration_fixed_layer_pending_position_mismatch")
                if _derived_position < 0:
                    raise ValueError("calibration_fixed_layer_pending_position_invalid")
                _target_kv_by_layer = []
                for _state in present_key_value_states:
                    if _derived_position >= int(_state[0].shape[2]):
                        raise ValueError("calibration_fixed_layer_pending_position_out_of_range")
                    _target_kv_by_layer.append(
                        (
                            _state[0][:, :, _derived_position : _derived_position + 1, :],
                            _state[1][:, :, _derived_position : _derived_position + 1, :],
                        )
                    )
                self._stage_fixed_layer_exact_cache_calibration_event(
                    hidden_by_layer=pending_fixed_layer_calibration_hidden_by_layer[_k],
                    key_value_by_layer=_target_kv_by_layer,
                    confidence=_record["confidence"],
                    decoder_position=_record["decoder_position"],
                    source_selected_token_id=_record["shallow_selected_token_id"],
                )
                # Complete this event before staging the next: the collector
                # has exactly one pending lifecycle slot.
                self.exact_cache_calibration_collector.commit_selected_token(
                    _record["shallow_selected_token_id"]
                )

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        self.stack_hidden_states = ()
        self.stack_hidden_metadata = ()
        self.stack_fixed_layer_calibration_records = ()
        self.stack_phase3c_source_hidden_records = ()
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        return hidden_states, present_key_value_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lm_head=None,
        cm_head=None,
    ):
        r""" 
        We have implemented the following inference strategy:

        1) Normal framework: Forward all transformer layers.
        2) Static framework: Only forward the pre-defined number of early layers.
        3) Early-Exit framework: Each token can exit the forward path if confidence is higher than threshold.
        4) Shallow-Deep framework: 
            While a few early layers are defined as 'Shallow' decoder, the entire network including Shallow is defined as 'Deep' decoder.
            Each token can skip the Deep decoder path if confidence at Shallow decoder is higher than threshold.
        """

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
            self.stack_hidden_states = ()
            self.stack_hidden_metadata = ()
            self.stack_fixed_layer_calibration_records = ()
            self.stack_phase3c_source_hidden_records = ()
            self.stack_conf, self.stack_pred = (), ()
            if self.is_decoder:
                collector = getattr(self, "exact_cache_calibration_collector", None)
                if collector is not None:
                    collector.abort_pending()
                # T5-parity generation lifecycle: reset generation-local
                # runtime diagnostic counters and (outside trainer-driven
                # evaluation) the accounting aggregates, so no counter ever
                # leaks from one sample's generation into the next. Runs
                # AFTER abort_pending so the collector's own reset_run()
                # never sees a stale pending event.
                self._begin_missing_kv_generation()
                # Same ordering as DeployT5Stack: increment first, then write
                # the binding row, so the row carries THIS generation's index.
                self._generation_index += 1
                self._record_missing_kv_generation_binding()

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if self.is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        elif self.config.encoder_attention_type == "local":
            extended_attention_mask = _get_local_attention_mask(attention_mask, self.block_len, inputs_embeds.device)
        else:
            extended_attention_mask = attention_mask

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = [] if use_cache else None
        all_hidden_states = None
        all_attentions = None
        all_cross_attentions = None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        skip_mask = False  # False: forward, and True: skip
        self.shallow2deep = False  # False: skip, and True: forward
        self.lm_logits = None  # to prevent calculating logits twice
        # Official CALM Task C1: per-token candidate decision history handed
        # to the aliased transaction (same forward-local lifetime as T5's).
        calm_phase3c_candidate_evaluations = []
        # Raw hidden entering each block, captured only while calibration is
        # enabled. Index i holds the input to block i, so at an exit from the
        # configured shallow source layer, entries 0..source_layer-1 exist
        # and h_source (the input to block source_layer) is appended by the
        # exit branch itself.
        exact_cache_calibration_prefix_hidden_states = (
            [] if self._exact_cache_calibration_enabled() else None
        )

        for i, layer_module in enumerate(self.block):
                
            # Static framework
            if self.is_decoder and self.config.static_exit_layer is not None:
                if i == self.config.static_exit_layer: break

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            
            # check that tokens are generated once in a time
            auto_reg = True if hidden_states.shape[1] == 1 else False
            if self.is_decoder and auto_reg and i == 0:
                self.block_op[i] += 1
                # T5-parity: one generated token per block-0 autoregressive
                # entry -- the denominator SumTrainer's per-sample accounting
                # reads (generated_token_count).
                self._missing_kv_accounting_obj().record_generated_token()
                            
            if self.is_decoder and auto_reg and i > 0:
                
                # Shallow-Deep framework 
                if self.use_shallow_deep and i == self.shallow_exit_layer:
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    start = datetime.datetime.now()
                    _hidden_states = self.dropout(self.final_layer_norm(hidden_states))
                    lm_logits = lm_head(_hidden_states) if not self.config.tie_word_embeddings \
                        else lm_head(_hidden_states * (self.config.d_model ** -0.5))
                        
                    skip_mask, conf = get_skip_mask(
                        lm_logits,
                        _hidden_states,
                        cm_head,
                        config=self.config,
                        adapt_threshold=self.bmm_threshold,
                        return_conf=True,
                    )
                    self.stack_conf = self.stack_conf + (conf,)
                    self.stack_pred = self.stack_pred + (lm_logits,)

                    if not skip_mask: self.block_op[i] += 1
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)

                    # if skip Deep decoder, store hidden_states at self.shallow_exit_layer
                    if skip_mask:
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        # T5-parity: every REAL shallow exit records its
                        # nominally skipped deep token-layer units (source i,
                        # len(self.block) layers), the counter the trainer
                        # derives early_exit_token_count from.
                        self._missing_kv_accounting_obj().record_nominal_skip(i, len(self.block), token_count=1)
                        self.lm_logits = lm_logits
                        if self.config.parallel_gen_token:
                            if use_cache:
                                for j in range(i, len(self.block)):
                                    present_key_value_states = present_key_value_states + [past_key_values[j],]
                            self.stack_hidden_states = self.stack_hidden_states + (hidden_states,)
                            # Task C1 Phase-3c source-hidden capture: on a real
                            # shallow exit, hidden_states here IS h_source (the
                            # raw hidden ENTERING block shallow_exit_layer --
                            # for the official LongT5 checkpoint, h3 = block2
                            # output = block3 input). Captured via the SAME
                            # DeployT5Stack record maker, appended for EVERY
                            # exit (None when restoration is disabled) so the
                            # tuple stays exactly aligned with FREE's own
                            # pending stack. The exact deep flush later
                            # overwrites targets shallow_exit_layer..N-1 from
                            # this record; the last already-exact self K/V
                            # stays block shallow_exit_layer-1, untouched.
                            phase3c_source_hidden_record = None
                            if self._phase3c_runtime_restoration_enabled():
                                restoration_decoder_position = infer_decoder_position(past_key_values)
                                if restoration_decoder_position is None:
                                    restoration_decoder_position = (
                                        self._infer_decoder_position_from_source_present_kv(
                                            present_key_value_states, i - 1
                                        )
                                    )
                                phase3c_source_hidden_record = (
                                    self._maybe_make_phase3c_source_hidden_record_for_skip(
                                        hidden_states,
                                        phase3c_source_hidden_layer=i,
                                        exit_layer=i,
                                        decoder_position=restoration_decoder_position,
                                        relative_index=len(self.stack_phase3c_source_hidden_records),
                                        confidence=conf,
                                    )
                                )
                            self.stack_phase3c_source_hidden_records = (
                                self.stack_phase3c_source_hidden_records + (phase3c_source_hidden_record,)
                            )
                            assert len(self.stack_phase3c_source_hidden_records) == len(self.stack_hidden_states)
                            # Calibration observer bookkeeping ONLY -- entirely
                            # inside this gate, so when calibration is disabled
                            # the FREE exit does exactly what it always did and
                            # stack_hidden_metadata /
                            # stack_fixed_layer_calibration_records both stay
                            # empty for the whole generation.
                            #
                            # This is the ACTUAL exit branch (skip_mask=True,
                            # confidence > threshold) -- the only branch allowed
                            # to originate a fixed-layer calibration event.
                            # hidden_states here is the raw hidden ENTERING
                            # block shallow_exit_layer (h_source) for THIS
                            # token, captured now so a later no-crossing token can
                            # never donate its identity to this deferred event.
                            # The exact deep hidden/K/V population is filled in
                            # later, at flush time.
                            if self._fixed_layer_exact_cache_calibration_active():
                                decoder_position = infer_decoder_position(past_key_values)
                                if decoder_position is None:
                                    # First decoder position: there is no prior
                                    # cache length to read yet, so reuse the
                                    # last already-exact layer's own present K/V.
                                    decoder_position = self._infer_decoder_position_from_source_present_kv(
                                        present_key_value_states, i - 1
                                    )
                                layer0_past_seq_len = (
                                    safe_cache_seq_len(past_key_values[0]) if len(past_key_values) else None
                                )
                                self.stack_hidden_metadata = self.stack_hidden_metadata + (
                                    make_skipped_token_metadata(
                                        relative_index=len(self.stack_hidden_metadata),
                                        decoder_position=decoder_position,
                                        layer0_past_seq_len=layer0_past_seq_len,
                                        exit_layer=i,
                                        confidence=conf,
                                    ),
                                )
                                fixed_layer_calibration_record = {
                                    "sample_context": dict(self._missing_kv_sample_context_fields()),
                                    "generation_index": int(self._generation_index),
                                    "decoder_position": (
                                        int(decoder_position) if decoder_position is not None else None
                                    ),
                                    "confidence": float(conf),
                                    "threshold": float(getattr(self.config, "shallow2deep_conf_threshold")),
                                    "shallow_selected_token_id": int(
                                        lm_logits.detach().argmax(-1).reshape(-1)[-1].item()
                                    ),
                                    "selected_token_committed": False,
                                    "hidden_prefix_0_to_source_layer": [
                                        tensor.detach().cpu().contiguous()
                                        for tensor in exact_cache_calibration_prefix_hidden_states
                                    ]
                                    + [hidden_states.detach().cpu().contiguous()],
                                }
                                self.exact_cache_calibration_collector.record_fixed_layer_exit_observed()
                                self.stack_fixed_layer_calibration_records = (
                                    self.stack_fixed_layer_calibration_records
                                    + (fixed_layer_calibration_record,)
                                )
                                # Pending tuples must stay exactly aligned with
                                # FREE's own stack_hidden_states.
                                assert len(self.stack_hidden_states) == len(self.stack_hidden_metadata)
                                assert len(self.stack_fixed_layer_calibration_records) == len(
                                    self.stack_hidden_metadata
                                )
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
                        break

                    if not skip_mask:
                        self.shallow2deep = True
                        if self.config.parallel_gen_token and len(self.stack_hidden_states):
                            self.parallel_tokens_shallow += len(self.stack_hidden_states)
                            self.parallel_tokens_deep += 1
                            
                            # PURE_MISSING_KV_RECOVERY_COMPUTE_COST shadow
                            # measurement (the SAME aliased Native FREE
                            # implementation DeployT5Stack runs at this exact
                            # point): measures BEFORE the production flush
                            # over the exact same pending workload, never
                            # mutating live state. The production flush below
                            # remains the reference trajectory.
                            pure_recovery_held_spans = None
                            if (
                                self._pure_recovery_cost_enabled()
                                and len(self.stack_hidden_states) > 0
                            ):
                                pure_recovery_held_spans = self._measure_pure_missing_kv_recovery(
                                    past_key_values,
                                    encoder_hidden_states,
                                    encoder_extended_attention_mask,
                                    encoder_decoder_position_bias,
                                    head_mask,
                                    cross_attn_head_mask,
                                    use_cache,
                                    output_attentions,
                                    infer_decoder_position(past_key_values),
                                )
                            # in Shallow-Deep decoder, generate the next token in a non-autoregressive manner
                            hidden_states, present_key_value_states = self.parallel_gen_token(
                                hidden_states,
                                attention_mask=extended_attention_mask,
                                position_bias=position_bias,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_extended_attention_mask=encoder_extended_attention_mask,
                                encoder_decoder_position_bias=encoder_decoder_position_bias,
                                head_mask=head_mask,
                                cross_attn_head_mask=cross_attn_head_mask,
                                past_key_values=past_key_values,
                                present_key_value_states=present_key_value_states,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                layer_idx=self.shallow_exit_layer,
                            )
                            if pure_recovery_held_spans is not None:
                                # Validation mode: compare the pending-only
                                # shadow K/V against the pending portion of
                                # the production mixed flush, then discard.
                                self._pure_recovery_validate_pending_parity(
                                    pure_recovery_held_spans, present_key_value_states
                                )
                                pure_recovery_held_spans = None

                            # Adaptive Threshold Estimator
                            if self.config.use_adapt_threshold:
                                # Calibration Set Update
                                self.lm_logits = self.lm_head(self.dropout(self.final_layer_norm(hidden_states)))
                                deep_pred = self.lm_logits.argmax(-1)
                                shallow_pred = torch.cat(self.stack_pred[-deep_pred.size(1):]).argmax(-1).view(-1)

                                self.stack_conf_all += self.stack_conf[-deep_pred.size(1):]
                                self.stack_ident_all += ((deep_pred.view(-1) == shallow_pred.view(-1)).long().cpu().numpy(),)
                                self.stack_conf, self.stack_pred = (), ()
                                
                            break

                # Official FREE CALM first-crossing Task C1 (Patch 1): the
                # opt-in missing-K/V transaction path, mirroring the accepted
                # DeployT5Stack branch one-for-one. Evaluated BEFORE the
                # original Early-Exit branch below, and only when the Task C1
                # runtime is explicitly enabled -- otherwise execution falls
                # through to the untouched upstream FREE baseline exactly as
                # before. `hidden_states` here is the raw block-i input h_i;
                # the confidence helper builds its OWN normalized temporary
                # and never mutates it, so the transaction's source_hidden
                # stays the raw h_s (never final_layer_norm(h_s)).
                elif self._is_calm_taskc1_runtime_enabled() and not skip_mask:
                    candidate_layers = self._calm_phase3c_candidate_layers()
                    if i not in candidate_layers:
                        self.block_op[i] += 1
                    else:
                        if self.config.use_synchronize:
                            torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        with self._missing_kv_component_timer_obj().time_block(
                            "confidence_time_ms",
                            device=hidden_states.device,
                        ):
                            confidence_hook = getattr(self, "_calm_phase3c_candidate_confidence_test_hook", None)
                            if callable(confidence_hook):
                                lm_logits, conf = confidence_hook(i, hidden_states)
                            else:
                                lm_logits, conf = compute_calm_candidate_logits_and_confidence(
                                    hidden_states,
                                    final_layer_norm=self.final_layer_norm,
                                    dropout=self.dropout,
                                    lm_head=lm_head,
                                    config=self.config,
                                )
                        threshold = float(getattr(self.kv_runtime_restorer, "candidate_threshold", CALM_THRESHOLD))
                        candidate_pass = float(conf) > threshold
                        decision = {
                            "candidate_layer": int(i),
                            "confidence": float(conf),
                            "threshold": threshold,
                            "threshold_comparator": "strict_gt",
                            "candidate_pass": bool(candidate_pass),
                        }
                        calm_phase3c_candidate_evaluations.append(decision)
                        self.kv_trace.record(
                            "kv_calm_phase3c_candidate_evaluation",
                            **decision,
                            candidate_layers=list(candidate_layers),
                            decoder_position=infer_decoder_position(past_key_values),
                            source_hidden_semantics="raw_block_input_h_{}".format(i),
                        )
                        if self.config.use_synchronize:
                            torch.cuda.synchronize()
                        self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                        if candidate_pass:
                            source_hidden = hidden_states.detach().clone()
                            self.lm_logits = lm_logits
                            decoder_position = infer_decoder_position(past_key_values)
                            self._missing_kv_accounting_obj().record_calm_first_crossing(i)
                            self._missing_kv_accounting_obj().record_nominal_skip(i, len(self.block), 1)
                            self._kv_runtime_restoration_counters["calm_phase3c_first_crossing_tokens"] += 1
                            self.kv_trace.record(
                                "kv_calm_phase3c_first_crossing",
                                restoration_method=getattr(self.kv_runtime_restorer, "method", None),
                                runtime_mode=getattr(self.kv_runtime_restorer, "runtime_mode", PHASE3C_RUNTIME_MODE),
                                selected_source_layer=int(i),
                                last_exact_kv_layer=int(i) - 1,
                                first_missing_target_layer=int(i),
                                decoder_position=decoder_position,
                                candidate_evaluations=list(calm_phase3c_candidate_evaluations),
                                preserved_exit_logits=True,
                                exact_reference_start_layer=int(i),
                                exact_reference_end_layer=len(self.block) - 1,
                                exact_catchup_avoided_token_layer_units=0,
                                speed_claim_valid=False,
                            )
                            hidden_states, present_key_value_states, _calm_phase3c_event = (
                                self._run_calm_phase3c_exact_reference_and_stage_cache(
                                    source_layer=i,
                                    source_hidden=source_hidden,
                                    exit_logits=lm_logits,
                                    confidence=conf,
                                    candidate_evaluations=calm_phase3c_candidate_evaluations,
                                    hidden_states=hidden_states,
                                    extended_attention_mask=extended_attention_mask,
                                    position_bias=position_bias,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_extended_attention_mask=encoder_extended_attention_mask,
                                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                                    head_mask=head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    past_key_values=past_key_values,
                                    present_key_value_states=present_key_value_states,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    calibration_prefix_hidden_states=exact_cache_calibration_prefix_hidden_states,
                                )
                            )
                            break
                        self.block_op[i] += 1
                        if int(i) == int(candidate_layers[-1]):
                            self._missing_kv_accounting_obj().record_calm_full_depth_fallback()
                            self._kv_runtime_restoration_counters["calm_phase3c_full_depth_fallback_tokens"] += 1
                            is_exact_catchup = (
                                getattr(self.kv_runtime_restorer, "method", None) == EXACT_CATCHUP_METHOD
                            )
                            no_crossing_semantics = {}
                            if is_exact_catchup:
                                no_crossing_semantics = {
                                    "policy_no_crossing": True,
                                    "restoration_error_fallback": False,
                                }
                                calibration_collector = getattr(self, "exact_cache_calibration_collector", None)
                                if calibration_collector is not None:
                                    calibration_collector.stage_no_crossing()
                            self.kv_trace.record(
                                "kv_calm_phase3c_full_depth_fallback",
                                candidate_evaluations=list(calm_phase3c_candidate_evaluations),
                                decoder_position=infer_decoder_position(past_key_values),
                                restoration_requested=False,
                                exact_reference_duplicate_branch=False,
                                speed_claim_valid=False,
                                **no_crossing_semantics,
                            )

                # Early-Exit framework
                elif self.use_early_exit and not skip_mask:
                    if self.exit_min_layer is not None and i < self.exit_min_layer: 
                        self.block_op[i] += 1
                    else:
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        _hidden_states = self.dropout(self.final_layer_norm(hidden_states))
                        lm_logits = lm_head(_hidden_states) if not self.config.tie_word_embeddings \
                            else lm_head(_hidden_states * (self.config.d_model ** -0.5))
                            
                        skip_mask = get_skip_mask(
                            lm_logits,
                            _hidden_states,
                            cm_head,
                            config=self.config,
                            pos_time=past_key_values[i][0].shape[2] + 1 if past_key_values[i] is not None else 1
                        )
                        if not skip_mask: self.block_op[i] += 1                    
                        if skip_mask: self.lm_logits = lm_logits
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                    
                # Normal framework
                elif (not self.use_shallow_deep and not self.use_early_exit):
                    self.block_op[i] += 1

            if exact_cache_calibration_prefix_hidden_states is not None:
                if len(exact_cache_calibration_prefix_hidden_states) != int(i):
                    raise ValueError("calibration_prefix_hidden_capture_order_invalid")
                exact_cache_calibration_prefix_hidden_states.append(
                    hidden_states.detach().cpu().contiguous()
                )
            past_key_value = past_key_values[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=skip_mask,
            )

            if self.is_decoder:
                if self.config.use_early_exit: prefix = 'time_exit_' if skip_mask else 'time_'
                elif self.config.use_shallow_deep: prefix = 'time_parallel_' if self.shallow2deep else 'time_'
                else: prefix = 'time_'
                for idx, t in enumerate(layer_module.key_value_gen_time): self.deploy_time[prefix + 'key_value_gen'][idx] += t
                for idx, t in enumerate(layer_module.attn_time): self.deploy_time[prefix + 'attn'][idx] += t
                self.deploy_time[prefix + 'ffn'] += layer_module.ffn_time

            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + [present_key_value_state,]
            if self.config.use_synchronize: torch.cuda.synchronize()
            if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if not skip_mask and self.lm_logits is None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class DeployLongT5ForConditionalGeneration(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.static_exit_layer = None
        self.encoder = DeployLongT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = DeployLongT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.decoder.lm_head = self.lm_head
        if self.config.exit_conf_type == 'meta' or self.config.shallow2deep_conf_type == "meta":
            self.cm_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model, bias=True),
                nn.ReLU(),
                nn.Linear(config.d_model, 2, bias=True),
            )
        else:
            self.cm_head = None

        # RollBack policy
        self.rollback_num = 0
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # BMM
        self.bmm_update_iter = 0
        self.bmm_update_max_iter = 100

        self.deploy_time = {
            'time_encoder_forward': datetime.timedelta(),
            'time_decoder_forward': datetime.timedelta(),
            'time_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_ffn': datetime.timedelta(),
            'time_confidence': datetime.timedelta(),
            'time_exit_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_exit_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_exit_ffn': datetime.timedelta(),
            'time_parallel_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_parallel_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_parallel_ffn': datetime.timedelta(),
            'time_others': datetime.timedelta(),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        DeployT5ForConditionalGeneration class is for a deployment scenario,
        where the decoder models are communicating with only one user (i.e., the batch size of 1).

        Here, for the faster inference, we have implemented non-autoregressive hidden_state copying in Shallow-Deep framework.
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        encoder_outputs, decoder_outputs = self.forward_impl(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                                                            head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs,
                                                            past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache,
                                                            output_attentions, output_hidden_states, return_dict)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if self.decoder.lm_logits is None:  # token has not skipped
            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)
            
            lm_logits = self.lm_head(sequence_output)
        else: lm_logits = self.decoder.lm_logits
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)
        
        if self.config.rollback_conf_threshold is None:
            lm_logits = lm_logits[:, [-1], :]
        loss = self.compute_model_loss(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def compute_model_loss(self, lm_logits=None, labels=None):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            assert lm_logits is not None
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss
    
    def forward_impl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_encoder_forward'] += (datetime.datetime.now() - start)
        
        hidden_states = encoder_outputs[0]

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            
        if past_key_values is None and len(self.decoder.stack_conf_all) > 0 and self.bmm_update_iter < self.bmm_update_max_iter:
            X = np.hstack(self.decoder.stack_conf_all)
            Y = np.hstack(self.decoder.stack_ident_all)
            self.decoder.bmm_model.fit(X, Y)
            
            self.decoder.bmm_threshold = self.decoder.bmm_model.predict_proba(0.3, 0.9)
            self.bmm_update_iter += 1
            
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lm_head=self.lm_head,
            cm_head=self.cm_head,
        )
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)
        for k, v in self.decoder.deploy_time.items():
            if type(v) != list: self.deploy_time[k] += v
            else: self.deploy_time[k] = [_d + _v for _d, _v in zip(self.deploy_time[k], v)]
        self.decoder._reset_time_measure()
        
        return encoder_outputs, decoder_outputs

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only

        # for RollBack policy
        self.rollback_candidates = ()
        self.pass_length_rollback = 0

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )       

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            
            # RollBack policy
            if self.config.use_shallow_deep and self.decoder.shallow2deep and not self.config.copy_skipped_hidden_states and self.config.rollback_conf_threshold is not None:
                if self.config.use_synchronize: torch.cuda.synchronize()
                start = datetime.datetime.now()

                seq_len = outputs.logits.size(1)
                if seq_len == 1:
                    # stack_hidden_states is empty, so do not need to RollBack
                    assert len(self.rollback_candidates) == 0
                    self.pass_length_rollback += 1
                
                else:
                    # we should check RollBack
                    assert seq_len - 1 == len(self.rollback_candidates)
                    
                    deep_logits = outputs.logits[:, :-1, :]
                    shallow_preds = torch.cat(self.rollback_candidates, dim=0)
                    rollback_loss = self.criterion(deep_logits.squeeze(0), shallow_preds)

                    for j, _loss in enumerate(rollback_loss):
                        if _loss.item() > self.config.rollback_conf_threshold:
                            # RollBack
                            outputs.logits = deep_logits[:, [j], :]
                            
                            # remove RollBacked tokens
                            input_ids = input_ids[:, :self.pass_length_rollback + 1]  # consider sos token
                            past_key_values = []
                            for past in outputs.past_key_values:
                                past_key_values += [[past[0][:, :, :self.pass_length_rollback + 1, :],  # self-attn key
                                                     past[1][:, :, :self.pass_length_rollback + 1, :],  # self-attn value
                                                     past[2],
                                                     past[3]],]
                            outputs.past_key_values = past_key_values

                            self.decoder.block_op[0] -= (seq_len - 1) - j
                            self.rollback_num += (seq_len - 1) - j
                            break
                        else:
                            self.pass_length_rollback += 1
                    
                    self.rollback_candidates = ()
                    self.pass_length_rollback += 1
                    
                if self.config.use_synchronize: torch.cuda.synchronize()
                self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # for RollBack, store Shallow decoder's predictions
            if self.config.use_shallow_deep and not self.decoder.shallow2deep and not self.config.copy_skipped_hidden_states and self.config.rollback_conf_threshold is not None:
                self.rollback_candidates += (next_tokens,)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Verify the token the generation loop actually selected is the
            # same one the shallow exit recorded, BEFORE it is appended.
            # Mirrors the DeployT5ForConditionalGeneration.greedy_search hook
            # placement exactly; a mismatch fails closed.
            if hasattr(self.decoder, "_commit_exact_cache_calibration_selected_token"):
                self.decoder._commit_exact_cache_calibration_selected_token(next_tokens)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        # Calibration-only exact tail finalization: only reachable here, after
        # the selected token was committed, appended to input_ids, the
        # stopping criterion evaluated and the loop broken -- i.e. after
        # generation termination is fully fixed. Uses model_kwargs, which
        # still holds the exact live tensors the last successful decoder
        # forward() used (past_key_values updated by
        # _update_model_kwargs_for_generation; encoder_outputs/attention_mask/
        # head_mask/cross_attn_head_mask are encoder-decoder-persistent) --
        # never reconstructed from stored metadata.
        if hasattr(self.decoder, "finalize_pending_fixed_layer_calibration_exits"):
            _tail_pending_count_before = len(getattr(self.decoder, "stack_hidden_states", ()) or ())
            try:
                _tail_encoder_outputs = model_kwargs.get("encoder_outputs")
                self.decoder.finalize_pending_fixed_layer_calibration_exits(
                    encoder_hidden_states=(
                        _tail_encoder_outputs[0] if _tail_encoder_outputs is not None else None
                    ),
                    encoder_attention_mask=model_kwargs.get("attention_mask"),
                    head_mask=model_kwargs.get("head_mask"),
                    cross_attn_head_mask=model_kwargs.get("cross_attn_head_mask"),
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            except Exception as _tail_exc:
                # Calibration-only: never let a tail-finalization bug crash
                # generation or change its already-final result. Recorded as a
                # collector failure rather than silently ignored.
                _tail_collector = getattr(self.decoder, "exact_cache_calibration_collector", None)
                if _tail_collector is not None and _tail_pending_count_before > 0:
                    _tail_collector.record_tail_finalization_failure(
                        _tail_pending_count_before,
                        stage="finalize_pending_fixed_layer_calibration_exits_call",
                        failure_type=type(_tail_exc).__name__,
                        reason=str(_tail_exc),
                    )

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
