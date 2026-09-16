"""
T5: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L19
"""
from typing import Optional, Tuple, Union, List, Callable

import contextlib
import json
import os
import copy
import math
import time
import datetime
import warnings
import numpy as np
import torch
# torch.set_num_threads(4)
# torch.set_num_interop_threads(4)
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
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention, 
    T5LayerFF,
    T5Block, 
    T5Stack, 
    T5ForConditionalGeneration
)
from transformers.models.t5.configuration_t5 import T5Config
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput
except ImportError:
    from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

    GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
    GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.utils import logging

from util import (
    compute_exit_lm_logits,
    get_skip_mask,
    BetaMixture1D,
) 
from our_kv_restoration import (
    H2OImportanceTracker,
    KVTraceRecorder,
    build_exact_recompute_verification_event,
    build_exact_recompute_dryrun_plan,
    build_h2o_key_mask_indices,
    build_h2o_mask_noop_plan,
    build_h2o_mask_plan,
    candidate_positions_from_metadata,
    expected_self_attn_kv_shapes,
    gather_recompute_hidden_states,
    infer_decoder_position,
    infer_pending_start_position_from_metadata,
    make_skipped_token_metadata,
    metadata_list_to_trace,
    metadata_to_trace_dict,
    safe_cache_seq_len,
    select_h2o_topk_dryrun,
    CALM_TASKC1_RUNTIME_METHODS,
    DIRECT_SHALLOW_KV_REUSE_METHOD,
    EXACT_CATCHUP_METHOD,
    EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
    RuntimeKVRestorationManager,
    PHASE3C_RUNTIME_MODE,
    PHASE3C_RUNTIME_RESTORATION_METHOD,
)
from our_kv_restoration.missing_kv_runtime_accounting import (
    MissingKVComponentTimer,
    MissingKVRuntimeAccounting,
    TASK_C2_BATCHED_INSERTION_POLICY_MODE,
    TASK_C2_DIRECT_INSERTION_POLICY_MODE,
)
from our_kv_restoration.early_exit_exact_cache_calibration import (
    ExactCacheCalibrationCollector,
    EVENT_ORIGIN_CALIBRATION_ONLY_TERMINAL_EXACT_FINALIZATION,
)
from our_kv_restoration.missing_kv_exact_catchup_overhead import (
    ExactCatchupOverheadEvent,
    ExactCatchupOverheadRecorder,
    RUNTIME_PATH_CANDIDATE_FIRST_CROSSING,
    RUNTIME_PATH_FIXED_SOURCE_PARALLEL_FLUSH,
    aggregate_exact_catchup_overhead,
    resolve_exact_catchup_event_decoder_position,
)
from our_kv_restoration.missing_kv_dump_provenance import (
    HIDDEN_LOGICAL_RECORD_TYPE,
    KV_LOGICAL_RECORD_TYPE,
    PACKED_GENERATION_STORAGE_FORMAT,
    append_jsonl as append_provenance_jsonl,
    canonical_json_sha256,
    make_dump_row_uid,
    populate_packed_record_identities,
    read_jsonl as read_provenance_jsonl,
    sha256_file,
)
from our_kv_restoration.missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_POLICY_NAME,
    CALM_THRESHOLD,
    calm_policy_sha256,
    build_calm_trace_row,
    compute_calm_candidate_logits_and_confidence,
    parse_candidate_layers,
    validate_calm_trace_rows,
)
from our_kv_restoration.f2a_frozen_schedule import (
    F2A_EVALUATION_PROTOCOL_NAME,
    F2A_METHOD_EXIT_CONDITIONED_HIDDEN_RESTORATION,
    F2A_METHOD_EXIT_HIDDEN_TARGET_PROJECTION,
    F2A_METHOD_FINAL_KV_RESTORATION,
    F2A_REQUIRED_METHODS,
    build_f2a_component_record,
    build_f2a_event,
    build_f2a_exact_shadow_parity_diagnostics,
    build_reference_generation_trajectory_row,
    cache_noninterference_snapshot,
    caches_share_writable_storage,
    clone_cache,
    f2a_logit_metrics,
    ordered_reference_generation_trajectory_sha256,
    patch_candidate_cache_for_full_event,
    reference_generation_trajectory_population_sha256,
    reference_trajectory_sha256,
    restore_f2a_method_block_from_hidden,
    restore_f2a_method_from_hidden,
    schedule_semantic_sha256,
    validate_reference_generation_trajectory_rows,
    validate_cache_noninterference,
    validate_exact_shadow_parity,
    validate_f2a_artifact_for_source_mode,
    validate_f2a_record_rows,
    validate_f2a_reference_position_bias,
    validate_f2a_schedule_rows,
)
from our_kv_restoration.phase3c_policy_artifact import (
    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
    SOURCE_LAYER_MODE_FIXED,
    SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
    artifact_policy_sha256,
    load_phase3c_policy_artifact,
)
from our_kv_restoration.runtime_kv_restoration import (
    STACKED_LEARNED_TARGETS_UNSUPPORTED_STATUS,
    RuntimeKVRestorationResult,
    _is_nonfinite_error_message,
)

logger = logging.get_logger(__name__)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]


def free_skip_decision_reason(skip_mask, use_adapt_threshold=False, adaptive_threshold_active=False, effective_threshold_available=False):
    if use_adapt_threshold and adaptive_threshold_active:
        return "adaptive_threshold_accept" if skip_mask else "adaptive_threshold_reject"
    if effective_threshold_available:
        return "confidence_gt_effective_threshold" if skip_mask else "confidence_le_effective_threshold"
    return "unknown_threshold_source"


class DeployT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False
        self._last_query_states = None
        self._last_position_bias = None

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        skip_mask=False,
        gen_cross_attn_key_value=False,
        stack_hidden_states=None,
        layer_idx=None,
        kv_importance_tracker=None,
        staged_pending_self_kv=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).

        ``staged_pending_self_kv`` (default None: every existing caller is
        byte-identical) is a PRIVATE, already-validated (restored_key,
        restored_value) pair of shape [batch, n_heads, N, d_kv] staged by a
        successful FREE-aligned Batched Task C2 transaction for THIS layer's
        N pending early-exit tokens. When present on the self-attention path,
        the key/value projection publishes

            torch.cat([old_past, staged_pending, current], dim=2)

        in ONE final cache materialization -- the pending block was never
        pre-materialized into the live cache. Self-attention only; the
        cross-attention layer never receives it.
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length        

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            if query_length is None:
                if past_key_value[0] is not None: real_seq_length += past_key_value[0].shape[2]
                if stack_hidden_states is not None: real_seq_length += stack_hidden_states.shape[1]
                # Staged pending K/V occupy the cache positions between the
                # old past and the current token, so they count toward the
                # key length exactly as if they were already in the past.
                if staged_pending_self_kv is not None:
                    real_seq_length += staged_pending_self_kv[0].shape[2]
            else:
                real_seq_length += query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(states.shape[0], -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(states.shape[0], -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value, staged_pending_block=None):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    if staged_pending_block is not None:
                        # The ONE final publication of a successful Batched
                        # Task C2 flush: old exact past + validated restored
                        # pending block + exact current-token K/V, in exact
                        # sequence order, with no intermediate old+pending
                        # cache materialization.
                        hidden_states = torch.cat([past_key_value, staged_pending_block, hidden_states], dim=2)
                    else:
                        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states
        
        # get key/value states
        if self.is_decoder and key_value_states is None and stack_hidden_states is not None:
            _hidden_states = torch.cat((stack_hidden_states,) + (hidden_states,), dim=1)
        else:
            _hidden_states = hidden_states

        key_states = project(
            _hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
            staged_pending_block=staged_pending_self_kv[0] if staged_pending_self_kv is not None else None,
        )
        value_states = project(
            _hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
            staged_pending_block=staged_pending_self_kv[1] if staged_pending_self_kv is not None else None,
        )
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.key_value_gen_time = (datetime.datetime.now() - start)

        # non-autoregressively generate key_value_states for the past skipped tokens
        if gen_cross_attn_key_value:
            return [key_states, value_states]
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=hidden_states.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                
        if skip_mask:
            self._last_query_states = None
            attn_output = None
        else:
            # get query states
            query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
            if (
                self.is_decoder
                and key_value_states is None
                and not gen_cross_attn_key_value
                and (
                    getattr(self.config, "kv_attention_diag_dump_enabled", False)
                    or getattr(self.config, "kv_full_attention_diag_dump_enabled", False)
                )
            ):
                self._last_query_states = query_states.detach()
            else:
                self._last_query_states = None
                self._last_position_bias = None

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if self.pruned_heads:
                mask = torch.ones(position_bias.shape[1])
                mask[list(self.pruned_heads)] = 0
                position_bias_masked = position_bias[:, mask.bool()]
            else:
                position_bias_masked = position_bias
            if self._last_query_states is not None:
                self._last_position_bias = position_bias_masked.detach() if position_bias_masked is not None else None
            scores += position_bias_masked

            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            if (
                self.is_decoder
                and key_value_states is None
                and not skip_mask
                and not gen_cross_attn_key_value
                and kv_importance_tracker is not None
                and kv_importance_tracker.enabled
                and layer_idx is not None
            ):
                kv_importance_tracker.update(layer_idx, attn_weights)

            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = self.o(attn_output)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.attn_ffn_time = (datetime.datetime.now() - start)
        present_key_value_state = [key_states, value_states] if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs


class DeployT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.SelfAttention = DeployT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        skip_mask=False,
        stack_hidden_states=None,
        layer_idx=None,
        kv_importance_tracker=None,
        staged_pending_self_kv=None,
    ):
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_stack_hidden_states = self.layer_norm(stack_hidden_states) if stack_hidden_states is not None else None
        if self.config.is_decoder and getattr(self.config, "kv_all_layer_hidden_dump_enabled", False):
            self._last_raw_hidden_states_for_hidden_dump = hidden_states.detach()
            self._last_normed_hidden_states_for_hidden_dump = normed_hidden_states.detach()
        else:
            self._last_raw_hidden_states_for_hidden_dump = None
            self._last_normed_hidden_states_for_hidden_dump = None
        if self.config.use_synchronize: torch.cuda.synchronize()
        norm_time = (datetime.datetime.now() - start)

        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skip_mask=skip_mask,
            stack_hidden_states=normed_stack_hidden_states,
            layer_idx=layer_idx,
            kv_importance_tracker=kv_importance_tracker,
            staged_pending_self_kv=staged_pending_self_kv,
        )
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if not skip_mask:
            hidden_states = hidden_states + self.dropout(attention_output[0])
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        else:
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them     

        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.config.is_decoder:
            self.attn_ffn_time = self.SelfAttention.attn_ffn_time + norm_time + (datetime.datetime.now() - start)
            self.key_value_gen_time = self.SelfAttention.key_value_gen_time
        return outputs


class DeployT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.EncDecAttention = DeployT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        skip_mask=False,
        parallel_mask=False,
        gen_cross_attn_key_value=False,
    ):
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if (not skip_mask and not gen_cross_attn_key_value) or parallel_mask:
            normed_hidden_states = self.layer_norm(hidden_states)
        else: normed_hidden_states = hidden_states
        if self.config.use_synchronize: torch.cuda.synchronize()
        norm_time = (datetime.datetime.now() - start)
        
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            skip_mask=skip_mask,
            gen_cross_attn_key_value=gen_cross_attn_key_value,
        )
        if gen_cross_attn_key_value:
            self.key_value_gen_time = self.EncDecAttention.key_value_gen_time
            return attention_output  # non-autoregressively generated key_value_states
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if not skip_mask:
            hidden_states = hidden_states + self.dropout(attention_output[0])
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        else:
            outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them   

        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.config.is_decoder:
            self.attn_ffn_time = self.EncDecAttention.attn_ffn_time + norm_time + (datetime.datetime.now() - start)
            self.key_value_gen_time = self.EncDecAttention.key_value_gen_time 
        return outputs


class DeployT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(DeployT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(DeployT5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

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
        staged_pending_self_kv=None,
    ):

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
            staged_pending_self_kv=staged_pending_self_kv,
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


class DeployT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [DeployT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.device_map = None
        self.gradient_checkpointing = False

        # Early-Exit framework
        self.use_early_exit = config.use_early_exit
        self.exit_min_layer = config.exit_min_layer
            
        # Shallow-Deep Module
        self.use_shallow_deep = config.use_shallow_deep
        self.shallow_exit_layer = config.shallow_exit_layer
        if self.is_decoder and config.use_shallow_deep:
            assert config.shallow_exit_layer > 0 and config.shallow_exit_layer < len(self.block)
        
        # Synchronized Parallel Decoding
        self.block_op = [0] * config.num_layers  # to calculate the average number of forward block layers
        self.parallel_tokens_shallow = 0  # how much tokens are used in parallel decoding as stack_hidden_states
        self.parallel_tokens_deep = 0  # how much tokens are used in parallel decoding with skip_mask = False
        self.stack_hidden_states = ()  # store hidden_states that do not forward Deep decoder
        self.stack_hidden_metadata = ()  # JSON-safe metadata aligned with stack_hidden_states
        self.stack_source_kv_records = ()  # source-layer K/V dump records aligned with stack_hidden_metadata
        self.stack_phase3c_source_hidden_records = ()  # Phase 3c raw-hidden source records aligned with stack_hidden_metadata
        self.stack_fixed_layer_calibration_records = ()  # Native FREE fixed-source-layer-6 exact-cache calibration records aligned with stack_hidden_metadata
        self.kv_runtime_restorer = None
        if self.is_decoder and getattr(config, "kv_runtime_restoration_enabled", False):
            self.kv_runtime_restorer = RuntimeKVRestorationManager.from_path(
                getattr(config, "kv_runtime_restoration_artifact", None),
                getattr(config, "kv_runtime_restoration_method", "source_procrustes"),
                threshold=getattr(config, "kv_runtime_restoration_threshold", None),
                model_config=config,
            )
        self.exact_cache_calibration_collector = None
        if self.is_decoder and bool(getattr(config, "kv_early_exit_exact_cache_calibration_enabled", False)):
            self.exact_cache_calibration_collector = ExactCacheCalibrationCollector.from_config(config)
        self.kv_f2a_artifact = None
        self.kv_f2a_policy_sha256 = None
        self.kv_f2a_candidate_policy_sha256 = None
        self.kv_f2a_artifact_sha256 = None
        if self.is_decoder and getattr(config, "kv_f2a_frozen_schedule_enabled", False):
            artifact_path = getattr(config, "kv_f2a_policy_artifact", None)
            if not artifact_path:
                raise ValueError("f2a_policy_artifact_required")
            source_mode = getattr(config, "kv_f2a_source_layer_mode", SOURCE_LAYER_MODE_FIXED)
            if source_mode == SOURCE_LAYER_MODE_FIXED:
                if not bool(getattr(config, "use_shallow_deep", False)) or bool(getattr(config, "use_early_exit", False)):
                    raise ValueError("f2a_fixed_mode_requires_free_shallow_deep_reference")
                if bool(getattr(config, "kv_runtime_restoration_enabled", False)):
                    raise ValueError("f2a_fixed_mode_rejects_runtime_restoration_overwrite")
            elif source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
                if not bool(getattr(config, "use_early_exit", False)) or bool(getattr(config, "use_shallow_deep", False)):
                    raise ValueError("f2a_calm_mode_requires_calm_early_exit_reference")
                if bool(getattr(config, "kv_runtime_restoration_enabled", False)):
                    raise ValueError("f2a_calm_mode_rejects_taskc1_overwrite")
            else:
                raise ValueError("f2a_source_layer_mode_unsupported")
            self.kv_f2a_artifact = load_phase3c_policy_artifact(artifact_path)
            validation = validate_f2a_artifact_for_source_mode(
                self.kv_f2a_artifact,
                source_layer_mode=source_mode,
                decoder_layer_count=getattr(config, "num_layers", len(self.block)),
            )
            if validation.get("status") != "ok":
                raise ValueError("F2a artifact validation failed: {}".format(validation.get("errors")))
            self.kv_f2a_policy_sha256 = artifact_policy_sha256(self.kv_f2a_artifact, require_authoritative=True)
            self.kv_f2a_candidate_policy_sha256 = calm_policy_sha256()
            from our_kv_restoration.missing_kv_dump_provenance import sha256_file as _f2a_sha256_file

            self.kv_f2a_artifact_sha256 = _f2a_sha256_file(artifact_path)
        self.missing_kv_accounting = MissingKVRuntimeAccounting(task_c1_exact_overwrite=True)
        self.missing_kv_component_timer = MissingKVComponentTimer(
            enabled=bool(getattr(config, "kv_runtime_component_timing_enabled", False)),
            backend=getattr(config, "kv_runtime_component_timing_backend", "auto"),
        )
        self.missing_kv_exact_catchup_overhead_recorder = ExactCatchupOverheadRecorder()
        self._missing_kv_evaluation_active = False
        self._kv_runtime_restoration_counters = {}
        self._reset_runtime_restoration_generation_state()
        self.reset_missing_kv_evaluation_aggregates(aggregation_scope="standalone_generation")
        self.kv_trace = KVTraceRecorder(
            enabled=self.is_decoder and getattr(config, "kv_trace_enabled", False),
            max_records=getattr(config, "kv_trace_max_records", 100000),
        )
        self.kv_importance = H2OImportanceTracker(
            enabled=self.is_decoder and getattr(config, "kv_importance_enabled", False),
            num_layers=config.num_layers,
            mode=getattr(config, "kv_importance_mode", "h2o_layer"),
            decay=getattr(config, "kv_importance_decay", 1.0),
            include_current=getattr(config, "kv_importance_include_current", True),
        )
        
        # Adaptive Threshold Estimator
        self.bmm_model = BetaMixture1D()
        self.bmm_threshold = None
        self.stack_conf, self.stack_pred = (), ()
        self.stack_conf_all, self.stack_ident_all = (), ()
        self._smoke_forced_flush_used = False
        self._tail_pending_recorded = False
        self._exact_catchup_dump_next_flush_index = 0
        self._source_kv_dumped_tokens = 0
        self._adjacent_anchor_dumped_tokens = 0
        self._all_layer_calib_dumped_records = 0
        self._all_layer_calib_dumped_token_ids = set()
        self._all_layer_calib_packed_records = 0
        self._all_layer_hidden_dumped_records = 0
        self._all_layer_hidden_dumped_token_ids = set()
        self._all_layer_hidden_dump_failures = 0
        self._all_layer_hidden_packed_records = 0
        self._missing_kv_packed_generation = None
        self._attention_diag_dumped_records = 0
        self._full_attention_diag_dumped_records = 0
        self._restoration_dryrun_record_count = 0
        self._eaes_exported_source_record_ids = set()
        self._eaes_export_next_flush_index = 0
        self._runtime_restoration_next_flush_id = 0
        self._f2a_schedule_rows = []
        self._f2a_records = []
        self._f2a_skipped_events = []
        self._f2a_reference_trajectory_rows = []
        self._f2a_reference_generated_token_ids = []
        self._f2a_reference_decision_trace = []
        self._f2a_current_reference_decision = None
        self._f2a_reference_generation_finalized = False
        self._f2a_debug_logit_records_written = 0
        self._f2a_current_prefix_token_ids = None
        self._f2a_current_decoder_input_token_id = None
        self._f2a_current_decoder_input_position = None
        self._f2a_current_predicted_token_position = None
        self._f2a_pending_calm_event = None
        self._generation_index = -1
        self._missing_kv_generation_sample_context = None
        if self.is_decoder:
            self._reset_time_measure()
        else:
            self.deploy_time = None

    def set_missing_kv_generation_sample_context(self, context):
        self._missing_kv_generation_sample_context = dict(context or {})

    def clear_missing_kv_generation_sample_context(self):
        self._missing_kv_generation_sample_context = None

    def _missing_kv_provenance_enabled(self):
        return bool(getattr(self.config, "missing_kv_provenance_enabled", False))

    def _missing_kv_dump_storage_format(self):
        return str(getattr(self.config, "missing_kv_dump_storage_format", "legacy_row_v1") or "legacy_row_v1")

    def _missing_kv_packed_generation_enabled(self):
        return self._missing_kv_provenance_enabled() and self._missing_kv_dump_storage_format() == PACKED_GENERATION_STORAGE_FORMAT

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

    def _attach_missing_kv_dump_row_uid(self, row):
        if self._missing_kv_provenance_enabled():
            row["dump_row_uid"] = make_dump_row_uid(row)
        return row

    def _record_missing_kv_generation_binding(self):
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

    def _calm_counterfactual_trace_enabled(self):
        return bool(getattr(self.config, "kv_calm_counterfactual_trace_enabled", False))

    def _calm_counterfactual_trace_path(self):
        path = getattr(self.config, "kv_calm_counterfactual_trace_output", None)
        return path if path else None

    def _calm_counterfactual_candidate_layers(self):
        return parse_candidate_layers(
            getattr(self.config, "kv_calm_counterfactual_candidate_layers", CALM_CANDIDATE_LAYERS)
        )

    def _calm_counterfactual_threshold(self):
        value = getattr(self.config, "kv_calm_counterfactual_threshold", CALM_THRESHOLD)
        return float(CALM_THRESHOLD if value is None else value)

    def _begin_missing_kv_packed_generation(self):
        if not (self.is_decoder and self._missing_kv_packed_generation_enabled()):
            return
        context = self._missing_kv_sample_context_fields()
        if context.get("missing_kv_sample_context_status") != "ok":
            raise ValueError("missing-KV packed dump requires sample context at generation reset")
        if self._missing_kv_packed_generation is not None:
            raise ValueError("missing-KV packed generation buffer was not finalized before next generation reset")
        self._missing_kv_packed_generation = {
            "generation_index": int(self._generation_index),
            "context": context,
            "hidden": {},
            "kv": {},
            "hidden_device_before_dump": None,
            "kv_device_before_dump": None,
            "hidden_dtype": None,
            "kv_dtype": None,
            "hidden_include_raw": bool(getattr(self.config, "kv_all_layer_hidden_dump_include_raw_hidden", False)),
            "hidden_include_normed": bool(getattr(self.config, "kv_all_layer_hidden_dump_include_normed_hidden", True)),
            "hidden_model_d_model": int(getattr(self.config, "d_model", 0)),
            "kv_model_num_heads": int(getattr(self.config, "num_heads", 0)),
            "kv_model_d_kv": int(getattr(self.config, "d_kv", 0)),
            "model_num_decoder_layers": len(self.block),
            "calm_trace_enabled": self._calm_counterfactual_trace_enabled(),
            "calm_trace": {},
            "calm_candidate_layers": tuple(self._calm_counterfactual_candidate_layers()),
            "calm_threshold": self._calm_counterfactual_threshold(),
        }

    def abort_missing_kv_generation_dump(self, reason="generation_failed"):
        if self._missing_kv_packed_generation is not None:
            self._missing_kv_packed_generation["abort_reason"] = str(reason)
        self._missing_kv_packed_generation = None
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is not None:
            collector.abort_pending()

    def _exact_cache_calibration_enabled(self):
        return getattr(self, "exact_cache_calibration_collector", None) is not None

    def _fixed_layer_exact_cache_calibration_active(self):
        collector = getattr(self, "exact_cache_calibration_collector", None)
        return collector is not None and getattr(collector, "source_layer_mode", None) == SOURCE_LAYER_MODE_FIXED

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
        """Stage one Native FREE fixed-source-layer exact-cache calibration
        event using data the official synchronized exact-deep path already
        computed. Performs no additional deep computation of its own.

        ``hidden_by_layer`` and ``key_value_by_layer`` must each span every
        decoder layer (0..len(self.block)-1), single position per layer for
        this token -- the collector's tensor population checks require full
        coverage even though only layers >= fixed_source_layer are ever used
        downstream.

        ``event_origin`` distinguishes the official synchronized parallel
        flush (default, when omitted) from calibration-only terminal exact
        finalization; it only changes staged provenance, never the tensors
        themselves."""

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
        # Deferred fixed-layer exits are not staged into the collector's own
        # pending slot at exit time (the exact hidden/K/V population isn't
        # available until a later synchronized flush). Instead, the most
        # recently created stack-aligned record (if any, and if not already
        # verified) is checked here, immediately, against the token the
        # generation loop actually selected -- never against a later
        # no-crossing token's identity. A mismatch invalidates generation
        # rather than silently collecting an unverified event.
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
        collector.commit_selected_token(actual_token_id)

    def _packed_generation_buffer(self):
        if self._missing_kv_packed_generation is None:
            self._begin_missing_kv_packed_generation()
        return self._missing_kv_packed_generation

    def _accumulate_packed_hidden_slice(
        self,
        *,
        layer_idx,
        token_position,
        token_offset,
        past_len,
        raw_slice=None,
        normed_slice=None,
        device_before_dump=None,
    ):
        buffer = self._packed_generation_buffer()
        if buffer is None:
            return
        key = (int(token_position), int(layer_idx))
        if key in buffer["hidden"]:
            raise ValueError("duplicate_packed_hidden_slice:pos{}_layer{}".format(token_position, layer_idx))
        record = {
            "token_offset_in_forward": int(token_offset),
            "self_attn_past_len_before_layer": past_len,
            "raw_hidden": None,
            "normed_hidden": None,
        }
        if raw_slice is not None:
            squeezed = raw_slice.squeeze(0).squeeze(0).contiguous()
            if int(squeezed.dim()) != 1:
                raise ValueError("unexpected_packed_raw_hidden_shape:{}".format(list(raw_slice.shape)))
            record["raw_hidden"] = squeezed
            buffer["hidden_dtype"] = str(squeezed.dtype).replace("torch.", "")
        if normed_slice is not None:
            squeezed = normed_slice.squeeze(0).squeeze(0).contiguous()
            if int(squeezed.dim()) != 1:
                raise ValueError("unexpected_packed_normed_hidden_shape:{}".format(list(normed_slice.shape)))
            record["normed_hidden"] = squeezed
            buffer["hidden_dtype"] = buffer["hidden_dtype"] or str(squeezed.dtype).replace("torch.", "")
        buffer["hidden_device_before_dump"] = buffer["hidden_device_before_dump"] or device_before_dump
        buffer["hidden"][key] = record

    def _accumulate_packed_kv_slice(
        self,
        *,
        layer_idx,
        token_position,
        token_offset,
        past_len,
        key_slice,
        value_slice,
        device_before_dump=None,
    ):
        buffer = self._packed_generation_buffer()
        if buffer is None:
            return
        key = (int(token_position), int(layer_idx))
        if key in buffer["kv"]:
            raise ValueError("duplicate_packed_kv_slice:pos{}_layer{}".format(token_position, layer_idx))
        key_tensor = key_slice.squeeze(0).squeeze(1).contiguous()
        value_tensor = value_slice.squeeze(0).squeeze(1).contiguous()
        if int(key_tensor.dim()) != 2 or int(value_tensor.dim()) != 2:
            raise ValueError("unexpected_packed_kv_slice_shape:key{}:value{}".format(list(key_slice.shape), list(value_slice.shape)))
        if tuple(key_tensor.shape) != tuple(value_tensor.shape):
            raise ValueError("packed_key_value_shape_mismatch")
        buffer["kv_dtype"] = str(key_tensor.dtype).replace("torch.", "")
        buffer["kv_device_before_dump"] = buffer["kv_device_before_dump"] or device_before_dump
        buffer["kv"][key] = {
            "token_offset_in_forward": int(token_offset),
            "self_attn_past_len_before_layer": past_len,
            "key": key_tensor,
            "value": value_tensor,
        }

    def _maybe_record_calm_counterfactual_confidence(
        self,
        *,
        layer_idx,
        hidden_states,
        past_key_values,
        auto_reg,
        lm_head,
    ):
        if not (self.is_decoder and self._calm_counterfactual_trace_enabled()):
            return
        if not self._missing_kv_packed_generation_enabled():
            raise ValueError("CALM counterfactual trace requires packed-generation dump storage")
        if self.use_shallow_deep or self.use_early_exit or self.config.static_exit_layer is not None:
            raise ValueError("CALM counterfactual trace requires full-depth decoder execution")
        if bool(getattr(self.config, "kv_runtime_restoration_enabled", False)):
            raise ValueError("CALM counterfactual trace requires kv_runtime_restoration_enabled=False")
        if bool(getattr(self.config, "kv_runtime_restoration_calm_enabled", False)):
            raise ValueError("CALM counterfactual trace requires kv_runtime_restoration_calm_enabled=False")
        if not auto_reg:
            raise ValueError("CALM counterfactual trace requires autoregressive single-token decoding")
        candidate_layers = self._calm_counterfactual_candidate_layers()
        if int(layer_idx) not in candidate_layers:
            return
        buffer = self._packed_generation_buffer()
        if buffer is None:
            return
        decoder_position = infer_decoder_position(past_key_values)
        if decoder_position is None:
            decoder_position = 0
        candidate_map = buffer.setdefault("calm_trace", {}).setdefault(int(decoder_position), {})
        if int(layer_idx) in candidate_map:
            raise ValueError("duplicate_calm_counterfactual_confidence:pos{}_layer{}".format(decoder_position, layer_idx))
        _lm_logits, confidence = compute_calm_candidate_logits_and_confidence(
            hidden_states,
            final_layer_norm=self.final_layer_norm,
            dropout=self.dropout,
            lm_head=lm_head,
            config=self.config,
        )
        candidate_map[int(layer_idx)] = confidence

    def _packed_generation_ranges(self, records, label):
        if not records:
            return None
        positions = sorted({int(key[0]) for key in records})
        layers = sorted({int(key[1]) for key in records})
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("{}_packed_decoder_positions_not_contiguous".format(label))
        if layers != list(range(layers[0], layers[-1] + 1)):
            raise ValueError("{}_packed_layers_not_contiguous".format(label))
        for position in positions:
            for layer_idx in layers:
                if (position, layer_idx) not in records:
                    raise ValueError("{}_packed_missing_position_layer:pos{}_layer{}".format(label, position, layer_idx))
        return positions, layers

    def _packed_common_manifest_fields(self, buffer, *, record_type, file_path, token_count, layer_count, positions, layers):
        row = {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": record_type,
            "dump_succeeded": True,
            "file_path": file_path,
            "stable_sample_id": buffer["context"].get("stable_sample_id"),
            "selected_order": buffer["context"].get("selected_order"),
            "raw_dataset_index": buffer["context"].get("raw_dataset_index"),
            "dataset_provided_id": buffer["context"].get("dataset_provided_id"),
            "missing_kv_sample_context_status": buffer["context"].get("missing_kv_sample_context_status"),
            "generation_index": int(buffer["generation_index"]),
            "token_count": int(token_count),
            "layer_count": int(layer_count),
            "decoder_position_start": int(positions[0]),
            "decoder_position_end_exclusive": int(positions[-1] + 1),
            "decoder_positions_contiguous": True,
            "layer_start": int(layers[0]),
            "layer_end_exclusive": int(layers[-1] + 1),
            "layers_contiguous": True,
            "model_num_decoder_layers": int(buffer["model_num_decoder_layers"]),
            "full_depth_execution": bool(
                not self.use_shallow_deep
                and not self.use_early_exit
                and self.config.static_exit_layer is None
            ),
            "use_shallow_deep": bool(self.use_shallow_deep),
            "use_early_exit": bool(self.use_early_exit),
            "static_exit_layer": self.config.static_exit_layer,
        }
        return row

    def _finalize_packed_hidden_generation(self, buffer):
        records = buffer.get("hidden") or {}
        ranges = self._packed_generation_ranges(records, "hidden")
        if ranges is None:
            raise ValueError("packed_hidden_records_missing")
        positions, layers = ranges
        raw_tensors = []
        normed_tensors = []
        for position in positions:
            raw_by_layer = []
            normed_by_layer = []
            for layer_idx in layers:
                item = records[(position, layer_idx)]
                if buffer["hidden_include_raw"]:
                    if item.get("raw_hidden") is None:
                        raise ValueError("packed_raw_hidden_missing:pos{}_layer{}".format(position, layer_idx))
                    raw_by_layer.append(item["raw_hidden"])
                if buffer["hidden_include_normed"]:
                    if item.get("normed_hidden") is None:
                        raise ValueError("packed_normed_hidden_missing:pos{}_layer{}".format(position, layer_idx))
                    normed_by_layer.append(item["normed_hidden"])
            if raw_by_layer:
                raw_tensors.append(torch.stack(raw_by_layer, dim=0))
            if normed_by_layer:
                normed_tensors.append(torch.stack(normed_by_layer, dim=0))
        payload = {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": "packed_generation_hidden",
            "metadata": None,
        }
        raw_hidden = torch.stack(raw_tensors, dim=0).contiguous() if raw_tensors else None
        normed_hidden = torch.stack(normed_tensors, dim=0).contiguous() if normed_tensors else None
        if raw_hidden is not None:
            payload["raw_hidden"] = raw_hidden
        if normed_hidden is not None:
            payload["normed_hidden"] = normed_hidden
        dump_dir = self._all_layer_hidden_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        file_path = os.path.join(dump_dir, "hidden_generation_{}.pt".format(buffer["generation_index"]))
        row = self._packed_common_manifest_fields(
            buffer,
            record_type="packed_generation_hidden",
            file_path=file_path,
            token_count=len(positions),
            layer_count=len(layers),
            positions=positions,
            layers=layers,
        )
        row.update(
            {
                "include_raw_hidden": bool(buffer["hidden_include_raw"]),
                "include_normed_hidden": bool(buffer["hidden_include_normed"]),
                "raw_hidden_shape": list(raw_hidden.shape) if raw_hidden is not None else None,
                "normed_hidden_shape": list(normed_hidden.shape) if normed_hidden is not None else None,
                "dtype": buffer.get("hidden_dtype"),
                "device_before_dump": buffer.get("hidden_device_before_dump"),
                "model_d_model": int(buffer.get("hidden_model_d_model", 0)),
            }
        )
        row = populate_packed_record_identities(row, logical_record_type=HIDDEN_LOGICAL_RECORD_TYPE)
        payload["metadata"] = dict(row)
        return row, payload, file_path

    def _finalize_packed_kv_generation(self, buffer):
        records = buffer.get("kv") or {}
        ranges = self._packed_generation_ranges(records, "kv")
        if ranges is None:
            raise ValueError("packed_kv_records_missing")
        positions, layers = ranges
        key_tensors = []
        value_tensors = []
        for position in positions:
            key_by_layer = []
            value_by_layer = []
            for layer_idx in layers:
                item = records[(position, layer_idx)]
                key_by_layer.append(item["key"])
                value_by_layer.append(item["value"])
            key_tensors.append(torch.stack(key_by_layer, dim=0))
            value_tensors.append(torch.stack(value_by_layer, dim=0))
        key_tensor = torch.stack(key_tensors, dim=0).contiguous()
        value_tensor = torch.stack(value_tensors, dim=0).contiguous()
        dump_dir = self._all_layer_calib_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        file_path = os.path.join(dump_dir, "kv_generation_{}.pt".format(buffer["generation_index"]))
        row = self._packed_common_manifest_fields(
            buffer,
            record_type="packed_generation_kv",
            file_path=file_path,
            token_count=len(positions),
            layer_count=len(layers),
            positions=positions,
            layers=layers,
        )
        row.update(
            {
                "key_shape": list(key_tensor.shape),
                "value_shape": list(value_tensor.shape),
                "dtype": buffer.get("kv_dtype"),
                "device_before_dump": buffer.get("kv_device_before_dump"),
                "model_num_heads": int(buffer.get("kv_model_num_heads", 0)),
                "model_d_kv": int(buffer.get("kv_model_d_kv", 0)),
            }
        )
        row = populate_packed_record_identities(row, logical_record_type=KV_LOGICAL_RECORD_TYPE)
        payload = {
            "manifest_schema_version": 2,
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
            "record_type": "packed_generation_kv",
            "key": key_tensor,
            "value": value_tensor,
            "metadata": dict(row),
        }
        return row, payload, file_path

    def _finalize_calm_trace_generation(self, buffer, hidden_row, kv_row):
        if not buffer.get("calm_trace_enabled"):
            return [], None
        output_path = self._calm_counterfactual_trace_path()
        if not output_path:
            raise ValueError("kv_calm_counterfactual_trace_output is required when CALM trace is enabled")
        positions = list(
            range(
                int(hidden_row.get("decoder_position_start", 0)),
                int(hidden_row.get("decoder_position_end_exclusive", 0)),
            )
        )
        candidate_layers = tuple(buffer.get("calm_candidate_layers") or self._calm_counterfactual_candidate_layers())
        threshold = float(buffer.get("calm_threshold", CALM_THRESHOLD))
        trace_records = buffer.get("calm_trace") or {}
        rows = []
        for position in positions:
            candidate_map = trace_records.get(int(position))
            if not isinstance(candidate_map, dict):
                raise ValueError("calm_trace_missing_token_position:{}".format(position))
            missing_layers = [layer for layer in candidate_layers if layer not in candidate_map]
            if missing_layers:
                raise ValueError(
                    "calm_trace_missing_candidate_layers:pos{}:{}".format(
                        position,
                        ",".join(str(layer) for layer in missing_layers),
                    )
                )
            rows.append(
                build_calm_trace_row(
                    sample_context=buffer["context"],
                    generation_index=int(buffer["generation_index"]),
                    decoder_position=int(position),
                    candidate_confidences=candidate_map,
                    model_num_decoder_layers=int(buffer["model_num_decoder_layers"]),
                    candidate_layers=candidate_layers,
                    threshold=threshold,
                )
            )
        binding_row = {
            "stable_sample_id": buffer["context"].get("stable_sample_id"),
            "selected_order": buffer["context"].get("selected_order"),
            "generation_index": int(buffer["generation_index"]),
            "raw_dataset_index": buffer["context"].get("raw_dataset_index"),
            "dataset_provided_id": buffer["context"].get("dataset_provided_id"),
        }
        validation = validate_calm_trace_rows(
            rows,
            population_rows=[dict(buffer["context"])],
            binding_rows=[binding_row],
            hidden_rows=[hidden_row],
            kv_rows=[kv_row],
            candidate_layers=candidate_layers,
            threshold=threshold,
        )
        if validation.get("status") != "ok":
            raise ValueError(
                "calm_trace_in_memory_validation_failed:{}".format(
                    ",".join(str(error) for error in validation.get("errors", []))
                )
            )
        return rows, output_path

    def _packed_manifest_path(self, modality):
        if modality == "hidden":
            dump_dir = self._all_layer_hidden_dump_dir()
            file_name = "all_layer_hidden_manifest.jsonl"
        else:
            dump_dir = self._all_layer_calib_dump_dir()
            file_name = "all_layer_kv_manifest.jsonl"
        if dump_dir is None:
            raise ValueError("missing_{}_packed_dump_dir".format(modality))
        os.makedirs(dump_dir, exist_ok=True)
        return os.path.join(dump_dir, file_name)

    def _packed_manifest_state(self, path):
        return {
            "exists": os.path.exists(path),
            "size": os.path.getsize(path) if os.path.exists(path) else 0,
        }

    def _restore_packed_manifest_state(self, path, state):
        try:
            if state.get("exists"):
                with open(path, "r+b") as handle:
                    handle.truncate(int(state.get("size", 0)))
            elif os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.warning("failed to roll back packed manifest %s", path, exc_info=True)

    def _cleanup_packed_paths(self, paths):
        for path in paths:
            if not path:
                continue
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                logger.warning("failed to remove packed dump file %s", path, exc_info=True)

    def _write_packed_payload_temp(self, payload, final_path):
        parent = os.path.dirname(final_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_path = "{}.tmp".format(final_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        return temp_path

    def _replace_packed_temp(self, temp_path, final_path):
        os.replace(temp_path, final_path)

    def _validate_in_memory_packed_artifact(self, row, payload, *, logical_record_type, expected_record_type):
        expected_row = populate_packed_record_identities(row, logical_record_type=logical_record_type)
        for key in ("logical_row_count", "logical_population_sha256", "logical_dump_row_uid_aggregate_sha256", "packed_record_uid"):
            if row.get(key) != expected_row.get(key):
                raise ValueError("packed_in_memory_{}_mismatch".format(key))
        if payload.get("manifest_schema_version") != 2:
            raise ValueError("packed_in_memory_payload_schema_mismatch")
        if payload.get("storage_format") != PACKED_GENERATION_STORAGE_FORMAT:
            raise ValueError("packed_in_memory_payload_storage_format_mismatch")
        if payload.get("record_type") != expected_record_type:
            raise ValueError("packed_in_memory_payload_record_type_mismatch")
        if payload.get("metadata") != row:
            raise ValueError("packed_in_memory_payload_metadata_mismatch")
        for tensor_key in ("raw_hidden", "normed_hidden", "key", "value"):
            tensor = payload.get(tensor_key)
            if tensor is None:
                continue
            if not torch.isfinite(tensor.detach()).all().item():
                raise ValueError(
                    "packed_in_memory_nonfinite_tensor:{}:{}".format(
                        expected_record_type,
                        tensor_key,
                    )
                )

    def finalize_missing_kv_generation_dump(self):
        if not self._missing_kv_packed_generation_enabled():
            return
        buffer = self._missing_kv_packed_generation
        if buffer is None:
            return
        temp_paths = []
        final_paths = []
        hidden_manifest = None
        kv_manifest = None
        calm_trace_path = None
        hidden_manifest_state = None
        kv_manifest_state = None
        calm_trace_state = None
        committed = False
        try:
            hidden_row, hidden_payload, hidden_path = self._finalize_packed_hidden_generation(buffer)
            kv_row, kv_payload, kv_path = self._finalize_packed_kv_generation(buffer)
            calm_trace_rows, calm_trace_path = self._finalize_calm_trace_generation(buffer, hidden_row, kv_row)
            self._validate_in_memory_packed_artifact(
                hidden_row,
                hidden_payload,
                logical_record_type=HIDDEN_LOGICAL_RECORD_TYPE,
                expected_record_type="packed_generation_hidden",
            )
            self._validate_in_memory_packed_artifact(
                kv_row,
                kv_payload,
                logical_record_type=KV_LOGICAL_RECORD_TYPE,
                expected_record_type="packed_generation_kv",
            )
            if os.path.exists(hidden_path):
                raise ValueError("packed_hidden_shard_already_exists:{}".format(hidden_path))
            if os.path.exists(kv_path):
                raise ValueError("packed_kv_shard_already_exists:{}".format(kv_path))
            temp_paths = ["{}.tmp".format(hidden_path), "{}.tmp".format(kv_path)]
            hidden_temp = self._write_packed_payload_temp(hidden_payload, hidden_path)
            kv_temp = self._write_packed_payload_temp(kv_payload, kv_path)
            self._replace_packed_temp(hidden_temp, hidden_path)
            final_paths.append(hidden_path)
            self._replace_packed_temp(kv_temp, kv_path)
            final_paths.append(kv_path)
            hidden_manifest = self._packed_manifest_path("hidden")
            kv_manifest = self._packed_manifest_path("kv")
            hidden_manifest_state = self._packed_manifest_state(hidden_manifest)
            kv_manifest_state = self._packed_manifest_state(kv_manifest)
            if calm_trace_path is not None:
                calm_trace_state = self._packed_manifest_state(calm_trace_path)
            try:
                self._append_jsonl(hidden_manifest, hidden_row)
                self._append_jsonl(kv_manifest, kv_row)
                if calm_trace_path is not None:
                    for trace_row in calm_trace_rows:
                        self._append_jsonl(calm_trace_path, trace_row)
            except Exception:
                self._restore_packed_manifest_state(hidden_manifest, hidden_manifest_state)
                self._restore_packed_manifest_state(kv_manifest, kv_manifest_state)
                if calm_trace_path is not None:
                    self._restore_packed_manifest_state(calm_trace_path, calm_trace_state or {"exists": False, "size": 0})
                self._cleanup_packed_paths(final_paths + temp_paths)
                raise
            self._all_layer_hidden_packed_records += 1
            self._all_layer_calib_packed_records += 1
            committed = True
            self._write_all_layer_hidden_dump_summary(self._all_layer_hidden_dump_dir())
        except Exception:
            if not committed:
                self._cleanup_packed_paths(final_paths + temp_paths)
            raise
        finally:
            self._missing_kv_packed_generation = None

    def _exact_catchup_trace_fields(
        self,
        start_layer,
        pending_skipped_tokens,
        pending_metadata_trace=None,
        metadata_positions=None,
        metadata_positions_available=False,
    ):
        if start_layer is None:
            end_layer = None
            num_catchup_layers = None
            missing_layer_range = []
        else:
            start_layer = int(start_layer)
            end_layer = len(self.block) - 1
            missing_layer_range = list(range(start_layer, len(self.block)))
            num_catchup_layers = len(missing_layer_range)
        pending_skipped_tokens = int(pending_skipped_tokens or 0)
        exact_catchup_token_layer_units = None
        if num_catchup_layers is not None:
            exact_catchup_token_layer_units = pending_skipped_tokens * num_catchup_layers

        batch_size = None
        if pending_skipped_tokens and isinstance(self.stack_hidden_states, tuple) and self.stack_hidden_states:
            shape = getattr(self.stack_hidden_states[0], "shape", None)
            if shape is not None and len(shape) >= 1:
                batch_size = int(shape[0])
        exact_catchup_kv_shapes = []
        if batch_size is not None and pending_skipped_tokens and missing_layer_range:
            kv_shape = [
                batch_size,
                int(getattr(self.config, "num_heads", 0)),
                pending_skipped_tokens,
                int(getattr(self.config, "d_kv", 0)),
            ]
            exact_catchup_kv_shapes = [
                {
                    "layer_idx": int(layer_idx),
                    "key_shape": list(kv_shape),
                    "value_shape": list(kv_shape),
                }
                for layer_idx in missing_layer_range
            ]

        return {
            "end_layer": end_layer,
            "num_catchup_layers": num_catchup_layers,
            "exact_catchup_token_layer_units": exact_catchup_token_layer_units,
            "exact_catchup_layer_range": missing_layer_range,
            "exact_catchup_decoder_positions": metadata_positions,
            "exact_catchup_positions_available": metadata_positions_available,
            "exact_catchup_metadata": pending_metadata_trace,
            "exact_catchup_kv_shapes": exact_catchup_kv_shapes,
            "exact_catchup_source": "parallel_gen_token",
            "exact_parallel_catchup": True,
            "kv_importance_enabled": bool(getattr(self.config, "kv_importance_enabled", False)),
            "kv_importance_mode": getattr(self.config, "kv_importance_mode", None),
            "raw_kv_dump_enabled": False,
        }

    def record_tail_pending_skips(self, reason="generation_end"):
        if not self.is_decoder:
            return
        pending_skipped_tokens = len(self.stack_hidden_states)
        pending_metadata = self.stack_hidden_metadata
        pending_metadata_count = len(pending_metadata) if isinstance(pending_metadata, tuple) else 0
        if pending_skipped_tokens <= 0 and pending_metadata_count <= 0:
            return
        if getattr(self, "_tail_pending_recorded", False):
            return

        # Under the FREE-aligned lazy batched schedule these terminal pending
        # exits are AVOIDED work, not lost work: generation ended before any
        # later token needed their deep K/V, so neither Phase-3c restoration
        # nor exact replay was performed for them, by design. Recorded through
        # this existing tail-pending hook rather than a new mechanism.
        if self._task_c2_batched_insertion_enabled():
            self._missing_kv_accounting_obj().record_task_c2_batched_terminal_pending(
                pending_skipped_tokens
            )

        # Pending actual exits that never reached a synchronized flush before
        # generation ended (or restarted for the next sample) must never be
        # silently dropped from the fixed-layer calibration record: there is
        # no official exact tail-flush path, so this is surfaced as an
        # explicit invalid/incomplete collection rather than fabricated or
        # ignored. This must run regardless of kv_trace being enabled.
        calibration_collector = getattr(self, "exact_cache_calibration_collector", None)
        if calibration_collector is not None and self._fixed_layer_exact_cache_calibration_active():
            calibration_collector.record_tail_pending_loss(pending_skipped_tokens, reason=reason)

        if not hasattr(self, "kv_trace") or not self.kv_trace.enabled:
            self._tail_pending_recorded = True
            return

        pending_metadata_trace, pending_metadata_truncated = metadata_list_to_trace(pending_metadata)
        metadata_positions = candidate_positions_from_metadata(pending_metadata)
        metadata_positions_available = bool(metadata_positions and any(position is not None for position in metadata_positions))
        metadata_positions_trace = metadata_positions
        if metadata_positions_trace is not None and len(metadata_positions_trace) > 128:
            metadata_positions_trace = metadata_positions_trace[:128]
            pending_metadata_truncated = True

        self._maybe_export_eaes_scores_for_pending(
            pending_metadata,
            self.stack_source_kv_records,
            flush_index=None,
            reason=reason,
        )

        self.kv_trace.record(
            "tail_pending_skips",
            reason=reason,
            pending_skipped_tokens=pending_skipped_tokens,
            pending_metadata_count=pending_metadata_count,
            pending_metadata=pending_metadata_trace,
            pending_metadata_truncated=pending_metadata_truncated,
            metadata_positions=metadata_positions_trace,
            metadata_positions_available=metadata_positions_available,
            start_layer=self.shallow_exit_layer,
            shallow_exit_layer=self.shallow_exit_layer,
            num_layers=len(self.block),
        )
        self._tail_pending_recorded = True

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                item = value.item()
                if isinstance(item, (str, int, float, bool)) or item is None:
                    return item
            except Exception:
                pass
        return str(value)

    def _reset_runtime_restoration_generation_state(self):
        if "config" in self.__dict__ and bool(getattr(self.config, "kv_f2a_frozen_schedule_enabled", False)):
            self.clear_f2a_generation_step_context()
        # Batched Task C2 cross-attention trust is generation-local by
        # construction: cross K/V are derived from THIS generation's encoder
        # states, so nothing learned about them may outlive the generation.
        # Dropping the dict here also drops the tensor references it holds.
        self._task_c2_batched_cross_kv_trust = {}
        self._kv_runtime_restoration_counters = {
            "pending_skipped_tokens": 0,
            "restored_token_count": 0,
            "restored_token_layer_units": 0,
            "exact_catchup_token_count": 0,
            "exact_catchup_token_layer_units": 0,
            "exact_catchup_computed_token_layer_units": 0,
            "exact_cache_retained_token_layer_units": 0,
            "exact_overwrite_token_layer_units": 0,
            "exact_catchup_avoided_token_layer_units": 0,
            "fallback_exact_count": 0,
            "missing_map_count": 0,
            "cache_shape_mismatch_count": 0,
            "nan_or_inf_count": 0,
            "unexpected_runtime_error_count": 0,
            "unsupported_batch_count": 0,
            "phase3c_restore_attempt_count": 0,
            "phase3c_restore_success_count": 0,
            "exact_overwrite_count": 0,
            "missing_threshold_count": 0,
            "missing_hidden_pair_count": 0,
            "missing_k_pair_count": 0,
            "missing_v_gap_count": 0,
            "calm_early_exit_tokens": 0,
            "calm_target_token_layer_units": 0,
            "calm_restored_token_layer_units": 0,
            "calm_state_copy_fallback_token_layer_units": 0,
            "calm_missing_map_count": 0,
            "calm_cache_shape_mismatch_count": 0,
            "calm_nan_or_inf_count": 0,
            "calm_unsupported_batch_count": 0,
            "calm_phase3c_first_crossing_tokens": 0,
            "calm_phase3c_full_depth_fallback_tokens": 0,
            "calm_phase3c_transaction_failure_tokens": 0,
            "calm_phase3c_requested_token_layer_units": 0,
            "calm_phase3c_succeeded_token_layer_units": 0,
            "calm_phase3c_overwritten_token_layer_units": 0,
            "calm_phase3c_failed_token_layer_units": 0,
        }

    def reset_missing_kv_evaluation_aggregates(self, aggregation_scope="standalone_generation"):
        active = bool(getattr(self, "_missing_kv_evaluation_active", False))
        self._missing_kv_accounting_obj().reset(
            aggregation_scope=aggregation_scope,
            evaluation_aggregation_active=active,
        )
        self._missing_kv_component_timer_obj().reset()
        self._missing_kv_exact_catchup_overhead_recorder_obj().reset()

    def begin_missing_kv_evaluation(self):
        self._missing_kv_evaluation_active = True
        self.reset_missing_kv_evaluation_aggregates(aggregation_scope="evaluation")
        self._reset_runtime_restoration_generation_state()
        # Pure-recovery shadow measurement state is evaluation-scoped.
        self._pure_recovery_events = []
        self._calm_pure_recovery_events = []
        if "pure_recovery_component_timer" in self.__dict__:
            self.__dict__["pure_recovery_component_timer"].reset()
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is not None:
            collector.reset_run()

    def end_missing_kv_evaluation(self):
        self._missing_kv_evaluation_active = False
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is not None:
            collector.close()
        if "missing_kv_accounting" in self.__dict__:
            self.__dict__["missing_kv_accounting"].set_lifecycle(
                aggregation_scope="evaluation",
                evaluation_aggregation_active=False,
            )

    def _snapshot_f2a_generation_step_context(self):
        """Capture the already-prepared current-step F2a context before a
        per-generation reset would otherwise wipe it. Returns None when F2a is
        disabled or no step context has been prepared yet."""

        if not self._f2a_enabled():
            return None
        if getattr(self, "_f2a_current_prefix_token_ids", None) is None:
            return None
        return {
            "prefix_token_ids": getattr(self, "_f2a_current_prefix_token_ids", None),
            "decoder_input_token_id": getattr(self, "_f2a_current_decoder_input_token_id", None),
            "decoder_input_position": getattr(self, "_f2a_current_decoder_input_position", None),
            "predicted_token_position": getattr(self, "_f2a_current_predicted_token_position", None),
            "reference_decision": copy.deepcopy(getattr(self, "_f2a_current_reference_decision", None)),
        }

    def _restore_f2a_generation_step_context(self, snapshot):
        """Restore a step context captured by ``_snapshot_f2a_generation_step_context``
        after the per-generation arrays/counters it depends on have already
        been reset. ``generated_token_offset`` is forced to 0 regardless of
        what it was at snapshot time: by construction this restores the FIRST
        decision of a freshly-reset (now empty) generation, and the snapshot
        may have been computed while a previous generation's trailing token
        count was still in scope."""

        if not snapshot or snapshot.get("prefix_token_ids") is None:
            return
        self._f2a_current_prefix_token_ids = snapshot["prefix_token_ids"]
        self._f2a_current_decoder_input_token_id = snapshot["decoder_input_token_id"]
        self._f2a_current_decoder_input_position = snapshot["decoder_input_position"]
        self._f2a_current_predicted_token_position = snapshot["predicted_token_position"]
        decision = snapshot.get("reference_decision")
        if decision is not None:
            decision = dict(decision)
            decision["generated_token_offset"] = 0
            self._f2a_current_reference_decision = decision

    def _begin_missing_kv_generation(self, *, preserve_current_f2a_step_context=True):
        preserved_f2a_step_context = (
            self._snapshot_f2a_generation_step_context() if preserve_current_f2a_step_context else None
        )
        if not bool(getattr(self, "_missing_kv_evaluation_active", False)):
            self.reset_missing_kv_evaluation_aggregates(aggregation_scope="standalone_generation")
        self._reset_runtime_restoration_generation_state()
        collector = getattr(self, "exact_cache_calibration_collector", None)
        if collector is not None:
            collector.reset_run()
        accounting = self._missing_kv_accounting_obj()
        accounting.set_lifecycle(
            aggregation_scope=(
                "evaluation"
                if bool(getattr(self, "_missing_kv_evaluation_active", False))
                else "standalone_generation"
            ),
            evaluation_aggregation_active=bool(getattr(self, "_missing_kv_evaluation_active", False)),
        )
        accounting.record_generation_local_reset()
        accounting.record_generation_start()
        if self._f2a_enabled():
            accounting.record_f2a_reference_generation()
            self._f2a_reference_generated_token_ids = []
            self._f2a_reference_decision_trace = []
            self._f2a_current_reference_decision = None
            self._f2a_reference_generation_finalized = False
            self._restore_f2a_generation_step_context(preserved_f2a_step_context)

    def _reset_runtime_restoration_counters(self):
        """Compatibility wrapper for older tests/callers that reset both scopes."""
        self._reset_runtime_restoration_generation_state()
        self.reset_missing_kv_evaluation_aggregates(aggregation_scope="standalone_generation")

    def _missing_kv_accounting_obj(self):
        if "missing_kv_accounting" not in self.__dict__:
            self.__dict__["missing_kv_accounting"] = MissingKVRuntimeAccounting(task_c1_exact_overwrite=True)
        return self.__dict__["missing_kv_accounting"]

    def _configure_missing_kv_accounting_policy(self):
        accounting = self._missing_kv_accounting_obj()
        config = getattr(self, "config", None)
        force_restore_all = bool(getattr(config, "kv_runtime_restoration_force_restore_all", False))
        recent_exact_window = 0 if force_restore_all else int(getattr(config, "kv_runtime_restoration_recent_exact_window", 0) or 0)
        restoration_method = getattr(config, "kv_runtime_restoration_method", None)
        restoration_enabled = bool(getattr(config, "kv_runtime_restoration_enabled", False))
        direct_insertion_enabled = bool(getattr(config, "kv_runtime_restoration_direct_insertion_enabled", False))
        batched_insertion_enabled = bool(getattr(config, "kv_runtime_restoration_batched_insertion_enabled", False))
        if restoration_enabled and restoration_method == EXACT_CATCHUP_METHOD:
            policy_mode = "exact_catchup_no_approximation"
            complete_required = False
        elif batched_insertion_enabled and restoration_enabled and restoration_method == PHASE3C_RUNTIME_RESTORATION_METHOD:
            # FREE-aligned lazy batched schedule. complete_required is
            # deliberately False: under lazy execution, early-exit tokens
            # still pending when generation ends are never restored at all
            # because no later token ever needed their deep K/V. Requiring
            # "all source-6 exits x 18" here would wrongly report that
            # correctly-avoided work as missing coverage.
            policy_mode = TASK_C2_BATCHED_INSERTION_POLICY_MODE
            complete_required = False
        elif direct_insertion_enabled and restoration_enabled and restoration_method == PHASE3C_RUNTIME_RESTORATION_METHOD:
            # Task C2 direct insertion never performs the Task C1 exact-
            # catchup-then-overwrite sequence for a successfully direct-
            # inserted token -- exported separately from the CALM Task C1
            # label below even though the underlying config combination
            # (method=phase3c_kv_final, force_restore_all=True,
            # recent_exact_window=0) is otherwise identical. A token that
            # falls back to the existing Task C1 pending-buffer path on
            # direct-insertion failure still uses that path's own
            # unmodified accounting, which complete_required=True continues
            # to validate for completeness.
            policy_mode = TASK_C2_DIRECT_INSERTION_POLICY_MODE
            complete_required = True
        elif restoration_enabled and restoration_method in CALM_TASKC1_RUNTIME_METHODS and force_restore_all and recent_exact_window == 0:
            policy_mode = "taskc1_force_restore_all_exact_catchup_then_overwrite"
            complete_required = True
        elif restoration_enabled:
            policy_mode = "selective_runtime_restoration"
            complete_required = False
        else:
            # Restoration-disabled FREE runs split by the one config flag
            # that defines the State/Exact arm distinction: Synchronized
            # Exact (copy_skipped_hidden_states=False) genuinely replays
            # pending skipped tokens through the missing deep blocks; State
            # Copying (True) generates deep K/V from the COPIED shallow
            # hidden states and performs no exact replay, so its exported
            # identity must never claim the exact-catch-up baseline.
            policy_mode = (
                "free_state_copying_baseline"
                if bool(getattr(config, "copy_skipped_hidden_states", False))
                else "free_exact_catchup_baseline"
            )
            complete_required = False
        accounting.configure_policy(
            restoration_policy_mode=policy_mode,
            force_restore_all=force_restore_all,
            recent_exact_window=recent_exact_window,
            complete_exact_catchup_coverage_required=complete_required,
            task_c1_exact_overwrite=policy_mode != "exact_catchup_no_approximation",
        )
        return accounting

    def _missing_kv_component_timer_obj(self):
        if "missing_kv_component_timer" not in self.__dict__:
            enabled = bool(getattr(getattr(self, "config", None), "kv_runtime_component_timing_enabled", False))
            backend = getattr(getattr(self, "config", None), "kv_runtime_component_timing_backend", "auto")
            self.__dict__["missing_kv_component_timer"] = MissingKVComponentTimer(enabled=enabled, backend=backend)
        return self.__dict__["missing_kv_component_timer"]

    def _missing_kv_exact_catchup_overhead_recorder_obj(self):
        if "missing_kv_exact_catchup_overhead_recorder" not in self.__dict__:
            self.__dict__["missing_kv_exact_catchup_overhead_recorder"] = ExactCatchupOverheadRecorder()
        return self.__dict__["missing_kv_exact_catchup_overhead_recorder"]

    def _exact_catchup_overhead_enabled(self):
        return bool(getattr(getattr(self, "config", None), "kv_exact_catchup_overhead_enabled", False))

    def _calm_taskc1_evidence_prefix(self, restoration_method):
        if restoration_method == PHASE3C_RUNTIME_RESTORATION_METHOD:
            return "phase3c"
        return restoration_method

    def _calm_taskc1_first_token_context_confirmed(self, present_key_value_states, source_layer):
        try:
            source_layer = int(source_layer)
        except (TypeError, ValueError):
            return False
        if source_layer <= 0 or present_key_value_states is None:
            return False
        try:
            if len(present_key_value_states) != source_layer:
                return False
        except TypeError:
            return False
        for layer_idx in range(source_layer):
            try:
                seq_len = safe_cache_seq_len(present_key_value_states[layer_idx])
            except Exception:
                return False
            if seq_len != 1:
                return False
        return True

    def _exact_catchup_overhead_sample_context_fields(self):
        """Independent of missing_kv_provenance_enabled -- an exact-catchup
        overhead event needs stable identity whenever the trainer happened to
        set a sample context, regardless of whether raw dump provenance is
        also being written."""

        context = getattr(self, "_missing_kv_generation_sample_context", None)
        if not context:
            return {"stable_sample_id": None, "selected_order": None, "raw_dataset_index": None}
        return {
            "stable_sample_id": context.get("stable_sample_id"),
            "selected_order": context.get("selected_order"),
            "raw_dataset_index": context.get("raw_dataset_index"),
        }

    def missing_kv_runtime_accounting_summary(self):
        timer = self._missing_kv_component_timer_obj()
        timer.finalize()
        timing = timer.to_summary()
        return self._configure_missing_kv_accounting_policy().to_summary(timing=timing)

    def missing_kv_runtime_accounting_metrics(self, metric_key_prefix="eval"):
        return self._configure_missing_kv_accounting_policy().scalar_metrics(metric_key_prefix=metric_key_prefix)

    def exact_catchup_overhead_events(self):
        return list(self._missing_kv_exact_catchup_overhead_recorder_obj().events)

    def exact_catchup_overhead_summary(self, *, generation_timing_summary=None):
        recorder = self._missing_kv_exact_catchup_overhead_recorder_obj()
        # missing_kv_runtime_accounting_summary() calls the component timer's
        # finalize(), which synchronizes once and resolves every pending
        # per-target CUDA-event timing callback. Only AFTER that may pending
        # live events be converted to their immutable to_dict() snapshot --
        # otherwise a committed candidate transaction could be serialized
        # with target timings that never got the chance to resolve.
        accounting_summary = self.missing_kv_runtime_accounting_summary()
        recorder.resolve_and_serialize_pending()
        return aggregate_exact_catchup_overhead(
            recorder.events,
            accounting_summary=accounting_summary,
            generation_timing_summary=generation_timing_summary,
            duplicate_diagnostics=recorder.duplicate_diagnostics(),
        )

    def _f2a_enabled(self):
        return bool(getattr(self, "is_decoder", False) and getattr(self.config, "kv_f2a_frozen_schedule_enabled", False))

    def _f2a_methods(self):
        raw = getattr(
            self.config,
            "kv_f2a_methods",
            ",".join(F2A_REQUIRED_METHODS),
        )
        methods = tuple(item.strip() for item in str(raw).split(",") if item.strip())
        return methods or F2A_REQUIRED_METHODS

    def _f2a_reference_identity(self):
        payload = {
            "protocol": F2A_EVALUATION_PROTOCOL_NAME,
            "generation_index": self._generation_index,
            "source_layer_mode": getattr(self.config, "kv_f2a_source_layer_mode", None),
            "threshold": float(getattr(self.config, "threshold", CALM_THRESHOLD)),
            "threshold_comparator": "strict_gt",
            "model_num_layers": len(self.block),
        }
        context = self._missing_kv_sample_context_fields()
        for key in ("stable_sample_id", "selected_order", "raw_dataset_index", "dataset_provided_id"):
            if key in context:
                payload[key] = context[key]
        return canonical_json_sha256(payload)

    def _f2a_prefix_token_ids(self):
        token_ids = getattr(self, "_f2a_current_prefix_token_ids", None)
        if token_ids is None:
            raise ValueError("f2a_reference_decoder_prefix_unavailable")
        return [int(item) for item in token_ids]

    def set_f2a_generation_step_context(
        self,
        *,
        prefix_token_ids,
        current_decoder_input_token_id,
        current_decoder_input_position=None,
        predicted_token_position,
    ):
        if not self._f2a_enabled():
            return
        self._f2a_current_prefix_token_ids = [int(item) for item in prefix_token_ids]
        self._f2a_current_decoder_input_token_id = int(current_decoder_input_token_id)
        self._f2a_current_decoder_input_position = int(
            current_decoder_input_position
            if current_decoder_input_position is not None
            else len(self._f2a_current_prefix_token_ids) - 1
        )
        self._f2a_current_predicted_token_position = int(predicted_token_position)
        self._f2a_current_reference_decision = {
            "generated_token_offset": len(getattr(self, "_f2a_reference_generated_token_ids", []) or []),
            "decoder_input_position": self._f2a_current_decoder_input_position,
            "predicted_token_position": self._f2a_current_predicted_token_position,
            "reference_decision_type": "full_depth_reference_forward",
            "reference_decision_layer": None,
            "source_layer_mode": getattr(self.config, "kv_f2a_source_layer_mode", None),
            "first_crossing_source_layer": None,
            "fallback_trigger_candidate_layer": None,
            "full_depth_fallback": False,
            "exact_catchup_flush_occurred": False,
            "pending_token_count": 0,
        }

    def clear_f2a_generation_step_context(self):
        self._f2a_current_prefix_token_ids = None
        self._f2a_current_decoder_input_token_id = None
        self._f2a_current_decoder_input_position = None
        self._f2a_current_predicted_token_position = None

    def _f2a_update_reference_decision(self, **fields):
        if not self._f2a_enabled():
            return
        decision = getattr(self, "_f2a_current_reference_decision", None)
        if not isinstance(decision, dict):
            decision = {
                "generated_token_offset": len(getattr(self, "_f2a_reference_generated_token_ids", []) or []),
                "decoder_input_position": getattr(self, "_f2a_current_decoder_input_position", None),
                "predicted_token_position": getattr(self, "_f2a_current_predicted_token_position", None),
                "source_layer_mode": getattr(self.config, "kv_f2a_source_layer_mode", None),
                "fallback_trigger_candidate_layer": None,
            }
        decision.setdefault("fallback_trigger_candidate_layer", None)
        decision.update(self._json_safe(fields))
        self._f2a_current_reference_decision = decision

    def record_f2a_reference_selected_token(self, token_id):
        if not self._f2a_enabled():
            return
        token_value = int(token_id.item() if hasattr(token_id, "item") else token_id)
        self._f2a_reference_generated_token_ids.append(token_value)
        decision = dict(getattr(self, "_f2a_current_reference_decision", None) or {})
        decision.setdefault("generated_token_offset", len(self._f2a_reference_generated_token_ids) - 1)
        decision["generated_token_id"] = token_value
        self._f2a_reference_decision_trace.append(self._json_safe(decision))

    def _f2a_write_reference_trajectory_summary(self):
        rows = list(getattr(self, "_f2a_reference_trajectory_rows", []) or [])
        path = getattr(self.config, "kv_f2a_reference_trajectory_summary_output", None)
        if not path:
            return
        binding_rows = []
        identity_binding_errors = []
        generation_binding_path = getattr(self.config, "missing_kv_generation_binding_output", None)
        if generation_binding_path:
            try:
                if os.path.exists(generation_binding_path):
                    binding_rows = read_provenance_jsonl(generation_binding_path)
            except Exception as exc:
                binding_rows = []
                identity_binding_errors.append("generation_binding_rows_unreadable:{}".format(exc))
        validation = validate_reference_generation_trajectory_rows(
            rows,
            generation_binding_rows=binding_rows,
            source_layer_mode=getattr(self.config, "kv_f2a_source_layer_mode", None),
            expected_decoder_layer_count=len(getattr(self, "block", []) or []),
        )
        summary = {
            "schema_version": 1,
            "record_type": "missing_kv_f2a_reference_generation_trajectory_summary",
            "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
            "status": validation.get("status"),
            "errors": validation.get("errors", []),
            "expected_generation_count": len(rows),
            "written_generation_count": len(rows),
            "complete_population_recording": validation.get("status") == "ok",
            "generation_trajectory_population_sha256": reference_generation_trajectory_population_sha256(rows),
            "ordered_generation_trajectory_sha256": ordered_reference_generation_trajectory_sha256(rows),
            "total_generated_token_count": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
            "generation_binding_sha256": None,
            "dataset_population_sha256": None,
            "decoding_configuration_sha256": getattr(self.config, "kv_f2a_decoding_configuration_sha256", None),
            "identity_binding_status": "ok",
            "identity_binding_errors": [],
        }
        binding_summary_path = getattr(self.config, "missing_kv_generation_binding_summary_output", None)
        if binding_summary_path:
            try:
                if os.path.exists(binding_summary_path):
                    with open(binding_summary_path, "r", encoding="utf-8") as handle:
                        binding_summary = json.load(handle)
                    summary["generation_binding_sha256"] = binding_summary.get("generation_sample_binding_sha256")
            except Exception as exc:
                identity_binding_errors.append("generation_binding_summary_unreadable:{}".format(exc))
        population_summary_path = getattr(self.config, "missing_kv_effective_population_summary_output", None)
        if population_summary_path:
            try:
                if os.path.exists(population_summary_path):
                    with open(population_summary_path, "r", encoding="utf-8") as handle:
                        population_summary = json.load(handle)
                    summary["dataset_population_sha256"] = population_summary.get("dataset_population_sha256")
            except Exception as exc:
                identity_binding_errors.append("effective_population_summary_unreadable:{}".format(exc))
        if identity_binding_errors:
            summary["identity_binding_status"] = "unavailable"
            summary["identity_binding_errors"] = identity_binding_errors
            summary["complete_population_recording"] = False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _f2a_current_input_token_id(self):
        token_id = getattr(self, "_f2a_current_decoder_input_token_id", None)
        if token_id is None:
            raise ValueError("f2a_current_decoder_input_token_unavailable")
        return int(token_id)

    def _f2a_current_input_position(self):
        position = getattr(self, "_f2a_current_decoder_input_position", None)
        if position is None:
            token_ids = getattr(self, "_f2a_current_prefix_token_ids", None)
            if token_ids is None:
                raise ValueError("f2a_current_decoder_input_position_unavailable")
            return len(token_ids) - 1
        return int(position)

    def _f2a_predicted_token_position(self, default_position):
        position = getattr(self, "_f2a_current_predicted_token_position", None)
        if position is None:
            if default_position is None:
                raise ValueError("f2a_predicted_token_position_unavailable")
            return int(default_position)
        return int(position)

    def _f2a_append_output_row(self, path, row):
        if not path:
            return
        append_provenance_jsonl(path, row)

    def _f2a_record_event_and_rows(self, event, records):
        self._f2a_schedule_rows.append(dict(event))
        self._f2a_records.extend(dict(record) for record in records)
        self._missing_kv_accounting_obj().record_f2a_frozen_event()
        self._f2a_append_output_row(getattr(self.config, "kv_f2a_schedule_output", None), event)
        for record in records:
            self._missing_kv_accounting_obj().record_f2a_candidate_replay(record.get("method"))
            self._f2a_append_output_row(getattr(self.config, "kv_f2a_records_output", None), record)

    def f2a_summary(self):
        schedule_rows = list(getattr(self, "_f2a_schedule_rows", []))
        record_rows = list(getattr(self, "_f2a_records", []))
        accounting_summary = self.missing_kv_runtime_accounting_summary()
        source_mode = getattr(self.config, "kv_f2a_source_layer_mode", None)
        method_names = list(self._f2a_methods())
        skipped_reason_counts = dict(accounting_summary.get("f2a_skipped_reason_counts", {}) or {})
        cap_skipped = int(accounting_summary.get("f2a_cap_excluded_event_count", 0) or 0)
        blocking_skipped = sum(
            int(count or 0)
            for reason, count in skipped_reason_counts.items()
            if str(reason) not in ("terminal_no_followup", "max_events_reached")
        )
        terminal_no_followup = int(accounting_summary.get("f2a_calm_terminal_no_followup_count", 0) or 0)
        failed_events = int(accounting_summary.get("f2a_failed_event_count", 0) or 0)
        finalized_events = int(accounting_summary.get("f2a_frozen_restoration_event_count", len(schedule_rows)) or 0)
        blocking_failure_events = failed_events
        if source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
            produced_events = int(accounting_summary.get("f2a_calm_event_produced_count", finalized_events) or 0)
            considered_events = int(
                accounting_summary.get(
                    "f2a_calm_event_considered_count",
                    finalized_events + terminal_no_followup + cap_skipped + blocking_failure_events,
                )
                or 0
            )
        else:
            produced_events = finalized_events
            considered_events = finalized_events + cap_skipped + blocking_failure_events
        reference_generation_count = int(accounting_summary.get("f2a_reference_generation_count", 0) or 0)
        generic_generation_count = int(accounting_summary.get("generation_count", 0) or 0)
        generic_generated_token_count = int((accounting_summary.get("counters") or {}).get("generated_token_count", 0) or 0)
        reference_generated_token_count = int(accounting_summary.get("f2a_reference_generated_token_count", 0) or 0)
        schedule_summary = {
            "expected_record_count": finalized_events,
            "written_record_count": len(schedule_rows),
            "skipped_record_count": blocking_skipped,
            "complete_population_recording": blocking_skipped == 0,
            "schedule_semantic_sha256": schedule_semantic_sha256(schedule_rows) if schedule_rows else None,
        }
        record_summary = {
            "expected_record_count": finalized_events * len(method_names),
            "written_record_count": len(record_rows),
            "skipped_record_count": blocking_skipped,
            "complete_population_recording": blocking_skipped == 0,
        }
        decoding_sha = getattr(self.config, "kv_f2a_decoding_configuration_sha256", None)
        f2a_generation_binding_rows = []
        f2a_identity_binding_errors = []
        generation_binding_path = getattr(self.config, "missing_kv_generation_binding_output", None)
        if generation_binding_path:
            try:
                if os.path.exists(generation_binding_path):
                    f2a_generation_binding_rows = read_provenance_jsonl(generation_binding_path)
            except Exception as exc:
                f2a_identity_binding_errors.append(
                    "generation_binding_read_failed:{}".format(exc)
                )
                f2a_generation_binding_rows = []
        trajectory_rows = list(getattr(self, "_f2a_reference_trajectory_rows", []) or [])
        trajectory_validation = validate_reference_generation_trajectory_rows(
            trajectory_rows,
            generation_binding_rows=f2a_generation_binding_rows,
            expected_generation_count=reference_generation_count,
            expected_generated_token_count=reference_generated_token_count,
            source_layer_mode=source_mode,
            expected_decoder_layer_count=len(getattr(self, "block", []) or []),
        )
        trajectory_complete = trajectory_validation.get("status") == "ok"
        trajectory_summary = {
            "schema_version": 1,
            "record_type": "missing_kv_f2a_reference_generation_trajectory_summary",
            "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
            "status": trajectory_validation.get("status"),
            "errors": trajectory_validation.get("errors", []),
            "expected_generation_count": len(trajectory_rows),
            "written_generation_count": len(trajectory_rows),
            "complete_population_recording": bool(trajectory_complete),
            "identity_binding_status": "ok" if not f2a_identity_binding_errors else "unavailable",
            "identity_binding_errors": list(f2a_identity_binding_errors),
            "generation_trajectory_population_sha256": reference_generation_trajectory_population_sha256(trajectory_rows),
            "ordered_generation_trajectory_sha256": ordered_reference_generation_trajectory_sha256(trajectory_rows),
            "total_generated_token_count": sum(int(row.get("generated_token_count", 0) or 0) for row in trajectory_rows),
            "decoding_configuration_sha256": decoding_sha,
        }
        trajectory_sha = (
            reference_trajectory_sha256(
                schedule_rows=schedule_rows,
                reference_generation_rows=trajectory_rows,
                generation_binding_rows=f2a_generation_binding_rows,
                source_layer_mode=source_mode,
                decoding_configuration_sha256=decoding_sha,
                reference_generation_count=reference_generation_count,
                reference_generated_token_count=reference_generated_token_count,
                terminal_exclusion_accounting={
                    "terminal_no_followup_event_count": terminal_no_followup,
                    "cap_excluded_event_count": cap_skipped,
                    "failed_event_count": failed_events,
                },
            )
            if trajectory_rows and decoding_sha not in (None, "")
            else None
        )
        schedule_validation = validate_f2a_schedule_rows(schedule_rows, summary=schedule_summary)
        record_validation = validate_f2a_record_rows(record_rows, schedule_rows=schedule_rows, summary=record_summary)
        status = "ok" if schedule_validation.get("status") == "ok" and record_validation.get("status") == "ok" else "failed"
        def _output_sha(path):
            if not path:
                return None
            try:
                return sha256_file(path) if os.path.exists(path) else None
            except Exception:
                return None
        return {
            "schema_version": 1,
            "producer_protocol": F2A_EVALUATION_PROTOCOL_NAME,
            "status": status,
            "runtime_started": True,
            "runtime_completed": True,
            "paper_claim_valid": False,
            "real_population_evaluated": False,
            "source_layer_mode": source_mode,
            "configured_methods": method_names,
            "method_names": method_names,
            "max_events": int(getattr(self.config, "kv_f2a_max_events", 0) or 0),
            "event_cap_enabled": int(getattr(self.config, "kv_f2a_max_events", 0) or 0) > 0,
            "generic_generation_count": generic_generation_count,
            "generic_generated_token_count": generic_generated_token_count,
            "reference_generation_count": reference_generation_count,
            "reference_generated_token_count": reference_generated_token_count,
            "reference_trajectory_sha256": trajectory_sha,
            "decoding_configuration_sha256": decoding_sha,
            "considered_event_count": considered_events,
            "produced_event_count": produced_events,
            "finalized_event_count": finalized_events,
            "terminal_no_followup_count": terminal_no_followup,
            "cap_skipped_event_count": cap_skipped,
            "cap_skipped_pending_token_count": int(accounting_summary.get("f2a_cap_excluded_pending_token_count", 0) or 0),
            "failed_event_count": failed_events,
            "written_schedule_event_count": len(schedule_rows),
            "written_component_record_count": len(record_rows),
            "schedule_file_sha256": _output_sha(getattr(self.config, "kv_f2a_schedule_output", None)),
            "records_file_sha256": _output_sha(getattr(self.config, "kv_f2a_records_output", None)),
            "reference_trajectory_file_sha256": _output_sha(getattr(self.config, "kv_f2a_reference_trajectory_output", None)),
            "reference_trajectory_summary_file_sha256": _output_sha(getattr(self.config, "kv_f2a_reference_trajectory_summary_output", None)),
            "schedule_semantic_sha256": schedule_summary.get("schedule_semantic_sha256"),
            "reference_generation_trajectory_summary": trajectory_summary,
            "schedule_validation_failure_count": int(accounting_summary.get("f2a_schedule_validation_failure_count", 0) or 0),
            "variant_replay_failure_count": int(accounting_summary.get("f2a_variant_replay_failure_count", 0) or 0),
            "nonfinite_metric_failure_count": int(accounting_summary.get("f2a_nonfinite_metric_failure_count", 0) or 0),
            "exact_shadow_parity_failure_count": int(accounting_summary.get("f2a_exact_shadow_parity_failure_count", 0) or 0),
            "reference_state_mutation_failure_count": int(accounting_summary.get("f2a_reference_state_mutation_failure_count", 0) or 0),
            "full_depth_fallback_count": int(accounting_summary.get("f2a_full_depth_fallback_count", 0) or 0),
            "candidate_replay_counts_by_method": dict(accounting_summary.get("f2a_candidate_replay_counts_by_method", {}) or {}),
            "schedule_event_count": len(schedule_rows),
            "record_count": len(record_rows),
            "skipped_events": list(getattr(self, "_f2a_skipped_events", [])),
            "schedule_summary": schedule_summary,
            "record_summary": record_summary,
            "schedule_validation": schedule_validation,
            "record_validation": record_validation,
            "accounting": accounting_summary,
        }

    def _f2a_target_self_attention_modules(self, target_layer):
        target_self_attention = self.block[target_layer].layer[0]
        return (
            target_self_attention.layer_norm,
            target_self_attention.SelfAttention.k,
            target_self_attention.SelfAttention.v,
        )

    def _f2a_logits_from_hidden(self, hidden_states, lm_head):
        if lm_head is None:
            lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            raise ValueError("f2a_lm_head_missing")
        hidden = hidden_states[:, [-1], :]
        sequence_output = self.final_layer_norm(hidden)
        sequence_output = self.dropout(sequence_output)
        if getattr(self.config, "tie_word_embeddings", False):
            sequence_output = sequence_output * (self.config.d_model ** -0.5)
        return lm_head(sequence_output)

    def _maybe_record_f2a_debug_logits(self, *, reference_logits, shadow_logits, diagnostics):
        """Opt-in, capped raw-logit diagnostic record for exact-shadow parity
        failures, using the existing KV_F2A_WRITE_DEBUG_LOGITS /
        KV_F2A_MAX_DEBUG_LOGIT_RECORDS contract. Normal runs never produce
        this record; ordinary parity-failure diagnostics (shapes, scalar
        stats) are recorded separately regardless of this flag."""

        if not bool(getattr(self.config, "kv_f2a_write_debug_logits", False)):
            return
        max_records = int(getattr(self.config, "kv_f2a_max_debug_logit_records", 0) or 0)
        if max_records <= 0 or self._f2a_debug_logit_records_written >= max_records:
            return
        self.kv_trace.record(
            "f2a_exact_shadow_parity_debug_logits",
            reference_logits=reference_logits.detach().cpu(),
            shadow_logits=shadow_logits.detach().cpu(),
            **diagnostics,
        )
        self._f2a_debug_logit_records_written += 1

    def _run_f2a_query_with_cache(
        self,
        *,
        source_layer,
        query_hidden,
        candidate_past_key_values,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
        initial_position_bias=None,
    ):
        """Replay a single-query fixed-layer forward starting at ``source_layer``.

        ``initial_position_bias``, when provided, seeds the FIRST layer's
        self-attention position bias (subsequently reused unchanged through
        deeper layers, matching reference T5 decoder behavior). This must be
        the exact reference-equivalent position-bias slice: ``source_layer``
        does not own the learned relative-attention-bias table (only decoder
        block 0 does), so leaving this ``None`` would silently synthesize a
        zero position bias inside ``DeployT5Attention.forward`` instead of
        reusing the reference's learned bias.
        """
        hidden = query_hidden.detach().clone()
        position_bias = initial_position_bias
        cross_position_bias = encoder_decoder_position_bias
        for target_layer in range(int(source_layer), len(self.block)):
            layer_outputs = self.block[target_layer](
                hidden,
                attention_mask=None,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=cross_position_bias,
                layer_head_mask=head_mask[target_layer] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[target_layer] if cross_attn_head_mask is not None else None,
                past_key_value=candidate_past_key_values[target_layer],
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=False,
                parallel_mask=False,
                layer_idx=target_layer,
                kv_importance_tracker=None,
            )
            hidden = layer_outputs[0]
            position_bias = layer_outputs[2] if len(layer_outputs) > 2 else None
            if self.is_decoder and encoder_hidden_states is not None:
                cross_position_bias = layer_outputs[4 if output_attentions else 3]
        return hidden

    def _f2a_candidate_cache_from_exact_states(
        self,
        *,
        source_layer,
        exact_states_by_layer,
        past_key_values,
        drop_current_token=True,
    ):
        candidate_cache = []
        source_layer = int(source_layer)
        for layer_idx in range(len(self.block)):
            if layer_idx < source_layer:
                if past_key_values is not None and past_key_values[layer_idx] is not None:
                    candidate_cache.append(
                        tuple(item.detach().clone() if isinstance(item, torch.Tensor) else item for item in past_key_values[layer_idx])
                    )
                else:
                    candidate_cache.append(None)
                continue
            exact_state = exact_states_by_layer.get(layer_idx)
            if exact_state is None or len(exact_state) < 2:
                raise ValueError("f2a_exact_reference_state_missing:{}".format(layer_idx))
            state = []
            for state_index, item in enumerate(exact_state):
                if isinstance(item, torch.Tensor) and state_index in (0, 1) and drop_current_token:
                    if int(item.shape[2]) < 1:
                        raise ValueError("f2a_exact_reference_state_empty:{}".format(layer_idx))
                    state.append(item[:, :, :-1, :].detach().clone())
                elif isinstance(item, torch.Tensor):
                    state.append(item.detach().clone())
                else:
                    state.append(item)
            candidate_cache.append(tuple(state))
        return candidate_cache

    def _maybe_run_f2a_fixed_shadow_replay(
        self,
        *,
        source_layer,
        query_hidden_at_source,
        pending_metadata,
        pending_phase3c_source_hidden_records,
        exact_states_by_layer,
        reference_hidden_states,
        reference_present_key_value_states,
        past_key_values,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
        lm_head,
        reference_position_bias,
    ):
        if not self._f2a_enabled():
            return
        if getattr(self.config, "kv_f2a_source_layer_mode", SOURCE_LAYER_MODE_FIXED) != SOURCE_LAYER_MODE_FIXED:
            return
        if self.kv_f2a_artifact is None:
            raise ValueError("f2a_policy_artifact_not_loaded")
        if lm_head is None or not use_cache:
            self._missing_kv_accounting_obj().record_f2a_skipped("missing_lm_head_or_cache")
            return
        pending_count = len(pending_metadata or ())
        if pending_count == 0:
            return
        if len(pending_phase3c_source_hidden_records or ()) != pending_count:
            self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event()
            raise ValueError("f2a_phase3c_source_hidden_records_missing")
        max_events = int(getattr(self.config, "kv_f2a_max_events", 0) or 0)
        if max_events > 0 and len(self._f2a_schedule_rows) >= max_events:
            self._missing_kv_accounting_obj().record_f2a_skipped("max_events_reached", 1)
            self._missing_kv_accounting_obj().record_f2a_cap_excluded_event(
                count=1,
                pending_token_count=pending_count,
            )
            return
        reference_snapshot = cache_noninterference_snapshot(
            reference_present_key_value_states,
            {
                "target_layers": list(range(int(source_layer), len(self.block))),
                "cache_positions": [],
            },
        )
        reference_logits = self._f2a_logits_from_hidden(reference_hidden_states[:, [-1], :], lm_head)
        context = self._missing_kv_sample_context_fields()
        stable_sample_id = context.get("stable_sample_id")
        if stable_sample_id in (None, ""):
            self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event()
            raise ValueError("f2a_stable_sample_id_unavailable")
        reference_identity = self._f2a_reference_identity()
        source_hiddens = []
        pending_positions = []
        restore_relative_indices = []
        past_len = safe_cache_seq_len(past_key_values[int(source_layer)]) if past_key_values is not None else 0
        past_len = 0 if past_len is None else int(past_len)
        # Fixed exact-shadow/candidate replay starts at a mid-network layer
        # that does not own the learned relative-attention-bias table (only
        # decoder block 0 does). Without the exact reference-equivalent
        # position bias, DeployT5Attention.forward would silently synthesize
        # a zero position bias instead -- fail closed rather than allow that.
        expected_key_length = past_len + pending_count + 1
        try:
            validated_reference_position_bias = validate_f2a_reference_position_bias(
                reference_position_bias,
                expected_key_length=expected_key_length,
                expected_num_heads=getattr(self.config, "num_heads", None),
                expected_dtype=reference_hidden_states.dtype,
                expected_device=reference_hidden_states.device,
            )
        except Exception as exc:
            self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event()
            raise ValueError(str(exc)) from exc
        cache_positions = []
        for relative_index, source_record in enumerate(pending_phase3c_source_hidden_records):
            source_hidden = source_record.get("raw_hidden")
            if source_hidden is None:
                self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
                self._missing_kv_accounting_obj().record_f2a_failed_event()
                raise ValueError("f2a_source_hidden_missing")
            decoder_position = source_record.get("decoder_position")
            if decoder_position is None:
                if relative_index < len(pending_metadata or ()):
                    decoder_position = (pending_metadata[relative_index] or {}).get("decoder_position")
            if decoder_position is None:
                decoder_position = past_len + relative_index
            pending_positions.append(int(decoder_position))
            restore_relative_indices.append(int(relative_index))
            cache_positions.append(past_len + int(relative_index))
            source_hiddens.append(source_hidden)
        source_hidden_block = torch.cat(source_hiddens, dim=1)
        event = build_f2a_event(
            stable_sample_id=stable_sample_id,
            generation_index=self._generation_index,
            decoder_position=int(infer_decoder_position(past_key_values)),
            predicted_token_position=self._f2a_predicted_token_position(infer_decoder_position(past_key_values)),
            current_decoder_input_position=self._f2a_current_input_position(),
            current_decoder_input_token_id=self._f2a_current_input_token_id(),
            prefix_token_ids=self._f2a_prefix_token_ids(),
            source_layer_mode=SOURCE_LAYER_MODE_FIXED,
            source_layer=int(source_layer),
            decoder_layer_count=len(self.block),
            pending_token_positions=pending_positions,
            restore_relative_indices=restore_relative_indices,
            cache_positions=cache_positions,
            reference_schedule_identity=reference_identity,
            reference_run_identity=reference_identity,
            artifact_file_sha256=self.kv_f2a_artifact_sha256,
            policy_sha256=self.kv_f2a_policy_sha256,
            threshold=float(getattr(self.config, "threshold", CALM_THRESHOLD)),
            threshold_comparator="strict_gt",
        )
        exact_candidate_cache = self._f2a_candidate_cache_from_exact_states(
            source_layer=int(source_layer),
            exact_states_by_layer=exact_states_by_layer,
            past_key_values=past_key_values,
        )
        shadow_hidden = self._run_f2a_query_with_cache(
            source_layer=int(source_layer),
            query_hidden=query_hidden_at_source,
            candidate_past_key_values=exact_candidate_cache,
            encoder_hidden_states=encoder_hidden_states,
            encoder_extended_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            initial_position_bias=validated_reference_position_bias.clone(),
        )
        shadow_logits = self._f2a_logits_from_hidden(shadow_hidden, lm_head)
        parity = validate_exact_shadow_parity(
            f2a_logit_metrics(reference_logits, shadow_logits),
            dtype=reference_logits.dtype,
        )
        if parity.get("status") != "ok":
            diagnostics = build_f2a_exact_shadow_parity_diagnostics(
                parity=parity,
                reference_logits=reference_logits,
                shadow_logits=shadow_logits,
                reference_position_bias=validated_reference_position_bias,
                reference_position_bias_source="reference_exact_catchup_last_query_slice",
                shadow_cache_sequence_length=past_len + pending_count,
                current_query_length=1,
                expected_key_length=expected_key_length,
            )
            self.kv_trace.record("f2a_exact_shadow_parity_failure_diagnostics", **diagnostics)
            self._maybe_record_f2a_debug_logits(
                reference_logits=reference_logits,
                shadow_logits=shadow_logits,
                diagnostics=diagnostics,
            )
            self._missing_kv_accounting_obj().record_f2a_exact_shadow_parity_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event(
                event_uid=event.get("frozen_event_uid")
            )
            raise ValueError("f2a_exact_shadow_parity_failed:{}".format(parity.get("errors")))
        records = []
        for method in self._f2a_methods():
            try:
                restored_by_layer = {}
                for target_layer in range(int(source_layer), len(self.block)):
                    exact_state = exact_states_by_layer.get(target_layer)
                    if exact_state is None or len(exact_state) < 2:
                        raise ValueError("f2a_exact_reference_state_missing:{}".format(target_layer))
                    layer_norm, key_projection, value_projection = self._f2a_target_self_attention_modules(target_layer)
                    restored_key, restored_value, _metadata = restore_f2a_method_block_from_hidden(
                        method=method,
                        artifact=self.kv_f2a_artifact,
                        source_hidden=source_hidden_block,
                        source_layer=int(source_layer),
                        target_layer=target_layer,
                        target_layer_norm=layer_norm,
                        key_projection=key_projection,
                        value_projection=value_projection,
                        threshold=float(getattr(self.config, "threshold", CALM_THRESHOLD)),
                        output_device=exact_state[0].device,
                        output_dtype=exact_state[0].dtype,
                    )
                    restored_by_layer[target_layer] = (restored_key, restored_value)
                candidate_cache = patch_candidate_cache_for_full_event(
                    exact_candidate_cache,
                    event=event,
                    restored_by_target_layer=restored_by_layer,
                )
                candidate_hidden = self._run_f2a_query_with_cache(
                    source_layer=int(source_layer),
                    query_hidden=query_hidden_at_source,
                    candidate_past_key_values=candidate_cache,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_extended_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    head_mask=head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    initial_position_bias=validated_reference_position_bias.clone(),
                )
                candidate_logits = self._f2a_logits_from_hidden(candidate_hidden, lm_head)
                metrics = f2a_logit_metrics(reference_logits, candidate_logits)
                records.append(
                    build_f2a_component_record(
                        event,
                        method=method,
                        metrics=metrics,
                        status="ok",
                        artifact_file_sha256=self.kv_f2a_artifact_sha256,
                        policy_sha256=self.kv_f2a_policy_sha256,
                        replay_diagnostics={
                            "exact_shadow_parity": parity,
                            "candidate_output_used_for_reference_continuation": False,
                            "candidate_cache_aliases_reference": False,
                            "pending_token_count": pending_count,
                        },
                    )
                )
            except Exception as exc:
                self._missing_kv_accounting_obj().record_f2a_variant_replay_failure()
                self._missing_kv_accounting_obj().record_f2a_failed_event(
                    event_uid=event.get("frozen_event_uid")
                )
                raise ValueError("f2a_candidate_replay_failed:{}:{}".format(method, exc)) from exc
        noninterference = validate_cache_noninterference(
            reference_snapshot,
            reference_present_key_value_states,
            event=event,
        )
        if noninterference.get("status") != "ok":
            self._missing_kv_accounting_obj().record_f2a_reference_state_mutation_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event(
                event_uid=event.get("frozen_event_uid")
            )
            raise ValueError("f2a_reference_cache_mutated:{}".format(noninterference.get("errors")))
        for record in records:
            record.setdefault("replay_diagnostics", {})["reference_cache_noninterference"] = noninterference
        self._f2a_record_event_and_rows(event, records)

    def _runtime_restoration_enabled(self):
        return self.kv_runtime_restorer is not None and bool(
            getattr(self.config, "kv_runtime_restoration_enabled", False)
        )

    def _phase3c_runtime_restoration_enabled(self):
        return (
            self._runtime_restoration_enabled()
            and getattr(self.kv_runtime_restorer, "method", None) == PHASE3C_RUNTIME_RESTORATION_METHOD
        )

    def _calm_runtime_restoration_enabled(self):
        return (
            self.is_decoder
            and self.use_early_exit
            and not self.use_shallow_deep
            and bool(getattr(self.config, "kv_runtime_restoration_enabled", False))
            and bool(getattr(self.config, "kv_runtime_restoration_calm_enabled", False))
            and self.kv_runtime_restorer is not None
        )

    def _is_calm_phase3c_taskc1_enabled(self):
        return (
            self._is_calm_taskc1_runtime_enabled()
            and getattr(self.kv_runtime_restorer, "method", None) == PHASE3C_RUNTIME_RESTORATION_METHOD
        )

    def _is_calm_taskc1_runtime_enabled(self):
        return (
            self.is_decoder
            and self.use_early_exit
            and not self.use_shallow_deep
            and bool(getattr(self.config, "kv_runtime_restoration_enabled", False))
            and bool(getattr(self.config, "kv_runtime_restoration_calm_enabled", False))
            and self.kv_runtime_restorer is not None
            and getattr(self.kv_runtime_restorer, "method", None) in CALM_TASKC1_RUNTIME_METHODS
            and getattr(self.kv_runtime_restorer, "runtime_source_mode", None)
            in (SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING, SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM)
        )

    def _is_f2a_calm_enabled(self):
        return (
            self._f2a_enabled()
            and self.is_decoder
            and self.use_early_exit
            and not self.use_shallow_deep
            and getattr(self.config, "kv_f2a_source_layer_mode", None) == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
        )

    def _calm_phase3c_candidate_layers(self):
        layers = getattr(self.kv_runtime_restorer, "candidate_source_layers", None) if self.kv_runtime_restorer is not None else None
        if layers:
            return tuple(int(item) for item in layers)
        return tuple(int(item) for item in CALM_CANDIDATE_LAYERS)

    def _call_calm_phase3c_transaction_test_hook(self, stage, event=None, **payload):
        hook = getattr(self, "_calm_phase3c_transaction_test_hook", None)
        if callable(hook):
            hook(stage, event, payload)

    def _calm_context_trace_fields(self, context):
        if not isinstance(context, dict):
            return {"enabled": False, "reason": "context_missing"}
        return {
            key: value
            for key, value in context.items()
            if key not in {"source_key", "source_value"}
        }

    def _build_calm_runtime_restoration_context(
        self,
        exit_start_layer,
        present_key_value_states,
        past_key_values,
        decoder_position,
        confidence=None,
    ):
        exit_start_layer = int(exit_start_layer)
        source_layer = exit_start_layer - 1
        context = {
            "enabled": False,
            "reason": "disabled",
            "exit_start_layer": exit_start_layer,
            "source_layer": source_layer,
            "target_layers": list(range(exit_start_layer, len(self.block))),
            "decoder_position": decoder_position,
            "confidence": confidence,
            "source_key": None,
            "source_value": None,
            "source_key_shape": None,
            "source_value_shape": None,
            "source_slice_start": None,
            "source_slice_end": None,
            "source_slice_mode": None,
            "source_past_len": None,
            "error_message": None,
        }
        if not self._calm_runtime_restoration_enabled():
            return context
        self._kv_runtime_restoration_counters["calm_early_exit_tokens"] += 1
        try:
            if source_layer < 0:
                raise ValueError("source_layer_unavailable")
            if present_key_value_states is None or len(present_key_value_states) <= source_layer:
                raise ValueError("source_present_key_value_state_unavailable")
            source_state = present_key_value_states[source_layer]
            if source_state is None or len(source_state) < 2:
                raise ValueError("source_state_missing_self_attention_kv")
            source_key_tensor = source_state[0]
            source_value_tensor = source_state[1]
            if source_key_tensor is None or source_value_tensor is None:
                raise ValueError("source_self_attention_kv_is_none")
            source_past = past_key_values[source_layer] if past_key_values is not None and len(past_key_values) > source_layer else None
            start, end, slice_mode, past_len = self._source_slice_for_skip_dump(source_key_tensor, source_past)
            if start is None:
                raise ValueError("could_not_infer_source_token_slice")
            source_key = source_key_tensor[:, :, start:end, :].detach().clone()
            source_value = source_value_tensor[:, :, start:end, :].detach().clone()
            context.update(
                {
                    "enabled": True,
                    "reason": "enabled",
                    "source_key": source_key,
                    "source_value": source_value,
                    "source_key_shape": list(source_key.shape),
                    "source_value_shape": list(source_value.shape),
                    "source_slice_start": int(start),
                    "source_slice_end": int(end),
                    "source_slice_mode": slice_mode,
                    "source_past_len": past_len,
                }
            )
        except Exception as exc:
            context["reason"] = "source_unavailable"
            context["error_message"] = str(exc)
        return context

    def _apply_calm_runtime_restoration_to_present_kv(
        self,
        layer_idx,
        present_key_value_state,
        past_key_value,
        calm_context,
    ):
        context = calm_context if isinstance(calm_context, dict) else {}
        event = {
            "enabled": bool(context.get("enabled")),
            "layer_idx": int(layer_idx),
            "target_layer": int(layer_idx),
            "source_layer": context.get("source_layer"),
            "exit_start_layer": context.get("exit_start_layer"),
            "decoder_position": context.get("decoder_position"),
            "restoration_method": getattr(self.config, "kv_runtime_restoration_method", None),
            "past_key_value_len": None,
            "cache_position": None,
            "cache_position_source": None,
            "present_key_shape_before": None,
            "present_value_shape_before": None,
            "source_key_shape": context.get("source_key_shape"),
            "source_value_shape": context.get("source_value_shape"),
            "restored_key_shape": None,
            "restored_value_shape": None,
            "actual_restoration_applied": False,
            "fallback_state_copy": True,
            "skip_reason": None,
            "error_message": None,
            "missing_map_count": 0,
            "cache_shape_mismatch_count": 0,
            "nan_or_inf_count": 0,
        }
        def _finish(state):
            event["counters"] = dict(self._kv_runtime_restoration_counters)
            return state, event

        if not context.get("enabled"):
            event["skip_reason"] = context.get("reason", "disabled")
            event["error_message"] = context.get("error_message")
            self._kv_runtime_restoration_counters["calm_target_token_layer_units"] += 1
            self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
            return _finish(present_key_value_state)
        self._kv_runtime_restoration_counters["calm_target_token_layer_units"] += 1
        if present_key_value_state is None or len(present_key_value_state) < 2:
            event["skip_reason"] = "present_key_value_state_unavailable"
            self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
            return _finish(present_key_value_state)
        key_tensor = present_key_value_state[0]
        value_tensor = present_key_value_state[1]
        if key_tensor is None or value_tensor is None:
            event["skip_reason"] = "self_attention_kv_unavailable"
            self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
            return _finish(present_key_value_state)
        event["present_key_shape_before"] = list(key_tensor.shape)
        event["present_value_shape_before"] = list(value_tensor.shape)
        if key_tensor.shape[0] != 1:
            self._kv_runtime_restoration_counters["calm_unsupported_batch_count"] += 1
            raise NotImplementedError("CALM runtime restored-K/V currently supports batch size 1 only")

        past_len = safe_cache_seq_len(past_key_value)
        event["past_key_value_len"] = past_len
        if past_len is not None and int(key_tensor.shape[2]) >= int(past_len) + 1:
            cache_position = int(past_len)
            event["cache_position_source"] = "past_key_value_len"
        elif int(key_tensor.shape[2]) >= 1:
            cache_position = int(key_tensor.shape[2]) - 1
            event["cache_position_source"] = "last_token"
        else:
            event["skip_reason"] = "could_not_infer_target_cache_position"
            self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
            return _finish(present_key_value_state)
        event["cache_position"] = cache_position

        source_layer = context.get("source_layer")
        source_key = context.get("source_key")
        source_value = context.get("source_value")
        try:
            if source_layer is None:
                raise ValueError("source_layer_missing")
            source_layer = int(source_layer)
            if int(layer_idx) <= source_layer:
                event["skip_reason"] = "target_not_deeper_than_source"
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            if not self.kv_runtime_restorer.has_pair(source_layer, int(layer_idx)):
                event["skip_reason"] = "missing_map"
                event["missing_map_count"] = 1
                self._kv_runtime_restoration_counters["calm_missing_map_count"] += 1
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            if source_key is None or source_value is None:
                raise ValueError("source_kv_missing_from_context")
            source_key = source_key.to(device=key_tensor.device, dtype=key_tensor.dtype)
            source_value = source_value.to(device=value_tensor.device, dtype=value_tensor.dtype)
            result = self.kv_runtime_restorer.restore(source_key, source_value, source_layer, int(layer_idx))
            if result.status == "missing_map":
                event["skip_reason"] = "missing_map"
                event["error_message"] = result.error_message
                event["missing_map_count"] = 1
                self._kv_runtime_restoration_counters["calm_missing_map_count"] += 1
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            if result.status != "ok":
                event["skip_reason"] = result.status
                event["error_message"] = result.error_message
                if result.error_message and ("NaN" in result.error_message or "Inf" in result.error_message):
                    event["nan_or_inf_count"] = 1
                    self._kv_runtime_restoration_counters["calm_nan_or_inf_count"] += 1
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            restored_key = result.restored_key.to(device=key_tensor.device, dtype=key_tensor.dtype)
            restored_value = result.restored_value.to(device=value_tensor.device, dtype=value_tensor.dtype)
            event["restored_key_shape"] = list(restored_key.shape)
            event["restored_value_shape"] = list(restored_value.shape)
            target_key_slice = key_tensor[:, :, cache_position : cache_position + 1, :]
            target_value_slice = value_tensor[:, :, cache_position : cache_position + 1, :]
            if restored_key.shape != target_key_slice.shape or restored_value.shape != target_value_slice.shape:
                event["skip_reason"] = "cache_shape_mismatch"
                event["cache_shape_mismatch_count"] = 1
                self._kv_runtime_restoration_counters["calm_cache_shape_mismatch_count"] += 1
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            if not torch.isfinite(restored_key).all() or not torch.isfinite(restored_value).all():
                event["skip_reason"] = "nan_or_inf"
                event["nan_or_inf_count"] = 1
                self._kv_runtime_restoration_counters["calm_nan_or_inf_count"] += 1
                self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
                return _finish(present_key_value_state)
            restored_key_tensor = key_tensor.clone()
            restored_value_tensor = value_tensor.clone()
            restored_key_tensor[:, :, cache_position : cache_position + 1, :] = restored_key
            restored_value_tensor[:, :, cache_position : cache_position + 1, :] = restored_value
            self._kv_runtime_restoration_counters["calm_restored_token_layer_units"] += 1
            event["actual_restoration_applied"] = True
            event["fallback_state_copy"] = False
            event["skip_reason"] = None
            return _finish((restored_key_tensor, restored_value_tensor) + tuple(present_key_value_state[2:]))
        except Exception as exc:
            event["skip_reason"] = "restore_exception"
            event["error_message"] = str(exc)
            self._kv_runtime_restoration_counters["calm_state_copy_fallback_token_layer_units"] += 1
            return _finish(present_key_value_state)

    def _attach_runtime_source_kv_for_skip(self, row, present_key_value_states, past_key_values, source_layer):
        if not self._runtime_restoration_enabled():
            return row
        if self._phase3c_runtime_restoration_enabled():
            row.update(
                {
                    "runtime_source_kv_required": False,
                    "runtime_source_kv_available": False,
                    "runtime_source_kv_skipped_reason": "phase3c_uses_source_hidden",
                }
            )
            return row
        row["runtime_source_kv_required"] = True
        try:
            if source_layer is None or int(source_layer) < 0:
                raise ValueError("source_layer_unavailable")
            source_layer = int(source_layer)
            if present_key_value_states is None or len(present_key_value_states) <= source_layer:
                raise ValueError("source_present_key_value_state_unavailable")
            source_past = past_key_values[source_layer] if past_key_values is not None and len(past_key_values) > source_layer else None
            source_state = present_key_value_states[source_layer]
            if source_state is None or len(source_state) < 2:
                raise ValueError("source_state_missing_self_attention_kv")
            key_tensor = source_state[0]
            value_tensor = source_state[1]
            if key_tensor is None or value_tensor is None:
                raise ValueError("source_self_attention_kv_is_none")
            start, end, slice_mode, past_len = self._source_slice_for_skip_dump(key_tensor, source_past)
            if start is None:
                raise ValueError("could_not_infer_source_token_slice")
            source_key = key_tensor[:, :, start:end, :].detach().clone()
            source_value = value_tensor[:, :, start:end, :].detach().clone()
            row.update(
                {
                    "runtime_source_kv_available": True,
                    "runtime_source_slice_start": start,
                    "runtime_source_slice_end": end,
                    "runtime_source_slice_mode": slice_mode,
                    "runtime_source_past_len": past_len,
                    "runtime_source_key_shape": list(source_key.shape),
                    "runtime_source_value_shape": list(source_value.shape),
                    "_runtime_source_key": source_key,
                    "_runtime_source_value": source_value,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "runtime_source_kv_available": False,
                    "runtime_source_error_message": str(exc),
                }
            )
        return row

    def _load_runtime_source_kv_from_record(self, source_record):
        if not isinstance(source_record, dict):
            raise ValueError("source_record_missing")
        source_key = source_record.get("_runtime_source_key")
        source_value = source_record.get("_runtime_source_value")
        if source_key is not None and source_value is not None:
            return source_key, source_value, "memory"
        file_path = source_record.get("source_kv_file_path") or source_record.get("file_path")
        if not file_path:
            raise ValueError("runtime_source_kv_unavailable")
        payload = torch.load(file_path, map_location="cpu")
        source_key = payload.get("source_key")
        source_value = payload.get("source_value")
        if source_key is None or source_value is None:
            raise ValueError("source_kv_file_missing_tensors")
        return source_key, source_value, "file"

    def _make_phase3c_source_hidden_record(
        self,
        raw_hidden,
        phase3c_source_hidden_layer,
        exit_layer,
        decoder_position,
        relative_index,
        confidence=None,
    ):
        raw_hidden_record = raw_hidden.detach().clone() if hasattr(raw_hidden, "detach") else raw_hidden
        calibration_source_mode = None
        calibration_layer_scope_match = False
        calibration_runtime_scope_match = False
        confidence_semantics_verified = False
        if self.kv_runtime_restorer is not None:
            calibration_source_mode = getattr(self.kv_runtime_restorer, "calibration_source_mode", None)
            calibration_layer_scope_match = bool(
                getattr(self.kv_runtime_restorer, "calibration_layer_scope_match", False)
            )
            calibration_runtime_scope_match = bool(
                getattr(self.kv_runtime_restorer, "calibration_runtime_scope_match", False)
            )
            confidence_semantics_verified = bool(
                getattr(self.kv_runtime_restorer, "confidence_semantics_verified", False)
            )
            generation_quality_claim_valid = bool(
                getattr(self.kv_runtime_restorer, "generation_quality_claim_valid", False)
            )
            fixed_layer_membership_semantics_verified = bool(
                getattr(self.kv_runtime_restorer, "fixed_layer_membership_semantics_verified", False)
            )
        else:
            generation_quality_claim_valid = False
            fixed_layer_membership_semantics_verified = False
        runtime_source_mode = "fixed_shallow_layer"
        return {
            "raw_hidden": raw_hidden_record,
            "phase3c_source_hidden_layer": int(phase3c_source_hidden_layer),
            "exit_layer": int(exit_layer),
            "catchup_start_layer": int(exit_layer),
            "decoder_position": decoder_position,
            "relative_index": int(relative_index),
            "confidence": confidence,
            "raw_hidden_shape": list(raw_hidden.shape) if hasattr(raw_hidden, "shape") else None,
            "calibration_source_mode": calibration_source_mode,
            "runtime_source_mode": runtime_source_mode,
            "calibration_layer_scope_match": calibration_layer_scope_match,
            "calibration_runtime_scope_match": calibration_runtime_scope_match,
            "confidence_semantics_verified": confidence_semantics_verified,
            "fixed_layer_membership_semantics_verified": fixed_layer_membership_semantics_verified,
            "generation_quality_claim_valid": generation_quality_claim_valid,
        }

    def _maybe_make_phase3c_source_hidden_record_for_skip(
        self,
        raw_hidden,
        phase3c_source_hidden_layer,
        exit_layer,
        decoder_position,
        relative_index,
        confidence=None,
    ):
        if not self._phase3c_runtime_restoration_enabled() and not self._f2a_enabled():
            return None
        return self._make_phase3c_source_hidden_record(
            raw_hidden,
            phase3c_source_hidden_layer=phase3c_source_hidden_layer,
            exit_layer=exit_layer,
            decoder_position=decoder_position,
            relative_index=relative_index,
            confidence=confidence,
        )

    def _phase3c_source_hidden_record_trace(self, record):
        if not isinstance(record, dict):
            return None
        return {
            key: self._json_safe(value)
            for key, value in record.items()
            if key != "raw_hidden"
        }

    def _runtime_restoration_plan(
        self,
        pending_skipped_tokens,
        pending_metadata,
        source_records,
        layer_idx,
        phase3c_source_hidden_records=None,
    ):
        if self._runtime_restoration_enabled():
            runtime_restoration_flush_id = int(getattr(self, "_runtime_restoration_next_flush_id", 0))
            self._runtime_restoration_next_flush_id = runtime_restoration_flush_id + 1
        else:
            runtime_restoration_flush_id = None
        catchup_start_layer = int(layer_idx)
        try:
            decoder_layer_count = len(self.block)
        except Exception:
            decoder_layer_count = catchup_start_layer
        expected_target_layers = list(range(catchup_start_layer, decoder_layer_count))
        plan = {
            "enabled": False,
            "reason": "disabled",
            "runtime_restoration_flush_id": runtime_restoration_flush_id,
            "restore_relative_indices": [],
            "exact_relative_indices": list(range(int(pending_skipped_tokens or 0))),
            "pending_skipped_tokens": int(pending_skipped_tokens or 0),
            "restoration_method": getattr(self.config, "kv_runtime_restoration_method", None),
            "recent_exact_window": getattr(self.config, "kv_runtime_restoration_recent_exact_window", 0),
            "max_pending_tokens": getattr(self.config, "kv_runtime_restoration_max_pending_tokens", 0),
            "force_restore_all": bool(getattr(self.config, "kv_runtime_restoration_force_restore_all", False)),
            "start_layer": layer_idx,
            "threshold": getattr(self.config, "kv_runtime_restoration_threshold", None),
            "runtime_mode": (
                PHASE3C_RUNTIME_MODE
                if self._phase3c_runtime_restoration_enabled()
                else "legacy_exact_overwrite"
            ),
            "exact_catchup_already_computed": True,
            "speed_claim_valid": False,
            "catchup_start_layer": catchup_start_layer,
            "catchup_end_layer": decoder_layer_count - 1,
            "expected_target_layers": expected_target_layers,
            "expected_target_layer_count": len(expected_target_layers),
            "expected_restore_relative_indices": [],
            "expected_token_layer_record_count": 0,
            "source_record_count": len(source_records or ()),
            "phase3c_source_hidden_record_count": len(phase3c_source_hidden_records or ()),
        }
        if not self._runtime_restoration_enabled():
            return plan
        if pending_skipped_tokens <= 0:
            plan["reason"] = "no_pending_tokens"
            return plan
        if self.config.copy_skipped_hidden_states:
            raise NotImplementedError("runtime restored-K/V requires copy_skipped_hidden_states=False")
        if self._phase3c_runtime_restoration_enabled():
            if len(phase3c_source_hidden_records or []) != int(pending_skipped_tokens):
                plan["reason"] = "phase3c_source_hidden_record_count_mismatch"
                plan["phase3c_source_hidden_record_count"] = len(phase3c_source_hidden_records or [])
                return plan
        elif len(source_records or []) != int(pending_skipped_tokens):
            plan["reason"] = "source_record_count_mismatch"
            plan["source_record_count"] = len(source_records or [])
            return plan
        max_pending = int(getattr(self.config, "kv_runtime_restoration_max_pending_tokens", 0) or 0)
        if max_pending > 0 and int(pending_skipped_tokens) > max_pending:
            plan["reason"] = "max_pending_tokens_exceeded"
            return plan
        if bool(getattr(self.config, "kv_runtime_restoration_force_restore_all", False)):
            recent_window = 0
        else:
            recent_window = int(getattr(self.config, "kv_runtime_restoration_recent_exact_window", 0) or 0)
        recent_window = max(0, min(int(pending_skipped_tokens), recent_window))
        restore_end = int(pending_skipped_tokens) - recent_window
        restore_indices = list(range(restore_end))
        exact_indices = list(range(restore_end, int(pending_skipped_tokens)))
        plan.update(
            {
                "enabled": True,
                "reason": "enabled",
                "restore_relative_indices": restore_indices,
                "exact_relative_indices": exact_indices,
                "restored_token_count": len(restore_indices),
                "exact_catchup_token_count": len(exact_indices),
                "pending_metadata_count": len(pending_metadata or ()),
                "expected_restore_relative_indices": list(restore_indices),
                "expected_token_layer_record_count": len(expected_target_layers) * len(restore_indices),
            }
        )
        self._kv_runtime_restoration_counters["pending_skipped_tokens"] += int(pending_skipped_tokens)
        self._kv_runtime_restoration_counters["restored_token_count"] += len(restore_indices)
        self._kv_runtime_restoration_counters["exact_catchup_token_count"] += len(exact_indices)
        return plan

    def _stage_runtime_restored_kv_slices(
        self,
        result,
        key_tensor,
        value_tensor,
        cache_position,
        span=1,
        defer_finite_validation=False,
    ):
        """Validate one restored K/V block against the cache slice it will
        occupy, without touching the live cache.

        ``span`` is the number of consecutive cache positions the restored
        block covers, starting at ``cache_position``. It defaults to 1, which
        is exactly the existing single-token behaviour every current caller
        relies on; the FREE-aligned lazy batched flush passes span=N to stage
        all N pending tokens as one contiguous block.

        ``defer_finite_validation`` defaults to False, so every existing
        caller keeps the per-layer nonfinite check here. The batched Task C2
        transaction sets it True and instead runs ONE aggregate finite check
        over the assembled candidate cache before it commits anything. The
        staged tensors are copied verbatim into that candidate, so the
        aggregate sees exactly the values this check would have seen. All
        shape, device and dtype validation below is unconditional in both
        modes."""

        if result.restored_key is None or result.restored_value is None:
            return None, None, "restored_kv_missing", "restored K/V result is incomplete"
        span = int(span)
        if span < 1:
            return None, None, "cache_shape_mismatch", "restored K/V span must be >= 1, got {}".format(span)
        target_key_slice = key_tensor[:, :, cache_position : cache_position + span, :]
        target_value_slice = value_tensor[:, :, cache_position : cache_position + span, :]
        try:
            staged_key = result.restored_key.to(device=target_key_slice.device, dtype=target_key_slice.dtype)
            staged_value = result.restored_value.to(device=target_value_slice.device, dtype=target_value_slice.dtype)
        except Exception as exc:
            return None, None, "cache_shape_mismatch", "restored K/V conversion failed: {}".format(exc)
        if staged_key.shape != target_key_slice.shape:
            return (
                None,
                None,
                "restored_key_shape_mismatch",
                "restored key shape {} does not match target key slice {}".format(
                    list(staged_key.shape),
                    list(target_key_slice.shape),
                ),
            )
        if staged_value.shape != target_value_slice.shape:
            return (
                None,
                None,
                "restored_value_shape_mismatch",
                "restored value shape {} does not match target value slice {}".format(
                    list(staged_value.shape),
                    list(target_value_slice.shape),
                ),
            )
        if staged_key.device != target_key_slice.device or staged_value.device != target_value_slice.device:
            return None, None, "cache_shape_mismatch", "restored K/V device mismatch after conversion"
        if staged_key.dtype != target_key_slice.dtype or staged_value.dtype != target_value_slice.dtype:
            return None, None, "cache_shape_mismatch", "restored K/V dtype mismatch after conversion"
        if not defer_finite_validation and (
            not torch.isfinite(staged_key).all().item() or not torch.isfinite(staged_value).all().item()
        ):
            return None, None, "nan_or_inf", "restored K/V contains NaN or Inf"
        return staged_key, staged_value, None, None

    def _commit_runtime_restored_kv_slices(
        self,
        restored_key_tensor,
        restored_value_tensor,
        cache_position,
        staged_key,
        staged_value,
        span=1,
        candidate_is_private=False,
    ):
        """Write the staged K/V block into a CLONE of the cache tensors and
        return the clone, so the caller can discard it and leave the live
        cache untouched on any later failure.

        ``span`` matches ``_stage_runtime_restored_kv_slices`` and defaults to
        the existing single-position behaviour.

        ``candidate_is_private`` is an opt-in assertion by the caller that
        ``restored_key_tensor``/``restored_value_tensor`` are freshly
        allocated temporaries which cannot alias the live model cache and are
        referenced nowhere else, so the defensive full-tensor clone is pure
        overhead and is skipped. It defaults to False, which keeps every
        existing caller byte-identical.

        Only the two Task C2 insertion helpers set it. Both build their
        candidate with ``torch.cat([old_cache_slice, zeros])``, which always
        allocates a new tensor and copies the old cache into it, and both
        abandon the ENTIRE transaction with a `_fail(...)` return if anything
        (including the commit test hook firing between the key and value
        writes) raises -- so a half-written private candidate is simply
        discarded and the authoritative past_key_values is never touched.

        The other two callers deliberately do NOT opt in: the CALM Task C1
        exact-overwrite path passes the exact reference state returned by a
        real block forward (retained for evidence/comparison after the
        commit), and the pending-token restoration loop rebinds its candidate
        across iterations and relies on an exception leaving the previous
        iteration's tensor intact.
        """

        span = int(span)
        if candidate_is_private:
            candidate_key_tensor = restored_key_tensor
            candidate_value_tensor = restored_value_tensor
        else:
            candidate_key_tensor = restored_key_tensor.clone()
            candidate_value_tensor = restored_value_tensor.clone()
        candidate_key_slice = candidate_key_tensor[:, :, cache_position : cache_position + span, :]
        candidate_value_slice = candidate_value_tensor[:, :, cache_position : cache_position + span, :]
        candidate_key_slice.copy_(staged_key)
        commit_hook = getattr(self, "_runtime_restoration_commit_test_hook", None)
        if callable(commit_hook):
            commit_hook(
                "after_key_candidate_write",
                candidate_key_tensor,
                candidate_value_tensor,
                cache_position,
            )
        candidate_value_slice.copy_(staged_value)
        return candidate_key_tensor, candidate_value_tensor

    def _run_f2a_followup_query_to_decision(
        self,
        *,
        decision_layer,
        decision_type,
        query_initial_hidden,
        candidate_past_key_values,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
        lm_head,
    ):
        decision_layer = int(decision_layer)
        hidden = query_initial_hidden.detach().clone()
        position_bias = None
        cross_position_bias = encoder_decoder_position_bias
        if str(decision_type) == "first_crossing":
            stop_exclusive = decision_layer
        elif str(decision_type) == "full_depth_fallback":
            stop_exclusive = len(self.block)
        else:
            raise ValueError("f2a_followup_decision_type_unsupported:{}".format(decision_type))
        for target_layer in range(0, stop_exclusive):
            layer_outputs = self.block[target_layer](
                hidden,
                attention_mask=None,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=cross_position_bias,
                layer_head_mask=head_mask[target_layer] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[target_layer] if cross_attn_head_mask is not None else None,
                past_key_value=candidate_past_key_values[target_layer],
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=False,
                parallel_mask=False,
                layer_idx=target_layer,
                kv_importance_tracker=None,
            )
            hidden = layer_outputs[0]
            position_bias = layer_outputs[2] if len(layer_outputs) > 2 else None
            if self.is_decoder and encoder_hidden_states is not None:
                cross_position_bias = layer_outputs[4 if output_attentions else 3]
        return self._f2a_logits_from_hidden(hidden, lm_head)

    def _finalize_f2a_pending_calm_event(
        self,
        *,
        followup_query_initial_hidden,
        followup_reference_decision_layer,
        followup_reference_decision_type,
        followup_reference_logits,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
        lm_head,
    ):
        pending = getattr(self, "_f2a_pending_calm_event", None)
        if not pending:
            return None
        try:
            event = dict(pending["event"])
            event["followup_decoder_input_token_id"] = self._f2a_current_input_token_id()
            event["followup_decoder_input_position"] = self._f2a_current_input_position()
            event["followup_predicted_token_position"] = self._f2a_predicted_token_position(self._f2a_current_input_position() + 1)
            followup_prefix = self._f2a_prefix_token_ids()
            event["followup_reference_decoder_prefix_token_ids"] = followup_prefix
            from our_kv_restoration.f2a_frozen_schedule import prefix_token_sha256, make_f2a_event_uid

            event["followup_prefix_token_sha256"] = prefix_token_sha256(followup_prefix)
            event["followup_reference_decision_layer"] = int(followup_reference_decision_layer)
            event["followup_reference_decision_type"] = str(followup_reference_decision_type)
            event["frozen_event_uid"] = make_f2a_event_uid(event)
            exact_cache = pending["exact_cache"]
            exact_snapshot = cache_noninterference_snapshot(exact_cache, event)
            exact_shadow_logits = self._run_f2a_followup_query_to_decision(
                decision_layer=followup_reference_decision_layer,
                decision_type=followup_reference_decision_type,
                query_initial_hidden=followup_query_initial_hidden,
                candidate_past_key_values=exact_cache,
                encoder_hidden_states=encoder_hidden_states,
                encoder_extended_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lm_head=lm_head,
            )
            parity = validate_exact_shadow_parity(
                f2a_logit_metrics(followup_reference_logits, exact_shadow_logits),
                dtype=followup_reference_logits.dtype,
            )
            if parity.get("status") != "ok":
                self._missing_kv_accounting_obj().record_f2a_exact_shadow_parity_failure()
                raise ValueError("f2a_calm_deferred_exact_shadow_parity_failed:{}".format(parity.get("errors")))
            records = []
            for method, candidate_cache in sorted(pending["candidate_caches"].items()):
                if caches_share_writable_storage(exact_cache, candidate_cache):
                    raise ValueError("f2a_calm_candidate_cache_aliases_reference:{}".format(method))
                candidate_logits = self._run_f2a_followup_query_to_decision(
                    decision_layer=followup_reference_decision_layer,
                    decision_type=followup_reference_decision_type,
                    query_initial_hidden=followup_query_initial_hidden,
                    candidate_past_key_values=candidate_cache,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_extended_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    head_mask=head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lm_head=lm_head,
                )
                records.append(
                    build_f2a_component_record(
                        event,
                        method=method,
                        metrics=f2a_logit_metrics(followup_reference_logits, candidate_logits),
                        status="ok",
                        artifact_file_sha256=self.kv_f2a_artifact_sha256,
                        policy_sha256=self.kv_f2a_policy_sha256,
                        replay_diagnostics={
                            "exact_shadow_parity": parity,
                            "candidate_output_used_for_reference_continuation": False,
                            "candidate_cache_aliases_reference": False,
                            "followup_query_frozen": True,
                            "followup_reference_schedule_frozen": True,
                            "candidate_schedule_recomputed": False,
                        },
                    )
                )
            noninterference = validate_cache_noninterference(exact_snapshot, exact_cache, event=event)
            if noninterference.get("status") != "ok":
                self._missing_kv_accounting_obj().record_f2a_reference_state_mutation_failure()
                raise ValueError("f2a_calm_deferred_reference_cache_mutated:{}".format(noninterference.get("errors")))
            for record in records:
                record.setdefault("replay_diagnostics", {})["reference_cache_noninterference"] = noninterference
            self._f2a_record_event_and_rows(event, records)
            self._missing_kv_accounting_obj().record_f2a_calm_event_finalized()
            self._f2a_pending_calm_event = None
            return event
        except Exception:
            failed_event_uid = None
            try:
                failed_event_uid = event.get("frozen_event_uid")
            except Exception:
                failed_event_uid = None
            self._missing_kv_accounting_obj().record_f2a_failed_event(
                event_uid=failed_event_uid
            )
            self._missing_kv_accounting_obj().record_f2a_calm_event_failure()
            self._f2a_pending_calm_event = None
            raise

    def finalize_f2a_generation_end(self, terminal_reason="unknown_generation_end"):
        if not self._f2a_enabled():
            return
        terminal_reason = str(terminal_reason or "unknown_generation_end")
        if getattr(self, "_f2a_reference_generation_finalized", False):
            return
        pending_calm_event = getattr(self, "_f2a_pending_calm_event", None)
        if pending_calm_event is not None:
            if terminal_reason == "generation_failure":
                event_uid = None
                try:
                    event_uid = (pending_calm_event.get("event") or {}).get("frozen_event_uid")
                except Exception:
                    event_uid = None
                accounting = self._missing_kv_accounting_obj()
                if event_uid not in (None, ""):
                    accounting.record_f2a_failed_event(event_uid=event_uid)
                else:
                    accounting.record_f2a_schedule_validation_failure()
                    accounting.record_f2a_failed_event()
                accounting.record_f2a_calm_event_failure()
                self._f2a_pending_calm_event = None
            else:
                self._missing_kv_accounting_obj().record_f2a_calm_terminal_no_followup()
                terminal_reason = "terminal_no_followup"
                if getattr(self, "_f2a_reference_decision_trace", None):
                    self._f2a_reference_decision_trace[-1]["terminal_no_followup"] = True
                    self._f2a_reference_decision_trace[-1]["terminal_reason"] = terminal_reason
                self._f2a_pending_calm_event = None
        context = self._missing_kv_sample_context_fields()
        stable_sample_id = context.get("stable_sample_id")
        if stable_sample_id in (None, ""):
            if getattr(self, "_f2a_reference_generated_token_ids", None):
                self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            return
        try:
            row = build_reference_generation_trajectory_row(
                stable_sample_id=stable_sample_id,
                generation_index=self._generation_index,
                selected_order=context.get("selected_order"),
                raw_dataset_index=context.get("raw_dataset_index"),
                dataset_provided_id=context.get("dataset_provided_id"),
                generated_token_ids=list(getattr(self, "_f2a_reference_generated_token_ids", []) or []),
                reference_decision_trace=list(getattr(self, "_f2a_reference_decision_trace", []) or []),
                terminal_reason=terminal_reason,
                generation_completed=terminal_reason != "generation_failure",
                decoder_layer_count=len(getattr(self, "block", []) or []),
            )
        except Exception:
            self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            raise
        self._f2a_reference_trajectory_rows = list(getattr(self, "_f2a_reference_trajectory_rows", []) or [])
        self._f2a_reference_trajectory_rows.append(dict(row))
        self._f2a_append_output_row(getattr(self.config, "kv_f2a_reference_trajectory_output", None), row)
        self._f2a_write_reference_trajectory_summary()
        self._f2a_reference_generation_finalized = True

    def _run_f2a_calm_exact_reference_and_shadow_replay(
        self,
        *,
        source_layer,
        source_hidden,
        exit_logits,
        confidence,
        candidate_evaluations,
        hidden_states,
        extended_attention_mask,
        position_bias,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        past_key_values,
        present_key_value_states,
        use_cache,
        output_attentions,
        lm_head,
    ):
        if not self._is_f2a_calm_enabled():
            return None
        if self.kv_f2a_artifact is None:
            raise ValueError("f2a_policy_artifact_not_loaded")
        if not use_cache:
            self._missing_kv_accounting_obj().record_f2a_skipped("calm_requires_use_cache")
            raise ValueError("f2a_calm_requires_use_cache")
        source_layer = int(source_layer)
        num_layers = len(self.block)
        target_layers = list(range(source_layer, num_layers))
        decoder_position = infer_decoder_position(past_key_values)
        reference_hidden = source_hidden
        reference_position_bias = position_bias
        reference_encoder_decoder_position_bias = encoder_decoder_position_bias
        exact_states_by_layer = {}
        exact_target_states = []
        for target_layer in target_layers:
            layer_module = self.block[target_layer]
            past_key_value = past_key_values[target_layer] if past_key_values is not None else None
            layer_outputs = layer_module(
                reference_hidden,
                attention_mask=extended_attention_mask,
                position_bias=reference_position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=reference_encoder_decoder_position_bias,
                layer_head_mask=head_mask[target_layer] if head_mask is not None else None,
                cross_attn_layer_head_mask=cross_attn_head_mask[target_layer] if cross_attn_head_mask is not None else None,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=False,
                layer_idx=target_layer,
                kv_importance_tracker=self.kv_importance if self.is_decoder else None,
            )
            self.block_op[target_layer] += 1
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            reference_hidden, exact_state = layer_outputs[:2]
            reference_position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                reference_encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if exact_state is None or len(exact_state) < 2:
                raise ValueError("f2a_calm_exact_state_missing:{}".format(target_layer))
            exact_states_by_layer[target_layer] = tuple(
                item.detach().clone() if isinstance(item, torch.Tensor) else item
                for item in exact_state
            )
            exact_target_states.append(tuple(item.detach().clone() if isinstance(item, torch.Tensor) else item for item in exact_state))
        prefix_states = list(present_key_value_states or [])
        if len(prefix_states) != source_layer:
            raise ValueError("f2a_calm_prefix_cache_length_mismatch")
        complete_exact_cache = prefix_states + exact_target_states
        if len(complete_exact_cache) != num_layers:
            raise ValueError("f2a_calm_exact_cache_length_mismatch")
        reference_identity = self._f2a_reference_identity()
        past_len = safe_cache_seq_len(past_key_values[source_layer]) if past_key_values is not None else 0
        past_len = 0 if past_len is None else int(past_len)
        evaluations = list(candidate_evaluations or [])
        context = self._missing_kv_sample_context_fields()
        if context.get("stable_sample_id") in (None, ""):
            self._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
            self._missing_kv_accounting_obj().record_f2a_failed_event()
            raise ValueError("f2a_stable_sample_id_unavailable")
        max_events = int(getattr(self.config, "kv_f2a_max_events", 0) or 0)
        self._missing_kv_accounting_obj().record_f2a_calm_event_considered()
        if max_events > 0 and len(self._f2a_schedule_rows) >= max_events:
            self._missing_kv_accounting_obj().record_f2a_calm_event_cap_skipped()
            self.lm_logits = exit_logits
            return source_hidden, complete_exact_cache, {"f2a_event_skipped": "max_events_reached"}
        event = build_f2a_event(
            stable_sample_id=context.get("stable_sample_id"),
            generation_index=self._generation_index,
            decoder_position=int(decoder_position),
            current_decoder_input_position=self._f2a_current_input_position(),
            predicted_token_position=self._f2a_predicted_token_position(decoder_position),
            current_decoder_input_token_id=self._f2a_current_input_token_id(),
            prefix_token_ids=self._f2a_prefix_token_ids(),
            source_layer_mode=SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            source_layer=source_layer,
            decoder_layer_count=num_layers,
            pending_token_positions=[int(decoder_position)],
            restore_relative_indices=[0],
            cache_positions=[past_len],
            reference_schedule_identity=reference_identity,
            reference_run_identity=reference_identity,
            artifact_file_sha256=self.kv_f2a_artifact_sha256,
            policy_sha256=self.kv_f2a_policy_sha256,
            candidate_policy_sha256=self.kv_f2a_candidate_policy_sha256,
            policy_name=CALM_POLICY_NAME,
            candidate_layers=CALM_CANDIDATE_LAYERS,
            source_confidence=float(confidence),
            candidate_evaluations=evaluations,
            event_timing="deferred_followup",
            restored_token_position=int(decoder_position),
            threshold=float(CALM_THRESHOLD),
            threshold_comparator="strict_gt",
            confidence_compute_dtype="float32",
            adaptive_threshold=False,
        )
        if safe_cache_seq_len(complete_exact_cache[source_layer]) != past_len + 1:
            raise ValueError("f2a_calm_exact_cache_after_token_length_mismatch")
        exact_candidate_cache = clone_cache(complete_exact_cache)
        candidate_caches = {}
        for method in self._f2a_methods():
            restored_by_layer = {}
            for target_layer in target_layers:
                exact_state = exact_states_by_layer[target_layer]
                layer_norm, key_projection, value_projection = self._f2a_target_self_attention_modules(target_layer)
                restored_key, restored_value, _metadata = restore_f2a_method_block_from_hidden(
                    method=method,
                    artifact=self.kv_f2a_artifact,
                    source_hidden=source_hidden,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    target_layer_norm=layer_norm,
                    key_projection=key_projection,
                    value_projection=value_projection,
                    threshold=float(CALM_THRESHOLD),
                    output_device=exact_state[0].device,
                    output_dtype=exact_state[0].dtype,
                )
                restored_by_layer[target_layer] = (restored_key, restored_value)
            candidate_cache = patch_candidate_cache_for_full_event(
                exact_candidate_cache,
                event=event,
                restored_by_target_layer=restored_by_layer,
            )
            if safe_cache_seq_len(candidate_cache[source_layer]) != past_len + 1:
                raise ValueError("f2a_calm_candidate_cache_after_token_length_mismatch")
            candidate_caches[str(method)] = candidate_cache
        self._f2a_pending_calm_event = {
            "event": event,
            "exact_cache": exact_candidate_cache,
            "candidate_caches": candidate_caches,
        }
        self._missing_kv_accounting_obj().record_f2a_calm_event_produced()
        self.lm_logits = exit_logits
        return source_hidden, complete_exact_cache, {"f2a_pending_event": event}

    def _calm_taskc1_authoritative_restoration_cache_positions(self, *, past_key_values, target_layers):
        positions = []
        positions_by_layer = {}
        for target_layer in target_layers:
            target_layer = int(target_layer)
            try:
                past_key_value = past_key_values[target_layer] if past_key_values is not None else None
            except (TypeError, IndexError):
                cache_position = None
            else:
                past_len = safe_cache_seq_len(past_key_value)
                cache_position = 0 if past_len is None else int(past_len)
            positions.append(cache_position)
            positions_by_layer[target_layer] = cache_position
        return positions, positions_by_layer

    def _run_calm_phase3c_exact_reference_and_stage_cache(
        self,
        *,
        source_layer,
        source_hidden,
        exit_logits,
        confidence,
        candidate_evaluations,
        hidden_states,
        extended_attention_mask,
        position_bias,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        past_key_values,
        present_key_value_states,
        use_cache,
        output_attentions,
        calibration_prefix_hidden_states=None,
    ):
        source_layer = int(source_layer)
        num_layers = len(self.block)
        target_layers = list(range(source_layer, num_layers))
        requested_units = len(target_layers)
        (
            authoritative_restoration_positions,
            authoritative_restoration_position_by_layer,
        ) = self._calm_taskc1_authoritative_restoration_cache_positions(
            past_key_values=past_key_values,
            target_layers=target_layers,
        )
        inferred_decoder_position = infer_decoder_position(past_key_values)
        decoder_position_resolution = resolve_exact_catchup_event_decoder_position(
            inferred_decoder_position=inferred_decoder_position,
            past_key_values=past_key_values,
            target_layers=target_layers,
            first_token_context=self._calm_taskc1_first_token_context_confirmed(
                present_key_value_states,
                source_layer,
            ),
            restoration_cache_positions=authoritative_restoration_positions,
        )
        decoder_position = (
            decoder_position_resolution.get("decoder_position")
            if decoder_position_resolution.get("status") == "ok"
            else inferred_decoder_position
        )
        restoration_method = getattr(self.kv_runtime_restorer, "method", None)
        runtime_source_mode = getattr(self.kv_runtime_restorer, "runtime_source_mode", None)
        is_exact_catchup = restoration_method == EXACT_CATCHUP_METHOD
        calibration_collector = getattr(self, "exact_cache_calibration_collector", None)
        calibration_hidden_by_layer = None
        if calibration_collector is not None:
            if not is_exact_catchup:
                raise ValueError("exact_cache_calibration_requires_exact_catchup")
            calibration_hidden_by_layer = list(calibration_prefix_hidden_states or [])
            if len(calibration_hidden_by_layer) != source_layer:
                raise ValueError(
                    "calibration_prefix_hidden_layer_count_mismatch:{}:{}".format(
                        len(calibration_hidden_by_layer), source_layer
                    )
                )
        runtime_mode = getattr(self.kv_runtime_restorer, "runtime_mode", PHASE3C_RUNTIME_MODE)
        evidence_prefix = self._calm_taskc1_evidence_prefix(restoration_method)

        # Exact-catch-up overhead event recording is diagnostic-only and
        # opt-in: any failure to construct/register an event (e.g. identity
        # fields not yet available this early in generation) must never
        # break the real transaction below, so this is deliberately isolated
        # in its own try/except rather than allowed to propagate.
        overhead_recorder = self._missing_kv_exact_catchup_overhead_recorder_obj()
        overhead_event = None
        if self._exact_catchup_overhead_enabled():
            try:
                sample_context = self._exact_catchup_overhead_sample_context_fields()
                if sample_context.get("stable_sample_id") not in (None, ""):
                    if decoder_position_resolution.get("status") != "ok":
                        overhead_recorder.record_missing_identity_skip(
                            reason="candidate_first_crossing_decoder_position_resolution_failed:{}".format(
                                decoder_position_resolution.get("reason") or "unknown"
                            )
                        )
                    else:
                        candidate_overhead_event = ExactCatchupOverheadEvent(
                            runtime_path=RUNTIME_PATH_CANDIDATE_FIRST_CROSSING,
                            transaction_type="candidate_first_crossing",
                            source_layer=source_layer,
                            decoder_layer_count=num_layers,
                            stable_sample_id=sample_context.get("stable_sample_id"),
                            selected_order=sample_context.get("selected_order"),
                            raw_dataset_index=sample_context.get("raw_dataset_index"),
                            generation_index=self._generation_index,
                            decoder_position=decoder_position,
                            candidate_policy_name=CALM_POLICY_NAME,
                            candidate_layers=list(CALM_CANDIDATE_LAYERS),
                            threshold=float(getattr(self.kv_runtime_restorer, "candidate_threshold", CALM_THRESHOLD)),
                            threshold_comparator=getattr(
                                self.kv_runtime_restorer, "candidate_threshold_comparator", "strict_gt"
                            ),
                            adaptive_threshold=False,
                        )
                        overhead_recorder.register(candidate_overhead_event)
                        overhead_event = candidate_overhead_event
                else:
                    # A missing stable identity must never be a silent no-op:
                    # instrumentation stays out of the real transaction below,
                    # but the run is explicitly marked invalid so full paper
                    # mode fails closed instead of reporting a quietly
                    # incomplete event population.
                    overhead_recorder.record_missing_identity_skip(
                        reason="candidate_first_crossing_stable_sample_id_unavailable"
                    )
            except Exception as exc:
                overhead_event = None
                overhead_recorder.record_missing_identity_skip(
                    reason="candidate_first_crossing_event_construction_failed:{}".format(type(exc).__name__)
                )

        def _initial_target_record(target_layer):
            return {
                "source_layer": source_layer,
                "target_layer": int(target_layer),
                "restoration_method": restoration_method,
                "runtime_mode": runtime_mode,
                "direct_source_kv_layer": None,
                "cache_position": None,
                "past_key_value_len": None,
                "target_key_shape": None,
                "target_value_shape": None,
                "status": "not_attempted_due_to_transaction_failure",
                "exact_reference_status": "not_attempted",
                "restoration_status": "not_attempted",
                "restoration_submode": None,
                "gap_bin": None,
                "regenerated_key_shape": None,
                "regenerated_value_shape": None,
                "restored_key_shape": None,
                "restored_value_shape": None,
                "cache_staging_status": "not_attempted",
                "candidate_cache_write_status": "not_attempted",
                "cache_commit_status": "local_candidate_write_not_attempted",
                "transaction_cache_install_status": "not_attempted",
                "final_status": "not_attempted_due_to_transaction_failure",
                "speed_claim_valid": False,
            }

        # Official CALM three-arm pure-recovery measurement event for THIS
        # first crossing (None unless the measurement is enabled on the
        # Official CALM + Exact live arm). Created after the transaction's
        # own source_hidden/prefix validation below; _record_failure marks
        # it invalid if the live transaction fails.
        calm_pure_event = None

        event = {
            "runtime_mode": runtime_mode,
            "restoration_method": restoration_method,
            "approximation_strategy": "none" if is_exact_catchup else restoration_method,
            "source_layer_mode": runtime_source_mode,
            "selected_source_layer": source_layer,
            "last_exact_kv_layer": source_layer - 1,
            "first_missing_target_layer": source_layer,
            "target_layers": target_layers,
            "target_layer_count": requested_units,
            "confidence": float(confidence),
            "threshold": float(getattr(self.kv_runtime_restorer, "candidate_threshold", CALM_THRESHOLD)),
            "threshold_comparator": getattr(self.kv_runtime_restorer, "candidate_threshold_comparator", "strict_gt"),
            "candidate_evaluations": list(candidate_evaluations or []),
            "preserved_exit_logits": True,
            "preserved_exit_logits_source": "{}_source_hidden_h_{}".format(evidence_prefix, source_layer),
            "exact_reference_start_layer": source_layer,
            "exact_reference_end_layer": num_layers - 1,
            "exact_reference_skip_mask": False,
            "exact_reference_restored_kv_used_for_hidden_propagation": False,
            "transaction_status": "started",
            "phase3c_restore_records": [_initial_target_record(target_layer) for target_layer in target_layers],
            "decoder_position": decoder_position,
            "decoder_position_resolution": decoder_position_resolution,
            "speed_claim_valid": False,
            "exact_catchup_avoided_token_layer_units": 0,
            "exact_catchup_required_token_layer_units": requested_units,
            "exact_catchup_executed_token_layer_units": 0,
            "exact_cache_written_token_layer_units": 0,
            "restoration_required_token_layer_units": 0 if is_exact_catchup else requested_units,
            "restoration_attempted_token_layer_units": 0,
            "restoration_token_record_count": 0 if is_exact_catchup else requested_units,
            "restoration_stage_entered": False,
            "approximate_overwrite_stage_entered": False,
            "catchup_failure_count": 0,
            "cache_write_failure_count": 0,
            "restoration_error_failure_count": 0,
            "restoration_error_fallback_count": 0,
            "unresolved_transaction_count": 0,
        }
        executed_exact_units = 0

        def _record_failure(stage, message):
            if calm_pure_event is not None and calm_pure_event.get("measurement_error") is None:
                # The three-arm event is only paper-valid over a COMMITTED
                # live exact transaction.
                calm_pure_event["measurement_error"] = "live_transaction_failed:{}".format(stage)
            event["transaction_status"] = "failed"
            event["failure_stage"] = stage
            event["error_message"] = str(message)
            event["exact_catchup_executed_token_layer_units"] = executed_exact_units
            event["restoration_attempted_token_layer_units"] = sum(
                1
                for record in event["phase3c_restore_records"]
                if record.get("restoration_status") not in {"not_attempted", "not_attempted_due_to_transaction_failure"}
            )
            for record in event["phase3c_restore_records"]:
                if record.get("final_status") == "candidate_ready_not_committed":
                    record["transaction_cache_install_status"] = "not_committed"
                    record["final_status"] = "computed_but_not_committed_due_to_transaction_failure"
                    record["status"] = "computed_but_not_committed_due_to_transaction_failure"
                elif record.get("final_status") is None:
                    if record.get("restoration_status") not in {
                        "not_attempted",
                        "not_attempted_due_to_transaction_failure",
                    }:
                        record["final_status"] = "failed"
                        record["status"] = "failed"
                    else:
                        record["final_status"] = "not_attempted_due_to_transaction_failure"
                        record["status"] = "not_attempted_due_to_transaction_failure"
                elif record.get("final_status") in {"not_attempted_due_to_transaction_failure", "failed"}:
                    record["status"] = record.get("final_status")
                if record.get("final_status") in {
                    "not_attempted_due_to_transaction_failure",
                    "failed",
                    "computed_but_not_committed_due_to_transaction_failure",
                }:
                    record.setdefault("failure_stage", stage)
            self._kv_runtime_restoration_counters["calm_phase3c_transaction_failure_tokens"] += 1
            if is_exact_catchup:
                event["catchup_failure_count"] = 1
            else:
                self._kv_runtime_restoration_counters["calm_phase3c_requested_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["calm_phase3c_failed_token_layer_units"] += requested_units
            self._kv_runtime_restoration_counters["exact_catchup_computed_token_layer_units"] += executed_exact_units
            self._kv_runtime_restoration_counters["exact_catchup_token_layer_units"] += executed_exact_units
            accounting = self._missing_kv_accounting_obj()
            if not is_exact_catchup:
                accounting.record_restoration_flush(enabled=True, restore_relative_count=requested_units)
                accounting.record_restoration_layer(
                    requested=requested_units,
                    succeeded=0,
                    overwritten=0,
                    fallback=0,
                    token_records=len(event["phase3c_restore_records"]),
                )
            accounting.record_calm_transaction_failure()
            if overhead_event is not None:
                try:
                    overhead_event.fail(stage=stage, reason=message)
                    overhead_recorder.finalize(overhead_event)
                except Exception:
                    pass
            if self.kv_trace.enabled:
                self.kv_trace.record("kv_calm_phase3c_taskc1_transaction", **event)

        try:
            event["failure_stage"] = "transaction_start_validation"
            self._missing_kv_accounting_obj().record_exact_catchup_required(
                pending_token_count=1,
                num_catchup_layers=requested_units,
            )
            self._call_calm_phase3c_transaction_test_hook("start", event, source_layer=source_layer)
            if not use_cache:
                raise ValueError("calm_phase3c_taskc1_requires_use_cache")
            if source_hidden is None or source_hidden.ndim != 3:
                raise ValueError("source_hidden_must_have_shape_batch_seq_model")
            if int(source_hidden.shape[0]) != 1 or int(source_hidden.shape[1]) != 1:
                raise ValueError("calm_phase3c_source_hidden_must_be_single_batch_single_token")
            if not torch.isfinite(source_hidden).all().item():
                raise ValueError("calm_phase3c_source_hidden_nonfinite")
            if present_key_value_states is None:
                raise ValueError("present_key_value_states_missing")
            prefix_states = list(present_key_value_states)
            if len(prefix_states) != source_layer:
                raise ValueError(
                    "candidate_cache_prefix_length_mismatch: expected {} got {}".format(
                        source_layer,
                        len(prefix_states),
                    )
                )

            # Three-arm CALM pure-recovery measurement: ONLY the event
            # metadata is bound here. The Exact arm accumulates from the
            # live target-layer replay below (timed in place, exactly once)
            # via _calm_pure_recovery_exact_target_time_block; the Ours and
            # State shadows run strictly AFTER that replay completes, so
            # the Exact paper baseline is measured on unperturbed state.
            if is_exact_catchup and self._calm_pure_recovery_cost_enabled():
                calm_pure_event = self._calm_pure_recovery_begin_event(
                    source_layer=source_layer,
                    source_hidden=source_hidden,
                    decoder_position=decoder_position,
                )

            reference_hidden = source_hidden
            reference_position_bias = position_bias
            reference_encoder_decoder_position_bias = encoder_decoder_position_bias
            candidate_target_states = []
            for target_layer in target_layers:
                record = event["phase3c_restore_records"][target_layer - source_layer]
                layer_module = self.block[target_layer]
                past_key_value = past_key_values[target_layer] if past_key_values is not None else None
                past_len = authoritative_restoration_position_by_layer.get(int(target_layer))
                past_len = 0 if past_len is None else int(past_len)
                record["cache_position"] = past_len
                record["past_key_value_len"] = past_len
                if calibration_hidden_by_layer is not None:
                    calibration_hidden_by_layer.append(reference_hidden.detach().cpu().contiguous())
                event["failure_stage"] = "before_exact_reference_layer"
                self._call_calm_phase3c_transaction_test_hook(
                    "before_exact_reference_layer",
                    event,
                    target_layer=target_layer,
                    past_len=past_len,
                )
                event["failure_stage"] = "exact_reference_layer"
                # Paper-facing candidate timing uses MissingKVComponentTimer's
                # deferred CUDA-event mechanism rather than a host
                # perf_counter() wrapped around an asynchronous CUDA call: a
                # timing *handle* is created before the block (so it can be
                # captured by the on_resolved closure passed into
                # time_block), but is only appended to the event's
                # per_target_timing -- and executed units only incremented --
                # by commit_target_executed() after this exact target-layer
                # call has actually returned successfully. This guarantees no
                # timing/executed-units are ever fabricated for a target whose
                # call raised, while still supporting CUDA's true deferred
                # (resolve-at-finalize) resolution.
                _overhead_timing_handle = None
                if overhead_event is not None:
                    try:
                        _overhead_timing_handle = overhead_event.begin_target_timing(target_layer=target_layer)
                    except Exception:
                        _overhead_timing_handle = None

                def _on_overhead_target_timing_resolved(elapsed_ms, backend, _handle=_overhead_timing_handle):
                    if _handle is not None:
                        ExactCatchupOverheadEvent.resolve_target_timing(
                            _handle, elapsed_ms=elapsed_ms, timing_backend=backend
                        )

                with self._missing_kv_component_timer_obj().time_block(
                    "exact_parallel_catchup_time_ms",
                    device=reference_hidden.device,
                    on_resolved=(
                        _on_overhead_target_timing_resolved if _overhead_timing_handle is not None else None
                    ),
                ), self._calm_pure_recovery_exact_target_time_block(
                    calm_pure_event, reference_hidden.device
                ):
                    layer_outputs = layer_module(
                        reference_hidden,
                        attention_mask=extended_attention_mask,
                        position_bias=reference_position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=reference_encoder_decoder_position_bias,
                        layer_head_mask=head_mask[target_layer],
                        cross_attn_layer_head_mask=cross_attn_head_mask[target_layer],
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        skip_mask=False,
                        layer_idx=target_layer,
                        kv_importance_tracker=self.kv_importance if self.is_decoder else None,
                    )
                self.block_op[target_layer] += 1
                self._missing_kv_accounting_obj().record_exact_catchup_executed(1)
                executed_exact_units += 1
                event["exact_catchup_executed_token_layer_units"] = executed_exact_units
                if overhead_event is not None and _overhead_timing_handle is not None:
                    try:
                        overhead_event.commit_target_executed(_overhead_timing_handle)
                    except Exception:
                        pass
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
                reference_hidden, exact_state = layer_outputs[:2]
                reference_position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    reference_encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
                record["exact_reference_status"] = "ok"
                event["failure_stage"] = "after_exact_reference_layer"
                self._call_calm_phase3c_transaction_test_hook(
                    "after_exact_reference_layer",
                    event,
                    target_layer=target_layer,
                    exact_state=exact_state,
                    executed_exact_units=executed_exact_units,
                )
                if exact_state is None or len(exact_state) < 2:
                    raise ValueError("target_state_missing_self_attention_kv:{}".format(target_layer))
                exact_key, exact_value = exact_state[:2]
                if exact_key is None or exact_value is None:
                    raise ValueError("target_self_attention_kv_none:{}".format(target_layer))
                if int(exact_key.shape[0]) != 1 or int(exact_value.shape[0]) != 1:
                    raise ValueError("calm_phase3c_batch_size_must_be_one")
                expected_seq_len = past_len + 1
                if int(exact_key.shape[2]) != expected_seq_len or int(exact_value.shape[2]) != expected_seq_len:
                    raise ValueError(
                        "target_cache_sequence_length_mismatch: layer={} expected={} key={} value={}".format(
                            target_layer,
                            expected_seq_len,
                            int(exact_key.shape[2]),
                            int(exact_value.shape[2]),
                        )
                    )
                record["target_key_shape"] = list(exact_key.shape)
                record["target_value_shape"] = list(exact_value.shape)
                if is_exact_catchup:
                    record.update(
                        {
                            "status": "exact_cache_ready",
                            "restoration_status": "not_requested",
                            "cache_staging_status": "not_applicable_exact_cache",
                            "candidate_cache_write_status": "not_applicable_exact_cache",
                            "cache_commit_status": "exact_cache_ready",
                            "transaction_cache_install_status": "pending",
                            "final_status": "exact_cache_ready",
                        }
                    )
                    candidate_target_states.append(exact_state)
                    continue
                event["restoration_stage_entered"] = True
                target_self_attention = layer_module.layer[0]
                event["failure_stage"] = "restore_from_hidden"
                with self._missing_kv_component_timer_obj().time_block(
                    "restoration_compute_time_ms",
                    device=exact_key.device,
                ):
                    if restoration_method == DIRECT_SHALLOW_KV_REUSE_METHOD:
                        event["failure_stage"] = "direct_source_kv_slice"
                        direct_source_kv_layer = source_layer - 1
                        record["direct_source_kv_layer"] = direct_source_kv_layer
                        if direct_source_kv_layer < 0 or len(prefix_states) <= direct_source_kv_layer:
                            raise ValueError("direct_source_kv_layer_unavailable:{}".format(direct_source_kv_layer))
                        direct_source_state = prefix_states[direct_source_kv_layer]
                        if direct_source_state is None or len(direct_source_state) < 2:
                            raise ValueError("direct_source_kv_state_missing:{}".format(direct_source_kv_layer))
                        direct_source_key, direct_source_value = direct_source_state[:2]
                        if direct_source_key is None or direct_source_value is None:
                            raise ValueError("direct_source_self_attention_kv_none:{}".format(direct_source_kv_layer))
                        if int(direct_source_key.shape[2]) < past_len + 1 or int(direct_source_value.shape[2]) < past_len + 1:
                            raise ValueError(
                                "direct_source_cache_sequence_length_mismatch: layer={} expected_at_least={} key={} value={}".format(
                                    direct_source_kv_layer,
                                    past_len + 1,
                                    int(direct_source_key.shape[2]),
                                    int(direct_source_value.shape[2]),
                                )
                            )
                        source_key_slice = direct_source_key[:, :, past_len:past_len + 1, :].to(
                            device=exact_key.device,
                            dtype=exact_key.dtype,
                        )
                        source_value_slice = direct_source_value[:, :, past_len:past_len + 1, :].to(
                            device=exact_value.device,
                            dtype=exact_value.dtype,
                        )
                        event["failure_stage"] = "direct_shallow_kv_reuse"
                        result = self.kv_runtime_restorer.restore(
                            source_key_slice,
                            source_value_slice,
                            direct_source_kv_layer,
                            target_layer,
                        )
                    elif restoration_method == EXIT_HIDDEN_TARGET_PROJECTION_METHOD:
                        event["failure_stage"] = "exit_hidden_target_projection"
                        result = self.kv_runtime_restorer.restore_from_hidden(
                            source_hidden,
                            source_layer,
                            target_layer,
                            target_self_attention.layer_norm,
                            target_self_attention.SelfAttention.k,
                            target_self_attention.SelfAttention.v,
                            output_device=exact_key.device,
                            output_dtype=exact_key.dtype,
                        )
                    else:
                        event["failure_stage"] = "restore_from_hidden"
                        result = self.kv_runtime_restorer.restore_from_hidden(
                            source_hidden,
                            source_layer,
                            target_layer,
                            target_self_attention.layer_norm,
                            target_self_attention.SelfAttention.k,
                            target_self_attention.SelfAttention.v,
                            output_device=exact_key.device,
                            output_dtype=exact_key.dtype,
                        )
                metadata = result.metadata or {}
                event["restoration_attempted_token_layer_units"] += 1
                record.update(
                    {
                        "status": result.status,
                        "restoration_status": result.status,
                        "restoration_submode": metadata.get("restoration_submode"),
                        "gap_bin": metadata.get("gap_bin"),
                        "regenerated_key_shape": metadata.get("regenerated_key_shape"),
                        "regenerated_value_shape": metadata.get("regenerated_value_shape"),
                        "restored_key_shape": metadata.get("restored_key_shape"),
                        "restored_value_shape": metadata.get("restored_value_shape"),
                        "final_status": None,
                    }
                )
                if result.status != "ok":
                    record["final_status"] = "failed"
                    record["status"] = "failed"
                    record["error_message"] = result.error_message
                    raise ValueError("{}_restore_failed:layer{}:{}".format(evidence_prefix, target_layer, result.status))
                event["failure_stage"] = "after_restoration"
                self._call_calm_phase3c_transaction_test_hook(
                    "after_restoration",
                    event,
                    target_layer=target_layer,
                    result=result,
                )
                event["failure_stage"] = "cache_staging"
                with self._missing_kv_component_timer_obj().time_block(
                    "cache_staging_time_ms",
                    device=exact_key.device,
                ):
                    staged_key, staged_value, staging_error, staging_message = self._stage_runtime_restored_kv_slices(
                        result,
                        exact_key,
                        exact_value,
                        past_len,
                    )
                if staging_error is not None:
                    record["cache_staging_status"] = "failed"
                    record["final_status"] = "failed"
                    record["status"] = "failed"
                    record["error_message"] = staging_message
                    raise ValueError("{}_cache_staging_failed:layer{}:{}".format(evidence_prefix, target_layer, staging_error))
                record["cache_staging_status"] = "ok"
                event["failure_stage"] = "after_staging"
                self._call_calm_phase3c_transaction_test_hook(
                    "after_staging",
                    event,
                    target_layer=target_layer,
                    staged_key=staged_key,
                    staged_value=staged_value,
                )
                event["failure_stage"] = "cache_commit"
                record["candidate_cache_write_status"] = "started"
                record["cache_commit_status"] = "local_candidate_write_started"
                event["approximate_overwrite_stage_entered"] = True
                with self._missing_kv_component_timer_obj().time_block(
                    "cache_commit_time_ms",
                    device=exact_key.device,
                ):
                    candidate_key, candidate_value = self._commit_runtime_restored_kv_slices(
                        exact_key,
                        exact_value,
                        past_len,
                        staged_key,
                        staged_value,
                    )
                record["candidate_cache_write_status"] = "ok"
                record["cache_commit_status"] = "local_candidate_write_ok"
                record["transaction_cache_install_status"] = "pending"
                record["final_status"] = "candidate_ready_not_committed"
                record["status"] = "candidate_ready_not_committed"
                candidate_target_states.append((candidate_key, candidate_value) + tuple(exact_state[2:]))
                event["failure_stage"] = "after_target_staged"
                self._call_calm_phase3c_transaction_test_hook(
                    "after_target_staged",
                    event,
                    target_layer=target_layer,
                    candidate_state=candidate_target_states[-1],
                )

            if calm_pure_event is not None:
                # The live Exact replay for this event has just completed
                # (exactly once); only now are the Ours and State shadows
                # measured on the same h_s (order: Exact -> Ours -> State).
                self._calm_pure_recovery_measure_shadows(
                    calm_pure_event, source_hidden, source_layer
                )

            event["failure_stage"] = "before_final_cache_list_construction"
            self._call_calm_phase3c_transaction_test_hook("before_final_cache_list_construction", event)
            complete_cache = prefix_states + candidate_target_states
            if len(complete_cache) != num_layers:
                raise ValueError("candidate_complete_cache_length_mismatch")
            event["failure_stage"] = "before_cache_install"
            self._call_calm_phase3c_transaction_test_hook("before_cache_install", event, complete_cache=complete_cache)
            accounting = self._missing_kv_accounting_obj()
            if is_exact_catchup:
                for record in event["phase3c_restore_records"]:
                    record["transaction_cache_install_status"] = "committed"
                    record["cache_commit_status"] = "exact_cache_committed"
                    record["final_status"] = "exact_cache_committed"
                    record["status"] = "exact_cache_committed"
                self._kv_runtime_restoration_counters["exact_catchup_computed_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["exact_cache_retained_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["exact_catchup_token_layer_units"] += requested_units
                event.update(
                    {
                        "transaction_status": "committed",
                        "actual_restoration_applied": False,
                        "restored_token_layer_units": 0,
                        "exact_catchup_token_layer_units": requested_units,
                        "exact_catchup_computed_token_layer_units": requested_units,
                        "exact_cache_written_token_layer_units": requested_units,
                        "exact_overwrite_token_layer_units": 0,
                        "exact_catchup_avoided_token_layer_units": 0,
                        "exact_reference_final_hidden_shape": list(reference_hidden.shape),
                        "returned_hidden_source": "{}_source_hidden_h_{}".format(evidence_prefix, source_layer),
                        "present_key_value_states_len_after": len(complete_cache),
                        "counters": dict(self._kv_runtime_restoration_counters),
                    }
                )
                accounting.record_exact_cache_written(requested_units)
                if calibration_collector is not None:
                    # Historical (candidate_first_crossing) callers always
                    # resolve to stage_first_crossing (the attribute default
                    # below matches its own prior unconditional call
                    # exactly). The official FREE CALM init path sets
                    # collector_stage_method_name="stage_official_free_calm_
                    # first_crossing" on the manager instance instead -- the
                    # transaction logic above is otherwise fully reused
                    # unchanged for both.
                    stage_method_name = getattr(
                        self.kv_runtime_restorer, "collector_stage_method_name", "stage_first_crossing"
                    )
                    getattr(calibration_collector, stage_method_name)(
                        sample_context=self._missing_kv_sample_context_fields(),
                        generation_index=self._generation_index,
                        decoder_position=decoder_position,
                        source_layer=source_layer,
                        confidence=confidence,
                        threshold=float(getattr(self.kv_runtime_restorer, "candidate_threshold", CALM_THRESHOLD)),
                        candidate_evaluations=candidate_evaluations,
                        hidden_by_layer=calibration_hidden_by_layer,
                        complete_cache=complete_cache,
                        target_records=event["phase3c_restore_records"],
                        source_selected_token_id=int(exit_logits.detach().argmax(dim=-1).reshape(-1)[0].item()),
                    )
            else:
                for record in event["phase3c_restore_records"]:
                    record["transaction_cache_install_status"] = "committed"
                    record["final_status"] = "restored_and_committed"
                    record["status"] = "restored_and_committed"
                accounting.record_restoration_flush(enabled=True, restore_relative_count=requested_units)
                accounting.record_restoration_layer(
                    requested=requested_units,
                    succeeded=requested_units,
                    overwritten=requested_units,
                    fallback=0,
                    token_records=requested_units,
                )
                self._kv_runtime_restoration_counters["calm_phase3c_requested_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["calm_phase3c_succeeded_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["calm_phase3c_overwritten_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["phase3c_restore_success_count"] += requested_units
                self._kv_runtime_restoration_counters["exact_overwrite_count"] += requested_units
                self._kv_runtime_restoration_counters["restored_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["exact_catchup_computed_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["exact_overwrite_token_layer_units"] += requested_units
                self._kv_runtime_restoration_counters["exact_catchup_token_layer_units"] += requested_units
                event.update(
                    {
                        "transaction_status": "committed",
                        "actual_restoration_applied": True,
                        "restored_token_layer_units": requested_units,
                        "exact_catchup_token_layer_units": requested_units,
                        "exact_catchup_computed_token_layer_units": requested_units,
                        "exact_overwrite_token_layer_units": requested_units,
                        "exact_catchup_avoided_token_layer_units": 0,
                        "exact_reference_final_hidden_shape": list(reference_hidden.shape),
                        "returned_hidden_source": "{}_source_hidden_h_{}".format(evidence_prefix, source_layer),
                        "present_key_value_states_len_after": len(complete_cache),
                        "counters": dict(self._kv_runtime_restoration_counters),
                    }
                )
            event.pop("failure_stage", None)
            if calm_pure_event is not None:
                # Explicit live-transaction commit gate: paper validity
                # requires the surrounding live Exact transaction to have
                # actually committed. Any failure path leaves this False.
                calm_pure_event["live_exact_transaction_committed"] = True
            self.lm_logits = exit_logits
            if overhead_event is not None:
                try:
                    overhead_event.commit()
                except Exception as exc:
                    # commit() rejects an executed != required invariant
                    # violation rather than silently normalizing it -- record
                    # it as an explicit failed transaction (never as a silent
                    # incomplete registration) so the reason is visible in
                    # the recorder's diagnostics.
                    try:
                        overhead_event.fail(stage="commit_invariant_violation", reason=str(exc))
                    except Exception:
                        pass
                try:
                    overhead_recorder.finalize(overhead_event)
                except Exception:
                    pass
            if self.kv_trace.enabled:
                self.kv_trace.record("kv_calm_phase3c_taskc1_transaction", **event)
            return source_hidden, complete_cache, event
        except Exception as exc:
            _record_failure(event.get("failure_stage", "transaction"), exc)
            raise

    def _apply_runtime_restoration_to_present_kv(
        self,
        layer_idx,
        present_key_value_state,
        past_key_value,
        pending_skipped_tokens,
        source_records,
        plan,
        layer_input_seq_len,
        pending_source_hidden_states=None,
        phase3c_source_hidden_records=None,
    ):
        is_phase3c = self._phase3c_runtime_restoration_enabled()
        event = {
            "enabled": bool(plan.get("enabled")),
            "runtime_restoration_flush_id": plan.get("runtime_restoration_flush_id"),
            "layer_idx": layer_idx,
            "target_layer": layer_idx,
            "pending_skipped_tokens": int(pending_skipped_tokens or 0),
            "restoration_method": plan.get("restoration_method"),
            "threshold": plan.get("threshold"),
            "runtime_mode": plan.get("runtime_mode"),
            "exact_catchup_already_computed": True,
            "speed_claim_valid": False,
            "calibration_source_mode": getattr(self.kv_runtime_restorer, "calibration_source_mode", None),
            "calibration_fixed_source_layer": getattr(self.kv_runtime_restorer, "calibration_fixed_source_layer", None),
            "runtime_source_mode": getattr(self.kv_runtime_restorer, "runtime_source_mode", None),
            "runtime_fixed_source_layer": getattr(self.kv_runtime_restorer, "runtime_fixed_source_layer", None),
            "calibration_layer_scope_match": getattr(
                self.kv_runtime_restorer,
                "calibration_layer_scope_match",
                None,
            ),
            "calibration_runtime_scope_match": getattr(
                self.kv_runtime_restorer,
                "calibration_runtime_scope_match",
                None,
            ),
            "confidence_semantics_verified": (
                getattr(self.kv_runtime_restorer, "confidence_semantics_verified", None)
                if is_phase3c
                else None
            ),
            "fixed_layer_membership_semantics_verified": (
                getattr(self.kv_runtime_restorer, "fixed_layer_membership_semantics_verified", None)
                if is_phase3c
                else None
            ),
            "generation_quality_claim_valid": (
                bool(getattr(self.kv_runtime_restorer, "generation_quality_claim_valid", False))
                if is_phase3c
                else None
            ),
            "recent_exact_window": plan.get("recent_exact_window"),
            "restore_relative_indices": list(plan.get("restore_relative_indices") or []),
            "exact_relative_indices": list(plan.get("exact_relative_indices") or []),
            "restored_relative_indices": [],
            "fallback_exact_relative_indices": [],
            "missing_map_relative_indices": [],
            "missing_hidden_pair_relative_indices": [],
            "missing_k_pair_relative_indices": [],
            "missing_v_gap_relative_indices": [],
            "missing_threshold_relative_indices": [],
            "cache_shape_mismatch_relative_indices": [],
            "nan_or_inf_relative_indices": [],
            "unexpected_runtime_error_relative_indices": [],
            "phase3c_restore_records": [],
            "actual_restoration_applied": False,
            "copy_skipped_hidden_states": self.config.copy_skipped_hidden_states,
        }
        finalized = False

        def _set_phase3c_final_status(record, restoration_status, staging_status, commit_status, final_status, fallback_reason=None, error_message=None):
            if not isinstance(record, dict):
                return
            record["restoration_status"] = restoration_status
            record["cache_staging_status"] = staging_status
            record["cache_commit_status"] = commit_status
            record["final_status"] = final_status
            record["status"] = final_status
            record["fallback_reason"] = fallback_reason
            if error_message is not None:
                record["error_message"] = error_message

        def _append_phase3c_record_once(record):
            if isinstance(record, dict) and record not in event["phase3c_restore_records"]:
                event["phase3c_restore_records"].append(record)

        def _finish(return_state):
            nonlocal finalized
            if finalized:
                return return_state, event
            finalized = True
            restored_units = len(event["restored_relative_indices"])
            fallback_units = len(event["fallback_exact_relative_indices"])
            missing_units = len(event["missing_map_relative_indices"])
            computed_units = int(pending_skipped_tokens or 0)
            exact_overwrite_units = restored_units
            exact_retained_units = max(0, computed_units - exact_overwrite_units)
            exact_avoided_units = 0
            exact_units = computed_units
            requested_units = len(plan.get("restore_relative_indices") or []) if plan.get("enabled") else 0
            self._missing_kv_accounting_obj().record_restoration_layer(
                requested=requested_units,
                succeeded=restored_units,
                overwritten=exact_overwrite_units,
                fallback=fallback_units,
                token_records=len(event.get("phase3c_restore_records") or []),
            )
            self._kv_runtime_restoration_counters["restored_token_layer_units"] += restored_units
            self._kv_runtime_restoration_counters["exact_catchup_computed_token_layer_units"] += computed_units
            self._kv_runtime_restoration_counters["exact_cache_retained_token_layer_units"] += exact_retained_units
            self._kv_runtime_restoration_counters["exact_overwrite_token_layer_units"] += exact_overwrite_units
            self._kv_runtime_restoration_counters["exact_catchup_avoided_token_layer_units"] += exact_avoided_units
            self._kv_runtime_restoration_counters["exact_catchup_token_layer_units"] += exact_units
            self._kv_runtime_restoration_counters["fallback_exact_count"] += fallback_units
            event.update(
                {
                    "restored_token_layer_units": restored_units,
                    "exact_catchup_token_layer_units": exact_units,
                    "exact_catchup_computed_token_layer_units": computed_units,
                    "exact_cache_retained_token_layer_units": exact_retained_units,
                    "exact_overwrite_token_layer_units": exact_overwrite_units,
                    "exact_catchup_avoided_token_layer_units": exact_avoided_units,
                    "fallback_exact_count": fallback_units,
                    "missing_map_count": missing_units,
                    "missing_threshold_count": len(event["missing_threshold_relative_indices"]),
                    "missing_hidden_pair_count": len(event["missing_hidden_pair_relative_indices"]),
                    "missing_k_pair_count": len(event["missing_k_pair_relative_indices"]),
                    "missing_v_gap_count": len(event["missing_v_gap_relative_indices"]),
                    "cache_shape_mismatch_count": len(event["cache_shape_mismatch_relative_indices"]),
                    "nan_or_inf_count": len(event["nan_or_inf_relative_indices"]),
                    "unexpected_runtime_error_count": len(event["unexpected_runtime_error_relative_indices"]),
                    "actual_restoration_applied": restored_units > 0,
                    "counters": dict(self._kv_runtime_restoration_counters),
                }
            )
            return return_state, event

        if not plan.get("enabled"):
            event["skip_reason"] = plan.get("reason")
            return _finish(present_key_value_state)
        if present_key_value_state is None or len(present_key_value_state) < 2:
            event["skip_reason"] = "present_key_value_state_unavailable"
            return _finish(present_key_value_state)
        key_tensor = present_key_value_state[0]
        value_tensor = present_key_value_state[1]
        if key_tensor is None or value_tensor is None:
            event["skip_reason"] = "self_attention_kv_unavailable"
            return _finish(present_key_value_state)
        if key_tensor.shape[0] != 1:
            self._kv_runtime_restoration_counters["unsupported_batch_count"] += 1
            raise NotImplementedError("runtime restored-K/V currently supports batch size 1 only")

        past_len = safe_cache_seq_len(past_key_value)
        past_len = 0 if past_len is None else int(past_len)
        event["past_key_value_len"] = past_len
        event["present_key_shape_before"] = list(key_tensor.shape)
        event["present_value_shape_before"] = list(value_tensor.shape)
        event["output_cache_dtype"] = str(key_tensor.dtype).replace("torch.", "")
        event["output_cache_device"] = str(key_tensor.device)
        event["layer_input_seq_len"] = int(layer_input_seq_len)

        restored_key_tensor = key_tensor.clone()
        restored_value_tensor = value_tensor.clone()
        for relative_idx in list(plan.get("restore_relative_indices") or []):
            relative_idx = int(relative_idx)
            source_record = None
            phase3c_source_record = None
            phase3c_record = None
            source_layer = None
            cache_position = None
            try:
                cache_position = past_len + relative_idx
                source_load_source = None
                if is_phase3c:
                    self._kv_runtime_restoration_counters["phase3c_restore_attempt_count"] += 1
                    phase3c_source_record = (
                        phase3c_source_hidden_records[relative_idx]
                        if phase3c_source_hidden_records is not None and relative_idx < len(phase3c_source_hidden_records)
                        else None
                    )
                    if isinstance(phase3c_source_record, dict):
                        source_layer = phase3c_source_record.get("phase3c_source_hidden_layer")
                    source_layer = int(source_layer) if source_layer is not None else None
                    phase3c_record = {
                        "runtime_restoration_flush_id": plan.get("runtime_restoration_flush_id"),
                        "relative_index": relative_idx,
                        "threshold": plan.get("threshold"),
                        "source_layer": source_layer,
                        "target_layer": int(layer_idx),
                        "gap": int(layer_idx) - int(source_layer) if source_layer is not None else None,
                        "gap_bin": None,
                        "cache_position": cache_position,
                        "past_key_value_len": past_len,
                        "source_hidden_shape": None,
                        "regenerated_key_shape": None,
                        "regenerated_value_shape": None,
                        "restored_key_shape": None,
                        "restored_value_shape": None,
                        "status": None,
                        "restoration_status": None,
                        "cache_staging_status": "not_attempted",
                        "cache_commit_status": "not_attempted",
                        "final_status": None,
                        "fallback_reason": None,
                        "phase3c_source_hidden_record": self._phase3c_source_hidden_record_trace(phase3c_source_record),
                    }
                if cache_position < 0 or cache_position >= int(restored_key_tensor.shape[2]):
                    if is_phase3c:
                        message = (
                            "cache_position_out_of_range: cache_position={} cache_seq_len={}".format(
                                cache_position,
                                int(restored_key_tensor.shape[2]),
                            )
                        )
                        _set_phase3c_final_status(
                            phase3c_record,
                            "not_attempted",
                            "not_attempted",
                            "not_attempted",
                            "fallback_exact",
                            "cache_position_out_of_range",
                            message,
                        )
                        _append_phase3c_record_once(phase3c_record)
                    event["cache_shape_mismatch_relative_indices"].append(relative_idx)
                    event["fallback_exact_relative_indices"].append(relative_idx)
                    event.setdefault("fallback_error_messages", []).append(
                        "cache_position_out_of_range: relative_idx={} cache_position={} cache_seq_len={}".format(
                            relative_idx,
                            cache_position,
                            int(restored_key_tensor.shape[2]),
                        )
                    )
                    self._kv_runtime_restoration_counters["cache_shape_mismatch_count"] += 1
                    continue

                if is_phase3c:
                    if source_layer is None:
                        _set_phase3c_final_status(
                            phase3c_record,
                            "phase3c_source_hidden_layer_missing",
                            "not_attempted",
                            "not_attempted",
                            "fallback_exact",
                            "phase3c_source_hidden_layer_missing",
                        )
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        _append_phase3c_record_once(phase3c_record)
                        continue
                    if int(layer_idx) < int(source_layer):
                        _set_phase3c_final_status(
                            phase3c_record,
                            "target_before_phase3c_source_hidden_layer",
                            "not_attempted",
                            "not_attempted",
                            "fallback_exact",
                            "target_before_phase3c_source_hidden_layer",
                        )
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        _append_phase3c_record_once(phase3c_record)
                        continue
                    if not isinstance(phase3c_source_record, dict) or phase3c_source_record.get("raw_hidden") is None:
                        _set_phase3c_final_status(
                            phase3c_record,
                            "source_hidden_missing",
                            "not_attempted",
                            "not_attempted",
                            "fallback_exact",
                            "source_hidden_missing",
                        )
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        _append_phase3c_record_once(phase3c_record)
                        continue
                    source_hidden = phase3c_source_record["raw_hidden"]
                    phase3c_record["source_hidden_shape"] = list(source_hidden.shape)
                    event.setdefault("source_hidden_shapes", []).append(list(source_hidden.shape))
                    event.setdefault("phase3c_source_hidden_layers", []).append(int(source_layer))
                    event.setdefault("phase3c_source_hidden_records", []).append(
                        self._phase3c_source_hidden_record_trace(phase3c_source_record)
                    )
                    target_self_attention = self.block[int(layer_idx)].layer[0]
                    with self._missing_kv_component_timer_obj().time_block(
                        "restoration_compute_time_ms",
                        device=key_tensor.device,
                    ):
                        result = self.kv_runtime_restorer.restore_from_hidden(
                            source_hidden,
                            source_layer,
                            int(layer_idx),
                            target_self_attention.layer_norm,
                            target_self_attention.SelfAttention.k,
                            target_self_attention.SelfAttention.v,
                            output_device=key_tensor.device,
                            output_dtype=key_tensor.dtype,
                        )
                    metadata = result.metadata or {}
                    phase3c_record.update(
                        {
                            "gap_bin": metadata.get("gap_bin"),
                            "regenerated_key_shape": metadata.get("regenerated_key_shape"),
                            "regenerated_value_shape": metadata.get("regenerated_value_shape"),
                            "restored_key_shape": metadata.get("restored_key_shape"),
                            "restored_value_shape": metadata.get("restored_value_shape"),
                            "status": result.status,
                            "fallback_reason": None if result.status == "ok" else result.status,
                            "error_message": result.error_message,
                            "restoration_submode": metadata.get("restoration_submode"),
                            "hidden_affine_applied": metadata.get("hidden_affine_applied"),
                            "k_correction_applied": metadata.get("k_correction_applied"),
                            "v_correction_applied": metadata.get("v_correction_applied"),
                            "calibration_source_mode": metadata.get("calibration_source_mode"),
                            "calibration_fixed_source_layer": metadata.get("calibration_fixed_source_layer"),
                            "runtime_source_mode": metadata.get("runtime_source_mode"),
                            "runtime_fixed_source_layer": metadata.get("runtime_fixed_source_layer"),
                            "calibration_layer_scope_match": metadata.get("calibration_layer_scope_match"),
                            "calibration_runtime_scope_match": metadata.get("calibration_runtime_scope_match"),
                            "confidence_semantics_verified": metadata.get("confidence_semantics_verified"),
                            "fixed_layer_membership_semantics_verified": metadata.get("fixed_layer_membership_semantics_verified"),
                            "generation_quality_claim_valid": metadata.get("generation_quality_claim_valid"),
                            "runtime_mode": metadata.get("runtime_mode", PHASE3C_RUNTIME_MODE),
                            "exact_catchup_already_computed": True,
                            "speed_claim_valid": False,
                        }
                    )
                    event.setdefault("phase3c_gap_bins", []).append(metadata.get("gap_bin"))
                    event.setdefault("regenerated_key_shapes", []).append(metadata.get("regenerated_key_shape"))
                    event.setdefault("regenerated_value_shapes", []).append(metadata.get("regenerated_value_shape"))
                    event.setdefault("restored_key_shapes", []).append(metadata.get("restored_key_shape"))
                    event.setdefault("restored_value_shapes", []).append(metadata.get("restored_value_shape"))
                    if result.status != "ok":
                        _set_phase3c_final_status(
                            phase3c_record,
                            result.status,
                            "not_attempted",
                            "not_attempted",
                            "fallback_exact",
                            result.status,
                            result.error_message,
                        )
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        if result.status == "missing_threshold":
                            event["missing_threshold_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["missing_threshold_count"] += 1
                        elif result.status == "missing_hidden_pair":
                            event["missing_hidden_pair_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["missing_hidden_pair_count"] += 1
                        elif result.status == "missing_k_pair":
                            event["missing_k_pair_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["missing_k_pair_count"] += 1
                        elif result.status == "missing_v_gap":
                            event["missing_v_gap_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["missing_v_gap_count"] += 1
                        elif result.status == "unsupported_batch":
                            self._kv_runtime_restoration_counters["unsupported_batch_count"] += 1
                        elif result.status == "nan_or_inf" or _is_nonfinite_error_message(result.error_message):
                            event["nan_or_inf_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["nan_or_inf_count"] += 1
                        _append_phase3c_record_once(phase3c_record)
                        continue
                else:
                    source_record = source_records[relative_idx] if relative_idx < len(source_records) else None
                    if isinstance(source_record, dict) and source_record.get("source_layer") is not None:
                        source_layer = int(source_record.get("source_layer"))
                    if source_layer is None:
                        raise ValueError("source_layer_missing")
                    if int(layer_idx) <= source_layer:
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        continue
                    if not self.kv_runtime_restorer.has_pair(source_layer, int(layer_idx)):
                        event["missing_map_relative_indices"].append(relative_idx)
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        self._kv_runtime_restoration_counters["missing_map_count"] += 1
                        continue
                    source_key, source_value, source_load_source = self._load_runtime_source_kv_from_record(source_record)
                    source_key = source_key.to(device=key_tensor.device, dtype=key_tensor.dtype)
                    source_value = source_value.to(device=value_tensor.device, dtype=value_tensor.dtype)
                    result = self.kv_runtime_restorer.restore(source_key, source_value, source_layer, int(layer_idx))
                    if result.status == "missing_map":
                        event["missing_map_relative_indices"].append(relative_idx)
                        self._kv_runtime_restoration_counters["missing_map_count"] += 1
                        continue
                    if result.status != "ok":
                        event["fallback_exact_relative_indices"].append(relative_idx)
                        if result.status == "nan_or_inf" or _is_nonfinite_error_message(result.error_message):
                            event["nan_or_inf_relative_indices"].append(relative_idx)
                            self._kv_runtime_restoration_counters["nan_or_inf_count"] += 1
                        continue

                with self._missing_kv_component_timer_obj().time_block(
                    "cache_staging_time_ms",
                    device=key_tensor.device,
                ):
                    staged_key, staged_value, staging_error, staging_message = self._stage_runtime_restored_kv_slices(
                        result,
                        restored_key_tensor,
                        restored_value_tensor,
                        cache_position,
                    )
                if staging_error is not None:
                    if is_phase3c:
                        _set_phase3c_final_status(
                            phase3c_record,
                            "ok",
                            "failed",
                            "not_attempted",
                            "fallback_exact",
                            staging_error,
                            staging_message,
                        )
                        _append_phase3c_record_once(phase3c_record)
                    event["fallback_exact_relative_indices"].append(relative_idx)
                    event.setdefault("fallback_error_messages", []).append(staging_message)
                    if staging_error == "nan_or_inf":
                        event["nan_or_inf_relative_indices"].append(relative_idx)
                        self._kv_runtime_restoration_counters["nan_or_inf_count"] += 1
                    else:
                        event["cache_shape_mismatch_relative_indices"].append(relative_idx)
                        self._kv_runtime_restoration_counters["cache_shape_mismatch_count"] += 1
                    continue
                try:
                    with self._missing_kv_component_timer_obj().time_block(
                        "cache_commit_time_ms",
                        device=key_tensor.device,
                    ):
                        restored_key_tensor, restored_value_tensor = self._commit_runtime_restored_kv_slices(
                            restored_key_tensor,
                            restored_value_tensor,
                            cache_position,
                            staged_key,
                            staged_value,
                        )
                except Exception as exc:
                    if is_phase3c:
                        _set_phase3c_final_status(
                            phase3c_record,
                            "ok",
                            "ok",
                            "failed",
                            "fallback_exact",
                            "atomic_cache_commit_failed",
                            str(exc),
                        )
                        _append_phase3c_record_once(phase3c_record)
                    event["cache_shape_mismatch_relative_indices"].append(relative_idx)
                    event["fallback_exact_relative_indices"].append(relative_idx)
                    event.setdefault("fallback_error_messages", []).append(
                        "atomic restored K/V commit failed: {}".format(exc)
                    )
                    self._kv_runtime_restoration_counters["cache_shape_mismatch_count"] += 1
                    continue
                if is_phase3c:
                    _set_phase3c_final_status(
                        phase3c_record,
                        "ok",
                        "ok",
                        "ok",
                        "restored",
                    )
                    _append_phase3c_record_once(phase3c_record)
                event["restored_relative_indices"].append(relative_idx)
                event.setdefault("restored_key_positions", []).append(cache_position)
                event.setdefault("source_layers", []).append(source_layer)
                event.setdefault("source_load_sources", []).append(source_load_source or "source_hidden")
                if is_phase3c:
                    self._kv_runtime_restoration_counters["phase3c_restore_success_count"] += 1
                    self._kv_runtime_restoration_counters["exact_overwrite_count"] += 1
            except Exception as exc:
                if is_phase3c:
                    if not isinstance(phase3c_record, dict):
                        phase3c_record = {
                            "runtime_restoration_flush_id": plan.get("runtime_restoration_flush_id"),
                            "relative_index": relative_idx,
                            "threshold": plan.get("threshold"),
                            "source_layer": source_layer,
                            "target_layer": int(layer_idx),
                            "gap": int(layer_idx) - int(source_layer) if source_layer is not None else None,
                            "gap_bin": None,
                            "cache_position": cache_position,
                            "source_hidden_shape": None,
                            "regenerated_key_shape": None,
                            "regenerated_value_shape": None,
                            "restored_key_shape": None,
                            "restored_value_shape": None,
                            "status": None,
                            "restoration_status": None,
                            "cache_staging_status": "not_attempted",
                            "cache_commit_status": "not_attempted",
                            "final_status": None,
                            "fallback_reason": None,
                            "phase3c_source_hidden_record": self._phase3c_source_hidden_record_trace(phase3c_source_record),
                        }
                    _set_phase3c_final_status(
                        phase3c_record,
                        "unexpected_runtime_error",
                        (
                            phase3c_record.get("cache_staging_status")
                            if phase3c_record.get("cache_staging_status") not in {None, "not_attempted"}
                            else "not_completed"
                        ),
                        phase3c_record.get("cache_commit_status") or "not_attempted",
                        "fallback_exact",
                        "unexpected_runtime_exception",
                        str(exc),
                    )
                    _append_phase3c_record_once(phase3c_record)
                    event["unexpected_runtime_error_relative_indices"].append(relative_idx)
                    self._kv_runtime_restoration_counters["unexpected_runtime_error_count"] += 1
                event["fallback_exact_relative_indices"].append(relative_idx)
                event.setdefault("fallback_error_messages", []).append(str(exc))

        new_present = (restored_key_tensor, restored_value_tensor) + tuple(present_key_value_state[2:])
        return _finish(new_present)

    def _task_c2_direct_insertion_enabled(self):
        """Task C2: Native FREE fixed-source-layer-6 direct K/V insertion.
        A narrow opt-in execution flag layered on the existing
        phase3c_kv_final restorer -- never a new restoration method. Full
        fail-closed configuration validation happens once, in
        util/additional_args.py's update_autoconfig(); this only reads the
        resulting flag plus the already-validated restorer/runtime state."""

        return (
            self.is_decoder
            and self.use_shallow_deep
            and not self.use_early_exit
            and bool(getattr(self.config, "kv_runtime_restoration_direct_insertion_enabled", False))
            and self._phase3c_runtime_restoration_enabled()
        )

    def prepare_phase3c_runtime_for_inference(self):
        """One-time Phase-3c INFERENCE SETUP for the fixed-source-6 stacked
        runtime: materialize the device-local fitted parameters and build +
        finite-validate the stacked learned-target bank NOW, outside any
        generation timing, so the first timed generate() reuses fully
        prepared, already-validated immutable state (no first-flush H2D
        transfers, no first-flush bank construction, no first-flush
        parameter finite scans).

        This is model/runtime setup, NOT a warm-up: it performs NO
        restoration arithmetic, computes NO restored K/V, runs NO forward
        pass, and touches NO cache. It only prepares immutable state through
        the SAME _stacked_learned_target_bank / _device_local_parameters
        infrastructure the runtime already uses -- never a duplicate build
        path. Idempotent: a second call finds the validated bank fingerprint
        unchanged and rebuilds nothing.

        It deliberately does NOT arm the timing-mode fast-reuse slot: the
        first real flush still takes the full validated lookup once, so any
        module/device/dtype change after preparation is caught by the
        existing fingerprint/device safety mechanisms exactly as before.

        Returns a small status dict for logging/tests."""

        if not (self.is_decoder and self._runtime_restoration_enabled()):
            return {"prepared": False, "reason": "runtime_restoration_disabled"}
        # SCOPE: this preparation exists ONLY for the Native FREE
        # fixed-source Batched Task C2 stacked runtime. The eligibility
        # predicate is the SAME one the runtime itself uses -- never a
        # second configuration check. Immediate/Task-C1/CALM-style Phase-3c
        # configurations are harmless no-ops here.
        if not self._task_c2_batched_insertion_enabled():
            return {"prepared": False, "reason": "batched_task_c2_disabled"}
        restorer = self.kv_runtime_restorer
        source_layer = int(self.shallow_exit_layer)
        learned_target_layers = list(range(source_layer + 1, len(self.block)))
        if not learned_target_layers:
            return {"prepared": False, "reason": "no_learned_target_layers"}
        norm_list = [self.block[layer].layer[0].layer_norm for layer in learned_target_layers]
        key_list = [self.block[layer].layer[0].SelfAttention.k for layer in learned_target_layers]
        value_list = [self.block[layer].layer[0].SelfAttention.v for layer in learned_target_layers]
        support = restorer._stacked_learned_module_support(norm_list, key_list, value_list)
        if support is None:
            # Synthetic/non-T5 fixtures: the runtime would use its sequential
            # compatibility path; nothing to pre-materialize.
            return {"prepared": False, "reason": "stacked_modules_unsupported"}
        module_device, module_dtype, variance_epsilon = support
        restorer._stacked_learned_target_bank(
            source_layer,
            learned_target_layers,
            norm_list,
            key_list,
            value_list,
            module_device,
            module_dtype,
            variance_epsilon,
            # Preparation ALWAYS validates: the bank enters the timed phase
            # already finite-validated, which is scientifically cleaner than
            # skipping validation -- and a later validation-ON caller finds a
            # genuinely validated bank, never silently trusts an unvalidated
            # one.
            validate_fitted_parameters_finite=True,
        )
        return {
            "prepared": True,
            "source_layer": source_layer,
            "learned_target_layers": learned_target_layers,
            "module_device": str(module_device),
            "module_dtype": str(module_dtype),
        }

    # ------------------------------------------------------------------
    # PURE_MISSING_KV_RECOVERY_COMPUTE_COST -- measurement-only shadow
    # instrumentation over the ordinary FREE Exact reference trajectory.
    # At each real flush with P>0 pending early-exit tokens it times, on the
    # SAME pending source hidden states: (a) a pending-token-only exact deep
    # replay (FREE pure recovery) and (b) the frozen Phase-3c stacked
    # restoration (Ours pure recovery). Both are shadows: results are
    # discarded (or compared for parity when validation is enabled) and
    # NEVER published; the live generation is untouched. Explicitly
    # excluded from both timers: the current non-exit token's normal deep
    # computation, cache staging/commit/publication, mask/position-bias
    # bookkeeping, cross-attention seeding, LM-head/confidence work, and
    # parameter-bank materialization (prepared once, untimed).
    # ------------------------------------------------------------------

    _PURE_RECOVERY_FREE_KEY = "pure_recovery_free_exact_replay_ms"
    _PURE_RECOVERY_OURS_KEY = "pure_recovery_ours_phase3c_ms"

    def _pure_recovery_cost_enabled(self):
        return (
            self.is_decoder
            and bool(getattr(self.config, "kv_pure_recovery_cost_enabled", False))
            and self.use_shallow_deep
            and not self.use_early_exit
            and not bool(self.config.copy_skipped_hidden_states)
        )

    def _pure_recovery_timer_obj(self):
        """Dedicated MissingKVComponentTimer instance (existing class; the
        CUDA-event backend defers every resolution to one finalize()). Kept
        separate from the production component timer so the measurement
        works without enabling the whole component-timing family."""

        if "pure_recovery_component_timer" not in self.__dict__:
            self.__dict__["pure_recovery_component_timer"] = MissingKVComponentTimer(
                enabled=True, backend="auto"
            )
        return self.__dict__["pure_recovery_component_timer"]

    def _pure_recovery_events_obj(self):
        if not hasattr(self, "_pure_recovery_events"):
            self._pure_recovery_events = []
        return self._pure_recovery_events

    def _pure_recovery_shadow_restorer_obj(self):
        """Shadow RuntimeKVRestorationManager built from the SAME frozen
        artifact/method/threshold arguments, used ONLY by this measurement.
        The live config keeps kv_runtime_restoration_enabled=False, so the
        production restoration paths never see it.

        The approved artifact SHA-256 is REQUIRED and verified against the
        actual file BEFORE deserialization -- the paper metric must be bound
        to the exact frozen artifact (SAMSum and CNN/DailyMail use different
        ones), never a post-load informational check."""

        if "pure_recovery_shadow_restorer" not in self.__dict__:
            from our_kv_restoration.missing_kv_dump_provenance import sha256_file
            from our_kv_restoration.runtime_kv_restoration import RuntimeKVRestorationManager

            artifact_path = getattr(self.config, "kv_runtime_restoration_artifact", None)
            expected_sha = getattr(self.config, "kv_runtime_restoration_artifact_sha256", None)
            if not artifact_path:
                raise ValueError("pure_recovery_artifact_path_missing")
            expected_sha = str(expected_sha or "").strip()
            if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
                raise ValueError(
                    "pure_recovery_artifact_sha256_missing_or_invalid: "
                    "kv_runtime_restoration_artifact_sha256 must be lowercase 64-hex, got {!r}".format(
                        expected_sha
                    )
                )
            actual_sha = sha256_file(artifact_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    "pure_recovery_artifact_sha256_mismatch: expected={} actual={} path={}".format(
                        expected_sha, actual_sha, artifact_path
                    )
                )
            self.__dict__["pure_recovery_artifact_binding"] = {
                "artifact_path": str(artifact_path),
                "expected_artifact_sha256": expected_sha,
                "actual_artifact_sha256": actual_sha,
                "verified_before_load": True,
            }
            self.__dict__["pure_recovery_shadow_restorer"] = RuntimeKVRestorationManager.from_path(
                artifact_path,
                getattr(self.config, "kv_runtime_restoration_method", "phase3c_kv_final"),
                threshold=getattr(self.config, "kv_runtime_restoration_threshold", None),
                model_config=self.config,
            )
        return self.__dict__["pure_recovery_shadow_restorer"]

    def _pure_recovery_prepare_shadow(self):
        """One-time, UNTIMED preparation of the shadow restorer's immutable
        state (device-local fitted parameters + validated stacked bank) via
        the same infrastructure production preparation uses. Bank
        construction is model/runtime setup and must never appear inside the
        per-event recovery timing."""

        if self.__dict__.get("pure_recovery_shadow_prepared"):
            return
        restorer = self._pure_recovery_shadow_restorer_obj()
        source_layer = int(self.shallow_exit_layer)
        learned_target_layers = list(range(source_layer + 1, len(self.block)))
        norm_list = [self.block[layer].layer[0].layer_norm for layer in learned_target_layers]
        key_list = [self.block[layer].layer[0].SelfAttention.k for layer in learned_target_layers]
        value_list = [self.block[layer].layer[0].SelfAttention.v for layer in learned_target_layers]
        support = restorer._stacked_learned_module_support(norm_list, key_list, value_list)
        if support is not None:
            module_device, module_dtype, variance_epsilon = support
            restorer._stacked_learned_target_bank(
                source_layer,
                learned_target_layers,
                norm_list,
                key_list,
                value_list,
                module_device,
                module_dtype,
                variance_epsilon,
                validate_fitted_parameters_finite=True,
            )
        self.__dict__["pure_recovery_shadow_prepared"] = True

    def _pure_recovery_prepare_exact_replay_inputs(
        self,
        pending_hidden,
        past_key_values,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
    ):
        """UNTIMED prepass for the pending-only exact replay: resolve every
        deep layer's past state (seeding cross K/V exactly like the
        production flush does when a layer has never run) and build the one
        shared causal mask + position bias for the pending-only span. This
        bookkeeping is deliberately outside the FREE pure-recovery timer,
        mirroring how the production flush computes it outside its own
        per-layer catch-up timing."""

        source_layer = int(self.shallow_exit_layer)
        pending_count = int(pending_hidden.shape[1])
        resolved_pasts = []
        for j in range(source_layer, len(self.block)):
            past_key_value = past_key_values[j]
            if past_key_value is None:
                past_key_value = self.block[j].gen_cross_attn_key_value(
                    pending_hidden,  # dummy, exactly like the production flush
                    attention_mask=None,
                    position_bias=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=head_mask[j] if head_mask is not None else None,
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[j] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            resolved_pasts.append(past_key_value)

        first_past = resolved_pasts[0]
        real_seq_length = pending_count
        if first_past[0] is not None:
            real_seq_length += int(first_past[0].shape[2])
        key_length = real_seq_length
        extended_attention_mask = None
        if self.config.parallel_causal_mask:
            attention_mask = torch.ones(
                pending_hidden.shape[0], real_seq_length, device=pending_hidden.device
            )
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, torch.Size([pending_hidden.shape[0], pending_count])
            )
        position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
            real_seq_length, key_length, device=pending_hidden.device
        )
        # Pending tokens occupy the LAST pending_count query positions --
        # identical to the production flush's slice for its own span.
        position_bias = position_bias[:, :, -pending_count:, :]
        if extended_attention_mask is not None:
            position_bias = position_bias + extended_attention_mask
        return {
            "source_layer": source_layer,
            "pending_count": pending_count,
            "resolved_pasts": resolved_pasts,
            "extended_attention_mask": extended_attention_mask,
            "position_bias": position_bias,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_extended_attention_mask": encoder_extended_attention_mask,
            "encoder_decoder_position_bias": encoder_decoder_position_bias,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
        }

    def _pure_recovery_run_pending_only_exact_replay(self, pending_hidden, prep, collect_spans):
        """TIMED body of the FREE pure-recovery measurement: exact deep
        replay of the PENDING TOKENS ONLY through blocks source..N-1,
        mirroring the validated production flush block-call pattern (same
        kwargs, same shared position bias, same present layout) minus the
        current non-exit token and minus every diagnostic/cache side effect.

        ``collect_spans`` is True only in parity-validation mode: per-layer
        (pending_key_span, pending_value_span) views of the shadow presents
        are then retained for the post-flush comparison. In the paper timing
        mode (validation OFF) NOTHING is retained -- each layer's present is
        dropped as soon as the next layer rebinds it, and the function
        returns None. The timed arithmetic is identical either way."""

        hidden_states = pending_hidden
        position_bias = prep["position_bias"]
        encoder_decoder_position_bias = prep["encoder_decoder_position_bias"]
        pending_count = prep["pending_count"]
        head_mask = prep["head_mask"]
        cross_attn_head_mask = prep["cross_attn_head_mask"]
        output_attentions = prep["output_attentions"]
        spans = [] if collect_spans else None
        for offset, j in enumerate(range(prep["source_layer"], len(self.block))):
            layer_outputs = self.block[j](
                hidden_states,
                attention_mask=prep["extended_attention_mask"],
                position_bias=position_bias,
                encoder_hidden_states=prep["encoder_hidden_states"],
                encoder_attention_mask=prep["encoder_extended_attention_mask"],
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[j] if head_mask is not None else None,
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[j] if cross_attn_head_mask is not None else None
                ),
                past_key_value=prep["resolved_pasts"][offset],
                use_cache=True,
                output_attentions=output_attentions,
                skip_mask=False,
                parallel_mask=True,
                stack_hidden_states=None,
                layer_idx=j,
                kv_importance_tracker=None,
            )
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if encoder_decoder_position_bias is not None or prep["encoder_hidden_states"] is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if collect_spans:
                spans.append(
                    (
                        present_key_value_state[0][:, :, -pending_count:, :],
                        present_key_value_state[1][:, :, -pending_count:, :],
                    )
                )
        return spans

    def _pure_recovery_run_ours_restoration(self, pending_hidden):
        """TIMED body of the Ours pure-recovery measurement: the frozen
        Phase-3c restoration arithmetic ONLY -- the same-layer target
        (== source layer) through the existing native restore_from_hidden()
        and every strictly deeper target through the existing stacked
        learned restoration -- ending exactly when the restored missing K/V
        tensors are ready. No cache staging, no candidate construction, no
        single-append publication, no current-token execution."""

        restorer = self._pure_recovery_shadow_restorer_obj()
        source_layer = int(self.shallow_exit_layer)
        source_attention = self.block[source_layer].layer[0]
        native = restorer.restore_from_hidden(
            pending_hidden,
            source_layer,
            source_layer,
            source_attention.layer_norm,
            source_attention.SelfAttention.k,
            source_attention.SelfAttention.v,
            output_device=pending_hidden.device,
            output_dtype=pending_hidden.dtype,
            source_hidden_prevalidated=True,
            defer_output_finite_validation=True,
        )
        learned_target_layers = list(range(source_layer + 1, len(self.block)))
        stacked = restorer.restore_learned_targets_from_hidden_stacked(
            pending_hidden,
            source_layer,
            learned_target_layers,
            [self.block[layer].layer[0].layer_norm for layer in learned_target_layers],
            [self.block[layer].layer[0].SelfAttention.k for layer in learned_target_layers],
            [self.block[layer].layer[0].SelfAttention.v for layer in learned_target_layers],
            output_device=pending_hidden.device,
            output_dtype=pending_hidden.dtype,
            source_hidden_prevalidated=True,
            defer_output_finite_validation=True,
            validate_fitted_parameters_finite=False,
        )
        return native, stacked

    def _measure_pure_missing_kv_recovery(
        self,
        past_key_values,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
        decoder_position,
    ):
        """Run both shadow measurements for THIS real FREE flush event over
        the exact same pending workload. Returns the held FREE shadow spans
        (for post-flush parity validation) when validation is enabled, else
        None. Never mutates live state; failures are recorded on the event
        and never break generation."""

        pending_count = len(self.stack_hidden_states)
        if pending_count <= 0:
            return None
        event = {
            "record_type": "pure_missing_kv_recovery_event",
            "generation_index": int(self._generation_index),
            "decoder_position": int(decoder_position) if decoder_position is not None else None,
            "source_layer": int(self.shallow_exit_layer),
            "pending_token_count": int(pending_count),
            "target_layer_count": int(len(self.block) - self.shallow_exit_layer),
            "requested_token_layer_units": int(
                pending_count * (len(self.block) - self.shallow_exit_layer)
            ),
            "free_pure_recovery_ms": None,
            "free_timing_backend": None,
            "ours_pure_recovery_ms": None,
            "ours_timing_backend": None,
            "measurement_error": None,
        }
        event.update(self._missing_kv_sample_context_fields())
        events = self._pure_recovery_events_obj()
        events.append(event)
        held_spans = None
        try:
            pending_hidden = torch.cat(self.stack_hidden_states, dim=1)
            # UNTIMED setup: shadow bank preparation (first event only) and
            # replay bookkeeping (mask/bias/cross seeding).
            self._pure_recovery_prepare_shadow()
            prep = self._pure_recovery_prepare_exact_replay_inputs(
                pending_hidden,
                past_key_values,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                head_mask,
                cross_attn_head_mask,
                use_cache,
                output_attentions,
            )
            timer = self._pure_recovery_timer_obj()

            def _on_free_resolved(elapsed_ms, backend, _event=event):
                _event["free_pure_recovery_ms"] = elapsed_ms
                _event["free_timing_backend"] = backend

            def _on_ours_resolved(elapsed_ms, backend, _event=event):
                _event["ours_pure_recovery_ms"] = elapsed_ms
                _event["ours_timing_backend"] = backend

            validation_enabled = bool(
                getattr(self.config, "kv_pure_recovery_cost_validation_enabled", False)
            )
            with timer.time_block(
                self._PURE_RECOVERY_FREE_KEY,
                device=pending_hidden.device,
                on_resolved=_on_free_resolved,
            ):
                free_spans = self._pure_recovery_run_pending_only_exact_replay(
                    pending_hidden, prep, collect_spans=validation_enabled
                )
            with timer.time_block(
                self._PURE_RECOVERY_OURS_KEY,
                device=pending_hidden.device,
                on_resolved=_on_ours_resolved,
            ):
                native, stacked = self._pure_recovery_run_ours_restoration(pending_hidden)
            # Post-timing sanity (untimed): both restorations produced
            # results for the same workload.
            event["ours_native_status"] = native.status
            event["ours_stacked_status"] = stacked.status
            if validation_enabled and free_spans is not None:
                held_spans = {
                    "pending_count": pending_count,
                    "source_layer": prep["source_layer"],
                    "free_spans": free_spans,
                    "event": event,
                }
            del native, stacked, free_spans
        except Exception as exc:
            event["measurement_error"] = "{}:{}".format(type(exc).__name__, exc)
            held_spans = None
        return held_spans

    def _pure_recovery_validate_pending_parity(self, held_spans, present_key_value_states):
        """Post-flush parity gate (validation mode only): the pending-only
        shadow K/V must match the pending portion of the production mixed
        flush for every deep layer. Records bounded scalar evidence on the
        event, then discards the shadow tensors."""

        event = held_spans["event"]
        pending_count = held_spans["pending_count"]
        source_layer = held_spans["source_layer"]
        max_abs = 0.0
        max_rel = 0.0
        layers_compared = 0
        allclose_all = True
        try:
            for offset, (shadow_key, shadow_value) in enumerate(held_spans["free_spans"]):
                layer = source_layer + offset
                production_state = present_key_value_states[layer]
                production_key = production_state[0][:, :, -(pending_count + 1) : -1, :]
                production_value = production_state[1][:, :, -(pending_count + 1) : -1, :]
                if production_key.shape != shadow_key.shape:
                    allclose_all = False
                    event["validation_error"] = "shape_mismatch_layer_{}".format(layer)
                    break
                for candidate, reference in (
                    (shadow_key, production_key),
                    (shadow_value, production_value),
                ):
                    diff = (candidate.float() - reference.float()).abs()
                    layer_abs = float(diff.max().item()) if diff.numel() else 0.0
                    denominator = reference.float().abs().clamp_min(1e-5)
                    layer_rel = float((diff / denominator).max().item()) if diff.numel() else 0.0
                    max_abs = max(max_abs, layer_abs)
                    max_rel = max(max_rel, layer_rel)
                    if not torch.allclose(candidate, reference, atol=1e-5, rtol=1e-4):
                        allclose_all = False
                layers_compared += 1
            event["validation"] = {
                "layers_compared": layers_compared,
                "pending_count": pending_count,
                "max_abs_diff": max_abs,
                "max_rel_diff_denominator_clamped": max_rel,
                "allclose_atol": 1e-5,
                "allclose_rtol": 1e-4,
                "allclose_all": bool(allclose_all),
            }
        except Exception as exc:
            event["validation_error"] = "{}:{}".format(type(exc).__name__, exc)

    def _pure_recovery_event_invalid_reasons(self, event, validation_required):
        """Per-event PAPER validity: every correctness condition must pass,
        never just 'both timings present'."""

        reasons = []
        if event.get("measurement_error") is not None:
            reasons.append("measurement_error")
        for side in ("free", "ours"):
            elapsed = event.get("{}_pure_recovery_ms".format(side))
            if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
                reasons.append("{}_timing_unresolved_or_invalid".format(side))
            backend = event.get("{}_timing_backend".format(side))
            if backend not in ("cuda_events", "cpu_perf_counter"):
                reasons.append("{}_timing_backend_invalid".format(side))
        if event.get("ours_native_status") != "ok":
            reasons.append("ours_native_status_not_ok")
        if event.get("ours_stacked_status") != "ok":
            reasons.append("ours_stacked_status_not_ok")
        if validation_required:
            if event.get("validation_error"):
                reasons.append("validation_error")
            validation = event.get("validation")
            if not validation:
                if "validation_error" not in [r for r in reasons]:
                    reasons.append("validation_missing")
            else:
                if validation.get("layers_compared") != event.get("target_layer_count"):
                    reasons.append("validation_layer_count_mismatch")
                if validation.get("pending_count") != event.get("pending_token_count"):
                    reasons.append("validation_pending_count_mismatch")
                if validation.get("allclose_all") is not True:
                    reasons.append("validation_allclose_failed")
        return reasons

    def pure_recovery_cost_summary(self):
        """Paper-facing PURE_MISSING_KV_RECOVERY_COMPUTE_COST summary for the
        resolved measurement protocol. Call AFTER the pure-recovery timer has
        been finalized so deferred CUDA timings are resolved. The two
        protocols are mutually exclusive by configuration (Native FREE
        requires use_shallow_deep, Official CALM requires use_early_exit),
        so exactly one event population can exist per evaluation."""

        if self._calm_pure_recovery_cost_enabled():
            return self._calm_pure_recovery_cost_summary()
        return self._native_pure_recovery_cost_summary()

    def _native_pure_recovery_cost_summary(self):
        """Aggregate the bounded per-event rows into the paper-facing
        summary. Call AFTER the pure-recovery timer has been finalized so
        deferred CUDA timings are resolved.

        FAIL-CLOSED: ``paper_metric_valid`` is True only when every recorded
        event passes every correctness condition (timings resolved on a real
        backend, Ours statuses ok, parity validation when enabled) AND the
        run-level timer is valid with zero errors and zero unresolved
        events. The paper reduction percentage is emitted ONLY when valid;
        raw totals are otherwise retained under explicitly diagnostic
        names."""

        timer = self.__dict__.get("pure_recovery_component_timer")
        events = list(getattr(self, "_pure_recovery_events", ()))
        validation_required = bool(
            getattr(self.config, "kv_pure_recovery_cost_validation_enabled", False)
        )
        event_invalid_reasons = {}
        resolved = []
        for index, event in enumerate(events):
            reasons = self._pure_recovery_event_invalid_reasons(event, validation_required)
            if reasons:
                event_invalid_reasons[str(index)] = reasons
            else:
                resolved.append(event)

        timer_summary = timer.to_summary() if timer is not None else None
        timer_valid = bool(
            timer_summary is not None
            and timer_summary.get("timing_valid") is True
            and not timer_summary.get("timing_errors")
            and int(timer_summary.get("unresolved_pending_event_count") or 0) == 0
        )

        paper_metric_valid = bool(
            events and not event_invalid_reasons and len(resolved) == len(events) and timer_valid
        )
        if not events:
            measurement_status = "no_events_measured"
        elif not timer_valid:
            measurement_status = "timer_invalid"
        elif any(
            reason.startswith("validation")
            for reasons in event_invalid_reasons.values()
            for reason in reasons
        ):
            measurement_status = "validation_failed"
        elif event_invalid_reasons:
            measurement_status = "partial_or_failed_events"
        else:
            measurement_status = "ok"

        free_total = sum(event["free_pure_recovery_ms"] for event in resolved)
        ours_total = sum(event["ours_pure_recovery_ms"] for event in resolved)
        pending_total = sum(event["pending_token_count"] for event in resolved)
        unit_total = sum(event["requested_token_layer_units"] for event in resolved)
        validations = [event.get("validation") for event in events if event.get("validation")]
        summary = {
            "metric": "PURE_MISSING_KV_RECOVERY_COMPUTE_COST",
            "measurement_protocol": "native_free_fixed_source",
            "measurement_status": measurement_status,
            "paper_metric_valid": paper_metric_valid,
            "paper_metric_invalid_reasons": (
                None
                if paper_metric_valid
                else {
                    "events_recorded": len(events),
                    "timer_valid": timer_valid,
                    "event_invalid_reasons": event_invalid_reasons,
                }
            ),
            "validation_required": validation_required,
            "artifact_binding": self.__dict__.get("pure_recovery_artifact_binding"),
            "events_recorded": len(events),
            "events_resolved": len(resolved),
            "pending_tokens_measured": pending_total,
            "token_layer_units_measured": unit_total,
            "free_pure_recovery_total_ms": free_total,
            "ours_pure_recovery_total_ms": ours_total,
            "free_ms_per_pending_token": (free_total / pending_total) if pending_total else None,
            "ours_ms_per_pending_token": (ours_total / pending_total) if pending_total else None,
            "free_us_per_token_layer_unit": (1000.0 * free_total / unit_total) if unit_total else None,
            "ours_us_per_token_layer_unit": (1000.0 * ours_total / unit_total) if unit_total else None,
            # PAPER-USABLE only when every validity condition passed; the raw
            # diagnostic value is retained separately and explicitly marked.
            "pure_recovery_reduction_percent": (
                100.0 * (free_total - ours_total) / free_total
                if paper_metric_valid and free_total > 0
                else None
            ),
            "diagnostic_reduction_percent_unvalidated": (
                100.0 * (free_total - ours_total) / free_total if free_total > 0 else None
            ),
            "validation_events": len(validations),
            "validation_all_allclose": (
                all(entry.get("allclose_all") for entry in validations) if validations else None
            ),
            "validation_max_abs_diff": (
                max(entry.get("max_abs_diff", 0.0) for entry in validations) if validations else None
            ),
            "timer_summary": timer_summary,
            "timer_valid": timer_valid,
            "scope_notes": {
                "reference_trajectory": "live FREE Exact generation (unchanged)",
                "free_timer_scope": (
                    "pending-token-only exact deep replay block calls (blocks "
                    "source..N-1); excludes current-token deep computation, "
                    "cross-attention seeding, mask/position-bias bookkeeping"
                ),
                "ours_timer_scope": (
                    "same-layer native restoration + stacked strictly-deeper "
                    "learned Phase-3c restoration arithmetic only; ends when "
                    "restored K/V are ready; excludes cache "
                    "staging/commit/publication and bank materialization "
                    "(prepared once, untimed)"
                ),
            },
            "events": events,
        }
        return summary

    # ------------------------------------------------------------------
    # PURE_MISSING_KV_RECOVERY_COMPUTE_COST -- Official FREE CALM protocol
    # (official_free_calm_first_crossing). The LIVE trajectory is Official
    # CALM + Exact K/V (exact_catchup). At every REAL first-crossing event
    # (source layer s, source hidden h_s = the raw hidden entering block s)
    # the SAME event is measured under all three recovery arms:
    #   STATE  -- shadow: sequential per-target exit-hidden projection
    #             (h_s -> target LayerNorm -> target W_K/W_V) via the
    #             existing artifact-free exit_hidden_target_projection
    #             manager, targets s..N-1 (conventional CALM State Copying
    #             recovery compute; deliberately NOT stacked/vectorized).
    #   EXACT  -- the ACTUAL live exact deep replay (blocks s..N-1) is
    #             timed in place, per target block call; no duplicate
    #             shadow replay is ever run.
    #   OURS   -- shadow: the frozen Phase-3c restoration (same-layer
    #             native target s + stacked learned targets s+1..N-1;
    #             source N-1 has no learned stack by construction).
    # State and Ours results are discarded after timing and never touch
    # the live cache/hidden/exit decision. The Phase-3c shadow artifact is
    # bound by its own measurement-only config fields (never the live
    # exact_catchup runtime fields) and SHA-verified before load.
    # ------------------------------------------------------------------

    _CALM_PURE_RECOVERY_STATE_KEY = "pure_recovery_state_copy_ms"
    _CALM_PURE_RECOVERY_STATE_STACKED_KEY = "pure_recovery_state_stacked_ms"
    _CALM_PURE_RECOVERY_EXACT_KEY = "pure_recovery_exact_replay_ms"
    _CALM_PURE_RECOVERY_OURS_KEY = "pure_recovery_ours_phase3c_ms"

    def _calm_pure_recovery_cost_enabled(self):
        """Official CALM three-arm measurement gate: only on the Official
        FREE CALM + Exact (exact_catchup) live trajectory. Full fail-closed
        configuration validation happens once in update_autoconfig(); this
        runtime gate additionally refuses any other live arm."""

        return (
            self.is_decoder
            and bool(getattr(self.config, "kv_pure_recovery_cost_enabled", False))
            and self.use_early_exit
            and not self.use_shallow_deep
            and not bool(getattr(self.config, "copy_skipped_hidden_states", False))
            and self._is_calm_taskc1_runtime_enabled()
            and getattr(self.kv_runtime_restorer, "method", None) == EXACT_CATCHUP_METHOD
            and getattr(self.kv_runtime_restorer, "runtime_source_mode", None)
            == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM
        )

    def _calm_pure_recovery_events_obj(self):
        if not hasattr(self, "_calm_pure_recovery_events"):
            self._calm_pure_recovery_events = []
        return self._calm_pure_recovery_events

    class _CalmPureRecoveryShadowConfigView:
        """Read-only view of the live config for constructing the
        measurement-only Phase-3c shadow restorer. The live runtime method is
        exact_catchup (artifact-free); the shadow manager IS a Phase-3c
        manager, so only the method field it validates against is presented
        as phase3c_kv_final. Nothing on the live config is mutated."""

        def __init__(self, config):
            object.__setattr__(self, "_calm_pure_recovery_live_config", config)

        def __getattr__(self, name):
            if name == "kv_runtime_restoration_method":
                return PHASE3C_RUNTIME_RESTORATION_METHOD
            return getattr(
                object.__getattribute__(self, "_calm_pure_recovery_live_config"), name
            )

    def _calm_pure_recovery_phase3c_restorer_obj(self):
        """Measurement-only Phase-3c shadow restorer bound to the dedicated
        kv_pure_recovery_cost_phase3c_artifact(+_sha256) fields -- NEVER the
        live exact_catchup runtime artifact/method fields. The approved
        SHA-256 is required and verified against the actual file BEFORE
        deserialization; mismatch fails closed."""

        if "calm_pure_recovery_phase3c_restorer" not in self.__dict__:
            from our_kv_restoration.missing_kv_dump_provenance import sha256_file
            from our_kv_restoration.runtime_kv_restoration import RuntimeKVRestorationManager

            artifact_path = getattr(self.config, "kv_pure_recovery_cost_phase3c_artifact", None)
            expected_sha = getattr(
                self.config, "kv_pure_recovery_cost_phase3c_artifact_sha256", None
            )
            if not artifact_path:
                raise ValueError("calm_pure_recovery_phase3c_artifact_path_missing")
            expected_sha = str(expected_sha or "").strip()
            if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
                raise ValueError(
                    "calm_pure_recovery_phase3c_artifact_sha256_missing_or_invalid: "
                    "kv_pure_recovery_cost_phase3c_artifact_sha256 must be lowercase 64-hex, "
                    "got {!r}".format(expected_sha)
                )
            actual_sha = sha256_file(artifact_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    "calm_pure_recovery_phase3c_artifact_sha256_mismatch: "
                    "expected={} actual={} path={}".format(expected_sha, actual_sha, artifact_path)
                )
            self.__dict__["calm_pure_recovery_artifact_binding"] = {
                "artifact_path": str(artifact_path),
                "expected_artifact_sha256": expected_sha,
                "actual_artifact_sha256": actual_sha,
                "verified_before_load": True,
            }
            self.__dict__["calm_pure_recovery_phase3c_restorer"] = (
                RuntimeKVRestorationManager.from_path(
                    artifact_path,
                    PHASE3C_RUNTIME_RESTORATION_METHOD,
                    threshold=getattr(self.config, "kv_runtime_restoration_threshold", None),
                    model_config=self._CalmPureRecoveryShadowConfigView(self.config),
                )
            )
        return self.__dict__["calm_pure_recovery_phase3c_restorer"]

    def _calm_pure_recovery_state_restorer_obj(self):
        """Measurement-only State Copying shadow manager: the EXISTING
        artifact-free exit_hidden_target_projection method (target LayerNorm
        + target W_K/W_V from the copied exit hidden, no hidden affine, no
        K/V corrections) -- exactly the conventional CALM State Copying
        missing-K/V recovery compute."""

        if "calm_pure_recovery_state_restorer" not in self.__dict__:
            from our_kv_restoration.runtime_kv_restoration import RuntimeKVRestorationManager

            self.__dict__["calm_pure_recovery_state_restorer"] = (
                RuntimeKVRestorationManager.from_path(
                    "",
                    EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
                    threshold=getattr(self.config, "kv_runtime_restoration_threshold", None),
                    model_config=self.config,
                )
            )
        return self.__dict__["calm_pure_recovery_state_restorer"]

    def _calm_pure_recovery_learned_target_modules(self, source_layer):
        learned_target_layers = list(range(int(source_layer) + 1, len(self.block)))
        norm_list = [self.block[layer].layer[0].layer_norm for layer in learned_target_layers]
        key_list = [self.block[layer].layer[0].SelfAttention.k for layer in learned_target_layers]
        value_list = [self.block[layer].layer[0].SelfAttention.v for layer in learned_target_layers]
        return learned_target_layers, norm_list, key_list, value_list

    def _calm_pure_recovery_prepare_source(self, source_layer):
        """UNTIMED per-event preparation: shadow restorer construction (with
        SHA-verified artifact binding) and the source-specific stacked
        parameter bank. CALM has a dynamic source, so a source change rebuilds
        the existing single-slot bank HERE (fingerprint mismatch), never
        inside the Ours timer; an unchanged source is a cheap fingerprint hit.
        The device-local fitted-parameter cache persists across sources."""

        restorer = self._calm_pure_recovery_phase3c_restorer_obj()
        state_restorer = self._calm_pure_recovery_state_restorer_obj()
        learned_target_layers, norm_list, key_list, value_list = (
            self._calm_pure_recovery_learned_target_modules(source_layer)
        )
        if learned_target_layers:
            support = restorer._stacked_learned_module_support(norm_list, key_list, value_list)
            if support is not None:
                module_device, module_dtype, variance_epsilon = support
                restorer._stacked_learned_target_bank(
                    int(source_layer),
                    learned_target_layers,
                    norm_list,
                    key_list,
                    value_list,
                    module_device,
                    module_dtype,
                    variance_epsilon,
                    validate_fitted_parameters_finite=True,
                )
        # State-Stacked fairness control: stack the s..N-1 projection
        # WEIGHTS (norm/W_K/W_V only, no fitted parameters) outside the
        # State-Stacked timer, mirroring the Ours bank rule above. Covers
        # source == N-1, whose single target still needs the bank.
        state_target_layers = list(range(int(source_layer), len(self.block)))
        state_norms = [self.block[layer].layer[0].layer_norm for layer in state_target_layers]
        state_keys = [self.block[layer].layer[0].SelfAttention.k for layer in state_target_layers]
        state_values = [self.block[layer].layer[0].SelfAttention.v for layer in state_target_layers]
        state_support = state_restorer._stacked_learned_module_support(
            state_norms, state_keys, state_values
        )
        if state_support is not None:
            state_device, state_dtype, state_epsilon = state_support
            state_restorer._stacked_exit_hidden_projection_bank(
                state_target_layers,
                state_norms,
                state_keys,
                state_values,
                state_device,
                state_dtype,
                state_epsilon,
            )

    def _calm_pure_recovery_run_state_stacked_restoration(self, source_hidden, source_layer):
        """TIMED body of the STATE-STACKED fairness-control arm: the SAME
        conventional State Copying mathematics as the sequential arm
        (h_s -> target LayerNorm -> target W_K/W_V for EVERY missing target
        s..N-1, target==s included), executed as ONE target-stacked
        operation through the existing stacked projection primitives. This
        is NOT an original CALM runtime implementation -- it exists only as
        the implementation-fair optimized timing control alongside the
        conventional sequential arm."""

        restorer = self._calm_pure_recovery_state_restorer_obj()
        source_layer = int(source_layer)
        target_layers = list(range(source_layer, len(self.block)))
        return restorer.restore_exit_hidden_targets_stacked(
            source_hidden,
            source_layer,
            target_layers,
            [self.block[layer].layer[0].layer_norm for layer in target_layers],
            [self.block[layer].layer[0].SelfAttention.k for layer in target_layers],
            [self.block[layer].layer[0].SelfAttention.v for layer in target_layers],
            output_device=source_hidden.device,
            output_dtype=source_hidden.dtype,
            source_hidden_prevalidated=True,
            defer_output_finite_validation=True,
        )

    def _calm_pure_recovery_run_state_restoration(self, source_hidden, source_layer):
        """TIMED body of the STATE arm: sequential per-target-layer
        exit-hidden projection for EVERY missing target s..N-1 (target==s
        included -- State Copying regenerates it from h_s through the
        target-s LayerNorm/W_K/W_V exactly like every deeper target). No
        hidden update, no attention/FFN, no cache publication."""

        restorer = self._calm_pure_recovery_state_restorer_obj()
        statuses = []
        for target_layer in range(int(source_layer), len(self.block)):
            target_self_attention = self.block[target_layer].layer[0]
            result = restorer.restore_from_hidden(
                source_hidden,
                source_layer,
                target_layer,
                target_self_attention.layer_norm,
                target_self_attention.SelfAttention.k,
                target_self_attention.SelfAttention.v,
                output_device=source_hidden.device,
                output_dtype=source_hidden.dtype,
                source_hidden_prevalidated=True,
                defer_output_finite_validation=True,
            )
            statuses.append(result.status)
        return statuses

    def _calm_pure_recovery_run_ours_restoration(self, source_hidden, source_layer):
        """TIMED body of the OURS arm: the frozen Phase-3c restoration
        arithmetic only -- same-layer native target s via the existing
        restore_from_hidden() path plus ONE stacked learned restoration for
        targets s+1..N-1 (skipped entirely when s == N-1). Ends when the
        restored K/V are ready; no staging/commit/publication."""

        restorer = self._calm_pure_recovery_phase3c_restorer_obj()
        source_layer = int(source_layer)
        source_self_attention = self.block[source_layer].layer[0]
        native = restorer.restore_from_hidden(
            source_hidden,
            source_layer,
            source_layer,
            source_self_attention.layer_norm,
            source_self_attention.SelfAttention.k,
            source_self_attention.SelfAttention.v,
            output_device=source_hidden.device,
            output_dtype=source_hidden.dtype,
            source_hidden_prevalidated=True,
            defer_output_finite_validation=True,
        )
        learned_target_layers, norm_list, key_list, value_list = (
            self._calm_pure_recovery_learned_target_modules(source_layer)
        )
        if not learned_target_layers:
            return native, None
        stacked = restorer.restore_learned_targets_from_hidden_stacked(
            source_hidden,
            source_layer,
            learned_target_layers,
            norm_list,
            key_list,
            value_list,
            output_device=source_hidden.device,
            output_dtype=source_hidden.dtype,
            source_hidden_prevalidated=True,
            defer_output_finite_validation=True,
            validate_fitted_parameters_finite=False,
        )
        return native, stacked

    def _calm_pure_recovery_begin_event(self, *, source_layer, source_hidden, decoder_position):
        """Create/bind ONLY the matched three-arm event metadata for THIS
        real CALM first-crossing and return the event dict. Per-event
        execution order is strictly:

            metadata binding (here)
            -> the LIVE Exact replay s..N-1, timed in place exactly once
               (per-target, via _calm_pure_recovery_exact_target_time_block)
            -> only after that replay completed, the untimed bank
               preparation and the Ours/State shadow timings
               (_calm_pure_recovery_measure_shadows)
            -> the live Exact transaction continues to its commit.

        The Exact paper baseline is therefore always measured BEFORE any
        shadow arithmetic or bank preparation can perturb GPU cache or
        utilization state. Nothing here touches live state."""

        source_layer = int(source_layer)
        num_layers = len(self.block)
        target_layer_count = num_layers - source_layer
        event = {
            "record_type": "calm_pure_missing_kv_recovery_event",
            "generation_index": int(getattr(self, "_generation_index", 0) or 0),
            "decoder_position": int(decoder_position) if decoder_position is not None else None,
            "source_layer": source_layer,
            "source_hidden_semantics": "raw_block_input_h_{}".format(source_layer),
            "pending_token_count": 1,
            "target_layer_count": target_layer_count,
            "requested_token_layer_units": target_layer_count,
            "state_pure_recovery_ms": None,
            "state_timing_backend": None,
            "state_stacked_pure_recovery_ms": None,
            "state_stacked_timing_backend": None,
            "exact_pure_recovery_ms": None,
            "exact_timing_backend": None,
            "exact_target_timings_expected": target_layer_count,
            "exact_target_timings_resolved": 0,
            "exact_timing_failures": 0,
            "ours_pure_recovery_ms": None,
            "ours_timing_backend": None,
            "live_exact_transaction_committed": False,
            "measurement_error": None,
        }
        event.update(self._missing_kv_sample_context_fields())
        self._calm_pure_recovery_events_obj().append(event)
        return event

    def _calm_pure_recovery_measure_shadows(self, event, source_hidden, source_layer):
        """Measure the shadow arms for THIS event, strictly AFTER the live
        Exact replay for the same event has completed (order:
        Exact -> Ours -> State-Stacked -> State-Sequential). Untimed setup
        (SHA-verified shadow restorers + the source-specific stacked banks)
        precedes the shadow timers. Never mutates live state; failures are
        recorded on the event and never break the live transaction."""

        source_layer = int(source_layer)
        target_layer_count = len(self.block) - source_layer
        try:
            # UNTIMED setup: SHA-verified shadow restorers + the
            # source-specific stacked banks (Ours learned + State weights).
            self._calm_pure_recovery_prepare_source(source_layer)
            timer = self._pure_recovery_timer_obj()

            def _on_state_resolved(elapsed_ms, backend, _event=event):
                _event["state_pure_recovery_ms"] = elapsed_ms
                _event["state_timing_backend"] = backend

            def _on_state_stacked_resolved(elapsed_ms, backend, _event=event):
                _event["state_stacked_pure_recovery_ms"] = elapsed_ms
                _event["state_stacked_timing_backend"] = backend

            def _on_ours_resolved(elapsed_ms, backend, _event=event):
                _event["ours_pure_recovery_ms"] = elapsed_ms
                _event["ours_timing_backend"] = backend

            with timer.time_block(
                self._CALM_PURE_RECOVERY_OURS_KEY,
                device=source_hidden.device,
                on_resolved=_on_ours_resolved,
            ):
                native, stacked = self._calm_pure_recovery_run_ours_restoration(
                    source_hidden, source_layer
                )
            with timer.time_block(
                self._CALM_PURE_RECOVERY_STATE_STACKED_KEY,
                device=source_hidden.device,
                on_resolved=_on_state_stacked_resolved,
            ):
                state_stacked = self._calm_pure_recovery_run_state_stacked_restoration(
                    source_hidden, source_layer
                )
            with timer.time_block(
                self._CALM_PURE_RECOVERY_STATE_KEY,
                device=source_hidden.device,
                on_resolved=_on_state_resolved,
            ):
                state_statuses = self._calm_pure_recovery_run_state_restoration(
                    source_hidden, source_layer
                )
            event["state_restoration_statuses"] = state_statuses
            event["state_all_targets_ok"] = bool(
                len(state_statuses) == target_layer_count
                and all(status == "ok" for status in state_statuses)
            )
            event["state_stacked_status"] = state_stacked.status
            event["state_stacked_target_count"] = (
                int(state_stacked.restored_key.shape[0])
                if state_stacked.restored_key is not None
                else None
            )
            event["ours_native_status"] = native.status
            event["ours_stacked_status"] = (
                stacked.status if stacked is not None else "skipped_no_learned_targets"
            )
            del native, stacked, state_stacked
        except Exception as exc:
            event["measurement_error"] = "{}:{}".format(type(exc).__name__, exc)
        return event

    def _calm_pure_recovery_exact_target_time_block(self, event, device):
        """Additional dedicated-timer context around ONE live exact
        target-block call (the EXACT arm times the actual live replay in
        place -- no duplicate replay exists). Null context when the CALM
        measurement is not active for this transaction."""

        if event is None:
            return contextlib.nullcontext()
        timer = self._pure_recovery_timer_obj()

        def _on_resolved(elapsed_ms, backend, _event=event):
            _event["exact_target_timings_resolved"] += 1
            if elapsed_ms is None or backend not in ("cuda_events", "cpu_perf_counter"):
                _event["exact_timing_failures"] += 1
                return
            _event["exact_pure_recovery_ms"] = float(
                _event.get("exact_pure_recovery_ms") or 0.0
            ) + float(elapsed_ms)
            prior = _event.get("exact_timing_backend")
            _event["exact_timing_backend"] = backend if prior in (None, backend) else "mixed"

        return timer.time_block(
            self._CALM_PURE_RECOVERY_EXACT_KEY, device=device, on_resolved=_on_resolved
        )

    def _calm_pure_recovery_event_invalid_reasons(self, event):
        """Per-event PAPER validity for the three-arm CALM measurement:
        every correctness condition must pass -- never just 'timings
        present'."""

        reasons = []
        num_layers = len(self.block)
        if event.get("measurement_error") is not None:
            reasons.append("measurement_error")
        if event.get("live_exact_transaction_committed") is not True:
            reasons.append("live_exact_transaction_not_committed")
        # Paper-facing sample/event identity is mandatory: an event that
        # cannot be bound to a stable sample and decoder position is never
        # paper-usable.
        stable_sample_id = event.get("stable_sample_id")
        if not isinstance(stable_sample_id, str) or not stable_sample_id.strip():
            reasons.append("stable_sample_id_missing")
        for identity_field in ("selected_order", "raw_dataset_index", "decoder_position"):
            value = event.get(identity_field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                reasons.append("{}_invalid".format(identity_field))
        source_layer = event.get("source_layer")
        candidate_layers = self._calm_phase3c_candidate_layers()
        if source_layer not in candidate_layers:
            reasons.append("source_layer_not_in_official_candidate_range")
        expected_targets = num_layers - int(source_layer) if source_layer is not None else None
        if event.get("target_layer_count") != expected_targets:
            reasons.append("target_layer_count_mismatch")
        if event.get("requested_token_layer_units") != expected_targets:
            reasons.append("requested_token_layer_units_mismatch")
        for side in ("state", "state_stacked", "exact", "ours"):
            elapsed = event.get("{}_pure_recovery_ms".format(side))
            if (
                not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
            ):
                reasons.append("{}_timing_unresolved_or_invalid".format(side))
            backend = event.get("{}_timing_backend".format(side))
            if backend not in ("cuda_events", "cpu_perf_counter"):
                reasons.append("{}_timing_backend_invalid".format(side))
        if event.get("exact_target_timings_resolved") != event.get("exact_target_timings_expected"):
            reasons.append("exact_target_timings_incomplete")
        if int(event.get("exact_timing_failures") or 0) != 0:
            reasons.append("exact_timing_failures")
        if event.get("state_all_targets_ok") is not True:
            reasons.append("state_restoration_status_not_ok")
        if event.get("state_stacked_status") != "ok":
            reasons.append("state_stacked_status_not_ok")
        if event.get("state_stacked_target_count") != expected_targets:
            reasons.append("state_stacked_target_count_mismatch")
        if event.get("ours_native_status") != "ok":
            reasons.append("ours_native_status_not_ok")
        stacked_status = event.get("ours_stacked_status")
        if source_layer is not None and int(source_layer) == num_layers - 1:
            if stacked_status != "skipped_no_learned_targets":
                reasons.append("ours_last_layer_source_stacked_not_skipped")
        elif stacked_status != "ok":
            reasons.append("ours_stacked_status_not_ok")
        return reasons

    def _calm_pure_recovery_cost_summary(self):
        """Paper-facing three-arm summary for the Official CALM protocol.
        FAIL-CLOSED exactly like the Native FREE summary: paper comparisons
        are emitted ONLY when every recorded event passes every per-event
        condition AND the run-level timer is valid; raw totals are otherwise
        retained under explicitly diagnostic names."""

        timer = self.__dict__.get("pure_recovery_component_timer")
        events = list(getattr(self, "_calm_pure_recovery_events", ()))
        event_invalid_reasons = {}
        resolved = []
        for index, event in enumerate(events):
            reasons = self._calm_pure_recovery_event_invalid_reasons(event)
            if reasons:
                event_invalid_reasons[str(index)] = reasons
            else:
                resolved.append(event)

        timer_summary = timer.to_summary() if timer is not None else None
        timer_valid = bool(
            timer_summary is not None
            and timer_summary.get("timing_valid") is True
            and not timer_summary.get("timing_errors")
            and int(timer_summary.get("unresolved_pending_event_count") or 0) == 0
        )
        paper_metric_valid = bool(
            events and not event_invalid_reasons and len(resolved) == len(events) and timer_valid
        )
        if not events:
            measurement_status = "no_events_measured"
        elif not timer_valid:
            measurement_status = "timer_invalid"
        elif event_invalid_reasons:
            measurement_status = "partial_or_failed_events"
        else:
            measurement_status = "ok"

        def _arm_total(arm):
            return sum(float(event["{}_pure_recovery_ms".format(arm)]) for event in resolved)

        state_total = _arm_total("state") if resolved else 0.0
        state_stacked_total = _arm_total("state_stacked") if resolved else 0.0
        exact_total = _arm_total("exact") if resolved else 0.0
        ours_total = _arm_total("ours") if resolved else 0.0
        unit_total = sum(int(event["requested_token_layer_units"]) for event in resolved)
        event_count = len(resolved)

        source_layer_counts = {}
        per_source = {}
        for event in resolved:
            source_layer = int(event["source_layer"])
            source_layer_counts[str(source_layer)] = (
                source_layer_counts.get(str(source_layer), 0) + 1
            )
            bucket = per_source.setdefault(
                source_layer,
                {
                    "source_layer": source_layer,
                    "event_count": 0,
                    "token_layer_units": 0,
                    "state_total_ms": 0.0,
                    "state_stacked_total_ms": 0.0,
                    "exact_total_ms": 0.0,
                    "ours_total_ms": 0.0,
                },
            )
            bucket["event_count"] += 1
            bucket["token_layer_units"] += int(event["requested_token_layer_units"])
            bucket["state_total_ms"] += float(event["state_pure_recovery_ms"])
            bucket["state_stacked_total_ms"] += float(event["state_stacked_pure_recovery_ms"])
            bucket["exact_total_ms"] += float(event["exact_pure_recovery_ms"])
            bucket["ours_total_ms"] += float(event["ours_pure_recovery_ms"])
        per_source_rows = []
        for source_layer in sorted(per_source):
            bucket = per_source[source_layer]
            units = bucket["token_layer_units"]
            for arm in ("state", "state_stacked", "exact", "ours"):
                bucket["{}_us_per_token_layer_unit".format(arm)] = (
                    1000.0 * bucket["{}_total_ms".format(arm)] / units if units else None
                )
            per_source_rows.append(bucket)

        def _reduction_percent(reference_total, candidate_total):
            if reference_total > 0:
                return 100.0 * (reference_total - candidate_total) / reference_total
            return None

        def _overhead_percent(reference_total, candidate_total):
            if reference_total > 0:
                return 100.0 * (candidate_total - reference_total) / reference_total
            return None

        summary = {
            "metric": "PURE_MISSING_KV_RECOVERY_COMPUTE_COST",
            "measurement_protocol": "official_free_calm_first_crossing",
            "measurement_status": measurement_status,
            "paper_metric_valid": paper_metric_valid,
            "paper_metric_invalid_reasons": (
                None
                if paper_metric_valid
                else {
                    "events_recorded": len(events),
                    "timer_valid": timer_valid,
                    "event_invalid_reasons": event_invalid_reasons,
                }
            ),
            "artifact_binding": self.__dict__.get("calm_pure_recovery_artifact_binding"),
            "events_recorded": len(events),
            "events_resolved": event_count,
            "total_requested_token_layer_units": unit_total,
            "source_layer_counts": source_layer_counts,
            "state_total_ms": state_total,
            "state_ms_per_event": (state_total / event_count) if event_count else None,
            "state_us_per_token_layer_unit": (
                1000.0 * state_total / unit_total if unit_total else None
            ),
            "state_stacked_total_ms": state_stacked_total,
            "state_stacked_ms_per_event": (
                (state_stacked_total / event_count) if event_count else None
            ),
            "state_stacked_us_per_token_layer_unit": (
                1000.0 * state_stacked_total / unit_total if unit_total else None
            ),
            "exact_total_ms": exact_total,
            "exact_ms_per_event": (exact_total / event_count) if event_count else None,
            "exact_us_per_token_layer_unit": (
                1000.0 * exact_total / unit_total if unit_total else None
            ),
            "ours_total_ms": ours_total,
            "ours_ms_per_event": (ours_total / event_count) if event_count else None,
            "ours_us_per_token_layer_unit": (
                1000.0 * ours_total / unit_total if unit_total else None
            ),
            # PAPER-USABLE comparisons only when every validity condition
            # passed; raw diagnostic values retained separately.
            "ours_vs_exact_reduction_percent": (
                _reduction_percent(exact_total, ours_total) if paper_metric_valid else None
            ),
            "state_vs_exact_reduction_percent": (
                _reduction_percent(exact_total, state_total) if paper_metric_valid else None
            ),
            "ours_vs_state_overhead_percent": (
                _overhead_percent(state_total, ours_total) if paper_metric_valid else None
            ),
            # Explicit-semantics aliases for the two historical sequential-
            # State fields above (state_* keeps meaning SEQUENTIAL State for
            # backward compatibility; these aliases carry the same values).
            "state_sequential_vs_exact_reduction_percent": (
                _reduction_percent(exact_total, state_total) if paper_metric_valid else None
            ),
            "ours_vs_state_sequential_overhead_percent": (
                _overhead_percent(state_total, ours_total) if paper_metric_valid else None
            ),
            # Implementation-fair comparisons against the STACKED State
            # control (same State mathematics, same class of target-wise
            # vectorization as Ours).
            "state_stacked_vs_exact_reduction_percent": (
                _reduction_percent(exact_total, state_stacked_total)
                if paper_metric_valid
                else None
            ),
            "ours_vs_state_stacked_overhead_percent": (
                _overhead_percent(state_stacked_total, ours_total)
                if paper_metric_valid
                else None
            ),
            "diagnostic_ours_vs_exact_reduction_percent_unvalidated": _reduction_percent(
                exact_total, ours_total
            ),
            "diagnostic_state_vs_exact_reduction_percent_unvalidated": _reduction_percent(
                exact_total, state_total
            ),
            "diagnostic_ours_vs_state_overhead_percent_unvalidated": _overhead_percent(
                state_total, ours_total
            ),
            "diagnostic_state_stacked_vs_exact_reduction_percent_unvalidated": _reduction_percent(
                exact_total, state_stacked_total
            ),
            "diagnostic_ours_vs_state_stacked_overhead_percent_unvalidated": _overhead_percent(
                state_stacked_total, ours_total
            ),
            "per_source_layer": per_source_rows,
            "timer_summary": timer_summary,
            "timer_valid": timer_valid,
            "scope_notes": {
                "reference_trajectory": "live Official CALM + Exact K/V generation (unchanged)",
                "state_sequential_role": "conventional CALM implementation",
                "state_stacked_role": (
                    "target-vectorized implementation-fair State Copying control "
                    "(same State mathematics; NOT an original CALM runtime "
                    "implementation)"
                ),
                "ours_role": "stacked Phase-3c recovery",
                "state_timer_scope": (
                    "SEQUENTIAL exit-hidden target projection (target LayerNorm + "
                    "target W_K/W_V) for every missing target s..N-1; no hidden "
                    "update, no attention/FFN, no cache publication"
                ),
                "state_stacked_timer_scope": (
                    "ONE target-stacked exit-hidden projection over all missing "
                    "targets s..N-1 (same State mathematics, vectorized like the "
                    "Ours stacked path); weight bank prepared per source, untimed; "
                    "no corrections, no cache publication"
                ),
                "exact_timer_scope": (
                    "the ACTUAL live exact deep replay's per-target block calls "
                    "(blocks s..N-1), timed in place; excludes confidence/LM-head "
                    "work, cache install, accounting and diagnostics"
                ),
                "ours_timer_scope": (
                    "same-layer native target-s restoration + one stacked Phase-3c "
                    "learned restoration (targets s+1..N-1; skipped for s==N-1); "
                    "ends when restored K/V are ready; excludes bank "
                    "materialization (prepared per source, untimed)"
                ),
                "not_an_end_to_end_speedup": (
                    "these values are pure missing-K/V recovery compute costs only"
                ),
            },
            "events": events,
        }
        return summary

    def _task_c2_batched_insertion_enabled(self):
        """Task C2 FREE-aligned lazy BATCHED K/V insertion. Same narrow
        opt-in layering as _task_c2_direct_insertion_enabled(), but a
        different execution SCHEDULE: FREE's own pending stack and flush
        trigger are reused untouched, and only what happens AT the existing
        flush point changes. Full fail-closed configuration validation --
        including mutual exclusion with direct insertion -- happens once, in
        util/additional_args.py's update_autoconfig()."""

        return (
            self.is_decoder
            and self.use_shallow_deep
            and not self.use_early_exit
            and bool(getattr(self.config, "kv_runtime_restoration_batched_insertion_enabled", False))
            and self._phase3c_runtime_restoration_enabled()
        )

    @staticmethod
    def _task_c2_cross_kv_identity(tensor):
        """Smallest identity that is already natural here: the tensor OBJECT
        plus its in-place mutation counter, shape, device and dtype.

        The batched flush reuses the caller's cross-attention entries
        verbatim, so the very same object reappears flush after flush -- that
        object is the identity. ``_version`` bumps on any in-place write, so a
        mutated-in-place tensor no longer matches. The dict holds a strong
        reference to the tensor it trusts, which also makes ``is`` safe: the
        object cannot be freed and its address reused while trusted."""

        return (
            int(getattr(tensor, "_version", -1)),
            tuple(tensor.shape),
            str(tensor.device),
            str(tensor.dtype),
        )

    def _task_c2_batched_cross_kv_is_trusted(self, target_layer, cross_key, cross_value):
        """True only if THIS exact pair of cross-attention tensors was already
        finite-validated during this generation and has not changed since.

        Deliberately keyed on object identity, never on the layer number
        alone: a new or replaced cross tensor at a trusted layer misses."""

        trust = getattr(self, "_task_c2_batched_cross_kv_trust", None)
        if not trust:
            return False
        entry = trust.get(int(target_layer))
        if entry is None:
            return False
        trusted_key, trusted_value, key_identity, value_identity = entry
        return (
            trusted_key is cross_key
            and trusted_value is cross_value
            and key_identity == self._task_c2_cross_kv_identity(cross_key)
            and value_identity == self._task_c2_cross_kv_identity(cross_value)
        )

    def _task_c2_batched_trust_cross_kv(self, target_layer, cross_key, cross_value):
        """Record trust for a cross-attention pair. Only ever called after the
        transaction-level finite validation that covered it has PASSED, so a
        tensor from a failed transaction is never trusted."""

        if getattr(self, "_task_c2_batched_cross_kv_trust", None) is None:
            self._task_c2_batched_cross_kv_trust = {}
        self._task_c2_batched_cross_kv_trust[int(target_layer)] = (
            cross_key,
            cross_value,
            self._task_c2_cross_kv_identity(cross_key),
            self._task_c2_cross_kv_identity(cross_value),
        )

    def _try_task_c2_fixed_source6_batched_insertion(
        self,
        source_layer,
        pending_source_hidden_states,
        pending_metadata,
        past_key_values,
        decoder_position,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
    ):
        """FREE-aligned lazy batched Phase-3c restoration of every pending
        early-exit token, executed at FREE's own existing synchronized flush
        point instead of exact deep replay.

        The N pending source hidden states already stored by FREE
        (self.stack_hidden_states, in generation order) are concatenated once
        into a single [1, N, d_model] tensor. Target 6 (same-layer) is
        restored through the existing native restore_from_hidden() path; the
        learned targets 7..23 are restored together in ONE stacked
        learned-target operation (restore_learned_targets_from_hidden_
        stacked, the frozen separate-K + separate-V ``stacked_bmm``
        candidate), produced directly in the learned self-attention cache's
        own dtype/device. The remaining per-target loop exists ONLY for the
        existing cache placement/staging. No target decoder block is executed
        for any pending token, so a successful Batched flush performs no
        exact replay for them. The current non-exit token is deliberately NOT
        restored: it is left to the caller's normal deep block loop, which
        appends its own exact K/V after the restored pending block.

        Everything is built in local temporaries (reusing the same
        _stage_runtime_restored_kv_slices/_commit_runtime_restored_kv_slices
        primitives as the Immediate path, now with span=N) and a brand-new
        past_key_values list is returned only if EVERY layer validated, so a
        failure can never leave a partial mutation behind and the caller can
        fall straight through to the unmodified parallel_gen_token() flush."""

        source_layer = int(source_layer)
        target_start = source_layer
        target_end = len(self.block)
        pending_source_hidden_states = tuple(pending_source_hidden_states or ())
        pending_token_count = len(pending_source_hidden_states)
        target_layer_count = max(0, target_end - target_start)
        requested_units = pending_token_count * target_layer_count
        result = {
            "success": False,
            "past_key_values": None,
            "pending_token_count": pending_token_count,
            "requested_units": requested_units,
            "inserted_units": 0,
            "failure_stage": None,
            "failure_type": None,
            "failure_message": None,
            "target_layers": list(range(target_start, target_end)),
            "decoder_position": int(decoder_position) if decoder_position is not None else None,
            "deep_cache_length_before": None,
            "deep_cache_length_after": None,
            "restoration_calls": 0,
            "stacked_learned_restoration": False,
            # Populated ONLY on a fully validated stacked success: per-layer
            # {target_layer: (restored_key, restored_value)} pending blocks
            # for the caller's single-append publication. None on every
            # failure and on the sequential compatibility path.
            "staged_pending_self_kv": None,
        }

        def _fail(stage, exc):
            result["failure_stage"] = stage
            result["failure_type"] = type(exc).__name__
            result["failure_message"] = str(exc)
            return result

        # Diagnostic finite-value validation switch (default ON). When
        # explicitly disabled for paper-facing timing, ONLY this
        # transaction's diagnostic NaN/Inf validation is skipped: the
        # per-transaction full-tensor scans, their validation-only scalar
        # readbacks, and the one-time fitted-parameter validation at cold
        # stacked-bank build. The restoration arithmetic, every shape/
        # layer-range/dtype/device structural check, cache staging, atomic
        # publication and accounting are byte-identical in both modes.
        # Fallback precision: the structural/exception fallback machinery
        # (shape/position/compatibility failures, exceptions) remains fully
        # available in both modes, but with validation OFF the
        # NONFINITE-DETECTION fallback intentionally cannot trigger --
        # that diagnostic is exactly what was disabled.
        finite_validation_enabled = bool(
            getattr(self.config, "kv_runtime_restoration_finite_validation_enabled", True)
        )

        try:
            if not use_cache:
                return _fail("precondition", ValueError("use_cache_false"))
            if pending_token_count <= 0:
                return _fail("precondition", ValueError("pending_stack_empty"))
            if target_layer_count <= 0:
                return _fail("precondition", ValueError("no_missing_target_layers"))
            if decoder_position is None or int(decoder_position) < 0:
                return _fail("precondition", ValueError("decoder_position_unavailable"))
            if past_key_values is None or len(past_key_values) != len(self.block):
                return _fail("precondition", ValueError("past_key_values_shape_invalid"))
            for index, pending_hidden in enumerate(pending_source_hidden_states):
                if (
                    pending_hidden is None
                    or pending_hidden.ndim != 3
                    or int(pending_hidden.shape[0]) != 1
                    or int(pending_hidden.shape[1]) != 1
                ):
                    return _fail(
                        "precondition",
                        ValueError(
                            "pending_source_hidden_shape_invalid:index_{}:{}".format(
                                index,
                                list(pending_hidden.shape) if pending_hidden is not None else None,
                            )
                        ),
                    )
            decoder_position = int(decoder_position)

            # One concatenation, in FREE's own chronological pending order:
            # index k is the k-th pending early exit, which occupies decoder
            # position deep_cache_length_before + k.
            try:
                source_hidden = torch.cat(pending_source_hidden_states, dim=1)
            except Exception as exc:
                return _fail("pending_concat", exc)
            if (
                source_hidden.ndim != 3
                or int(source_hidden.shape[0]) != 1
                or int(source_hidden.shape[1]) != pending_token_count
            ):
                return _fail(
                    "pending_concat",
                    ValueError("batched_source_hidden_shape_invalid:{}".format(list(source_hidden.shape))),
                )
            if finite_validation_enabled and not torch.isfinite(source_hidden).all().item():
                return _fail("precondition", ValueError("source_hidden_nonfinite"))

            # Every deep target layer must currently hold the same self-attn
            # cache length L, and FREE's own lag invariant must hold:
            # decoder_position == L + pending_count.
            deep_cache_length_before = None
            for target_layer in range(target_start, target_end):
                past_state = past_key_values[target_layer]
                layer_len = safe_cache_seq_len(past_state)
                layer_len = 0 if layer_len is None else int(layer_len)
                if deep_cache_length_before is None:
                    deep_cache_length_before = layer_len
                elif layer_len != deep_cache_length_before:
                    return _fail(
                        "cache_position:layer_{}".format(target_layer),
                        ValueError(
                            "deep_cache_layer_length_misaligned:{}!={}".format(
                                layer_len, deep_cache_length_before
                            )
                        ),
                    )
            deep_cache_length_before = int(deep_cache_length_before or 0)
            result["deep_cache_length_before"] = deep_cache_length_before
            if deep_cache_length_before + pending_token_count != decoder_position:
                return _fail(
                    "cache_position",
                    ValueError(
                        "deep_cache_pending_lag_invalid:{}+{}!={}".format(
                            deep_cache_length_before, pending_token_count, decoder_position
                        )
                    ),
                )

            # When FREE's own pending metadata carries positions, require them
            # to be exactly the contiguous chronological block this batch is
            # about to occupy -- restored pending tokens must never be
            # reordered or interleaved with the current token.
            metadata_positions = candidate_positions_from_metadata(pending_metadata)
            if metadata_positions is not None and len(metadata_positions) == pending_token_count:
                expected_positions = [deep_cache_length_before + k for k in range(pending_token_count)]
                if all(position is not None for position in metadata_positions):
                    if [int(position) for position in metadata_positions] != expected_positions:
                        return _fail(
                            "pending_order",
                            ValueError(
                                "pending_positions_not_chronological:{}!={}".format(
                                    [int(position) for position in metadata_positions], expected_positions
                                )
                            ),
                        )

            expected_heads = int(self.config.num_heads)
            expected_d_kv = int(self.config.d_kv)
            restoration_calls = 0

            # The frozen selected learned-target arithmetic: ONE stacked
            # separate-K + separate-V restoration covering every strictly
            # deeper learned target (7..23 for source 6) at once. Target 6
            # (same-layer) deliberately stays on the existing native
            # restore_from_hidden() path below -- it has no learned map and
            # must not be included in the learned stack. When the target
            # modules are not the T5 RMSNorm / bias-free Linear form the
            # stacked arithmetic vectorizes (synthetic scheduling fixtures),
            # the manager reports "unsupported" and this transaction keeps
            # its existing per-target sequential loop byte-identical.
            learned_target_layers = list(range(target_start + 1, target_end))

            # Canonical learned-cache dtype/device authority. The one-K-bank
            # + one-V-bank finite validation below is the transaction safety
            # authority ONLY if the stacked banks are produced in the exact
            # dtype/device of the self-attention cache they enter -- never
            # inferred from source_hidden, and never left to torch.cat()'s
            # floating-dtype promotion to "catch". Metadata-only reads: no
            # host-visible synchronization.
            learned_cache_device = None
            learned_cache_dtype = None
            for target_layer in learned_target_layers:
                past_state = past_key_values[target_layer]
                if past_state is None:
                    continue
                for slot in (0, 1):
                    tensor = past_state[slot] if len(past_state) > slot else None
                    if tensor is None or not isinstance(tensor, torch.Tensor):
                        continue
                    if learned_cache_device is None:
                        learned_cache_device = tensor.device
                        learned_cache_dtype = tensor.dtype
                    elif tensor.device != learned_cache_device or tensor.dtype != learned_cache_dtype:
                        return _fail(
                            "learned_cache_compatibility",
                            ValueError(
                                "learned_self_cache_dtype_device_inconsistent:layer_{}:slot_{}:{}/{}!={}/{}".format(
                                    target_layer,
                                    slot,
                                    tensor.device,
                                    tensor.dtype,
                                    learned_cache_device,
                                    learned_cache_dtype,
                                )
                            ),
                        )
            if learned_cache_device is None and learned_target_layers:
                # No learned self-K/V tensor exists yet (first flush of a
                # generation, past entries still None). The deep cache
                # contract: those entries are produced by these same target
                # projection modules, so the module weight dtype/device IS
                # the cache dtype/device.
                probe_weight = getattr(
                    self.block[learned_target_layers[0]].layer[0].SelfAttention.k, "weight", None
                )
                if not isinstance(probe_weight, torch.Tensor):
                    return _fail(
                        "learned_cache_compatibility",
                        ValueError("learned_cache_dtype_device_authority_unavailable"),
                    )
                learned_cache_device = probe_weight.device
                learned_cache_dtype = probe_weight.dtype

            stacked_key_bank = None
            stacked_value_bank = None
            if learned_target_layers:
                with self._missing_kv_component_timer_obj().time_block(
                    "task_c2_restoration_compute_time_ms",
                    device=source_hidden.device,
                ):
                    stacked_result = self.kv_runtime_restorer.restore_learned_targets_from_hidden_stacked(
                        source_hidden,
                        source_layer,
                        learned_target_layers,
                        [self.block[layer].layer[0].layer_norm for layer in learned_target_layers],
                        [self.block[layer].layer[0].SelfAttention.k for layer in learned_target_layers],
                        [self.block[layer].layer[0].SelfAttention.v for layer in learned_target_layers],
                        # The ACTUAL learned self-attention cache is the
                        # output dtype/device authority -- the banks come
                        # back as the exact post-cast values the candidate
                        # cache will hold.
                        output_device=learned_cache_device,
                        output_dtype=learned_cache_dtype,
                        # Finite-checked once above for this transaction.
                        source_hidden_prevalidated=True,
                        # Deferred to this transaction's aggregate check over
                        # the final post-cast candidate K/V below.
                        defer_output_finite_validation=True,
                        # In the explicit timing mode this also skips the
                        # one-time fitted-parameter NaN/Inf validation at
                        # cold bank build; such parameters stay marked
                        # unvalidated and are scanned before any later
                        # validation-ON use.
                        validate_fitted_parameters_finite=finite_validation_enabled,
                    )
                if stacked_result.status == "ok":
                    stacked_key_bank = stacked_result.restored_key
                    stacked_value_bank = stacked_result.restored_value
                    expected_bank_shape = [
                        len(learned_target_layers),
                        expected_heads,
                        pending_token_count,
                        expected_d_kv,
                    ]
                    if (
                        stacked_key_bank is None
                        or stacked_value_bank is None
                        or list(stacked_key_bank.shape) != expected_bank_shape
                        or list(stacked_value_bank.shape) != expected_bank_shape
                    ):
                        return _fail(
                            "stacked_learned_restoration",
                            ValueError(
                                "stacked_learned_bank_shape_invalid:{}".format(
                                    list(stacked_key_bank.shape)
                                    if stacked_key_bank is not None
                                    else None
                                )
                            ),
                        )
                    # The invariant the one-bank-scan validation stands on:
                    # both banks ARE in the learned cache's dtype/device.
                    # Metadata-only check, and it covers every narrow() slice
                    # staged below (views preserve dtype/device).
                    if (
                        stacked_key_bank.device != learned_cache_device
                        or stacked_key_bank.dtype != learned_cache_dtype
                        or stacked_value_bank.device != learned_cache_device
                        or stacked_value_bank.dtype != learned_cache_dtype
                    ):
                        return _fail(
                            "stacked_learned_restoration",
                            ValueError(
                                "stacked_learned_bank_dtype_device_mismatch:{}/{}!={}/{}".format(
                                    stacked_key_bank.device,
                                    stacked_key_bank.dtype,
                                    learned_cache_device,
                                    learned_cache_dtype,
                                )
                            ),
                        )
                    # One physical restoration operation for ALL learned
                    # targets together.
                    restoration_calls += 1
                    result["stacked_learned_restoration"] = True
                elif stacked_result.status == STACKED_LEARNED_TARGETS_UNSUPPORTED_STATUS:
                    stacked_key_bank = None
                    stacked_value_bank = None
                else:
                    return _fail(
                        "stacked_learned_restoration",
                        ValueError(
                            "{}:{}".format(stacked_result.status, stacked_result.error_message)
                        ),
                    )

            candidate_targets = []
            # SINGLE-APPEND staging (stacked production path only): the
            # restored pending K/V blocks are kept in this PRIVATE
            # transaction-local map instead of being pre-materialized into an
            # extended candidate cache. They become visible to the caller
            # only on a fully validated success, and the current non-exit
            # token's own deep self-attention then publishes
            # old + pending + current in ONE torch.cat per K/V (see
            # staged_pending_self_kv in DeployT5Attention.forward). On any
            # failure this local map simply dies with the call -- nothing is
            # ever staged into live state.
            staged_pending_self_kv = {}
            # Transaction-local only: (layer, key, value) triples whose
            # finiteness is not yet established. Bounded by the target-layer
            # count, discarded when this call returns.
            untrusted_cross = []
            for target_layer in range(target_start, target_end):
                if stacked_key_bank is not None and target_layer > target_start:
                    # Learned target: its K/V block is a view into the one
                    # stacked restoration computed above -- no per-target
                    # restore call. Index 0 of the bank is target_start + 1.
                    bank_index = target_layer - target_start - 1
                    restored_key_slice = stacked_key_bank.narrow(0, bank_index, 1)
                    restored_value_slice = stacked_value_bank.narrow(0, bank_index, 1)
                    # Cheap structural invariant (metadata only, no sync):
                    # the slice about to be staged is already the intended
                    # cache dtype/device, so staging's .to() is an identity
                    # and the bank scan below validates these exact values.
                    if (
                        restored_key_slice.device != learned_cache_device
                        or restored_key_slice.dtype != learned_cache_dtype
                        or restored_value_slice.device != learned_cache_device
                        or restored_value_slice.dtype != learned_cache_dtype
                    ):
                        return _fail(
                            "cache_staging:layer_{}".format(target_layer),
                            ValueError("stacked_learned_slice_dtype_device_mismatch"),
                        )
                    restore_result = RuntimeKVRestorationResult(
                        restored_key_slice,
                        restored_value_slice,
                        "ok",
                        None,
                        {
                            "target_layer": int(target_layer),
                            "restoration_submode": "phase3c_stacked_learned_slice",
                            "output_finite_validation_deferred": True,
                        },
                    )
                else:
                    # Target 6 (same-layer native projection), or every
                    # target of the sequential compatibility path: ONE
                    # batched restoration call per target layer, covering all
                    # N pending tokens together. There is deliberately no
                    # loop over pending tokens here.
                    with self._missing_kv_component_timer_obj().time_block(
                        "task_c2_restoration_compute_time_ms",
                        device=source_hidden.device,
                    ):
                        target_self_attention = self.block[target_layer].layer[0]
                        restore_result = self.kv_runtime_restorer.restore_from_hidden(
                            source_hidden,
                            source_layer,
                            target_layer,
                            target_self_attention.layer_norm,
                            target_self_attention.SelfAttention.k,
                            target_self_attention.SelfAttention.v,
                            # Same cache dtype/device authority as the
                            # stacked banks, so the whole candidate cache
                            # stays uniform and no torch.cat() dtype
                            # promotion can ever occur inside the candidate
                            # extension. (Identical to the previous behavior
                            # whenever source_hidden already matched the
                            # cache, which is every accepted production
                            # configuration.)
                            output_device=learned_cache_device or source_hidden.device,
                            output_dtype=learned_cache_dtype or source_hidden.dtype,
                            # The concatenated [1, N, d_model] pending block was
                            # finite-checked once above, before this loop, and is
                            # not modified by it -- so the per-target-layer
                            # recheck is a pure device synchronization.
                            source_hidden_prevalidated=True,
                            # Every intermediate finite check inside the restorer
                            # is a host-visible scalar readback. They are deferred
                            # to this transaction's single aggregate check over
                            # the final candidate K/V, which is strictly
                            # downstream of all of them and still runs before any
                            # authoritative commit.
                            defer_output_finite_validation=True,
                        )
                    restoration_calls += 1
                if restore_result.status != "ok":
                    return _fail(
                        "restore_from_hidden:layer_{}".format(target_layer),
                        ValueError("{}:{}".format(restore_result.status, restore_result.error_message)),
                    )
                restored_key = restore_result.restored_key
                restored_value = restore_result.restored_value
                if (
                    restored_key is None
                    or restored_value is None
                    or restored_key.shape != restored_value.shape
                    or int(restored_key.shape[0]) != 1
                    or int(restored_key.shape[1]) != expected_heads
                    or int(restored_key.shape[2]) != pending_token_count
                    or int(restored_key.shape[3]) != expected_d_kv
                ):
                    return _fail(
                        "restore_shape:layer_{}".format(target_layer),
                        ValueError(
                            "restored_kv_shape_invalid:{}".format(
                                list(restored_key.shape) if restored_key is not None else None
                            )
                        ),
                    )
                # A1: no finite check here. restored_key/restored_value are
                # exactly the tensors the restorer just produced and handed
                # back unmodified, so re-scanning them is a duplicate of the
                # restorer's own check. Both are now covered once, together,
                # by the transaction-level validation below.

                past_state = past_key_values[target_layer]
                self_key_ref = past_state[0] if (past_state is not None and len(past_state) > 0) else None
                self_value_ref = past_state[1] if (past_state is not None and len(past_state) > 1) else None
                if stacked_key_bank is not None:
                    # SINGLE-APPEND path (stacked production mode): no
                    # candidate cache extension, no zero-fill, no slice
                    # commit. The old exact self K/V pass through the
                    # candidate UNTOUCHED (a zero-length view stands in when
                    # this layer has never held any), and the validated
                    # restored block is staged privately for the ONE final
                    # old+pending+current publication inside the current
                    # token's own deep self-attention.
                    if target_layer == target_start and (
                        restored_key.device != learned_cache_device
                        or restored_key.dtype != learned_cache_dtype
                        or restored_value.device != learned_cache_device
                        or restored_value.dtype != learned_cache_dtype
                    ):
                        # Learned bank slices were already metadata-checked
                        # above; this covers the native target-6 result so
                        # the final single cat can never dtype-promote.
                        return _fail(
                            "cache_staging:layer_{}".format(target_layer),
                            ValueError("staged_pending_dtype_device_mismatch"),
                        )
                    staged_pending_self_kv[target_layer] = (restored_key, restored_value)
                    new_self_key = self_key_ref if self_key_ref is not None else restored_key.narrow(2, 0, 0)
                    new_self_value = (
                        self_value_ref if self_value_ref is not None else restored_value.narrow(2, 0, 0)
                    )
                else:
                    # Sequential compatibility path (synthetic fixtures whose
                    # modules the stacked arithmetic cannot vectorize): keep
                    # the existing pre-extended candidate flow byte-identical.
                    empty_key = self_key_ref if self_key_ref is not None else restored_key.narrow(2, 0, 0)
                    empty_value = self_value_ref if self_value_ref is not None else restored_value.narrow(2, 0, 0)
                    extended_key = torch.cat([empty_key, torch.zeros_like(restored_key)], dim=2)
                    extended_value = torch.cat([empty_value, torch.zeros_like(restored_value)], dim=2)
                    with self._missing_kv_component_timer_obj().time_block(
                        "task_c2_cache_staging_time_ms",
                        device=source_hidden.device,
                    ):
                        staged_key, staged_value, staging_error, staging_message = self._stage_runtime_restored_kv_slices(
                            restore_result,
                            extended_key,
                            extended_value,
                            deep_cache_length_before,
                            span=pending_token_count,
                            # Shape/device/dtype staging checks stay on; only the
                            # nonfinite scan moves to the aggregate below, which
                            # inspects the very bytes these staged tensors are
                            # copied into.
                            defer_finite_validation=True,
                        )
                    if staging_error is not None:
                        return _fail(
                            "cache_staging:layer_{}".format(target_layer),
                            ValueError("{}:{}".format(staging_error, staging_message)),
                        )
                    try:
                        with self._missing_kv_component_timer_obj().time_block(
                            "task_c2_cache_commit_time_ms",
                            device=source_hidden.device,
                        ):
                            new_self_key, new_self_value = self._commit_runtime_restored_kv_slices(
                                extended_key,
                                extended_value,
                                deep_cache_length_before,
                                staged_key,
                                staged_value,
                                span=pending_token_count,
                                # extended_key/extended_value are this iteration's
                                # own torch.cat() results: private, unaliased, and
                                # discarded wholesale if anything below fails.
                                candidate_is_private=True,
                            )
                    except Exception as exc:
                        return _fail("cache_commit:layer_{}".format(target_layer), exc)
                    expected_length = deep_cache_length_before + pending_token_count
                    if (
                        int(new_self_key.shape[2]) != expected_length
                        or int(new_self_value.shape[2]) != expected_length
                    ):
                        return _fail(
                            "cache_length:layer_{}".format(target_layer),
                            ValueError("new_self_cache_length_invalid"),
                        )

                # Cross-attention cache is method-irrelevant here and must
                # survive untouched: reuse the caller's own entries verbatim,
                # only seeding them when this layer has never held any.
                cross_key_ref = past_state[2] if (past_state is not None and len(past_state) > 2) else None
                cross_value_ref = past_state[3] if (past_state is not None and len(past_state) > 3) else None
                if cross_key_ref is None or cross_value_ref is None:
                    try:
                        cross_seed = self.block[target_layer].gen_cross_attn_key_value(
                            source_hidden,
                            attention_mask=None,
                            position_bias=None,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_extended_attention_mask,
                            encoder_decoder_position_bias=encoder_decoder_position_bias,
                            layer_head_mask=head_mask[target_layer] if head_mask is not None else None,
                            cross_attn_layer_head_mask=(
                                cross_attn_head_mask[target_layer] if cross_attn_head_mask is not None else None
                            ),
                            past_key_value=None,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )
                    except Exception as exc:
                        return _fail("cross_attention_init:layer_{}".format(target_layer), exc)
                    cross_key_ref, cross_value_ref = cross_seed[2], cross_seed[3]
                if cross_key_ref is None or cross_value_ref is None:
                    return _fail(
                        "cross_attention_init:layer_{}".format(target_layer),
                        ValueError("cross_kv_unavailable_or_nonfinite"),
                    )
                # A2: cross-attention K/V are reused verbatim and are NOT
                # produced by Phase-3c restoration, so once this exact pair
                # has been finite-validated in this generation it cannot have
                # silently gone nonfinite. A miss (first sight, replaced
                # tensor, or in-place mutation) falls into the aggregate below
                # and is only trusted after that aggregate passes. With
                # finite validation disabled there is no aggregate to feed,
                # so nothing is collected -- and nothing is ever marked
                # trusted, since trust means "was finite-validated".
                if finite_validation_enabled and not self._task_c2_batched_cross_kv_is_trusted(
                    target_layer, cross_key_ref, cross_value_ref
                ):
                    untrusted_cross.append((target_layer, cross_key_ref, cross_value_ref))

                candidate_targets.append([new_self_key, new_self_value, cross_key_ref, cross_value_ref])

            result["restoration_calls"] = restoration_calls
            if len(candidate_targets) != target_layer_count:
                return _fail("target_count", ValueError("candidate_target_count_mismatch"))
            # restoration_calls counts PHYSICAL restoration operations (a
            # diagnostic, audited across producers/consumers): stacked mode
            # performs 1 native same-layer restoration + 1 stacked learned
            # restoration; the sequential compatibility path keeps its one
            # call per target layer. LOGICAL coverage stays N x
            # target_layer_count either way, carried by target_layers /
            # requested_units / inserted_units.
            expected_restoration_calls = (
                1 + (1 if learned_target_layers else 0)
                if stacked_key_bank is not None
                else target_layer_count
            )
            if restoration_calls != expected_restoration_calls:
                return _fail("target_count", ValueError("restoration_call_count_mismatch"))

            candidate_past_key_values = list(past_key_values[:target_start]) + candidate_targets
            if len(candidate_past_key_values) != len(self.block):
                return _fail("candidate_cache_length", ValueError("candidate_cache_length_invalid"))
            expected_deep_length = deep_cache_length_before + pending_token_count
            # Single-append candidates deliberately still hold the OLD self
            # cache length: the restored block is staged privately and only
            # enters the cache in the final old+pending+current publication.
            # The compatibility path keeps its pre-extended length.
            candidate_expected_deep_length = (
                deep_cache_length_before if stacked_key_bank is not None else expected_deep_length
            )
            for target_layer in range(target_start, target_end):
                layer_len = safe_cache_seq_len(candidate_past_key_values[target_layer])
                layer_len = 0 if layer_len is None else int(layer_len)
                if layer_len != candidate_expected_deep_length:
                    return _fail(
                        "candidate_cache_consistency:layer_{}".format(target_layer),
                        ValueError(
                            "layer_self_cache_length_mismatch:{}!={}".format(
                                layer_len, candidate_expected_deep_length
                            )
                        ),
                    )
            if stacked_key_bank is not None and len(staged_pending_self_kv) != target_layer_count:
                return _fail("target_count", ValueError("staged_pending_count_mismatch"))

            # A3: THE transaction-level finite validation. Every per-layer,
            # per-intermediate finite check deferred above converges here into
            # ONE host-visible decision, taken on the final candidate data
            # that is about to be committed, while past_key_values is still
            # completely untouched.
            #
            # The restored spans are the newly written positions only: the
            # [0, deep_cache_length_before) prefix came from the authoritative
            # cache and was already valid, and re-scanning it every flush is
            # what made this path quadratic in cache length.
            #
            # Nonfinite values cannot hide from this check. NaN propagates
            # through every op in the chain, and Inf either stays Inf or
            # becomes NaN (Inf*0, Inf-Inf); nothing in the hidden affine,
            # LayerNorm, K/V projections, K affine or V Procrustes can turn a
            # nonfinite intermediate back into a finite committed value. The
            # scan also runs after the output dtype cast, so a finite float32
            # value that overflows in a narrower cache dtype is caught too.
            try:
                aggregate = None
                if not finite_validation_enabled:
                    # Explicitly selected timing mode: no diagnostic scan
                    # feeds the aggregate, it stays None, and the decision
                    # below therefore requires no scalar readback either.
                    # (untrusted_cross is empty in this mode by construction,
                    # so the cross loop and the trust-marking loop below are
                    # no-ops.)
                    pass
                elif stacked_key_bank is not None:
                    # Native same-layer target (target_start): the staged
                    # restored block IS what the final single cat will
                    # publish verbatim (same dtype/device by the checks
                    # above), so validating it here keeps the required
                    # native transaction validation on the exact bytes that
                    # can become cache state.
                    native_key_block, native_value_block = staged_pending_self_kv[target_start]
                    aggregate = torch.isfinite(native_key_block).all() & torch.isfinite(
                        native_value_block
                    ).all()
                    # Learned targets: ONE device-side finite reduction for
                    # the whole learned K bank and ONE for the V bank --
                    # never one scan pair per learned target. These banks are
                    # the exact POST-CAST values destined for the cache: the
                    # manager cast them to the output/cache dtype before
                    # returning, each candidate extension was built with
                    # torch.zeros_like(<bank slice>) (same dtype/device, or
                    # the torch.cat above would already have failed), so the
                    # staging .to() was an identity and _commit copied these
                    # very values verbatim into the candidate spans.
                    learned_finite = torch.isfinite(stacked_key_bank).all() & torch.isfinite(
                        stacked_value_bank
                    ).all()
                    aggregate = aggregate & learned_finite
                else:
                    for offset, entry in enumerate(candidate_targets):
                        restored_key_span = entry[0].narrow(
                            2, deep_cache_length_before, pending_token_count
                        )
                        restored_value_span = entry[1].narrow(
                            2, deep_cache_length_before, pending_token_count
                        )
                        layer_finite = torch.isfinite(restored_key_span).all() & torch.isfinite(
                            restored_value_span
                        ).all()
                        aggregate = layer_finite if aggregate is None else (aggregate & layer_finite)
                        del offset
                for _layer, cross_key_ref, cross_value_ref in untrusted_cross:
                    cross_finite = torch.isfinite(cross_key_ref).all() & torch.isfinite(
                        cross_value_ref
                    ).all()
                    aggregate = cross_finite if aggregate is None else (aggregate & cross_finite)
                # The one and only scalar readback of this transaction.
                transaction_finite = True if aggregate is None else bool(aggregate.item())
            except Exception as exc:
                return _fail("transaction_finite_validation", exc)
            if not transaction_finite:
                return _fail(
                    "transaction_finite_validation",
                    ValueError("restored_kv_nonfinite"),
                )

            # Only now, with the whole transaction proven finite, may these
            # cross-attention tensors become trusted for later flushes of this
            # same generation.
            for target_layer, cross_key_ref, cross_value_ref in untrusted_cross:
                self._task_c2_batched_trust_cross_kv(target_layer, cross_key_ref, cross_value_ref)

            result["success"] = True
            result["past_key_values"] = candidate_past_key_values
            result["inserted_units"] = requested_units
            # Post-publication logical length (old + restored pending): on
            # the single-append path the candidate itself still holds the
            # old length, and this length is realized by the final combined
            # publication.
            result["deep_cache_length_after"] = expected_deep_length
            if stacked_key_bank is not None:
                result["staged_pending_self_kv"] = staged_pending_self_kv
            return result
        except Exception as exc:
            return _fail("unexpected_exception", exc)

    def _try_task_c2_fixed_source6_direct_insertion(
        self,
        source_layer,
        source_hidden,
        past_key_values,
        present_key_value_states,
        decoder_position,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        head_mask,
        cross_attn_head_mask,
        use_cache,
        output_attentions,
    ):
        """Attempt to immediately restore and atomically install exact-
        shaped K/V for every missing deep target layer
        (source_layer..len(self.block)-1) of one Native FREE
        fixed-source-layer-6 exiting token, using
        self.kv_runtime_restorer.restore_from_hidden() directly from the
        original source_hidden (h6) for EVERY target independently -- a
        restored target's own output is never chained into another
        target's input. No target decoder block is ever executed.

        Builds every candidate target-layer cache only in local temporary
        objects (reusing the existing atomic
        _stage_runtime_restored_kv_slices/_commit_runtime_restored_kv_slices
        helpers already used by the Task C1 exact-overwrite path) and
        returns a plain result dict; present_key_value_states/
        past_key_values are only ever read, never mutated, so any failure
        leaves the caller free to fall back to the existing Task C1
        pending-buffer path with no partial installation."""

        source_layer = int(source_layer)
        target_start = source_layer
        target_end = len(self.block)
        requested_units = max(0, target_end - target_start)
        result = {
            "success": False,
            "complete_cache": None,
            "requested_units": requested_units,
            "inserted_units": 0,
            "failure_stage": None,
            "failure_type": None,
            "failure_message": None,
            "target_layers": list(range(target_start, target_end)),
            "decoder_position": int(decoder_position) if decoder_position is not None else None,
        }

        def _fail(stage, exc):
            result["failure_stage"] = stage
            result["failure_type"] = type(exc).__name__
            result["failure_message"] = str(exc)
            return result

        try:
            if not use_cache:
                return _fail("precondition", ValueError("use_cache_false"))
            if len(self.stack_hidden_states):
                # A prior exit is still unresolved (pending Task C1 flush):
                # target-layer caches are not at decoder_position yet, so
                # direct insertion cannot be atomic against them here. This
                # token falls back to the existing pending-buffer path too,
                # joining the same later synchronized flush.
                return _fail("precondition", ValueError("pending_stack_nonempty"))
            if (
                source_hidden is None
                or source_hidden.ndim != 3
                or int(source_hidden.shape[0]) != 1
                or int(source_hidden.shape[1]) != 1
            ):
                return _fail(
                    "precondition",
                    ValueError(
                        "source_hidden_shape_invalid:{}".format(
                            list(source_hidden.shape) if source_hidden is not None else None
                        )
                    ),
                )
            if not torch.isfinite(source_hidden).all().item():
                return _fail("precondition", ValueError("source_hidden_nonfinite"))
            if decoder_position is None or int(decoder_position) < 0:
                return _fail("precondition", ValueError("decoder_position_unavailable"))
            if requested_units <= 0:
                return _fail("precondition", ValueError("no_missing_target_layers"))
            if past_key_values is None or len(past_key_values) != len(self.block):
                return _fail("precondition", ValueError("past_key_values_shape_invalid"))
            if present_key_value_states is None or len(present_key_value_states) < target_start:
                return _fail("precondition", ValueError("present_key_value_states_prefix_unavailable"))

            decoder_position = int(decoder_position)
            candidate_targets = []
            for target_layer in range(target_start, target_end):
                target_self_attention = self.block[target_layer].layer[0]
                with self._missing_kv_component_timer_obj().time_block(
                    "task_c2_restoration_compute_time_ms",
                    device=source_hidden.device,
                ):
                    restore_result = self.kv_runtime_restorer.restore_from_hidden(
                        source_hidden,
                        source_layer,
                        target_layer,
                        target_self_attention.layer_norm,
                        target_self_attention.SelfAttention.k,
                        target_self_attention.SelfAttention.v,
                        output_device=source_hidden.device,
                        output_dtype=source_hidden.dtype,
                        # Identical guarantee to the batched helper: this
                        # exiting token's h6 was finite-checked once above,
                        # before this loop. Immediate and Batched receive the
                        # same optimization so the arms stay comparable.
                        source_hidden_prevalidated=True,
                    )
                if restore_result.status != "ok":
                    return _fail(
                        "restore_from_hidden:layer_{}".format(target_layer),
                        ValueError("{}:{}".format(restore_result.status, restore_result.error_message)),
                    )
                restored_key = restore_result.restored_key
                restored_value = restore_result.restored_value
                # Read from self.config (guaranteed present -- already relied
                # on elsewhere in this class) rather than the projection
                # module's own attribute names, which are not guaranteed
                # across every SelfAttention implementation.
                expected_heads = int(self.config.num_heads)
                expected_d_kv = int(self.config.d_kv)
                if (
                    restored_key is None
                    or restored_value is None
                    or restored_key.shape != restored_value.shape
                    or int(restored_key.shape[0]) != 1
                    or int(restored_key.shape[1]) != expected_heads
                    or int(restored_key.shape[2]) != 1
                    or int(restored_key.shape[3]) != expected_d_kv
                ):
                    return _fail(
                        "restore_shape:layer_{}".format(target_layer),
                        ValueError(
                            "restored_kv_shape_invalid:{}".format(
                                list(restored_key.shape) if restored_key is not None else None
                            )
                        ),
                    )
                if not torch.isfinite(restored_key).all().item() or not torch.isfinite(restored_value).all().item():
                    return _fail("restore_nonfinite:layer_{}".format(target_layer), ValueError("restored_kv_nonfinite"))

                past_state = past_key_values[target_layer]
                self_key_ref = past_state[0] if (past_state is not None and len(past_state) > 0) else None
                self_value_ref = past_state[1] if (past_state is not None and len(past_state) > 1) else None
                past_self_len = int(self_key_ref.shape[2]) if self_key_ref is not None else 0
                if past_self_len != decoder_position:
                    return _fail(
                        "cache_position:layer_{}".format(target_layer),
                        ValueError(
                            "past_self_cache_length_mismatch:{}!={}".format(past_self_len, decoder_position)
                        ),
                    )

                empty_key = self_key_ref if self_key_ref is not None else restored_key.narrow(2, 0, 0)
                empty_value = self_value_ref if self_value_ref is not None else restored_value.narrow(2, 0, 0)
                extended_key = torch.cat([empty_key, torch.zeros_like(restored_key)], dim=2)
                extended_value = torch.cat([empty_value, torch.zeros_like(restored_value)], dim=2)
                with self._missing_kv_component_timer_obj().time_block(
                    "task_c2_cache_staging_time_ms",
                    device=source_hidden.device,
                ):
                    staged_key, staged_value, staging_error, staging_message = self._stage_runtime_restored_kv_slices(
                        restore_result, extended_key, extended_value, past_self_len,
                    )
                if staging_error is not None:
                    return _fail(
                        "cache_staging:layer_{}".format(target_layer),
                        ValueError("{}:{}".format(staging_error, staging_message)),
                    )
                try:
                    with self._missing_kv_component_timer_obj().time_block(
                        "task_c2_cache_commit_time_ms",
                        device=source_hidden.device,
                    ):
                        new_self_key, new_self_value = self._commit_runtime_restored_kv_slices(
                            extended_key,
                            extended_value,
                            past_self_len,
                            staged_key,
                            staged_value,
                            # Same private-temporary guarantee as the batched
                            # helper: Immediate and Batched share this
                            # optimization so the arms stay comparable.
                            candidate_is_private=True,
                        )
                except Exception as exc:
                    return _fail("cache_commit:layer_{}".format(target_layer), exc)
                if (
                    int(new_self_key.shape[2]) != decoder_position + 1
                    or int(new_self_value.shape[2]) != decoder_position + 1
                ):
                    return _fail(
                        "cache_length:layer_{}".format(target_layer),
                        ValueError("new_self_cache_length_invalid"),
                    )

                cross_key_ref = past_state[2] if (past_state is not None and len(past_state) > 2) else None
                cross_value_ref = past_state[3] if (past_state is not None and len(past_state) > 3) else None
                if cross_key_ref is None or cross_value_ref is None:
                    try:
                        cross_seed = self.block[target_layer].gen_cross_attn_key_value(
                            source_hidden,
                            attention_mask=None,
                            position_bias=None,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_extended_attention_mask,
                            encoder_decoder_position_bias=encoder_decoder_position_bias,
                            layer_head_mask=head_mask[target_layer] if head_mask is not None else None,
                            cross_attn_layer_head_mask=(
                                cross_attn_head_mask[target_layer] if cross_attn_head_mask is not None else None
                            ),
                            past_key_value=None,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )
                    except Exception as exc:
                        return _fail("cross_attention_init:layer_{}".format(target_layer), exc)
                    cross_key_ref, cross_value_ref = cross_seed[2], cross_seed[3]
                if (
                    cross_key_ref is None
                    or cross_value_ref is None
                    or not torch.isfinite(cross_key_ref).all().item()
                    or not torch.isfinite(cross_value_ref).all().item()
                ):
                    return _fail(
                        "cross_attention_init:layer_{}".format(target_layer),
                        ValueError("cross_kv_unavailable_or_nonfinite"),
                    )

                # A plain list, matching the existing per-layer present_key_value_state
                # convention produced by normal block forward() (list, not tuple).
                candidate_targets.append([new_self_key, new_self_value, cross_key_ref, cross_value_ref])

            if len(candidate_targets) != requested_units:
                return _fail("target_count", ValueError("candidate_target_count_mismatch"))

            complete_cache = list(present_key_value_states[:target_start]) + candidate_targets
            if len(complete_cache) != len(self.block):
                return _fail("candidate_cache_length", ValueError("candidate_cache_length_invalid"))
            for layer_idx, state in enumerate(complete_cache):
                layer_len = safe_cache_seq_len(state)
                if layer_len != decoder_position + 1:
                    return _fail(
                        "candidate_cache_consistency:layer_{}".format(layer_idx),
                        ValueError("layer_self_cache_length_mismatch:{}!={}".format(layer_len, decoder_position + 1)),
                    )

            result["success"] = True
            result["complete_cache"] = complete_cache
            result["inserted_units"] = requested_units
            return result
        except Exception as exc:
            return _fail("unexpected_exception", exc)

    def _exact_catchup_dump_dir(self):
        dump_dir = getattr(self.config, "kv_exact_catchup_dump_dir", None)
        if dump_dir:
            return dump_dir
        trace_path = getattr(self.config, "kv_trace_path", None)
        if trace_path:
            trace_parent = os.path.dirname(trace_path)
            if trace_parent:
                return os.path.join(trace_parent, "exact_kv_dumps")
        return None

    def _source_dump_dir(self):
        dump_dir = getattr(self.config, "kv_source_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._exact_catchup_dump_dir()

    def _adjacent_anchor_dump_dir(self):
        dump_dir = getattr(self.config, "kv_adjacent_anchor_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._source_dump_dir()

    def _all_layer_calib_dump_dir(self):
        dump_dir = getattr(self.config, "kv_all_layer_calib_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._exact_catchup_dump_dir()

    def _all_layer_hidden_dump_dir(self):
        dump_dir = getattr(self.config, "kv_all_layer_hidden_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._all_layer_calib_dump_dir()

    def _attention_diag_dump_dir(self):
        dump_dir = getattr(self.config, "kv_attention_diag_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._exact_catchup_dump_dir()

    def _full_attention_diag_dump_dir(self):
        dump_dir = getattr(self.config, "kv_full_attention_diag_dump_dir", None)
        if dump_dir:
            return dump_dir
        return self._exact_catchup_dump_dir()

    def _eaes_score_jsonl_path(self):
        path = getattr(self.config, "kv_restore_eaes_score_jsonl", None)
        if path:
            return path
        dump_dir = self._exact_catchup_dump_dir()
        if dump_dir:
            return os.path.join(dump_dir, "eaes_scores.jsonl")
        return None

    def _append_jsonl(self, path, row):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._json_safe(row), sort_keys=True) + "\n")

    def _append_eaes_score_row(self, row):
        if not getattr(self.config, "kv_restore_dump_eaes_scores", False):
            return False
        path = self._eaes_score_jsonl_path()
        if path is None:
            return False
        self._append_jsonl(path, row)
        return True

    def _eaes_v1_score_for_position(self, source_layer, position):
        result = {
            "eaes_score_version": "eaes_v1_received_attention",
            "eaes_score_definition": (
                "sum of accumulated received decoder self-attention mass at "
                "eligible early/source layers 0..source_layer"
            ),
            "eaes_received_attention_sum": None,
            "eaes_observation_count": 0,
            "eaes_score": None,
            "eligible_layer_count": 0,
            "eligible_layers": [],
            "score_status": "missing_eaes_evidence",
        }
        if source_layer is None or position is None:
            result["score_status"] = "missing_source_layer_or_position"
            return result
        if not hasattr(self, "kv_importance") or self.kv_importance is None or not self.kv_importance.enabled:
            result["score_status"] = "kv_importance_disabled"
            return result

        source_layer = int(source_layer)
        position = int(position)
        if source_layer < 0 or position < 0:
            result["score_status"] = "invalid_source_layer_or_position"
            return result

        eligible_layers = list(range(0, source_layer + 1))
        result["eligible_layers"] = eligible_layers
        result["eligible_layer_count"] = len(eligible_layers)
        total = 0.0
        observations = 0
        observed_layers = []
        for layer_idx in eligible_layers:
            scores = self.kv_importance.get_scores(layer_idx)
            if scores is None:
                continue
            try:
                score_len = int(scores.shape[0]) if hasattr(scores, "shape") else len(scores)
            except TypeError:
                continue
            if position >= score_len:
                continue
            try:
                value = scores[position]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                value = float(value)
            except Exception:
                continue
            total += value
            observations += 1
            observed_layers.append(layer_idx)

        result["eaes_received_attention_sum"] = total
        result["eaes_observation_count"] = observations
        result["observed_layers"] = observed_layers
        if observations > 0:
            result["eaes_score"] = total
            result["score_status"] = "ok"
        return result

    def _maybe_export_eaes_scores_for_pending(self, pending_metadata, source_records, flush_index=None, reason="flush"):
        if not getattr(self.config, "kv_restore_dump_eaes_scores", False):
            return {"enabled": False, "exported": 0, "skipped": 0}
        path = self._eaes_score_jsonl_path()
        if path is None:
            return {"enabled": True, "exported": 0, "skipped": 0, "reason": "missing_eaes_score_jsonl_path"}

        pending_metadata = tuple(pending_metadata or ())
        source_records = tuple(source_records or ())
        exported = 0
        skipped = 0
        pending_count = max(len(pending_metadata), len(source_records))
        for idx in range(pending_count):
            metadata = pending_metadata[idx] if idx < len(pending_metadata) else None
            source_record = source_records[idx] if idx < len(source_records) else None
            metadata_dict = metadata_to_trace_dict(metadata)
            if not isinstance(source_record, dict):
                source_record = {}
            source_record_id = source_record.get("source_record_id")
            if source_record_id is None:
                skipped += 1
                continue
            source_record_id = str(source_record_id)
            if source_record_id in self._eaes_exported_source_record_ids:
                skipped += 1
                continue

            source_layer = source_record.get("source_layer")
            if source_layer is None and metadata_dict is not None and metadata_dict.get("exit_layer") is not None:
                source_layer = int(metadata_dict.get("exit_layer")) - 1
            decoder_position = source_record.get("decoder_position")
            if decoder_position is None and metadata_dict is not None:
                decoder_position = metadata_dict.get("decoder_position")

            score_fields = self._eaes_v1_score_for_position(source_layer, decoder_position)
            row = {
                "source_record_id": source_record_id,
                "source_token_uid": source_record_id,
                "status": score_fields.get("score_status"),
                "generation_index": source_record.get("generation_index", self._generation_index),
                "sample_id": source_record.get("generation_index", self._generation_index),
                "flush_index": flush_index,
                "export_reason": reason,
                "token_index": decoder_position,
                "decoder_position": decoder_position,
                "source_layer": source_layer,
                "exit_layer": source_record.get("exit_layer"),
                "catchup_start_layer": source_record.get("catchup_start_layer"),
                "pending_relative_index": source_record.get("pending_relative_index"),
                "relative_index": source_record.get("relative_index"),
                "recency_rank": (pending_count - 1 - idx) if pending_count else None,
                "last_seen_step": decoder_position,
                "skipped_token_metadata": metadata_dict,
            }
            row.update(score_fields)
            self._append_eaes_score_row(row)
            self._eaes_exported_source_record_ids.add(source_record_id)
            exported += 1

        if hasattr(self, "kv_trace") and self.kv_trace.enabled:
            self.kv_trace.record(
                "eaes_score_export",
                status="ok",
                path=path,
                export_reason=reason,
                flush_index=flush_index,
                pending_record_count=pending_count,
                exported_count=exported,
                skipped_count=skipped,
                score_version="eaes_v1_received_attention",
            )
        return {"enabled": True, "exported": exported, "skipped": skipped, "path": path}

    def _append_exact_catchup_manifest(self, row):
        dump_dir = self._exact_catchup_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "manifest.jsonl"), row)

    def _append_source_kv_manifest(self, row):
        dump_dir = self._source_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        manifest_row = {
            key: value
            for key, value in dict(row).items()
            if not str(key).startswith("_runtime_")
        }
        self._append_jsonl(os.path.join(dump_dir, "source_manifest.jsonl"), manifest_row)

    def _append_adjacent_anchor_manifest(self, row):
        dump_dir = self._adjacent_anchor_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "adjacent_anchor_manifest.jsonl"), row)

    def _append_all_layer_calib_manifest(self, row):
        dump_dir = self._all_layer_calib_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "all_layer_kv_manifest.jsonl"), row)

    def _append_all_layer_hidden_manifest(self, row):
        dump_dir = self._all_layer_hidden_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "all_layer_hidden_manifest.jsonl"), row)

    def _append_attention_diag_manifest(self, row):
        dump_dir = self._attention_diag_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "attention_diag_manifest.jsonl"), row)

    def _append_full_attention_diag_manifest(self, row):
        dump_dir = self._full_attention_diag_dump_dir()
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        self._append_jsonl(os.path.join(dump_dir, "full_attention_diag_manifest.jsonl"), row)

    def _append_restoration_dryrun_manifest(self, row):
        if not getattr(self.config, "kv_restoration_dryrun_enabled", False):
            return
        max_records = int(getattr(self.config, "kv_restoration_dryrun_max_records", 100) or 0)
        if self._restoration_dryrun_record_count >= max_records:
            return
        path = getattr(self.config, "kv_restoration_dryrun_manifest_path", None)
        if path is None:
            dump_dir = self._exact_catchup_dump_dir()
            if dump_dir is None:
                return
            path = os.path.join(dump_dir, "restoration_dryrun_manifest.jsonl")
        row = dict(row)
        row.setdefault("dryrun_status", "restoration_utility_unavailable")
        row.setdefault(
            "message",
            "Runtime K_hat/V_hat construction is not wired into FREE exact catch-up yet; no restoration metrics were computed.",
        )
        self._append_jsonl(path, row)
        self._restoration_dryrun_record_count += 1

    def _parse_exact_catchup_dump_layers(self):
        layers = getattr(self.config, "kv_exact_catchup_dump_layers", None)
        if layers is None or str(layers).strip() == "":
            return None
        normalized = str(layers).strip()
        if normalized.lower() in {"all", "*"}:
            return None
        parsed = set()
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part))
        return parsed

    def _parse_attention_diag_dump_layers(self):
        layers = getattr(self.config, "kv_attention_diag_dump_layers", None)
        if layers is None or str(layers).strip() == "":
            return None
        normalized = str(layers).strip()
        if normalized.lower() in {"all", "*"}:
            return None
        parsed = set()
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part))
        return parsed

    def _parse_full_attention_diag_dump_layers(self):
        layers = getattr(self.config, "kv_full_attention_diag_dump_layers", None)
        if layers is None or str(layers).strip() == "":
            return None
        normalized = str(layers).strip()
        if normalized.lower() in {"all", "*"}:
            return None
        parsed = set()
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part))
        return parsed

    def _parse_all_layer_calib_dump_layers(self):
        layers = getattr(self.config, "kv_all_layer_calib_dump_layers", None)
        if layers is None or str(layers).strip() == "":
            return None
        normalized = str(layers).strip()
        if normalized.lower() in {"all", "*"}:
            return None
        parsed = set()
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part))
        return parsed

    def _parse_all_layer_hidden_dump_layers(self):
        layers = getattr(self.config, "kv_all_layer_hidden_dump_layers", None)
        if layers is None or str(layers).strip() == "":
            return None
        normalized = str(layers).strip()
        if normalized.lower() in {"all", "*"}:
            return None
        parsed = set()
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part))
        return parsed

    def _begin_exact_catchup_dump_flush(self, start_layer, pending_skipped_tokens):
        context = {
            "enabled": False,
            "reason": None,
            "flush_index": None,
            "dump_dir": None,
            "dumped_layers": 0,
            "requested_layers": None,
        }
        if not getattr(self.config, "kv_exact_catchup_dump_enabled", False):
            context["reason"] = "disabled"
            return context
        pending_skipped_tokens = int(pending_skipped_tokens or 0)
        if pending_skipped_tokens <= 0:
            context["reason"] = "no_pending_tokens"
            return context
        dump_dir = self._exact_catchup_dump_dir()
        if dump_dir is None:
            context["reason"] = "missing_dump_dir"
            return context
        max_flushes = int(getattr(self.config, "kv_exact_catchup_dump_max_flushes", 4) or 0)
        if self._exact_catchup_dump_next_flush_index >= max_flushes:
            context["reason"] = "max_flushes_reached"
            return context
        os.makedirs(dump_dir, exist_ok=True)
        context.update(
            {
                "enabled": True,
                "reason": "enabled",
                "flush_index": self._exact_catchup_dump_next_flush_index,
                "dump_dir": dump_dir,
                "requested_layers": self._parse_exact_catchup_dump_layers(),
            }
        )
        self._exact_catchup_dump_next_flush_index += 1
        return context

    def _convert_exact_catchup_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_exact_catchup_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_exact_catchup_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_exact_catchup_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_source_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_source_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_source_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_source_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_adjacent_anchor_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_adjacent_anchor_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_adjacent_anchor_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_adjacent_anchor_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_all_layer_calib_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_all_layer_calib_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_all_layer_calib_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_all_layer_calib_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_all_layer_hidden_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_all_layer_hidden_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_all_layer_hidden_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_all_layer_hidden_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_attention_diag_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_attention_diag_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_attention_diag_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_attention_diag_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _convert_full_attention_diag_dump_tensor(self, tensor):
        tensor = tensor.detach()
        dtype_name = getattr(self.config, "kv_full_attention_diag_dump_dtype", "float16")
        if dtype_name == "float16":
            tensor = tensor.to(dtype=torch.float16)
        elif dtype_name == "bfloat16":
            tensor = tensor.to(dtype=torch.bfloat16)
        elif dtype_name == "float32":
            tensor = tensor.to(dtype=torch.float32)
        elif dtype_name != "original":
            raise ValueError("Unsupported kv_full_attention_diag_dump_dtype: {}".format(dtype_name))
        if getattr(self.config, "kv_full_attention_diag_dump_cpu", True):
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _pending_slice_for_exact_catchup_dump(self, key_tensor, past_key_value, pending_skipped_tokens):
        pending_skipped_tokens = int(pending_skipped_tokens or 0)
        total_len = int(key_tensor.shape[2])
        past_len = safe_cache_seq_len(past_key_value)
        if past_len is not None and total_len >= int(past_len) + pending_skipped_tokens:
            start = int(past_len)
            return start, start + pending_skipped_tokens, "past_plus_new", int(past_len)
        if total_len >= pending_skipped_tokens:
            return 0, pending_skipped_tokens, "new_only", past_len
        return None, None, "unknown", past_len

    def _source_slice_for_skip_dump(self, key_tensor, past_key_value):
        total_len = int(key_tensor.shape[2])
        past_len = safe_cache_seq_len(past_key_value)
        if past_len is not None and total_len >= int(past_len) + 1:
            start = int(past_len)
            return start, start + 1, "past_plus_new", int(past_len)
        if total_len >= 1:
            return total_len - 1, total_len, "last_token", past_len
        return None, None, "unknown", past_len

    def _new_token_slice_for_all_layer_calib_dump(self, key_tensor, past_key_value, logical_past_len=None):
        total_len = int(key_tensor.shape[2])
        # ``logical_past_len`` (default None: behavior identical for every
        # existing caller) carries the LOGICAL past length when the
        # single-append Batched path staged restored pending K/V privately:
        # the physical past_key_value then deliberately still holds only the
        # old cache, but the pending positions are NOT newly generated by
        # this forward and must not be classified as such.
        past_len = (
            int(logical_past_len) if logical_past_len is not None else safe_cache_seq_len(past_key_value)
        )
        if past_len is not None and total_len >= int(past_len):
            start = int(past_len)
            return start, total_len, "past_plus_new", int(past_len)
        if total_len >= 1:
            return 0, total_len, "new_only", past_len
        return None, None, "unknown", past_len

    def _all_layer_calib_record_id(self, generation_index, decoder_position, layer_idx):
        return "gen{}_pos{}_layer{}".format(
            generation_index,
            decoder_position if decoder_position is not None else "none",
            layer_idx if layer_idx is not None else "none",
        )

    def _maybe_dump_all_layer_calib_kv(
        self,
        layer_idx,
        present_key_value_state,
        past_key_value,
        skip_mask=False,
        logical_past_len=None,
    ):
        if not (self.is_decoder and getattr(self.config, "kv_all_layer_calib_dump_enabled", False)):
            return
        requested_layers = self._parse_all_layer_calib_dump_layers()
        if requested_layers is not None and int(layer_idx) not in requested_layers:
            return
        dump_dir = self._all_layer_calib_dump_dir()
        row_base = {
            "record_type": "all_layer_calib_kv",
            "generation_index": self._generation_index,
            "layer_idx": int(layer_idx),
            "token_source": "decoder_self_attention_present_key_value_state",
            "full_depth_execution": bool(
                not self.use_shallow_deep
                and not self.use_early_exit
                and self.config.static_exit_layer is None
                and not skip_mask
            ),
            "use_shallow_deep": bool(self.use_shallow_deep),
            "use_early_exit": bool(self.use_early_exit),
            "static_exit_layer": self.config.static_exit_layer,
            "skip_mask": bool(skip_mask),
        }
        row_base.update(self._missing_kv_sample_context_fields())
        if dump_dir is None:
            row = dict(row_base)
            row.update({"dump_succeeded": False, "skip_reason": "missing_dump_dir"})
            self._append_all_layer_calib_manifest(row)
            return
        max_tokens = int(getattr(self.config, "kv_all_layer_calib_dump_max_tokens", 128) or 0)
        try:
            if present_key_value_state is None or len(present_key_value_state) < 2:
                raise ValueError("present_key_value_state_missing_self_attention_kv")
            key_tensor = present_key_value_state[0]
            value_tensor = present_key_value_state[1]
            if key_tensor is None or value_tensor is None:
                raise ValueError("self_attention_kv_is_none")
            if len(key_tensor.shape) < 4 or len(value_tensor.shape) < 4:
                raise ValueError("unexpected_kv_rank")
            if key_tensor.shape != value_tensor.shape:
                raise ValueError("key_value_shape_mismatch")
            start, end, slice_mode, past_len = self._new_token_slice_for_all_layer_calib_dump(
                key_tensor, past_key_value, logical_past_len=logical_past_len
            )
            if start is None or end is None or end <= start:
                raise ValueError("could_not_infer_new_token_slice")
            os.makedirs(dump_dir, exist_ok=True)
            for token_offset, token_position in enumerate(range(int(start), int(end))):
                token_id = (int(self._generation_index), int(token_position))
                if token_id not in self._all_layer_calib_dumped_token_ids:
                    if max_tokens > 0 and len(self._all_layer_calib_dumped_token_ids) >= max_tokens:
                        row = dict(row_base)
                        row.update(
                            {
                                "dump_succeeded": False,
                                "skip_reason": "max_tokens_reached",
                                "decoder_position": int(token_position),
                                "token_index": int(token_position),
                                "slice_mode": slice_mode,
                                "self_attn_past_len_before_layer": past_len,
                            }
                        )
                        self._attach_missing_kv_dump_row_uid(row)
                        self._append_all_layer_calib_manifest(row)
                        continue
                    self._all_layer_calib_dumped_token_ids.add(token_id)
                record_id = self._all_layer_calib_record_id(self._generation_index, token_position, layer_idx)
                key_slice = self._convert_all_layer_calib_dump_tensor(key_tensor[:, :, token_position : token_position + 1, :])
                value_slice = self._convert_all_layer_calib_dump_tensor(value_tensor[:, :, token_position : token_position + 1, :])
                if self._missing_kv_packed_generation_enabled():
                    self._accumulate_packed_kv_slice(
                        layer_idx=layer_idx,
                        token_position=token_position,
                        token_offset=token_offset,
                        past_len=past_len,
                        key_slice=key_slice,
                        value_slice=value_slice,
                        device_before_dump=str(key_tensor.device),
                    )
                    self._all_layer_calib_dumped_records += 1
                    continue
                file_name = "all_layer_{}.pt".format(record_id)
                file_path = os.path.join(dump_dir, file_name)
                row = dict(row_base)
                row.update(
                    {
                        "dump_succeeded": True,
                        "record_id": record_id,
                        "file_path": file_path,
                        "generation_index": self._generation_index,
                        "decoder_position": int(token_position),
                        "token_index": int(token_position),
                        "token_offset_in_forward": int(token_offset),
                        "dtype": str(key_slice.dtype).replace("torch.", ""),
                        "device_before_dump": str(key_tensor.device),
                        "slice_mode": slice_mode,
                        "self_attn_past_len_before_layer": past_len,
                        "present_key_shape": list(key_tensor.shape),
                        "present_value_shape": list(value_tensor.shape),
                        "key_shape": list(key_slice.shape),
                        "value_shape": list(value_slice.shape),
                        "model_num_decoder_layers": len(self.block),
                        "model_num_heads": int(getattr(self.config, "num_heads", 0)),
                        "model_d_kv": int(getattr(self.config, "d_kv", 0)),
                    }
                )
                self._attach_missing_kv_dump_row_uid(row)
                payload = {
                    "key": key_slice,
                    "value": value_slice,
                    "metadata": dict(row),
                }
                torch.save(payload, file_path)
                self._all_layer_calib_dumped_records += 1
                self._append_all_layer_calib_manifest(row)
        except Exception as exc:
            if self._missing_kv_packed_generation_enabled():
                self.abort_missing_kv_generation_dump(reason=str(exc))
                raise
            row = dict(row_base)
            row.update({"dump_succeeded": False, "error_message": str(exc)})
            self._append_all_layer_calib_manifest(row)

    def _write_all_layer_hidden_dump_summary(self, dump_dir):
        if dump_dir is None:
            return
        summary = {
            "record_type": "all_layer_hidden_dump_summary",
            "dump_dir": dump_dir,
            "dumped_records": int(self._all_layer_hidden_dumped_records),
            "packed_generation_records": int(self._all_layer_hidden_packed_records),
            "storage_format": self._missing_kv_dump_storage_format(),
            "unique_token_positions": len(self._all_layer_hidden_dumped_token_ids),
            "failure_rows": int(self._all_layer_hidden_dump_failures),
            "include_raw_hidden": bool(getattr(self.config, "kv_all_layer_hidden_dump_include_raw_hidden", False)),
            "include_normed_hidden": bool(getattr(self.config, "kv_all_layer_hidden_dump_include_normed_hidden", True)),
            "dtype": getattr(self.config, "kv_all_layer_hidden_dump_dtype", "float16"),
            "cpu": bool(getattr(self.config, "kv_all_layer_hidden_dump_cpu", True)),
            "max_tokens": getattr(self.config, "kv_all_layer_hidden_dump_max_tokens", 128),
            "max_flushes": getattr(self.config, "kv_all_layer_hidden_dump_max_flushes", None),
            "requested_layers": getattr(self.config, "kv_all_layer_hidden_dump_layers", None),
        }
        path = os.path.join(dump_dir, "all_layer_hidden_dump_summary.json")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self._json_safe(summary), indent=2, sort_keys=True))

    def _maybe_dump_all_layer_hidden_states(
        self,
        layer_idx,
        raw_hidden_states,
        normed_hidden_states,
        past_key_value,
        skip_mask=False,
        logical_past_len=None,
    ):
        if not (self.is_decoder and getattr(self.config, "kv_all_layer_hidden_dump_enabled", False)):
            return
        requested_layers = self._parse_all_layer_hidden_dump_layers()
        if requested_layers is not None and int(layer_idx) not in requested_layers:
            return
        max_flushes = getattr(self.config, "kv_all_layer_hidden_dump_max_flushes", None)
        if max_flushes is not None and int(max_flushes) > 0 and int(self._generation_index) >= int(max_flushes):
            return
        dump_dir = self._all_layer_hidden_dump_dir()
        include_raw = bool(getattr(self.config, "kv_all_layer_hidden_dump_include_raw_hidden", False))
        include_normed = bool(getattr(self.config, "kv_all_layer_hidden_dump_include_normed_hidden", True))
        row_base = {
            "record_type": "all_layer_hidden_state",
            "generation_index": self._generation_index,
            "layer_idx": int(layer_idx),
            "token_source": "decoder_self_attention_hidden_state_before_kv_projection",
            "full_depth_execution": bool(
                not self.use_shallow_deep
                and not self.use_early_exit
                and self.config.static_exit_layer is None
                and not skip_mask
            ),
            "use_shallow_deep": bool(self.use_shallow_deep),
            "use_early_exit": bool(self.use_early_exit),
            "static_exit_layer": self.config.static_exit_layer,
            "skip_mask": bool(skip_mask),
            "include_raw_hidden": include_raw,
            "include_normed_hidden": include_normed,
            "model_name_or_path": getattr(self.config, "_name_or_path", None),
        }
        row_base.update(self._missing_kv_sample_context_fields())
        if dump_dir is None:
            row = dict(row_base)
            row.update({"dump_succeeded": False, "skip_reason": "missing_dump_dir"})
            self._all_layer_hidden_dump_failures += 1
            self._append_all_layer_hidden_manifest(row)
            return
        max_tokens = int(getattr(self.config, "kv_all_layer_hidden_dump_max_tokens", 128) or 0)
        try:
            if not include_raw and not include_normed:
                raise ValueError("no_hidden_tensors_requested")
            hidden_for_shape = normed_hidden_states if normed_hidden_states is not None else raw_hidden_states
            if hidden_for_shape is None:
                raise ValueError("hidden_state_unavailable")
            if len(hidden_for_shape.shape) != 3:
                raise ValueError("unexpected_hidden_rank")
            seq_len = int(hidden_for_shape.shape[1])
            # Same logical-position contract as the K/V calib dump above: the
            # hidden tensor itself is untouched, only its token-position
            # label honors privately staged pending K/V.
            past_len = (
                int(logical_past_len)
                if logical_past_len is not None
                else safe_cache_seq_len(past_key_value)
            )
            token_start = int(past_len) if past_len is not None else 0
            os.makedirs(dump_dir, exist_ok=True)
            for token_offset in range(seq_len):
                token_position = token_start + int(token_offset)
                token_id = (int(self._generation_index), int(token_position))
                if token_id not in self._all_layer_hidden_dumped_token_ids:
                    if max_tokens > 0 and len(self._all_layer_hidden_dumped_token_ids) >= max_tokens:
                        row = dict(row_base)
                        row.update(
                            {
                                "dump_succeeded": False,
                                "skip_reason": "max_tokens_reached",
                                "decoder_position": int(token_position),
                                "token_index": int(token_position),
                                "token_offset_in_forward": int(token_offset),
                                "self_attn_past_len_before_layer": past_len,
                            }
                        )
                        self._attach_missing_kv_dump_row_uid(row)
                        self._all_layer_hidden_dump_failures += 1
                        self._append_all_layer_hidden_manifest(row)
                        continue
                    self._all_layer_hidden_dumped_token_ids.add(token_id)
                record_id = self._all_layer_calib_record_id(self._generation_index, token_position, layer_idx)
                payload = {"metadata": None}
                row = dict(row_base)
                row.update(
                    {
                        "dump_succeeded": True,
                        "record_id": record_id,
                        "file_path": os.path.join(dump_dir, "all_layer_hidden_{}.pt".format(record_id)),
                        "generation_index": self._generation_index,
                        "decoder_position": int(token_position),
                        "token_index": int(token_position),
                        "token_offset_in_forward": int(token_offset),
                        "self_attn_past_len_before_layer": past_len,
                        "raw_hidden_shape": None,
                        "normed_hidden_shape": None,
                        "hidden_input_shape": list(hidden_for_shape.shape),
                        "model_num_decoder_layers": len(self.block),
                        "model_d_model": int(getattr(self.config, "d_model", 0)),
                    }
                )
                dtype_name = None
                if include_raw and raw_hidden_states is not None:
                    raw_slice = self._convert_all_layer_hidden_dump_tensor(
                        raw_hidden_states[:, token_offset : token_offset + 1, :]
                    )
                    payload["raw_hidden_state"] = raw_slice
                    row["raw_hidden_shape"] = list(raw_slice.shape)
                    dtype_name = str(raw_slice.dtype).replace("torch.", "")
                if include_normed and normed_hidden_states is not None:
                    normed_slice = self._convert_all_layer_hidden_dump_tensor(
                        normed_hidden_states[:, token_offset : token_offset + 1, :]
                    )
                    payload["normed_hidden_state"] = normed_slice
                    row["normed_hidden_shape"] = list(normed_slice.shape)
                    dtype_name = dtype_name or str(normed_slice.dtype).replace("torch.", "")
                row["dtype"] = dtype_name
                row["device_before_dump"] = str(hidden_for_shape.device)
                self._attach_missing_kv_dump_row_uid(row)
                if self._missing_kv_packed_generation_enabled():
                    self._accumulate_packed_hidden_slice(
                        layer_idx=layer_idx,
                        token_position=token_position,
                        token_offset=token_offset,
                        past_len=past_len,
                        raw_slice=payload.get("raw_hidden_state"),
                        normed_slice=payload.get("normed_hidden_state"),
                        device_before_dump=str(hidden_for_shape.device),
                    )
                    self._all_layer_hidden_dumped_records += 1
                    continue
                payload["metadata"] = dict(row)
                torch.save(payload, row["file_path"])
                self._all_layer_hidden_dumped_records += 1
                self._append_all_layer_hidden_manifest(row)
            self._write_all_layer_hidden_dump_summary(dump_dir)
        except Exception as exc:
            if self._missing_kv_packed_generation_enabled():
                self.abort_missing_kv_generation_dump(reason=str(exc))
                raise
            row = dict(row_base)
            row.update({"dump_succeeded": False, "error_message": str(exc)})
            self._all_layer_hidden_dump_failures += 1
            self._append_all_layer_hidden_manifest(row)
            self._write_all_layer_hidden_dump_summary(dump_dir)

    def _source_record_id(self, generation_index, decoder_position, relative_index, source_layer):
        return "gen{}_pos{}_rel{}_src{}".format(
            generation_index,
            decoder_position if decoder_position is not None else "none",
            relative_index if relative_index is not None else "none",
            source_layer if source_layer is not None else "none",
        )

    def _infer_decoder_position_from_source_present_kv(self, present_key_value_states, source_layer):
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
        except Exception:
            return None

    def _maybe_dump_adjacent_anchor_for_skip(
        self,
        source_record_row,
        source_hidden_states,
        source_layer,
        exit_layer,
        decoder_position,
        relative_index,
    ):
        if not getattr(self.config, "kv_adjacent_anchor_dump_enabled", False):
            return None
        source_record_id = source_record_row.get("source_record_id") if isinstance(source_record_row, dict) else None
        anchor_layer = None
        try:
            anchor_layer = int(source_layer) + 1 if source_layer is not None else None
        except (TypeError, ValueError):
            anchor_layer = None
        row = {
            "record_type": "adjacent_anchor_kv",
            "source_record_id": source_record_id,
            "source_token_uid": source_record_id,
            "generation_index": self._generation_index,
            "decoder_position": decoder_position,
            "token_index": decoder_position,
            "pending_relative_index": relative_index,
            "relative_index": relative_index,
            "exit_layer": exit_layer,
            "source_layer": source_layer,
            "anchor_layer": anchor_layer,
            "anchor_source": "exact_adjacent_projection",
            "projection_only": True,
            "projection_scope": "self_attention_key_value_projection_only",
            "projection_note": (
                "Anchor K/V are produced by applying the adjacent layer self-attention "
                "layer norm and K/V projections to the exit hidden state; attention, "
                "output projection, cross-attention, and FFN are not executed."
            ),
        }
        dump_dir = self._adjacent_anchor_dump_dir()
        if dump_dir is None:
            row.update({"dump_succeeded": False, "skip_reason": "missing_dump_dir"})
            self._append_adjacent_anchor_manifest(row)
            return row
        max_tokens = int(getattr(self.config, "kv_adjacent_anchor_dump_max_tokens", 128) or 0)
        if self._adjacent_anchor_dumped_tokens >= max_tokens:
            row.update({"dump_succeeded": False, "skip_reason": "max_tokens_reached"})
            self._append_adjacent_anchor_manifest(row)
            return row
        try:
            if source_record_id is None:
                raise ValueError("source_record_id_unavailable")
            if anchor_layer is None or anchor_layer < 0 or anchor_layer >= len(self.block):
                raise ValueError("anchor_layer_unavailable")
            if source_hidden_states is None:
                raise ValueError("source_hidden_state_unavailable")
            if len(source_hidden_states.shape) != 3:
                raise ValueError("unexpected_source_hidden_rank: {}".format(list(source_hidden_states.shape)))
            hidden_seq_len = int(source_hidden_states.shape[1])
            if hidden_seq_len <= 0:
                raise ValueError("source_hidden_seq_len_zero")
            if hidden_seq_len == 1:
                hidden_slice = source_hidden_states
                hidden_slice_mode = "single_token"
                hidden_slice_start = 0
                hidden_slice_end = 1
            else:
                hidden_slice = source_hidden_states[:, -1:, :]
                hidden_slice_mode = "last_token"
                hidden_slice_start = hidden_seq_len - 1
                hidden_slice_end = hidden_seq_len

            adjacent_self_attn_layer = self.block[anchor_layer].layer[0]
            adjacent_attention = adjacent_self_attn_layer.SelfAttention
            with torch.no_grad():
                normed_hidden = adjacent_self_attn_layer.layer_norm(hidden_slice)
                key_proj = adjacent_attention.k(normed_hidden)
                value_proj = adjacent_attention.v(normed_hidden)
                batch_size = int(key_proj.shape[0])
                token_count = int(key_proj.shape[1])
                num_heads = int(adjacent_attention.n_heads)
                head_dim = int(adjacent_attention.key_value_proj_dim)
                anchor_key = key_proj.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2).contiguous()
                anchor_value = value_proj.view(batch_size, token_count, num_heads, head_dim).transpose(1, 2).contiguous()

            anchor_key_dump = self._convert_adjacent_anchor_dump_tensor(anchor_key)
            anchor_value_dump = self._convert_adjacent_anchor_dump_tensor(anchor_value)
            os.makedirs(dump_dir, exist_ok=True)
            anchor_record_id = "anchor_{}_layer_{:02d}".format(source_record_id, anchor_layer)
            file_name = "{}.pt".format(anchor_record_id)
            file_path = os.path.join(dump_dir, file_name)
            row.update(
                {
                    "dump_succeeded": True,
                    "anchor_record_id": anchor_record_id,
                    "file_path": file_path,
                    "anchor_kv_file_path": file_path,
                    "dtype": str(anchor_key_dump.dtype).replace("torch.", ""),
                    "device_before_dump": str(source_hidden_states.device),
                    "source_hidden_shape": list(source_hidden_states.shape),
                    "hidden_slice_mode": hidden_slice_mode,
                    "hidden_slice_start": hidden_slice_start,
                    "hidden_slice_end": hidden_slice_end,
                    "anchor_key_shape": list(anchor_key.shape),
                    "anchor_value_shape": list(anchor_value.shape),
                    "anchor_tokens_dumped": int(anchor_key.shape[2]),
                }
            )
            payload = {
                "anchor_key": anchor_key_dump,
                "anchor_value": anchor_value_dump,
                "metadata": dict(row),
            }
            torch.save(payload, file_path)
            self._adjacent_anchor_dumped_tokens += int(anchor_key.shape[2])
            self._append_adjacent_anchor_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "adjacent_anchor_kv_dump",
                    dump_succeeded=True,
                    source_record_id=source_record_id,
                    anchor_record_id=anchor_record_id,
                    generation_index=self._generation_index,
                    decoder_position=decoder_position,
                    source_layer=source_layer,
                    anchor_layer=anchor_layer,
                    file_path=file_path,
                )
            return row
        except Exception as exc:
            row.update({"dump_succeeded": False, "error_message": str(exc)})
            self._append_adjacent_anchor_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "adjacent_anchor_kv_dump",
                    dump_succeeded=False,
                    source_record_id=source_record_id,
                    generation_index=self._generation_index,
                    decoder_position=decoder_position,
                    source_layer=source_layer,
                    anchor_layer=anchor_layer,
                    error_message=str(exc),
                )
            return row

    def _maybe_dump_source_kv_for_skip(
        self,
        present_key_value_states,
        past_key_values,
        source_layer,
        exit_layer,
        skipped_metadata,
        decoder_position,
        confidence,
        source_hidden_states=None,
    ):
        metadata_dict = metadata_to_trace_dict(skipped_metadata)
        decoder_position_source = "past_key_values" if decoder_position is not None else "missing"
        decoder_position_fallback_used = False
        if decoder_position is None:
            fallback_position = self._infer_decoder_position_from_source_present_kv(
                present_key_value_states,
                source_layer,
            )
            if fallback_position is not None:
                decoder_position = fallback_position
                decoder_position_source = "source_present_key_value_fallback"
                decoder_position_fallback_used = True
        if metadata_dict is not None and decoder_position is not None:
            metadata_dict = dict(metadata_dict)
            metadata_dict["decoder_position"] = decoder_position
        relative_index = metadata_dict.get("relative_index") if metadata_dict else None
        record_id = self._source_record_id(
            self._generation_index,
            decoder_position,
            relative_index,
            source_layer,
        )
        row = {
            "record_type": "source_kv",
            "source_record_id": record_id,
            "source_token_uid": record_id,
            "generation_index": self._generation_index,
            "decoder_position": decoder_position,
            "pending_relative_index": relative_index,
            "relative_index": relative_index,
            "exit_layer": exit_layer,
            "catchup_start_layer": exit_layer,
            "source_layer": source_layer,
            "decoder_position_source": decoder_position_source,
            "decoder_position_fallback_used": decoder_position_fallback_used,
            "token_index": decoder_position,
            "confidence": confidence,
            "skipped_token_metadata": metadata_dict,
            "kv_source": "free_source_layer_before_skip",
        }
        anchor_row = self._maybe_dump_adjacent_anchor_for_skip(
            row,
            source_hidden_states,
            source_layer,
            exit_layer,
            decoder_position,
            relative_index,
        )
        if isinstance(anchor_row, dict):
            row.update(
                {
                    "adjacent_anchor_dump_succeeded": anchor_row.get("dump_succeeded"),
                    "adjacent_anchor_record_id": anchor_row.get("anchor_record_id"),
                    "adjacent_anchor_file_path": anchor_row.get("file_path"),
                    "adjacent_anchor_layer": anchor_row.get("anchor_layer"),
                    "adjacent_anchor_skip_reason": anchor_row.get("skip_reason"),
                    "adjacent_anchor_error_message": anchor_row.get("error_message"),
                }
            )
        self._attach_runtime_source_kv_for_skip(
            row,
            present_key_value_states,
            past_key_values,
            source_layer,
        )
        if not getattr(self.config, "kv_source_dump_enabled", False):
            row.update({"dump_succeeded": False, "skip_reason": "disabled"})
            return row
        dump_dir = self._source_dump_dir()
        if dump_dir is None:
            row.update({"dump_succeeded": False, "skip_reason": "missing_dump_dir"})
            self._append_source_kv_manifest(row)
            return row
        max_tokens = int(getattr(self.config, "kv_source_dump_max_tokens", 128) or 0)
        if self._source_kv_dumped_tokens >= max_tokens:
            row.update({"dump_succeeded": False, "skip_reason": "max_tokens_reached"})
            self._append_source_kv_manifest(row)
            return row
        try:
            if source_layer is None or int(source_layer) < 0:
                raise ValueError("source_layer_unavailable")
            source_layer = int(source_layer)
            if present_key_value_states is None or len(present_key_value_states) <= source_layer:
                raise ValueError("source_present_key_value_state_unavailable")
            source_past = past_key_values[source_layer] if past_key_values is not None and len(past_key_values) > source_layer else None
            source_state = present_key_value_states[source_layer]
            if source_state is None or len(source_state) < 2:
                raise ValueError("source_state_missing_self_attention_kv")
            key_tensor = source_state[0]
            value_tensor = source_state[1]
            if key_tensor is None or value_tensor is None:
                raise ValueError("source_self_attention_kv_is_none")
            start, end, slice_mode, past_len = self._source_slice_for_skip_dump(key_tensor, source_past)
            if start is None:
                raise ValueError("could_not_infer_source_token_slice")

            source_key = self._convert_source_dump_tensor(key_tensor[:, :, start:end, :])
            source_value = self._convert_source_dump_tensor(value_tensor[:, :, start:end, :])
            os.makedirs(dump_dir, exist_ok=True)
            file_name = "source_{}_layer_{:02d}.pt".format(record_id, source_layer)
            file_path = os.path.join(dump_dir, file_name)
            row.update(
                {
                    "dump_succeeded": True,
                    "file_path": file_path,
                    "source_kv_file_path": file_path,
                    "dtype": str(source_key.dtype).replace("torch.", ""),
                    "device_before_dump": str(key_tensor.device),
                    "slice_mode": slice_mode,
                    "self_attn_past_len_before_layer": past_len,
                    "present_key_shape": list(key_tensor.shape),
                    "present_value_shape": list(value_tensor.shape),
                    "source_slice_start": start,
                    "source_slice_end": end,
                    "source_tokens_dumped": int(end - start),
                }
            )
            payload_metadata = {
                key: value
                for key, value in dict(row).items()
                if not str(key).startswith("_runtime_")
            }
            payload = {
                "source_key": source_key,
                "source_value": source_value,
                "metadata": payload_metadata,
            }
            torch.save(payload, file_path)
            self._source_kv_dumped_tokens += 1
            self._append_source_kv_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "source_kv_dump",
                    dump_succeeded=True,
                    source_record_id=record_id,
                    generation_index=self._generation_index,
                    decoder_position=decoder_position,
                    source_layer=source_layer,
                    exit_layer=exit_layer,
                    file_path=file_path,
                    slice_mode=slice_mode,
                )
            return row
        except Exception as exc:
            row.update({"dump_succeeded": False, "error_message": str(exc)})
            self._append_source_kv_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "source_kv_dump",
                    dump_succeeded=False,
                    source_record_id=record_id,
                    generation_index=self._generation_index,
                    decoder_position=decoder_position,
                    source_layer=source_layer,
                    exit_layer=exit_layer,
                    error_message=str(exc),
            )
            return row

    def _load_source_kv_for_attention_diag(self, source_file_paths, expected_count):
        source_keys = []
        source_values = []
        loaded_paths = []
        for path in list(source_file_paths or [])[:expected_count]:
            if not path:
                continue
            payload = torch.load(path, map_location="cpu")
            source_key = payload.get("source_key")
            source_value = payload.get("source_value")
            if source_key is None or source_value is None:
                raise ValueError("source payload missing source_key/source_value: {}".format(path))
            source_keys.append(source_key)
            source_values.append(source_value)
            loaded_paths.append(path)
        if len(source_keys) != expected_count:
            raise ValueError("source_kv_count_mismatch expected={} loaded={}".format(expected_count, len(source_keys)))
        return torch.cat(source_keys, dim=2), torch.cat(source_values, dim=2), loaded_paths

    def _query_slice_for_attention_diag(self, query_states, dumped_pending):
        if query_states is None:
            raise ValueError("query_unavailable")
        if len(query_states.shape) != 4:
            raise ValueError("unexpected_query_rank: {}".format(list(query_states.shape)))
        query_len = int(query_states.shape[2])
        if query_len <= 0:
            raise ValueError("query_length_zero")
        query_slice_start = 0
        if query_len >= int(dumped_pending):
            query_slice_end = int(dumped_pending)
            query_slice_mode = "pending_prefix"
            metric_scope = "restricted_pending_token_block"
        elif query_len == 1:
            query_slice_end = 1
            query_slice_mode = "consumer_current_token"
            metric_scope = "restricted_consumer_to_missing_block"
        else:
            query_slice_end = query_len
            query_slice_mode = "short_query_all"
            metric_scope = "restricted_short_query_to_missing_block"
        query = query_states[:, :, query_slice_start:query_slice_end, :]
        return query, query_slice_mode, query_slice_start, query_slice_end, metric_scope

    def _maybe_dump_attention_diag(
        self,
        layer_idx,
        flush_index,
        query_states,
        exact_key,
        exact_value,
        source_file_paths,
        source_record_ids,
        source_layer,
        target_slice_mode,
        pending_skipped_tokens,
        dumped_pending,
        metadata_positions,
        row_base,
    ):
        if not getattr(self.config, "kv_attention_diag_dump_enabled", False):
            return
        dump_dir = self._attention_diag_dump_dir()
        row = {
            "record_type": "attention_diag",
            "status": "pending",
            "flush_index": flush_index,
            "source_layer": source_layer,
            "target_layer": int(layer_idx),
            "layer_gap": int(layer_idx) - int(source_layer) if source_layer is not None else None,
            "source_record_ids": list(source_record_ids or []),
            "pending_decoder_positions": metadata_positions,
            "pending_token_count": int(dumped_pending or 0),
            "pending_skipped_tokens": int(pending_skipped_tokens or 0),
            "target_slice_mode": target_slice_mode,
            "kv_source": "restricted_missing_block_attention_diag",
            "metric_scope": "restricted_missing_block_only",
        }
        if dump_dir is None:
            row.update({"status": "missing_dump_dir", "reason": "kv_attention_diag_dump_dir_unavailable"})
            self._append_attention_diag_manifest(row)
            return
        if int(dumped_pending or 0) <= 0:
            row.update({"status": "no_pending_tokens", "reason": "no_dumped_pending_tokens"})
            self._append_attention_diag_manifest(row)
            return
        max_records = int(getattr(self.config, "kv_attention_diag_dump_max_records", 1024) or 0)
        if self._attention_diag_dumped_records >= max_records:
            row.update({"status": "max_records_reached", "reason": "kv_attention_diag_dump_max_records_reached"})
            self._append_attention_diag_manifest(row)
            return
        requested_layers = self._parse_attention_diag_dump_layers()
        if requested_layers is not None and int(layer_idx) not in requested_layers:
            return
        try:
            query, query_slice_mode, query_slice_start, query_slice_end, metric_scope = self._query_slice_for_attention_diag(
                query_states,
                dumped_pending,
            )
            source_key, source_value, loaded_source_paths = self._load_source_kv_for_attention_diag(
                source_file_paths,
                int(dumped_pending),
            )
            if list(source_key.shape) != list(exact_key.shape) or list(source_value.shape) != list(exact_value.shape):
                raise ValueError(
                    "source_exact_shape_mismatch source_key={} exact_key={} source_value={} exact_value={}".format(
                        list(source_key.shape),
                        list(exact_key.shape),
                        list(source_value.shape),
                        list(exact_value.shape),
                    )
                )

            query_dump = self._convert_attention_diag_dump_tensor(query)
            exact_key_dump = self._convert_attention_diag_dump_tensor(exact_key)
            exact_value_dump = self._convert_attention_diag_dump_tensor(exact_value)
            source_key_dump = self._convert_attention_diag_dump_tensor(source_key)
            source_value_dump = self._convert_attention_diag_dump_tensor(source_value)

            os.makedirs(dump_dir, exist_ok=True)
            file_name = "attention_flush_{}_layer_{:02d}_record_{:06d}.pt".format(
                "none" if flush_index is None else "{:06d}".format(int(flush_index)),
                int(layer_idx),
                int(self._attention_diag_dumped_records),
            )
            file_path = os.path.join(dump_dir, file_name)
            row.update(
                {
                    "status": "ok",
                    "file_path": file_path,
                    "dtype": str(query_dump.dtype).replace("torch.", ""),
                    "query_shape": list(query_dump.shape),
                    "exact_key_shape": list(exact_key_dump.shape),
                    "exact_value_shape": list(exact_value_dump.shape),
                    "source_key_shape": list(source_key_dump.shape),
                    "source_value_shape": list(source_value_dump.shape),
                    "query_slice_mode": query_slice_mode,
                    "query_slice_start": query_slice_start,
                    "query_slice_end": query_slice_end,
                    "query_token_count": int(query_slice_end - query_slice_start),
                    "key_value_token_count": int(dumped_pending),
                    "metric_scope": metric_scope,
                    "source_slice_mode": "source_manifest_slices",
                    "source_kv_file_paths": loaded_source_paths,
                    "target_kv_file_path": row_base.get("target_kv_file_path") or row_base.get("file_path"),
                    "target_present_key_shape": row_base.get("present_key_shape"),
                    "target_pending_slice_start": row_base.get("pending_slice_start"),
                    "target_pending_slice_end": row_base.get("pending_slice_end"),
                }
            )
            payload = {
                "query": query_dump,
                "exact_key": exact_key_dump,
                "exact_value": exact_value_dump,
                "source_key": source_key_dump,
                "source_value": source_value_dump,
                "metadata": dict(row),
            }
            torch.save(payload, file_path)
            self._attention_diag_dumped_records += 1
            self._append_attention_diag_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "attention_diag_dump",
                    status="ok",
                    file_path=file_path,
                    flush_index=flush_index,
                    layer_idx=int(layer_idx),
                    pending_token_count=int(dumped_pending),
                    metric_scope="restricted_missing_block_only",
                )
        except Exception as exc:
            row.update({"status": "query_unavailable" if "query_unavailable" in str(exc) else "dump_error", "reason": str(exc)})
            self._append_attention_diag_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "attention_diag_dump",
                    status=row.get("status"),
                    flush_index=flush_index,
                    layer_idx=int(layer_idx),
                    reason=str(exc),
                )

    def _maybe_dump_full_attention_diag(
        self,
        layer_idx,
        flush_index,
        query_states,
        position_bias,
        full_key,
        full_value,
        missing_block_start,
        missing_block_end,
        source_file_paths,
        source_record_ids,
        source_layer,
        target_slice_mode,
        pending_skipped_tokens,
        dumped_pending,
        metadata_positions,
        row_base,
    ):
        if not getattr(self.config, "kv_full_attention_diag_dump_enabled", False):
            return
        dump_dir = self._full_attention_diag_dump_dir()
        row = {
            "record_type": "full_attention_diag",
            "status": "pending",
            "flush_index": flush_index,
            "source_layer": source_layer,
            "target_layer": int(layer_idx),
            "layer_gap": int(layer_idx) - int(source_layer) if source_layer is not None else None,
            "source_record_ids": list(source_record_ids or []),
            "pending_decoder_positions": metadata_positions,
            "pending_token_count": int(dumped_pending or 0),
            "pending_skipped_tokens": int(pending_skipped_tokens or 0),
            "target_slice_mode": target_slice_mode,
            "kv_source": "full_cache_decoder_self_attention_diag",
            "metric_scope": "full_cache_decoder_self_attention",
            "missing_block_start": int(missing_block_start) if missing_block_start is not None else None,
            "missing_block_end": int(missing_block_end) if missing_block_end is not None else None,
        }
        if dump_dir is None:
            row.update({"status": "missing_dump_dir", "reason": "kv_full_attention_diag_dump_dir_unavailable"})
            self._append_full_attention_diag_manifest(row)
            return
        if int(dumped_pending or 0) <= 0:
            row.update({"status": "no_pending_tokens", "reason": "no_dumped_pending_tokens"})
            self._append_full_attention_diag_manifest(row)
            return
        max_records = int(getattr(self.config, "kv_full_attention_diag_dump_max_records", 256) or 0)
        if self._full_attention_diag_dumped_records >= max_records:
            row.update({"status": "max_records_reached", "reason": "kv_full_attention_diag_dump_max_records_reached"})
            self._append_full_attention_diag_manifest(row)
            return
        requested_layers = self._parse_full_attention_diag_dump_layers()
        if requested_layers is not None and int(layer_idx) not in requested_layers:
            return
        try:
            if full_key is None or full_value is None:
                raise ValueError("full_key_value_unavailable")
            if len(full_key.shape) != 4 or len(full_value.shape) != 4:
                raise ValueError("unexpected_full_kv_rank key={} value={}".format(list(full_key.shape), list(full_value.shape)))
            if list(full_key.shape) != list(full_value.shape):
                raise ValueError("full_key_value_shape_mismatch")
            full_key_token_count = int(full_key.shape[2])
            max_key_tokens = int(getattr(self.config, "kv_full_attention_diag_dump_max_key_tokens", 512) or 0)
            if max_key_tokens > 0 and full_key_token_count > max_key_tokens:
                row.update(
                    {
                        "status": "full_key_too_long",
                        "reason": "full_key_token_count_exceeds_cap",
                        "full_key_token_count": full_key_token_count,
                        "max_key_tokens": max_key_tokens,
                    }
                )
                self._append_full_attention_diag_manifest(row)
                return
            missing_block_start = int(missing_block_start)
            missing_block_end = int(missing_block_end)
            if missing_block_start < 0 or missing_block_end > full_key_token_count or missing_block_end <= missing_block_start:
                raise ValueError("invalid_missing_block_slice")
            missing_block_token_count = int(missing_block_end - missing_block_start)
            if missing_block_token_count != int(dumped_pending):
                raise ValueError(
                    "missing_block_token_count_mismatch block={} dumped_pending={}".format(
                        missing_block_token_count,
                        int(dumped_pending),
                    )
                )
            query, query_slice_mode, query_slice_start, query_slice_end, _restricted_scope = self._query_slice_for_attention_diag(
                query_states,
                dumped_pending,
            )
            if query.shape[0] != full_key.shape[0] or query.shape[1] != full_key.shape[1] or query.shape[-1] != full_key.shape[-1]:
                raise ValueError(
                    "query_full_key_shape_mismatch query={} full_key={}".format(
                        list(query.shape),
                        list(full_key.shape),
                    )
                )
            source_key, source_value, loaded_source_paths = self._load_source_kv_for_attention_diag(
                source_file_paths,
                int(dumped_pending),
            )
            exact_missing_key = full_key[:, :, missing_block_start:missing_block_end, :]
            exact_missing_value = full_value[:, :, missing_block_start:missing_block_end, :]
            if list(source_key.shape) != list(exact_missing_key.shape) or list(source_value.shape) != list(exact_missing_value.shape):
                raise ValueError(
                    "source_exact_shape_mismatch source_key={} exact_key={} source_value={} exact_value={}".format(
                        list(source_key.shape),
                        list(exact_missing_key.shape),
                        list(source_value.shape),
                        list(exact_missing_value.shape),
                    )
                )

            position_bias_dump = None
            position_bias_included = False
            position_bias_shape = None
            if bool(getattr(self.config, "kv_full_attention_diag_dump_include_position_bias", True)) and position_bias is not None:
                if len(position_bias.shape) == 4 and int(position_bias.shape[-1]) == full_key_token_count:
                    if int(position_bias.shape[2]) >= query_slice_end:
                        position_bias_slice = position_bias[:, :, query_slice_start:query_slice_end, :]
                        position_bias_dump = self._convert_full_attention_diag_dump_tensor(position_bias_slice)
                        position_bias_included = True
                        position_bias_shape = list(position_bias_dump.shape)

            query_dump = self._convert_full_attention_diag_dump_tensor(query)
            full_key_dump = self._convert_full_attention_diag_dump_tensor(full_key)
            full_value_dump = self._convert_full_attention_diag_dump_tensor(full_value)
            source_key_dump = self._convert_full_attention_diag_dump_tensor(source_key)
            source_value_dump = self._convert_full_attention_diag_dump_tensor(source_value)
            exact_missing_key_dump = self._convert_full_attention_diag_dump_tensor(exact_missing_key)
            exact_missing_value_dump = self._convert_full_attention_diag_dump_tensor(exact_missing_value)

            os.makedirs(dump_dir, exist_ok=True)
            file_name = "full_attention_flush_{}_layer_{:02d}_record_{:06d}.pt".format(
                "none" if flush_index is None else "{:06d}".format(int(flush_index)),
                int(layer_idx),
                int(self._full_attention_diag_dumped_records),
            )
            file_path = os.path.join(dump_dir, file_name)
            row.update(
                {
                    "status": "ok",
                    "file_path": file_path,
                    "dtype": str(query_dump.dtype).replace("torch.", ""),
                    "query_shape": list(query_dump.shape),
                    "full_exact_key_shape": list(full_key_dump.shape),
                    "full_exact_value_shape": list(full_value_dump.shape),
                    "missing_block_source_key_shape": list(source_key_dump.shape),
                    "missing_block_source_value_shape": list(source_value_dump.shape),
                    "exact_missing_key_shape": list(exact_missing_key_dump.shape),
                    "exact_missing_value_shape": list(exact_missing_value_dump.shape),
                    "query_slice_mode": query_slice_mode,
                    "query_slice_start": query_slice_start,
                    "query_slice_end": query_slice_end,
                    "query_token_count": int(query_slice_end - query_slice_start),
                    "key_value_token_count": missing_block_token_count,
                    "full_key_token_count": full_key_token_count,
                    "missing_block_token_count": missing_block_token_count,
                    "metric_scope": "full_cache_decoder_self_attention",
                    "source_slice_mode": "source_manifest_slices",
                    "source_kv_file_paths": loaded_source_paths,
                    "target_kv_file_path": row_base.get("target_kv_file_path") or row_base.get("file_path"),
                    "target_present_key_shape": row_base.get("present_key_shape"),
                    "target_pending_slice_start": row_base.get("pending_slice_start"),
                    "target_pending_slice_end": row_base.get("pending_slice_end"),
                    "position_bias_included": position_bias_included,
                    "position_bias_shape": position_bias_shape,
                }
            )
            payload = {
                "query": query_dump,
                "full_exact_key": full_key_dump,
                "full_exact_value": full_value_dump,
                "missing_block_source_key": source_key_dump,
                "missing_block_source_value": source_value_dump,
                "exact_missing_key": exact_missing_key_dump,
                "exact_missing_value": exact_missing_value_dump,
                "missing_block_start": missing_block_start,
                "missing_block_end": missing_block_end,
                "metadata": dict(row),
            }
            if position_bias_dump is not None:
                payload["position_bias"] = position_bias_dump
            torch.save(payload, file_path)
            self._full_attention_diag_dumped_records += 1
            self._append_full_attention_diag_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "full_attention_diag_dump",
                    status="ok",
                    file_path=file_path,
                    flush_index=flush_index,
                    layer_idx=int(layer_idx),
                    pending_token_count=int(dumped_pending),
                    full_key_token_count=full_key_token_count,
                    metric_scope="full_cache_decoder_self_attention",
                )
        except Exception as exc:
            row.update({"status": "query_unavailable" if "query_unavailable" in str(exc) else "dump_error", "reason": str(exc)})
            self._append_full_attention_diag_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "full_attention_diag_dump",
                    status=row.get("status"),
                    flush_index=flush_index,
                    layer_idx=int(layer_idx),
                    reason=str(exc),
                )

    def _maybe_dump_exact_catchup_kv(
        self,
        dump_context,
        layer_idx,
        present_key_value_state,
        past_key_value,
        position_bias,
        pending_skipped_tokens,
        metadata_positions,
        exact_catchup_trace,
        hidden_states_seq_len,
    ):
        if not dump_context.get("enabled"):
            return
        flush_index = dump_context.get("flush_index")
        max_layers = int(getattr(self.config, "kv_exact_catchup_dump_max_layers", 4) or 0)
        requested_layers = dump_context.get("requested_layers")
        manifest_base = {
            "record_type": "target_kv",
            "flush_index": flush_index,
            "layer_idx": int(layer_idx),
            "target_layer": int(layer_idx),
            "start_layer": (
                exact_catchup_trace.get("exact_catchup_layer_range", [None])[0]
                if exact_catchup_trace.get("exact_catchup_layer_range")
                else None
            ),
            "end_layer": exact_catchup_trace.get("end_layer"),
            "num_catchup_layers": exact_catchup_trace.get("num_catchup_layers"),
            "pending_skipped_tokens": int(pending_skipped_tokens or 0),
            "pending_decoder_positions": metadata_positions,
            "metadata_positions": metadata_positions,
            "copy_skipped_hidden_states": bool(self.config.copy_skipped_hidden_states),
            "parallel_causal_mask": bool(self.config.parallel_causal_mask),
            "hidden_states_seq_len": int(hidden_states_seq_len),
            "kv_source": "free_exact_parallel_catchup",
        }
        if requested_layers is not None and int(layer_idx) not in requested_layers:
            return
        if dump_context.get("dumped_layers", 0) >= max_layers:
            if requested_layers is None:
                return
            row = dict(manifest_base)
            row.update({"dump_succeeded": False, "skip_reason": "max_layers_reached"})
            self._append_exact_catchup_manifest(row)
            return

        row = dict(manifest_base)
        try:
            if present_key_value_state is None or len(present_key_value_state) < 2:
                raise ValueError("present_key_value_state missing self-attention key/value")
            key_tensor = present_key_value_state[0]
            value_tensor = present_key_value_state[1]
            if key_tensor is None or value_tensor is None:
                raise ValueError("present self-attention key/value is None")
            if len(key_tensor.shape) < 4 or len(value_tensor.shape) < 4:
                raise ValueError("unexpected K/V rank: key={} value={}".format(key_tensor.shape, value_tensor.shape))
            if key_tensor.shape[2] != value_tensor.shape[2]:
                raise ValueError("key/value sequence lengths differ")

            pending_start, pending_end, slice_mode, past_len = self._pending_slice_for_exact_catchup_dump(
                key_tensor,
                past_key_value,
                pending_skipped_tokens,
            )
            if pending_start is None:
                raise ValueError("could not infer pending-token K/V slice")
            max_tokens = int(getattr(self.config, "kv_exact_catchup_dump_max_tokens_per_flush", 8) or 0)
            dumped_pending = min(int(pending_skipped_tokens or 0), max_tokens)
            pending_end = pending_start + dumped_pending
            include_current = bool(getattr(self.config, "kv_exact_catchup_dump_include_current_token", False))
            current_start = pending_start + int(pending_skipped_tokens or 0)
            current_end = current_start + 1
            current_slice_available = include_current and current_end <= int(key_tensor.shape[2])
            source_records = list(self.stack_source_kv_records or [])
            linked_source_records = source_records[:dumped_pending]
            source_record_ids = [
                record.get("source_record_id")
                for record in linked_source_records
                if isinstance(record, dict)
            ]
            source_file_paths = [
                record.get("source_kv_file_path") or record.get("file_path")
                for record in linked_source_records
                if isinstance(record, dict) and (record.get("source_kv_file_path") or record.get("file_path"))
            ]
            source_layers = sorted(
                {
                    int(record.get("source_layer"))
                    for record in linked_source_records
                    if isinstance(record, dict) and record.get("source_layer") is not None
                }
            )
            source_layer = source_layers[0] if len(source_layers) == 1 else None

            exact_key = self._convert_exact_catchup_dump_tensor(key_tensor[:, :, pending_start:pending_end, :])
            exact_value = self._convert_exact_catchup_dump_tensor(value_tensor[:, :, pending_start:pending_end, :])
            row.update(
                {
                    "dump_succeeded": True,
                    "dtype": str(exact_key.dtype).replace("torch.", ""),
                    "device_before_dump": str(key_tensor.device),
                    "slice_mode": slice_mode,
                    "self_attn_past_len_before_layer": past_len,
                    "present_key_shape": list(key_tensor.shape),
                    "present_value_shape": list(value_tensor.shape),
                    "pending_slice_start": pending_start,
                    "pending_slice_end": pending_end,
                    "current_slice_start": current_start if include_current else None,
                    "current_slice_end": current_end if current_slice_available else None,
                    "current_slice_included": bool(current_slice_available),
                    "pending_tokens_dumped": dumped_pending,
                    "pending_tokens_truncated": dumped_pending < int(pending_skipped_tokens or 0),
                    "source_layer": source_layer,
                    "source_record_ids": source_record_ids,
                    "source_kv_file_paths": source_file_paths,
                    "source_records_available": len(source_record_ids),
                    "source_records_missing": max(0, dumped_pending - len(source_record_ids)),
                }
            )
            payload = {
                "exact_key": exact_key,
                "exact_value": exact_value,
                "metadata": dict(row),
            }
            if current_slice_available:
                payload["current_key"] = self._convert_exact_catchup_dump_tensor(key_tensor[:, :, current_start:current_end, :])
                payload["current_value"] = self._convert_exact_catchup_dump_tensor(value_tensor[:, :, current_start:current_end, :])

            file_name = "flush_{:06d}_layer_{:02d}.pt".format(int(flush_index), int(layer_idx))
            file_path = os.path.join(dump_context["dump_dir"], file_name)
            row["file_path"] = file_path
            row["target_kv_file_path"] = file_path
            payload["metadata"]["file_path"] = file_path
            payload["metadata"]["target_kv_file_path"] = file_path
            query_states = getattr(self.block[int(layer_idx)].layer[0].SelfAttention, "_last_query_states", None)
            self_attn_position_bias = getattr(self.block[int(layer_idx)].layer[0].SelfAttention, "_last_position_bias", None)
            self._maybe_dump_attention_diag(
                layer_idx=layer_idx,
                flush_index=flush_index,
                query_states=query_states,
                exact_key=exact_key,
                exact_value=exact_value,
                source_file_paths=source_file_paths,
                source_record_ids=source_record_ids,
                source_layer=source_layer,
                target_slice_mode=slice_mode,
                pending_skipped_tokens=pending_skipped_tokens,
                dumped_pending=dumped_pending,
                metadata_positions=metadata_positions,
                row_base=row,
            )
            self._maybe_dump_full_attention_diag(
                layer_idx=layer_idx,
                flush_index=flush_index,
                query_states=query_states,
                position_bias=self_attn_position_bias if self_attn_position_bias is not None else position_bias,
                full_key=key_tensor,
                full_value=value_tensor,
                missing_block_start=pending_start,
                missing_block_end=pending_end,
                source_file_paths=source_file_paths,
                source_record_ids=source_record_ids,
                source_layer=source_layer,
                target_slice_mode=slice_mode,
                pending_skipped_tokens=pending_skipped_tokens,
                dumped_pending=dumped_pending,
                metadata_positions=metadata_positions,
                row_base=row,
            )
            torch.save(payload, file_path)
            dump_context["dumped_layers"] = dump_context.get("dumped_layers", 0) + 1
            self._append_exact_catchup_manifest(row)
            self._append_restoration_dryrun_manifest(
                {
                    "flush_index": flush_index,
                    "layer_idx": int(layer_idx),
                    "exact_kv_file_path": file_path,
                    "pending_tokens_dumped": dumped_pending,
                    "kv_source": "free_exact_parallel_catchup",
                }
            )
            self.kv_trace.record(
                "exact_catchup_kv_dump",
                dump_succeeded=True,
                flush_index=flush_index,
                layer_idx=int(layer_idx),
                file_path=file_path,
                pending_tokens_dumped=dumped_pending,
                pending_tokens_truncated=dumped_pending < int(pending_skipped_tokens or 0),
                slice_mode=slice_mode,
            )
        except Exception as exc:
            row.update({"dump_succeeded": False, "error_message": str(exc)})
            self._append_exact_catchup_manifest(row)
            if hasattr(self, "kv_trace") and self.kv_trace.enabled:
                self.kv_trace.record(
                    "exact_catchup_kv_dump",
                    dump_succeeded=False,
                    flush_index=flush_index,
                    layer_idx=int(layer_idx),
                    error_message=str(exc),
                )

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

    def _run_exact_recompute_verification(
        self,
        plan,
        encoder_hidden_states=None,
        encoder_extended_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
    ):
        recompute_relative_indices = plan.get("recompute_relative_indices", [])
        verify_hidden_states, invalid_relative_indices = gather_recompute_hidden_states(
            self.stack_hidden_states,
            recompute_relative_indices,
        )
        missing_layer_range = plan.get("missing_layer_range") or []
        if invalid_relative_indices:
            return build_exact_recompute_verification_event(
                plan,
                input_hidden_states=None,
                recomputed_hidden_states=None,
                verification_status="shape_only",
                invalid_relative_indices=invalid_relative_indices,
            )
        if verify_hidden_states is None:
            return build_exact_recompute_verification_event(
                plan,
                input_hidden_states=None,
                recomputed_hidden_states=None,
                verification_status="skipped_empty_selection",
            )

        oracle_self_attn_kv_shapes = expected_self_attn_kv_shapes(
            verify_hidden_states,
            self.block[0].layer[0].SelfAttention.n_heads,
            self.block[0].layer[0].SelfAttention.key_value_proj_dim,
            missing_layer_range,
        )
        input_hidden_states = verify_hidden_states
        try:
            with torch.no_grad():
                local_hidden_states = verify_hidden_states
                local_position_bias = None
                for layer_idx in missing_layer_range:
                    local_attention_mask = torch.ones(
                        local_hidden_states.shape[0],
                        local_hidden_states.shape[1],
                        device=local_hidden_states.device,
                    )
                    local_extended_attention_mask = self.get_extended_attention_mask(
                        local_attention_mask,
                        torch.Size([local_hidden_states.shape[0], local_hidden_states.shape[1]]),
                    )
                    layer_outputs = self.block[layer_idx](
                        local_hidden_states,
                        attention_mask=local_extended_attention_mask,
                        position_bias=local_position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=None,
                        layer_head_mask=head_mask[layer_idx] if head_mask is not None else None,
                        cross_attn_layer_head_mask=cross_attn_head_mask[layer_idx] if cross_attn_head_mask is not None else None,
                        past_key_value=None,
                        use_cache=False,
                        output_attentions=False,
                        skip_mask=False,
                        parallel_mask=True,
                        stack_hidden_states=None,
                        layer_idx=layer_idx,
                        kv_importance_tracker=None,
                    )
                    local_hidden_states = layer_outputs[0].detach()
                    local_position_bias = layer_outputs[1] if len(layer_outputs) > 1 else None
            return build_exact_recompute_verification_event(
                plan,
                input_hidden_states=input_hidden_states,
                recomputed_hidden_states=local_hidden_states,
                verification_status="ok",
                oracle_self_attn_kv_shapes=oracle_self_attn_kv_shapes,
            )
        except Exception as exc:
            return build_exact_recompute_verification_event(
                plan,
                input_hidden_states=input_hidden_states,
                recomputed_hidden_states=None,
                verification_status="shape_only",
                oracle_self_attn_kv_shapes=oracle_self_attn_kv_shapes,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

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
        kv_importance_tracker,
        record_deploy_time,
    ):
        """Execute one target decoder block (layer j, j >= source layer)
        against a batch of already-shallow-computed hidden states,
        computing the attention mask and relative-position bias on first
        use and reusing them for subsequent layers in the same batch.

        This is a narrow execution helper used exclusively by
        calibration-only terminal exact finalization
        (finalize_pending_fixed_layer_calibration_exits). It is NOT called
        by the official synchronized parallel flush (parallel_gen_token),
        which still uses its own existing inline block-execution logic
        (interleaved with H2O masking, kv_trace, missing-KV accounting, and
        runtime restoration that this helper deliberately does not
        reproduce) -- that path was left unmodified to avoid regression
        risk in already-accepted, heavily-interleaved code. This helper's
        block-call sequence mirrors parallel_gen_token's non-H2O inline
        semantics (attention-mask/position-bias construction,
        gen_cross_attn_key_value fallback, the block(...) call itself, and
        output unpacking) closely enough that the two are expected to
        produce identical per-pending-token results; that equivalence is
        established empirically by the production-decoder-block parity
        test (test_native_free_source6_model_call_path.py), not by a
        shared call site. No accounting, tracing, H2O masking, or runtime
        restoration happens here; the only caller remains responsible for
        that itself.

        kv_importance_tracker/record_deploy_time are explicit so terminal
        finalization (calibration collection cost only) never pollutes the
        shared kv_importance tracker or FREE/exact-catchup runtime timing
        fields -- this helper's only caller always passes None/False.
        """
        past_key_value = past_key_values[j]
        if past_key_value is None:
            # if past_key_values is not defined, it implies that all previous tokens have skipped Deep decoder
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
            if record_deploy_time:
                self.deploy_time['time_parallel_key_value_gen'][1] += self.block[j].key_value_gen_time

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
                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1):, :]
                if extended_attention_mask is not None:
                    position_bias = position_bias + extended_attention_mask  # (batch_size, n_heads, seq_length, key_length)

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
            layer_idx=j,
            kv_importance_tracker=kv_importance_tracker,
        )
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

        if record_deploy_time:
            for idx, t in enumerate(self.block[j].key_value_gen_time): self.deploy_time['time_parallel_key_value_gen'][idx] += t
            for idx, t in enumerate(self.block[j].attn_time): self.deploy_time['time_parallel_attn'][idx] += t
            self.deploy_time['time_parallel_ffn'] += self.block[j].ffn_time

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
        """Calibration-only terminal exact finalization for Native FREE
        fixed-source-layer-6 pending exits still unresolved when generation
        has just terminated. Must be called only after the current token is
        selected, its selected-token commit verification has completed, it
        has been appended to input_ids, and the stopping criterion has made
        generation termination final -- i.e. from the generation-finalization
        point, before record_tail_pending_skips(reason="generation_end").

        Executes the target decoder blocks (via the narrow
        _execute_deep_target_layer helper, mirroring parallel_gen_token's
        non-H2O inline block-execution semantics -- see that helper's own
        docstring) against the exact same past K/V cache the official
        synchronized parallel flush would use, but processes ONLY the
        already-generated pending tokens: no new token is generated, no
        dummy no-crossing token is appended, the LM head is never called,
        and no confidence/threshold/early-exit runtime state is touched.
        Equivalence with the normal flush's own (separate, unmodified)
        inline computation is established empirically by the
        production-decoder-block parity test, not by a shared call site.

        On full success, the pending stack is cleared exactly like a normal
        synchronized flush -- so record_tail_pending_skips() naturally
        becomes a no-op afterward. On any failure, only the events that
        actually finalized are removed from the pending stack (never a
        successfully finalized event is left for record_tail_pending_skips
        to also count), and the remainder is left in place so that existing
        fail-closed fallback still records it.
        """

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

        # Tail input invariants -- fail closed, never fabricate. None of
        # these were even attempted, so any violation here makes the whole
        # pending batch "remaining_unfinalized", not "failed".
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
                    kv_importance_tracker=None,
                    record_deploy_time=False,
                )
                if not torch.isfinite(hidden_states).all():
                    raise ValueError("calibration_tail_finalization_nonfinite_hidden")
                if present_key_value_state is None or not torch.isfinite(present_key_value_state[0]).all() or not torch.isfinite(present_key_value_state[1]).all():
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
                # The collector's tensor population check requires full
                # 0..len(self.block)-1 coverage (only layers >= source_layer
                # are ever used downstream). Layers below source_layer were
                # never touched by this finalization -- each pending token
                # already passed through them normally at its own exit time,
                # so their exact K/V already live in the untouched input
                # past_key_values at this same absolute position.
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
            self.stack_hidden_states = self.stack_hidden_states[finalized_count:]
            self.stack_hidden_metadata = self.stack_hidden_metadata[finalized_count:]
            self.stack_source_kv_records = self.stack_source_kv_records[finalized_count:]
            self.stack_phase3c_source_hidden_records = self.stack_phase3c_source_hidden_records[finalized_count:]
            self.stack_fixed_layer_calibration_records = self.stack_fixed_layer_calibration_records[finalized_count:]
            return

        # Every pending exit finalized successfully: clear the pending stack
        # exactly like a normal synchronized flush, so
        # record_tail_pending_skips() naturally observes nothing left.
        self.stack_hidden_states = ()
        self.stack_hidden_metadata = ()
        self.stack_source_kv_records = ()
        self.stack_phase3c_source_hidden_records = ()
        self.stack_fixed_layer_calibration_records = ()
        self._tail_pending_recorded = False

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
        lm_head=None,
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
        _fixed_layer_calibration_active_here = self._fixed_layer_exact_cache_calibration_active()
        pending_skipped_tokens = len(self.stack_hidden_states)
        present_key_value_states_len_before = len(present_key_value_states) if present_key_value_states is not None else None
        decoder_position = infer_decoder_position(past_key_values)
        self_attn_past_len_at_start = None
        if past_key_values is not None and layer_idx is not None:
            try:
                self_attn_past_len_at_start = safe_cache_seq_len(past_key_values[layer_idx])
            except (TypeError, IndexError):
                self_attn_past_len_at_start = None
        pending_metadata = self.stack_hidden_metadata
        if isinstance(self.stack_hidden_states, tuple) and isinstance(pending_metadata, tuple):
            assert len(self.stack_hidden_states) == len(pending_metadata)
        pending_source_hidden_states = self.stack_hidden_states
        pending_source_records = self.stack_source_kv_records
        pending_phase3c_source_hidden_records = self.stack_phase3c_source_hidden_records
        pending_fixed_layer_calibration_records = self.stack_fixed_layer_calibration_records
        f2a_query_hidden_at_source = hidden_states.detach().clone() if self._f2a_enabled() else None
        f2a_exact_states_by_layer = {}
        if isinstance(pending_source_records, tuple) and isinstance(pending_metadata, tuple):
            assert len(pending_source_records) == len(pending_metadata)
        if isinstance(pending_phase3c_source_hidden_records, tuple) and isinstance(pending_metadata, tuple):
            assert len(pending_phase3c_source_hidden_records) == len(pending_metadata)
        if isinstance(pending_fixed_layer_calibration_records, tuple) and isinstance(pending_metadata, tuple):
            assert len(pending_fixed_layer_calibration_records) == len(pending_metadata)
        # Per-pending-event accumulating hidden-by-layer lists: each starts
        # as a copy of that event's own stored raw hidden for layers
        # 0..fixed_source_layer (captured at its own exit time, never from
        # this flush's current no-crossing token), then gains one entry per
        # deeper target layer as the for-j loop below actually executes.
        pending_fixed_layer_calibration_hidden_by_layer = (
            [
                list(record["hidden_prefix_0_to_source_layer"]) if record is not None else None
                for record in pending_fixed_layer_calibration_records
            ]
            if _fixed_layer_calibration_active_here
            else None
        )
        if _fixed_layer_calibration_active_here:
            for record in pending_fixed_layer_calibration_records:
                if record is None or not record.get("selected_token_committed"):
                    raise ValueError("calibration_fixed_layer_pending_record_missing_or_unverified")
        pending_metadata_trace, pending_metadata_truncated = metadata_list_to_trace(pending_metadata)
        metadata_positions = candidate_positions_from_metadata(pending_metadata)
        metadata_positions_available = bool(metadata_positions and any(position is not None for position in metadata_positions))
        metadata_positions_trace = metadata_positions
        if metadata_positions_trace is not None and len(metadata_positions_trace) > 128:
            metadata_positions_trace = metadata_positions_trace[:128]
            pending_metadata_truncated = True
        pending_start_position = infer_pending_start_position_from_metadata(pending_metadata)
        if pending_start_position is None:
            pending_start_position = self_attn_past_len_at_start
        if pending_start_position is None:
            pending_start_position = decoder_position
        candidate_positions_source = "metadata" if metadata_positions_available else (
            "pending_start_position" if pending_start_position is not None else "relative_only"
        )
        kv_restore_policy = getattr(self.config, "kv_restore_policy", "none")
        h2o_mask_plan = None

        if (
            self.is_decoder
            and kv_restore_policy in {
                "h2o_topk_dryrun",
                "h2o_mask_noop",
                "h2o_exact_recompute_dryrun",
                "h2o_exact_recompute_verify",
                "h2o_mask",
            }
            and pending_skipped_tokens > 0
            and (
                kv_restore_policy == "h2o_mask"
                or (
                    self.kv_trace.enabled
                    and (
                        kv_restore_policy in {"h2o_exact_recompute_dryrun", "h2o_exact_recompute_verify"}
                        or getattr(self.config, "kv_restore_log_candidates", True)
                    )
                )
            )
        ):
            selection = select_h2o_topk_dryrun(
                self.kv_importance,
                target_layer=layer_idx,
                pending_count=pending_skipped_tokens,
                pending_start_position=pending_start_position,
                candidate_positions=metadata_positions if metadata_positions_available else None,
                shallow_exit_layer=self.shallow_exit_layer,
                topk=getattr(self.config, "kv_restore_topk", 0),
                importance_layer_mode=getattr(self.config, "kv_restore_importance_layer", "target"),
                recent_window=getattr(self.config, "kv_restore_recent_window", 0),
            )
            if self.kv_trace.enabled and (
                kv_restore_policy in {"h2o_exact_recompute_dryrun", "h2o_exact_recompute_verify"}
                or getattr(self.config, "kv_restore_log_candidates", True)
            ):
                self.kv_trace.record(
                    "kv_restore_h2o_topk_dryrun",
                    selection=selection,
                    copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
                    parallel_causal_mask=self.config.parallel_causal_mask,
                    decoder_position=decoder_position,
                    self_attn_past_len_at_start=self_attn_past_len_at_start,
                    pending_metadata=pending_metadata_trace,
                    pending_metadata_count=len(pending_metadata),
                    pending_metadata_truncated=pending_metadata_truncated,
                    metadata_positions=metadata_positions_trace,
                    metadata_positions_available=metadata_positions_available,
                    candidate_positions_source=selection.get("candidate_positions_source"),
                )
            if kv_restore_policy == "h2o_mask_noop":
                plan = build_h2o_mask_noop_plan(
                    selection,
                    pending_metadata=pending_metadata,
                    metadata_positions=metadata_positions_trace,
                    metadata_positions_available=metadata_positions_available,
                    pending_metadata_truncated=pending_metadata_truncated,
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("h2o_mask_noop_plan", **plan)
            elif kv_restore_policy == "h2o_mask":
                h2o_mask_plan = build_h2o_mask_plan(
                    selection,
                    pending_metadata=pending_metadata,
                    metadata_positions=metadata_positions_trace,
                    metadata_positions_available=metadata_positions_available,
                    pending_metadata_truncated=pending_metadata_truncated,
                    policy="h2o_mask",
                    actual_mask_applied=True,
                    mask_mode="key_mask",
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("h2o_mask_plan", **h2o_mask_plan)
            elif kv_restore_policy == "h2o_exact_recompute_dryrun":
                plan = build_exact_recompute_dryrun_plan(
                    selection,
                    pending_metadata=pending_metadata,
                    metadata_positions=metadata_positions_trace,
                    metadata_positions_available=metadata_positions_available,
                    pending_metadata_truncated=pending_metadata_truncated,
                    target_layer=layer_idx,
                    final_layer=len(self.block),
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("exact_recompute_dryrun_plan", **plan)
            elif kv_restore_policy == "h2o_exact_recompute_verify":
                plan = build_exact_recompute_dryrun_plan(
                    selection,
                    pending_metadata=pending_metadata,
                    metadata_positions=metadata_positions_trace,
                    metadata_positions_available=metadata_positions_available,
                    pending_metadata_truncated=pending_metadata_truncated,
                    target_layer=layer_idx,
                    final_layer=len(self.block),
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("exact_recompute_dryrun_plan", **plan)
                    verification_event = self._run_exact_recompute_verification(
                        plan,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_extended_attention_mask=encoder_extended_attention_mask,
                        head_mask=head_mask,
                        cross_attn_head_mask=cross_attn_head_mask,
                    )
                    self.kv_trace.record("exact_recompute_verification", **verification_event)
        exact_catchup_trace = self._exact_catchup_trace_fields(
            layer_idx,
            pending_skipped_tokens,
            pending_metadata_trace=pending_metadata_trace,
            metadata_positions=metadata_positions_trace,
            metadata_positions_available=metadata_positions_available,
        )
        exact_dump_context = self._begin_exact_catchup_dump_flush(layer_idx, pending_skipped_tokens)
        runtime_restoration_plan = self._runtime_restoration_plan(
            pending_skipped_tokens,
            pending_metadata,
            pending_source_records,
            layer_idx,
            phase3c_source_hidden_records=pending_phase3c_source_hidden_records,
        )
        # Record only the REQUIRED units (and flush count) here, before any
        # target-layer call has actually run. Recording EXECUTED units must
        # wait until each target layer's forward call actually returns
        # (see the `record_exact_catchup_executed` call inside the `for j`
        # loop below) -- otherwise a mid-loop failure would be reported as if
        # the complete planned flush had executed.
        self._missing_kv_accounting_obj().record_exact_catchup_required(
            pending_skipped_tokens,
            exact_catchup_trace.get("num_catchup_layers") or 0,
        )
        # Exact-catch-up overhead event recording is diagnostic-only and
        # opt-in; any failure here (e.g. identity fields unavailable) must
        # never break the real flush below, hence the isolated try/except.
        # Registered here (required units known); committed after the `for j`
        # loop completes without exception. There is no wrapping try/except
        # around that loop itself (an exception there already aborts this
        # generation step entirely) so a mid-loop failure leaves this event
        # permanently in "registered" state -- evaluation finalization
        # reports any such incomplete event as invalid rather than
        # fabricating a completion it never reached.
        fixed_source_overhead_event = None
        if self._exact_catchup_overhead_enabled():
            _fixed_source_overhead_recorder = self._missing_kv_exact_catchup_overhead_recorder_obj()
            try:
                sample_context = self._exact_catchup_overhead_sample_context_fields()
                if sample_context.get("stable_sample_id") not in (None, ""):
                    candidate_overhead_event = ExactCatchupOverheadEvent(
                        runtime_path=RUNTIME_PATH_FIXED_SOURCE_PARALLEL_FLUSH,
                        transaction_type="fixed_source_parallel_flush",
                        source_layer=layer_idx,
                        decoder_layer_count=len(self.block),
                        stable_sample_id=sample_context.get("stable_sample_id"),
                        selected_order=sample_context.get("selected_order"),
                        raw_dataset_index=sample_context.get("raw_dataset_index"),
                        generation_index=self._generation_index,
                        decoder_position=decoder_position,
                        pending_skipped_token_count=pending_skipped_tokens,
                        candidate_policy_name="fixed_source_parallel_flush_v1",
                    )
                    _fixed_source_overhead_recorder.register(candidate_overhead_event)
                    fixed_source_overhead_event = candidate_overhead_event
                else:
                    # Never a silent no-op -- see the matching candidate-path
                    # comment above: a missing identity must invalidate the
                    # run explicitly so full paper mode fails closed.
                    _fixed_source_overhead_recorder.record_missing_identity_skip(
                        reason="fixed_source_parallel_flush_stable_sample_id_unavailable"
                    )
            except Exception as exc:
                fixed_source_overhead_event = None
                _fixed_source_overhead_recorder.record_missing_identity_skip(
                    reason="fixed_source_parallel_flush_event_construction_failed:{}".format(type(exc).__name__)
                )
        self._missing_kv_accounting_obj().record_restoration_flush(
            runtime_restoration_plan.get("enabled"),
            len(runtime_restoration_plan.get("restore_relative_indices") or []),
        )
        if self.kv_trace.enabled and self._runtime_restoration_enabled():
            self.kv_trace.record("kv_runtime_restoration_flush_plan", **runtime_restoration_plan)
        self.kv_trace.record(
            "parallel_gen_token_enter",
            start_layer=layer_idx,
            pending_skipped_tokens=pending_skipped_tokens,
            hidden_states_seq_len=hidden_states.shape[1],
            copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
            present_key_value_states_len_before=present_key_value_states_len_before,
            decoder_position=decoder_position,
            pending_metadata=pending_metadata_trace,
            pending_metadata_count=len(pending_metadata),
            pending_metadata_truncated=pending_metadata_truncated,
            metadata_positions=metadata_positions_trace,
            metadata_positions_available=metadata_positions_available,
            candidate_positions_source=candidate_positions_source,
            exact_kv_dump_enabled=exact_dump_context.get("enabled"),
            exact_kv_dump_flush_index=exact_dump_context.get("flush_index"),
            exact_kv_dump_dir=exact_dump_context.get("dump_dir"),
            exact_kv_dump_reason=exact_dump_context.get("reason"),
            kv_runtime_restoration_enabled=runtime_restoration_plan.get("enabled"),
            runtime_restoration_flush_id=runtime_restoration_plan.get("runtime_restoration_flush_id"),
            kv_runtime_restoration_method=runtime_restoration_plan.get("restoration_method"),
            kv_runtime_restoration_recent_exact_window=runtime_restoration_plan.get("recent_exact_window"),
            kv_runtime_restoration_restore_relative_indices=runtime_restoration_plan.get("restore_relative_indices"),
            kv_runtime_restoration_exact_relative_indices=runtime_restoration_plan.get("exact_relative_indices"),
            kv_runtime_restoration_expected_target_layers=runtime_restoration_plan.get("expected_target_layers"),
            kv_runtime_restoration_expected_token_layer_record_count=runtime_restoration_plan.get("expected_token_layer_record_count"),
            phase3c_source_hidden_record_count=len(pending_phase3c_source_hidden_records or ()),
            **exact_catchup_trace,
        )
        
        if not self.config.copy_skipped_hidden_states:
            hidden_states = torch.cat(self.stack_hidden_states + (hidden_states,), dim=1)            
            # reset and re-calculate based on the length of hidden_states
            extended_attention_mask, position_bias = None, None
        else:
            self.stack_hidden_states = torch.cat(self.stack_hidden_states, dim=1)
            extended_attention_mask = attention_mask
        h2o_mask_active = kv_restore_policy == "h2o_mask" and h2o_mask_plan is not None
        if h2o_mask_active:
            if self.config.copy_skipped_hidden_states or not self.config.parallel_causal_mask:
                raise NotImplementedError(
                    "kv_restore_policy='h2o_mask' v1 requires copy_skipped_hidden_states=False "
                    "and parallel_causal_mask=True."
                )

        for j in range(layer_idx, len(self.block)):

            if _fixed_layer_calibration_active_here:
                # hidden_states here spans every pending exit position
                # (columns 0..N-1, in generation order) followed by the
                # current no-crossing token (last column). Each pending
                # event's own pre-block-j hidden must come from ITS OWN
                # column -- never from the current token's column (-1).
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
            if h2o_mask_active:
                extended_attention_mask, position_bias = None, None
            past_key_value_was_none = past_key_value is None
            if self.kv_trace.enabled:
                self_attn_past_len = safe_cache_seq_len(past_key_value)
                stack_hidden_states_seq_len = None
                if self.config.copy_skipped_hidden_states and hasattr(self.stack_hidden_states, "shape"):
                    stack_hidden_states_seq_len = self.stack_hidden_states.shape[1]
                hidden_states_seq_len = hidden_states.shape[1]
                hidden_states_processed = hidden_states_seq_len + (stack_hidden_states_seq_len or 0)
                self.kv_trace.record(
                    "parallel_layer",
                    layer_idx=j,
                    start_layer=layer_idx,
                    end_layer=exact_catchup_trace.get("end_layer"),
                    num_catchup_layers=exact_catchup_trace.get("num_catchup_layers"),
                    pending_skipped_tokens=pending_skipped_tokens,
                    past_key_value_was_none=past_key_value_was_none,
                    self_attn_past_len=self_attn_past_len,
                    hidden_states_processed=hidden_states_processed,
                    hidden_states_seq_len=hidden_states_seq_len,
                    stack_hidden_states_seq_len=stack_hidden_states_seq_len,
                    copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
                )
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
                    if h2o_mask_active:
                        past_key_value_len = safe_cache_seq_len(past_key_value)
                        past_key_value_len = 0 if past_key_value_len is None else past_key_value_len
                        h2o_mask_indices = build_h2o_key_mask_indices(
                            h2o_mask_plan,
                            pending_count=pending_skipped_tokens,
                            past_key_value_len=past_key_value_len,
                            hidden_states_seq_len=hidden_states.shape[1],
                        )
                        masked_key_positions = h2o_mask_indices["masked_key_positions"]
                        if masked_key_positions:
                            attention_mask[:, masked_key_positions] = 0
                        self.kv_trace.record(
                            "h2o_mask_applied",
                            policy="h2o_mask",
                            mask_mode="key_mask",
                            layer_idx=j,
                            target_layer=layer_idx,
                            pending_count=pending_skipped_tokens,
                            selected_relative_indices=h2o_mask_indices["kept_relative_indices"],
                            non_selected_relative_indices=h2o_mask_indices["masked_relative_indices"],
                            masked_relative_indices=h2o_mask_indices["masked_relative_indices"],
                            masked_key_positions=masked_key_positions,
                            kept_relative_indices=h2o_mask_indices["kept_relative_indices"],
                            current_token_local_indices=h2o_mask_indices["current_token_local_indices"],
                            past_key_value_len=h2o_mask_indices["past_key_value_len"],
                            hidden_states_seq_len=h2o_mask_indices["hidden_states_seq_len"],
                            real_seq_length=h2o_mask_indices["real_seq_length"],
                            would_mask_count=len(h2o_mask_indices["masked_relative_indices"]),
                            actual_mask_applied=True,
                            copy_skipped_hidden_states=False,
                            parallel_causal_mask=True,
                        )
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

            # Host-side wall clock for the overhead event's per-target-layer
            # row. This block processes the P pending skipped tokens AND the
            # current deep token TOGETHER (mixed_parallel_layer_execution),
            # unlike the candidate path's pure single-token exact-reference
            # call -- so this timing is explicitly labeled as a different
            # timing stratum and must never be silently merged with the
            # candidate path's pure immediate exact-reference timing.
            _overhead_target_start = time.perf_counter()
            try:
                with self._missing_kv_component_timer_obj().time_block(
                    "exact_parallel_catchup_time_ms",
                    device=hidden_states.device,
                ):
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
                        layer_idx=j,
                        kv_importance_tracker=self.kv_importance,
                    )
            except Exception as exc:
                # Explicitly finalize this event as failed before the
                # original exception propagates, instead of leaving it
                # permanently in "registered" state. Preserves the original
                # exception/traceback unchanged (bare `raise`); never
                # fabricates a per-target timing for the layer that failed.
                if fixed_source_overhead_event is not None:
                    try:
                        fixed_source_overhead_event.fail(
                            stage="fixed_source_target_layer_{}".format(j),
                            reason=str(exc),
                        )
                        self._missing_kv_exact_catchup_overhead_recorder_obj().finalize(fixed_source_overhead_event)
                    except Exception:
                        pass
                raise
            _overhead_target_elapsed_ms = (time.perf_counter() - _overhead_target_start) * 1000.0
            # This target layer's forward call has now actually returned, so
            # the P pending skipped tokens' worth of catch-up at this layer
            # are genuinely executed -- record executed units only here, never
            # before the call (see the required-only recording above the loop).
            self._missing_kv_accounting_obj().record_exact_catchup_executed(pending_skipped_tokens)
            if fixed_source_overhead_event is not None:
                try:
                    fixed_source_overhead_event.record_target_executed(
                        target_layer=j,
                        elapsed_ms=_overhead_target_elapsed_ms,
                        timing_backend="host_perf_counter_mixed_parallel_layer",
                        executed_pending_token_layer_units=pending_skipped_tokens,
                    )
                except Exception:
                    pass
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
            exact_present_key_value_state = present_key_value_state
            if self._f2a_enabled() and exact_present_key_value_state is not None:
                f2a_exact_states_by_layer[int(j)] = tuple(
                    item.detach().clone() if isinstance(item, torch.Tensor) else item
                    for item in exact_present_key_value_state
                )
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = None if h2o_mask_active else layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            runtime_restoration_event = None
            if use_cache and self._runtime_restoration_enabled():
                present_key_value_state, runtime_restoration_event = self._apply_runtime_restoration_to_present_kv(
                    j,
                    present_key_value_state,
                    past_key_value,
                    pending_skipped_tokens,
                    pending_source_records,
                    runtime_restoration_plan,
                    layer_input_seq_len,
                    pending_source_hidden_states=pending_source_hidden_states,
                    phase3c_source_hidden_records=pending_phase3c_source_hidden_records,
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("kv_runtime_restoration_layer", **runtime_restoration_event)
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + [present_key_value_state,]
            self._maybe_dump_exact_catchup_kv(
                exact_dump_context,
                j,
                exact_present_key_value_state,
                past_key_value,
                position_bias,
                pending_skipped_tokens,
                metadata_positions_trace,
                exact_catchup_trace,
                layer_input_seq_len,
            )
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        if _fixed_layer_calibration_active_here:
            # present_key_value_states now spans every layer, each with one
            # position per pending exit (columns self_attn_past_len_at_start
            # .. self_attn_past_len_at_start+N-1, in generation order)
            # followed by the current no-crossing token's own position.
            # Commit exactly one fitting event per pending exit -- never
            # reduced to a single event, and never using the current
            # token's own (-1) position.
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
            # Every prior pending exit is now committed. Only after that may
            # this token's own no-crossing accounting be staged; the normal
            # generation callback resolves it once the actual selected token
            # is known.
            self.exact_cache_calibration_collector.stage_no_crossing()

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        self.stack_hidden_states = ()
        self.stack_hidden_metadata = ()
        self.stack_source_kv_records = ()
        self.stack_phase3c_source_hidden_records = ()
        self.stack_fixed_layer_calibration_records = ()
        self._tail_pending_recorded = False
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        # The `for j` loop above completed for every missing target layer
        # without an exception reaching here, so this flush's planned exact
        # catch-up genuinely executed in full.
        if fixed_source_overhead_event is not None:
            try:
                fixed_source_overhead_event.commit()
            except Exception as exc:
                try:
                    fixed_source_overhead_event.fail(stage="commit_invariant_violation", reason=str(exc))
                except Exception:
                    pass
            try:
                self._missing_kv_exact_catchup_overhead_recorder_obj().finalize(fixed_source_overhead_event)
            except Exception:
                pass
        self.kv_trace.record(
            "parallel_gen_token_exit",
            start_layer=layer_idx,
            pending_skipped_tokens=pending_skipped_tokens,
            pending_metadata_count=len(pending_metadata),
            output_hidden_states_seq_len=hidden_states.shape[1],
            copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
            present_key_value_states_len_before=present_key_value_states_len_before,
            present_key_value_states_len_after=len(present_key_value_states) if present_key_value_states is not None else None,
            exact_kv_dump_enabled=exact_dump_context.get("enabled"),
            exact_kv_dump_flush_index=exact_dump_context.get("flush_index"),
            exact_kv_dump_dir=exact_dump_context.get("dump_dir"),
            exact_kv_dump_dumped_layers=exact_dump_context.get("dumped_layers"),
            exact_kv_dump_reason=exact_dump_context.get("reason"),
            kv_runtime_restoration_enabled=runtime_restoration_plan.get("enabled"),
            kv_runtime_restoration_counters=dict(self._kv_runtime_restoration_counters),
            **exact_catchup_trace,
        )
        # The reference exact catch-up computes its self-attention position
        # bias once (via block[0]'s learned relative-attention-bias table,
        # since only block 0 owns it) and reuses the same tensor unchanged
        # through every subsequent layer. Capture its last (current-query)
        # row here -- before it goes out of scope -- so fixed-layer F2a
        # shadow/candidate replay can reuse the exact reference semantics
        # instead of silently synthesizing a zero position bias at a
        # mid-network layer that does not own the learned bias table.
        f2a_reference_position_bias = (
            position_bias[:, :, -1:, :].detach().clone()
            if self._f2a_enabled() and isinstance(position_bias, torch.Tensor)
            else None
        )
        self._maybe_run_f2a_fixed_shadow_replay(
            source_layer=layer_idx,
            query_hidden_at_source=f2a_query_hidden_at_source,
            pending_metadata=pending_metadata,
            pending_phase3c_source_hidden_records=pending_phase3c_source_hidden_records,
            exact_states_by_layer=f2a_exact_states_by_layer,
            reference_hidden_states=hidden_states,
            reference_present_key_value_states=present_key_value_states,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_extended_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lm_head=lm_head,
            reference_position_bias=f2a_reference_position_bias,
        )

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
            if self.is_decoder:
                self.record_tail_pending_skips(reason="generation_reset_with_pending_stack")
            past_key_values = [None] * len(self.block)
            self.stack_hidden_states = ()
            self.stack_hidden_metadata = ()
            self.stack_source_kv_records = ()
            self.stack_phase3c_source_hidden_records = ()
            self.stack_fixed_layer_calibration_records = ()
            self.stack_conf, self.stack_pred = (), ()
            self._smoke_forced_flush_used = False
            self._tail_pending_recorded = False
            self._begin_missing_kv_generation()
            self._generation_index += 1
            self._record_missing_kv_generation_binding()
            self._begin_missing_kv_packed_generation()
            if self.is_decoder:
                self.kv_importance.reset()
                self.kv_trace.record(
                    "generation_reset",
                    is_decoder=self.is_decoder,
                    num_layers=len(self.block),
                    use_shallow_deep=self.use_shallow_deep,
                    use_early_exit=self.use_early_exit,
                    static_exit_layer=self.config.static_exit_layer,
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

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
        f2a_followup_query_initial_hidden = None
        if self._is_f2a_calm_enabled():
            f2a_followup_query_initial_hidden = hidden_states.detach().clone()
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        skip_mask = False  # False: forward, and True: skip
        # Per-layer validated pending K/V staged by a successful Batched
        # Task C2 flush in THIS forward pass only ({layer: (key, value)});
        # consumed by the current token's deep self-attention for its ONE
        # combined old+pending+current cache publication. Forward-local by
        # design: it can never leak into a later token or generation.
        task_c2_staged_pending_self_kv = None
        calm_runtime_restoration_context = None
        calm_phase3c_candidate_evaluations = []
        exact_cache_calibration_prefix_hidden_states = [] if self._exact_cache_calibration_enabled() else None
        self.shallow2deep = False  # False: skip, and True: forward
        self.lm_logits = None  # to prevent calculating logits twice

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
                accounting = self._missing_kv_accounting_obj()
                accounting.record_generated_token()
                if self._f2a_enabled():
                    accounting.record_f2a_reference_generated_token()
            self._maybe_record_calm_counterfactual_confidence(
                layer_idx=i,
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                auto_reg=auto_reg,
                lm_head=lm_head,
            )
                            
            if self.is_decoder and auto_reg and i > 0:
                
                # Shallow-Deep framework 
                if self.use_shallow_deep and i == self.shallow_exit_layer:
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    start = datetime.datetime.now()
                    with self._missing_kv_component_timer_obj().time_block(
                        "confidence_time_ms",
                        device=hidden_states.device,
                    ):
                        _hidden_states = self.dropout(self.final_layer_norm(hidden_states))
                        lm_logits = compute_exit_lm_logits(_hidden_states, lm_head, self.config)
                        # The official FREE shallow-selected token: the exact-deep
                        # computation below (when this token crosses) only makes the
                        # K/V cache exact for future tokens, it never re-selects the
                        # current token's identity.
                        fixed_layer_calibration_shallow_selected_token_id = (
                            int(lm_logits.detach().argmax(dim=-1).reshape(-1)[0].item())
                            if self._fixed_layer_exact_cache_calibration_active()
                            else None
                        )

                        skip_mask, conf = get_skip_mask(
                            lm_logits,
                            _hidden_states,
                            cm_head,
                            config=self.config,
                            adapt_threshold=self.bmm_threshold,
                            return_conf=True,
                        )
                    nominal_threshold = getattr(self.config, "shallow2deep_conf_threshold", None)
                    use_adapt_threshold = getattr(self.config, "use_adapt_threshold", False)
                    effective_threshold = self.bmm_threshold if self.bmm_threshold is not None else nominal_threshold
                    effective_threshold_available = effective_threshold is not None
                    skip_decision_reason = free_skip_decision_reason(
                        skip_mask,
                        use_adapt_threshold=use_adapt_threshold,
                        adaptive_threshold_active=self.bmm_threshold is not None,
                        effective_threshold_available=effective_threshold_available,
                    )
                    force_after = getattr(self.config, "smoke_force_flush_after_skips", None)
                    force_once = getattr(self.config, "smoke_force_flush_once", True)
                    forced_flush_used_before = self._smoke_forced_flush_used
                    should_force_flush = (
                        self.is_decoder
                        and self.use_shallow_deep
                        and force_after is not None
                        and int(force_after) > 0
                        and len(self.stack_hidden_states) >= int(force_after)
                        and (not force_once or not forced_flush_used_before)
                    )
                    if should_force_flush and skip_mask:
                        original_skip_mask = skip_mask
                        skip_mask = False
                        self._smoke_forced_flush_used = True
                        self.kv_trace.record(
                            "smoke_force_flush_override",
                            force_after=int(force_after),
                            pending_skipped_tokens=len(self.stack_hidden_states),
                            original_skip_mask=original_skip_mask,
                            forced_skip_mask=skip_mask,
                            confidence=conf,
                            threshold=nominal_threshold,
                            nominal_threshold=nominal_threshold,
                            effective_threshold=effective_threshold,
                            effective_threshold_available=effective_threshold_available,
                            use_adapt_threshold=use_adapt_threshold,
                            skip_decision_reason="smoke_force_flush_override",
                            exit_layer=i,
                            force_once=force_once,
                            forced_flush_used_before=forced_flush_used_before,
                        )
                    self.stack_conf = self.stack_conf + (conf,)
                    self.stack_pred = self.stack_pred + (lm_logits,)
                    
                    if not skip_mask: self.block_op[i] += 1
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                    stack_hidden_states_len_before = len(self.stack_hidden_states)
                    stack_hidden_metadata_len_before = len(self.stack_hidden_metadata)

                    # if skip Deep decoder, store hidden_states at self.shallow_exit_layer
                    if skip_mask:
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        self._missing_kv_accounting_obj().record_nominal_skip(i, len(self.block), token_count=1)
                        self.lm_logits = lm_logits
                        decoder_position = infer_decoder_position(past_key_values)
                        layer0_past_seq_len = safe_cache_seq_len(past_key_values[0]) if len(past_key_values) else None

                        # Task C2: for this exiting token only, attempt to
                        # immediately restore and atomically install the
                        # complete missing deep-layer K/V directly from h6,
                        # bypassing the pending-buffer/parallel_gen_token
                        # path entirely. Must run before ANY of that path's
                        # bookkeeping (stack_hidden_states/stack_hidden_metadata/
                        # stack_source_kv_records/stack_phase3c_source_hidden_records)
                        # is touched, so a failure can fall straight through
                        # into the existing Task C1 code completely unchanged.
                        task_c2_installed = False
                        if self._task_c2_direct_insertion_enabled():
                            task_c2_decoder_position = decoder_position
                            if task_c2_decoder_position is None:
                                # infer_decoder_position(past_key_values) is
                                # unavailable exactly at the first decoder
                                # position (no prior past K/V length to read
                                # yet). The last already-exact source layer's
                                # own present K/V (i-1, already computed this
                                # forward call) already carries that same
                                # position -- reuse the identical fallback
                                # the existing Task C1 pending-buffer path
                                # below already uses
                                # (_maybe_dump_source_kv_for_skip), scoped
                                # only to this Task C2 attempt so a genuinely
                                # first-token exit is not needlessly routed
                                # into Task C1 fallback, without touching any
                                # existing Task C1 position bookkeeping.
                                task_c2_decoder_position = self._infer_decoder_position_from_source_present_kv(
                                    present_key_value_states,
                                    i - 1,
                                )
                            task_c2_result = self._try_task_c2_fixed_source6_direct_insertion(
                                i,
                                hidden_states,
                                past_key_values,
                                present_key_value_states,
                                task_c2_decoder_position,
                                encoder_hidden_states,
                                encoder_extended_attention_mask,
                                encoder_decoder_position_bias,
                                head_mask,
                                cross_attn_head_mask,
                                use_cache,
                                output_attentions,
                            )
                            self._missing_kv_accounting_obj().record_task_c2_direct_insertion_attempt(
                                success=task_c2_result["success"],
                                fallback=not task_c2_result["success"],
                                requested_units=task_c2_result["requested_units"],
                                inserted_units=task_c2_result["inserted_units"],
                            )
                            self.kv_trace.record(
                                "kv_task_c2_direct_insertion_transaction",
                                source_layer=i,
                                first_missing_target_layer=i,
                                target_layers=task_c2_result["target_layers"],
                                decoder_position=task_c2_result["decoder_position"],
                                confidence=conf,
                                requested_token_layer_units=task_c2_result["requested_units"],
                                inserted_token_layer_units=task_c2_result["inserted_units"],
                                success=task_c2_result["success"],
                                failure_stage=task_c2_result["failure_stage"],
                                failure_type=task_c2_result["failure_type"],
                                failure_message=task_c2_result["failure_message"],
                                # These describe only the direct-insertion
                                # ATTEMPT itself (always false, on success or
                                # failure alike) -- never the token's full
                                # eventual lifecycle. On failure the token
                                # falls through into the existing Task C1
                                # pending-buffer path immediately after this
                                # event, which is what fallback_requested/
                                # fallback_path below records.
                                direct_attempt_exact_target_block_executed=False,
                                direct_attempt_pending_buffer_insertion_used=False,
                                partial_cache_committed=False,
                                fallback_requested=not task_c2_result["success"],
                                fallback_path=None if task_c2_result["success"] else "task_c1_pending_buffer",
                                generation_index=self._generation_index,
                            )
                            if task_c2_result["success"]:
                                present_key_value_states = task_c2_result["complete_cache"]
                                task_c2_installed = True

                        if task_c2_installed:
                            if self.config.use_synchronize: torch.cuda.synchronize()
                            if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
                            break

                        skipped_metadata = None
                        if self.config.parallel_gen_token:
                            if use_cache:
                                for j in range(i, len(self.block)):
                                    present_key_value_states = present_key_value_states + [past_key_values[j],]
                            self.stack_hidden_states = self.stack_hidden_states + (hidden_states,)
                            relative_index = len(self.stack_hidden_metadata)
                            skipped_metadata = make_skipped_token_metadata(
                                relative_index=relative_index,
                                decoder_position=decoder_position,
                                layer0_past_seq_len=layer0_past_seq_len,
                                exit_layer=i,
                                confidence=conf,
                            )
                            legacy_source_kv_layer = i - 1
                            phase3c_source_hidden_layer = i
                            catchup_start_layer = i
                            source_layer_for_restoration = legacy_source_kv_layer
                            phase3c_source_hidden_record = self._maybe_make_phase3c_source_hidden_record_for_skip(
                                hidden_states,
                                phase3c_source_hidden_layer=phase3c_source_hidden_layer,
                                exit_layer=i,
                                decoder_position=decoder_position,
                                relative_index=relative_index,
                                confidence=conf,
                            )
                            source_kv_record = self._maybe_dump_source_kv_for_skip(
                                present_key_value_states,
                                past_key_values,
                                legacy_source_kv_layer,
                                i,
                                skipped_metadata,
                                decoder_position,
                                conf,
                                source_hidden_states=hidden_states,
                            )
                            if (
                                isinstance(source_kv_record, dict)
                                and source_kv_record.get("decoder_position_fallback_used")
                                and source_kv_record.get("decoder_position") is not None
                            ):
                                decoder_position = source_kv_record.get("decoder_position")
                                skipped_metadata = make_skipped_token_metadata(
                                    relative_index=relative_index,
                                    decoder_position=decoder_position,
                                    layer0_past_seq_len=layer0_past_seq_len,
                                    exit_layer=i,
                                    confidence=conf,
                                )
                                if isinstance(phase3c_source_hidden_record, dict):
                                    phase3c_source_hidden_record["decoder_position"] = decoder_position
                            self.stack_hidden_metadata = self.stack_hidden_metadata + (skipped_metadata,)
                            self.stack_source_kv_records = self.stack_source_kv_records + (source_kv_record,)
                            self.stack_phase3c_source_hidden_records = self.stack_phase3c_source_hidden_records + (
                                phase3c_source_hidden_record,
                            )
                            # This is the ACTUAL exit branch (skip_mask=True,
                            # confidence > threshold): the only branch allowed
                            # to originate a fixed-layer calibration event.
                            # hidden_states here is the raw hidden entering
                            # block shallow_exit_layer (h6) for THIS token --
                            # captured now because a later no-crossing token
                            # must never donate its own identity to this
                            # deferred event. The exact deep hidden/K/V
                            # population is filled in later, at flush time,
                            # inside parallel_gen_token.
                            fixed_layer_calibration_record = None
                            if self._fixed_layer_exact_cache_calibration_active():
                                fixed_layer_calibration_record = {
                                    "sample_context": dict(self._missing_kv_sample_context_fields()),
                                    "generation_index": int(self._generation_index),
                                    "decoder_position": int(decoder_position) if decoder_position is not None else None,
                                    "confidence": float(conf),
                                    "threshold": float(getattr(self.config, "shallow2deep_conf_threshold")),
                                    "shallow_selected_token_id": fixed_layer_calibration_shallow_selected_token_id,
                                    "selected_token_committed": False,
                                    "hidden_prefix_0_to_source_layer": [
                                        tensor.detach().cpu().contiguous()
                                        for tensor in exact_cache_calibration_prefix_hidden_states
                                    ]
                                    + [hidden_states.detach().cpu().contiguous()],
                                }
                                self.exact_cache_calibration_collector.record_fixed_layer_exit_observed()
                            self.stack_fixed_layer_calibration_records = self.stack_fixed_layer_calibration_records + (
                                fixed_layer_calibration_record,
                            )
                            if isinstance(self.stack_hidden_states, tuple) and isinstance(self.stack_hidden_metadata, tuple):
                                assert len(self.stack_hidden_states) == len(self.stack_hidden_metadata)
                                assert len(self.stack_source_kv_records) == len(self.stack_hidden_metadata)
                                assert len(self.stack_phase3c_source_hidden_records) == len(self.stack_hidden_metadata)
                                assert len(self.stack_fixed_layer_calibration_records) == len(self.stack_hidden_metadata)
                            self._tail_pending_recorded = False
                        pending_metadata_trace, pending_metadata_truncated = metadata_list_to_trace(self.stack_hidden_metadata)
                        metadata_positions = candidate_positions_from_metadata(self.stack_hidden_metadata)
                        metadata_positions_available = bool(metadata_positions and any(position is not None for position in metadata_positions))
                        metadata_positions_trace = metadata_positions
                        if metadata_positions_trace is not None and len(metadata_positions_trace) > 128:
                            metadata_positions_trace = metadata_positions_trace[:128]
                            pending_metadata_truncated = True
                        
                        self.kv_trace.record(
                            "shallow_deep_skip",
                            exit_layer=i,
                            stack_hidden_states_len_before=stack_hidden_states_len_before,
                            stack_hidden_states_len_after=len(self.stack_hidden_states),
                            stack_hidden_metadata_len_before=stack_hidden_metadata_len_before,
                            stack_hidden_metadata_len_after=len(self.stack_hidden_metadata),
                            use_cache=use_cache,
                            parallel_gen_token=self.config.parallel_gen_token,
                            copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
                            confidence=conf,
                            nominal_threshold=nominal_threshold,
                            effective_threshold=effective_threshold,
                            effective_threshold_available=effective_threshold_available,
                            use_adapt_threshold=use_adapt_threshold,
                            skip_decision_reason=skip_decision_reason,
                            decoder_position=decoder_position,
                            decoder_position_source=(
                                source_kv_record.get("decoder_position_source")
                                if isinstance(source_kv_record, dict)
                                else ("past_key_values" if decoder_position is not None else "missing")
                            ),
                            decoder_position_fallback_used=(
                                bool(source_kv_record.get("decoder_position_fallback_used"))
                                if isinstance(source_kv_record, dict)
                                else False
                            ),
                            generation_index=self._generation_index,
                            source_layer_for_restoration=source_layer_for_restoration,
                            legacy_source_kv_layer=legacy_source_kv_layer,
                            phase3c_source_hidden_layer=phase3c_source_hidden_layer,
                            catchup_start_layer=catchup_start_layer,
                            layer0_past_seq_len=layer0_past_seq_len,
                            skipped_token_metadata=metadata_to_trace_dict(skipped_metadata),
                            source_kv_record=source_kv_record,
                            phase3c_source_hidden_record=self._phase3c_source_hidden_record_trace(phase3c_source_hidden_record),
                            pending_metadata=pending_metadata_trace,
                            pending_metadata_count=len(self.stack_hidden_metadata),
                            pending_metadata_truncated=pending_metadata_truncated,
                            metadata_positions=metadata_positions_trace,
                            metadata_positions_available=metadata_positions_available,
                        )
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
                        self._f2a_update_reference_decision(
                            reference_decision_type="fixed_shallow_skip",
                            reference_decision_layer=int(i),
                            source_layer_mode=SOURCE_LAYER_MODE_FIXED,
                            full_depth_fallback=False,
                            exact_catchup_flush_occurred=False,
                            pending_token_count=len(self.stack_hidden_metadata),
                        )
                        break

                    if not skip_mask:
                        self.shallow2deep = True
                        fixed_layer_calibration_active = self._fixed_layer_exact_cache_calibration_active()
                        # if self.config.parallel_gen_token:
                        if self.config.parallel_gen_token and len(self.stack_hidden_states):
                            self.parallel_tokens_shallow += len(self.stack_hidden_states)
                            self.parallel_tokens_deep += 1
                            pending_skipped_tokens = len(self.stack_hidden_states)
                            pending_metadata = self.stack_hidden_metadata
                            if isinstance(self.stack_hidden_states, tuple) and isinstance(pending_metadata, tuple):
                                assert len(self.stack_hidden_states) == len(pending_metadata)
                                assert len(self.stack_phase3c_source_hidden_records) == len(pending_metadata)
                            pending_metadata_trace, pending_metadata_truncated = metadata_list_to_trace(pending_metadata)
                            metadata_positions = candidate_positions_from_metadata(pending_metadata)
                            metadata_positions_available = bool(metadata_positions and any(position is not None for position in metadata_positions))
                            metadata_positions_trace = metadata_positions
                            if metadata_positions_trace is not None and len(metadata_positions_trace) > 128:
                                metadata_positions_trace = metadata_positions_trace[:128]
                                pending_metadata_truncated = True
                            present_key_value_states_len_before = len(present_key_value_states) if present_key_value_states is not None else None
                            exact_catchup_trace = self._exact_catchup_trace_fields(
                                self.shallow_exit_layer,
                                pending_skipped_tokens,
                                pending_metadata_trace=pending_metadata_trace,
                                metadata_positions=metadata_positions_trace,
                                metadata_positions_available=metadata_positions_available,
                            )
                            eaes_flush_index = self._eaes_export_next_flush_index
                            self._eaes_export_next_flush_index += 1
                            eaes_export = self._maybe_export_eaes_scores_for_pending(
                                pending_metadata,
                                self.stack_source_kv_records,
                                flush_index=eaes_flush_index,
                                reason="shallow_deep_flush_start",
                            )
                            self.kv_trace.record(
                                "shallow_deep_flush_start",
                                start_layer=self.shallow_exit_layer,
                                pending_skipped_tokens=pending_skipped_tokens,
                                copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
                                present_key_value_states_len_before=present_key_value_states_len_before,
                                decoder_position=infer_decoder_position(past_key_values),
                                pending_metadata=pending_metadata_trace,
                                pending_metadata_count=len(pending_metadata),
                                pending_metadata_truncated=pending_metadata_truncated,
                                metadata_positions=metadata_positions_trace,
                                metadata_positions_available=metadata_positions_available,
                                eaes_score_export=eaes_export,
                                **exact_catchup_trace,
                            )
                            self._f2a_update_reference_decision(
                                reference_decision_type="fixed_exact_catchup_flush",
                                reference_decision_layer=int(self.shallow_exit_layer),
                                source_layer_mode=SOURCE_LAYER_MODE_FIXED,
                                full_depth_fallback=False,
                                exact_catchup_flush_occurred=True,
                                pending_token_count=int(pending_skipped_tokens),
                            )

                            # Task C2 FREE-aligned lazy BATCHED insertion.
                            # This is the ONLY behavioural change at FREE's
                            # existing flush point: instead of exact-replaying
                            # the pending tokens through the deep blocks, the
                            # already-accepted Phase-3c restoration produces
                            # all N pending tokens' deep K/V in one batched
                            # call per target layer, and the CURRENT non-exit
                            # token then simply continues through the normal
                            # deep block loop below (no break), appending its
                            # own exact K/V after the restored pending block.
                            # On any failure nothing has been mutated, so
                            # execution falls straight through to the
                            # unmodified parallel_gen_token() flush.
                            batched_installed = False
                            if self._task_c2_batched_insertion_enabled():
                                batched_result = self._try_task_c2_fixed_source6_batched_insertion(
                                    self.shallow_exit_layer,
                                    self.stack_hidden_states,
                                    pending_metadata,
                                    past_key_values,
                                    infer_decoder_position(past_key_values),
                                    encoder_hidden_states,
                                    encoder_extended_attention_mask,
                                    encoder_decoder_position_bias,
                                    head_mask,
                                    cross_attn_head_mask,
                                    use_cache,
                                    output_attentions,
                                )
                                self._missing_kv_accounting_obj().record_task_c2_batched_flush_attempt(
                                    success=batched_result["success"],
                                    fallback=not batched_result["success"],
                                    pending_token_count=batched_result["pending_token_count"],
                                    requested_units=batched_result["requested_units"],
                                    inserted_units=batched_result["inserted_units"],
                                )
                                # Guarded like the other restoration trace
                                # sites: with tracing disabled (paper-facing
                                # timing) not even the kwargs of this
                                # diagnostic record are constructed.
                                if self.kv_trace.enabled:
                                    self.kv_trace.record(
                                        "kv_task_c2_batched_insertion_transaction",
                                        source_layer=self.shallow_exit_layer,
                                        target_layers=batched_result["target_layers"],
                                        decoder_position=batched_result["decoder_position"],
                                        pending_token_count=batched_result["pending_token_count"],
                                        requested_token_layer_units=batched_result["requested_units"],
                                        inserted_token_layer_units=batched_result["inserted_units"],
                                        deep_cache_length_before=batched_result["deep_cache_length_before"],
                                        deep_cache_length_after=batched_result["deep_cache_length_after"],
                                        restoration_calls=batched_result["restoration_calls"],
                                        stacked_learned_restoration=batched_result["stacked_learned_restoration"],
                                        success=batched_result["success"],
                                        failure_stage=batched_result["failure_stage"],
                                        failure_type=batched_result["failure_type"],
                                        failure_message=batched_result["failure_message"],
                                        # A successful batched flush never runs a
                                        # deep block for any pending token and
                                        # never enters parallel_gen_token at all.
                                        pending_exact_target_block_executed=False,
                                        partial_cache_committed=False,
                                        fallback_requested=not batched_result["success"],
                                        fallback_path=(
                                            None if batched_result["success"] else "free_parallel_gen_token"
                                        ),
                                        generation_index=self._generation_index,
                                    )
                                if batched_result["success"]:
                                    past_key_values = batched_result["past_key_values"]
                                    # Single-append publication: the validated
                                    # pending K/V blocks staged by the stacked
                                    # transaction ride along for the remaining
                                    # deep layers of THIS forward pass only
                                    # (None on the sequential compatibility
                                    # path, whose candidate is pre-extended).
                                    task_c2_staged_pending_self_kv = batched_result.get(
                                        "staged_pending_self_kv"
                                    )
                                    # Consume FREE's pending stack exactly the
                                    # way parallel_gen_token() itself does.
                                    self.stack_hidden_states = ()
                                    self.stack_hidden_metadata = ()
                                    self.stack_source_kv_records = ()
                                    self.stack_phase3c_source_hidden_records = ()
                                    self.stack_fixed_layer_calibration_records = ()
                                    self._tail_pending_recorded = False
                                    batched_installed = True

                            if not batched_installed:
                                # PURE_MISSING_KV_RECOVERY_COMPUTE_COST shadow
                                # measurement: runs BEFORE the production flush
                                # over the exact same pending workload, never
                                # mutating live state. The production flush
                                # below remains the reference trajectory.
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
                                # The N pending exit events (if any) and this
                                # token's own no-crossing accounting are staged
                                # and committed inside parallel_gen_token itself,
                                # once the exact hidden/K/V population for each
                                # pending event actually exists -- never here,
                                # and never from this (no-crossing) token's own
                                # confidence/position/token-id.
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
                                    lm_head=lm_head,
                                )
                                if pure_recovery_held_spans is not None:
                                    # Validation mode: compare the pending-only
                                    # shadow K/V against the pending portion of
                                    # the production mixed flush, then discard.
                                    self._pure_recovery_validate_pending_parity(
                                        pure_recovery_held_spans, present_key_value_states
                                    )
                                    pure_recovery_held_spans = None
                                self.kv_trace.record(
                                    "shallow_deep_flush_end",
                                    start_layer=self.shallow_exit_layer,
                                    pending_skipped_tokens=pending_skipped_tokens,
                                    copy_skipped_hidden_states=self.config.copy_skipped_hidden_states,
                                    present_key_value_states_len_before=present_key_value_states_len_before,
                                    present_key_value_states_len_after=len(present_key_value_states) if present_key_value_states is not None else None,
                                    **exact_catchup_trace,
                                )

                                # Adaptive Threshold Estimator
                                if self.config.use_adapt_threshold:
                                    # Calibration Set Update
                                    self.lm_logits = self.lm_head(self.dropout(self.final_layer_norm(hidden_states)))
                                    deep_pred = self.lm_logits.argmax(-1)
                                    shallow_pred = torch.cat(self.stack_pred).argmax(-1).view(-1)

                                    self.stack_conf_all += self.stack_conf
                                    self.stack_ident_all += ((deep_pred.view(-1) == shallow_pred.view(-1)).long().cpu().numpy(),)
                                    self.stack_conf, self.stack_pred = (), ()

                                break
                            # Successful batched flush: no break. The current
                            # non-exit token falls through to the ordinary
                            # deep block loop below and executes layers
                            # source_layer..N-1 normally, now attending over
                            # the restored pending K/V as valid past.
                        elif fixed_layer_calibration_active:
                            # No pending exits to flush alongside this
                            # no-crossing token: it is never a fitting event
                            # in its own right. The block loop below simply
                            # continues normally through the remaining layers
                            # for this same token (no parallel_gen_token
                            # call); only its no-crossing accounting is
                            # staged here. The normal generation callback
                            # (_commit_exact_cache_calibration_selected_token)
                            # resolves this pending slot once the actual
                            # selected token is known.
                            self.exact_cache_calibration_collector.stage_no_crossing()

                # CALM candidate-first-crossing F2a exact-reference shadow replay.
                elif self._is_f2a_calm_enabled() and not skip_mask:
                    candidate_layers = tuple(int(item) for item in CALM_CANDIDATE_LAYERS)
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
                        threshold = float(CALM_THRESHOLD)
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
                            "kv_f2a_calm_candidate_evaluation",
                            **decision,
                            candidate_layers=list(candidate_layers),
                            decoder_position=infer_decoder_position(past_key_values),
                            source_hidden_semantics="raw_block_input_h_{}".format(i),
                        )
                        if self.config.use_synchronize:
                            torch.cuda.synchronize()
                        self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                        if candidate_pass:
                            self._f2a_update_reference_decision(
                                reference_decision_type="first_crossing",
                                reference_decision_layer=int(i),
                                source_layer_mode=SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
                                first_crossing_source_layer=int(i),
                                full_depth_fallback=False,
                                exact_catchup_flush_occurred=True,
                                pending_token_count=1,
                                candidate_evaluations=list(calm_phase3c_candidate_evaluations),
                            )
                            self._finalize_f2a_pending_calm_event(
                                followup_query_initial_hidden=f2a_followup_query_initial_hidden,
                                followup_reference_decision_layer=i,
                                followup_reference_decision_type="first_crossing",
                                followup_reference_logits=lm_logits,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_extended_attention_mask=encoder_extended_attention_mask,
                                encoder_decoder_position_bias=encoder_decoder_position_bias,
                                head_mask=head_mask,
                                cross_attn_head_mask=cross_attn_head_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                lm_head=lm_head,
                            )
                            source_hidden = hidden_states.detach().clone()
                            self.lm_logits = lm_logits
                            self._missing_kv_accounting_obj().record_calm_first_crossing(i)
                            self._missing_kv_accounting_obj().record_nominal_skip(i, len(self.block), 1)
                            hidden_states, present_key_value_states, _f2a_event = (
                                self._run_f2a_calm_exact_reference_and_shadow_replay(
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
                                    lm_head=lm_head,
                                )
                            )
                            break
                        self.block_op[i] += 1
                        if int(i) == int(candidate_layers[-1]):
                            self._f2a_update_reference_decision(
                                reference_decision_type="full_depth_fallback",
                                reference_decision_layer=len(self.block) - 1,
                                source_layer_mode=SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
                                first_crossing_source_layer=None,
                                fallback_trigger_candidate_layer=int(i),
                                full_depth_fallback=True,
                                exact_catchup_flush_occurred=False,
                                pending_token_count=0,
                                candidate_evaluations=list(calm_phase3c_candidate_evaluations),
                            )
                            self._missing_kv_accounting_obj().record_calm_full_depth_fallback()
                            self._missing_kv_accounting_obj().record_f2a_full_depth_fallback()
                            self.kv_trace.record(
                                "kv_f2a_calm_full_depth_fallback",
                                candidate_evaluations=list(calm_phase3c_candidate_evaluations),
                                decoder_position=infer_decoder_position(past_key_values),
                                restoration_requested=False,
                                exact_reference_duplicate_branch=False,
                                f2a_event_recorded=False,
                                speed_claim_valid=False,
                            )

                # CALM candidate-first-crossing Task C1 exact-overwrite runtime.
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
                        lm_logits = compute_exit_lm_logits(_hidden_states, lm_head, self.config)
                            
                        skip_mask = get_skip_mask(
                            lm_logits,
                            _hidden_states,
                            cm_head,
                            config=self.config,
                            pos_time=past_key_values[i][0].shape[2] + 1 if past_key_values[i] is not None else 1
                        )
                        if not skip_mask: self.block_op[i] += 1                    
                        if skip_mask:
                            decoder_position = infer_decoder_position(past_key_values)
                            layer0_past_seq_len = safe_cache_seq_len(past_key_values[0]) if len(past_key_values) else None
                            self.kv_trace.record(
                                "early_exit",
                                exit_layer=i,
                                exit_min_layer=self.exit_min_layer,
                                exit_conf_type=self.config.exit_conf_type,
                                exit_conf_threshold=self.config.exit_conf_threshold,
                                decoder_position=decoder_position,
                                layer0_past_seq_len=layer0_past_seq_len,
                            )
                            if self._calm_runtime_restoration_enabled():
                                calm_runtime_restoration_context = self._build_calm_runtime_restoration_context(
                                    exit_start_layer=i,
                                    present_key_value_states=present_key_value_states,
                                    past_key_values=past_key_values,
                                    decoder_position=decoder_position,
                                    confidence=None,
                                )
                                self.kv_trace.record(
                                    "kv_calm_runtime_restoration_token_start",
                                    **self._calm_context_trace_fields(calm_runtime_restoration_context),
                                )
                            self.lm_logits = lm_logits
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                    
                # Normal framework
                elif (not self.use_shallow_deep and not self.use_early_exit):
                    self.block_op[i] += 1
                
            if exact_cache_calibration_prefix_hidden_states is not None:
                if len(exact_cache_calibration_prefix_hidden_states) != int(i):
                    raise ValueError("calibration_prefix_hidden_capture_order_invalid")
                exact_cache_calibration_prefix_hidden_states.append(hidden_states.detach().cpu().contiguous())
            past_key_value = past_key_values[i]
            task_c2_staged_pending_for_layer = (
                task_c2_staged_pending_self_kv.get(i)
                if task_c2_staged_pending_self_kv is not None
                else None
            )
            # Diagnostic-only LOGICAL past length: with the single-append
            # layout the physical past_key_value deliberately still holds
            # only the old cache while the restored pending block is staged
            # privately, so position bookkeeping must add the pending count.
            # None whenever nothing is staged, which keeps every other path's
            # diagnostics byte-identical. Metadata-only; no scans, no sync.
            task_c2_logical_past_len = None
            if task_c2_staged_pending_for_layer is not None:
                staged_key_block, staged_value_block = task_c2_staged_pending_for_layer
                if int(staged_key_block.shape[2]) != int(staged_value_block.shape[2]):
                    raise ValueError("task_c2_staged_pending_kv_length_mismatch")
                task_c2_logical_past_len = int(safe_cache_seq_len(past_key_value) or 0) + int(
                    staged_key_block.shape[2]
                )
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
                layer_idx=i,
                kv_importance_tracker=self.kv_importance if self.is_decoder else None,
                staged_pending_self_kv=task_c2_staged_pending_for_layer,
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
            if (
                use_cache
                and self._calm_runtime_restoration_enabled()
                and skip_mask
                and calm_runtime_restoration_context is not None
                and i >= int(calm_runtime_restoration_context.get("exit_start_layer", len(self.block)))
            ):
                present_key_value_state, calm_runtime_restoration_event = self._apply_calm_runtime_restoration_to_present_kv(
                    i,
                    present_key_value_state,
                    past_key_value,
                    calm_runtime_restoration_context,
                )
                if self.kv_trace.enabled:
                    self.kv_trace.record("kv_calm_runtime_restoration_layer", **calm_runtime_restoration_event)
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + [present_key_value_state,]
            self._maybe_dump_all_layer_calib_kv(
                layer_idx=i,
                present_key_value_state=present_key_value_state,
                past_key_value=past_key_value,
                skip_mask=skip_mask,
                logical_past_len=task_c2_logical_past_len,
            )
            if self.is_decoder and getattr(self.config, "kv_all_layer_hidden_dump_enabled", False):
                self_attn_layer = layer_module.layer[0]
                self._maybe_dump_all_layer_hidden_states(
                    layer_idx=i,
                    raw_hidden_states=getattr(self_attn_layer, "_last_raw_hidden_states_for_hidden_dump", None),
                    normed_hidden_states=getattr(self_attn_layer, "_last_normed_hidden_states_for_hidden_dump", None),
                    past_key_value=past_key_value,
                    skip_mask=skip_mask,
                    logical_past_len=task_c2_logical_past_len,
                )
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if (
            self._is_f2a_calm_enabled()
            and getattr(self, "_f2a_pending_calm_event", None) is not None
            and self.lm_logits is None
        ):
            fallback_logits = self._f2a_logits_from_hidden(hidden_states, lm_head)
            self._finalize_f2a_pending_calm_event(
                followup_query_initial_hidden=f2a_followup_query_initial_hidden,
                followup_reference_decision_layer=len(self.block) - 1,
                followup_reference_decision_type="full_depth_fallback",
                followup_reference_logits=fallback_logits,
                encoder_hidden_states=encoder_hidden_states,
                encoder_extended_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lm_head=lm_head,
            )
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


class DeployT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.static_exit_layer = None
        self.encoder = DeployT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = DeployT5Stack(decoder_config, self.shared)

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
        self.bmm_update_max_iter = 300
        
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
            'time_estimate_conf': datetime.timedelta(),
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
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)

        if self.decoder.shallow2deep: 
            self.decoder.stack_conf, self.decoder.stack_pred = (), ()
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

    def flush_kv_trace(self, path=None):
        trace_path = path if path is not None else getattr(self.config, "kv_trace_path", None)
        if trace_path is None or not hasattr(self.decoder, "kv_trace"):
            return
        self.decoder.kv_trace.flush(trace_path)

    def _record_tail_pending_skips(self, reason="generation_end"):
        if hasattr(self, "decoder") and hasattr(self.decoder, "record_tail_pending_skips"):
            self.decoder.record_tail_pending_skips(reason=reason)

    def _maybe_flush_kv_trace(self):
        if (
            getattr(self.config, "kv_trace_enabled", False)
            and getattr(self.config, "kv_trace_path", None) is not None
            and getattr(self.config, "kv_trace_flush_on_generate_end", True)
        ):
            self.flush_kv_trace()

    def dump_kv_importance(self, path=None):
        dump_path = path if path is not None else getattr(self.config, "kv_importance_dump_path", None)
        if dump_path is None or not hasattr(self.decoder, "kv_importance"):
            return
        self.decoder.kv_importance.dump(dump_path)

    def _maybe_dump_kv_importance(self):
        if (
            getattr(self.config, "kv_importance_enabled", False)
            and getattr(self.config, "kv_importance_dump_path", None) is not None
            and getattr(self.config, "kv_importance_flush_on_generate_end", True)
        ):
            self.dump_kv_importance()

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
        f2a_terminal_reason = "unknown_generation_end"

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
            f2a_decoder = getattr(self, "decoder", None)
            f2a_context_set = False
            if f2a_decoder is not None and hasattr(f2a_decoder, "set_f2a_generation_step_context"):
                try:
                    prefix_ids = input_ids.detach().cpu().reshape(input_ids.shape[0], -1)[0].tolist()
                    current_input = int(prefix_ids[-1])
                    current_input_position = int(len(prefix_ids) - 1)
                    predicted_position = int(len(prefix_ids))
                    f2a_decoder.set_f2a_generation_step_context(
                        prefix_token_ids=prefix_ids,
                        current_decoder_input_token_id=current_input,
                        current_decoder_input_position=current_input_position,
                        predicted_token_position=predicted_position,
                    )
                    f2a_context_set = True
                except Exception:
                    if bool(getattr(getattr(f2a_decoder, "config", None), "kv_f2a_frozen_schedule_enabled", False)):
                        raise

            # forward pass to get next token
            try:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            except Exception:
                calibration_collector = getattr(self.decoder, "exact_cache_calibration_collector", None)
                if calibration_collector is not None:
                    calibration_collector.abort_pending()
                if hasattr(self.decoder, "finalize_f2a_generation_end"):
                    try:
                        self._record_tail_pending_skips(reason="generation_failure")
                        self.decoder.finalize_f2a_generation_end(terminal_reason="generation_failure")
                        self._maybe_flush_kv_trace()
                    except Exception as cleanup_exc:
                        try:
                            setattr(self.decoder, "_f2a_generation_failure_cleanup_error", repr(cleanup_exc))
                            if hasattr(self.decoder, "_missing_kv_accounting_obj"):
                                self.decoder._missing_kv_accounting_obj().record_f2a_schedule_validation_failure()
                        except Exception:
                            pass
                raise
            finally:
                if f2a_context_set and hasattr(f2a_decoder, "clear_f2a_generation_step_context"):
                    f2a_decoder.clear_f2a_generation_step_context()

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

            if hasattr(self.decoder, "_commit_exact_cache_calibration_selected_token"):
                self.decoder._commit_exact_cache_calibration_selected_token(next_tokens)

            if f2a_decoder is not None and hasattr(f2a_decoder, "record_f2a_reference_selected_token"):
                f2a_decoder.record_f2a_reference_selected_token(next_tokens.reshape(-1)[0])

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
                    f2a_terminal_reason = "eos"

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True
                if f2a_terminal_reason != "eos":
                    try:
                        configured_max_length = getattr(stopping_criteria, "max_length", None)
                        f2a_terminal_reason = (
                            "max_length"
                            if configured_max_length is not None and int(input_ids.shape[-1]) >= int(configured_max_length)
                            else "stopping_criteria"
                        )
                    except Exception:
                        f2a_terminal_reason = "stopping_criteria"

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        # Calibration-only exact tail finalization: only reachable here,
        # after the selected token is committed, appended to input_ids, the
        # stopping criterion evaluated, and the loop broken -- i.e. after
        # generation termination is fully fixed. Uses model_kwargs, which
        # still holds the exact live tensors the last successful decoder
        # forward() call used (past_key_values updated by
        # _update_model_kwargs_for_generation; encoder_outputs/attention_mask
        # /head_mask/cross_attn_head_mask are encoder-decoder-persistent and
        # untouched by it) -- never reconstructed from stored metadata.
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
                # generation or change its already-final result. Record it as
                # a failure (not silently dropped) so the fail-closed
                # record_tail_pending_skips() fallback below and the
                # collector's own status/conservation checks still see it.
                _tail_collector = getattr(self.decoder, "exact_cache_calibration_collector", None)
                if _tail_collector is not None and _tail_pending_count_before > 0:
                    _tail_collector.record_tail_finalization_failure(
                        _tail_pending_count_before,
                        stage="finalize_pending_fixed_layer_calibration_exits_call",
                        failure_type=type(_tail_exc).__name__,
                        reason=str(_tail_exc),
                    )
        self._record_tail_pending_skips(reason="generation_end")
        if hasattr(self.decoder, "finalize_f2a_generation_end"):
            self.decoder.finalize_f2a_generation_end(terminal_reason=f2a_terminal_reason)
        self._maybe_flush_kv_trace()
        self._maybe_dump_kv_importance()

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
