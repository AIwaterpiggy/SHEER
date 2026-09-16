from .kv_trace import (
    KVTraceRecorder,
    infer_decoder_position,
    safe_cache_seq_len,
)
from .h2o_importance import H2OImportanceTracker
from .exact_recompute_plan import build_exact_recompute_dryrun_plan
from .exact_recompute_verify import (
    build_exact_recompute_verification_event,
    expected_self_attn_kv_shapes,
    gather_recompute_hidden_states,
)
from .h2o_mask_plan import build_h2o_mask_noop_plan, build_h2o_mask_plan
from .h2o_masking import build_h2o_key_mask_indices
from .position_bookkeeping import (
    SkippedTokenMetadata,
    candidate_positions_from_metadata,
    infer_pending_start_position_from_metadata,
    make_skipped_token_metadata,
    metadata_list_to_trace,
    metadata_to_trace_dict,
)
from .selection_policy import (
    resolve_importance_layer,
    select_h2o_topk_dryrun,
)
from .runtime_kv_restoration import (
    ARTIFACT_FREE_CALM_TASKC1_METHODS,
    CALM_TASKC1_RUNTIME_METHODS,
    DIRECT_SHALLOW_KV_REUSE_METHOD,
    EXACT_CATCHUP_METHOD,
    EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
    HIDDEN_PROJECTION_RUNTIME_METHODS,
    LEGACY_RUNTIME_RESTORATION_METHODS,
    PHASE3C_RUNTIME_MODE,
    PHASE3C_RUNTIME_RESTORATION_METHOD,
    RuntimeKVRestorationManager,
    RuntimeKVRestorationResult,
    SUPPORTED_RUNTIME_RESTORATION_METHODS,
)

__all__ = [
    "H2OImportanceTracker",
    "KVTraceRecorder",
    "SkippedTokenMetadata",
    "build_exact_recompute_dryrun_plan",
    "build_exact_recompute_verification_event",
    "build_h2o_key_mask_indices",
    "build_h2o_mask_noop_plan",
    "build_h2o_mask_plan",
    "candidate_positions_from_metadata",
    "expected_self_attn_kv_shapes",
    "ARTIFACT_FREE_CALM_TASKC1_METHODS",
    "CALM_TASKC1_RUNTIME_METHODS",
    "DIRECT_SHALLOW_KV_REUSE_METHOD",
    "EXACT_CATCHUP_METHOD",
    "EXIT_HIDDEN_TARGET_PROJECTION_METHOD",
    "gather_recompute_hidden_states",
    "HIDDEN_PROJECTION_RUNTIME_METHODS",
    "infer_decoder_position",
    "infer_pending_start_position_from_metadata",
    "LEGACY_RUNTIME_RESTORATION_METHODS",
    "make_skipped_token_metadata",
    "metadata_list_to_trace",
    "metadata_to_trace_dict",
    "PHASE3C_RUNTIME_MODE",
    "PHASE3C_RUNTIME_RESTORATION_METHOD",
    "resolve_importance_layer",
    "RuntimeKVRestorationManager",
    "RuntimeKVRestorationResult",
    "safe_cache_seq_len",
    "select_h2o_topk_dryrun",
    "SUPPORTED_RUNTIME_RESTORATION_METHODS",
]
