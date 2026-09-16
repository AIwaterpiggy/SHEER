import warnings
from dataclasses import dataclass, field
from typing import Optional, List


def is_behavior_changing_kv_restore_policy(policy):
    return policy in {"h2o_mask"}


@dataclass
class AdditionalArguments:
    """
    Arguments for accelerating decoder models.
    """

    # deployment scenario
    deploy_scenario: Optional[bool] = field(
        default=False, metadata={"help": ("Assume a deploying scneario for the accurate measurement.")},
    )
    use_synchronize: Optional[bool] = field(
        default=True, metadata={"help": ("Use synchronize when measuring the inference time.")},
    )
    kv_trace_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Enable in-memory KV-cache and early-exit trace recording.")},
    )
    kv_trace_path: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for KV trace records.")},
    )
    kv_trace_max_records: Optional[int] = field(
        default=100000, metadata={"help": ("Maximum number of KV trace records kept in memory before dropping new records.")},
    )
    kv_trace_flush_on_generate_end: Optional[bool] = field(
        default=True, metadata={"help": ("Flush KV trace records at the end of deploy greedy generation when a path is set.")},
    )
    kv_importance_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Enable H2O-style decoder self-attention importance tracking.")},
    )
    kv_importance_mode: Optional[str] = field(
        default="h2o_layer", metadata={"help": ("KV importance tracking mode. Currently only h2o_layer is supported.")},
    )
    kv_importance_decay: Optional[float] = field(
        default=1.0, metadata={"help": ("Exponential decay factor applied before adding new attention importance.")},
    )
    kv_importance_include_current: Optional[bool] = field(
        default=True, metadata={"help": ("Include the current key position in decoder self-attention importance.")},
    )
    kv_importance_dump_path: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for dumping accumulated KV importance scores.")},
    )
    kv_importance_flush_on_generate_end: Optional[bool] = field(
        default=True, metadata={"help": ("Dump KV importance scores at the end of deploy greedy generation when a path is set.")},
    )
    kv_restore_policy: Optional[str] = field(
        default="none", metadata={"help": ("Selective KV restoration policy. Supports none, h2o_topk_dryrun, h2o_mask_noop, h2o_exact_recompute_dryrun, h2o_exact_recompute_verify, and h2o_mask v1.")}
    )
    kv_restore_topk: Optional[int] = field(
        default=0, metadata={"help": ("Number of pending skipped tokens that a dry-run restoration policy would select.")}
    )
    kv_restore_importance_layer: Optional[str] = field(
        default="target", metadata={"help": ("Importance layer used by dry-run restoration scoring: target, shallow_exit, or previous.")}
    )
    kv_restore_recent_window: Optional[int] = field(
        default=0, metadata={"help": ("Number of most recent pending skipped tokens to flag in dry-run restoration logs.")}
    )
    kv_restore_log_candidates: Optional[bool] = field(
        default=True, metadata={"help": ("Log dry-run restoration candidate selections into the KV trace.")}
    )
    kv_runtime_restoration_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Master switch for opt-in runtime restored-K/V quality-test paths. Disabled by default.")}
    )
    kv_runtime_restoration_calm_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: allow runtime restored-K/V overwrite in the CALM early-exit path when kv_runtime_restoration_enabled is also True. Disabled by default.")}
    )
    kv_runtime_restoration_official_free_calm_enabled: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Opt-in: route CALM Task C1 runtime restoration (kv_runtime_restoration_calm_enabled=True) "
                "through the official FREE CALM-style production early-exit candidate range/threshold "
                "(models/deploying_t5.py's existing CALM Task C1 exact-overwrite branch, not the plain "
                "State Copying use_early_exit branch) instead of the frozen historical (4,6,8,10)/0.9 "
                "candidate-first-crossing research policy. The candidate range is derived from "
                "exit_min_layer/num_layers and kv_runtime_restoration_threshold must equal "
                "exit_conf_threshold -- never a separately frozen number. The historical candidate_first_"
                "crossing policy is completely unaffected when this stays False (the default)."
            )
        },
    )
    kv_runtime_restoration_artifact: Optional[str] = field(
        default=None, metadata={"help": ("Torch .pt artifact containing runtime source->target K/V restoration maps.")}
    )
    kv_runtime_restoration_method: Optional[str] = field(
        default="source_procrustes", metadata={"help": ("Runtime K/V restoration method: source_procrustes, k_affine_v_procrustes, exact_catchup, direct_shallow_kv_reuse, exit_hidden_target_projection, or phase3c_kv_final.")}
    )
    kv_runtime_restoration_threshold: Optional[float] = field(
        default=None, metadata={"help": ("Exact artifact threshold to use for phase3c_kv_final runtime restoration. No nearest-threshold fallback is used.")}
    )
    kv_runtime_restoration_recent_exact_window: Optional[int] = field(
        default=0, metadata={"help": ("Keep the most recent N pending skipped tokens on exact FREE catch-up; restore older pending tokens.")}
    )
    kv_runtime_restoration_max_pending_tokens: Optional[int] = field(
        default=0, metadata={"help": ("If >0, disable runtime restoration for a flush with more than this many pending skipped tokens.")}
    )
    kv_runtime_restoration_debug: Optional[bool] = field(
        default=False, metadata={"help": ("Emit additional runtime restored-K/V diagnostics in the KV trace.")}
    )
    kv_runtime_restoration_force_restore_all: Optional[bool] = field(
        default=False, metadata={"help": ("Ignore recent_exact_window and restore all pending skipped tokens when runtime restoration is enabled.")}
    )
    kv_runtime_restoration_direct_insertion_enabled: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Task C2: for the Native FREE fixed-source-layer-6 shallow-deep path only, "
                "immediately restore and atomically insert missing deep-layer K/V for an exiting "
                "token instead of deferring it to the pending-buffer exact synchronized flush. "
                "Falls back to the existing Task C1 pending-buffer path on any failure. "
                "Disabled by default; requires kv_runtime_restoration_enabled=True, "
                "kv_runtime_restoration_method=phase3c_kv_final, and is mutually exclusive with "
                "kv_early_exit_exact_cache_calibration_enabled."
            )
        },
    )
    kv_runtime_restoration_batched_insertion_enabled: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "FREE-aligned lazy batched Phase-3c restoration: for the Native FREE "
                "fixed-source-layer-6 shallow-deep path only, keep FREE's existing pending-token "
                "scheduling (early exits accumulate in the pending stack untouched) but, at the "
                "existing synchronized flush point, restore ALL pending tokens' deep K/V in one "
                "batched Phase-3c operation per target layer and insert them, instead of exact "
                "deep replay. The current non-exit token then executes the normal deep path. "
                "Falls back to the existing parallel_gen_token() flush on any failure. "
                "Disabled by default; requires the same fail-closed configuration as "
                "kv_runtime_restoration_direct_insertion_enabled, and is mutually exclusive "
                "with it."
            )
        },
    )
    kv_runtime_restoration_finite_validation_enabled: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Diagnostic finite-value validation performed by the Batched Task C2 restoration "
                "transaction (source-hidden precondition scan, native target-6 restored K/V span "
                "scans, the aggregate learned K/V bank scans, untrusted cross-K/V scans, their "
                "validation-only scalar readbacks, and the one-time fitted-parameter NaN/Inf "
                "validation at cold stacked-bank build). Defaults to True: correctness/debug "
                "behavior is unchanged unless this is EXPLICITLY set False for paper-facing "
                "timing runs after correctness has been established. The restoration arithmetic, "
                "shape/dtype/device structural checks, cache staging and atomic publication are "
                "identical in both modes; the structural/exception fallback machinery also "
                "remains available in both, but with validation off the nonfinite-DETECTION "
                "fallback intentionally cannot trigger, because that diagnostic is disabled."
            )
        },
    )
    kv_pure_recovery_cost_enabled: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Measurement-only shadow instrumentation for the paper metric "
                "PURE_MISSING_KV_RECOVERY_COMPUTE_COST. On each real Native FREE flush with "
                "P>0 pending early-exit tokens, times (a) a pending-token-only exact deep "
                "replay and (b) the frozen Phase-3c stacked restoration over the SAME pending "
                "source hidden states, using the existing MissingKVComponentTimer CUDA-event "
                "backend. The live generation trajectory remains ordinary FREE Exact; shadow "
                "results are discarded and never published to the cache. Requires the frozen "
                "Native FREE Exact paper configuration with runtime restoration DISABLED and "
                "kv_runtime_restoration_artifact set (used only to build the shadow restorer)."
            )
        },
    )
    kv_pure_recovery_cost_output: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Output JSON path for the pure missing-K/V recovery cost summary "
                "(aggregates plus bounded per-event scalar rows; no tensors)."
            )
        },
    )
    kv_pure_recovery_cost_validation_enabled: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Additionally validate, per measured flush, that the pending-only exact "
                "replay K/V match the pending portion of the production mixed FREE flush "
                "(shape + allclose with recorded max abs/rel differences). Diagnostic-only; "
                "holds the shadow tensors until the flush completes, then discards them. "
                "Native FREE protocol only: the Official CALM three-arm measurement has no "
                "shadow exact replay to compare (its Exact arm IS the live replay) and its "
                "State/Ours arms are approximations by design, so this flag must stay False "
                "there."
            )
        },
    )
    kv_pure_recovery_cost_phase3c_artifact: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Measurement-only Phase-3c artifact path for the Official CALM three-arm "
                "PURE_MISSING_KV_RECOVERY_COMPUTE_COST shadow (Ours arm). Deliberately "
                "separate from kv_runtime_restoration_artifact: the live Official CALM + "
                "Exact runtime is artifact-free exact_catchup and its runtime fields are "
                "never overloaded for the measurement. Required (with the SHA field below) "
                "when kv_pure_recovery_cost_enabled is used on the Official CALM protocol; "
                "ignored by the Native FREE protocol."
            )
        },
    )
    kv_pure_recovery_cost_phase3c_artifact_sha256: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Approved lowercase 64-hex SHA-256 of kv_pure_recovery_cost_phase3c_artifact, "
                "verified against the actual file BEFORE deserialization; any mismatch fails "
                "closed and the shadow restorer is never built."
            )
        },
    )
    kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Development-only opt-in allowing Native FREE fixed source layer 6 to consume only "
                "the source-6 subset of a candidate-first-crossing Phase 3c artifact."
            )
        },
    )
    kv_runtime_restoration_artifact_sha256: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Expected SHA-256 of kv_runtime_restoration_artifact. Required, format-validated, and "
                "verified against the actual file before deserialization whenever "
                "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime is True."
            )
        },
    )
    kv_early_exit_exact_cache_calibration_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in fitting-population collector for actual first crossings on the exact_catchup trajectory.")},
    )
    kv_early_exit_exact_cache_calibration_output_dir: Optional[str] = field(
        default=None, metadata={"help": ("Output directory for bounded exact-cache calibration shards and manifests.")}
    )
    kv_early_exit_exact_cache_calibration_fitting_source_manifest: Optional[str] = field(
        default=None, metadata={"help": ("Accepted source manifest that binds the approved fitting-409 stable IDs.")}
    )
    kv_early_exit_exact_cache_calibration_approved_corrective_manifest: Optional[str] = field(
        default=None,
        metadata={"help": ("Centrally approved corrective manifest used as the fitting-population root of trust.")},
    )
    kv_early_exit_exact_cache_calibration_max_events_per_flush: Optional[int] = field(
        default=8, metadata={"help": ("Maximum detached CPU calibration events retained before a shard flush.")}
    )
    kv_early_exit_exact_cache_calibration_source_layer_mode: Optional[str] = field(
        default="candidate_first_crossing",
        metadata={
            "help": (
                "Collector source-layer policy: candidate_first_crossing (existing CALM multi-source "
                "trajectory, frozen (4,6,8,10)/0.9 policy), fixed_layer (Native FREE fixed-shallow-layer "
                "exact-cache free-running collection; legacy T5/CNN routes remain fixed at source6, while "
                "the Multi-News population-contract route requires the explicitly supplied "
                "fixed_source_layer to match the actual FREE shallow_exit_layer and the approved "
                "Multi-News contract), or official_free_calm_first_crossing (official FREE CALM-style "
                "production early-exit path; candidate range/threshold derived from exit_min_layer/"
                "num_layers/exit_conf_threshold, never the frozen historical tuple)."
            )
        },
    )
    kv_early_exit_exact_cache_calibration_fixed_source_layer: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Required when kv_early_exit_exact_cache_calibration_source_layer_mode is fixed_layer. "
                "Legacy Native FREE T5/CNN routes require 6. The Multi-News contract route requires an "
                "explicitly supplied value that matches both the runtime shallow_exit_layer and the "
                "approved population contract."
            )
        },
    )
    kv_early_exit_exact_cache_calibration_population_contract: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a centrally approved Native FREE fixed-layer train-calibration population "
                "contract. CNN/DailyMail uses its existing source6 contract (see "
                "our_kv_restoration.cnndm_source6_population_contract); Multi-News uses its own separately "
                "typed approved contract. The two contract implementations stay separate and are never "
                "interchangeable. Supplying this opts the exact-cache calibration collector into the "
                "approved train-population branch instead of the SAMSum legacy corrective/source-manifest "
                "branch. Requires kv_early_exit_exact_cache_calibration_enabled=True, "
                "source_layer_mode=fixed_layer, an explicit fixed_source_layer matching the route's "
                "approved contract, kv_early_exit_exact_cache_calibration_dataset_split=train, and "
                "missing_kv_provenance_enabled=True."
            )
        },
    )
    kv_early_exit_exact_cache_calibration_population_contract_sha256: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Expected SHA-256 of kv_early_exit_exact_cache_calibration_population_contract. Required "
                "whenever the contract path is set; verified against the actual file before any contract "
                "content is trusted -- the contract file is never self-authorizing."
            )
        },
    )
    kv_early_exit_exact_cache_calibration_dataset_split: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Opt-in effective evaluation split for --do_eval. Only the explicit value 'train' is "
                "supported, and only together with an approved fixed-layer train-calibration population "
                "contract above (CNN/DailyMail or Multi-News); any other combination fails closed. Leave "
                "unset for the default validation split."
            )
        },
    )
    kv_runtime_component_timing_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in missing-KV component timing. Disabled by default to avoid timing overhead.")}
    )
    kv_runtime_component_timing_backend: Optional[str] = field(
        default="auto", metadata={"help": ("Missing-KV component timing backend: auto, cuda_events, cpu_perf_counter, or disabled.")}
    )
    kv_runtime_accounting_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for aggregate missing-KV runtime accounting summary.")}
    )
    missing_kv_per_sample_accounting_output: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional JSONL path for one stable-sample accounting delta row per generation."
            )
        },
    )
    kv_runtime_component_timing_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for missing-KV component timing summary.")}
    )
    kv_exact_catchup_overhead_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in per-transaction exact catch-up overhead event recording. Disabled by default; independent of kv_runtime_component_timing_enabled.")},
    )
    kv_exact_catchup_event_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for exact catch-up overhead events.")}
    )
    kv_exact_catchup_summary_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for the cross-validated exact catch-up overhead aggregate summary.")}
    )
    kv_exact_catchup_csv_dir: Optional[str] = field(
        default=None, metadata={"help": ("Optional directory for exact catch-up overhead overall/by-runtime-path/by-source/by-gap CSV tables.")}
    )
    kv_generation_timing_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in generation-only wall-time timer around gen_model.generate() only. Disabled by default; independent of kv_exact_catchup_overhead_enabled.")},
    )
    kv_generation_timing_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for the generation-only wall-time timing summary.")}
    )
    kv_exact_catchup_require_complete_events: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in: fail evaluation if the exact catch-up overhead event population is incomplete or fails cross-validation.")},
    )
    kv_f2a_frozen_schedule_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in F2a frozen-prefix/frozen-exit-schedule shadow replay. Disabled by default.")},
    )
    kv_f2a_source_layer_mode: Optional[str] = field(
        default="fixed_layer",
        metadata={"help": ("F2a source mode: fixed_layer or candidate_first_crossing.")},
    )
    kv_f2a_policy_artifact: Optional[str] = field(
        default=None,
        metadata={"help": ("Phase 3c artifact used only for F2a candidate shadow replay.")},
    )
    kv_f2a_schedule_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSONL output path for F2a frozen schedule events.")},
    )
    kv_f2a_records_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSONL output path for F2a per-method replay records.")},
    )
    kv_f2a_summary_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSON output path for F2a schedule/replay summary.")},
    )
    kv_f2a_reference_trajectory_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSONL output path for F2a reference generation trajectory rows.")},
    )
    kv_f2a_reference_trajectory_summary_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSON output path for F2a reference generation trajectory summary.")},
    )
    kv_f2a_methods: Optional[str] = field(
        default="exit_hidden_target_projection,exit_conditioned_hidden_restoration,final_kv_restoration",
        metadata={"help": ("Comma-separated F2a component methods. Paper default is the three primary variants.")},
    )
    kv_f2a_max_events: Optional[int] = field(
        default=0,
        metadata={"help": ("Optional cap for engineering F2a event capture. Zero means uncapped.")},
    )
    kv_f2a_write_debug_logits: Optional[bool] = field(
        default=False,
        metadata={"help": ("Engineering-only: include capped raw logits in F2a records. Disabled for paper-facing output.")},
    )
    kv_f2a_max_debug_logit_records: Optional[int] = field(
        default=0,
        metadata={"help": ("Maximum F2a records with debug logits when kv_f2a_write_debug_logits is enabled.")},
    )
    kv_f2a_decoding_configuration_sha256: Optional[str] = field(
        default=None,
        metadata={"help": ("Paper-facing F2a decoding-configuration identity used for run trajectory binding.")},
    )
    save_eval_predictions: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: export eval predictions already generated by SumTrainer evaluation.")}
    )
    eval_predictions_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for eval prediction export. If unset, writes eval_predictions.jsonl under output_dir.")}
    )
    missing_kv_provenance_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: write missing-KV dump provenance manifests and bind generated rows to effective eval samples.")}
    )
    missing_kv_effective_population_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for the effective evaluation population manifest.")}
    )
    missing_kv_effective_population_summary_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for the effective evaluation population summary.")}
    )
    missing_kv_selected_stable_sample_ids_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional JSON/JSONL file listing stable_sample_id values to select from the validation population "
                "after provenance stable IDs are derived. Used by F2b artifact-held-out evaluation."
            )
        },
    )
    missing_kv_f2b_population_identity_file: Optional[str] = field(
        default=None,
        metadata={"help": ("Optional F2b artifact-held-out population identity JSON to embed in provenance summaries.")},
    )
    missing_kv_generation_binding_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path binding effective samples to generation indices.")}
    )
    missing_kv_generation_binding_summary_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for generation-to-sample binding validation summary.")}
    )
    missing_kv_tokenizer_identity_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for tokenizer semantic identity metadata.")}
    )
    missing_kv_checkpoint_inventory_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for F2b local checkpoint inventory identity.")}
    )
    missing_kv_decoding_configuration_identity_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for F2b effective decoding configuration identity.")}
    )
    missing_kv_candidate_policy_identity_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for F2b effective candidate-policy identity.")}
    )
    missing_kv_runtime_method_identity_output: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSON path for F2b runtime method identity.")}
    )
    missing_kv_dump_storage_format: Optional[str] = field(
        default="legacy_row_v1",
        metadata={"help": ("Missing-KV all-layer dump storage format: legacy_row_v1 or packed_generation_v1.")},
    )
    kv_calm_counterfactual_trace_enabled: Optional[bool] = field(
        default=False,
        metadata={"help": ("Opt-in: record counterfactual CALM candidate first-crossing confidence trace rows during full-depth calibration.")},
    )
    kv_calm_counterfactual_trace_output: Optional[str] = field(
        default=None,
        metadata={"help": ("JSONL output path for counterfactual CALM candidate confidence trace rows.")},
    )
    kv_calm_counterfactual_candidate_layers: Optional[str] = field(
        default="4,6,8,10",
        metadata={"help": ("Comma-separated decoder candidate layer indices for counterfactual CALM trace collection.")},
    )
    kv_calm_counterfactual_threshold: Optional[float] = field(
        default=0.9,
        metadata={"help": ("Strict greater-than threshold for counterfactual CALM first-crossing trace collection.")},
    )
    kv_restore_dump_eaes_scores: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: export EAES v1 received-attention scores for skipped source records. Disabled by default.")}
    )
    kv_restore_eaes_score_jsonl: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for EAES score export. If omitted, write eaes_scores.jsonl under the exact K/V dump directory when available.")}
    )
    kv_exact_catchup_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump exact FREE parallel catch-up self-attention K/V slices to .pt shards. Disabled by default.")}
    )
    kv_exact_catchup_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for exact catch-up K/V dump shards and manifest.jsonl. If omitted, use a sibling of kv_trace_path when possible.")}
    )
    kv_exact_catchup_dump_max_flushes: Optional[int] = field(
        default=4, metadata={"help": ("Maximum number of FREE parallel catch-up flushes to dump when exact K/V dumping is enabled.")}
    )
    kv_exact_catchup_dump_max_tokens_per_flush: Optional[int] = field(
        default=8, metadata={"help": ("Maximum pending skipped tokens per flush to include in exact K/V dump shards.")}
    )
    kv_exact_catchup_dump_max_layers: Optional[int] = field(
        default=4, metadata={"help": ("Maximum catch-up layers per flush to dump when exact K/V dumping is enabled.")}
    )
    kv_exact_catchup_dump_layers: Optional[str] = field(
        default=None, metadata={"help": ("Optional comma-separated catch-up layer indices to dump. Leave unset to use the first capped layers in the catch-up range.")}
    )
    kv_exact_catchup_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for exact K/V dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_exact_catchup_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move exact K/V dump tensors to CPU before saving.")}
    )
    kv_exact_catchup_dump_include_current_token: Optional[bool] = field(
        default=False, metadata={"help": ("Also dump the current deep token K/V slice after pending skipped-token slices. Disabled by default.")}
    )
    kv_source_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump source-layer self-attention K/V for skipped tokens, aligned with exact catch-up target dumps.")}
    )
    kv_source_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for source K/V shards and source_manifest.jsonl. If omitted, reuse exact catch-up dump dir.")}
    )
    kv_source_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for source K/V dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_source_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move source K/V dump tensors to CPU before saving.")}
    )
    kv_source_dump_max_tokens: Optional[int] = field(
        default=128, metadata={"help": ("Maximum skipped-token source K/V shards to dump.")}
    )
    kv_adjacent_anchor_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump exact-adjacent self-attention K/V projection anchors from skipped-token exit hidden states. Disabled by default.")}
    )
    kv_adjacent_anchor_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for adjacent-anchor K/V shards and adjacent_anchor_manifest.jsonl. If omitted, reuse source/exact K/V dump dir.")}
    )
    kv_adjacent_anchor_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for adjacent-anchor K/V dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_adjacent_anchor_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move adjacent-anchor K/V dump tensors to CPU before saving.")}
    )
    kv_adjacent_anchor_dump_max_tokens: Optional[int] = field(
        default=128, metadata={"help": ("Maximum skipped-token adjacent-anchor K/V shards to dump.")}
    )
    kv_all_layer_calib_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump decoder self-attention K/V for selected generated tokens across decoder layers for offline restoration calibration. Disabled by default.")}
    )
    kv_all_layer_calib_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for all-layer calibration K/V shards and all_layer_kv_manifest.jsonl. If omitted, reuse exact K/V dump dir when available.")}
    )
    kv_all_layer_calib_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for all-layer calibration K/V dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_all_layer_calib_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move all-layer calibration K/V tensors to CPU before saving.")}
    )
    kv_all_layer_calib_dump_max_tokens: Optional[int] = field(
        default=128, metadata={"help": ("Maximum unique decoder token positions to dump for all-layer calibration.")}
    )
    kv_all_layer_calib_dump_layers: Optional[str] = field(
        default=None, metadata={"help": ("Optional comma-separated decoder layer indices to dump for all-layer calibration. Leave unset or use all to dump all decoder layers.")}
    )
    kv_all_layer_hidden_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump decoder self-attention raw/normed hidden states aligned with all-layer K/V calibration shards. Disabled by default.")}
    )
    kv_all_layer_hidden_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for all-layer hidden-state shards and all_layer_hidden_manifest.jsonl. If omitted, reuse all-layer K/V dump dir when available.")}
    )
    kv_all_layer_hidden_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for all-layer hidden-state dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_all_layer_hidden_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move all-layer hidden-state dump tensors to CPU before saving.")}
    )
    kv_all_layer_hidden_dump_max_tokens: Optional[int] = field(
        default=128, metadata={"help": ("Maximum unique decoder token positions to dump for all-layer hidden-state calibration.")}
    )
    kv_all_layer_hidden_dump_max_flushes: Optional[int] = field(
        default=None, metadata={"help": ("Optional cap on generation/reset indices to dump for all-layer hidden-state calibration. Leave unset for no cap.")}
    )
    kv_all_layer_hidden_dump_include_raw_hidden: Optional[bool] = field(
        default=False, metadata={"help": ("Include decoder layer input hidden state before self-attention layer norm in all-layer hidden dumps.")}
    )
    kv_all_layer_hidden_dump_include_normed_hidden: Optional[bool] = field(
        default=True, metadata={"help": ("Include decoder self-attention layer-normalized hidden state before K/V projection in all-layer hidden dumps.")}
    )
    kv_all_layer_hidden_dump_layers: Optional[str] = field(
        default=None, metadata={"help": ("Optional comma-separated decoder layer indices to dump for all-layer hidden-state calibration. Leave unset or use all to dump all decoder layers.")}
    )
    kv_attention_diag_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump restricted attention-output diagnostic Q/K/V shards for exact catch-up records. Disabled by default.")}
    )
    kv_attention_diag_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for attention diagnostic shards and attention_diag_manifest.jsonl. If omitted, reuse exact catch-up dump dir.")}
    )
    kv_attention_diag_dump_max_records: Optional[int] = field(
        default=1024, metadata={"help": ("Maximum restricted attention diagnostic records to dump.")}
    )
    kv_attention_diag_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for attention diagnostic dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_attention_diag_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move attention diagnostic tensors to CPU before saving.")}
    )
    kv_attention_diag_dump_layers: Optional[str] = field(
        default=None, metadata={"help": ("Optional comma-separated target layer indices for attention diagnostic dumping. Leave unset to use dumped exact catch-up layers.")}
    )
    kv_full_attention_diag_dump_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in: dump full-cache decoder self-attention diagnostic shards for exact catch-up records. Disabled by default.")}
    )
    kv_full_attention_diag_dump_dir: Optional[str] = field(
        default=None, metadata={"help": ("Directory for full-cache attention diagnostic shards and full_attention_diag_manifest.jsonl. If omitted, reuse exact catch-up dump dir.")}
    )
    kv_full_attention_diag_dump_max_records: Optional[int] = field(
        default=256, metadata={"help": ("Maximum full-cache attention diagnostic records to dump.")}
    )
    kv_full_attention_diag_dump_dtype: Optional[str] = field(
        default="float16", metadata={"help": ("Dtype for full-cache attention diagnostic dump tensors: original, float16, bfloat16, or float32.")}
    )
    kv_full_attention_diag_dump_cpu: Optional[bool] = field(
        default=True, metadata={"help": ("Move full-cache attention diagnostic tensors to CPU before saving.")}
    )
    kv_full_attention_diag_dump_layers: Optional[str] = field(
        default=None, metadata={"help": ("Optional comma-separated target layer indices for full-cache attention diagnostic dumping. Leave unset to use dumped exact catch-up layers.")}
    )
    kv_full_attention_diag_dump_max_key_tokens: Optional[int] = field(
        default=512, metadata={"help": ("Skip full-cache attention diagnostic records whose full self-attention key length exceeds this cap.")}
    )
    kv_full_attention_diag_dump_include_position_bias: Optional[bool] = field(
        default=True, metadata={"help": ("Include the decoder self-attention position-bias tensor in full-cache diagnostic shards when available.")}
    )
    kv_restoration_dryrun_enabled: Optional[bool] = field(
        default=False, metadata={"help": ("Opt-in placeholder dry-run manifest for comparing exact catch-up K/V against runtime restoration utilities when they are wired.")}
    )
    kv_restoration_dryrun_manifest_path: Optional[str] = field(
        default=None, metadata={"help": ("Optional JSONL path for restoration dry-run records. If omitted, write under the exact K/V dump directory.")}
    )
    kv_restoration_dryrun_max_records: Optional[int] = field(
        default=100, metadata={"help": ("Maximum restoration dry-run manifest rows to write.")}
    )
    smoke_force_flush_after_skips: Optional[int] = field(
        default=None, metadata={"help": ("Smoke-only: force one shallow-deep non-skip after this many pending skipped tokens.")}
    )
    smoke_force_flush_once: Optional[bool] = field(
        default=True, metadata={"help": ("Smoke-only: force the shallow-deep flush at most once per generation.")}
    )

    # train the intermediate layers as well
    output_hidden_states_decoder: Optional[bool] = field(
        default=False, metadata={"help": ("Output all hidden states in decoder model to train intermedidate layers.")},
    )
    intermediate_loss_fn: Optional[str] = field(
        default=None, metadata={"help": ("Choose the loss function to train intermediate layers as well.")},
    )
    distill_layer_alpha: Optional[float] = field(
        default=None, metadata={"help": ("Distillation interpolation hyperparameter between CrossEntropy and KL divergence.")}
    )
    do_layer_transformation: Optional[bool] = field(
        default=False, metadata={"help": ("Whether or not use transformation for student (shallow decoder) hidden states")}
    )

    # static: output all tokens after a specific layer, not the end of decodoer layer
    static_exit_layer: Optional[int] = field(
        default=None, metadata={"help": ("Choose an exit block for all tokens (i.e., exit tokens after [static_exit_layer] block).")},
    )

    # early-exit: output tokens based on confidence in decoder layers
    use_early_exit: Optional[bool] = field(
        default=False, metadata={"help": ("Use early-exit framework in decoder model.")}
    )
    exit_conf_type: Optional[str] = field(
        default=None, metadata={"help": ("Select the type of confidence measure.")},
    )   
    exit_conf_threshold: Optional[float] = field(
        default=1., metadata={"help": ("Default threshold value for early-exit framework.")},
    )
    exit_position_temp: Optional[float] = field(
        default=None, metadata={"help": ("Temperature value for decaying confidence threshold")},
    )
    exit_min_layer: Optional[int] = field(
        default=None, metadata={"help": ("To address unstable text generation and training, exit after certain layers.")},
    )   
    train_meta_cm_head: Optional[bool] = field(
        default=False, metadata={"help": ("Train cm (confidence measure) head to align last hidden_states when exit_conf_type is set to meta.")}
    )
        
    # shallow-deep framework
    use_shallow_deep: Optional[bool] = field(
        default=False, metadata={"help": ("Use shallow-deep decoder framework in decoder model.")}
    )
    shallow_exit_layer: Optional[int] = field(
        default=None, metadata={"help": ("Number of layers for shallow decoder model.")}
    )
    shallow2deep_conf_type: Optional[str] = field(
        default=None, metadata={"help": ("Select the type of confidence measure for chaning shallow to deep decoder.")},
    )   
    shallow2deep_conf_threshold: Optional[float] = field(
        default=1., metadata={"help": ("Default threshold value in Shallow-Deep framework.")},
    )
    parallel_gen_token: Optional[bool] = field(
        default=True, metadata={"help": ("With the previous skipped tokens, generate the next token of Deep decoder in a non-autoregressive manner.")},
    )
    copy_skipped_hidden_states: Optional[bool] = field(
        default=False, metadata={"help": ("For the previous skipped tokens, copy hidden_states and generate key_value of Deep decoder.")},
    )
    parallel_causal_mask: Optional[bool] = field(
        default=True, metadata={"help": ("Using causal masking for sequence parallel computing when shallow2deep occurs.")}
    )
    rollback_conf_threshold: Optional[float] = field(
        default=None, metadata={"help": ("Default threshold value for RollBack policy in Shallow-Deep framework.")},
    )
        
    # adpative threshold estimator
    use_adapt_threshold: Optional[bool] = field(
        default=False, metadata={"help": ("Using adaptive threshold estimator for FREE framework.")},
    )
    
    # low rank adaptation
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": ("Using low-rank adaptation for large language models")}
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": ("Default rank value of lora")},
    )
    lora_alpha: Optional[float] = field(
        default=8, metadata={"help": ("Default alpha value of lora")},
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": ("Default dropout value of lora")}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": ("Change target modules of lora")}
    )
    

def update_autoconfig(config, additional_args, **kwargs):

    # assertion
    if additional_args.intermediate_loss_fn is not None:
        assert additional_args.output_hidden_states_decoder
    if additional_args.train_meta_cm_head:
        assert additional_args.output_hidden_states_decoder
        assert additional_args.intermediate_loss_fn is None  # when training cm_head, model should be fully fine-tuned
    if additional_args.use_shallow_deep:
        assert additional_args.shallow_exit_layer is not None
    if not additional_args.parallel_causal_mask:
        assert not additional_args.copy_skipped_hidden_states
        assert additional_args.rollback_conf_threshold is None
    if additional_args.rollback_conf_threshold is not None:
        assert not additional_args.copy_skipped_hidden_states
    if additional_args.kv_importance_enabled:
        assert additional_args.kv_importance_mode == "h2o_layer"
        assert additional_args.kv_importance_decay is not None
        assert additional_args.kv_importance_decay >= 0.0 and additional_args.kv_importance_decay <= 1.0
    valid_restore_policies = ["none", "h2o_topk_dryrun", "h2o_mask_noop", "h2o_exact_recompute_dryrun", "h2o_exact_recompute_verify", "h2o_mask"]
    valid_restore_importance_layers = ["target", "shallow_exit", "previous"]
    assert additional_args.kv_restore_policy in valid_restore_policies
    assert additional_args.kv_restore_importance_layer in valid_restore_importance_layers
    if additional_args.kv_restore_policy == "h2o_mask":
        unsupported_reasons = []
        if not additional_args.use_shallow_deep:
            unsupported_reasons.append("use_shallow_deep must be True")
        if not additional_args.parallel_gen_token:
            unsupported_reasons.append("parallel_gen_token must be True")
        if additional_args.copy_skipped_hidden_states:
            unsupported_reasons.append("copy_skipped_hidden_states must be False")
        if not additional_args.parallel_causal_mask:
            unsupported_reasons.append("parallel_causal_mask must be True")
        if unsupported_reasons:
            raise NotImplementedError(
                "kv_restore_policy='h2o_mask' v1 supports only T5 shallow-deep parallel flush with "
                "copy_skipped_hidden_states=False and parallel_causal_mask=True: {}".format(
                    "; ".join(unsupported_reasons)
                )
            )
    if additional_args.smoke_force_flush_after_skips is not None:
        assert additional_args.smoke_force_flush_after_skips >= 0
    valid_runtime_restoration_methods = [
        "source_procrustes",
        "k_affine_v_procrustes",
        "exact_catchup",
        "direct_shallow_kv_reuse",
        "exit_hidden_target_projection",
        "phase3c_kv_final",
    ]
    if additional_args.kv_runtime_restoration_method not in valid_runtime_restoration_methods:
        raise ValueError(
            "Unsupported kv_runtime_restoration_method: {}".format(
                additional_args.kv_runtime_restoration_method
            )
        )
    if additional_args.kv_runtime_restoration_recent_exact_window is not None:
        assert additional_args.kv_runtime_restoration_recent_exact_window >= 0
    if additional_args.kv_runtime_restoration_max_pending_tokens is not None:
        assert additional_args.kv_runtime_restoration_max_pending_tokens >= 0
    if additional_args.kv_runtime_restoration_threshold is not None:
        assert additional_args.kv_runtime_restoration_threshold >= 0.0
    if additional_args.kv_runtime_restoration_official_free_calm_enabled and not (
        additional_args.kv_runtime_restoration_enabled and additional_args.kv_runtime_restoration_calm_enabled
    ):
        raise ValueError(
            "kv_runtime_restoration_official_free_calm_enabled requires kv_runtime_restoration_enabled=True "
            "and kv_runtime_restoration_calm_enabled=True"
        )
    if additional_args.kv_runtime_restoration_enabled:
        unsupported_reasons = []
        artifact_free_taskc1_runtime = additional_args.kv_runtime_restoration_method in {
            "exact_catchup",
            "direct_shallow_kv_reuse",
            "exit_hidden_target_projection",
        }
        if not artifact_free_taskc1_runtime and not additional_args.kv_runtime_restoration_artifact:
            unsupported_reasons.append("kv_runtime_restoration_artifact must be set")
        phase3c_runtime = additional_args.kv_runtime_restoration_method == "phase3c_kv_final"
        calm_taskc1_runtime = phase3c_runtime or artifact_free_taskc1_runtime
        if artifact_free_taskc1_runtime and additional_args.kv_runtime_restoration_artifact:
            unsupported_reasons.append("{} does not use kv_runtime_restoration_artifact".format(additional_args.kv_runtime_restoration_method))
        if artifact_free_taskc1_runtime and not additional_args.kv_runtime_restoration_calm_enabled:
            unsupported_reasons.append("{} is only supported for CALM Task C1 runtime restoration".format(additional_args.kv_runtime_restoration_method))
        if calm_taskc1_runtime:
            if additional_args.kv_runtime_restoration_threshold is None:
                unsupported_reasons.append("kv_runtime_restoration_threshold must be set for CALM Task C1 runtime restoration")
            if additional_args.use_adapt_threshold:
                unsupported_reasons.append("use_adapt_threshold must be False for CALM Task C1 runtime restoration")
            if additional_args.kv_runtime_restoration_calm_enabled:
                if additional_args.kv_runtime_restoration_official_free_calm_enabled:
                    # Official FREE CALM path: the runtime restoration
                    # threshold must equal the model's own exit_conf_threshold
                    # (the value that actually governs the early-exit
                    # decision this restoration attaches to) -- never a
                    # separately frozen number. The historical branch below
                    # (else) is completely unaffected.
                    if additional_args.exit_conf_threshold is None:
                        unsupported_reasons.append(
                            "exit_conf_threshold must be set for official FREE CALM Task C1 runtime restoration"
                        )
                    elif additional_args.kv_runtime_restoration_threshold is not None and abs(
                        float(additional_args.kv_runtime_restoration_threshold)
                        - float(additional_args.exit_conf_threshold)
                    ) > 1e-12:
                        unsupported_reasons.append(
                            "kv_runtime_restoration_threshold must equal exit_conf_threshold for official "
                            "FREE CALM Task C1 runtime restoration"
                        )
                    # The Task C1 branch always computes confidence via the
                    # existing softmax top-1/top-2 margin helper and compares
                    # against a single constant threshold (no per-position
                    # decay). That is only equivalent to the original FREE
                    # get_skip_mask() decision when exit_conf_type=="softmax"
                    # (not "meta"/None) and exit_position_temp is None (no
                    # threshold decay) -- fail closed otherwise instead of
                    # silently diverging from the State Copying arm.
                    if additional_args.exit_conf_type != "softmax":
                        unsupported_reasons.append(
                            "exit_conf_type must be softmax for official FREE CALM Task C1 runtime restoration"
                        )
                    if additional_args.exit_position_temp is not None:
                        unsupported_reasons.append(
                            "exit_position_temp must be unset for official FREE CALM Task C1 runtime restoration"
                        )
                elif additional_args.kv_runtime_restoration_threshold is not None and abs(
                    float(additional_args.kv_runtime_restoration_threshold) - 0.9
                ) > 1e-12:
                    unsupported_reasons.append("kv_runtime_restoration_threshold must be 0.9 for CALM Task C1 runtime restoration")
                if not additional_args.kv_runtime_restoration_force_restore_all:
                    unsupported_reasons.append("kv_runtime_restoration_force_restore_all must be True for CALM Task C1 runtime restoration")
                if int(additional_args.kv_runtime_restoration_recent_exact_window or 0) != 0:
                    unsupported_reasons.append("kv_runtime_restoration_recent_exact_window must be 0 for CALM Task C1 runtime restoration")
                if not additional_args.use_early_exit:
                    unsupported_reasons.append("use_early_exit must be True for CALM Task C1 runtime restoration")
                if additional_args.use_shallow_deep:
                    unsupported_reasons.append("use_shallow_deep must be False for CALM Task C1 runtime restoration")
                if additional_args.static_exit_layer is not None:
                    unsupported_reasons.append("static_exit_layer must be unset for CALM Task C1 runtime restoration")
            else:
                if not additional_args.use_shallow_deep:
                    unsupported_reasons.append("use_shallow_deep must be True")
                if not additional_args.parallel_gen_token:
                    unsupported_reasons.append("parallel_gen_token must be True")
                if additional_args.copy_skipped_hidden_states:
                    unsupported_reasons.append("copy_skipped_hidden_states must be False")
        elif additional_args.kv_runtime_restoration_calm_enabled:
            if not additional_args.use_early_exit:
                unsupported_reasons.append("use_early_exit must be True for CALM runtime restoration")
            if additional_args.use_shallow_deep:
                unsupported_reasons.append("use_shallow_deep must be False for CALM runtime restoration")
        else:
            if not additional_args.use_shallow_deep:
                unsupported_reasons.append("use_shallow_deep must be True")
            if not additional_args.parallel_gen_token:
                unsupported_reasons.append("parallel_gen_token must be True")
            if additional_args.copy_skipped_hidden_states:
                unsupported_reasons.append("copy_skipped_hidden_states must be False")
        if unsupported_reasons:
            raise NotImplementedError(
                "Runtime restored-K/V insertion currently supports either FREE shallow-deep "
                "parallel catch-up with copy_skipped_hidden_states=False or the opt-in CALM "
                "quality-test overwrite path: {}".format(
                    "; ".join(unsupported_reasons)
                )
            )
    if (
        additional_args.kv_runtime_restoration_direct_insertion_enabled
        and additional_args.kv_runtime_restoration_batched_insertion_enabled
    ):
        # Immediate (per-exit direct insertion) and FREE-aligned lazy batched
        # insertion are two mutually exclusive execution SCHEDULES over the
        # same accepted Phase-3c restoration. Enabling both is a
        # configuration error, never a silent pick-one.
        raise NotImplementedError(
            "kv_runtime_restoration_direct_insertion_enabled (Immediate Task C2) and "
            "kv_runtime_restoration_batched_insertion_enabled (FREE-aligned lazy batched "
            "Task C2) are mutually exclusive execution schedules; enable exactly one."
        )
    if (
        additional_args.kv_runtime_restoration_direct_insertion_enabled
        or additional_args.kv_runtime_restoration_batched_insertion_enabled
    ):
        # Task C2: fail-closed to exactly the supported Native FREE
        # fixed-source-layer-6 configuration. parallel_gen_token=True stays
        # required even though Task C2 success never calls
        # parallel_gen_token() itself, because a failed direct-insertion
        # attempt falls back to the existing Task C1 pending-buffer path,
        # whose later synchronized exact catch-up depends on it. The batched
        # schedule requires exactly the same configuration and falls back to
        # exactly the same path, so both share one validation block.
        task_c2_mode_label = (
            "Task C2 direct K/V insertion"
            if additional_args.kv_runtime_restoration_direct_insertion_enabled
            else "Task C2 FREE-aligned lazy batched K/V insertion"
        )
        direct_insertion_reasons = []
        if not additional_args.kv_runtime_restoration_enabled:
            direct_insertion_reasons.append("kv_runtime_restoration_enabled must be True")
        if additional_args.kv_runtime_restoration_method != "phase3c_kv_final":
            direct_insertion_reasons.append("kv_runtime_restoration_method must be phase3c_kv_final")
        if not additional_args.kv_runtime_restoration_artifact:
            direct_insertion_reasons.append("kv_runtime_restoration_artifact must be set")
        if not additional_args.use_shallow_deep:
            direct_insertion_reasons.append("use_shallow_deep must be True")
        if additional_args.use_early_exit:
            direct_insertion_reasons.append("use_early_exit must be False")
        if additional_args.shallow_exit_layer != 6:
            direct_insertion_reasons.append("shallow_exit_layer must be 6")
        if additional_args.shallow2deep_conf_threshold != 0.9:
            direct_insertion_reasons.append("shallow2deep_conf_threshold must be 0.9")
        if additional_args.kv_runtime_restoration_threshold is None:
            direct_insertion_reasons.append("kv_runtime_restoration_threshold must be set")
        elif abs(float(additional_args.kv_runtime_restoration_threshold) - 0.9) > 1e-12:
            direct_insertion_reasons.append("kv_runtime_restoration_threshold must be 0.9")
        if additional_args.use_adapt_threshold:
            direct_insertion_reasons.append("use_adapt_threshold must be False")
        if not additional_args.parallel_gen_token:
            direct_insertion_reasons.append("parallel_gen_token must be True")
        if not additional_args.parallel_causal_mask:
            direct_insertion_reasons.append("parallel_causal_mask must be True")
        if additional_args.copy_skipped_hidden_states:
            direct_insertion_reasons.append("copy_skipped_hidden_states must be False")
        if additional_args.kv_runtime_restoration_calm_enabled:
            direct_insertion_reasons.append("kv_runtime_restoration_calm_enabled must be False")
        if not additional_args.kv_runtime_restoration_force_restore_all:
            direct_insertion_reasons.append("kv_runtime_restoration_force_restore_all must be True")
        if int(additional_args.kv_runtime_restoration_recent_exact_window or 0) != 0:
            direct_insertion_reasons.append("kv_runtime_restoration_recent_exact_window must be 0")
        if additional_args.static_exit_layer is not None:
            direct_insertion_reasons.append("static_exit_layer must be unset")
        if additional_args.kv_early_exit_exact_cache_calibration_enabled:
            direct_insertion_reasons.append(
                "kv_early_exit_exact_cache_calibration_enabled must be False (Task C2 direct "
                "insertion and exact-cache calibration collection are mutually exclusive)"
            )
        if direct_insertion_reasons:
            raise NotImplementedError(
                "{} requires the exact supported Native FREE "
                "fixed-source-layer-6 configuration: {}".format(
                    task_c2_mode_label, "; ".join(direct_insertion_reasons)
                )
            )
    if additional_args.kv_pure_recovery_cost_enabled and additional_args.use_early_exit:
        # Official CALM three-arm PURE_MISSING_KV_RECOVERY_COMPUTE_COST
        # (protocol official_free_calm_first_crossing): the LIVE trajectory
        # must be exactly the accepted Official FREE CALM + Exact
        # (exact_catchup) reference configuration -- State and Phase-3c are
        # measured as shadows over the same real first-crossing events. The
        # candidate range is derived at runtime from exit_min_layer ..
        # num_layers-1, never a hard-coded tuple. The Phase-3c shadow is
        # bound by its OWN measurement-only artifact fields; the live
        # exact_catchup runtime fields are never overloaded.
        calm_pure_recovery_reasons = []
        if additional_args.use_shallow_deep:
            calm_pure_recovery_reasons.append("use_shallow_deep must be False")
        if additional_args.use_adapt_threshold:
            calm_pure_recovery_reasons.append("use_adapt_threshold must be False")
        if additional_args.copy_skipped_hidden_states:
            calm_pure_recovery_reasons.append("copy_skipped_hidden_states must be False")
        if additional_args.static_exit_layer is not None:
            calm_pure_recovery_reasons.append("static_exit_layer must be unset")
        if not additional_args.kv_runtime_restoration_enabled:
            calm_pure_recovery_reasons.append("kv_runtime_restoration_enabled must be True")
        if not additional_args.kv_runtime_restoration_calm_enabled:
            calm_pure_recovery_reasons.append("kv_runtime_restoration_calm_enabled must be True")
        if not additional_args.kv_runtime_restoration_official_free_calm_enabled:
            calm_pure_recovery_reasons.append(
                "kv_runtime_restoration_official_free_calm_enabled must be True"
            )
        if additional_args.kv_runtime_restoration_method != "exact_catchup":
            calm_pure_recovery_reasons.append(
                "kv_runtime_restoration_method must be exact_catchup (the live reference "
                "trajectory is Official CALM + Exact; State and Phase-3c are shadow-only)"
            )
        if not additional_args.kv_runtime_restoration_force_restore_all:
            calm_pure_recovery_reasons.append("kv_runtime_restoration_force_restore_all must be True")
        if int(additional_args.kv_runtime_restoration_recent_exact_window or 0) != 0:
            calm_pure_recovery_reasons.append("kv_runtime_restoration_recent_exact_window must be 0")
        if additional_args.kv_runtime_restoration_artifact:
            calm_pure_recovery_reasons.append(
                "kv_runtime_restoration_artifact must be unset (exact_catchup is artifact-free; "
                "the measurement artifact uses kv_pure_recovery_cost_phase3c_artifact)"
            )
        if additional_args.exit_min_layer is None:
            calm_pure_recovery_reasons.append("exit_min_layer must be set")
        if additional_args.exit_conf_type != "softmax":
            calm_pure_recovery_reasons.append("exit_conf_type must be softmax")
        if additional_args.exit_conf_threshold is None:
            calm_pure_recovery_reasons.append("exit_conf_threshold must be set")
        elif abs(float(additional_args.exit_conf_threshold) - 0.9) > 1e-12:
            calm_pure_recovery_reasons.append("exit_conf_threshold must be 0.9")
        if additional_args.kv_runtime_restoration_threshold is None:
            calm_pure_recovery_reasons.append("kv_runtime_restoration_threshold must be set")
        elif additional_args.exit_conf_threshold is not None and abs(
            float(additional_args.kv_runtime_restoration_threshold)
            - float(additional_args.exit_conf_threshold)
        ) > 1e-12:
            calm_pure_recovery_reasons.append(
                "kv_runtime_restoration_threshold must equal exit_conf_threshold"
            )
        if (additional_args.kv_restore_policy or "none") != "none":
            calm_pure_recovery_reasons.append(
                "kv_restore_policy must be 'none' (the reference trajectory must be ordinary "
                "Official CALM + Exact)"
            )
        if additional_args.kv_pure_recovery_cost_validation_enabled:
            calm_pure_recovery_reasons.append(
                "kv_pure_recovery_cost_validation_enabled must be False (parity validation is "
                "Native-FREE-only; the CALM Exact arm IS the live replay and the State/Phase-3c "
                "arms are approximations by design)"
            )
        if not additional_args.missing_kv_provenance_enabled:
            calm_pure_recovery_reasons.append(
                "missing_kv_provenance_enabled must be True (every paper-facing three-arm event "
                "must carry stable sample identity; identity-less events fail closed)"
            )
        if not additional_args.kv_pure_recovery_cost_phase3c_artifact:
            calm_pure_recovery_reasons.append(
                "kv_pure_recovery_cost_phase3c_artifact must be set (consumed only by the "
                "measurement's shadow Phase-3c restorer)"
            )
        calm_pure_recovery_sha = str(
            additional_args.kv_pure_recovery_cost_phase3c_artifact_sha256 or ""
        ).strip()
        if len(calm_pure_recovery_sha) != 64 or any(
            c not in "0123456789abcdef" for c in calm_pure_recovery_sha
        ):
            calm_pure_recovery_reasons.append(
                "kv_pure_recovery_cost_phase3c_artifact_sha256 must be the approved lowercase "
                "64-hex artifact SHA (verified against the file before the shadow restorer "
                "loads it)"
            )
        if calm_pure_recovery_reasons:
            raise NotImplementedError(
                "kv_pure_recovery_cost_enabled on the Official CALM protocol requires the exact "
                "Official FREE CALM + Exact reference configuration: {}".format(
                    "; ".join(calm_pure_recovery_reasons)
                )
            )
    elif additional_args.kv_pure_recovery_cost_enabled:
        # Pure missing-K/V recovery cost is a SHADOW measurement over the
        # ordinary FREE Exact reference trajectory: fail closed to exactly
        # that paper configuration. Runtime restoration must be DISABLED
        # (the live trajectory stays FREE Exact; the artifact is consumed
        # only by the measurement's shadow restorer).
        pure_recovery_reasons = []
        if not additional_args.use_shallow_deep:
            pure_recovery_reasons.append("use_shallow_deep must be True")
        if additional_args.use_early_exit:
            pure_recovery_reasons.append("use_early_exit must be False")
        # Approved fixed-source protocols only (mirrors the fixed_layer
        # calibration route's dataset-scoped source authority): the official
        # LongT5 Multi-News FREE checkpoint uses shallow/source layer 3;
        # every other accepted Native FREE protocol stays frozen at the
        # SAMSum source layer 6. Never inferred, never retuned.
        if kwargs.get('dataset_name') == "multi_news":
            if additional_args.shallow_exit_layer != 3:
                pure_recovery_reasons.append(
                    "shallow_exit_layer must be 3 for the official multi_news LongT5 protocol"
                )
        elif additional_args.shallow_exit_layer != 6:
            pure_recovery_reasons.append("shallow_exit_layer must be 6")
        if additional_args.shallow2deep_conf_threshold != 0.9:
            pure_recovery_reasons.append("shallow2deep_conf_threshold must be 0.9")
        if additional_args.use_adapt_threshold:
            pure_recovery_reasons.append("use_adapt_threshold must be False")
        if not additional_args.parallel_gen_token:
            pure_recovery_reasons.append("parallel_gen_token must be True")
        if not additional_args.parallel_causal_mask:
            pure_recovery_reasons.append("parallel_causal_mask must be True")
        if additional_args.copy_skipped_hidden_states:
            pure_recovery_reasons.append("copy_skipped_hidden_states must be False")
        if additional_args.static_exit_layer is not None:
            pure_recovery_reasons.append("static_exit_layer must be unset")
        if additional_args.kv_runtime_restoration_enabled:
            pure_recovery_reasons.append(
                "kv_runtime_restoration_enabled must be False (the live reference "
                "trajectory must remain FREE Exact; Ours is shadow-only)"
            )
        if not additional_args.kv_runtime_restoration_artifact:
            pure_recovery_reasons.append(
                "kv_runtime_restoration_artifact must be set (consumed only by the "
                "shadow Phase-3c restorer)"
            )
        pure_recovery_sha = str(additional_args.kv_runtime_restoration_artifact_sha256 or "").strip()
        if len(pure_recovery_sha) != 64 or any(
            c not in "0123456789abcdef" for c in pure_recovery_sha
        ):
            pure_recovery_reasons.append(
                "kv_runtime_restoration_artifact_sha256 must be the approved lowercase "
                "64-hex artifact SHA (verified against the file before the shadow "
                "restorer loads it)"
            )
        if (additional_args.kv_restore_policy or "none") != "none":
            pure_recovery_reasons.append(
                "kv_restore_policy must be 'none' (the reference trajectory must be "
                "ordinary Native FREE Exact)"
            )
        if additional_args.kv_runtime_restoration_method != "phase3c_kv_final":
            pure_recovery_reasons.append("kv_runtime_restoration_method must be phase3c_kv_final")
        if additional_args.kv_runtime_restoration_threshold is None:
            pure_recovery_reasons.append("kv_runtime_restoration_threshold must be set")
        elif abs(float(additional_args.kv_runtime_restoration_threshold) - 0.9) > 1e-12:
            pure_recovery_reasons.append("kv_runtime_restoration_threshold must be 0.9")
        if pure_recovery_reasons:
            raise NotImplementedError(
                "kv_pure_recovery_cost_enabled requires the exact Native FREE Exact "
                "paper configuration: {}".format("; ".join(pure_recovery_reasons))
            )
    if additional_args.kv_f2a_frozen_schedule_enabled:
        f2a_reasons = []
        if additional_args.kv_f2a_source_layer_mode not in {"fixed_layer", "candidate_first_crossing"}:
            f2a_reasons.append("kv_f2a_source_layer_mode must be fixed_layer or candidate_first_crossing")
        if not additional_args.kv_f2a_policy_artifact:
            f2a_reasons.append("kv_f2a_policy_artifact must be set")
        for field_name in (
            "kv_f2a_schedule_output",
            "kv_f2a_records_output",
            "kv_f2a_summary_output",
            "kv_f2a_reference_trajectory_output",
            "kv_f2a_reference_trajectory_summary_output",
        ):
            if not getattr(additional_args, field_name, None):
                f2a_reasons.append("{} must be set".format(field_name))
        if additional_args.kv_f2a_max_events is not None and int(additional_args.kv_f2a_max_events) < 0:
            f2a_reasons.append("kv_f2a_max_events must be non-negative")
        if additional_args.kv_f2a_max_debug_logit_records is not None and int(additional_args.kv_f2a_max_debug_logit_records) < 0:
            f2a_reasons.append("kv_f2a_max_debug_logit_records must be non-negative")
        methods = [
            item.strip()
            for item in str(additional_args.kv_f2a_methods or "").split(",")
            if item.strip()
        ]
        required_methods = {
            "exit_hidden_target_projection",
            "exit_conditioned_hidden_restoration",
            "final_kv_restoration",
        }
        if set(methods) != required_methods or len(methods) != len(required_methods):
            f2a_reasons.append("kv_f2a_methods must be the three frozen F2a methods")
        if additional_args.kv_f2a_write_debug_logits and int(additional_args.kv_f2a_max_debug_logit_records or 0) <= 0:
            f2a_reasons.append("kv_f2a_max_debug_logit_records must be >0 when debug logits are enabled")
        if f2a_reasons:
            raise NotImplementedError("F2a frozen-schedule replay requires explicit paper-safe plumbing: {}".format("; ".join(f2a_reasons)))
    # Validation-only route marker (never used to compute/assign a source
    # layer -- only to pick which fixed-layer contract applies below).
    # run_summarization.py passes data_args.dataset_name through kwargs;
    # callers that never pass it (most existing tests) get None, which
    # always resolves to the legacy/CNN source6 requirement, unchanged.
    dataset_name = kwargs.get('dataset_name')
    # Computed once, before the calibration block below, so the fixed_layer
    # branch's legacy-SAMSum-manifest requirement (which must NOT apply in
    # CNN contract mode) and the dedicated CNN validation block further down
    # both consult the exact same predicate rather than two subtly
    # different conditions.
    cnn_population_contract_options_supplied = bool(
        additional_args.kv_early_exit_exact_cache_calibration_population_contract
        or additional_args.kv_early_exit_exact_cache_calibration_population_contract_sha256
        or additional_args.kv_early_exit_exact_cache_calibration_dataset_split
    )
    if additional_args.kv_early_exit_exact_cache_calibration_enabled:
        calibration_reasons = []
        calibration_source_mode = additional_args.kv_early_exit_exact_cache_calibration_source_layer_mode
        if calibration_source_mode not in (
            "candidate_first_crossing",
            "fixed_layer",
            "official_free_calm_first_crossing",
        ):
            calibration_reasons.append(
                "kv_early_exit_exact_cache_calibration_source_layer_mode must be "
                "candidate_first_crossing, fixed_layer, or official_free_calm_first_crossing"
            )
        elif calibration_source_mode == "fixed_layer":
            # The official synchronized FREE shallow-deep path is not the CALM
            # candidate_first_crossing exact_catchup runtime restoration method:
            # collection here rides the existing synchronized parallel flush,
            # so runtime restoration must stay off entirely.
            if additional_args.kv_runtime_restoration_enabled:
                calibration_reasons.append("kv_runtime_restoration_enabled must be False for fixed_layer calibration")
            if additional_args.kv_runtime_restoration_calm_enabled:
                calibration_reasons.append("kv_runtime_restoration_calm_enabled must be False for fixed_layer calibration")
            if additional_args.kv_runtime_restoration_artifact:
                calibration_reasons.append("kv_runtime_restoration_artifact must be empty for fixed_layer calibration")
            if additional_args.kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime:
                calibration_reasons.append(
                    "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime must be False "
                    "for fixed_layer calibration"
                )
            if not additional_args.use_shallow_deep:
                calibration_reasons.append("use_shallow_deep must be True for fixed_layer calibration")
            if additional_args.use_early_exit:
                calibration_reasons.append("use_early_exit must be False for fixed_layer calibration")
            if dataset_name == "multi_news":
                # Official LongT5 Multi-News FREE route: the approved source
                # layer is whatever the externally approved Multi-News
                # population contract and the actual runtime
                # config.shallow_exit_layer agree on (currently 3 for the
                # approved official checkpoint protocol, never inferred from
                # dataset_name itself) -- a consistency check, not
                # source-layer selection. The contract independently binds
                # and re-verifies the actual approved value (see
                # ExactCacheCalibrationCollector._from_multinews_population_
                # contract).
                multinews_fixed_source_layer = additional_args.kv_early_exit_exact_cache_calibration_fixed_source_layer
                if isinstance(multinews_fixed_source_layer, bool) or not isinstance(
                    multinews_fixed_source_layer, int
                ):
                    calibration_reasons.append(
                        "kv_early_exit_exact_cache_calibration_fixed_source_layer must be an explicit "
                        "integer for multi_news fixed_layer calibration"
                    )
                elif additional_args.shallow_exit_layer != multinews_fixed_source_layer:
                    calibration_reasons.append(
                        "shallow_exit_layer must equal kv_early_exit_exact_cache_calibration_fixed_source_layer "
                        "for multi_news fixed_layer calibration"
                    )
            else:
                if int(additional_args.kv_early_exit_exact_cache_calibration_fixed_source_layer or 0) != 6:
                    calibration_reasons.append("kv_early_exit_exact_cache_calibration_fixed_source_layer must be 6")
                if additional_args.shallow_exit_layer != 6:
                    calibration_reasons.append("shallow_exit_layer must be 6 for fixed_layer calibration")
            if additional_args.use_adapt_threshold:
                calibration_reasons.append("use_adapt_threshold must be False for fixed_layer calibration")
            if not additional_args.parallel_gen_token:
                calibration_reasons.append("parallel_gen_token must be True for fixed_layer calibration")
            if additional_args.copy_skipped_hidden_states:
                calibration_reasons.append("copy_skipped_hidden_states must be False for fixed_layer calibration")
            if additional_args.shallow2deep_conf_type != "softmax":
                calibration_reasons.append("shallow2deep_conf_type must be softmax for fixed_layer calibration")
            if additional_args.shallow2deep_conf_threshold != 0.9:
                calibration_reasons.append("shallow2deep_conf_threshold must be 0.9 for fixed_layer calibration")
            if not additional_args.parallel_causal_mask:
                calibration_reasons.append("parallel_causal_mask must be True for fixed_layer calibration")
            if additional_args.static_exit_layer is not None:
                calibration_reasons.append("static_exit_layer must be unset for fixed_layer calibration")
            if additional_args.rollback_conf_threshold is not None:
                calibration_reasons.append("rollback_conf_threshold must be unset for fixed_layer calibration")
            if int(additional_args.smoke_force_flush_after_skips or 0) != 0:
                calibration_reasons.append("smoke_force_flush_after_skips must be disabled for fixed_layer calibration")
        elif calibration_source_mode == "official_free_calm_first_crossing":
            # Official FREE CALM-style production early-exit calibration
            # collection: same artifact-free exact_catchup shape as the
            # historical candidate_first_crossing branch below, but must be
            # explicitly routed through the official path (never silently
            # reuses the historical (4,6,8,10)/0.9 policy).
            if not additional_args.kv_runtime_restoration_enabled:
                calibration_reasons.append("kv_runtime_restoration_enabled must be True")
            if not additional_args.kv_runtime_restoration_calm_enabled:
                calibration_reasons.append("kv_runtime_restoration_calm_enabled must be True")
            if not additional_args.kv_runtime_restoration_official_free_calm_enabled:
                calibration_reasons.append(
                    "kv_runtime_restoration_official_free_calm_enabled must be True for "
                    "official_free_calm_first_crossing calibration"
                )
            if additional_args.kv_runtime_restoration_method != "exact_catchup":
                calibration_reasons.append("kv_runtime_restoration_method must be exact_catchup")
            if additional_args.kv_runtime_restoration_artifact:
                calibration_reasons.append("exact-cache calibration is artifact-free")
            if additional_args.exit_conf_threshold is None:
                calibration_reasons.append("exit_conf_threshold must be set for official_free_calm_first_crossing calibration")
            if additional_args.exit_min_layer is None:
                calibration_reasons.append("exit_min_layer must be set for official_free_calm_first_crossing calibration")
            # Same runtime-equivalence requirement as the official FREE CALM
            # Task C1 runtime block above: the collector's confidence
            # computation is only equivalent to the original FREE
            # get_skip_mask() decision under exit_conf_type=="softmax" and
            # exit_position_temp=None.
            if additional_args.exit_conf_type != "softmax":
                calibration_reasons.append("exit_conf_type must be softmax for official_free_calm_first_crossing calibration")
            if additional_args.exit_position_temp is not None:
                calibration_reasons.append("exit_position_temp must be unset for official_free_calm_first_crossing calibration")
        else:
            # candidate_first_crossing: preserved exactly as before.
            if not additional_args.kv_runtime_restoration_enabled:
                calibration_reasons.append("kv_runtime_restoration_enabled must be True")
            if not additional_args.kv_runtime_restoration_calm_enabled:
                calibration_reasons.append("kv_runtime_restoration_calm_enabled must be True")
            if additional_args.kv_runtime_restoration_method != "exact_catchup":
                calibration_reasons.append("kv_runtime_restoration_method must be exact_catchup")
            if additional_args.kv_runtime_restoration_artifact:
                calibration_reasons.append("exact-cache calibration is artifact-free")
        if not additional_args.missing_kv_provenance_enabled:
            calibration_reasons.append("missing_kv_provenance_enabled must be True")
        if not additional_args.missing_kv_selected_stable_sample_ids_file:
            calibration_reasons.append("missing_kv_selected_stable_sample_ids_file must bind fitting selection")
        if not additional_args.kv_early_exit_exact_cache_calibration_output_dir:
            calibration_reasons.append("kv_early_exit_exact_cache_calibration_output_dir must be set")
        if not cnn_population_contract_options_supplied and calibration_source_mode != "official_free_calm_first_crossing":
            # Legacy SAMSum population binding only -- the CNN/DailyMail
            # train-calibration contract branch (validated separately,
            # below) binds its population via the population contract
            # instead, and the official FREE CALM branch binds its
            # population purely via missing_kv_selected_stable_sample_ids_
            # file (see ExactCacheCalibrationCollector._from_official_free_
            # calm_config) -- neither must ever be blocked on these two
            # SAMSum-only manifests.
            if not additional_args.kv_early_exit_exact_cache_calibration_fitting_source_manifest:
                calibration_reasons.append("fitting source manifest must be set")
            if not additional_args.kv_early_exit_exact_cache_calibration_approved_corrective_manifest:
                calibration_reasons.append("approved corrective manifest must be set")
        if int(additional_args.kv_early_exit_exact_cache_calibration_max_events_per_flush or 0) <= 0:
            calibration_reasons.append("max events per flush must be positive")
        if calibration_reasons:
            raise ValueError("exact-cache calibration configuration invalid: {}".format("; ".join(calibration_reasons)))
    official_free_calm_train_route_requested = bool(
        additional_args.kv_early_exit_exact_cache_calibration_dataset_split
        and additional_args.kv_early_exit_exact_cache_calibration_source_layer_mode
        == "official_free_calm_first_crossing"
    )
    if official_free_calm_train_route_requested:
        # Official FREE CALM CNN/DailyMail train-calibration opt-in: the
        # authoritative fitting population is
        # missing_kv_selected_stable_sample_ids_file itself (see
        # run_summarization.py's official pre-construction gate) -- this
        # route never depends on, and must be kept separate from, the Native
        # FREE fixed-source-6 population contract validated in the elif
        # branch below. dataset_split truthiness alone used to route this
        # combination into that Native branch (which then always failed
        # closed on source_layer_mode != fixed_layer); this dedicated branch
        # is the fix.
        official_reasons = []
        if not additional_args.kv_early_exit_exact_cache_calibration_enabled:
            official_reasons.append("kv_early_exit_exact_cache_calibration_enabled must be True")
        if official_reasons:
            raise NotImplementedError(
                "Official FREE CALM CNN/DailyMail train-calibration route requires the exact supported "
                "configuration: {}".format("; ".join(official_reasons))
            )
    elif cnn_population_contract_options_supplied:
        # Narrow CNN/DailyMail train-calibration opt-in. When none of these
        # three options are supplied, this block never runs and existing
        # SAMSum behavior (including the fixed_layer branch validated above)
        # is completely unaffected.
        cnn_reasons = []
        if not additional_args.kv_early_exit_exact_cache_calibration_enabled:
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_enabled must be True")
        if additional_args.kv_early_exit_exact_cache_calibration_source_layer_mode != "fixed_layer":
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_source_layer_mode must be fixed_layer")
        if dataset_name == "multi_news":
            # The exact fixed_source_layer == shallow_exit_layer match is
            # already required above (whenever calibration is enabled, which
            # this branch also independently requires below); this route
            # only needs an explicit integer here, never a literal 6 -- the
            # Multi-News population contract binds the actual approved value.
            multinews_fixed_source_layer = additional_args.kv_early_exit_exact_cache_calibration_fixed_source_layer
            if isinstance(multinews_fixed_source_layer, bool) or not isinstance(multinews_fixed_source_layer, int):
                cnn_reasons.append(
                    "kv_early_exit_exact_cache_calibration_fixed_source_layer must be an explicit integer "
                    "for multi_news fixed_layer calibration"
                )
        elif int(additional_args.kv_early_exit_exact_cache_calibration_fixed_source_layer or 0) != 6:
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_fixed_source_layer must be 6")
        if additional_args.kv_early_exit_exact_cache_calibration_dataset_split != "train":
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_dataset_split must be train")
        if not additional_args.missing_kv_provenance_enabled:
            cnn_reasons.append("missing_kv_provenance_enabled must be True")
        if not additional_args.kv_early_exit_exact_cache_calibration_population_contract:
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_population_contract must be set")
        if not additional_args.kv_early_exit_exact_cache_calibration_population_contract_sha256:
            cnn_reasons.append("kv_early_exit_exact_cache_calibration_population_contract_sha256 must be set")
        if cnn_reasons:
            raise NotImplementedError(
                "CNN/DailyMail train-calibration population contract requires the exact supported "
                "configuration: {}".format("; ".join(cnn_reasons))
            )
    if additional_args.kv_exact_catchup_require_complete_events and not additional_args.kv_exact_catchup_overhead_enabled:
        raise ValueError("kv_exact_catchup_require_complete_events requires kv_exact_catchup_overhead_enabled=True")
    if additional_args.kv_exact_catchup_dump_max_flushes is not None:
        assert additional_args.kv_exact_catchup_dump_max_flushes >= 0
    if additional_args.kv_exact_catchup_dump_max_tokens_per_flush is not None:
        assert additional_args.kv_exact_catchup_dump_max_tokens_per_flush >= 0
    if additional_args.kv_exact_catchup_dump_max_layers is not None:
        assert additional_args.kv_exact_catchup_dump_max_layers >= 0
    if additional_args.kv_exact_catchup_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_exact_catchup_dump_dtype: {}".format(additional_args.kv_exact_catchup_dump_dtype))
    if additional_args.kv_source_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_source_dump_dtype: {}".format(additional_args.kv_source_dump_dtype))
    if additional_args.kv_source_dump_max_tokens is not None:
        assert additional_args.kv_source_dump_max_tokens >= 0
    if additional_args.kv_adjacent_anchor_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_adjacent_anchor_dump_dtype: {}".format(additional_args.kv_adjacent_anchor_dump_dtype))
    if additional_args.kv_adjacent_anchor_dump_max_tokens is not None:
        assert additional_args.kv_adjacent_anchor_dump_max_tokens >= 0
    if additional_args.kv_all_layer_calib_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_all_layer_calib_dump_dtype: {}".format(additional_args.kv_all_layer_calib_dump_dtype))
    if additional_args.kv_all_layer_calib_dump_max_tokens is not None:
        assert additional_args.kv_all_layer_calib_dump_max_tokens >= 0
    if additional_args.kv_all_layer_hidden_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_all_layer_hidden_dump_dtype: {}".format(additional_args.kv_all_layer_hidden_dump_dtype))
    if additional_args.kv_all_layer_hidden_dump_max_tokens is not None:
        assert additional_args.kv_all_layer_hidden_dump_max_tokens >= 0
    if additional_args.kv_all_layer_hidden_dump_max_flushes is not None:
        assert additional_args.kv_all_layer_hidden_dump_max_flushes >= 0
    if additional_args.kv_all_layer_hidden_dump_enabled:
        if not additional_args.kv_all_layer_hidden_dump_include_raw_hidden and not additional_args.kv_all_layer_hidden_dump_include_normed_hidden:
            raise ValueError(
                "kv_all_layer_hidden_dump_enabled requires at least one of "
                "kv_all_layer_hidden_dump_include_raw_hidden or "
                "kv_all_layer_hidden_dump_include_normed_hidden"
            )
    if additional_args.missing_kv_dump_storage_format not in ["legacy_row_v1", "packed_generation_v1"]:
        raise ValueError(
            "Unsupported missing_kv_dump_storage_format: {}".format(
                additional_args.missing_kv_dump_storage_format
            )
        )
    if additional_args.kv_calm_counterfactual_trace_enabled:
        if not additional_args.missing_kv_provenance_enabled:
            raise ValueError("kv_calm_counterfactual_trace_enabled requires missing_kv_provenance_enabled")
        if additional_args.missing_kv_dump_storage_format != "packed_generation_v1":
            raise ValueError("kv_calm_counterfactual_trace_enabled requires packed_generation_v1 storage")
        if not additional_args.kv_all_layer_hidden_dump_enabled:
            raise ValueError("kv_calm_counterfactual_trace_enabled requires all-layer hidden dumping")
        if not additional_args.kv_all_layer_hidden_dump_include_raw_hidden:
            raise ValueError("kv_calm_counterfactual_trace_enabled requires raw hidden dumping")
        if not additional_args.kv_all_layer_calib_dump_enabled:
            raise ValueError("kv_calm_counterfactual_trace_enabled requires all-layer K/V dumping")
        if not additional_args.kv_calm_counterfactual_trace_output:
            raise ValueError("kv_calm_counterfactual_trace_output is required when CALM trace is enabled")
        if additional_args.use_shallow_deep or additional_args.use_early_exit:
            raise ValueError("CALM counterfactual trace requires full-depth execution, not FREE early-exit execution")
        if additional_args.static_exit_layer is not None:
            raise ValueError("CALM counterfactual trace requires static_exit_layer unset")
        if additional_args.kv_runtime_restoration_enabled:
            raise ValueError("CALM counterfactual trace requires runtime restoration disabled")
        if additional_args.kv_runtime_restoration_calm_enabled:
            raise ValueError("CALM counterfactual trace requires kv_runtime_restoration_calm_enabled disabled")
        if additional_args.use_adapt_threshold:
            raise ValueError("CALM counterfactual trace requires use_adapt_threshold disabled")
        try:
            candidate_layers = [
                int(part.strip())
                for part in str(additional_args.kv_calm_counterfactual_candidate_layers).split(",")
                if part.strip()
            ]
        except Exception as exc:
            raise ValueError("Invalid kv_calm_counterfactual_candidate_layers") from exc
        if candidate_layers != sorted(candidate_layers) or len(set(candidate_layers)) != len(candidate_layers):
            raise ValueError("kv_calm_counterfactual_candidate_layers must be sorted unique integers")
        if not candidate_layers or any(layer < 0 for layer in candidate_layers):
            raise ValueError("kv_calm_counterfactual_candidate_layers must be non-empty non-negative integers")
        if float(additional_args.kv_calm_counterfactual_threshold) != 0.9:
            raise ValueError("kv_calm_counterfactual_threshold is frozen at 0.9 for this trace policy")
    if additional_args.kv_attention_diag_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_attention_diag_dump_dtype: {}".format(additional_args.kv_attention_diag_dump_dtype))
    if additional_args.kv_attention_diag_dump_max_records is not None:
        assert additional_args.kv_attention_diag_dump_max_records >= 0
    if additional_args.kv_full_attention_diag_dump_dtype not in ["original", "float16", "bfloat16", "float32"]:
        raise ValueError("Unsupported kv_full_attention_diag_dump_dtype: {}".format(additional_args.kv_full_attention_diag_dump_dtype))
    if additional_args.kv_full_attention_diag_dump_max_records is not None:
        assert additional_args.kv_full_attention_diag_dump_max_records >= 0
    if additional_args.kv_full_attention_diag_dump_max_key_tokens is not None:
        assert additional_args.kv_full_attention_diag_dump_max_key_tokens >= 0
    if additional_args.kv_restoration_dryrun_max_records is not None:
        assert additional_args.kv_restoration_dryrun_max_records >= 0
    if additional_args.kv_restore_policy in ["h2o_topk_dryrun", "h2o_mask_noop", "h2o_exact_recompute_dryrun", "h2o_exact_recompute_verify", "h2o_mask"] and not additional_args.kv_importance_enabled:
        warnings.warn(
            "kv_restore_policy='{}' is most useful with kv_importance_enabled=True; ".format(additional_args.kv_restore_policy) +
            "dry-run candidate logs will not include scores without importance tracking.",
            UserWarning,
        )
    if additional_args.kv_restore_dump_eaes_scores and not additional_args.kv_importance_enabled:
        warnings.warn(
            "kv_restore_dump_eaes_scores=True requires kv_importance_enabled=True to observe early-layer "
            "attention evidence; missing EAES scores will be reported instead of fabricated.",
            UserWarning,
        )

    deploy_config = {
        'use_synchronize': additional_args.use_synchronize,
        'kv_trace_enabled': additional_args.kv_trace_enabled,
        'kv_trace_path': additional_args.kv_trace_path,
        'kv_trace_max_records': additional_args.kv_trace_max_records,
        'kv_trace_flush_on_generate_end': additional_args.kv_trace_flush_on_generate_end,
        'kv_importance_enabled': additional_args.kv_importance_enabled,
        'kv_importance_mode': additional_args.kv_importance_mode,
        'kv_importance_decay': additional_args.kv_importance_decay,
        'kv_importance_include_current': additional_args.kv_importance_include_current,
        'kv_importance_dump_path': additional_args.kv_importance_dump_path,
        'kv_importance_flush_on_generate_end': additional_args.kv_importance_flush_on_generate_end,
        'kv_restore_policy': additional_args.kv_restore_policy,
        'kv_restore_topk': additional_args.kv_restore_topk,
        'kv_restore_importance_layer': additional_args.kv_restore_importance_layer,
        'kv_restore_recent_window': additional_args.kv_restore_recent_window,
        'kv_restore_log_candidates': additional_args.kv_restore_log_candidates,
        'kv_runtime_restoration_enabled': additional_args.kv_runtime_restoration_enabled,
        'kv_runtime_restoration_calm_enabled': additional_args.kv_runtime_restoration_calm_enabled,
        'kv_runtime_restoration_official_free_calm_enabled': additional_args.kv_runtime_restoration_official_free_calm_enabled,
        'kv_runtime_restoration_artifact': additional_args.kv_runtime_restoration_artifact,
        'kv_runtime_restoration_method': additional_args.kv_runtime_restoration_method,
        'kv_runtime_restoration_threshold': additional_args.kv_runtime_restoration_threshold,
        'kv_runtime_restoration_recent_exact_window': additional_args.kv_runtime_restoration_recent_exact_window,
        'kv_runtime_restoration_max_pending_tokens': additional_args.kv_runtime_restoration_max_pending_tokens,
        'kv_runtime_restoration_debug': additional_args.kv_runtime_restoration_debug,
        'kv_runtime_restoration_force_restore_all': additional_args.kv_runtime_restoration_force_restore_all,
        'kv_runtime_restoration_direct_insertion_enabled': additional_args.kv_runtime_restoration_direct_insertion_enabled,
        'kv_runtime_restoration_batched_insertion_enabled': additional_args.kv_runtime_restoration_batched_insertion_enabled,
        'kv_runtime_restoration_finite_validation_enabled': additional_args.kv_runtime_restoration_finite_validation_enabled,
        'kv_pure_recovery_cost_enabled': additional_args.kv_pure_recovery_cost_enabled,
        'kv_pure_recovery_cost_output': additional_args.kv_pure_recovery_cost_output,
        'kv_pure_recovery_cost_validation_enabled': additional_args.kv_pure_recovery_cost_validation_enabled,
        'kv_pure_recovery_cost_phase3c_artifact': additional_args.kv_pure_recovery_cost_phase3c_artifact,
        'kv_pure_recovery_cost_phase3c_artifact_sha256': additional_args.kv_pure_recovery_cost_phase3c_artifact_sha256,
        'kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime': additional_args.kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime,
        'kv_runtime_restoration_artifact_sha256': additional_args.kv_runtime_restoration_artifact_sha256,
        'kv_early_exit_exact_cache_calibration_enabled': additional_args.kv_early_exit_exact_cache_calibration_enabled,
        'kv_early_exit_exact_cache_calibration_output_dir': additional_args.kv_early_exit_exact_cache_calibration_output_dir,
        'kv_early_exit_exact_cache_calibration_fitting_source_manifest': additional_args.kv_early_exit_exact_cache_calibration_fitting_source_manifest,
        'kv_early_exit_exact_cache_calibration_approved_corrective_manifest': additional_args.kv_early_exit_exact_cache_calibration_approved_corrective_manifest,
        'kv_early_exit_exact_cache_calibration_max_events_per_flush': additional_args.kv_early_exit_exact_cache_calibration_max_events_per_flush,
        'kv_early_exit_exact_cache_calibration_source_layer_mode': additional_args.kv_early_exit_exact_cache_calibration_source_layer_mode,
        'kv_early_exit_exact_cache_calibration_fixed_source_layer': additional_args.kv_early_exit_exact_cache_calibration_fixed_source_layer,
        'kv_early_exit_exact_cache_calibration_population_contract': additional_args.kv_early_exit_exact_cache_calibration_population_contract,
        'kv_early_exit_exact_cache_calibration_population_contract_sha256': additional_args.kv_early_exit_exact_cache_calibration_population_contract_sha256,
        'kv_early_exit_exact_cache_calibration_dataset_split': additional_args.kv_early_exit_exact_cache_calibration_dataset_split,
        'kv_runtime_component_timing_enabled': additional_args.kv_runtime_component_timing_enabled,
        'kv_runtime_component_timing_backend': additional_args.kv_runtime_component_timing_backend,
        'kv_runtime_accounting_output': additional_args.kv_runtime_accounting_output,
        'missing_kv_per_sample_accounting_output': additional_args.missing_kv_per_sample_accounting_output,
        'kv_runtime_component_timing_output': additional_args.kv_runtime_component_timing_output,
        'kv_exact_catchup_overhead_enabled': additional_args.kv_exact_catchup_overhead_enabled,
        'kv_exact_catchup_event_output': additional_args.kv_exact_catchup_event_output,
        'kv_exact_catchup_summary_output': additional_args.kv_exact_catchup_summary_output,
        'kv_exact_catchup_csv_dir': additional_args.kv_exact_catchup_csv_dir,
        'kv_generation_timing_enabled': additional_args.kv_generation_timing_enabled,
        'kv_generation_timing_output': additional_args.kv_generation_timing_output,
        'kv_exact_catchup_require_complete_events': additional_args.kv_exact_catchup_require_complete_events,
        'kv_f2a_frozen_schedule_enabled': additional_args.kv_f2a_frozen_schedule_enabled,
        'kv_f2a_source_layer_mode': additional_args.kv_f2a_source_layer_mode,
        'kv_f2a_policy_artifact': additional_args.kv_f2a_policy_artifact,
        'kv_f2a_schedule_output': additional_args.kv_f2a_schedule_output,
        'kv_f2a_records_output': additional_args.kv_f2a_records_output,
        'kv_f2a_summary_output': additional_args.kv_f2a_summary_output,
        'kv_f2a_reference_trajectory_output': additional_args.kv_f2a_reference_trajectory_output,
        'kv_f2a_reference_trajectory_summary_output': additional_args.kv_f2a_reference_trajectory_summary_output,
        'kv_f2a_methods': additional_args.kv_f2a_methods,
        'kv_f2a_max_events': additional_args.kv_f2a_max_events,
        'kv_f2a_write_debug_logits': additional_args.kv_f2a_write_debug_logits,
        'kv_f2a_max_debug_logit_records': additional_args.kv_f2a_max_debug_logit_records,
        'kv_f2a_decoding_configuration_sha256': additional_args.kv_f2a_decoding_configuration_sha256,
        'save_eval_predictions': additional_args.save_eval_predictions,
        'eval_predictions_output': additional_args.eval_predictions_output,
        'missing_kv_provenance_enabled': additional_args.missing_kv_provenance_enabled,
        'missing_kv_effective_population_output': additional_args.missing_kv_effective_population_output,
        'missing_kv_effective_population_summary_output': additional_args.missing_kv_effective_population_summary_output,
        'missing_kv_selected_stable_sample_ids_file': additional_args.missing_kv_selected_stable_sample_ids_file,
        'missing_kv_f2b_population_identity_file': additional_args.missing_kv_f2b_population_identity_file,
        'missing_kv_generation_binding_output': additional_args.missing_kv_generation_binding_output,
        'missing_kv_generation_binding_summary_output': additional_args.missing_kv_generation_binding_summary_output,
        'missing_kv_tokenizer_identity_output': additional_args.missing_kv_tokenizer_identity_output,
        'missing_kv_checkpoint_inventory_output': additional_args.missing_kv_checkpoint_inventory_output,
        'missing_kv_decoding_configuration_identity_output': additional_args.missing_kv_decoding_configuration_identity_output,
        'missing_kv_candidate_policy_identity_output': additional_args.missing_kv_candidate_policy_identity_output,
        'missing_kv_runtime_method_identity_output': additional_args.missing_kv_runtime_method_identity_output,
        'missing_kv_dump_storage_format': additional_args.missing_kv_dump_storage_format,
        'kv_calm_counterfactual_trace_enabled': additional_args.kv_calm_counterfactual_trace_enabled,
        'kv_calm_counterfactual_trace_output': additional_args.kv_calm_counterfactual_trace_output,
        'kv_calm_counterfactual_candidate_layers': additional_args.kv_calm_counterfactual_candidate_layers,
        'kv_calm_counterfactual_threshold': additional_args.kv_calm_counterfactual_threshold,
        'kv_restore_dump_eaes_scores': additional_args.kv_restore_dump_eaes_scores,
        'kv_restore_eaes_score_jsonl': additional_args.kv_restore_eaes_score_jsonl,
        'kv_exact_catchup_dump_enabled': additional_args.kv_exact_catchup_dump_enabled,
        'kv_exact_catchup_dump_dir': additional_args.kv_exact_catchup_dump_dir,
        'kv_exact_catchup_dump_max_flushes': additional_args.kv_exact_catchup_dump_max_flushes,
        'kv_exact_catchup_dump_max_tokens_per_flush': additional_args.kv_exact_catchup_dump_max_tokens_per_flush,
        'kv_exact_catchup_dump_max_layers': additional_args.kv_exact_catchup_dump_max_layers,
        'kv_exact_catchup_dump_layers': additional_args.kv_exact_catchup_dump_layers,
        'kv_exact_catchup_dump_dtype': additional_args.kv_exact_catchup_dump_dtype,
        'kv_exact_catchup_dump_cpu': additional_args.kv_exact_catchup_dump_cpu,
        'kv_exact_catchup_dump_include_current_token': additional_args.kv_exact_catchup_dump_include_current_token,
        'kv_source_dump_enabled': additional_args.kv_source_dump_enabled,
        'kv_source_dump_dir': additional_args.kv_source_dump_dir,
        'kv_source_dump_dtype': additional_args.kv_source_dump_dtype,
        'kv_source_dump_cpu': additional_args.kv_source_dump_cpu,
        'kv_source_dump_max_tokens': additional_args.kv_source_dump_max_tokens,
        'kv_adjacent_anchor_dump_enabled': additional_args.kv_adjacent_anchor_dump_enabled,
        'kv_adjacent_anchor_dump_dir': additional_args.kv_adjacent_anchor_dump_dir,
        'kv_adjacent_anchor_dump_dtype': additional_args.kv_adjacent_anchor_dump_dtype,
        'kv_adjacent_anchor_dump_cpu': additional_args.kv_adjacent_anchor_dump_cpu,
        'kv_adjacent_anchor_dump_max_tokens': additional_args.kv_adjacent_anchor_dump_max_tokens,
        'kv_all_layer_calib_dump_enabled': additional_args.kv_all_layer_calib_dump_enabled,
        'kv_all_layer_calib_dump_dir': additional_args.kv_all_layer_calib_dump_dir,
        'kv_all_layer_calib_dump_dtype': additional_args.kv_all_layer_calib_dump_dtype,
        'kv_all_layer_calib_dump_cpu': additional_args.kv_all_layer_calib_dump_cpu,
        'kv_all_layer_calib_dump_max_tokens': additional_args.kv_all_layer_calib_dump_max_tokens,
        'kv_all_layer_calib_dump_layers': additional_args.kv_all_layer_calib_dump_layers,
        'kv_all_layer_hidden_dump_enabled': additional_args.kv_all_layer_hidden_dump_enabled,
        'kv_all_layer_hidden_dump_dir': additional_args.kv_all_layer_hidden_dump_dir,
        'kv_all_layer_hidden_dump_dtype': additional_args.kv_all_layer_hidden_dump_dtype,
        'kv_all_layer_hidden_dump_cpu': additional_args.kv_all_layer_hidden_dump_cpu,
        'kv_all_layer_hidden_dump_max_tokens': additional_args.kv_all_layer_hidden_dump_max_tokens,
        'kv_all_layer_hidden_dump_max_flushes': additional_args.kv_all_layer_hidden_dump_max_flushes,
        'kv_all_layer_hidden_dump_include_raw_hidden': additional_args.kv_all_layer_hidden_dump_include_raw_hidden,
        'kv_all_layer_hidden_dump_include_normed_hidden': additional_args.kv_all_layer_hidden_dump_include_normed_hidden,
        'kv_all_layer_hidden_dump_layers': additional_args.kv_all_layer_hidden_dump_layers,
        'kv_attention_diag_dump_enabled': additional_args.kv_attention_diag_dump_enabled,
        'kv_attention_diag_dump_dir': additional_args.kv_attention_diag_dump_dir,
        'kv_attention_diag_dump_max_records': additional_args.kv_attention_diag_dump_max_records,
        'kv_attention_diag_dump_dtype': additional_args.kv_attention_diag_dump_dtype,
        'kv_attention_diag_dump_cpu': additional_args.kv_attention_diag_dump_cpu,
        'kv_attention_diag_dump_layers': additional_args.kv_attention_diag_dump_layers,
        'kv_full_attention_diag_dump_enabled': additional_args.kv_full_attention_diag_dump_enabled,
        'kv_full_attention_diag_dump_dir': additional_args.kv_full_attention_diag_dump_dir,
        'kv_full_attention_diag_dump_max_records': additional_args.kv_full_attention_diag_dump_max_records,
        'kv_full_attention_diag_dump_dtype': additional_args.kv_full_attention_diag_dump_dtype,
        'kv_full_attention_diag_dump_cpu': additional_args.kv_full_attention_diag_dump_cpu,
        'kv_full_attention_diag_dump_layers': additional_args.kv_full_attention_diag_dump_layers,
        'kv_full_attention_diag_dump_max_key_tokens': additional_args.kv_full_attention_diag_dump_max_key_tokens,
        'kv_full_attention_diag_dump_include_position_bias': additional_args.kv_full_attention_diag_dump_include_position_bias,
        'kv_restoration_dryrun_enabled': additional_args.kv_restoration_dryrun_enabled,
        'kv_restoration_dryrun_manifest_path': additional_args.kv_restoration_dryrun_manifest_path,
        'kv_restoration_dryrun_max_records': additional_args.kv_restoration_dryrun_max_records,
        'smoke_force_flush_after_skips': additional_args.smoke_force_flush_after_skips,
        'smoke_force_flush_once': additional_args.smoke_force_flush_once,
    }
    config.update(deploy_config)
    
    inter_config = {
        'output_hidden_states_decoder': additional_args.output_hidden_states_decoder,
        'intermediate_loss_fn': additional_args.intermediate_loss_fn,
        'distill_layer_alpha': additional_args.distill_layer_alpha,
        'do_layer_transformation': additional_args.do_layer_transformation,
    }
    config.update(inter_config)

    static_config = {
        'static_exit_layer': additional_args.static_exit_layer,
    }
    config.update(static_config)
    
    early_exit_config = {
        'use_early_exit': additional_args.use_early_exit,
        'exit_conf_type': additional_args.exit_conf_type,
        'exit_conf_threshold': additional_args.exit_conf_threshold,
        'exit_position_temp': additional_args.exit_position_temp,
        'exit_min_layer': additional_args.exit_min_layer,
        'train_meta_cm_head': additional_args.train_meta_cm_head,
        'max_answer_length': kwargs.get('max_answer_length', None),
    }
    config.update(early_exit_config)
    
    shallow_deep_config = {
        'use_shallow_deep': additional_args.use_shallow_deep,
        'shallow_exit_layer': additional_args.shallow_exit_layer,
        'shallow2deep_conf_type': additional_args.shallow2deep_conf_type,
        'shallow2deep_conf_threshold': additional_args.shallow2deep_conf_threshold,  
        'parallel_gen_token': additional_args.parallel_gen_token,  
        'copy_skipped_hidden_states': additional_args.copy_skipped_hidden_states,  
        'rollback_conf_threshold': additional_args.rollback_conf_threshold,
        'parallel_causal_mask': additional_args.parallel_causal_mask
    }
    config.update(shallow_deep_config)
    
    adaptive_threshold_config = {
        "use_adapt_threshold": additional_args.use_adapt_threshold,
    }
    config.update(adaptive_threshold_config)
    
    lora_config = {
        "use_lora": additional_args.use_lora,
        "lora_rank": additional_args.lora_rank,
        "lora_alpha": additional_args.lora_alpha,
        "lora_dropout": additional_args.lora_dropout,
        "lora_target_modules": additional_args.lora_target_modules,
    }
    config.update(lora_config)

    return config
