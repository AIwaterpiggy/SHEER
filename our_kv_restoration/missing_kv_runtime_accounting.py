"""Runtime accounting helpers for missing deep-layer K/V experiments.

The counters in this module are behavior-preserving diagnostics.  A token-layer
unit means one decoder token at one decoder layer.  Task C1 computes exact FREE
catch-up before overwriting selected exact K/V slices, so avoided exact catch-up
units must remain zero and no speed claim is valid.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional


COUNTER_KEYS = (
    "generated_token_count",
    "nominal_skipped_token_layer_units",
    "exact_catchup_required_token_layer_units",
    "exact_catchup_executed_token_layer_units",
    "exact_cache_written_token_layer_units",
    "exact_catchup_avoided_token_layer_units",
    "restoration_requested_token_layer_units",
    "restoration_succeeded_token_layer_units",
    "restoration_failed_token_layer_units",
    "restoration_overwritten_token_layer_units",
    "fallback_token_layer_units",
    "fallback_event_count",
    "pending_token_count",
    "pending_token_layer_units",
    "restoration_flush_count",
    "restoration_layer_event_count",
    "restoration_token_record_count",
    "exact_catchup_flush_count",
)


TIMING_COMPONENTS = (
    "confidence_time_ms",
    "exact_parallel_catchup_time_ms",
    "restoration_compute_time_ms",
    "cache_staging_time_ms",
    "cache_commit_time_ms",
    "restoration_end_to_end_time_ms",
    # Native FREE fixed-source-layer-6 Task C2 direct K/V insertion: kept
    # distinct from the Task C1 exact-overwrite components above (this is
    # calibration/timer-plumbing evidence only, never a paper speed claim).
    "task_c2_restoration_compute_time_ms",
    "task_c2_cache_staging_time_ms",
    "task_c2_cache_commit_time_ms",
    "task_c2_insertion_end_to_end_time_ms",
)

# Distinct accounting policy mode for successful Task C2 direct-insertion
# tokens -- never Task C1 exact-overwrite semantics.
TASK_C2_DIRECT_INSERTION_POLICY_MODE = "phase3c_fixed_source6_direct_insertion"

# Distinct accounting policy mode for the FREE-aligned lazy BATCHED schedule:
# FREE's own pending stack and flush trigger are reused unchanged, but the
# flush restores all pending tokens with Phase-3c instead of exact-replaying
# them. Completeness is deliberately NOT required under this mode (terminal
# pending tokens are intentionally never restored).
TASK_C2_BATCHED_INSERTION_POLICY_MODE = "phase3c_fixed_source6_batched_lazy_insertion"

MISSING_KV_RUNTIME_ACCOUNTING_SCHEMA_VERSION = 2
MISSING_KV_RUNTIME_ACCOUNTING_RECORD_TYPE = "missing_kv_runtime_accounting"
MISSING_KV_RUNTIME_ACCOUNTING_PRODUCER_PROTOCOL = "missing_kv_runtime_accounting_v2"


def _as_nonnegative_int(value: Any) -> int:
    if value is None:
        return 0
    result = int(value)
    if result < 0:
        raise ValueError("missing-KV accounting counters must be non-negative")
    return result


@dataclass
class MissingKVRuntimeAccounting:
    """Aggregate missing-KV work counters for one evaluation/generation run."""

    counters: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in COUNTER_KEYS})
    task_c1_exact_overwrite: bool = True
    restoration_policy_mode: str = "unspecified"
    force_restore_all: bool = False
    recent_exact_window: int = 0
    complete_exact_catchup_coverage_required: bool = False
    aggregation_scope: str = "standalone_generation"
    evaluation_aggregation_active: bool = False
    generation_count: int = 0
    aggregate_reset_count: int = 0
    generation_local_reset_count: int = 0
    calm_first_crossing_token_count: int = 0
    calm_full_depth_fallback_token_count: int = 0
    calm_source_layer_counts: Dict[str, int] = field(default_factory=dict)
    calm_transaction_failure_token_count: int = 0
    generation_quality_run_valid: bool = True
    f2a_reference_generation_count: int = 0
    f2a_reference_generated_token_count: int = 0
    f2a_frozen_restoration_event_count: int = 0
    f2a_candidate_replay_counts_by_method: Dict[str, int] = field(default_factory=dict)
    f2a_full_depth_fallback_count: int = 0
    f2a_skipped_reason_counts: Dict[str, int] = field(default_factory=dict)
    f2a_schedule_validation_failure_count: int = 0
    f2a_variant_replay_failure_count: int = 0
    f2a_nonfinite_metric_failure_count: int = 0
    f2a_reference_state_mutation_failure_count: int = 0
    f2a_calm_event_considered_count: int = 0
    f2a_calm_event_produced_count: int = 0
    f2a_calm_event_finalized_count: int = 0
    f2a_calm_terminal_no_followup_count: int = 0
    f2a_calm_event_failure_count: int = 0
    f2a_calm_event_cap_skipped_count: int = 0
    f2a_exact_shadow_parity_failure_count: int = 0
    f2a_cap_excluded_event_count: int = 0
    f2a_cap_excluded_pending_token_count: int = 0
    f2a_terminal_no_followup_event_count: int = 0
    f2a_failed_event_count: int = 0
    f2a_failure_occurrence_count: int = 0
    f2a_blocking_failure_event_count: int = 0
    _f2a_failed_event_keys: set = field(default_factory=set, repr=False)
    # Native FREE fixed-source-layer-6 Task C2 direct K/V insertion.
    # Deliberately separate from the generic Task C1 `counters` dict: a
    # successful Task C2 token never requests/executes exact catch-up, so
    # folding its avoided work into `exact_catchup_avoided_token_layer_units`
    # would incorrectly require avoided <= exact_catchup_required for units
    # that were never part of an exact-catchup request at all.
    task_c2_requested_exit_tokens: int = 0
    task_c2_succeeded_exit_tokens: int = 0
    task_c2_failed_exit_tokens: int = 0
    task_c2_fallback_exit_tokens: int = 0
    task_c2_requested_token_layer_units: int = 0
    task_c2_inserted_token_layer_units: int = 0
    task_c2_exact_catchup_avoided_token_layer_units: int = 0

    # FREE-aligned lazy BATCHED Task C2 insertion. Kept separate from the
    # Immediate counters above because the unit of work differs: Immediate
    # attempts once per exiting token, batched attempts once per FREE flush
    # covering N pending tokens at a time. Terminal pending tokens are
    # intentionally never restored under the lazy schedule, so restored
    # tokens are counted independently of source-6 exits and completeness
    # must NOT be required (exits x 18 is the wrong denominator here).
    task_c2_batched_flush_attempts: int = 0
    task_c2_batched_flush_successes: int = 0
    task_c2_batched_flush_failures: int = 0
    task_c2_batched_fallback_flushes: int = 0
    task_c2_batched_restored_pending_tokens: int = 0
    task_c2_batched_requested_token_layer_units: int = 0
    task_c2_batched_inserted_token_layer_units: int = 0
    task_c2_batched_exact_catchup_avoided_token_layer_units: int = 0
    task_c2_batched_terminal_pending_tokens_avoided: int = 0

    def configure_policy(
        self,
        *,
        restoration_policy_mode: Optional[str] = None,
        force_restore_all: Optional[bool] = None,
        recent_exact_window: Optional[int] = None,
        complete_exact_catchup_coverage_required: Optional[bool] = None,
        task_c1_exact_overwrite: Optional[bool] = None,
    ) -> None:
        if restoration_policy_mode is not None:
            self.restoration_policy_mode = str(restoration_policy_mode)
        if force_restore_all is not None:
            self.force_restore_all = bool(force_restore_all)
        if recent_exact_window is not None:
            self.recent_exact_window = _as_nonnegative_int(recent_exact_window)
        if complete_exact_catchup_coverage_required is not None:
            self.complete_exact_catchup_coverage_required = bool(complete_exact_catchup_coverage_required)
        if task_c1_exact_overwrite is not None:
            self.task_c1_exact_overwrite = bool(task_c1_exact_overwrite)

    def reset(
        self,
        *,
        aggregation_scope: Optional[str] = None,
        evaluation_aggregation_active: Optional[bool] = None,
    ) -> None:
        for key in COUNTER_KEYS:
            self.counters[key] = 0
        self.generation_count = 0
        self.generation_local_reset_count = 0
        self.calm_first_crossing_token_count = 0
        self.calm_full_depth_fallback_token_count = 0
        self.calm_source_layer_counts = {}
        self.calm_transaction_failure_token_count = 0
        self.generation_quality_run_valid = True
        self.f2a_reference_generation_count = 0
        self.f2a_reference_generated_token_count = 0
        self.f2a_frozen_restoration_event_count = 0
        self.f2a_candidate_replay_counts_by_method = {}
        self.f2a_full_depth_fallback_count = 0
        self.f2a_skipped_reason_counts = {}
        self.f2a_schedule_validation_failure_count = 0
        self.f2a_variant_replay_failure_count = 0
        self.f2a_nonfinite_metric_failure_count = 0
        self.f2a_reference_state_mutation_failure_count = 0
        self.f2a_calm_event_considered_count = 0
        self.f2a_calm_event_produced_count = 0
        self.f2a_calm_event_finalized_count = 0
        self.f2a_calm_terminal_no_followup_count = 0
        self.f2a_calm_event_failure_count = 0
        self.f2a_calm_event_cap_skipped_count = 0
        self.f2a_exact_shadow_parity_failure_count = 0
        self.f2a_cap_excluded_event_count = 0
        self.f2a_cap_excluded_pending_token_count = 0
        self.f2a_terminal_no_followup_event_count = 0
        self.f2a_failed_event_count = 0
        self.f2a_failure_occurrence_count = 0
        self.f2a_blocking_failure_event_count = 0
        self._f2a_failed_event_keys = set()
        self.task_c2_requested_exit_tokens = 0
        self.task_c2_succeeded_exit_tokens = 0
        self.task_c2_failed_exit_tokens = 0
        self.task_c2_fallback_exit_tokens = 0
        self.task_c2_requested_token_layer_units = 0
        self.task_c2_inserted_token_layer_units = 0
        self.task_c2_exact_catchup_avoided_token_layer_units = 0
        self.task_c2_batched_flush_attempts = 0
        self.task_c2_batched_flush_successes = 0
        self.task_c2_batched_flush_failures = 0
        self.task_c2_batched_fallback_flushes = 0
        self.task_c2_batched_restored_pending_tokens = 0
        self.task_c2_batched_requested_token_layer_units = 0
        self.task_c2_batched_inserted_token_layer_units = 0
        self.task_c2_batched_exact_catchup_avoided_token_layer_units = 0
        self.task_c2_batched_terminal_pending_tokens_avoided = 0
        self.aggregate_reset_count += 1
        if aggregation_scope is not None:
            self.aggregation_scope = str(aggregation_scope)
        if evaluation_aggregation_active is not None:
            self.evaluation_aggregation_active = bool(evaluation_aggregation_active)

    def set_lifecycle(
        self,
        *,
        aggregation_scope: Optional[str] = None,
        evaluation_aggregation_active: Optional[bool] = None,
    ) -> None:
        if aggregation_scope is not None:
            self.aggregation_scope = str(aggregation_scope)
        if evaluation_aggregation_active is not None:
            self.evaluation_aggregation_active = bool(evaluation_aggregation_active)

    def record_generation_start(self) -> None:
        self.generation_count += 1

    def record_generation_local_reset(self) -> None:
        self.generation_local_reset_count += 1

    def add(self, key: str, value: Any = 1) -> None:
        if key not in self.counters:
            raise KeyError("unknown missing-KV counter: {}".format(key))
        self.counters[key] += _as_nonnegative_int(value)

    def record_generated_token(self, count: int = 1) -> None:
        self.add("generated_token_count", count)

    def record_calm_first_crossing(self, source_layer: int, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        if count == 0:
            return
        self.calm_first_crossing_token_count += count
        key = str(int(source_layer))
        self.calm_source_layer_counts[key] = self.calm_source_layer_counts.get(key, 0) + count

    def record_calm_full_depth_fallback(self, count: int = 1) -> None:
        self.calm_full_depth_fallback_token_count += _as_nonnegative_int(count)

    def record_calm_transaction_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.calm_transaction_failure_token_count += count
        if count:
            self.generation_quality_run_valid = False

    def record_f2a_reference_generation(self, count: int = 1) -> None:
        self.f2a_reference_generation_count += _as_nonnegative_int(count)

    def record_f2a_reference_generated_token(self, count: int = 1) -> None:
        self.f2a_reference_generated_token_count += _as_nonnegative_int(count)

    def record_f2a_frozen_event(self, count: int = 1) -> None:
        self.f2a_frozen_restoration_event_count += _as_nonnegative_int(count)

    def record_f2a_candidate_replay(self, method: str, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        if count == 0:
            return
        method_key = str(method)
        self.f2a_candidate_replay_counts_by_method[method_key] = (
            self.f2a_candidate_replay_counts_by_method.get(method_key, 0) + count
        )

    def record_f2a_full_depth_fallback(self, count: int = 1) -> None:
        self.f2a_full_depth_fallback_count += _as_nonnegative_int(count)

    def record_f2a_skipped(self, reason: str, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        if count == 0:
            return
        reason_key = str(reason)
        self.f2a_skipped_reason_counts[reason_key] = self.f2a_skipped_reason_counts.get(reason_key, 0) + count

    def record_f2a_schedule_validation_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_schedule_validation_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_variant_replay_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_variant_replay_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_nonfinite_metric_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_nonfinite_metric_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_reference_state_mutation_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_reference_state_mutation_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_failed_event(self, count: int = 1, event_uid: Optional[str] = None) -> None:
        count = _as_nonnegative_int(count)
        if count == 0:
            return
        if event_uid not in (None, ""):
            key = str(event_uid)
            if key in self._f2a_failed_event_keys:
                return
            self._f2a_failed_event_keys.add(key)
            self.f2a_failed_event_count += 1
            return
        self.f2a_failed_event_count += count

    def record_f2a_calm_event_considered(self, count: int = 1) -> None:
        self.f2a_calm_event_considered_count += _as_nonnegative_int(count)

    def record_f2a_calm_event_produced(self, count: int = 1) -> None:
        self.f2a_calm_event_produced_count += _as_nonnegative_int(count)

    def record_f2a_calm_event_finalized(self, count: int = 1) -> None:
        self.f2a_calm_event_finalized_count += _as_nonnegative_int(count)

    def record_f2a_calm_terminal_no_followup(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_calm_terminal_no_followup_count += count
        self.f2a_terminal_no_followup_event_count += count

    def record_f2a_calm_event_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_calm_event_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_calm_event_cap_skipped(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_calm_event_cap_skipped_count += count
        self.record_f2a_cap_excluded_event(count=count, pending_token_count=count)

    def record_f2a_exact_shadow_parity_failure(self, count: int = 1) -> None:
        count = _as_nonnegative_int(count)
        self.f2a_exact_shadow_parity_failure_count += count
        self.f2a_failure_occurrence_count += count
        self.f2a_blocking_failure_event_count += count

    def record_f2a_cap_excluded_event(self, count: int = 1, pending_token_count: int = 0) -> None:
        count = _as_nonnegative_int(count)
        pending = _as_nonnegative_int(pending_token_count)
        self.f2a_cap_excluded_event_count += count
        self.f2a_cap_excluded_pending_token_count += pending

    def record_nominal_skip(self, source_layer: int, decoder_layer_count: int, token_count: int = 1) -> None:
        units = max(0, int(decoder_layer_count) - int(source_layer)) * _as_nonnegative_int(token_count)
        self.add("nominal_skipped_token_layer_units", units)

    def record_exact_catchup_flush(
        self,
        pending_token_count: int,
        num_catchup_layers: int,
    ) -> None:
        units = self.record_exact_catchup_required(
            pending_token_count=pending_token_count,
            num_catchup_layers=num_catchup_layers,
        )
        self.record_exact_catchup_executed(units)

    def record_exact_catchup_required(
        self,
        pending_token_count: int,
        num_catchup_layers: int,
    ) -> int:
        pending = _as_nonnegative_int(pending_token_count)
        layers = _as_nonnegative_int(num_catchup_layers)
        units = pending * layers
        if pending > 0:
            self.add("exact_catchup_flush_count", 1)
        self.add("pending_token_count", pending)
        self.add("pending_token_layer_units", units)
        self.add("exact_catchup_required_token_layer_units", units)
        return units

    def record_exact_catchup_executed(self, token_layer_units: int) -> None:
        units = _as_nonnegative_int(token_layer_units)
        self.add("exact_catchup_executed_token_layer_units", units)

    def record_exact_cache_written(self, token_layer_units: int) -> None:
        self.add("exact_cache_written_token_layer_units", token_layer_units)

    def record_restoration_flush(self, enabled: bool, restore_relative_count: int) -> None:
        if bool(enabled) and _as_nonnegative_int(restore_relative_count) > 0:
            self.add("restoration_flush_count", 1)

    def record_restoration_layer(
        self,
        requested: int,
        succeeded: int,
        overwritten: int,
        fallback: int,
        token_records: Optional[int] = None,
    ) -> None:
        requested = _as_nonnegative_int(requested)
        succeeded = _as_nonnegative_int(succeeded)
        overwritten = _as_nonnegative_int(overwritten)
        fallback = _as_nonnegative_int(fallback)
        token_records = requested if token_records is None else _as_nonnegative_int(token_records)
        failed = requested - succeeded if succeeded <= requested else 0
        if requested > 0:
            self.add("restoration_layer_event_count", 1)
        self.add("restoration_requested_token_layer_units", requested)
        self.add("restoration_succeeded_token_layer_units", succeeded)
        self.add("restoration_failed_token_layer_units", failed)
        self.add("restoration_overwritten_token_layer_units", overwritten)
        self.add("restoration_token_record_count", token_records)
        self.add("fallback_token_layer_units", fallback)
        if fallback > 0:
            self.add("fallback_event_count", 1)

    def record_task_c2_direct_insertion_attempt(
        self,
        *,
        success: bool,
        fallback: bool,
        requested_units: int,
        inserted_units: int = 0,
    ) -> None:
        """Record one Native FREE fixed-source-layer-6 Task C2 direct-
        insertion attempt for one exited token. ``requested_units`` is
        ``len(self.block) - source_layer`` regardless of outcome.
        ``inserted_units`` (only meaningful on success) is credited both as
        inserted work and as exact-catchup avoided work, since a successful
        Task C2 token never requests or executes exact catch-up for those
        layers at all."""

        requested_units = _as_nonnegative_int(requested_units)
        self.task_c2_requested_exit_tokens += 1
        self.task_c2_requested_token_layer_units += requested_units
        if success:
            inserted_units = _as_nonnegative_int(inserted_units)
            self.task_c2_succeeded_exit_tokens += 1
            self.task_c2_inserted_token_layer_units += inserted_units
            self.task_c2_exact_catchup_avoided_token_layer_units += inserted_units
        else:
            self.task_c2_failed_exit_tokens += 1
            if fallback:
                self.task_c2_fallback_exit_tokens += 1

    def record_task_c2_batched_flush_attempt(
        self,
        *,
        success: bool,
        fallback: bool,
        pending_token_count: int,
        requested_units: int,
        inserted_units: int = 0,
    ) -> None:
        """Record one FREE-aligned lazy BATCHED Task C2 flush attempt.

        The unit here is one FREE synchronized-flush point covering
        ``pending_token_count`` pending early-exit tokens at once, unlike the
        Immediate recorder above whose unit is a single exiting token.
        ``requested_units`` is ``pending_token_count * (len(block) -
        source_layer)``. On success the inserted units are credited both as
        inserted work and as exact catch-up avoided, because a successful
        batched flush never exact-replays any pending token."""

        pending_token_count = _as_nonnegative_int(pending_token_count)
        requested_units = _as_nonnegative_int(requested_units)
        self.task_c2_batched_flush_attempts += 1
        self.task_c2_batched_requested_token_layer_units += requested_units
        if success:
            inserted_units = _as_nonnegative_int(inserted_units)
            self.task_c2_batched_flush_successes += 1
            self.task_c2_batched_restored_pending_tokens += pending_token_count
            self.task_c2_batched_inserted_token_layer_units += inserted_units
            self.task_c2_batched_exact_catchup_avoided_token_layer_units += inserted_units
        else:
            self.task_c2_batched_flush_failures += 1
            if fallback:
                self.task_c2_batched_fallback_flushes += 1

    def record_task_c2_batched_terminal_pending(self, pending_token_count: int) -> None:
        """Record early-exit tokens still pending when generation ended under
        the lazy batched schedule. Their deep K/V were never needed by any
        later token, so neither Phase-3c restoration nor exact replay was
        performed for them -- this is avoided work, not missing work, and is
        exactly why batched completeness must not be measured as
        ``source-6 exits x 18``."""

        self.task_c2_batched_terminal_pending_tokens_avoided += _as_nonnegative_int(pending_token_count)

    def validate_task_c2_batched(self) -> Dict[str, Any]:
        errors = []
        if self.task_c2_batched_flush_attempts != (
            self.task_c2_batched_flush_successes + self.task_c2_batched_flush_failures
        ):
            errors.append("task_c2_batched_flush_attempts_mismatch")
        if self.task_c2_batched_fallback_flushes > self.task_c2_batched_flush_failures:
            errors.append("task_c2_batched_fallback_exceeds_failed")
        if self.task_c2_batched_inserted_token_layer_units > self.task_c2_batched_requested_token_layer_units:
            errors.append("task_c2_batched_inserted_exceeds_requested")
        if (
            self.task_c2_batched_exact_catchup_avoided_token_layer_units
            > self.task_c2_batched_requested_token_layer_units
        ):
            errors.append("task_c2_batched_avoided_exceeds_requested")
        if self.task_c2_batched_inserted_token_layer_units and not self.task_c2_batched_restored_pending_tokens:
            errors.append("task_c2_batched_inserted_without_restored_tokens")
        return {"status": "ok" if not errors else "invalid", "errors": errors}

    def validate_task_c2(self) -> Dict[str, Any]:
        errors = []
        if self.task_c2_requested_exit_tokens != (
            self.task_c2_succeeded_exit_tokens + self.task_c2_failed_exit_tokens
        ):
            errors.append("task_c2_requested_exit_tokens_mismatch")
        if self.task_c2_fallback_exit_tokens > self.task_c2_failed_exit_tokens:
            errors.append("task_c2_fallback_exceeds_failed")
        if self.task_c2_inserted_token_layer_units > self.task_c2_requested_token_layer_units:
            errors.append("task_c2_inserted_exceeds_requested")
        if self.task_c2_exact_catchup_avoided_token_layer_units > self.task_c2_requested_token_layer_units:
            errors.append("task_c2_avoided_exceeds_requested")
        return {"status": "ok" if not errors else "invalid", "errors": errors}

    def net_avoided_token_layer_units(self) -> int:
        """Diagnostic-only derived quantity: nominal minus executed exact
        catch-up units. This is distinct from ``exact_catchup_avoided_token_layer_units``,
        which is an explicit-avoidance counter that must stay zero under Task
        C1's exact-overwrite semantics. A positive value here from a failed or
        incomplete transaction is diagnostic only and must not be treated as a
        paper-valid speedup."""

        c = self.counters
        return c["nominal_skipped_token_layer_units"] - c["exact_catchup_executed_token_layer_units"]

    def validate(self) -> Dict[str, Any]:
        errors = []
        c = self.counters
        if c["restoration_succeeded_token_layer_units"] > c["restoration_requested_token_layer_units"]:
            errors.append("restoration_succeeded_exceeds_requested")
        if c["restoration_overwritten_token_layer_units"] > c["restoration_requested_token_layer_units"]:
            errors.append("restoration_overwritten_exceeds_requested")
        if (
            c["restoration_succeeded_token_layer_units"]
            + c["restoration_failed_token_layer_units"]
            != c["restoration_requested_token_layer_units"]
        ):
            errors.append("restoration_requested_mismatch")
        if c["restoration_overwritten_token_layer_units"] > c["restoration_succeeded_token_layer_units"]:
            errors.append("restoration_overwritten_exceeds_succeeded")
        if self.complete_exact_catchup_coverage_required:
            if c["restoration_token_record_count"] != c["restoration_requested_token_layer_units"]:
                errors.append("restoration_token_record_count_mismatch")
            if c["restoration_layer_event_count"] > c["restoration_token_record_count"]:
                errors.append("invalid_restoration_flush_layer_relationship")
            if c["restoration_flush_count"] > c["restoration_layer_event_count"] and c["restoration_flush_count"] > 0:
                errors.append("invalid_restoration_flush_layer_relationship")
        if c["exact_catchup_avoided_token_layer_units"] > c["exact_catchup_required_token_layer_units"]:
            errors.append("exact_catchup_avoided_exceeds_required")
        if self.task_c1_exact_overwrite:
            if c["exact_catchup_executed_token_layer_units"] != c["exact_catchup_required_token_layer_units"]:
                errors.append("task_c1_exact_catchup_executed_mismatch")
            if c["exact_catchup_avoided_token_layer_units"] != 0:
                errors.append("task_c1_exact_catchup_avoided_nonzero")
        if self.complete_exact_catchup_coverage_required:
            required = c["exact_catchup_required_token_layer_units"]
            if c["restoration_requested_token_layer_units"] != required:
                errors.append("taskc1_restoration_requested_exact_catchup_mismatch")
            if c["restoration_succeeded_token_layer_units"] != required:
                errors.append("taskc1_restoration_succeeded_exact_catchup_mismatch")
            if c["restoration_overwritten_token_layer_units"] != required:
                errors.append("taskc1_restoration_overwritten_exact_catchup_mismatch")
            if c["restoration_token_record_count"] != required:
                errors.append("taskc1_restoration_record_exact_catchup_mismatch")
        if self.restoration_policy_mode == "exact_catchup_no_approximation":
            required = c["exact_catchup_required_token_layer_units"]
            if c["exact_catchup_executed_token_layer_units"] != required:
                errors.append("exact_catchup_executed_required_mismatch")
            if c["exact_cache_written_token_layer_units"] != required:
                errors.append("exact_cache_written_required_mismatch")
            if c["exact_catchup_avoided_token_layer_units"] != 0:
                errors.append("exact_catchup_avoided_nonzero")
            if self.calm_first_crossing_token_count + self.calm_full_depth_fallback_token_count != c["generated_token_count"]:
                errors.append("exact_catchup_token_population_mismatch")
            for key in (
                "restoration_requested_token_layer_units",
                "restoration_succeeded_token_layer_units",
                "restoration_failed_token_layer_units",
                "restoration_overwritten_token_layer_units",
                "fallback_token_layer_units",
                "fallback_event_count",
            ):
                if c[key] != 0:
                    errors.append("exact_catchup_{}_nonzero".format(key))
        return {"status": "ok" if not errors else "invalid", "errors": errors}

    def to_summary(self, timing: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        validation = self.validate()
        return {
            "schema_version": MISSING_KV_RUNTIME_ACCOUNTING_SCHEMA_VERSION,
            "record_type": MISSING_KV_RUNTIME_ACCOUNTING_RECORD_TYPE,
            "producer_protocol": MISSING_KV_RUNTIME_ACCOUNTING_PRODUCER_PROTOCOL,
            "measurement_role": "missing_kv_runtime_accounting",
            "counter_unit": "decoder_token_layer",
            # Diagnostic-only derived quantity -- see net_avoided_token_layer_units()
            # docstring. Reported as invalid (rather than folded into the
            # blocking `validation` errors below) when negative, since a
            # negative value only arises from partial/isolated accounting
            # (e.g. exact-catchup recorded without matching nominal-skip
            # tracking) and must never silently poison unrelated counter
            # validation.
            "net_avoided_token_layer_units": self.net_avoided_token_layer_units(),
            "net_avoided_token_layer_units_valid": self.net_avoided_token_layer_units() >= 0,
            "task_c1_exact_overwrite": self.task_c1_exact_overwrite,
            "restoration_policy_mode": self.restoration_policy_mode,
            "force_restore_all": self.force_restore_all,
            "recent_exact_window": self.recent_exact_window,
            "complete_exact_catchup_coverage_required": self.complete_exact_catchup_coverage_required,
            "complete_exact_catchup_coverage_satisfied": (
                validation["status"] == "ok"
                if self.complete_exact_catchup_coverage_required
                else None
            ),
            "aggregation_scope": self.aggregation_scope,
            "evaluation_aggregation_active_at_export": self.evaluation_aggregation_active,
            "generation_count": self.generation_count,
            "aggregate_reset_count": self.aggregate_reset_count,
            "generation_local_reset_count": self.generation_local_reset_count,
            "calm_first_crossing_token_count": self.calm_first_crossing_token_count,
            "calm_full_depth_fallback_token_count": self.calm_full_depth_fallback_token_count,
            "policy_no_crossing_full_depth_token_count": self.calm_full_depth_fallback_token_count,
            "calm_source_layer_counts": dict(sorted(self.calm_source_layer_counts.items())),
            "calm_transaction_failure_token_count": self.calm_transaction_failure_token_count,
            "f2a_reference_generation_count": self.f2a_reference_generation_count,
            "f2a_reference_generated_token_count": self.f2a_reference_generated_token_count,
            "f2a_frozen_restoration_event_count": self.f2a_frozen_restoration_event_count,
            "f2a_candidate_replay_counts_by_method": dict(sorted(self.f2a_candidate_replay_counts_by_method.items())),
            "f2a_full_depth_fallback_count": self.f2a_full_depth_fallback_count,
            "f2a_skipped_reason_counts": dict(sorted(self.f2a_skipped_reason_counts.items())),
            "f2a_schedule_validation_failure_count": self.f2a_schedule_validation_failure_count,
            "f2a_variant_replay_failure_count": self.f2a_variant_replay_failure_count,
            "f2a_nonfinite_metric_failure_count": self.f2a_nonfinite_metric_failure_count,
            "f2a_reference_state_mutation_failure_count": self.f2a_reference_state_mutation_failure_count,
            "f2a_calm_event_considered_count": self.f2a_calm_event_considered_count,
            "f2a_calm_event_produced_count": self.f2a_calm_event_produced_count,
            "f2a_calm_event_finalized_count": self.f2a_calm_event_finalized_count,
            "f2a_calm_terminal_no_followup_count": self.f2a_calm_terminal_no_followup_count,
            "f2a_calm_event_failure_count": self.f2a_calm_event_failure_count,
            "f2a_calm_event_cap_skipped_count": self.f2a_calm_event_cap_skipped_count,
            "f2a_exact_shadow_parity_failure_count": self.f2a_exact_shadow_parity_failure_count,
            "f2a_cap_excluded_event_count": self.f2a_cap_excluded_event_count,
            "f2a_cap_excluded_pending_token_count": self.f2a_cap_excluded_pending_token_count,
            "f2a_terminal_no_followup_event_count": self.f2a_terminal_no_followup_event_count,
            "f2a_failed_event_count": self.f2a_failed_event_count,
            "f2a_failure_occurrence_count": self.f2a_failure_occurrence_count,
            "f2a_blocking_failure_event_count": self.f2a_blocking_failure_event_count,
            "task_c2_policy_mode": TASK_C2_DIRECT_INSERTION_POLICY_MODE,
            "task_c2_requested_exit_tokens": self.task_c2_requested_exit_tokens,
            "task_c2_succeeded_exit_tokens": self.task_c2_succeeded_exit_tokens,
            "task_c2_failed_exit_tokens": self.task_c2_failed_exit_tokens,
            "task_c2_fallback_exit_tokens": self.task_c2_fallback_exit_tokens,
            "task_c2_requested_token_layer_units": self.task_c2_requested_token_layer_units,
            "task_c2_inserted_token_layer_units": self.task_c2_inserted_token_layer_units,
            "task_c2_exact_catchup_avoided_token_layer_units": self.task_c2_exact_catchup_avoided_token_layer_units,
            "task_c2_validation": self.validate_task_c2(),
            "task_c2_batched_policy_mode": TASK_C2_BATCHED_INSERTION_POLICY_MODE,
            "task_c2_batched_flush_attempts": self.task_c2_batched_flush_attempts,
            "task_c2_batched_flush_successes": self.task_c2_batched_flush_successes,
            "task_c2_batched_flush_failures": self.task_c2_batched_flush_failures,
            "task_c2_batched_fallback_flushes": self.task_c2_batched_fallback_flushes,
            "task_c2_batched_restored_pending_tokens": self.task_c2_batched_restored_pending_tokens,
            "task_c2_batched_requested_token_layer_units": self.task_c2_batched_requested_token_layer_units,
            "task_c2_batched_inserted_token_layer_units": self.task_c2_batched_inserted_token_layer_units,
            "task_c2_batched_exact_catchup_avoided_token_layer_units": (
                self.task_c2_batched_exact_catchup_avoided_token_layer_units
            ),
            "task_c2_batched_terminal_pending_tokens_avoided": (
                self.task_c2_batched_terminal_pending_tokens_avoided
            ),
            "task_c2_batched_validation": self.validate_task_c2_batched(),
            "task_c2_speed_claim_valid": False,
            "generation_quality_run_valid": bool(self.generation_quality_run_valid),
            "speed_claim_valid": False,
            "counters": dict(self.counters),
            "validation": validation,
            "timing": dict(timing or {}),
        }

    def scalar_metrics(self, metric_key_prefix: str = "eval") -> Dict[str, Any]:
        validation = self.validate()
        metrics = {
            "{}_missing_kv_{}".format(metric_key_prefix, key): value
            for key, value in self.counters.items()
        }
        metrics["{}_missing_kv_accounting_valid".format(metric_key_prefix)] = validation["status"] == "ok"
        metrics["{}_missing_kv_speed_claim_valid".format(metric_key_prefix)] = False
        metrics["{}_missing_kv_net_avoided_token_layer_units".format(metric_key_prefix)] = self.net_avoided_token_layer_units()
        metrics["{}_missing_kv_generation_count".format(metric_key_prefix)] = self.generation_count
        metrics["{}_missing_kv_calm_first_crossing_token_count".format(metric_key_prefix)] = self.calm_first_crossing_token_count
        metrics["{}_missing_kv_calm_full_depth_fallback_token_count".format(metric_key_prefix)] = self.calm_full_depth_fallback_token_count
        metrics["{}_missing_kv_policy_no_crossing_full_depth_token_count".format(metric_key_prefix)] = self.calm_full_depth_fallback_token_count
        metrics["{}_missing_kv_calm_transaction_failure_token_count".format(metric_key_prefix)] = self.calm_transaction_failure_token_count
        metrics["{}_missing_kv_generation_quality_run_valid".format(metric_key_prefix)] = bool(self.generation_quality_run_valid)
        metrics["{}_missing_kv_f2a_reference_generation_count".format(metric_key_prefix)] = self.f2a_reference_generation_count
        metrics["{}_missing_kv_f2a_reference_generated_token_count".format(metric_key_prefix)] = self.f2a_reference_generated_token_count
        metrics["{}_missing_kv_f2a_frozen_restoration_event_count".format(metric_key_prefix)] = self.f2a_frozen_restoration_event_count
        metrics["{}_missing_kv_f2a_full_depth_fallback_count".format(metric_key_prefix)] = self.f2a_full_depth_fallback_count
        metrics["{}_missing_kv_f2a_schedule_validation_failure_count".format(metric_key_prefix)] = self.f2a_schedule_validation_failure_count
        metrics["{}_missing_kv_f2a_variant_replay_failure_count".format(metric_key_prefix)] = self.f2a_variant_replay_failure_count
        metrics["{}_missing_kv_f2a_nonfinite_metric_failure_count".format(metric_key_prefix)] = self.f2a_nonfinite_metric_failure_count
        metrics["{}_missing_kv_f2a_reference_state_mutation_failure_count".format(metric_key_prefix)] = self.f2a_reference_state_mutation_failure_count
        metrics["{}_missing_kv_f2a_calm_event_considered_count".format(metric_key_prefix)] = self.f2a_calm_event_considered_count
        metrics["{}_missing_kv_f2a_calm_event_produced_count".format(metric_key_prefix)] = self.f2a_calm_event_produced_count
        metrics["{}_missing_kv_f2a_calm_event_finalized_count".format(metric_key_prefix)] = self.f2a_calm_event_finalized_count
        metrics["{}_missing_kv_f2a_calm_terminal_no_followup_count".format(metric_key_prefix)] = self.f2a_calm_terminal_no_followup_count
        metrics["{}_missing_kv_f2a_calm_event_failure_count".format(metric_key_prefix)] = self.f2a_calm_event_failure_count
        metrics["{}_missing_kv_f2a_calm_event_cap_skipped_count".format(metric_key_prefix)] = self.f2a_calm_event_cap_skipped_count
        metrics["{}_missing_kv_f2a_exact_shadow_parity_failure_count".format(metric_key_prefix)] = self.f2a_exact_shadow_parity_failure_count
        metrics["{}_missing_kv_f2a_cap_excluded_event_count".format(metric_key_prefix)] = self.f2a_cap_excluded_event_count
        metrics["{}_missing_kv_f2a_cap_excluded_pending_token_count".format(metric_key_prefix)] = self.f2a_cap_excluded_pending_token_count
        metrics["{}_missing_kv_f2a_terminal_no_followup_event_count".format(metric_key_prefix)] = self.f2a_terminal_no_followup_event_count
        metrics["{}_missing_kv_f2a_failed_event_count".format(metric_key_prefix)] = self.f2a_failed_event_count
        metrics["{}_missing_kv_f2a_failure_occurrence_count".format(metric_key_prefix)] = self.f2a_failure_occurrence_count
        metrics["{}_missing_kv_f2a_blocking_failure_event_count".format(metric_key_prefix)] = self.f2a_blocking_failure_event_count
        metrics["{}_missing_kv_task_c2_requested_exit_tokens".format(metric_key_prefix)] = self.task_c2_requested_exit_tokens
        metrics["{}_missing_kv_task_c2_succeeded_exit_tokens".format(metric_key_prefix)] = self.task_c2_succeeded_exit_tokens
        metrics["{}_missing_kv_task_c2_failed_exit_tokens".format(metric_key_prefix)] = self.task_c2_failed_exit_tokens
        metrics["{}_missing_kv_task_c2_fallback_exit_tokens".format(metric_key_prefix)] = self.task_c2_fallback_exit_tokens
        metrics["{}_missing_kv_task_c2_validation_valid".format(metric_key_prefix)] = (
            self.validate_task_c2()["status"] == "ok"
        )
        return metrics


class MissingKVComponentTimer:
    """Optional component timer.

    CPU timing uses ``time.perf_counter`` and resolves immediately at each
    block's exit. CUDA timing uses deferred resolution: ``time_block`` only
    records a start/end CUDA event pair and appends it to a pending list --
    it never synchronizes. ``finalize()`` must be called once (typically at
    evaluation completion) to synchronize a single time and resolve every
    pending event pair into ``values_ms``. This keeps a source-4 event with
    up to 20 target layers from synchronizing 20 times.

    When CUDA timing is unavailable the component is explicitly absent
    instead of reported as zero. Timing failures (event creation, record,
    synchronize, or elapsed-time errors; negative/NaN/Inf elapsed values)
    never raise out of ``time_block``/``finalize`` -- they are captured in
    ``timing_errors`` and surfaced via ``timing_valid`` in ``to_summary()``,
    so a diagnostic timing failure can never silently break, or be confused
    with, the underlying model computation it wraps.
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: Optional[str] = None,
        cuda_module: Optional[Any] = None,
    ) -> None:
        self.enabled = bool(enabled)
        requested_backend = backend or ("auto" if self.enabled else "disabled")
        if not self.enabled:
            requested_backend = "disabled"
        self.cuda_module = cuda_module
        if self.enabled and requested_backend not in {"auto", "cuda_events", "cpu_perf_counter", "disabled"}:
            raise ValueError("unsupported missing-KV timing backend: {}".format(requested_backend))
        if self.enabled and requested_backend == "disabled":
            self.enabled = False
        self.timing_backend_requested = requested_backend
        self.values_ms: Dict[str, float] = {}
        self.component_backends: Dict[str, str] = {}
        self.synchronization_semantics = "disabled"
        self._pending_cuda_events = []
        self._timing_errors = []

    @staticmethod
    def _load_cuda_module() -> Optional[Any]:
        try:
            import torch  # type: ignore

            return torch.cuda
        except Exception:
            return None

    @staticmethod
    def _cuda_available(cuda_module: Any) -> bool:
        is_available = getattr(cuda_module, "is_available", None)
        if callable(is_available):
            try:
                return bool(is_available())
            except Exception:
                return False
        return hasattr(cuda_module, "Event")

    def reset(self) -> None:
        self.values_ms.clear()
        self.component_backends.clear()
        self.synchronization_semantics = "disabled"
        self._pending_cuda_events = []
        self._timing_errors = []

    def _resolve_backend_for_device(self, device: Optional[Any] = None) -> str:
        if not self.enabled:
            return "disabled"
        backend = self.timing_backend_requested
        if backend == "auto":
            device_type = getattr(device, "type", None)
            if str(device_type) == "cuda":
                backend = "cuda_events"
            else:
                backend = "cpu_perf_counter"
        if backend == "cuda_events":
            self.cuda_module = self.cuda_module or self._load_cuda_module()
            if self.cuda_module is None or not self._cuda_available(self.cuda_module):
                self.synchronization_semantics = "cuda_events_unavailable"
                return "cuda_events_unavailable"
            self.synchronization_semantics = "cuda_events_deferred_synchronize_at_finalize"
            return "cuda_events"
        if backend == "cpu_perf_counter":
            device_type = getattr(device, "type", None)
            if str(device_type) == "cuda":
                self.synchronization_semantics = "host_launch_timing_for_async_cuda"
            else:
                self.synchronization_semantics = "host_wall_clock"
            return "cpu_perf_counter"
        self.synchronization_semantics = backend
        return backend

    @contextlib.contextmanager
    def time_block(
        self,
        component_name: str,
        device: Optional[Any] = None,
        on_resolved: Optional[Any] = None,
    ) -> Iterator[None]:
        """``on_resolved``, when given, is called exactly once as
        ``on_resolved(elapsed_ms, backend)`` with ``elapsed_ms=None`` when no
        timing could be produced (disabled/unavailable/failed) -- immediately
        for CPU timing, or later from ``finalize()`` for deferred CUDA
        timing. This lets a caller (e.g. a per-target-layer overhead event)
        capture this specific invocation's elapsed time without changing the
        aggregate per-component accounting in ``values_ms``.
        """

        if not self.enabled:
            yield
            return
        backend = self._resolve_backend_for_device(device)
        if backend == "cuda_events":
            try:
                start_event = self.cuda_module.Event(enable_timing=True)
                end_event = self.cuda_module.Event(enable_timing=True)
                start_event.record()
            except Exception as exc:
                self._timing_errors.append(
                    "cuda_event_creation_or_start_record_failed:{}:{}".format(component_name, exc)
                )
                self.component_backends[component_name] = "cuda_events_error"
                if on_resolved is not None:
                    on_resolved(None, "cuda_events_error")
                yield
                return
            try:
                yield
            finally:
                try:
                    end_event.record()
                except Exception as exc:
                    self._timing_errors.append(
                        "cuda_event_end_record_failed:{}:{}".format(component_name, exc)
                    )
                    self.component_backends[component_name] = "cuda_events_error"
                    if on_resolved is not None:
                        on_resolved(None, "cuda_events_error")
                else:
                    # Deferred resolution: append the pair and move on. No
                    # synchronize and no elapsed_time() call here -- those
                    # happen exactly once, for every pending pair, in
                    # finalize().
                    self._pending_cuda_events.append((component_name, start_event, end_event, on_resolved))
                    self.component_backends[component_name] = "cuda_events"
            return
        if backend in {"cuda_events_unavailable", "disabled"}:
            self.component_backends[component_name] = backend
            if on_resolved is not None:
                on_resolved(None, backend)
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.values_ms[component_name] = self.values_ms.get(component_name, 0.0) + elapsed_ms
            self.component_backends[component_name] = "cpu_perf_counter"
            if on_resolved is not None:
                on_resolved(elapsed_ms, "cpu_perf_counter")

    def finalize(self) -> None:
        """Synchronize once (if any CUDA events are pending) and resolve
        every pending event pair into ``values_ms``. Idempotent: safe to call
        multiple times, and a no-op when nothing is pending."""

        if not self.enabled or not self._pending_cuda_events:
            return
        pending = self._pending_cuda_events
        self._pending_cuda_events = []
        synchronize = getattr(self.cuda_module, "synchronize", None) if self.cuda_module is not None else None
        try:
            if callable(synchronize):
                synchronize()
            else:
                self._timing_errors.append("cuda_synchronize_unavailable")
                for component_name, _start_event, _end_event, on_resolved in pending:
                    self._timing_errors.append(
                        "cuda_elapsed_time_unresolved_after_sync_unavailable:{}".format(component_name)
                    )
                    if on_resolved is not None:
                        on_resolved(None, "cuda_events_error")
                return
        except Exception as exc:
            self._timing_errors.append("cuda_synchronize_failed:{}".format(exc))
            for component_name, _start_event, _end_event, on_resolved in pending:
                self._timing_errors.append(
                    "cuda_elapsed_time_unresolved_after_sync_failure:{}".format(component_name)
                )
                if on_resolved is not None:
                    on_resolved(None, "cuda_events_error")
            return
        for component_name, start_event, end_event, on_resolved in pending:
            try:
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception as exc:
                self._timing_errors.append("cuda_elapsed_time_failed:{}:{}".format(component_name, exc))
                if on_resolved is not None:
                    on_resolved(None, "cuda_events_error")
                continue
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                self._timing_errors.append(
                    "cuda_elapsed_time_invalid:{}:{}".format(component_name, elapsed_ms)
                )
                if on_resolved is not None:
                    on_resolved(None, "cuda_events_error")
                continue
            self.values_ms[component_name] = self.values_ms.get(component_name, 0.0) + elapsed_ms
            if on_resolved is not None:
                on_resolved(elapsed_ms, "cuda_events")

    def _aggregate_resolved_backend(self) -> str:
        if not self.enabled:
            return "disabled"
        backends = set(self.component_backends.values())
        if not backends:
            return "not_resolved"
        if len(backends) == 1:
            return next(iter(backends))
        return "mixed"

    def to_summary(self) -> Dict[str, Any]:
        values = {name: self.values_ms[name] for name in sorted(self.values_ms)}
        aggregate_sources = [
            "restoration_compute_time_ms",
            "cache_staging_time_ms",
            "cache_commit_time_ms",
        ]
        if "restoration_end_to_end_time_ms" not in values and any(name in values for name in aggregate_sources):
            values["restoration_end_to_end_time_ms"] = sum(values.get(name, 0.0) for name in aggregate_sources)
            aggregate_backends = {
                self.component_backends.get(name)
                for name in aggregate_sources
                if name in self.component_backends
            }
            if len(aggregate_backends) == 1:
                self.component_backends["restoration_end_to_end_time_ms"] = next(iter(aggregate_backends))
            elif len(aggregate_backends) > 1:
                self.component_backends["restoration_end_to_end_time_ms"] = "mixed"
        # Task C2 direct-insertion timing components, kept distinct from the
        # Task C1 exact-overwrite ones above -- same derivation pattern, not
        # a second timing framework.
        task_c2_aggregate_sources = [
            "task_c2_restoration_compute_time_ms",
            "task_c2_cache_staging_time_ms",
            "task_c2_cache_commit_time_ms",
        ]
        if "task_c2_insertion_end_to_end_time_ms" not in values and any(
            name in values for name in task_c2_aggregate_sources
        ):
            values["task_c2_insertion_end_to_end_time_ms"] = sum(
                values.get(name, 0.0) for name in task_c2_aggregate_sources
            )
            task_c2_aggregate_backends = {
                self.component_backends.get(name)
                for name in task_c2_aggregate_sources
                if name in self.component_backends
            }
            if len(task_c2_aggregate_backends) == 1:
                self.component_backends["task_c2_insertion_end_to_end_time_ms"] = next(iter(task_c2_aggregate_backends))
            elif len(task_c2_aggregate_backends) > 1:
                self.component_backends["task_c2_insertion_end_to_end_time_ms"] = "mixed"
        unavailable = [name for name in TIMING_COMPONENTS if name not in values]
        resolved_backend = self._aggregate_resolved_backend()
        # Unresolved pending events (finalize() was never called, or a
        # synchronize/elapsed-time failure left some unresolved) invalidate
        # timing output rather than silently reporting a partial result.
        timing_valid = not self._timing_errors and not self._pending_cuda_events
        return {
            "timing_enabled": self.enabled,
            "timing_backend": resolved_backend,
            "timing_backend_requested": self.timing_backend_requested,
            "timing_backend_resolved": resolved_backend,
            "component_backends": dict(sorted(self.component_backends.items())),
            "timing_is_synchronized": self.synchronization_semantics in {
                "cuda_events_deferred_synchronize_at_finalize",
                "host_wall_clock",
            },
            "synchronization_semantics": self.synchronization_semantics,
            "timing_scope": "runtime_missing_kv_components",
            "restoration_end_to_end_source": (
                "sum_of_non_overlapping_compute_staging_commit"
                if "restoration_end_to_end_time_ms" in values
                and any(name in self.values_ms for name in aggregate_sources)
                else None
            ),
            "timing_overhead_warning": (
                "component timing is diagnostic and can perturb runtime" if self.enabled else None
            ),
            "components_ms": values,
            "unavailable_components": unavailable,
            "timing_valid": timing_valid,
            "timing_errors": list(self._timing_errors),
            "unresolved_pending_event_count": len(self._pending_cuda_events),
            "speed_claim_valid": False,
        }


class GenerationWallTimer:
    """Opt-in run-level generation-only wall-time timer.

    Wraps exactly one call -- ``gen_model.generate(...)`` in
    ``SumTrainer.prediction_step`` -- and nothing else: not dataset
    preprocessing, tokenization, data loading, ROUGE computation, prediction
    decoding, JSON/CSV writing, or archive creation. It is a run-level
    denominator, reset once per evaluation and accumulated across every
    sample-level ``generate()`` call; it must never add synchronization
    inside individual decoding steps.
    """

    def __init__(self, enabled: bool = False, backend: Optional[str] = None) -> None:
        self.enabled = bool(enabled)
        requested = backend or ("auto" if self.enabled else "disabled")
        if not self.enabled:
            requested = "disabled"
        self.timing_backend_requested = requested
        self.generation_wall_time_ms = 0.0
        self.generation_call_count = 0
        self.generation_failed_call_count = 0
        self._errors: list = []
        self.synchronization_semantics = "disabled"

    def reset(self) -> None:
        self.generation_wall_time_ms = 0.0
        self.generation_call_count = 0
        self.generation_failed_call_count = 0
        self._errors = []
        self.synchronization_semantics = "disabled"

    @contextlib.contextmanager
    def time_generation_call(self, device: Optional[Any] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        is_cuda = str(getattr(device, "type", None)) == "cuda"
        cuda_module = None
        if is_cuda:
            try:
                import torch  # type: ignore

                cuda_module = torch.cuda
                synchronize = getattr(cuda_module, "synchronize", None)
                if callable(synchronize):
                    synchronize()
                self.synchronization_semantics = "sync_before_and_after_generate"
            except Exception as exc:
                self._errors.append("pre_generate_synchronize_failed:{}".format(exc))
                cuda_module = None
        else:
            self.synchronization_semantics = "host_wall_clock"
        start = time.perf_counter()
        try:
            yield
        except Exception:
            # A failed generate() call must never contribute a successful
            # duration to the denominator.
            self.generation_failed_call_count += 1
            raise
        else:
            if cuda_module is not None:
                try:
                    cuda_module.synchronize()
                except Exception as exc:
                    self._errors.append("post_generate_synchronize_failed:{}".format(exc))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.generation_wall_time_ms += elapsed_ms
            self.generation_call_count += 1

    def to_summary(self) -> Dict[str, Any]:
        valid = self.enabled and not self._errors
        return {
            "schema_version": 1,
            "generation_timing_enabled": self.enabled,
            "generation_timing_backend": self.timing_backend_requested,
            "generation_wall_time_ms": self.generation_wall_time_ms if self.enabled else None,
            "generation_call_count": self.generation_call_count,
            "generation_failed_call_count": self.generation_failed_call_count,
            "synchronization_semantics": self.synchronization_semantics,
            "generation_timing_valid": valid,
            "generation_timing_errors": list(self._errors),
            "measurement_scope": "gen_model.generate_call_only",
            "excludes": [
                "dataset_preprocessing",
                "tokenization",
                "data_loading",
                "rouge_computation",
                "prediction_decoding",
                "json_writing",
                "csv_generation",
                "archive_creation",
            ],
            "speed_claim_valid": False,
        }
