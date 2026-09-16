"""Exact catch-up overhead accounting for the early-exit missing deep-layer
K/V restoration paper.

This module answers one question: how much nominal early-exit computation is
later re-executed by exact catch-up in order to create missing deeper-layer
self-attention K/V?

It is a scalar-only per-transaction event recorder and cross-validated
aggregator. It is deliberately separate from, and never a replacement for,
``MissingKVRuntimeAccounting`` in ``missing_kv_runtime_accounting.py`` -- the
aggregate counters there remain authoritative; this module's per-event
records must reduce back to those same totals (see
``cross_validate_exact_catchup_overhead``).

Task C1 always executes exact catch-up before overwriting cache slices with
approximate restored K/V, so for every successful transaction recorded here:
    exact_catchup_executed_token_layer_units == exact_catchup_required_token_layer_units
    speed_claim_valid = False
Nothing in this module changes that. It measures the motivation for
eventually removing exact catch-up (Task C2); it does not implement Task C2
and it does not claim any speedup.

Never serialize hidden states, K/V tensors, logits, model parameters, or
unbounded raw identity arrays through this module -- only scalar/bounded
metadata.
"""

from __future__ import annotations

import math
import numbers
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from our_kv_restoration.missing_kv_dump_provenance import canonical_json_sha256, canonical_json_text


EVENT_SCHEMA_VERSION = 1
EVENT_RECORD_TYPE = "missing_kv_exact_catchup_overhead_event"
EVENT_IDENTITY_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
RUN_CONTEXT_IDENTITY_SCHEMA_VERSION = 2

# Canonical run_context.json content-identity fields (Blocker 2): binds a
# generated exact_catchup_summary.json to the exact run_context.json used for
# that run, so a full-mode summary can never be silently paired with a
# different run's run_context.json. Deliberately excludes filesystem paths --
# only the semantic policy/provenance content a run_context.json carries.
_RUN_CONTEXT_IDENTITY_FIELDS = (
    "schema_version",
    "mode",
    "run_instance_id",
    "run_started_at_utc",
    "run_finished_at_utc",
    "population_capped",
    "max_eval_samples",
    "candidate_policy_name",
    "candidate_layers",
    "threshold",
    "threshold_comparator",
    "adaptive_threshold",
    "paper_candidate_valid",
    "speed_claim_valid",
    "start_repository_commit",
    "start_git_branch",
    "start_tracked_worktree_clean",
    "end_repository_commit",
    "end_git_branch",
    "end_tracked_worktree_clean",
    "git_run_stability",
)


def compute_run_context_identity_sha256(run_context: Mapping[str, Any]) -> str:
    """Canonical content identity for a run_context.json payload (Blocker 2).

    Reuses ``canonical_json_sha256`` exactly as-is -- never a second
    canonicalization format -- over a fixed, explicit field subset, exactly
    like the existing event-population identity
    (``event_population_sha256``/``event_uid_population_sha256`` above). Both
    the runner (after delegated runtime success, before finalizer
    invocation) and the finalizer (to independently recompute and compare)
    call this same function, so the identity definition never drifts between
    the two call sites.
    """

    payload = {field: run_context.get(field) for field in _RUN_CONTEXT_IDENTITY_FIELDS}
    return canonical_json_sha256(payload)

RUNTIME_PATH_CANDIDATE_FIRST_CROSSING = "candidate_first_crossing"
RUNTIME_PATH_FIXED_SOURCE_PARALLEL_FLUSH = "fixed_source_parallel_flush"

TRANSACTION_STATUS_REGISTERED = "registered"
TRANSACTION_STATUS_COMMITTED = "committed"
TRANSACTION_STATUS_FAILED = "failed"

# Frozen gap-bin boundaries shared with the rest of the Phase 3c project,
# plus "same_layer" for gap == 0 (the first target layer in a catch-up run,
# i.e. target_layer == source_layer).
_GAP_BIN_BOUNDARIES = (
    ("1-2", 1, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9-12", 9, 12),
    ("13-16", 13, 16),
    ("17-999", 17, 999999),
)
GAP_BIN_LABELS = ("same_layer",) + tuple(label for label, _start, _end in _GAP_BIN_BOUNDARIES)


def gap_bin_for_gap(gap: int) -> str:
    gap = int(gap)
    if gap == 0:
        return "same_layer"
    if gap < 0:
        raise ValueError("exact_catchup_overhead_negative_gap: {}".format(gap))
    for label, start, end in _GAP_BIN_BOUNDARIES:
        if start <= gap <= end:
            return label
    return "17-999"


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def strict_nonnegative_json_number(value: Any) -> tuple[Optional[float], bool, str]:
    """Paper-facing JSON-number validation.

    Unlike ``_finite_or_none()``, this never coerces strings or booleans.
    It is intentionally used only at paper validation boundaries so legacy
    diagnostic aggregation can keep its tolerant behavior.
    """

    if value is None:
        return None, False, "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, False, "invalid_type"
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None, False, "invalid"
    return number, True, "ok"


def strict_nonnegative_json_int(value: Any) -> tuple[Optional[int], bool, str]:
    """Paper-facing nonnegative JSON-integer validation."""

    if value is None:
        return None, False, "missing"
    if isinstance(value, bool) or not isinstance(value, int):
        return None, False, "invalid_type"
    if value < 0:
        return None, False, "invalid"
    return value, True, "ok"


def _dedupe_preserving_order(items: Sequence[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        text = str(item)
        if text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


_EVENT_IDENTITY_FIELDS = (
    "event_identity_schema_version",
    "stable_sample_id",
    "generation_index",
    "decoder_position",
    "runtime_path",
    "source_layer",
    "candidate_policy_identity",
    "transaction_type",
)


def compute_event_uid(identity_fields: Mapping[str, Any]) -> str:
    """Build a stable ``event_uid`` from canonical semantic identity fields
    (never from filesystem paths or Python object ids).

    Fails closed: raises ``ValueError`` when a required identity field is
    missing, ``None``, or empty, since a paper-facing event without stable
    identity can never be safely deduplicated or cross-validated.
    """

    missing = [key for key in _EVENT_IDENTITY_FIELDS if identity_fields.get(key) in (None, "")]
    if missing:
        raise ValueError(
            "exact_catchup_overhead_missing_identity_fields: {}".format(sorted(missing))
        )
    canonical = {key: identity_fields[key] for key in _EVENT_IDENTITY_FIELDS}
    return canonical_json_sha256(canonical)


def _strict_position_int(value: Any) -> tuple[Optional[int], Optional[str]]:
    if isinstance(value, bool):
        return None, "invalid_position_type"
    if not isinstance(value, numbers.Integral):
        return None, "invalid_position_type"
    position = int(value)
    if position < 0:
        return None, "negative_position"
    return position, None


def _self_attention_cache_seq_len_for_position_resolution(layer_cache: Any) -> tuple[Optional[int], str]:
    if layer_cache is None:
        return None, "absent"
    try:
        if len(layer_cache) == 0:
            return None, "malformed_cache_object"
        key_states = layer_cache[0]
    except (TypeError, IndexError):
        return None, "malformed_cache_object"
    if key_states is None:
        return None, "malformed_cache_object"
    shape = getattr(key_states, "shape", None)
    if shape is None or len(shape) < 3:
        return None, "malformed_cache_object"
    try:
        seq_len = int(shape[2])
    except (TypeError, ValueError):
        return None, "malformed_cache_object"
    if seq_len < 0:
        return None, "malformed_cache_object"
    return seq_len, "ok"


def _target_past_cache_position_rows(
    *,
    past_key_values: Any,
    target_layers: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    if past_key_values is None:
        for target_layer in target_layers:
            rows.append(
                {
                    "target_layer": int(target_layer),
                    "past_cache_status": "absent",
                    "past_key_value_len": None,
                    "cache_position": 0,
                    "semantically_empty": True,
                }
            )
        return rows, errors
    try:
        past_count = len(past_key_values)
    except TypeError:
        return rows, ["malformed_past_key_values"]
    for target_layer in target_layers:
        target_layer = int(target_layer)
        if target_layer < 0 or target_layer >= past_count:
            rows.append(
                {
                    "target_layer": target_layer,
                    "past_cache_status": "missing",
                    "past_key_value_len": None,
                    "cache_position": None,
                    "semantically_empty": False,
                }
            )
            errors.append("missing_cache_position")
            continue
        seq_len, status = _self_attention_cache_seq_len_for_position_resolution(past_key_values[target_layer])
        if status == "absent":
            rows.append(
                {
                    "target_layer": target_layer,
                    "past_cache_status": status,
                    "past_key_value_len": None,
                    "cache_position": 0,
                    "semantically_empty": True,
                }
            )
        elif status == "ok":
            rows.append(
                {
                    "target_layer": target_layer,
                    "past_cache_status": status,
                    "past_key_value_len": seq_len,
                    "cache_position": seq_len,
                    "semantically_empty": seq_len == 0,
                }
            )
        else:
            rows.append(
                {
                    "target_layer": target_layer,
                    "past_cache_status": status,
                    "past_key_value_len": None,
                    "cache_position": None,
                    "semantically_empty": False,
                }
            )
            errors.append(status)
    return rows, _dedupe_preserving_order(errors)


def resolve_exact_catchup_event_decoder_position(
    *,
    inferred_decoder_position: Any,
    past_key_values: Any,
    target_layers: Sequence[int],
    first_token_context: bool,
    restoration_cache_positions: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Resolve the decoder position used in exact-catchup event identity.

    The first decoder token is the only case where the normal past-cache
    inference may legitimately return ``None``.  Recovery to position 0 is
    allowed only when the target-layer past cache is semantically empty, the
    transaction's authoritative cache positions are all exactly 0, and the
    caller independently confirms first-token runtime context.
    """

    target_layers = [int(layer) for layer in (target_layers or [])]
    cache_rows, cache_errors = _target_past_cache_position_rows(
        past_key_values=past_key_values,
        target_layers=target_layers,
    )
    if restoration_cache_positions is None:
        restoration_positions = []
        cache_errors.append("restoration_cache_positions_missing")
    else:
        restoration_positions = []
        for value in restoration_cache_positions:
            position, reason = _strict_position_int(value)
            restoration_positions.append(position)
            if reason is not None:
                cache_errors.append("invalid_cache_position")
        if len(restoration_positions) != len(target_layers):
            cache_errors.append("cache_position_count_mismatch")
    cache_errors = _dedupe_preserving_order(cache_errors)

    result = {
        "decoder_position": None,
        "decoder_position_source": "unresolved",
        "status": "failed",
        "reason": None,
        "inferred_decoder_position": inferred_decoder_position,
        "first_token_context": bool(first_token_context),
        "target_layers": list(target_layers),
        "restoration_cache_positions": list(restoration_positions),
        "past_cache_positions": [row.get("cache_position") for row in cache_rows],
        "past_cache_statuses": [
            {
                "target_layer": row.get("target_layer"),
                "past_cache_status": row.get("past_cache_status"),
                "past_key_value_len": row.get("past_key_value_len"),
                "semantically_empty": row.get("semantically_empty"),
            }
            for row in cache_rows
        ],
    }

    if not target_layers:
        result["reason"] = "target_layers_missing"
        return result
    if inferred_decoder_position is not None:
        position, reason = _strict_position_int(inferred_decoder_position)
        if reason is not None:
            result["reason"] = "explicit_decoder_position_{}".format(reason)
            return result
        if cache_errors:
            result["reason"] = cache_errors[0]
            return result
        comparable_positions = [pos for pos in restoration_positions if pos is not None]
        if comparable_positions and any(pos != position for pos in comparable_positions):
            result["reason"] = "explicit_position_conflicts_with_cache_position"
            return result
        result.update(
            {
                "decoder_position": position,
                "decoder_position_source": "inferred_decoder_position",
                "status": "ok",
                "reason": None,
            }
        )
        return result

    if cache_errors:
        result["reason"] = cache_errors[0]
        return result
    if any(position is None for position in restoration_positions):
        result["reason"] = "missing_cache_position"
        return result
    unique_positions = sorted(set(int(position) for position in restoration_positions))
    if len(unique_positions) != 1:
        result["reason"] = "mixed_cache_positions"
        return result
    only_position = unique_positions[0]
    if any(row.get("semantically_empty") is not True for row in cache_rows):
        result["reason"] = "nonempty_past_cache"
        return result
    if only_position != 0:
        result["reason"] = "nonzero_first_token_cache_position"
        return result
    if not bool(first_token_context):
        result["reason"] = "first_token_context_not_confirmed"
        return result
    result.update(
        {
            "decoder_position": 0,
            "decoder_position_source": "first_token_empty_cache_resolution",
            "status": "ok",
            "reason": None,
        }
    )
    return result


@dataclass
class ExactCatchupOverheadEvent:
    """Builds one exact catch-up overhead transaction event.

    Lifecycle: constructed (required units known immediately) ->
    ``record_target_executed`` zero or more times -> exactly one of
    ``commit()``/``fail()``. Calling any finalizing method twice, or in the
    wrong order, raises -- this is deliberate: silent double-finalization is
    exactly the bug this module exists to prevent.
    """

    runtime_path: str
    transaction_type: str
    source_layer: int
    decoder_layer_count: int
    stable_sample_id: Optional[str] = None
    selected_order: Optional[int] = None
    raw_dataset_index: Optional[int] = None
    generation_index: Optional[int] = None
    decoder_position: Optional[int] = None
    pending_skipped_token_count: int = 1
    candidate_policy_name: Optional[str] = None
    candidate_layers: Optional[Sequence[int]] = None
    threshold: Optional[float] = None
    threshold_comparator: Optional[str] = None
    adaptive_threshold: Optional[bool] = None
    full_depth_fallback: bool = False

    def __post_init__(self) -> None:
        self.source_layer = int(self.source_layer)
        self.decoder_layer_count = int(self.decoder_layer_count)
        self.pending_skipped_token_count = int(self.pending_skipped_token_count)
        self.last_exact_kv_layer = self.source_layer - 1
        self.first_missing_target_layer = self.source_layer
        self.last_missing_target_layer = self.decoder_layer_count - 1
        self.required_target_layer_count = max(0, self.decoder_layer_count - self.source_layer)
        self.nominal_skipped_token_layer_units = (
            self.required_target_layer_count * self.pending_skipped_token_count
        )
        self.exact_catchup_required_token_layer_units = self.nominal_skipped_token_layer_units
        self.exact_catchup_executed_token_layer_units = 0
        self.per_target_timing: List[Dict[str, Any]] = []
        self.transaction_status = TRANSACTION_STATUS_REGISTERED
        self.failure_stage: Optional[str] = None
        self.failure_reason: Optional[str] = None
        self._committed = False
        self._failed = False
        self.event_uid: Optional[str] = None

    def record_target_executed(
        self,
        *,
        target_layer: int,
        elapsed_ms: Optional[float] = None,
        timing_backend: Optional[str] = None,
        executed_pending_token_layer_units: Optional[int] = None,
    ) -> None:
        """Immediate-value API: use when the elapsed time is already known
        at the call site (CPU timing, or a diagnostic non-paper stratum).
        For CUDA deferred timing, use ``begin_target_timing`` /
        ``commit_target_executed`` instead -- see their docstrings."""

        if self._committed or self._failed:
            raise ValueError("exact_catchup_overhead_event_already_finalized")
        units = (
            self.pending_skipped_token_count
            if executed_pending_token_layer_units is None
            else int(executed_pending_token_layer_units)
        )
        gap = int(target_layer) - self.source_layer
        self.exact_catchup_executed_token_layer_units += units
        self.per_target_timing.append(
            {
                "target_layer": int(target_layer),
                "gap": gap,
                "gap_bin": gap_bin_for_gap(gap),
                "executed_pending_token_layer_units": units,
                "elapsed_ms": _finite_or_none(elapsed_ms),
                "timing_backend": timing_backend,
                "timing_resolved": True,
            }
        )

    def begin_target_timing(
        self,
        *,
        target_layer: int,
        executed_pending_token_layer_units: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reserve a per-target timing handle *before* the target-layer call
        runs, so it can be passed as (or wrapped by) a component timer's
        ``on_resolved`` callback. The handle is a plain mutable dict that
        ``resolve_target_timing`` mutates in place -- it is intentionally
        NOT yet appended to ``per_target_timing``: only
        ``commit_target_executed`` (called once the target-layer call has
        actually returned successfully) appends it and increments executed
        units. If the call fails, the caller simply never commits the
        handle, so no timing is ever fabricated for a target that did not
        complete.
        """

        if self._committed or self._failed:
            raise ValueError("exact_catchup_overhead_event_already_finalized")
        gap = int(target_layer) - self.source_layer
        return {
            "target_layer": int(target_layer),
            "gap": gap,
            "gap_bin": gap_bin_for_gap(gap),
            "executed_pending_token_layer_units": (
                self.pending_skipped_token_count
                if executed_pending_token_layer_units is None
                else int(executed_pending_token_layer_units)
            ),
            "elapsed_ms": None,
            "timing_backend": None,
            "timing_resolved": False,
        }

    @staticmethod
    def resolve_target_timing(handle: Dict[str, Any], *, elapsed_ms: Optional[float], timing_backend: Optional[str]) -> None:
        """Mutate a handle from ``begin_target_timing`` once its deferred
        timing resolves -- immediately for CPU timing, or later (at
        component-timer finalization) for deferred CUDA-event timing. Safe
        to call whether or not the handle has since been committed."""

        handle["elapsed_ms"] = _finite_or_none(elapsed_ms)
        handle["timing_backend"] = timing_backend
        handle["timing_resolved"] = True

    def commit_target_executed(self, handle: Dict[str, Any]) -> None:
        """Call once the target-layer call represented by ``handle`` has
        actually returned successfully: appends the handle to
        ``per_target_timing`` (whether or not its deferred timing has
        resolved yet -- ``resolve_target_timing`` may still fire later) and
        increments executed units. Never call this for a target whose call
        raised -- that is exactly the case ``begin_target_timing`` exists to
        keep un-fabricated."""

        if self._committed or self._failed:
            raise ValueError("exact_catchup_overhead_event_already_finalized")
        self.exact_catchup_executed_token_layer_units += handle["executed_pending_token_layer_units"]
        self.per_target_timing.append(handle)

    @property
    def timing_fully_resolved(self) -> bool:
        return all(bool(row.get("timing_resolved")) for row in self.per_target_timing)

    def commit(self) -> None:
        if self._failed:
            raise ValueError("exact_catchup_overhead_event_already_failed")
        if self._committed:
            raise ValueError("exact_catchup_overhead_event_already_committed")
        if self.exact_catchup_executed_token_layer_units != self.exact_catchup_required_token_layer_units:
            raise ValueError(
                "exact_catchup_overhead_commit_requires_executed_equals_required: "
                "executed={} required={}".format(
                    self.exact_catchup_executed_token_layer_units,
                    self.exact_catchup_required_token_layer_units,
                )
            )
        self._committed = True
        self.transaction_status = TRANSACTION_STATUS_COMMITTED
        # exact_catchup_required_token_layer_units is the immutable planned
        # value computed at construction (Section F) -- it is never
        # overwritten here, even though it now (by the check above) equals
        # the executed value for every successful commit.

    def fail(self, *, stage: str, reason: Any) -> None:
        if self._committed:
            raise ValueError("exact_catchup_overhead_event_already_committed")
        if self._failed:
            raise ValueError("exact_catchup_overhead_event_already_failed")
        self._failed = True
        self.transaction_status = TRANSACTION_STATUS_FAILED
        self.failure_stage = str(stage)
        self.failure_reason = str(reason)
        # required/executed are preserved exactly as accumulated so far --
        # a failed transaction may legitimately have executed < required.

    @property
    def net_avoided_token_layer_units(self) -> int:
        return self.nominal_skipped_token_layer_units - self.exact_catchup_executed_token_layer_units

    def identity_fields(self) -> Dict[str, Any]:
        return {
            "event_identity_schema_version": EVENT_IDENTITY_SCHEMA_VERSION,
            "stable_sample_id": self.stable_sample_id,
            "generation_index": self.generation_index,
            "decoder_position": self.decoder_position,
            "runtime_path": self.runtime_path,
            "source_layer": self.source_layer,
            "candidate_policy_identity": self.candidate_policy_name,
            "transaction_type": self.transaction_type,
        }

    def finalize_uid(self) -> str:
        if self.event_uid is None:
            self.event_uid = compute_event_uid(self.identity_fields())
        return self.event_uid

    def to_dict(self) -> Dict[str, Any]:
        if self.transaction_status not in (TRANSACTION_STATUS_COMMITTED, TRANSACTION_STATUS_FAILED):
            raise ValueError(
                "exact_catchup_overhead_event_not_finalized: status={}".format(self.transaction_status)
            )
        self.finalize_uid()
        timed = [row["elapsed_ms"] for row in self.per_target_timing if row["elapsed_ms"] is not None]
        exact_catchup_elapsed_ms = sum(timed) if timed else None
        backends = {row["timing_backend"] for row in self.per_target_timing if row["timing_backend"]}
        if len(backends) == 1:
            timing_backend = next(iter(backends))
        elif len(backends) > 1:
            timing_backend = "mixed"
        else:
            timing_backend = None
        timing_available = bool(self.per_target_timing) and len(timed) == len(self.per_target_timing)
        timing_fully_resolved = self.timing_fully_resolved
        # Paper measurement state is a run-level property (requires
        # generation timing, component-timer validity, and cross-validation
        # across the whole event population), so this event-level field must
        # never assert it directly. The aggregate computes paper measurement
        # state; the finalizer is the sole authoritative location for
        # paper_candidate_valid.
        if self.runtime_path == RUNTIME_PATH_FIXED_SOURCE_PARALLEL_FLUSH:
            paper_timing_eligibility = "ineligible_fixed_source_mixed_timing"
        elif self.transaction_status != TRANSACTION_STATUS_COMMITTED:
            paper_timing_eligibility = "ineligible_failed_transaction"
        else:
            paper_timing_eligibility = "requires_run_level_validation"
        return {
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_uid": self.event_uid,
            "stable_sample_id": self.stable_sample_id,
            "selected_order": self.selected_order,
            "raw_dataset_index": self.raw_dataset_index,
            "generation_index": self.generation_index,
            "decoder_position": self.decoder_position,
            "runtime_path": self.runtime_path,
            "transaction_type": self.transaction_type,
            "source_layer": self.source_layer,
            "last_exact_kv_layer": self.last_exact_kv_layer,
            "first_missing_target_layer": self.first_missing_target_layer,
            "last_missing_target_layer": self.last_missing_target_layer,
            "decoder_layer_count": self.decoder_layer_count,
            "required_target_layer_count": self.required_target_layer_count,
            "pending_skipped_token_count": self.pending_skipped_token_count,
            "nominal_skipped_token_layer_units": self.nominal_skipped_token_layer_units,
            "exact_catchup_required_token_layer_units": self.exact_catchup_required_token_layer_units,
            "exact_catchup_executed_token_layer_units": self.exact_catchup_executed_token_layer_units,
            "net_avoided_token_layer_units": self.net_avoided_token_layer_units,
            # Shallow-copy each row: this dict is meant to be an immutable
            # snapshot. The live handles (shared with any not-yet-fired
            # on_resolved callback) must never be able to silently mutate an
            # already-serialized event after the fact.
            "per_target_timing": [dict(row) for row in self.per_target_timing],
            "exact_catchup_elapsed_ms": exact_catchup_elapsed_ms,
            "timing_backend": timing_backend,
            "timing_available": timing_available,
            "timing_fully_resolved": timing_fully_resolved,
            "candidate_policy_name": self.candidate_policy_name,
            "candidate_layers": list(self.candidate_layers) if self.candidate_layers is not None else None,
            "threshold": self.threshold,
            "threshold_comparator": self.threshold_comparator,
            "adaptive_threshold": self.adaptive_threshold,
            "transaction_status": self.transaction_status,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "full_depth_fallback": bool(self.full_depth_fallback),
            "transaction_committed": self.transaction_status == TRANSACTION_STATUS_COMMITTED,
            "semantic_transaction_valid": self.transaction_status == TRANSACTION_STATUS_COMMITTED,
            "paper_timing_eligibility": paper_timing_eligibility,
            # The finalizer is the sole authoritative location for
            # paper_candidate_valid. The aggregate only reports
            # paper_measurement_valid.
            "paper_candidate_valid": False,
        }


class ExactCatchupOverheadRecorder:
    """Owns event-uid lifecycle protection and the finalized-event
    collection for one evaluation/generation run.

    Enforces: the same ``event_uid`` must not be committed twice; a failed
    UID must not later be committed; a committed UID must not later be
    failed; any duplicate registration/finalization invalidates the run and
    never mutates aggregate counters a second time. Duplicate diagnostics are
    bounded -- never an unbounded raw duplicate-event list.

    Semantic transaction completion (``finalize``) is deliberately separate
    from timing resolution and serialization: ``finalize`` marks the
    commit/fail lifecycle immediately but holds the *live* event object
    (whose per-target timing handles may still be waiting on a deferred CUDA
    callback). Only ``resolve_and_serialize_pending()`` -- which callers must
    invoke after the component timer has been finalized -- converts pending
    live events into their immutable ``to_dict()`` snapshot and appends them
    to ``events``. This is what guarantees a committed transaction is never
    converted to an immutable dictionary before its deferred CUDA callbacks
    have populated the target timings.
    """

    _MAX_DUPLICATE_EXAMPLES = 20

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._pending_events: List[ExactCatchupOverheadEvent] = []
        self._registered_uids = set()
        self._committed_uids = set()
        self._failed_uids = set()
        self.duplicate_registration_count = 0
        self.duplicate_finalization_count = 0
        self.missing_identity_skip_count = 0
        self._missing_identity_reason_examples: List[str] = []
        self.invalid_run = False
        self.invalid_reasons: List[str] = []
        self._duplicate_uid_examples: List[str] = []

    def reset(self) -> None:
        self.__init__()

    def _record_duplicate_example(self, uid: str) -> None:
        if len(self._duplicate_uid_examples) < self._MAX_DUPLICATE_EXAMPLES:
            self._duplicate_uid_examples.append(uid)

    def record_missing_identity_skip(self, *, reason: str = "stable_sample_id_unavailable") -> None:
        """Call when overhead recording is enabled but a transaction's
        stable identity is unavailable, so no event could be safely
        constructed. This must never be a silent no-op: it marks the run
        invalid and keeps a bounded count/example so paper-strict full mode
        fails closed instead of reporting a clean run with a quietly
        incomplete event population."""

        self.missing_identity_skip_count += 1
        self.invalid_run = True
        if len(self._missing_identity_reason_examples) < self._MAX_DUPLICATE_EXAMPLES:
            self._missing_identity_reason_examples.append(str(reason))

    def register(self, event: ExactCatchupOverheadEvent) -> str:
        uid = event.finalize_uid()
        if uid in self._registered_uids:
            self.duplicate_registration_count += 1
            self._record_duplicate_example(uid)
            self.invalid_run = True
            self.invalid_reasons.append("duplicate_registration:{}".format(uid))
        else:
            self._registered_uids.add(uid)
        return uid

    def finalize(self, event: ExactCatchupOverheadEvent) -> None:
        """Mark semantic transaction completion (commit/fail) immediately.
        Does NOT serialize yet: the live event is held in a pending list
        until ``resolve_and_serialize_pending()`` is called, which must only
        happen after any deferred CUDA timing has been resolved."""

        uid = event.finalize_uid()
        if event.transaction_status == TRANSACTION_STATUS_COMMITTED:
            if uid in self._failed_uids:
                self.invalid_run = True
                self.invalid_reasons.append("committed_after_failed:{}".format(uid))
                self.duplicate_finalization_count += 1
                self._record_duplicate_example(uid)
                return
            if uid in self._committed_uids:
                self.invalid_run = True
                self.invalid_reasons.append("duplicate_commit:{}".format(uid))
                self.duplicate_finalization_count += 1
                self._record_duplicate_example(uid)
                return
            self._committed_uids.add(uid)
        elif event.transaction_status == TRANSACTION_STATUS_FAILED:
            if uid in self._committed_uids:
                self.invalid_run = True
                self.invalid_reasons.append("failed_after_committed:{}".format(uid))
                self.duplicate_finalization_count += 1
                self._record_duplicate_example(uid)
                return
            if uid in self._failed_uids:
                self.invalid_run = True
                self.invalid_reasons.append("duplicate_failure:{}".format(uid))
                self.duplicate_finalization_count += 1
                self._record_duplicate_example(uid)
                return
            self._failed_uids.add(uid)
        else:
            raise ValueError(
                "exact_catchup_overhead_event_not_finalized: status={}".format(event.transaction_status)
            )
        self._pending_events.append(event)

    def resolve_and_serialize_pending(self) -> int:
        """Convert every pending live event into its immutable ``to_dict()``
        snapshot and append it to ``events``. Callers must invoke this only
        after the component timer's deferred CUDA events have been resolved
        (i.e. after ``MissingKVComponentTimer.finalize()``). Idempotent: a
        second call with nothing newly pending is a no-op. Returns the
        number of events serialized by this call."""

        pending = self._pending_events
        self._pending_events = []
        for event in pending:
            self.events.append(event.to_dict())
        return len(pending)

    @property
    def pending_event_count(self) -> int:
        return len(self._pending_events)

    def duplicate_diagnostics(self) -> Dict[str, Any]:
        """Side-effect-free: safe to call at any point, including mid-run.

        ``incomplete_registered_event_count`` covers transactions that were
        registered (required units known) but never reached commit()/fail()
        -- e.g. an exception propagated out of a flush loop that has no
        wrapping try/except of its own. This never fabricates a completion
        the transaction never reached; it reports the population as invalid
        instead.
        """

        incomplete_uids = self._registered_uids - self._committed_uids - self._failed_uids
        invalid_reasons = list(self.invalid_reasons)
        if incomplete_uids:
            invalid_reasons.append(
                "incomplete_registered_events:{}".format(len(incomplete_uids))
            )
        if self.missing_identity_skip_count:
            invalid_reasons.append(
                "missing_identity_skips:{}".format(self.missing_identity_skip_count)
            )
        return {
            "registered_event_count": len(self._registered_uids),
            "committed_event_count": len(self._committed_uids),
            "failed_event_count": len(self._failed_uids),
            "incomplete_registered_event_count": len(incomplete_uids),
            "pending_unserialized_event_count": len(self._pending_events),
            "duplicate_registration_count": self.duplicate_registration_count,
            "duplicate_finalization_count": self.duplicate_finalization_count,
            "duplicate_uid_examples": list(self._duplicate_uid_examples),
            "missing_identity_skip_count": self.missing_identity_skip_count,
            "missing_identity_reason_examples": list(self._missing_identity_reason_examples),
            "invalid_run": self.invalid_run or bool(incomplete_uids),
            "invalid_reasons": invalid_reasons,
        }


def _new_event_row() -> Dict[str, Any]:
    return {
        "event_count": 0,
        "successful_event_count": 0,
        "failed_event_count": 0,
        "full_depth_fallback_count": 0,
        "pending_skipped_token_count": 0,
        "nominal_skipped_token_layer_units": 0,
        "exact_catchup_required_token_layer_units": 0,
        "exact_catchup_executed_token_layer_units": 0,
        "net_avoided_token_layer_units": 0,
        "exact_catchup_elapsed_ms": 0.0,
        "_elapsed_event_count": 0,
    }


def _update_event_row(row: Dict[str, Any], event: Mapping[str, Any]) -> None:
    row["event_count"] += 1
    if event["transaction_status"] == TRANSACTION_STATUS_COMMITTED:
        row["successful_event_count"] += 1
    elif event["transaction_status"] == TRANSACTION_STATUS_FAILED:
        row["failed_event_count"] += 1
    if event.get("full_depth_fallback"):
        row["full_depth_fallback_count"] += 1
    row["pending_skipped_token_count"] += int(event["pending_skipped_token_count"])
    row["nominal_skipped_token_layer_units"] += int(event["nominal_skipped_token_layer_units"])
    row["exact_catchup_required_token_layer_units"] += int(event["exact_catchup_required_token_layer_units"])
    row["exact_catchup_executed_token_layer_units"] += int(event["exact_catchup_executed_token_layer_units"])
    row["net_avoided_token_layer_units"] += int(event["net_avoided_token_layer_units"])
    elapsed = _finite_or_none(event.get("exact_catchup_elapsed_ms"))
    if elapsed is not None:
        row["exact_catchup_elapsed_ms"] += elapsed
        row["_elapsed_event_count"] += 1


def _finalize_event_row(row: Dict[str, Any], *, generation_wall_time_ms: Optional[float] = None) -> Dict[str, Any]:
    elapsed_count = row.pop("_elapsed_event_count")
    executed = row["exact_catchup_executed_token_layer_units"]
    row["mean_elapsed_ms_per_event"] = (
        row["exact_catchup_elapsed_ms"] / elapsed_count if elapsed_count else None
    )
    row["mean_elapsed_ms_per_executed_unit"] = (
        row["exact_catchup_elapsed_ms"] / executed if executed and elapsed_count else None
    )
    if row["event_count"] == 0:
        row["timing_availability"] = "none"
    elif elapsed_count == row["event_count"]:
        row["timing_availability"] = "complete"
    elif elapsed_count > 0:
        row["timing_availability"] = "partial"
    else:
        row["timing_availability"] = "none"
    if generation_wall_time_ms is not None and generation_wall_time_ms > 0:
        row["catchup_share_of_generation_time"] = row["exact_catchup_elapsed_ms"] / generation_wall_time_ms
    else:
        row["catchup_share_of_generation_time"] = None
    return row


def _new_gap_row() -> Dict[str, Any]:
    return {
        "target_layer_occurrence_count": 0,
        "executed_pending_token_layer_units": 0,
        "exact_catchup_elapsed_ms": 0.0,
        "_elapsed_count": 0,
    }


def _update_gap_row(row: Dict[str, Any], target: Mapping[str, Any]) -> None:
    row["target_layer_occurrence_count"] += 1
    row["executed_pending_token_layer_units"] += int(target.get("executed_pending_token_layer_units") or 0)
    elapsed = _finite_or_none(target.get("elapsed_ms"))
    if elapsed is not None:
        row["exact_catchup_elapsed_ms"] += elapsed
        row["_elapsed_count"] += 1


def _finalize_gap_row(row: Dict[str, Any]) -> Dict[str, Any]:
    elapsed_count = row.pop("_elapsed_count")
    row["mean_elapsed_ms_per_target_layer_occurrence"] = (
        row["exact_catchup_elapsed_ms"] / elapsed_count if elapsed_count else None
    )
    row["timing_availability"] = (
        "complete"
        if row["target_layer_occurrence_count"] and elapsed_count == row["target_layer_occurrence_count"]
        else ("partial" if elapsed_count > 0 else "none")
    )
    return row


PAPER_TIMING_STATUS_OK = "ok"
PAPER_TIMING_STATUS_NO_CANDIDATE_EVENTS = "structurally_valid_statistically_unusable"
PAPER_TIMING_STATUS_INVALID = "invalid"


def recompute_candidate_timing_facts(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recompute paper-candidate timing facts directly from event rows.

    This is deliberately smaller than the full aggregate: it only derives the
    candidate-first-crossing committed-event timing facts that a finalizer must
    independently verify from the JSONL event population. Fixed-source timing
    and failed candidate transactions are excluded by definition.

    The authoritative paper elapsed total is recomputed from resolved
    per-target timing rows. The cached event-level
    ``exact_catchup_elapsed_ms`` is checked only for consistency with those
    rows, so a tampered event total cannot become authoritative.
    """

    committed_candidate_events = [
        event
        for event in events
        if event.get("runtime_path") == RUNTIME_PATH_CANDIDATE_FIRST_CROSSING
        and event.get("transaction_status") == TRANSACTION_STATUS_COMMITTED
    ]
    candidate_event_count = len(committed_candidate_events)
    candidate_target_occurrence_count = 0
    candidate_timed_target_occurrence_count = 0
    candidate_timed_event_count = 0
    backend_set = set()
    paper_catchup_elapsed_from_targets = 0.0
    timing_errors: List[str] = []

    def _is_nonnegative_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    for event in committed_candidate_events:
        targets_raw = event.get("per_target_timing")
        if targets_raw is None:
            targets = []
        elif isinstance(targets_raw, list):
            targets = targets_raw
        else:
            targets = []
            timing_errors.append("candidate_target_timing_invalid_type")

        required_target_count = event.get("required_target_layer_count")
        if _is_nonnegative_int(required_target_count) and len(targets) != required_target_count:
            timing_errors.append("candidate_target_occurrence_count_event_mismatch")

        event_fully_timed = bool(targets) and event.get("timing_fully_resolved") is True
        event_elapsed_sum_from_targets = 0.0
        saw_resolved_target = False
        for target in targets:
            candidate_target_occurrence_count += 1
            if not isinstance(target, Mapping):
                timing_errors.append("candidate_target_timing_invalid_type")
                event_fully_timed = False
                continue

            timing_resolved = target.get("timing_resolved")
            if not isinstance(timing_resolved, bool):
                timing_errors.append("candidate_target_timing_resolved_invalid_type")
                event_fully_timed = False

            backend = target.get("timing_backend")
            if backend is not None and not isinstance(backend, str):
                timing_errors.append("candidate_target_timing_backend_invalid_type")
                event_fully_timed = False
            elif backend is not None:
                backend_set.add(backend)

            elapsed, elapsed_valid, elapsed_error = strict_nonnegative_json_number(target.get("elapsed_ms"))
            if timing_resolved is True and not elapsed_valid:
                timing_errors.append(
                    "candidate_target_elapsed_invalid_type"
                    if elapsed_error in ("missing", "invalid_type")
                    else "candidate_target_elapsed_invalid"
                )
                event_fully_timed = False
            elif timing_resolved is True:
                candidate_timed_target_occurrence_count += 1
                saw_resolved_target = True
                event_elapsed_sum_from_targets += elapsed
            else:
                event_fully_timed = False

        if event_fully_timed:
            candidate_timed_event_count += 1
            paper_catchup_elapsed_from_targets += event_elapsed_sum_from_targets

        event_elapsed, event_elapsed_valid, event_elapsed_error = strict_nonnegative_json_number(
            event.get("exact_catchup_elapsed_ms")
        )
        if targets or saw_resolved_target:
            if not event_elapsed_valid:
                timing_errors.append(
                    "candidate_event_elapsed_invalid_type"
                    if event_elapsed_error in ("missing", "invalid_type")
                    else "candidate_event_elapsed_invalid"
                )
            elif abs(event_elapsed - event_elapsed_sum_from_targets) > 1e-6:
                timing_errors.append("candidate_event_elapsed_mismatch")

    candidate_timing_complete = (
        candidate_event_count > 0
        and candidate_timed_event_count == candidate_event_count
        and candidate_timed_target_occurrence_count == candidate_target_occurrence_count
    )
    candidate_timing_backend_set = sorted(backend_set)
    candidate_cuda_event_timing_valid = candidate_timing_complete and backend_set == {"cuda_events"}
    if candidate_event_count == 0 or not candidate_cuda_event_timing_valid:
        paper_catchup_elapsed_ms = None
    else:
        paper_catchup_elapsed_ms = paper_catchup_elapsed_from_targets
    timing_errors = _dedupe_preserving_order(timing_errors)
    return {
        "candidate_event_count": candidate_event_count,
        "candidate_timed_event_count": candidate_timed_event_count,
        "candidate_target_occurrence_count": candidate_target_occurrence_count,
        "candidate_timed_target_occurrence_count": candidate_timed_target_occurrence_count,
        "candidate_timing_complete": candidate_timing_complete,
        "candidate_timing_backend_set": candidate_timing_backend_set,
        "candidate_cuda_event_timing_valid": candidate_cuda_event_timing_valid,
        "paper_catchup_elapsed_ms": paper_catchup_elapsed_ms,
        "candidate_timing_errors": timing_errors,
    }


def _component_timing_state_from_accounting(accounting_summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    component_timing = (accounting_summary or {}).get("timing") or {}
    component_backends = component_timing.get("component_backends") or {}
    return {
        "component_timing_valid": component_timing.get("timing_valid") is True,
        "component_timing_errors": list(component_timing.get("timing_errors") or []),
        "component_unresolved_pending_event_count": component_timing.get("unresolved_pending_event_count"),
        "component_timing_backend_resolved": component_timing.get("timing_backend_resolved"),
        "exact_catchup_component_timing_backend": component_backends.get("exact_parallel_catchup_time_ms"),
    }


def _component_timing_state_from_paper_timing(paper_timing: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    paper_timing = paper_timing or {}
    return {
        "component_timing_valid": paper_timing.get("component_timing_valid") is True,
        "component_timing_errors": list(paper_timing.get("component_timing_errors") or []),
        "component_unresolved_pending_event_count": paper_timing.get("component_unresolved_pending_event_count"),
        "component_timing_backend_resolved": paper_timing.get("component_timing_backend_resolved"),
        "exact_catchup_component_timing_backend": paper_timing.get("exact_catchup_component_timing_backend"),
    }


def derive_paper_measurement_state(
    *,
    candidate_timing_facts: Mapping[str, Any],
    component_timing_state: Optional[Mapping[str, Any]] = None,
    accounting_summary: Optional[Mapping[str, Any]] = None,
    generation_timing_summary: Optional[Mapping[str, Any]],
    cross_validation: Mapping[str, Any],
    dup_diagnostics: Mapping[str, Any],
    generation_wall_time_ms: Optional[float],
    generation_timing_valid: Optional[bool],
    run_valid: bool,
    accounting_valid: bool,
    candidate_decision_accounting: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive authoritative paper measurement state from checked inputs.

    This is the shared pure gate used by both the aggregate and the
    finalizer. In particular, finalization never trusts the summary's coarse
    ``paper_measurement_status``/``paper_measurement_valid`` fields; it
    recomputes them through this helper from event-derived candidate timing
    facts plus the timing/accounting diagnostics already serialized by the
    real run.
    """

    candidate_event_count = int(candidate_timing_facts.get("candidate_event_count") or 0)
    candidate_timing_complete = candidate_timing_facts.get("candidate_timing_complete") is True
    candidate_timing_backend_set = list(candidate_timing_facts.get("candidate_timing_backend_set") or [])
    candidate_cuda_event_timing_valid = candidate_timing_facts.get("candidate_cuda_event_timing_valid") is True
    candidate_timing_errors = list(candidate_timing_facts.get("candidate_timing_errors") or [])
    paper_catchup_elapsed_ms = candidate_timing_facts.get("paper_catchup_elapsed_ms")

    if component_timing_state is None:
        component_timing_state = _component_timing_state_from_accounting(accounting_summary)
    component_timing_valid = component_timing_state.get("component_timing_valid") is True
    component_timing_errors = list(component_timing_state.get("component_timing_errors") or [])
    component_unresolved_pending_event_count_raw = component_timing_state.get(
        "component_unresolved_pending_event_count"
    )
    component_unresolved_pending_event_count, component_unresolved_count_valid, _component_unresolved_error = (
        strict_nonnegative_json_int(component_unresolved_pending_event_count_raw)
    )
    component_timing_backend_resolved = component_timing_state.get("component_timing_backend_resolved")
    exact_catchup_component_backend = component_timing_state.get("exact_catchup_component_timing_backend")
    exact_catchup_component_cuda_event_valid = exact_catchup_component_backend == "cuda_events"

    generation_wall_time_value, generation_wall_time_value_valid, generation_wall_time_error = (
        strict_nonnegative_json_number(generation_wall_time_ms)
    )
    valid_generation_denominator = (
        generation_wall_time_value_valid
        and generation_wall_time_value > 0
        and generation_timing_valid is True
    )
    paper_catchup_share_of_generation_time = (
        paper_catchup_elapsed_ms / generation_wall_time_value
        if paper_catchup_elapsed_ms is not None and valid_generation_denominator
        else None
    )

    component_timing_state_absent = (
        component_timing_valid is False
        and not component_timing_errors
        and component_timing_backend_resolved is None
        and exact_catchup_component_backend is None
    )
    component_unresolved_missing_allowed = (
        candidate_event_count == 0
        and _component_unresolved_error == "missing"
        and component_timing_state_absent
    )
    component_unresolved_schema_invalid = (
        not component_unresolved_count_valid and not component_unresolved_missing_allowed
    )

    component_errors: List[str] = []
    if not component_timing_valid:
        component_errors.append("component_timing_invalid")
    component_errors.extend(str(item) for item in component_timing_errors)
    if component_unresolved_schema_invalid:
        component_errors.append("component_unresolved_pending_event_count_invalid_type")
    elif component_unresolved_pending_event_count != 0:
        component_errors.append(
            "component_timing_has_unresolved_events:{}".format(component_unresolved_pending_event_count)
        )
    if exact_catchup_component_backend != "cuda_events":
        component_errors.append(
            "exact_catchup_component_backend_not_cuda_events:{}".format(exact_catchup_component_backend)
        )
    component_errors = _dedupe_preserving_order(component_errors)

    schema_errors: List[str] = []
    schema_errors.extend(str(item) for item in candidate_timing_errors)
    if component_unresolved_schema_invalid:
        schema_errors.append("component_unresolved_pending_event_count_invalid_type")
    if generation_wall_time_error == "invalid_type":
        schema_errors.append("generation_wall_time_invalid_type")
    elif generation_wall_time_error == "invalid":
        schema_errors.append("generation_wall_time_nonpositive_or_missing")
    elif generation_timing_valid is True and (
        not generation_wall_time_value_valid or generation_wall_time_value <= 0
    ):
        schema_errors.append("generation_wall_time_nonpositive_or_missing")
    schema_errors = _dedupe_preserving_order(schema_errors)

    errors: List[str] = []
    if schema_errors:
        errors.extend(schema_errors)
        if (
            candidate_event_count > 0
            and not candidate_cuda_event_timing_valid
            and candidate_timing_backend_set != ["cuda_events"]
        ):
            errors.append("candidate_timing_backend_not_cuda_events:{}".format(candidate_timing_backend_set))
        errors = _dedupe_preserving_order(errors)
        status = PAPER_TIMING_STATUS_INVALID
    elif candidate_event_count == 0:
        status = PAPER_TIMING_STATUS_NO_CANDIDATE_EVENTS
    elif not candidate_timing_complete:
        errors.append("candidate_timing_incomplete")
        status = PAPER_TIMING_STATUS_INVALID
    elif not candidate_cuda_event_timing_valid:
        errors.append("candidate_timing_backend_not_cuda_events:{}".format(candidate_timing_backend_set))
        status = PAPER_TIMING_STATUS_INVALID
    elif component_errors:
        errors.extend(component_errors)
        status = PAPER_TIMING_STATUS_INVALID
    elif generation_timing_summary is None or generation_timing_valid is not True:
        errors.append("generation_timing_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    elif generation_wall_time_error == "invalid_type":
        errors.append("generation_wall_time_invalid_type")
        status = PAPER_TIMING_STATUS_INVALID
    elif not valid_generation_denominator:
        errors.append("generation_wall_time_nonpositive_or_missing")
        status = PAPER_TIMING_STATUS_INVALID
    elif cross_validation.get("status") != "ok":
        errors.append("cross_validation_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    elif dup_diagnostics.get("invalid_run"):
        errors.append("event_population_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    elif not accounting_valid:
        errors.append("accounting_validation_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    elif (candidate_decision_accounting or {}).get("status") != "ok":
        errors.append("candidate_decision_accounting_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    elif not run_valid:
        errors.append("summary_run_invalid")
        status = PAPER_TIMING_STATUS_INVALID
    else:
        status = PAPER_TIMING_STATUS_OK

    paper_measurement_valid = status == PAPER_TIMING_STATUS_OK
    if paper_measurement_valid:
        paper_measurement_status = PAPER_TIMING_STATUS_OK
    elif status == PAPER_TIMING_STATUS_NO_CANDIDATE_EVENTS and run_valid:
        paper_measurement_status = PAPER_TIMING_STATUS_NO_CANDIDATE_EVENTS
    else:
        paper_measurement_status = PAPER_TIMING_STATUS_INVALID

    paper_measurement_errors = list(errors)
    if not run_valid and "summary_run_invalid" not in paper_measurement_errors:
        if not accounting_valid and "accounting_validation_invalid" not in paper_measurement_errors:
            paper_measurement_errors.append("accounting_validation_invalid")
        if (
            (candidate_decision_accounting or {}).get("status") != "ok"
            and "candidate_decision_accounting_invalid" not in paper_measurement_errors
        ):
            paper_measurement_errors.append("candidate_decision_accounting_invalid")
        if cross_validation.get("status") != "ok" and "cross_validation_invalid" not in paper_measurement_errors:
            paper_measurement_errors.append("cross_validation_invalid")
        if dup_diagnostics.get("invalid_run") and "event_population_invalid" not in paper_measurement_errors:
            paper_measurement_errors.append("event_population_invalid")
        paper_measurement_errors.append("summary_run_invalid")

    return {
        "paper_timing_status": status,
        "paper_timing_errors": errors,
        "component_timing_valid": component_timing_valid,
        "component_timing_errors": component_timing_errors,
        "component_unresolved_pending_event_count": component_unresolved_pending_event_count_raw,
        "component_timing_backend_resolved": component_timing_backend_resolved,
        "exact_catchup_component_timing_backend": exact_catchup_component_backend,
        "exact_catchup_component_cuda_event_valid": exact_catchup_component_cuda_event_valid,
        "fixed_source_mixed_timing_excluded_from_paper_numerator": True,
        "generation_timing_valid": generation_timing_valid,
        "generation_timing_backend": (
            generation_timing_summary.get("generation_timing_backend") if generation_timing_summary else None
        ),
        "paper_catchup_elapsed_ms": paper_catchup_elapsed_ms,
        "paper_catchup_share_of_generation_time": paper_catchup_share_of_generation_time,
        "summary_run_valid": bool(run_valid),
        "accounting_validation_valid": bool(accounting_valid),
        "candidate_decision_accounting_valid": (candidate_decision_accounting or {}).get("status") == "ok",
        "paper_measurement_valid": paper_measurement_valid,
        "paper_measurement_status": paper_measurement_status,
        "paper_measurement_errors": paper_measurement_errors,
    }


def _compute_paper_strict_timing(
    events: Sequence[Mapping[str, Any]],
    *,
    accounting_summary: Optional[Mapping[str, Any]],
    generation_timing_summary: Optional[Mapping[str, Any]],
    cross_validation: Mapping[str, Any],
    dup_diagnostics: Mapping[str, Any],
    generation_wall_time_ms: Optional[float],
    generation_timing_valid: Optional[bool],
    run_valid: bool,
    accounting_valid: bool,
    candidate_decision_accounting: Mapping[str, Any],
) -> Dict[str, Any]:
    """Paper-facing candidate-first-crossing timing gates (Section E).

    Deliberately restricted to the candidate-first-crossing runtime path --
    fixed-source diagnostic timing never contributes to
    ``paper_catchup_elapsed_ms`` (see
    ``fixed_source_mixed_timing_excluded_from_paper_numerator``, which is
    always True here precisely because this function only ever iterates
    ``RUNTIME_PATH_CANDIDATE_FIRST_CROSSING`` events).

    A population with zero candidate events is reported as
    ``PAPER_TIMING_STATUS_NO_CANDIDATE_EVENTS`` rather than fabricating a
    zero-valued latency or silently treating it the same as a genuine
    failure -- callers (the finalizer) decide, with full-run context, whether
    that is acceptable for a given mode.
    """

    timing_facts = recompute_candidate_timing_facts(events)
    measurement_state = derive_paper_measurement_state(
        candidate_timing_facts=timing_facts,
        accounting_summary=accounting_summary,
        generation_timing_summary=generation_timing_summary,
        cross_validation=cross_validation,
        dup_diagnostics=dup_diagnostics,
        generation_wall_time_ms=generation_wall_time_ms,
        generation_timing_valid=generation_timing_valid,
        run_valid=run_valid,
        accounting_valid=accounting_valid,
        candidate_decision_accounting=candidate_decision_accounting,
    )

    return {
        **measurement_state,
        "candidate_event_count": timing_facts["candidate_event_count"],
        "candidate_timed_event_count": timing_facts["candidate_timed_event_count"],
        "candidate_target_occurrence_count": timing_facts["candidate_target_occurrence_count"],
        "candidate_timed_target_occurrence_count": timing_facts["candidate_timed_target_occurrence_count"],
        "candidate_timing_complete": timing_facts["candidate_timing_complete"],
        "candidate_timing_backend_set": timing_facts["candidate_timing_backend_set"],
        "candidate_cuda_event_timing_valid": timing_facts["candidate_cuda_event_timing_valid"],
        "candidate_timing_errors": timing_facts["candidate_timing_errors"],
        # Kept for schema compatibility -- paper-candidate validity requires
        # full-mode/uncapped run-context information this function never
        # receives, so it is never authoritatively asserted here. The
        # finalizer (which loads run_context.json) is the sole authoritative
        # location for paper_candidate_valid.
        "paper_candidate_valid": False,
        "paper_candidate_validity_scope": "requires_finalizer_run_context_validation",
    }


def _compute_candidate_decision_accounting(
    events: Sequence[Mapping[str, Any]],
    *,
    accounting_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Section I: report CALM candidate first-crossing / full-depth-fallback
    counts from the authoritative ``MissingKVRuntimeAccounting`` summary --
    never by fabricating a zero-duration fallback event, since full-depth
    fallback legitimately creates no exact-catchup event at all."""

    first_crossing = None
    fallback = None
    if accounting_summary is not None:
        first_crossing = accounting_summary.get("calm_first_crossing_token_count")
        fallback = accounting_summary.get("calm_full_depth_fallback_token_count")
    total = None
    first_crossing_rate = None
    full_depth_fallback_rate = None
    if first_crossing is not None and fallback is not None:
        total = int(first_crossing) + int(fallback)
        if total > 0:
            first_crossing_rate = first_crossing / total
            full_depth_fallback_rate = fallback / total
    candidate_pending_sum = sum(
        int(event["pending_skipped_token_count"])
        for event in events
        if event.get("runtime_path") == RUNTIME_PATH_CANDIDATE_FIRST_CROSSING
        and event.get("transaction_status") == TRANSACTION_STATUS_COMMITTED
    )
    errors: List[str] = []
    if first_crossing is not None and candidate_pending_sum != int(first_crossing):
        errors.append("candidate_event_pending_token_sum_vs_first_crossing_mismatch")
    return {
        "first_crossing_token_count": first_crossing,
        "full_depth_fallback_token_count": fallback,
        "total_candidate_decision_token_count": total,
        "first_crossing_rate": first_crossing_rate,
        "full_depth_fallback_rate": full_depth_fallback_rate,
        "candidate_event_pending_token_sum": candidate_pending_sum,
        "errors": errors,
        "status": "ok" if not errors else "invalid",
    }


def cross_validate_exact_catchup_overhead(
    events: Sequence[Mapping[str, Any]],
    *,
    accounting_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-check event-level sums against each other and (when supplied)
    against MissingKVRuntimeAccounting's aggregate counters. Never silently
    intersects a mismatched population -- any mismatch is reported as an
    explicit error rather than dropped.
    """

    errors: List[str] = []
    sum_nominal = sum(int(event["nominal_skipped_token_layer_units"]) for event in events)
    sum_required = sum(int(event["exact_catchup_required_token_layer_units"]) for event in events)
    sum_executed = sum(int(event["exact_catchup_executed_token_layer_units"]) for event in events)
    per_target_executed_sum = 0
    per_target_elapsed_sum = 0.0
    event_elapsed_sum = 0.0
    for event in events:
        targets = event.get("per_target_timing") or []
        per_target_executed_sum += sum(int(t["executed_pending_token_layer_units"]) for t in targets)
        per_target_elapsed_sum += sum(_finite_or_none(t.get("elapsed_ms")) or 0.0 for t in targets)
        event_elapsed_sum += _finite_or_none(event.get("exact_catchup_elapsed_ms")) or 0.0
    if per_target_executed_sum != sum_executed:
        errors.append("per_target_executed_units_sum_mismatch")
    if abs(per_target_elapsed_sum - event_elapsed_sum) > 1e-6:
        errors.append("per_target_elapsed_time_sum_mismatch")
    if accounting_summary is not None:
        counters = accounting_summary.get("counters", {})
        if int(counters.get("nominal_skipped_token_layer_units", -1)) != sum_nominal:
            errors.append("accounting_nominal_units_mismatch")
        if int(counters.get("exact_catchup_required_token_layer_units", -1)) != sum_required:
            errors.append("accounting_required_units_mismatch")
        if int(counters.get("exact_catchup_executed_token_layer_units", -1)) != sum_executed:
            errors.append("accounting_executed_units_mismatch")
    return {
        "status": "ok" if not errors else "invalid",
        "errors": errors,
        "sum_nominal_skipped_token_layer_units": sum_nominal,
        "sum_exact_catchup_required_token_layer_units": sum_required,
        "sum_exact_catchup_executed_token_layer_units": sum_executed,
        "sum_per_target_executed_token_layer_units": per_target_executed_sum,
        "sum_per_target_elapsed_ms": per_target_elapsed_sum,
        "sum_event_exact_catchup_elapsed_ms": event_elapsed_sum,
    }


def aggregate_exact_catchup_overhead(
    events: Sequence[Mapping[str, Any]],
    *,
    accounting_summary: Optional[Mapping[str, Any]] = None,
    generation_timing_summary: Optional[Mapping[str, Any]] = None,
    duplicate_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build cross-validated overall / by-runtime-path / by-source-layer /
    by-gap aggregates for a finalized event population.

    ``by_gap`` rows aggregate at target-layer-occurrence granularity (one
    event can contribute many target-layer rows across many gap bins), which
    is intentionally a different unit than the event-level rows above it --
    see the ``sum(gap target rows) == sum(per-target totals)`` cross-check.
    """

    events = list(events)
    overall = _new_event_row()
    by_runtime_path: Dict[str, Any] = {}
    by_source_layer: Dict[str, Any] = {}
    by_gap: Dict[str, Any] = {}
    # Runtime-path-stratified breakdowns (Major 2): candidate_first_crossing
    # (single-token immediate exact-reference, deferred CUDA-event timing)
    # and fixed_source_parallel_flush (pending+current-token mixed parallel
    # execution, host diagnostic timing) have incompatible timing semantics
    # and must never be blended into one homogeneous per-key row. These reuse
    # the exact same row constructors/updaters/finalizers as the legacy mixed
    # tables below -- no aggregation math is duplicated.
    by_runtime_path_and_source_layer: Dict[str, Dict[str, Any]] = {}
    by_runtime_path_and_gap: Dict[str, Dict[str, Any]] = {}

    generation_wall_time_raw = None
    generation_wall_time_ms = None
    generation_timing_valid = None
    if generation_timing_summary is not None:
        generation_wall_time_raw = generation_timing_summary.get("generation_wall_time_ms")
        generation_wall_time_ms = _finite_or_none(generation_wall_time_raw)
        generation_timing_valid = bool(generation_timing_summary.get("generation_timing_valid"))

    for event in events:
        _update_event_row(overall, event)
        runtime_path_key = str(event["runtime_path"])
        path_row = by_runtime_path.setdefault(runtime_path_key, _new_event_row())
        _update_event_row(path_row, event)
        source_row = by_source_layer.setdefault(str(event["source_layer"]), _new_event_row())
        _update_event_row(source_row, event)
        stratified_source_row = by_runtime_path_and_source_layer.setdefault(runtime_path_key, {}).setdefault(
            str(event["source_layer"]), _new_event_row()
        )
        _update_event_row(stratified_source_row, event)
        stratified_gap_bucket = by_runtime_path_and_gap.setdefault(runtime_path_key, {})
        for target in event.get("per_target_timing") or []:
            gap_row = by_gap.setdefault(str(target["gap_bin"]), _new_gap_row())
            _update_gap_row(gap_row, target)
            stratified_gap_row = stratified_gap_bucket.setdefault(str(target["gap_bin"]), _new_gap_row())
            _update_gap_row(stratified_gap_row, target)

    valid_generation_denominator = (
        generation_wall_time_ms is not None and generation_wall_time_ms > 0 and generation_timing_valid is True
    )
    _finalize_event_row(
        overall,
        generation_wall_time_ms=generation_wall_time_ms if valid_generation_denominator else None,
    )
    for row in by_runtime_path.values():
        _finalize_event_row(row)
    for row in by_source_layer.values():
        _finalize_event_row(row)
    for row in by_gap.values():
        _finalize_gap_row(row)
    for bucket in by_runtime_path_and_source_layer.values():
        for row in bucket.values():
            _finalize_event_row(row)
    for bucket in by_runtime_path_and_gap.values():
        for row in bucket.values():
            _finalize_gap_row(row)

    # These legacy mixed tables (overall/by_source_layer/by_gap) are kept for
    # backward compatibility, but they merge candidate and fixed-source
    # timing populations -- explicitly marked diagnostic-only so they are
    # never mistaken for a paper-eligible table. Use the runtime-path-
    # stratified tables above for paper-safe breakdowns; the paper-facing
    # candidate-only numerator lives under paper_timing regardless.
    runtime_path_set = sorted({str(event.get("runtime_path")) for event in events})
    timing_backend_set = sorted({str(event.get("timing_backend")) for event in events if event.get("timing_backend")})
    # Section 9: "mixed" is only factually accurate when more than one
    # runtime path actually contributed to these tables -- a single-path
    # population (or an empty one) is labeled accordingly. paper_table_eligible
    # remains False either way: these tables are never the paper-safe source.
    timing_population_semantics = (
        "single_runtime_path_diagnostic_only"
        if len(runtime_path_set) <= 1
        else "mixed_runtime_paths_diagnostic_only"
    )
    overall["timing_population_semantics"] = timing_population_semantics
    overall["paper_table_eligible"] = False
    overall["timing_backend_set"] = timing_backend_set
    overall["runtime_path_set"] = runtime_path_set
    legacy_mixed_aggregate_diagnostics = {
        "applies_to": ["overall", "by_source_layer", "by_gap"],
        "timing_population_semantics": timing_population_semantics,
        "paper_table_eligible": False,
        "runtime_path_set": runtime_path_set,
        "timing_backend_set": timing_backend_set,
        "note": (
            "overall/by_source_layer/by_gap merge candidate_first_crossing and "
            "fixed_source_parallel_flush timing for backward compatibility "
            "only -- use by_runtime_path_and_source_layer / "
            "by_runtime_path_and_gap for paper-safe stratified breakdowns. "
            "The paper-facing candidate-only numerator remains under "
            "paper_timing regardless."
        ),
    }

    # Canonical, order-preserving event-population identity (Major 1): reuses
    # canonical_json_sha256 exactly as-is (lists are hashed in their given
    # order; only dict keys within each row are canonicalized/sorted), so
    # reordering, mutating, or duplicating a row changes the digest. This is
    # computed from the same immutable event-row snapshots passed in here --
    # callers must write these same rows to exact_catchup_events.jsonl.
    event_row_count = len(events)
    event_population_sha256 = canonical_json_sha256(events)
    event_uid_population_sha256 = canonical_json_sha256([str(event.get("event_uid")) for event in events])

    cross_validation = cross_validate_exact_catchup_overhead(events, accounting_summary=accounting_summary)
    dup_diagnostics = dict(duplicate_diagnostics or {})
    accounting_valid = (
        accounting_summary is None or accounting_summary.get("validation", {}).get("status") == "ok"
    )
    event_population_valid = not dup_diagnostics.get("invalid_run", False)
    candidate_decision_accounting = _compute_candidate_decision_accounting(
        events, accounting_summary=accounting_summary
    )
    run_valid = (
        cross_validation["status"] == "ok"
        and accounting_valid
        and event_population_valid
        and candidate_decision_accounting["status"] == "ok"
    )

    def _sort_gap_key(label: str) -> int:
        try:
            return GAP_BIN_LABELS.index(label)
        except ValueError:
            return len(GAP_BIN_LABELS)

    paper_timing = _compute_paper_strict_timing(
        events,
        accounting_summary=accounting_summary,
        generation_timing_summary=generation_timing_summary,
        cross_validation=cross_validation,
        dup_diagnostics=dup_diagnostics,
        generation_wall_time_ms=generation_wall_time_raw,
        generation_timing_valid=generation_timing_valid,
        run_valid=run_valid,
        accounting_valid=accounting_valid,
        candidate_decision_accounting=candidate_decision_accounting,
    )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "event_row_count": event_row_count,
        "event_population_sha256": event_population_sha256,
        "event_uid_population_sha256": event_uid_population_sha256,
        "overall": overall,
        "by_runtime_path": dict(sorted(by_runtime_path.items())),
        "by_source_layer": dict(sorted(by_source_layer.items(), key=lambda item: int(item[0]))),
        "by_gap": dict(sorted(by_gap.items(), key=lambda item: _sort_gap_key(item[0]))),
        "by_runtime_path_and_source_layer": {
            rp: dict(sorted(bucket.items(), key=lambda item: int(item[0])))
            for rp, bucket in sorted(by_runtime_path_and_source_layer.items())
        },
        "by_runtime_path_and_gap": {
            rp: dict(sorted(bucket.items(), key=lambda item: _sort_gap_key(item[0])))
            for rp, bucket in sorted(by_runtime_path_and_gap.items())
        },
        "legacy_mixed_aggregate_diagnostics": legacy_mixed_aggregate_diagnostics,
        "cross_validation": cross_validation,
        "duplicate_diagnostics": dup_diagnostics,
        "generation_wall_time_ms": generation_wall_time_raw,
        "generation_timing_valid": generation_timing_valid,
        "paper_timing": paper_timing,
        "candidate_decision_accounting": candidate_decision_accounting,
        "run_valid": run_valid,
        "speed_claim_valid": False,
    }


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".exact_catchup_overhead_tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_events_jsonl_atomic(path: str, events: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(canonical_json_text(event) + "\n" for event in events)
    _atomic_write_text(path, text)


def write_json_atomic(path: str, payload: Mapping[str, Any]) -> None:
    import json

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


_CSV_COLUMNS_BY_TABLE = {
    "overall": (
        "event_count",
        "successful_event_count",
        "failed_event_count",
        "full_depth_fallback_count",
        "pending_skipped_token_count",
        "nominal_skipped_token_layer_units",
        "exact_catchup_required_token_layer_units",
        "exact_catchup_executed_token_layer_units",
        "net_avoided_token_layer_units",
        "exact_catchup_elapsed_ms",
        "mean_elapsed_ms_per_event",
        "mean_elapsed_ms_per_executed_unit",
        "timing_availability",
        "catchup_share_of_generation_time",
    ),
    "by_gap": (
        "target_layer_occurrence_count",
        "executed_pending_token_layer_units",
        "exact_catchup_elapsed_ms",
        "mean_elapsed_ms_per_target_layer_occurrence",
        "timing_availability",
    ),
}


def write_aggregate_csv_atomic(path: str, rows_by_key: Mapping[str, Mapping[str, Any]], *, key_column: str, table: str) -> None:
    columns = _CSV_COLUMNS_BY_TABLE[table]
    lines = [",".join([key_column] + list(columns))]
    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        values = [str(key)]
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    _atomic_write_text(path, "\n".join(lines) + "\n")


def flatten_runtime_path_stratified_rows(nested: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Flatten a ``{runtime_path: {key: row}}`` stratified aggregate (e.g.
    ``by_runtime_path_and_source_layer`` / ``by_runtime_path_and_gap``) into a
    single flat mapping keyed by ``"<runtime_path>:<key>"``, so it can be
    written with the existing :func:`write_aggregate_csv_atomic` unchanged --
    candidate and fixed-source rows are never placed under the same key
    without a runtime_path prefix distinguishing them."""

    flat: Dict[str, Any] = {}
    for runtime_path, bucket in nested.items():
        for key, row in bucket.items():
            flat["{}:{}".format(runtime_path, key)] = row
    return flat
