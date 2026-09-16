"""Runtime helpers for opt-in restored K/V insertion.

This module intentionally does not decide when restoration should happen.  It
only loads compact calibration artifacts and applies a requested source->target
K/V transform to one token slice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .phase3c_policy_artifact import (
    ARTIFACT_TYPE as PHASE3C_ARTIFACT_TYPE,
    LATEST_SCHEMA_VERSION as PHASE3C_LATEST_SCHEMA_VERSION,
    SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
    SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
    SUPPORTED_SCHEMA_VERSIONS as PHASE3C_SUPPORTED_SCHEMA_VERSIONS,
    apply_hidden_diagonal_affine,
    apply_k_head_channel_affine,
    apply_v_headwise_procrustes,
    get_hidden_layer_pair_parameters,
    get_k_layer_pair_parameters,
    get_threshold_policy_parameters,
    get_v_gap_bin_parameters,
    load_phase3c_policy_artifact,
    phase3c_fixed_source_subset_coverage_validation,
    phase3c_runtime_coverage_validation,
    threshold_key_to_float,
    threshold_to_key,
    validate_candidate_first_crossing_semantics,
    validate_exit_confidence_semantics,
    validate_fixed_layer_membership_semantics,
    validate_official_free_calm_semantics,
    validate_phase3c_policy_artifact,
)
from .missing_kv_calm_trace import (
    CALM_CANDIDATE_LAYERS,
    CALM_CONFIDENCE_COMPUTE_DTYPE,
    CALM_CONFIDENCE_TYPE,
    CALM_THRESHOLD,
    CALM_THRESHOLD_COMPARATOR,
    OFFICIAL_FREE_CALM_POLICY_NAME,
    official_free_calm_candidate_layers,
)
from .missing_kv_dump_provenance import sha256_file

_PRELIMINARY_ARTIFACT_SHA256_RE = re.compile(r"[0-9a-f]{64}")


SUPPORTED_RUNTIME_RESTORATION_METHODS = {
    "source_procrustes",
    "k_affine_v_procrustes",
    "exact_catchup",
    "direct_shallow_kv_reuse",
    "exit_hidden_target_projection",
    "phase3c_kv_final",
}

LEGACY_RUNTIME_RESTORATION_METHODS = {
    "source_procrustes",
    "k_affine_v_procrustes",
}

PHASE3C_RUNTIME_RESTORATION_METHOD = "phase3c_kv_final"
PHASE3C_RUNTIME_MODE = "phase3c_exact_overwrite_smoke"
# Returned by restore_learned_targets_from_hidden_stacked() when the supplied
# target modules are not the T5-RMSNorm/bias-free-Linear form the stacked
# arithmetic vectorizes (e.g. synthetic scheduling fixtures with a plain
# nn.LayerNorm). The caller then keeps its existing sequential
# restore_from_hidden() loop, whose module contract is fully generic.
STACKED_LEARNED_TARGETS_UNSUPPORTED_STATUS = "stacked_learned_targets_unsupported"
EXACT_CATCHUP_METHOD = "exact_catchup"
DIRECT_SHALLOW_KV_REUSE_METHOD = "direct_shallow_kv_reuse"
EXIT_HIDDEN_TARGET_PROJECTION_METHOD = "exit_hidden_target_projection"
ARTIFACT_FREE_CALM_TASKC1_METHODS = {
    EXACT_CATCHUP_METHOD,
    DIRECT_SHALLOW_KV_REUSE_METHOD,
    EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
}
HIDDEN_PROJECTION_RUNTIME_METHODS = {
    PHASE3C_RUNTIME_RESTORATION_METHOD,
    EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
}
CALM_TASKC1_RUNTIME_METHODS = ARTIFACT_FREE_CALM_TASKC1_METHODS | {PHASE3C_RUNTIME_RESTORATION_METHOD}


def _is_nonfinite_error_message(message: Any) -> bool:
    normalized = str(message).lower()
    if "only finite values" in normalized:
        return True
    if "non-finite" in normalized or "nonfinite" in normalized:
        return True
    if "must be finite" in normalized or "finite value" in normalized:
        return True
    return bool(re.search(r"(?<![a-z0-9_])(nan|inf|infinity|infinite)(?![a-z0-9_])", normalized))


def _pair_key(source_layer: int, target_layer: int) -> str:
    return "{}->{}".format(int(source_layer), int(target_layer))


# ---------------------------------------------------------------------------
# Stacked learned-target restoration arithmetic (frozen `stacked_bmm`
# candidate). Re-expressed VERBATIM from the GPU-validated diagnostic
# scripts/benchmark_phase3c_fused_kv_projection.py (_stacked_t5_rms_norm /
# _stacked_projection_input / _stacked_bmm_restore /
# _stacked_downstream_corrections). Only the selected separate-K/separate-V
# candidate is carried into production -- the rejected fused-KV variant is
# deliberately absent. The rejected candidate search is CLOSED.
# ---------------------------------------------------------------------------


def _stacked_t5_rms_norm(
    hidden_states: torch.Tensor,
    norm_weights: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Vectorized equivalent of the per-target T5LayerNorm calls: float32
    variance -> rsqrt(variance + epsilon) -> module-dtype handling ->
    target-specific norm weight multiplication. NOT torch.nn.LayerNorm (no
    mean subtraction)."""

    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    if norm_weights.dtype in (torch.float16, torch.bfloat16):
        hidden_states = hidden_states.to(norm_weights.dtype)
    return norm_weights[:, None, :] * hidden_states


def _stacked_projection_input(
    source_hidden: torch.Tensor,
    bank: Mapping[str, Any],
) -> torch.Tensor:
    hidden_hat = (
        source_hidden.float() * bank["hidden_scale"][:, None, :]
        + bank["hidden_bias"][:, None, :]
    )
    hidden_hat = hidden_hat.to(dtype=bank["module_dtype"])
    return _stacked_t5_rms_norm(
        hidden_hat,
        bank["norm_weights"],
        bank["variance_epsilon"],
    )


def _stacked_downstream_corrections(
    key_flat: torch.Tensor,
    value_flat: torch.Tensor,
    bank: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_count = int(bank["target_count"])
    num_heads = int(bank["num_heads"])
    d_kv = int(bank["d_kv"])
    seq_len = int(key_flat.shape[1])

    key_regenerated = (
        key_flat.reshape(target_count, seq_len, num_heads, d_kv)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    value_regenerated = (
        value_flat.reshape(target_count, seq_len, num_heads, d_kv)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    restored_key = (
        key_regenerated.float() * bank["k_scale"][:, :, None, :]
        + bank["k_bias"][:, :, None, :]
    )

    # The validated target x head Procrustes BMM: T*H independent
    # [N, d_kv] @ [d_kv, d_kv] transforms.
    value_input = value_regenerated.float().reshape(
        target_count * num_heads, seq_len, d_kv
    )
    rotations = bank["v_rotations"].reshape(target_count * num_heads, d_kv, d_kv)
    restored_value = torch.bmm(value_input, rotations)
    restored_value = restored_value + bank["v_biases"].reshape(
        target_count * num_heads, 1, d_kv
    )
    restored_value = restored_value.reshape(target_count, num_heads, seq_len, d_kv)
    return restored_key, restored_value


def _stacked_bmm_learned_restore(
    source_hidden: torch.Tensor,
    bank: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The frozen separate-K + separate-V stacked candidate. nn.Linear stores
    weight as [out_features, in_features], so the projection is
    bmm(input, W.transpose(1, 2)) -- orientation is intentional and must not
    be silently changed."""

    projected_input = _stacked_projection_input(source_hidden, bank)
    key_flat = torch.bmm(projected_input, bank["k_weights"].transpose(1, 2))
    value_flat = torch.bmm(projected_input, bank["v_weights"].transpose(1, 2))
    return _stacked_downstream_corrections(key_flat, value_flat, bank)


def _to_device_dtype(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype)


def _apply_procrustes(source: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    rotations = _to_device_dtype(params["rotations"], source.device, torch.float32)
    biases = _to_device_dtype(params["biases"], source.device, torch.float32)
    source_fit = source.float().squeeze(2)
    if source_fit.ndim != 3:
        raise ValueError("expected source K/V slice shape [batch, heads, 1, dim]")
    pred = torch.einsum("bhd,hde->bhe", source_fit, rotations) + biases.unsqueeze(0)
    return pred.unsqueeze(2).to(dtype=source.dtype)


def _apply_affine(source: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    scale = _to_device_dtype(params["scale"], source.device, torch.float32)
    bias = _to_device_dtype(params["bias"], source.device, torch.float32)
    return (scale * source.float() + bias).to(dtype=source.dtype)


def apply_transform(source: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    kind = params.get("kind")
    if kind == "procrustes_head":
        return _apply_procrustes(source, params)
    if kind == "affine_head_channel_diag":
        return _apply_affine(source, params)
    raise ValueError("unsupported runtime restoration transform kind: {}".format(kind))


def _read_model_config_value(model_config: Any, *names: str) -> Optional[Any]:
    if model_config is None:
        return None
    for name in names:
        if isinstance(model_config, dict) and name in model_config:
            return model_config[name]
        if hasattr(model_config, name):
            return getattr(model_config, name)
    return None


def _verify_preliminary_source6_artifact_sha256(path: str, model_config: Optional[Any]) -> None:
    """SHA-before-load gate for the preliminary candidate-first-crossing
    source-6 override. Must run strictly before the artifact is
    deserialized. Reuses the existing sha256_file helper; introduces no new
    provenance framework."""

    expected = _read_model_config_value(model_config, "kv_runtime_restoration_artifact_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "kv_runtime_restoration_artifact_sha256 is required when "
            "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime is True"
        )
    if _PRELIMINARY_ARTIFACT_SHA256_RE.fullmatch(expected) is None:
        raise ValueError(
            "kv_runtime_restoration_artifact_sha256 must be exactly 64 lowercase hexadecimal "
            "characters, got {!r}".format(expected)
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            "preliminary source-6 artifact SHA-256 mismatch: expected={} actual={}".format(expected, actual)
        )


@dataclass
class RuntimeKVRestorationResult:
    restored_key: Optional[torch.Tensor]
    restored_value: Optional[torch.Tensor]
    status: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RuntimeKVRestorationManager:
    """Loads and applies restored K/V maps from a torch artifact."""

    def __init__(self, artifact: Dict[str, Any], method: str, threshold: Optional[float] = None, model_config: Optional[Any] = None):
        self.artifact = artifact
        self.method = method
        if method not in SUPPORTED_RUNTIME_RESTORATION_METHODS:
            raise ValueError("unsupported runtime restoration method: {}".format(method))
        self.threshold = threshold
        self.threshold_key = None
        self.threshold_payload = None
        self.model_spec = {}
        self.fit_config = {}
        self.policy_config = {}
        self.calibration_source_mode = None
        self.calibration_fixed_source_layer = None
        self.runtime_source_mode = None
        self.runtime_fixed_source_layer = None
        self.calibration_layer_scope_match = False
        self.calibration_runtime_scope_match = False
        self.confidence_semantics_verified = False
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = False
        self.generation_quality_claim_valid = False
        self.artifact_runtime_compatible = False
        self.artifact_paper_population_match = False
        self.artifact_use_classification = "blocked_pending_source6_artifact"
        self.paper_facing_artifact_valid = False
        self.preliminary_source6_subset_enabled = False
        self.runtime_coverage_validation = None
        self.gap_bins = []
        self.candidate_source_layers = []
        self.candidate_threshold = None
        self.candidate_threshold_comparator = None
        self.candidate_policy_sha256 = None
        # Which ExactCacheCalibrationCollector method models/deploying_t5.py's
        # shared CALM Task C1 transaction helper should call when a
        # calibration collector is attached. Every existing (historical)
        # code path leaves this at its default; only the official FREE CALM
        # init paths below override it.
        self.collector_stage_method_name = "stage_first_crossing"
        # Lazy device-local copies of the IMMUTABLE fitted Phase-3c
        # parameters. The artifact is loaded on CPU, so without this every
        # restoration re-issues a host->device transfer for each of the six
        # parameter tensors, for every target layer, on every flush. The
        # values never change during inference, so one materialization per
        # (parameter, device) is enough.
        #
        # Inference-only and non-persistent: it is built from the artifact,
        # never written back to it, never serialized, and has no effect on
        # artifact content or SHA. Keyed by the fully-qualified tensor device
        # so a tensor can never be handed back for a different device.
        self._device_parameter_cache: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
        # Keys of _device_parameter_cache whose tensors have actually been
        # finite-validated. Kept separate so the explicit Batched timing mode
        # (finite validation disabled) can materialize parameters WITHOUT the
        # one-time NaN/Inf scan while a later validation-ON request in the
        # same process can never silently inherit an unvalidated mapping as
        # trusted -- it validates the cached tensors first.
        self._device_parameter_validated: set = set()
        # Lazy single-slot stacked learned-target parameter bank for the
        # Batched Task C2 flush (fitted Phase-3c parameters AND the target
        # T5 RMSNorm/W_K/W_V weights, stacked once per compatible
        # (targets, device, dtype, weight-identity) state). Deliberately NOT
        # built here: from_pretrained() may load real checkpoint weights
        # AFTER construction, so stacking model weights at construction time
        # could freeze random initialization. Built on the FIRST real
        # stacked restoration call instead -- see
        # _stacked_learned_target_bank(). Inference-only, non-persistent,
        # never serialized, never written to the artifact.
        # (fingerprint, bank, fitted_parameters_finite_validated) triple.
        self._stacked_learned_parameter_bank: Optional[Tuple[Tuple[Any, ...], Dict[str, Any], bool]] = None
        # Timing-mode-only fast-reuse slot: ((source, targets, module ids),
        # (device, dtype, epsilon), bank-object reference). Consulted ONLY
        # when validate_fitted_parameters_finite=False; the identity check
        # against the live bank slot means a rebuilt bank or different
        # module objects always fall back to the full validated path.
        self._stacked_learned_bank_fast_reuse: Optional[Tuple[Any, Any, Any]] = None
        # Measurement-only single-slot WEIGHT bank for the stacked
        # exit-hidden (State Copying) projection fairness-control arm --
        # per-target norm/W_K/W_V stacks only, NO fitted parameters. See
        # restore_exit_hidden_targets_stacked().
        self._stacked_exit_hidden_projection_bank_slot: Optional[Tuple[Any, Dict[str, Any]]] = None
        self.runtime_mode = PHASE3C_RUNTIME_MODE if self.is_phase3c else self.method
        if self.is_artifact_free_calm_taskc1:
            self._init_artifact_free_calm_taskc1(model_config)
        elif self.is_phase3c:
            self._init_phase3c(artifact, threshold, model_config)
        else:
            self._init_legacy(artifact, method)

    @classmethod
    def from_path(
        cls,
        path: str,
        method: str,
        threshold: Optional[float] = None,
        model_config: Optional[Any] = None,
    ) -> "RuntimeKVRestorationManager":
        if method in ARTIFACT_FREE_CALM_TASKC1_METHODS:
            if path:
                raise ValueError("{} does not use kv_runtime_restoration_artifact".format(method))
            return cls({}, method, threshold=threshold, model_config=model_config)
        if method == PHASE3C_RUNTIME_RESTORATION_METHOD:
            if not path:
                raise ValueError("phase3c_kv_final requires kv_runtime_restoration_artifact")
            if bool(
                _read_model_config_value(
                    model_config,
                    "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime",
                )
            ):
                _verify_preliminary_source6_artifact_sha256(path, model_config)
            artifact = load_phase3c_policy_artifact(path)
            return cls(artifact, method, threshold=threshold, model_config=model_config)
        artifact = torch.load(path, map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError("runtime restoration artifact must be a dictionary")
        return cls(artifact, method, threshold=threshold, model_config=model_config)

    @property
    def is_phase3c(self) -> bool:
        return self.method == PHASE3C_RUNTIME_RESTORATION_METHOD

    @property
    def is_artifact_free_calm_taskc1(self) -> bool:
        return self.method in ARTIFACT_FREE_CALM_TASKC1_METHODS

    @property
    def uses_hidden_projection(self) -> bool:
        return self.method in HIDDEN_PROJECTION_RUNTIME_METHODS

    def _init_artifact_free_calm_taskc1(self, model_config: Optional[Any]) -> None:
        official_free_calm = bool(
            self._config_value(model_config, "kv_runtime_restoration_official_free_calm_enabled")
        )
        if official_free_calm:
            self._init_official_free_calm_artifact_free_taskc1(model_config)
            return
        mismatches = []
        required_flags = {
            "kv_runtime_restoration_enabled": True,
            "kv_runtime_restoration_calm_enabled": True,
            "use_early_exit": True,
            "use_shallow_deep": False,
            "use_adapt_threshold": False,
            "kv_runtime_restoration_force_restore_all": True,
        }
        for field, expected in required_flags.items():
            actual = bool(self._config_value(model_config, field))
            if actual is not bool(expected):
                mismatches.append("{} must be {}".format(field, expected))
        static_exit_layer = self._config_value(model_config, "static_exit_layer")
        if static_exit_layer is not None:
            mismatches.append("static_exit_layer must be None")
        recent_exact_window = self._config_value(model_config, "kv_runtime_restoration_recent_exact_window")
        if recent_exact_window is not None and int(recent_exact_window) != 0:
            mismatches.append("kv_runtime_restoration_recent_exact_window must be 0")
        runtime_threshold = self._config_value(model_config, "kv_runtime_restoration_threshold")
        if runtime_threshold is None:
            mismatches.append("kv_runtime_restoration_threshold must be {}".format(CALM_THRESHOLD))
        elif abs(float(runtime_threshold) - float(CALM_THRESHOLD)) > 1e-12:
            mismatches.append("kv_runtime_restoration_threshold must be {}".format(CALM_THRESHOLD))
        decoder_layer_count = self._config_value(model_config, "num_layers", "decoder_layer_count")
        if decoder_layer_count is not None and int(decoder_layer_count) != 24:
            mismatches.append("candidate_first_crossing runtime decoder_layer_count must be 24")
        if mismatches:
            raise ValueError(
                "{} Task C1 runtime config mismatch: {}".format(
                    self.method,
                    "; ".join(mismatches),
                )
            )
        self.threshold = float(runtime_threshold) if runtime_threshold is not None else float(CALM_THRESHOLD)
        self.threshold_key = threshold_to_key(self.threshold)
        self.runtime_mode = (
            "exact_catchup"
            if self.method == EXACT_CATCHUP_METHOD
            else "{}_exact_overwrite_smoke".format(self.method)
        )
        self.runtime_source_mode = SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
        self.runtime_fixed_source_layer = None
        self.calibration_source_mode = "runtime_builtin"
        self.calibration_fixed_source_layer = None
        self.calibration_layer_scope_match = True
        self.calibration_runtime_scope_match = True
        self.confidence_semantics_verified = True
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = True
        self.generation_quality_claim_valid = True
        self.candidate_source_layers = list(CALM_CANDIDATE_LAYERS)
        self.candidate_threshold = float(CALM_THRESHOLD)
        self.candidate_threshold_comparator = str(CALM_THRESHOLD_COMPARATOR)
        self.candidate_policy_sha256 = None
        num_layers = int(decoder_layer_count) if decoder_layer_count is not None else 24
        d_model = self._config_value(model_config, "d_model")
        num_heads = self._config_value(model_config, "num_heads")
        d_kv = self._config_value(model_config, "d_kv")
        self.model_spec = {
            "decoder_layer_count": num_layers,
            "d_model": int(d_model) if d_model is not None else None,
            "num_heads": int(num_heads) if num_heads is not None else None,
            "d_kv": int(d_kv) if d_kv is not None else None,
        }
        self.fit_config = {
            "source_layer_mode": SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING,
            "candidate_first_crossing_semantics": {
                "candidate_exit_layers": list(CALM_CANDIDATE_LAYERS),
                "candidate_evaluation_order": list(CALM_CANDIDATE_LAYERS),
                "threshold": float(CALM_THRESHOLD),
                "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
                "confidence_type": CALM_CONFIDENCE_TYPE,
                "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
                "adaptive_threshold": False,
            },
        }
        self.policy_config = {}
        self.method_payload = {}
        self.pairs = {}

    def _init_official_free_calm_artifact_free_taskc1(self, model_config: Optional[Any]) -> None:
        """Official FREE CALM-style production early-exit counterpart of
        ``_init_artifact_free_calm_taskc1`` above (that method's own
        historical (4,6,8,10)/frozen-0.9/24-layer checks are left completely
        untouched -- this is a parallel branch, only reached when the caller
        explicitly opts in via kv_runtime_restoration_official_free_calm_enabled).

        The runtime threshold must equal the model's own
        ``exit_conf_threshold`` (the value that actually governs the
        early-exit decision this restoration attaches to) -- never a
        separately frozen number -- and the candidate range is derived from
        ``exit_min_layer``/``num_layers``, never hard-coded.
        """

        mismatches = []
        required_flags = {
            "kv_runtime_restoration_enabled": True,
            "kv_runtime_restoration_calm_enabled": True,
            "use_early_exit": True,
            "use_shallow_deep": False,
            "use_adapt_threshold": False,
            "kv_runtime_restoration_force_restore_all": True,
        }
        for field, expected in required_flags.items():
            actual = bool(self._config_value(model_config, field))
            if actual is not bool(expected):
                mismatches.append("{} must be {}".format(field, expected))
        static_exit_layer = self._config_value(model_config, "static_exit_layer")
        if static_exit_layer is not None:
            mismatches.append("static_exit_layer must be None")
        recent_exact_window = self._config_value(model_config, "kv_runtime_restoration_recent_exact_window")
        if recent_exact_window is not None and int(recent_exact_window) != 0:
            mismatches.append("kv_runtime_restoration_recent_exact_window must be 0")
        exit_conf_threshold = self._config_value(model_config, "exit_conf_threshold")
        if exit_conf_threshold is None:
            mismatches.append("exit_conf_threshold must be set")
        runtime_threshold = self._config_value(model_config, "kv_runtime_restoration_threshold")
        if runtime_threshold is None:
            mismatches.append("kv_runtime_restoration_threshold must be set")
        elif exit_conf_threshold is not None and abs(float(runtime_threshold) - float(exit_conf_threshold)) > 1e-12:
            mismatches.append("kv_runtime_restoration_threshold must equal exit_conf_threshold")
        # Runtime-decision parity with the original FREE get_skip_mask():
        # only holds when exit_conf_type=="softmax" (the Task C1 branch
        # always uses the existing softmax top-1/top-2 margin helper) and
        # exit_position_temp is None (the Task C1 branch compares against a
        # single constant threshold, with no per-position decay).
        exit_conf_type = self._config_value(model_config, "exit_conf_type")
        if exit_conf_type != "softmax":
            mismatches.append("exit_conf_type must be softmax")
        exit_position_temp = self._config_value(model_config, "exit_position_temp")
        if exit_position_temp is not None:
            mismatches.append("exit_position_temp must be None")
        exit_min_layer = self._config_value(model_config, "exit_min_layer")
        if exit_min_layer is None:
            mismatches.append("exit_min_layer must be set")
        decoder_layer_count = self._config_value(model_config, "num_layers", "decoder_layer_count")
        if decoder_layer_count is None:
            mismatches.append("num_layers must be set")
        if mismatches:
            raise ValueError(
                "{} official_free_calm Task C1 runtime config mismatch: {}".format(
                    self.method,
                    "; ".join(mismatches),
                )
            )
        candidate_layers = list(
            official_free_calm_candidate_layers(
                exit_min_layer=int(exit_min_layer), num_decoder_layers=int(decoder_layer_count)
            )
        )
        self.threshold = float(runtime_threshold)
        self.threshold_key = threshold_to_key(self.threshold)
        self.runtime_mode = (
            "exact_catchup"
            if self.method == EXACT_CATCHUP_METHOD
            else "{}_exact_overwrite_smoke".format(self.method)
        )
        self.runtime_source_mode = SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM
        self.runtime_fixed_source_layer = None
        self.calibration_source_mode = "runtime_builtin"
        self.calibration_fixed_source_layer = None
        self.calibration_layer_scope_match = True
        self.calibration_runtime_scope_match = True
        self.confidence_semantics_verified = True
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = True
        self.generation_quality_claim_valid = True
        self.candidate_source_layers = candidate_layers
        self.candidate_threshold = self.threshold
        self.candidate_threshold_comparator = str(CALM_THRESHOLD_COMPARATOR)
        self.candidate_policy_sha256 = None
        self.collector_stage_method_name = "stage_official_free_calm_first_crossing"
        num_layers = int(decoder_layer_count)
        d_model = self._config_value(model_config, "d_model")
        num_heads = self._config_value(model_config, "num_heads")
        d_kv = self._config_value(model_config, "d_kv")
        self.model_spec = {
            "decoder_layer_count": num_layers,
            "d_model": int(d_model) if d_model is not None else None,
            "num_heads": int(num_heads) if num_heads is not None else None,
            "d_kv": int(d_kv) if d_kv is not None else None,
        }
        self.fit_config = {
            "source_layer_mode": SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM,
            "official_free_calm_semantics": {
                "candidate_exit_layers": candidate_layers,
                "candidate_evaluation_order": candidate_layers,
                "threshold": self.threshold,
                "threshold_comparator": CALM_THRESHOLD_COMPARATOR,
                "confidence_type": CALM_CONFIDENCE_TYPE,
                "confidence_compute_dtype": CALM_CONFIDENCE_COMPUTE_DTYPE,
                "adaptive_threshold": False,
            },
        }
        self.policy_config = {}
        self.method_payload = {}
        self.pairs = {}

    def _init_legacy(self, artifact: Dict[str, Any], method: str) -> None:
        if artifact.get("artifact_type") == PHASE3C_ARTIFACT_TYPE:
            raise ValueError(
                "Phase 3c artifact requires kv_runtime_restoration_method={}".format(
                    PHASE3C_RUNTIME_RESTORATION_METHOD
                )
            )
        methods = artifact.get("methods") or {}
        if method not in methods:
            raise KeyError("runtime restoration artifact missing method: {}".format(method))
        self.method_payload = methods[method]
        self.pairs = self.method_payload.get("pairs") or {}

    def _init_phase3c(self, artifact: Dict[str, Any], threshold: Optional[float], model_config: Optional[Any]) -> None:
        validate_phase3c_policy_artifact(artifact)
        if artifact.get("artifact_type") != PHASE3C_ARTIFACT_TYPE:
            raise ValueError("phase3c_kv_final requires artifact_type={!r}".format(PHASE3C_ARTIFACT_TYPE))
        if artifact.get("schema_version") not in PHASE3C_SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported Phase 3c schema_version: {}".format(artifact.get("schema_version")))
        params_by_threshold = artifact.get("parameters_by_threshold") or {}
        threshold_keys = sorted(params_by_threshold.keys(), key=threshold_key_to_float)
        if threshold is None:
            if len(threshold_keys) != 1:
                raise ValueError(
                    "phase3c_kv_final requires explicit kv_runtime_restoration_threshold when artifact has "
                    "multiple thresholds: {}".format(threshold_keys)
                )
            threshold_key = threshold_keys[0]
            threshold = threshold_key_to_float(threshold_key)
        else:
            threshold_key = threshold_to_key(threshold)
        # Exact lookup only; no nearest-threshold fallback.
        self.threshold_payload = get_threshold_policy_parameters(artifact, threshold)
        self.threshold = float(threshold)
        self.threshold_key = threshold_key
        self.model_spec = dict(artifact.get("model_spec") or {})
        self.fit_config = dict(artifact.get("fit_config") or {})
        self.policy_config = dict(artifact.get("policy_config") or {})
        self.calibration_source_mode = self.fit_config.get("source_layer_mode") or self.policy_config.get("source_layer_mode")
        self.calibration_fixed_source_layer = self.fit_config.get("fixed_source_layer")
        self.gap_bins = list(self.fit_config.get("gap_bins") or [])
        preliminary_candidate_source6 = bool(
            self.calibration_source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
            and self._config_value(
                model_config,
                "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime",
            )
        )
        if preliminary_candidate_source6:
            self.runtime_coverage_validation = phase3c_fixed_source_subset_coverage_validation(
                artifact,
                threshold=self.threshold,
                fixed_source_layer=6,
                decoder_layer_count=24,
            )
        else:
            self.runtime_coverage_validation = phase3c_runtime_coverage_validation(artifact)
        if self.runtime_coverage_validation.get("status") == "failed":
            raise ValueError("Phase 3c artifact runtime coverage validation failed: {}".format(self.runtime_coverage_validation))
        if bool(self._config_value(model_config, "use_adapt_threshold")):
            raise NotImplementedError("use_adapt_threshold must be False for phase3c_kv_final Task C1")
        if self.calibration_source_mode == SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING:
            if preliminary_candidate_source6:
                self._init_phase3c_preliminary_candidate_source6_fixed_runtime(
                    artifact, model_config
                )
            else:
                self._init_phase3c_candidate_first_crossing(artifact, model_config)
        elif self.calibration_source_mode == SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM:
            self._init_phase3c_official_free_calm(artifact, model_config)
        else:
            self._init_phase3c_fixed_layer(artifact, model_config)
        self._validate_phase3c_model_config(model_config)
        self.method_payload = self.threshold_payload
        self.pairs = self.threshold_payload.get("hidden_by_layer_pair") or {}

    def _init_phase3c_fixed_layer(self, artifact: Dict[str, Any], model_config: Optional[Any]) -> None:
        # A genuine fixed_layer artifact is schema-version-agnostic:
        # validate_phase3c_policy_artifact's own source_layer_mode==
        # SOURCE_LAYER_MODE_FIXED branch (unlike candidate_first_crossing/
        # official_free_calm) never restricts schema_version, and nothing
        # downstream here (confidence/membership semantics, threshold
        # parameter lookup, hidden/K/V restoration) branches on it either.
        # This previously required exact equality with the legacy
        # SCHEMA_VERSION (1), which rejected the Native FREE fixed-source-6
        # producer's own schema_version=LATEST_SCHEMA_VERSION (2) artifacts
        # -- reuse the same repository-declared supported-version set the
        # general _init_phase3c gate above already checks, rather than a
        # second, narrower one.
        if artifact.get("schema_version") not in PHASE3C_SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                "unsupported Phase 3c runtime schema_version for fixed-layer Task C1: {}".format(
                    artifact.get("schema_version")
                )
            )
        if bool(self._config_value(model_config, "kv_runtime_restoration_calm_enabled")):
            raise ValueError("fixed-layer phase3c_kv_final Task C1 requires kv_runtime_restoration_calm_enabled=False")
        self.runtime_source_mode = "fixed_shallow_layer"
        runtime_fixed = self._config_value(model_config, "shallow_exit_layer")
        self.runtime_fixed_source_layer = int(runtime_fixed) if runtime_fixed is not None else None
        self.calibration_layer_scope_match = bool(
            self.calibration_source_mode == "fixed_layer"
            and self.calibration_fixed_source_layer is not None
            and self.runtime_fixed_source_layer is not None
            and int(self.calibration_fixed_source_layer) == int(self.runtime_fixed_source_layer)
        )
        self.calibration_runtime_scope_match = self.calibration_layer_scope_match
        self.confidence_semantics_verified = bool(validate_exit_confidence_semantics(self.fit_config))
        self.fixed_layer_membership_semantics_verified = bool(
            self.calibration_layer_scope_match
            and validate_fixed_layer_membership_semantics(self.fit_config)
        )
        self.generation_quality_claim_valid = bool(
            self.calibration_layer_scope_match
            and self.confidence_semantics_verified
            and self.fixed_layer_membership_semantics_verified
        )
        # A genuine fixed-layer artifact must continue to use this existing
        # path unchanged under any runtime config, including Native FREE
        # (use_shallow_deep=True): no extra opt-in and no forced downgrade to
        # preliminary status. A prior revision added an unconditional
        # "blocked pending preliminary flag" gate here that also
        # mislabeled a genuine fixed-layer artifact as
        # artifact_use_classification="preliminary_runtime_only" whenever the
        # opt-in was set; that block has been removed.
        #
        # Replace the __init__ "blocked" defaults with the actual outcome of
        # this successful runtime load. Paper-facing acceptance is never
        # granted here: it belongs to central approval and external
        # evidence, not merely runtime compatibility, so
        # paper_facing_artifact_valid stays False regardless of how strong
        # the runtime-side checks above are.
        self.artifact_runtime_compatible = True
        self.artifact_paper_population_match = bool(
            self.calibration_layer_scope_match
            and self.confidence_semantics_verified
            and self.fixed_layer_membership_semantics_verified
        )
        self.artifact_use_classification = "fixed_layer_runtime"
        self.paper_facing_artifact_valid = False
        self.preliminary_source6_subset_enabled = False

    def _init_phase3c_preliminary_candidate_source6_fixed_runtime(
        self,
        artifact: Dict[str, Any],
        model_config: Optional[Any],
    ) -> None:
        """Development-only source-6 dispatch for a multi-source artifact.

        The calibration semantics remain candidate-first-crossing.  Only the
        existing source-6 maps are exposed to the Native FREE fixed shallow
        runtime; no artifact field is rewritten or reinterpreted globally.
        """

        if artifact.get("schema_version") != PHASE3C_LATEST_SCHEMA_VERSION:
            raise ValueError(
                "preliminary candidate source-6 runtime requires schema_version {}".format(
                    PHASE3C_LATEST_SCHEMA_VERSION
                )
            )
        if not bool(validate_candidate_first_crossing_semantics(self.fit_config)):
            raise ValueError("candidate_first_crossing Phase 3c artifact semantics mismatch")
        required_flags = {
            "kv_runtime_restoration_enabled": True,
            "kv_runtime_restoration_calm_enabled": False,
            "use_shallow_deep": True,
            "use_early_exit": False,
            "use_adapt_threshold": False,
            "kv_runtime_restoration_force_restore_all": True,
            "kv_runtime_restoration_allow_preliminary_candidate_source6_fixed_runtime": True,
        }
        mismatches = []
        for field, expected in required_flags.items():
            actual = bool(self._config_value(model_config, field))
            if actual is not bool(expected):
                mismatches.append("{} must be {}".format(field, expected))
        runtime_fixed = self._config_value(model_config, "shallow_exit_layer")
        if runtime_fixed is None or int(runtime_fixed) != 6:
            mismatches.append("shallow_exit_layer must be 6")
        recent_exact_window = self._config_value(
            model_config, "kv_runtime_restoration_recent_exact_window"
        )
        if recent_exact_window is not None and int(recent_exact_window) != 0:
            mismatches.append("kv_runtime_restoration_recent_exact_window must be 0")
        method = self._config_value(model_config, "kv_runtime_restoration_method")
        if method is not None and str(method) != PHASE3C_RUNTIME_RESTORATION_METHOD:
            mismatches.append(
                "kv_runtime_restoration_method must be {}".format(
                    PHASE3C_RUNTIME_RESTORATION_METHOD
                )
            )
        if abs(float(self.threshold) - float(CALM_THRESHOLD)) > 1e-12:
            mismatches.append("artifact runtime threshold must be {}".format(CALM_THRESHOLD))
        runtime_layer_count = self._config_value(
            model_config, "num_layers", "decoder_layer_count"
        )
        if runtime_layer_count is None or int(runtime_layer_count) != 24:
            mismatches.append("runtime decoder_layer_count must be 24")
        if mismatches:
            raise ValueError(
                "preliminary candidate source-6 fixed runtime config mismatch: {}".format(
                    "; ".join(mismatches)
                )
            )

        # Reuse the exact same frozen-contract fields the existing CALM
        # candidate_first_crossing path checks (_init_phase3c_candidate_first_crossing
        # below) -- no new confidence/policy definition is introduced here.
        semantics = self.fit_config.get("candidate_first_crossing_semantics") or {}
        candidate_layers = [int(item) for item in semantics.get("candidate_exit_layers") or []]
        if candidate_layers != list(CALM_CANDIDATE_LAYERS):
            raise ValueError(
                "preliminary candidate source-6 fixed runtime requires the frozen candidate layers "
                "{}: got {}".format(list(CALM_CANDIDATE_LAYERS), candidate_layers)
            )
        if semantics.get("candidate_evaluation_order") != list(CALM_CANDIDATE_LAYERS):
            raise ValueError("candidate_first_crossing evaluation order mismatch")
        if 6 not in candidate_layers:
            raise ValueError("candidate_first_crossing artifact does not include source layer 6")
        if semantics.get("threshold_comparator") != CALM_THRESHOLD_COMPARATOR:
            raise ValueError("candidate_first_crossing threshold comparator mismatch")
        if semantics.get("confidence_type") != CALM_CONFIDENCE_TYPE:
            raise ValueError("candidate_first_crossing confidence type mismatch")
        if semantics.get("confidence_compute_dtype") != CALM_CONFIDENCE_COMPUTE_DTYPE:
            raise ValueError("candidate_first_crossing confidence dtype mismatch")
        if bool(semantics.get("adaptive_threshold")):
            raise ValueError("candidate_first_crossing adaptive threshold must be disabled")

        self.runtime_source_mode = "fixed_shallow_layer"
        self.runtime_fixed_source_layer = 6
        self.calibration_layer_scope_match = False
        self.calibration_runtime_scope_match = False
        self.confidence_semantics_verified = True
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = True
        self.generation_quality_claim_valid = False
        self.artifact_runtime_compatible = True
        self.artifact_paper_population_match = False
        self.artifact_use_classification = "preliminary_runtime_only"
        self.paper_facing_artifact_valid = False
        self.preliminary_source6_subset_enabled = True
        self.candidate_source_layers = candidate_layers
        self.candidate_threshold = float(semantics.get("threshold"))
        self.candidate_threshold_comparator = str(semantics.get("threshold_comparator"))
        self.candidate_policy_sha256 = str(semantics.get("policy_sha256"))

    def _init_phase3c_candidate_first_crossing(self, artifact: Dict[str, Any], model_config: Optional[Any]) -> None:
        if artifact.get("schema_version") != PHASE3C_LATEST_SCHEMA_VERSION:
            raise ValueError(
                "candidate_first_crossing Phase 3c Task C1 requires schema_version {}".format(
                    PHASE3C_LATEST_SCHEMA_VERSION
                )
            )
        if not bool(validate_candidate_first_crossing_semantics(self.fit_config)):
            raise ValueError("candidate_first_crossing Phase 3c artifact semantics mismatch")
        required_flags = {
            "kv_runtime_restoration_enabled": True,
            "kv_runtime_restoration_calm_enabled": True,
            "use_early_exit": True,
            "use_shallow_deep": False,
            "use_adapt_threshold": False,
            "kv_runtime_restoration_force_restore_all": True,
            # This artifact's source_layer_mode is the frozen historical
            # (4,6,8,10)/0.9 policy, never the official FREE CALM-style
            # candidate range/threshold -- the runtime flag must agree, or a
            # server misconfiguration could silently run the historical
            # confidence/threshold semantics against a runtime believed to
            # be in official mode (or vice versa).
            "kv_runtime_restoration_official_free_calm_enabled": False,
        }
        mismatches = []
        for field, expected in required_flags.items():
            actual = bool(self._config_value(model_config, field))
            if actual is not bool(expected):
                mismatches.append("{} must be {}".format(field, expected))
        static_exit_layer = self._config_value(model_config, "static_exit_layer")
        if static_exit_layer is not None:
            mismatches.append("static_exit_layer must be None")
        recent_exact_window = self._config_value(model_config, "kv_runtime_restoration_recent_exact_window")
        if recent_exact_window is not None and int(recent_exact_window) != 0:
            mismatches.append("kv_runtime_restoration_recent_exact_window must be 0")
        method = self._config_value(model_config, "kv_runtime_restoration_method")
        if method is not None and str(method) != PHASE3C_RUNTIME_RESTORATION_METHOD:
            mismatches.append("kv_runtime_restoration_method must be {}".format(PHASE3C_RUNTIME_RESTORATION_METHOD))
        runtime_threshold = self._config_value(model_config, "kv_runtime_restoration_threshold")
        if runtime_threshold is not None and abs(float(runtime_threshold) - float(CALM_THRESHOLD)) > 1e-12:
            mismatches.append("kv_runtime_restoration_threshold must be {}".format(CALM_THRESHOLD))
        if abs(float(self.threshold) - float(CALM_THRESHOLD)) > 1e-12:
            mismatches.append("artifact runtime threshold must be {}".format(CALM_THRESHOLD))
        artifact_decoder_layer_count = int(self.model_spec.get("decoder_layer_count", 0) or 0)
        runtime_decoder_layer_count = self._config_value(model_config, "num_layers", "decoder_layer_count")
        if artifact_decoder_layer_count != 24:
            mismatches.append(
                "candidate_first_crossing frozen decoder_layer_count must be 24 in artifact, got {}".format(
                    artifact_decoder_layer_count
                )
            )
        if runtime_decoder_layer_count is None or int(runtime_decoder_layer_count) != 24:
            mismatches.append(
                "candidate_first_crossing frozen runtime decoder_layer_count must be 24, got {}".format(
                    runtime_decoder_layer_count
                )
            )
        if mismatches:
            raise ValueError("candidate_first_crossing Phase 3c Task C1 runtime config mismatch: {}".format("; ".join(mismatches)))

        semantics = self.fit_config.get("candidate_first_crossing_semantics") or {}
        candidate_layers = [int(item) for item in semantics.get("candidate_exit_layers") or []]
        if candidate_layers != list(CALM_CANDIDATE_LAYERS):
            raise ValueError("candidate_first_crossing candidate layers mismatch: {}".format(candidate_layers))
        if semantics.get("candidate_evaluation_order") != list(CALM_CANDIDATE_LAYERS):
            raise ValueError("candidate_first_crossing evaluation order mismatch")
        if semantics.get("threshold_comparator") != CALM_THRESHOLD_COMPARATOR:
            raise ValueError("candidate_first_crossing threshold comparator mismatch")
        if semantics.get("confidence_type") != CALM_CONFIDENCE_TYPE:
            raise ValueError("candidate_first_crossing confidence type mismatch")
        if semantics.get("confidence_compute_dtype") != CALM_CONFIDENCE_COMPUTE_DTYPE:
            raise ValueError("candidate_first_crossing confidence dtype mismatch")
        if bool(semantics.get("adaptive_threshold")):
            raise ValueError("candidate_first_crossing adaptive threshold must be disabled")

        self.runtime_source_mode = SOURCE_LAYER_MODE_CANDIDATE_FIRST_CROSSING
        self.runtime_fixed_source_layer = None
        self.calibration_layer_scope_match = True
        self.confidence_semantics_verified = True
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = True
        self.calibration_runtime_scope_match = True
        self.generation_quality_claim_valid = True
        # artifact_paper_population_match=True here means only that the
        # artifact calibration population matches the controlled
        # candidate-first-crossing runtime actually being executed; it is
        # not a Native FREE paper-facing claim, so
        # paper_facing_artifact_valid stays False.
        self.artifact_runtime_compatible = True
        self.artifact_paper_population_match = True
        self.artifact_use_classification = "candidate_first_crossing_runtime"
        self.paper_facing_artifact_valid = False
        self.preliminary_source6_subset_enabled = False
        self.candidate_source_layers = candidate_layers
        self.candidate_threshold = float(semantics.get("threshold"))
        self.candidate_threshold_comparator = str(semantics.get("threshold_comparator"))
        self.candidate_policy_sha256 = str(semantics.get("policy_sha256"))

    def _init_phase3c_official_free_calm(self, artifact: Dict[str, Any], model_config: Optional[Any]) -> None:
        """Official FREE CALM-style production early-exit counterpart of
        ``_init_phase3c_candidate_first_crossing`` above (left completely
        unmodified). Same shape of runtime-config/artifact-semantics
        cross-checks, but validated via ``validate_official_free_calm_
        semantics`` (structural: contiguous candidate range, comparator,
        confidence semantics -- never exact equality to the frozen
        (4,6,8,10)/0.9 tuple) and against the model's own ``exit_conf_
        threshold``/``exit_min_layer``/``num_layers`` instead of frozen
        constants."""

        if artifact.get("schema_version") != PHASE3C_LATEST_SCHEMA_VERSION:
            raise ValueError(
                "official_free_calm Phase 3c Task C1 requires schema_version {}".format(
                    PHASE3C_LATEST_SCHEMA_VERSION
                )
            )
        if not bool(validate_official_free_calm_semantics(self.fit_config)):
            raise ValueError("official_free_calm Phase 3c artifact semantics mismatch")
        required_flags = {
            "kv_runtime_restoration_enabled": True,
            "kv_runtime_restoration_calm_enabled": True,
            "use_early_exit": True,
            "use_shallow_deep": False,
            "use_adapt_threshold": False,
            "kv_runtime_restoration_force_restore_all": True,
            # The converse of the same binding check in
            # _init_phase3c_candidate_first_crossing above: this artifact's
            # source_layer_mode is official_free_calm_first_crossing, so the
            # runtime must have explicitly opted into official mode too.
            "kv_runtime_restoration_official_free_calm_enabled": True,
        }
        mismatches = []
        for field, expected in required_flags.items():
            actual = bool(self._config_value(model_config, field))
            if actual is not bool(expected):
                mismatches.append("{} must be {}".format(field, expected))
        static_exit_layer = self._config_value(model_config, "static_exit_layer")
        if static_exit_layer is not None:
            mismatches.append("static_exit_layer must be None")
        recent_exact_window = self._config_value(model_config, "kv_runtime_restoration_recent_exact_window")
        if recent_exact_window is not None and int(recent_exact_window) != 0:
            mismatches.append("kv_runtime_restoration_recent_exact_window must be 0")
        method = self._config_value(model_config, "kv_runtime_restoration_method")
        if method is not None and str(method) != PHASE3C_RUNTIME_RESTORATION_METHOD:
            mismatches.append("kv_runtime_restoration_method must be {}".format(PHASE3C_RUNTIME_RESTORATION_METHOD))
        exit_conf_threshold = self._config_value(model_config, "exit_conf_threshold")
        runtime_threshold = self._config_value(model_config, "kv_runtime_restoration_threshold")
        if exit_conf_threshold is None:
            mismatches.append("exit_conf_threshold must be set")
        if runtime_threshold is not None and exit_conf_threshold is not None and abs(
            float(runtime_threshold) - float(exit_conf_threshold)
        ) > 1e-12:
            mismatches.append("kv_runtime_restoration_threshold must equal exit_conf_threshold")
        if exit_conf_threshold is not None and abs(float(self.threshold) - float(exit_conf_threshold)) > 1e-12:
            mismatches.append("artifact runtime threshold must equal exit_conf_threshold")
        # Same runtime-decision parity requirement as
        # _init_official_free_calm_artifact_free_taskc1 above.
        exit_conf_type = self._config_value(model_config, "exit_conf_type")
        if exit_conf_type != "softmax":
            mismatches.append("exit_conf_type must be softmax")
        exit_position_temp = self._config_value(model_config, "exit_position_temp")
        if exit_position_temp is not None:
            mismatches.append("exit_position_temp must be None")
        exit_min_layer = self._config_value(model_config, "exit_min_layer")
        if exit_min_layer is None:
            mismatches.append("exit_min_layer must be set")
        artifact_decoder_layer_count = int(self.model_spec.get("decoder_layer_count", 0) or 0)
        runtime_decoder_layer_count = self._config_value(model_config, "num_layers", "decoder_layer_count")
        if runtime_decoder_layer_count is None or int(runtime_decoder_layer_count) != artifact_decoder_layer_count:
            mismatches.append(
                "official_free_calm runtime decoder_layer_count must equal the artifact's {}, got {}".format(
                    artifact_decoder_layer_count, runtime_decoder_layer_count
                )
            )
        if mismatches:
            raise ValueError("official_free_calm Phase 3c Task C1 runtime config mismatch: {}".format("; ".join(mismatches)))

        semantics = self.fit_config.get("official_free_calm_semantics") or {}
        candidate_layers = [int(item) for item in semantics.get("candidate_exit_layers") or []]
        if exit_min_layer is not None and runtime_decoder_layer_count is not None:
            expected_runtime_candidate_layers = list(
                official_free_calm_candidate_layers(
                    exit_min_layer=int(exit_min_layer), num_decoder_layers=int(runtime_decoder_layer_count)
                )
            )
            if candidate_layers != expected_runtime_candidate_layers:
                raise ValueError(
                    "official_free_calm artifact candidate_exit_layers does not match the runtime "
                    "exit_min_layer/num_layers-derived range"
                )

        self.runtime_source_mode = SOURCE_LAYER_MODE_OFFICIAL_FREE_CALM
        self.runtime_fixed_source_layer = None
        self.calibration_layer_scope_match = True
        self.confidence_semantics_verified = True
        self.fixed_layer_membership_semantics_verified = False
        self.candidate_policy_semantics_verified = True
        self.calibration_runtime_scope_match = True
        self.generation_quality_claim_valid = True
        self.artifact_runtime_compatible = True
        self.artifact_paper_population_match = True
        self.artifact_use_classification = "official_free_calm_runtime"
        self.paper_facing_artifact_valid = False
        self.preliminary_source6_subset_enabled = False
        self.candidate_source_layers = candidate_layers
        self.candidate_threshold = float(semantics.get("threshold"))
        self.candidate_threshold_comparator = str(semantics.get("threshold_comparator"))
        self.candidate_policy_sha256 = str(semantics.get("policy_sha256"))
        self.collector_stage_method_name = "stage_official_free_calm_first_crossing"

    def _config_value(self, model_config: Any, *names: str) -> Optional[int]:
        return _read_model_config_value(model_config, *names)

    def _validate_phase3c_model_config(self, model_config: Optional[Any]) -> None:
        if model_config is None:
            return
        expected = {
            "decoder_layer_count": int(self.model_spec.get("decoder_layer_count")),
            "d_model": int(self.model_spec.get("d_model")),
            "num_heads": int(self.model_spec.get("num_heads")),
            "d_kv": int(self.model_spec.get("d_kv")),
        }
        actual = {
            "decoder_layer_count": self._config_value(model_config, "num_layers", "decoder_layer_count"),
            "d_model": self._config_value(model_config, "d_model"),
            "num_heads": self._config_value(model_config, "num_heads"),
            "d_kv": self._config_value(model_config, "d_kv"),
        }
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value is not None and int(actual_value) != int(expected_value):
                mismatches.append("{} expected {} got {}".format(key, expected_value, actual_value))
        if mismatches:
            raise ValueError("Phase 3c artifact/model config mismatch: {}".format("; ".join(mismatches)))

    def select_phase3c_gap_bin(self, source_layer: int, target_layer: int) -> str:
        gap = int(target_layer) - int(source_layer)
        for gap_bin in self.gap_bins:
            label = gap_bin.get("label")
            start = int(gap_bin.get("start"))
            end = int(gap_bin.get("end"))
            if start <= gap <= end:
                return str(label)
        raise ValueError("missing_v_gap_bin_for_gap: gap={} source_layer={} target_layer={}".format(gap, source_layer, target_layer))

    def has_pair(self, source_layer: int, target_layer: int) -> bool:
        if self.method == DIRECT_SHALLOW_KV_REUSE_METHOD:
            return int(target_layer) > int(source_layer)
        if self.method == EXIT_HIDDEN_TARGET_PROJECTION_METHOD:
            return int(target_layer) >= int(source_layer)
        if self.is_phase3c:
            if self.preliminary_source6_subset_enabled and int(source_layer) != 6:
                # The preliminary override exposes only the source-6 slice of
                # a multi-source candidate_first_crossing artifact; source
                # 4/8/10 must never be reachable through this mode even
                # though their maps are structurally present in the artifact.
                return False
            try:
                get_hidden_layer_pair_parameters(self.artifact, self.threshold, source_layer, target_layer)
                get_k_layer_pair_parameters(self.artifact, self.threshold, source_layer, target_layer)
                gap_bin = self.select_phase3c_gap_bin(source_layer, target_layer)
                get_v_gap_bin_parameters(self.artifact, self.threshold, gap_bin)
                return True
            except Exception:
                return False
        return _pair_key(source_layer, target_layer) in self.pairs

    def restore(
        self,
        source_key: torch.Tensor,
        source_value: torch.Tensor,
        source_layer: int,
        target_layer: int,
    ) -> RuntimeKVRestorationResult:
        if self.method == DIRECT_SHALLOW_KV_REUSE_METHOD:
            metadata = {
                "runtime_mode": self.runtime_mode,
                "source_layer": int(source_layer),
                "target_layer": int(target_layer),
                "restoration_submode": DIRECT_SHALLOW_KV_REUSE_METHOD,
                "direct_source_kv_layer": int(source_layer),
                "exact_catchup_already_computed": True,
                "speed_claim_valid": False,
            }
            try:
                if int(target_layer) <= int(source_layer):
                    return RuntimeKVRestorationResult(
                        None,
                        None,
                        "target_not_deeper_than_source",
                        "direct shallow K/V reuse requires target_layer > source_kv_layer",
                        metadata,
                    )
                if source_key.ndim != 4 or source_value.ndim != 4:
                    raise ValueError("source K/V must have shape [batch, heads, 1, d_kv]")
                if int(source_key.shape[0]) != 1 or int(source_value.shape[0]) != 1:
                    return RuntimeKVRestorationResult(
                        None,
                        None,
                        "unsupported_batch",
                        "direct_shallow_kv_reuse currently supports batch size 1 only",
                        metadata,
                    )
                if int(source_key.shape[2]) != 1 or int(source_value.shape[2]) != 1:
                    raise ValueError("source K/V slice must contain one token")
                if source_key.shape != source_value.shape:
                    raise ValueError("source K/V shape mismatch")
                if not torch.isfinite(source_key).all() or not torch.isfinite(source_value).all():
                    raise ValueError("direct source K/V contains NaN or Inf")
                restored_key = source_key.detach().clone()
                restored_value = source_value.detach().clone()
                metadata["restored_key_shape"] = list(restored_key.shape)
                metadata["restored_value_shape"] = list(restored_value.shape)
                return RuntimeKVRestorationResult(restored_key, restored_value, "ok", None, metadata)
            except Exception as exc:
                status = "nan_or_inf" if _is_nonfinite_error_message(exc) else "restore_error"
                return RuntimeKVRestorationResult(None, None, status, str(exc), metadata)
        if self.is_phase3c:
            return RuntimeKVRestorationResult(
                None,
                None,
                "restore_error",
                "phase3c_kv_final requires restore_from_hidden, not source K/V tensors",
                {"runtime_mode": PHASE3C_RUNTIME_MODE},
            )
        key = _pair_key(source_layer, target_layer)
        pair = self.pairs.get(key)
        if pair is None:
            return RuntimeKVRestorationResult(None, None, "missing_map", "missing pair map {}".format(key))
        try:
            restored_key = apply_transform(source_key, pair["k"])
            restored_value = apply_transform(source_value, pair["v"])
            if restored_key.shape != source_key.shape or restored_value.shape != source_value.shape:
                raise ValueError(
                    "restored shape mismatch key={} value={} source_key={} source_value={}".format(
                        list(restored_key.shape),
                        list(restored_value.shape),
                        list(source_key.shape),
                        list(source_value.shape),
                    )
                )
            if not torch.isfinite(restored_key).all() or not torch.isfinite(restored_value).all():
                raise ValueError("restored K/V contains NaN or Inf")
            return RuntimeKVRestorationResult(restored_key, restored_value, "ok", None)
        except Exception as exc:
            status = "nan_or_inf" if _is_nonfinite_error_message(exc) else "restore_error"
            return RuntimeKVRestorationResult(None, None, status, str(exc))

    def _device_local_parameters(
        self,
        kind: str,
        identity: str,
        device: Any,
        loader,
        *,
        validate_finite: bool = True,
    ) -> Mapping[str, Any]:
        """Return the fitted Phase-3c parameter mapping for ``identity``,
        already materialized on ``device``.

        First access for a given (kind, identity, device) converts the CPU
        artifact tensors to the float32 form the apply_* helpers require and
        moves them to the device once. Later accesses return that same
        mapping, so ``_as_float_tensor(...)`` (already float32 -> returns the
        same object) and ``.to(device=...)`` (already resident -> returns the
        same object) inside those helpers become no-ops instead of transfers.
        The mathematical definitions are therefore untouched.

        The cached device is re-verified on every hit: a mapping is only
        reused if every tensor in it still reports the requested device, so a
        stale or wrong-device entry can never be handed out.

        Finiteness is validated exactly ONCE, at materialization, BEFORE the
        mapping is inserted into the cache -- and membership in
        ``_device_parameter_validated`` (not mere cache presence) is the
        trust marker: anything validated is immutable for inference, so
        validating callers may skip revalidating it (each such check is a
        ``.item()`` device synchronization). A mapping that fails validation
        is never cached and the failure propagates as a nonfinite
        ValueError, exactly as the generic path would have raised it.

        ``validate_finite=False`` is the EXPLICIT Batched timing mode only
        (kv_runtime_restoration_finite_validation_enabled=False): the
        mapping is materialized without the one-time NaN/Inf scan and is NOT
        marked validated. If a validation-ON caller later requests the same
        cached mapping, the scan runs then, before the mapping is treated as
        trusted -- an unvalidated mapping can never silently inherit trust.
        """

        device = torch.device(device) if not isinstance(device, torch.device) else device
        key = (str(kind), str(identity), str(device))
        cached = self._device_parameter_cache.get(key)
        if cached is not None and all(
            value.device == device
            for value in cached.values()
            if isinstance(value, torch.Tensor)
        ):
            if validate_finite and key not in self._device_parameter_validated:
                # Materialized earlier while validation was explicitly off;
                # this validating caller must not trust it unscanned.
                for name, value in cached.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                        raise ValueError(
                            "phase3c {}.{} for {} must contain only finite values".format(
                                kind, name, identity
                            )
                        )
                self._device_parameter_validated.add(key)
            return cached

        source = loader()
        materialized = {}
        for name, value in dict(source).items():
            if not isinstance(value, torch.Tensor):
                materialized[name] = value
                continue
            tensor = value.detach().to(device=device, dtype=torch.float32)
            # The one and only finite check for this parameter on this
            # device. It runs BEFORE the mapping becomes reusable, so an
            # invalid parameter can never be cached and then trusted.
            # Skipped ONLY in the explicit timing mode, which also leaves
            # the mapping unmarked so it can never be trusted later without
            # this scan.
            if validate_finite and not torch.isfinite(tensor).all().item():
                raise ValueError(
                    "phase3c {}.{} for {} must contain only finite values".format(kind, name, identity)
                )
            materialized[name] = tensor
        self._device_parameter_cache[key] = materialized
        if validate_finite:
            self._device_parameter_validated.add(key)
        else:
            self._device_parameter_validated.discard(key)
        return materialized

    def clear_device_parameter_cache(self) -> None:
        """Drop every device-local parameter copy. Purely a lifecycle helper
        (tests, device changes); the fitted values themselves are unaffected
        because the cache is only ever built FROM the artifact."""

        self._device_parameter_cache.clear()
        self._device_parameter_validated.clear()
        self._stacked_learned_parameter_bank = None
        self._stacked_learned_bank_fast_reuse = None
        self._stacked_exit_hidden_projection_bank_slot = None

    @staticmethod
    def _stacked_module_weight_identity(weight: torch.Tensor) -> Tuple[Any, ...]:
        """Cheap identity/staleness key for one model weight tensor. A
        replaced tensor changes id/data_ptr, an in-place mutation (including
        load_state_dict's copy_) bumps _version, and a device/dtype move
        changes the remaining fields -- so a stale stacked bank can never be
        handed back after any relevant model state change."""

        return (
            id(weight),
            int(weight._version),
            weight.data_ptr(),
            str(weight.device),
            str(weight.dtype),
        )

    @staticmethod
    def _stacked_learned_module_support(
        target_layer_norms,
        key_projections,
        value_projections,
    ):
        """Return (module_device, module_dtype, variance_epsilon) when the
        supplied modules are the exact T5 form the stacked arithmetic
        vectorizes -- T5 RMSNorm (weight + variance_epsilon, no mean
        subtraction) and bias-free Linear K/V projections, all with one
        uniform device/dtype. Returns None otherwise (e.g. synthetic
        scheduling fixtures with a plain nn.LayerNorm), in which case the
        caller keeps the generic sequential restore_from_hidden() path."""

        weights = []
        epsilons = []
        for norm in target_layer_norms:
            epsilon = getattr(norm, "variance_epsilon", None)
            weight = getattr(norm, "weight", None)
            if epsilon is None or not isinstance(weight, torch.Tensor) or weight.ndim != 1:
                return None
            epsilons.append(float(epsilon))
            weights.append(weight)
        for module in list(key_projections) + list(value_projections):
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                return None
            if getattr(module, "bias", None) is not None:
                return None
            weights.append(weight)
        if not weights or len(set(epsilons)) != 1:
            return None
        devices = {weight.device for weight in weights}
        dtypes = {weight.dtype for weight in weights}
        if len(devices) != 1 or len(dtypes) != 1:
            return None
        return next(iter(devices)), next(iter(dtypes)), epsilons[0]

    def _stacked_learned_target_bank(
        self,
        source_layer: int,
        target_layers,
        target_layer_norms,
        key_projections,
        value_projections,
        module_device: torch.device,
        module_dtype: torch.dtype,
        variance_epsilon: float,
        validate_fitted_parameters_finite: bool = True,
    ) -> Mapping[str, Any]:
        """Return the lazy single-slot stacked parameter bank, rebuilding it
        only when the fingerprint (targets, device, dtype, epsilon, actual
        model weight identity/version) no longer matches. Fitted Phase-3c
        parameters are materialized through the SAME _device_local_parameters
        cache the sequential path uses (moved to the device once) and stacked
        here exactly once -- never re-transferred and never re-stacked per
        flush.

        The cached slot also records whether its fitted parameters were
        finite-validated at build time. A bank built in the explicit timing
        mode (validation off) is NOT reused by a later validation-ON request
        in the same process: that request rebuilds through
        _device_local_parameters with validation on, which scans the cached
        (unvalidated) mappings before trusting them."""

        norm_list = list(target_layer_norms)
        key_list = list(key_projections)
        value_list = list(value_projections)
        fingerprint = (
            self.threshold_key,
            int(source_layer),
            tuple(int(layer) for layer in target_layers),
            str(module_device),
            str(module_dtype),
            float(variance_epsilon),
            tuple(
                self._stacked_module_weight_identity(module.weight)
                for module in norm_list + key_list + value_list
            ),
        )
        cached = self._stacked_learned_parameter_bank
        if (
            cached is not None
            and cached[0] == fingerprint
            and (cached[2] or not validate_fitted_parameters_finite)
        ):
            return cached[1]
        bank = self._build_stacked_learned_target_bank(
            int(source_layer),
            [int(layer) for layer in target_layers],
            norm_list,
            key_list,
            value_list,
            module_device,
            module_dtype,
            float(variance_epsilon),
            validate_fitted_parameters_finite=bool(validate_fitted_parameters_finite),
        )
        self._stacked_learned_parameter_bank = (
            fingerprint,
            bank,
            bool(validate_fitted_parameters_finite),
        )
        return bank

    def _build_stacked_learned_target_bank(
        self,
        source_layer: int,
        target_layers,
        target_layer_norms,
        key_projections,
        value_projections,
        module_device: torch.device,
        module_dtype: torch.dtype,
        variance_epsilon: float,
        validate_fitted_parameters_finite: bool = True,
    ) -> Dict[str, Any]:
        threshold_key = threshold_to_key(self.threshold)
        hidden_params = []
        k_params = []
        v_params = []
        for target_layer in target_layers:
            pair_identity = "{}:{}->{}".format(threshold_key, source_layer, target_layer)
            hidden_params.append(
                self._device_local_parameters(
                    "hidden",
                    pair_identity,
                    module_device,
                    lambda target_layer=target_layer: get_hidden_layer_pair_parameters(
                        self.artifact, self.threshold, source_layer, target_layer
                    ),
                    validate_finite=validate_fitted_parameters_finite,
                )
            )
            k_params.append(
                self._device_local_parameters(
                    "k",
                    pair_identity,
                    module_device,
                    lambda target_layer=target_layer: get_k_layer_pair_parameters(
                        self.artifact, self.threshold, source_layer, target_layer
                    ),
                    validate_finite=validate_fitted_parameters_finite,
                )
            )
            gap_bin = self.select_phase3c_gap_bin(source_layer, target_layer)
            v_params.append(
                self._device_local_parameters(
                    "v",
                    "{}:{}".format(threshold_key, gap_bin),
                    module_device,
                    lambda gap_bin=gap_bin: get_v_gap_bin_parameters(
                        self.artifact, self.threshold, gap_bin
                    ),
                    validate_finite=validate_fitted_parameters_finite,
                )
            )
        num_heads = int(self.model_spec["num_heads"])
        d_kv = int(self.model_spec["d_kv"])
        # torch.stack copies, so the bank is an immutable inference-only
        # snapshot; staleness against later model mutation is handled by the
        # weight-identity fingerprint above.
        return {
            "target_count": len(target_layers),
            "num_heads": num_heads,
            "d_kv": d_kv,
            "module_dtype": module_dtype,
            "variance_epsilon": float(variance_epsilon),
            "hidden_scale": torch.stack([entry["scale"] for entry in hidden_params], dim=0).contiguous(),
            "hidden_bias": torch.stack([entry["bias"] for entry in hidden_params], dim=0).contiguous(),
            "norm_weights": torch.stack(
                [norm.weight.detach() for norm in target_layer_norms], dim=0
            ).contiguous(),
            "k_weights": torch.stack(
                [module.weight.detach() for module in key_projections], dim=0
            ).contiguous(),
            "v_weights": torch.stack(
                [module.weight.detach() for module in value_projections], dim=0
            ).contiguous(),
            "k_scale": torch.stack([entry["scale"] for entry in k_params], dim=0).contiguous(),
            "k_bias": torch.stack([entry["bias"] for entry in k_params], dim=0).contiguous(),
            "v_rotations": torch.stack([entry["rotations"] for entry in v_params], dim=0).contiguous(),
            "v_biases": torch.stack([entry["biases"] for entry in v_params], dim=0).contiguous(),
        }

    def restore_learned_targets_from_hidden_stacked(
        self,
        source_hidden: torch.Tensor,
        source_layer: int,
        target_layers,
        target_layer_norms,
        key_projections,
        value_projections,
        *,
        output_device: Optional[torch.device] = None,
        output_dtype: Optional[torch.dtype] = None,
        source_hidden_prevalidated: bool = False,
        defer_output_finite_validation: bool = False,
        validate_fitted_parameters_finite: bool = True,
    ) -> RuntimeKVRestorationResult:
        """Restore ALL strictly-deeper learned target layers from one source
        hidden block in ONE stacked operation (the frozen separate-K +
        separate-V ``stacked_bmm`` candidate).

        ``validate_fitted_parameters_finite`` (default True) controls the
        one-time NaN/Inf validation of fitted Phase-3c parameters at
        device materialization / bank build. It is set False ONLY by the
        Batched Task C2 timing mode
        (kv_runtime_restoration_finite_validation_enabled=False), so even a
        cold first stacked restoration performs zero diagnostic finite
        scans. Parameters materialized unvalidated are never later treated
        as validated: a subsequent validation-ON request scans them first.

        Returns restored_key/restored_value BANKS of shape
        [target_count, num_heads, seq_len, d_kv]; index i corresponds to
        ``target_layers[i]``, so for the Native FREE source-6 protocol index
        0 -> target 7 ... index 16 -> target 23. The same-layer target
        (target == source_layer) is deliberately REJECTED here: it has no
        learned map and must stay on the existing native
        restore_from_hidden() path.

        The two flags have exactly the semantics documented on
        restore_from_hidden(). The final K/V banks are cast to
        output_device/output_dtype BEFORE the (optionally deferred) finite
        validation, so a value that is finite in float32 but overflows in a
        narrower cache dtype is still caught."""

        metadata: Dict[str, Any] = {
            "runtime_mode": self.runtime_mode,
            "restoration_method": self.method,
            "threshold": self.threshold,
            "threshold_key": self.threshold_key,
            "source_layer": int(source_layer),
            "target_layers": [int(layer) for layer in target_layers],
            "restoration_submode": "phase3c_stacked_learned_targets",
            "stacked_candidate": "stacked_bmm_separate_k_v",
            "output_finite_validation_deferred": bool(defer_output_finite_validation),
        }
        try:
            if self.method != PHASE3C_RUNTIME_RESTORATION_METHOD:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "restore_error",
                    "stacked learned-target restoration is only supported for phase3c_kv_final",
                    metadata,
                )
            source_layer = int(source_layer)
            target_layers = [int(layer) for layer in target_layers]
            norm_list = list(target_layer_norms)
            key_list = list(key_projections)
            value_list = list(value_projections)
            if not target_layers:
                return RuntimeKVRestorationResult(
                    None, None, "restore_error", "target_layers must be non-empty", metadata
                )
            if len(norm_list) != len(target_layers) or len(key_list) != len(target_layers) or len(
                value_list
            ) != len(target_layers):
                return RuntimeKVRestorationResult(
                    None, None, "restore_error", "target module list length mismatch", metadata
                )
            if any(layer <= source_layer for layer in target_layers):
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "restore_error",
                    "stacked learned targets must be strictly deeper than source layer {} "
                    "(the same-layer target stays on the native restore_from_hidden path)".format(
                        source_layer
                    ),
                    metadata,
                )
            if sorted(target_layers) != target_layers or len(set(target_layers)) != len(target_layers):
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "restore_error",
                    "target_layers must be strictly increasing and unique",
                    metadata,
                )
            if source_hidden.ndim != 3:
                raise ValueError(
                    "source_hidden must have shape [batch, seq_len, d_model], got {}".format(
                        tuple(source_hidden.shape)
                    )
                )
            if int(source_hidden.shape[0]) != 1:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "unsupported_batch",
                    "{} currently supports batch size 1 only".format(self.method),
                    metadata,
                )
            if int(source_hidden.shape[1]) < 1:
                raise ValueError(
                    "source_hidden must contain at least one token, got seq_len={}".format(
                        int(source_hidden.shape[1])
                    )
                )
            if self.preliminary_source6_subset_enabled and source_layer != 6:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "preliminary_source6_subset_rejected_source_layer",
                    "preliminary source-6 subset runtime only accepts source_layer=6, got {}".format(
                        source_layer
                    ),
                    metadata,
                )
            if not source_hidden_prevalidated and not torch.isfinite(source_hidden).all().item():
                return RuntimeKVRestorationResult(
                    None, None, "nan_or_inf", "source_hidden contains NaN or Inf", metadata
                )
            # Explicit timing-mode static-invariant hoist
            # (validate_fitted_parameters_finite=False, the SAME flag that
            # already gates every other diagnostic validation of immutable
            # state on this path): the per-flush module-support scan and the
            # full per-weight bank staleness fingerprint validate IMMUTABLE
            # module/parameter state and are re-established here once, on
            # first use, then reused while the caller keeps handing back the
            # very same module objects and the very same bank object. The
            # reuse key is (source, targets, module ids) plus an IDENTITY
            # check against the cached bank object, so a rebuilt/replaced
            # bank or any different module list can never be fast-reused.
            # Validation-ON calls never touch this path: they re-run the
            # full support scan and weight fingerprint every call, exactly
            # as reviewed (in-place weight mutation detection included).
            fast_reuse_support = None
            module_identity_key = None
            if not validate_fitted_parameters_finite:
                module_identity_key = (
                    source_layer,
                    tuple(target_layers),
                    tuple(id(module) for module in norm_list + key_list + value_list),
                )
                cached_fast = self._stacked_learned_bank_fast_reuse
                cached_bank_slot = self._stacked_learned_parameter_bank
                if (
                    cached_fast is not None
                    and cached_bank_slot is not None
                    and cached_fast[0] == module_identity_key
                    and cached_fast[2] is cached_bank_slot[1]
                ):
                    fast_reuse_support = cached_fast[1]
            if fast_reuse_support is not None:
                module_device, module_dtype, variance_epsilon = fast_reuse_support
                bank = self._stacked_learned_parameter_bank[1]
            else:
                support = self._stacked_learned_module_support(norm_list, key_list, value_list)
                if support is None:
                    return RuntimeKVRestorationResult(
                        None,
                        None,
                        STACKED_LEARNED_TARGETS_UNSUPPORTED_STATUS,
                        "target modules are not the T5 RMSNorm / bias-free Linear form the "
                        "stacked learned-target arithmetic vectorizes",
                        metadata,
                    )
                module_device, module_dtype, variance_epsilon = support
                bank = self._stacked_learned_target_bank(
                    source_layer,
                    target_layers,
                    norm_list,
                    key_list,
                    value_list,
                    module_device,
                    module_dtype,
                    variance_epsilon,
                    validate_fitted_parameters_finite=bool(validate_fitted_parameters_finite),
                )
                if module_identity_key is not None:
                    # Timing mode only: arm fast reuse for the NEXT flush,
                    # tied to this exact bank object.
                    self._stacked_learned_bank_fast_reuse = (module_identity_key, support, bank)
            output_device = output_device or module_device
            output_dtype = output_dtype or module_dtype
            restored_key, restored_value = _stacked_bmm_learned_restore(
                source_hidden.to(device=module_device), bank
            )
            metadata["restored_key_shape"] = list(restored_key.shape)
            metadata["restored_value_shape"] = list(restored_value.shape)
            # Same output contract as the single-target path: cast to the
            # output/cache dtype FIRST, then (unless the batched transaction
            # defers to its aggregate) finite-validate the exact post-cast
            # values that are destined for the cache.
            restored_key = restored_key.to(device=output_device, dtype=output_dtype)
            restored_value = restored_value.to(device=output_device, dtype=output_dtype)
            if not defer_output_finite_validation and (
                not torch.isfinite(restored_key).all() or not torch.isfinite(restored_value).all()
            ):
                raise ValueError("{} restored K/V contains NaN or Inf".format(self.method))
            return RuntimeKVRestorationResult(restored_key, restored_value, "ok", None, metadata)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("missing_artifact_threshold"):
                status = "missing_threshold"
            elif message.startswith("missing_hidden_layer_pair"):
                status = "missing_hidden_pair"
            elif message.startswith("missing_k_layer_pair"):
                status = "missing_k_pair"
            elif message.startswith("missing_v_gap_bin"):
                status = "missing_v_gap"
            elif _is_nonfinite_error_message(message):
                status = "nan_or_inf"
            else:
                status = "restore_error"
            return RuntimeKVRestorationResult(None, None, status, message, metadata)
        except Exception as exc:  # pragma: no cover - defensive
            return RuntimeKVRestorationResult(None, None, "restore_error", str(exc), metadata)

    def _stacked_exit_hidden_projection_bank(
        self,
        target_layers,
        target_layer_norms,
        key_projections,
        value_projections,
        module_device: torch.device,
        module_dtype: torch.dtype,
        variance_epsilon: float,
    ) -> Mapping[str, Any]:
        """Single-slot stacked WEIGHT bank for the exit-hidden (State
        Copying) target projection fairness control: per-target T5 RMSNorm
        weights and W_K/W_V, stacked once and reused while the
        weight-identity fingerprint matches (same staleness rule as the
        learned bank). Contains NO fitted parameters -- State Copying has
        none -- so there is nothing to finite-validate beyond the model
        weights themselves."""

        norm_list = list(target_layer_norms)
        key_list = list(key_projections)
        value_list = list(value_projections)
        fingerprint = (
            tuple(int(layer) for layer in target_layers),
            str(module_device),
            str(module_dtype),
            float(variance_epsilon),
            tuple(
                self._stacked_module_weight_identity(module.weight)
                for module in norm_list + key_list + value_list
            ),
        )
        cached = getattr(self, "_stacked_exit_hidden_projection_bank_slot", None)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        bank = {
            "target_count": len(norm_list),
            "num_heads": int(self.model_spec["num_heads"]),
            "d_kv": int(self.model_spec["d_kv"]),
            "module_dtype": module_dtype,
            "variance_epsilon": float(variance_epsilon),
            "norm_weights": torch.stack(
                [norm.weight.detach() for norm in norm_list], dim=0
            ).contiguous(),
            "k_weights": torch.stack(
                [module.weight.detach() for module in key_list], dim=0
            ).contiguous(),
            "v_weights": torch.stack(
                [module.weight.detach() for module in value_list], dim=0
            ).contiguous(),
        }
        self._stacked_exit_hidden_projection_bank_slot = (fingerprint, bank)
        return bank

    def restore_exit_hidden_targets_stacked(
        self,
        source_hidden: torch.Tensor,
        source_layer: int,
        target_layers,
        target_layer_norms,
        key_projections,
        value_projections,
        *,
        output_device: Optional[torch.device] = None,
        output_dtype: Optional[torch.dtype] = None,
        source_hidden_prevalidated: bool = False,
        defer_output_finite_validation: bool = False,
    ) -> RuntimeKVRestorationResult:
        """Target-wise VECTORIZED conventional State Copying projection (the
        timing-fairness control arm). For every supplied target t -- the
        same-layer target t == source IS allowed, State Copying treats it
        identically to every deeper target --

            h_s -> target-t T5 RMSNorm -> target W_K_t / W_V_t
            -> [target, head, token, d_kv]

        executed as ONE stacked operation, reusing the frozen
        ``stacked_bmm`` primitives' projection portion (_stacked_t5_rms_norm
        plus the separate-K/separate-V bmm orientation). Deliberately NO
        hidden diagonal affine, NO K affine correction, NO V Procrustes:
        this is NOT Phase-3c, and it must remain numerically equivalent to
        running the existing sequential exit_hidden_target_projection
        restore_from_hidden() per target, which stays the semantic
        authority. Returns restored_key/restored_value BANKS of shape
        [target_count, num_heads, seq_len, d_kv]; index i corresponds to
        ``target_layers[i]``."""

        metadata: Dict[str, Any] = {
            "runtime_mode": self.runtime_mode,
            "restoration_method": self.method,
            "threshold": self.threshold,
            "threshold_key": self.threshold_key,
            "source_layer": int(source_layer),
            "target_layers": [int(layer) for layer in target_layers],
            "restoration_submode": "exit_hidden_target_projection_stacked",
            "stacked_candidate": "stacked_bmm_separate_k_v_projection_only",
            "hidden_affine_applied": False,
            "k_correction_applied": False,
            "v_correction_applied": False,
            "output_finite_validation_deferred": bool(defer_output_finite_validation),
        }
        try:
            if self.method != EXIT_HIDDEN_TARGET_PROJECTION_METHOD:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "restore_error",
                    "stacked exit-hidden target projection is only supported for "
                    "exit_hidden_target_projection",
                    metadata,
                )
            source_layer = int(source_layer)
            target_layers = [int(layer) for layer in target_layers]
            norm_list = list(target_layer_norms)
            key_list = list(key_projections)
            value_list = list(value_projections)
            if not target_layers:
                return RuntimeKVRestorationResult(
                    None, None, "restore_error", "target_layers must be non-empty", metadata
                )
            if len(norm_list) != len(target_layers) or len(key_list) != len(target_layers) or len(
                value_list
            ) != len(target_layers):
                return RuntimeKVRestorationResult(
                    None, None, "restore_error", "target module list length mismatch", metadata
                )
            if any(layer < source_layer for layer in target_layers):
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "target_before_source",
                    "exit-hidden stacked targets must not be shallower than source layer {}".format(
                        source_layer
                    ),
                    metadata,
                )
            if sorted(target_layers) != target_layers or len(set(target_layers)) != len(target_layers):
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "restore_error",
                    "target_layers must be strictly increasing and unique",
                    metadata,
                )
            if source_hidden.ndim != 3:
                raise ValueError(
                    "source_hidden must have shape [batch, seq_len, d_model], got {}".format(
                        tuple(source_hidden.shape)
                    )
                )
            if int(source_hidden.shape[0]) != 1:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "unsupported_batch",
                    "{} currently supports batch size 1 only".format(self.method),
                    metadata,
                )
            if int(source_hidden.shape[1]) < 1:
                raise ValueError(
                    "source_hidden must contain at least one token, got seq_len={}".format(
                        int(source_hidden.shape[1])
                    )
                )
            if not source_hidden_prevalidated and not torch.isfinite(source_hidden).all().item():
                return RuntimeKVRestorationResult(
                    None, None, "nan_or_inf", "source_hidden contains NaN or Inf", metadata
                )
            support = self._stacked_learned_module_support(norm_list, key_list, value_list)
            if support is None:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    STACKED_LEARNED_TARGETS_UNSUPPORTED_STATUS,
                    "target modules are not the T5 RMSNorm / bias-free Linear form the "
                    "stacked arithmetic vectorizes",
                    metadata,
                )
            module_device, module_dtype, variance_epsilon = support
            bank = self._stacked_exit_hidden_projection_bank(
                target_layers,
                norm_list,
                key_list,
                value_list,
                module_device,
                module_dtype,
                variance_epsilon,
            )
            output_device = output_device or module_device
            output_dtype = output_dtype or module_dtype
            # Identical per-target arithmetic to the sequential path, just
            # target-stacked: the shared h_s (no hidden affine) through the
            # per-target RMSNorm weights, then the separate-K/separate-V
            # bmm (same nn.Linear weight orientation note as
            # _stacked_bmm_learned_restore), then the same
            # [T, N, H*d_kv] -> [T, H, N, d_kv] layout as
            # _stacked_downstream_corrections' regeneration step -- with no
            # downstream corrections.
            hidden = source_hidden.to(device=module_device, dtype=module_dtype)
            projected = _stacked_t5_rms_norm(
                hidden, bank["norm_weights"], bank["variance_epsilon"]
            )
            key_flat = torch.bmm(projected, bank["k_weights"].transpose(1, 2))
            value_flat = torch.bmm(projected, bank["v_weights"].transpose(1, 2))
            target_count = int(bank["target_count"])
            num_heads = int(bank["num_heads"])
            d_kv = int(bank["d_kv"])
            seq_len = int(key_flat.shape[1])
            restored_key = (
                key_flat.reshape(target_count, seq_len, num_heads, d_kv)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            restored_value = (
                value_flat.reshape(target_count, seq_len, num_heads, d_kv)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            metadata["restored_key_shape"] = list(restored_key.shape)
            metadata["restored_value_shape"] = list(restored_value.shape)
            restored_key = restored_key.to(device=output_device, dtype=output_dtype)
            restored_value = restored_value.to(device=output_device, dtype=output_dtype)
            if not defer_output_finite_validation and (
                not torch.isfinite(restored_key).all() or not torch.isfinite(restored_value).all()
            ):
                raise ValueError("{} restored K/V contains NaN or Inf".format(self.method))
            return RuntimeKVRestorationResult(restored_key, restored_value, "ok", None, metadata)
        except ValueError as exc:
            message = str(exc)
            status = "nan_or_inf" if _is_nonfinite_error_message(message) else "restore_error"
            return RuntimeKVRestorationResult(None, None, status, message, metadata)
        except Exception as exc:  # pragma: no cover - defensive
            return RuntimeKVRestorationResult(None, None, "restore_error", str(exc), metadata)

    def restore_from_hidden(
        self,
        source_hidden: torch.Tensor,
        source_layer: int,
        target_layer: int,
        target_layer_norm: torch.nn.Module,
        key_projection: torch.nn.Module,
        value_projection: torch.nn.Module,
        *,
        output_device: Optional[torch.device] = None,
        output_dtype: Optional[torch.dtype] = None,
        source_hidden_prevalidated: bool = False,
        defer_output_finite_validation: bool = False,
    ) -> RuntimeKVRestorationResult:
        """Restore target-layer K/V from a source hidden state.

        ``source_hidden_prevalidated`` defaults to False, so Task C1,
        diagnostics, offline uses, tests and any external caller keep the
        existing finite validation of ``source_hidden`` unchanged.

        A caller may pass True ONLY after it has itself finite-validated this
        exact tensor. Both Task C2 helpers do: they validate once per
        transaction (Immediate: the one exiting token's h6; Batched: the
        concatenated [1, N, d_model] pending block) before their target-layer
        loop, then reuse that same unmodified tensor for all 18 targets.
        Revalidating it per target layer costs a ``.item()`` device
        synchronization each time and can never find anything new. Shape,
        batch, dtype/device and all newly computed output checks are
        unaffected by this flag.

        ``defer_output_finite_validation`` also defaults to False, so every
        existing caller keeps the per-call intermediate and output finite
        checks exactly as before. Only the BATCHED Task C2 transaction sets it
        True: that caller runs one aggregate finite check over the final
        candidate K/V it is about to commit, which is strictly downstream of
        every intermediate this flag skips. The restoration math, shapes,
        dtypes, devices and returned tensors are byte-identical either way --
        only WHEN a nonfinite value is reported changes, never WHETHER.

        A result produced with this flag set is marked
        ``metadata["output_finite_validation_deferred"] = True`` so it is
        self-describing: any caller that has NOT performed the aggregate
        check can see that this result is not yet finite-verified.
        """

        if not self.uses_hidden_projection:
            return RuntimeKVRestorationResult(
                None,
                None,
                "restore_error",
                "restore_from_hidden is only supported for phase3c_kv_final or exit_hidden_target_projection",
                None,
            )
        metadata: Dict[str, Any] = {
            "runtime_mode": self.runtime_mode,
            "restoration_method": self.method,
            "threshold": self.threshold,
            "threshold_key": self.threshold_key,
            "source_layer": int(source_layer),
            "target_layer": int(target_layer),
            "gap": int(target_layer) - int(source_layer),
            "gap_bin": None,
            "source_hidden_shape": list(source_hidden.shape) if hasattr(source_hidden, "shape") else None,
            "regenerated_key_shape": None,
            "regenerated_value_shape": None,
            "restored_key_shape": None,
            "restored_value_shape": None,
            "restoration_submode": None,
            "hidden_affine_applied": None,
            "k_correction_applied": None,
            "v_correction_applied": None,
            "calibration_source_mode": self.calibration_source_mode,
            "calibration_fixed_source_layer": self.calibration_fixed_source_layer,
            "runtime_source_mode": self.runtime_source_mode,
            "runtime_fixed_source_layer": self.runtime_fixed_source_layer,
            "calibration_layer_scope_match": self.calibration_layer_scope_match,
            "calibration_runtime_scope_match": self.calibration_runtime_scope_match,
            "confidence_semantics_verified": self.confidence_semantics_verified,
            "fixed_layer_membership_semantics_verified": self.fixed_layer_membership_semantics_verified,
            "generation_quality_claim_valid": self.generation_quality_claim_valid,
            "artifact_runtime_compatible": self.artifact_runtime_compatible,
            "artifact_paper_population_match": self.artifact_paper_population_match,
            "artifact_use_classification": self.artifact_use_classification,
            "paper_facing_artifact_valid": self.paper_facing_artifact_valid,
            "preliminary_source6_subset_enabled": self.preliminary_source6_subset_enabled,
            "exact_catchup_already_computed": True,
            "speed_claim_valid": False,
        }
        try:
            if source_hidden.ndim != 3:
                raise ValueError("source_hidden must have shape [batch, seq_len, d_model], got {}".format(tuple(source_hidden.shape)))
            if int(source_hidden.shape[0]) != 1:
                return RuntimeKVRestorationResult(None, None, "unsupported_batch", "{} currently supports batch size 1 only".format(self.method), metadata)
            if int(source_hidden.shape[1]) < 1:
                # Every stage of the restoration below is strictly
                # position-wise, so an N-token source hidden produces exactly
                # the concatenation of the N single-token results:
                #   apply_hidden_diagonal_affine  h * scale + bias, broadcast over [B,T,D]
                #   target LayerNorm              normalizes the last dim per position
                #   target K/V projections        nn.Linear, per position
                #   view/transpose                uses the actual seq_len
                #   apply_k_head_channel_affine   scale/bias broadcast over [B,H,T,D]
                #   apply_v_headwise_procrustes   einsum bhtd,hde->bhte, per position
                # None of them mixes positions, and no fitted parameter or
                # mathematical definition changes. Only the historical
                # one-token-only guard is relaxed here, so the FREE-aligned
                # lazy batched flush can restore all N pending early-exit
                # tokens in a single call per target layer.
                raise ValueError("source_hidden must contain at least one token, got seq_len={}".format(int(source_hidden.shape[1])))
            # Skipped only when the caller has already finite-validated this
            # exact tensor for this transaction (see
            # source_hidden_prevalidated above); every other caller keeps it.
            if not source_hidden_prevalidated and not torch.isfinite(source_hidden).all().item():
                return RuntimeKVRestorationResult(None, None, "nan_or_inf", "source_hidden contains NaN or Inf", metadata)
            source_layer = int(source_layer)
            target_layer = int(target_layer)
            if self.preliminary_source6_subset_enabled and source_layer != 6:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "preliminary_source6_subset_rejected_source_layer",
                    "preliminary source-6 subset runtime only accepts source_layer=6, got {}".format(
                        source_layer
                    ),
                    metadata,
                )
            if target_layer < source_layer:
                return RuntimeKVRestorationResult(
                    None,
                    None,
                    "target_before_source",
                    "target_layer {} is shallower than source hidden layer {}".format(target_layer, source_layer),
                    metadata,
                )

            param = next(key_projection.parameters(), None)
            if param is None:
                param = next(value_projection.parameters(), None)
            module_device = param.device if param is not None else source_hidden.device
            module_dtype = param.dtype if param is not None else source_hidden.dtype
            output_device = output_device or module_device
            output_dtype = output_dtype or module_dtype

            if self.method == EXIT_HIDDEN_TARGET_PROJECTION_METHOD:
                metadata.update(
                    {
                        "restoration_submode": EXIT_HIDDEN_TARGET_PROJECTION_METHOD,
                        "hidden_affine_applied": False,
                        "k_correction_applied": False,
                        "v_correction_applied": False,
                        "gap_bin": "artifact_free_projection",
                    }
                )
                hidden_hat = source_hidden.to(device=module_device, dtype=module_dtype)
            elif target_layer == source_layer:
                metadata.update(
                    {
                        "restoration_submode": "same_layer_exact_projection",
                        "hidden_affine_applied": False,
                        "k_correction_applied": False,
                        "v_correction_applied": False,
                    }
                )
                hidden_hat = source_hidden.to(device=module_device, dtype=module_dtype)
            else:
                # Same fitted parameters as before, just materialized once
                # per device instead of transferred on every call. The V
                # gap-bin entry is shared by every target layer in that bin,
                # so it is keyed by the bin and reused across them.
                threshold_key = threshold_to_key(self.threshold)
                pair_identity = "{}:{}->{}".format(threshold_key, int(source_layer), int(target_layer))
                hidden_params = self._device_local_parameters(
                    "hidden",
                    pair_identity,
                    module_device,
                    lambda: get_hidden_layer_pair_parameters(
                        self.artifact, self.threshold, source_layer, target_layer
                    ),
                )
                k_params = self._device_local_parameters(
                    "k",
                    pair_identity,
                    module_device,
                    lambda: get_k_layer_pair_parameters(
                        self.artifact, self.threshold, source_layer, target_layer
                    ),
                )
                gap_bin = self.select_phase3c_gap_bin(source_layer, target_layer)
                metadata["gap_bin"] = gap_bin
                v_params = self._device_local_parameters(
                    "v",
                    "{}:{}".format(threshold_key, gap_bin),
                    module_device,
                    lambda: get_v_gap_bin_parameters(self.artifact, self.threshold, gap_bin),
                )
                metadata.update(
                    {
                        "restoration_submode": "phase3c_artifact_restoration",
                        "hidden_affine_applied": True,
                        "k_correction_applied": True,
                        "v_correction_applied": True,
                    }
                )
                hidden_hat = apply_hidden_diagonal_affine(
                    source_hidden.to(device=module_device),
                    hidden_params,
                    # The source hidden was validated once by the caller for
                    # this whole transaction; the fitted parameters were
                    # validated once when they entered the device cache.
                    # Re-checking either here would only re-synchronize.
                    validate_input_finite=not source_hidden_prevalidated,
                    validate_parameter_finite=False,
                    validate_output_finite=not defer_output_finite_validation,
                )
                hidden_hat = hidden_hat.to(device=module_device, dtype=module_dtype)
            normed_hidden = target_layer_norm(hidden_hat)
            key_flat = key_projection(normed_hidden)
            value_flat = value_projection(normed_hidden)
            batch, seq_len, _ = key_flat.shape
            num_heads = int(self.model_spec["num_heads"])
            d_kv = int(self.model_spec["d_kv"])
            if key_flat.shape[-1] != num_heads * d_kv:
                raise ValueError(
                    "target K projection output dim mismatch: expected {}, got {}".format(
                        num_heads * d_kv,
                        int(key_flat.shape[-1]),
                    )
                )
            if value_flat.shape[-1] != num_heads * d_kv:
                raise ValueError(
                    "target V projection output dim mismatch: expected {}, got {}".format(
                        num_heads * d_kv,
                        int(value_flat.shape[-1]),
                    )
                )
            key_regenerated = key_flat.view(batch, seq_len, num_heads, d_kv).transpose(1, 2).contiguous()
            value_regenerated = value_flat.view(batch, seq_len, num_heads, d_kv).transpose(1, 2).contiguous()
            metadata["regenerated_key_shape"] = list(key_regenerated.shape)
            metadata["regenerated_value_shape"] = list(value_regenerated.shape)

            if target_layer == source_layer or self.method == EXIT_HIDDEN_TARGET_PROJECTION_METHOD:
                restored_key = key_regenerated
                restored_value = value_regenerated
                if target_layer == source_layer and self.method == PHASE3C_RUNTIME_RESTORATION_METHOD:
                    metadata["gap_bin"] = "same_layer_exact_projection"
            else:
                # key_regenerated/value_regenerated are newly computed for
                # THIS target layer, so their input checks stay on for every
                # ordinary caller; only the immutable cached fitted parameters
                # skip revalidation. The batched Task C2 transaction defers
                # both the input and output checks to its single aggregate
                # validation of the final candidate K/V.
                restored_key = apply_k_head_channel_affine(
                    key_regenerated,
                    k_params,
                    validate_input_finite=not defer_output_finite_validation,
                    validate_parameter_finite=False,
                    validate_output_finite=not defer_output_finite_validation,
                )
                restored_value = apply_v_headwise_procrustes(
                    value_regenerated,
                    v_params,
                    validate_input_finite=not defer_output_finite_validation,
                    validate_parameter_finite=False,
                    validate_output_finite=not defer_output_finite_validation,
                )
            metadata["restored_key_shape"] = list(restored_key.shape)
            metadata["restored_value_shape"] = list(restored_value.shape)
            restored_key = restored_key.to(device=output_device, dtype=output_dtype)
            restored_value = restored_value.to(device=output_device, dtype=output_dtype)
            # Deferred ONLY for the batched Task C2 transaction, which runs one
            # aggregate finite check over these exact values after they have
            # been staged into the candidate cache -- i.e. still strictly
            # before any authoritative commit. The check stays here (after the
            # output cast) for every other caller, so a finite float32 value
            # that overflows to Inf in a narrower output_dtype is caught in
            # both modes.
            metadata["output_finite_validation_deferred"] = bool(defer_output_finite_validation)
            if not defer_output_finite_validation and (
                not torch.isfinite(restored_key).all() or not torch.isfinite(restored_value).all()
            ):
                raise ValueError("{} restored K/V contains NaN or Inf".format(self.method))
            return RuntimeKVRestorationResult(restored_key, restored_value, "ok", None, metadata)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("missing_artifact_threshold"):
                status = "missing_threshold"
            elif message.startswith("missing_hidden_layer_pair"):
                status = "missing_hidden_pair"
            elif message.startswith("missing_k_layer_pair"):
                status = "missing_k_pair"
            elif message.startswith("missing_v_gap_bin") or message.startswith("missing_v_gap_bin_for_gap"):
                status = "missing_v_gap"
            elif _is_nonfinite_error_message(message):
                status = "nan_or_inf"
            else:
                status = "restore_error"
            return RuntimeKVRestorationResult(None, None, status, message, metadata)
        except Exception as exc:
            message = str(exc)
            status = "nan_or_inf" if _is_nonfinite_error_message(message) else "restore_error"
            return RuntimeKVRestorationResult(None, None, status, message, metadata)


__all__ = [
    "ARTIFACT_FREE_CALM_TASKC1_METHODS",
    "CALM_TASKC1_RUNTIME_METHODS",
    "DIRECT_SHALLOW_KV_REUSE_METHOD",
    "EXACT_CATCHUP_METHOD",
    "EXIT_HIDDEN_TARGET_PROJECTION_METHOD",
    "HIDDEN_PROJECTION_RUNTIME_METHODS",
    "LEGACY_RUNTIME_RESTORATION_METHODS",
    "PHASE3C_RUNTIME_MODE",
    "PHASE3C_RUNTIME_RESTORATION_METHOD",
    "RuntimeKVRestorationManager",
    "RuntimeKVRestorationResult",
    "SUPPORTED_RUNTIME_RESTORATION_METHODS",
]
