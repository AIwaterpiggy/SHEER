def _shape_list(tensor):
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def gather_recompute_hidden_states(stack_hidden_states, recompute_relative_indices):
    """Gather pending hidden states by relative index without mutating the stack."""
    recompute_relative_indices = [int(idx) for idx in (recompute_relative_indices or [])]
    invalid_relative_indices = [
        idx
        for idx in recompute_relative_indices
        if idx < 0 or idx >= len(stack_hidden_states)
    ]
    if invalid_relative_indices:
        return None, invalid_relative_indices
    if not recompute_relative_indices:
        return None, []

    selected_hidden_states = tuple(stack_hidden_states[idx] for idx in recompute_relative_indices)
    try:
        import torch
    except Exception:  # pragma: no cover - torch is expected in this repo.
        return None, recompute_relative_indices
    return torch.cat(selected_hidden_states, dim=1).detach().clone(), []


def expected_self_attn_kv_shapes(hidden_states, num_heads, key_value_proj_dim, missing_layer_range):
    hidden_shape = _shape_list(hidden_states)
    if hidden_shape is None or len(hidden_shape) < 2:
        return []
    kv_shape = [hidden_shape[0], int(num_heads), hidden_shape[1], int(key_value_proj_dim)]
    return [
        {
            "layer_idx": int(layer_idx),
            "key_shape": list(kv_shape),
            "value_shape": list(kv_shape),
        }
        for layer_idx in (missing_layer_range or [])
    ]


def build_exact_recompute_verification_event(
    plan,
    input_hidden_states=None,
    recomputed_hidden_states=None,
    verification_status="shape_only",
    invalid_relative_indices=None,
    oracle_self_attn_kv_shapes=None,
    error_type=None,
    error_message=None,
):
    """Build JSON-safe trace metadata for behavior-preserving recompute verification."""
    plan = plan or {}
    return {
        "policy": "h2o_exact_recompute_verify",
        "actual_recompute_applied": False,
        "behavior_changing": False,
        "would_replace_kv": False,
        "target_layer": plan.get("target_layer"),
        "final_layer": plan.get("final_layer"),
        "missing_layer_range": plan.get("missing_layer_range"),
        "recompute_relative_indices": plan.get("recompute_relative_indices"),
        "recompute_positions": plan.get("recompute_positions"),
        "recompute_metadata": plan.get("recompute_metadata"),
        "recompute_count": plan.get("recompute_count"),
        "input_hidden_shape": _shape_list(input_hidden_states),
        "recomputed_hidden_shape": _shape_list(recomputed_hidden_states),
        "verification_status": verification_status,
        "cache_mutation_allowed": False,
        "kv_replacement_applied": False,
        "oracle_self_attn_kv_shapes": oracle_self_attn_kv_shapes,
        "invalid_relative_indices": invalid_relative_indices or [],
        "error_type": error_type,
        "error_message": error_message,
    }
