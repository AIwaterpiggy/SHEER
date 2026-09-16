def _as_list(value):
    return value if isinstance(value, list) else []


def build_h2o_mask_plan(
    selection,
    pending_metadata=None,
    metadata_positions=None,
    metadata_positions_available=False,
    pending_metadata_truncated=False,
    policy="h2o_mask",
    actual_mask_applied=True,
    mask_mode="key_mask",
):
    """Build a JSON-safe mask plan from a dry-run H2O selection."""
    selection = selection or {}
    candidate_relative_indices = _as_list(selection.get("candidate_relative_indices"))
    full_candidate_relative_indices = _as_list(selection.get("_full_candidate_relative_indices")) or candidate_relative_indices
    candidate_positions = selection.get("candidate_positions")
    if candidate_positions is not None and not isinstance(candidate_positions, list):
        candidate_positions = None

    selected_relative_indices = _as_list(selection.get("selected_relative_indices"))
    full_selected_relative_indices = _as_list(selection.get("_full_selected_relative_indices")) or selected_relative_indices
    selected_positions = _as_list(selection.get("selected_positions"))
    selected_scores = _as_list(selection.get("selected_scores"))
    selected_relative_set = set(selected_relative_indices)
    full_selected_relative_set = set(full_selected_relative_indices)

    non_selected_relative_indices = [
        relative_idx
        for relative_idx in candidate_relative_indices
        if relative_idx not in selected_relative_set
    ]
    full_non_selected_relative_indices = [
        relative_idx
        for relative_idx in full_candidate_relative_indices
        if relative_idx not in full_selected_relative_set
    ]

    non_selected_positions = []
    if candidate_positions is not None:
        position_by_relative_idx = {
            relative_idx: candidate_positions[offset]
            for offset, relative_idx in enumerate(candidate_relative_indices)
            if offset < len(candidate_positions)
        }
        non_selected_positions = [
            position_by_relative_idx.get(relative_idx)
            for relative_idx in non_selected_relative_indices
        ]
    elif non_selected_relative_indices:
        non_selected_positions = [None for _ in non_selected_relative_indices]

    pending_metadata_count = None if pending_metadata is None else len(pending_metadata)

    plan = {
        "policy": policy,
        "target_layer": selection.get("target_layer"),
        "importance_layer": selection.get("importance_layer"),
        "pending_count": selection.get("pending_count"),
        "candidate_positions": candidate_positions,
        "candidate_relative_indices": candidate_relative_indices,
        "selected_positions": selected_positions,
        "selected_relative_indices": selected_relative_indices,
        "selected_scores": selected_scores,
        "non_selected_positions": non_selected_positions,
        "non_selected_relative_indices": non_selected_relative_indices,
        "recent_positions": selection.get("recent_positions"),
        "recent_relative_indices": selection.get("recent_relative_indices"),
        "topk": selection.get("topk"),
        "has_importance_scores": selection.get("has_importance_scores"),
        "candidate_positions_source": selection.get("candidate_positions_source"),
        "used_explicit_metadata_positions": selection.get("used_explicit_metadata_positions"),
        "would_keep_count": len(selected_relative_indices),
        "would_mask_count": len(non_selected_relative_indices),
        "actual_mask_applied": bool(actual_mask_applied),
        "mask_mode": mask_mode,
        "pending_metadata_count": pending_metadata_count,
        "pending_metadata_truncated": pending_metadata_truncated,
        "metadata_positions": metadata_positions,
        "metadata_positions_available": bool(metadata_positions_available),
        "candidate_log_truncated": bool(selection.get("candidate_log_truncated", False)),
        "partition_is_partial": bool(selection.get("candidate_log_truncated", False)),
    }
    if actual_mask_applied:
        plan["_full_candidate_relative_indices"] = full_candidate_relative_indices
        plan["_full_selected_relative_indices"] = full_selected_relative_indices
        plan["_full_non_selected_relative_indices"] = full_non_selected_relative_indices
    return plan


def build_h2o_mask_noop_plan(
    selection,
    pending_metadata=None,
    metadata_positions=None,
    metadata_positions_available=False,
    pending_metadata_truncated=False,
):
    """Build a JSON-safe no-op mask plan from a dry-run H2O selection."""
    return build_h2o_mask_plan(
        selection,
        pending_metadata=pending_metadata,
        metadata_positions=metadata_positions,
        metadata_positions_available=metadata_positions_available,
        pending_metadata_truncated=pending_metadata_truncated,
        policy="h2o_mask_noop",
        actual_mask_applied=False,
        mask_mode="noop",
    )
