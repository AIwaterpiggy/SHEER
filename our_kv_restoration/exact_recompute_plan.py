def _as_list(value):
    return value if isinstance(value, list) else []


def _positions_for_relative_indices(relative_indices, candidate_relative_indices, candidate_positions):
    if candidate_positions is None:
        return [None for _ in relative_indices]
    position_by_relative_idx = {
        relative_idx: candidate_positions[offset]
        for offset, relative_idx in enumerate(candidate_relative_indices)
        if offset < len(candidate_positions)
    }
    return [position_by_relative_idx.get(relative_idx) for relative_idx in relative_indices]


def _metadata_for_relative_indices(relative_indices, pending_metadata):
    if pending_metadata is None:
        return []
    try:
        from .position_bookkeeping import metadata_to_trace_dict
    except Exception:
        return []
    metadata_items = list(pending_metadata)
    selected_metadata = []
    for relative_idx in relative_indices:
        if 0 <= int(relative_idx) < len(metadata_items):
            selected_metadata.append(metadata_to_trace_dict(metadata_items[int(relative_idx)]))
        else:
            selected_metadata.append(None)
    return selected_metadata


def build_exact_recompute_dryrun_plan(
    selection,
    pending_metadata=None,
    metadata_positions=None,
    metadata_positions_available=False,
    pending_metadata_truncated=False,
    target_layer=None,
    final_layer=None,
):
    """Build a JSON-safe exact-recompute dry-run plan from H2O selection."""
    selection = selection or {}
    candidate_relative_indices = _as_list(selection.get("_full_candidate_relative_indices"))
    if not candidate_relative_indices:
        candidate_relative_indices = _as_list(selection.get("candidate_relative_indices"))
    candidate_positions = selection.get("candidate_positions")
    if candidate_positions is not None and not isinstance(candidate_positions, list):
        candidate_positions = None

    selected_relative_indices = _as_list(selection.get("_full_selected_relative_indices"))
    if not selected_relative_indices:
        selected_relative_indices = _as_list(selection.get("selected_relative_indices"))
    selected_scores = _as_list(selection.get("selected_scores"))
    selected_positions = _positions_for_relative_indices(
        selected_relative_indices,
        candidate_relative_indices,
        candidate_positions,
    )
    if all(position is None for position in selected_positions):
        selected_positions = _as_list(selection.get("selected_positions"))

    recompute_relative_indices = list(selected_relative_indices)
    recompute_positions = list(selected_positions)
    recompute_metadata = _metadata_for_relative_indices(recompute_relative_indices, pending_metadata)
    recompute_relative_set = set(recompute_relative_indices)

    non_recompute_relative_indices = [
        relative_idx
        for relative_idx in candidate_relative_indices
        if relative_idx not in recompute_relative_set
    ]
    non_recompute_positions = _positions_for_relative_indices(
        non_recompute_relative_indices,
        candidate_relative_indices,
        candidate_positions,
    )

    if target_layer is None:
        target_layer = selection.get("target_layer")
    if target_layer is not None:
        target_layer = int(target_layer)
    if final_layer is not None:
        final_layer = int(final_layer)
    missing_layer_range = []
    if target_layer is not None and final_layer is not None:
        missing_layer_range = list(range(target_layer, final_layer))

    pending_metadata_count = None if pending_metadata is None else len(pending_metadata)

    return {
        "policy": "h2o_exact_recompute_dryrun",
        "mode": "dryrun",
        "actual_recompute_applied": False,
        "target_layer": target_layer,
        "final_layer": final_layer,
        "missing_layer_range": missing_layer_range,
        "pending_count": selection.get("pending_count"),
        "candidate_relative_indices": candidate_relative_indices,
        "candidate_positions": candidate_positions,
        "selected_relative_indices": selected_relative_indices,
        "selected_positions": selected_positions,
        "selected_scores": selected_scores,
        "recompute_relative_indices": recompute_relative_indices,
        "recompute_positions": recompute_positions,
        "recompute_metadata": recompute_metadata,
        "recompute_count": len(recompute_relative_indices),
        "non_recompute_relative_indices": non_recompute_relative_indices,
        "non_recompute_positions": non_recompute_positions,
        "non_recompute_count": len(non_recompute_relative_indices),
        "candidate_positions_source": selection.get("candidate_positions_source"),
        "used_explicit_metadata_positions": selection.get("used_explicit_metadata_positions"),
        "metadata_positions": metadata_positions,
        "metadata_positions_available": bool(metadata_positions_available),
        "pending_metadata_count": pending_metadata_count,
        "pending_metadata_truncated": pending_metadata_truncated,
        "candidate_log_truncated": bool(selection.get("candidate_log_truncated", False)),
        "partition_is_partial": bool(selection.get("candidate_log_truncated", False)),
    }
