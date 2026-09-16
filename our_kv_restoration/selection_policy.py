def resolve_importance_layer(target_layer, shallow_exit_layer=None, mode="target"):
    if target_layer is None:
        return None
    target_layer = int(target_layer)

    if mode == "target":
        return target_layer
    if mode == "shallow_exit":
        return None if shallow_exit_layer is None else int(shallow_exit_layer)
    if mode == "previous":
        return target_layer - 1 if target_layer > 0 else None
    raise ValueError("Unsupported importance layer mode: {}".format(mode))


def _truncate(items, max_items):
    if items is None:
        return None, False
    return items[:max_items], len(items) > max_items


def _score_at(scores, position):
    try:
        return float(scores[position].detach().cpu().item())
    except AttributeError:
        return float(scores[position])


def select_h2o_topk_dryrun(
    importance_tracker,
    target_layer,
    pending_count,
    pending_start_position=None,
    candidate_positions=None,
    candidate_relative_indices=None,
    shallow_exit_layer=None,
    topk=0,
    importance_layer_mode="target",
    recent_window=0,
    max_logged_candidates=128,
):
    pending_count = int(pending_count or 0)
    importance_layer = resolve_importance_layer(
        target_layer,
        shallow_exit_layer=shallow_exit_layer,
        mode=importance_layer_mode,
    )

    if candidate_relative_indices is None:
        candidate_relative_indices = list(range(pending_count))
    else:
        candidate_relative_indices = list(candidate_relative_indices)
    candidate_offsets = list(range(len(candidate_relative_indices)))

    used_explicit_metadata_positions = candidate_positions is not None
    if candidate_positions is not None:
        candidate_positions = list(candidate_positions)
        candidate_positions_source = "metadata"
        if len(candidate_positions) < len(candidate_relative_indices):
            candidate_positions.extend([None] * (len(candidate_relative_indices) - len(candidate_positions)))
        elif len(candidate_positions) > len(candidate_relative_indices):
            candidate_positions = candidate_positions[: len(candidate_relative_indices)]
    elif pending_start_position is None:
        candidate_positions = None
        candidate_positions_source = "relative_only"
    else:
        start = int(pending_start_position)
        candidate_positions = [start + idx for idx in candidate_relative_indices]
        candidate_positions_source = "pending_start_position"

    recent_window = max(0, int(recent_window or 0))
    recent_relative_indices = []
    recent_offsets = []
    if recent_window > 0 and pending_count > 0:
        recent_start = max(0, len(candidate_offsets) - recent_window)
        recent_offsets = candidate_offsets[recent_start:]
        recent_relative_indices = [candidate_relative_indices[idx] for idx in recent_offsets]
    recent_positions = None
    if candidate_positions is not None:
        recent_positions = [candidate_positions[idx] for idx in recent_offsets]

    selected = []
    has_importance_scores = False
    if (
        importance_tracker is not None
        and importance_layer is not None
        and candidate_positions is not None
    ):
        scores = importance_tracker.get_scores(importance_layer)
        if scores is not None:
            has_importance_scores = True
            score_len = int(scores.shape[0]) if hasattr(scores, "shape") else len(scores)
            for relative_idx, position in zip(candidate_relative_indices, candidate_positions):
                if position is not None:
                    position = int(position)
                    if 0 <= position < score_len:
                        selected.append((relative_idx, position, _score_at(scores, position)))

    if topk is None or int(topk) < 0:
        selected_count = len(selected)
    else:
        selected_count = min(int(topk), len(selected))

    if selected_count == 0:
        selected = []
    else:
        selected = sorted(selected, key=lambda item: item[2], reverse=True)[:selected_count]

    selected_relative_indices = [item[0] for item in selected]
    selected_positions = [item[1] for item in selected]
    selected_scores = [item[2] for item in selected]

    candidate_positions_logged, candidate_pos_truncated = _truncate(candidate_positions, max_logged_candidates)
    candidate_relative_logged, candidate_rel_truncated = _truncate(candidate_relative_indices, max_logged_candidates)
    selected_positions_logged, selected_pos_truncated = _truncate(selected_positions, max_logged_candidates)
    selected_relative_logged, selected_rel_truncated = _truncate(selected_relative_indices, max_logged_candidates)
    selected_scores_logged, selected_scores_truncated = _truncate(selected_scores, max_logged_candidates)
    recent_positions_logged, recent_pos_truncated = _truncate(recent_positions, max_logged_candidates)
    recent_relative_logged, recent_rel_truncated = _truncate(recent_relative_indices, max_logged_candidates)

    return {
        "policy": "h2o_topk_dryrun",
        "target_layer": None if target_layer is None else int(target_layer),
        "importance_layer": importance_layer,
        "pending_count": pending_count,
        "pending_start_position": pending_start_position,
        "candidate_positions": candidate_positions_logged,
        "candidate_relative_indices": candidate_relative_logged,
        "selected_positions": selected_positions_logged,
        "selected_relative_indices": selected_relative_logged,
        "selected_scores": selected_scores_logged,
        "recent_relative_indices": recent_relative_logged,
        "recent_positions": recent_positions_logged,
        "_full_candidate_relative_indices": candidate_relative_indices,
        "_full_selected_relative_indices": selected_relative_indices,
        "topk": topk,
        "has_importance_scores": has_importance_scores,
        "used_explicit_metadata_positions": used_explicit_metadata_positions,
        "candidate_positions_source": candidate_positions_source,
        "candidate_log_truncated": any(
            [
                candidate_pos_truncated,
                candidate_rel_truncated,
                selected_pos_truncated,
                selected_rel_truncated,
                selected_scores_truncated,
                recent_pos_truncated,
                recent_rel_truncated,
            ]
        ),
    }
