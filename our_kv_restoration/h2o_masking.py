def _as_int_list(values):
    if values is None:
        return []
    return [int(value) for value in values]


def build_h2o_key_mask_indices(
    selection_or_plan,
    pending_count,
    past_key_value_len,
    hidden_states_seq_len,
):
    pending_count = int(pending_count or 0)
    past_key_value_len = int(past_key_value_len or 0)
    hidden_states_seq_len = int(hidden_states_seq_len or 0)
    real_seq_length = past_key_value_len + hidden_states_seq_len

    selected_relative_indices = _as_int_list(
        selection_or_plan.get("_full_selected_relative_indices", selection_or_plan.get("selected_relative_indices"))
    )
    raw_non_selected_relative_indices = selection_or_plan.get(
        "_full_non_selected_relative_indices",
        selection_or_plan.get("non_selected_relative_indices"),
    )
    non_selected_relative_indices = _as_int_list(raw_non_selected_relative_indices)
    if raw_non_selected_relative_indices is None:
        selected_set = set(selected_relative_indices)
        non_selected_relative_indices = [
            int(relative_idx)
            for relative_idx in selection_or_plan.get(
                "_full_candidate_relative_indices",
                selection_or_plan.get("candidate_relative_indices", []),
            )
            if int(relative_idx) not in selected_set
        ]

    checked_relative_indices = selected_relative_indices + non_selected_relative_indices
    invalid_relative_indices = [
        relative_idx
        for relative_idx in checked_relative_indices
        if relative_idx < 0 or relative_idx >= pending_count
    ]
    if hidden_states_seq_len < pending_count:
        invalid_relative_indices.extend(list(range(hidden_states_seq_len, pending_count)))

    if invalid_relative_indices:
        raise ValueError(
            "Invalid h2o_mask relative indices for pending_count={}: {}".format(
                pending_count,
                sorted(set(invalid_relative_indices)),
            )
        )

    masked_key_positions = [
        past_key_value_len + relative_idx
        for relative_idx in non_selected_relative_indices
    ]
    invalid_key_positions = [
        key_position
        for key_position in masked_key_positions
        if key_position < 0 or key_position >= real_seq_length
    ]
    if invalid_key_positions:
        raise ValueError(
            "Invalid h2o_mask key positions for real_seq_length={}: {}".format(
                real_seq_length,
                sorted(set(invalid_key_positions)),
            )
        )

    current_token_local_indices = list(range(pending_count, hidden_states_seq_len))

    return {
        "masked_relative_indices": non_selected_relative_indices,
        "masked_key_positions": masked_key_positions,
        "kept_relative_indices": selected_relative_indices,
        "current_token_local_indices": current_token_local_indices,
        "past_key_value_len": past_key_value_len,
        "hidden_states_seq_len": hidden_states_seq_len,
        "real_seq_length": real_seq_length,
        "invalid_relative_indices": [],
    }
