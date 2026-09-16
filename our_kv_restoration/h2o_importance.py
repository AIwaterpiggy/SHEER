import json
import os

import torch


class H2OImportanceTracker:
    """Layer-wise accumulated decoder self-attention importance."""

    def __init__(
        self,
        enabled=False,
        num_layers=0,
        mode="h2o_layer",
        decay=1.0,
        include_current=True,
    ):
        self.enabled = bool(enabled)
        self.num_layers = int(num_layers or 0)
        self.mode = mode
        self.decay = 1.0 if decay is None else float(decay)
        self.include_current = bool(include_current)
        self.scores = [None for _ in range(self.num_layers)]

        if self.enabled and self.mode != "h2o_layer":
            raise ValueError("Unsupported KV importance mode: {}".format(self.mode))
        if self.enabled and (self.decay < 0.0 or self.decay > 1.0):
            raise ValueError("kv_importance_decay must be in [0.0, 1.0]")

    def reset(self):
        self.scores = [None for _ in range(self.num_layers)]

    def _ensure_layer(self, layer_idx):
        if layer_idx is None:
            return None
        layer_idx = int(layer_idx)
        if layer_idx < 0:
            return None
        if layer_idx >= len(self.scores):
            self.scores.extend([None for _ in range(layer_idx + 1 - len(self.scores))])
            self.num_layers = len(self.scores)
        return layer_idx

    def update(self, layer_idx: int, attn_weights):
        if not self.enabled:
            return
        layer_idx = self._ensure_layer(layer_idx)
        if layer_idx is None or attn_weights is None:
            return
        if len(attn_weights.shape) != 4:
            return

        with torch.no_grad():
            weights = attn_weights.detach()
            query_len = weights.shape[-2]
            key_len = weights.shape[-1]
            if key_len == 0:
                return

            token_scores = weights.float().mean(dim=(0, 1, 2))
            if not self.include_current and query_len == 1:
                token_scores = token_scores.clone()
                token_scores[-1] = 0.0

            current_scores = self.scores[layer_idx]
            if current_scores is None:
                current_scores = torch.zeros(
                    key_len,
                    dtype=token_scores.dtype,
                    device=token_scores.device,
                )
            elif current_scores.device != token_scores.device:
                current_scores = current_scores.to(token_scores.device)

            if current_scores.numel() < key_len:
                expanded_scores = torch.zeros(
                    key_len,
                    dtype=token_scores.dtype,
                    device=token_scores.device,
                )
                expanded_scores[: current_scores.numel()] = current_scores
                current_scores = expanded_scores
            elif current_scores.numel() > key_len:
                token_scores = torch.nn.functional.pad(token_scores, (0, current_scores.numel() - key_len))

            if self.decay != 1.0:
                current_scores = current_scores * self.decay
            self.scores[layer_idx] = current_scores + token_scores

    def get_scores(self, layer_idx: int):
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= len(self.scores):
            return None
        return self.scores[layer_idx]

    def topk(self, layer_idx: int, candidate_positions=None, k=None):
        scores = self.get_scores(layer_idx)
        if scores is None or scores.numel() == 0:
            return []

        with torch.no_grad():
            selected_positions = candidate_positions
            selected_scores = scores.detach()
            if selected_positions is not None:
                selected_positions = torch.as_tensor(selected_positions, device=selected_scores.device, dtype=torch.long)
                if selected_positions.numel() == 0:
                    return []
                selected_scores = selected_scores.index_select(0, selected_positions)
            else:
                selected_positions = torch.arange(selected_scores.numel(), device=selected_scores.device)

            if k is None or k > selected_scores.numel():
                k = int(selected_scores.numel())
            if k <= 0:
                return []

            values, indices = torch.topk(selected_scores, k=k)
            positions = selected_positions.index_select(0, indices)
            return [
                {"position": int(position), "score": float(value)}
                for position, value in zip(positions.cpu().tolist(), values.cpu().tolist())
            ]

    def state_dict_cpu(self):
        return {
            "enabled": self.enabled,
            "num_layers": self.num_layers,
            "mode": self.mode,
            "decay": self.decay,
            "include_current": self.include_current,
            "scores": [
                None if layer_scores is None else layer_scores.detach().cpu().tolist()
                for layer_scores in self.scores
            ],
        }

    def dump(self, path: str):
        if path is None:
            return
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.state_dict_cpu(), handle, sort_keys=True)
