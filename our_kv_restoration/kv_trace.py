import json
import math
import os
from numbers import Number

MAX_SANITIZED_ITEMS = 128
MAX_SANITIZED_DICT_ITEMS = 64

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in this repo, but tracing should import cleanly.
    torch = None


def _is_torch_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor)


def _safe_scalar(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value
    return None


def _sanitize_value(value):
    scalar = _safe_scalar(value)
    if scalar is not None or value is None:
        return scalar

    if _is_torch_tensor(value):
        tensor = value.detach()
        if tensor.numel() == 1 and not tensor.is_cuda:
            return _sanitize_value(tensor.item())
        shape = [int(dim) for dim in tensor.shape]
        return "tensor(shape={}, device={})".format(shape, tensor.device)

    if isinstance(value, Number):
        return _sanitize_value(value.item() if hasattr(value, "item") else float(value))

    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SANITIZED_ITEMS:
            return "{}(len={})".format(type(value).__name__, len(value))
        return [_sanitize_value(item) for item in value]

    if isinstance(value, dict):
        if len(value) > MAX_SANITIZED_DICT_ITEMS:
            return "dict(len={})".format(len(value))
        return {str(key): _sanitize_value(item) for key, item in value.items()}

    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            item = None
        if item is not None and item is not value:
            return _sanitize_value(item)

    return str(value)


def safe_cache_seq_len(past_key_value):
    """Return self-attention cache sequence length for one layer cache."""
    if past_key_value is None:
        return None
    try:
        if len(past_key_value) == 0:
            return None
        key_states = past_key_value[0]
    except (TypeError, IndexError):
        return None

    if key_states is None:
        return None
    shape = getattr(key_states, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    try:
        return int(shape[2])
    except (TypeError, ValueError):
        return None


def infer_decoder_position(past_key_values):
    """Infer current decoder position from layer-0 self-attention cache length."""
    if past_key_values is None:
        return None
    try:
        if len(past_key_values) == 0:
            return None
        layer0_past = past_key_values[0]
    except (TypeError, IndexError):
        return None
    return safe_cache_seq_len(layer0_past)


class KVTraceRecorder:
    """Small in-memory JSONL trace recorder for deploy-time KV experiments."""

    def __init__(self, enabled=False, max_records=100000):
        self.enabled = bool(enabled)
        self.max_records = None if max_records is None else max(0, int(max_records))
        self.records = []

    def __len__(self):
        return len(self.records)

    def record(self, event_type: str, **kwargs):
        if not self.enabled:
            return
        if self.max_records is not None and len(self.records) >= self.max_records:
            return

        record = {"event_type": str(event_type)}
        for key, value in kwargs.items():
            record[str(key)] = _sanitize_value(value)
        self.records.append(record)

    def flush(self, path: str):
        if path is None or not self.records:
            return

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "a", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.clear()

    def clear(self):
        self.records.clear()
