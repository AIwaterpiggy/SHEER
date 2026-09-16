from dataclasses import asdict, dataclass
from numbers import Number
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - this helper should import even if torch is unavailable.
    torch = None


MAX_METADATA_TRACE_ITEMS = 128


@dataclass
class SkippedTokenMetadata:
    relative_index: int
    decoder_position: Optional[int]
    layer0_past_seq_len: Optional[int]
    exit_layer: Optional[int]
    confidence: Optional[float] = None


def _safe_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Number):
        return int(value)
    if hasattr(value, "item") and not _is_cuda_tensor(value):
        try:
            return int(value.item())
        except Exception:
            return None
    return None


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Number):
        return float(value)
    if hasattr(value, "item") and not _is_cuda_tensor(value):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _is_cuda_tensor(value):
    return torch is not None and isinstance(value, torch.Tensor) and value.is_cuda


def make_skipped_token_metadata(
    relative_index,
    decoder_position=None,
    layer0_past_seq_len=None,
    exit_layer=None,
    confidence=None,
):
    return SkippedTokenMetadata(
        relative_index=_safe_int(relative_index) or 0,
        decoder_position=_safe_int(decoder_position),
        layer0_past_seq_len=_safe_int(layer0_past_seq_len),
        exit_layer=_safe_int(exit_layer),
        confidence=_safe_float(confidence),
    )


def metadata_to_trace_dict(metadata):
    if metadata is None:
        return None
    if isinstance(metadata, SkippedTokenMetadata):
        return asdict(metadata)
    return {
        "relative_index": _safe_int(metadata.get("relative_index")),
        "decoder_position": _safe_int(metadata.get("decoder_position")),
        "layer0_past_seq_len": _safe_int(metadata.get("layer0_past_seq_len")),
        "exit_layer": _safe_int(metadata.get("exit_layer")),
        "confidence": _safe_float(metadata.get("confidence")),
    }


def metadata_list_to_trace(metadata_list, max_items=MAX_METADATA_TRACE_ITEMS):
    if metadata_list is None:
        return [], False
    metadata_items = list(metadata_list)
    truncated = len(metadata_items) > max_items
    return [metadata_to_trace_dict(item) for item in metadata_items[:max_items]], truncated


def infer_pending_start_position_from_metadata(metadata_list):
    if not metadata_list:
        return None
    for metadata in metadata_list:
        trace_dict = metadata_to_trace_dict(metadata)
        if trace_dict is not None and trace_dict["decoder_position"] is not None:
            return trace_dict["decoder_position"]
    return None


def candidate_positions_from_metadata(metadata_list):
    if metadata_list is None:
        return None
    return [
        metadata_to_trace_dict(metadata)["decoder_position"]
        for metadata in metadata_list
    ]
