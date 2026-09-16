"""Canonical provenance helpers for missing-KV dump runs.

The helpers in this module deliberately avoid absolute paths in cryptographic
identities unless a caller explicitly records them as non-identity metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional at import time for lightweight tests.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is present in normal runs.
    _np = None

try:
    import torch as _torch
except Exception:  # pragma: no cover - torch is present in normal runs.
    _torch = None


MANIFEST_SCHEMA_VERSION = 1
PACKED_MANIFEST_SCHEMA_VERSION = 2
LEGACY_DUMP_STORAGE_FORMAT = "legacy_row_v1"
PACKED_GENERATION_STORAGE_FORMAT = "packed_generation_v1"
HIDDEN_LOGICAL_RECORD_TYPE = "all_layer_hidden_state"
KV_LOGICAL_RECORD_TYPE = "all_layer_calib_kv"

MODEL_INDEX_FILES = ("pytorch_model.bin.index.json", "model.safetensors.index.json")
MODEL_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
MODEL_WEIGHT_GLOBS = ("pytorch_model-*.bin", "model-*.safetensors")
TOKENIZER_ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
TOKENIZER_ASSET_ATTRS = (
    "vocab_file",
    "merges_file",
    "spiece_model_file",
    "sp_model_file",
    "tokenizer_file",
)


def expected_samsum_paper_dump_config() -> Dict[str, Any]:
    """Return the frozen SAMSum full-depth calibration protocol."""

    return {
        "producer_protocol": "full_depth_calibration",
        "dump_storage_format": PACKED_GENERATION_STORAGE_FORMAT,
        "dataset_name": "knkarthick/samsum",
        "dataset_split": "validation",
        "text_column": "dialogue",
        "summary_column": "summary",
        "max_source_length": 512,
        "max_target_length": 128,
        "num_beams": 1,
        "per_device_eval_batch_size": 1,
        "use_early_exit": False,
        "use_shallow_deep": False,
        "runtime_restoration_enabled": False,
        "fixed_analysis_source_layer": 6,
        "analysis_confidence_definition": "softmax_top1_top2_margin",
        "analysis_confidence_compute_dtype": "float32",
        "analysis_threshold": 0.9,
        "analysis_threshold_comparator": "strict_gt",
        "adaptive_threshold": False,
        "raw_hidden_enabled": True,
        "normed_hidden_enabled": True,
        "requested_dump_layers": "all",
        "requested_hidden_dump_dtype": "original",
        "requested_kv_dump_dtype": "original",
        "observed_raw_hidden_dtypes": ["float32"],
        "observed_normed_hidden_dtypes": ["float32"],
        "observed_key_dtypes": ["float32"],
        "observed_value_dtypes": ["float32"],
    }


class ProvenanceValidationError(ValueError):
    """Raised when a provenance identity or manifest is structurally invalid."""


def normalize_for_canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ProvenanceValidationError("non_finite_float_not_canonical")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if _np is not None and isinstance(value, _np.generic):
        return normalize_for_canonical_json(value.item())
    if _np is not None and isinstance(value, _np.ndarray):
        return normalize_for_canonical_json(value.tolist())
    if _torch is not None and isinstance(value, _torch.Tensor):
        if value.numel() > 4096:
            raise ProvenanceValidationError("tensor_too_large_for_canonical_json")
        return normalize_for_canonical_json(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): normalize_for_canonical_json(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, tuple):
        return [normalize_for_canonical_json(item) for item in value]
    if isinstance(value, list):
        return [normalize_for_canonical_json(item) for item in value]
    if isinstance(value, set):
        normalized = [normalize_for_canonical_json(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json_text(item))
    raise TypeError("Unsupported canonical JSON type: {}".format(type(value).__name__))


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        normalize_for_canonical_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: Any) -> str:
    if text is None:
        text = ""
    return sha256_bytes(str(text).encode("utf-8"))


def token_sequence_sha256(token_ids: Sequence[Any]) -> str:
    return canonical_json_sha256([int(token_id) for token_id in token_ids])


def effective_token_count(token_ids: Sequence[Any], pad_token_id: Optional[int] = None) -> int:
    count = 0
    for token_id in token_ids:
        value = int(token_id)
        if value == -100:
            continue
        if pad_token_id is not None and value == int(pad_token_id):
            continue
        count += 1
    return count


def stable_sample_id(
    *,
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    split: str,
    raw_dataset_index: int,
    dataset_provided_id: Any,
    source_text_sha256: str,
    reference_text_sha256: str,
) -> str:
    payload = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "split": split,
        "raw_dataset_index": int(raw_dataset_index),
        "dataset_provided_id": None if dataset_provided_id is None else str(dataset_provided_id),
        "source_text_sha256": source_text_sha256,
        "reference_text_sha256": reference_text_sha256,
    }
    return canonical_json_sha256(payload)


def write_json_file(path: os.PathLike[str] | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalize_for_canonical_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json_text(row) + "\n")


def append_jsonl(path: os.PathLike[str] | str, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_text(row) + "\n")


def read_jsonl(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_record(root: Path, path: Path) -> Dict[str, Any]:
    return {
        "relative_path": _relative_posix(path, root),
        "byte_size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def ordered_file_inventory(root: os.PathLike[str] | str, relative_paths: Iterable[str]) -> Dict[str, Any]:
    root_path = Path(root)
    records = []
    seen = set()
    normalized_paths = [Path(rel).as_posix() for rel in relative_paths]
    duplicates = sorted({rel for rel, count in Counter(normalized_paths).items() if count > 1})
    if duplicates:
        raise ProvenanceValidationError("duplicate_inventory_path:{}".format(",".join(duplicates)))
    for rel in sorted(normalized_paths):
        if rel in seen:
            raise ProvenanceValidationError("duplicate_inventory_path:{}".format(rel))
        seen.add(rel)
        candidate = (root_path / rel).resolve()
        try:
            candidate.relative_to(root_path.resolve())
        except ValueError as exc:
            raise ProvenanceValidationError("inventory_path_outside_root:{}".format(rel)) from exc
        if not candidate.is_file():
            raise ProvenanceValidationError("inventory_file_missing:{}".format(rel))
        records.append(_file_record(root_path, candidate))
    return {
        "files": records,
        "file_count": len(records),
        "aggregate_sha256": canonical_json_sha256(records),
    }


def _index_referenced_shards(index_path: Path) -> List[str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ProvenanceValidationError("model_index_missing_weight_map:{}".format(index_path.name))
    return sorted({str(rel).replace("\\", "/") for rel in weight_map.values()})


def model_checkpoint_identity(model_root: os.PathLike[str] | str) -> Dict[str, Any]:
    root = Path(model_root)
    if not root.exists():
        raise ProvenanceValidationError("model_root_missing:{}".format(root))
    if not root.is_dir():
        raise ProvenanceValidationError("model_root_not_directory:{}".format(root))
    if not (root / "config.json").is_file():
        raise ProvenanceValidationError("model_config_missing")

    rel_paths = {"config.json"}
    index_files = [name for name in MODEL_INDEX_FILES if (root / name).is_file()]
    if index_files:
        for index_name in index_files:
            rel_paths.add(index_name)
            for shard in _index_referenced_shards(root / index_name):
                if not (root / shard).is_file():
                    raise ProvenanceValidationError("model_index_referenced_shard_missing:{}".format(shard))
                rel_paths.add(shard)
    else:
        for name in MODEL_WEIGHT_FILES:
            if (root / name).is_file():
                rel_paths.add(name)
        for pattern in MODEL_WEIGHT_GLOBS:
            for path in root.glob(pattern):
                if path.is_file():
                    rel_paths.add(_relative_posix(path, root))
    weight_paths = [rel for rel in rel_paths if rel != "config.json"]
    if not weight_paths:
        raise ProvenanceValidationError("model_weight_file_missing")

    inventory = ordered_file_inventory(root, rel_paths)
    generation_config = None
    generation_config_path = root / "generation_config.json"
    if generation_config_path.is_file():
        generation_config = _file_record(root, generation_config_path)
        try:
            generation_config["metadata"] = json.loads(generation_config_path.read_text(encoding="utf-8"))
        except Exception:
            generation_config["metadata"] = None

    try:
        config_metadata = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except Exception:
        config_metadata = None
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "identity_type": "model_checkpoint",
        "included_file_inventory": inventory["files"],
        "included_file_count": inventory["file_count"],
        "model_checkpoint_identity_sha256": inventory["aggregate_sha256"],
        "generation_config_file": generation_config,
        "config_metadata": config_metadata,
    }
    return payload


def _tokenizer_asset_inventory(
    tokenizer_root: Optional[os.PathLike[str] | str],
    tokenizer: Any = None,
) -> Dict[str, Any]:
    discovered_paths: List[Path] = []
    if tokenizer_root is None:
        root = None
    else:
        root = Path(tokenizer_root)
        if root.is_dir():
            discovered_paths.extend(root / name for name in TOKENIZER_ASSET_FILES if (root / name).is_file())
    if tokenizer is not None:
        candidates = []
        for attr in TOKENIZER_ASSET_ATTRS:
            value = getattr(tokenizer, attr, None)
            if value:
                candidates.append(value)
        init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        if isinstance(init_kwargs, Mapping):
            for key, value in init_kwargs.items():
                if _is_path_like_key(key) and isinstance(value, str):
                    candidates.append(value)
        for candidate in candidates:
            path = Path(str(candidate))
            if path.is_file():
                discovered_paths.append(path)
    if not discovered_paths:
        return {
            "files": [],
            "file_count": 0,
            "aggregate_sha256": canonical_json_sha256([]),
            "identity_sources": {"asset_files": "unavailable"},
        }
    records = []
    seen_digests = set()
    for path in sorted({p.resolve() for p in discovered_paths}, key=lambda item: item.as_posix()):
        digest = sha256_file(path)
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        records.append(
            {
                "asset_name": path.name,
                "byte_size": int(path.stat().st_size),
                "sha256": digest,
            }
        )
    return {
        "files": sorted(records, key=lambda item: (item["asset_name"], item["sha256"])),
        "file_count": len(records),
        "aggregate_sha256": canonical_json_sha256(sorted(records, key=lambda item: (item["asset_name"], item["sha256"]))),
        "identity_sources": {"asset_files": "available" if records else "unavailable"},
    }


def _best_effort_jsonable(value: Any) -> Any:
    try:
        normalize_for_canonical_json(value)
        return value
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(key): _best_effort_jsonable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple, set)):
        return [_best_effort_jsonable(item) for item in value]
    return str(value)


_PATH_LIKE_KEYS = (
    "path",
    "file",
    "dir",
    "directory",
    "cache",
    "name_or_path",
    "pretrained",
    "tokenizer_file",
    "vocab_file",
    "merges_file",
    "spiece",
)


def _is_path_like_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _PATH_LIKE_KEYS)


def _is_path_like_value(value: Any) -> bool:
    if isinstance(value, Path):
        return True
    if not isinstance(value, str):
        return False
    text = value.replace("\\", "/")
    if text.startswith(("/", "./", "../", "~")):
        return True
    if ":" in text[:3]:  # Windows drive path.
        return True
    return any(marker in text.lower() for marker in ("/cache/", "/tokenizer", "/snapshots/", "/models--"))


def _path_independent_tokenizer_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key in sorted(value, key=lambda item: str(item)):
            if _is_path_like_key(key):
                continue
            item = value[key]
            if _is_path_like_value(item):
                continue
            sanitized[str(key)] = _path_independent_tokenizer_config(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if _is_path_like_value(item):
                continue
            out.append(_path_independent_tokenizer_config(item))
        return out
    return _best_effort_jsonable(value)


def tokenizer_identity(
    tokenizer: Any,
    *,
    requested_identifier: Optional[str] = None,
    tokenizer_root: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else getattr(tokenizer, "vocab", {})
    added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    special_token_ids = {}
    for key in sorted(special_tokens_map):
        token_value = special_tokens_map.get(key)
        try:
            special_token_ids[key] = tokenizer.convert_tokens_to_ids(token_value)
        except Exception:
            special_token_ids[key] = getattr(tokenizer, "{}_id".format(key), None)
    semantic_payload = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab": {str(token): int(idx) for token, idx in sorted(vocab.items(), key=lambda item: (int(item[1]), str(item[0])))},
        "added_vocab": {str(token): int(idx) for token, idx in sorted(added_vocab.items(), key=lambda item: str(item[0]))},
        "special_tokens_map": _best_effort_jsonable(special_tokens_map),
        "special_token_ids": special_token_ids,
        "model_input_names": list(getattr(tokenizer, "model_input_names", []) or []),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "behavior_config": _path_independent_tokenizer_config(getattr(tokenizer, "init_kwargs", {}) or {}),
    }
    backend_serialization_sha256 = None
    backend_status = "unavailable"
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        try:
            backend_serialization_sha256 = sha256_bytes(backend.to_str().encode("utf-8"))
            backend_status = "available"
        except Exception:
            backend_status = "error"
    if backend_serialization_sha256 is not None:
        semantic_payload["backend_tokenizer_serialization_sha256"] = backend_serialization_sha256
    asset_inventory = _tokenizer_asset_inventory(tokenizer_root, tokenizer=tokenizer)
    path_metadata = {
        "requested_identifier": requested_identifier,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_root": None if tokenizer_root is None else str(tokenizer_root),
        "resolved_tokenizer_root": None
        if tokenizer_root is None or not Path(tokenizer_root).exists()
        else str(Path(tokenizer_root).resolve()),
    }
    identity_payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "identity_type": "tokenizer",
        "semantic_content": semantic_payload,
        "path_metadata": path_metadata,
        "tokenizer_asset_inventory": asset_inventory["files"],
        "tokenizer_asset_inventory_sha256": asset_inventory["aggregate_sha256"],
        "identity_sources": {
            "backend_tokenizer_serialization": backend_status,
            **asset_inventory.get("identity_sources", {}),
        },
    }
    identity_payload["tokenizer_identity_sha256"] = canonical_json_sha256(
        {
            "semantic_content": semantic_payload,
            "tokenizer_asset_inventory": asset_inventory["files"],
        }
    )
    return normalize_for_canonical_json(identity_payload)


def validate_tokenizer_identity_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    errors = []
    embedded_digest = payload.get("tokenizer_identity_sha256")
    semantic_content = payload.get("semantic_content")
    asset_inventory = payload.get("tokenizer_asset_inventory")
    if not embedded_digest:
        errors.append("tokenizer_identity_sha256_missing")
    if not isinstance(semantic_content, Mapping):
        errors.append("tokenizer_semantic_content_missing")
    if not isinstance(asset_inventory, list):
        errors.append("tokenizer_asset_inventory_missing")
        asset_inventory = []
    recomputed = None
    if isinstance(semantic_content, Mapping):
        recomputed = canonical_json_sha256(
            {
                "semantic_content": semantic_content,
                "tokenizer_asset_inventory": asset_inventory,
            }
        )
        if embedded_digest and recomputed != embedded_digest:
            errors.append("tokenizer_identity_sha256_mismatch")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "embedded_tokenizer_identity_sha256": embedded_digest,
        "recomputed_tokenizer_identity_sha256": recomputed,
    }


def make_effective_population_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_split_row_count: int,
    post_cap_row_count: int,
    dataset_fingerprint: Optional[str],
    raw_validation_dataset_fingerprint: Optional[str] = None,
    post_cap_dataset_fingerprint: Optional[str] = None,
    post_preprocessing_dataset_fingerprint: Optional[str] = None,
    dataset_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    stable_ids = [str(row["stable_sample_id"]) for row in records]
    selected_orders = [int(row["selected_order"]) for row in records]
    duplicate_count = len(stable_ids) - len(set(stable_ids))
    expected_orders = list(range(len(records)))
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if duplicate_count == 0 and selected_orders == expected_orders else "failed",
        "raw_split_row_count": int(raw_split_row_count),
        "post_cap_row_count": int(post_cap_row_count),
        "empty_or_invalid_removed_count": int(post_cap_row_count) - int(len(records)),
        "effective_tokenized_row_count": int(len(records)),
        "selected_order_coverage": {
            "contiguous_zero_based": selected_orders == expected_orders,
            "min_selected_order": min(selected_orders) if selected_orders else None,
            "max_selected_order": max(selected_orders) if selected_orders else None,
        },
        "duplicate_stable_sample_id_count": int(duplicate_count),
        "dataset_fingerprint": dataset_fingerprint,
        "raw_validation_dataset_fingerprint": raw_validation_dataset_fingerprint,
        "post_cap_dataset_fingerprint": post_cap_dataset_fingerprint,
        "post_preprocessing_dataset_fingerprint": post_preprocessing_dataset_fingerprint,
        "dataset_population_sha256": canonical_json_sha256(list(records)),
        "dataset_metadata": dict(dataset_metadata or {}),
    }


def validate_effective_population_sidecar(
    records: Sequence[Mapping[str, Any]],
    supplied_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors = []
    if not isinstance(supplied_summary, Mapping):
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "failed",
            "errors": ["effective_population_summary_missing"],
            "recomputed_effective_tokenized_row_count": len(records),
            "recomputed_dataset_population_sha256": canonical_json_sha256(list(records)),
        }
    stable_ids = [str(row.get("stable_sample_id")) for row in records]
    selected_orders = [int(row.get("selected_order", -1)) for row in records]
    expected_orders = list(range(len(records)))
    duplicate_count = len(stable_ids) - len(set(stable_ids))
    recomputed_sha = canonical_json_sha256(list(records))
    if supplied_summary.get("status") != "ok":
        errors.append("effective_population_summary_status_not_ok")
    if int(supplied_summary.get("effective_tokenized_row_count", -1)) != len(records):
        errors.append("effective_population_row_count_mismatch")
    if supplied_summary.get("dataset_population_sha256") != recomputed_sha:
        errors.append("dataset_population_sha256_mismatch")
    coverage = supplied_summary.get("selected_order_coverage", {})
    if selected_orders != expected_orders:
        errors.append("effective_population_selected_order_not_contiguous")
    if isinstance(coverage, Mapping) and bool(coverage.get("contiguous_zero_based")) != (selected_orders == expected_orders):
        errors.append("effective_population_selected_order_coverage_mismatch")
    if int(supplied_summary.get("duplicate_stable_sample_id_count", -1)) != duplicate_count:
        errors.append("duplicate_stable_sample_id_count_mismatch")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "recomputed_effective_tokenized_row_count": len(records),
        "supplied_effective_tokenized_row_count": supplied_summary.get("effective_tokenized_row_count"),
        "recomputed_duplicate_stable_sample_id_count": duplicate_count,
        "supplied_duplicate_stable_sample_id_count": supplied_summary.get("duplicate_stable_sample_id_count"),
        "selected_order_contiguous_zero_based": selected_orders == expected_orders,
        "recomputed_dataset_population_sha256": recomputed_sha,
        "supplied_dataset_population_sha256": supplied_summary.get("dataset_population_sha256"),
    }


def validate_generation_bindings(
    population_records: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors = []
    population_by_order = {int(row["selected_order"]): row for row in population_records}
    order_counts = Counter(int(row.get("selected_order", -1)) for row in binding_rows)
    generation_counts = Counter(int(row.get("generation_index", -1)) for row in binding_rows)
    stable_counts = Counter(str(row.get("stable_sample_id")) for row in binding_rows)
    expected_orders = set(population_by_order)
    actual_orders = {int(row.get("selected_order", -1)) for row in binding_rows}
    for order, count in order_counts.items():
        if count > 1:
            errors.append("duplicate_selected_order:{}".format(order))
    for generation_index, count in generation_counts.items():
        if count > 1:
            errors.append("duplicate_generation_index:{}".format(generation_index))
    for stable_id, count in stable_counts.items():
        if count > 1:
            errors.append("duplicate_stable_sample_id:{}".format(stable_id))
    missing_orders = sorted(expected_orders - actual_orders)
    unknown_orders = sorted(actual_orders - expected_orders)
    if missing_orders:
        errors.append("ungenerated_effective_samples")
    if unknown_orders:
        errors.append("unknown_generation_selected_order")
    for row in binding_rows:
        order = int(row.get("selected_order", -1))
        expected = population_by_order.get(order)
        if expected is None:
            continue
        if str(row.get("stable_sample_id")) != str(expected.get("stable_sample_id")):
            errors.append("stable_sample_id_mismatch:{}".format(order))
    summary = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if not errors and len(binding_rows) == len(population_records) else "failed",
        "effective_sample_count": len(population_records),
        "generation_binding_count": len(binding_rows),
        "unique_selected_order_count": len(order_counts),
        "unique_generation_index_count": len(generation_counts),
        "unique_stable_sample_id_count": len(stable_counts),
        "missing_selected_orders": missing_orders[:50],
        "unknown_selected_orders": unknown_orders[:50],
        "errors": errors,
        "generation_sample_binding_sha256": canonical_json_sha256(list(binding_rows)),
    }
    if len(binding_rows) != len(population_records):
        summary["errors"].append("effective_sample_generation_count_mismatch")
    summary["status"] = "ok" if not summary["errors"] else "failed"
    return summary


def validate_generation_binding_sidecar(
    population_records: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
    supplied_summary: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    recomputed = validate_generation_bindings(population_records, binding_rows)
    errors = []
    if not isinstance(supplied_summary, Mapping):
        errors.append("generation_binding_summary_missing")
    else:
        for key in (
            "status",
            "effective_sample_count",
            "generation_binding_count",
            "unique_selected_order_count",
            "unique_generation_index_count",
            "unique_stable_sample_id_count",
            "generation_sample_binding_sha256",
        ):
            if supplied_summary.get(key) != recomputed.get(key):
                errors.append("generation_binding_summary_{}_mismatch".format(key))
        for key in ("missing_selected_orders", "unknown_selected_orders"):
            if list(supplied_summary.get(key, [])) != list(recomputed.get(key, [])):
                errors.append("generation_binding_summary_{}_mismatch".format(key))
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if not errors and recomputed.get("status") == "ok" else "failed",
        "errors": errors + ([] if recomputed.get("status") == "ok" else ["generation_binding_recomputed_invalid"]),
        "recomputed": recomputed,
        "supplied_status": supplied_summary.get("status") if isinstance(supplied_summary, Mapping) else None,
        "supplied_generation_sample_binding_sha256": supplied_summary.get("generation_sample_binding_sha256")
        if isinstance(supplied_summary, Mapping)
        else None,
    }


def _identity_tuple(row: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (
        row.get("stable_sample_id"),
        row.get("selected_order"),
        row.get("raw_dataset_index"),
        row.get("dataset_provided_id"),
    )


def validate_dump_rows_against_generation_bindings(
    population_records: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
    hidden_manifest_rows: Sequence[Mapping[str, Any]],
    kv_manifest_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors = []
    population_by_order = {int(row["selected_order"]): row for row in population_records}
    bindings_by_generation = defaultdict(list)
    for row in binding_rows:
        try:
            bindings_by_generation[int(row["generation_index"])].append(row)
        except (KeyError, TypeError, ValueError):
            errors.append("binding_missing_generation_index")

    canonical_binding_by_generation = {}
    multiple_generation_identity_count = 0
    for generation_index, rows in bindings_by_generation.items():
        identity_set = {_identity_tuple(row) for row in rows}
        if len(identity_set) != 1:
            multiple_generation_identity_count += 1
            errors.append("one_generation_associated_with_multiple_sample_identities:{}".format(generation_index))
            continue
        binding = rows[0]
        order = int(binding.get("selected_order", -1))
        expected_population = population_by_order.get(order)
        if expected_population is None:
            errors.append("binding_unknown_selected_order:{}".format(order))
        elif _identity_tuple(binding) != _identity_tuple(expected_population):
            errors.append("binding_population_identity_mismatch:{}".format(generation_index))
        canonical_binding_by_generation[generation_index] = binding

    summary_counts = {
        "validated_hidden_row_count": 0,
        "validated_kv_row_count": 0,
        "unknown_generation_count": 0,
        "sample_mismatch_count": 0,
        "selected_order_mismatch_count": 0,
        "raw_index_mismatch_count": 0,
        "dataset_id_mismatch_count": 0,
    }
    bound_generations = set()
    bound_samples = set()

    def successful(rows):
        return [row for row in rows if row.get("dump_succeeded") and row.get("file_path")]

    def validate_row(row: Mapping[str, Any], label: str) -> None:
        try:
            generation_index = int(row["generation_index"])
        except (KeyError, TypeError, ValueError):
            errors.append("orphan_{}_row_missing_generation_index".format(label))
            summary_counts["unknown_generation_count"] += 1
            return
        binding = canonical_binding_by_generation.get(generation_index)
        if binding is None:
            errors.append("unknown_dump_generation_index:{}:{}".format(label, generation_index))
            errors.append("orphan_{}_row".format(label))
            summary_counts["unknown_generation_count"] += 1
            return
        if str(row.get("stable_sample_id")) != str(binding.get("stable_sample_id")):
            errors.append("dump_stable_sample_id_mismatch:{}:{}".format(label, generation_index))
            summary_counts["sample_mismatch_count"] += 1
        try:
            selected_order_matches = int(row.get("selected_order")) == int(binding.get("selected_order"))
        except (TypeError, ValueError):
            selected_order_matches = False
        if not selected_order_matches:
            errors.append("dump_selected_order_mismatch:{}:{}".format(label, generation_index))
            summary_counts["selected_order_mismatch_count"] += 1
        try:
            raw_index_matches = int(row.get("raw_dataset_index")) == int(binding.get("raw_dataset_index"))
        except (TypeError, ValueError):
            raw_index_matches = False
        if not raw_index_matches:
            errors.append("dump_raw_dataset_index_mismatch:{}:{}".format(label, generation_index))
            summary_counts["raw_index_mismatch_count"] += 1
        row_dataset_id = row.get("dataset_provided_id")
        binding_dataset_id = binding.get("dataset_provided_id")
        if (None if row_dataset_id is None else str(row_dataset_id)) != (
            None if binding_dataset_id is None else str(binding_dataset_id)
        ):
            errors.append("dump_dataset_provided_id_mismatch:{}:{}".format(label, generation_index))
            summary_counts["dataset_id_mismatch_count"] += 1
        bound_generations.add(generation_index)
        bound_samples.add(str(binding.get("stable_sample_id")))
        summary_counts["validated_{}_row_count".format("kv" if label == "kv" else "hidden")] += 1

    for row in successful(hidden_manifest_rows):
        validate_row(row, "hidden")
    for row in successful(kv_manifest_rows):
        validate_row(row, "kv")

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "validated_hidden_row_count": summary_counts["validated_hidden_row_count"],
        "validated_kv_row_count": summary_counts["validated_kv_row_count"],
        "unique_bound_generation_count": len(bound_generations),
        "unique_bound_sample_count": len(bound_samples),
        "unknown_generation_count": summary_counts["unknown_generation_count"],
        "sample_mismatch_count": summary_counts["sample_mismatch_count"],
        "selected_order_mismatch_count": summary_counts["selected_order_mismatch_count"],
        "raw_index_mismatch_count": summary_counts["raw_index_mismatch_count"],
        "dataset_id_mismatch_count": summary_counts["dataset_id_mismatch_count"],
        "multiple_identity_generation_count": multiple_generation_identity_count,
    }


def make_dump_row_uid(row: Mapping[str, Any]) -> str:
    payload = {
        "stable_sample_id": row.get("stable_sample_id"),
        "generation_index": row.get("generation_index"),
        "decoder_position": row.get("decoder_position"),
        "token_index": row.get("token_index"),
        "layer_idx": row.get("layer_idx"),
        "record_type": row.get("record_type"),
    }
    return canonical_json_sha256(payload)


def logical_dump_row_uid(
    *,
    stable_sample_id: Any,
    generation_index: int,
    decoder_position: int,
    token_index: int,
    layer_idx: int,
    record_type: str,
) -> str:
    return make_dump_row_uid(
        {
            "stable_sample_id": stable_sample_id,
            "generation_index": int(generation_index),
            "decoder_position": int(decoder_position),
            "token_index": int(token_index),
            "layer_idx": int(layer_idx),
            "record_type": record_type,
        }
    )


def logical_population_keys_for_packed_record(row: Mapping[str, Any]) -> List[Tuple[str, int, int, int]]:
    token_count = int(row.get("token_count", 0))
    layer_count = int(row.get("layer_count", 0))
    position_start = int(row.get("decoder_position_start", 0))
    layer_start = int(row.get("layer_start", 0))
    stable_id = str(row.get("stable_sample_id"))
    generation_index = int(row.get("generation_index"))
    return [
        (stable_id, generation_index, position_start + token_offset, layer_start + layer_offset)
        for token_offset in range(token_count)
        for layer_offset in range(layer_count)
    ]


def logical_population_sha256(keys: Iterable[Tuple[Any, int, int, int]]) -> str:
    normalized = sorted(
        [[str(stable), int(generation), int(position), int(layer)] for stable, generation, position, layer in keys],
        key=lambda item: canonical_json_text(item),
    )
    return canonical_json_sha256(normalized)


def logical_dump_row_uid_aggregate_sha256_for_packed_record(
    row: Mapping[str, Any],
    *,
    logical_record_type: str,
) -> str:
    uids = [
        logical_dump_row_uid(
            stable_sample_id=stable,
            generation_index=generation,
            decoder_position=position,
            token_index=position,
            layer_idx=layer,
            record_type=logical_record_type,
        )
        for stable, generation, position, layer in logical_population_keys_for_packed_record(row)
    ]
    return canonical_json_sha256(sorted(uids))


def make_packed_record_uid(row: Mapping[str, Any]) -> str:
    payload = {
        "manifest_schema_version": int(row.get("manifest_schema_version", PACKED_MANIFEST_SCHEMA_VERSION)),
        "storage_format": row.get("storage_format"),
        "record_type": row.get("record_type"),
        "stable_sample_id": row.get("stable_sample_id"),
        "selected_order": row.get("selected_order"),
        "raw_dataset_index": row.get("raw_dataset_index"),
        "dataset_provided_id": row.get("dataset_provided_id"),
        "generation_index": row.get("generation_index"),
        "token_count": row.get("token_count"),
        "layer_count": row.get("layer_count"),
        "decoder_position_start": row.get("decoder_position_start"),
        "decoder_position_end_exclusive": row.get("decoder_position_end_exclusive"),
        "decoder_positions_contiguous": row.get("decoder_positions_contiguous"),
        "layer_start": row.get("layer_start"),
        "layer_end_exclusive": row.get("layer_end_exclusive"),
        "layers_contiguous": row.get("layers_contiguous"),
        "raw_hidden_shape": row.get("raw_hidden_shape"),
        "normed_hidden_shape": row.get("normed_hidden_shape"),
        "include_raw_hidden": row.get("include_raw_hidden"),
        "include_normed_hidden": row.get("include_normed_hidden"),
        "key_shape": row.get("key_shape"),
        "value_shape": row.get("value_shape"),
        "dtype": row.get("dtype"),
        "model_num_decoder_layers": row.get("model_num_decoder_layers"),
        "model_d_model": row.get("model_d_model"),
        "model_num_heads": row.get("model_num_heads"),
        "model_d_kv": row.get("model_d_kv"),
        "logical_row_count": row.get("logical_row_count"),
        "logical_population_sha256": row.get("logical_population_sha256"),
        "logical_dump_row_uid_aggregate_sha256": row.get("logical_dump_row_uid_aggregate_sha256"),
    }
    return canonical_json_sha256(payload)


def populate_packed_record_identities(row: Mapping[str, Any], *, logical_record_type: str) -> Dict[str, Any]:
    out = dict(row)
    keys = logical_population_keys_for_packed_record(out)
    out["logical_row_count"] = len(keys)
    out["logical_population_sha256"] = logical_population_sha256(keys)
    out["logical_dump_row_uid_aggregate_sha256"] = logical_dump_row_uid_aggregate_sha256_for_packed_record(
        out,
        logical_record_type=logical_record_type,
    )
    out["packed_record_uid"] = make_packed_record_uid(out)
    return out


def preprocessing_identity(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "identity_type": "missing_kv_preprocessing",
            "preprocessing_schema_version": 1,
            **dict(payload),
        }
    )


def dump_population_key(row: Mapping[str, Any]) -> Optional[Tuple[str, int, int, int]]:
    try:
        stable_id = str(row["stable_sample_id"])
        return (
            stable_id,
            int(row["generation_index"]),
            int(row["decoder_position"]),
            int(row["layer_idx"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def dump_row_storage_format(row: Mapping[str, Any]) -> str:
    schema = int(row.get("manifest_schema_version", 1) or 1)
    if schema == PACKED_MANIFEST_SCHEMA_VERSION:
        return str(row.get("storage_format") or "")
    return str(row.get("storage_format") or LEGACY_DUMP_STORAGE_FORMAT)


def _is_packed_manifest_row(row: Mapping[str, Any]) -> bool:
    try:
        schema = int(row.get("manifest_schema_version", 1) or 1)
    except (TypeError, ValueError):
        schema = None
    return schema == PACKED_MANIFEST_SCHEMA_VERSION or row.get("storage_format") == PACKED_GENERATION_STORAGE_FORMAT


def _resolve_manifest_file_path(file_path: Any, root: Path) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    if not file_path:
        return None, None, "missing_file_path"
    candidate = Path(str(file_path))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        rel = candidate.relative_to(root).as_posix()
    except ValueError:
        # Packed D1/D2 manifests are content-addressed and may be relocated as a
        # bundle. Older rows may still carry the original absolute path, so only
        # allow a relocation fallback for deterministic packed shard filenames.
        name = Path(str(file_path)).name
        if name.startswith(("hidden_generation_", "kv_generation_")) and name.endswith(".pt"):
            fallback = (root / name).resolve()
            try:
                rel = fallback.relative_to(root).as_posix()
            except ValueError:
                return candidate, None, "shard_path_outside_dump_root:{}".format(file_path)
            if fallback.is_file():
                return fallback, rel, None
        return candidate, None, "shard_path_outside_dump_root:{}".format(file_path)
    return candidate, rel, None


def canonical_shard_inventory_record(
    *,
    relative_path: str,
    byte_size: int,
    sha256: str,
    manifest_reference_count: int,
    logical_reference_count: int,
    storage_format: str,
    modality: str,
    generation_index: Any,
) -> Dict[str, Any]:
    """Canonical shard-inventory record shape shared by every caller that
    builds a tensor-shard inventory aggregate, whether by re-hashing live
    files (see ``_manifest_shard_inventory`` below) or by reconstructing the
    identity from already-authenticated small manifest metadata (see
    ``official_free_calm_phase3c_adapter.derive_official_free_calm_dump_run_
    binding``, which never re-reads the referenced multi-GB tensor shards).
    Factored out so both paths compute byte-identical ``aggregate_sha256``
    values for the same underlying shard identities -- one inventory
    semantics implementation, never two."""

    return {
        "relative_path": relative_path,
        "byte_size": int(byte_size),
        "sha256": sha256,
        "manifest_reference_count": int(manifest_reference_count),
        "logical_reference_count": int(logical_reference_count),
        "storage_format": storage_format,
        "modality": modality,
        "generation_index": generation_index,
    }


def shard_inventory_from_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of ``canonical_shard_inventory_record`` dicts into
    the same ``{files, file_count, total_tensor_bytes, aggregate_sha256}``
    shape ``_manifest_shard_inventory`` returns. Callers that need
    ``duplicate_referenced_shard_paths`` populated (only meaningful when the
    records were derived from live-file scanning) set it themselves after
    calling this -- it always starts empty here."""

    ordered = sorted(records, key=lambda item: item["relative_path"])
    return {
        "files": ordered,
        "file_count": len(ordered),
        "duplicate_referenced_shard_paths": [],
        "total_tensor_bytes": int(sum(int(record["byte_size"]) for record in ordered)),
        "aggregate_sha256": canonical_json_sha256(ordered),
    }


def _manifest_shard_inventory(
    rows: Sequence[Mapping[str, Any]],
    *,
    dump_root: os.PathLike[str] | str,
    modality: str,
) -> Tuple[Dict[str, Any], List[str]]:
    root = Path(dump_root).resolve()
    errors = []
    records_by_rel = {}
    path_counts = Counter()
    storage_formats_by_rel = defaultdict(set)
    for row in rows:
        candidate, rel, path_error = _resolve_manifest_file_path(row.get("file_path"), root)
        if path_error:
            errors.append(path_error)
            continue
        if not candidate.is_file():
            errors.append("missing_referenced_shard:{}".format(rel))
            continue
        digest = sha256_file(candidate)
        path_counts[rel] += 1
        storage_formats_by_rel[rel].add(dump_row_storage_format(row))
        existing = records_by_rel.get(rel)
        if existing is not None and existing["sha256"] != digest:
            errors.append("same_path_conflicting_digest:{}".format(rel))
        if existing is None:
            records_by_rel[rel] = canonical_shard_inventory_record(
                relative_path=rel,
                byte_size=candidate.stat().st_size,
                sha256=digest,
                manifest_reference_count=0,
                logical_reference_count=0,
                storage_format=dump_row_storage_format(row),
                modality=modality,
                generation_index=row.get("generation_index"),
            )
        elif (
            existing.get("storage_format") != dump_row_storage_format(row)
            or existing.get("modality") != modality
            or existing.get("generation_index") != row.get("generation_index")
        ):
            errors.append("same_path_conflicting_identity:{}".format(rel))
        records_by_rel[rel]["manifest_reference_count"] += 1
        if dump_row_storage_format(row) == PACKED_GENERATION_STORAGE_FORMAT:
            records_by_rel[rel]["logical_reference_count"] += int(row.get("logical_row_count", 0) or 0)
        else:
            records_by_rel[rel]["logical_reference_count"] += 1
    duplicate_paths = sorted(
        rel
        for rel, count in path_counts.items()
        if count > 1 and LEGACY_DUMP_STORAGE_FORMAT in storage_formats_by_rel.get(rel, set())
    )
    for rel in duplicate_paths:
        errors.append("duplicate_referenced_shard_path:{}".format(rel))
    inventory = shard_inventory_from_records(records_by_rel.values())
    inventory["duplicate_referenced_shard_paths"] = duplicate_paths[:50]
    return inventory, errors


def _packed_expected_record_type(label: str) -> str:
    return "packed_generation_hidden" if label == "hidden" else "packed_generation_kv"


def _packed_logical_record_type(label: str) -> str:
    return HIDDEN_LOGICAL_RECORD_TYPE if label == "hidden" else KV_LOGICAL_RECORD_TYPE


_PACKED_METADATA_COMPARE_FIELDS = (
    "manifest_schema_version",
    "storage_format",
    "record_type",
    "stable_sample_id",
    "selected_order",
    "raw_dataset_index",
    "dataset_provided_id",
    "generation_index",
    "token_count",
    "layer_count",
    "decoder_position_start",
    "decoder_position_end_exclusive",
    "decoder_positions_contiguous",
    "layer_start",
    "layer_end_exclusive",
    "layers_contiguous",
    "dtype",
    "model_num_decoder_layers",
    "model_d_model",
    "model_num_heads",
    "model_d_kv",
    "include_raw_hidden",
    "include_normed_hidden",
    "raw_hidden_shape",
    "normed_hidden_shape",
    "key_shape",
    "value_shape",
    "logical_row_count",
    "logical_population_sha256",
    "logical_dump_row_uid_aggregate_sha256",
    "packed_record_uid",
)


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return normalize_for_canonical_json(left) == normalize_for_canonical_json(right)
    except Exception:
        return left == right


def _positive_int_field(row: Mapping[str, Any], field: str, *, label: str, errors: List[str]) -> Optional[int]:
    try:
        value = int(row.get(field))
    except (TypeError, ValueError):
        errors.append("{}_packed_invalid_{}".format(label, field))
        return None
    if value <= 0:
        errors.append("{}_packed_nonpositive_{}".format(label, field))
        return None
    return value


def _shape_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return None


def _value_category(value: Any) -> str:
    if _torch is not None and isinstance(value, _torch.Tensor):
        scalar = value.detach().cpu()
        if bool(_torch.isnan(scalar)):
            return "nan"
        if bool(_torch.isposinf(scalar)):
            return "positive_inf"
        if bool(_torch.isneginf(scalar)):
            return "negative_inf"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nonfinite"
    if number != number:
        return "nan"
    if number == float("inf"):
        return "positive_inf"
    if number == float("-inf"):
        return "negative_inf"
    return "nonfinite"


def _tensor_numerical_summary(
    tensor: Any,
    *,
    row: Mapping[str, Any],
    relative_path: Optional[str],
    tensor_key: str,
    preview_limit: int = 32,
) -> Dict[str, Any]:
    if _torch is None or not isinstance(tensor, _torch.Tensor):
        return {
            "tensor_key": tensor_key,
            "relative_path": relative_path,
            "generation_index": row.get("generation_index"),
            "stable_sample_id": row.get("stable_sample_id"),
            "status": "failed",
            "error": "not_a_torch_tensor",
            "dtype": None,
            "shape": None,
            "numel": 0,
            "finite_count": 0,
            "nan_count": 0,
            "positive_inf_count": 0,
            "negative_inf_count": 0,
            "nonfinite_count": 0,
            "finite_min": None,
            "finite_max": None,
            "finite_max_abs": None,
            "nonfinite_location_preview": [],
        }
    detached = tensor.detach().cpu()
    finite_mask = _torch.isfinite(detached)
    nan_mask = _torch.isnan(detached)
    posinf_mask = _torch.isposinf(detached)
    neginf_mask = _torch.isneginf(detached)
    nonfinite_mask = ~finite_mask
    finite_values = detached[finite_mask]
    finite_min = None
    finite_max = None
    finite_max_abs = None
    if int(finite_values.numel()) > 0:
        finite_float = finite_values.float()
        finite_min = float(finite_float.min().item())
        finite_max = float(finite_float.max().item())
        finite_max_abs = float(finite_float.abs().max().item())
    preview = []
    nonfinite_indices = _torch.nonzero(nonfinite_mask, as_tuple=False)
    pos_start = int(row.get("decoder_position_start", 0) or 0)
    layer_start = int(row.get("layer_start", 0) or 0)
    for coord_tensor in nonfinite_indices[:preview_limit]:
        coords = [int(item) for item in coord_tensor.tolist()]
        token_offset = coords[0] if len(coords) >= 1 else None
        layer_offset = coords[1] if len(coords) >= 2 else None
        value = detached[tuple(coords)] if coords else detached
        preview.append(
            {
                "relative_shard_path": relative_path,
                "generation_index": row.get("generation_index"),
                "stable_sample_id": row.get("stable_sample_id"),
                "tensor_key": tensor_key,
                "token_offset": token_offset,
                "decoder_position": (pos_start + token_offset) if token_offset is not None else None,
                "layer_offset": layer_offset,
                "layer_idx": (layer_start + layer_offset) if layer_offset is not None else None,
                "remaining_tensor_coordinates": coords[2:] if len(coords) > 2 else [],
                "value_category": _value_category(value),
            }
        )
    return {
        "tensor_key": tensor_key,
        "relative_path": relative_path,
        "generation_index": row.get("generation_index"),
        "stable_sample_id": row.get("stable_sample_id"),
        "status": "ok",
        "dtype": str(detached.dtype).replace("torch.", ""),
        "shape": [int(item) for item in detached.shape],
        "numel": int(detached.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(nan_mask.sum().item()),
        "positive_inf_count": int(posinf_mask.sum().item()),
        "negative_inf_count": int(neginf_mask.sum().item()),
        "nonfinite_count": int(nonfinite_mask.sum().item()),
        "finite_min": finite_min,
        "finite_max": finite_max,
        "finite_max_abs": finite_max_abs,
        "nonfinite_location_preview": preview,
    }


def _merge_numerical_summaries(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ("raw_hidden", "normed_hidden", "key", "value")
    by_key = {
        key: {
            "shard_count": 0,
            "numel": 0,
            "finite_count": 0,
            "nan_count": 0,
            "positive_inf_count": 0,
            "negative_inf_count": 0,
            "nonfinite_count": 0,
            "global_finite_min": None,
            "global_finite_max": None,
            "global_finite_max_abs": None,
            "observed_dtypes": [],
        }
        for key in tensor_keys
    }
    dtype_sets = {key: set() for key in tensor_keys}
    preview_by_key = {key: [] for key in tensor_keys}
    affected_shards = set()
    affected_generations = set()
    total_nonfinite = 0
    for record in records:
        key = str(record.get("tensor_key"))
        if key not in by_key:
            continue
        summary = by_key[key]
        summary["shard_count"] += 1
        for count_key in ("numel", "finite_count", "nan_count", "positive_inf_count", "negative_inf_count", "nonfinite_count"):
            summary[count_key] += int(record.get(count_key, 0) or 0)
        dtype = record.get("dtype")
        if dtype:
            dtype_sets[key].add(str(dtype))
        for source_key, target_key, reducer in (
            ("finite_min", "global_finite_min", min),
            ("finite_max", "global_finite_max", max),
            ("finite_max_abs", "global_finite_max_abs", max),
        ):
            value = record.get(source_key)
            if value is None:
                continue
            summary[target_key] = float(value) if summary[target_key] is None else reducer(float(summary[target_key]), float(value))
        nonfinite = int(record.get("nonfinite_count", 0) or 0)
        total_nonfinite += nonfinite
        if nonfinite:
            if record.get("relative_path") is not None:
                affected_shards.add(str(record.get("relative_path")))
            if record.get("generation_index") is not None:
                try:
                    affected_generations.add(int(record.get("generation_index")))
                except (TypeError, ValueError):
                    affected_generations.add(str(record.get("generation_index")))
            remaining = max(0, 32 - len(preview_by_key[key]))
            if remaining:
                preview_by_key[key].extend(list(record.get("nonfinite_location_preview", []))[:remaining])
    for key in tensor_keys:
        by_key[key]["observed_dtypes"] = sorted(dtype_sets[key])
    preview = []
    for key in tensor_keys:
        preview.extend(preview_by_key[key])
    return {
        "status": "ok" if total_nonfinite == 0 else "failed",
        "tensor_summaries": by_key,
        "total_nonfinite_count": int(total_nonfinite),
        "affected_shard_count": len(affected_shards),
        "affected_generation_indices": sorted(affected_generations, key=lambda item: str(item))[:50],
        "nonfinite_location_preview": preview,
        "observed_dtype_sets": {
            "raw_hidden": by_key["raw_hidden"]["observed_dtypes"],
            "normed_hidden": by_key["normed_hidden"]["observed_dtypes"],
            "key": by_key["key"]["observed_dtypes"],
            "value": by_key["value"]["observed_dtypes"],
        },
    }


def _validate_packed_payload(
    row: Mapping[str, Any],
    *,
    dump_root: os.PathLike[str] | str,
    label: str,
    numerical_records: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    errors: List[str] = []
    if _torch is None:
        errors.append("{}_packed_payload_torch_unavailable".format(label))
        return errors
    root = Path(dump_root).resolve()
    candidate, rel, path_error = _resolve_manifest_file_path(row.get("file_path"), root)
    if path_error:
        errors.append("{}_{}".format(label, path_error))
        return errors
    if not candidate.is_file():
        errors.append("{}_packed_payload_missing:{}".format(label, rel))
        return errors
    try:
        payload = _torch.load(str(candidate), map_location="cpu")
    except Exception as exc:
        errors.append("{}_packed_payload_load_error:{}".format(label, str(exc)))
        return errors
    if not isinstance(payload, Mapping):
        errors.append("{}_packed_payload_not_mapping".format(label))
        return errors
    expected_record_type = _packed_expected_record_type(label)
    if int(payload.get("manifest_schema_version", -1)) != PACKED_MANIFEST_SCHEMA_VERSION:
        errors.append("{}_packed_payload_schema_mismatch".format(label))
    if payload.get("storage_format") != PACKED_GENERATION_STORAGE_FORMAT:
        errors.append("{}_packed_payload_storage_format_mismatch".format(label))
    if payload.get("record_type") != expected_record_type:
        errors.append("{}_packed_payload_record_type_mismatch".format(label))
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("{}_packed_payload_metadata_missing".format(label))
        metadata = {}
    else:
        recomputed_metadata = populate_packed_record_identities(
            metadata,
            logical_record_type=_packed_logical_record_type(label),
        )
        for key in (
            "logical_row_count",
            "logical_population_sha256",
            "logical_dump_row_uid_aggregate_sha256",
            "packed_record_uid",
        ):
            if metadata.get(key) != recomputed_metadata.get(key):
                errors.append("{}_packed_payload_metadata_{}_mismatch".format(label, key))
        if recomputed_metadata.get("packed_record_uid") != row.get("packed_record_uid"):
            errors.append("{}_packed_payload_identity_mismatch".format(label))
        for field in _PACKED_METADATA_COMPARE_FIELDS:
            if not _canonical_equal(metadata.get(field), row.get(field)):
                errors.append("{}_packed_payload_metadata_manifest_mismatch:{}".format(label, field))

    dtype_name = row.get("dtype")
    token_count = _positive_int_field(row, "token_count", label=label, errors=errors)
    layer_count = _positive_int_field(row, "layer_count", label=label, errors=errors)
    model_num_decoder_layers = _positive_int_field(row, "model_num_decoder_layers", label=label, errors=errors)
    if token_count is None or layer_count is None or model_num_decoder_layers is None:
        return errors
    if layer_count != model_num_decoder_layers:
        errors.append("{}_packed_payload_model_layer_count_mismatch".format(label))
    if label == "hidden":
        model_d_model = _positive_int_field(row, "model_d_model", label=label, errors=errors)
        if model_d_model is None:
            return errors
        canonical_shape = [token_count, layer_count, model_d_model]
        requested = {
            "raw_hidden": bool(row.get("include_raw_hidden")),
            "normed_hidden": bool(row.get("include_normed_hidden")),
        }
        for tensor_key, shape_key in (("raw_hidden", "raw_hidden_shape"), ("normed_hidden", "normed_hidden_shape")):
            expected_shape = _shape_list(row.get(shape_key))
            tensor = payload.get(tensor_key)
            if not requested[tensor_key]:
                if expected_shape is not None:
                    errors.append("{}_packed_payload_unrequested_shape_declared:{}".format(label, tensor_key))
                if tensor is not None:
                    errors.append("{}_packed_payload_unrequested_tensor_present:{}".format(label, tensor_key))
                continue
            if expected_shape != canonical_shape:
                errors.append("{}_packed_payload_declared_shape_not_canonical:{}".format(label, tensor_key))
            if tensor is None:
                errors.append("{}_packed_payload_missing_tensor:{}".format(label, tensor_key))
                continue
            if not hasattr(tensor, "shape") or int(tensor.dim()) != 3:
                errors.append("{}_packed_payload_wrong_rank:{}".format(label, tensor_key))
                continue
            if list(tensor.shape) != canonical_shape:
                errors.append("{}_packed_payload_shape_mismatch:{}".format(label, tensor_key))
            if expected_shape is not None and list(tensor.shape) != expected_shape:
                errors.append("{}_packed_payload_manifest_shape_mismatch:{}".format(label, tensor_key))
            if int(tensor.shape[0]) != token_count or int(tensor.shape[1]) != layer_count:
                errors.append("{}_packed_payload_axis_mismatch:{}".format(label, tensor_key))
            if dtype_name and str(tensor.dtype).replace("torch.", "") != str(dtype_name):
                errors.append("{}_packed_payload_dtype_mismatch:{}".format(label, tensor_key))
            numerical_summary = _tensor_numerical_summary(tensor, row=row, relative_path=rel, tensor_key=tensor_key)
            if numerical_records is not None:
                numerical_records.append(numerical_summary)
            if int(numerical_summary.get("nonfinite_count", 0) or 0) > 0:
                errors.append("{}_packed_nonfinite_tensor:{}".format(label, tensor_key))
    else:
        model_num_heads = _positive_int_field(row, "model_num_heads", label=label, errors=errors)
        model_d_kv = _positive_int_field(row, "model_d_kv", label=label, errors=errors)
        if model_num_heads is None or model_d_kv is None:
            return errors
        canonical_shape = [token_count, layer_count, model_num_heads, model_d_kv]
        for tensor_key, shape_key in (("key", "key_shape"), ("value", "value_shape")):
            expected_shape = _shape_list(row.get(shape_key))
            tensor = payload.get(tensor_key)
            if expected_shape != canonical_shape:
                errors.append("{}_packed_payload_declared_shape_not_canonical:{}".format(label, tensor_key))
            if tensor is None:
                errors.append("{}_packed_payload_missing_tensor:{}".format(label, tensor_key))
                continue
            if not hasattr(tensor, "shape") or int(tensor.dim()) != 4:
                errors.append("{}_packed_payload_wrong_rank:{}".format(label, tensor_key))
                continue
            if list(tensor.shape) != canonical_shape:
                errors.append("{}_packed_payload_shape_mismatch:{}".format(label, tensor_key))
            if expected_shape is not None and list(tensor.shape) != expected_shape:
                errors.append("{}_packed_payload_manifest_shape_mismatch:{}".format(label, tensor_key))
            if int(tensor.shape[0]) != token_count or int(tensor.shape[1]) != layer_count:
                errors.append("{}_packed_payload_axis_mismatch:{}".format(label, tensor_key))
            if dtype_name and str(tensor.dtype).replace("torch.", "") != str(dtype_name):
                errors.append("{}_packed_payload_dtype_mismatch:{}".format(label, tensor_key))
            numerical_summary = _tensor_numerical_summary(tensor, row=row, relative_path=rel, tensor_key=tensor_key)
            if numerical_records is not None:
                numerical_records.append(numerical_summary)
            if int(numerical_summary.get("nonfinite_count", 0) or 0) > 0:
                errors.append("{}_packed_nonfinite_tensor:{}".format(label, tensor_key))
    return errors


def validate_dump_population(
    hidden_manifest_rows: Sequence[Mapping[str, Any]],
    kv_manifest_rows: Sequence[Mapping[str, Any]],
    *,
    dump_root: os.PathLike[str] | str,
    allow_token_segment_packed_rows: bool = False,
) -> Dict[str, Any]:
    """Validate a hidden/KV dump population and return its summary.

    ``allow_token_segment_packed_rows`` (default False -- existing behavior
    unchanged for every current caller) opts in to the Official CALM
    calibration collector's token-segment packed layout: one packed row per
    dumped first-crossing token EVENT rather than one row per generation, so
    multiple rows may legitimately share a ``generation_index`` and start at
    the event's own nonzero ``decoder_position_start``. Everything else --
    axis/contiguity checks, packed identity recomputation, full tensor
    payload loading/shape/dtype/finiteness validation, numerical summaries,
    shard inventories, and the logical token-layer population identities
    (which remain the authority that rejects duplicated or overlapping
    segments) -- applies identically in both modes; duplicate detection and
    hidden/KV cross-modality alignment simply key on the
    ``(generation_index, decoder_position_start,
    decoder_position_end_exclusive)`` segment instead of the bare
    generation.
    """

    errors = []
    packed_numerical_records: List[Dict[str, Any]] = []
    failed_manifest_rows = {
        "hidden": {"count": 0, "generation_indices": [], "reasons": []},
        "kv": {"count": 0, "generation_indices": [], "reasons": []},
    }

    def successful(rows, label):
        out = []
        for row in rows:
            if _is_packed_manifest_row(row) and row.get("dump_succeeded") is not True:
                failed_manifest_rows[label]["count"] += 1
                if row.get("generation_index") is not None:
                    try:
                        failed_manifest_rows[label]["generation_indices"].append(int(row.get("generation_index")))
                    except (TypeError, ValueError):
                        failed_manifest_rows[label]["generation_indices"].append(str(row.get("generation_index")))
                reason = row.get("skip_reason") or row.get("error_message") or row.get("failure_reason") or "dump_succeeded_not_true"
                failed_manifest_rows[label]["reasons"].append(str(reason))
                continue
            if row.get("skip_reason") == "max_tokens_reached":
                errors.append("dump_cap_reached")
            if row.get("dump_succeeded") and row.get("file_path"):
                out.append(row)
        return out

    hidden_rows = successful(hidden_manifest_rows, "hidden")
    kv_rows = successful(kv_manifest_rows, "kv")
    for label in ("hidden", "kv"):
        if failed_manifest_rows[label]["count"]:
            errors.append("{}_packed_failed_manifest_rows".format(label))
    if not hidden_rows:
        errors.append("no_successful_hidden_rows")
    if not kv_rows:
        errors.append("no_successful_kv_rows")

    successful_formats = {
        label: sorted({dump_row_storage_format(row) for row in rows})
        for label, rows in (("hidden", hidden_rows), ("kv", kv_rows))
    }
    for label, formats in successful_formats.items():
        if len(formats) > 1:
            errors.append("{}_mixed_dump_storage_formats".format(label))
        for storage_format in formats:
            if storage_format not in {LEGACY_DUMP_STORAGE_FORMAT, PACKED_GENERATION_STORAGE_FORMAT}:
                errors.append("{}_unsupported_dump_storage_format:{}".format(label, storage_format))
    combined_formats = sorted(set(successful_formats.get("hidden", [])) | set(successful_formats.get("kv", [])))
    if len(combined_formats) > 1:
        errors.append("hidden_kv_storage_format_mismatch")
    storage_format = combined_formats[0] if len(combined_formats) == 1 else None

    def summarize_rows(rows, label):
        key_counts = Counter()
        uid_counts = Counter()
        per_layer = Counter()
        per_sample = Counter()
        per_generation = Counter()
        malformed = 0
        expected_num_layers = None
        row_uid_mismatch_count = 0
        for row in rows:
            if not row.get("stable_sample_id"):
                errors.append("{}_missing_stable_sample_id".format(label))
            if not row.get("dump_row_uid"):
                errors.append("{}_missing_dump_row_uid".format(label))
            elif make_dump_row_uid(row) != row.get("dump_row_uid"):
                row_uid_mismatch_count += 1
                errors.append("{}_dump_row_uid_mismatch".format(label))
            key = dump_population_key(row)
            if key is None:
                malformed += 1
                errors.append("{}_missing_semantic_row_key".format(label))
                continue
            key_counts[key] += 1
            uid_counts[str(row.get("dump_row_uid"))] += 1
            per_layer[key[3]] += 1
            per_sample[key[0]] += 1
            per_generation[key[1]] += 1
            layer_total = row.get("model_num_decoder_layers")
            if layer_total is not None:
                expected_num_layers = max(int(layer_total), expected_num_layers or 0)
        duplicates = [canonical_json_text(key) for key, count in key_counts.items() if count > 1]
        duplicate_uids = [uid for uid, count in uid_counts.items() if count > 1]
        if malformed:
            errors.append("{}_malformed_rows".format(label))
        if duplicates:
            errors.append("{}_duplicate_population_identities".format(label))
        if duplicate_uids:
            errors.append("{}_duplicate_dump_row_uids".format(label))
        if row_uid_mismatch_count:
            errors.append("{}_dump_row_uid_validation_failed".format(label))
        expected_layers = list(range(expected_num_layers)) if expected_num_layers else []
        missing_expected_layers = {}
        if expected_layers:
            by_token = defaultdict(set)
            for key in key_counts:
                by_token[key[:3]].add(key[3])
            for token_key, layers in by_token.items():
                missing = sorted(set(expected_layers) - layers)
                if missing:
                    missing_expected_layers[canonical_json_text(token_key)] = missing
            if missing_expected_layers:
                errors.append("{}_missing_expected_layers".format(label))
        return {
            "row_count": len(rows),
            "token_layer_count": len(key_counts),
            "malformed_row_count": malformed,
            "duplicate_identity_count": len(duplicates),
            "duplicate_dump_row_uid_count": len(duplicate_uids),
            "dump_row_uid_mismatch_count": int(row_uid_mismatch_count),
            "duplicate_identities": duplicates[:50],
            "duplicate_dump_row_uids": duplicate_uids[:50],
            "population_keys": set(key_counts),
            "population_sha256": canonical_json_sha256(sorted([list(key) for key in key_counts], key=lambda item: canonical_json_text(item))),
            "per_layer_counts": {str(key): int(value) for key, value in sorted(per_layer.items())},
            "per_sample_counts": {str(key): int(value) for key, value in sorted(per_sample.items())},
            "per_generation_counts": {str(key): int(value) for key, value in sorted(per_generation.items())},
            "missing_expected_layers": dict(list(missing_expected_layers.items())[:50]),
            "successful_generations": sorted({int(key[1]) for key in key_counts}),
            "storage_format": LEGACY_DUMP_STORAGE_FORMAT,
        }

    def summarize_packed_rows(rows, label):
        key_counts = Counter()
        per_layer = Counter()
        per_sample = Counter()
        per_generation = Counter()
        generation_counts = Counter()
        malformed = 0
        uid_mismatches = []
        payload_errors = []
        generation_rows = {}
        model_num_layers = None
        for row in rows:
            record_prefix = "{}_generation_{}".format(label, row.get("generation_index"))
            if row.get("record_type") != _packed_expected_record_type(label):
                errors.append("{}_packed_record_type_mismatch".format(label))
            if int(row.get("manifest_schema_version", -1)) != PACKED_MANIFEST_SCHEMA_VERSION:
                errors.append("{}_packed_schema_version_mismatch".format(label))
            if row.get("storage_format") != PACKED_GENERATION_STORAGE_FORMAT:
                errors.append("{}_packed_storage_format_mismatch".format(label))
            try:
                generation_index = int(row["generation_index"])
                token_count = int(row["token_count"])
                layer_count = int(row["layer_count"])
                pos_start = int(row["decoder_position_start"])
                pos_end = int(row["decoder_position_end_exclusive"])
                layer_start = int(row["layer_start"])
                layer_end = int(row["layer_end_exclusive"])
            except (KeyError, TypeError, ValueError):
                malformed += 1
                errors.append("{}_packed_missing_axis_identity".format(label))
                continue
            # Row identity for duplicate detection and hidden/KV alignment.
            # Default full-generation layout: one row per generation, so the
            # bare generation_index is the identity and a repeat is an
            # error. Token-segment layout (explicit opt-in): one row per
            # dumped token event, so the identity is the
            # (generation, position-range) segment -- repeats of the SAME
            # segment are still an error, and any partial overlap between
            # distinct segments is rejected by the logical token-layer
            # population uniqueness check below, which stays authoritative.
            if allow_token_segment_packed_rows:
                row_identity = (generation_index, pos_start, pos_end)
                duplicate_error = "{}_duplicate_packed_token_segment".format(label)
            else:
                row_identity = generation_index
                duplicate_error = "{}_duplicate_packed_generation".format(label)
            generation_counts[row_identity] += 1
            if generation_counts[row_identity] > 1:
                errors.append(duplicate_error)
            generation_rows[row_identity] = row
            if not row.get("stable_sample_id"):
                errors.append("{}_packed_missing_stable_sample_id".format(label))
            if token_count <= 0:
                errors.append("{}_packed_nonpositive_token_count".format(label))
            if layer_count <= 0:
                errors.append("{}_packed_nonpositive_layer_count".format(label))
            # Token segments legitimately start at the event's own decoder
            # position; a negative start still fails via the position-range
            # check below.
            if not allow_token_segment_packed_rows and pos_start != 0:
                errors.append("{}_packed_position_start_not_zero".format(label))
            if pos_start < 0 or pos_end - pos_start != token_count:
                errors.append("{}_packed_position_range_mismatch".format(label))
            if not bool(row.get("decoder_positions_contiguous")):
                errors.append("{}_packed_positions_not_contiguous".format(label))
            if layer_start != 0:
                errors.append("{}_packed_layer_start_not_zero".format(label))
            if layer_end - layer_start != layer_count:
                errors.append("{}_packed_layer_range_mismatch".format(label))
            if not bool(row.get("layers_contiguous")):
                errors.append("{}_packed_layers_not_contiguous".format(label))
            layer_total = row.get("model_num_decoder_layers")
            if layer_total is not None:
                layer_total = int(layer_total)
                model_num_layers = max(layer_total, model_num_layers or 0)
                if layer_count != layer_total or layer_end != layer_total:
                    errors.append("{}_packed_model_layer_count_mismatch".format(label))
            expected = populate_packed_record_identities(row, logical_record_type=_packed_logical_record_type(label))
            for key in ("logical_row_count", "logical_population_sha256", "logical_dump_row_uid_aggregate_sha256", "packed_record_uid"):
                if row.get(key) != expected.get(key):
                    uid_mismatches.append("{}:{}".format(record_prefix, key))
                    errors.append("{}_packed_{}_mismatch".format(label, key))
            payload_errors.extend(
                _validate_packed_payload(
                    row,
                    dump_root=dump_root,
                    label=label,
                    numerical_records=packed_numerical_records,
                )
            )
            logical_keys = logical_population_keys_for_packed_record(row)
            if len(logical_keys) != int(row.get("logical_row_count", len(logical_keys))):
                errors.append("{}_packed_logical_row_count_mismatch".format(label))
            for key in logical_keys:
                key_counts[key] += 1
                per_layer[key[3]] += 1
                per_sample[key[0]] += 1
                per_generation[key[1]] += 1
        errors.extend(payload_errors)
        duplicates = [canonical_json_text(key) for key, count in key_counts.items() if count > 1]
        if malformed:
            errors.append("{}_packed_malformed_rows".format(label))
        if duplicates:
            errors.append("{}_duplicate_population_identities".format(label))
        expected_layers = list(range(model_num_layers)) if model_num_layers else []
        missing_expected_layers = {}
        if expected_layers:
            by_token = defaultdict(set)
            for key in key_counts:
                by_token[key[:3]].add(key[3])
            for token_key, layers in by_token.items():
                missing = sorted(set(expected_layers) - layers)
                if missing:
                    missing_expected_layers[canonical_json_text(token_key)] = missing
            if missing_expected_layers:
                errors.append("{}_missing_expected_layers".format(label))
        return {
            "row_count": int(sum(int(row.get("logical_row_count", 0) or 0) for row in rows)),
            "physical_manifest_row_count": len(rows),
            "token_layer_count": len(key_counts),
            "malformed_row_count": malformed,
            "duplicate_identity_count": len(duplicates),
            "duplicate_dump_row_uid_count": 0,
            "dump_row_uid_mismatch_count": 0,
            "packed_identity_mismatch_count": len(uid_mismatches),
            "packed_identity_mismatches": uid_mismatches[:50],
            "duplicate_identities": duplicates[:50],
            "duplicate_dump_row_uids": [],
            "population_keys": set(key_counts),
            "population_sha256": logical_population_sha256(key_counts.keys()),
            "logical_dump_row_uid_aggregate_sha256": canonical_json_sha256(
                sorted(
                    [
                        logical_dump_row_uid(
                            stable_sample_id=key[0],
                            generation_index=key[1],
                            decoder_position=key[2],
                            token_index=key[2],
                            layer_idx=key[3],
                            record_type=_packed_logical_record_type(label),
                        )
                        for key in key_counts
                    ]
                )
            ),
            "per_layer_counts": {str(key): int(value) for key, value in sorted(per_layer.items())},
            "per_sample_counts": {str(key): int(value) for key, value in sorted(per_sample.items())},
            "per_generation_counts": {str(key): int(value) for key, value in sorted(per_generation.items())},
            "missing_expected_layers": dict(list(missing_expected_layers.items())[:50]),
            "successful_generations": sorted(
                {key[0] for key in generation_counts}
                if allow_token_segment_packed_rows
                else generation_counts
            ),
            "duplicate_generation_indices": sorted(
                [list(key) if allow_token_segment_packed_rows else key
                 for key, count in generation_counts.items() if count > 1]
            )[:50],
            "packed_rows_by_generation": generation_rows,
            "payload_error_count": len(payload_errors),
            "payload_errors": payload_errors[:50],
            "storage_format": PACKED_GENERATION_STORAGE_FORMAT,
        }

    if storage_format == PACKED_GENERATION_STORAGE_FORMAT:
        hidden_summary = summarize_packed_rows(hidden_rows, "hidden")
        kv_summary = summarize_packed_rows(kv_rows, "kv")
    else:
        hidden_summary = summarize_rows(hidden_rows, "hidden")
        kv_summary = summarize_rows(kv_rows, "kv")

    if storage_format == PACKED_GENERATION_STORAGE_FORMAT:
        for generation_index in sorted(set(hidden_summary.get("packed_rows_by_generation", {})) & set(kv_summary.get("packed_rows_by_generation", {}))):
            hidden_row = hidden_summary["packed_rows_by_generation"][generation_index]
            kv_row = kv_summary["packed_rows_by_generation"][generation_index]
            for field in (
                "stable_sample_id",
                "selected_order",
                "raw_dataset_index",
                "dataset_provided_id",
                "token_count",
                "decoder_position_start",
                "decoder_position_end_exclusive",
                "decoder_positions_contiguous",
                "layer_start",
                "layer_end_exclusive",
                "layers_contiguous",
            ):
                if hidden_row.get(field) != kv_row.get(field):
                    errors.append("packed_hidden_kv_axis_or_identity_mismatch:{}:{}".format(generation_index, field))
        hidden_summary.pop("packed_rows_by_generation", None)
        kv_summary.pop("packed_rows_by_generation", None)

    hidden_inventory, hidden_shard_errors = _manifest_shard_inventory(hidden_rows, dump_root=dump_root, modality="hidden")
    kv_inventory, kv_shard_errors = _manifest_shard_inventory(kv_rows, dump_root=dump_root, modality="kv")
    errors.extend("hidden_{}".format(error) for error in hidden_shard_errors)
    errors.extend("kv_{}".format(error) for error in kv_shard_errors)
    numerical_by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in packed_numerical_records:
        if record.get("relative_path") is not None:
            numerical_by_rel[str(record.get("relative_path"))].append(record)
    for inventory in (hidden_inventory, kv_inventory):
        for file_record in inventory.get("files", []):
            summaries = numerical_by_rel.get(str(file_record.get("relative_path")), [])
            file_record["numerical_summaries"] = sorted(
                summaries,
                key=lambda item: str(item.get("tensor_key")),
            )
    numerical_validation = _merge_numerical_summaries(packed_numerical_records)
    if storage_format == PACKED_GENERATION_STORAGE_FORMAT and numerical_validation.get("status") != "ok":
        errors.append("packed_numerical_validation_failed")

    hidden_keys = hidden_summary.pop("population_keys")
    kv_keys = kv_summary.pop("population_keys")
    hidden_only = sorted(hidden_keys - kv_keys, key=lambda item: canonical_json_text(item))
    kv_only = sorted(kv_keys - hidden_keys, key=lambda item: canonical_json_text(item))
    if hidden_only:
        errors.append("hidden_only_population_identities")
    if kv_only:
        errors.append("kv_only_population_identities")
    matched = sorted(hidden_keys & kv_keys, key=lambda item: canonical_json_text(item))
    combined_inventory = sorted(hidden_inventory["files"] + kv_inventory["files"], key=lambda item: (item["modality"], item["relative_path"]))
    hidden_physical_count = int(hidden_inventory.get("file_count", 0))
    kv_physical_count = int(kv_inventory.get("file_count", 0))
    logical_total = int(hidden_summary.get("row_count", 0)) + int(kv_summary.get("row_count", 0))
    physical_total = hidden_physical_count + kv_physical_count
    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "storage_format": storage_format,
        "hidden_failed_manifest_row_count": int(failed_manifest_rows["hidden"]["count"]),
        "kv_failed_manifest_row_count": int(failed_manifest_rows["kv"]["count"]),
        "hidden_failed_manifest_generation_indices": sorted(failed_manifest_rows["hidden"]["generation_indices"], key=lambda item: str(item))[:50],
        "kv_failed_manifest_generation_indices": sorted(failed_manifest_rows["kv"]["generation_indices"], key=lambda item: str(item))[:50],
        "hidden_failed_manifest_reasons": sorted(set(failed_manifest_rows["hidden"]["reasons"]))[:50],
        "kv_failed_manifest_reasons": sorted(set(failed_manifest_rows["kv"]["reasons"]))[:50],
        "hidden": hidden_summary,
        "kv": kv_summary,
        "hidden_only_identities": [list(key) for key in hidden_only[:50]],
        "kv_only_identities": [list(key) for key in kv_only[:50]],
        "hidden_population_sha256": hidden_summary["population_sha256"],
        "kv_population_sha256": kv_summary["population_sha256"],
        "matched_hidden_kv_population_sha256": canonical_json_sha256([list(key) for key in matched]),
        "numerical_validation": numerical_validation,
        "observed_dtype_sets": numerical_validation.get("observed_dtype_sets", {}),
        "hidden_shard_inventory": hidden_inventory,
        "kv_shard_inventory": kv_inventory,
        "combined_tensor_shard_aggregate_sha256": canonical_json_sha256(combined_inventory),
        "tensor_shard_inventory_records": combined_inventory,
        "unique_physical_shard_count": physical_total,
        "hidden_physical_shard_count": hidden_physical_count,
        "kv_physical_shard_count": kv_physical_count,
        "total_tensor_bytes": int(hidden_inventory.get("total_tensor_bytes", 0)) + int(kv_inventory.get("total_tensor_bytes", 0)),
        "logical_hidden_row_count": int(hidden_summary.get("row_count", 0)),
        "logical_kv_row_count": int(kv_summary.get("row_count", 0)),
        "packing_ratio": (float(logical_total) / float(physical_total)) if physical_total else None,
        "unique_sample_count": len({key[0] for key in hidden_keys | kv_keys}),
        "unique_generation_count": len({key[1] for key in hidden_keys | kv_keys}),
        "unique_generated_token_count": len({key[:3] for key in hidden_keys | kv_keys}),
    }
