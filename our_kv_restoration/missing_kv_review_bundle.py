"""Bounded review-bundle archive helpers for missing-KV paper evidence."""

from __future__ import annotations

import json
import gzip
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .missing_kv_dump_provenance import sha256_file


REVIEW_BUNDLE_SCHEMA_VERSION = 1
BOUNDED_JSONL_PREVIEW_SCHEMA_VERSION = 1
DEFAULT_REVIEW_BUNDLE_MAX_FILE_BYTES = 5_000_000
DEFAULT_BOUNDED_JSONL_PREVIEW_MAX_BYTES = 1_000_000
DEFAULT_REVIEW_BUNDLE_ALLOWED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".tsv",
    ".txt",
}
REVIEW_BUNDLE_EXCLUDED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pth",
    ".pt",
    ".safetensors",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_row_count(path: Path) -> int:
    count = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_bounded_jsonl_preview(
    *,
    full_records_path: Path,
    preview_path: Path,
    manifest_path: Path,
    preview_rows: Iterable[Mapping[str, Any]],
    selection_rule: str,
    record_schema: str,
    max_preview_bytes: int = DEFAULT_BOUNDED_JSONL_PREVIEW_MAX_BYTES,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a deterministic JSONL preview bound to an unchanged full file."""

    full_records_path = Path(full_records_path)
    preview_path = Path(preview_path)
    manifest_path = Path(manifest_path)
    if not full_records_path.is_file():
        raise FileNotFoundError("bounded_preview_full_records_missing")
    if not isinstance(max_preview_bytes, int) or isinstance(max_preview_bytes, bool) or max_preview_bytes <= 0:
        raise ValueError("bounded_preview_max_bytes_invalid")

    encoded_rows: List[bytes] = []
    for row in preview_rows:
        if not isinstance(row, Mapping):
            raise TypeError("bounded_preview_row_not_mapping")
        encoded_rows.append(
            (json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    payload = b"".join(encoded_rows)
    if len(payload) > max_preview_bytes:
        raise ValueError("bounded_preview_size_limit_exceeded")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(payload)
    manifest = {
        "schema_version": BOUNDED_JSONL_PREVIEW_SCHEMA_VERSION,
        "record_schema": str(record_schema),
        "selection_rule": str(selection_rule),
        "full_records_path": str(full_records_path),
        "full_records_sha256": sha256_file(full_records_path),
        "full_records_row_count": _jsonl_row_count(full_records_path),
        "full_records_byte_size": full_records_path.stat().st_size,
        "preview_path": str(preview_path),
        "preview_row_count": len(encoded_rows),
        "preview_sha256": sha256_file(preview_path),
        "preview_byte_size": len(payload),
        "max_preview_bytes": max_preview_bytes,
    }
    if metadata:
        manifest.update(_json_safe(metadata))
    write_json(manifest_path, manifest)
    manifest["preview_manifest_path"] = str(manifest_path)
    manifest["preview_manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _safe_label(label: Any, fallback: str = "evidence") -> str:
    text = str(label or fallback)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text).strip("._")
    return safe or fallback


def _normalize_member_path(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    if normalized in ("", ".", ".."):
        return None
    if normalized.startswith("/") or normalized.startswith("//"):
        return None
    if len(normalized) >= 2 and normalized[1] == ":":
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None
    parts = path.parts
    if any(part in ("", ".", "..") for part in parts):
        return None
    return "/".join(parts) or None


def _copy_file_specs(
    *,
    bundle_dir: Path,
    files: Sequence[Mapping[str, Any]],
    max_file_bytes: int,
    allowed_suffixes: set[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    members: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    used_rel_paths: set[str] = set()
    files_dir = bundle_dir / "files"
    for index, spec in enumerate(files):
        label = str(spec.get("label") or "evidence_{}".format(index))
        required = bool(spec.get("required", False))
        source_raw = spec.get("path")
        if source_raw in (None, ""):
            excluded.append({"label": label, "path": None, "required": required, "reason": "required_path_missing" if required else "optional_path_missing"})
            continue
        source = Path(str(source_raw))
        if not source.is_file():
            excluded.append({"label": label, "path": str(source), "required": required, "reason": "required_file_missing" if required else "optional_file_missing"})
            continue
        suffix = source.suffix.lower()
        size = source.stat().st_size
        if suffix in REVIEW_BUNDLE_EXCLUDED_SUFFIXES:
            excluded.append({"label": label, "path": str(source), "required": required, "size_bytes": size, "reason": "large_artifact_suffix_excluded"})
            continue
        if suffix not in allowed_suffixes:
            excluded.append({"label": label, "path": str(source), "required": required, "size_bytes": size, "reason": "suffix_not_review_bundle_allowed"})
            continue
        if size > int(max_file_bytes):
            excluded.append({"label": label, "path": str(source), "required": required, "size_bytes": size, "reason": "file_too_large_for_review_bundle"})
            continue
        requested_member_path = spec.get("member_path")
        normalized_member_path = (
            _normalize_member_path(str(requested_member_path))
            if requested_member_path not in (None, "")
            else None
        )
        if requested_member_path not in (None, "") and normalized_member_path is None:
            excluded.append(
                {
                    "label": label,
                    "path": str(source),
                    "required": required,
                    "size_bytes": size,
                    "reason": "member_path_invalid",
                }
            )
            continue
        safe_name = _safe_label(label)
        rel = (
            Path(normalized_member_path)
            if normalized_member_path is not None
            else Path("files") / "{}{}".format(safe_name, suffix)
        )
        counter = 1
        while str(rel).replace("\\", "/") in used_rel_paths:
            if normalized_member_path is not None:
                excluded.append(
                    {
                        "label": label,
                        "path": str(source),
                        "required": required,
                        "size_bytes": size,
                        "reason": "member_path_duplicate",
                    }
                )
                rel = None
                break
            rel = Path("files") / "{}_{}{}".format(safe_name, counter, suffix)
            counter += 1
        if rel is None:
            continue
        rel_text = str(rel).replace("\\", "/")
        used_rel_paths.add(rel_text)
        dest = bundle_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, dest)
        except Exception as exc:
            excluded.append({"label": label, "path": str(source), "required": required, "size_bytes": size, "reason": "copy_failed:{}".format(type(exc).__name__)})
            continue
        members.append(
            {
                "label": label,
                "relative_path": rel_text,
                "size_bytes": size,
                "sha256": sha256_file(dest),
            }
        )
    return members, excluded


def _write_archive(bundle_dir: Path, archive_path: Path, members: Sequence[Mapping[str, Any]]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0, compresslevel=9) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for member in sorted(members, key=lambda row: str(row.get("relative_path"))):
                    rel = str(member.get("relative_path"))
                    source = bundle_dir / rel
                    info = tarfile.TarInfo(rel)
                    info.size = source.stat().st_size
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)


def verify_review_bundle_archive(
    archive_path: Path,
    expected_members: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = 1024 * 1024,
) -> Dict[str, Any]:
    archive_path = Path(archive_path)
    failures: List[str] = []
    expected_by_path = {str(row.get("relative_path")): dict(row) for row in expected_members}
    if len(expected_by_path) != len(expected_members):
        failures.append("review_bundle_manifest_duplicate_member_path")
    observed_paths: List[str] = []
    observed_rows: List[Dict[str, Any]] = []
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                rel = _normalize_member_path(member.name)
                if rel is None:
                    failures.append("review_bundle_archive_member_path_unsafe")
                    continue
                if rel in observed_paths:
                    failures.append("review_bundle_archive_duplicate_member_path:{}".format(rel))
                    continue
                observed_paths.append(rel)
                if not member.isfile():
                    failures.append("review_bundle_archive_member_not_file:{}".format(rel))
                    continue
                expected = expected_by_path.get(rel)
                if expected is None:
                    failures.append("review_bundle_archive_unexpected_member:{}".format(rel))
                    continue
                if int(member.size) != int(expected.get("size_bytes", -1)):
                    failures.append("review_bundle_archive_member_size_mismatch:{}".format(rel))
                digest = hashlib.sha256()
                stream = archive.extractfile(member)
                if stream is None:
                    failures.append("review_bundle_archive_member_extract_failed:{}".format(rel))
                    continue
                with stream:
                    while True:
                        chunk = stream.read(chunk_size)
                        if not chunk:
                            break
                        digest.update(chunk)
                observed_sha = digest.hexdigest()
                if observed_sha != expected.get("sha256"):
                    failures.append("review_bundle_archive_member_sha256_mismatch:{}".format(rel))
                observed_rows.append({"relative_path": rel, "size_bytes": int(member.size), "sha256": observed_sha})
    except Exception as exc:
        failures.append("review_bundle_archive_open_failed:{}".format(type(exc).__name__))
    expected_paths = set(expected_by_path)
    observed_set = set(observed_paths)
    for rel in sorted(expected_paths - observed_set):
        failures.append("review_bundle_archive_missing_member:{}".format(rel))
    return {
        "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path) if archive_path.is_file() else None,
        "expected_member_count": len(expected_members),
        "observed_member_count": len(observed_rows),
        "members": observed_rows,
    }


def create_review_bundle_archive(
    *,
    bundle_dir: Path,
    files: Sequence[Mapping[str, Any]],
    bundle_type: str,
    archive_path: Optional[Path] = None,
    max_file_bytes: int = DEFAULT_REVIEW_BUNDLE_MAX_FILE_BYTES,
    allowed_suffixes: Optional[set[str]] = None,
) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(allowed_suffixes or DEFAULT_REVIEW_BUNDLE_ALLOWED_SUFFIXES)
    members, excluded = _copy_file_specs(
        bundle_dir=bundle_dir,
        files=files,
        max_file_bytes=max_file_bytes,
        allowed_suffixes=allowed,
    )
    members = sorted(members, key=lambda row: str(row.get("relative_path")))
    if archive_path is None:
        archive_path = bundle_dir.with_suffix(".tar.gz")
    archive_path = Path(archive_path)
    _write_archive(bundle_dir, archive_path, members)
    verification = verify_review_bundle_archive(archive_path, members)
    required_failures = [row for row in excluded if row.get("required") is True]
    failures: List[str] = []
    if required_failures:
        failures.append("review_bundle_required_file_missing")
    if verification.get("status") != "ok":
        failures.append("review_bundle_archive_verification_failed")
        failures.extend(verification.get("failures") or [])
    status = "ok" if not failures else "failed"
    manifest = {
        "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
        "status": status,
        "failures": failures,
        "bundle_type": bundle_type,
        "bundle_dir": str(bundle_dir),
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path) if archive_path.is_file() else None,
        "member_count": len(members),
        "included_file_count": len(members),
        "excluded_file_count": len(excluded),
        "members": members,
        "files": members,
        "excluded_files": excluded,
        "archive_verification": verification,
    }
    write_json(bundle_dir / "bundle_manifest.json", manifest)
    manifest["bundle_manifest_sha256"] = sha256_file(bundle_dir / "bundle_manifest.json")
    return manifest


__all__ = [
    "BOUNDED_JSONL_PREVIEW_SCHEMA_VERSION",
    "DEFAULT_BOUNDED_JSONL_PREVIEW_MAX_BYTES",
    "DEFAULT_REVIEW_BUNDLE_MAX_FILE_BYTES",
    "REVIEW_BUNDLE_SCHEMA_VERSION",
    "create_review_bundle_archive",
    "verify_review_bundle_archive",
    "write_bounded_jsonl_preview",
]
