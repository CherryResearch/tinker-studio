from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IMPORTED_SOURCES_PATH = Path("processed") / "imported_sources.jsonl"
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
JSON_EXTENSIONS = {".json"}
JSONL_EXTENSIONS = {".jsonl", ".ndjson"}
CSV_EXTENSIONS = {".csv", ".tsv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local notes, poetry, or Google Keep exports into the Tinker dataset.")
    parser.add_argument("--input", required=True, help="Input file or folder.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root to update.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "google_keep", "poetry", "notes", "longform"),
        default="auto",
        help="How to interpret the input rows.",
    )
    parser.add_argument("--label", default="", help="Optional label to attach to every imported row.")
    parser.add_argument("--preview", action="store_true", help="Preview rows and counts without writing imported_sources.jsonl.")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True, help="Append to existing imports.")
    parser.add_argument("--replace", dest="append", action="store_false", help="Replace existing imported rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = dataset_root / IMPORTED_SOURCES_PATH
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    rows = list(import_sources(input_path, source_type=args.source_type, label=args.label))
    if not rows:
        raise RuntimeError(f"No importable text rows found under {input_path}")

    existing_rows = read_jsonl(output_path) if args.append and output_path.exists() else []
    merged_rows = dedupe_rows(existing_rows + rows)
    if args.preview:
        print(f"Preview rows: {len(rows)}")
        print(f"Existing rows: {len(existing_rows)}")
        print(f"Total after import: {len(merged_rows)}")
        print(format_counts(merged_rows))
        print("Sample:")
        for row in rows[:5]:
            preview = {
                "source_type": row.get("source_type"),
                "title": row.get("title"),
                "color": row.get("color"),
                "labels": row.get("labels"),
                "word_count": row.get("word_count"),
                "source_path": row.get("source_path"),
                "text_preview": str(row.get("text") or "")[:220],
            }
            print(json.dumps(preview, ensure_ascii=False))
        return 0

    write_jsonl(output_path, merged_rows)

    print(f"Imported rows: {len(rows)}")
    print(f"Total imported rows: {len(merged_rows)}")
    print(f"Output: {output_path}")
    print(format_counts(merged_rows))
    return 0


def import_sources(input_path: Path, *, source_type: str, label: str) -> Iterable[dict[str, Any]]:
    paths = discover_input_paths(input_path)
    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for path in paths:
        for row in rows_from_path(path, source_type=source_type, label=label, imported_at=imported_at):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            row["id"] = stable_id(row)
            row["word_count"] = len(text.split())
            row["char_count"] = len(text)
            yield row


def discover_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    supported = TEXT_EXTENSIONS | JSON_EXTENSIONS | JSONL_EXTENSIONS | CSV_EXTENSIONS
    return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in supported)


def rows_from_path(path: Path, *, source_type: str, label: str, imported_at: str) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in JSON_EXTENSIONS:
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from rows_from_json(data, path=path, source_type=source_type, label=label, imported_at=imported_at)
        return
    if suffix in JSONL_EXTENSIONS:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield normalize_mapping_row(
                        json.loads(line),
                        path=path,
                        source_type=resolve_source_type(source_type, path),
                        label=label,
                        imported_at=imported_at,
                    )
        return
    if suffix in CSV_EXTENSIONS:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield normalize_mapping_row(
                    row,
                    path=path,
                    source_type=resolve_source_type(source_type, path),
                    label=label,
                    imported_at=imported_at,
                )
        return
    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        yield build_row(
            source_type=resolve_source_type(source_type, path, {"text": text}),
            title=path.stem,
            text=text,
            path=path,
            label=label,
            imported_at=imported_at,
        )


def rows_from_json(
    data: Any,
    *,
    path: Path,
    source_type: str,
    label: str,
    imported_at: str,
) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield normalize_mapping_row(
                    item,
                    path=path,
                    source_type=resolve_source_type(source_type, path, item),
                    label=label,
                    imported_at=imported_at,
                )
        return
    if isinstance(data, dict):
        yield normalize_mapping_row(
            data,
            path=path,
            source_type=resolve_source_type(source_type, path, data),
            label=label,
            imported_at=imported_at,
        )


def resolve_source_type(source_type: str, path: Path, row: dict[str, Any] | None = None) -> str:
    if source_type != "auto":
        return source_type
    row = row or {}
    keep_markers = {"textContent", "listContent", "userEditedTimestampUsec", "isArchived", "isTrashed"}
    if any(key in row for key in keep_markers):
        return "google_keep"
    text = str(row.get("text") or row.get("content") or "")
    if path.suffix.lower() in TEXT_EXTENSIONS and looks_like_poetry(text):
        return "poetry"
    return "notes"


def normalize_mapping_row(
    row: dict[str, Any],
    *,
    path: Path,
    source_type: str,
    label: str,
    imported_at: str,
) -> dict[str, Any]:
    if source_type == "google_keep":
        return normalize_google_keep_row(row, path=path, label=label, imported_at=imported_at)
    text = first_text(row, "text", "body", "content", "note", "poem", "rendered_text")
    title = first_text(row, "title", "name", "heading") or path.stem
    labels = normalize_labels(row.get("labels") or row.get("tags") or label)
    return build_row(
        source_type=source_type,
        title=title,
        text=text,
        path=path,
        label=label,
        imported_at=imported_at,
        color=str(row.get("color") or ""),
        labels=labels,
        created_at=first_text(row, "created_at", "created", "timestamp"),
        updated_at=first_text(row, "updated_at", "updated", "modified"),
        metadata={key: value for key, value in row.items() if key not in {"text", "body", "content", "note", "poem"}},
    )


def normalize_google_keep_row(row: dict[str, Any], *, path: Path, label: str, imported_at: str) -> dict[str, Any]:
    text_parts: list[str] = []
    title = str(row.get("title") or path.stem).strip()
    text_content = str(row.get("textContent") or "").strip()
    if text_content:
        text_parts.append(text_content)
    list_content = row.get("listContent")
    if isinstance(list_content, list):
        list_items = []
        for item in list_content:
            if isinstance(item, dict):
                item_text = str(item.get("text") or "").strip()
                if item_text:
                    marker = "[x]" if item.get("isChecked") else "[ ]"
                    list_items.append(f"{marker} {item_text}")
        if list_items:
            text_parts.append("\n".join(list_items))
    labels = normalize_labels(row.get("labels"))
    if label:
        labels.append(label)
    return build_row(
        source_type="google_keep",
        title=title,
        text="\n\n".join(text_parts).strip(),
        path=path,
        label=label,
        imported_at=imported_at,
        color=str(row.get("color") or ""),
        labels=unique(labels),
        created_at=timestamp_from_keep_usec(row.get("createdTimestampUsec")),
        updated_at=timestamp_from_keep_usec(row.get("userEditedTimestampUsec")),
        metadata={
            "is_archived": bool(row.get("isArchived")),
            "is_trashed": bool(row.get("isTrashed")),
        },
    )


def build_row(
    *,
    source_type: str,
    title: str,
    text: str,
    path: Path,
    label: str,
    imported_at: str,
    color: str = "",
    labels: list[str] | None = None,
    created_at: str = "",
    updated_at: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_labels = unique([*(labels or []), *([label] if label else [])])
    return {
        "source_type": source_type,
        "title": title,
        "text": text.strip(),
        "color": color,
        "labels": row_labels,
        "created_at": created_at,
        "updated_at": updated_at,
        "source_path": str(path),
        "imported_at_utc": imported_at,
        "metadata": metadata or {},
    }


def first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("name") or item.get("label") or "").strip()
            else:
                text = str(item).strip()
            if text:
                labels.append(text)
        return labels
    return [str(value).strip()] if str(value).strip() else []


def timestamp_from_keep_usec(value: Any) -> str:
    try:
        usec = int(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(usec / 1_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(row: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "source_type": row.get("source_type"),
                "title": row.get("title"),
                "text": row.get("text"),
                "source_path": row.get("source_path"),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:18]
    return f"{row.get('source_type')}-{digest}"


def looks_like_poetry(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    short_lines = sum(len(line.split()) <= 9 for line in lines)
    return short_lines / len(lines) >= 0.62


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for row in rows:
        row_id = str(row.get("id") or stable_id(row))
        if row_id in seen:
            continue
        row["id"] = row_id
        seen.add(row_id)
        deduped.append(row)
    return deduped


def format_counts(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        source_type = str(row.get("source_type") or "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    return "Counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
