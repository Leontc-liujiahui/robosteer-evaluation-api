"""Build and query a small, server-side index of authoritative Level 2 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


CSV_FAMILIES = {"speed", "amplitude", "direction", "trajectory", "body_restrain"}
VIDEO_FAMILIES = {"order", "times"}
FAMILIES = CSV_FAMILIES | VIDEO_FAMILIES


def normalize_family(value: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", value.casefold())
    return "body_restrain" if name == "bodyrestrain" else name


def lookup(index_path: Path, task_id: str) -> dict | None:
    connection = sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def _paired_text_id(task_id: str, family: str) -> str:
    for modality in ("audio", "video"):
        prefix = f"L2_{family}_{modality}_"
        if task_id.startswith(prefix):
            return f"L2_{family}_text_{task_id[len(prefix):]}"
    return task_id


def _order_labels(task_id: str, text_rows: dict[str, str]) -> tuple[str, str, str] | None:
    text_id = _paired_text_id(task_id, "order")
    match = re.fullmatch(r"(.+)_p[01]", text_id)
    if match is None:
        return None
    p1_prompt = text_rows.get(match.group(1) + "_p1")
    p0_prompt = text_rows.get(match.group(1) + "_p0")
    if p1_prompt is None or p0_prompt is None:
        return None
    parsed = re.fullmatch(r"do (.+) after doing (.+)", p1_prompt.strip())
    if parsed is None:
        return None
    second, first = (part.strip() for part in parsed.groups())
    if not first or not second or p0_prompt.strip() != f"{first} then {second}":
        return None
    sample_match = re.fullmatch(r"L2_order_(?:text|audio|video)_(.+)", task_id)
    if sample_match is None:
        return None
    sample_id = sample_match.group(1)
    even = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 2 == 0
    return (first, second, "A") if even else (second, first, "B")


def _times_labels(task_id: str, text_rows: dict[str, str]) -> tuple[str, int] | None:
    prompt = text_rows.get(_paired_text_id(task_id, "times"))
    if prompt is None:
        return None
    parsed = re.fullmatch(r"repeat (.+) ([234]) times", prompt.strip())
    if parsed is None or not parsed.group(1).strip():
        return None
    count = int(parsed.group(2))
    if not task_id.endswith(f"_{count}x"):
        return None
    return parsed.group(1).strip(), count


def build_index(dataset_root: Path, destination: Path) -> dict[str, int]:
    dataset_root = dataset_root.expanduser().resolve()
    task_root = dataset_root / "Tasks" / "Level2"
    if not task_root.is_dir():
        raise FileNotFoundError(f"missing Level 2 task tree: {task_root}")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".building")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    counts = {name: 0 for name in FAMILIES}
    try:
        connection.execute("""CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, constraint_name TEXT NOT NULL, task_type TEXT,
            groundtruth TEXT, action_a TEXT, action_b TEXT, target_first TEXT,
            action TEXT, target_count INTEGER
        )""")
        text_rows: dict[str, str] = {}
        videos: list[tuple[str, str, str]] = []
        for path in sorted(task_root.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            task_id = metadata["task_id"]
            family = normalize_family(str(metadata["task_family"]))
            if family not in FAMILIES:
                continue
            task_type = str(metadata.get("task_type", ""))
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"invalid task ID in {path}")
            if family in CSV_FAMILIES:
                raw = payload["ground_truth"]["motion_parameters"]
                reference = (dataset_root / raw).resolve()
                if not reference.is_relative_to(dataset_root) or not reference.is_dir():
                    raise ValueError(f"reference motion missing or outside dataset: {path}")
                for filename in ("joint_pos.csv", "body_pos.csv", "body_quat.csv"):
                    if not (reference / filename).is_file():
                        raise ValueError(f"reference motion missing {filename}: {path}")
                connection.execute(
                    "INSERT INTO tasks (task_id,constraint_name,task_type,groundtruth) VALUES (?,?,?,?)",
                    (task_id, family, task_type, str(reference.relative_to(dataset_root))),
                )
            else:
                videos.append((task_id, family, task_type))
                if task_id.startswith(f"L2_{family}_text_"):
                    prompt = payload.get("input", {}).get("modalities", {}).get("text")
                    if isinstance(prompt, str):
                        text_rows[task_id] = prompt
        for task_id, family, task_type in videos:
            if family == "order":
                labels = _order_labels(task_id, text_rows)
                if labels is None:
                    continue
                connection.execute(
                    "INSERT INTO tasks (task_id,constraint_name,task_type,action_a,action_b,target_first) VALUES (?,?,?,?,?,?)",
                    (task_id, family, task_type, *labels),
                )
            else:
                labels = _times_labels(task_id, text_rows)
                if labels is None:
                    continue
                connection.execute(
                    "INSERT INTO tasks (task_id,constraint_name,task_type,action,target_count) VALUES (?,?,?,?,?)",
                    (task_id, family, task_type, *labels),
                )
        connection.commit()
        for family in counts:
            counts[family] = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE constraint_name = ?", (family,)
            ).fetchone()[0]
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    temporary.replace(destination)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Index RoboSteer Level 2 task metadata")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_index(args.dataset_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
