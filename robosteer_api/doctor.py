"""Check that a CPU server can load every Level 2 evaluation dependency."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys

from .config import Settings
from .task_index import FAMILIES


def inspect(settings: Settings) -> dict:
    checks: dict[str, object] = {}
    for tool in ("ffmpeg", "ffprobe"):
        checks[tool] = shutil.which(tool) is not None

    connection = sqlite3.connect(settings.task_index.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        counts = dict(connection.execute(
            "SELECT constraint_name, COUNT(*) FROM tasks GROUP BY constraint_name"
        ))
    finally:
        connection.close()
    checks["indexed_tasks"] = {family: counts.get(family, 0) for family in sorted(FAMILIES)}

    if str(settings.core_root) not in sys.path:
        sys.path.insert(0, str(settings.core_root))
    try:
        from scripts.evaluation.data.motion import load_qpos_36  # noqa: F401
        from scripts.level2.ir import compute_ir2  # noqa: F401
        from scripts.level2.order_vllm import parse_order_output  # noqa: F401
        from scripts.level2.times_vllm import parse_times_output  # noqa: F401
        checks["evaluator_imports"] = True
    except (ImportError, OSError) as exc:
        checks["evaluator_imports"] = f"{type(exc).__name__}: {exc}"

    try:
        from scripts.evaluation.core_metric.g_MPJPE import GlobalMPJPEEvaluator
        GlobalMPJPEEvaluator(device="cpu")
        checks["g1_cpu_kinematics"] = True
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        checks["g1_cpu_kinematics"] = f"{type(exc).__name__}: {exc}"

    ok = (checks["ffmpeg"] and checks["ffprobe"]
          and all(counts.get(family, 0) > 0 for family in FAMILIES)
          and checks["evaluator_imports"] is True
          and checks["g1_cpu_kinematics"] is True)
    return {"ready": bool(ok), "checks": checks}


def main() -> None:
    report = inspect(Settings.from_env())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
