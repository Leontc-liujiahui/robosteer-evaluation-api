"""Single-task adapter around the existing, authoritative IR₂ implementation."""

from __future__ import annotations

from pathlib import Path
import sys

from .config import Settings


class EvaluationFailure(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def evaluate_csv(settings: Settings, task: dict, clip: Path) -> dict:
    reference = (settings.dataset_root / task["groundtruth"]).resolve()
    if (not reference.is_relative_to(settings.dataset_root) or not reference.is_dir()
            or any(not (reference / name).is_file() for name in
                   ("joint_pos.csv", "body_pos.csv", "body_quat.csv"))):
        raise EvaluationFailure("REFERENCE_UNAVAILABLE", "Benchmark reference is unavailable.", 503)
    if str(settings.core_root) not in sys.path:
        sys.path.insert(0, str(settings.core_root))
    try:
        from scripts.evaluation.data.motion import MotionIndex, load_qpos_36
        from scripts.level2.ir import compute_ir2
    except ImportError as exc:
        raise EvaluationFailure("EVALUATOR_UNAVAILABLE", "Evaluator dependencies are unavailable.", 503) from exc

    try:
        # The existing loader is authoritative for numeric widths, finite values,
        # quaternion validity, and shortest shared frame count.
        load_qpos_36(clip)
    except (ValueError, OSError) as exc:
        raise EvaluationFailure("INVALID_MOTION", "Uploaded motion CSV files are invalid.") from exc

    task_id = task["task_id"]
    family = task["constraint_name"]
    prediction = MotionIndex(clip.parent, {task_id: clip}, {})
    groundtruth = MotionIndex(reference.parent, {task_id: reference}, {})
    metadata = {task_id: {
        "task_id": task_id,
        "task_family": family,
        "task_type": task["task_type"],
        "base_motion_groundtruth": str(reference),
    }}
    try:
        score = compute_ir2(
            family=family,
            prediction=prediction,
            base_groundtruth=groundtruth,
            task_metadata_by_id=metadata,
            prediction_fps=settings.prediction_fps,
            groundtruth_fps=settings.groundtruth_fps,
            device="cpu",
        )
    except FileNotFoundError as exc:
        raise EvaluationFailure("EVALUATOR_UNAVAILABLE", "Evaluator assets are unavailable.", 503) from exc
    except ImportError as exc:
        raise EvaluationFailure("EVALUATOR_UNAVAILABLE", "Evaluator dependencies are unavailable.", 503) from exc
    except (RuntimeError, ValueError, OSError) as exc:
        raise EvaluationFailure("EVALUATION_FAILED", "Motion could not be evaluated against this task.") from exc
    if score.sample_ids != [task_id] or len(score.sample_values) != 1:
        raise EvaluationFailure("EVALUATION_FAILED", "Motion could not be evaluated against this task.")
    value = float(score.sample_values[0])
    return {"satisfied": bool(value >= 1.0), "score": value, "message": "Constraint satisfied." if value >= 1.0 else "Constraint not satisfied."}
