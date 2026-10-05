"""Runtime configuration; data and evaluator dependencies stay outside this repo."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    core_root: Path
    task_index: Path
    prediction_fps: float = 50.0
    groundtruth_fps: float = 50.0
    max_upload_bytes: int = 64 * 1024 * 1024
    max_video_frames: int = 16
    max_concurrent_evaluations: int = 2
    vlm_timeout_seconds: float = 120.0
    allowed_origin: str = "https://robosteer.github.io"

    @classmethod
    def from_env(cls) -> "Settings":
        def required(name: str) -> Path:
            value = os.environ.get(name)
            if not value:
                raise ValueError(f"{name} is required")
            return Path(value).expanduser().resolve()

        settings = cls(
            dataset_root=required("ROBOOSTEER_DATASET_ROOT"),
            core_root=required("ROBOOSTEER_CORE_ROOT"),
            task_index=required("ROBOOSTEER_TASK_INDEX"),
            prediction_fps=float(os.environ.get("ROBOOSTEER_PREDICTION_FPS", "50")),
            groundtruth_fps=float(os.environ.get("ROBOOSTEER_GROUNDTRUTH_FPS", "50")),
            max_upload_bytes=int(os.environ.get("ROBOOSTEER_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024))),
            max_video_frames=int(os.environ.get("ROBOOSTEER_MAX_VIDEO_FRAMES", "16")),
            max_concurrent_evaluations=int(os.environ.get("ROBOOSTEER_MAX_CONCURRENT_EVALUATIONS", "2")),
            vlm_timeout_seconds=float(os.environ.get("ROBOOSTEER_VLM_TIMEOUT_SECONDS", "120")),
        )
        if not settings.dataset_root.is_dir():
            raise ValueError("ROBOOSTEER_DATASET_ROOT must be a directory")
        if not (settings.core_root / "scripts" / "level2" / "ir.py").is_file():
            raise ValueError("ROBOOSTEER_CORE_ROOT must contain the existing evaluator scripts")
        if not settings.task_index.is_file():
            raise ValueError("ROBOOSTEER_TASK_INDEX must point to an existing SQLite index")
        if settings.prediction_fps <= 0 or settings.groundtruth_fps <= 0:
            raise ValueError("FPS values must be positive")
        if (settings.max_upload_bytes <= 0 or not 1 <= settings.max_video_frames <= 64
                or settings.max_concurrent_evaluations <= 0):
            raise ValueError("invalid upload or video frame limit")
        return settings
