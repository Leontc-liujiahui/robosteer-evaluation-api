"""Locate the OMG dependency in a source checkout or a downloaded model bundle."""

from pathlib import Path


def omg_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    bundled = project_root / "vendor" / "OMG"
    if (bundled / "src" / "omg" / "__init__.py").is_file():
        return bundled
    bundle = project_root / "models-hf"
    for name in (
        "motion_encoder", "text_motion_evaluator", "audio_motion_evaluator",
        "rhythm_motion_evaluator", "trajectory_motion_evaluator",
        "video_motion_evaluator_human", "video_motion_evaluator_skel",
    ):
        candidate = bundle / name / "omg"
        if (candidate / "src" / "omg" / "__init__.py").is_file():
            return candidate
    return project_root.parent / "cxt" / "OMG"
