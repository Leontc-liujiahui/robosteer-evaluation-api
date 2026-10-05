# Shared evaluation engine

This package contains reusable data loading, encoders, conventional metrics,
and runtime code. It is intentionally not the public evaluation interface.

Use the three level-specific launchers under `scripts/examples/`:

- `run_level1.sh` passes explicit Level-1 prediction, motion GT and condition;
- `run_level2.sh` passes a constrained prediction plus the corresponding
  Level-1 base prediction, GT and condition; and
- `run_level3.sh` evaluates text/audio/video generation segments and duration-weights their BS_level2 scores; image boundaries are excluded.

The legacy `scripts/evaluation/evaluate.py` and `scripts/evaluation/run_evaluation.sh` are retained here only for
legacy conventional-metric runs. New benchmark results should use the
level-specific entry points so that IR and BS are computed with the correct
hierarchy.
