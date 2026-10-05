# Evaluation architecture

The active evaluation path is:

```text
scripts/evaluation/run_evaluation.sh -> scripts/evaluation/evaluate.py -> pipeline/
                                      |-- data/
                                      |-- preprocessing/
                                      |-- encoders/
                                      `-- core_metric/
```

The three input streams are independent:

- `--prediction`: generated motion CSV directories;
- `--motion-groundtruth`: optional reference motion CSV directories;
- `--instruction-groundtruth`: optional instruction `manifest.json`.

Selected metrics determine which optional stream is required. FID requires
motion ground truth, MM-Distance requires instruction ground truth, and
Diversity requires neither.

There is no instruction-modality CLI argument. The instruction manifest names
the registered encoder:

```json
{
  "schema": "liujiahui.instruction.v1",
  "encoder": "rhythm_encoder",
  "samples": [
    {"sample_id": "example_00001", "path": "features/example_00001.npy"}
  ]
}
```

Encoder definitions are maintained directly in
`liujiahui/models/registry.json`. Checkpoint paths in that file are relative to
the registry directory.

Run:

```bash
bash liujiahui/scripts/evaluation/run_evaluation.sh \
  --prediction liujiahui/prediction/ListenDenoiseAction/MUL_RHY \
  --motion-groundtruth liujiahui/groundtruth/motion_generation \
  --output liujiahui/results/rhythm_generation/LDA \
  --metrics FID,Diversity
```

Use `--instruction-groundtruth` when the selected metric requires it. Use
`--list-metrics` to inspect implemented metric dependencies.
