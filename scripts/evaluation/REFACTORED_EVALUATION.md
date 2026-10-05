# Level-aware evaluation contract

- Level 1 receives explicit `prediction`, `motion-groundtruth`, and
  `condition-groundtruth` paths. Task JSON is used only for timing, condition
  manifest materialization, and the task-specific IR_1 rule.
- Level 2 receives its constrained prediction plus explicit Level-1 base
  prediction, GT, and condition. The evaluator restricts those three base
  inputs to IDs represented by the Level-2 task before computing BG, fixes
  IR_1 to 1, and then computes the family-specific IR_2.
- Level 3 treats every non-image text/audio/video item as a generation subtask
  with `IR_1 = IR_2 = 1`, scores each modality cohort as `BS_level2 = BG`, and
  duration-weights those scores within a task before macro-averaging.

The public scripts are in `scripts/level1`, `scripts/level2`,
`scripts/level3`, and `scripts/examples`; shared implementation lives here.
