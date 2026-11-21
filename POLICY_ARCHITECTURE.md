# Default expert architecture — user instruction, 2026-11-19

Tracking and recovery experts use the same architecture:
- Actor sensor: projected gravity (3), body angular velocity (3), joint position relative to home (29), joint velocity relative to home (29), previous action (29).
- History: ten consecutive 93-dimensional sensor frames.
- Current root height and linear velocity are excluded from Actor inputs.
- Live command: unchanged OfficialSM motion command (640 dimensions). Hidden get-up references, phase and simulator state never enter either expert.
- Plain MLP: 1570 -> 2048 -> 1024 -> 512 -> 256 -> 29, using the existing SM LayerNorm/GELU blocks and independent observation/command normalizers.
- Critic and world-model privileged states retain full state, including height and linear velocity.
- Gate remains its existing four-frame proprioceptive MLP and supervision; this architecture change applies to the two action experts.

New StableMimic runs must use a compatible ten-frame MLP seed checkpoint. Do not silently load a single-frame MultiEncoderAE checkpoint or bypass strict state-dict loading.

## One training entry, reset distribution only

Use `python -m supermimic.scripts.train_stable_official` for all new variants.
`--tracking-reset-probability 1` selects tracking-origin episodes only; `0` selects recovery-origin episodes only; `0.5` selects equal reset probability. All three keep the same dual-MLP Actor, gate and observation/history handling. Reset labels select the corresponding reward/termination regime; a successful recovery can still progress to transition and tracking. Existing single-actor training artifacts are retained as historical baselines, not the entry for future runs.
