# StableMimic on SuperMimic

This repository contains the StableMimic implementation used for the G1 and Roban
experiments on the official SuperMimic world-model backend. The checked-in
configuration comes from Roban Box v15 v003, the 100,000-iteration run with 33
tracking clips and 95 fall recovery references.

The actor receives 10 frames of proprioception and a continuously advancing
tracking reference in every phase. Recovery references supervise the recovery
reward and provide privileged inputs to the critic. Tracking and Recovery each
account for half of resets. After the recovery end-pose check passes, a 1.5-second
transition blends the recovery and tracking rewards. The Roban v003 configuration
uses zero action noise, a 20-second episode cap, and up to ±30° of Recovery
joint-angle randomization within the robot's soft limits.

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Clone this repository and
the pinned [mjlab_research](https://github.com/HybridRobotics/mjlab_research)
dependency as siblings.

```bash
git clone git@github.com:toonasinensis/stablemimic-supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git
git -C mjlab_research checkout dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
cd stablemimic-supermimic
uv sync
```

Provide the local Roban Box v15 training XML with its meshes and the matching
same-stem URDF. The URDF supplies the actuator effort and speed limits. Set
`ROBAN_RECOVERY_XML` to that XML. The motion loader consumes 50 Hz NPZ files with
`fps`, `joint_names`, `root_pos_w`, `root_quat_w` (wxyz), `root_lin_vel_w`,
`root_ang_vel_w`, `joint_pos`, and `joint_vel`. Robot assets, reference data, and
checkpoints are local inputs.

## Reproduce Roban v003

If you have the v002 source recovery clips and their manifest, prepare the v003
references. The script filters out clip 65 and the ground-source clips, retimes
the remaining fall clips by 1.5× at 50 Hz, and keeps the 3-second static tail.
`configs/roban_boxv15_fall95_manifest.json` records the original clip numbers and
the corresponding input and output digests.

```bash
export ROBAN_RECOVERY_XML="$PWD/assets/roban_handball_boxv15_sm.xml"
uv run python scripts/retime_recovery.py \
  --source-dir data/recovery_v002 \
  --source-manifest data/recovery_v002_manifest.json \
  --robot-xml "$ROBAN_RECOVERY_XML" \
  --output-dir data/recovery_slow15
```

Run the same scratch-training configuration on one CUDA GPU. The original
checkpoint only supplied configuration values to the scratch run. The exported
configuration supplies those values directly.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl \
uv run python -m supermimic.scripts.train_stable_official \
  --base-config configs/roban_boxv15_v003.json --from-scratch --robot roban \
  --tracking-files 'data/tracking/*.npz' \
  --recovery-files 'data/recovery_slow15/*.npz' \
  --tracking-reset-probability 0.5 --recovery-reset-mode clip_start \
  --recovery-joint-noise-deg 30 --episode-seconds 20 \
  --num-envs 4096 --iterations 100001 --save-interval 1000 --seed 42 \
  --output-dir runs/roban-boxv15-v003
```

TensorBoard events are in the run's `tensorboard/` directory. For CPU playback
with the native MuJoCo viewer, keep `config.json` next to the checkpoint.

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=glfw TORCHDYNAMO_DISABLE=1 \
uv run python scripts/play_roban_cpu.py \
  --checkpoint runs/roban-boxv15-v003/model_100000.pt \
  --motion-files 'data/recovery_slow15/*.npz' \
  --tracking-files 'data/tracking/*.npz' --no-early-reset
```

## Source and license

Based on [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)
at `abc124c1bd5976341034668af3d12c087f2f1f55`. See the [SuperMimic project
page](https://supermimic.github.io) and [paper](https://arxiv.org/abs/2610.28480).
SuperMimic acknowledges ControlVAE, robotic_world_model, and DreamerV3.

Copyright 2026 The SuperMimic Authors. The upstream Apache-2.0 license is
included in [LICENSE](LICENSE).

## G1 Recovery v021B

This configuration uses 40 LAFAN tracking clips and 200 CCRP recovery clips
at 50 Hz. The recovery clips are retimed by 1.5× and include a 3-second static
standing tail. Each Recovery episode starts at the first frame of a clip.

The phase gate opens when the recovery reference enters its standing tail,
or at 8 seconds for longer clips. When the gate is open, an end-pose similarity
of at least 0.70 triggers Transition and pays the one-shot success bonus.
Transition blends the recovery and tracking rewards over 1.5 seconds.
The world model uses soft Recovery failure with temperature 0.01 and a
straight-through success bonus with temperature 0.03 after the gate opens.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=egl \
uv run python -m supermimic.scripts.train_stable_official \
  --base-config configs/g1_recovery_v021b.json --from-scratch --robot g1 \
  --tracking-files 'data/g1/tracking/*.npz' \
  --recovery-files 'data/g1/recovery/*.npz' \
  --recovery-timing-gate phase --recovery-reset-mode clip_start \
  --tracking-reset-probability 0.5 --episode-seconds 10 \
  --num-envs 4096 --iterations 100001 --save-interval 1000 --seed 42 \
  --output-dir runs/g1-recovery-v021b
```
