# StableMimic on SuperMimic

G1 Recovery v021B trains a single policy to get up and return to tracking using
the official SuperMimic world-model backend.

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Clone this repository and the
pinned [mjlab_research](https://github.com/HybridRobotics/mjlab_research)
dependency as siblings.

```bash
git clone https://github.com/toonasinensis/stablemimic-supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git
git -C mjlab_research checkout dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
cd stablemimic-supermimic
uv sync
```

## Train G1 Recovery v021B

Place the prepared G1 reference clips in these directories before starting the
run. Each file is a 50 Hz NPZ in the robot's joint order.

| Directory | Training input |
| --- | --- |
| `data/g1/tracking/` | 40 retargeted LAFAN tracking clips |
| `data/g1/recovery/` | 200 G1 CCRP get-up clips retimed by 1.5× with a 3-second static standing tail |

Recovery resets sample a clip and start at its first frame. The actor gets 10
frames of proprioception and a continuously advancing LAFAN tracking command in
every phase.

The phase gate opens when the recovery reference enters its standing tail,
or at 8 seconds for longer clips. When the gate is open, an end-pose similarity
of at least 0.70 triggers Transition and pays the one-shot success bonus.
Transition blends the recovery and tracking rewards over 1.5 seconds.
The world model uses soft Recovery failure with temperature 0.01 and a
straight-through success bonus with temperature 0.03 after the gate opens. The
preset leaves the separate `recovery_stand_timeout` disabled.

The phase recovery baseline from [TARS job 35797](https://open.tars-ai.com/pai/distributed-training/detail/35797)
uses HEFT-inspired regularization, randomized PD gains, and twice the actor's
proprioception noise. The verified run used a single RTX 4090 with PyTorch
2.14.1+cu126, MuJoCo and MuJoCo Warp 3.16.0, and Warp 1.19.0.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled MUJOCO_GL=disable OMP_NUM_THREADS=8 \
uv run python -m supermimic.scripts.train_stable_official \
  --base-config configs/g1_recovery_v021b.json --from-scratch --robot g1 \
  --tracking-files 'data/g1/tracking/*.npz' \
  --recovery-files 'data/g1/recovery/*.npz' \
  --recovery-timing-gate phase --recovery-reset-mode clip_start \
  --heft-regularization --pd-gain-dr --obs-noise-scale 2 \
  --tracking-reset-probability 0.5 --episode-seconds 10 \
  --num-envs 4096 --iterations 100001 --save-interval 1000 --seed 42 \
  --output-dir runs/g1-recovery-phase-obs2
```

TensorBoard events are saved under `runs/g1-recovery-phase-obs2/tensorboard/`.

## Source and license

Based on [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)
at `abc124c1bd5976341034668af3d12c087f2f1f55`. See the [SuperMimic project
page](https://supermimic.github.io) and [paper](https://arxiv.org/abs/2610.28480).
SuperMimic acknowledges ControlVAE, robotic_world_model, and DreamerV3.

Copyright 2026 The SuperMimic Authors. The upstream Apache-2.0 license is
included in [LICENSE](LICENSE).
