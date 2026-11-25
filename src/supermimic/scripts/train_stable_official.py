"""Train one reference-conditioned SM actor with tracking and recovery resets."""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.stable_official_runner import (
  StableOfficialRunner,
  StableOfficialRunnerCfg,
)
from supermimic.tasks.stable_official import (
  recovery_stand_timed_out,
  stable_official_cfg,
)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  source = p.add_mutually_exclusive_group(required=True)
  source.add_argument("--seed-checkpoint")
  source.add_argument(
    "--base-config", type=Path, help="JSON runner config for a scratch run."
  )
  p.add_argument("--resume-checkpoint")
  p.add_argument("--obs-noise-scale", type=float, default=1.0)
  p.add_argument("--recovery-timing-gate", choices=("off", "phase"), default="off")
  p.add_argument(
    "--heft-regularization",
    action="store_true",
    help="Use the three HEFT-inspired G1 regularizers.",
  )
  p.add_argument(
    "--pd-gain-dr",
    action="store_true",
    help="Randomize G1 Kp and Kd independently at every reset.",
  )
  p.add_argument("--from-scratch", action="store_true")
  p.add_argument("--robot", choices=("g1", "roban"), default="g1")
  p.add_argument(
    "--recovery-reset-mode", choices=("frame", "clip_start"), default="frame"
  )
  p.add_argument("--recovery-joint-noise-deg", type=float)
  p.add_argument("--episode-seconds", type=float)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--tracking-files", required=True)
  p.add_argument("--recovery-files", required=True)
  p.add_argument(
    "--recovery-standing-tail-s",
    type=float,
    default=0.0,
    help="Duration of the static standing hold appended to every recovery clip.",
  )
  p.add_argument("--output-dir", required=True)
  p.add_argument("--iterations", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--save-interval", type=int, default=5000)
  p.add_argument("--tracking-reset-probability", type=float, default=0.5)
  a = p.parse_args()
  if not math.isfinite(a.obs_noise_scale) or a.obs_noise_scale < 0:
    p.error("--obs-noise-scale must be finite and nonnegative")
  if a.base_config is not None and not a.from_scratch:
    p.error("--base-config requires --from-scratch")
  if a.robot != "g1" and (a.heft_regularization or a.pd_gain_dr):
    p.error("--heft-regularization and --pd-gain-dr require --robot g1")
  if a.pd_gain_dr and not a.from_scratch:
    p.error(
      "--pd-gain-dr requires --from-scratch because the world model inputs change"
    )
  if not 0.0 <= a.tracking_reset_probability <= 1.0:
    p.error("--tracking-reset-probability must be between 0 and 1")
  if not 0 <= a.recovery_standing_tail_s < float("inf"):
    p.error("--recovery-standing-tail-s must be finite and nonnegative")
  if (
    a.recovery_joint_noise_deg is not None
    and not 0 <= a.recovery_joint_noise_deg < float("inf")
  ):
    p.error("--recovery-joint-noise-deg must be finite and nonnegative")
  if a.episode_seconds is not None and not 0 < a.episode_seconds < float("inf"):
    p.error("--episode-seconds must be finite and positive")
  configure_torch_backends()
  torch.manual_seed(a.seed)
  if a.base_config is not None:
    base = load_runner_cfg(json.loads(a.base_config.read_text()))
  else:
    parent = torch.load(a.seed_checkpoint, map_location="cpu", weights_only=False)
    base = load_runner_cfg({"runner_cfg": parent["cfg"]})
    for key in ("actor", "world_model"):
      tensors = (
        parent[key].values() if key == "actor" else parent[key]["dynamics"].values()
      )
      if not all(torch.isfinite(t).all() for t in tensors if t.is_floating_point()):
        raise RuntimeError(f"Nonfinite parent {key}")
  cfg = StableOfficialRunnerCfg(
    **{f.name: getattr(base, f.name) for f in fields(base)},
    seed_checkpoint=a.seed_checkpoint or "",
  )
  cfg.from_scratch = a.from_scratch
  cfg.recovery_reset_mode = a.recovery_reset_mode
  cfg.seed = a.seed
  cfg.recovery_probability = 1.0 - a.tracking_reset_probability
  cfg.resume_checkpoint = a.resume_checkpoint
  cfg.heft_regularization = a.heft_regularization
  cfg.pd_gain_dr = a.pd_gain_dr
  cfg.obs_noise_scale = a.obs_noise_scale
  cfg.output_dir = a.output_dir
  cfg.run_name = a.output_dir.rsplit("/", 1)[-1]
  cfg.max_iterations = a.iterations
  cfg.save_interval = a.save_interval
  assert (
    cfg.policy.noise == 0
    and cfg.history_length == 10
    and cfg.actor_obs == "proprio"
    and cfg.actor_type == "mlp"
  )
  if a.seed_checkpoint is not None:
    del parent
  ecfg = stable_official_cfg(
    a.tracking_files,
    a.recovery_files,
    robot=a.robot,
    heft_regularization=cfg.heft_regularization,
    pd_gain_dr=cfg.pd_gain_dr,
  )
  ecfg.commands["motion"].recovery_probability = cfg.recovery_probability
  ecfg.commands["motion"].recovery_reset_mode = cfg.recovery_reset_mode
  ecfg.commands["motion"].recovery_timing_gate = a.recovery_timing_gate
  if a.recovery_joint_noise_deg is not None:
    ecfg.commands["motion"].recovery_joint_position_noise = math.radians(
      a.recovery_joint_noise_deg
    )
  if a.episode_seconds is not None:
    ecfg.episode_length_s = a.episode_seconds
    ecfg.commands["motion"].recovery_timeout_s = a.episode_seconds
  ecfg.commands["motion"].recovery_standing_tail_s = a.recovery_standing_tail_s
  if a.recovery_standing_tail_s:
    ecfg.terminations["recovery_stand_timeout"] = TerminationTermCfg(
      func=recovery_stand_timed_out
    )
  ecfg.seed = a.seed
  ecfg.scene.num_envs = a.num_envs
  print(
    f"[INFO] Single-MLP runner: tracking reset probability={a.tracking_reset_probability}, "
    f"recovery reset probability={cfg.recovery_probability}; "
    "10-frame proprio history + tracking command in every phase; "
    f"HEFT regularization={cfg.heft_regularization}, PD gain DR={cfg.pd_gain_dr}; "
    f"actor observation noise scale={cfg.obs_noise_scale:g}",
    flush=True,
  )
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(ecfg, device="cuda:0"))
  StableOfficialRunner(
    env, cfg, task_id=f"SuperMimic-StableMimic-Official-{a.robot}", device="cuda:0"
  ).run()


if __name__ == "__main__":
  main()
