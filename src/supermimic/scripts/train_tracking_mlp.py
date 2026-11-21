"""Official SM tracking with a ten-frame proprioceptive plain MLP actor."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.tasks.env_cfgs import gencoord_env_cfg


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--config-checkpoint", required=True)
  p.add_argument("--tracking-files", required=True)
  p.add_argument("--output-dir", required=True)
  p.add_argument("--iterations", type=int, required=True)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--save-interval", type=int, default=5000)
  a = p.parse_args()
  configure_torch_backends()
  # Read configuration only. All model/optimizer weights start fresh.
  parent = torch.load(a.config_checkpoint, map_location="cpu", weights_only=False)
  cfg = load_runner_cfg({"runner_cfg": parent["cfg"]})
  original = asdict(cfg)
  del parent
  cfg.actor_type = "mlp"
  cfg.actor_obs = "proprio"
  cfg.history_length = 10
  cfg.slot_mix = {"motion": 1.0, "velocity": 0.0}
  cfg.policy.noise = 0.0
  cfg.output_dir = a.output_dir
  cfg.run_name = Path(a.output_dir).name
  cfg.max_iterations = a.iterations
  cfg.save_interval = a.save_interval
  changed = {
    k: {"before": original[k], "after": v}
    for k, v in asdict(cfg).items()
    if original[k] != v
  }
  print("[INFO] Changes from OfficialSM config:", json.dumps(changed), flush=True)
  ecfg = gencoord_env_cfg()
  ecfg.commands["motion"].motion_file = a.tracking_files
  ecfg.commands["motion"].sampling_mode = (
    "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
  )
  ecfg.scene.num_envs = a.num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(ecfg, device="cuda:0"))
  runner = SuperMimicRunner(env, cfg, task_id="SuperMimic-GenCoord-G1", device="cuda:0")
  assert runner._representation.actor_obs_mode == "proprio"
  assert runner._representation.proprio_obs_dim(29) == 93
  assert not any(key.startswith("recovery") for key in ecfg.rewards)
  assert not hasattr(runner.motion_cmd, "phase")
  print(
    "[INFO] Tracking=1.0; actor=MLP; sensor=10x93; current height/linear velocity excluded",
    flush=True,
  )
  runner.run()


if __name__ == "__main__":
  main()
