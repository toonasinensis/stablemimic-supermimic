"""Single-environment Roban playback in the native CPU MuJoCo viewer."""

import argparse
import copy
import glob
import json
import math
import queue
import time
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from tensordict import TensorDict

from supermimic.models import PlainMotionActor
from supermimic.representations import (
  NUM_GOAL_FRAMES,
  _assemble_goal,
  goal_window_indices,
)
from supermimic.runners.stable_official_runner import add_xy
from supermimic.tasks.roban_recovery import get_roban_xml_path
from supermimic.tasks.stable_official import (
  RECOVERY,
  RECOVERY_KEYS,
  TRACKING,
  TRANSITION,
  RecoveryObjective,
  StableOfficialMotionCfg,
  recovery_scores,
  recovery_similarity_terms,
  stable_official_cfg,
)


def joint_values(value, names):
  if isinstance(value, (int, float)):
    return np.full(len(names), value, dtype=np.float64)
  result = np.zeros(len(names))
  ids, _, values = resolve_matching_names_values(value, names)
  result[ids] = values
  return result


class Player:
  def __init__(
    self,
    checkpoint,
    motion_files,
    motion_index,
    start_frame,
    tracking_file,
    episode_seconds=None,
    early_reset=True,
  ):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained = saved["cfg"]
    self.iteration = saved["iteration"]
    assert trained["actor_type"] == "mlp" and trained["actor_obs"] == "proprio"
    cfg = stable_official_cfg(tracking_file, motion_files, robot="roban")
    cfg.scene.num_envs = 1
    cfg.scene.sensors = ()
    scene = Scene(cfg.scene, device="cpu")
    self.model = scene.compile()
    cfg.sim.mujoco.apply(self.model)
    self.data = mj.MjData(self.model)
    # Rendering-only model: never used by physics or the policy.
    self.ghost_model = copy.deepcopy(self.model)
    self.ghost_model.geom_matid[:] = -1
    self.ghost_model.geom_rgba[:] = (0.15, 0.65, 1.0, 0.4)
    self.recovery_ghost_model = copy.deepcopy(self.ghost_model)
    self.recovery_ghost_model.geom_rgba[:] = (1.0, 0.48, 0.12, 0.4)
    robot = scene["robot"]
    action = cfg.actions["joint_pos"]
    _, names = robot.find_joints_by_actuator_names(action.actuator_names)
    self.names = list(names)
    num_joints = len(self.names)
    assert num_joints == 21
    sensor_dim = 6 + 3 * num_joints
    goal_dim = NUM_GOAL_FRAMES * (2 * num_joints + 6)
    self.actor = PlainMotionActor(
      sensor_dim,
      num_joints,
      SimpleNamespace(name="motion", goal_dim=goal_dim),
      tuple(trained["actor_hidden_dims"]),
      trained["history_length"],
    )
    self.actor.load_state_dict(saved["actor"], strict=True)
    self.actor.eval()
    joint_ids = np.array([self.model.joint("robot/" + n).id for n in names])
    self.qpos_ids = self.model.jnt_qposadr[joint_ids]
    self.qvel_ids = self.model.jnt_dofadr[joint_ids]
    self.ctrl_ids = np.array(
      [np.flatnonzero(self.model.actuator_trnid[:, 0] == j).item() for j in joint_ids]
    )
    urdf = ElementTree.parse(get_roban_xml_path().with_suffix(".urdf")).getroot()
    limits = {
      joint.attrib["name"]: joint.find("limit") for joint in urdf.findall("joint")
    }
    self.speed_limit = np.array(
      [float(limits[name].attrib["velocity"]) for name in self.names]
    )
    self.force_limit = np.array(
      [float(limits[name].attrib["effort"]) for name in self.names]
    )
    np.testing.assert_allclose(
      self.model.actuator_forcerange[self.ctrl_ids],
      np.column_stack((-self.force_limit, self.force_limit)),
    )
    joint_pos = robot.cfg.init_state.joint_pos
    # Match mjlab's EntityData, which uses the scene keyframe only as a fallback.
    self.home = (
      joint_values(joint_pos, names)
      if joint_pos is not None
      else self.model.key("init_state").qpos[self.qpos_ids].copy()
    )
    self.home_vel = joint_values(robot.cfg.init_state.joint_vel, names)
    self.scale = joint_values(action.scale, names)
    self.offset = (
      self.home if action.use_default_offset else joint_values(action.offset, names)
    )
    self.base = self.model.body("robot/base_link").id
    self.score_body_ids = np.array(
      [
        self.model.body("robot/" + name).id
        for name in cfg.commands["motion"].body_names
      ]
    )
    run_config = json.loads((Path(checkpoint).parent / "config.json").read_text())
    env_cfg = run_config["env_cfg"]
    saved_motion = env_cfg["commands"]["motion"]
    score_cfg = StableOfficialMotionCfg(
      entity_name=cfg.commands["motion"].entity_name,
      resampling_time_range=cfg.commands["motion"].resampling_time_range,
      motion_file=tracking_file,
      body_names=cfg.commands["motion"].body_names,
      anchor_body_name=cfg.commands["motion"].anchor_body_name,
      **{
        name: saved_motion[name]
        for name in (
          "failure_similarity",
          "terminal_similarity",
          "error_timeout_s",
          "recovery_timeout_s",
          "transition_s",
          "recovery_lift",
          "recovery_multiplier",
          "success_bonus",
          "transition_coefficient",
          "recovery_standing_tail_s",
          "recovery_stand_timeout_s",
        )
      },
    )
    self.score_cfg = score_cfg
    gyro = self.model.sensor("robot/imu_ang_vel").adr[0]
    self.gyro = slice(gyro, gyro + 3)
    self.decimation = cfg.decimation
    self.period = self.model.opt.timestep * self.decimation
    limit = score_cfg.recovery_timeout_s if episode_seconds is None else episode_seconds
    self.max_steps = max(1, math.ceil(limit / self.period))
    self.early_reset = early_reset
    self.fk = ForwardKinematics(
      self.model,
      torch.tensor(self.score_body_ids),
      torch.tensor(self.qpos_ids),
      torch.arange(len(self.score_body_ids)),
    )
    self.tracking = MotionLoader(tracking_file, self.fk)
    self.tracking_name = Path(tracking_file).name
    print(f"Tracking command: {self.tracking_name}", flush=True)
    self.tracking_time = torch.zeros(1, dtype=torch.long)
    self.rng = np.random.default_rng(42)
    self.pos_limit = cfg.terminations["anchor_pos"].params["threshold"]
    self.ori_limit = cfg.terminations["anchor_ori"].params["threshold"]
    self.history = np.zeros((trained["history_length"], sensor_dim), dtype=np.float32)
    self.last_action = np.zeros(num_joints, dtype=np.float32)
    self.paths = sorted(glob.glob(motion_files))
    if not self.paths:
      raise FileNotFoundError(motion_files)
    self.start_frame = start_frame
    self.load_clip(motion_index)

  def load_clip(self, index):
    self.index = index % len(self.paths)
    self.path = Path(self.paths[self.index])
    with np.load(self.path, allow_pickle=False) as clip:
      assert clip["joint_names"].tolist() == self.names
      assert abs(float(clip["fps"].item()) - 1 / self.period) < 1e-3
      self.clip = {
        k: clip[k].astype(np.float32)
        for k in (
          "root_pos_w",
          "root_quat_w",
          "root_lin_vel_w",
          "root_ang_vel_w",
          "joint_pos",
          "joint_vel",
        )
      }
    self.length = len(self.clip["joint_pos"])
    self.recovery = MotionLoader(str(self.path), self.fk)
    self.recovery.cache_body_states()
    self.objective = RecoveryObjective(self.score_cfg, self.recovery, self.period)
    self.terminal_reference = self.recovery.body_state(torch.tensor([self.length - 1]))
    self.reset()
    print(f"Clip {self.index + 1}/{len(self.paths)}: {self.path.name}", flush=True)

  def draw_reference(self, viewer, visualizer, side_by_side):
    phase = int(self.state[1])
    idx = int(self.tracking_time)
    pose = self.data.qpos.copy()
    pose[:3] = self.tracking.root_pos_w[idx].numpy()
    pose[:2] += self.state[4][0].numpy()
    pose[3:7] = self.tracking.root_quat_w[idx].numpy()
    pose[self.qpos_ids] = self.tracking.joint_pos[idx].numpy()
    if side_by_side:
      pose[1] += 1.1
    visualizer.add_ghost_entity(pose, self.ghost_model)

    if phase in (RECOVERY, TRANSITION):
      frame = min(self.frame, self.length - 1) if phase == RECOVERY else self.length - 1
      pose[:3] = self.clip["root_pos_w"][frame]
      pose[3:7] = self.clip["root_quat_w"][frame]
      pose[self.qpos_ids] = self.clip["joint_pos"][frame]
      if side_by_side:
        pose[1] -= 1.1
      visualizer.add_ghost_entity(pose, self.recovery_ghost_model)
    # Keep the reference out of native mouse selection/perturbation.
    for i in range(viewer.user_scn.ngeom):
      geom = viewer.user_scn.geoms[i]
      geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
      geom.objid = -1
      geom.category = mj.mjtCatBit.mjCAT_DECOR

  def score_body_state(self, data):
    ids = self.score_body_ids
    pos, quat = data.xpos[ids], data.xquat[ids]
    cvel = data.cvel[ids]
    # Same link-origin world velocities used by mjlab EntityData.
    linear = cvel[:, 3:] - np.cross(cvel[:, :3], data.subtree_com[self.base] - pos)
    return tuple(
      torch.from_numpy(a.astype(np.float32)[None])
      for a in (pos, quat, linear, cvel[:, :3])
    )

  def sensor(self):
    d = self.data
    return np.concatenate(
      (
        -d.xmat[self.base, 6:9],
        d.sensordata[self.gyro],
        d.qpos[self.qpos_ids] - self.home,
        d.qvel[self.qvel_ids] - self.home_vel,
        self.last_action,
      )
    ).astype(np.float32)

  def reset(self):
    m, d = self.model, self.data
    self.frame = min(self.start_frame, self.length - 1)
    k, c = self.frame, self.clip
    mj.mj_resetData(m, d)
    m.actuator_forcerange[self.ctrl_ids] = np.column_stack(
      (-self.force_limit, self.force_limit)
    )
    d.qpos[:3] = c["root_pos_w"][k]
    d.qpos[3:7] = c["root_quat_w"][k]
    d.qpos[self.qpos_ids] = c["joint_pos"][k]
    d.qvel[:3] = c["root_lin_vel_w"][k]
    mj.mju_rotVecQuat(
      d.qvel[3:6],
      c["root_ang_vel_w"][k].astype(np.float64),
      d.qpos[3:7] * (1, -1, -1, -1),
    )
    d.qvel[self.qvel_ids] = c["joint_vel"][k]
    d.qpos[2] += self.score_cfg.recovery_lift
    d.ctrl[self.ctrl_ids] = d.qpos[self.qpos_ids]
    mj.mj_forward(m, d)
    self.last_action.fill(0)
    self.history[:] = self.sensor()
    self.steps = 0
    self.tracking_time = torch.zeros_like(self.tracking_time)
    self.ignored_terminations = set()
    self.state = (
      torch.tensor([k]),
      torch.tensor([RECOVERY]),
      torch.zeros(1, dtype=torch.long),
      torch.zeros(1, dtype=torch.long),
      torch.zeros(1, 2),
      torch.zeros(1, dtype=torch.bool),
      torch.ones(1, dtype=torch.bool),
      torch.zeros(1, dtype=torch.long),
    )
    self.min_score, self.longest_bad = 1.0, 0

  def physical_state(self):
    pos, quat, lin, ang = self.score_body_state(self.data)
    return (pos, matrix_from_quat(quat), lin, ang), torch.cat(
      (pos[:, 0], quat[:, 0]), dim=-1
    )

  @torch.inference_mode()
  def step(self):
    m, d = self.model, self.data
    self.history[:-1] = self.history[1:]
    self.history[-1] = self.sensor()
    _, aux = self.physical_state()
    idx = goal_window_indices(self.tracking, self.tracking_time)
    root = add_xy(aux, self.state[4], -1)
    content = torch.cat(
      (self.tracking.joint_pos[idx], self.tracking.joint_vel[idx]), dim=-1
    ).flatten(1)
    _, command = _assemble_goal(
      content,
      root[:, :3],
      root[:, 3:7],
      self.tracking.root_pos_w[idx],
      self.tracking.root_quat_w[idx],
    )
    obs = TensorDict(
      {"history": torch.from_numpy(self.history[None]), "goal_motion": command},
      batch_size=[1],
    )
    self.last_action[:] = self.actor(obs)[0].numpy()
    d.ctrl[self.ctrl_ids] = self.last_action * self.scale + self.offset
    for _ in range(self.decimation):
      speed = d.qvel[self.qvel_ids]
      m.actuator_forcerange[self.ctrl_ids, 0] = np.where(
        speed < -self.speed_limit, 0.0, -self.force_limit
      )
      m.actuator_forcerange[self.ctrl_ids, 1] = np.where(
        speed > self.speed_limit, 0.0, self.force_limit
      )
      mj.mj_step(m, d)
    mj.mj_forward(m, d)
    self.steps += 1
    bodies, aux = self.physical_state()
    frame, phase = self.state[:2]
    ref = self.recovery.body_state((frame + 1).clamp(max=self.length - 1))
    instant = recovery_similarity_terms(recovery_scores(bodies, ref)[0])
    score = float(instant.sum())
    end_terms = recovery_similarity_terms(
      recovery_scores(bodies, self.terminal_reference)[0]
    )
    terminal = float(end_terms.sum())
    _, _, failed, _, self.state, success, completed, stand_timeout = (
      self.objective.advance(
        bodies,
        aux,
        self.tracking,
        self.tracking_time,
        self.state,
      )
    )
    motion_end = int(self.tracking_time) + 1 >= self.tracking.time_step_total
    self.tracking_time = (self.tracking_time + 1).clamp(
      max=self.tracking.time_step_total - 1
    )
    self.frame = int(self.state[0])
    count = int(self.state[2])
    if int(phase) == RECOVERY:
      self.min_score = min(self.min_score, score)
      self.longest_bad = max(self.longest_bad, count)
    reason = "recovery_error" if bool(failed) else None
    if reason is None and bool(stand_timeout):
      reason = "recovery_stand_timeout"
    timed_out = self.steps >= self.max_steps
    if int(phase) == TRACKING and reason is None:
      tracking_frame = int(self.tracking_time)
      target = self.tracking.root_pos_w[tracking_frame, 2]
      if abs(float(aux[0, 2] - target)) > self.pos_limit:
        reason = "anchor_pos"
      else:
        target_up = matrix_from_quat(self.tracking.root_quat_w[tracking_frame])[2, 2]
        if abs(float(bodies[1][0, 0, 2, 2] - target_up)) > self.ori_limit:
          reason = "anchor_ori"
      if reason is None and motion_end:
        reason = "motion_end"
    if reason and not self.early_reset:
      if reason not in self.ignored_terminations:
        print(
          f"IGNORED {reason} at {self.steps * self.period:.2f}s | early reset off",
          flush=True,
        )
        self.ignored_terminations.add(reason)
      reason = None
    if timed_out:
      reason = f"{reason}+time_out" if reason else "time_out"
    if self.steps % 25 == 0 or reason or bool(success) or bool(completed):
      phase_name = {
        RECOVERY: "recovery",
        TRANSITION: "transition",
        TRACKING: "tracking",
      }[int(phase)]
      print(
        f"t={self.steps * self.period:5.2f}s {phase_name} S_next={score:.4f} "
        f"S_end={terminal:.4f} bad={count}/{self.objective.failure_steps} "
        f"stand_wait={int(self.state[-1]) * self.period:.2f}s "
        f"{' -> transition' if bool(success) else ''}{' -> tracking' if bool(completed) else ''}",
        flush=True,
      )
      for label, terms in (("S_next", instant), ("S_end", end_terms)):
        parts = " ".join(
          f"{name}={value:.5g}"
          for name, value in zip(RECOVERY_KEYS[:4], terms.tolist(), strict=True)
        )
        print(f"  {label} contributions: {parts}", flush=True)
    if reason:
      print(
        f"RESET {reason} | min recovery S_next={self.min_score:.4f} "
        f"longest below {self.score_cfg.failure_similarity:g}: "
        f"{self.longest_bad * self.period:.2f}s",
        flush=True,
      )
      self.reset()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--motion-files", required=True)
  p.add_argument("--motion-index", type=int, default=0)
  p.add_argument("--start-frame", type=int, default=0)
  p.add_argument("--tracking-files", required=True)
  p.add_argument("--tracking-index", type=int, default=0)
  p.add_argument(
    "--episode-seconds", type=float, help="Override the CPU play episode time limit."
  )
  p.add_argument(
    "--early-reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Allow reasons other than timeout to reset the CPU playback.",
  )
  args = p.parse_args()
  if args.episode_seconds is not None and (
    not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0
  ):
    p.error("--episode-seconds must be finite and greater than zero")
  torch.set_num_threads(1)
  tracking_paths = sorted(glob.glob(args.tracking_files))
  if not tracking_paths:
    p.error(f"No tracking clips matched {args.tracking_files}")
  tracking_file = tracking_paths[args.tracking_index % len(tracking_paths)]
  player = Player(
    args.checkpoint,
    args.motion_files,
    args.motion_index,
    args.start_frame,
    tracking_file,
    args.episode_seconds,
    args.early_reset,
  )
  keys = queue.SimpleQueue()
  paused, speed, backlog = False, 1.0, 0.0
  show_reference, side_by_side = True, True
  m, d = player.model, player.data
  with mujoco.viewer.launch_passive(
    m, d, key_callback=keys.put, show_left_ui=False, show_right_ui=False
  ) as viewer:
    with viewer.lock():
      viewer.cam.lookat[:] = d.xpos[player.base] + np.array([0.0, 0.55, 0.0])
      viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.0, 135.0, -20.0
      viewer.opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = True
      viewer.opt.flags[mj.mjtVisFlag.mjVIS_PERTOBJ] = True
    visualizer = MujocoNativeDebugVisualizer(viewer.user_scn, m, 0)
    print(
      f"Loaded model_{player.iteration} | one Roban S2.2 CPU | Mimic PD, delay 0, S17 speed gate\n"
      f"Episode timeout: {player.max_steps * player.period:g}s | "
      f"Early reset: {'on' if player.early_reset else 'off'} (T toggles)\n"
      f"recovery_error: S_next < {player.score_cfg.failure_similarity:g} for "
      f"{player.score_cfg.error_timeout_s:g}s\n"
      "Blue = Actor tracking command in every phase. Orange = get-up reference for reward. R reset, N/P switch, F random, Space pause.",
      flush=True,
    )
    last = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      if not paused:
        backlog = min(backlog + (now - last) * speed, 0.12 * max(speed, 1))
      last = now
      with viewer.lock():
        while not keys.empty():
          key = keys.get()
          if key == glfw.KEY_SPACE:
            paused = not paused
            backlog = 0.0
          elif key in (glfw.KEY_R, glfw.KEY_ENTER):
            player.reset()
            backlog = 0.0
          elif key == glfw.KEY_T:
            player.early_reset = not player.early_reset
            player.ignored_terminations.clear()
            print(f"Early reset: {'on' if player.early_reset else 'off'}", flush=True)
          elif key == glfw.KEY_G:
            show_reference = not show_reference
          elif key == glfw.KEY_B:
            side_by_side = not side_by_side
          elif key == glfw.KEY_F:
            player.load_clip(int(player.rng.integers(len(player.paths))))
            backlog = 0.0
          elif key in (glfw.KEY_N, glfw.KEY_P):
            player.load_clip(player.index + (1 if key == glfw.KEY_N else -1))
            backlog = 0.0
          elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            speed = min(speed * 2, 8.0)
            print(f"Speed: {speed}x", flush=True)
          elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            speed = max(speed / 2, 0.125)
            print(f"Speed: {speed}x", flush=True)
        while not paused and backlog >= player.period:
          player.step()
          backlog -= player.period
        viewer.user_scn.ngeom = 0
        if show_reference:
          player.draw_reference(viewer, visualizer, side_by_side)
      phase = {RECOVERY: "Recovery", TRANSITION: "Transition", TRACKING: "Tracking"}[
        int(player.state[1])
      ]
      viewer.set_texts(
        (
          mj.mjtFont.mjFONT_NORMAL,
          mj.mjtGridPos.mjGRID_TOPLEFT,
          f"{phase} | Blue: Actor tracking ({player.tracking_name}, frame {int(player.tracking_time)})\n"
          f"Orange: recovery reward reference | clip {player.index + 1}/{len(player.paths)}",
          "",
        )
      )
      viewer.sync()
      time.sleep(0.002)


if __name__ == "__main__":
  main()
