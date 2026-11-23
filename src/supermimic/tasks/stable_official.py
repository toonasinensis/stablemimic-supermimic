"""Recovery episodes and transitions around the unchanged official SM tracker."""

from dataclasses import dataclass, fields, replace

import torch
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
)
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils.helpers import rotation_error_magnitude

TRACKING, RECOVERY, TRANSITION = range(3)
STATE_FIELDS = (
  "getup_frame",
  "phase",
  "bad_steps",
  "transition_steps",
  "command_xy",
  "gate_tracking_override",
  "recovery_episode",
  "reference_stand_steps",
)
RECOVERY_KEYS = (
  "root_height",
  "root_orientation",
  "body_position",
  "body_orientation",
  "body_linear_velocity",
  "body_angular_velocity",
)


def bounds(motion, frame):
  clip = torch.searchsorted(motion.motion_ends, frame, right=True)
  starts = torch.cat((motion.motion_ends.new_zeros(1), motion.motion_ends[:-1]))
  return starts[clip], motion.motion_ends[clip] - 1


def translated_bodies(motion, frame, xy):
  pos, quat, lin, ang = motion.body_state(frame)
  shift = torch.cat((xy, torch.zeros_like(xy[:, :1])), dim=-1)
  return pos + shift[:, None], quat, lin, ang


def recovery_heading_delta(rotation, reference_rotation):
  relative = rotation @ reference_rotation.transpose(-1, -2)
  sine = relative[:, 1, 0] - relative[:, 0, 1]
  cosine = relative[:, 0, 0] + relative[:, 1, 1]
  valid = sine.square() + cosine.square() > 1e-12
  yaw = torch.atan2(torch.where(valid, sine, 0.0), torch.where(valid, cosine, 1.0))
  return yaw


def align_recovery_reference(bodies, reference):
  """Rotate the reference about world Z to match the robot's root heading.

  Fit yaw only: root height, tilt and articulated pose remain recovery targets.
  At an upside-down yaw ambiguity choose zero; keep atan2 gradients finite.
  """
  _, rot, _, _ = bodies
  target, quat, target_lin, target_ang = reference
  target_rot = matrix_from_quat(quat)
  yaw = recovery_heading_delta(rot[:, 0], target_rot[:, 0])
  c, s = yaw.cos(), yaw.sin()
  zero, one = torch.zeros_like(c), torch.ones_like(c)
  heading = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1)
  heading = heading.reshape(-1, 1, 3, 3)
  aligned_pos = target[:, :1] + (
    heading @ (target - target[:, :1]).unsqueeze(-1)
  ).squeeze(-1)
  return (
    aligned_pos,
    heading @ target_rot,
    (heading @ target_lin.unsqueeze(-1)).squeeze(-1),
    (heading @ target_ang.unsqueeze(-1)).squeeze(-1),
  )


def recovery_scores(bodies, reference):
  pos, rot, lin, ang = bodies
  target, target_rot, target_lin, target_ang = align_recovery_reference(
    bodies, reference
  )
  delta = (pos - pos[:, :1]) - (target - target[:, :1])
  orientation = rotation_error_magnitude(rot, target_rot).square()
  return torch.stack(
    (
      0.5 * torch.exp(-(pos[:, 0, 2] - target[:, 0, 2]).square() / 0.3**2),
      0.5 * torch.exp(-orientation[:, 0] / 0.4**2),
      torch.exp(-delta.square().sum(-1).mean(-1) / 0.3**2),
      torch.exp(-orientation.mean(-1) / 0.4**2),
      torch.exp(-(lin - target_lin).square().sum(-1).mean(-1)),
      torch.exp(-(ang - target_ang).square().sum(-1).mean(-1) / 3.14**2),
    ),
    dim=-1,
  )


def recovery_similarity_terms(scores):
  # The pose kernels have maximum weights 0.5, 0.5, 1 and 1.
  return scores[..., :4] / 3.0


@dataclass(kw_only=True)
class StableOfficialMotionCfg(MotionCommandCfg):
  recovery_file: str = ""
  torso_body_name: str = "torso_link"
  recovery_reset_mode: str = "frame"
  recovery_joint_position_noise: float | None = None
  recovery_probability: float = 0.5
  recovery_multiplier: float = 1.0
  recovery_timeout_s: float = 10.0
  recovery_timing_gate: str = "off"
  recovery_standing_tail_s: float = 0.0
  recovery_stand_timeout_s: float = 2.0
  recovery_lift: float = 0.03
  failure_similarity: float = 0.05
  terminal_similarity: float = 0.70
  error_timeout_s: float = 2.0
  transition_s: float = 1.5
  transition_coefficient: float = 1.0
  success_bonus: float = 1.0
  gate_ce_weight: float = 0.1
  gate_transition_weight: float = 4.0
  gate_consistency_weight: float = 0.01
  expert_alignment_weight: float = 0.01

  def build(self, env):
    return StableOfficialMotion(self, env)


class RecoveryObjective:
  def __init__(self, cfg, motion, dt):
    self.cfg, self.motion, self.dt = cfg, motion, dt
    self.failure_steps = round(cfg.error_timeout_s / dt)
    self.standing_tail_frames = round(cfg.recovery_standing_tail_s / dt)
    self.stand_timeout_steps = round(cfg.recovery_stand_timeout_s / dt)
    if self.standing_tail_frames:
      lengths = torch.diff(motion.motion_ends, prepend=motion.motion_ends.new_zeros(1))
      if self.stand_timeout_steps < 1 or (lengths <= self.standing_tail_frames).any():
        raise ValueError(
          "Invalid recovery standing tail or timeout for the motion clips"
        )
    self.transition_frames = round(cfg.transition_s / dt)
    self.timing_gate_tail_frames = round(3.0 / dt)
    self.timing_gate_cap = round((cfg.recovery_timeout_s - cfg.transition_s - 0.5) / dt)

  def weights(self, phase, transition, override):
    alpha = (transition / self.transition_frames).clamp(0, 1)
    tracking = torch.where(phase == TRANSITION, alpha, (phase == TRACKING).float())
    tracking = torch.where((phase == RECOVERY) & override, 1.0, tracking)
    return torch.stack((tracking, 1 - tracking), dim=-1)

  def advance(
    self, bodies, aux, nominal, motion_times, state, *, return_similarity=False
  ):
    frame, phase, bad, transition, xy, override, episode, stand_steps = state
    first, last = bounds(self.motion, frame)
    successor = (frame + 1).minimum(last)
    scores = recovery_scores(bodies, self.motion.body_state(successor))
    terminal_scores = recovery_scores(bodies, self.motion.body_state(last))
    terminal = recovery_similarity_terms(terminal_scores).sum(-1)
    current = recovery_similarity_terms(scores).sum(-1)
    rec = phase == RECOVERY
    bonus_ready = torch.ones_like(rec)
    if self.cfg.recovery_timing_gate == "phase":
      gate_frame = (
        (last + 1 - self.timing_gate_tail_frames)
        .minimum(first + self.timing_gate_cap)
        .maximum(first)
      )
      bonus_ready = successor >= gate_frame
    success = rec & (terminal >= self.cfg.terminal_similarity)
    success = success & bonus_ready
    bad_next = torch.where(
      rec & ~success & (current < self.cfg.failure_similarity), bad + 1, 0
    )
    failed = rec & (bad_next >= self.failure_steps)
    # A step is spent in the tail once its source reference is in the hold.
    in_standing_tail = frame >= last - self.standing_tail_frames + 1
    stand_steps_next = torch.where(
      rec & ~success & in_standing_tail, stand_steps + 1, 0
    )
    stand_timed_out = rec & ~success & (stand_steps_next >= self.stand_timeout_steps)
    # Exhausting a demonstrator is not a reset: hold its terminal while recovery finishes.
    transition_next = torch.where(phase == TRANSITION, transition + 1, 0)
    completed = (phase == TRANSITION) & (transition_next >= self.transition_frames)
    next_phase = torch.where(
      completed, TRACKING, torch.where(success, TRANSITION, phase)
    )
    next_frame = torch.where(rec, successor, frame)
    aligned = aux[:, :2] - nominal.root_pos_w[motion_times, :2]
    next_xy = torch.where(success[:, None], aligned, xy).detach()
    alpha = (transition / self.transition_frames).clamp(0, 1)
    track_scale = torch.where(
      phase == TRANSITION,
      alpha * self.cfg.transition_coefficient,
      (phase == TRACKING).float(),
    )
    # Fade out imitation of the get-up endpoint as command imitation fades in.
    in_transition = phase == TRANSITION
    recovery_scale = torch.where(in_transition, 1.0 - alpha, rec.float())
    recovery_target_scores = torch.where(
      in_transition[:, None], terminal_scores, scores
    )
    rates = (
      recovery_target_scores * (recovery_scale * self.cfg.recovery_multiplier)[:, None]
    )
    bonus = success.float() * self.cfg.success_bonus / self.dt
    result = (
      rates,
      bonus,
      failed,
      track_scale,
      (
        next_frame,
        next_phase,
        bad_next,
        transition_next,
        next_xy,
        override,
        episode,
        stand_steps_next,
      ),
      success,
      completed,
      stand_timed_out,
    )
    if return_similarity:
      return (*result, current, terminal, bonus_ready)
    return result


class StableOfficialMotion(MotionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.recovery = MotionLoader(cfg.recovery_file, self.motion.fk, self.device)
    self.objective = RecoveryObjective(cfg, self.recovery, env.step_dt)
    self.getup_frame = self.time_steps.clone()
    self.phase = torch.zeros_like(self.time_steps)
    self.bad_steps = self.phase.clone()
    self.reference_stand_steps = self.phase.clone()
    self.transition_steps = self.phase.clone()
    self.command_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.gate_tracking_override = torch.zeros_like(self.phase, dtype=torch.bool)
    self.recovery_episode = self.gate_tracking_override.clone()
    self.recovery_reached_transition = self.gate_tracking_override.clone()
    self.standing_steps = self.phase.clone()
    self.stood_for_1s = self.gate_tracking_override.clone()
    self.torso_index = cfg.body_names.index(cfg.torso_body_name)
    self.step_phase = self.phase.clone()
    self.track_scale = torch.ones(self.num_envs, device=self.device)
    self.recovery_rates = torch.zeros(self.num_envs, 6, device=self.device)
    self.success_rate = torch.zeros(self.num_envs, device=self.device)
    self.failed = torch.zeros_like(self.phase, dtype=torch.bool)
    self.stand_timed_out = torch.zeros_like(self.failed)
    self.failure_weights = torch.ones(self.recovery.time_step_total, device=self.device)
    self.initial_frame = self.phase.clone()
    self.metrics["recovery_reset_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def phase_state(self):
    return tuple(getattr(self, key) for key in STATE_FIELDS)

  def set_time_steps(self, env_ids, frames):
    super().set_time_steps(env_ids, frames)
    if hasattr(self, "command_xy"):
      self._body_pos_w[env_ids, :, :2] += self.command_xy[env_ids, None, :]

  def _record_recovery_episode_outcomes(self, env_ids):
    # Log one Bernoulli outcome per finished recovery-origin episode. The shared
    # logger concatenates these vectors, giving an episode-weighted success rate.
    ended_recovery = self.recovery_episode[env_ids]
    outcomes = self.recovery_reached_transition[env_ids][ended_recovery].float()
    self._env.extras.setdefault("log", {})[
      "Recovery/entered_transition_episode_fraction"
    ] = outcomes.clone()
    self._env.extras["log"]["Recovery/stood_for_1s_episode_fraction"] = (
      self.stood_for_1s[env_ids][ended_recovery].float().clone()
    )
    self.recovery_reached_transition[env_ids] = False
    self.standing_steps[env_ids] = 0
    self.stood_for_1s[env_ids] = False

  def _resample_command(self, env_ids):
    self._record_recovery_episode_outcomes(env_ids)
    # Keep native tracking RSI distribution, perturbations, joint limits and motor reset.
    self.command_xy[env_ids] = 0
    super()._resample_command(env_ids)
    count = len(env_ids)
    rec = torch.rand(count, device=self.device) < self.cfg.recovery_probability
    if self.cfg.recovery_reset_mode == "clip_start":
      ends = self.recovery.motion_ends
      starts = torch.cat((ends.new_zeros(1), ends[:-1]))
      frame = starts[torch.randint(len(starts), (count,), device=self.device)]
    else:
      sampled = torch.randint(
        self.recovery.time_step_total, (count,), device=self.device
      )
      hard = torch.multinomial(self.failure_weights, count, replacement=True)
      frame = torch.where(torch.rand(count, device=self.device) < 0.5, sampled, hard)
    self.phase[env_ids] = rec.long()
    self.step_phase[env_ids] = self.phase[env_ids]
    self.recovery_episode[env_ids] = rec
    self.getup_frame[env_ids] = frame
    self.initial_frame[env_ids] = frame
    self.bad_steps[env_ids] = 0
    self.reference_stand_steps[env_ids] = 0
    self.transition_steps[env_ids] = 0
    ids = env_ids[rec]
    frame = frame[rec]
    root = self.recovery.root_pos_w[frame].clone()
    root[:, 2] += self.cfg.recovery_lift
    n = len(ids)

    def noise(ranges):
      return torch.stack(
        [
          torch.empty(n, device=self.device).uniform_(*ranges.get(k, (0.0, 0.0)))
          for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        dim=-1,
      )

    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    pose = noise(self.cfg.pose_range)
    vel = noise(self.cfg.velocity_range)
    root += pose[:, :3]
    quat = quat_mul(
      quat_from_euler_xyz(*pose[:, 3:].unbind(-1)), self.recovery.root_quat_w[frame]
    )
    q = self.recovery.joint_pos[frame]
    noise_amplitude = self.cfg.recovery_joint_position_noise
    if noise_amplitude is None:
      q = q + torch.empty_like(q).uniform_(*self.cfg.joint_position_range)
    else:
      # Sample within the feasible interval to avoid piling up resets at a limit.
      lower, upper = self.robot.data.soft_joint_pos_limits[ids].unbind(-1)
      left = (q - noise_amplitude).clamp(lower, upper)
      right = (q + noise_amplitude).clamp(lower, upper)
      q = torch.lerp(left, right, torch.rand_like(q))
    self._write_reference_state_to_sim(
      ids,
      root + self._env.scene.env_origins[ids],
      quat,
      self.recovery.root_lin_vel_w[frame] + vel[:, :3],
      self.recovery.root_ang_vel_w[frame] + vel[:, 3:],
      q,
      self.recovery.joint_vel[frame],
    )
    nominal_height = self.motion.root_pos_w[self.time_steps[env_ids], 2]
    reset_height = torch.where(
      rec,
      self.recovery.root_pos_w[self.getup_frame[env_ids], 2] + self.cfg.recovery_lift,
      nominal_height,
    )
    self.gate_tracking_override[env_ids] = rec & (reset_height >= 0.8 * nominal_height)
    self.metrics["recovery_reset_fraction"][env_ids] = rec.float()
    self._pending_forward = True

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    _, last = bounds(self.motion, self.time_steps[ids])
    self.set_time_steps(ids, (self.time_steps[ids] + 1).minimum(last))
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()


def advance_phase(env):
  cmd = env.command_manager.get_term("motion")
  data = cmd.robot.data
  aux = torch.cat(
    (data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w), dim=-1
  )
  bodies = (
    cmd.robot_body_pos_w - env.scene.env_origins[:, None],
    matrix_from_quat(cmd.robot_body_quat_w),
    cmd.robot_body_lin_vel_w,
    cmd.robot_body_ang_vel_w,
  )
  # Observation-only diagnostics: these never alter rewards, phases or dones.
  base_height = aux[:, 2]
  torso_up = bodies[1][:, cmd.torso_index, 2, 2]
  standing = (
    (base_height >= 0.7)
    & (torso_up >= 0.8660254)
    & (data.root_link_lin_vel_w.norm(dim=-1) <= 0.3)
  )
  cmd.standing_steps.copy_(torch.where(standing, cmd.standing_steps + 1, 0))
  cmd.stood_for_1s.logical_or_(cmd.standing_steps >= round(1.0 / env.step_dt))
  cmd.step_phase.copy_(cmd.phase)
  rates, bonus, failed, scale, state, success, completed, stand_timeout = (
    cmd.objective.advance(bodies, aux, cmd.motion, cmd.time_steps, cmd.phase_state())
  )
  cmd.recovery_rates.copy_(rates)
  cmd.success_rate.copy_(bonus)
  cmd.recovery_reached_transition.logical_or_(success)
  cmd.failed.copy_(failed)
  cmd.stand_timed_out.copy_(stand_timeout)
  cmd.track_scale.copy_(scale)
  cmd.failure_weights.scatter_add_(
    0, cmd.initial_frame, (failed | stand_timeout).float()
  )
  for key, value in zip(STATE_FIELDS, state, strict=True):
    getattr(cmd, key).copy_(value)
  env.extras.setdefault("log", {}).update(
    {
      "Recovery/base_height": base_height.mean(),
      "Recovery/torso_up": torso_up.mean(),
      "Recovery/standing_pose_fraction": standing.float().mean(),
      "Recovery/success_fraction": success.float().mean(),
      "Recovery/transition_completed_fraction": completed.float().mean(),
      "Phase/tracking_fraction": (cmd.step_phase == TRACKING).float().mean(),
      "Phase/recovery_fraction": (cmd.step_phase == RECOVERY).float().mean(),
      "Phase/transition_fraction": (cmd.step_phase == TRANSITION).float().mean(),
    }
  )
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def tracking_termination(env, func, **params):
  cmd = env.command_manager.get_term("motion")
  return func(env, **params) & (cmd.step_phase == TRACKING)


def time_out(env):
  cmd = env.command_manager.get_term("motion")
  limit = torch.where(
    cmd.recovery_episode, cmd.cfg.recovery_timeout_s, env.cfg.episode_length_s
  )
  return env.episode_length_buf * env.step_dt >= limit


def recovery_failed(env):
  return env.command_manager.get_term("motion").failed


def recovery_stand_timed_out(env):
  return env.command_manager.get_term("motion").stand_timed_out


def track_reward(env, func, **params):
  return func(env, **params) * env.command_manager.get_term("motion").track_scale


def recovery_reward(env, index):
  return env.command_manager.get_term("motion").recovery_rates[:, index]


def success_reward(env):
  return env.command_manager.get_term("motion").success_rate


def stable_official_cfg(tracking, recovery, robot="g1"):
  cfg = gencoord_env_cfg()
  original = cfg.commands["motion"]
  if robot == "roban":
    from supermimic.tasks.roban_motor import (
      RobanS17PositionActionCfg,
      roban_mimic_actuators,
    )
    from supermimic.tasks.roban_recovery import (
      ROBAN_TRACKING_BODIES,
      get_roban_robot_cfg,
    )

    entity = get_roban_robot_cfg()
    entity.articulation = replace(
      entity.articulation, actuators=roban_mimic_actuators()
    )
    cfg.scene.entities = {"robot": entity}
    cfg.scene.sensors = ()
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    original.anchor_body_name = "base_link"
    original.body_names = ROBAN_TRACKING_BODIES
    cfg.events["foot_friction"].params[
      "asset_cfg"
    ].geom_names = r"^[lr]_foot_padded_hull$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.viewer.body_name = "base_link"
    cfg.actions["joint_pos"].scale = {".*": 0.5}
    cfg.actions["joint_pos"] = RobanS17PositionActionCfg(
      **vars(cfg.actions["joint_pos"])
    )
  cmd = StableOfficialMotionCfg(
    **{f.name: getattr(original, f.name) for f in fields(MotionCommandCfg)}
  )
  if robot == "roban":
    cmd.torso_body_name = "waist_yaw_link"
  cmd.motion_file = tracking
  cmd.recovery_file = recovery
  cmd.sampling_mode = "external"
  cfg.commands = {"motion": cmd}
  terms = {"phase_update": TerminationTermCfg(func=advance_phase)}
  for name, term in cfg.terminations.items():
    if name == "time_out":
      term.func = time_out
    else:
      term.params = {"func": term.func, **term.params}
      term.func = tracking_termination
    terms[name] = term
  terms["recovery_error"] = TerminationTermCfg(func=recovery_failed)
  cfg.terminations = terms
  for name, term in cfg.rewards.items():
    if name.startswith("motion_"):
      term.params = {"func": term.func, **term.params}
      term.func = track_reward
  for i, name in enumerate(RECOVERY_KEYS):
    cfg.rewards["recovery_" + name] = RewardTermCfg(
      func=recovery_reward, weight=1.0, params={"index": i}
    )
  cfg.rewards["recovery_success"] = RewardTermCfg(func=success_reward, weight=1.0)
  return cfg
