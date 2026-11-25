"""Official SM with one tracking-command actor and privileged recovery targets."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.learners.policy_learner import ActorCriticPolicyLearner
from supermimic.models import build_mlp
from supermimic.replay_buffer import FieldSpec
from supermimic.runners.reset_sampling import ReturnSampler
from supermimic.runners.supermimic_runner import (
  SuperMimicRunner,
  SuperMimicRunnerCfg,
  build_actor,
)
from supermimic.sim_context import joint_order_ctrl_ids, nominal_pd_gains
from supermimic.tasks.g1_regularization import (
  JOINT_POS_WEIGHT,
  JOINT_TORQUE_WEIGHT,
  JOINT_VEL_WEIGHT,
  joint_pos_cost,
  joint_torque_cost,
  joint_vel_cost,
  requested_torque,
  torque_reference,
)
from supermimic.tasks.stable_official import (
  RECOVERY,
  STATE_FIELDS,
  TRACKING,
  TRANSITION,
  bounds,
  translated_bodies,
)
from supermimic.utils.reference_cache import cache_reference_bodies
from supermimic.utils.tracking_reward import tracking_reward


def add_xy(aux, xy, sign=1):
  return torch.cat((aux[:, :2] + sign * xy, aux[:, 2:]), dim=-1)


class TrackingReturnSampler(ReturnSampler):
  """Original return calculation, excluding entire recovery-origin episodes."""

  def record_tracking(self, frames, rewards, terminated, clip_end, timed_out, active):
    if not hasattr(self, "valid"):
      self.valid = torch.zeros_like(self._terminated)
    self.valid[:, self._step] = active
    return super().record(frames, rewards, terminated, clip_end, timed_out)

  def _update(self):
    returns = self._returns()
    dones = self._dones()
    num_envs, horizon = dones.shape
    episode = dones.long().cumsum(1) - dones.long()
    envs = torch.arange(num_envs, device=dones.device)[:, None]
    key = (envs * horizon + episode) * self.bin_count + self._bins(self._frames)
    keys, inverse = torch.unique(key.reshape(-1), return_inverse=True)
    valid = self.valid.reshape(-1).float()
    visits = torch.zeros(keys.numel(), device=returns.device).scatter_add_(
      0, inverse, valid
    )
    samples = torch.zeros_like(visits).scatter_add_(
      0, inverse, returns.reshape(-1) * valid
    ) / visits.clamp_min(1)
    bins = keys % self.bin_count
    new = torch.zeros_like(self.count).scatter_add_(0, bins, (visits > 0).float())
    total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
    self.count = self.count.clamp(max=self.cfg.memory) + new
    self.value = torch.where(
      new > 0,
      self.value + (total - new * self.value) / self.count.clamp_min(1),
      self.value,
    )


class StableOfficialWorldEnv(WorldModelEnv):
  _use_heft_regularization = False

  def bind_recovery(self, command):
    self.objective = command.objective
    # Both unscaled imitation kernels have a maximum of 5.
    cfg = command.cfg
    max_scale = max(1.0, cfg.recovery_multiplier, cfg.transition_coefficient)
    self.max_step_reward = 5.0 * max_scale * self.representation.dt + cfg.success_bonus

  def bind_heft_regularization(self, env, sim_context):
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if tuple(action.target_names) != robot.joint_names or action.cfg.clip is not None:
      raise ValueError("Expected unclipped position commands in G1 joint order")
    self._heft_kp, self._heft_kd = nominal_pd_gains(env, joint_order_ctrl_ids(robot))
    self._heft_torque_reference = torque_reference(robot)
    self._heft_sim_context = sim_context
    self._heft_pd_columns = sim_context.find("pd_gains")
    self._heft_encoder_columns = sim_context.find("encoder_bias")
    self._heft_substeps = env.cfg.decimation
    self._use_heft_regularization = True

  def _heft_reward_rate(self, joints, prev_joints, action, sim_ctx):
    pos, vel = joints.chunk(2, dim=-1)
    _, prev_vel = prev_joints.chunk(2, dim=-1)
    # Approximate the last two substeps using the model's constant-acceleration step.
    mean_vel = vel + (prev_vel - vel) / (2.0 * self._heft_substeps)
    kp, kd = self._heft_kp, self._heft_kd
    if self._heft_pd_columns is not None:
      scales = self._heft_sim_context.denormalize(sim_ctx, self._heft_pd_columns)
      kp_scale, kd_scale = scales.chunk(2, dim=-1)
      kp, kd = kp * kp_scale, kd * kd_scale
    target = self.dynamics.proc(action)
    if self._heft_encoder_columns is not None:
      target = target - self._heft_sim_context.denormalize(
        sim_ctx, self._heft_encoder_columns
      )
    request = requested_torque(pos, vel, target, kp, kd)
    return (
      JOINT_VEL_WEIGHT * joint_vel_cost(mean_vel)
      + JOINT_POS_WEIGHT
      * joint_pos_cost(pos, self.representation._soft_joint_pos_limits)
      + JOINT_TORQUE_WEIGHT * joint_torque_cost(request, self._heft_torque_reference)
    )

  def sample_initial_states(self):
    return super().sample_initial_states()

  def _on_reset(self, batch):
    self.phase_state = tuple(batch[k].detach() for k in STATE_FIELDS)
    super()._on_reset(batch)

  def _update_goal(self):
    xy = self.phase_state[4]
    self._ref_body_state = translated_bodies(self.motion, self.motion_times, xy)
    self._wm_goal_obs, self._actor_goal_obs = self.representation.compute_wm_goal(
      self.motion, self.motion_times, add_xy(self.aux_state, xy, -1)
    )

  def _compute_commands(self):
    _, last = bounds(self.motion, self.motion_times)
    self.motion_times = (self.motion_times + 1).minimum(last)
    self._update_goal()

  def _critic_obs(self):
    base = super()._critic_obs()
    frame, phase, bad, transition, _, _, _, _ = self.phase_state
    _, last = bounds(self.objective.motion, frame)
    frame = (frame + 1).minimum(last)
    motion = self.objective.motion
    hidden = torch.cat(
      (
        motion.root_pos_w[frame],
        motion.root_quat_w[frame],
        motion.joint_pos[frame],
        motion.root_lin_vel_w[frame],
        motion.root_ang_vel_w[frame],
        motion.joint_vel[frame],
      ),
      dim=-1,
    )
    hidden = hidden.clone()
    hidden[:, :2] = 0
    hidden = torch.where((phase == RECOVERY)[:, None], hidden, 0.0)
    flags = torch.stack(
      (
        (phase == TRACKING).float(),
        (phase == RECOVERY).float(),
        (phase == TRANSITION).float(),
        bad / self.objective.failure_steps,
        transition / self.objective.transition_frames,
      ),
      dim=-1,
    )
    return torch.cat((base, hidden, flags), dim=-1)

  def get_observations(self):
    return TensorDict(
      {
        "history": self._sensor_history(),
        "goal_motion": self._actor_goal_obs,
        "critic": self._critic_obs(),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions):
    next_aux, next_state, joints, term, reward = self._wm_step(
      self.rep_state, self.aux_state, actions, self.last_actions, self.sim_ctx
    )
    self.rep_state, self.aux_state, self.joint_state, self.last_actions = (
      next_state,
      next_aux,
      joints,
      actions,
    )
    self._compute_commands()
    if self.history_length > 1:
      self._past_sensors = torch.cat(
        (self._past_sensors[:, 1:], self._latest_sensor[:, None]), dim=1
      )
    self._latest_sensor = self._synth_sensor()
    return self.get_observations(), reward, term, {}

  def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    aux, state = self.dynamics(rep_state, action, aux_state, sim_ctx)
    joints = self.representation.joint_state(state)
    bodies = self.representation.world_frame_bodies(aux, state)
    phase = self.phase_state[1]
    (
      rec_rates,
      bonus,
      failed,
      scale,
      next_phase,
      success,
      _,
      stand_timeout,
      current,
      terminal,
      bonus_ready,
    ) = self.objective.advance(
      bodies,
      aux,
      self.motion,
      self.motion_times,
      self.phase_state,
      return_similarity=True,
    )
    tracking = tracking_reward(
      aux,
      *bodies,
      joints,
      action,
      last_action,
      *self._ref_body_state,
      self.representation._soft_joint_pos_limits,
      self.representation.dt,
    )
    q, _ = joints.chunk(2, dim=-1)
    lo, hi = self.representation._soft_joint_pos_limits.unbind(-1)
    action_penalty = 0.1 * (action - last_action).square().sum(-1)
    joint_limit_penalty = 2 * ((lo - q).clamp_min(0) + (q - hi).clamp_min(0)).sum(-1)
    original_penalty = action_penalty + joint_limit_penalty
    regularization = -original_penalty
    if self._use_heft_regularization:
      prev_joints = self.representation.joint_state(rep_state)
      regularization = (
        self._heft_reward_rate(joints, prev_joints, action, sim_ctx) - action_penalty
      )
    # Scale only imitation, preserving the official regularizers in every phase.
    reward = (
      scale * (tracking + self.representation.dt * original_penalty)
      + self.representation.dt * regularization
    )
    reward += self.representation.dt * (rec_rates.sum(-1) + bonus)
    recovery_term = (failed | stand_timeout).float()
    objective = self.objective
    if objective.cfg.recovery_timing_gate == "phase":
      in_recovery = phase == RECOVERY
      eligible = (
        in_recovery & ~success & (self.phase_state[2] >= objective.failure_steps - 1)
      )
      failure_probability = torch.sigmoid(
        (objective.cfg.failure_similarity - current) / 0.01
      )
      recovery_term = torch.where(
        stand_timeout, 1.0, torch.where(eligible, failure_probability, 0.0)
      )
      success_probability = torch.sigmoid(
        (terminal - objective.cfg.terminal_similarity) / 0.03
      )
      reward += objective.cfg.success_bonus * torch.where(
        in_recovery & bonus_ready,
        success_probability - success_probability.detach(),
        0.0,
      )
    nominal_term = super()._compute_term_prob(aux, *self._ref_body_state[:2])
    # A clip cutoff holds the last command during imagined rollouts.
    term = torch.where(phase == TRACKING, nominal_term, recovery_term)
    self.phase_state = next_phase
    return aux, state, joints, term, reward


@dataclass
class StableOfficialRunnerCfg(SuperMimicRunnerCfg):
  actor_type: str = "mlp"
  actor_obs: str = "proprio"
  history_length: int = 10
  from_scratch: bool = False
  reference_mode: str = "tracking_command_only"
  recovery_reset_mode: str = "frame"
  seed: int = 42
  recovery_probability: float = 0.5
  seed_checkpoint: str = ""
  resume_checkpoint: str | None = None
  heft_regularization: bool = False
  pd_gain_dr: bool = False


def termination_episode_metrics(manager, dones):
  """One one-hot outcome per ended episode, ready for the shared logger."""
  names = [name for name in manager.active_terms if name != "phase_update"]
  # Prefer true failures, then a specific clip end, then generic timeout.
  names.sort(
    key=lambda name: (
      manager.get_term_cfg(name).time_out,
      name
      != ("motion_end" if manager.get_term_cfg(name).time_out else "recovery_error"),
    )
  )
  remaining = torch.ones_like(dones[dones])
  metrics = {}
  for name in names:
    hit = manager.get_term(name)[dones]
    metrics[f"Episode_Termination/{name}"] = (remaining & hit).float()
    remaining = remaining & ~hit
  return metrics


class StableOfficialRunner(SuperMimicRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Tracking and recovery-origin episodes both end at 10 s.
    self.logger.max_episode_length = round(
      self.motion_cmd.cfg.recovery_timeout_s / self.env.unwrapped.step_dt
    )

  def _training_start_step(self):
    if self.cfg.resume_checkpoint is None:
      return 0
    saved = torch.load(
      self.cfg.resume_checkpoint, map_location=self.device, weights_only=False
    )
    runtime = {
      "output_dir",
      "run_name",
      "max_iterations",
      "resume_checkpoint",
      "save_interval",
    }
    legacy_defaults = {
      "heft_regularization": False,
      "pd_gain_dr": False,
      "obs_noise_scale": 1.0,
    }
    for key, value in asdict(self.cfg).items():
      saved_value = saved["cfg"].get(key, legacy_defaults.get(key))
      if key not in runtime and saved_value != value:
        raise ValueError(f"Resume config differs for {key}")
    self.policy_learner.actor.load_state_dict(saved["actor"], strict=True)
    self.policy_learner.critic.load_state_dict(saved["critic"], strict=True)
    self._wm.load_bundle(saved["world_model"])
    for key, optimizers in (
      ("actor_optimizers", self.policy_learner.actor_optimizers),
      ("critic_optimizers", self.policy_learner.critic_optimizers),
      ("wm_optimizers", self.world_model_learner.optimizers),
    ):
      for optimizer, state in zip(optimizers, saved[key], strict=True):
        optimizer.load_state_dict(state)
    from pathlib import Path

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_dir = Path(self.cfg.resume_checkpoint).parent / "tensorboard"
    if event_dir.is_dir():
      events = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
      events.Reload()
      last_log = saved["iteration"] // self.cfg.log_interval
      tags = events.Tags()["scalars"]
      self.logger.tot_time = sum(
        value.value
        for tag in (
          "Perf/collection_time",
          "Perf/wm_learning_time",
          "Perf/policy_learning_time",
        )
        if tag in tags
        for value in events.Scalars(tag)
        if value.step <= last_log
      )
    start = saved["iteration"] + 1
    if not 0 < start < self.cfg.max_iterations:
      raise ValueError(f"Resume iteration {start} exceeds target")
    print(
      f"[INFO] Resumed full actor/critic/WM and optimizer states at iteration {start}; replay and reset-sampler statistics rebuild",
      flush=True,
    )
    return start

  def _buffer_schema(self):
    schema = super()._buffer_schema()
    for name in STATE_FIELDS:
      schema[name] = (
        FieldSpec(2)
        if name == "command_xy"
        else FieldSpec(
          dtype=torch.bool
          if name in ("gate_tracking_override", "recovery_episode")
          else torch.long
        )
      )
    return schema

  def _init_policy(self):
    cfg = self.cfg
    self.reference_caches = [
      cache_reference_bodies(self.motion),
      cache_reference_bodies(self.motion_cmd.recovery),
    ]
    self.motion_cmd._body_state = self.motion.body_state
    seed = (
      None
      if cfg.from_scratch
      else torch.load(cfg.seed_checkpoint, map_location=self.device, weights_only=False)
    )
    torch.manual_seed(cfg.seed)
    tracking = build_actor(cfg, self._wm, self.device)
    if seed is not None:
      tracking.load_state_dict(seed["actor"], strict=True)
      self._wm.load_bundle(seed["world_model"])
    self._active_task_specs = [
      ts for ts in tracking.task_specs if cfg.slot_mix.get(ts.name, 0) > 0
    ]
    actor = tracking
    _, base_dim = self._policy_input_dims()
    privileged_dim = 13 + 2 * self._num_actions + 5
    critic = build_mlp(
      base_dim + privileged_dim, cfg.critic_hidden_dims, output_dim=1
    ).to(self.device)
    if seed is not None:
      weights = deepcopy(seed["critic"])
      weights["0.weight"] = torch.cat(
        (
          weights["0.weight"],
          torch.zeros(weights["0.weight"].shape[0], privileged_dim, device=self.device),
        ),
        dim=-1,
      )
      critic.load_state_dict(weights, strict=True)
    self.wm_env = StableOfficialWorldEnv(
      self._dynamics,
      self._representation,
      self.env,
      cfg.policy.batch_size,
      self.replay_buffer,
      cfg.history_length,
      cfg.termination_softness,
      self.device,
    )
    self.wm_env.bind_recovery(self.motion_cmd)
    if cfg.heft_regularization:
      self.wm_env.bind_heft_regularization(self.env.unwrapped, self._sim_context)
    self.policy_learner = ActorCriticPolicyLearner(
      actor, critic, self.wm_env, cfg.policy, self.device
    )
    self.wm_env.step = torch.compile(self.wm_env.step)
    self.wm_env.reset = torch.compile(self.wm_env.reset)
    self._configure_reset_sampling()
    origin = (
      "from scratch (no model weights loaded)"
      if seed is None
      else f"from step {seed['iteration']}"
    )
    print(
      f"[INFO] Initialized single reference-conditioned MLP, critic and WM {origin}",
      flush=True,
    )

  def _configure_reset_sampling(self):
    super()._configure_reset_sampling()
    self._sampler = TrackingReturnSampler(
      self.cfg.return_sampler,
      self.motion_cmd.bin_count,
      self.motion.time_step_total,
      self.env.num_envs,
      5 * self.env.unwrapped.step_dt,
      self.device,
    )

  def _build_sim_obs_td(
    self, rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
  ):
    return super()._build_sim_obs_td(
      rep_state,
      add_xy(aux_state, self.motion_cmd.command_xy, -1),
      sim_actor_obs,
      motion_times,
      sim_ctx,
    )

  def _step_and_store(self, actions, curr_states, actor_td=None, extra_fields=None):
    fields = {
      k: v.clone()
      for k, v in zip(STATE_FIELDS, self.motion_cmd.phase_state(), strict=True)
    }
    return super()._step_and_store(
      actions, curr_states, actor_td, dict(extra_fields or {}) | fields
    )

  def step(self):
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr = obs["full_state"].to(self.device)
      state = self._representation.represent(curr)
      aux = self._representation.extract_aux_state(curr)
      ctx = self._sim_context.read(self.env.unwrapped)
      times = self.motion_cmd.time_steps.clone()
      active = ~self.motion_cmd.recovery_episode.clone()
      actor_td = self._build_sim_obs_td(state, aux, obs["actor"], times, ctx)
      actions = self.policy_learner.actor(actor_td)
      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr,
        actor_td,
        {"sim_ctx": ctx}
        | ({"actor_obs": obs["actor"]} if self._stores_actor_obs else {}),
      )
      next_sensor = next_obs["actor"]
      self._advance_sim_history(next_sensor, dones)
      manager = self.env.unwrapped.termination_manager
      log_key = "episode" if "episode" in extras else "log"
      logged = {
        key: value
        for key, value in extras.get(log_key, {}).items()
        if not key.startswith("Episode_Termination/")
      }
      logged.update(termination_episode_metrics(manager, dones.bool()))
      self.logger.process_env_step(rewards, dones.bool(), {log_key: logged})
      truncated = manager.time_outs.bool()
      clip_end = manager.get_term("motion_end").bool()
      weights = self._sampler.record_tracking(
        times,
        rewards,
        manager.terminated.bool() & ~truncated,
        clip_end,
        truncated & ~clip_end,
        active,
      )
      if weights is not None:
        self.motion_cmd.set_sampling_distribution(weights)
    self._flush_normalization()
    return self.env.num_envs
