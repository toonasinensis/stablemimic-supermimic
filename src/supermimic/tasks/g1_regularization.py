"""HEFT student-inspired regularizers for the G1 recovery experiment."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import ManagerTermBase, RewardTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from supermimic.sim_context import joint_order_ctrl_ids, read_pd_gains

JOINT_VEL_WEIGHT = -0.0005
JOINT_POS_WEIGHT = -1.0
JOINT_TORQUE_WEIGHT = -0.01

# HEFT g1_5 nominal torque limits, used as the reward reference in N m.
_TORQUE_REFERENCE = {
  ".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint": 25.0,
  ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 88.0,
  ".*_(hip_roll|knee)_joint": 139.0,
  ".*_wrist_(pitch|yaw)_joint": 5.0,
  "waist_(pitch|roll)_joint|.*_ankle_(pitch|roll)_joint": 35.0,
}


def torque_reference(robot) -> torch.Tensor:
  indices, _, values = resolve_matching_names_values(
    _TORQUE_REFERENCE, robot.joint_names
  )
  if len(indices) != 29 or robot.num_joints != 29:
    raise ValueError("Expected 29 HEFT g1_5 torque references")
  reference = torch.empty_like(robot.data.default_joint_pos[0])
  reference[indices] = reference.new_tensor(values)
  return reference


def smooth_excess(excess: torch.Tensor, width: float) -> torch.Tensor:
  ramp = excess.clamp(0.0, width)
  return torch.where(
    excess < width, ramp.square() / (2.0 * width), excess - width / 2.0
  )


def joint_vel_cost(mean_vel: torch.Tensor) -> torch.Tensor:
  return mean_vel.square().sum(dim=-1)


def joint_pos_cost(pos: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
  lo, hi = limits.unbind(-1)
  return (smooth_excess(lo - pos, 0.01) + smooth_excess(pos - hi, 0.01)).sum(-1) / 0.1


def requested_torque(pos, vel, target, kp, kd) -> torch.Tensor:
  return kp * (target - pos) - kd * vel


def joint_torque_cost(request: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
  return smooth_excess(request.abs() / (0.75 * reference) - 1.0, 0.05).sum(-1)


class G1RegularizedPositionAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.previous_substep_velocity = torch.zeros_like(self._entity.data.joint_vel)

  def apply_actions(self) -> None:
    # The velocity immediately before the last substep is the penultimate sample.
    self.previous_substep_velocity.copy_(self._entity.data.joint_vel)
    super().apply_actions()


@dataclass(kw_only=True)
class G1RegularizedPositionActionCfg(JointPositionActionCfg):
  def build(self, env):
    return G1RegularizedPositionAction(self, env)


def sim_joint_vel_cost(env):
  action = env.action_manager.get_term("joint_pos")
  vel = env.scene["robot"].data.joint_vel
  return joint_vel_cost(0.5 * (action.previous_substep_velocity + vel))


def sim_joint_pos_cost(env):
  data = env.scene["robot"].data
  return joint_pos_cost(data.joint_pos, data.soft_joint_pos_limits)


class SimJointTorqueCost(ManagerTermBase):
  def __init__(self, cfg, env):
    super().__init__(env)
    self.robot = env.scene["robot"]
    self.ctrl_ids = joint_order_ctrl_ids(self.robot)
    self.reference = torque_reference(self.robot)

  def __call__(self, env):
    data = self.robot.data
    kp, kd = read_pd_gains(env, self.ctrl_ids)
    request = requested_torque(
      data.joint_pos, data.joint_vel, data.joint_pos_target, kp, kd
    )
    return joint_torque_cost(request, self.reference)


def configure_g1_regularization(cfg) -> None:
  cfg.actions["joint_pos"] = G1RegularizedPositionActionCfg(
    **vars(cfg.actions["joint_pos"])
  )
  cfg.rewards.update(
    joint_limit=RewardTermCfg(func=sim_joint_pos_cost, weight=JOINT_POS_WEIGHT),
    joint_vel_l2=RewardTermCfg(func=sim_joint_vel_cost, weight=JOINT_VEL_WEIGHT),
    joint_torque_limits=RewardTermCfg(
      func=SimJointTorqueCost, weight=JOINT_TORQUE_WEIGHT
    ),
  )
