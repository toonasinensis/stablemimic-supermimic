"""Roban S17 torque limits for the MuJoCo position drives."""

from dataclasses import dataclass
from math import isclose
from re import fullmatch
from xml.etree import ElementTree

import mujoco
import torch
import warp as wp
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from supermimic.utils.gencoord_helpers import decompose_rep_state

from .roban_recovery import get_roban_xml_path

# RobanS2-TPPO-Parkour, tasks/personal/thl/mimic/robot_cfg.py.
# Joint pattern, stiffness, damping, armature, rated effort.
_MIMIC_DRIVES = (
  ("waist_yaw_joint", 40.179, 2.557, 0.03178707, 80.0),
  ("leg_[lr]1_joint", 90.179, 3.557, 0.0493, 150.0),
  ("leg_[lr][24]_joint", 150.098, 8.308, 0.0493, 150.0),
  ("leg_[lr]3_joint", 40.179, 3.557, 0.03178707, 70.0),
  ("leg_[lr][56]_joint", 34.25, 1.907, 0.036, 74.0),
  ("zarm_[lr]1_joint", 14.25, 0.907, 0.015, 18.6),
  ("zarm_[lr][234]_joint", 14.25, 0.907, 0.018, 37.0),
)


def roban_mimic_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(pattern,),
      stiffness=stiffness,
      damping=damping,
      armature=armature,
      effort_limit=effort,
      frictionloss=0.0,
      delay_min_lag=0,
      delay_max_lag=0,
    )
    for pattern, stiffness, damping, armature, effort in _MIMIC_DRIVES
  )


@wp.kernel
def _update_motor_torque_bounds(
  ctrl_ids: wp.array(dtype=wp.int32),
  dof_ids: wp.array(dtype=wp.int32),
  effort_and_speed: wp.array(dtype=wp.vec2),
  qvel: wp.array2d(dtype=wp.float32),
  force_range: wp.array2d(dtype=wp.vec2),
):
  world, motor = wp.tid()
  limits = effort_and_speed[motor]
  speed = qvel[world, dof_ids[motor]]
  lower = -limits[0]
  upper = limits[0]
  if speed < -limits[1]:
    lower = 0.0
  if speed > limits[1]:
    upper = 0.0
  force_range[world, ctrl_ids[motor]] = wp.vec2(lower, upper)


@dataclass(kw_only=True)
class RobanS17PositionActionCfg(JointPositionActionCfg):
  """Use the effort and velocity limits from the URDF beside the robot XML."""

  def build(self, env: ManagerBasedRlEnv) -> JointPositionAction:
    return RobanS17PositionAction(self, env)


class RobanS17PositionAction(JointPositionAction):
  def __init__(self, cfg: RobanS17PositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._sim = env.sim
    model = self._sim.mj_model
    urdf = get_roban_xml_path().with_suffix(".urdf")
    urdf_limits = {
      joint.attrib["name"]: joint.find("limit")
      for joint in ElementTree.parse(urdf).getroot().findall("joint")
    }

    ctrl_ids = self._entity.indexing.ctrl_ids.cpu().tolist()
    dof_ids = []
    effort_and_speed = []
    pd_parameters = {}
    for ctrl_id in ctrl_ids:
      joint_id = int(model.actuator_trnid[ctrl_id, 0])
      name = model.joint(joint_id).name.rsplit("/", 1)[-1]
      limit = urdf_limits.get(name)
      if limit is None:
        raise ValueError(f"{urdf} has no motor limits for {name}")
      effort = float(limit.attrib["effort"])
      speed = float(limit.attrib["velocity"])
      lower, upper = model.actuator_forcerange[ctrl_id]
      if not (
        effort > 0.0
        and speed > 0.0
        and model.actuator_trntype[ctrl_id] == mujoco.mjtTrn.mjTRN_JOINT
        and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        and model.actuator_gear[ctrl_id, 0] == 1.0
        and model.actuator_forcelimited[ctrl_id]
        and isclose(lower, -effort)
        and isclose(upper, effort)
      ):
        raise ValueError(f"Expected a 1:1 hinge drive with S17 URDF limits for {name}")
      dof = int(model.jnt_dofadr[joint_id])
      expected = next(
        (kp, kd, arm, rated)
        for pattern, kp, kd, arm, rated in _MIMIC_DRIVES
        if fullmatch(pattern, name)
      )
      actual = (
        model.actuator_gainprm[ctrl_id, 0],
        -model.actuator_biasprm[ctrl_id, 2],
        model.dof_armature[dof],
        effort,
      )
      if not all(
        isclose(a, e, rel_tol=1e-6) for a, e in zip(actual, expected, strict=True)
      ):
        raise ValueError(
          f"Compiled Roban drive mismatch for {name}: {actual} vs {expected}"
        )
      print(
        f"[Roban drive] {name}: Kp={actual[0]:g} Kd={actual[1]:g} "
        f"armature={actual[2]:g} effort={effort:g} speed={speed:g} delay=0",
        flush=True,
      )
      dof_ids.append(dof)
      effort_and_speed.append((effort, speed))
      pd_parameters[name] = (
        model.actuator_gainprm[ctrl_id, 0],
        -model.actuator_biasprm[ctrl_id, 2],
        effort,
      )

    joint_names = self._entity.joint_names
    if tuple(self.target_names) != tuple(joint_names):
      raise ValueError("Roban CCRP position targets must follow the robot joint order")
    self._pd_parameters = torch.tensor(
      [pd_parameters[name] for name in joint_names],
      device=env.device,
      dtype=torch.float32,
    )
    device = self._sim.wp_device
    self._ctrl_ids = wp.array(ctrl_ids, dtype=wp.int32, device=device)
    self._dof_ids = wp.array(dof_ids, dtype=wp.int32, device=device)
    self._effort_and_speed = wp.array(effort_and_speed, dtype=wp.vec2, device=device)
    self._launch_dim = (env.num_envs, len(ctrl_ids))
    self._sim.expand_model_fields(("actuator_forcerange",))

  def torque_cost(
    self, previous_state: torch.Tensor, position_target: torch.Tensor
  ) -> torch.Tensor:
    """Square normalized PD demand before clipping, at the start of a control step."""
    q, qd = decompose_rep_state(previous_state)[-2:]
    kp, kd, effort = self._pd_parameters.unbind(-1)
    torque = kp * (position_target - q) - kd * qd
    return (torque / effort).square().mean(-1)

  def apply_actions(self) -> None:
    super().apply_actions()
    # The action manager calls this before every physics substep. Leave the
    # built-in PD and the braking side of its force range in MuJoCo.
    wp.launch(
      _update_motor_torque_bounds,
      dim=self._launch_dim,
      inputs=[
        self._ctrl_ids,
        self._dof_ids,
        self._effort_and_speed,
        self._sim.wp_data.qvel,
      ],
      outputs=[self._sim.wp_model.actuator_forcerange],
      device=self._sim.wp_device,
    )
