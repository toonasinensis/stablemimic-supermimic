"""Roban S2.2 plant and task settings for native SuperMimic recovery."""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ROBAN_RECOVERY_TASK = "SuperMimic-Recovery-RobanS22"


def get_roban_xml_path() -> Path:
  return Path(os.environ["ROBAN_RECOVERY_XML"]).expanduser()


def get_spec():
  return mujoco.MjSpec.from_file(str(get_roban_xml_path()))


ACTUATOR_DELAY_MIN_LAG = 0
ACTUATOR_DELAY_MAX_LAG = 4

ROBAN_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l4_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r4_joint",
  ),
  stiffness=80.0,
  damping=5.0,
  effort_limit=150.0,
  armature=0.02,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ACTUATOR_LEG_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("leg_l3_joint", "leg_r3_joint"),
  stiffness=50.0,
  damping=3.0,
  effort_limit=70.0,
  armature=0.01,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ACTUATOR_ANKLE_SERIAL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r5_joint",
    "leg_r6_joint",
  ),
  stiffness=40.0,
  damping=3.0,
  effort_limit=74.0,
  armature=0.01,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ACTUATOR_ANKLE = ROBAN_ACTUATOR_ANKLE_SERIAL

ROBAN_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=50.0,
  damping=3.0,
  effort_limit=80.0,
  armature=0.012,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=("zarm_l1_joint", "zarm_r1_joint"),
  stiffness=15.0,
  damping=1.0,
  effort_limit=14.1,
  armature=0.002,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
  ),
  stiffness=20.0,
  damping=1.5,
  effort_limit=37.0,
  armature=0.005,
  delay_min_lag=ACTUATOR_DELAY_MIN_LAG,
  delay_max_lag=ACTUATOR_DELAY_MAX_LAG,
)

ROBAN_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ROBAN_ACTUATOR_LEG,
    ROBAN_ACTUATOR_LEG_YAW,
    ROBAN_ACTUATOR_ANKLE,
    ROBAN_ACTUATOR_WAIST,
    ROBAN_ACTUATOR_SHOULDER,
    ROBAN_ACTUATOR_ARM,
  ),
  soft_joint_pos_limit_factor=0.9,
)

DEFAULT_INIT = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def get_roban_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=DEFAULT_INIT,
    spec_fn=get_spec,
    articulation=ROBAN_ARTICULATION,
  )


ROBAN_TRACKING_BODIES = (
  "base_link",
  "leg_l3_link",
  "leg_l4_link",
  "leg_l6_link",
  "leg_r3_link",
  "leg_r4_link",
  "leg_r6_link",
  "waist_yaw_link",
  "zarm_l1_link",
  "zarm_l2_link",
  "zarm_l3_link",
  "zarm_l4_link",
  "zarm_r1_link",
  "zarm_r2_link",
  "zarm_r3_link",
  "zarm_r4_link",
)
