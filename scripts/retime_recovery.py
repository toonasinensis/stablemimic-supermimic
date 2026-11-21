"""Retime fall-sourced Box v15 Recovery references at a fixed 50 Hz."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path, required=True)
parser.add_argument("--source-manifest", type=Path, required=True)
parser.add_argument("--robot-xml", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
source_manifest = json.loads(args.source_manifest.read_text())
paths = sorted(args.source_dir.glob("*.npz"))
records = {row["file"]: row for row in source_manifest["clips"]}
excluded_clip = "fallAndGetUp2_subject3_supine_getup_002460_002784.npz"
assert paths[64].name == excluded_clip
assert len(paths) == len(records) == 109
factor = 1.5
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
fields = (
  "root_pos_w",
  "root_quat_w",
  "joint_pos",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_vel",
)
model = mj.MjModel.from_xml_path(str(args.robot_xml))
data = mj.MjData(model)
meshes, spheres = {}, []
for geom in np.flatnonzero(model.geom_contype):
  if model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    meshes[geom] = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
  elif model.geom_type[geom] == mj.mjtGeom.mjGEOM_SPHERE:
    spheres.append(geom)
assert len(meshes) == 28 and len(spheres) == 2


def restore_floor_clearance(motion, count, hold):
  joint_addresses = [
    int(model.joint(name).qposadr[0]) for name in motion["joint_names"]
  ]
  lift = np.zeros(count)
  for frame in range(count):
    data.qpos[:3] = motion["root_pos_w"][frame]
    data.qpos[3:7] = motion["root_quat_w"][frame]
    data.qpos[joint_addresses] = motion["joint_pos"][frame]
    mj.mj_kinematics(model, data)
    bottom = min(
      *(
        (v @ data.geom_xmat[g, 6:9]).min() + data.geom_xpos[g, 2]
        for g, v in meshes.items()
      ),
      *(data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in spheres),
    )
    lift[frame] = max(0.0, -bottom) if bottom < -1e-6 else 0.0
  motion["root_pos_w"][:, 2] += np.pad(lift, (0, hold), mode="edge")
  motion["root_lin_vel_w"][:count, 2] += np.gradient(lift, 1 / 50)
  return float(lift.max())


clips, excluded = [], []
for index, source in enumerate(paths, start=1):
  if source.name == excluded_clip or not source.name.startswith(
    ("fallAndGetUp", "pushAndFall")
  ):
    excluded.append(
      dict(
        original_index=index,
        file=source.name,
        reason="user_excluded_65"
        if source.name == excluded_clip
        else "non_fall_source",
      )
    )
    continue
  record = records[source.name]
  assert hashlib.sha256(source.read_bytes()).hexdigest() == record["sha256"]
  with np.load(source, allow_pickle=False) as clip:
    motion = dict(clip)
  count, hold = record["recovery_frames"], record["static_tail_frames"]
  assert float(motion["fps"].item()) == 50 and len(motion["joint_pos"]) == count + hold
  grid = np.arange(count)
  new_count = round((count - 1) * factor) + 1
  sample_at = np.linspace(0, count - 1, new_count)
  velocity_scale = (count - 1) / (new_count - 1)
  for key in fields:
    frames = motion[key][:count]
    if key == "root_quat_w":
      retimed = Slerp(grid, Rotation.from_quat(frames, scalar_first=True))(
        sample_at
      ).as_quat(scalar_first=True)
      retimed[0] = frames[0]
      if np.dot(retimed[-1], frames[-1]) < 0:
        retimed[-1] = -retimed[-1]
    else:
      retimed = np.stack(
        [np.interp(sample_at, grid, channel) for channel in frames.T], axis=-1
      )
    if "vel" in key:
      retimed *= velocity_scale
      tail = np.zeros_like(motion[key][count:])
    else:
      tail = np.repeat(retimed[-1:], hold, axis=0)
    motion[key] = np.concatenate((retimed, tail)).astype(frames.dtype)
    assert np.isfinite(motion[key]).all()
  floor_lift = restore_floor_clearance(motion, new_count, hold)
  path = out / source.name
  np.savez_compressed(path, **motion)
  clips.append(
    dict(
      original_index=index,
      file=path.name,
      source_frames=count,
      recovery_frames=new_count,
      static_tail_frames=hold,
      extra_floor_lift_m=floor_lift,
      source_sha256=record["sha256"],
      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
  )
assert len(clips) == 95 and len(excluded) == 14
assert {path.name for path in out.glob("*.npz")} == {row["file"] for row in clips}
(out / "dataset_manifest.json").write_text(
  json.dumps(
    dict(
      fps=50,
      duration_factor=factor,
      static_tail_s=3,
      source_families=["fallAndGetUp", "pushAndFall"],
      source_manifest_sha256=hashlib.sha256(
        args.source_manifest.read_bytes()
      ).hexdigest(),
      clips=clips,
      excluded=excluded,
    ),
    indent=2,
  )
  + "\n"
)
print(
  f"Prepared {len(clips)} clips; {sum(row['recovery_frames'] + row['static_tail_frames'] for row in clips)} frames at 50 Hz."
)
