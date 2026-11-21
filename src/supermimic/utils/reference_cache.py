"""Precompute immutable motion-body FK; predicted-state FK stays differentiable."""

import torch


class ReferenceBodyCache:
  def __init__(self, motion, chunk_size=2048):
    self.original = motion.body_state
    frames = torch.arange(motion.time_step_total, device=motion.joint_pos.device)
    with torch.no_grad():
      self.table = torch.cat(
        [torch.cat(self.original(chunk), dim=-1) for chunk in frames.split(chunk_size)]
      )

  def __call__(self, frames):
    return self.table[frames].split((3, 4, 3, 3), dim=-1)


def cache_reference_bodies(motion):
  cache = ReferenceBodyCache(motion)
  motion.body_state = cache
  return cache
