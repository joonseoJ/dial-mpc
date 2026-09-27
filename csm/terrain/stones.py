"""Go2 on a stepping-stone strip whose unseen cells may be holes.

The world: a start pad, a strip of NX x NY stones, an end pad.  Every stone is
either solid (top at z = 0, flush with the pads) or a hole (top HOLE_DEPTH
below).  Some stones are seen before the walk starts; the rest are covered
(puddles, debris, occlusion) and only a probability of being a hole is known.

Terrain is carried as the vector of stone tops, `tops` (NX*NY,), and enters the
physics through `geom_pos` alone.  `terrain_sys(sys, tops)` returns the system
for one terrain, so a planner can vmap its rollouts over M hypothesised
terrains with nothing else duplicated.

The Go2 reward is the stock walking objective, unchanged: stones are flush
with the pads, so on solid ground every term means what it means on the flat,
and a foot that drops into a hole is priced by the gait, height and upright
terms the way any stumble is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from brax.io import mjcf
from brax.envs.base import State

from dial_mpc.envs.unitree_go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
from dial_mpc.utils.io_utils import get_model_path

# ---- strip geometry -------------------------------------------------------
X0 = 0.45            # first stone's rear edge (m); the robot's head starts at 0.28
CX = 0.15            # stone length along the walk
NX = 12              # stones along x -> 1.8 m strip
NY = 2               # lanes: left feet walk y in (0, 0.3), right feet (-0.3, 0)
CY = 0.30
GAP = 0.004          # clearance between stones, smaller than a foot (0.035 m)
HALF_Z = 0.25        # stone half height
HOLE_DEPTH = 0.30    # a hole's top sits this far below the walking surface
FLOOR_Z = -0.35      # the floor under everything, so a foot in a hole lands
PAD_BACK = 1.0       # start pad extends this far behind x = 0
PAD_FRONT = 3.0      # end pad length
N_CELL = NX * NY
SOLID, HOLE = 0.0, -HOLE_DEPTH

SCENE = "mjx_scene_force_stones.xml"


def configure(ny: int, cy: float):
    """Change the strip's lanes before an env is built.

    The default two 0.30 m lanes leave a foot no sideways choice: each foot
    line has one lane, so avoiding an unseen stone can only be done by
    changing stride.  Five 0.14 m lanes centred on y = 0, +-0.14, +-0.28 put
    the feet (y = +-0.142) on lane centres and let a sidestep of one lane
    carry them onto different stones -- a route choice, priced by the
    lateral-velocity term of the tracking row.
    """
    global NY, CY, N_CELL, SCENE
    NY, CY = ny, cy
    N_CELL = NX * NY
    SCENE = "mjx_scene_force_stones.xml" if (ny, cy) == (2, 0.30) else \
        f"mjx_scene_force_stones_{ny}x{int(round(cy * 100))}.xml"


def cell_centres():
    """(N_CELL, 2) xy centre of every stone, index = ix * NY + iy."""
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    x = X0 + (ix + 0.5) * CX
    y = -CY * NY / 2 + (iy + 0.5) * CY
    return np.stack([x.ravel(), y.ravel()], -1)


def write_scene() -> str:
    """Generate the scene next to the Go2 model (the include is relative)."""
    ctr = cell_centres()
    end = X0 + NX * CX
    geoms = [
        f'<geom name="floor" size="0 0 0.05" type="plane" pos="0 0 {FLOOR_Z}" material="groundplane"/>',
        f'<geom name="pad_start" type="box" contype="1" conaffinity="0" rgba="0.55 0.55 0.5 1" '
        f'size="{(X0 + PAD_BACK) / 2:.4f} 0.6 {HALF_Z}" pos="{(X0 - PAD_BACK) / 2:.4f} 0 {-HALF_Z}"/>',
        f'<geom name="pad_end" type="box" contype="1" conaffinity="0" rgba="0.55 0.55 0.5 1" '
        f'size="{PAD_FRONT / 2:.4f} 0.6 {HALF_Z}" pos="{end + PAD_FRONT / 2:.4f} 0 {-HALF_Z}"/>',
    ]
    for i, (x, y) in enumerate(ctr):
        geoms.append(
            f'<geom name="stone{i}" type="box" contype="1" conaffinity="0" rgba="0.45 0.5 0.6 1" '
            f'size="{CX / 2 - GAP:.4f} {CY / 2 - GAP:.4f} {HALF_Z}" pos="{x:.4f} {y:.4f} {-HALF_Z}"/>')
    # Side walls are left out on purpose: a foot that misses the strip
    # sideways falls, as it would on a real one.
    xml = f"""<mujoco model="go2 stones">
  <include file="mjx_go2_force.xml"/>
  <statistic center="1 0 0.1" extent="1.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 1.5" dir="0 0 -1" directional="true"/>
    {chr(10).join('    ' + g for g in geoms).lstrip()}
  </worldbody>
</mujoco>
"""
    path = get_model_path("unitree_go2", SCENE)
    with open(path, "w") as f:
        f.write(xml)
    return str(path)


@dataclass
class StonesEnvConfig(UnitreeGo2EnvConfig):
    pass


class StonesEnv(UnitreeGo2Env):
    """The stock Go2 walk on the stone strip.  `self.sys` holds whatever
    terrain was last installed; planners never rely on it and pass their own."""

    def make_system(self, config):
        write_scene()
        sys = mjcf.load(get_model_path("unitree_go2", SCENE))
        return sys.tree_replace({"opt.timestep": config.timestep})

    def __init__(self, config: StonesEnvConfig):
        super().__init__(config)
        mj = self.sys.mj_model
        self.stone_ids = jnp.array([
            mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, f"stone{i}") for i in range(N_CELL)])
        self.base_sys = self.sys

    def terrain_sys(self, tops):
        """The system for one terrain (`tops` (N_CELL,), stone top heights)."""
        pos = self.base_sys.geom_pos.at[self.stone_ids, 2].set(tops - HALF_Z)
        return self.base_sys.tree_replace({"geom_pos": pos})

    def step_in(self, tops, state: State, action) -> State:
        """`env.step` under terrain `tops`.  Brax's pipeline reads `self.sys`
        at trace time, so swap it for the duration of the trace."""
        old = self.sys
        self.sys = self.terrain_sys(tops)
        try:
            return self.step(state, action)
        finally:
            self.sys = old

    def reset(self, rng):
        state = super().reset(rng)
        return state


# ---- terrains and beliefs ---------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """How an episode's strip is drawn.

    Each stone is seen with probability `p_seen`.  Seen stones are known
    exactly.  An unseen stone carries a hole probability drawn from
    `p_levels` (what the robot can guess from what it looks like), and its
    true state is drawn from that probability, so the belief is calibrated.
    Seen stones are holes with probability `p_seen_hole`.
    """
    p_seen: float = 0.4
    p_seen_hole: float = 0.25
    p_levels: tuple = (0.1, 0.3, 0.5)
    clear_first: int = 0        # stones at the start forced solid and seen


def sample_episode(key, lay: Layout):
    """-> tops_true (N_CELL,), p_hole (N_CELL,) belief (0/1 where seen)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    seen = jax.random.uniform(k1, (N_CELL,)) < lay.p_seen
    p_unseen = jnp.asarray(lay.p_levels)[jax.random.randint(k2, (N_CELL,), 0, len(lay.p_levels))]
    p_true = jnp.where(seen, lay.p_seen_hole, p_unseen)
    hole = jax.random.uniform(k3, (N_CELL,)) < p_true
    if lay.clear_first:
        first = jnp.arange(N_CELL) < lay.clear_first * NY
        hole = hole & ~first
        seen = seen | first
    belief = jnp.where(seen, hole.astype(jnp.float32), p_unseen)
    return jnp.where(hole, HOLE, SOLID), belief


def sample_particles(key, belief, m):
    """(m, N_CELL) terrains drawn independently from the per-stone belief."""
    hole = jax.random.uniform(key, (m, N_CELL)) < belief[None]
    return jnp.where(hole, HOLE, SOLID)


def representative(belief, kind):
    """Single-terrain stand-ins for the belief."""
    if kind == "optimistic":            # every unseen stone solid
        return jnp.where(belief >= 1.0, HOLE, SOLID)
    if kind == "conservative":          # every unseen stone a hole
        return jnp.where(belief > 0.0, HOLE, SOLID)
    if kind == "likely":                # each stone its more likely state
        return jnp.where(belief > 0.5, HOLE, SOLID)
    raise ValueError(kind)
