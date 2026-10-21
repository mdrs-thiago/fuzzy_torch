"""Model descriptions for rendering and cross-validation.

The environments do **not** need these to run. Dynamics are hand-derived torch ODEs,
the way Gymnasium's own classic-control envs work, so there is no physics engine in
the loop. These files serve two offline purposes:

1. **Rendering.** MuJoCo draws the scene from a state we hand it: we set ``qpos``
   from the torch state and call ``mj_forward``. ``mj_step`` is never called, so
   MuJoCo never integrates the plant -- the dynamics stay ours, differentiable and
   batchable.
2. **Cross-validation.** The same model, integrated by MuJoCo's own RK4, is an
   independent check on our derivation. The analytical invariants are partly
   self-consistent (our equations of motion against our own energy function, both
   from one Lagrangian); an engine that derives its own equations from a description
   containing none is what catches a mis-derivation.
"""
import pathlib

ASSET_DIR = pathlib.Path(__file__).parent


def quadrotor_mjcf(target=(0.0, 0.0, 1.5), offwidth: int = 1280,
                    offheight: int = 720) -> str:
    """Asset-free MuJoCo scene for the renderer-only quadrotor view."""
    tx, ty, tz = (float(v) for v in target)
    return f"""<mujoco model="quadrotor">
  <visual><global offwidth="{offwidth}" offheight="{offheight}" fovy="48"/>
    <map znear=".03" zfar="60"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".30 .47 .69" rgb2=".82 .90 .98"
             width="512" height="512"/>
    <texture name="ground_grid" type="2d" builtin="checker" rgb1=".18 .23 .20"
             rgb2=".24 .30 .27" width="256" height="256"/>
    <material name="ground" texture="ground_grid" texrepeat="12 12" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="-5 -4 10" dir=".4 .3 -1" directional="true" diffuse=".9 .9 .9"/>
    <geom type="plane" size="20 20 .1" material="ground"/>
    <geom type="cylinder" pos="{tx} {ty} .012" size=".45 .012" rgba=".10 .62 .96 .45"/>
    <geom type="capsule" fromto="{tx} {ty} 0 {tx} {ty} {tz}" size=".012" rgba=".12 .64 .95 .45"/>
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom type="ellipsoid" pos="0 0 0" size=".15 .11 .045" rgba=".08 .11 .16 1"/>
      <geom type="capsule" fromto="-.23 -.23 0 .23 .23 0" size=".018" rgba=".20 .24 .30 1"/>
      <geom type="capsule" fromto="-.23 .23 0 .23 -.23 0" size=".018" rgba=".20 .24 .30 1"/>
      <geom type="cylinder" pos=".23 .23 0" size=".050 .025" rgba=".06 .07 .08 1"/>
      <geom type="cylinder" pos="-.23 .23 0" size=".050 .025" rgba=".06 .07 .08 1"/>
      <geom type="cylinder" pos="-.23 -.23 0" size=".050 .025" rgba=".06 .07 .08 1"/>
      <geom type="cylinder" pos=".23 -.23 0" size=".050 .025" rgba=".06 .07 .08 1"/>
      <geom type="cylinder" pos=".23 .23 .028" size=".16 .004" rgba=".24 .58 .94 .46"/>
      <geom type="cylinder" pos="-.23 .23 .028" size=".16 .004" rgba=".96 .28 .16 .46"/>
      <geom type="cylinder" pos="-.23 -.23 .028" size=".16 .004" rgba=".24 .58 .94 .46"/>
      <geom type="cylinder" pos=".23 -.23 .028" size=".16 .004" rgba=".96 .28 .16 .46"/>
      <geom type="box" pos=".10 0 -.055" size=".055 .040 .018" rgba=".85 .87 .90 1"/>
      <geom type="sphere" pos=".13 0 -.076" size=".035" rgba=".05 .08 .10 1"/>
    </body>
  </worldbody>
</mujoco>"""


def crane_mjcf(trolley_mass: float = 2.0,
               payload_mass: float = 0.5,
               cable_length: float = 2.0,
               gravity: float = 9.81,
               x_limit: float = 10.0,
               target_x: float = 5.0,
               timestep: float = 1e-4,
               offwidth: int = 1280,
               offheight: int = 720) -> str:
    """
    MJCF for the gantry crane: a prismatic trolley carrying a revolute pendulum.

    Physics comes from the explicit ``<inertial>`` elements -- the payload is a point
    mass, so its own rotational inertia is negligible and set to 1e-9. Every
    decorative geom carries ``density="0"`` and no collision flags, so the gantry,
    cable and crate change the appearance without touching the dynamics.

    The swing axis is ``0 -1 0`` so a positive joint angle puts the payload at
    ``+L*sin(theta)`` in x, matching this package's sign convention.
    """
    tiny = 1e-9
    span = x_limit + 1.4
    beam_z = cable_length + 4.0
    cable_vis = cable_length - 0.28          # visual cable, shortened by the hook
    return f"""<mujoco model="gantry_crane">
  <option integrator="RK4" gravity="0 0 {-gravity}" timestep="{timestep}"/>

  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <headlight ambient="0.40 0.40 0.44" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>
    <rgba haze="0.87 0.90 0.94 1"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.05" zfar="80" shadowclip="10"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.66 0.82" rgb2="0.93 0.95 0.98"
             width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.48 0.52 0.58"
             rgb2="0.76 0.79 0.84" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="22 22" texuniform="true"
              reflectance="0.0" specular="0.1" shininess="0.1"/>
    <material name="steel"  rgba="0.46 0.49 0.57 1" specular="0.55" shininess="0.55"/>
    <material name="beam"   rgba="0.33 0.36 0.44 1" specular="0.45" shininess="0.45"/>
    <material name="cab"    rgba="0.15 0.17 0.23 1" specular="0.7"  shininess="0.7"/>
    <material name="cable"  rgba="0.26 0.27 0.31 1" specular="0.35"/>
    <material name="crate"  rgba="0.80 0.44 0.20 1" specular="0.15" shininess="0.1"/>
    <material name="target" rgba="0.16 0.55 0.85 0.55"/>
  </asset>

  <worldbody>
    <light name="key" pos="{span * 0.5} -12 {beam_z + 8}" dir="-0.3 0.55 -1"
           directional="true" diffuse="0.72 0.72 0.70" specular="0.2 0.2 0.2"
           castshadow="true"/>
    <light name="fill" pos="{-span} 9 {beam_z + 5}" dir="0.45 -0.4 -1"
           directional="true" diffuse="0.20 0.22 0.26" castshadow="false"/>

    <geom name="ground" type="plane" size="{span + 6} {span * 0.6} 0.1" material="floor"/>

    <geom name="leg_l"  type="box" pos="{-span} 0 {beam_z / 2}"
          size="0.22 0.22 {beam_z / 2}" material="steel"/>
    <geom name="leg_r"  type="box" pos="{span} 0 {beam_z / 2}"
          size="0.22 0.22 {beam_z / 2}" material="steel"/>
    <geom name="foot_l" type="box" pos="{-span} 0 0.12" size="0.75 1.6 0.12" material="steel"/>
    <geom name="foot_r" type="box" pos="{span} 0 0.12"  size="0.75 1.6 0.12" material="steel"/>
    <geom name="runway" type="box" pos="0 0 {beam_z + 0.1}" size="{span + 0.4} 0.30 0.22"
          material="beam"/>
    <geom name="rail"   type="box" pos="0 0 {beam_z - 0.14}" size="{span + 0.2} 0.10 0.05"
          material="cab"/>

    <geom name="target_mark" type="cylinder" pos="{target_x} 0 0.02" size="0.7 0.02"
          material="target"/>
    <geom name="target_post" type="capsule"
          fromto="{target_x} 0 0 {target_x} 0 {beam_z - cable_length - 0.6}"
          size="0.05" material="target"/>

    <body name="trolley" pos="0 0 {beam_z}">
      <joint name="slide" type="slide" axis="1 0 0" damping="0"/>
      <inertial pos="0 0 0" mass="{trolley_mass}"
                diaginertia="{tiny} {tiny} {tiny}"/>
      <geom name="cab" type="box" pos="0 0 -0.16" size="0.62 0.42 0.24"
            material="cab" density="0" contype="0" conaffinity="0"/>
      <geom name="wheel_l" type="cylinder" pos="-0.42 0 0.02" euler="90 0 0"
            size="0.14 0.34" material="steel" density="0" contype="0" conaffinity="0"/>
      <geom name="wheel_r" type="cylinder" pos="0.42 0 0.02" euler="90 0 0"
            size="0.14 0.34" material="steel" density="0" contype="0" conaffinity="0"/>

      <body name="payload" pos="0 0 0">
        <joint name="swing" type="hinge" axis="0 -1 0" damping="0"/>
        <inertial pos="0 0 {-cable_length}" mass="{payload_mass}"
                  diaginertia="{tiny} {tiny} {tiny}"/>
        <geom name="cable" type="capsule" fromto="0 0 -0.34 0 0 {-cable_vis}"
              size="0.035" material="cable" density="0" contype="0" conaffinity="0"/>
        <geom name="hook" type="box" pos="0 0 {-cable_vis - 0.09}" size="0.10 0.16 0.09"
              material="steel" density="0" contype="0" conaffinity="0"/>
        <geom name="crate" type="box" pos="0 0 {-cable_length - 0.32}"
              size="0.62 0.55 0.45" material="crate" density="0"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="slide" gear="1" ctrlrange="-1000 1000"/>
  </actuator>
</mujoco>
"""
