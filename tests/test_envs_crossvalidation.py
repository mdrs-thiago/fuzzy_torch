"""Cross-validate the hand-derived torch dynamics against an independent engine.

The analytical invariants in ``test_envs_crane.py`` are strong but partly
self-consistent: energy conservation checks our equations of motion against our own
energy function, and a shared mistake in the Lagrangian both were derived from would
satisfy them together. The small-angle period test is genuinely independent but only
probes the linearised regime.

This closes that gap. The same body is described in MJCF and handed to MuJoCo, which
derives and integrates its own equations of motion from a description containing no
equations. MuJoCo is configured with RK4, matching our integrator's order, so
agreement is limited by the physics rather than by discretisation -- which is why the
tolerances here are far tighter than an engine with a first-order integrator allows.

Needs no GL context: only ``mj_step``, never the renderer.
"""
import pytest
import torch

mujoco = pytest.importorskip("mujoco",
                             reason="cross-validation needs mujoco (pip install 'torch-fis[viz]')")

from fuzzy_torch.envs import CranePlant, rk4_step  # noqa: E402
from fuzzy_torch.envs.assets import crane_mjcf  # noqa: E402

MASS_TROLLEY, MASS_PAYLOAD, LENGTH, G = 2.0, 0.5, 2.0, 9.81
DT = 1e-4
STEPS = 8000          # 0.8 s -- long enough for a large-angle swing to diverge if wrong


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_string(
        crane_mjcf(MASS_TROLLEY, MASS_PAYLOAD, LENGTH, G, timestep=DT))
    return model, mujoco.MjData(model)


def mujoco_rollout(mj, theta0, force):
    model, data = mj
    mujoco.mj_resetData(model, data)
    data.qpos[:] = [0.0, theta0]
    data.qvel[:] = [0.0, 0.0]
    for _ in range(STEPS):
        data.ctrl[0] = force
        mujoco.mj_step(model, data)
    return torch.tensor([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]],
                        dtype=torch.float64)


def torch_rollout(theta0, force):
    plant = CranePlant(trolley_mass=MASS_TROLLEY, payload_mass=MASS_PAYLOAD,
                       cable_length=LENGTH, gravity=G).double()
    state = plant.initial_state(1, randomize=False)
    state[0, 2] = theta0
    u = torch.tensor([[force]], dtype=torch.float64)
    for _ in range(STEPS):
        state = rk4_step(plant.derivatives, state, u, DT)
    return state[0]


CASES = [
    (0.20, 0.0),     # free swing, small
    (0.60, 0.0),     # free swing, well outside the small-angle regime
    (0.00, 10.0),    # pure forcing, sway excited by the trolley
    (0.30, -8.0),    # both at once, opposing signs
]


@pytest.mark.parametrize("theta0, force", CASES)
def test_matches_mujoco(mj, theta0, force):
    ours = torch_rollout(theta0, force)
    theirs = mujoco_rollout(mj, theta0, force)
    assert torch.allclose(ours, theirs, atol=1e-6), (
        f"torch {ours.tolist()} vs mujoco {theirs.tolist()}")


def test_agreement_is_far_tighter_than_discretisation_error(mj):
    """Guards the tolerance: agreement must be near round-off, not merely 'close'.

    If it ever degrades to ~1e-4 the physics has drifted, even though a loose
    tolerance would still pass.
    """
    worst = max(
        float((torch_rollout(t, f) - mujoco_rollout(mj, t, f)).abs().max())
        for t, f in CASES
    )
    assert worst < 1e-6, f"worst disagreement {worst:.3e}"


def test_decorative_geoms_do_not_change_the_dynamics(mj):
    """The MJCF carries a gantry, cable and crate for rendering.

    They are ``density="0"`` with explicit ``<inertial>`` on every body, so they must
    contribute no mass. A mismatch here would silently corrupt every comparison above.
    """
    model, _ = mj
    assert model.body_mass.sum() == pytest.approx(MASS_TROLLEY + MASS_PAYLOAD, rel=1e-12)


def test_disagrees_with_an_inverted_pendulum(mj):
    """Guards the guard: the comparison must be able to *fail*.

    verify_physics.py implements cart-pole -- gravity destabilising, plus a 4/3
    uniform-rod factor -- under the crane name. A passing test above only means
    something if the check rejects that model.
    """
    theta0 = 0.6

    def inverted(state, u):
        x_dot, theta, omega = state[..., 1], state[..., 2], state[..., 3]
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        temp = ((u[..., 0] + MASS_PAYLOAD * LENGTH * omega ** 2 * sin_t)
                / (MASS_TROLLEY + MASS_PAYLOAD))
        theta_acc = ((G * sin_t - cos_t * temp)
                     / (LENGTH * (4.0 / 3.0
                                  - MASS_PAYLOAD * cos_t ** 2 / (MASS_TROLLEY + MASS_PAYLOAD))))
        x_acc = temp - MASS_PAYLOAD * LENGTH * theta_acc * cos_t / (MASS_TROLLEY + MASS_PAYLOAD)
        return torch.stack([x_dot, x_acc, omega, theta_acc], dim=-1)

    state = torch.zeros((1, 4), dtype=torch.float64)
    state[0, 2] = theta0
    u = torch.zeros((1, 1), dtype=torch.float64)
    for _ in range(STEPS):
        state = rk4_step(inverted, state, u, DT)

    assert not torch.allclose(state[0], mujoco_rollout(mj, theta0, 0.0), atol=1e-6)
