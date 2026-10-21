"""Contract and physics tests for the 3-D quadrotor plant."""
import pytest
import torch

from fuzzy_torch.envs import ControlMode, QuadrotorPlant


def test_hover_thrust_cancels_gravity_when_level():
    drone = QuadrotorPlant(linear_drag=0.0, angular_drag=0.0)
    state = drone.initial_state(1, randomize=False)
    hover = torch.full((1, 4), float(drone.m * drone.g / 4.0))
    derivative = drone.derivatives(state, drone.map_action(hover, state))
    assert torch.allclose(derivative[0, 3:6], torch.zeros(3), atol=1e-6)
    assert torch.allclose(derivative[0, 10:13], torch.zeros(3), atol=1e-6)


def test_rotor_mixing_generates_the_expected_roll_torque():
    drone = QuadrotorPlant()
    state = drone.initial_state(1, randomize=False)
    physical = drone.map_action(torch.tensor([[1.0, 1.0, 0.0, 0.0]]), state)
    assert physical[0, 1] > 0.0
    assert float(physical[0, 2]) == pytest.approx(0.0)


def test_rate_thrust_mode_closes_the_inner_rate_loop():
    drone = QuadrotorPlant(mode=ControlMode.RATE_THRUST)
    state = drone.initial_state(1, randomize=False)
    action = torch.tensor([[float(drone.m * drone.g), 1.0, -2.0, .5]])
    physical = drone.map_action(action, state)
    assert physical.shape == (1, 4)
    assert torch.all(physical[0, 1:] * action[0, 1:] > 0.0)


def test_drone_energy_is_conserved_in_unforced_drag_free_flight():
    drone = QuadrotorPlant(linear_drag=0.0, angular_drag=0.0)
    state = drone.initial_state(1, randomize=False).double()
    state[0, 2], state[0, 3:6], state[0, 10:13] = 3.0, torch.tensor([1.0, -.5, .7]), torch.tensor([.3, .2, -.1])
    drone = drone.double()
    u = torch.zeros((1, 4), dtype=torch.float64)
    energy0 = drone.energy(state)
    from fuzzy_torch.envs import rk4_step
    for _ in range(500):
        state = rk4_step(drone.derivatives, state, u, .001)
    assert float(drone.energy(state)) == pytest.approx(float(energy0), rel=2e-5, abs=2e-5)


@pytest.mark.parametrize('mode', [ControlMode.MOTOR_THRUST, ControlMode.RATE_THRUST])
def test_quadrotor_gym_contract(mode):
    pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env
    from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv
    env = FuzzyControlEnv(QuadrotorPlant(mode=mode), episode_steps=10)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (4,)
