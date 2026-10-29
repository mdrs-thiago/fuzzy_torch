"""The laptop protocol of kart_net, over localhost (no SuperTuxKart needed)."""
import threading
import time

import numpy as np
import pytest

from fuzzy_torch.envs.kart_net import KartHub, join


def _wait(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError('timed out')
        time.sleep(0.01)


@pytest.fixture
def hub():
    hub = KartHub('127.0.0.1', 0, timeout=0.5)
    yield hub
    hub.close()


def _laptop(hub, name, driver):
    connection = join(f'127.0.0.1:{hub.address[1]}', name, driver)
    threading.Thread(target=connection.run, daemon=True).start()
    return connection


def test_a_laptop_drives_its_kart_one_step_late(hub):
    _laptop(hub, 'Ana', lambda obs: np.array([0.5, -obs[0]]))
    _wait(lambda: hub.names == ['Ana'])
    driver = hub.drivers['Ana']
    driver.reset()
    assert driver(np.array([0.3, 0, 0, 0, 0, 0])).tolist() == [0.0, 0.0]   # nothing heard yet
    _wait(lambda: driver._action[0] == 0.5)
    assert driver(np.zeros(6)) == pytest.approx([0.5, -0.3])               # the previous answer


def test_names_are_unique_ignoring_case_and_spacing(hub):
    _laptop(hub, 'Ana Souza', lambda obs: np.zeros(2))
    _wait(lambda: hub.names == ['Ana Souza'])
    with pytest.raises(ConnectionError, match='already connected'):
        join(f'127.0.0.1:{hub.address[1]}', 'ana  souza', lambda obs: np.zeros(2))


def test_a_silent_laptop_times_out_and_a_closed_one_disconnects(hub):
    gate = threading.Event()

    def stuck(obs):
        gate.wait()
        return np.ones(2)

    connection = _laptop(hub, 'Slow', stuck)
    _wait(lambda: hub.names == ['Slow'])
    driver = hub.drivers['Slow']
    driver.reset()
    driver(np.zeros(6))
    time.sleep(0.6)
    with pytest.raises(TimeoutError):
        driver(np.zeros(6))
    gate.set()
    connection.sock.close()
    _wait(lambda: not driver.connected or not hub.names)
    with pytest.raises(ConnectionError):
        driver(np.zeros(6))
        driver(np.zeros(6))


def test_a_shared_spec_gives_the_projector_fuzzy_panels(hub):
    import json
    import pathlib
    spec = json.loads((pathlib.Path(__file__).resolve().parents[1]
                       / 'examples/race_control/example_driver.json').read_text())
    join(f'127.0.0.1:{hub.address[1]}', 'Ana', lambda obs: np.zeros(2), spec)
    _wait(lambda: 'Ana' in hub.drivers and hub.drivers['Ana'].steering is not None)
    assert hub.drivers['Ana'].throttle is not None


def test_bad_hellos_are_refused(hub):
    with pytest.raises(ConnectionError, match='name'):
        join(f'127.0.0.1:{hub.address[1]}', ' ', lambda obs: np.zeros(2))
