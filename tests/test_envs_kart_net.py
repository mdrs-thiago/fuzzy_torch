"""The laptop protocol of kart_net, over localhost (no SuperTuxKart needed)."""
import json
import pathlib
import threading
import time

import numpy as np
import pytest

pytest.importorskip('websockets')

from fuzzy_torch.envs.kart_net import KartHub, join, ws_url  # noqa: E402

CODE = 'K7QX2'


def _wait(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise AssertionError('timed out')
        time.sleep(0.01)


@pytest.fixture
def hub():
    hub = KartHub('127.0.0.1', 0, code=CODE, timeout=0.5)
    yield hub
    hub.close()


def _address(hub):
    return f'127.0.0.1:{hub.address[1]}'


def _laptop(hub, name, driver, code=CODE):
    connection = join(_address(hub), name, code, driver)
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


def test_the_join_code_is_required_but_not_case_sensitive(hub):
    with pytest.raises(ConnectionError, match='join code'):
        join(_address(hub), 'Eve', 'WRONG', lambda obs: np.zeros(2))
    _laptop(hub, 'Ana', lambda obs: np.zeros(2), code=CODE.lower())
    _wait(lambda: hub.names == ['Ana'])


def test_names_are_unique_ignoring_case_and_spacing(hub):
    _laptop(hub, 'Ana Souza', lambda obs: np.zeros(2))
    _wait(lambda: hub.names == ['Ana Souza'])
    with pytest.raises(ConnectionError, match='already connected'):
        join(_address(hub), 'ana  souza', CODE, lambda obs: np.zeros(2))


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
    connection.close()
    _wait(lambda: not driver.connected)
    with pytest.raises(ConnectionError):
        driver(np.zeros(6))


def test_a_shared_spec_gives_the_race_screen_fuzzy_panels(hub):
    spec = json.loads((pathlib.Path(__file__).resolve().parents[1]
                       / 'examples/race_control/example_driver.json').read_text())
    join(_address(hub), 'Ana', CODE, lambda obs: np.zeros(2), spec)
    _wait(lambda: 'Ana' in hub.drivers and hub.drivers['Ana'].steering is not None)
    assert hub.drivers['Ana'].throttle is not None


def test_bad_names_are_refused(hub):
    with pytest.raises(ConnectionError, match='name'):
        join(_address(hub), ' ', CODE, lambda obs: np.zeros(2))


def test_addresses_students_type_become_websocket_urls():
    assert ws_url('192.168.0.10:9000') == 'ws://192.168.0.10:9000'
    assert ws_url('https://abcd.ngrok-free.app/') == 'wss://abcd.ngrok-free.app'
    assert ws_url('http://localhost:9000') == 'ws://localhost:9000'
    assert ws_url('wss://example.org/race') == 'wss://example.org/race'
