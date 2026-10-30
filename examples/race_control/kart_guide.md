# Fuzzy Kart: student guide

You write a fuzzy driver: two fuzzy systems, one for the **steering** and one for
the **throttle**. On race day it drives a kart in
[SuperTuxKart](https://supertuxkart.net), the open-source kart game, against your
classmates and AI karts. The projector shows your rules firing as your kart races.

Your driver is a JSON file in the Race Control format (see this folder's
[README](README.md)). Start from `kart_starter.json`, a working driver tuned for
SuperTuxKart, and improve it.

## What your driver sees

Every 0.05 s your driver gets six numbers and answers with `[throttle, steering]`.

| input | meaning in SuperTuxKart | typical values |
| --- | --- | --- |
| `cross_track_error` | how far you are from the middle of the road, in half road widths: ±1 is the road edge, positive = left of the middle | −0.5 … 0.5 on the road |
| `heading_error` | road direction minus your heading, radians: positive = the road turns away to your left | −0.2 … 0.2, up to ±0.8 |
| `speed` | forward speed, scaled so the kart's top speed (22 m/s) reads 4 | 2 … 3.5 when racing |
| `yaw_rate` | how fast you are turning, positive = turning left | −1.3 … 1.3 |
| `lookahead_curvature` | the bend about 10 m ahead, positive = left bend | −0.4 … 0.4 |
| `curve` | how sharp that bend is, `abs(lookahead_curvature)` | 0 … 0.45 |
| `progress` | fraction of the race completed | 0 … 1 |

Each system may use up to three inputs. An input outside its `range` is held at
the nearest end of the range, so give your outermost terms shoulders that stay at 1.

Outputs: `throttle` in `[-1, 1]` (negative brakes; there is no reverse), `steering`
in `[-1, 1]` (positive turns left).

## Testing your driver

**On the lecture circuits** (fast, nothing extra to install):

```bash
python examples/race_control/grand_prix.py my_driver.json
```

**In SuperTuxKart** (needs `pip install 'torch-fis[kart]'`; the first run downloads
the game, about 700 MB):

```bash
python examples/supertuxkart_demo.py my_driver.json                        # three tracks
python examples/supertuxkart_demo.py my_driver.json --track zengarden --live
python examples/kart_party.py --driver my_driver.json --ai 3               # with your rules on screen
```

The tracks are `lighthouse`, `zengarden`, `hacienda`, `cornfield_crossing`,
`snowmountain` and `scotland`. A driver that only works on one track is overfitted.

**In Python**, to tune it with code:

```python
import json
from fuzzy_torch.envs import SuperTuxKartEnv
from fuzzy_torch.envs.controllers import ConfigRaceDriver

driver = ConfigRaceDriver(json.load(open('my_driver.json')))
env = SuperTuxKartEnv('zengarden', opponents=3)
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(driver(obs))
    if terminated or truncated:
        break
print(info['race_time'], info['position'], info['rescues'])
```

## Tuning your driver automatically

Once your rules make sense, let the computer fine-tune the numbers:

```bash
python examples/kart_tune.py my_driver.json            # writes my_driver_tuned.json
python examples/kart_tune.py my_driver.json --lag 3    # if you will race from your laptop
```

It only moves membership-function breakpoints and TSK rule outputs. Your inputs,
terms and rules stay as you wrote them, so you can still explain every rule. It
races solo on three tracks, keeps a change only when the total time drops, then
tests the result on three other tracks. Being slower on those means it fitted the
tuning tracks rather than learning to drive: tune on more tracks (`--tracks ...`).
A run takes 5 to 10 minutes. Tuning can't fix a missing rule: look at what your
driver does wrong first, then tune.

## On race day

Your instructor runs the race on the projector. You take part in one of two ways:

- **Submit your JSON** on the Race Control page, as for the lecture circuits. Your
  latest submission races.
- **Drive from your laptop**, if your instructor opens the race for laptops:

  ```bash
  python examples/kart_join.py <address> --code <code> --name "Your Name" --driver my_driver.json
  ```

  The address and the code are shown at the top of the race screen.

  Your window shows your own rules firing while your kart races. Your file is read
  again before each race, so you can change your rules between heats. A laptop that
  stops answering for a second coasts until it answers again, so keep the window
  open and the laptop awake.

With more than 8 students, the class races in heats of up to 8, and the 8 fastest
overall meet in a final.

## Tips

- **Full throttle needs a crisp 1.0.** A Mamdani "full" term defuzzifies to its
  centroid, about 0.8, so the kart never uses full power. The starter's throttle is
  a zero-order TSK (Sugeno) system for that reason: `"then": 1.0` is exactly 1.0.
- **Look ahead.** Steering on `cross_track_error` and `heading_error` alone reacts
  only once the kart is already off line. Adding `lookahead_curvature` starts the
  turn before the bend, as the starter does.
- **Grass is slow, walls are slower.** Leaving the road is allowed, but costs time.
  A kart stuck for 2 s is rescued, which costs more.
- **Watch the inputs, not just the result.** `kart_party.py` shows which rules fire
  at every moment; a rule that never fires, or fires at the wrong time, is the place
  to start.
