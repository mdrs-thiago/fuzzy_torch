# fuzzy_torch

A differentiable Fuzzy Inference System (FIS) library built on **PyTorch**.

Design and evaluate **Mamdani** and **Takagi-Sugeno-Kang (TSK)** fuzzy systems with batched
tensor operations, differentiable membership functions, and composable rules. Every parameter
is an `nn.Parameter`, so a fuzzy system trains with any `torch.optim` optimizer and moves to
the GPU with `.to(device)` like any other `nn.Module`.

## Installation

```bash
pip install torch-fis              # core library
pip install "torch-fis[envs]"      # + Gymnasium control environments
pip install "torch-fis[viz]"       # + MuJoCo rendering and matplotlib
```

The distribution is called `torch-fis`; the import name is `fuzzy_torch`:

```python
import fuzzy_torch
```

The latest development version installs straight from GitHub:

```bash
pip install git+https://github.com/mdrs-thiago/fuzzy_torch.git
```

Or for local development:

```bash
git clone https://github.com/mdrs-thiago/fuzzy_torch.git
cd fuzzy_torch
pip install -e .
```

## Quick Start - Mamdani

```python
import torch
from fuzzy_torch import (
    FuzzyVariable, FuzzyRule, FuzzyModule, Centroid,
    TriangularMF, TrapezoidalMF,
)

# 1. Variables carry a name, a range, and named linguistic terms.
service = FuzzyVariable('Service', 0, 10)
service.add_term('Poor',      TriangularMF(0, 0, 5))
service.add_term('Good',      TriangularMF(0, 5, 10))
service.add_term('Excellent', TriangularMF(5, 10, 10))

food = FuzzyVariable('Food', 0, 10)
food.add_term('Rancid',    TrapezoidalMF(0, 0, 1, 3))
food.add_term('Delicious', TrapezoidalMF(7, 9, 10, 10))

tip = FuzzyVariable('Tip', 0, 30)
tip.add_term('Low',    TriangularMF(0, 0, 15))
tip.add_term('Medium', TriangularMF(0, 15, 30))
tip.add_term('High',   TriangularMF(15, 30, 30))

# 2. Rules: an antecedent (or list of them, AND-connected) and a consequent.
rules = [
    FuzzyRule([(service, 'Poor'), (food, 'Rancid')],       (tip, 'Low')),
    FuzzyRule((service, 'Good'),                            (tip, 'Medium')),
    FuzzyRule([(service, 'Excellent'), (food, 'Delicious')], (tip, 'High')),
]

# 3. Build and run.
fis = FuzzyModule(inputs=[service, food], output=tip, rules=rules,
                  defuzz_method=Centroid(0, 30))

out = fis({'Service': torch.tensor([6.5]), 'Food': torch.tensor([9.8])})
print(f"Tip: {out.item():.2f}")

fis.explain({'Service': torch.tensor(6.5), 'Food': torch.tensor(9.8)})
```

Inputs may be 0-dim scalars, `(batch,)` or `(batch, 1)` tensors, or plain Python numbers.
A 0-dim input gives a 0-dim output; everything else gives `(batch,)`.

## Quick Start - TSK (Sugeno) of any order

Swap the consequent fuzzy set for a `TSKConsequent` and the system switches to TSK inference:
firing strengths weight per-rule crisp polynomials, `sum(w_i * f_i) / sum(w_i)`. No
defuzzification is involved, and the output variable needs no terms.

```python
import torch
from fuzzy_torch import (
    FuzzyVariable, FuzzyRule, TSKSystem, TSKConsequent, GaussianMF,
)

x = FuzzyVariable('X', 0, 10)
x.add_term('Low',  GaussianMF(2.0, 2.0))
x.add_term('High', GaussianMF(8.0, 2.0))

y = FuzzyVariable('Y', 0, 100)          # range only; no terms needed

rules = [
    FuzzyRule((x, 'Low'),  TSKConsequent(order=1)),   # f(x) = c0 + c1*x
    FuzzyRule((x, 'High'), TSKConsequent(order=1)),
]
fis = TSKSystem(inputs=[x], output=y, rules=rules)

# The coefficients are nn.Parameters -- train them like anything else.
inputs  = torch.linspace(0, 10, 200)
targets = 3.0 + 2.0 * inputs

opt = torch.optim.Adam(fis.parameters(), lr=0.05)
for _ in range(1000):
    opt.zero_grad()
    loss = ((fis({'X': inputs}) - targets) ** 2).mean()
    loss.backward()
    opt.step()

fis.print_rules()
```

`order` is the total polynomial degree, and the basis is the **full multivariate polynomial**:
every monomial of degree `<= order`, cross terms included. For `n` inputs at order `p` that is
`C(n + p, p)` coefficients per rule.

| inputs | order | monomials |
|--------|-------|-----------|
| 2 | 0 | `1` |
| 2 | 1 | `1, a, b` |
| 2 | 2 | `1, a, b, a², a·b, b²` |
| 2 | 3 | the above plus `a³, a²b, ab², b³` (10 terms) |

By default `TSKConsequent(normalize=True)` rescales each input from its variable range onto
`[-1, 1]` before raising powers, which keeps high orders conditioned -- a raw cubic term on a
`[0, 30]` universe reaches 27000 and would swamp the constant. The printed coefficients then
apply to the rescaled inputs; pass `normalize=False` to work in raw units.

For a consequent that is not a polynomial at all, subclass `Consequent` and implement
`forward(X: (batch, n_inputs)) -> (batch,)`.

## Visualisation

Sets, variables and systems can draw themselves with matplotlib
(`pip install "torch-fis[plot]"`). Each method returns the matplotlib axes (or figure),
and accepts `ax=` to draw into an existing one.

```python
TriangularMF(0, 5, 10).plot()            # one membership function
service.plot()                           # every term of a variable
service.plot(value=6.5)                  # ... with the degrees reached at 6.5

fis.plot_inference({'Service': 6.5, 'Food': 9.8})   # rule viewer, Mamdani or TSK
fis.plot_surface()                                  # output over the first two inputs
fis.plot_surface('Service', fixed={'Food': 9.0})    # a 1-D slice
```

`plot_inference` is the graphical counterpart of `explain()`: one row per rule
showing each antecedent set cut at the degree the input reaches, the firing strength,
and the implied output set (or, for TSK, the rule's crisp value); the last row shows the
aggregated set and the crisp output. Rule bases larger than 12 rules show only the rules
that fire, unless you pass `active_only=False`.

## Systems as JSON

A system can also be described as plain data and built with `build_system`; the
inverse, `system_to_config`, exports a (possibly trained) system back to it:

```python
from fuzzy_torch import build_system, system_to_config

fis = build_system({
    "inputs": {"error": {"range": [-1, 1], "terms": {
        "neg": ["trap", -1, -1, -0.5, 0], "zero": ["tri", -0.5, 0, 0.5], "pos": ["trap", 0, 0.5, 1, 1]}}},
    "output": {"name": "u", "range": [-1, 1], "terms": {
        "down": ["tri", -1, -1, 0], "hold": ["tri", -1, 0, 1], "up": ["tri", 0, 1, 1]}},
    "rules": [{"if": {"error": "neg"}, "then": "up"},
              {"if": {"error": "zero"}, "then": "hold"},
              {"if": {"error": "pos"}, "then": "down"}],
})
```

A rule whose `then` is a number or a coefficient list makes a TSK system instead.
Malformed specs raise `ConfigError` naming the field, e.g. `rules[1].if.error: unknown term`.
The full format is documented in `config.py`.

## Choosing the mode

The mode is inferred from the rules, so `FuzzyModule` handles both. `MamdaniSystem` and
`TSKSystem` are the same class with an assertion, for when you want the intent stated in code.
A rule base may not mix the two kinds of consequent.

## Configuration

```python
FuzzyModule(
    inputs, output, rules,
    defuzz_method=None,   # Mamdani: Centroid (default), Bisector, MeanOfMaximum
    implication=None,     # Mamdani: Min (default, clipping) or Product (scaling)
    aggregation=None,     # Mamdani: Max (default) or ProbabilisticSum
    tsk_normalize=True,   # TSK: weighted average (True) or weighted sum (False)
    no_rule_value=float('nan'),
    chunk_rules=None,
)
```

**When no rule fires**, there is no meaningful crisp answer, so the output is `NaN` rather
than a plausible-looking number. Set `no_rule_value` (e.g. to the midpoint of the output
range) if you would rather have a fallback. Gradients of the batch items that *did* fire are
unaffected either way.

**Only `Centroid` is differentiable.** `Bisector` and `MeanOfMaximum` rely on hard index
selection, so no gradient reaches the membership functions; constructing a system with one
emits a warning. Each class declares this as `.differentiable`.

## Performance notes

Mamdani inference is `O(batch × rules × resolution)` however it is arranged, so the wins come
from removing redundant work rather than from wider tensors. Against the previous
implementation on the same rule bases (CPU, 4 threads, three runs, min-of-repeats):

| rules × batch × resolution | speedup |
|---|---|
| 25 × 64 × 201    | 2.8-3.1× |
| 25 × 1024 × 201  | 1.9-2.1× |
| 120 × 1024 × 201 | 1.8-1.9× |
| 400 × 4096 × 201 | 1.2-1.6× |
| 120 × 8192 × 501 | 1.0-1.1× |

Ratios rather than absolute times: the absolute numbers moved by 4× between runs on a loaded
machine, while the ratios stayed put. The gain is largest on small-to-medium rule bases, where
the removed per-rule overhead dominates; at the top end the work is memory-bound and the two
implementations converge.

Rules are folded one at a time by default. Batching them into a single
`(batch, rules, resolution)` tensor was measurably *slower* at every size tried on CPU -- it
is memory-bound, and 400 rules over a batch of 4096 materialises 1.3 GB. `chunk_rules` sets
the block width if you want to try the wide form on a GPU.

The consequent grid is cached in `eval()` mode and recomputed in `train()`, where the output
membership parameters are still moving.

## Features

- **Membership functions**: Triangular, Trapezoidal, Gaussian, Bell, Sigmoid --
  correct at degenerate shoulders (`TriangularMF(0, 0, 5)` is 1.0 at `x = 0`) and finite
  under autograd everywhere.
- **T-Norms / S-Norms**: Min, Max, Product, Probabilistic Sum, Łukasiewicz.
- **Inference**: Mamdani (configurable implication and aggregation) and TSK of any order.
- **Defuzzification**: Centroid, Bisector, Mean of Maximum.
- **Differentiable**: end to end, over membership parameters and TSK coefficients alike.
- **GPU-ready**: `nn.Module` throughout; `.to(device)` moves the whole system.
- **Visualisation**: `.plot()` on sets and variables; `plot_inference` and `plot_surface` on systems.

## Watching simulations in a notebook

`record` runs one episode headlessly and returns a video that plays inline in Jupyter,
VS Code or Colab; `compare` shows several side by side. It renders with MuJoCo (3-D)
when an OpenGL context is available and falls back to pygame (2-D) otherwise.

```python
from fuzzy_torch.envs import RacingEnv, record, compare
from fuzzy_torch.envs.controllers import FuzzyRaceDriver

video = record(RacingEnv('chicane', render_mode='rgb_array'), FuzzyRaceDriver())
video                       # last line of a cell -> animated playback
video.info['lap_time']      # every info dict, observation and raw state is kept
video.save('lap.gif')       # or .webp
```

Make the environment with `render_mode='human'` instead and the same call opens a live
window. `examples/notebooks/02_fuzzy_control_envs.ipynb` uses this throughout.

## Quadrotor environment

The optional ``envs`` extra also includes a differentiable 3-D quadrotor.  Its
13-state rigid-body model uses world position and velocity, a scalar-first unit
quaternion, and body angular rates.  The default action is four physical rotor
thrusts; ``ControlMode.RATE_THRUST`` supplies a practical higher-level interface
of collective thrust plus roll, pitch, and yaw-rate requests.

```python
from fuzzy_torch.envs import ControlMode, QuadrotorPlant
from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv

drone = QuadrotorPlant(mode=ControlMode.MOTOR_THRUST)
env = FuzzyControlEnv(drone, render_mode='human')
observation, info = env.reset(seed=0)
hover = [float(drone.m * drone.g / 4)] * 4
observation, reward, terminated, truncated, info = env.step(hover)
```

MuJoCo draws an asset-free 3-D quadrotor when available; the pygame fallback is a
top-down flight view with altitude and target readouts.

Run the baseline demo to generate a tracking plot, or use the live 3-D viewer:

```bash
.venv/bin/python examples/drone_demo.py --out out/
.venv/bin/python examples/drone_demo.py --live --backend mujoco
```

## Fuzzy Grand Prix environment

The optional ``envs`` extra includes a closed-circuit Gymnasium time-trial for the
dynamic ``DiffDriveRobot`` plant.  It has three comparable lecture circuits:
``oval``, ``chicane`` and ``technical``.  A lap ends only at the finish line;
leaving the track is a DNF, so the reported ``lap_time`` cannot be improved by
cutting corners.

```python
from fuzzy_torch.envs import RacingEnv
from fuzzy_torch.envs.controllers import FuzzyRaceDriver

env = RacingEnv('chicane')
driver = FuzzyRaceDriver()
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(driver(obs))
    if terminated or truncated:
        break
print(info['lap_time'], info['off_track'])  # seconds, or None for a DNF
```

The observation is deliberately track-relative rather than a raw world pose:
``[cross_track_error, heading_error, speed, yaw_rate, lookahead_curvature, progress]``.
This gives a small, explainable fuzzy-controller input space and lets the same rule
base be evaluated on every circuit.  See ``examples/racing_demo.py`` for a complete
three-circuit run and plot.  Add ``--live`` to open the animated circuit window:

```bash
pip install -e '.[viz]'
python examples/racing_demo.py --live --backend mujoco
```

The MuJoCo view is a stylised 3-D kart, full road surface, alternating kerbs and a
follow camera.  The underlying vehicle dynamics remain the differentiable torch model,
so this is a visualisation of the same Gymnasium state that the controller receives.

For a lecture, [`examples/race_control/`](examples/race_control/README.md) runs a live
leaderboard: students submit a name plus a JSON driver from the browser, and every
entry is raced on all three circuits, ranked, and replayable.

### Racing in SuperTuxKart

[SuperTuxKart](https://supertuxkart.net), the open-source kart racer, runs as a
Gymnasium environment through its Python bindings (`pystk2`): real 3-D tracks, AI
opponents, items, drifting and nitro. `SuperTuxKartEnv` uses the same interface as
`RacingEnv`, with the same six features and the same `[throttle, steering]` action, so
any fuzzy driver, including a Race Control JSON driver, races in it unchanged:

```python
from fuzzy_torch.envs import SuperTuxKartEnv
from fuzzy_torch.envs.controllers import FuzzyRaceDriver

env = SuperTuxKartEnv('zengarden', opponents=3, laps=1)
driver = FuzzyRaceDriver()
obs, info = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, info = env.step(driver(obs))
    if terminated or truncated:
        break
print(info['race_time'], info['position'], '/', info['num_karts'])
```

The features are computed from the game state, which pystk2 exposes in full: the kart's
position, heading, velocity and yaw rate, and the track's centre line with its road
width. With the default `units='grand_prix'` they are rescaled to the lecture circuits'
scale: a cross-track error of ±1 is the road edge, and the kart's top speed reads 4.
Use `units='si'` for raw metres and seconds. The game reports more in `info`
(`position`, `lap`, `off_track`, `rescues`). `extra_actions=True` adds drift, nitro and
item buttons.

```bash
pip install -e '.[kart]'           # first start downloads the game assets (~700 MB)
python examples/supertuxkart_demo.py                                 # 3 tracks, headless
python examples/supertuxkart_demo.py my_driver.json --track lighthouse --live
python examples/supertuxkart_demo.py --gif out/race.gif
```

`examples/race_control/kart_starter.json` is a driver tuned for SuperTuxKart, and
[`kart_guide.md`](examples/race_control/kart_guide.md) is the guide for students.

pystk2 runs one game per process. It starts either headless or with graphics, and
holds one race at a time, so render in a fresh process (a notebook kernel included).

#### A class race, with the fuzzy logic on screen

`KartRace` puts several fuzzy drivers in one race, each in its own kart, and
`examples/kart_party.py` shows it on a projector. The window has the followed kart's
chase camera. Next to it, the driver's fuzzy systems update every step: each input
on its membership functions, the strongest rules and their firing strengths, and the
output set with the crisp action. A track map and live standings sit underneath.

```bash
# one driver against AI karts
python examples/kart_party.py --driver examples/race_control/example_driver.json --ai 3

# the whole class: each student's latest Race Control submission
python examples/kart_party.py --store examples/race_control/leaderboard.json --track zengarden

# or a folder of JSON drivers named after the students; --headless is a dry run
python examples/kart_party.py --folder submissions/ --headless
```

SuperTuxKart renders at most 8 chase cameras per race, so a larger class races in
heats of up to 8, and the 8 fastest overall meet in a final. In the window:
`Left`/`Right` (or a click on the standings) follows another driver, `L` follows
the leader, `Space` pauses, `Enter` starts the next race, and `Esc` quits. A driver
that raises an error coasts and is flagged, so one broken file cannot stop the race.

#### Students driving from their laptops

With `--join PORT`, students drive their own karts from their laptops. The race still
runs on your machine. Each laptop runs the student's driver and gets its kart's
observations over WebSockets. A laptop needs the **join code**, which is shown over
the race camera with the address, so students can read both from a screen share.

```bash
# your machine; the code is random unless you pass --code
python examples/kart_party.py --join 9000 --ai 2
# each laptop (needs only torch-fis[envs], not SuperTuxKart)
python examples/kart_join.py 192.168.0.10:9000 --code K7QX2 --name "Ana Souza" --driver my_driver.json
```

Students outside your network can join through an HTTP tunnel. WebSockets pass
through ngrok's free plan, which has no raw TCP:

```bash
ngrok http 9000                      # prints https://abcd-1234.ngrok-free.app
python examples/kart_party.py --join 9000 --public-url https://abcd-1234.ngrok-free.app
python examples/kart_join.py https://abcd-1234.ngrok-free.app --code K7QX2 --name "Ana Souza" --driver my_driver.json
```

The laptop window shows the student's own fuzzy systems firing as the kart races.
The driver file is read again before each race, so students can change their rules
between heats. A Python controller can join too:
`fuzzy_torch.envs.kart_net.join(server, name, code, controller).run()`.

The server never waits for a laptop. It uses the newest action each laptop has sent,
so a remote kart steers one step (50 ms) late plus the network round trip. Up to
about 200 ms of round trip, races are only slightly slower. A laptop silent for a
second coasts until it answers again. Because laptops run their own code, the server
cannot check that it is a fuzzy controller; for a graded race, use JSON submissions.

## Student fuzzy-control workspace

The three-case-study starter workspace lives in
[`examples/student/`](examples/student/README.md).  It fixes the simulation and
evaluation contracts while supplying neutral, trainable TSK fuzzy controllers for
the crane, kart, and quadrotor.  Students can therefore design rules and fine-tune
membership/consequent parameters without accidentally changing the benchmark.

## Upgrading from 0.1

**Existing systems will produce different numbers**, because 0.1 computed several things
incorrectly. The changes that move results:

- **Membership functions with a degenerate shoulder** (`a == b` or `b == c`) returned `0` at
  their flat endpoint instead of `1`. `TriangularMF(0, 0, 5)` was 0.0 at `x = 0`, and
  `TrapezoidalMF(7, 9, 10, 10)` was 0.0 at `x = 10`. Any rule base using shoulder sets at the
  ends of a range -- the common case -- was firing weakly or not at all there.
- A `+1e-6` guard in the denominators biased ordinary interior values as well
  (`TriangularMF(0, 0, 5)` gave `0.49999991` at the midpoint).
- **When no rule fires**, 0.1 returned `0.0`, which is not even inside a `[0, 30]` output
  range. It is now `NaN` by default -- pass `no_rule_value=` for the old-style fallback.
- `Bisector` was one grid step low on symmetric sets.
- `GaussianMF` guarded its division with `sigma + 1e-6`, which only protects sigma
  approaching zero from above and puts a singularity at `sigma = -1e-6` (NaN value *and*
  NaN gradient). It now uses `|sigma|` clamped off zero, so sigma crossing zero during
  training stays finite.

Also worth knowing:

- `pip install` from 0.1 shipped **no Python modules at all** (the package discovery config
  matched nothing), so any working 0.1 install was a source checkout.
- `FuzzyRule` now registers its variables as submodules, so `rule.parameters()`,
  `rule.to(device)` and `rule.state_dict()` work on a standalone rule.
- Scalar (0-dim) inputs used to raise `IndexError`; they now work and return a 0-dim output.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
