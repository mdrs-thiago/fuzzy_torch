# Fuzzy-control case studies

This folder is the student-editable boundary for the three environments.  Do not
change plant dynamics, rewards, or evaluation scenarios: put membership functions,
rules, and optional training code in `fuzzy_starters.py`.  That keeps results from
different rule bases comparable.

## Workflow

1. Start with one of the provided controllers.  Each is a trainable TSK fuzzy
   system with three Gaussian terms per input and zeroed consequents.
2. Give each rule a meaningful initial consequent or replace the grid with your
   own Mamdani rule base.  Keep the public input and output contract below.
3. Call `.train()` and optimise `controller.parameters()` with PyTorch if you
   want to fine-tune membership functions and TSK consequents.  Call `.eval()`
   for reporting; controllers return NumPy actions for the Gym environments.
4. Evaluate only on the named scenarios/tracks and report your seed, rule count,
   membership-function family, and the resulting metric.

## Contracts and suggested metrics

| Case | Student inputs | Required action | Primary report |
| --- | --- | --- | --- |
| Crane | `[x_error, x_error_dot, theta_error, theta_error_dot]` | trolley force `[N]` | IAE, residual sway, control effort |
| Kart | steering: `[cross_track_error, heading_error]`; speed: `[abs(curvature), speed]` | `[throttle, steering]` in `[-1, 1]` | completed-lap time; DNF otherwise |
| Drone | position/velocity errors, plus roll/pitch/yaw and body rates | `[collective_thrust, roll_rate, pitch_rate, yaw_rate]` | final 3-D position error and integrated position error |

`DroneStarter` adds hover thrust itself, so its vertical FIS returns a **thrust
correction**, not an absolute thrust.  `CraneStarter` consumes the output of
`ErrorFeatures`; `RaceStarter` consumes `RacingEnv`'s published observation;
`DroneStarter` consumes the raw state and target exposed by `FuzzyControlEnv`.

The examples already demonstrate the fixed environments:

```bash
.venv/bin/python examples/crane_demo.py --out out/
.venv/bin/python examples/racing_demo.py --track oval
.venv/bin/python examples/drone_demo.py --out out/
```

To use a starter, import it from an experiment or notebook after adding this
folder to `sys.path`:

```python
from fuzzy_starters import CraneStarter, RaceStarter, DroneStarter
controller = RaceStarter()
```

The starter systems deliberately do **not** hide a tuned solution.  They cover
the whole input space and produce safe neutral actions, leaving the actual fuzzy
design and fine-tuning as the coursework.
