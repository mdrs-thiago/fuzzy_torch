# Fuzzy Grand Prix: Race Control

A live leaderboard for the kart time trial. Students open a web page, type their
name, paste a **JSON fuzzy driver**, and watch it race the three circuits (`oval`,
`chicane`, `technical`). Entries are ranked by total lap time. Each tile shows the
racing lines, and clicking one opens a replay against the leader's ghost, the
membership functions and rule table, and the JSON itself.

Above the grid, an **Onboard** wall has one screen per circuit that loops the
lap-record holder's lap from a chase camera, in a 3-D scene matching the MuJoCo
one. The replay dialog has the same onboard view next to the map. It is drawn in
the browser with three.js, fetched from jsDelivr. Without internet access the
screens fall back to a moving dot on the circuit map.

```bash
pip install -e '.[envs]'
python examples/race_control/server.py            # http://127.0.0.1:8080
```

Use a fresh results file per class with `--store results/2026-10-16.json`.

## Sharing it with the class

Submissions are data, not code: a driver can only choose membership functions,
rules and operators from a fixed vocabulary, with size limits (125 rules, 3 inputs
and 7 terms per system). It is therefore safe to put the page on the internet:

```bash
ngrok http 8080            # or: python server.py --host 0.0.0.0 for the classroom LAN
```

Share the ngrok URL as-is. At start-up the server also prints an **instructor
URL** (`/?key=…`); opened with that key, each entry's JSON tab has a *Delete entry*
button for removing joke names or test runs. `--control-key` fixes the key.

An entry takes roughly 10 s to race; `--workers N` races N entries in parallel
(default: half your cores, at most 4). A kart that stops making
progress for 4 s is retired as *Stalled*, so a broken driver cannot hold the queue
for the full 45 s limit. Each name may have one run waiting at a time; the board
shows every name's best run.

## Before submitting: check locally

```bash
python examples/race_control/grand_prix.py my_driver.json
python examples/race_control/grand_prix.py my_driver.json --live --track chicane   # 3-D view
```

This is the exact evaluation the server runs, so the times match. `--live` is also
the way to show any leaderboard entry in MuJoCo: download its JSON from the entry
dialog and pass it here.

## The driver format

A driver is two fuzzy systems. `throttle` outputs the throttle (negative values
brake to a stop; there is no reverse), and `steering` outputs the steering request.
Both outputs are clipped to `[-1, 1]`. `example_driver.json` is the baseline driver
(56.3 s total), and it is the place to start.

```json
{
  "description": "optional one-liner shown on the leaderboard",
  "throttle": { ...fuzzy system... },
  "steering": { ...fuzzy system... }
}
```

### Inputs

Each system picks up to three of these, by name:

| name | meaning | typical range |
| --- | --- | --- |
| `cross_track_error` | metres from the centre line, positive = kart is left of it | ±1 (off track beyond) |
| `heading_error` | track direction minus kart heading, radians | ±π |
| `speed` | forward speed, m/s | 0 – 4 |
| `yaw_rate` | turning rate, rad/s | ±3.2 |
| `lookahead_curvature` | signed curvature 1.8 m ahead, 1/m (positive = left bend) | ±1.5 |
| `curve` | `abs(lookahead_curvature)`: how sharp the next bend is | 0 – 1.5 |
| `progress` | fraction of the lap completed | 0 – 1 |

An input beyond its `range` is held at the nearest end of the range, so a kart
pushed far off the road keeps firing its outermost terms and steers back.

### A fuzzy system

```json
{
  "inputs": {
    "cross_track_error": {
      "range": [-1, 1],
      "terms": {
        "right":  ["trap", -1, -1, -0.28, 0],
        "centre": ["tri", -0.28, 0, 0.28],
        "left":   ["trap", 0, 0.28, 1, 1]
      }
    }
  },
  "output": {
    "range": [-1, 1],
    "terms": { "right": ["tri", -1, -1, 0], "none": ["tri", -1, 0, 1], "left": ["tri", 0, 1, 1] }
  },
  "rules": [
    {"if": {"cross_track_error": "right"},  "then": "left"},
    {"if": {"cross_track_error": "centre"}, "then": "none"},
    {"if": {"cross_track_error": "left"},   "then": "right"}
  ]
}
```

**Membership functions** are `[shape, parameters…]`:

| shape | parameters | |
| --- | --- | --- |
| `tri` | a, b, c | triangle, peak at b; `a == b` makes a left shoulder |
| `trap` | a, b, c, d | trapezoid, flat top from b to c |
| `gauss` | mu, sigma | Gaussian |
| `bell` | a, b, c | generalised bell: width a, slope b, centre c |
| `sigmoid` | b, c | slope b (negative = falling), centre c |

**Rules** AND together every condition in `if`; a rule may name only some of the
inputs. What you write in `then` selects the inference method:

- a term name, `"then": "left"`: **Mamdani**. The output needs `terms`.
- a number, `"then": 0.8`: **zero-order TSK (Sugeno)**. The output needs only a `range`.
- a list, `"then": [c0, c1, c2]`: **first-order TSK**, `c0 + c1·x1 + c2·x2` in raw
  input units, with the inputs in the order they are listed.

All the rules in one system must use the same kind of `then`. The two systems may
differ, for example TSK steering with Mamdani throttle.

**Optional settings:** `"and": "min" | "product"`, `"implication": "min" | "product"`,
`"aggregation": "max" | "probsum"`, `"defuzz": "centroid" | "bisector" | "mom"`, and
`"default"`, the output when no rule fires (0 if omitted).

Mistakes are reported with the path to the problem, for example
`steering.rules[2].if.heading_error: unknown term 'leftt' (available: right, centre, left)`.

## Race day in SuperTuxKart

The same JSON drivers race each other in SuperTuxKart, the open-source kart game,
with the fuzzy logic of the followed kart drawn live next to the race:

```bash
pip install -e '.[kart]'
python examples/kart_party.py --store examples/race_control/leaderboard.json
```

Students start from `kart_starter.json` and read [`kart_guide.md`](kart_guide.md).
See "Racing in SuperTuxKart" in the main README.

## From PyTorch to the leaderboard

Every fuzzy system in `fuzzy_torch` is differentiable, so a driver can be tuned by
gradient descent and then exported to this format:

```python
import json
from fuzzy_torch import system_to_config
# ...train `steering` and `throttle` (FuzzyModule objects) however you like...
json.dump({'throttle': system_to_config(throttle),
           'steering': system_to_config(steering)}, open('my_driver.json', 'w'), indent=2)
```

`fuzzy_torch.build_system(spec)` goes the other way: it turns a spec, including a
leaderboard entry, back into a trainable `FuzzyModule`.

## API

| method | path | |
| --- | --- | --- |
| GET | `/api/leaderboard` | standings, records, queue (ETag-cached; the page polls it) |
| GET | `/api/tracks` | circuit centre lines and the replay sample interval |
| GET | `/api/example` | `example_driver.json` |
| POST | `/api/submit` | `{"name": "...", "spec": {...}}` → `{"id", "position"}`, or 422 with the error |
| GET | `/api/entries/<id>` | status, lap results, paths and spec of one run |
| DELETE | `/api/entries/<id>?key=…` | instructor only |
