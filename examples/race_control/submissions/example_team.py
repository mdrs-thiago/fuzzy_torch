"""Copy this file, rename it, and tune the fuzzy systems in your own controller."""
from fuzzy_torch.envs.controllers import FuzzyRaceDriver

STUDENT_ID = "DEMO-01"


def build_controller():
    return FuzzyRaceDriver()


def describe(controller):
    """Optional: data shown when a spectator opens your kart tile."""
    return {
        "title": "DEMO-01 · Two Mamdani systems",
        "systems": {
            "steering": {"purpose": "cross-track + heading → steering", "rules": 9},
            "throttle": {"purpose": "curvature + speed → throttle", "rules": 9},
        },
    }
