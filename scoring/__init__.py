"""Pure raw-stats -> fantasy-points scoring, driven by a league's real settings.

Never hardcodes a scoring format (see AGENTS.md). Flow:

    settings payload --adapters--> ScoringRules --.
                                                  |--> score(stats, rules) -> points
    box score --extract--> canonical stat dict --'

The same ``score`` call produces model-training labels and application-layer
projections -- one implementation, no drift.
"""

from __future__ import annotations

from .adapters import normalize_sleeper, normalize_yahoo
from .rules import ScoringRules, Tier, YardageBonus, score
from . import stat_keys

__all__ = [
    "score",
    "ScoringRules",
    "YardageBonus",
    "Tier",
    "normalize_sleeper",
    "normalize_yahoo",
    "stat_keys",
]
