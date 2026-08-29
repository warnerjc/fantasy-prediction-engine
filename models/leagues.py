"""Load a league's ScoringRules from specifications/league-configs/.

Sprint leagues that need a draft ranking (see specifications/draft-sprint-plan.md):
``sleeper`` -> Sleeper 1356741521163968513 (live scoring captured to a config)
``yahoo``   -> Yahoo 236625 "keepitcrooked" (hand-captured scoring)
"""

from __future__ import annotations

import json
from pathlib import Path

from scoring import ScoringRules, normalize_sleeper, normalize_yahoo

CONFIG_DIR = Path(__file__).resolve().parents[1] / "specifications" / "league-configs"

LEAGUES = {
    "sleeper": ("sleeper-1356741521163968513-scoring.json", normalize_sleeper, "scoring_settings"),
    "yahoo": ("yahoo-236625-scoring.json", normalize_yahoo, None),
}


def load_rules(league: str) -> ScoringRules:
    if league not in LEAGUES:
        raise KeyError(f"unknown league {league!r}; have {list(LEAGUES)}")
    fname, normalize, subkey = LEAGUES[league]
    cfg = json.loads((CONFIG_DIR / fname).read_text())
    return normalize(cfg[subkey] if subkey else cfg)
