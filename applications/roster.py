"""Parse a league's roster settings into starter slots and per-position
replacement ranks — the baseline for value-over-replacement.

Replacement rank for a position = ``teams × (dedicated starters + expected share
of flex slots)``. The player projected at that rank is "freely available", so a
player's draft value is ``projected_points − replacement_points[position]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# how a flex slot's usage splits across eligible positions (empirical-ish)
_FLEX_SPLIT = {
    frozenset({"RB", "WR", "TE"}): {"RB": 0.40, "WR": 0.45, "TE": 0.15},
    frozenset({"RB", "WR"}): {"RB": 0.45, "WR": 0.55},
    frozenset({"WR", "TE"}): {"WR": 0.70, "TE": 0.30},
    frozenset({"QB", "RB", "WR", "TE"}): {"QB": 0.10, "RB": 0.36, "WR": 0.40, "TE": 0.14},
}

# Sleeper roster_positions tokens / Yahoo config keys -> flex eligibility
_FLEX_TOKENS = {
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "REC_FLEX": {"WR", "TE"},
    "FLEX_WRT": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "SUPERFLEX": {"QB", "RB", "WR", "TE"},
}
_SKIP = {"BN", "IR", "TAXI"}
SCORABLE = ("QB", "RB", "WR", "TE", "K", "DEF")

# "last starter" as the replacement baseline overvalues positions that are cheap
# to replace off waivers (QB/TE much more so than RB/WR). Push their baseline
# deeper by this factor — standard VBD practice without ADP data.
_BASELINE_MULT = {"QB": 1.6, "RB": 1.0, "WR": 1.0, "TE": 1.3, "K": 1.0, "DEF": 1.0}


@dataclass(frozen=True)
class RosterSpec:
    teams: int
    dedicated: dict[str, int]                 # position -> dedicated starter slots
    flex: list[frozenset[str]] = field(default_factory=list)   # one entry per flex slot

    def flex_allocation(self) -> dict[str, float]:
        alloc: dict[str, float] = {p: 0.0 for p in SCORABLE}
        for elig in self.flex:
            split = _FLEX_SPLIT.get(frozenset(elig))
            if split is None:                # even split fallback
                split = {p: 1 / len(elig) for p in elig}
            for pos, frac in split.items():
                alloc[pos] += frac
        return alloc

    def replacement_rank(self) -> dict[str, int]:
        alloc = self.flex_allocation()
        ranks = {}
        for pos in SCORABLE:
            slots = self.dedicated.get(pos, 0) + alloc.get(pos, 0.0)
            ranks[pos] = max(1, math.ceil(self.teams * slots * _BASELINE_MULT.get(pos, 1.0)))
        return ranks


def _positions_from_sleeper(tokens: list[str]) -> tuple[dict[str, int], list[frozenset[str]]]:
    dedicated: dict[str, int] = {}
    flex: list[frozenset[str]] = []
    for tok in tokens:
        if tok in _SKIP:
            continue
        if tok in _FLEX_TOKENS:
            flex.append(frozenset(_FLEX_TOKENS[tok]))
        else:
            pos = "DEF" if tok in ("DEF", "DST") else tok
            dedicated[pos] = dedicated.get(pos, 0) + 1
    return dedicated, flex


def roster_spec(league_config: dict) -> RosterSpec:
    """``league_config`` is a parsed specifications/league-configs/*.json."""
    teams = league_config.get("teams") or league_config.get("total_rosters") or 12
    rp = league_config.get("roster_positions")

    if isinstance(rp, list):                      # Sleeper shape
        dedicated, flex = _positions_from_sleeper(rp)
    elif isinstance(rp, dict):                     # Yahoo config shape
        dedicated, flex = {}, []
        for tok, count in rp.items():
            if tok in _SKIP:
                continue
            if tok in _FLEX_TOKENS:
                flex.extend([frozenset(_FLEX_TOKENS[tok])] * int(count))
            else:
                pos = "DEF" if tok in ("DEF", "DST") else tok
                dedicated[pos] = dedicated.get(pos, 0) + int(count)
    else:
        raise ValueError("league_config has no roster_positions")

    return RosterSpec(teams=int(teams), dedicated=dedicated, flex=flex)
