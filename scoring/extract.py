"""Turn source box-score rows into canonical stat dicts.

The scoring function needs stats in the canonical vocabulary; nflverse gives us
its own column names. This module is the stat-side mirror of ``scoring.adapters``
(which does the same job for league settings).

Offense (QB/RB/WR/TE) is covered from ``nfl.import_weekly_data``. Kicking and DST
inputs are *not* in that release -- they require play-by-play aggregation and are
landed by the ``/data`` pipeline; ``kicking_stats`` / ``dst_stats`` are declared
here as the target shape so callers can code against them now.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from . import stat_keys as K

# nflverse weekly_data column -> canonical key, for the simple 1:1 stats.
_NFLVERSE_WEEKLY = {
    "completions": K.PASS_CMP,
    "attempts": K.PASS_ATT,
    "passing_yards": K.PASS_YD,
    "passing_tds": K.PASS_TD,
    "interceptions": K.PASS_INT,
    "passing_2pt_conversions": K.PASS_2PT,
    "sacks": K.PASS_SACK,
    "carries": K.RUSH_ATT,
    "rushing_yards": K.RUSH_YD,
    "rushing_tds": K.RUSH_TD,
    "rushing_2pt_conversions": K.RUSH_2PT,
    "receptions": K.REC,
    "targets": K.REC_TGT,
    "receiving_yards": K.REC_YD,
    "receiving_tds": K.REC_TD,
    "receiving_2pt_conversions": K.REC_2PT,
    "special_teams_tds": K.RET_TD,
}

_FUMBLE_LOST_COLS = ("rushing_fumbles_lost", "sack_fumbles_lost", "receiving_fumbles_lost")


def canonical_offense_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Map an ``import_weekly_data`` frame to canonical stat columns.

    Returns a new frame with the identifier columns (``player_id``, ``season``,
    ``week``, ``position``, ``recent_team``, ``opponent_team`` when present) plus
    one column per canonical stat. Missing source columns are treated as 0.
    """
    out = pd.DataFrame(index=weekly.index)
    for id_col in ("player_id", "player_display_name", "position", "season", "week",
                   "recent_team", "opponent_team"):
        if id_col in weekly.columns:
            out[id_col] = weekly[id_col]

    for src, canonical in _NFLVERSE_WEEKLY.items():
        out[canonical] = pd.to_numeric(weekly.get(src), errors="coerce").fillna(0) \
            if src in weekly.columns else 0

    fum = sum(
        pd.to_numeric(weekly[c], errors="coerce").fillna(0)
        for c in _FUMBLE_LOST_COLS if c in weekly.columns
    )
    out[K.FUM_LOST] = fum if not isinstance(fum, int) else 0

    # platforms that don't split 2pt by play type read a single combined stat
    out[K.TWO_PT] = out[K.PASS_2PT] + out[K.RUSH_2PT] + out[K.REC_2PT]
    return out


def stats_dict(row: Mapping[str, Any]) -> dict[str, float]:
    """One canonical-stats row (from ``canonical_offense_stats``) as a plain dict."""
    return {k: float(v) for k, v in row.items() if k in K.ALL_KEYS and pd.notna(v)}
