"""Turn source box-score rows into canonical stat dicts.

The scoring function needs stats in the canonical vocabulary; nflverse gives us
its own column names. This module is the stat-side mirror of ``scoring.adapters``
(which does the same job for league settings).

Offense comes from nflreadpy's ``load_player_stats``; kicking and team-defense
lines come from the ``kicking_stats`` / ``team_defense_stats`` tables that
``/data`` derives from ``load_player_stats`` (K rows) and ``load_team_stats``.
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
    """Map a ``player_week_stats`` frame to canonical stat columns.

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
    """One canonical-stats row (from any extractor here) as a plain dict."""
    return {k: float(v) for k, v in row.items() if k in K.ALL_KEYS and pd.notna(v)}


# data.kicking_stats column -> canonical key (source cols are already bucketed)
_KICKING = {
    "fg_made": K.K_FG_MADE, "fg_missed": K.K_FG_MISSED, "fg_made_yds": K.K_FG_MADE_YDS,
    "fg_made_0_19": K.K_FG_MADE_0_19, "fg_made_20_29": K.K_FG_MADE_20_29,
    "fg_made_30_39": K.K_FG_MADE_30_39, "fg_made_40_49": K.K_FG_MADE_40_49,
    "fg_made_50p": K.K_FG_MADE_50P,
    "fg_missed_0_19": K.K_FG_MISSED_0_19, "fg_missed_20_29": K.K_FG_MISSED_20_29,
    "fg_missed_30_39": K.K_FG_MISSED_30_39, "fg_missed_40_49": K.K_FG_MISSED_40_49,
    "fg_missed_50p": K.K_FG_MISSED_50P,
    "xp_made": K.K_XP_MADE, "xp_missed": K.K_XP_MISSED,
}

# data.team_defense_stats already uses canonical dst_* names; just carry them through
_DST = {c: c for c in (
    K.DST_SACK, K.DST_INT, K.DST_FUM_REC, K.DST_SAFETY, K.DST_TD, K.DST_BLK_KICK,
    K.DST_PTS_ALLOWED, K.DST_YDS_ALLOWED,
)}


def canonical_kicking_stats(kicking: pd.DataFrame) -> pd.DataFrame:
    """`data.kicking_stats` rows -> canonical stat columns, keyed by
    `player_id` (= `kicker_player_id`), `season`, `week`."""
    out = kicking[[c for c in ("kicker_player_id", "season", "week", "game_type", "team")
                   if c in kicking.columns]].rename(columns={"kicker_player_id": "player_id"})
    for src, canonical in _KICKING.items():
        out[canonical] = pd.to_numeric(kicking.get(src), errors="coerce").fillna(0) \
            if src in kicking.columns else 0
    return out


def canonical_dst_stats(team_defense: pd.DataFrame) -> pd.DataFrame:
    """`data.team_defense_stats` rows -> canonical stat columns, keyed by
    `team` (= `defense_team`), `season`, `week`."""
    out = team_defense[[c for c in ("defense_team", "season", "week", "game_type")
                        if c in team_defense.columns]].rename(columns={"defense_team": "team"})
    for src, canonical in _DST.items():
        out[canonical] = pd.to_numeric(team_defense.get(src), errors="coerce").fillna(0) \
            if src in team_defense.columns else 0
    return out
