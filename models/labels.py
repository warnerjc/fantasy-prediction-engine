"""Season labels: fantasy points and PPG per player/team/season, via /scoring.

Labels are league-specific — pass the ``ScoringRules`` for the league being
drafted. Features are league-agnostic (usage), so a per-league model is just a
relabel + retrain (seconds). The *same* ``scoring.score`` used here is used by
the application layer — one implementation, no drift.
"""

from __future__ import annotations

import pandas as pd

from scoring import ScoringRules, score
from scoring.extract import (
    canonical_dst_stats,
    canonical_kicking_stats,
    canonical_offense_stats,
    stats_dict,
)

OFFENSE = ("QB", "RB", "WR", "TE")


def _drop_final_week(df: pd.DataFrame, week_col: str = "week") -> pd.DataFrame:
    """Drop the last REG week of each season -- played after every fantasy
    championship by locked-seed teams resting starters, so it's fantasy-dead and
    statistically distorted. Final week is 17 for 2015-2020, 18 for 2021 on."""
    if df.empty:
        return df
    final = df.groupby("season")[week_col].transform("max")
    return df[df[week_col] < final]


def _score_rows(canonical: pd.DataFrame, rules: ScoringRules, position_col: str | None) -> pd.Series:
    pos = canonical[position_col] if position_col and position_col in canonical else None
    return pd.Series(
        [
            score(stats_dict(row), rules, position=(pos.iloc[i] if pos is not None else None))
            for i, row in enumerate(canonical.to_dict("records"))
        ],
        index=canonical.index,
    )


def season_labels(
    pws: pd.DataFrame,
    kicking_stats: pd.DataFrame,
    team_defense_stats: pd.DataFrame,
    rules: ScoringRules,
    season_types: tuple[str, ...] = ("REG",),
    drop_final_week: bool = False,
) -> pd.DataFrame:
    """One row per (player_id, season, position): games, fantasy_points, ppg.
    For DEF, ``player_id`` is the team abbreviation.

    ``drop_final_week`` excludes each season's last REG week (fantasy-dead,
    resting starters) -- the v1 draft-projection caller opts in; see
    ``_drop_final_week``.
    """
    frames = []

    off = pws[pws["season_type"].isin(season_types)].copy()
    if drop_final_week:
        off = _drop_final_week(off)
    can = canonical_offense_stats(off)
    can["fp"] = _score_rows(can, rules, "position")
    can = can[can["position"].isin(OFFENSE)]
    frames.append(
        can.groupby(["player_id", "season", "position"], as_index=False)
        .agg(games=("week", "nunique"), fantasy_points=("fp", "sum"))
    )

    kick = kicking_stats[kicking_stats["game_type"].isin(season_types)].copy()
    if drop_final_week:
        kick = _drop_final_week(kick)
    if not kick.empty:
        ck = canonical_kicking_stats(kick)
        ck["fp"] = _score_rows(ck, rules, None)
        g = ck.groupby(["player_id", "season"], as_index=False).agg(
            games=("week", "nunique"), fantasy_points=("fp", "sum")
        )
        g["position"] = "K"
        frames.append(g)

    dst = team_defense_stats[team_defense_stats["game_type"].isin(season_types)].copy()
    if drop_final_week:
        dst = _drop_final_week(dst)
    if not dst.empty:
        cd = canonical_dst_stats(dst)
        cd["fp"] = _score_rows(cd, rules, None)
        g = cd.groupby(["team", "season"], as_index=False).agg(
            games=("week", "nunique"), fantasy_points=("fp", "sum")
        )
        g = g.rename(columns={"team": "player_id"})
        g["position"] = "DEF"
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    out["ppg"] = out["fantasy_points"] / out["games"]
    return out


def week_labels(
    pws: pd.DataFrame,
    kicking_stats: pd.DataFrame,
    team_defense_stats: pd.DataFrame,
    rules: ScoringRules,
    season_types: tuple[str, ...] = ("REG",),
) -> pd.DataFrame:
    """One row per (player_id, season, week, position): ``week_points``.

    The weekly-grain counterpart of ``season_labels`` -- same scoring path
    (``scoring.score`` via ``_score_rows``), no aggregation to a season and no
    division by games. For DEF, ``player_id`` is the team abbreviation. Every
    game is a prediction target (no ``drop_final_week``); the v2 weekly windows
    do not drop it either.
    """
    frames = []

    off = pws[pws["season_type"].isin(season_types)].copy()
    can = canonical_offense_stats(off)
    can["fp"] = _score_rows(can, rules, "position")
    can = can[can["position"].isin(OFFENSE)]
    frames.append(
        can.groupby(["player_id", "season", "week", "position"], as_index=False)
        .agg(week_points=("fp", "sum"))
    )

    kick = kicking_stats[kicking_stats["game_type"].isin(season_types)].copy()
    if not kick.empty:
        ck = canonical_kicking_stats(kick)
        ck["fp"] = _score_rows(ck, rules, None)
        g = ck.groupby(["player_id", "season", "week"], as_index=False).agg(
            week_points=("fp", "sum")
        )
        g["position"] = "K"
        frames.append(g)

    dst = team_defense_stats[team_defense_stats["game_type"].isin(season_types)].copy()
    if not dst.empty:
        cd = canonical_dst_stats(dst)
        cd["fp"] = _score_rows(cd, rules, None)
        g = cd.groupby(["team", "season", "week"], as_index=False).agg(
            week_points=("fp", "sum")
        )
        g = g.rename(columns={"team": "player_id"})
        g["position"] = "DEF"
        frames.append(g)

    return pd.concat(frames, ignore_index=True)
