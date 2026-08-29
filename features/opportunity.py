"""Priority-1 features: usage / opportunity, aggregated over a window.

Opportunity (targets, carries, snaps, shares) predicts future fantasy production
far better than efficiency (yards per touch), which regresses hard year over year.
Efficiency columns are still emitted — cheap, and the model can weight them down.

All inputs here are **outcome data**: they come only from ``visible`` rows, which
``visible_weeks`` has already restricted to weeks strictly before the as-of point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# player-week stat -> (window total column, per-game column)
_COUNTING = {
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "carries": "carries",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "attempts": "pass_att",
    "completions": "pass_cmp",
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "interceptions": "pass_int",
    "receiving_air_yards": "rec_air_yd",
}


def _safe_div(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


def opportunity_features(visible: pd.DataFrame, player_col: str = "player_id") -> pd.DataFrame:
    """One row per player: usage totals, per-game rates, and team shares over the
    window in ``visible`` (output of ``features.visible_weeks`` on player_week_stats).
    """
    df = visible.copy()
    for col in _COUNTING:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)

    # team totals in the exact (season, week, team) cells the player appeared in,
    # so shares stay correct across mid-window team changes.
    team = (
        df.groupby(["season", "week", "team"], dropna=False)[["attempts", "carries", "targets", "receiving_air_yards"]]
        .sum()
        .rename(columns={
            "attempts": "team_pass_att", "carries": "team_carries",
            "targets": "team_targets", "receiving_air_yards": "team_air_yards",
        })
        .reset_index()
    )
    df = df.merge(team, on=["season", "week", "team"], how="left")
    df["w_target_share"] = _safe_div(df["targets"], df["team_targets"])
    df["w_rush_share"] = _safe_div(df["carries"], df["team_carries"])
    df["w_air_yards_share"] = _safe_div(df["receiving_air_yards"], df["team_air_yards"])

    grp = df.groupby(player_col, sort=False)
    out = pd.DataFrame(index=grp.size().index)
    out["games"] = grp.size()

    for src, name in _COUNTING.items():
        total = grp[src].sum()
        out[name] = total
        out[f"{name}_pg"] = total / out["games"]

    for w in ("w_target_share", "w_rush_share", "w_air_yards_share"):
        out[w.removeprefix("w_")] = grp[w].mean()

    # WOPR (weighted opportunity rating) — standard opportunity composite
    out["wopr"] = 1.5 * out["target_share"].fillna(0) + 0.7 * out["air_yards_share"].fillna(0)

    # efficiency (regresses; kept for the model to weigh, not to lead on)
    out["yards_per_target"] = _safe_div(out["rec_yd"], out["targets"])
    out["yards_per_carry"] = _safe_div(out["rush_yd"], out["carries"])
    out["yards_per_att"] = _safe_div(out["pass_yd"], out["pass_att"])
    out["catch_rate"] = _safe_div(out["receptions"], out["targets"])

    out["most_recent_team"] = grp["team"].last()
    out["most_recent_pos"] = grp["position"].last() if "position" in df.columns else np.nan

    return out.reset_index().rename(columns={"index": player_col})


def snap_features(
    visible_snaps: pd.DataFrame, player_col: str = "gsis_id"
) -> pd.DataFrame:
    """Snap-share aggregates over the window. ``visible_snaps`` is
    ``features.visible_weeks(snap_counts, ..., season_type_col="game_type")``.
    """
    df = visible_snaps.copy()
    df = df[df[player_col].notna()]
    for c in ("offense_pct", "offense_snaps", "st_pct"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    grp = df.groupby(player_col, sort=False)
    out = pd.DataFrame(index=grp.size().index)
    out["snap_games"] = grp["offense_pct"].count()
    out["off_snap_pct_mean"] = grp["offense_pct"].mean()
    out["off_snap_pct_last"] = grp["offense_pct"].last()
    out["off_snap_pct_max"] = grp["offense_pct"].max()
    out["st_snap_pct_mean"] = grp["st_pct"].mean()
    return out.reset_index().rename(columns={"index": player_col})
