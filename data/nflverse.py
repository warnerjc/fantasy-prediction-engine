"""nflverse pulls, each returned at the grain its target table expects.

Source cadence: nflverse regenerates the current season's releases within a day
of each game and applies stat corrections for ~2 weeks after. Historical seasons
are stable. All pulls here are therefore safe to re-run; ``data.db.upsert``
handles the overwrite.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import nfl_data_py as nfl
import pandas as pd

# columns from import_weekly_data we deliberately do NOT land in player_week_stats
_WEEKLY_DROP = {"player_name", "headshot_url", "position_group"}
# ...everything else is kept: raw counts + nflverse-provided rates (target_share,
# air_yards_share, wopr, racr, pacr, dakota, *_epa). Rates are cheap to store and
# a feature may want them; recomputing target_share etc. is a feature concern.


def _seasons(seasons: Iterable[int]) -> list[int]:
    return sorted({int(s) for s in seasons})


def weekly_player_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Offensive box-score production, one row per (player, season, week, season_type).

    This is the feed for `player_week_stats` — the source of truth. `recent_team`
    is renamed `team`; `opponent_team` -> `opponent`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = nfl.import_weekly_data(_seasons(seasons), downcast=True)

    df = df.drop(columns=[c for c in _WEEKLY_DROP if c in df.columns])
    df = df.rename(columns={"recent_team": "team", "opponent_team": "opponent"})
    df["season_type"] = df["season_type"].fillna("REG")
    return df.reset_index(drop=True)


def snap_counts(seasons: Iterable[int], crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly snap counts/pcts. Keyed by `pfr_player_id`; `gsis_id` attached via
    the crosswalk so downstream joins to `player_week_stats` are ID-based, not
    name-based."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = nfl.import_snap_counts(_seasons(seasons))

    if crosswalk is None:
        crosswalk = player_ids()
    xwalk = (
        crosswalk[["pfr_id", "gsis_id"]]
        .dropna()
        .drop_duplicates("pfr_id", keep=False)  # drop pfr ids that map to >1 gsis id
        .rename(columns={"pfr_id": "pfr_player_id"})
    )
    df = df.merge(xwalk, on="pfr_player_id", how="left")
    if df.duplicated(["pfr_player_id", "season", "week", "game_type", "team"]).any():
        raise AssertionError("snap_counts still fanned out after 1:1 crosswalk merge")
    return df.reset_index(drop=True)


def injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly injury reports, keyed by `gsis_id` (matches player_week_stats.player_id)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = nfl.import_injuries(_seasons(seasons))
    df["game_type"] = df["game_type"].fillna("REG")
    # A report gets re-issued within a week (Questionable -> Out); keep the latest.
    pk = ["gsis_id", "season", "week", "game_type", "team"]
    df = (
        df.sort_values("date_modified")
        .drop_duplicates(pk, keep="last")
        .reset_index(drop=True)
    )
    return df


def schedules(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw game rows — one per game. Carries rest days, roof/surface/weather, and
    the closing Vegas spread/total used to derive team implied points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = nfl.import_schedules(_seasons(seasons))
    return df.reset_index(drop=True)


def team_week(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Explode `schedules` to one row per (season, week, game_type, team) with the
    pre-game-known context a feature can use without leakage: opponent, home/away,
    rest days, div game, roof/surface/weather, and the Vegas implied team total.

    Implied total: `total_line/2` split by `spread_line` (nflverse convention:
    positive spread_line = home favored), so favored teams get the higher total.
    """
    s = schedules_df
    common = s[["game_id", "season", "week", "game_type", "gameday", "weekday",
                "gametime", "roof", "surface", "temp", "wind", "div_game",
                "spread_line", "total_line"]].copy()

    half_total = common["total_line"] / 2
    half_spread = common["spread_line"] / 2

    home = common.assign(
        team=s["home_team"], opponent=s["away_team"], is_home=1,
        rest=s["home_rest"], implied_total=half_total + half_spread,
        team_spread=-s["spread_line"],
    )
    away = common.assign(
        team=s["away_team"], opponent=s["home_team"], is_home=0,
        rest=s["away_rest"], implied_total=half_total - half_spread,
        team_spread=s["spread_line"],
    )
    out = pd.concat([home, away], ignore_index=True)
    out["game_type"] = out["game_type"].fillna("REG")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def player_ids() -> pd.DataFrame:
    """Cross-reference table: gsis_id <-> pfr_id <-> sleeper_id <-> yahoo_id <-> espn_id.
    From nflverse's maintained `import_ids` release — do not rebuild from names."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = nfl.import_ids()

    keep = ["gsis_id", "pfr_id", "sleeper_id", "yahoo_id", "espn_id", "mfl_id",
            "sportradar_id", "name", "merge_name", "position", "team", "birthdate",
            "draft_year", "draft_round", "draft_pick", "draft_ovr"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"])
    for c in ("sleeper_id", "yahoo_id", "espn_id", "mfl_id"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df.reset_index(drop=True)
