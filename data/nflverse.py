"""nflverse pulls (via ``nflreadpy``), each returned at the grain its target
table expects.

``nflreadpy`` is the maintained nflverse Python client — unlike the older
``nfl_data_py`` it serves the current season's ``player_stats`` / ``team_stats``
releases (which now also carry kicker FG-by-distance and team-defense box scores
natively, so no play-by-play aggregation is needed). It returns polars frames;
everything here converts to pandas at the boundary.

Source cadence: nflverse regenerates the current season's releases within a day
of each game and applies stat corrections for ~2 weeks after. Historical seasons
are stable. All pulls here are safe to re-run; ``data.db.upsert`` handles the
overwrite.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import nflreadpy as nr
import numpy as np
import pandas as pd

# player_stats columns we deliberately do NOT land in player_week_stats
_WEEKLY_DROP = {"player_name", "headshot_url", "position_group"}
_OFFENSE_POS = {"QB", "RB", "WR", "TE", "FB"}


def _seasons(seasons: Iterable[int]) -> list[int]:
    return sorted({int(s) for s in seasons})


def _load(loader: str, seasons: Iterable[int] | None = None) -> pd.DataFrame:
    fn = getattr(nr, loader)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(seasons=_seasons(seasons)) if seasons is not None else fn()
    return out.to_pandas() if hasattr(out, "to_pandas") else out


def player_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw `load_player_stats` (all positions + kickers). Pulled once per build and
    fed to `weekly_player_stats` and `kicking_stats`."""
    return _load("load_player_stats", seasons)


def team_stats(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw `load_team_stats` — team-week box score incl. `def_*` for DST."""
    return _load("load_team_stats", seasons)


def weekly_player_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Offensive box-score production, one row per (player, season, week, season_type).

    Feed for `player_week_stats` — the source of truth. Renamed to keep the
    historical column names: `opponent_team`->`opponent`,
    `passing_interceptions`->`interceptions`, `sacks_suffered`->`sacks`.
    """
    df = player_stats_df[player_stats_df["position"].isin(_OFFENSE_POS)].copy()
    df = df.drop(columns=[c for c in _WEEKLY_DROP if c in df.columns])
    df = df.rename(columns={
        "opponent_team": "opponent",
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
    })
    df["season_type"] = df["season_type"].fillna("REG")
    return df.reset_index(drop=True)


def snap_counts(seasons: Iterable[int], crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly snap counts/pcts. Keyed by `pfr_player_id`; `gsis_id` attached via
    the crosswalk so downstream joins are ID-based, not name-based."""
    df = _load("load_snap_counts", seasons)
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
        raise AssertionError("snap_counts fanned out after 1:1 crosswalk merge")
    return df.reset_index(drop=True)


def injuries(seasons: Iterable[int]) -> pd.DataFrame:
    """Weekly injury reports, keyed by `gsis_id`."""
    df = _load("load_injuries", seasons)
    df["game_type"] = df["game_type"].fillna("REG")
    pk = ["gsis_id", "season", "week", "game_type", "team"]
    sort_col = "date_modified" if "date_modified" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col)
    # a report gets re-issued within a week (Questionable -> Out); keep the latest
    return df.drop_duplicates(pk, keep="last").reset_index(drop=True)


def schedules(seasons: Iterable[int]) -> pd.DataFrame:
    """Raw game rows — rest days, roof/surface/weather, closing spread/total."""
    return _load("load_schedules", seasons).reset_index(drop=True)


def team_week(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Explode `schedules` to one row per (season, week, game_type, team) with the
    pre-game-known context a feature can use without leakage. Implied total:
    `total_line/2` split by `spread_line` (positive = home favored)."""
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


def seasonal_rosters(seasons: Iterable[int]) -> pd.DataFrame:
    """One row per (player_id, season): the player's team + status/experience.
    Team is pre-Week-1-known, so `changed_team` derived from it is not leakage."""
    df = _load("load_rosters", seasons).rename(columns={"gsis_id": "player_id"})
    keep = ["player_id", "season", "team", "position", "status", "years_exp", "entry_year"]
    df = df[[c for c in keep if c in df.columns]]
    return df.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season"]).reset_index(drop=True)


# --- kicker & team-defense: native in nflreadpy's stat releases -------------

_FG_MADE = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49"]
_FG_MISS = ["fg_missed_0_19", "fg_missed_20_29", "fg_missed_30_39", "fg_missed_40_49"]


def kicking_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Weekly kicker lines from the `load_player_stats` frame (FG-by-distance is
    native). Keyed `(kicker_player_id, season, week, game_type, team)`."""
    k = player_stats_df[player_stats_df["position"] == "K"].copy()
    for c in _FG_MADE + _FG_MISS + ["fg_made", "fg_missed", "fg_made_distance",
                                    "fg_made_50_59", "fg_made_60_",
                                    "fg_missed_50_59", "fg_missed_60_",
                                    "pat_made", "pat_missed"]:
        k[c] = pd.to_numeric(k[c], errors="coerce").fillna(0) if c in k.columns else 0

    out = pd.DataFrame({
        "kicker_player_id": k["player_id"].values,
        "season": k["season"].values, "week": k["week"].values,
        "game_type": k["season_type"].fillna("REG").values,
        "team": k["team"].values,
        "fg_made": k["fg_made"].values,
        "fg_missed": k["fg_missed"].values,
        "fg_made_yds": k["fg_made_distance"].values,
        "fg_made_50p": (k["fg_made_50_59"] + k["fg_made_60_"]).values,
        "fg_missed_50p": (k["fg_missed_50_59"] + k["fg_missed_60_"]).values,
        "xp_made": k["pat_made"].values,
        "xp_missed": k["pat_missed"].values,
    })
    for c in _FG_MADE + _FG_MISS:
        out[c] = k[c].values
    return out.reset_index(drop=True)


def team_defense_stats(team_stats: pd.DataFrame, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Weekly team-defense lines. `def_*` box score from `load_team_stats`; yards
    allowed = the opponent's offensive yards in that game; points allowed = the
    opponent's final score. Keyed `(defense_team, season, week, game_type)`."""
    ts = team_stats.copy()
    for c in ("passing_yards", "rushing_yards", "def_sacks", "def_interceptions",
              "def_fumbles", "def_safeties", "def_tds", "def_fg_blocks",
              "def_pat_blocks", "def_punt_blocks"):
        ts[c] = pd.to_numeric(ts.get(c), errors="coerce").fillna(0)
    ts["off_yards"] = ts["passing_yards"] + ts["rushing_yards"]

    out = pd.DataFrame({
        "season": ts["season"], "week": ts["week"],
        "game_type": ts["season_type"].fillna("REG"),
        "defense_team": ts["team"], "game_id": ts["game_id"],
        "opponent_team": ts["opponent_team"],
        "dst_sack": ts["def_sacks"], "dst_int": ts["def_interceptions"],
        "dst_fum_rec": ts["def_fumbles"], "dst_safety": ts["def_safeties"],
        "dst_td": ts["def_tds"],
        "dst_blk_kick": ts["def_fg_blocks"] + ts["def_pat_blocks"] + ts["def_punt_blocks"],
    })

    # yards allowed: opponent's offensive yards in the same game
    opp_yards = ts[["game_id", "team", "off_yards"]].rename(
        columns={"team": "opponent_team", "off_yards": "dst_yds_allowed"})
    out = out.merge(opp_yards, on=["game_id", "opponent_team"], how="left")

    # points allowed: opponent's final score
    sc = schedules_df
    scores = pd.concat([
        pd.DataFrame({"game_id": sc["game_id"], "defense_team": sc["home_team"],
                      "dst_pts_allowed": sc["away_score"]}),
        pd.DataFrame({"game_id": sc["game_id"], "defense_team": sc["away_team"],
                      "dst_pts_allowed": sc["home_score"]}),
    ])
    out = out.merge(scores, on=["game_id", "defense_team"], how="left")

    for c in ("dst_sack", "dst_int", "dst_fum_rec", "dst_safety", "dst_td", "dst_blk_kick"):
        out[c] = out[c].round().astype(int)
    return (
        out.drop(columns=["game_id", "opponent_team"])
        .drop_duplicates(["defense_team", "season", "week", "game_type"])
        .reset_index(drop=True)
    )


def player_ids() -> pd.DataFrame:
    """Cross-reference: gsis_id <-> pfr_id <-> sleeper_id <-> yahoo_id <-> espn_id.
    From nflverse's maintained `load_ff_playerids` — do not rebuild from names."""
    df = _load("load_ff_playerids")
    keep = ["gsis_id", "pfr_id", "sleeper_id", "yahoo_id", "espn_id", "mfl_id",
            "sportradar_id", "name", "merge_name", "position", "team", "birthdate",
            "draft_year", "draft_round", "draft_pick", "draft_ovr"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["gsis_id"]).drop_duplicates(subset=["gsis_id"])
    for c in ("sleeper_id", "yahoo_id", "espn_id", "mfl_id"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df.reset_index(drop=True)
