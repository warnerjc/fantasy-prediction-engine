"""Feature matrices for kicker and team-defense projections.

Same window engine as the offense features (``visible_weeks`` + ``AsOf`` /
``Window``), but sourced from ``kicking_stats`` / ``team_defense_stats`` rather
than ``player_week_stats``. K and DEF carry far less year-over-year signal than
skill players, so these sets are deliberately small — prior-season own
production plus a team-quality proxy from ``team_week``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context import context_features
from .window import AsOf, Window, visible_weeks

_KICK_CONTEXT = ["implied_total", "team_spread", "is_dome", "is_outdoors",
                 "temp", "wind", "is_home", "rest", "short_week"]


def _team_quality(team_week: pd.DataFrame, as_of: AsOf, window: Window) -> pd.DataFrame:
    """Prior-window mean implied total / spread per team — offense-strength proxy
    (more points → more FG/XP chances for K; game script for DEF)."""
    tw = team_week.rename(columns={"game_type": "season_type"})
    vis = visible_weeks(tw, as_of, window, player_col="team", season_type_col="season_type")
    g = vis.groupby("team")
    return pd.DataFrame({
        "team_implied_total_prior": g["implied_total"].mean(),
        "team_spread_prior": g["team_spread"].mean(),
        "team_games_prior": g.size(),
    }).reset_index()


def kicker_feature_matrix(
    kicking_stats: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
) -> pd.DataFrame:
    """One row per kicker for predicting ``target_season``. Keyed
    (``player_id``, ``target_season``)."""
    window = window or Window.prior_season()
    as_of = AsOf(target_season, 1)

    k = kicking_stats.rename(columns={"kicker_player_id": "player_id"})
    vis = visible_weeks(k, as_of, window, player_col="player_id", season_type_col="game_type")
    if vis.empty:
        return pd.DataFrame()

    for c in ("fg_made", "fg_missed", "fg_made_50p", "fg_made_yds", "xp_made", "xp_missed"):
        vis[c] = pd.to_numeric(vis.get(c), errors="coerce").fillna(0)

    g = vis.groupby("player_id")
    out = pd.DataFrame(index=g.size().index)
    out["games"] = g.size()
    out["fg_made_pg"] = g["fg_made"].sum() / out["games"]
    out["fg_att_pg"] = (g["fg_made"].sum() + g["fg_missed"].sum()) / out["games"]
    out["fg_pct"] = g["fg_made"].sum() / (g["fg_made"].sum() + g["fg_missed"].sum()).replace(0, np.nan)
    out["fg_made_50p_pg"] = g["fg_made_50p"].sum() / out["games"]
    out["fg_made_yds_pg"] = g["fg_made_yds"].sum() / out["games"]
    out["xp_made_pg"] = g["xp_made"].sum() / out["games"]
    out["most_recent_team"] = g["team"].last()

    out = out.reset_index().merge(
        _team_quality(team_week, as_of, window),
        left_on="most_recent_team", right_on="team", how="left",
    ).drop(columns=["team"], errors="ignore")
    out.insert(1, "target_season", target_season)
    out["position"] = "K"
    return out


def defense_feature_matrix(
    team_defense_stats: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
) -> pd.DataFrame:
    """One row per team defense for predicting ``target_season``. Keyed
    (``player_id`` = team abbrev, ``target_season``)."""
    window = window or Window.prior_season()
    as_of = AsOf(target_season, 1)

    d = team_defense_stats.rename(columns={"defense_team": "team"})
    vis = visible_weeks(d, as_of, window, player_col="team", season_type_col="game_type")
    if vis.empty:
        return pd.DataFrame()

    stat_cols = ["dst_sack", "dst_int", "dst_fum_rec", "dst_safety", "dst_td",
                 "dst_blk_kick", "dst_pts_allowed", "dst_yds_allowed"]
    for c in stat_cols:
        vis[c] = pd.to_numeric(vis.get(c), errors="coerce").fillna(0)

    g = vis.groupby("team")
    out = pd.DataFrame(index=g.size().index)
    out["games"] = g.size()
    for c in stat_cols:
        out[f"{c}_pg_prior"] = g[c].sum() / out["games"]
    out["takeaways_pg_prior"] = (g["dst_int"].sum() + g["dst_fum_rec"].sum()) / out["games"]

    out = out.reset_index().rename(columns={"team": "player_id"})
    out["most_recent_team"] = out["player_id"]
    out = out.merge(
        _team_quality(team_week, as_of, window),
        left_on="player_id", right_on="team", how="left",
    ).drop(columns=["team"], errors="ignore")
    out.insert(1, "target_season", target_season)
    out["position"] = "DEF"
    return out


# --- v2 weekly K / DEF matrices ---------------------------------------------

def _weekly_trailing_agg(
    source: pd.DataFrame, id_col: str, as_of: AsOf, window: Window,
    stat_cols: list[str],
) -> pd.DataFrame | None:
    """Sum ``stat_cols`` over each entity's trailing window before ``as_of`` and
    divide by games. Returns None when the window is empty."""
    vis = visible_weeks(source, as_of, window, player_col=id_col, season_type_col="game_type")
    if vis.empty:
        return None
    for c in stat_cols:
        vis[c] = pd.to_numeric(vis.get(c), errors="coerce").fillna(0)
    g = vis.groupby(id_col)
    out = pd.DataFrame(index=g.size().index)
    out["games"] = g.size()
    for c in stat_cols:
        out[f"{c}_pg"] = g[c].sum() / out["games"]
    out["most_recent_team"] = g["team"].last() if "team" in vis.columns else None
    return out.reset_index()


def week_kicker_matrix(
    kicking_stats: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
) -> pd.DataFrame:
    """One row per kicker-week played in ``target_season`` (REG). Keyed
    (``player_id``, ``target_season``, ``week``). Trailing own production +
    ``context_features`` (implied total / venue) for the as-of game."""
    window = window or Window.trailing(n_games=8, max_seasons_back=2)
    k = kicking_stats.rename(columns={"kicker_player_id": "player_id"})
    reg = k[(k["season"] == target_season) & (k["game_type"] == "REG")]
    if reg.empty:
        return pd.DataFrame()
    skel = reg[["player_id", "season", "week", "team"]].drop_duplicates(
        subset=["player_id", "season", "week"])

    stat_cols = ["fg_made", "fg_missed", "fg_made_50p", "fg_made_yds", "xp_made", "xp_missed"]
    per_week = []
    for wk in sorted(skel["week"].unique()):
        as_of = AsOf(target_season, int(wk))
        agg = _weekly_trailing_agg(k, "player_id", as_of, window, stat_cols)
        if agg is None:
            continue
        agg["fg_pct"] = agg["fg_made_pg"] / (agg["fg_made_pg"] + agg["fg_missed_pg"]).replace(0, np.nan)
        ctx = context_features(team_week, as_of)
        ctx = ctx[["team"] + [c for c in _KICK_CONTEXT if c in ctx.columns]]
        w = skel[skel["week"] == wk].merge(
            agg.drop(columns=["most_recent_team"], errors="ignore"), on="player_id", how="left")
        w = w.merge(ctx, on="team", how="left")
        per_week.append(w)

    if not per_week:
        return pd.DataFrame()
    out = pd.concat(per_week, ignore_index=True).rename(columns={"season": "target_season"})
    out["position"] = "K"
    return out.reset_index(drop=True)


def week_defense_matrix(
    team_defense_stats: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
) -> pd.DataFrame:
    """One row per team-defense-week played in ``target_season`` (REG). Keyed
    (``player_id`` = team abbrev, ``target_season``, ``week``)."""
    window = window or Window.trailing(n_games=8, max_seasons_back=2)
    d = team_defense_stats.rename(columns={"defense_team": "team"})
    reg = d[(d["season"] == target_season) & (d["game_type"] == "REG")]
    if reg.empty:
        return pd.DataFrame()
    skel = reg[["team", "season", "week"]].drop_duplicates(subset=["team", "season", "week"])

    stat_cols = ["dst_sack", "dst_int", "dst_fum_rec", "dst_safety", "dst_td",
                 "dst_blk_kick", "dst_pts_allowed", "dst_yds_allowed"]
    per_week = []
    for wk in sorted(skel["week"].unique()):
        as_of = AsOf(target_season, int(wk))
        agg = _weekly_trailing_agg(d, "team", as_of, window, stat_cols)
        if agg is None:
            continue
        agg["takeaways_pg"] = agg["dst_int_pg"] + agg["dst_fum_rec_pg"]
        ctx = context_features(team_week, as_of)
        ctx = ctx[["team"] + [c for c in _KICK_CONTEXT if c in ctx.columns]]
        w = skel[skel["week"] == wk].merge(
            agg.drop(columns=["most_recent_team"], errors="ignore"), on="team", how="left")
        w = w.merge(ctx, on="team", how="left")
        per_week.append(w)

    if not per_week:
        return pd.DataFrame()
    out = pd.concat(per_week, ignore_index=True).rename(
        columns={"season": "target_season", "team": "player_id"})
    out["most_recent_team"] = out["player_id"]
    out["position"] = "DEF"
    return out.reset_index(drop=True)
