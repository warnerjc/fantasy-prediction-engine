"""Team / situation change features.

The 2024 backtest's biggest misses were players who changed teams (Saquon
NYG→PHI proj RB19→finished 1, Henry, J.Taylor): a prior-usage model sees last
year's role and can't see the new team's opportunity or offense. These features
close that gap.

Category: **pre-game-known**. Free agency and trades settle in the offseason, so a
player's team for the upcoming season and the opportunity vacated on it are known
before Week 1 — same status as opponent / home-away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .window import AsOf, Window, visible_weeks


def _modal(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def team_change_features(
    pws: pd.DataFrame,
    seasonal_rosters: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
) -> pd.DataFrame:
    """One row per player (present in the window), keyed ``player_id``:
    ``changed_team`` + the new team's prior-season offense environment + the
    target/carry share vacated on the new team by departed players.
    """
    window = window or Window.prior_season()
    as_of = AsOf(target_season, 1)
    vis = visible_weeks(pws, as_of, window)
    if vis.empty:
        return pd.DataFrame()

    for c in ("targets", "carries", "attempts"):
        vis[c] = pd.to_numeric(vis.get(c), errors="coerce").fillna(0)

    tgt_team = (
        seasonal_rosters[seasonal_rosters["season"] == target_season]
        .drop_duplicates("player_id")
        .set_index("player_id")["team"]
    )
    prior_team = vis.groupby("player_id")["team"].agg(_modal)

    out = pd.DataFrame({"prior_team": prior_team})
    mapped = pd.Series(out.index.map(tgt_team), index=out.index)
    out["new_team"] = mapped.fillna(out["prior_team"])
    out["changed_team"] = (out["new_team"] != out["prior_team"]).astype("int64")

    # --- new team's prior-season offensive environment (most recent window season)
    last = vis[vis["season"] == vis["season"].max()]
    team_wk = last.groupby(["team", "season", "week"]).agg(
        pass_att=("attempts", "sum"), plays=("targets", "sum"),
    )
    team_env = team_wk.groupby(level="team").agg(
        new_team_pass_att_pg=("pass_att", "mean"),
    )
    rush_wk = last.groupby(["team", "season", "week"])["carries"].sum()
    team_env["new_team_rush_att_pg"] = rush_wk.groupby(level="team").mean()
    team_env["new_team_pass_ratio"] = team_env["new_team_pass_att_pg"] / (
        team_env["new_team_pass_att_pg"] + team_env["new_team_rush_att_pg"]
    )

    tw = team_week[team_week["season"] == vis["season"].max()]
    team_env["new_team_implied_total_prior"] = (
        tw.groupby("team")["implied_total"].mean()
    )

    out = out.merge(team_env, left_on="new_team", right_index=True, how="left")

    # --- opportunity vacated on the new team: window usage of players who were on
    #     that team then and are NOT on it for target_season
    pl = vis.groupby("player_id").agg(
        wteam=("team", _modal), tgt=("targets", "sum"), car=("carries", "sum")
    )
    pl["now"] = pd.Series(pl.index.map(tgt_team), index=pl.index).fillna("__gone__")
    left = pl[pl["wteam"] != pl["now"]]
    vac = left.groupby("wteam").agg(vac_tgt=("tgt", "sum"), vac_car=("car", "sum"))
    team_tot = pl.groupby("wteam").agg(tot_tgt=("tgt", "sum"), tot_car=("car", "sum"))
    vac = vac.join(team_tot)
    vac["vacated_tgt_share_new_team"] = vac["vac_tgt"] / vac["tot_tgt"].replace(0, np.nan)
    vac["vacated_rush_share_new_team"] = vac["vac_car"] / vac["tot_car"].replace(0, np.nan)

    out = out.merge(
        vac[["vacated_tgt_share_new_team", "vacated_rush_share_new_team"]],
        left_on="new_team", right_index=True, how="left",
    )
    out[["vacated_tgt_share_new_team", "vacated_rush_share_new_team"]] = (
        out[["vacated_tgt_share_new_team", "vacated_rush_share_new_team"]].fillna(0)
    )

    return out.reset_index().rename(columns={"index": "player_id"}).drop(columns=["prior_team"])
