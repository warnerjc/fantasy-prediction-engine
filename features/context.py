"""Pre-game-known context for the as-of week itself: Vegas line, venue, rest.

Unlike opportunity features, these describe the as-of game and are legitimately
known before kickoff, so they read the as-of (season, week) row of ``team_week``
directly — no window, no leakage. v1 season-grain projection doesn't use these
(they're per-game); v2 weekly does, joined on the player's team.
"""

from __future__ import annotations

import pandas as pd

from .window import AsOf

_DOME_ROOFS = {"dome", "closed"}


def context_features(team_week: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """One row per team for ``as_of`` — opponent, home/away, rest, implied total,
    spread, venue. Join to players on ``team``.
    """
    tw = team_week[
        (team_week["season"] == as_of.season)
        & (team_week["week"] == as_of.week)
        & (team_week["game_type"].isin(("REG", "POST")))
    ].copy()

    keep = ["team", "opponent", "is_home", "rest", "div_game",
            "implied_total", "team_spread", "roof", "temp", "wind"]
    out = tw[[c for c in keep if c in tw.columns]].copy()

    roof = out.get("roof")
    out["is_dome"] = roof.isin(_DOME_ROOFS).astype("Int64") if roof is not None else pd.NA
    out["is_outdoors"] = roof.eq("outdoors").astype("Int64") if roof is not None else pd.NA
    out["short_week"] = (pd.to_numeric(out.get("rest"), errors="coerce") <= 5).astype("Int64")

    return out.reset_index(drop=True)
