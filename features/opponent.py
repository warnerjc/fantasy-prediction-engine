"""Priority-2 feature: opponent defense allowed, by position, over a window.

"Allowed" = production of opposing scorers against that defense. Computed from the
same ``visible`` player-weeks the opportunity features use (outcome data, so
strictly pre-as-of), grouped by the *defense faced* (``opponent``) and the
scorer's position.

v1 season-grain doesn't join this per-game; v2 weekly joins each player's row to
``def_<POS>_*`` for their upcoming opponent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ALLOWED = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "receptions": "rec",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "targets": "tgt",
    "carries": "carries",
}
_POSITIONS = ("QB", "RB", "WR", "TE")


def opponent_allowed_features(visible: pd.DataFrame) -> pd.DataFrame:
    """One row per defense team, columns ``def_<POS>_<stat>_pg`` = mean production
    that defense allowed to each position per game over the window.
    """
    df = visible.copy()
    df = df[df["position"].isin(_POSITIONS)]
    for c in _ALLOWED:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    # games each defense played in the window = distinct (season, week) it appears
    # in as the opponent of a tracked scorer
    games = (
        df.groupby("opponent")[["season", "week"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
        .rename("def_games")
    )

    totals = df.groupby(["opponent", "position"])[list(_ALLOWED)].sum()
    totals = totals.rename(columns=_ALLOWED)

    per_game = totals.div(games, axis=0, level=0)
    wide = per_game.unstack("position")
    wide.columns = [f"def_{pos}_{stat}_pg" for stat, pos in wide.columns]
    wide = wide.join(games)

    return wide.reset_index().rename(columns={"opponent": "defense_team"})
