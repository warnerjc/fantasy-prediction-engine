"""As-of point + window: selecting the player-weeks visible when predicting a game.

This is the shared engine. A feature function calls ``visible_weeks`` to get the
historical rows it is allowed to see, then aggregates them — it never does the
"which weeks count" filtering itself, so v1 (prior season) and v2 (trailing N
games) reuse identical feature code with a different ``Window``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def week_index(season, week) -> int:
    """Orderable absolute week key (weeks reset each season)."""
    return season * 100 + week


@dataclass(frozen=True)
class AsOf:
    """The game about to be played. Only data from strictly before this is visible.

    For a season-grain (draft) projection of season S, use ``AsOf(S, 1)``: nothing
    from season S is visible, all of S-1 is.
    """

    season: int
    week: int = 1

    @property
    def index(self) -> int:
        return week_index(self.season, self.week)


@dataclass(frozen=True)
class Window:
    """How far back ``visible_weeks`` looks from an ``AsOf`` point.

    - ``kind="prior_season"``: every week of the ``n_seasons`` seasons before the
      as-of season (v1 draft grain).
    - ``kind="trailing"``: the most recent ``n_games`` games *each player actually
      played*, strictly before the as-of point, spanning the season boundary if
      needed (v2 weekly grain). ``max_seasons_back`` caps how stale those games
      may be — with ``max_seasons_back=2`` and an as-of season S, only games from
      S and S-1 count, so a player who last suited up years ago contributes
      nothing (and doesn't pollute the opponent-allowed aggregation).

    ``drop_final_week`` (prior_season only): also drop the last REG week of each
    visible season. That week is played after every fantasy league's championship
    by locked-seed teams resting starters — fantasy-dead and statistically
    distorted. The v1 draft-grain call sites opt in; v2 trailing windows don't.
    """

    kind: str = "prior_season"
    n_seasons: int = 1
    n_games: int | None = None
    max_seasons_back: int | None = None
    season_types: tuple[str, ...] = ("REG",)
    drop_final_week: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("prior_season", "trailing"):
            raise ValueError(f"unknown window kind {self.kind!r}")
        if self.kind == "trailing" and not self.n_games:
            raise ValueError("trailing window needs n_games")

    @classmethod
    def prior_season(cls, n_seasons: int = 1, **kw) -> "Window":
        return cls(kind="prior_season", n_seasons=n_seasons, **kw)

    @classmethod
    def trailing(cls, n_games: int, **kw) -> "Window":
        return cls(kind="trailing", n_games=n_games, **kw)


def visible_weeks(
    pws: pd.DataFrame,
    as_of: AsOf,
    window: Window,
    player_col: str = "player_id",
    season_type_col: str = "season_type",
) -> pd.DataFrame:
    """Rows of ``pws`` visible when predicting ``as_of``, per ``window``.

    ``pws`` must have ``season``, ``week`` and a season-type column (named
    ``season_type_col`` — ``player_week_stats`` uses ``season_type``, the nflverse
    raw tables use ``game_type``). Output is a copy with a ``week_index`` column
    added, sorted by (player, week_index).
    """
    df = pws[pws[season_type_col].isin(window.season_types)].copy()
    df["week_index"] = week_index(df["season"], df["week"])

    if window.kind == "prior_season":
        lo = as_of.season - window.n_seasons
        df = df[(df["season"] >= lo) & (df["season"] < as_of.season)]
        if window.drop_final_week and not df.empty:
            final = df.groupby("season")["week"].transform("max")
            df = df[df["week"] < final]
    else:  # trailing
        df = df[df["week_index"] < as_of.index]
        if window.max_seasons_back is not None:
            df = df[df["season"] >= as_of.season - window.max_seasons_back]
        df = df.sort_values([player_col, "week_index"])
        df = df.groupby(player_col, sort=False).tail(window.n_games)

    return df.sort_values([player_col, "week_index"]).reset_index(drop=True)
