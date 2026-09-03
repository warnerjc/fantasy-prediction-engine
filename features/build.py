"""Assemble a model-ready feature matrix from the feature functions.

``season_feature_matrix`` is the v1 (draft) entry point: for predicting season S,
it builds one row per player from their season S-1 usage + priors. It calls the
same window/opportunity/snap functions v2 will call with a trailing window — the
only v1-specific thing here is the choice of ``AsOf(S, 1)`` and
``Window.prior_season()``.

No labels here. The model layer joins the scored target (see /scoring, /models).
"""

from __future__ import annotations

import pandas as pd

from .context import context_features
from .identity import identity_features
from .opponent import opponent_allowed_features
from .opportunity import opportunity_features, snap_features
from .team_change import team_change_features
from .window import AsOf, Window, visible_weeks

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def season_feature_matrix(
    pws: pd.DataFrame,
    snap_counts: pd.DataFrame,
    player_ids: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
    positions: tuple[str, ...] = SKILL_POSITIONS,
    seasonal_rosters: pd.DataFrame | None = None,
    team_week: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature row per player, for predicting ``target_season``. Keyed by
    (``player_id``, ``target_season``). ``window`` defaults to the full prior season.
    Pass ``seasonal_rosters`` + ``team_week`` to include team-change features.
    """
    window = window or Window.prior_season()
    as_of = AsOf(target_season, 1)

    vis = visible_weeks(pws, as_of, window)
    if vis.empty:
        return pd.DataFrame()

    vis_snaps = visible_weeks(
        snap_counts, as_of, window, player_col="gsis_id", season_type_col="game_type"
    )

    feats = opportunity_features(vis)
    feats = feats.merge(
        snap_features(vis_snaps), left_on="player_id", right_on="gsis_id", how="left"
    ).drop(columns=["gsis_id"], errors="ignore")
    feats = feats.merge(identity_features(player_ids, as_of), on="player_id", how="left")

    if seasonal_rosters is not None and team_week is not None:
        tc = team_change_features(pws, seasonal_rosters, team_week, target_season, window)
        if not tc.empty:
            feats = feats.merge(tc, on="player_id", how="left")

    feats = feats[feats["most_recent_pos"].isin(positions)].copy()
    feats.insert(1, "target_season", target_season)
    return feats.reset_index(drop=True)


def training_frame(
    pws: pd.DataFrame,
    snap_counts: pd.DataFrame,
    player_ids: pd.DataFrame,
    target_seasons: list[int],
    window: Window | None = None,
    positions: tuple[str, ...] = SKILL_POSITIONS,
    seasonal_rosters: pd.DataFrame | None = None,
    team_week: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``season_feature_matrix`` stacked over several target seasons — the X matrix
    for walk-forward training. The model attaches y and splits by ``target_season``.
    """
    frames = [
        season_feature_matrix(pws, snap_counts, player_ids, s, window, positions,
                              seasonal_rosters, team_week)
        for s in sorted(target_seasons)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --- v2 weekly matrix --------------------------------------------------------

# context_features columns that are model inputs (drop the raw `roof` string and
# the duplicate `opponent`, which the skeleton already carries)
_CONTEXT_KEEP = ["is_home", "rest", "div_game", "implied_total", "team_spread",
                 "temp", "wind", "is_dome", "is_outdoors", "short_week"]


def week_feature_matrix(
    pws: pd.DataFrame,
    snap_counts: pd.DataFrame,
    player_ids: pd.DataFrame,
    team_week: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
    positions: tuple[str, ...] = SKILL_POSITIONS,
) -> pd.DataFrame:
    """One row per player-week actually played in ``target_season`` (REG), keyed
    (``player_id``, ``target_season``, ``week``). The v2 (start/sit) entry point.

    Each row is built ``AsOf(target_season, week)`` with a **trailing** window:
    the player's last N games strictly before that week, crossing the season
    boundary. Same opportunity / snap / identity functions v1 calls with a
    prior-season window, plus the per-game context the season matrix omits —
    Vegas implied total / spread, venue, rest (``context_features`` on the
    player's team) and the opponent defense's allowed-by-position profile
    (``opponent_allowed_features`` on the player's Week-N opponent).
    """
    window = window or Window.trailing(n_games=6, max_seasons_back=2)
    if window.kind != "trailing":
        raise ValueError("week_feature_matrix needs a trailing window")

    reg = pws[(pws["season"] == target_season) & (pws["season_type"] == "REG")]
    reg = reg[reg["position"].isin(positions)]
    if reg.empty:
        return pd.DataFrame()
    skel = (reg[["player_id", "season", "week", "team", "opponent", "position"]]
            .drop_duplicates(subset=["player_id", "season", "week"]))

    per_week = []
    for wk in sorted(skel["week"].unique()):
        as_of = AsOf(target_season, int(wk))
        vis = visible_weeks(pws, as_of, window)
        if vis.empty:
            continue
        vis_snaps = visible_weeks(snap_counts, as_of, window,
                                  player_col="gsis_id", season_type_col="game_type")

        feats = opportunity_features(vis)
        feats = feats.merge(snap_features(vis_snaps), left_on="player_id",
                            right_on="gsis_id", how="left").drop(columns=["gsis_id"], errors="ignore")
        feats = feats.merge(identity_features(player_ids, as_of), on="player_id", how="left")

        ctx = context_features(team_week, as_of)
        ctx = ctx[["team"] + [c for c in _CONTEXT_KEEP if c in ctx.columns]]
        # opponent_allowed over the same visible rows -> allowed-by-position per
        # defense faced. (Per-player trailing rows grouped by opponent: a coarse
        # weekly proxy, consistent with the season matrix's join semantics.)
        oppo = opponent_allowed_features(vis)

        w = skel[skel["week"] == wk].merge(feats, on="player_id", how="left")
        w = w.merge(ctx, on="team", how="left")
        w = w.merge(oppo, left_on="opponent", right_on="defense_team", how="left") \
             .drop(columns=["defense_team"], errors="ignore")
        per_week.append(w)

    if not per_week:
        return pd.DataFrame()
    out = pd.concat(per_week, ignore_index=True).rename(columns={"season": "target_season"})
    # skeleton position is the pre-game truth; opportunity's window-derived
    # most_recent_pos can disagree after a position reclass — keep both, model
    # ignores the string.
    return out.reset_index(drop=True)
