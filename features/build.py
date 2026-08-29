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

from .identity import identity_features
from .opportunity import opportunity_features, snap_features
from .window import AsOf, Window, visible_weeks

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def season_feature_matrix(
    pws: pd.DataFrame,
    snap_counts: pd.DataFrame,
    player_ids: pd.DataFrame,
    target_season: int,
    window: Window | None = None,
    positions: tuple[str, ...] = SKILL_POSITIONS,
) -> pd.DataFrame:
    """Feature row per player, for predicting ``target_season``. Keyed by
    (``player_id``, ``target_season``). ``window`` defaults to the full prior season.
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
) -> pd.DataFrame:
    """``season_feature_matrix`` stacked over several target seasons — the X matrix
    for walk-forward training. The model attaches y and splits by ``target_season``.
    """
    frames = [
        season_feature_matrix(pws, snap_counts, player_ids, s, window, positions)
        for s in sorted(target_seasons)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
