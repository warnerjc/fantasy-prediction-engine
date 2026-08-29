"""Feature functions on player_week_stats + raw context tables.

Every function is parameterized by an **as-of point** (the game about to be
played) and a **window** (how far back to look). v1 calls them with
``Window.prior_season()``; v2 will call the *same* functions with
``Window.trailing(n_games=4)``. There is no v1-only or v2-only fork of a feature.

Leakage rule: a feature "as of" (season, week) may read outcome stats only from
strictly-earlier weeks. Pre-game-known context (opponent, home/away, rest, Vegas
line, reported injury status) is allowed for the as-of week itself.
"""

from .build import season_feature_matrix, training_frame
from .context import context_features
from .identity import identity_features
from .opponent import opponent_allowed_features
from .opportunity import opportunity_features, snap_features
from .window import AsOf, Window, visible_weeks, week_index

__all__ = [
    "AsOf",
    "Window",
    "visible_weeks",
    "week_index",
    "opportunity_features",
    "snap_features",
    "opponent_allowed_features",
    "context_features",
    "identity_features",
    "season_feature_matrix",
    "training_frame",
]
