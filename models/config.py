"""Training config. Everything the training loop varies is here, not in its body.

The invariant test (AGENTS.md): adding the v2 weekly quantile model should be a
new ``ModelConfig`` (different ``target``, ``objective``, ``quantiles``, split),
not an edit to ``pipeline.train_one``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# columns that are identifiers / metadata, never model inputs
NON_FEATURE_COLS = frozenset({
    "player_id", "target_season", "season", "week", "position", "team",
    "opponent", "defense_team",
    "most_recent_team", "most_recent_pos", "gsis_id",
    "fantasy_points", "ppg", "games", "label_games", "label_points",
    # weekly grain (v2)
    "week_points", "label_week_played",
})


@dataclass(frozen=True)
class ModelConfig:
    position: str
    target: str = "ppg"                 # column in the assembled frame to predict
    grain: str = "season"               # "season" (v1 draft) | "week" (v2 start/sit)
    # trailing games each player's window looks back (grain="week" only); None ->
    # the season-grain full-prior-season window
    feature_window_games: int | None = None
    objective: str = "tweedie"          # LightGBM objective
    tweedie_variance_power: float = 1.3
    learning_rate: float = 0.03
    n_estimators: int = 600
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    # a training row's feature (prior) season needs at least this many games to
    # carry a real usage signal; its target season needs min_label_games for a
    # stable label. Neither filters inference.
    min_feature_games: int = 4
    min_label_games: int = 4
    min_train_seasons: int = 3
    sample_weight_by_label_games: bool = True
    quantiles: tuple[float, ...] = ()   # v1 empty -> mean only
    random_state: int = 42             # LightGBM subsampling is stochastic; pin it
    # feature-name prefixes to drop for this position (a walk-forward A/B showed
    # team-change features help RB/WR but slightly hurt QB/TE)
    exclude_feature_prefixes: tuple[str, ...] = ()

    def with_(self, **kw) -> "ModelConfig":
        return replace(self, **kw)


# per-position starting points. K/DEF get a shorter tree / more regularization —
# far less signal, easy to overfit.
_TEAM_CHANGE = ("changed_team", "new_team_", "vacated_")

DEFAULT_CONFIGS: dict[str, ModelConfig] = {
    "QB": ModelConfig("QB", exclude_feature_prefixes=_TEAM_CHANGE),
    "RB": ModelConfig("RB"),
    "WR": ModelConfig("WR"),
    "TE": ModelConfig("TE", exclude_feature_prefixes=_TEAM_CHANGE),
    "K": ModelConfig("K", num_leaves=15, n_estimators=300, min_child_samples=30,
                     min_feature_games=6, min_label_games=6),
    # DEF PPG goes negative in some leagues (points-allowed penalties) -> L2, not tweedie
    "DEF": ModelConfig("DEF", objective="regression", num_leaves=15, n_estimators=300,
                       min_child_samples=15, min_feature_games=6, min_label_games=6),
}


# --- v2 weekly quantile models ------------------------------------------------
# A new config, not a new training path (AGENTS.md invariant): same
# ``pipeline.train_one`` reads ``target="week_points"``, ``grain="week"``, a
# trailing feature window, and real ``quantiles``. Weekly fantasy points are more
# zero-inflated / right-skewed than a season PPG average, so the Tweedie variance
# power runs a touch higher. Sample weighting by games is season-grain only.
_WEEKLY_QUANTILES = (0.1, 0.5, 0.9)


def _weekly(pos: str, **kw) -> ModelConfig:
    base = DEFAULT_CONFIGS[pos]
    return base.with_(
        target="week_points",
        grain="week",
        feature_window_games=kw.pop("feature_window_games", 6),
        quantiles=_WEEKLY_QUANTILES,
        sample_weight_by_label_games=False,
        min_feature_games=kw.pop("min_feature_games", 3),
        **kw,
    )


WEEKLY_CONFIGS: dict[str, ModelConfig] = {
    "QB": _weekly("QB", tweedie_variance_power=1.5),
    "RB": _weekly("RB", tweedie_variance_power=1.5),
    "WR": _weekly("WR", tweedie_variance_power=1.5),
    "TE": _weekly("TE", tweedie_variance_power=1.5),
    "K": _weekly("K", feature_window_games=8, min_feature_games=4),
    # weekly DEF points go negative -> regression (train_one also guards y<0)
    "DEF": _weekly("DEF", feature_window_games=8, min_feature_games=4),
}
