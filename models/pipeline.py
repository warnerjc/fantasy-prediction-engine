"""Assemble → train → walk-forward evaluate → project. Config-driven throughout.

``train_one`` never names a target column, objective, or split boundary — it
reads them off ``ModelConfig``. v2 (weekly quantiles) is a new config, not an
edit here.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import NON_FEATURE_COLS, ModelConfig
from .prediction import predictions_frame


def assemble_position(
    feature_matrix: pd.DataFrame, labels: pd.DataFrame, position: str
) -> pd.DataFrame:
    """Join per-target-season feature rows to their label (label season == the
    season being predicted). One row per (player_id, target_season)."""
    feats = feature_matrix.copy()
    lab = labels[labels["position"] == position][
        ["player_id", "season", "games", "fantasy_points", "ppg"]
    ].rename(columns={
        "season": "target_season", "games": "label_games",
        "fantasy_points": "label_points",
    })
    merged = feats.merge(lab, on=["player_id", "target_season"], how="left")
    merged["position"] = position
    if "games" in merged.columns:
        merged = merged.rename(columns={"games": "prior_games"})
    return merged


def feature_columns(frame: pd.DataFrame, config: ModelConfig | None = None) -> list[str]:
    exclude = config.exclude_feature_prefixes if config else ()
    cols = []
    for c in frame.columns:
        if c in NON_FEATURE_COLS or c.startswith(exclude):
            continue
        if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]):
            cols.append(c)
    return cols


def _lgbm(config: ModelConfig, objective: str | None = None, alpha: float | None = None):
    params = dict(
        objective=objective or config.objective,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        subsample_freq=1,
        random_state=config.random_state,
        verbose=-1,
    )
    if (objective or config.objective) == "tweedie":
        params["tweedie_variance_power"] = config.tweedie_variance_power
    if alpha is not None:
        params["alpha"] = alpha
    return lgb.LGBMRegressor(**params)


@dataclass
class FittedModel:
    config: ModelConfig
    features: list[str]
    mean_model: lgb.LGBMRegressor
    quantile_models: dict[float, lgb.LGBMRegressor]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features]
        out = predictions_frame(self.mean_model.predict(X), index=frame.index)
        for q, m in self.quantile_models.items():
            out[f"pred_p{int(q * 100)}"] = m.predict(X)
        return out


def _training_mask(frame: pd.DataFrame, config: ModelConfig) -> pd.Series:
    m = frame["label_points"].notna() & (frame["label_games"] >= config.min_label_games)
    if "prior_games" in frame.columns:
        m &= frame["prior_games"] >= config.min_feature_games
    return m


def train_one(frame: pd.DataFrame, config: ModelConfig) -> FittedModel:
    feats = feature_columns(frame, config)
    train = frame[_training_mask(frame, config)]
    X, y = train[feats], train[config.target]
    w = train["label_games"] if config.sample_weight_by_label_games else None

    objective = config.objective
    if objective == "tweedie" and (y < 0).any():
        objective = "regression"  # tweedie needs y >= 0
    mean_model = _lgbm(config, objective=objective).fit(X, y, sample_weight=w)
    qmodels: dict[float, lgb.LGBMRegressor] = {}
    for q in config.quantiles:
        qmodels[q] = _lgbm(config, objective="quantile", alpha=q).fit(X, y, sample_weight=w)
    return FittedModel(config, feats, mean_model, qmodels)


def _metrics(pred: pd.Series, actual: pd.Series, top_n: int) -> dict:
    ok = pred.notna() & actual.notna()
    p, a = pred[ok], actual[ok]
    if len(p) < 5:
        return {"n": len(p), "spearman": np.nan, "mae": np.nan, f"top{top_n}_hit": np.nan}
    top_pred = set(p.sort_values(ascending=False).head(top_n).index)
    top_act = set(a.sort_values(ascending=False).head(top_n).index)
    return {
        "n": len(p),
        "spearman": spearmanr(p, a).statistic,
        "mae": float((p - a).abs().mean()),
        f"top{top_n}_hit": len(top_pred & top_act) / top_n,
    }


_TOP_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "K": 12, "DEF": 12}


def walk_forward(
    assembled: pd.DataFrame, config: ModelConfig, eval_seasons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on target_season < S, predict S, for each S in ``eval_seasons`` with
    enough prior seasons. Returns (per-season metrics, OOF predictions)."""
    rows, oof = [], []
    top_n = _TOP_N.get(config.position, 24)
    for s in sorted(eval_seasons):
        train = assembled[assembled["target_season"] < s]
        if train["target_season"].nunique() < config.min_train_seasons:
            continue
        test = assembled[assembled["target_season"] == s].copy()
        if test.empty:
            continue
        model = train_one(train, config)
        pred = model.predict(test)
        test = pd.concat([test, pred], axis=1)
        m = _metrics(test.set_index("player_id")["pred_mean"],
                     test.set_index("player_id")[config.target], top_n)
        m = {"position": config.position, "season": s, **m}
        rows.append(m)
        oof.append(test[["player_id", "target_season", "position", config.target,
                         "label_points", "label_games", "pred_mean"]])
    metrics = pd.DataFrame(rows)
    oof_df = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
    return metrics, oof_df


def project_position(
    assembled: pd.DataFrame, config: ModelConfig, target_season: int
) -> pd.DataFrame:
    """Train on everything before ``target_season``, predict it. Ranked by
    projected PPG, with a naive projected-season-points column."""
    train = assembled[assembled["target_season"] < target_season]
    test = assembled[assembled["target_season"] == target_season].copy()
    if test.empty or train.empty:
        return pd.DataFrame()

    model = train_one(train, config)
    test = pd.concat([test.reset_index(drop=True),
                      model.predict(test.reset_index(drop=True))], axis=1)

    games_assumption = _games_assumption(config.position)
    test["proj_ppg"] = test["pred_mean"]
    test["proj_points"] = test["proj_ppg"] * games_assumption
    test = test.sort_values("proj_ppg", ascending=False)
    test["pos_rank"] = np.arange(1, len(test) + 1)
    keep = ["pos_rank", "player_id", "position", "most_recent_team", "target_season",
            "proj_ppg", "proj_points", "pred_p10", "pred_p50", "pred_p90"]
    return test[[c for c in keep if c in test.columns]].reset_index(drop=True)


def _games_assumption(position: str) -> float:
    # naive v1 games-played reconstruction; a real durability model is later work
    return {"QB": 16.0, "RB": 15.0, "WR": 15.5, "TE": 15.0, "K": 17.0, "DEF": 17.0}.get(position, 15.5)
