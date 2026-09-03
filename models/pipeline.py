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
    feature_matrix: pd.DataFrame, labels: pd.DataFrame, position: str,
    grain: str = "season",
) -> pd.DataFrame:
    """Join feature rows to their label.

    ``grain="season"`` (v1): label season == the season predicted, one row per
    (player_id, target_season). ``grain="week"`` (v2): ``week_labels`` joined on
    (player_id, target_season, week), one row per player-week, with a
    ``label_week_played`` flag.
    """
    feats = feature_matrix.copy()
    pos_lab = labels[labels["position"] == position]

    if grain == "week":
        lab = pos_lab[["player_id", "season", "week", "week_points"]].rename(
            columns={"season": "target_season"})
        merged = feats.merge(lab, on=["player_id", "target_season", "week"], how="left")
        merged["label_week_played"] = merged["week_points"].notna()
    else:
        lab = pos_lab[["player_id", "season", "games", "fantasy_points", "ppg"]].rename(
            columns={"season": "target_season", "games": "label_games",
                     "fantasy_points": "label_points"})
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
    if config.grain == "week":
        m = frame[config.target].notna()          # the player-week was actually played
    else:
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


def _pinball(y: pd.Series, q: pd.Series, alpha: float) -> float:
    d = y - q
    return float(np.where(d >= 0, alpha * d, (alpha - 1.0) * d).mean())


def _weekly_metrics(test: pd.DataFrame, target: str, top_n: int) -> dict:
    """Player-week grain: rank ρ / MAE pooled over all player-weeks, top-N hit
    averaged per week, and (when quantiles are populated) empirical p10–p90
    coverage + p50 pinball loss."""
    ok = test["pred_mean"].notna() & test[target].notna()
    t = test[ok]
    base = {"n": len(t), "spearman": np.nan, "mae": np.nan, f"top{top_n}_hit": np.nan,
            "coverage_80": np.nan, "pinball_p50": np.nan}
    if len(t) < 10:
        return base
    base["spearman"] = spearmanr(t["pred_mean"], t[target]).statistic
    base["mae"] = float((t["pred_mean"] - t[target]).abs().mean())

    hits = []
    for _, g in t.groupby("week"):
        if len(g) < top_n:
            continue
        tp = set(g.sort_values("pred_mean", ascending=False).head(top_n).index)
        ta = set(g.sort_values(target, ascending=False).head(top_n).index)
        hits.append(len(tp & ta) / top_n)
    base[f"top{top_n}_hit"] = float(np.mean(hits)) if hits else np.nan

    if "pred_p10" in t.columns and t["pred_p10"].notna().any():
        q = t.dropna(subset=["pred_p10", "pred_p50", "pred_p90"])
        if len(q) >= 10:
            inside = (q[target] >= q["pred_p10"]) & (q[target] <= q["pred_p90"])
            base["coverage_80"] = float(inside.mean())
            base["pinball_p50"] = _pinball(q[target], q["pred_p50"], 0.5)
    return base


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

        if config.grain == "week":
            m = _weekly_metrics(test, config.target, top_n)
            oof_cols = ["player_id", "target_season", "week", "position", config.target,
                        "pred_mean", "pred_p10", "pred_p50", "pred_p90"]
        else:
            m = _metrics(test.set_index("player_id")["pred_mean"],
                         test.set_index("player_id")[config.target], top_n)
            oof_cols = ["player_id", "target_season", "position", config.target,
                        "label_points", "label_games", "pred_mean"]
        rows.append({"position": config.position, "season": s, **m})
        oof.append(test[[c for c in oof_cols if c in test.columns]])
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
