"""Model-layer tests: label computation, config-driven training, walk-forward."""

import numpy as np
import pandas as pd
import pytest

from scoring import ScoringRules, stat_keys as K
from models import ModelConfig, season_labels
from models.config import NON_FEATURE_COLS
from models.pipeline import assemble_position, feature_columns, train_one, walk_forward
from models.prediction import Prediction, predictions_frame


# --- labels ----------------------------------------------------------------

def _pw(pid, pos, team, season, week, **stats):
    base = dict(player_id=pid, position=pos, team=team, opponent="X", season=season,
                week=week, season_type="REG", receptions=0, receiving_yards=0,
                receiving_tds=0, rushing_yards=0, rushing_tds=0, carries=0, targets=0,
                attempts=0, completions=0, passing_yards=0, passing_tds=0, interceptions=0)
    base.update(stats)
    return base


def test_season_labels_scores_and_averages_ppg():
    pws = pd.DataFrame([
        _pw("wr1", "WR", "AA", 2023, 1, receptions=5, receiving_yards=100, receiving_tds=1),
        _pw("wr1", "WR", "AA", 2023, 2, receptions=5, receiving_yards=50),
    ])
    rules = ScoringRules(per_unit={K.REC: 1.0, K.REC_YD: 0.1, K.REC_TD: 6.0})
    empty_k = pd.DataFrame(columns=["kicker_player_id", "season", "week", "game_type"])
    empty_d = pd.DataFrame(columns=["defense_team", "season", "week", "game_type"])

    lab = season_labels(pws, empty_k, empty_d, rules).set_index("player_id")
    # wk1: 5 + 10 + 6 = 21 ; wk2: 5 + 5 = 10 ; total 31 over 2 games
    assert lab.loc["wr1", "fantasy_points"] == pytest.approx(31.0)
    assert lab.loc["wr1", "games"] == 2
    assert lab.loc["wr1", "ppg"] == pytest.approx(15.5)


# --- prediction shape -----------------------------------------------------

def test_prediction_v1_has_mean_only():
    p = Prediction(mean=12.3)
    assert p.p10 is None and p.p50 is None and p.p90 is None
    assert p.as_dict() == {"mean": 12.3, "p10": None, "p50": None, "p90": None}


def test_predictions_frame_carries_full_schema():
    f = predictions_frame([1.0, 2.0])
    assert list(f.columns) == ["pred_mean", "pred_p10", "pred_p50", "pred_p90"]
    assert f["pred_p10"].isna().all()


# --- assemble / features -------------------------------------------------

def _assembled(n_seasons=6, per_season=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2016, 2016 + n_seasons):
        for i in range(per_season):
            usage = rng.gamma(3, 2)
            rows.append(dict(
                player_id=f"p{i}", target_season=s, position="WR",
                most_recent_team="AA", most_recent_pos="WR",
                prior_games=int(rng.integers(4, 18)),
                targets_pg=usage, rec_yd_pg=usage * 8 + rng.normal(0, 5),
                target_share=usage / 40,
                label_games=int(rng.integers(4, 18)),
                label_points=0.0, ppg=usage * 0.9 + rng.normal(0, 2),
            ))
    df = pd.DataFrame(rows)
    df["label_points"] = df["ppg"] * df["label_games"]
    return df


def test_feature_columns_excludes_metadata_and_label():
    df = _assembled()
    cols = feature_columns(df)
    assert "ppg" not in cols and "label_points" not in cols and "player_id" not in cols
    assert set(cols) <= (set(df.columns) - NON_FEATURE_COLS)
    assert "targets_pg" in cols and "rec_yd_pg" in cols


def test_config_is_swappable_without_touching_train_one():
    df = _assembled()
    base = ModelConfig("WR")
    # a different target column + objective is a config change only
    df["ppg_alt"] = df["ppg"] * 1.5
    alt = base.with_(target="ppg_alt", objective="regression", n_estimators=50)
    m1 = train_one(df, base.with_(n_estimators=50))
    m2 = train_one(df, alt)
    assert m1.mean_model.predict(df[m1.features]).mean() == pytest.approx(
        m2.mean_model.predict(df[m2.features]).mean() / 1.5, rel=0.25
    )


def test_walk_forward_trains_only_on_earlier_seasons():
    df = _assembled(n_seasons=6)
    cfg = ModelConfig("WR", n_estimators=40, min_train_seasons=3)
    metrics, oof = walk_forward(df, cfg, eval_seasons=[2019, 2020, 2021])
    assert list(metrics["season"]) == [2019, 2020, 2021]
    assert (metrics["n"] > 0).all()
    # signal recoverable on synthetic data where ppg is ~linear in usage
    assert metrics["spearman"].mean() > 0.5
    assert set(oof["target_season"]) == {2019, 2020, 2021}


def test_min_label_games_filters_training_rows_not_inference():
    df = _assembled()
    df.loc[df.index[:50], "label_games"] = 1        # below the floor
    cfg = ModelConfig("WR", min_label_games=4, min_feature_games=0, n_estimators=30)
    model = train_one(df, cfg)
    # still predicts for every row, including the filtered-out ones
    preds = model.predict(df)
    assert len(preds) == len(df) and preds["pred_mean"].notna().all()
