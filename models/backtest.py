"""Repeatable projected-vs-actual backtest + naive / market baselines.

    python -m models.backtest --league sleeper --season 2024
    python -m models.backtest --league yahoo --season 2020-2025
    python -m models.backtest --league sleeper                 # all eval seasons

For each held-out season S the model is trained on ``target_season < S`` and asked
to predict S -- the *same* walk-forward split ``models.build`` validates on, so
these numbers line up with ``<league>_walkforward.csv``. Predictions are joined to
actual PPG and scored against three cheap baselines:

    last_ppg   the player's PPG the previous season (naive persistence)
    ewma_ppg   recency-weighted mean of the prior two seasons
    market_adp preseason ADP for season S (fantasyfootballcalculator, historical)

The point is honesty about model value: if ``last_ppg`` or ``market_adp`` is within
a hair of the model on rank correlation, the LightGBM layer is not adding much and
the draft board should lean harder on the market blend.

Writes:
    models/output/<league>_backtest_<season>.csv   per player, biggest rank misses first
    models/output/<league>_baselines_<tag>.csv     model vs baselines, per season/position
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# the ADP baseline reuses the app layer's cached FFC client rather than
# re-implementing the HTTP + on-disk cache. backtest.py is a diagnostic, not part
# of the train/project path, so this one-way dependency is acceptable.
from applications.adp import fetch_adp, normalize_name

from .build import OFFENSE, _feature_matrix, _load_tables
from .config import DEFAULT_CONFIGS
from .labels import season_labels
from .leagues import load_rules
from .pipeline import _metrics, assemble_position, walk_forward

OUT_DIR = Path(__file__).resolve().parent / "output"

# fantasyfootballcalculator format closest to each league's scoring (mirrors the
# map in applications/adp.py). Historical ADP boards barely move between 10 and 12
# teams, so a single --adp-teams knob is enough.
_ADP_FORMAT = {"sleeper": "half-ppr", "yahoo": "ppr"}
_EWMA_WEIGHTS = (0.65, 0.35)   # prior season, season before that
_MIN_LABEL_GAMES = 4           # a <4-game season is too noisy to grade against


# --- season slices -------------------------------------------------------------

def _labels_by_pos_season(labels: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Labels + a normalized name for ADP matching. One row per player/season/pos."""
    out = labels.merge(names, on="player_id", how="left")
    out["name"] = out["name"].fillna(out["player_id"])       # DEF: id is the team abbr
    out["norm_name"] = out["name"].map(normalize_name)
    return out


def _actuals(labels: pd.DataFrame, season: int, position: str) -> pd.DataFrame:
    a = labels[(labels["season"] == season) & (labels["position"] == position)]
    a = a[a["games"] >= _MIN_LABEL_GAMES]
    return a.set_index("player_id")[["name", "norm_name", "ppg", "games"]]


def _prior_ppg(labels: pd.DataFrame, season: int, position: str, back: int) -> pd.Series:
    p = labels[(labels["season"] == season - back) & (labels["position"] == position)]
    return p.set_index("player_id")["ppg"]


def _ewma_ppg(labels: pd.DataFrame, season: int, position: str) -> pd.Series:
    s1 = _prior_ppg(labels, season, position, 1)
    s2 = _prior_ppg(labels, season, position, 2)
    w1, w2 = _EWMA_WEIGHTS
    df = pd.concat({"s1": s1, "s2": s2}, axis=1)
    # players with only s1 fall back to s1 alone; drop players with neither
    df["ewma"] = np.where(
        df["s2"].notna(),
        df["s1"].fillna(df["s2"]) * w1 + df["s2"] * w2,
        df["s1"],
    )
    return df["ewma"].dropna()


def _adp_scores(league: str, adp_teams: int, season: int, position: str,
                actuals: pd.DataFrame) -> pd.Series:
    """-ADP (higher = drafted earlier), indexed by player_id, matched on name."""
    try:
        adp = fetch_adp(_ADP_FORMAT.get(league, "half-ppr"), adp_teams, season)
    except Exception as e:                                    # network / API hiccup
        print(f"    (ADP for {season} unavailable: {e})")
        return pd.Series(dtype=float)
    if adp.empty:
        return pd.Series(dtype=float)
    adp = adp[adp["position"] == position][["norm_name", "adp"]]
    m = actuals.reset_index().merge(adp, on="norm_name", how="left")
    return m.set_index("player_id")["adp"].mul(-1.0).dropna()


# --- per-season evaluation ----------------------------------------------------

def _oof_for_season(assembled: pd.DataFrame, config, season: int) -> pd.DataFrame:
    _, oof = walk_forward(assembled, config, [season])
    return oof


def _grade(name: str, pred: pd.Series, actual: pd.Series, top_n: int,
           season: int, position: str, rank_only: bool = False) -> dict:
    m = _metrics(pred, actual, top_n)
    hit_key = f"top{top_n}_hit"
    return {
        "season": season, "position": position, "method": name,
        "n": m["n"], "spearman": m["spearman"],
        "mae": np.nan if rank_only else m["mae"],   # ADP isn't in PPG units
        "topN_hit": m[hit_key],
    }


def run(league: str, seasons: list[int] | None, adp_teams: int, all_players: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tbl = _load_tables()
    rules = load_rules(league)
    labels = season_labels(tbl["player_week_stats"], tbl["kicking_stats"],
                           tbl["team_defense_stats"], rules, drop_final_week=True)
    names = tbl["player_ids"][["gsis_id", "name"]].rename(columns={"gsis_id": "player_id"})
    lab = _labels_by_pos_season(labels, names)

    data_seasons = sorted(tbl["player_week_stats"]["season"].unique())
    first_target = data_seasons[0] + 2
    # a walk-forward season needs min_train_seasons of prior targets before it
    min_train = max(c.min_train_seasons for c in DEFAULT_CONFIGS.values())
    eval_pool = [s for s in data_seasons if s >= first_target + min_train]
    seasons = seasons or eval_pool
    seasons = [s for s in seasons if s in set(data_seasons)]
    if not seasons:
        print("no backtestable seasons in that range"); return
    target_seasons = list(range(first_target, max(seasons) + 1))

    metric_rows: list[dict] = []
    per_player: list[pd.DataFrame] = []

    for pos, config in DEFAULT_CONFIGS.items():
        fm = _feature_matrix(pos, tbl, target_seasons)
        if fm.empty:
            continue
        assembled = assemble_position(fm, labels, pos)
        top_n = {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "K": 12, "DEF": 12}[pos]

        for s in seasons:
            oof = _oof_for_season(assembled, config, s)
            if oof.empty:
                continue
            act = _actuals(lab, s, pos)
            if act.empty:
                continue
            actual = act["ppg"]

            model_pred = oof.set_index("player_id")["pred_mean"].reindex(actual.index)
            last = _prior_ppg(lab, s, pos, 1).reindex(actual.index)
            ewma = _ewma_ppg(lab, s, pos).reindex(actual.index)
            adp = _adp_scores(league, adp_teams, s, pos, act).reindex(actual.index)

            metric_rows.append(_grade("model", model_pred, actual, top_n, s, pos))
            metric_rows.append(_grade("last_ppg", last, actual, top_n, s, pos))
            metric_rows.append(_grade("ewma_ppg", ewma, actual, top_n, s, pos))
            metric_rows.append(_grade("market_adp", adp, actual, top_n, s, pos, rank_only=True))

            pp = pd.DataFrame({
                "season": s, "position": pos,
                "name": act["name"], "team_games": act["games"],
                "proj_ppg": model_pred, "actual_ppg": actual,
                "last_ppg": last,
            }).reset_index()
            pp["ppg_error"] = pp["proj_ppg"] - pp["actual_ppg"]
            pp["proj_pos_rank"] = pp["proj_ppg"].rank(ascending=False, method="min")
            pp["actual_pos_rank"] = pp["actual_ppg"].rank(ascending=False, method="min")
            pp["rank_delta"] = pp["proj_pos_rank"] - pp["actual_pos_rank"]
            # a miss only cost you if the player was draftable on projection or
            # actually finished draftable -- ignore the deep-bench rank churn
            pp["draftable"] = (pp["proj_pos_rank"] <= 2 * top_n) | (pp["actual_pos_rank"] <= 2 * top_n)
            per_player.append(pp)

    if not metric_rows:
        print("nothing to grade -- check that models.build runs for this league")
        return

    metrics = pd.DataFrame(metric_rows)
    _print_summary(metrics)

    tag = f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
    mpath = OUT_DIR / f"{league}_baselines_{tag}.csv"
    metrics.to_csv(mpath, index=False)

    players = pd.concat(per_player, ignore_index=True)
    if not all_players:
        players = players[players["draftable"]]
    players = players.sort_values(
        ["season", "rank_delta"], key=lambda c: c.abs() if c.name == "rank_delta" else c,
        ascending=[True, False],
    )
    cols = ["season", "position", "name", "team_games", "proj_ppg", "actual_ppg",
            "ppg_error", "last_ppg", "proj_pos_rank", "actual_pos_rank", "rank_delta",
            "draftable"]
    ppath = OUT_DIR / f"{league}_backtest_{tag}.csv"
    players[cols].round(2).to_csv(ppath, index=False)
    print(f"\nper-season baseline metrics -> {mpath}")
    print(f"per-player projected-vs-actual (biggest rank misses first) -> {ppath}")


def _print_summary(metrics: pd.DataFrame) -> None:
    """Model vs baselines, averaged over seasons, per position; then the pooled row."""
    piv = (metrics.groupby(["position", "method"])[["spearman", "mae", "topN_hit"]]
           .mean().reset_index())
    order = {"model": 0, "last_ppg": 1, "ewma_ppg": 2, "market_adp": 3}
    piv = piv.sort_values(["position", "method"], key=lambda c: c.map(order).fillna(9)
                          if c.name == "method" else c)

    n_seasons = metrics["season"].nunique()
    print(f"\n{'pos':>4} {'method':>11} {'rho':>7} {'MAE':>7} {'topN':>6}   "
          f"(mean over {n_seasons} season{'s' if n_seasons != 1 else ''})")
    print("-" * 48)
    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        sub = piv[piv["position"] == pos]
        if sub.empty:
            continue
        base_rho = sub[sub["method"] == "model"]["spearman"].iloc[0]
        for r in sub.itertuples():
            if pd.isna(r.spearman) and pd.isna(r.topN_hit):
                continue                                    # method had no coverage (e.g. DST ADP)
            mark = ""
            if r.method != "model" and pd.notna(r.spearman) and r.spearman >= base_rho - 0.02:
                mark = "  <- matches model"
            mae = f"{r.mae:6.2f}" if pd.notna(r.mae) else "     ."
            rho = f"{r.spearman:7.3f}" if pd.notna(r.spearman) else "      ."
            print(f"{pos:>4} {r.method:>11} {rho} {mae} {r.topN_hit:6.2f}{mark}")
        print()


def _parse_season(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=["sleeper", "yahoo"])
    ap.add_argument("--season", default=None,
                    help="YYYY, or YYYY-YYYY range; default: all walk-forwardable seasons")
    ap.add_argument("--adp-teams", type=int, default=12,
                    help="team count for the historical ADP baseline (default 12)")
    ap.add_argument("--all", action="store_true",
                    help="keep deep-bench players in the per-player CSV (default: draftable range only)")
    args = ap.parse_args()
    run(args.league, _parse_season(args.season), args.adp_teams, args.all)


if __name__ == "__main__":
    main()
