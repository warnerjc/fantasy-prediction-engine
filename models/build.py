"""Train + validate + project all six position models for a league.

    python -m models.build --league sleeper
    python -m models.build --league yahoo --project 2025

Walk-forward metrics per held-out season are printed; ranked projections for the
target season are written to models/output/<league>_projections.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.db import read_sql
from features import (
    defense_feature_matrix,
    kicker_feature_matrix,
    season_feature_matrix,
    week_defense_matrix,
    week_feature_matrix,
    week_kicker_matrix,
)
from features.window import Window

from .config import DEFAULT_CONFIGS, WEEKLY_CONFIGS
from .labels import season_labels, week_labels
from .leagues import load_rules
from .pipeline import assemble_position, project_position, walk_forward

OUT_DIR = Path(__file__).resolve().parent / "output"
OFFENSE = ("QB", "RB", "WR", "TE")


def _load_tables() -> dict[str, pd.DataFrame]:
    return {t: read_sql(f"SELECT * FROM {t}") for t in
            ("player_week_stats", "snap_counts", "player_ids", "team_week",
             "kicking_stats", "team_defense_stats", "seasonal_rosters")}


def _feature_matrix(position: str, tbl: dict, target_seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in target_seasons:
        if position in OFFENSE:
            fm = season_feature_matrix(tbl["player_week_stats"], tbl["snap_counts"],
                                       tbl["player_ids"], s, Window.prior_season(n_seasons=2, drop_final_week=True),
                                       seasonal_rosters=tbl["seasonal_rosters"],
                                       team_week=tbl["team_week"])
            fm = fm[fm["most_recent_pos"] == position] if not fm.empty else fm
        elif position == "K":
            fm = kicker_feature_matrix(tbl["kicking_stats"], tbl["team_week"], s,
                                       Window.prior_season(n_seasons=2, drop_final_week=True))
        else:  # DEF
            fm = defense_feature_matrix(tbl["team_defense_stats"], tbl["team_week"], s,
                                        Window.prior_season(n_seasons=2, drop_final_week=True))
        if not fm.empty:
            frames.append(fm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _weekly_offense_matrix(tbl: dict, target_seasons: list[int], window: Window) -> pd.DataFrame:
    """week_feature_matrix stacked over seasons, all skill positions at once
    (the trailing window / opportunity shares need every player, so building it
    per-position would just repeat the work)."""
    frames = []
    for s in target_seasons:
        fm = week_feature_matrix(tbl["player_week_stats"], tbl["snap_counts"],
                                 tbl["player_ids"], tbl["team_week"], s, window)
        if not fm.empty:
            frames.append(fm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _weekly_special_matrix(pos: str, tbl: dict, target_seasons: list[int], window: Window) -> pd.DataFrame:
    fn = week_kicker_matrix if pos == "K" else week_defense_matrix
    src = "kicking_stats" if pos == "K" else "team_defense_stats"
    frames = [fn(tbl[src], tbl["team_week"], s, window) for s in target_seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_weekly(league: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tbl = _load_tables()
    rules = load_rules(league)
    wk_lab = week_labels(tbl["player_week_stats"], tbl["kicking_stats"],
                         tbl["team_defense_stats"], rules)

    seasons = sorted(tbl["player_week_stats"]["season"].unique())
    target_seasons = list(range(seasons[0] + 1, seasons[-1] + 1))   # trailing window crosses the boundary
    eval_seasons = target_seasons                                   # walk_forward skips the under-trained ones

    # skill matrix once, K/DEF windows differ so build them per position
    skill_window = Window.trailing(
        n_games=WEEKLY_CONFIGS["WR"].feature_window_games, max_seasons_back=2)
    off_fm = _weekly_offense_matrix(tbl, target_seasons, skill_window)

    all_metrics = []
    for pos, config in WEEKLY_CONFIGS.items():
        if pos in OFFENSE:
            fm = off_fm[off_fm["position"] == pos] if not off_fm.empty else off_fm
        else:
            fm = _weekly_special_matrix(pos, tbl, target_seasons,
                                        Window.trailing(n_games=config.feature_window_games,
                                                        max_seasons_back=2))
        if fm is None or fm.empty:
            print(f"{pos}: no weekly features, skipping")
            continue
        assembled = assemble_position(fm, wk_lab, pos, grain="week")
        metrics, _ = walk_forward(assembled, config, eval_seasons)
        if metrics.empty:
            continue
        all_metrics.append(metrics)
        avg = metrics[["spearman", "mae", "coverage_80", "pinball_p50"]].mean()
        hit = metrics[[c for c in metrics.columns if c.startswith("top")]].mean().mean()
        print(f"{pos:>4}  seasons={len(metrics)}  rho={avg['spearman']:.3f}  "
              f"MAE={avg['mae']:.2f}  topN_hit={hit:.2f}  "
              f"cover80={avg['coverage_80']:.2f}  pinball={avg['pinball_p50']:.2f}")

    if all_metrics:
        path = OUT_DIR / f"{league}_weekly_walkforward.csv"
        pd.concat(all_metrics, ignore_index=True).to_csv(path, index=False)
        print(f"\nper-held-out-season weekly metrics -> {path}")


def run(league: str, project_season: int | None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tbl = _load_tables()
    rules = load_rules(league)
    labels = season_labels(tbl["player_week_stats"], tbl["kicking_stats"],
                           tbl["team_defense_stats"], rules, drop_final_week=True)

    seasons = sorted(tbl["player_week_stats"]["season"].unique())
    first_target = seasons[0] + 2                      # need 2 prior seasons of features
    project_season = project_season or seasons[-1] + 1
    target_seasons = list(range(first_target, project_season + 1))
    eval_seasons = [s for s in target_seasons if s <= seasons[-1]]

    all_metrics, all_projections = [], []
    for pos, config in DEFAULT_CONFIGS.items():
        fm = _feature_matrix(pos, tbl, target_seasons)
        if fm.empty:
            print(f"{pos}: no features, skipping")
            continue
        assembled = assemble_position(fm, labels, pos)

        metrics, _ = walk_forward(assembled, config, eval_seasons)
        all_metrics.append(metrics)
        if not metrics.empty:
            avg = metrics[["spearman", "mae"]].mean()
            hit = metrics[[c for c in metrics.columns if c.startswith("top")]].mean().mean()
            print(f"{pos:>4}  seasons={len(metrics)}  spearman={avg['spearman']:.3f}  "
                  f"MAE={avg['mae']:.2f}  topN_hit={hit:.2f}")

        proj = project_position(assembled, config, project_season)
        if not proj.empty:
            all_projections.append(proj)

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(OUT_DIR / f"{league}_walkforward.csv", index=False)
    if all_projections:
        out = pd.concat(all_projections, ignore_index=True)
        names = tbl["player_ids"][["gsis_id", "name"]].rename(columns={"gsis_id": "player_id"})
        out = out.merge(names, on="player_id", how="left")
        out["name"] = out["name"].fillna(out["player_id"])   # DEF: player_id is the team
        out = out[["pos_rank", "position", "name", "most_recent_team", "proj_ppg",
                   "proj_points", "pred_p10", "pred_p50", "pred_p90", "player_id",
                   "target_season"]]
        path = OUT_DIR / f"{league}_projections.csv"
        out.to_csv(path, index=False)
        print(f"\nprojections for {project_season} -> {path}  ({len(out)} players)")
        for pos in DEFAULT_CONFIGS:
            top = out[out["position"] == pos].head(5)
            if not top.empty:
                shown = ", ".join(f"{r.name} ({r.proj_ppg:.1f})" for r in top.itertuples())
                print(f"  {pos:>4}: {shown}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=["sleeper", "yahoo"])
    ap.add_argument("--project", type=int, default=None, help="season to project (default: latest+1)")
    ap.add_argument("--grain", choices=["season", "week"], default="season",
                    help="season = v1 draft projection; week = v2 start/sit walk-forward")
    args = ap.parse_args()
    if args.grain == "week":
        run_weekly(args.league)
    else:
        run(args.league, args.project)


if __name__ == "__main__":
    main()
