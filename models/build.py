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
)
from features.window import Window

from .config import DEFAULT_CONFIGS
from .labels import season_labels
from .leagues import load_rules
from .pipeline import assemble_position, project_position, walk_forward

OUT_DIR = Path(__file__).resolve().parent / "output"
OFFENSE = ("QB", "RB", "WR", "TE")


def _load_tables() -> dict[str, pd.DataFrame]:
    return {t: read_sql(f"SELECT * FROM {t}") for t in
            ("player_week_stats", "snap_counts", "player_ids", "team_week",
             "kicking_stats", "team_defense_stats")}


def _feature_matrix(position: str, tbl: dict, target_seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in target_seasons:
        if position in OFFENSE:
            fm = season_feature_matrix(tbl["player_week_stats"], tbl["snap_counts"],
                                       tbl["player_ids"], s, Window.prior_season(n_seasons=2))
            fm = fm[fm["most_recent_pos"] == position] if not fm.empty else fm
        elif position == "K":
            fm = kicker_feature_matrix(tbl["kicking_stats"], tbl["team_week"], s,
                                       Window.prior_season(n_seasons=2))
        else:  # DEF
            fm = defense_feature_matrix(tbl["team_defense_stats"], tbl["team_week"], s,
                                        Window.prior_season(n_seasons=2))
        if not fm.empty:
            frames.append(fm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(league: str, project_season: int | None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tbl = _load_tables()
    rules = load_rules(league)
    labels = season_labels(tbl["player_week_stats"], tbl["kicking_stats"],
                           tbl["team_defense_stats"], rules)

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
    args = ap.parse_args()
    run(args.league, args.project)


if __name__ == "__main__":
    main()
