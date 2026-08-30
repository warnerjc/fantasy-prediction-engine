# /models — training + inference

Six per-position models (QB/RB/WR/TE/K/DEF), LightGBM, **config-driven** and
**walk-forward validated**. Consumes `/features` rows and `/scoring`-computed
labels. Produces `Prediction(mean, p10, p50, p90)` — v1 fills `mean`, quantiles
stay `None`.

```
python -m models.build --league sleeper            # train + validate + project latest+1
python -m models.build --league yahoo --project 2025
```

Writes `models/output/<league>_projections.csv` (ranked, with `proj_ppg` and a
naive `proj_points`) and `<league>_walkforward.csv` (per-held-out-season metrics).

## Flow

```
player_week_stats ─┐
kicking_stats      ├─ season_labels(rules) ─► points / games / ppg  per player-season
team_defense_stats ┘        (league-specific; uses the same scoring.score as the app)

/features season_feature_matrix / kicker_ / defense_  ─► X per (player, target_season)

  assemble_position(X, labels)  ─►  walk_forward(...)          ─► per-season metrics
                                └►  project_position(target)   ─► ranked projection
```

Features are **league-agnostic** (usage); only the label changes per league, so a
per-league model is a relabel + retrain (seconds) — that's why `--league` is a
build flag, not a separate codebase.

## Config-driven (`config.py`)

`ModelConfig` holds everything the training loop varies: `target` column,
`objective` (`tweedie` for non-negative offense/K, `regression` for DEF whose PPG
can go negative), tree params, the training-row filters, and `quantiles`
(empty in v1). The invariant test: **v2's weekly quantile model is a new
`ModelConfig` (`target="week_points"`, `quantiles=(.1,.5,.9)`, weekly split), not
an edit to `pipeline.train_one`.**

`min_feature_games` / `min_label_games` filter *training* rows (a prior season
with 2 games carries no signal; a target season with 2 games is a noisy label) —
they never filter inference. Training rows are also sample-weighted by
`label_games`, so a 6-game season counts, just less than a 16-game one.

## Validation

`walk_forward` trains on `target_season < S`, predicts `S`, for each held-out `S`
with ≥ `min_train_seasons` prior. Metrics per season (never one blended number):
Spearman rank correlation (what matters for a draft board), MAE on PPG, and
top-N hit rate (N = 12 QB/TE/K/DEF, 24 RB/WR).

Current (2015–24 data, both sprint leagues): **QB/WR ρ≈0.67, RB≈0.58, TE≈0.60**,
top-N hit 0.52–0.66. **K and DEF ρ≈0.05–0.09** — near-noise, as expected;
year-over-year kicker/defense fantasy output is barely predictable. Draft K/DEF
late and stream them; the projections are there for completeness, not edge.

Gain-importance sanity check (leakage guard): WR led by `rec_yd_pg` then
`target_share` / `receptions_pg` / snap %; RB led by `off_snap_pct_mean` then
`rush_yd_pg` / `carries`. Opportunity dominates efficiency — the leakage
discipline in `/features` held.

## Backtest + baselines (`backtest.py`)

`python -m models.backtest --league sleeper [--season YYYY | YYYY-YYYY]`
(`bin/backtest`). Re-runs the walk-forward split per held-out season, joins
predictions to actual PPG, and grades the model against three cheap baselines:

- **`last_ppg`** — the player's PPG the prior season (naive persistence)
- **`ewma_ppg`** — recency-weighted mean of the prior two seasons (0.65 / 0.35)
- **`market_adp`** — preseason ADP for that season (FFC historical; rank metrics
  only, ADP isn't a PPG estimate)

Writes `output/<league>_baselines_<tag>.csv` (per season/position/method) and
`output/<league>_backtest_<tag>.csv` (per player, biggest rank misses first;
`--all` keeps the deep-bench churn, default is draftable range only).

**Finding (2020–25, both leagues):** on Spearman the model is a **wash with
`ewma_ppg`** for every skill position — QB 0.69 vs 0.67, WR 0.77 vs 0.77, and it
slightly *trails* the naive baseline for RB (0.73 vs 0.74) and TE (0.68 vs 0.71).
`market_adp` has lower rank ρ but the **best top-N hit rate everywhere** (RB 0.74,
TE 0.76 vs the model's 0.69 / 0.53) — the crowd is better at identifying *which*
players finish top-tier. For **DEF** the model (ρ≈0.05) is clearly *worse* than
just using `ewma_ppg` (ρ≈0.25). Takeaways: the ADP blend in the draft board
(`draft_tool._blend_market`) is earning its keep; a v2 model that can't beat
`ewma_ppg` on held-out seasons isn't worth shipping over it; consider dropping the
DEF model for a straight prior-year table. Roadmap to actually beat ADP:
`specifications/draft-sprint-plan.md` → Appendix A.

## Team-change features (walk-forward A/B)

`changed_team` + new-team environment + vacated opportunity share (see
`features/team_change.py`) gave a small net gain for **RB/WR** (WR ρ 0.69→0.70,
RB 0.58→0.59, MAE down ~0.03) and a small drag on **QB/TE**, so the QB/TE configs
set `exclude_feature_prefixes` to drop them. They did **not** rescue the marquee
single-season breakouts (Saquon proj RB19 → finished RB1): last year's production
dominates the split gain and the historical sample of "vet RB → better spot" is
too thin. Genuinely hard; not a tuning fix.

## Known v1 gaps

- **True rookies** (no prior NFL season) can't be projected by a prior-usage
  model — they fall out of the matrix. Needs a draft-capital baseline or ADP
  slot-in at the application layer.
- **Games-played** is a flat per-position assumption (`_games_assumption`), not a
  durability model — fine for ranking, rough for projected season totals.
- **Quantiles** are `None` (v1 mean-only by design).
