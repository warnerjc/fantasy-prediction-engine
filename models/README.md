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
can go negative), `grain` (`"season"` v1 / `"week"` v2), `feature_window_games`
(the trailing window for weekly), tree params, the training-row filters, and
`quantiles` (empty in v1). The invariant held: **v2's weekly quantile model is
`WEEKLY_CONFIGS` — `target="week_points"`, `grain="week"`, `quantiles=(.1,.5,.9)`
— not an edit to `pipeline.train_one`.**

`min_feature_games` / `min_label_games` filter *training* rows (a prior season
with 2 games carries no signal; a target season with 2 games is a noisy label) —
they never filter inference. Training rows are also sample-weighted by
`label_games`, so a 6-game season counts, just less than a 16-game one.

## Validation

`walk_forward` trains on `target_season < S`, predicts `S`, for each held-out `S`
with ≥ `min_train_seasons` prior. Metrics per season (never one blended number):
Spearman rank correlation (what matters for a draft board), MAE on PPG, and
top-N hit rate (N = 12 QB/TE/K/DEF, 24 RB/WR).

Current (2015–25 data, walk-forward 2020–25, both sprint leagues):
**WR ρ≈0.75, RB≈0.73, TE≈0.70, QB≈0.66**, top-N hit 0.53–0.66. **K and DEF
ρ≈0.10** — near-noise, as expected; year-over-year kicker/defense fantasy output
is barely predictable. Draft K/DEF late and stream them; the projections are there
for completeness, not edge.

Labels and prior-season feature windows span **weeks 1 through (final REG week −
1)** — 1–16 pre-2021, 1–17 from 2021 (`Window(drop_final_week=True)` /
`season_labels(drop_final_week=True)`, opted in at the v1 call sites). The final
NFL week is played after every fantasy championship by locked-seed teams resting
starters — fantasy-dead and distorted. Excluding it lifted **TE ρ ~+0.03** and
**DEF ~+0.06** (the low-volume positions a single weird week swings most), cost
QB/WR ~0.02, and left the 2026 draftable-range board essentially unchanged
(rank ρ 0.98 vs the full-REG label).

Gain-importance sanity check (leakage guard): WR led by `rec_yd_pg` then
`target_share` / `receptions_pg` / snap %; RB led by `off_snap_pct_mean` then
`rush_yd_pg` / `carries`. Opportunity dominates efficiency — the leakage
discipline in `/features` held.

## v2 weekly model (`--grain week`)

`bin/refresh-models --league <l> --grain week` runs the weekly walk-forward for
all six positions and writes `<league>_weekly_walkforward.csv`. It's a **new
config, not a new training path**: `WEEKLY_CONFIGS` sets `target="week_points"`,
`grain="week"`, `feature_window_games` (6 skill / 8 K-DEF), `quantiles=(.1,.5,.9)`;
`pipeline.train_one` is unchanged. Features come from
`features.week_feature_matrix` (trailing per-player window + per-game Vegas /
opponent / venue context); labels from `models.week_labels` (same `scoring.score`
path as `season_labels`, at week grain). Split is still walk-forward by season —
train on all weeks of `target_season < S`, score every player-week of `S`.

Metrics are player-week grain: rank ρ / MAE pooled over player-weeks, top-N hit
averaged per week, and — from the real quantile models — empirical p10–p90
**coverage** and p50 **pinball loss**.

Walk-forward 2019–25, both leagues (sleeper / yahoo):

| pos | model ρ | MAE | topN | p10–p90 cover |
|---|---|---|---|---|
| QB  | 0.49 / 0.50 | 8.7 / 8.3 | 0.52 | 0.62 |
| RB  | 0.67 / 0.66 | 5.1 / 4.3 | 0.60 | 0.70 |
| WR  | 0.62 / 0.61 | 4.5 / 3.8 | 0.44 | 0.69 |
| TE  | 0.54 / 0.52 | 3.3 / 2.9 | 0.45 | 0.66 |
| K   | 0.10 / 0.10 | 3.9 / 4.4 | 0.44 | 0.73 |
| DEF | 0.30 / 0.29 | 5.9 / 4.4 | 0.52 | 0.75 |

## Weekly backtest — model vs rolling averages (`backtest.py --grain week`)

`bin/backtest --league <l> --grain week [--season …]`. Same walk-forward split,
graded against four cheap in-season baselines (each strictly from the player's
earlier games):

- **`trailing_mean`** — mean of the last N games (crosses the season boundary)
- **`trailing_ewma`** — span-N EWMA of prior games
- **`season_to_date`** — mean of this season's earlier weeks
- **`last_game`** — the previous game's points

Writes `<league>_weekly_baselines_<tag>.csv` and `<league>_weekly_backtest_<tag>.csv`
(per player-week, biggest model misses first).

**Verdict (2019–25, both leagues): ship it.** Unlike the season model — a wash
with `ewma_ppg` — the weekly model **beats `trailing_ewma` on rank ρ at every
position**: QB +0.02–0.03, WR +0.013, TE +0.012–0.018, K +0.03, and **DEF
+0.15** (0.30 vs 0.14 — the per-game context roughly doubles DEF rank
correlation, reversing the season-grain finding that the DEF model *lost* to
ewma). MAE is lower everywhere. **RB is the one wash** (0.665 vs 0.662), same as
season grain — last week's box score already carries the RB signal. `last_game`
alone is clearly the worst baseline — weekly variance is real, and a trailing
window matters.

**Open: quantile calibration.** p10–p90 coverage lands 0.62–0.75 vs the 0.80
nominal — the LightGBM quantile models run too narrow (worst at QB). Before the
intervals drive a start/sit confidence call they need widening (fit .05/.95 and
relabel, or a post-hoc conformal adjustment on a held-out slice). The mean
projection is trustworthy now; the spread is not yet.

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

**Finding (2020–25, both leagues, fantasy-relevant-weeks label):** on Spearman the
model is a **wash with `ewma_ppg`** for every skill position — QB 0.67 vs 0.65,
WR 0.76 vs 0.77, TE 0.71 vs 0.72, and it slightly *trails* on RB (0.73 vs 0.75).
`market_adp` has lower rank ρ but the **best top-N hit rate everywhere** (RB 0.74,
TE 0.78 vs the model's 0.67 / 0.56) — the crowd is better at identifying *which*
players finish top-tier. For **DEF** the model (ρ≈0.10) is still clearly *worse*
than just using `ewma_ppg` (ρ≈0.24). Takeaways: the ADP blend in the draft board
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
