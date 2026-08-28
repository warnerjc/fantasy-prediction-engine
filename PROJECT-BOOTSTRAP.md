# Fantasy Football Prediction System — Project Summary

## Goal
Build a custom fantasy football prediction system for standard head-to-head (H2H) leagues (13/14/15-week seasons, Sleeper and Yahoo), hosted locally. Two end-user tools:

1. **Draft assistant** — ranks/recommends players during a live draft, given league settings, league size, draft position, and draft type (snake/auction).
2. **Weekly start/sit tool** — projects each rostered player's expected fantasy points (with floor/ceiling) for a given week, to support start/sit decisions.

Explicitly **not** in scope: DFS/best-ball optimization, ownership modeling, or anything requiring ceiling-obsessed lineup construction. This is for standard H2H roster management only.

## Key design decisions made so far

- **v1 (build first): season-grain draft tool.** Predicts season-total (or points-per-game) projections using prior-season rolling stats as features. Powers the draft assistant in time for upcoming drafts.
- **v2 (build later): weekly quantile model.** Predicts a full distribution of weekly points per player (e.g., p10/p25/p50/p75/p90) rather than a single number, since "highest score probability" is underspecified — the right metric depends on the decision (start/sit vs. streaming vs. protecting a lead). A distribution lets every downstream question be computed on top of it.
- **Architected so v1 → v2 requires no refactor**, by:
  - Always storing and engineering features at **weekly grain** (`player_week_stats` as the source of truth), even though v1 only predicts season totals. Season totals are a `GROUP BY` aggregation on top of the weekly table, not a separate data source.
  - Writing feature functions parameterized by an "as-of week / window" so v1 (full prior season window) and v2 (rolling 3-5 week window) reuse the same functions with different parameters.
  - Typing model predictions as a dict/object (`{mean, p10, p50, p90}`) from day one, even though v1 only populates `mean` — avoids touching downstream consumers when v2 adds quantiles.
  - Keeping model training config-driven (target column + loss/objective + temporal split strategy swappable) rather than hardcoded, so v2 is a new config, not new architecture.
  - Building a standalone `/scoring` module (raw stats → fantasy points, given a league's actual scoring settings pulled from the API) used identically by both training labels and the application layer — never hardcoding "standard PPR" assumptions.

## Planned architecture

```
/data          -> pipeline: pulls nflverse + Sleeper/Yahoo + injuries + Vegas lines into SQLite (weekly grain)
/features      -> feature functions operating on player-week rows, parameterized by as-of week/window
/scoring       -> raw stats -> fantasy points, given a league's scoring settings dict
/models        -> training + inference, parameterized by target column + objective + temporal split
/applications  -> draft_tool.py, weekly_tool.py (consumers of /models + /scoring)
```

**Core DB table:** `player_week_stats` (player, season, week, team, opponent, snaps, targets, carries, red zone touches, etc.) — the single source of truth; season-level views are aggregations on top of it, never a parallel table.

**Local hosting:** WSL2 + Docker Compose (via Rancher Desktop) as the runtime, SQLite (or Parquet) for storage, Windows Task Scheduler triggering scheduled container runs for weekly data refresh, a persistent Streamlit/Flask dashboard container for viewing projections. No cloud infra, no GPU needed — LightGBM trains in seconds-to-minutes on this data size. See `AGENTS.md` → Confirmed hosting decisions for the full reasoning.

## Modeling approach

- **Model type:** Gradient-boosted trees (LightGBM/XGBoost) — handle tabular sports data well, interpretable via SHAP, don't need massive data. One model per position (QB/RB/WR/TE/K/DEF), since scoring dynamics differ substantially.
- **v1 target:** season-total fantasy points (or points-per-game), trained on prior-season rolling stats.
- **v2 target:** weekly fantasy points, quantile/Tweedie/Negative Binomial loss to capture the right-skewed, zero-inflated nature of fantasy scoring and produce floor/ceiling, not just a mean.
- **Validation:** walk-forward by season (train on earlier years, test on a later held-out year) — never random shuffle, to avoid leaking future info into the past.
- **Features prioritize opportunity over efficiency:** targets, carries, red zone touches, snap %, target share (rolling windows), opponent defense allowed-by-position, Vegas implied team total, home/away, rest days.

## v2 addition (later): weekly adjustment layer

Sits between the base model output and the application layer — not a retrained model, but a rules layer that:
- Downweights or zeroes out players who are OUT/IR; widens the distribution (doesn't just move the median) for Questionable tags
- Swaps in backup player context when a starter is out
- Adjusts team implied total / re-runs affected players when Vegas lines move
- Optionally uses an LLM to parse beat-reporter/injury news into structured features (`starter_confidence`, `role_change_flag`) feeding this layer — a natural fit given prior GenAI coursework

## Data sources confirmed

- **nflverse (`nfl_data_py` / `nflreadpy` in Python)** — primary source, pulled via package functions rather than manual CSV downloads:
  - `stats_player` (weekly, offense) — core production stats: targets, receptions, yards, TDs, carries. Goes back to 1999, far more than the 5 years originally assumed.
  - Snap counts — separate nflverse release, needed for snap % (not in `stats_player`).
  - Injuries — separate nflverse release.
  - Schedules — opponent, home/away, rest days.
  - Red zone touches — not directly in `stats_player`; needs deriving from raw play-by-play filtered to red zone plays.
  - Opponent defense allowed-by-position — not provided directly; computed by aggregating `stats_player` by opposing team.
- **Sleeper API** — league settings, rosters, scoring (public, no auth) — build first, lower friction.
- **Yahoo API** — same purpose, OAuth-based, more friction — real API/OAuth integration still
  deferred, but a Yahoo league is supported sooner than that via a manually-entered scoring
  config (see `specifications/draft-sprint-plan.md`), since the ranking engine only needs a
  league's scoring settings, not a live API connection, to produce a correct draft ranking.
- **Vegas lines/odds** — external odds API (e.g., The Odds API) for implied team totals, spreads.
- **Weather** — NWS API, for outdoor stadiums.

## Immediate next step (in progress)

Write the data pipeline script: pull `stats_player` (weekly) plus snap counts, injuries, and schedules via `nfl_data_py`, land them into the SQLite `player_week_stats` schema described above. Then build the v1 season-grain model on top.

## User context
- Building for own use in standard H2H Sleeper/Yahoo leagues (13/14/15-week seasons).
- Has GenAI for Business Applications coursework (UT Austin) but limited hands-on since — gap is mainly classical/tabular ML (pandas, scikit-learn/XGBoost, sports feature engineering), not LLM application work.
- Wants the draft tool ready in time to test in upcoming drafts, but recognizes (correctly) that it depends on the projection model being built first, even in a lighter v1 form.
