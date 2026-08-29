# /features — model-ready features from player_week_stats

Every feature function is parameterized by an **as-of point** (`AsOf(season, week)`
— the game about to be played) and a **window** (`Window` — how far back to look).
v1 calls them with `Window.prior_season()`; v2 will call the *same* functions with
`Window.trailing(n_games=4)`. No feature has a v1-only / v2-only fork.

## Leakage rule

A feature "as of" `(season, week)` reads outcome stats **only from strictly
earlier weeks**. `visible_weeks()` does that filtering once; feature functions
consume its output and never re-filter. Pre-game-known context (opponent,
home/away, rest, Vegas line, reported injury status) is allowed for the as-of
week itself — that's what `context_features` uses.

## The engine — `window.py`

| | |
|---|---|
| `AsOf(season, week=1)` | the game being predicted; nothing from `>= (season, week)` is visible |
| `Window.prior_season(n_seasons=1)` | every REG week of the N seasons before the as-of season (v1) |
| `Window.trailing(n_games)` | the last N games *each player actually played* before the as-of point, crossing the season boundary (v2) |
| `visible_weeks(df, as_of, window, ...)` | rows of `df` visible at `as_of` per `window`; adds `week_index` (`season*100 + week`, orderable). Works on `player_week_stats` (`season_type`) and the nflverse raw tables (`season_type_col="game_type"`). |

## Feature functions

| function | input | output (one row per…) | category |
|---|---|---|---|
| `opportunity_features(visible)` | `visible_weeks(pws, …)` | `player_id` — usage totals + `_pg` rates + `target_share` / `rush_share` / `air_yards_share` (team-share, per-week mean) + `wopr` + efficiency (`yards_per_target`, …) + `most_recent_team` / `most_recent_pos` | outcome |
| `snap_features(visible_snaps)` | `visible_weeks(snap_counts, …, season_type_col="game_type")` | `gsis_id` — `off_snap_pct_mean` / `_last` / `_max`, `st_snap_pct_mean`, `snap_games` | outcome |
| `opponent_allowed_features(visible)` | `visible_weeks(pws, …)` | `defense_team` — `def_<QB\|RB\|WR\|TE>_<stat>_pg` allowed, `def_games` | outcome (opposing scorers) |
| `context_features(team_week, as_of)` | `team_week` | `team` — `opponent`, `is_home`, `rest`, `implied_total`, `team_spread`, `div_game`, `is_dome` / `is_outdoors`, `short_week` | pre-game-known |
| `identity_features(player_ids, as_of)` | `player_ids` | `player_id` — `age`, `years_exp`, `is_rookie`, `draft_round`, `draft_ovr`, `undrafted` | pre-game-known |
| `team_change_features(pws, seasonal_rosters, team_week, target_season, window=None)` | `player_week_stats` + `seasonal_rosters` + `team_week` | `player_id` — `changed_team`, new team's prior pass ratio / implied total, `vacated_tgt_share_new_team` / `vacated_rush_share_new_team` (share of the new team's prior opportunity held by players who left) | pre-game-known (offseason moves settle before Week 1) |
| `kicker_feature_matrix(kicking_stats, team_week, target_season, window=None)` | `kicking_stats` + `team_week` | `player_id` (kicker) — prior FG made/att per game, FG%, 50+/game, `xp_made_pg`, + team implied-total proxy | outcome + pre-game-known |
| `defense_feature_matrix(team_defense_stats, team_week, target_season, window=None)` | `team_defense_stats` + `team_week` | `player_id` (team) — prior `dst_*_pg`, `takeaways_pg_prior`, + team spread/implied-total proxy | outcome + pre-game-known |

## Assembly — `build.py`

- `season_feature_matrix(pws, snap_counts, player_ids, target_season, window=None)` —
  the **v1 draft entry point**. For predicting season S: `AsOf(S, 1)` +
  `Window.prior_season()`, one row per skill player (QB/RB/WR/TE) built from their
  S-1 usage + priors. Keyed `(player_id, target_season)`. **No label** — the model
  layer joins the scored target.
- `training_frame(…, target_seasons)` — `season_feature_matrix` stacked over
  several seasons = the X matrix for walk-forward training; the model splits on
  `target_season`.

Pass `seasonal_rosters` + `team_week` to `season_feature_matrix` to add
team-change features (a walk-forward A/B showed they help RB/WR: WR ρ 0.69→0.70,
RB 0.58→0.59, both MAE down slightly; they slightly hurt QB/TE, so the QB/TE
model configs exclude them). They did **not** fix the big single-season breakout
misses (Saquon RB19→RB1) — prior-year production still dominates the trees and
there aren't enough historical "veteran RB → better situation" examples to learn
the pattern. Open problem.

`context_features` / `opponent_allowed_features` are **not** folded into
`season_feature_matrix` — they're per-game and belong to the v2 weekly matrix.
They're built and as-of-parameterized now so v2 is a call-site change, not new code.

## Not here yet

- **Red-zone touches / carries inside the 10** — needs the play-by-play pull
  (deferred; see `data/README.md`).
- **Strength-of-schedule for the season projection** — could average
  `opponent_allowed` over a team's S-1 slate; not built for the sprint.
- **Injury-report features** (`starter_confidence`, etc.) — v2 adjustment layer.
