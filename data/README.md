# /data — ingestion layer

Pulls raw stats + context into a local SQLite store (`data/nfl.db`, git-ignored).

```
python -m data.build --seasons 2015-2024     # full (re)build
python -m data.build --seasons 2025          # weekly in-season refresh
```

Every table has an explicit primary key and is written with `INSERT OR REPLACE`
(`data.db.upsert`), so re-running is safe and picks up nflverse stat corrections
(which land for ~2 weeks after each game). No append-only tables.

## Tables

| table | grain / PK | source | notes |
|---|---|---|---|
| `player_week_stats` | `(player_id, season, week, season_type)` | `nfl.import_weekly_data` | **The source of truth.** Offensive box-score production. `player_id` = nflverse `gsis_id`. `recent_team`→`team`, `opponent_team`→`opponent`. Season totals are a `GROUP BY` on this — never a second table. A player with no row for a week (bye/inactive/no production) is *absent*, not a zero row. |
| `snap_counts` | `(pfr_player_id, season, week, game_type, team)` | `nfl.import_snap_counts` | Off/def/ST snaps + pct. `gsis_id` attached via the `player_ids` crosswalk (>99% match for QB/RB/WR/TE; ~18% of *all* rows unmatched — mostly OL/DL not in the crosswalk). `team` is in the PK because PFR occasionally shares one id between two players. |
| `injuries` | `(gsis_id, season, week, game_type, team)` | `nfl.import_injuries` | Weekly practice/game report. Re-issued reports are deduped to the latest `date_modified`. `team` in PK handles mid-week trades. |
| `schedules` | `(game_id)` | `nfl.import_schedules` | One row per game. Carries rest days, roof/surface/`temp`/`wind`, and the closing Vegas `spread_line` / `total_line`. |
| `team_week` | `(season, week, game_type, team)` | derived from `schedules` | `schedules` exploded to one row per team with pre-game-known context only: `opponent`, `is_home`, `rest`, `div_game`, weather, `implied_total` (`total_line/2 ± spread_line/2`), `team_spread`. Safe for as-of-week features — no outcome columns. |
| `kicking_stats` | `(kicker_player_id, season, week, game_type, team)` | derived from `import_pbp_data` | Weekly kicker line: `fg_made`/`fg_missed` + by-distance buckets (`fg_made_0_19 … _50p`, same for missed; blocked counts as missed), `fg_made_yds`, `xp_made`/`xp_missed`. `kicker_player_id` is a gsis id. |
| `team_defense_stats` | `(defense_team, season, week, game_type)` | derived from `import_pbp_data` | Weekly DST line, already in canonical `dst_*` names: `dst_sack`, `dst_int`, `dst_fum_rec` (= offensive fumbles lost), `dst_safety`, `dst_td` (INT/fumble/return TDs the team scored), `dst_blk_kick`, `dst_pts_allowed`, `dst_yds_allowed`. 4th-down stops and defensive 2pt returns omitted (rare / low fantasy value). |
| `seasonal_rosters` | `(player_id, season)` | `nfl.import_seasonal_rosters` | Player's team + status + experience for a season. Team is pre-Week-1-known (free agency settles in the offseason), so `changed_team` features derived from it are not leakage. |
| `player_ids` | `(gsis_id)` | `nfl.import_ids` | Cross-reference: `gsis_id ↔ pfr_id ↔ sleeper_id ↔ yahoo_id ↔ espn_id`. Use this for every cross-source join — never join on name/team. Rows with a null `gsis_id` are dropped (a handful of never-active players). |

Play-by-play (`import_pbp_data`, ~50k rows/season × ~400 cols) is pulled during
each build and used to derive `kicking_stats` + `team_defense_stats`, but is **not
stored whole** — it's large and the two derived tables are all the sprint needs.
Landing raw (or red-zone-filtered) PBP for opportunity features is a later step.

## Not landed yet (out of scope for the draft sprint)

- **Raw / red-zone play-by-play** — PBP is pulled and aggregated (see above) but
  not stored row-level. Red-zone touches & carries-inside-10 for opportunity
  features need a stored (filtered) PBP table — a later step.
- **Vegas odds API / weather API** — `schedules` already carries closing
  spread/total and basic weather, enough for v1. Live odds movement is a v2
  weekly-tool concern.
- **Sleeper / Yahoo league data** (rosters, live draft state) — pulled by the
  application layer at draft time, not stored here.

## Refresh cadence

nflverse regenerates the current season's releases within a day of each game.
In-season: `python -m data.build --seasons <current>` weekly (Tue/Wed, after stat
corrections settle). Historical seasons are stable — no need to re-pull them.
