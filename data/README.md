# /data — ingestion layer

Pulls raw stats + context into a local SQLite store (`data/nfl.db`, git-ignored)
via **`nflreadpy`** — the maintained nflverse Python client. (The older
`nfl_data_py` was used first but can't fetch the current season's `player_stats`
release; `nflreadpy` also carries kicker FG-by-distance and team-defense box
scores natively, so no play-by-play aggregation is needed.)

```
python -m data.build --seasons 2015-2025     # full (re)build
python -m data.build --seasons 2026           # weekly in-season refresh
```

Every table has an explicit primary key and is written with `INSERT OR REPLACE`
(`data.db.upsert`), so re-running is safe and picks up nflverse stat corrections
(which land for ~2 weeks after each game). No append-only tables. Full build is
~30s (no PBP pull).

## Tables

| table | grain / PK | source | notes |
|---|---|---|---|
| `player_week_stats` | `(player_id, season, week, season_type)` | `load_player_stats` (offense) | **The source of truth.** `player_id` = `gsis_id`. Renamed to keep stable column names: `opponent_team`→`opponent`, `passing_interceptions`→`interceptions`, `sacks_suffered`→`sacks`. Season totals are a `GROUP BY` on this — never a second table. A player with no row for a week (bye/inactive/no production) is *absent*, not a zero row. |
| `kicking_stats` | `(kicker_player_id, season, week, game_type, team)` | `load_player_stats` (K) | FG made/missed + native by-distance buckets (`fg_made_0_19 … _40_49`, plus `50_59`+`60_` combined into `fg_made_50p`), `fg_made_yds`, `xp_made`/`xp_missed`. |
| `team_defense_stats` | `(defense_team, season, week, game_type)` | `load_team_stats` + `schedules` | Canonical `dst_*` names: `dst_sack`, `dst_int`, `dst_fum_rec`, `dst_safety`, `dst_td`, `dst_blk_kick` (FG+PAT+punt blocks), `dst_yds_allowed` (opponent's offensive yards that game), `dst_pts_allowed` (opponent's final score). Special-teams return TDs and 4th-down stops omitted (rare / not in the release). |
| `snap_counts` | `(pfr_player_id, season, week, game_type, team)` | `load_snap_counts` | Off/def/ST snaps + pct. `gsis_id` attached via the `player_ids` crosswalk (>99% match for QB/RB/WR/TE; ~18% of *all* rows unmatched — mostly OL/DL). `team` in PK because PFR occasionally shares one id between two players. |
| `injuries` | `(gsis_id, season, week, game_type, team)` | `load_injuries` | Weekly practice/game report. Re-issued reports deduped to the latest `date_modified`. `team` in PK handles mid-week trades. |
| `schedules` | `(game_id)` | `load_schedules` | One row per game. Rest days, roof/surface/`temp`/`wind`, closing `spread_line` / `total_line`, final scores. |
| `team_week` | `(season, week, game_type, team)` | derived from `schedules` | One row per team, pre-game-known context only: `opponent`, `is_home`, `rest`, `div_game`, weather, `implied_total` (`total_line/2 ± spread_line/2`), `team_spread`. No outcome columns. |
| `seasonal_rosters` | `(player_id, season)` | `load_rosters` | Player's team + status + experience for a season. Team is pre-Week-1-known, so `changed_team` features derived from it are not leakage. |
| `player_ids` | `(gsis_id)` | `load_ff_playerids` | Cross-reference: `gsis_id ↔ pfr_id ↔ sleeper_id ↔ yahoo_id ↔ espn_id`. Use for every cross-source join — never join on name/team. Null-`gsis_id` rows dropped. |

## Not landed yet (out of scope for the draft sprint)

- **Red-zone / route-level play-by-play** — for red-zone touches, carries inside
  the 10, aDOT. `nflreadpy.load_pbp` has it; not pulled for v1.
- **Vegas odds API / weather API** — `schedules` already carries closing
  spread/total and basic weather, enough for v1.
- **Sleeper / Yahoo league data** — pulled by the application layer at draft time.

## Refresh cadence

nflverse regenerates the current season's releases within a day of each game.
In-season: `python -m data.build --seasons <current>` weekly (Tue/Wed, after stat
corrections settle). Historical seasons are stable.
