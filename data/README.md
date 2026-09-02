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

## Yahoo Fantasy API (read-only)

`data/yahoo.py` is a thin OAuth2 client for pulling Yahoo league config directly
(vs. the hand-captured `specifications/league-configs/yahoo-236625-scoring.json`).
Credentials load from `.env` at the repo root — copy `.env.example`, create a
Yahoo OAuth app at <https://developer.yahoo.com/apps/create/> (Redirect URI
`https://localhost:8000/callback`), and paste in the Client ID / Secret.

**Fantasy API access is approval-gated.** Apply at
<https://sports.yahoo.com/developer/access/> (read-only — write access is
discontinued) and wait for Yahoo's approval email; the self-serve "Fantasy
Sports → Read" permission checkbox no longer exists in the app UI. Until the
account is approved, every fantasy endpoint returns
`401 additional_authorization_required`.

```
bin/yahoo-auth login --manual            # one-time; caches tokens in data/cache/ (git-ignored)
bin/yahoo-auth whoami                     # prove the token works
bin/yahoo-auth leagues                    # list your NFL leagues + league_keys
bin/yahoo-auth settings nfl.l.236625      # dump a league's raw settings JSON
bin/yahoo-auth raw <fantasy/v2 path>      # ad-hoc endpoint exploration
```

### Which login flow — always use `--manual` on this machine

`bin/yahoo-auth login` (no flag) runs a local HTTPS listener for the OAuth
redirect, but on WSL2 the Windows→WSL `localhost` forwarding is flaky and Chrome
often hides the self-signed-cert "proceed" link, so the redirect dead-ends on
"this site can't be reached".

**On this WSL2 setup, always run `bin/yahoo-auth login --manual`.** It prints the
URL; you approve in the browser, get redirected to a page that won't load, then
copy the **full URL from the address bar** and paste it at the prompt. The command
pulls the `code` out of it. (If the pasted URL says `error=...` instead of
`code=...`, it tells you and exits — nothing was authorized.)

### Do I need to log in again on Yahoo draft day?

**Usually no.** `login` is a one-time step: the refresh token is cached in
`data/cache/yahoo_token.json` and every `bin/yahoo-auth` call auto-refreshes the
short-lived access token from it. That keeps working for months.

You only need to re-run `bin/yahoo-auth login --manual` if:

- a command prints **`no Yahoo token cached`** (the token file was deleted — e.g.
  someone cleared `data/cache/`, or the repo was re-cloned), or
- a command fails refreshing with **`invalid_grant`** / **`stored token has no
  refresh_token`** (you revoked the app's access in your Yahoo account, or Yahoo
  expired the grant after long inactivity).

**Draft-day checklist (do this the day before, not 5 minutes before):**

```
bin/yahoo-auth whoami            # succeeds -> you're set, nothing else to do
                                 # fails    -> bin/yahoo-auth login --manual, then whoami again
bin/yahoo-auth leagues           # confirm keepitcrooked's league_key is listed
```

The Yahoo league drafts offline, so there is no live pick polling to keep a token
warm during the draft itself — the only thing that touches Yahoo is pulling league
config beforehand.

### Not wired into scoring yet

The API's `league/settings` shape differs from the hand-captured config
`scoring.normalize_yahoo` expects — mapping one to the other is the next step. For
now the CLI just dumps raw JSON.

## Not landed yet (out of scope for the draft sprint)

- **Red-zone / route-level play-by-play** — for red-zone touches, carries inside
  the 10, aDOT. `nflreadpy.load_pbp` has it; not pulled for v1.
- **Vegas odds API / weather API** — `schedules` already carries closing
  spread/total and basic weather, enough for v1.
- **Sleeper league data** — pulled by the application layer at draft time.
- **Yahoo league data** — read-only API client exists (`data/yahoo.py`, see above);
  not yet mapped into the scoring config.

## Refresh cadence

nflverse regenerates the current season's releases within a day of each game.
In-season: `python -m data.build --seasons <current>` weekly (Tue/Wed, after stat
corrections settle). Historical seasons are stable.
