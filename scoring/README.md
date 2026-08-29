# /scoring — raw stats → fantasy points

Pure, league-agnostic scoring. Never hardcodes a format (see AGENTS.md). The same
`score()` call produces model-training labels and application-layer projections.

```
league settings ──normalize_sleeper / normalize_yahoo──▶ ScoringRules ─┐
                                                                       ├─▶ score(stats, rules, position) ─▶ points
box score ──canonical_offense_stats / stats_dict──▶ canonical stat dict ┘
```

Everything is expressed in **one canonical stat vocabulary**. Each platform names
its knobs differently and nflverse names its columns differently again; the
adapters (`adapters.py`) and the extractor (`extract.py`) are the *only* code that
knows those foreign names. Downstream code only sees canonical keys.

> The tables below are generated from the mapping dicts in code — if they drift,
> the code (`stat_keys.py`, `adapters.py`, `extract.py`) is the source of truth.

## Canonical keys (`scoring/stat_keys.py`)

### Offense

| key | meaning |
|---|---|
| `pass_cmp` / `pass_att` | completions / attempts |
| `pass_yd` / `pass_td` / `pass_int` | passing yards / TDs / interceptions thrown |
| `pass_2pt` | 2-pt conversion passes |
| `pass_sack` | sacks taken (as the passer) |
| `rush_att` / `rush_yd` / `rush_td` | carries / rush yards / rush TDs |
| `rush_2pt` | 2-pt conversion runs |
| `rec` / `rec_tgt` | receptions / targets |
| `rec_yd` / `rec_td` | receiving yards / TDs |
| `rec_2pt` | 2-pt conversion receptions |
| `two_pt` | any 2-pt conversion (platforms that don't split by play type) |
| `fum_lost` | fumbles lost |
| `fum_rec_td` | offensive fumble-recovery TD |
| `ret_yd` / `ret_td` | kick+punt return yards / TDs credited to the returner |

### Kicking

| key | meaning |
|---|---|
| `k_xp_made` / `k_xp_missed` | extra points made / missed |
| `k_fg_made` / `k_fg_missed` | field goals made / missed, any distance (count) |
| `k_fg_made_yds` | summed distance of made FGs (for per-yard scoring, Sleeper `fgm_yds`) |
| `k_fg_made_0_19 … _20_29 … _30_39 … _40_49 … _50p` | made FGs by distance bucket |
| `k_fg_missed_0_19 … _50p` | missed FGs by distance bucket |

### Team defense / special teams (DST)

| key | meaning |
|---|---|
| `dst_sack` / `dst_int` / `dst_fum_rec` / `dst_ff` | sacks / interceptions / fumble recoveries / forced fumbles |
| `dst_safety` / `dst_blk_kick` | safeties / blocked kicks |
| `dst_td` | defensive TD (INT/fumble return) |
| `dst_ret_td` | kick/punt/blocked-kick return TD by the ST unit |
| `dst_2pt_return` | defensive 2-pt return / returned blocked XP |
| `dst_4th_down_stop` | 4th-down stop (Yahoo) |
| `dst_pts_allowed` | points allowed this game → scored via a **tier table**, not per-unit |
| `dst_yds_allowed` | total yards allowed this game → **tier table** |

### Virtual keys (yardage-bonus targets only)

`rush_rec_yd` = `rush_yd` + `rec_yd`  ·  `pass_rush_yd` = `pass_yd` + `rush_yd`

## Sleeper `scoring_settings` → canonical (`normalize_sleeper`)

Per-unit (`count × points`):

| Sleeper | canonical | | Sleeper | canonical |
|---|---|---|---|---|
| `pass_cmp` | `pass_cmp` | | `fgm_yds` | `k_fg_made_yds` |
| `pass_att` | `pass_att` | | `fgm` | `k_fg_made` |
| `pass_yd` | `pass_yd` | | `fgmiss` | `k_fg_missed` |
| `pass_td` | `pass_td` | | `fgm_0_19 … _50p` | `k_fg_made_0_19 … _50p` |
| `pass_int` | `pass_int` | | `fgmiss_0_19 … _50p` | `k_fg_missed_0_19 … _50p` |
| `pass_2pt` | `pass_2pt` | | `xpm` / `xpmiss` | `k_xp_made` / `k_xp_missed` |
| `pass_sack` | `pass_sack` | | `sack` | `dst_sack` |
| `rush_att` | `rush_att` | | `int` | `dst_int` |
| `rush_yd` | `rush_yd` | | `fum_rec` | `dst_fum_rec` |
| `rush_td` | `rush_td` | | `ff` | `dst_ff` |
| `rush_2pt` | `rush_2pt` | | `safe` | `dst_safety` |
| `rec` | `rec` | | `blk_kick` | `dst_blk_kick` |
| `rec_tgt` | `rec_tgt` | | `def_td` | `dst_td` |
| `rec_yd` | `rec_yd` | | `def_2pt` | `dst_2pt_return` |
| `rec_td` | `rec_td` | | `st_td` | `dst_ret_td` |
| `rec_2pt` | `rec_2pt` | | | |
| `fum_lost` | `fum_lost` | | | |
| `fum_rec_td` | `fum_rec_td` | | | |

Yardage bonuses (flat points awarded once at threshold):

| Sleeper | fires when |
|---|---|
| `bonus_pass_yd_300` / `_400` | `pass_yd` ≥ 300 / 400 |
| `bonus_rush_yd_100` / `_200` | `rush_yd` ≥ 100 / 200 |
| `bonus_rec_yd_100` / `_200` | `rec_yd` ≥ 100 / 200 |
| `bonus_rush_rec_yd_100` / `_200` | `rush_yd + rec_yd` ≥ 100 / 200 |

Tier tables: `pts_allow_0 / _1_6 / _7_13 / _14_20 / _21_27 / _28_34 / _35p` →
`dst_pts_allowed` tiers; `yds_allow_0_100 … _550p` → `dst_yds_allowed` tiers.

TE premium: `bonus_rec_te` → `position_bonuses["TE"] = {rec: <pts>}`.

**Not mapped:** `st_fum_rec`, `st_ff`, `def_st_td`, `def_st_fum_rec`, `def_st_ff`
— special-teams-vs-defense splits of events the DST source (`load_team_stats`)
only reports as combined totals; mapping them would double-count.

## Yahoo config → canonical (`normalize_yahoo`)

Consumes the hand-captured `specifications/league-configs/*.json` (nested under
`scoring.offense` / `scoring.kicking` / `scoring.defense_special_teams`). Real
Yahoo API integration is deferred.

| Yahoo (offense) | canonical | | Yahoo (kicking) | canonical |
|---|---|---|---|---|
| `completion` | `pass_cmp` | | `fg_made_0_19 … _50_plus` | `k_fg_made_0_19 … _50p` |
| `passing_yard` | `pass_yd` | | `fg_missed_0_19 … _50_plus` | `k_fg_missed_0_19 … _50p` |
| `passing_td` | `pass_td` | | `pat_made` / `pat_missed` | `k_xp_made` / `k_xp_missed` |
| `interception_thrown` | `pass_int` | | | |
| `rushing_yard` / `rushing_td` | `rush_yd` / `rush_td` | | **Yahoo (DST)** | |
| `reception` | `rec` | | `sack` | `dst_sack` |
| `receiving_yard` / `receiving_td` | `rec_yd` / `rec_td` | | `interception` | `dst_int` |
| `return_yard` / `return_td` | `ret_yd` / `ret_td` | | `fumble_recovery` | `dst_fum_rec` |
| `two_point_conversion` | `two_pt` | | `forced_fumble` | `dst_ff` |
| `fumble_lost` | `fum_lost` | | `defensive_touchdown` | `dst_td` |
| `offensive_fumble_return_td` | `fum_rec_td` | | `safety` | `dst_safety` |
| | | | `block_kick` | `dst_blk_kick` |
| | | | `kick_punt_return_td` | `dst_ret_td` |
| | | | `fourth_down_stop` | `dst_4th_down_stop` |
| | | | `extra_point_returned` | `dst_2pt_return` |

Tier tables: `points_allowed_0 … _35_plus` → `dst_pts_allowed`;
`yards_allowed_0_99 … _500_plus` → `dst_yds_allowed`.

## `player_week_stats` → canonical (`canonical_offense_stats`)

Offense only. Kicking → `canonical_kicking_stats` (from `kicking_stats`); DST →
`canonical_dst_stats` (from `team_defense_stats`). See `data/README.md`.

| nflverse column | canonical | | nflverse column | canonical |
|---|---|---|---|---|
| `completions` / `attempts` | `pass_cmp` / `pass_att` | | `receptions` / `targets` | `rec` / `rec_tgt` |
| `passing_yards` / `passing_tds` | `pass_yd` / `pass_td` | | `receiving_yards` / `receiving_tds` | `rec_yd` / `rec_td` |
| `interceptions` | `pass_int` | | `receiving_2pt_conversions` | `rec_2pt` |
| `passing_2pt_conversions` | `pass_2pt` | | `special_teams_tds` | `ret_td` |
| `sacks` | `pass_sack` | | `rushing_fumbles_lost` + `sack_fumbles_lost` + `receiving_fumbles_lost` | `fum_lost` (summed) |
| `carries` / `rushing_yards` / `rushing_tds` | `rush_att` / `rush_yd` / `rush_td` | | *(derived)* `pass_2pt`+`rush_2pt`+`rec_2pt` | `two_pt` |
| `rushing_2pt_conversions` | `rush_2pt` | | | |

Not available from this release: `ret_yd` (return yards), red-zone touches — both
need play-by-play.
