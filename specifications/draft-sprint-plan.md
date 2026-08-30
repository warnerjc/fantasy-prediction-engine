---
status: active
scope: draft assistant only — 2 of the user's 4 leagues actually need it (see registry below)
deadline: Sleeper draft Wed 2026-09-02; Yahoo (236625) draft date TBD, offline draft type
---

# Draft Sprint Plan

A tightly-scoped build to have a working, trustworthy draft assistant ready before the Sleeper
draft (Wed 09-02) and, ideally, before the Yahoo draft (date not yet known). See
`product-architect.md` Q4 for the full reasoning behind the scope cuts below — this file is the
execution plan, that file is the decision record.

## League registry

| League | Platform | ID | Type | Status | Draft-assistant needed? |
|---|---|---|---|---|---|
| (unnamed) | Sleeper | `1356741521163968513` | Redraft | Drafts Wed 2026-09-02 | **Yes — live, via Sleeper API** |
| (unnamed) | Sleeper | `1314665599955132416` | Dynasty | Already drafted | No — out of scope for v1 regardless (dynasty non-goal) |
| (unnamed) | Yahoo | `775326` | Redraft | Already drafted | No |
| keepitcrooked | Yahoo | `236625` | Redraft | **Offline draft, date TBD** | **Yes — static ranked list; scoring config already captured, see [`league-configs/yahoo-236625-scoring.json`](league-configs/yahoo-236625-scoring.json)** |

Scope is smaller than originally planned: only **2 leagues** need active draft-assistant work
this sprint. `236625`'s "Offline Draft" type means there's no live Yahoo draft-room state to
poll even if we wanted it — the static-ranked-list approach isn't a sprint compromise for this
league, it's the only approach that applies. The two already-drafted leagues need nothing from
this sprint; they'll matter once the weekly tool is built post-sprint.

## What this sprint explicitly does NOT include

Deferred to after both drafts, with no loss of long-term direction — these come back once
there's no deadline pressure:

- Weekly start/sit tool, and everything tied to it (persistent dashboard, remote access via
  Tailscale/Cloudflare Tunnel)
- v2: weekly quantile model, adjustment/injury layer
- Dynasty draft support (the 4th Sleeper league) — needs a multi-year value model v1 doesn't have
- Docker/Rancher Desktop containerization — build and run in a plain WSL2 Python venv this week;
  containerize afterward, wrapping already-working code
- Real Yahoo API/OAuth integration — Yahoo league is served via a manually-entered scoring
  config instead (see below)
- Full nflverse history back to 1999 — pull ~5-10 seasons for the sprint model; backfill later

## What "done" looks like by Wednesday

A ranked, scoring-adjusted player list, correct for each of the 2 leagues that actually need one
this sprint (`1356741521163968513` on Sleeper, `236625` on Yahoo — see registry above), that the
user can draft from — at minimum as a static list to manually check off picks against. Live
auto-refresh-as-picks-happen is a bonus layer on top for the Sleeper league only, not a
requirement for "done."

## Day-by-day

### Thu 08-27 night — planning wrap-up (this session)
- Close out `/specifications` — done.
- League IDs and Yahoo scoring config captured — done (see registry above).
- No code yet.

### Fri 08-28 — core pipeline + model (the big day)
- **First 5 minutes:** throwaway venv on WSL2's existing Python 3.14, `pip install` the real
  sprint dependencies (pandas, LightGBM, scikit-learn, `nfl_data_py`/`nflreadpy`, pyarrow if
  needed). If wheels resolve cleanly, keep 3.14 and move on — no version downgrade needed. Only
  if something specifically fails to find a wheel, install 3.12 (see `prereq-checklist.md` for
  the fallback commands) and redo the venv on that instead. Don't decide this in advance.
- WSL2 Python venv, no Docker.
- Pull nflverse weekly stats, ~5-10 seasons, via `nfl_data_py`/`nflreadpy`.
- Build `player_week_stats` (SQLite), weekly grain, per the `AGENTS.md` invariant — not cut,
  it's cheap to do right the first time.
- Build `/scoring`: pure function of (raw stats, scoring settings dict). For `1356741521163968513`,
  the settings dict comes from a live Sleeper API pull; for `236625`, load
  `league-configs/yahoo-236625-scoring.json` directly — same function, two ways to populate the
  input. Watch the fractional-points note in that file (continuous multiply, not floor-division,
  for yardage bonuses).
- Build feature functions, as-of/window parameterized even though only "full prior season" is
  used this sprint — same reasoning, no extra cost to do it right.
- Train a first-pass model on season-total fantasy points, walk-forward validated against at
  least one held-out season. Per-position if time allows; a single blended model as a fallback
  if not — upgrade to per-position on Sunday if needed.
- **End-of-day deliverable:** a ranked list for each of the 2 leagues that actually need one
  (`1356741521163968513`, `236625`), correct against each league's actual scoring settings.
  This alone is a usable draft tool even with zero UI.

#### Progress log — Fri 08-28

*(Note: the initial repo scaffold + a first-pass `/scoring` were done in a separate
Copilot session; reviewed and largely rebuilt here.)*

- [x] **venv on Python 3.14.4** — full sprint stack (see `requirements.txt`) installs cleanly,
  all wheels, no source builds. **3.12 fallback not needed** — `prereq-checklist.md` updated.
- [x] **Data lib: `nflreadpy`, not `nfl_data_py`.** Started on `nfl_data_py` 0.3.2 (already
  installed) — but its `import_weekly_data` **can't fetch the 2025 season** (404, stale
  release), which would have shipped a tool that projects the *wrong year*. Migrated `/data`
  to `nflreadpy` (maintained client; also serves kicker FG buckets + team-defense box scores
  natively → no play-by-play pull). Everything downstream reads SQLite, so blast radius = 1
  file. Caught before the draft.
- [x] **`/scoring` rebuilt** around a canonical stat vocabulary + platform adapters
  (`normalize_sleeper` / `normalize_yahoo`), not a hand-typed key guess. Handles half/full/no
  PPR, 4-vs-6pt pass TD, Sleeper yardage-threshold bonuses, kicker distance buckets, DST
  points/yards-allowed tiers, TE-premium hook. Hand-verified vs a real line (Chase W5 2024)
  under **both** sprint leagues; 25 tests. Full mapping tables in `scoring/README.md`.
  Deferred: Sleeper `st_fum_rec` / `def_st_*` DST split (needs the PBP DST extractor;
  would double-count otherwise — TODO in `adapters.py`).
- [x] **`/data` pipeline** — `python -m data.build --seasons 2015-2025` (~30s) lands nine
  tables: `player_week_stats` (66k rows, weekly grain, source of truth), `kicking_stats` +
  `team_defense_stats` (native from nflreadpy's stat releases), `snap_counts`, `injuries`,
  `seasonal_rosters`, `schedules` + derived `team_week` (Vegas implied team total), and the
  `player_ids` crosswalk. Idempotent INSERT-OR-REPLACE per PK. Dupe hazards found + handled
  (PFR reuses some player ids; injury reports re-issued mid-week). 11 tests. `data/README.md`.
  **Not landed:** red-zone / route-level play-by-play; external Vegas/weather APIs.
- [x] **`/features`** — `window.py` engine (`AsOf` + `Window.prior_season()` / `.trailing()`,
  `visible_weeks()` does leakage filtering once). Functions: `opportunity_features` (usage
  totals/rates, target/rush/air-yards share, WOPR), `snap_features`, `opponent_allowed_features`
  (`def_<POS>_<stat>_pg`), `context_features` (Vegas implied total/spread, home/rest/venue),
  `identity_features` (age, exp, draft capital). `build.season_feature_matrix` /
  `training_frame` assemble the v1 matrix (player × target_season, no label). Opponent/context
  built but held out of the v1 matrix — they're the v2 weekly join. Hand-checked vs Chase 2023.
  12 tests incl. leakage guards. `features/README.md`.
- [x] **v1 model** — six per-position LightGBM models (QB/RB/WR/TE/K/DEF), **PPG target**,
  config-driven (`ModelConfig`: target/objective/split/filters/quantiles), walk-forward by
  season. Decisions made: PPG (not season-total — same unit v2 predicts); per-position for all
  six incl. K/DEF (pulled K/DEF stats from PBP to do it). Labels league-specific via
  `scoring.score`, features league-agnostic → per-league model is a relabel. `Prediction`
  typed `{mean,p10,p50,p90}`, v1 fills mean only.
  `python -m models.build --league sleeper|yahoo` → ranked `models/output/<league>_projections.csv`.
  Walk-forward (2015–24): QB/WR ρ≈0.67, RB≈0.58, TE≈0.60; K/DEF ρ≈0.05–0.09 (near-noise,
  expected). 2025 projections pass the eye test. `models/README.md`.
- [x] **team-change features** — `changed_team` + new-team environment + vacated opportunity
  share (`features/team_change.py`, `seasonal_rosters` table). Small net gain for RB/WR,
  excluded from QB/TE. Does not fix marquee breakouts (Saquon RB19→1) — open problem, documented.
- [x] **`/applications/draft_tool.py`** — snake draft board with value-over-replacement (VBD),
  gap-based tiers, live Sleeper draft-state polling (`--watch`), snake pick math + "likely
  available at your next pick", drafted-player filtering. Both leagues verified. Auction not
  supported (different logic — flagged). `applications/README.md`.
  `python -m applications.draft_tool --league sleeper --slot N --watch`

**Well ahead of schedule** — the full v1 stack (data → scoring → features → models → draft
tool) is done Friday. 55 tests. Remaining is validation + polish, not building:

- [x] Rookie handling — `applications/adp.py` pulls crowd ADP (fantasyfootballcalculator.com,
  cached), `board.py` places every unmatched top-180 ADP player on the board with a VBD
  imputed from ADP (isotonic on players that have both). Tagged `(adp)`; `adp` column shown
  for everyone. Sleeper-id backfill from the cached player directory so they drop off when drafted.
- [x] Model-vs-market — draft board now blends model VBD with an ADP-implied VBD
  (`_blend_market`, isotonic curve, default `--blend 0.7`). Reins in the model rating elite
  QBs ~2 rounds ahead of ADP. `--blend 1.0` = pure model.
- [x] Static list — `python -m applications.draft_tool --league <l> --export` →
  `models/output/<league>_board.csv` (full ranked, all positions, tiers, model/market/blended VBD).
- [x] Dry-run without a real mock — `applications/mock.py` (local full-draft sim + `--sims N`
  Monte Carlo, both leagues from configs); `draft_tool --replay --draft <id>` fast-forwards a
  completed Sleeper draft through the live view. `--watch` plumbing exercised without a room.
- [x] `python -m models.backtest --league <l> [--season Y|Y-Y]` (`bin/backtest`) — repeatable
  projected-vs-actual view + baselines (`last_ppg`, `ewma_ppg`, historical `market_adp`).
  Finding (2020–25): model is a wash with `ewma_ppg` on rank ρ for every skill position and
  worse than `ewma_ppg` for DEF; ADP has the best top-N hit. Validates leaning on the `--blend`
  ADP mix. See `models/README.md` → Backtest + baselines, and Appendix A for the roadmap.
- [ ] Tune `_BASELINE_MULT` / flex splits / `--blend` weight — the mock sim drafts QB/TE
  earlier than typical, a signal the blend could lean more toward market for those.

### Sun 08-30 night — validate + refine
- Sanity-check the model: SHAP/feature-importance should show targets/carries/red-zone
  touches/snaps dominating. If something else dominates, treat it as a leakage bug, not a
  finding — chase it down before trusting the output.
- Move to per-position models if Friday only got a blended one.
- Confirm the ranked list is pulling correct, current rosters/scoring for both leagues.

### Mon 08-31 night — draft-day app
- Pull live roster/draft state from Sleeper's API.
- Layer auto-refresh-as-picks-happen UI on top of the ranked list (Sleeper leagues only —
  stretch goal, not required for "done").
- Yahoo league: confirm the static ranked list works standalone for manual use.

### Tue 09-01 — buffer + dry run
- If the live-polling UI exists, dry-run it against a Sleeper mock draft to catch bugs before
  the real thing.
- Fix anything broken; otherwise treat the static ranked list as locked and ready.

### Wed 09-02 — Sleeper draft
- Use it. Static ranked list is the floor regardless of whether live polling made it in.

### Yahoo draft — whenever it lands
- Same static ranked list, manual scoring config already built Friday. No additional
  Yahoo-specific engineering required unless real API integration is explicitly requested later.

---

## Appendix A — variables that could beat market ADP (post-sprint backlog)

Context: the `bin/backtest` baseline run (2020–25, both leagues) showed the v1 model is a
**wash with a recency-weighted 2-yr PPG average** on rank correlation for every skill
position, and market ADP has the best top-N hit rate everywhere. On draft-day *ranking
accuracy* alone the tabular-usage model has a ceiling near ADP. To clear it we need signal
that is both predictive **and** underweighted by consensus — ADP already efficiently prices
name value, last year's box score, draft capital, obvious depth-chart moves, and beat-writer
narrative.

The tiers below are roughly in expected-value order. None are sprint work; they are the
modeling roadmap once the drafts are done. "Already built" flags features that exist in
`/features` but are held out of the v1 season matrix.

### Tier 1 — team environment (the model's biggest blind spot; features half-built)

- **Vegas preseason team strength** — season win totals → projected team points-for /
  implied pace. `context_features` (`implied_total`, `team_spread`, weather) already exist
  weekly-grain but are **not wired into the season model**. Wiring a target-season
  team-environment block in is the cheapest high-leverage change available.
- **Scheme / pace** — pass rate over expected (PROE), neutral-script pass rate,
  seconds/play, new-OC / new-HC flag.
- **Target & carry competition, done properly** — explicit vacated *and* added volume when
  teammates arrive/depart, weighted by quality of the competition. The current
  `team_change` features are a blunt version that barely moved the needle.

### Tier 2 — player trajectory

- **Structured age × position curves** — RB cliff ~28, WR year-3 leap, TE year-3/4. The
  model has raw `age` / `years_exp` as tree inputs but no structured curve; this directly
  targets the documented "over-projects aging vets off injury" bias.
- **Durability model** — replace the flat `_games_assumption` per position with
  games-missed-last-2-seasons + injury type + age. Fixes the Mostert / B. Robinson bias.

### Tier 3 — rookies (currently zero independent signal — placed by pure ADP)

- **Draft capital** — `draft_ovr` is already in `player_ids`, just unused for players with
  no prior NFL season. A rookie-only model keyed on draft capital is the single best start.
- **College production** — dominator rating, breakout age, yards per route run, college
  target share adjusted for competition.
- **Athletic testing** — RAS / 40 (modest for WR/RB).
- **Landing spot** — projected team pass rate, depth-chart slot, vacated volume.

### Tier 4 — beating ADP directly (market microstructure)

- **ADP trend / velocity** — is a player's ADP rising or falling over the last ~2 weeks of
  drafts.
- **ADP dispersion** — `stdev` / `high` / `low` are already fetched from FFC and unused.
  High dispersion = market disagreement = opportunity.
- **Cross-source ADP delta** — Sleeper vs Underdog vs NFFC vs FFC; gaps between casual and
  sharp pools are real inefficiencies.
- **Expert consensus vs ADP** — FantasyPros ECR minus ADP.

### Tier 5 — the real structural edge (not draft day)

- **v2 weekly quantile model + injury/news adjustment layer.** Static free lists cannot
  react to Thursday injury news or a mid-season target-share shift. That is a persistent
  in-season edge; the draft-day ranking is a one-shot where the market is hard to beat.
