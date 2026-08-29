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

- [x] **venv on Python 3.14.4** — full sprint stack (`pandas` 3.0.5, `lightgbm` 4.7.0,
  `scikit-learn` 1.9.0, `nfl_data_py` 0.3.2, `numpy`, `scipy`, `pyarrow`, `fastparquet`,
  `sqlite-utils`) installs cleanly, all wheels, no source builds. **3.12 fallback not needed**
  — `prereq-checklist.md` updated. (Docker image still says pin 3.12; revisit, 3.14 looks fine.)
- [x] **`/scoring` rebuilt** around a canonical stat vocabulary + platform adapters
  (`normalize_sleeper` / `normalize_yahoo`), not a hand-typed key guess. Handles half/full/no
  PPR, 4-vs-6pt pass TD, Sleeper yardage-threshold bonuses, kicker distance buckets, DST
  points/yards-allowed tiers, TE-premium hook. Hand-verified vs a real line (Chase W5 2024)
  under **both** sprint leagues; 25 tests. Full mapping tables in `scoring/README.md`.
  Deferred: Sleeper `st_fum_rec` / `def_st_*` DST split (needs the PBP DST extractor;
  would double-count otherwise — TODO in `adapters.py`).
- [x] **`/data` pipeline** — `python -m data.build --seasons 2015-2024` lands six tables:
  `player_week_stats` (54.5k rows, weekly grain, the source of truth), `snap_counts`,
  `injuries`, `schedules` + derived `team_week` (opponent / home-away / rest / **Vegas implied
  team total**), and the `player_ids` gsis↔pfr↔sleeper↔yahoo crosswalk. Idempotent
  INSERT-OR-REPLACE on each table's PK (absorbs nflverse stat corrections). Real dupe hazards
  found + handled (PFR reuses some player ids across two people; injury reports get re-issued
  mid-week). 7 tests. Schema doc in `data/README.md`.
  **Not landed (post-sprint, per scope above):** play-by-play → red-zone touches, kicker
  FG-by-distance, DST stat lines; external Vegas/weather APIs.
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

- [ ] Rookie handling — model omits rookies (no prior NFL season); the board just doesn't show
  them. Need a rookie ADP list cross-reference for early rounds. Decide at app layer.
- [ ] Tuesday: dry-run `--watch` against a Sleeper mock draft.
- [ ] `python -m models.build --backtest <season>` — wire the projected-vs-actual view in as a
  repeatable command (nice-to-have).
- [ ] Tune `_BASELINE_MULT` / flex splits in `applications/roster.py` after a mock draft.

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
