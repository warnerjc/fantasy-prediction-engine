# AGENTS.md — Fantasy Football Prediction System

This file is the shared foundation for any agent (human or AI) working in this repo.
It encodes architectural invariants that must survive regardless of which specific
technologies end up chosen. See `PROJECT-BOOTSTRAP.md` for the full narrative writeup.

**Status: active sprint.** Building a scoped-down draft assistant against a hard deadline —
Sleeper league draft Wed 2026-09-02, Yahoo league draft date TBD. See
`specifications/draft-sprint-plan.md` for the day-by-day plan and what's deliberately deferred.
**For the duration of this sprint**, two of the "confirmed" decisions below are temporarily
suspended: no Docker (plain WSL2 venv) and a trimmed ~5-10 season nflverse pull (not full
1999+) — both revert to the confirmed decision once the sprint ends. Everything else below
(architecture, invariants, scoring/leakage discipline) applies at full strength even during the
sprint — the deadline changes *what* gets built, not *how carefully*.

Treat every specific technology named below (SQLite, LightGBM, Python, Streamlit, nflverse,
etc.) as the *current best guess* outside of what's explicitly confirmed. If a choice matters
and isn't confirmed, say so — don't silently build around it.

## What this system is

Two end-user tools for standard head-to-head (H2H) fantasy football leagues (13/14/15-week
seasons, Sleeper and Yahoo):

1. **Draft assistant** — ranks/recommends players live during a draft, given league settings,
   league size, draft position, draft type (snake/auction).
2. **Weekly start/sit tool** — projects each rostered player's expected fantasy points
   (with floor/ceiling) for a given week.

**Explicitly out of scope:** DFS/best-ball optimization, ownership modeling, ceiling-obsessed
lineup construction. This is standard H2H roster management only. If a request drifts toward
those, flag it rather than quietly building it.

## Architecture

```
/data          -> pipeline: nflverse + Sleeper/Yahoo + injuries + Vegas lines -> SQLite (weekly grain)
/features      -> feature functions on player-week rows, parameterized by as-of week/window
/scoring       -> raw stats -> fantasy points, given a league's actual scoring settings
/models        -> training + inference, parameterized by target column + objective + temporal split
/applications  -> draft_tool.py, weekly_tool.py (consumers of /models + /scoring)
```

Each directory has a matching persona skill in `.claude/skills/` — see "Personas" below.

## Non-negotiable invariants

These exist so v1 (season-grain draft tool) doesn't require a rewrite for v2 (weekly quantile
model + adjustment layer). Violating them creates rework later — treat them as load-bearing,
not stylistic preferences:

- **`player_week_stats` is the single source of truth**, at weekly grain. Season totals are a
  `GROUP BY` aggregation on top of it, computed on demand — never a parallel table or a
  separately-pulled "season stats" source.
- **Feature functions are parameterized by an as-of week / window.** v1 uses the full prior
  season as the window; v2 uses a rolling 3-5 week window. Same function, different config —
  never a v1-only and v2-only fork of the same feature logic.
- **No leakage:** a feature computed "as of week N" must never read week N or later outcome
  data. Pre-game-known information (opponent, Vegas line, home/away, rest days) is fine even
  though it's "about" week N; box-score outcomes are not.
- **Predictions are typed as `{mean, p10, p50, p90}` from day one**, even in v1 where only
  `mean` is populated. Downstream consumers (draft/weekly tools) should never need to change
  shape when v2 adds real quantiles.
- **Training config is swappable, not hardcoded**: target column, loss/objective, and temporal
  split strategy are config, so v2 is a new config file, not new architecture.
- **`/scoring` never hardcodes a scoring format.** It's a pure function of (raw stats, a
  league's scoring settings dict pulled from the API). Standard PPR is one input, not an
  assumption baked into the code. Training labels and the application layer must call the
  exact same scoring function — never two implementations that can drift.
- **Validation is walk-forward by season** (train on earlier years, test on a later held-out
  year). Never a random shuffle split — it leaks future info into the past for time-series
  sports data.
- **One model per position** (QB/RB/WR/TE/K/DEF) — scoring dynamics differ enough that a
  shared model is the wrong default, not a premature optimization.

## Personas

Each persona is a `.claude/skills/` entry scoped to one architecture boundary above, invoked
with `/persona-name`. Use the one matching the layer you're working in; each persona's file
lists what it explicitly defers to the others.

| Persona | Owns | Skill |
|---|---|---|
| Data engineer | `/data` — ingestion, SQLite schema, source freshness, ID crosswalks | `data-engineer` |
| Feature engineer | `/features` — as-of/window feature functions, leakage discipline | `feature-engineer` |
| Scoring engineer | `/scoring` — stats → fantasy points, league scoring settings | `scoring-engineer` |
| ML modeler | `/models` — training, walk-forward validation, config-driven objectives | `ml-modeler` |
| Application engineer | `/applications` — draft_tool.py, weekly_tool.py, dashboard | `app-engineer` |
| Product/architecture | Cross-cutting scoping, tradeoff decisions, PM collaboration | `product-architect` |

## Confirmed hosting decisions

Resolved via `/specifications` — treat these as settled unless something changes materially:

- **Draft tool is local-only.** Runs on the same Windows machine, used from the same home
  network every time. No remote access needed. The UI auto-updates as picks happen (a polling
  loop against live draft state, not a one-shot render — Sleeper has no push/webhook for draft
  events), kept open as a browser tab through the draft.
- **Weekly tool is a persistent, long-running local service**, not a run-on-demand script — it
  needs to stay up across the week so it can be checked repeatedly as injury news/Vegas lines
  move. That means auto-start/restart-on-crash matters, not just "run it when needed."
- **Weekly tool must be reachable from a phone (or another device — work machine, etc.) when
  away from home** (some game-day decisions happen off the home network) — via **Tailscale or
  Cloudflare Tunnel**, explicitly *not* port-forwarding/DDNS. No open inbound ports on the home
  router, full stop. Final pick between the two is an implementation-time detail, not an open
  question.
- **Repo lives in the WSL2-native filesystem**, not the Windows filesystem — at
  `/home/warnerjc/dev/gsv-fantasy-football` (Ubuntu distro), reachable from Windows at
  `\\wsl$\Ubuntu\home\warnerjc\dev\gsv-fantasy-football`. Chosen over staying on `C:\` because
  Docker bind mounts perform meaningfully worse crossing the `/mnt/c` 9P bridge than against a
  native WSL2 path.
- **Runtime environment: WSL2 + Docker Compose**, using **Rancher Desktop** (already installed)
  as the container engine rather than Docker Desktop. Confirm Rancher Desktop is set to the
  `dockerd (moby)` engine (not `containerd`) so standard `docker compose` works as documented.
  The persistent weekly dashboard runs as a container with a restart policy
  (`restart: unless-stopped`); the weekly refresh is a one-off container run
  (`docker compose run --rm pipeline`) triggered by Windows Task Scheduler.
- **The machine is fully shut down (not sleeping) when not in use** — this is a known, *accepted*
  gap against the remote-access decision above: Tailscale/Cloudflare Tunnel can't reach a
  powered-off machine, so away-from-home dashboard access only works on weekends it happens to
  already be on. Not solved for v1 by design (the user mostly works from home and would rather
  not power the machine on just for this). Wake-on-LAN over Tailscale is a possible future
  enhancement, not a v1 requirement — note if it's ever pursued that Tailscale doesn't extend LAN
  broadcast, so it likely still needs an always-on relay on the LAN, and the AT&T-provided router
  may not support the configurability needed. Don't build around WoL working; don't build around
  the machine always being reachable either — both are explicitly out of scope for v1.

- **Compute confirmed sufficient, CPU-only.** i7-12700KF (12C/20T), 64 GB RAM — comfortably
  enough for CPU-only LightGBM/XGBoost training and walk-forward validation across full
  nflverse history without trimming the pull window. A GPU is present (~8 GB VRAM) but
  deliberately unused — not worth the CUDA/container-passthrough setup cost at this data scale.
  Don't reach for GPU acceleration unless a specific, measured bottleneck justifies it.

## Open local-system questions

This is a locally-hosted, single-user project, so several architecture decisions (storage
location/format, scheduling reliability, whether the app needs to be reachable beyond
localhost, how much history to pull) depend on facts about the actual machine and workflow
this runs on. Those questions live in `/specifications`, one file per persona that has
open questions — check there before assuming a hosting/environment detail, and add a question
there rather than guessing if a persona hits an unanswered one.

## Working conventions

- Confirm the current tech stack decision before assuming one from `PROJECT-BOOTSTRAP.md` —
  that doc is a snapshot, not a spec, and this project is explicitly mid-refinement.
- No cloud infra, no GPU assumed. This is a local-hosting project (cron/Task Scheduler for
  refresh jobs) sized for one user's leagues, not a multi-tenant service — don't over-build.
- Prefer walking through a design decision with the `product-architect` persona before writing
  code that locks in an architectural choice not yet confirmed.
