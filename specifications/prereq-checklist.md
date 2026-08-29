# Pre-req Checklist

Concrete setup steps derived from the decisions recorded in `AGENTS.md` and `/specifications`.
This is the "get the environment ready" list — no project code depends on anything here being
done in a particular order, but nothing in `/data` can really start until the Docker/Python
items are settled.

**During `draft-sprint-plan.md`, most of this file is deferred** — Docker, remote access, and
credentials sections below are all sprint-irrelevant (see each section for why). The one item
that's a real blocker is Python — flagged first.

## Tonight, before Friday

- [x] **Install `sqlite3` CLI** — done.
- [x] Python 3.12 — **resolved: not needed.** Created `.venv` on WSL2's system Python 3.14.4
      (Fri 2026-08-28) and `pip install`ed the full sprint stack — pandas 3.0.5, lightgbm 4.7.0,
      scikit-learn 1.9.0, nfl_data_py 0.3.2, numpy 2.5.2, scipy 1.18.1, pyarrow 25.0.1,
      fastparquet, sqlite-utils. All wheels resolved cleanly on 3.14, no source builds. The
      wheel-availability worry didn't materialize; staying on 3.14 for the sprint venv. (The
      post-sprint Docker image still pins 3.12 per the section below — revisit that too, since
      3.14 now looks fine.)

## Repo location

- [x] Move the repo into the WSL2-native filesystem — done, now at
      `/home/warnerjc/dev/gsv-fantasy-football` (Ubuntu distro).
- [ ] Re-open the project in your editor pointed at the WSL2 path (e.g. VS Code "Reopen in WSL",
      or `code .` from a WSL2 terminal in that directory).
- [ ] Confirm you're working from the WSL2 copy for a bit, then delete the old Windows-side
      folder at `C:\Users\jcwar\Desktop\development\gsv-fantasy-football`. Not automatic —
      deliberate step once you've verified nothing's missing.

## Container runtime — not needed for the sprint, but already confirmed working

The draft sprint itself runs in a plain WSL2 venv, no Docker (see `draft-sprint-plan.md`) — so
none of this blocked anything this week. Verified anyway ahead of the post-sprint
containerization work, so it's one less thing to check later:

- [x] Rancher Desktop confirmed set to **`dockerd (moby)`** engine.
- [x] `docker` / `docker compose` confirmed available from a WSL2 Ubuntu terminal.
- [x] Rancher Desktop's WSL integration confirmed enabled for the Ubuntu distro.

## Python (post-sprint / Docker context)

- [x] Resolved: host Python installs (native Windows 3.12.4, WSL2 system 3.14.4) don't matter
      once containerized — the project's Docker image will pin its own version.
- [ ] When the first Dockerfile is written (post-sprint), pin the base image to **Python 3.12**
      (e.g. `python:3.12-slim`) — same reasoning as the sprint venv above, wheel-availability
      safety over the very new 3.14.

## GUI tooling (optional, for manually inspecting the SQLite DB)

- [ ] Install [DB Browser for SQLite](https://sqlitebrowser.org/) on Windows.
- [ ] Once the DB file exists, confirm it can be opened either via the WSL2 UNC path
      (`\\wsl$\Ubuntu\home\warnerjc\dev\gsv-fantasy-football\...`) or, more reliably, by pointing
      DB Browser at a bind-mounted path Docker exposes back to the Windows side.

## Remote access — deferred, not needed for the sprint

Only relevant once the weekly dashboard exists (post-sprint, per `AGENTS.md`). Nothing to do
here before either draft.

- [ ] Install **Tailscale** or set up **Cloudflare Tunnel** — final pick is an implementation-
      time detail (see `AGENTS.md`).
- [ ] If Tailscale: install on the Windows machine and on your phone (and work machine, per
      `data-engineer.md` Q2), sign into the same tailnet.
- [ ] If Cloudflare Tunnel: Cloudflare account + a domain to route the tunnel through, install
      `cloudflared` on the Windows/WSL2 side.

## Credentials — deferred, not needed for the sprint

- [ ] Create an account/API key for **The Odds API** (the-odds-api.com) or an equivalent
      Vegas-lines provider — **not actually needed for the draft-sprint model.** Vegas implied
      team totals are a weekly-game signal; they don't meaningfully apply to a season-total
      draft ranking. Revisit once the weekly tool (v2) is being built, where it's genuinely
      relevant.
- [ ] Recreate the **Yahoo Developer app** (Client ID/Secret for OAuth) — **not needed for the
      sprint either.** Yahoo league `236625` is served via the manually-captured scoring config
      (`league-configs/yahoo-236625-scoring.json`), no API call required. Only matters if real
      Yahoo API integration (auto-pull, live polling) is built later, per the original
      "Yahoo — add later" plan.
- [x] Sleeper API needs no credentials (public, no auth) — nothing to do here, ever.

## GPU acceleration — explicitly not needed

Confirmed in `ml-modeler.md`: CPU-only LightGBM/XGBoost is sufficient for this project's data
scale (hundreds of thousands of rows at most). **Not setting up WSL2 GPU passthrough** —
skipping the NVIDIA driver/Container Toolkit/Rancher Desktop GPU-passthrough setup described at
https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute entirely for now. That setup
targets CUDA-heavy workloads (deep learning) this project doesn't have. Revisit only if a
concrete future workload actually needs it (e.g., running a local LLM for the v2 injury-news
idea, if that's ever built to run locally rather than via an API) — at that point, also verify
Rancher Desktop's current GPU-passthrough maturity specifically, since it's less established
there than in Docker Desktop.

## Not yet started (deferred until pipeline work actually begins)

- Docker Compose file(s) for the pipeline/app services.
- Dockerfile(s) pinning the Python 3.12 base image.
- Windows Task Scheduler job definitions for the weekly refresh.
