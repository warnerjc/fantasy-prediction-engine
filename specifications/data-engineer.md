---
persona: data-engineer
status: resolved — environment and uptime/remote-access sub-decisions closed; Python shell
  confirmation and Odds/Yahoo credential setup remain as pre-req-checklist to-dos
---

# Data Engineer — Local System Questions

These drive decisions in `/data`: storage format/location, refresh scheduling, and whether
credential-gated sources (Yahoo, Vegas odds) are ready to integrate now or later.

## Machine & uptime

1. Is the machine you'd run this on your everyday Windows 11 desktop/laptop, and is it
   typically on/awake during a normal week (relevant for whether a Task Scheduler job set for,
   say, Tuesday night actually fires) — or does it sleep/shut down such that we should plan for
   "run on next wake" / manual-trigger fallback instead of relying on an exact scheduled time?

   *Answer:* Personal Windows machine. Typically on weekday nights and most weekend days/nights.
   It doesn't sleep — it's fully **shut down** (not sleeping) whenever it's not in use.

   ⚠️ **Flagged tension, not yet resolved:** `AGENTS.md` already confirms the weekly dashboard
   must be reachable via Tailscale/Cloudflare Tunnel when away from home on game-day weekends.
   Both of those require the machine to be powered on — a full shutdown means no tunnel, no
   dashboard, regardless of which one we pick. "Typically on most weekend days/nights" may
   already cover most game-day usage, but this needs an explicit resolution rather than an
   assumption. See open sub-decision below.

2. Is this the only machine involved, or could you see wanting to check projections from
   another device on your home network (not necessarily draft day — just casually during the
   week)?

   *Answer:* Could see checking from a work machine or phone via Tailscale/Cloudflare Tunnel —
   consistent with (and slightly broadens) the remote-access decision already in `AGENTS.md`;
   "reachable by phone" there should be read as "reachable by any of my devices," not phone-only.

## Storage

3. Preferred drive/folder for the SQLite DB and any raw pulled data (Parquet/CSV)? Any existing
   project-storage convention you use, or should we just default to a folder inside this repo?

   *Answer:* Open question turned back to us — user has WSL2 + Ubuntu already set up and is open
   to containerizing the app, asked for a recommendation on the best way to move forward. This
   is bigger than a storage-path question — it's an environment/runtime decision. See open
   sub-decision below.

4. Roughly how much free disk space is available? nflverse weekly stats are small, but if we
   pull full history back to 1999 (rather than a bounded lookback like last 5-10 years) plus
   play-by-play for red-zone-touch derivation, size adds up — worth knowing before deciding the
   pull window.

   *Answer:* 393 GB free on `C:`, 986 GB free on `Z:` (Z: is mainly used for game storage — not
   assumed as project space unless stated otherwise). Either way, plenty of headroom for full
   nflverse history + play-by-play; disk space is not a constraint on the pull-window decision.

## Environment

5. Do you already have Python installed, and do you have a preferred environment/package
   manager (venv+pip, conda/miniconda, uv, poetry)? Anything already set up on this machine we
   should reuse rather than starting fresh?

   *Answer:* Likely yes, but unconfirmed/inconsistent across shells — `which python` in Git Bash
   returns `/c/Python312/python` (native Windows Python 3.12), but running `python3` in
   PowerShell unexpectedly dropped into what looks like a **WSL2 Ubuntu bash session**
   (`warnerjc@warnerjc:/mnt/c/WINDOWS/system32$`) running Python 3.14.4, then showed
   `[1]+ Stopped python3`. That's not normal PowerShell behavior — worth untangling before
   relying on either. See confirmation commands below; this ties directly into the environment
   decision (native Windows / WSL2 / Docker) since the answer changes what "Python" even means
   on this machine.

6. Any existing SQLite tooling you use or prefer (DB Browser for SQLite, DBeaver, a VS Code
   extension) for poking at the data directly outside the app? Not required, but shapes whether
   it's worth also building lightweight SQL views for manual inspection.

   *Answer:* No strong preference; familiar with pgAdmin (Postgres-specific, not applicable to
   SQLite directly) but open to a recommendation, ideally something GUI-based that works cleanly
   across Windows/WSL2.

   **Recommendation:** [DB Browser for SQLite](https://sqlitebrowser.org/) (a.k.a. DB4S) — free,
   native Windows installer, simple spreadsheet-like GUI for browsing/editing/querying. Works
   fine whether the `.db` file lives on the native Windows filesystem or is accessed from Windows
   via the `\\wsl$\<distro>\...` path if it ends up living inside WSL2. If the environment
   decision below lands on Postgres-in-Docker instead of SQLite, pgAdmin becomes directly
   relevant again since you already know it — worth keeping in mind, not deciding now.

## Credentials

7. Do you already have an account/API key for The Odds API (or another Vegas-lines source), or
   does that need to be created as part of setup?

   *Answer:* Need to create it — request a pre-req checklist (tracked as a to-do, not answered
   here; will be produced once the environment decision below is settled so the checklist
   reflects the actual setup).

8. Do you already have a Yahoo Developer app registered (Client ID/Secret for OAuth), or is
   that still to be created? (Bootstrap plan has Yahoo as "add later" — this just confirms
   whether "later" has any head start.)

   *Answer:* Had one a while back, will most likely need to recreate it.

## Resolved sub-decisions (from the answers above)

- **Environment/runtime — WSL2 + Docker Compose**, using **Rancher Desktop** (already installed)
  rather than Docker Desktop. Pre-req-checklist item: confirm Rancher Desktop's container engine
  is set to `dockerd (moby)`, not `containerd`, so plain `docker compose` works as documented
  rather than needing `nerdctl compose`. See `AGENTS.md` → Confirmed hosting decisions.
- **Uptime vs. remote access — not solved for v1.** User mostly works from home and would rather
  not have to power the machine on just for remote access. Away-from-home dashboard access works
  opportunistically only on weekends the machine happens to already be on. Wake-on-LAN over
  Tailscale is a real future option but is *not* as simple as flipping a BIOS setting: Tailscale
  is a routed overlay, not a LAN broadcast extension, so delivering a WoL magic packet to a fully
  powered-off machine typically still needs some always-on relay on the same LAN (a Tailscale
  subnet router, or a router with native "Wake on WAN" support) — and AT&T-provided residential
  gateways are usually locked down for this kind of configurability without bridge mode. Not
  worth pursuing until it's a recurring real problem. See `AGENTS.md` → Confirmed hosting
  decisions.

## Resolved: Python confusion

- **Native Windows:** `python`/`py` correctly resolve to the real interpreter at
  `C:\Python312\python.exe` (3.12.4). `python3` resolves to a **Microsoft Store
  app-execution-alias stub**, not a real interpreter — that's what caused the odd behavior
  earlier. Use `python`/`py` on Windows, never `python3`.
- **WSL2 Ubuntu:** system `python3` is 3.14.4 (`/usr/bin/python3`), unusually new for an Ubuntu
  system interpreter — worth being aware it's not a typical LTS-default version, but not
  investigated further because it doesn't matter (next point).
- **Neither host install is what the project actually runs on.** Docker base images pin their
  own Python version, independent of the host. **Decision: pin the Docker image to Python
  3.12** (matches the native Windows install for mental-model consistency, and is a safer bet
  than 3.14 for wheel availability across LightGBM/nflverse/pandas/scikit-learn, which is a very
  recent release some scientific-computing packages may not have caught up to yet). Don't
  install project dependencies into either host Python — everything project-related happens
  inside containers.

## Resolved: where the repo lives

Moved into the WSL2-native filesystem: **`/home/warnerjc/dev/gsv-fantasy-football`** (Ubuntu
distro), copied from the prior Windows-side location. From Windows, that path is also reachable
at `\\wsl$\Ubuntu\home\warnerjc\dev\gsv-fantasy-football` (e.g. to open via VS Code's
"WSL"/"Remote Development" extension, or point Windows-side GUI tools like DB Browser for
SQLite at the DB file).

**Two copies currently exist** — the original at
`C:\Users\jcwar\Desktop\development\gsv-fantasy-football` and the new one in WSL2. The original
is *not yet deleted*. Next steps to finish this migration:

1. Re-open the project in your editor pointed at the WSL2 path (VS Code: "Reopen in WSL" /
   `code .` from inside a WSL2 terminal at `~/dev/gsv-fantasy-football`).
2. Start future Claude Code sessions from inside WSL2 at that path, so it becomes the sole
   working copy going forward (this session's tools are still anchored at the original Windows
   path — see note in chat).
3. Once you've confirmed the WSL2 copy is current and you're working from it, delete the old
   Windows-side folder. Not done automatically — that's a deliberate step for you to take once
   you've verified nothing was left behind.

## Still open (pre-req-checklist items, not architecture decisions)

- Decide + execute the repo-location move above.
- Create an Odds API account/key.
- Recreate the Yahoo Developer app (Client ID/Secret).

See [`prereq-checklist.md`](prereq-checklist.md) for the full setup checklist.
