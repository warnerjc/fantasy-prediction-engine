---
persona: product-architect
status: resolved, with an urgent scope decision raised by Q4 — see bottom
---

# Product / Architecture — Local System Questions

These are the cross-cutting questions that shape how much operational polish is worth
investing in, versus building the minimum that gets you through this draft season.

1. Is this a single-season build (get through this year's draft, iterate casually after), or
   something you intend to maintain and improve year over year? Affects whether pulling full
   nflverse history (1999+) vs. a bounded lookback is worth it, and whether DB backup/versioning
   is worth setting up now versus later.

   *Answer:* Multi-year — intended to be built on iteratively, season after season.
   **Decision:** pull full nflverse history (1999+) rather than a bounded lookback (compute
   headroom already confirmed via `ml-modeler.md`), and DB backup/versioning is worth setting
   up as real infrastructure rather than deferred — this isn't a throwaway script.

2. How many leagues do you expect to run this against at once (just your own, or also
   friends'/family leagues with different scoring settings)? Confirms the multi-scoring-config
   design in `/scoring` is actually exercised, not just theoretical.

   *Answer:* Four leagues, all the user's own: 3 regular H2H + 1 dynasty H2H. Also floated a
   longer-term possibility of releasing this as a public SaaS product eventually — explicitly
   not something to build for now, but worth a light design note (below).

   **Design notes:**
   - The `/scoring` per-league-settings design is genuinely exercised from day one (4 different
     scoring configs, not a hypothetical) — good, no speculative complexity here.
   - **Dynasty is a real scope question, not just a config variation.** The weekly start/sit
     tool works identically for dynasty (same-season projection, doesn't care how the roster
     was built). But the **draft assistant** does not translate cleanly: v1's season-total
     projection is a redraft concept. A dynasty rookie/startup draft needs multi-year asset
     value (a rookie WR's year-1 projection understates their dynasty value), which v1
     explicitly doesn't model. Flagging this now so it isn't discovered mid-draft: **v1's draft
     assistant should be scoped to the 3 redraft leagues; dynasty draft support is an explicit
     non-goal until there's a multi-year value model**, not an oversight.
   - **SaaS-later note (light touch, not a build task):** the core `/data`, `/features`,
     `/scoring`, `/models` layers are already user-agnostic (scoring settings and rosters are
     parameterized per league via API, not hardcoded to one person) — that part wouldn't need
     rework for multi-tenancy later. The single-user assumptions live in `/applications` and the
     hosting layer (one set of API credentials, local Docker, one SQLite file) — those would
     need real rework for SaaS, and that's fine; per the same principle as v1→v2, don't build
     multi-tenant auth/billing/isolation now for a maybe-later possibility. Just don't bury
     single-user assumptions inside `/data`/`/features`/`/scoring`/`/models` where they'd be
     costly to dig out later — keep them confined to `/applications`.

3. Any local environment constraints worth knowing up front — corporate laptop restrictions,
   antivirus/firewall software that tends to flag scheduled scripts or local web servers, WSL
   availability if we ever wanted a Linux-like environment instead of native Windows?

   *Answer:* Resolved via `data-engineer.md` — WSL2 with Ubuntu is already set up, and Rancher
   Desktop (not Docker Desktop) is already installed. This became the confirmed runtime decision:
   WSL2 + Docker Compose via Rancher Desktop. See `AGENTS.md` → Confirmed hosting decisions.
   Also confirmed here: personal machine, no corporate/antivirus constraints to design around.

4. Rough timeline pressure: how soon is "your upcoming draft"? This determines how much of v1
   needs to be genuinely done (versus a manual/spreadsheet fallback being acceptable for this
   season while v1 finishes) and sequences the specification/build work accordingly.

   *Answer:* Originally stated as "5 days out"; corrected to **Sleeper league drafts Wednesday,
   September 2, 2026** (6 days out from the 2026-08-27 planning session). There is also a
   **second draft, a Yahoo redraft league, date/time not yet known.**

   ⚠️ **Real scope problem, resolved as a deliberate sprint plan** (not silently absorbed, not
   silently cut). Zero implementation existed as of this session — everything so far was
   architecture/environment planning. Resolved approach, agreed with the user:

   - **AI-assisted build**, coding done by Claude across remaining sessions
     (2026-08-27 night: planning wrap-up · Fri 08-28: core pipeline/model build ·
     Sun 08-30 & Mon 08-31 nights: model refinement + draft-day app ·
     Tue 09-01: dry run + buffer · Wed 09-02: Sleeper draft).
   - **Scope narrowed to draft-assistant only** for this sprint — weekly start/sit tool, v2
     quantiles/adjustment layer, and remote access (Tailscale/Cloudflare) all wait; none are
     needed before either draft.
   - **Docker/containerization deferred past the sprint.** Build and run directly in a WSL2
     Python venv this week; wrap it in the already-decided WSL2 + Docker Compose setup
     afterward once there's no deadline pressure. This is a temporary sprint-only deviation
     from the confirmed runtime decision in `AGENTS.md`, not a reversal of it.
   - **nflverse pull window trimmed to ~5-10 seasons for the sprint model**, not the full
     1999+ history — full backfill happens later at leisure (compute isn't the constraint,
     time is).
   - **Actual league/platform breakdown, now confirmed with real IDs** (correcting an earlier
     assumption that the 3 "regular H2H" leagues were all on Sleeper): 1 Sleeper redraft
     (`1356741521163968513`, drafts Wed 2026-09-02) + 1 Sleeper dynasty (`1314665599955132416`,
     already drafted, out of scope) + 2 Yahoo redraft (`775326`, already drafted, needs nothing;
     `236625` "keepitcrooked", offline draft type, date TBD). **So the sprint's draft-assistant
     work serves exactly 2 leagues**, not 4 — smaller scope than originally planned. Full
     registry and the Yahoo scoring config: `draft-sprint-plan.md`.
   - **Yahoo support this sprint = manual scoring-config input, not real API/OAuth
     integration.** User will pull the Yahoo league's scoring settings manually and hand them
     to the tool as a config dict — reuses the same `/scoring` + model + ranking pipeline with
     zero OAuth risk. Real Yahoo API integration (auto-pull, live polling) stays deferred to
     after both drafts, consistent with the original "Yahoo — add later" plan; it just no
     longer blocks having a working tool for that draft.
   - **Live auto-updating-as-picks-happen UI stays Sleeper-only** for the sprint (only Sleeper
     has a public no-auth API for live draft state) and is itself a stretch goal, not
     guaranteed. The Yahoo draft, and Sleeper if live polling doesn't get finished in time,
     both fall back to the same static ranked list with manual pick tracking — this fallback
     is not a separately-built v0, it's simply the model's natural output before a live-UI
     layer is added on top.

   Full day-by-day build plan: [`draft-sprint-plan.md`](draft-sprint-plan.md).
