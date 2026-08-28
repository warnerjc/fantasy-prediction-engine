---
persona: app-engineer
status: partially resolved
---

# Application Engineer — Local System Questions

These decide whether a purely local-only tool (e.g., Streamlit bound to localhost) is
sufficient, or whether draft-day/weekly usage patterns require reachability beyond the machine
running the pipeline.

## Draft day

1. On draft day, what device will you actually have the draft assistant open on — the same
   Windows machine that runs the pipeline, a different laptop, or your phone?

   *Answer:* Same Windows machine — keeping it simple for now.

2. Will drafts typically happen from home (same network as the machine running this), or could
   you be drafting from somewhere else (in-person draft night, a friend's place, travel)? This
   is the key fork: local-network-only is a much smaller build than needing remote access
   (e.g., a Tailscale/VPN tunnel, or hosting somewhere reachable from anywhere).

   *Answer:* Same network, every time.

3. During a live draft, do you want something you keep open in a browser tab throughout (a
   dashboard that updates as picks happen), or something you re-run per-pick (CLI/script you
   invoke to get the current recommendation)?

   *Answer:* Browser tab that automatically updates after each pick.

## Weekly start/sit

4. For the weekly tool, is a "run a script, read the output" workflow (CLI output or a
   generated report file) good enough, or do you want a persistent dashboard you can check back
   on throughout the week as injury news/Vegas lines move?

   *Answer:* Persistent dashboard, checked throughout the week.

5. Would you want this reachable from your phone during the week (e.g., checking start/sit
   Sunday morning away from your desk), or is "at my desk on my usual machine" the realistic
   usage pattern?

   *Answer:* Mostly at the desk, but needs to be reachable from phone too — some game-day
   weekends I'm away from home and start/sit calls have to happen from my phone.

## Design implications from the above

- **Draft tool is local-network-only** — no remote access needed for draft day. Simplifies that
  half of the build considerably: a Streamlit (or similar) app bound to the local machine,
  reachable at `localhost`/LAN IP from the same machine, is sufficient.
- **"Auto-updates after each pick" means the draft tool needs to poll live draft state**, not
  just compute once and sit static. Sleeper doesn't push webhooks for draft events (as of the
  current API), so this will be a polling loop against the draft endpoint with the UI
  auto-refreshing (e.g., Streamlit's autorefresh pattern) rather than a one-shot render.
- **"Persistent dashboard" means the weekly tool is a long-running local service**, not a
  script invoked on demand — it needs to stay up across the week, which has implications for
  how it's started (auto-start on boot / restart-on-crash) and ties back to the uptime question
  in `data-engineer.md` Q1.
- **Phone reachability during the week is a real fork from "local network only."** Unlike the
  draft tool, the weekly dashboard needs to be reachable from outside the home network on
  weekends away from home. **Resolved:** Tailscale or Cloudflare Tunnel — no port forwarding,
  no open inbound ports on the home router. Either avoids exposing the home network directly;
  Tailscale needs the Tailscale app installed on the phone but no domain, Cloudflare Tunnel
  needs a domain but is reachable from a plain browser bookmark with no app install. Final pick
  between the two can happen at implementation time — both satisfy the "no exposed ports"
  requirement. See `AGENTS.md` → Confirmed hosting decisions.
