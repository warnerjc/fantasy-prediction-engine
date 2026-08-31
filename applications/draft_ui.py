"""Browser draft assistant. A tiny Flask server + one auto-refreshing page.

    python -m applications.draft_ui --league sleeper --draft <id> --slot 4
    # then open http://localhost:8000

A background thread polls the Sleeper draft every few seconds; when a pick lands
it recomputes the recommendation (and, on a round boundary, re-runs the strategy
simulation). The page just fetches `/state` and redraws — nothing blocks.

Recommendations never tell you to reach: a player whose ADP is well past your
next pick is tagged "could wait" and listed separately. Cross-check the room's
own ADP board for timing.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback

from flask import Flask, jsonify

from models.leagues import load_rules

from . import sleeper
from .draft_tool import _build, _load_config, _projection_season
from .draftplan import choose_initial_strategy, evaluate, recommend
from .roster import roster_spec

POLL_SECONDS = 1.5            # ~1.3 Sleeper req/s (draft_state = 2 calls) — safe, and mocks fly
EVAL_THROUGH_ROUND = 11       # strategy sim only matters while the plan is still live
EVAL_SIMS = 14


class AppState:
    def __init__(self, league: str, draft_id: str, my_slot: int):
        self.league, self.draft_id, self.my_slot = league, draft_id, my_slot
        cfg = _load_config(league)
        self.spec = roster_spec(cfg)
        self.rules = load_rules(league)
        self.rounds = None
        self.board = _build(league, self.spec, use_adp=True,
                            season=_projection_season(league), blend=0.7)
        self.strategy, why = choose_initial_strategy(self.spec, self.rules)
        self.strategy_log = [{"round": 0, "to": self.strategy.name, "why": why}]
        self.lock = threading.Lock()
        self.payload = {"status": "starting"}
        self.seen_picks = -1
        self.evaluated_round = -1
        self.eval_thread: threading.Thread | None = None
        self.first_eval_done = False
        self.live_pick = 0          # latest pick count seen from Sleeper (every poll)
        self.computed_ts = 0.0      # when `payload` was last recomputed

    # -- the poll/recommend loop (fast, never blocks on the sim) ---------
    def run_forever(self):
        while True:
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
            time.sleep(POLL_SECONDS)

    def _drafted_board(self, st):
        return self.board.with_drafted(
            {str(p["player_id"]) for p in st["picks"] if p.get("player_id")})

    def _tick(self):
        st = sleeper.draft_state(self.draft_id)
        self.rounds = st.get("rounds") or self.rounds or 16
        n = len(st["picks"])
        self.live_pick = n
        if st["status"] == "pre_draft":
            self._set({"status": "pre_draft", "strategy": self._strat_blob(),
                       "message": "Waiting for the draft to start…"})
            return
        if n == self.seen_picks and self.payload.get("status") != "starting":
            return

        board = self._drafted_board(st)
        rec = recommend(board, st, self.my_slot, self.spec, self.strategy, self.rounds)
        self.seen_picks = n                     # only after a successful compute
        my_next = rec.get("my_next_pick")
        your_turn = my_next == rec["current_pick"]
        rec.update(status=st["status"], picks_made=n,
                   total_picks=self.spec.teams * self.rounds,
                   strategy=self._strat_blob(), your_turn=your_turn,
                   refining=self._eval_running())
        self.computed_ts = time.time()
        self._set(rec)
        # only re-check strategy when you have a comfortable buffer — never in the
        # picks right before your turn (recs must be stable when you're on the clock)
        if my_next and (my_next - n) > 5:
            self._maybe_eval(st, n)

    # -- strategy simulation, off the hot path --------------------------
    def _eval_running(self) -> bool:
        return self.eval_thread is not None and self.eval_thread.is_alive()

    def _maybe_eval(self, st, n):
        cur_round = n // self.spec.teams + 1
        if (cur_round == self.evaluated_round or cur_round > EVAL_THROUGH_ROUND
                or st["status"] == "complete" or self._eval_running()):
            return
        self.evaluated_round = cur_round
        self.eval_thread = threading.Thread(
            target=self._do_eval, args=(self._drafted_board(st), st, cur_round), daemon=True)
        self.eval_thread.start()

    def _do_eval(self, board, st, cur_round):
        try:
            ev = evaluate(board, st, self.my_slot, self.spec, self.rounds,
                          self.rules, n_sims=EVAL_SIMS)
            # first sim: adopt the winner outright (the pre-draft guess was just a
            # placeholder). after that: small hysteresis so noise doesn't flip-flop.
            change = ev["best"].name != self.strategy.name
            if change and (not self.first_eval_done or ev["gap"] >= 4):
                with self.lock:
                    self.strategy_log.append({"round": cur_round, "to": ev["best"].name,
                                              "why": "sim" if not self.first_eval_done
                                              else f"sim +{ev['gap']:.0f} pts"})
                self.strategy = ev["best"]
            self.first_eval_done = True
        except Exception:
            traceback.print_exc()

    def _strat_blob(self):
        with self.lock:
            return {"name": self.strategy.name, "blurb": self.strategy.blurb,
                    "log": list(self.strategy_log)}

    def _set(self, payload):
        with self.lock:
            self.payload = payload

    def get(self):
        with self.lock:
            return dict(self.payload)


def create_app(state: AppState) -> Flask:
    app = Flask(__name__)

    @app.get("/state")
    def _state():
        p = state.get()
        p["live_pick"] = state.live_pick
        caught_up = p.get("picks_made") == state.live_pick
        p["caught_up"] = caught_up
        stale_for = round(time.time() - state.computed_ts, 1) if state.computed_ts else None
        # what the user cares about: can I trust the recs shown right now?
        #   updating -> a pick landed, recs are being recomputed (wait ~1-2s)
        #   ready    -> recs match the live board; strategy is frozen for this pick
        #   stuck    -> no successful recompute in >12s while the draft is live
        if not caught_up and stale_for and stale_for > 12:
            p["engine_state"] = "stuck"
        else:
            p["engine_state"] = "updating" if not caught_up else "ready"
        p["stale_for"] = stale_for
        return jsonify(p)

    @app.get("/")
    def _index():
        return _PAGE, 200, {"Cache-Control": "no-store"}

    return app


_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>draft</title><style>
 :root{color-scheme:dark}
 body{margin:0;background:#12141a;color:#e8e8ea;font:16px/1.4 system-ui,sans-serif}
 #app{max-width:760px;margin:0 auto;padding:18px}
 h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#8a8f9a;margin:22px 0 8px}
 .hdr{font-size:15px;color:#b8bcc4;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}
 .clock{font-size:26px;font-weight:700;color:#fff}
 .strat{background:#1c1f27;border-radius:10px;padding:12px 14px;margin-top:10px}
 .strat b{color:#7db4ff;font-size:17px}
 .strat .why{color:#9aa0ab;font-size:14px;margin-top:2px}
 .strat .log{color:#6f7580;font-size:12px;margin-top:6px}
 .card{background:#1c1f27;border-left:4px solid #7db4ff;border-radius:8px;padding:10px 13px;margin:7px 0;display:flex;justify-content:space-between;align-items:baseline;gap:10px}
 .card .nm{font-size:19px;font-weight:700}
 .card .meta{color:#9aa0ab;font-size:13px}
 .card .tm{text-align:right;font-size:13px;color:#c7ccd4;white-space:nowrap}
 .card.wait{border-left-color:#4a4f5a;opacity:.85}
 .card.wait .nm{font-size:16px;font-weight:600}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td,th{text-align:left;padding:5px 8px;border-bottom:1px solid #23262f}
 th{color:#8a8f9a;font-weight:600}
 .run{color:#ff9b5c;font-weight:700}
 .roster span{display:inline-block;background:#1c1f27;border-radius:6px;padding:3px 9px;margin:3px 4px 0 0;font-size:13px}
 .big{font-size:20px;color:#fff;margin:30px 0}
 .banner{border-radius:10px;padding:14px 16px;margin:10px 0 12px;font-size:21px;font-weight:700;border:2px solid transparent}
 .banner.ready{background:#173a1f;color:#8be8a4;border-color:#2f7a44}
 .banner.updating{background:#3a2e12;color:#ffcf8a;border-color:#c98f28;animation:pulse 1s ease-in-out infinite}
 .banner.idle{background:#1c1f27;color:#c7ccd4;font-weight:600;font-size:16px}
 @keyframes pulse{0%,100%{border-color:#c98f28}50%{border-color:#6b4f16}}
 .sub{font-size:12px;color:#6f7580;margin:-6px 0 12px}
 .take{font-size:16px;color:#e8e8ea;background:#1c1f27;border-radius:8px;padding:11px 14px;margin:0 0 14px}
 .beat{font-size:11px;color:#5a5f6a;text-align:right;margin-bottom:6px}
 .beat.dead{color:#ff7a7a}
</style></head><body><div id=beat class=beat>connecting…</div><div id=app>loading…</div>
<script>
const g=id=>document.getElementById(id)
const num=x=>(x==null||Number.isNaN(x))?'–':x
function card(r,wait){
 return `<div class="card${wait?' wait':''}">
  <div><div class=nm>${r.name}</div><div class=meta>${r.position} · ${r.team||''} · proj ${num(r.proj_ppg)} · ADP ${num(r.adp)}</div></div>
  <div class=tm>${wait? (r.note||'') : (r.timing||'')}${r.why&&!wait?'<br>'+r.why:''}</div></div>`
}
function land(L){
 let rows=Object.entries(L||{}).map(([p,v])=>`<tr><td>${p}</td>
  <td>${num(v.startable_left)} left ${v.running?'<span class=run>· RUN</span>':''}</td>
  <td>${v.best? v.best.name+' ('+num(v.best.proj_ppg)+')':'–'}</td>
  <td>${v.gone_by_next||0} gone by your pick</td></tr>`).join('')
 return `<table><tr><th>pos</th><th>startable</th><th>best available</th><th></th></tr>${rows}</table>`
}
let lastOk=Date.now()
function render(s){
 const a=g('app')
 if(s.status==='pre_draft'||s.status==='starting'){
   a.innerHTML=`<div class=strat><b>${s.strategy?.name||'…'}</b><div class=why>${s.strategy?.blurb||''}</div></div>
   <div class=big>${s.message||'Starting…'}</div>`; return }
 const st=s.engine_state, yt=s.your_turn
 let banner
 if(st==='stuck') banner=`<div class="banner updating">⚠ engine stuck (${s.stale_for}s) — check the terminal, or use Sleeper's ADP for this pick</div>`
 else if(yt && st==='ready') banner=`<div class="banner ready">✅ YOUR PICK #${s.current_pick} — recs ready</div>`
 else if(yt) banner=`<div class="banner updating">⏳ YOUR PICK #${s.current_pick} — updating…</div>`
 else if(st!=='ready') banner=`<div class="banner idle">pick ${s.live_pick} · catching up…</div>`
 else banner=`<div class="banner idle">pick ${s.current_pick} · you're up at #${num(s.my_next_pick)} (${num(s.picks_until_next)} away)</div>`
 const log=s.strategy?.log||[]
 let h=`<div class=hdr><div>pick <b>${s.current_pick}</b> / ${s.total_picks} · ${s.status}</div><div>Round ${s.round}</div></div>
  ${banner}
  <div class=strat><b>${s.strategy?.name||'…'}</b><div class=why>${s.strategy?.blurb||''}</div>
   ${log.length>1?`<div class=log>${log.map(l=>l.round?`R${l.round}→${l.to} (${l.why})`:`start: ${l.to}`).join(' · ')}</div>`:''}</div>`
 if(s.takeaway) h+=`<div class=take>💡 ${s.takeaway}</div>`
 h+=`<h2>Draft now</h2>${(s.recommendations||[]).map(r=>card(r,false)).join('')}`
 if((s.wait||[]).length) h+=`<h2>Can wait — target later</h2>${s.wait.map(r=>card(r,true)).join('')}`
 const rc=s.my_roster?.players||{}
 h+=`<h2>Your roster (${s.my_roster?.n||0})</h2><div class=roster>`+
   Object.entries(rc).map(([p,ns])=>`<span>${p}: ${ns.join(', ')}</span>`).join('')+`</div>`
 h+=`<h2>Landscape</h2>${land(s.landscape)}`
 a.innerHTML=h
}
async function tick(){
 let s
 try{ const r=await fetch('/state',{cache:'no-store'}); s=await r.json() }
 catch(e){ g('beat').className='beat dead'; g('beat').textContent='server unreachable'; return }
 lastOk=Date.now()
 try{ render(s) }
 catch(e){ g('app').innerHTML=`<div class=take>render error: ${e.message} — recs may be stale</div>`+g('app').innerHTML }
 g('beat').className='beat'
 g('beat').textContent=`updated ${new Date().toLocaleTimeString()}`
}
tick(); setInterval(tick,1500)
setInterval(()=>{ if(Date.now()-lastOk>6000){ g('beat').className='beat dead'; g('beat').textContent='no update in '+Math.round((Date.now()-lastOk)/1000)+'s' } },2000)
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=["sleeper", "yahoo"])
    ap.add_argument("--draft", required=True, help="Sleeper draft id (from the draft-room URL)")
    ap.add_argument("--slot", type=int, required=True, help="your draft position (1 = first)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    state = AppState(args.league, args.draft, args.slot)
    threading.Thread(target=state.run_forever, daemon=True).start()
    print(f"draft assistant: http://localhost:{args.port}  "
          f"(league {args.league}, draft {args.draft}, slot {args.slot})")
    create_app(state).run(port=args.port, threaded=True)


if __name__ == "__main__":
    main()
