"""Snake-draft assistant: a value-over-replacement board that updates live.

    python -m applications.draft_tool --league sleeper --slot 7 --watch
    python -m applications.draft_tool --league yahoo            # static board, manual draft

Reads projections from models/output/<league>_projections.csv — run
`python -m models.build --league <league>` first. Auction drafts are not
supported yet (different logic — budget pacing, not positional scarcity).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from models.leagues import CONFIG_DIR, LEAGUES
from . import sleeper
from .board import build_board
from .roster import roster_spec

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

_LEAGUE_ID = {"sleeper": "1356741521163968513", "yahoo": "236625"}
_COLS = ["overall_rank", "tier", "name", "position", "most_recent_team", "proj_ppg", "vbd"]


def _load_config(league: str) -> dict:
    fname = LEAGUES[league][0]
    return json.loads((CONFIG_DIR / fname).read_text())


def _fmt(df: pd.DataFrame) -> str:
    show = df[[c for c in _COLS if c in df.columns]].copy()
    for c in ("proj_ppg", "vbd"):
        if c in show:
            show[c] = show[c].round(1)
    return show.to_string(index=False)


def _render(board, state: dict | None, slot: int | None) -> None:
    print(f"\n{'=' * 78}")
    if state:
        print(f"pick {state['next_pick_no']} / {state['teams'] * state['rounds']}  "
              f"({len(state['picks'])} made)   draft {state['status']}")
    print(f"{'=' * 78}\nBEST AVAILABLE (by value over replacement)\n")
    print(_fmt(board.top(20)))

    print("\nBEST AT EACH POSITION\n")
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        p = board.by_position(pos, 5)
        if not p.empty:
            names = "  ".join(f"{r.name} ({r.proj_ppg:.1f})" for r in p.itertuples())
            print(f"  {pos:>4}  {names}")

    if slot and state:
        t = board.my_targets(slot, state["teams"], state["rounds"], state["next_pick_no"])
        if t["my_next_pick"]:
            print(f"\nYOUR NEXT PICK: #{t['my_next_pick']}  ({t['picks_until_mine']} picks away)")
            print("  likely available to you:")
            print(_fmt(t["likely_available"]).replace("\n", "\n  "))


def run(league: str, slot: int | None, watch: bool, draft_id: str | None, interval: int) -> None:
    cfg = _load_config(league)
    spec = roster_spec(cfg)
    board = build_board(league, spec)
    print(f"{league}: {spec.teams} teams, replacement ranks {spec.replacement_rank()}")

    if league == "yahoo" or not watch:
        state = None
        if league == "sleeper":
            try:
                did = draft_id or sleeper.draft_id_for_league(_LEAGUE_ID["sleeper"])
                st = sleeper.draft_state(did)
                board = board.with_drafted(sleeper.drafted_sleeper_ids(did))
                state = st
            except Exception as e:  # offline / API down -> static board still works
                print(f"(no live draft state: {e})")
        _render(board, state, slot)
        if league == "yahoo":
            print("\nYahoo offline draft — check picks off this list manually.")
        return

    did = draft_id or sleeper.draft_id_for_league(_LEAGUE_ID["sleeper"])
    print(f"watching Sleeper draft {did}  (every {interval}s, Ctrl-C to stop)")
    seen = -1
    while True:
        try:
            st = sleeper.draft_state(did)
            if len(st["picks"]) != seen:
                seen = len(st["picks"])
                _render(board.with_drafted({p["player_id"] for p in st["picks"] if p.get("player_id")}),
                        st, slot)
            if st["status"] == "complete":
                print("\ndraft complete.")
                return
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=["sleeper", "yahoo"])
    ap.add_argument("--slot", type=int, default=None, help="your draft position (1 = first overall)")
    ap.add_argument("--watch", action="store_true", help="poll live Sleeper draft state")
    ap.add_argument("--draft", default=None, help="Sleeper draft id (default: look up from league)")
    ap.add_argument("--interval", type=int, default=15, help="poll seconds when --watch")
    args = ap.parse_args()
    run(args.league, args.slot, args.watch, args.draft, args.interval)


if __name__ == "__main__":
    main()
