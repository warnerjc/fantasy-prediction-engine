"""Draft strategy selection + pick recommendation.

Sits on top of the value board (``board.DraftBoard``). Given the live draft state
and your slot it (1) picks a draft strategy and re-checks it each round by
simulating the rest of the draft under every strategy, and (2) recommends players
that fit that strategy **and are available at or after their ADP** -- it never
tells you to reach. A player the model likes who should last goes on a "wait"
list tagged with your next pick number, and you cross-check the room's ADP board.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .board import DraftBoard, _snake_pick_numbers
from .mock import DraftStart, roster_value, simulate_draft
from .roster import SCORABLE, RosterSpec

# --- strategies --------------------------------------------------------------

_BANDS = ((1, 3), (4, 7), (8, 11), (12, 99))   # round bands the weights are keyed on


@dataclass(frozen=True)
class Strategy:
    name: str
    blurb: str
    weights: dict[str, tuple[float, float, float, float]]   # position -> weight per band

    def weight(self, position: str, rnd: int) -> float:
        w = self.weights.get(position, (1.0, 1.0, 1.0, 1.0))
        for i, (lo, hi) in enumerate(_BANDS):
            if lo <= rnd <= hi:
                return w[i]
        return w[-1]


# Strategy is a *tilt*, not a script: a small early-rounds nudge toward a position
# group. `evaluate` runs the rest-of-draft simulator from your actual slot + roster
# + board state under each tilt and picks the one that ends with the best starting
# lineup -- so the choice is search output, not a table anyone hand-fit. Weights
# stay near 1.0 in round 1 so a value that falls to you is never passed over.
STRATEGIES: dict[str, Strategy] = {
    "bpa": Strategy("bpa", "best player available — no positional tilt", {}),
    "rb_early": Strategy("rb_early", "lean RB the first few rounds",
                         {"RB": (1.25, 1.2, 0.95, 1.0), "WR": (0.95, 1.0, 1.05, 1.0)}),
    "wr_early": Strategy("wr_early", "lean WR/TE early, RB from the mid rounds",
                         {"WR": (1.2, 1.3, 0.95, 1.0), "RB": (0.9, 0.72, 1.25, 1.05),
                          "TE": (1.1, 1.1, 0.95, 1.0)}),
    "qb_early": Strategy("qb_early", "grab a top QB early (superflex)",
                         {"QB": (1.35, 1.2, 0.9, 1.0)}),
}


def _ppr(rules) -> float:
    return float(rules.per_unit.get("rec", 0.0))


def _superflex(spec: RosterSpec) -> bool:
    return spec.dedicated.get("QB", 0) + sum(1 for f in spec.flex if "QB" in f) >= 2


def candidate_strategies(spec: RosterSpec, rules=None) -> list[str]:
    """Tilts `evaluate` weighs. `qb_early` is only in play for superflex."""
    names = ["bpa", "rb_early", "wr_early"]
    if _superflex(spec):
        names.append("qb_early")
    return names


def _effective_slots(spec: RosterSpec) -> dict[str, float]:
    """Startable slots per position = dedicated + expected flex share."""
    alloc = spec.flex_allocation()
    return {p: spec.dedicated.get(p, 0) + alloc.get(p, 0.0) for p in SCORABLE}


def choose_initial_strategy(spec: RosterSpec, rules) -> tuple[Strategy, str]:
    """A cheap first guess, used only until the first `evaluate` runs (right after
    your round-1 pick). The simulator takes over from there."""
    if _superflex(spec):
        return STRATEGIES["qb_early"], "superflex — starting guess, sim confirms after round 1"
    slots = _effective_slots(spec)
    rb, wr = slots["RB"], slots["WR"]
    if wr >= 2.6 and rb <= 1.8:
        return STRATEGIES["wr_early"], (f"{wr:.1f} WR / {rb:.1f} RB startable slots — "
                                       "starting guess: WR-lean; sim confirms after round 1")
    if rb >= 2.4:
        return STRATEGIES["rb_early"], f"{rb:.1f} RB slots — starting guess: RB-lean; sim confirms"
    return STRATEGIES["bpa"], "starting guess: best available; sim confirms after round 1"


# --- draft-state helpers ----------------------------------------------------

def _pos_by_id(board: DraftBoard) -> dict[str, str]:
    return dict(zip(board.players["sleeper_id"].astype(str), board.players["position"]))


def _pick_pos(pick: dict, pos_by_id: dict[str, str]) -> str | None:
    pid = pick.get("player_id")
    return pos_by_id.get(str(pid)) or (pick.get("metadata") or {}).get("position")


def _start_state(board: DraftBoard, picks: list[dict], spec: RosterSpec) -> DraftStart:
    pos_by_id = _pos_by_id(board)
    rosters = {s: Counter() for s in range(1, spec.teams + 1)}
    for p in picks:
        slot, pos = p.get("draft_slot"), _pick_pos(p, pos_by_id)
        if slot in rosters and pos:
            rosters[slot][pos] += 1
    return DraftStart(rosters=rosters, pick_no=len(picks) + 1)


def _my_drafted_rows(board: DraftBoard, picks: list[dict], my_slot: int) -> pd.DataFrame:
    ids = {str(p["player_id"]) for p in picks
           if p.get("draft_slot") == my_slot and p.get("player_id")}
    return board.players[board.players["sleeper_id"].astype(str).isin(ids)]


# --- strategy re-evaluation (simulation) -----------------------------------

def evaluate(board: DraftBoard, draft_state: dict, my_slot: int, spec: RosterSpec,
             rounds: int, rules=None, n_sims: int = 16, seed: int = 0) -> dict:
    """Simulate the rest of the draft from your actual slot + roster + board state
    under each tilt; return the one whose starting lineup ends up most valuable.
    The choice is search output — no hand-picked default."""
    start = _start_state(board, draft_state["picks"], spec)
    my_rows = _my_drafted_rows(board, draft_state["picks"], my_slot)
    rng = np.random.default_rng(seed)

    scores: dict[str, float] = {}
    for name in candidate_strategies(spec, rules):
        strat = STRATEGIES[name]
        pts = []
        for _ in range(n_sims):
            _, mine = simulate_draft(
                board, spec, my_slot, rounds,
                np.random.default_rng(rng.integers(1 << 32)),
                my_strategy=strat, start=start,
            )
            full = pd.concat([my_rows, mine], ignore_index=True) if not mine.empty else my_rows
            pts.append(roster_value(full, spec))
        scores[name] = float(np.mean(pts)) if pts else 0.0

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
    return {"best": STRATEGIES[ranked[0][0]], "ranked": ranked, "gap": gap}


# --- recommendation -------------------------------------------------------

_startable = _effective_slots

# how much bench depth is actually worth drafting beyond the starting lineup. RB/WR
# want real depth (byes, flex, injuries); a backup TE / QB (1-QB) barely plays.
_DEPTH_TARGET = {"RB": 2.3, "WR": 2.3}


def _targets(spec: RosterSpec) -> dict[str, float]:
    s = _effective_slots(spec)
    return {p: s[p] + _DEPTH_TARGET.get(p, 0.0) for p in SCORABLE}


def _fit(pos: str, have: int, targets: dict[str, float], rnd: int, rounds: int,
         one_and_done: set[str]) -> float:
    """How much you still want another `pos`, ~0.05–1.0."""
    if pos in ("K", "DEF"):
        return 1.0 if rnd >= rounds - 2 else 0.04
    if pos in one_and_done:                 # backup barely plays -> stop wanting one
        return 1.0 if have < 1 else 0.1
    t = max(targets.get(pos, 1.0), 1.0)
    return max(0.15, 1.0 - have / t)


def _scarcity(startable_left: int, picks_until_next: int, teams: int, running: bool) -> float:
    """Boost a position that will thin out before your next pick. `startable_left`
    is the count of players still in a startable tier at that position."""
    # rough count of picks that will land on this position before you're up again
    pressure = max(1, round(picks_until_next * 0.35))
    if startable_left <= pressure:
        boost = 2.2          # runs dry before your next pick — last chance
    elif startable_left <= pressure + 3:
        boost = 1.5
    else:
        boost = 1.0
    return boost * (1.2 if running else 1.0)


def _landscape(board: DraftBoard, picks: list[dict], spec: RosterSpec,
               my_next: int | None) -> dict:
    pos_by_id = _pos_by_id(board)
    recent = Counter(_pick_pos(p, pos_by_id) for p in picks[-spec.teams:])
    # only players we have a market read on — a no-ADP name (injured, buried,
    # or just un-ranked) isn't part of "what's left and when does it go"
    avail = board.available[board.available["adp"].notna()]
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        pv = avail[avail["position"] == pos].sort_values("vbd", ascending=False)
        best = pv.head(1)
        out[pos] = {
            "startable_left": int((pv["vbd"] > 0).sum()),
            "gone_by_next": int((pv["adp"] < my_next).sum()) if my_next else 0,
            "running": recent.get(pos, 0) >= max(3, spec.teams // 3),
            "best": None if best.empty else {
                "name": best["name"].iloc[0],
                "proj_ppg": _round(best["proj_ppg"].iloc[0]),
                "adp": _round(best["adp"].iloc[0]),
            },
        }
    return out


def recommend(board: DraftBoard, draft_state: dict, my_slot: int, spec: RosterSpec,
              strategy: Strategy, rounds: int) -> dict:
    picks = draft_state["picks"]
    cur = len(picks) + 1
    rnd = (cur - 1) // spec.teams + 1
    snake = _snake_pick_numbers(my_slot, spec.teams, rounds)
    my_next = next((p for p in snake if p >= cur), None)

    pos_by_id = _pos_by_id(board)
    my_counts = Counter(_pick_pos(p, pos_by_id) for p in picks
                        if p.get("draft_slot") == my_slot)
    targets = _targets(spec)
    one_and_done = {"TE"} | ({"QB"} if not _superflex(spec) else set())

    land = _landscape(board, picks, spec, my_next)
    until_next = (my_next - cur) if my_next else spec.teams
    # scarcity only matters where you still want another body at the position
    scar = {p: (_scarcity(land[p]["startable_left"], until_next, spec.teams, land[p]["running"])
                if my_counts.get(p, 0) < targets.get(p, 0) - 0.5 else 1.0)
            for p in ("QB", "RB", "WR", "TE")}

    avail = board.available
    cand = avail[avail["adp"].notna()].copy()                 # need an ADP to reason about timing
    if cand.empty:
        cand = avail.copy()

    shift = cand["vbd"].min()
    cand["wt"] = cand["position"].map(lambda p: strategy.weight(p, rnd))
    cand["ft"] = cand["position"].map(
        lambda p: _fit(p, my_counts.get(p, 0), targets, rnd, rounds, one_and_done))
    cand["scar"] = cand["position"].map(lambda p: scar.get(p, 1.0))
    # a run you've already missed isn't scarcity: once the startable tier at a
    # position is gone, a below-replacement body there shouldn't get a scarcity
    # boost that lifts it over real value at another position (the Ollie Gordon
    # "RB is running, take him now" at -17 VBD case)
    cand["scar"] = np.where(cand["vbd"] > 0, cand["scar"], 1.0)
    # a player with only an ADP and no projection is a pure dart -- don't lead with
    # one when projected players are on the board
    cand["dart"] = np.where(cand["proj_ppg"].isna() | (cand["source"] == "adp"), 0.55, 1.0)
    # never-reach: a player whose ADP is past your next pick can be had later, so
    # drafting him now wastes the pick. Steep -- if everything at a position you
    # "need" will clearly last, the tool should pivot to where value is going now.
    horizon = (my_next or cur) + 4
    cand["urg"] = np.clip(horizon / cand["adp"].clip(lower=1), 0.25, 1.0)
    # lightly compress the value spread so need/scarcity/timing can reorder *within*
    # a value band -- but not so much that a falling elite gets passed over
    cand["val"] = (cand["vbd"] - shift + 1.0) ** 0.85
    cand["sc"] = cand["wt"] * cand["ft"] * cand["scar"] * cand["urg"] * cand["dart"] * cand["val"]

    ranked = cand.sort_values("sc", ascending=False)
    # elite-value override: if the best raw value on the board towers over the rest
    # (a top pick fell to you), recommend him regardless of strategy.
    by_vbd = cand.sort_values("vbd", ascending=False)
    if len(by_vbd) >= 4:
        v = by_vbd["vbd"].to_numpy()
        if v[0] - v[3] > 22 and by_vbd.iloc[0]["name"] != ranked.iloc[0]["name"]:
            elite = by_vbd.iloc[[0]]
            ranked = pd.concat([elite, ranked[ranked["name"] != elite.iloc[0]["name"]]])
    takeaway, lead = _takeaway(ranked, cand, land, my_counts, targets, my_next)
    # the headline names the pick to make — put that player at the top of the cards
    recs = ranked.head(6)
    if lead is not None and lead in set(cand["name"]):
        recs = pd.concat([cand[cand["name"] == lead], recs[recs["name"] != lead]]).head(6)

    # what waiting buys you: strong strategy-fit players whose ADP says they last
    will_last = cand[cand["adp"] > horizon].copy()
    will_last["sc2"] = will_last["wt"] * will_last["ft"] * (will_last["vbd"] - shift + 1.0)
    wait = will_last.sort_values("sc2", ascending=False).head(6)

    return {
        "current_pick": cur, "round": rnd, "my_next_pick": my_next,
        "picks_until_next": (my_next - cur) if my_next else None,
        "strategy": {"name": strategy.name, "blurb": strategy.blurb},
        "my_roster": _roster_summary(board, picks, my_slot, spec),
        "takeaway": takeaway,
        "recommendations": [_rec_row(r, cur, my_next) for r in recs.itertuples()],
        "wait": [_wait_row(r, my_next) for r in wait.itertuples()],
        "landscape": land,
    }


def _takeaway(ranked, cand, land, my_counts, targets, my_next) -> tuple[str, str | None]:
    """One-line strategic read + the player name it points at (to lead the cards)."""
    if ranked.empty:
        return "Board's thin — take the best player left.", None
    top = ranked.iloc[0]
    for p in ("RB", "WR", "TE", "QB"):                     # a need that's going now
        short = my_counts.get(p, 0) < targets.get(p, 0) - 0.3
        if (short and land.get(p, {}).get("running")
                and 1 <= land[p]["startable_left"] <= 6):
            # name a player we can actually recommend (has an ADP, clears replacement)
            pool = cand[(cand["position"] == p) & (cand["vbd"] > 0)].sort_values("vbd", ascending=False)
            if pool.empty:
                continue
            nm = pool.iloc[0]["name"]
            return f"{p} is running and you still need one — take {nm} now.", nm
    if my_next and top["adp"] > my_next:                   # the best fit will keep — take value first
        going = cand[cand["adp"] <= my_next].sort_values("vbd", ascending=False)
        if not going.empty:
            g = going.iloc[0]
            return (f"Nothing you need is going now — take the best value ({g['name']}, "
                    f"{g['position']}); {top['name']} ({top['position']}) should keep for "
                    f"your next pick."), g["name"]
    return f"Best fit on the board: {top['name']} ({top['position']}).", top["name"]


# --- payload formatting --------------------------------------------------

def _round(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _rec_row(r, cur: int, my_next: int | None) -> dict:
    adp = None if r.adp is None or np.isnan(r.adp) else r.adp
    if adp is None:
        timing = "no ADP"
    elif adp <= cur:
        timing = f"value — past ADP {adp:.0f}"
    elif my_next and adp <= my_next:
        timing = f"going now — likely gone by your #{my_next}"
    else:
        timing = f"could wait — ADP {adp:.0f}" + (f", lasts past #{my_next}" if my_next else "")
    why = []
    if r.wt > 1.05:
        why.append("fits your plan")
    if r.ft > 0.6:
        why.append(f"need at {r.position}")
    return {
        "name": r.name, "position": r.position, "team": r.most_recent_team,
        "proj_ppg": _round(r.proj_ppg), "adp": _round(r.adp),
        "sleeper_adp": _round(getattr(r, "sleeper_adp", None)),
        "tier": None if pd.isna(r.tier) else int(r.tier),
        "timing": timing,
        "why": ", ".join(why) or "best fit available",
    }


def _wait_row(r, my_next: int | None) -> dict:
    return {
        "name": r.name, "position": r.position, "team": r.most_recent_team,
        "proj_ppg": _round(r.proj_ppg), "adp": _round(r.adp),
        "sleeper_adp": _round(getattr(r, "sleeper_adp", None)),
        "note": f"ADP {r.adp:.0f} — should last to your #{my_next}" if my_next else f"ADP {r.adp:.0f}",
    }


def _roster_summary(board: DraftBoard, picks: list[dict], my_slot: int, spec: RosterSpec) -> dict:
    ids = [str(p["player_id"]) for p in picks
           if p.get("draft_slot") == my_slot and p.get("player_id")]
    rows = board.players[board.players["sleeper_id"].astype(str).isin(ids)]
    by_pos = {p: rows[rows["position"] == p]["name"].tolist() for p in SCORABLE}
    by_pos = {p: v for p, v in by_pos.items() if v}
    return {"counts": {p: len(v) for p, v in by_pos.items()}, "players": by_pos,
            "n": len(ids)}
