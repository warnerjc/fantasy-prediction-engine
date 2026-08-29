"""Simulated / Monte-Carlo mock drafts.

Runs a full snake draft locally: the other N-1 seats pick roughly by ADP (with
tunable randomness and positional-need awareness), our seat picks off the live
``DraftBoard`` exactly as ``draft_tool`` would. Two uses:

1. **Dry run** — exercises the same board code path as ``--watch`` (drafted-player
   filtering, VBD ranking, roster needs) without needing a real draft room.
2. **Strategy check** — ``--sims N`` runs it many times to see the roster you tend
   to end up with from a given seat, and where value falls.

    python -m applications.mock --league sleeper --slot 5
    python -m applications.mock --league sleeper --slot 5 --sims 200
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .board import DraftBoard
from .draft_tool import _build, _load_config
from .roster import RosterSpec, SCORABLE, roster_spec

import math

# bench depth a drafter will carry per position, on top of starters + flex share
_BENCH = {"QB": 1, "RB": 4, "WR": 4, "TE": 1, "K": 0, "DEF": 0}


def _position_caps(spec: RosterSpec) -> dict[str, int]:
    """Max a team will roster at each position (starters + flex + bench)."""
    alloc = spec.flex_allocation()
    return {p: spec.dedicated.get(p, 0) + math.ceil(alloc.get(p, 0.0)) + _BENCH.get(p, 0)
            for p in SCORABLE}


def _required_slots(spec: RosterSpec) -> dict[str, int]:
    """Positions a legal roster must fill: dedicated starters + 1 per flex-eligible
    slot's most-likely position, counting K/DEF."""
    req = dict(spec.dedicated)
    for elig in spec.flex:
        req["RB" if "RB" in elig else next(iter(elig))] = req.get(
            "RB" if "RB" in elig else next(iter(elig)), 0) + 1
    return req


def _snake_order(teams: int, rounds: int) -> list[int]:
    order = []
    for r in range(rounds):
        seats = range(1, teams + 1) if r % 2 == 0 else range(teams, 0, -1)
        order.extend(seats)
    return order


@dataclass
class _Roster:
    counts: Counter = field(default_factory=Counter)

    def add(self, pos: str) -> None:
        self.counts[pos] += 1

    def fit(self, pos: str, startable: float) -> float:
        """Roster-fit multiplier for taking another `pos`: >0 while below startable
        need, ~0 for the first bench body, negative once stacking past that."""
        have = self.counts[pos]
        if have < startable:
            return 1.0 - have / max(startable, 1)      # 1.0 empty -> ~0 at startable
        return -0.5 * (have - startable)               # penalty for piling on


def simulate_draft(
    board: DraftBoard,
    spec: RosterSpec,
    my_slot: int,
    rounds: int,
    rng: np.random.Generator,
    opp_temp: float = 6.0,
    my_need_weight: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One full draft. Returns (all picks, my roster)."""
    caps = _position_caps(spec)
    required = _required_slots(spec)
    alloc = spec.flex_allocation()
    startable = {p: spec.dedicated.get(p, 0) + alloc.get(p, 0.0) for p in SCORABLE}
    pool = board.players.dropna(subset=["vbd"]).sort_values("vbd", ascending=False).reset_index(drop=True)
    pool = pool.copy()
    pool["adp_fill"] = pool["adp"].fillna(pool["overall_rank"] + 40)
    taken: set[int] = set()
    rosters = {s: _Roster() for s in range(1, spec.teams + 1)}
    my_picks_left = {s: sum(1 for x in _snake_order(spec.teams, rounds) if x == s) for s in rosters}
    picks = []

    for pick_no, slot in enumerate(_snake_order(spec.teams, rounds), start=1):
        avail = pool[~pool.index.isin(taken)]
        if avail.empty:
            break
        rd = rosters[slot]
        my_picks_left[slot] -= 1

        under_cap = avail[avail["position"].map(lambda p: rd.counts[p] < caps.get(p, 99))]
        unmet = {p: n - rd.counts[p] for p, n in required.items() if rd.counts[p] < n}

        if my_picks_left[slot] < sum(unmet.values()):        # must start filling required slots
            forced = avail[avail["position"].isin(unmet)]
            pick_from = forced if not forced.empty else (under_cap if not under_cap.empty else avail)
        else:
            pick_from = under_cap if not under_cap.empty else avail

        if slot == my_slot:
            fit = pick_from["position"].map(lambda p: rd.fit(p, startable[p]))
            score = pick_from["vbd"] + my_need_weight * fit * pick_from["vbd"].clip(lower=0).mean()
            choice = score.idxmax()
        else:
            cand = pick_from.nsmallest(14, "adp_fill")
            logits = -cand["adp_fill"].to_numpy() / opp_temp
            w = np.exp(logits - logits.max())
            w = w / w.sum()
            choice = rng.choice(cand.index.to_numpy(), p=w)

        taken.add(choice)
        row = pool.loc[choice]
        rosters[slot].add(row["position"])
        picks.append({
            "pick_no": pick_no, "round": (pick_no - 1) // spec.teams + 1, "slot": slot,
            "name": row["name"], "position": row["position"], "team": row["most_recent_team"],
            "adp": row["adp"], "vbd": row["vbd"], "is_mine": slot == my_slot,
        })

    pdf = pd.DataFrame(picks)
    return pdf, pdf[pdf["is_mine"]].reset_index(drop=True)


def _print_single(picks: pd.DataFrame, mine: pd.DataFrame, spec: RosterSpec) -> None:
    print("\nMY PICKS")
    for r in mine.itertuples():
        adp = f"adp {r.adp:.0f}" if pd.notna(r.adp) else "adp  -"
        print(f"  R{r.round:>2} (#{r.pick_no:>3})  {r.position:<3} {r.name:<24} {adp:>8}  vbd {r.vbd:6.1f}")
    comp = ", ".join(f"{n}{p}" for p, n in sorted(Counter(mine['position']).items()))
    print(f"\n  roster: {comp}   total VBD captured: {mine['vbd'].sum():.0f}")


def _print_mc(results: list[pd.DataFrame], spec: RosterSpec, slot: int) -> None:
    tot = np.array([m["vbd"].sum() for m in results])
    print(f"\n{len(results)} sims from slot {slot}")
    print(f"  VBD captured: mean {tot.mean():.0f}   sd {tot.std():.0f}   "
          f"range {tot.min():.0f}–{tot.max():.0f}")

    by_round: dict[int, Counter] = {}
    for m in results:
        for r in m.itertuples():
            by_round.setdefault(r.round, Counter())[f"{r.name} ({r.position})"] += 1
    print("\n  most common pick by round:")
    for rd in sorted(by_round)[:8]:
        top = by_round[rd].most_common(3)
        print(f"   R{rd:>2}  " + "   ".join(f"{n} {c*100//len(results)}%" for n, c in top))

    pos_first = {p: [] for p in ("QB", "RB", "WR", "TE")}
    for m in results:
        for p in pos_first:
            got = m[m["position"] == p]
            pos_first[p].append(got["round"].min() if not got.empty else np.nan)
    print("\n  round you typically land your first:")
    for p, rounds in pos_first.items():
        arr = np.array(rounds, float)
        print(f"   {p}: {np.nanmedian(arr):.0f}  (mean {np.nanmean(arr):.1f})")


def _round_count(cfg: dict) -> int:
    rp = cfg.get("roster_positions")
    return len(rp) if isinstance(rp, list) else sum(int(v) for v in rp.values())


def run(league: str, slot: int, sims: int, blend: float, opp_temp: float, seed: int) -> None:
    cfg = _load_config(league)
    spec = roster_spec(cfg)
    board = _build(league, spec, use_adp=True, season=None, blend=blend)
    n_rounds = _round_count(cfg)

    print(f"{league}: {spec.teams} teams x {n_rounds} rounds, your slot {slot}")
    rng = np.random.default_rng(seed)

    if sims <= 1:
        picks, mine = simulate_draft(board, spec, slot, n_rounds, rng, opp_temp)
        _print_single(picks, mine, spec)
        return

    results = []
    for _ in range(sims):
        _, mine = simulate_draft(board, spec, slot, n_rounds,
                                 np.random.default_rng(rng.integers(1 << 32)), opp_temp)
        results.append(mine)
    _print_mc(results, spec, slot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", required=True, choices=["sleeper", "yahoo"])
    ap.add_argument("--slot", type=int, required=True, help="your draft position")
    ap.add_argument("--sims", type=int, default=1, help="1 = show a full draft; N = Monte Carlo summary")
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--opp-temp", type=float, default=6.0, help="opponent randomness (higher = more chaos)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.league, args.slot, args.sims, args.blend, args.opp_temp, args.seed)


if __name__ == "__main__":
    main()
