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


@dataclass
class DraftStart:
    """Resume a simulation from a real draft in progress."""
    rosters: dict[int, Counter]   # slot -> positions already drafted by that slot
    pick_no: int                  # the next overall pick number (1-based)


def roster_value(roster: pd.DataFrame, spec: RosterSpec) -> float:
    """Value of a roster's *optimal starting lineup*: fill each dedicated slot with
    the position's best player, each flex with the best eligible player left, and
    sum their ``vbd`` (points over positional replacement — already discounts QB,
    where streaming is cheap). Bench doesn't count."""
    if roster.empty or "vbd" not in roster.columns:
        return 0.0
    pool = roster.dropna(subset=["vbd"]).sort_values("vbd", ascending=False)
    used: set = set()
    total = 0.0
    for pos, n in spec.dedicated.items():
        picks = pool[(pool["position"] == pos) & ~pool.index.isin(used)].head(n)
        total += float(picks["vbd"].sum())
        used |= set(picks.index)
    for elig in spec.flex:
        cand = pool[pool["position"].isin(elig) & ~pool.index.isin(used)].head(1)
        total += float(cand["vbd"].sum())
        used |= set(cand.index)
    return total


def simulate_draft(
    board: DraftBoard,
    spec: RosterSpec,
    my_slot: int,
    rounds: int,
    rng: np.random.Generator,
    opp_temp: float = 6.0,
    my_need_weight: float = 0.35,
    *,
    my_strategy=None,
    start: DraftStart | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One full (or resumed) draft. Returns (picks made this call, my picks this call).

    ``start`` resumes an in-progress draft: seed each slot's roster counts and the
    next pick number. ``my_strategy`` (a ``draftplan.Strategy``) re-weights our
    seat's pick by ``strategy.weight(position, round)``.
    """
    caps = _position_caps(spec)
    required = _required_slots(spec)
    alloc = spec.flex_allocation()
    startable = {p: spec.dedicated.get(p, 0) + alloc.get(p, 0.0) for p in SCORABLE}

    # cap the pool: nobody past ~(picks + a buffer) gets drafted, and the hot loop
    # is O(pool) per pick.
    keep = spec.teams * rounds + 60
    pool = (board.available.dropna(subset=["vbd"])
            .sort_values("vbd", ascending=False).head(keep).reset_index(drop=True))
    n = len(pool)
    pos = pool["position"].to_numpy()
    vbd = pool["vbd"].to_numpy(dtype=float)
    adp_fill = pool["adp"].fillna(pool["overall_rank"] + 40).to_numpy(dtype=float)
    vbd_pos_mean = float(np.clip(vbd, 0, None).mean())
    alive = np.ones(n, dtype=bool)

    rosters = {s: _Roster(Counter(start.rosters.get(s, {})) if start else Counter())
               for s in range(1, spec.teams + 1)}
    first = start.pick_no if start else 1
    order = _snake_order(spec.teams, rounds)
    my_picks_left = {s: sum(1 for i, x in enumerate(order, 1) if x == s and i >= first)
                     for s in rosters}
    picks = []

    for pick_no, slot in enumerate(order, start=1):
        if pick_no < first:
            continue
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        rd = rosters[slot]
        rnd = (pick_no - 1) // spec.teams + 1
        my_picks_left[slot] -= 1
        apos = pos[idx]

        under_cap = idx[np.array([rd.counts[p] < caps.get(p, 99) for p in apos])]
        unmet = {p: req - rd.counts[p] for p, req in required.items() if rd.counts[p] < req}
        if my_picks_left[slot] < sum(unmet.values()) and unmet:      # must fill required slots
            forced = idx[np.isin(apos, list(unmet))]
            cand_idx = forced if forced.size else (under_cap if under_cap.size else idx)
        else:
            cand_idx = under_cap if under_cap.size else idx

        cpos = pos[cand_idx]
        if slot == my_slot:
            fit = np.array([rd.fit(p, startable[p]) for p in cpos])
            score = vbd[cand_idx] + my_need_weight * fit * vbd_pos_mean
            if my_strategy is not None:
                score = score * np.array([my_strategy.weight(p, rnd) for p in cpos])
            choice = int(cand_idx[np.argmax(score)])
        else:
            k = min(14, cand_idx.size)
            near = cand_idx[np.argsort(adp_fill[cand_idx])[:k]]
            logits = -adp_fill[near] / opp_temp
            w = np.exp(logits - logits.max())
            choice = int(rng.choice(near, p=w / w.sum()))

        alive[choice] = False
        rosters[slot].add(pos[choice])
        picks.append({
            "pick_no": pick_no, "round": rnd, "slot": slot,
            "name": pool.at[choice, "name"], "position": pos[choice],
            "team": pool.at[choice, "most_recent_team"],
            "adp": pool.at[choice, "adp"], "vbd": vbd[choice],
            "proj_points": pool.at[choice, "proj_points"] if "proj_points" in pool.columns else np.nan,
            "is_mine": slot == my_slot,
        })

    pdf = pd.DataFrame(picks)
    mine = pdf[pdf["is_mine"]].reset_index(drop=True) if not pdf.empty else pdf
    return pdf, mine


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
