"""The draft board: model projections + value-over-replacement + live draft state.

Consumes ``models/output/<league>_projections.csv`` (the ``{mean,...}``-shaped
projection, of which v1 fills the mean) as a black box. Adds cross-position value
via ``roster.RosterSpec`` and filters players already drafted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.db import read_sql
from .roster import RosterSpec, SCORABLE

PROJECTIONS_DIR = Path(__file__).resolve().parents[1] / "models" / "output"

# a within-position projected-PPG gap this large starts a new tier
_TIER_GAP = {"QB": 1.5, "RB": 1.2, "WR": 1.0, "TE": 0.9, "K": 0.6, "DEF": 1.0}


def _snake_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    out = []
    for r in range(1, rounds + 1):
        out.append((r - 1) * teams + slot if r % 2 else r * teams - slot + 1)
    return out


@dataclass
class DraftBoard:
    players: pd.DataFrame           # full pool, with vbd / tier / replacement
    spec: RosterSpec
    drafted: set[str]               # sleeper ids

    @property
    def available(self) -> pd.DataFrame:
        return self.players[~self.players["sleeper_id"].isin(self.drafted)].copy()

    def top(self, n: int = 30) -> pd.DataFrame:
        return self.available.sort_values("vbd", ascending=False).head(n)

    def by_position(self, position: str, n: int = 20) -> pd.DataFrame:
        p = self.available[self.available["position"] == position]
        return p.sort_values("proj_ppg", ascending=False).head(n)

    def with_drafted(self, sleeper_ids: set[str]) -> "DraftBoard":
        return DraftBoard(self.players, self.spec, set(sleeper_ids))

    def my_targets(self, slot: int, teams: int, rounds: int, next_pick_no: int) -> dict:
        """What is realistically available at my next pick, assuming picks between
        now and then come off the top of the board."""
        picks = _snake_pick_numbers(slot, teams, rounds)
        my_next = next((p for p in picks if p >= next_pick_no), None)
        avail = self.available.sort_values("vbd", ascending=False).reset_index(drop=True)
        if my_next is None:
            return {"my_next_pick": None, "likely_gone": avail.head(0), "likely_available": avail.head(15)}
        gone_by = max(0, my_next - next_pick_no)
        return {
            "my_next_pick": my_next,
            "picks_until_mine": gone_by,
            "likely_gone": avail.head(gone_by),
            "likely_available": avail.iloc[gone_by:gone_by + 15],
        }


def _assign_tiers(df: pd.DataFrame) -> pd.Series:
    tiers = pd.Series(1, index=df.index, dtype=int)
    for pos, grp in df.groupby("position"):
        gap = _TIER_GAP.get(pos, 1.0)
        ordered = grp.sort_values("proj_ppg", ascending=False)
        t, prev = 1, None
        for idx, ppg in ordered["proj_ppg"].items():
            if prev is not None and prev - ppg > gap:
                t += 1
            tiers.at[idx] = t
            prev = ppg
    return tiers


def build_board(league: str, spec: RosterSpec, projections_dir: Path = PROJECTIONS_DIR) -> DraftBoard:
    proj = pd.read_csv(projections_dir / f"{league}_projections.csv")
    proj = proj[proj["position"].isin(SCORABLE)].copy()

    xref = read_sql("SELECT gsis_id, sleeper_id FROM player_ids").rename(columns={"gsis_id": "player_id"})
    proj = proj.merge(xref, on="player_id", how="left")
    # DEF rows carry the team abbrev as player_id; Sleeper uses that as its DEF id
    proj["sleeper_id"] = proj["sleeper_id"].fillna(proj["player_id"]).astype(str)

    repl_rank = spec.replacement_rank()
    proj["pos_rank"] = proj.groupby("position")["proj_points"].rank(ascending=False, method="first")
    replacement_points = {}
    for pos in SCORABLE:
        pool = proj[proj["position"] == pos].sort_values("proj_points", ascending=False)
        r = min(repl_rank.get(pos, len(pool)), len(pool))
        replacement_points[pos] = float(pool["proj_points"].iloc[r - 1]) if r else 0.0

    proj["replacement_points"] = proj["position"].map(replacement_points)
    proj["vbd"] = proj["proj_points"] - proj["replacement_points"]
    proj["tier"] = _assign_tiers(proj)
    proj["overall_rank"] = proj["vbd"].rank(ascending=False, method="first").astype(int)

    return DraftBoard(proj.sort_values("vbd", ascending=False).reset_index(drop=True), spec, set())
