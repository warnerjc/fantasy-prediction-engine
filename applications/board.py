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
from sklearn.isotonic import IsotonicRegression

from data.db import read_sql
from .adp import normalize_name
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
    tiers = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for pos, grp in df.groupby("position"):
        gap = _TIER_GAP.get(pos, 1.0)
        ordered = grp[grp["proj_ppg"].notna()].sort_values("proj_ppg", ascending=False)
        t, prev = 1, None
        for idx, ppg in ordered["proj_ppg"].items():
            if prev is not None and prev - ppg > gap:
                t += 1
            tiers.at[idx] = t
            prev = ppg
    return tiers


def _attach_adp(proj: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    proj = proj.copy()
    proj["norm_name"] = proj["name"].map(normalize_name)
    if adp is None or adp.empty:
        proj["adp"] = np.nan
        return proj.drop(columns=["norm_name"])
    key = adp[["norm_name", "position", "adp"]].drop_duplicates(["norm_name", "position"])
    proj = proj.merge(key, on=["norm_name", "position"], how="left")
    return proj.drop(columns=["norm_name"])


def _unprojected_from_adp(proj, adp, sleeper_players, max_adp) -> pd.DataFrame:
    """Rows for ADP players with no model projection (rookies / missing season),
    VBD imputed from ADP by isotonic regression on players who have both."""
    if adp is None or adp.empty:
        return pd.DataFrame()
    have = set(zip(proj["name"].map(normalize_name), proj["position"]))
    matched = pd.Series(
        [(n, p) in have for n, p in zip(adp["norm_name"], adp["position"])],
        index=adp.index,
    )
    # only offense — K/DEF name matching to ADP is unreliable and they're low-stakes
    miss = adp[~matched & adp["position"].isin(("QB", "RB", "WR", "TE"))
               & (adp["adp"] <= max_adp)].copy()
    fit = proj.dropna(subset=["adp", "vbd"])
    if miss.empty or len(fit) < 10:
        return pd.DataFrame()

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(fit["adp"], fit["vbd"])
    out = pd.DataFrame({
        "name": miss["name"].values,
        "position": miss["position"].values,
        "most_recent_team": miss["team"].values,
        "player_id": "adp:" + miss["name"].str.replace(r"\s+", "_", regex=True),
        "proj_ppg": np.nan, "proj_points": np.nan,
        "adp": miss["adp"].values,
        "vbd": iso.predict(miss["adp"].values),
        "source": "adp",
    })
    if sleeper_players is not None and not sleeper_players.empty:
        sp = sleeper_players.drop_duplicates(["norm_name", "position"])
        out["norm_name"] = out["name"].map(normalize_name)
        out = out.merge(sp[["norm_name", "position", "sleeper_id", "is_rookie"]],
                        on=["norm_name", "position"], how="left").drop(columns=["norm_name"])
    return out


def build_board(
    league: str,
    spec: RosterSpec,
    projections_dir: Path = PROJECTIONS_DIR,
    adp: pd.DataFrame | None = None,
    sleeper_players: pd.DataFrame | None = None,
    max_adp: int = 180,
) -> DraftBoard:
    proj = pd.read_csv(projections_dir / f"{league}_projections.csv")
    proj = proj[proj["position"].isin(SCORABLE)].copy()

    xref = read_sql("SELECT gsis_id, sleeper_id FROM player_ids").rename(columns={"gsis_id": "player_id"})
    proj = proj.merge(xref, on="player_id", how="left")
    # DEF rows carry the team abbrev as player_id; Sleeper uses that as its DEF id
    proj["sleeper_id"] = proj["sleeper_id"].fillna(proj["player_id"]).astype(str)
    proj["source"] = "model"

    repl_rank = spec.replacement_rank()
    replacement_points = {}
    for pos in SCORABLE:
        pool = proj[proj["position"] == pos].sort_values("proj_points", ascending=False)
        r = min(repl_rank.get(pos, len(pool)), len(pool))
        replacement_points[pos] = float(pool["proj_points"].iloc[r - 1]) if r else 0.0
    proj["replacement_points"] = proj["position"].map(replacement_points)
    proj["vbd"] = proj["proj_points"] - proj["replacement_points"]

    proj = _attach_adp(proj, adp)
    extra = _unprojected_from_adp(proj, adp, sleeper_players, max_adp)
    board = pd.concat([proj, extra], ignore_index=True) if not extra.empty else proj

    board["is_rookie"] = board.get("is_rookie", 0)
    board["is_rookie"] = board["is_rookie"].fillna(0).astype(int)
    board["sleeper_id"] = board["sleeper_id"].astype("string")
    board["tier"] = _assign_tiers(board)
    board["overall_rank"] = board["vbd"].rank(ascending=False, method="first").astype(int)

    return DraftBoard(board.sort_values("vbd", ascending=False).reset_index(drop=True), spec, set())
