"""Draft-tool tests: roster parsing, replacement ranks, VBD, snake pick math."""

import numpy as np
import pandas as pd
import pytest

from applications.adp import normalize_name
from applications.board import DraftBoard, _assign_tiers, _snake_pick_numbers, _unprojected_from_adp
from applications.roster import RosterSpec, roster_spec


# --- roster parsing --------------------------------------------------------

def test_roster_spec_from_sleeper_tokens():
    cfg = {"total_rosters": 12, "roster_positions":
           ["QB", "RB", "WR", "TE", "WRRB_FLEX", "REC_FLEX", "REC_FLEX", "K", "DEF"] + ["BN"] * 7}
    spec = roster_spec(cfg)
    assert spec.teams == 12
    assert spec.dedicated == {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "K": 1, "DEF": 1}
    assert len(spec.flex) == 3


def test_roster_spec_from_yahoo_dict():
    cfg = {"teams": 10, "roster_positions": {"QB": 1, "WR": 3, "RB": 2, "TE": 1,
                                             "FLEX_WRT": 1, "K": 1, "DEF": 1, "BN": 6}}
    spec = roster_spec(cfg)
    assert spec.teams == 10
    assert spec.dedicated["WR"] == 3 and spec.dedicated["RB"] == 2
    assert spec.flex == [frozenset({"RB", "WR", "TE"})]


def test_replacement_rank_pushes_qb_te_deeper_than_last_starter():
    spec = RosterSpec(teams=12, dedicated={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "K": 1, "DEF": 1},
                      flex=[frozenset({"RB", "WR"})])
    r = spec.replacement_rank()
    assert r["QB"] > 12            # deeper than one-per-team (waiver QB is fine)
    assert r["TE"] > 12
    assert r["K"] == 12 and r["DEF"] == 12
    assert r["RB"] >= 12           # 1 dedicated + flex share


# --- snake pick math -----------------------------------------------------

def test_snake_pick_numbers_reverse_each_round():
    picks = _snake_pick_numbers(slot=3, teams=12, rounds=4)
    assert picks == [3, 22, 27, 46]      # 3, (24-3+1), (24+3), (48-3+1)

    first = _snake_pick_numbers(slot=1, teams=10, rounds=3)
    assert first == [1, 20, 21]


# --- board -------------------------------------------------------------------

def _board():
    rows = []
    for i in range(40):
        rows.append(dict(player_id=f"rb{i}", sleeper_id=f"s_rb{i}", name=f"RB{i}",
                         position="RB", most_recent_team="AA",
                         proj_ppg=20 - i * 0.4, proj_points=(20 - i * 0.4) * 15))
    for i in range(40):
        rows.append(dict(player_id=f"wr{i}", sleeper_id=f"s_wr{i}", name=f"WR{i}",
                         position="WR", most_recent_team="BB",
                         proj_ppg=18 - i * 0.3, proj_points=(18 - i * 0.3) * 15))
    proj = pd.DataFrame(rows)
    spec = RosterSpec(teams=10, dedicated={"RB": 2, "WR": 2}, flex=[])
    repl = spec.replacement_rank()
    proj["replacement_points"] = proj["position"].map(
        {p: proj[proj.position == p].sort_values("proj_points", ascending=False)
              ["proj_points"].iloc[repl[p] - 1] for p in ("RB", "WR")})
    proj["vbd"] = proj["proj_points"] - proj["replacement_points"]
    proj["tier"] = _assign_tiers(proj)
    proj["overall_rank"] = proj["vbd"].rank(ascending=False, method="first").astype(int)
    return DraftBoard(proj.sort_values("vbd", ascending=False).reset_index(drop=True), spec, set())


def test_vbd_is_zero_at_replacement_and_positive_above():
    b = _board()
    rb = b.players[b.players.position == "RB"].sort_values("proj_points", ascending=False)
    # RB replacement rank = teams * 2 = 20 -> the 20th RB has ~0 vbd
    assert rb["vbd"].iloc[19] == pytest.approx(0.0, abs=1e-6)
    assert (rb["vbd"].iloc[:19] > 0).all()
    assert (rb["vbd"].iloc[20:] < 0).all()


def test_drafted_players_drop_off_the_board():
    b = _board()
    top5 = set(b.top(5)["sleeper_id"])
    b2 = b.with_drafted(top5)
    assert not top5 & set(b2.available["sleeper_id"])
    assert len(b2.available) == len(b.players) - 5


def test_normalize_name_strips_suffixes_and_punctuation():
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("D.J. Moore") == "dj moore"
    assert normalize_name("Michael Pittman II") == "michael pittman"


def test_unprojected_from_adp_imputes_vbd_monotonically():
    # 20 projected players with a clean decreasing vbd-vs-adp relationship
    proj = pd.DataFrame({
        "name": [f"P{i}" for i in range(20)],
        "position": ["RB"] * 20,
        "adp": np.arange(1, 21) * 3.0,
        "vbd": 200 - np.arange(1, 21) * 8.0,
    })
    adp = pd.DataFrame({
        "name": ["Rookie A", "Rookie B", "P3"],
        "norm_name": ["rookie a", "rookie b", "p3"],
        "position": ["RB", "RB", "RB"],
        "team": ["LV", "NE", "X"],
        "adp": [12.0, 40.0, 9.0],       # P3 is already projected -> skipped
    })
    out = _unprojected_from_adp(proj, adp, sleeper_players=None, max_adp=180)
    assert set(out["name"]) == {"Rookie A", "Rookie B"}          # P3 excluded
    a = out.set_index("name")["vbd"]
    assert a["Rookie A"] > a["Rookie B"]                          # earlier ADP -> more value
    assert out["proj_ppg"].isna().all() and (out["source"] == "adp").all()


def test_my_targets_splits_gone_vs_available():
    b = _board()
    t = b.my_targets(slot=3, teams=10, rounds=15, next_pick_no=1)
    assert t["my_next_pick"] == 3
    assert len(t["likely_gone"]) == 2            # picks 1 and 2 go first
    gone_ids = set(t["likely_gone"]["sleeper_id"])
    assert gone_ids == set(b.top(2)["sleeper_id"])
