"""Draft-tool tests: roster parsing, replacement ranks, VBD, snake pick math."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from applications.adp import normalize_name
from applications.board import (
    DraftBoard,
    _assign_tiers,
    _blend_market,
    _snake_pick_numbers,
    _unprojected_from_adp,
)
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


def _proj_with_adp(n=20):
    return pd.DataFrame({
        "name": [f"P{i}" for i in range(n)], "position": ["RB"] * n, "source": ["model"] * n,
        "adp": np.arange(1, n + 1) * 3.0, "vbd": 200.0 - np.arange(1, n + 1) * 8.0,
    })


def test_unprojected_from_adp_selects_unmatched_and_leaves_vbd_nan():
    adp = pd.DataFrame({
        "name": ["Rookie A", "Rookie B", "P3"],
        "norm_name": ["rookie a", "rookie b", "p3"],
        "position": ["RB", "RB", "RB"], "team": ["LV", "NE", "X"],
        "adp": [12.0, 40.0, 9.0],       # P3 is already projected -> skipped
    })
    out = _unprojected_from_adp(_proj_with_adp(), adp, sleeper_players=None, max_adp=180)
    assert set(out["name"]) == {"Rookie A", "Rookie B"}
    assert out["vbd"].isna().all() and (out["source"] == "adp").all()


def test_blend_market_fills_rookies_and_pulls_model_toward_adp():
    board = pd.concat([
        _proj_with_adp(),
        pd.DataFrame({"name": ["RookA", "RookB"], "position": ["RB", "RB"],
                      "source": ["adp", "adp"], "adp": [12.0, 40.0], "vbd": [np.nan, np.nan]}),
    ], ignore_index=True)

    blended = _blend_market(board, weight=0.5)
    r = blended.set_index("name")
    # rookies get the ADP-implied value, monotonic in ADP
    assert r.loc["RookA", "vbd"] > r.loc["RookB", "vbd"]
    assert pd.notna(r.loc["RookA", "vbd"])
    # a model player whose model VBD is far from the ADP curve is pulled toward it
    p0 = r.loc["P0"]
    assert min(p0["model_vbd"], p0["market_vbd"]) <= p0["vbd"] <= max(p0["model_vbd"], p0["market_vbd"])

    pure = _blend_market(board, weight=1.0)
    assert (pure[pure.source == "model"]["vbd"]
            == pure[pure.source == "model"]["model_vbd"]).all()      # model untouched
    assert pd.notna(pure.set_index("name").loc["RookA", "vbd"])      # rookies still filled


def test_my_targets_splits_gone_vs_available():
    b = _board()
    t = b.my_targets(slot=3, teams=10, rounds=15, next_pick_no=1)
    assert t["my_next_pick"] == 3
    assert len(t["likely_gone"]) == 2            # picks 1 and 2 go first
    gone_ids = set(t["likely_gone"]["sleeper_id"])
    assert gone_ids == set(b.top(2)["sleeper_id"])


# --- mock draft simulator ------------------------------------------------

def _sim_board(n_per_pos=30):
    from applications.board import DraftBoard
    rows = []
    for pos, base in (("QB", 300), ("RB", 260), ("WR", 250), ("TE", 200), ("K", 120), ("DEF", 130)):
        for i in range(n_per_pos):
            rows.append(dict(player_id=f"{pos}{i}", sleeper_id=f"s_{pos}{i}", name=f"{pos}{i}",
                             position=pos, most_recent_team="AA", adp=np.nan,
                             proj_ppg=(base - i * 6) / 15,
                             vbd=float(base - i * 6), overall_rank=0, source="model"))
    df = pd.DataFrame(rows)
    df["overall_rank"] = df["vbd"].rank(ascending=False, method="first").astype(int)
    return DraftBoard(df.sort_values("vbd", ascending=False).reset_index(drop=True),
                      RosterSpec(teams=10, dedicated={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
                                 flex=[frozenset({"RB", "WR", "TE"})]), set())


def test_simulate_draft_fills_a_legal_roster_and_no_duplicates():
    from applications.mock import simulate_draft
    b = _sim_board()
    picks, mine = simulate_draft(b, b.spec, my_slot=4, rounds=15,
                                 rng=np.random.default_rng(0))
    assert len(picks) == 10 * 15
    assert picks["name"].is_unique                       # nobody drafted twice
    assert len(mine) == 15
    counts = Counter(mine["position"])
    assert counts["QB"] >= 1 and counts["K"] >= 1 and counts["DEF"] >= 1   # required slots filled
    assert counts["QB"] <= 3 and counts["TE"] <= 3        # not stacking absurdly


def test_simulate_draft_is_deterministic_under_seed():
    from applications.mock import simulate_draft
    b = _sim_board()
    a1 = simulate_draft(b, b.spec, 4, 15, np.random.default_rng(7))[0]
    a2 = simulate_draft(b, b.spec, 4, 15, np.random.default_rng(7))[0]
    assert a1.equals(a2)
