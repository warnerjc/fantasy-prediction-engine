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


def test_build_board_sleeper_id_matches_sleeper_pick_ids(tmp_path, monkeypatch):
    # regression: player_ids stored sleeper_id float-formatted ("9493.0"), which
    # never matched Sleeper's pick ids ("9493") -> drafted players never left the
    # board during a live draft.
    import applications.board as bd

    proj = pd.DataFrame({
        "position": ["WR", "RB", "DEF"], "name": ["Puka Nacua", "Bijan Robinson", "SEA"],
        "most_recent_team": ["LA", "ATL", "SEA"], "player_id": ["00-0039075", "00-0009509", "SEA"],
        "proj_ppg": [17.9, 22.4, 10.7], "proj_points": [277.0, 337.0, 182.0],
        "target_season": [2026, 2026, 2026],
    })
    proj.to_csv(tmp_path / "sleeper_projections.csv", index=False)
    monkeypatch.setattr(bd, "read_sql", lambda q: pd.DataFrame(
        {"gsis_id": ["00-0039075", "00-0009509"], "sleeper_id": pd.array(["9493.0", "9509.0"], dtype="string")}))

    spec = RosterSpec(teams=12, dedicated={"WR": 2, "RB": 2, "DEF": 1}, flex=[])
    board = bd.build_board("sleeper", spec, projections_dir=tmp_path, adp=None)

    assert set(board.players["sleeper_id"]) == {"9493", "9509", "SEA"}   # no ".0"
    drafted = board.with_drafted({"9493"})                              # Sleeper pick id form
    assert "Puka Nacua" not in set(drafted.available["name"])


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


def test_blend_market_floors_model_at_the_market_value():
    # a model row the model tanks (committee RB) must not fall below what its ADP
    # implies -- otherwise it drops off the draftable board entirely.
    n = 25
    anchor = pd.DataFrame({
        "name": [f"P{i}" for i in range(n)], "position": ["RB"] * n, "source": ["model"] * n,
        "adp": np.linspace(1, 180, n), "vbd": np.linspace(150, -30, n),
    })
    tanked = pd.DataFrame({"name": ["Committee"], "position": ["RB"], "source": ["model"],
                           "adp": [120.0], "vbd": [-140.0]})   # model hates him, market drafts him
    out = _blend_market(pd.concat([anchor, tanked], ignore_index=True), weight=0.7).set_index("name")
    r = out.loc["Committee"]
    assert r["vbd"] >= r["market_vbd"] - 1e-6           # floored at market
    assert r["vbd"] > r["model_vbd"]                    # lifted off the model's number


def test_blend_market_per_position_weight_and_no_adp_pin():
    # anchor rows spanning the whole ADP range so the isotonic curve has a real,
    # low floor; then two QBs the model loves equally — one with a late ADP, one
    # the market isn't drafting at all.
    n = 30
    rb = pd.DataFrame({
        "name": [f"P{i}" for i in range(n)], "position": ["RB"] * n, "source": ["model"] * n,
        "adp": np.linspace(1, 180, n), "vbd": np.linspace(180, 2, n),
    })
    q = pd.DataFrame({
        "name": ["QBadp", "QBnoadp"], "position": ["QB", "QB"], "source": ["model", "model"],
        "adp": [140.0, np.nan], "vbd": [160.0, 160.0],
    })
    out = _blend_market(pd.concat([rb, q], ignore_index=True),
                        weight=0.7, per_pos={"QB": 0.3}).set_index("name")

    qa = out.loc["QBadp"]
    assert qa["vbd"] == pytest.approx(0.3 * qa["model_vbd"] + 0.7 * qa["market_vbd"], rel=1e-6)
    # QBadp blended toward the low market value sits well under its model VBD
    assert qa["vbd"] < 0.5 * qa["model_vbd"]
    # no-ADP QB: pinned to the bottom of the curve, then blended -> far below pure model
    floor = out.loc[out["market_vbd"].notna(), "market_vbd"].min()
    assert out.loc["QBnoadp", "market_vbd"] == pytest.approx(floor)
    assert out.loc["QBnoadp", "vbd"] < 0.5 * 160
    # an RB with no override still uses the passed 0.7 weight
    r0 = out.loc["P0"]
    assert r0["vbd"] == pytest.approx(0.7 * r0["model_vbd"] + 0.3 * r0["market_vbd"], rel=1e-6)


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
            vbd = float(base - i * 6)
            rows.append(dict(player_id=f"{pos}{i}", sleeper_id=f"s_{pos}{i}", name=f"{pos}{i}",
                             position=pos, most_recent_team="AA",
                             proj_ppg=(base - i * 6) / 15, proj_points=vbd + 120,
                             vbd=vbd, tier=1 + i // 6, overall_rank=0, source="model"))
    df = pd.DataFrame(rows)
    df["overall_rank"] = df["vbd"].rank(ascending=False, method="first").astype(int)
    df["adp"] = df["overall_rank"].astype(float)
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


def test_simulate_draft_resumes_from_a_partial_state():
    from applications.mock import DraftStart, simulate_draft
    b = _sim_board()
    start = DraftStart(rosters={s: Counter({"RB": 1}) for s in range(1, 11)}, pick_no=31)
    picks, mine = simulate_draft(b, b.spec, 4, 15, np.random.default_rng(0), start=start)
    assert picks["pick_no"].min() == 31 and picks["pick_no"].max() == 150   # resumes, finishes
    assert picks["name"].is_unique
    # RB2 slot: 10 teams already hold 1 RB -> most add at most 2 more before caps
    assert Counter(mine["position"])["RB"] <= 3


def test_roster_value_scores_the_optimal_lineup():
    from applications.mock import roster_value
    b = _sim_board()
    spec = RosterSpec(teams=10, dedicated={"QB": 1, "RB": 1}, flex=[frozenset({"RB", "WR"})])
    roster = b.players.set_index("name").loc[["QB0", "RB0", "RB5", "WR0"]].reset_index()
    # starts QB0 + best two of {RB0, RB5, WR0} for RB + flex
    got = roster_value(roster, spec)
    top3 = roster.nlargest(3, "vbd")["vbd"].sum()
    assert got == pytest.approx(roster.loc[roster.name == "QB0", "vbd"].iloc[0]
                                + roster.nlargest(2, "vbd").query("position != 'QB'")["vbd"].sum(),
                                rel=0.01) or got == pytest.approx(top3, rel=0.01)


# --- draft strategy + recommendation ------------------------------------

def test_choose_initial_strategy_reads_roster_shape(monkeypatch):
    from applications.draftplan import choose_initial_strategy

    class R:  # minimal ScoringRules stand-in
        per_unit = {"rec": 0.5}
        position_bonuses: dict = {}

    wr_heavy = RosterSpec(teams=12, dedicated={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
                          flex=[frozenset({"RB", "WR"}), frozenset({"WR", "TE"}), frozenset({"WR", "TE"})])
    s, why = choose_initial_strategy(wr_heavy, R())
    assert s.name == "wr_early"          # WR-heavy roster -> WR-lean starting guess

    rb_heavy = RosterSpec(teams=12, dedicated={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
                          flex=[frozenset({"RB", "WR"})])
    R.per_unit = {"rec": 0.0}
    assert choose_initial_strategy(rb_heavy, R())[0].name == "rb_early"

    superflex = RosterSpec(teams=12, dedicated={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
                           flex=[frozenset({"QB", "RB", "WR", "TE"})])
    assert choose_initial_strategy(superflex, R())[0].name == "qb_early"


def _draft_state(b, picks_made):
    """A Sleeper-style draft_state: fill `picks_made` picks off the top of the board."""
    order = []
    teams = b.spec.teams
    for r in range(picks_made // teams + 2):
        seats = range(1, teams + 1) if r % 2 == 0 else range(teams, 0, -1)
        order += list(seats)
    top = b.players.sort_values("vbd", ascending=False).head(picks_made)
    picks = [{"player_id": sid, "draft_slot": order[i], "pick_no": i + 1,
              "metadata": {"position": pos}}
             for i, (sid, pos) in enumerate(zip(top["sleeper_id"], top["position"]))]
    return {"picks": picks, "status": "drafting", "teams": teams, "rounds": 15}


def test_recommend_never_reaches_and_lists_players_to_wait_on():
    from applications.draftplan import STRATEGIES, recommend
    b = _sim_board()
    st = _draft_state(b, picks_made=24)          # your slot-4 pick #25 in a 10-team
    rec = recommend(b, st, my_slot=4, spec=b.spec, strategy=STRATEGIES["bpa"], rounds=15)

    assert rec["current_pick"] == 25 and rec["my_next_pick"] == 37
    names = {r["name"] for r in rec["recommendations"]}
    waits = {w["name"] for w in rec["wait"]}
    # a "wait" player's ADP is past your next pick; a "draft now" player's isn't far past it
    for w in rec["wait"]:
        assert w["adp"] > rec["my_next_pick"]
    assert names and not (names & waits)
    assert rec["landscape"]["RB"]["startable_left"] >= 0


# --- 2026 roster overrides ------------------------------------------------

def test_roster_2026_overrides_relabel_team_and_drop_out(tmp_path, monkeypatch):
    import json
    from applications import roster_2026 as r26

    cfg = tmp_path / "ov.json"
    cfg.write_text(json.dumps({
        "team": {"A.J. Brown": "NE", "D.J. Moore": "BUF"},
        "out": {"Joe Mixon": "unsigned"},
        "adp": {"Josh Jacobs": 146},
    }))
    monkeypatch.setattr(r26, "_OVERRIDES", cfg)
    team, out, adp = r26.load_overrides(cfg)
    assert team == {"aj brown": "NE", "dj moore": "BUF"}
    assert out == {"joe mixon"}
    assert adp == {"josh jacobs": 146.0}

    proj = pd.DataFrame({
        "name": ["A.J. Brown", "DJ Moore", "Justin Jefferson", "Joe Mixon"],
        "most_recent_team": ["PHI", "CHI", "MIN", "HOU"],
    })
    labeled = r26.apply_team_labels(proj)
    assert list(labeled["most_recent_team"]) == ["NE", "BUF", "MIN", "HOU"]
    assert list(labeled["team_changed"]) == [True, True, False, False]
    assert list(labeled["team_source"]) == ["override", "override", "model", "model"]

    board = proj.assign(vbd=[10.0, 9, 8, 7])
    kept, dropped = r26.drop_unavailable(board)
    assert dropped == ["Joe Mixon"]
    assert "Joe Mixon" not in set(kept["name"])

    b2 = pd.DataFrame({"name": ["Josh Jacobs", "Bijan Robinson"], "adp": [28.0, 2.0]})
    forced, changed = r26.apply_adp_overrides(b2)
    assert changed == ["Josh Jacobs"]
    assert forced.set_index("name").loc["Josh Jacobs", "adp"] == 146.0
    assert forced.set_index("name").loc["Bijan Robinson", "adp"] == 2.0


def test_roster_2026_missing_override_file_is_a_noop(tmp_path, monkeypatch):
    from applications import roster_2026 as r26
    monkeypatch.setattr(r26, "_OVERRIDES", tmp_path / "does-not-exist.json")
    assert r26.load_overrides(r26._OVERRIDES) == ({}, set(), {})
    proj = pd.DataFrame({"name": ["X"], "most_recent_team": ["AA"]})
    out = r26.apply_team_labels(proj)
    assert list(out["most_recent_team"]) == ["AA"] and not out["team_changed"].any()


# --- scarcity floor (don't chase a need with a below-replacement body) ----

def _board_with_exhausted_rb():
    """WR has real value left; RB is all below replacement; a K drags the value
    floor down (so the negative RB still gets a sizeable `val`)."""
    rows = []
    for i in range(8):
        rows.append(dict(player_id=f"wr{i}", sleeper_id=f"s_wr{i}", name=f"WR{i}",
                         position="WR", most_recent_team="AA", proj_ppg=15 - i,
                         proj_points=200 - i * 8, vbd=30.0 - i * 4, tier=1 + i // 3,
                         source="model", adp=float(20 + i * 4)))
    for i in range(5):
        rows.append(dict(player_id=f"rb{i}", sleeper_id=f"s_rb{i}", name=f"RB{i}",
                         position="RB", most_recent_team="BB", proj_ppg=6 - i * 0.3,
                         proj_points=90 - i * 5, vbd=-10.0 - i * 2, tier=4,
                         source="model", adp=float(30 + i * 3)))
    rows.append(dict(player_id="k0", sleeper_id="s_k0", name="K0", position="K",
                     most_recent_team="CC", proj_ppg=8, proj_points=8, vbd=-45.0,
                     tier=6, source="model", adp=150.0))
    df = pd.DataFrame(rows)
    df["overall_rank"] = df["vbd"].rank(ascending=False, method="first").astype(int)
    spec = RosterSpec(teams=10, dedicated={"RB": 2, "WR": 2, "K": 1}, flex=[])
    return DraftBoard(df.sort_values("vbd", ascending=False).reset_index(drop=True), spec, set())


def test_recommend_does_not_chase_a_need_below_replacement():
    from applications.draftplan import STRATEGIES, recommend
    b = _board_with_exhausted_rb()
    # a handful of picks in; you hold nothing yet, RB pool is all negative-VBD
    picks = [{"player_id": f"x{i}", "draft_slot": (i % 10) + 1, "pick_no": i + 1,
              "metadata": {"position": "WR"}} for i in range(6)]
    st = {"picks": picks, "status": "drafting", "teams": 10, "rounds": 15}
    rec = recommend(b, st, my_slot=4, spec=b.spec, strategy=STRATEGIES["bpa"], rounds=15)

    top = rec["recommendations"][0]
    assert top["position"] == "WR" and top["proj_ppg"] is not None
    # the headline must not point at a below-replacement RB
    neg_rbs = {r.name for r in b.players.itertuples() if r.position == "RB" and r.vbd <= 0}
    assert not any(nm in rec["takeaway"] for nm in neg_rbs)


def test_takeaway_never_names_a_player_absent_from_the_card_lists():
    """A high-VBD player with no ADP (injured / buried / un-ranked) must not be
    the headline — the takeaway pick has to be something the user can see."""
    from applications.draftplan import STRATEGIES, recommend
    b = _sim_board()
    # blank out ADP for the top few WRs -> they can't be reasoned about on timing
    top_wr = b.players[b.players["position"] == "WR"].nlargest(3, "vbd")["name"]
    b.players.loc[b.players["name"].isin(top_wr), "adp"] = np.nan
    # a WR run so the "running, take X" branch is in play
    picks = [{"player_id": f"w{i}", "draft_slot": (i % 10) + 1, "pick_no": i + 1,
              "metadata": {"position": "WR"}} for i in range(9)]
    st = {"picks": picks, "status": "drafting", "teams": 10, "rounds": 15}
    rec = recommend(b, st, my_slot=4, spec=b.spec, strategy=STRATEGIES["wr_early"], rounds=15)

    listed = {r["name"] for r in rec["recommendations"]} | {w["name"] for w in rec["wait"]}
    hit = [nm for nm in b.players["name"] if nm in rec["takeaway"]]
    assert hit and all(nm in listed for nm in hit)          # every name it drops is visible
    assert not any(nm in rec["takeaway"] for nm in top_wr)  # not the no-ADP studs


# --- Sleeper empirical ADP + dual-ADP board column -----------------------

def test_sleeper_adp_aggregates_pick_numbers(monkeypatch, tmp_path):
    from applications import adp as adpmod
    monkeypatch.setattr(adpmod, "CACHE_DIR", tmp_path)

    drafts = {
        "d1": [{"pick_no": 3, "metadata": {"first_name": "Al", "last_name": "Pierce",
                                           "position": "WR", "team": "IND"}},
               {"pick_no": 10, "metadata": {"first_name": "Sam", "last_name": "Laporta",
                                            "position": "TE", "team": "DET"}}],
        "d2": [{"pick_no": 7, "metadata": {"first_name": "Al", "last_name": "Pierce",
                                           "position": "WR", "team": "IND"}}],
    }
    monkeypatch.setattr(adpmod.sleeper, "picks", lambda d: drafts[d])
    out = adpmod.sleeper_adp(["d1", "d2"], ttl_hours=0)

    assert set(out.columns) >= {"name", "norm_name", "position", "team", "adp",
                                "times_drafted", "stdev"}
    pierce = out.set_index("norm_name").loc["al pierce"]
    assert pierce["adp"] == pytest.approx(5.0)          # mean of 3 and 7
    assert pierce["times_drafted"] == 2
    assert out.set_index("norm_name").loc["sam laporta", "times_drafted"] == 1


def test_build_board_ref_adp_adds_column_without_touching_vbd(tmp_path, monkeypatch):
    import applications.board as bd

    proj = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(12)],
        "name": [f"RB{i}" for i in range(6)] + [f"WR{i}" for i in range(6)],
        "position": ["RB"] * 6 + ["WR"] * 6,
        "most_recent_team": ["AA"] * 12,
        "proj_points": np.linspace(200, 90, 12), "proj_ppg": np.linspace(14, 6, 12),
        "target_season": 2026,
    })
    proj.to_csv(tmp_path / "sleeper_projections.csv", index=False)
    monkeypatch.setattr(bd, "read_sql", lambda *_a, **_k: pd.DataFrame(
        {"player_id": proj["player_id"], "sleeper_id": proj["player_id"]}))

    spec = RosterSpec(teams=10, dedicated={"RB": 2, "WR": 2}, flex=[])
    base = bd.build_board("sleeper", spec, projections_dir=tmp_path, adp=None)
    ref = pd.DataFrame({"norm_name": ["rb0", "wr0"], "position": ["RB", "WR"], "adp": [5.0, 8.0]})
    withref = bd.build_board("sleeper", spec, projections_dir=tmp_path, adp=None, ref_adp=ref)

    assert "sleeper_adp" in withref.players.columns
    m = withref.players.set_index("name")
    assert m.loc["RB0", "sleeper_adp"] == 5.0 and pd.isna(m.loc["RB1", "sleeper_adp"])
    # vbd untouched
    a = base.players.set_index("name")["vbd"]
    b2 = withref.players.set_index("name")["vbd"]
    assert (a.sort_index().to_numpy() == pytest.approx(b2.sort_index().to_numpy()))
