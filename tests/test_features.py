"""Feature tests — synthetic player-weeks, hand-checked aggregates, leakage guards."""

import numpy as np
import pandas as pd
import pytest

from features import (
    AsOf,
    Window,
    context_features,
    defense_feature_matrix,
    identity_features,
    kicker_feature_matrix,
    opponent_allowed_features,
    opportunity_features,
    season_feature_matrix,
    visible_weeks,
    week_index,
)


def _pws_row(player, pos, team, season, week, opp="OPP", **stats):
    base = dict(
        player_id=player, position=pos, team=team, opponent=opp,
        season=season, week=week, season_type="REG",
        targets=0, receptions=0, receiving_yards=0, receiving_tds=0, receiving_air_yards=0,
        carries=0, rushing_yards=0, rushing_tds=0,
        attempts=0, completions=0, passing_yards=0, passing_tds=0, interceptions=0,
    )
    base.update(stats)
    return base


@pytest.fixture
def pws():
    rows = []
    # WR1 on AAA: 2022 + 2023, plus a 2024 week that must never leak into a 2024 as-of
    for wk in range(1, 5):
        rows.append(_pws_row("wr1", "WR", "AAA", 2023, wk, targets=10, receptions=6,
                             receiving_yards=90, receiving_air_yards=120, receiving_tds=1))
    rows.append(_pws_row("wr1", "WR", "AAA", 2022, 1, targets=4, receptions=3, receiving_yards=30))
    rows.append(_pws_row("wr1", "WR", "AAA", 2024, 1, targets=99, receiving_yards=999))
    # WR2 on AAA 2023: the rest of the team's targets, so shares are checkable
    for wk in range(1, 5):
        rows.append(_pws_row("wr2", "WR", "AAA", 2023, wk, targets=10, receptions=7,
                             receiving_yards=70, receiving_air_yards=80))
    # a RB whose production is what defense BBB "allowed"
    for wk in range(1, 5):
        rows.append(_pws_row("rb1", "RB", "CCC", 2023, wk, opp="BBB",
                             carries=15, rushing_yards=75, rushing_tds=1))
    return pd.DataFrame(rows)


# --- window / leakage ---------------------------------------------------------

def test_week_index_orders_across_seasons():
    assert week_index(2023, 18) < week_index(2024, 1)


def test_prior_season_window_excludes_as_of_season(pws):
    vis = visible_weeks(pws, AsOf(2024, 1), Window.prior_season())
    assert set(vis["season"]) == {2023}
    assert not (vis["player_id"].eq("wr1") & vis["season"].eq(2024)).any()


def test_prior_season_n_seasons_reaches_further_back(pws):
    vis = visible_weeks(pws, AsOf(2024, 1), Window.prior_season(n_seasons=2))
    assert set(vis["season"]) == {2022, 2023}


def test_trailing_window_takes_last_n_games_played(pws):
    vis = visible_weeks(pws, AsOf(2023, 4), Window.trailing(2))
    wr1 = vis[vis["player_id"] == "wr1"]
    assert list(wr1["week"]) == [2, 3]            # strictly before wk4, last 2
    assert wr1["week_index"].max() < AsOf(2023, 4).index


# --- opportunity ------------------------------------------------------------

def test_opportunity_totals_rates_and_shares(pws):
    vis = visible_weeks(pws, AsOf(2024, 1), Window.prior_season())
    feats = opportunity_features(vis).set_index("player_id")
    wr1 = feats.loc["wr1"]
    assert wr1["games"] == 4
    assert wr1["targets"] == 40 and wr1["targets_pg"] == 10
    assert wr1["rec_yd"] == 360 and wr1["rec_yd_pg"] == 90
    # wr1 had 10 of 20 team targets every week -> 0.5 target share
    assert wr1["target_share"] == pytest.approx(0.5)
    # air yards 120 of 200 -> 0.6
    assert wr1["air_yards_share"] == pytest.approx(0.6)
    assert wr1["wopr"] == pytest.approx(1.5 * 0.5 + 0.7 * 0.6)
    assert wr1["yards_per_target"] == pytest.approx(90 / 10)


def test_opponent_allowed_is_per_game_by_position(pws):
    vis = visible_weeks(pws, AsOf(2024, 1), Window.prior_season())
    opp = opponent_allowed_features(vis).set_index("defense_team")
    # BBB faced rb1 4 times, 75 rush yds each
    assert opp.loc["BBB", "def_RB_rush_yd_pg"] == pytest.approx(75.0)
    assert opp.loc["BBB", "def_RB_rush_td_pg"] == pytest.approx(1.0)
    assert opp.loc["BBB", "def_games"] == 4


# --- context ---------------------------------------------------------------

def test_context_features_for_as_of_week():
    tw = pd.DataFrame([
        dict(season=2024, week=1, game_type="REG", team="KC", opponent="BAL", is_home=1,
             rest=10, div_game=0, implied_total=27.0, team_spread=-3.0, roof="outdoors",
             temp=70, wind=5),
        dict(season=2024, week=1, game_type="REG", team="DET", opponent="LA", is_home=1,
             rest=7, div_game=0, implied_total=26.0, team_spread=-4.0, roof="dome",
             temp=None, wind=None),
        dict(season=2024, week=2, game_type="REG", team="KC", opponent="CIN", is_home=0,
             rest=7, div_game=0, implied_total=24.0, team_spread=-1.0, roof="outdoors",
             temp=75, wind=3),
    ])
    ctx = context_features(tw, AsOf(2024, 1)).set_index("team")
    assert len(ctx) == 2                          # only week 1
    assert ctx.loc["KC", "is_dome"] == 0 and ctx.loc["KC", "is_outdoors"] == 1
    assert ctx.loc["DET", "is_dome"] == 1
    assert ctx.loc["KC", "short_week"] == 0


# --- identity -------------------------------------------------------------

def test_identity_age_experience_and_draft():
    pids = pd.DataFrame([
        dict(gsis_id="p1", birthdate="2000-03-01", draft_year=2021, draft_round=1, draft_ovr=5),
        dict(gsis_id="p2", birthdate="1995-01-01", draft_year=None, draft_round=None, draft_ovr=None),
    ])
    idn = identity_features(pids, AsOf(2024, 1)).set_index("player_id")
    assert idn.loc["p1", "age"] == pytest.approx(24.5, abs=0.1)
    assert idn.loc["p1", "years_exp"] == 3
    assert idn.loc["p1", "undrafted"] == 0
    assert idn.loc["p2", "undrafted"] == 1
    assert idn.loc["p2", "is_rookie"] == 0


# --- assembled matrix --------------------------------------------------------

def test_team_change_features_flag_and_vacated_share():
    from features import team_change_features

    rows = []
    # rb1 on OLD 2023, moves to NEW 2024. rb2 was on NEW 2023, leaves (vacates carries).
    for wk in range(1, 6):
        rows.append(_pws_row("rb1", "RB", "OLD", 2023, wk, carries=10, rushing_yards=40))
        rows.append(_pws_row("rb2", "RB", "NEW", 2023, wk, carries=20, rushing_yards=80))
        rows.append(_pws_row("rb3", "RB", "NEW", 2023, wk, carries=5, rushing_yards=15))
    pws = pd.DataFrame(rows)
    rosters = pd.DataFrame([
        dict(player_id="rb1", season=2024, team="NEW"),   # moved OLD -> NEW
        dict(player_id="rb2", season=2024, team="ELSE"),   # left NEW
        dict(player_id="rb3", season=2024, team="NEW"),    # stayed
    ])
    tw = pd.DataFrame([dict(season=2023, week=w, game_type="REG", team=t, opponent="X",
                            is_home=1, rest=7, div_game=0, implied_total=22.0,
                            team_spread=0.0, roof="dome", temp=None, wind=None)
                       for w in range(1, 6) for t in ("OLD", "NEW")])

    tc = team_change_features(pws, rosters, tw, 2024).set_index("player_id")
    assert tc.loc["rb1", "changed_team"] == 1
    assert tc.loc["rb3", "changed_team"] == 0
    # NEW's 2023 carries: rb2 100 + rb3 25 = 125; rb2 (100) departed -> 0.8 vacated
    assert tc.loc["rb1", "vacated_rush_share_new_team"] == pytest.approx(0.8)


def test_kicker_and_defense_matrices_are_prior_season_only():
    tw = pd.DataFrame([
        dict(season=2023, week=w, game_type="REG", team="GB", opponent="X", is_home=1,
             rest=7, div_game=0, implied_total=24.0, team_spread=-2.0, roof="dome",
             temp=None, wind=None)
        for w in range(1, 5)
    ])
    kick = pd.DataFrame([
        dict(kicker_player_id="k1", season=2023, week=w, game_type="REG", team="GB",
             fg_made=2, fg_missed=0, fg_made_50p=1, fg_made_yds=80, xp_made=3, xp_missed=0)
        for w in range(1, 5)
    ] + [dict(kicker_player_id="k1", season=2024, week=1, game_type="REG", team="GB",
              fg_made=9, fg_missed=9, fg_made_50p=9, fg_made_yds=999, xp_made=9, xp_missed=9)])
    km = kicker_feature_matrix(kick, tw, 2024).set_index("player_id")
    assert km.loc["k1", "games"] == 4                     # 2023 only
    assert km.loc["k1", "fg_made_pg"] == 2
    assert km.loc["k1", "team_implied_total_prior"] == pytest.approx(24.0)

    d = pd.DataFrame([
        dict(defense_team="GB", season=2023, week=w, game_type="REG", dst_sack=3, dst_int=1,
             dst_fum_rec=1, dst_safety=0, dst_td=0, dst_blk_kick=0, dst_pts_allowed=20,
             dst_yds_allowed=330)
        for w in range(1, 5)
    ])
    dm = defense_feature_matrix(d, tw, 2024).set_index("player_id")
    assert dm.loc["GB", "dst_sack_pg_prior"] == pytest.approx(3.0)
    assert dm.loc["GB", "takeaways_pg_prior"] == pytest.approx(2.0)


def test_season_feature_matrix_is_keyed_and_leak_free(pws):
    snaps = pd.DataFrame(columns=["gsis_id", "season", "week", "game_type", "offense_pct",
                                  "offense_snaps", "st_pct"])
    pids = pd.DataFrame([dict(gsis_id="wr1", birthdate="2000-01-01", draft_year=2021,
                              draft_round=2, draft_ovr=40)])
    fm = season_feature_matrix(pws, snaps, pids, 2024)
    assert {"player_id", "target_season"}.issubset(fm.columns)
    assert (fm["target_season"] == 2024).all()
    wr1 = fm[fm["player_id"] == "wr1"].iloc[0]
    assert wr1["targets"] == 40                   # 2023 only, not the 99 from 2024
