"""Data-layer unit tests (no network): upsert semantics + schedule derivation."""

import numpy as np
import pandas as pd
import pytest

from data.build import parse_seasons
from data.db import PRIMARY_KEYS, connect, read_sql, upsert
from data.nflverse import kicking_stats, team_defense_stats, team_week


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


def test_parse_seasons_ranges_and_lists():
    assert parse_seasons("2015-2018") == [2015, 2016, 2017, 2018]
    assert parse_seasons("2023,2025") == [2023, 2025]
    assert parse_seasons("2024") == [2024]


def test_upsert_is_idempotent_on_pk(conn):
    df = pd.DataFrame({
        "player_id": ["a", "b"], "season": [2024, 2024], "week": [1, 1],
        "season_type": ["REG", "REG"], "passing_yards": [300.0, 0.0],
    })
    assert upsert(conn, "player_week_stats", df) == 2
    # re-run with a corrected stat line -> overwrite, not duplicate
    df.loc[0, "passing_yards"] = 325.0
    upsert(conn, "player_week_stats", df)
    out = read_sql("SELECT * FROM player_week_stats ORDER BY player_id", conn)
    assert len(out) == 2
    assert out.loc[0, "passing_yards"] == 325.0


def test_upsert_rejects_pk_collision_in_batch(conn):
    df = pd.DataFrame({
        "gsis_id": ["x", "x"], "season": [2024, 2024], "week": [3, 3],
        "game_type": ["REG", "REG"], "team": ["GB", "GB"],
    })
    with pytest.raises(ValueError, match="collide on PK"):
        upsert(conn, "injuries", df)


def test_upsert_handles_na_nat_and_numpy(conn):
    df = pd.DataFrame({
        "gsis_id": ["x"], "season": [2024], "week": [3], "game_type": ["REG"],
        "team": ["GB"], "report_status": [pd.NA],
        "date_modified": [pd.Timestamp("2024-11-10T12:00:00Z")],
        "n": [np.int32(5)],
    })
    assert upsert(conn, "injuries", df) == 1
    out = read_sql("SELECT * FROM injuries", conn)
    assert out.loc[0, "report_status"] is None
    assert out.loc[0, "n"] == 5


def test_new_source_column_triggers_alter(conn):
    upsert(conn, "schedules", pd.DataFrame({"game_id": ["g1"], "total_line": [45.0]}))
    upsert(conn, "schedules", pd.DataFrame({"game_id": ["g1"], "total_line": [45.0], "wind": [8.0]}))
    out = read_sql("SELECT * FROM schedules", conn)
    assert "wind" in out.columns and out.loc[0, "wind"] == 8.0


def test_team_week_derives_home_away_rest_and_implied_total():
    sched = pd.DataFrame([{
        "game_id": "2024_01_BAL_KC", "season": 2024, "week": 1, "game_type": "REG",
        "gameday": "2024-09-05", "weekday": "Thursday", "gametime": "20:20",
        "away_team": "BAL", "home_team": "KC", "away_rest": 7, "home_rest": 10,
        "roof": "outdoors", "surface": "grass", "temp": 67.0, "wind": 8.0,
        "div_game": 0, "spread_line": 3.0, "total_line": 46.0,
    }])
    tw = team_week(sched).set_index("team")

    # KC favored by 3 at home: implied 46/2 + 3/2 = 24.5, BAL gets 21.5
    assert tw.loc["KC", "is_home"] == 1
    assert tw.loc["KC", "rest"] == 10
    assert tw.loc["KC", "implied_total"] == pytest.approx(24.5)
    assert tw.loc["BAL", "implied_total"] == pytest.approx(21.5)
    assert tw.loc["BAL", "opponent"] == "KC"
    # spreads are from each team's perspective
    assert tw.loc["KC", "team_spread"] == pytest.approx(-3.0)
    assert tw.loc["BAL", "team_spread"] == pytest.approx(3.0)


def test_every_registered_table_has_pk():
    for pk in PRIMARY_KEYS.values():
        assert pk and all(isinstance(c, str) for c in pk)


def test_kicking_stats_from_player_stats_maps_and_combines_50plus():
    ps = pd.DataFrame([{
        "position": "K", "player_id": "00-k1", "season": 2024, "week": 1,
        "season_type": "REG", "team": "GB",
        "fg_made": 3, "fg_missed": 2, "fg_made_distance": 122,
        "fg_made_20_29": 1, "fg_made_40_49": 1, "fg_made_50_59": 1, "fg_made_60_": 0,
        "fg_missed_40_49": 1, "fg_missed_50_59": 0, "fg_missed_60_": 1,
        "pat_made": 3, "pat_missed": 0,
    }])
    k = kicking_stats(ps).iloc[0]
    assert k["fg_made"] == 3 and k["fg_missed"] == 2
    assert k["fg_made_50p"] == 1 and k["fg_missed_50p"] == 1     # 50_59 + 60_ combined
    assert k["fg_made_yds"] == 122 and k["xp_made"] == 3
    assert k["kicker_player_id"] == "00-k1" and k["game_type"] == "REG"


def test_team_defense_stats_from_team_stats_and_schedules():
    ts = pd.DataFrame([
        dict(game_id="2024_01_CHI_GB", season=2024, week=1, season_type="REG",
             team="GB", opponent_team="CHI", passing_yards=250, rushing_yards=120,
             def_sacks=3, def_interceptions=1, def_fumbles=1, def_safeties=0,
             def_tds=1, def_fg_blocks=0, def_pat_blocks=0, def_punt_blocks=1),
        dict(game_id="2024_01_CHI_GB", season=2024, week=1, season_type="REG",
             team="CHI", opponent_team="GB", passing_yards=180, rushing_yards=60,
             def_sacks=2, def_interceptions=0, def_fumbles=0, def_safeties=0,
             def_tds=0, def_fg_blocks=0, def_pat_blocks=0, def_punt_blocks=0),
    ])
    sched = pd.DataFrame([dict(game_id="2024_01_CHI_GB", home_team="GB", away_team="CHI",
                               home_score=24, away_score=17)])
    d = team_defense_stats(ts, sched).set_index("defense_team")
    assert d.loc["GB", "dst_sack"] == 3
    assert d.loc["GB", "dst_blk_kick"] == 1                     # punt block
    assert d.loc["GB", "dst_pts_allowed"] == 17                 # CHI scored 17
    assert d.loc["GB", "dst_yds_allowed"] == 240                # CHI offense 180 + 60
    assert d.loc["CHI", "dst_pts_allowed"] == 24
