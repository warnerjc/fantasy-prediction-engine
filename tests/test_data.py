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


def _pbp_fg(dist, result, kicker="00-k1", team="GB", wk=1):
    return dict(season=2023, week=wk, season_type="REG", posteam=team, defteam="CHI",
                game_id=f"2023_0{wk}_CHI_GB", home_team="GB", away_team="CHI",
                home_score=24, away_score=17,
                field_goal_attempt=1, extra_point_attempt=0, kick_distance=dist,
                field_goal_result=result, extra_point_result=None,
                kicker_player_id=kicker, touchdown=0, td_team=None,
                sack=0, interception=0, fumble_lost=0, safety=0, yards_gained=0)


def test_kicking_stats_buckets_makes_and_misses_by_distance():
    pbp = pd.DataFrame([
        _pbp_fg(25, "made"), _pbp_fg(45, "made"), _pbp_fg(52, "made"),
        _pbp_fg(48, "missed"), _pbp_fg(55, "blocked"),
    ])
    k = kicking_stats(pbp).iloc[0]
    assert k["fg_made"] == 3 and k["fg_missed"] == 2
    assert k["fg_made_20_29"] == 1 and k["fg_made_40_49"] == 1 and k["fg_made_50p"] == 1
    assert k["fg_missed_40_49"] == 1 and k["fg_missed_50p"] == 1   # blocked counts as missed
    assert k["fg_made_yds"] == 25 + 45 + 52


def test_team_defense_stats_aggregates_and_points_allowed():
    rows = []
    for _ in range(3):
        rows.append(dict(season=2023, week=1, season_type="REG", game_id="2023_01_CHI_GB",
                         home_team="GB", away_team="CHI", home_score=24, away_score=17,
                         posteam="CHI", defteam="GB", sack=1, interception=0, fumble_lost=0,
                         safety=0, yards_gained=5, touchdown=0, td_team=None,
                         field_goal_result=None, extra_point_result=None))
    rows.append({**rows[0], "sack": 0, "interception": 1, "touchdown": 1,
                 "td_team": "GB", "yards_gained": 30})
    d = team_defense_stats(pd.DataFrame(rows))
    gb = d[d["defense_team"] == "GB"].iloc[0]
    assert gb["dst_sack"] == 3
    assert gb["dst_int"] == 1
    assert gb["dst_td"] == 1
    assert gb["dst_pts_allowed"] == 17          # GB (home) conceded away_score
    assert gb["dst_yds_allowed"] == 45
