"""Data-layer unit tests (no network): upsert semantics + schedule derivation."""

import numpy as np
import pandas as pd
import pytest

from data.build import parse_seasons
from data.db import PRIMARY_KEYS, connect, read_sql, upsert
from data.nflverse import team_week


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
