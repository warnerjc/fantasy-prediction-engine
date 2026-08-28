from scoring.league_scoring import calculate_fantasy_points


def test_standard_ppr_scoring_uses_fractional_yardage():
    settings = {
        "pass_yards": 0.04,
        "pass_td": 4,
        "rush_yards": 0.1,
        "rush_td": 6,
        "rec": 1,
        "rec_yards": 0.1,
        "rec_td": 6,
        "fum_lost": -2,
    }

    stats = {
        "passing_yards": 250,
        "passing_touchdowns": 2,
        "rushing_yards": 80,
        "rushing_touchdowns": 1,
        "receptions": 5,
        "receiving_yards": 60,
        "receiving_touchdowns": 1,
        "fumbles_lost": 1,
    }

    total = calculate_fantasy_points(stats, settings)

    expected = (
        250 * 0.04
        + 2 * 4
        + 80 * 0.1
        + 1 * 6
        + 5 * 1
        + 60 * 0.1
        + 1 * 6
        + 1 * -2
    )

    assert total == expected


def test_missing_stat_maps_to_zero():
    settings = {"pass_yards": 0.04, "rec": 1}
    stats = {"passing_yards": 100}

    assert calculate_fantasy_points(stats, settings) == 100 * 0.04
