"""Pure scoring utilities for fantasy football point calculations.

This module is intentionally league-agnostic: it takes raw player stat values and
applies a scoring-settings dict. The same function is used for model training and
for final draft/weekly application logic.
"""

from __future__ import annotations


STAT_ALIASES = {
    "pass_yards": ("passing_yards", "pass_yards"),
    "pass_td": ("passing_touchdowns", "pass_td"),
    "pass_int": ("passing_interceptions", "pass_int"),
    "rush_yards": ("rushing_yards", "rush_yards"),
    "rush_td": ("rushing_touchdowns", "rush_td"),
    "rec": ("receptions", "rec"),
    "rec_yards": ("receiving_yards", "rec_yards"),
    "rec_td": ("receiving_touchdowns", "rec_td"),
    "fum_lost": ("fumbles_lost", "fum_lost"),
    "fum_rec": ("fumbles_recovered", "fum_rec"),
    "two_pt_pass": ("two_point_passes", "two_pt_pass"),
    "two_pt_rush": ("two_point_runs", "two_pt_rush"),
    "two_pt_rec": ("two_point_receptions", "two_pt_rec"),
}


def _coerce_stat_value(stats: dict, stat_name: str):
    """Find the raw stat value for a settings key, allowing common alias names."""
    candidates = [stat_name]
    candidates.extend(STAT_ALIASES.get(stat_name, ()))

    for candidate in candidates:
        if candidate in stats and stats[candidate] is not None:
            return stats[candidate]

    return 0


def calculate_fantasy_points(stats: dict, settings: dict) -> float:
    """Convert raw stat dict into fantasy points using a scoring settings dict.

    Settings keys are expected to be stat names like "pass_yards", "rec", etc.,
    with values representing points per occurrence. Missing stats simply contribute 0.
    """
    total = 0.0

    for stat_name, points_per_unit in settings.items():
        value = _coerce_stat_value(stats, stat_name)
        if value == 0 and stat_name not in stats and stat_name not in STAT_ALIASES:
            continue
        total += float(value) * float(points_per_unit)

    return total
