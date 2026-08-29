"""The pure scoring function and its normalized rule representation.

``ScoringRules`` is what every league's settings get normalized *into* (by
``scoring.adapters``). ``score()`` is the single implementation of
raw-stats -> fantasy-points, called identically by model-training label
generation and by the draft/weekly application layer. There is deliberately no
default scoring format here: a caller with no real league settings must build a
``ScoringRules`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from . import stat_keys


@dataclass(frozen=True)
class YardageBonus:
    """Flat bonus awarded once when a (possibly combined) yardage stat hits a threshold.

    ``stat`` is either a canonical yardage key (e.g. ``"pass_yd"``) or a virtual
    combined key from ``stat_keys.COMBINED_YARDAGE`` (e.g. ``"rush_rec_yd"``).
    """

    stat: str
    threshold: float
    points: float


@dataclass(frozen=True)
class Tier:
    """Closed interval ``[low, high]`` mapping to a point value.

    Used for DST points-allowed and yards-allowed tables. ``high`` may be
    ``math.inf`` for the open top bucket.
    """

    low: float
    high: float
    points: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class ScoringRules:
    """A league's scoring settings, normalized to the canonical vocabulary.

    ``per_unit`` covers everything that is (count * points): yards, TDs, INTs,
    receptions, made kicks, DST sacks, etc. ``yardage_bonuses`` and the two tier
    tables cover the non-linear pieces.
    """

    per_unit: Mapping[str, float] = field(default_factory=dict)
    yardage_bonuses: tuple[YardageBonus, ...] = ()
    pts_allowed_tiers: tuple[Tier, ...] = ()
    yds_allowed_tiers: tuple[Tier, ...] = ()
    # position -> {canonical_stat: extra points per unit}, e.g. TE receiving premium
    position_bonuses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    source: str = "manual"          # "sleeper" | "yahoo" | "manual"
    raw: Mapping = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        keys = set(self.per_unit)
        for extra in self.position_bonuses.values():
            keys |= set(extra)
        unknown = keys - stat_keys.ALL_KEYS
        if unknown:
            raise ValueError(f"non-canonical scoring keys: {sorted(unknown)}")


def _yardage_for(stats: Mapping[str, float], stat: str) -> float:
    if stat in stat_keys.COMBINED_YARDAGE:
        return sum(float(stats.get(k, 0) or 0) for k in stat_keys.COMBINED_YARDAGE[stat])
    return float(stats.get(stat, 0) or 0)


def _tier_points(tiers: tuple[Tier, ...], value: float | None) -> float:
    if value is None or not tiers:
        return 0.0
    for tier in tiers:
        if tier.contains(float(value)):
            return tier.points
    return 0.0


def score(
    stats: Mapping[str, float],
    rules: ScoringRules,
    position: str | None = None,
) -> float:
    """Fantasy points for one player-week (or team-week for DST).

    ``stats`` is a canonical stat dict (see ``scoring.stat_keys``). Any key absent
    from ``stats`` contributes zero; any stat present that the league does not
    score is simply ignored. ``position`` (e.g. ``"TE"``) activates any matching
    entry in ``rules.position_bonuses`` — needed for TE receiving premiums.
    """
    total = 0.0

    per_unit = dict(rules.per_unit)
    if position and position in rules.position_bonuses:
        for key, extra in rules.position_bonuses[position].items():
            per_unit[key] = per_unit.get(key, 0.0) + extra

    for key, pts in per_unit.items():
        value = stats.get(key)
        if value:
            total += float(value) * float(pts)

    for bonus in rules.yardage_bonuses:
        if _yardage_for(stats, bonus.stat) >= bonus.threshold:
            total += bonus.points

    total += _tier_points(rules.pts_allowed_tiers, stats.get(stat_keys.DST_PTS_ALLOWED))
    total += _tier_points(rules.yds_allowed_tiers, stats.get(stat_keys.DST_YDS_ALLOWED))

    return total
