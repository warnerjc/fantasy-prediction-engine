"""Translate a platform's league-settings payload into a ``ScoringRules``.

These are the *only* places that know Sleeper's / Yahoo's key names. Everything
downstream works in the canonical vocabulary from ``scoring.stat_keys``.

- ``normalize_sleeper`` consumes the ``scoring_settings`` dict from
  ``GET https://api.sleeper.app/v1/league/<id>`` (flat, e.g. ``{"pass_yd": 0.04}``).
- ``normalize_yahoo`` consumes the hand-captured config under
  ``specifications/league-configs/*.json`` (nested under ``scoring.offense`` etc.).
  Real Yahoo API integration is deferred; when it lands it gets its own adapter
  or this one is pointed at the API shape.
"""

from __future__ import annotations

import math
from typing import Mapping

from . import stat_keys as K
from .rules import ScoringRules, Tier, YardageBonus

# --------------------------------------------------------------------------
# Sleeper
# --------------------------------------------------------------------------

# Sleeper scoring_settings key -> canonical per-unit key. Only 1:1 count*points
# knobs live here; bonuses / tiers / TE premium are handled separately below.
_SLEEPER_PER_UNIT = {
    # passing
    "pass_cmp": K.PASS_CMP, "pass_att": K.PASS_ATT, "pass_yd": K.PASS_YD,
    "pass_td": K.PASS_TD, "pass_int": K.PASS_INT, "pass_2pt": K.PASS_2PT,
    "pass_sack": K.PASS_SACK,
    # rushing
    "rush_att": K.RUSH_ATT, "rush_yd": K.RUSH_YD, "rush_td": K.RUSH_TD,
    "rush_2pt": K.RUSH_2PT,
    # receiving
    "rec": K.REC, "rec_tgt": K.REC_TGT, "rec_yd": K.REC_YD, "rec_td": K.REC_TD,
    "rec_2pt": K.REC_2PT,
    # misc offense
    "fum_lost": K.FUM_LOST, "fum_rec_td": K.FUM_REC_TD,
    # kicking
    "xpm": K.K_XP_MADE, "xpmiss": K.K_XP_MISSED, "fgm_yds": K.K_FG_MADE_YDS,
    "fgm": K.K_FG_MADE, "fgmiss": K.K_FG_MISSED,
    "fgm_0_19": K.K_FG_MADE_0_19, "fgm_20_29": K.K_FG_MADE_20_29,
    "fgm_30_39": K.K_FG_MADE_30_39, "fgm_40_49": K.K_FG_MADE_40_49,
    "fgm_50p": K.K_FG_MADE_50P,
    "fgmiss_0_19": K.K_FG_MISSED_0_19, "fgmiss_20_29": K.K_FG_MISSED_20_29,
    "fgmiss_30_39": K.K_FG_MISSED_30_39, "fgmiss_40_49": K.K_FG_MISSED_40_49,
    "fgmiss_50p": K.K_FG_MISSED_50P,
    # DST. Sleeper also exposes st_fum_rec / st_ff / def_st_* variants that are
    # separate events from the plain def_* keys; splitting special-teams vs
    # defensive recoveries needs a play-by-play DST extractor that doesn't exist
    # yet, so those variants are intentionally NOT mapped here (TODO: revisit when
    # /data lands DST stats). st_td stays -- a return TD is unambiguous.
    "sack": K.DST_SACK, "int": K.DST_INT, "fum_rec": K.DST_FUM_REC,
    "ff": K.DST_FF, "safe": K.DST_SAFETY, "blk_kick": K.DST_BLK_KICK,
    "def_td": K.DST_TD, "def_2pt": K.DST_2PT_RETURN, "st_td": K.DST_RET_TD,
}

# bonus_<stat>_<threshold> -> (canonical yardage key, threshold)
_SLEEPER_BONUS = {
    "bonus_pass_yd_300": (K.PASS_YD, 300), "bonus_pass_yd_400": (K.PASS_YD, 400),
    "bonus_rush_yd_100": (K.RUSH_YD, 100), "bonus_rush_yd_200": (K.RUSH_YD, 200),
    "bonus_rec_yd_100": (K.REC_YD, 100), "bonus_rec_yd_200": (K.REC_YD, 200),
    "bonus_rush_rec_yd_100": ("rush_rec_yd", 100),
    "bonus_rush_rec_yd_200": ("rush_rec_yd", 200),
}

_SLEEPER_PTS_ALLOWED = [
    ("pts_allow_0", 0, 0), ("pts_allow_1_6", 1, 6), ("pts_allow_7_13", 7, 13),
    ("pts_allow_14_20", 14, 20), ("pts_allow_21_27", 21, 27),
    ("pts_allow_28_34", 28, 34), ("pts_allow_35p", 35, math.inf),
]

_SLEEPER_YDS_ALLOWED = [
    ("yds_allow_0_100", 0, 99), ("yds_allow_100_199", 100, 199),
    ("yds_allow_200_299", 200, 299), ("yds_allow_300_349", 300, 349),
    ("yds_allow_350_399", 350, 399), ("yds_allow_400_449", 400, 449),
    ("yds_allow_450_499", 450, 499), ("yds_allow_500_549", 500, 549),
    ("yds_allow_550p", 550, math.inf),
]


def _nonzero(settings: Mapping, key: str) -> bool:
    return key in settings and settings[key] not in (None, 0, 0.0)


def normalize_sleeper(scoring_settings: Mapping[str, float]) -> ScoringRules:
    per_unit: dict[str, float] = {}
    for sleeper_key, canonical in _SLEEPER_PER_UNIT.items():
        if _nonzero(scoring_settings, sleeper_key):
            if canonical in per_unit:
                raise ValueError(
                    f"two Sleeper keys map to {canonical!r}; mapping is ambiguous"
                )
            per_unit[canonical] = float(scoring_settings[sleeper_key])

    bonuses = tuple(
        YardageBonus(stat=stat, threshold=thr, points=float(scoring_settings[key]))
        for key, (stat, thr) in _SLEEPER_BONUS.items()
        if _nonzero(scoring_settings, key)
    )

    pts_tiers = tuple(
        Tier(lo, hi, float(scoring_settings[key]))
        for key, lo, hi in _SLEEPER_PTS_ALLOWED
        if key in scoring_settings
    )
    yds_tiers = tuple(
        Tier(lo, hi, float(scoring_settings[key]))
        for key, lo, hi in _SLEEPER_YDS_ALLOWED
        if key in scoring_settings
    )

    position_bonuses: dict[str, dict[str, float]] = {}
    if _nonzero(scoring_settings, "bonus_rec_te"):
        position_bonuses["TE"] = {K.REC: float(scoring_settings["bonus_rec_te"])}

    return ScoringRules(
        per_unit=per_unit,
        yardage_bonuses=bonuses,
        pts_allowed_tiers=pts_tiers,
        yds_allowed_tiers=yds_tiers,
        position_bonuses=position_bonuses,
        source="sleeper",
        raw=dict(scoring_settings),
    )


# --------------------------------------------------------------------------
# Yahoo (hand-captured config shape)
# --------------------------------------------------------------------------

_YAHOO_OFFENSE = {
    "completion": K.PASS_CMP, "passing_yard": K.PASS_YD, "passing_td": K.PASS_TD,
    "interception_thrown": K.PASS_INT,
    "rushing_yard": K.RUSH_YD, "rushing_td": K.RUSH_TD,
    "reception": K.REC, "receiving_yard": K.REC_YD, "receiving_td": K.REC_TD,
    "return_yard": K.RET_YD, "return_td": K.RET_TD,
    "two_point_conversion": K.TWO_PT,
    "fumble_lost": K.FUM_LOST, "offensive_fumble_return_td": K.FUM_REC_TD,
}

_YAHOO_KICKING = {
    "fg_made_0_19": K.K_FG_MADE_0_19, "fg_made_20_29": K.K_FG_MADE_20_29,
    "fg_made_30_39": K.K_FG_MADE_30_39, "fg_made_40_49": K.K_FG_MADE_40_49,
    "fg_made_50_plus": K.K_FG_MADE_50P,
    "fg_missed_0_19": K.K_FG_MISSED_0_19, "fg_missed_20_29": K.K_FG_MISSED_20_29,
    "fg_missed_30_39": K.K_FG_MISSED_30_39, "fg_missed_40_49": K.K_FG_MISSED_40_49,
    "fg_missed_50_plus": K.K_FG_MISSED_50P,
    "pat_made": K.K_XP_MADE, "pat_missed": K.K_XP_MISSED,
}

_YAHOO_DST = {
    "sack": K.DST_SACK, "interception": K.DST_INT, "fumble_recovery": K.DST_FUM_REC,
    "forced_fumble": K.DST_FF, "defensive_touchdown": K.DST_TD, "safety": K.DST_SAFETY,
    "block_kick": K.DST_BLK_KICK, "kick_punt_return_td": K.DST_RET_TD,
    "fourth_down_stop": K.DST_4TH_DOWN_STOP, "extra_point_returned": K.DST_2PT_RETURN,
}

_YAHOO_PTS_ALLOWED = {
    "points_allowed_0": (0, 0), "points_allowed_1_6": (1, 6),
    "points_allowed_7_13": (7, 13), "points_allowed_14_20": (14, 20),
    "points_allowed_21_27": (21, 27), "points_allowed_28_34": (28, 34),
    "points_allowed_35_plus": (35, math.inf),
}

_YAHOO_YDS_ALLOWED = {
    "yards_allowed_0_99": (0, 99), "yards_allowed_100_199": (100, 199),
    "yards_allowed_200_299": (200, 299), "yards_allowed_300_399": (300, 399),
    "yards_allowed_400_499": (400, 499), "yards_allowed_500_plus": (500, math.inf),
}


def normalize_yahoo(config: Mapping) -> ScoringRules:
    """``config`` is a parsed league-config JSON (see specifications/league-configs/)."""
    scoring = config.get("scoring", config)  # tolerate being handed the inner dict
    flat: dict[str, float] = {}
    for section in ("offense", "kicking", "defense_special_teams"):
        flat.update(scoring.get(section, {}))

    per_unit: dict[str, float] = {}
    for group in (_YAHOO_OFFENSE, _YAHOO_KICKING, _YAHOO_DST):
        for yahoo_key, canonical in group.items():
            if yahoo_key in flat and flat[yahoo_key] is not None:
                per_unit[canonical] = per_unit.get(canonical, 0.0) + float(flat[yahoo_key])

    pts_tiers = tuple(
        Tier(lo, hi, float(flat[key]))
        for key, (lo, hi) in _YAHOO_PTS_ALLOWED.items()
        if key in flat
    )
    yds_tiers = tuple(
        Tier(lo, hi, float(flat[key]))
        for key, (lo, hi) in _YAHOO_YDS_ALLOWED.items()
        if key in flat
    )

    return ScoringRules(
        per_unit=per_unit,
        pts_allowed_tiers=pts_tiers,
        yds_allowed_tiers=yds_tiers,
        source="yahoo",
        raw=dict(config),
    )
