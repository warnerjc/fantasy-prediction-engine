"""Scoring tests.

Every non-trivial case is hand-computed in the assert (per the scoring-engineer
working method) rather than asserting against a magic number.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from scoring import ScoringRules, normalize_sleeper, normalize_yahoo, score, stat_keys as K
from scoring.extract import (
    canonical_dst_stats,
    canonical_kicking_stats,
    canonical_offense_stats,
    stats_dict,
)
from scoring.rules import Tier, YardageBonus

FIXTURES = Path(__file__).parent / "fixtures"
YAHOO_CONFIG = Path(__file__).resolve().parents[1] / "specifications" / "league-configs" / "yahoo-236625-scoring.json"


@pytest.fixture
def sleeper_rules() -> ScoringRules:
    fx = json.loads((FIXTURES / "sleeper_1356741521163968513.json").read_text())
    return normalize_sleeper(fx["scoring_settings"])


@pytest.fixture
def yahoo_rules() -> ScoringRules:
    return normalize_yahoo(json.loads(YAHOO_CONFIG.read_text()))


# --- real player-week, both leagues -------------------------------------------

# Ja'Marr Chase, Week 5 2024 vs BAL: 10 rec, 193 rec yds, 2 rec TD.
CHASE_W5 = {K.REC: 10, K.REC_YD: 193, K.REC_TD: 2}


def test_chase_under_sleeper_half_ppr_with_bonus(sleeper_rules):
    pts = score(CHASE_W5, sleeper_rules, position="WR")
    # rec 10*0.5 + yds 193*0.1 + td 2*6 + rush/rec-yd-100 bonus (193>=100) 5
    assert pts == pytest.approx(10 * 0.5 + 193 * 0.1 + 2 * 6 + 5)


def test_chase_under_yahoo(yahoo_rules):
    pts = score(CHASE_W5, yahoo_rules, position="WR")
    # reception 10*0.25 + yds 193*0.1 + td 2*6, no bonuses in this league
    assert pts == pytest.approx(10 * 0.25 + 193 * 0.1 + 2 * 6)


def test_qb_passing_yardage_bonus_fires_once(sleeper_rules):
    stats = {K.PASS_YD: 320, K.PASS_TD: 3, K.PASS_INT: 1, K.RUSH_YD: 15}
    pts = score(stats, sleeper_rules, position="QB")
    # 320*0.04 + 3*6 + 1*-2 + 15*0.1 + bonus_pass_yd_300 (5), not _400
    assert pts == pytest.approx(320 * 0.04 + 18 - 2 + 1.5 + 5)


# --- kicking ------------------------------------------------------------------

def test_kicker_distance_buckets_yahoo(yahoo_rules):
    stats = {K.K_FG_MADE_20_29: 1, K.K_FG_MADE_40_49: 2, K.K_FG_MADE_50P: 1,
             K.K_XP_MADE: 3, K.K_FG_MISSED_30_39: 1}
    pts = score(stats, yahoo_rules, position="K")
    assert pts == pytest.approx(1 * 3 + 2 * 5 + 1 * 6 + 3 * 1 + 1 * -1)


def test_kicker_per_yard_sleeper(sleeper_rules):
    stats = {K.K_FG_MADE_YDS: 120, K.K_XP_MADE: 2, K.K_FG_MISSED_20_29: 1}
    pts = score(stats, sleeper_rules, position="K")
    assert pts == pytest.approx(120 * 0.1 + 2 * 1 + 1 * -2)


# --- DST --------------------------------------------------------------------

def test_dst_points_allowed_tier_sleeper(sleeper_rules):
    stats = {K.DST_SACK: 4, K.DST_INT: 2, K.DST_TD: 1, K.DST_PTS_ALLOWED: 10}
    pts = score(stats, sleeper_rules, position="DEF")
    # sack 2, int 2, def_td 6, pts_allow_7_13 -> 7
    assert pts == pytest.approx(4 * 2 + 2 * 2 + 1 * 6 + 7)


def test_dst_yards_allowed_tier_yahoo(yahoo_rules):
    stats = {K.DST_SACK: 3, K.DST_YDS_ALLOWED: 512, K.DST_PTS_ALLOWED: 24}
    pts = score(stats, yahoo_rules, position="DEF")
    # sack 1*3, yards_allowed_500_plus -2, points_allowed_21_27 -> 0
    assert pts == pytest.approx(3 * 1 - 2 + 0)


# --- edge cases ------------------------------------------------------------

def test_bonus_at_exact_threshold_counts():
    rules = ScoringRules(
        per_unit={K.RUSH_YD: 0.1},
        yardage_bonuses=(YardageBonus(K.RUSH_YD, 100, 3),),
    )
    assert score({K.RUSH_YD: 100}, rules) == pytest.approx(10 + 3)
    assert score({K.RUSH_YD: 99}, rules) == pytest.approx(9.9)


def test_unscored_stat_is_ignored():
    rules = ScoringRules(per_unit={K.REC: 1.0})
    assert score({K.REC: 4, K.PASS_INT: 3}, rules) == pytest.approx(4.0)


def test_missing_stat_is_zero():
    rules = ScoringRules(per_unit={K.REC: 1.0, K.REC_YD: 0.1})
    assert score({K.REC_YD: 50}, rules) == pytest.approx(5.0)


def test_non_canonical_per_unit_key_rejected():
    with pytest.raises(ValueError):
        ScoringRules(per_unit={"points_per_touchdown": 6})


# --- standard / no-bonus leagues ------------------------------------------

# a realistic standard NON-PPR Sleeper payload: full key set, PPR + every bonus 0
_STANDARD_NON_PPR_SLEEPER = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 0, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2, "fum_lost": -2,
    "bonus_pass_yd_300": 0, "bonus_rush_yd_100": 0, "bonus_rec_yd_100": 0,
    "bonus_rush_rec_yd_100": 0, "bonus_rush_rec_yd_200": 0,
    "pts_allow_0": 10, "pts_allow_1_6": 7, "pts_allow_7_13": 4,
    "pts_allow_14_20": 1, "pts_allow_21_27": 0, "pts_allow_28_34": -1,
    "pts_allow_35p": -4,
}


def test_standard_non_ppr_drops_receptions_and_bonuses():
    rules = normalize_sleeper(_STANDARD_NON_PPR_SLEEPER)
    assert K.REC not in rules.per_unit          # rec: 0 -> not scored
    assert rules.yardage_bonuses == ()          # all bonuses 0 -> none
    assert rules.yds_allowed_tiers == ()        # no yds_allow_* keys in payload
    # Chase W5: 193*0.1 + 2*6, receptions worth nothing
    assert score(CHASE_W5, rules, position="WR") == pytest.approx(193 * 0.1 + 2 * 6)


def test_minimal_hand_built_rules_no_bonuses_no_tiers():
    rules = ScoringRules(per_unit={
        K.PASS_YD: 0.04, K.PASS_TD: 6, K.PASS_INT: -2,
        K.RUSH_YD: 0.1, K.RUSH_TD: 6,
        K.REC: 0.5, K.REC_YD: 0.1, K.REC_TD: 6, K.FUM_LOST: -2,
    })
    assert score(CHASE_W5, rules, position="WR") == pytest.approx(5 + 19.3 + 12)
    # DST / kicker stats present but unscored by these rules -> ignored, no crash
    assert score({K.DST_SACK: 3, K.DST_PTS_ALLOWED: 7}, rules) == pytest.approx(0.0)


def test_open_top_tier_uses_inf():
    rules = ScoringRules(pts_allowed_tiers=(Tier(35, math.inf, -5),))
    assert score({K.DST_PTS_ALLOWED: 60}, rules) == pytest.approx(-5)


# --- adapters produce sane rules ------------------------------------------

def test_sleeper_adapter_reads_fixture(sleeper_rules):
    assert sleeper_rules.source == "sleeper"
    assert sleeper_rules.per_unit[K.REC] == pytest.approx(0.5)      # half PPR
    assert sleeper_rules.per_unit[K.PASS_TD] == pytest.approx(6)
    assert any(b.stat == "rush_rec_yd" and b.threshold == 100 for b in sleeper_rules.yardage_bonuses)
    assert len(sleeper_rules.pts_allowed_tiers) == 7


def test_yahoo_adapter_reads_config(yahoo_rules):
    assert yahoo_rules.source == "yahoo"
    assert yahoo_rules.per_unit[K.REC] == pytest.approx(0.25)
    assert yahoo_rules.per_unit[K.PASS_INT] == pytest.approx(-3)
    assert yahoo_rules.per_unit[K.PASS_CMP] == pytest.approx(0.15)
    assert any(t.low == 500 for t in yahoo_rules.yds_allowed_tiers)


# --- nflverse stat extraction --------------------------------------------

def test_canonical_offense_stats_maps_and_sums_fumbles():
    weekly = pd.DataFrame([{
        "player_id": "00-1", "position": "RB", "season": 2024, "week": 1,
        "carries": 12, "rushing_yards": 65, "rushing_tds": 1,
        "receptions": 3, "receiving_yards": 20, "targets": 4,
        "rushing_fumbles_lost": 1, "receiving_fumbles_lost": 1, "sack_fumbles_lost": 0,
        "rushing_2pt_conversions": 1,
    }])
    out = canonical_offense_stats(weekly)
    row = out.iloc[0]
    assert row[K.RUSH_YD] == 65
    assert row[K.FUM_LOST] == 2
    assert row[K.TWO_PT] == 1
    d = stats_dict(row)
    assert d[K.RUSH_ATT] == 12 and K.FUM_LOST in d


def test_canonical_kicking_stats_score_under_yahoo(yahoo_rules):
    kicking = pd.DataFrame([{
        "kicker_player_id": "00-k1", "season": 2024, "week": 1,
        "fg_made_20_29": 1, "fg_made_40_49": 2, "fg_made_50p": 1,
        "fg_missed_30_39": 1, "xp_made": 3, "fg_made_yds": 180,
    }])
    d = stats_dict(canonical_kicking_stats(kicking).iloc[0])
    assert score(d, yahoo_rules, position="K") == pytest.approx(
        1 * 3 + 2 * 5 + 1 * 6 + 1 * -1 + 3 * 1
    )


def test_canonical_dst_stats_score_under_sleeper(sleeper_rules):
    team_def = pd.DataFrame([{
        "defense_team": "BAL", "season": 2024, "week": 1,
        "dst_sack": 5, "dst_int": 2, "dst_fum_rec": 1, "dst_td": 1,
        "dst_pts_allowed": 10, "dst_yds_allowed": 300,
    }])
    d = stats_dict(canonical_dst_stats(team_def).iloc[0])
    # sack 2*5, int 2*2, fum_rec 2*1, def_td 6, pts_allow_7_13 -> 7
    assert score(d, sleeper_rules, position="DEF") == pytest.approx(10 + 4 + 2 + 6 + 7)


def test_extracted_stats_score_consistently(sleeper_rules):
    weekly = pd.DataFrame([{
        "player_id": "00-2", "position": "WR", "season": 2024, "week": 5,
        "receptions": 10, "receiving_yards": 193, "receiving_tds": 2,
    }])
    d = stats_dict(canonical_offense_stats(weekly).iloc[0])
    assert score(d, sleeper_rules, position="WR") == pytest.approx(
        10 * 0.5 + 193 * 0.1 + 2 * 6 + 5
    )
