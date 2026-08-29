"""Canonical fantasy stat vocabulary.

Every platform (Sleeper, Yahoo) names its scoring knobs differently, and nflverse
names its box-score columns differently again. This module defines the *one*
internal vocabulary that both league-settings adapters (``scoring.adapters``) and
stat extractors (``scoring.extract``) translate into, so ``scoring.rules.score``
only ever has to reason about one set of names.

A "canonical stat dict" is ``{canonical_key: numeric_value}`` for a single
player-week (for offense/kicking) or team-week (for DST). Any key may be absent;
absent means zero contribution (see ``scoring.rules.score``).
"""

from __future__ import annotations

# --- Passing -----------------------------------------------------------------
PASS_CMP = "pass_cmp"           # completions
PASS_ATT = "pass_att"           # attempts
PASS_YD = "pass_yd"             # passing yards
PASS_TD = "pass_td"             # passing touchdowns
PASS_INT = "pass_int"           # interceptions thrown
PASS_2PT = "pass_2pt"           # 2-point conversion passes
PASS_SACK = "pass_sack"         # sacks taken (as the passer)

# --- Rushing ---------------------------------------------------------------
RUSH_ATT = "rush_att"           # carries
RUSH_YD = "rush_yd"             # rushing yards
RUSH_TD = "rush_td"             # rushing touchdowns
RUSH_2PT = "rush_2pt"           # 2-point conversion runs

# --- Receiving -----------------------------------------------------------
REC = "rec"                     # receptions
REC_TGT = "rec_tgt"             # targets
REC_YD = "rec_yd"               # receiving yards
REC_TD = "rec_td"               # receiving touchdowns
REC_2PT = "rec_2pt"             # 2-point conversion receptions

# --- Misc offense ------------------------------------------------------
TWO_PT = "two_pt"               # any 2-point conversion (platforms that don't split by type)
FUM_LOST = "fum_lost"           # fumbles lost
FUM_REC_TD = "fum_rec_td"       # offensive fumble-recovery touchdown
RET_YD = "ret_yd"               # kick + punt return yards (credited to the returner)
RET_TD = "ret_td"              # kick + punt return touchdowns (credited to the returner)

# --- Kicking -----------------------------------------------------------
K_XP_MADE = "k_xp_made"
K_XP_MISSED = "k_xp_missed"
K_FG_MADE = "k_fg_made"                 # made FGs, any distance (count)
K_FG_MISSED = "k_fg_missed"             # missed FGs, any distance (count)
K_FG_MADE_YDS = "k_fg_made_yds"         # summed distance of made FGs (Sleeper `fgm_yds`: pts per yard)
K_FG_MADE_0_19 = "k_fg_made_0_19"
K_FG_MADE_20_29 = "k_fg_made_20_29"
K_FG_MADE_30_39 = "k_fg_made_30_39"
K_FG_MADE_40_49 = "k_fg_made_40_49"
K_FG_MADE_50P = "k_fg_made_50p"
K_FG_MISSED_0_19 = "k_fg_missed_0_19"
K_FG_MISSED_20_29 = "k_fg_missed_20_29"
K_FG_MISSED_30_39 = "k_fg_missed_30_39"
K_FG_MISSED_40_49 = "k_fg_missed_40_49"
K_FG_MISSED_50P = "k_fg_missed_50p"

# --- Team defense / special teams (DST) ------------------------------
DST_SACK = "dst_sack"
DST_INT = "dst_int"
DST_FUM_REC = "dst_fum_rec"
DST_FF = "dst_ff"                       # forced fumbles
DST_SAFETY = "dst_safety"
DST_BLK_KICK = "dst_blk_kick"
DST_TD = "dst_td"                       # defensive touchdown (INT/fumble return)
DST_RET_TD = "dst_ret_td"              # kick/punt/blocked-kick return touchdown by the ST unit
DST_2PT_RETURN = "dst_2pt_return"       # defensive 2-point return (Yahoo `extra_point_returned` == blocked XP returned)
DST_4TH_DOWN_STOP = "dst_4th_down_stop"
DST_PTS_ALLOWED = "dst_pts_allowed"     # points allowed this game -> scored via a tier table, not per-unit
DST_YDS_ALLOWED = "dst_yds_allowed"     # total yards allowed this game -> tier table

# Virtual keys usable only as a yardage-bonus target: summed on the fly by score().
COMBINED_YARDAGE = {
    "rush_rec_yd": (RUSH_YD, REC_YD),
    "pass_rush_yd": (PASS_YD, RUSH_YD),
}

ALL_KEYS = frozenset(
    v
    for k, v in globals().items()
    if k.isupper() and isinstance(v, str) and not k.startswith("_")
)
