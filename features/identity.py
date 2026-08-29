"""Player priors known before a snap is played: age, experience, draft capital.

All pre-game-known. Strong for a season projection — draft capital and age carry
the signal for ascending young players that last year's box score misses.
Source: the ``player_ids`` crosswalk (nflverse ``import_ids``).
"""

from __future__ import annotations

import pandas as pd

from .window import AsOf

# season is taken to "start" here for age purposes
_SEASON_START = "-09-01"


def identity_features(player_ids: pd.DataFrame, as_of: AsOf) -> pd.DataFrame:
    """One row per ``gsis_id`` with age / experience / draft columns as of the
    as-of season. Returned keyed as ``player_id`` to join to the feature matrix.
    """
    df = player_ids.copy()
    df = df[df["gsis_id"].notna()]

    born = pd.to_datetime(df.get("birthdate"), errors="coerce")
    season_start = pd.Timestamp(f"{as_of.season}{_SEASON_START}")
    df["age"] = (season_start - born).dt.days / 365.25

    draft_year = pd.to_numeric(df.get("draft_year"), errors="coerce")
    df["years_exp"] = as_of.season - draft_year
    df["is_rookie"] = (df["years_exp"] <= 0).astype("Int64")
    df["draft_round"] = pd.to_numeric(df.get("draft_round"), errors="coerce")
    df["draft_ovr"] = pd.to_numeric(df.get("draft_ovr"), errors="coerce")
    df["undrafted"] = draft_year.isna().astype("Int64")  # 1 = no draft record

    keep = ["gsis_id", "age", "years_exp", "is_rookie", "draft_round", "draft_ovr", "undrafted"]
    return df[keep].rename(columns={"gsis_id": "player_id"}).reset_index(drop=True)
