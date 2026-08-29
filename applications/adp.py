"""Fantasy ADP (average draft position), for placing players the model can't
project — rookies and anyone with no prior NFL season.

Source: fantasyfootballcalculator.com's public ADP API (crowd-sourced from real
mock/live drafts, no auth). Cached to disk with a TTL so a live-draft refresh
loop doesn't hammer it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "output"
_API = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
_TIMEOUT = 15

# closest FFC format to each league's scoring
FORMAT_BY_LEAGUE = {"sleeper": "half-ppr", "yahoo": "ppr"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    n = re.sub(r"[.'`]", "", str(name).lower())
    parts = [p for p in re.split(r"\s+", n) if p and p not in _SUFFIXES]
    return " ".join(parts)


def fetch_adp(fmt: str, teams: int, year: int, ttl_hours: float = 12.0) -> pd.DataFrame:
    """Columns: name, norm_name, position, team, adp, times_drafted, stdev.
    Served from cache unless the cache is missing or older than ``ttl_hours``."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"adp_{fmt}_{teams}_{year}.json"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_hours * 3600

    if fresh:
        payload = json.loads(cache.read_text())
    else:
        r = requests.get(_API.format(fmt=fmt), params={"teams": teams, "year": year},
                         timeout=_TIMEOUT, headers={"User-Agent": "gsv-fantasy-football"})
        r.raise_for_status()
        payload = r.json()
        cache.write_text(json.dumps(payload))

    rows = payload.get("players", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[["name", "position", "team", "adp", "times_drafted", "stdev"]].copy()
    df["position"] = df["position"].replace({"PK": "K", "DST": "DEF"})
    df["norm_name"] = df["name"].map(normalize_name)
    return df.sort_values("adp").reset_index(drop=True)


def adp_for_league(league: str, teams: int, year: int, **kw) -> pd.DataFrame:
    return fetch_adp(FORMAT_BY_LEAGUE.get(league, "half-ppr"), teams, year, **kw)
