"""Fantasy ADP (average draft position), for placing players the model can't
project — rookies and anyone with no prior NFL season.

Source: fantasyfootballcalculator.com's public ADP API (crowd-sourced from real
mock/live drafts, no auth). Cached to disk with a TTL so a live-draft refresh
loop doesn't hammer it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import sleeper

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


def sleeper_adp(draft_ids: list[str], ttl_hours: float = 12.0) -> pd.DataFrame:
    """Empirical ADP from completed Sleeper drafts: mean overall pick number for
    each player across ``draft_ids``. Same columns as ``fetch_adp`` so it is a
    drop-in for display next to the crowd ADP.

    Sleeper publishes no usable public ADP endpoint (the ``sleeper.com/graphql``
    ADP fields are gone), but ``/draft/<id>/picks`` is public and reliable — point
    this at a handful of recent mocks of the league you're drafting and it reflects
    that room's scoring/roster, which the fantasyfootballcalculator crowd does not.
    """
    ids = [str(d) for d in draft_ids if d]
    if not ids:
        return pd.DataFrame()

    CACHE_DIR.mkdir(exist_ok=True)
    tag = hashlib.sha1(",".join(sorted(ids)).encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"adp_sleeper_{tag}.json"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_hours * 3600

    if fresh:
        picks = json.loads(cache.read_text())
    else:
        picks = []
        for did in ids:
            picks.extend(sleeper.picks(did))
        cache.write_text(json.dumps(picks))

    rows = []
    for p in picks:
        m = p.get("metadata") or {}
        name = f"{m.get('first_name', '')} {m.get('last_name', '')}".strip()
        pos = m.get("position")
        if not name or not pos or p.get("pick_no") is None:
            continue
        rows.append({"name": name, "position": pos, "team": m.get("team"),
                     "pick_no": float(p["pick_no"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["position"] = df["position"].replace({"PK": "K", "DST": "DEF"})
    df["norm_name"] = df["name"].map(normalize_name)

    g = df.groupby(["norm_name", "position"], as_index=False).agg(
        name=("name", "first"), team=("team", "first"),
        adp=("pick_no", "mean"), times_drafted=("pick_no", "size"),
        stdev=("pick_no", "std"))
    g["stdev"] = g["stdev"].fillna(0.0)
    return g[["name", "position", "team", "adp", "times_drafted", "stdev", "norm_name"]] \
        .sort_values("adp").reset_index(drop=True)
