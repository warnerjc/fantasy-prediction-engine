"""Thin Sleeper API client for live draft state. Public API, no auth.

Only what the draft tool needs: the league's draft id, draft metadata (type,
teams, order), and the picks made so far. Sleeper has no push for draft events —
the tool polls this.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.sleeper.app/v1"
_TIMEOUT = 30
_CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "output"


def _get(path: str):
    r = requests.get(f"{BASE}/{path}", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def all_players(ttl_hours: float = 24.0) -> pd.DataFrame:
    """The full Sleeper player directory (~14 MB), cached to disk. Columns:
    sleeper_id, name, norm_name, position, team, years_exp, is_rookie."""
    from .adp import normalize_name

    _CACHE_DIR.mkdir(exist_ok=True)
    cache = _CACHE_DIR / "sleeper_players.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_hours * 3600:
        raw = json.loads(cache.read_text())
    else:
        raw = _get("players/nfl")
        cache.write_text(json.dumps(raw))

    rows = []
    for sid, p in raw.items():
        pos = p.get("position")
        if pos not in ("QB", "RB", "WR", "TE", "K", "DEF"):
            continue
        name = p.get("full_name") or p.get("last_name") or sid
        yx = p.get("years_exp")
        rows.append({
            "sleeper_id": sid, "name": name, "norm_name": normalize_name(name),
            "position": pos, "team": p.get("team"), "years_exp": yx,
            "is_rookie": int(yx == 0) if yx is not None else 0,
        })
    return pd.DataFrame(rows)


def league(league_id: str) -> dict:
    return _get(f"league/{league_id}")


def draft_id_for_league(league_id: str) -> str:
    d = league(league_id).get("draft_id")
    if not d:
        raise RuntimeError(f"league {league_id} has no draft_id yet")
    return d


def draft(draft_id: str) -> dict:
    return _get(f"draft/{draft_id}")


def picks(draft_id: str) -> list[dict]:
    """All picks made so far, in order. Each has ``player_id`` (a Sleeper id),
    ``picked_by``, ``round``, ``pick_no``, ``draft_slot``."""
    return _get(f"draft/{draft_id}/picks")


def drafted_sleeper_ids(draft_id: str) -> set[str]:
    return {p["player_id"] for p in picks(draft_id) if p.get("player_id")}


def draft_state(draft_id: str) -> dict:
    """Normalized snapshot: type, teams, rounds, status, slot->roster map,
    picks so far, and next overall pick number."""
    d = draft(draft_id)
    made = picks(draft_id)
    s = d.get("settings", {})
    return {
        "type": d.get("type"),
        "status": d.get("status"),
        "teams": s.get("teams"),
        "rounds": s.get("rounds"),
        "slot_to_roster_id": d.get("slot_to_roster_id") or {},
        "draft_order": d.get("draft_order") or {},
        "picks": made,
        "next_pick_no": len(made) + 1,
    }
