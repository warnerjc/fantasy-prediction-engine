"""Thin Sleeper API client for live draft state. Public API, no auth.

Only what the draft tool needs: the league's draft id, draft metadata (type,
teams, order), and the picks made so far. Sleeper has no push for draft events —
the tool polls this.
"""

from __future__ import annotations

import requests

BASE = "https://api.sleeper.app/v1"
_TIMEOUT = 10


def _get(path: str):
    r = requests.get(f"{BASE}/{path}", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


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
