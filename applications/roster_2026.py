"""Current-season team + availability corrections for the draft board.

The model carries every player's *last-season* team (from game data) and can't see
offseason moves, cuts, or suspensions. The obvious automated fix — nflverse's
upcoming-season roster release — is **not trustworthy at cut time**: the 2026 file
had A.J. Brown on NE correctly but also Isiah Pacheco on DET and Justin Fields on
KC (both wrong), and dropped rostered veterans (Diggs, Najee, Deebo…). So the only
input here is a hand-maintained JSON:
`specifications/league-configs/roster-2026-overrides.json`.

`overrides.team` relabels a player's team; `overrides.out` removes them from the
board (retired / unsigned / suspended); `overrides.adp` forces a player's ADP when
the cached crowd value lags breaking news (e.g. a suspension tanking his draft
stock). Nothing here touches the model's projection — team/availability labels and
the ADP that feeds the market blend. Revisit the automated path once nflverse's
roster data firms up (usually the first week or two of September).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .adp import normalize_name

_OVERRIDES = (
    Path(__file__).resolve().parents[1]
    / "specifications" / "league-configs" / "roster-2026-overrides.json"
)


def load_overrides(path: Path = _OVERRIDES) -> tuple[dict[str, str], set[str], dict[str, float]]:
    """`({norm_name: team}, {norm_name}, {norm_name: adp})` — team relabels, the
    OUT set, and forced ADP values."""
    try:
        cfg = json.loads(path.read_text())
    except FileNotFoundError:
        return {}, set(), {}
    team = {normalize_name(k): v for k, v in cfg.get("team", {}).items()}
    out = {normalize_name(k) for k in cfg.get("out", {})}
    adp = {normalize_name(k): float(v) for k, v in cfg.get("adp", {}).items()}
    return team, out, adp


def apply_team_labels(proj: pd.DataFrame) -> pd.DataFrame:
    """Relabel `most_recent_team` from `overrides.team`. Adds `team_changed`
    (bool) and `team_source` ('override' | 'model')."""
    proj = proj.copy()
    team, _, _ = load_overrides()
    mapped = proj["name"].map(normalize_name).map(team)
    proj["team_changed"] = mapped.notna() & mapped.ne(proj["most_recent_team"])
    proj["team_source"] = pd.Series("model", index=proj.index).mask(mapped.notna(), "override")
    proj["most_recent_team"] = mapped.fillna(proj["most_recent_team"])
    return proj


def drop_unavailable(board: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove players on the OUT list (any `source`). Returns `(board, dropped_names)`."""
    _, out, _ = load_overrides()
    if not out:
        return board, []
    mask = board["name"].map(normalize_name).isin(out)
    dropped = sorted(board.loc[mask, "name"].unique().tolist())
    return board.loc[~mask].copy(), dropped


def apply_adp_overrides(board: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Force `adp` for players whose real draft-room value has moved off the cached
    crowd ADP (news the feed hasn't caught). Runs before the market blend so the
    corrected ADP flows into `market_vbd`. Returns `(board, changed_names)`."""
    _, _, adp = load_overrides()
    if not adp:
        return board, []
    board = board.copy()
    norm = board["name"].map(normalize_name)
    hit = norm.isin(adp)
    board.loc[hit, "adp"] = norm[hit].map(adp)
    return board, sorted(board.loc[hit, "name"].unique().tolist())
