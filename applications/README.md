# /applications — end-user tools

Consumes `/models` projections and `/scoring` as black boxes. Never re-derives a
projection or a point value here.

## draft_tool.py — snake draft assistant

```
python -m models.build --league sleeper          # refresh projections first
python -m applications.draft_tool --league sleeper --slot 7 --watch
python -m applications.draft_tool --league yahoo               # static board, offline draft
```

| flag | |
|---|---|
| `--league sleeper\|yahoo` | which league config + projections to use |
| `--slot N` | your draft position (1 = first overall) → "your next pick" targeting |
| `--watch` | poll live Sleeper draft state every `--interval` s (default 15), re-render on each new pick |
| `--draft <id>` | Sleeper draft id (default: looked up from the league) |

Auction drafts are **not** supported — different logic (budget pacing, not
positional scarcity). `run()` is where that would branch.

### What it does

1. **Value over replacement (VBD).** `roster.py` parses the league's
   `roster_positions` into dedicated starter slots + flex slots (with an
   eligibility→position split), then a per-position replacement rank:
   `teams × (starters + flex share) × baseline_mult`. `baseline_mult` pushes
   QB/TE deeper (1.6 / 1.3) because "last starter" overvalues positions that
   stream cheaply off waivers. A player's board value is
   `projected_season_points − replacement_points[position]` — this is what makes
   an RB and a QB comparable on one list. Tune `_BASELINE_MULT` in `roster.py`
   to taste.
2. **Live draft state.** `sleeper.py` polls `/draft/<id>/picks`; drafted players
   drop off the board. Yahoo `236625` is an offline draft with no pollable state
   — the static board is the tool.
3. **Tiers.** Within a position, a projected-PPG gap over a per-position
   threshold starts a new tier — the "draft any of these, then there's a cliff"
   signal.
4. **Your next pick.** With `--slot`, computes your snake pick numbers and, by
   assuming picks come off the top of the board, splits the board into "likely
   gone" vs "likely there for you."

### Consuming the prediction shape

Projections are `{mean, p10, p50, p90}` (v1 fills only `mean` → `proj_ppg`). The
board shows a single value and no floor/ceiling. When v2 populates real
quantiles, `proj_p10`/`proj_p90` columns already ride through
`models/output/*.csv` — the board can show a range without a rewrite.

### Known gaps

- **Rookies** aren't in the projections (no prior NFL season) — they won't appear
  on the board at all. Cross-check a rookie ADP list for the early rounds.
- **VBD baselines are heuristic** (no ADP data). They're a `roster.py` constant,
  easy to adjust after a mock draft.
- Auction, and the weekly start/sit tool, are later.
