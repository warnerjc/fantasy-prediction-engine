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
| `--no-adp` | model-only board, skip the ADP fetch |
| `--season` | ADP season (default: the projections' target season) |
| `--blend W` | VBD = `W·model + (1−W)·ADP-implied` (default 0.7; `1.0` = pure model) |
| `--export` | write the full ranked board to `models/output/<league>_board.csv` (the static list) |
| `--replay --draft <id>` | fast-forward a completed Sleeper draft through the live view (dry-run of the `--watch` path) |

Auction drafts are **not** supported — different logic (budget pacing, not
positional scarcity). `run()` is where that would branch.

### What it does

1. **Value over replacement (VBD).** `roster.py` parses the league's
   `roster_positions` into dedicated starter slots + flex slots (with an
   eligibility→position split), then a per-position replacement rank:
   `teams × (starters + flex share) × baseline_mult`. `baseline_mult` pushes
   QB/TE deeper (1.4 / 1.15) because "last starter" overvalues positions that
   stream cheaply off waivers. Model VBD is
   `projected_season_points − replacement_points[position]`.
   **Then it's blended with the market** (`_blend_market`): an isotonic curve maps
   ADP → VBD (fit on RB/WR/K/DEF only — the positions whose model VBD we trust),
   and the final VBD is `W·model + (1−W)·market`. `W` is `--blend` (default 0.7),
   **overridden per position by `_BLEND_BY_POS`** — QB `0.35`, TE `0.5`, because
   the model's year-over-year rank for those is near-noise and it rates efficient
   veterans (Stafford, McBride) ~2 rounds ahead of ADP. A QB/TE the model likes
   that has no ADP at all (market isn't drafting it) is pinned to the bottom of
   the curve, not floated on pure model value. `--blend 1.0` turns the whole blend
   off. Tune `_BASELINE_MULT` in `roster.py` too.
2. **Live draft state.** `sleeper.py` polls `/draft/<id>/picks`; drafted players
   drop off the board. Yahoo `236625` is an offline draft with no pollable state
   — the static board is the tool.
3. **Tiers.** Within a position, a projected-PPG gap over a per-position
   threshold starts a new tier — the "draft any of these, then there's a cliff"
   signal.
4. **Your next pick.** With `--slot`, computes your snake pick numbers and, by
   assuming picks come off the top of the board, splits the board into "likely
   gone" vs "likely there for you."
5. **Rookies / unprojected players.** The model can't project anyone with no
   prior NFL season. `adp.py` pulls crowd ADP (fantasyfootballcalculator.com,
   public API, disk-cached with a TTL), and `board.py` places every unmatched
   top-180 ADP player on the board at a VBD **imputed from ADP** — isotonic
   regression on the players who have both a projection and an ADP. Those rows
   are tagged `(adp)` with no `proj_ppg`, and also listed in their own section.
   The `adp` column is shown for *every* player, so model-vs-market gaps (value
   or reach) are visible at a glance. Sleeper ids for these rows come from the
   cached Sleeper player directory, so they still drop off when drafted.
6. **2026 team / availability / ADP corrections** (`roster_2026.py`). The model
   labels players with their 2025 team, can't see offseason moves / cuts /
   suspensions, and blends against a cached crowd ADP that lags breaking news.
   `specifications/league-configs/roster-2026-overrides.json` is the
   hand-maintained fix: `team` relabels `most_recent_team` (adds a `team_source`
   column: `override` / `model` / `adp`), `out` drops a player, `adp` forces a
   player's ADP so a news-driven draft-stock move (a suspension, say) flows into
   the market blend. Seeded from a scrape of `nfl.com/sitemap/html/rosters/2026`
   cross-checked vs nflverse + the user's Sleeper export — nflverse's `roster_2026`
   release alone is **not** trusted (at cut time it had Pacheco on DET, Fields on
   KC, and dropped rostered vets). Edit the JSON up to draft day; revisit the
   automated path once nflverse firms up (early September).

### Consuming the prediction shape

Projections are `{mean, p10, p50, p90}` (v1 fills only `mean` → `proj_ppg`). The
board shows a single value and no floor/ceiling. When v2 populates real
quantiles, `proj_p10`/`proj_p90` columns already ride through
`models/output/*.csv` — the board can show a range without a rewrite.

## mock.py — simulated / Monte-Carlo drafts

```
python -m applications.mock --league sleeper --slot 5            # one full draft
python -m applications.mock --league sleeper --slot 5 --sims 200 # MC summary
```

Runs a full snake draft locally: the other seats pick ~by ADP (softmax over
`-ADP / opp_temp`, positions killed at their roster cap), our seat picks off the
live `DraftBoard` (highest VBD, with a roster-fit adjustment and a hard "fill
K/DEF/starters before you run out of picks" rule). It exercises the **same board
code path as `--watch`** without a draft room.

- one run → prints our picks round-by-round + final roster + total VBD captured
- `--sims N` → most-common pick per round, VBD-captured distribution, the round
  you typically land your first QB/RB/WR/TE

  After the `_BLEND_BY_POS` tuning (Sun 08-30), the sleeper sim lands its first QB
  round 6-7 and first TE round 3 (was QB round 4, two QBs by round 5). Yahoo stays
  RB-heavy with a late first WR — correct for a 0.25-PPR 10-team league, not a bug.

The sim drafter is greedy-VBD with no roster-need logic, not a real draft AI — its
output is a **sanity check on the board and a read on where the model/blend
leans**, not a strategy to copy. It still doubles up at TE (McBride *and* Bowers)
because nothing tells it "I already have a starter there."

### Known gaps

- **Rookie values are pure ADP**, not a model view — imputed from where the crowd
  drafts them. Fine for slotting; no independent signal.
- **VBD baselines are heuristic** (`_BASELINE_MULT` / flex splits in `roster.py`).
  Easy to adjust after a mock draft.
- ADP format is the closest FFC preset to each league (`half-ppr` / `ppr`), not an
  exact scoring match.
- Auction, and the weekly start/sit tool, are later.
