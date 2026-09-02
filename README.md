# Fantasy Football Prediction System

Custom draft assistant (and later, a weekly start/sit tool) for standard H2H
leagues. See `PROJECT-BOOTSTRAP.md` for the full design and `AGENTS.md` for the
architecture invariants.

## Running it

All commands run from the repo root (`~/dev/gsv-fantasy-football`). The wrapper
scripts in `bin/` use the project's virtualenv automatically — you don't need to
activate anything.

```bash
bin/draft  --league sleeper --export          # write the full ranked board to a CSV
bin/draft  --league sleeper --slot 7           # print the board for your draft slot
bin/draft  --league sleeper --slot 7 --watch   # live: auto-refresh as picks happen
bin/draft  --league yahoo                       # Yahoo (offline draft) — static board

bin/mock   --league sleeper --slot 5            # simulate one full draft
bin/mock   --league sleeper --slot 5 --sims 200 # Monte Carlo: 200 simulated drafts

bin/refresh-data   --seasons 2015-2026          # re-pull nflverse into SQLite (~30s)
bin/refresh-models --league sleeper             # retrain + rewrite projections
bin/refresh-models --league yahoo

bin/backtest --league sleeper                   # projected-vs-actual + baselines (~20s)
bin/backtest --league sleeper --season 2024     # one held-out season

bin/yahoo-auth login --manual                   # one-time Yahoo OAuth (needs .env — see .env.example)
bin/yahoo-auth whoami                           # check the cached token still works (do this pre-draft)
bin/yahoo-auth leagues                          # list your Yahoo NFL leagues

bin/test                                         # run the test suite
```

Credentials for external APIs (currently just Yahoo) live in a git-ignored `.env`
at the repo root — `cp .env.example .env` and fill it in. Any script reads them
via `config.get(...)`.

**Before a real draft:** run `bin/refresh-data --seasons 2015-<current year>`
then `bin/refresh-models --league sleeper` (and `--league yahoo`) so the
projections use the latest completed season.

Outputs land in `models/output/`:
- `<league>_projections.csv` — raw model projections (PPG)
- `<league>_board.csv` — full ranked draft board with value-over-replacement (from `--export`)
- `<league>_walkforward.csv` — model accuracy by held-out season

### If a wrapper script won't run

`chmod +x bin/*` once. Or run the underlying command directly:
`.venv/bin/python -m applications.draft_tool --league sleeper --export`

### First-time setup (already done on this machine)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Layout

| dir | what |
|---|---|
| `data/` | nflverse → SQLite (`nfl.db`); `data/README.md` |
| `scoring/` | raw stats → fantasy points, per league scoring; `scoring/README.md` |
| `features/` | model features (as-of/window parameterized); `features/README.md` |
| `models/` | per-position LightGBM, walk-forward validated; `models/README.md` |
| `applications/` | `draft_tool.py`, `mock.py`; `applications/README.md` |
| `specifications/` | sprint plan, league configs, persona Q&A |
