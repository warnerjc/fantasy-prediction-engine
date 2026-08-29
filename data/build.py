"""Build / refresh the SQLite store.

    python -m data.build --seasons 2015-2024        # full (re)build
    python -m data.build --seasons 2025             # weekly in-season refresh

Every step upserts on its table's primary key, so re-running is safe and picks up
nflverse stat corrections. Sleeper/Yahoo league data is pulled by the application
layer at draft time, not here.
"""

from __future__ import annotations

import argparse
import time

from . import nflverse
from .db import DB_PATH, PRIMARY_KEYS, connect, upsert


def parse_seasons(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"no seasons parsed from {spec!r}")
    return sorted(out)


def build(seasons: list[int], db_path=DB_PATH) -> dict[str, int]:
    conn = connect(db_path)
    counts: dict[str, int] = {}
    try:
        def step(name: str, df):
            t = time.perf_counter()
            n = upsert(conn, name, df)
            counts[name] = n
            print(f"  {name:<18} {n:>7,} rows  ({time.perf_counter() - t:.1f}s)")

        print(f"Seasons {seasons[0]}-{seasons[-1]} -> {db_path}")

        xwalk = nflverse.player_ids()
        step("player_ids", xwalk)
        step("player_week_stats", nflverse.weekly_player_stats(seasons))
        step("snap_counts", nflverse.snap_counts(seasons, crosswalk=xwalk))
        step("injuries", nflverse.injuries(seasons))
        step("seasonal_rosters", nflverse.seasonal_rosters(seasons))

        sched = nflverse.schedules(seasons)
        step("schedules", sched)
        step("team_week", nflverse.team_week(sched))

        print("  pulling play-by-play (large)...")
        pbp = nflverse.play_by_play(seasons)
        step("kicking_stats", nflverse.kicking_stats(pbp))
        step("team_defense_stats", nflverse.team_defense_stats(pbp))

        _sanity(conn)
        return counts
    finally:
        conn.close()


def _sanity(conn) -> None:
    pws = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT player_id), MIN(season), MAX(season) FROM player_week_stats"
    ).fetchone()
    print(f"  player_week_stats: {pws[0]:,} rows, {pws[1]:,} players, {pws[2]}-{pws[3]}")
    unmatched = conn.execute(
        "SELECT COUNT(*) FROM snap_counts WHERE gsis_id IS NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM snap_counts").fetchone()[0]
    if total:
        print(f"  snap_counts without gsis_id: {unmatched:,} / {total:,} "
              f"({100 * unmatched / total:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", required=True, type=parse_seasons,
                    help="e.g. '2015-2024' or '2023,2024' or '2025'")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    build(args.seasons, db_path=args.db)
    print("done. tables:", ", ".join(PRIMARY_KEYS))


if __name__ == "__main__":
    main()
