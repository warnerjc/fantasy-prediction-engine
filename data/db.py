"""SQLite store: location, connection, and the idempotent upsert helper.

Refresh semantics: every table has an explicit primary key and is written with
``upsert`` (INSERT OR REPLACE on the PK), so a weekly re-run overwrites revised
rows instead of duplicating them. nflverse revises stat lines for a week or two
after games (stat corrections), so append-only would rot the table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "nfl.db"

# table -> primary key columns. A player with no row for a week (bye / inactive /
# not rostered) is intentionally absent, not a zero row.
PRIMARY_KEYS: dict[str, list[str]] = {
    "player_week_stats": ["player_id", "season", "week", "season_type"],
    # team is in the PK because PFR occasionally reuses one player id for two
    # different people (e.g. two "Jalen Davis"), who then collide on a shared week.
    "snap_counts": ["pfr_player_id", "season", "week", "game_type", "team"],
    "injuries": ["gsis_id", "season", "week", "game_type", "team"],
    "schedules": ["game_id"],
    "team_week": ["season", "week", "game_type", "team"],
    "player_ids": ["gsis_id"],
}


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Create ``table`` if needed and INSERT-OR-REPLACE every row of ``df``.

    The PK comes from ``PRIMARY_KEYS``. Column set is taken from ``df``; a schema
    change (new source column) is applied with ALTER TABLE ADD COLUMN.
    """
    if table not in PRIMARY_KEYS:
        raise KeyError(f"no primary key registered for {table!r}")
    if df.empty:
        return 0

    pk = PRIMARY_KEYS[table]
    missing = [c for c in pk if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: dataframe missing PK columns {missing}")
    if df.duplicated(pk).any():
        dupes = df[df.duplicated(pk, keep=False)][pk]
        raise ValueError(f"{table}: {len(dupes)} rows collide on PK before write:\n{dupes.head()}")

    cols = list(df.columns)

    existing = _table_columns(conn, table)
    if existing is None:
        col_defs = ", ".join(_quote(c) for c in cols)
        pk_def = ", ".join(_quote(c) for c in pk)
        conn.execute(f"CREATE TABLE {_quote(table)} ({col_defs}, PRIMARY KEY ({pk_def}))")
    else:
        for c in cols:
            if c not in existing:
                conn.execute(f"ALTER TABLE {_quote(table)} ADD COLUMN {_quote(c)}")

    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(_quote(c) for c in cols)
    sql = f"INSERT OR REPLACE INTO {_quote(table)} ({col_list}) VALUES ({placeholders})"
    conn.executemany(sql, _rows(df))
    conn.commit()
    return len(df)


def _rows(df: pd.DataFrame):
    """Yield PK-safe tuples: pandas/numpy NA -> None, numpy scalars -> Python."""
    for row in df.itertuples(index=False, name=None):
        clean = []
        for v in row:
            if v is None or v is pd.NaT or (np.ndim(v) == 0 and pd.isna(v)):
                clean.append(None)
            elif isinstance(v, np.generic):
                clean.append(v.item())
            elif isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
                clean.append(v.isoformat())
            else:
                clean.append(v)
        yield tuple(clean)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str] | None:
    rows = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    return {r[1] for r in rows} if rows else None


def read_sql(query: str, conn: sqlite3.Connection | None = None, **kwargs) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect()
    try:
        return pd.read_sql_query(query, conn, **kwargs)
    finally:
        if own:
            conn.close()
