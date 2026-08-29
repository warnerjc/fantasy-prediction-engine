"""Data-ingestion layer: pulls raw stats/context into the SQLite store.

`player_week_stats` is the single source of truth, at weekly grain (see AGENTS.md).
Season-level numbers are always a GROUP BY on top of it, never a second table.
Adjacent raw tables (`snap_counts`, `injuries`, `schedules`, `player_ids`) support
feature engineering but do not duplicate box-score production.
"""
