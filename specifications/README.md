# Specifications

This directory holds open questions the project's personas (see `AGENTS.md`) need answered
about *your* local system before certain architecture decisions in `PROJECT-BOOTSTRAP.md` can
firm up. It exists because this is a locally-hosted, single-user project — the right answer to
"SQLite or Parquet," "Streamlit or CLI," "Task Scheduler or something else" depends on hardware,
network, and workflow facts that live with you, not in the codebase.

## How this works

- Each file is one persona's questions, scoped to decisions that persona actually owns.
- Answer inline under each question (or reply in chat and we'll transcribe here) — this
  directory is meant to accumulate answers over time, not be filled out in one sitting.
- Once a question is answered and it settles an architecture decision, the `product-architect`
  persona should fold the resolution back into `AGENTS.md` or `PROJECT-BOOTSTRAP.md` as a
  confirmed decision, and the question here can be marked resolved (or removed if it's fully
  absorbed elsewhere). This directory is a scratchpad for gathering constraints, not the
  permanent record of decisions made.
- `feature-engineer` and `scoring-engineer` have no file here — their work operates on data
  already landed in the DB and has no dependency on local hardware/network/OS specifics. If
  that changes, add a file for them.

## Files

- [`data-engineer.md`](data-engineer.md) — storage, environment, credentials, machine uptime
- [`ml-modeler.md`](ml-modeler.md) — compute specs for training
- [`app-engineer.md`](app-engineer.md) — how/where you'll actually use the tools, incl. draft day
- [`product-architect.md`](product-architect.md) — ops cadence, project lifespan, environment constraints
- [`prereq-checklist.md`](prereq-checklist.md) — concrete setup steps derived from the above, once questions turn into action items
- [`draft-sprint-plan.md`](draft-sprint-plan.md) — active, time-boxed execution plan for the upcoming Sleeper (09-02) and Yahoo (TBD) drafts; temporarily supersedes some `AGENTS.md` defaults (no Docker yet, trimmed history pull) for this sprint only
