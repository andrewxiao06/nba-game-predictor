# CLAUDE.md — NBA Game Predictor

## How I want you to work on this project

This is a **feature/goal-first project now**. I'll pick up the underlying ML
and data-science concepts separately (outside this repo). In here, optimize
for shipping working milestones from PLAN.md efficiently — don't slow down
for in-line teaching.

### 1. Prioritize shipping the roadmap
- Move through PLAN.md's milestones with minimal friction. Default to
  implementing the next reasonable step rather than waiting for me to spell
  it out.
- You can chain multiple related steps together (e.g. write the SQL and the
  script that runs it) instead of stopping after each file.
- Keep explanations brief — a line or two on *why* a non-obvious decision was
  made (e.g. why a split is season-based) is enough. No extended concept
  walkthroughs unless I ask.

### 2. Just write the code
- Don't withhold full implementations to make me work it out — write the
  complete solution.
- If something is genuinely ambiguous (conflicting requirements, a design
  choice with real tradeoffs), make the call and note it, or ask if it's a
  decision only I can make.

### 3. Still respect the project's guardrails
- Keep following PLAN.md's scope, SQL-not-ORM rule, and file structure.
- Don't silently expand scope beyond what's in PLAN.md's roadmap/backlog —
  if you think something should move up, say so, but keep going on the
  current milestone.
- Update PLAN.md/bug tables as you go rather than leaving them stale.

### 4. Pace
- Favor throughput over pausing to check understanding.
- It's still fine to flag something worth knowing, but don't block progress
  on it — a short note is enough.

## Project context
- Full roadmap, bugs, and decisions live in `PLAN.md`. Read it before working.
- Current status: Week 1 (SQL storage) and Week 2 (feature engineering, incl.
  player-availability) complete. `features.csv` built: 10,616 games, 22 cols.
  Next: Week 3, `train.py` (baseline model).
- Scope: single-game winner prediction only. Not a championship/series predictor.
- SQL requirement: raw SQL only (`sqlite3`), no ORM (no SQLAlchemy). Used as the
  storage/query layer — joins and aggregation happen in SQL, feature engineering
  still happens in pandas.
- Player-level data (box-score aggregates: minutes, PTS/REB/AST, star player
  availability) is in scope for v1, not deferred to v2.
