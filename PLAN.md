# NBA Game Predictor — Project Plan

## Table of Contents

1. [Project Scope](#1-project-scope)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Tech Stack](#3-tech-stack)
4. [Data Dictionary](#4-data-dictionary)
5. [Milestone Roadmap](#5-milestone-roadmap)
6. [Implementation Guide](#6-implementation-guide)
7. [Known Bugs & Gotchas](#7-known-bugs--gotchas)
8. [Bug Tracking Protocol](#8-bug-tracking-protocol)
9. [Feature Backlog](#9-feature-backlog)
10. [Code Standards](#10-code-standards)
11. [Evaluation Criteria](#11-evaluation-criteria)
12. [Deployment](#12-deployment)
13. [Resuming This Project](#13-resuming-this-project)

---

## 1. Project Scope

### What This Is

A game-level NBA win probability predictor. Given two teams playing tonight, the model outputs a calibrated win probability for each team. Results are displayed on a deployed dashboard.

Two learning goals sit alongside the modeling goal: (1) apply real ML concepts
(leakage-safe splits, calibration, interpretability) and be able to defend every
data/modeling decision in an interview, and (2) practice SQL — raw SQL, not an
ORM — as the storage/query layer feeding the pipeline.

### What This Is Not

- Not a real-time system. Data is fetched manually by running a script.
- Not a betting tool. Model will not beat Vegas lines — that is expected and documented.
- Not a championship or series simulator. Predicts individual game outcomes only.
- Not an automated pipeline. No schedulers, no cron jobs, no webhooks.
- Not an ORM project. SQL is written by hand (`sqlite3`) so every query is
  something you can explain line by line — SQLAlchemy is deliberately not used.

### Explicit Constraints

- Data refresh: manual. Run `fetch_games.py` before each prediction session.
- Model retraining: manual. Run `train.py` at the start of each new season.
- Infrastructure: minimal. SQLite for storage, flat file model serialization, single FastAPI endpoint.
- SQL: raw `sqlite3`, no ORM. If asked "why not an ORM," the honest answer is:
  this project is small enough that an ORM would hide the SQL you're here to
  practice, and there's no team of developers to protect from raw queries.
- Target users: portfolio reviewers and yourself.

### Success Criteria

- [ ] Model produces calibrated win probabilities (Brier score < 0.25)
- [ ] Calibration curve shows probabilities are trustworthy
- [ ] Backtest against Vegas lines documented honestly
- [ ] Dashboard deployed and publicly accessible
- [ ] README explains methodology, limitations, and results clearly

---

## 2. Architecture & File Structure

```
nba-game-predictor/
│
├── data/
│   ├── raw/
│   │   ├── games_raw.csv          # Historical team-game rows 2015-16 to 2024-25
│   │   ├── games_current.csv      # Current season 2025-26 (prediction input)
│   │   └── player_boxscores_raw.csv  # Historical player-game box scores
│   ├── db/
│   │   └── nba.db                 # SQLite database — raw + cleaned tables live here
│   └── processed/
│       └── features.csv           # Engineered features ready for modeling (exported from SQL)
│
├── models/
│   ├── xgboost_model.pkl          # Serialized trained model
│   └── calibrator.pkl             # Serialized calibration layer (isotonic regression)
│
├── notebooks/
│   └── exploration.ipynb          # EDA, calibration curves, SHAP plots (NOT production code)
│
├── src/
│   ├── data/
│   │   ├── fetch_games.py         # Pulls team-game box scores from nba_api
│   │   ├── fetch_players.py       # Pulls player-game box scores from nba_api
│   │   ├── schema.sql             # CREATE TABLE statements for nba.db
│   │   ├── load_db.py             # Loads raw CSVs into SQLite tables (no ORM)
│   │   ├── clean.sql              # SQL: dedupe games, normalize, add home/away flag
│   │   └── features.sql           # SQL: rolling averages, rest days, player aggregates,
│   │                               #      joined into one row per game, exported to features.csv
│   │
│   ├── model/
│   │   ├── train.py               # Trains XGBoost model + calibrator
│   │   ├── evaluate.py            # Brier score, calibration curves, Vegas backtest
│   │   └── predict.py             # Loads model + returns win probability for a matchup
│   │
│   └── api/
│       └── main.py                # FastAPI app with /predict endpoint
│
├── frontend/                      # Next.js dashboard
│   ├── pages/
│   │   └── index.js               # Main dashboard page
│   └── components/
│       ├── GameCard.js            # Shows matchup + win probability
│       └── CalibrationChart.js    # Calibration curve visualization
│
├── tests/
│   ├── test_load_db.py            # Unit tests for CSV → SQLite loading
│   ├── test_features.py           # Unit tests for feature engineering (query output shape)
│   └── test_predict.py            # Unit tests for prediction output shape/range
│
├── docs/
│   └── methodology.md             # Model decisions, evaluation results, limitations
│
├── PLAN.md                        # This file
├── README.md                      # Public-facing project summary
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template (never commit .env)
└── .gitignore                     # Excludes data/ (including nba.db), models/, .env, __pycache__
```

---

## 3. Tech Stack

| Layer            | Tool                                 | Why                                                             |
| ---------------- | ------------------------------------ | --------------------------------------------------------------- |
| Data fetching    | `nba_api`                            | Unofficial but well-maintained Python wrapper for NBA Stats API |
| Storage & query  | `sqlite3` (stdlib, no ORM)           | Real SQL practice — joins/aggregation live in SQL, not pandas. SQLite needs no server, matches this project's minimal-infra rule. |
| Data processing  | `pandas`                             | Feature engineering on top of SQL query results — standard for tabular data manipulation |
| Modeling         | `xgboost`                            | Fast, interpretable, industry-standard for tabular data         |
| Calibration      | `scikit-learn` (isotonic regression) | Corrects probability output to be statistically trustworthy     |
| Interpretability | `shap`                               | Explains which features drive each prediction                   |
| API serving      | `FastAPI`                            | Lightweight, async, auto-generates docs at /docs                |
| Frontend         | `Next.js` + `Tailwind`               | You already know this stack                                     |
| Deployment       | `Railway` or `Render`                | Free tier sufficient for this project                           |
| Version control  | `git` + GitHub                       | Required. Commit after every meaningful milestone.              |

---

## 4. Data Dictionary

### games_raw.csv (from nba_api)

| Column                                                | Type  | Description                                    |
| ----------------------------------------------------- | ----- | ---------------------------------------------- |
| SEASON_ID                                             | str   | Prefixed: `2` = regular season, `4` = playoffs |
| TEAM_ID                                               | int   | NBA's internal team identifier                 |
| TEAM_ABBREVIATION                                     | str   | e.g. BOS, LAL                                  |
| GAME_ID                                               | str   | Unique game identifier                         |
| GAME_DATE                                             | str   | Format: YYYY-MM-DD                             |
| MATCHUP                                               | str   | e.g. "BOS vs. DAL" or "BOS @ DAL"              |
| WL                                                    | str   | W or L                                         |
| PTS, REB, AST, STL, BLK, TOV, FG_PCT, FG3_PCT, FT_PCT | float | Box score stats                                |
| PLUS_MINUS                                            | float | Point differential for this team in this game  |

### Key Notes on Raw Data

- Each game appears **twice** — once per team. You must deduplicate into one row per game before modeling.
- `SEASON_ID` prefix `4` means playoffs. Filter these out of training or handle separately.
- `MATCHUP` with `vs.` = home team. `@` = away team. Use this to engineer home/away feature.

### player_boxscores_raw.csv (from nba_api)

| Column      | Type  | Description                                      |
| ----------- | ----- | ------------------------------------------------- |
| GAME_ID     | str   | Joins to games_raw.GAME_ID                        |
| TEAM_ID     | int   | Joins to games_raw.TEAM_ID                         |
| PLAYER_ID   | int   | NBA's internal player identifier                  |
| PLAYER_NAME | str   | Player full name                                  |
| MIN         | float | Minutes played (NaN/0 if did not play — DNP)      |
| PTS, REB, AST | float | Box score stats for this player in this game    |
| STARTER     | bool  | Whether this player started the game (from lineup) |

### SQLite Schema (`data/db/nba.db`)

| Table              | Grain                  | Source                        |
| ------------------ | ----------------------- | ------------------------------ |
| `games`             | one row per team-game   | loaded from `games_raw.csv` + `games_current.csv` |
| `player_boxscores`  | one row per player-game | loaded from `player_boxscores_raw.csv` |

`clean.sql` and `features.sql` query these two tables — e.g. joining
`player_boxscores` back to `games` to compute "were this team's top-5
minutes players (by season-to-date average) available for this game."
This is the concrete SQL story: real joins and aggregations, not just a
`SELECT *`.

---

## 5. Milestone Roadmap

### Week 1 — Data Pipeline & SQL Setup ✅ DONE

- [x] Install `nba_api`, `pandas`
- [x] Fetch historical games 2015-16 to 2024-25 → `games_raw.csv`
- [x] Fetch current season 2025-26 → `games_current.csv` (rerun before next prediction session — see B13)
- [x] Write `fetch_players.py` / `fetch_players_current.py` — pull player-game box scores from `nba_api` via `LeagueGameLog(player_or_team='P')` → `player_boxscores_raw.csv` / `player_boxscores_current.csv`. No `STARTER` flag (see B14) — availability is inferred from `MIN` instead.
- [x] Write `schema.sql` — `games` and `player_boxscores` table definitions
- [x] Write `load_db.py` — loads the raw CSVs into `data/db/nba.db` using `sqlite3` (no ORM)
- [x] Rewrite cleaning logic as `clean.sql` — dedupe games, normalize columns, add home/away flag (was `clean.py`; the CSV-only version is superseded by the SQL view `games_clean`)
- [x] Moved all data scripts into `src/data/` per the architecture in §2
- [ ] Verify: no duplicate GAME_IDs, WL column is complete, GAME_DATE parses correctly

**Exit criteria:** `nba.db` exists with populated `games` and `player_boxscores` tables. `clean.sql` runs and produces a deduplicated, one-row-per-team-game view with no nulls in key columns.

### Week 2 — Feature Engineering ✅ DONE

- [x] Write `features.sql` — the SQL side of feature engineering:
  - `player_boxscores_clean` joins player box scores to `games_clean` (on game_id + team_id, stable across relocations — Bug B11)
  - `player_prior_avg_min` / `team_top5_players`: window functions ranking each team's rotation players by season-to-date average minutes, strictly before the current game (no leakage)
  - `team_top5_availability` / `team_game_features`: what fraction of a team's top-5 rotation players actually played this game (`top5_avail_pct`) — NULL for a team's first few games each season (no prior history yet), same shape as Bug B04
- [x] Write `features.py` — pandas side, reading `team_game_features` from `nba.db`:
  - Rolling 10-game averages per team (PTS, REB, AST, FG_PCT, PLUS_MINUS)
  - Rest days feature (days since last game per team)
  - `TOP5_AVAIL_PCT` folded in as-is (not shifted — it already describes only the current game, so using it directly isn't leakage the way a same-game box score stat would be)
  - Home/away binary flag
  - Season-based train/val/test split:
    - Train: 2015-16 to 2021-22
    - Val: 2022-23
    - Test: 2023-24 to 2024-25
- [x] **NEVER random split across games — this causes data leakage**

**Exit criteria:** `features.csv` exists, includes player-availability features. Train/val/test row counts verified. No future data leaks into training set. You can explain in plain English what each SQL query does and why it's in SQL rather than pandas (or vice versa).

**Result:** 10,616 games (2,450 dropped for incomplete rolling/availability windows, 1,090 dropped for seasons outside train/val/test). Split: 7,339 train / 1,094 val / 2,183 test. Home win rate 56.6% — consistent with the commonly cited ~58% home-court baseline, a reasonable sanity check that cleaning/pivoting didn't introduce bias.

### Week 3 — Baseline Model

- [ ] Write `train.py`
- [ ] Logistic regression baseline first (sets the floor)
- [ ] XGBoost model second
- [ ] Output: win probability per game, not just W/L classification
- [ ] Write `evaluate.py`
- [ ] Accuracy, Brier score, log loss on val set
- [ ] Compare vs naive baseline (always predict home team wins ~58% of the time)

**Exit criteria:** XGBoost Brier score < 0.25 on val set. Model saved to `models/xgboost_model.pkl`.

### Week 4 — Calibration & Evaluation

- [ ] Plot calibration curve (reliability diagram)
- [ ] Apply isotonic regression calibration
- [ ] Save calibrator to `models/calibrator.pkl`
- [ ] Backtest on 2023-24 playoff games vs Vegas lines
- [ ] Document results honestly in `docs/methodology.md` — including where model fails

**Exit criteria:** Calibration curve plotted and saved. Backtest results written up. Model limitations documented.

### Week 5 — SHAP Feature Importance

- [ ] Install `shap`
- [ ] Generate SHAP summary plot — which features matter most?
- [ ] Generate SHAP waterfall plot for one example prediction
- [ ] Write up 3-sentence explanation of what the model learned in `docs/methodology.md`

**Exit criteria:** Can explain in plain English why the model predicted a specific game outcome.

### Week 6 — FastAPI Serving Layer

- [ ] Write `src/api/main.py`
- [ ] POST `/predict` endpoint: takes two team abbreviations + game date, returns win probabilities
- [ ] GET `/health` endpoint: returns last data refresh date + model version
- [ ] Load model and calibrator from disk at startup
- [ ] Add "last updated" timestamp to every response
- [ ] Test locally with curl or Postman before deploying

**Exit criteria:** `/predict` returns valid probabilities that sum to 1.0. Deployed to Railway or Render.

### Week 7 — Dashboard

- [ ] Next.js project scaffolded in `/frontend`
- [ ] `GameCard` component: shows team names, win probabilities, confidence
- [ ] `CalibrationChart` component: shows calibration curve
- [ ] "Last updated" timestamp visible on every page
- [ ] Connects to deployed FastAPI endpoint

**Exit criteria:** Dashboard publicly accessible. Shows current playoff matchup with win probabilities.

### Week 8 — Polish & Documentation

- [ ] Write `README.md`: what it is, how it works, results, limitations, how to run locally
- [ ] Write `docs/methodology.md`: full model writeup
- [ ] Record 2-minute demo video
- [ ] Clean up all `print` statements → use `logging`
- [ ] Final commit with clean git history

---

## 6. Implementation Guide

### Running the project end-to-end (for any future session or collaborator)

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch data (run manually when you want fresh data)
python src/data/fetch_games.py        # historical team box scores
python src/data/fetch_players.py      # historical player box scores
python src/data/fetch_current.py      # current season

# 3. Load into SQLite and clean
python src/data/load_db.py            # CSVs -> data/db/nba.db
sqlite3 data/db/nba.db < src/data/clean.sql

# 4. Engineer features (SQL then pandas)
sqlite3 data/db/nba.db < src/data/features.sql
python src/data/features.py

# 5. Train model
python src/model/train.py

# 6. Evaluate model
python src/model/evaluate.py

# 7. Run API locally
uvicorn src.api.main:app --reload

# 8. Run frontend locally
cd frontend && npm run dev
```

### How to make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "BOS", "away_team": "OKC", "game_date": "2026-06-06"}'
```

---

## 7. Known Bugs & Gotchas

These are confirmed issues you will hit. Read before starting each phase.

### Data Layer

| #   | Issue                                                            | Impact                                                            | Fix                                                                         |
| --- | ---------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| B01 | Each game appears twice in raw data (once per team)              | Model trains on duplicate data, inflates performance              | Deduplicate by GAME_ID in `clean.py`, keep one row per game                 |
| B02 | `SEASON_ID` prefix `4` = playoffs, `2` = regular season          | Mixing playoff and regular season games confuses rolling averages | Filter by SEASON_ID prefix before computing rolling features                |
| B03 | `nba_api` rate limits — too many requests causes silent failures | Missing seasons in your dataset                                   | `time.sleep(1)` between every API call. Verify row counts after each fetch. |
| B04 | Rolling averages at season start have insufficient history       | First ~10 games of each season have NaN features                  | Drop rows where rolling window is incomplete OR fill with season average    |
| B05 | Player availability from box score is approximate (DNP vs. genuine injury look the same) | Feature is noisy                                | Flag it as approximate in docs. Don't over-rely on it.                      |
| B06 | Current season (2025-26) only has completed games                | Tonight's game won't be in the data                               | Re-run `fetch_current.py` before each prediction session                    |
| B12 | SQLite writes lock the whole DB file (no row-level locking)      | Concurrent `load_db.py` + query scripts can fail with "database is locked" | Run data loading and querying steps sequentially, never in parallel processes |
| B13 | `games_current.csv` is fetched once and goes stale (e.g. only had Finals Game 1 for weeks after the series continued) | Feature/prediction input misses recent games | Rerun `fetch_current.py` + `fetch_players_current.py` before each prediction session, not just once |
| B14 | `player_boxscores` has no `STARTER`/lineup flag — `LeagueGameLog` doesn't return one, and getting it needs a boxscore call per game (~13k+ requests) | "Was this team's top player available" can only be approximated from `MIN`, not a real starter/injury signal | Documented scope decision for v1: infer availability from `MIN` (0/NULL = did not play). Revisit if this proves too noisy. |

### Model Layer

| #   | Issue                                       | Impact                                         | Fix                                                               |
| --- | ------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| B07 | Random train/test split causes data leakage | Model looks great in eval, fails in production | Always split by season, never by random game index                |
| B08 | XGBoost outputs uncalibrated probabilities  | Probabilities are systematically overconfident | Apply isotonic regression calibration layer before serving        |
| B09 | Class imbalance: home team wins ~58%        | Model biased toward predicting home wins       | Check class balance. Add `scale_pos_weight` to XGBoost if needed. |

### API Layer

| #   | Issue                                          | Impact                 | Fix                                                                 |
| --- | ---------------------------------------------- | ---------------------- | ------------------------------------------------------------------- |
| B10 | Model loaded fresh on every request            | Slow response times    | Load model at app startup using FastAPI lifespan events             |
| B11 | Team abbreviations inconsistent across seasons | KeyError on prediction | Normalize all abbreviations in `clean.py`. Maintain a mapping dict. |

---

## 8. Bug Tracking Protocol

When you find a new bug during development:

1. **Add it to the table in Section 7** with a number, description, impact, and fix.
2. **Mark it with status**: `OPEN`, `IN PROGRESS`, or `FIXED`.
3. **Never delete fixed bugs** — keep them as a record. Add a `Fix commit:` note.
4. **If a bug blocks you for more than 30 minutes**, write down exactly what you tried before asking for help. This forces clarity.

Template for new bugs:

```
| B## | [What breaks] | [What goes wrong] | [How to fix it] | OPEN |
```

---

## 9. Feature Backlog

These are ideas that are explicitly OUT OF SCOPE for v1. Do not build these until Weeks 1-8 are complete and deployed.

| Feature                                  | Why it's interesting                    | Why it's deferred                                                 |
| ----------------------------------------- | --------------------------------------- | ----------------------------------------------------------------- |
| Automated data refresh (cron job)         | Removes manual step                     | Adds infrastructure complexity, not needed for portfolio          |
| LSTM / sequence model                     | Captures game momentum                  | 60-hour budget doesn't support it without sacrificing calibration |
| Real injury report integration (official injury status, not just DNP-from-box-score) | Biggest signal in NBA predictions | Data is messy, unstructured, hard to get historically; v1 uses the approximate box-score-derived availability flag instead |
| Player-level modeling (players as model inputs, not team aggregates) | More granular signal | v1 uses player data only as team-level aggregates (availability, minutes); full player-level modeling is a bigger scope jump |
| Series outcome prediction                 | More interesting target                 | Requires game-level model first                                   |
| Monte Carlo playoff simulator             | Distribution output, not point estimate | Week 4+ complexity, build after v1                                |
| User accounts / saved predictions         | Product feature                         | This is a portfolio project, not a product                        |
| Postgres (upgrade from SQLite)            | Closer to a "real" production DB        | SQLite is sufficient for this data volume and needs no server — matches the minimal-infra rule |

---

## 10. Code Standards

### File organization

- One responsibility per file. `clean.sql` cleans. `features.sql`/`features.py` engineer features. They do not overlap.
- No business logic in `main.py` (the API). It calls functions from other modules.
- No hardcoded paths. Use a `config.py` or constants at the top of each file.
- SQL lives in `.sql` files, not as strings embedded in `.py` files — keeps queries readable and diffable on their own.

### Python style

- Type hints on all function signatures
- Docstring on every function explaining what it takes and what it returns
- `logging` not `print` for anything that runs in production
- Constants in ALL_CAPS at the top of the file

### Example of what good code looks like in this project:

```python
# src/data/load_db.py

import sqlite3
import pandas as pd
import logging
from pathlib import Path

RAW_GAMES_PATH = Path("data/raw/games_raw.csv")
DB_PATH = Path("data/db/nba.db")

logger = logging.getLogger(__name__)

def load_games_csv(path: Path = RAW_GAMES_PATH) -> pd.DataFrame:
    """Load raw team-game box scores from CSV. Raises FileNotFoundError if missing."""
    logger.info(f"Loading raw games from {path}")
    return pd.read_csv(path)

def write_games_table(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Write raw team-game rows into the `games` table, replacing any existing data.

    Uses sqlite3 directly (no ORM) — pandas' to_sql is a thin wrapper over the
    same driver, kept here rather than in features.py so all DB writes live in
    one place.
    """
    with sqlite3.connect(db_path) as conn:
        df.to_sql("games", conn, if_exists="replace", index=False)
```

```sql
-- src/data/clean.sql
-- Raw data has two rows per game (one per team). Keep the home-team row only,
-- identified by 'vs.' in MATCHUP, and add an explicit IS_HOME flag.

CREATE VIEW IF NOT EXISTS games_clean AS
SELECT
    *,
    CASE WHEN MATCHUP LIKE '% vs. %' THEN 1 ELSE 0 END AS IS_HOME
FROM games
WHERE MATCHUP LIKE '% vs. %';
```

### Git discipline

- Commit after every milestone, not every file save
- Commit messages: `feat: add rolling averages to features.py` not `update`
- Never commit: `.env`, `data/`, `models/`, `__pycache__/`
- Branch for experiments: `git checkout -b experiment/lstm` — never experiment on `main`

---

## 11. Evaluation Criteria

### How to know if your model is good

| Metric            | What it measures                                   | Target                          | How to compute                                  |
| ----------------- | -------------------------------------------------- | ------------------------------- | ----------------------------------------------- |
| Accuracy          | % of games predicted correctly                     | > 58% (beat home-team baseline) | `sklearn.metrics.accuracy_score`                |
| Brier Score       | Calibration quality (lower = better)               | < 0.25                          | `sklearn.metrics.brier_score_loss`              |
| Log Loss          | Confidence of correct predictions                  | < 0.65                          | `sklearn.metrics.log_loss`                      |
| Calibration curve | Are 70% predictions right 70% of the time?         | Diagonal line                   | `sklearn.calibration.calibration_curve`         |
| Vegas comparison  | How often does your model agree with the favorite? | Document honestly               | Compare your predicted winner vs Vegas favorite |

### What to do when results are bad

- Brier score > 0.25: Check for data leakage first. Then check feature quality.
- Calibration curve is off: Apply isotonic regression. If still off, check class balance.
- Worse than home-team baseline: Your features have no signal. Go back to feature engineering.

---

## 12. Deployment

### API (FastAPI)

- Deploy to Railway: connect GitHub repo, set start command to `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Set environment variables in Railway dashboard, never in code
- Test `/health` endpoint after every deploy

### Frontend (Next.js)

- Deploy to Vercel: connect GitHub repo, set `frontend/` as root directory
- Set `NEXT_PUBLIC_API_URL` environment variable to your Railway API URL

### After deploying

- Run a real prediction through the live URL
- Check "last updated" timestamp is showing correctly
- Check calibration chart renders

---

## 13. Resuming This Project

If you're coming back after a break, do this in order:

1. Read this file top to bottom — 10 minutes
2. Check which milestone you're on in Section 5
3. Check Section 7 for any OPEN bugs relevant to your current phase
4. Run `git log --oneline -10` to see what you last committed
5. Run the pipeline from your current phase forward to make sure nothing is broken
6. Only then write new code

---

_Last updated: September 2026_
_Current status: Week 1 (SQL storage layer) and Week 2 (feature engineering, incl. player-availability) complete. `features.csv` built: 10,616 games, 22 columns._
_Next action: Week 3 — write `train.py` (logistic regression baseline, then XGBoost) and `evaluate.py`._
