"""Engineer matchup features from the SQLite feature views.

Input  : data/db/nba.db, view `team_game_features` (one row per team per game,
         built by clean.sql + features.sql: dedup, home/away, player-availability)
Output : data/processed/features.csv (one row per game = one matchup)

Each output row describes a single game with each team's form measured strictly
*before* that game (no leakage, Bug B07): rolling 10-game averages, rest days,
player-top5-availability, and a home/away framing. Target is HOME_WIN. A SPLIT
column assigns each game to train/val/test by season so the modeling step never
splits randomly.

Run:
    sqlite3 data/db/nba.db < src/data/clean.sql
    sqlite3 data/db/nba.db < src/data/features.sql
    python src/data/features.py
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/db/nba.db")
FEATURES_PATH = Path("data/processed/features.csv")

# Box-score stats we average over a rolling window per team.
ROLLING_STATS = ["PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS"]
ROLLING_WINDOW = 10

# Season-based split (Bug B07: never split randomly across games). SEASON is the
# starting year, e.g. 2021 == the 2021-22 season.
TRAIN_SEASONS = range(2015, 2022)  # 2015-16 .. 2021-22
VAL_SEASONS = {2022}               # 2022-23
TEST_SEASONS = {2023, 2024}        # 2023-24 .. 2024-25

logger = logging.getLogger(__name__)


def load_clean(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the team_game_features view (games_clean + player availability).

    Raises FileNotFoundError if the database is missing, sqlite3.OperationalError
    if the view hasn't been created yet (run clean.sql then features.sql first).
    """
    logger.info("Loading team_game_features from %s", db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Expected database at {db_path}; run load_db.py first.")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM team_game_features", conn)
    df = df.rename(columns={c: c.upper() for c in df.columns})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add REST_DAYS: days since this team's previous game in the same season.

    Rest is physical, so it spans game types within a season. The first game of
    a team's season has no prior game and is left as NaN (dropped downstream).
    """
    df = df.copy()
    prev_date = df.groupby(["TEAM_ABBREVIATION", "SEASON"])["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - prev_date).dt.days
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ROLL_<stat>: mean of the prior ROLLING_WINDOW games for that team.

    Computed per (SEASON, SEASON_TYPE, team) so playoff and regular-season form
    don't bleed together (Bug B02), and shifted by one game so the current game
    is never included (Bug B07). Requires a full window; partial windows are NaN
    and dropped later (Bug B04).
    """
    df = df.copy()
    grp = df.groupby(["SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION"])
    for stat in ROLLING_STATS:
        df[f"ROLL_{stat}"] = grp[stat].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        )
    return df


def build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot two team-rows per game into one matchup row (home vs away)."""
    # TOP5_AVAIL_PCT is not shifted like the rolling stats: it already describes
    # only this game (were the team's rotation players available tonight), so
    # using it as-is is not leakage the way a same-game box score stat would be.
    feature_cols = ["REST_DAYS", "TOP5_AVAIL_PCT"] + [f"ROLL_{s}" for s in ROLLING_STATS]
    keep = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION", "WL"] + feature_cols

    home = df[df["IS_HOME"] == 1][keep].copy()
    away = df[df["IS_HOME"] == 0][keep].copy()

    home = home.rename(columns={"TEAM_ABBREVIATION": "HOME_TEAM"})
    away = away.rename(columns={"TEAM_ABBREVIATION": "AWAY_TEAM"})
    home = home.rename(columns={c: f"HOME_{c}" for c in feature_cols})
    away = away.rename(columns={c: f"AWAY_{c}" for c in feature_cols})

    # Game-level fields come from the home row; merge away features on GAME_ID.
    away_keep = ["GAME_ID", "AWAY_TEAM"] + [f"AWAY_{c}" for c in feature_cols]
    matchups = home.merge(away[away_keep], on="GAME_ID", how="inner")

    matchups["HOME_WIN"] = (matchups["WL"] == "W").astype(int)
    matchups = matchups.drop(columns=["WL"])
    return matchups


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add SPLIT (train/val/test) by season; drop seasons outside all splits."""
    df = df.copy()

    def which(season: int) -> str:
        if season in TRAIN_SEASONS:
            return "train"
        if season in VAL_SEASONS:
            return "val"
        if season in TEST_SEASONS:
            return "test"
        return "none"

    df["SPLIT"] = df["SEASON"].map(which)
    dropped = (df["SPLIT"] == "none").sum()
    if dropped:
        logger.warning("Dropping %d games from seasons outside train/val/test", dropped)
    return df[df["SPLIT"] != "none"].copy()


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline on cleaned game rows."""
    df = add_rest_days(df)
    df = add_rolling_features(df)
    matchups = build_matchups(df)

    feature_cols = [
        c for c in matchups.columns
        if c.startswith(("HOME_ROLL", "AWAY_ROLL", "HOME_REST", "AWAY_REST", "HOME_TOP5", "AWAY_TOP5"))
    ]
    before = len(matchups)
    matchups = matchups.dropna(subset=feature_cols).copy()  # B04: incomplete windows
    logger.info("Dropped %d matchups with incomplete feature windows", before - len(matchups))

    matchups = assign_split(matchups)
    matchups = matchups.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return matchups


def verify(df: pd.DataFrame) -> None:
    """Sanity-check the engineered features and report split counts."""
    feature_cols = [c for c in df.columns if c.startswith(("HOME_", "AWAY_")) and c not in ("HOME_TEAM", "AWAY_TEAM")]
    assert df[feature_cols].isna().sum().sum() == 0, "NaNs remain in feature columns"
    assert df["GAME_ID"].is_unique, "duplicate games in feature matrix"
    assert set(df["HOME_WIN"].unique()) <= {0, 1}, "HOME_WIN not binary"

    counts = df["SPLIT"].value_counts().to_dict()
    logger.info("Split counts: %s", {k: counts.get(k, 0) for k in ("train", "val", "test")})
    logger.info("Home win rate: %.3f over %d games", df["HOME_WIN"].mean(), len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    features = engineer(load_clean())
    verify(features)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    logger.info("Wrote %d rows, %d columns to %s", len(features), features.shape[1], FEATURES_PATH)


if __name__ == "__main__":
    main()
