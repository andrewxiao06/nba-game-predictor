"""Load raw CSVs into data/db/nba.db.

Applies schema.sql (drop + recreate tables), then loads each CSV as-is, tagged
with a SOURCE column ('historical' vs 'current') so downstream SQL can tell
them apart without relying on file provenance. No cleaning happens here —
that's clean.sql's job.

Run:
    python src/data/load_db.py
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_PATH = Path("src/data/schema.sql")
DB_PATH = Path("data/db/nba.db")

GAMES_RAW_PATH = Path("data/raw/games_raw.csv")
GAMES_CURRENT_PATH = Path("data/raw/games_current.csv")
PLAYER_BOXSCORES_RAW_PATH = Path("data/raw/player_boxscores_raw.csv")
PLAYER_BOXSCORES_CURRENT_PATH = Path("data/raw/player_boxscores_current.csv")

logger = logging.getLogger(__name__)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate games/player_boxscores per schema.sql."""
    logger.info("Applying schema from %s", SCHEMA_PATH)
    conn.executescript(SCHEMA_PATH.read_text())


def table_columns(table: str, conn: sqlite3.Connection) -> list[str]:
    """Return `table`'s column names, per schema.sql."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def load_csv(path: Path, table: str, source: str, conn: sqlite3.Connection) -> None:
    """Load one raw CSV into `table`, keeping only columns schema.sql defines.

    nba_api endpoints return extra columns (e.g. FANTASY_PTS, VIDEO_AVAILABLE)
    that aren't part of this project's schema — drop them rather than widen
    the table to match every field the API happens to return.
    """
    if not path.exists():
        logger.warning("Skipping %s -> %s (file not found)", path, table)
        return
    df = pd.read_csv(path, dtype={"GAME_ID": str, "SEASON_ID": str})
    df.columns = [c.lower() for c in df.columns]
    df["source"] = source

    wanted = table_columns(table, conn)
    extra = [c for c in df.columns if c not in wanted]
    if extra:
        logger.info("Dropping columns not in %s schema: %s", table, extra)
    df = df[[c for c in wanted if c in df.columns]]

    df.to_sql(table, conn, if_exists="append", index=False)
    logger.info("Loaded %d rows from %s into %s (source=%s)", len(df), path, table, source)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        apply_schema(conn)
        load_csv(GAMES_RAW_PATH, "games", "historical", conn)
        load_csv(GAMES_CURRENT_PATH, "games", "current", conn)
        load_csv(PLAYER_BOXSCORES_RAW_PATH, "player_boxscores", "historical", conn)
        load_csv(PLAYER_BOXSCORES_CURRENT_PATH, "player_boxscores", "current", conn)


if __name__ == "__main__":
    main()
