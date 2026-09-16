"""Fetch the current NBA season (2025-26) player box scores into
player_boxscores_current.csv.

Same shape as fetch_players.py, just the current season set and output file.
Run this alongside fetch_current.py before each prediction session (Bug B06).

Run:
    python src/data/fetch_players_current.py
"""

from fetch_players import fetch_all, SEASON_TYPES

SEASONS = ["2025-26"]
OUTPUT_PATH = "data/raw/player_boxscores_current.csv"


def main() -> None:
    boxscores = fetch_all(SEASONS, SEASON_TYPES)
    boxscores.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Wrote {boxscores.shape[0]} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
