"""Fetch historical NBA player box scores (2015-16 through 2024-25) into
player_boxscores_raw.csv.

Uses LeagueGameLog(player_or_team_abbreviation='P'), one call per
(season, season_type) — not one call per game or per player — so this stays
cheap: ~10 seasons x 3 season types = 30 calls, same rate-limit pattern as
fetch_games.py (Bug B03). Playin and playoffs didn't exist as separate types
in the earliest seasons; nba_api just returns 0 rows for those, which is fine.

No STARTER/lineup flag: that requires BoxScoreTraditionalV2 called once per
game (~13k+ requests for this window), which isn't worth the API load for v1.
Player availability is inferred downstream from MIN instead (approximate,
same caveat as bug B05 in PLAN.md).

Run:
    python src/data/fetch_players.py
"""

import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.library.parameters import SeasonTypeAllStar

SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

SEASON_TYPES = [
    SeasonTypeAllStar.regular,
    SeasonTypeAllStar.playoffs,
    SeasonTypeAllStar.playin,
]

OUTPUT_PATH = "data/raw/player_boxscores_raw.csv"


def fetch_season_type(season: str, season_type: str) -> pd.DataFrame:
    """Fetch one (season, season_type) of player-game logs."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
    )
    return log.get_data_frames()[0]


def fetch_all(seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    """Fetch all seasons x season types and return them concatenated."""
    frames = []
    for season in seasons:
        for season_type in season_types:
            print(f"Fetching {season} {season_type}...")
            df = fetch_season_type(season, season_type)
            print(f"  {len(df)} rows")
            if len(df):
                frames.append(df)
            time.sleep(1)  # respect rate limits (B03)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    boxscores = fetch_all(SEASONS, SEASON_TYPES)
    boxscores.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Wrote {boxscores.shape[0]} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
