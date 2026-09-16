-- Table definitions for data/db/nba.db.
--
-- `games` and `player_boxscores` hold the raw rows loaded from CSV, unchanged
-- (load_db.py just tags each row with its SOURCE file). All cleaning and
-- feature logic lives in clean.sql / features.sql, not here.

DROP TABLE IF EXISTS games;
CREATE TABLE games (
    season_id           TEXT,
    team_id              INTEGER,
    team_abbreviation    TEXT,
    team_name            TEXT,
    game_id              TEXT,
    game_date            TEXT,
    matchup              TEXT,
    wl                   TEXT,
    min                  INTEGER,
    pts                  INTEGER,
    fgm                  INTEGER,
    fga                  INTEGER,
    fg_pct               REAL,
    fg3m                 INTEGER,
    fg3a                 INTEGER,
    fg3_pct              REAL,
    ftm                  INTEGER,
    fta                  INTEGER,
    ft_pct               REAL,
    oreb                 INTEGER,
    dreb                 INTEGER,
    reb                  INTEGER,
    ast                  INTEGER,
    stl                  INTEGER,
    blk                  INTEGER,
    tov                  INTEGER,
    pf                   INTEGER,
    plus_minus           REAL,
    source               TEXT  -- 'historical' (games_raw.csv) or 'current' (games_current.csv)
);

CREATE INDEX idx_games_game_id ON games(game_id);
CREATE INDEX idx_games_team_date ON games(team_abbreviation, game_date);

-- One row per player per game. Loaded from LeagueGameLog(player_or_team='P'),
-- so there is no STARTER/lineup flag (that requires a per-game boxscore call,
-- ~13k+ requests for the historical window — not worth it for v1). Player
-- availability is inferred downstream from MIN: NULL/0 minutes means the
-- player did not play, which approximates "unavailable" without saying why
-- (DNP-rest and DNP-injury look identical). Flagged as approximate, same as
-- bug B05 in PLAN.md.
DROP TABLE IF EXISTS player_boxscores;
CREATE TABLE player_boxscores (
    season_id           TEXT,
    player_id            INTEGER,
    player_name          TEXT,
    team_id              INTEGER,
    team_abbreviation    TEXT,
    team_name            TEXT,
    game_id              TEXT,
    game_date            TEXT,
    matchup              TEXT,
    wl                   TEXT,
    min                  REAL,
    pts                  INTEGER,
    fgm                  INTEGER,
    fga                  INTEGER,
    fg_pct               REAL,
    fg3m                 INTEGER,
    fg3a                 INTEGER,
    fg3_pct              REAL,
    ftm                  INTEGER,
    fta                  INTEGER,
    ft_pct               REAL,
    oreb                 INTEGER,
    dreb                 INTEGER,
    reb                  INTEGER,
    ast                  INTEGER,
    stl                  INTEGER,
    blk                  INTEGER,
    tov                  INTEGER,
    pf                   INTEGER,
    plus_minus           REAL,
    source               TEXT  -- 'historical' or 'current'
);

CREATE INDEX idx_pbox_game_id ON player_boxscores(game_id);
CREATE INDEX idx_pbox_team_game ON player_boxscores(team_abbreviation, game_date);
CREATE INDEX idx_pbox_player ON player_boxscores(player_id, game_date);
