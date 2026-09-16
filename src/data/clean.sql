-- Clean/normalize `games` into a view: one row per team per game (two rows
-- per game), competitive game types only, with home/away and season fields
-- added. Dedup to a single matchup row happens later, at the feature-join
-- step in features.sql. This is the SQL replacement for the old clean.py.
--
-- Run:
--     sqlite3 data/db/nba.db < src/data/clean.sql

DROP VIEW IF EXISTS games_clean;
CREATE VIEW games_clean AS
WITH typed AS (
    -- SEASON_ID is a 5-char string: first char is the game-type prefix, last
    -- four are the season's starting year (Bug B02: real data has 6 prefixes,
    -- not just 2/4 as originally assumed).
    SELECT
        *,
        CASE substr(season_id, 1, 1)
            WHEN '1' THEN 'preseason'
            WHEN '2' THEN 'regular_season'
            WHEN '3' THEN 'allstar'
            WHEN '4' THEN 'playoffs'
            WHEN '5' THEN 'playin'
            WHEN '6' THEN 'tournament'
            ELSE 'unknown'
        END AS season_type,
        CAST(substr(season_id, 2) AS INTEGER) AS season
    FROM games
),
filtered AS (
    -- Preseason and all-star games are not competitive; drop them.
    SELECT * FROM typed
    WHERE season_type IN ('regular_season', 'playoffs', 'playin', 'tournament')
),
normalized AS (
    -- Bug B11: team abbreviations drift across seasons. Map historical codes
    -- onto each franchise's current abbreviation so joins never miss.
    SELECT
        *,
        CASE team_abbreviation
            WHEN 'NOH' THEN 'NOP'  -- New Orleans Hornets -> Pelicans
            WHEN 'NJN' THEN 'BKN'  -- New Jersey Nets -> Brooklyn
            WHEN 'NOK' THEN 'NOP'  -- New Orleans/Oklahoma City Hornets -> Pelicans
            WHEN 'SEA' THEN 'OKC'  -- Seattle SuperSonics -> Oklahoma City Thunder
            WHEN 'VAN' THEN 'MEM'  -- Vancouver Grizzlies -> Memphis
            WHEN 'CHH' THEN 'CHA'  -- Charlotte Hornets (original) -> Charlotte
            ELSE team_abbreviation
        END AS team_abbreviation_norm,
        CASE WHEN matchup LIKE '% vs. %' THEN 1 ELSE 0 END AS is_home,
        -- MATCHUP is always "XXX vs. YYY" (home) or "XXX @ YYY" (away), and
        -- team abbreviations are always 3 letters, so the opponent is
        -- everything after the fixed-width separator.
        CASE
            WHEN matchup LIKE '% vs. %' THEN substr(matchup, 9)
            ELSE substr(matchup, 7)
        END AS opponent_raw
    FROM filtered
),
with_opponent AS (
    SELECT
        *,
        CASE opponent_raw
            WHEN 'NOH' THEN 'NOP'
            WHEN 'NJN' THEN 'BKN'
            WHEN 'NOK' THEN 'NOP'
            WHEN 'SEA' THEN 'OKC'
            WHEN 'VAN' THEN 'MEM'
            WHEN 'CHH' THEN 'CHA'
            ELSE opponent_raw
        END AS opponent
    FROM normalized
    -- Data integrity: drop rows with nulls in key columns.
    WHERE game_id IS NOT NULL
      AND game_date IS NOT NULL
      AND team_abbreviation IS NOT NULL
      AND matchup IS NOT NULL
      AND wl IS NOT NULL
      AND pts IS NOT NULL
),
complete_games AS (
    -- Keep only games with exactly 2 team rows.
    SELECT *, COUNT(*) OVER (PARTITION BY game_id) AS team_rows
    FROM with_opponent
)
SELECT
    season_id,
    season,
    season_type,
    team_id,
    team_abbreviation_norm AS team_abbreviation,
    team_name,
    game_id,
    game_date,
    matchup,
    opponent,
    is_home,
    wl,
    min, pts, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct,
    oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus,
    source
FROM complete_games
WHERE team_rows = 2;
