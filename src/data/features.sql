-- SQL side of feature engineering: joins `player_boxscores` onto `games_clean`
-- to produce a team-game-level player-availability feature, then exposes one
-- clean view (`team_game_features`) for features.py to read via pandas.
--
-- Availability logic: for each team-game, take every player who has already
-- played for that team this season (strictly before this game -- no leakage),
-- rank them by average minutes per game played, take the top 5, and compute
-- what fraction of those 5 actually played (MIN > 0) in this game. That's
-- `top5_avail_pct`. It's still an approximation -- a DNP from rest and a DNP
-- from injury look identical (Bug B05).
--
-- Bug B16: player_boxscores only contains players who appeared in a game (no
-- DNP rows), so the top 5 must come from the team's *prior* games, not from
-- this game's box score -- otherwise absent players are invisible and the
-- feature is ~always 1.0.
--
-- Run:
--     sqlite3 data/db/nba.db < src/data/features.sql

-- One row per player-game, restricted to games that survived cleaning
-- (games_clean already drops preseason/all-star/incomplete games). Joined on
-- (game_id, team_id) rather than team_abbreviation because team_id is stable
-- across relocations/renames while abbreviations are not (Bug B11).
DROP VIEW IF EXISTS player_boxscores_clean;
CREATE VIEW player_boxscores_clean AS
SELECT
    pb.player_id,
    pb.player_name,
    pb.team_id,
    gc.team_abbreviation,
    pb.game_id,
    gc.game_date,
    gc.season,
    gc.season_type,
    pb.min
FROM player_boxscores pb
JOIN games_clean gc
    ON pb.game_id = gc.game_id AND pb.team_id = gc.team_id;

-- Superseded views from the old (Bug B16) availability logic.
DROP VIEW IF EXISTS player_prior_avg_min;
DROP VIEW IF EXISTS team_top5_players;

-- Each player's stint with a team in a season. Materialized as a table (not a
-- view) because the trade check below looks it up once per candidate row.
DROP TABLE IF EXISTS player_stints;
CREATE TABLE player_stints AS
SELECT player_id, team_id, season, MIN(game_date) AS first_date
FROM player_boxscores_clean
GROUP BY player_id, team_id, season;
CREATE INDEX idx_stints_player ON player_stints(player_id, season);

-- One row per team-game: how many of the team's top-5 players (by minutes
-- per game played in prior games this season) played in this game.
DROP TABLE IF EXISTS team_top5_availability;
CREATE TABLE team_top5_availability AS
WITH team_games AS (
    SELECT DISTINCT team_id, game_id, game_date, season FROM games_clean
),
-- Every (team-game, player) pair for players who played for that team at
-- some point this season; game_min is NULL when the player sat out.
grid AS (
    SELECT
        tg.team_id, tg.game_id, tg.game_date, tg.season,
        s.player_id, s.first_date AS stint_first,
        pb.min AS game_min
    FROM team_games tg
    JOIN player_stints s
        ON s.team_id = tg.team_id AND s.season = tg.season
    LEFT JOIN player_boxscores pb
        ON pb.player_id = s.player_id AND pb.game_id = tg.game_id AND pb.team_id = tg.team_id
),
-- Running totals over this team's earlier games only (frame ends 1 PRECEDING,
-- so tonight's game never informs who counts as a top-5 player).
prior AS (
    SELECT
        *,
        SUM(COALESCE(game_min, 0)) OVER w AS prior_min_sum,
        SUM(CASE WHEN game_min > 0 THEN 1 ELSE 0 END) OVER w AS prior_games_played
    FROM grid
    WINDOW w AS (
        PARTITION BY team_id, season, player_id
        ORDER BY game_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
-- Rank by minutes per game *played*, so an injured star still ranks as a top-5
-- player while sitting out. Players who have since joined another team are
-- excluded -- a traded player isn't "missing."
ranked AS (
    SELECT
        p.team_id, p.game_id, p.game_min,
        ROW_NUMBER() OVER (
            PARTITION BY p.team_id, p.game_id
            ORDER BY p.prior_min_sum * 1.0 / p.prior_games_played DESC, p.player_id
        ) AS min_rank
    FROM prior p
    WHERE p.prior_games_played > 0
      AND NOT EXISTS (
          SELECT 1 FROM player_stints o
          WHERE o.player_id = p.player_id
            AND o.season = p.season
            AND o.team_id != p.team_id
            AND o.first_date > p.stint_first
            AND o.first_date < p.game_date
      )
)
SELECT
    team_id,
    game_id,
    COUNT(*) AS top5_count,
    SUM(CASE WHEN game_min > 0 THEN 1 ELSE 0 END) AS top5_played
FROM ranked
WHERE min_rank <= 5
GROUP BY team_id, game_id;

-- Final output: games_clean plus the availability feature. top5_avail_pct is
-- NULL for a team's first game each season (no prior games to rank players
-- from) -- handled like Bug B04 downstream (dropped).
DROP VIEW IF EXISTS team_game_features;
CREATE VIEW team_game_features AS
SELECT
    gc.*,
    a.top5_played * 1.0 / NULLIF(a.top5_count, 0) AS top5_avail_pct
FROM games_clean gc
LEFT JOIN team_top5_availability a
    ON gc.team_id = a.team_id AND gc.game_id = a.game_id;
