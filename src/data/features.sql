-- SQL side of feature engineering: joins `player_boxscores` onto `games_clean`
-- to produce a team-game-level player-availability feature, then exposes one
-- clean view (`team_game_features`) for features.py to read via pandas.
--
-- Availability logic: for each team-game, rank that team's players by their
-- average minutes over *prior* games this season (season-to-date, strictly
-- before the current game date -- no leakage), take the top 5, and compute
-- what fraction of them actually played (MIN > 0) in this game. That's
-- `top5_avail_pct`. It's an approximation of "were the team's normal
-- rotation players out" -- see bug B05 in PLAN.md: a DNP from rest and a DNP
-- from injury look identical in this data.
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

-- Each player's average minutes over all *prior* games in the same
-- season/season_type, as of just before this game. window frame explicitly
-- excludes the current row so the current game can never leak into its own
-- "is this a rotation player" determination.
DROP VIEW IF EXISTS player_prior_avg_min;
CREATE VIEW player_prior_avg_min AS
SELECT
    player_id,
    team_id,
    game_id,
    game_date,
    season,
    season_type,
    AVG(min) OVER (
        PARTITION BY player_id, season, season_type
        ORDER BY game_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_avg_min
FROM player_boxscores_clean;

-- Rank each team's players, per game, by that prior season-to-date average.
-- Players with no prior games this season (first appearance) have a NULL
-- average and are excluded -- they can't yet be judged a rotation player.
DROP VIEW IF EXISTS team_top5_players;
CREATE VIEW team_top5_players AS
SELECT
    team_id,
    game_id,
    player_id,
    RANK() OVER (
        PARTITION BY team_id, game_id ORDER BY prior_avg_min DESC
    ) AS min_rank
FROM player_prior_avg_min
WHERE prior_avg_min IS NOT NULL;

-- For each team-game, how many of the top-5 rotation players (by the ranking
-- above) actually appeared (MIN > 0) in this game.
DROP VIEW IF EXISTS team_top5_availability;
CREATE VIEW team_top5_availability AS
SELECT
    t.team_id,
    t.game_id,
    COUNT(*) AS top5_count,
    SUM(CASE WHEN pb.min > 0 THEN 1 ELSE 0 END) AS top5_played
FROM team_top5_players t
JOIN player_boxscores_clean pb
    ON t.player_id = pb.player_id
   AND t.game_id = pb.game_id
   AND t.team_id = pb.team_id
WHERE t.min_rank <= 5
GROUP BY t.team_id, t.game_id;

-- Final output: games_clean plus the availability feature. top5_avail_pct is
-- NULL for a team's first ~handful of games each season (no prior-minutes
-- history yet to rank players from) -- same shape as the rolling-average NaN
-- issue in Bug B04, handled the same way downstream (dropped).
DROP VIEW IF EXISTS team_game_features;
CREATE VIEW team_game_features AS
SELECT
    gc.*,
    a.top5_played * 1.0 / NULLIF(a.top5_count, 0) AS top5_avail_pct
FROM games_clean gc
LEFT JOIN team_top5_availability a
    ON gc.team_id = a.team_id AND gc.game_id = a.game_id;
