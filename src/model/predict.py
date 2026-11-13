"""Predict a win probability for an upcoming (not-yet-played) matchup.

Unlike train.py/evaluate.py, which score games that already happened, this
computes each team's rolling 10-game form and rest days *as of a given date*
directly from the database, then feeds that single matchup row through the
saved logistic regression model.

Known limitation: TOP5_AVAIL_PCT (player availability) is computed from a
game's own box score -- it can't be known before the game is played. Real
injury-report data isn't integrated (see PLAN.md Feature Backlog), so this
defaults to 1.0 (assume full roster available) unless overridden with
--home-avail / --away-avail.

Uses the UNCALIBRATED logistic regression, not the isotonic-calibrated one:
evaluation on the held-out test set showed the raw model is already well
calibrated, and the isotonic calibrator (fit on only ~1,094 val games) made
accuracy, Brier score, and log loss all slightly worse (see
plots/calibration_curve.png).

Run:
    python src/model/predict.py --home BOS --away LAL --date 2026-11-01
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from train import MODEL_DIR

logger = logging.getLogger(__name__)

# Mirrors src/data/features.py's DB_PATH/ROLLING_STATS/ROLLING_WINDOW/MAX_REST_DAYS -- not
# imported directly since that module lives in a different package (src/data)
# with no shared path setup. Keep these in sync if features.py changes.
DB_PATH = Path("data/db/nba.db")
ROLLING_STATS = ["PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS"]
ROLLING_WINDOW = 10
MAX_REST_DAYS = 7

# Must match the column order train.py's X was built with (NON_FEATURE_COLS
# dropped, HOME_* then AWAY_*) -- scikit-learn doesn't check column names,
# only position, so getting this order wrong silently produces garbage.
FEATURE_ORDER = [
    "HOME_REST_DAYS", "HOME_TOP5_AVAIL_PCT", "HOME_ROLL_WIN_PCT",
    "HOME_ROLL_PTS", "HOME_ROLL_REB", "HOME_ROLL_AST", "HOME_ROLL_FG_PCT", "HOME_ROLL_PLUS_MINUS",
    "AWAY_REST_DAYS", "AWAY_TOP5_AVAIL_PCT", "AWAY_ROLL_WIN_PCT",
    "AWAY_ROLL_PTS", "AWAY_ROLL_REB", "AWAY_ROLL_AST", "AWAY_ROLL_FG_PCT", "AWAY_ROLL_PLUS_MINUS",
]


def load_team_games(team: str, before_date: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load a team's completed games strictly before a given date, most recent first.

    Crosses season boundaries on purpose, matching features.py (Bug B15): early
    in a season, the window is filled with last season's trailing games.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT game_date, pts, reb, ast, fg_pct, plus_minus, wl
            FROM games_clean
            WHERE team_abbreviation = ? AND game_date < ?
            ORDER BY game_date DESC
            """,
            conn,
            params=(team, before_date),
        )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def compute_team_features(team: str, as_of_date: str, avail_pct: float) -> dict:
    """Compute REST_DAYS and ROLL_* features for one team as of a given date.

    Mirrors features.py's add_rest_days/add_rolling_features logic, but
    against the team's most recent real games instead of a fixed historical
    row, since there's no "current game" row to look backward from yet.
    """
    games = load_team_games(team, as_of_date)
    if len(games) < ROLLING_WINDOW:
        raise ValueError(
            f"{team} has only {len(games)} games before {as_of_date}, "
            f"need {ROLLING_WINDOW} for a full rolling window."
        )

    rest_days = min((pd.Timestamp(as_of_date) - games.iloc[0]["game_date"]).days, MAX_REST_DAYS)
    last_n = games.iloc[:ROLLING_WINDOW]

    features = {"REST_DAYS": rest_days, "TOP5_AVAIL_PCT": avail_pct}
    for stat in ROLLING_STATS:
        features[f"ROLL_{stat}"] = last_n[stat.lower()].mean()
    features["ROLL_WIN_PCT"] = (last_n["wl"] == "W").mean()
    return features


def build_matchup_row(home: str, away: str, date: str, home_avail: float, away_avail: float) -> pd.DataFrame:
    """Build the single-row feature vector for a home/away matchup on a date."""
    home_feats = compute_team_features(home, date, home_avail)
    away_feats = compute_team_features(away, date, away_avail)

    row = {f"HOME_{k}": v for k, v in home_feats.items()}
    row.update({f"AWAY_{k}": v for k, v in away_feats.items()})
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict(home: str, away: str, date: str, home_avail: float = 1.0, away_avail: float = 1.0) -> float:
    """Return P(home team wins) for a given matchup and date."""
    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    X = build_matchup_row(home, away, date, home_avail, away_avail)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0, 1]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Predict a win probability for an upcoming NBA matchup.")
    parser.add_argument("--home", required=True, help="Home team abbreviation, e.g. BOS")
    parser.add_argument("--away", required=True, help="Away team abbreviation, e.g. LAL")
    parser.add_argument("--date", required=True, help="Game date, YYYY-MM-DD")
    parser.add_argument("--home-avail", type=float, default=1.0, help="Home top-5 availability fraction (default 1.0 = full roster)")
    parser.add_argument("--away-avail", type=float, default=1.0, help="Away top-5 availability fraction (default 1.0 = full roster)")
    args = parser.parse_args()

    home_win_prob = predict(args.home, args.away, args.date, args.home_avail, args.away_avail)
    print(f"\n{args.home} (home) vs {args.away} (away) on {args.date}")
    print(f"  P({args.home} wins) = {home_win_prob:.3f}")
    print(f"  P({args.away} wins) = {1 - home_win_prob:.3f}\n")


if __name__ == "__main__":
    main()
