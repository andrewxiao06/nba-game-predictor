import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_DIR = Path("models")

NON_FEATURE_COLS = [
    "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
    "HOME_TEAM", "AWAY_TEAM", "HOME_WIN", "SPLIT",
]

logger = logging.getLogger(__name__)

def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load features from a CSV file."""
    logger.info(f"Loading features from {path}")
    return pd.read_csv(path)

def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop NON_FEATURE_COLS to build X. HOME_WIN is y."""
    X = df.drop(columns=NON_FEATURE_COLS)
    y = df["HOME_WIN"]
    return X, y

def split_data(
    df: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Split the DataFrame into train/val/test using the existing SPLIT column."""
    logger.info("Splitting data into train, validation, and test sets")
    train_df = df[df["SPLIT"] == "train"]
    val_df = df[df["SPLIT"] == "val"]
    test_df = df[df["SPLIT"] == "test"]

    return get_X_y(train_df), get_X_y(val_df), get_X_y(test_df)

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on train features only."""
    logger.info("Fitting StandardScaler on training data")
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train a logistic regression model to fit baseline logistic regression."""
    logger.info("Training Logistic Regression model")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """fit xgboost (no scaling needed - tree models dont care about feature scale).

    Shallow/conservative hyperparameters, not XGBoost defaults: with ~7.3k train
    rows and 14 features, the default max_depth=6 overfits badly (test log loss
    0.736 vs. 0.630 here). Picked via a small manual sweep over max_depth/
    n_estimators/learning_rate, selected on val log loss (not test -- test is
    only touched once, afterward, to confirm), not a proper CV grid search.
    """
    logger.info("Training XGBoost model")
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        max_depth=1,
        n_estimators=200,
        learning_rate=0.1,
    )
    model.fit(X_train, y_train)
    return model

def fit_calibrator(raw_probs, y_val: pd.Series) -> IsotonicRegression:
    """Fit isotonic regression mapping a model's raw predict_proba output to a
    calibrated probability, using held-out val predictions (never train predictions
    — the model already fit those, so they'd look falsely well-calibrated)."""
    logger.info("Fitting isotonic calibrator on validation predictions")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_val)
    return calibrator

def save_artifacts(
    model: LogisticRegression | XGBClassifier,
    model_name: str,
    scaler: StandardScaler | None = None,
    calibrator: IsotonicRegression | None = None,
) -> None:
    """Save a model (and its scaler/calibrator, if present) to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}_model.pkl"

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    if scaler is not None:
        scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
        logger.info(f"Saving scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path)

    if calibrator is not None:
        calibrator_path = MODEL_DIR / f"{model_name}_calibrator.pkl"
        logger.info(f"Saving calibrator to {calibrator_path}")
        joblib.dump(calibrator, calibrator_path)

def main() -> None:
    """orchestrate: load > split > get_X_y > scale > train both modesl > save."""
    logging.basicConfig(level=logging.INFO)

    df = load_features()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    # Fit scaler on training data
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model, then calibrate on held-out val predictions
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    lr_val_raw_probs = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_calibrator = fit_calibrator(lr_val_raw_probs, y_val)
    save_artifacts(lr_model, "logistic_regression", scaler=scaler, calibrator=lr_calibrator)

    # Train XGBoost model (no scaling needed)
    xgb_model = train_xgboost(X_train, y_train)
    save_artifacts(xgb_model, "xgboost")

if __name__ == "__main__":
    main()
