import logging

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

PROB_CLIP_EPS = 1e-7

from train import FEATURES_PATH, MODEL_DIR, load_features, split_data

logger = logging.getLogger(__name__)

def load_model_artifacts(model_name: str) -> tuple:
    """Load model, scaler, and calibrator (whichever exist) from MODEL_DIR.
    Return (model, scaler_or_None, calibrator_or_None)."""
    model_path = MODEL_DIR / f"{model_name}_model.pkl"
    scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
    calibrator_path = MODEL_DIR / f"{model_name}_calibrator.pkl"

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    if scaler_path.exists():
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        logger.info("No scaler found for this model.")
        scaler = None

    if calibrator_path.exists():
        logger.info(f"Loading calibrator from {calibrator_path}")
        calibrator = joblib.load(calibrator_path)
    else:
        logger.info("No calibrator found for this model.")
        calibrator = None

    return model, scaler, calibrator

def evaluate_model(model, X, y_true, scaler=None, calibrator=None) -> dict:
    """Score a model on X/y_true. Apply scaler.transform(X) first if given,
    then calibrator.predict(raw_probs) if given.
    Use model.predict_proba(X)[:, 1] for probabilities, not .predict()."""
    if scaler is not None:
        X = scaler.transform(X)

    y_prob = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        y_prob = calibrator.predict(y_prob)
    # Isotonic calibration is a step function and can output exact 0/1 probabilities
    # for small validation bins; log_loss punishes a wrong prediction at exactly 0/1
    # with near-infinite loss, so clip away from the boundary before scoring.
    y_prob = np.clip(y_prob, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }

def naive_baseline(y_train, y_true) -> dict:
    """Always predict the training set's home-win rate as the probability
    (not a hard 0/1 label — that would blow up log_loss when wrong)."""
    home_win_rate = y_train.mean()
    y_prob_baseline = [home_win_rate] * len(y_true)
    y_pred_baseline = [1] * len(y_true)

    return {
        "accuracy": accuracy_score(y_true, y_pred_baseline),
        "brier_score": brier_score_loss(y_true, y_prob_baseline),
        "log_loss": log_loss(y_true, y_prob_baseline),
    }

def main() -> None:
    """Evaluate logistic regression, xgboost, and the naive baseline on the
    held-out test split, then print/log a comparison table.

    Test (not val) is used here because the logistic regression calibrator
    was fit on val predictions in train.py — scoring it on val again would be
    leakage (it would look artificially well-calibrated). Test has never been
    touched by training or calibration, so it's the honest number."""
    logging.basicConfig(level=logging.INFO)

    df = load_features(FEATURES_PATH)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    for model_name in ["logistic_regression", "xgboost"]:
        model, scaler, calibrator = load_model_artifacts(model_name)
        metrics = evaluate_model(model, X_test, y_test, scaler, calibrator)
        logger.info(f"Metrics for {model_name} (test): {metrics}")

    baseline_metrics = naive_baseline(y_train, y_test)
    logger.info(f"Metrics for naive baseline (test): {baseline_metrics}")

if __name__ == "__main__":
    main()
