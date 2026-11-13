"""Visualizations for the trained logistic regression model.

Not part of the production pipeline (train.py / evaluate.py) -- this is an
exploratory/portfolio tool, matching the notebooks/exploration.ipynb carve-out
in PLAN.md's architecture (EDA and plots live outside the modeling pipeline).

Produces three PNGs in PLOTS_DIR:
    decision_boundary.png   -- 2-feature slice of the model's decision boundary
    calibration_curve.png   -- reliability diagram, uncalibrated vs. isotonic
    coefficients.png        -- logistic regression coefficients, sorted by magnitude
    combined_score.png      -- all features collapsed into one score vs. win probability
    feature_correlation.png -- how strongly each pair of features moves together

Run:
    python src/model/visualize.py
"""

import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train import MODEL_DIR, load_features, split_data

PLOTS_DIR = Path("plots")

logger = logging.getLogger(__name__)


def plot_decision_boundary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_x: str,
    feature_y: str,
    save_path: Path,
) -> None:
    """Fit a simplified 2-feature-only logistic regression (for visualization
    only -- not the production model, which uses all 14 features and has a
    hyperplane boundary that can't be drawn directly) and plot its decision
    boundary against real test-set outcomes."""
    X2_train = X_train[[feature_x, feature_y]].values
    X2_test = X_test[[feature_x, feature_y]].values

    scaler = StandardScaler().fit(X2_train)
    X2_train_s = scaler.transform(X2_train)
    X2_test_s = scaler.transform(X2_test)

    viz_model = LogisticRegression().fit(X2_train_s, y_train)

    x_min, x_max = X2_test_s[:, 0].min() - 1, X2_test_s[:, 0].max() + 1
    y_min, y_max = X2_test_s[:, 1].min() - 1, X2_test_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    probs = viz_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2, linestyles="--")

    rng = np.random.RandomState(42)
    sample = rng.choice(len(X2_test_s), size=min(400, len(X2_test_s)), replace=False)
    colors = np.where(y_test.values[sample] == 1, "darkblue", "darkred")
    ax.scatter(
        X2_test_s[sample, 0], X2_test_s[sample, 1],
        c=colors, s=15, alpha=0.7, edgecolors="white", linewidths=0.3,
    )

    ax.set_xlabel(f"{feature_x} (standardized)")
    ax.set_ylabel(f"{feature_y} (standardized)")
    ax.set_title(
        "2-feature logistic regression decision boundary (test set)\n"
        "Blue = home win, Red = home loss | dashed line = 50% boundary"
    )
    plt.colorbar(contour, label="P(home win)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_calibration_curve(
    y_test: pd.Series,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    save_path: Path,
) -> None:
    """Reliability diagram: for games where the model said 'X% chance,' how
    often did the home team actually win? A perfectly calibrated model falls
    on the diagonal. Compares raw vs. isotonic-calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for label, probs in [("Uncalibrated", raw_probs), ("Isotonic-calibrated", calibrated_probs)]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual home wins")
    ax.set_title("Calibration curve (reliability diagram) -- test set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_coefficients(model: LogisticRegression, feature_names: list[str], save_path: Path) -> None:
    """Bar chart of logistic regression coefficients, sorted by magnitude --
    the direct, interpretable 'what did the model learn' answer."""
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values(key=abs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["darkred" if c < 0 else "darkblue" for c in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("Logistic regression coefficients\nBlue = raises home-win odds, Red = lowers it")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_combined_score(model: LogisticRegression, X_scaled: np.ndarray, y_true: pd.Series, save_path: Path) -> None:
    """Collapse all features into one axis: each game's weighted sum of features
    (the log-odds score), plotted against the S-curve that turns it into a
    probability, with the actual win rate of real games overlaid."""
    score = X_scaled @ model.coef_[0] + model.intercept_[0]
    xs = np.linspace(score.min() - 0.5, score.max() + 0.5, 300)

    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-0.03, 0.03, len(score))
    ax.scatter(score, y_true.values + jitter, s=6, alpha=0.15, color="gray", label="Real games (1 = home win, 0 = loss)")

    bins = pd.qcut(score, 12)
    binned = pd.DataFrame({"score": score, "won": y_true.values}).groupby(bins, observed=True).mean()
    ax.scatter(binned["score"], binned["won"], s=60, color="darkred", zorder=3, label="Actual home win rate (binned)")

    ax.plot(xs, 1 / (1 + np.exp(-xs)), color="darkblue", linewidth=2.5, label="Model: P(home win)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Combined score = sum of (feature x weight) + intercept, all 16 features")
    ax.set_ylabel("Probability home team wins")
    ax.set_title("Every feature at once: combined score -> win probability (test set)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_feature_correlation(X: pd.DataFrame, save_path: Path) -> None:
    """Heatmap of how strongly each pair of features moves together
    (+1 = always rise together, -1 = opposite, 0 = unrelated)."""
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("Feature correlations (train set)")
    plt.colorbar(im, label="Correlation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_features()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")
    calibrator = joblib.load(MODEL_DIR / "logistic_regression_calibrator.pkl")

    X_test_scaled = scaler.transform(X_test)
    raw_probs = model.predict_proba(X_test_scaled)[:, 1]
    calibrated_probs = np.clip(calibrator.predict(raw_probs), 0.02, 0.98)

    plot_decision_boundary(
        X_train, y_train, X_test, y_test,
        feature_x="HOME_ROLL_PLUS_MINUS", feature_y="HOME_REST_DAYS",
        save_path=PLOTS_DIR / "decision_boundary.png",
    )
    plot_calibration_curve(y_test, raw_probs, calibrated_probs, PLOTS_DIR / "calibration_curve.png")
    plot_coefficients(model, list(X_train.columns), PLOTS_DIR / "coefficients.png")
    plot_combined_score(model, X_test_scaled, y_test, PLOTS_DIR / "combined_score.png")
    plot_feature_correlation(X_train, PLOTS_DIR / "feature_correlation.png")


if __name__ == "__main__":
    main()
