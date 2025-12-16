from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import yaml
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay

from telco_churn.data import fetch_telco, clean_telco_df, split_stratified


def main(config_path: str, model_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # Load data (same way as training)
    df = fetch_telco(openml_id=cfg["data"]["openml_id"])
    X, y = clean_telco_df(
        df,
        target_col=cfg["data"]["target_col"],
        drop_cols=cfg["data"].get("drop_cols", []),
    )
    split = split_stratified(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )

    # Load trained model
    model = joblib.load(model_path)
    y_proba = model.predict_proba(split.X_test)[:, 1]

    figures_dir = Path(cfg["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(split.y_test, y_proba)
    roc_path = figures_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Calibration curve
    plt.figure()
    CalibrationDisplay.from_predictions(split.y_test, y_proba, n_bins=10)
    cal_path = figures_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(cal_path, dpi=200)
    plt.close()

    print("Wrote:", roc_path)
    print("Wrote:", cal_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="artifacts/model.joblib")
    args = ap.parse_args()
    main(args.config, args.model)
