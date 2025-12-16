from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.data import clean_telco_df, fetch_telco, split_stratified
from telco_churn.features import build_preprocessor
from telco_churn.metrics import classification_report_proba, pick_threshold_by_cost


def build_model(name: str, max_iter: int = 2000):
    name = name.lower().strip()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=max_iter, class_weight="balanced")
    if name in {"hgb", "hist_gb", "histgradientboosting"}:
        return HistGradientBoostingClassifier()
    raise ValueError(f"Unknown model name: {name}")


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # 1) Load + clean
    df = fetch_telco(openml_id=cfg["data"]["openml_id"])
    X, y = clean_telco_df(
        df,
        target_col=cfg["data"]["target_col"],
        drop_cols=cfg["data"].get("drop_cols", []),
    )

    # 2) Split
    split = split_stratified(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )

    # 3) Build pipeline
    pre = build_preprocessor(split.X_train)
    clf = build_model(cfg["model"]["name"], max_iter=cfg["model"].get("max_iter", 2000))
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    # 4) Train
    pipe.fit(split.X_train, split.y_train)

    # 5) Evaluate
    y_proba = pipe.predict_proba(split.X_test)[:, 1]
    report = classification_report_proba(split.y_test, y_proba)
    thresh = pick_threshold_by_cost(split.y_test, y_proba)

    # 6) Save artifacts
    artifacts_dir = Path(cfg["outputs"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, artifacts_dir / "model.joblib")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(
            {
                "roc_auc": report.roc_auc,
                "pr_auc": report.pr_auc,
                "brier": report.brier,
                "recommended_threshold": thresh,
                "config": cfg,
            },
            indent=2,
        )
    )

    print("Saved:", artifacts_dir / "model.joblib")
    print("Saved:", artifacts_dir / "metrics.json")
    print(f"ROC-AUC={report.roc_auc:.4f}  PR-AUC={report.pr_auc:.4f}  Brier={report.brier:.4f}")
    print("Threshold suggestion:", thresh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    main(args.config)
