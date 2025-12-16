from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main(model_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)

    X = pd.read_csv(input_csv)
    proba = model.predict_proba(X)[:, 1]

    out = X.copy()
    out["churn_risk"] = proba

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Wrote:", str(out_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to artifacts/model.joblib")
    ap.add_argument("--input", required=True, help="CSV containing feature columns (no target column).")
    ap.add_argument("--output", default="artifacts/predictions.csv")
    args = ap.parse_args()

    main(args.model, args.input, args.output)
