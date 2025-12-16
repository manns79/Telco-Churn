import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.features import build_preprocessor


def test_pipeline_trains_on_tiny_data():
    X = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6],
        "MonthlyCharges": [20.0, 30.0, 25.0, 80.0, 90.0, 70.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Month-to-month",
            "Two year",
            "Two year",
            "One year",
        ],
    })
    y = pd.Series([1, 0, 1, 0, 0, 0])

    pre = build_preprocessor(X)
    pipe = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=500))])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    assert len(proba) == len(X)
    assert (proba >= 0).all() and (proba <= 1).all()
