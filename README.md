![CI](https://github.com/manns79/Telco-Churn/actions/workflows/ci.yml/badge.svg)

# Telco-Churn

Predict customer churn using a reproducible scikit-learn pipeline with:
- end-to-end preprocessing (ColumnTransformer + one-hot encoding)
- probabilistic evaluation (ROC-AUC, PR-AUC, Brier score)
- a simple cost-aware threshold recommendation
- saved model artifact for batch scoring
- unit tests + GitHub Actions CI

## Quickstart

```bash
python -m venv .venv
# Windows (cmd)
.venv\Scripts\activate.bat
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .

python -m telco_churn.train --config configs/base.yaml
