# Release Notes (2026-04)

## Scope
This release adds statistical reporting with visual outputs, robust dataset handling, and stable CI coverage.

## Data Source
- UCI Student Performance dataset (ID 320): `student-por.csv` (Portuguese course)

## Reporting Added
- New `analysis/` package with:
  - `report.py`, `plotting.py`, `json_util.py`, module entrypoint
- Generated outputs:
  - `reports/summary.json`
  - `reports/REPORT.md`
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/class_balance.png`

## Latest Report Snapshot
- Test ROC-AUC: `0.8023`
- Class balance: `not_at_risk=549`, `at_risk=100`
- Confusion matrix (test):
  - TN=106, FP=4
  - FN=17, TP=3

## Reliability and CI
- Added ZIP-based UCI fetch fallback via `app/uci_fetch.py`.
- Ensured local/offline stability with vendored `data/student-por.csv`.
- CI runs tests and `python -m analysis` smoke step.
- Upgraded actions to:
  - `actions/checkout@v6`
  - `actions/setup-python@v6`

## Latest CI Status
- Latest successful run: https://github.com/milos-plavsic/personalized-learning-path-agent/actions/runs/24447653019

## Dependency Notes
- Core stack includes `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `fastapi`, `pytest`, `httpx`.
