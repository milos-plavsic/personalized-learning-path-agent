from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from app.datasets import DATA_SOURCE, load_student_portuguese


def run_risk_model_finetune(random_state: int = 42) -> dict:
    """Tune the at-risk classifier via randomized search on held-out data."""
    n_iter = int(os.getenv("FINETUNE_N_ITER", "12"))
    df = load_student_portuguese()
    y = (df["G3"] < 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    param = {
        "n_estimators": randint(80, 350),
        "max_depth": [None, 6, 10, 14],
        "min_samples_leaf": randint(1, 5),
    }
    base = RandomForestClassifier(
        random_state=random_state, n_jobs=2, class_weight="balanced"
    )
    search = RandomizedSearchCV(
        base,
        param,
        n_iter=n_iter,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    best = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in search.best_params_.items()}
    return {
        "best_params": best,
        "test_roc_auc": auc,
        "n_iter": n_iter,
        "data_source": DATA_SOURCE,
    }


def main() -> None:
    out = run_risk_model_finetune()
    print("At-risk model hyperparameter fine-tune")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
