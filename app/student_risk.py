from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from app.datasets import DATA_SOURCE, load_student_portuguese


def _prepare_xy() -> tuple[pd.DataFrame, np.ndarray]:
    df = load_student_portuguese()
    y = (df["G3"] < 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def cohort_risk_summary() -> dict:
    X, y = _prepare_xy()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
    )
    auc = float(
        np.mean(
            cross_val_score(
                clf,
                X,
                y,
                cv=3,
                scoring="roc_auc",
                n_jobs=1,
            )
        )
    )
    clf.fit(X, y)
    names = np.array(X.columns)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    priorities = [f"Address driver feature `{names[i]}` (importance {importances[i]:.3f})" for i in top_idx]
    return {
        "cohort": "UCI Portuguese course students (secondary school)",
        "n_students": int(len(y)),
        "at_risk_rate": float(y.mean()),
        "cv_roc_auc_mean": auc,
        "suggested_priorities": priorities,
    }


def generate_plan(goal: str) -> dict:
    summary = cohort_risk_summary()
    return {
        "goal": goal,
        "weekly_hours": 8,
        "next_module": "Study skills + formative assessment loop",
        "estimated_weeks": 12,
        **summary,
        "data_source": DATA_SOURCE,
    }
