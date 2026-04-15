from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from analysis.json_util import dumps_pretty
from analysis.plotting import class_balance_bar, confusion_matrix_heatmap
from app.datasets import DATA_SOURCE, load_student_portuguese


def generate_report(out_dir: Path | None = None, random_state: int = 42) -> dict:
    out = Path(out_dir or "reports")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = load_student_portuguese()
    y = (df["G3"] < 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    X = pd.get_dummies(X, drop_first=True)

    balance = {"not_at_risk": int((y == 0).sum()), "at_risk": int((y == 1).sum())}
    class_balance_bar(balance, fig_dir / "class_balance.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=4,
    )
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_hat)
    auc = float(roc_auc_score(y_test, proba))
    report = classification_report(
        y_test, y_hat, target_names=["not_at_risk", "at_risk"], output_dict=True
    )

    summary = {
        "data_source": DATA_SOURCE,
        "task": "at_risk_binary",
        "n_samples": int(len(y)),
        "class_balance": balance,
        "test_roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (out / "summary.json").write_text(dumps_pretty(summary), encoding="utf-8")

    confusion_matrix_heatmap(
        cm, ["not_at_risk", "at_risk"], fig_dir / "confusion_matrix.png"
    )

    md = "\n".join(
        [
            "# At-risk model — statistical report",
            "",
            f"**Data:** {DATA_SOURCE}",
            "",
            "## Test ROC-AUC",
            "",
            f"{auc:.4f}",
            "",
            "## Confusion matrix (test set)",
            "",
            "See `figures/confusion_matrix.png`.",
            "",
            "## Classification report (test)",
            "",
            "```json",
            dumps_pretty(report),
            "```",
        ]
    )
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return {"output_dir": str(out.resolve()), "test_roc_auc": auc}


def main() -> None:
    print(dumps_pretty(generate_report()))


if __name__ == "__main__":
    main()
