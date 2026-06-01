from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.datasets import DATA_SOURCE, load_student_portuguese
from app.orchestration_policy import (
    confidence_label,
    decide_loop,
    normalized_stability,
    normalize_threshold,
    weighted_confidence,
)


class IterationRisk(TypedDict):
    iteration: int
    include_prior_grades: bool
    n_estimators: int
    max_depth: int | None
    test_roc_auc: float
    at_risk_f1: float
    cv_auc_mean: float
    cv_auc_std: float
    confidence_score: float


class RiskState(TypedDict, total=False):
    goal: str
    confidence_threshold: float
    max_iterations: int
    random_state: int

    iteration: int
    include_prior_grades: bool
    n_estimators: int
    max_depth: int | None

    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    test_auc: float
    at_risk_f1: float
    cv_auc_mean: float
    cv_auc_std: float

    confidence_score: float
    confidence_label: str
    continue_loop: bool
    stop_reason: str

    history: list[IterationRisk]
    decisions: list[str]
    plan_text: str


def _validate(state: RiskState) -> RiskState:
    return {
        "goal": state.get("goal", "improve grade outcomes"),
        "confidence_threshold": normalize_threshold(state.get("confidence_threshold", 0.72)),
        "max_iterations": max(1, int(state.get("max_iterations", 3))),
        "random_state": int(state.get("random_state", 42)),
        "iteration": 0,
        "history": [],
        "decisions": [],
    }


def _load(state: RiskState) -> RiskState:
    return {"df": load_student_portuguese()}


def _plan(state: RiskState) -> RiskState:
    it = int(state["iteration"]) + 1
    include_prior = it >= 2
    n_estimators = 220 if it == 1 else 360
    max_depth: int | None = 10 if it == 1 else None
    decision = (
        f"iteration={it}: include_prior_grades={include_prior}, "
        f"n_estimators={n_estimators}, max_depth={max_depth}"
    )
    return {
        "iteration": it,
        "include_prior_grades": include_prior,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "decisions": [*state["decisions"], decision],
    }


def _prep(state: RiskState) -> RiskState:
    df = state["df"]
    y = (df["G3"] < 10).astype(int).to_numpy()
    cols = ["G3"]
    if not state["include_prior_grades"]:
        cols.extend(["G1", "G2"])
    X = pd.get_dummies(df.drop(columns=cols), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=state["random_state"],
        stratify=y,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def _train_eval(state: RiskState) -> RiskState:
    model = RandomForestClassifier(
        n_estimators=state["n_estimators"],
        max_depth=state["max_depth"],
        class_weight="balanced",
        random_state=state["random_state"],
        n_jobs=4,
    )
    model.fit(state["X_train"], state["y_train"])
    proba = model.predict_proba(state["X_test"])[:, 1]
    pred = (proba >= 0.5).astype(int)

    test_auc = float(roc_auc_score(state["y_test"], proba))
    at_risk_f1 = float(f1_score(state["y_test"], pred, pos_label=1, zero_division=0))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=state["random_state"])
    cv_auc = cross_val_score(model, state["X_train"], state["y_train"], cv=cv, scoring="roc_auc", n_jobs=1)

    return {
        "test_auc": test_auc,
        "at_risk_f1": at_risk_f1,
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
    }


def _assess(state: RiskState) -> RiskState:
    components = {
        "primary_quality": state["test_auc"],
        "secondary_quality": state["at_risk_f1"],
        "stability": normalized_stability(state["cv_auc_std"]),
    }
    score = weighted_confidence(components)
    conf_label = confidence_label(score)
    loop = decide_loop(
        confidence_score=score,
        confidence_threshold=state["confidence_threshold"],
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )

    h: IterationRisk = {
        "iteration": state["iteration"],
        "include_prior_grades": state["include_prior_grades"],
        "n_estimators": state["n_estimators"],
        "max_depth": state["max_depth"],
        "test_roc_auc": state["test_auc"],
        "at_risk_f1": state["at_risk_f1"],
        "cv_auc_mean": state["cv_auc_mean"],
        "cv_auc_std": state["cv_auc_std"],
        "confidence_score": score,
    }

    return {
        "confidence_score": score,
        "confidence_label": conf_label,
        "continue_loop": loop["continue_loop"],
        "stop_reason": loop["stop_reason"],
        "history": [*state["history"], h],
    }


def _route(state: RiskState) -> Literal["plan", "finalize"]:
    return "plan" if state["continue_loop"] else "finalize"


def _finalize(state: RiskState) -> RiskState:
    score = state["confidence_score"]
    if score >= 0.8:
        intensity = "maintain current study rhythm and target advanced exercises"
    elif score >= 0.6:
        intensity = "add weekly remediation on weak subjects and monitor attendance"
    else:
        intensity = "use intensive intervention with tutor support and short-cycle reviews"

    plan = (
        f"Goal: {state['goal']}. Priority action: {intensity}. "
        f"Track risk trend weekly and reevaluate after each assessment cycle."
    )
    return {"plan_text": plan}


def build_risk_graph():
    g = StateGraph(RiskState)
    g.add_node("validate", _validate)
    g.add_node("load", _load)
    g.add_node("plan", _plan)
    g.add_node("prep", _prep)
    g.add_node("train_eval", _train_eval)
    g.add_node("assess", _assess)
    g.add_node("finalize", _finalize)

    g.set_entry_point("validate")
    g.add_edge("validate", "load")
    g.add_edge("load", "plan")
    g.add_edge("plan", "prep")
    g.add_edge("prep", "train_eval")
    g.add_edge("train_eval", "assess")
    g.add_conditional_edges("assess", _route, {"plan": "plan", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_RISK_GRAPH = build_risk_graph()


def generate_plan(
    goal: str,
    *,
    confidence_threshold: float = 0.72,
    max_iterations: int = 3,
    random_state: int = 42,
) -> dict:
    out = _RISK_GRAPH.invoke(
        {
            "goal": goal,
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
            "random_state": random_state,
        }
    )
    return {
        "goal": out["goal"],
        "plan": out["plan_text"],
        "risk_model": "RandomForestClassifier",
        "test_roc_auc": out["test_auc"],
        "at_risk_f1": out["at_risk_f1"],
        "cv_roc_auc_mean": out["cv_auc_mean"],
        "cv_roc_auc_std": out["cv_auc_std"],
        "confidence_score": out["confidence_score"],
        "confidence_label": out["confidence_label"],
        "confidence_threshold": out["confidence_threshold"],
        "iterations": out["iteration"],
        "loop_terminated_reason": out["stop_reason"],
        "iteration_history": out["history"],
        "decision_log": out["decisions"],
        "data_source": DATA_SOURCE,
    }
