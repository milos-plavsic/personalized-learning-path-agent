from app.student_risk import generate_plan


def test_student_risk_loop_hits_max_iterations_when_threshold_strict() -> None:
    out = generate_plan("reduce risk", confidence_threshold=0.99, max_iterations=2, random_state=0)
    assert out["iterations"] == 2
    assert out["loop_terminated_reason"] == "max_iterations_reached"


def test_student_risk_contains_decision_trace() -> None:
    out = generate_plan("reduce risk", confidence_threshold=0.1, max_iterations=3, random_state=0)
    assert len(out["iteration_history"]) == out["iterations"]
    assert len(out["decision_log"]) == out["iterations"]
