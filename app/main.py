import os

from app.student_risk import generate_plan


def main() -> None:
    goal = os.getenv("DEMO_GOAL", "Improve Portuguese course outcomes")
    result = generate_plan(goal)
    print("Personalized Learning Path Agent (education cohort model)")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
