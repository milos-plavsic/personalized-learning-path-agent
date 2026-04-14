import os


def generate_plan(goal: str) -> dict:
    return {
        "goal": goal,
        "weekly_hours": 8,
        "next_module": "model evaluation fundamentals",
        "estimated_weeks": 12,
    }


def main() -> None:
    goal = os.getenv("DEMO_GOAL", "Become an ML engineer")
    result = generate_plan(goal)
    print("Personalized Learning Path Agent")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
