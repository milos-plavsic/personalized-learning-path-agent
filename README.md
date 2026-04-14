# 07 - Personalized Learning Path Agent

An adaptive learning assistant that builds and continuously updates skill roadmaps using user goals, progress signals, and prerequisite graph reasoning.

## Why This Project Stands Out

- Product-oriented AI use case with measurable user outcomes.
- Combines recommendation logic and agentic planning.
- Good showcase for memory/state and personalization quality.

## Core Capabilities

- User skill profile and goal intake.
- Prerequisite graph traversal to identify skill gaps.
- Weekly personalized roadmap generation.
- Progress-aware plan adjustment and motivation nudges.
- Time-to-mastery estimates using lightweight predictive model.

## Suggested Tech Stack

- Python 3.11+
- `langgraph`, `networkx`, `scikit-learn`, `fastapi`, `sqlite`
- Optional UI: Streamlit or Next.js frontend

## Architecture (Graph)

`profile_ingest -> goal_parser -> prerequisite_mapper -> gap_analyzer -> roadmap_planner -> schedule_optimizer -> feedback_processor -> roadmap_update`

## Usage Suggestions

- Start with one track (e.g., ML engineering fundamentals).
- Keep roadmap outputs versioned for progress analytics.
- Add explicit "why this step now" explanations.

## Portfolio Additions

- Skill heatmap before/after 4-week progression.
- Cohort comparison mode (anonymous aggregates).
- Coach mode that adapts based on missed milestones.

## Milestones

- `v0.1`: static roadmap generation.
- `v0.2`: prerequisite graph + dynamic updates.
- `v0.3`: prediction of completion and bottlenecks.
- `v1.0`: multi-user API and polished UI.

## Demo Scenarios

1. 12-week path from beginner to ML deployment basics.
2. Interview prep plan based on weak topics.
3. Team onboarding plan for backend engineers entering AI.
