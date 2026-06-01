from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.learner_store import get_roadmap, save_roadmap
from app.student_risk import generate_plan
from finetune.tuner import run_risk_model_finetune

app = FastAPI(title="Personalized Learning Path Agent", version="0.3.0")

_ui = Path(__file__).resolve().parent / "static"
if _ui.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui), html=True), name="roadmap-ui")


class RoadmapRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    confidence_threshold: float = Field(0.72, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class RoadmapPersistRequest(RoadmapRequest):
    user_id: str = Field("demo-user", min_length=1)


@app.post("/v1/roadmap")
def roadmap(body: RoadmapPersistRequest) -> dict:
    plan = generate_plan(
        body.goal,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )
    save_roadmap(body.user_id, body.goal, plan)
    return plan


@app.get("/v1/roadmap/{user_id}")
def roadmap_get(user_id: str) -> dict:
    stored = get_roadmap(user_id)
    if not stored:
        raise HTTPException(status_code=404, detail="no roadmap for user")
    return stored


@app.post("/v1/finetune/risk_model")
def finetune_risk() -> dict:
    return run_risk_model_finetune()
