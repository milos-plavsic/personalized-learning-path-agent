from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.student_risk import generate_plan
from finetune.tuner import run_risk_model_finetune

app = FastAPI(title="Personalized Learning Path Agent", version="0.2.0")


class RoadmapRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    confidence_threshold: float = Field(0.72, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/roadmap")
def roadmap(body: RoadmapRequest) -> dict:
    return generate_plan(
        body.goal,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )


@app.post("/v1/finetune/risk_model")
def finetune_risk() -> dict:
    return run_risk_model_finetune()
