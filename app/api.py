from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.student_risk import generate_plan

app = FastAPI(title="Personalized Learning Path Agent", version="0.1.0")


class RoadmapRequest(BaseModel):
    goal: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/roadmap")
def roadmap(body: RoadmapRequest) -> dict:
    return generate_plan(body.goal)
