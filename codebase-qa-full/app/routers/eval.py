"""
eval.py — run the retrieval benchmark on demand

POST /eval/run?repo_id=...&max_questions=100

This is the endpoint you'd wire into CI: re-ingest a repo, hit this
endpoint, and fail the build if recall_at_5 drops below some threshold
compared to the last run. The project doesn't include a CI config (that's
specific to your pipeline), but the numbers this returns are exactly what
you'd gate on.
"""

from fastapi import APIRouter, Request, HTTPException, Query as QParam
from app.eval.runner import run_eval
from app.routers.repos import get_registry

router = APIRouter()


@router.post("/run")
async def run_eval_endpoint(
    request: Request,
    repo_id: str = QParam(...),
    max_questions: int = QParam(default=100, ge=1, le=500),
):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {registry[repo_id]['status']}")

    cfg, qdrant = request.app.state.settings, request.app.state.qdrant
    return await run_eval(repo_id, qdrant, cfg, max_questions=max_questions)
