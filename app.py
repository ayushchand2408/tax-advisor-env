"""
app.py — OpenEnv HTTP Server for TaxAdvisorEnv
Exposes reset(), step(), and state() as HTTP endpoints
so the OpenEnv validator can interact with the environment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import uvicorn
from env import TaxAdvisorEnv, TaxAction, grade_task

app = FastAPI(title="TaxAdvisorEnv", version="1.0.0")

# Global environment instance (one per server)
_envs: dict[str, TaxAdvisorEnv] = {}

# ─── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 0
    profile_index: int = 0

class StepRequest(BaseModel):
    tool_name: str
    arguments: dict = {}

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "TaxAdvisorEnv",
        "version": "1.0.0",
        "description": "OpenEnv-compliant tax filing environment",
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return initial observation."""
    env = TaxAdvisorEnv(task_id=req.task_id, profile_index=req.profile_index)
    _envs["default"] = env
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "state": env.state().model_dump(),
    }

@app.post("/step")
def step(req: StepRequest):
    """Take one action and return observation, reward, done, info."""
    env = _envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    action = TaxAction(tool_name=req.tool_name, arguments=req.arguments)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
        "score": grade_task(env),
    }

@app.get("/state")
def state():
    """Return current environment state."""
    env = _envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state().model_dump()

@app.get("/tasks")
def tasks():
    """Return available tasks."""
    return {
        "tasks": [
            {"id": 0, "name": "Tax Calculation", "difficulty": "easy"},
            {"id": 1, "name": "Deduction Finder", "difficulty": "medium"},
            {"id": 2, "name": "Complete Tax Filing", "difficulty": "hard"},
        ]
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
