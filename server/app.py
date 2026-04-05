"""
server/app.py — OpenEnv HTTP Server for TaxAdvisorEnv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from env import TaxAdvisorEnv, TaxAction, grade_task

app = FastAPI(title="TaxAdvisorEnv", version="1.0.0")
_envs: dict = {}

class ResetRequest(BaseModel):
    task_id: int = 0
    profile_index: int = 0

class StepRequest(BaseModel):
    tool_name: str
    arguments: dict = {}

@app.get("/")
def root():
    return {"name": "TaxAdvisorEnv", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    env = TaxAdvisorEnv(task_id=req.task_id, profile_index=req.profile_index)
    _envs["default"] = env
    obs = env.reset()
    return {"observation": obs.model_dump(), "state": env.state().model_dump()}

@app.post("/step")
def step(req: StepRequest):
    env = _envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    action = TaxAction(tool_name=req.tool_name, arguments=req.arguments)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info, "score": grade_task(env)}

@app.get("/state")
def state():
    env = _envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state().model_dump()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
