from typing import Optional, Literal
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from src.environment import PreferenceEnvironment
from src.models import (
    GenerateAction, JudgeAction, RefinementAction,
    StepResult, GenerateObservation, JudgeObservation,
    RefineObservation, DoneObservation, ErrorObservation,
    TASK_CONFIG
)

app = FastAPI(title="tv_preference_env", version="0.1.0")

DATASET_PATH = Path("data/preference_dataset.json")
env = PreferenceEnvironment(DATASET_PATH)

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    example_id: Optional[str] = None

class StepRequest(BaseModel):
    action_type: Literal["generate", "judge", "refine"]
    action: dict   # raw dict, parsed by server based on action_type

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> GenerateObservation:
    try:
        return env.reset(
            task_id=request.task_id,
            example_id=request.example_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/step")
def step(request: StepRequest) -> StepResult:
    try:
        # Parse action based on action_type
        if request.action_type == "generate":
            action = GenerateAction(**request.action)
        elif request.action_type == "judge":
            action = JudgeAction(**request.action)
        elif request.action_type == "refine":
            action = RefinementAction(**request.action)
        else:
            raise HTTPException(status_code=422, detail=f"Unknown action_type: {request.action_type}")
        
        return env.step(action)
        
    except (ValueError, ValidationError) as e:
        # Catch Pydantic parsing errors and send clean 422
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")
    except Exception as e:
        # Catch MDP or other runtime errors and send clean 500
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/info")
def info() -> dict:
    return {
        "name":               "tv_preference_env",
        "version":            "0.1.0",
        "description":        "RL environment for training LLM preference judgment",
        "tasks":              list(TASK_CONFIG.keys()),
        "max_possible_reward": 0.80,
        "passing_threshold":   0.50,
        "action_space": {
            "generate": GenerateAction.model_json_schema(),
            "judge":    JudgeAction.model_json_schema(),
            "refine":   RefinementAction.model_json_schema(),
        },
        "observation_space": {
            "generate": GenerateObservation.model_json_schema(),
            "judge":    JudgeObservation.model_json_schema(),
            "refine":   RefineObservation.model_json_schema(),
            "done":     DoneObservation.model_json_schema(),
            "error":    ErrorObservation.model_json_schema(),
        },
    }

@app.get("/state")
def state() -> dict:
    try:
        return env.state_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve state: {str(e)}")