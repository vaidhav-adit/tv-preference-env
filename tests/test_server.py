from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "tv_preference_env"
    
    # Check action space
    assert "generate" in data["action_space"]
    assert "judge" in data["action_space"]
    
    # Check observation space (Must be exactly the 4 defined in spec)
    assert "generate" in data["observation_space"]
    assert "error" in data["observation_space"]
    assert "done" in data["observation_space"]

def test_reset_endpoint():
    # Test without payload (should randomly select)
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["phase"] == "generate"

    # Test with explicit payload
    response = client.post(
        "/reset",
        json={"task_id": "task_1_easy", "example_id": "example_001"}
    )
    assert response.status_code == 200
    assert response.json()["phase"] == "generate"

def test_step_endpoint():
    # Setup initial state
    client.post("/reset", json={"task_id": "task_1_easy", "example_id": "example_001"})
    
    # Valid generate step (must meet 20 char minimum from the Pydantic spec)
    payload = {
        "action_type": "generate",
        "action": {
            "response_text": "This is a test response that is long enough to easily pass the twenty character minimum validation length."
        }
    }
    response = client.post("/step", json=payload)
    
    # Check that it accepted the action and moved to the judge phase
    assert response.status_code == 200
    assert response.json()["observation"]["phase"] == "judge"

def test_state_endpoint():
    response = client.get("/state")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "phase" in response.json()