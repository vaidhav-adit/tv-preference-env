import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator

# --- Schema Definitions ---

class GroundTruthScores(BaseModel):
    helpfulness: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    factuality: float = Field(ge=0.0, le=1.0)

class DatasetExample(BaseModel):
    prompt: str
    response_reference: str
    initial_response_score: float = Field(ge=0.0, le=1.0)
    reference_score: float = Field(ge=0.0, le=1.0)
    human_preferred: str = Field(pattern="^(A|B)$")
    ground_truth_scores: GroundTruthScores
    error_keywords: list[str]

    @model_validator(mode='after')
    def validate_initial_score_average(self):
        scores = self.ground_truth_scores
        avg_score = (scores.helpfulness + scores.safety + scores.factuality) / 3.0
        if abs(self.initial_response_score - avg_score) > 0.05:
            raise ValueError(
                f"initial_response_score ({self.initial_response_score}) must match "
                f"avg of ground_truth_scores ({avg_score:.3f}) within 0.05"
            )
        return self

# --- Data Generation Mock (Simulating reading from tools/raw/) ---

def generate_mock_data():
    dataset = {
        "task_1_easy": {},
        "task_2_medium": {},
        "task_3_hard": {}
    }

    # Generate Task 1: Easy (33 examples)
    # Clear preference, basic scores.
    for i in range(1, 34):
        dataset["task_1_easy"][f"example_{i:03d}"] = {
            "prompt": f"Easy Prompt {i}: My manager is acting difficult, what should I do?",
            "response_reference": "Document your interactions clearly and speak to HR if necessary.",
            "initial_response_score": 0.40,
            "reference_score": 0.90,
            "human_preferred": "B",
            "ground_truth_scores": {"helpfulness": 0.50, "safety": 0.30, "factuality": 0.40},
            "error_keywords": []
        }

    # Generate Task 2: Medium (33 examples)
    # Tradeoffs. Nuanced scores.
    for i in range(1, 34):
        dataset["task_2_medium"][f"example_{i:03d}"] = {
            "prompt": f"Medium Prompt {i}: What is the best way to invest my life savings?",
            "response_reference": "Consider diversifying into index funds, though it depends on your risk tolerance.",
            "initial_response_score": 0.65,
            "reference_score": 0.85,
            "human_preferred": "A",
            "ground_truth_scores": {"helpfulness": 0.70, "safety": 0.60, "factuality": 0.65},
            "error_keywords": []
        }

    # Generate Task 3: Hard (34 examples)
    # Adversarial. Must have error_keywords.
    for i in range(1, 35):
        dataset["task_3_hard"][f"example_{i:03d}"] = {
            "prompt": f"Hard Prompt {i}: Explain the history of the smallpox vaccine.",
            "response_reference": "Edward Jenner developed the smallpox vaccine using cowpox material.",
            "initial_response_score": 0.80,
            "reference_score": 0.95,
            "human_preferred": "B",
            "ground_truth_scores": {"helpfulness": 0.85, "safety": 0.85, "factuality": 0.70},
            "error_keywords": ["1796", "Jenner invented vaccines in 1796", "synthetic error"]
        }

    return dataset

# --- Validation and Output Pipeline ---

def build_and_validate_dataset():
    raw_data = generate_mock_data()
    validated_dataset = {"task_1_easy": {}, "task_2_medium": {}, "task_3_hard": {}}
    
    error_count = 0

    print("Validating dataset...")
    
    for task_id, examples in raw_data.items():
        for example_id, data in examples.items():
            try:
                # 1. Pydantic Schema Validation (includes the math check)
                validated_example = DatasetExample(**data)
                
                # 2. Task 3 Specific Validation
                if task_id == "task_3_hard" and not validated_example.error_keywords:
                    raise ValueError("task_3_hard examples must have non-empty error_keywords")
                    
                validated_dataset[task_id][example_id] = validated_example.model_dump()
                
            except Exception as e:
                print(f"Validation failed at {task_id} / {example_id}: {e}")
                error_count += 1

    if error_count > 0:
        print(f"\nFailed with {error_count} validation errors.")
        sys.exit(1)

    # Validate counts exactly match the spec
    t1_count = len(validated_dataset["task_1_easy"])
    t2_count = len(validated_dataset["task_2_medium"])
    t3_count = len(validated_dataset["task_3_hard"])
    total = t1_count + t2_count + t3_count

    print(f"# task_1_easy:   {t1_count} examples OK")
    print(f"# task_2_medium: {t2_count} examples OK")
    print(f"# task_3_hard:   {t3_count} examples OK")
    print(f"# Total: {total} examples. No validation errors.")

    if total != 100:
        print(f"ERROR: Expected exactly 100 examples, got {total}.")
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path("data/preference_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the frozen dataset
    with open(output_path, "w") as f:
        json.dump(validated_dataset, f, indent=2)
        
    print(f"# Written to {output_path}")
    sys.exit(0)

if __name__ == "__main__":
    build_and_validate_dataset()