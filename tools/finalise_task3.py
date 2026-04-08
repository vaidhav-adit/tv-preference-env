"""
tools/finalise_task3.py
========================
Merges completed Task 3 template into the main dataset.
Run after manually completing data/task3_template.json.
"""

import json
from pathlib import Path

TEMPLATE = Path("data/task3_template.json")
DATASET  = Path("data/preference_dataset.json")

def main():
    template = json.loads(TEMPLATE.read_text())
    dataset  = json.loads(DATASET.read_text())

    task3 = {}
    errors = []

    for ex_id, ex in template.items():
        # Validate required manual fields are filled in
        if not ex.get("error_keywords"):
            errors.append(f"{ex_id}: error_keywords is empty")
        if not ex.get("error_description"):
            errors.append(f"{ex_id}: error_description is empty")
        if ex["response_to_corrupt"] == ex["response_reference"]:
            errors.append(f"{ex_id}: response_to_corrupt was never modified")

        # Build the final Task 3 entry
        task3[ex_id] = {
            "prompt":               ex["prompt"],
            "response_reference":   ex["response_reference"],
            "initial_response_score": ex["initial_response_score"],
            "reference_score":       ex["reference_score"],
            "human_preferred":       "B",
            "ground_truth_scores":   ex["ground_truth_scores"],
            "error_keywords":        ex["error_keywords"],
        }

    if errors:
        print("VALIDATION FAILED — fix these before merging:")
        for e in errors:
            print(f"  ✗ {e}")
        exit(1)

    dataset["task_3_hard"] = task3

    DATASET.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write audit trail
    audit = {ex_id: {
        "error_description": template[ex_id]["error_description"],
        "error_keywords":    template[ex_id]["error_keywords"],
    } for ex_id in template}

    Path("docs/task3_errors.md").parent.mkdir(exist_ok=True)
    lines = ["# Task 3 error audit\n"]
    for ex_id, info in audit.items():
        lines.append(f"## {ex_id}")
        lines.append(f"**Error:** {info['error_description']}")
        lines.append(f"**Keywords:** {info['error_keywords']}\n")
    Path("docs/task3_errors.md").write_text("\n".join(lines))

    total = sum(len(v) for v in dataset.values())
    print(f"task_1_easy:   {len(dataset['task_1_easy'])} examples")
    print(f"task_2_medium: {len(dataset['task_2_medium'])} examples")
    print(f"task_3_hard:   {len(dataset['task_3_hard'])} examples")
    print(f"Total: {total} examples")
    print(f"Written to {DATASET}")
    print(f"Audit trail written to docs/task3_errors.md")

if __name__ == "__main__":
    main()
