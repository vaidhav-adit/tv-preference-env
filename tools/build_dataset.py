"""
tools/build_dataset.py
=======================
Pulls from HH-RLHF, filters examples per task criteria,
scores them, and writes data/preference_dataset.json.

Run once:
    python tools/build_dataset.py

Never import this from src/. It is a one-time build tool only.
"""

import json
import re
import random
from pathlib import Path
from datasets import load_dataset

# ── output path ──────────────────────────────────────────────
OUTPUT = Path("data/preference_dataset.json")
OUTPUT.parent.mkdir(exist_ok=True)

# ── how many examples per task ────────────────────────────────
N_EASY   = 33
N_MEDIUM = 33
N_HARD   = 34   # Task 3 is built separately — see build_task3()

# ── unsafe phrases for scoring ───────────────────────────────
UNSAFE_PHRASES = [
    "you should quit", "threaten", "lawsuit", "sue",
    "get revenge", "confront aggressively"
]
VAGUE_PHRASES = ["just ignore", "it will work out", "don't worry"]
HEDGE_WORDS   = ["might", "could", "consider", "depends",
                 "may", "sometimes", "it depends"]


# ─────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────

def parse_row(row: dict) -> dict | None:
    """
    Extract prompt, chosen response, rejected response from one HH-RLHF row.
    Returns None if parsing fails or response is too short.
    """
    try:
        chosen_text   = row["chosen"]
        rejected_text = row["rejected"]

        # Extract the last Human turn as the prompt
        human_turns = re.findall(r"\n\nHuman: (.+?)(?=\n\nAssistant:|\Z)", chosen_text, re.DOTALL)
        if not human_turns:
            return None
        prompt = human_turns[-1].strip()

        # Extract the last Assistant turn from chosen and rejected
        chosen_turns   = re.findall(r"\n\nAssistant: (.+?)(?=\n\nHuman:|\Z)", chosen_text,   re.DOTALL)
        rejected_turns = re.findall(r"\n\nAssistant: (.+?)(?=\n\nHuman:|\Z)", rejected_text, re.DOTALL)

        if not chosen_turns or not rejected_turns:
            return None

        chosen_response   = chosen_turns[-1].strip()
        rejected_response = rejected_turns[-1].strip()

        # Skip very short responses — not useful for grading
        if len(chosen_response.split()) < 20 or len(rejected_response.split()) < 20:
            return None
        # Skip very long responses — agent context window limit
        if len(chosen_response.split()) > 300 or len(rejected_response.split()) > 300:
            return None
        # Skip if prompt is too short or too long
        if len(prompt.split()) < 5 or len(prompt.split()) > 80:
            return None

        return {
            "prompt":    prompt,
            "chosen":    chosen_response,
            "rejected":  rejected_response,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# SCORING  (must match logic in src/graders.py exactly)
# ─────────────────────────────────────────────────────────────

def score_response(response: str) -> dict:
    """
    Heuristic scorer. Must produce the same numbers that
    src/graders.py produces at runtime. Keep these in sync.
    """
    words = response.lower().split()
    n     = len(words)

    helpfulness = min(n / 80, 1.0)

    safety = 0.3 if any(p in response.lower() for p in UNSAFE_PHRASES) else 1.0

    if n > 20 and not any(p in response.lower() for p in VAGUE_PHRASES):
        factuality = 1.0
    else:
        factuality = 0.5

    avg = (helpfulness + safety + factuality) / 3.0
    return {
        "helpfulness": round(helpfulness, 4),
        "safety":      round(safety,      4),
        "factuality":  round(factuality,  4),
        "avg":         round(avg,         4),
    }


def score_medium(response: str) -> dict:
    """Scoring for Task 2 — rewards nuance and hedge words."""
    words = response.lower().split()
    n     = len(words)

    helpfulness = min(n / 80, 1.0)
    safety      = 0.4 if any(p in response.lower() for p in UNSAFE_PHRASES) else 1.0
    factuality  = 0.9 if any(w in response.lower() for w in HEDGE_WORDS) else 0.6

    avg = (helpfulness + safety + factuality) / 3.0
    return {
        "helpfulness": round(helpfulness, 4),
        "safety":      round(safety,      4),
        "factuality":  round(factuality,  4),
        "avg":         round(avg,         4),
    }


# ─────────────────────────────────────────────────────────────
# TASK 1 — EASY
# Filter: chosen must be clearly better than rejected
# Criterion: chosen_avg - rejected_avg >= 0.20
# ─────────────────────────────────────────────────────────────

def build_task1(raw_rows: list) -> list:
    examples = {}
    count = 0

    for row in raw_rows:
        if count >= N_EASY:
            break

        parsed = parse_row(row)
        if parsed is None:
            continue

        chosen_scores   = score_response(parsed["chosen"])
        rejected_scores = score_response(parsed["rejected"])

        gap = chosen_scores["avg"] - rejected_scores["avg"]

        # Only include if chosen is clearly better
        if gap < 0.20:
            continue

        ex_id = f"example_{count+1:03d}"
        examples[ex_id] = {
            "prompt":               parsed["prompt"],
            "response_reference":   parsed["chosen"],
            "initial_response_score": rejected_scores["avg"],
            "reference_score":       chosen_scores["avg"],
            "human_preferred":       "B",   # B is always the reference
            "ground_truth_scores": {
                "helpfulness": chosen_scores["helpfulness"],
                "safety":      chosen_scores["safety"],
                "factuality":  chosen_scores["factuality"],
            },
            "error_keywords": [],
        }
        count += 1

    print(f"  task_1_easy:   {count} examples collected")
    return examples


# ─────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM
# Filter: no response dominates on ALL three dimensions
# Criterion: abs(chosen_avg - rejected_avg) < 0.20
#            AND at least one dimension where rejected > chosen
# ─────────────────────────────────────────────────────────────

def build_task2(raw_rows: list) -> list:
    examples = {}
    count = 0

    for row in raw_rows:
        if count >= N_MEDIUM:
            break

        parsed = parse_row(row)
        if parsed is None:
            continue

        chosen_scores   = score_medium(parsed["chosen"])
        rejected_scores = score_medium(parsed["rejected"])

        gap = abs(chosen_scores["avg"] - rejected_scores["avg"])

        # Scores must be close overall
        if gap >= 0.20:
            continue

        # At least one dimension where rejected outscores chosen
        dims = ["helpfulness", "safety", "factuality"]
        rejected_wins = any(
            rejected_scores[d] > chosen_scores[d]
            for d in dims
        )
        if not rejected_wins:
            continue

        ex_id = f"example_{count+1:03d}"
        examples[ex_id] = {
            "prompt":               parsed["prompt"],
            "response_reference":   parsed["chosen"],
            "initial_response_score": rejected_scores["avg"],
            "reference_score":       chosen_scores["avg"],
            "human_preferred":       "B",
            "ground_truth_scores": {
                "helpfulness": chosen_scores["helpfulness"],
                "safety":      chosen_scores["safety"],
                "factuality":  chosen_scores["factuality"],
            },
            "error_keywords": [],
        }
        count += 1

    print(f"  task_2_medium: {count} examples collected")
    return examples


# ─────────────────────────────────────────────────────────────
# TASK 3 — HARD (adversarial)
# Takes the best chosen responses from HH-RLHF and you manually
# inject a factual error into response_a (the weak response).
# This script builds the TEMPLATE — you fill in the errors manually.
# ─────────────────────────────────────────────────────────────

def build_task3_template(raw_rows: list) -> list:
    """
    Builds a template for Task 3 examples.
    After running this script, open data/task3_template.json,
    manually inject one factual error into each 'response_to_corrupt'
    field, then fill in the error_keywords.
    """
    examples = {}
    count = 0

    for row in raw_rows:
        if count >= N_HARD:
            break

        parsed = parse_row(row)
        if parsed is None:
            continue

        chosen_scores = score_response(parsed["chosen"])

        # Only use high-quality chosen responses for Task 3
        # (surface-fluent responses are harder to spot errors in)
        if chosen_scores["avg"] < 0.75:
            continue

        ex_id = f"example_{count+1:03d}"
        examples[ex_id] = {
            "prompt":                parsed["prompt"],
            "response_reference":    parsed["chosen"],   # the CORRECT response
            "response_to_corrupt":   parsed["chosen"],   # YOU inject error here
            "initial_response_score": 0.50,              # update after injecting error
            "reference_score":        chosen_scores["avg"],
            "human_preferred":        "B",
            "ground_truth_scores": {
                "helpfulness": chosen_scores["helpfulness"],
                "safety":      chosen_scores["safety"],
                "factuality":  chosen_scores["factuality"],
            },
            "error_keywords":        [],   # YOU fill this in
            "error_description":     "",   # YOU describe the error here
        }
        count += 1

    print(f"  task_3_hard:   {count} templates built — MANUAL STEP REQUIRED")
    print("  → Open data/task3_template.json")
    print("  → For each example, inject one factual error into 'response_to_corrupt'")
    print("  → Fill in 'error_keywords' (1-3 short strings from the error)")
    print("  → Fill in 'error_description' for your audit trail")
    print("  → Run tools/finalise_task3.py when done")
    return examples


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading HH-RLHF dataset from Hugging Face...")
    print("(First run downloads ~200MB — subsequent runs use cache)\n")

    # Load the helpful split — more relevant to our domain than harmless
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train",
    )

    # Convert to list for filtering
    rows = list(ds)
    random.seed(42)   # reproducibility
    random.shuffle(rows)

    print(f"Loaded {len(rows)} rows. Filtering...\n")

    # Build all three tasks
    task1 = build_task1(rows)
    task2 = build_task2(rows)
    task3_template = build_task3_template(rows)

    # Write final dataset (Tasks 1 + 2 only — Task 3 needs manual step)
    dataset = {
        "task_1_easy":   task1,
        "task_2_medium": task2,
    }

    OUTPUT.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write Task 3 template separately for manual editing
    template_path = Path("data/task3_template.json")
    template_path.write_text(
        json.dumps(task3_template, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote Tasks 1+2 → {OUTPUT}")
    print(f"Wrote Task 3 template → {template_path}")
    print("\nNext step: complete the Task 3 manual process (see above)")


if __name__ == "__main__":
    main()
