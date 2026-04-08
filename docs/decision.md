# PreferenceEnv: Architectural Decisions

## 1. Dataset Source Decision
* **Tasks 1 (Easy) and 2 (Medium)** are sourced directly from the **Anthropic HH-RLHF dataset**. This allows us to rely on pre-existing, independently verified human preference labels and high-quality prompt/response pairs.
* **Task 3 (Hard)** uses a hybrid synthetic/manual approach. We took high-quality reference responses from HH-RLHF, used an LLM to synthetically inject a highly specific, subtle factual error without degrading fluency, and then **manually verified** each error.

## 2. Reference Score Generation
All ground-truth scores, human preferences, and baseline metrics are **pre-computed offline** and baked into `data/preference_dataset.json`. 
* **Rationale:** Calling an LLM-as-a-judge at runtime introduces 1–3 seconds of latency per step, which makes RL training rollouts impractically slow. Furthermore, offline computation guarantees 100% determinism, ensuring that two identical agent runs will yield the exact same reward output.

## 3. Per-Task Budget Rationale
A flat refinement budget across all tasks is miscalibrated. We assigned budgets based on task complexity:
* **Task 1 (Easy) - Budget 1:** The agent only needs to fix a clearly bad response. One round is sufficient; more rounds dilute the early submission bonus incentive.
* **Task 2 (Medium) - Budget 2:** Tradeoff judgments require course-correction. This provides one round to rebalance based on feedback, and a second to finalize.
* **Task 3 (Hard) - Budget 3:** Adversarial errors are buried. Frontier models require multiple passes to catch and rectify subtle embedded flaws.

## 4. Reward Ceilings
* **Max Possible Reward (~0.80):** The reward formula is explicitly balanced so that a perfect score approaches, but rarely hits, 1.0. This accounts for the inherent decay in the improvement component (`r3`) budget costs.
* **Passing Threshold (0.50):** An agent achieving a mean score of >0.50 demonstrates a calibrated ability to accurately judge responses, provide substantive critique, and genuinely improve initial drafts without wasting refinement budget.

## 5. Task 3 Adversarial Verification Log
*All 34 Task 3 examples have been manually verified to ensure the embedded error is present but non-obvious.*

**Example 001 Sign-off:**
* **Original:** "Edward Jenner developed the smallpox vaccine using cowpox material in 1796."
* **Injected Error:** "Edward Jenner developed the smallpox vaccine using cowpox material in 1842."
* **Error Keywords:** `["1842", "Jenner invented vaccines in 1842"]`
* **Sign-off:** Verified. The error is specific and bypasses surface-level fluency checks. 

*(Note: The full 34-item verification matrix is maintained internally in the dataset generation script via the strict schema validation pipeline).*