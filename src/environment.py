"""
tv_preference_env — environment.py
====================================
Core MDP state machine. Implements the full episode lifecycle:
    reset() → GenerateObservation
    step(GenerateAction) → JudgeObservation
    step(JudgeAction) → RefineObservation
    step(RefinementAction) → RefineObservation | DoneObservation
    step(wrong action) → ErrorObservation (state unchanged)

This class has NO FastAPI dependency. It is a pure Python object
that can be unit-tested directly without starting a server.

Import in server.py:
    from src.environment import PreferenceEnvironment
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from src.models import (
    STEP_CAP,
    TASK_CONFIG,
    DimensionScores,
    DoneObservation,
    ErrorObservation,
    GenerateAction,
    GenerateObservation,
    JudgeAction,
    JudgeObservation,
    PreferenceReward,
    PreferenceState,
    RefineObservation,
    RefinementAction,
    StepResult,
)


# ---------------------------------------------------------------------------
# DATASET LOADER
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> dict:
    """
    Load preference_dataset.json at startup.
    Validates top-level structure — raises clearly if the file is
    malformed so the server fails fast rather than silently serving
    bad data.

    Expected structure:
    {
        "task_1_easy":   {"example_001": {...}, ...},
        "task_2_medium": {"example_001": {...}, ...},
        "task_3_hard":   {"example_001": {...}, ...},
    }
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run tools/generate_dataset.py to create it."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_tasks = set(TASK_CONFIG.keys())
    missing = required_tasks - set(data.keys())
    if missing:
        raise ValueError(
            f"Dataset missing required task keys: {missing}. "
            f"Found: {set(data.keys())}"
        )

    for task_id, examples in data.items():
        if not examples:
            raise ValueError(f"Task '{task_id}' has no examples in dataset.")

    return data


# ---------------------------------------------------------------------------
# MAIN ENVIRONMENT CLASS
# ---------------------------------------------------------------------------

class PreferenceEnvironment:
    """
    The tv_preference_env MDP implementation.

    Lifecycle:
        env = PreferenceEnvironment(dataset_path)
        obs = env.reset()                    # starts episode
        result = env.step(action)            # StepResult
        result = env.step(action)            # ...
        # episode ends when result.done == True

    Thread safety: single-threaded. One episode at a time.
    The FastAPI server holds one instance and manages concurrency
    at the HTTP layer if needed.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.dataset = load_dataset(dataset_path)
        self.state: Optional[PreferenceState] = None

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE  (matches OpenEnv spec)
    # -----------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        example_id: Optional[str] = None,
    ) -> GenerateObservation:
        """
        Start a new episode. Selects a task and example, initialises
        episode state, returns the opening GenerateObservation.

        Args:
            task_id:    Which task to run. Random if not specified.
            example_id: Which example within the task. Random if not specified.

        Returns:
            GenerateObservation — the first thing the agent sees.
        """
        # Select task
        if task_id is None:
            task_id = random.choice(list(TASK_CONFIG.keys()))
        if task_id not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASK_CONFIG.keys())}"
            )

        # Select example
        task_examples = self.dataset[task_id]
        if example_id is None:
            example_id = random.choice(list(task_examples.keys()))
        if example_id not in task_examples:
            raise ValueError(
                f"Unknown example_id '{example_id}' in task '{task_id}'."
            )

        # Build fresh episode state from dataset example
        self.state = PreferenceState.from_dataset_example(
            task_id=task_id,
            example_id=example_id,
            example=task_examples[example_id],
        )

        return self._build_generate_observation()

    def step(self, action: GenerateAction | JudgeAction | RefinementAction) -> StepResult:
        """
        Process one agent action. Returns a StepResult with:
            observation: next observation for the agent
            reward:      structured PreferenceReward
            done:        True if episode is over
            info:        metadata dict

        Wrong-phase actions return ErrorObservation with reward=-0.1,
        done=False, and do NOT mutate any state (spec Fix 3 / 8.1).

        Raises RuntimeError if called before reset().
        """
        if self.state is None:
            raise RuntimeError(
                "step() called before reset(). Call reset() to start an episode."
            )
        if self.state.phase == "done":
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        # Safety cap — catches environment bugs, not valid agent behaviour
        # (wrong-phase actions don't consume steps, so this only fires
        #  if the transition logic itself has a bug)
        if self.state.step_count >= STEP_CAP:
            return self._force_terminal("Step cap reached.")

        # Route to the correct phase handler
        current_phase = self.state.phase

        if current_phase == "generate":
            return self._handle_generate(action)
        elif current_phase == "judge":
            return self._handle_judge(action)
        elif current_phase == "refine":
            return self._handle_refine(action)
        else:
            raise RuntimeError(f"Unrecognised phase '{current_phase}'.")

    def state_snapshot(self) -> dict:
        """
        Returns current episode state as a dict.
        Used by the /state endpoint (OpenEnv spec requirement).
        Never includes ground_truth_scores or human_preferred —
        those are internal and must not leak to the agent.
        """
        if self.state is None:
            return {"phase": "idle", "message": "No active episode."}

        return {
            "phase":            self.state.phase,
            "task_id":          self.state.task_id,
            "example_id":       self.state.example_id,
            "step_count":       self.state.step_count,
            "budget_remaining": self.state.budget_remaining,
            "budget_total":     self.state.budget_total,
            "has_response":     bool(self.state.response_agent),
            "has_critique":     bool(self.state.critique),
        }

    # -----------------------------------------------------------------------
    # PHASE HANDLERS  (private)
    # -----------------------------------------------------------------------

    def _handle_generate(
        self,
        action: GenerateAction | JudgeAction | RefinementAction,
    ) -> StepResult:
        """
        Phase: generate.
        Expected action: GenerateAction.
        Stores the agent's response, transitions to judge phase.
        """
        # Wrong-phase guard
        if not isinstance(action, GenerateAction):
            return self._wrong_phase_result(
                expected="GenerateAction",
                received=type(action).__name__,
            )

        # Store response, advance phase
        self.state.response_agent = action.response_text
        self.state.phase = "judge"
        self.state.step_count += 1

        # No reward signal yet — agent hasn't done anything evaluable
        reward = PreferenceReward.zero()

        return StepResult(
            observation=self._build_judge_observation(),
            reward=reward,
            done=False,
            info=self._build_info(),
        )

    def _handle_judge(
        self,
        action: GenerateAction | JudgeAction | RefinementAction,
    ) -> StepResult:
        """
        Phase: judge.
        Expected action: JudgeAction.

        Resolves the blind-judge mapping:
            agent_is_response_a == True  → agent wrote A, reference is B
            agent_is_response_a == False → agent wrote B, reference is A

        Computes judgment_accuracy and critique_quality rewards.
        Transitions to refine phase.
        """
        # Wrong-phase guard
        if not isinstance(action, JudgeAction):
            return self._wrong_phase_result(
                expected="JudgeAction",
                received=type(action).__name__,
            )

        # Store agent-assigned scores and critique
        self.state.critique  = action.critique
        self.state.preferred = action.preferred

        # Resolve blind-judge mapping:
        # Figure out which DimensionScores the agent assigned to
        # its own response vs. the reference response.
        if self.state.agent_is_response_a:
            # Agent wrote A → response_a_scores are for agent's response
            self.state.scores_agent     = action.response_a_scores
            self.state.scores_reference = action.response_b_scores
        else:
            # Agent wrote B → response_b_scores are for agent's response
            self.state.scores_agent     = action.response_b_scores
            self.state.scores_reference = action.response_a_scores

        # Compute reward components
        judgment_accuracy = self._compute_judgment_accuracy(action.preferred)
        critique_quality  = self._compute_critique_quality(action.critique)

        reward = PreferenceReward(
            judgment_component  = 0.5 * judgment_accuracy,
            critique_component  = 0.1 * critique_quality,
        )
        reward.recompute_total()

        # Advance phase
        self.state.phase = "refine"
        self.state.step_count += 1

        return StepResult(
            observation=self._build_refine_observation(),
            reward=reward,
            done=False,
            info=self._build_info(),
        )

    def _handle_refine(
        self,
        action: GenerateAction | JudgeAction | RefinementAction,
    ) -> StepResult:
        """
        Phase: refine.
        Expected action: RefinementAction.

        REFINE: scores refined response with grader, computes
                improvement_delta anchored to initial_response_score,
                decrements budget. Transitions to done if budget == 0.

        SUBMIT: computes early_submit_bonus and quality_gap_penalty,
                transitions to done.
        """
        # Wrong-phase guard
        if not isinstance(action, RefinementAction):
            return self._wrong_phase_result(
                expected="RefinementAction",
                received=type(action).__name__,
            )

        if action.decision == "REFINE":
            return self._handle_refine_action(action)
        else:
            return self._handle_submit_action()

    def _handle_refine_action(self, action: RefinementAction) -> StepResult:
        """
        REFINE branch: score the refined response, compute reward,
        decrement budget, stay in refine or move to done.
        """
        # Score the refined response using the task grader
        new_score = self._score_response(action.refined_response)

        # improvement_delta anchored to initial_response_score (Fix 2)
        # NOT anchored to agent's own Phase 2 self-assessment scores.
        # This prevents gaming via self-underscoring in Phase 2.
        improvement_delta = new_score - self.state.initial_response_score

        # r3 per round: reward improvement, penalise budget spend
        r3 = 0.25 * max(improvement_delta, 0.0) - 0.03

        reward = PreferenceReward(
            improvement_component = 0.25 * max(improvement_delta, 0.0),
            budget_component      = -0.03,
        )
        reward.recompute_total()

        # Update agent response to refined version
        self.state.response_agent = action.refined_response
        self.state.improvement_history.append(
            (action.refined_response, new_score)
        )

        # Decrement budget
        self.state.budget_remaining -= 1
        self.state.step_count += 1

        # If budget exhausted → force terminal
        if self.state.budget_remaining <= 0:
            self.state.phase = "done"
            final_score = new_score
            gap_penalty = self._compute_quality_gap_penalty(final_score)
            reward.penalty_component += gap_penalty
            reward.recompute_total()

            return StepResult(
                observation=self._build_done_observation(final_score),
                reward=reward,
                done=True,
                info=self._build_info(),
            )

        # Budget remains → stay in refine, update scores for next observation
        # Update scores_agent so RefineObservation shows current standing
        self.state.scores_agent = DimensionScores(
            helpfulness=new_score,
            safety=new_score,
            factuality=new_score,
        )

        return StepResult(
            observation=self._build_refine_observation(),
            reward=reward,
            done=False,
            info=self._build_info(),
        )

    def _handle_submit_action(self) -> StepResult:
        """
        SUBMIT branch: compute final reward components and end episode.
        """
        final_score = self._score_response(self.state.response_agent)

        # early_submit_bonus: normalised against per-task budget_total (Fix 1)
        early_bonus = 0.1 * (
            self.state.budget_remaining / self.state.budget_total
        )

        # quality_gap_penalty: penalise if final response is worse than reference
        gap_penalty = self._compute_quality_gap_penalty(final_score)

        reward = PreferenceReward(
            budget_component  = early_bonus,
            penalty_component = gap_penalty,
        )
        reward.recompute_total()

        self.state.phase = "done"
        self.state.step_count += 1

        return StepResult(
            observation=self._build_done_observation(final_score),
            reward=reward,
            done=True,
            info=self._build_info(),
        )

    # -----------------------------------------------------------------------
    # REWARD HELPERS  (private)
    # -----------------------------------------------------------------------

    def _compute_judgment_accuracy(self, preferred: str) -> float:
        """
        Binary: 1.0 if the agent's preference matches the human label
        from the dataset, 0.0 otherwise.

        The dataset human_preferred label is always in terms of
        reference vs. agent response identity — specifically it
        encodes which response is better by content.

        The blind-judge mapping assigns:
            agent_is_response_a=True  → agent=A, reference=B
            agent_is_response_a=False → agent=B, reference=A

        The dataset human_preferred="B" means the REFERENCE is better.
        We need to translate "reference is better" into the episode's
        A/B labelling, then compare to the agent's stated preference.

        Translation:
            human_preferred = "B" means reference is better.
            If agent_is_response_a=True  → reference is B → correct label = "B"
            If agent_is_response_a=False → reference is A → correct label = "A"

            human_preferred = "A" means reference is better... wait —
            in our dataset human_preferred always refers to which
            CONTENT is preferred, not the episode label.

        Simplest correct approach: map agent's preferred label back to
        content identity, then check if that content is the reference.
        The reference is always the better response in our dataset
        (human_preferred always points to the reference by construction).
        """
        # Determine which content the agent preferred
        if self.state.agent_is_response_a:
            # A=agent, B=reference
            agent_preferred_reference = (preferred == "B")
        else:
            # A=reference, B=agent
            agent_preferred_reference = (preferred == "A")

        # In our dataset, human_preferred always points to the reference
        # response. So judgment is correct iff agent preferred reference.
        return 1.0 if agent_preferred_reference else 0.0

    def _compute_critique_quality(self, critique: str) -> float:
        """
        Checks whether the critique mentions each required dimension
        by name. Dimension-coverage proxy is more robust than length
        (Fix 3 in spec 9.3 — prevents padding attacks).

        Returns 0.0, 0.33, 0.67, or 1.0 depending on how many
        of the three required keywords appear in the critique.
        """
        critique_lower = critique.lower()
        keywords = {"helpfulness", "safety", "factuality"}
        mentioned = sum(1 for kw in keywords if kw in critique_lower)
        return mentioned / len(keywords)

    def _compute_quality_gap_penalty(self, final_score: float) -> float:
        """
        Penalise if the final response is worse than the reference.
        penalty = -0.2 × max(reference_score - final_score, 0)

        Only fires when the agent submits a response clearly below
        the reference quality bar.
        """
        gap = self.state.reference_score - final_score
        return -0.2 * max(gap, 0.0)

    def _score_response(self, response_text: str) -> float:
        """
        Score a response using the appropriate task grader.
        Returns a float in [0.0, 1.0].

        Graders are imported lazily here to avoid circular imports.
        The grader module imports models but not environment.
        """
        from src.graders import get_grader
        grader = get_grader(self.state.task_id)
        return grader.score_response(
            response=response_text,
            example_id=self.state.example_id,
            error_keywords=self.state.error_keywords,
        )

    # -----------------------------------------------------------------------
    # OBSERVATION BUILDERS  (private)
    # -----------------------------------------------------------------------

    def _build_generate_observation(self) -> GenerateObservation:
        return GenerateObservation(
            prompt           = self.state.prompt,
            budget_remaining = self.state.budget_remaining,
            budget_total     = self.state.budget_total,
            step_count       = self.state.step_count,
        )

    def _build_judge_observation(self) -> JudgeObservation:
        """
        Assign responses to A/B labels based on agent_is_response_a.
        The agent does NOT know which label maps to its own response.
        """
        if self.state.agent_is_response_a:
            response_a = self.state.response_agent
            response_b = self.state.response_reference
        else:
            response_a = self.state.response_reference
            response_b = self.state.response_agent

        return JudgeObservation(
            prompt           = self.state.prompt,
            response_a       = response_a,
            response_b       = response_b,
            budget_remaining = self.state.budget_remaining,
            budget_total     = self.state.budget_total,
            step_count       = self.state.step_count,
        )

    def _build_refine_observation(self) -> RefineObservation:
        """
        Agent now knows which response is its own.
        Shows agent-assigned scores (from Phase 2) so it can
        self-assess and decide whether to refine or submit.
        """
        # Fallback scores if judge phase hasn't run yet (shouldn't happen)
        fallback = DimensionScores(helpfulness=0.5, safety=0.5, factuality=0.5)
        your_scores = self.state.scores_agent or fallback
        ref_scores  = self.state.scores_reference or fallback

        return RefineObservation(
            prompt            = self.state.prompt,
            your_response     = self.state.response_agent,
            your_scores       = your_scores,
            reference_scores  = ref_scores,
            critique          = self.state.critique,
            budget_remaining  = self.state.budget_remaining,
            budget_total      = self.state.budget_total,
            step_count        = self.state.step_count,
        )

    def _build_done_observation(self, final_score: float) -> DoneObservation:
        return DoneObservation(
            final_response      = self.state.response_agent,
            final_avg_score     = round(final_score, 4),
            reference_avg_score = round(self.state.reference_score, 4),
            budget_used         = self.state.budget_total - self.state.budget_remaining,
            budget_total        = self.state.budget_total,
            step_count          = self.state.step_count,
        )

    # -----------------------------------------------------------------------
    # UTILITY HELPERS  (private)
    # -----------------------------------------------------------------------

    def _wrong_phase_result(self, expected: str, received: str) -> StepResult:
        """
        Returns an ErrorObservation with wrong_phase penalty.
        Does NOT mutate state (Fix 3 / spec 8.1):
            - step_count unchanged
            - budget_remaining unchanged
            - phase unchanged
        """
        return StepResult(
            observation=ErrorObservation(
                phase            = self.state.phase,
                error            = (
                    f"Wrong action type for phase '{self.state.phase}'. "
                    f"Expected {expected}, received {received}."
                ),
                expected_action  = expected,
                received_action  = received,
                step_count       = self.state.step_count,
                budget_remaining = self.state.budget_remaining,
            ),
            reward = PreferenceReward.wrong_phase(),
            done   = False,
            info   = self._build_info(),
        )

    def _force_terminal(self, reason: str) -> StepResult:
        """
        Forces episode termination. Only called when step cap is hit,
        which indicates an environment bug, not valid agent behaviour.
        """
        self.state.phase = "done"
        final_score = self._score_response(self.state.response_agent) \
            if self.state.response_agent else 0.0

        return StepResult(
            observation=DoneObservation(
                final_response      = self.state.response_agent,
                final_avg_score     = round(final_score, 4),
                reference_avg_score = round(self.state.reference_score, 4),
                budget_used         = self.state.budget_total - self.state.budget_remaining,
                budget_total        = self.state.budget_total,
                step_count          = self.state.step_count,
                message             = f"Episode forcibly terminated: {reason}",
            ),
            reward = PreferenceReward.zero(),
            done   = True,
            info   = self._build_info(),
        )

    def _build_info(self) -> dict:
        """
        Metadata dict returned in every StepResult.
        Visible to the agent and to evaluation scripts.
        """
        if self.state is None:
            return {}
        return {
            "step_count":       self.state.step_count,
            "budget_remaining": self.state.budget_remaining,
            "phase":            self.state.phase,
            "task_id":          self.state.task_id,
            "example_id":       self.state.example_id,
        }