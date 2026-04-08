# Checklist and Instructions

Hackathon Compliance Rules — Everything You Must Follow

SECTION 1 — Submission Requirements (the hard ones)
1. Your inference script must be named exactly inference.py and placed at the repo root. Not scripts/baseline.py, not src/inference.py. The evaluator looks for this exact filename at the root. Wrong name or wrong location = automatic failure.
2. inference.py must use the OpenAI client with exactly these three environment variable names:
API_BASE_URL   — the LLM API endpoint
MODEL_NAME     — the model identifier
HF_TOKEN       — your API key
The evaluator sets these three variables before running your script. If you use TOGETHER_API_KEY or any other name, your script crashes on their machine.
3. Your environment must be deployed to a public Hugging Face Space using Docker mode. Not Gradio. Docker. The HF Space URL is what you paste into the Scaler submission dashboard.
4. HF Spaces uses port 7860, not 8000. Your Dockerfile must say EXPOSE 7860 and uvicorn must start with --port 7860. This is different from local development. Confirmed from real submission repos already deployed for this hackathon.
5. docker build && docker run must pass cleanly on a fresh machine before you submit. Test this yourself. Common failures: dataset JSON not copied into the image, wrong uvicorn start command, missing requirements.
6. Submit your HF Space URL in the Scaler dashboard before April 8. The code on GitHub alone does not count as a submission. The URL must be pasted into the dashboard. Do this early — HF Spaces takes 10–15 minutes to build on first deploy.

SECTION 2 — OpenEnv Spec Compliance
7. Implement all four endpoints: POST /reset, POST /step, GET /info, GET /state. The validate CLI checks all four. Missing any one fails the spec check.
8. /reset must return valid observation JSON with a phase field equal to "generate". Test with a raw curl before deploying.
9. /step must always return all four fields: observation, reward, done, info. Every single step response — including error responses for wrong-phase actions — must have all four fields. reward must be a structured object, not a bare float. done must be a boolean.
10. All reward values must be finite floats — never NaN, never Infinity. The validate CLI explicitly checks this. If your reward engine ever divides by zero (for example when budget_total is 0), add a guard before the division.
11. Wrong-phase actions must return HTTP 200 with an ErrorObservation, not a 500 error. The validate CLI deliberately sends wrong-phase actions to test this. Wrap your entire /step handler in try/except.
12. openenv.yaml must exist at the repo root. Required fields: name, version, description, tasks, endpoints. The validate CLI reads this file directly.
13. All Pydantic models must use typed fields. No bare dict, no Any. This is required for schema generation in /info and for the LLM scoring part of evaluation.

SECTION 3 — Task and Grader Requirements
14. You must have exactly 3 tasks — easy, medium, hard. This is a hard rubric requirement. All three must be reachable by passing task_id to /reset.
15. All graders must return scores between 0.0 and 1.0, and must be fully deterministic. No LLM calls inside graders. No network calls. No randomness. Same input always produces the same output. Add a clamp at the end of every grader: return max(0.0, min(1.0, score)).
16. The hard task must genuinely challenge frontier models. The rubric specifically asks this. Our Task 3 adversarial design satisfies it. A task that is just longer or more verbose does not count as hard.
17. Reward must provide signal throughout the episode, not just at the end. The rubric explicitly penalises sparse binary rewards. Our design awards partial reward at the judge phase and per-round improvement signal during refinement. This requirement is already satisfied in our design — don't remove the mid-episode rewards.

SECTION 4 — Code Quality
18. Python version must be 3.10, 3.11, or 3.12. Use FROM python:3.11-slim in Dockerfile.
19. Never hardcode any API key anywhere in your codebase. Not in inference.py, not in any config file, not in the Dockerfile. Everything comes from environment variables. Add .env to .gitignore before your first commit.
20. The dataset JSON must be baked into the Docker image at build time. Use COPY data/ ./data/ in your Dockerfile. The server must start with zero network calls. If you download data at startup it will fail on HF Spaces.
21. The tools/ folder must never be imported by the server. Dataset generation scripts are one-time build tools. If server.py or environment.py imports anything from tools/, that is a design error.
22. Write unit tests for your graders and reward engine. The rubric scores code quality. A submission with no tests will score lower. Minimum: one test per grader with a known input/output pair.

SECTION 5 — inference.py Specific Rules
23. temperature=0 is mandatory. The rubric requires a reproducible baseline score. Two runs on the same machine must produce identical output. temperature=0 is the only way to guarantee this.
24. inference.py must run all 3 tasks and print a score for each. If it only runs one task the other two are scored as zero.
25. Wrap every LLM response parse in try/except. LLMs sometimes return invalid JSON. On failure: log the error, record reward 0.0 for that episode, continue to the next task. A crashed script produces no scores at all.
26. The LLM system prompt must instruct JSON-only output. Include exactly: "Respond with ONLY valid JSON. No explanation, no markdown, no preamble." Without this the LLM returns prose and your parser breaks.
27. The default ENV_BASE_URL must point to your live HF Space, not localhost. The evaluator runs your script against your deployed environment. If the default is localhost it will fail on their machine unless they set the variable manually.

SECTION 6 — Non-Technical Rules
28. Both team members must register individually on the Scaler dashboard. One person registering is not enough. The team is linked through the dashboard.
29. Your code becomes an open-source contribution under BSD-3-Clause license. Do not include any proprietary data, code, or anything you do not have the right to open-source. The HH-RLHF dataset is publicly licensed — that is fine.
30. Join the official Discord: discord.gg/Dedhy5pkWD. Last-minute rule changes and technical clarifications are posted there first. Not being on Discord means you may miss critical updates before the deadline.
31. Watch the Round 1 bootcamp recording before submitting. Ben Burtenshaw from Hugging Face walks through a real submission. Watch at: youtube.com/live/kkCNMz0Ptd8. It is 1–2 hours and will prevent common mistakes.

SECTION 7 — README Requirements
32. README must explain what the environment simulates and why it matters — one clear paragraph. This is the first thing evaluators read and it directly maps to the 30% real-world utility score.
33. README must describe all 3 tasks with difficulty rationale. For each task: what the agent does, why it is that difficulty, what a good score looks like vs a bad score.
34. README must include the actual output from running inference.py. Paste the score table. This proves your script works end to end.
35. README must include your HF Space URL prominently near the top. Make it easy for the evaluator to find without reading the whole file.

SECTION 8 — The Night Before Checklist
Run these in order the night before you submit:
36. docker build -t tv-preference-env . && docker run -p 7860:7860 tv-preference-env — must complete with no errors.
37. curl -X POST http://localhost:7860/reset — must return valid JSON.
38. pip install openenv-core && openenv validate https://your-space.hf.space — must exit code 0.
39. Run inference.py twice and confirm outputs are identical — reproducibility check.
40. ls inference.py from repo root — must exist. This is the single most common submission failure.
41. git log --all -p | grep -i "api_key" — must return nothing that is not an os.environ reference.
42. Paste the HF Space URL into the Scaler dashboard — the submission is not complete until you do this.