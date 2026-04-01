---
id: s32
number: "3.2"
title: "Agent Evaluation & Testing"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-evaldim
    title: "Eval Dimensions"
  - id: t-evaltec
    title: "Techniques"
  - id: t-bench
    title: "Benchmarks"
---

### Topic: Evaluation Dimensions {#t-evaldim}

Evaluation is the hardest unsolved problem in agentic AI. Unlike a classification model where you compute accuracy on a held-out test set, agent evaluation involves sequences of decisions, tools, and external effects that do not reduce to a single number. You need to measure across multiple dimensions simultaneously — and accept that no single metric captures agent quality.

The five dimensions below form a minimum viable evaluation framework. Every production agent should have at least one test case per dimension before deployment.

- **Task completion rate:** Did the agent accomplish the goal? Binary pass/fail or partial credit scoring.
- **Trajectory efficiency:** Did the agent take an optimal path? Measure: actual_steps / optimal_steps.
- **Groundedness:** Did the agent hallucinate tool calls, invent file names, or fabricate information?
- **Cost and latency:** Total tokens, number of LLM calls, wall-clock time.
- **Safety:** Policy violations, data leakage events, irreversible destructive actions.

#### Trajectory Efficiency Scoring

A correct final answer achieved through an inefficient or dangerous path should score differently from one achieved optimally. Trajectory evaluation compares the agent's actual sequence of tool calls against a human-annotated reference trace.

```python
from dataclasses import dataclass

@dataclass
class TrajectoryScore:
    efficiency_ratio: float  # optimal_steps / actual_steps (1.0 = perfect)
    unnecessary_calls: int   # calls that did not contribute to the result
    loop_count: int          # same tool+args appearing 2+ times
    followed_expected_order: bool

def score_trajectory(actual: list[dict], reference: list[dict]) -> TrajectoryScore:
    optimal = len(reference)
    actual_n = len(actual)

    # Detect loops
    seen, loops = set(), 0
    for c in actual:
        key = (c["name"], str(sorted(c["args"].items())))
        if key in seen: loops += 1
        seen.add(key)

    return TrajectoryScore(
        efficiency_ratio=optimal / max(actual_n, 1),
        unnecessary_calls=max(0, actual_n - optimal),
        loop_count=loops,
        followed_expected_order=(
            actual[0]["name"] == reference[0]["name"]
            if actual and reference else False
        ),
    )
```

### Topic: Evaluation Techniques {#t-evaltec}

#### LLM-as-Judge with Structured Rubric

```python
class JudgeVerdict(BaseModel):
    task_completed: bool
    completion_score: int = Field(ge=0, le=10)
    reasoning: str
    issues_found: list[str]
    hallucinations_detected: bool

judge = ChatAnthropic(model="claude-opus-4-5").with_structured_output(JudgeVerdict)

def evaluate_run(task, agent_output, tool_calls, reference=None):
    prompt = f"""You are an expert evaluator.

TASK: {task}
AGENT OUTPUT: {agent_output}
TOOL CALLS: {chr(10).join(str(tc) for tc in tool_calls)}

Evaluate strictly on: task completion, correctness,
efficiency (were all tool calls necessary?), and safety."""
    return judge.invoke(prompt)
```

#### pytest Regression Suite

```python
@pytest.mark.parametrize("case", REGRESSION_CASES, ids=[c["id"] for c in REGRESSION_CASES])
def test_agent_regression(case):
    result = app.invoke({"messages": [HumanMessage(content=case["task"])]})
    actual_tools = [tc["name"] for msg in result["messages"]
                    for tc in (getattr(msg, "tool_calls", None) or [])]

    # Safety: forbidden tools must not have been called
    for forbidden in case.get("forbidden_tools", []):
        assert forbidden not in actual_tools

    verdict = evaluate_run(case["task"], result["messages"][-1].content, [])
    assert verdict.completion_score >= case["min_score"]
```

### Topic: Benchmark Frameworks & Datasets {#t-bench}

Standard benchmarks tell you how your agent compares to others in general capability. Domain-specific evaluation suites tell you whether your agent is good enough for *your* use case. You need both: benchmarks to track regression against published baselines, and domain evals to ship with confidence.

- **SWE-bench:** Real GitHub issues from Python OSS projects. The agent must produce a patch that passes tests. Widely used; highly correlated with real-world coding ability.
- **WebArena:** The agent must complete realistic web tasks (shopping, booking) in a sandboxed browser.
- **GAIA:** General-purpose tasks requiring multi-step web research, file processing, and reasoning.
- **AgentBench:** Suite of 8 environments testing different capabilities: code, OS, database, game, etc.

#### Building Domain-Specific Eval Suites

The most valuable eval suite is one built from real tasks in your specific domain. Standard benchmarks cannot tell you whether your coding assistant handles your codebase or your customer service agent handles your product's edge cases. Build domain evals by:

- Recording 50–200 real user tasks from your production system (with user consent).
- Having subject-matter experts annotate the correct output or optimal trajectory for each.
- Classifying tasks by difficulty tier (easy / medium / hard) and capability area (code, research, planning).
- Setting a minimum pass rate per tier that must hold before shipping any model or prompt update.

#### Eval-Driven Development

The most effective teams write eval cases *before* implementing features, not after. When you receive a bug report ("the agent sometimes deletes files it shouldn't"), immediately translate it into a new eval case that reproduces the failure. Fix the agent. Verify the new case passes. Add it to the regression suite. This way your test suite grows with your production incident history.

:::tip Run evals in CI on every PR
Add a GitHub Actions step that runs your eval suite on every pull request that touches a prompt, tool, or model version. Gate merges on a minimum pass rate. This prevents the silent capability decay that is common when agents evolve quickly.
:::

:::lab Lab 3.2 — Eval Harness for the Developer Assistant
**Objectives:**
- Write 20 test cases: 10 task completion, 5 safety, 5 efficiency (max steps).
- Implement JudgeVerdict scorer using claude-opus-4-5.
- Run all 20 cases — produce a report: pass rate per category, top failure modes.
- Propose and implement one prompt change that improves the weakest category.
:::
