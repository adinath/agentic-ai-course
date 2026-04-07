---
id: s32
number: "3.2"
title: "A2A Communication & Agent Evaluation"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-a2a
    title: "Agent-to-Agent Communication"
  - id: t-eval
    title: "Agent Evaluation"
---

Multi-agent systems need a common language. They also need an honest report card. This session covers both: the emerging protocols that let agents talk to each other across organisational boundaries, and the evaluation frameworks that tell you whether any of it is actually working.

### Topic: Agent-to-Agent Communication {#t-a2a}

#### Why Standardised A2A Communication Matters

Without a standard protocol, every multi-agent integration requires custom code. You end up with:
- Agent A calling Agent B via a proprietary REST API
- Agent B calling Agent C via a different proprietary SDK
- No interoperability, no reuse, no ecosystem

The solution: define a shared protocol for agent discovery, task delegation, and result streaming — just as HTTP standardised web communication.

Two protocols are emerging as standards: **A2A** (Google's Agent-to-Agent Protocol) and **ACP** (IBM's Agent Communication Protocol).

#### A2A — Agent to Agent Protocol

A2A is an open protocol introduced by Google in 2025. It enables agents to discover each other's capabilities via an **Agent Card** (a JSON manifest) and exchange tasks/results via standardised HTTP+SSE endpoints.

```
┌─────────────────────────────────────────────────────────────────┐
│                    A2A PROTOCOL FLOW                             │
│                                                                  │
│  Client Agent                        Remote Agent               │
│  ─────────────                        ──────────────            │
│                                                                  │
│  GET /.well-known/agent.json  ──▶   Returns Agent Card          │
│  (discover capabilities)            (name, skills, endpoint)    │
│                                                                  │
│  POST /tasks                  ──▶   Task accepted               │
│  (delegate task)                    (returns task_id)           │
│                                                                  │
│  GET /tasks/{id}/stream       ──▶   SSE stream of updates       │
│  (poll for results)                 (working → completed)       │
└─────────────────────────────────────────────────────────────────┘
```

**Agent Card** — the agent's public capability manifest:

```json
{
  "name": "security-review-agent",
  "version": "1.0.0",
  "description": "Specialised agent for security vulnerability detection in code diffs",
  "endpoint": "https://agents.example.com/security",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "review_diff",
      "name": "Review Code Diff",
      "description": "Analyse a git diff for security vulnerabilities",
      "inputModes": ["text"],
      "outputModes": ["text", "structured"]
    },
    {
      "id": "scan_dependencies",
      "name": "Scan Dependencies",
      "description": "Check dependencies for known CVEs",
      "inputModes": ["file"],
      "outputModes": ["structured"]
    }
  ]
}
```

**Implementing an A2A server:**

```python
# a2a_server.py — Expose your LangGraph agent via A2A protocol
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio, uuid, json

app = FastAPI()

# In-memory task store (use Redis/Postgres in production)
tasks: dict[str, dict] = {}

# Your existing LangGraph agent
security_agent = build_security_agent()

# --- Agent Card endpoint (discovery) ---
@app.get("/.well-known/agent.json")
async def get_agent_card():
    return {
        "name": "security-review-agent",
        "version": "1.0.0",
        "description": "Code security vulnerability detector",
        "endpoint": "http://localhost:8000",
        "capabilities": {"streaming": True},
        "skills": [{
            "id": "review_diff",
            "name": "Review Code Diff",
            "description": "Find security issues in a git diff",
            "inputModes": ["text"],
            "outputModes": ["text"],
        }],
    }

class TaskRequest(BaseModel):
    skill_id: str
    input: str
    session_id: Optional[str] = None

# --- Task submission endpoint ---
@app.post("/tasks")
async def create_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "working", "result": None}

    # Run agent in background
    asyncio.create_task(run_agent_task(task_id, request))
    return {"task_id": task_id, "status": "working"}

async def run_agent_task(task_id: str, request: TaskRequest):
    try:
        result = await security_agent.ainvoke({
            "messages": [{"role": "user", "content": request.input}]
        })
        tasks[task_id] = {
            "status": "completed",
            "result": result["messages"][-1].content,
        }
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

# --- Streaming result endpoint ---
@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    async def event_generator():
        while True:
            task = tasks.get(task_id, {})
            status = task.get("status", "unknown")

            if status == "working":
                yield f"data: {json.dumps({'status': 'working'})}\n\n"
                await asyncio.sleep(0.5)
            elif status == "completed":
                yield f"data: {json.dumps({'status': 'completed', 'result': task['result']})}\n\n"
                break
            elif status == "failed":
                yield f"data: {json.dumps({'status': 'failed', 'error': task.get('error')})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Implementing an A2A client:**

```python
# a2a_client.py — Discover and delegate to remote agents
import httpx, asyncio, json

class A2AClient:
    """Client for communicating with A2A-compliant agents."""

    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.card: dict = {}

    async def discover(self) -> dict:
        """Fetch and cache the agent's capability card."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.agent_url}/.well-known/agent.json")
            response.raise_for_status()
            self.card = response.json()
        return self.card

    async def delegate(self, skill_id: str, input_text: str) -> str:
        """Delegate a task to the remote agent and stream results."""
        if not self.card:
            await self.discover()

        # Validate skill exists
        skill_ids = [s["id"] for s in self.card.get("skills", [])]
        if skill_id not in skill_ids:
            raise ValueError(f"Skill '{skill_id}' not found. Available: {skill_ids}")

        async with httpx.AsyncClient(timeout=120) as client:
            # Submit task
            task_response = await client.post(
                f"{self.agent_url}/tasks",
                json={"skill_id": skill_id, "input": input_text},
            )
            task_response.raise_for_status()
            task_id = task_response.json()["task_id"]

            # Stream results via SSE
            async with client.stream("GET", f"{self.agent_url}/tasks/{task_id}/stream") as stream:
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event["status"] == "completed":
                            return event["result"]
                        elif event["status"] == "failed":
                            raise RuntimeError(f"Remote agent failed: {event.get('error')}")


# Usage: discover and use a remote agent
async def main():
    client = A2AClient("http://localhost:8000")
    card = await client.discover()
    print(f"Connected to: {card['name']} — {len(card['skills'])} skills available")

    result = await client.delegate(
        skill_id="review_diff",
        input_text="diff --git a/auth.py ...",
    )
    print(f"Security review:\n{result}")
```

#### ACP — Agent Communication Protocol

ACP (IBM's Agent Communication Protocol) takes a different approach — it focuses on asynchronous, event-driven communication via a message bus, making it better suited for long-running, complex multi-agent workflows.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACP vs A2A COMPARISON                         │
├─────────────────────────┬───────────────────────────────────────┤
│         A2A             │              ACP                       │
├─────────────────────────┼───────────────────────────────────────┤
│ Request/response + SSE  │ Async message bus (pub/sub)           │
│ Per-agent endpoints     │ Central broker / router               │
│ REST + JSON             │ Events with typed schemas             │
│ Good for: single task   │ Good for: complex workflows           │
│ delegation              │ with many agents over time            │
│ Simpler to implement    │ Better for long-running pipelines     │
└─────────────────────────┴───────────────────────────────────────┘
```

```python
# ACP-style agent communication via message queue (Redis Streams)
import redis.asyncio as redis
import json, asyncio
from uuid import uuid4

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

class ACPMessage:
    """Typed message for agent-to-agent communication."""
    def __init__(self, sender: str, recipient: str, task: str,
                 task_id: str = None, correlation_id: str = None):
        self.sender = sender
        self.recipient = recipient
        self.task = task
        self.task_id = task_id or str(uuid4())
        self.correlation_id = correlation_id or self.task_id

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "task": self.task,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
        }

async def send_task(msg: ACPMessage):
    """Send a task to a specific agent via its message stream."""
    stream_key = f"agent:{msg.recipient}:inbox"
    await r.xadd(stream_key, msg.to_dict())

async def receive_tasks(agent_name: str, handler):
    """Listen for incoming tasks on this agent's stream."""
    stream_key = f"agent:{agent_name}:inbox"
    last_id = "$"  # start from new messages only
    while True:
        messages = await r.xread({stream_key: last_id}, block=1000, count=10)
        for stream, msgs in (messages or []):
            for msg_id, msg_data in msgs:
                last_id = msg_id
                await handler(ACPMessage(**msg_data))

# Usage: Orchestrator sends to Security agent
async def orchestrator_workflow(pr_diff: str):
    task_id = str(uuid4())
    await send_task(ACPMessage(
        sender="orchestrator",
        recipient="security-agent",
        task=pr_diff,
        task_id=task_id,
    ))
    print(f"Task {task_id} dispatched to security-agent")
```

### Topic: Agent Evaluation {#t-eval}

#### Why Evaluation is Hard for Agents

A correct final answer achieved through an inefficient or unsafe path is not a success. You need to evaluate:
- **Did it work?** (task completion)
- **How did it work?** (tool trajectory)
- **Was it accurate?** (tool call precision)
- **At what cost?** (tokens, latency, money)

#### Tool Call Accuracy

Measures whether the agent called the right tools with the right arguments. A tool called with wrong arguments that happens to produce a correct output by coincidence is a failure — it will fail on the next similar task.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolCallAccuracyScore:
    correct_tool_selected: bool      # right tool for the task
    correct_arguments: bool          # arguments match expected values
    argument_accuracy: float         # 0-1, fraction of args that match
    hallucinated_tool: bool          # tool name does not exist
    unnecessary_calls: int           # calls that did not contribute to outcome

def evaluate_tool_call_accuracy(
    actual_calls: list[dict],
    expected_calls: list[dict],      # human-annotated reference
    tool_registry: dict,
) -> ToolCallAccuracyScore:
    """
    Compare actual tool calls against the expected (reference) calls.

    actual_calls format: [{"name": "read_file", "args": {"path": "src/api.py"}}]
    expected_calls format: same
    """
    hallucinated = any(c["name"] not in tool_registry for c in actual_calls)

    if not expected_calls:
        return ToolCallAccuracyScore(
            correct_tool_selected=False, correct_arguments=False,
            argument_accuracy=0.0, hallucinated_tool=hallucinated,
            unnecessary_calls=len(actual_calls),
        )

    # Check if expected tools were called (order-independent)
    actual_tool_names = {c["name"] for c in actual_calls}
    expected_tool_names = {c["name"] for c in expected_calls}
    correct_tools = expected_tool_names.issubset(actual_tool_names)

    # Argument accuracy: fraction of expected arg values that match
    matched_args = 0
    total_args = 0
    for exp in expected_calls:
        actual_match = next(
            (a for a in actual_calls if a["name"] == exp["name"]), None
        )
        if actual_match:
            for key, val in exp["args"].items():
                total_args += 1
                if actual_match["args"].get(key) == val:
                    matched_args += 1

    arg_accuracy = matched_args / max(total_args, 1)
    unnecessary = max(0, len(actual_calls) - len(expected_calls))

    return ToolCallAccuracyScore(
        correct_tool_selected=correct_tools,
        correct_arguments=arg_accuracy >= 0.9,
        argument_accuracy=arg_accuracy,
        hallucinated_tool=hallucinated,
        unnecessary_calls=unnecessary,
    )
```

#### Tool Trajectory Evaluation

Trajectory evaluation goes beyond individual tool calls — it assesses the *sequence* and *pattern* of tool usage. A good trajectory is efficient, ordered logically, and free of loops.

```python
@dataclass
class TrajectoryScore:
    efficiency_ratio: float          # reference_steps / actual_steps (1.0 = perfect)
    followed_expected_order: bool    # did the agent follow the expected sequence?
    loop_count: int                  # how many times did it repeat the same call?
    unnecessary_calls: int           # calls with no contribution to the outcome
    backtrack_count: int             # how many times did it undo previous work?
    overall_score: float             # 0-1 composite score

def score_trajectory(
    actual: list[dict],
    reference: list[dict],
) -> TrajectoryScore:
    """
    Score the agent's tool call trajectory against a reference.

    A lower actual:reference ratio means the agent took more steps than needed.
    """
    optimal_n = len(reference)
    actual_n = len(actual)

    # Detect loops: same (tool, args) appearing more than once
    seen: set = set()
    loops = 0
    for call in actual:
        key = (call["name"], json.dumps(sorted(call.get("args", {}).items())))
        if key in seen:
            loops += 1
        seen.add(key)

    # Detect backtracks: write → read same file → write again
    backtracks = 0
    writes = {}
    for i, call in enumerate(actual):
        if call["name"] in ("write_file", "edit_file"):
            path = call.get("args", {}).get("path", "")
            writes[path] = i
        elif call["name"] == "read_file":
            path = call.get("args", {}).get("path", "")
            if path in writes:
                backtracks += 1

    efficiency = optimal_n / max(actual_n, 1)
    order_match = (
        len(actual) > 0 and len(reference) > 0
        and actual[0]["name"] == reference[0]["name"]
    )
    unnecessary = max(0, actual_n - optimal_n)

    # Composite score penalises loops and backtracks heavily
    overall = max(0.0, efficiency - (loops * 0.1) - (backtracks * 0.05))

    return TrajectoryScore(
        efficiency_ratio=round(efficiency, 3),
        followed_expected_order=order_match,
        loop_count=loops,
        unnecessary_calls=unnecessary,
        backtrack_count=backtracks,
        overall_score=round(min(1.0, overall), 3),
    )


# LLM-as-Judge for trajectory quality
class TrajectoryJudgement(BaseModel):
    is_optimal: bool
    efficiency_score: int = Field(ge=0, le=10)
    reasoning: str
    redundant_steps: list[str]
    missing_steps: list[str]

def judge_trajectory(task: str, actual: list[dict], reference: list[dict]) -> TrajectoryJudgement:
    """Use an LLM to qualitatively assess the trajectory."""
    judge = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TrajectoryJudgement)
    actual_str = "\n".join(f"{i+1}. {c['name']}({c.get('args', {})})" for i, c in enumerate(actual))
    ref_str = "\n".join(f"{i+1}. {c['name']}({c.get('args', {})})" for i, c in enumerate(reference))

    return judge.invoke(
        f"Task: {task}\n\n"
        f"Reference trajectory (optimal):\n{ref_str}\n\n"
        f"Actual trajectory:\n{actual_str}\n\n"
        "Evaluate whether the actual trajectory was efficient and correct."
    )
```

#### Building a Regression Eval Suite

```python
import pytest
from typing import TypedDict

class EvalCase(TypedDict):
    id: str
    task: str
    expected_tool_calls: list[dict]   # reference trajectory
    min_completion_score: float       # 0-1
    forbidden_tools: list[str]        # safety: must NOT be called
    max_iterations: int               # efficiency: must complete within N steps

# Annotate your cases with expected trajectories
EVAL_CASES: list[EvalCase] = [
    {
        "id": "fix-auth-bug",
        "task": "Fix the 401 error in the login endpoint",
        "expected_tool_calls": [
            {"name": "read_file", "args": {"path": "src/auth.py"}},
            {"name": "write_file", "args": {"path": "src/auth.py"}},
            {"name": "run_tests", "args": {"test_path": "tests/test_auth.py"}},
        ],
        "min_completion_score": 0.8,
        "forbidden_tools": ["delete_file", "drop_table"],
        "max_iterations": 8,
    },
]

@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_agent_trajectory(case: EvalCase):
    result = app.invoke({"messages": [HumanMessage(content=case["task"])]})

    # Collect all tool calls from the run
    actual_tool_calls = [
        {"name": tc["name"], "args": tc["args"]}
        for msg in result["messages"]
        for tc in getattr(msg, "tool_calls", [])
    ]

    # Safety: forbidden tools must not have been called
    actual_names = {c["name"] for c in actual_tool_calls}
    for forbidden in case.get("forbidden_tools", []):
        assert forbidden not in actual_names, f"Agent called forbidden tool: {forbidden}"

    # Efficiency: must complete within max iterations
    assert len(actual_tool_calls) <= case["max_iterations"], (
        f"Agent took {len(actual_tool_calls)} tool calls, max allowed: {case['max_iterations']}"
    )

    # Tool call accuracy
    accuracy = evaluate_tool_call_accuracy(
        actual_tool_calls,
        case["expected_tool_calls"],
        tool_registry,
    )
    assert not accuracy.hallucinated_tool, "Agent hallucinated a tool name"

    # Trajectory scoring
    traj = score_trajectory(actual_tool_calls, case["expected_tool_calls"])
    assert traj.loop_count == 0, f"Agent looped {traj.loop_count} times"
    assert traj.efficiency_ratio >= 0.5, f"Trajectory too inefficient: {traj.efficiency_ratio}"
```

:::lab Lab 3.2 — A2A Integration + Eval Harness
**Objectives:**
- Deploy the security review agent as an A2A server (FastAPI + SSE).
- Write an A2A client that discovers capabilities and delegates review tasks.
- Annotate 10 test cases with expected tool trajectories.
- Run the eval suite — report: tool call accuracy, trajectory scores, loops detected.
- Fix the worst-performing case and re-run to verify improvement.
:::
