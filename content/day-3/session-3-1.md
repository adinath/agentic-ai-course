---
id: s31
number: "3.1"
title: "Multi-Agent System Design"
time: "9:00–11:00 AM"
duration: "2 hours"
topics:
  - id: t-whymulti
    title: "Why Multi-Agent?"
  - id: t-topology
    title: "Topologies"
  - id: t-coord
    title: "Coordination"
---

### Topic: Why Multi-Agent? {#t-whymulti}

A single large agent can handle many tasks, but it has fundamental limits. Context windows fill up when a task requires simultaneously reasoning about 20 files. A general-purpose agent cannot match a specialist agent tuned — via prompt, memory, and tools — for a specific domain. And a single agent is a single point of failure: one bad decision derails the whole task.

Multi-agent systems solve these problems by distributing work across coordinated agents. Each agent is small, focused, and independently testable. The cost is coordination overhead and added architectural complexity — so only reach for multi-agent when the single-agent ceiling is genuinely limiting you.

- **Parallelism:** Independent subtasks run concurrently (e.g., review 5 PRs simultaneously).
- **Specialisation:** Different parts require different expertise or models (small fast model for classification, large model for reasoning).
- **Scale:** Route many request types to the most cost-effective handler.
- **Reliability:** Multiple agents independently attempt a task; supervisor picks the best (ensemble/consensus).

:::warning Multi-agent is not always the answer
Only reach for multi-agent when the single-agent ceiling is genuinely limiting you. Multi-agent adds coordination overhead and complexity. Start with the simplest architecture that works.
:::

### Topic: Topology Patterns {#t-topology}

#### Orchestrator-Worker

The most common pattern. An Orchestrator receives the top-level task, decomposes it, dispatches to Workers, collects results, and synthesises a final answer.

```python
def dispatch(state) -> OrchestratorState:
    """Dispatch all workers in parallel."""
    import asyncio
    async def run_all():
        s, p, st = await asyncio.gather(
            security_agent.ainvoke({"task": state["pr_diff"]}),
            perf_agent.ainvoke({"task": state["pr_diff"]}),
            style_agent.ainvoke({"task": state["pr_diff"]}),
        )
        return s["result"], p["result"], st["result"]
    s, p, st = asyncio.run(run_all())
    return {"security_result": s, "perf_result": p, "style_result": st}
```

#### Mixture of Agents (MoA)

Runs multiple agents in parallel on the same task, then a high-capability aggregator synthesises the best response. Dramatically improves quality on complex tasks — at the cost of 3–5× more LLM calls.

```python
from langchain_core.messages import HumanMessage

# Multiple proposer models for diversity
proposers = [
    ChatAnthropic(model="claude-sonnet-4-5"),
    ChatAnthropic(model="claude-sonnet-4-5", temperature=0.9),
    ChatOpenAI(model="gpt-4o"),
]
aggregator = ChatAnthropic(model="claude-opus-4-5")

async def mixture_of_agents(task: str) -> str:
    # .ainvoke() expects a list of messages, not a raw string
    responses = await asyncio.gather(
        *[m.ainvoke([HumanMessage(content=task)]) for m in proposers]
    )
    proposals = [r.content for r in responses]
    combined = "\n\n".join(f"Proposal {i+1}:\n{p}" for i, p in enumerate(proposals))
    final = await aggregator.ainvoke([
        HumanMessage(content=f"Task: {task}\n\n{combined}\n\nSynthesise the best answer.")
    ])
    return final.content
```

### Topic: Communication & Coordination {#t-coord}

#### Typed Task Contracts

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class TaskContract(BaseModel):
    task_id: str
    parent_task_id: Optional[str] = None
    assignee: str
    task_type: Literal["code", "review", "test", "deploy"]
    instruction: str
    context: dict = Field(default_factory=dict)
    priority: Literal["critical", "high", "normal", "low"] = "normal"

class TaskResult(BaseModel):
    task_id: str
    status: Literal["success", "failure", "partial"]
    output: str
    artifacts: dict = Field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
```

:::lab Lab 3.1 — Multi-Agent Code Review System
**Objectives:**
- Build: Orchestrator → [Security Agent, Performance Agent, Style Agent] → Synthesis Agent.
- Use parallel async dispatch for the three worker agents.
- Define typed `TaskContract` and `TaskResult` Pydantic models.
- Measure speedup: sequential execution vs. parallel dispatch.
:::
