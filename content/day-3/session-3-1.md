---
id: s31
number: "3.1"
title: "Multi-Agent Topologies"
time: "9:00–11:00 AM"
duration: "2 hours"
topics:
  - id: t-whymulti
    title: "Why Multi-Agent?"
  - id: t-topologies
    title: "Topology Patterns"
---

A single agent is a generalist. A multi-agent system is a team. Both have their place — but only one can handle a 500-file codebase simultaneously. Understand the six topology patterns so you can pick the right one, not just the first one you read about.

### Topic: Why Multi-Agent? {#t-whymulti}

Single agents hit a ceiling. Multi-agent systems push that ceiling up — at the cost of coordination complexity.

**Reach for multi-agent when:**
- Task requires more context than fits in one window (e.g., reviewing an entire codebase simultaneously)
- Subtasks are genuinely parallel and independent (e.g., security + performance + style review)
- Different parts require different expertise/models (cheap fast model for classification, expensive model for architecture)
- You need redundancy (multiple agents attempt the same task, best answer wins)

**Do NOT reach for multi-agent when:**
- A single agent with good tools can handle it — most tasks fall here
- You are still debugging the single-agent version
- Coordination overhead exceeds the benefit (adding 3 agents to save 5 minutes is a bad trade)

:::warning Multi-agent is not always the answer
Every agent you add is a new failure point, a new context to manage, and a new source of coordination bugs. Start with the simplest architecture that works.
:::

### Topic: Topology Patterns {#t-topologies}

Six canonical patterns. Each solves a different problem. Know all six; use only what the task requires.

#### 1. Supervisor → Worker

The most common pattern. A Supervisor receives the top-level task, decomposes it, dispatches to specialised Workers, collects results, and synthesises the final answer.

```
┌─────────────────────────────────────────────────────────┐
│                  SUPERVISOR-WORKER                       │
│                                                         │
│              [Supervisor Agent]                         │
│              ↙      ↓      ↘                            │
│    [Worker A] [Worker B] [Worker C]                     │
│        ↓          ↓          ↓                          │
│              [Supervisor Agent]                         │
│                   ↓                                     │
│             [Final Answer]                              │
└─────────────────────────────────────────────────────────┘
```

```python
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict
import asyncio

class SupervisorState(TypedDict):
    task: str
    security_result: str
    performance_result: str
    style_result: str
    final_review: str

# Specialised worker agents
security_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
perf_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
style_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
supervisor_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

async def security_worker(state: SupervisorState) -> dict:
    response = await security_llm.ainvoke([
        SystemMessage(content="You are a security expert. Find vulnerabilities in this code diff."),
        HumanMessage(content=state["task"]),
    ])
    return {"security_result": response.content}

async def performance_worker(state: SupervisorState) -> dict:
    response = await perf_llm.ainvoke([
        SystemMessage(content="You are a performance engineer. Find inefficiencies in this code diff."),
        HumanMessage(content=state["task"]),
    ])
    return {"performance_result": response.content}

async def style_worker(state: SupervisorState) -> dict:
    response = await style_llm.ainvoke([
        SystemMessage(content="You are a code style reviewer. Check PEP8, naming, and clarity."),
        HumanMessage(content=state["task"]),
    ])
    return {"style_result": response.content}

def dispatch_workers(state: SupervisorState) -> dict:
    """Dispatch all workers in parallel — 3x faster than sequential."""
    loop = asyncio.new_event_loop()
    sec, perf, style = loop.run_until_complete(asyncio.gather(
        security_worker(state),
        performance_worker(state),
        style_worker(state),
    ))
    loop.close()
    return {**sec, **perf, **style}

def synthesise(state: SupervisorState) -> dict:
    response = supervisor_llm.invoke([
        SystemMessage(content="Synthesise these specialist reviews into a final PR review."),
        HumanMessage(content=(
            f"Security review:\n{state['security_result']}\n\n"
            f"Performance review:\n{state['performance_result']}\n\n"
            f"Style review:\n{state['style_result']}"
        )),
    ])
    return {"final_review": response.content}
```

#### 2. Router → Expert

A Router classifies incoming requests and dispatches each to the most appropriate Expert agent. The Router itself is lightweight — it just classifies and routes.

```
┌─────────────────────────────────────────────────────────┐
│                   ROUTER-EXPERT                          │
│                                                         │
│         [User Request]                                  │
│               ↓                                         │
│          [Router Agent]                                 │
│         ↙    ↓    ↘    ↘                               │
│   [Code] [Data] [Search] [General]                     │
│  Expert  Expert  Expert   Expert                        │
└─────────────────────────────────────────────────────────┘
```

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    category: Literal["code", "data", "search", "general"]
    reasoning: str = Field(description="One sentence explaining the routing decision")
    confidence: float = Field(ge=0.0, le=1.0)

router_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
router = router_llm.with_structured_output(RouteDecision)

def route_request(state: dict) -> str:
    """Route the request to the appropriate expert agent."""
    task = state["messages"][-1].content
    decision = router.invoke(
        f"Classify this task for routing to a specialist agent:\n\n{task}"
    )

    # Fallback to general if confidence is low
    if decision.confidence < 0.6:
        return "general"

    return decision.category

# Build router graph
g = StateGraph({"messages": list, "result": str})
g.add_node("route", lambda s: {})  # no-op node to trigger routing
g.add_node("code_expert", code_expert_agent)
g.add_node("data_expert", data_expert_agent)
g.add_node("search_expert", search_expert_agent)
g.add_node("general_expert", general_agent)

g.add_edge(START, "route")
g.add_conditional_edges("route", route_request, {
    "code": "code_expert",
    "data": "data_expert",
    "search": "search_expert",
    "general": "general_expert",
})
for expert in ["code_expert", "data_expert", "search_expert", "general_expert"]:
    g.add_edge(expert, END)
```

#### 3. Orchestrator

An Orchestrator dynamically directs multiple agents at runtime — it decides in real-time which agent to call next based on intermediate results. Unlike Supervisor-Worker (fixed dispatch), the Orchestrator's routing decisions are adaptive.

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                         │
│         [Orchestrator] ←── dynamic decisions            │
│         ↙    ↕    ↘                                     │
│   [Agent A] [Agent B] [Agent C]                         │
│      ↓          ↓          ↓                            │
│      results fed back to Orchestrator                   │
└─────────────────────────────────────────────────────────┘
```

```python
class OrchestratorState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    next_agent: str  # orchestrator decides this dynamically
    results: dict

class NextAction(BaseModel):
    next_agent: Literal["researcher", "coder", "tester", "done"]
    reasoning: str

orchestrator_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
orchestrator = orchestrator_llm.with_structured_output(NextAction)

def orchestrate(state: OrchestratorState) -> dict:
    """Dynamically decide which agent to invoke next."""
    results_summary = json.dumps(state.get("results", {}), indent=2)
    action = orchestrator.invoke([
        SystemMessage(content="You are an orchestrator. Based on the task and results so far, decide the next action."),
        HumanMessage(content=f"Task: {state['task']}\nResults so far:\n{results_summary}"),
    ])
    return {"next_agent": action.next_agent}

def route_orchestrator(state: OrchestratorState) -> str:
    return state["next_agent"]  # orchestrator decides dynamically

g.add_conditional_edges("orchestrate", route_orchestrator, {
    "researcher": "researcher_agent",
    "coder": "coder_agent",
    "tester": "tester_agent",
    "done": END,
})
```

#### 4. Round Table

All agents see the same task and contribute independently. A moderator synthesises the contributions. Great for creative problem-solving, code review, and generating diverse perspectives.

```
┌─────────────────────────────────────────────────────────┐
│                    ROUND TABLE                           │
│                                                         │
│                [Task / Proposal]                        │
│              ↙    ↙    ↓    ↘    ↘                      │
│  [Agent A] [B] [C] [D] [E]                             │
│      ↓      ↓   ↓   ↓   ↓                              │
│              [Moderator / Synthesiser]                  │
│                    ↓                                    │
│              [Consensus Answer]                         │
└─────────────────────────────────────────────────────────┘
```

```python
async def round_table(task: str, n_participants: int = 3) -> str:
    """Run multiple agents on the same task and synthesise responses."""
    # All participants see the same prompt; vary temperature for diversity
    participants = [
        ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0),
        ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7),
        ChatAnthropic(model="claude-sonnet-4-6", temperature=1.0),
    ]

    # Gather all perspectives in parallel
    responses = await asyncio.gather(
        *[p.ainvoke([HumanMessage(content=task)]) for p in participants]
    )
    proposals = [r.content for r in responses]

    # Moderator synthesises
    moderator = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    combined = "\n\n---\n\n".join(
        f"Perspective {i+1}:\n{p}" for i, p in enumerate(proposals)
    )
    final = await moderator.ainvoke([
        SystemMessage(content="Synthesise these perspectives into the best possible answer. Preserve unique insights from each."),
        HumanMessage(content=f"Task: {task}\n\n{combined}"),
    ])
    return final.content
```

#### 5. Tree of Agents

A hierarchical decomposition where the root agent decomposes the task, child agents further decompose their subtasks, and leaf agents execute. Results propagate back up the tree. Useful for very large tasks.

```
┌─────────────────────────────────────────────────────────┐
│                   TREE OF AGENTS                         │
│                                                         │
│                    [Root Agent]                         │
│                   ↙         ↘                           │
│          [Branch A]       [Branch B]                    │
│          ↙    ↘           ↙     ↘                       │
│       [Leaf] [Leaf]   [Leaf]  [Leaf]                    │
│          ↑    ↑           ↑     ↑                       │
│          results flow up the tree                       │
└─────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass, field

@dataclass
class AgentNode:
    task: str
    depth: int
    children: list['AgentNode'] = field(default_factory=list)
    result: str = ""

MAX_DEPTH = 3
MIN_TASK_TOKENS = 50  # stop decomposing if task is small enough

def build_agent_tree(task: str, depth: int = 0) -> AgentNode:
    """Recursively decompose task into a tree of sub-tasks."""
    node = AgentNode(task=task, depth=depth)

    # Leaf condition: too deep or task is small enough to execute directly
    if depth >= MAX_DEPTH or len(task.split()) < MIN_TASK_TOKENS:
        node.result = execute_leaf_task(task)
        return node

    # Decompose
    subtasks = decompose_task(task)
    node.children = [build_agent_tree(st, depth + 1) for st in subtasks]

    # Synthesise children results
    children_results = "\n\n".join(
        f"Subtask: {c.task}\nResult: {c.result}" for c in node.children
    )
    node.result = synthesise_results(task, children_results)
    return node
```

#### 6. Pipeline

Agents are arranged in a linear sequence — each agent's output is the next agent's input. Best for deterministic workflows with ordered stages (e.g., research → draft → review → publish).

```
┌─────────────────────────────────────────────────────────┐
│                      PIPELINE                            │
│                                                         │
│  Input ──▶ [Stage 1] ──▶ [Stage 2] ──▶ [Stage 3] ──▶  │
│            Research      Draft          Review   Output │
└─────────────────────────────────────────────────────────┘
```

```python
class PipelineState(TypedDict):
    original_task: str
    research_output: str
    draft_output: str
    review_output: str
    final_output: str

def research_stage(state: PipelineState) -> dict:
    """Stage 1: gather information."""
    result = research_agent.invoke({"task": state["original_task"]})
    return {"research_output": result["messages"][-1].content}

def draft_stage(state: PipelineState) -> dict:
    """Stage 2: draft using research output."""
    result = draft_llm.invoke([
        SystemMessage(content="Write a technical blog post based on the research."),
        HumanMessage(content=f"Research:\n{state['research_output']}"),
    ])
    return {"draft_output": result.content}

def review_stage(state: PipelineState) -> dict:
    """Stage 3: review and refine the draft."""
    result = review_llm.invoke([
        SystemMessage(content="Review and improve this draft for accuracy and clarity."),
        HumanMessage(content=state["draft_output"]),
    ])
    return {"final_output": result.content}

# Build linear pipeline
g = StateGraph(PipelineState)
g.add_node("research", research_stage)
g.add_node("draft", draft_stage)
g.add_node("review", review_stage)
g.add_edge(START, "research")
g.add_edge("research", "draft")
g.add_edge("draft", "review")
g.add_edge("review", END)
pipeline = g.compile()
```

#### Topology Selection Guide

```
┌──────────────────────┬──────────────────────────────────────────┐
│  Pattern             │  Use When                                │
├──────────────────────┼──────────────────────────────────────────┤
│  Supervisor-Worker   │  Parallel, independent subtasks          │
│  Router-Expert       │  Many request types, clear categories    │
│  Orchestrator        │  Dynamic workflows, adaptive decisions   │
│  Round Table         │  Needs diverse perspectives / consensus  │
│  Tree of Agents      │  Massive tasks, recursive decomposition  │
│  Pipeline            │  Ordered stages, deterministic workflow  │
└──────────────────────┴──────────────────────────────────────────┘
```

:::lab Lab 3.1 — Multi-Agent PR Review System
**Objectives:**
- Implement the Supervisor-Worker pattern: Supervisor → [Security, Performance, Style] → Synthesis.
- Use async parallel dispatch for the three workers.
- Measure speedup: sequential vs. parallel execution.
- Extend: add a Router in front that checks if the diff is large enough to need all three workers (small diffs → single General reviewer).
:::
