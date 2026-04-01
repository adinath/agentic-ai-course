---
id: s13
number: "1.3"
title: "Planning, Reasoning & Control Flow"
time: "3:00–5:00 PM"
duration: "2 hours"
topics:
  - id: t-planning
    title: "Planning Strategies"
  - id: t-routing
    title: "Structured Routing"
  - id: t-reflection
    title: "Self-Correction"
---

Deep dive into how agents plan multi-step tasks. Compare planning strategies and implement task decomposition, replanning on failure, and structured output to drive reliable control flow.

### Topic: Planning Strategies {#t-planning}

#### ReAct: Interleaved Reasoning and Acting

ReAct (Reasoning + Acting) alternates between producing a brief reasoning trace (*Thought*) and selecting an action (*Action*). Best for tasks where each step reveals what the next step should be.

:::example ReAct trace example
**Thought:** I need to count open GitHub issues. I should use list_github_issues.<br/>
**Action:** list_github_issues(repo="acme/backend", state="open")<br/>
**Observation:** 23 items returned<br/>
**Answer:** There are 23 open issues in the acme/backend repository.
:::

#### Plan-and-Execute: Upfront Decomposition

Generates an explicit plan before taking any actions. When a step fails, the agent can replan from that point rather than starting over.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    current_step: int
    results: list
    final_answer: str

def planner(state):
    """Generate step-by-step plan."""
    response = planner_llm.invoke([
        SystemMessage(content="Decompose into 3-7 steps. Output JSON: {'steps': [...]}"),
        HumanMessage(content=state["task"]),
    ])
    data = json.loads(response.content)
    return {"plan": data["steps"], "current_step": 0}

def should_continue(state):
    if state["current_step"] >= len(state["plan"]):
        return "done"
    return "execute"
```

### Topic: Structured Outputs for Control Flow {#t-routing}

Structured output is how you make an agent's decisions machine-readable. Instead of parsing natural language like "I think I should call the data agent next", you ask the model to produce a JSON object that your orchestration code can route on *deterministically*. This is far more reliable than free-form text routing and prevents the model from selecting non-existent routes.

#### JSON Schema-Driven Routing

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    category: Literal["code", "data", "search", "general"]
    reasoning: str = Field(description="One sentence explaining the classification")
    priority: Literal["high", "normal", "low"]

llm = ChatAnthropic(model="claude-haiku-4-5")
router = llm.with_structured_output(RouteDecision)

decision = router.invoke("Plot monthly revenue from our Postgres DB")
print(decision.category)   # → "data"

agent_map = {"code": code_agent, "data": data_agent, ...}
result = agent_map[decision.category].invoke({"task": user_input})
```

#### Conditional Edges in LangGraph

LangGraph's `add_conditional_edges` is the mechanism for structured control flow. A routing function inspects the current state and returns the name of the next node to execute. This is more reliable than asking the model to choose its own next step in natural language.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class WorkflowState(TypedDict):
    task: str
    category: str
    result: str

def classify_node(state: WorkflowState) -> dict:
    decision = router.invoke(state["task"])
    return {"category": decision.category}

def route_after_classify(state: WorkflowState) -> str:
    """Return node name — this is the routing function."""
    return f"{state['category']}_agent"

g = StateGraph(WorkflowState)
g.add_node("classify", classify_node)
g.add_node("code_agent", run_code_agent)
g.add_node("data_agent", run_data_agent)
g.add_node("search_agent", run_search_agent)

g.add_edge(START, "classify")           # set_entry_point() is deprecated
# Conditional edges: routing function maps state → node name
g.add_conditional_edges("classify", route_after_classify, {
    "code_agent": "code_agent",
    "data_agent": "data_agent",
    "search_agent": "search_agent",
})
g.add_edge("code_agent", END)
g.add_edge("data_agent", END)
g.add_edge("search_agent", END)
app = g.compile()
```

#### Deterministic Guardrails

Structured outputs from the model still need to be validated before use. Always check the model's output against the allowed values rather than passing it directly to your routing logic — the model can occasionally produce values outside the `Literal` constraint in rare edge cases.

:::warning Always validate structured output before routing
Even with `with_structured_output` and a Pydantic model, validate that the returned value is in the expected set before using it as a dictionary key. An unexpected value will raise a `KeyError` that halts the agent silently.
:::

### Topic: Reflection & Self-Correction {#t-reflection}

#### Critic-Actor Pattern

Two model calls collaborate: the **Actor** generates a solution and the **Critic** evaluates it. If rejected, the Critic provides specific feedback that the Actor uses to revise. This loop repeats until approved or max attempts reached.

```python
def actor(state):
    prompt = state["task"]
    if state.get("feedback"):
        prompt += f"\n\nPrevious attempt failed. Feedback: {state['feedback']}\nPlease revise."
    response = actor_llm.invoke(prompt)
    return {"solution": response.content, "attempts": state.get("attempts", 0) + 1}

def should_retry(state):
    if state["approved"] or state["attempts"] >= 3:
        return "done"
    return "actor"

graph.add_conditional_edges("critic", should_retry,
    {"actor": "actor", "done": END})
```

#### Step-Back Prompting

Step-back prompting asks the agent to first answer a higher-level, simpler version of the question before tackling the specific task. This activates relevant background knowledge and reduces the chance of jumping to an incorrect solution. It is particularly effective for debugging, algorithm design, and architecture decisions.

:::example Step-back in practice
**Task:** "Fix the race condition in our Redis-based session store."

**Step-back question:** "What are the general causes of race conditions in distributed caches and their standard solutions?"

The step-back answer (SETNX, Lua scripts, atomic ops, advisory locks) now primes the model with the right concepts before it reads the actual code.
:::

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-sonnet-4-5")

def step_back_then_solve(task: str) -> str:
    """Apply step-back prompting before solving a complex task."""
    # Step 1: ask the abstract/general question first
    step_back_q = llm.invoke(
        f"What is a more general or abstract version of this question?\n\nTask: {task}"
    ).content

    background = llm.invoke(
        f"Answer this general question with key principles and patterns:\n\n{step_back_q}"
    ).content

    # Step 2: solve the original task with the background primed
    solution = llm.invoke([
        HumanMessage(content=(
            f"Background knowledge:\n{background}\n\n"
            f"Now solve the specific task:\n{task}"
        ))
    ])
    return solution.content
```

#### Logging Reasoning Traces

Every agent in production should log its complete reasoning traces — the full sequence of messages, tool calls, and results. Without traces, debugging an agent failure is nearly impossible because the model's internal reasoning is otherwise invisible. Structure logs as JSON for easy querying.

```python
import json, time, logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent.trace")

class TraceLogger(BaseCallbackHandler):
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.start = time.time()
        self.steps = []

    def on_tool_start(self, tool, input_str, **kwargs):
        self.steps.append({"step": len(self.steps)+1,
                           "type": "tool_call", "tool": tool["name"],
                           "input": str(input_str)[:300]})

    def on_tool_end(self, output, **kwargs):
        self.steps[-1]["output"] = str(output)[:300]

    def on_chain_end(self, outputs, **kwargs):
        logger.info(json.dumps({
            "run_id": self.run_id,
            "duration_s": round(time.time() - self.start, 2),
            "total_steps": len(self.steps),
            "trace": self.steps,
        }))
```

:::lab Lab 1.3 — Plan-and-Execute Agent
**Objectives:**
- Implement a Plan-and-Execute agent: "Add input validation and tests to the /users POST endpoint."
- The Planner (claude-sonnet) decomposes; the Executor (claude-haiku + tools) runs each step.
- Add a self-correction loop: if `run_tests` fails, agent replans from that step.
- Compare step count and token usage vs. the ReAct agent from Lab 1.1.
:::
