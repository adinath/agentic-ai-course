---
id: s11
number: "1.1"
title: "What Makes an Agent?"
time: "9:00–11:00 AM"
duration: "2 hours"
type: "Concepts & Architecture"
topics:
  - id: t-prao
    title: "The PRAO Loop"
  - id: t-primitives
    title: "Core Primitives"
  - id: t-frameworks
    title: "Frameworks"
---

Transition from thinking about LLMs as completion engines to thinking about agents as autonomous actors with goals, tools, and decision loops. Understand how tools like Claude Code work under the hood.

### Topic: The Agent Mental Model {#t-prao}

An LLM completion is stateless: you send a prompt, receive a response, the model forgets everything. An **agent** persists across multiple steps — it perceives its environment, reasons about what to do, takes an action, and observes the result before reasoning again. This cycle is the **Perception → Reasoning → Action → Observation (PRAO) loop**.

#### The PRAO Loop

- **Perception:** Inputs fed into the model — user messages, tool results, retrieved memories, environment state.
- **Reasoning:** The model's internal chain-of-thought; deciding which action to take and why.
- **Action:** Calling a tool, writing a file, sending an HTTP request, or replying to the user.
- **Observation:** The result returned by the action, appended to context so the model can reason next turn.

:::tip Mental model vs. implementation
In code, the loop is: call LLM → parse tool calls → execute tools → append results → call LLM again. The "reasoning" phase is invisible — it happens inside the model's forward pass. Your job is to shape the inputs (perception) and handle the outputs (actions) reliably.
:::

#### Reactive vs. Goal-Directed vs. Planning Agents

- **Reactive agents** (simplest) respond directly to the current input with no memory or planning. A customer service chatbot is reactive.
- **Goal-directed agents** maintain an objective and decide step-by-step how to achieve it. They use tools but do not plan ahead explicitly.
- **Planning agents** (most capable) generate a multi-step plan before acting. They can anticipate obstacles, parallelise, and replan on failure.

:::example Real-world mapping
**Reactive:** GitHub Copilot inline completion<br/>**Goal-directed:** Claude Code running a single task ("fix this bug")<br/>**Planning:** Devin-style agents decomposing a full feature request
:::

#### Why Agents Fail — Common Pitfalls

- **Infinite loops** — the agent calls the same tool repeatedly expecting different results.
- **Context overflow** — accumulated tool results exhaust the context window.
- **Hallucinated tool calls** — the model invents tool names or arguments that don't exist.
- **Goal drift** — the agent pursues a subgoal so aggressively it forgets the original task.
- **Overconfidence** — acting on the first plan without verifying preconditions.

### Topic: Core Agent Primitives {#t-primitives}

Every agent framework is built from the same handful of primitives. Mastering these before touching any framework saves enormous debugging time.

#### Tools and Function Calling

A tool is a function the LLM is allowed to call. You define its name, description, and JSON Schema for its parameters. A well-designed schema is the single most important thing you can do for agent reliability.

```python
# Minimal tool definition with LangChain (v0.3+)
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path.

    Args:
        path: Absolute or relative path to the file.
    Returns:
        The full text content of the file.
    """
    with open(path) as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, overwriting if it exists."""
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"
```

#### System Prompts as Agent Constitution

The system prompt defines who the agent is, what it can and cannot do, and the norms it must follow. A good developer-assistant system prompt includes role, capabilities, constraints, and output format.

```python
SYSTEM_PROMPT = """
You are an autonomous software engineering assistant.

CAPABILITIES
  - read_file(path): read source files
  - write_file(path, content): write or overwrite files
  - run_tests(test_path): execute pytest and return results

RULES
  1. Always read a file before modifying it.
  2. Run tests after every code change.
  3. If tests fail twice in a row, stop and report.
  4. Never delete files unless explicitly requested.

REASONING FORMAT
  Before each tool call, output one line: "Plan: <reason>"
"""
```

#### Stop Conditions and Safety Rails

Always configure three termination conditions: **max iterations**, a **task completion signal**, and an **error threshold**.

```python
MAX_ITERATIONS = 20
consecutive_errors = 0
messages = [{"role": "user", "content": task}]

for i in range(MAX_ITERATIONS):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:          # terminal state
        break

    for tc in response.tool_calls:
        try:
            result = tool_registry[tc["name"]].invoke(tc["args"])
            consecutive_errors = 0
        except Exception as e:
            result = f"ERROR: {e}"
            consecutive_errors += 1
        messages.append({"role": "tool", "content": str(result)})

    if consecutive_errors >= 3:
        break  # escalate to human
```

### Topic: Agentic Frameworks Landscape {#t-frameworks}

#### LangGraph — Recommended for Production

LangGraph models an agent as a directed graph where nodes are Python functions and edges are routing conditions. State flows as a typed dictionary — making control flow explicit, debuggable, and resumable.

:::tip Package versions (as of 2025)
`langgraph>=1.1`, `langchain-core>=1.2`, `langchain-anthropic>=1.3` — install with `pip install langgraph langchain-anthropic`
:::

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, add_messages
from typing import TypedDict, Annotated

# add_messages is a reducer that handles message merging correctly
# (replaces messages by ID, appends new ones) — never use operator.add for messages
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
llm_with_tools = llm.bind_tools([read_file, write_file, run_tests])

def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([read_file, write_file, run_tests]))
graph.add_edge(START, "agent")          # set_entry_point() is deprecated
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")        # loop back
app = graph.compile()
```

:::tip Why LangGraph over a simple loop?
State is typed, the graph is serializable (pause & resume), streaming and human interrupts are built-in, and LangSmith tracing works out-of-the-box.
:::

#### Model Context Protocol (MCP)

MCP is an open standard (Anthropic, 2024) that separates tool servers from agent clients. Instead of bundling tools inside your agent code, you run a lightweight tool server with a standardised interface. Any MCP-compatible agent can then discover and call those tools dynamically — this is how Claude Code connects to your file system and terminal.

#### Framework Selection Guide

- **LangGraph:** Production agents needing explicit state, checkpoints, and human-in-the-loop.
- **AutoGen:** Multi-agent conversations where agents communicate in natural language turns.
- **CrewAI:** Role-based agent teams with a simple declarative API — great for prototyping.
- **Custom loop:** Maximum control, minimum dependencies; best for simple or unusual architectures.

:::lab Lab 1.1 — Build a ReAct Loop from Scratch
**Objectives:**
- Implement a bare-bones ReAct loop without any framework.
- Give the agent two tools: `bash_exec` and `read_file`.
- Task: "How many TODO comments are in this codebase? List files and line numbers."
- Log every tool call and reasoning step to understand the decision loop.
:::
