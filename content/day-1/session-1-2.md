---
id: s12
number: "1.2"
title: "Frameworks, Tools & ReAct"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-frameworks
    title: "Framework Introduction"
  - id: t-tooldesign
    title: "Tool Calling & Design"
  - id: t-react
    title: "ReAct Agent"
---

Frameworks are training wheels for agents — helpful until they get in the way. Understand the landscape, set up LangGraph and Ollama, then build a proper ReAct agent with iteration limits and graceful error handling.

### Topic: Framework Introduction {#t-frameworks}

#### Framework Landscape

Every agent framework is an opinionated wrapper around the same PRAO loop you built in session 1.1. Choose based on what you need to control — and which platform you live on.

| Framework | Strength | Use when |
|---|---|---|
| **LangGraph** ★ | Explicit state graphs, checkpoints, time-travel debugging, human-in-the-loop | Production agents needing tight control flow |
| **CrewAI** | Declarative role-based teams, fast scaffolding | Quick prototypes, role-driven workflows |
| **AutoGen** | Multi-agent conversations in natural language | Research, debate, collaborative tasks |
| **LlamaIndex** | Event-driven Workflows, 300+ data connectors, best-in-class RAG & indexing | Data/RAG-first agents — doc Q&A, knowledge bases, structured extraction |
| **Google ADK** | Multi-language (Py/TS/Go/Java), Session/State/Memory, A2A native | Production agents on Vertex AI / Gemini Enterprise |
| **AWS Strands** | Model-driven loop, MCP-native, OpenTelemetry observability | Lean agents on AWS (Lambda, Bedrock AgentCore) |
| **Custom loop** | Maximum control, zero overhead | Unusual architectures, teaching the basics |

**LangGraph is the recommended default for production work** — explicit state, serialisable graphs, streaming, checkpointing, and human-in-the-loop built in. It's also the framework used throughout this course.

#### Hosted vs. Local Models

Two runtime choices, with very different trade-offs. **Use both** in development; pick the right one per task.

| Channel | Examples | Use for |
|---|---|---|
| **Hosted** (cloud API) | Anthropic Claude, OpenAI, Gemini, Bedrock | Production reasoning, complex code, multi-modal, anything customer-facing |
| **Local** (on-device) | Ollama (llama3.2, qwen2.5-coder), vLLM | Air-gapped dev, cost-sensitive batch jobs, eval harnesses, integration tests |

#### LangGraph Setup

```bash
pip install "langgraph>=1.1" "langchain-anthropic>=1.4" "langchain-core>=1.2"
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, add_messages
from typing import TypedDict, Annotated

# State is a typed dict — the single source of truth for the entire graph run
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # add_messages reducer: replaces by ID, appends new — never use operator.add

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
```

#### Using Hosted Models (Anthropic Claude)

```python
import os
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

# Direct SDK — maximum control, lowest abstraction
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

# LangChain wrapper — integrates with the ecosystem (tools, callbacks, tracing)
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=4096,
)

# Model selection guide
MODELS = {
    "claude-haiku-4-5-20251001":  "Fast classification, routing, simple tools",
    "claude-sonnet-4-6": "General reasoning, coding, multi-step tasks",  # sweet spot
    "claude-opus-4-6":   "Complex architecture, strategic decisions, hard evals",
}
```

#### Using Local Models (Ollama)

Ollama runs open-weight models locally — no API key, no data leaves your machine. Essential for air-gapped environments or cost-sensitive development.

```bash
pip install "langchain-ollama>=0.2"
```

```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2          # 3B params — fast, good for tool calling
ollama pull qwen2.5-coder:7b  # optimised for code tasks
ollama serve                   # starts on http://localhost:11434
```

```python
from langchain_ollama import ChatOllama

# Drop-in replacement for ChatAnthropic in most LangGraph workflows
llm_local = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url="http://localhost:11434",
)

# Test tool calling capability
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm_with_tools = llm_local.bind_tools([add])
response = llm_with_tools.invoke("What is 17 + 25?")
print(response.tool_calls)  # → [{"name": "add", "args": {"a": 17, "b": 25}}]
```

:::tip Hosted vs. Local — when to use each
**Hosted (Claude):** Production, complex reasoning, high-stakes decisions, multi-modal.<br/>
**Local (Ollama):** Development, rapid iteration, cost-sensitive batch workloads, offline/air-gapped.
:::

### Topic: Tool Calling & Design {#t-tooldesign}

Tools are the agent's hands. A poorly designed tool schema is the number-one cause of agent unreliability — the model can only work with what you describe.

#### Tool Definition Best Practices

```python
from langchain_core.tools import tool
from pathlib import Path

WORKSPACE = Path("./workspace").resolve()

# ❌ Bad: vague, no safety, no structured return
@tool
def query(q: str) -> str:
    """Query the database."""
    return db.execute(q)

# ✅ Good: precise contract, validation, structured return, safety
@tool
def query_database(
    sql: str,
    database: str = "staging",
    timeout_seconds: int = 30,
) -> dict:
    """Execute a read-only SELECT query against the application database.

    Args:
        sql: A valid SELECT statement. INSERT/UPDATE/DELETE will be rejected.
        database: Target database. One of: staging, analytics. (NOT production)
        timeout_seconds: Query timeout in seconds. Maximum 120.

    Returns:
        {"rows": [...], "columns": [...], "row_count": N, "execution_ms": N}
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries permitted. Use staging, not production.")
    if database == "production":
        raise PermissionError("Direct production queries not allowed. Use analytics replica.")
    # ... execute and return structured result


@tool
def read_file(path: str) -> str:
    """Read the contents of a file within the workspace directory.

    Args:
        path: Relative path from workspace root (e.g. 'src/main.py').

    Returns:
        File contents as string. Raises PermissionError if path escapes workspace.
    """
    resolved = (WORKSPACE / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE)):
        raise PermissionError(f"Path outside workspace: {path}")
    return resolved.read_text()
```

#### Tool Design Checklist

- **Precise names** — `query_database` not `query`. The name is the model's primary signal.
- **Docstring as contract** — describe what it does, what each arg means, what it returns, and any constraints.
- **Structured returns** — return dicts/objects, not raw strings. The model can reason about structured data more reliably.
- **Validation up front** — reject bad inputs immediately with a clear error message. The model will read the error and self-correct.
- **Scope constraints** — never give an agent more access than it needs for the current task.

#### Binding Tools to a Model in LangGraph

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    resolved = (WORKSPACE / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE)):
        raise PermissionError(f"Path outside workspace: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"

@tool
def run_tests(test_path: str = "tests/") -> str:
    """Run pytest on the given path and return the output."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
        capture_output=True, text=True, cwd=WORKSPACE, timeout=60
    )
    return (result.stdout + result.stderr)[-3000:]  # last 3K chars

tools = [read_file, write_file, run_tests]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools, handle_tool_errors=True)  # handles parallel tool calls and surfaces errors
```

### Topic: ReAct Agent {#t-react}

ReAct (Reasoning + Acting) is the dominant pattern for tool-using agents. The model alternates between producing a reasoning trace and selecting an action. LangGraph makes a ReAct agent a first-class citizen.

#### Full ReAct Agent with LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, add_messages
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
tools = [read_file, write_file, run_tests]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools, handle_tool_errors=True)

def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # routes to tools or END
graph.add_edge("tools", "agent")                       # loop back after tools
app = graph.compile()

# Run it
result = app.invoke({"messages": [HumanMessage("Fix the failing test in tests/test_api.py")]})
print(result["messages"][-1].content)
```

#### Max Iterations — Preventing Infinite Loops

A ReAct agent without iteration limits will loop forever when it hits an obstacle. Always set a hard cap.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int  # track loop count explicitly

MAX_ITERATIONS = 15

def call_model(state: AgentState) -> dict:
    iteration = state.get("iteration_count", 0)
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": iteration + 1,
    }

def should_continue(state: AgentState) -> str:
    """Route: continue to tools, stop at limit, or finish naturally."""
    last_message = state["messages"][-1]

    # Natural completion — model decided it's done
    if not getattr(last_message, "tool_calls", None):
        return "end"

    # Hard safety limit
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"

    return "tools"

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
graph.add_edge("tools", "agent")
app = graph.compile()
```

#### Error Scenarios & Recovery

Agents encounter three classes of errors. Handle each differently:

```python
from langchain_core.messages import ToolMessage, HumanMessage
import json

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    consecutive_errors: int    # track error streaks
    error_log: list[str]       # audit trail for debugging

def safe_tool_executor(state: AgentState) -> dict:
    """Execute tools with per-class error handling."""
    last_message = state["messages"][-1]
    results = []
    consecutive_errors = state.get("consecutive_errors", 0)

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        try:
            # Class 1: Tool not found — model hallucinated a tool name
            if tool_name not in tool_registry:
                raise ValueError(
                    f"Tool '{tool_name}' does not exist. "
                    f"Available tools: {list(tool_registry.keys())}"
                )

            result = tool_registry[tool_name].invoke(tool_args)
            consecutive_errors = 0  # reset on success

            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))

        except PermissionError as e:
            # Class 2: Policy violation — block and explain, do not retry
            consecutive_errors += 1
            results.append(ToolMessage(
                content=f"BLOCKED: {e}. This action is not permitted. Do not retry.",
                tool_call_id=tool_call["id"],
            ))

        except Exception as e:
            # Class 3: Transient failure — explain and let agent retry with correction
            consecutive_errors += 1
            results.append(ToolMessage(
                content=f"ERROR: {type(e).__name__}: {e}. Please check your arguments and retry.",
                tool_call_id=tool_call["id"],
            ))

    # Abort if error streak is too long
    if consecutive_errors >= 3:
        results.append(HumanMessage(
            content="⚠️ Agent aborted after 3 consecutive errors. Please review the error log."
        ))

    return {
        "messages": results,
        "consecutive_errors": consecutive_errors,
    }

# Wire safe_tool_executor instead of ToolNode
graph.add_node("tools", safe_tool_executor)
```

:::example ReAct trace
**Thought (inside model):** I need to fix the failing test. First I should read the test file to understand what it expects.<br/>
**Action:** `read_file(path="tests/test_api.py")`<br/>
**Observation:** `def test_create_user(): ... assert response.status_code == 201`<br/>
**Thought:** Now I need to read the actual implementation to find the mismatch.<br/>
**Action:** `read_file(path="src/api.py")`<br/>
**Observation:** `return JSONResponse(status_code=200, ...)`  ← bug found<br/>
**Action:** `write_file(path="src/api.py", content=...)`<br/>
**Action:** `run_tests(test_path="tests/test_api.py")`<br/>
**Observation:** `1 passed`<br/>
**Final:** The test was failing because the status code was 200 instead of 201. Fixed.
:::

:::lab Lab 1.2 — ReAct Agent with Error Recovery
**Objectives:**
- Build a LangGraph ReAct agent with `read_file`, `write_file`, `run_tests`.
- Set `MAX_ITERATIONS = 10` and verify it stops gracefully.
- Deliberately inject a bad tool call (wrong file path) — observe the error message the model receives.
- Task: "The test `tests/test_calculator.py` is failing. Fix the bug in `src/calculator.py`."
- Measure: how many iterations did the agent need? What was the error recovery pattern?
:::
