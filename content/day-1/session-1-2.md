---
id: s12
number: "1.2"
title: "Tools, Actions & Environment Interaction"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-tooldesign
    title: "Tool Design"
  - id: t-execution
    title: "Execution Patterns"
  - id: t-toolcat
    title: "Tool Categories"
---

Design and implement robust tool interfaces. Cover synchronous and async tool execution, error handling, and safe interaction with external systems — files, APIs, databases, and browsers.

### Topic: Tool Design Principles {#t-tooldesign}

A tool is an API contract between your agent and the world. The quality of that contract determines whether your agent succeeds or fails. Poor tool design is the number-one cause of agent unreliability in production systems.

#### Schema Precision

```python
# ❌ Bad: vague parameter names, no descriptions
@tool
def query_db(q: str) -> str:
    """Query the database.""" ...

# ✅ Good: precise contract with constraints
@tool
def query_database(
    sql: str,
    database: str = "production",
    timeout_seconds: int = 30,
) -> dict:
    """Execute a read-only SELECT query.

    Args:
        sql: A valid SELECT statement. INSERT/UPDATE/DELETE rejected.
        database: One of: production, staging, analytics.
        timeout_seconds: Query timeout. Max 120.
    Returns:
        {"rows": [...], "columns": [...], "row_count": N}
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries permitted.")
```

#### Idempotency & Reversibility

- **Idempotent read tools:** `read_file`, `search_code`, `query_database` — safe to call multiple times.
- **Idempotent write tools:** `write_file` (overwrites) is idempotent; `append_to_file` is NOT.
- **Reversible write tools:** Always create a backup before modifying. Return a "revert" mechanism.

### Topic: Execution Patterns {#t-execution}

#### Parallel Tool Calls

When a model returns multiple tool calls in one response, they can run in parallel — one of the biggest latency wins available. LangGraph's `ToolNode` handles this automatically.

```python
from langgraph.prebuilt import ToolNode
import asyncio

# ToolNode runs parallel tool calls automatically
tool_node = ToolNode(tools=[read_file, search_code, run_tests])

# Under the hood it does something like:
async def _parallel_execute(tool_calls):
    tasks = [
        TOOLS[tc["name"]].ainvoke(tc["args"])
        for tc in tool_calls
    ]
    return await asyncio.gather(*tasks)  # all run concurrently
```

#### Sandboxing: Safe Code Execution

When an agent can execute code, sandboxing is mandatory. Options in increasing isolation:

- **subprocess with timeout** — basic protection, not isolated.
- **Docker exec** — isolated filesystem; agent cannot escape unless you mount volumes.
- **E2B (e2b.dev)** — managed cloud sandboxes with a simple Python SDK, best for SaaS agents.
- **Firecracker microVMs** — production-grade isolation used by cloud providers.

```python
from e2b_code_interpreter import Sandbox

@tool
def run_python_code(code: str) -> str:
    """Execute Python in an isolated cloud sandbox.
    No access to host filesystem or network.
    """
    with Sandbox() as sbx:
        execution = sbx.run_code(code)
        return {
            "stdout": execution.text,
            "stderr": [str(e) for e in execution.errors],
        }
```

### Topic: Real-World Tool Categories {#t-toolcat}

Production agents need tools that interact with real systems. These fall into several broad categories, each with distinct design patterns and safety considerations. Understanding the category helps you pick the right abstraction and error-handling strategy.

#### File System & Shell Tools

The most common category for developer agents. Always scope file access to a specific directory and validate paths to prevent directory traversal. Run untrusted code in an isolated sandbox rather than directly on the host.

```python
import subprocess
from pathlib import Path

WORKSPACE = Path("/workspace").resolve()

def _safe_path(path: str) -> Path:
    resolved = (WORKSPACE / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE)):
        raise PermissionError(f"Path outside workspace: {path}")
    return resolved

@tool
def run_shell(command: str) -> str:
    """Run a shell command inside /workspace. Destructive commands blocked."""
    BLOCKED = ["rm -rf", "curl ", "wget ", "sudo "]
    if any(b in command for b in BLOCKED):
        raise PermissionError(f"Blocked: {command}")
    result = subprocess.run(
        command, shell=True, capture_output=True,
        text=True, cwd=WORKSPACE, timeout=30
    )
    return (result.stdout + result.stderr)[:4000]
```

#### Web Search & Browser Automation

Web access is one of the most powerful capabilities you can give an agent. **Tavily** is optimised for agents — it returns clean extracted text rather than raw HTML, greatly reducing context bloat. For interactive pages (forms, login flows, SPAs), use Playwright browser automation.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# Tavily: best for agents — returns clean extracted text
search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_raw_content=False,  # keep context compact
)
results = search.invoke("LangGraph 1.1 latest release notes")
# → [{"url": "...", "content": "..."}, ...]

# Playwright for interactive browser sessions
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
)
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

browser = await create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
tools = toolkit.get_tools()
# Provides: navigate_browser, click_element, fill_form,
#           extract_text, get_elements, current_page_url
```

#### Database & API Tools

Agents frequently need to query databases or call external APIs. Always use parameterised queries to prevent SQL injection, and never expose raw database credentials in tool descriptions — inject them via environment variables at runtime.

```python
import httpx, os
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Safe SQL query tool (read-only by default)
db = SQLDatabase.from_uri(os.environ["DATABASE_URL"])
sql_tool = QuerySQLDataBaseTool(db=db)

# Generic REST API tool with auth header injection
@tool
async def call_api(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    """Call an internal REST API endpoint.
    Args:
        endpoint: Path relative to API base, e.g. /users/42
        method: HTTP method: GET, POST, PUT, PATCH.
        body: JSON body for POST/PUT/PATCH requests.
    """
    base = os.environ["API_BASE_URL"]
    headers = {"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, base + endpoint,
                                     json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
```

#### Human-in-the-Loop: Approval Gates

For irreversible or high-risk actions (sending emails, deploying to production), pause the agent and ask a human before proceeding. LangGraph supports this natively via the `interrupt()` function — the preferred API since LangGraph 1.0.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage

# Add a dedicated review node that calls interrupt()
def human_review_node(state):
    """Pause execution and surface the pending tool calls for human approval."""
    pending = state["messages"][-1].tool_calls
    decision = interrupt({
        "question": "Approve the following tool calls?",
        "pending_calls": pending,
    })
    # decision is whatever the human sends back via Command(resume=...)
    return {"approved": decision}

checkpointer = MemorySaver()
# Wire review_node before tools in your graph, then compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-42"}}

# First run — hits interrupt() inside human_review_node and pauses
state = app.invoke({"messages": [HumanMessage("Deploy v2.1 to prod")]}, config)
print("Interrupted — pending:", state)

# Human reviews and resumes with a decision
final = app.invoke(Command(resume=True), config)   # or resume=False to abort
```

:::lab Lab 1.2 — Developer Assistant with 5 Tools
**Objectives:**
- Build an agent with: `read_file`, `write_file`, `run_tests`, `search_web`, `call_api`.
- Task: "Find the latest httpx changelog and update requirements.txt, then run tests."
- Observe the agent's tool chaining strategy end-to-end.
- Add `interrupt_before=["tools"]` and test the human approval flow.
:::
