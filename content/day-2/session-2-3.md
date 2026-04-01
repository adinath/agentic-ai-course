---
id: s23
number: "2.3"
title: "Building Production-Grade Single Agents"
time: "3:00–5:00 PM"
duration: "2 hours"
topics:
  - id: t-observ
    title: "Observability"
  - id: t-stream
    title: "Streaming & Interrupts"
  - id: t-deploy
    title: "Deployment"
---

### Topic: Observability & Tracing {#t-observ}

An unobservable agent is an untrustworthy agent. In production, you need to answer: What did the agent do? Why did it make that decision? How long did each step take? How much did it cost? Without structured tracing, you are debugging blind. The good news: LangChain's callback system lets you add comprehensive observability without changing your agent logic.

#### Langfuse Integration

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-lf-...", secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)

# Attach to a LangGraph run — all calls traced automatically
result = app.invoke(
    {"messages": [HumanMessage(content=task)]},
    config={
        "callbacks": [langfuse_handler],
        "run_name": f"task-{task_id}"
    },
)
```

#### Structured JSON Logging for Alerts

Beyond tracing platforms, every agent should emit structured JSON logs that capture the key signals for alerting: step count, token consumption, tool error rate, and wall-clock latency. These feed into your log aggregator (Datadog, CloudWatch, Loki) and drive alert rules.

```python
import json, time, logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent.ops")

class OpsMetricsHandler(BaseCallbackHandler):
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.start = time.time()
        self.tool_calls = 0
        self.tool_errors = 0

    def on_tool_start(self, tool, input_str, **kwargs):
        self.tool_calls += 1
        logger.info(json.dumps({"event": "tool_start", "run_id": self.run_id,
                                "tool": tool["name"], "input": str(input_str)[:200]}))

    def on_tool_error(self, error, **kwargs):
        self.tool_errors += 1
        logger.error(json.dumps({"event": "tool_error", "run_id": self.run_id,
                                  "error": str(error)[:400]}))

    def on_chain_end(self, outputs, **kwargs):
        error_rate = self.tool_errors / max(1, self.tool_calls)
        logger.info(json.dumps({
            "event": "run_complete", "run_id": self.run_id,
            "duration_s": round(time.time() - self.start, 2),
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "error_rate": round(error_rate, 3),
        }))
        # Alert if error rate exceeds threshold
        if error_rate > 0.5:
            send_alert(f"Agent {self.run_id}: high error rate {error_rate:.0%}")
```

#### Replay Debugging

The most powerful debugging technique for agents is **run replay** — reproducing any historical run exactly from its stored trace. With LangSmith or Langfuse, every run's complete input/output chain is stored. You can click "replay" on a failing run to reproduce it locally, add breakpoints, and iterate on the system prompt or tool logic without waiting for the bug to occur again in production.

:::tip Build a replay harness from day one
Store every agent run as a JSON file: `{run_id, task, messages, tool_calls, final_output, metadata}`. A 20-line replay script that loads this file and re-runs the agent saves hours of debugging time when something goes wrong in production.
:::

### Topic: Streaming & Interrupts {#t-stream}

#### Token-Level Streaming with LangGraph

`stream_mode="messages"` streams token-by-token from every LLM node. Use `stream_mode="updates"` instead to stream full node outputs (less chatty, easier to process server-side).

```python
async def stream_agent(task: str):
    # stream_mode="messages" → AIMessageChunk per token from every LLM node
    async for chunk, metadata in app.astream(
        {"messages": [HumanMessage(content=task)]},
        stream_mode="messages",
    ):
        if chunk.content:
            yield chunk.content   # stream tokens to client

        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                if tc.get("name"):
                    yield f"\n[Calling tool: {tc['name']}]\n"

# Alternatively: stream full state updates per node (simpler for dashboards)
async def stream_updates(task: str):
    async for update in app.astream(
        {"messages": [HumanMessage(content=task)]},
        stream_mode="updates",   # {"node_name": state_delta} per step
    ):
        node_name, delta = next(iter(update.items()))
        print(f"[{node_name}] produced: {delta}")
```

### Topic: Deployment & Operations {#t-deploy}

#### Checkpointer Options

LangGraph ships with in-process memory by default. For durable persistence across restarts, choose a backend that fits your stack:

```python
from langgraph.checkpoint.memory import MemorySaver          # dev / testing
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver           # local / single-process
# pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver       # production
# pip install langgraph-checkpoint-redis
from langgraph.checkpoint.redis import RedisSaver             # high-throughput production

checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

# Always pass thread_id — this is the persistent conversation key
config = {"configurable": {"thread_id": "user-session-xyz"}}
result = app.invoke({"messages": [HumanMessage("do X")]}, config)
```

#### Queue-Based Async Execution with Celery

```python
from celery import Celery
celery_app = Celery("agent_worker", broker="redis://localhost:6379/0")

@celery_app.task(bind=True, max_retries=3)
def run_agent_task(self, task_id: str, user_task: str):
    try:
        config = {"configurable": {"thread_id": task_id}}
        result = agent_app.invoke(
            {"messages": [HumanMessage(content=user_task)]},
            config=config,
        )
        store_result(task_id, result["messages"][-1].content)
    except Exception as exc:
        self.retry(exc=exc, countdown=30)

# FastAPI: return task ID immediately, process in background
@app_api.post("/agent/tasks")
async def create_task(user_task: str) -> dict:
    task_id = str(uuid.uuid4())
    run_agent_task.delay(task_id, user_task)
    return {"task_id": task_id, "status": "queued"}
```

:::lab Lab 2.3 — Production Agent Deployment
**Objectives:**
- Deploy as a FastAPI service with streaming endpoint `/agent/stream` (SSE).
- Add Langfuse tracing on all LLM calls.
- SQLite checkpointer with a `human_review` node using `interrupt()` for tool approval gates.
- Stress test: 10 concurrent tasks — measure p50/p99 latency.
:::
