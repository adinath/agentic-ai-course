---
id: s11
number: "1.1"
title: "What is an Agent?"
time: "9:00–11:00 AM"
duration: "2 hours"
type: "Concepts & Architecture"
topics:
  - id: t-whatsagent
    title: "What is an AI Agent?"
  - id: t-agenttypes
    title: "Types of Agents"
  - id: t-prao
    title: "The PRAO Loop"
---

Welcome to the day where "just a chatbot" stops being an excuse. By session end you will be able to distinguish an agent from an LLM completion, classify agents by capability tier, and explain exactly why your agent sometimes gets stuck in a loop — without blaming the model.

### Topic: What is an AI Agent? {#t-whatsagent}

An LLM on its own is a **completion engine**: you send text, it generates text, it forgets everything. Useful — but about as autonomous as a very smart calculator.

An **AI Agent** is an LLM wired into a control loop that lets it:
1. **Perceive** — receive inputs from the world (messages, tool results, documents, sensor data).
2. **Reason** — decide what to do next via the model's internal forward pass.
3. **Act** — call tools, write files, send messages, query APIs.
4. **Observe** — capture the result of the action and feed it back for the next reasoning step.

This cycle — **Perception → Reasoning → Action → Observation (PRAO)** — is what distinguishes an agent from a one-shot LLM call.

#### The PRAO Loop

```
┌──────────────────────────────────────────────────────┐
│                     AI AGENT                         │
│                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │ Perceive │───▶│  Reason  │───▶│    Action    │  │
│   └──────────┘    └──────────┘    └──────────────┘  │
│         ▲                               │            │
│         │         Observation           │            │
│         └───────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
              │                     │
           Inputs               Environment
        (user, tools,          (files, APIs,
         memory)                databases)
```

:::tip Mental model vs. implementation
In code: call LLM → parse tool calls → execute tools → append results → call LLM again. The "reasoning" phase happens invisibly inside the model. Your job is to shape the inputs reliably and handle the outputs safely.
:::

#### Why Agents Fail — The Uncomfortable Truth

- **Infinite loops** — agent calls the same tool repeatedly expecting different results (the LLM definition of insanity).
- **Context overflow** — accumulated tool results exhaust the 200K context window faster than you expect.
- **Hallucinated tool calls** — model invents tool names or argument values that do not exist.
- **Goal drift** — agent optimises a subgoal so aggressively it forgets the original task.
- **Overconfidence** — acts on the first plan without verifying preconditions.

None of these are model bugs. They are architectural problems you can solve with proper loop design.

### Topic: Types of Agents {#t-agenttypes}

Not all agents are created equal. They form a capability ladder — each tier adds autonomy, complexity, and failure modes.

#### Tier 1 — Reactive Agents

Respond directly to the current input. No memory, no planning, no tools. Fast and deterministic.

```
Input ──▶ LLM ──▶ Output
```

**Examples:** GitHub Copilot inline completion, simple Q&A chatbots.
**Use when:** The task is fully self-contained in a single turn.

#### Tier 2 — Tool-Using Agents (ReAct)

Maintain a reasoning trace and can call tools to gather information before answering. Still largely reactive — each step is determined by the previous observation.

```
User Message
     │
     ▼
  LLM Reasons ──▶ Tool Call ──▶ Tool Result
     │◀──────────────────────────────────┘
     ▼
  LLM Reasons ──▶ Final Answer
```

**Examples:** Claude Code running a single bug fix, ChatGPT with web search.
**Use when:** Task requires external data or code execution but fits in one context window.

#### Tier 3 — Planning Agents

Generate a multi-step plan before acting. Can anticipate obstacles, parallelise subtasks, and replan on failure. The plan itself is a first-class object in the state.

```
Task ──▶ Planner ──▶ [Step 1, Step 2, Step 3]
                           │
                    Executor ──▶ Tools ──▶ Results
                           │
                    Replanner (on failure)
```

**Examples:** Devin-style coding agents, research agents that decompose a complex question.
**Use when:** Task spans multiple sessions, files, or requires coordination across capabilities.

#### Tier 4 — Multi-Agent Systems

Multiple specialised agents coordinated by an orchestrator. Each agent has its own context, tools, and memory. The system can parallelize independent subtasks and handle workloads that exceed a single context window.

**Examples:** PR review pipeline with security + performance + style agents running in parallel.
**Use when:** Single-agent ceiling genuinely limits you — not before.

:::example Real-world mapping
**Reactive:** GitHub Copilot<br/>
**Tool-using:** Claude Code / single task<br/>
**Planning:** Devin / full feature request<br/>
**Multi-agent:** Enterprise CI pipeline with specialized review agents
:::

#### Choosing the Right Tier

```python
def choose_agent_tier(task: str) -> str:
    """
    Decision framework for agent architecture selection.
    Start at the lowest tier that can handle the task.
    """
    # Single turn, no external data needed → Reactive
    if is_self_contained(task) and no_tools_needed(task):
        return "reactive"

    # Needs tools but fits in one context window → Tool-using
    if needs_tools(task) and fits_single_context(task):
        return "react_agent"

    # Multi-step, needs planning, may need replanning → Planning
    if requires_planning(task) or has_dependencies(task):
        return "plan_and_execute"

    # Exceeds context, parallel subtasks, specialist knowledge → Multi-agent
    return "multi_agent"
```

### Topic: The PRAO Loop in Code {#t-prao}

Before touching any framework, implement the loop from scratch. Every framework is just this with more bells and whistles.

```python
# Bare-bones PRAO agent loop — production-ready error handling included
import json
from anthropic import Anthropic

client = Anthropic()
MAX_ITERATIONS = 20

def run_agent(task: str, tools: list, tool_registry: dict) -> str:
    """
    Core PRAO loop. Returns the agent's final response.

    Args:
        task: The user's task description.
        tools: List of tool schemas (Anthropic tool format).
        tool_registry: Dict mapping tool name → callable.
    """
    messages = [{"role": "user", "content": task}]
    consecutive_errors = 0

    for iteration in range(MAX_ITERATIONS):
        # --- REASONING: call the LLM ---
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # Append assistant message (PRAO: this is the "Action" decision)
        messages.append({"role": "assistant", "content": response.content})

        # --- TERMINAL STATE: no tool calls → agent is done ---
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # --- ACTION: execute tool calls ---
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input

            try:
                if tool_name not in tool_registry:
                    raise ValueError(f"Unknown tool: {tool_name}")
                result = tool_registry[tool_name](**tool_args)
                consecutive_errors = 0
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
            except Exception as e:
                consecutive_errors += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"ERROR: {e}",
                    "is_error": True,
                })

        # --- OBSERVATION: append tool results to context ---
        messages.append({"role": "user", "content": tool_results})

        # Safety rail: abort after 3 consecutive errors
        if consecutive_errors >= 3:
            return "Agent stopped: too many consecutive tool errors."

    return "Agent stopped: max iterations reached."


# Example usage
if __name__ == "__main__":
    import os

    def read_file(path: str) -> str:
        with open(path) as f:
            return f.read()

    tools = [{
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"],
        },
    }]

    result = run_agent(
        task="Read the file README.md and summarise it in 3 bullet points.",
        tools=tools,
        tool_registry={"read_file": read_file},
    )
    print(result)
```

:::lab Lab 1.1 — Implement the PRAO Loop from Scratch
**Objectives:**
- Implement `run_agent()` without any framework (just the Anthropic SDK).
- Give the agent two tools: `bash_exec(command)` and `read_file(path)`.
- Task: "Count all TODO comments in this repo, list the file names and line numbers."
- Log every iteration: tool name, args, result (first 200 chars), and iteration number.
- Observe: how does the agent decide it's done? What triggers the `end_turn` stop reason?
:::
