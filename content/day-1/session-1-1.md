---
id: s11
number: "1.1"
title: "What is an Agent?"
time: "9:00–11:00 AM"
duration: "2 hours"
type: "Concepts & Architecture"
topics:
  - id: t-whatsagent
    title: "LLM vs. Agent"
  - id: t-prao
    title: "The PRAO Loop"
  - id: t-agenttypes
    title: "Classic Agent Types"
  - id: t-ladder
    title: "Capability Ladder & PEAS"
---

Welcome to the day where "just a chatbot" stops being an excuse. By session end you will be able to distinguish an agent from an LLM completion, classify any agent — classic or LLM-era — by capability tier, and explain exactly why your agent sometimes gets stuck in a loop without blaming the model.

### Topic: LLM vs. Agent {#t-whatsagent}

A bare LLM is a **completion engine**: text in, text out, no memory between calls. A single forward pass through the network. You send a prompt, it generates tokens, it forgets everything. Useful — but about as autonomous as a very smart calculator.

```
┌───────────────────────────────┐     ┌──────────────────────────────────────┐
│ LLM (completion)              │     │ Agent                                │
│ Input → Model → Output        │     │ Perceive → Reason → Act → Observe → ↻│
│                               │     │                                      │
│ Stateless · one shot          │     │ Stateful loop · uses tools           │
│ Forgets everything            │     │ Accumulates context · adapts         │
└───────────────────────────────┘     └──────────────────────────────────────┘
```

An **AI agent** is the *same* LLM wrapped in a feedback loop with tools and memory. The model still does only one thing — produce the next token — but it now operates inside a control structure that lets it sense the world, decide, act, and check the result.

:::tip Mental model vs. implementation
In code: call LLM → parse tool calls → execute tools → append results → call LLM again. The "reasoning" phase happens invisibly inside the model. Your job is to shape the inputs reliably, run the tools safely, and stop the loop when work is done.
:::

### Topic: The PRAO Loop {#t-prao}

The differentiator is a four-step cycle: **Perception → Reasoning → Action → Observation (PRAO)**. The loop-back from O to P is the one line of architecture that turns a stateless model into an autonomous agent.

```
┌────────────────────────────────────────────────────────────────┐
│                          AGENT                                 │
│                                                                │
│   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐  │
│   │ Perceive │──▶│  Reason  │──▶│ Action  │──▶│ Observation│  │
│   │ user msg │   │ LLM call │   │ tool    │   │ result     │  │
│   │ tool out │   │ pick     │   │ API hit │   │ env feed   │  │
│   └──────────┘   └──────────┘   └─────────┘   └─────┬──────┘  │
│         ▲                                            │         │
│         └─────────── feeds next turn ────────────────┘         │
│                  (until done or stopped)                       │
└────────────────────────────────────────────────────────────────┘
```

| Phase | What happens | Concrete example |
|-------|--------------|------------------|
| **Perception** | Read the world | User prompt, file content, sensor reading, prior tool result |
| **Reasoning** | Decide the next move | Single LLM forward pass — produce text or tool call |
| **Action** | Change the world | Call a tool, hit an API, write a file, send a message |
| **Observation** | See what happened | Tool result, exception, environment feedback |

#### Why Agents Fail — The Architectural Truth

None of these are model bugs. They are loop-design problems you must solve in *your* code, not the LLM:

- **Infinite loops** — agent calls the same tool repeatedly expecting different results.
- **Context overflow** — accumulated tool results exhaust the 200K window faster than you expect.
- **Hallucinated tool calls** — model invents a tool name or argument that does not exist.
- **Goal drift** — agent optimises a sub-goal so aggressively it forgets the original task.
- **Overconfidence** — acts on the first plan without verifying preconditions.

#### The PRAO Loop in Code

Before touching any framework, implement the loop from scratch. Every framework is just this with more bells and whistles.

```python
import json
from anthropic import Anthropic

client = Anthropic()
MAX_ITERATIONS = 20

def run_agent(task: str, tools: list, tool_registry: dict) -> str:
    """Bare-bones PRAO loop. Returns the agent's final response."""
    messages = [{"role": "user", "content": task}]
    consecutive_errors = 0

    for iteration in range(MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                if block.name not in tool_registry:
                    raise ValueError(f"Unknown tool: {block.name}")
                result = tool_registry[block.name](**block.input)
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

        messages.append({"role": "user", "content": tool_results})

        if consecutive_errors >= 3:
            return "Agent stopped: too many consecutive tool errors."

    return "Agent stopped: max iterations reached."
```

### Topic: Classic Agent Types {#t-agenttypes}

Russell & Norvig's *Artificial Intelligence: A Modern Approach* (AIMA) classifies agents into five canonical architectures. Every modern LLM agent is, structurally, one of these — usually a **learning, utility-based** agent where language is both percept and action. Knowing the taxonomy stops you from reaching for "multi-agent" when a reflex rule will do.

#### 1. Simple Reflex Agent

A direct lookup from current percept to action. No memory, no world model. The fastest agent — and the most fragile.

```
ENVIRONMENT
  │ percept
  ▼
┌─────────────────┐         ┌──────────────────────┐
│ "What is the    │ ──────▶ │ Condition–Action     │
│  world like     │         │ Rules                │
│  now?"          │         │ if X → do Y          │
└─────────────────┘         └──────────────────────┘
                                     │
                                     ▼
                                 ACTUATORS
```

- **Key idea:** maps percept directly to action via condition–action rules.
- **Works only** in fully observable environments.
- **Real examples:** thermostat, regex spam filter, deterministic form validator, rate-limit middleware.
- **LLM-era equivalent:** stateless prompt → completion (e.g. inline code suggestion).

#### 2. Model-Based Reflex Agent

Adds an internal **state** that tracks what cannot currently be perceived. State is updated each tick from the percept, the last action, and a model of how the world evolves.

```
percept ──┐
          ▼
   ┌───────────┐    ┌──────────────────┐    ┌───────────────────┐
   │  State    │───▶│ How world evolves│───▶│ Condition–Action  │──▶ action
   │ (memory)  │    │ What my acts do  │    │ Rules             │
   └───────────┘    └──────────────────┘    └───────────────────┘
        ▲
        └────────── last_action feedback ──────────
```

- **Key idea:** maintains a world model so the agent can act in *partially observable* environments.
- **Real examples:** Roomba mapping cleaned rooms, conversational chatbot with short-term memory, TCP retransmit logic.
- **LLM-era equivalent:** ReAct loop with conversation memory + tool calls.

#### 3. Goal-Based Agent

Acts to *reach* an explicit goal. Considers the future. Action is chosen by simulating outcomes against the goal — search and planning live here.

```
percept ──▶ State ──▶ "What happens if I do A?" ──▶ Action selector
                                                       │ goal-aware
                                                       ▼
                                                    ACTUATORS
                              ▲
                              │
                            Goals (deliver_pkg, reach_X)
```

- **Key idea:** action chosen by simulating outcomes against an explicit goal.
- **Real examples:** GPS routing, chess engine minimax, build-system dependency resolver.
- **LLM-era equivalent:** Plan-and-Execute agents (Devin, AutoGPT-style).

#### 4. Utility-Based Agent

Generalises goals to a continuous **utility function** `U(state) → ℝ`. Trades off competing objectives (speed vs. cost vs. safety) and handles uncertainty via expected utility — `argmax_a U(predict(state, a))`.

- **Key idea:** picks the action with the highest expected utility, not just *a* goal-satisfying one.
- **Real examples:** ride-share dispatcher, autonomous-driving cost map, A/B test bandit.
- **LLM-era equivalent:** cost / latency / quality routing — picking Haiku vs. Sonnet vs. Opus per query.

#### 5. Learning Agent

A **meta-architecture** that wraps any of the four prior types. Four parts: **Critic** measures performance against a fixed standard; **Learning Element** edits the **Performance Element**; **Problem Generator** suggests exploratory actions.

```
                     ┌─────────────┐
       feedback ───▶ │   Critic    │ ──── changes ────┐
                     └─────────────┘                  ▼
                                              ┌──────────────────┐
   percept ──▶ Performance Element ───────▶  │ Learning Element │
                       │                      └──────────────────┘
                       ▼                              │
                    action ◀──── exploratory ──── Problem Generator
```

- **Key idea:** improves over time. Any prior type can become a learning agent.
- **Real examples:** AlphaGo (self-play), RLHF-tuned LLMs, eval-driven prompt tuning, online recommenders with bandit exploration.

### Topic: Capability Ladder & PEAS {#t-ladder}

Each type adds *one* capability the previous one lacked. Pick the simplest type that solves the problem — complexity must be earned, never assumed.

```
Reflex  ─+state─▶  Model-Based  ─+goals─▶  Goal-Based
                                              │
                                              + preferences
                                              ▼
              Learning  ◀─ +adaptation ─  Utility-Based
```

#### Classic Theory → LLM Era

| Classic type | LLM-era manifestation | Example |
|---|---|---|
| Simple Reflex | Stateless prompt → completion | Inline code suggestion |
| Model-Based Reflex | Conversation memory + tool calls | ReAct loop, Claude Code |
| Goal-Based | Plan-and-execute agents | Devin, AutoGPT |
| Utility-Based | Cost / latency / quality routing | Model routing, judge-based selection |
| Learning | RLHF, in-context learning, eval-driven loops | RLHF-tuned models, prompt-tuning harnesses |

#### PEAS — The 30-Year-Old Vocabulary That Still Works

Before building any agent, specify **PEAS**:

- **P**erformance measure — how do you know it's working? (Latency, accuracy, refusal rate, user score.)
- **E**nvironment — what does the agent operate in? (Codebase, web, OS, internal API.)
- **A**ctuators — what tools / APIs can it use?
- **S**ensors — what inputs does it read? (User text, file content, retrieval, sensor data.)

:::tip Skip PEAS at your peril
Most "the agent doesn't work" production bugs are unstated PEAS. The agent's actuator list is fuzzy, its performance measure is "vibes", or its environment includes a system the developer didn't realise the agent could reach. Write PEAS down on day one and revisit it whenever you change a tool.
:::

:::example The cardinal sin
Reaching for "learning" or "multi-agent" when a reflex rule would do. Most production systems labelled "AI agents" are actually goal-based agents wrapped in a thin tool harness. That is *fine* — it is also enough.
:::

:::lab Lab 1.1 — Implement the PRAO Loop from Scratch
**Objectives:**
- Implement `run_agent()` without any framework (just the Anthropic SDK).
- Give the agent two tools: `bash_exec(command)` and `read_file(path)`.
- Task: "Count all TODO comments in this repo, list the file names and line numbers."
- Log every iteration: tool name, args, result (first 200 chars), and iteration number.
- Observe: how does the agent decide it's done? What triggers the `end_turn` stop reason?
- Reflect: classify your agent on the AIMA ladder. Which capabilities did you skip — and why?
:::
