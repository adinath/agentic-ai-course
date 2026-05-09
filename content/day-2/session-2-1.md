---
id: s21
number: "2.1"
title: "AIFSD & MCP"
time: "9:00–11:00 AM"
duration: "2 hours"
topics:
  - id: t-aifsd
    title: "AI-First Software Development"
  - id: t-sensai
    title: "Sens-AI Framework"
  - id: t-rulescommandsskills
    title: "Rules, Commands & Skills"
  - id: t-harness
    title: "Harness Engineering"
  - id: t-mcp
    title: "Model Context Protocol"
  - id: t-skills
    title: "Agent Skills"
---

Yesterday you built agents that execute tasks. Today you learn to *develop with* agents as first-class collaborators. AIFSD is the methodology; the Sens-AI Framework is the mental model; Rules/Commands/Skills is the configuration layer; MCP is the wiring standard.

### Topic: AI-First Software Development (AIFSD) {#t-aifsd}

#### The Shift

Traditional development: developer writes code, runs tests, ships.
AI-First Software Development (AIFSD): developer + AI agent pair-program. The agent handles mechanical execution while the developer provides strategic direction — framing, constraints, judgment.

The critical distinction: AI is a **collaborator**, not an autocomplete on steroids. That distinction changes how you interact with it.

```
┌─────────────────────────────────────────────────────────────────┐
│                  AIFSD COLLABORATION MODEL                       │
│                                                                  │
│   Developer                         AI Agent                    │
│  ──────────                         ────────                    │
│  Strategic intent       ◀──▶        Task execution              │
│  Problem framing        ◀──▶        Code generation             │
│  Architecture decisions ◀──▶        Pattern matching            │
│  Quality judgment       ◀──▶        Tool orchestration          │
│  Context provision      ◀──▶        Test execution              │
└─────────────────────────────────────────────────────────────────┘
```

#### The Rehash Loop — Your Early Warning Signal

The most common failure mode in AI-assisted development is the **rehash loop**: the AI keeps generating slight variations of the same wrong answer no matter how you adjust the prompt. Most developers respond by tweaking prompts further. That doesn't fix it.

The rehash loop is not a prompt problem. It is a signal that the AI has exhausted the context you've given it. Stop prompting. Apply the Sens-AI habits instead.

### Topic: Sens-AI Framework {#t-sensai}

The **Sens-AI Framework** was introduced by Andrew Stellman in O'Reilly Radar (2025). It defines five critical thinking habits that turn AI from an autocomplete tool into a genuine engineering collaborator — without losing your own judgment in the process.

:::info Source
Sens-AI Framework by Andrew Stellman · O'Reilly Radar 2025
Reference: oreilly.com/radar/the-sens-ai-framework/
:::

#### The Reinforcing Loop

The five habits are not a checklist — they form a **reinforcing loop**. Critical Thinking reveals what Context was missing; the loop restarts with better inputs, compounding quality over time.

```
┌──────────────────────────────────────────────────────────────────┐
│             SENS-AI REINFORCING LOOP                              │
│                                                                   │
│   Context → Research → Problem Framing → Refining                │
│      ↑                                       │                   │
│      └─────────── Critical Thinking ◀────────┘                   │
│                                                                   │
│   "Critical Thinking reveals what Context was missing —           │
│    the loop restarts, compounding quality over time."             │
└──────────────────────────────────────────────────────────────────┘
```

---

#### Habit 01 — Context

> *Paying attention to what information you supply to the model, trying to figure out what else it needs to know, and supplying it clearly — code, comments, structure, intent, constraints, and what you've already tried.*

AI fills gaps with assumptions — and those assumptions are almost always wrong for your specific situation. Rich context is not a courtesy; it is the difference between a useful answer and a plausible-looking hallucination.

**Thin context vs. rich context:**

```
❌ Thin: "Fix the bug in my code."
   No language, no framework, no expected behaviour, no constraints.

✅ Rich: "This Express JWT middleware returns 200 on expired tokens
   instead of 401. The token is verified with jsonwebtoken v9.
   I've already tried catching the TokenExpiredError explicitly
   but it's not reached."
```

**What rich context includes:**
- The relevant code snippet (not the whole file)
- Your intent — what it is *supposed* to do, not just what it does
- Constraints — performance, readability, team conventions, architecture
- What you've already tried — so AI doesn't repeat failed paths
- The domain — "this is a high-frequency trading system" changes everything

:::tip Watch for context drift
In a multi-turn conversation, the model's understanding degrades over time. When output starts to feel off, ask the AI to restate what it thinks the code does. If the summary is wrong, you've found the drift — and the gap to fill.
:::

---

#### Habit 02 — Research

> *Actively using AI and external sources to deepen **your own** understanding of the problem — running examples, consulting documentation, checking references to verify what's really going on. The goal is you getting smarter, not just collecting info to paste.*

Most developers try to *prompt their way out* of a stuck situation. Research says: stop prompting, start understanding. The rehash loop isn't a prompt problem — it's a knowledge gap. Fill the gap first.

**Research sources — AI is one, not the only one:**

| Source | Use For |
|--------|---------|
| Ask AI to explain itself | Fastest first step — ask it to explain what it just generated, not fix it. If the explanation is wrong, you've found the gap. |
| Official documentation | AI training data may be stale. Go to source for library APIs, framework versions, breaking changes. |
| Your own codebase | How was this solved before? Existing patterns are context the AI will never have. |
| Stack Overflow / GitHub Issues | Known bugs, version incompatibilities, edge cases the community has hit. |

:::info Research feeds framing
What you learn through research becomes the raw material for a sharper problem definition. You can't frame a problem you don't understand. Research isn't a detour — it's the on-ramp to Habit 3.
:::

---

#### Habit 03 — Problem Framing

> *Using the information you've gathered to define the problem more clearly so the model can respond more usefully. Most AI failures are framing failures — the AI solved the problem you described, not the problem you had.*

Prompt engineering is requirements engineering. The challenges are the same ones software teams have grappled with for decades: scoping, specifying functional and nonfunctional requirements, communicating intent precisely.

**The four elements of a well-framed prompt:**

| Element | What it means |
|---------|---------------|
| What you want | A specific functional outcome — "accept input as a string parameter" not "make it more testable." |
| Constraints | Performance, readability, team standards, architectural decisions. |
| What you **don't** want | Explicitly close off overengineered paths — "no new dependencies", "don't change the public interface." |
| Definition of done | How will you know the answer is right? Prevents stopping too early or going too far. |

**Vague frame vs. precise frame:**

```
❌ Vague: "Make this code more testable."
   AI adds interfaces, mock objects, dependency injection throughout —
   a miniature framework where a simple refactor was needed.

✅ Precise: "Refactor this method to accept a string parameter instead
   of reading from stdin, so I can pass test input directly.
   Don't change the class structure."
   → One surgical change. No abstraction layers added.
```

---

#### Habit 04 — Refining

> *Iterating your prompts deliberately. This isn't about random tweaks; it's about making targeted changes based on what the model got right and what it missed, and using those results to guide the next step.*

**Vibe coding vs. deliberate refining:**

```
❌ Vibe coding (random tweaks):
   Prompt → output → "make it more X" → same answer rephrased → repeat.
   Leads directly to the rehash loop.

✅ Deliberate refining:
   Read output → diagnose what was right, what was specifically wrong
   → make one targeted change → re-prompt.
```

**The deliberate refining loop:**

1. **Read the output** — don't skim; understand what was actually produced.
2. **Diagnose specifically** — "the error handling is right but business logic is in the controller" — not "this is bad."
3. **Make one targeted change** — address the specific gap you identified.
4. **Also refine the code** — clean names, reduce duplication, verify architecture fit. Don't stop when it compiles.

:::tip Agile parallel
Just as Agile shifted requirements from static specs to living conversations, prompt refining shifts AI interaction from single-shot commands to iterative improvement — except you must infer what's missing from the output rather than having the AI ask clarifying questions.
:::

---

#### Habit 05 — Critical Thinking

> *Judging the quality of AI output rather than simply accepting it. Does the suggestion make sense? Is it correct, relevant, plausible? This habit is especially important because AI sounds confident even when it's wrong.*

AI doesn't truly understand your codebase. It mimics patterns from training, producing answers that look right. It's very often correct — which is why vibe coding works at all. But it rarely accounts for overall architecture, long-term strategy, or good design principles. Those gaps are your responsibility.

**Three concrete techniques:**

**A. Generate alternatives** — Ask for 2-3 solutions. Comparing them reveals different assumptions and trade-offs, and keeps your own judgment engaged rather than passive.

**B. Use AI as its own critic** — After it generates code, ask it to review that same code for problems. The context shift surfaces edge cases and design issues it didn't catch the first time.

**C. Read, run, debug — then ask "will this make sense in 6 months?"** — Is coupling creeping in? Are names clear? Does this fit the architecture? Catching this now is far cheaper than after it's woven through the codebase.

**Signs the habit is taking hold in your team:**
- Asking "why did AI choose this pattern?" not just "does it work?"
- Relating AI output to readability, separation of concerns, testability
- Reviewing AI-generated code with the same rigour as human-written code
- Treating AI failures as learning opportunities, not blame

:::warning The long-term reality
More convincing output will require more sophisticated evaluation. Models will keep improving. What won't change is the need for developers to think critically about the code in front of them. As AI gets better, this habit becomes *more* important — not less.
:::

---

#### Sens-AI Quick Reference

| # | Habit | The question to ask yourself |
|---|-------|------------------------------|
| 01 | Context | "What does AI need to know that it doesn't?" |
| 02 | Research | "Do I understand this well enough to guide AI?" |
| 03 | Problem Framing | "Am I solving the right problem, precisely defined?" |
| 04 | Refining | "What specifically was wrong, and what one thing should I change?" |
| 05 | Critical Thinking | "Is this correct, sound, and will it hold up in 6 months?" |

### Topic: Rules, Commands & Skills {#t-rulescommandsskills}

With the Sens-AI habits as your mental model, the next step is configuring your AI agent's operating environment. In Claude Code and similar tools, this is done through three mechanisms: **Rules**, **Commands**, and **Skills**.

#### Rules — Agent Operating Constraints

Rules define who the agent is, what it can and cannot do, and the norms it must follow. In Claude Code, rules live in a `CLAUDE.md` file checked into the repository — every session loads them automatically.

Good rules reflect the Sens-AI habits: they provide **Context** (domain, stack, team conventions) and **Framing** (what good output looks like) upfront, so the agent never starts from a blank slate.

```markdown
# CLAUDE.md — Rules for the payments-service repo

## Role
Senior backend engineer on the payments-service.
Stack: Python 3.12, FastAPI, PostgreSQL, Redis.

## Constraints
- Never modify existing database migrations — always create new ones.
- Run the full test suite before declaring any fix complete.
- Never commit secrets, API keys, or credentials.
- All new endpoints must have input validation and return typed responses.
- Use `structlog` for logging, never the standard `logging` module.

## Escalation Rules
- If a change touches auth or payments logic, ask for human review.
- If tests fail after 2 attempts, stop and report root cause — don't guess further.

## Definition of Done (Refining habit applied to agent output)
- Tests pass
- New code has type hints
- No new linting errors
- PR description explains *why*, not just *what*
```

#### Commands — Predefined Workflows

Commands are slash-invoked agent workflows. They apply the **Problem Framing** habit in advance — the prompt is pre-engineered for each recurring task so you don't re-frame from scratch every time.

```markdown
# Available commands in this project

/review  — Review the current git diff: bugs, security issues, style violations.
/test    — Run the test suite and report failures with suggested root causes.
/migrate — Generate a new Alembic migration for pending model changes.
/doc <fn> — Generate or update the docstring for the given function.
/explain <file> — Explain what this file does and its role in the system.
/audit   — Scan for security vulnerabilities and generate a remediation report.
```

#### Skills — Reusable Agent Workflows

Skills are self-contained agent sub-procedures with a typed input/output contract. They operationalise the **Critical Thinking** habit: each skill includes an explicit evaluation step so the agent assesses its own output before returning it.

```python
# skills/code_review_skill.py
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

class ReviewFinding(BaseModel):
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["security", "correctness", "performance", "style"]
    line_range: str
    description: str
    suggestion: str

class ReviewResult(BaseModel):
    findings: list[ReviewFinding]
    overall_verdict: Literal["approve", "request_changes", "comment"]
    summary: str
    # Critical Thinking step: self-assessment of review quality
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this review's completeness")
    potential_missed_issues: list[str] = Field(description="Issues this review may have missed")

REVIEW_SYSTEM = """You are a senior engineer performing a code review.
Analyse the diff for: security vulnerabilities, correctness issues,
performance problems, and style violations.
Be specific — cite line numbers and provide actionable suggestions.
After reviewing, honestly assess what you might have missed (Critical Thinking)."""

def code_review_skill(diff: str, context: str = "") -> ReviewResult:
    """Reusable code review skill — applies all Sens-AI habits internally."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    reviewer = llm.with_structured_output(ReviewResult)
    return reviewer.invoke([
        SystemMessage(content=REVIEW_SYSTEM),
        HumanMessage(content=f"Context: {context}\n\nDiff:\n```diff\n{diff}\n```"),
    ])
```

#### Sub-Agents: Local and Cloud

Rules/Commands/Skills configure a single agent. Sub-agents take delegation further — specialist agents that a parent agent calls for specific work, either in-process (local) or over HTTP (cloud).

```python
import asyncio, httpx, os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# LOCAL sub-agent — runs in the same Python process
def build_security_agent():
    """Security specialist — fast, no network, shares process memory."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    llm_with_tools = llm.bind_tools([scan_dependencies, check_secrets, audit_sql])

    def call_model(state):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    g = StateGraph({"messages": list})
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode([scan_dependencies, check_secrets, audit_sql], handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()

security_agent = build_security_agent()

# CLOUD sub-agent — HTTP delegation to a remote specialist service
async def invoke_cloud_agent(agent_url: str, task: str) -> str:
    """Delegate to a remotely deployed agent. Returns its final answer."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{agent_url}/runs",
            json={"input": {"messages": [{"role": "user", "content": task}]}},
            headers={"Authorization": f"Bearer {os.environ['AGENT_API_KEY']}"},
        )
        response.raise_for_status()
        result = response.json()
        return result["output"]["messages"][-1]["content"]

# Parent orchestrator: parallel local + cloud delegation
async def orchestrate_review(pr_diff: str) -> dict:
    local_result, cloud_result = await asyncio.gather(
        security_agent.ainvoke({"messages": [HumanMessage(pr_diff)]}),
        invoke_cloud_agent(os.environ["PERF_AGENT_URL"], pr_diff),
    )
    return {
        "security": local_result["messages"][-1].content,
        "performance": cloud_result,
    }
```

### Topic: Harness Engineering {#t-harness}

> *Agent = Model + Harness.*
> — Birgitta Böckeler · martinfowler.com/articles/harness-engineering · 2026

LLMs are non-deterministic, don't know your codebase, and think in tokens — not in your domain. The **harness** is everything around the model that closes that gap: the rules, skills, MCPs, linters, tests, and review agents that **steer** it before it acts and **catch** it after. Rules / commands / skills aren't *the* harness — they're its *building blocks*.

```
                     User Harness
       ┌─────────── you build this ───────────┐
       │   Builder Harness (system prompt,    │
       │   retrieval, orchestration)          │
       │   ┌────────────────────────────┐     │
       │   │       Model                │     │
       │   │     tokens in / out        │     │
       │   └────────────────────────────┘     │
       │     rules · skills · MCPs ·          │
       │     linters · review agents          │
       └──────────────────────────────────────┘

   FEEDFORWARD (guides)        FEEDBACK (sensors)
   steer before it acts        observe after it acts
   AGENTS.md · skills · LSP    linters · tests · review agents
```

#### Two Axes — Direction × Execution

The strongest harness mixes all four cells. Feedforward-only and the agent never finds out whether the rules worked. Feedback-only and it repeats the same mistakes. Computational catches structure cheaply; inferential catches semantics expensively.

|  | Computational (deterministic) | Inferential (LLM-driven) |
|---|---|---|
| **Feedforward · guides** | Type systems, language servers, `.editorconfig`, codemods, scaffolding, framework conventions. *Cheap, reliable, every run.* | `AGENTS.md`, `CLAUDE.md`, skills, how-tos, architecture docs, MCPs that surface team knowledge. *Slower, encodes intent.* |
| **Feedback · sensors** | Linters, type-checkers, unit tests, ArchUnit, dep-cruiser, mutation testing, coverage gates. *Best when their messages are written for the agent.* | Review agents, LLM-as-judge, response samplers, log-anomaly detectors, drift scanners. *Expensive — schedule them off the hot path.* |

#### Three Things to Regulate

| Harness | What it covers | Tooling maturity |
|---|---|---|
| **Maintainability** | Internal code quality: duplication, complexity, coverage, style, drift | **HIGH** — decades of pre-existing tooling (linters, type-checkers, ArchUnit). Plug in and pay back immediately. |
| **Architecture Fitness** | Non-functional: performance, observability, security posture, resilience | **EMERGING** — encoded as fitness functions. Workable, mostly bespoke per team. |
| **Behaviour** | Does the app actually do what the user wanted? | **OPEN PROBLEM** — spec in, AI-written tests out. Approved-fixtures and human-curated suites help, but the gap is real. *Where supervision still earns its keep.* |

:::tip Harnessability is a property of your codebase
Strong types, clear module boundaries, mature frameworks → easy to harness. Untyped, monolithic, legacy → the harness is hardest to build exactly where it's most needed. **Greenfield teams should design for harnessability on day one.**
:::

#### The Steering Loop — Keep Quality Left

Spread checks by cost. The earlier a sensor fires, the cheaper the fix.

```
  GUIDES (feedforward)              SENSORS (feedback)              SCOPE
  ─────────────────────             ────────────────────             ─────
  AGENTS.md, skills, LSP    ─▶ AGENT ─▶  linters, type-check,        PRE-COMMIT
  ref docs, codemods                     unit tests, /code-review        cheap · every change
                                              │
                                              ▼  fixes self-applied
                                         human review                 PRE-INTEGRATION
                                              │                            judgment · social accountability
                                              ▼
                                  ─── INTEGRATION ───
                                              │
                                              ▼
                                         rerun fast sensors           POST-INTEGRATION
                                         + mutation tests             broader · slower
                                         + /architecture-review
                                              │
                                              ▼
                                  CONTINUOUS DRIFT & HEALTH          OFF-PATH
                                  /find-dead-code · dependabot              run on a schedule
                                  /coverage-quality · log-anomalies         feeds new commits
                                  SLO watchers · janitor agents
```

**Three operating principles:**

- **Iterate the harness, not just the prompt.** When the same mistake recurs, don't re-prompt — *add a guide or a sensor*. Coding agents make custom linters and structural tests cheap to write.
- **Aim humans at the hard part.** A good harness doesn't replace human review — it *redirects* it. Computational sensors handle structure; inferential handles most semantics; humans handle correctness, intent, trade-offs.
- **Treat the harness as code.** Version it, review it, regress on it. The harness is now part of your engineering output.

### Topic: Model Context Protocol (MCP) {#t-mcp}

#### What MCP Solves

Before MCP: every agent needed custom integration code per tool. **N agents × M tools = N·M integrations.**
After MCP: write a tool server once, any compatible agent uses it. **N + M.**

MCP is an open standard introduced by Anthropic in late 2024. It defines three roles communicating over **JSON-RPC 2.0** — usually transported over `stdio` (local subprocess) or **streamable HTTP** (remote, OAuth-secured).

#### Three Roles, One Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                  MCP HOST (your AI app)                          │
│  ┌─────────────────────────────────────────┐                    │
│  │           LLM Engine                    │                    │
│  │   Claude · GPT · Gemini · Llama         │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │   Client #1    │   │   Client #2    │   │   Client #3    │  │
│  │  stdio         │   │  http+sse      │   │  stdio         │  │
│  └───────┬────────┘   └───────┬────────┘   └───────┬────────┘  │
└──────────┼────────────────────┼─────────────────────┼───────────┘
           │ JSON-RPC 2.0       │                     │
           ▼                    ▼                     ▼
   ┌──────────────┐    ┌──────────────┐      ┌──────────────┐
   │ Filesystem   │    │   GitHub     │      │  PostgreSQL  │
   │ MCP Server   │    │ MCP Server   │      │  MCP Server  │
   └──────────────┘    └──────────────┘      └──────────────┘
```

**HOST** — the user-facing AI app (Claude Desktop, Cursor, Windsurf, Continue, your LangGraph agent). Owns the LLM, UI, *and the consent layer* — every tool call passes through host approval before reaching a server. Five responsibilities: spawn clients · aggregate capabilities · route LLM tool calls to the right client · gate consent · own the UI / auth.

**CLIENT** — an in-process module of the host. Holds *exactly one* stateful connection to *exactly one* server. `N servers ⇒ N clients`. Speaks JSON-RPC. Lifecycle: `initialize` → server returns `capabilities, tools, resources, prompts` → `notifications/initialized` → runtime `tools/call` requests.

**SERVER** — a standalone process exposing **three capability types**:

| Type | What it is | Examples |
|---|---|---|
| **Tools** | Executable functions, model-invoked, side effects allowed | `write_file`, `run_query`, `send_email` |
| **Resources** | Read-only data sources, addressed by URI | `file://README.md`, `postgres://schema` |
| **Prompts** | User-triggered templates exposed via slash-menu | `/summarize_pr`, `/explain_query` |

:::tip Mental model
Host = the browser tab. LLM = the engine inside it. Each MCP client = one fetch connection (one per origin). Servers = the websites being fetched. The deal: wrap any system once as an MCP server — local CLI, internal API, vector DB — and *every* MCP host can use it without bespoke glue.
:::

#### Using MCP Servers from LangGraph

```bash
pip install "langchain-mcp-adapters>=0.1" mcp
npm install -g @modelcontextprotocol/server-filesystem
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

async def build_mcp_agent():
    async with MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
                "transport": "stdio",
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "transport": "stdio",
                "env": {"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
            },
        }
    ) as client:
        tools = await client.get_tools()  # auto-discovered — no schema writing
        print(f"Discovered {len(tools)} tools from MCP servers")

        agent = create_react_agent(
            ChatAnthropic(model="claude-sonnet-4-6", temperature=0),
            tools,
        )
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "List all open PRs in the repo"}]
        })
        return result
```

#### Building a Custom MCP Server

```python
# my_tools_server.py — expose your tools to any MCP-compatible agent
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio, subprocess

app = Server("dev-tools")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_linter",
            description="Run ruff linter on a file and return issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File to lint"},
                    "fix": {"type": "boolean", "default": False},
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="get_test_coverage",
            description="Return test coverage % for the given module.",
            inputSchema={
                "type": "object",
                "properties": {"module": {"type": "string"}},
                "required": ["module"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "run_linter":
        cmd = ["ruff", "check", arguments["file_path"]]
        if arguments.get("fix"):
            cmd.append("--fix")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return [types.TextContent(type="text", text=result.stdout + result.stderr)]

    if name == "get_test_coverage":
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/",
             f"--cov={arguments['module']}", "--cov-report=term-missing", "-q"],
            capture_output=True, text=True, timeout=60,
        )
        return [types.TextContent(type="text", text=result.stdout)]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

#### Connecting to Claude Code

Configure MCP servers in `.claude/settings.json` (project-level) or `~/.claude/settings.json` (global):

```json
{
  "mcpServers": {
    "dev-tools": {
      "command": "python",
      "args": ["./scripts/my_tools_server.py"],
      "description": "Custom dev tools: linter, coverage"
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": { "DATABASE_URL": "${DATABASE_URL}" }
    }
  }
}
```

Claude Code discovers and uses these tools automatically — no restart required.

### Topic: Agent Skills {#t-skills}

**Agent Skills** are an Anthropic open standard for packaging *knowledge and workflows* an agent can read on demand. A skill is just a folder with a `SKILL.md` and optional scripts/references — version-controlled, portable, and adopted by Claude · Cursor · GitHub Copilot · VS Code · Codex · Gemini CLI · OpenHands · Goose · 25+ others. **Author once, run anywhere.**

#### Anatomy of a Skill

```
📁 my-skill/                   ← just a folder
├── 📄 SKILL.md                ← REQUIRED: YAML frontmatter + markdown
├── 📁 scripts/                ← optional: executables the agent runs
│   ├── lint.sh
│   └── coverage.py
├── 📁 references/             ← optional: docs the agent reads on demand
│   └── style-guide.md
└── 📁 assets/                 ← optional: templates the agent uses
    └── template.md
```

`SKILL.md` is the heart:

```markdown
---
name: pr-review
description: Reviews PRs for security, tests, and style.
---

# How to Review a PR

1. Fetch diff via `gh pr diff`
2. Run scripts/lint.sh
3. Check tests ≥ 80% coverage
4. Post inline comments
```

**YAML frontmatter** for discovery (loaded into every session as a tiny metadata blob). **Markdown body** for execution (loaded only when the skill is *activated*).

#### Progressive Disclosure — Three Stages

Skills load lazily. You can hand an agent **100 skills for under 5K tokens of metadata**; only the matching one expands, only the needed files load.

```
1 · DISCOVERY                  2 · ACTIVATION                 3 · EXECUTION
─────────────                  ─────────────                  ─────────────
at startup                     task matches a skill           agent runs the steps
loaded:                        loaded:                        loaded:
  name + description            + full SKILL.md (1 only)        + scripts on demand
  for ALL skills                  others stay metadata-only      + references on demand

  ≈ 50 tokens × N skills       full body of matched skill     only what the step needs
  tiny baseline footprint      ≈ a few hundred more tokens     no bloat · no waste
```

#### MCP vs. Skills — Complementary, Not Either-Or

A skill tells the agent **what** to do; MCP gives it the **hands** to actually do it. Don't pick — compose. Capability without competence is reckless. Competence without capability is just advice.

| Dimension | **MCP** — the hands | **Skills** — the brain |
|---|---|---|
| Layer | Transport / wire protocol | Knowledge / workflow package |
| Format | JSON-RPC over stdio / streamable HTTP | Folder of markdown + `SKILL.md` |
| Provides | Tools · resources · prompts | Instructions · scripts · references |
| Loaded | Connection live for whole session | Progressive disclosure (3 stages) |
| Lives | External process, local or remote | In the LLM context window |
| Authored by | Platform / SRE / vendor | Domain expert, team lead |
| Answers | **CAN** the agent reach the system? | **SHOULD** the agent do these steps? |

```
┌─────────────────────────┐  calls   ┌─────────────────────────┐
│   SKILL · pr-review     │ ───────▶ │      MCP SERVERS        │
│ 1. github.get_diff(id)  │          │ github · get_diff       │
│ 2. run scripts/lint.sh  │ ◀─────── │ filesystem · read       │
│ 3. github.post_review() │ returns  │ postgres · query        │
│   tells WHAT to do      │  data    │  gives HANDS to do it   │
└─────────────────────────┘          └─────────────────────────┘
```

:::lab Lab 2.1 — Sens-AI Habits + MCP Server
**Objectives:**
- Take a "stuck" coding scenario (agent in a rehash loop) and apply all 5 Sens-AI habits to break it. Document what each habit changed.
- Write a `CLAUDE.md` for a sample project with rules, 3 commands, and 1 skill that applies Critical Thinking.
- Build a custom MCP server: `run_linter`, `run_tests`, `get_coverage`.
- Connect it to Claude Code and verify it uses your tool.
- Reflect: which Sens-AI habit is hardest to consistently apply when working with agents?
:::
