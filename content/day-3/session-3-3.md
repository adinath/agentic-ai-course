---
id: s33
number: "3.3"
title: "Capstone Project"
time: "3:00–5:00 PM"
duration: "2 hours"
topics:
  - id: t-capstone
    title: "Capstone Workshop"
  - id: t-arch
    title: "Architecture Design"
  - id: t-present
    title: "Peer Review"
---

You have three days of theory, tools, and code behind you. Now you use it. The capstone is not a homework assignment — it is the blueprint for a real agent system you could build and ship.

### Topic: Capstone Project Overview {#t-capstone}

Teams of 2–3 design and begin implementing a complete multi-agent system. The deliverables are concrete, the constraints are real, and the peer review is honest.

#### Capstone Problem Prompts

Choose one. If none fits your use case, propose an equivalent with instructor approval.

---

**Option A — Autonomous PR Reviewer**

An agent system that monitors a GitHub repo, automatically reviews every new PR, and posts a structured review comment. Large diffs trigger the full Supervisor-Worker topology (security + performance + style agents). Small diffs route to a single General reviewer. Blocking reviews require human approval before posting.

```
User Story: As a developer, I open a PR and receive a thorough automated review
within 60 seconds — without bothering my teammates for routine feedback.
```

**Acceptance criteria:**
- Correct topology selection based on diff size
- Structured review comment with severity-tagged findings
- Human approval gate for blocking reviews
- Langfuse traces for every run
- Eval suite: 10 test PRs (5 small, 5 large) with minimum 80% accuracy

---

**Option B — Intelligent Incident Responder**

A multi-agent system that monitors alerts (PagerDuty / simulated), triages them using runbooks, attempts automated remediation, and escalates to a human only if automation fails. Uses a knowledge graph of known incidents and their resolutions as semantic memory.

```
User Story: As an on-call engineer, I receive 70% fewer pages because the agent
handles routine incidents automatically — and pages me only for genuine unknowns.
```

**Acceptance criteria:**
- Correct triage classification (sev1/sev2/sev3) with ≥85% accuracy
- Successful automated remediation for at least 3 incident types
- Episodic memory: learns from each resolved incident
- Full Langfuse traces per incident
- Guardrails: no irreversible actions without human approval

---

**Option C — Research Synthesiser**

A pipeline-topology agent that takes a technical question, searches web and internal documentation, extracts relevant information with citations, and produces a structured technical report. Self-curating memory: stores high-value sources for future reuse.

```
User Story: As a developer, I ask a complex technical question and receive a cited,
accurate report in under 2 minutes — drawn from both the web and our internal docs.
```

**Acceptance criteria:**
- Correct retrieval from both web (Tavily) and internal docs (ChromaDB)
- Cited sources for every factual claim
- Ragas evaluation: faithfulness ≥ 0.85
- Self-curating memory: sources indexed after first retrieval
- Report quality scored by LLM judge ≥ 7/10

---

### Topic: Architecture Design Template {#t-arch}

Complete the following for your chosen problem. This is your design document — be specific.

#### 1. Agent Topology

Draw your agent graph (use ASCII art or a description). Identify:
- Which topology pattern(s) from Session 3.1?
- How many agents? What role does each play?
- What triggers routing decisions?

```
Example (Option A):

[GitHub Webhook] ──▶ [Router Agent]
                           ↓
              diff_size < 300 lines?
               ↙ yes              ↘ no
   [General Reviewer]    [Supervisor]
                         ↙   ↓    ↘
               [Security] [Perf] [Style]
                         ↘   ↓   ↙
                       [Synthesiser]
                            ↓
              review_verdict == "blocking"?
               ↙ yes              ↘ no
       [Human Approval]    [Post Comment]
```

#### 2. Memory Design

For each memory type you use, specify:

| Type | Storage | Written When | Retrieved When | Retention |
|------|---------|--------------|----------------|-----------|
| Session | LangGraph state | Every step | Every step | 1 session |
| Episodic | ChromaDB | After each run | Before each run | Permanent |
| Semantic | ChromaDB | At setup + curated | Per query | Permanent |
| Procedural | CLAUDE.md | At setup | System prompt | Permanent |

#### 3. Tool Inventory

List every tool with its safety classification:

| Tool | Category | Safety Class | Max Impact |
|------|----------|--------------|------------|
| read_file | filesystem | read-only | none |
| write_file | filesystem | reversible-write | local file |
| post_comment | github | external-action | visible to team |
| deploy | infrastructure | destructive | production |

#### 4. Context Strategy

- Total token budget: ___K
- What goes in each slot? Priority order?
- How do you handle context overflow (compression, summarisation, truncation)?
- Do you use prompt caching?

#### 5. Evaluation Plan

Write 10 test cases you would run before shipping. For each:
- Task description
- Expected tool trajectory (which tools, in what order)
- Minimum completion score
- Forbidden actions

```python
# Template for your eval case
eval_case = {
    "id": "your-case-id",
    "task": "Describe the task the agent receives",
    "expected_tool_calls": [
        {"name": "tool_name", "args": {"key": "value"}},
    ],
    "min_completion_score": 0.8,
    "forbidden_tools": ["delete_file", "drop_table"],
    "max_iterations": 10,
}
```

#### 6. Safety Layer

| Failure Mode | Likelihood | Mitigation |
|--------------|------------|------------|
| Agent posts incorrect review | Medium | Human approval for blocking reviews |
| Prompt injection via PR content | Low | Input sanitisation + content wrapping |
| Infinite loop on complex diff | Medium | MAX_ITERATIONS = 15 hard limit |

### Topic: Implementation Sprint & Peer Review {#t-present}

**Sprint (45 minutes)**

Implement the skeleton of your system:
1. ✅ LangGraph graph with all nodes defined (even if some are stubs)
2. ✅ At least 3 tools implemented and tested
3. ✅ CLAUDE.md with rules, commands, and skills
4. ✅ One eval case that actually runs end-to-end

**Peer Review (15 minutes per team)**

Present to the room. Peer reviewers score on four dimensions:

```python
class CapstoneReview(BaseModel):
    completeness: int = Field(ge=0, le=5, description="All required sections addressed?")
    safety_coverage: int = Field(ge=0, le=5, description="Failure modes identified and mitigated?")
    eval_quality: int = Field(ge=0, le=5, description="Test cases specific and measurable?")
    architectural_fit: int = Field(ge=0, le=5, description="Topology matches the problem shape?")
    biggest_risk: str = Field(description="What is the most likely way this system fails in production?")
    strongest_aspect: str = Field(description="What is done particularly well?")
```

**Feedback rules:**
- No vague praise ("looks good!") — be specific
- Every critique must include a concrete alternative
- Identify the single biggest production risk you see

#### Post-Course: What to Build Next

You now have the full toolbox. Here is a practical next-step ladder:

```
Week 1: Ship a single-agent tool for your team
        (code reviewer, doc generator, test writer)
        → Apply Day 1 + Day 2.3 (observability from day one)

Week 2-3: Add memory to your agent
        → Apply Day 2.2 (episodic + semantic)
        → Measure: does memory improve task completion rate?

Month 1: Scale to multi-agent
        → Apply Day 3.1 topology patterns
        → Only if single agent hits a genuine ceiling

Month 2: Evaluate and harden
        → Apply Day 3.2 eval harness
        → Target: 80% pass rate before sharing with users
```

:::lab Lab 3.3 — Capstone Sprint
**Objectives:**
- Complete the Architecture Design Template for your chosen problem.
- Implement the skeleton: graph, 3+ tools, CLAUDE.md, 1 working eval case.
- Present for peer review using the `CapstoneReview` rubric.
- Each team must identify and address the "biggest risk" raised in peer feedback before the session ends.
:::

---

*Congratulations — you just went from "what is an agent" to "how do I trust this in production". The agents are watching. Make them behave.*
