---
id: s34
number: "3.4"
title: "Extended: Agent Performance Engineering"
time: "5:30–7:30 PM"
duration: "2 hours"
type: "Extended Session"
topics:
  - id: t-perf-problem
    title: "The Performance Tax"
  - id: t-prompt-cache
    title: "Prompt Caching"
  - id: t-semantic-cache
    title: "Semantic Caching"
  - id: t-model-routing
    title: "Right-Sized Models"
  - id: t-async-parallel
    title: "Async & Parallelism"
  - id: t-token-budget
    title: "Token Budget Management"
---

Your agent works. Now make it fast enough that users don't rage-quit, and cheap enough that your cloud bill doesn't trigger a board meeting. Performance engineering for agentic systems is not an afterthought — it is the difference between a demo and a product.

### Topic: The Performance Tax {#t-perf-problem}

Every agentic interaction is a pipeline of LLM calls. Each call costs tokens; each token costs money; every round-trip costs latency. The compound effect is brutal.

#### Why Agents Are Expensive By Default

A typical 10-step ReAct loop on a 5,000-token system prompt:

- **Each call** re-sends the full system prompt
- **10 calls** × 5,000 tokens = 50,000 input tokens per task
- At Claude Sonnet 4.6 pricing ($3/MTok input), that is **$0.15 per task**
- At 1,000 tasks/day → **$150/day** just in input tokens, before output

The system prompt alone — with tool definitions and instructions — is often the biggest cost driver, and it is identical across every single call.

:::warning The Compound Cost Trap
Multi-agent systems multiply this. A 5-agent pipeline where each agent has a 3,000-token system prompt and runs 5 steps internally costs **5 × 3,000 × 5 = 75,000 input tokens** before a single useful output token is generated. Design for cost from day one.
:::

#### The Four Performance Levers

- **Prompt Caching** — pay once to store your system prompt at the API layer; subsequent calls read from cache at 10% cost (Anthropic) or 50% cost (OpenAI)
- **Semantic Caching** — store LLM responses and serve identical/similar queries from cache without hitting the LLM at all
- **Model Routing** — use the smallest model that can do the job correctly
- **Async Parallelism** — run independent steps concurrently instead of sequentially

Applied together, these typically deliver **3–10× cost reduction** and **2–5× latency improvement** on real agentic workloads.

### Topic: Prompt Caching {#t-prompt-cache}

Prompt caching is a server-side feature where the LLM provider stores the KV (key-value) cache for your prompt prefix, so subsequent calls that share that prefix skip the prefill computation. You pay a reduced rate for cache reads.

#### How It Works

When you send a prompt:

```
[System: 10,000 tokens — CACHED]
[Conversation history: 2,000 tokens — not cached]
[New user message: 50 tokens — not cached]
```

The first call writes the cache (slightly more expensive). Every subsequent call with the same system prompt prefix reads from cache at a fraction of the normal cost.

#### Anthropic Prompt Caching

Anthropic's implementation requires you to explicitly mark cache breakpoints using `cache_control`. This gives you control over *exactly* what gets cached.

**Source:** https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

**Pricing (as of April 2026):**
- Cache write: 125% of standard input token price (slight overhead to populate cache)
- Cache read: 10% of standard input token price (**90% discount**)
- Cache TTL: 5 minutes (resets on every hit)
- Minimum cacheable block: 1,024 tokens (Sonnet/Haiku), 2,048 tokens (Opus)

```python
import anthropic
from pathlib import Path

client = anthropic.Anthropic()

# Load your stable system prompt — tool definitions, domain knowledge, rules.
# The longer this is, the bigger the savings.
SYSTEM_PROMPT = Path("agent_system_prompt.txt").read_text()  # e.g. 5,000+ tokens

def agent_call(user_message: str, conversation_history: list[dict] | None = None) -> str:
    """Call Claude with prompt caching on the system prompt."""
    messages = conversation_history or []
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Mark this block for caching.
                # Everything up to this breakpoint will be cached.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )

    # Inspect cache utilisation — log this in production (Langfuse/Langsmith)
    usage = response.usage
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read  = getattr(usage, "cache_read_input_tokens", 0)
    regular     = usage.input_tokens

    print(f"Tokens | write: {cache_write:>6} | read: {cache_read:>6} | regular: {regular:>6}")

    return response.content[0].text
```

#### Caching Multiple Breakpoints

You can cache up to **4 breakpoints** in a single request. This is useful for caching tool definitions separately from conversation history:

```python
import anthropic

client = anthropic.Anthropic()

# Large, stable tool definitions (often 2,000–4,000 tokens)
TOOL_DEFINITIONS = load_tool_definitions()  # returns list[dict]

# Domain knowledge (e.g. a runbook, code base summary)
DOMAIN_KNOWLEDGE = load_domain_knowledge()  # long string

def agent_call_multi_cache(user_message: str, recent_history: list[dict]) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": DOMAIN_KNOWLEDGE,
                "cache_control": {"type": "ephemeral"},  # Breakpoint 1
            }
        ],
        tools=TOOL_DEFINITIONS,
        # Pass tool_choice to enable tool use
        tool_choice={"type": "auto"},
        messages=[
            # Cache the first N turns of a long conversation
            *[
                {**msg, "cache_control": {"type": "ephemeral"}}
                if i == len(recent_history) - 4  # Breakpoint at turn N-4
                else msg
                for i, msg in enumerate(recent_history)
            ],
            {"role": "user", "content": user_message},
        ],
    )
    return response.content
```

#### OpenAI Prompt Caching

OpenAI's implementation is automatic — no explicit marking required. Inputs of 1,024+ tokens are eligible. Cache discount is 50% (vs Anthropic's 90%).

**Source:** https://platform.openai.com/docs/guides/prompt-caching

```python
from openai import OpenAI

client = OpenAI()

# OpenAI caches automatically — just structure your prompt with the
# stable prefix first and the dynamic content last.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": LONG_STABLE_SYSTEM_PROMPT},  # Cached automatically
        {"role": "user",   "content": dynamic_user_message},        # Dynamic — not cached
    ],
)

# Check cache usage in response
print(response.usage.prompt_tokens_details)
# PromptTokensDetails(cached_tokens=4096, audio_tokens=0)
```

:::tip Structuring Prompts for Maximum Cache Hits
Place the most stable, longest content **first** (system prompt, tool definitions, domain knowledge). Put dynamic content (user message, current date, session-specific context) **last**. The cache only applies to prefixes — if you put dynamic content first, you break the cache on every call.
:::

#### Savings Estimator

```python
def estimate_prompt_cache_savings(
    system_prompt_tokens: int,
    calls_per_day: int,
    input_price_per_mtok: float = 3.0,  # Claude Sonnet 4.6 price
) -> dict:
    """Calculate projected daily savings from enabling prompt caching."""
    # Without cache: every call pays full price for the system prompt
    cost_without_cache = (system_prompt_tokens / 1_000_000) * input_price_per_mtok * calls_per_day

    # With cache: first call is a write (125%), rest are reads (10%)
    cost_first_call = (system_prompt_tokens / 1_000_000) * input_price_per_mtok * 1.25
    cost_reads = (system_prompt_tokens / 1_000_000) * input_price_per_mtok * 0.10 * (calls_per_day - 1)
    cost_with_cache = cost_first_call + cost_reads

    return {
        "without_cache_usd": round(cost_without_cache, 4),
        "with_cache_usd":    round(cost_with_cache, 4),
        "savings_usd":       round(cost_without_cache - cost_with_cache, 4),
        "savings_pct":       round((1 - cost_with_cache / cost_without_cache) * 100, 1),
    }

# Example: 5,000-token system prompt, 500 calls/day
result = estimate_prompt_cache_savings(5_000, 500)
# → {'savings_pct': 86.3, 'savings_usd': 0.638, 'without_cache_usd': 0.750, ...}
```

:::info Cache TTL and Warm-up
The Anthropic cache TTL is **5 minutes**, reset on every hit. For agents with bursty traffic, this works well. For cron-style agents that run once per hour, the cache will be cold most of the time — calculate your actual hit rate before projecting savings.
:::

### Topic: Semantic Caching {#t-semantic-cache}

Prompt caching saves on repeated calls with the *same* system prompt. Semantic caching goes further: it stores LLM *responses* and returns cached answers for *semantically similar* queries — no LLM call needed at all.

#### LangChain Semantic Cache

LangChain's caching layer wraps any LLM. Set it once at startup and all subsequent calls route through the cache automatically.

**Source:** https://python.langchain.com/docs/how_to/llm_caching/

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemorySemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# One-time setup — embeddings model converts queries to vectors for similarity search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

set_llm_cache(
    InMemorySemanticCache(
        embedding=embeddings,
        # Similarity threshold: 0.95 = very conservative (near-exact match only)
        # 0.85 = more aggressive (similar but not identical queries hit cache)
        # Tune based on acceptable staleness for your use case
        score_threshold=0.95,
    )
)

# All LLM calls through this instance automatically use the cache
llm = ChatAnthropic(model="claude-sonnet-4-6")

# First call — cache miss, calls LLM and stores response
response1 = llm.invoke("What is the capital of France?")

# Second call — exact match, returns cached response instantly (0ms LLM latency)
response2 = llm.invoke("What is the capital of France?")

# Third call — semantic match at 0.97 similarity, returns cached response
response3 = llm.invoke("What's France's capital city?")
```

For production, replace `InMemorySemanticCache` with `RedisSemanticCache` (persists across restarts, shared across instances):

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Redis + vector similarity — survives restarts, sharable across multiple agent instances
set_llm_cache(
    RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=embeddings,
        score_threshold=0.95,
    )
)
```

#### GPTCache — Dedicated Caching Layer

For higher-volume deployments, GPTCache is a standalone caching library designed specifically for LLM workloads. It supports multiple similarity backends (FAISS, Milvus, Hnswlib) and adapters for LangChain and direct API clients.

**Source:** https://github.com/zilliztech/GPTCache

```python
from gptcache import cache
from gptcache.adapter import openai  # Drop-in OpenAI replacement
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Initialise GPTCache with FAISS backend
onnx = Onnx()
data_manager = get_data_manager(
    CacheBase("sqlite"),          # Metadata store
    VectorBase("faiss", dimension=onnx.dimension),  # Vector store
)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()  # Uses OPENAI_API_KEY env var

# Use gptcache's openai adapter — same API as openai.OpenAI()
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

#### When Semantic Caching Helps vs Hurts

- **Helps:** FAQ-style queries, repeated lookups in the same domain, classification tasks, extraction from stable templates
- **Hurts / avoid:** Real-time data queries ("what is the current BTC price?"), queries that depend on user context or session state, creative generation where diversity is the point
- **Threshold tuning:** Start at 0.95 (conservative). Monitor cache hit rate and false positives in production before lowering.

:::warning Semantic Cache Staleness
A cached response from 3 months ago may be wrong today. Tag cached entries with a creation timestamp and TTL. Invalidate the cache when your underlying data changes. For agent tools (search, database queries), prefer prompt caching or model routing over semantic caching — those tools exist specifically because the answer can change.
:::

### Topic: Right-Sized Models {#t-model-routing}

The most impactful single performance decision is **which model you use for each task**. Claude Haiku 4.5 is ~20× cheaper per token than Claude Opus 4.6 and ~3× faster. For tasks that don't require deep reasoning, using Opus is like hiring a senior architect to rename a variable.

#### Model Selection Matrix

Use this framework to pick the right model for each operation in your agent:

- **Haiku 4.5** — Classification, entity extraction, simple Q&A, routing decisions, yes/no guards, formatting, JSON extraction from structured text. Anything where the "thinking" is minimal.
- **Sonnet 4.6** — Code generation, summarisation, multi-step reasoning, tool use orchestration, RAG synthesis. The default for most agentic tasks.
- **Opus 4.6** — Complex multi-step planning, cross-document reasoning, research synthesis, architect-level decisions, tasks requiring nuanced judgment. Use sparingly.

#### Dynamic Model Routing in LangGraph

```python
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Pre-initialise model clients at startup — avoids per-call overhead
MODELS = {
    "fast":     ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512),
    "balanced": ChatAnthropic(model="claude-sonnet-4-6",         max_tokens=2048),
    "powerful": ChatAnthropic(model="claude-opus-4-6",           max_tokens=4096),
}

# Use Haiku to classify task complexity — cheap, fast, accurate enough
ROUTER = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=10)

class AgentState(BaseModel):
    task: str
    tier: str = ""
    result: str = ""

def classify_task(state: AgentState) -> AgentState:
    """Route to the correct model tier based on task complexity."""
    prompt = (
        "Classify the complexity of this task as exactly one word: simple, medium, or complex.\n"
        f"Task: {state.task}\n"
        "Reply with only the word."
    )
    tier = ROUTER.invoke(prompt).content.strip().lower()
    # Sanitise — only accept known tiers
    if tier not in ("simple", "medium", "complex"):
        tier = "medium"  # Safe default

    tier_map = {"simple": "fast", "medium": "balanced", "complex": "powerful"}
    return state.model_copy(update={"tier": tier_map[tier]})

def execute_task(state: AgentState) -> AgentState:
    """Execute the task using the routed model."""
    llm = MODELS[state.tier]
    result = llm.invoke(state.task).content
    return state.model_copy(update={"result": result})

# Build the routing graph
graph = StateGraph(AgentState)
graph.add_node("classify", classify_task)
graph.add_node("execute", execute_task)
graph.set_entry_point("classify")
graph.add_edge("classify", "execute")
graph.add_edge("execute", END)

agent = graph.compile()

# Example usage
result = agent.invoke(AgentState(task="What is 2 + 2?"))
print(f"Model tier used: {result['tier']}")   # → fast (Haiku)
print(f"Result: {result['result']}")
```

#### RouteLLM — Learned Routing

For high-volume deployments, Stanford's RouteLLM trains a small classifier that learns *from your data* when strong vs weak models are needed. It consistently achieves 2–4× cost reduction with <5% quality loss on real benchmarks.

**Source:** https://github.com/lm-sys/RouteLLM

```python
# pip install routellm
from routellm.controller import Controller

# Controller wraps two models: strong (expensive) and weak (cheap)
controller = Controller(
    routers=["mf"],  # Matrix Factorisation router
    strong_model="claude-opus-4-6",
    weak_model="claude-haiku-4-5-20251001",
)

# threshold=0.11856 means "use strong model only when confidence is high"
# This was calibrated by RouteLLM to cut GPT-4 calls by 40% with minimal quality loss
response = controller.chat.completions.create(
    model="router-mf-0.11856",
    messages=[{"role": "user", "content": user_query}],
)
```

:::tip Cost vs Quality Trade-off
There is no universal answer to which model to use. Run an offline eval on your specific task distribution. Measure quality at each tier. Accept quality loss only where you can quantify that it does not affect user outcomes. "Haiku for classification, Sonnet for generation, Opus for planning" is a sensible starting default, not a law.
:::

### Topic: Async & Parallelism {#t-async-parallel}

Sequential tool calls compound latency. If your agent reads 5 documents before writing a summary, those reads can run in parallel. Python's `asyncio` and LangGraph's native async support make this straightforward.

#### Parallel LLM Calls with asyncio

```python
import asyncio
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

# BAD: sequential — total latency = sum of all call latencies
async def summarise_sequential(documents: list[str]) -> list[str]:
    results = []
    for doc in documents:
        result = await llm.ainvoke(f"Summarise in one sentence: {doc}")
        results.append(result.content)
    return results  # Latency: N × avg_call_latency

# GOOD: parallel — total latency ≈ max single call latency
async def summarise_parallel(documents: list[str]) -> list[str]:
    tasks = [
        llm.ainvoke(f"Summarise in one sentence: {doc}")
        for doc in documents
    ]
    results = await asyncio.gather(*tasks)
    return [r.content for r in results]  # Latency: ~1× avg_call_latency

# For 5 documents averaging 1.5s each:
# Sequential: ~7.5s  |  Parallel: ~1.7s  (includes overhead)
```

#### Parallel Nodes in LangGraph

LangGraph's Send API dispatches work to multiple graph nodes in parallel and fan-in the results:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel

class ResearchState(BaseModel):
    query: str
    sources: list[str] = []
    summaries: list[str] = []
    final_report: str = ""

async def fetch_sources(state: ResearchState) -> dict:
    """Fan-out: dispatch one summarisation task per source in parallel."""
    return [
        Send("summarise_source", {"source": src, "query": state.query})
        for src in state.sources
    ]

async def summarise_source(state: dict) -> dict:
    """Worker: summarise one source. Runs in parallel with other workers."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001")  # Cheap for extraction
    summary = await llm.ainvoke(
        f"Extract facts relevant to '{state['query']}' from: {state['source']}"
    )
    return {"summaries": [summary.content]}

async def write_report(state: ResearchState) -> dict:
    """Fan-in: all summaries are available, write the final report."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")  # Better model for synthesis
    combined = "\n---\n".join(state.summaries)
    report = await llm.ainvoke(f"Write a report for '{state.query}' using: {combined}")
    return {"final_report": report.content}

# Build graph with parallel fan-out
graph = StateGraph(ResearchState)
graph.add_node("fetch_sources", fetch_sources)
graph.add_node("summarise_source", summarise_source)
graph.add_node("write_report", write_report)

graph.add_conditional_edges("fetch_sources", lambda s: s, ["summarise_source"])
graph.add_edge("summarise_source", "write_report")
graph.set_entry_point("fetch_sources")
graph.add_edge("write_report", END)
```

#### Concurrency Limits

Unbounded parallelism will hit rate limits or overwhelm downstream services. Use `asyncio.Semaphore` to cap concurrent calls:

```python
import asyncio
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")
# Cap at 5 concurrent LLM calls to stay within rate limits
SEMAPHORE = asyncio.Semaphore(5)

async def rate_limited_call(prompt: str) -> str:
    async with SEMAPHORE:
        result = await llm.ainvoke(prompt)
        return result.content

async def process_batch(prompts: list[str]) -> list[str]:
    tasks = [rate_limited_call(p) for p in prompts]
    return await asyncio.gather(*tasks)
```

### Topic: Token Budget Management {#t-token-budget}

Every token you send is a token you pay for. System prompts bloat over time as developers append edge cases. Context windows fill with stale history. Output tokens are often unbounded when they should not be.

#### Audit Your System Prompt

```python
import anthropic

client = anthropic.Anthropic()

def audit_prompt_token_cost(system_prompt: str, model: str = "claude-sonnet-4-6") -> dict:
    """Count tokens in your system prompt before committing to it."""
    response = client.messages.count_tokens(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": "ping"}],
    )
    price_per_mtok = 3.0  # Claude Sonnet 4.6 — check docs for current pricing
    tokens = response.input_tokens
    daily_cost_at_1k_calls = (tokens / 1_000_000) * price_per_mtok * 1_000

    return {
        "tokens": tokens,
        "estimated_daily_cost_usd_at_1k_calls": round(daily_cost_at_1k_calls, 4),
        "tip": "Enable prompt caching if tokens > 1024 and calls > 100/day",
    }
```

#### Cap Response Length Per Operation

Not every step in your agent needs a 4,096-token response. A routing decision needs 10 tokens. A yes/no guard needs 5. An extraction step needs however many tokens the extracted JSON is — rarely more than 512.

```python
from anthropic import Anthropic

client = Anthropic()

# Routing decision: tiny budget
def classify_intent(user_message: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,  # 20 tokens is plenty for a single-word classification
        messages=[
            {
                "role": "user",
                "content": f"Classify as: search, calculate, create, or other.\nInput: {user_message}\nReply with ONE word."
            }
        ],
    )
    return response.content[0].text.strip().lower()

# Data extraction: bounded by structure
def extract_entities(text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,  # JSON with 5-10 fields won't exceed this
        messages=[
            {
                "role": "user",
                "content": f"Extract name, date, and amount as JSON. Text: {text}"
            }
        ],
    )
    import json
    return json.loads(response.content[0].text)

# Final user-facing answer: generous budget
def answer_question(question: str, context: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,  # Enough for a thorough answer, not an essay
        messages=[
            {
                "role": "user",
                "content": f"Answer concisely using only the context provided.\nContext: {context}\nQuestion: {question}"
            }
        ],
    )
    return response.content[0].text
```

#### Context Window Hygiene

```python
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

def trim_conversation_history(
    messages: list[BaseMessage],
    max_tokens: int = 8_000,
    tokenizer_fn=None,
) -> list[BaseMessage]:
    """
    Keep the system message + most recent N exchanges that fit within max_tokens.
    Drops oldest turns first. Never drops the system message.
    """
    if not messages:
        return messages

    # Always keep system message
    system = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    # Rough token estimate: 4 chars ≈ 1 token (good enough for trimming heuristic)
    def estimate_tokens(m: BaseMessage) -> int:
        return len(m.content) // 4

    total = sum(estimate_tokens(m) for m in system)
    kept = []

    # Walk backwards — keep most recent messages
    for msg in reversed(non_system):
        cost = estimate_tokens(msg)
        if total + cost > max_tokens:
            break
        kept.insert(0, msg)
        total += cost

    return system + kept
```

:::tip The Three Rules of Token Hygiene
**1. Measure before you optimise.** Use `client.messages.count_tokens()` (Anthropic) or tiktoken (OpenAI) to know your actual token counts — not guesses.
**2. Set `max_tokens` on every call.** Leaving it at default (`4096`) on a routing step is like ordering a full dinner when you just want the check.
**3. Compress before you truncate.** Summarise old conversation turns into a single "conversation summary" message rather than dropping them entirely. Dropped context = hallucinated context.
:::

:::lab Lab 3.4 — Performance Audit Sprint
**Objectives:**
- Instrument your capstone agent with token usage logging (read + write + cache tokens per call).
- Enable Anthropic prompt caching on the system prompt and measure the savings using the estimator function.
- Add a model routing layer: classify tasks as simple/medium/complex and route to Haiku/Sonnet/Opus accordingly.
- Wrap at least two parallel tool calls with `asyncio.gather` instead of sequential await.
- Run the agent 10 times and record: total tokens per run, cache hit rate, p95 latency before and after.
- Target: ≥40% reduction in input token cost, ≥30% reduction in p95 latency.
:::

---

*Performance engineering is not about making your agent faster at the wrong thing. Measure first. Optimise the bottleneck. Ship it. Measure again. The cycle never ends — but at least now your cloud bill won't require a separate line item in the quarterly report.*
