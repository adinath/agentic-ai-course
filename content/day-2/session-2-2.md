---
id: s22
number: "2.2"
title: "Context Management & Window Engineering"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-ctxfund
    title: "Window Fundamentals"
  - id: t-compress
    title: "Compression"
  - id: t-dynctx
    title: "Dynamic Construction"
---

### Topic: Context Window Fundamentals {#t-ctxfund}

The context window is the agent's working memory. Everything the model knows about the current task must fit inside it. Modern models have large windows (200K tokens for Claude, 128K for GPT-4o), but agents burn through them faster than you expect — tool results accumulate, long documents get injected, and each retrieval round adds more text. Without a deliberate token budget strategy, you will hit limits at the worst possible moment.

:::example Token budget for a 100K context agent
```
   4,000 tokens → System prompt (persona, rules, tool docs)
  10,000 tokens → Retrieved memory (5 chunks × 2,000 tokens)
  30,000 tokens → Conversation history (last 10 turns)
  20,000 tokens → Tool results (last 5 calls × 4,000 tokens)
  20,000 tokens → Current document being processed
  16,000 tokens → Reserved for model output
  ─────────────────────────────────
 100,000 tokens TOTAL
```
:::

- **Lost-in-the-middle:** LLMs reliably recall information at the beginning and end of context, but underperform on information in the middle. Put critical instructions at the top.
- **Context poisoning:** Stale or incorrect tool results accumulate and mislead the model. Tag errors clearly and strip irrelevant history.

#### Model Context Limits & Trade-offs

Not all context windows are created equal. Larger windows cost more per token and add latency — but smaller windows force more aggressive context management. Choose based on your task profile:

```python
# Model context window reference (as of mid-2025)
MODEL_LIMITS = {
    "claude-opus-4-5":   {"context": 200_000, "output": 32_000},
    "claude-sonnet-4-5": {"context": 200_000, "output": 16_000},
    "claude-haiku-4-5":  {"context": 200_000, "output": 8_000},
    "gpt-4o":            {"context": 128_000, "output": 16_000},
    "gemini-1.5-pro":    {"context": 2_000_000, "output": 8_000},
}

# Compute usable budget (reserve 20% for output)
def input_budget(model: str) -> int:
    limits = MODEL_LIMITS[model]
    return limits["context"] - limits["output"]

# Practical guidance:
# Use claude-haiku for fast tool classification (small context needed)
# Use claude-sonnet for multi-file reasoning (needs 50-100K)
# Use claude-opus for complex architectural decisions (full window)
```

### Topic: Compression & Summarization {#t-compress}

#### Rolling Window with Summaries

```python
MAX_RECENT = 10   # keep last N messages verbatim

def compress_history(messages, existing_summary=""):
    if len(messages) <= MAX_RECENT:
        return messages, existing_summary

    old    = messages[:-MAX_RECENT]
    recent = messages[-MAX_RECENT:]
    old_text = "\n".join(f"{m.type}: {m.content[:300]}" for m in old)

    new_summary = summariser.invoke(
        f"Existing summary: {existing_summary}\n\n"
        f"New messages:\n{old_text}\n\nUpdate summary. Max 200 words."
    ).content

    summary_msg = SystemMessage(content=f"[CONVERSATION SUMMARY]\n{new_summary}")
    return [summary_msg] + recent, new_summary
```

### Topic: Dynamic Context Construction {#t-dynctx}

Rather than appending content to context naively, build the context programmatically each turn using a template that assigns a token budget to each slot. When any slot overflows, lower-priority slots are trimmed first — ensuring the most critical content (system prompt, recent messages) is never lost.

#### Priority-Weighted Context Builder

```python
from dataclasses import dataclass

@dataclass
class ContextSlot:
    name: str
    content: str
    priority: int    # higher = keep longer under budget pressure
    max_tokens: int

def build_context(slots: list[ContextSlot], total_budget: int) -> str:
    """Build context string respecting token budget using priority order."""
    slots_sorted = sorted(slots, key=lambda s: s.priority, reverse=True)
    used, parts = 0, []
    for slot in slots_sorted:
        available = min(slot.max_tokens, (total_budget - used) * 4)
        if available <= 0:
            break
        trimmed = slot.content[:available]
        parts.append(f"[{slot.name.upper()}]\n{trimmed}")
        used += len(trimmed) // 4
    return "\n\n".join(parts)

# Example: assemble context before each agent step
context = build_context([
    ContextSlot("system",  SYSTEM_PROMPT,    priority=10, max_tokens=2000),
    ContextSlot("memory",  retrieved_memory, priority=8,  max_tokens=4000),
    ContextSlot("history", conversation,     priority=6,  max_tokens=12000),
    ContextSlot("file",    current_file,     priority=4,  max_tokens=8000),
], total_budget=24000)
```

#### Prompt Caching with Anthropic API

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": LONG_SYSTEM_PROMPT,   # e.g. 10K tokens of docs
        "cache_control": {"type": "ephemeral"},  # mark for caching
    }],
    messages=[{"role": "user", "content": user_message}],
)

# Cached calls cost ~10% of normal input token price
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

:::lab Lab 2.2 — Context Manager Class
**Objectives:**
- Implement a `ContextManager` class that builds agent prompts under a configurable token budget.
- Slots: system_prompt, retrieved_memories, tool_docs, recent_history, current_file.
- Support priority-based trimming: system_prompt is never trimmed; recent_history is last resort.
- Benchmark: 10 coding tasks — measure task-success-rate vs. tokens-consumed.
:::
