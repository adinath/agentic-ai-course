---
id: s22
number: "2.2"
title: "Context Engineering & Memory"
time: "12:00–2:00 PM"
duration: "2 hours"
topics:
  - id: t-ctxeng
    title: "Context Engineering"
  - id: t-memory
    title: "Memory Management"
---

Context is the agent's entire world. Memory is how it survives across sessions. Get these wrong and you have an expensive goldfish. Get them right and you have a collaborator that genuinely improves with use.

### Topic: Context Engineering {#t-ctxeng}

#### What is Context Engineering?

Context engineering is the discipline of deliberately constructing what goes into the model's context window at each step — not just appending everything and hoping for the best.

The context window is finite. A 200K-token window sounds generous until you account for system prompt, tools, retrieved memory, conversation history, and the current document. You will fill it faster than you expect.

```
Token budget for a typical 100K context agent:
─────────────────────────────────────────────
   4,000 tokens → System prompt + rules
  10,000 tokens → Retrieved memory (5 chunks × 2K)
  30,000 tokens → Conversation history (last 10 turns)
  20,000 tokens → Tool results (last 5 calls × 4K)
  20,000 tokens → Current document / code file
  16,000 tokens → Reserved for model output
─────────────────────────────────────────────
 100,000 tokens TOTAL
```

**Key context engineering principles:**

- **Lost-in-the-middle:** LLMs reliably recall information at the beginning and end of context, but miss the middle. Put critical instructions at the top.
- **Context poisoning:** Stale or incorrect tool results accumulate and mislead. Tag errors clearly, strip irrelevant history.
- **Priority ordering:** Not all content is equally important. Design a hierarchy.

#### Priority-Weighted Context Builder

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContextSlot:
    name: str
    content: str
    priority: int     # higher = kept first under budget pressure
    max_tokens: int
    required: bool = False  # required slots are never trimmed

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4

def build_context(slots: list[ContextSlot], total_budget: int) -> str:
    """Build context string respecting token budget, trimming by priority."""
    # Required slots always included — calculate their cost first
    required = [s for s in slots if s.required]
    optional = sorted([s for s in slots if not s.required],
                      key=lambda s: s.priority, reverse=True)

    used = sum(min(estimate_tokens(s.content), s.max_tokens) for s in required)
    parts = [f"[{s.name.upper()}]\n{s.content[:s.max_tokens * 4]}" for s in required]

    for slot in optional:
        available_chars = min(
            slot.max_tokens * 4,
            (total_budget - used) * 4,
        )
        if available_chars <= 0:
            continue
        trimmed = slot.content[:available_chars]
        parts.append(f"[{slot.name.upper()}]\n{trimmed}")
        used += estimate_tokens(trimmed)

    return "\n\n".join(parts)


# Usage: assemble context before each agent step
def build_agent_context(
    system_prompt: str,
    retrieved_memory: str,
    conversation_history: str,
    current_file: str,
    tool_docs: str,
) -> str:
    return build_context([
        ContextSlot("system",   system_prompt,       priority=10, max_tokens=2_000, required=True),
        ContextSlot("tools",    tool_docs,            priority=9,  max_tokens=1_500, required=True),
        ContextSlot("memory",   retrieved_memory,     priority=8,  max_tokens=4_000),
        ContextSlot("history",  conversation_history, priority=6,  max_tokens=12_000),
        ContextSlot("file",     current_file,         priority=4,  max_tokens=8_000),
    ], total_budget=24_000)
```

#### Rolling Window Compression

When conversation history grows beyond budget, compress old messages into a summary. Keep recent messages verbatim for precision; summarise the rest for efficiency.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AnyMessage

summariser = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
MAX_RECENT_MESSAGES = 10

def compress_history(
    messages: list[AnyMessage],
    existing_summary: str = "",
) -> tuple[list[AnyMessage], str]:
    """Compress old messages into a rolling summary, keep recent ones verbatim."""
    if len(messages) <= MAX_RECENT_MESSAGES:
        return messages, existing_summary

    old_messages = messages[:-MAX_RECENT_MESSAGES]
    recent_messages = messages[-MAX_RECENT_MESSAGES:]

    old_text = "\n".join(
        f"{m.type}: {m.content[:400]}" for m in old_messages
    )

    new_summary = summariser.invoke(
        f"Existing summary: {existing_summary}\n\n"
        f"New messages to add:\n{old_text}\n\n"
        "Update the summary to include these messages. Max 250 words. "
        "Focus on: decisions made, tools used, key findings, unresolved issues."
    ).content

    summary_msg = SystemMessage(
        content=f"[CONVERSATION SUMMARY — {len(old_messages)} compressed messages]\n{new_summary}"
    )
    return [summary_msg] + recent_messages, new_summary


# Prompt caching — reduces cost 90% for repeated system prompts
import anthropic

def create_with_cache(system_prompt: str, user_message: str) -> str:
    """Use Anthropic prompt caching to cut costs on repeated system prompts."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},  # cache this prefix
        }],
        messages=[{"role": "user", "content": user_message}],
    )
    # Cached input tokens cost ~10% of normal price
    print(f"Cache hits: {response.usage.cache_read_input_tokens} tokens")
    return response.content[0].text
```

### Topic: Memory Management {#t-memory}

Memory is not one thing — it is four different systems, each with a different purpose, storage backend, and retrieval pattern.

#### The Four Memory Types

| Type | Human analogy | Storage | Lifetime | Loaded |
|---|---|---|---|---|
| **Session (Working)** | Short-term scratchpad | LangGraph `AgentState` | One conversation | Always |
| **Semantic** | "Python is a language" — facts you just *know* | Vector DB (Chroma/Qdrant/Pinecone) | Persistent | Top-K per query |
| **Episodic** | "Last time I debugged this, drawing helped" | Vector DB of episode records | Persistent | Top-K few-shots |
| **Procedural** | "I know how to ride a bicycle" — muscle memory | System prompt / `CLAUDE.md` | Persistent | At startup, every turn |

Each memory type has the same three-step lifecycle: **WRITE → STORE → USE.** What changes is *when* it's written, *what* schema it uses, and *how* it gets back into context.

#### Semantic Memory — Facts & Knowledge

> **Human analogy:** "Python is a programming language." You don't remember *where* you learned it — you just know it. Semantic memory holds durable, decontextualised facts the agent leans on without re-telling history.

| Stage | What happens |
|---|---|
| **01 · WRITE** | A `memory_manager` reads the chat and pulls out facts, preferences, entities & relations. Hot-path (instant) or background (after-the-fact reflection). |
| **02 · STORE** | Two patterns: **Collection** — many docs, vector-indexed, deduplicate over time. **Profile** — one schema-bound doc; new info overwrites the latest state. |
| **03 · USE** | Similarity search ranked by relevance · recency · importance. Top-K facts injected into the system prompt before the LLM reasons. |

```
"I work at Acme on the ML team — mostly NLP."
     │ extract  ⟶  memory_manager
     ▼
Memory(content="User: ML engineer at Acme, NLP focus", importance=HIGH)
     │ persist  ⟶  namespace: ("acme", "{user_id}", "facts")
     ▼
Next session  ⟶  similarity_search(query)  ⟶  top-K  ⟶  system prompt
```

#### Episodic Memory — Past Experiences

> **Human analogy:** "Last time I debugged a recursion bug, drawing a tree on paper unstuck me." You replay *what worked before* — situation, reasoning, action, outcome. Experience-shaped few-shots for the agent.

| Stage | What happens |
|---|---|
| **01 · WRITE** | After a successful run. Background reflection. Capture only runs the agent (or a critic) judges worth remembering. Failed runs stay out unless tagged. |
| **02 · STORE** | **Episode schema:** `observation` (the situation) · `thoughts` (reasoning) · `action` (what was done) · `result` (why it worked). |
| **03 · USE** | On a new task, similarity-match the situation; inject top-K episodes as in-context few-shots. The agent imitates the moves that worked last time. |

```python
Episode(
    observation = "User stuck on recursion in binary tree traversal",
    thoughts    = "Use a treehouse-village metaphor — concrete first",
    action      = "Reframed problem; outlined 1.left 2.right 3.add-self",
    result      = "User clicked. Worked because it tied logic to a picture.",
)
# stored in episodes collection · retrieved next time recursion comes up
```

#### Procedural Memory — How-To & System Behaviour

> **Human analogy:** "I know how to ride a bicycle." Muscle memory. You don't recite the rules — you just *do* the right thing. For agents this is the system prompt and rules: identity, style, workflow, safety policy.

| Stage | What happens |
|---|---|
| **01 · WRITE** | Seed with a hand-written system prompt. A *prompt optimiser* (`create_prompt_optimizer`) reflects on trajectories & user scores, proposes a sharpened prompt — meta-prompt or gradient-style. |
| **02 · STORE** | One canonical system prompt (versioned, reviewable) *or* a collection of rule snippets indexed by situation — pulled in like skills when relevant. |
| **03 · USE** | Always-on context. Sits at the top of every prompt — never trimmed. Refined through feedback loops, not per-turn search. |

```
v1: "You are a helpful assistant."
        │
        │ feedback: user_score = 0  ("show practical example, not theory")
        ▼  optimizer.invoke({trajectories, prompt})
v2: "You are a helpful assistant. For programming questions:
       1. Open with a concrete code example
       2. Defer theory unless asked
       3. Adapt fast when the user redirects."
```

#### Session Memory (Working Memory)

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, add_messages

class AgentState(TypedDict):
    # Session memory — lives for one conversation thread
    messages: Annotated[list[AnyMessage], add_messages]
    current_task: Optional[str]
    iteration_count: int
    intermediate_results: list[str]

# MemorySaver keeps state in-process — fine for single sessions
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# thread_id is the session key — same ID = same conversation
config = {"configurable": {"thread_id": "user-session-abc123"}}
result = app.invoke({"messages": [HumanMessage("Fix the login bug")]}, config)

# Resuming the same session later — full history is restored
result2 = app.invoke({"messages": [HumanMessage("Now write tests for it")]}, config)
```

#### Episodic Memory

```python
import json
from pathlib import Path
from datetime import datetime
from langchain_chroma import Chroma
from langchain_anthropic import AnthropicEmbeddings
from langchain_core.documents import Document

embeddings = AnthropicEmbeddings(model="voyage-3")
episode_store = Chroma(
    collection_name="agent_episodes",
    embedding_function=embeddings,
    persist_directory="./memory/episodes",
)

summariser = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

def save_episode(task: str, messages: list, outcome: str):
    """Compress a completed agent session into episodic memory."""
    convo = "\n".join(f"{m.type}: {str(m.content)[:400]}" for m in messages)
    summary = summariser.invoke(
        f"Summarise this agent session in 5 bullet points.\n"
        f"Focus on: what was attempted, what worked, key learnings, gotchas.\n\n"
        f"Task: {task}\nOutcome: {outcome}\nSession:\n{convo}"
    ).content

    doc = Document(
        page_content=f"Task: {task}\nOutcome: {outcome}\n\n{summary}",
        metadata={
            "task": task,
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "episode",
        },
    )
    episode_store.add_documents([doc])

def recall_episodes(current_task: str, k: int = 3) -> list[str]:
    """Retrieve past episodes relevant to the current task."""
    docs = episode_store.similarity_search(current_task, k=k)
    return [d.page_content for d in docs]

# Usage: inject episodic memories at session start
past_episodes = recall_episodes("Fix authentication bug in FastAPI service")
episodic_context = "\n\n---\n\n".join(past_episodes)
```

#### Semantic Memory

```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Build semantic memory from codebase documentation
semantic_store = Chroma(
    collection_name="semantic_knowledge",
    embedding_function=embeddings,
    persist_directory="./memory/semantic",
)

def index_knowledge_base(docs_dir: str):
    """Index all markdown/text files in a directory into semantic memory."""
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    semantic_store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} documents")

@tool
def search_semantic_memory(query: str, k: int = 4) -> list[dict]:
    """Search the semantic knowledge base for facts, docs, or patterns.

    Use this before attempting tasks that require domain knowledge or
    API documentation.

    Args:
        query: What you're looking for. Be specific.
        k: Number of results (1-5).
    """
    results = semantic_store.similarity_search_with_score(query, k=k)
    return [
        {"content": doc.page_content, "source": doc.metadata.get("source", "?"),
         "relevance": round(1 - score, 3)}
        for doc, score in results
    ]
```

#### Procedural Memory

```python
# Procedural memory: how-to knowledge that shapes agent behaviour
# Stored as structured snippets, loaded into system prompt at startup

PROCEDURAL_LIBRARY = {
    "database_migrations": """
PROCEDURE: Database Migration
1. Never modify existing migration files — create new ones.
2. Always test migration on staging before production.
3. Include a rollback function in every migration.
4. Command: `alembic revision --autogenerate -m "description"`
""",

    "api_testing": """
PROCEDURE: Testing API Endpoints
1. Test happy path, invalid input, and auth failure for every endpoint.
2. Use TestClient from fastapi.testclient — do not start a real server.
3. Assert status code AND response body structure.
4. Mock external dependencies (databases, third-party APIs).
""",

    "git_workflow": """
PROCEDURE: Git Workflow
1. Never commit directly to main — always use feature branches.
2. Branch naming: feat/*, fix/*, chore/*, docs/*
3. Run tests before every commit.
4. Write descriptive commit messages: "<type>: <what changed and why>"
""",
}

def load_relevant_procedures(task: str) -> str:
    """Select relevant procedural memories based on the current task."""
    # Simple keyword matching — can be replaced with embedding search
    task_lower = task.lower()
    relevant = []

    if any(kw in task_lower for kw in ["migration", "database", "schema"]):
        relevant.append(PROCEDURAL_LIBRARY["database_migrations"])
    if any(kw in task_lower for kw in ["test", "endpoint", "api"]):
        relevant.append(PROCEDURAL_LIBRARY["api_testing"])
    if any(kw in task_lower for kw in ["commit", "branch", "git", "pr"]):
        relevant.append(PROCEDURAL_LIBRARY["git_workflow"])

    return "\n\n".join(relevant) if relevant else ""
```

#### Assembling All Memory Types at Session Start

```python
def build_agent_with_memory(task: str, user_id: str) -> dict:
    """
    Assemble all memory types before starting an agent session.
    This is the memory orchestration pattern for production agents.
    """
    # 1. Episodic — what happened in similar past sessions
    past_episodes = recall_episodes(task, k=2)

    # 2. Semantic — domain knowledge from documentation
    semantic_context = search_semantic_memory.invoke({"query": task, "k": 3})
    semantic_text = "\n---\n".join(r["content"] for r in semantic_context)

    # 3. Procedural — how-to knowledge for the task type
    procedures = load_relevant_procedures(task)

    # 4. Build system prompt with all memory injected
    system_prompt = f"""You are an expert software engineer.

{f"RELEVANT PAST EPISODES:{chr(10)}{chr(10).join(past_episodes)}" if past_episodes else ""}

{f"DOMAIN KNOWLEDGE:{chr(10)}{semantic_text}" if semantic_text else ""}

{f"PROCEDURES TO FOLLOW:{chr(10)}{procedures}" if procedures else ""}

Always use tools to verify assumptions. Run tests after every code change."""

    return {
        "system_prompt": system_prompt,
        "initial_state": {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task),
            ],
            "current_task": task,
            "iteration_count": 0,
            "intermediate_results": [],
        }
    }
```

:::lab Lab 2.2 — Agent with Full Memory Stack
**Objectives:**
- Build an agent with all four memory types: session (LangGraph state), episodic (ChromaDB), semantic (ChromaDB), procedural (CLAUDE.md).
- Run task: "Add rate limiting to the /login endpoint."
- After completion, save the episode to episodic memory.
- Run the same task again — verify the agent uses the stored episode.
- Measure: tokens used in session 1 vs. session 2 with memory warm.
:::
