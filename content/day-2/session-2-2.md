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

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT MEMORY TAXONOMY                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SESSION MEMORY (Working Memory)                        │   │
│  │  Scope: current conversation only                       │   │
│  │  Storage: LangGraph AgentState (in-process dict)        │   │
│  │  Access: immediate, O(1)                                │   │
│  │  Example: current task, intermediate results            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  EPISODIC MEMORY                                        │   │
│  │  Scope: logs of past agent runs                         │   │
│  │  Storage: PostgreSQL + embeddings                       │   │
│  │  Access: vector similarity search                       │   │
│  │  Example: "Last time I fixed auth errors, I did X"      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SEMANTIC MEMORY                                        │   │
│  │  Scope: domain knowledge, facts, documentation          │   │
│  │  Storage: vector database (Chroma, Qdrant, Pinecone)    │   │
│  │  Access: similarity search by query                     │   │
│  │  Example: API docs, codebase patterns, team norms       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  PROCEDURAL MEMORY                                      │   │
│  │  Scope: how-to knowledge, heuristics, workflows         │   │
│  │  Storage: system prompt snippets, CLAUDE.md files       │   │
│  │  Access: loaded at startup, retrieved by tag            │   │
│  │  Example: "Always run tests before committing"          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
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
