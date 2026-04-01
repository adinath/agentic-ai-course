---
id: s21
number: "2.1"
title: "Memory Architecture for Agents"
time: "9:00–11:00 AM"
duration: "2 hours"
topics:
  - id: t-memtax
    title: "Memory Taxonomy"
  - id: t-rag
    title: "Vector Memory & RAG"
  - id: t-episodic
    title: "Persistent Memory"
---

Understand the four types of agent memory, design trade-offs, and implement each. Move beyond naive context stuffing to purposeful, scalable memory systems that persist across sessions.

### Topic: Memory Taxonomy {#t-memtax}

A human developer working on a complex task uses multiple types of memory simultaneously — they remember the current bug (working memory), recall how they solved a similar issue last month (episodic), know what a closure is without looking it up (semantic), and can type without thinking (procedural). Agent memory systems mirror this taxonomy. Designing the right combination for your use case is one of the highest-leverage architectural decisions you will make.

:::info Practical mapping to storage systems
Working memory → LangGraph AgentState (in-process dict)<br/>
Episodic memory → PostgreSQL table of run logs + embeddings<br/>
Semantic memory → Vector database (Chroma, Qdrant, Pinecone)<br/>
Procedural → System prompt snippets, retrieved at startup
:::

- **In-context (working) memory:** Fast and immediately accessible but finite and lost when the conversation ends.
- **External memory:** Stored in vector databases or KV stores. Persists across sessions; accessed by retrieval.
- **Episodic memory:** Logs of past agent runs — what worked, what failed. Prevents repeating past mistakes.
- **Semantic/procedural memory:** Distilled facts and heuristics injected into the system prompt at task start.

#### Choosing a Memory Write Strategy

Not everything that happens in a session is worth storing. Writing too much creates a noisy memory store where irrelevant retrievals crowd out useful ones. Writing too little means the agent never learns from experience. A practical rule of thumb:

- **Always write:** Successful task solutions, discovered edge cases, confirmed API behaviours.
- **Never write:** Failed intermediate attempts, raw tool output dumps, routine status messages.
- **Write conditionally:** Anything the agent marks as a "new learning" with confidence above a threshold (e.g., 0.7).

### Topic: Vector Memory & RAG for Agents {#t-rag}

Retrieval-Augmented Generation (RAG) is the most commonly used external memory pattern. Before acting, the agent retrieves relevant knowledge from a vector store and injects it into the context. The challenge is retrieval quality — irrelevant or noisy retrieved text degrades agent performance more than having no external memory at all.

#### Embedding & Chunking Strategies

How you chunk and embed documents determines what you can retrieve. The three main strategies, in order of increasing quality and cost:

- **Fixed-size chunking:** Split by token count (e.g. 512 tokens, 50-token overlap). Simple and fast; may split mid-concept.
- **Semantic chunking:** Split at sentence boundaries based on embedding similarity shifts. Better semantic coherence; requires an embedding pass over the text first.
- **Proposition chunking:** Extract self-contained factual statements ("The API rate limit is 60 requests per minute"). Best retrieval precision; slowest to build.

:::tip Start with fixed-size, move to semantic when you have eval data
Fixed-size chunking is fast to iterate on. Once you have a retrieval eval dataset showing where it fails (split concepts, missing context), migrate to semantic chunking for those document types specifically.
:::

#### Setting Up a Vector Memory Store

```python
from langchain_chroma import Chroma
from langchain_anthropic import AnthropicEmbeddings
from langchain_core.documents import Document

embeddings = AnthropicEmbeddings(model="voyage-3")
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./memory_db",
)

def save_memory(task: str, solution: str, tags: list[str]):
    doc = Document(
        page_content=f"Task: {task}\nSolution: {solution}",
        metadata={"tags": tags, "timestamp": datetime.utcnow().isoformat()},
    )
    vectorstore.add_documents([doc])

def get_relevant_memories(task: str, k: int = 3) -> list[str]:
    docs = vectorstore.similarity_search(task, k=k)
    return [d.page_content for d in docs]
```

#### Agentic RAG: On-Demand Retrieval Tool

```python
@tool
def search_memory(query: str, k: int = 3) -> list[dict]:
    """Search past solutions relevant to the current query.
    Call this before starting a task to check if we've solved
    something similar before.

    Args:
        query: Natural language description of what you need.
        k: Number of results (1-5).
    """
    docs = vectorstore.similarity_search(query, k=k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
```

### Topic: Persistent & Episodic Memory {#t-episodic}

#### Summarization-Based Memory Compression

After a long agent run, a cheap fast model summarizes the full conversation into a compact episodic memory that can be retrieved later — reducing storage costs while preserving the key learnings.

```python
summariser = ChatAnthropic(model="claude-haiku-4-5")

def compress_and_store(session_messages: list, task: str):
    conversation_text = "\n".join(
        f"{m.type}: {m.content[:500]}" for m in session_messages
    )
    summary = summariser.invoke([
        HumanMessage(
            f"Summarize this agent session in 3-5 bullet points.\n"
            f"Focus on: what was attempted, what succeeded, key learnings.\n\n"
            f"{conversation_text}"
        )
    ])
    save_memory(task=task, solution=summary.content, tags=["episode"])
```

#### User & Session Profile Stores

For agents that interact with the same user repeatedly, maintaining a structured user profile dramatically improves personalisation. The profile stores preferences, coding style, past decisions, and project context that shapes future responses — without needing to rediscover them each session.

```python
import json
from pathlib import Path

class UserProfileStore:
    def __init__(self, storage_dir: str = "./profiles"):
        self.dir = Path(storage_dir)
        self.dir.mkdir(exist_ok=True)

    def get(self, user_id: str) -> dict:
        path = self.dir / f"{user_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"user_id": user_id, "preferences": {}, "history_summary": ""}

    def update(self, user_id: str, updates: dict):
        profile = self.get(user_id)
        profile.update(updates)
        (self.dir / f"{user_id}.json").write_text(json.dumps(profile, indent=2))

profiles = UserProfileStore()

# Before a session: inject profile into system prompt
profile = profiles.get(user_id)
system = (f"User preferences: {json.dumps(profile['preferences'])}\n"
          f"Past session context: {profile['history_summary']}")
```

:::lab Lab 2.1 — Developer Assistant with Vector Memory
**Objectives:**
- Extend the Day 1 assistant with ChromaDB vector memory.
- Before each task: retrieve and inject top-3 relevant past solutions.
- After each task: summarize the session and write back to memory.
- Run the same bug-fix task twice — observe how the second run uses memory.
:::
