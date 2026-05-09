---
id: s13
number: "1.3"
title: "Plan & Execute + RAG"
time: "3:00–5:00 PM"
duration: "2 hours"
topics:
  - id: t-planexecute
    title: "Plan and Execute"
  - id: t-rag
    title: "RAG for Agents"
  - id: t-chunking
    title: "Chunking Strategies"
  - id: t-kg
    title: "Knowledge Graphs & GraphRAG"
  - id: t-rageval
    title: "RAG Evaluation with Ragas"
---

ReAct is reactive — it figures out the next step only after seeing the previous result. Plan-and-Execute is proactive — it lays out the full roadmap before the first tool call. RAG gives the agent a long-term memory it can actually trust.

### Topic: Plan and Execute {#t-planexecute}

#### Why Planning Beats Pure ReAct for Complex Tasks

ReAct works brilliantly for exploratory tasks. It falls apart when:
- Steps have dependencies that require upfront coordination.
- A mid-task failure means you want to replan from that point, not restart.
- Token costs are high and you want the LLM to think once rather than 15 times.

Plan-and-Execute separates **strategic thinking** (Planner, uses a powerful model once) from **tactical execution** (Executor, uses a cheaper model per step).

```
┌─────────────────────────────────────────────────────────────┐
│            PLAN-AND-EXECUTE ARCHITECTURE                     │
│                                                              │
│  Task ──▶ Planner ──▶ [Step 1, Step 2, Step 3, Step 4]     │
│                              │                               │
│                        Executor Loop                         │
│                         ┌────┴────┐                          │
│                         │ Step N  │──▶ Tools ──▶ Result     │
│                         └────┬────┘                          │
│                   ┌──────────┴──────────┐                    │
│                  Pass                  Fail                  │
│                   │                     │                    │
│              Next Step            Replanner                  │
│                                        │                     │
│                                New Plan from N               │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation with LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Optional
import json

class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    current_step: int
    step_results: List[str]
    final_answer: Optional[str]
    replan_count: int

# Planner: uses a capable model, called once (or on replan)
planner_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

def planner(state: PlanExecuteState) -> dict:
    """Decompose the task into sequential steps."""
    response = planner_llm.invoke([
        SystemMessage(content=(
            "You are a planning agent. Decompose the task into 3-7 concrete, "
            "executable steps. Output ONLY valid JSON: {\"steps\": [\"step1\", ...]}"
        )),
        HumanMessage(content=state["task"]),
    ])

    try:
        data = json.loads(response.content)
        steps = data["steps"]
    except (json.JSONDecodeError, KeyError):
        # Fallback: treat the whole response as a single step
        steps = [response.content]

    return {
        "plan": steps,
        "current_step": 0,
        "step_results": [],
    }

# Executor: cheaper model, called once per step
executor_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
executor_llm_with_tools = executor_llm.bind_tools([read_file, write_file, run_tests])

def executor(state: PlanExecuteState) -> dict:
    """Execute the current plan step using tools."""
    step = state["plan"][state["current_step"]]
    prior_results = "\n".join(
        f"Step {i+1} result: {r}" for i, r in enumerate(state["step_results"])
    )

    messages = [
        SystemMessage(content="Execute the given step using available tools."),
        HumanMessage(content=(
            f"Overall task: {state['task']}\n\n"
            f"Previous step results:\n{prior_results}\n\n"
            f"Current step to execute: {step}"
        )),
    ]

    # Mini ReAct loop for this step (max 5 iterations)
    for _ in range(5):
        response = executor_llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            result = response.content
            break

        # Execute tools
        from langchain_core.messages import ToolMessage
        for tc in response.tool_calls:
            try:
                tool_result = tool_registry[tc["name"]].invoke(tc["args"])
            except Exception as e:
                tool_result = f"ERROR: {e}"
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
    else:
        result = "Step did not complete within iteration limit."

    new_results = state["step_results"] + [result]
    return {
        "step_results": new_results,
        "current_step": state["current_step"] + 1,
    }

def replanner(state: PlanExecuteState) -> dict:
    """Generate a new plan starting from the failed step."""
    completed = "\n".join(
        f"✓ {step}: {result}"
        for step, result in zip(state["plan"], state["step_results"])
    )
    response = planner_llm.invoke([
        SystemMessage(content=(
            "You are a replanning agent. Given completed steps and a failure, "
            "generate a NEW plan for the remaining work. "
            "Output JSON: {\"steps\": [\"step1\", ...]}"
        )),
        HumanMessage(content=(
            f"Original task: {state['task']}\n\n"
            f"Completed:\n{completed}\n\n"
            f"Last step failed. Generate a revised plan for the remaining work."
        )),
    ])
    try:
        data = json.loads(response.content)
        new_plan = data["steps"]
    except (json.JSONDecodeError, KeyError):
        new_plan = [response.content]

    return {
        "plan": state["step_results"] + new_plan,  # preserve completed steps
        "current_step": len(state["step_results"]),
        "replan_count": state.get("replan_count", 0) + 1,
    }

def synthesiser(state: PlanExecuteState) -> dict:
    """Combine all step results into a coherent final answer."""
    results_summary = "\n".join(
        f"Step {i+1} ({step}): {result}"
        for i, (step, result) in enumerate(zip(state["plan"], state["step_results"]))
    )
    response = planner_llm.invoke([
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Execution results:\n{results_summary}\n\n"
            "Summarise the outcome for the user."
        )),
    ])
    return {"final_answer": response.content}

def route_after_execute(state: PlanExecuteState) -> str:
    """Decide: next step, replan, or synthesise."""
    last_result = state["step_results"][-1] if state["step_results"] else ""
    all_steps_done = state["current_step"] >= len(state["plan"])

    if all_steps_done:
        return "synthesise"
    if "ERROR" in last_result and state.get("replan_count", 0) < 2:
        return "replan"
    if all_steps_done:
        return "synthesise"
    return "execute"

# Build the graph
g = StateGraph(PlanExecuteState)
g.add_node("planner", planner)
g.add_node("executor", executor)
g.add_node("replanner", replanner)
g.add_node("synthesiser", synthesiser)

g.add_edge(START, "planner")
g.add_edge("planner", "executor")
g.add_conditional_edges("executor", route_after_execute, {
    "execute": "executor",
    "replan": "replanner",
    "synthesise": "synthesiser",
})
g.add_edge("replanner", "executor")
g.add_edge("synthesiser", END)

plan_execute_app = g.compile()
```

### Topic: RAG for Agents {#t-rag}

RAG (Retrieval-Augmented Generation) gives agents access to knowledge that exceeds the context window. Instead of stuffing all documentation into every prompt, the agent retrieves only the relevant chunks at query time.

#### How RAG Works in an Agent Context

```
Query ──▶ Embed ──▶ Vector Search ──▶ Top-K Chunks
                                           │
                                    Inject into Context
                                           │
                                        LLM ──▶ Answer
```

The key difference from "RAG for chatbots" is that in an agent, **retrieval is a tool** — the agent decides *when* to search, *what* to search for, and *how many* results to retrieve based on the current task state.

#### Setting Up a Vector Store

```bash
pip install "langchain-chroma>=0.2" "langchain-anthropic>=1.4" chromadb
```

```python
from langchain_chroma import Chroma
from langchain_anthropic import AnthropicEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from datetime import datetime

# Voyage-3 embeddings — optimised for retrieval (Anthropic hosted)
embeddings = AnthropicEmbeddings(model="voyage-3")

vectorstore = Chroma(
    collection_name="course_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

def index_documents(docs: list[dict]):
    """Chunk and embed documents into the vector store."""
    documents = [
        Document(
            page_content=doc["content"],
            metadata={
                "source": doc["source"],
                "indexed_at": datetime.utcnow().isoformat(),
            }
        )
        for doc in docs
    ]
    vectorstore.add_documents(documents)
    print(f"Indexed {len(documents)} documents")
```

#### Agentic RAG Tool

```python
@tool
def search_knowledge_base(query: str, k: int = 4) -> list[dict]:
    """Search the knowledge base for information relevant to the query.

    Use this tool BEFORE attempting to answer questions about domain knowledge,
    APIs, internal documentation, or past solutions.

    Args:
        query: Natural language description of what you need. Be specific.
        k: Number of results to return (1-5). Default 4.

    Returns:
        List of {"content": "...", "source": "...", "score": 0.0-1.0}
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(1 - score, 3),  # convert distance to similarity
        }
        for doc, score in results
    ]
```

### Topic: Chunking Strategies {#t-chunking}

The retriever can only find what the chunker stored. Chunking is the most under-rated knob in RAG — bad chunking starves the LLM of context no matter how good your embeddings are.

#### Why Chunk Size Matters

```
   Fragmented           Sweet spot              Diluted
┌──────────────┬────────────────────────┬──────────────────┐
│ no surrounding│  256 – 1024 tokens   │ multiple topics  │
│   context     │   (domain-dependent)  │   per chunk      │
└──────────────┴────────────────────────┴──────────────────┘
   ~50 tok          ~512 tok                ~4096 tok
```

There is no universal answer — domain *and* embedding model dictate the band. The three failure modes:

- **Too small** — snippets ripped from context. Retrieval finds the right line, but the LLM has nothing to reason with.
- **Too large** — multiple topics averaged into one embedding vector. Cosine similarity drops, retrieval misses, irrelevant text fills the context window.
- **Wrong boundary** — cuts mid-sentence or mid-code-block. Embedding distorted, LLM sees grammatically broken text.

Run an A/B sweep: index the same corpus at 128/256/512/1024 tokens, replay your eval set, pick the cheapest size that hits your faithfulness threshold.

#### Strategy 1 — Fixed-Size + Overlap

The baseline. Cut every `chunk_size` characters/tokens with a sliding overlap so context isn't lost at boundaries.

```
…the quick brown fox jumps over the lazy dog. The dog barked at the moon…

CHUNKS  size = 512  overlap = 50
[ chunk 1  0–512 ]
       [ chunk 2  462–974 ]   ← 50-token overlap with chunk 1
              [ chunk 3  924–1436 ]
```

`stride = chunk_size − overlap`. Typical overlap: 10–20% of `chunk_size`.

- **When to use:** plain unstructured text, demos, baseline benchmarks. Fast and predictable.
- **Watch out for:** cuts code blocks, tables, JSON, and Markdown headings in half. Storage grows linearly with overlap.

#### Strategy 2 — Recursive Character Splitting

Try natural separators in priority order, fall back to smaller units only when a chunk is still too big.

```
\n\n   →   \n   →   ". "   →   " "   →   char
paragraph  line     sentence    word     last resort
priority 1 …                              priority 5

Each step: if any chunk ≤ chunk_size  →  ✓ keep this split
```

- **When to use:** default for prose — docs, articles, blog posts, support tickets. Almost always beats fixed-size.
- **Watch out for:** the default separator list is English-centric. Code, JSON, and structured formats need a custom list (`Language.PYTHON`, `Language.MARKDOWN`).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Generic prose
prose_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Python source code — splits on def/class/etc. before falling back to lines
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=120,
)
```

#### Strategy 3 — Document-Aware Chunking

When the source has structure, exploit it. Markdown headings, HTML sections, RST, JSON, and code already carry author-intended boundaries — use them directly *and* preserve the hierarchy as searchable metadata.

```
SOURCE                              CHUNKS WITH HIERARCHY
# Authentication       ─────▶       Chunk 1 · Authentication
                                    { h1: "Authentication" }
## OAuth                            Chunk 2 · OAuth
                                    { h1: "Authentication", h2: "OAuth" }
### PKCE                            Chunk 3 · PKCE
                                    { h1: "Authentication", h2: "OAuth", h3: "PKCE" }
## API Keys                         Chunk 4 · API Keys
                                    { h1: "Authentication", h2: "API Keys" }
```

Each chunk inherits its heading path. The LLM gets the topic context for free; metadata filters let you scope queries (e.g. `where h2 = "OAuth"`).

- **When to use:** Markdown, HTML, source code, RST, JSON. Anywhere structure exists.
- **Watch out for:** sections vary wildly in size — combine with a size-based splitter inside each section. Useless on plain prose.

#### Strategy 4 — Semantic Chunking

Embed every sentence, measure cosine distance between adjacent sentences, cut where the distance peaks above a threshold (e.g. the 95th percentile of all distances). Each chunk now contains a single coherent topic.

```
distance
  1.0 ┤    ▲                ▲                            ▲
      │    │                │                            │
  0.5 ┤────│────────────────│────────────────────────────│──── threshold
      │    │     │  │       │   │  │   │   │  │   │      │  │
  0.0 ┤────●─────●──●───────●───●──●───●───●──●───●──────●──●──
        topic A    │ topic B    │      topic C           │ topic D
                  cut          cut                      cut
```

- **When to use:** long-form technical writing where topics shift mid-section — research papers, transcripts, mixed-topic blog posts. Quality over speed.
- **Watch out for:** slow (embeds every sentence at index time), threshold tuning required, falls back to recursive splitting if all sentences look similar.

#### Strategy 5 — Parent-Child (Small-to-Big) Retrieval

Embed *small* precise chunks for retrieval, but return their *large* parent for context. Best of both worlds: precise matching, rich context for the LLM.

```
                        ┌──────────────────────────┐
                        │   Parent Document        │
                        │  ~2000 tok · NOT embedded│
                        └─────┬─────┬─────┬────────┘
                              │     │     │
                    ┌─────────┘     │     └─────────┐
                    ▼               ▼               ▼
                 Child 1         Child 3        Child 4
              ~200 tok          ← match        ~200 tok
              embedded         (cosine 0.91)   embedded
                                   │
                  query ──── 1. match small ──┘
                            2. return parent (large) for context
```

Implementation: `ParentDocumentRetriever` (LangChain), `SmallToBigRetriever` (LlamaIndex).

#### What's Next — 2024+ Techniques

- **Contextual Retrieval (Anthropic, 2024)** — prepend an LLM-generated 50-100 token context blurb to each chunk before embedding. ~49% retrieval-failure reduction on standard benchmarks. Pairs with prompt caching to keep cost flat.
- **Late Chunking (Jina AI, 2024)** — embed the whole document with a long-context embedding model, *then* slice the resulting token embeddings. Each chunk's vector now reflects document-wide context.

### Topic: Knowledge Graphs & GraphRAG {#t-kg}

Vector search retrieves what is **similar**. Knowledge graphs retrieve what is **related**. For multi-hop questions ("X of Y of Z"), similarity quietly fails and relations win.

#### When Vector RAG Hits a Wall

```
QUERY:  "What products does the company that acquired Acme sell?"

VECTOR RAG (top-K chunks)              KNOWLEDGE GRAPH (traversal)

▸ "Acme was acquired in Q2 2024…"      [Acme] ──acquired_by──▶ [TechCo]
▸ "TechCo's product line includes…"                                │
▸ "Tech-sector M&A trends…"  (distract)                          sells
                                                                   ▼
✗ LLM must synthesise across chunks      ✓ Direct multi-hop traversal
   no single chunk has the chained         answer = the nodes returned
   answer                                  by the query
```

#### Six Wins for Knowledge Graphs

| Dimension | Vector RAG | Knowledge Graph |
|---|---|---|
| Multi-hop reasoning | Brittle — synthesis across chunks | Native — graph traversal |
| Explainability | Opaque cosine scores | The path *is* the answer |
| Determinism | Embedding drift between models | Exact traversal — same query, same path |
| Schema | None — free text | Typed entities & relations |
| Aggregations | Very limited | Count, group, filter, shortest-path |
| Phantom entities | Hallucination risk | Only existing nodes can be returned |

**Best-fit domains:** org charts, supply chains, compliance & regulation, code dependencies, drug interactions, fraud rings — anywhere relations *are* the data.

**The trade-off:** index-time cost rises. Entity extraction (LLM-driven) and schema discipline are required. Setup is days, not minutes.

#### A Lightweight In-Memory Knowledge Graph

```python
import networkx as nx
from langchain_core.tools import tool

kg = nx.DiGraph()
kg.add_nodes_from([
    ("LangGraph", {"type": "framework", "version": "1.1"}),
    ("StateGraph", {"type": "class", "module": "langgraph.graph"}),
    ("ToolNode",   {"type": "class", "module": "langgraph.prebuilt"}),
])
kg.add_edges_from([
    ("LangGraph",  "StateGraph", {"relation": "provides"}),
    ("LangGraph",  "ToolNode",   {"relation": "provides"}),
    ("StateGraph", "ToolNode",   {"relation": "can_contain"}),
])

@tool
def query_knowledge_graph(entity: str, relation: str | None = None) -> list[dict]:
    """Look up an entity and its outgoing relationships.

    Args:
        entity: The entity to look up (e.g. 'LangGraph').
        relation: Optional filter by relation type (e.g. 'provides').
    Returns:
        List of {"from", "relation", "to", "attributes"} edges.
    """
    if entity not in kg:
        return [{"error": f"Entity '{entity}' not in graph"}]
    results = []
    for source, target, data in kg.edges(entity, data=True):
        if relation is None or data.get("relation") == relation:
            results.append({
                "from": source,
                "relation": data.get("relation"),
                "to": target,
                "attributes": kg.nodes[target],
            })
    return results
```

For production-scale graphs use **Neo4j** (Cypher queries) or **Memgraph** (open-source, Bolt-compatible). Both have first-class LangChain integrations.

#### GraphRAG — The Hybrid Pattern

The strongest RAG architecture in 2025 is hybrid: **vector for entry**, **graph for expansion**, **LLM for grounding**.

```
QUERY-TIME · HYBRID GRAPHRAG

Query  ──▶  Vector seed  ──▶  Graph expand  ──▶  Subgraph  ──▶  LLM
            (entry nodes)     (1–2 hop)         (facts as     (grounded
                                                 context)      answer)

  SIMILARITY            RELATIONS              GROUNDING

Index-time: LLM extracts entities + relations from each chunk
            → graph DB & vector index of entity descriptions
```

- **Use both when:** relational queries on top of an unstructured corpus, audit-required answers, "list / count / who-related-to-whom" questions over enterprise knowledge.
- **Stick with vector RAG when:** pure semantic search, long-form prose Q&A, fast iteration, no clear entity schema. Don't pay the GraphRAG tax for similarity-only problems.

### Topic: RAG Evaluation with Ragas {#t-rageval}

You cannot improve what you cannot measure. Ragas is the standard library for evaluating RAG pipelines — it measures whether your retrieval is actually helping the agent answer correctly.

```bash
pip install "ragas>=0.2" datasets
```

#### How Ragas Actually Works

Ragas decomposes each answer into atomic claims (or sentences, or entities), then uses an **LLM-as-judge** to score every claim against the retrieved context, the ground truth, or the question itself.

```
EVALUATION SAMPLE                                METRIC SCORES (0–1)
┌─────────────────────────────┐                 ┌───────────────────┐
│ question      "Who acquired │                 │ Faithfulness 0.92 │
│               Acme & when?" │ ──▶ LLM ──▶     │ Answer Rel.  0.88 │
│ answer        "TechCo Q2…"  │     judge       │ Ctx Precision 0.75│
│ retrieved_ctx [chunks]      │     (gpt-4o-mini)│ Ctx Recall   0.81 │
│ ground_truth  (optional)    │                 └───────────────────┘
└─────────────────────────────┘
```

#### The Four Ragas Metrics

| Metric | What it catches | Needs ground truth? |
|---|---|---|
| **Faithfulness** | Hallucinations — claims in the answer that aren't supported by retrieved context | No |
| **Answer Relevancy** | Drift — answer not aligned with the question | No |
| **Context Precision** | Noisy retrieval — irrelevant chunks ranked high | Yes (ideal) / No (LLM-only mode) |
| **Context Recall** | Missed relevant info — retriever didn't return what was needed | Yes |

Three of four metrics work **without ground truth**, so you can run them on live production traffic. Wire the result into CI: fail the build when Faithfulness drops below your threshold and the hallucination never ships.

```python
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)
```

#### Running Ragas Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
from datasets import Dataset

# Collect evaluation data from agent runs
eval_data = {
    "question": [
        "What is the max context window for claude-sonnet-4-6?",
        "How do I add a tool to a LangGraph agent?",
    ],
    "answer": [
        "Claude Sonnet 4.6 supports a 200,000 token context window.",
        "Use llm.bind_tools([your_tool]) and add a ToolNode to the graph.",
    ],
    "contexts": [
        ["claude-sonnet-4-6 has a 200K token context window..."],
        ["ToolNode executes tool calls...", "bind_tools() attaches tools to a model..."],
    ],
    "ground_truth": [
        "200,000 tokens",
        "Call llm.bind_tools(tools) and add ToolNode to the graph",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Wrap Anthropic models for Ragas
evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6"))
evaluator_embeddings = LangchainEmbeddingsWrapper(AnthropicEmbeddings(model="voyage-3"))

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print(result)
# Expected output:
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.85, 'context_recall': 0.91}
```

#### Interpreting Results & Fixing Issues

```python
def diagnose_rag_pipeline(scores: dict) -> list[str]:
    """Diagnose RAG quality issues from Ragas scores."""
    issues = []

    if scores.get("faithfulness", 1) < 0.7:
        issues.append(
            "⚠️ LOW FAITHFULNESS: Model is hallucinating beyond retrieved context. "
            "Fix: strengthen system prompt ('only answer from provided context'), "
            "increase k (retrieve more chunks), improve chunk quality."
        )

    if scores.get("context_precision", 1) < 0.7:
        issues.append(
            "⚠️ LOW CONTEXT PRECISION: Retrieving noisy/irrelevant chunks. "
            "Fix: use hybrid search (vector + BM25), add metadata filters, "
            "switch from fixed-size to semantic chunking."
        )

    if scores.get("context_recall", 1) < 0.7:
        issues.append(
            "⚠️ LOW CONTEXT RECALL: Missing relevant information in retrieval. "
            "Fix: increase k, check embedding model alignment with queries, "
            "review chunk size (too large = diluted embeddings)."
        )

    if scores.get("answer_relevancy", 1) < 0.7:
        issues.append(
            "⚠️ LOW ANSWER RELEVANCY: Responses drift from the question. "
            "Fix: improve system prompt specificity, add output format constraints."
        )

    return issues if issues else ["✅ All metrics above threshold — RAG pipeline healthy."]
```

:::lab Lab 1.3 — Plan-and-Execute Agent with RAG
**Objectives:**
- Build a Plan-and-Execute agent for: "Audit the codebase for security vulnerabilities and generate a report."
- Add a `search_knowledge_base` tool backed by ChromaDB (index OWASP Top-10 summaries).
- Planner uses `claude-sonnet-4-6`; Executor uses `claude-haiku-4-5-20251001`.
- After the run, evaluate retrieval quality with Ragas on 5 question/answer pairs.
- Compare task completion: agent with RAG vs. agent without — measure difference in answer quality.
:::
