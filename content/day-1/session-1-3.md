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
  - id: t-rageval
    title: "RAG Evaluation"
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
executor_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
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
pip install langchain-chroma langchain-anthropic chromadb
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

#### Knowledge Graphs for Structured Retrieval

Vector search is powerful but returns unordered chunks — it loses the *relationships* between entities. A knowledge graph stores entities and their connections, enabling structured traversal queries.

```
┌──────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                        │
│                                                           │
│  [LangGraph] ──uses──▶ [StateGraph]                      │
│      │                      │                            │
│    ──has──▶ [ToolNode] ──executes──▶ [Tool]             │
│      │                                                   │
│    ──integrates──▶ [Anthropic Claude]                    │
│                         │                                │
│                    ──supports──▶ [Tool Calling]          │
└──────────────────────────────────────────────────────────┘
```

```python
# Using NetworkX for lightweight in-memory knowledge graphs
import networkx as nx
from langchain_core.tools import tool

# Build a knowledge graph of your codebase architecture
kg = nx.DiGraph()
kg.add_nodes_from([
    ("LangGraph", {"type": "framework", "version": "1.1"}),
    ("StateGraph", {"type": "class", "module": "langgraph.graph"}),
    ("ToolNode", {"type": "class", "module": "langgraph.prebuilt"}),
])
kg.add_edges_from([
    ("LangGraph", "StateGraph", {"relation": "provides"}),
    ("LangGraph", "ToolNode", {"relation": "provides"}),
    ("StateGraph", "ToolNode", {"relation": "can_contain"}),
])

@tool
def query_knowledge_graph(entity: str, relation: str = None) -> list[dict]:
    """Query the knowledge graph for an entity and its relationships.

    Args:
        entity: The entity name to look up (e.g. 'LangGraph', 'StateGraph').
        relation: Optional. Filter by relation type (e.g. 'provides', 'depends_on').

    Returns:
        List of {"from": "...", "relation": "...", "to": "...", "attributes": {...}}
    """
    if entity not in kg:
        return [{"error": f"Entity '{entity}' not in knowledge graph"}]

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

### Topic: RAG Evaluation with Ragas {#t-rageval}

You cannot improve what you cannot measure. Ragas is the standard library for evaluating RAG pipelines — it measures whether your retrieval is actually helping the agent answer correctly.

```bash
pip install ragas datasets
```

#### The Four Ragas Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│  RAGAS EVALUATION METRICS                                        │
├────────────────────────┬────────────────────────────────────────┤
│  Faithfulness          │ Does the answer match the retrieved     │
│                        │ context? (hallucination detector)       │
├────────────────────────┼────────────────────────────────────────┤
│  Answer Relevancy      │ Is the answer relevant to the question? │
│                        │ (response quality)                      │
├────────────────────────┼────────────────────────────────────────┤
│  Context Precision     │ Are the retrieved chunks actually useful?│
│                        │ (retrieval precision)                   │
├────────────────────────┼────────────────────────────────────────┤
│  Context Recall        │ Did we retrieve everything needed?      │
│                        │ (retrieval completeness)                │
└────────────────────────┴────────────────────────────────────────┘
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
- Planner uses `claude-sonnet-4-6`; Executor uses `claude-haiku-4-5`.
- After the run, evaluate retrieval quality with Ragas on 5 question/answer pairs.
- Compare task completion: agent with RAG vs. agent without — measure difference in answer quality.
:::
