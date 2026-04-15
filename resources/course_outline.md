# AI Agents Training Curriculum

## Day 1 — LLM Based Agents

**Fundamentals:** Introduction to Agents and type of Agents

- What is an AI Agent?
- Types of Agents
- Framework Introduction
  - Framework landscape
  - Langchain & Langgraph
  - Using hosted models
  - Using local models (Ollama)
- Tool calling
- Tool design
- React Agent
  - Max iterations
  - Error scenarios
- Plan And Execute
- RAG
  - Knowledge graph
  - RAG Evaluation (Ragas)

---

## Day 2 — Single Agents

- AIFSD
  - Sense AI Framework
  - Rules, Commands, Skills
  - Sub Agents - Local / Cloud agents
  - Agent First development
- MCP
- Context Engineering
- Memory management
  - Session memory
  - Long term memory
    - Episodic memory
    - Semantic memory
    - Procedural Memory
- Observability (Langfuse)
- Guardrails

---

## Day 3 — Multi-Agent System

- Topologies
  - Supervisor - Worker
  - Router - Expert
  - Orchestrator
  - Round Table
  - Tree of Agents
  - Pipeline
- Agent to agent communication
  - A2A (Agent To Agent)
  - ACP (Agent Client Protocol)
- Agent Evaluation
  - Tool call accuracy
  - Tool Trajectory
- Capstone Project

---

## Day 3 — Extended Session: Agent Performance Engineering

- The Performance Tax
  - Cost anatomy of agentic loops
  - Four performance levers
- Prompt Caching
  - Anthropic cache_control (explicit breakpoints, 90% read discount)
  - OpenAI automatic prefix caching (50% discount)
  - Structuring prompts for maximum cache hits
- Semantic Caching
  - LangChain InMemorySemanticCache / RedisSemanticCache
  - GPTCache (FAISS / Milvus backends)
  - Threshold tuning and TTL management
- Right-Sized Models
  - Model selection matrix (Haiku / Sonnet / Opus)
  - Dynamic routing in LangGraph
  - RouteLLM learned routing (Stanford)
- Async & Parallelism
  - asyncio.gather for parallel LLM calls
  - LangGraph Send API (fan-out / fan-in)
  - Concurrency limits with asyncio.Semaphore
- Token Budget Management
  - Prompt token auditing (count_tokens API)
  - Per-operation max_tokens discipline
  - Context window hygiene and history trimming
