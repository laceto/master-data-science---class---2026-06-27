# LangChain DeepAgents: The Batteries-Included Agent Harness

> **Current version:** 0.6.11 (June 18, 2026) · MIT License · Python ≥ 3.11  
> **Repository:** [github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)  
> **Docs:** [docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview)

---

## Table of Contents

1. [What is DeepAgents?](#1-what-is-deepagents)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation](#3-installation)
4. [Quickstart](#4-quickstart)
5. [Core Capabilities](#5-core-capabilities)
   - 5.1 [Planning & Task Decomposition](#51-planning--task-decomposition)
   - 5.2 [Filesystem](#52-filesystem)
   - 5.3 [Context Management](#53-context-management)
   - 5.4 [Persistent Memory (AGENTS.md)](#54-persistent-memory-agentsmd)
   - 5.5 [Skills (SKILL.md)](#55-skills-skillmd)
   - 5.6 [Subagents (Inline)](#56-subagents-inline)
   - 5.7 [Async Subagents (Background)](#57-async-subagents-background)
   - 5.8 [Human-in-the-Loop](#58-human-in-the-loop)
   - 5.9 [Shell Execution](#59-shell-execution)
6. [Middleware System](#6-middleware-system)
7. [Model Support](#7-model-support)
8. [Advanced Configuration](#8-advanced-configuration)
9. [Deep Research Example](#9-deep-research-example)
10. [Production & Observability](#10-production--observability)
11. [Guiding Principles](#11-guiding-principles)
12. [Version History & Key Milestones](#12-version-history--key-milestones)
13. [Resources](#13-resources)

---

## 1. What is DeepAgents?

**DeepAgents** is an open-source, opinionated agent harness built by LangChain for long-running, multi-step autonomous tasks. It sits as a layer above LangGraph and LangChain core, bundling in the building blocks that nearly every production agent needs — planning, file management, context compression, memory, subagent orchestration, and skills — so you can focus on domain logic rather than infrastructure.

> "An opinionated agent that runs out of the box. Extend, override, or replace any piece."

DeepAgents occupies the middle ground in LangChain's ecosystem:

```
LangGraph (low-level graph runtime)
    ↑
create_agent (lightweight wrapper)
    ↑
DeepAgents / create_deep_agent (batteries-included harness)  ← you are here
```

Key differentiators from a basic ReAct agent:

| Feature | Basic Agent | DeepAgents |
|---|---|---|
| Planning / todos | Manual | Built-in `write_todos` |
| File system access | Manual | Pluggable backends |
| Context compression | Manual | Automatic |
| Memory across sessions | Manual | `AGENTS.md` |
| Reusable behaviors | Manual | `SKILL.md` |
| Subagent delegation | Manual | Built-in `task` tool |
| Async background agents | Not available | v0.5+ |
| Human-in-the-loop | Manual | Built-in |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Your Application               │
└───────────────────────┬─────────────────────────┘
                        │ invoke / stream
┌───────────────────────▼─────────────────────────┐
│              create_deep_agent()                │
│                                                 │
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ Planning │  │ Filesystem │  │   Memory    │ │
│  │(write_   │  │(read/write │  │ (AGENTS.md) │ │
│  │  todos)  │  │ /edit/find)│  │             │ │
│  └──────────┘  └────────────┘  └─────────────┘ │
│                                                 │
│  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
│  │  Skills  │  │  Context   │  │  Subagents  │ │
│  │(SKILL.md)│  │Compression │  │(sync/async) │ │
│  └──────────┘  └────────────┘  └─────────────┘ │
│                                                 │
│  ┌────────────────────────────────────────────┐ │
│  │           Middleware Pipeline              │ │
│  └────────────────────────────────────────────┘ │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│              LangGraph Runtime                  │
│   (streaming, persistence, checkpointing)       │
└─────────────────────────────────────────────────┘
```

The framework is built on three pillars:

1. **LangGraph** provides the durable execution runtime — streaming, persistence, and checkpointing.
2. **Middleware** is how DeepAgents extends behavior: each capability (filesystem, memory, skills, subagents) is a middleware component that can be configured, replaced, or removed.
3. **Convention over configuration** — project structure (AGENTS.md, skills/, subagents/, tools.json) defines agent behavior without custom code.

---

## 3. Installation

```bash
# Using pip
pip install deepagents

# Using uv (recommended)
uv add deepagents

# With optional JavaScript sandbox support
uv add "deepagents[quickjs]"
```

**Requirements:** Python ≥ 3.11, < 4.0

Install a model provider:

```bash
# OpenAI
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# Google
pip install langchain-google-genai

# For local models (Ollama, vLLM, etc.)
pip install langchain-ollama
```

---

## 4. Quickstart

The minimal example needs only three parameters:

```python
from deepagents import create_deep_agent

# Define a custom tool (any Python function decorated with @tool)
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22°C."

# Create the agent
agent = create_deep_agent(
    model="openai:gpt-4o",          # any LLM with tool calling
    tools=[get_weather],             # your custom tools
    system_prompt="You are a helpful research assistant.",
)

# Run it
result = agent.invoke({"messages": "What is the weather in Paris?"})
print(result["messages"][-1].content)
```

With just those three parameters you automatically get:

- A built-in planner (`write_todos`)
- File system read/write tools
- Automatic context compression
- Subagent delegation via the `task` tool
- Cross-session memory via `AGENTS.md`

---

## 5. Core Capabilities

### 5.1 Planning & Task Decomposition

DeepAgents ships with a `write_todos` tool that the agent uses automatically before executing complex tasks. Unlike a basic ReAct agent that simply reacts to each step, a deep agent:

1. Reads the objective
2. Calls `write_todos` to create a structured plan
3. Executes steps one at a time (or in parallel via subagents)
4. Updates the todo list as it learns more
5. Marks items complete and adapts if the plan needs to change

```
User: "Research the top 5 Python web frameworks and write a comparison report"

Agent (internal):
  write_todos([
    "Search for popular Python web frameworks",
    "Gather performance benchmarks for each",
    "Collect community/ecosystem statistics",
    "Write comparison table",
    "Write executive summary",
  ])
  → Executes each step, delegating heavy searches to subagents
```

### 5.2 Filesystem

The virtual filesystem is the backbone of DeepAgents. It is used internally by skills, memory, code execution, and context management — and exposed as tools to the agent.

**Built-in filesystem tools:**

| Tool | Description |
|---|---|
| `read_file` | Read a file's content |
| `write_file` | Create or overwrite a file |
| `edit_file` | Apply targeted edits to a file |
| `list_dir` | List directory contents |
| `find_files` | Search files by pattern |

**Pluggable backends:**

```python
from deepagents import create_deep_agent
from deepagents.filesystem import LocalFilesystemBackend, InMemoryFilesystemBackend

# Local disk (default)
agent = create_deep_agent(
    model="openai:gpt-4o",
    filesystem_backend=LocalFilesystemBackend(root="/tmp/agent_workspace"),
)

# In-memory (ephemeral, good for testing)
agent = create_deep_agent(
    model="openai:gpt-4o",
    filesystem_backend=InMemoryFilesystemBackend(),
)
```

Available backends: **local disk**, **in-memory state**, **LangGraph store**, **composite routing**, or a **custom backend** with granular read/write permission rules.

### 5.3 Context Management

Long-running agents inevitably fill up the context window. DeepAgents handles this automatically through four mechanisms:

| Layer | What it does |
|---|---|
| **Thread summarization** | Compresses conversation history when tokens grow large |
| **Tool output offloading** | Saves large tool results to disk; passes a reference instead |
| **Subagent isolation** | Subtasks run in isolated context windows; only the result comes back |
| **Prompt caching** | Static prompt sections are cache-eligible on supported models |

This means the agent can run for hours on complex tasks without hitting context limits — a key differentiator for "deep" agent work.

### 5.4 Persistent Memory (AGENTS.md)

Memory in DeepAgents follows the `AGENTS.md` convention (similar to how `CLAUDE.md` works). These files are loaded at agent startup and always present in the system prompt.

```markdown
# AGENTS.md

## Preferences
- Always write Python code with type hints.
- Use `uv` for package management, not pip.
- Prefer async code where possible.

## Project Context
This project is a FastAPI backend for an e-commerce platform.
Database: PostgreSQL 16. ORM: SQLAlchemy 2.0.

## Known Issues
- The `orders` table has a performance problem with large date ranges.
  Use `EXPLAIN ANALYZE` before writing new queries.
```

Memory backends are pluggable: `StateBackend` (in-memory per session), `StoreBackend` (persistent across sessions via LangGraph store), or `FilesystemBackend`.

### 5.5 Skills (SKILL.md)

Skills implement **progressive disclosure** — instead of loading every tool and instruction upfront, you define specialized capabilities in `SKILL.md` files. The agent only sees a name and description; the full instructions are loaded on demand.

```markdown
---
name: data-analysis
description: Advanced data analysis with pandas and matplotlib
version: 1.0.0
---

# Data Analysis Skill

When this skill is active:
1. Always start with `df.info()` and `df.describe()` to understand the data.
2. Check for missing values with `df.isnull().sum()`.
3. Use matplotlib for visualizations; save plots to `./output/`.
4. Write a summary of findings to `./output/summary.md`.

## Tools Available
- pandas, numpy, matplotlib, seaborn
- scipy for statistical tests
```

Skills use the [agentskills.io](https://agentskills.io) spec format (YAML frontmatter + markdown body). They reduce prompt bloat and keep the agent focused.

### 5.6 Subagents (Inline)

Inline subagents let the main agent delegate subtasks to child agents that run in isolated context windows. The main agent blocks until the subagent completes and receives only the final result.

```python
from deepagents import create_deep_agent

# Define a specialized subagent
research_agent = create_deep_agent(
    model="openai:gpt-4o",
    system_prompt="You are a specialist in web research. Find accurate, cited information.",
    tools=[web_search_tool],
)

# Main agent can spawn it
main_agent = create_deep_agent(
    model="openai:gpt-4o",
    subagents={"researcher": research_agent},
    system_prompt="You are a project manager. Delegate research to the researcher subagent.",
)
```

The `task` tool is automatically added to the main agent, allowing it to write:

```
task(agent="researcher", instructions="Find the top 3 Python ML frameworks by GitHub stars as of 2026")
```

**When to use inline subagents:**
- Tasks that are fast and predictable
- When you need the result before proceeding
- Simpler operational model

### 5.7 Async Subagents (Background)

Introduced in **v0.5**, async subagents run independently on a remote Agent Protocol server. The main agent gets a task ID immediately and can continue working on other things.

```python
from deepagents import create_deep_agent
from deepagents.subagents import AsyncSubagentConfig

agent = create_deep_agent(
    model="openai:gpt-4o",
    async_subagents=[
        AsyncSubagentConfig(
            name="deep-researcher",
            url="https://my-agent-server.example.com",
            description="Long-running research agent for multi-hour tasks",
        )
    ],
)
```

**Five new tools for async task management:**

| Tool | Description |
|---|---|
| `start_async_task` | Start a background task, returns task ID |
| `check_async_task` | Poll the status and partial results |
| `update_async_task` | Send new instructions mid-execution |
| `cancel_async_task` | Cancel a running task |
| `list_async_tasks` | List all background tasks and statuses |

**When to use async subagents:**
- Tasks that take minutes to hours (deep research, large-scale code analysis)
- When you need concurrency across multiple independent workstreams
- When you need to steer or cancel in-progress work
- Multi-step data pipelines

**Comparison: Inline vs Async Subagents**

| Aspect | Inline | Async |
|---|---|---|
| Blocks main agent | Yes | No |
| Concurrency | Sequential | Parallel |
| Steering mid-task | No | Yes (`update_async_task`) |
| Operational complexity | Low | Higher (job tracking) |
| Best for | Fast, predictable tasks | Long-running tasks |

### 5.8 Human-in-the-Loop

DeepAgents supports interrupting the agent to request human approval before tool calls execute.

```python
from deepagents import create_deep_agent
from deepagents.hitl import ApprovalConfig

agent = create_deep_agent(
    model="openai:gpt-4o",
    approval=ApprovalConfig(
        tools=["write_file", "shell"],   # require approval for these tools
    ),
)
```

When an approval-gated tool is called, the agent pauses and waits. A human can:
- **Approve** — let the tool call proceed as-is
- **Edit** — modify the arguments before execution
- **Reject** — block the call (the agent adapts its plan)

For async subagents, `update_async_task` effectively provides human-in-the-loop steering mid-execution — the previous run is interrupted and the subagent restarts with updated instructions plus the full prior conversation history.

### 5.9 Shell Execution

The agent can run shell commands inside a configurable sandbox:

```python
from deepagents import create_deep_agent
from deepagents.shell import SandboxConfig

agent = create_deep_agent(
    model="openai:gpt-4o",
    shell=SandboxConfig(
        allowed_commands=["python", "pip", "git", "npm"],
        working_dir="/tmp/workspace",
    ),
)
```

The optional `quickjs` extra adds a JavaScript sandbox for safe client-side script execution.

---

## 6. Middleware System

Every capability in DeepAgents is implemented as **middleware**. This is the extensibility mechanism that lets you add, remove, or replace any behavior without forking the library.

```python
from deepagents import AgentMiddleware, MiddlewareContext

class MyLoggingMiddleware(AgentMiddleware):
    async def on_tool_call(self, ctx: MiddlewareContext, tool_name: str, args: dict):
        print(f"[LOG] Tool call: {tool_name}({args})")
        return await ctx.next(tool_name, args)

agent = create_deep_agent(
    model="openai:gpt-4o",
    middleware=[MyLoggingMiddleware()],
)
```

**What middleware can do:**
- Inject additional tools into the agent
- Extend the agent's state schema
- Modify the system prompt dynamically
- Intercept and transform model calls
- Intercept and transform tool calls
- Implement custom approval flows

**Built-in middleware components** (all replaceable):

| Middleware | Responsibility |
|---|---|
| `FilesystemMiddleware` | Read/write/edit/find file tools |
| `MemoryMiddleware` | Load AGENTS.md into state at startup |
| `SkillsMiddleware` | Progressive skill loading on demand |
| `SubagentMiddleware` | Inline subagent delegation via `task` |
| `AsyncSubagentMiddleware` | Background task management (v0.5+) |
| `ContextMiddleware` | Thread summarization & output offloading |
| `ApprovalMiddleware` | Human-in-the-loop tool approval |
| `ShellMiddleware` | Sandboxed shell command execution |

---

## 7. Model Support

DeepAgents is **model-agnostic** — it works with any LLM that supports tool calling.

**Model string format:** `provider:model-name`

Install the OpenAI provider:

```bash
pip install langchain-openai
export OPENAI_API_KEY="sk-..."
```

```python
from deepagents import create_deep_agent

# OpenAI — recommended default
create_deep_agent(model="openai:gpt-4o")          # best balance of speed & capability
create_deep_agent(model="openai:gpt-4o-mini")     # cheaper, faster, lighter tasks
create_deep_agent(model="openai:gpt-4.1")         # latest flagship
create_deep_agent(model="openai:o3")              # reasoning-optimized for complex planning

# Other frontier APIs (also supported)
create_deep_agent(model="anthropic:claude-opus-4-8")
create_deep_agent(model="google:gemini-2.5-pro")

# Open-weight via hosted providers
create_deep_agent(model="baseten:llama-3.3-70b")
create_deep_agent(model="fireworks:qwen2.5-72b")

# Self-hosted / local
create_deep_agent(model="ollama:llama3.3")
create_deep_agent(model="vllm:mistral-7b")
```

For long-horizon tasks, **`openai:gpt-4o`** or **`openai:gpt-4.1`** are the recommended defaults — they offer large context windows, reliable tool calling, and strong planning capability.

---

## 8. Advanced Configuration

### Project Structure Convention

```
my-agent/
├── AGENTS.md          ← persistent memory / instructions loaded at startup
├── skills/
│   ├── data-analysis.md  ← SKILL.md files for on-demand capabilities
│   ├── code-review.md
│   └── web-research.md
├── subagents/
│   ├── researcher.py     ← specialized subagent definitions
│   └── coder.py
├── tools.json            ← MCP server configuration
└── main.py               ← agent entry point
```

### MCP Server Integration

DeepAgents supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers as tool sources:

```python
from deepagents import create_deep_agent
from deepagents.mcp import MCPServerConfig

agent = create_deep_agent(
    model="openai:gpt-4o",
    mcp_servers=[
        MCPServerConfig(
            name="filesystem-server",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        ),
        MCPServerConfig(
            name="github-server",
            command=["npx", "-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "your-token"},
        ),
    ],
)
```

Or via `tools.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    }
  }
}
```

### Streaming

```python
async for event in agent.astream_events(
    {"messages": "Research AI safety papers from 2025-2026"},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n[Tool: {event['name']}]")
```

---

## 9. Deep Research Example

The canonical example in the DeepAgents repository is `deep_research` — a multi-agent system that performs comprehensive research on any topic.

```python
from deepagents import create_deep_agent
from langchain_tavily import TavilySearch

# Specialized research subagent
search_agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[TavilySearch(max_results=10)],
    system_prompt="""You are a research specialist. 
    Given a research question, find comprehensive, cited information.
    Always verify facts from multiple sources.
    Save your findings to a file before returning.""",
)

# Main orchestrator agent
orchestrator = create_deep_agent(
    model="openai:gpt-4o",
    subagents={"researcher": search_agent},
    system_prompt="""You are a research director managing a team of researchers.
    
    For any research task:
    1. Use write_todos to create a detailed research plan
    2. Break the topic into independent research questions
    3. Delegate each question to the researcher subagent in parallel
    4. Synthesize results into a structured report
    5. Save the final report to report.md""",
)

result = orchestrator.invoke({
    "messages": "Write a comprehensive report on the state of AI agent frameworks in 2026"
})
```

**What happens internally:**

```
Orchestrator:
  write_todos([
    "Research LangChain/DeepAgents capabilities",
    "Research CrewAI capabilities",
    "Research AutoGen capabilities",
    "Research LlamaIndex Workflows",
    "Gather performance benchmarks",
    "Compile comparison matrix",
    "Write executive summary",
  ])
  
  → task(agent="researcher", instructions="Research LangChain/DeepAgents...")
  → task(agent="researcher", instructions="Research CrewAI...")
  → task(agent="researcher", instructions="Research AutoGen...")
  [results come back from subagents]
  
  → write_file("report.md", <synthesized content>)
```

---

## 10. Production & Observability

### Managed Deep Agents

LangChain offers a **Managed Deep Agents** service — a hosted platform that handles the operational complexity of running production agents, including:
- Persistent storage for filesystem and memory backends
- Async subagent execution infrastructure
- Auto-scaling

### Checkpointing & Resuming

Built on LangGraph, every agent run can be checkpointed and resumed:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")

agent = create_deep_agent(
    model="openai:gpt-4o",
    checkpointer=checkpointer,
)

# Start a run with a thread ID
config = {"configurable": {"thread_id": "research-task-001"}}
agent.invoke({"messages": "Start the research..."}, config=config)

# Resume later (after interruption)
agent.invoke({"messages": "Continue"}, config=config)
```

---

## 11. Guiding Principles

DeepAgents is designed around three principles:

1. **Opinionated** — Defaults are carefully tuned for long-horizon, multi-step work. You don't have to configure everything to get a capable agent.

2. **Extensible** — Every component (filesystem backend, middleware, memory backend, subagent) can be overridden or replaced. You never have to fork the library.

3. **Model-agnostic** — Any LLM with tool calling support works. No vendor lock-in.

---

## 12. Version History & Key Milestones

| Version | Date | Key Changes |
|---|---|---|
| **0.1** | 2025 | Initial release: `create_deep_agent`, planning, filesystem, context management |
| **0.3** | Late 2025 | Skills (SKILL.md), improved memory, MCP server integration |
| **0.4** | Early 2026 | Inline subagents with isolated contexts, `task` tool |
| **0.5** | April 2026 | **Async subagents** — background execution via Agent Protocol; 5 new async task tools |
| **0.6** | May 2026 | Managed Deep Agents hosted service; improved observability |
| **0.6.11** | June 18, 2026 | Current stable release |

---

## 13. Resources

- **Official Docs:** [docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview)
- **GitHub Repository:** [github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- **PyPI Package:** [pypi.org/project/deepagents](https://pypi.org/project/deepagents/)
- **API Reference:** [reference.langchain.com/python/deepagents](https://reference.langchain.com/python/deepagents)
- **LangChain Deep Agents Product Page:** [langchain.com/deep-agents](https://www.langchain.com/deep-agents)
- **Blog — v0.5 Release:** [langchain.com/blog/deep-agents-v0-5](https://www.langchain.com/blog/deep-agents-v0-5)
- **Blog — Managed Deep Agents:** [langchain.com/blog/introducing-managed-deep-agents](https://www.langchain.com/blog/introducing-managed-deep-agents)
- **DeepWiki:** [deepwiki.com/langchain-ai/deepagents](https://deepwiki.com/langchain-ai/deepagents)
- **DataCamp Tutorial:** [datacamp.com/tutorial/deep-agents](https://www.datacamp.com/tutorial/deep-agents)
- **Medium — Building Deep Agents (Michiel Horstman):** [michielh.medium.com](https://michielh.medium.com/deep-agents-in-langchain-building-the-next-generation-of-autonomous-ai-systems-with-langgraph-3787b67e1805)
- **Medium — SKILL.md Deep Dive (A B Vijay Kumar):** [abvijaykumar.medium.com](https://abvijaykumar.medium.com/building-deep-agents-skill-md-with-langchain-074176c66dec)

---

*Sources consulted: LangChain official docs, GitHub repository (24.9k stars, 186 releases), PyPI package page, LangChain blog, DataCamp, Medium community tutorials, DEV Community.*
