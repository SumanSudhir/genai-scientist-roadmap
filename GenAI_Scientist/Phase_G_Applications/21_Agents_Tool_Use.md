# Topic 21: AI Agents & Tool Use

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 21 of 28
> **Scope**: Agent definition and architecture, tool use / function calling, agent patterns (ReAct, Plan-and-Execute, Reflexion), multi-agent systems, memory systems (short-term, long-term, working), code generation agents, agent evaluation, frameworks (LangChain, LlamaIndex, AutoGen, CrewAI), failure modes and limitations
> **Why this matters**: Agents represent the frontier of applied AI — models that don't just answer questions but *take actions* in the world. Every major AI lab is investing in agents (OpenAI's function calling, Anthropic's tool use, Google's Gemini actions). Interviewers will ask you to design agent systems, reason about failure modes, and explain the theoretical foundations of agent architectures. This is where NLP meets systems engineering.
> **Previous**: [Topic 20: RAG](20_RAG.md)
> **Next**: [Topic 22: Multimodal AI](22_Multimodal_AI.md)

---

## Table of Contents

1. [What Is an AI Agent?](#1-what-is-an-ai-agent)
2. [Tool Use & Function Calling](#2-tool-use--function-calling)
3. [ReAct — Reasoning + Acting](#3-react--reasoning--acting)
4. [Plan-and-Execute Agents](#4-plan-and-execute-agents)
5. [Reflexion — Self-Reflective Agents](#5-reflexion--self-reflective-agents)
6. [Memory Systems for Agents](#6-memory-systems-for-agents)
7. [Multi-Agent Systems](#7-multi-agent-systems)
8. [Code Generation Agents](#8-code-generation-agents)
9. [Agent Evaluation](#9-agent-evaluation)
10. [Frameworks & Implementation](#10-frameworks--implementation)
11. [Failure Modes & Limitations](#11-failure-modes--limitations)
12. [The Future of Agents (2025-2026)](#12-the-future-of-agents-2025-2026)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. What Is an AI Agent?

### 1.1 Definition

An **AI agent** is a system that uses an LLM as its reasoning core to **perceive** its environment, **decide** what actions to take, and **execute** those actions to achieve a goal — in a loop.

$$
\text{Agent} = \text{LLM (reasoning)} + \text{Tools (actions)} + \text{Memory (state)} + \text{Loop (autonomy)}
$$

The critical distinction from a simple chatbot: **agents act, not just respond**. A chatbot answers "What's the weather?" with text. An agent calls a weather API, reads the result, and decides whether to also check tomorrow's forecast based on the user's intent.

### 1.2 The Agent Loop

Every agent follows some variant of this loop:

```
┌──────────────────────────────────────────┐
│                                          │
│    ┌─────────┐                           │
│    │ Observe │ ◄── Environment state,    │
│    └────┬────┘     tool results,         │
│         │          user input            │
│         ▼                                │
│    ┌─────────┐                           │
│    │  Think  │ ◄── LLM reasoning         │
│    └────┬────┘     (decide next action)  │
│         │                                │
│         ▼                                │
│    ┌─────────┐                           │
│    │   Act   │ ──► Call tools, generate  │
│    └────┬────┘     text, write code      │
│         │                                │
│         ▼                                │
│    ┌─────────┐                           │
│    │Evaluate │ ──► Is the goal achieved? │
│    └────┬────┘                           │
│         │                                │
│    No ──┘     Yes ──► Return result      │
│                                          │
└──────────────────────────────────────────┘
```

### 1.3 Agent vs Pipeline vs Chatbot

| Aspect | Chatbot | Pipeline (RAG) | Agent |
|--------|---------|---------------|-------|
| Decision-making | None — responds to each message | Minimal — fixed retrieval flow | **Dynamic — decides what to do next** |
| Tool use | None | Fixed (search → generate) | **Flexible — chooses tools based on context** |
| Multi-step | Single turn | Fixed steps | **Variable-length reasoning chains** |
| Autonomy | None | None | **High — pursues goals independently** |
| Error recovery | None | None | **Can retry, reflect, try alternatives** |
| Example | "The capital of France is Paris" | Retrieve docs → Generate answer | Search API → Parse results → Calculate → Present |

### 1.4 Why Now? What Changed

Agents require three capabilities that only recently converged:

1. **Reliable instruction following**: The LLM must consistently understand and follow complex instructions. Models before GPT-3.5-turbo/GPT-4 were too unreliable.

2. **Structured output**: Function calling / tool use requires the LLM to generate precise JSON or structured formats. This emerged with fine-tuned tool-use models (2023+).

3. **Long context + strong reasoning**: Agent loops can accumulate thousands of tokens of history. Models needed both large context windows and strong reasoning to maintain coherent multi-step plans.

---

## 2. Tool Use & Function Calling

### 2.1 What Is Tool Use?

Tool use enables an LLM to **invoke external functions** during generation. Instead of generating a text response, the model generates a **structured function call** that the system executes, returning the result to the model.

```
User: "What's the weather in Paris right now?"

Without tools:
  LLM: "I don't have real-time data, but Paris typically..."  (hallucination risk)

With tools:
  LLM: [calls get_weather(city="Paris")]
  System: executes API call → {"temp": 18, "condition": "cloudy"}
  LLM: "It's currently 18°C and cloudy in Paris."  (grounded in real data)
```

### 2.2 How Function Calling Works

The model is provided with **tool definitions** (schema) and can generate **tool calls** as part of its response:

**Step 1: Define tools**

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

**Step 2: Model generates tool call**

```json
{
  "tool_call": {
    "name": "get_weather",
    "arguments": {"city": "Paris", "units": "celsius"}
  }
}
```

**Step 3: System executes and returns result**

```json
{
  "tool_result": {
    "temperature": 18,
    "condition": "cloudy",
    "humidity": 72
  }
}
```

**Step 4: Model generates final response using the result**

### 2.3 How LLMs Learn to Use Tools

Tool use is taught through **fine-tuning on tool-use data**:

1. **Training data**: Thousands of examples of (instruction, tool definition, correct tool call, tool result, final answer) tuples
2. **Structured generation**: The model learns to output valid JSON matching the tool schema
3. **Decision making**: The model learns when to call a tool vs. answer directly, and which tool to call

Some models (GPT-4, Claude, Gemini) have tool use built into their training. Open-source models (Llama, Mistral) can be fine-tuned for tool use with datasets like ToolBench, Gorilla, or API-Bank.

### 2.4 Parallel and Sequential Tool Calls

**Parallel**: Call multiple independent tools simultaneously

```
User: "Compare weather in Paris and Tokyo"

Tool calls (parallel):
  1. get_weather(city="Paris")
  2. get_weather(city="Tokyo")

→ Both execute simultaneously, results returned together
```

**Sequential**: Each tool call depends on the previous result

```
User: "Find the CEO of the company with the highest stock price today"

Step 1: get_top_stocks() → ["AAPL", "MSFT", "NVDA"]
Step 2: get_stock_price("AAPL") → $185.20  (highest)
Step 3: get_company_info("AAPL") → {"ceo": "Tim Cook"}
```

### 2.5 Tool Use Challenges

| Challenge | Description | Mitigation |
|-----------|------------|------------|
| **Schema hallucination** | Model invents parameters not in the schema | Strict schema validation, constrained generation |
| **Wrong tool selection** | Calls search when it should call calculator | Better tool descriptions, few-shot examples |
| **Argument errors** | Passes wrong types or values | JSON schema validation, retry with error message |
| **Unnecessary calls** | Uses tools when it already knows the answer | System prompt: "Only use tools when needed" |
| **Infinite loops** | Keeps calling tools without converging | Max iteration limits, loop detection |

---

## 3. ReAct — Reasoning + Acting

### 3.1 The Framework

ReAct (Yao et al., 2023) is the foundational agent pattern. It interleaves **reasoning** (thinking out loud) with **acting** (tool calls) in an alternating sequence:

$$
\text{Thought}_1 \to \text{Action}_1 \to \text{Observation}_1 \to \text{Thought}_2 \to \text{Action}_2 \to \cdots
$$

### 3.2 How It Works

```
Question: "What is the elevation of the birth city of the inventor of the telephone?"

Thought 1: I need to find who invented the telephone.
Action 1: search("inventor of the telephone")
Observation 1: Alexander Graham Bell invented the telephone.

Thought 2: Now I need to find where Alexander Graham Bell was born.
Action 2: search("Alexander Graham Bell birthplace")
Observation 2: Alexander Graham Bell was born in Edinburgh, Scotland.

Thought 3: Now I need the elevation of Edinburgh.
Action 3: search("Edinburgh Scotland elevation")
Observation 3: Edinburgh has an elevation of approximately 47 meters.

Thought 4: I now have all the information I need.
Action 4: finish("The elevation of Edinburgh, the birth city of Alexander
           Graham Bell (inventor of the telephone), is approximately 47 meters.")
```

### 3.3 Why Reasoning Helps

The explicit "Thought" steps are critical:

1. **Decomposition**: Complex queries are broken into manageable sub-problems
2. **Planning**: The model articulates what information it needs before acting
3. **Integration**: After each observation, the model reflects on what it learned and what's still needed
4. **Error recovery**: If an action returns unexpected results, the model can reason about alternatives

**Without reasoning (Act-only)**:
```
Action 1: search("elevation birth city inventor telephone")
Observation 1: No relevant results found.  (query too complex)
→ Agent stuck or gives wrong answer
```

**With reasoning (ReAct)**:
The model decomposes the question into three clear sub-queries, each of which is simple enough for the search tool.

### 3.4 ReAct vs Chain-of-Thought

| Aspect | Chain-of-Thought (CoT) | ReAct |
|--------|----------------------|-------|
| Reasoning | Yes (internal) | Yes (explicit) |
| Actions/Tools | No | **Yes** |
| Grounding | Parametric knowledge only | **External information via tools** |
| Hallucination risk | High (reasoning from memory) | **Lower (verified by tool results)** |
| Multi-step | Yes (in text) | Yes (interleaved with actions) |

ReAct = CoT + tool use. The reasoning is "grounded" by real observations from the environment.

### 3.5 Limitations of ReAct

1. **Myopic planning**: Each thought only considers the immediate next action. No global plan for the entire task.
2. **Context accumulation**: Each step adds to the context window. Long chains can exhaust the context or cause "lost in the middle" effects.
3. **Greedy search**: ReAct commits to each action — no backtracking if a step leads to a dead end.
4. **Dependent on tool quality**: The agent is only as good as its tools. A bad search API produces bad observations.

---

## 4. Plan-and-Execute Agents

### 4.1 The Idea

Plan-and-Execute addresses ReAct's myopic planning by separating the process into two phases:

1. **Plan**: Generate a complete plan for the entire task upfront
2. **Execute**: Execute each step of the plan, potentially re-planning if things change

$$
\text{Plan}(q) = [s_1, s_2, \ldots, s_n] \quad \xrightarrow{\text{Execute}} \quad s_1 \to r_1, \; s_2 \to r_2, \; \ldots
$$

### 4.2 Architecture

```
┌─────────────────────────────────────────────┐
│                  PLANNER                     │
│  (LLM generates high-level plan)            │
│                                             │
│  1. Search for X                            │
│  2. Calculate Y using result of step 1      │
│  3. Compare with Z                          │
│  4. Synthesize final answer                 │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│                 EXECUTOR                     │
│  (LLM/Agent executes each step)             │
│                                             │
│  Step 1: [tool call] → result               │
│  Step 2: [tool call with step 1 result] →   │
│  Step 3: ...                                │
│                                             │
│  If step fails → return to PLANNER          │
│                  for re-planning             │
└─────────────────────────────────────────────┘
```

### 4.3 Plan-and-Execute vs ReAct

| Aspect | ReAct | Plan-and-Execute |
|--------|-------|-----------------|
| Planning | **Step-by-step** (decide each action after previous result) | **Upfront** (plan all steps before executing) |
| Adaptability | High (adjusts after every step) | Lower (follows the plan, re-plans on failure) |
| Efficiency | May wander (no global view) | More directed (plan guides execution) |
| Complex tasks | Struggles with multi-constraint problems | Better at tasks requiring coordination |
| Token usage | Accumulates all history | Planner sees compact plan; executor sees local context |

### 4.4 When to Use Each

- **ReAct**: Simple multi-step tasks, exploration, tasks where the next step depends heavily on the previous result
- **Plan-and-Execute**: Complex tasks with multiple constraints, tasks where steps are largely independent, tasks requiring a clear structure

### 4.5 Adaptive Planning

Modern systems combine both: start with a plan, execute steps using ReAct-style reasoning, and re-plan when observations diverge from expectations:

```
Plan: [Step 1, Step 2, Step 3, Step 4]
Execute Step 1 → success
Execute Step 2 → unexpected result
→ Re-plan: [New Step 2b, Step 3', Step 4']
Execute Step 2b → success
...
```

---

## 5. Reflexion — Self-Reflective Agents

### 5.1 The Key Insight

Reflexion (Shinn et al., 2023) adds **self-reflection** to the agent loop. After an attempt fails (or partially succeeds), the agent **reflects on what went wrong** and uses that reflection to improve its next attempt.

$$
\text{Attempt}_1 \to \text{Evaluate} \to \text{Reflect} \to \text{Attempt}_2 \to \cdots
$$

### 5.2 The Three Components

**1. Actor**: The agent that performs actions (ReAct-style)

**2. Evaluator**: Determines whether the task was completed successfully. Can be:
- External (unit tests for code, ground truth for QA)
- Internal (LLM self-evaluation: "Was this answer correct and complete?")
- Heuristic (did the agent converge? Did it use too many steps?)

**3. Self-Reflection**: When the evaluator detects failure, the LLM generates a reflection:

```
Attempt 1 result: Code throws IndexError on line 12

Reflection: "The error occurred because I assumed the list would always
have at least 3 elements. I need to add a bounds check before accessing
index 2. I should also handle the empty list case."

Attempt 2: [fixes the code based on reflection]
```

### 5.3 The Reflexion Loop

```
┌────────────────────────────────────────────┐
│                                            │
│  Task ──► Actor (attempt) ──► Result       │
│                                  │         │
│                           Evaluator        │
│                           /        \       │
│                      Success      Failure  │
│                        │            │      │
│                     Return     Reflect     │
│                                  │         │
│                           Store in         │
│                           memory           │
│                                  │         │
│                           Next attempt     │
│                           (with reflection │
│                            in context)     │
│                                            │
└────────────────────────────────────────────┘
```

### 5.4 Why Reflection Works

1. **Learning from mistakes**: Unlike ReAct which has no mechanism to improve after failure, Reflexion explicitly identifies error patterns and corrects them.

2. **Persistent memory**: Reflections are stored and included in subsequent attempts, acting as "episodic memory" of what went wrong. The agent doesn't repeat the same mistakes.

3. **More targeted fixes**: Instead of blindly retrying, the agent has a specific diagnosis of what to change. This is much more efficient than random exploration.

### 5.5 Empirical Results

On HumanEval (code generation):
- Standard GPT-4: ~67% pass@1
- GPT-4 + Reflexion (3 attempts): ~91% pass@1

The improvement is dramatic because code has a clear evaluator (unit tests) and reflection can precisely identify bugs.

### 5.6 Limitations

1. **Requires good evaluation**: Reflexion is only as good as the evaluator. For open-ended tasks (creative writing, strategy), evaluation is subjective and self-reflection may not converge.
2. **Cost**: Multiple attempts = multiple LLM calls. 3 attempts × reflection = ~4x the cost of a single attempt.
3. **Context growth**: Each attempt + reflection adds to the context. Deep reflection chains may exhaust the context window.

---

## 6. Memory Systems for Agents

### 6.1 Why Memory Matters

Without memory, agents are stateless — each interaction starts from scratch. Memory enables:
- Maintaining context across a long conversation
- Remembering past tool results and decisions
- Accumulating knowledge over time
- Avoiding repeated work

### 6.2 Three Types of Memory

#### Short-Term Memory (Working Context)

The current conversation history and recent tool results, held in the LLM's context window.

| Aspect | Details |
|--------|---------|
| Storage | LLM context window |
| Capacity | 4K-128K tokens (model-dependent) |
| Persistence | Current session only |
| Access | Automatic (part of the prompt) |

**Challenge**: Context windows are finite. For long agent sessions, older messages must be summarized or evicted.

**Strategies**:
- **Sliding window**: Keep only the last $K$ messages
- **Summarization**: Periodically summarize older messages into a compact form
- **Importance-based retention**: Keep messages tagged as important; evict routine ones

#### Long-Term Memory (Persistent Knowledge)

Information that persists across sessions, stored externally:

| Aspect | Details |
|--------|---------|
| Storage | Vector database, key-value store, knowledge graph |
| Capacity | Unlimited (external storage) |
| Persistence | Across sessions |
| Access | Retrieval (RAG-style search) |

**Implementation**: Save important facts, preferences, and past interactions to a vector database. Before each response, retrieve relevant memories:

```
User: "Schedule a meeting with Sarah"

Agent retrieves from long-term memory:
- "Sarah prefers morning meetings (10am-12pm)"
- "Sarah's calendar is managed through Google Calendar"
- "Previous meeting with Sarah was about Q3 planning"

Agent: "I'll schedule a morning meeting with Sarah through
Google Calendar. Shall I book it for Q3 planning follow-up?"
```

#### Working Memory (Scratchpad)

A structured space for the agent to store intermediate results, plans, and state:

| Aspect | Details |
|--------|---------|
| Storage | Dedicated section of the prompt or external store |
| Capacity | Limited (part of context) |
| Persistence | Current task only |
| Access | Explicit read/write by the agent |

**Example**: A coding agent's scratchpad:

```
## Working Memory
- Current file: src/utils.py
- Bug identified: race condition in line 45
- Fix approach: Add mutex lock around shared resource
- Tests to run: test_concurrent_access, test_race_condition
- Status: Fix applied, running tests...
```

### 6.3 Memory Architecture Summary

```
┌────────────────────────────────────────────────────────┐
│                    Agent                                │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Short-Term Memory (Context Window)        │  │
│  │  [System prompt] [Recent messages] [Tool results] │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                              │
│            ┌────────────┼────────────┐                 │
│            ▼            ▼            ▼                 │
│   ┌──────────────┐ ┌─────────┐ ┌──────────────┐      │
│   │  Long-Term   │ │ Working │ │  Episodic    │      │
│   │  Memory      │ │ Memory  │ │  Memory      │      │
│   │  (Vector DB) │ │(scratch)│ │ (reflections)│      │
│   └──────────────┘ └─────────┘ └──────────────┘      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 7. Multi-Agent Systems

### 7.1 Why Multiple Agents?

A single agent has limitations: one LLM context, one "personality," one set of tools. Multi-agent systems assign different roles to different agents, enabling:

1. **Specialization**: Each agent focuses on what it does best
2. **Parallelism**: Multiple agents work simultaneously
3. **Debate/verification**: Agents check each other's work
4. **Complexity management**: Break a large task into agent-sized subtasks

### 7.2 Multi-Agent Patterns

#### Pattern 1: Supervisor

A central "supervisor" agent delegates subtasks to specialized worker agents:

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │  (router +   │
                    │   orchestrator)│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Research │ │  Code    │ │  Review  │
        │  Agent   │ │  Agent   │ │  Agent   │
        └──────────┘ └──────────┘ └──────────┘
```

**How it works**: User request → Supervisor analyzes → Delegates to appropriate worker → Worker returns result → Supervisor synthesizes and delegates next step.

#### Pattern 2: Debate / Verification

Multiple agents independently solve the same problem, then debate or vote:

```
Agent A ──► Solution A ──┐
                         ├──► Debate / Merge ──► Final Answer
Agent B ──► Solution B ──┘

Or:
Agent A (Generator) ──► Draft ──► Agent B (Critic) ──► Feedback ──► Agent A (Revise) ──► ...
```

**Use case**: Code review (one agent writes, another reviews), fact-checking (multiple agents verify claims independently).

#### Pattern 3: Assembly Line (Pipeline)

Each agent handles one stage of a multi-stage process:

```
Agent 1 ──► Agent 2 ──► Agent 3 ──► Agent 4
(Research)  (Analyze)   (Write)     (Edit)
```

**Use case**: Content creation (research → outline → draft → edit → publish).

#### Pattern 4: Collaborative Swarm

Agents communicate peer-to-peer without a central coordinator:

```
Agent A ◄──► Agent B
  ▲  ▼          ▲  ▼
Agent C ◄──► Agent D
```

**Use case**: Complex problem-solving where different perspectives interact dynamically.

### 7.3 Communication Protocols

| Protocol | Description | Trade-off |
|----------|-------------|-----------|
| **Shared memory** | Agents read/write to a common state | Simple but potential conflicts |
| **Message passing** | Agents send structured messages to each other | Clean but higher latency |
| **Blackboard** | Central shared workspace all agents can see | Transparent but can be noisy |
| **Conversation** | Agents communicate in natural language | Flexible but verbose and lossy |

### 7.4 Multi-Agent Challenges

1. **Coordination overhead**: More agents = more communication = more tokens = higher cost and latency
2. **Error propagation**: One agent's mistake can cascade through the system
3. **Redundant work**: Without careful coordination, agents may duplicate effort
4. **Convergence**: Debates may not converge; agents may disagree indefinitely
5. **Debugging**: Tracing failures across multiple agents is significantly harder than debugging a single agent

### 7.5 When Multi-Agent > Single Agent

| Use Multi-Agent | Use Single Agent |
|----------------|-----------------|
| Task requires diverse expertise | Task is straightforward |
| Different sub-tasks need different tools | All tools can be given to one agent |
| Verification/review is critical | Speed is the priority |
| Sub-tasks can run in parallel | Tasks are inherently sequential |
| System needs checks and balances | Trust in single model is sufficient |

---

## 8. Code Generation Agents

### 8.1 Why Code Agents Are Special

Code generation is the most natural domain for agents because code has:

1. **Automatic evaluation**: Unit tests provide unambiguous success/failure signals
2. **Rich tool ecosystem**: File systems, compilers, debuggers, test runners, linters
3. **Structured output**: Code is formally structured — easier for models to generate than free-form text
4. **Iterative improvement**: Write → test → debug → fix is a natural agent loop

### 8.2 The Code Agent Loop

```
Task: "Write a function to merge two sorted linked lists"
        │
        ▼
   ┌──────────┐
   │  Plan    │ ── Understand requirements, design approach
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  Write   │ ── Generate code
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  Test    │ ── Run unit tests / execute code
   └────┬─────┘
        │
   Pass? ──Yes──► Done
        │
       No
        │
        ▼
   ┌──────────┐
   │  Debug   │ ── Analyze error, read traceback
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  Fix     │ ── Modify code based on diagnosis
   └────┬─────┘
        │
        └──► Back to Test
```

### 8.3 Tools for Code Agents

| Tool | Purpose |
|------|---------|
| **File read/write** | Read existing code, write new code |
| **Shell execution** | Run scripts, install dependencies, execute commands |
| **Test runner** | Run unit tests, integration tests |
| **Linter/formatter** | Check code style, find potential bugs |
| **Search (codebase)** | Find relevant functions, classes, patterns |
| **Search (web/docs)** | Look up API documentation, Stack Overflow |
| **Git operations** | Commit, branch, diff, blame |
| **Debugger** | Set breakpoints, inspect variables, step through code |

### 8.4 Notable Code Agents

| Agent | Key Feature |
|-------|------------|
| **Devin** (Cognition, 2024) | Full software engineering agent with sandboxed environment |
| **SWE-Agent** (Princeton) | Terminal + file editor interface for resolving GitHub issues |
| **OpenHands** (formerly OpenDevin) | Open-source, sandboxed code agent |
| **Aider** | CLI pair programming agent |
| **Claude Code** (Anthropic) | Terminal-based coding agent with tool use |
| **Cursor / Copilot** | IDE-integrated coding assistants with agent-like features |

### 8.5 SWE-bench: The Standard Benchmark

**SWE-bench** (Jimenez et al., 2024) evaluates code agents on real GitHub issues:

- 2,294 real issue-PR pairs from 12 popular Python repositories
- Agent must: read the issue, explore the codebase, write a patch, pass the existing tests
- **SWE-bench Lite**: 300 instance subset for faster evaluation

| Agent | SWE-bench Lite (% resolved) |
|-------|----------------------------|
| Claude 3.5 Sonnet + simple scaffolding | ~49% |
| OpenHands + Claude 3.5 Sonnet | ~53% |
| Devin | ~14% (initial) → improving |
| SWE-Agent + GPT-4 | ~18% |
| Amazon Q Developer Agent | ~38% |

Performance is improving rapidly. The gap between AI and human software engineers is narrowing for well-defined tasks.

---

## 9. Agent Evaluation

### 9.1 Why Agent Evaluation Is Hard

Agents operate in open-ended environments with:
- Variable-length execution trajectories
- Non-deterministic tool outputs
- Multiple valid approaches to the same task
- Compound errors (one wrong step invalidates everything after it)

### 9.2 Evaluation Dimensions

| Dimension | What It Measures | How to Evaluate |
|-----------|-----------------|-----------------|
| **Task completion** | Did the agent achieve the goal? | Binary success/partial score, ground truth comparison |
| **Efficiency** | How many steps / tokens / tool calls? | Count steps, measure cost |
| **Correctness** | Are intermediate steps and final answer correct? | Step-by-step verification |
| **Safety** | Did the agent avoid harmful actions? | Sandbox monitoring, policy checks |
| **Robustness** | Does it handle edge cases and errors? | Adversarial inputs, tool failures |
| **Cost** | Total LLM tokens + tool calls + time | Dollar cost per task |

### 9.3 Key Benchmarks

| Benchmark | Domain | Metric | Key Feature |
|-----------|--------|--------|-------------|
| **SWE-bench** | Software engineering | % issues resolved | Real GitHub issues |
| **WebArena** | Web browsing | % tasks completed | Realistic web tasks on live sites |
| **GAIA** | General AI assistant | % correct answers | Questions requiring tool use |
| **ToolBench** | Tool use | Pass rate | 16K+ real-world APIs |
| **AgentBench** | Multi-environment | Composite score | 8 environments (OS, DB, web, etc.) |
| **HumanEval** | Code generation | pass@k | Function-level code synthesis |
| **MATH / GSM8K** | Math reasoning | % correct | With tool use (calculator) |

### 9.4 Evaluation Best Practices

1. **Sandbox everything**: Agents execute code, call APIs, modify files. Always run in isolated environments.
2. **Measure cost alongside accuracy**: An agent that solves 90% of tasks at $5/task may be worse than one solving 85% at $0.50/task.
3. **Test failure recovery**: Intentionally introduce tool failures and see if the agent adapts.
4. **Human evaluation for open-ended tasks**: Automated metrics can't fully capture the quality of complex agent behaviors.
5. **Track trajectory, not just outcome**: Understanding *how* the agent reached its answer reveals systematic strengths and weaknesses.

---

## 10. Frameworks & Implementation

### 10.1 Framework Comparison

| Framework | Philosophy | Best For | Key Feature |
|-----------|-----------|----------|-------------|
| **LangChain** | Composable chains and agents | General-purpose agent building | Largest ecosystem, most integrations |
| **LlamaIndex** | Data-centric (connect LLMs to data) | RAG-heavy agent systems | Strong data connectors |
| **AutoGen** (Microsoft) | Multi-agent conversation | Multi-agent systems | Agent-to-agent chat protocol |
| **CrewAI** | Role-based multi-agent | Team-based task execution | Simple role/task assignment |
| **LangGraph** | Graph-based agent workflows | Complex, stateful agent flows | Cycles, state management, persistence |
| **Semantic Kernel** (Microsoft) | Enterprise AI orchestration | Enterprise integration | .NET/Python, strong Azure integration |
| **Anthropic Agent SDK** | Minimal agent scaffolding | Building with Claude models | Lightweight, focused |

### 10.2 LangGraph — The Modern Standard

LangGraph (from the LangChain team) has emerged as the leading framework for complex agents because it models agents as **state machines** (graphs):

```
         ┌─────────┐
         │  Start  │
         └────┬────┘
              │
              ▼
         ┌─────────┐
    ┌────│  Route  │────┐
    │    └─────────┘    │
    ▼                   ▼
┌────────┐        ┌─────────┐
│Research│        │  Answer  │
│  Node  │        │  Node   │
└───┬────┘        └────┬────┘
    │                  │
    ▼                  │
┌────────┐             │
│Analyze │             │
│  Node  │             │
└───┬────┘             │
    │                  │
    └──────┬───────────┘
           ▼
      ┌─────────┐
      │   End   │
      └─────────┘
```

**Key advantages**:
- **Cycles**: Supports loops (retry, reflect, iterate) — essential for agents
- **State persistence**: Save and resume agent state across sessions
- **Human-in-the-loop**: Pause execution for human approval at critical steps
- **Streaming**: Stream intermediate results to the user
- **Checkpointing**: Record every state for debugging and replay

### 10.3 Architecture Decision: Framework vs Custom

| Use Framework | Build Custom |
|--------------|-------------|
| Rapid prototyping | Highly specific requirements |
| Standard agent patterns | Extreme performance needs |
| Team needs shared abstractions | Minimal dependency footprint |
| Rich ecosystem needed | Full control over every detail |

**The practical advice**: Start with a framework (LangGraph) for prototyping. Extract to custom code only where the framework creates friction. Many production systems use frameworks for orchestration but custom code for critical paths.

---

## 11. Failure Modes & Limitations

### 11.1 The Compounding Error Problem

Each step in an agent's execution has some probability of error. Over $n$ steps:

$$
P(\text{all steps correct}) = \prod_{i=1}^{n} (1 - p_{\text{error},i}) \approx (1 - p)^n
$$

For a 10-step task with 5% error per step:

$$
P(\text{success}) = (0.95)^{10} = 0.60
$$

**40% chance of failure** — even with 95% per-step accuracy. This is why long agent chains are fragile.

```
Steps:     1     2     3     4     5     6     7     8     9    10
Success:  95%   90%   86%   81%   77%   74%   70%   66%   63%  60%
                                                                 ↑
                                             After 10 steps, only 60% succeed
```

### 11.2 Taxonomy of Agent Failures

| Failure Mode | Description | Example |
|-------------|-------------|---------|
| **Planning failure** | Agent creates a wrong or impossible plan | Plans to divide by zero, or misunderstands the task |
| **Tool selection error** | Chooses the wrong tool for the job | Uses search when calculator is needed |
| **Argument error** | Passes incorrect arguments to a tool | `search("weather in Paris")` instead of `get_weather("Paris")` |
| **Hallucinated action** | Invents a tool or action that doesn't exist | Calls `send_email()` when no email tool is available |
| **Observation misinterpretation** | Misreads or ignores a tool's output | Tool returns an error, agent treats it as success |
| **Infinite loop** | Repeats the same actions without progress | Keeps searching for the same query with the same results |
| **Goal drift** | Loses track of the original objective over many steps | Starts solving a different problem than what was asked |
| **Premature termination** | Stops before the task is complete | Returns a partial answer without checking completeness |
| **Safety violation** | Takes harmful or unauthorized actions | Deletes files, sends unintended messages, exposes data |

### 11.3 Mitigation Strategies

| Strategy | How It Helps |
|----------|-------------|
| **Max step limits** | Prevents infinite loops and runaway costs |
| **Sandboxing** | Contains damage from incorrect actions (especially code execution) |
| **Human-in-the-loop** | Require approval for high-stakes actions (delete, send, publish) |
| **Reflection / retry** | Identify and correct errors (Reflexion pattern) |
| **Guardrails** | Input/output filtering to prevent harmful actions |
| **Verification steps** | Explicitly check intermediate results before proceeding |
| **Fallback to simpler methods** | If the agent approach fails, fall back to non-agent pipeline |
| **Observation validation** | Parse and validate tool results before feeding back to the LLM |
| **Cost budgets** | Hard limits on total tokens / tool calls per task |

### 11.4 The Reliability Frontier

Current state (2025-2026):

| Task Complexity | Agent Reliability | Example |
|----------------|------------------|---------|
| 1-2 steps | ~95% | Simple QA with one tool call |
| 3-5 steps | ~75-85% | Multi-hop research, simple code tasks |
| 5-10 steps | ~50-70% | Complex code modifications, multi-step analysis |
| 10-20 steps | ~30-50% | Full software engineering tasks, complex research |
| 20+ steps | ~10-30% | End-to-end project execution |

The gap between "impressive demos" and "reliable production systems" is the central challenge in agent research.

---

## 12. The Future of Agents (2025-2026)

### 12.1 Key Trends

**1. Computer Use Agents**: Models that directly control computer interfaces (clicking, typing, navigating). Anthropic's "computer use" and OpenAI's "operator" allow agents to interact with any application through its GUI — no API required.

**2. Test-Time Compute for Agents**: Models like o1/o3 that "think longer" before acting. More reasoning = better planning = fewer errors per step. This could dramatically shift the reliability frontier.

**3. Agent-Specific Training**: Fine-tuning models specifically for agent tasks (tool use, planning, error recovery) rather than relying on general-purpose chat models.

**4. Formal Verification**: Using formal methods to verify agent plans before execution. Especially important for high-stakes domains (finance, healthcare).

**5. Agent-to-Agent Protocols**: Standardized protocols for agents to communicate, negotiate, and collaborate — analogous to how HTTP standardized web communication.

### 12.2 Open Problems

1. **Reliability at scale**: How to achieve >99% reliability for 10+ step tasks?
2. **Safe autonomy**: How to give agents more autonomy without more risk?
3. **Cost efficiency**: Current agents use 10-100x more tokens than non-agent approaches. How to reduce this?
4. **Evaluation**: How to evaluate open-ended agent capabilities beyond narrow benchmarks?
5. **Long-horizon planning**: Current agents plan well for 5-10 steps. How to plan reliably for 100+ steps?

---

## 13. Interview Questions & Answers

### Q1: Design an AI agent that can answer questions about a company's codebase.

**A**: I'll design a code QA agent with ReAct-style reasoning and specialized tools.

**Architecture**:

```
User question → Agent (LLM) → Think → Act → Observe → ... → Answer
```

**Tools available to the agent**:
1. `search_code(query)`: Semantic search over the codebase (pre-indexed with code embeddings)
2. `grep_code(pattern, file_glob)`: Exact text search (regex over files)
3. `read_file(path, start_line, end_line)`: Read specific file sections
4. `list_directory(path)`: Explore directory structure
5. `get_git_history(file_path, n)`: Recent commits affecting a file
6. `run_command(cmd)`: Execute shell commands in a sandbox (read-only access)

**Memory**:
- **Short-term**: Conversation history + tool results in context
- **Long-term**: Vector index of code chunks (functions, classes, modules) with metadata (file path, docstrings, dependencies)
- **Working**: Scratchpad for noting relevant files, function signatures, call chains discovered so far

**Example execution**:

```
User: "How does the authentication middleware work?"

Thought: I need to find the authentication middleware in the codebase.
Action: search_code("authentication middleware")
Observation: Found matches in src/middleware/auth.py (85% relevance),
             src/utils/jwt.py (72%), tests/test_auth.py (68%)

Thought: Let me read the main auth middleware file.
Action: read_file("src/middleware/auth.py", 1, 100)
Observation: [code content showing AuthMiddleware class with
              verify_token, check_permissions methods]

Thought: This uses JWT verification. Let me check the JWT utility.
Action: read_file("src/utils/jwt.py", 1, 50)
Observation: [JWT creation and verification logic]

Thought: I now understand the flow. Let me synthesize an answer.
Action: finish("The authentication middleware (src/middleware/auth.py)
        works as follows: 1. Extracts the JWT from the Authorization
        header... 2. Verifies the token using... 3. Checks permissions...")
```

**Safety**: All code execution is sandboxed and read-only. The agent cannot modify the codebase.

**Evaluation**: Measure on a curated set of codebase questions with ground-truth answers. Track: answer correctness, number of tool calls (efficiency), latency.

---

### Q2: What is the Reflexion pattern? How does self-reflection improve agent performance?

**A**: Reflexion adds a **reflect-and-retry** loop to the standard agent pattern. When an attempt fails, the agent:

1. **Evaluates** the attempt (using external feedback like test results, or self-evaluation)
2. **Reflects** on what went wrong (generates a natural language diagnosis)
3. **Retries** with the reflection as additional context

**Why it works**: Standard agents (ReAct) have no mechanism to learn from failure within a task. If step 5 fails, the agent might try the same failing approach again. Reflexion creates **episodic memory** — a record of what was tried and why it failed — that guides subsequent attempts toward different strategies.

**Concrete example** (code generation):

```
Attempt 1: Write merge_sorted_lists function
→ Test result: FAIL — IndexError on empty list input

Reflection: "My implementation assumed both lists are non-empty.
I need to handle the base cases where either list is empty
by returning the other list directly."

Attempt 2: Write merge_sorted_lists with base case handling
→ Test result: FAIL — Wrong output for lists with duplicates

Reflection: "The merge logic uses strict less-than comparison,
which skips duplicate elements. I should use less-than-or-equal
so duplicates from both lists are included."

Attempt 3: Fix comparison operator
→ Test result: PASS — All 15 test cases pass
```

**Quantitative impact**: On HumanEval, Reflexion improves GPT-4 from ~67% to ~91% pass@1. The effect is largest when:
- Clear evaluation signals exist (tests, metrics)
- Common errors are identifiable (off-by-one, edge cases)
- The fix is localized (not a fundamental approach change)

**Limitations**: Doesn't help when the fundamental approach is wrong (reflection says "try a different algorithm" but the model doesn't know a better one), or when evaluation is ambiguous (subjective quality tasks).

---

### Q3: How do you prevent an agent from taking harmful actions?

**A**: A defense-in-depth approach with multiple layers:

**Layer 1: Tool Design (Principle of Least Privilege)**
- Only give the agent tools it needs. A Q&A agent doesn't need `delete_file`.
- Make tools safe by default: `search` is read-only; `write_file` requires explicit paths.
- Implement tool-level permissions: some tools are always allowed, others require approval.

**Layer 2: Input/Output Guardrails**
- Filter agent inputs for prompt injection attempts
- Validate tool arguments against schemas before execution
- Filter agent outputs for sensitive information (PII, credentials)

**Layer 3: Human-in-the-Loop**
- Classify actions by risk level:
  - **Low risk** (search, read): Execute automatically
  - **Medium risk** (write file, API call): Log and execute, notify user
  - **High risk** (delete, send message, external API): Require explicit human approval
- Present the planned action to the user before execution

**Layer 4: Sandboxing**
- Execute code in isolated containers (Docker, sandboxed VM)
- Network restrictions: only allow approved API endpoints
- File system isolation: agent can only access designated directories
- Resource limits: CPU, memory, execution time caps

**Layer 5: Monitoring and Kill Switches**
- Real-time monitoring of agent actions
- Anomaly detection: flag unusual patterns (too many API calls, accessing unexpected files)
- Hard stop: immediately terminate the agent if safety thresholds are exceeded
- Complete audit trail: log every thought, action, and observation

**Layer 6: Post-Hoc Review**
- Automated review of completed agent sessions
- Flag sessions with unusual patterns for human review
- Feedback loop: use identified failures to improve guardrails

The key principle: **never give an agent more capability than it needs, and always monitor what it does with the capability it has**.

---

### Q4: What are the failure modes of multi-step agents? How do you mitigate them?

**A**: The fundamental problem: **errors compound**. With per-step success rate $p$, a 10-step task succeeds with probability $p^{10}$. At $p = 0.95$: only 60% success for 10 steps.

**Five major failure modes and mitigations**:

**1. Compounding errors**
- Problem: Each incorrect step corrupts all subsequent steps
- Mitigation: Verification checkpoints after each step. "Did step $n$ produce a reasonable result before proceeding to step $n+1$?" Reflexion pattern for recovery.

**2. Goal drift**
- Problem: Over many steps, the agent loses track of the original objective
- Mitigation: Include the original task in every LLM call (not just the first). Periodically re-evaluate: "Am I still working toward the original goal?"

**3. Context window exhaustion**
- Problem: After many steps, the accumulated history exceeds the context window
- Mitigation: Summarize older steps, keep only recent detailed history. Working memory for key facts. Parent prompts that compress the trajectory.

**4. Infinite loops**
- Problem: Agent repeats the same action expecting different results
- Mitigation: Max step limits, loop detection (detect if the same action has been tried before), forced action diversity ("Try a different approach").

**5. Cascading tool failures**
- Problem: An API is down, a file doesn't exist, a command fails — the agent doesn't handle the error gracefully
- Mitigation: Tool error handling (wrap every tool call in try-catch, return informative error messages), fallback tools (if primary search fails, try secondary), timeout limits per tool.

**The meta-mitigation**: Design for **graceful degradation**. When the agent detects it's struggling (too many retries, conflicting information, cost budget exceeded), it should fall back to a simpler approach or ask for human help rather than continuing to fail.

---

### Q5: Compare ReAct, Plan-and-Execute, and Reflexion. When would you choose each?

**A**:

| Aspect | ReAct | Plan-and-Execute | Reflexion |
|--------|-------|-----------------|-----------|
| **Planning** | None (step-by-step) | Upfront plan | None (but reflects on failures) |
| **Adaptability** | High (re-decides each step) | Medium (follows plan, re-plans on failure) | High (learns from each attempt) |
| **Error handling** | Weak (no explicit recovery) | Medium (re-plan on failure) | **Strong** (diagnoses and fixes errors) |
| **Token efficiency** | Medium | Higher (plan is compact) | Low (multiple full attempts) |
| **Best for** | Exploratory tasks, simple multi-step | Complex tasks with clear structure | Tasks with clear success criteria |

**Choose ReAct when**:
- The task is exploratory (you don't know the steps in advance)
- Each step strongly depends on the previous result
- The task is relatively simple (3-5 steps)
- Example: "Find information about X and summarize it"

**Choose Plan-and-Execute when**:
- The task has a clear structure that can be planned upfront
- Sub-steps are relatively independent
- The task is complex enough that a plan helps avoid getting lost
- Example: "Research competitors A, B, C, compare their pricing, and create a report"

**Choose Reflexion when**:
- The task has a clear evaluator (tests, metrics, ground truth)
- First attempts are likely to have specific, fixable errors
- The cost of multiple attempts is acceptable
- Example: "Write a function that passes these unit tests"

**In practice**: Combine them. Use Plan-and-Execute for the high-level structure, ReAct for executing each step, and Reflexion when a step fails. This gives you the benefits of all three.

---

### Q6: How do memory systems work in agents? Design a memory architecture for a customer support agent.

**A**: For a customer support agent handling ongoing relationships:

**Short-Term Memory** (conversation context):
- Current conversation messages (user + agent)
- Recent tool results (order lookups, account info)
- Implementation: LLM context window, last 20 messages + summarized older messages

**Long-Term Memory** (persistent):
- Customer profile: name, plan, history, preferences
- Past interactions: summarized previous support conversations
- Known issues: previously reported problems and resolutions
- Implementation: Vector database (Qdrant) with per-customer namespace

**Working Memory** (current task state):
- Current issue being resolved
- Steps taken so far
- Pending actions (waiting for customer response, escalation in progress)
- Implementation: Structured JSON in the system prompt, updated each turn

**Episodic Memory** (reflection):
- "Last time this customer asked about billing, they were frustrated by the process. Be extra clear and patient."
- "For refund requests, always check the return policy first — skipping this caused an error last time."
- Implementation: Retrieved from vector DB based on current context

**Architecture**:

```
Customer message
       │
       ▼
Retrieve from Long-Term Memory:
  - Customer profile → "Premium customer since 2022, prefers email"
  - Past interactions → "Had billing issue in Jan, resolved via refund"
  - Similar resolved issues → "Other customers with this error needed..."
       │
       ▼
Construct prompt:
  [System prompt + Working memory + Long-term memories + Conversation]
       │
       ▼
LLM generates response (potentially with tool calls)
       │
       ▼
Update memories:
  - Working memory: Update task state
  - Long-term: Save new interaction summary (async)
  - Episodic: If issue resolved, save resolution pattern
```

---

### Q7: What is the difference between an agent and a pipeline? Give a concrete example where each is better.

**A**: The key distinction is **dynamic decision-making**:

**Pipeline**: Fixed sequence of steps, determined at design time. Every input follows the same path.

**Agent**: Dynamic sequence of steps, determined at runtime by the LLM. Different inputs may follow completely different paths.

**Pipeline is better — customer FAQ system**:

```
User question → Embed → Search FAQ database → Re-rank → Format answer
```

Every question follows the same 4 steps. There's no decision to make — the pipeline is deterministic and efficient. Adding agent-style reasoning would increase cost and latency for no benefit. The task is well-defined and the flow is static.

**Agent is better — IT support ticket resolution**:

```
Ticket: "I can't connect to the VPN from my new laptop"

Agent thinks: This could be several issues. Let me diagnose.
→ Check: Is the VPN client installed? (calls asset management API)
→ Observation: Yes, installed yesterday
→ Check: Is the laptop on the approved device list? (calls security API)
→ Observation: No — new device not yet registered
→ Action: Register device in security system
→ Action: Ask user to restart VPN client
→ Check: Is the issue resolved? (ask user)
→ If no → Escalate to network team with diagnostic info
```

The agent dynamically chooses which checks to run based on intermediate results. Different tickets require different diagnostic paths — a fixed pipeline can't handle this variation. The agent's ability to branch, retry, and adapt is essential.

**Rule of thumb**: If you can draw the flowchart before seeing the input, use a pipeline. If the flowchart depends on the input, use an agent.

---

### Q8: How would you design a system for safe, autonomous agent execution?

**A**: A layered safety architecture:

**1. Permission Tiers**:
```
Tier 0 (Always allowed):  Read files, search, calculate
Tier 1 (Logged):          Write files, API reads
Tier 2 (Confirmed):       Send messages, modify data
Tier 3 (Human required):  Delete data, financial transactions, external comms
```

**2. Execution Sandbox**:
- Isolated container per agent session
- Network allowlist (only approved endpoints)
- File system is session-scoped (can't access other sessions)
- Resource limits (60s max per tool execution, 1GB memory)

**3. Action Validation Pipeline**:
```
Agent proposes action
       │
       ▼
Schema validation ── Is the action well-formed?
       │
       ▼
Policy check ── Is the action allowed by the permission tier?
       │
       ▼
Anomaly detection ── Is this action unusual for this context?
       │
       ▼
Execute (if all pass) or Request approval (if any flag)
```

**4. Monitoring**:
- Real-time dashboard showing active agents and their actions
- Alerts on: permission escalation attempts, repeated failures, cost threshold exceeded
- Complete audit log: every thought, action, observation, and decision

**5. Graceful degradation**:
- If the agent exceeds step limit: summarize progress and ask for human guidance
- If a tool fails repeatedly: skip and try alternative approach
- If cost budget exceeded: return best partial result with explanation

**6. Testing regime**:
- Red-team the agent: try to trick it into harmful actions via adversarial inputs
- Chaos testing: randomly fail tools to test error handling
- Regression tests: curated set of tasks the agent must handle safely

---

### Q9: Explain the compounding error problem. Why does it matter more for agents than for single-turn LLM use?

**A**: In a single-turn LLM call, there's one opportunity for error. If the model is 95% accurate, you get a correct answer 95% of the time.

In an agent executing $n$ sequential steps, each step must be correct for the final result to be correct. Assuming independence:

$$
P(\text{success}) = \prod_{i=1}^{n} P(\text{step } i \text{ correct}) = (1-p)^n
$$

| Steps | Per-step accuracy 95% | Per-step accuracy 99% |
|-------|----------------------|----------------------|
| 1 | 95.0% | 99.0% |
| 3 | 85.7% | 97.0% |
| 5 | 77.4% | 95.1% |
| 10 | 59.9% | 90.4% |
| 20 | 35.8% | 81.8% |
| 50 | 7.7% | 60.5% |

Even at 99% per-step accuracy, a 50-step task only succeeds 60% of the time. This is why:

1. **Minimizing steps is critical**: Always prefer fewer, more powerful steps over many small steps
2. **Per-step reliability must be extremely high**: Going from 95% to 99% per step doubles success rate at 20 steps
3. **Verification and recovery matter**: Checking intermediate results and correcting errors breaks the multiplicative chain
4. **Agent architectures prioritize reliability**: This is why Reflexion, human-in-the-loop, and checkpoint verification exist — they interrupt the compounding

**Comparison with single-turn**: A RAG pipeline with 3 fixed steps has this problem too, but it's manageable (0.95³ = 86%). An autonomous agent with 15+ dynamic steps faces a qualitatively different challenge.

The practical implication: **design agents for the minimum number of steps**, and invest heavily in per-step reliability through better models, better tools, and verification.

---

### Q10: What is the current state of AI agents? What works, what doesn't, and what's needed?

**A**:

**What works well (2025-2026)**:

- **Code agents** (3-10 step tasks): SWE-bench Lite resolution rates >50%. Writing functions, fixing bugs, refactoring — agents are genuinely useful. Clear evaluation (tests) enables Reflexion-style improvement.
- **RAG agents**: Multi-query retrieval, query decomposition, and agentic RAG reliably improve over naive RAG. The agent decides what to search for and when to stop.
- **Data analysis agents**: Given a dataset, agents can explore, visualize, and analyze with high reliability. Tools (pandas, SQL) are deterministic and well-structured.
- **Simple tool use** (1-3 tools, 1-3 steps): Weather lookup, calendar management, calculator — near-human reliability.

**What's fragile**:

- **Long-horizon tasks** (20+ steps): Too much error compounding. Agents lose track of goals, accumulate irrelevant context, and make cascading errors.
- **Web browsing agents**: Websites are messy, dynamic, and unpredictable. Agents fail on CAPTCHAs, pop-ups, dynamic content, login flows.
- **Multi-agent systems**: Coordination overhead, error propagation between agents, and debugging complexity limit practical value. Most production systems use single agents.
- **Open-ended tasks**: "Write a marketing strategy" — no clear success criteria, so agents can't self-evaluate or improve.

**What's needed**:

1. **Better planning**: Current agents plan myopically. We need models that can reason about 50-step plans and adapt them dynamically.
2. **Formal verification of plans**: Before executing, verify that the plan is logically sound and will achieve the goal.
3. **Cheaper reasoning**: Agents use 10-100x more tokens than non-agent approaches. Models need to be more efficient at tool-use reasoning.
4. **Standardized tool interfaces**: Like REST APIs standardized web services, we need standard interfaces for agent tools.
5. **Better evaluation**: Current benchmarks test narrow capabilities. We need evaluation frameworks that capture real-world agent utility.
6. **Safety guarantees**: Mathematical guarantees (not just guardrails) that agents won't take harmful actions, especially as autonomy increases.

The trajectory is clear: agents are moving from "impressive demos" to "reliable production tools," but we're still in the early stages of that transition. The gap is being closed by better models (o1/o3 reasoning), better frameworks (LangGraph), and better evaluation (SWE-bench, WebArena).

---

*Agents represent the frontier of applied AI — where language models become capable actors in the world. Next: [Topic 22: Multimodal AI](22_Multimodal_AI.md) — extending beyond text to images, audio, and video.*
