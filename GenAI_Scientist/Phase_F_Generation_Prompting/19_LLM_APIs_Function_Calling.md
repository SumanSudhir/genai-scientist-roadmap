# Topic 19: LLM APIs & Function Calling

> **Interview Weight**: ★★★★☆ — Production-critical knowledge. Function calling mechanics, cost estimation, model routing, and LLM gateway design are [H]-level questions at Google, Amazon, and AI startups. Every senior engineer working with LLMs is expected to know these cold.

## Table of Contents

1. [LLM API Fundamentals](#1-llm-api-fundamentals)
2. [Chat Roles & Message Structure](#2-chat-roles--message-structure)
3. [Streaming](#3-streaming)
4. [Function Calling / Tool Use](#4-function-calling--tool-use)
5. [Parallel Function Calling](#5-parallel-function-calling)
6. [Cost Estimation & Token Pricing](#6-cost-estimation--token-pricing)
7. [Provider Comparison](#7-provider-comparison)
8. [Rate Limiting, Retry & Fallback](#8-rate-limiting-retry--fallback)
9. [Semantic Caching](#9-semantic-caching)
10. [Model Routing](#10-model-routing)
11. [Hosted API vs Self-Hosted OSS Models](#11-hosted-api-vs-self-hosted-oss-models)
12. [LLM Gateway Design](#12-llm-gateway-design)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. LLM API Fundamentals

### Core Parameters

Every LLM API call has these key parameters:

| Parameter | Type | Effect |
|-----------|------|--------|
| `model` | string | Which model to use (cost/quality trade-off) |
| `messages` | list | Conversation history (system + user + assistant turns) |
| `temperature` | float 0–2 | Randomness: 0 = greedy, 1 = default, >1 = creative |
| `max_tokens` | int | Maximum output tokens to generate (cost control) |
| `top_p` | float 0–1 | Nucleus sampling threshold (alternative to temperature) |
| `stream` | bool | Return tokens as they're generated vs all at once |
| `stop` | list[str] | Stop generation when these strings appear |
| `n` | int | Number of completions to generate (1 = default) |
| `seed` | int | Make output reproducible (near-deterministic) |
| `response_format` | object | Force JSON output or specific schema |
| `tools` | list | Function definitions the model can call |
| `tool_choice` | string | "auto", "none", or force a specific tool |

### The Request-Response Cycle

```
Client                          LLM API Server
  │                                  │
  │── POST /v1/chat/completions ─────►│
  │   {model, messages, temp, ...}   │
  │                                  │ prefill (process input tokens)
  │                                  │ decode  (generate output tokens)
  │◄── 200 OK ──────────────────────│
  │   {id, choices, usage: {        │
  │     prompt_tokens: 150,         │
  │     completion_tokens: 80,      │
  │     total_tokens: 230           │
  │   }}                            │
```

**Usage object** is critical for cost tracking — always log it.

---

## 2. Chat Roles & Message Structure

### The Three Roles

```python
messages = [
    {
        "role": "system",       # Developer instructions — highest priority
        "content": "You are a concise assistant. Answer in ≤ 2 sentences."
    },
    {
        "role": "user",         # What the user said
        "content": "What is the capital of France?"
    },
    {
        "role": "assistant",    # What the model previously said (for multi-turn)
        "content": "The capital of France is Paris."
    },
    {
        "role": "user",         # Next user turn
        "content": "What is its population?"
    }
]
```

### Priority Order

```
System prompt  ─── highest priority (RLHF-trained to follow system over user)
  │
  ▼
Assistant turns ─── model's own previous outputs (trusted)
  │
  ▼
User turns ──────── lowest trust (may contain injections)
```

### Multi-Turn Context Management

The API is **stateless** — you must send the full conversation history every time. This has implications:

```
Turn 1: send [system, user_1]                         → 200 tokens
Turn 2: send [system, user_1, assistant_1, user_2]    → 350 tokens
Turn 3: send [system, user_1, ..., user_3]            → 550 tokens

Cost grows with conversation length!
```

**Strategies to control cost**:
- **Sliding window**: keep last N turns, drop oldest
- **Summarization**: periodically summarize early turns into one message
- **Selective retention**: keep system + last 5 turns + any "pinned" messages

---

## 3. Streaming

### What It Is

Instead of waiting for the full response, stream tokens as they're generated using **Server-Sent Events (SSE)**:

```
Without streaming:
  Client waits 8 seconds → gets full 200-word response at once
  User sees nothing for 8 seconds → poor UX

With streaming:
  After 200ms → "The capital..."
  After 400ms → "The capital of France..."
  After 600ms → "The capital of France is..."
  Progressive rendering → user starts reading immediately
```

### SSE Protocol

```
HTTP response header: Content-Type: text/event-stream

Streaming chunks:
  data: {"choices": [{"delta": {"content": "The"}}]}
  data: {"choices": [{"delta": {"content": " capital"}}]}
  data: {"choices": [{"delta": {"content": " of"}}]}
  ...
  data: {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
  data: [DONE]
```

### Why Streaming Matters

- **Perceived latency**: Time-to-first-token (TTFT) matters more to users than total time
- **Interruption**: User can stop generation early if the answer is already clear
- **Progressive trust**: Users can catch hallucinations mid-generation and provide correction
- **Token budget control**: Can stop generation when answer is complete (before `max_tokens`)

### Implementation Pattern

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True
)

full_response = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    full_response += delta
    print(delta, end="", flush=True)  # stream to terminal/UI
```

---

## 4. Function Calling / Tool Use

### The Core Mechanism

Function calling allows the LLM to **request execution of functions** — the model does not execute code itself, it outputs a structured call that your code executes:

```
Client → API: "What's the weather in Paris?"  +  tool definitions
API   → Client: {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
Client → [executes get_weather("Paris") → returns "22°C, sunny"]
Client → API: original messages + tool_result: "22°C, sunny"
API   → Client: "The weather in Paris is currently 22°C and sunny."
```

### Tool Definition Schema

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Use when asked about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris' or 'New York'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

### Full Round-Trip Flow

```
Step 1 — Send user query + tool definitions:
  messages = [{"role": "user", "content": "What's the weather in Paris?"}]
  response = client.chat.completions.create(model=..., messages=messages, tools=tools)

Step 2 — Model decides to call a tool:
  response.choices[0].finish_reason == "tool_calls"
  tool_call = response.choices[0].message.tool_calls[0]
  # tool_call.function.name = "get_weather"
  # tool_call.function.arguments = '{"city": "Paris", "unit": "celsius"}'

Step 3 — Execute the function:
  args = json.loads(tool_call.function.arguments)
  result = get_weather(**args)  # → "22°C, sunny"

Step 4 — Send result back to model:
  messages.append(response.choices[0].message)          # assistant's tool_call message
  messages.append({
      "role": "tool",
      "tool_call_id": tool_call.id,
      "content": result
  })
  final_response = client.chat.completions.create(model=..., messages=messages, tools=tools)

Step 5 — Model generates final answer:
  "The weather in Paris is currently 22°C and sunny."
```

### `tool_choice` Parameter

| Value | Behavior |
|-------|---------|
| `"auto"` | Model decides whether to call a tool (default) |
| `"none"` | Model must respond without calling any tool |
| `{"type": "function", "function": {"name": "X"}}` | Force the model to call function X |

---

## 5. Parallel Function Calling

### What It Is

When a query requires multiple independent tools, modern models can emit **multiple tool calls in a single response** — they're all executed in parallel:

```
User: "Compare the weather in Paris and Tokyo right now."

Without parallel calling (2 round trips):
  Round 1: model calls get_weather("Paris") → wait → get result
  Round 2: model calls get_weather("Tokyo") → wait → get result
  Latency: 2 × API call time

With parallel calling (1 round trip):
  Round 1: model emits BOTH calls simultaneously:
    tool_calls[0]: get_weather("Paris")
    tool_calls[1]: get_weather("Tokyo")
  Execute both in parallel → get both results
  Send both tool results in one message batch → model generates answer
  Latency: 1 × API call time  (≈ 2× faster)
```

### Implementation

```python
# Model returns multiple tool_calls
tool_calls = response.choices[0].message.tool_calls

# Execute all in parallel
import concurrent.futures
def execute_tool(tc):
    args = json.loads(tc.function.arguments)
    return tc.id, get_weather(**args)

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = dict(executor.map(execute_tool, tool_calls))

# Send all results back together
for tc in tool_calls:
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": results[tc.id]
    })
```

### When Parallel Calling Fires

Model uses parallel calling when:
- Multiple tools are needed for the same response
- The tools are **independent** (result of one doesn't affect the call of another)
- Examples: weather in multiple cities, stock prices for multiple tickers, parallel database lookups

---

## 6. Cost Estimation & Token Pricing

### How Pricing Works

LLM APIs charge separately for **input tokens** (your prompt) and **output tokens** (the response). Output tokens cost 2–5× more than input tokens because:
- Output is generated autoregressively (one token at a time, sequential)
- Input is processed in parallel (full prefill in one pass)

### Cost Calculation

$$
\text{Cost} = \frac{\text{input tokens}}{1{,}000{,}000} \times P_{\text{in}} + \frac{\text{output tokens}}{1{,}000{,}000} \times P_{\text{out}}
$$

### Worked Example (GPT-4o pricing, April 2025)

```
Request:
  System prompt:  500 tokens
  Few-shot examples: 800 tokens
  User query:     100 tokens
  Total input:    1,400 tokens

Response:
  Model output:   350 tokens

GPT-4o pricing: $2.50 / 1M input, $10.00 / 1M output

Cost per call:
  Input:  (1,400 / 1,000,000) × $2.50 = $0.0035
  Output: (350   / 1,000,000) × $10.00 = $0.0035
  Total:  $0.0070 per call

At 1M requests/day:
  $7,000 / day = $210,000 / month
```

### Cost Optimization Levers

| Lever | Savings | Trade-off |
|-------|---------|-----------|
| **Shorter system prompt** | 10–40% | May lose instruction precision |
| **Fewer few-shot examples** | 20–50% | May reduce accuracy |
| **Smaller model** (GPT-4o → GPT-4o-mini) | 50–95% | Quality drop on hard tasks |
| **Prompt caching** | 50–90% on cached portion | Anthropic/OpenAI supported |
| **Semantic caching** | 20–60% | New infrastructure needed |
| **Output length control** | 20–50% | Need `max_tokens` + prompt engineering |
| **Batching** | 50% (OpenAI Batch API) | 24hr latency acceptable |

### Prompt Caching

Anthropic and OpenAI cache the **prefix** of your prompt (system prompt + static few-shot examples). Subsequent calls with the same prefix pay only for the new portion:

```
Call 1: [system(500) + few-shot(800) + query_1(100)] = 1,400 tokens at full price
Call 2: [system(500) + few-shot(800) + query_2(100)]
        → Cached: 1,300 tokens at ~10% price
        → New: 100 tokens at full price
        → Savings: ~85% on input cost
```

---

## 7. Provider Comparison

### Key API Differences

| Feature | OpenAI | Anthropic (Claude) | Google (Gemini) |
|---------|--------|-------------------|-----------------|
| **Top model** | GPT-4o | Claude Sonnet 4.6 | Gemini 1.5 Pro |
| **Context window** | 128K | 200K | 1M |
| **Function calling** | Yes (parallel) | Yes (tool use) | Yes (parallel) |
| **Structured output** | JSON Schema | Tool-based JSON | JSON mode |
| **Streaming** | SSE | SSE | SSE |
| **Prompt caching** | Yes | Yes (explicit) | Yes |
| **Batch API** | Yes (50% discount) | Yes | Yes |
| **Multimodal** | Yes (vision, audio) | Yes (vision) | Yes (vision, audio, video) |
| **Rate limits** | Tier-based | Tier-based | Project-based |

### Message Format Differences

```python
# OpenAI / most providers
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}

# Anthropic (system is separate param, not in messages)
client.messages.create(
    system="You are a helpful assistant.",  # separate!
    messages=[{"role": "user", "content": "..."}]
)

# Tool results
# OpenAI: role="tool", tool_call_id=...
# Anthropic: role="user", content=[{"type": "tool_result", "tool_use_id": ...}]
```

Use **LiteLLM** or **OpenRouter** to abstract over provider differences with a unified API.

---

## 8. Rate Limiting, Retry & Fallback

### Production Failure Modes

```
Error 429 — Rate limit exceeded:
  Too many requests per minute (RPM) or tokens per minute (TPM)
  Solution: exponential backoff + retry

Error 500/503 — Server error / unavailable:
  Provider outage or overload
  Solution: retry with backoff, then failover to alternate provider

Error 400 — Bad request:
  Malformed message, context too long, invalid tool schema
  Solution: do NOT retry (same request will fail again); fix the request

Timeout:
  Provider taking > your SLA
  Solution: cancel, log, retry, or serve cached response
```

### Exponential Backoff with Jitter

```python
import time, random

def call_with_retry(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)  # jitter prevents thundering herd
            time.sleep(wait)

# Backoff schedule: ~1s, ~2s, ~4s, ~8s, ~16s
# Jitter: prevents all clients retrying simultaneously after a rate-limit spike
```

### Failover Strategy

```
Primary:   OpenAI GPT-4o        (best quality)
Secondary: Anthropic Claude     (failover on OpenAI outage)
Tertiary:  Self-hosted Llama 3  (failover on both; always available)

Decision logic:
  1. Try primary
  2. If 429/503, wait + retry up to N times
  3. If still failing, route to secondary
  4. Log all failovers for incident analysis
```

---

## 9. Semantic Caching

### Exact-Match vs Semantic Caching

```
Exact-match cache:
  Key: hash(prompt string)
  "What is the capital of France?" → cached → "Paris"
  "What's the capital of France?"  → MISS (different string, same meaning)
  Hit rate: ~5–15% (only exact repeats)

Semantic cache:
  Key: embedding of query
  "What is the capital of France?" → embed → [0.12, -0.34, ...]
                                   → nearest neighbor search
  "What's the capital of France?"  → embed → [0.11, -0.35, ...]
                                   → cosine similarity = 0.98 → HIT
  Hit rate: ~20–60% depending on query distribution
```

### Implementation

```
Query arrives
    │
    ▼
Embed query → embedding vector
    │
    ▼
Vector similarity search in cache (e.g., Redis + FAISS)
    │
    ├── similarity ≥ threshold (e.g., 0.95)?
    │       → Return cached response (free!)
    │
    └── similarity < threshold?
            → Call LLM API
            → Store (query embedding, response) in cache
            → Return response

Threshold tuning:
  0.99 → very conservative, fewer false hits, lower hit rate
  0.90 → aggressive, more hits, risk of returning wrong cached response
  0.95 → typical production sweet spot
```

### When Semantic Caching Helps Most

- High query volume with natural repetition (FAQ chatbots, search assistants)
- Queries cluster around common intents (customer support: "track my order", "cancel subscription")
- Cost-sensitive applications where 30%+ hit rate is achievable

### Risks

- **Semantic collision**: "How do I cancel my subscription?" and "How do I cancel my order?" may be similar in embedding space but need different answers → use domain-specific embeddings + lower threshold
- **Stale cache**: cached response may become outdated → TTL expiry
- **Privacy**: don't cache responses containing user-specific data

---

## 10. Model Routing

### The Cost-Quality Trade-off

```
Query: "What is 2 + 2?"
  → GPT-4o costs $0.007 and gives "4"
  → GPT-4o-mini costs $0.0002 and gives "4"
  → 35× cost difference for identical output

Query: "Analyze the strategic implications of this 10-page acquisition proposal..."
  → GPT-4o-mini may miss nuance
  → GPT-4o is worth the cost
```

**Model router**: classifies query complexity and sends to the appropriate model tier.

### Routing Strategies

**Strategy 1 — Rule-based (fast, zero overhead)**:
```python
def route(query: str) -> str:
    if len(query.split()) < 20:          return "gpt-4o-mini"
    if any(k in query for k in SIMPLE_KEYWORDS): return "gpt-4o-mini"
    return "gpt-4o"
```

**Strategy 2 — Classifier-based (accurate, ~5ms overhead)**:
```python
# Fine-tuned small classifier: input=query, output=complexity score 0-1
score = complexity_classifier(query)
model = "gpt-4o" if score > 0.6 else "gpt-4o-mini"
```

**Strategy 3 — Cascading (best quality, higher latency)**:
```python
# Try small model first; escalate if confidence is low
response, confidence = small_model_with_confidence(query)
if confidence < 0.8:
    response = large_model(query)
```

### Routing Features for Classifier

| Feature | Signal |
|---------|--------|
| Query length | Longer queries tend to be harder |
| Sentence count | Multi-sentence = multi-step reasoning |
| Domain keywords | "analyze", "compare", "synthesize" → hard |
| Code presence | Code tasks generally harder |
| Ambiguity score | Low-perplexity under small model → clear/easy |
| Historical accuracy | Track small model accuracy by query type |

### Cost Savings from Routing

```
Baseline (all GPT-4o):       $7,000/day for 1M requests
With routing (60% to mini):  $7,000 × 0.40 + $200 × 0.60 = $2,920/day
Savings:                     58% cost reduction
```

---

## 11. Hosted API vs Self-Hosted OSS Models

### Decision Framework for Healthcare Application

| Dimension | Hosted API (OpenAI/Anthropic) | Self-Hosted (Llama 3 on H100) |
|-----------|-------------------------------|-------------------------------|
| **Data privacy** | Data leaves your infrastructure → HIPAA risk | Data stays on-prem → HIPAA compliant |
| **Quality** | GPT-4o: state-of-the-art | Llama 3 70B: 80–90% of GPT-4o quality |
| **Cost at scale** | $0.007/call → $7K/day at 1M calls | $3K/month for 2× H100 → $0.0001/call at scale |
| **Latency** | 200–800ms (network + queue) | 50–200ms (on-prem, no network hop) |
| **Setup time** | Minutes (API key) | Weeks (infra, deployment, ops) |
| **Maintenance** | Zero (provider handles) | High (updates, scaling, monitoring) |
| **Customization** | Prompt engineering only | Full fine-tuning, LoRA adapters |
| **Availability SLA** | 99.9% (provider SLA) | You own the SLA |

### Cost Crossover Analysis

```
At what volume does self-hosting become cheaper?

Hosted GPT-4o-mini: $0.0002/call
Self-hosted setup:  $3,000/month infra + $5,000 one-time setup

Break-even: $3,000 / $0.0002 = 15,000,000 calls/month = 500K calls/day

→ Self-hosting is worth it above ~500K requests/day for mini-tier quality
→ Above ~50K requests/day for GPT-4o-quality workloads
```

### Healthcare-Specific Recommendation

```
For PHI (Protected Health Information) data:
  → Self-host: HIPAA requires data not leave your control
  → Use: Llama 3 70B or Mistral + on-prem H100 cluster
  → Add: BAA (Business Associate Agreement) if using hosted

For non-PHI queries (general health info):
  → Hosted API is acceptable and easier
  → Faster iteration, zero infra cost
```

---

## 12. LLM Gateway Design

### What an LLM Gateway Does

A centralized proxy that sits between your application and LLM providers:

```
App 1 ──┐
App 2 ──┼──► LLM Gateway ──► OpenAI
App 3 ──┘       │         ──► Anthropic
                │         ──► Self-hosted Llama
                │
         Responsibilities:
         ✓ Authentication & API key management
         ✓ Rate limiting per user/team
         ✓ Request logging & audit trail
         ✓ Cost tracking per team/project
         ✓ Model routing
         ✓ Retry & fallback logic
         ✓ Response caching
         ✓ PII scrubbing
```

### Gateway Architecture

```
                      ┌─────────────────────────────────────────┐
                      │              LLM Gateway                  │
                      │                                          │
Request ──► Auth ──► Rate   ──► PII  ──► Cache  ──► Router ──► Provider
           check    limiter    scrub    lookup    (model/     (OpenAI /
                    per key                       provider)   Anthropic)
                       │                              │
                    Reject                       Log + track
                    (429)                        cost + latency
                                                      │
Response ◄─────────────────────── Output validation ◄┘
              (PII scan, safety check, format check)
```

### Key Design Decisions

**1. Rate limiting strategy**:
```
Per API key: 60 RPM, 100K TPM  (prevent runaway costs)
Per team:    500 RPM, 2M TPM   (aggregate team budget)
Global:      10K RPM            (protect downstream providers)
Algorithm: Token bucket (handles bursts) vs Fixed window (simpler)
```

**2. Cost tracking**:
```
Every response has usage.prompt_tokens + usage.completion_tokens
Log: {timestamp, user_id, team_id, model, input_tokens, output_tokens, cost, latency}
Dashboard: cost per team per day, top consumers, anomaly alerts
```

**3. Logging for compliance**:
```
Healthcare/Finance: log full prompt + response (encrypted, access-controlled)
Consumer apps: log metadata only (no PII in logs)
Retention: 30–90 days for debugging, 7 years for compliance
```

**4. Open-source gateway options**:

| Gateway | Key Features |
|---------|-------------|
| **LiteLLM Proxy** | 100+ model support, cost tracking, load balancing |
| **PortKey** | Analytics, guardrails, semantic caching |
| **Helicone** | Logging, caching, custom models |
| **Kong AI Gateway** | Enterprise, built on Kong |

---

## 13. Interview Questions & Answers

### Q1: What is function calling in LLM APIs? Walk through the full round-trip.

**Answer**: Function calling lets the LLM request execution of your application's functions. The flow is: (1) you send the user query plus a list of tool definitions (name, description, JSON Schema for parameters); (2) the model decides whether a tool is needed — if so, it returns a special `tool_calls` response with the function name and arguments as JSON; (3) your code executes the function with those arguments and gets a result; (4) you append the tool result to the conversation with role `"tool"` and the `tool_call_id`; (5) the model reads the result and generates the final natural language response. The critical insight: **the model never executes code** — it only outputs structured JSON describing what it wants to call. Your application is fully in control of what gets executed. This enables safe integration with sensitive systems because you can validate, authorize, or block tool calls before execution.

### Q2: How do you estimate the cost of an LLM API call? What is the input/output token pricing difference?

**Answer**: Cost = (input_tokens / 1M) × price_per_1M_input + (output_tokens / 1M) × price_per_1M_output. Output tokens are 3–5× more expensive than input tokens because output is generated sequentially (one autoregressive step per token, using full GPU compute for each), while input is processed in a single parallel forward pass (prefill). For GPT-4o at $2.50/$10 per 1M tokens: a call with 1,000 input tokens and 200 output tokens costs $0.0025 + $0.0020 = $0.0045. At 1M calls/day this is $4,500/day. Key cost levers: shorten system prompts (saves input tokens), use `max_tokens` to cap output (saves output tokens), use smaller models for simple tasks (90% cheaper), enable prompt caching for repeated prefixes (saves 50–90% on repeated input), use the Batch API for async workloads (50% discount).

### Q3: What is semantic caching for LLM APIs? How is it different from exact-match caching?

**Answer**: Exact-match caching stores (hash(prompt) → response) and only hits for byte-identical prompts — effective for repeated identical queries (5–15% hit rate in most apps). Semantic caching stores (embedding(prompt) → response) and retrieves cached responses for **semantically similar** queries above a cosine similarity threshold (typically 0.95). "What is the capital of France?" and "What's France's capital city?" are different strings but have embeddings with cosine similarity ~0.98 — semantic cache returns the same answer without an API call. Implementation: embed each incoming query, run approximate nearest-neighbor search against a vector store of cached (embedding, response) pairs, return if similarity ≥ threshold, otherwise call LLM and cache the new pair. Hit rate: 20–60% for FAQ/support bots with clustered intents. Main risk: semantic collision — different questions with similar embeddings getting wrong cached answers; mitigate with higher threshold + domain-specific embeddings + TTL expiry for time-sensitive data.

### Q4: Design a cost-optimization strategy for an LLM app processing 1M requests/day.

**Answer**: Baseline at $7K/day (all GPT-4o), target 70% reduction. Layer the optimizations: (1) **Model routing** (biggest win, ~50-60% savings): fine-tune a small classifier to route ~60% of simple queries to GPT-4o-mini ($200/day for those) vs GPT-4o for the complex 40% — net cost ~$3,000/day; (2) **Semantic caching** (~20% additional reduction): embed queries, cache in Redis+FAISS, return cached responses for similarity ≥ 0.95. For a support bot with repetitive queries, expect 30% cache hit rate → saves 30% of remaining API calls; (3) **Prompt compression**: audit and trim system prompts (often 20-30% of input tokens are redundant instructions) + reduce few-shot examples from 5 to 2; (4) **Prompt caching**: use OpenAI/Anthropic prefix caching for shared system prompt + few-shot examples — saves ~70% on those input tokens; (5) **Batch API** for non-urgent workloads (analytics, batch processing): 50% discount, 24h SLA. Combined: $7K → ~$1.5K/day (78% reduction) with careful monitoring of quality regressions at each step.

### Q5: How do you implement rate limiting, retry logic, and fallback for production LLM usage?

**Answer**: Three separate concerns. **Rate limiting**: implement a token bucket at the gateway layer — each API key gets a bucket of N tokens (requests) that refills at R rate per minute. Exceed the bucket → 429 returned immediately without hitting the upstream provider. Set limits per user, per team, and globally. **Retry logic**: use exponential backoff with jitter for transient errors (429, 503, timeout). Start at 1s, double each retry, add random jitter (0–1s) to prevent thundering herd, cap at 5 retries with ~30s max wait. Never retry 400 errors (malformed request) or 401 (auth failure) — they'll fail every time. **Fallback**: define a provider hierarchy (GPT-4o → Claude → Llama self-hosted). After N retries on primary fail, route to secondary. Track failover events in metrics. Design the fallback model to have the same tool schemas and message format (use LiteLLM or similar abstraction). Alert on-call when failover exceeds 1% of traffic — it signals a provider incident or config problem.

### Q6: Compare hosted APIs vs self-hosted OSS models for a healthcare application.

**Answer**: The dominant concern in healthcare is **HIPAA compliance** — PHI (Protected Health Information) cannot be sent to third-party APIs without a Business Associate Agreement (BAA). OpenAI, Anthropic, and Google all offer BAAs, so hosted APIs are technically HIPAA-compliant with a BAA in place. However, many healthcare organizations prefer self-hosted for defense-in-depth: data never leaves the network boundary, no risk of provider data retention policy violations, and full audit control. **Quality**: GPT-4o still leads Llama 3 70B on complex medical reasoning by ~5–15% on clinical benchmarks. For straightforward tasks (note summarization, ICD coding), Llama 3 70B is 85–95% as good. **Cost**: at 1M+ calls/month, self-hosted H100 clusters ($3–8K/month amortized) beat OpenAI ($0.007/call = $7K/day at 1M/day). **Recommendation**: hybrid — use self-hosted Llama 3 70B for routine, high-volume tasks with PHI; use hosted GPT-4o with BAA for low-volume, high-complexity clinical decision support where quality difference matters.
