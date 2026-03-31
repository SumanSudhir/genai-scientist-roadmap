# Topic 13: Working with LLM APIs

## Table of Contents
1. [The LLM API Landscape](#1-the-llm-api-landscape)
2. [OpenAI API — Chat Completions](#2-openai-api--chat-completions)
3. [Understanding Parameters: Temperature, top_p, & More](#3-understanding-parameters-temperature-top_p--more)
4. [Streaming Responses](#4-streaming-responses)
5. [Function Calling / Tool Use](#5-function-calling--tool-use)
6. [Structured Outputs (JSON Mode)](#6-structured-outputs-json-mode)
7. [Anthropic API (Claude)](#7-anthropic-api-claude)
8. [Open-Source Model Serving: Ollama](#8-open-source-model-serving-ollama)
9. [Production Serving with vLLM](#9-production-serving-with-vllm)
10. [Prompt Engineering Essentials](#10-prompt-engineering-essentials)
11. [Multi-Turn Conversations & Memory](#11-multi-turn-conversations--memory)
12. [Cost Management & Optimization](#12-cost-management--optimization)
13. [Error Handling & Reliability](#13-error-handling--reliability)
14. [Practice Exercises](#14-practice-exercises)
15. [Mini-Project: Multi-Turn Assistant with Memory](#15-mini-project-multi-turn-assistant-with-memory)
16. [Interview Questions & Answers](#16-interview-questions--answers)

---

## 1. The LLM API Landscape

### The Three Ways to Use LLMs

```
┌─────────────────────────────────────────────────────────────────┐
│                   How to Access LLMs                            │
├───────────────────┬───────────────────┬─────────────────────────┤
│   Cloud APIs      │   Self-Hosted     │   Local Inference       │
│   (Managed)       │   (Your GPUs)     │   (Your Machine)        │
├───────────────────┼───────────────────┼─────────────────────────┤
│ OpenAI, Anthropic │ vLLM, TGI,        │ Ollama, llama.cpp,      │
│ Google, AWS       │ TensorRT-LLM      │ LM Studio               │
├───────────────────┼───────────────────┼─────────────────────────┤
│ ✓ No infra needed │ ✓ Full control    │ ✓ Free, private          │
│ ✓ Latest models   │ ✓ Data privacy    │ ✓ No internet needed     │
│ ✗ Per-token cost  │ ✗ GPU cost + ops  │ ✗ Limited model size     │
│ ✗ Data sent out   │ ✗ You manage it   │ ✗ Slower (CPU/small GPU) │
├───────────────────┼───────────────────┼─────────────────────────┤
│ Best for:         │ Best for:         │ Best for:                │
│ Prototyping,      │ High volume,      │ Development, testing,    │
│ small-medium      │ data-sensitive,   │ offline use, learning    │
│ scale             │ custom models     │                          │
└───────────────────┴───────────────────┴─────────────────────────┘
```

### Model Selection Cheat Sheet (2025-26)

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Best quality, cost not an issue | GPT-4o / Claude Opus | Frontier capability |
| Good quality, cost-conscious | GPT-4o-mini / Claude Haiku | 90% quality, 10% cost |
| Open-source, self-hosted | Llama 3.1 70B / Mistral Large | Full control |
| Fast local dev | Llama 3.2 3B via Ollama | Free, instant |
| Code generation | GPT-4o / Claude Sonnet / DeepSeek Coder | Best at code |
| Embedding / retrieval | text-embedding-3-small / BGE | Optimized for search |

---

## 2. OpenAI API — Chat Completions

### Installation & Setup

```python
# Install
# pip install openai

import os
from openai import OpenAI

# Initialize client — reads OPENAI_API_KEY from environment
client = OpenAI()

# Or explicitly pass the key (not recommended for production)
# client = OpenAI(api_key="sk-...")
```

### Basic Chat Completion

```python
def chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Send a single prompt and get a response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# Usage
answer = chat("What is gradient descent in one sentence?")
print(answer)
```

### The Message Format

The chat API uses a list of messages with roles:

```python
messages = [
    # System message: sets behavior, personality, constraints
    {"role": "system", "content": "You are a Python expert. Be concise."},

    # User message: the human's input
    {"role": "user", "content": "How do I read a CSV file?"},

    # Assistant message: the model's previous response (for multi-turn)
    {"role": "assistant", "content": "Use pandas: `pd.read_csv('file.csv')`"},

    # User follow-up
    {"role": "user", "content": "What if it's tab-separated?"},
]
```

### The Response Object

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

# Key fields
print(response.choices[0].message.content)   # The actual text response
print(response.choices[0].message.role)       # "assistant"
print(response.choices[0].finish_reason)      # "stop", "length", "tool_calls"
print(response.model)                         # Model used
print(response.usage.prompt_tokens)           # Input tokens
print(response.usage.completion_tokens)       # Output tokens
print(response.usage.total_tokens)            # Total tokens
```

---

## 3. Understanding Parameters: Temperature, top_p, & More

### Temperature

Controls randomness. Mathematically, temperature scales the logits before softmax:

$$
P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

```python
# Deterministic (good for factual tasks, code, extraction)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0,   # Always picks the most likely token
)

# Creative (good for brainstorming, stories, marketing copy)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem about Python"}],
    temperature=1.2,  # More random, more creative
)
```

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0 | Deterministic, always picks top token | Code, math, extraction, classification |
| 0.1-0.3 | Mostly deterministic, slight variation | Factual Q&A, summarization |
| 0.7-1.0 | Balanced creativity | General chat, writing |
| 1.0-1.5 | High creativity, less coherent | Brainstorming, creative fiction |

### top_p (Nucleus Sampling)

Instead of a fixed temperature, restrict to the smallest set of tokens whose cumulative probability exceeds $p$:

$$
\text{Top-p set} = \min \{S : \sum_{x_i \in S} P(x_i) \geq p\}
$$

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    temperature=1.0,
    top_p=0.9,  # Only sample from tokens in the top 90% probability mass
)
```

**Rule of thumb**: Adjust temperature OR top_p, not both. OpenAI recommends changing one while keeping the other at default.

### Other Parameters

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=500,           # Max output tokens (controls cost and length)
    stop=["\n\n", "END"],     # Stop generation at these sequences
    presence_penalty=0.5,     # Penalize tokens that appeared (encourages new topics)
    frequency_penalty=0.5,    # Penalize tokens by frequency (reduces repetition)
    n=3,                      # Generate 3 completions (for self-consistency)
    seed=42,                  # For reproducibility (best-effort)
)

# Access multiple completions
for i, choice in enumerate(response.choices):
    print(f"Choice {i}: {choice.message.content}")
```

---

## 4. Streaming Responses

Streaming returns tokens as they're generated instead of waiting for the full response. Critical for real-time chat UIs.

### Basic Streaming

```python
def chat_stream(prompt: str, model: str = "gpt-4o-mini"):
    """Stream a response token by token."""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=True,  # Enable streaming
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            full_response += delta.content

    print()  # Newline at the end
    return full_response


# Usage — tokens print as they arrive
response = chat_stream("Explain transformers in 3 sentences")
```

### Streaming with Async (for web servers)

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def chat_stream_async(prompt: str):
    """Async streaming for use in FastAPI / web frameworks."""
    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_response += delta.content
            yield delta.content  # Yield each token for SSE

    return full_response
```

### FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(prompt: str):
    """Stream LLM response via Server-Sent Events."""
    async def generate():
        stream = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 5. Function Calling / Tool Use

Function calling lets the LLM decide **when** to call a function and **what arguments** to pass. The LLM doesn't execute the function — your code does.

### The Flow

```
┌──────┐    1. User message         ┌──────┐
│ User │ ──────────────────────────► │ LLM  │
└──────┘                             └──┬───┘
                                        │
                     2. LLM decides to   │
                        call a function  │
                                        ▼
                              ┌──────────────────┐
                              │ tool_calls:       │
                              │   name: get_weather│
                              │   args: {city:    │
                              │         "London"} │
                              └────────┬─────────┘
                                       │
                    3. YOUR CODE        │
                       executes the     ▼
                       function    ┌──────────┐
                                   │ get_weather│
                                   │ ("London") │
                                   └─────┬──────┘
                                         │
                    4. Send result back   │
                       to LLM            ▼
                              ┌──────────────────┐
                              │ tool result:      │
                              │ "London: 15°C,   │
                              │  partly cloudy"   │
                              └────────┬─────────┘
                                       │
                    5. LLM generates    │
                       final response   ▼
                              ┌──────────────────┐
                              │ "The weather in   │
                              │  London is 15°C   │
                              │  and partly cloudy"│
                              └──────────────────┘
```

### Implementation

```python
import json

# Step 1: Define the tools (function schemas)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g., 'London'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g., '2 * 3 + 5'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# Step 2: Define the actual function implementations
def get_weather(city: str, unit: str = "celsius") -> str:
    """Simulate a weather API call."""
    # In production, call a real weather API here
    weather_data = {
        "london": {"temp": 15, "condition": "partly cloudy"},
        "tokyo": {"temp": 22, "condition": "sunny"},
        "new york": {"temp": 8, "condition": "rainy"},
    }
    data = weather_data.get(city.lower(), {"temp": 20, "condition": "unknown"})
    return json.dumps({"city": city, "temperature": data["temp"], "unit": unit, "condition": data["condition"]})


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Map function names to implementations
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# Step 3: The complete function calling loop
def chat_with_tools(user_message: str) -> str:
    """Chat with automatic tool use."""
    messages = [
        {"role": "system", "content": "You can check weather and do math. Use the tools when needed."},
        {"role": "user", "content": user_message},
    ]

    # First API call — LLM may decide to call tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
    )

    message = response.choices[0].message

    # Check if the LLM wants to call tools
    if message.tool_calls:
        # Add the assistant's message (with tool calls) to history
        messages.append(message)

        # Execute each tool call
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  [Tool Call] {func_name}({func_args})")

            # Call the actual function
            result = TOOL_FUNCTIONS[func_name](**func_args)

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Second API call — LLM generates final response using tool results
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

    return response.choices[0].message.content


# Usage
print(chat_with_tools("What's the weather in London?"))
# [Tool Call] get_weather({'city': 'London'})
# "The weather in London is 15°C and partly cloudy."

print(chat_with_tools("What is 15 * 37 + 42?"))
# [Tool Call] calculate({'expression': '15 * 37 + 42'})
# "15 × 37 + 42 = 597"
```

### Parallel Tool Calls

The LLM can call multiple tools in a single response:

```python
# "What's the weather in London and Tokyo?"
# → LLM returns TWO tool_calls:
#   tool_calls[0]: get_weather(city="London")
#   tool_calls[1]: get_weather(city="Tokyo")
# → Your code executes both, sends both results back
# → LLM combines: "London is 15°C, Tokyo is 22°C"
```

### tool_choice Parameter

```python
# Let the model decide whether to use tools
tool_choice="auto"          # Default — model decides

# Force the model to call a specific function
tool_choice={"type": "function", "function": {"name": "get_weather"}}

# Force the model to call at least one tool
tool_choice="required"

# Prevent any tool use
tool_choice="none"
```

---

## 6. Structured Outputs (JSON Mode)

### Basic JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract information and return JSON."},
        {"role": "user", "content": "John Smith is a 32-year-old software engineer at Google in Mountain View."},
    ],
    response_format={"type": "json_object"},  # Force JSON output
)

result = json.loads(response.choices[0].message.content)
print(result)
# {"name": "John Smith", "age": 32, "job": "software engineer", "company": "Google", "location": "Mountain View"}
```

### Structured Outputs with Pydantic (OpenAI SDK)

```python
from pydantic import BaseModel
from typing import Optional


class PersonInfo(BaseModel):
    name: str
    age: Optional[int]
    job_title: str
    company: str
    location: Optional[str]


response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract person information from the text."},
        {"role": "user", "content": "John Smith is a 32-year-old software engineer at Google in Mountain View."},
    ],
    response_format=PersonInfo,
)

person = response.choices[0].message.parsed
print(person.name)       # "John Smith"
print(person.age)        # 32
print(person.company)    # "Google"
print(type(person))      # <class 'PersonInfo'>
```

### Using the Instructor Library (Works with Any Provider)

```python
# pip install instructor

import instructor
from pydantic import BaseModel, Field
from typing import List

# Patch the OpenAI client
client_instructor = instructor.from_openai(OpenAI())


class Skill(BaseModel):
    name: str
    proficiency: str = Field(description="beginner, intermediate, or expert")


class ResumeExtraction(BaseModel):
    name: str
    email: Optional[str]
    years_of_experience: int
    skills: List[Skill]
    summary: str = Field(description="One-sentence professional summary")


resume_text = """
Jane Doe - jane.doe@email.com
Senior ML Engineer with 7 years of experience.
Expert in PyTorch, Python, and transformer models.
Intermediate in Kubernetes and AWS.
Built recommendation systems serving 10M users.
"""

result = client_instructor.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Extract structured info from this resume:\n\n{resume_text}"},
    ],
    response_model=ResumeExtraction,
)

print(result.model_dump_json(indent=2))
# {
#   "name": "Jane Doe",
#   "email": "jane.doe@email.com",
#   "years_of_experience": 7,
#   "skills": [
#     {"name": "PyTorch", "proficiency": "expert"},
#     {"name": "Python", "proficiency": "expert"},
#     {"name": "Transformer models", "proficiency": "expert"},
#     {"name": "Kubernetes", "proficiency": "intermediate"},
#     {"name": "AWS", "proficiency": "intermediate"}
#   ],
#   "summary": "Senior ML Engineer specializing in recommendation systems and deep learning."
# }
```

---

## 7. Anthropic API (Claude)

### Setup & Basic Usage

```python
# pip install anthropic

from anthropic import Anthropic

anthropic_client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment

response = anthropic_client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system="You are a helpful Python tutor. Be concise.",  # System prompt is separate
    messages=[
        {"role": "user", "content": "Explain list comprehensions with an example."},
    ],
)

print(response.content[0].text)
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

### Key Differences from OpenAI

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt | Inside `messages` list | Separate `system` parameter |
| Response content | `response.choices[0].message.content` (string) | `response.content[0].text` (list of blocks) |
| Streaming | `stream=True` returns chunks | `stream=True` returns event stream |
| Tool use | `tools` + `tool_calls` | `tools` + `tool_use` content blocks |
| JSON mode | `response_format={"type": "json_object"}` | No native JSON mode — use tool use or prompting |
| Max tokens | Optional (has default) | **Required** — must always specify |

### Claude Streaming

```python
def chat_stream_claude(prompt: str):
    """Stream Claude responses."""
    with anthropic_client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()
```

### Claude Tool Use

```python
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {  # Note: "input_schema" not "parameters"
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)

# Check for tool use in response
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Args: {block.input}, ID: {block.id}")
```

---

## 8. Open-Source Model Serving: Ollama

### Setup

```bash
# Install Ollama (macOS)
# brew install ollama

# Or download from https://ollama.com

# Pull a model
# ollama pull llama3.2:3b
# ollama pull mistral
# ollama pull phi3:mini

# Run a model (starts serving on localhost:11434)
# ollama run llama3.2:3b
```

### Using Ollama with Python

```python
# pip install ollama

import ollama

# Basic chat
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ],
)
print(response["message"]["content"])
```

### Ollama with OpenAI-Compatible API

Ollama exposes an OpenAI-compatible endpoint, so you can use the OpenAI SDK:

```python
from openai import OpenAI

# Point to local Ollama server
local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required but unused
)

response = local_client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Explain gradient descent briefly."},
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

This is powerful — you can switch between OpenAI and local models by just changing `base_url` and `model`. Write your code once, swap providers easily.

### Ollama Streaming

```python
# Stream responses
stream = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Write a haiku about Python"}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
```

---

## 9. Production Serving with vLLM

### When to Use vLLM

Use vLLM when you need high-throughput, production-grade serving of open-source models. It provides paged attention, continuous batching, and an OpenAI-compatible API.

### Setup & Serving

```bash
# pip install vllm

# Start vLLM server (OpenAI-compatible)
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --port 8000 \
#     --max-model-len 4096
```

### Using vLLM from Python

```python
# vLLM exposes the same API as OpenAI — use the same client!
vllm_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = vllm_client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is attention in transformers?"},
    ],
    temperature=0.7,
    max_tokens=500,
)
print(response.choices[0].message.content)
```

### Provider-Agnostic Client Pattern

```python
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str       # "openai", "ollama", "vllm"
    model: str
    base_url: str | None = None
    api_key: str | None = None


def get_client(config: LLMConfig) -> OpenAI:
    """Create an OpenAI-compatible client for any provider."""
    providers = {
        "openai": {"base_url": "https://api.openai.com/v1", "api_key": os.getenv("OPENAI_API_KEY")},
        "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        "vllm": {"base_url": "http://localhost:8000/v1", "api_key": "not-needed"},
    }

    settings = providers[config.provider]
    return OpenAI(
        base_url=config.base_url or settings["base_url"],
        api_key=config.api_key or settings["api_key"],
    )


# Switch providers with one line
config = LLMConfig(provider="ollama", model="llama3.2:3b")
# config = LLMConfig(provider="openai", model="gpt-4o-mini")
# config = LLMConfig(provider="vllm", model="meta-llama/Llama-3.1-8B-Instruct")

client = get_client(config)
```

---

## 10. Prompt Engineering Essentials

### Zero-Shot Prompting

No examples — just instructions:

```python
def classify_sentiment(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral. Reply with one word only."},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()

print(classify_sentiment("This product is amazing!"))   # "positive"
print(classify_sentiment("Terrible service, never again")) # "negative"
```

### Few-Shot Prompting

Provide examples in the prompt:

```python
def classify_intent(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Classify the user's intent. Respond with exactly one of:
order_status, refund, product_question, complaint, other

Examples:
"Where is my package?" → order_status
"I want my money back" → refund
"Does this come in blue?" → product_question
"This broke after one day!" → complaint
"Can I speak to a manager?" → other"""},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

print(classify_intent("When will my order arrive?"))  # "order_status"
print(classify_intent("This is the worst product"))   # "complaint"
```

### Chain-of-Thought (CoT)

Ask the model to reason step by step:

```python
def solve_math(problem: str) -> str:
    """Solve a math problem with chain-of-thought reasoning."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Solve the problem step by step. Show your reasoning, then give the final answer on a new line starting with 'Answer: '."},
            {"role": "user", "content": problem},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# Without CoT: model might get this wrong
# With CoT: model reasons through each step
result = solve_math(
    "A store sells notebooks for $3 each. If you buy 5 or more, you get a 20% discount. "
    "How much do 7 notebooks cost?"
)
print(result)
# Step 1: Base price = 7 × $3 = $21
# Step 2: Since 7 ≥ 5, apply 20% discount
# Step 3: Discount = $21 × 0.20 = $4.20
# Step 4: Final price = $21 - $4.20 = $16.80
# Answer: $16.80
```

### System Prompt Best Practices

```python
# GOOD system prompt — specific, constrained, with format
system_prompt = """You are a customer support agent for TechCo.

Rules:
- Only answer questions about TechCo products
- If you don't know the answer, say "Let me connect you with a specialist"
- Never make up product features
- Be friendly but concise
- Always end with "Is there anything else I can help with?"

Response format:
- Keep responses under 100 words
- Use bullet points for multiple items"""

# BAD system prompt — vague, unconstrained
system_prompt_bad = "You are a helpful assistant. Help the user."
```

---

## 11. Multi-Turn Conversations & Memory

### Basic Multi-Turn Chat

```python
class ChatSession:
    """Manages a multi-turn conversation with history."""

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant."):
        self.model = model
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining history."""
        self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def get_token_count(self) -> int:
        """Rough estimate of token count (4 chars ≈ 1 token)."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4


# Usage
session = ChatSession(system_prompt="You are a Python tutor. Be concise.")
print(session.chat("What is a list comprehension?"))
print(session.chat("Show me an example with filtering"))
print(session.chat("How does it compare to a for loop in speed?"))
# Each response is aware of the full conversation history
```

### Sliding Window Memory (Handle Long Conversations)

```python
class SlidingWindowChat:
    """Chat with a sliding window to manage context length."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        max_messages: int = 20,  # Keep last N messages
    ):
        self.model = model
        self.system_message = {"role": "system", "content": system_prompt}
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        # Trim history if too long (keep system + last N messages)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        # Always include system message at the start
        api_messages = [self.system_message] + self.messages

        response = client.chat.completions.create(
            model=self.model,
            messages=api_messages,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message
```

### Summary-Based Memory (for Very Long Conversations)

```python
class SummaryMemoryChat:
    """Summarize old messages to fit in context window."""

    def __init__(self, model: str = "gpt-4o-mini", max_recent: int = 10):
        self.model = model
        self.system_prompt = "You are a helpful assistant."
        self.summary: str = ""       # Summary of older conversation
        self.recent: list[dict] = [] # Recent messages (kept verbatim)
        self.max_recent = max_recent

    def _summarize(self, messages: list[dict]) -> str:
        """Summarize a list of messages into a brief paragraph."""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Capture key facts and decisions."},
                {"role": "user", "content": conversation_text},
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content

    def chat(self, user_message: str) -> str:
        self.recent.append({"role": "user", "content": user_message})

        # If recent messages exceed limit, summarize the older half
        if len(self.recent) > self.max_recent:
            older = self.recent[:len(self.recent) // 2]
            new_summary = self._summarize(older)
            if self.summary:
                self.summary = f"{self.summary}\n{new_summary}"
            else:
                self.summary = new_summary
            self.recent = self.recent[len(self.recent) // 2:]

        # Build messages with summary context
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}",
            })
        messages.extend(self.recent)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        assistant_message = response.choices[0].message.content
        self.recent.append({"role": "assistant", "content": assistant_message})

        return assistant_message
```

---

## 12. Cost Management & Optimization

### Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a given text and model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """Estimate API cost in USD."""
    # Pricing as of early 2026 (check for current prices)
    pricing = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    }
    rates = pricing.get(model, pricing["gpt-4o-mini"])
    return input_tokens * rates["input"] + output_tokens * rates["output"]


# Example
text = "What is the capital of France?"
tokens = count_tokens(text)
print(f"Tokens: {tokens}")
print(f"Cost for 1000 requests (est): ${estimate_cost(tokens * 1000, 50 * 1000, 'gpt-4o-mini'):.4f}")
```

### Cost Comparison Table

| Model | Input (per 1M tokens) | Output (per 1M tokens) | 1M Requests (avg 200 in + 100 out) |
|-------|----------------------|----------------------|-------------------------------------|
| GPT-4o | $2.50 | $10.00 | $1,500 |
| GPT-4o-mini | $0.15 | $0.60 | $90 |
| Claude Sonnet | $3.00 | $15.00 | $2,100 |
| Claude Haiku | $0.25 | $1.25 | $175 |
| Llama 3.1 8B (self-hosted) | ~$0.05* | ~$0.05* | ~$10* |

*Self-hosted costs depend on GPU — amortized cost per token on an A100 at ~$2/hr.

### Cost Optimization Strategies

```python
import hashlib
from functools import lru_cache


# Strategy 1: Caching — avoid repeated API calls
class LLMCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self):
        self.cache: dict[str, str] = {}

    def _make_key(self, messages: list[dict], model: str) -> str:
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list[dict], model: str) -> str | None:
        key = self._make_key(messages, model)
        return self.cache.get(key)

    def set(self, messages: list[dict], model: str, response: str):
        key = self._make_key(messages, model)
        self.cache[key] = response


cache = LLMCache()


def cached_chat(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Chat with caching — identical requests return cached response."""
    cached = cache.get(messages, model)
    if cached:
        return cached

    response = client.chat.completions.create(model=model, messages=messages)
    result = response.choices[0].message.content

    cache.set(messages, model, result)
    return result


# Strategy 2: Model routing — use cheap models for easy tasks
def smart_chat(user_message: str) -> str:
    """Route to cheap or expensive model based on complexity."""
    # Simple heuristic: short queries → cheap model
    if len(user_message.split()) < 20:
        model = "gpt-4o-mini"
    else:
        model = "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.choices[0].message.content


# Strategy 3: Limit output tokens
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain transformers"}],
    max_tokens=150,  # Caps cost — shorter responses are cheaper
)
```

---

## 13. Error Handling & Reliability

### Robust API Client

```python
import time
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)


def chat_with_retry(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Chat with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30.0,  # 30-second timeout
            )
            return response.choices[0].message.content

        except RateLimitError:
            # Rate limited — wait and retry with exponential backoff
            delay = base_delay * (2 ** attempt)
            print(f"Rate limited. Retrying in {delay}s...")
            time.sleep(delay)

        except APITimeoutError:
            # Timeout — retry
            print(f"Timeout. Retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(base_delay)

        except APIConnectionError:
            # Network error — retry
            print(f"Connection error. Retrying in {base_delay}s...")
            time.sleep(base_delay)

        except APIError as e:
            # Other API errors (500s, etc.)
            if e.status_code and e.status_code >= 500:
                print(f"Server error {e.status_code}. Retrying...")
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise  # Don't retry client errors (400s)

    raise RuntimeError(f"Failed after {max_retries} retries")
```

### Using Tenacity for Advanced Retry Logic

```python
# pip install tenacity

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
)
def reliable_chat(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Production-ready chat with automatic retry."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=30.0,
    )
    return response.choices[0].message.content
```

### Fallback to Another Provider

```python
def chat_with_fallback(messages: list[dict]) -> str:
    """Try OpenAI first, fall back to Anthropic on failure."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            timeout=15.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI failed: {e}. Falling back to Claude...")
        # Convert message format for Anthropic
        system_msg = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthropic_messages.append(m)

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_msg,
            messages=anthropic_messages,
        )
        return response.content[0].text
```

---

## 14. Practice Exercises

- [ ] **Exercise 1**: Build a CLI chatbot using OpenAI API with streaming. Support multi-turn conversation with `/clear` to reset history.

- [ ] **Exercise 2**: Implement function calling — create 3 tools (weather lookup, calculator, URL fetcher) and build the complete tool-calling loop that handles parallel tool calls.

- [ ] **Exercise 3**: Run Llama 3 locally with Ollama. Build a simple Python chat interface using the OpenAI-compatible API. Compare response quality vs GPT-4o-mini on 10 test prompts.

- [ ] **Exercise 4**: Build a structured data extractor using Pydantic + Instructor that extracts job postings from raw text into `{title, company, location, salary_range, skills: List[str], remote: bool}`.

- [ ] **Exercise 5**: Implement the provider-agnostic client pattern. Write code that can switch between OpenAI, Ollama, and vLLM by changing a single config variable.

---

## 15. Mini-Project: Multi-Turn Assistant with Memory

Build a complete conversational assistant with:

### Requirements
1. Multi-turn conversation with history
2. System prompt defining the assistant's persona
3. Streaming responses
4. Sliding window memory (keep last 20 messages)
5. Summary-based compression for older messages
6. Token counting and cost tracking per session
7. Support both OpenAI and Ollama backends
8. Graceful error handling with retry

### Starter Architecture

```python
"""
Multi-turn conversational assistant with memory management.

Usage:
    python assistant.py --provider openai --model gpt-4o-mini
    python assistant.py --provider ollama --model llama3.2:3b
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class SessionStats:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    start_time: float = field(default_factory=time.time)


class Assistant:
    def __init__(self, provider: str, model: str, system_prompt: str, max_recent: int = 20):
        self.model = model
        self.max_recent = max_recent
        self.system_prompt = system_prompt
        self.summary = ""
        self.recent_messages: list[dict] = []
        self.stats = SessionStats()

        # Initialize provider-specific client
        if provider == "openai":
            self.client = OpenAI()
        elif provider == "ollama":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _build_messages(self) -> list[dict]:
        """Build the message list with system prompt and memory."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"Earlier conversation summary: {self.summary}"})
        messages.extend(self.recent_messages)
        return messages

    def _compress_memory(self):
        """Summarize older messages when history gets too long."""
        if len(self.recent_messages) <= self.max_recent:
            return

        # Summarize the older half
        split = len(self.recent_messages) // 2
        older = self.recent_messages[:split]

        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=200,
            temperature=0,
        )

        new_summary = response.choices[0].message.content
        self.summary = f"{self.summary}\n{new_summary}".strip() if self.summary else new_summary
        self.recent_messages = self.recent_messages[split:]

    def chat(self, user_message: str) -> str:
        """Send a message and stream the response."""
        self.recent_messages.append({"role": "user", "content": user_message})
        self._compress_memory()

        messages = self._build_messages()

        # Stream the response
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
        print()

        self.recent_messages.append({"role": "assistant", "content": full_response})
        self.stats.total_requests += 1

        return full_response

    def print_stats(self):
        elapsed = time.time() - self.stats.start_time
        print(f"\n--- Session Stats ---")
        print(f"Requests: {self.stats.total_requests}")
        print(f"Messages in memory: {len(self.recent_messages)}")
        print(f"Has summary: {'Yes' if self.summary else 'No'}")
        print(f"Session duration: {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Multi-turn LLM assistant")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--system", default="You are a helpful, concise assistant.")
    args = parser.parse_args()

    assistant = Assistant(provider=args.provider, model=args.model, system_prompt=args.system)
    print(f"Assistant ready ({args.provider}/{args.model}). Type /quit to exit, /stats for stats, /clear to reset.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/stats":
            assistant.print_stats()
            continue
        if user_input == "/clear":
            assistant.recent_messages.clear()
            assistant.summary = ""
            print("Conversation cleared.")
            continue

        print("Assistant: ", end="")
        assistant.chat(user_input)

    assistant.print_stats()


if __name__ == "__main__":
    main()
```

---

## 16. Interview Questions & Answers

### Q1: How does function calling work in OpenAI's API? Walk through the message flow.

**Answer**: Function calling is a multi-step process. (1) You define tools as JSON schemas describing function names, descriptions, and parameter types. (2) You send the user's message along with the tool definitions to the API. (3) The LLM analyzes the query and decides whether to call a tool — if yes, it returns a response with `tool_calls` containing the function name and arguments (as JSON). The LLM does NOT execute the function. (4) Your code parses the tool call, executes the actual function, and gets the result. (5) You send the tool result back to the API as a message with `role: "tool"` and the matching `tool_call_id`. (6) The LLM generates a final natural language response incorporating the tool result. The LLM can also make parallel tool calls (e.g., checking weather in two cities simultaneously), and you handle each independently before sending all results back.

### Q2: What is the difference between temperature and top_p? When do you adjust each?

**Answer**: Both control randomness but differently. **Temperature** scales the logits before softmax: $P(x_i) = e^{z_i/T} / \sum_j e^{z_j/T}$. At $T=0$, it always picks the highest-probability token (deterministic). At $T>1$, the distribution flattens and rare tokens become more likely. **Top-p (nucleus sampling)** dynamically restricts the token pool to the smallest set whose cumulative probability exceeds $p$. With $p=0.1$, only the most confident tokens are considered; with $p=0.95$, nearly all tokens are in play. **When to use**: Temperature=0 for deterministic tasks (classification, extraction, code). Temperature 0.7-1.0 for creative tasks. Adjust one at a time — don't combine both. In practice, temperature is the more commonly tuned parameter because it's simpler to reason about.

### Q3: How would you estimate the cost of an LLM-powered feature serving 100K users/day?

**Answer**: Work backwards from usage patterns. Estimate (1) average queries per user per day (e.g., 3), (2) average input tokens per query (system prompt + user message + context, e.g., 500), (3) average output tokens per query (e.g., 200). So: 100K users × 3 queries × (500 input + 200 output) = 300K queries/day. With GPT-4o-mini at $0.15/1M input and $0.60/1M output: daily cost = 300K × 500 × $0.15/1M + 300K × 200 × $0.60/1M = $22.50 + $36 = **$58.50/day ≈ $1,755/month**. To reduce costs: (1) semantic caching for repeated queries (20-30% hit rate), (2) shorter system prompts, (3) model routing — use GPT-4o-mini for simple queries (70%), GPT-4o only for complex ones. With these optimizations, realistically $800-1000/month.

### Q4: Compare running Llama locally with Ollama vs using an API. When would you choose each?

**Answer**: **Ollama/local**: Free after hardware cost, complete data privacy (nothing leaves your machine), no internet needed, great for development and testing. Limitations: model size limited by your RAM/VRAM (a 7B model needs ~4GB RAM quantized), inference is slower than cloud GPUs, no access to frontier models (GPT-4o, Claude Opus). **API**: Access to best models, no hardware management, scales instantly, pay-per-use pricing. Limitations: per-token cost adds up at scale, data sent to third party, network latency, rate limits. **Choose local** when: developing/prototyping (saves money during iteration), data is sensitive/regulated (healthcare, finance), you're building for offline environments, or you need a fine-tuned open-source model. **Choose API** when: you need frontier-quality responses, volume is low-medium (cheaper than GPU costs), time-to-market matters, or you need features like function calling that open-source models may not support well.

### Q5: How do you handle multi-turn conversations that exceed the context window?

**Answer**: Three strategies in order of sophistication. **(1) Sliding window**: Keep only the last $N$ messages, drop the oldest. Simple but loses early context. Good for casual chat where recent context matters most. **(2) Summary-based compression**: When the message list exceeds a threshold, summarize the older half into a 2-3 sentence summary using the LLM itself. Keep the summary as a system message and recent messages verbatim. Preserves key facts while reducing tokens. Costs one extra API call per compression. **(3) Hybrid approach**: Sliding window for messages + a persistent summary of the full conversation + optionally a vector store of key facts for retrieval when relevant. This is what production chatbots use. **Implementation detail**: Always keep the system prompt outside the window — it should never be dropped. Monitor token count before each API call to avoid exceeding the model's context limit.

### Q6: What is structured output? How do you ensure an LLM returns valid JSON?

**Answer**: Structured output means forcing the LLM to return data in a specific format (usually JSON matching a schema). Three approaches: **(1) OpenAI's JSON mode** — set `response_format={"type": "json_object"}` and instruct the model to return JSON in the prompt. Guarantees valid JSON but not schema compliance. **(2) OpenAI's structured outputs** — pass a Pydantic model as `response_format`, and the API constrains generation to match the schema exactly. Most reliable but OpenAI-specific. **(3) Instructor library** — wraps any LLM provider, validates outputs against Pydantic models, and automatically retries if validation fails. Works with OpenAI, Anthropic, Ollama. In production, use approach 2 or 3 because they guarantee both valid JSON AND schema compliance. Always define clear Pydantic models with field descriptions and use `temperature=0` for extraction tasks.

### Q7: How do you implement error handling and reliability for LLM API calls?

**Answer**: Production LLM applications need multiple layers of resilience. **(1) Retry with exponential backoff**: Wrap API calls with retries for transient errors (rate limits, timeouts, 500s). Use exponential backoff: wait $\text{base} \times 2^{\text{attempt}}$ seconds between retries. Libraries like `tenacity` make this clean. **(2) Timeouts**: Set explicit timeouts (15-30s for normal calls) to prevent hanging. **(3) Fallback providers**: If your primary provider fails, fall back to another (OpenAI → Anthropic → local Ollama). Requires abstracting the client behind a common interface. **(4) Circuit breaker**: After N consecutive failures, stop calling the failing provider for a cooldown period. **(5) Input validation**: Validate and truncate inputs before sending to avoid context length errors. **(6) Output validation**: Parse and validate LLM responses — if the output doesn't match expected format, retry or return a default. **(7) Logging**: Log every API call (prompt, response, tokens, latency, errors) for debugging and cost tracking.

### Q8: Design a provider-agnostic LLM client. How do you make your code work with OpenAI, Anthropic, and local models?

**Answer**: The key insight is that Ollama and vLLM both expose **OpenAI-compatible APIs**. So the architecture is: create an `OpenAI` client instance with a custom `base_url` that points to the right backend. For OpenAI: default URL. For Ollama: `http://localhost:11434/v1`. For vLLM: `http://localhost:8000/v1`. All three use the same `chat.completions.create()` interface. **For Anthropic**, which has a different API format, create an adapter that translates between the OpenAI message format and Anthropic's format (system prompt extraction, content block handling). In practice: define a `LLMConfig` dataclass with provider, model, base_url, and api_key fields. A factory function returns the appropriate client. Your application code only interacts with the OpenAI-compatible interface, and swapping providers is a config change, not a code change. This pattern is essential for production — it lets you run cheap local models in development, A/B test providers in production, and fail over between providers automatically.

### Q9: When would you fine-tune a model vs use better prompting vs use RAG?

**Answer**: **Start with prompting** — always. It's the cheapest and fastest approach. Use zero-shot, then few-shot, then chain-of-thought. If prompting gets you 80%+ of desired quality, stop here. **Use RAG** when the model needs access to specific/current/private knowledge it wasn't trained on: company docs, product catalogs, recent events. RAG adds 200-500ms latency per retrieval but requires no training. **Fine-tune** when you need: (1) consistent output format/style the model doesn't naturally produce, (2) domain-specific behavior (medical, legal, financial terminology), (3) cost reduction — a fine-tuned small model can replace expensive API calls with a cheap self-hosted model. Fine-tuning requires labeled data (100+ examples minimum, 1K+ for good results) and compute. **The decision framework**: No data, general task → prompting. Have documents, need factual grounding → RAG. Have labeled examples, need behavior change → fine-tuning. Many production systems combine all three: fine-tuned model + RAG + careful prompt engineering.

### Q10: How does streaming work under the hood? Why is it important for user experience?

**Answer**: Without streaming, the user sees nothing until the entire response is generated — for a 500-token response at 50 tok/s, that's a 10-second wait staring at a blank screen. With streaming, tokens appear as they're generated, so the user sees the first token within ~200ms (TTFT). **Under the hood**: The API returns a stream of **Server-Sent Events (SSE)**. Each event contains a `delta` with partial content (usually 1-3 tokens). Your client reads these events incrementally and renders them. **Implementation**: Set `stream=True` in the API call. Instead of `response.choices[0].message.content`, iterate over the stream and concatenate `chunk.choices[0].delta.content`. **For web apps**: Use SSE or WebSockets to forward the token stream to the frontend. FastAPI's `StreamingResponse` handles this. **Gotchas**: (1) You can't know the total token count until the stream ends. (2) Function calls in streaming mode arrive as partial JSON that must be assembled. (3) Error handling is harder — errors can occur mid-stream.

---

*End of Topic 13: Working with LLM APIs*
