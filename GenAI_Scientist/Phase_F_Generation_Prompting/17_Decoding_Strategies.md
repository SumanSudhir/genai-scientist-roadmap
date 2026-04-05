# Topic 17: Decoding Strategies & Text Generation

> **Interview Weight**: ★★★★☆ — Temperature, top-k, top-p are asked at every LLM interview (even junior). Beam search length normalization, contrastive decoding, and constrained generation are senior [H]-level questions. Know all sampling methods with numbers.

## Table of Contents

1. [The Decoding Problem](#1-the-decoding-problem)
2. [Greedy Decoding](#2-greedy-decoding)
3. [Beam Search](#3-beam-search)
4. [Temperature Sampling](#4-temperature-sampling)
5. [Top-k Sampling](#5-top-k-sampling)
6. [Top-p (Nucleus) Sampling](#6-top-p-nucleus-sampling)
7. [Min-p Sampling](#7-min-p-sampling)
8. [Repetition & Frequency Penalties](#8-repetition--frequency-penalties)
9. [Contrastive Decoding](#9-contrastive-decoding)
10. [Constrained & Structured Generation](#10-constrained--structured-generation)
11. [Decoding Strategy Comparison](#11-decoding-strategy-comparison)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. The Decoding Problem

### What Decoding Is

A trained LLM outputs a **probability distribution over the vocabulary at each step**:

$$
P(y_t \mid y_{<t}, x) \in \mathbb{R}^{|V|}
$$

For GPT-style models, $|V| \approx 32{,}000$–$128{,}000$. Decoding is the algorithm that converts these distributions into an actual token sequence.

### The Core Tension

```
Quality ◄────────────────────────► Diversity

Greedy / Beam Search              Temperature > 1
(deterministic, repetitive,       (random, creative,
 often degenerate)                 sometimes incoherent)

          ↑
    The sweet spot:
    top-p / top-k at moderate temperature
    (used by ChatGPT, Claude, etc.)
```

### What Decoding Affects

| Property | Decoding Control |
|----------|-----------------|
| Factual accuracy | Lower temperature, greedy/beam for factual recall |
| Creativity | Higher temperature, nucleus sampling |
| Diversity across samples | Temperature + top-p |
| Repetition avoidance | Repetition penalty |
| Output format compliance | Constrained generation |
| Speed | Speculative decoding (see Topic 23) |

---

## 2. Greedy Decoding

### How It Works

At each step, select the single highest-probability token:

$$
y_t = \arg\max_{w \in V} P(w \mid y_{<t}, x)
$$

```
Step 1: logits = [3.2, 1.1, 0.8, -0.4, ...]
        softmax → [0.52, 0.12, 0.09, 0.03, ...]
        pick argmax → token 0 ("Paris")

Step 2: given "Paris", logits = [...]
        pick argmax → token 42 ("is")

...continues deterministically
```

### The Exposure Bias Problem

Greedy decoding always picks the locally best token, but local optimality ≠ global optimality:

```
Sentence: "The bank ___"

Greedy picks: "of" (highest prob for first word)
Continues: "of the river" → reasonable

But the globally best completion might be:
"account balance" (requires picking lower-prob "account" first)
Greedy never reaches this because "account" was not the argmax at step 1.
```

### When to Use

- Factual Q&A with deterministic answers
- Fill-in-the-blank tasks
- Classification-style generation (always want same output for same input)
- Debug mode (reproducible outputs)

---

## 3. Beam Search

### The Idea

Instead of one candidate (greedy), maintain **B candidates (beams)** at each step and keep the top-B by cumulative log-probability:

$$
\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)
$$

### Example with B=2

```
Vocabulary: {cat, dog, sat, slept}
Prompt: "The"

Step 1 — Expand each beam with top tokens:
  Beam 1: "The" → expand:
    "The cat"   log P = -0.5
    "The dog"   log P = -0.7
    "The sat"   log P = -2.1  ← pruned

  Keep top-2: ["The cat" (-0.5), "The dog" (-0.7)]

Step 2 — Expand surviving beams:
  "The cat" → "The cat sat"    log P = -0.5 + (-0.3) = -0.8
  "The cat" → "The cat slept"  log P = -0.5 + (-0.9) = -1.4
  "The dog" → "The dog sat"    log P = -0.7 + (-0.4) = -1.1
  "The dog" → "The dog slept"  log P = -0.7 + (-0.6) = -1.3

  Keep top-2: ["The cat sat" (-0.8), "The dog sat" (-1.1)]

Final: return highest-score complete sequence
```

### Length Normalization

Beam search has a **length bias** — longer sequences accumulate more negative log-probabilities and score lower. Fix with length normalization:

$$
\text{score}_{\text{norm}}(y_{1:T}) = \frac{1}{T^\alpha} \sum_{t=1}^{T} \log P(y_t \mid y_{<t})
$$

where $\alpha \in [0, 1]$ is a length penalty ($\alpha = 0$ = no normalization, $\alpha = 1$ = full normalization). Typical value: $\alpha = 0.6$–$0.7$.

```
Without normalization:
  Short answer (5 tokens):  score = -3.5  → -3.5 / 1 = -3.5
  Long answer  (15 tokens): score = -8.1  → -8.1 / 1 = -8.1  ← loses

With α = 0.6:
  Short answer: -3.5 / 5^0.6 = -3.5 / 2.63 = -1.33
  Long answer:  -8.1 / 15^0.6 = -8.1 / 5.07 = -1.60  ← closer, more fair
```

### Beam Search Limitations

- **Repetition**: High-scoring beams often repeat phrases ("the the the")
- **Generic outputs**: Finds the mode of the distribution → safe, boring text
- **Compute**: B× more expensive than greedy
- **Not used for LLM chat**: Too repetitive and deterministic for open-ended generation

**Used for**: Machine translation (WMT), summarization (when faithfulness > diversity), structured prediction.

---

## 4. Temperature Sampling

### The Formula

Scale the logits by temperature $T$ before softmax:

$$
P_T(y_t = w) = \frac{\exp(z_w / T)}{\sum_{w'} \exp(z_{w'} / T)}
$$

where $z_w$ is the raw logit for token $w$.

### Worked Example

```
Logits for 4 tokens: z = [3.0, 2.0, 1.0, 0.0]

T = 1.0 (standard softmax):
  exp(z) = [20.09, 7.39, 2.72, 1.00]  sum = 31.20
  P = [0.644, 0.237, 0.087, 0.032]
  → "Paris" dominates but others have chance

T = 0.5 (sharper / more confident):
  exp(z/0.5) = exp([6.0, 4.0, 2.0, 0.0]) = [403.4, 54.6, 7.39, 1.00]
  sum = 466.4
  P = [0.865, 0.117, 0.016, 0.002]
  → "Paris" overwhelms; near-greedy

T = 0.1 (almost greedy):
  exp(z/0.1) = exp([30, 20, 10, 0]) = [1.07e13, 4.85e8, 22026, 1]
  P ≈ [1.000, 0.000, 0.000, 0.000]
  → Effectively greedy

T = 2.0 (flatter / more random):
  exp(z/2.0) = exp([1.5, 1.0, 0.5, 0.0]) = [4.48, 2.72, 1.65, 1.00]
  sum = 9.85
  P = [0.455, 0.276, 0.167, 0.102]
  → Much more uniform; rare tokens get significant probability
```

### Effect Summary

```
T → 0   : Greedy (argmax)        → deterministic, repetitive
T = 0.7 : Focused sampling       → good for factual + some creativity (default for many)
T = 1.0 : Raw model distribution → balanced
T > 1   : Flatter distribution   → more creative, more errors
T → ∞   : Uniform distribution   → completely random
```

### Practical Defaults

| Use Case | Temperature |
|----------|------------|
| Factual Q&A, code generation | 0.0–0.3 |
| Chatbot (helpful + natural) | 0.7–1.0 |
| Creative writing, brainstorming | 1.0–1.4 |
| Data augmentation (diverse) | 1.2–2.0 |

---

## 5. Top-k Sampling

### How It Works

Before sampling, keep only the top-k highest-probability tokens and renormalize:

$$
P_k(w) = \begin{cases} P(w) / Z_k & \text{if } w \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}
$$

where $Z_k = \sum_{w \in \text{top-}k} P(w)$.

### Worked Example (k = 3)

```
Full distribution (vocab size 5):
  "Paris"   → 0.50
  "London"  → 0.25
  "Berlin"  → 0.15
  "Tokyo"   → 0.07
  "Rome"    → 0.03

Top-3 mask:
  Keep: "Paris" (0.50), "London" (0.25), "Berlin" (0.15)
  Zero: "Tokyo", "Rome"

Renormalize (Z_k = 0.50 + 0.25 + 0.15 = 0.90):
  "Paris"  → 0.50 / 0.90 = 0.556
  "London" → 0.25 / 0.90 = 0.278
  "Berlin" → 0.15 / 0.90 = 0.167

Sample from {0.556, 0.278, 0.167}
```

### The Problem with Fixed k

```
Scenario A — peaked distribution:
  P = [0.95, 0.03, 0.01, 0.005, 0.005]  ← model is very confident
  With k=50: includes 49 tokens the model almost never wants
  → Unnecessary noise

Scenario B — flat distribution:
  P = [0.05, 0.04, 0.04, 0.04, 0.03, ...]  ← model is uncertain
  With k=10: cuts off many reasonable continuations
  → Artificially restrictive
```

**The fix**: top-p (nucleus) sampling dynamically adapts k based on the distribution shape.

---

## 6. Top-p (Nucleus) Sampling

### How It Works

Keep the **smallest set of tokens** whose cumulative probability exceeds threshold $p$:

$$
V_p = \arg\min_{V' \subseteq V} \left\{ |V'| : \sum_{w \in V'} P(w) \geq p \right\}
$$

Tokens are sorted by probability descending; add tokens until the cumulative mass ≥ p.

### Worked Example (p = 0.9)

```
Sorted probabilities:
  Token     Prob    Cumulative
  "Paris"   0.50    0.50
  "London"  0.25    0.75
  "Berlin"  0.15    0.90  ← cumulative hits 0.90 here → STOP
  "Tokyo"   0.07    0.97
  "Rome"    0.03    1.00

Nucleus = {"Paris", "London", "Berlin"}

Renormalize:
  "Paris"  → 0.50 / 0.90 = 0.556
  "London" → 0.25 / 0.90 = 0.278
  "Berlin" → 0.15 / 0.90 = 0.167
```

### Why Nucleus Sampling Adapts

```
Peaked distribution (p = 0.9):
  "Paris" alone = 0.95 ≥ 0.9 → nucleus = 1 token (near-greedy)

Flat distribution (p = 0.9):
  Need 20 tokens to reach 0.90 cumulative → nucleus = 20 tokens
```

The nucleus size automatically adjusts: small when the model is confident, large when uncertain. This is why top-p outperforms top-k in practice.

### Combining Temperature + Top-p

**Common in production**: Apply temperature first (reshape distribution), then apply top-p:

```
Step 1: raw logits → apply T=0.8 → new probabilities
Step 2: apply top-p=0.9 to the temperature-adjusted distribution
Step 3: sample

This is the default for most LLM APIs (OpenAI, Anthropic, etc.)
```

---

## 7. Min-p Sampling

### The Idea (2024)

Min-p sets a **relative minimum probability threshold** based on the top token's probability:

$$
P_{\min}(w) = p_{\min} \times \max_{w'} P(w')
$$

Any token with probability below $P_{\min}$ is filtered out.

### Example (p_min = 0.05)

```
Top token probability: P("Paris") = 0.60
Threshold: 0.05 × 0.60 = 0.030

Keep tokens with P(w) ≥ 0.030:
  "Paris"  0.60 ✓
  "London" 0.25 ✓
  "Berlin" 0.08 ✓
  "Tokyo"  0.04 ✓
  "Rome"   0.02 ✗  (0.02 < 0.030)

If top token = 0.95 (very confident):
  Threshold = 0.05 × 0.95 = 0.0475 → only very likely tokens pass
  
If top token = 0.15 (very uncertain):
  Threshold = 0.05 × 0.15 = 0.0075 → many tokens pass (more diverse)
```

**Advantage over top-p**: Self-calibrating based on model confidence at each step rather than absolute cumulative mass.

---

## 8. Repetition & Frequency Penalties

### The Repetition Problem

Without penalties, LLMs often generate repetitive text:
```
"The cat sat on the mat. The cat sat on the mat. The cat sat..."
```
This happens because the model's probability of a token increases after it was generated — it's in the context now.

### Presence Penalty

If a token has appeared in the generated text at all, reduce its logit by a fixed amount $\alpha$:

$$
z'_w = z_w - \alpha \cdot \mathbb{1}[w \in \text{generated tokens}]
$$

Binary — same penalty regardless of how many times the token appeared.

### Frequency Penalty

Scale the penalty by how many times the token has appeared:

$$
z'_w = z_w - \beta \cdot \text{count}(w, \text{generated tokens})
$$

Penalizes repeated tokens more the more they appear.

### Repetition Penalty (Multiplicative)

Used in HuggingFace transformers:

$$
z'_w = \begin{cases} z_w / \theta & \text{if } z_w > 0 \text{ and } w \in \text{seen} \\ z_w \times \theta & \text{if } z_w < 0 \text{ and } w \in \text{seen} \end{cases}
$$

where $\theta > 1$ (typically 1.1–1.3). Positive logits are divided (reduced), negative logits are multiplied (made more negative).

### Worked Example (θ = 1.2)

```
Token "the" appears 3 times already.
Original logit: z = 2.5 (positive, high probability)
After penalty:  z' = 2.5 / 1.2 = 2.08

Token "quantum" never appeared.
Original logit: z = 1.8
After penalty:  z' = 1.8 (unchanged)

Effect: "the" less likely to be selected again → less repetition
```

---

## 9. Contrastive Decoding

### The Motivation

LLMs sometimes generate generic, low-information text. A **weak (smaller) model** is even more likely to generate this generic text — it can serve as a "negative" signal.

**Key idea** (Li et al., 2022): Good tokens are those the **expert (large) model** prefers over the **amateur (small) model**:

$$
\text{CD score}(w) = \log P_{\text{expert}}(w | x) - \log P_{\text{amateur}}(w | x)
$$

### How It Works

```
Token "Paris":
  Expert model (70B):  P = 0.82  → log P = -0.20
  Amateur model (7B):  P = 0.65  → log P = -0.43
  CD score = -0.20 - (-0.43) = +0.23  ← expert strongly prefers this

Token "city" (generic):
  Expert model (70B):  P = 0.12  → log P = -2.12
  Amateur model (7B):  P = 0.20  → log P = -1.61
  CD score = -2.12 - (-1.61) = -0.51  ← amateur likes this more → penalize
```

**Effect**: Filters out generic tokens that both models agree on, amplifying the expert's unique strengths.

### Adaptive Plausibility Constraint

Raw contrastive decoding can amplify errors. Apply a plausibility mask first:

$$
V_{\text{CD}} = \{w : P_{\text{expert}}(w) \geq \alpha \cdot \max_{w'} P_{\text{expert}}(w')\}
$$

Only consider tokens the expert model finds plausible (e.g., $\alpha = 0.1$), then pick the one with the highest CD score from this set.

### When Contrastive Decoding Helps

- Long-form generation (articles, stories) — reduces generic phrases
- Open-domain QA — improves factual precision
- Less useful for short responses where generic text isn't a problem

---

## 10. Constrained & Structured Generation

### The Problem

Production systems often need **structured outputs** (JSON, XML, code). Standard sampling may generate malformed JSON 5–30% of the time.

### Token Masking Approach

At each step, mask out tokens that would make the output invalid given the current partially-generated structure:

```
Generating JSON: {"name": "
Current state: we are inside a string value
Valid next tokens: any character token, closing quote "
Invalid: } ] : { (would break JSON structure)

Apply mask → set logits of invalid tokens to -∞ → they get zero probability after softmax
```

### Finite-State Machine Approach

Model the JSON grammar as a finite-state machine. At each state, only certain tokens are valid transitions:

```
State: EXPECT_VALUE
  Valid tokens: " (string start), 0-9 (number start), t (true), f (false), n (null), [ (array), { (object)
  
State: IN_STRING  
  Valid tokens: any except unescaped "
  
State: EXPECT_COMMA_OR_CLOSE
  Valid tokens: , (more items) or } (close object)
```

### Implementation in Practice

- **Outlines** (open-source): FSM-based, regex and JSON schema support
- **llama.cpp grammar**: BNF grammar specification
- **OpenAI Structured Outputs**: Server-side token masking with JSON Schema
- **LMQL**: Full query language for constrained generation

### Constrained Generation for Code

```
Target: valid Python function
Constraints:
  - Must start with "def "
  - Parentheses must be balanced
  - Indentation must be consistent (multiples of 4 spaces)

Each token generated is checked against the grammar;
invalid tokens are masked before sampling.
```

---

## 11. Decoding Strategy Comparison

```
Strategy        Deterministic?  Quality   Diversity  Speed    Use Case
──────────────────────────────────────────────────────────────────────
Greedy          Yes             Medium    None       Fastest  Factual QA, classification
Beam Search     Yes             High      None       Slow     Translation, summarization
Temperature     No              Variable  Tunable    Fast     General chat
Top-k           No              Good      Medium     Fast     General chat
Top-p           No              Good      Adaptive   Fast     Default for most LLMs
Top-p + Temp    No              Best      Tunable    Fast     Production standard
Min-p           No              Good      Adaptive   Fast     High-quality generation
Contrastive     No              Highest   Low        2× slow  Long-form, factual
Constrained     Mixed           N/A       N/A        Varies   JSON/code output
```

### Production Defaults (what major APIs actually use)

| Provider | Default Strategy |
|----------|-----------------|
| OpenAI (GPT-4o) | temperature=1.0, top_p=1.0 (but internally tuned) |
| Anthropic (Claude) | temperature=1.0 (user-exposed), top_p and top_k available |
| Google (Gemini) | temperature=0.9, top_p=1.0, top_k=none |
| HuggingFace (most models) | do_sample=False (greedy) by default |

### The Diversity–Quality Frontier

```
Diversity
  ▲
  │  T=2.0 (random)
  │    ×
  │         ×  T=1.5
  │              ×  top-p=0.95, T=1.0
  │                    ×  top-p=0.9, T=0.8  ← sweet spot for chat
  │                          ×  beam search
  │                               ×  greedy
  └────────────────────────────────────────► Quality (factual, coherent)
```

---

## 12. Interview Questions & Answers

### Q1: What is temperature in LLM sampling? What happens at T=0, T=1, and T>1?

**Answer**: Temperature $T$ scales the logits before softmax: $P_T(w) \propto \exp(z_w / T)$. At **T→0**: dividing by a tiny number makes large logits enormous relative to small ones → distribution concentrates entirely on the argmax → equivalent to greedy decoding, deterministic. At **T=1**: standard softmax, raw model probabilities. At **T>1**: dividing by a number > 1 brings logits closer together → distribution flattens → rare tokens get relatively more probability → more diverse, creative, but also more likely to generate errors. Intuition: temperature controls the model's "confidence" — low T makes the model more decisive, high T makes it more exploratory.

### Q2: What is nucleus (top-p) sampling? Why is it preferred over top-k?

**Answer**: Top-p (nucleus) sampling keeps the smallest set of tokens whose cumulative probability ≥ p, then renormalizes and samples. E.g., p=0.9: if "Paris" alone has 0.95 probability, the nucleus is just {"Paris"} (near-greedy). If the distribution is flat (10 tokens each with 0.10), the nucleus includes all 10. **Why it's better than top-k**: top-k uses a fixed cutoff (always keep exactly k tokens) regardless of the distribution shape. When the model is very confident, k=50 includes 49 nearly-impossible tokens, adding noise. When the model is uncertain, k=50 may cut off many reasonable options. Top-p adapts: small nucleus when confident, large nucleus when uncertain. This dynamic adaptation to the model's confidence at each step is why top-p is the production standard.

### Q3: You're building a chatbot. Walk through how you'd set temperature, top-k, and top-p.

**Answer**: For a helpful chatbot, I'd use **temperature=0.7–0.8** (focused but not robotic), **top-p=0.9** (adaptive nucleus), and no top-k (top-p is sufficient). Rationale: temperature 0.7 prevents the flatness of T=1.0 that allows rare nonsensical tokens, while avoiding the boring determinism of T<0.3. Top-p=0.9 ensures we never sample truly improbable tokens while adapting to the model's confidence at each step. For a **creative writing chatbot**, I'd raise temperature to 1.0–1.2 and possibly top-p to 0.95 to allow more diversity. For a **code generation assistant**, I'd lower temperature to 0.2–0.4 (code needs to be syntactically correct) or use constrained generation with the language grammar. For **factual Q&A**, temperature=0 (greedy) or beam search gives the most consistent, deterministic answers.

### Q4: What is beam search length normalization? Why is it necessary?

**Answer**: Beam search scores sequences by their cumulative log-probability: $\sum_t \log P(y_t)$. Since all log-probabilities are negative, longer sequences always score lower than shorter ones, creating a **length bias** — beam search unfairly prefers shorter sequences. Length normalization divides by $T^\alpha$ where $T$ is sequence length and $\alpha \in [0,1]$: $\text{score} = \frac{1}{T^\alpha} \sum_t \log P(y_t)$. At $\alpha=0$: no normalization (length bias). At $\alpha=1$: full normalization (averages log-probability, favors longer sequences). Typical $\alpha=0.6$: partial normalization balancing the two extremes. Without normalization, a translation model would always produce "Yes" or "No" instead of complete sentences. With over-normalization ($\alpha=1$), verbose outputs are unfairly preferred.

### Q5: What is contrastive decoding? When would you use it?

**Answer**: Contrastive decoding (Li et al., 2022) scores tokens by the difference in log-probability between an expert (large) model and an amateur (small) model: $\text{CD}(w) = \log P_{\text{expert}}(w) - \log P_{\text{amateur}}(w)$. Tokens that the expert strongly prefers over the amateur are promoted; generic tokens that both models agree on (like filler phrases or common stopwords) are penalized because the amateur assigns them similar probability. **When to use**: long-form generation tasks where generic, repetitive, or uninformative text is a problem (essays, articles, detailed explanations). **When not to**: short responses, factual Q&A, code (where correctness > novelty, and the overhead of running two models is wasteful). Downside: 2× inference cost since you run both models at every step.

### Q6: How do you force an LLM to output valid JSON? What is the token masking approach?

**Answer**: At each decoding step, maintain a state machine representing the current position in the JSON grammar. At each state, compute which tokens are syntactically valid continuations and set all other tokens' logits to $-\infty$ before sampling — they get zero probability after softmax. Example: if we're in state "inside string value", we block `}`, `]`, `:`, `{` (unescaped) since they'd break JSON syntax. Only character tokens and closing quotes are valid. This ensures 100% grammatically valid JSON regardless of what the model "wants" to generate. The grammar constraints don't prevent semantic errors (wrong field values) but guarantee structural validity. Libraries like Outlines implement this with regular expressions and JSON Schema; OpenAI's Structured Outputs mode does this server-side.
