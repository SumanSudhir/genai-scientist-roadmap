# Topic 13: Encoder-Decoder Models & Unified Frameworks

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 13 of 28
> **Scope**: T5 (text-to-text, span corruption, C4), BART (denoising autoencoder), mBART, UL2 (unified pretraining), Prefix LM, the encoder-decoder vs decoder-only debate, why decoder-only won
> **Why this matters**: Encoder-decoder models represent a fundamentally different design philosophy — one that was the dominant paradigm for conditional generation tasks (translation, summarization) before decoder-only models took over. Understanding T5 and BART shows interviewers you can reason about architectural trade-offs, not just recite the latest model. The "why did decoder-only win?" question is a favorite in AI Scientist interviews because it tests deep understanding of scaling, training objectives, and emergent capabilities.
> **Previous**: [Topic 12: Decoder Models (GPT & Open-Source LLMs)](12_GPT_Open_Source_LLMs.md)
> **Next**: [Topic 14: Pretraining LLMs](14_Pretraining.md)

---

## Table of Contents

1. [The Encoder-Decoder Philosophy](#1-the-encoder-decoder-philosophy)
2. [T5 — Text-to-Text Transfer Transformer](#2-t5--text-to-text-transfer-transformer)
3. [T5's Pretraining: Span Corruption](#3-t5s-pretraining-span-corruption)
4. [T5's Systematic Exploration](#4-t5s-systematic-exploration)
5. [BART — Denoising Sequence-to-Sequence](#5-bart--denoising-sequence-to-sequence)
6. [Prefix LM — The Hybrid Approach](#6-prefix-lm--the-hybrid-approach)
7. [Flan-T5 & Instruction-Tuned Encoder-Decoders](#7-flan-t5--instruction-tuned-encoder-decoders)
8. [The Great Debate: Why Decoder-Only Won](#8-the-great-debate-why-decoder-only-won)
9. [When Encoder-Decoder Still Wins](#9-when-encoder-decoder-still-wins)
10. [Interview Questions & Answers](#10-interview-questions--answers)

---

## 1. The Encoder-Decoder Philosophy

### 1.1 The Core Idea

Encoder-decoder models separate the **understanding** of the input from the **generation** of the output:

$$
P(y_1, \ldots, y_m \mid x_1, \ldots, x_n) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, \text{Enc}(x_1, \ldots, x_n))
$$

- **Encoder**: Processes the full input bidirectionally → rich contextual representations
- **Decoder**: Generates the output autoregressively, attending to the encoder via cross-attention

This is the original transformer design from "Attention Is All You Need" (see [Topic 9](09_Transformer_Architecture.md)).

### 1.2 Why Separate Encoder and Decoder?

The intuition: **understanding and generation are different skills**.

**Understanding** (encoder's job):
- Benefits from **bidirectional** context — seeing both left and right of every token
- Needs to build complete representations of the *entire* input before any output is generated
- Doesn't need to be autoregressive — can process all positions in parallel

**Generation** (decoder's job):
- Must be **autoregressive** — each output token depends on previous output tokens
- Needs **cross-attention** to selectively read from the input representation
- Output length may differ from input length (translation: different word count; summarization: much shorter)

### 1.3 The Three Attention Patterns (Recap)

```
Encoder Self-Attention:        Bidirectional (every token ↔ every token)
████████
████████
████████
████████

Decoder Masked Self-Attention: Causal (each token → only previous tokens)
█░░░░░░░
██░░░░░░
███░░░░░
████░░░░

Decoder Cross-Attention:       Full (every decoder token → every encoder token)
████████
████████
████████
████████
(rows = decoder positions, columns = encoder positions)
```

### 1.4 Encoder-Decoder vs Decoder-Only: Structural Comparison

For a task with input length $n$ and output length $m$:

**Encoder-Decoder**:
- Encoder: $O(n^2)$ attention (bidirectional, full input)
- Decoder: $O(m^2)$ self-attention + $O(mn)$ cross-attention per layer
- Input processed once; decoder reads from cached encoder output

**Decoder-Only** (input + output concatenated):
- Total: $O((n+m)^2)$ causal attention
- Input processed together with output; reprocessed (via KV cache) during generation

**When encoder-decoder is structurally better**: When $n \gg m$ (long input, short output) — the encoder processes the long input once with efficient bidirectional attention, and the decoder generates the short output while cross-attending to it. For summarization of a 10K-token document into a 200-token summary, this is more efficient than a decoder-only model processing 10,200 tokens causally.

---

## 2. T5 — Text-to-Text Transfer Transformer

### 2.1 The Unifying Idea

T5 (Raffel et al., 2020) proposed that **every NLP task can be framed as text-to-text**: the model takes a text input and produces a text output.

$$
\text{input text} \longrightarrow \text{T5} \longrightarrow \text{output text}
$$

Examples:

```
Translation:
  Input:  "translate English to German: That is good."
  Output: "Das ist gut."

Summarization:
  Input:  "summarize: state authorities dispatched emergency crews..."
  Output: "emergency crews were dispatched after a storm."

Classification (sentiment):
  Input:  "sst2 sentence: this movie was terrible."
  Output: "negative"

Regression (similarity):
  Input:  "stsb sentence1: The cat sat. sentence2: A cat was sitting."
  Output: "4.2"

Question Answering:
  Input:  "question: Who is the president? context: Joe Biden is the president..."
  Output: "Joe Biden"
```

**Why this matters**: A single model, a single training procedure, and a single decoding process handles *every* NLP task. No task-specific heads, no custom architectures — just text in, text out.

### 2.2 Architecture

T5 is a standard encoder-decoder transformer with minor modifications:

| Parameter | T5-Small | T5-Base | T5-Large | T5-3B | T5-11B |
|-----------|----------|---------|----------|-------|--------|
| Encoder layers | 6 | 12 | 24 | 24 | 24 |
| Decoder layers | 6 | 12 | 24 | 24 | 24 |
| $d_{\text{model}}$ | 512 | 768 | 1024 | 1024 | 1024 |
| $d_{ff}$ | 2048 | 3072 | 4096 | 16384 | 65536 |
| Heads | 8 | 12 | 16 | 32 | 128 |
| Parameters | 60M | 220M | 770M | 3B | 11B |

**Design choices**:
- **Pre-LN** (LayerNorm before attention and FFN)
- **Relative position bias** (learned, bucketed — not sinusoidal, not RoPE; see [Topic 10](10_Positional_Encodings.md))
- **No bias terms** in linear layers
- **ReLU activation** (not GELU or SwiGLU — T5 predates their adoption)
- **Shared encoder-decoder embeddings** (weight tying)
- **SentencePiece** vocabulary with 32,000 tokens

### 2.3 Relative Position Bias in T5

T5 uses a unique positional encoding: **learned relative position biases** added directly to attention scores.

Relative distances are mapped to **logarithmic buckets**:

$$
\text{score}_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + b_h[\text{bucket}(i - j)]
$$

where $b_h$ is a learned bias table per head $h$, and $\text{bucket}(\cdot)$ maps distances to 32 buckets:

- Small distances ($|i-j| \leq 8$): One bucket per distance (fine-grained)
- Larger distances: Logarithmic bucketing (coarser as distance increases)
- Beyond max distance: All mapped to the same bucket

**Advantages**: Learned, flexible, relative. **Disadvantages**: Finite number of buckets limits extrapolation; less elegant than RoPE.

---

## 3. T5's Pretraining: Span Corruption

### 3.1 The Objective

T5's pretraining objective is **span corruption** (also called "span denoising"): randomly mask contiguous spans of tokens and train the model to generate only the missing spans.

**Input**: Original text with spans replaced by sentinel tokens `<X>`, `<Y>`, `<Z>`, ...

**Target**: Only the masked content, prefixed by its corresponding sentinel token.

```
Original:  "Thank you for inviting me to your party last week."

Input:     "Thank you <X> me to your party <Y> week."
Target:    "<X> for inviting <Y> last <Z>"
```

where `<X>`, `<Y>`, `<Z>` are special sentinel tokens (unique per span).

### 3.2 Mathematical Formulation

Let $\mathbf{x}$ be the original sequence and $\mathcal{S}$ be the set of masked spans. The training loss:

$$
\mathcal{L} = -\sum_{s \in \mathcal{S}} \sum_{t \in s} \log P(x_t \mid x_{\text{before } t \text{ in target}}, \text{Enc}(\tilde{\mathbf{x}}))
$$

where $\tilde{\mathbf{x}}$ is the corrupted input (with sentinel tokens).

### 3.3 Masking Parameters

The T5 paper systematically tested these:

| Parameter | T5's Choice |
|-----------|------------|
| Corruption rate | **15%** of tokens masked |
| Average span length | **3** tokens |
| Number of spans | $\approx 0.15n / 3 = 0.05n$ (5% of sequence length) |

### 3.4 Span Corruption vs MLM vs CLM

| Aspect | MLM (BERT) | Span Corruption (T5) | CLM (GPT) |
|--------|-----------|---------------------|-----------|
| **What's masked** | Individual tokens | Contiguous spans | Nothing (predict next) |
| **Input** | Full sequence with [MASK] | Corrupted sequence with sentinels | Left context |
| **Target** | Original masked tokens | Only the span contents | Every next token |
| **Architecture** | Encoder-only | Encoder-decoder | Decoder-only |
| **Bidirectional context** | Yes (encoder) | Yes (encoder) | No (causal) |
| **Target length** | Same as masked count | Shorter than input (only spans) | Same as input |
| **Training efficiency** | 15% tokens | ~15% tokens (but target is very short) | 100% tokens |

**Why span corruption for encoder-decoder?**

1. **Natural for seq2seq**: The encoder reads the corrupted input; the decoder generates the missing pieces. This matches the model's architecture — encoder understands, decoder generates.

2. **Short targets**: The decoder only generates the masked content (~15% of the original), making training much more efficient than having the decoder reproduce the entire input.

3. **Span-level reasoning**: Masking spans (not individual tokens) forces the model to predict coherent multi-token phrases, learning better compositional understanding.

### 3.5 Why Not Use CLM for Encoder-Decoder?

You could train an encoder-decoder with CLM by feeding the full input to the encoder and generating the full output with the decoder. But this would:
- Duplicate effort (the decoder essentially regenerates what the encoder already processed)
- Provide no pretraining signal for the cross-attention mechanism
- Lose the efficiency of short target sequences

Span corruption naturally trains all three components: encoder (reads corrupted input), cross-attention (decoder reads encoder output to fill spans), and decoder (generates span content autoregressively).

### 3.6 T5 Span Corruption — Visual Walkthrough

Original sentence: "The quick brown fox jumps over the lazy dog"

**Step 1**: Randomly select spans (avg length 3, ~15% of tokens masked):

Spans selected: ["quick brown"] at positions 1-2 and ["lazy"] at position 7

**Step 2**: Replace each span with a single sentinel token:

```
Input:  "The <X> fox jumps over the <Y> dog"
Target: "<X> quick brown <Y> lazy <EOS>"
```

**Why this design**:
- Corrupting SPANS (not individual tokens) makes the task harder → richer representations
- Sentinel tokens encode "which gap" so the decoder knows where each span goes
- Predicting only the masked spans (not the full sequence) is more compute-efficient than full seq2seq
- T5 can denoise multiple spans in one forward pass

**Comparison to BERT**:

```
BERT:  [CLS] The [MASK] brown [MASK] jumps ... [SEP]  → predict individual tokens
T5:    "The <X> fox jumps ..."  → generate "quick brown" for <X>
```

T5's span corruption creates harder reconstruction tasks and forces the model to understand longer-range dependencies within the corrupted span.

---

## 4. T5's Systematic Exploration

One of T5's greatest contributions was its **systematic ablation study** — testing dozens of design choices to identify what matters. This is a model for how AI research should be conducted.

### 4.1 What They Tested

The paper explored a combinatorial space of design choices:

#### Pretraining Objectives Tested

| Objective | Description | Result |
|-----------|-------------|--------|
| **Prefix LM** | Bidirectional on prefix, causal on suffix | Decent but not best |
| **BERT-style MLM** (adapted for enc-dec) | Predict masked tokens | Good for understanding tasks |
| **Deshuffling** | Reconstruct shuffled sentence | Poor |
| **Span corruption** (various configs) | Predict masked spans | **Best overall** |

#### Corruption Strategies Tested

| Strategy | Description | Result |
|----------|-------------|--------|
| Individual tokens | BERT-style random token masking | Good |
| Spans (mean length 3) | Mask contiguous spans | **Best** |
| Spans (mean length 5) | Longer spans | Slightly worse |
| Spans (mean length 10) | Very long spans | Worse (too hard) |

#### Corruption Rates Tested

| Rate | Result |
|------|--------|
| 10% | Slightly undertrained |
| **15%** | **Best** |
| 25% | Slightly worse (too much corruption) |
| 50% | Significantly worse |

#### Architecture Variants Tested

| Variant | Description | Result |
|---------|-------------|--------|
| **Encoder-decoder** | Separate encoder and decoder | **Best** for same compute budget |
| Decoder-only (causal LM) | GPT-style | Worse at same param count |
| Prefix LM | Shared params, bidirectional prefix | Close to enc-dec |
| Shared encoder-decoder | Share all weights between enc and dec | Slightly worse than unshared |

### 4.2 The Key Finding on Architecture

At the same number of parameters and the same compute budget:

$$
\text{Encoder-Decoder} > \text{Prefix LM} > \text{Decoder-Only}
$$

**Why?** The encoder-decoder has a crucial advantage: the encoder uses **bidirectional attention** (no causal mask). For a model with $L$ total layers split as $L/2$ encoder + $L/2$ decoder, the encoder provides $L/2$ layers of bidirectional processing — strictly more powerful than $L/2$ layers of causal processing.

**But the comparison is nuanced**: A fair comparison should use the same **FLOPs**, not the same **parameters**. An encoder-decoder with $L$ encoder layers and $L$ decoder layers has $\sim 2L$ layers of computation. A decoder-only model with $2L$ layers uses similar FLOPs but has $\sim 1.5\times$ the parameters (no cross-attention needed). The T5 paper acknowledged this complexity.

### 4.3 Other Key Findings

1. **More data always helps**: Training on more data (with the same compute budget per example) consistently improved performance.
2. **Larger models are more sample-efficient**: Larger models extract more learning per training token.
3. **Multi-task pretraining hurts**: Mixing supervised task data into pretraining (before fine-tuning) didn't help — unsupervised pretraining was sufficient.
4. **Fine-tuning all parameters works best**: Compared to adapter layers or only fine-tuning the output head, updating all parameters gave the best results.

### 4.4 T5 Architecture Search — Key Findings Table

T5's main contribution was systematic ablations. Summary of key findings:

| What was tested | Winner | Why it matters |
|-----------------|--------|----------------|
| Architecture | Encoder-decoder > decoder-only (for supervised tasks) | But decoder-only won later for generality |
| Pre-training objective | Span corruption > MLM > CLM | Spans create harder tasks |
| Scale | More is better | Confirmed scaling laws |
| Multi-task training | Helps on average, hurts some tasks | Task-specific fine-tuning still best |
| Vocab size | Larger helps up to ~32K | Diminishing returns beyond |

**T5 family sizes** (for interview memory):

```
T5-small:    60M params
T5-base:    220M params
T5-large:   770M params
T5-XL:        3B params
T5-XXL:      11B params
Flan-T5-XXL: 11B params  (instruction-tuned, much better at following instructions)
```

---

## 5. BART — Denoising Sequence-to-Sequence

### 5.1 The Idea

BART (Lewis et al., 2020) generalizes the denoising pretraining idea: corrupt the input with various noise functions, then train the encoder-decoder to reconstruct the original.

$$
\text{Corrupted input} \xrightarrow{\text{Encoder}} \text{Representations} \xrightarrow{\text{Decoder}} \text{Original input}
$$

Unlike T5 (which only generates the missing spans), BART's decoder reconstructs the **entire original sequence**.

### 5.2 Corruption Strategies

BART tested five corruption strategies:

| Strategy | Description | Example |
|----------|-------------|---------|
| **Token Masking** | Replace random tokens with [MASK] | "The cat [MASK] on the [MASK]" |
| **Token Deletion** | Delete random tokens entirely | "The sat on mat" |
| **Text Infilling** | Replace spans with a single [MASK] | "The [MASK] the mat" (span "cat sat on" → one mask) |
| **Sentence Permutation** | Shuffle sentence order | "On the mat. The cat sat." |
| **Document Rotation** | Rotate document to start at a random token | "on the mat. The cat sat" |

### 5.3 BART vs T5: Key Differences

| Aspect | T5 | BART |
|--------|-----|------|
| **Decoder target** | Only the masked spans (short target) | **Entire original sequence** (full reconstruction) |
| **Corruption** | Span corruption only | Multiple strategies (text infilling best) |
| **Input to encoder** | Corrupted text with sentinel tokens | Corrupted text with [MASK] or deletions |
| **Positional encoding** | Relative position bias | Learned absolute |
| **Activation** | ReLU | GELU |
| **Base model** | Custom architecture | Same config as RoBERTa (for fair comparison) |

### 5.4 BART's Best Configuration

After testing all corruption strategies, the best combination was:

1. **Text infilling** with $\lambda = 3$ (Poisson distribution for span lengths, mean = 3)
2. **Sentence permutation** (shuffle sentence order)

Text infilling was the single most important corruption — it forces the model to determine both *what* was removed and *how long* the removed span was (since one [MASK] can replace any-length span).

### 5.5 BART for Generation Tasks

BART excels at generation tasks because its decoder is trained to produce fluent, complete text:

**Summarization**: Input the document to the encoder; the decoder generates the summary. BART achieved state-of-the-art on CNN/DailyMail and XSum.

**Machine Translation**: BART can be adapted by replacing the encoder's embedding layer with a randomly initialized encoder for the source language, then fine-tuning the full model.

**Abstractive QA**: Given context + question, generate the answer.

### 5.6 BART vs BERT for Understanding Tasks

BART can also be used for understanding tasks:

- **Classification**: Use the final decoder token's representation (analogous to BERT's [CLS])
- **Span extraction**: The decoder generates the answer span as text

But BART is generally **not preferred** for understanding tasks — it's more complex than BERT and doesn't offer significant advantages for classification or NER. BART's strength is in **generation**.

---

## 6. Prefix LM — The Hybrid Approach

### 6.1 The Idea

A Prefix LM uses a single transformer (no separate encoder/decoder), but treats the input differently from the output:

- **Prefix** (the input): Bidirectional attention — all input tokens can attend to each other
- **Suffix** (the output): Causal attention — each output token attends to all input tokens and previous output tokens

```
Attention mask for Prefix LM (prefix = 4 tokens, suffix = 3 tokens):

         Prefix    Suffix
     ┌─── p1 p2 p3 p4 │ s1 s2 s3
  p1 │  ✓  ✓  ✓  ✓  │  ✗  ✗  ✗
  p2 │  ✓  ✓  ✓  ✓  │  ✗  ✗  ✗
  p3 │  ✓  ✓  ✓  ✓  │  ✗  ✗  ✗
  p4 │  ✓  ✓  ✓  ✓  │  ✗  ✗  ✗
  ───┼──────────────┼──────────
  s1 │  ✓  ✓  ✓  ✓  │  ✓  ✗  ✗
  s2 │  ✓  ✓  ✓  ✓  │  ✓  ✓  ✗
  s3 │  ✓  ✓  ✓  ✓  │  ✓  ✓  ✓
```

### 6.2 Advantages

1. **Bidirectional encoding**: Like an encoder-decoder, the prefix tokens get bidirectional context
2. **Shared parameters**: Unlike encoder-decoder, all layers share the same weights (no separate encoder/decoder parameters). This is more parameter-efficient
3. **Simpler architecture**: Just one transformer stack with a modified attention mask

### 6.3 Who Uses Prefix LM

| Model | Type | Notes |
|-------|------|-------|
| UniLM (Microsoft) | Prefix LM | Early unified model |
| PaLM (Google, partially) | Can operate as Prefix LM | Flexible attention masking |
| U-PaLM | Prefix LM variant | UL2 applied to PaLM |
| GLM (Tsinghua) | Prefix LM with autoregressive blank infilling | Chinese LLM foundation |

### 6.4 Prefix LM vs Encoder-Decoder vs Decoder-Only

| Aspect | Encoder-Decoder | Prefix LM | Decoder-Only |
|--------|----------------|-----------|-------------|
| Attention on input | Bidirectional | Bidirectional | Causal |
| Attention on output | Causal + cross-attn | Causal | Causal |
| Parameters | 2 stacks (more params per FLOP) | 1 stack (shared) | 1 stack |
| Flexibility | Fixed input/output split | Flexible prefix length | Fully causal |
| Best for | Conditional generation | Conditional generation | Everything (at scale) |

**T5's finding**: Encoder-decoder slightly outperforms Prefix LM at the same compute budget, because having dedicated encoder layers (with separate parameters) gives more modeling capacity than shared layers.

---

## 7. Flan-T5 & Instruction-Tuned Encoder-Decoders

### 7.1 The Flan Recipe

**Flan** (Fine-tuned Language Net) is a methodology, not a model. It involves:

1. Take a pretrained model (T5, PaLM, UL2)
2. Fine-tune it on a massive collection of instruction-following tasks
3. The resulting model follows instructions much better

### 7.2 Flan-T5

Flan-T5 (Chung et al., 2022) applied the Flan recipe to T5:

- **1,836 tasks** organized into task clusters
- **Templates**: Each task is presented in multiple natural language templates
- **Chain-of-thought**: Some examples include step-by-step reasoning
- **Sizes**: Flan-T5-Small (80M) to Flan-T5-XXL (11B)

### 7.3 Results

Flan-T5 dramatically improved T5's zero-shot and few-shot performance:

| Model | Zero-shot MMLU | Few-shot MMLU |
|-------|---------------|---------------|
| T5-XXL (11B) | ~30% | ~45% |
| Flan-T5-XXL (11B) | ~50% | ~55% |
| GPT-3 (175B) | ~43% | ~53% |

**Flan-T5-XXL (11B) matched GPT-3 (175B)** on several benchmarks — a 16× parameter efficiency advantage. This showed that instruction tuning is enormously valuable.

### 7.4 Why Instruction Tuning Matters for Encoder-Decoders

Without instruction tuning, T5 only knows to continue text in the span-corruption style. It doesn't naturally follow instructions like "Summarize this article" or "Translate to French."

Instruction tuning teaches the model to:
1. Parse natural language instructions
2. Map instructions to the appropriate behavior
3. Generate appropriate output format

This bridges the gap between T5's pretraining (span filling) and real-world usage (following instructions).

---

## 8. The Great Debate: Why Decoder-Only Won

This section addresses one of the most important questions in modern NLP: why did decoder-only models become the dominant paradigm, despite encoder-decoder models having structural advantages for many tasks?

### 8.1 The Convergence

```
2017-2019: Three paradigms competing
├── Encoder-only (BERT)     → Classification, embeddings
├── Encoder-decoder (T5)    → Translation, summarization
└── Decoder-only (GPT)      → Generation

2020-2022: Decoder-only begins dominating
├── GPT-3 shows ICL
├── Scaling laws favor simplicity
└── Emergent abilities appear in decoder-only first

2023-2026: Decoder-only is default
├── GPT-4, Llama, Mistral, Claude → All decoder-only
├── Encoder-only survives (embeddings, classification)
└── Encoder-decoder fading (limited to T5-based systems)
```

### 8.2 The Seven Reasons Decoder-Only Won

#### Reason 1: Simplicity Scales

Decoder-only has **one attention pattern** (causal) and **one training objective** (next-token prediction). Encoder-decoder has three attention patterns (bidirectional, causal, cross-attention) and a more complex training setup.

Simplicity matters because:
- Fewer engineering decisions → fewer places to make mistakes at scale
- Simpler to optimize for distributed training across thousands of GPUs
- Simpler to optimize for inference (KV cache is straightforward)

#### Reason 2: In-Context Learning Emerged in Decoder-Only First

GPT-3's in-context learning was a game-changer: the model could perform new tasks from examples in the prompt, **without any fine-tuning**.

Encoder-decoder models showed ICL too (Flan-T5), but:
- It emerged less reliably
- Required instruction tuning to activate
- Was generally weaker than in decoder-only models of equivalent compute

**Why?** The causal pretraining objective naturally teaches the model to extract patterns from context and apply them — this is essentially what next-token prediction does. The model learns to "learn from its context" as a byproduct of predicting next tokens.

#### Reason 3: Universality

A decoder-only model can handle **any** task by generating the appropriate output:

- Classification: Generate the label ("positive", "negative")
- Translation: Generate in the target language
- Summarization: Generate the summary
- Code: Generate code
- Reasoning: Generate step-by-step

An encoder-decoder model technically can too, but the text-to-text framing feels more natural for decoder-only (just keep generating) than for encoder-decoder (must define what's "input" vs "output").

#### Reason 4: Scaling Laws Are Better Understood

Kaplan (2020) and Chinchilla (2022) scaling laws were derived for decoder-only (causal LM) models. The compute-performance relationship is well-characterized:

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{0.076} + \left(\frac{D_c}{D}\right)^{0.095} + L_{\infty}
$$

Encoder-decoder scaling laws are less studied and less clean, making it harder to plan billion-dollar training runs.

#### Reason 5: Training Efficiency

CLM provides training signal on **100%** of tokens. Span corruption provides signal on only ~15% of tokens (the masked spans). For the same compute budget, CLM extracts more learning:

$$
\text{Effective training signal: CLM} = n \text{ tokens per sequence}
$$

$$
\text{Effective training signal: span corruption} = 0.15n \text{ tokens per sequence}
$$

To match CLM's training signal, span corruption needs ~6.7× more data — a significant disadvantage at scale.

#### Reason 6: Inference Simplicity

Decoder-only inference is simple: maintain a KV cache, generate one token at a time.

Encoder-decoder inference has two stages:
1. Encode the full input (one forward pass through the encoder)
2. Generate output tokens (autoregressive, with cross-attention to encoder)

The cross-attention adds complexity (separate KV cache for encoder outputs) and memory overhead. At massive scale, this engineering complexity matters.

#### Reason 7: Emergent Capabilities Favor Scale

The most exciting capabilities (reasoning, code generation, creative writing, tool use) emerge at very large scale (100B+ parameters). The research community invested its scaling budget in decoder-only models, creating a **flywheel effect**: more research → better understanding → more investment → better models → more research.

Encoder-decoder models maxed out at T5-11B. By the time researchers might have pushed to 100B+ encoder-decoder, decoder-only was already winning at that scale.

### 8.3 The Counter-Arguments (Why Encoder-Decoder Should Have Won)

1. **Bidirectional encoding is provably stronger**: For understanding the input, bidirectional attention is strictly more powerful than causal attention. A bidirectional encoder can capture relationships that a causal model must infer indirectly.

2. **Structural alignment**: For tasks with clear input-output structure (translation, summarization), encoder-decoder's separation mirrors the task structure.

3. **Compute efficiency**: For long-input-short-output tasks, encoder-decoder is more FLOP-efficient (process the long input once, generate the short output).

4. **T5's ablation showed encoder-decoder was better**: At equal compute, encoder-decoder outperformed decoder-only in T5's experiments.

**Why these arguments lost**: They're all correct in principle, but scale effects dominated. At 175B+ parameters with ICL, the structural advantages of encoder-decoder became marginal while the simplicity advantages of decoder-only became decisive.

### 8.4 The Definitive Comparison

| Factor | Encoder-Decoder | Decoder-Only | Winner |
|--------|----------------|-------------|--------|
| Architecture simplicity | Complex (3 attention types) | Simple (1 attention type) | **Decoder** |
| Training objective | Span corruption (15% signal) | CLM (100% signal) | **Decoder** |
| In-context learning | Weaker, requires instruction tuning | Strong, emerges naturally | **Decoder** |
| Bidirectional understanding | Yes (encoder) | No (causal only) | **Enc-Dec** |
| Long input + short output | More efficient | Less efficient | **Enc-Dec** |
| Scaling laws | Less studied | Well characterized | **Decoder** |
| Inference engineering | More complex (2 stages) | Simpler (1 stage + KV cache) | **Decoder** |
| Universality | Need to define input/output split | Everything is generation | **Decoder** |
| Maximum scale achieved | 11B (T5) | 1.8T+ (GPT-4) | **Decoder** |

### 8.5 Encoder-Decoder vs Decoder-Only — When to Use Each

| Criterion | Encoder-Decoder (T5, BART) | Decoder-Only (GPT, Llama) |
|-----------|---------------------------|---------------------------|
| Task | Seq2seq: translation, summarization, QA | Open-ended generation, chat |
| Input-output relationship | Fixed input → variable output | Variable context → continuation |
| Training efficiency | Two separate stacks → more compute per sample | Simpler architecture |
| Fine-tuning ease | Works great with labeled (input, output) pairs | Works great with instruction tuning |
| KV cache at inference | Must store encoder outputs + decoder KV | Just decoder KV |
| Why decoder-only won | — | Scales better, more flexible, emergent ICL |

**When encoder-decoder still wins**:
- Machine translation (T5/mT5 competitive with GPT-4 on many language pairs)
- Structured prediction tasks with strict input→output format
- Low-resource settings where you have labeled (input, output) pairs
- Classification/NLU where you fine-tune on small datasets

---

## 9. When Encoder-Decoder Still Wins

Despite decoder-only dominance, encoder-decoder models remain the best choice for specific scenarios:

### 9.1 Machine Translation (Still)

For dedicated translation systems (not general-purpose LLMs), encoder-decoder is preferred:
- Bidirectional source encoding is critical for languages with different word orders
- The cross-attention alignment is interpretable and helps quality
- Models like NLLB (No Language Left Behind) use encoder-decoder for 200+ languages
- More efficient: encode source once, generate target

### 9.2 Speech Recognition (Whisper)

OpenAI's Whisper is an encoder-decoder transformer:
- Encoder: Processes the audio spectrogram bidirectionally
- Decoder: Generates the text transcript autoregressively
- Cross-attention creates a natural audio-text alignment

### 9.3 Structured Output Tasks

Tasks where the output has a specific structure that differs from the input:
- **Code compilation**: Source language → target language
- **Data-to-text**: Structured data → natural language description
- **SQL generation**: Natural language → SQL query (some systems prefer encoder-decoder)

### 9.4 Low-Resource Efficiency

For small models (< 1B params), encoder-decoder can be more parameter-efficient:
- Flan-T5-Small (80M) outperforms decoder-only models of similar size
- The bidirectional encoder compensates for limited parameter capacity
- At small scale, the structural advantages outweigh the simplicity advantages

### 9.5 Production Systems with Fixed Structure

When the input-output structure is fixed and well-defined, encoder-decoder's structural match to the task provides engineering benefits:
- Clear separation of encoding and decoding stages
- Encoder output can be cached and reused for different decodings
- Cross-attention provides interpretable attention patterns for debugging

---

## 10. Interview Questions & Answers

### Q1: How does T5's span corruption differ from BERT's token masking?

**A**: Three fundamental differences:

**1. What's masked**:
- BERT: Individual tokens, independently sampled (15% of tokens)
- T5: Contiguous spans of tokens (mean length 3, ~15% of tokens total)

Span masking is harder — the model must predict coherent multi-word phrases, not just individual words from context. This encourages learning compositional semantics.

**2. How the target is constructed**:
- BERT: Predicts the original token at each masked position (encoder-only, same-length output)
- T5: The decoder generates only the masked content, prefixed by sentinel tokens. Input: "Thank you `<X>` your party `<Y>`". Target: "`<X>` for inviting me to `<Y>` last week `<Z>`"

T5's target is much shorter than the input — typically ~15% the length. This makes decoder training efficient.

**3. Architecture fit**:
- BERT's MLM is designed for encoder-only (predict in-place)
- T5's span corruption is designed for encoder-decoder (encoder reads corrupted input, decoder generates the missing pieces)

Span corruption naturally exercises all three components: the encoder learns to understand corrupted text, cross-attention learns to align decoder queries with relevant encoder positions, and the decoder learns to generate coherent text. BERT's MLM only trains the encoder.

---

### Q2: Why did the field converge on decoder-only despite encoder-decoder being more flexible?

**A**: Seven factors drove this convergence, in order of importance:

**1. In-context learning**: The most transformative capability of modern LLMs — performing tasks from examples in the prompt without fine-tuning — emerged strongly and naturally in decoder-only models trained with CLM. Encoder-decoder models showed ICL too, but less reliably and requiring instruction tuning to activate.

**2. Training efficiency**: CLM provides gradient signal on 100% of tokens; span corruption provides signal on ~15%. For billion-dollar training runs, this 6.7× efficiency gap is decisive.

**3. Simplicity at scale**: One attention pattern, one training objective, one inference pipeline. At 1000+ GPU training runs, every source of complexity is a potential failure point. Decoder-only's simplicity translates to engineering reliability.

**4. Scaling law predictability**: The compute-performance relationship for CLM is well-characterized (Chinchilla). Organizations can confidently predict the outcome of a $100M training run. Encoder-decoder scaling is less understood.

**5. Universality**: Any task can be framed as "generate the next tokens." No need to decide what's input vs output, no need for separate encoder/decoder configurations.

**6. Inference optimization**: KV cache, speculative decoding, continuous batching — all inference optimizations are simpler for decoder-only. The cross-attention in encoder-decoder adds a second KV cache and complicates batch management.

**7. Momentum**: Once GPT-3 showed the path, research investment concentrated on decoder-only. More papers, better tools, more data recipes, more scaling insights — creating a flywheel that was hard for encoder-decoder to compete with.

**The bottom line**: Encoder-decoder's structural advantages (bidirectional encoding, natural input-output separation) are real but marginal at scale. Decoder-only's practical advantages (simplicity, efficiency, emergent capabilities) compound with scale.

---

### Q3: When would you still choose an encoder-decoder model over a decoder-only model?

**A**: Four scenarios where encoder-decoder is the better choice:

**1. Dedicated translation systems**: When building a specialized translation model (not a general-purpose LLM), encoder-decoder is more efficient. Bidirectional encoding of the source sentence is critical for handling different word orders across languages. The cross-attention creates natural alignment. NLLB (Meta's translation model for 200+ languages) uses encoder-decoder.

**2. Small model regime (<1B params)**: At small scale, encoder-decoder's structural advantages outweigh decoder-only's simplicity. Flan-T5-Small (80M) outperforms decoder-only models of similar size. The bidirectional encoder compensates for limited capacity.

**3. Fixed input-output structure with long inputs**: When the input is much longer than the output (e.g., document summarization with 10K input → 200 word summary), encoder-decoder is more efficient. The encoder processes the long input once; the decoder generates the short output with cross-attention. A decoder-only model would process all 10K+ tokens causally.

**4. Speech and audio tasks**: Whisper (speech recognition), and many TTS systems use encoder-decoder. The encoder processes the continuous signal (audio spectrogram) bidirectionally; the decoder generates the discrete output (text or audio tokens). The modality difference makes the separation natural.

**When NOT to choose encoder-decoder**: For general-purpose AI, chat, reasoning, or code generation — decoder-only is strictly better at the scales where these capabilities emerge (7B+).

---

### Q4: What is the prefix LM approach? How does it combine bidirectional and causal attention?

**A**: A Prefix LM uses a **single** transformer stack (shared parameters, unlike encoder-decoder) with a **mixed attention mask**:

- **Prefix tokens** (the input): Full bidirectional attention — every prefix token can attend to every other prefix token
- **Suffix tokens** (the output): Causal attention — each suffix token attends to all prefix tokens plus previous suffix tokens

$$
\text{Mask}_{ij} = \begin{cases}
\text{attend} & \text{if } i, j \in \text{prefix} \quad \text{(bidirectional)} \\
\text{attend} & \text{if } i \in \text{suffix}, j \in \text{prefix} \quad \text{(full cross)} \\
\text{attend} & \text{if } i \in \text{suffix}, j \in \text{suffix}, j \leq i \quad \text{(causal)} \\
\text{block} & \text{otherwise}
\end{cases}
$$

**Compared to encoder-decoder**: Prefix LM uses shared parameters (more parameter-efficient) but has less modeling capacity (no separate encoder/decoder specialization). T5's ablation showed encoder-decoder slightly outperforms Prefix LM at equal compute.

**Compared to decoder-only**: Prefix LM gives the input bidirectional attention (better understanding) but at the cost of flexibility — you must decide where the prefix ends and the suffix begins.

**Who uses it**: UniLM, GLM (the foundation for ChatGLM), some configurations of PaLM. It's a middle ground that hasn't become dominant because decoder-only's simplicity won at scale.

---

### Q5: Compare the pretraining objectives: CLM, MLM, span corruption, and UL2's Mixture of Denoisers. When is each best?

**A**:

| Objective | Signal Density | Bidirectional? | Architecture | Strongest Capability |
|-----------|---------------|---------------|-------------|---------------------|
| **CLM** | 100% (every token) | No (causal) | Decoder-only | Generation, ICL, reasoning |
| **MLM** | 15% (masked only) | Yes | Encoder-only | Understanding, classification |
| **Span corruption** | ~15% (span targets) | Yes (encoder) | Encoder-decoder | Conditional generation |
| **UL2 MoD** | Mixed (R:15%, S:variable, X:50%) | Yes (encoder) | Encoder-decoder | Versatility across all tasks |

**CLM is best when**: You want a general-purpose model, need ICL/reasoning, plan to scale to 10B+, or prioritize inference simplicity. It's the default choice for modern LLMs.

**MLM is best when**: You need high-quality bidirectional representations for classification, NER, or embeddings, and won't need generation. DeBERTa v3 with ELECTRA-style RTD is the modern instantiation.

**Span corruption is best when**: You're building an encoder-decoder for conditional generation (translation, summarization) at moderate scale (<11B params). T5/Flan-T5 is the mature implementation.

**UL2 MoD is best when**: You want a single model to excel at both understanding and generation tasks. The mixture of objectives creates a more versatile model than any single objective. Best for encoder-decoder models where you need broad capability.

---

### Q6: How does BART's denoising differ from T5's span corruption?

**A**: The key difference is in **what the decoder must generate**:

**T5 (span corruption)**:
- Input to encoder: Text with sentinel tokens replacing spans
- Decoder generates: Only the masked content (short target)
- Example — Input: "Thank you `<X>` your party `<Y>`" → Target: "`<X>` for inviting me to `<Y>` last week `<Z>`"

**BART (denoising)**:
- Input to encoder: Corrupted text (masked, deleted, shuffled, etc.)
- Decoder generates: The **entire original text** (full reconstruction)
- Example — Input: "Thank you [MASK] your party [MASK]" → Target: "Thank you for inviting me to your party last week."

**Implications**:

1. **Target length**: T5's target is ~15% of the input. BART's target is the full sequence. This makes T5's decoder more efficient to train.

2. **Corruption diversity**: T5 uses only span corruption. BART tests five strategies (token masking, deletion, infilling, sentence permutation, document rotation) and combines the best ones. The diversity of corruption creates more robust representations.

3. **Text infilling vs span corruption**: BART's text infilling replaces a variable-length span with a **single** [MASK] token. The model must infer both the content and the length of the missing span. T5 uses one sentinel per span, so the number of spans is known from the sentinels. BART's approach is harder and arguably teaches better understanding.

4. **Generation quality**: BART produces better generation quality (the decoder is trained to produce complete, fluent text). T5 produces better few-shot performance (span corruption is more efficient for the encoder).

---

### Q7: Explain Flan-T5 and why instruction tuning was so impactful for encoder-decoder models.

**A**: **Flan-T5** is T5 fine-tuned on 1,836 tasks phrased as natural language instructions, with multiple templates per task and chain-of-thought examples.

**Why instruction tuning was transformative for encoder-decoder models**:

T5's pretraining (span corruption) teaches the model to fill in missing text — but it doesn't teach the model to follow human instructions like "Summarize this article" or "Translate to French." There's a **gap** between the pretraining format and how users actually want to interact with the model.

Instruction tuning bridges this gap by:

1. **Teaching instruction parsing**: The model learns that "summarize:", "translate:", "answer:" are instructions, not text to continue.

2. **Activating latent capabilities**: T5 already has strong language understanding from pretraining. Instruction tuning doesn't teach new knowledge — it teaches the model to *apply* its existing knowledge in response to instructions.

3. **Enabling zero-shot generalization**: After seeing 1,836 tasks, the model can generalize to new tasks it wasn't trained on, by decomposing them into patterns it has seen.

**The quantitative impact**: Flan-T5-XXL (11B) matched GPT-3 (175B) on several benchmarks. This is a 16× parameter efficiency gain, entirely from instruction tuning.

**Why this was especially impactful for encoder-decoder**: Decoder-only models pretrained with CLM naturally learn to follow context patterns (because next-token prediction is inherently pattern-following). Encoder-decoder models pretrained with span corruption don't get this for free — they need explicit instruction tuning to "unlock" instruction following.

---

### Q8: How do encoder-decoder models handle the KV cache during inference?

**A**: Encoder-decoder inference has **two separate KV caches**:

**1. Encoder KV cache** (compute once, reuse):
- The full input is processed through the encoder in a single forward pass
- The encoder's output $\mathbf{H}_{\text{enc}} = [\mathbf{h}_1, \ldots, \mathbf{h}_n]$ is stored
- For cross-attention, the keys and values derived from encoder outputs are cached per decoder layer:

$$
\mathbf{K}_{\text{cross}}^{(l)} = \mathbf{H}_{\text{enc}} \mathbf{W}_K^{(l)}, \quad \mathbf{V}_{\text{cross}}^{(l)} = \mathbf{H}_{\text{enc}} \mathbf{W}_V^{(l)}
$$

These never change during generation — computed once and reused for every decoder step.

**2. Decoder self-attention KV cache** (grows with each step):
- Just like decoder-only models, the decoder's self-attention keys and values are cached and appended at each generation step
- At step $t$: cache contains K, V for decoder positions $1, \ldots, t-1$

**Total memory**: $\text{KV}_{\text{encoder}} + \text{KV}_{\text{decoder}}$

$$
\text{Memory} = \underbrace{2 \times L_{\text{dec}} \times h \times d_k \times n}_{\text{cross-attention cache (fixed)}} + \underbrace{2 \times L_{\text{dec}} \times h \times d_k \times m}_{\text{self-attention cache (grows)}}
$$

where $n$ is input length and $m$ is current output length.

**Compared to decoder-only**: A decoder-only model has one KV cache of size proportional to $n + m$. The encoder-decoder model has separate caches but the cross-attention cache doesn't grow during generation — an advantage when $n \gg m$.

---

### Q9: What was the most important finding from T5's systematic study?

**A**: The single most impactful finding was the **architecture comparison**: at equivalent compute budgets, encoder-decoder consistently outperformed decoder-only models on the benchmarks tested.

But the **most practically important** finding was something simpler: **scale and data matter more than architectural details**.

Specifically:
1. **More data always helps**: Increasing C4 size consistently improved results, with no saturation observed. This foreshadowed the data-scaling era.
2. **Pretraining objective differences are modest**: Span corruption, MLM, prefix LM — the differences between objectives were small (1-3%) compared to the effect of scale.
3. **Full fine-tuning beats parameter-efficient methods**: Updating all parameters during fine-tuning consistently beat adapter layers or prompt tuning.
4. **Training longer helps**: Even when models started to overfit on the pretraining data, continued training improved downstream performance.

**The meta-lesson**: T5 showed that NLP was becoming an **empirical science** where systematic experimentation and scale were more important than clever architectural innovations. This insight directly led to the scaling-focused approach (GPT-3, Chinchilla, Llama) that dominates today.

---

### Q10: If you were building a translation system in 2025, would you use encoder-decoder or decoder-only?

**A**: It depends on the requirements:

**Dedicated, high-volume translation system** → **Encoder-decoder**:
- Use NLLB-style encoder-decoder (based on Meta's No Language Left Behind)
- Bidirectional source encoding handles word-order differences across languages
- More efficient: encode source once, decode target with cross-attention
- Lower inference cost per translation at scale
- Interpretable cross-attention alignment for debugging
- Well-established: production translation systems (Google Translate, DeepL) still use encoder-decoder variants

**Translation as one capability among many** → **Decoder-only**:
- Use Llama 3 or GPT-4 with translation prompts
- Quality is competitive for high-resource language pairs
- Maintains all other capabilities (reasoning, code, conversation)
- Simpler deployment (one model for everything)
- Better for low-resource languages where ICL/prompting with examples can help

**The hybrid approach** (increasingly common):
- Decoder-only model as the main backbone
- Use the decoder-only LLM to generate training data (synthetic translations)
- Distill into a smaller encoder-decoder for production deployment
- Get the quality of the LLM with the efficiency of encoder-decoder at inference

This mirrors the broader trend: decoder-only for capability development and research, specialized architectures for production deployment.

---

*This completes Phase D: Language Models (Topics 11-13). You now understand all three architectural families — encoder-only, decoder-only, and encoder-decoder — including their strengths, weaknesses, and when to use each. Next: [Topic 14: Pretraining LLMs](14_Pretraining.md) — beginning Phase E: Training Large Language Models.*
