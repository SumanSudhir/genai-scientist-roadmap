# Topic 12: Decoder Models (GPT Family & Open-Source LLMs)

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 12 of 28
> **Scope**: Autoregressive language modeling, GPT-1/2/3/4, scaling laws (Kaplan, Chinchilla), Llama 1/2/3, Mistral & Mixtral (MoE), Phi, Gemma, Qwen, DeepSeek, architectural innovations across the family
> **Why this matters**: Decoder-only models are the dominant paradigm in AI. Every frontier model — GPT-4, Claude, Gemini, Llama 3 — is decoder-only. You must know the evolution from GPT-1's simple idea to today's trillion-parameter systems, the scaling laws that guide training decisions, and the architectural innovations in each model family. Interviewers will ask you to compare models, explain scaling decisions, and reason about trade-offs.
> **Previous**: [Topic 11: Encoder Models (BERT Family)](11_BERT_Family.md)
> **Next**: [Topic 13: Encoder-Decoder & Unified Models](13_Encoder_Decoder_Models.md)

---

## Table of Contents

1. [Autoregressive Language Modeling — The Foundation](#1-autoregressive-language-modeling--the-foundation)
2. [GPT-1 — Generative Pretraining (2018)](#2-gpt-1--generative-pretraining-2018)
3. [GPT-2 — Zero-Shot Transfer (2019)](#3-gpt-2--zero-shot-transfer-2019)
4. [GPT-3 — In-Context Learning at Scale (2020)](#4-gpt-3--in-context-learning-at-scale-2020)
5. [Scaling Laws — The Science of Bigger Models](#5-scaling-laws--the-science-of-bigger-models)
6. [GPT-4 and Beyond (2023-2025)](#6-gpt-4-and-beyond-2023-2025)
7. [Llama Family — Open-Source Revolution](#7-llama-family--open-source-revolution)
8. [Mistral & Mixtral — Efficiency Through Sparsity](#8-mistral--mixtral--efficiency-through-sparsity)
9. [Mixture of Experts (MoE) — Deep Dive](#9-mixture-of-experts-moe--deep-dive)
10. [Architectural Innovations Timeline](#10-architectural-innovations-timeline)
11. [Interview Questions & Answers](#11-interview-questions--answers)

---

## 1. Autoregressive Language Modeling — The Foundation

### 1.1 The Objective

Every decoder-only model is trained on the same fundamental objective: **predict the next token**.

Given a sequence of tokens $x_1, x_2, \ldots, x_{n-1}$, predict $x_n$:

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

The training loss is the negative log-likelihood:

$$
\mathcal{L} = -\sum_{t=1}^{n} \log P(x_t \mid x_{<t})
$$

This is called **Causal Language Modeling (CLM)** because each token is predicted using only **past** (causal) context.

### 1.2 Why This Simple Objective Works

Next-token prediction seems trivially simple — yet it produces models capable of reasoning, coding, and creative writing. Why?

**1. Compression requires understanding**: To predict the next word well, the model must understand syntax, semantics, world knowledge, logic, and style. The loss function forces the model to compress all of language into its parameters.

**2. 100% training efficiency**: Unlike MLM (15% of tokens), every single token provides gradient signal. This makes CLM dramatically more efficient per training example.

**3. Natural generation**: A model trained to predict the next token can directly generate text by sampling from its own predictions — no architectural modifications needed.

**4. Scalability**: The objective is the same whether you train on 1B or 15T tokens. No task-specific engineering. Just add more data and compute.

### 1.3 Training vs Inference

**Training** (teacher forcing): The model sees the true sequence and predicts each token from the ground truth context. All positions computed in parallel using the causal mask.

$$
\text{Loss at position } t: -\log P(x_t \mid x_1, \ldots, x_{t-1}) \quad \text{(using true } x_1, \ldots, x_{t-1}\text{)}
$$

**Inference** (autoregressive): The model generates one token at a time, feeding its own predictions back as input:

$$
\hat{x}_t \sim P(x_t \mid x_1, \ldots, x_{t-2}, \hat{x}_{t-1})
$$

This creates a **train-test discrepancy** (called **exposure bias** — see [Topic 7](07_Sequence_Modeling.md)): during training the model always sees true tokens, but during inference it sees its own (potentially incorrect) predictions. In practice, large well-trained models are robust to this.

---

## 2. GPT-1 — Generative Pretraining (2018)

### 2.1 The Key Idea

GPT-1 (Radford et al., 2018) combined two ideas:

1. **Unsupervised pretraining**: Train a transformer decoder on a large unlabeled corpus using CLM
2. **Supervised fine-tuning**: Adapt the pretrained model to specific tasks with labeled data

This was parallel to BERT's approach but with a causal (left-to-right) objective instead of MLM.

### 2.2 Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden size ($d$) | 768 |
| Attention heads | 12 |
| FFN dimension | 3072 |
| Parameters | 117M |
| Context length | 512 |
| Vocabulary | BPE, 40,000 tokens |
| Activation | GELU |
| Positional encoding | Learned absolute |

GPT-1 is essentially BERT-Base in decoder-only form — same dimensions, same number of layers, similar parameter count.

### 2.3 Pretraining

- **Data**: BooksCorpus (~800M words) — the same corpus used for BERT
- **Objective**: Standard CLM (next-token prediction)

### 2.4 Fine-Tuning

GPT-1 fine-tuned by reformulating each task as a sequence with special delimiter tokens:

```
Classification:   [START] text [EXTRACT] → label
Entailment:       [START] premise [DELIM] hypothesis [EXTRACT] → label
Similarity:       [START] text1 [DELIM] text2 [EXTRACT] → score
Multiple Choice:  [START] context [DELIM] answer_k [EXTRACT] → score (for each k)
```

The final hidden state at the `[EXTRACT]` position was used for classification — similar to BERT's [CLS] token.

**Combined loss during fine-tuning**:

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{CLM}}
$$

Adding the CLM loss as an auxiliary objective during fine-tuning helped regularization and faster convergence.

### 2.5 Significance

GPT-1 proved that **generative pretraining helps discriminative tasks** — a left-to-right model could match or beat task-specific architectures after fine-tuning. But it was still in the pretrain-finetune paradigm. GPT-2 would change that.

---

## 3. GPT-2 — Zero-Shot Transfer (2019)

### 3.1 The Revolutionary Insight

GPT-2's key insight: **a language model trained on enough diverse data can perform tasks without any fine-tuning**.

Instead of fine-tuning on labeled data, you present the task in natural language and let the model generate the answer:

```
Translation:    "Translate English to French: cheese =>"  → "fromage"
Summarization:  "Article: [text] TL;DR:"                 → summary
QA:             "Q: What is the capital of France? A:"    → "Paris"
```

The model was never explicitly trained on translation, summarization, or QA. It learned these capabilities implicitly from predicting next tokens in diverse web text.

### 3.2 Significance

GPT-2 demonstrated the **zero-shot paradigm**: scale up the model and data, and tasks emerge without explicit training. This was the first hint that language models might be **general-purpose AI systems**, not just text predictors.

---

## 4. GPT-3 — In-Context Learning at Scale (2020)

### 4.1 The Scale

GPT-3 was a 100× scale-up from GPT-2:

| Parameter | GPT-2 XL | GPT-3 175B |
|-----------|----------|------------|
| Parameters | 1.5B | **175B** |
| Layers | 48 | **96** |
| $d_{\text{model}}$ | 1600 | **12288** |
| Heads | 25 | **96** |
| $d_{\text{head}}$ | 64 | 128 |
| FFN dimension | 6400 | 49152 |
| Context length | 1024 | **2048** |
| Training tokens | ~10B | **300B** |
| Training compute | — | ~3.14 × 10²³ FLOPs |

### 4.2 The GPT-3 Model Family

GPT-3 was actually 8 models of different sizes, trained to study scaling:

| Model | Params | Layers | $d$ | Heads |
|-------|--------|--------|-----|-------|
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 |
| GPT-3 Large | 760M | 24 | 1536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2048 | 16 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 |
| GPT-3 13B | 13B | 40 | 5140 | 40 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** |

### 4.3 In-Context Learning (ICL)

GPT-3's defining discovery: **large models can learn from examples provided in the prompt, without any gradient updates**.

Three paradigms:

**Zero-shot**: Task description only
```
Translate English to French:
cheese =>
```

**One-shot**: One example
```
Translate English to French:
sea otter => loutre de mer
cheese =>
```

**Few-shot**: Several examples
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe en peluche
cheese =>
```

**The remarkable finding**: Few-shot performance improved consistently with model scale. At 175B parameters, few-shot prompting often matched or exceeded fine-tuned BERT models — without updating a single weight.

### 4.4 Emergent Abilities

GPT-3 revealed that certain capabilities **emerge** at sufficient scale — they're absent in smaller models and appear suddenly:

- **In-context learning**: Barely works at 1B; consistently works at 100B+
- **Chain-of-thought reasoning**: Absent in small models; emerges at ~100B
- **Arithmetic**: GPT-3 175B can do 2-digit addition; smaller models can't
- **Code generation**: Meaningful code only at 10B+

This led to the hypothesis that scale itself creates qualitatively new capabilities — not just quantitative improvement.

### 4.5 Training Data

| Source | Tokens (B) | Weight |
|--------|-----------|--------|
| Common Crawl (filtered) | 410 | 60% |
| WebText2 | 19 | 22% |
| Books1 | 12 | 8% |
| Books2 | 55 | 8% |
| Wikipedia | 3 | 3% |

**Data mixing**: Sources were weighted differently from their proportion in the dataset. Higher-quality sources (books, Wikipedia) were **upsampled** — seen more times during training than their raw fraction would suggest.

### 4.6 Limitations Acknowledged

The GPT-3 paper was unusually candid about limitations:

1. **Repetition**: Tends to repeat phrases and lose coherence in long outputs
2. **Reasoning**: Struggles with logical reasoning, especially multi-step
3. **Bias**: Reflects biases present in training data
4. **Factuality**: Confidently generates plausible-sounding but incorrect facts
5. **Context window**: 2048 tokens was limiting for many real-world tasks

### 4.7 In-Context Learning — How It Works

GPT-3's most surprising property: it can learn new tasks from examples in the prompt.

**3-shot sentiment classification example**:

```
Prompt:
  "The food was terrible." → negative
  "Great service, loved it!" → positive
  "It was okay I guess." → neutral
  "The best meal of my life!" → ???

GPT-3 output: "positive"
```

No gradient updates — this happens at inference time. The model simply completes the pattern.

**What's actually happening**:
1. The model has seen millions of similar format patterns during pretraining
2. The examples in the prompt act as a "task description" that activates relevant knowledge
3. "Induction heads" (attention heads that copy past patterns) match [example → label] pairs
4. The model continues the pattern with the test input

**Why this is remarkable**:
- 0-shot: no examples, just instruction ("Classify sentiment: ...")
- 1-shot: one example
- Few-shot (k-shot): k examples (typically k=3-10)
- Performance scales with k and model size — tiny models can't do it, large ones can

**The emergent capability**: GPT-3 (175B) showed this clearly; GPT-2 (1.5B) showed it weakly; earlier models couldn't do it at all. This suggests ICL is an emergent capability that appears above a parameter threshold (~10-50B for reliable few-shot learning).

---

## 5. Scaling Laws — The Science of Bigger Models

### 5.1 Why Scaling Laws Matter

Scaling laws tell us **how model performance improves** as we increase:
- $N$: Number of parameters
- $D$: Amount of training data (tokens)
- $C$: Compute budget (FLOPs)

These laws are critical for planning LLM training: should you make the model bigger or train it longer? The answer determines billion-dollar compute budgets.

### 5.2 Kaplan Scaling Laws (OpenAI, 2020)

Kaplan et al. found that test loss follows **power laws** in each factor:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

where $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$.

**Key finding**: Performance improves predictably as a smooth function of scale. There are no sharp transitions (contradicting the "emergent abilities" narrative, depending on the metric).

**Kaplan's recommendation**: For a fixed compute budget, make the model **as large as possible** and train on relatively fewer tokens. They suggested:

$$
N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}
$$

This means: scaling parameters is more important than scaling data. Double the compute? Mostly grow the model size.

### 5.3 Chinchilla Scaling Laws (DeepMind, 2022)

The Chinchilla paper (Hoffmann et al., 2022) fundamentally revised Kaplan's recommendations.

**Chinchilla's finding**: For a fixed compute budget, parameters and data should scale **equally**:

$$
N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

The optimal ratio: approximately **20 tokens per parameter**.

$$
D_{\text{opt}} \approx 20 \times N
$$

**The implication**: GPT-3 was **massively undertrained**. With 175B parameters, it should have been trained on $\sim 3.5T$ tokens (it used 300B — roughly 10× too few).

| Model | Params | Training Tokens | Tokens/Param | Chinchilla Optimal? |
|-------|--------|----------------|--------------|-------------------|
| GPT-3 | 175B | 300B | 1.7 | Very undertrained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| Llama 1 (65B) | 65B | 1.4T | 21.5 | Near-optimal |
| Llama 2 (70B) | 70B | 2T | 28.6 | Over-trained (intentionally) |
| Llama 3 (70B) | 70B | 15T | 214 | Massively over-trained |

### 5.4 Beyond Chinchilla: Inference-Optimal Scaling

Chinchilla optimizes for **training compute**. But in production, **inference cost** dominates. A smaller model trained on more data is cheaper to deploy:

$$
\text{Inference cost} \propto N \quad (\text{per token})
$$

$$
\text{Training cost} \propto 6ND \quad (\text{total FLOPs} \approx 6 \times N \times D)
$$

If you'll serve the model to millions of users, it's worth spending more on training (higher $D$) to get a smaller model ($N$) that's cheaper per inference. This is why:

- **Llama 3 8B** was trained on 15T tokens (1875 tokens/param) — vastly overtrained by Chinchilla's standard
- **Phi-3** was trained on 3.3T tokens for just 3.8B params

The new paradigm: **train small models on enormous data** for deployment efficiency.

### 5.5 The Compute-Optimal Frontier

```
                    │  Performance
                    │
                    │         ╱ Chinchilla-optimal
                    │       ╱   (equal N and D scaling)
                    │     ╱
                    │   ╱  ╱ Kaplan-style
                    │  ╱ ╱   (prioritize N)
                    │╱╱
                    │╱   ╱ Inference-optimal
                    │  ╱   (prioritize D, keep N small)
                    │╱
                    └────────────────────── Compute budget
```

### 5.6 FLOPs Estimation

The standard approximation for transformer training FLOPs:

$$
C \approx 6ND
$$

where:
- $C$ = total training FLOPs
- $N$ = number of parameters
- $D$ = number of training tokens

The factor 6 comes from: each token requires ~2 FLOPs per parameter for the forward pass and ~4 FLOPs for the backward pass (2× forward for gradient computation, 2× for gradient accumulation).

**Example**: Llama 3 70B on 15T tokens:

$$
C \approx 6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24} \text{ FLOPs}
$$

On H100 GPUs at ~50% utilization (500 TFLOP/s effective):

$$
\text{GPU-hours} = \frac{6.3 \times 10^{24}}{500 \times 10^{12} \times 3600} \approx 3.5M \text{ GPU-hours}
$$

At ~$2/GPU-hour: **~$7M** in compute.

### 5.7 Chinchilla Scaling Law — Worked Example

Chinchilla (Hoffmann et al., 2022) found the compute-optimal recipe:

$$
N_{\text{opt}} \propto C^{0.5} \quad \text{(model size)}, \qquad D_{\text{opt}} \propto C^{0.5} \quad \text{(training tokens)}
$$

More precisely: optimal N/D ratio ≈ **20 tokens per parameter**.

**Example**: Given a compute budget of $6 \times 10^{23}$ FLOPs (roughly GPT-3's budget):

```
GPT-3's approach:  N=175B params, D=300B tokens   (ratio ≈ 1.7 tokens/param)
Chinchilla:        N=70B params,  D=1.4T tokens   (ratio ≈ 20 tokens/param)
```

Chinchilla 70B (compute-matched to GPT-3 175B) outperforms GPT-3 on most benchmarks!

**The insight**: GPT-3 was severely undertrained relative to its parameter count. Models should be trained on ~20× more tokens than they have parameters.

**Practical impact on modern models**:

```
Llama 2 7B:  trained on 2T tokens  → 286 tokens/param  (far more than 20×)
Llama 2 70B: trained on 2T tokens  → 29 tokens/param   (close to Chinchilla optimal)
Llama 3 8B:  trained on 15T tokens → 1875 tokens/param (massively over-trained)
```

Modern models violate Chinchilla because **inference cost matters**: a smaller, longer-trained model is cheaper to serve even if not compute-optimal to train. Training cost is one-time; serving cost is per-query.

---

## 6. GPT-4 and Beyond (2023-2025)

### 6.1 What We Know

OpenAI released minimal technical details about GPT-4. Based on public information and credible reports:

| Aspect | GPT-4 (2023) | GPT-4o (2024) | o1/o3 (2024-2025) |
|--------|-------------|---------------|-------------------|
| Architecture | Decoder-only (rumored MoE) | Decoder-only (rumored MoE) | Decoder-only + reasoning |
| Parameters | ~1.8T (rumored, 8×220B MoE) | Unknown | Unknown |
| Context | 8K / 32K / 128K | 128K | 128K-200K |
| Modality | Text + image input | Text + image + audio I/O | Text |
| Key capability | Multimodal, strong reasoning | Faster, natively multimodal | Extended reasoning (CoT) |

### 6.2 GPT-4's Innovations (Inferred)

**Mixture of Experts (MoE)**: Widely believed to use 8 expert FFNs per layer with top-2 routing. This means ~1.8T total parameters but only ~220B activated per forward pass — enabling GPT-3-level inference cost with much more total capacity.

**Multimodal training**: GPT-4 processes images natively — not through a separate vision encoder bolted on, but integrated into the main transformer training.

**RLHF at scale**: Extensive alignment training using RLHF (see [Topic 16](16_Alignment_RLHF_DPO.md)) to improve helpfulness, harmlessness, and instruction following.

### 6.3 The o1/o3 Paradigm: Test-Time Compute

OpenAI's o1 and o3 models introduced a new scaling axis: **test-time compute**.

Instead of just producing an answer, the model generates extended **chains of reasoning** (thinking tokens) before answering:

$$
\text{Answer quality} \propto f(\text{model size}, \text{training compute}, \text{inference compute})
$$

**Key insight**: You can improve performance by spending more compute at inference time (longer reasoning chains), not just by training bigger models. This is analogous to how humans spend more time thinking about harder problems.

The reasoning tokens are trained via **process reward models** — rewarding each step of reasoning, not just the final answer. This is covered in [Topic 27: Research Frontiers](27_Research_Frontiers.md).

---

## 7. Llama Family — Open-Source Revolution

### 7.1 Why Llama Matters

Meta's Llama models transformed the LLM landscape by releasing **high-quality model weights** to the research community. Before Llama (Feb 2023), the only competitive LLMs were proprietary (GPT-3/4, PaLM). After Llama, the open-source ecosystem exploded.

### 7.2 Llama 1 (February 2023)

**Key contribution**: Showed that smaller, well-trained models can match much larger undertrained models.

| Model | Params | Layers | $d$ | Heads | $d_{ff}$ | Context | Training Tokens |
|-------|--------|--------|-----|-------|-----------|---------|----------------|
| Llama 7B | 6.7B | 32 | 4096 | 32 | 11008 | 2048 | 1.0T |
| Llama 13B | 13.0B | 40 | 5120 | 40 | 13824 | 2048 | 1.0T |
| Llama 33B | 32.5B | 60 | 6656 | 52 | 17920 | 2048 | 1.4T |
| Llama 65B | 65.2B | 80 | 8192 | 64 | 22016 | 2048 | 1.4T |

**Architecture innovations** (vs GPT-3):

| Component | GPT-3 | Llama 1 |
|-----------|-------|---------|
| Normalization | Pre-LN (LayerNorm) | **Pre-RMSNorm** |
| Activation | GELU | **SwiGLU** |
| Positional encoding | Learned absolute | **RoPE** |
| Bias terms | Yes | **No biases** |
| Weight tying | Yes | **No** |

**Result**: Llama 65B matched GPT-3 175B despite being 2.7× smaller. The key: Chinchilla-optimal training (1.4T tokens for 65B params ≈ 21.5 tokens/param).

### 7.3 Llama 2 (July 2023)

**Key contributions**: Longer context, more data, grouped-query attention, alignment via RLHF.

| Model | Params | Layers | $d$ | Q Heads | KV Heads | $d_{ff}$ | Context | Tokens |
|-------|--------|--------|-----|---------|----------|-----------|---------|--------|
| Llama 2 7B | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 11008 | 4096 | 2.0T |
| Llama 2 13B | 13.0B | 40 | 5120 | 40 | 40 (MHA) | 13824 | 4096 | 2.0T |
| Llama 2 70B | 68.7B | 80 | 8192 | 64 | **8 (GQA)** | 28672 | 4096 | 2.0T |

**Changes from Llama 1**:

1. **More training data**: 2T tokens (vs 1-1.4T), with 40% more data overall
2. **Doubled context**: 4096 tokens (vs 2048)
3. **Grouped-Query Attention (GQA)**: The 70B model uses 8 KV heads shared across 64 query heads — 8× KV cache reduction. Smaller models kept full MHA.
4. **Llama 2 Chat**: Aligned versions using SFT + RLHF (see [Topic 16](16_Alignment_RLHF_DPO.md))

**GQA in Llama 2 70B**: 64 query heads grouped into 8 groups, each group sharing one KV head. This was critical for serving the 70B model efficiently — KV cache at 4096 context would otherwise require ~40GB in FP16.

### 7.4 Llama 3 (April 2024) and Llama 3.1 (July 2024)

**Key contributions**: Massive scale-up in data, vocabulary, and context length. Llama 3 showed that you can keep architecture constant and get enormous gains from more (and better) data.

| Model | Params | Layers | $d$ | Q Heads | KV Heads | $d_{ff}$ | Context | Tokens | Vocab |
|-------|--------|--------|-----|---------|----------|-----------|---------|--------|-------|
| Llama 3 8B | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 8192 | 15T | 128K |
| Llama 3 70B | 70.6B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 8192 | 15T | 128K |
| Llama 3.1 405B | 405B | 126 | 16384 | 128 | 8 (GQA) | 53248 | 128K | 15T | 128K |

**What changed from Llama 2**:

1. **15T training tokens** (7.5× increase from 2T) — far beyond Chinchilla-optimal, prioritizing inference efficiency
2. **128K vocabulary** (4× increase from 32K) — better multilingual support, shorter sequences for same content
3. **GQA everywhere** — even 8B model uses GQA (8 KV heads for 32 Q heads)
4. **Increased FFN dimension**: 14336 for 8B model (vs 11008 in Llama 2 7B) — more capacity
5. **Extended context to 128K** (Llama 3.1) via RoPE with $\theta = 500000$
6. **Data quality**: Extensive filtering, deduplication, domain classification, and quality scoring

### 7.5 Llama 3 Data Pipeline

The data pipeline is as important as the architecture:

```
Raw Web Data (hundreds of TB)
       │
       ▼
  URL Deduplication ─── Remove exact duplicate URLs
       │
       ▼
  Text Extraction ──── HTML → clean text
       │
       ▼
  Quality Filtering ── Classifier trained on Wikipedia-like quality
       │
       ▼
  Deduplication ────── MinHash + exact dedup
       │
       ▼
  PII Removal ──────── Remove personal information
       │
       ▼
  Domain Classification ── Code, math, science, etc.
       │
       ▼
  Safety Filtering ──── Remove toxic/harmful content
       │
       ▼
  Data Mixing ──────── Weight domains for optimal performance
       │
       ▼
  Final: ~15T tokens
```

**Key data decisions**:
- Upsample code and math data (improves reasoning)
- Include multilingual data (~5% of total) for 128K vocab utilization
- Synthetic data for specific capabilities (math reasoning, tool use)

### 7.6 The Llama Architecture Summary

All Llama variants share these design choices:

| Component | Choice |
|-----------|--------|
| Architecture | Decoder-only |
| Normalization | Pre-RMSNorm |
| FFN activation | SwiGLU |
| Positional encoding | RoPE |
| Attention | GQA (in larger models / Llama 3) |
| Bias terms | None |
| Weight tying | No |
| Tokenizer | SentencePiece BPE |

This has become the **de facto standard architecture** for open-source LLMs. Almost every model released since Llama 1 follows this template.

---

## 8. Mistral & Mixtral — Efficiency Through Sparsity

### 8.1 Mistral 7B (September 2023)

Mistral 7B was notable for outperforming Llama 2 13B despite having half the parameters.

**Key innovations**:

| Feature | Llama 2 7B | Mistral 7B |
|---------|-----------|------------|
| GQA | No (full MHA) | **Yes** (8 KV heads, 32 Q heads) |
| Sliding Window Attention | No | **Yes** (window = 4096) |
| Context length | 4096 | 8192 (effective: much longer via SWA) |
| Performance | — | **Beats Llama 2 13B** |

**Sliding Window Attention (SWA)**: Instead of attending to all previous tokens, each layer attends only to the last $W$ tokens (window size):

$$
\text{Attention}_{ij} = \begin{cases} \text{softmax}\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right) & \text{if } i - W < j \leq i \\ 0 & \text{otherwise} \end{cases}
$$

```
Full causal attention (Llama):     Sliding window (Mistral, W=3):
████░░░░                           ███░░░░░
█████░░░                           ░███░░░░
██████░░                           ░░███░░░
███████░                           ░░░███░░
████████                           ░░░░███░
```

**But information still propagates far**: With $L$ layers and window $W$, information can travel up to $L \times W$ positions through the network. For Mistral (32 layers × 4096 window): effective receptive field of ~131K tokens.

**Memory savings**: KV cache only needs to store $W$ tokens (rolling buffer) instead of the entire sequence. For $W = 4096$: KV cache is fixed regardless of sequence length.

### 8.2 Mixtral 8x7B (December 2023)

Mixtral is a **Mixture of Experts (MoE)** model — one of the most important architecture innovations for scaling LLMs.

| Parameter | Mistral 7B | Mixtral 8x7B |
|-----------|-----------|--------------|
| Total parameters | 7.2B | **46.7B** |
| Active parameters (per token) | 7.2B | **12.9B** |
| Experts | 1 (dense) | **8** (top-2 active) |
| Layers | 32 | 32 |
| $d_{\text{model}}$ | 4096 | 4096 |
| Performance | ≈ Llama 2 13B | **≈ Llama 2 70B** |
| Inference cost | ~7B MACs | ~13B MACs |

Mixtral achieves **Llama 2 70B performance** at roughly **1/5 the inference cost**.

---

## 9. Mixture of Experts (MoE) — Deep Dive

MoE is critical enough to warrant its own section. It's the key architectural innovation enabling efficient scaling.

### 9.1 The Core Idea

In a standard transformer, every token passes through the same FFN in each layer. In an MoE transformer, the FFN is replaced by **multiple expert FFNs**, and a **router** selects which experts process each token.

**Standard FFN**:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x})
$$

**MoE FFN**:

$$
\text{MoE}(\mathbf{x}) = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot \text{FFN}_i(\mathbf{x})
$$

where $g_i(\mathbf{x})$ is the gating weight for expert $i$.

### 9.2 The Router (Gating Network)

The router determines which experts process each token:

$$
\mathbf{g}(\mathbf{x}) = \text{TopK}\left(\text{softmax}(\mathbf{W}_g \cdot \mathbf{x})\right)
$$

where $\mathbf{W}_g \in \mathbb{R}^{E \times d}$ is the router weight matrix.

**Top-K routing**: Only the top $K$ experts (typically $K = 2$) are activated. The rest have zero weight:

$$
g_i(\mathbf{x}) = \begin{cases} \text{softmax}_i(\mathbf{W}_g \mathbf{x}) & \text{if } i \in \text{TopK} \\ 0 & \text{otherwise} \end{cases}
$$

The final output is the **weighted sum** of the active experts:

$$
\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} g_i(\mathbf{x}) \cdot \text{FFN}_i(\mathbf{x})
$$

### 9.3 Visual Representation

```
Token x ──► Router ──► [Expert 1: g=0.6] ──►  0.6 × FFN₁(x)
              │        [Expert 2: g=0.0] ──►  (not computed)
              │        [Expert 3: g=0.4] ──►  0.4 × FFN₃(x)  ──► sum ──► output
              │        [Expert 4: g=0.0] ──►  (not computed)
              │        ...
              │        [Expert 8: g=0.0] ──►  (not computed)
              │
              └── Selects top-2 experts based on gating scores
```

### 9.4 Why MoE Works

**More parameters, same compute**: With 8 experts and top-2 routing, the model has 8× more FFN parameters but activates only 2/8 = 25% per token. Total parameters ≈ 8× dense, but FLOPs per token ≈ same as 2× dense.

**Expert specialization**: Different experts learn to handle different types of tokens/patterns:
- Some experts may specialize in code
- Others in mathematical reasoning
- Others in specific languages or domains
- This specialization is emergent — not programmed

**Efficient scaling**: To make a model more knowledgeable, add more experts rather than making each layer wider. Expert addition is cheaper than width increase because only a subset is active.

### 9.5 The Load Balancing Problem

The biggest challenge in MoE: ensuring all experts are used roughly equally.

**Problem**: Without intervention, the router often collapses — sending most tokens to 1-2 "popular" experts while others are unused. This wastes capacity.

**Solution**: Add an auxiliary **load balancing loss**:

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i
$$

where:
- $f_i$ = fraction of tokens routed to expert $i$ (actual load)
- $p_i$ = average router probability for expert $i$ (predicted preference)
- $\alpha$ = balancing coefficient (typically 0.01)

This loss encourages uniform distribution of tokens across experts. Minimized when $f_i = p_i = 1/E$ for all experts.

### 9.6 MoE Trade-offs

| Advantage | Disadvantage |
|-----------|--------------|
| More total parameters at same inference cost | Higher memory (all expert weights must be loaded) |
| Expert specialization | Load balancing is finicky |
| Scales well to very large models | Communication overhead in distributed training |
| Faster training per FLOP | Harder to fine-tune (expert routing may break) |
| | Batch sizes must be large enough to utilize all experts |

### 9.7 MoE in Practice

| Model | Experts | Active | Total Params | Active Params | Performance |
|-------|---------|--------|-------------|--------------|-------------|
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B | ≈ Llama 2 70B |
| Mixtral 8x22B | 8 | 2 | 176B | 39B | ≈ Llama 3 70B |
| DeepSeek-V2 | 160 | 6 | 236B | 21B | Competitive with 70B |
| GPT-4 (rumored) | 8 | 2 | ~1.8T | ~220B | Frontier |
| DeepSeek-V3 | 256 | 8 | 671B | 37B | Competitive with GPT-4 |

### 9.8 MoE Routing — Concrete Example

**Architecture**: 8 experts, top-2 routing (each token uses 2 of 8 experts).

For token "Paris" in a Mixtral-style model:

1. Gating network computes scores for all 8 experts:

$$
\mathbf{g} = \text{softmax}(\mathbf{W}_{\text{gate}} \times \text{token\_embedding})
$$

$$
\mathbf{g} = [0.28,\ 0.05,\ 0.31,\ 0.08,\ 0.11,\ 0.04,\ 0.09,\ 0.04]
$$

2. Select top-2: **Expert 3** (0.31) and **Expert 1** (0.28)

3. Renormalize weights: $[0.28/(0.28+0.31),\ 0.31/(0.28+0.31)] = [0.47,\ 0.53]$

4. Compute FFN for selected experts only:

$$
\text{output} = 0.47 \times \text{Expert}_1(\text{"Paris"}) + 0.53 \times \text{Expert}_3(\text{"Paris"})
$$

5. Experts 2, 4, 5, 6, 7, 8 do NOT process this token — no compute for them.

```
Token       Experts chosen         Experts skipped
"Paris"  →  [Expert1, Expert3]     [Expert2,4,5,6,7,8] — not activated
"the"    →  [Expert2, Expert5]     [Expert1,3,4,6,7,8] — not activated
"jumped" →  [Expert4, Expert1]     [Expert2,3,5,6,7,8] — not activated
```

**Compute savings**: 8 experts, 2 active → use 25% of FFN parameters per token.
If each expert has $d_{\text{ff}} = 14336$ (Mixtral), effective FFN per token = $2 \times 14336 = 28{,}672$ vs a dense model with all 8 experts always = $8 \times 14336 = 114{,}688$.
Total parameters: 8× more than a single expert, but compute: same as ~2× expert.

**Load balancing loss**: Without it, the router might always pick expert 1 (collapse). An auxiliary loss penalizes uneven expert utilization.

### 9.9 Small Language Models — Quick Reference

A counter-trend to "bigger is better": small models (1–14B) trained on carefully curated data can dramatically outperform their size class.

| Model | Params | Key Innovation | Use Case |
|-------|--------|---------------|----------|
| Phi-3 Mini | 3.8B | Synthetic + heavily filtered data ("textbooks") | On-device, reasoning |
| Phi-3 Medium | 14B | Data curriculum: easy → hard progression | Edge deployment |
| Gemma 2 2B | 2.6B | Knowledge distillation from larger teacher | Mobile inference |
| Gemma 2 9B | 9.2B | Local + global alternating attention layers | Mid-range tasks |
| Gemma 2 27B | 27.2B | Logit soft-capping for training stability | Server, matches Llama 3 70B |
| SmolLM | 0.1–1.7B | Ultra-compact for edge/browser deployment | Embedded AI |

**Key insight**: Data quality > quantity > model size. Phi-3 Mini (3.8B on 3.3T tokens) matches models 10× its size.

---

> **Note — Global LLM Landscape**: DeepSeek-R1 and Qwen2.5 demonstrate that non-US labs can match GPT-4 quality. DeepSeek uses MoE with 671B total / 37B active params. Both are open-weights. Key for interviews: awareness that the LLM landscape is now truly global.

---

## 10. Architectural Innovations Timeline

### 12.1 Chronological Evolution

```
2017 │ Original Transformer (Vaswani et al.)
     │  Post-LN, ReLU FFN, Sinusoidal PE, Encoder-Decoder
     │
2018 │ GPT-1 (OpenAI)
     │  Decoder-only, Learned PE, GELU, fine-tuning paradigm
     │
2019 │ GPT-2 (OpenAI)
     │  Pre-LN, byte-level BPE, zero-shot transfer
     │
2020 │ GPT-3 (OpenAI)
     │  Scale: 175B params, in-context learning, few-shot
     │
2020 │ Kaplan Scaling Laws
     │  Power law relationships: loss vs N, D, C
     │
2021 │ RoPE (Su et al.)
     │  Rotary position embeddings — became standard
     │
2022 │ Chinchilla (DeepMind)
     │  20 tokens/param optimal, revised scaling laws
     │
2022 │ PaLM (Google)
     │  540B, parallel attn+FFN, SwiGLU
     │
2023 │ Llama 1 (Meta)
     │  RMSNorm + SwiGLU + RoPE, open weights, Chinchilla-optimal
     │
2023 │ Llama 2 (Meta)
     │  GQA, longer context, RLHF alignment
     │
2023 │ Mistral 7B
     │  Sliding window attention, efficient GQA
     │
2023 │ Mixtral 8x7B
     │  Open-source MoE, top-2 routing
     │
2023 │ Phi-1/1.5 (Microsoft)
     │  Synthetic data, data quality > model size
     │
2024 │ Llama 3 (Meta)
     │  15T tokens, 128K vocab, massive overtraining
     │
2024 │ DeepSeek-V2
     │  MLA (compressed KV), fine-grained MoE (160 experts)
     │
2024 │ Gemma 2 (Google)
     │  Knowledge distillation, logit soft-capping
     │
2024 │ DeepSeek-V3
     │  256 experts, $5.5M training cost, multi-token prediction
     │
2025 │ DeepSeek-R1
     │  RL-trained reasoning, no SFT needed
```

### 12.2 Key Architectural Components Adoption

| Component | Introduced | Now Standard? |
|-----------|-----------|--------------|
| Pre-LN | GPT-2 (2019) | Yes — universal |
| GELU | GPT-1 (2018) | Replaced by SwiGLU |
| SwiGLU | PaLM (2022) | Yes — in all major LLMs |
| RoPE | RoFormer (2021) | Yes — in all major LLMs |
| GQA | Ainslie et al. (2023) | Yes — standard for models ≥ 7B |
| RMSNorm | Zhang & Sennrich (2019) | Yes — replaced LayerNorm |
| MoE | Switch Transformer (2021) | Growing — used in frontier models |
| No biases | Llama 1 (2023) | Yes — common practice |
| MLA | DeepSeek-V2 (2024) | Emerging — potential standard |

---

## 11. Interview Questions & Answers

### Q1: What are scaling laws? What did Chinchilla show about compute-optimal training?

**A**: Scaling laws describe how model performance (test loss) improves as a function of model size ($N$), training data ($D$), and compute ($C$). They follow **power laws**:

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}
$$

**Kaplan et al. (2020)** found that performance improves predictably with scale, and recommended prioritizing model size over data: $N_{\text{opt}} \propto C^{0.73}$.

**Chinchilla (2022)** revised this fundamentally. By training 400+ models across different sizes and data budgets, they found that parameters and data should scale **equally**:

$$
N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}
$$

The optimal ratio is approximately **20 tokens per parameter**.

**Impact**: This showed that GPT-3 (175B params, 300B tokens = 1.7 tokens/param) was massively undertrained. A 70B model trained on 1.4T tokens (Chinchilla) outperformed GPT-3 despite being 2.5× smaller.

**Modern nuance**: Chinchilla optimizes for training compute. For inference-heavy production use, it's better to **overtrain** smaller models: Llama 3 8B uses 15T tokens (1875 tokens/param). The extra training cost is amortized over millions of inference requests.

---

### Q2: Compare Llama 2 and Llama 3 architectures. What changed and why?

**A**: The core architecture is identical (Pre-RMSNorm, SwiGLU, RoPE, decoder-only). The differences are in scale and design choices:

| Aspect | Llama 2 70B | Llama 3 70B | Why the change |
|--------|------------|------------|----------------|
| Training tokens | 2T | **15T** | Inference-optimal: more data → better quality per inference FLOP |
| Vocabulary | 32K | **128K** | Better multilingual, shorter sequences, less compute per input |
| GQA | 8 KV heads | 8 KV heads | Unchanged — already optimal |
| Context | 4K | 8K (→128K in 3.1) | User demand + RoPE scaling ($\theta = 500000$) |
| FFN dimension (8B) | 11008 | **14336** | More capacity per layer |

**The biggest change is data, not architecture**. Llama 3 uses 7.5× more training tokens with more aggressive quality filtering. This shows that at the current frontier, the architecture is largely solved — differentiation comes from data.

**Vocabulary increase** from 32K to 128K is significant: it reduces token count for equivalent text by ~15%, directly reducing inference cost. The embedding layer grows, but embeddings are a small fraction of total parameters at this scale.

---

### Q3: What is Mixture of Experts? How does Mixtral use it? What are the routing mechanisms?

**A**: **MoE** replaces the single FFN in each transformer layer with multiple "expert" FFNs and a learned router:

$$
\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} g_i(\mathbf{x}) \cdot \text{FFN}_i(\mathbf{x})
$$

The **router** computes a probability distribution over experts and selects the top-K:

$$
g_i(\mathbf{x}) = \frac{\exp((\mathbf{W}_g \mathbf{x})_i)}{\sum_{j \in \text{TopK}} \exp((\mathbf{W}_g \mathbf{x})_j)} \quad \text{for } i \in \text{TopK}
$$

**Mixtral 8x7B** uses:
- **8 experts** per layer, each a standard SwiGLU FFN
- **Top-2 routing**: Each token activates exactly 2 experts
- **46.7B total params**, **12.9B active** per token
- Attention layers are shared (not MoE) — only the FFN is replaced
- Result: Llama 2 70B performance at ~1/5 inference cost

**Routing mechanisms in practice**:
- **Top-K**: Select highest-scoring experts (Mixtral, GPT-4 rumored)
- **Expert choice**: Each expert selects its preferred tokens (inverse routing)
- **Soft routing**: All experts contribute with continuous weights (less common, higher cost)

**Load balancing**: Without auxiliary loss, some experts become "dead" (never selected). The standard fix is an auxiliary loss:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i
$$

DeepSeek-V3 innovated by using a bias term in the router instead, avoiding the auxiliary loss's negative impact on model quality.

---

### Q4: Why are small language models (Phi, Gemma) performing so well despite fewer parameters?

**A**: Four factors explain the small model revolution:

**1. Data quality over quantity**: Phi models showed that training on carefully curated, synthetic "textbook-quality" data dramatically improves learning efficiency. A 3.8B model trained on 3.3T tokens of high-quality data outperforms a 13B model trained on lower-quality web data.

**2. Massive overtraining**: Modern small models train far beyond Chinchilla-optimal:
- Chinchilla-optimal for 3.8B: ~76B tokens (20×)
- Phi-3 Mini actual: 3.3T tokens (868×)

Each additional token provides diminishing but non-zero improvement. The extra training cost is justified because inference (where the small model is cheaper) far outweighs training cost.

**3. Knowledge distillation**: Gemma 2 and other models use soft labels from larger teacher models. The small model learns not just the correct answer but the teacher's full probability distribution, transferring nuanced knowledge that raw data alone can't provide.

**4. Architecture inheritance**: Small models now use the same optimized architecture (RMSNorm, SwiGLU, RoPE, GQA) that was validated at 70-400B scale. These design choices are scale-independent — they help at 3B just as much as at 70B.

**The implication for practice**: For many production tasks, a 3-8B model is sufficient. The cost difference is massive: serving a 3.8B model requires one GPU; serving a 70B model requires 2-8 GPUs. If a small model achieves 90% of the quality at 10% of the cost, it's the rational choice.

---

### Q5: Explain the evolution from GPT-1 to GPT-3. What key insight drove each generation?

**A**: Each generation was driven by a single key insight:

**GPT-1 (2018) — "Pretraining transfers"**: A causal language model pretrained on unlabeled text, then fine-tuned on labeled data, outperforms training from scratch on each task. Key insight: **the same base model can be adapted to many tasks**.
- 117M params, BooksCorpus, fine-tune per task

**GPT-2 (2019) — "Tasks are prompts"**: Scale the model and data enough, and tasks emerge without fine-tuning. You just need to prompt the model correctly. Key insight: **language modeling is meta-learning — the model learns to learn from its context**.
- 1.5B params, WebText (40GB), zero-shot via prompting

**GPT-3 (2020) — "Scale is all you need"**: Increase parameters by 100× and capabilities explode. Few-shot in-context learning, basic reasoning, and arithmetic emerge. Key insight: **capabilities emerge at sufficient scale that are absent in smaller models**.
- 175B params, 300B tokens, few-shot ICL, emergent abilities

**The trajectory**:
```
GPT-1:  Pretrain → Fine-tune → Task         (one model per task)
GPT-2:  Pretrain → Prompt → Task             (one model, no training)
GPT-3:  Pretrain → Examples in prompt → Task (one model learns from examples)
```

Each step reduced the amount of task-specific engineering needed, converging toward **general-purpose AI systems** that work across tasks without modification.

---

### Q6: What is the KV cache memory requirement for Llama 3 70B? How does GQA help?

**A**: For each token in the sequence, we cache the K and V vectors across all layers and KV heads.

**KV cache formula**:

$$
\text{Cache size} = 2 \times L \times h_{KV} \times d_k \times n \times \text{bytes per element}
$$

For Llama 3 70B ($L = 80, h_{KV} = 8, d_k = 128, n = \text{sequence length}$):

**With GQA (8 KV heads)**:

$$
\text{Cache} = 2 \times 80 \times 8 \times 128 \times n \times 2 \text{ bytes (FP16)} = 327{,}680 \times n \text{ bytes}
$$

At $n = 8192$: $327{,}680 \times 8192 = 2.68 \text{ GB}$ per sequence.
At $n = 128{,}000$: $327{,}680 \times 128{,}000 = 41.9 \text{ GB}$ per sequence.

**Without GQA (hypothetical, 64 KV heads)**:

$$
\text{Cache} = 2 \times 80 \times 64 \times 128 \times n \times 2 = 2{,}621{,}440 \times n \text{ bytes}
$$

At $n = 8192$: $21.5 \text{ GB}$ — **8× larger**.

**GQA provides 8× KV cache reduction** (from 64 to 8 KV heads), making long-context inference feasible. Without GQA, serving 128K context would require ~335 GB of KV cache per sequence — impossible on current hardware.

---

### Q7: How does DeepSeek's Multi-head Latent Attention (MLA) differ from GQA?

**A**: Both MLA and GQA reduce KV cache size, but through different mechanisms:

**GQA** shares KV heads across query heads. With 8 KV groups for 64 query heads: 8× reduction. The full-dimensional K and V vectors are cached.

**MLA** compresses K and V into a **low-rank latent representation** before caching:

$$
\mathbf{c}_t = \mathbf{W}^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
$$

Only $\mathbf{c}_t$ (dimension $d_c \ll d$) is cached. During attention, it's decompressed:

$$
\mathbf{k}_t = \mathbf{W}^{UK} \mathbf{c}_t, \quad \mathbf{v}_t = \mathbf{W}^{UV} \mathbf{c}_t
$$

**Comparison**:

| Aspect | GQA | MLA |
|--------|-----|-----|
| Cache reduction | $h_Q / h_{KV}$ (e.g., 8×) | $d / d_c$ (e.g., 20-50×) |
| Cache per token per layer | $2 \times h_{KV} \times d_k$ | $d_c$ (single vector) |
| Quality impact | Minimal (proven at scale) | Minimal (validated by DeepSeek) |
| Extra computation | None | Decompression matmuls during attention |
| Adoption | Universal (Llama, Mistral, etc.) | DeepSeek only (so far) |

MLA achieves much greater cache reduction (~93% vs GQA's ~87.5% for 8-group GQA) but adds computational overhead for the decompression. It represents a potentially important direction for extremely long-context models.

---

### Q8: Explain sliding window attention. How does information propagate beyond the window?

**A**: Sliding window attention (SWA) restricts each token to attending only to the previous $W$ tokens (the "window"), instead of all previous tokens:

$$
\text{Attn}(i, j) = 0 \quad \text{if } j < i - W
$$

For Mistral 7B: $W = 4096$.

**Information propagation**: While each layer can only see $W$ positions back, information from distant positions propagates **through intermediate layers**. Layer 1 at position $i$ attends to positions $[i-W, i]$. Layer 2 at position $i$ attends to Layer 1 outputs at $[i-W, i]$, which themselves contain information from $[i-2W, i]$. After $L$ layers:

$$
\text{Effective receptive field} = L \times W
$$

For Mistral: $32 \times 4096 = 131{,}072$ positions.

```
Layer 3:  Position i sees info originally from [i-3W, i]
Layer 2:  Position i sees info originally from [i-2W, i]
Layer 1:  Position i sees info originally from [i-W, i]
```

**Trade-offs**:
- **Advantage**: Fixed KV cache size ($W$ per layer, regardless of sequence length) → O(1) memory
- **Disadvantage**: Information from distant positions is indirect (passed through multiple layers and transformations), so it may be "diluted." For tasks requiring precise recall of distant details, SWA may underperform full attention.

In practice, most important context is local, and SWA's efficiency makes long-sequence inference practical.

---

### Q9: What are the key differences between training for compute-optimality vs inference-optimality?

**A**: These are two different optimization targets:

**Compute-optimal (Chinchilla)**: Minimize training cost for a target performance level.

$$
\text{Minimize } C = 6ND \quad \text{subject to } L(N, D) \leq L_{\text{target}}
$$

Result: $D_{\text{opt}} \approx 20N$. Use a large model and moderate data.

**Inference-optimal**: Minimize per-token inference cost for a target performance level.

$$
\text{Minimize } N \quad \text{subject to } L(N, D) \leq L_{\text{target}} \text{ (for some feasible } D\text{)}
$$

Result: Use the **smallest model possible** and train it on **as much data as feasible**. $D \gg 20N$.

**Comparison**:

| Aspect | Compute-Optimal | Inference-Optimal |
|--------|----------------|-------------------|
| Goal | Min training cost | Min serving cost |
| $D/N$ ratio | ~20 | 200-2000+ |
| Model size | Larger | Smaller |
| Training cost | Lower | Higher (more tokens) |
| Inference cost | Higher (bigger model) | Lower (smaller model) |
| Example | Chinchilla 70B / 1.4T tokens | Llama 3 8B / 15T tokens |

**When to choose which**:
- **Research/one-off**: Compute-optimal (minimize total cost)
- **Production/high-volume**: Inference-optimal (amortize training across millions of requests)
- **API service**: Inference-optimal (per-query cost dominates)

This is why Llama 3 8B trained on 15T tokens despite Chinchilla suggesting ~160B would suffice — Meta optimized for the billions of inference requests the model would serve.

---

### Q10: If you were starting an LLM project today, what model would you choose and why?

**A**: It depends on the use case. Here's my decision framework:

**General-purpose chat/reasoning (API)**: Use a frontier API model (GPT-4o, Claude, Gemini). Best quality, no infrastructure burden.

**Production deployment (cost-sensitive)**:
- **Llama 3.1 8B** fine-tuned for your task. 15T tokens of pretraining gives excellent base capabilities. Single GPU deployment ($0.10-0.20/M tokens self-hosted).
- **Mistral 7B** if you need sliding window attention for long sequences.

**Maximum open-source quality**:
- **Llama 3.1 70B** for dense model. Best open-source quality in most benchmarks.
- **Mixtral 8x22B** if you have the memory but want lower inference cost than a 70B dense model.

**Research / limited compute**:
- **Phi-3 3.8B** or **Gemma 2 9B** — excellent quality per parameter, single consumer GPU.

**Reasoning-heavy tasks**:
- **DeepSeek-R1** (open) or **QwQ 32B** — specialized reasoning models.
- Or use **o1/o3** via API for the best reasoning quality.

**Key considerations**:
1. **Fine-tuning**: Always try fine-tuning a small model before jumping to a larger one. A fine-tuned 8B model often beats a prompted 70B model on specific tasks.
2. **License**: Check commercial use terms (Llama's license, Mistral's Apache 2.0, etc.)
3. **Context length**: If you need 128K context, choose Llama 3.1 or Mistral variants with proper RoPE scaling.
4. **Latency budget**: If you need <100ms responses, you're limited to ~7B models on single GPUs.

---

*This topic covers the decoder model family — the dominant paradigm in modern AI. Next: [Topic 13: Encoder-Decoder & Unified Models](13_Encoder_Decoder_Models.md) — T5, BART, and the debate over why decoder-only won.*
