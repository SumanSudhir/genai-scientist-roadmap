# Topic 9: The Transformer Architecture (Deep Dive)

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 9 of 28
> **Scope**: Full encoder-decoder transformer, encoder/decoder blocks, FFN, layer normalization, residual connections, parameter counting, the three architectural variants (encoder-only, decoder-only, encoder-decoder), modern design choices
> **Why this matters**: The transformer is THE architecture. Every LLM, every vision transformer, every modern AI system is built on this blueprint. You must be able to draw it from memory, count its parameters, explain every design choice, and articulate why modern variants deviate from the original.
> **Previous**: [Topic 8: Attention Mechanisms](08_Attention_Mechanisms.md)
> **Next**: [Topic 10: Positional Encodings](10_Positional_Encodings.md)

---

## Table of Contents

1. [The Big Picture — What the Transformer Solves](#1-the-big-picture--what-the-transformer-solves)
2. [Input Pipeline: From Tokens to Vectors](#2-input-pipeline-from-tokens-to-vectors)
3. [The Encoder Block](#3-the-encoder-block)
4. [The Decoder Block](#4-the-decoder-block)
5. [Multi-Head Attention (Recap & Context)](#5-multi-head-attention-recap--context)
6. [The Feed-Forward Network (FFN)](#6-the-feed-forward-network-ffn)
7. [Residual Connections — Why Depth Is Possible](#7-residual-connections--why-depth-is-possible)
8. [Layer Normalization — Stabilizing Training](#8-layer-normalization--stabilizing-training)
9. [Pre-LN vs Post-LN — The Critical Design Choice](#9-pre-ln-vs-post-ln--the-critical-design-choice)
10. [The Output Head](#10-the-output-head)
11. [Full Architecture Walkthrough — Putting It All Together](#11-full-architecture-walkthrough--putting-it-all-together)
12. [Parameter Counting — Know Your Model Size](#12-parameter-counting--know-your-model-size)
13. [The Three Variants — Encoder-Only, Decoder-Only, Encoder-Decoder](#13-the-three-variants--encoder-only-decoder-only-encoder-decoder)
14. [Modern Transformer Design Choices (2023-2026)](#14-modern-transformer-design-choices-2023-2026)
15. [Common Misconceptions](#15-common-misconceptions)
16. [Interview Questions & Answers](#16-interview-questions--answers)

---

## 1. The Big Picture — What the Transformer Solves

Before diving into components, understand the problem the transformer solves and why it was revolutionary.

### 1.1 The Three Problems with RNN-Based Seq2Seq

| Problem | RNN/LSTM | Transformer |
|---------|----------|-------------|
| **Sequential computation** | Must process tokens one-by-one; can't parallelize over time steps | Processes all tokens simultaneously; fully parallelizable |
| **Long-range dependencies** | Signal decays over distance despite LSTM gates | Direct connection between any two positions via attention |
| **Information bottleneck** | Entire input compressed into single context vector | Every decoder step can attend to all encoder positions |

### 1.2 The Core Insight

The transformer's key insight: **replace recurrence entirely with attention**. Instead of propagating information through hidden states over time, let every position directly attend to every other position.

This gives us:

1. **Parallelism**: All positions are processed simultaneously during training (unlike RNNs)
2. **Constant path length**: Any two positions are connected by a single attention operation, regardless of distance
3. **Flexible information flow**: The model learns *which* positions to attend to, rather than relying on fixed sequential ordering

### 1.3 The Original Paper

"Attention Is All You Need" (Vaswani et al., 2017) introduced the transformer for machine translation. The architecture they proposed is an **encoder-decoder** model:

- **Encoder**: Reads the source sentence, builds rich representations
- **Decoder**: Generates the target sentence one token at a time, attending to the encoder's output

This original design was quickly adapted into three variants (Section 13), but understanding the full encoder-decoder version is essential — every variant is a subset of it.

---

## 2. Input Pipeline: From Tokens to Vectors

Before any attention or computation happens, raw tokens must be converted into continuous vectors the transformer can process.

### 2.1 Token Embedding

Each token in the vocabulary is mapped to a dense vector:

$$
\mathbf{e}_i = \text{Embed}(x_i) \in \mathbb{R}^{d_{\text{model}}}
$$

where $x_i$ is the token ID and $d_{\text{model}}$ is the model's hidden dimension (512 in the original paper, 4096+ in modern LLMs).

The embedding matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d_{\text{model}}}$ has one row per vocabulary token. This is a learned parameter.

**Scaling**: The original paper scales embeddings by $\sqrt{d_{\text{model}}}$:

$$
\mathbf{e}_i = \mathbf{W}_E[x_i] \cdot \sqrt{d_{\text{model}}}
$$

**Why scale?** Embedding vectors are initialized with small values (variance $\approx 1/d_{\text{model}}$). Without scaling, they would be much smaller in magnitude than positional encodings, drowning out the token identity signal. Multiplying by $\sqrt{d_{\text{model}}}$ brings them to a comparable scale.

### 2.2 Positional Encoding

Transformers have no inherent notion of position — attention treats the input as a **set**, not a sequence. Positional encodings inject order information.

The original paper uses **sinusoidal positional encodings**:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

where $pos$ is the position in the sequence and $i$ is the dimension index.

The final input representation combines both:

$$
\mathbf{h}_i^{(0)} = \text{Embed}(x_i) \cdot \sqrt{d_{\text{model}}} + PE(i)
$$

> **Note**: Positional encodings are covered in depth in [Topic 10](10_Positional_Encodings.md). Here we just need to know they exist and are added to the input.

### 2.3 Input Dropout

After combining embeddings and positional encodings, dropout is applied:

$$
\mathbf{h}_i^{(0)} = \text{Dropout}\left(\text{Embed}(x_i) \cdot \sqrt{d_{\text{model}}} + PE(i)\right)
$$

This regularizes the input representation. The original paper uses dropout $= 0.1$.

### 2.4 Visual Summary

```
Token IDs:     [x_1,  x_2,  x_3, ..., x_n]
                 |      |      |          |
Embed (lookup): [e_1,  e_2,  e_3, ..., e_n]    × sqrt(d_model)
                 +      +      +          +
Positional:    [PE_1, PE_2, PE_3, ..., PE_n]
                 |      |      |          |
Dropout:       [h_1,  h_2,  h_3, ..., h_n]    → Input to Encoder/Decoder
```

---

## 3. The Encoder Block

The encoder transforms input representations into rich, contextual representations that capture the meaning of each position in the context of the full sequence.

### 3.1 Single Encoder Block

Each encoder block consists of two sub-layers:

```
Input
  │
  ├──────────────────┐
  │                  │
  ▼                  │
Multi-Head           │  (Residual Connection)
Self-Attention       │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
Layer Norm
  │
  ├──────────────────┐
  │                  │
  ▼                  │
Feed-Forward         │  (Residual Connection)
Network (FFN)        │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
Layer Norm
  │
Output
```

Mathematically (Post-LN, the original paper's formulation):

$$
\mathbf{z}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{MultiHeadAttn}\left(\mathbf{h}^{(l-1)}, \mathbf{h}^{(l-1)}, \mathbf{h}^{(l-1)}\right)\right)
$$

$$
\mathbf{h}^{(l)} = \text{LayerNorm}\left(\mathbf{z}^{(l)} + \text{FFN}\left(\mathbf{z}^{(l)}\right)\right)
$$

where $\mathbf{h}^{(l-1)}$ is the input to layer $l$, and $\text{MultiHeadAttn}(Q, K, V)$ takes queries, keys, and values (all the same in self-attention).

### 3.2 Stacking Encoder Blocks

The encoder is a stack of $N$ identical blocks (6 in the original paper):

$$
\text{Encoder}(\mathbf{X}) = \text{Block}_N \circ \text{Block}_{N-1} \circ \cdots \circ \text{Block}_1(\mathbf{X})
$$

Each block refines the representations. Lower layers tend to capture local/syntactic patterns; higher layers capture more abstract/semantic relationships.

### 3.3 What the Encoder Produces

The encoder's final output is a sequence of vectors $\{\mathbf{h}_1^{(N)}, \mathbf{h}_2^{(N)}, \ldots, \mathbf{h}_n^{(N)}\}$, one per input position. Each vector is a **contextual representation** — it encodes not just the token at that position, but its relationship to every other position in the input.

These vectors are what the decoder attends to via cross-attention.

---

## 4. The Decoder Block

The decoder generates output tokens one at a time (autoregressively), using both its own prior outputs and the encoder's representations.

### 4.1 Single Decoder Block

Each decoder block has **three** sub-layers (compared to the encoder's two):

```
Input (previous decoder outputs)
  │
  ├──────────────────┐
  │                  │
  ▼                  │
Masked Multi-Head    │  (Residual)
Self-Attention       │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
Layer Norm
  │
  ├──────────────────┐
  │                  │
  ▼                  │
Multi-Head           │  (Residual)
Cross-Attention      │
(Q=decoder, KV=encoder)
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
Layer Norm
  │
  ├──────────────────┐
  │                  │
  ▼                  │
Feed-Forward         │  (Residual)
Network (FFN)        │
  │                  │
  ▼                  │
  + ◄────────────────┘
  │
  ▼
Layer Norm
  │
Output
```

Mathematically (Post-LN):

$$
\mathbf{a}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{MaskedMultiHeadAttn}\left(\mathbf{h}^{(l-1)}, \mathbf{h}^{(l-1)}, \mathbf{h}^{(l-1)}\right)\right)
$$

$$
\mathbf{b}^{(l)} = \text{LayerNorm}\left(\mathbf{a}^{(l)} + \text{MultiHeadAttn}\left(\mathbf{a}^{(l)}, \mathbf{h}_{\text{enc}}, \mathbf{h}_{\text{enc}}\right)\right)
$$

$$
\mathbf{h}^{(l)} = \text{LayerNorm}\left(\mathbf{b}^{(l)} + \text{FFN}\left(\mathbf{b}^{(l)}\right)\right)
$$

### 4.2 The Three Attention Patterns

This is critical to understand — the decoder uses three distinct types of attention:

| Sub-Layer | Q from | K, V from | Mask? | Purpose |
|-----------|--------|-----------|-------|---------|
| **Masked Self-Attention** | Decoder | Decoder | Yes (causal) | Let each output position attend to previous output positions only |
| **Cross-Attention** | Decoder | Encoder | No | Let each output position attend to all input positions |
| **Encoder Self-Attention** (in encoder) | Encoder | Encoder | No | Let each input position attend to all other input positions |

### 4.3 Why Masked (Causal) Self-Attention?

During training, the decoder sees all target tokens simultaneously (for parallelism). But at inference time, it generates one token at a time — it shouldn't be able to "peek" at future tokens.

The causal mask enforces this:

$$
\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (can attend)} \\ -\infty & \text{if } j > i \text{ (blocked)} \end{cases}
$$

Applied to attention scores before softmax:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \text{Mask}\right)\mathbf{V}
$$

The $-\infty$ values become 0 after softmax, effectively preventing information flow from future positions.

```
Causal Attention Mask (4 tokens):

         Key positions
         1    2    3    4
    1  [ 0   -∞   -∞   -∞ ]     Token 1 sees only itself
Q   2  [ 0    0   -∞   -∞ ]     Token 2 sees 1, 2
    3  [ 0    0    0   -∞ ]     Token 3 sees 1, 2, 3
    4  [ 0    0    0    0  ]     Token 4 sees all
```

### 4.4 Why Cross-Attention?

Cross-attention is how the decoder "reads" the source input. At each decoder position:

- **Queries** come from the decoder (what am I looking for?)
- **Keys and Values** come from the encoder (what's available in the input?)

This is conceptually similar to Bahdanau attention but generalized to multi-head scaled dot-product form.

**Example**: Translating "The cat sat on the mat" to French:
- When generating "le", the decoder's query might strongly attend to "the" in the encoder output
- When generating "chat", it attends to "cat"
- The attention pattern forms a soft alignment between source and target

---

## 5. Multi-Head Attention (Recap & Context)

See [Topic 8](08_Attention_Mechanisms.md) for the full derivation. Here we recap the essentials as they appear in the transformer.

### 5.1 The Formulation

Given input vectors, we project them into $h$ separate heads:

$$
\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)
$$

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O
$$

where:
- $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ are per-head projections
- $\mathbf{W}^O \in \mathbb{R}^{hd_k \times d_{\text{model}}}$ is the output projection
- $d_k = d_{\text{model}} / h$ (dimension per head)

### 5.2 Original Paper Configuration

| Parameter | Value |
|-----------|-------|
| $d_{\text{model}}$ | 512 |
| $h$ (number of heads) | 8 |
| $d_k = d_v$ | 64 |
| $N$ (layers) | 6 |

### 5.3 What Different Heads Learn

Research has shown that individual attention heads specialize:
- **Syntactic heads**: Track subject-verb agreement, dependency parsing
- **Positional heads**: Attend to specific relative positions (previous token, next token)
- **Rare token heads**: Attend to infrequent or important tokens
- **Separator heads**: Attend to delimiter or punctuation tokens

This specialization is why multi-head attention outperforms single-head attention with the same total dimension.

---

## 6. The Feed-Forward Network (FFN)

The FFN is the often-overlooked workhorse of the transformer. It sits in every block and accounts for **two-thirds** of the model's parameters.

### 6.1 Standard FFN

The FFN is a simple two-layer MLP applied independently to each position:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right) + \mathbf{b}_2
$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ — expands the dimension
- $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ — projects back down
- $d_{ff} = 4 \times d_{\text{model}}$ (original paper: $d_{ff} = 2048$ for $d_{\text{model}} = 512$)
- $\sigma$ = ReLU in the original paper

```
    d_model          d_ff            d_model
   ┌─────┐      ┌──────────┐      ┌─────┐
x ─┤ W_1 ├─────►│   ReLU   ├─────►│ W_2 ├──► output
   └─────┘      └──────────┘      └─────┘
    (512)         (2048)            (512)
              "expand"          "compress"
```

### 6.2 Why 4x Expansion?

The expansion factor of 4 is not arbitrary:

1. **Capacity**: The FFN needs enough capacity to store factual knowledge. Research shows FFN layers act as **key-value memories** — each neuron in the intermediate layer corresponds to a pattern detector, and its output weight encodes the associated information.

2. **Bottleneck design**: The expand-then-compress pattern forces the network to create sparse, distributed representations in the high-dimensional space, then compress the relevant information back down.

3. **Empirical finding**: The original paper tested different ratios; $4 \times$ gave the best trade-off between performance and computation.

### 6.3 Modern FFN Variants

#### GELU Activation (BERT, GPT)

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
$$

GELU provides a smooth, non-monotonic activation that allows small negative gradients (unlike ReLU's hard zero).

#### SwiGLU (Llama, PaLM, Mistral)

SwiGLU replaces the standard FFN with a **gated** variant:

$$
\text{SwiGLU}(\mathbf{x}) = \left(\text{Swish}(\mathbf{x}\mathbf{W}_1) \odot (\mathbf{x}\mathbf{W}_3)\right)\mathbf{W}_2
$$

where:
- $\text{Swish}(x) = x \cdot \sigma(x)$ (also called SiLU)
- $\odot$ is element-wise multiplication
- $\mathbf{W}_3$ is a third weight matrix (the "gate")

**Why SwiGLU?** The gating mechanism allows the network to selectively activate dimensions. Empirically, SwiGLU gives ~1-2% improvement across benchmarks compared to ReLU or GELU FFNs.

**Parameter note**: SwiGLU has 3 weight matrices instead of 2. To keep the parameter count similar, the intermediate dimension is reduced from $4d$ to $\frac{8d}{3}$ (approximately $2.67d$).

### 6.4 FFN as Key-Value Memory

A landmark insight from "Transformer Feed-Forward Layers Are Key-Value Memories" (Geva et al., 2021):

- Each row of $\mathbf{W}_1$ acts as a **key** (pattern detector)
- The corresponding column of $\mathbf{W}_2$ acts as a **value** (associated output)
- The ReLU activation acts as a **gate** (only fire for matching patterns)

This means:

$$
\text{FFN}(\mathbf{x}) = \sum_{i=1}^{d_{ff}} \text{ReLU}(\mathbf{w}_{1,i}^T \mathbf{x}) \cdot \mathbf{w}_{2,i}
$$

Each term: IF the input matches key $i$ (positive dot product), THEN add value $i$ to the output, scaled by the match strength.

**This is why larger FFNs = more knowledge**: More neurons mean more key-value pairs, which means more factual knowledge the model can store.

### 6.5 Position-Wise Application

A crucial detail: the FFN is applied **independently and identically** to each position in the sequence. There is no interaction between positions within the FFN — that's the attention layer's job.

Think of it as:
- **Attention** = communication between positions (the "telephone")
- **FFN** = processing within each position (the "brain")

---

## 7. Residual Connections — Why Depth Is Possible

### 7.1 The Problem Without Residuals

Consider a network with $L$ layers, where each layer computes $\mathbf{h}^{(l)} = f^{(l)}(\mathbf{h}^{(l-1)})$. During backpropagation:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \cdot \prod_{l=1}^{L} \frac{\partial f^{(l)}}{\partial \mathbf{h}^{(l-1)}}
$$

This product of $L$ Jacobian matrices causes:
- **Vanishing gradients** if $\|J_l\| < 1$ consistently → gradient shrinks exponentially
- **Exploding gradients** if $\|J_l\| > 1$ consistently → gradient grows exponentially

A 96-layer transformer without residual connections would be **untrainable**.

### 7.2 The Residual Connection

Instead of $\mathbf{h}^{(l)} = f^{(l)}(\mathbf{h}^{(l-1)})$, we compute:

$$
\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + f^{(l)}(\mathbf{h}^{(l-1)})
$$

The function $f^{(l)}$ now only needs to learn the **residual** — the difference from the input.

### 7.3 Why This Fixes Gradient Flow

The gradient through a residual connection:

$$
\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} = \mathbf{I} + \frac{\partial f^{(l)}}{\partial \mathbf{h}^{(l-1)}}
$$

The identity matrix $\mathbf{I}$ guarantees a direct gradient path. Even if $\frac{\partial f^{(l)}}{\partial \mathbf{h}^{(l-1)}}$ is small, the gradient is at least $\mathbf{I}$.

Over $L$ layers, there are now $2^L$ possible paths for gradient flow (each layer can either contribute its Jacobian or pass through via identity). This exponential number of paths prevents the vanishing problem.

### 7.4 Residual Connections in the Transformer

Every sub-layer (attention and FFN) in the transformer has a residual connection:

$$
\text{output} = \text{input} + \text{SubLayer}(\text{input})
$$

This is why the transformer requires all sub-layers to have the same output dimension as $d_{\text{model}}$ — the addition requires matching shapes.

### 7.5 Intuitive View

Think of residual connections as creating an **ensemble of paths** through the network:

```
Input ─────────────────────────────────────────► Output
  │                                               ▲
  └──► Layer 1 ──────────────────────────────────►+
  │       │                                       ▲
  └───────└──► Layer 2 ──────────────────────────►+
  │       │       │                               ▲
  └───────└───────└──► Layer 3 ──────────────────►+
```

The final output is effectively a sum over all possible subsets of layers. The network behaves like an **ensemble** of shallower networks.

---

## 8. Layer Normalization — Stabilizing Training

### 8.1 Why Normalization?

Neural networks are sensitive to the **scale** of activations. As values pass through many layers, they can grow or shrink uncontrollably. Normalization keeps activations in a stable range, which:

1. Allows higher learning rates (faster training)
2. Reduces sensitivity to weight initialization
3. Acts as a mild regularizer

### 8.2 Layer Norm vs Batch Norm

**Batch Normalization** (BatchNorm) normalizes across the **batch dimension** for each feature:

$$
\text{BatchNorm}(x_{ij}) = \frac{x_{ij} - \mu_j^{(\text{batch})}}{\sqrt{(\sigma_j^{(\text{batch})})^2 + \epsilon}}
$$

where $\mu_j^{(\text{batch})}$ is the mean of feature $j$ across all samples in the batch.

**Layer Normalization** (LayerNorm) normalizes across the **feature dimension** for each sample:

$$
\text{LayerNorm}(\mathbf{x}_i) = \frac{\mathbf{x}_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}
$$

where:

$$
\mu_i = \frac{1}{d} \sum_{j=1}^{d} x_{ij}, \qquad \sigma_i^2 = \frac{1}{d} \sum_{j=1}^{d} (x_{ij} - \mu_i)^2
$$

and $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are learned scale and shift parameters.

### 8.3 Why Transformers Use LayerNorm, Not BatchNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch size | Yes — fails with small batches | No — each sample normalized independently |
| Handles variable-length sequences | Poorly — what's the "batch" for padding tokens? | Fine — each token normalized independently |
| Training vs inference | Different behavior (uses running stats at inference) | Same behavior always |
| Works with autoregressive generation | Problematic — batch size is often 1 | Perfect |

The key reasons:
1. **Variable sequence lengths** in NLP make batch statistics unreliable
2. **Autoregressive generation** (batch size = 1) makes BatchNorm useless
3. **Consistency** between training and inference (no running mean/variance)

### 8.4 RMSNorm — The Modern Simplification

Many modern transformers (Llama, Mistral, Gemma) use **RMSNorm** instead of LayerNorm:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}
$$

where:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

**Differences from LayerNorm**:
- No mean subtraction (re-centering)
- No learned bias $\boldsymbol{\beta}$
- Only learned scale $\boldsymbol{\gamma}$

**Why?** The mean subtraction in LayerNorm provides negligible benefit in practice, but adds computation. RMSNorm is 10-15% faster with equivalent performance.

---

## 9. Pre-LN vs Post-LN — The Critical Design Choice

This is one of the most important architectural decisions in transformer design, and a common interview question.

### 9.1 Post-LN (Original Paper)

The original transformer places LayerNorm **after** the residual addition:

$$
\mathbf{h}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{SubLayer}(\mathbf{h}^{(l-1)})\right)
$$

```
x ──┬──► SubLayer ──► + ──► LayerNorm ──► output
    └────────────────►↑
```

### 9.2 Pre-LN (Modern Standard)

Modern transformers place LayerNorm **before** the sub-layer:

$$
\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + \text{SubLayer}\left(\text{LayerNorm}(\mathbf{h}^{(l-1)})\right)
$$

```
x ──┬──► LayerNorm ──► SubLayer ──► + ──► output
    └───────────────────────────────►↑
```

### 9.3 Why Pre-LN Won

The difference seems trivial but has profound consequences for training stability:

**Post-LN problems**:

1. **Gradient amplification at initialization**: In Post-LN, the residual branch goes through LayerNorm. At initialization, this creates gradient amplification — the gradient at layer $l$ is roughly $O(L^{1/2})$ times larger than at layer $L$ (the output). This makes deeper layers update much more slowly.

2. **Requires careful warmup**: Post-LN transformers need extensive learning rate warmup (thousands of steps) to avoid divergence. Without warmup, they often fail to train.

3. **Output scale grows**: Without normalization on the residual path, the magnitude of the residual stream grows with depth.

**Pre-LN advantages**:

1. **Well-behaved gradients**: The gradient flows through the residual connection without passing through LayerNorm, keeping gradients stable across depth.

2. **No warmup needed**: Pre-LN transformers are much more robust to learning rate choices and can often train without warmup.

3. **Easier to scale**: Pre-LN is why we can train 100+ layer transformers reliably.

**Pre-LN disadvantage**:

1. **Slightly lower final performance**: Some studies find Post-LN achieves marginally better performance *when it converges*. This is because Pre-LN's direct residual path can lead to later layers contributing less (the "representation collapse" concern).

### 9.4 Post-LN with Fixes

Some architectures use Post-LN with additional tricks:
- **DeepNorm** (Microsoft): Scale the residual connection by $\alpha$ and apply a special initialization
- **Admin** initialization: Initialize layers so that Post-LN behaves like Pre-LN early in training

### 9.5 Summary Table

| Aspect | Post-LN | Pre-LN |
|--------|---------|--------|
| LayerNorm position | After residual add | Before sub-layer |
| Training stability | Fragile, needs warmup | Robust, minimal warmup |
| Final performance | Slightly higher (when it works) | Very slightly lower |
| Used in practice | BERT, original Transformer | GPT-2/3/4, Llama, Mistral, most modern LLMs |
| Deep networks (96+ layers) | Difficult to train | Standard approach |

### 9.6 The Final LayerNorm

In Pre-LN architectures, a **final LayerNorm** is applied after the last block:

$$
\mathbf{h}_{\text{out}} = \text{LayerNorm}(\mathbf{h}^{(L)})
$$

This is necessary because in Pre-LN, the output of the last block is an unnormalized residual sum. Without this final norm, the output magnitudes would grow with depth.

---

## 10. The Output Head

The output head converts the decoder's final hidden states into a probability distribution over the vocabulary.

### 10.1 Linear Projection + Softmax

$$
P(y_t = w \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_{\text{head}} \cdot \mathbf{h}_t^{(L)} + \mathbf{b}_{\text{head}})_w
$$

where:
- $\mathbf{h}_t^{(L)} \in \mathbb{R}^{d_{\text{model}}}$ is the decoder's output at position $t$
- $\mathbf{W}_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$ projects to vocabulary size
- The softmax produces a probability for each token in the vocabulary

### 10.2 Weight Tying

A widely used technique: **tie the output projection weights with the embedding matrix**:

$$
\mathbf{W}_{\text{head}} = \mathbf{W}_E
$$

**Why this works**:
- The embedding maps tokens → vectors. The output head maps vectors → tokens. These are inverse operations — it makes sense to use the same weight matrix (transposed).
- Reduces parameters significantly (for $V = 32000$ and $d = 4096$, this saves ~131M parameters)
- Acts as regularization — forces the output space and embedding space to be consistent
- Used in the original transformer, GPT-2, and many modern models

**Who doesn't tie**: Some modern models (Llama 2/3, Mistral) do **not** tie weights, as they found untied weights give better performance at scale.

### 10.3 Temperature and Sampling

At inference time, the raw logits $\mathbf{z} = \mathbf{W}_{\text{head}} \cdot \mathbf{h}_t^{(L)}$ can be modified before softmax:

$$
P(y_t = w) = \frac{\exp(z_w / T)}{\sum_{w'} \exp(z_{w'} / T)}
$$

where $T$ is the temperature. This is covered in detail in [Topic 17: Decoding Strategies](17_Decoding_Strategies.md).

---

## 11. Full Architecture Walkthrough — Putting It All Together

### 11.1 The Complete Encoder-Decoder Transformer

```
                    ┌─────────────────────────────────────────┐
                    │              OUTPUT HEAD                  │
                    │  Linear (d_model → V) + Softmax          │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────────┐
                    │           DECODER (× N layers)           │
                    │                                          │
                    │  ┌─────────────────────────────────────┐ │
                    │  │ FFN + Add & Norm                    │ │
                    │  ├─────────────────────────────────────┤ │
                    │  │ Cross-Attention + Add & Norm        │◄├───── Encoder Output
                    │  │ (Q=decoder, K,V=encoder)            │ │
                    │  ├─────────────────────────────────────┤ │
                    │  │ Masked Self-Attention + Add & Norm  │ │
                    │  └─────────────────────────────────────┘ │
                    │                                          │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────────┐
                    │    Output Embedding + Positional Enc     │
                    └──────────────────┬──────────────────────┘
                                       │
                                 Output Tokens
                              (shifted right)


    ┌──────────────────────────────────────────────────┐
    │              ENCODER (× N layers)                 │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐  │
    │  │ FFN + Add & Norm                            │  │
    │  ├─────────────────────────────────────────────┤  │
    │  │ Self-Attention + Add & Norm                 │  │
    │  └─────────────────────────────────────────────┘  │
    │                                                   │
    └──────────────────────┬───────────────────────────┘
                           │
    ┌──────────────────────┴───────────────────────────┐
    │       Input Embedding + Positional Encoding       │
    └──────────────────────┬───────────────────────────┘
                           │
                      Input Tokens
```

### 11.2 Data Flow for Machine Translation

Let's trace a concrete example: translating "I love cats" → "J'aime les chats"

**Step 1: Encode the input**
```
Input tokens:  ["I", "love", "cats"]
Embed + PE:    [h_I, h_love, h_cats]
                    │
              6 × Encoder Blocks
                    │
Encoder out:   [e_I, e_love, e_cats]    ← Rich contextual representations
```

**Step 2: Decode autoregressively**

```
Time step 1: Input = [<start>]
  Masked self-attn:  <start> attends to itself
  Cross-attn:        <start> attends to [e_I, e_love, e_cats]
  FFN:               processes
  Output head:       P(y_1 | <start>) → "J'aime" (highest probability)

Time step 2: Input = [<start>, J'aime]
  Masked self-attn:  J'aime attends to [<start>, J'aime]
  Cross-attn:        J'aime attends to [e_I, e_love, e_cats]
  Output head:       P(y_2 | <start>, J'aime) → "les"

Time step 3: Input = [<start>, J'aime, les]
  ...continues until <end> token is generated
```

### 11.3 Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Decoder input** | Entire target sequence (teacher forcing) | One token at a time (autoregressive) |
| **Parallelism** | Full — all positions computed simultaneously | Sequential — each token depends on previous |
| **Causal mask** | Applied to prevent future leakage | Naturally satisfied (future tokens don't exist yet) |
| **Speed** | Fast (parallel) | Slow (sequential), mitigated by KV cache |

---

## 12. Parameter Counting — Know Your Model Size

Being able to count parameters is essential for interviews and for understanding model scale.

### 12.1 Notation

| Symbol | Meaning | Original Paper |
|--------|---------|----------------|
| $d$ | Model dimension ($d_{\text{model}}$) | 512 |
| $h$ | Number of attention heads | 8 |
| $d_k$ | Head dimension ($d/h$) | 64 |
| $d_{ff}$ | FFN intermediate dimension | 2048 |
| $V$ | Vocabulary size | 37000 |
| $L$ | Number of layers | 6 |

### 12.2 Parameters Per Component

#### Embedding Layer

$$
\text{Params}_{\text{embed}} = V \times d
$$

(One $d$-dimensional vector per vocabulary token.)

#### Single Attention Sub-Layer

Four weight matrices: $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d}$ and $\mathbf{W}^O \in \mathbb{R}^{d \times d}$

$$
\text{Params}_{\text{attn}} = 4d^2
$$

(Plus $4d$ for biases, often omitted in modern models.)

**Note**: This is the same whether you think of it as $h$ heads with $d \times d_k$ each or one big $d \times d$ matrix — the total is identical.

#### Single FFN Sub-Layer (Standard)

Two weight matrices: $\mathbf{W}_1 \in \mathbb{R}^{d_{ff} \times d}$ and $\mathbf{W}_2 \in \mathbb{R}^{d \times d_{ff}}$

$$
\text{Params}_{\text{FFN}} = 2 \times d \times d_{ff} = 2 \times d \times 4d = 8d^2
$$

(Plus $d_{ff} + d$ for biases.)

#### Single FFN Sub-Layer (SwiGLU)

Three weight matrices: $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d_{ff} \times d}$ and $\mathbf{W}_2 \in \mathbb{R}^{d \times d_{ff}}$

With reduced $d_{ff} = \frac{8d}{3}$:

$$
\text{Params}_{\text{SwiGLU}} = 3 \times d \times \frac{8d}{3} = 8d^2
$$

(Same total as standard FFN — by design.)

#### Layer Normalization

$$
\text{Params}_{\text{LN}} = 2d \quad (\boldsymbol{\gamma} \text{ and } \boldsymbol{\beta})
$$

Per sub-layer. Negligible compared to attention and FFN.

#### Output Head (Unembedding)

$$
\text{Params}_{\text{head}} = V \times d
$$

If weight-tied with embeddings, this is 0 additional parameters.

### 12.3 Total Parameter Count

#### Encoder-Decoder Transformer (Original)

$$
\text{Encoder layer} = \underbrace{4d^2}_{\text{self-attn}} + \underbrace{8d^2}_{\text{FFN}} + \underbrace{\text{small}}_{\text{LN}} = 12d^2
$$

$$
\text{Decoder layer} = \underbrace{4d^2}_{\text{masked self-attn}} + \underbrace{4d^2}_{\text{cross-attn}} + \underbrace{8d^2}_{\text{FFN}} + \underbrace{\text{small}}_{\text{LN}} = 16d^2
$$

$$
\text{Total} \approx L \times 12d^2 + L \times 16d^2 + 2Vd = L \times 28d^2 + 2Vd
$$

For the original paper ($d = 512, L = 6, V = 37000$):

$$
\approx 6 \times 28 \times 512^2 + 2 \times 37000 \times 512 \approx 44M + 38M \approx 65M \text{ params}
$$

#### Decoder-Only Transformer (GPT-Style)

$$
\text{Per layer} = \underbrace{4d^2}_{\text{self-attn}} + \underbrace{8d^2}_{\text{FFN}} = 12d^2
$$

$$
\text{Total} \approx 12Ld^2 + Vd \quad (\text{with weight tying, one embedding})
$$

### 12.4 Real-World Examples

| Model | $L$ | $d$ | $h$ | $d_{ff}$ | Vocab | Total Params |
|-------|-----|-----|-----|-----------|-------|--------------|
| GPT-2 Small | 12 | 768 | 12 | 3072 | 50257 | 124M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 50257 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 50257 | 175B |
| Llama 2 7B | 32 | 4096 | 32 | 11008 | 32000 | 6.7B |
| Llama 2 70B | 80 | 8192 | 64 | 28672 | 32000 | 68.7B |
| Llama 3 8B | 32 | 4096 | 32 | 14336 | 128256 | 8.0B |
| Llama 3 70B | 80 | 8192 | 64 | 28672 | 128256 | 70.6B |

### 12.5 Where Do the Parameters Live?

For a typical decoder-only model:

```
FFN weights:        ~65% of total parameters
Attention weights:  ~30% of total parameters
Embeddings:         ~3-5% of total parameters
LayerNorm:          ~0.01% of total parameters
```

The FFN dominates. This is why techniques like MoE (Mixture of Experts) target the FFN — it's where most of the model's knowledge is stored.

---

## 13. The Three Variants — Encoder-Only, Decoder-Only, Encoder-Decoder

The original transformer is encoder-decoder, but two major simplifications emerged.

### 13.1 Encoder-Only (BERT-Style)

**Architecture**: Only the encoder stack. No decoder, no cross-attention, no causal mask.

**Key characteristics**:
- **Bidirectional attention**: Every token attends to every other token (no masking)
- **Pretraining**: Masked Language Modeling (MLM) — predict randomly masked tokens
- **Output**: Rich contextual representations for each position
- **Use cases**: Classification, NER, sentence embeddings, retrieval

```
Input:    The [MASK] sat on the mat
           ↓    ↓    ↓   ↓  ↓   ↓
        ┌─────────────────────────┐
        │     Encoder × N         │
        │  (bidirectional attn)   │
        └─────────────────────────┘
           ↓    ↓    ↓   ↓  ↓   ↓
Output: [h₁] [h₂] [h₃] ... ... [h₆]
              ↓
         predict "cat"
```

**Models**: BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT

### 13.2 Decoder-Only (GPT-Style)

**Architecture**: Only the decoder stack, but without cross-attention (since there's no encoder). Just masked self-attention + FFN.

**Key characteristics**:
- **Causal (unidirectional) attention**: Each token only attends to itself and previous tokens
- **Pretraining**: Next-token prediction (causal language modeling)
- **Output**: Predicted next token at each position
- **Use cases**: Text generation, chat, reasoning, general-purpose AI

```
Input:    The cat sat on the
           ↓    ↓   ↓   ↓  ↓
        ┌─────────────────────┐
        │   Decoder × N        │
        │  (causal attention)  │
        └─────────────────────┘
           ↓    ↓   ↓   ↓  ↓
Output:  cat  sat  on  the mat
        (next token predictions)
```

**Models**: GPT-1/2/3/4, Llama 1/2/3, Mistral, Claude, Gemini

### 13.3 Encoder-Decoder (T5-Style)

**Architecture**: Full encoder + decoder as in the original paper.

**Key characteristics**:
- **Encoder**: Bidirectional attention over the input
- **Decoder**: Causal attention over generated output + cross-attention to encoder
- **Pretraining**: Span corruption (T5), denoising (BART)
- **Use cases**: Translation, summarization, structured output tasks

```
Input:    The <X> sat on the <Y>         Output:  <X> cat <Y> mat
           ↓   ↓   ↓   ↓  ↓   ↓                   ↓   ↓   ↓   ↓
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │     Encoder × N          │───►│     Decoder × N          │
        │  (bidirectional attn)    │    │  (causal + cross-attn)   │
        └──────────────────────────┘    └──────────────────────────┘
```

**Models**: T5, BART, mBART, UL2, Flan-T5

### 13.4 Comparison

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder |
|--------|-------------|--------------|-----------------|
| **Attention** | Bidirectional | Causal | Both |
| **Pretraining** | MLM | CLM | Span corruption / denoising |
| **Strength** | Understanding | Generation | Conditional generation |
| **Inference speed** | Fast (single pass) | Slow (autoregressive) | Medium |
| **Dominant for** | Embeddings, classification | General-purpose AI, chat | Translation, summarization |
| **Scaling trend** | Plateaued at ~1B | Scaled to 1T+ | Scaled to ~13B (T5-XXL) |
| **Current status** | Still used (embeddings) | **Dominant paradigm** | Declining use |

### 13.5 Why Decoder-Only Won

The field has converged on decoder-only as the default. Why?

1. **Simplicity**: One attention pattern (causal), one training objective (next-token prediction). Fewer moving parts = easier to scale.

2. **Emergent abilities**: In-context learning, chain-of-thought reasoning, and instruction following all emerged naturally from causal LM pretraining at scale. Encoder-only and encoder-decoder models showed these less reliably.

3. **Universality**: A decoder-only model can do everything:
   - Classification → generate the label
   - Translation → generate in the target language
   - Summarization → generate the summary
   - Q&A → generate the answer

4. **Scaling laws**: Decoder-only models follow predictable scaling laws (Chinchilla), making them easier to plan and budget.

5. **Inference efficiency**: The KV cache works most naturally with causal attention, and the single-stack architecture is simpler to optimize.

---

## 14. Modern Transformer Design Choices (2023-2026)

Modern LLMs deviate significantly from the original 2017 design. Here's what changed and why.

### 14.1 Architecture Comparison

| Design Choice | Original (2017) | GPT-3 (2020) | Llama 2 (2023) | Llama 3 (2024) |
|---------------|-----------------|--------------|----------------|----------------|
| **Type** | Encoder-Decoder | Decoder-only | Decoder-only | Decoder-only |
| **Norm** | Post-LN | Pre-LN | Pre-RMSNorm | Pre-RMSNorm |
| **Activation** | ReLU | GELU | SwiGLU | SwiGLU |
| **Positional** | Sinusoidal | Learned | RoPE | RoPE |
| **Attention** | MHA | MHA | GQA (70B) | GQA |
| **Bias terms** | Yes | Yes | No | No |
| **Vocab size** | 37K | 50K | 32K | 128K |
| **Context** | 512 | 2048 | 4096 | 8192-128K |
| **Weight tying** | Yes | Yes | No | No |

### 14.2 Grouped-Query Attention (GQA)

Standard Multi-Head Attention (MHA) gives each head its own Q, K, V projections. This creates large KV caches at inference.

**GQA** groups multiple query heads to share the same K, V heads:

```
MHA (8 heads):        GQA (8Q, 2KV groups):    MQA (8Q, 1KV group):
Q₁ K₁ V₁             Q₁ Q₂ Q₃ Q₄  K₁ V₁      Q₁ Q₂ ... Q₈  K₁ V₁
Q₂ K₂ V₂             Q₅ Q₆ Q₇ Q₈  K₂ V₂
Q₃ K₃ V₃
Q₄ K₄ V₄
...                   8 Q heads, 2 KV heads      8 Q heads, 1 KV head
Q₈ K₈ V₈
```

**KV cache savings**:
- MHA: $2 \times L \times h \times d_k \times n$ (per sequence, for all layers)
- GQA with $g$ groups: $2 \times L \times g \times d_k \times n$
- MQA ($g=1$): $2 \times L \times d_k \times n$

For Llama 2 70B with GQA (8 KV heads vs 64 Q heads): **8× reduction** in KV cache size.

### 14.3 No Bias Terms

Most modern LLMs remove bias terms from linear layers:

$$
\mathbf{y} = \mathbf{W}\mathbf{x} \quad \text{(no + b)}
$$

**Why?** Biases add minimal expressivity but complicate certain techniques:
- Tensor parallelism needs to handle bias reduction across GPUs
- LoRA and other PEFT methods are cleaner without biases
- Empirically, removing biases doesn't hurt performance

### 14.4 Larger Vocabularies

Trend: vocabulary sizes are growing.

| Model | Vocab Size |
|-------|-----------|
| GPT-2 | 50,257 |
| Llama 2 | 32,000 |
| Llama 3 | 128,256 |
| GPT-4 | ~100K (estimated) |
| Gemma 2 | 256,000 |

**Why larger?** Larger vocabularies mean:
- Fewer tokens per input → shorter sequences → less computation
- Better multilingual coverage
- Better handling of code, math, special characters

**Trade-off**: Larger embedding tables → more parameters, but the embedding layer is a small fraction of total params for large models.

### 14.5 Parallel Attention + FFN

Some architectures (PaLM, GPT-J) compute attention and FFN **in parallel** instead of sequentially:

**Sequential** (standard):

$$
\mathbf{h} = \mathbf{x} + \text{FFN}(\text{LN}(\mathbf{x} + \text{Attn}(\text{LN}(\mathbf{x}))))
$$

**Parallel**:

$$
\mathbf{h} = \mathbf{x} + \text{Attn}(\text{LN}(\mathbf{x})) + \text{FFN}(\text{LN}(\mathbf{x}))
$$

**Advantage**: Attention and FFN can be computed simultaneously → ~15% faster training.
**Disadvantage**: Slightly worse performance at small scale; negligible difference at large scale.

---

## 15. Common Misconceptions

### Misconception 1: "Transformers understand sequence order through attention"

**Reality**: Attention is permutation-equivariant. Without positional encodings, `[cat, sat, mat]` and `[mat, cat, sat]` would produce identical attention patterns. Positional encodings are the *only* source of order information.

### Misconception 2: "The FFN is just a non-linear transformation"

**Reality**: The FFN stores the majority of the model's factual knowledge. It acts as a key-value memory where each neuron detects a pattern and outputs associated information. This is why scaling the FFN (more neurons) directly increases the model's knowledge capacity.

### Misconception 3: "Bigger model = more layers"

**Reality**: You can make a model bigger by increasing width ($d_{\text{model}}$), depth ($L$), or FFN dimension ($d_{ff}$). Modern evidence suggests **width** scales more efficiently than depth. GPT-3 175B has 96 layers; Llama 3 70B has 80 layers but is more capable due to better width/depth ratio and more training data.

### Misconception 4: "Encoder-decoder is better for generation tasks"

**Reality**: Decoder-only models have proven equally capable at generation tasks (translation, summarization) when scaled sufficiently. The decoder-only architecture is simpler and scales more predictably, which is why it became the dominant paradigm.

### Misconception 5: "Residual connections are just skip connections"

**Reality**: While inspired by ResNet's skip connections, the transformer's residual connections play a more fundamental role. They create a **residual stream** — a shared communication bus that all layers read from and write to. Each layer adds a small delta to this stream. The final representation is the sum of the original input and all layers' contributions.

---

## 16. Interview Questions & Answers

### Q1: Draw the full transformer architecture from memory and explain every component.

**A**: *(Draw the diagram from Section 11.1)* The transformer has three main parts:

**Input pipeline**: Token embeddings (scaled by $\sqrt{d_{\text{model}}}$) plus positional encodings. This converts discrete tokens into continuous vectors with position information.

**Encoder** (N identical blocks): Each block has (1) multi-head self-attention where every token attends to every other token bidirectionally, followed by Add & Norm, then (2) a position-wise FFN that expands to $4d$ then compresses back, followed by Add & Norm.

**Decoder** (N identical blocks): Each block has (1) **masked** multi-head self-attention (causal mask prevents attending to future), (2) multi-head **cross-attention** where queries come from the decoder but keys/values come from the encoder's output, and (3) a position-wise FFN. Each sub-layer has Add & Norm.

**Output head**: Linear projection to vocabulary size followed by softmax.

Key design choices: residual connections around every sub-layer enable gradient flow through deep networks; LayerNorm stabilizes training; multi-head attention allows attending to different aspects simultaneously.

---

### Q2: How many parameters does a transformer with L layers, d_model hidden size, and V vocab size have? Break it down by component.

**A**: For a decoder-only transformer (the dominant paradigm):

**Per layer**:
- Self-attention: $4d^2$ (four matrices $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O$, each $d \times d$)
- FFN (standard): $8d^2$ (two matrices: $d \times 4d$ and $4d \times d$)
- LayerNorm: $4d$ (negligible — two norms × $2d$ each)
- **Total per layer**: $\approx 12d^2$

**Global**:
- Token embeddings: $Vd$
- Output head: $Vd$ (0 if weight-tied)
- Final LayerNorm: $2d$

**Total**: $\approx 12Ld^2 + 2Vd$ (with weight tying) or $\approx 12Ld^2 + Vd$ (untied, shared embedding).

**Example**: Llama 2 7B — $L=32, d=4096, V=32000$
- Layer params: $12 \times 32 \times 4096^2 \approx 6.4B$ (SwiGLU is $\approx 8d^2$ in FFN × 3 matrices, but with reduced $d_{ff}$ it stays roughly $8d^2$ total — same ballpark)
- Embeddings: $2 \times 32000 \times 4096 \approx 0.26B$
- Total: $\approx 6.7B$ ✓

---

### Q3: Why Pre-LN instead of Post-LN? What training stability issue does it solve?

**A**: In **Post-LN**, the LayerNorm sits after the residual addition:

$$
\text{out} = \text{LN}(x + \text{SubLayer}(x))
$$

This creates gradient amplification at early layers. The gradient norm at layer $l$ scales as $O(\sqrt{L - l})$ — layers closer to the input receive much larger gradient updates. This causes:
1. Training instability, especially at high learning rates
2. Requirement for extensive warmup (thousands of steps of tiny learning rates)
3. Difficulty training very deep models (96+ layers)

In **Pre-LN**, LayerNorm is applied before the sub-layer:

$$
\text{out} = x + \text{SubLayer}(\text{LN}(x))
$$

The gradient flows through the residual connection **without passing through normalization**, maintaining a clean gradient highway. This gives:
1. Well-conditioned gradients across all depths
2. No warmup needed (or minimal)
3. Reliable training of very deep networks

The trade-off: Pre-LN may give very slightly lower final performance than Post-LN *when Post-LN manages to converge*. In practice, Pre-LN's reliability far outweighs this marginal difference, which is why every major LLM (GPT-2/3/4, Llama, Mistral, Claude) uses Pre-LN.

---

### Q4: Why is the FFN dimension 4x the model dimension? What role does the FFN play?

**A**: The FFN plays a fundamentally different role than attention:

**Attention** = inter-position communication. It determines *which* positions are relevant to each other and aggregates information across positions.

**FFN** = per-position processing. It transforms the representation at each position independently. Research (Geva et al., 2021) shows the FFN functions as a **key-value memory**:

$$
\text{FFN}(x) = \sum_i \text{ReLU}(w_{1,i}^T x) \cdot w_{2,i}
$$

Each of the $4d$ neurons acts as a pattern detector (key) paired with an associated output (value). More neurons = more knowledge.

**Why 4x?** It's a capacity choice. The model needs enough intermediate neurons to:
1. Store factual knowledge (e.g., "Paris is the capital of France")
2. Perform feature transformations (compose attention outputs)
3. Implement non-linear computations that attention alone can't

The 4x ratio was empirically determined in the original paper. Some modern models use different ratios (Llama uses $\approx 2.67x$ with SwiGLU to keep parameter count similar after adding the gate matrix). The key insight is that the FFN must be significantly wider than $d_{\text{model}}$ to have sufficient capacity.

**The FFN accounts for ~65% of all parameters** — this is where most of the model's "knowledge" lives.

---

### Q5: Compare encoder-only, decoder-only, and encoder-decoder. When would you use each?

**A**:

| Architecture | Attention | Pretraining | Best For |
|-------------|-----------|-------------|----------|
| **Encoder-only** (BERT) | Bidirectional | MLM (predict masked tokens) | Understanding: classification, NER, embeddings, retrieval |
| **Decoder-only** (GPT) | Causal | CLM (predict next token) | Generation: chat, reasoning, general-purpose AI |
| **Encoder-decoder** (T5) | Bidirectional encoder + causal decoder with cross-attention | Span corruption, denoising | Conditional generation: translation, summarization |

**When to use each**:

- **Encoder-only**: When you need to *understand* text — text classification (sentiment, spam), named entity recognition, computing semantic embeddings for search/retrieval. BERT-family models are still the best choice for high-quality, efficient embeddings.

- **Decoder-only**: The default choice for almost everything in 2024-2026. If you need text generation, reasoning, chat, or a general-purpose model, use decoder-only. The scaling laws are well understood, emergent abilities (ICL, CoT) appear reliably, and the architecture is simplest to optimize.

- **Encoder-decoder**: When you have a clear input→output structure and the encoder's bidirectional processing of the input genuinely helps. Translation and summarization are the classic cases. However, decoder-only models have largely caught up here too.

**Why decoder-only dominates**: Simplicity (one attention type, one training objective), predictable scaling, emergent abilities at scale, and universality (any NLP task can be framed as text generation).

---

### Q6: What happens if you remove residual connections from a 96-layer transformer?

**A**: The model becomes **untrainable**. Three things go catastrophically wrong:

1. **Vanishing gradients**: Without the identity shortcut, gradients must propagate through 96 consecutive matrix multiplications. Even with careful initialization, the gradient norm decays exponentially — lower layers receive essentially zero gradient signal and don't learn.

2. **Signal degradation**: The forward pass also suffers. The input representation is transformed 96 times with no skip connections — by the final layer, the original token identity is completely lost. The model can't form useful representations.

3. **Rank collapse**: Without residual connections, deep networks suffer from **representation rank collapse** — the representations across different inputs converge to a low-dimensional subspace. The model loses the ability to distinguish between different inputs.

With residual connections, the gradient at any layer includes the identity term: $\frac{\partial h^{(l)}}{\partial h^{(l-1)}} = I + \frac{\partial f^{(l)}}{\partial h^{(l-1)}}$. This guarantees a direct gradient path from the loss to every layer, regardless of depth. The network effectively behaves as an **ensemble of shallower networks** — there are $2^{96}$ paths through the network, and the gradient can flow through any of them.

---

### Q7: How does the KV cache work and why is it essential for inference?

**A**: During autoregressive generation, the model generates one token at a time. Without caching, generating token $t$ requires recomputing attention over all $t$ positions, making the total generation cost $O(n^2)$ per token.

**The KV cache** stores the key and value vectors from all previous positions:

At step $t$:
1. Compute Q, K, V only for the **new** token (position $t$)
2. Append the new K, V to the cached K, V from steps $1$ to $t-1$
3. Compute attention: new Q attends to all cached K, V

This reduces per-token computation from $O(n \cdot d)$ attention to $O(d)$ for the new token's Q/K/V computation, plus $O(n \cdot d)$ for the attention with cached values. The total generation cost drops from $O(n^2 \cdot d)$ to $O(n \cdot d)$.

**Cache size**: For each layer, we store K and V tensors of shape $[n, d_k]$ per head. Total cache: $2 \times L \times h \times n \times d_k$ elements. For Llama 2 70B with a 4096-length sequence in FP16, this is about 20GB — which is why GQA (reducing KV heads) is so important.

---

### Q8: Explain weight tying. Why do some models use it and others don't?

**A**: Weight tying means using the same weight matrix for the input embedding and the output projection:

$$
\mathbf{W}_{\text{output}} = \mathbf{W}_{\text{embed}} \in \mathbb{R}^{V \times d}
$$

**Why it works**: The embedding maps tokens → vectors, and the output head maps vectors → logits over tokens. These are conceptually inverse operations. If token "cat" maps to vector $\mathbf{v}$, the output should assign high logit to "cat" when the hidden state is close to $\mathbf{v}$. Using the same weights enforces this consistency.

**Advantages**:
- Saves $V \times d$ parameters (e.g., 131M for $V=32K, d=4096$)
- Regularization effect — constrains the output space
- Works well empirically, especially for smaller models

**Why some models don't use it** (Llama 2/3, Mistral):
- At large scale, the parameter savings are a tiny fraction of total params
- Untied weights give the model more freedom — the optimal embedding space and output space may differ
- When vocabulary is very large (128K+), the embedding layer is a significant compute bottleneck; having a separate output matrix allows optimizing each independently

---

### Q9: What is the "residual stream" view of transformers?

**A**: The **residual stream** interpretation (popularized by Elhage et al., Anthropic) provides a powerful mental model:

Think of the transformer as having a single "highway" of information — the **residual stream** — that flows from input to output. Each layer (attention and FFN) **reads from** this stream and **writes a small update** back to it.

$$
\mathbf{h}^{(L)} = \mathbf{h}^{(0)} + \sum_{l=1}^{L} \left(\text{Attn}^{(l)}(\cdot) + \text{FFN}^{(l)}(\cdot)\right)
$$

The final representation is the **sum** of the original input embedding plus the outputs of all $2L$ sub-layers. This view reveals:

1. **Superposition**: The residual stream must carry information for many different features simultaneously in a limited-dimensional space. Features are stored in "superposition" — overlapping, non-orthogonal directions.

2. **Direct paths**: A token's original embedding persists through the residual stream. Even at layer 96, the model can "read" the original token identity.

3. **Layer independence**: Each layer's contribution is additive. In principle, you could reorder or even remove layers, and the model might still partially function (this has been verified experimentally).

4. **Information bottleneck**: The residual stream has fixed width $d_{\text{model}}$. Everything the model knows must fit through this bottleneck at every layer — this constrains what the model can represent.

This view is fundamental to mechanistic interpretability research and helps reason about how transformers process information.

---

### Q10: If you were designing a new transformer from scratch today, what would you choose and why?

**A**: Based on current best practices (2024-2026):

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Architecture** | Decoder-only | Simplicity, proven scaling, emergent abilities |
| **Normalization** | Pre-RMSNorm | Stable training, faster than LayerNorm, no warmup needed |
| **Activation** | SwiGLU | ~1-2% improvement over GELU/ReLU |
| **Positional encoding** | RoPE | Relative position, extrapolation-friendly, proven at scale |
| **Attention** | GQA (8 KV groups) | Balanced quality and inference efficiency |
| **Bias terms** | None | Simpler, no quality loss |
| **Weight tying** | No (at scale) | More expressivity, marginal parameter cost |
| **Vocabulary** | 128K+ (SentencePiece/BPE) | Shorter sequences, multilingual |
| **FFN ratio** | $\frac{8}{3}d$ with SwiGLU | Same param count as 4d ReLU FFN |
| **Context length** | 8K+ with RoPE scaling | Modern applications need long context |
| **Initialization** | Small init for attention, standard for FFN | Training stability |

This is essentially the Llama 3 recipe, which represents the current consensus for efficient, high-quality transformer design.

---

*This topic covers the blueprint. Next: [Topic 10: Positional Encodings](10_Positional_Encodings.md) — a deep dive into how transformers understand position, from sinusoidal to RoPE to ALiBi.*
