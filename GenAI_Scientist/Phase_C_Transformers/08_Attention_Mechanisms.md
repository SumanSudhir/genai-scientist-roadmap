# Topic 8: Attention Mechanisms

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 8 of 28
> **Scope**: Seq2Seq attention (Bahdanau, Luong), self-attention, scaled dot-product attention, multi-head attention, cross-attention, causal masking, modern variants (GQA, MQA, Flash Attention, sliding window)
> **Why this matters**: Attention IS the transformer. Every forward pass through GPT, Llama, or BERT is a sequence of attention operations. You need to know it cold -- the math, the intuition, the variants, and the computational trade-offs.
> **Previous**: [Topic 7: Sequence Modeling](07_Sequence_Modeling.md)
> **Next**: [Topic 9: Transformer Architecture](09_Transformer_Architecture.md)

---

## Table of Contents

1. [The Seq2Seq Bottleneck — Why Attention Was Invented](#1-the-seq2seq-bottleneck--why-attention-was-invented)
2. [Bahdanau Attention (Additive)](#2-bahdanau-attention-additive)
3. [Luong Attention (Multiplicative)](#3-luong-attention-multiplicative)
4. [Self-Attention — The Key Insight](#4-self-attention--the-key-insight)
5. [Scaled Dot-Product Attention](#5-scaled-dot-product-attention)
6. [Multi-Head Attention](#6-multi-head-attention)
7. [Cross-Attention](#7-cross-attention)
8. [Causal (Masked) Attention](#8-causal-masked-attention)
9. [Computational Complexity of Attention](#9-computational-complexity-of-attention)
10. [Modern Attention Variants](#10-modern-attention-variants)
11. [What Attention Heads Actually Learn](#11-what-attention-heads-actually-learn)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. The Seq2Seq Bottleneck — Why Attention Was Invented

To understand attention, you must first understand the problem it solves.

### 1.1 The Bottleneck Problem

In a basic Seq2Seq model (encoder-decoder RNN), the encoder processes the entire input sequence and compresses it into a single fixed-size vector — the **context vector** — which is the final hidden state of the encoder.

```
Encoder:   x_1 -> x_2 -> x_3 -> ... -> x_T -> [context vector c]
                                                       |
Decoder:                                         c -> y_1 -> y_2 -> y_3
```

The decoder must generate the entire output from this one vector.

**Why this fails**:

1. **Information bottleneck**: A 50-word sentence and a 500-word paragraph are both squeezed into the same 512-dimensional vector. Long inputs inevitably lose information.

2. **Vanishing signal**: The encoder processes sequentially. By the time it reaches the end, early tokens have been processed through many transformations. Their signal is diluted, even with LSTMs.

3. **No alignment**: The decoder has no idea which part of the input is relevant to the current output position. When translating "Le chat est sur le tapis" to "The cat is on the mat," the decoder generating "cat" should focus on "chat" — but it only sees a blended summary of everything.

**Empirical evidence**: Seq2Seq performance degraded sharply as input length increased beyond ~20-30 tokens. Something better was needed.

### 1.2 The Insight

Instead of compressing the entire input into one vector, let the decoder **look at all encoder hidden states** at every decoding step and **dynamically choose** which ones are most relevant.

This is attention: a mechanism for the decoder to selectively focus on different parts of the input at each step.

---

## 2. Bahdanau Attention (Additive)

**Paper**: "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)

This was the original attention mechanism. It revolutionized machine translation and planted the seed for transformers.

### 2.1 The Mechanism

At each decoder time step t, Bahdanau attention computes a **different** context vector by taking a weighted combination of all encoder hidden states.

**The three steps** (repeated at every decoder step):

**Step 1 — Score**: Compute a relevance score between the decoder's current state and each encoder hidden state:

$$e_{t,i} = \mathbf{v}^T \tanh(W_1 \mathbf{s}_{t-1} + W_2 \mathbf{h}_i)$$

Where:
- $\mathbf{s}_{t-1}$ is the decoder's hidden state from the previous step (what the decoder has generated so far)
- $\mathbf{h}_i$ is the i-th encoder hidden state (representation of input position i)
- $W_1$, $W_2$ are learned weight matrices, $\mathbf{v}$ is a learned vector
- The tanh + linear combination forms a small neural network that learns to judge relevance

**Step 2 — Normalize**: Convert scores to a probability distribution using softmax:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

The $\alpha$ values are the **attention weights**. They sum to 1 and indicate how much the decoder should "attend to" each encoder position at this step.

**Step 3 — Weighted sum**: Compute the context vector as the weighted combination of encoder hidden states:

$$\mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{h}_i$$

This context vector $\mathbf{c}_t$ is then fed into the decoder (typically concatenated with the decoder's input) to produce the output at step t.

### 2.2 What Changed

```
WITHOUT attention:
  Encoder: x_1, x_2, ..., x_T -> single vector c
  Decoder: c -> y_1 -> y_2 -> y_3

WITH attention:
  Encoder: x_1, x_2, ..., x_T -> h_1, h_2, ..., h_T  (keep ALL hidden states)
  Decoder step t: look at ALL h_i, compute relevance, create custom c_t
                  c_t + s_{t-1} -> y_t
```

The decoder now has access to a "soft pointer" into the input at every step. When generating "cat," it assigns high attention weight to "chat." When generating "mat," it attends to "tapis."

### 2.3 Why "Additive"

The scoring function uses addition inside a neural network:

$$e_{t,i} = \mathbf{v}^T \tanh(W_1 \mathbf{s}_{t-1} + W_2 \mathbf{h}_i)$$

The decoder state and encoder state are mapped to a common space and **added** together, then passed through tanh and a linear layer. This is a small feedforward network with one hidden layer.

**Computational cost**: Requires a full forward pass through this small network for every (decoder position, encoder position) pair. For output length $T_{\text{out}}$ and input length $T_{\text{in}}$, that's $T_{\text{out}} \times T_{\text{in}}$ evaluations.

### 2.4 Visualizing Alignment

One of the most compelling results from Bahdanau et al. was plotting attention weights $\alpha_{t,i}$ as a heatmap. Each **row** = one decoder step (output word being generated), each **column** = one encoder position (input word).

```
Translating: "Le chat est sur le tapis" → "The cat is on the mat"

              Le    chat   est   sur    le   tapis
         The [0.85  0.05  0.04  0.03   0.02  0.01]   ← "The" mostly attends to "Le"
         cat [0.04  0.88  0.03  0.02   0.02  0.01]   ← "cat" strongly attends to "chat"
          is [0.02  0.03  0.87  0.04   0.02  0.02]   ← "is"  strongly attends to "est"
          on [0.01  0.02  0.03  0.88   0.04  0.02]   ← "on"  strongly attends to "sur"
         the [0.02  0.02  0.02  0.04   0.85  0.05]   ← "the" strongly attends to "le"
         mat [0.01  0.01  0.02  0.03   0.05  0.88]   ← "mat" strongly attends to "tapis"
```

The near-diagonal pattern emerges **without ever being told** about word alignment — the model learned it purely from translation data. For language pairs with different word order (German-English), the learned pattern is non-monotonic but still shows correct cross-lingual alignments.

This was stunning in 2015: a neural network learning interpretable, linguistically meaningful alignment as a byproduct of training to minimize translation loss.

---

## 3. Luong Attention (Multiplicative)

**Paper**: "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)

### 3.1 Simplified Scoring

Luong proposed replacing the neural network with a simpler dot-product-based score:

**Dot product** (simplest):

$$e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$$

**General** (learned):

$$e_{t,i} = \mathbf{s}_t^T W \mathbf{h}_i$$

**Concat** (similar to Bahdanau):

$$e_{t,i} = \mathbf{v}^T \tanh(W [\mathbf{s}_t; \mathbf{h}_i])$$

### 3.2 Bahdanau vs Luong

| Aspect | Bahdanau (Additive) | Luong (Multiplicative) |
|--------|--------------------|-----------------------|
| Score function | $\mathbf{v}^T \tanh(W_1 \mathbf{s} + W_2 \mathbf{h})$ | $\mathbf{s}^T W \mathbf{h}$ (or $\mathbf{s}^T \mathbf{h}$) |
| Decoder state used | $\mathbf{s}_{t-1}$ (previous) | $\mathbf{s}_t$ (current) |
| Computation | Slower (neural network) | Faster (matrix multiply) |
| When context is used | Before computing $\mathbf{s}_t$ | After computing $\mathbf{s}_t$ |
| Accuracy | Slightly better on small data | Slightly better on large data |

### 3.3 Why This Matters for Transformers

The dot-product score from Luong attention is the direct ancestor of the transformer's attention mechanism. The key evolution:

Luong (2015): $e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$ (decoder state dot encoder state)

Transformer (2017): $e_{t,i} = (\mathbf{x}_t W_Q)^T (\mathbf{x}_i W_K) = \mathbf{q}_t^T \mathbf{k}_i$ (query dot key, with learned projections)

The transformer generalizes Luong attention by:
1. Adding **learned projections** ($W_Q$, $W_K$, $W_V$) instead of using raw hidden states
2. **Scaling** by $1/\sqrt{d_k}$ for numerical stability
3. Applying it as **self-attention** (not just cross-attention)
4. Using **multiple heads** in parallel

---

## 4. Self-Attention — The Key Insight

### 4.1 From Cross-Attention to Self-Attention

In Seq2Seq attention, the decoder attends to the encoder — two different sequences interact. This is **cross-attention**.

**Self-attention** asks a different question: what if a sequence attends **to itself**? Each token looks at every other token in the same sequence to build a contextualized representation.

```
Cross-attention:  decoder token t looks at all encoder tokens
Self-attention:   token t looks at all tokens in the same sequence (including itself)
```

### 4.2 Why Self-Attention Is Powerful

Consider the sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to? "The animal." A human resolves this instantly by looking at the broader context.

With self-attention, the representation of "it" is computed by attending to every other word. The model can learn to assign high attention weight from "it" to "animal," effectively resolving the coreference. No recurrence needed, no sequential processing — just pairwise comparisons.

### 4.3 The Three Advantages Over RNNs

**1. Long-range dependencies**: In an RNN, information from position 1 must pass through every intermediate position to reach position 100. At each step, information is transformed and potentially lost. With self-attention, position 100 can directly attend to position 1 — the path length is $O(1)$.

**2. Parallelization**: RNNs must process sequentially ($\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$). Self-attention computes all pairwise relationships simultaneously. All positions interact in parallel. This makes transformers dramatically faster to train on GPUs.

**3. No information bottleneck**: Every token's representation is a weighted combination of ALL other tokens' representations, with weights determined by relevance. No fixed-size bottleneck.

```
Information path length:
  RNN:            O(n)     — token 1 to token n passes through n-1 intermediate states
  Self-Attention: O(1)     — token 1 directly attends to token n
  CNN:            O(log_k(n)) — depends on kernel size and number of layers
```

### 4.4 Visualizing Self-Attention

Consider: **"The animal didn't cross the street because it was too tired."**

A simplified view of what a **coreference-resolving** attention head might learn for the token "it":

```
              The  animal  didn't  cross  street  because   it   was  tired
     The    [ 0.5    0.1     0.0    0.1     0.1     0.0    0.1   0.0   0.1 ]
  animal    [ 0.1    0.5     0.0    0.1     0.0     0.0    0.2   0.0   0.1 ]
  didn't    [ 0.1    0.1     0.4    0.2     0.1     0.0    0.0   0.0   0.1 ]
   cross    [ 0.1    0.1     0.1    0.4     0.2     0.1    0.0   0.0   0.0 ]
  street    [ 0.1    0.0     0.1    0.2     0.4     0.1    0.0   0.0   0.1 ]
 because    [ 0.1    0.1     0.1    0.1     0.1     0.4    0.0   0.0   0.1 ]
      it    [ 0.0    0.7     0.0    0.0     0.1     0.0    0.2   0.0   0.0 ]  ← "it" → "animal"
     was    [ 0.0    0.1     0.1    0.0     0.0     0.1    0.3   0.4   0.0 ]
   tired    [ 0.0    0.1     0.0    0.0     0.0     0.1    0.3   0.1   0.4 ]
```

Look at the row for **"it"**: attention weight **0.70 on "animal"**. This is how the model resolves the pronoun — the output representation of "it" after this layer is a blend dominated by "animal"'s value vector. When the FFN layer processes "it" in the next step, it is effectively processing something that **knows it refers to the animal**.

No explicit coreference annotation was given — this pattern emerges purely from the model learning to predict text. This is self-attention's most profound capability: building context-aware representations where a word's meaning is shaped by every word it attends to.

---

## 5. Scaled Dot-Product Attention

This is the core attention mechanism used in all transformers. You need to be able to derive it, explain every choice, and identify what breaks if you change anything.

### 5.1 The Q, K, V Framework

Given an input sequence $X$ of shape $(T, d_{\text{model}})$, we compute three representations:

$$Q = X W_Q \quad \text{(Queries, shape: } T \times d_k\text{)}$$
$$K = X W_K \quad \text{(Keys, shape: } T \times d_k\text{)}$$
$$V = X W_V \quad \text{(Values, shape: } T \times d_v\text{)}$$

Where $W_Q$, $W_K$, $W_V$ are learned projection matrices.

**Intuition for Q, K, V**:

Think of a library search:
- **Query (Q)**: "What am I looking for?" — the current token's question about what context it needs
- **Key (K)**: "What do I contain?" — each token's advertisement of its content
- **Value (V)**: "What information do I provide?" — each token's actual content to contribute

When you search a library:
1. You compare your query against the key (title/description) of each book
2. You select the books whose keys best match your query
3. You read the values (content) of those books

### 5.2 The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's trace through this step by step.

**Step 1 — Compute similarity scores**:

$$S = QK^T \quad \text{(shape: } T \times T\text{)}$$

$$S_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \text{how relevant position } j \text{ is to position } i$$

This is a matrix of dot products. $S_{ij}$ is large when query $i$ and key $j$ point in similar directions in the $d_k$-dimensional space.

**Step 2 — Scale by $\sqrt{d_k}$**:

$$S_{\text{scaled}} = \frac{S}{\sqrt{d_k}}$$

**Why scale?** This is critical. Without scaling, the dot products grow with $d_k$.

If $\mathbf{q}$ and $\mathbf{k}$ are random vectors with entries drawn independently from $\mathcal{N}(0, 1)$, then:

$$\mathbf{q}^T \mathbf{k} = \sum_{j=1}^{d_k} q_j k_j$$

Each term $q_j k_j$ has mean 0 and variance 1 (product of two independent standard normals has variance 1). By the CLT, the sum has:

$$\mathbb{E}[\mathbf{q}^T \mathbf{k}] = 0$$
$$\text{Var}[\mathbf{q}^T \mathbf{k}] = d_k$$
$$\text{Std}[\mathbf{q}^T \mathbf{k}] = \sqrt{d_k}$$

So for $d_k = 64$, typical dot products are in the range $[-8, 8]$. For $d_k = 512$, they're in $[-22, 22]$.

**What happens if you don't scale**: Large dot products push softmax into regions where it saturates — the output becomes nearly one-hot (one value ≈ 1, rest ≈ 0). In these saturated regions:
- Gradients through softmax are nearly zero (vanishing gradient)
- The model can't learn nuanced attention patterns (it's forced to attend to just one position)
- Training becomes unstable

Dividing by $\sqrt{d_k}$ normalizes the variance to 1, keeping dot products in a reasonable range regardless of $d_k$.

**Step 3 — Apply softmax**:

$$A = \text{softmax}(S_{\text{scaled}}) \quad \text{(shape: } T \times T\text{, rows sum to 1)}$$

$$A_{ij} = \frac{\exp(S_{\text{scaled},ij})}{\sum_k \exp(S_{\text{scaled},ik})}$$

Each row of $A$ is a probability distribution over all positions. $A_{ij}$ represents how much position $i$ should attend to position $j$.

**Step 4 — Weighted combination of values**:

$$\text{Output} = AV \quad \text{(shape: } T \times d_v\text{)}$$

$$\text{output}_i = \sum_j A_{ij} \mathbf{v}_j$$

Each output position is a weighted average of all value vectors, with weights determined by the attention scores. This is the contextualized representation.

### 5.3 The Complete Picture

```
Input X (T x d_model)
    |
    ├── X @ W_Q -> Q (T x d_k)
    ├── X @ W_K -> K (T x d_k)
    └── X @ W_V -> V (T x d_v)
            |
    Q @ K^T -> Scores (T x T)        [pairwise similarities]
            |
    / sqrt(d_k) -> Scaled Scores      [numerical stability]
            |
    + Mask (optional) -> Masked Scores [causal masking]
            |
    softmax -> Attention Weights (T x T)  [valid probability distribution]
            |
    @ V -> Output (T x d_v)           [weighted combination]
```

### 5.4 Tensor Shapes (with Batch Dimension)

In practice, we process a batch of B sequences, each of length T:

```
X:      (B, T, d_model)
W_Q:    (d_model, d_k)
W_K:    (d_model, d_k)
W_V:    (d_model, d_v)

Q:      (B, T, d_k)
K:      (B, T, d_k)
V:      (B, T, d_v)

Q @ K^T: (B, T, T)     — the T x T attention matrix for each sequence in the batch
softmax: (B, T, T)
@ V:     (B, T, d_v)    — the output
```

The $(B, T, T)$ attention matrix is the memory bottleneck. For $T = 4096$ and $B = 32$ in FP16: $32 \times 4096 \times 4096 \times 2$ bytes = 1 GB just for one attention layer's scores. With 32 layers, that's 32 GB. This is why long sequences are so expensive.

### 5.5 Concrete Numerical Walkthrough

To make the formula tangible, let's trace through attention on **"I love Paris"** with $d_k = 2$ (toy numbers — real models use $d_k = 64$ or 128).

**Given Q, K, V after projection** (made up for illustration):

$$Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Rows = tokens: "I" (row 0), "love" (row 1), "Paris" (row 2). $V$ is set as an identity matrix so the output is easy to read: output[i][j] = how much token $i$ borrows from token $j$.

---

**Step 1 — Raw scores** $S = QK^T$:

$$S_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$$

$$S = \begin{bmatrix}
1{\cdot}1 + 0{\cdot}0 & 1{\cdot}1 + 0{\cdot}1 & 1{\cdot}0 + 0{\cdot}1 \\
1{\cdot}1 + 1{\cdot}0 & 1{\cdot}1 + 1{\cdot}1 & 1{\cdot}0 + 1{\cdot}1 \\
0{\cdot}1 + 1{\cdot}0 & 0{\cdot}1 + 1{\cdot}1 & 0{\cdot}0 + 1{\cdot}1
\end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 1 \end{bmatrix}$$

**Reading $S$**: Entry $S_{10} = 1$, $S_{11} = 2$, $S_{12} = 1$ — "love"'s query $[1,1]$ most strongly matches its own key $[1,1]$ (dot product = 2). "I"'s query $[1,0]$ scores 0 against "Paris" whose key is $[0,1]$ — orthogonal vectors, no similarity.

---

**Step 2 — Scale** by $\sqrt{d_k} = \sqrt{2} \approx 1.41$:

$$S_{\text{scaled}} = \begin{bmatrix} 0.71 & 0.71 & 0.00 \\ 0.71 & 1.41 & 0.71 \\ 0.00 & 0.71 & 0.71 \end{bmatrix}$$

Without this step, the scores stay as integers 0, 1, 2. For $d_k = 64$, dot products would be in the range $[-8, +8]$, pushing softmax toward near-one-hot outputs and killing gradients.

---

**Step 3 — Softmax** (applied row-wise):

$$\text{Row 0 (I):} \quad \frac{e^{0.71},\ e^{0.71},\ e^{0}}{e^{0.71}+e^{0.71}+e^{0}} = \frac{2.03,\ 2.03,\ 1.00}{5.06} = [0.40,\ 0.40,\ 0.20]$$

$$\text{Row 1 (love):} \quad \frac{e^{0.71},\ e^{1.41},\ e^{0.71}}{e^{0.71}+e^{1.41}+e^{0.71}} = \frac{2.03,\ 4.10,\ 2.03}{8.16} = [0.25,\ 0.50,\ 0.25]$$

$$\text{Row 2 (Paris):} \quad \frac{e^{0},\ e^{0.71},\ e^{0.71}}{e^{0}+e^{0.71}+e^{0.71}} = \frac{1.00,\ 2.03,\ 2.03}{5.06} = [0.20,\ 0.40,\ 0.40]$$

$$A = \begin{bmatrix} 0.40 & 0.40 & 0.20 \\ 0.25 & 0.50 & 0.25 \\ 0.20 & 0.40 & 0.40 \end{bmatrix}$$

**Reading $A$** (each row sums to 1):
- **"I"** splits attention roughly: 40% on itself, 40% on "love", 20% on "Paris"
- **"love"** focuses mostly on itself (50%), splits evenly to neighbors (25% each)
- **"Paris"** ignores "I" mostly (20%), splits equally between "love" and itself (40% each)

---

**Step 4 — Output** $= AV$:

Since $V$ is identity, the output is just the attention matrix $A$ itself:

$$\text{Output} = AV = \begin{bmatrix} 0.40 & 0.40 & 0.20 \\ 0.25 & 0.50 & 0.25 \\ 0.20 & 0.40 & 0.40 \end{bmatrix}$$

Each row is the **new contextualized representation** of that token:

```
"I"     → [0.40, 0.40, 0.20]  — blends itself, "love", "Paris"
"love"  → [0.25, 0.50, 0.25]  — mostly itself, slight awareness of context
"Paris" → [0.20, 0.40, 0.40]  — leans toward "love" (the verb governing it)
```

**The key insight**: "Paris" is no longer just its original embedding. Its representation now carries 40% information from "love" — making it aware of its grammatical role as the object of "love". This is **contextualization**: the same word gets different representations depending on the context it appears in.

Compare: "Paris" in "I love Paris" vs "Paris is cold" — the attention would pull in different context, giving different output representations, which is exactly what we want for downstream tasks.

---

## 6. Multi-Head Attention

### 6.1 Motivation

A single attention head computes one set of attention weights — one "view" of which tokens are relevant to which. But relevance is multidimensional:

- One head might capture **syntactic** relationships (subject-verb agreement)
- Another might capture **semantic** similarity (synonyms, coreferences)
- Another might capture **positional** patterns (attending to the previous token, or to punctuation)

A single attention can only compute one weighted average. Multi-head attention runs multiple attention operations in parallel, each learning a different relationship.

### 6.2 The Mechanism

Instead of one attention with $d_{\text{model}}$ dimensions, split into $h$ heads, each with $d_k = d_{\text{model}} / h$ dimensions:

For head $i$ ($i = 1, \ldots, h$):

$$Q_i = X W_Q^i \quad (T \times d_k, \text{ where } d_k = d_{\text{model}} / h)$$
$$K_i = X W_K^i \quad (T \times d_k)$$
$$V_i = X W_V^i \quad (T \times d_v, \text{ where } d_v = d_{\text{model}} / h)$$
$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad (T \times d_v)$$

Concatenate all heads:

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) \quad (T \times d_{\text{model}})$$

Final projection:

$$\text{Output} = \text{MultiHead} \cdot W_O \quad (T \times d_{\text{model}})$$

Where $W_O$ is a $(d_{\text{model}} \times d_{\text{model}})$ output projection matrix.

### 6.3 Why Not Just One Big Attention?

You might think: a single attention head with $d_{\text{model}}$ dimensions can represent any attention pattern. Why split?

**The key insight**: Different heads operate in **different subspaces**. Each head's $W_Q$, $W_K$, $W_V$ project the input into a different $d_k$-dimensional subspace, and attention is computed independently in each subspace.

This allows the model to simultaneously attend to information from different representation subspaces at different positions. With a single head, these different aspects would be blended into one average, losing information.

**Analogy**: Imagine describing a person. One "head" describes their appearance, another their personality, another their role. Each perspective captures different information. Averaging all descriptions into one number loses specifics; keeping them separate preserves the full picture.

### 6.4 Parameter Count

Per head: $W_Q^i$ $(d_{\text{model}} \times d_k)$ + $W_K^i$ $(d_{\text{model}} \times d_k)$ + $W_V^i$ $(d_{\text{model}} \times d_v)$ = $3 \cdot d_{\text{model}} \cdot d_k$ = $3 \cdot d_{\text{model}} \cdot (d_{\text{model}} / h)$

All $h$ heads: $h \cdot 3 \cdot d_{\text{model}} \cdot (d_{\text{model}} / h) = 3 \cdot d_{\text{model}}^2$

Output $W_O$: $d_{\text{model}} \cdot d_{\text{model}} = d_{\text{model}}^2$

Total: $4 \cdot d_{\text{model}}^2$ (same as a single head with Q, K, V, O projections!)

Multi-head attention has the **same parameter count** as single-head attention with the same $d_{\text{model}}$. The splitting is free — you get multiple perspectives without extra parameters.

### 6.5 Typical Configurations

| Model | d_model | h (heads) | d_k = d_model/h |
|-------|---------|-----------|-----------------|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 (175B) | 12288 | 96 | 128 |
| Llama 2 7B | 4096 | 32 | 128 |
| Llama 2 70B | 8192 | 64 | 128 |

Note: $d_k = 64$ or 128 is nearly universal. When models grow larger, they add more heads (not bigger heads).

### 6.6 Two Heads, Two Perspectives

The same sentence, processed by two different heads simultaneously. Consider: **"The cat sat on the mat"**

**Head 1 — Syntactic (subject tracking)**:
```
         The   cat   sat    on   the   mat
   The [ 0.5   0.4   0.0   0.0   0.1   0.0 ]
   cat [ 0.2   0.6   0.1   0.0   0.1   0.0 ]
   sat [ 0.1   0.5   0.2   0.1   0.1   0.0 ]  ← "sat" attends to "cat" (its subject)
    on [ 0.1   0.1   0.1   0.4   0.2   0.1 ]
   the [ 0.2   0.1   0.0   0.1   0.5   0.1 ]
   mat [ 0.1   0.1   0.0   0.2   0.4   0.2 ]
```

**Head 2 — Positional (previous token)**:
```
         The   cat   sat    on   the   mat
   The [ 1.0   0.0   0.0   0.0   0.0   0.0 ]
   cat [ 0.9   0.1   0.0   0.0   0.0   0.0 ]  ← "cat" attends mostly to "The"
   sat [ 0.0   0.9   0.1   0.0   0.0   0.0 ]  ← "sat" attends mostly to "cat"
    on [ 0.0   0.0   0.9   0.1   0.0   0.0 ]
   the [ 0.0   0.0   0.0   0.9   0.1   0.0 ]
   mat [ 0.0   0.0   0.0   0.0   0.9   0.1 ]
```

Neither pattern is programmed — both **emerge from training**:
- Head 1 learned that verbs should look at their subjects (useful for understanding grammar)
- Head 2 learned a simple bigram pattern (look left — useful for language modeling)

The output projection $W_O$ then combines both heads' outputs: the final representation of "sat" carries both its subject identity (from Head 1: "the cat is the subject") and its local context (from Head 2: "it follows 'cat'").

This is why splitting into multiple heads is useful — a single head would have to choose between these two patterns, or blend them into a compromised average that captures neither well.

---

## 7. Cross-Attention

### 7.1 What It Is

In self-attention, Q, K, V all come from the same sequence. In **cross-attention**, Q comes from one sequence and K, V come from a different sequence.

```
Self-attention:     Q, K, V all from X     (sequence attends to itself)
Cross-attention:    Q from X, K and V from Y   (X attends to Y)
```

### 7.2 Where It Appears

**Original transformer (encoder-decoder)**:
- Encoder: self-attention (input attends to itself)
- Decoder: masked self-attention (output attends to itself, causally)
- Decoder: **cross-attention** (output attends to encoder output)

```
Decoder step:
  1. Masked self-attention on decoder tokens     (Q, K, V from decoder)
  2. Cross-attention: decoder queries, encoder keys/values
     Q = decoder_hidden @ W_Q
     K = encoder_output @ W_K     <-- from encoder
     V = encoder_output @ W_V     <-- from encoder
  3. Feed-forward network
```

The cross-attention lets the decoder "look at" the input while generating the output. When translating, the decoder at position t uses its query to search through the encoder's key-value pairs to find which input tokens are relevant for generating output token t.

**In multimodal models** (LLaVA, Flamingo):
- Q from the language model
- K, V from the vision encoder
- This lets the language model "see" the image while generating text

**In RAG systems** (conceptually):
- Q from the user query
- K, V from the retrieved documents
- Though in practice, RAG usually concatenates context into the prompt rather than using architectural cross-attention

### 7.3 Cross-Attention Shapes

```
Decoder hidden states:  (B, T_dec, d_model)
Encoder output:         (B, T_enc, d_model)

Q = decoder @ W_Q:     (B, T_dec, d_k)
K = encoder @ W_K:     (B, T_enc, d_k)
V = encoder @ W_V:     (B, T_enc, d_v)

Scores = Q @ K^T:      (B, T_dec, T_enc)    ← rectangular, not square!
@ V:                    (B, T_dec, d_v)
```

The attention matrix is $(T_{\text{dec}} \times T_{\text{enc}})$, not square. Each decoder position distributes attention over all encoder positions.

---

## 8. Causal (Masked) Attention

### 8.1 The Problem

In autoregressive language models (GPT, Llama), the model predicts the next token given all previous tokens:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

Token $t$ must NOT be able to see tokens $t+1, t+2, \ldots, T$. If it could, it would just copy the answer instead of learning to predict.

But self-attention computes: $\text{output}_t = \sum_j A_{tj} \mathbf{v}_j$ over ALL positions $j$ including future ones. We need to prevent attention to future positions.

### 8.2 The Causal Mask

Apply a mask to the attention scores BEFORE softmax:

```
Mask = upper triangular matrix of -infinity

     [  0   -inf  -inf  -inf ]
M =  [  0     0   -inf  -inf ]
     [  0     0     0   -inf ]
     [  0     0     0     0  ]

Masked_Scores = Scores + M
Attention_Weights = softmax(Masked_Scores)
```

After adding $-\infty$ to future positions, softmax converts them to 0:

$$\text{softmax}(\ldots, -\infty, \ldots) \to (\ldots, 0, \ldots)$$

The resulting attention weights:
```
     [ a11   0     0     0   ]     Position 1 attends only to position 1
A =  [ a21  a22    0     0   ]     Position 2 attends to positions 1-2
     [ a31  a32   a33    0   ]     Position 3 attends to positions 1-3
     [ a41  a42   a43   a44  ]     Position 4 attends to positions 1-4
```

This is a lower-triangular matrix. Each row sums to 1 (valid probability distribution over the positions it can see).

### 8.3 Why -Infinity and Not Zero

We mask the **scores** (before softmax), not the **weights** (after softmax).

If we set scores to 0 instead of $-\infty$:

$$\text{softmax}(\ldots, 0, \ldots) \to \text{some positive value (not zero!)}$$

Softmax of 0 is $\exp(0)/\text{Sum} = 1/\text{Sum}$, which is not zero. The future tokens would still receive some attention. Only $-\infty$ guarantees zero attention after softmax, because $\exp(-\infty) = 0$.

### 8.4 Causal Attention During Training vs Inference

**Training**: The entire sequence is processed at once. The causal mask ensures position $t$ only sees positions $1 \ldots t$. All positions are computed in parallel — the mask handles the causality constraint.

```
Training (parallel):
  Input:  [The, cat, sat, on]  (all positions at once)
  Mask:   lower triangular
  Output: [cat, sat, on, the]  (predictions for next token at each position)
  Loss:   computed at all positions simultaneously
```

**Inference**: Tokens are generated one at a time. At step $t$, the model only has tokens $1 \ldots t$, so there's nothing future to mask. But the KV cache stores past computations (see Topic 17).

```
Inference (sequential):
  Step 1: Input [The]       -> predict "cat"
  Step 2: Input [The, cat]  -> predict "sat"
  Step 3: Input [The, cat, sat] -> predict "on"
  ...
```

### 8.5 No Mask in Encoder Models

BERT uses **bidirectional** self-attention — every token attends to every other token (including "future" ones). There is no causal mask.

This is possible because BERT's task is understanding (MLM), not generation. It fills in [MASK] tokens using full context. BERT cannot generate text autoregressively — it's not trained to.

```
Encoder (BERT):   Full attention matrix (no mask)
     [ a11  a12  a13  a14 ]
A =  [ a21  a22  a23  a24 ]     Every position sees every position
     [ a31  a32  a33  a34 ]
     [ a41  a42  a43  a44 ]

Decoder (GPT):    Causal mask (lower triangular)
     [ a11   0    0    0  ]
A =  [ a21  a22   0    0  ]     Each position sees only past + self
     [ a31  a32  a33   0  ]
     [ a41  a42  a43  a44 ]
```

---

## 9. Computational Complexity of Attention

### 9.1 Time Complexity

For sequence length $T$ and head dimension $d_k$:

$QK^T$: $O(T^2 \cdot d_k)$ — $T$ queries, each dot-producted with $T$ keys

softmax: $O(T^2)$ — normalize each of $T$ rows

Attention $\times V$: $O(T^2 \cdot d_v)$ — $T$ outputs, each a weighted sum of $T$ values

Total: $O(T^2 \cdot d)$ — quadratic in sequence length

### 9.2 Memory Complexity

The attention score matrix is $(T \times T)$ per head per layer.

Score matrix: $O(T^2)$ per head

All heads: $O(h \cdot T^2)$

All layers: $O(L \cdot h \cdot T^2)$

For Llama 2 70B with $T=4096$, $L=80$, $h=64$:

$$\text{Attention matrices alone: } 80 \times 64 \times 4096^2 \times 2 \text{ bytes (FP16)} \approx 172 \text{ GB}$$

This is why long context is so expensive and why efficient attention variants (Section 11) are essential.

### 9.3 The Quadratic Bottleneck

```
T=512:    T^2 = 262K        (manageable)
T=2048:   T^2 = 4.2M        (standard)
T=8192:   T^2 = 67M         (expensive)
T=32768:  T^2 = 1.07B       (very expensive)
T=131072: T^2 = 17.2B       (requires Flash Attention + efficient implementation)
```

The quadratic scaling means:
- Doubling sequence length -> 4x more compute and memory for attention
- Going from 2K to 128K context -> 4096x more attention compute
- This is the fundamental limit of standard transformer attention

---

## 10. Modern Attention Variants

Research has produced many variants to make attention faster, more memory-efficient, or more capable.

### 10.1 Multi-Query Attention (MQA)

**Paper**: "Fast Transformer Decoding" (Shazeer, 2019)

**Idea**: All attention heads share the SAME K and V projections. Only Q has separate projections per head.

```
Standard MHA:     h separate Q, K, V projections    -> h * 3 * d_model * d_k params
Multi-Query:      h separate Q, 1 shared K, 1 shared V -> h * d_model * d_k + 2 * d_model * d_k params
```

**Why it helps**: During inference, the KV cache stores K and V for all past tokens. With MHA, the cache is $h$ times larger. MQA reduces KV cache by $h$-fold, dramatically reducing memory bandwidth during autoregressive generation.

**Trade-off**: Slight quality degradation because all heads share the same key-value representation. They can only differ in how they query.

**Used by**: PaLM, Falcon.

### 10.2 Grouped-Query Attention (GQA)

**Paper**: "GQA: Training Generalized Multi-Query Attention from Multi-Head Checkpoints" (Ainslie et al., 2023)

**Idea**: A compromise between MHA and MQA. Group heads into $G$ groups (where $1 < G < h$). Each group shares K and V.

```
MHA:   G = h     (every head has its own K, V)  — full quality, largest KV cache
MQA:   G = 1     (all heads share K, V)         — lowest quality, smallest KV cache
GQA:   G = h/g   (g heads share K, V)           — balanced quality and efficiency
```

**Typical**: Llama 2 70B uses GQA with 8 KV groups for 64 query heads (8 KV heads, each shared by 8 query heads). Llama 2 7B uses standard MHA.

**Why GQA matters**: It's the standard for large models now. It provides near-MHA quality with near-MQA efficiency. You can even convert an MHA-trained model to GQA by mean-pooling KV projections within groups.

### 10.3 Flash Attention

**Paper**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

**The problem**: Standard attention materializes the full $T \times T$ score matrix in GPU high-bandwidth memory (HBM). For long sequences, this matrix is enormous. Even worse, reading/writing this matrix to/from HBM is the bottleneck — not the compute itself. Attention is **memory-bandwidth-bound**, not compute-bound.

**The insight**: Never materialize the full $T \times T$ matrix. Instead, compute attention in blocks (tiles) that fit in GPU SRAM (fast on-chip memory), and accumulate results.

**Algorithm sketch**:
1. Divide Q, K, V into blocks that fit in SRAM
2. For each block of Q:
   a. Load the block of Q into SRAM
   b. For each block of K, V:
      - Load K, V block into SRAM
      - Compute block of scores ($Q_{\text{block}} K_{\text{block}}^T$)
      - Apply block-wise softmax with online numerically-stable accumulation
      - Update the output accumulator
3. Write the final output back to HBM

**Key technique**: Online softmax computation. Standard softmax requires the max over the entire row to be stable. Flash Attention computes a running max and rescales on-the-fly as new blocks are processed.

**Results**:
- **Memory**: $O(T)$ instead of $O(T^2)$ — no materialized attention matrix
- **Speed**: 2-4x faster than standard attention (fewer HBM reads/writes)
- **Exact**: No approximation — mathematically identical to standard attention
- **Enables long context**: Without Flash Attention, 128K context would be infeasible

**Flash Attention 2** (2023): Further optimizations (better parallelism, fewer non-matmul FLOPs). ~2x faster than Flash Attention 1.

**Flash Attention 3** (2024): Exploits Hopper GPU features (asynchronous execution, FP8).

### 10.4 Sliding Window Attention

**Used by**: Mistral 7B

**Idea**: Each token only attends to a fixed window of $W$ recent tokens, not the entire sequence.

Standard: Token $t$ attends to positions $[1, 2, \ldots, t]$ — $O(T^2)$ total

Sliding window: Token $t$ attends to positions $[t - W + 1, \ldots, t]$ — $O(T \cdot W)$ total

For $W = 4096$ and $T = 32768$, this is 8x cheaper.

**But doesn't this lose long-range information?** Not entirely. Because of the stacking of layers:
- Layer 1: Each token sees $W$ tokens
- Layer 2: Each token sees tokens that saw $W$ tokens, so effective range is $2W$
- Layer $L$: Effective receptive field is $L \cdot W$

With $L = 32$ layers and $W = 4096$: effective range = 131,072 tokens. Information propagates across layers, similar to how stacked CNN layers build large receptive fields.

---

## 11. What Attention Heads Actually Learn

### 11.1 Empirical Findings

Research on interpreting attention heads (Clark et al. 2019, Voita et al. 2019) has revealed that different heads specialize:

**Syntactic heads**:
- Attend from a verb to its subject ("The cat [sat]" — "sat" attends to "cat")
- Attend from a pronoun to its antecedent ("it" attends to "the animal")
- Attend to the previous token (bigram pattern)

**Positional heads**:
- Attend to the immediately preceding token (very common — almost every model has these)
- Attend to the first token (beginning of sentence/sequence)
- Attend to nearby tokens within a fixed window

**Semantic heads**:
- Attend to semantically related words
- Attend within named entities

**Copy/induction heads** (Olsson et al. 2022):
- These are particularly important for in-context learning
- An **induction head** implements the pattern: if sequence A B ... A appeared, predict B after the second A
- They compose two heads: one that looks at the previous token, and one that finds where that token appeared before

---

## 12. Interview Questions & Answers

### Q1: Derive scaled dot-product attention from scratch. Why divide by $\sqrt{d_k}$?

**Answer**: We want a mechanism where each position computes a weighted average of all positions' representations, with weights based on pairwise relevance.

Start with a query $\mathbf{q}_i$ (what position $i$ is looking for) and keys $\mathbf{k}_j$ (what each position $j$ offers). The natural measure of relevance is the dot product:

$$s_{ij} = \mathbf{q}_i^T \mathbf{k}_j$$

Convert scores to a valid probability distribution using softmax:

$$\alpha_{ij} = \text{softmax}(s_{ij}) = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}$$

Compute the output as the weighted sum of value vectors:

$$\text{output}_i = \sum_j \alpha_{ij} \mathbf{v}_j$$

In matrix form: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T) V$.

Now, why scale by $\sqrt{d_k}$? If $\mathbf{q}$ and $\mathbf{k}$ have entries from $\mathcal{N}(0, 1)$, then $\mathbf{q}^T \mathbf{k}$ has mean 0 and variance $d_k$. For large $d_k$ (e.g., 128), the dot products have large magnitude — some entries of $QK^T$ will be +30 or -30.

When softmax receives inputs with large magnitude, it saturates: the largest value gets probability ~1 and everything else gets ~0. In these saturated regions, the gradients are nearly zero (softmax gradient involves terms like $\alpha(1 - \alpha)$, which is ~0 when $\alpha$ is near 0 or 1). This causes:
- Vanishing gradients through the attention layer
- The model can't learn nuanced, distributed attention patterns
- Training is slow or unstable

Dividing by $\sqrt{d_k}$ normalizes the variance of the dot products back to 1, keeping them in a range where softmax has meaningful gradients. The complete formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Q2: What would happen if you didn't scale? What if you used $1/d_k$ instead of $1/\sqrt{d_k}$?

**Answer**: Without scaling, the attention would become nearly hard (one-hot) attention for large $d_k$, because the dot products would push softmax into saturation. The model would essentially attend to only one position per query, losing the ability to aggregate information from multiple positions. Gradients would vanish, making training extremely slow or impossible.

If you used $1/d_k$ instead of $1/\sqrt{d_k}$, the scores would be over-dampened. The variance would be $1/d_k$ instead of 1, making all scores very close to zero. Softmax of near-zero values produces near-uniform distributions. The attention would become too diffuse — attending roughly equally to everything — losing the ability to discriminate between relevant and irrelevant positions.

The $1/\sqrt{d_k}$ is the Goldilocks scaling: it keeps the variance at exactly 1, ensuring softmax operates in its informative regime where it can produce both peaked and distributed attention patterns depending on the actual query-key similarity.

### Q3: Why multiple heads instead of one large attention? What do different heads capture?

**Answer**: Multi-head attention runs $h$ independent attention operations in parallel, each in a $d_k = d_{\text{model}}/h$ dimensional subspace.

The key advantage is **representational diversity**. A single attention head computes one set of attention weights — one "view" of relevance. But relevance is multidimensional. The same pair of tokens might be related syntactically (subject-verb), semantically (synonym), positionally (adjacent), or structurally (opening and closing bracket).

With multiple heads, each head can specialize in a different type of relationship. Empirically:
- Some heads learn to attend to the previous token (positional/local patterns)
- Some heads track subject-verb agreement across long distances
- Some heads resolve coreference ("it" -> "the cat")
- "Induction heads" implement in-context learning by finding and copying past patterns

Critically, multi-head attention has the **same parameter count** as single-head attention with the same total dimension. The projections for all heads together are exactly equivalent to the single-head projections. So multiple heads give you representational diversity for free.

The output projection $W_O$ then learns to combine the different perspectives from all heads into a single representation that's most useful for the next layer.

### Q4: Explain cross-attention. When is it used vs self-attention?

**Answer**: In self-attention, Q, K, and V all derive from the same sequence — each position attends to every other position in the same sequence.

In cross-attention, Q comes from one sequence (typically the decoder) and K, V come from a different sequence (typically the encoder output). This lets one sequence "look at" another.

The mechanism is identical — $QK^T / \sqrt{d_k}$, softmax, weighted sum of $V$ — but Q and K/V come from different sources.

**Where cross-attention is used**:
- **Encoder-decoder transformers** (T5, BART, original transformer): The decoder uses cross-attention to attend to the encoder's output at each layer. This is how the decoder knows what the input says.
- **Multimodal models**: Language model queries attend to vision encoder key-values (Flamingo, LLaVA-style models), allowing the LLM to "see" image features.
- **Retrieval-augmented models**: Some architectures (FiD, RETRO) use cross-attention to integrate retrieved documents.

**Why GPT/Llama don't need cross-attention**: Decoder-only models have no separate encoder. Everything — the input prompt and the generated output — is one sequence. Self-attention handles both "understanding the input" and "generating the output" in a single stream. This simplicity is one reason decoder-only models became dominant.

### Q5: What is the causal mask? Why is it necessary for autoregressive generation?

**Answer**: The causal mask is a lower-triangular mask applied to the attention scores before softmax, setting all positions above the diagonal to $-\infty$.

It's necessary because autoregressive models are trained to predict each token given only the preceding tokens: $P(x_t \mid x_1, \ldots, x_{t-1})$. If token $t$ could attend to token $t+1$, the model would learn to "cheat" by looking at the future instead of learning to predict it.

The mask is applied as:

$$\text{Scores}_{\text{masked}} = \text{Scores} + M, \quad \text{where } M_{ij} = 0 \text{ if } j \leq i, \text{ else } -\infty$$

After softmax, positions with $-\infty$ scores receive zero attention weight, effectively making them invisible.

During training, the entire sequence is processed in one forward pass (for efficiency), and the causal mask ensures each position can only see past positions. This lets us compute the loss at all positions simultaneously while maintaining the autoregressive property.

During inference, tokens are generated sequentially and the model naturally only has access to past tokens. The mask isn't strictly needed, but the KV cache implementation typically follows the same pattern.

Encoder models (BERT) don't use causal masking because they need full bidirectional context for understanding tasks.

### Q6: How does Flash Attention achieve $O(T)$ memory instead of $O(T^2)$?

**Answer**: Standard attention materializes the full $T \times T$ score matrix $S = QK^T$ in GPU memory (HBM). For $T = 128K$, this matrix alone is $128K \times 128K = 16.4$ billion entries $\approx$ 33 GB in FP16. This is both a memory problem and a speed problem — reading and writing this matrix to/from HBM is the actual bottleneck.

Flash Attention's key insight is that we never need the full $T \times T$ matrix at once. We only need each row of the attention weights to compute the corresponding output. So instead of materializing the whole matrix, we compute attention in tiles:

1. Divide Q into blocks of size $B_q$ and K, V into blocks of size $B_{kv}$ (chosen to fit in SRAM)
2. For each Q block, iterate over all K, V blocks:
   - Compute the block of scores in SRAM (fast on-chip memory)
   - Update a running softmax using an online algorithm (track running max and sum for numerical stability)
   - Accumulate the output contribution
3. Write only the final output $(T \times d_v)$ back to HBM

The critical trick is the **online softmax**: standard softmax requires knowing the maximum across the entire row, which requires seeing all K blocks first. Flash Attention maintains a running maximum and rescales accumulated values when a new maximum is found. This is mathematically exact — the result is bit-for-bit identical to standard attention.

Memory: we never store the $T \times T$ matrix. We only store $O(T \cdot d)$ for the output and $O(B)$ for the temporary tile in SRAM. Total HBM memory is $O(T)$, not $O(T^2)$.

### Q7: Compare MHA, MQA, and GQA. What are the trade-offs?

**Answer**: All three compute multi-head attention but differ in how K and V projections are shared across heads.

**MHA (Multi-Head Attention)**: Each of $h$ heads has its own Q, K, V projections. Full expressiveness. KV cache during inference stores $h$ separate key-value pairs per token per layer.

**MQA (Multi-Query Attention)**: Each head has its own Q projection, but all heads share ONE K and ONE V projection. KV cache is $h$ times smaller. But all heads are forced to attend based on the same key-value representation — they can only differ in what they query, not what they match against. Quality degrades noticeably.

**GQA (Grouped-Query Attention)**: Compromise. Heads are divided into $G$ groups. Within each group, heads share K and V. KV cache is $h/G$ times smaller than MHA. Quality is very close to MHA when $G$ is chosen well (e.g., $G = 8$ for $h = 64$).

```
Model           | Heads (h) | KV heads (G) | KV cache size | Quality
MHA             | 64        | 64           | 64 * d_k      | Best
GQA (Llama 2)   | 64        | 8            | 8 * d_k       | Near-MHA
MQA             | 64        | 1            | 1 * d_k       | Noticeably worse
```

GQA has become the standard for large models because the KV cache is the main memory bottleneck during inference. For a 70B model serving long sequences, the KV cache can easily exceed 100 GB. Reducing it by 8x (with 8 KV groups) makes serving feasible on fewer GPUs with minimal quality loss.

### Q8: Explain the difference between Bahdanau and Luong attention. How did they lead to the transformer?

**Answer**: Both solve the Seq2Seq bottleneck by letting the decoder attend to all encoder hidden states, but they differ in the scoring mechanism.

**Bahdanau (additive)**: Uses a small neural network to score relevance:

$$e_{t,i} = \mathbf{v}^T \tanh(W_1 \mathbf{s}_{t-1} + W_2 \mathbf{h}_i)$$

Uses the previous decoder state $\mathbf{s}_{t-1}$, applies attention before the decoder RNN step. More expressive but slower.

**Luong (multiplicative)**: Uses a dot product (or bilinear form):

$$e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i \quad \text{(dot)} \quad \text{or} \quad \mathbf{s}_t^T W \mathbf{h}_i \quad \text{(general)}$$

Uses the current decoder state $\mathbf{s}_t$, applies attention after the decoder step. Simpler and faster.

The evolution to the transformer:
1. Luong showed dot-product attention works well — no need for a neural network scorer
2. The transformer generalized this by adding learned projections ($W_Q$, $W_K$, $W_V$) so the model learns what to query and what to match on
3. Added scaling by $1/\sqrt{d_k}$ for numerical stability
4. Applied attention as self-attention (not just cross-attention), removing the need for recurrence entirely
5. Added multiple heads for representational diversity

The transformer kept the core idea (weighted sum of values based on query-key similarity) but removed all recurrence, enabling full parallelization.

### Q9: What is an induction head? Why does it matter for in-context learning?

**Answer**: An induction head is a specific circuit composed of two attention heads working together that implements a pattern-matching and copying mechanism.

The pattern it implements: if the sequence contains "...A B ... A", the induction head predicts B will follow the second occurrence of A. It essentially does: "I've seen this token before. What followed it last time? Predict that."

It works through composition of two heads:
1. **Previous-token head** (in an earlier layer): Attends to the token before the current one, creating a representation that encodes "what token preceded me"
2. **Induction head** (in a later layer): Searches for previous occurrences of the current token's predecessor pattern and copies the token that followed

This is important because it's the mechanism underlying in-context learning. When you give an LLM few-shot examples like:
```
France -> Paris
Germany -> Berlin
Japan ->
```
The induction head recognizes the pattern (country -> capital) and copies the appropriate answer. Olsson et al. (2022) showed that induction heads emerge during training at a specific point that correlates with the sudden improvement in in-context learning ability.

### Q10: Attention is $O(T^2)$. What are the practical consequences, and what are the main approaches to address it?

**Answer**: The quadratic complexity means that doubling the sequence length quadruples the compute and memory for attention. This creates concrete problems:

- Training on 4K context uses 16x more attention compute than 1K context
- Serving a model with 128K context uses 4096x more than 1K
- The attention score matrix for $T=128K$ is $128K \times 128K = 16.4B$ entries per head per layer

**Practical consequences**:
- Context length is the main constraint on model capability (can't process long documents, codebases, conversations)
- Inference cost scales quadratically with conversation length
- Batch sizes must decrease as context length increases (memory bound)

**Main approaches**:

1. **Efficient exact attention** (Flash Attention): Doesn't change the math, just computes it smarter. Reduces memory from $O(T^2)$ to $O(T)$ by tiling and online softmax. 2-4x faster. This is the most widely adopted solution.

2. **Sparse attention** (sliding window, Longformer, BigBird): Only compute attention for a subset of position pairs. Reduces compute to $O(T \cdot W)$ where $W$ is window size. Works well for many tasks but can miss very long-range dependencies.

3. **Linear attention** (Linear Transformer, RWKV): Replace softmax with a kernel that allows reordering the computation from $O(T^2 \cdot d)$ to $O(T \cdot d^2)$. Significant quality trade-off.

4. **State space models** (Mamba): Entirely different architecture that processes sequences in $O(T)$ time with no attention at all. Emerging alternative, covered in Topic 27.

5. **KV cache compression**: During inference, compress or evict old key-value pairs to limit memory growth. Techniques include: eviction based on attention weight (discard rarely-attended positions), quantization of cached KV pairs, or windowed + sink token caching.

In practice, the field has converged on: Flash Attention for training and moderate contexts + sliding window or GQA for inference efficiency + RoPE scaling for extending beyond trained context length.

---

*Next: [Topic 9: Transformer Architecture](09_Transformer_Architecture.md)*
