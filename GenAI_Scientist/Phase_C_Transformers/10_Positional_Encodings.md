# Topic 10: Positional Encodings (Deep Dive)

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 10 of 28
> **Scope**: Why position matters, sinusoidal encodings, learned embeddings, relative position (Shaw, Transformer-XL), RoPE, ALiBi, context length extension (NTK-aware scaling, YaRN, Position Interpolation)
> **Why this matters**: Without positional information, a transformer treats "dog bites man" identically to "man bites dog." Positional encoding is what gives the transformer its sense of order. RoPE is in every modern LLM — you need to understand the math cold, including how models extend to longer contexts than they were trained on.
> **Previous**: [Topic 9: Transformer Architecture](09_Transformer_Architecture.md)
> **Next**: [Topic 11: Encoder Models (BERT Family)](11_BERT_Family.md)

---

## Table of Contents

1. [Why Transformers Need Positional Information](#1-why-transformers-need-positional-information)
2. [Sinusoidal Positional Encoding (Original Paper)](#2-sinusoidal-positional-encoding-original-paper)
3. [Learned Absolute Positional Embeddings](#3-learned-absolute-positional-embeddings)
4. [Relative Positional Encoding (Shaw et al.)](#4-relative-positional-encoding-shaw-et-al)
5. [Transformer-XL: Relative Position in Recurrence](#5-transformer-xl-relative-position-in-recurrence)
6. [Rotary Position Embedding (RoPE)](#6-rotary-position-embedding-rope)
7. [ALiBi — Attention with Linear Biases](#7-alibi--attention-with-linear-biases)
8. [Context Length Extension — The Frontier Problem](#8-context-length-extension--the-frontier-problem)
9. [Comparison of All Methods](#9-comparison-of-all-methods)
10. [Positional Encodings in Vision and Beyond](#10-positional-encodings-in-vision-and-beyond)
11. [Interview Questions & Answers](#11-interview-questions--answers)

---

## 1. Why Transformers Need Positional Information

### 1.1 The Permutation Invariance Problem

Self-attention computes:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Consider what happens if we permute the input. Let $\pi$ be any permutation of positions. If the input is $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n]$, the permuted input is $\mathbf{X}_\pi = [\mathbf{x}_{\pi(1)}, \mathbf{x}_{\pi(2)}, \ldots, \mathbf{x}_{\pi(n)}]$.

The attention output for the permuted input is simply the permuted attention output of the original:

$$
\text{Attention}(\mathbf{X}_\pi \mathbf{W}^Q, \mathbf{X}_\pi \mathbf{W}^K, \mathbf{X}_\pi \mathbf{W}^V) = \pi \circ \text{Attention}(\mathbf{X}\mathbf{W}^Q, \mathbf{X}\mathbf{W}^K, \mathbf{X}\mathbf{W}^V)
$$

This is **permutation equivariance** — the transformer processes inputs as a **set**, not a **sequence**. Without positional information:

```
"The cat sat on the mat"     ← Same attention patterns
"mat the on sat cat The"     ← Identical outputs (just permuted)
```

The model cannot distinguish "dog bites man" from "man bites dog." Language is inherently sequential — word order carries meaning. We must inject position information.

### 1.2 Where to Inject Position

There are three strategies for injecting positional information:

| Strategy | Where | Examples |
|----------|-------|---------|
| **Add to input** | Modify the input embeddings | Sinusoidal, Learned absolute |
| **Modify attention scores** | Bias the attention logits | ALiBi, T5 relative bias |
| **Modify Q/K representations** | Rotate or transform Q and K | RoPE, Relative position (Shaw) |

Each has different trade-offs for expressivity, extrapolation, and efficiency.

### 1.3 Desirable Properties

An ideal positional encoding should:

1. **Uniquely identify each position** — different positions get different encodings
2. **Capture relative distance** — the model should easily determine that positions 3 and 5 are closer than positions 3 and 50
3. **Generalize to unseen lengths** — work on sequences longer than those seen during training
4. **Be efficient** — minimal additional computation
5. **Be translation-invariant** — the relationship between positions $i$ and $j$ should depend only on $i - j$, not on absolute values

No single method achieves all five perfectly. The evolution of positional encodings is the story of trading off between these properties.

---

## 2. Sinusoidal Positional Encoding (Original Paper)

### 2.1 The Formulation

Vaswani et al. (2017) proposed encoding position using sine and cosine functions of different frequencies:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

where:
- $pos$ is the position in the sequence (0, 1, 2, ...)
- $i$ is the dimension index ($0 \leq i < d_{\text{model}}/2$)
- Even dimensions get sine, odd dimensions get cosine

The positional encoding vector for position $pos$ is:

$$
PE(pos) = \begin{bmatrix} \sin(\omega_0 \cdot pos) \\ \cos(\omega_0 \cdot pos) \\ \sin(\omega_1 \cdot pos) \\ \cos(\omega_1 \cdot pos) \\ \vdots \\ \sin(\omega_{d/2-1} \cdot pos) \\ \cos(\omega_{d/2-1} \cdot pos) \end{bmatrix}
$$

where $\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$.

### 2.2 The Frequency Spectrum

The frequencies form a **geometric progression** from high to low:

$$
\omega_i = 10000^{-2i/d_{\text{model}}}
$$

| Dimension pair $(2i, 2i+1)$ | Frequency $\omega_i$ | Wavelength $\lambda = 2\pi/\omega$ |
|------------------------------|----------------------|--------------------------------------|
| $(0, 1)$ — first pair | $1.0$ | $2\pi \approx 6.3$ positions |
| $(d/4, d/4+1)$ — middle | $0.01$ | $\approx 628$ positions |
| $(d-2, d-1)$ — last pair | $0.0001$ | $\approx 62{,}832$ positions |

**Low-dimensional** pairs oscillate rapidly — they distinguish nearby positions (fine-grained position).
**High-dimensional** pairs oscillate slowly — they distinguish distant positions (coarse-grained position).

Think of it like a **clock**: the seconds hand (fast) tells you the exact time within a minute; the hours hand (slow) tells you the rough time of day. Together, they uniquely identify any point in time.

```
Dimension 0 (high freq):  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿   (changes every position)
Dimension 64 (med freq):  ∿    ∿    ∿    ∿      (changes every ~10 positions)
Dimension 510 (low freq): ∿                      (changes every ~10000 positions)
                          pos 0              pos 100
```

### 2.3 Why Sine and Cosine?

The choice of $\sin$ and $\cos$ isn't arbitrary. For any fixed offset $k$, the positional encoding at position $pos + k$ can be expressed as a **linear transformation** of the encoding at position $pos$:

$$
\begin{bmatrix} \sin(\omega(pos + k)) \\ \cos(\omega(pos + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix} \begin{bmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{bmatrix}
$$

This is a **rotation matrix** $R_k$. The relative position $k$ is encoded as a rotation.

**Why this matters**: Since the transformation from $PE(pos)$ to $PE(pos+k)$ is linear (a rotation), the model can learn to attend to relative positions via a linear operation on the positional encodings. The dot product $PE(pos)^T PE(pos + k)$ depends only on $k$, not on $pos$ — encoding relative distance naturally.

### 2.4 Properties of Sinusoidal Encodings

**Unique identification**: Each position gets a unique vector (like a binary encoding but continuous).

**Bounded**: All values are in $[-1, 1]$, regardless of position. No magnitude explosion for long sequences.

**Smooth distance**: $\|PE(pos) - PE(pos + k)\|$ varies smoothly with $k$, with nearby positions being more similar.

**Dot product encodes relative position**:

$$
PE(pos_1)^T PE(pos_2) = \sum_{i=0}^{d/2-1} \cos(\omega_i(pos_1 - pos_2))
$$

This depends only on $pos_1 - pos_2$, not on the absolute positions.

### 2.5 Limitations

1. **No learning**: The sinusoidal encoding is fixed — the model can't optimize the positional signal during training.

2. **Additive interference**: Adding PE to the token embedding conflates two types of information (identity and position) in the same vector. The model must learn to disentangle them.

3. **Extrapolation**: While theoretically the sinusoidal functions extend to any length, in practice models trained on length $n$ don't generalize well to length $> n$. The model hasn't learned to *use* those position values.

4. **Not used in modern LLMs**: Every major model since GPT-2 has moved to alternative approaches (learned, RoPE, or ALiBi).

---

## 3. Learned Absolute Positional Embeddings

### 3.1 The Idea

Instead of using fixed mathematical functions, simply **learn** a position embedding for each position:

$$
\mathbf{h}_i^{(0)} = \text{TokenEmbed}(x_i) + \mathbf{p}_i
$$

where $\mathbf{p}_i \in \mathbb{R}^{d_{\text{model}}}$ is a learned vector for position $i$, stored in a learnable embedding matrix $\mathbf{P} \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$.

### 3.2 Who Uses This

| Model | Max Length | Learned PE Params |
|-------|-----------|-------------------|
| BERT | 512 | 512 × 768 = 393K |
| GPT-2 | 1024 | 1024 × 768/1600 |
| GPT-3 | 2048 | 2048 × 12288 = 25M |

### 3.3 Advantages

1. **Flexibility**: The model can learn arbitrary position-dependent patterns. If certain positions are special (e.g., position 0 for [CLS] in BERT), the learned embedding can capture this.

2. **Simplicity**: Easy to implement — just another embedding lookup table.

3. **Strong empirical performance**: For models that don't need to extrapolate beyond their training length, learned embeddings perform as well as or better than sinusoidal.

### 3.4 Limitations

1. **Hard length limit**: The model has embeddings for positions $0$ to $n_{\max} - 1$. It literally cannot process position $n_{\max}$. There's no embedding for it.

2. **No parameter sharing across positions**: Position 100 and position 101 share no parameters, even though they're adjacent. The model must independently learn that nearby positions behave similarly.

3. **No extrapolation**: This is the critical weakness. A model trained with max length 2048 cannot handle length 2049. This motivated the search for relative position methods.

4. **Absolute, not relative**: The model sees absolute position. The relationship between "the 3rd and 5th tokens" must be learned separately from "the 103rd and 105th tokens," even though the relative offset is the same.

### 3.5 What Learned Embeddings Actually Learn

Visualization of learned position embeddings (from BERT and GPT-2) reveals:

- **Low dimensions**: Smooth, quasi-sinusoidal patterns — the model rediscovers the sinusoidal structure
- **High dimensions**: More noisy, less structured patterns
- **Nearby positions**: Have similar embeddings (cosine similarity is high)
- **Distant positions**: Have dissimilar embeddings

This suggests that the sinusoidal design was a reasonable inductive bias — the model learns something similar when given freedom.

---

## 4. Relative Positional Encoding (Shaw et al.)

### 4.1 Motivation

Absolute positional encodings tell the model "this token is at position 5." But for many NLP tasks, what matters is **relative position**: "this token is 3 positions to the left of that token."

Shaw et al. (2018) proposed modifying the attention mechanism itself to incorporate relative position.

### 4.2 The Formulation

In standard attention, the score between positions $i$ and $j$ is:

$$
e_{ij} = \frac{(\mathbf{x}_i \mathbf{W}^Q)(\mathbf{x}_j \mathbf{W}^K)^T}{\sqrt{d_k}}
$$

Shaw et al. add a **relative position bias** to both the attention scores and the values:

$$
e_{ij} = \frac{(\mathbf{x}_i \mathbf{W}^Q)(\mathbf{x}_j \mathbf{W}^K + \mathbf{a}_{ij}^K)^T}{\sqrt{d_k}}
$$

$$
z_i = \sum_j \alpha_{ij} (\mathbf{x}_j \mathbf{W}^V + \mathbf{a}_{ij}^V)
$$

where $\mathbf{a}_{ij}^K, \mathbf{a}_{ij}^V \in \mathbb{R}^{d_k}$ are learned relative position embeddings that depend only on the **offset** $i - j$, not on the absolute positions.

### 4.3 Clipping

To keep the number of learnable parameters finite, the relative distance is clipped:

$$
\mathbf{a}_{ij}^K = \mathbf{w}_{\text{clip}(i-j, -k, k)}^K
$$

where $\text{clip}(x, a, b) = \max(a, \min(b, x))$ and $k$ is the maximum relative distance (e.g., $k = 16$). Beyond distance $k$, the model treats all distant positions the same.

This means we only need to learn $2k + 1$ relative position embeddings instead of $n^2$.

### 4.4 Impact

This was the first successful relative position approach. Key insight: **the attention mechanism itself should encode relative position**, rather than adding position to the input.

**Limitation**: Still requires modifying the attention computation in a way that's not trivially parallelizable on GPUs — the per-pair bias $\mathbf{a}_{ij}$ is awkward to implement efficiently.

---

## 5. Transformer-XL: Relative Position in Recurrence

### 5.1 Context

Transformer-XL (Dai et al., 2019) was designed for **very long sequences**. It introduced segment-level recurrence — processing long documents in chunks while carrying hidden states from previous chunks.

This required a rethinking of positional encoding, because the same position index (e.g., position 3) occurs in every chunk.

### 5.2 The Relative Position Decomposition

Transformer-XL decomposes the standard attention score into four terms. Start with the absolute position attention score:

$$
e_{ij} = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^Q {\mathbf{W}^K}^T (\mathbf{x}_j + \mathbf{p}_j)^T
$$

Expanding:

$$
e_{ij} = \underbrace{\mathbf{x}_i \mathbf{W}^Q {\mathbf{W}^K}^T \mathbf{x}_j^T}_{(a) \text{ content→content}} + \underbrace{\mathbf{x}_i \mathbf{W}^Q {\mathbf{W}^K}^T \mathbf{p}_j^T}_{(b) \text{ content→position}} + \underbrace{\mathbf{p}_i \mathbf{W}^Q {\mathbf{W}^K}^T \mathbf{x}_j^T}_{(c) \text{ position→content}} + \underbrace{\mathbf{p}_i \mathbf{W}^Q {\mathbf{W}^K}^T \mathbf{p}_j^T}_{(d) \text{ position→position}}
$$

Transformer-XL replaces this with a **relative** version:

$$
e_{ij} = \underbrace{\mathbf{x}_i \mathbf{W}^Q {\mathbf{W}_E^K}^T \mathbf{x}_j^T}_{(a) \text{ content-based addressing}} + \underbrace{\mathbf{x}_i \mathbf{W}^Q {\mathbf{W}_R^K}^T \mathbf{R}_{i-j}^T}_{(b) \text{ content-dependent position bias}} + \underbrace{\mathbf{u} \cdot {\mathbf{W}_E^K}^T \mathbf{x}_j^T}_{(c) \text{ global content bias}} + \underbrace{\mathbf{v} \cdot {\mathbf{W}_R^K}^T \mathbf{R}_{i-j}^T}_{(d) \text{ global position bias}}
$$

Key changes:
- **Terms (b) and (d)**: Replace absolute $\mathbf{p}_j$ with relative sinusoidal encoding $\mathbf{R}_{i-j}$
- **Term (c)**: Replace query-specific $\mathbf{p}_i \mathbf{W}^Q$ with a global learnable bias $\mathbf{u}$ (since the query's absolute position shouldn't matter for content-based attention)
- **Term (d)**: Similarly replace with global bias $\mathbf{v}$

### 5.3 Legacy

Transformer-XL's decomposition was influential but complex. The key insight — **replace absolute positions with relative offsets in the attention score** — directly inspired RoPE and ALiBi, which achieved the same goal more elegantly.

---

## 6. Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) is the positional encoding used in virtually every modern LLM: **Llama 1/2/3, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Phi, CodeLlama**. Understanding RoPE deeply is essential.

### 6.1 The Core Idea

RoPE encodes position by **rotating** the query and key vectors. The rotation angle depends on the position, so the dot product between a rotated query and a rotated key naturally encodes their **relative position**.

**Key insight**: If we rotate $\mathbf{q}$ at position $m$ by angle $m\theta$ and rotate $\mathbf{k}$ at position $n$ by angle $n\theta$, then their dot product:

$$
\text{Re}[\mathbf{q}_m^* \mathbf{k}_n] = \text{Re}[\mathbf{q}^* e^{-im\theta} \cdot \mathbf{k} e^{in\theta}] = \text{Re}[\mathbf{q}^* \mathbf{k} \cdot e^{i(n-m)\theta}]
$$

depends only on $n - m$ (the relative position), not on the absolute positions $m$ or $n$.

### 6.2 Mathematical Formulation

#### Working in 2D First

Consider a 2D query-key pair. At position $m$, we rotate the query by angle $m\theta$:

$$
\mathbf{R}_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

The rotated query and key are:

$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{q}, \qquad \mathbf{k}_n = \mathbf{R}_n \mathbf{k}
$$

The attention score is:

$$
\mathbf{q}_m^T \mathbf{k}_n = (\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}
$$

because rotation matrices have the property $\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{n-m}$.

The dot product depends only on the **relative position** $n - m$. This is exactly what we want.

#### Extending to $d$-Dimensions

For a $d$-dimensional vector, we pair up dimensions $(0,1), (2,3), (4,5), \ldots, (d-2, d-1)$ and apply **independent 2D rotations** to each pair, with different frequencies:

$$
\mathbf{R}_m^{(d)} = \begin{bmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & & & \\ \sin(m\theta_0) & \cos(m\theta_0) & & & \\ & & \cos(m\theta_1) & -\sin(m\theta_1) & \\ & & \sin(m\theta_1) & \cos(m\theta_1) & \\ & & & & \ddots \\ & & & & & \cos(m\theta_{d/2-1}) & -\sin(m\theta_{d/2-1}) \\ & & & & & \sin(m\theta_{d/2-1}) & \cos(m\theta_{d/2-1}) \end{bmatrix}
$$

where the frequencies are:

$$
\theta_i = 10000^{-2i/d}
$$

This is the same frequency scheme as sinusoidal encodings — a geometric progression from high to low frequency.

### 6.3 Efficient Implementation

The full rotation matrix is sparse (block-diagonal with 2×2 blocks). In practice, RoPE is computed element-wise without forming the matrix:

For a vector $\mathbf{x} = [x_0, x_1, x_2, x_3, \ldots, x_{d-2}, x_{d-1}]$:

$$
\text{RoPE}(\mathbf{x}, m) = \begin{bmatrix} x_0 \cos(m\theta_0) - x_1 \sin(m\theta_0) \\ x_0 \sin(m\theta_0) + x_1 \cos(m\theta_0) \\ x_2 \cos(m\theta_1) - x_3 \sin(m\theta_1) \\ x_2 \sin(m\theta_1) + x_3 \cos(m\theta_1) \\ \vdots \end{bmatrix}
$$

This can be expressed compactly as:

$$
\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})
$$

where $\text{rotate\_half}$ swaps pairs and negates: $[x_0, x_1, x_2, x_3] \to [-x_1, x_0, -x_3, x_2]$.

This is a simple, efficient operation — no matrix multiplication needed.

### 6.4 Applied to Attention

RoPE is applied to the queries and keys **after** the linear projections but **before** the dot product:

$$
\text{Attention}_{ij} = \frac{(\mathbf{R}_i \mathbf{W}^Q \mathbf{x}_i)^T (\mathbf{R}_j \mathbf{W}^K \mathbf{x}_j)}{\sqrt{d_k}}
$$

Note: RoPE is **not** applied to values $\mathbf{V}$. The values carry content information — position should affect *which* tokens we attend to (Q, K), not *what* information we retrieve (V).

### 6.5 Why RoPE Works So Well

1. **Relative position naturally**: The dot product $\mathbf{q}_m^T \mathbf{k}_n$ depends on $m - n$, not absolute positions. No need for explicit relative position biases.

2. **No additive interference**: Unlike sinusoidal or learned embeddings that are *added* to the input (mixing content and position), RoPE is a *multiplicative* transformation (rotation). Content and position information don't interfere.

3. **Long-range decay**: The dot product between distant positions naturally decays:

$$
\mathbf{q}_m^T \mathbf{k}_n = \sum_{i=0}^{d/2-1} \left[(\mathbf{q}_{2i} \mathbf{k}_{2i} + \mathbf{q}_{2i+1}\mathbf{k}_{2i+1})\cos((m-n)\theta_i) + (\mathbf{q}_{2i}\mathbf{k}_{2i+1} - \mathbf{q}_{2i+1}\mathbf{k}_{2i})\sin((m-n)\theta_i)\right]
$$

For large $|m-n|$, the high-frequency terms oscillate rapidly and tend to cancel, creating a natural distance-based decay. The model attends more strongly to nearby positions by default.

4. **Compatibility with KV cache**: RoPE is applied independently to each position's Q and K. When using a KV cache, previously computed K values (already rotated) remain valid — no need to recompute.

5. **No additional parameters**: RoPE is entirely determined by the position and frequency formula. Zero learnable parameters for position encoding.

### 6.6 Visualizing RoPE

Think of each 2D dimension pair as a **clock hand**:

```
Position 0:    Position 1:    Position 2:    Position 5:
    |              /              ─              \
    |  θ=0°      /  θ=36°       θ=72°           θ=180°
    |            /

(High-frequency dimension pair — rotates quickly)

Position 0:    Position 1:    Position 2:    Position 5:
    |              |              |              |
    |  θ=0°      |  θ=3.6°     |  θ=7.2°      / θ=18°
    |            |              |             /

(Low-frequency dimension pair — rotates slowly)
```

Each position corresponds to a unique combination of rotation angles across all dimension pairs. The relative rotation between two positions depends only on their distance.

---

## 7. ALiBi — Attention with Linear Biases

### 7.1 The Idea

ALiBi (Press et al., 2022) takes a radically different approach: instead of encoding position in the input or in Q/K, it adds a **static, linear bias** directly to the attention scores:

$$
\text{score}_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|
$$

where $m$ is a head-specific slope that penalizes attention by distance.

That's it. No positional encodings in the input. No rotation. Just a linear penalty on distance in the attention scores.

### 7.2 The Slopes

Each attention head gets a different slope $m_h$, forming a geometric sequence:

$$
m_h = 2^{-8h/H}
$$

For a model with $H = 8$ heads:

$$
m = \left[\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\right]
$$

- **Head 1** ($m = 1/2$): Strong distance penalty → very local attention
- **Head 8** ($m = 1/256$): Weak distance penalty → nearly global attention

This automatically creates a **multi-scale** position representation:

```
Head 1 (m=1/2):     Strong local focus
Attention: ████▓▓░░░░░░░░░░░░░░

Head 4 (m=1/16):    Medium range
Attention: ████████████▓▓▓░░░░░

Head 8 (m=1/256):   Nearly global
Attention: █████████████████████
```

### 7.3 Attention Score Matrix

For a 5-token sequence with slope $m$:

$$
\text{Bias} = -m \cdot \begin{bmatrix} 0 & 1 & 2 & 3 & 4 \\ 1 & 0 & 1 & 2 & 3 \\ 2 & 1 & 0 & 1 & 2 \\ 3 & 2 & 1 & 0 & 1 \\ 4 & 3 & 2 & 1 & 0 \end{bmatrix}
$$

(For causal attention, the upper triangle is masked to $-\infty$ anyway.)

### 7.4 Why ALiBi Extrapolates Well

ALiBi was designed specifically for **length extrapolation** — performing well on sequences longer than those seen during training.

**Why it works**: The linear bias $-m \cdot |i - j|$ is defined for any distance. Whether the distance is 100 or 100,000, the formula applies. There are no learned parameters that could fail on unseen distances.

The authors showed that models trained with ALiBi on 1024 tokens could perform well on sequences up to 2048+ tokens without any modification — a significant improvement over sinusoidal and learned positional encodings.

### 7.5 Properties

| Property | ALiBi |
|----------|-------|
| Additional parameters | **Zero** — slopes are fixed, not learned |
| Relative position | Yes — bias depends on $|i - j|$ |
| Extrapolation | Excellent — best of the non-scaling methods |
| Complexity | $O(1)$ per attention entry (just an addition) |
| Modifies | Attention scores only (not embeddings or Q/K) |

### 7.6 Limitations

1. **Hard linear assumption**: ALiBi assumes attention should decay linearly with distance. This may not always be optimal — some tasks require attending strongly to specific distant positions.

2. **No learned flexibility**: The slopes are fixed. The model can't learn task-specific position patterns.

3. **Performance gap**: At large scale, RoPE-based models (Llama) have generally outperformed ALiBi-based models (BLOOM, MPT), though it's difficult to isolate the positional encoding's contribution from other design choices.

### 7.7 Who Uses ALiBi

| Model | Positional Encoding |
|-------|-------------------|
| BLOOM (BigScience) | ALiBi |
| MPT (MosaicML) | ALiBi |
| BLOOM-176B | ALiBi |

Most models since 2023 have chosen RoPE over ALiBi.

---

## 8. Context Length Extension — The Frontier Problem

One of the most active research areas in LLM development: how to make a model work on sequences longer than its training length.

### 8.1 The Problem

A model trained with max length $L_{\text{train}}$ encounters sequences of length $L_{\text{test}} > L_{\text{train}}$. What happens?

**With learned absolute embeddings**: Complete failure. There's no embedding for position $> L_{\text{train}}$.

**With sinusoidal encodings**: Poor performance. The model hasn't learned to use those position values.

**With RoPE**: Degradation. The rotation frequencies were calibrated for $L_{\text{train}}$. At longer lengths, high-frequency dimensions rotate so fast they create meaningless patterns. Performance degrades significantly.

**With ALiBi**: Gradual degradation. Works better than others, but still imperfect.

### 8.2 Position Interpolation (PI)

Chen et al. (2023) proposed a simple idea: instead of **extrapolating** RoPE to new positions, **interpolate** by scaling down the position indices:

$$
\text{RoPE-PI}(\mathbf{x}, m) = \text{RoPE}\left(\mathbf{x}, m \cdot \frac{L_{\text{train}}}{L_{\text{target}}}\right)
$$

If trained on length 4096 and targeting length 16384, scale factor $s = 4096/16384 = 0.25$. Position 16384 becomes position 4096 in the model's coordinate system.

**Intuition**: The model has seen all rotation angles between $0$ and $L_{\text{train}} \cdot \theta$. By scaling, we ensure all positions map to this familiar range.

**Trade-off**: Nearby positions now have very similar rotation angles (they're squeezed together). This can hurt **local** resolution — the model struggles to distinguish adjacent tokens. Requires fine-tuning on a small amount of long-context data.

```
Original (L=4096):     |─────────────────────|
                       0                     4096
                       Positions 0 to 4096 fill this space

PI for L=16384:        |─────────────────────|
                       0                     4096
                       Positions 0 to 16384 squeezed into this space
                       → Local resolution reduced by 4×
```

### 8.3 NTK-Aware Interpolation

NTK-aware scaling (Reddit user "bloc97", 2023) addresses PI's weakness by scaling frequencies **non-uniformly**:

**Key insight**: High-frequency dimensions (which encode local position) shouldn't be compressed as much as low-frequency dimensions (which encode global position).

The base frequency is modified:

$$
\theta_i' = \theta_i \cdot \alpha^{-2i/(d-2)} = 10000^{-2i/d} \cdot \alpha^{-2i/(d-2)}
$$

where $\alpha$ is the scaling factor (e.g., 4 for 4× length extension).

This effectively:
- **Preserves** high-frequency components (local position resolution maintained)
- **Compresses** low-frequency components (which can afford it — they change slowly anyway)

```
Frequency:     High ◄──────────────────────────► Low

PI:            Compress ──── uniformly ──── Compress
               (hurts local)              (fine)

NTK-aware:     Preserve ────────────── Compress
               (local OK)              (global OK)
```

### 8.4 YaRN (Yet another RoPE extensioN)

YaRN (Peng et al., 2023) combines NTK-aware interpolation with a **temperature correction**:

$$
\text{Attention} = \text{softmax}\left(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k} \cdot t}\right)
$$

where $t > 1$ is a temperature that compensates for the entropy increase caused by longer sequences.

YaRN also divides dimensions into three groups:

1. **High frequency** (no scaling): These dimensions oscillate fast enough that extrapolation works fine
2. **Medium frequency** (NTK-interpolated): A smooth blend between no scaling and full interpolation
3. **Low frequency** (fully interpolated): These must be compressed to stay in the learned range

The boundary between groups is determined by the **wavelength** relative to the training length:
- If wavelength $\lambda < L_{\text{train}}$: The model has seen full cycles → extrapolation is safe
- If wavelength $\lambda > L_{\text{train}}$: The model hasn't seen full cycles → must interpolate

### 8.5 Dynamic NTK Scaling

Used in practice by many systems (e.g., together.ai, vLLM):

The scaling factor $\alpha$ is computed **dynamically** based on the current sequence length:

$$
\alpha = \max\left(1, \frac{L_{\text{current}}}{L_{\text{train}}}\right)
$$

- When $L_{\text{current}} \leq L_{\text{train}}$: No scaling ($\alpha = 1$), standard RoPE
- When $L_{\text{current}} > L_{\text{train}}$: Scale increases proportionally

This allows a single model to handle any length without pre-specifying the target length.

### 8.6 How Llama 3 Does It

Llama 3 uses a variant of RoPE scaling with a **modified frequency base**:

- Llama 2: Trained on 4K context, base $\theta = 10000$
- Llama 3: Trained on 8K context, extended to 128K using RoPE scaling with base $\theta = 500000$

The larger base frequency means lower-frequency rotation rates across all dimensions, naturally supporting longer sequences. Combined with continued pretraining on long documents, Llama 3 achieves robust 128K context.

### 8.7 Summary: Context Extension Methods

| Method | Approach | Fine-tuning needed? | Local resolution | Long-range |
|--------|----------|---------------------|-----------------|------------|
| **Position Interpolation** | Scale all positions uniformly | Yes (small amount) | Reduced | Good |
| **NTK-aware** | Scale frequencies non-uniformly | Yes (less) | Preserved | Good |
| **YaRN** | NTK + temperature + dimension grouping | Yes (minimal) | Preserved | Excellent |
| **Dynamic NTK** | NTK with adaptive scaling | No | Preserved | Good |
| **Increased base freq** | Change $\theta$ base + continued pretraining | Yes (significant) | Good | Excellent |

---

## 9. Comparison of All Methods

### 9.1 Feature Comparison

| Method | Type | Params | Relative Pos | Extrapolation | Used In |
|--------|------|--------|-------------|---------------|---------|
| **Sinusoidal** | Input additive | 0 | Implicit (via dot product) | Poor | Original Transformer |
| **Learned Absolute** | Input additive | $n_{\max} \times d$ | No | None | BERT, GPT-2/3 |
| **Shaw Relative** | Attention bias | $(2k+1) \times d_k$ | Yes (explicit) | Moderate | Various |
| **Transformer-XL** | Attention decomposition | Moderate | Yes | Moderate | Transformer-XL |
| **RoPE** | Q/K rotation | 0 | Yes (via rotation) | Moderate (extensible) | **Llama, Mistral, Qwen, Gemma, Phi** |
| **ALiBi** | Attention score bias | 0 | Yes (linear decay) | Good | BLOOM, MPT |
| **T5 Relative Bias** | Learned attention bias | Small (per head, per bucket) | Yes | Moderate | T5, Flan-T5 |

### 9.2 The Evolution Timeline

```
2017: Sinusoidal PE (Vaswani et al.)
       └─ Fixed, mathematical, elegant but limited

2018: Learned Absolute (BERT, GPT-2)
       └─ Flexible but can't extrapolate

2018: Shaw Relative PE
       └─ First successful relative approach

2019: Transformer-XL Relative PE
       └─ Complex decomposition, influenced later work

2020: T5 Relative Bias
       └─ Learned bucketized relative bias

2021: RoPE (Su et al.)
       └─ Elegant rotation-based, became dominant

2022: ALiBi (Press et al.)
       └─ Simplest approach, good extrapolation

2023: Position Interpolation, NTK-aware scaling, YaRN
       └─ Extending RoPE to longer contexts

2024: Large base frequency (Llama 3: θ=500K)
       └─ Native long context through scaling + continued pretraining
```

### 9.3 The Current Winner: RoPE

RoPE dominates modern LLMs for several reasons:

1. **Relative position naturally** — the core requirement
2. **Zero additional parameters** — all information from the formula
3. **Compatible with KV cache** — essential for efficient inference
4. **Extensible** — many proven methods to extend context (PI, NTK, YaRN)
5. **Simple, efficient implementation** — element-wise operations
6. **Empirically strong** — best performance in practice at scale

The main question in 2024-2026 is not *whether* to use RoPE, but *how to extend it* for longer contexts.

---

## 10. Positional Encodings in Vision and Beyond

### 10.1 Vision Transformers (ViT)

ViT patches an image into a grid and treats each patch as a "token." Position encodings tell the model where each patch is in the 2D image.

**Approaches**:
- **Learned 1D** (original ViT): Flatten the 2D grid to 1D and use learned absolute embeddings. Works surprisingly well — the model learns the 2D structure from data.
- **Learned 2D**: Separate embeddings for row and column positions.
- **RoPE-2D**: Apply RoPE with 2D rotations (used in some vision-language models).

### 10.2 Audio and Speech

For audio spectrograms (time-frequency representation):
- Sinusoidal along the time axis
- Sometimes 2D positional encodings (time + frequency)

### 10.3 Multimodal Models

Models like LLaVA or GPT-4V must handle both text tokens and image patches in the same sequence. Positional encodings must:
- Give text tokens sequential positions
- Give image patches 2D spatial positions
- Allow the model to understand the relationship between text and image positions

Common approach: use the text model's positional encoding (e.g., RoPE) for all tokens, with image patches occupying sequential positions in the text stream.

---

## 11. Interview Questions & Answers

### Q1: Why can't transformers understand word order without positional encoding?

**A**: Self-attention is **permutation equivariant**. The attention score between any two tokens depends only on their *content* (the dot product of their projected representations), not on *where* they appear in the sequence. If you permute the input tokens, the attention outputs are simply permuted — the model cannot detect that any reordering occurred.

Mathematically: let $\pi$ be any permutation. Then $\text{Attn}(\pi(\mathbf{X})) = \pi(\text{Attn}(\mathbf{X}))$. The model treats the input as an **unordered set**, not a sequence. Since natural language is fundamentally ordered ("dog bites man" ≠ "man bites dog"), we must inject position information.

There are three strategies: (1) add positional vectors to the input embeddings (sinusoidal, learned), (2) modify the attention scores with position-dependent biases (ALiBi, T5 bias), or (3) rotate Q/K vectors by position-dependent angles (RoPE). Modern LLMs predominantly use RoPE.

---

### Q2: What is the key insight behind RoPE? How does it encode relative position?

**A**: RoPE's key insight is that **rotation in 2D encodes relative position through the dot product**.

If we rotate query $\mathbf{q}$ at position $m$ by angle $m\theta$ and key $\mathbf{k}$ at position $n$ by angle $n\theta$, their dot product:

$$
(\mathbf{R}_m\mathbf{q})^T(\mathbf{R}_n\mathbf{k}) = \mathbf{q}^T\mathbf{R}_m^T\mathbf{R}_n\mathbf{k} = \mathbf{q}^T\mathbf{R}_{n-m}\mathbf{k}
$$

depends only on the **relative position** $n - m$, because $\mathbf{R}_m^T\mathbf{R}_n = \mathbf{R}_{n-m}$ (rotation matrices compose by adding angles).

For $d$-dimensional vectors, RoPE pairs dimensions $(2i, 2i+1)$ and applies independent 2D rotations with frequencies $\theta_i = 10000^{-2i/d}$, creating a multi-scale relative position representation.

**Why RoPE is preferred over alternatives**:
- Relative position emerges naturally from the math (no explicit bias terms)
- No additional parameters (unlike learned approaches)
- Multiplicative, not additive — doesn't interfere with content information
- Compatible with KV cache (rotations are position-independent)
- Extensible to longer contexts via PI, NTK-aware scaling, or YaRN

---

### Q3: Compare sinusoidal, learned, RoPE, and ALiBi. Trade-offs of each?

**A**:

| | Sinusoidal | Learned Absolute | RoPE | ALiBi |
|--|-----------|-----------------|------|-------|
| **How** | Add fixed sin/cos to input | Add learned vectors to input | Rotate Q, K by position | Subtract linear distance from attention scores |
| **Params** | 0 | $n_{\max} \times d$ | 0 | 0 |
| **Relative position** | Implicit (via dot product) | No | Yes (via rotation) | Yes (linear decay) |
| **Extrapolation** | Poor | Impossible (hard cutoff) | Moderate (extensible) | Good |
| **Local resolution** | Good | Good | Good | Fixed by slope |
| **Flexibility** | None (fixed formula) | High (fully learned) | Moderate | Low (fixed linear bias) |
| **Content-position separation** | Poor (additive mixing) | Poor (additive mixing) | Good (multiplicative) | Perfect (separate bias) |
| **Used today** | Legacy | BERT, some smaller models | **Dominant** (Llama, Mistral, etc.) | BLOOM, MPT |

**When to choose each**:
- **Sinusoidal**: Historical interest only; no reason to use in new models
- **Learned absolute**: When max length is fixed and known (e.g., BERT-style models for classification)
- **RoPE**: Default choice for any new LLM — proven at scale, extensible
- **ALiBi**: When you need out-of-the-box length extrapolation without fine-tuning

---

### Q4: How do models extend their context length beyond what they were trained on?

**A**: The main approaches for extending RoPE-based models:

**1. Position Interpolation (PI)**: Scale all position indices by $L_{\text{train}} / L_{\text{target}}$ so they fit within the trained range. Simple but reduces local resolution (nearby tokens get very similar positions). Requires minimal fine-tuning.

**2. NTK-aware scaling**: Scale frequencies non-uniformly — high-frequency (local) dimensions are preserved while low-frequency (global) dimensions are compressed. This maintains local resolution while extending range. The base $\theta$ is effectively increased by a factor $\alpha$.

**3. YaRN**: Combines NTK-aware scaling with temperature correction and dimension-wise grouping. Divides dimensions into high-frequency (no scaling), medium (blended), and low-frequency (fully interpolated). Currently the best quality for post-hoc extension.

**4. Increased base frequency + continued pretraining**: Change the base from $\theta = 10000$ to $\theta = 500000$ (Llama 3's approach). This natively spaces out all rotation frequencies for longer sequences, but requires significant continued pretraining on long documents.

**Key insight**: The fundamental problem is that high-frequency RoPE dimensions rotate so fast that they "wrap around" multiple times at long distances, creating ambiguous position signals. All extension methods address this by either compressing positions into the familiar range (interpolation) or slowing down the rotations (frequency scaling).

---

### Q5: Why does RoPE not apply rotations to the value vectors?

**A**: RoPE is applied only to queries and keys, not values. The reasoning is based on what each component does:

- **Q and K** determine *which* tokens to attend to. This is where position matters — the model should know relative position when computing attention weights.
- **V** determines *what information* to retrieve from each position. Once we know *which* tokens are relevant (via position-aware attention weights), the actual content we extract should not be position-dependent.

Mathematically, if we applied RoPE to V:

$$
\text{output}_i = \sum_j \alpha_{ij} \mathbf{R}_j \mathbf{v}_j
$$

The position-dependent rotation on $\mathbf{v}_j$ would distort the content information. The output at position $i$ would depend on the absolute position of attended tokens, not just their content — which is undesirable.

Additionally, not applying RoPE to V has a practical benefit: the value vectors in the KV cache don't need positional information, which simplifies implementation.

---

### Q6: Explain the dot product property of sinusoidal encodings that makes them encode relative position.

**A**: For sinusoidal encodings, the dot product between two position vectors depends only on their distance:

$$
PE(pos_1)^T \cdot PE(pos_2) = \sum_{i=0}^{d/2-1} \left[\sin(\omega_i \cdot pos_1)\sin(\omega_i \cdot pos_2) + \cos(\omega_i \cdot pos_1)\cos(\omega_i \cdot pos_2)\right]
$$

Using the product-to-sum identity $\cos(A - B) = \cos A \cos B + \sin A \sin B$:

$$
PE(pos_1)^T \cdot PE(pos_2) = \sum_{i=0}^{d/2-1} \cos(\omega_i \cdot (pos_1 - pos_2))
$$

This depends only on $pos_1 - pos_2$, not on the absolute positions.

Furthermore, this connection to rotations is exactly what inspired RoPE: the sinusoidal identity $\sin(A+B) = \sin A \cos B + \cos A \sin B$ is the rotation formula. RoPE essentially takes this implicit rotational structure in sinusoidal encodings and makes it explicit by directly rotating Q and K vectors.

---

### Q7: What is the relationship between RoPE's base frequency and context length?

**A**: RoPE's frequencies are $\theta_i = \text{base}^{-2i/d}$, where $\text{base} = 10000$ in the original formulation.

The **wavelength** (number of positions for a full 360° rotation) of dimension pair $i$ is:

$$
\lambda_i = 2\pi \cdot \text{base}^{2i/d}
$$

- Fastest dimension ($i = 0$): $\lambda_0 = 2\pi \approx 6.3$ positions
- Slowest dimension ($i = d/2 - 1$): $\lambda_{\max} = 2\pi \cdot \text{base} \approx 62{,}832$ positions

For the model to reliably distinguish positions, it needs at least one dimension whose wavelength is longer than the sequence length. With $\text{base} = 10000$, the maximum wavelength is $\approx 62K$ — sufficient for training lengths up to ~4K (the model needs margin).

**To support longer contexts**, increase the base:
- Llama 2 ($\text{base} = 10000$): 4K context
- Llama 3 ($\text{base} = 500000$): 128K context
- CodeLlama ($\text{base} = 1000000$): 100K context

With $\text{base} = 500000$: $\lambda_{\max} = 2\pi \times 500000 \approx 3.14M$ positions — far more than 128K.

Increasing the base **slows down all rotations**, giving the model more room before positions become ambiguous. The trade-off is potentially reduced local resolution (nearby positions have more similar rotations), but in practice the fast dimensions still oscillate quickly enough.

---

### Q8: If you were building a new LLM, how would you choose the positional encoding?

**A**: In 2024-2026, the answer is almost certainly **RoPE** with appropriate base frequency for the target context length. Here's the decision process:

**Step 1: Choose RoPE** (unless you have a specific reason not to). It's proven at scale in Llama, Mistral, Qwen, Gemma, Phi, and virtually every other modern LLM.

**Step 2: Set the base frequency** based on target context length:
- Target ≤ 8K: $\theta = 10000$ (standard)
- Target 16K–32K: $\theta = 50000$–$100000$
- Target 128K+: $\theta = 500000$+ (Llama 3 approach)

**Step 3: Plan for extension**. Even if you train on 8K, design for future extension:
- Use RoPE (naturally extensible)
- Reserve compute budget for continued pretraining on long data
- Have NTK/YaRN scaling ready as a quick fix

**When you might choose differently**:
- **ALiBi**: If zero-shot length extrapolation (no fine-tuning) is critical and you can't afford continued pretraining
- **Learned absolute**: For a BERT-like encoder model that will never exceed its training length (classification, embeddings)
- **No positional encoding**: Some architectures experiment with learning position implicitly through causal masking (the mask itself encodes position for decoder-only models — position $i$ can attend to exactly $i$ tokens, uniquely identifying it). This remains experimental.

---

### Q9: What happens at the boundary when a model sees sequences longer than its training length?

**A**: The failure mode depends on the positional encoding:

**Learned absolute**: Catastrophic failure. Position indices beyond $n_{\max}$ have no embedding. The model crashes or produces garbage.

**Sinusoidal**: The math produces valid values, but the model has never learned to use them. Performance degrades rapidly — perplexity shoots up, coherence drops. The model doesn't understand what "position 5000" means if it only trained on length 2048.

**RoPE (without extension)**: Degradation is gradual but significant. High-frequency dimension pairs "wrap around" — position 4097 has the same fast-rotating components as position 1. This creates **aliasing**: the model confuses distant positions with nearby positions. Mid-frequency dimensions provide some differentiation, but overall quality drops sharply beyond ~1.5× the training length.

**ALiBi**: Graceful degradation. The linear bias $-m|i-j|$ is defined for any distance. The main issue is that at very long distances, the bias becomes very large and negative, effectively masking all distant tokens. This limits practical extrapolation to ~2× training length without modifications.

**RoPE with scaling (PI/NTK/YaRN)**: After fine-tuning, these methods work well at extended lengths. Without fine-tuning, NTK-aware and dynamic scaling provide reasonable performance at ~2-4× training length, with some quality drop.

---

### Q10: Explain the T5 relative position bias. How does it differ from other relative approaches?

**A**: T5 uses **learned relative position biases** added directly to attention scores, but with a twist: positions are **bucketed logarithmically** to reduce the number of learnable parameters.

The bias for relative position $\delta = i - j$ is:

$$
\text{score}_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + b_{h}[\text{bucket}(\delta)]
$$

where $b_h$ is a learned bias table per head $h$, and $\text{bucket}(\cdot)$ maps relative positions to a fixed number of buckets (32 by default).

**The bucketing scheme**:
- Small distances ($|\delta| \leq 8$): Each distance gets its own bucket (fine-grained)
- Larger distances: Logarithmic bucketing — distances 9-16 share a bucket, 17-32 share another, etc.

```
Distance:  0  1  2  3  4  5  6  7  8  9-16  17-32  33-64  65-128
Bucket:    0  1  2  3  4  5  6  7  8    9     10     11     12
```

**How it differs from other approaches**:
- **vs Shaw**: Simpler — just a scalar bias per bucket per head, not a vector
- **vs RoPE**: Less elegant mathematically, but explicitly learned
- **vs ALiBi**: Learned (more flexible) rather than fixed linear

**Limitation**: The number of buckets is fixed. Beyond the maximum bucket distance (~128), all positions are treated identically. This limits extrapolation.

T5's approach influenced later work but was ultimately superseded by RoPE's parameter-free approach.

---

*This topic completes Phase C: The Transformer Revolution. You now have the complete transformer picture — attention (Topic 8), architecture (Topic 9), and position (Topic 10). Next: [Topic 11: Encoder Models (BERT Family)](11_BERT_Family.md) — beginning Phase D: Language Models.*
