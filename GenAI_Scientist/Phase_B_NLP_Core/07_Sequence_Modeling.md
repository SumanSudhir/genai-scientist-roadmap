# Topic 7: Sequence Modeling (RNNs, LSTMs, Seq2Seq)

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 7 of 28
> **Scope**: Vanilla RNNs, vanishing/exploding gradients, LSTMs, GRUs, bidirectional RNNs, Seq2Seq, teacher forcing, exposure bias, the information bottleneck
> **Why this matters**: Understanding the problems that existed before transformers helps you appreciate WHY transformers work. Interviewers love "trace the evolution" questions. The information bottleneck problem in Seq2Seq directly motivated the attention mechanism — the single most important idea in modern AI.
> **Prev**: [Topic 6: Text Preprocessing & Embeddings](06_Text_Preprocessing_Embeddings.md)
> **Next**: [Topic 8: Attention Mechanisms](../Phase_C_Transformers/08_Attention_Mechanisms.md)

---

## Table of Contents

1. [Why Sequence Modeling?](#1-why-sequence-modeling)
2. [Vanilla RNNs](#2-vanilla-rnns)
3. [Backpropagation Through Time (BPTT)](#3-backpropagation-through-time-bptt)
4. [Vanishing and Exploding Gradients](#4-vanishing-and-exploding-gradients)
5. [Long Short-Term Memory (LSTM)](#5-long-short-term-memory-lstm)
6. [Gated Recurrent Unit (GRU)](#6-gated-recurrent-unit-gru)
7. [Bidirectional RNNs](#7-bidirectional-rnns)
8. [Sequence-to-Sequence (Seq2Seq)](#8-sequence-to-sequence-seq2seq)
9. [Teacher Forcing & Exposure Bias](#9-teacher-forcing--exposure-bias)
10. [The Information Bottleneck & the Road to Attention](#10-the-information-bottleneck--the-road-to-attention)
11. [Why Transformers Replaced RNNs](#11-why-transformers-replaced-rnns)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. Why Sequence Modeling?

Language is inherently sequential. The meaning of a word depends on what came before (and after) it:

- "The bank was flooded" — river bank (context: flooding)
- "The bank was closed" — financial bank (context: business hours)

Unlike images (where pixels can be processed in any order), text has **temporal structure** — word order matters. "Dog bites man" vs "Man bites dog" have the same words but completely different meanings.

**Sequence modeling** is the problem of building models that can process ordered sequences and maintain memory of what they've seen. Before transformers (2017), the dominant approach was **recurrent neural networks (RNNs)** and their variants.

**Historical timeline**:
- **1986**: RNNs introduced (Rumelhart, Hinton, Williams)
- **1990**: Vanishing gradient problem identified (Bengio, Simard, Frasconi)
- **1997**: LSTM introduced (Hochreiter & Schmidhuber)
- **2014**: GRU introduced (Cho et al.), Seq2Seq breakthrough (Sutskever et al.)
- **2014-2015**: Attention mechanism added to Seq2Seq (Bahdanau, Luong)
- **2017**: Transformer — "Attention Is All You Need" — RNNs become obsolete for most tasks

Understanding this arc from RNN → LSTM → Seq2Seq → Attention → Transformer is essential interview material.

---

## 2. Vanilla RNNs

### 2.1 Architecture

An RNN processes a sequence $(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T)$ one element at a time, maintaining a **hidden state** $\mathbf{h}_t$ that serves as the model's "memory":

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

where:
- $\mathbf{x}_t \in \mathbb{R}^d$ is the input at time step $t$
- $\mathbf{h}_t \in \mathbb{R}^h$ is the hidden state at time $t$ (the "memory")
- $\mathbf{y}_t$ is the output at time $t$
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ is the recurrent weight matrix (hidden-to-hidden)
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$ is the input weight matrix (input-to-hidden)
- $\mathbf{W}_{hy} \in \mathbb{R}^{o \times h}$ is the output weight matrix (hidden-to-output)

**Key insight**: The same weights $(\mathbf{W}_{hh}, \mathbf{W}_{xh})$ are shared across all time steps. This is **parameter sharing** — the network has the same number of parameters regardless of sequence length.

### 2.2 Unrolling Through Time

An RNN can be "unrolled" into a deep feedforward network where each time step is a layer:

```
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ... → xₜ → [hₜ]
         ↓           ↓           ↓                    ↓
        y₁          y₂          y₃                   yₜ
```

This unrolled view makes it clear that:
1. An RNN processing a sequence of length $T$ is equivalent to a $T$-layer deep network
2. All layers share the same weights (weight tying)
3. Gradients must flow through $T$ layers during backpropagation — this is where problems arise

### 2.3 Types of RNN Architectures

| Pattern | Input → Output | Example |
|---------|---------------|---------|
| One-to-one | Single → Single | Standard feedforward (not really RNN) |
| One-to-many | Single → Sequence | Image captioning |
| Many-to-one | Sequence → Single | Sentiment classification |
| Many-to-many (aligned) | Sequence → Same-length sequence | POS tagging, NER |
| Many-to-many (unaligned) | Sequence → Different-length sequence | Translation (Seq2Seq) |

### 2.4 What the Hidden State Captures

The hidden state $\mathbf{h}_t$ is a **compressed representation of the entire sequence up to time $t$**. It must encode everything relevant from $(\mathbf{x}_1, ..., \mathbf{x}_t)$ into a fixed-size vector of dimension $h$.

**Problem**: As sequences get longer, the hidden state must compress more and more information into the same fixed-size vector. Information from early time steps gets progressively overwritten by newer information. This is a **lossy compression** that worsens with sequence length.

---

## 3. Backpropagation Through Time (BPTT)

### 3.1 The Algorithm

BPTT is simply backpropagation applied to the unrolled RNN. The total loss over a sequence is:

$$\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$

The gradient of $\mathcal{L}$ with respect to the shared weight matrix $\mathbf{W}_{hh}$ sums contributions from all time steps:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$

For each time step $t$, the gradient must flow back through all preceding hidden states:

$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

The critical term is $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}$, which involves a chain of Jacobians:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}}$$

### 3.2 The Jacobian Chain

Each factor in the product is:

$$\frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \text{diag}(\tanh'(\mathbf{z}_i)) \cdot \mathbf{W}_{hh}$$

where $\mathbf{z}_i = \mathbf{W}_{hh}\mathbf{h}_{i-1} + \mathbf{W}_{xh}\mathbf{x}_i + \mathbf{b}_h$ and $\tanh'(z) = 1 - \tanh^2(z) \in (0, 1]$.

So the full Jacobian chain is:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \text{diag}(\tanh'(\mathbf{z}_i)) \cdot \mathbf{W}_{hh}$$

This is a product of $(t - k)$ matrices. The behavior of this product — whether it shrinks, grows, or stays stable — determines whether RNNs can learn long-range dependencies.

### 3.3 Truncated BPTT

**Full BPTT** requires storing all $T$ hidden states and propagating gradients through all $T$ steps. For long sequences ($T$ = 1000+), this is both memory-intensive and prone to gradient issues.

**Truncated BPTT**: Only backpropagate through the last $\tau$ steps (typically $\tau$ = 20-50):

$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}} \approx \sum_{k=t-\tau}^{t} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \cdot \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

This reduces memory from $O(T)$ to $O(\tau)$ and limits gradient instability, but it means the model **cannot learn dependencies longer than $\tau$ steps** — a fundamental limitation.

---

## 4. Vanishing and Exploding Gradients

### 4.1 The Mathematical Root Cause

From the Jacobian chain:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \text{diag}(\tanh'(\mathbf{z}_i)) \cdot \mathbf{W}_{hh}$$

The norm of this product is bounded by:

$$\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\right\| \leq \prod_{i=k+1}^{t} \|\text{diag}(\tanh'(\mathbf{z}_i))\| \cdot \|\mathbf{W}_{hh}\|$$

Since $\tanh'(z) \leq 1$, we have $\|\text{diag}(\tanh'(\mathbf{z}_i))\| \leq 1$.

Let $\lambda_{\max}$ be the largest singular value of $\mathbf{W}_{hh}$:

$$\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\right\| \leq \lambda_{\max}^{t-k}$$

### 4.2 Vanishing Gradients ($\lambda_{\max} < 1$)

If $\lambda_{\max} < 1$, then $\lambda_{\max}^{t-k} \to 0$ exponentially as the gap $(t - k)$ grows.

**Consequence**: Gradients from the loss at time $t$ cannot effectively reach hidden states at time $k$ when $(t - k)$ is large. The model **cannot learn long-range dependencies**.

**Example**: In a sequence of 100 words, if the model needs to connect the subject at position 5 to a verb at position 95, the gradient from position 95 to position 5 passes through 90 Jacobian multiplications. With $\lambda_{\max} = 0.9$: $0.9^{90} \approx 10^{-4}$. The gradient is nearly zero — the model cannot learn this dependency.

**Practical effect**: The model behaves as if it only has a short-term memory of 10-20 steps, regardless of sequence length. Information from the beginning of a paragraph is effectively invisible when processing the end.

### 4.3 Exploding Gradients ($\lambda_{\max} > 1$)

If $\lambda_{\max} > 1$, then $\lambda_{\max}^{t-k} \to \infty$ exponentially.

**Consequence**: Gradients grow to enormous values, causing weight updates that are so large they destabilize the entire network. Weights oscillate wildly, loss becomes NaN.

**Easier to fix than vanishing**: Gradient clipping provides a simple solution:

$$\mathbf{g} \leftarrow \begin{cases} \frac{g_{\max}}{\|\mathbf{g}\|} \cdot \mathbf{g} & \text{if } \|\mathbf{g}\| > g_{\max} \\ \mathbf{g} & \text{otherwise} \end{cases}$$

This caps the gradient norm at $g_{\max}$ (typically 1.0 or 5.0) without changing gradient direction. Used universally in RNN and transformer training.

### 4.4 The Asymmetry

Vanishing gradients are **much harder** to fix than exploding gradients:

- **Exploding**: Gradient clipping is simple and effective. The gradient still carries directional information even when clipped.
- **Vanishing**: You can't "amplify" vanishing gradients without introducing noise. The signal-to-noise ratio degrades. Multiplying small gradients by a constant just increases noise proportionally.

This asymmetry motivated the development of gated architectures (LSTM, GRU) that provide controlled pathways for gradient flow.

---

## 5. Long Short-Term Memory (LSTM)

### 5.1 The Core Innovation: The Cell State

**Key idea** (Hochreiter & Schmidhuber, 1997): Add a separate **cell state** $\mathbf{c}_t$ that runs through the entire sequence with **additive** updates (not multiplicative). Additive updates allow gradients to flow unchanged across many time steps — the same principle as residual connections in transformers.

The cell state is the LSTM's "long-term memory". Three **gates** control what information enters, stays in, and leaves the cell state.

### 5.2 The Four Components

An LSTM at time step $t$ computes:

**1. Forget Gate** — What to erase from memory:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

Output is in $(0, 1)^h$. Values near 0 mean "forget this component of the cell state". Values near 1 mean "keep it".

**2. Input Gate** — What new information to write to memory:

$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**3. Candidate Cell State** — The new information to potentially add:

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**4. Output Gate** — What to expose as the hidden state:

$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

### 5.3 State Updates

**Cell state update** (the key equation):

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

This is a **gated additive update**:
- $\mathbf{f}_t \odot \mathbf{c}_{t-1}$: Selectively keep old memory
- $\mathbf{i}_t \odot \tilde{\mathbf{c}}_t$: Selectively add new information

**Hidden state** (output):

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

The output gate controls how much of the cell state is exposed. The tanh squashes the cell state to $(-1, 1)$ before gating.

### 5.4 Why the Cell State Solves Vanishing Gradients

Consider the gradient of $\mathbf{c}_t$ with respect to $\mathbf{c}_{t-1}$:

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t) + \text{(terms involving } \frac{\partial \mathbf{f}_t}{\partial \mathbf{c}_{t-1}} \text{, etc.)}$$

The dominant term is $\text{diag}(\mathbf{f}_t)$. If the forget gate is close to 1 (which it learns to be for important information), then:

$$\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_k} \approx \prod_{t=k+1}^{T} \text{diag}(\mathbf{f}_t)$$

When $\mathbf{f}_t \approx 1$, this product stays close to the identity matrix, regardless of how many steps $(T - k)$ separate them. **Gradients flow through the cell state nearly unchanged.**

Compare to the vanilla RNN:
- **Vanilla RNN**: $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(\tanh'(\cdot)) \cdot \mathbf{W}_{hh}$ — multiplicative, prone to vanishing
- **LSTM cell state**: $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \text{diag}(\mathbf{f}_t)$ — additive path, gradient preserved

This is exactly analogous to **residual connections** in transformers: provide a direct, additive path for gradient flow that bypasses the non-linear transformations.

### 5.5 Gate Behaviors: Intuition

**Example**: Processing "The cat, which had been sitting on the warm windowsill all afternoon watching the birds, **jumped**."

- **Forget gate**: After reading "jumped", forget the clause about sitting/watching (no longer relevant for predicting the next word)
- **Input gate**: Write "jumped" (the main verb, grammatically crucial) into memory with high input gate value
- **Output gate**: Expose the fact that the subject ("cat") performed an action ("jumped") for the next prediction

The LSTM learns these gating patterns from data. It learns that opening parenthetical clauses should not erase the subject from memory, that main verbs should be strongly stored, etc.

### 5.6 LSTM Variants

**Peephole connections** (Gers & Schmidhuber, 2000): Let gates also look at the cell state:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t, \mathbf{c}_{t-1}] + \mathbf{b}_f)$$

Gives gates more information but adds parameters. Mixed results empirically.

**Coupled forget-input gate**: Constrain $\mathbf{i}_t = 1 - \mathbf{f}_t$, so the model can only add new information by forgetting old information. Reduces parameters with minimal quality loss.

### 5.7 LSTM Parameter Count

For hidden size $h$ and input size $d$:
- Each gate: $(h + d) \times h$ weights + $h$ biases
- 4 gates total: $4 \times [(h + d) \times h + h] = 4h(h + d) + 4h = 4h(h + d + 1)$

For $h = 512$, $d = 300$: $4 \times 512 \times (512 + 300 + 1) = 1,664,512$ parameters per layer.

---

## 6. Gated Recurrent Unit (GRU)

### 6.1 Simplification of LSTM

**GRU** (Cho et al., 2014): Combines the forget and input gates into a single **update gate**, and merges the cell state and hidden state. Fewer parameters, similar performance.

### 6.2 GRU Equations

**Update gate** (how much of the old state to keep):

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Reset gate** (how much of the old state to use when computing the candidate):

$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Candidate hidden state**:

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden state update**:

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### 6.3 Key Differences from LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| State vectors | 2 (cell state $\mathbf{c}_t$ + hidden state $\mathbf{h}_t$) | 1 (hidden state $\mathbf{h}_t$ only) |
| Parameters | $4h(h + d + 1)$ | $3h(h + d + 1)$ (25% fewer) |
| Forget-input coupling | Independent | Coupled: $\mathbf{i}_t = \mathbf{z}_t$, $\mathbf{f}_t = 1 - \mathbf{z}_t$ |
| Output gating | Separate output gate | No output gate — full state is exposed |

### 6.4 GRU Update Gate as Interpolation

The update equation $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$ is a **linear interpolation** between the old hidden state and the candidate:

- $\mathbf{z}_t = 0$: Keep old state completely (ignore new input)
- $\mathbf{z}_t = 1$: Replace entirely with candidate (forget old state)
- $0 < \mathbf{z}_t < 1$: Blend old and new

This interpolation is equivalent to an LSTM with coupled forget-input gates.

### 6.5 Reset Gate Intuition

When $\mathbf{r}_t \approx 0$: The candidate computation ignores the previous hidden state entirely: $\tilde{\mathbf{h}}_t \approx \tanh(\mathbf{W}_h [\mathbf{0}, \mathbf{x}_t])$. This allows the model to "forget" the past and compute a fresh state from the current input alone — useful at sentence boundaries or topic changes.

When $\mathbf{r}_t \approx 1$: The full previous hidden state flows into the candidate computation, enabling smooth continuation of existing state.

### 6.6 LSTM vs GRU: When to Choose

| Scenario | Prefer |
|----------|--------|
| Long sequences (>500 steps) | LSTM (separate cell state is slightly better for very long-range) |
| Limited compute/data | GRU (fewer parameters, faster training) |
| Real-time applications | GRU (fewer operations per step) |
| Very complex dependencies | LSTM (more expressive gating) |
| In practice (2014-2017 era) | Performance is usually comparable — try both |

**Empirical findings** (Chung et al., 2014; Greff et al., 2015): Across many tasks, LSTM and GRU perform comparably. The forget gate is the most critical component of LSTM — ablating other components matters less. The key innovation is gated additive state updates, not the specific gating architecture.

---

## 7. Bidirectional RNNs

### 7.1 Motivation

A standard (unidirectional) RNN processes sequences left-to-right. The hidden state $\mathbf{h}_t$ captures context from $(\mathbf{x}_1, ..., \mathbf{x}_t)$ — only the **past**.

But for many NLP tasks, **future context matters**:
- "He deposited money in the **bank**" — knowing "money" comes before helps
- "He sat on the river **bank**" — knowing "river" comes before helps
- But what if the disambiguating word comes *after*? "The **bank** was flooded by the river" — "river" comes after "bank"

### 7.2 Architecture

A bidirectional RNN runs **two separate RNNs** over the sequence:

**Forward RNN** (left to right):

$$\overrightarrow{\mathbf{h}}_t = f(\overrightarrow{\mathbf{h}}_{t-1}, \mathbf{x}_t)$$

**Backward RNN** (right to left):

$$\overleftarrow{\mathbf{h}}_t = f(\overleftarrow{\mathbf{h}}_{t+1}, \mathbf{x}_t)$$

**Combined representation**:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2h}$$

The concatenation captures both past context (from forward RNN) and future context (from backward RNN).

### 7.3 When Bidirectional Works (and When It Doesn't)

**Works well for understanding tasks** where the full sequence is available at once:
- Named Entity Recognition (NER)
- Part-of-Speech tagging
- Sentiment classification
- Question answering (given the full passage)
- BERT's architecture is essentially a bidirectional model (via masked language modeling)

**Does NOT work for generation tasks** where you must predict one token at a time:
- Language modeling (can't look at future tokens you haven't generated yet)
- Machine translation (at decode time, future target tokens don't exist)
- Any autoregressive generation

**Connection to transformers**: BERT uses bidirectional attention (every token attends to every other). GPT uses causal (unidirectional) attention (each token only attends to previous tokens). This is the fundamental architectural difference between encoder models (understanding) and decoder models (generation).

### 7.4 Deep Bidirectional RNNs

Stack multiple bidirectional layers. Layer $l$ takes the output of layer $l-1$ as input:

$$\overrightarrow{\mathbf{h}}_t^{(l)} = f(\overrightarrow{\mathbf{h}}_{t-1}^{(l)}, \mathbf{h}_t^{(l-1)})$$

where $\mathbf{h}_t^{(l-1)} = [\overrightarrow{\mathbf{h}}_t^{(l-1)}; \overleftarrow{\mathbf{h}}_t^{(l-1)}]$.

Typical depth: 2-4 layers for RNNs (deeper RNNs are hard to train without residual connections). Compare to transformers: 12-96 layers (enabled by residual connections and layer normalization).

---

## 8. Sequence-to-Sequence (Seq2Seq)

### 8.1 The Breakthrough

**Seq2Seq** (Sutskever et al., 2014; Cho et al., 2014) enabled mapping **variable-length input sequences to variable-length output sequences** — the architecture that made neural machine translation practical.

Before Seq2Seq, NMT required word alignment models, phrase tables, and language models combined in complex pipelines. Seq2Seq replaced all of this with a single end-to-end neural network.

### 8.2 Architecture: Encoder-Decoder

**Encoder**: Reads the input sequence and compresses it into a fixed-size vector.

$$\mathbf{h}_t^{enc} = \text{LSTM}_{enc}(\mathbf{h}_{t-1}^{enc}, \mathbf{x}_t)$$

The final hidden state $\mathbf{h}_T^{enc}$ (and cell state $\mathbf{c}_T^{enc}$ for LSTM) is the **context vector** — a fixed-size summary of the entire input.

**Decoder**: Generates the output sequence one token at a time, starting from the context vector.

$$\mathbf{h}_t^{dec} = \text{LSTM}_{dec}(\mathbf{h}_{t-1}^{dec}, \mathbf{y}_{t-1})$$

$$P(\mathbf{y}_t \mid \mathbf{y}_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{h}_t^{dec})$$

The decoder is initialized with the encoder's final state: $\mathbf{h}_0^{dec} = \mathbf{h}_T^{enc}$ (and $\mathbf{c}_0^{dec} = \mathbf{c}_T^{enc}$).

### 8.3 Training

The model is trained to maximize the likelihood of the correct output sequence:

$$\mathcal{L} = -\sum_{t=1}^{T'} \log P(y_t^* \mid y_1^*, ..., y_{t-1}^*, \mathbf{x})$$

where $y_t^*$ is the ground-truth token at step $t$ and $T'$ is the target sequence length.

### 8.4 Inference: Greedy vs Beam Search

**Greedy decoding**: At each step, pick the token with highest probability:

$$y_t = \arg\max_{y} P(y \mid y_{1:t-1}, \mathbf{x})$$

Fast but suboptimal — a locally best choice can lead to a globally bad sequence.

**Beam search**: Maintain $B$ (beam width) candidates at each step. At each position, expand each candidate by all possible next tokens, score them, and keep the top $B$.

- Beam width $B = 1$: Greedy decoding
- Beam width $B = 5$: Standard for machine translation
- Larger $B$: Better search but diminishing returns and slower

**Beam search score** (with length normalization):

$$\text{score}(y_{1:t}) = \frac{1}{t^\alpha} \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, \mathbf{x})$$

where $\alpha$ (typically 0.6-0.7) prevents beam search from preferring shorter sequences.

### 8.5 Practical Tricks from the Original Paper

Sutskever et al. (2014) found several crucial tricks:

1. **Reverse the input sequence**: Feeding "ABC" as "CBA" so the decoder starts generating from the part of the input closest to the last encoder hidden state. This reduced the distance between corresponding input-output words, significantly improving performance.

2. **Deep LSTMs**: 4-layer LSTMs for both encoder and decoder.

3. **Ensemble**: Average predictions from 5 independently trained models.

These tricks were necessary band-aids for a fundamental limitation: the information bottleneck.

---

## 9. Teacher Forcing & Exposure Bias

### 9.1 Teacher Forcing

**During training**, the decoder receives the **ground-truth** previous token as input at each step:

$$\mathbf{h}_t^{dec} = \text{LSTM}_{dec}(\mathbf{h}_{t-1}^{dec}, \mathbf{y}_{t-1}^*)$$

This is **teacher forcing** — the "teacher" (ground truth) forces the correct input at each step, regardless of what the model would have predicted.

**Why teacher forcing is used**:
1. **Faster convergence**: The model always gets correct context, so learning is more stable
2. **Parallelizable within a sequence**: Each step's input is known in advance (ground truth), allowing some computational tricks
3. **Avoids compounding errors during training**: Without teacher forcing, an early mistake cascades through the entire sequence

### 9.2 Exposure Bias

**The problem**: During training, the model always sees ground-truth previous tokens. During inference, it sees its own (potentially incorrect) predictions. This train-test mismatch is **exposure bias**.

**Example**: Translating "The cat sat on the mat":
- **Training**: Each decoder step gets the correct previous word. The model never learns to recover from mistakes.
- **Inference**: If the model predicts "The dog..." at step 2 (wrong!), it has never encountered this situation during training. It may produce increasingly nonsensical output.

**Exposure bias causes**:
1. **Error accumulation**: Small errors compound — each mistake makes subsequent predictions worse
2. **Degenerate outputs**: The model enters states never seen during training and produces garbage
3. **Brittleness**: The model performs well on data similar to training but fails on slightly different inputs

### 9.3 Mitigations

**Scheduled sampling** (Bengio et al., 2015): During training, randomly use the model's own prediction (instead of ground truth) with increasing probability as training progresses:

$$P(\text{use model prediction}) = \epsilon_t$$

where $\epsilon_t$ increases from 0 to some value (e.g., 0.25) over training. This gradually exposes the model to its own errors.

**Minimum risk training**: Optimize expected loss under the model's own sampling distribution, not the teacher-forced distribution.

**Reinforcement learning approaches**: REINFORCE algorithm to optimize sequence-level metrics (BLEU, ROUGE) directly.

**How transformers handle this**: Autoregressive transformers (GPT) also use teacher forcing during training. But because transformers process all positions in parallel (with causal masking), each position is simultaneously trained to predict the next token given all previous ground-truth tokens. The exposure bias problem still exists in principle, but empirically is less severe because:
1. Self-attention can directly access any previous token (no compression bottleneck)
2. Much larger training corpora reduce the distributional gap
3. Techniques like nucleus sampling and temperature scaling at inference mitigate degenerate outputs

---

## 10. The Information Bottleneck & the Road to Attention

### 10.1 The Fundamental Problem

In Seq2Seq, the entire input sequence must be compressed into a **single fixed-size vector** — the encoder's final hidden state:

$$\text{context} = \mathbf{h}_T^{enc} \in \mathbb{R}^h$$

For $h = 1024$ (typical), this is 1024 floating-point numbers that must encode the complete meaning of a sentence that could be 50+ words long.

**This is a lossy bottleneck**. Information theory tells us that you cannot losslessly compress an arbitrary-length sequence into a fixed-size representation. Early tokens get progressively overwritten as the encoder processes more tokens.

### 10.2 Evidence of the Bottleneck

**Performance degrades with sequence length**: Cho et al. (2014) showed that Seq2Seq BLEU scores dropped sharply for sentences longer than ~20 words. The context vector simply couldn't hold enough information.

**Reversing input helps**: Sutskever's trick of reversing the input sequence worked because it placed the first few input tokens closest to the decoder initialization. These tokens are most relevant for starting the output. But this only partially mitigates the problem — it doesn't help with information in the middle of long sequences.

### 10.3 The Solution: Attention

Instead of compressing the entire input into one vector, let the decoder **look at all encoder hidden states** at each generation step, and **selectively focus on the most relevant parts**:

$$\text{context}_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot \mathbf{h}_i^{enc}$$

where $\alpha_{t,i}$ is the "attention weight" — how much the decoder should focus on encoder position $i$ when generating output token $t$.

This is the **attention mechanism** (Bahdanau et al., 2014), and it solved the bottleneck problem definitively. It's the direct ancestor of the self-attention mechanism in transformers.

**The key insight**: Instead of asking "compress everything into one vector", attention asks "which parts of the input are relevant to what I'm generating right now?" This replaces a fixed compression with an adaptive, query-dependent retrieval — the same principle behind RAG systems today.

### 10.4 From Attention to Transformers

The progression:
1. **Seq2Seq** (2014): Fixed context vector → bottleneck
2. **Attention** (2014-2015): Dynamic context → solves bottleneck, but still sequential RNN
3. **Self-attention** (2017): Apply attention to the sequence itself (not just encoder-decoder) → captures relationships within a sequence
4. **Transformer** (2017): Use ONLY attention (no RNN) → parallelizable, scalable, now the dominant architecture

The transformer's key realization: if attention is doing all the heavy lifting, **why keep the RNN at all?** Remove the recurrence, use only attention, and you get a model that's both more powerful and more parallelizable.

---

## 11. Why Transformers Replaced RNNs

### 11.1 The Three Fundamental Limitations of RNNs

**1. Sequential computation (no parallelization)**

RNNs process tokens one at a time: $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$. You cannot compute $\mathbf{h}_{100}$ until you've computed $\mathbf{h}_1$ through $\mathbf{h}_{99}$.

- Time complexity for a sequence of length $T$: $O(T)$ sequential operations
- Cannot utilize GPU parallelism (GPUs excel at doing thousands of operations simultaneously)
- Training a long sequence takes $T$ times longer than a single step

**Transformers**: Self-attention processes all positions simultaneously. $O(1)$ sequential operations (all positions computed in parallel). Entire sequences processed in one GPU operation.

**2. Limited long-range dependency**

Despite LSTMs/GRUs, practical RNNs struggle with dependencies beyond ~100-200 tokens. The gradient still decays (just much more slowly than vanilla RNNs). The hidden state still undergoes lossy compression.

**Transformers**: Self-attention directly connects every pair of positions. A token at position 1 can attend to a token at position 10,000 with equal architectural capacity. The path length between any two tokens is $O(1)$ (one attention operation), vs $O(T)$ for RNNs.

**3. Compression bottleneck**

RNN hidden states are fixed-size vectors that must compress all previous information. This works for short sequences but degrades for long ones.

**Transformers**: The KV cache stores separate representations for every token. Nothing is compressed — the model has full access to all previous tokens at all times.

### 11.2 The Numbers

| Metric | RNN (LSTM) | Transformer |
|--------|-----------|-------------|
| Path length (position 1 to T) | $O(T)$ | $O(1)$ |
| Sequential operations | $O(T)$ | $O(1)$ |
| Computation per layer | $O(T \cdot h^2)$ | $O(T^2 \cdot d + T \cdot d^2)$ |
| Max practical sequence length | ~500-1000 | ~128K-1M+ |
| Training parallelism | Low (sequential) | High (fully parallel) |
| State-of-the-art NLP (2024) | Obsolete | Universal |

### 11.3 What RNNs Did Right

Despite being replaced, RNNs contributed crucial ideas that live on in transformers:

1. **Gating**: LSTM gates inspired the GLU (Gated Linear Unit) in modern transformer FFNs (SwiGLU in Llama)
2. **Additive state updates**: LSTM cell state's additive updates are the same principle as residual connections
3. **Autoregressive generation**: The teacher-forcing training / autoregressive inference paradigm is exactly how GPT works
4. **Encoder-decoder**: Seq2Seq's encoder-decoder architecture is directly used in T5, BART

### 11.4 Are RNNs Truly Dead?

**Mostly, yes**, for NLP. But there's a renaissance in **linear RNNs**:

- **Mamba** (Gu & Dao, 2023): State Space Model with selective gating, achieves $O(T)$ computation (vs transformer's $O(T^2)$) with competitive quality. Linear in sequence length, making very long sequences feasible.
- **RWKV**: RNN-transformer hybrid that combines recurrent computation with parallel training.
- **Jamba**: Hybrid Mamba + Transformer architecture.

These models revive the RNN's efficient $O(T)$ inference while solving the parallelization and long-range dependency problems. They're research frontier (Topic 27), not yet mainstream for language modeling, but actively competitive with transformers for certain use cases.

---

## 12. Interview Questions & Answers

### Q1: Draw the LSTM cell and explain each gate's role. Why is the cell state the key innovation?

**Answer**: The LSTM cell has four main computations and two state vectors:

**States**:
- $\mathbf{c}_t$ (cell state): Long-term memory, updated additively
- $\mathbf{h}_t$ (hidden state): Short-term output, derived from cell state via output gate

**Gates** (all use sigmoid → output in (0,1)):
1. **Forget gate** $\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$: Decides which components of the old cell state to erase. Near 0 = forget, near 1 = remember.

2. **Input gate** $\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$: Decides which components of the new candidate to write to memory.

3. **Output gate** $\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$: Decides which components of the cell state to expose as the hidden state output.

**Candidate**: $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$ — the proposed new information.

**Cell state update**: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$

**Why the cell state is the key innovation**: The cell state provides an **additive gradient highway**. In vanilla RNNs, the gradient must pass through multiplicative weight matrices at each step: $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(\tanh') \cdot \mathbf{W}_{hh}$. In LSTM, the gradient through the cell state is: $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t)$. When the forget gate is near 1, this is approximately the identity — gradients flow through time steps without vanishing. This is the same principle as residual connections in transformers, discovered 18 years earlier.

### Q2: What is teacher forcing? What is exposure bias? How are they related?

**Answer**: **Teacher forcing** is a training strategy for autoregressive models where the ground-truth previous token is fed as input at each step, instead of the model's own prediction. At step $t$, the decoder receives $y_{t-1}^*$ (ground truth) rather than $\hat{y}_{t-1}$ (model's prediction).

**Benefits**: Stable training, faster convergence, each step gets correct context so gradients are meaningful.

**Exposure bias** is the resulting train-test mismatch. During training, the model always sees correct prefixes. During inference, it sees its own (potentially incorrect) predictions. The model has never been "exposed" to its own errors during training, so it doesn't know how to recover from mistakes.

**They are causally related**: Teacher forcing directly causes exposure bias. If we trained the model on its own predictions (scheduled sampling, reinforcement learning), exposure bias would decrease — but training would be much more unstable because early in training, the model's predictions are nearly random, providing meaningless context for later steps.

**Modern context**: GPT-family models also use teacher forcing (via causal masked attention during training). The exposure bias problem still exists but is less impactful because: (a) self-attention gives direct access to all previous tokens (no compression), reducing error compounding; (b) massive training corpora make the model robust to diverse inputs; (c) sampling strategies (nucleus sampling, temperature) prevent degenerate loops at inference time.

### Q3: Why is the Seq2Seq bottleneck problem important? What solution was proposed?

**Answer**: The Seq2Seq encoder compresses the entire input sequence into a single fixed-size vector $\mathbf{h}_T^{enc} \in \mathbb{R}^h$. For a hidden size of $h = 1024$, every input sentence — whether 5 words or 50 words — must be represented by the same 1024 numbers.

**Why this is important**:
1. **Information loss**: Long sentences lose information — details from early positions are overwritten. Cho et al. (2014) showed BLEU scores dropping sharply for sentences >20 words.
2. **Unequal treatment**: Information at the end of the input is better preserved than information at the beginning (recency bias).
3. **Fundamental limitation**: No amount of LSTM capacity can avoid lossy compression when mapping variable-length inputs to fixed-size representations.

**The solution — Attention** (Bahdanau et al., 2014): Instead of one context vector, let the decoder dynamically attend to all encoder hidden states at each generation step:

$$\text{context}_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot \mathbf{h}_i^{enc}$$

where $\alpha_{t,i} = \text{softmax}(\text{score}(\mathbf{h}_{t-1}^{dec}, \mathbf{h}_i^{enc}))$ gives higher weight to encoder positions relevant to the current decoder step.

This removes the bottleneck entirely — the decoder has access to the full input at every step, with attention serving as a soft retrieval mechanism. This idea was so powerful that it led to the transformer architecture, where self-attention replaced recurrence altogether.

### Q4: Why did transformers replace RNNs? Give at least 3 concrete reasons.

**Answer**:

**1. Parallelization**: RNNs process tokens sequentially — $\mathbf{h}_t$ requires $\mathbf{h}_{t-1}$. A sequence of 1000 tokens requires 1000 sequential operations. Transformers process all positions in parallel via self-attention — one matrix multiplication handles all positions simultaneously. This makes transformers 10-100x faster to train on GPUs, enabling scaling to trillions of tokens.

**2. Long-range dependencies**: In RNNs, information from position 1 must survive $T-1$ sequential hidden state updates to reach position $T$. Even with LSTMs, practical effective memory is ~100-200 tokens. In transformers, self-attention directly connects any two positions with a path length of $O(1)$. Position 1 can directly attend to position 10,000 in a single operation.

**3. No compression bottleneck**: RNN hidden states are fixed-size vectors that compress all previous context. Transformers maintain separate key-value pairs for every token in the KV cache — nothing is compressed. The model has full, uncompressed access to all previous tokens at all times.

**Additional reasons**: (4) Transformers are simpler to optimize — no complex gating dynamics, straightforward gradient flow through residual connections. (5) Transformers scale predictably — scaling laws (Chinchilla) provide clear recipes for optimal model size vs data, which don't exist for RNNs. (6) Hardware alignment — attention is fundamentally matrix multiplication, which is exactly what GPUs/TPUs are optimized for.

### Q5: Compare LSTM and GRU. When would you choose each?

**Answer**: GRU simplifies LSTM by merging the cell state and hidden state into one, and coupling the forget and input gates into a single update gate.

**LSTM**: 3 gates (forget, input, output), 2 state vectors ($\mathbf{c}_t$, $\mathbf{h}_t$), $4h(h+d+1)$ parameters per layer. The separate cell state provides a dedicated long-term memory pathway, and the output gate controls how much is exposed.

**GRU**: 2 gates (update, reset), 1 state vector ($\mathbf{h}_t$), $3h(h+d+1)$ parameters (25% fewer). The update gate interpolates between old and new states: $\mathbf{h}_t = (1-\mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$.

**Choose LSTM when**: Very long sequences where the separate cell state might help, you have abundant compute and data, or the task requires fine-grained control of what information is output (the output gate provides this).

**Choose GRU when**: Smaller datasets (fewer parameters reduce overfitting), latency-sensitive applications (fewer computations per step), or as a default — GRU is simpler and performs comparably in most empirical comparisons.

**The honest answer**: For any task where the LSTM/GRU choice matters, you should probably be using a transformer instead. The main reason to know the distinction is for interview purposes and to understand the historical evolution.

### Q6: Explain the vanishing gradient problem mathematically. Why is it harder to fix than exploding gradients?

**Answer**: In a vanilla RNN, the gradient from time $t$ to time $k$ passes through a chain of Jacobians:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} = \prod_{i=k+1}^{t} \text{diag}(\tanh'(\mathbf{z}_i)) \cdot \mathbf{W}_{hh}$$

Since $\tanh'(z) \leq 1$ and letting $\lambda_{\max}$ be the largest singular value of $\mathbf{W}_{hh}$:

$$\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k}\right\| \leq \lambda_{\max}^{t-k}$$

When $\lambda_{\max} < 1$: gradients decay exponentially with distance $(t-k)$. For a gap of 50 steps with $\lambda_{\max} = 0.9$: $0.9^{50} \approx 0.005$. The gradient is effectively zero — the model cannot learn this dependency.

**Why exploding is easier to fix**: Exploding gradients ($\lambda_{\max} > 1$) are solved by gradient clipping — cap the gradient norm at a threshold. The clipped gradient still points in the correct direction, so the model still learns; it just takes smaller steps.

**Why vanishing is harder**: You can't simply "scale up" vanishing gradients because:
1. A gradient near zero contains no useful directional information — it's noise
2. Multiplying noise by a constant gives you louder noise, not signal
3. The problem is structural: the signal has already been destroyed by the repeated multiplications

The solution requires **architectural changes** that provide additive (not multiplicative) gradient pathways: LSTM cell states, residual connections, or eliminating recurrence entirely (transformers).

### Q7: What is the relationship between LSTM cell states and residual connections in transformers?

**Answer**: Both solve the same problem (vanishing gradients in deep computation graphs) using the same mathematical principle (additive state updates).

**LSTM cell state**: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$

When $\mathbf{f}_t \approx 1$: $\mathbf{c}_t \approx \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ — additive update to previous state.

Gradient: $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \mathbf{I}$ — identity, no vanishing.

**Transformer residual**: $\mathbf{h}_l = \mathbf{h}_{l-1} + \text{Sublayer}(\mathbf{h}_{l-1})$

Gradient: $\frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} = \mathbf{I} + \frac{\partial \text{Sublayer}}{\partial \mathbf{h}_{l-1}}$ — always includes identity.

Both provide a "gradient highway" — a path through which gradients can flow unimpeded across many layers (LSTM) or many time steps (transformer). The identity component ensures the gradient never vanishes completely.

The transformer's approach is cleaner: the residual is always an unmodified addition of the identity ($\mathbf{h}_{l-1}$). The LSTM's approach is gated: the forget gate $\mathbf{f}_t$ can potentially close the gradient highway if needed. This gating gives LSTMs more flexibility but transformers more stability.

### Q8: How does beam search work? Why is it used instead of greedy decoding?

**Answer**: **Greedy decoding** picks the highest-probability token at each step: $y_t = \arg\max P(y \mid y_{<t}, x)$. This is fast ($O(V)$ per step) but myopic — it can miss sequences where an initially lower-probability token leads to a much better overall sequence.

**Example**: $P(\text{"the"}) = 0.5$, $P(\text{"a"}) = 0.4$. Greedy picks "the". But $P(\text{"dog"} \mid \text{"a"}) = 0.9$ while $P(\text{"dog"} \mid \text{"the"}) = 0.1$. "A dog" (joint 0.36) beats "the dog" (joint 0.05), but greedy committed to "the" and can't backtrack.

**Beam search** maintains $B$ candidate sequences (beams). At each step:
1. For each beam, compute probability of all $V$ possible next tokens → $B \times V$ candidates
2. Score each candidate by cumulative log probability: $\sum_{i=1}^t \log P(y_i \mid y_{<i}, x)$
3. Keep only the top $B$ candidates
4. Stop when all beams have produced an EOS token

**Length normalization** is critical: without it, beam search heavily favors shorter sequences (fewer log probabilities to sum, less negative). The typical normalization divides the score by $T^\alpha$ where $\alpha \in [0.6, 0.7]$.

**Beam search limitations**: It's still an approximate search (not guaranteed to find the global optimum). For open-ended generation (chatbots, creative writing), beam search produces generic, repetitive text because it optimizes likelihood, not diversity. Modern LLMs use sampling-based methods (nucleus sampling, temperature) instead, which produce more natural and varied text.

### Q9: What is scheduled sampling and why was it proposed?

**Answer**: Scheduled sampling (Bengio et al., 2015) was proposed to address exposure bias — the mismatch between teacher forcing (training) and autoregressive generation (inference).

**The approach**: At each decoder step, with probability $\epsilon_t$, use the model's own prediction $\hat{y}_{t-1}$ instead of the ground truth $y_{t-1}^*$. The probability $\epsilon_t$ starts at 0 (pure teacher forcing) and increases during training according to a schedule:

- **Linear decay**: $\epsilon_t = \min(\epsilon_{max}, k \cdot t)$
- **Exponential decay**: $\epsilon_t = 1 - k^t$  
- **Inverse sigmoid**: $\epsilon_t = 1 - k/(k + \exp(t/k))$

**Intuition**: Early in training, the model's predictions are garbage — feeding them as input would produce meaningless gradients. So we start with teacher forcing to establish a reasonable model. Gradually, we introduce the model's own predictions, exposing it to its own error distribution so it learns to recover from mistakes.

**Limitations**: Scheduled sampling is a heuristic — the curriculum schedule requires tuning, and the effectiveness is inconsistent across tasks. It also doesn't address the fundamental issue that the model is trained with a different objective (cross-entropy per token) than what we evaluate (sequence-level BLEU/ROUGE).

### Q10: Trace the evolution from RNNs to transformers. What problem did each step solve?

**Answer**:

**Step 1: Vanilla RNN (1986)** — First model to process sequences with shared weights. **Problem it solved**: Handling variable-length sequences. **Problem it created**: Vanishing/exploding gradients — cannot learn long-range dependencies.

**Step 2: LSTM (1997)** — Added gated cell state with additive updates. **Problem it solved**: Vanishing gradients — cell state provides gradient highway. **Problem it created**: Still sequential, still compresses history into fixed-size state, limited to ~200 token effective memory.

**Step 3: Seq2Seq (2014)** — Encoder-decoder architecture for variable-length input → variable-length output. **Problem it solved**: Machine translation, summarization, any sequence mapping. **Problem it created**: Information bottleneck — entire input compressed into one vector.

**Step 4: Attention (2014-2015)** — Decoder dynamically attends to all encoder states. **Problem it solved**: Information bottleneck — decoder has full access to all input positions. **Problem it created**: Still relies on sequential RNN backbone — slow, limited scaling.

**Step 5: Self-attention (2017)** — Apply attention within a sequence (not just encoder→decoder). **Problem it solved**: Captures relationships between all position pairs in a sequence.

**Step 6: Transformer (2017)** — Use only attention, remove RNN entirely. **Problem it solved**: Full parallelization (process all positions simultaneously), $O(1)$ path length between any positions, scales to billions of parameters. This is the architecture behind BERT, GPT, Llama, and every modern LLM.

Each step solved the most critical limitation of the previous step, and the progression reveals a clear theme: **making information flow freer and computation more parallel**.

---

*Next topic: [Topic 8: Attention Mechanisms](../Phase_C_Transformers/08_Attention_Mechanisms.md)*
