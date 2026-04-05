# Topic 27: Research Frontiers (Mamba, MoE, Test-Time Compute)

## Table of Contents

1. [Beyond Transformers: State Space Models](#1-beyond-transformers-state-space-models)
2. [Mixture of Experts (MoE)](#2-mixture-of-experts-moe)
3. [Test-Time Compute & Reasoning Models](#3-test-time-compute--reasoning-models)
4. [Long-Context Architectures](#4-long-context-architectures)
5. [Synthetic Data & Small Language Models](#5-synthetic-data--small-language-models)
6. [Mechanistic Interpretability (Research Angle)](#6-mechanistic-interpretability-research-angle)
7. [Retrieval-Augmented Pretraining](#7-retrieval-augmented-pretraining)
8. [Architecture Trends: What Comes After Transformers?](#8-architecture-trends-what-comes-after-transformers)
9. [Interview Q&A](#9-interview-qa)

---

## 1. Beyond Transformers: State Space Models

### Why Look Beyond Transformers?

Transformers have two core limitations at scale:
1. **O(n²) attention complexity**: Processing a 1M-token document requires $10^{12}$ attention operations
2. **O(n) KV cache memory during inference**: Each generated token requires caching all previous keys and values

State Space Models (SSMs) offer O(n) training and O(1) inference state — a fundamentally different trade-off.

### State Space Models: The Math

SSMs are inspired by linear dynamical systems from control theory:

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

Where $h(t)$ is the hidden state, $x(t)$ is the input, $y(t)$ is the output.

For discrete sequences (discretized with step size $\Delta$):
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

**Recurrent view**: O(1) inference — only need to maintain hidden state $h_t$  
**Convolutional view**: During training, SSM is equivalent to a convolution, allowing parallelization

$$y = x * \bar{K}, \quad \bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$$

This is O(n log n) with FFT, much better than O(n²) attention.

### S4 (Structured State Spaces for Sequences)

Key innovation: parameterize $A$ as a diagonal plus low-rank matrix (DPLR) — called the **HiPPO matrix** — which is specifically designed to memorize history efficiently.

The HiPPO matrix $A$ projects the input history onto Legendre polynomials, giving theoretically optimal compression of long-range history.

### Mamba (Gu & Dao, 2023)

**Core innovation**: Selective state spaces — make $B$, $C$, and $\Delta$ **input-dependent**.

Standard S4: $A$, $B$, $C$, $\Delta$ are fixed parameters  
Mamba: $B(x_t)$, $C(x_t)$, $\Delta(x_t)$ — computed from the input at each step

```
For each token x_t:
  Δ, B, C = linear_projection(x_t)  # input-dependent
  h_t = discretize(A, B, Δ) * h_{t-1} + B * x_t
  y_t = C * h_t
```

**Why this matters**: The model can now selectively remember or forget information based on the content. A fixed $A$ matrix must treat all inputs equally.

**Hardware-aware algorithm**: Mamba uses a parallel scan algorithm optimized for GPU memory bandwidth — avoiding the materialization of intermediate states.

### Mamba vs Transformer Comparison

| Property | Transformer | Mamba |
|---------|------------|-------|
| **Training complexity** | O(n²) | O(n log n) or O(n) |
| **Inference state** | O(n) KV cache | O(d²) fixed hidden state |
| **Parallelization (train)** | Full | Full (via parallel scan) |
| **Long-context performance** | Good (with RoPE+Flash) | Strong in theory, mixed in practice |
| **Quality on language tasks** | Excellent | ~Competitive at 3B, gap widens at scale |
| **Recallability** | Perfect (exact attention) | Lossy (fixed state size) |

**Key limitation**: Mamba's hidden state has fixed size. For tasks requiring exact recall of long-range information (e.g., retrieve a specific sentence from 100K tokens earlier), attention is better.

### Hybrid Architectures: Jamba & Mamba-2

**Jamba** (AI21, 2024): Interleave Mamba layers with transformer attention layers
- Every N Mamba layers, insert 1 attention layer
- Attention layers handle tasks requiring exact recall
- Mamba layers handle long-range dependency accumulation cheaply

```
Layer 1: Mamba
Layer 2: Mamba
Layer 3: Mamba
Layer 4: Attention  ← every 4th layer
...
```

Result: 256K effective context window with ~70% lower memory than pure transformer

**Mamba-2**: Restructures selective SSM as a semiseparable matrix, enabling more efficient CUDA kernels and better hardware utilization.

---

## 2. Mixture of Experts (MoE)

### Core Idea

Instead of one large dense FFN that activates all parameters for every token, have **many smaller expert FFNs** and a **router** that selects which experts process each token.

```
Token x
  ↓
Router: g(x) = softmax(W_r · x)
  ↓
Top-K selection: select experts {e₁, e₂} with highest gate scores
  ↓
y = g₁(x) · FFN₁(x) + g₂(x) · FFN₂(x)
```

**Key property**: Large *total* parameters, small *active* parameters per token.

| Model | Total Params | Active Params/Token | Experts |
|-------|-------------|--------------------|---------| 
| Mixtral 8×7B | 47B | 13B | 8, top-2 |
| GPT-4 (estimated) | ~1.8T | ~200B | ~16, top-2 |
| Grok-1 | 314B | 86B | 8, top-2 |

### MoE Architecture (Mixtral 8×7B)

Every other FFN layer is replaced with a MoE layer:

```
Standard transformer layer:
  x → Attention → LayerNorm → FFN → LayerNorm → output

MoE transformer layer:
  x → Attention → LayerNorm → MoE(FFN_1...FFN_8, Router) → LayerNorm → output
```

**Router** (token-choice routing):
$$G(x) = \text{TopK}(\text{softmax}(W_r x), k=2)$$
$$y = \sum_{i \in \text{TopK}} G_i(x) \cdot \text{FFN}_i(x)$$

Each expert is a standard SwiGLU FFN. The router is a learned linear layer.

### Load Balancing Problem

Without explicit regularization, the router collapses: a few experts get all tokens (expert collapse).

**Auxiliary load-balancing loss** (Switch Transformer, GShard):

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{E} f_i \cdot P_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$ (discrete)
- $P_i$ = average router probability for expert $i$ (differentiable)
- $\alpha$ = hyperparameter (typically 0.01)

Goal: $f_i \approx 1/E$ for all experts.

**Expert capacity**: Limit how many tokens each expert can process per batch. Tokens that exceed capacity are dropped (capacity factor $C$, typically 1.0–2.0).

### Expert Specialization

After training, experts specialize implicitly:
- Some experts handle specific syntax (punctuation, code)
- Some handle semantic domains (medical, legal, scientific)
- Some handle specific languages

This specialization is emergent — not engineered.

### MoE Training Challenges

1. **Communication overhead**: In distributed training, token routing across GPUs requires all-to-all communication
2. **Expert collapse**: Without load balancing, training is unstable
3. **Uneven compute**: Variable number of tokens per expert → GPU idle time
4. **Memory**: All expert weights must fit in memory (or be sharded across GPUs)

### Fine-Grained vs Coarse MoE

**Standard MoE** (Mixtral): Large experts, few of them  
**Fine-grained MoE** (DeepSeekMoE): More, smaller experts → more flexibility

DeepSeek MoE splits each expert into smaller "atoms" and routes to more of them:
- 64 fine-grained experts, top-6 routing
- Plus 2 "shared experts" always active (handle common knowledge)
- Result: Same compute as top-2/8 but higher quality

---

## 3. Test-Time Compute & Reasoning Models

### What is Test-Time Compute?

**Training-time compute**: FLOPs spent training the model  
**Test-time compute (inference-time compute)**: FLOPs spent generating each answer

The insight: you can trade *more inference compute* for *better answers*, independent of model size.

```
Standard LLM:    Input → 1 forward pass → Output
Reasoning LLM:   Input → N forward passes (internal thinking) → Output
```

### Chain-of-Thought as Test-Time Compute

CoT forces the model to use more output tokens before giving an answer. Each token in the chain uses one forward pass.

```
Question: "If 3 workers take 4 days to build a wall, how many days for 6 workers?"
Standard: "2 days"
CoT:      "3 workers × 4 days = 12 worker-days total.
           6 workers → 12 / 6 = 2 days. Answer: 2 days"
```

More tokens = more compute = better accuracy on reasoning tasks. This is the simplest form of test-time scaling.

### OpenAI o1/o3: Chain-of-Thought at Scale

o1 (September 2024): A reasoning model trained to spend many tokens on an internal "thinking" chain before answering.

**Key details:**
- Internal scratchpad with extended reasoning (not shown to user)
- Trained with RL to search for correct answers, not just plausible-sounding ones
- At inference, more "thinking tokens" = higher accuracy

**Test-Time Compute Scaling Law** (Snell et al. 2024):
$$\text{Accuracy} \propto \log(\text{test-time compute})$$

You can achieve the same accuracy as a 10× larger model by using 10× more inference compute.

```
o1-mini (small) + 10× compute ≈ GPT-4o performance on math benchmarks
```

### Approaches to Test-Time Compute

**1. Best-of-N (BoN)**
Generate N responses, score each with a reward model or verifier, pick the best.
- Simple, effective
- Compute scales as O(N)
- Works well when verification is cheap (math: check if answer correct)

**2. Self-Consistency** (Wang et al. 2022)
Sample N reasoning chains, majority-vote on final answer.
- No reward model needed
- Accuracy: 56% → 74% on GSM8K with N=40

**3. Tree of Thought (ToT)**
Explore answer space as a tree — generate multiple partial reasoning paths, evaluate each node, prune, and expand promising branches.

```
Root: Problem statement
├── Branch A: Approach via algebra
│   ├── A1: Simplify → dead end
│   └── A2: Substitute → promising
│       └── A2.1: Solve → answer
└── Branch B: Approach via geometry
    └── B1: Draw diagram → promising
```

**4. Process Reward Models (PRMs)**
Instead of scoring final answers (outcome reward), score each reasoning step.
- PRM trains: given (problem, step_1, ..., step_k), predict whether step k+1 is correct
- PRM800K dataset (OpenAI): 800K step-level labels on math solutions

PRMs enable better beam search over reasoning chains.

**5. MCTS (Monte Carlo Tree Search)**
Full tree search with:
- **Selection**: UCB score to balance exploration/exploitation
- **Expansion**: Generate next reasoning step
- **Simulation**: Roll out to completion
- **Backpropagation**: Update value estimates

Used in AlphaCode 2 for competitive programming.

### When Does Test-Time Compute Help?

| Task Type | Benefit | Why |
|---------|---------|-----|
| Math / logic | High | Ground truth verification is cheap; more search = better |
| Coding | High | Can run code to verify |
| Complex reasoning | High | Multiple steps, easy to check consistency |
| Factual QA | Medium | Benefit if retrieval/verification possible |
| Creative writing | Low | No objective correct answer |
| Simple chat | Low | Single correct response, no need for search |

---

## 4. Long-Context Architectures

### The Challenge

Training on 1M token sequences is expensive:
- O(n²) attention: 1M tokens → $10^{12}$ attention operations per layer
- Storing activations for backprop: enormous memory

Several approaches tackle this at different levels.

### Sparse Attention

Instead of every token attending to every other token, restrict attention patterns:

**Sliding window attention** (Mistral/Longformer): Each token attends only to nearby W tokens.
$$\text{Complexity}: O(n \cdot W)$$

**Global + local** (BigBird, Longformer): A few global tokens attend to everything; rest use sliding window.

**Dilated attention** (LongNet): Exponentially increasing gaps between attended positions.

### RoPE Context Extension

Models trained with 4K context struggle at 32K. Solutions:

**Linear interpolation**: Scale position indices down. If max position was 4096, at inference position 8192 becomes 4096. Simple but loses high-frequency information.

**NTK-aware scaling** (LocalLLaMA 2023): Change the base frequency of RoPE:
$$\theta_i \rightarrow \theta_i \cdot \left(\frac{L'}{L}\right)^{2i/d}$$

This preserves high-frequency components better.

**YaRN**: Dynamic NTK scaling + attention temperature correction. Used in Mistral for 128K context.

**LongRoPE**: Find non-uniform rescaling factors per dimension via evolutionary search. Used in Phi-3 for 128K context.

### Memory-Augmented Transformers

**MemGPT**: Treat the LLM like an OS — explicit paging of context in/out of "active memory"
**Infini-Attention** (Google, 2024): Compressive memory alongside local attention; KV cache is compressed into a fixed-size matrix and continually updated

### Streaming Inference: StreamingLLM

Observation: LLM attention sinks — the first few tokens receive disproportionately high attention regardless of content. This is because initial tokens collect "garbage" attention from positions that don't need to attend anywhere specific.

**StreamingLLM**: Keep the first K "sink tokens" in the KV cache always. Use a sliding window for the rest. Enables infinite-length generation with fixed memory.

---

## 5. Synthetic Data & Small Language Models

### The Synthetic Data Opportunity

High-quality data is scarce. But a capable LLM can *generate* training data for smaller models.

**Key finding** (Gunasekar et al. 2023 — Phi-1): A 1.3B parameter model trained on "textbook-quality" synthetic Python data outperforms 10× larger models trained on raw code.

### Textbook Is All You Need (Phi series)

**Data curation philosophy:**
- Web data has low educational signal-to-noise ratio
- Textbooks are written to explain concepts clearly, with worked examples
- Synthetic "textbook" data generated by GPT-4 can match real textbook quality

**Phi-1.5 recipe:**
1. Curate ~7B tokens of "high educational value" web data (heuristic filtering)
2. Generate ~20B tokens of synthetic "textbooks" and "exercises" with GPT-4
3. Train 1.3B parameter model on this mix
4. Result: beats much larger models on reasoning benchmarks

**Why does it work?**
- Textbook-quality data is dense in concepts
- Few redundant or irrelevant examples
- Coverage is broad (unlike web data which over-represents certain domains)

### Microsoft's Orca Approach

Orca trains small models to mimic the *reasoning process* of large models, not just their outputs.

**Teacher traces**: Ask GPT-4 to explain its step-by-step reasoning, not just give an answer. Train the small model on the full trace.

```
Standard distillation:
Input: "What is 15% of 80?"
Output: "12"   ← small model learns to predict "12"

Orca distillation:
Input: "What is 15% of 80?"
Output: "15% means 15 per hundred. 15/100 × 80 = 0.15 × 80 = 12. Answer: 12"
← small model learns the reasoning process
```

### LIMA (Less Is More for Alignment)

Finding: 1,000 carefully curated instruction-following examples is sufficient for alignment — quality >>> quantity.

- LIMA: 1K examples, hand-curated for diversity and quality
- Result: beats models fine-tuned on 50K+ examples on many tasks

**Implication**: The model's *knowledge* comes from pretraining. Fine-tuning teaches the *format* of helpful responses. You don't need millions of fine-tuning examples.

### Distillation via Speculative Decoding

**Speculative decoding as curriculum**: Use the draft model's error patterns to identify what it struggles with → generate more training data for those cases.

---

## 6. Mechanistic Interpretability (Research Angle)

*(See also Topic 26 §8 for the safety connection)*

### Superposition Hypothesis (Deep Dive)

A neural network with $d$ dimensions can represent $n >> d$ features simultaneously by storing them as almost-orthogonal directions.

**Toy model experiment** (Anthropic 2022):
- Train a 2-hidden-unit MLP to reconstruct 5-dimensional sparse inputs
- Find that the 2 units encode all 5 features via superposition
- Features appear as pentagon vertices in the 2D hidden space

**Implication for interpretability**: You cannot interpret individual neurons — they're polysemantic. You must find the *feature directions*.

### Sparse Autoencoders (SAEs) — State of the Art

**Goal**: Find the sparse feature directions in superposition.

**Method**:
1. Run the model on a large corpus, collect layer activations
2. Train an SAE: wide sparse autoencoder $h \rightarrow f \rightarrow h'$
3. Each active dimension of $f$ corresponds to one interpretable feature

**Results (Anthropic, 2024)**:
- SAE on Claude Sonnet finds ~34M features across layers
- Features include: "Golden Gate Bridge" (a specific location concept), "sycophancy", "recursive function calls", human names
- Steering by adding the "banana" feature direction causes model to mention bananas in unrelated contexts

### Double Descent & Grokking

**Grokking** (Power et al. 2022): Models first overfit (training loss ↓, val loss stays high), then after extended training, suddenly generalize (val loss drops sharply).

This happens because:
1. The model first memorizes training examples (low-efficiency algorithm)
2. With more training, it "discovers" a generalizing algorithm (efficient representation)

Implication: training longer than needed for training loss convergence can improve generalization.

**Double descent**: Test error decreases → increases at interpolation threshold → decreases again as model size grows into the over-parameterized regime.

### Circuits Research

**Goal**: Find the smallest computational subgraph responsible for a specific capability.

**Indirect Object Identification (IOI) circuit** (Wang et al. 2022):
- Task: "Mary gave John a present. John gave ___" → predict "Mary"
- Circuit identified: 3 types of attention heads work together
  - **S-inhibition heads**: Attend to duplicate tokens, suppress them
  - **Name Mover heads**: Copy name of IOI (Mary) to output position
  - **Duplicate Token heads**: Identify that John appeared twice

This full circuit was identified in GPT-2 Small (12 layers, 85M params).

---

## 7. Retrieval-Augmented Pretraining

### RETRO (Retrieval Enhanced Transformer, DeepMind 2022)

**Idea**: Instead of compressing all world knowledge into model weights during pretraining, retrieve relevant documents *during* pretraining.

**Architecture**:
```
Input: "The capital of France is ___"
  ↓ chunk and encode
Retrieval: find nearest neighbors in a 2T-token database
  ↓ chunked cross-attention
Retrieved: ["Paris is the capital of France and largest city..."]
  ↓
Model: "Paris"
```

**Chunked cross-attention**: Divide input into chunks of C tokens. Each chunk attends to its retrieved documents via cross-attention.

**Results**:
- 7.5B RETRO matches 175B GPT-3 on perplexity
- Because knowledge is stored in the retrieval database, not parameters
- Database is updatable post-training (unlike weights)

**Why it's powerful**:
- Knowledge cutoff is the database, not training time
- Updating knowledge = updating the database (no retraining)
- Model can cite sources

### Challenges vs RAG

| | RETRO | RAG (applied post-training) |
|--|-------|---------------------------|
| When retrieval happens | During training & inference | Only at inference |
| Architecture change | Yes (chunked cross-attention) | No |
| Training cost | High | None |
| Retrieval integration | Tight (trained jointly) | Loose (prompt-based) |
| Update knowledge | Update DB | Update DB |

---

## 8. Architecture Trends: What Comes After Transformers?

### The Bitter Lesson (Rich Sutton, 2019)

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

Specific clever architectures consistently lose to simpler ones that scale better. Examples:
- Hand-crafted chess evaluation functions → self-play RL
- Linguistic feature engineering → end-to-end deep learning
- Convolutional inductive biases → ViT (pure attention on image patches)

**Implication**: The architecture that scales better usually wins, even if it's less principled.

### Current Architectural Convergence

Modern frontier LLMs have converged on:
- **Pre-LayerNorm** (stability)
- **RMSNorm** (efficiency over LayerNorm)
- **RoPE** (length extrapolation)
- **SwiGLU FFN** (empirically better than ReLU)
- **GQA** (memory efficiency)
- **No absolute positional embeddings** (RoPE handles it)
- **~30:70 attention:FFN parameter ratio**

### What Transformers Can't Do Well

1. **Exact recall over very long contexts**: Attention softmax distributes probability mass; small probability → small gradient → forgotten
2. **State tracking** (implicit): RNNs maintain explicit state; transformers must reconstruct state from context each time
3. **Efficient inference for very long sequences**: O(n) KV cache grows unboundedly

### Candidate Next Architectures

**SSM + Attention hybrids** (Jamba, Zamba, Falcon Mamba):
- SSM for long-range compression
- Sparse attention for exact recall
- Promising but not yet proven at full scale

**Linear attention**:
- Rewrite attention as kernel feature map, enabling O(n) inference
- Quality gap vs softmax attention on language tasks remains
- RetNet, RWKV, GLA (Gated Linear Attention) are leading candidates

**Structured State Space Duality** (Mamba-2):
- SSM is equivalent to a specific form of linear attention
- Unifies the two research directions theoretically

### Scaling Laws for Next-Generation Architectures

The critical question: does the new architecture have a better **inference efficiency frontier**?

```
Quality vs Compute at Inference:
                    
Quality   ●  Transformer (dense)
  ↑       ●  MoE (more params, same active compute)
          ●  SSM (same quality, less inference compute?)
          
          → Inference compute
```

MoE already shifts the frontier. SSMs may shift it further for long-context tasks. The field is actively evaluating.

---

## 9. Interview Q&A

**Q: What is a State Space Model (SSM)? How does it differ from a transformer?**

A: An SSM models sequences via a linear dynamical system:
$$h_t = Ah_{t-1} + Bx_t, \quad y_t = Ch_t$$

Key differences:

| | Transformer | SSM (Mamba) |
|--|------------|-------------|
| Complexity | O(n²) attention | O(n log n) or O(n) |
| Inference state | O(n) KV cache | O(d²) fixed hidden state |
| Recall | Exact (all tokens in cache) | Lossy (fixed state) |
| Selectivity | Attention weights are input-dependent | Mamba: A,B,C input-dependent |

SSMs are theoretically superior for very long sequences but have a quality gap on standard language benchmarks at current scales.

---

**Q: What is Mixture of Experts (MoE)? What is a router?**

A: MoE replaces dense FFN layers with N expert FFNs plus a router that selects K of them per token.

```
Router: g(x) = TopK(softmax(W_r · x), K)
Output: y = Σᵢ gᵢ(x) · FFNᵢ(x)
```

This gives large *total* parameters (more knowledge capacity) with small *active* parameters per token (same compute as a smaller dense model).

**Router**: A learned linear layer that maps each token to a distribution over experts, then selects the top-K. Key challenge: without regularization, routing collapses (a few experts get all tokens). Auxiliary load-balancing loss enforces uniform routing.

---

**Q: What is test-time compute? How do models like o1/o3 use "thinking" at inference?**

A: Test-time compute means spending more FLOPs per query at inference to improve answer quality — instead of (or in addition to) using a larger model.

o1/o3 work by:
1. Training with RL to generate long internal "thinking chains" before answering
2. At inference, the model explores multiple reasoning paths internally
3. More thinking tokens = more accurate answers on reasoning tasks

Scaling law: accuracy scales roughly as $\log(\text{test-time compute})$.

**When it helps**: Tasks with verifiable correct answers (math, code). **When it doesn't**: Creative writing, simple factual Q&A.

---

**Q: Explain Mamba. How does its selective state space mechanism work?**

A: Mamba (Gu & Dao 2023) extends structured SSMs with **input-dependent** parameters.

Standard SSMs: $A$, $B$, $C$, $\Delta$ are fixed for all inputs → the model cannot selectively focus on or ignore tokens.

Mamba: $B(x_t)$, $C(x_t)$, $\Delta(x_t)$ are computed from the current input via linear projections.

This selectivity means:
- Large $\Delta$ → large time step → current input strongly affects hidden state (focus on this token)
- Small $\Delta$ → small time step → hidden state barely changes (ignore this token)

Mamba also uses a hardware-aware parallel scan algorithm to compute the recurrence efficiently on GPUs, avoiding materializing intermediate states.

---

**Q: Why do MoE models have more total parameters but use fewer per-token?**

A: In an MoE layer, only K out of N experts process each token (typically top-2 out of 8).

Example — Mixtral 8×7B:
- 8 experts, each is a full 7B FFN
- Only 2 experts active per token
- Active parameters per token: ~13B (2 × 7B, minus shared attention)
- Total parameters: ~47B

The router's computation is negligible. So you get the quality of a 47B model (large knowledge capacity) at the inference cost of a ~13B dense model.

**Routing challenges**:
1. **Load balancing**: Without regularization, a few experts get all tokens (expert collapse). Auxiliary loss enforces uniform routing.
2. **Expert capacity**: Each expert can only process C tokens per batch. Tokens exceeding capacity are dropped (affects quality).
3. **Communication overhead**: In tensor-parallel training, tokens must be all-to-all communicated to the assigned expert's GPU.

---

**Q: Compare Mamba vs Transformer: training parallelization, inference speed, long-context performance, and language quality.**

A:

**Training parallelization**:
- Transformer: fully parallelizable via attention (can compute all positions simultaneously)
- Mamba: uses parallel scan (O(n log n)), nearly as parallelizable as attention in practice

**Inference speed**:
- Transformer: O(n) KV cache per request; decode step is fast but memory grows
- Mamba: O(1) hidden state; decode step is a simple matrix multiply — faster and memory-constant

**Long-context performance**:
- Transformer: theoretically O(n²) but with Flash Attention and RoPE extension handles 128K+ tokens well
- Mamba: fixed hidden state means lossy compression; recall of specific early content degrades
- Hybrid (Jamba): best of both worlds

**Language quality**:
- At 3B parameters: Mamba ≈ Transformer on standard benchmarks
- At 7B+: Transformer still leads on language modeling, instruction following
- Mamba leads on some long-context retrieval tasks

---

**Q: What is the relationship between test-time compute scaling and chain-of-thought reasoning?**

A: CoT is the simplest form of test-time compute scaling — each reasoning step is a forward pass.

The deeper connection:
- **Search**: CoT allows the model to generate and evaluate multiple candidate reasoning paths (self-consistency, ToT, MCTS)
- **Verification**: Test-time compute is most useful when answers can be verified (math: check by computing; code: run the program)
- **Compute budget**: As you allow more tokens (more test-time compute), accuracy improves logarithmically

**Is test-time compute a new scaling axis?**

Yes — alongside model parameters and training tokens, test-time compute is a third axis:
$$\text{Performance} \approx f(N_\text{params}, D_\text{tokens}, C_\text{test-time})$$

OpenAI's results with o1/o3 suggest test-time compute scaling is at least as powerful as doubling model size for reasoning tasks. But they are not interchangeable — a small model with huge test-time compute can't match a large model on knowledge-intensive tasks.

---

**Q: What are the limitations of the transformer architecture that next-generation architectures try to address?**

A:

**1. Quadratic attention complexity**: O(n²) time and O(n) KV cache memory.
- Addressed by: sparse attention (Mistral), linear attention (RetNet, RWKV), SSMs (Mamba)

**2. Fixed context window**: Models don't naturally generalize beyond training context.
- Addressed by: RoPE extensions (YaRN, LongRoPE), retrieval augmentation

**3. Stateless inference**: Transformer regenerates all context at each step — no persistent state.
- Addressed by: SSMs (fixed hidden state), MemGPT, Infini-Attention

**4. No explicit memory management**: All tokens equally in context; no compression of old content.
- Addressed by: Mamba (selective gating), RETRO (external database), compressive transformers

**5. MoE underutilization**: Dense models compute over all parameters for every token.
- Addressed by: Mixture of Experts (conditional computation)

**Honest assessment**: No successor has definitively beaten dense transformers at scale on all tasks. Hybrids (MoE + attention, Mamba + attention) are the most promising current direction.

---

**Q: What is the "bitter lesson" (Rich Sutton)? How does it apply to current trends in AI architecture design?**

A: Sutton's 2019 essay argues: over 70 years of AI research, methods that leverage general computation and scaling consistently outperform methods that leverage human knowledge and hand-designed structure.

Historical examples:
- Chess: symbolic evaluation → deep RL self-play
- Speech: phoneme engineering → end-to-end deep learning
- Vision: CNN inductive biases → ViT (transformers on raw patches)

**Application to current trends:**

*Supports*: Scaling transformers (simple architecture + more compute + more data) has been the biggest recent win. MoE extends this by making compute more efficient.

*Implication for SSMs*: Mamba's selective mechanism is a clever inductive bias. If Sutton is right, a simple architecture with better scaling properties will eventually win — even if Mamba is smarter per compute.

*Implication for test-time compute*: This is a general computational approach (search) rather than a clever representation. The bitter lesson predicts this will scale well — and o1/o3 results support this.

*Tension*: Human knowledge still matters during data quality curation, evaluation design, and alignment. The lesson is more nuanced than "ignore domain knowledge always."
