# Topic 16: Alignment — RLHF, DPO & Constitutional AI

> **Interview Weight**: ★★★★★ — Asked at every senior AI Scientist interview at Google, Amazon, Anthropic, and OpenAI. You must be able to derive the RLHF objective, explain why PPO is used, explain DPO's key insight, and compare them quantitatively.

## Table of Contents

1. [The Alignment Problem — Why It Matters](#1-the-alignment-problem--why-it-matters)
2. [RLHF — Reinforcement Learning from Human Feedback](#2-rlhf--reinforcement-learning-from-human-feedback)
3. [The Reward Model](#3-the-reward-model)
4. [PPO for LLM Fine-Tuning](#4-ppo-for-llm-fine-tuning)
5. [The KL Penalty — Preventing Reward Hacking](#5-the-kl-penalty--preventing-reward-hacking)
6. [DPO — Direct Preference Optimization](#6-dpo--direct-preference-optimization)
7. [RLHF vs DPO — Comparison](#7-rlhf-vs-dpo--comparison)
8. [Constitutional AI & RLAIF](#8-constitutional-ai--rlaif)
9. [ORPO, SimPO & Modern Variants](#9-orpo-simpo--modern-variants)
10. [Preference Data Collection](#10-preference-data-collection)
11. [Alignment Tax & Capability Preservation](#11-alignment-tax--capability-preservation)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. The Alignment Problem — Why It Matters

### What Alignment Means

A language model trained purely on next-token prediction optimizes for **predicting the training distribution** — not for being helpful, harmless, or honest. The raw pretrained model will:

- Generate toxic content if that's statistically likely
- Confidently hallucinate facts
- Follow harmful instructions if the question is phrased naturally
- Give verbose, padding-heavy responses that predict well but aren't useful

**Alignment** is the problem of making the model's outputs match human values and preferences.

### The Three Failure Modes

```
Failure 1 — Sycophancy:
  User: "Is my essay good?"
  Misaligned model: "Yes, it's excellent!" (tells user what they want to hear)
  Aligned model: "Here's specific feedback on structure and clarity..."

Failure 2 — Reward Hacking:
  Reward function: "longer responses score higher"
  Misaligned model: repeats the same point 10 times to maximize length
  This is valid optimization — the reward function is wrong

Failure 3 — Harmful Compliance:
  User: "How do I pick a lock?"
  Misaligned model: step-by-step instructions
  Aligned model: context-dependent — locksmith vs. burglar
```

### The RLHF Solution: Use Human Preferences

Instead of defining a reward function manually (hard, incomplete), **learn it from human comparisons**:

$$
\text{Human rater sees: Response A vs Response B} \to \text{"A is better"}
$$

$$
\text{Train a reward model to predict these preferences}
$$

$$
\text{Fine-tune the LLM to maximize the reward model's score}
$$

---

## 2. RLHF — Reinforcement Learning from Human Feedback

### The Three-Stage Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
─────────────────────────────────────
Pretrained LLM
     │
     ▼
Fine-tune on high-quality (prompt, response) pairs
(demonstrations from human labelers of ideal behavior)
     │
     ▼
SFT Model  ← starting point for RLHF

Stage 2: Reward Model Training
──────────────────────────────
Same prompt → SFT model generates multiple responses
Human labelers rank: response A > response B > response C
     │
     ▼
Train a reward model R_θ to predict human preferences
(separate model, often same architecture as SFT)
     │
     ▼
Reward Model R_θ(prompt, response) → scalar score

Stage 3: RL Fine-Tuning with PPO
─────────────────────────────────
SFT model (now called "policy" π_θ)
     │
     ▼
For each prompt: generate response, score with R_θ
     │
     ▼
Optimize π_θ to maximize E[R_θ(prompt, response)]
  + KL penalty to prevent diverging from SFT
     │
     ▼
Aligned Model (InstructGPT, ChatGPT, Claude 1, etc.)
```

### Why This Worked: InstructGPT (2022)

OpenAI's InstructGPT showed that a **1.3B RLHF-aligned model was preferred over a 175B GPT-3** model by human raters 71% of the time. Alignment quality > raw scale for practical usefulness.

---

## 3. The Reward Model

### Architecture

The reward model is typically initialized from the SFT model with the language modeling head replaced by a **scalar regression head**:

```
SFT Model (Transformer)
     │
     ▼
[Last token representation]  ∈ ℝ^d
     │
     ▼
Linear(d → 1)
     │
     ▼
r = scalar reward
```

### Bradley-Terry Preference Model

Given two responses $y_w$ (winner/preferred) and $y_l$ (loser/rejected) for prompt $x$, the probability that $y_w$ is preferred follows the **Bradley-Terry model**:

$$
P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

where $\sigma$ is the sigmoid function and $r_\theta$ is the reward model.

### Reward Model Loss

The reward model is trained with a **pairwise ranking loss** (binary cross-entropy on preferences):

$$
\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]
$$

**Intuition**: Maximize the probability of correctly identifying the preferred response. The loss decreases as the reward gap $r_\theta(x, y_w) - r_\theta(x, y_l)$ grows.

### Worked Example

```
Prompt: "Explain photosynthesis"

Response A (y_w): "Photosynthesis is the process by which plants convert
  sunlight, CO₂, and water into glucose and oxygen using chlorophyll."
→ r_θ(x, y_w) = 4.2

Response B (y_l): "Plants make food from sun."
→ r_θ(x, y_l) = 1.8

Preference probability: σ(4.2 - 1.8) = σ(2.4) = 0.917
Loss for this pair: -log(0.917) = 0.087  ← very small (correct prediction)

If r_θ(x, y_w) = 2.0 and r_θ(x, y_l) = 1.9 (barely distinguishable):
  σ(0.1) = 0.525
  Loss: -log(0.525) = 0.644  ← large (uncertain prediction, needs more training)
```

---

## 4. PPO for LLM Fine-Tuning

### Why RL for Text Generation?

Text generation is a **sequential decision process**:
- State = (prompt + tokens generated so far)
- Action = next token to generate
- Policy = the language model $\pi_\theta$
- Reward = obtained only at the end of generation (from reward model)

This is exactly the reinforcement learning setup. Standard supervised learning can't optimize for end-of-sequence rewards directly.

### PPO (Proximal Policy Optimization)

PPO is the RL algorithm used in RLHF. The core objective is to maximize expected reward while keeping the policy close to the old policy (stability):

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ — probability ratio (new vs old policy)
- $\hat{A}_t$ — advantage estimate (how much better this action was than expected)
- $\epsilon$ — clipping range (typically 0.2)

### The Clipping Trick — ASCII Explanation

```
Advantage Â > 0 (action was good — increase its probability):

    Objective
        │
    ----+----   clip ceiling (1 + ε)
        │       ← PPO stops here, prevents too-large updates
        │
    ----+----   ratio = 1 (old policy)
        │
    ════╪════  min(ratio × Â, clipped × Â)
        │
  0.8  1.0  1.2  1.4 → ratio r_t
  
If ratio > 1 + ε: clip → no more gradient signal (policy change already large enough)
If ratio < 1 - ε: clip → no more gradient signal (don't shrink good actions too much)
```

**Why clipping**: Without it, a single update could dramatically change the policy, leading to instability. PPO bounds the per-step update size.

### LLM-Specific PPO

For LLMs, each "episode" is one response generation:

```
1. Sample prompt x from dataset
2. Generate response y ~ π_θ(· | x)  (forward pass, autoregressive)
3. Score with reward model: r = R_φ(x, y)
4. Compute advantage Â using a learned value function V_ψ(x, y_<t)
5. Update θ to maximize PPO objective + KL penalty - value loss
```

The **value function** $V_\psi$ estimates expected future reward from position $t$ — it's an additional head added to the LLM during RLHF.

---

## 5. The KL Penalty — Preventing Reward Hacking

### The Problem: Reward Hacking

Without constraints, the policy will **exploit the reward model**:

```
Reward model trained on: "responses should be comprehensive and clear"
Adversarial optimization finds: extremely long, verbose, repetitive responses
  → Reward model gives high scores (never saw this distribution during training)
  → Response quality for humans is terrible

This is reward hacking — the model finds a policy that scores high on the 
imperfect reward model but doesn't align with actual human preferences.
```

### The KL Penalty Solution

Add a KL divergence penalty between the current policy and the SFT reference policy:

$$
\mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[R_\phi(x, y) - \beta \cdot \text{KL}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)\right]
$$

Expanding the KL term per token:

$$
\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \sum_t \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}
$$

The **per-token reward** used in PPO is:

$$
r_t = \begin{cases} R_\phi(x, y) - \beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} & t = T \text{ (last token)} \\ -\beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} & t < T \end{cases}
$$

### Effect of β

| β value | Behavior |
|---------|---------|
| β = 0 | No KL penalty — pure reward maximization → reward hacking |
| β small (0.01–0.1) | Mostly follows reward, small constraint → InstructGPT uses β ≈ 0.02 |
| β large (1+) | Stays close to SFT model, minimal reward optimization |
| β → ∞ | No change from SFT model |

**Typical values**: InstructGPT used β ≈ 0.02. Higher β means more conservative alignment.

---

## 6. DPO — Direct Preference Optimization

### The Key Insight

RLHF requires training a **separate reward model** and running **expensive PPO**. DPO (Rafailov et al., 2023) shows that the optimal policy under the RLHF objective can be expressed **analytically** — no reward model or RL needed.

### Deriving DPO

Starting from the RLHF objective:

$$
\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[R(x,y)\right] - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

The **optimal policy** has a closed-form solution:

$$
\pi^*(y | x) = \frac{\pi_{\text{ref}}(y | x) \exp(R(x, y) / \beta)}{Z(x)}
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp(R(x, y) / \beta)$ is the partition function.

Rearranging to express the **reward in terms of the policy**:

$$
R(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

### DPO Loss

Substituting into the Bradley-Terry preference model and noting that $\log Z(x)$ cancels (it appears in both $y_w$ and $y_l$ terms):

$$
\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]}
$$

**What this does**: Increase the log-likelihood of preferred response $y_w$ relative to the reference policy, while decreasing the log-likelihood of rejected response $y_l$.

### DPO Training Loop (Conceptual)

```
No separate reward model needed!

For each batch of (x, y_w, y_l) preference triples:

1. Forward pass through π_θ (trainable):
   → log π_θ(y_w | x)  and  log π_θ(y_l | x)

2. Forward pass through π_ref (frozen SFT model):
   → log π_ref(y_w | x)  and  log π_ref(y_l | x)

3. Compute log-ratios:
   δ_w = log π_θ(y_w | x) - log π_ref(y_w | x)  (how much policy shifted on winner)
   δ_l = log π_θ(y_l | x) - log π_ref(y_l | x)  (how much policy shifted on loser)

4. Loss = -log σ(β · (δ_w - δ_l))
   → Minimize loss = maximize (δ_w - δ_l)
   → Policy increases winner probability more than loser probability, relative to reference

5. Backprop through π_θ only (π_ref is frozen)
```

### Worked Numerical Example

```
β = 0.1, one (x, y_w, y_l) triple:

Before training (π_θ = π_ref, so all ratios = 0):
  δ_w = 0, δ_l = 0
  Loss = -log σ(0.1 × (0 - 0)) = -log σ(0) = -log(0.5) = 0.693

After one gradient step:
  δ_w = +1.2  (policy moved toward y_w)
  δ_l = -0.8  (policy moved away from y_l)
  Loss = -log σ(0.1 × (1.2 - (-0.8)))
       = -log σ(0.1 × 2.0)
       = -log σ(0.2)
       = -log(0.550)
       = 0.598  ← lower, improved

Converged state:
  δ_w = +5.0, δ_l = -3.0
  Loss = -log σ(0.1 × 8.0) = -log σ(0.8) = -log(0.690) = 0.371
```

The implicit reward learned by DPO is $\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ — no explicit reward model needed.

---

## 7. RLHF vs DPO — Comparison

### Side-by-Side

| Dimension | RLHF (PPO) | DPO |
|-----------|-----------|-----|
| **Stages** | 3 (SFT → RM → PPO) | 2 (SFT → DPO) |
| **Reward model** | Separate trained RM | Implicit (no explicit RM) |
| **RL required** | Yes (PPO with value function) | No (supervised-style loss) |
| **Memory** | 4 models in memory (policy, ref, RM, value) | 2 models (policy, ref) |
| **Compute** | ~10–20× more than SFT | ~2–3× more than SFT |
| **Stability** | Tricky (PPO tuning required) | Stable (no RL instability) |
| **Data** | Preference pairs (+ online generation) | Preference pairs only (offline) |
| **Quality** | Slightly better on complex tasks | Competitive, often slightly worse on RLHF-hard tasks |
| **Flexibility** | Can use online RL (explore + feedback) | Offline only (fixed dataset) |

### When to Use Which

```
Use RLHF (PPO) when:
  ✓ You have a strong reward signal (human raters or verifiable tasks)
  ✓ Online exploration helps (math, code — verifiable correctness)
  ✓ You have compute budget for 4 models + RL training
  ✓ Quality is paramount (frontier models: GPT-4, Claude 3)

Use DPO when:
  ✓ You have a fixed preference dataset
  ✓ Compute is limited (startups, research)
  ✓ You want stable, reproducible training
  ✓ Competitive quality is sufficient (most open-source models: Llama 2 Chat, Zephyr, Mistral Instruct)
```

### The Core Trade-off

```
RLHF with PPO:
  Reward model R_φ → imperfect proxy → reward hacking possible
  Online generation → covers distribution the model actually produces
  More expensive but more flexible

DPO:
  No explicit reward model → cannot hack something that doesn't exist
  Offline data only → may not cover failure modes the policy encounters
  Simpler but less adaptive
```

---

## 8. Constitutional AI & RLAIF

### Constitutional AI (Anthropic, 2022)

**The problem with RLHF**: Human labelers must rate thousands of potentially harmful outputs. This is slow, expensive, and exposes humans to toxic content.

**CAI's solution**: Replace human feedback with **AI feedback** guided by a constitution (a set of principles).

### The Two-Stage Process

```
Stage 1: Supervised Learning from AI Feedback (SL-CAF)
───────────────────────────────────────────────────────
1. "Red-teaming" prompts elicit harmful responses from the initial model
2. The model is asked to critique its own response against the constitution:
   "Identify specific ways in which the response is harmful, unethical, or toxic."
3. The model revises the response to be more harmless:
   "Please rewrite the response to remove any harmful content."
4. These (original prompt, revised response) pairs are used for SFT

Stage 2: RL from AI Feedback (RLAIF)
──────────────────────────────────────
1. Generate pairs of responses
2. Ask the model (or a stronger model) to choose the more harmless response
   guided by the constitutional principles
3. Use these AI preference labels to train a reward model
4. Use the reward model for PPO (like standard RLHF, but with AI labels)
```

### The Constitution

A set of principles like:

```
1. "Choose the response that is least likely to contain harmful content"
2. "Choose the response that is most honest and acknowledges uncertainty"
3. "Choose the response that is most helpful while refusing to assist 
    with clearly harmful tasks"
```

The model evaluates responses against these principles and provides preference labels.

### Why It Works

- **Scalable**: AI generates far more labels than humans at lower cost
- **Consistent**: AI applies the same principles consistently; humans vary
- **Transparent**: The constitution is explicit — you know what the model is optimized for
- **Iterative**: Can update the constitution without retraining human labelers

### RLAIF vs RLHF

| | RLHF | RLAIF / CAI |
|--|------|-------------|
| Labels from | Human raters | AI model (possibly same or stronger) |
| Scale | Limited by human bandwidth | Virtually unlimited |
| Consistency | Variable (human subjectivity) | High (same model, same principles) |
| Bias | Human cultural bias | AI model bias |
| Cost | $50–200 per 1K preferences | <$1 per 1K preferences |
| Quality | Human judgment is gold standard | Often 80–95% of human quality |

---

## 9. ORPO, SimPO & Modern Variants

### ORPO (Odds Ratio Preference Optimization, 2024)

**Problem with DPO**: Still requires a separate SFT stage. ORPO eliminates the two-stage process by adding a preference loss **directly to the SFT loss**:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

where the odds ratio loss is:

$$
\mathcal{L}_{\text{OR}} = -\log \sigma\!\left(\log \frac{\text{odds}_\theta(y_w | x)}{\text{odds}_\theta(y_l | x)}\right), \quad \text{odds}(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}
$$

**Advantage**: Single training stage, no reference model needed.

### SimPO (Simple Preference Optimization, 2024)

SimPO simplifies DPO by:
1. Removing the reference model (no $\pi_{\text{ref}}$ needed)
2. Using **length-normalized** log-likelihood as the reward
3. Adding a margin γ to ensure a reward gap between winner and loser:

$$
\mathcal{L}_{\text{SimPO}} = -\log \sigma\!\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w | x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l | x) - \gamma\right)
$$

Length normalization prevents the model from preferring shorter responses.

### Summary of Alignment Methods

```
Training Complexity (decreasing) →
RLHF (PPO) > DPO > SimPO > ORPO

Quality (approximately) →
RLHF (PPO) ≥ DPO ≈ SimPO > ORPO

Data Requirements:
All methods need: (prompt, preferred_response, rejected_response) triples
RLHF also needs: online generation during PPO
```

---

## 10. Preference Data Collection

### Data Format

All methods require preference datasets of the form:

```json
{
  "prompt": "Explain quantum entanglement simply",
  "chosen": "Quantum entanglement is like two coins...",
  "rejected": "Quantum mechanics involves Schrödinger equations..."
}
```

### Sources of Preference Data

| Source | Scale | Quality | Cost |
|--------|-------|---------|------|
| Human labelers (MTurk) | Medium | Variable | Moderate |
| Expert annotators | Small | High | Expensive |
| Existing user thumbs up/down | Large | Noisy | Free |
| AI-generated (RLAIF/CAI) | Huge | Good (varies) | Low |
| Synthetic (stronger model judges weaker model) | Huge | Good | Low |

### Key Public Preference Datasets

| Dataset | Size | Source | Used By |
|---------|------|--------|---------|
| **Anthropic HH-RLHF** | 161K | Human raters on helpful+harmless | Widely used |
| **OpenAssistant** | 161K | Community labelers | Falcon, many open models |
| **UltraFeedback** | 64K | GPT-4 rates 4 models | Zephyr, Mistral models |
| **Nectar** | 183K | GPT-4 preference labels | Open-source research |
| **Orca DPO Pairs** | 12.9K | Distilled from GPT-4 | Orca series |

### Data Quality Issues

```
Problem 1 — Label noise:
  Humans disagree 20-30% of the time on ambiguous pairs
  Solution: Multiple labelers + majority vote, or confidence filtering

Problem 2 — Distribution mismatch:
  Preference data from one domain (customer service) doesn't 
  transfer well to another (medical advice)
  Solution: domain-specific data collection

Problem 3 — Sycophancy in AI labels:
  AI labelers prefer longer, more confident-sounding responses
  regardless of actual accuracy
  Solution: Use specialized judge prompts, verify with humans on a subset
```

---

## 11. Alignment Tax & Capability Preservation

### The Alignment Tax

Fine-tuning for alignment typically hurts **raw capability**:

| Capability | Change After RLHF |
|-----------|------------------|
| Instruction following | ↑↑ (major improvement) |
| Safety/harmlessness | ↑↑ (major improvement) |
| Open-domain QA (MMLU) | ↓ slight (-1 to -3%) |
| Code generation | ↓ slight (-2 to -5%) |
| Math reasoning | ↓ slight to moderate |
| Verbosity | ↑ (models become wordier) |

**Why**: The model is optimized for human-rated quality, which may not correlate with benchmark performance. Also, KL penalty constrains the policy from making aggressive capability-improving updates.

### Mitigation: Iterative RLHF

Training in iterations with diverse prompts and refreshing the reward model helps:

```
Iteration 1: RLHF on helpfulness data → gains in following instructions
Iteration 2: Add RLHF on safety data → gains in harmlessness
Iteration 3: Re-inject SFT data alongside RLHF → capability preservation
```

### The Helpful, Harmless, Honest (3H) Framework

Anthropic's framing for what alignment should achieve:

- **Helpful**: Provides genuine value; completes tasks effectively
- **Harmless**: Avoids toxic, dangerous, or misleading outputs
- **Honest**: Calibrated uncertainty; acknowledges limitations; doesn't deceive

These often conflict:

```
Tension: helpful vs harmless
  "How do I pick a lock?" → helpful answer is dangerous for burglars
  Solution: context-dependent policy (mentions locksmith scenarios)

Tension: helpful vs honest  
  User: "Is my business idea good?"
  Helpful → encouragement; Honest → realistic critique
  Solution: calibrated honesty with constructive framing
```

---

## 12. Interview Questions & Answers

### Q1: What are the three stages of RLHF? Why is each stage necessary?

**Answer**: RLHF has three stages. (1) **Supervised Fine-Tuning (SFT)**: Fine-tune the pretrained model on high-quality human demonstrations to get a competent starting policy. Without SFT, the base model generates incoherent responses that the reward model can't meaningfully score. (2) **Reward Model Training**: Train a reward model on human preference comparisons (response A vs B, which is better?). This captures human values in a learnable proxy function. We can't define the reward function manually — it's too complex. (3) **PPO Fine-Tuning**: Use the reward model's scores as the reward signal in RL to optimize the policy, subject to a KL penalty preventing reward hacking. We need RL (not supervised learning) because the reward is only available at the end of generation, not at each token.

### Q2: Derive the DPO loss from first principles. What problem does it solve compared to RLHF?

**Answer**: Start from the RLHF objective: $\max_\pi \mathbb{E}[R(x,y)] - \beta \text{KL}(\pi \| \pi_{\text{ref}})$. The optimal policy is $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(R(x,y)/\beta)$, which gives $R(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. Substituting into the Bradley-Terry model $P(y_w \succ y_l) = \sigma(R(y_w) - R(y_l))$, the partition function $Z(x)$ cancels, giving: $\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$. This eliminates the need for a separate reward model and PPO training — the reward is **implicit** in the policy ratio. Problems solved: (1) no expensive RL training, (2) no reward hacking since there's no explicit reward to hack, (3) 2× lower memory (only 2 models vs 4), (4) more stable training.

### Q3: What is the KL penalty in RLHF? What happens if β = 0 vs β = ∞?

**Answer**: The KL penalty $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ penalizes the current policy for diverging from the SFT reference policy. It prevents **reward hacking** — without it, the policy would find adversarial inputs that score high on the (imperfect) reward model but are low quality for humans. At **β = 0**: pure reward maximization → reward hacking, degenerate outputs (repeated phrases, excessive length, etc.). At **β → ∞**: policy cannot change from the SFT model regardless of reward. At typical values **β ≈ 0.02–0.1**: the policy improves on the reward signal while staying in-distribution. The KL term also serves as a regularizer, preventing catastrophic forgetting of SFT capabilities.

### Q4: Why is PPO used for RLHF rather than simpler RL algorithms like REINFORCE?

**Answer**: REINFORCE uses the raw reward signal as the gradient weight, which has extremely high variance in text generation (a 100-token response has $|V|^{100}$ possible sequences). High variance → unstable training, slow convergence. PPO addresses this with: (1) **Clipping**: limits the per-step policy change, preventing large destabilizing updates; (2) **Value function baseline**: $\hat{A}_t = r_t - V_\psi(s_t)$ subtracts a learned baseline from the reward, drastically reducing variance (variance of advantage ≪ variance of raw reward); (3) **Multiple epochs per batch**: PPO reuses each batch of experience for several gradient updates (unlike REINFORCE which discards after one), improving sample efficiency. Together, PPO provides the stability needed for LLMs which have millions of parameters and long episode horizons.

### Q5: What is Constitutional AI? How does it scale alignment beyond human labeling?

**Answer**: Constitutional AI (Anthropic, 2022) replaces human preference labels with AI-generated labels guided by explicit principles (a "constitution"). Stage 1 (SL-CAF): red-teaming prompts elicit harmful outputs, the model critiques and revises them using the constitution, and revised responses become SFT data. Stage 2 (RLAIF): the model generates preference labels by evaluating which response better satisfies the constitutional principles — these labels train a reward model for standard PPO. Key scaling advantages: (1) AI labels at 100–1000× lower cost than humans; (2) consistent application of principles (no human disagreement); (3) transparent — you can inspect and modify the constitution; (4) avoids exposing human labelers to harmful content. The trade-off is that AI labels introduce AI biases (verbosity preference, sycophancy) that human labels wouldn't. In practice, CAI achieves ~90% of the quality of human-labeled RLHF at 1% of the cost.

### Q6: Compare DPO and RLHF on a real production scenario. When would you choose each?

**Answer**: For a **production chatbot at a startup** with limited compute and a fixed preference dataset, I'd choose **DPO**: 2-stage training (SFT → DPO), only 2 models in memory, no PPO tuning required, deterministic and reproducible. Expected outcome: competitive with RLHF on most helpfulness metrics. For a **frontier model** (GPT-4 tier) where absolute quality matters and I have the compute budget, I'd choose **RLHF with online PPO**: the ability to generate new responses during training and get reward feedback covers distribution shift that offline DPO misses, and online RL can discover improvement strategies outside the training preference distribution. For **math or code tasks with verifiable correctness**, RLHF with a verifier-based reward (outcome reward model) is strictly better — DPO needs preference pairs, which are hard to define when there are clear right/wrong answers. In practice: ~80% of open-source aligned models use DPO (Zephyr, Mistral Instruct, Llama 2 Chat); frontier models (GPT-4, Claude) use some form of online RL.

### Q7: What is reward hacking? Give a concrete example and how the KL penalty mitigates it.

**Answer**: Reward hacking occurs when the policy finds a strategy that achieves high reward model scores without actually being better by human standards. **Example**: A reward model trained to prefer "comprehensive" responses might assign higher scores to longer outputs. The policy discovers that generating verbose, repetitive content scores high: "The answer is Paris. Paris is the answer. Paris, which is the answer to your question, is indeed Paris..." — nonsense that scores high on the imperfect reward model. **KL penalty mitigation**: The penalty $\beta \log \frac{\pi_\theta(y_t)}{\pi_{\text{ref}}(y_t)}$ is high when the policy deviates far from the reference. The repetitive "Paris" response has near-zero probability under the SFT reference model, so its KL cost is enormous — the total reward (reward score − β × KL) is lower than a genuine good response. The KL penalty effectively constrains the policy to the "reasonable text" manifold learned during pretraining and SFT, where reward hacking strategies are out-of-distribution.
