# Topic 25: Evaluation & Benchmarking

## Table of Contents
1. [Why Evaluation Is Hard for LLMs](#1-why-evaluation-is-hard-for-llms)
2. [Perplexity](#2-perplexity)
3. [BLEU](#3-bleu)
4. [ROUGE](#4-rouge)
5. [METEOR, ChrF & BERTScore](#5-meteor-chrf--bertscore)
6. [Putting Metrics Together: When to Use What](#6-putting-metrics-together-when-to-use-what)
7. [LLM Benchmarks: Static Evaluations](#7-llm-benchmarks-static-evaluations)
8. [Human Evaluation Methods](#8-human-evaluation-methods)
9. [Chatbot Arena & Elo Ranking](#9-chatbot-arena--elo-ranking)
10. [LLM-as-a-Judge](#10-llm-as-a-judge)
11. [Evaluating Specific Capabilities](#11-evaluating-specific-capabilities)
12. [Benchmark Contamination](#12-benchmark-contamination)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. Why Evaluation Is Hard for LLMs

### The Core Challenge

Traditional ML has clear metrics (accuracy, F1, AUC). LLMs generate **open-ended text** where:
- Multiple correct answers exist
- Quality is multidimensional (fluency, accuracy, helpfulness, safety, style)
- Ground truth is subjective or unavailable
- Tasks are diverse — one metric can't cover all capabilities

### The Evaluation Taxonomy

```
┌──────────────────────────────────────────────────┐
│              LLM Evaluation Methods               │
├────────────────┬────────────────┬────────────────┤
│  Automatic     │  Human         │  LLM-as-Judge  │
│  Metrics       │  Evaluation    │                │
├────────────────┼────────────────┼────────────────┤
│ • Perplexity   │ • Likert scale │ • Pointwise    │
│ • BLEU/ROUGE   │ • Pairwise     │   scoring      │
│ • BERTScore    │   comparison   │ • Pairwise     │
│ • Exact match  │ • Elo ranking  │   comparison   │
│ • Pass@k       │ • Annotation   │ • Reference-   │
│                │   agreement    │   guided       │
├────────────────┼────────────────┼────────────────┤
│ Cheap, fast,   │ Gold standard, │ Scalable,      │
│ reproducible   │ expensive,     │ moderate cost, │
│ but limited    │ slow           │ has biases     │
└────────────────┴────────────────┴────────────────┘
```

### Goodhart's Law in LLM Evaluation

> "When a measure becomes a target, it ceases to be a good measure."

This is pervasive in LLM evaluation:
- Models optimized for MMLU may memorize answers rather than truly understand
- Models trained to be verbose score higher on some LLM-as-Judge setups
- Benchmark scores improve while real-world utility stays flat

---

## 2. Perplexity

### Definition

Perplexity measures how well a language model predicts a held-out test set. For a sequence $w_1, w_2, \ldots, w_N$:

$$
\text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N}
$$

Equivalently, using the chain rule:

$$
\text{PPL}(W) = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)
$$

This is the **exponentiated average negative log-likelihood** — the geometric mean of the inverse token probabilities.

### Intuition

$$
\text{PPL} = 2^{H(P, Q)}
$$

where $H(P, Q)$ is the cross-entropy between the true distribution $P$ and the model distribution $Q$. Perplexity represents the **effective number of choices** the model is uncertain between at each step.

| Perplexity | Interpretation |
|-----------|---------------|
| 1 | Perfect prediction (model is certain and correct) |
| 10 | Model is choosing among ~10 equally likely options |
| 100 | Model is choosing among ~100 options (poor) |
| Vocabulary size $V$ | Random guessing |

### Example

```
Sentence: "The cat sat on the ___"

Good model:  P(mat) = 0.3, P(floor) = 0.2, P(rug) = 0.15, ...
             PPL for this position ≈ low (concentrated probability)

Bad model:   P(mat) = 0.001, P(floor) = 0.001, P(banana) = 0.001, ...
             PPL for this position ≈ high (spread probability)
```

### Limitations of Perplexity

| Limitation | Explanation |
|-----------|------------|
| **Not comparable across tokenizers** | Different tokenizations → different $N$ → different PPL |
| **Doesn't measure generation quality** | A model with low PPL can still generate poor text |
| **Ignores task performance** | Low PPL ≠ good at following instructions, reasoning, coding |
| **Sensitive to domain** | A model trained on code will have high PPL on poetry |
| **Doesn't capture factuality** | Confidently wrong predictions still yield low PPL |
| **Open-ended generation** | Multiple valid continuations exist — penalizing one valid choice is wrong |

**Bottom line**: Perplexity is useful for comparing models on the same data with the same tokenizer during pretraining. It is **insufficient** as a standalone evaluation for modern LLMs.

---

## 3. BLEU

### Definition

**BLEU** (Bilingual Evaluation Understudy) measures **precision** of n-gram overlap between a candidate and reference(s). Originally designed for machine translation.

### Modified Precision

Standard precision would overcount repeated n-grams. BLEU uses **clipped counts**:

$$
p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \min\!\left(\text{Count}(\text{n-gram}), \text{Max\_Ref\_Count}(\text{n-gram})\right)}{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \text{Count}(\text{n-gram})}
$$

Each n-gram's count is clipped to its maximum occurrence in any single reference.

### Brevity Penalty

Since BLEU is precision-based, a very short translation could score perfectly. The **brevity penalty (BP)** penalizes short candidates:

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}
$$

where $c$ = candidate length, $r$ = effective reference length.

### Final BLEU Score

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

Standard BLEU-4 uses $N = 4$ with uniform weights $w_n = 1/4$:

$$
\text{BLEU-4} = \text{BP} \cdot (p_1 \cdot p_2 \cdot p_3 \cdot p_4)^{1/4}
$$

### Worked Example

```
Reference:  "The cat is on the mat"
Candidate:  "The the the the the the"

Unigram precision (naive): 6/6 = 1.0  ← Everything matches "the"!

Clipped unigram precision:
  "the" appears 6 times in candidate, max 2 times in reference
  Clipped count = min(6, 2) = 2
  p₁ = 2/6 = 0.33  ← Much more reasonable

Reference:  "The cat is on the mat"
Candidate:  "The cat sat on the mat"

p₁ = 5/6  (all unigrams match except "sat" vs "is")
p₂ = 3/5  ("the cat", "on the", "the mat" match)
p₃ = 1/4  ("on the mat" matches)
p₄ = 0/3  (no 4-gram matches)

BLEU-4 = BP × (5/6 × 3/5 × 1/4 × 0/3)^{1/4} = 0  ← any p_n = 0 kills the score
```

### Limitations

| Limitation | Issue |
|-----------|-------|
| **No recall** | Doesn't penalize missing important content |
| **N-gram only** | "The dog bit the man" vs "The man bit the dog" score equally |
| **No synonyms** | "automobile" vs "car" gets zero credit |
| **Brittle at sentence level** | Designed for corpus-level, unreliable per-sentence |
| **Favors short, safe outputs** | Doesn't reward creativity or fluency |

### BLEU Calculation — Full Step-by-Step (BLEU-2)

```
Reference:  "the cat sat on the mat"   (6 tokens)
Hypothesis: "the cat sat on a mat"     (6 tokens)

Step 1 — Unigram precision (clipped):
  Hypothesis unigrams: {the:1, cat:1, sat:1, on:1, a:1, mat:1}
  Reference  unigrams: {the:2, cat:1, sat:1, on:1, the:2, mat:1}

  Matches (clipped to reference max):
    the → min(1, 2) = 1  ✓
    cat → min(1, 1) = 1  ✓
    sat → min(1, 1) = 1  ✓
    on  → min(1, 1) = 1  ✓
    a   → min(1, 0) = 0  ✗  (not in reference)
    mat → min(1, 1) = 1  ✓

  p₁ = (1+1+1+1+0+1) / 6 = 5/6 ≈ 0.833

Step 2 — Bigram precision (clipped):
  Hypothesis bigrams: (the,cat), (cat,sat), (sat,on), (on,a), (a,mat)   5 bigrams
  Reference  bigrams: (the,cat), (cat,sat), (sat,on), (on,the), (the,mat)

  Matches:
    (the,cat) ✓, (cat,sat) ✓, (sat,on) ✓
    (on,a) ✗, (a,mat) ✗

  p₂ = 3/5 = 0.600

Step 3 — Brevity Penalty:
  c = 6 (hypothesis), r = 6 (reference) → c = r → BP = 1.0

Step 4 — BLEU-2:
  BLEU-2 = BP × (p₁ × p₂)^(1/2)
          = 1.0 × (0.833 × 0.600)^0.5
          = 1.0 × (0.500)^0.5
          = 1.0 × 0.707
          ≈ 0.707

The one wrong word ("a" instead of "the") drops BLEU-2 from 1.0 to 0.707.
BLEU-4 would be 0 here (no 4-gram overlap at all from just one wrong word in 6).
```

---

## 4. ROUGE

### Definition

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures **recall** of n-gram overlap. Originally designed for summarization.

### ROUGE Variants

**ROUGE-N** (n-gram recall):

$$
\text{ROUGE-N} = \frac{\sum_{\text{n-gram} \in \text{Reference}} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{\text{n-gram} \in \text{Reference}} \text{Count}(\text{n-gram})}
$$

**ROUGE-L** (Longest Common Subsequence):

$$
R_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{m}, \quad P_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{n}
$$

$$
\text{ROUGE-L} = F_{\text{lcs}} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}
$$

where $m$ = reference length, $n$ = candidate length, $\beta$ typically set to favor recall.

**ROUGE-LSum**: Applies ROUGE-L at the sentence level, then aggregates (for multi-sentence summaries).

### BLEU vs ROUGE: Side-by-Side

| | BLEU | ROUGE |
|--|------|-------|
| **Orientation** | Precision (what % of candidate is in reference) | Recall (what % of reference is in candidate) |
| **Task** | Machine translation | Summarization |
| **Key question** | "Is the generated text accurate?" | "Does the summary cover the source?" |
| **Brevity** | Has brevity penalty | No explicit brevity penalty |
| **Score range** | 0–1 (corpus), 0–100 (reported) | 0–1 |

### Worked Example

```
Reference:  "The cat sat on the mat in the room"
Candidate:  "The cat on the mat"

BLEU (precision focus):
  Unigrams in candidate that appear in reference: 5/5 = 1.0
  → BLEU says: "Everything in the candidate is correct" ✓

ROUGE-1 (recall focus):
  Unigrams in reference that appear in candidate: 5/8 = 0.625
  → ROUGE says: "The candidate is missing content" (sat, in, room)
```

### ROUGE-L — LCS Worked Example

**Task**: Evaluate a 2-sentence summary.

```
Reference:  "the cat sat on the mat"
Candidate:  "the cat on the floor"

Step 1 — Find Longest Common Subsequence (LCS):
  LCS must be in-order (not necessarily contiguous)

  Reference: the  cat  sat  on  the  mat
  Candidate: the  cat  on   the floor

  LCS options:
    "the cat on the" = length 4  ← longest
    (matches: the[1]↔the[1], cat[2]↔cat[2], on[4]↔on[3], the[5]↔the[4])

  LCS length = 4

Step 2 — Compute Recall and Precision:
  m = |reference| = 6
  n = |candidate|  = 5

  R_lcs = LCS / m = 4/6 = 0.667
  P_lcs = LCS / n = 4/5 = 0.800

Step 3 — F-score (β = 1, equal weight):
  ROUGE-L = 2 × R × P / (R + P)
           = 2 × 0.667 × 0.800 / (0.667 + 0.800)
           = 1.067 / 1.467
           ≈ 0.727

Compare to ROUGE-1:
  Matching unigrams: {the, cat, on, the} = 4 matched in reference  
  ROUGE-1 recall = 4/6 = 0.667  (same as R_lcs here since LCS = token recall)
  But ROUGE-L also penalizes out-of-order matches; ROUGE-1 doesn't.
```

**Key insight**: ROUGE-L captures that order matters. A candidate that has all the right words in wrong order scores lower on ROUGE-L than ROUGE-1.

---

## 5. METEOR, ChrF & BERTScore

### METEOR
- Extends BLEU with stemming + synonym matching (via WordNet) and a fluency penalty for non-contiguous matches
- Recall-weighted F-score: $F = \frac{10PR}{R + 9P}$, final score multiplied by $(1 - \text{fragmentation penalty})$
- Correlates better with human judgment than BLEU for translation; rarely asked at depth in MNC interviews

### ChrF
- Character n-gram F-score (precision + recall at character level, averaged over n=1..6)
- No tokenization dependency; robust to morphological variation ("walking" partially matches "walked")
- Useful for morphologically rich languages; standard secondary metric in MT evaluation

### BERTScore
- Encodes reference + candidate with BERT, greedy-matches token embeddings by cosine similarity
- Captures synonymy: "automobile" ↔ "car" scores ~0.89 cosine — BLEU/ROUGE score 0 for this
- Better human correlation than n-gram metrics; use when semantic equivalence matters (summarization, paraphrase)

---

## 6. Putting Metrics Together: When to Use What

| Metric | Best For | Advantages | Limitations |
|--------|---------|------------|-------------|
| **Perplexity** | Pretraining evaluation, LM comparison | Cheap, well-defined | Not task-specific, tokenizer-dependent |
| **BLEU** | Machine translation | Standard, interpretable | Precision-only, no synonyms |
| **ROUGE** | Summarization | Recall-focused, captures coverage | No semantics, surface-level |
| **METEOR** | Translation (better than BLEU) | Synonyms, stemming, word order | Slow, English-centric WordNet |
| **ChrF** | Translation (multilingual) | Language-agnostic, morphology-robust | Character-level can miss meaning |
| **BERTScore** | Any generation task | Semantic similarity, paraphrase-aware | Compute cost, model-dependent |
| **Exact Match** | QA, code generation | Simple, unambiguous | Too strict for open-ended tasks |
| **Pass@k** | Code generation | Accounts for stochasticity | Expensive (multiple samples) |

### The Modern Consensus

For evaluating LLMs in 2024+:
1. **Automatic metrics** (BLEU, ROUGE) are baselines — necessary but not sufficient
2. **BERTScore** or embedding-based metrics for better semantic evaluation
3. **LLM-as-Judge** for scalable qualitative evaluation
4. **Human evaluation** remains the gold standard for final decisions
5. **Task-specific benchmarks** (MMLU, HumanEval, etc.) for capability profiling

---

## 7. LLM Benchmarks: Static Evaluations

### Major Benchmarks

| Benchmark | What It Tests | Format | Size | Key Detail |
|-----------|-------------|--------|------|-----------|
| **MMLU** | Multitask knowledge (57 subjects) | 4-choice MCQ | 14K | Ranges from elementary to professional level |
| **HellaSwag** | Commonsense reasoning | 4-choice completion | 10K | Adversarially filtered to be hard for models |
| **ARC** | Science reasoning (grade 3-9) | MCQ | 7.8K | Challenge set filters out easy questions |
| **TruthfulQA** | Truthfulness | Generation + MCQ | 817 | Tests for common misconceptions |
| **WinoGrande** | Coreference / commonsense | Fill-in-the-blank | 44K | Pronoun resolution requiring world knowledge |
| **GSM8K** | Grade school math | Free-form | 8.5K | Chain-of-thought reasoning needed |
| **MATH** | Competition math | Free-form | 12.5K | Much harder; tests formal reasoning |
| **HumanEval** | Code generation | Function completion | 164 | Python only; pass@k metric |
| **MBPP** | Basic programming | Function + tests | 974 | Broader than HumanEval |
| **BBH** | Hard BIG-Bench tasks | Mixed | 6.5K | 23 tasks where LLMs previously failed |
| **IFEval** | Instruction following | Constrained generation | 541 | Verifiable format constraints |
| **GPQA** | Graduate-level science | MCQ | 448 | Expert-validated, very hard |

### The Open LLM Leaderboard (HuggingFace)

Originally evaluated models on 6 benchmarks: MMLU, HellaSwag, ARC, TruthfulQA, WinoGrande, GSM8K.

**Leaderboard v2** replaced saturated benchmarks with harder ones: MMLU-Pro, GPQA, MuSR, MATH, BBH, IFEval.

### How Benchmarks Are Administered

**Few-shot evaluation** (standard for most benchmarks):

```
Prompt format (5-shot MMLU example):

Question: What is the capital of France?
A) London  B) Paris  C) Berlin  D) Madrid
Answer: B

Question: What is the boiling point of water?
A) 50°C  B) 100°C  C) 150°C  D) 200°C
Answer: B

[... 3 more examples ...]

Question: {test_question}
A) {option_a}  B) {option_b}  C) {option_c}  D) {option_d}
Answer:
```

The model's next-token probability over "A", "B", "C", "D" determines the answer.

### Benchmark Aggregation

**Average score** across benchmarks is misleading — it hides per-task variance. Better approaches:

- Report per-task scores alongside the average
- Use **radar charts** to visualize capability profiles
- Weight benchmarks by importance for your use case

```
                   MMLU
                  90│
                 ╱  │
    IFEval  85 ╱   │╲  88 HellaSwag
              ╱    │  ╲
             ╱     │   ╲
    MATH  72 ──────┼────── 91 ARC
              ╲    │   ╱
               ╲   │  ╱
    GPQA  65    ╲  │╱  82 GSM8K
                  ╲│
                   │
                 HumanEval 78

Model A: Strong on knowledge, weaker on hard reasoning
```

---

## 8. Human Evaluation Methods

### Likert Scale Rating

Annotators rate each response on a fixed scale:

```
Rate the response on each dimension (1-5):

Fluency:      [1]  [2]  [3]  [4]  [5]
Relevance:    [1]  [2]  [3]  [4]  [5]
Accuracy:     [1]  [2]  [3]  [4]  [5]
Helpfulness:  [1]  [2]  [3]  [4]  [5]
```

**Pros**: Rich multidimensional signal, simple to administer.
**Cons**: Subjective scales — one annotator's "4" is another's "3". Requires calibration.

### Pairwise Comparison

Annotators choose which response is better (A, B, or tie):

```
Query: "Explain quantum entanglement"

Response A: [Model X output]
Response B: [Model Y output]

Which response is better?  [A]  [B]  [Tie]
```

**Pros**: Easier and more reliable than absolute scoring — humans are better at comparisons than absolute ratings.
**Cons**: Doesn't provide absolute quality measures. Quadratic comparisons needed for $K$ models: $\binom{K}{2}$.

### Inter-Annotator Agreement

Measures how much annotators agree with each other:

**Cohen's Kappa** (2 annotators):

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ = observed agreement, $p_e$ = expected agreement by chance.

| Kappa Range | Interpretation |
|------------|---------------|
| < 0 | Worse than chance |
| 0.0 – 0.20 | Slight agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| 0.81 – 1.00 | Almost perfect agreement |

**Krippendorff's Alpha** (multiple annotators, handles missing data):

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

where $D_o$ = observed disagreement, $D_e$ = expected disagreement. Works with any number of annotators and any scale type (nominal, ordinal, interval, ratio). Target: $\alpha > 0.667$ for reliable annotations, $\alpha > 0.8$ for strong agreement.

### Best Practices for Human Evaluation

1. **Clear annotation guidelines** with examples of each score level
2. **Calibration round** — annotators discuss disagreements on a pilot set
3. **Minimum 3 annotators per item** to estimate agreement and aggregate via majority vote
4. **Randomize order** of model outputs to avoid position bias
5. **Blind evaluation** — annotators shouldn't know which model generated which response

---

## 9. Chatbot Arena & Elo Ranking

### The Problem with Static Benchmarks

Static benchmarks have fundamental limitations:
- **Contamination**: Models may have seen benchmark data during training
- **Saturation**: Top models all score >90%, making differentiation meaningless
- **Narrow scope**: MCQ benchmarks miss open-ended generation quality
- **Gaming**: Models can be optimized specifically for benchmark formats

### Chatbot Arena (LMSYS)

A **crowdsourced, live evaluation platform** where real users compare LLM outputs:

```
┌─────────────────────────────────────────────┐
│               Chatbot Arena                  │
│                                             │
│  User enters a prompt                       │
│            │                                │
│            ▼                                │
│  ┌──────────────┐  ┌──────────────┐        │
│  │   Model A    │  │   Model B    │        │
│  │  (anonymous) │  │  (anonymous) │        │
│  │              │  │              │        │
│  │  Response A  │  │  Response B  │        │
│  └──────────────┘  └──────────────┘        │
│            │                │               │
│            ▼                ▼               │
│     User votes: A wins / B wins / Tie      │
│            │                                │
│            ▼                                │
│     Elo ratings updated                    │
└─────────────────────────────────────────────┘
```

### Elo Rating System

Borrowed from chess. After each comparison:

**Expected score** for model A against model B:

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

**Rating update** after outcome $S_A$ (1 = win, 0.5 = tie, 0 = loss):

$$
R_A' = R_A + K(S_A - E_A)
$$

where $K$ is the update factor (typically 4-32).

**Example**:
```
Model A (Elo 1200) vs Model B (Elo 1400):
E_A = 1 / (1 + 10^(200/400)) = 1 / (1 + 3.16) = 0.24

If A wins (upset):
R_A' = 1200 + 32 × (1 - 0.24) = 1200 + 24.3 = 1224
R_B' = 1400 + 32 × (0 - 0.76) = 1400 - 24.3 = 1376

If B wins (expected):
R_A' = 1200 + 32 × (0 - 0.24) = 1200 - 7.7 = 1192
R_B' = 1400 + 32 × (1 - 0.76) = 1400 + 7.7 = 1408
```

Upsets cause large rating changes; expected outcomes cause small ones.

### Why Elo > Static Benchmarks

| Advantage | Explanation |
|----------|------------|
| **Real user prompts** | Tests what people actually ask, not curated benchmarks |
| **Contamination-resistant** | New prompts every day — can't be in training data |
| **Open-ended** | Tests generation quality, not just MCQ accuracy |
| **Self-calibrating** | Elo naturally ranks models on a continuous scale |
| **Constantly updated** | Rankings reflect current model versions |

### Limitations of Arena-Style Evaluation

- **User bias**: Users may prefer verbose, confident-sounding answers
- **Prompt distribution**: Skewed toward certain demographics/topics
- **Not reproducible**: Can't exactly replicate a ranking
- **Vote quality**: Some users vote randomly or with bias
- **Statistical confidence**: Rare matchups have high uncertainty

### Bradley-Terry Model

The Arena uses the **Bradley-Terry model** (a generalization of Elo):

$$
P(i \text{ beats } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}
$$

where $\beta_i$ is the strength parameter of model $i$. Estimated via maximum likelihood over all observed battles. More statistically principled than incremental Elo updates.

---

## 10. LLM-as-a-Judge

### Motivation

Human evaluation is expensive (~$1-5 per judgment) and slow. LLM-as-Judge uses a strong LLM (GPT-4, Claude) to evaluate other LLMs at scale.

### Evaluation Modes

**Pointwise scoring**:

```
Rate the following response on a scale of 1-10.

Question: {question}
Response: {response}

Evaluation criteria:
- Accuracy: Is the information correct?
- Completeness: Does it cover all aspects?
- Clarity: Is it easy to understand?

Score: ___
Justification: ___
```

**Pairwise comparison**:

```
Which response is better? Respond with "A", "B", or "Tie".

Question: {question}
Response A: {response_a}
Response B: {response_b}

Your judgment: ___
```

**Reference-guided**:

```
Given the reference answer, evaluate the response.

Question: {question}
Reference: {gold_answer}
Response: {model_response}

Score (1-5): ___
```

### Known Biases

| Bias | Description | Mitigation |
|------|-----------|-----------|
| **Position bias** | Prefers the first (or last) response in pairwise comparisons | Swap order and average, or use single-response evaluation |
| **Verbosity bias** | Prefers longer, more detailed responses | Explicitly instruct to ignore length; penalize unnecessary verbosity |
| **Self-enhancement bias** | LLM prefers outputs in its own style | Use a different model family as judge |
| **Leniency bias** | Tends to give high scores across the board | Use comparative ranking instead of absolute scores |
| **Sycophancy** | Agrees with the premise of the prompt | Validate with adversarial prompts |
| **Format bias** | Prefers well-formatted (markdown, bullet points) responses | Normalize formatting before judging |

### Calibration and Validation

**Always validate LLM-as-Judge against human annotations**:

1. Collect human judgments on a sample ($N \geq 200$)
2. Compute agreement between LLM judge and humans
3. Measure: Cohen's Kappa, Spearman correlation, accuracy on pairwise preferences
4. If agreement is low, revise the judging prompt or switch models

Typical agreement rates:
- GPT-4 as judge vs human preferences: **~80-85%** agreement
- This is comparable to inter-annotator agreement among humans (~80%)

### Multi-Judge Panels

Use multiple LLM judges and aggregate:

$$
\text{Score}_{\text{final}} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}_k
$$

Or use majority vote for pairwise decisions. Different judge models reduce individual biases.

---

## 11. Evaluating Specific Capabilities

### Factuality Evaluation

| Method | How It Works | Granularity |
|--------|-------------|-------------|
| **Claim decomposition** | Split response into atomic claims, verify each | Per-claim |
| **NLI-based** | Use NLI model to check entailment against source | Per-claim |
| **Self-consistency** | Generate $N$ times; inconsistent claims likely wrong | Per-response |
| **FActScore** | Decompose into atomic facts, check each against Wikipedia | Per-fact |
| **Citation verification** | Check if cited sources actually support claims | Per-citation |

**FActScore** (Factual Precision Score):

$$
\text{FActScore} = \frac{\text{Number of supported atomic facts}}{\text{Total number of atomic facts}}
$$

### Reasoning Evaluation

```
Types of reasoning benchmarks:

Mathematical:    GSM8K (grade school), MATH (competition), AIME
Logical:         BBH (Boolean expressions, causal judgment)
Commonsense:     HellaSwag, WinoGrande, PIQA
Scientific:      ARC, GPQA (graduate-level)
Multi-step:      MuSR (multi-step soft reasoning)
```

**Chain-of-thought evaluation**: Check both the **final answer** and the **reasoning process**. A correct answer with flawed reasoning is a concern (may break on harder problems).

### Instruction Following

**IFEval** (Instruction Following Evaluation): Tests verifiable format constraints:

```
"Write a poem about spring. The poem must:
1. Have exactly 4 stanzas
2. Each stanza must have 4 lines
3. Include the word 'bloom' at least 3 times
4. End with a question"
```

Evaluation is **programmatic** — check each constraint mechanically. No subjectivity.

### Safety Evaluation

| Benchmark | Tests |
|-----------|-------|
| **ToxiGen** | Implicit toxic language generation |
| **RealToxicityPrompts** | Toxicity when completing prompts |
| **BBQ** | Bias across 9 social dimensions |
| **XSTest** | Exaggerated safety behavior (over-refusal) |
| **HarmBench** | Standardized attack/defense evaluation |

The **dual failure mode**: Models can be unsafe (generates harmful content) or over-safe (refuses benign requests). Both are failures.

```
Under-refusal:  User asks for harmful instructions → Model complies   ✗
Over-refusal:   User asks about historical wars → "I can't discuss violence"  ✗
Correct:        Distinguish harmful intent from legitimate queries    ✓
```

---

## 12. Benchmark Contamination

### The Problem

If benchmark data appears in a model's training corpus, benchmark scores become meaningless — the model may have **memorized** answers rather than demonstrating genuine capability.

### Types of Contamination

| Type | Description | Detection Difficulty |
|------|-----------|---------------------|
| **Direct** | Exact benchmark examples in training data | Easy (n-gram matching) |
| **Indirect** | Paraphrased or reformulated versions | Hard |
| **Rephrased** | Same question, different wording | Very hard |
| **Capability overlap** | Training data teaches the exact skill tested | Impossible to distinguish from genuine learning |

### Detection Methods

**N-gram overlap**:

$$
\text{Contamination}(x) = \max_{d \in \mathcal{D}_{\text{train}}} \frac{|\text{n-grams}(x) \cap \text{n-grams}(d)|}{|\text{n-grams}(x)|}
$$

Flag examples where a high fraction of n-grams appear verbatim in training data.

**Canary strings**: Embed unique identifiable strings in benchmark data; if they appear in model outputs, contamination is confirmed.

**Performance gap analysis**: Compare performance on the original benchmark vs. rephrased versions:

$$
\Delta = \text{Score}_{\text{original}} - \text{Score}_{\text{rephrased}}
$$

Large $\Delta$ suggests memorization rather than understanding.

### Mitigation Strategies

| Strategy | Approach |
|----------|---------|
| **Private test sets** | Keep evaluation data secret (not published) |
| **Dynamic benchmarks** | Generate new test instances regularly |
| **Temporal cutoffs** | Use data created after training cutoff |
| **Decontamination** | Remove benchmark-similar data from training (imperfect) |
| **Live evaluation** | Chatbot Arena style — new prompts from real users |

### Notable Cases

- GPT-4's original technical report acknowledged possible contamination on some benchmarks
- Several open-source models showed suspiciously high scores that dropped dramatically on rephrased versions
- This has driven the shift toward **live, human-based evaluation** (Arena) as the most trusted signal

### Contamination — Worked Example (How Train/Test Overlap Inflates Scores)

```
Benchmark: MMLU — "What is the capital of France?"
Correct answer: "Paris"

Scenario A (Clean model):
  Training data: Wikipedia articles, news (no MMLU examples)
  Model must reason from general knowledge → 72% accuracy on MMLU

Scenario B (Contaminated model):
  Training data: Wikipedia + a dataset dump that included MMLU test questions
  The model saw: "Question: What is the capital of France? Answer: Paris"
  
  At eval time, model recognizes the question pattern:
  "What is the capital of France?" → pattern-matches to training example → "Paris"
  
  Result: 91% MMLU accuracy — but 19 points comes from memorization

How to detect:
  1. Hash the MMLU questions (MD5)
  2. Compare against all training data document fingerprints
  3. Find: 8.3% of MMLU questions appear verbatim in training data
  
  Controlled experiment:
    Original MMLU accuracy: 91%
    Rephrased MMLU ("Name the capital city of France"): 76%
    Gap = 15 points → strong evidence of contamination

Real example pattern (from research):
  Model A: MMLU = 88% (training data scraped from web in 2023)
  Model B: MMLU = 73% (carefully decontaminated training data)
  Model A's gap vs. rephrased: 14 points → likely contaminated
  Model B's gap vs. rephrased: 1 point → clean
```

**Interview answer**: "I'd detect contamination by (1) n-gram matching benchmark questions against training data, (2) measuring performance gap between original and rephrased versions, and (3) using temporal cutoffs — test on data created after the training cutoff date."

---

## 13. Interview Questions & Answers

### Q1: Why is perplexity not sufficient to evaluate LLMs? What are its limitations?

**Answer**: Perplexity measures how well a model predicts a held-out test set — it's the exponentiated average negative log-likelihood. While useful during pretraining as a training signal, it has critical limitations for evaluating modern LLMs. (1) **Not comparable across tokenizers**: different tokenizations change the sequence length $N$, making perplexity incomparable (a BPE vs SentencePiece tokenizer will give different values for the same model quality). (2) **Doesn't measure task performance**: a model with low perplexity can still fail at instruction following, reasoning, or factuality — it just means the model assigns high probability to the test text. (3) **Doesn't measure generation quality**: perplexity evaluates the probability of existing text, not the quality of text the model generates. (4) **Multiple valid continuations**: for open-ended tasks, many correct responses exist — penalizing one valid completion is inappropriate. (5) **Sensitive to domain**: a code model has high perplexity on poetry and vice versa. For these reasons, modern LLM evaluation relies on task-specific benchmarks, human evaluation, and LLM-as-Judge.

### Q2: How does BLEU differ from ROUGE? When do you use each?

**Answer**: **BLEU** is **precision-oriented**: it measures what fraction of n-grams in the candidate appear in the reference. It asks "Is everything the model generated correct?" It includes a brevity penalty to prevent trivially short outputs. **ROUGE** is **recall-oriented**: it measures what fraction of n-grams in the reference appear in the candidate. It asks "Did the model capture the key content?" Use **BLEU for translation** — precision matters because we want every translated word to be accurate. Use **ROUGE for summarization** — recall matters because we want the summary to cover the important points from the source. In practice, both metrics are limited: they're purely lexical (can't handle synonyms), sensitive to surface-level choices, and don't capture semantic meaning. For modern evaluation, they serve as cheap baselines alongside BERTScore (semantic) and LLM-as-Judge (holistic).

### Q3: What is the Chatbot Arena? Why is Elo ranking considered more reliable than static benchmarks?

**Answer**: Chatbot Arena (LMSYS) is a crowdsourced evaluation platform where users submit prompts to two anonymous models, see both responses, and vote for the better one. Elo ratings (borrowed from chess) are computed from these pairwise comparisons. It's considered more reliable for several reasons: (1) **Contamination-resistant** — prompts are novel, real-user queries that can't be in training data. (2) **Tests real-world usage** — covers the actual distribution of user needs, not curated benchmark tasks. (3) **Open-ended evaluation** — measures generation quality, helpfulness, and reasoning holistically, not just MCQ accuracy. (4) **Continuously updated** — rankings reflect current model versions and evolving user needs. (5) **Self-calibrating** — Elo naturally produces a continuous ranking with confidence intervals. Limitations include user biases (verbosity preference), non-reproducibility, and prompt distribution skew. The underlying Bradley-Terry model provides a statistically principled framework, and with 1M+ votes, the rankings are quite stable.

### Q4: What are the biases in LLM-as-a-Judge evaluation? How do you mitigate them?

**Answer**: Five major biases: (1) **Position bias** — the judge prefers whichever response appears first (or last) in pairwise comparison. Mitigation: evaluate in both orderings and average. (2) **Verbosity bias** — prefers longer, more detailed responses regardless of quality. Mitigation: explicitly instruct the judge to evaluate conciseness; penalize unnecessary detail. (3) **Self-enhancement bias** — LLMs prefer text in their own style. Mitigation: use a different model family as judge (e.g., Claude judging GPT outputs). (4) **Leniency/central tendency** — judges avoid extreme scores, clustering around 3-4 on a 5-point scale. Mitigation: use pairwise comparison instead of absolute scoring. (5) **Format bias** — prefers well-structured responses (markdown, bullets). Mitigation: normalize formatting before evaluation. **General mitigations**: validate against human judgments on a calibration set (target >80% agreement), use multi-judge panels with different models, and always report confidence intervals.

### Q5: Explain BERTScore. Why is it better than BLEU/ROUGE for evaluating generation quality?

**Answer**: BERTScore computes semantic similarity using contextual embeddings. It encodes reference and candidate tokens with BERT, computes pairwise cosine similarities, then uses greedy matching — each token in one sequence matches its most similar token in the other. Precision, recall, and F1 are computed from these maximum similarities. It's better than BLEU/ROUGE because: (1) **Captures synonymy** — "automobile" and "car" have high cosine similarity in BERT space, while BLEU gives zero credit. (2) **Handles paraphrases** — "The dog chased the cat" and "The cat was pursued by the dog" score highly despite different surface forms. (3) **Contextual** — the same word in different contexts gets different embeddings, capturing polysemy. (4) **Correlates better with human judgments** across multiple generation tasks. Limitations: it requires a good encoder model, is slower than n-gram metrics, and can be fooled by semantically similar but factually different text (e.g., "Paris is the capital of Germany" would score well against "Berlin is the capital of Germany" since the sentence structures are similar).

### Q6: How would you evaluate a RAG system? What metrics matter at each stage?

**Answer**: Evaluate each component then end-to-end. **Retrieval**: Recall@K (fraction of relevant documents in top-K, target >90%), MRR (mean reciprocal rank of first relevant doc), Precision@K. These measure whether the right information reaches the LLM. **Generation**: Faithfulness (does the answer only use retrieved information? Measured via NLI), answer relevance (does it address the question? LLM-as-Judge), completeness. **End-to-end**: Answer correctness against a curated QA set, citation accuracy. The **RAGAS framework** automates four metrics: context precision, context recall, faithfulness, answer relevance. For production, track online metrics: user satisfaction (thumbs up/down rate), follow-up question rate (lower is better — the first answer was sufficient), task completion rate. Maintain a test set of 200+ questions with gold answers, run weekly regression tests, and perform monthly human evaluation on a sample. Critically, evaluation should be **component-level** so you know whether to improve retrieval or generation.

### Q7: What is benchmark contamination? Why is it a serious problem?

**Answer**: Benchmark contamination occurs when evaluation data leaks into a model's training corpus. The model memorizes answers rather than demonstrating genuine capability, inflating scores artificially. It's serious because it **undermines trust** in the entire evaluation ecosystem — we can't compare models fairly if some have seen the test data. **Types**: Direct contamination (exact benchmark text in training data), indirect (paraphrased versions), rephrased (same question, different wording). **Detection**: N-gram overlap analysis between training data and benchmarks, canary strings, performance gap between original and rephrased versions (large gap suggests memorization). **Mitigation**: Private test sets, dynamic benchmarks (generate new instances regularly), temporal cutoffs (use data created after training), live evaluation (Arena-style). This problem has driven the field toward continuously refreshed evaluations, private held-out test sets, and human-based live evaluation as the most trusted signals.

### Q8: How do you evaluate safety and detect over-refusal in LLMs?

**Answer**: Safety evaluation has two failure modes: **under-refusal** (complying with harmful requests) and **over-refusal** (refusing benign requests). Both are measured. For under-refusal: benchmark suites like HarmBench test whether models resist adversarial attacks (jailbreaks, role-play attacks, encoding attacks). Red-teaming by domain experts systematically probes boundaries. For over-refusal: benchmarks like XSTest present borderline-but-safe queries (e.g., "How do I kill a process in Linux?") and measure whether the model appropriately responds. **Metrics**: refusal rate on harmful prompts (higher is better), refusal rate on benign prompts (lower is better), and the gap between them. **Toxicity**: evaluate generated text with toxicity classifiers (Perspective API, custom models) on both adversarial and natural prompts. **Bias**: BBQ benchmark tests bias across 9 social dimensions. The key insight is that safety evaluation requires testing both directions — you need both an attack benchmark and a benign benchmark to find the right balance.

### Q9: Compare pass@k for code evaluation with standard accuracy. Why is it needed?

**Answer**: Standard accuracy evaluates a single model output against a gold answer. For code generation, this is insufficient because: (1) LLMs are stochastic — the same prompt might produce a correct solution on one sample and an incorrect one on another. (2) Multiple valid implementations exist — there's no single "correct" answer. **Pass@k** measures the probability that at least one of $k$ generated samples passes all test cases. The unbiased estimator (Codex paper):

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ = total samples generated, $c$ = number that pass. In practice, generate $n$ samples (e.g., 200), count $c$ correct, compute pass@k for $k = 1, 5, 10$. **Pass@1** measures reliability (does the model usually get it right?), while **pass@10** measures capability (can the model ever get it right, with retries?). The gap between pass@1 and pass@10 indicates how much sampling/selection strategies like best-of-n could improve practical performance.

### Q10: Design an evaluation pipeline for a production LLM application. What layers would you include?

**Answer**: Five-layer evaluation stack, from cheapest/fastest to most expensive/reliable. **Layer 1 — Deterministic unit tests**: Format validation (valid JSON, length limits), required field presence, regex patterns, citation format. Run on every inference, zero marginal cost. **Layer 2 — Automatic metrics**: ROUGE/BERTScore against reference answers, embedding similarity, NLI-based faithfulness checking. Run on every output, low cost. **Layer 3 — LLM-as-Judge**: Multi-criteria scoring (accuracy, relevance, helpfulness, safety) using GPT-4 or Claude. Run on a sample (10-20% of traffic), moderate cost. Validate against human judgments quarterly. **Layer 4 — Human evaluation**: Expert review of 200+ sampled outputs monthly. Pairwise comparison when testing a new model/prompt. High cost but gold standard. **Layer 5 — Online metrics**: User thumbs up/down, regeneration rate, task completion rate, session length. Continuous, passive collection. Run regression tests on every prompt/model/pipeline change. Alert if any metric drops >2% (quality) or >0% (safety). This layered approach gives fast feedback on every change while maintaining the reliability of periodic human evaluation.

---

*End of Topic 25: Evaluation & Benchmarking*
