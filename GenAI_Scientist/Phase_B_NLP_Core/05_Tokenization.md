# Topic 5: Tokenization

> **Scope**: The science of converting raw text into tokens — BPE, WordPiece, Unigram, SentencePiece, byte-level BPE. Mathematical foundations, vocabulary design, and impact on model performance.

---

## Table of Contents

1. [Why Tokenization Matters](#1-why-tokenization-matters)
2. [Tokenization Strategies — The Spectrum](#2-tokenization-strategies)
3. [Byte-Pair Encoding (BPE)](#3-bpe)
4. [WordPiece](#4-wordpiece)
5. [Byte-Level BPE](#5-byte-level-bpe)
6. [Special Tokens](#6-special-tokens)
7. [Vocabulary Size Trade-offs](#7-vocabulary-size)
8. [Tokenization & Model Performance](#8-tokenization-and-performance)
9. [Comparing Tokenizers Across Models](#9-comparing-tokenizers)
10. [Interview Questions & Answers](#10-interview-qa)

---

## 1. Why Tokenization Matters

Tokenization is the **first transformation** in the LLM pipeline. Every downstream decision — vocabulary size, sequence length, inference speed, multilingual ability — traces back to the tokenizer.

```
Raw text ──→ Tokenizer ──→ Token IDs ──→ Embedding Layer ──→ Transformer
                │
                ├── Determines vocabulary (what the model "sees")
                ├── Determines sequence length (how many tokens per text)
                ├── Determines multilingual equity (tokens per word per language)
                └── Fixed after training (cannot change without retraining)
```

The tokenizer is trained **before** the model and remains frozen throughout pretraining, fine-tuning, and inference. A bad tokenizer cannot be compensated for by a better model.

---

## 2. Tokenization Strategies — The Spectrum

```
┌──────────────────────────────────────────────────────────────────┐
│              The Tokenization Spectrum                             │
│                                                                    │
│  Character-level          Subword                    Word-level    │
│  ─────────────           ────────                   ──────────    │
│  a, b, c, ...            un, happi, ness            unhappiness   │
│                                                                    │
│  Vocab: ~256              Vocab: 30K-128K            Vocab: 100K+ │
│  Very long sequences      Balanced                   Short seqs   │
│  No OOV problem           Rare OOV                   OOV problem  │
│  Slow training            Good trade-off             Fast if known│
│                                                                    │
│  ◄─────── shorter vocab, longer sequences ──────────────────────► │
│  ◄─────── better OOV handling ──────────────────────────────────► │
│  ◄─────── worse efficiency ──── sweet spot ──── worse coverage ──►│
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Word-Level Tokenization

Split on whitespace and punctuation. Each unique word gets a token ID.

**Problems**:
- **Huge vocabulary**: English alone has 170K+ common words, plus inflections, compound words, names
- **Out-of-Vocabulary (OOV)**: Any unseen word maps to `<unk>` — all information lost
- **No morphological sharing**: "run", "running", "runner" are completely independent tokens; the model can't share learned structure

### 2.2 Character-Level Tokenization

Each character is a token. Vocabulary is tiny (~256 for UTF-8 bytes or ~100 for ASCII).

**Problems**:
- **Extremely long sequences**: "Tokenization" = 12 characters = 12 tokens. A typical document becomes 5-10× longer.
- **Harder to learn**: The model must learn spelling, morphology, and word boundaries from scratch — all through attention patterns over individual characters

### 2.3 Subword Tokenization (The Winner)

Split words into meaningful subunits. Frequent words stay whole; rare words are broken into pieces.

$$\text{"unhappiness"} \rightarrow \text{["un", "happi", "ness"]}$$

This is the universal choice for modern LLMs. All major subword methods (BPE, WordPiece, Unigram) follow the same principle: **learn a vocabulary from data** that balances vocabulary size against sequence length.

---

## 3. Byte-Pair Encoding (BPE)

The most widely used tokenization algorithm. Used by GPT-2, GPT-3, GPT-4, Llama, Mistral, and most modern LLMs.

### 3.1 Algorithm

BPE is a bottom-up algorithm that iteratively merges the most frequent pair of adjacent tokens.

**Training (vocabulary construction)**:

```
Input: training corpus, desired vocabulary size V

1. Initialize vocabulary with all individual characters in the corpus
2. Represent the corpus as a sequence of characters
3. Repeat until |vocabulary| = V:
   a. Count all adjacent token pairs in the corpus
   b. Find the most frequent pair (a, b)
   c. Merge (a, b) into a new token "ab"
   d. Add "ab" to the vocabulary
   e. Replace all occurrences of (a, b) with "ab" in the corpus
4. Return: vocabulary + merge rules (ordered list of merges)
```

### 3.2 Worked Example

```
Corpus (with word frequencies):
  "low"   : 5
  "lower" : 2
  "newest": 6
  "widest": 3

Step 0: Character vocabulary
  {l, o, w, e, r, n, s, t, i, d}

Represent as character sequences (with end-of-word marker ·):
  l o w ·       (×5)
  l o w e r ·   (×2)
  n e w e s t · (×6)
  w i d e s t · (×3)

Step 1: Count pairs
  (e, s) = 9    ← most frequent (6 from "newest" + 3 from "widest")
  (s, t) = 9    ← tied
  (l, o) = 7
  (o, w) = 7
  ...

Merge (e, s) → "es":
  l o w ·       (×5)
  l o w e r ·   (×2)
  n e w es t ·  (×6)
  w i d es t ·  (×3)

Step 2: Count pairs again
  (es, t) = 9   ← most frequent
  (l, o) = 7
  ...

Merge (es, t) → "est":
  l o w ·       (×5)
  l o w e r ·   (×2)
  n e w est ·   (×6)
  w i d est ·   (×3)

Step 3:
  (l, o) = 7    ← most frequent

Merge (l, o) → "lo":
  lo w ·        (×5)
  lo w e r ·    (×2)
  n e w est ·   (×6)
  w i d est ·   (×3)

Step 4:
  (lo, w) = 7   ← most frequent

Merge (lo, w) → "low":
  low ·         (×5)
  low e r ·     (×2)
  n e w est ·   (×6)
  w i d est ·   (×3)

... and so on until vocabulary reaches desired size.
```

### 3.3 Encoding (Tokenization at Inference)

Given a trained BPE vocabulary and merge rules, tokenize new text by applying merges in the same order they were learned:

```
Input: "lowest"

Step 0: Split into characters
  l o w e s t

Step 1: Apply merge rules in order
  Rule 1: (e, s) → es     →  l o w es t
  Rule 2: (es, t) → est   →  l o w est
  Rule 3: (l, o) → lo     →  lo w est
  Rule 4: (lo, w) → low   →  low est

Result: ["low", "est"]
```

The model has never seen "lowest" during tokenizer training, but it composes it from known subwords — capturing both the base word "low" and the superlative "est".

### 3.4 Properties

- **Deterministic**: Given the merge rules, tokenization is unique
- **Greedy**: Always merges the earliest applicable rule (left to right)
- **Data-driven**: Vocabulary reflects the frequency distribution of the training corpus
- **Compact representation**: Common words become single tokens; rare words decompose into frequent subunits

### 3.5 Complexity

- Training: $O(N \cdot V)$ where $N$ = corpus size and $V$ = vocabulary size (each merge requires a pass over the corpus)
- Encoding: $O(n^2)$ for a word of length $n$ in the worst case (with efficient implementations: $O(n \log n)$)

### 3.6 BPE Step-by-Step Walkthrough

Starting corpus: ["low", "lower", "newest", "wider"]
(With end-of-word marker: ["l o w </w>", "l o w e r </w>", "n e w e s t </w>", "w i d e r </w>"])

Initial vocabulary: all individual characters {l, o, w, </w>, e, r, n, s, t, i, d}

**Iteration 1**: Count all adjacent pairs:
  "l o" appears 2x  (in "low", "lower")
  "o w" appears 2x
  "w </w>" appears 1x
  "e r" appears 2x  (in "lower", "wider")
  ... (many others)
  
  Most frequent: "e r" (2x) → MERGE into new token "er"
  Vocabulary now includes "er"

**Iteration 2**: After merging "er":
  Corpus: ["l o w </w>", "l o w er </w>", "n e w e s t </w>", "w i d er </w>"]
  Count pairs again... "l o" still 2x → MERGE into "lo"

**After ~10 iterations**:
  Common tokens: "lo", "low", "er", "est", "new", "wide", "newer", "lowest"
  
  "newest" → ["new", "est"] (2 tokens — recognized pattern)
  "unknown_word" → [character-level fallback] — never OOV!

---

## 4. WordPiece

Used by BERT, DistilBERT, and ELECTRA. Similar to BPE but uses a different merge criterion.

### 4.1 Key Difference from BPE

BPE merges the most **frequent** pair. WordPiece merges the pair that maximizes the **likelihood** of the training corpus under a language model.

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This is equivalent to selecting the pair whose merge produces the largest increase in the unigram language model likelihood:

$$\Delta \mathcal{L} = \log P(ab) - \log P(a) - \log P(b) = \log \frac{P(ab)}{P(a) \cdot P(b)}$$

This is the **pointwise mutual information (PMI)** of tokens $a$ and $b$.

### 4.2 Intuition: Frequency vs Likelihood

| Criterion | BPE (Frequency) | WordPiece (Likelihood/PMI) |
|-----------|-----------------|---------------------------|
| Merges | Most frequent pair | Pair with highest mutual information |
| Bias | Favors common bigrams | Favors pairs that co-occur more than chance |
| Effect | "th" merges early (very common) | "qu" might merge early (almost always together) |

WordPiece prefers merging pairs that are **strongly associated** (high PMI), even if individually less frequent. This produces a vocabulary where subwords are more linguistically meaningful.

### 4.3 The `##` Prefix Convention

WordPiece marks continuation tokens (non-initial subwords) with `##`:

$$\text{"unhappiness"} \rightarrow \text{["un", "##happi", "##ness"]}$$

This distinguishes word-initial tokens from mid-word tokens, allowing the model to recognize word boundaries:
- `"un"` at position 0 = word start (prefix meaning "not")
- `"##happi"` = continuation of the same word

BPE does not use this convention — it relies on whitespace encoding (e.g., `Ġ` prefix for space-preceded tokens in GPT-2).

> **SentencePiece note**: SentencePiece is a language-independent tokenizer framework that applies BPE or Unigram at the byte/character level without needing pre-tokenization. Used by: T5, Llama, Mistral, many multilingual models.

---

## 5. Byte-Level BPE

GPT-2 introduced byte-level BPE, which operates on **raw bytes** rather than Unicode characters. This is distinct from SentencePiece's approach.

### 5.1 The Problem with Character-Level BPE

Standard BPE operates on Unicode characters. Unicode has ~150,000 characters across all scripts. The base vocabulary (before any merges) would need to include all of these.

### 5.2 The Byte-Level Solution

Instead of starting with Unicode characters, start with the **256 possible byte values** (0x00 to 0xFF):

$$\text{Base vocabulary} = \{0, 1, 2, \ldots, 255\} \quad (|\text{base}| = 256)$$

Any Unicode text can be encoded as a sequence of UTF-8 bytes. BPE merges then operate on these byte sequences:

```
"café" in UTF-8 bytes: [99, 97, 102, 195, 169]
                        c   a   f    é (2 bytes)

BPE can merge byte pairs:
  [99, 97] → "ca"
  [102, 195, 169] → "fé" (merged across byte boundary)
```

### 5.3 Advantages

1. **Zero OOV**: Any byte sequence can be tokenized — no `<unk>` tokens ever
2. **Tiny base vocabulary**: 256 bytes vs ~150K Unicode characters
3. **Language-agnostic**: Works for any language, emoji, code, binary formats
4. **No pre-tokenization rules needed**: No language-specific splitting

### 5.4 The GPT-2 Encoding Trick

GPT-2 maps the 256 bytes to printable Unicode characters to avoid control characters in the vocabulary. This is a cosmetic mapping — the algorithm is still operating on bytes.

### 5.5 Disadvantage: Fertility for Non-Latin Scripts

Byte-level BPE with a small vocabulary produces many tokens per character for non-Latin scripts:

| Script | Bytes per Character | Effect |
|--------|-------------------|--------|
| ASCII (English) | 1 byte | Efficient — similar to char-level |
| Latin extended (French, German) | 1-2 bytes | Slightly longer |
| CJK (Chinese, Japanese, Korean) | 3 bytes | 3× more tokens per character |
| Emoji | 4 bytes | 4× more tokens |

This means the same content in Chinese requires ~3× more tokens than English, making inference ~3× slower and more expensive. Larger vocabularies (Llama 3: 128K) mitigate this by learning CJK characters as single tokens.

---

## 6. Special Tokens

Special tokens are reserved tokens with specific roles that the model learns during pretraining.

### 6.1 Common Special Tokens

```
┌─────────────────────────────────────────────────────────────┐
│                    Special Token Zoo                          │
├──────────┬─────────────────────────────────────────────────┤
│ Token    │ Purpose                                          │
├──────────┼─────────────────────────────────────────────────┤
│ [CLS]    │ Classification token (BERT). Aggregates          │
│          │ sequence representation for classification.      │
│ [SEP]    │ Separator between segments (BERT).               │
│ [MASK]   │ Placeholder for masked tokens in MLM (BERT).    │
│ [PAD]    │ Padding token for batch processing.              │
│ [UNK]    │ Unknown token for OOV words.                     │
│ <s>      │ Beginning of sequence (Llama, GPT).              │
│ </s>     │ End of sequence / end of generation.             │
│ <|endoftext|> │ Document separator (GPT-2/3).               │
│ <|im_start|>  │ Chat role marker (ChatML format).           │
│ <|im_end|>    │ End of chat turn (ChatML format).           │
│ <extra_id_N>  │ Sentinel tokens for span corruption (T5).   │
└──────────┴─────────────────────────────────────────────────┘
```

### 6.2 Structural Role

Special tokens serve as **control signals** that the model learns to interpret during pretraining:

```
BERT input:
  [CLS] The movie was great [SEP] I loved the acting [SEP] [PAD] [PAD]
    ↓                         ↓                        ↓
  Classification           Segment                  Ignored
  embedding               boundary                  (masked out)

GPT chat input:
  <|im_start|>system
  You are helpful.<|im_end|>
  <|im_start|>user
  What is ML?<|im_end|>
  <|im_start|>assistant
```

### 6.3 Adding Special Tokens

When adding new special tokens to a pretrained model:

1. Add token to the tokenizer vocabulary
2. Resize the model's embedding matrix: $E \in \mathbb{R}^{|V| \times d} \rightarrow E' \in \mathbb{R}^{(|V|+k) \times d}$
3. Initialize new embeddings (random or copy from similar tokens)
4. Fine-tune to learn the new token's representation

The new embeddings start random and need gradient signal to become useful — a cold start that requires training data containing the new tokens.

---

## 7. Vocabulary Size Trade-offs

The vocabulary size $|V|$ creates a three-way trade-off:

### 7.1 The Trade-off Triangle

$$\text{Vocab size } |V| \quad \leftrightarrow \quad \text{Sequence length} \quad \leftrightarrow \quad \text{Embedding parameters}$$

**Larger vocabulary**:
- Fewer tokens per text (shorter sequences) → faster inference
- More parameters in embedding layer: $|V| \times d_{\text{model}}$
- Better coverage of rare words and non-English scripts
- Each token seen less often during training (sparser signal)

**Smaller vocabulary**:
- More tokens per text (longer sequences) → slower inference, uses more context window
- Fewer embedding parameters
- Better per-token training signal (each token seen more often)
- Poor coverage of non-English, domain-specific terms

### 7.2 Embedding Layer Cost

The embedding matrix accounts for a significant fraction of total parameters in smaller models:

$$\text{Embedding params} = |V| \times d_{\text{model}}$$

| Model | Vocab Size | $d_{\text{model}}$ | Embedding Params | Total Params | Embedding % |
|-------|-----------|-------------------|-----------------|-------------|------------|
| BERT-base | 30,522 | 768 | 23.4M | 110M | 21% |
| GPT-2 | 50,257 | 768 | 38.6M | 117M | 33% |
| Llama 2 (7B) | 32,000 | 4,096 | 131M | 7B | 1.9% |
| Llama 3 (8B) | 128,256 | 4,096 | 525M | 8B | 6.6% |

For large models, the embedding cost is a small fraction. For small models, vocabulary size meaningfully affects the parameter budget.

### 7.3 Sequence Length Impact

$$\text{Tokens per word} \approx f(|V|, \text{language})$$

| Vocab Size | English Tokens/Word | Chinese Tokens/Char | Inference Cost |
|-----------|--------------------|--------------------|---------------|
| 32K | ~1.3 | ~3.8 | Higher (longer sequences) |
| 50K | ~1.2 | ~3.2 | Medium |
| 128K | ~1.1 | ~1.8 | Lower (shorter sequences) |

Llama 3's jump from 32K to 128K vocabulary reduced Chinese token count by ~2×, making inference nearly twice as fast for Chinese text.

### 7.4 Tokenization Strategies Compared

```
Strategy        | OOV handling | Vocab size | Seq length | Example: "ChatGPT"
----------------|--------------|------------|------------|--------------------
Word-level      | FAIL (UNK)   | 50K-100K   | Short      | ["Chat", "G", "PT"] — OOV
Character-level | Perfect      | ~100        | Very long  | ["C","h","a","t","G","P","T"]
BPE             | Never OOV    | 30K-100K   | Medium     | ["Chat", "G", "PT"] or ["ChatGPT"]
Byte-level BPE  | Never OOV    | 256+merges | Medium     | Always representable in bytes
```

**The fundamental tradeoff:**
  Large vocabulary → fewer tokens per sentence (efficient) but rare tokens learned poorly
  Small vocabulary → more tokens per sentence (expensive attention) but each token well-trained

GPT-2/3/4 use ~50K vocab. Llama uses ~32K. Models with 100K+ vocab tokens become less common because
the embedding table (vocab_size × d_model) gets large and rare tokens are under-trained.

---

## 8. Tokenization & Model Performance

### 8.1 Tokenization Determines What the Model Can See

The model operates entirely in token space. If the tokenizer splits "COVID-19" into ["CO", "VID", "-", "19"], the model must learn through attention patterns that these four tokens represent a single entity. If the tokenizer keeps it as a single token "COVID-19", the model has direct access to the concept.

### 8.2 Impact on Arithmetic and Reasoning

LLMs famously struggle with arithmetic. Tokenization is a contributing factor:

```
"12345 + 67890"

Tokenization A: ["123", "45", " +", " 678", "90"]
  → Digits are split across token boundaries
  → Model can't easily "see" individual digits
  → Addition requires cross-token digit extraction

Tokenization B: ["1", "2", "3", "4", "5", " +", " 6", "7", "8", "9", "0"]
  → Each digit is its own token
  → Much easier for the model to align and compute
```

This is why some models use digit-level tokenization for numbers, at the cost of longer sequences.

### 8.3 Impact on Code

Code has unique tokenization challenges:
- Indentation matters (Python): spaces/tabs must be preserved
- Special characters are frequent: `{`, `}`, `->`, `::`, `===`
- Variable names should ideally stay as single tokens
- Common patterns: `def`, `self.`, `import`, `return`

Models trained with code-aware tokenizers (larger vocabulary with code-specific tokens) perform better on coding benchmarks.

### 8.4 The Glitch Token Phenomenon

Some tokens in the vocabulary are trained on very little data, leading to bizarre model behavior when these tokens appear:

- Tokens from the long tail of the training data
- Artifacts from data processing (URLs, file paths, encoded strings)
- Tokens that the model has essentially never seen in context

When prompted with these tokens, models may produce incoherent or unexpected outputs. This is because the embedding for the rare token was poorly learned during pretraining.

---

## 9. Comparing Tokenizers Across Models

### 9.1 Architecture Summary

```
┌──────────────────────────────────────────────────────────────────┐
│              Tokenizer Architectures by Model                     │
├───────────────┬─────────────┬──────────┬────────────────────────┤
│ Model         │ Algorithm   │ Vocab    │ Key Design Choices     │
├───────────────┼─────────────┼──────────┼────────────────────────┤
│ BERT          │ WordPiece   │ 30,522   │ ## continuation, [CLS] │
│ GPT-2         │ Byte BPE    │ 50,257   │ Ġ space prefix         │
│ T5            │ SP+Unigram  │ 32,000   │ ▁ space, sentinels     │
│ Llama 2       │ SP+BPE      │ 32,000   │ ▁ space, byte fallback │
│ Llama 3       │ Tiktoken BPE│ 128,256  │ Large vocab, multilingual│
│ Mistral       │ SP+BPE      │ 32,000   │ Similar to Llama 2     │
│ GPT-4         │ Byte BPE    │ 100,277  │ cl100k_base, tiktoken  │
│ Gemini        │ SP+BPE      │ 256,000  │ Largest vocab          │
└───────────────┴─────────────┴──────────┴────────────────────────┘
```

### 9.2 The Trend

```
         2018          2019-2022       2023-2024        2025+
         BERT          GPT-2/3         Llama 3          Gemini
         30K           50K             128K             256K
         WordPiece     Byte BPE        Byte BPE         SP+BPE
         English-      English-        Multilingual     Highly
         focused       focused         improved         multilingual

    ◄── smaller vocab ──────────────────── larger vocab ──────►
    ◄── more tokens per text ──── fewer tokens per text ──────►
    ◄── English-centric ────────── multilingual-equitable ────►
```

The field is converging on **larger vocabularies** (100K+) with **byte-level fallback**, prioritizing:
1. Efficiency across languages (lower fertility)
2. Better code handling
3. Shorter sequences (faster inference)

---

## 10. Interview Questions & Answers

### Q1: Walk through the BPE algorithm step by step. How does it build the vocabulary?

**A**: BPE starts with a base vocabulary of individual characters (or bytes). It then iteratively: (1) counts all adjacent token pairs in the corpus, (2) finds the most frequent pair, (3) merges that pair into a new token added to the vocabulary, (4) replaces all occurrences of the pair in the corpus. This repeats until the vocabulary reaches the desired size.

The result is a merge list (ordered sequence of merge rules) and a vocabulary. At inference time, the same merge rules are applied in order to tokenize new text. Common words become single tokens early; rare words are decomposed into known subwords. The key property is that every possible input can be tokenized using the base characters as a fallback.

---

### Q2: Why did GPT-2 switch to byte-level BPE? What problem does it solve?

**A**: Character-level BPE requires having all Unicode characters in the base vocabulary — over 150,000 characters across all scripts. This is impractical. Byte-level BPE operates on the 256 possible byte values instead. Since any text can be represented as UTF-8 bytes, this guarantees:

1. **Zero OOV**: Any byte sequence can be tokenized — no `<unk>` tokens ever
2. **Tiny base vocabulary**: 256 entries vs 150K+ Unicode characters
3. **Language-agnostic**: Works for any language, emoji, code, or even binary data

The trade-off is that non-ASCII characters (CJK, Cyrillic, etc.) require multiple bytes per character, increasing fertility. This is mitigated by BPE merges learning common multi-byte patterns as single tokens, and by using larger vocabulary sizes (GPT-4: 100K, Llama 3: 128K).

---

### Q3: Why does Llama use SentencePiece? How does it handle multilingual text?

**A**: Llama uses SentencePiece because it's **language-agnostic** — it treats input as raw character/byte sequences without requiring language-specific pre-tokenization rules. Traditional tokenizers need separate rules for English (split on spaces), Chinese (character-level), Japanese (morphological analysis), etc. SentencePiece handles all languages with a single algorithm.

For multilingual text, SentencePiece encodes spaces as `▁` (visible underscore), allowing lossless reconstruction. It uses byte-level fallback for characters not in the vocabulary, so no language produces `<unk>` tokens. Llama 3 increased vocabulary from 32K to 128K specifically to improve multilingual efficiency, reducing Chinese fertility from ~3.8 to ~1.8 tokens per character.

---

### Q4: A model tokenizes "unhappiness" as ["un", "happiness"] vs ["unhapp", "iness"]. Which is better and why?

**A**: ["un", "happiness"] is better because the split aligns with **morphological boundaries**:

- "un" = negation prefix (compositional meaning: "not")
- "happiness" = base word (a known concept)

The model can leverage what it learned about "un-" (negation in "undo", "unclear", "unfair") and "happiness" independently. The representation is compositional.

["unhapp", "iness"] splits at an arbitrary character boundary. Neither "unhapp" nor "iness" carries independent meaning. The model must learn from scratch that this combination means "unhappiness" — it can't compose it from known parts.

In practice, BPE usually produces morphologically reasonable splits because morpheme boundaries tend to be frequency boundaries — common prefixes and suffixes merge early during vocabulary construction because they appear frequently.

---

### Q5: Compare BPE, WordPiece, and Unigram. What are the key differences?

**A**:

| Property | BPE | WordPiece | Unigram |
|----------|-----|-----------|---------|
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) |
| Merge criterion | Most frequent pair | Highest PMI pair | N/A (removal minimizes loss) |
| Tokenization | Deterministic (apply merge rules) | Deterministic (greedy longest-match) | Probabilistic (Viterbi for best, sampling possible) |
| Subword regularization | No | No | Yes (multiple valid tokenizations) |
| Used by | GPT, Llama, Mistral | BERT, DistilBERT | T5, ALBERT, XLNet |

**When to choose**: BPE is the default for generative models (simple, well-understood). WordPiece works well for encoder models. Unigram is preferred when subword regularization is desired (improves robustness, especially for translation).

---

### Q6: What is subword regularization? Why does it help?

**A**: Subword regularization (Kudo, 2018) is unique to the Unigram tokenizer. Because the Unigram model assigns probabilities to tokens, multiple valid tokenizations exist for any word. Instead of always using the most probable (Viterbi) tokenization, you can **sample** different tokenizations during training:

$$x \sim P(x | W) \propto \prod_i p(x_i)$$

For example, "international" might be tokenized as:
- ["inter", "national"] (most probable)
- ["in", "ter", "nation", "al"] (sampled)
- ["intern", "ation", "al"] (sampled)

This acts as data augmentation at the tokenization level — the model sees different breakdowns of the same word across training steps, making it more robust to tokenization artifacts. Kudo showed 1-2 BLEU point improvements on machine translation.

BPE cannot do this because its tokenization is deterministic — the merge rules define a unique segmentation.

---

### Q7: How does vocabulary size affect inference cost?

**A**: Vocabulary size affects inference through two mechanisms:

**1. Sequence length** (dominant effect): Larger vocabulary → fewer tokens per text → fewer forward passes during autoregressive generation. If Llama 3 (128K vocab) produces 20% fewer tokens than Llama 2 (32K vocab) for the same content, inference is roughly 20% faster.

$$\text{Inference cost} \propto T \cdot C_{\text{per-token}}$$

where $T$ = number of tokens (reduced by larger vocab).

**2. Softmax computation**: The final layer computes softmax over $|V|$ logits:

$$P(x_t | x_{<t}) = \text{softmax}(h_t W_{\text{head}}^\top) \in \mathbb{R}^{|V|}$$

Larger vocabulary means a larger matrix multiply ($d \times |V|$) and softmax. But this is typically a small fraction of total compute compared to the attention layers.

Net effect: larger vocabulary almost always reduces total inference cost because the sequence length reduction outweighs the per-token softmax increase.

---

### Q8: What happens if you use a tokenizer trained on one model with a different model?

**A**: This generally doesn't work and can fail catastrophically. The tokenizer and model are tightly coupled because:

1. **Embedding matrix alignment**: The model's embedding matrix $E \in \mathbb{R}^{|V| \times d}$ has exactly $|V|$ rows, one per token in the tokenizer's vocabulary. A different tokenizer with different vocabulary size or token ordering would produce IDs that map to the wrong (or nonexistent) embedding rows.

2. **Learned token semantics**: During pretraining, the model learned that token ID 1234 means "the" (for example). A different tokenizer might assign ID 1234 to "dog" — complete semantic mismatch.

3. **Special token mismatch**: Different models use different special tokens (`[CLS]` vs `<s>`, `[SEP]` vs `</s>`). Using the wrong special tokens breaks the model's control flow.

The only safe scenario is using the **exact tokenizer** that the model was trained with, or a tokenizer that produces the same vocabulary with the same ID mapping.

---

### Q9: How would you design a tokenizer for a domain-specific model (e.g., biomedical)?

**A**: Start with an existing general tokenizer and extend it, or train a new one on domain data:

**Option 1: Train from scratch** on domain corpus + general corpus:
1. Collect a representative domain corpus (medical papers, clinical notes)
2. Mix with general text (prevent domain overfitting): ~70% domain, 30% general
3. Train BPE/Unigram with vocabulary size ~50K-100K
4. Verify that domain-specific terms (drug names, gene names, medical codes) are single tokens or at most 2 tokens

**Option 2: Extend existing tokenizer**:
1. Identify domain terms that are poorly tokenized (high fertility)
2. Add them to the vocabulary as new tokens
3. Resize model embeddings and fine-tune

**Key verification**: Compute fertility on domain-specific test set. Good domain tokenization: "acetaminophen" → 1-2 tokens. Bad: "acetaminophen" → ["ac", "et", "amin", "ophen"] (4 tokens, losing the word as a unit).

---

### Q10: Why is the tokenizer considered a bottleneck for LLM reasoning about numbers and code?

**A**: The tokenizer introduces inconsistent boundaries that break structural patterns:

**Numbers**: "12345" might tokenize as ["123", "45"] or ["1234", "5"] — different numbers get different splits. The model can't reliably extract individual digits because they're hidden within multi-digit tokens. Arithmetic requires digit-level alignment, but tokenization makes this unpredictable.

**Code**: Indentation (critical in Python) might be tokenized inconsistently. "    " (4 spaces) could be one token or four separate space tokens depending on context. Variable names split arbitrarily: "getUserName" → ["get", "User", "Name"] in one context but ["getUser", "Name"] in another.

**The fundamental issue**: Tokenization is a lossy transformation that discards character-level structure. The model must reconstruct this structure through attention patterns — possible but inefficient, especially for tasks that inherently operate at the character/digit level.

Research directions: (1) character-level models (ByT5), (2) digit-level tokenization for numbers, (3) hybrid approaches that provide character-level information alongside subword tokens.
