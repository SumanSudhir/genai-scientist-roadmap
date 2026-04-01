# Topic 6: Text Preprocessing & Word Embeddings

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 6 of 28
> **Scope**: Classical text preprocessing, Bag of Words, TF-IDF, Word2Vec, GloVe, FastText, static vs contextual embeddings, sentence embeddings, embedding spaces
> **Why this matters**: Embeddings are the foundation of everything in NLP. RAG systems use embedding models. Semantic search relies on vector similarity. Understanding how we went from one-hot vectors to contextual representations explains WHY transformers are so powerful.
> **Prev**: [Topic 5: Tokenization](05_Tokenization.md)
> **Next**: [Topic 7: Sequence Modeling](07_Sequence_Modeling.md)

---

## Table of Contents

1. [Classical Text Preprocessing](#1-classical-text-preprocessing)
2. [Bag of Words (BoW)](#2-bag-of-words-bow)
3. [TF-IDF](#3-tf-idf)
4. [Word2Vec](#4-word2vec)
5. [GloVe](#5-glove)
6. [FastText](#6-fasttext)
7. [Static vs Contextual Embeddings](#7-static-vs-contextual-embeddings)
8. [Sentence & Document Embeddings](#8-sentence--document-embeddings)
9. [Embedding Spaces: Properties & Operations](#9-embedding-spaces-properties--operations)
10. [Modern Embedding Models for RAG & Search](#10-modern-embedding-models-for-rag--search)
11. [Interview Questions & Answers](#11-interview-questions--answers)

---

## 1. Classical Text Preprocessing

Before the era of neural NLP, text preprocessing was critical because models couldn't learn invariances on their own. Understanding these techniques matters because: (a) they appear in interview "trace the evolution" questions, (b) BM25 and TF-IDF still power hybrid search in RAG systems, and (c) they reveal the problems that neural approaches solved.

### 1.1 Lowercasing

Convert all text to lowercase to reduce vocabulary size.

- "The Cat sat on THE mat" → "the cat sat on the mat"
- Reduces vocabulary by ~30-50% in typical English corpora

**When it hurts**: "US" (country) vs "us" (pronoun), "IT" (industry) vs "it" (pronoun), "Apple" (company) vs "apple" (fruit). Modern tokenizers handle this via subword units and case-sensitive vocabularies — BERT's uncased model lowercases, but the cased model preserves case.

### 1.2 Stemming

Reduce words to their root form by chopping suffixes using heuristic rules.

**Porter Stemmer** (most common):
- "running" → "run"
- "studies" → "studi" (not a real word!)
- "university" → "univers"

**Snowball Stemmer**: Improved Porter with better rules for multiple languages.

**Problems**: Over-stemming ("university" → "univers"), under-stemming ("alumnus" / "alumni" not merged), produces non-words. Fast but crude.

### 1.3 Lemmatization

Reduce words to their dictionary form (lemma) using linguistic knowledge.

- "running" → "run" (verb lemma)
- "better" → "good" (adjective lemma)
- "studies" → "study" (not "studi"!)
- "mice" → "mouse"

**Requires**: Part-of-speech information and a morphological dictionary (e.g., WordNet). Slower but more accurate than stemming.

**SpaCy vs NLTK**: SpaCy uses lookup tables + rules (fast). NLTK's WordNetLemmatizer requires POS tag as input.

### 1.4 Stop Word Removal

Remove high-frequency, low-information words: "the", "is", "at", "which", "on", etc.

**Common stop word lists**: NLTK (179 words), SpaCy (326 words), scikit-learn's CountVectorizer.

**When it hurts**: "To be or not to be" → "" (entire sentence is stop words). "The Who" (band name). In modern NLP, we generally don't remove stop words — transformer models learn to handle them, and they carry syntactic information.

### 1.5 Text Cleaning

- **HTML/XML tag removal**: Strip `<p>`, `<div>`, etc.
- **URL removal**: Regex patterns for http/https links
- **Special character handling**: Decide what to keep (punctuation, numbers, emoji)
- **Unicode normalization**: NFKD/NFC forms for consistent representation
- **Whitespace normalization**: Collapse multiple spaces, handle tabs/newlines

### 1.6 When Preprocessing Still Matters

| Context | Preprocessing Needed? | Why |
|---------|----------------------|-----|
| Modern LLMs (GPT, Llama) | Minimal | Tokenizers handle raw text |
| BM25 search in RAG | Yes | Stemming/lowercasing improve recall |
| TF-IDF features | Yes | Reduces vocabulary, improves signal |
| Keyword extraction | Sometimes | Lemmatization groups word forms |
| Data cleaning for pretraining | Yes | HTML, dedup, quality filtering |

---

## 2. Bag of Words (BoW)

### 2.1 The Representation

Represent text as a vector of word counts, ignoring order completely.

**Vocabulary**: $V = \{w_1, w_2, ..., w_{|V|}\}$ — the set of all unique words in the corpus.

**Document vector**: $\mathbf{d} \in \mathbb{R}^{|V|}$ where $d_i$ = count of word $w_i$ in the document.

**Example**:
- Vocabulary: ["cat", "sat", "on", "mat", "the", "dog"]
- "the cat sat on the mat" → [1, 1, 1, 1, 2, 0]
- "the dog sat on the mat" → [0, 1, 1, 1, 2, 1]

### 2.2 Properties and Limitations

**What BoW captures**: Which words are present and how often. Two documents about similar topics will share many words, giving high cosine similarity.

**What BoW loses**:
1. **Word order**: "Dog bites man" and "Man bites dog" have identical BoW representations
2. **Semantics**: "Good" and "excellent" are as different as "good" and "terrible"
3. **Context**: "Bank" (financial) and "bank" (river) have the same feature
4. **Sparsity**: With $|V| = 100,000$, most entries are zero — extremely high-dimensional, sparse vectors

### 2.3 N-gram Extension

To partially capture word order, use n-grams (sequences of n words) as features:

- Unigrams: ["the", "cat", "sat"]
- Bigrams: ["the cat", "cat sat", "sat on"]
- Trigrams: ["the cat sat", "cat sat on"]

This explodes the vocabulary size ($|V|^n$ possible n-grams) but captures local context. In practice, bigrams and trigrams with frequency filtering work reasonably well.

### 2.4 Binary BoW

Instead of counts, use binary indicators: $d_i = 1$ if word $w_i$ appears, $0$ otherwise. Often works as well as count-based BoW for classification tasks because word presence is more informative than exact frequency in many cases.

---

## 3. TF-IDF

### 3.1 Motivation

Raw word counts are misleading. "The" appears everywhere — its high count doesn't distinguish documents. We need a weighting scheme that values **distinctive** words.

### 3.2 Term Frequency (TF)

How often a term appears in a document, normalized:

$$\text{TF}(t, d) = \frac{f(t, d)}{\sum_{t' \in d} f(t', d)}$$

where $f(t, d)$ is the raw count of term $t$ in document $d$.

**Variants**:
- **Raw count**: $f(t, d)$
- **Log normalization**: $1 + \log(f(t, d))$ — dampens the effect of very frequent terms
- **Boolean**: $1$ if $t \in d$, else $0$

### 3.3 Inverse Document Frequency (IDF)

How rare a term is across the corpus:

$$\text{IDF}(t) = \log\frac{N}{|\{d \in D : t \in d\}|}$$

where $N$ is the total number of documents and the denominator is the number of documents containing $t$.

**Intuition**: Words appearing in many documents (e.g., "the", "is") get low IDF. Words appearing in few documents (e.g., "quantum", "mitochondria") get high IDF.

**IDF values** (example with 10,000 documents):
- "the" appears in 9,900 docs → $\text{IDF} = \log(10000/9900) \approx 0.004$
- "machine" appears in 500 docs → $\text{IDF} = \log(10000/500) = 3.0$
- "eigendecomposition" appears in 5 docs → $\text{IDF} = \log(10000/5) = 7.6$

### 3.4 TF-IDF Score

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Properties**:
- High for terms that are frequent in the document BUT rare in the corpus (distinctive terms)
- Low for terms that are rare in the document (not relevant)
- Low for terms that are common everywhere (not distinctive)

### 3.5 Connection to Information Theory

IDF is closely related to information content:

$$\text{IDF}(t) = \log\frac{N}{df(t)} \approx -\log P(t)$$

where $P(t) = df(t)/N$ is the probability of a random document containing term $t$. This is the **self-information** (surprisal) of encountering $t$. Rare terms carry more information — exactly the intuition behind IDF.

### 3.6 TF-IDF in Modern Systems

**BM25** (Best Matching 25) is the modern evolution of TF-IDF used in Elasticsearch, Solr, and as the sparse retrieval component in hybrid RAG:

$$\text{BM25}(t, d) = \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$$

where $k_1$ (typically 1.2) controls TF saturation and $b$ (typically 0.75) controls document length normalization. BM25 is a probabilistic retrieval model that provides a principled improvement over raw TF-IDF.

**Why BM25 still matters**: In RAG hybrid search, BM25 captures **lexical matching** (exact keyword overlap) that dense retrieval misses. The combination of BM25 (sparse) + embedding similarity (dense) with reciprocal rank fusion consistently outperforms either alone.

---

## 4. Word2Vec

### 4.1 The Key Insight

**Distributional hypothesis** (Firth, 1957): "You shall know a word by the company it keeps."

Words that appear in similar contexts have similar meanings. Word2Vec operationalizes this by learning vector representations where words with similar contexts end up close together in vector space.

**Before Word2Vec**: Words were one-hot vectors — "king" and "queen" were as different as "king" and "banana" (all pairwise distances equal). No notion of similarity.

**After Word2Vec**: Words are dense vectors in $\mathbb{R}^d$ (typically $d = 300$) where cosine similarity captures semantic relationships: $\cos(\text{king}, \text{queen}) \gg \cos(\text{king}, \text{banana})$.

### 4.2 Skip-gram Model

**Objective**: Given a center word, predict the surrounding context words.

For a vocabulary of size $|V|$ and embedding dimension $d$:
- **Input embedding matrix**: $\mathbf{W} \in \mathbb{R}^{|V| \times d}$ (word as center)
- **Output embedding matrix**: $\mathbf{W}' \in \mathbb{R}^{|V| \times d}$ (word as context)

For center word $w_c$ and context word $w_o$ within a window of size $m$:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{v}'_{w_o}{}^T \mathbf{v}_{w_c})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}'_w{}^T \mathbf{v}_{w_c})}$$

where $\mathbf{v}_{w_c}$ is the center embedding and $\mathbf{v}'_{w_o}$ is the context embedding.

**Training objective** (maximize log probability):

$$\mathcal{J} = \frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

where $T$ is the corpus size and $m$ is the context window size.

### 4.3 The Softmax Problem and Negative Sampling

**Problem**: Computing $P(w_o \mid w_c)$ requires normalizing over the entire vocabulary ($|V|$ often 100K-1M). This is a $O(|V|)$ computation for every training step — prohibitively expensive.

**Negative Sampling** (Mikolov et al., 2013): Instead of computing the full softmax, convert the problem into binary classification:

- For the true (center, context) pair: predict 1
- For $k$ randomly sampled "negative" pairs: predict 0

**Negative sampling objective**:

$$\mathcal{J} = \log \sigma(\mathbf{v}'_{w_o}{}^T \mathbf{v}_{w_c}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}\left[\log \sigma(-\mathbf{v}'_{w_i}{}^T \mathbf{v}_{w_c})\right]$$

where $\sigma$ is the sigmoid function, $k$ is the number of negatives (typically 5-20), and $P_n(w) \propto f(w)^{3/4}$ is the noise distribution (word frequency raised to the 3/4 power).

**Why $f(w)^{3/4}$?** Raising frequency to 3/4 slightly flattens the distribution, giving rare words a higher chance of being sampled as negatives. Pure frequency sampling would almost never sample rare words, and the model would never learn to distinguish them.

**Complexity reduction**: From $O(|V|)$ per step to $O(k)$ per step — a massive speedup.

### 4.4 CBOW (Continuous Bag of Words)

**Objective**: Given the context words, predict the center word.

$$P(w_c \mid w_{c-m}, ..., w_{c+m}) = \frac{\exp(\mathbf{v}'_{w_c}{}^T \bar{\mathbf{v}})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}'_w{}^T \bar{\mathbf{v}})}$$

where $\bar{\mathbf{v}} = \frac{1}{2m}\sum_{j=-m, j\neq 0}^{m} \mathbf{v}_{w_{c+j}}$ is the average of context word embeddings.

**Skip-gram vs CBOW**:

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Predicts | Context from center | Center from context |
| Rare words | Better (each word gets many updates) | Worse (rare words averaged out) |
| Training speed | Slower | Faster |
| Data efficiency | Better for small corpora | Better for large corpora |
| Typical use | Default choice | When speed matters |

### 4.5 Training Details

**Subsampling of frequent words**: Very common words ("the", "a", "is") provide less information. Each word $w$ is discarded during training with probability:

$$P(\text{discard}) = 1 - \sqrt{\frac{t}{f(w)}}$$

where $f(w)$ is the word's frequency and $t$ is a threshold (typically $10^{-5}$). This speeds up training and improves quality — "the" appearing next to any word doesn't tell us much.

**Window size**: Typically 5-10. Smaller windows capture more syntactic relationships ("big" ↔ "large"). Larger windows capture more topical/semantic relationships ("dog" ↔ "veterinarian").

**Embedding dimension**: Typically 100-300. Higher dimensions capture more nuance but risk overfitting with limited data. The original Word2Vec paper used $d = 300$.

---

## 5. GloVe

### 5.1 Motivation: Global vs Local

**Word2Vec's limitation**: It only uses local context windows. It doesn't directly leverage the global co-occurrence statistics of the corpus.

**GloVe's insight** (Pennington et al., 2014): Word meaning is captured by the **ratios** of co-occurrence probabilities, not the raw probabilities themselves.

### 5.2 Co-occurrence Matrix

**Build a co-occurrence matrix** $\mathbf{X}$ where $X_{ij}$ = number of times word $j$ appears in the context of word $i$ across the entire corpus.

Let $P_{ij} = P(j \mid i) = X_{ij} / X_i$ be the probability of word $j$ appearing in the context of word $i$.

### 5.3 The Ratio Insight

Consider words "ice" and "steam" with probe words:

| Probe word $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Ratio $\frac{P(k|\text{ice})}{P(k|\text{steam})}$ |
|----------------|----------------------|------------------------|--------------------------------------------------|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | 8.9 |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | 0.085 |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 1.36 |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | 0.96 |

- "Solid" is much more associated with "ice" → ratio >> 1
- "Gas" is much more associated with "steam" → ratio << 1
- "Water" is related to both → ratio ≈ 1
- "Fashion" is unrelated to both → ratio ≈ 1

**The ratios discriminate meaning better than raw probabilities.**

### 5.4 GloVe Objective

GloVe learns embeddings such that the dot product of word vectors approximates the log of co-occurrence:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$$

**Weighted least squares objective**:

$$\mathcal{J} = \sum_{i,j=1}^{|V|} f(X_{ij})\left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

where $f(X_{ij})$ is a weighting function:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

with $x_{\max} = 100$ and $\alpha = 3/4$ typically.

**Why the weighting**: Without $f$, very frequent co-occurrences (like "the" + "of") would dominate the loss. The capping function prevents this while still giving more weight to more informative (moderately frequent) co-occurrences.

### 5.5 GloVe vs Word2Vec

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Training signal | Local context windows | Global co-occurrence matrix |
| Objective | Predict context/center words | Reconstruct log co-occurrence |
| Training method | Online (SGD on corpus) | Batch (matrix factorization) |
| Scalability | Streams through corpus | Must build co-occurrence matrix first |
| Performance | Slightly better on syntactic tasks | Slightly better on semantic tasks |
| Theory | Neural network | Connects to SVD/matrix factorization |

**Levy and Goldberg (2014)** showed that Word2Vec with negative sampling is implicitly factorizing a **shifted PMI (Pointwise Mutual Information) matrix**: $\mathbf{w}_i^T \mathbf{w}_j \approx \text{PMI}(i, j) - \log k$. This means Word2Vec and GloVe are doing fundamentally similar things from different angles — both are capturing co-occurrence statistics in low-rank vector spaces.

---

## 6. FastText

### 6.1 The OOV Problem

**Word2Vec and GloVe's fatal flaw**: They learn one vector per word. If a word wasn't in the training vocabulary, it gets no embedding. This is the **out-of-vocabulary (OOV) problem**.

- Misspellings: "languge" has no embedding even though it's clearly "language"
- Morphological variants: "unhappily" might not be in vocabulary even if "happy" is
- Rare words: Technical terms, proper nouns, neologisms

### 6.2 Subword Embeddings

**FastText's solution** (Bojanowski et al., 2017): Represent each word as a **bag of character n-grams**, plus the word itself.

For word "where" with $n = 3$:
- Add boundary markers: `<where>`
- Extract character n-grams: `<wh`, `whe`, `her`, `ere`, `re>`
- Plus the full word: `<where>`

**Word embedding** = sum (or average) of all its n-gram embeddings:

$$\mathbf{v}_{\text{where}} = \mathbf{v}_{<wh} + \mathbf{v}_{whe} + \mathbf{v}_{her} + \mathbf{v}_{ere} + \mathbf{v}_{re>} + \mathbf{v}_{<where>}$$

### 6.3 Why Subwords Work

1. **OOV handling**: Even unseen words share n-grams with known words. "languge" shares n-grams with "language", so it gets a reasonable (if imperfect) embedding.

2. **Morphology**: "unhappily" = "unh" + "nha" + "hap" + "app" + "ppi" + "pil" + "ily" + ... Many of these overlap with "happy", "happily", "unhappy", automatically capturing morphological relationships.

3. **Rare words**: Even words seen only once or twice benefit from the n-gram representations learned from more frequent words sharing those n-grams.

### 6.4 FastText Training

FastText uses the **skip-gram objective** (same as Word2Vec) but replaces the center word embedding with the sum of its n-gram embeddings:

$$\text{score}(w_c, w_o) = \sum_{g \in G(w_c)} \mathbf{z}_g^T \mathbf{v}'_{w_o}$$

where $G(w_c)$ is the set of n-grams of word $w_c$ and $\mathbf{z}_g$ is the n-gram embedding.

**Typical settings**: Character n-grams from length 3 to 6. Hash n-grams into a fixed-size bucket (e.g., 2M buckets) to limit memory.

### 6.5 FastText vs Word2Vec vs GloVe

| Aspect | Word2Vec | GloVe | FastText |
|--------|----------|-------|----------|
| OOV handling | None | None | Yes (via n-grams) |
| Morphology | Not captured | Not captured | Captured |
| Rare words | Poor embeddings | Poor embeddings | Good (n-gram sharing) |
| Memory | $|V| \times d$ | $|V| \times d$ | $(|V| + B) \times d$ ($B$ = n-gram buckets) |
| Languages | English-centric | English-centric | Good for morphologically rich languages |
| Speed | Fast | Fast (after matrix built) | Slightly slower (more lookups) |

---

## 7. Static vs Contextual Embeddings

### 7.1 The Polysemy Problem

**Static embeddings** (Word2Vec, GloVe, FastText) assign **one vector per word**, regardless of context:

- "I went to the **bank** to deposit money" → bank = financial institution
- "I sat on the river **bank**" → bank = riverbank
- "You can **bank** on it" → bank = rely on

All three instances get the **exact same embedding**. This is fundamentally wrong — the meaning is different in each case. The static embedding becomes an awkward average of all possible senses.

### 7.2 The Contextual Revolution

**Contextual embeddings** (ELMo, BERT, GPT) produce **different vectors for the same word depending on context**:

- ELMo (2018): Bidirectional LSTM contextualizes word representations
- BERT (2018): Transformer encoder produces deeply contextualized embeddings
- GPT (2018+): Transformer decoder produces context-dependent representations

**BERT embedding for "bank"**:
- "I went to the bank to deposit money" → $\mathbf{h}_{\text{bank}}^{(1)} \in \mathbb{R}^{768}$
- "I sat on the river bank" → $\mathbf{h}_{\text{bank}}^{(2)} \in \mathbb{R}^{768}$
- $\cos(\mathbf{h}_{\text{bank}}^{(1)}, \mathbf{h}_{\text{bank}}^{(2)})$ is LOW — BERT knows these are different

### 7.3 How Contextual Embeddings Work

In a transformer, the representation of each token at layer $l$ depends on **all other tokens** in the sequence through self-attention:

$$\mathbf{h}_i^{(l)} = \text{TransformerLayer}_l(\mathbf{h}_1^{(l-1)}, ..., \mathbf{h}_n^{(l-1)})$$

Each layer refines the representation. Lower layers capture syntax (POS tags, dependency structure). Higher layers capture semantics (word sense, entity type, sentiment).

**Layer-wise behavior** (discovered via probing studies):
- **Layer 0** (embeddings): Static, context-free — similar to Word2Vec
- **Layers 1-4**: Mostly syntactic — POS tags, local agreement
- **Layers 5-8**: Increasingly semantic — word sense disambiguation
- **Layers 9-12**: Task-specific — the representations most useful for downstream tasks

### 7.4 Static vs Contextual: When to Use Each

| Use Case | Static | Contextual | Why |
|----------|--------|-----------|-----|
| Word similarity/analogy benchmarks | Good | Overkill | Simple task, static is sufficient |
| Named entity recognition | Decent | Much better | Context determines entity type |
| Sentiment analysis | Decent | Much better | "Not bad" requires context |
| Semantic search (RAG) | Inadequate | Essential | Query-document matching needs context |
| Machine translation | Inadequate | Essential | Same word has different translations by context |
| Initialization for downstream | Yes (pre-transformer era) | Yes (BERT era) | Contextual gives much better starting point |

---

## 8. Sentence & Document Embeddings

### 8.1 The Challenge

Word embeddings represent individual words. But for tasks like semantic search, document clustering, and RAG, we need embeddings for entire sentences or documents.

**Naive approaches**:
1. **Average word embeddings**: $\mathbf{s} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{w}_i$. Simple but loses word order and weighs all words equally. "Dog bites man" ≈ "Man bites dog".
2. **Weighted average**: Weight by IDF scores. Better — common words contribute less. But still loses order.
3. **[CLS] token from BERT**: Use the special classification token's output. Surprisingly bad for similarity tasks because BERT wasn't trained for similarity — [CLS] is trained for next sentence prediction, not semantic meaning.

### 8.2 Sentence-BERT (SBERT)

**Problem with BERT for similarity**: Using BERT directly requires feeding both sentences through BERT together (cross-encoder), which is $O(n^2)$ for all pairs. Comparing 10,000 sentences requires 50M forward passes — taking ~65 hours.

**Sentence-BERT** (Reimers & Gurevych, 2019): Fine-tune BERT as a **bi-encoder** that produces good sentence embeddings independently:

**Architecture**:
1. Feed sentence A through BERT → pool to get $\mathbf{u}$
2. Feed sentence B through BERT → pool to get $\mathbf{v}$
3. Compute similarity: $\cos(\mathbf{u}, \mathbf{v})$

**Pooling strategies**:
- **Mean pooling**: Average all token embeddings (best for SBERT)
- **[CLS] token**: Use only the CLS token embedding
- **Max pooling**: Element-wise maximum across token embeddings

**Training**:
- **NLI (Natural Language Inference) data**: Sentence pairs labeled as entailment, contradiction, neutral
- **Siamese/triplet network**: Minimize distance for entailment pairs, maximize for contradiction
- **Objective**: Softmax over the concatenated/difference of sentence embeddings

**Result**: 10,000 sentence comparisons now takes ~5 seconds (embeddings computed once, similarity is just dot product). Quality is nearly as good as the full cross-encoder BERT.

### 8.3 Contrastive Learning for Embeddings

Modern embedding models are trained with **contrastive learning**:

$$\mathcal{L}_i = -\log\frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i^+)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j)/\tau)}$$

**Positive pairs**: Same document in different augmentations, question-answer pairs, paraphrases.
**Negative pairs**: Other documents in the batch (in-batch negatives).
**Hard negatives**: Documents that are semantically close but not matches — crucial for quality.

### 8.4 Universal Sentence Encoder (USE)

Google's USE (2018) provides sentence embeddings trained on multiple tasks simultaneously:
- Skip-thought (predict surrounding sentences)
- Conversational response prediction
- Natural language inference

Two variants: Transformer-based (more accurate) and DAN (Deep Averaging Network, faster).

### 8.5 Doc2Vec

**Paragraph Vector** (Le & Mikolov, 2014): Extension of Word2Vec to documents.

**Distributed Memory (PV-DM)**: Like CBOW but with a document ID vector concatenated to the input:

$$P(w_t \mid w_{t-k}, ..., w_{t-1}, d) = \text{softmax}(\mathbf{W}[\mathbf{v}_{w_{t-k}}; ...; \mathbf{v}_{w_{t-1}}; \mathbf{v}_d])$$

The document vector $\mathbf{v}_d$ acts as a memory that captures the topic of the document.

Largely superseded by SBERT and modern embedding models, but conceptually important.

---

## 9. Embedding Spaces: Properties & Operations

### 9.1 Vector Arithmetic (Analogies)

The most famous property of Word2Vec embeddings:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

**More examples**:
- $\text{Paris} - \text{France} + \text{Italy} \approx \text{Rome}$ (capital-country)
- $\text{walking} - \text{walk} + \text{swim} \approx \text{swimming}$ (tense)
- $\text{bigger} - \text{big} + \text{small} \approx \text{smaller}$ (comparative)

**Why this works**: Word2Vec learns a linear structure where semantic relationships are captured as direction vectors. The "royalty" direction is roughly $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} \approx \mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{woman}}$.

**Limitations**: Analogies work well for frequent, well-represented relationships but fail for complex or rare relationships. They also suffer from bias: $\text{man} - \text{woman} + \text{doctor} \approx \text{nurse}$ reflects societal biases in the training data.

### 9.2 Similarity Measures

**Cosine similarity** (most common for embeddings):

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2} \sqrt{\sum_i v_i^2}}$$

Range: $[-1, 1]$ where 1 = identical direction, 0 = orthogonal, -1 = opposite.

**Why cosine over Euclidean for text**: Cosine ignores magnitude and focuses on direction. Two documents about the same topic but of different lengths would have different Euclidean distances but similar cosine similarity. The "direction" in embedding space captures meaning, while "magnitude" often captures frequency or length.

**Dot product**: $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\mathbf{u}, \mathbf{v})$. Used in attention (combines direction similarity with magnitude). Some embedding models are trained for dot product similarity (where magnitude carries information about relevance).

### 9.3 Clustering in Embedding Space

Embeddings naturally cluster by semantic similarity:
- Word embeddings: Country names cluster together, verbs cluster together
- Sentence embeddings: Documents about similar topics cluster together
- CLIP embeddings: Images and their descriptions cluster together

This is the foundation of:
- **Nearest neighbor retrieval**: Find the $k$ closest embeddings to a query (RAG)
- **Semantic clustering**: Group similar documents without labels
- **Anomaly detection**: Flag inputs far from any known cluster

### 9.4 Embedding Bias

Word embeddings encode societal biases from training data:

- Gender bias: "programmer" is closer to "man" than "woman"
- Racial bias: Names associated with certain ethnicities cluster with stereotypical attributes
- Professional bias: "CEO" closer to "man", "secretary" closer to "woman"

**Debiasing methods**:
1. **Post-hoc projection**: Identify the "gender direction" and project embeddings onto the orthogonal subspace (Bolukbasi et al., 2016)
2. **Data augmentation**: Balance gendered examples in training data
3. **Counterfactual data**: "He is a nurse" / "She is a nurse" with equal weight

**For interviews**: Know that bias exists, know one mitigation approach, and know it's an active research area with no perfect solution.

---

## 10. Modern Embedding Models for RAG & Search

### 10.1 The Embedding Model Landscape (2024-2026)

The field has moved far beyond Word2Vec. Modern embedding models are specifically designed for retrieval tasks:

| Model | Org | Dimensions | Max Tokens | Key Innovation |
|-------|-----|-----------|------------|---------------|
| text-embedding-3-large | OpenAI | 3072 | 8191 | Matryoshka embeddings (variable dim) |
| text-embedding-3-small | OpenAI | 1536 | 8191 | Cost-efficient |
| BGE-large-en-v1.5 | BAAI | 1024 | 512 | RetroMAE pretraining |
| E5-large-v2 | Microsoft | 1024 | 512 | "query: " / "passage: " prefixes |
| GTE-large | Alibaba | 1024 | 512 | Multi-task training |
| Cohere embed-v3 | Cohere | 1024 | 512 | Input type classification |
| Jina-embeddings-v2 | Jina | 768 | 8192 | Long context embeddings |
| Nomic-embed-text | Nomic | 768 | 8192 | Open-source, long context |

### 10.2 Training Modern Embedding Models

**Stage 1: Pretraining** (unsupervised)
- Start from a pretrained transformer (BERT, RoBERTa, or similar)
- Sometimes further pretrained with RetroMAE (masked autoencoder for retrieval)

**Stage 2: Contrastive fine-tuning**
- Positive pairs: question-answer, query-document, paraphrases
- In-batch negatives: other documents in the same batch
- Hard negatives: BM25-retrieved or cross-encoder-mined negatives
- Loss: InfoNCE with temperature $\tau \approx 0.02$

**Stage 3: Knowledge distillation** (optional)
- Distill from a cross-encoder (more accurate but slower) into the bi-encoder
- Train the bi-encoder to match the cross-encoder's similarity scores

### 10.3 Matryoshka Representation Learning (MRL)

**Problem**: Different applications need different dimensionalities. Search might need 256 dims for speed, classification might benefit from 1024 dims.

**MRL** (Kusupati et al., 2022): Train the model so that the **first $d'$ dimensions** of the full $d$-dimensional embedding are a good $d'$-dimensional embedding, for any $d' \leq d$.

**Training**: Add contrastive loss terms at multiple truncation points:

$$\mathcal{L} = \sum_{d' \in \{32, 64, 128, 256, 512, 1024, ...\}} \alpha_{d'} \cdot \mathcal{L}_{d'}$$

where $\mathcal{L}_{d'}$ is the contrastive loss using only the first $d'$ dimensions.

**Result**: You can truncate embeddings at deployment time to trade quality for speed/storage, without retraining. OpenAI's text-embedding-3 models support this.

### 10.4 Bi-Encoder vs Cross-Encoder

**Bi-encoder** (SBERT-style):
- Encode query and document independently
- Similarity = dot product or cosine of embeddings
- **Fast**: Embed once, compare with any query
- **Scalable**: Works with approximate nearest neighbor (ANN) indexes
- **Quality**: Good but not as fine-grained

**Cross-encoder** (BERT-style):
- Concatenate query and document, feed through one model
- Output = similarity score (regression head)
- **Slow**: Must run the model for every (query, document) pair
- **Not indexable**: Can't pre-compute anything
- **Quality**: Better — models full token-level interaction

**In practice** (RAG pipeline): Use bi-encoder for initial retrieval (top 100-1000), then cross-encoder for reranking (top 10-20). This gives cross-encoder quality at bi-encoder speed.

### 10.5 ColBERT: Late Interaction

**ColBERT** (Khattab & Zaharia, 2020): A middle ground between bi-encoder and cross-encoder.

1. Encode query tokens independently: $\{q_1, q_2, ..., q_m\}$
2. Encode document tokens independently: $\{d_1, d_2, ..., d_n\}$
3. Compute MaxSim: $\text{score} = \sum_i \max_j (q_i^T d_j)$

**Token-level interaction** at the scoring stage, but **independent encoding** at the embedding stage. This allows document embeddings to be precomputed and indexed while still capturing fine-grained relevance.

---

## 11. Interview Questions & Answers

### Q1: Derive the Skip-gram objective. Why is negative sampling necessary?

**Answer**: Skip-gram aims to maximize the probability of context words given a center word. For center word $w_c$ in a corpus of $T$ words with context window $m$:

$$\mathcal{J} = \frac{1}{T}\sum_{t=1}^{T}\sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} \mid w_t)$$

The conditional probability uses softmax over the vocabulary:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{v}'_{w_o}{}^T \mathbf{v}_{w_c})}{\sum_{w=1}^{|V|}\exp(\mathbf{v}'_w{}^T \mathbf{v}_{w_c})}$$

The denominator sums over the entire vocabulary $|V|$ (often 100K-1M words). This normalization constant must be computed at every training step, making each step $O(|V|)$ — prohibitively expensive.

**Negative sampling** reformulates this as binary classification. Instead of predicting the exact context word via softmax, we train a binary classifier to distinguish real (center, context) pairs from random pairs:

$$\mathcal{J} = \log\sigma(\mathbf{v}'_{w_o}{}^T \mathbf{v}_{w_c}) + \sum_{i=1}^{k}\mathbb{E}_{w_i \sim P_n}\left[\log\sigma(-\mathbf{v}'_{w_i}{}^T \mathbf{v}_{w_c})\right]$$

This reduces per-step cost from $O(|V|)$ to $O(k)$ where $k \approx 5\text{-}20$. Levy & Goldberg (2014) showed this implicitly factorizes a shifted PMI matrix, providing theoretical grounding for this approximation.

### Q2: How does GloVe differ from Word2Vec? What "global" information does it capture?

**Answer**: **Word2Vec** is a **predictive** model that uses local context windows. During training, it slides a window across the corpus and predicts context words from center words (skip-gram) or vice versa (CBOW). Each training example uses only a small window of text.

**GloVe** is a **count-based** model that first builds a **global co-occurrence matrix** $X_{ij}$ counting how often word $j$ appears near word $i$ across the entire corpus. It then factorizes this matrix by learning embeddings such that $\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}$.

The "global" information is the aggregate co-occurrence statistics. While Word2Vec processes one window at a time (local), GloVe's co-occurrence matrix captures the full distributional pattern across the entire corpus in one shot. This allows GloVe to learn from the ratios of co-occurrence probabilities — for example, "ice" co-occurring with "solid" much more than "steam" does, captured globally across millions of documents.

In practice, Levy & Goldberg showed the two approaches are mathematically related — Word2Vec with negative sampling implicitly factorizes a shifted PMI matrix, which is closely related to GloVe's log co-occurrence matrix. Empirical performance is similar; the choice often comes down to implementation preference.

### Q3: Why can't Word2Vec handle polysemy? How do contextual embeddings solve this?

**Answer**: Word2Vec assigns **one vector per word type** — the word "bank" gets a single embedding regardless of whether it means a financial institution or a riverbank. This embedding becomes an awkward average of all senses, weighted by their frequency in the training data.

If "financial bank" appears 70% of the time and "river bank" 30%, the embedding will be closer to financial-related words, making it systematically misleading for river bank contexts.

**Contextual embeddings** (BERT, GPT) solve this through self-attention. Each token's representation at layer $l$ is a function of **all tokens in the sequence**:

$$\mathbf{h}_i^{(l)} = f(\mathbf{h}_1^{(l-1)}, ..., \mathbf{h}_n^{(l-1)})$$

When "bank" appears next to "river" and "fishing", attention incorporates these context words, pushing the representation toward the river bank sense. When it appears next to "deposit" and "account", the representation shifts toward the financial sense.

Probing studies show this happens progressively through layers: layer 0 embeddings are context-free (like Word2Vec), and by layers 6-8, word sense disambiguation has largely occurred based on context.

### Q4: Explain how Sentence-BERT creates sentence embeddings and why cosine similarity works.

**Answer**: BERT was not designed for sentence similarity. Using BERT as a cross-encoder (feeding sentence pairs together) is accurate but $O(n^2)$ for $n$ sentences. BERT's [CLS] token output, taken independently per sentence, performs poorly for similarity because it was trained for NSP, not semantic representation.

**Sentence-BERT** fine-tunes BERT in a **Siamese architecture**:
1. Feed sentence A through BERT, apply mean pooling over token outputs → embedding $\mathbf{u}$
2. Feed sentence B through the same BERT, apply mean pooling → embedding $\mathbf{v}$
3. Train on NLI data using a classification objective over $[\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|]$, or a contrastive objective directly on similarity.

After training, the mean-pooled output is a semantically meaningful sentence embedding.

**Why cosine similarity works**: Cosine measures the angle between vectors, ignoring magnitude. SBERT's contrastive training explicitly optimizes for cosine similarity — positive pairs are pushed to have cosine near 1, negatives near 0 (or -1). The training objective shapes the embedding space so that the direction of a vector encodes semantic content, making cosine the natural similarity measure. Additionally, cosine is invariant to sentence length (via magnitude normalization), which is important since longer sentences shouldn't automatically be "more similar" to other long sentences.

### Q5: What is TF-IDF? How does it relate to information theory?

**Answer**: TF-IDF weights each term by its frequency in a document (TF) multiplied by its rarity across the corpus (IDF):

$$\text{TF-IDF}(t, d) = \frac{f(t,d)}{\sum_{t'} f(t', d)} \times \log\frac{N}{df(t)}$$

**IDF as self-information**: $\text{IDF}(t) = \log\frac{N}{df(t)} \approx -\log P(t)$ where $P(t) = df(t)/N$ is the probability a random document contains term $t$. This is exactly the **self-information** (surprisal) from information theory. Rare events carry more information — seeing "eigendecomposition" in a document tells you much more about its topic than seeing "the".

**Connection to entropy**: A document's TF-IDF representation concentrates weight on its most informative terms — those with high surprisal. This is related to the principle of maximum discrimination: the best features for distinguishing documents are those with high variance in their occurrence pattern across the corpus.

**BM25 as the modern successor**: BM25 adds term frequency saturation ($k_1$ parameter: after a word appears enough times, additional occurrences add diminishing value) and document length normalization ($b$ parameter: longer documents shouldn't be penalized for having more word occurrences). BM25 remains the gold standard for lexical matching in hybrid RAG systems.

### Q6: Compare bi-encoder and cross-encoder approaches for semantic search. When do you use each?

**Answer**: 

**Bi-encoder**: Encodes query and document independently into fixed-size vectors. Similarity is computed via dot product or cosine. Documents can be pre-embedded and indexed in a vector database (FAISS, Qdrant, etc.) for sub-millisecond retrieval.
- **Pros**: Fast ($O(1)$ per query-doc comparison after indexing), scalable to billions of documents
- **Cons**: No token-level interaction between query and document — can miss subtle relevance signals

**Cross-encoder**: Concatenates query and document as a single input: `[CLS] query [SEP] document [SEP]`. Full self-attention over all tokens captures detailed interactions.
- **Pros**: Much more accurate — models word-level alignment and complex relevance patterns
- **Cons**: $O(n)$ forward passes per query (one per candidate document), cannot be pre-computed

**In practice** (the two-stage pipeline):
1. **Stage 1 — Retrieval**: Bi-encoder retrieves top-100 candidates from millions/billions of documents. Sub-second latency.
2. **Stage 2 — Reranking**: Cross-encoder reranks the top-100 to produce final top-10. Adds ~100ms latency.

This gives you cross-encoder quality at bi-encoder scale. ColBERT offers a middle ground with token-level late interaction that's faster than cross-encoding but more accurate than simple bi-encoding.

### Q7: What is the relationship between Word2Vec and matrix factorization?

**Answer**: Levy & Goldberg (2014) proved that Word2Vec's skip-gram with negative sampling is **implicitly factorizing a shifted PMI matrix**.

Specifically, the skip-gram objective converges to a solution where:

$$\mathbf{w}_i^T \mathbf{c}_j = \text{PMI}(i, j) - \log k$$

where $\text{PMI}(i, j) = \log\frac{P(i, j)}{P(i)P(j)}$ is the Pointwise Mutual Information and $k$ is the number of negative samples.

This means Word2Vec is essentially performing **low-rank matrix factorization** (like SVD) on the PMI matrix. The PMI matrix captures how much more (or less) likely two words are to co-occur than chance.

GloVe makes this explicit: it directly factorizes the log co-occurrence matrix $\log X_{ij}$.

**Implication**: The linear algebraic structure of word embeddings (analogies, clustering) emerges from this matrix factorization perspective. The embedding space captures the principal directions of variation in word co-occurrence patterns — which correspond to semantic dimensions like gender, tense, country-capital relationships, etc.

### Q8: How do modern embedding models differ from Word2Vec/GloVe?

**Answer**: Modern embedding models (BGE, E5, text-embedding-3, etc.) differ in several fundamental ways:

1. **Architecture**: Transformer-based (BERT/RoBERTa backbone) vs shallow networks (Word2Vec) or matrix factorization (GloVe). Transformers produce contextual representations.

2. **Granularity**: Modern models produce sentence/passage embeddings, not just word embeddings. They're trained end-to-end for retrieval tasks.

3. **Training**: Multi-stage contrastive learning with carefully curated positive pairs (query-passage matches from search logs) and hard negatives (BM25-mined or cross-encoder-mined). Word2Vec only used co-occurrence.

4. **Scale**: Trained on millions of query-document pairs with cross-encoder distillation. Word2Vec trained on raw text co-occurrence.

5. **Features**: Support long inputs (8K+ tokens), instruction prefixes ("query: " vs "passage: "), Matryoshka dimensionality reduction, and typed inputs (search query vs classification).

6. **Quality gap**: On the MTEB benchmark (Massive Text Embedding Benchmark), modern models score 60-70% average, while Word2Vec-based approaches score below 40%. The gap is largest on retrieval and clustering tasks.

### Q9: Explain the concept of embedding spaces and why analogy arithmetic works in Word2Vec.

**Answer**: Word2Vec learns to map words into a continuous vector space $\mathbb{R}^d$ such that semantic relationships correspond to geometric relationships. Specifically, **consistent semantic relationships are encoded as approximately parallel direction vectors**.

The king-queen example: $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{queen}} \approx \mathbf{v}_{\text{man}} - \mathbf{v}_{\text{woman}}$. This means the "gender" direction is approximately constant across word pairs that differ primarily in gender.

**Why this emerges from training**: The skip-gram objective forces words with similar contexts to have similar embeddings. "King" and "queen" appear in nearly identical contexts (royalty, throne, crown), except where gender-specific words appear. The optimization naturally separates the "royalty" component (shared) from the "gender" component (different), creating a linear subspace for each relationship type.

**From the matrix factorization view**: The embedding space captures the principal components of the PMI matrix. Different semantic relationships (gender, tense, geography) correspond to different dimensions or linear combinations of dimensions. Since PMI is a linear function of co-occurrence probabilities, and relationships manifest as consistent co-occurrence shifts, the result is approximately linear vector arithmetic.

**Limitations**: This works best for simple, frequent, consistent relationships. It fails for: (a) complex semantic relationships, (b) polysemous words (averaged senses), (c) rare words (insufficient training signal), (d) cultural/social biases (reflects training data biases as geometric relationships).

### Q10: How would you choose an embedding model for a new RAG system?

**Answer**: Decision framework:

1. **Check MTEB leaderboard** for your language and task type (retrieval, clustering, classification). This gives a baseline ranking.

2. **Evaluate on your domain**: Generic benchmarks don't capture domain-specific performance. Test on 200-500 query-document pairs from your actual use case. Key metrics: Recall@10, MRR@10, NDCG@10.

3. **Consider constraints**:
   - **Latency**: Smaller dimensions (256-512) for real-time. Matryoshka models let you tune this.
   - **Cost**: Open-source (BGE, E5, Nomic) vs API (OpenAI, Cohere). For 10M+ documents, hosting your own model is often cheaper.
   - **Input length**: Most models max at 512 tokens. For long documents, use Jina v2 (8K) or chunk documents.
   - **Language**: Multilingual models (E5-multilingual, BGE-M3) if non-English content.

4. **Architecture choices**:
   - Bi-encoder only: Fastest, simplest
   - Bi-encoder + cross-encoder reranker: Best quality at moderate latency
   - ColBERT: Good middle ground for token-level matching

5. **Fine-tuning**: If generic models underperform on your domain by >5%, fine-tune on domain-specific query-document pairs. Even 1000-5000 pairs with hard negatives can significantly boost performance. Use a cross-encoder teacher for distillation if you can afford it.

---

*Next topic: [Topic 7: Sequence Modeling](07_Sequence_Modeling.md)*
