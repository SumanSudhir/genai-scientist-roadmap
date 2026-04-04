# Topic 20: Retrieval-Augmented Generation (RAG)

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 20 of 28
> **Scope**: RAG pipeline end-to-end, embedding models, vector databases, ANN algorithms (HNSW, IVF, PQ), chunking strategies, retrieval methods (dense, sparse, hybrid), re-ranking (cross-encoder, ColBERT), advanced RAG (HyDE, multi-query, RAPTOR, agentic RAG, CRAG), evaluation (RAGAS), RAG vs fine-tuning vs long context
> **Why this matters**: RAG is asked in almost every applied AI interview. It's the primary method for grounding LLMs in factual, current, or proprietary knowledge. Building a production RAG system requires understanding embeddings, retrieval algorithms, chunking, re-ranking, and evaluation — a full-stack AI engineering challenge. Interviewers want you to design a RAG system from scratch and reason about trade-offs at every stage.
> **Previous**: [Topic 19: In-Context Learning (Theory)](19_In_Context_Learning.md)
> **Next**: [Topic 21: Agents & Tool Use](21_Agents_Tool_Use.md)

---

## Table of Contents

1. [Why RAG Exists — The Problem It Solves](#1-why-rag-exists--the-problem-it-solves)
2. [The RAG Pipeline — End to End](#2-the-rag-pipeline--end-to-end)
3. [Embedding Models — Turning Text into Vectors](#3-embedding-models--turning-text-into-vectors)
4. [Chunking Strategies — How to Split Documents](#4-chunking-strategies--how-to-split-documents)
5. [Vector Databases & ANN Algorithms](#5-vector-databases--ann-algorithms)
6. [Retrieval Methods — Dense, Sparse, Hybrid](#6-retrieval-methods--dense-sparse-hybrid)
7. [Re-Ranking — The Second Stage](#7-re-ranking--the-second-stage)
8. [Advanced RAG Techniques](#8-advanced-rag-techniques)
9. [RAG Evaluation — How to Measure Quality](#9-rag-evaluation--how-to-measure-quality)
10. [RAG vs Fine-Tuning vs Long Context](#10-rag-vs-fine-tuning-vs-long-context)
11. [Production RAG — System Design Considerations](#11-production-rag--system-design-considerations)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. Why RAG Exists — The Problem It Solves

### 1.1 The Limitations of Standalone LLMs

LLMs have three fundamental knowledge problems:

**1. Knowledge cutoff**: The model only knows what was in its training data. It cannot answer questions about events after its training date.

**2. Hallucination**: When the model doesn't know something, it often generates plausible-sounding but incorrect information — confidently.

**3. No access to private data**: The model has never seen your company's internal documents, codebase, or customer data. It cannot answer questions about them.

### 1.2 The RAG Solution

**Retrieval-Augmented Generation** (Lewis et al., 2020) solves these problems by giving the LLM access to external knowledge at inference time:

$$
P(y \mid q) = P(y \mid q, \text{retrieve}(q, \mathcal{D}))
$$

Instead of relying solely on parametric knowledge (stored in weights), the model also uses **non-parametric knowledge** (retrieved documents):

```
User query: "What was our Q3 revenue?"
        │
        ▼
    Retrieve relevant documents from company database
        │
        ▼
    Augment the prompt with retrieved context
        │
        ▼
    LLM generates answer grounded in the retrieved documents
        │
        ▼
    Answer: "Q3 revenue was $4.2M, a 15% increase YoY."
    [Source: Q3 Financial Report, page 12]
```

### 1.3 Why RAG, Not Just Fine-Tuning?

| Problem | Fine-Tuning | RAG |
|---------|------------|-----|
| Knowledge updates | Must retrain (expensive) | Update the document store (cheap) |
| Factual grounding | Model still hallucinates | Can cite sources, verify claims |
| Private data | Data baked into weights (privacy risk) | Documents stay in your infra |
| Cost | $100-$10,000+ per fine-tune run | ~$0.001 per query (embedding + retrieval) |
| Transparency | Black box ("how did it know this?") | Citations ("retrieved from doc X, page Y") |

RAG is the **default first approach** for grounding LLMs in specific knowledge. Fine-tuning is for behavior/style changes (see [Topic 15](15_Fine_Tuning_PEFT.md)).

---

## 2. The RAG Pipeline — End to End

### 2.1 The Two Phases

RAG has two distinct phases:

**Phase 1: Indexing** (offline, done once)

$$
\text{Documents} \xrightarrow{\text{chunk}} \text{Chunks} \xrightarrow{\text{embed}} \text{Vectors} \xrightarrow{\text{index}} \text{Vector DB}
$$

**Phase 2: Querying** (online, per user query)

$$
\text{Query} \xrightarrow{\text{embed}} \text{Query Vector} \xrightarrow{\text{search}} \text{Top-K Chunks} \xrightarrow{\text{augment}} \text{LLM Prompt} \xrightarrow{\text{generate}} \text{Answer}
$$

### 2.2 Full Pipeline Diagram

```
                         INDEXING (Offline)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Documents ──► Chunking ──► Embedding ──► Vector Store   │
│  (PDF, HTML,   (split into   (dense      (FAISS,        │
│   Markdown,     overlapping   vectors)    Pinecone,      │
│   etc.)         passages)                 Qdrant)        │
│                                                          │
└──────────────────────────────────────────────────────────┘

                         QUERYING (Online)
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  User Query ──► Embed Query ──► Vector Search ──►        │
│                                  (ANN: top-K)            │
│                                       │                  │
│                                       ▼                  │
│                                  (Optional)              │
│                                  Re-Ranking              │
│                                       │                  │
│                                       ▼                  │
│                                  Augment Prompt          │
│                                       │                  │
│                                       ▼                  │
│                                  LLM Generation          │
│                                       │                  │
│                                       ▼                  │
│                                  Answer + Sources        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.3 The Augmented Prompt

The retrieved chunks are inserted into the LLM prompt:

```
System: You are a helpful assistant. Answer questions based on the
provided context. If the context doesn't contain the answer, say
"I don't have enough information."

Context:
[Retrieved Chunk 1]: "Q3 2025 revenue reached $4.2M, representing
a 15% year-over-year increase driven by enterprise expansion..."

[Retrieved Chunk 2]: "The enterprise segment contributed $3.1M,
up from $2.4M in Q3 2024..."

[Retrieved Chunk 3]: "Operating expenses were $3.8M, resulting in
an operating margin of 9.5%..."

User: What was our Q3 revenue and what drove the growth?
```

The LLM then generates an answer **grounded** in the retrieved context, rather than relying on its parametric knowledge.

---

## 3. Embedding Models — Turning Text into Vectors

### 3.1 What Embedding Models Do

An embedding model maps text to a dense vector in a continuous space where **semantic similarity corresponds to geometric proximity**:

$$
f: \text{text} \rightarrow \mathbb{R}^d
$$

$$
\text{sim}(A, B) = \cos(\mathbf{e}_A, \mathbf{e}_B) = \frac{\mathbf{e}_A \cdot \mathbf{e}_B}{\|\mathbf{e}_A\| \|\mathbf{e}_B\|}
$$

Texts with similar meaning should have high cosine similarity, regardless of surface-level wording:
- "What is the capital of France?" ↔ "France's capital city" → sim ≈ 0.92
- "What is the capital of France?" ↔ "French cuisine recipes" → sim ≈ 0.35

### 3.2 How Embedding Models Are Trained

Modern embedding models (BGE, E5, GTE, OpenAI) use **contrastive learning**:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau)}{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+) / \tau) + \sum_{j} \exp(\text{sim}(\mathbf{q}, \mathbf{d}_j^-) / \tau)}
$$

where:
- $(\mathbf{q}, \mathbf{d}^+)$ is a positive pair (query and its relevant document)
- $\mathbf{d}_j^-$ are negative documents (not relevant to the query)
- $\tau$ is a temperature parameter

This is the **InfoNCE loss** — the same principle as CLIP (see [Topic 22](22_Multimodal_AI.md)).

### 3.3 Training Data Sources

| Stage | Data | Purpose |
|-------|------|---------|
| **Pretraining** | Unsupervised text pairs (title-body, question-answer, etc.) | Learn general semantic similarity |
| **Fine-tuning (hard negatives)** | Curated pairs with hard negatives | Learn fine-grained distinctions |
| **Fine-tuning (instructions)** | Task-prefixed pairs ("Represent this query for retrieval: ...") | Task-aware embeddings |

**Hard negatives** are critical: instead of random negatives (easy to distinguish), use negatives that are semantically similar but not relevant. Example:
- Query: "Python error handling best practices"
- Hard negative: "Python variable naming conventions" (same topic, wrong subtopic)
- Easy negative: "Italian pasta recipes" (obviously irrelevant)

### 3.4 Key Embedding Models (2024-2026)

| Model | Dimensions | Max Tokens | Key Feature | Open? |
|-------|-----------|-----------|-------------|-------|
| **BGE-large** (BAAI) | 1024 | 512 | Instruction-aware, strong retrieval | Yes |
| **BGE-M3** (BAAI) | 1024 | 8192 | Multi-lingual, multi-granularity, multi-function | Yes |
| **E5-large-v2** (Microsoft) | 1024 | 512 | "Embed Everything Everywhere" | Yes |
| **E5-Mistral-7B** (Microsoft) | 4096 | 32768 | LLM-based embeddings | Yes |
| **GTE-large** (Alibaba) | 1024 | 512 | Strong retrieval + classification | Yes |
| **Nomic Embed v1.5** | 768 | 8192 | Matryoshka, long context | Yes |
| **text-embedding-3-large** (OpenAI) | 3072 | 8191 | Matryoshka, strong commercial | No |
| **Cohere Embed v3** | 1024 | 512 | Multilingual, compression | No |
| **Voyage-3** (Voyage AI) | 1024 | 32000 | Strong code + retrieval | No |

### 3.5 Bi-Encoder vs Cross-Encoder

This is a fundamental architectural distinction:

**Bi-encoder** (used for retrieval):
- Encode query and document **independently**
- Compare via cosine similarity
- Fast: precompute document embeddings, compare with O(1) per pair
- Scalable: search millions of documents in milliseconds

$$
\text{score}(q, d) = \cos(f(q), f(d))
$$

**Cross-encoder** (used for re-ranking):
- Encode query and document **together** through the full model
- Produces a single relevance score
- Slow: requires a full forward pass per (query, document) pair
- Accurate: captures fine-grained query-document interactions

$$
\text{score}(q, d) = g([\text{CLS}] \; q \; [\text{SEP}] \; d \; [\text{SEP}])
$$

```
Bi-encoder:                    Cross-encoder:
Query ──► Encoder ──► vec_q    Query + Doc ──► Encoder ──► score
                       ↓
Doc ──► Encoder ──► vec_d      (full interaction between
                       ↓        query and document tokens)
          cosine(vec_q, vec_d)
```

**In RAG**: Use bi-encoder for initial retrieval (fast, over millions of docs), then cross-encoder for re-ranking the top-K results (slow, but only over 10-100 docs).

### 3.6 Matryoshka Representation Learning (MRL)

A technique where embeddings are trained so that **any prefix of the full embedding is a valid, lower-dimensional embedding**:

$$
\text{Quality}(\mathbf{e}[1:k]) \approx \text{Quality}(\mathbf{e}[1:d]) \quad \text{for reasonable } k
$$

Example: A 1024-dim embedding can be truncated to 256 dims with only ~2% quality loss.

**Why this matters**: Storage and search cost scale with dimensionality. MRL lets you choose the quality-cost trade-off at deployment time without retraining.

Used by: OpenAI text-embedding-3, Nomic Embed, BGE-M3.

---

## 4. Chunking Strategies — How to Split Documents

Chunking is how you break long documents into passages for embedding and retrieval. It's deceptively important — poor chunking is one of the most common causes of RAG failure.

### 4.1 Why Chunking Matters

1. **Embedding models have token limits** (512-8192 tokens). Documents must fit within these limits.
2. **Retrieval precision**: Small, focused chunks are more likely to be relevant to a specific query than entire documents.
3. **Context window limits**: The LLM's context window constrains how many chunks can be included.
4. **Semantic coherence**: A chunk should contain a complete, self-contained idea. Splitting mid-sentence or mid-paragraph degrades retrieval quality.

### 4.2 Chunking Methods

#### Fixed-Size Chunking

Split text into chunks of exactly $N$ tokens with overlap of $O$ tokens:

```
Document: [████████████████████████████████████████]

Chunk 1:  [████████████]
Chunk 2:       [████████████]         (overlap)
Chunk 3:            [████████████]
Chunk 4:                 [████████████]
```

| Parameter | Typical Value |
|-----------|--------------|
| Chunk size | 256-1024 tokens |
| Overlap | 10-20% of chunk size |

**Pros**: Simple, consistent chunk sizes, easy to implement.
**Cons**: Splits mid-sentence, mid-paragraph, or mid-section. No semantic awareness.

#### Recursive Character Splitting

Split hierarchically using a list of separators, trying each in order:

```python
separators = ["\n\n", "\n", ". ", " ", ""]
# Try to split by paragraphs first, then sentences, then words
```

If a chunk exceeds the size limit, split at the highest-level separator that fits.

**Pros**: Respects document structure better than fixed-size.
**Cons**: Chunks vary in size; still doesn't understand semantics.

#### Semantic Chunking

Use an embedding model to detect **topic boundaries**:

1. Embed each sentence
2. Compute cosine similarity between consecutive sentences
3. Split where similarity drops below a threshold (topic change)

```
Sentence similarities: [0.92, 0.89, 0.91, 0.34, 0.88, 0.90, 0.85, 0.28, 0.91]
                                            ↑ split here              ↑ split here
```

**Pros**: Chunks are semantically coherent — each chunk is about one topic.
**Cons**: Expensive (must embed every sentence), variable chunk sizes.

#### Document-Structure-Aware Chunking

Use the document's inherent structure:

| Document Type | Chunking Strategy |
|--------------|------------------|
| Markdown/HTML | Split by headers (h1, h2, h3) |
| PDF | Split by sections, pages |
| Code | Split by function/class/file |
| Legal/Academic | Split by section/article/paragraph |
| Conversation logs | Split by turns or topics |

**Pros**: Preserves the author's intended organization.
**Cons**: Requires document-type-specific parsers; sections may be too long or too short.

### 4.3 The Chunk Size Trade-off

| Smaller Chunks (128-256 tokens) | Larger Chunks (512-1024 tokens) |
|-------------------------------|-------------------------------|
| More precise retrieval | More context per chunk |
| May lose surrounding context | May include irrelevant information |
| More chunks to search | Fewer chunks to search |
| Higher recall (find more relevant pieces) | Higher self-containedness |

**The sweet spot** for most use cases: **256-512 tokens** with 10-20% overlap.

### 4.4 Parent-Child Chunking (Small-to-Big)

A sophisticated strategy that retrieves with **small chunks** but returns **larger context**:

```
Document
├── Parent Chunk (1024 tokens) ──── returned to LLM
│   ├── Child Chunk 1 (256 tokens) ──── used for retrieval
│   ├── Child Chunk 2 (256 tokens) ──── used for retrieval
│   └── Child Chunk 3 (256 tokens) ──── used for retrieval
├── Parent Chunk (1024 tokens)
│   ├── Child Chunk 4 (256 tokens)
...
```

1. **Index** the small child chunks (precise retrieval)
2. When a child chunk matches, **return the parent chunk** (more context)

**Why this works**: Small chunks give precise retrieval; large parent chunks give the LLM enough context to understand the answer. Best of both worlds.

### 4.5 Contextual Retrieval (Anthropic, 2024)

Prepend **contextual descriptions** to each chunk before embedding:

```
Original chunk: "Revenue increased by 15% to $4.2M."

Contextualized chunk: "This chunk is from the Q3 2025 Financial Report,
Section 3: Revenue Analysis. Revenue increased by 15% to $4.2M."
```

The added context helps the embedding model understand what the chunk is about, even without seeing the surrounding document. Anthropic showed this reduces retrieval failures by ~49%.

---

## 5. Vector Databases & ANN Algorithms

### 5.1 The Problem: Exact Nearest Neighbor Is Too Slow

Given $N$ document vectors of dimension $d$, finding the exact nearest neighbor to a query vector requires:

$$
\text{Time}_{\text{exact}} = O(N \cdot d)
$$

For $N = 10M$ documents and $d = 1024$: ~10 billion operations per query. Too slow for real-time applications.

**Solution**: Approximate Nearest Neighbor (ANN) algorithms that trade a small amount of accuracy for massive speedup.

### 5.2 Key ANN Algorithms

#### HNSW (Hierarchical Navigable Small World)

The most popular ANN algorithm for RAG.

**Idea**: Build a multi-layer graph where:
- Bottom layer: All vectors connected to their nearest neighbors
- Upper layers: Progressively sparser, containing only a subset of vectors
- Search: Start at the top layer (coarse), navigate down to the bottom (fine)

```
Layer 2:    A ─────────── D              (sparse: few nodes)
            │             │
Layer 1:    A ──── C ──── D ──── F       (medium density)
            │      │      │      │
Layer 0:    A ─ B ─ C ─ D ─ E ─ F ─ G   (dense: all nodes)
```

**Search process**:
1. Start at a random entry point in the top layer
2. Greedily move to the neighbor closest to the query
3. When no closer neighbor exists, drop to the next layer
4. Repeat until reaching layer 0
5. Return the nearest neighbors found

**Complexity**: $O(\log N)$ search time (exponential speedup over brute force).

**Properties**:
| Aspect | HNSW |
|--------|------|
| Build time | $O(N \log N)$ |
| Search time | $O(\log N)$ |
| Memory | High (stores the graph + all vectors) |
| Accuracy | Very high (recall > 95% typical) |
| Dynamic updates | Yes (can add/remove vectors) |

#### IVF (Inverted File Index)

**Idea**: Partition the vector space into clusters using k-means. At query time, only search the nearest clusters.

1. **Indexing**: Run k-means to create $C$ centroids (e.g., $C = 1024$). Assign each vector to its nearest centroid.
2. **Search**: Find the $n_{\text{probe}}$ nearest centroids to the query, then search only vectors in those clusters.

$$
\text{Search cost} = O(n_{\text{probe}} \cdot N/C \cdot d)
$$

With $C = 1024$ and $n_{\text{probe}} = 10$: search only ~1% of vectors.

```
┌────────────┐  ┌────────────┐  ┌────────────┐
│ Cluster 1  │  │ Cluster 2  │  │ Cluster 3  │
│  ● ● ●     │  │  ● ● ●     │  │  ● ●       │
│  ● ●       │  │  ● ● ● ●   │  │  ● ● ●     │
│  ★ centroid│  │  ★ centroid│  │  ★ centroid│
└────────────┘  └────────────┘  └────────────┘

Query ◆ → Find nearest centroids → Search only those clusters
```

#### Product Quantization (PQ)

**Idea**: Compress vectors to reduce memory and speed up distance computation.

1. Split each $d$-dimensional vector into $M$ sub-vectors of dimension $d/M$
2. Cluster each sub-vector space independently into $K$ centroids (e.g., $K = 256$)
3. Represent each vector as $M$ centroid IDs (each 1 byte if $K = 256$)

$$
\text{Original}: d \times 4 \text{ bytes (float32)} = 4096 \text{ bytes for } d = 1024
$$

$$
\text{Compressed}: M \text{ bytes} = 128 \text{ bytes for } M = 128 \quad \text{(32× compression)}
$$

**Distance computation**: Precompute distances from the query to all centroids, then look up and sum — much faster than computing full dot products.

#### Combined: IVF-PQ

The practical choice for large-scale search: IVF for coarse partitioning + PQ for compressed storage.

$$
\text{IVF-PQ}: O(n_{\text{probe}} \cdot N/C \cdot M) \text{ with } M \ll d
$$

### 5.3 Vector Database Comparison

| Database | Type | ANN Algorithm | Key Feature | Best For |
|----------|------|--------------|-------------|----------|
| **FAISS** (Meta) | Library | IVF, HNSW, PQ, flat | Highly optimized, GPU support | High-performance, self-hosted |
| **Pinecone** | Managed cloud | Proprietary | Fully managed, serverless | Quick start, no ops burden |
| **Qdrant** | Self-hosted/cloud | HNSW | Rust-based, fast, rich filtering | Production self-hosted |
| **Weaviate** | Self-hosted/cloud | HNSW | GraphQL API, hybrid search | Hybrid search needs |
| **Chroma** | Embedded | HNSW (hnswlib) | Simple API, Python-native | Prototyping, small scale |
| **Milvus** | Self-hosted/cloud | IVF, HNSW, DiskANN | Distributed, billion-scale | Very large scale |
| **pgvector** | PostgreSQL extension | IVF, HNSW | Integrates with existing Postgres | Already using Postgres |

### 5.4 Choosing the Right Setup

```
< 100K vectors    → FAISS (flat) or Chroma — exact search is fine
100K-10M vectors  → Qdrant or FAISS (HNSW) — single machine
10M-1B vectors    → Milvus or Pinecone — distributed
> 1B vectors      → Milvus or custom FAISS clusters — distributed + PQ
```

---

## 6. Retrieval Methods — Dense, Sparse, Hybrid

### 6.1 Dense Retrieval (Semantic Search)

Use embedding models to encode query and documents into dense vectors. Retrieve by vector similarity.

$$
\text{score}(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d)
$$

**Strengths**:
- Captures **semantic** similarity ("automobile" ↔ "car")
- Handles paraphrases and synonyms
- Works across languages (with multilingual models)

**Weaknesses**:
- Struggles with **exact keyword matching** ("error code E4032" may not match "E4032")
- Requires good embedding models — poor models give poor retrieval
- Computationally more expensive than sparse methods

### 6.2 Sparse Retrieval (Lexical Search)

Traditional keyword-based retrieval using term frequency statistics.

#### BM25 (Best Match 25)

The gold standard for sparse retrieval:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}
$$

where:
- $f(t, d)$ = frequency of term $t$ in document $d$
- $|d|$ = document length, $avgdl$ = average document length
- $\text{IDF}(t) = \log\frac{N - n(t) + 0.5}{n(t) + 0.5}$ (inverse document frequency)
- $k_1 \approx 1.5$, $b \approx 0.75$ (tuning parameters)

**Strengths**:
- Excellent at **exact keyword matching** (product names, error codes, IDs)
- Fast — inverted index lookup is $O(1)$ per term
- No ML model needed — purely statistical
- Battle-tested over decades

**Weaknesses**:
- No semantic understanding ("automobile" won't match "car")
- Fails with vocabulary mismatch between query and document
- Sensitive to query phrasing

#### Learned Sparse Retrieval (SPLADE)

**SPLADE** (Formal et al., 2021) uses a BERT-like model to produce **sparse** representations where each dimension corresponds to a vocabulary term:

$$
\mathbf{w}_d = \max_{t \in d} \log(1 + \text{ReLU}(\text{MLM}(t))) \in \mathbb{R}^V
$$

This creates a sparse vector (most dimensions are 0) that can be searched with inverted indexes but captures semantic expansion (the model assigns weight to related terms not in the original text).

### 6.3 Hybrid Search (Dense + Sparse)

Combine dense and sparse retrieval for the best of both worlds:

$$
\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{dense}}(q, d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(q, d)
$$

where $\alpha \in [0, 1]$ controls the balance. Typical: $\alpha = 0.5$-$0.7$.

#### Reciprocal Rank Fusion (RRF)

A popular alternative to weighted combination. Instead of combining scores, combine **rankings**:

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
$$

where $k$ is a constant (typically 60) and $\text{rank}_r(d)$ is the rank of document $d$ in ranker $r$'s results.

**Why RRF works**: Score scales differ between dense and sparse retrievers — directly combining them requires careful normalization. RRF sidesteps this by only using ranks, which are naturally on the same scale.

### 6.4 When to Use Which

| Scenario | Best Method |
|----------|------------|
| Semantic questions ("What causes inflation?") | Dense |
| Exact keywords (product IDs, error codes, names) | Sparse (BM25) |
| General-purpose RAG system | **Hybrid** (dense + BM25 + RRF) |
| Multilingual retrieval | Dense (multilingual embeddings) |
| Very large corpus (100M+ docs) | Sparse (BM25 for initial filtering) + Dense re-ranking |

**The industry consensus**: Hybrid search with RRF is the default for production RAG. It captures both semantic similarity and keyword matching with minimal engineering overhead.

---

## 7. Re-Ranking — The Second Stage

### 7.1 Why Re-Rank?

Bi-encoder retrieval is fast but imprecise — it encodes query and document independently, missing fine-grained interactions. Re-ranking applies a more powerful (but slower) model to the top-K results:

```
Retrieval (bi-encoder):  Millions of docs → Top 100     (fast, approximate)
Re-ranking (cross-encoder): Top 100 → Top 10            (slow, precise)
Prompt augmentation:     Top 10 → LLM context            (final selection)
```

### 7.2 Cross-Encoder Re-Ranking

A cross-encoder (typically BERT/DeBERTa-based) encodes the query-document pair **together**:

$$
\text{score}(q, d) = \sigma(\mathbf{W} \cdot \text{Encoder}([\text{CLS}] \; q \; [\text{SEP}] \; d)_{\text{[CLS]}} + b)
$$

The model sees the full interaction between query and document tokens through self-attention — capturing nuances that bi-encoders miss.

**Example**:
- Query: "When was Python released?"
- Doc A: "Python 3.0 was released on December 3, 2008." → score: 0.95
- Doc B: "Python is a popular programming language." → score: 0.3

A bi-encoder might rank both similarly (both mention "Python"). The cross-encoder understands that only Doc A answers the "when" question.

### 7.3 ColBERT — Late Interaction

ColBERT (Khattab & Zaharia, 2020) is a compromise between bi-encoder and cross-encoder:

**Idea**: Encode query and document independently (like bi-encoder), but compute similarity at the **token level** instead of the **sequence level**:

$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i^T \mathbf{d}_j
$$

For each query token, find the maximum similarity with any document token, then sum.

```
Query tokens:    [When] [was] [Python] [released]
                   ↕      ↕      ↕        ↕
Doc tokens:      [Python] [3.0] [was] [released] [on] [Dec] [3] [2008]

Each query token finds its best-matching doc token:
  "When"    → max match with "2008"?  (temporal)
  "was"     → max match with "was"    (exact)
  "Python"  → max match with "Python" (exact)
  "released"→ max match with "released"(exact)
Sum of max similarities → final score
```

**Advantages**:
- Document token embeddings can be precomputed and indexed
- Much faster than cross-encoder at query time
- Better quality than simple bi-encoder (token-level matching captures more nuance)

### 7.4 Re-Ranking with LLMs

Modern approach: use the LLM itself to re-rank:

```
Rank these documents by relevance to the query: "{query}"

[1] {doc_1}
[2] {doc_2}
[3] {doc_3}
...

Ranking:
```

**Advantages**: Captures complex reasoning about relevance. **Disadvantages**: Expensive (LLM inference per re-ranking step), high latency.

### 7.5 The Full Retrieval Stack

```
Stage 1: BM25 (sparse)           → 1000 candidates    (< 10ms)
Stage 2: Dense retrieval (ANN)   → 100 candidates     (< 50ms)
Stage 3: Cross-encoder re-rank   → Top 10             (< 200ms)
Stage 4: Augment prompt          → Top 3-5 chunks
Stage 5: LLM generation          → Answer             (500ms-2s)
```

Total latency: ~1-3 seconds for a high-quality RAG response.

---

## 8. Advanced RAG Techniques

### 8.1 HyDE — Hypothetical Document Embeddings

**Problem**: The query is often short and vague ("Q3 revenue?"), while documents are detailed and specific. The semantic gap between queries and documents hurts retrieval.

**Solution** (Gao et al., 2023): Generate a **hypothetical answer** to the query, then use the hypothetical answer (not the query) for retrieval.

$$
q \xrightarrow{\text{LLM}} \hat{d} \xrightarrow{\text{embed}} \mathbf{e}_{\hat{d}} \xrightarrow{\text{search}} \text{retrieve}(\mathbf{e}_{\hat{d}}, \mathcal{D})
$$

```
Query:      "What was Q3 revenue?"
                │
                ▼ (LLM generates hypothetical answer)
Hypothetical: "Q3 revenue was approximately $4 million, showing growth
               over the previous quarter driven by enterprise sales."
                │
                ▼ (embed the hypothetical, not the query)
Retrieve documents similar to the hypothetical answer
```

**Why it works**: The hypothetical document is in "document space" — its embedding is closer to actual relevant documents than the short query's embedding.

**When it helps**: Vague queries, complex questions, queries with different vocabulary than the corpus.

**When it hurts**: The LLM may hallucinate wrong details in the hypothetical, retrieving incorrect documents. Adds latency (LLM call before retrieval).

### 8.2 Multi-Query RAG

**Problem**: A single query may not capture all aspects of the user's information need.

**Solution**: Generate multiple paraphrases/variations of the query, retrieve for each, then merge results.

```
Original:   "How does RAG compare to fine-tuning?"
                │
                ▼ (LLM generates variations)
Variation 1: "Differences between retrieval augmented generation and model fine-tuning"
Variation 2: "When should I use RAG vs fine-tune an LLM?"
Variation 3: "RAG advantages and disadvantages compared to fine-tuning"
                │
                ▼ (retrieve for each, then union + deduplicate)
Final results: top-K from merged, deduplicated results
```

### 8.3 Query Decomposition

**Problem**: Complex queries require information from multiple parts of the knowledge base.

**Solution**: Break the query into sub-queries, retrieve for each, combine results.

```
Query: "Compare Q3 and Q4 revenue and explain the trend"
            │
            ▼ (decompose)
Sub-query 1: "Q3 revenue figures"
Sub-query 2: "Q4 revenue figures"
Sub-query 3: "Revenue trend analysis 2025"
            │
            ▼ (retrieve for each, combine)
Feed all retrieved chunks to LLM for synthesis
```

### 8.4 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

**Problem**: Standard RAG retrieves at one granularity (chunk-level). It can't answer questions that require synthesizing information across multiple chunks.

**Solution** (Sarthi et al., 2024): Build a **tree** of summaries:

```
Level 2:    [Summary of entire document]
                /                    \
Level 1:  [Summary of §1-3]     [Summary of §4-6]
              /    |    \            /    |    \
Level 0:  [§1]  [§2]  [§3]      [§4]  [§5]  [§6]
```

1. Cluster leaf chunks by similarity
2. Summarize each cluster (using an LLM)
3. Recursively cluster and summarize until reaching a single root summary
4. At query time, search across **all levels** of the tree

**Why it works**: Low-level chunks answer specific questions. High-level summaries answer questions that span multiple sections. The tree provides multi-granularity retrieval.

### 8.5 Corrective RAG (CRAG)

**Problem**: Retrieved documents may not actually be relevant to the query. Including irrelevant documents in the prompt can mislead the LLM.

**Solution** (Yan et al., 2024): Add a **retrieval evaluator** that scores each retrieved document's relevance:

```
Query → Retrieve Top-K → Evaluate Relevance of Each Chunk
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               Correct    Ambiguous   Incorrect
                 │            │          │
            Use as-is     Refine    Discard/
                          query     Web search
                              │          │
                              ▼          ▼
                        Augment prompt with
                        refined/alternative results
```

If the evaluator determines the retrieved documents are **incorrect** or **ambiguous**, CRAG can:
- Strip out knowledge, triggering the LLM to rely on its own knowledge
- Reformulate the query and re-retrieve
- Fall back to web search

### 8.6 Self-RAG (Self-Reflective RAG)

**Idea** (Asai et al., 2023): Train the LLM to generate **special tokens** that control the retrieval process:

- `[Retrieve]`: "I need to look something up"
- `[No Retrieve]`: "I can answer from my own knowledge"
- `[Relevant]`: "This retrieved passage is relevant"
- `[Irrelevant]`: "This passage doesn't help"
- `[Supported]`: "My answer is supported by the evidence"

The model decides **when** to retrieve, **whether** the retrieval is useful, and **whether** its answer is grounded — all within the generation process.

### 8.7 Agentic RAG

Combine RAG with AI agent capabilities (see [Topic 21](21_Agents_Tool_Use.md)):

```
User: "How did our churn rate change across Q1-Q4?"
        │
Agent decides: Need data from multiple quarters
        │
        ├── Tool: search("Q1 churn rate") → retrieve
        ├── Tool: search("Q2 churn rate") → retrieve
        ├── Tool: search("Q3 churn rate") → retrieve
        ├── Tool: search("Q4 churn rate") → retrieve
        │
Agent: Synthesizes all results, identifies trend
        │
Answer: "Churn decreased from 8.2% in Q1 to 5.1% in Q4,
         a 37% improvement driven by..."
```

The agent can:
- Issue multiple queries dynamically
- Decide when it has enough information
- Use tools (calculators, APIs) alongside retrieval
- Backtrack if initial retrieval was insufficient

---

## 9. RAG Evaluation — How to Measure Quality

### 9.1 The Evaluation Challenge

RAG quality depends on **two independent components**: retrieval quality and generation quality. A failure in either causes a bad answer.

```
                        Retrieval
                    Good        Bad
              ┌──────────┬──────────┐
Generation    │  ✓ Great │  ✗ Hall- │
   Good       │  answer  │  ucinated│
              ├──────────┼──────────┤
Generation    │  ✗ Poor  │  ✗✗ Bad  │
   Bad        │  summary │  all over│
              └──────────┴──────────┘
```

### 9.2 Component Metrics

#### Retrieval Metrics

**Context Relevance**: Are the retrieved chunks relevant to the query?

$$
\text{Context Relevance} = \frac{\text{Number of relevant chunks retrieved}}{\text{Total chunks retrieved}}
$$

**Context Recall**: Did we retrieve all the necessary information?

$$
\text{Context Recall} = \frac{\text{Relevant information retrieved}}{\text{Total relevant information in corpus}}
$$

#### Generation Metrics

**Faithfulness (Groundedness)**: Is the answer supported by the retrieved context?

$$
\text{Faithfulness} = \frac{\text{Claims in answer supported by context}}{\text{Total claims in answer}}
$$

A faithfulness of 0.8 means 80% of the LLM's claims can be traced to the retrieved documents. The other 20% may be hallucinated.

**Answer Relevance**: Does the answer actually address the question?

$$
\text{Answer Relevance} = \text{sim}(\text{query}, \text{answer})
$$

### 9.3 RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment — Es et al., 2023) is the most widely used RAG evaluation framework. It automates evaluation using an LLM as judge:

| Metric | What It Measures | How It's Computed |
|--------|-----------------|------------------|
| **Faithfulness** | Is the answer grounded in context? | LLM extracts claims from answer, checks each against context |
| **Answer Relevancy** | Does the answer address the question? | Generate questions from the answer, compare to original query |
| **Context Precision** | Are relevant chunks ranked higher? | LLM judges relevance of each chunk; check ranking |
| **Context Recall** | Is all needed info retrieved? | Compare reference answer sentences to context |

**RAGAS Score** (overall):

$$
\text{RAGAS} = \text{Harmonic Mean}(\text{Faithfulness}, \text{Answer Relevancy}, \text{Context Precision}, \text{Context Recall})
$$

### 9.4 Beyond Automated Metrics

| Method | What It Catches |
|--------|----------------|
| **Human evaluation** | Nuanced quality issues, factual errors, tone |
| **A/B testing** | Real-world user preference between RAG configurations |
| **Failure analysis** | Systematic categorization of error types |
| **LLM-as-judge** | Scalable quality assessment (but has biases) |

### 9.5 Common Failure Modes

| Failure Mode | Cause | Fix |
|-------------|-------|-----|
| **Wrong chunks retrieved** | Poor embedding model or chunking | Better embeddings, hybrid search, re-ranking |
| **Right chunks, wrong answer** | LLM ignores or misinterprets context | Better prompts, instruction to cite sources |
| **Not enough context** | Relevant info split across chunks | Parent-child chunking, multi-query |
| **Contradictory chunks** | Different versions of same information | Deduplication, recency filtering |
| **Hallucination despite context** | LLM generates beyond what context supports | Faithfulness checking, constrained generation |
| **Latency too high** | Too many retrieval stages or large context | Reduce stages, compress context, cache |

---

## 10. RAG vs Fine-Tuning vs Long Context

### 10.1 The Three Approaches to Giving LLMs Knowledge

| Approach | How It Works | When Knowledge Changes |
|----------|-------------|----------------------|
| **RAG** | Retrieve relevant docs at query time | Update the document store (instant) |
| **Fine-tuning** | Train the model on your data | Retrain the model (expensive) |
| **Long context** | Stuff all relevant data into the prompt | Re-compute every query (expensive) |

### 10.2 Detailed Comparison

| Aspect | RAG | Fine-Tuning | Long Context |
|--------|-----|------------|--------------|
| **Knowledge update** | Instant (add/remove docs) | Requires retraining | Instant (change prompt) |
| **Factual accuracy** | High (citable sources) | Medium (may hallucinate) | High (in context) |
| **Cost at query time** | Low (retrieval + generation) | Low (just generation) | High (long prompts are expensive) |
| **Cost to set up** | Medium (indexing pipeline) | High (training) | Low (just a prompt) |
| **Scalability (data size)** | Millions of documents | Limited by training | Limited by context window |
| **Behavior changes** | No (just adds knowledge) | Yes (changes style, format) | Partial (via instructions) |
| **Latency** | Medium (retrieval + generation) | Low | High (long context processing) |
| **Privacy** | Data stays in your infra | Data baked into weights | Data sent per query |

### 10.3 Decision Framework

```
                    ┌──────────────────────────┐
                    │  What do you need?        │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
        Factual knowledge  Behavioral change    Small dataset
        from documents     (style, format)      (< 100 docs)
              │                  │                   │
              ▼                  ▼                   ▼
            RAG              Fine-tuning         Long context
                                                 (stuff into prompt)
```

**Use RAG when**:
- Knowledge changes frequently (news, docs, databases)
- You need citations and verifiability
- The knowledge base is large (1K+ documents)
- You can't afford to retrain the model

**Use fine-tuning when**:
- You need to change the model's behavior, style, or format
- The knowledge is stable and unlikely to change
- You have high-quality training examples (not just documents)
- You need lower latency (no retrieval step)

**Use long context when**:
- The total data fits in the context window (< 128K tokens)
- You need the model to reason over the entire dataset simultaneously
- Setup speed is critical (no indexing needed)
- The data changes frequently and is small

### 10.4 The Hybrid Approach

In practice, the best systems combine multiple approaches:

1. **Fine-tune** the model for your domain's style and format
2. **RAG** for factual knowledge that changes
3. **Long context** for the specific conversation history

Example: A customer support bot:
- Fine-tuned on support conversations (learns the right tone and format)
- RAG for product documentation (updated as products change)
- Long context for the current conversation thread

---

## 11. Production RAG — System Design Considerations

### 11.1 Architecture for Scale

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                   API Gateway                            │
│  (rate limiting, auth, request routing)                  │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌─────┴─────┐   ┌─────┴─────┐
    │ Query   │    │ Retrieval  │   │ Generation │
    │Processing│   │ Service    │   │ Service    │
    │(rewrite,│    │(embedding, │   │(LLM call,  │
    │ expand) │    │ search,    │   │ prompt      │
    │         │    │ re-rank)   │   │ assembly)   │
    └─────────┘    └─────┬─────┘   └────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────┴───┐ ┌───┴───┐ ┌───┴────┐
         │Vector  │ │BM25   │ │Metadata│
         │  DB    │ │Index  │ │ Store  │
         └────────┘ └───────┘ └────────┘
```

### 11.2 Key Production Considerations

| Consideration | Strategy |
|--------------|----------|
| **Latency** | Cache frequent queries; pre-compute embeddings; use faster models for re-ranking |
| **Cost** | Cache LLM responses; use smaller models where possible; batch embedding calls |
| **Freshness** | Incremental indexing; webhook-triggered re-indexing; timestamp-based filtering |
| **Access control** | Per-user document permissions; filter results by user's access level |
| **Monitoring** | Track retrieval quality, LLM faithfulness, user feedback, latency percentiles |
| **Feedback loop** | Log queries + retrieved docs + answers + user ratings; use for evaluation and improvement |

### 11.3 Caching Strategies

**Semantic caching**: Cache answers for semantically similar queries:

$$
\text{If } \cos(\mathbf{e}_{q_{\text{new}}}, \mathbf{e}_{q_{\text{cached}}}) > \theta: \text{ return cached answer}
$$

This avoids redundant LLM calls for paraphrased questions.

**Embedding caching**: Cache computed embeddings for documents and frequently asked queries.

**Result caching**: Cache the top-K retrieval results for popular queries (with TTL for freshness).

---

## 12. Interview Questions & Answers

### Q1: Design a RAG system for enterprise document Q&A. Walk through every component.

**A**: I'll design for a company with ~100K documents (PDFs, Confluence pages, Slack threads), 1000 daily users, <3 second latency requirement.

**Indexing pipeline**:
1. **Document ingestion**: Connectors for PDF, Confluence API, Slack API. Extract text, preserve metadata (author, date, source, permissions).
2. **Chunking**: Recursive character splitting at 512 tokens with 50-token overlap. For structured docs (Confluence), split by headers. Prepend contextual descriptions (title, section header) to each chunk.
3. **Embedding**: BGE-large-en (1024 dims). Batch process, ~1 hour for 100K docs on single GPU.
4. **Indexing**: Qdrant (self-hosted) with HNSW index. Store metadata alongside vectors for filtering.
5. **BM25 index**: Elasticsearch for sparse retrieval alongside dense.

**Query pipeline**:
1. **Query processing** (~50ms): Classify query intent. For complex queries, decompose into sub-queries.
2. **Hybrid retrieval** (~100ms): Dense search (top-100 from Qdrant) + BM25 (top-100 from Elasticsearch). Merge with RRF. Apply access control filters.
3. **Re-ranking** (~200ms): Cross-encoder (DeBERTa-based) re-ranks top-100 to top-10.
4. **Prompt assembly**: Top 5 chunks inserted into prompt with source metadata. System prompt instructs the LLM to cite sources and say "I don't know" when appropriate.
5. **Generation** (~1-2s): GPT-4o or Llama 3.1 70B. Stream response for perceived latency reduction.
6. **Post-processing**: Extract citations, format response, log for monitoring.

**Monitoring**: Track context relevance (automated RAGAS), user thumbs up/down, retrieval latency P95, hallucination rate (sampled human review).

---

### Q2: What is the bi-encoder vs cross-encoder trade-off in retrieval?

**A**:

**Bi-encoder**: Encodes query and document **independently** into fixed-size vectors. Similarity computed via cosine distance.

$$
\text{score} = \cos(f(q), f(d))
$$

- **Speed**: O(1) per comparison after document pre-encoding. Can search millions of docs in milliseconds using ANN.
- **Quality**: Limited — query and document don't "see" each other during encoding. The model must compress all meaning into a single vector, losing fine-grained interactions.
- **Use**: First-stage retrieval from large corpus.

**Cross-encoder**: Encodes query and document **together** through full transformer attention.

$$
\text{score} = g([\text{CLS}] \; q \; [\text{SEP}] \; d)
$$

- **Speed**: O(n²) per pair (full attention over concatenated input). Must run the model once per (query, document) pair.
- **Quality**: Superior — every query token can attend to every document token, capturing nuanced relevance.
- **Use**: Re-ranking top-K candidates from bi-encoder.

**The trade-off**: Bi-encoder sacrifices quality for speed; cross-encoder sacrifices speed for quality. In production, use both in a **cascade**: bi-encoder retrieves top-100 (milliseconds), cross-encoder re-ranks to top-10 (hundreds of milliseconds).

**ColBERT** is the middle ground: token-level bi-encoder. Documents are pre-encoded at the token level. At query time, compute token-to-token similarities without a joint forward pass. Faster than cross-encoder, better than simple bi-encoder.

---

### Q3: How does HyDE improve retrieval? When does it help vs hurt?

**A**: **HyDE** (Hypothetical Document Embeddings) generates a hypothetical answer to the query using the LLM, then uses that hypothetical for retrieval instead of the original query.

**How it helps**:

The fundamental problem: queries and documents live in different "linguistic spaces." A query is short ("Q3 revenue?") while a relevant document is long and detailed ("Q3 2025 revenue reached $4.2M..."). Their embeddings may not be close despite being semantically related.

HyDE bridges this gap. The LLM generates a hypothetical document that:
1. Uses document-like language (longer, more detailed)
2. Contains key terms likely to appear in the actual document
3. Has an embedding closer to relevant documents than the short query

**When it helps**:
- Vague or underspecified queries ("Tell me about the project")
- Complex questions requiring multi-hop reasoning
- Queries using different vocabulary than the corpus
- Domain-specific questions where the user doesn't know the exact terminology

**When it hurts**:
- **Hallucinated details mislead retrieval**: If the LLM generates wrong details in the hypothetical (e.g., wrong revenue figure), it may retrieve documents matching those wrong details instead of the correct ones
- **Simple factoid queries**: "What is the CEO's name?" — the query is already specific enough; HyDE adds latency without benefit
- **Latency-sensitive applications**: HyDE adds an LLM generation step before retrieval (300ms-2s)
- **Queries about rare or unusual topics**: The LLM may generate a generic hypothetical that doesn't help retrieve the specific niche information needed

---

### Q4: When would you choose RAG vs fine-tuning vs stuffing everything in context?

**A**: The decision depends on data size, update frequency, and the nature of the task:

**Choose RAG when**:
- Knowledge base is large (>100 documents, doesn't fit in context)
- Data changes frequently (weekly/daily updates)
- You need citations and traceability
- Multiple users with different access permissions
- Budget doesn't allow frequent retraining

**Choose fine-tuning when**:
- You need to change the model's *behavior* (style, format, domain adaptation)
- Knowledge is stable (medical guidelines, legal codes)
- You need minimum latency (no retrieval step)
- You have high-quality input-output pairs (not just documents)
- Example: "Always respond in our company's tone" → fine-tune

**Choose long context when**:
- Total data is small (<128K tokens, maybe <50 pages)
- You need the model to reason over ALL the data simultaneously
- Data changes frequently but is small
- Setup speed is critical (no indexing or training)
- Example: "Analyze this 20-page contract" → stuff in context

**The practical reality**: Most production systems use a hybrid:
- Fine-tune for behavior + RAG for knowledge
- RAG for large corpus + long context for current session/conversation
- The lines are blurring: models with 1M+ context windows reduce the need for RAG on smaller corpora

---

### Q5: How do you evaluate a RAG system? What metrics matter?

**A**: RAG evaluation must cover both retrieval and generation independently:

**Retrieval metrics** (is the right information found?):

1. **Context Precision**: Fraction of retrieved chunks that are actually relevant. Low precision = noise in the context that may confuse the LLM.

2. **Context Recall**: Fraction of relevant information in the corpus that was actually retrieved. Low recall = the LLM doesn't have the information needed to answer correctly.

3. **MRR (Mean Reciprocal Rank)**: Average of $1/\text{rank}$ of the first relevant result. Measures whether relevant chunks appear near the top.

**Generation metrics** (is the answer correct and grounded?):

4. **Faithfulness**: Fraction of claims in the answer that are supported by the retrieved context. Critical — low faithfulness means the LLM is hallucinating despite having context.

5. **Answer Relevance**: Does the answer actually address the question? Measured by semantic similarity between the query and the answer.

6. **Answer Correctness**: Does the answer match the ground truth (if available)?

**End-to-end metrics**:

7. **User satisfaction**: Thumbs up/down, explicit ratings. The ultimate metric.

8. **Task completion rate**: For goal-oriented systems — did the user achieve their objective?

**The RAGAS framework** automates metrics 1-5 using an LLM as judge. For production, I'd combine automated RAGAS evaluation (daily batch), sampled human evaluation (weekly), and continuous user feedback (always on).

**The most important metric**: **Faithfulness**. A RAG system that retrieves well but generates unfaithful answers is worse than one that says "I don't know" — unfaithful answers erode user trust.

---

### Q6: Explain HNSW. Why is it the default ANN algorithm?

**A**: **HNSW** (Hierarchical Navigable Small World) is a graph-based ANN algorithm that builds a multi-layer proximity graph for fast nearest-neighbor search.

**Construction**:
1. Create multiple layers (0 to $L_{\max}$). Each vector is inserted into layers 0 through a randomly assigned layer (geometric distribution — most vectors only in layer 0, few in the top layer).
2. In each layer, connect each vector to its $M$ nearest neighbors (typically $M = 16$-$64$).
3. Higher layers have fewer nodes and longer-range connections (express lanes).

**Search** (for query $\mathbf{q}$):
1. Enter at the top layer's entry point
2. Greedily navigate to the nearest neighbor of $\mathbf{q}$ in this layer
3. Use that point as the entry for the next layer down
4. Repeat until reaching layer 0
5. In layer 0, do a more thorough local search (expanding to $ef$ candidates)
6. Return the $K$ nearest neighbors found

**Why it's the default**:

1. **Speed**: $O(\log N)$ search time. For 10M vectors: ~100 distance computations vs 10M for brute force. Sub-millisecond queries.

2. **Quality**: Recall > 95% is typical with proper tuning. You can trade off speed for quality by adjusting $ef$ (search beam width).

3. **Dynamic**: Can add and remove vectors without rebuilding the entire index (unlike IVF which requires re-clustering).

4. **No training required**: Unlike IVF (needs k-means clustering), HNSW builds the graph incrementally.

5. **Memory-resident**: The graph is in memory, enabling consistent low-latency queries.

**Trade-off**: HNSW uses more memory than IVF-PQ (stores the full graph + vectors). For very large corpora (1B+ vectors), IVF-PQ or DiskANN may be more practical.

---

### Q7: What is the chunking problem? How does parent-child chunking solve it?

**A**: The **chunking dilemma**: Small chunks give precise retrieval (a 100-token chunk about "Q3 revenue" is more likely to match a query about Q3 revenue than a 2000-token chunk covering the entire financial report). But small chunks lack context — the LLM may not have enough information to generate a good answer from a 100-token snippet.

Larger chunks have the opposite problem: more context per chunk, but lower retrieval precision (the chunk matches many queries loosely instead of few queries precisely).

**Parent-child chunking** solves this by decoupling retrieval granularity from context granularity:

1. **Create small child chunks** (128-256 tokens) and index them for retrieval
2. **Map each child to its parent chunk** (512-1024 tokens, containing the child plus surrounding context)
3. **Retrieve using child chunks** (precise matching)
4. **Return parent chunks to the LLM** (rich context)

```
Parent (1024 tokens): "Section 3: Revenue Analysis. Q3 revenue reached $4.2M,
  a 15% increase YoY. Enterprise segment contributed $3.1M. SMB contributed
  $1.1M. Growth was driven by three key factors: (1) expansion of enterprise
  contracts by 23%, (2) new product launches in July, (3) improved retention..."

Child 1 (256 tokens): "Q3 revenue reached $4.2M, a 15% increase YoY.
  Enterprise segment contributed $3.1M."  ← Matches "Q3 revenue" query

Child 2 (256 tokens): "Growth was driven by three key factors:
  (1) expansion of enterprise contracts by 23%..."  ← Matches "growth drivers"
```

If the query is "What was Q3 revenue?", Child 1 is retrieved (precise match). But the LLM receives the full Parent chunk, which includes the breakdown and growth factors — enough context for a complete answer.

---

### Q8: Design a hybrid search system using dense + sparse retrieval with RRF.

**A**:

**Components**:
1. **Dense retriever**: BGE-large embedding model + Qdrant (HNSW index)
2. **Sparse retriever**: BM25 via Elasticsearch
3. **Fusion**: Reciprocal Rank Fusion

**Indexing**:
```
Document → Chunk (512 tokens, 50 overlap)
   │
   ├──► Embed (BGE-large) ──► Store in Qdrant
   │
   └──► Tokenize ──► Store in Elasticsearch (BM25 index)
```

**Query**:
```
Query "error code E4032 authentication failure"
   │
   ├──► Embed query ──► Qdrant search (top-100) ──► Ranked list A
   │
   └──► BM25 query ──► Elasticsearch search (top-100) ──► Ranked list B
   │
   ▼
   RRF Fusion:
   For each document d in (A ∪ B):
     score(d) = 1/(60 + rank_A(d)) + 1/(60 + rank_B(d))
     (If d not in a list, use rank = 1000)
   │
   ▼
   Sort by RRF score, return top-K
```

**Why this combination works for this query**:
- "error code E4032" — BM25 excels (exact keyword match)
- "authentication failure" — Dense retrieval excels (semantic: "auth error", "login failed", "access denied" are all similar)
- RRF combines both signals without needing score normalization

**RRF math for a document ranked #3 in dense and #1 in sparse**:

$$
\text{RRF} = \frac{1}{60 + 3} + \frac{1}{60 + 1} = \frac{1}{63} + \frac{1}{61} \approx 0.0159 + 0.0164 = 0.0323
$$

The constant $k = 60$ dampens the impact of rank differences — the difference between rank 1 and rank 5 is small, but the difference between rank 1 and rank 100 is significant.

---

### Q9: What are the failure modes of RAG? How do you debug each one?

**A**: Five major failure modes, in order of frequency:

**1. Retrieval failure — wrong chunks retrieved**
- **Symptom**: Answer is about the wrong topic or contains information from irrelevant documents
- **Diagnosis**: Log retrieved chunks alongside answers. Manually inspect for relevance.
- **Fixes**: Better embedding model, hybrid search, re-ranking, improved chunking, contextual retrieval (add descriptions to chunks)

**2. Retrieval gap — relevant information not retrieved**
- **Symptom**: Answer says "I don't have enough information" when the corpus contains the answer
- **Diagnosis**: Search for the answer manually in the corpus. Check if it exists in indexed chunks.
- **Fixes**: Multi-query retrieval, query expansion, HyDE, reduce chunk size (smaller chunks are more specific), increase top-K

**3. Unfaithful generation — LLM ignores or contradicts context**
- **Symptom**: Answer contains claims not in the retrieved chunks, or contradicts them
- **Diagnosis**: Compare answer claims against retrieved chunks (automated with RAGAS faithfulness metric)
- **Fixes**: Stronger system prompt ("Only answer based on the provided context"), better LLM, citation enforcement, faithfulness re-ranking of multiple generated answers

**4. Context window overflow — too much or too little context**
- **Symptom**: Answer misses information from later chunks (LLM "lost in the middle"), or answer is based on insufficient context
- **Diagnosis**: Vary the number of chunks and measure answer quality
- **Fixes**: Re-rank to put most relevant chunks first, compress/summarize context, use parent-child chunking, increase context window

**5. Stale or contradictory information**
- **Symptom**: Answer cites outdated information or combines contradictory sources
- **Diagnosis**: Check timestamps and versions of retrieved chunks
- **Fixes**: Timestamp-based filtering (prefer recent), deduplication, version tracking, metadata filtering

**Debugging workflow**:
```
Bad answer
   ├── Check retrieved chunks → Are they relevant? → No → Retrieval problem
   │                                                → Yes ↓
   ├── Check if answer matches chunks → Does the LLM follow the context? → No → Generation problem
   │                                                                      → Yes ↓
   └── Check if correct info is in corpus → Is it retrievable? → No → Indexing/chunking problem
```

---

### Q10: Long context windows (1M+ tokens) are getting cheaper. Will RAG become obsolete?

**A**: No, but the boundary between RAG and long context will shift. Here's the nuanced analysis:

**What long context replaces**:
- RAG for small corpora (<1000 pages): Just stuff everything in context
- RAG for session-level context: Conversation history, recent documents
- Simple Q&A over a few documents: No indexing overhead

**What RAG still provides that long context doesn't**:

1. **Cost at scale**: Processing 1M tokens per query costs ~$15 (at current GPT-4 pricing). A RAG system retrieving 5 chunks costs ~$0.01. For 10K queries/day: $150K vs $100/day. RAG is 1000× cheaper.

2. **Latency**: Processing 1M tokens takes 10-30 seconds. RAG retrieval takes <100ms. For real-time applications, RAG wins.

3. **Corpus size**: Even 1M tokens is ~750 pages. Enterprise knowledge bases have millions of pages. Long context can't hold the entire corpus.

4. **"Lost in the middle"**: LLMs struggle to attend to information in the middle of very long contexts (Liu et al., 2023). Relevant information at position 500K may be ignored. RAG puts the most relevant information at the top of the context.

5. **Dynamic updates**: Adding a new document to RAG takes seconds (embed + index). Adding it to a long-context approach means re-sending everything every query.

6. **Citations**: RAG naturally provides source attribution. Long context makes it harder to trace which specific document the answer came from.

**The likely future**: Hybrid approaches where long context handles session state and recently accessed documents, while RAG handles the full knowledge base. The "RAG or long context?" question is already being answered as "both."

---

*RAG is the bridge between LLMs and real-world knowledge. Next: [Topic 21: Agents & Tool Use](21_Agents_Tool_Use.md) — giving LLMs the ability to act, not just answer.*
