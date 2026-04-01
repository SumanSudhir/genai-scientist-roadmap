# Generative AI Scientist -- Complete Interview Roadmap

> **Target Audience**: Students, Professionals, and aspiring AI / Research Scientists
> **Approach**: Depth-first, theory-heavy, minimal code -- learn the underlying mechanics like a researcher to build a rock-solid foundation.

## Suggested Study Order & File Map

Study topics sequentially using this map as a comprehensive guide to mastering Generative AI concepts.

```
Phase A: Foundations (Topics 1-4)         -- The bedrock. Skip at your peril.
Phase B: NLP Core (Topics 5-7)           -- Classical NLP before the transformer era.
Phase C: Transformers (Topics 8-10)      -- The architecture that changed everything.
Phase D: Language Models (Topics 11-13)  -- From BERT to GPT-4 to Llama 3.
Phase E: Training LLMs (Topics 14-16)    -- Pretraining, fine-tuning, alignment.
Phase F: Generation &Tic (Topics 17-19)    -- Decoding, prompting, in-context learning.
Phase G: Applications (Topics 20-22)     -- RAG, agents, multimodal.
Phase H: Production (Topics 23-24)       -- Serving, optimization, system design.
Phase I: Frontier (Topics 25-27)         -- Cutting-edge research, evaluation, safety.
Phase J: Interview Prep (Topic 28)       -- Strategy, behavioral, mock patterns.
```

Each topic below lists: what it covers, why it matters for an AI Scientist interview, and key questions you should be able to answer after studying it.

---

## Phase A: Mathematical & ML Foundations

These are non-negotiable. Every AI Scientist interview starts here. Interviewers use these to gauge your depth -- surface-level answers immediately disqualify.

---

### Topic 1: Probability, Statistics & Information Theory

**What you'll learn**:
- Probability fundamentals (Bayes' theorem, conditional probability, independence)
- Common distributions (Gaussian, Bernoulli, Categorical, Poisson, Multinomial)
- Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP)
- Expectation, variance, covariance, correlation
- Information theory: entropy, cross-entropy, KL divergence, mutual information
- Hypothesis testing, p-values, confidence intervals
- A/B testing design and statistical significance

**Why it matters for AI Scientist**:
Cross-entropy is your training loss. KL divergence is in RLHF, VAEs, and distillation. MLE is how every language model is trained. Interviewers at research labs will ask you to derive things from first principles -- you need this vocabulary fluently.

**You should be able to answer**:
- Why is cross-entropy the standard loss for language models? Derive it from MLE.
- What is KL divergence? Why is it asymmetric? When does this asymmetry matter?
- Explain the relationship between entropy, cross-entropy, and KL divergence.
- How would you design an A/B test to compare two LLM variants?

---

### Topic 2: Linear Algebra & Optimization

**What you'll learn**:
- Vectors, matrices, tensor operations
- Eigenvalues, eigenvectors, SVD, PCA
- Matrix factorization and its role in embeddings
- Gradient descent (batch, mini-batch, stochastic)
- Optimizers: SGD with momentum, AdaGrad, RMSProp, Adam, AdamW
- Learning rate schedules (warmup, cosine decay, linear decay)
- Convexity, saddle points, loss landscapes
- Gradient clipping, gradient accumulation

**Why it matters for AI Scientist**:
Attention is matrix multiplication. LoRA is low-rank matrix factorization. AdamW is in every LLM training recipe. Understanding optimization deeply separates scientists from practitioners.

**You should be able to answer**:
- Why does Adam work better than SGD for transformers?
- What is the difference between Adam and AdamW? Why does it matter?
- Explain LoRA in terms of linear algebra (low-rank decomposition of weight updates).
- What is the role of warmup in learning rate scheduling?

---

### Topic 3: Classical Machine Learning

**What you'll learn**:
- Supervised learning: linear regression, logistic regression (derive the gradient), SVMs, decision trees, random forests, gradient boosting (XGBoost)
- Unsupervised learning: K-Means, DBSCAN, hierarchical clustering, PCA, t-SNE, UMAP
- Bias-variance tradeoff (deep understanding, not just the diagram)
- Regularization: L1 (Lasso), L2 (Ridge) -- geometric AND probabilistic interpretations
- Feature engineering, feature selection, feature importance
- Overfitting: causes, detection, prevention (cross-validation, early stopping, dropout, data augmentation)

**Why it matters for AI Scientist**:
These are your "warm-up" interview questions. If you stumble on bias-variance or can't explain why L1 induces sparsity, the interview is over before you reach the LLM questions.

**You should be able to answer**:
- Derive the gradient update for logistic regression.
- Explain bias-variance tradeoff with a concrete example. How does model complexity relate?
- Why does L1 produce sparse weights? (Geometric explanation with the diamond constraint.)
- When would you use XGBoost over a neural network?

---

### Topic 4: Deep Learning Foundations

**What you'll learn**:
- Perceptrons, MLPs, universal approximation theorem
- Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish, SwiGLU
- Backpropagation: chain rule, computational graphs, gradient flow
- Weight initialization: Xavier/Glorot, He/Kaiming -- why they matter
- Batch Normalization vs Layer Normalization (when and why)
- Dropout (training vs inference, inverted dropout)
- Residual connections (why they're critical for deep networks)
- CNNs (convolutions, pooling, ResNet) -- brief, for multimodal context
- Loss functions: cross-entropy, MSE, contrastive loss, triplet loss

**Why it matters for AI Scientist**:
Every component here appears in transformers. Layer Norm is in every transformer block. Residual connections are what make deep transformers trainable. GELU/SwiGLU are the activations in modern LLMs. You need to understand WHY each design choice was made.

**You should be able to answer**:
- Why do transformers use Layer Norm instead of Batch Norm?
- Explain the vanishing gradient problem. How do residual connections solve it?
- Why GELU over ReLU in transformers? What about SwiGLU in Llama?
- What happens if you remove residual connections from a 96-layer transformer?

---

## Phase B: NLP Before Transformers

Understanding the problems that existed before transformers helps you appreciate WHY transformers work. Interviewers love "trace the evolution" questions.

---

### Topic 5: Tokenization

**Covers**: Your existing `01_Tokenization.md` -- will be expanded.

**What you'll learn**:
- Word-level, character-level, subword tokenization
- BPE (Byte Pair Encoding) -- used by GPT models
- WordPiece -- used by BERT
- Unigram LM -- used by T5, ALBERT
- SentencePiece -- language-agnostic wrapper (used by Llama)
- Byte-level BPE (GPT-2/3/4) -- why bytes solve the OOV problem completely
- Vocabulary size trade-offs (small = long sequences, large = sparse embeddings)
- Special tokens ([CLS], [SEP], [PAD], <s>, </s>, <unk>)
- Tokenizer's impact on model performance and multilingual capability

**You should be able to answer**:
- Walk through BPE algorithm step by step. How does it build the vocabulary?
- Why did GPT-2 switch to byte-level BPE? What problem does it solve?
- Why does Llama use SentencePiece? How does it handle multilingual text?
- A model tokenizes "unhappiness" as ["un", "happiness"] vs ["unhapp", "iness"]. Which is better and why?

---

### Topic 6: Text Preprocessing & Word Embeddings

**Covers**: Your existing `02_Text_Preprocessing_Embeddings.md` -- will be expanded.

**What you'll learn**:
- Classical preprocessing: lowercasing, stemming, lemmatization, stop word removal
- Bag of Words, TF-IDF (derivation and intuition)
- Word2Vec: Skip-gram and CBOW (objectives, negative sampling, why it works)
- GloVe: global co-occurrence matrix factorization
- FastText: subword embeddings (handles OOV and morphology)
- Static vs contextual embeddings -- the paradigm shift
- Sentence embeddings: Sentence-BERT, Universal Sentence Encoder
- Embedding spaces: analogy arithmetic, clustering, nearest neighbors

**You should be able to answer**:
- Derive the Skip-gram objective. Why is negative sampling necessary?
- How does GloVe differ from Word2Vec? What "global" information does it capture?
- Why can't Word2Vec handle polysemy ("bank" = river bank vs financial bank)?
- Explain how Sentence-BERT creates sentence embeddings and why cosine similarity works.

---

### Topic 7: Sequence Modeling (RNNs, LSTMs, Seq2Seq)

**Covers**: Your existing `03_Sequence_Modeling.md` -- will be expanded.

**What you'll learn**:
- Vanilla RNNs: architecture, hidden state, unrolling through time
- Vanishing and exploding gradients (mathematical explanation via Jacobian)
- LSTM: forget gate, input gate, output gate, cell state -- how each solves the gradient problem
- GRU: simplified gating (reset gate, update gate)
- Bidirectional RNNs: why looking both ways helps for understanding tasks
- Seq2Seq: encoder-decoder paradigm, teacher forcing, exposure bias
- The information bottleneck problem -- why a single context vector fails
- Historical importance: these models defined NLP from 2013-2017

**You should be able to answer**:
- Draw the LSTM cell and explain each gate's role. Why is the cell state the key innovation?
- What is teacher forcing? What is exposure bias? How are they related?
- Why is the Seq2Seq bottleneck problem important? What solution was proposed? (Leads to Topic 8.)
- Why did transformers replace RNNs? Give at least 3 concrete reasons.

---

## Phase C: The Transformer Revolution

This is the core of any GenAI interview. You need to know transformers cold -- math, intuition, implementation logic, and every variant.

---

### Topic 8: Attention Mechanisms

**Covers**: Your existing `04_Attention_Mechanisms.md` -- will be expanded with modern variants.

**What you'll learn**:
- Bahdanau (additive) attention -- solving the Seq2Seq bottleneck
- Luong (multiplicative) attention -- simpler, faster alternative
- Self-attention -- attending to yourself (the key insight for transformers)
- Scaled dot-product attention: Q, K, V formulation, why we scale by sqrt(d_k)
- Multi-head attention: why multiple heads, what different heads learn
- Cross-attention: how the decoder attends to the encoder
- Causal (masked) attention: preventing future token leakage
- Attention as soft dictionary lookup (the intuitive explanation)
- Computational complexity: O(n^2) in sequence length -- why this matters

**You should be able to answer**:
- Derive scaled dot-product attention from scratch. Why divide by sqrt(d_k)?
- What would happen if you didn't scale? (Softmax saturation, vanishing gradients.)
- Why multiple heads instead of one large attention? What do different heads capture?
- Explain cross-attention. When is it used vs self-attention?
- What is the causal mask? Why is it necessary for autoregressive generation?

---

### Topic 9: The Transformer Architecture (Deep Dive)

**What you'll learn**:
- Full encoder-decoder transformer ("Attention Is All You Need")
- Input pipeline: token embedding + positional encoding
- Encoder block: self-attention -> Add&Norm -> FFN -> Add&Norm
- Decoder block: masked self-attention -> Add&Norm -> cross-attention -> Add&Norm -> FFN -> Add&Norm
- Feed-forward network: two linear layers with activation (d_model -> 4*d_model -> d_model)
- Layer normalization: Pre-LN vs Post-LN (why Pre-LN is now standard)
- Residual connections: why they're essential at depth
- Output head: linear projection + softmax over vocabulary
- Parameter counting: how to compute the number of parameters in a transformer
- The three variants: encoder-only, decoder-only, encoder-decoder

**You should be able to answer**:
- Draw the full transformer architecture from memory and label every component.
- How many parameters does a transformer with L layers, d_model hidden size, and V vocab size have? Break it down by component.
- Why Pre-LN instead of Post-LN? What training stability issue does it solve?
- Why is the FFN dimension 4x the model dimension? What role does the FFN play?
- Compare encoder-only, decoder-only, and encoder-decoder. When would you use each?

---

### Topic 10: Positional Encodings (Deep Dive)

**What you'll learn**:
- Why transformers need positional information (permutation invariance)
- Sinusoidal positional encoding (original paper): derivation and properties
- Learned absolute positional embeddings (BERT, GPT-2)
- Relative positional encoding (Shaw et al., Transformer-XL)
- Rotary Position Embedding (RoPE) -- used in Llama, Mistral
  - How RoPE encodes relative position through rotation matrices
  - Why RoPE enables length extrapolation
- ALiBi (Attention with Linear Biases) -- used in BLOOM, MPT
- NTK-aware RoPE scaling and YaRN -- extending context length
- Position interpolation vs extrapolation

**You should be able to answer**:
- Why can't transformers understand word order without positional encoding?
- What is the key insight behind RoPE? How does it encode relative position?
- Compare sinusoidal, learned, RoPE, and ALiBi. Trade-offs of each?
- How do models extend their context length beyond what they were trained on? (RoPE scaling, NTK-aware interpolation.)

---

## Phase D: Language Models

The evolution from BERT to GPT-4 to Llama 3. You need to know each model family, its training objective, architectural innovations, and why it matters.

---

### Topic 11: Encoder Models (BERT Family)

**Covers**: Expanded from your existing `05_Language_Models.md`.

**What you'll learn**:
- BERT: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
  - Why bidirectional context matters for understanding tasks
  - [CLS] token, [MASK] token, segment embeddings
  - Fine-tuning BERT for classification, NER, QA
- RoBERTa: what it changed (no NSP, dynamic masking, more data, larger batches)
- ALBERT: factorized embeddings, cross-layer parameter sharing
- DeBERTa: disentangled attention (separate content and position)
- ELECTRA: replaced token detection (more efficient pretraining)
- Why encoder-only models are still relevant: embeddings, retrieval, classification
- Comparison: when to use BERT-family vs GPT-family

**You should be able to answer**:
- How does MLM work? Why mask 15%? What is the 80/10/10 masking strategy?
- Why did RoBERTa remove NSP? What evidence showed it was harmful?
- How does DeBERTa's disentangled attention improve over BERT?
- For a text classification task with 10K labeled examples, would you use BERT or GPT-4? Why?

---

### Topic 12: Decoder Models (GPT Family & Open-Source LLMs)

**What you'll learn**:
- Autoregressive language modeling: P(x_t | x_1, ..., x_{t-1})
- GPT-1: generative pretraining + discriminative fine-tuning
- GPT-2: zero-shot task transfer via prompting
- GPT-3: few-shot in-context learning, emergent abilities, scaling
  - Architecture details: 175B parameters, 96 layers, 96 heads
- GPT-4: multimodal, improved reasoning (what we know publicly)
- Scaling laws (Chinchilla): compute-optimal training (tokens vs parameters)
- Llama 1/2/3 family:
  - Architecture innovations: RoPE, SwiGLU activation, GQA, Pre-LN
  - Training data scales: 1.4T -> 2T -> 15T tokens
  - Why Llama matters for open-source AI
- Mistral & Mixtral:
  - Sliding window attention
  - Mixture of Experts (MoE) architecture
- Phi (Microsoft): small language models that punch above their weight
- Gemma (Google), Qwen (Alibaba), DeepSeek

**You should be able to answer**:
- What are scaling laws? What did Chinchilla show about compute-optimal training?
- Compare Llama 2 and Llama 3 architectures. What changed and why?
- What is Mixture of Experts? How does Mixtral use it? What are the routing mechanisms?
- Why are small language models (Phi, Gemma) performing so well despite fewer parameters?

---

### Topic 13: Encoder-Decoder Models & Unified Frameworks

**What you'll learn**:
- T5: Text-to-Text Transfer Transformer
  - Everything is text-to-text: classification, translation, summarization
  - Span corruption pretraining objective
  - C4 dataset
- BART: denoising autoencoder approach
  - Multiple corruption strategies (masking, deletion, permutation, rotation)
- UL2: unifying different pretraining objectives
- Prefix LM: bidirectional on prefix, causal on suffix
- When encoder-decoder still wins over decoder-only (structured output tasks)
- The debate: why decoder-only became dominant
  - Simplicity, scalability, in-context learning, emergence

**You should be able to answer**:
- How does T5's span corruption differ from BERT's token masking?
- Why did the field converge on decoder-only despite encoder-decoder being more flexible?
- When would you still choose an encoder-decoder model over a decoder-only model?
- What is the prefix LM approach? How does it combine bidirectional and causal attention?

---

## Phase E: Training Large Language Models

This is where AI Scientist interviews go deep. You need to understand the full pipeline from data curation to alignment.

---

### Topic 14: Pretraining LLMs

**What you'll learn**:
- Training objectives: CLM, MLM, span corruption, prefix LM
- Data pipeline:
  - Sources: Common Crawl, Wikipedia, books, code (GitHub), scientific papers
  - Filtering: quality heuristics, deduplication (MinHash, exact dedup), toxicity filtering, PII removal
  - Data mixing ratios and their impact on model capabilities
- Tokenizer training: when and how to train a custom tokenizer
- Distributed training:
  - Data parallelism, tensor parallelism, pipeline parallelism
  - ZeRO (Zero Redundancy Optimizer) stages 1/2/3
  - FSDP (Fully Sharded Data Parallel)
- Mixed precision training: FP16, BF16, loss scaling
- Training stability: gradient clipping, weight decay, warm-up
- Compute requirements: FLOPs, GPU-hours, cost estimates
- Training infrastructure: NVIDIA A100/H100, TPUs, custom chips

**You should be able to answer**:
- How do you estimate the FLOPs needed to train a 70B parameter model on 2T tokens?
- What is the difference between data parallelism, tensor parallelism, and pipeline parallelism?
- Why is BF16 preferred over FP16 for LLM training?
- How does data quality and deduplication affect model performance?

---

### Topic 15: Fine-Tuning & Parameter-Efficient Methods

**Covers**: Expanded from your existing `06_Training_Fine_Tuning.md`.

**What you'll learn**:
- Full fine-tuning: when it's justified, catastrophic forgetting
- LoRA (Low-Rank Adaptation):
  - Mathematical formulation: W' = W + BA where B is d x r, A is r x d
  - Why low-rank works (intrinsic dimensionality hypothesis)
  - Rank selection, alpha scaling, which layers to adapt
- QLoRA: LoRA + 4-bit quantization (NF4 data type)
  - Double quantization, paged optimizers
  - Why this is revolutionary: fine-tune 65B models on a single 48GB GPU
- Prefix Tuning: prepend learnable vectors to each layer
- Prompt Tuning: learn soft prompts in the embedding space
- Adapter layers: bottleneck modules inserted between transformer layers
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- Instruction tuning:
  - Datasets: FLAN, Alpaca, Dolly, Orca, OpenHermes
  - How instruction format affects model behavior
- When to fine-tune vs when to prompt vs when to use RAG

**You should be able to answer**:
- Derive the LoRA update rule. Why does low-rank approximation work for fine-tuning?
- How does QLoRA achieve fine-tuning on a single consumer GPU? What are the three innovations?
- Compare LoRA, prefix tuning, and prompt tuning. When would you choose each?
- A client has 500 labeled examples. Should they fine-tune, use few-shot prompting, or build RAG? Walk through the decision framework.

---

### Topic 16: Alignment -- RLHF, DPO & Beyond

**What you'll learn**:
- Why alignment is necessary (helpful, harmless, honest)
- The RLHF pipeline:
  - Step 1: Supervised Fine-Tuning (SFT) on demonstrations
  - Step 2: Reward model training from human preference pairs
  - Step 3: PPO optimization with KL penalty
  - The objective: max E[R(x,y)] - beta * KL(pi_theta || pi_ref)
- Reward hacking and how to mitigate it
- DPO (Direct Preference Optimization):
  - Eliminates the reward model entirely
  - Derives the reward from the policy directly
  - Mathematical relationship to RLHF objective
  - Why DPO is simpler and often preferred
- Constitutional AI (Anthropic):
  - AI feedback instead of human feedback
  - Principles-based self-critique
- RLAIF, IPO (Identity Preference Optimization), KTO
- Reward model architectures and training
- Human preference data collection: rating vs ranking vs comparison
- Open problems: reward model overoptimization, distribution shift

**You should be able to answer**:
- Walk through the full RLHF pipeline. What is the role of the KL penalty?
- Derive the DPO objective. How does it avoid training a separate reward model?
- What is reward hacking? Give an example and explain how to mitigate it.
- Compare RLHF and DPO. When would you prefer each?
- What is Constitutional AI? How is it different from RLHF?

---

## Phase F: Generation, Prompting & Reasoning

How models generate text, and how we control that generation. Critical for applied AI scientist roles.

---

### Topic 17: Decoding Strategies & Text Generation

**What you'll learn**:
- Autoregressive generation: how transformers generate one token at a time
- Greedy decoding: always pick the highest probability token
- Beam search: keep top-K candidates, trade-off exploration vs exploitation
- Sampling-based methods:
  - Temperature scaling: controlling randomness
  - Top-k sampling: restrict to top K most likely tokens
  - Top-p (nucleus) sampling: dynamic threshold based on cumulative probability
  - Typical sampling, eta sampling
- Repetition penalty, frequency penalty, presence penalty
- Min-p sampling (recent innovation)
- Speculative decoding: draft with small model, verify with large model
- Structured generation: constraining output to JSON, regex, grammar
- KV cache: how it speeds up autoregressive generation
- Stopping criteria: EOS token, max length, stop sequences

**You should be able to answer**:
- Explain temperature scaling mathematically. What happens at T=0, T=1, T>>1?
- What is nucleus sampling? Why is it preferred over top-k in practice?
- How does speculative decoding achieve 2-3x speedup without quality loss?
- How does the KV cache work? Why does it make generation O(n) per token instead of O(n^2)?

---

### Topic 18: Prompt Engineering & In-Context Learning

**What you'll learn**:
- Zero-shot prompting: task description only
- Few-shot prompting: providing examples in context
- Chain-of-Thought (CoT): "Let's think step by step"
  - Why it works (decomposition, intermediate reasoning)
  - Zero-shot CoT vs few-shot CoT
- Self-consistency: sample multiple reasoning paths, majority vote
- Tree of Thoughts (ToT): explore multiple branches with backtracking
- ReAct: interleave reasoning and action (tool use)
- Program-of-Thought: generate code to solve problems
- Retrieval-augmented prompting (brief -- leads to Topic 20)
- Prompt sensitivity and robustness
- System prompts, role prompts, instruction hierarchy
- Prompt injection and defenses

**You should be able to answer**:
- Why does Chain-of-Thought prompting improve performance on reasoning tasks?
- What is self-consistency? How does it improve over single CoT?
- Explain the ReAct framework. How does it combine reasoning and acting?
- What is prompt injection? How would you defend against it in a production system?

---

### Topic 19: In-Context Learning (Theory)

**What you'll learn**:
- What is in-context learning (ICL)? Learning without gradient updates
- Empirical findings:
  - Emerges at scale (typically >10B parameters)
  - Sensitive to example format, order, and label distribution
  - Works even with random labels (surprising finding)
- Theoretical explanations:
  - ICL as implicit gradient descent (Akyurek et al., Oswald et al.)
  - ICL as Bayesian inference
  - Transformer as meta-learner
- Task vectors and function vectors in ICL
- Limitations of ICL vs fine-tuning
- When ICL fails: distribution shift, complex reasoning

**You should be able to answer**:
- What evidence suggests that ICL performs implicit gradient descent?
- Why does ICL work even with random labels? What does this tell us about what models are doing?
- When would fine-tuning outperform ICL, and vice versa?
- How does the number and quality of examples affect ICL performance?

---

## Phase G: Applications & Systems

The applied side -- building real systems with LLMs. AI Scientists need to bridge research and application.

---

### Topic 20: Retrieval-Augmented Generation (RAG)

**What you'll learn**:
- The RAG pipeline: query -> embed -> retrieve -> augment prompt -> generate
- Why RAG: reduce hallucinations, access current/private knowledge, cite sources
- Embedding models:
  - Sentence-BERT, OpenAI embeddings, BGE, E5, GTE
  - Training embedding models: contrastive learning, hard negatives
- Vector databases: FAISS, Pinecone, Weaviate, Qdrant, Chroma, Milvus
  - Approximate nearest neighbor (ANN) algorithms: HNSW, IVF, PQ
- Chunking strategies: fixed-size, semantic, recursive, document-structure-aware
- Retrieval methods:
  - Dense retrieval (bi-encoder)
  - Sparse retrieval (BM25, TF-IDF)
  - Hybrid search (dense + sparse with reciprocal rank fusion)
- Re-ranking: cross-encoder reranking, ColBERT
- Advanced RAG:
  - HyDE (Hypothetical Document Embeddings)
  - Multi-query RAG, query decomposition
  - RAPTOR (tree-structured retrieval)
  - Agentic RAG, self-reflective RAG (CRAG)
  - Parent-child chunking, contextual retrieval
- RAG evaluation: context relevance, answer faithfulness, RAGAS framework
- RAG vs fine-tuning vs long context: decision framework

**You should be able to answer**:
- Design a RAG system for enterprise document Q&A. Walk through every component.
- What is the bi-encoder vs cross-encoder trade-off in retrieval?
- How does HyDE improve retrieval? When does it help vs hurt?
- When would you choose RAG vs fine-tuning vs stuffing everything in context?
- How do you evaluate a RAG system? What metrics matter?

---

### Topic 21: AI Agents & Tool Use

**What you'll learn**:
- What is an AI agent? (Perception -> reasoning -> action loop)
- Tool use / function calling:
  - How LLMs learn to use tools (fine-tuning on tool-use data)
  - JSON function calling (OpenAI, Anthropic paradigm)
- Agent frameworks:
  - ReAct (reasoning + acting)
  - Plan-and-Execute
  - Reflexion (self-reflection and retry)
- Multi-agent systems: debate, collaboration, specialization
- Memory systems:
  - Short-term (conversation context)
  - Long-term (vector DB, knowledge graph)
  - Working memory (scratchpad)
- Code generation agents: code interpreter, sandbox execution
- Agent evaluation: task completion, efficiency, safety
- Frameworks: LangChain, LlamaIndex, AutoGen, CrewAI
- Limitations: compounding errors, hallucinated actions, safety risks

**You should be able to answer**:
- Design an AI agent that can answer questions about a company's codebase.
- What is the Reflexion pattern? How does self-reflection improve agent performance?
- How do you prevent an agent from taking harmful actions?
- What are the failure modes of multi-step agents? How do you mitigate them?

---

### Topic 22: Multimodal AI

**What you'll learn**:
- Vision-Language Models:
  - CLIP: contrastive learning to align image and text embeddings
  - LLaVA: visual instruction tuning (connect vision encoder to LLM)
  - GPT-4V, Gemini: native multimodal models
  - Architecture patterns: frozen vision encoder + projection layer + LLM
- Image generation:
  - Diffusion models: forward process (add noise), reverse process (denoise)
  - DDPM, DDIM, classifier-free guidance
  - Latent diffusion (Stable Diffusion): why work in latent space
  - Text-to-image: DALL-E, Stable Diffusion, Midjourney
  - ControlNet, IP-Adapter: controllable generation
- Video models: Sora (OpenAI), video understanding
- Audio/Speech:
  - Whisper: robust speech recognition
  - Text-to-Speech: VALL-E, Bark
  - AudioLM, MusicLM
- Multimodal embeddings and retrieval
- Cross-modal attention patterns

**You should be able to answer**:
- How does CLIP align images and text? What is contrastive learning?
- Explain the diffusion process. What are the forward and reverse steps?
- How does LLaVA connect a vision encoder to an LLM? What is visual instruction tuning?
- What is classifier-free guidance? Why does it improve image quality?

---

## Phase H: Production & System Design

AI Scientists at top companies must understand production constraints. System design is a standard interview round.

---

### Topic 23: LLM Inference Optimization & Deployment

**What you'll learn**:
- Quantization:
  - FP32 -> FP16 -> INT8 -> INT4 precision hierarchy
  - Post-Training Quantization: GPTQ, AWQ, SqueezeLLM
  - Quantization-Aware Training (QAT)
  - GGUF format for CPU inference
  - NF4 (NormalFloat4) -- the data type behind QLoRA
- Knowledge distillation:
  - Teacher-student framework
  - Soft labels, intermediate layer matching
  - DistilBERT as a case study
- Pruning: structured vs unstructured, magnitude-based, movement pruning
- Inference optimization:
  - KV cache management and paged attention (vLLM)
  - Continuous batching
  - Flash Attention (1, 2, 3): IO-aware attention computation
  - Tensor parallelism for inference
- Serving frameworks: vLLM, TGI (HuggingFace), TensorRT-LLM, SGLang
- API platforms: OpenAI, Anthropic, AWS Bedrock, Google Vertex AI
- Latency vs throughput vs cost trade-offs
- Batching strategies: static, dynamic, continuous

**You should be able to answer**:
- Explain how GPTQ quantization works. What is the trade-off vs AWQ?
- How does Flash Attention reduce memory from O(n^2) to O(n)? What is the key insight?
- What is paged attention (vLLM)? Why does it improve throughput?
- You need to serve a 70B model. Walk through the hardware and optimization decisions.

---

### Topic 24: ML System Design for GenAI

**What you'll learn**:
- System design framework:
  1. Requirements: functional, non-functional (latency, scale, cost)
  2. Data: sources, labeling, pipelines, feature engineering
  3. Model: architecture selection, baselines, training strategy
  4. Serving: inference pipeline, caching, load balancing
  5. Evaluation: offline metrics, online metrics, A/B testing
  6. Monitoring: drift detection, quality alerts, feedback loops
  7. Iteration: retraining triggers, continuous improvement
- Practice designs:
  - Conversational AI chatbot (multi-turn, guardrails, RAG)
  - RAG-based enterprise Q&A system
  - Code generation assistant
  - Content moderation at scale (text + image)
  - LLM-powered search and recommendation
  - Document summarization pipeline
  - Multi-agent customer service system
- Cost optimization: model selection, caching, prompt compression, routing
- Monitoring: hallucination detection, user satisfaction, token usage
- Guardrails: input/output filtering, PII detection, topic boundaries

**You should be able to answer**:
- Design a RAG-based Q&A system for 10K concurrent users. Cover end-to-end architecture.
- Design a content moderation system using LLMs. How do you handle scale and latency?
- How would you implement guardrails for a customer-facing chatbot?
- Your LLM chatbot costs $50K/month. How do you reduce costs by 5x without significant quality loss?

---

## Phase I: Research Frontiers & Evaluation

What distinguishes an AI Scientist from an engineer: understanding the frontier.

---

### Topic 25: Evaluation & Benchmarking

**What you'll learn**:
- Language model metrics:
  - Perplexity: definition, derivation, limitations
  - BLEU: precision-based n-gram overlap (translation)
  - ROUGE: recall-based n-gram overlap (summarization)
  - METEOR: synonym-aware, stemming-aware
  - BERTScore: semantic similarity using contextual embeddings
  - ChrF: character-level F-score
- LLM benchmarks:
  - MMLU: multitask language understanding (57 subjects)
  - HellaSwag: commonsense reasoning
  - TruthfulQA: measuring truthfulness
  - HumanEval: code generation correctness
  - GSM8K: grade school math
  - MATH: competition-level math
  - ARC: science reasoning
  - MT-Bench, Chatbot Arena (LMSYS): human preference
- Human evaluation methods:
  - Likert scale, pairwise comparison, Elo ranking
  - Inter-annotator agreement (Cohen's kappa, Krippendorff's alpha)
- LLM-as-a-Judge: using LLMs to evaluate other LLMs
  - Biases: position bias, verbosity bias, self-enhancement bias
- Evaluation of specific capabilities: factuality, reasoning, instruction following, safety
- Contamination: benchmark leakage into training data

**You should be able to answer**:
- Why is perplexity not sufficient to evaluate LLMs? What are its limitations?
- How does BLEU differ from ROUGE? When do you use each?
- What is the Chatbot Arena? Why is Elo ranking considered more reliable than static benchmarks?
- What are the biases in LLM-as-a-Judge evaluation? How do you mitigate them?

---

### Topic 26: Safety, Ethics & Responsible AI

**What you'll learn**:
- Hallucinations:
  - Intrinsic vs extrinsic hallucinations
  - Causes: training data issues, decoding artifacts, knowledge gaps
  - Mitigation: RAG, self-consistency, confidence calibration, citations
  - Detection methods: factual verification, self-check
- Bias and fairness:
  - Types: allocative, representational, stereotypical
  - Sources: training data, annotation, model architecture
  - Measurement: demographic parity, equalized odds, disparate impact
  - Mitigation: data balancing, debiasing techniques, red teaming
- Privacy:
  - Training data memorization (extractable vs discoverable)
  - Differential privacy in ML
  - PII detection and removal
  - Federated learning for privacy-preserving training
- Jailbreaking and adversarial attacks:
  - Types: prompt injection, role-play attacks, encoding attacks
  - Defenses: input filtering, output filtering, constitutional AI, perplexity filtering
- Watermarking: detecting AI-generated text
- AI regulation: EU AI Act, responsible scaling policies
- Copyright and training data: legal considerations

**You should be able to answer**:
- What causes hallucinations in LLMs? Propose 3 mitigation strategies.
- How would you audit an LLM for bias? What metrics would you use?
- What is the difference between prompt injection and jailbreaking?
- How does watermarking for LLM-generated text work?

---

### Topic 27: Research Frontiers & Emerging Topics (2024-2026)

**What you'll learn**:
- Reasoning and planning:
  - Chain-of-Thought, Tree of Thoughts, Graph of Thoughts
  - Process reward models vs outcome reward models
  - Test-time compute scaling (thinking tokens, o1/o3 paradigm)
  - Reasoning without explicit CoT (internalized reasoning)
- Long context:
  - Current: 128K-1M+ tokens
  - Techniques: RoPE scaling, YaRN, ring attention
  - Long-context vs RAG: when to use which
- Architecture innovations:
  - State Space Models: Mamba (S4, S6), linear-time sequence modeling
  - Mamba vs Transformer trade-offs
  - Hybrid architectures (Jamba: Mamba + Transformer)
  - RWKV: RNN-transformer hybrid
- Mixture of Experts at scale: routing, load balancing, expert specialization
- Synthetic data for training: self-play, LLM-generated training data
- Mechanistic interpretability:
  - Superposition, features, circuits
  - Sparse autoencoders for feature extraction
  - Probing, activation patching, causal tracing
- Small language models: efficiency via distillation, pruning, architecture search
- Continual learning / model merging: TIES, DARE, model soups
- Multi-token prediction (Meta's approach)

**You should be able to answer**:
- What are State Space Models? How does Mamba achieve linear-time sequence modeling?
- What is test-time compute scaling? How does it differ from scaling model size?
- What is mechanistic interpretability? What has it revealed about how transformers work?
- How does synthetic data improve LLM training? What are the risks (model collapse)?

---

## Phase J: Interview Strategy

---

### Topic 28: Interview Strategy, Behavioral & Communication

**What you'll learn**:
- Interview round types and what each tests:
  - ML fundamentals / breadth round
  - ML depth / specialization round
  - Coding round (DSA + ML coding)
  - ML system design round
  - Research discussion / paper presentation
  - Behavioral round
- How to explain technical concepts clearly:
  - The "zoom in, zoom out" technique
  - Using analogies effectively
  - Knowing when to go into math vs stay intuitive
- Behavioral preparation:
  - STAR format (Situation, Task, Action, Result)
  - Common questions: biggest impact, failure, disagreement, ambiguity
  - Company-specific values (Amazon LPs, Meta values, Google culture)
- Research presentation:
  - How to discuss your past work compellingly
  - Framing contributions, acknowledging limitations
- Common pitfalls:
  - Going too deep too fast without checking understanding
  - Not asking clarifying questions
  - Memorizing without understanding
- Questions to ask interviewers (that show depth)

---

## Topic-to-File Mapping

When you ask me to generate notes for a topic, I'll create/update the corresponding file:

| Topic | File Name | Status |
|-------|-----------|--------|
| 1. Probability, Statistics & Info Theory | `01_Probability_Statistics.md` | Done |
| 2. Linear Algebra & Optimization | `02_Linear_Algebra_Optimization.md` | Done |
| 3. Classical Machine Learning | `03_Classical_ML.md` | Done |
| 4. Deep Learning Foundations | `04_Deep_Learning_Foundations.md` | Pending |
| 5. Tokenization | `05_Tokenization.md` | Done |
| 6. Text Preprocessing & Embeddings | `06_Text_Preprocessing_Embeddings.md` | Pending |
| 7. Sequence Modeling | `07_Sequence_Modeling.md` | Pending |
| 8. Attention Mechanisms | `08_Attention_Mechanisms.md` | Done |
| 9. Transformer Architecture | `09_Transformer_Architecture.md` | Done |
| 10. Positional Encodings | `10_Positional_Encodings.md` | Done |
| 11. Encoder Models (BERT Family) | `11_BERT_Family.md` | Done |
| 12. Decoder Models (GPT & LLMs) | `12_GPT_Open_Source_LLMs.md` | Done |
| 13. Encoder-Decoder & Unified Models | `13_Encoder_Decoder_Models.md` | Done |
| 14. Pretraining LLMs | `14_Pretraining.md` | Done |
| 15. Fine-Tuning & PEFT | `15_Fine_Tuning_PEFT.md` | Done |
| 16. Alignment (RLHF, DPO) | `16_Alignment_RLHF_DPO.md` | Pending |
| 17. Decoding & Text Generation | `17_Decoding_Strategies.md` | Pending |
| 18. Prompt Engineering | `18_Prompt_Engineering.md` | Pending |
| 19. In-Context Learning (Theory) | `19_In_Context_Learning.md` | Pending |
| 20. RAG | `20_RAG.md` | Done |
| 21. Agents & Tool Use | `21_Agents_Tool_Use.md` | Done |
| 22. Multimodal AI | `22_Multimodal_AI.md` | Done |
| 23. Inference Optimization | `23_Inference_Optimization.md` | Done |
| 24. ML System Design | `24_ML_System_Design.md` | Done |
| 25. Evaluation & Benchmarking | `25_Evaluation_Benchmarks.md` | Done |
| 26. Safety & Ethics | `26_Safety_Ethics.md` | Pending |
| 27. Research Frontiers | `27_Research_Frontiers.md` | Pending |
| 28. Interview Strategy | `28_Interview_Strategy.md` | Pending |

---

## Suggested Study Order & Time Allocation

For someone with ML/DL background (adjust if starting fresh):

| Phase | Topics | Suggested Time | Priority |
|-------|--------|----------------|----------|
| A: Foundations | 1-4 | 1 week | Review if strong, study if weak |
| B: NLP Core | 5-7 | 3-4 days | Review your existing notes |
| C: Transformers | 8-10 | 1 week | **Critical -- master this cold** |
| D: Language Models | 11-13 | 1 week | **Critical -- know every model family** |
| E: Training LLMs | 14-16 | 1 week | **Critical -- RLHF/DPO asked in every interview** |
| F: Generation & Prompting | 17-19 | 4-5 days | Important for applied roles |
| G: Applications | 20-22 | 1 week | **RAG is almost always asked** |
| H: Production | 23-24 | 4-5 days | System design round prep |
| I: Frontiers | 25-27 | 4-5 days | Differentiator -- shows research awareness |
| J: Interview | 28 | Ongoing | Behavioral prep throughout |

**Total**: ~6-8 weeks at 4-5 hours/day

---


*Created: February 2026 | Target: AI Scientist (GenAI/LLM focus)*
