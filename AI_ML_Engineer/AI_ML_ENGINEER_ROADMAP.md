# AI/ML Engineer -- Complete Interview & Learning Roadmap

> **Target Roles**: ML Engineer, AI Engineer, Data Scientist, MLOps Engineer, NLP Engineer
> **Approach**: Practice-first, code-heavy, build real systems -- learn like an engineer, not just a theorist
> **How to use**: Study topics in order. Ask me to generate detailed notes for each topic one by one.

---

### Formatting Guidelines (for Claude -- follow these when generating notes)

- **Code**: Heavy. Every concept gets a working implementation. Use Python with PyTorch, sklearn, FastAPI, etc. Code should be production-quality with type hints, error handling, and comments.
- **Equations**: Use LaTeX. Block equations with `$$...$$`, inline math with `$...$`. Only where they clarify -- this is not a math course.
- **Diagrams**: Use Mermaid diagrams inside ```mermaid code blocks for architecture flows, pipelines, and system designs. Notion renders these.
- **ASCII art**: Use for model architectures, data flow, and comparison layouts where Mermaid isn't suitable.
- **Structure**: Every topic file must include:
  - Table of Contents
  - Concepts + Code interleaved (explain, then implement)
  - Practice exercises with checkboxes `- [ ]`
  - A Mini-Project section
  - An "Interview Questions & Answers" section at the end

---

## How This Roadmap Is Organized

```
Phase A: Foundations (Topics 1-2)          -- Python + just enough math.
Phase B: Classical ML (Topics 3-5)         -- The models that still win on tabular data.
Phase C: Deep Learning (Topics 6-9)        -- Build neural nets from scratch, master PyTorch.
Phase D: NLP & Transformers (Topics 10-12) -- Text processing through fine-tuning.
Phase E: LLMs & GenAI (Topics 13-16)      -- APIs, RAG, agents, prompting -- the 2025 stack.
Phase F: MLOps & Production (Topics 17-20) -- Deploy, monitor, scale -- what makes you an engineer.
Phase G: Specialization (Topics 21-25)     -- CV, RecSys, time series, optimization, cloud.
Phase H: Interview Prep (Topics 26-28)     -- DSA, mock interviews, portfolio.
```

Each topic below lists: what it covers, why it matters for an ML Engineer interview, key questions you should be able to answer, and hands-on practice exercises.

---

## Phase A: Python & Math Foundations

You can't build ML without solid Python and enough math to understand what's happening under the hood. This phase is fast -- we cover only what's needed, not a full math degree.

---

### Topic 1: Python for ML Engineers

**What you'll learn**:
- Python essentials (if you already know Python, skim this):
  - Data structures: lists, dicts, sets, tuples, defaultdict, Counter
  - List comprehensions, generators, decorators
  - OOP: classes, inheritance, dunder methods, dataclasses
  - Type hints and why they matter in production
- NumPy:
  - Array creation, indexing, slicing, broadcasting
  - Vectorized operations vs loops (why NumPy is 100x faster)
  - Linear algebra: dot, matmul, einsum
  - Random number generation, seeding for reproducibility
- Pandas:
  - DataFrame operations: filter, groupby, merge, pivot
  - Handling missing data, dtypes, memory optimization
  - Method chaining for clean data pipelines
- Visualization: Matplotlib, Seaborn basics
- Python tooling: virtual environments, pip, git basics

**Why it matters for ML Engineer**:
Every ML engineer interview starts with a coding round. You'll be writing Python under time pressure -- fluency with NumPy, Pandas, and Python idioms is non-negotiable. Production ML code requires OOP, type hints, and clean architecture.

**You should be able to answer**:
- What is broadcasting in NumPy? Give an example where it avoids an explicit loop.
- When would you use a generator instead of a list comprehension?
- How do you handle a DataFrame with 50% missing values in a column?
- Write a decorator that caches function results (memoization).

**Practice**:
- [ ] Implement matrix multiplication from scratch, then compare with NumPy
- [ ] Build a data cleaning pipeline for a messy CSV dataset
- [ ] Write a decorator that times any function
- [ ] Mini-project: EDA notebook on a Kaggle dataset (Titanic or House Prices)

**File**: `01_Python_For_ML.md`

---

### Topic 2: Math Essentials (The ML Engineer's Subset)

**What you'll learn**:
- Linear algebra (practical):
  - Vectors, matrices, transpose, inverse
  - Dot product as similarity, matrix multiply as transformation
  - Eigenvalues/eigenvectors (intuition for PCA)
  - SVD (used in recommendations, embeddings)
- Calculus (just enough):
  - Derivatives and the chain rule (backbone of backpropagation)
  - Partial derivatives, gradients
  - Gradient descent: intuition and the update rule
- Probability & statistics (practical):
  - Bayes' theorem (classification intuition)
  - Distributions: Normal, Bernoulli, Uniform
  - Mean, variance, standard deviation
  - Hypothesis testing: p-values, A/B testing basics
  - Correlation vs causation
- Information theory:
  - Entropy, cross-entropy, KL divergence (loss functions!)

**Why it matters for ML Engineer**:
You don't need to prove theorems, but you do need to understand why gradient descent converges, what cross-entropy loss is doing, and how PCA reduces dimensions. Every "explain this model" interview question requires math intuition.

**You should be able to answer**:
- Derive the gradient descent update rule for linear regression.
- What is cross-entropy loss? Why is it preferred over MSE for classification?
- Explain PCA in terms of eigenvalues. What do the eigenvalues represent?
- How would you design an A/B test to compare two recommendation algorithms?

**Practice**:
- [ ] Implement gradient descent from scratch for y = mx + b
- [ ] Compute PCA on a dataset using eigenvectors (then verify with sklearn)
- [ ] Simulate A/B test: generate data, compute p-value, make decision
- [ ] Mini-project: Build a linear regression from scratch using only NumPy

**File**: `02_Math_Essentials.md`

---

## Phase B: Classical Machine Learning

Every ML engineer must understand classical ML deeply -- not just call sklearn. You need to know what's happening inside, when to use what, and how to debug models. Gradient boosting still wins most tabular data competitions.

---

### Topic 3: Supervised Learning -- Regression

**What you'll learn**:
- Linear regression:
  - Normal equation (closed-form) and gradient descent (iterative)
  - Feature scaling: StandardScaler, MinMaxScaler -- why it matters
  - Regularization: L1 (Lasso), L2 (Ridge), ElasticNet
  - Polynomial features and overfitting
- Evaluation metrics:
  - MSE, RMSE, MAE, R-squared -- when to use which
- Practical sklearn workflow:
  - train_test_split, Pipeline, ColumnTransformer
  - Cross-validation: KFold, StratifiedKFold
- Feature engineering:
  - Handling categorical variables: OneHot, Label, Target encoding
  - Handling missing values: imputation strategies
  - Feature selection: correlation, mutual information, RFE

**Why it matters for ML Engineer**:
Regression is the "hello world" of ML interviews. More importantly, sklearn Pipelines and feature engineering are daily production work. Interviewers test whether you can build a complete, reproducible ML pipeline -- not just call `.fit()`.

**You should be able to answer**:
- Implement linear regression with gradient descent from scratch.
- What is the difference between L1 and L2 regularization? When would you choose each?
- How do you handle categorical variables with 1000+ categories?
- Walk through building a production sklearn Pipeline with preprocessing and model.

**Practice**:
- [ ] Implement linear regression from scratch (gradient descent + normal equation)
- [ ] Build a full regression pipeline on Kaggle House Prices dataset
- [ ] Compare L1 vs L2 regularization -- visualize coefficient paths
- [ ] Mini-project: End-to-end regression with feature engineering, CV, and model selection

**File**: `03_Supervised_Regression.md`

---

### Topic 4: Supervised Learning -- Classification

**What you'll learn**:
- Logistic regression:
  - Sigmoid function, log-loss, decision boundary
  - Multi-class: one-vs-rest, softmax
- Decision trees:
  - Splitting criteria: Gini impurity, entropy, information gain
  - Pruning: pre-pruning (max_depth, min_samples) vs post-pruning
  - Visualizing decision trees
- Ensemble methods:
  - Random Forest: bagging + feature randomness
  - Gradient Boosting: XGBoost, LightGBM, CatBoost
  - Why ensembles almost always win on tabular data
- Support Vector Machines:
  - Maximum margin, kernel trick (RBF, polynomial)
  - When SVMs still make sense
- Evaluation metrics:
  - Confusion matrix, accuracy, precision, recall, F1
  - ROC-AUC, PR-AUC -- when to use which
  - Handling imbalanced classes: SMOTE, class weights, threshold tuning

**Why it matters for ML Engineer**:
Classification is the most common ML task in production -- spam detection, churn prediction, fraud detection. XGBoost/LightGBM are the go-to models for tabular data at most companies. Understanding precision/recall trade-offs is critical for real-world systems where false positives and false negatives have different costs.

**You should be able to answer**:
- Implement logistic regression from scratch using gradient descent.
- Why is AUC-ROC misleading for imbalanced datasets? What should you use instead?
- Compare Random Forest vs XGBoost. When would you pick each?
- How do you tune the threshold for a fraud detection model where false negatives cost 100x more than false positives?

**Practice**:
- [ ] Implement logistic regression from scratch (gradient descent + sigmoid)
- [ ] Build a decision tree from scratch (recursive splitting)
- [ ] Kaggle competition: Classify with XGBoost, tune hyperparameters
- [ ] Mini-project: Credit card fraud detection (imbalanced classification)

**File**: `04_Supervised_Classification.md`

---

### Topic 5: Unsupervised Learning & Dimensionality Reduction

**What you'll learn**:
- Clustering:
  - K-Means: algorithm, elbow method, silhouette score
  - DBSCAN: density-based, handles arbitrary shapes
  - Hierarchical clustering: dendrogram
- Dimensionality reduction:
  - PCA: variance maximization, scree plot, choosing components
  - t-SNE: neighbor preservation, perplexity parameter
  - UMAP: faster, better global structure than t-SNE
- Anomaly detection:
  - Isolation Forest, Local Outlier Factor
  - Statistical methods: Z-score, IQR

**Why it matters for ML Engineer**:
Customer segmentation, anomaly detection, and embedding visualization are core ML engineering tasks. PCA is used for feature reduction in high-dimensional pipelines. t-SNE/UMAP are essential for debugging embeddings and understanding model representations.

**You should be able to answer**:
- Implement K-Means from scratch. What is the convergence guarantee?
- How do you choose the number of clusters? Compare elbow method vs silhouette score.
- When would you use DBSCAN over K-Means?
- You have 10,000-dimensional features. Walk through a dimensionality reduction strategy.

**Practice**:
- [ ] Implement K-Means from scratch
- [ ] Cluster customer segments on an e-commerce dataset
- [ ] Visualize high-dimensional embeddings with t-SNE and UMAP
- [ ] Mini-project: Anomaly detection on network intrusion dataset

**File**: `05_Unsupervised_Learning.md`

---

## Phase C: Deep Learning Foundations

The core of modern AI engineering. You must be fluent in building, training, and debugging neural networks. This phase takes you from a single neuron to convolutional and recurrent architectures.

---

### Topic 6: Neural Networks from Scratch

**What you'll learn**:
- The neuron: weighted sum + activation
- Activation functions: ReLU, Sigmoid, Tanh, GELU, Swish
- Forward pass and computation graphs
- Backpropagation: chain rule applied layer by layer
- Loss functions: MSE, cross-entropy, binary cross-entropy
- Optimizers:
  - SGD, SGD with momentum
  - Adam, AdamW -- why Adam is the default
  - Learning rate scheduling: cosine, warmup, step decay
- Weight initialization: Xavier, He, why it matters
- Regularization: Dropout, weight decay, early stopping, batch norm

**Why it matters for ML Engineer**:
"Implement a neural network from scratch" is one of the most common ML coding interview questions. Understanding backpropagation deeply lets you debug training issues (vanishing gradients, NaN losses, slow convergence) that sklearn users never encounter.

**You should be able to answer**:
- Implement forward and backward pass for a 2-layer MLP from scratch.
- Why does He initialization work better than random for ReLU networks?
- Explain Adam optimizer. What are the two moments it tracks and why?
- Your model's loss is NaN after 100 steps. Walk through your debugging process.

**Practice**:
- [ ] Implement a 2-layer neural network from scratch in NumPy (forward + backward)
- [ ] Train it on MNIST -- achieve >95% accuracy
- [ ] Visualize gradients and activations to understand vanishing/exploding gradients
- [ ] Mini-project: Build a digit classifier from scratch, then rebuild in PyTorch

**File**: `06_Neural_Networks_Scratch.md`

---

### Topic 7: PyTorch Mastery

**What you'll learn**:
- Tensors: creation, operations, GPU transfer, autograd
- nn.Module: building models with layers
- Datasets and DataLoaders: custom datasets, batching, shuffling, num_workers
- Training loop: forward -> loss -> backward -> step -> zero_grad
- Saving/loading: state_dict, checkpoints
- Mixed precision training: torch.cuda.amp
- Debugging: gradient checking, hooks, profiling
- Multi-GPU: DataParallel, DistributedDataParallel basics
- TorchScript and ONNX export

**Why it matters for ML Engineer**:
PyTorch is the industry standard for ML engineering. Every production ML team expects fluency -- not just calling existing models but writing custom training loops, datasets, and model architectures. Multi-GPU training and model export are production-critical skills.

**You should be able to answer**:
- Write a complete PyTorch training loop from memory (no looking up syntax).
- What is the difference between `model.eval()` and `torch.no_grad()`?
- How does `autograd` compute gradients? What is the computation graph?
- Your DataLoader is bottlenecking training. How do you diagnose and fix it?

**Practice**:
- [ ] Rebuild the MNIST classifier in PyTorch (nn.Module, DataLoader)
- [ ] Implement a custom Dataset class for an image folder
- [ ] Train with mixed precision and compare speed
- [ ] Mini-project: Image classification on CIFAR-10 with CNN (build training pipeline from scratch)

**File**: `07_PyTorch_Mastery.md`

---

### Topic 8: Convolutional Neural Networks (CNNs)

**What you'll learn**:
- Convolution operation: filters, stride, padding, output size formula
- Pooling: max pooling, average pooling, global average pooling
- Classic architectures: LeNet -> AlexNet -> VGG -> ResNet -> EfficientNet
- Residual connections: why they enable deep networks
- Transfer learning:
  - Feature extraction (freeze backbone)
  - Fine-tuning (unfreeze last N layers)
  - When to use which
- Data augmentation: RandomCrop, Flip, ColorJitter, Mixup, CutMix
- Object detection overview: YOLO, Faster R-CNN (concepts)

**Why it matters for ML Engineer**:
Transfer learning with pretrained CNNs is one of the most practical skills in ML -- you can solve most image tasks in a day with a fine-tuned ResNet. Data augmentation and the output size formula come up constantly in both interviews and production work.

**You should be able to answer**:
- Calculate the output dimensions of a conv layer given input size, kernel, stride, and padding.
- What are residual connections? Why do they enable training of very deep networks?
- When do you freeze the backbone vs fine-tune end-to-end?
- Compare YOLO vs Faster R-CNN for object detection. Trade-offs?

**Practice**:
- [ ] Implement a CNN from scratch in PyTorch
- [ ] Transfer learning: Fine-tune ResNet50 on a custom dataset (Flowers, Food-101)
- [ ] Visualize conv filters and feature maps
- [ ] Mini-project: Build an image classifier with data augmentation, transfer learning, achieve >90% on a custom dataset

**File**: `08_CNNs.md`

---

### Topic 9: Recurrent Networks & Sequence Models

**What you'll learn**:
- RNN: architecture, hidden state, backpropagation through time (BPTT)
- Vanishing/exploding gradients in RNNs
- LSTM: forget gate, input gate, output gate, cell state
- GRU: simplified LSTM (reset + update gates)
- Bidirectional RNNs
- Sequence-to-sequence: encoder-decoder for translation
- Why transformers replaced RNNs (parallelization, long-range dependencies)
- 1D CNNs for sequences (an alternative to RNNs)

**Why it matters for ML Engineer**:
While transformers dominate NLP, LSTMs remain relevant for time series, streaming data, and edge deployment. Understanding RNNs helps you appreciate WHY transformers work better. Seq2seq is the foundation for all encoder-decoder models.

**You should be able to answer**:
- Implement an LSTM cell from scratch. Explain each gate's purpose.
- Why do RNNs suffer from vanishing gradients? How does LSTM solve this?
- Why did transformers replace RNNs? Give 3 concrete technical reasons.
- When would you still choose an LSTM over a transformer in production?

**Practice**:
- [ ] Implement a character-level RNN from scratch
- [ ] Build a sentiment classifier with LSTM in PyTorch
- [ ] Compare RNN vs LSTM vs GRU on a time series task
- [ ] Mini-project: Text generation with character-level LSTM (train on Shakespeare)

**File**: `09_RNNs_Sequences.md`

---

## Phase D: NLP & Transformers

The heart of modern AI. Every ML engineer in 2025+ needs transformer fluency -- not just using HuggingFace, but understanding the internals well enough to debug, optimize, and extend.

---

### Topic 10: NLP Fundamentals & Text Processing

**What you'll learn**:
- Text preprocessing:
  - Tokenization: word, subword (BPE, WordPiece, SentencePiece)
  - Lowercasing, stemming, lemmatization
  - Stop word removal -- when it helps, when it hurts
- Text representation:
  - Bag of Words, TF-IDF
  - Word embeddings: Word2Vec (CBOW, Skip-gram), GloVe
  - Why static embeddings failed for polysemy
- Text classification with traditional ML:
  - TF-IDF + Logistic Regression (strong baseline!)
  - Naive Bayes for text
- Regular expressions for text extraction
- spaCy and NLTK practical usage

**Why it matters for ML Engineer**:
TF-IDF + LogReg remains a shockingly strong baseline that beats many overengineered solutions. Understanding tokenization is critical because it directly impacts model performance. In interviews, "build a text classifier" is a standard question -- knowing the simple baselines earns trust before you mention transformers.

**You should be able to answer**:
- Implement TF-IDF from scratch. What does the IDF term do?
- How does BPE tokenization work? Walk through the algorithm step by step.
- When would TF-IDF + LogReg beat a fine-tuned BERT model?
- Your NLP model performs poorly on misspelled text. What are your options?

**Practice**:
- [ ] Build a text classification pipeline: preprocessing -> TF-IDF -> LogReg
- [ ] Train Word2Vec on a custom corpus, visualize embeddings
- [ ] Implement TF-IDF from scratch
- [ ] Mini-project: Spam classifier (SMS Spam Collection dataset) -- compare Naive Bayes vs TF-IDF+LogReg vs simple neural network

**File**: `10_NLP_Fundamentals.md`

---

### Topic 11: Transformers & Attention

**What you'll learn**:
- Self-attention: Query, Key, Value mechanism
- Scaled dot-product attention: why scale by sqrt(d)
- Multi-head attention: parallel attention with different learned projections
- Transformer architecture:
  - Encoder block: self-attention -> add & norm -> FFN -> add & norm
  - Decoder block: masked self-attention -> cross-attention -> FFN
  - Positional encoding: sinusoidal and learned
- BERT: bidirectional encoder, MLM + NSP, [CLS] token
- GPT: autoregressive decoder, causal masking
- Using HuggingFace Transformers library:
  - AutoModel, AutoTokenizer, pipeline API
  - Tokenizer internals: input_ids, attention_mask, token_type_ids

**Why it matters for ML Engineer**:
Transformers are THE architecture of modern AI. You need to understand them well enough to debug attention issues, implement custom heads, and know when encoder vs decoder models are appropriate. HuggingFace is the standard library -- fluency is expected.

**You should be able to answer**:
- Implement scaled dot-product attention in PyTorch from scratch.
- Why do we scale by sqrt(d_k)? What happens without scaling?
- Compare BERT (encoder) vs GPT (decoder). When do you use each?
- How does the HuggingFace tokenizer handle out-of-vocabulary words?

**Practice**:
- [ ] Implement scaled dot-product attention from scratch in PyTorch
- [ ] Implement a single transformer encoder block from scratch
- [ ] Use HuggingFace pipeline for sentiment analysis, NER, summarization
- [ ] Mini-project: Fine-tune BERT for text classification on IMDB reviews (HuggingFace Trainer)

**File**: `11_Transformers_Attention.md`

---

### Topic 12: Fine-Tuning & Transfer Learning for NLP

**What you'll learn**:
- Why fine-tuning works: pretrain on large data, adapt to task
- Full fine-tuning vs feature extraction
- HuggingFace Trainer API:
  - TrainingArguments, evaluation strategy, callbacks
  - Logging to W&B or TensorBoard
- Parameter-efficient fine-tuning (PEFT):
  - LoRA: low-rank adaptation -- how and why it works
  - QLoRA: LoRA + 4-bit quantization
  - Prefix tuning, prompt tuning
- When to fine-tune vs when to prompt:
  - Fine-tune: specific format, domain adaptation, consistent behavior
  - Prompt: quick iteration, diverse tasks, no training data
- Dataset preparation:
  - Data formatting for instruction tuning
  - Alpaca format, ChatML format, ShareGPT format

**Why it matters for ML Engineer**:
Fine-tuning is the most practical LLM skill. LoRA/QLoRA made it possible to fine-tune 70B models on consumer hardware. Knowing when to fine-tune vs prompt vs RAG is a decision you'll make weekly. HuggingFace Trainer is the standard tool -- you must be fluent.

**You should be able to answer**:
- What is LoRA? Explain it in terms of matrix factorization.
- When would you fine-tune vs use few-shot prompting vs build RAG?
- How does QLoRA enable fine-tuning a 70B model on a single GPU?
- Walk through preparing a dataset for instruction tuning. What format would you use?

**Practice**:
- [ ] Fine-tune BERT for NER on CoNLL-2003
- [ ] Fine-tune a small LLM (Llama 3.2 1B or Phi-3 mini) with LoRA using PEFT library
- [ ] Compare full fine-tuning vs LoRA on a classification task (accuracy + training time)
- [ ] Mini-project: Fine-tune an open-source LLM for a custom task (e.g., customer support classification) using QLoRA

**File**: `12_Fine_Tuning_NLP.md`

---

## Phase E: LLMs & GenAI Engineering

The most in-demand skills for AI/ML engineers in 2025-26. Building with LLMs, not just understanding them. This phase is about the practical engineering of GenAI systems.

---

### Topic 13: Working with LLM APIs

**What you'll learn**:
- OpenAI API:
  - Chat completions, system/user/assistant messages
  - Temperature, top_p, max_tokens, stop sequences
  - Streaming responses
  - Function calling / tool use
  - Structured outputs (JSON mode)
- Anthropic API (Claude):
  - Messages API, system prompts
  - Tool use pattern
- Open-source model serving:
  - Ollama for local inference
  - vLLM for production serving
  - HuggingFace Inference API
- Prompt engineering:
  - Zero-shot, few-shot, chain-of-thought
  - System prompts, role-based prompting
  - Output formatting and parsing
- Cost management:
  - Token counting, cost estimation
  - Caching strategies
  - Model selection by task complexity

**Why it matters for ML Engineer**:
LLM APIs are now as fundamental as REST APIs. Every AI product uses them. Knowing how to stream responses, manage costs, implement function calling, and choose between models is daily work for AI engineers. Companies ask "build an LLM-powered feature" in take-home assignments.

**You should be able to answer**:
- How does function calling work in OpenAI's API? Walk through the message flow.
- What is the difference between temperature and top_p? When do you adjust each?
- How would you estimate the cost of an LLM-powered feature serving 100K users/day?
- Compare running Llama locally with Ollama vs using an API. When would you choose each?

**Practice**:
- [ ] Build a CLI chatbot using OpenAI API with streaming
- [ ] Implement function calling: LLM calls a weather API
- [ ] Run Llama 3 locally with Ollama, build a chat interface
- [ ] Mini-project: Build a multi-turn conversational assistant with memory (conversation history management)

**File**: `13_LLM_APIs.md`

---

### Topic 14: RAG (Retrieval-Augmented Generation)

**What you'll learn**:
- RAG pipeline end-to-end:
  - Document loading (PDF, HTML, markdown, code)
  - Chunking strategies: fixed, recursive, semantic
  - Embedding models: OpenAI, Sentence-Transformers, Cohere
  - Vector databases: ChromaDB, FAISS, Pinecone, Qdrant
  - Retrieval: similarity search, MMR (maximal marginal relevance)
  - Prompt construction: query + context + instructions
  - Generation with citations
- Advanced RAG:
  - Hybrid search: dense + sparse (BM25)
  - Re-ranking with cross-encoders
  - Parent-child chunking
  - Multi-query retrieval
  - Contextual compression
- LangChain and LlamaIndex:
  - Document loaders, text splitters, vector stores
  - Chains and retrieval chains
  - Comparison: when to use which framework
- Evaluation:
  - Retrieval metrics: recall@k, MRR
  - Generation metrics: faithfulness, relevance
  - RAGAS framework

**Why it matters for ML Engineer**:
RAG is the #1 most-asked GenAI engineering topic in interviews. It's how most companies add LLMs to their products without fine-tuning. Building a good RAG system requires engineering judgment -- chunking strategy, embedding choice, retrieval method, and prompt design all compound.

**You should be able to answer**:
- Build a RAG pipeline from scratch without frameworks. What are the components?
- How do you choose chunk size? What are the trade-offs?
- Compare dense retrieval vs BM25 vs hybrid. When does each win?
- Your RAG system returns irrelevant documents. Walk through your debugging process.

**Practice**:
- [ ] Build a basic RAG pipeline from scratch (no framework -- just embeddings + FAISS + LLM)
- [ ] Rebuild using LangChain, compare code complexity
- [ ] Implement hybrid search (BM25 + dense) with re-ranking
- [ ] Mini-project: "Chat with your PDF" app -- upload PDF, ask questions, get cited answers

**File**: `14_RAG.md`

---

### Topic 15: AI Agents & Tool Use

**What you'll learn**:
- Agent architecture: LLM + tools + memory + loop
- ReAct pattern: Reasoning + Acting
- Tool/function calling:
  - Defining tool schemas
  - LLM decides which tool to call and with what arguments
  - Parsing and executing tool calls
  - Handling errors and retries
- Memory for agents:
  - Short-term (conversation buffer)
  - Long-term (vector store of past interactions)
  - Working memory (scratchpad for current task)
- Multi-agent systems:
  - Supervisor pattern
  - Debate pattern
  - Pipeline pattern
- Frameworks:
  - LangGraph: stateful agent graphs
  - CrewAI: multi-agent orchestration
  - OpenAI Assistants API
- Agent evaluation and reliability

**Why it matters for ML Engineer**:
Agents are the next evolution of LLM applications. Companies are building AI agents for customer service, code generation, data analysis, and workflow automation. Understanding the ReAct pattern, tool calling, and agent reliability is becoming a core engineering skill.

**You should be able to answer**:
- Build a ReAct agent from scratch. What is the thought-action-observation loop?
- How does LangGraph differ from LangChain? When do you need stateful agents?
- Your agent gets stuck in an infinite loop. What are the common causes and fixes?
- Design a multi-agent system for customer support. How do agents hand off tasks?

**Practice**:
- [ ] Build a ReAct agent from scratch (no framework) with 3 tools (calculator, search, file reader)
- [ ] Build an agent using LangGraph with conditional routing
- [ ] Implement a multi-agent system: researcher + writer + reviewer
- [ ] Mini-project: Build a coding assistant agent that can read files, run code, and fix errors

**File**: `15_Agents_Tool_Use.md`

---

### Topic 16: Prompt Engineering & Advanced Techniques

**What you'll learn**:
- Prompting fundamentals:
  - Clear instructions, role assignment, examples
  - Output formatting: JSON, markdown, structured
- Advanced techniques:
  - Chain-of-thought (CoT) and zero-shot CoT ("Let's think step by step")
  - Self-consistency: sample multiple reasoning paths, take majority vote
  - Tree-of-thought: branching exploration
  - ReAct prompting for agents
- Structured output:
  - JSON mode, Pydantic models
  - Instructor library for reliable extraction
  - Retry and validation strategies
- Prompt optimization:
  - DSPy: programmatic prompt optimization
  - Prompt templates and versioning
- Guardrails:
  - Input validation
  - Output validation
  - Content filtering
  - Prompt injection defense

**Why it matters for ML Engineer**:
Prompt engineering is often the highest-ROI optimization for LLM applications. Before fine-tuning or complex architectures, a better prompt can solve the problem. Structured output extraction is critical for production systems. Guardrails are non-negotiable for customer-facing applications.

**You should be able to answer**:
- How does chain-of-thought improve LLM performance? Give an example.
- Implement structured data extraction from unstructured text using Pydantic.
- What is prompt injection? Build a basic defense.
- Compare DSPy vs manual prompt engineering. When is programmatic optimization worth it?

**Practice**:
- [ ] Build a structured data extractor: resume -> JSON (name, skills, experience)
- [ ] Implement chain-of-thought vs direct prompting, compare accuracy on math problems
- [ ] Build a prompt injection detector
- [ ] Mini-project: Build an AI-powered form filler -- extracts structured data from unstructured text using Pydantic + Instructor

**File**: `16_Prompt_Engineering.md`

---

## Phase F: MLOps & Production

What separates a notebook ML practitioner from a production ML engineer. This phase is about deployment, monitoring, and reliability. If you can't deploy it, it doesn't count.

---

### Topic 17: ML System Design & Architecture

**What you'll learn**:
- ML system design framework:
  - Requirements -> Data -> Model -> Serving -> Evaluation -> Monitoring
- Design patterns:
  - Online vs batch prediction
  - Feature stores: what and why
  - Model registry
  - A/B testing infrastructure
- Data engineering basics for ML:
  - ETL pipelines
  - Data versioning (DVC)
  - Data validation (Great Expectations)
- Architecture patterns:
  - Monolith vs microservices for ML
  - Event-driven ML pipelines
  - Lambda architecture for real-time + batch

**Why it matters for ML Engineer**:
System design is a dedicated interview round at most companies. You need to design end-to-end ML systems -- from data ingestion to model serving to monitoring. This is the round that separates senior engineers from juniors.

**You should be able to answer**:
- Design a recommendation system for an e-commerce platform. Walk through every component.
- When do you use online prediction vs batch prediction?
- What is a feature store? Why does it prevent training-serving skew?
- Design an A/B testing framework for comparing two ML models in production.

**Practice**:
- [ ] Design a recommendation system on paper (whiteboard style)
- [ ] Design a fraud detection system (real-time + batch)
- [ ] Design a RAG-based Q&A system for 10K users
- [ ] Mini-project: Write a complete system design document for an LLM chatbot (requirements, architecture, evaluation plan)

**File**: `17_ML_System_Design.md`

---

### Topic 18: Model Deployment & Serving

**What you'll learn**:
- REST API serving:
  - FastAPI for ML models
  - Request/response schemas with Pydantic
  - Async endpoints for LLM streaming
  - Error handling, health checks
- Containerization:
  - Docker for ML: writing Dockerfiles, GPU support
  - Docker Compose for multi-service setups
  - Optimizing image sizes
- Model serving frameworks:
  - TorchServe, Triton Inference Server
  - vLLM for LLM serving
  - BentoML for packaging
- Cloud deployment:
  - AWS: SageMaker, EC2, Lambda
  - GCP: Vertex AI, Cloud Run
  - Azure: Azure ML
- Serverless ML:
  - AWS Lambda + API Gateway
  - Modal for serverless GPU
  - Replicate for model hosting

**Why it matters for ML Engineer**:
If you can't deploy a model, you're a data scientist, not an ML engineer. FastAPI + Docker is the minimum stack every ML engineer needs. In interviews, "how would you deploy this?" follows every model design question.

**You should be able to answer**:
- Write a FastAPI endpoint that serves a PyTorch model with proper error handling.
- Write a Dockerfile for a Python ML application with GPU support.
- Compare SageMaker vs self-hosted EC2 for model serving. Trade-offs?
- Your deployed model's latency doubled overnight. Walk through your investigation.

**Practice**:
- [ ] Wrap a sklearn model in a FastAPI endpoint with Pydantic schemas
- [ ] Dockerize the API, run it in a container
- [ ] Deploy an LLM with vLLM, expose via FastAPI
- [ ] Mini-project: Deploy a RAG application -- FastAPI backend + Streamlit frontend + Docker Compose

**File**: `18_Model_Deployment.md`

---

### Topic 19: MLOps -- Experiment Tracking, CI/CD, Monitoring

**What you'll learn**:
- Experiment tracking:
  - MLflow: logging params, metrics, artifacts, model registry
  - Weights & Biases: experiment tracking, hyperparameter sweeps
  - Comparing runs, reproducing experiments
- CI/CD for ML:
  - GitHub Actions for ML pipelines
  - Automated testing: unit tests, data tests, model tests
  - Model validation gates before deployment
- Monitoring in production:
  - Data drift detection
  - Model performance monitoring
  - Logging and alerting (Prometheus, Grafana)
  - LLM-specific monitoring: token usage, latency, quality
- Feature stores:
  - Feast: offline + online feature serving
  - Why feature stores prevent training-serving skew
- Pipeline orchestration:
  - Airflow, Prefect, or Dagster
  - Scheduled retraining pipelines

**Why it matters for ML Engineer**:
MLOps is what makes ML engineering a real engineering discipline. Experiment tracking ensures reproducibility. CI/CD prevents bad models from reaching production. Monitoring catches drift before users notice. These are the skills that senior ML engineers are hired for.

**You should be able to answer**:
- How do you detect data drift in production? What metrics would you track?
- Walk through setting up a CI/CD pipeline for an ML model deployment.
- Compare MLflow vs W&B. When would you choose each?
- Your model's accuracy dropped 5% over the last month. What's your investigation plan?

**Practice**:
- [ ] Set up MLflow for experiment tracking, log a training run
- [ ] Set up W&B for a PyTorch training run with hyperparameter sweep
- [ ] Write GitHub Actions CI: lint + test + model validation
- [ ] Mini-project: End-to-end MLOps pipeline -- train model -> log to MLflow -> validate -> deploy to FastAPI -> monitor with Prometheus

**File**: `19_MLOps.md`

---

### Topic 20: Databases & Data Infrastructure for ML

**What you'll learn**:
- SQL for ML engineers:
  - Joins, window functions, CTEs, aggregations
  - Query optimization basics
  - Writing efficient feature extraction queries
- NoSQL:
  - MongoDB for document storage
  - Redis for caching and feature stores
- Vector databases:
  - FAISS, ChromaDB, Pinecone, Qdrant, Weaviate
  - Indexing algorithms: HNSW, IVF, PQ
  - When to use which
- Data warehousing:
  - BigQuery, Snowflake, Redshift basics
  - Partitioning, clustering for performance
- Streaming:
  - Kafka basics for real-time ML pipelines
  - Event-driven architecture

**Why it matters for ML Engineer**:
Data infrastructure is the unsung hero of ML engineering. SQL is used in every ML interview. Vector databases are essential for RAG and embedding search. Redis caching can reduce your serving latency by 10x. These skills separate ML engineers from notebook practitioners.

**You should be able to answer**:
- Write a SQL query using window functions to compute rolling 7-day averages.
- Compare FAISS vs Pinecone vs ChromaDB. When would you use each?
- How does HNSW indexing work at a high level? What are the trade-offs vs brute force?
- Design a real-time feature pipeline using Kafka + Redis.

**Practice**:
- [ ] SQL challenge: 20 LeetCode SQL problems (Medium)
- [ ] Set up ChromaDB, index 1000 documents, query with metadata filters
- [ ] Build a Redis-backed feature cache for an ML API
- [ ] Mini-project: Build a data pipeline -- ingest from API -> store in PostgreSQL -> extract features -> serve via Redis cache

**File**: `20_Databases_Infrastructure.md`

---

## Phase G: Specialization & Advanced Topics

Choose 2-3 topics based on your target role. Not all are required -- pick what aligns with your career goals. These are the topics that make you a specialist, not a generalist.

---

### Topic 21: Computer Vision in Production

**What you'll learn**:
- Image classification pipeline (end-to-end production)
- Object detection: YOLO v8, deployment considerations
- Image segmentation: U-Net, SAM (Segment Anything)
- OCR: Tesseract, PaddleOCR, document understanding
- Multimodal: CLIP for image-text, LLaVA for visual QA
- Edge deployment: ONNX, TensorRT, Core ML
- Video processing: frame extraction, tracking

**Why it matters for ML Engineer**:
Computer vision powers products from autonomous driving to medical imaging to retail analytics. YOLO and SAM are practical tools that can be deployed quickly. Edge deployment (ONNX, TensorRT) is increasingly important as models move to mobile and IoT devices.

**You should be able to answer**:
- Train and deploy a YOLO model for custom object detection. Walk through the pipeline.
- How do you export a PyTorch model to ONNX? What are the limitations?
- Compare YOLO vs Faster R-CNN for real-time object detection. Trade-offs?
- Design an OCR pipeline for processing scanned invoices.

**Practice**:
- [ ] Train YOLO v8 on a custom object detection dataset
- [ ] Build an OCR pipeline for invoices/receipts
- [ ] Deploy an image classifier to a mobile-friendly API (ONNX export)
- [ ] Mini-project: Build a visual search engine -- upload image, find similar products

**File**: `21_Computer_Vision_Production.md`

---

### Topic 22: Recommendation Systems

**What you'll learn**:
- Collaborative filtering:
  - User-based and item-based
  - Matrix factorization (SVD, ALS)
- Content-based filtering:
  - TF-IDF features, embedding-based similarity
- Deep learning for recommendations:
  - Two-tower models (candidate generation + ranking)
  - Neural collaborative filtering
  - Session-based recommendations with transformers
- System design:
  - Candidate generation -> ranking -> re-ranking pipeline
  - Feature engineering for recommendations
  - Handling cold start
- Evaluation: precision@k, recall@k, NDCG, MAP

**Why it matters for ML Engineer**:
Recommendation systems drive revenue at every tech company -- Netflix, Amazon, Spotify, YouTube. The two-tower architecture (candidate generation + ranking) is used universally. This is one of the most common ML system design interview questions.

**You should be able to answer**:
- Implement matrix factorization for collaborative filtering from scratch.
- Design a two-tower recommendation model. What goes in each tower?
- How do you handle the cold start problem for new users and new items?
- Walk through the full candidate generation -> ranking -> re-ranking pipeline.

**Practice**:
- [ ] Build a collaborative filtering recommender from scratch (matrix factorization)
- [ ] Build a content-based recommender using sentence embeddings
- [ ] Implement a two-tower model in PyTorch
- [ ] Mini-project: Movie recommendation system with collaborative + content hybrid approach

**File**: `22_Recommendation_Systems.md`

---

### Topic 23: Time Series & Forecasting

**What you'll learn**:
- Classical methods: ARIMA, SARIMA, Exponential Smoothing
- Feature engineering for time series:
  - Lag features, rolling statistics, date features
  - Fourier features for seasonality
- ML for time series: XGBoost, LightGBM with time features
- Deep learning: LSTM, Temporal Convolutional Networks
- Modern approaches: N-BEATS, TFT (Temporal Fusion Transformer), TimesFM
- Evaluation: MAE, MAPE, RMSE, proper cross-validation for time series

**Why it matters for ML Engineer**:
Demand forecasting, stock prediction, anomaly detection in metrics -- time series is everywhere. The key insight is that time series cross-validation is fundamentally different from standard CV. XGBoost with engineered features often beats deep learning on tabular time series.

**You should be able to answer**:
- Why can't you use standard K-Fold CV for time series? What do you use instead?
- Compare ARIMA vs XGBoost vs LSTM for forecasting. When does each win?
- What are lag features? How do you choose the right number of lags?
- Design a demand forecasting system for an e-commerce company.

**Practice**:
- [ ] Forecast store sales with ARIMA and Prophet
- [ ] Build a time series classifier for activity recognition
- [ ] Compare XGBoost vs LSTM on a forecasting task
- [ ] Mini-project: Demand forecasting pipeline -- data ingestion -> feature engineering -> model -> evaluation -> API

**File**: `23_Time_Series.md`

---

### Topic 24: LLM Inference Optimization

**What you'll learn**:
- Quantization: FP16 -> INT8 -> INT4 (GPTQ, AWQ, GGUF)
- KV cache management and memory optimization
- Flash Attention: why and how
- Serving frameworks: vLLM, TGI, TensorRT-LLM, SGLang
- Speculative decoding
- Batching: static -> continuous batching
- Benchmarking: latency, throughput, cost per token
- GPU selection and sizing

**Why it matters for ML Engineer**:
Serving LLMs at scale is one of the biggest engineering challenges in AI. Quantization, KV cache optimization, and continuous batching can reduce costs by 5-10x. Companies ask "how would you serve a 70B model?" in system design interviews.

**You should be able to answer**:
- Quantize a model to INT4. What is the quality vs speed trade-off?
- How does vLLM's paged attention improve over naive KV cache management?
- Compare vLLM vs TGI vs TensorRT-LLM. When would you choose each?
- Walk through serving a 70B model: hardware, quantization, batching, and framework choices.

**Practice**:
- [ ] Quantize a 7B model to INT4 using AutoGPTQ, compare quality and speed
- [ ] Set up vLLM, benchmark throughput at different batch sizes
- [ ] Compare GGUF CPU inference vs GPU inference on a 7B model
- [ ] Mini-project: Serve an open-source LLM in production -- vLLM + FastAPI + load testing with locust

**File**: `24_LLM_Inference_Optimization.md`

---

### Topic 25: Cloud & Infrastructure for ML

**What you'll learn**:
- AWS for ML:
  - EC2 (GPU instances), S3, SageMaker
  - Lambda for serverless, Step Functions for orchestration
  - IAM basics for security
- GCP for ML:
  - Vertex AI, Cloud Run, GCS
  - TPU basics
- Kubernetes basics:
  - Pods, deployments, services
  - Horizontal pod autoscaler
  - GPU scheduling
- Terraform basics for infrastructure as code
- Cost optimization:
  - Spot/preemptible instances
  - Right-sizing GPU instances
  - Auto-scaling strategies

**Why it matters for ML Engineer**:
Production ML runs on the cloud. Knowing AWS/GCP is expected for ML engineering roles. Kubernetes is increasingly required for scaling ML services. Cost optimization is a constant concern -- GPU costs add up fast.

**You should be able to answer**:
- Compare SageMaker vs EC2 for model training. When is SageMaker worth the premium?
- Walk through deploying an ML model on Kubernetes with auto-scaling.
- How do spot instances work? What are the risks for ML training?
- Your GPU costs are $20K/month. How do you reduce them by 3x?

**Practice**:
- [ ] Deploy a model on AWS SageMaker
- [ ] Set up a GPU instance on any cloud, train a model
- [ ] Containerize an ML app and deploy to Kubernetes (minikube locally)
- [ ] Mini-project: Deploy a complete ML application to AWS (EC2 + S3 + RDS + Docker)

**File**: `25_Cloud_Infrastructure.md`

---

## Phase H: Interview Prep & Portfolio

Consolidate everything, build your portfolio, and prepare for interviews. The best portfolio project is worth more than 100 LeetCode problems.

---

### Topic 26: DSA for ML Engineer Interviews

**What you'll learn**:
- Data structures: arrays, hash maps, trees, graphs, heaps, tries
- Algorithms: sorting, searching, BFS/DFS, dynamic programming
- ML-specific DSA:
  - Top-K problems (heap) -- used in recommendation, search
  - Similarity search (KD-tree, LSH)
  - Graph algorithms for social networks
  - Sampling algorithms (reservoir sampling)
- Complexity analysis: Big-O for time and space
- Practice strategy: LeetCode patterns, not random grinding

**Why it matters for ML Engineer**:
Most ML engineer interviews include a coding round with DSA problems. The bar is lower than SWE interviews (Medium difficulty is typical), but you still need to be solid. ML-specific problems (top-K, sampling, similarity search) come up frequently and are your opportunity to stand out.

**You should be able to answer**:
- Implement a min-heap from scratch. Use it to solve top-K efficiently.
- Find the K nearest neighbors in a dataset. What is the time complexity?
- Implement reservoir sampling for streaming data.
- Solve a graph problem: find connected components in a social network.

**Practice**:
- [ ] Solve 100 LeetCode problems (Easy: 30, Medium: 50, Hard: 20)
- [ ] Focus on: arrays, strings, hash maps, trees, graphs, DP
- [ ] Implement a min-heap from scratch
- [ ] Practice 10 ML-specific DSA problems (top-K, sampling, similarity)

**File**: `26_DSA_Interview.md`

---

### Topic 27: ML Interview Questions & System Design

**What you'll learn**:
- ML theory questions: bias-variance, overfitting, regularization, gradient descent
- ML coding: implement from scratch (logistic regression, k-means, decision tree)
- ML system design: recommendation system, fraud detection, search ranking, chatbot
- Behavioral: STAR method, leadership principles (Amazon), growth mindset
- Take-home projects: strategy for data science take-home assignments

**Why it matters for ML Engineer**:
This is the culmination of everything. ML interviews test breadth (do you know the fundamentals?) and depth (can you design a system?). Behavioral rounds are make-or-break -- companies reject technically strong candidates who can't communicate or collaborate.

**You should be able to answer**:
- Explain bias-variance tradeoff with a concrete example.
- Implement logistic regression from scratch in 30 minutes.
- Design a fraud detection system for a payments company (45-minute system design).
- Tell me about a time you disagreed with a teammate about a technical decision. (STAR format)

**Practice**:
- [ ] Mock interview: explain gradient descent to a non-technical person
- [ ] Mock interview: design a recommendation system (45 min)
- [ ] Implement logistic regression from scratch in 30 minutes
- [ ] Mini-project: Prepare a portfolio of 3-5 polished projects with READMEs

**File**: `27_ML_Interview_Prep.md`

---

### Topic 28: Portfolio Projects & GitHub

**What you'll learn**:
- Portfolio strategy: what projects impress hiring managers
- Project documentation: READMEs, architecture diagrams, demo videos
- GitHub profile optimization
- Blog writing: technical blog posts as a career accelerator
- Kaggle: competitions, notebooks, discussions
- Open source contributions: how to start

**Why it matters for ML Engineer**:
Your GitHub is your resume. A well-documented portfolio project demonstrates more skill than any certification. Hiring managers spend 30 seconds scanning your GitHub -- make it count. One deployed end-to-end project beats ten Jupyter notebooks.

**Recommended Portfolio Projects** (pick 3-5):

| Project | Skills Demonstrated |
|---------|-------------------|
| End-to-end ML pipeline (train -> deploy -> monitor) | MLOps, deployment, monitoring |
| RAG chatbot for a specific domain | LLMs, RAG, vector DB, deployment |
| AI agent with tool use | Agents, function calling, prompt engineering |
| Image classification with deployment | CV, transfer learning, Docker, API |
| Recommendation system | RecSys, data engineering, evaluation |
| Fine-tuned LLM for a specific task | Fine-tuning, LoRA, evaluation |
| Real-time dashboard with ML predictions | Full-stack, streaming, visualization |

**You should be able to answer**:
- Walk me through your most complex project. What decisions did you make and why?
- Show me the README. Can I run your project in 5 minutes?
- What would you do differently if you built this again?
- How did you evaluate your model? What were the limitations?

**Practice**:
- [ ] Build and deploy 3 portfolio projects with polished READMEs
- [ ] Write 2 technical blog posts about your projects
- [ ] Contribute to 1 open-source ML project (even a small fix counts)
- [ ] Mini-project: Create your complete GitHub portfolio -- profile README, pinned repos, project documentation

**File**: `28_Portfolio_Projects.md`

---

## Progress Tracker

| Topic | File Name | Status |
|-------|-----------|--------|
| 1. Python for ML Engineers | `01_Python_For_ML.md` | Pending |
| 2. Math Essentials | `02_Math_Essentials.md` | Pending |
| 3. Supervised Learning -- Regression | `03_Supervised_Regression.md` | Pending |
| 4. Supervised Learning -- Classification | `04_Supervised_Classification.md` | Pending |
| 5. Unsupervised Learning | `05_Unsupervised_Learning.md` | Pending |
| 6. Neural Networks from Scratch | `06_Neural_Networks_Scratch.md` | Pending |
| 7. PyTorch Mastery | `07_PyTorch_Mastery.md` | Pending |
| 8. CNNs | `08_CNNs.md` | Pending |
| 9. RNNs & Sequence Models | `09_RNNs_Sequences.md` | Pending |
| 10. NLP Fundamentals | `10_NLP_Fundamentals.md` | Pending |
| 11. Transformers & Attention | `11_Transformers_Attention.md` | Pending |
| 12. Fine-Tuning & Transfer Learning | `12_Fine_Tuning_NLP.md` | Pending |
| 13. Working with LLM APIs | `13_LLM_APIs.md` | Done |
| 14. RAG | `14_RAG.md` | Done |
| 15. AI Agents & Tool Use | `15_Agents_Tool_Use.md` | Done |
| 16. Prompt Engineering | `16_Prompt_Engineering.md` | Done |
| 17. ML System Design | `17_ML_System_Design.md` | Done |
| 18. Model Deployment & Serving | `18_Model_Deployment.md` | Done |
| 19. MLOps | `19_MLOps.md` | Done |
| 20. Databases & Infrastructure | `20_Databases_Infrastructure.md` | Done |
| 21. Computer Vision in Production | `21_Computer_Vision_Production.md` | Pending |
| 22. Recommendation Systems | `22_Recommendation_Systems.md` | Pending |
| 23. Time Series & Forecasting | `23_Time_Series.md` | Pending |
| 24. LLM Inference Optimization | `24_LLM_Inference_Optimization.md` | Pending |
| 25. Cloud & Infrastructure | `25_Cloud_Infrastructure.md` | Pending |
| 26. DSA for ML Interviews | `26_DSA_Interview.md` | Pending |
| 27. ML Interview Questions | `27_ML_Interview_Prep.md` | Pending |
| 28. Portfolio Projects | `28_Portfolio_Projects.md` | Pending |

---

## Suggested Study Order & Time Allocation

| Phase | Topics | Suggested Time | Priority |
|-------|--------|----------------|----------|
| A: Foundations | 1-2 | 1 week | Skim if strong, study if weak |
| B: Classical ML | 3-5 | 1.5 weeks | **Build from scratch -- don't just use sklearn** |
| C: Deep Learning | 6-9 | 2 weeks | **Critical -- implement everything in PyTorch** |
| D: NLP & Transformers | 10-12 | 1.5 weeks | **Critical -- transformers + fine-tuning** |
| E: LLMs & GenAI | 13-16 | 2 weeks | **Critical -- the most in-demand skills right now** |
| F: MLOps & Production | 17-20 | 2 weeks | **Critical -- what makes you an engineer, not a notebook user** |
| G: Specialization | 21-25 | 2 weeks | Pick 2-3 based on target role |
| H: Interview Prep | 26-28 | Ongoing | DSA + mock interviews + portfolio throughout |

**Total**: ~12-16 weeks at 3-4 hours/day

---

## Key Differences: This Track vs AI Scientist Track

| Dimension | AI/ML Engineer (This Track) | AI Scientist (Other Track) |
|-----------|---------------------------|---------------------------|
| **Focus** | Building & deploying ML systems | Understanding theory & research |
| **Code** | Heavy -- implement everything | Minimal -- theory-focused |
| **Math depth** | Enough to understand models | Deep -- proofs, derivations |
| **Publications** | Not required | Strongly valued |
| **Key skills** | PyTorch, APIs, Docker, Cloud, MLOps | Math, paper reading, novel methods |
| **Practice** | Projects, Kaggle, deployments | Problem sets, paper analysis |
| **Interview** | Coding + system design + ML | ML theory + research + math |
| **Target roles** | ML Engineer, AI Engineer, Data Scientist | Research Scientist, Applied Scientist |

---

## How to Use This With Me

1. Say: **"Generate notes for Topic X"** (e.g., "Generate notes for Topic 13: LLM APIs")
2. I'll create comprehensive, practice-heavy notes with:
   - Concepts explained clearly with code examples
   - Working implementations you can run
   - Practice exercises with checkboxes
   - A mini-project to build
   - Interview Q&A at the end of each topic
3. Study one topic per session, complete the practice exercises
4. **The practice is the most important part -- don't just read, BUILD.**

---

*Created: February 2026 | Target: AI/ML Engineer, AI Engineer, Data Scientist*
