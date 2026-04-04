# AI Interview Question Bank — Reverse Preparation

> **Purpose**: Self-testing question bank for AI Scientist / ML Engineer roles
> **Format**: Questions only (no answers). Test yourself, then refer to your study notes.
> **Difficulty**: `[E]` Easy (conceptual) · `[M]` Medium (analytical) · `[H]` Hard (system design / derivation)
> **Coverage**: 35 merged topics across GenAI Scientist + AI/ML Engineer tracks

---

## Topic 1: Probability, Statistics & Information Theory

1. `[E]` What is Bayes' theorem? Write it out and explain each term.
2. `[E]` What is the difference between a probability distribution and a probability density function?
3. `[E]` Name 5 common probability distributions and give a real-world use case for each.
4. `[E]` What is the difference between MLE and MAP estimation?
5. `[E]` What is entropy in information theory? What does high entropy mean?
6. `[M]` Derive the MLE estimate for the parameters of a Gaussian distribution.
7. `[M]` Explain the relationship between entropy, cross-entropy, and KL divergence. Write the formulas.
8. `[M]` Why is cross-entropy used as the loss function for language models? Connect it to MLE.
9. `[M]` What is KL divergence? Why is it asymmetric, and when does this asymmetry matter in ML?
10. `[M]` Explain mutual information. How is it related to entropy? Where is it used in ML?
11. `[M]` How would you design an A/B test to compare two LLM variants? What sample size formula would you use?
12. `[M]` What is the difference between Type I and Type II errors? How do they relate to precision and recall?
13. `[H]` Derive the evidence lower bound (ELBO) starting from the log-likelihood. Where does KL divergence appear?
14. `[H]` Explain why minimizing cross-entropy is equivalent to minimizing KL divergence when the target distribution is fixed.
15. `[H]` You're running an A/B test comparing two ranking models. After 2 weeks, Model B shows +2% improvement in click-through rate with p-value = 0.08. What do you do?
16. `[H]` Derive the Fisher information matrix for a Bernoulli distribution. How is it used in Elastic Weight Consolidation?
17. `[H]` What is the connection between entropy and the minimum description length principle? How does this relate to model selection?

---

## Topic 2: Linear Algebra & Optimization

1. `[E]` What is the difference between an eigenvalue and a singular value?
2. `[E]` What is PCA? What problem does it solve?
3. `[E]` Explain the difference between SGD, mini-batch gradient descent, and batch gradient descent.
4. `[E]` What is the role of the learning rate in gradient descent?
5. `[E]` What does it mean for a function to be convex? Why does convexity matter in optimization?
6. `[M]` Explain SVD (Singular Value Decomposition). How is it connected to PCA?
7. `[M]` Compare Adam, AdamW, and SGD with momentum. When would you use each?
8. `[M]` What is the role of weight decay? How does AdamW differ from Adam with L2 regularization?
9. `[M]` Explain cosine learning rate scheduling. Why is warmup used at the start of training?
10. `[M]` What are saddle points? Why are they a bigger problem than local minima in deep learning?
11. `[M]` Explain gradient clipping. When and why is it necessary?
12. `[M]` What is gradient accumulation? When would you use it?
13. `[H]` Explain the connection between low-rank matrix factorization and LoRA. Why does this work for adapting large models?
14. `[H]` Derive the gradient descent update for linear regression using matrix calculus. What is the normal equation?
15. `[H]` Why does the Adam optimizer maintain both first and second moment estimates? Derive the bias correction terms.
16. `[H]` Explain the loss landscape of deep neural networks. How do skip connections in ResNets change the loss landscape?
17. `[H]` What is the relationship between the condition number of the Hessian matrix and training difficulty? How do optimizers like Adam address ill-conditioning?

---

## Topic 3: Classical ML (Regression, Classification, Regularization)

1. `[E]` What is the bias-variance trade-off? How do you diagnose high bias vs high variance?
2. `[E]` What is the difference between L1 and L2 regularization? Which one leads to sparse solutions and why?
3. `[E]` Explain precision, recall, and F1-score. When would you optimize for each?
4. `[E]` What is the difference between a generative and a discriminative model? Give examples.
5. `[E]` What is cross-validation? Why use k-fold instead of a single train/test split?
6. `[E]` Explain logistic regression. What is the sigmoid function and why is it used?
7. `[M]` How does a Random Forest reduce variance compared to a single decision tree? What is bagging?
8. `[M]` Explain gradient boosting (XGBoost/LightGBM). How does it differ from Random Forest?
9. `[M]` You have a dataset with 95% class 0 and 5% class 1. What problems arise and how do you handle them?
10. `[M]` Explain the ROC curve and AUC. When is AUC misleading?
11. `[M]` What is feature importance in tree-based models? Compare permutation importance vs built-in importance.
12. `[M]` Derive the gradient descent update for logistic regression.
13. `[M]` What is the kernel trick in SVMs? Explain RBF kernel intuition.
14. `[H]` You're building a fraud detection model. Walk through your complete pipeline: data handling, model choice, metrics, deployment considerations.
15. `[H]` Compare XGBoost vs deep learning for tabular data. When does each win and why?
16. `[H]` Derive the closed-form solution for Ridge regression. How does the regularization term change the eigenvalues of the design matrix?
17. `[H]` What is calibration in classification? How do you check if a model's predicted probabilities are well-calibrated? How do you fix poor calibration?
18. `[H]` Explain the relationship between maximum margin classification (SVM) and regularization. Why does maximizing the margin reduce overfitting?

---

## Topic 4: Unsupervised Learning & Dimensionality Reduction

1. `[E]` What is K-Means clustering? What are its assumptions?
2. `[E]` What is the difference between PCA and t-SNE?
3. `[E]` What is the elbow method for choosing K in K-Means?
4. `[E]` What is the difference between hard and soft clustering?
5. `[M]` How does DBSCAN work? What are its advantages over K-Means?
6. `[M]` Explain UMAP. How does it differ from t-SNE in terms of global structure preservation?
7. `[M]` What is the silhouette score? How do you use it to evaluate clustering quality?
8. `[M]` Explain Gaussian Mixture Models (GMMs). How does EM algorithm fit them?
9. `[M]` When would you use PCA vs autoencoders for dimensionality reduction?
10. `[M]` What is anomaly detection? Compare isolation forest vs autoencoders for anomaly detection.
11. `[H]` You have 10 million user embeddings and need to cluster them. K-Means is too slow. What alternatives do you have?
12. `[H]` Explain the connection between PCA and the eigendecomposition of the covariance matrix. Derive it.
13. `[H]` How would you evaluate the quality of embeddings produced by an LLM? What clustering or retrieval-based metrics would you use?
14. `[H]` Explain hierarchical clustering (agglomerative vs divisive). When would you prefer it over K-Means?
15. `[H]` You need to reduce a 768-dimensional embedding to 128 dimensions for a production system. Compare PCA, random projection, and learned projections. What are the trade-offs?

---

## Topic 5: Deep Learning Foundations (MLPs, Backprop, Activations)

1. `[E]` What is backpropagation? Explain it in terms of the chain rule.
2. `[E]` What is the vanishing gradient problem? Which activations cause it?
3. `[E]` Compare ReLU, GELU, and SwiGLU activation functions.
4. `[E]` What is dropout and why does it help with regularization?
5. `[E]` What is the difference between batch normalization and layer normalization?
6. `[E]` What is a residual connection? Why is it important for deep networks?
7. `[M]` Explain the Xavier/Glorot and He initialization strategies. When do you use each?
8. `[M]` Why has layer normalization replaced batch normalization in transformers?
9. `[M]` Derive the backpropagation equations for a two-layer MLP with ReLU activation.
10. `[M]` What is the dying ReLU problem? How do Leaky ReLU and GELU address it?
11. `[M]` Explain the lottery ticket hypothesis. What are its implications for model pruning?
12. `[M]` What is RMSNorm and why do modern LLMs (Llama) use it instead of LayerNorm?
13. `[H]` Why do residual connections help gradient flow? Explain mathematically how they affect the Jacobian of the network.
14. `[H]` Implement the forward and backward pass of a linear layer from scratch. Show the gradient computation.
15. `[H]` What is the relationship between the width of a neural network and its expressive power? Explain the universal approximation theorem.
16. `[H]` Explain why SwiGLU works better than standard FFN in transformers. What is the gating mechanism doing?
17. `[H]` Compare Pre-LN vs Post-LN transformer architectures. Why did the field move to Pre-LN? What is the training stability trade-off?

---

## Topic 6: Python & PyTorch for ML

1. `[E]` What is the difference between a Python list and a NumPy array?
2. `[E]` Explain Python generators. Why are they useful for ML data loading?
3. `[E]` What is `torch.no_grad()` and when do you use it?
4. `[E]` What is the difference between `model.eval()` and `model.train()` in PyTorch?
5. `[E]` What is a PyTorch DataLoader? What parameters does it take?
6. `[M]` Explain the PyTorch autograd system. What is the computation graph?
7. `[M]` How do you implement a custom `nn.Module` in PyTorch? What methods must you define?
8. `[M]` What is mixed precision training in PyTorch? How does `torch.cuda.amp` work?
9. `[M]` Explain `DistributedDataParallel` (DDP). How does it differ from `DataParallel`?
10. `[M]` How would you profile a PyTorch model to find performance bottlenecks?
11. `[M]` What are Python decorators? Implement a timing decorator that measures function execution time.
12. `[M]` Explain the difference between `deepcopy` and regular assignment for PyTorch model weights.
13. `[H]` Implement a complete training loop in PyTorch with mixed precision, gradient accumulation, and gradient clipping.
14. `[H]` How would you implement a custom learning rate scheduler that does warmup followed by cosine decay?
15. `[H]` What is `torch.compile`? How does it speed up training? What are its limitations?
16. `[H]` Implement a custom PyTorch Dataset for variable-length text sequences with dynamic batching and padding.
17. `[H]` Explain how PyTorch handles memory on GPU. What causes OOM errors and how do you diagnose/fix them?

---

## Topic 7: CNNs & Computer Vision

1. `[E]` What is a convolution operation? What are filters/kernels?
2. `[E]` What is the difference between stride and padding in a convolutional layer?
3. `[E]` What is pooling? Compare max pooling and average pooling.
4. `[E]` What is transfer learning in computer vision? Why does it work?
5. `[M]` Explain the ResNet architecture. Why do skip connections enable training of very deep networks?
6. `[M]` How do you calculate the output size of a convolutional layer given input size, kernel size, stride, and padding?
7. `[M]` What is data augmentation? List 5 common augmentation techniques for image classification.
8. `[M]` Explain the difference between object detection (YOLO) and semantic segmentation (U-Net).
9. `[M]` What is batch normalization in CNNs? How does it affect training dynamics?
10. `[M]` Compare EfficientNet and ResNet architectures. What is compound scaling?
11. `[H]` Design a production image classification pipeline. Cover data collection, augmentation, model selection, training, evaluation, and deployment.
12. `[H]` How does YOLO achieve real-time object detection? Explain the architecture and loss function.
13. `[H]` What is the relationship between receptive field size and network depth? How do dilated convolutions change this?
14. `[H]` Compare CNN-based approaches vs Vision Transformers (ViT) for image classification. When does each win?
15. `[H]` Explain how CLIP connects vision and language. What is contrastive learning and how does it work in CLIP's training?
16. `[H]` You need to deploy an image classification model that runs at 30 FPS on a mobile device. Walk through your optimization pipeline (quantization, pruning, ONNX export, etc.).

---

## Topic 8: RNNs, LSTMs & Sequence Models

1. `[E]` What is the basic RNN architecture? Draw the unrolled computation graph.
2. `[E]` What is the vanishing gradient problem in RNNs?
3. `[E]` What are the three gates in an LSTM and what does each do?
4. `[E]` What is the difference between an LSTM and a GRU?
5. `[M]` How does the LSTM cell state solve the vanishing gradient problem? Explain mathematically.
6. `[M]` What is the Seq2Seq architecture? How is it used for machine translation?
7. `[M]` What is teacher forcing? What problem does it cause (exposure bias)?
8. `[M]` Explain bidirectional RNNs. When are they applicable and when are they not?
9. `[M]` How do you handle variable-length sequences in RNNs? What is packing/padding?
10. `[H]` Why did transformers replace RNNs for most NLP tasks? Compare computational complexity, parallelization, and long-range dependencies.
11. `[H]` Derive the LSTM forward equations. Show how gradients flow through the cell state.
12. `[H]` What is attention in the context of Seq2Seq? How does Bahdanau attention work?
13. `[H]` You need to build a real-time speech-to-text system. Would you use an RNN, Transformer, or hybrid? Justify your choice.
14. `[H]` Explain the CTC (Connectionist Temporal Classification) loss. Where is it used?
15. `[H]` Compare RNNs, 1D CNNs, and Transformers for time series forecasting. What are the trade-offs?

---

## Topic 9: Tokenization & Text Preprocessing

1. `[E]` What is tokenization? Why can't we just split on spaces?
2. `[E]` What is the difference between word-level, character-level, and subword tokenization?
3. `[E]` What is BPE (Byte-Pair Encoding)? Explain the algorithm.
4. `[E]` What is a vocabulary? What happens when a word is not in the vocabulary (OOV problem)?
5. `[M]` Compare BPE, WordPiece, and Unigram tokenization. What are the differences?
6. `[M]` What is SentencePiece? How does it differ from HuggingFace tokenizers?
7. `[M]` What is byte-level BPE? Why do GPT-2 and Llama use it?
8. `[M]` How do you train a custom tokenizer? When would you need to?
9. `[M]` What is tokenizer fertility? Why does vocabulary size matter for multilingual models?
10. `[M]` Explain text preprocessing steps for NLP: lowercasing, stemming, lemmatization, stop word removal. When are these harmful?
11. `[H]` You're building a multilingual model. Llama's tokenizer produces 3.8 tokens per Chinese word but 1.3 per English word. How do you fix this?
12. `[H]` Explain the Unigram tokenization algorithm. How does it use the EM algorithm?
13. `[H]` What is the trade-off between vocabulary size and sequence length? Derive the relationship between vocab size, model parameters, and inference cost.
14. `[H]` How would you design a tokenizer for a code-generation model? What special considerations apply?
15. `[H]` Compare the tokenizers of GPT-4, Llama 3, and Claude. How do their vocabulary sizes and strategies differ?

---

## Topic 10: Word Embeddings (Word2Vec, GloVe, Sentence Embeddings)

1. `[E]` What is a word embedding? Why are they better than one-hot encoding?
2. `[E]` Explain the distributional hypothesis: "You shall know a word by the company it keeps."
3. `[E]` What is the difference between Word2Vec CBOW and Skip-gram?
4. `[E]` What is TF-IDF? What does each component measure?
5. `[M]` How does Word2Vec learn word embeddings? Explain negative sampling.
6. `[M]` What is GloVe? How does it combine global co-occurrence statistics with local context windows?
7. `[M]` What is FastText? How does it handle out-of-vocabulary words?
8. `[M]` Explain Sentence-BERT. How does it produce sentence-level embeddings?
9. `[M]` How do you evaluate embedding quality? Explain intrinsic (analogy tasks) vs extrinsic evaluation.
10. `[M]` What is the difference between static embeddings (Word2Vec) and contextual embeddings (BERT)?
11. `[H]` Derive the Skip-gram objective function with negative sampling.
12. `[H]` Why do word embeddings exhibit linear algebraic properties (e.g., king - man + woman ≈ queen)?
13. `[H]` Compare embedding models for RAG: OpenAI text-embedding-3-small vs all-MiniLM-L6-v2 vs BGE-large. What are the trade-offs?
14. `[H]` How would you fine-tune an embedding model for a specific domain (e.g., legal documents)? What is contrastive fine-tuning?
15. `[H]` What is the relationship between Word2Vec and matrix factorization (Levy & Goldberg 2014)?

---

## Topic 11: Attention Mechanisms

1. `[E]` What is attention in neural networks? What problem does it solve?
2. `[E]` What is self-attention? How does it differ from cross-attention?
3. `[E]` What are Query, Key, and Value in attention? Give an analogy.
4. `[E]` Why do we scale by $\sqrt{d_k}$ in scaled dot-product attention?
5. `[E]` What is multi-head attention? Why use multiple heads instead of a single large attention?
6. `[M]` Write out the full scaled dot-product attention formula. What is the shape of each tensor?
7. `[M]` What is a causal attention mask? Why is it needed for autoregressive models?
8. `[M]` Explain the difference between Bahdanau (additive) and Luong (multiplicative) attention.
9. `[M]` What is the computational complexity of self-attention? Why is it O(n²)?
10. `[M]` How does Grouped-Query Attention (GQA) reduce memory and compute compared to Multi-Head Attention?
11. `[M]` What is Multi-Query Attention (MQA)? How does it compare to GQA?
12. `[H]` Why does attention work as a "soft lookup table"? Explain the information-theoretic interpretation.
13. `[H]` Derive the gradients of the attention mechanism. How do gradients flow through the softmax?
14. `[H]` What is Flash Attention? How does it reduce memory from O(n²) to O(n) without approximation?
15. `[H]` Explain the connection between attention and kernel methods. What is linear attention?
16. `[H]` Design an attention variant for a model that needs to process 1M token sequences. What approximations or architectural changes would you make?
17. `[H]` What are attention sinks? Why do the first few tokens in autoregressive models receive disproportionately high attention?

---

## Topic 12: Transformer Architecture

1. `[E]` Draw the transformer encoder and decoder architecture. Label all components.
2. `[E]` What is the role of the feed-forward network (FFN) in each transformer layer?
3. `[E]` What is the purpose of residual connections in transformers?
4. `[E]` What is the difference between a transformer encoder and a transformer decoder?
5. `[M]` How do you count the parameters in a transformer model? Walk through each component.
6. `[M]` What is Pre-LN vs Post-LN? Why did the field move to Pre-LN?
7. `[M]` Explain the SwiGLU FFN used in modern transformers (Llama, PaLM). How does it differ from standard ReLU FFN?
8. `[M]` What is the KV cache? How does it speed up autoregressive generation?
9. `[M]` Calculate the number of parameters in a transformer with d=4096, L=32, H=32, V=32000.
10. `[M]` How does the context window length affect transformer performance and cost?
11. `[H]` Why do transformers need positional information? What happens if you remove positional encodings entirely?
12. `[H]` Explain the relationship between transformer depth (number of layers) and width (hidden dimension). How do scaling laws inform this trade-off?
13. `[H]` Compare the architectural choices of GPT-4, Llama 3, and Mistral. What are the key differences (attention type, FFN, normalization)?
14. `[H]` What is Mixture of Experts (MoE)? How does Mixtral use it? What are the routing challenges?
15. `[H]` Explain why transformers are more parallelizable than RNNs. What is the training time complexity?
16. `[H]` Design a transformer architecture for processing documents of 100K tokens. Address attention complexity, memory, and positional encoding.

---

## Topic 13: Positional Encodings

1. `[E]` Why do transformers need positional encodings? What information would be lost without them?
2. `[E]` What are sinusoidal positional encodings? Why did the original transformer use them?
3. `[E]` What is the difference between absolute and relative positional encodings?
4. `[M]` Explain Rotary Position Embeddings (RoPE). How do they encode position in the attention computation?
5. `[M]` What is ALiBi (Attention with Linear Biases)? How does it encode position?
6. `[M]` Compare learned positional embeddings vs sinusoidal. When does each work better?
7. `[M]` What is the length generalization problem? Why do models trained on 4K tokens struggle at 32K tokens?
8. `[M]` How does RoPE enable length extrapolation better than absolute positional encodings?
9. `[H]` Derive the RoPE rotation matrix. Show how it encodes relative positions in the dot product.
10. `[H]` What is NTK-aware scaling for RoPE? How does it extend context length?
11. `[H]` Explain YaRN (Yet another RoPE extensioN). How does it improve on NTK-aware scaling?
12. `[H]` Compare RoPE, ALiBi, and learned embeddings in terms of: length generalization, training cost, and downstream performance.
13. `[H]` You need to extend a model trained with 8K context to 128K context. What positional encoding changes and training strategies would you use?
14. `[H]` How do positional encodings interact with the KV cache during generation? What inefficiencies arise?
15. `[H]` What is the theoretical basis for sinusoidal encodings? How do they relate to Fourier features?

---

## Topic 14: BERT & Encoder Models

1. `[E]` What is BERT? What pretraining objectives does it use?
2. `[E]` What is Masked Language Modeling (MLM)? How does the 80/10/10 masking strategy work?
3. `[E]` What is the `[CLS]` token? How is it used for classification tasks?
4. `[E]` What is the difference between BERT-base and BERT-large?
5. `[M]` How does RoBERTa improve upon BERT? List the key changes.
6. `[M]` What is DeBERTa? What is disentangled attention?
7. `[M]` What is ELECTRA? How does its replaced token detection objective differ from MLM?
8. `[M]` Explain ALBERT's parameter reduction techniques (cross-layer sharing, factorized embedding).
9. `[M]` How do you fine-tune BERT for text classification? Draw the architecture.
10. `[M]` What is Next Sentence Prediction (NSP)? Why did RoBERTa remove it?
11. `[H]` Why can't BERT be used for text generation? What architectural constraints prevent it?
12. `[H]` Compare MLM vs CLM for learning representations. When is bidirectional context helpful and when is it harmful?
13. `[H]` You need to build a semantic search system. Would you use BERT, Sentence-BERT, or a decoder model? Explain your reasoning.
14. `[H]` How does BERT's attention pattern differ from GPT's? Visualize the difference in attention masks.
15. `[H]` What is the train-test mismatch in MLM (the `[MASK]` token never appears at inference)? How significant is this problem?
16. `[H]` Explain knowledge distillation from BERT. How does DistilBERT achieve 97% of BERT's performance with 60% fewer parameters?

---

## Topic 15: GPT, Decoder Models & Open-Source LLMs

1. `[E]` What is the GPT architecture? How does it differ from BERT?
2. `[E]` What is causal language modeling (next-token prediction)?
3. `[E]` What is the difference between GPT-3 and GPT-4?
4. `[E]` Name 5 open-source LLMs and their organizations.
5. `[M]` Explain the scaling laws for LLMs (Kaplan et al. and Chinchilla). What is the key insight?
6. `[M]` Compare Llama 2 and Llama 3. What architectural and training improvements were made?
7. `[M]` What is Mistral? What is sliding window attention?
8. `[M]` What is Mixtral (Mixture of Experts)? How does the routing mechanism work?
9. `[M]` Explain the concept of emergent abilities in LLMs. Give examples.
10. `[M]` What is the Chinchilla-optimal compute allocation? Why were earlier models like GPT-3 considered undertrained?
11. `[H]` Compare GPT-4, Claude, Llama 3, and Gemini in terms of architecture, training data, and capabilities. What can you infer about their differences?
12. `[H]` Why did the field move from large undertrained models (GPT-3: 175B params, 300B tokens) to smaller overtrained models (Llama 3: 70B params, 15T tokens)?
13. `[H]` Explain the Phi series of small language models. How do they achieve strong performance at 1-3B parameters?
14. `[H]` What are the trade-offs between open-source and closed-source LLMs for production use? Consider quality, cost, privacy, and customization.
15. `[H]` Design a benchmark suite to evaluate a new LLM. What capabilities would you test and what metrics would you use?
16. `[H]` Explain how Grouped-Query Attention (used in Llama 2/3) affects the compute-memory trade-off during inference.

---

## Topic 16: Encoder-Decoder Models (T5, BART, UL2)

1. `[E]` What is the encoder-decoder architecture? How do encoder and decoder interact?
2. `[E]` What is T5? What is its "text-to-text" framework?
3. `[E]` What is span corruption (T5's pretraining objective)?
4. `[E]` What is BART? How does it combine BERT-like and GPT-like ideas?
5. `[M]` How does cross-attention work in encoder-decoder models? Which components come from the encoder vs decoder?
6. `[M]` What is prefix LM? How does it differ from full encoder-decoder and pure decoder architectures?
7. `[M]` Compare T5, BART, and GPT for summarization. What are the strengths of each?
8. `[M]` What is UL2? How does its Mixture of Denoisers (R, S, X denoisers) unify different pretraining objectives?
9. `[M]` What is FLAN-T5? How does instruction tuning improve T5?
10. `[H]` Why has the field moved away from encoder-decoder architectures toward decoder-only? What was lost and what was gained?
11. `[H]` Compare the training efficiency of MLM (BERT), span corruption (T5), and CLM (GPT). Which makes best use of each token?
12. `[H]` How does mT5 handle multilingual tasks? What is the "curse of multilinguality"?
13. `[H]` Design a seq-to-seq system for translating code between programming languages. What architecture would you choose and why?
14. `[H]` What is the difference between "stuff", "map-reduce", and "refine" strategies for processing long documents in seq-to-seq models?
15. `[H]` Explain the architectural connections between encoder-decoder models, prefix LMs, and the UL2 unification.

---

## Topic 17: Pretraining LLMs (Data, Distributed Training, Scaling)

1. `[E]` What are the two phases of training an LLM (pretraining vs fine-tuning)?
2. `[E]` Name 5 common data sources for LLM pretraining.
3. `[E]` What is data parallelism in distributed training?
4. `[E]` What is BF16? Why is it used instead of FP32 for training?
5. `[M]` Explain the difference between data parallelism, tensor parallelism, and pipeline parallelism.
6. `[M]` What is the formula for estimating FLOPs needed to train a model? Calculate for a 70B model on 2T tokens.
7. `[M]` Why is BF16 preferred over FP16 for LLM training? What problem does FP16 have?
8. `[M]` What is data deduplication? Why is it important for pretraining? Explain MinHash.
9. `[M]` What are data mixing ratios? How does changing the proportion of code vs web data affect model capabilities?
10. `[M]` What is gradient accumulation and why is it used in LLM training?
11. `[H]` Explain ZeRO stages 1, 2, and 3. How much memory does each save?
12. `[H]` What is the pipeline parallelism bubble problem? How do you calculate the bubble fraction?
13. `[H]` Derive the Chinchilla scaling laws. What is the optimal ratio of parameters to tokens?
14. `[H]` Compare the training recipes of GPT-3, Llama 2, and Llama 3. What evolved and why?
15. `[H]` You're training a 13B parameter model. You have 64 H100 GPUs across 8 nodes. Design the parallelism strategy (TP, PP, DP).
16. `[H]` What is Model FLOPs Utilization (MFU)? Why is it hard to achieve > 50% MFU?
17. `[H]` Explain how data quality filtering works at scale. Compare heuristic filters, classifier-based filters, and the FineWeb approach.
18. `[H]` What is curriculum learning / data annealing in LLM training? How does Llama 3 use it?

---

## Topic 18: Fine-Tuning & PEFT (LoRA, QLoRA, Adapters, DoRA)

1. `[E]` What is the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT)?
2. `[E]` What is LoRA? Explain the core idea in one sentence.
3. `[E]` What is catastrophic forgetting?
4. `[E]` What is instruction tuning?
5. `[M]` Derive the LoRA update rule: $W = W_0 + BA$. How is it initialized?
6. `[M]` What is the α/r scaling factor in LoRA? Why is it needed?
7. `[M]` How does QLoRA achieve fine-tuning on a single GPU? What are the three innovations?
8. `[M]` Compare LoRA, prefix tuning, and prompt tuning. When would you use each?
9. `[M]` Why does low-rank adaptation work for fine-tuning? Explain the intrinsic dimensionality hypothesis.
10. `[M]` How do you choose the LoRA rank? What are the trade-offs of higher vs lower rank?
11. `[M]` Which layers should you apply LoRA to? What does the research say?
12. `[H]` What is DoRA (Weight-Decomposed Low-Rank Adaptation)? How does it improve over LoRA?
13. `[H]` What is NF4 (NormalFloat)? Why is it better than standard INT4 for quantizing LLM weights?
14. `[H]` Compare LoRA quality vs full fine-tuning quality. When is the gap largest?
15. `[H]` How do you merge LoRA weights at inference time? What are the implications for multi-task serving?
16. `[H]` A client has 500 labeled examples. Should they fine-tune, use few-shot prompting, or build RAG? Walk through the decision framework.
17. `[H]` Explain AdaLoRA. How does it adaptively allocate rank across different weight matrices?
18. `[H]` Compare instruction tuning datasets: FLAN, Alpaca, Orca, OpenHermes. How does dataset quality affect the fine-tuned model?

---

## Topic 19: Alignment (RLHF, DPO, Constitutional AI)

1. `[E]` What does "alignment" mean for LLMs? What are the three H's?
2. `[E]` What is RLHF? List the three steps of the RLHF pipeline.
3. `[E]` What is a reward model? What data is it trained on?
4. `[E]` What is DPO (Direct Preference Optimization)? How does it simplify RLHF?
5. `[M]` Write the RLHF objective: $\max E[R(x,y)] - \beta \cdot KL(\pi_\theta \| \pi_{ref})$. Explain each term.
6. `[M]` What is reward hacking? Give an example and explain how KL regularization mitigates it.
7. `[M]` Explain the PPO algorithm at a high level. Why is it used for RLHF instead of simpler RL algorithms?
8. `[M]` Derive the DPO loss function. How does it eliminate the reward model?
9. `[M]` What is Constitutional AI (Anthropic)? How does it differ from standard RLHF?
10. `[M]` What is the difference between SFT (Supervised Fine-Tuning) and alignment? Why are both needed?
11. `[H]` What is the mathematical relationship between the RLHF objective and DPO? How does DPO derive an implicit reward from the policy?
12. `[H]` Compare RLHF, DPO, IPO, and KTO. What are the trade-offs of each alignment method?
13. `[H]` What is RLAIF (RL from AI Feedback)? How does it reduce the cost of human annotation?
14. `[H]` You're aligning a customer-facing chatbot. Design the full pipeline: data collection (preference pairs), reward modeling, and optimization. What pitfalls do you watch for?
15. `[H]` What is the "alignment tax"? Does alignment make models less capable? What does the research show?
16. `[H]` Explain Anthropic's iterative RLHF approach. How does red-teaming fit into the alignment pipeline?
17. `[H]` What is the relationship between RLHF and the Bradley-Terry model of human preferences?

---

## Topic 20: Decoding Strategies & Text Generation

1. `[E]` What is greedy decoding? What is its main limitation?
2. `[E]` What is temperature in LLM sampling? What happens at T=0 vs T=1 vs T>1?
3. `[E]` What is top-k sampling?
4. `[E]` What is nucleus (top-p) sampling?
5. `[M]` Compare greedy, beam search, top-k, and top-p decoding. When would you use each?
6. `[M]` What is the repetition penalty? How is it implemented?
7. `[M]` What is the KV cache? How does it reduce the computational cost of autoregressive generation?
8. `[M]` What is streaming in LLM APIs? How is it implemented (SSE)?
9. `[M]` Explain the trade-off between diversity and quality in text generation. How do you tune it?
10. `[H]` What is speculative decoding? How does it use a small model to speed up a large model?
11. `[H]` What is contrastive decoding? How does it improve generation quality?
12. `[H]` Explain how beam search handles the length bias problem. What is length normalization?
13. `[H]` What is guided/constrained generation? How do you force an LLM to output valid JSON?
14. `[H]` Design a decoding strategy for a code-generation model. What properties should it optimize for (vs a chatbot)?
15. `[H]` What is the relationship between perplexity and generation quality? Can a model with lower perplexity generate worse text?
16. `[H]` Explain min-p sampling. How does it adapt the sampling threshold based on the model's confidence?

---

## Topic 21: Prompt Engineering & Advanced Techniques

1. `[E]` What is the difference between zero-shot, one-shot, and few-shot prompting?
2. `[E]` What is chain-of-thought (CoT) prompting? Why does adding "Let's think step by step" help?
3. `[E]` What is a system prompt? How does it affect model behavior?
4. `[E]` What is structured output (JSON mode)? Why is it useful for production systems?
5. `[M]` Compare CoT, self-consistency, and tree-of-thought. When would you use each?
6. `[M]` What is ReAct prompting? How does it combine reasoning and acting?
7. `[M]` How does the Instructor library work with Pydantic for structured extraction?
8. `[M]` What is DSPy? How does it programmatically optimize prompts?
9. `[M]` Explain prompt injection. Give 3 examples of injection attacks.
10. `[M]` How do you build a multi-layer defense against prompt injection?
11. `[H]` Implement a self-consistency pipeline: sample N reasoning paths, extract answers, majority vote. When does this improve accuracy?
12. `[H]` Compare DSPy vs manual prompt engineering. When is programmatic optimization worth the setup cost?
13. `[H]` Design a guardrails system for a customer-facing LLM chatbot (input validation, output validation, content policy).
14. `[H]` What is the difference between OpenAI's `response_format: json_object` and Structured Outputs (`json_schema`)? When would you use each?
15. `[H]` You're building a data extraction pipeline that works 95% of the time. How do you get to 99%? Walk through your debugging and improvement process.
16. `[H]` What are the theoretical reasons that in-context learning works? Is the model doing gradient descent, Bayesian inference, or something else?
17. `[H]` Design a prompt management system for a team of 10 ML engineers. How do you version, test, and deploy prompts?

---

## Topic 22: In-Context Learning (Theory)

1. `[E]` What is in-context learning (ICL)? How does it differ from fine-tuning?
2. `[E]` What is a demonstration in ICL? What does it consist of?
3. `[M]` Does ICL learn new tasks or just activate existing capabilities? What does the research say?
4. `[M]` How does the order of few-shot examples affect ICL performance?
5. `[M]` What is the role of the label space in ICL? Does the model use the actual labels or just the format?
6. `[M]` Explain the hypothesis that ICL performs implicit gradient descent (Akyürek et al., 2022).
7. `[M]` What is the Bayesian interpretation of ICL (Xie et al., 2022)?
8. `[H]` How do transformers implement ICL mechanistically? What do the attention heads learn?
9. `[H]` What are "induction heads"? How do they enable ICL?
10. `[H]` Does ICL performance correlate with model scale? What are emergent abilities in the context of ICL?
11. `[H]` Compare ICL with fine-tuning on the same number of examples. Under what conditions does each win?
12. `[H]` What is the connection between ICL and meta-learning? How does pretraining create a meta-learner?
13. `[H]` Design an experiment to test whether ICL is doing task identification vs task learning. How would you distinguish the two?
14. `[H]` How does the pretraining distribution affect ICL capability? Why do some models exhibit stronger ICL than others?
15. `[H]` What is the role of the attention mechanism in enabling ICL? Could an RNN exhibit similar in-context learning?

---

## Topic 23: LLM APIs & Function Calling

1. `[E]` What are the key parameters of an LLM API call (model, temperature, max_tokens, etc.)?
2. `[E]` What is function calling / tool use in LLM APIs?
3. `[E]` What is streaming? Why is it important for user experience?
4. `[E]` What is the difference between the system, user, and assistant roles in chat APIs?
5. `[M]` How do you estimate the cost of an LLM API call? What is the difference between input and output token pricing?
6. `[M]` How does OpenAI's function calling work? Walk through the schema definition, model response, and function execution.
7. `[M]` What is parallel function calling? When does the model invoke multiple tools simultaneously?
8. `[M]` Compare the OpenAI, Anthropic, and Google Gemini APIs. What are the key differences?
9. `[M]` How do you run open-source models locally using Ollama or vLLM?
10. `[H]` Design a cost-optimization strategy for an LLM-powered application processing 1M requests/day. Consider model routing, caching, and batching.
11. `[H]` How do you implement rate limiting, retry logic, and fallback for production LLM API usage?
12. `[H]` Compare hosted APIs vs self-hosted open-source models for a healthcare application. Consider cost, latency, privacy, and quality.
13. `[H]` What is semantic caching for LLM APIs? How does it differ from exact-match caching?
14. `[H]` Implement a model router that sends simple queries to a small model and complex queries to a large model. How do you classify query complexity?
15. `[H]` Design an LLM gateway/proxy that handles authentication, rate limiting, logging, cost tracking, and model routing.

---

## Topic 24: RAG (Retrieval-Augmented Generation)

1. `[E]` What is RAG? Why is it needed?
2. `[E]` What are the two phases of a RAG pipeline (indexing and querying)?
3. `[E]` What is a vector database? Name 3 examples.
4. `[E]` What is cosine similarity? Why is it used for comparing embeddings?
5. `[E]` What is chunking? Why can't you just embed entire documents?
6. `[M]` Compare fixed-size, recursive, and semantic chunking strategies. What are the trade-offs?
7. `[M]` What is hybrid search (BM25 + dense)? Why is it better than dense-only retrieval?
8. `[M]` What is a cross-encoder re-ranker? Why use a two-stage pipeline (bi-encoder → cross-encoder)?
9. `[M]` What is MMR (Maximal Marginal Relevance)? What problem does it solve?
10. `[M]` How do you choose chunk size? What are the trade-offs of small vs large chunks?
11. `[M]` What is Reciprocal Rank Fusion (RRF)? Why is it better than raw score combination for hybrid search?
12. `[H]` Your RAG system returns irrelevant documents. Walk through your systematic debugging process.
13. `[H]` What is parent-child chunking? How does it solve the chunk-size dilemma?
14. `[H]` Design a production RAG system for a company with 100K internal documents. Cover ingestion, retrieval, generation, and evaluation.
15. `[H]` How do you evaluate a RAG system? Explain retrieval metrics (Recall@K, MRR, NDCG) and generation metrics (faithfulness, relevancy).
16. `[H]` What are the failure modes of RAG? List 5 and explain how to address each.
17. `[H]` Compare RAG vs fine-tuning vs long-context models. When does each win?
18. `[H]` What is RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)? How does it improve over flat RAG?

---

## Topic 25: AI Agents & Tool Use

1. `[E]` What is an AI agent? How does it differ from a simple LLM call?
2. `[E]` What are the four components of an agent (LLM + tools + memory + loop)?
3. `[E]` What is the ReAct pattern? How does it alternate between reasoning and acting?
4. `[E]` What is a tool/function schema? What information does it contain?
5. `[M]` How does an agent decide which tool to use? Explain the role of tool descriptions.
6. `[M]` What are the different types of agent memory (short-term, long-term, working memory)?
7. `[M]` What is LangGraph? How does it enable stateful, multi-step agents?
8. `[M]` What is the plan-and-execute agent pattern? How does it differ from ReAct?
9. `[M]` How do you handle tool call errors and retries in an agent loop?
10. `[M]` What is a multi-agent system? When would you use multiple agents instead of one?
11. `[H]` Design an agent that can research a topic, synthesize information from multiple sources, and write a report. What tools and architecture would you use?
12. `[H]` What is Reflexion? How does self-reflection improve agent performance?
13. `[H]` How do you evaluate agent performance? What metrics capture task completion, efficiency, and reliability?
14. `[H]` What are the reliability challenges of agents? How do you prevent infinite loops, budget overruns, and hallucinated tool calls?
15. `[H]` Compare LangChain agents, LangGraph, CrewAI, and AutoGen. When would you use each framework?
16. `[H]` Design a multi-agent system for code review: one agent reads the PR, one checks for bugs, one checks style, and a coordinator synthesizes feedback.
17. `[H]` What is the theoretical limit of agent capabilities? How does the agent's performance scale with the underlying LLM's capabilities?

---

## Topic 26: Multimodal AI (Vision-Language, Diffusion)

1. `[E]` What is multimodal AI? Give 3 examples of multimodal tasks.
2. `[E]` What is CLIP? How does it connect images and text?
3. `[E]` What is a diffusion model? Explain the forward and reverse process at a high level.
4. `[E]` What is GPT-4V / GPT-4o? How does it handle images?
5. `[M]` Explain contrastive learning in CLIP. What is the InfoNCE loss?
6. `[M]` How does LLaVA (Large Language and Vision Assistant) work? How is it trained?
7. `[M]` What is Stable Diffusion? Explain the latent diffusion architecture.
8. `[M]` What is a vision encoder (ViT)? How does it convert images to tokens?
9. `[M]` Compare vision-language models: CLIP, BLIP-2, LLaVA, Flamingo. What are the architectural differences?
10. `[M]` What is text-to-image generation? How does the model condition on text prompts?
11. `[H]` Derive the diffusion model objective (denoising score matching). What is the connection to the ELBO?
12. `[H]` How does classifier-free guidance work in diffusion models? Why does it improve image quality?
13. `[H]` Design a multimodal RAG system that can answer questions about documents containing text, tables, and images.
14. `[H]` What is Whisper? How does it handle multilingual speech recognition?
15. `[H]` Compare early fusion vs late fusion for multimodal models. What are the trade-offs?
16. `[H]` What are the key challenges in training vision-language models? Address data alignment, training instability, and hallucination.
17. `[H]` How would you build a production image search system using CLIP embeddings? Address indexing, latency, and relevance.

---

## Topic 27: LLM Inference Optimization (Quantization, KV Cache, Flash Attention)

1. `[E]` Why is LLM inference expensive? What are the main bottlenecks?
2. `[E]` What is quantization? What does it mean to go from FP16 to INT4?
3. `[E]` What is the KV cache and why is it needed for autoregressive generation?
4. `[E]` What is the difference between the prefill and decode phases of LLM inference?
5. `[M]` Compare GPTQ, AWQ, and GGUF quantization methods. What are the trade-offs?
6. `[M]` What is Flash Attention? How does it reduce memory usage without approximation?
7. `[M]` What is continuous batching? Why is it important for serving multiple users?
8. `[M]` What is vLLM? What is PagedAttention?
9. `[M]` Explain the difference between weight-only quantization and activation quantization.
10. `[M]` What is knowledge distillation? How can you use a large model to train a smaller one?
11. `[H]` What is speculative decoding? How does it use a draft model to speed up generation? What determines the speedup?
12. `[H]` You need to serve a 70B model to 100 concurrent users with < 200ms time-to-first-token. Design the serving infrastructure.
13. `[H]` Explain the memory-bound vs compute-bound distinction in transformer inference. How does batch size affect this?
14. `[H]` What is model pruning? Compare structured vs unstructured pruning. How much can you prune without quality loss?
15. `[H]` Compare serving frameworks: vLLM, TGI (Text Generation Inference), TensorRT-LLM, and Triton. When would you use each?
16. `[H]` What is tensor parallelism during inference? How does it differ from tensor parallelism during training?
17. `[H]` Calculate the memory requirements for serving a 70B model with KV cache for batch size 32 and sequence length 4096.

---

## Topic 28: Evaluation & Benchmarking

1. `[E]` What is perplexity? How do you interpret it?
2. `[E]` What are BLEU and ROUGE scores? When are they used?
3. `[E]` Name 5 popular LLM benchmarks and what they measure.
4. `[E]` What is LLM-as-a-judge evaluation? Why is it used?
5. `[M]` What is BERTScore? How does it improve upon BLEU/ROUGE?
6. `[M]` Explain the MMLU benchmark. What does it measure and what are its limitations?
7. `[M]` How do you evaluate a chatbot? What metrics capture helpfulness, safety, and fluency?
8. `[M]` What is contamination in LLM evaluation? How do you detect and mitigate it?
9. `[M]` What is the difference between automatic evaluation, human evaluation, and model-based evaluation?
10. `[M]` What is HumanEval? How do you evaluate code generation models?
11. `[H]` Design an evaluation framework for a RAG-based Q&A system. Cover retrieval quality, answer quality, and faithfulness.
12. `[H]` You're comparing two LLMs for a production task. Design a rigorous evaluation protocol that goes beyond benchmark scores.
13. `[H]` What is the Chatbot Arena (LMSYS)? How does Elo rating work for LLM evaluation? What are its limitations?
14. `[H]` Why are LLM benchmarks becoming saturated? How should the field design next-generation evaluation?
15. `[H]` Explain the RAGAS framework for RAG evaluation. What are faithfulness, answer relevancy, context precision, and context recall?
16. `[H]` How do you evaluate an AI agent? What metrics capture task completion rate, efficiency, tool usage accuracy, and reliability?
17. `[H]` What is the difference between capability evaluation and safety evaluation? Design a red-teaming protocol for an LLM.

---

## Topic 29: ML System Design

1. `[E]` What are the key components of an ML system (data, features, model, serving, monitoring)?
2. `[E]` What is the difference between online prediction and batch prediction?
3. `[E]` What is a feature store? Why is it needed?
4. `[M]` Walk through a 6-step ML system design framework for an interview.
5. `[M]` What is training-serving skew? How do you prevent it?
6. `[M]` What is an A/B test for ML models? How do you calculate statistical significance?
7. `[M]` What is data drift? How do you detect it in production?
8. `[M]` Explain the feature engineering pipeline for a recommendation system.
9. `[M]` What is a model registry? How does it fit into the ML lifecycle?
10. `[H]` Design a recommendation system for an e-commerce platform. Cover data, features, model, serving, and monitoring.
11. `[H]` Design a fraud detection system that processes 10,000 transactions per second. What architecture would you use?
12. `[H]` Design a RAG-based Q&A system for enterprise documents. Cover ingestion, retrieval, generation, guardrails, and evaluation.
13. `[H]` Design a content moderation system for a social media platform. How do you handle text, images, and videos?
14. `[H]` What is the cold start problem in ML systems? How do you handle it in recommendations and search ranking?
15. `[H]` You're designing a search ranking system. Compare pointwise, pairwise, and listwise learning-to-rank approaches.
16. `[H]` How do you design an ML system that needs to handle data from multiple countries with different privacy regulations (GDPR, CCPA)?
17. `[H]` Design a real-time personalization system. How do you balance model freshness, latency, and compute cost?

---

## Topic 30: Model Deployment & Serving

1. `[E]` What is FastAPI? Why is it popular for serving ML models?
2. `[E]` What is Docker? Why is containerization important for ML deployment?
3. `[E]` What is the difference between a CPU and GPU deployment?
4. `[M]` How do you serve a PyTorch model with FastAPI? What are the key considerations?
5. `[M]` What is model serialization? Compare Pickle, ONNX, and TorchScript.
6. `[M]` What is a multi-stage Docker build? Why is it important for ML containers?
7. `[M]` How do you implement health checks, logging, and monitoring for a deployed model?
8. `[M]` What is ONNX Runtime? How does it speed up inference?
9. `[M]` Compare AWS SageMaker, GCP Vertex AI, and Azure ML for model deployment.
10. `[H]` Design a scalable model serving architecture that handles 10K requests per second with auto-scaling.
11. `[H]` How do you implement model versioning and A/B testing in production?
12. `[H]` What is canary deployment? How does it differ from blue-green deployment for ML models?
13. `[H]` Design a streaming LLM serving system. How do you handle Server-Sent Events (SSE) and WebSocket connections?
14. `[H]` What is serverless ML deployment (e.g., Modal, AWS Lambda)? When is it appropriate and when is it not?
15. `[H]` How do you load test an ML API? Design a load testing strategy using Locust or k6.
16. `[H]` You have a model that takes 500ms per inference. Users need < 100ms latency. What optimization strategies would you explore?

---

## Topic 31: MLOps (Tracking, CI/CD, Monitoring)

1. `[E]` What is MLOps? How does it differ from DevOps?
2. `[E]` What is experiment tracking? Why is it important?
3. `[E]` What is MLflow? What are its key components?
4. `[E]` What is CI/CD for ML? How does it differ from software CI/CD?
5. `[M]` Compare MLflow and Weights & Biases (W&B). When would you use each?
6. `[M]` Design a GitHub Actions CI/CD pipeline for an ML model. What stages would it include?
7. `[M]` What are ML-specific tests (data validation, model quality checks, bias checks)?
8. `[M]` What is a model registry? How does it manage model versions and deployment stages?
9. `[M]` Explain data drift detection. What statistical tests would you use (KS test, PSI)?
10. `[M]` What is Prometheus + Grafana monitoring? How do you use them for ML services?
11. `[H]` Design a complete MLOps pipeline from data ingestion to model deployment with automated retraining.
12. `[H]` How do you detect and handle concept drift (not just data drift)? When should you retrain vs update vs rollback?
13. `[H]` Design a monitoring system for an LLM-powered chatbot. What metrics do you track beyond standard ML metrics?
14. `[H]` What is a feature store (e.g., Feast, Tecton)? How does it solve the training-serving skew problem?
15. `[H]` Compare pipeline orchestration tools: Airflow, Prefect, Dagster. When would you use each?
16. `[H]` You noticed your model's accuracy dropped 5% over the past week in production. Walk through your debugging process.
17. `[H]` Design a quality gate system that automatically blocks bad models from reaching production.

---

## Topic 32: Databases & Data Infrastructure for ML

1. `[E]` What is the difference between SQL and NoSQL databases? When would you use each?
2. `[E]` What are SQL JOINs? Explain INNER, LEFT, RIGHT, and FULL JOIN.
3. `[E]` What is Redis? What are its common use cases in ML systems?
4. `[E]` What is a vector database? How does it differ from a traditional database?
5. `[M]` Write a SQL query using window functions (ROW_NUMBER, RANK, LAG) for feature engineering.
6. `[M]` Compare FAISS, ChromaDB, Pinecone, and Qdrant as vector databases. What are the trade-offs?
7. `[M]` Explain HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search.
8. `[M]` How do you use Redis for ML prediction caching and rate limiting?
9. `[M]` What is PostgreSQL? How would you store ML features and model metadata in it?
10. `[M]` What is Apache Kafka? How is it used in real-time ML pipelines?
11. `[H]` Design a data infrastructure for a real-time recommendation system. Cover data ingestion (Kafka), feature store (Redis), model serving, and logging (PostgreSQL).
12. `[H]` Compare IVF, HNSW, and Product Quantization for vector search. When would you use each indexing strategy?
13. `[H]` You need to store and query 100M embeddings with < 10ms latency. Design the vector search infrastructure.
14. `[H]` What are the consistency vs availability trade-offs (CAP theorem) for ML serving systems?
15. `[H]` Design an ETL pipeline that ingests data from 5 different sources, cleans it, computes features, and stores them for both training and real-time serving.
16. `[H]` Compare data warehouses (BigQuery, Snowflake, Redshift) for ML analytics. What features matter most?

---

## Topic 33: Safety, Ethics & Responsible AI

1. `[E]` What is AI hallucination? Give 3 examples.
2. `[E]` What is bias in ML? Name 3 types of bias (selection, measurement, representation).
3. `[E]` What is the difference between fairness, accountability, and transparency in AI?
4. `[E]` What is prompt injection? How is it different from jailbreaking?
5. `[M]` How do you detect and mitigate hallucination in LLM outputs?
6. `[M]` What is differential privacy? How is it applied in ML training?
7. `[M]` Explain the EU AI Act risk categories. What requirements apply to high-risk AI systems?
8. `[M]` What is red-teaming for LLMs? How do you systematically test for safety failures?
9. `[M]` What is model watermarking? How can you detect AI-generated text?
10. `[M]` How do you measure bias in a language model? What metrics and benchmarks exist?
11. `[H]` Design a safety evaluation framework for a customer-facing LLM. Cover hallucination detection, bias testing, toxicity, and prompt injection.
12. `[H]` What is the tension between model capability and safety? How do you balance them during alignment?
13. `[H]` Explain the concept of AI alignment at a philosophical level. What is the control problem?
14. `[H]` How do you build a content moderation system that works across languages and cultures?
15. `[H]` What is mechanistic interpretability? How can it help with AI safety?
16. `[H]` Design a responsible AI governance framework for an organization deploying LLMs. What policies, processes, and technical controls are needed?
17. `[H]` What is the "deceptive alignment" concern? How could a model learn to behave well during evaluation but not in deployment?

---

## Topic 34: Research Frontiers (Mamba, MoE, Test-Time Compute)

1. `[E]` What is a State Space Model (SSM)? How does it differ from a transformer?
2. `[E]` What is Mixture of Experts (MoE)? What is a router?
3. `[E]` What is test-time compute? How do models like o1/o3 use "thinking" at inference?
4. `[M]` Explain Mamba. How does its selective state space mechanism work?
5. `[M]` What is the Jamba architecture (Mamba + Transformer hybrid)?
6. `[M]` How does MoE routing work? What is the load balancing problem?
7. `[M]` What is synthetic data for LLM training? How are models like Phi trained on it?
8. `[M]` Explain long-context architectures. How do models handle 100K+ token contexts?
9. `[M]` What is mechanistic interpretability? What are superposition and features in this context?
10. `[H]` Compare Mamba vs Transformer on: parallelizability during training, inference speed, long-context performance, and quality on language tasks.
11. `[H]` Why do MoE models have more total parameters but use fewer per-token? What are the routing challenges (load balancing, expert collapse)?
12. `[H]` What is the relationship between test-time compute scaling and chain-of-thought reasoning? Is test-time compute a new scaling axis?
13. `[H]` Explain the key results from mechanistic interpretability research (e.g., induction heads, superposition hypothesis). What are the implications?
14. `[H]` What is retrieval-augmented pretraining (RETRO)? How does it differ from RAG applied post-training?
15. `[H]` What are the limitations of the transformer architecture that next-generation architectures try to address?
16. `[H]` Design a research experiment to test whether Mamba can replace transformers for a specific task. What would you measure and how?
17. `[H]` What is the "bitter lesson" (Rich Sutton)? How does it apply to current trends in AI architecture design?

---

## Topic 35: Recommendation Systems & Time Series

1. `[E]` What is collaborative filtering? What is the cold start problem?
2. `[E]` What is the difference between content-based and collaborative filtering?
3. `[E]` What is a time series? What makes it different from regular tabular data?
4. `[E]` What is stationarity? Why is it important for forecasting?
5. `[M]` Explain matrix factorization for recommendations. How does it decompose the user-item matrix?
6. `[M]` What is the two-tower model for recommendations? How does it enable real-time serving?
7. `[M]` Explain ARIMA. What are the (p, d, q) parameters?
8. `[M]` How do you create lag features for ML-based time series forecasting?
9. `[M]` What is the difference between point forecasting and probabilistic forecasting?
10. `[M]` How do you properly do cross-validation for time series data? Why can't you use standard k-fold?
11. `[H]` Design a recommendation system for a video streaming platform. Cover candidate generation, ranking, re-ranking, and diversity.
12. `[H]` Compare deep learning approaches to recommendations: NCF, DeepFM, two-tower, sequence-based (SASRec). When does each win?
13. `[H]` You're forecasting demand for 10,000 products across 500 stores. How would you structure this problem? Would you use one global model or individual models?
14. `[H]` How do you handle multiple seasonalities (daily, weekly, yearly) in time series forecasting?
15. `[H]` What is the candidate generation → ranking → re-ranking pipeline in production recommendation systems? What models and infrastructure are used at each stage?
16. `[H]` Compare XGBoost, LSTM, and Transformer-based models for time series forecasting. When does each approach win?
17. `[H]` How do you evaluate a recommendation system? Explain offline metrics (NDCG, MAP), online metrics (CTR, engagement), and the gap between them.

---

## Summary

| Difficulty | Count |
|-----------|-------|
| `[E]` Easy | ~155 |
| `[M]` Medium | ~185 |
| `[H]` Hard | ~255 |
| **Total** | **~595** |

> **How to use**: Pick a topic, cover the `[E]` questions first to verify foundations, then test yourself on `[M]` for depth, and `[H]` for interview-level performance. Refer to your study notes for answers.
