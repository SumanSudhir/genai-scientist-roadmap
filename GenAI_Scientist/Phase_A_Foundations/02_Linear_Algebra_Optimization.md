# Topic 2: Linear Algebra & Optimization

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 2 of 28
> **Scope**: Vectors, matrices, tensors, eigendecomposition, SVD, PCA, gradient descent, optimizers (Adam, AdamW), learning rate schedules, loss landscapes
> **Why this matters**: Attention is matrix multiplication. LoRA is low-rank factorization. AdamW is in every LLM training run. If you can't think in matrices and gradients, transformers are a black box.
> **Previous**: [Topic 1: Probability, Statistics & Information Theory](01_Probability_Statistics.md)
> **Next**: [Topic 3: Classical Machine Learning](03_Classical_ML.md)

---

## Table of Contents

1. [Vectors & Matrices](#1-vectors--matrices)
2. [Matrix Operations That Power Deep Learning](#2-matrix-operations-that-power-deep-learning)
3. [Eigenvalues, Eigenvectors & Eigendecomposition](#3-eigenvalues-eigenvectors--eigendecomposition)
4. [Singular Value Decomposition (SVD)](#4-singular-value-decomposition-svd)
5. [PCA — Principal Component Analysis](#5-pca--principal-component-analysis)
6. [Matrix Factorization in Embeddings & LoRA](#6-matrix-factorization-in-embeddings--lora)
7. [Gradient Descent — The Engine of Learning](#7-gradient-descent--the-engine-of-learning)
8. [Optimizers — From SGD to AdamW](#8-optimizers--from-sgd-to-adamw)
9. [Learning Rate Schedules](#9-learning-rate-schedules)
10. [Loss Landscapes, Saddle Points & Training Dynamics](#10-loss-landscapes-saddle-points--training-dynamics)
11. [Gradient Clipping & Gradient Accumulation](#11-gradient-clipping--gradient-accumulation)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. Vectors & Matrices

### 1.1 Vectors

A vector is an ordered list of numbers. In ML, a vector typically represents a point in some feature space or a single data sample.

$$\mathbf{x} = [x_1, x_2, \ldots, x_d]^T \quad \text{(column vector in } \mathbb{R}^d\text{)}$$

**Key operations**:

**Dot product** (inner product):

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta$$

The dot product is the foundation of:
- **Attention**: $\mathbf{Q} \cdot \mathbf{K}$ computes similarity between query and key vectors
- **Linear layers**: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ is a collection of dot products
- **Cosine similarity**: $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$, used everywhere in retrieval and embeddings

**Norms**:

$$\text{L1 norm: } \|\mathbf{x}\|_1 = \sum |x_i| \quad \text{(Manhattan distance, sparsity-inducing regularizer)}$$

$$\text{L2 norm: } \|\mathbf{x}\|_2 = \sqrt{\sum x_i^2} \quad \text{(Euclidean distance, standard regularizer)}$$

$$\text{L}_\infty \text{ norm: } \|\mathbf{x}\|_\infty = \max |x_i| \quad \text{(Chebyshev distance)}$$

- L2 norm is used in: weight decay, gradient clipping, normalizing embeddings
- L1 norm is used in: L1 regularization (Lasso), promotes sparsity

**Cosine similarity**:

$$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \cdot \|\mathbf{b}\|_2}$$

Range: [-1, 1]. Value of 1 means same direction, 0 means orthogonal, -1 means opposite.

Why it dominates in NLP: cosine similarity measures the angle between vectors, ignoring magnitude. Two documents about the same topic will point in similar directions in embedding space regardless of their length. This is why retrieval systems (RAG, semantic search) normalize embeddings and use cosine similarity or equivalently dot product on normalized vectors.

### 1.2 Matrices

A matrix is a 2D array of numbers. In deep learning, matrices represent:
- **Weight matrices**: Each linear layer is a matrix $W$ of shape (output_dim, input_dim)
- **Batches of data**: A batch of $B$ sequences each of length $T$ with embedding dimension $d$ is shape $(B, T, d)$
- **Attention scores**: The $QK^T$ product produces a $(T, T)$ matrix of pairwise similarities

**Key matrix properties**:

**Transpose**: $(A^T)_{ij} = A_{ji}$. Flips rows and columns.

If $A$ is $(m \times n)$, then $A^T$ is $(n \times m)$.

**Symmetric matrix**: $A = A^T$. Covariance matrices are always symmetric.

**Identity matrix**: $I$, where $I_{ii} = 1$ and $I_{ij} = 0$ for $i \neq j$. Acts as multiplication identity: $AI = IA = A$.

**Inverse**: $A^{-1}$ such that $A A^{-1} = I$. Exists only for square, non-singular matrices.

**Orthogonal matrix**: $Q^T Q = Q Q^T = I$, meaning $Q^{-1} = Q^T$. The columns form an orthonormal basis. Orthogonal matrices preserve norms and angles -- they're pure rotations (and reflections).

**Rank**: The number of linearly independent rows (or columns). A $(m \times n)$ matrix has rank at most $\min(m, n)$. Rank is central to understanding LoRA -- the key insight is that weight updates during fine-tuning often have low rank.

**Positive semi-definite (PSD)**: A symmetric matrix where $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$. Covariance matrices are always PSD. All eigenvalues of a PSD matrix are $\geq 0$.

### 1.3 Tensors

A tensor is the generalization of matrices to arbitrary dimensions. Deep learning operates on tensors everywhere.

```
Scalar:  0-dimensional tensor  (a single number)
Vector:  1-dimensional tensor  (shape: [d])
Matrix:  2-dimensional tensor  (shape: [m, n])
3D tensor: shape [B, T, d]    (a batch of sequences -- the standard transformer input)
4D tensor: shape [B, C, H, W] (a batch of images in CNN format)
```

**Tensor operations in transformers**:

The core of self-attention involves these tensor operations on a batch of shape $(B, T, d)$:

```
Q = X @ W_Q    # (B, T, d) @ (d, d_k) -> (B, T, d_k)  matrix multiply
K = X @ W_K    # (B, T, d) @ (d, d_k) -> (B, T, d_k)
V = X @ W_V    # (B, T, d) @ (d, d_v) -> (B, T, d_v)

scores = Q @ K^T / sqrt(d_k)   # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
weights = softmax(scores)       # (B, T, T)
output = weights @ V            # (B, T, T) @ (B, T, d_v) -> (B, T, d_v)
```

Every step is a matrix multiplication or element-wise operation. No loops over tokens. This is why transformers are so GPU-friendly -- they reduce to dense linear algebra that GPUs are optimized for.

**Broadcasting**: When operating on tensors of different shapes, smaller tensors are "broadcast" to match. For example, adding a bias vector $(d,)$ to a batch $(B, T, d)$ broadcasts the bias across all batch elements and positions.

---

## 2. Matrix Operations That Power Deep Learning

### 2.1 Matrix Multiplication

The single most important operation in deep learning. Every linear layer, every attention computation, every embedding lookup is a matrix multiply.

If $A$ is $(m \times k)$ and $B$ is $(k \times n)$, then $C = AB$ is $(m \times n)$:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

**Computational complexity**: $O(m \cdot k \cdot n)$ for standard matrix multiplication. For a linear layer with input dimension $d_{\text{in}}$ and output dimension $d_{\text{out}}$ processing a batch of $B$ samples: $O(B \cdot d_{\text{in}} \cdot d_{\text{out}})$.

**Why this matters for LLMs**: A transformer with hidden dimension $d = 4096$ and FFN dimension $4d = 16384$ does matrix multiplies of size $(B \cdot T, 4096) \times (4096, 16384)$ at every layer. For $B=32$, $T=2048$, that's $65536 \times 4096 \times 16384 \approx 4.4$ trillion multiply-adds PER layer. A 32-layer model does this 32 times (plus attention). This is why LLM training requires thousands of GPUs.

### 2.2 Hadamard (Element-wise) Product

$C = A \odot B$ (element-wise, same shape):

$$C_{ij} = A_{ij} \cdot B_{ij}$$

Used in: gating mechanisms (LSTM gates, GRU gates, SwiGLU activation in Llama).

SwiGLU activation:

$$\text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{x} W_1) \odot (\mathbf{x} W_2)$$

The element-wise product between the two branches is a Hadamard product acting as a gate -- one branch controls the flow of information from the other.

### 2.3 Outer Product

$$\mathbf{a} \otimes \mathbf{b} = \mathbf{a} \mathbf{b}^T$$

If $\mathbf{a}$ is $(m,)$ and $\mathbf{b}$ is $(n,)$, result is $(m \times n)$.

Used in: attention (the weighted sum of value vectors can be viewed as a sum of outer products), understanding rank-1 matrices.

A rank-1 matrix is an outer product of two vectors: $M = \mathbf{u} \mathbf{v}^T$. LoRA initializes its update as a product of two low-rank matrices, which is a sum of rank-1 outer products.

---

## 3. Eigenvalues, Eigenvectors & Eigendecomposition

### 3.1 Definition

For a square matrix $A$, an eigenvector $\mathbf{v}$ and eigenvalue $\lambda$ satisfy:

$$A \mathbf{v} = \lambda \mathbf{v}$$

The matrix $A$ stretches the vector $\mathbf{v}$ by a factor of $\lambda$ without changing its direction.

**Intuition**: Eigenvectors are the "natural axes" of a linear transformation. Along these axes, the transformation is just scaling (no rotation). The eigenvalue tells you how much scaling happens along each axis.

### 3.2 Computing Eigenvalues

Eigenvalues are roots of the characteristic polynomial:

$$\det(A - \lambda I) = 0$$

For a 2x2 matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$\lambda^2 - (a+d)\lambda + (ad - bc) = 0$$

where $(a+d)$ is the trace and $(ad-bc)$ is the determinant.

### 3.3 Eigendecomposition

If $A$ is a square matrix with $n$ linearly independent eigenvectors, it can be decomposed as:

$$A = V \Lambda V^{-1}$$

Where:
- $V$ is the matrix of eigenvectors (columns)
- $\Lambda$ is a diagonal matrix of eigenvalues

For a symmetric matrix (like covariance matrices), $V$ is orthogonal ($V^{-1} = V^T$), so:

$$A = V \Lambda V^T$$

This is called the spectral decomposition. It says: any symmetric matrix is just scaling along orthogonal axes.

### 3.4 Where Eigenvalues Appear in ML

**PCA**: The principal components are the eigenvectors of the covariance matrix. The eigenvalues tell you how much variance each component captures.

**Spectral clustering**: Uses eigenvectors of the graph Laplacian to find clusters.

**Training dynamics**: The eigenvalues of the Hessian (second derivative matrix of the loss) determine the curvature of the loss landscape. Large eigenvalues = steep directions, small eigenvalues = flat directions. This curvature determines how fast we can learn and what learning rate is stable.

**Vanishing/exploding gradients**: In RNNs, gradients involve products of weight matrices across time steps. The eigenvalues of the weight matrix determine whether gradients vanish ($|\lambda| < 1$) or explode ($|\lambda| > 1$).

---

## 4. Singular Value Decomposition (SVD)

### 4.1 Definition

SVD works for ANY matrix, not just square ones. For a matrix $A$ of shape $(m \times n)$:

$$A = U \Sigma V^T$$

Where:
- $U$ is $(m \times m)$, orthogonal -- left singular vectors
- $\Sigma$ is $(m \times n)$, diagonal -- singular values ($\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$)
- $V$ is $(n \times n)$, orthogonal -- right singular vectors

The singular values are always non-negative and conventionally sorted in decreasing order.

### 4.2 Relationship to Eigendecomposition

$$A^T A = V \Sigma^2 V^T \quad \text{(eigendecomposition of } A^T A \text{, eigenvalues are } \sigma_i^2\text{)}$$

$$A A^T = U \Sigma^2 U^T \quad \text{(eigendecomposition of } A A^T\text{)}$$

So singular values are square roots of eigenvalues of $A^T A$.

### 4.3 Low-Rank Approximation (The Eckart-Young Theorem)

This is perhaps the most important result for understanding LoRA and compression.

The best rank-$r$ approximation of $A$ (in Frobenius norm) is obtained by keeping only the top $r$ singular values:

$$A_r = U_r \Sigma_r V_r^T$$

Where $U_r$ is $(m \times r)$, $\Sigma_r$ is $(r \times r)$, $V_r$ is $(n \times r)$.

**The approximation error**:

$$\|A - A_r\|_F = \sqrt{\sigma_{r+1}^2 + \sigma_{r+2}^2 + \ldots + \sigma_{\min(m,n)}^2}$$

If the singular values decay rapidly (many small singular values), then a low-rank approximation captures most of the information with far fewer parameters.

**Parameter savings**: Original matrix $A$ has $m \cdot n$ parameters. Rank-$r$ approximation stores $U_r$ ($m \cdot r$) + $\Sigma_r$ ($r$) + $V_r$ ($n \cdot r$) $\approx (m + n) \cdot r$ parameters. When $r \ll \min(m, n)$, this is a massive reduction.

### 4.4 SVD in Practice

**Dimensionality reduction**: Truncated SVD is equivalent to PCA (for centered data).

**Latent Semantic Analysis (LSA)**: Apply SVD to the term-document matrix. The reduced dimensions capture latent semantic relationships. "Doctor" and "physician" end up close in the reduced space even if they never co-occur in the same document.

**Matrix completion**: Netflix Prize -- predict missing entries in a user-movie rating matrix by assuming it's approximately low-rank (users and movies both live in a low-dimensional taste space).

**Compression**: Compress weight matrices by keeping only the top singular values. This is the theoretical foundation for why LoRA works.

---

## 5. PCA — Principal Component Analysis

### 5.1 The Goal

Given high-dimensional data, find a lower-dimensional representation that preserves as much variance as possible.

### 5.2 The Algorithm

**Step 1**: Center the data (subtract the mean):

$$X_{\text{centered}} = X - \bar{X}$$

**Step 2**: Compute the covariance matrix:

$$C = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}} \quad \text{(shape: } d \times d\text{)}$$

**Step 3**: Eigendecompose the covariance matrix:

$$C = V \Lambda V^T$$

The eigenvectors (columns of $V$) are the **principal components** -- the directions of maximum variance. The eigenvalues (diagonal of $\Lambda$) tell you how much variance each component captures.

**Step 4**: Project data onto the top $k$ principal components:

$$X_{\text{reduced}} = X_{\text{centered}} V_k \quad \text{(where } V_k \text{ contains the top } k \text{ eigenvectors)}$$

### 5.3 Equivalence to SVD

PCA on centered data $X$ is equivalent to computing the SVD of $X$:

$$X = U \Sigma V^T$$

The right singular vectors $V$ are the principal components, and the singular values relate to the eigenvalues of the covariance matrix by: $\lambda_i = \sigma_i^2 / n$.

### 5.4 Explained Variance

The fraction of total variance captured by the top $k$ components:

$$\text{Explained variance ratio} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

In practice, you choose $k$ such that this ratio exceeds some threshold (e.g., 95%).

### 5.5 PCA in the Context of GenAI

**Embedding visualization**: Reduce 768-dimensional BERT embeddings or 4096-dimensional Llama embeddings to 2D/3D for visualization. (Though t-SNE/UMAP are usually preferred for visualization because they preserve local structure better.)

**Intrinsic dimensionality**: Research shows that the weight updates during fine-tuning lie in a low-dimensional subspace. Aghajanyan et al. (2020) measured this "intrinsic dimensionality" -- for BERT fine-tuning on many tasks, effective dimensionality is ~200-800, far less than the millions of parameters. This is the theoretical motivation for LoRA.

**Data preprocessing**: PCA can reduce feature dimensionality before feeding to traditional ML models, removing noise and speeding up training.

---

## 6. Matrix Factorization in Embeddings & LoRA

### 6.1 Embeddings as Matrix Lookup

An embedding layer is a matrix $E$ of shape $(V \times d)$, where $V$ is vocabulary size and $d$ is embedding dimension.

```
E = [[e_1],     # embedding for token 0
     [e_2],     # embedding for token 1
     ...
     [e_V]]     # embedding for token V-1
```

Looking up the embedding for token ID $i$ is just selecting row $i$: $E[i]$. For a batch of token IDs, this is a matrix gather operation.

The embedding matrix for a large LLM is significant: with $V = 32{,}000$ and $d = 4096$, $E$ has 131 million parameters -- and that's before any transformer layers.

### 6.2 GloVe as Matrix Factorization

GloVe word embeddings are explicitly trained via matrix factorization. The co-occurrence matrix $M$ ($V \times V$) where $M_{ij}$ = how often word $i$ appears near word $j$ is approximately factorized as:

$$M \approx W W'^T$$

Where $W$ and $W'$ are $(V \times d)$ embedding matrices. This is a low-rank factorization of the co-occurrence statistics.

Word2Vec can also be shown to implicitly factorize a matrix related to pointwise mutual information (PMI) of word co-occurrences.

### 6.3 LoRA — Low-Rank Adaptation

LoRA is the most important parameter-efficient fine-tuning method. Its foundation is pure linear algebra.

**The key insight**: When fine-tuning a pretrained model, the weight change $\Delta W = W_{\text{finetuned}} - W_{\text{pretrained}}$ has low intrinsic rank. We don't need a full $(d \times d)$ update matrix.

**The formulation**:

Instead of updating the full weight matrix:

$$W' = W + \Delta W \quad (\Delta W \text{ is } d \times d \text{, millions of parameters})$$

Factorize $\Delta W$ as a product of two low-rank matrices:

$$W' = W + BA$$

Where:
- $W$ is $(d \times d)$, FROZEN (not updated)
- $B$ is $(d \times r)$, TRAINABLE
- $A$ is $(r \times d)$, TRAINABLE
- $r \ll d$ (typical: $r = 8$ or $16$, $d = 4096$)

**Parameter savings**:

```
Full fine-tuning: d * d = 4096 * 4096 = 16.7M parameters per layer
LoRA (r=8):       d * r + r * d = 4096 * 8 * 2 = 65.5K parameters per layer

Reduction: ~256x fewer trainable parameters
```

**Why it works (the linear algebra reason)**:

$BA$ produces a matrix of rank at most $r$. By the Eckart-Young theorem, the best rank-$r$ approximation captures the top $r$ singular value directions. Research shows the actual weight updates during fine-tuning have rapidly decaying singular values -- most of the "information" in the update is concentrated in a few directions. LoRA restricts the update to this low-rank subspace, which is sufficient for most tasks.

**Initialization**: $A$ is initialized from a Gaussian, $B$ is initialized to zero. This means the initial LoRA update is $BA = 0$, so the model starts exactly at the pretrained weights. Training then learns the low-rank update.

**Scaling factor**: The update is scaled by $\alpha / r$:

$$W' = W + \frac{\alpha}{r} BA$$

$\alpha$ is a hyperparameter (typically $\alpha = r$ or $\alpha = 2r$) that controls the learning rate effectively applied to the LoRA update.

### 6.4 QLoRA — Quantized LoRA

QLoRA combines LoRA with quantization:

1. **Quantize W to 4-bit** (NF4 data type): Reduces memory for storing frozen weights by 8x
2. **Apply LoRA** on top: Only the small $B$, $A$ matrices are in full precision
3. **Double quantization**: Even the quantization constants are quantized

Result: Fine-tune a 65B model on a single 48GB GPU. This is what made LLM fine-tuning accessible to researchers without massive compute.

---

## 7. Gradient Descent — The Engine of Learning

### 7.1 The Optimization Problem

Training a model means finding parameters $\theta$ that minimize a loss function:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

$$\text{Where } \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; \theta), y_i)$$

For language models:

$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

### 7.2 Gradient Descent (Batch)

Update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

Where $\eta$ is the learning rate and $\nabla \mathcal{L}$ is the gradient of the loss with respect to all parameters.

**The gradient** points in the direction of steepest increase of the loss. Moving in the negative gradient direction reduces the loss (locally).

**Problem**: Computing $\nabla \mathcal{L}$ over the entire dataset is expensive. For a dataset of billions of tokens, one gradient computation requires a full forward and backward pass over all data.

### 7.3 Stochastic Gradient Descent (SGD)

Approximate the full gradient with a gradient computed on a single random sample:

$$\theta_{t+1} = \theta_t - \eta \nabla \ell(x_i; \theta_t)$$

**Properties**:
- Much cheaper per step (one sample instead of entire dataset)
- Noisy gradient estimate (high variance)
- The noise can actually help escape shallow local minima
- Converges in expectation: $\mathbb{E}[\nabla \ell(x_i; \theta)] = \nabla \mathcal{L}(\theta)$

### 7.4 Mini-Batch SGD

The practical middle ground: compute gradient over a mini-batch of $B$ samples:

$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \text{batch}} \nabla \ell(x_i; \theta_t)$$

**Typical batch sizes**:
- Computer vision: 32 - 256
- NLP / LLMs: 256 - 4096 (often in tokens: 500K - 4M tokens per batch)
- Larger batches reduce gradient noise, allow larger learning rates, but require more memory

**Why mini-batch works**: The gradient estimate's variance decreases as $1/B$ (by the law of large numbers). Batch size $B = 256$ gives a gradient estimate with 16x less variance than $B = 1$, while costing 256x more compute. There are diminishing returns -- doubling $B$ halves variance but doubles compute.

### 7.5 Backpropagation

Backpropagation is how we efficiently compute gradients. It's just the chain rule applied systematically through a computational graph.

For a composition $f = f_3(f_2(f_1(x)))$:

$$\frac{df}{dx} = \frac{df_3}{df_2} \cdot \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

**Forward pass**: Compute outputs layer by layer, storing intermediate activations.
**Backward pass**: Compute gradients layer by layer in reverse, using stored activations.

**Key insight**: Backprop computes ALL gradients in one backward pass with the same time complexity as the forward pass. Without backprop, computing the gradient for each parameter separately would be astronomically expensive.

**Memory cost**: We must store all intermediate activations from the forward pass for use in the backward pass. For a transformer with $L$ layers, this means storing $L$ sets of activations of shape $(B, T, d)$. This is a major memory bottleneck and why techniques like gradient checkpointing (recomputing activations instead of storing them) are used.

---

## 8. Optimizers — From SGD to AdamW

### 8.1 SGD with Momentum

Plain SGD oscillates in ravines (directions with high curvature) and moves slowly along the valley floor. Momentum smooths this out.

$$v_t = \beta v_{t-1} + \nabla \mathcal{L}(\theta_t) \quad \text{(accumulate velocity)}$$

$$\theta_{t+1} = \theta_t - \eta v_t \quad \text{(update with velocity)}$$

$\beta$ is the momentum coefficient, typically 0.9.

**Intuition**: Think of a ball rolling down a hill. Momentum allows it to accumulate speed along consistent gradient directions and dampen oscillations in inconsistent directions.

**Nesterov momentum**: Look ahead before computing the gradient:

$$v_t = \beta v_{t-1} + \nabla \mathcal{L}(\theta_t - \eta \beta v_{t-1})$$

This "peeking" ahead gives a more accurate gradient and slightly faster convergence.

### 8.2 AdaGrad (Adaptive Gradient)

Different parameters may need different learning rates. Parameters updated frequently (common features) should have smaller learning rates; parameters updated rarely (rare features) should have larger learning rates.

$$g_t = \nabla \mathcal{L}(\theta_t)$$

$$G_t = G_{t-1} + g_t^2 \quad \text{(accumulate squared gradients, element-wise)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

**Problem**: $G_t$ monotonically increases, so the effective learning rate monotonically decreases. Eventually it becomes infinitesimally small and learning stops entirely. This is catastrophic for long training runs like LLM pretraining.

### 8.3 RMSProp (Root Mean Square Propagation)

Fix AdaGrad's problem with an exponentially decaying average of squared gradients:

$$g_t = \nabla \mathcal{L}(\theta_t)$$

$$v_t = \beta v_{t-1} + (1 - \beta) g_t^2 \quad \text{(exponential moving average of squared grads)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$

$\beta$ is typically 0.99. The exponential moving average "forgets" old gradients, so the effective learning rate doesn't decay to zero.

### 8.4 Adam (Adaptive Moment Estimation)

Adam combines momentum (first moment) with RMSProp (second moment):

$$g_t = \nabla \mathcal{L}(\theta_t)$$

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment: momentum)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment: adaptive learning rate)}$$

Bias correction (crucial for early steps when $m$ and $v$ are biased toward zero):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$ (or lower for transformers).

**Why Adam works well**:
- **Momentum** ($m_t$): Smooth gradients, faster convergence along consistent directions
- **Adaptive learning rates** ($v_t$): Parameters with large gradients get smaller effective learning rates, parameters with small gradients get larger ones
- **Bias correction**: Compensates for the zero-initialization of $m$ and $v$, which is particularly important in the first few hundred steps

**Why Adam dominates for transformers**: Transformers have complex loss landscapes with varying curvature across parameters (attention heads, FFN layers, layer norm, embeddings all have very different gradient scales). Adam's per-parameter adaptive learning rate handles this naturally. SGD with a single learning rate struggles.

### 8.5 AdamW (Adam with Decoupled Weight Decay)

The critical difference between Adam and AdamW, and why it matters.

**The problem with L2 regularization in Adam**:

In standard SGD, L2 regularization and weight decay are equivalent:

$$\text{SGD + L2: } \theta = \theta - \eta (\nabla \mathcal{L} + \lambda \theta)$$

$$\text{SGD + weight decay: } \theta = \theta - \eta \nabla \mathcal{L} - \eta \lambda \theta$$

These are the same thing.

But in Adam, they are NOT the same. L2 regularization adds the penalty to the gradient BEFORE the adaptive scaling:

$$\text{Adam + L2: } g_t = \nabla \mathcal{L} + \lambda \theta \quad (\lambda \theta \text{ gets scaled by } 1/\sqrt{v_t})$$

The adaptive scaling effectively reduces the weight decay for parameters with large gradients and increases it for parameters with small gradients. This is unintended and harmful.

**AdamW decouples weight decay from the adaptive gradient**:

$$g_t = \nabla \mathcal{L}(\theta_t) \quad \text{(no } \lambda\theta \text{ in gradient)}$$

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \underbrace{\frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam update (adaptive)}} - \underbrace{\eta \lambda \theta_t}_{\text{Weight decay (applied directly)}}$$

The weight decay term $\eta \lambda \theta_t$ is applied uniformly to all parameters, independent of the adaptive gradient scaling. This is the mathematically correct way to regularize with adaptive optimizers.

**Practical impact**: AdamW is the standard optimizer for ALL modern LLM training (GPT, Llama, Mistral, etc.). Using Adam with L2 instead of AdamW leads to measurably worse generalization.

**Typical LLM training hyperparameters**:

```
AdamW with:
  eta (learning rate) = 1e-4 to 3e-4 (peak, after warmup)
  beta_1 = 0.9
  beta_2 = 0.95 (slightly lower than default 0.999 for training stability)
  weight_decay = 0.1
  epsilon = 1e-8
```

### 8.6 Optimizer Comparison Summary

```
Optimizer    | Adaptive LR | Momentum | Memory   | Best For
-------------|-------------|----------|----------|------------------
SGD          | No          | No       | O(n)     | Convex problems, vision (with tuning)
SGD+Momentum | No          | Yes      | O(2n)    | Vision (ResNet, etc.)
AdaGrad      | Yes         | No       | O(2n)    | Sparse data, NLP embeddings
RMSProp      | Yes         | No       | O(2n)    | RNNs, non-stationary
Adam         | Yes         | Yes      | O(3n)    | General purpose
AdamW        | Yes         | Yes      | O(3n)    | Transformers, LLMs (standard)
```

Note: Adam/AdamW requires 3x the parameter memory (parameters + first moment + second moment). For a 70B model, that's ~210B values = 840GB in FP32. This is why mixed precision and optimizer state sharding (ZeRO) are essential.

---

## 9. Learning Rate Schedules

The learning rate is arguably the most important hyperparameter. A fixed learning rate is almost never optimal. Modern LLM training uses carefully designed schedules.

### 9.1 Why Schedules Matter

- **Too high at the start**: Training diverges (loss explodes)
- **Too high later**: Can't converge to a good minimum (bounces around)
- **Too low throughout**: Wastes compute, gets stuck in suboptimal regions
- **Sweet spot**: Start moderate, increase briefly (warmup), then gradually decrease

### 9.2 Warmup

**What**: Linearly increase the learning rate from ~0 to the peak learning rate over the first N steps (typically 1-5% of total training).

$$\eta_t = \eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}} \quad \text{for } t < T_{\text{warmup}}$$

**Why warmup is necessary for transformers**:

At initialization, the model's representations are random. The Adam optimizer's second moment estimates ($v_t$) are initialized to zero and take many steps to become accurate. Without warmup:
- Early gradients are large and noisy
- Adam's $v_t$ estimates are poor (biased toward zero), leading to excessively large updates
- The model can diverge or land in a bad region of the loss landscape from which it never recovers

Warmup gives the optimizer time to build accurate moment estimates before taking large steps. Empirically, removing warmup from transformer training often causes training failure.

**Typical warmup**: 1,000-2,000 steps for LLM pretraining.

### 9.3 Cosine Decay

After warmup, the most common schedule for LLM training:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Where $T$ is the total number of training steps.

**Shape**: Starts at $\eta_{\text{peak}}$, slowly decreases (slow at first, fast in the middle, slow near the end), approaches $\eta_{\min}$.

**Why cosine**: It provides a smooth, gradual decay. The slow decrease near the end allows the model to "settle" into a good minimum. Empirically outperforms linear decay and step decay for LLMs.

**Typical**: $\eta_{\min} = 0.1 \cdot \eta_{\text{peak}}$ (decay to 10% of peak).

### 9.4 Linear Decay

$$\eta_t = \eta_{\text{peak}} \left(1 - \frac{t}{T}\right)$$

Simpler than cosine. Used in some BERT-style training. Generally slightly worse than cosine for LLMs.

### 9.5 Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / S \rfloor}$$

Drop the learning rate by factor $\gamma$ every $S$ steps. Common in vision (ResNets), less common for LLMs.

### 9.6 The Standard LLM Schedule

```
Phase 1: Linear warmup        (0 -> eta_peak over ~2000 steps)
Phase 2: Cosine decay          (eta_peak -> 0.1*eta_peak over remaining steps)

                  eta_peak
                   /\
                  /  \
                 /    \___
                /         \___
               /              \___
              /                   \___
    0 -------/                        \___  eta_min
             |        |                    |
          warmup    midpoint            end
```

---

## 10. Loss Landscapes, Saddle Points & Training Dynamics

### 10.1 The Loss Landscape

The loss function $\mathcal{L}(\theta)$ defines a surface in parameter space. For a model with $N$ parameters, this is an $N$-dimensional surface. For a 7B parameter model, the loss landscape is a surface in 7-billion-dimensional space.

**Key features of deep learning loss landscapes**:
- Highly non-convex (many local minima)
- High-dimensional (billions of dimensions)
- Connected low-loss regions (mode connectivity)
- Sharp vs flat minima

### 10.2 Convexity

**Convex function**: Any line segment between two points on the graph lies above the graph. Formally: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ for $\lambda \in [0,1]$.

**Properties of convex optimization**:
- Every local minimum is a global minimum
- Gradient descent converges to the global minimum
- Linear regression, logistic regression have convex losses

**Deep learning losses are NOT convex**. But interesting things happen in high dimensions.

### 10.3 Saddle Points

In high dimensions, saddle points are far more common than local minima.

**Why**: At a critical point (gradient = 0), each of the $N$ dimensions of the Hessian contributes either a positive eigenvalue (curving up = local min direction) or negative eigenvalue (curving down = local max direction). For a true local minimum, ALL $N$ eigenvalues must be positive. The probability of this happening by chance in very high dimensions is vanishingly small.

**Result**: Most critical points in deep learning are saddle points, not local minima. The challenge is not "getting stuck in bad local minima" -- it's navigating through saddle points efficiently.

**SGD noise helps**: The stochastic noise in mini-batch SGD naturally perturbs the model away from saddle points. This is one reason SGD-based optimizers work better than exact gradient methods in deep learning.

### 10.4 Sharp vs Flat Minima

**Observation**: Models that converge to "flat" minima (wide basins in the loss landscape) generalize better than those in "sharp" minima (narrow basins).

**Intuition**: A flat minimum is robust -- small perturbations to the weights don't significantly increase the loss. A sharp minimum is fragile -- a tiny weight change causes a big loss increase. Since test data differs from training data, we want robustness.

**What promotes flat minima**:
- Larger batch sizes tend to find sharper minima (one argument for not making batches too large)
- SGD noise pushes toward flat regions (the noise scale is larger in sharper regions, creating an implicit bias)
- Weight decay / L2 regularization biases toward simpler solutions
- Learning rate warmup avoids sharp, pathological regions early in training

### 10.5 The Hessian

The Hessian $H$ is the matrix of second derivatives:

$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}$$

**What it tells us**:
- Eigenvalues of $H$ give the curvature along each eigenvector direction
- Large positive eigenvalue = steep curvature (need small learning rate in this direction)
- Near-zero eigenvalue = flat direction (can take large steps)
- Negative eigenvalue = saddle point direction

**For deep learning**: The Hessian is $N \times N$ (billions x billions), so we never compute it explicitly. But its spectral properties inform optimizer design:
- Adam's adaptive learning rates implicitly approximate per-parameter curvature (using the second moment as a diagonal approximation to the Hessian)
- The maximum eigenvalue of the Hessian determines the maximum stable learning rate: $\eta < 2 / \lambda_{\max}$

---

## 11. Gradient Clipping & Gradient Accumulation

### 11.1 Gradient Clipping

**The problem**: During training, gradients can occasionally spike to very large values (due to bad batches, initialization effects, or loss landscape pathologies). A single large gradient step can destroy hours of training by pushing the model to a terrible part of the loss landscape.

**Solution**: Clip gradients to a maximum norm before the update step.

**Gradient clipping by norm** (the standard approach):

$$\mathbf{g} = \nabla \mathcal{L}(\theta)$$

$$\text{if } \|\mathbf{g}\|_2 > \text{max\_norm}: \quad \mathbf{g} = \mathbf{g} \cdot \frac{\text{max\_norm}}{\|\mathbf{g}\|_2}$$

This scales the entire gradient vector down so its norm equals max\_norm, while preserving the direction.

**Gradient clipping by value** (less common):

$$g_i = \text{clamp}(g_i, -\text{max\_value}, \text{max\_value})$$

Clips each gradient element independently. Can change the direction of the gradient.

**Standard practice for LLMs**: max\_norm = 1.0. This is used in virtually all LLM training (GPT, Llama, Mistral).

**Why it matters**: Without gradient clipping, transformer training is unstable. Attention scores can occasionally produce very large gradients (especially early in training before attention patterns stabilize), and a single unclipped step can cause the loss to spike to infinity (training divergence).

### 11.2 Gradient Accumulation

**The problem**: Effective batch sizes for LLMs are often 1M-4M tokens, but GPU memory can only hold a fraction of that per forward/backward pass.

**Solution**: Accumulate gradients over multiple micro-batches before taking an optimizer step.

```
Effective batch size = micro_batch_size * gradient_accumulation_steps * num_GPUs

Example:
  micro_batch_size = 4 (sequences per GPU per step)
  gradient_accumulation_steps = 8
  num_GPUs = 64
  Effective batch = 4 * 8 * 64 = 2048 sequences
```

**How it works**:
1. Forward + backward pass on micro-batch 1 -> accumulate gradients
2. Forward + backward pass on micro-batch 2 -> add to accumulated gradients
3. ... repeat for gradient_accumulation_steps ...
4. Divide accumulated gradient by gradient_accumulation_steps
5. Optimizer step (update weights)
6. Zero gradients, repeat

**Key property**: Mathematically equivalent to a single large batch (ignoring batch norm, which isn't used in transformers anyway). The gradient of a sum is the sum of gradients.

**Trade-off**: Gradient accumulation is slower per effective batch (sequential micro-batches) but uses less memory. Data parallelism across GPUs processes micro-batches in parallel.

---

## 12. Interview Questions & Answers

### Q1: Why does Adam work better than SGD for transformers?

**Answer**: Transformers have parameters with vastly different gradient scales. Attention weight matrices, FFN layers, layer norm parameters, and embedding layers all have different magnitudes and curvatures. SGD applies the same learning rate to all parameters -- this means the learning rate is either too large for some parameters (causing instability) or too small for others (causing slow learning).

Adam solves this with per-parameter adaptive learning rates. The second moment estimate $v_t$ tracks the typical gradient magnitude for each parameter and scales the update inversely. Parameters with large, frequent gradients get smaller effective learning rates; parameters with small, rare gradients get larger ones.

Additionally, the momentum in Adam (first moment $m_t$) smooths noisy gradient signals, which is important because transformer gradients can be quite noisy, especially through the attention mechanism where softmax can produce sharp or saturated distributions.

Empirically, transformer training with SGD requires extremely careful learning rate tuning and often still underperforms Adam/AdamW.

### Q2: What is the difference between Adam and AdamW? Why does it matter?

**Answer**: The difference is in how weight decay is applied.

In Adam with L2 regularization, the weight decay term ($\lambda \theta$) is added to the gradient BEFORE the adaptive scaling:

$$g_t = \nabla \mathcal{L} + \lambda \theta$$

$$\text{update} = \frac{g_t}{\sqrt{v_t}} \quad (\lambda \theta \text{ gets scaled by } 1/\sqrt{v_t})$$

This means the effective weight decay varies per parameter based on gradient history. Parameters with large historical gradients (large $v_t$) get less effective regularization, and vice versa. This is unintended.

In AdamW, weight decay is applied AFTER and independently of the adaptive gradient:

$$g_t = \nabla \mathcal{L} \quad \text{(no } \lambda\theta\text{)}$$

$$\text{adam\_update} = \frac{g_t}{\sqrt{v_t}}$$

$$\theta = \theta - \eta \cdot \text{adam\_update} - \eta \lambda \theta$$

The weight decay acts uniformly on all parameters regardless of gradient magnitude. This matches the theoretical intent of regularization (a Gaussian prior on weights).

It matters because AdamW consistently achieves better generalization than Adam+L2 for transformers. The Loshchilov & Hutter (2019) paper demonstrated this clearly, and AdamW has been the standard for all major LLM training since.

### Q3: Explain LoRA in terms of linear algebra.

**Answer**: LoRA exploits the fact that weight updates during fine-tuning are low-rank.

A pretrained weight matrix $W$ is $(d \times d)$ with $d^2$ parameters. During fine-tuning, the update $\Delta W = W_{\text{finetuned}} - W_{\text{pretrained}}$ should theoretically require $d^2$ parameters to represent. But research shows $\Delta W$ has rapidly decaying singular values -- most of its "information" lives in a low-rank subspace.

LoRA constrains $\Delta W$ to be rank $r$ by factorizing it as:

$$\Delta W = BA, \quad \text{where } B \text{ is } (d \times r) \text{ and } A \text{ is } (r \times d), \quad r \ll d$$

By the Eckart-Young theorem from SVD theory, the best rank-$r$ approximation captures the top $r$ singular value directions. Since the actual $\Delta W$ is approximately low-rank, restricting to rank $r$ loses little information.

This reduces trainable parameters from $d^2$ to $2dr$. For $d = 4096$ and $r = 8$, that's a 256x reduction. The frozen $W$ stays in memory (possibly quantized to 4-bit in QLoRA), and only the small $B$, $A$ matrices are updated.

The linear algebra insight is that in the space of all possible weight updates, the useful updates for a specific task live in a much lower-dimensional subspace. LoRA directly parameterizes this subspace.

### Q4: What is the role of warmup in learning rate scheduling?

**Answer**: Warmup serves two purposes:

**1. Stabilizing Adam's moment estimates**: At step 0, Adam's first moment ($m_0 = 0$) and second moment ($v_0 = 0$) contain no information. Despite bias correction, the early estimates are noisy. With a high learning rate, these noisy estimates lead to large, erratic updates that can push the model into bad regions. Warmup uses a small learning rate for the first ~1000-2000 steps, giving the optimizer time to build accurate moment estimates before taking aggressive steps.

**2. Avoiding sharp loss landscape regions early on**: At initialization, transformer parameters are random, and the loss landscape near random initialization contains sharp, unstable regions. A large learning rate can push the model into pathological areas (e.g., attention distributions that are extremely peaked or completely uniform). Warmup allows the model to gently move away from initialization toward smoother regions before increasing the learning rate.

Empirically, removing warmup from transformer training often causes loss spikes or complete divergence in the first few hundred steps. The standard practice is 1-2% of total training steps as warmup. For a 300K step training run, 2000-6000 warmup steps.

### Q5: Explain SVD. How is it related to PCA? How is it related to LoRA?

**Answer**: SVD decomposes any $(m \times n)$ matrix $A$ into:

$$A = U \Sigma V^T$$

where $U$ is orthogonal $(m \times m)$, $\Sigma$ is diagonal with non-negative singular values $(m \times n)$, and $V$ is orthogonal $(n \times n)$.

**Relation to PCA**: PCA on centered data $X$ finds directions of maximum variance. These are the eigenvectors of the covariance matrix $X^T X$. But $X^T X = V \Sigma^2 V^T$ (from SVD of $X$). So the principal components ARE the right singular vectors $V$, and the variance along each component is $\sigma_i^2 / n$. PCA is SVD applied to the data matrix.

**Relation to LoRA**: The Eckart-Young theorem states that the best rank-$r$ approximation of $A$ (minimizing Frobenius norm error) is obtained by keeping only the top $r$ singular values: $A_r = U_r \Sigma_r V_r^T$. LoRA parameterizes weight updates as a rank-$r$ matrix $BA$. While LoRA doesn't explicitly compute SVD, it learns a rank-$r$ update through gradient descent. The theoretical justification is that if the true weight update has rapidly decaying singular values (which it empirically does), then a rank-$r$ parameterization can capture the essential update with minimal loss. LoRA is learning an implicit low-rank SVD of the weight update.

### Q6: What are saddle points? Why are they more common than local minima in high dimensions?

**Answer**: A saddle point is a critical point (gradient = 0) that is a minimum in some directions and a maximum in others -- like a mountain pass.

At any critical point, the Hessian (matrix of second derivatives) has $N$ eigenvalues. For a true local minimum, ALL $N$ eigenvalues must be positive. For a saddle point, at least one eigenvalue is negative.

In very high dimensions ($N$ = billions for LLMs), imagine each eigenvalue being positive or negative with roughly equal probability (a simplification, but captures the intuition). The probability that ALL $N$ eigenvalues are positive is approximately $0.5^N$, which is vanishingly small for large $N$.

More precisely, random matrix theory and empirical studies of neural network loss surfaces show that critical points with lower loss tend to have fewer negative eigenvalues. The global minimum has all positive eigenvalues, but as the loss value at the critical point increases, more eigenvalues become negative, making it a saddle point rather than a local minimum.

This means for practical deep learning optimization, the concern is not "stuck in bad local minima" but rather "slow progress through saddle point plateaus." SGD with momentum and adaptive optimizers like Adam help navigate through saddle points by exploiting the curvature difference across dimensions.

### Q7: What is the vanishing gradient problem? How does it relate to eigenvalues?

**Answer**: In deep networks (and especially RNNs), gradients are computed via backpropagation, which involves multiplying Jacobian matrices across layers:

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial z_L} \cdot \frac{\partial z_L}{\partial z_{L-1}} \cdots \frac{\partial z_2}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

Each $\frac{\partial z_{l+1}}{\partial z_l}$ is a Jacobian matrix. The gradient is a product of $L$ such matrices.

The eigenvalues of these Jacobian matrices determine what happens:
- If the largest eigenvalue $< 1$ consistently: the product of matrices shrinks exponentially with depth. Gradients vanish -> early layers don't learn.
- If the largest eigenvalue $> 1$ consistently: the product grows exponentially. Gradients explode -> unstable training.

For RNNs, the same weight matrix $W$ is multiplied $T$ times (for sequence length $T$), so the issue is whether eigenvalues of $W$ are above or below 1 in magnitude.

**Solutions**:
- **Residual connections**: Transform $\frac{\partial z_{l+1}}{\partial z_l}$ from a pure matrix multiply to $I + f'(z_l)$, where the identity $I$ ensures eigenvalues are at least 1. This creates a "gradient highway" and is the primary reason transformers can be hundreds of layers deep.
- **Layer normalization**: Stabilizes the scale of activations across layers.
- **Careful initialization**: Xavier/He initialization sets weight scales so eigenvalues are approximately 1.
- **Gradient clipping**: Prevents explosion (but doesn't fix vanishing).

### Q8: A 7B parameter model uses AdamW. How much memory is needed for the optimizer state?

**Answer**: AdamW stores three values per parameter:
1. **Parameters ($\theta$)**: 7B values
2. **First moment ($m$)**: 7B values
3. **Second moment ($v$)**: 7B values

Total: $7B \times 3 = 21B$ values for the optimizer state (including parameters).

In full precision (FP32, 4 bytes each):

$$21B \times 4 \text{ bytes} = 84 \text{ GB just for optimizer state}$$

Plus:
- Gradients: 7B × 4 bytes = 28 GB (FP32) or 14 GB (FP16)
- Activations: varies with batch size and sequence length, but typically tens of GB

Total memory easily exceeds 100 GB for a single 7B model, which is why:
- **Mixed precision** (FP16/BF16 for forward/backward, FP32 for optimizer): Reduces gradient and activation memory
- **ZeRO** (DeepSpeed): Shards optimizer state, gradients, and parameters across GPUs. ZeRO Stage 3 divides all three, so each GPU holds only $1/N_{\text{gpu}}$ of the total.
- **Gradient checkpointing**: Recomputes activations during backward pass instead of storing them, trading compute for memory.

For a 70B model: optimizer state alone is 840 GB in FP32. This is why training requires at minimum 8-16 GPUs with 80GB each (A100/H100) with ZeRO-3 sharding.

### Q9: What is cosine learning rate decay? Why is it preferred for LLM training?

**Answer**: Cosine decay follows the formula:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

It starts at $\eta_{\text{peak}}$ and smoothly decreases to $\eta_{\min}$ following a half-cosine curve.

The shape has desirable properties:
- **Slow initial decay**: Near the peak, the cosine curve is flat, so the learning rate stays high for a while, allowing efficient exploration of the loss landscape.
- **Gradual acceleration**: In the middle, the decay accelerates, moving through the transition zone smoothly.
- **Slow final decay**: Near the end, the curve flattens again, allowing the model to carefully settle into a minimum without overshooting.

Compared to alternatives:
- **Linear decay** decreases at a constant rate, which may be too aggressive early on or too slow at the end.
- **Step decay** introduces abrupt changes that can destabilize training.

Cosine decay has become the standard for LLM pretraining (used by Llama, Mistral, GPT, and others) because it consistently achieves lower final loss than alternatives across a wide range of model sizes and training durations. The smooth annealing toward the end is particularly beneficial for finding good minima.

### Q10: Explain the relationship between batch size, learning rate, and training stability.

**Answer**: There is a well-known linear scaling rule:

If you multiply batch size by $k$, multiply learning rate by $k$.

**Why**: With batch size $B$, the gradient variance is roughly $1/B$ (law of large numbers). A larger batch gives a more accurate gradient estimate, which means you can take larger steps without overshooting. The gradient noise decreases as $\sqrt{B}$, so the signal-to-noise ratio increases, allowing a proportionally larger learning rate.

**Practical limits**: This linear scaling holds up to a critical batch size, beyond which increasing batch size stops improving training efficiency (you're already estimating the gradient very accurately). For LLMs, this critical batch size is often in the millions of tokens.

**Training stability**: Very large learning rates (even with large batches) can exceed the maximum stable learning rate determined by the loss landscape curvature (related to the maximum eigenvalue of the Hessian: $\eta < 2/\lambda_{\max}$). Warmup is essential to avoid this instability.

**The trade-off**: Larger batches are more computationally efficient (better GPU utilization, more parallelism) but can lead to sharper minima (worse generalization) and have diminishing returns on convergence speed. The optimal batch size balances compute efficiency, convergence speed, and generalization.

---

*Next: [Topic 3: Classical Machine Learning](09_Classical_ML.md)*
