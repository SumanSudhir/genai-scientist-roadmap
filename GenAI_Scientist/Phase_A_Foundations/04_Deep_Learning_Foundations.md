# Topic 4: Deep Learning Foundations

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 4 of 28
> **Scope**: Perceptrons, MLPs, activation functions, backpropagation, weight initialization, normalization, dropout, residual connections, CNNs, loss functions
> **Why this matters**: Every component here appears in transformers. Layer Norm is in every transformer block. Residual connections are what make deep transformers trainable. GELU/SwiGLU are the activations in modern LLMs. You need to understand WHY each design choice was made.
> **Prev**: [Topic 3: Classical Machine Learning](../Phase_A_Foundations/03_Classical_ML.md)
> **Next**: [Topic 5: Tokenization](../Phase_B_NLP_Core/05_Tokenization.md)

---

## Table of Contents

1. [Perceptrons & Multi-Layer Perceptrons](#1-perceptrons--multi-layer-perceptrons)
2. [Activation Functions](#2-activation-functions)
3. [Backpropagation](#3-backpropagation)
4. [Weight Initialization](#4-weight-initialization)
5. [Normalization Techniques](#5-normalization-techniques)
6. [Dropout](#6-dropout)
7. [Residual Connections](#7-residual-connections)
8. [Convolutional Neural Networks (Brief)](#8-convolutional-neural-networks-brief)
9. [Loss Functions](#9-loss-functions)
10. [Interview Questions & Answers](#10-interview-questions--answers)

---

## 1. Perceptrons & Multi-Layer Perceptrons

### 1.1 The Perceptron

The perceptron is the simplest neural network -- a single neuron that computes a weighted sum of inputs and passes it through a step function:

$$y = \text{step}\left(\sum_{i=1}^{n} w_i x_i + b\right) = \text{step}(\mathbf{w}^T \mathbf{x} + b)$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ is the input vector
- $\mathbf{w} \in \mathbb{R}^n$ is the weight vector
- $b$ is the bias
- step$(z) = 1$ if $z \geq 0$, else $0$

**Perceptron Learning Rule**: If the perceptron misclassifies a point $(x_i, y_i)$:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta(y_i - \hat{y}_i)\mathbf{x}_i$$

**Perceptron Convergence Theorem**: If the data is linearly separable, the perceptron learning algorithm converges in a finite number of steps. This was proven by Rosenblatt (1962).

**The XOR Problem**: Minsky and Papert (1969) showed that a single perceptron cannot learn the XOR function because XOR is not linearly separable. This result nearly killed neural network research for a decade. The solution: stack multiple layers.

### 1.2 Multi-Layer Perceptrons (MLPs)

An MLP is a stack of fully connected layers with non-linear activations between them:

$$\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\vdots$$
$$\mathbf{y} = \mathbf{W}_L \mathbf{h}_{L-1} + \mathbf{b}_L$$

where $\sigma$ is a non-linear activation function and $\mathbf{W}_i, \mathbf{b}_i$ are the weights and biases of layer $i$.

**Key insight**: Without non-linear activations, an MLP collapses to a single linear transformation no matter how many layers you stack. Composing linear functions gives another linear function:

$$\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W}'\mathbf{x} + \mathbf{b}'$$

The non-linearity is what gives depth its power.

**MLPs in Transformers**: The feed-forward network (FFN) in every transformer block is an MLP:

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$, with $d_{ff} = 4 \times d_{model}$ typically. This FFN is applied independently to each token position. Research suggests the FFN acts as a **key-value memory**, storing factual knowledge learned during pretraining (Geva et al., 2021). Each row of $\mathbf{W}_1$ is a "key" that detects a pattern, and the corresponding column of $\mathbf{W}_2$ is the "value" that gets written to the residual stream.

### 1.3 Universal Approximation Theorem

**Theorem** (Cybenko, 1989; Hornik, 1991): A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision, given a suitable activation function (e.g., sigmoid).

More formally: For any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\epsilon > 0$, there exists a network $g$ with one hidden layer such that:

$$\sup_{\mathbf{x} \in [0,1]^n} |f(\mathbf{x}) - g(\mathbf{x})| < \epsilon$$

**What it says**: One hidden layer is *sufficient* to represent any continuous function.

**What it does NOT say**:
- It says nothing about how **many** neurons are needed (could be exponentially many)
- It says nothing about whether gradient descent can **find** those weights
- It says nothing about **generalization** -- it's about representation, not learning

**Why depth helps in practice**: While one wide layer can represent anything, deep networks are exponentially more **parameter-efficient**. Functions that require exponentially many neurons in a shallow network can be represented with polynomially many in a deep network. This is the **depth separation** result.

**Relevance to transformers**: Transformers are deep networks (GPT-3 has 96 layers). The universal approximation theorem tells us that, in principle, even a single transformer layer with a sufficiently wide FFN could approximate any function. But in practice, depth provides the compositional structure that language understanding requires -- each layer can build on abstractions from previous layers.

---

## 2. Activation Functions

Activation functions introduce non-linearity. The choice of activation profoundly affects gradient flow, training stability, and model performance. Modern LLMs have converged on very specific choices -- understanding why requires understanding the full landscape.

### 2.1 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
- Output range: $(0, 1)$ -- interpretable as a probability
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Maximum derivative value: $0.25$ (at $z = 0$)

**Problems**:
1. **Vanishing gradients**: For $|z| > 5$, $\sigma'(z) \approx 0$. In a deep network, gradients are products of these derivatives (chain rule), so they shrink exponentially with depth. A 10-layer network with sigmoid activations has gradients that are at most $(0.25)^{10} \approx 10^{-6}$ of the output gradient.
2. **Not zero-centered**: Outputs are always positive, which means gradients on weights are always the same sign as the input gradient. This creates a zig-zag pattern in gradient updates, slowing convergence.
3. **Exponential computation**: $e^{-z}$ is expensive compared to simpler functions.

**Where it's still used**: Final layer for binary classification, gates in LSTMs, attention scores (via softmax, which is a generalization of sigmoid).

### 2.2 Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

**Properties**:
- Output range: $(-1, 1)$ -- zero-centered (fixes sigmoid's problem #2)
- Derivative: $\tanh'(z) = 1 - \tanh^2(z)$
- Maximum derivative value: $1.0$ (at $z = 0$)

**Improvement over sigmoid**: Zero-centered outputs lead to more stable gradient updates. But it still saturates for large $|z|$, so vanishing gradients remain a problem in deep networks.

**Where it's still used**: LSTM cell state updates, some normalization contexts.

### 2.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

**Properties**:
- Derivative: $1$ for $z > 0$, $0$ for $z < 0$ (undefined at $z = 0$, conventionally set to $0$)
- No saturation for positive values -- gradients don't vanish for $z > 0$
- Extremely cheap to compute
- Introduces sparsity: ~50% of neurons output zero for typical inputs

**The ReLU revolution** (Nair & Hinton, 2010; Krizhevsky et al., 2012): ReLU enabled training of much deeper networks (AlexNet, VGG, ResNet) because gradients flow freely through positive activations. The sparsity also acts as implicit regularization.

**The dying ReLU problem**: If a neuron's weights update such that all inputs produce $z < 0$, the gradient is permanently zero -- the neuron is "dead" and can never recover. This can happen with high learning rates. In practice, 10-40% of neurons can die during training.

### 2.4 Leaky ReLU and Variants

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

where $\alpha$ is a small constant (typically $0.01$).

**Fixes the dying ReLU problem**: Even for $z < 0$, the gradient is $\alpha \neq 0$, so neurons can always recover.

**Parametric ReLU (PReLU)**: Makes $\alpha$ a learnable parameter per channel. He et al. (2015) showed PReLU improved ImageNet accuracy.

**ELU (Exponential Linear Unit)**:

$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

ELU has smoother negative values, which can push the mean activation closer to zero (like batch normalization, without the extra computation).

### 2.5 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(Z \leq z)$$

where $\Phi(z)$ is the CDF of the standard normal distribution $\mathcal{N}(0, 1)$.

**Approximation** (commonly used):

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)$$

**Intuition**: GELU is a **smooth, probabilistic version of ReLU**. Instead of a hard threshold at zero:
- For very positive $z$: $\Phi(z) \approx 1$, so $\text{GELU}(z) \approx z$ (like ReLU)
- For very negative $z$: $\Phi(z) \approx 0$, so $\text{GELU}(z) \approx 0$ (like ReLU)
- Near zero: smooth transition, weighted by the probability that a Gaussian random variable is less than $z$

**Why GELU for transformers**: Introduced by Hendrycks and Gimpel (2016), GELU became the default activation in BERT, GPT-2, and GPT-3. The smoothness helps optimization -- no sharp kink at zero means gradients change smoothly, which matters for the large-scale optimization done in LLM training. The slight negative region (GELU can output small negative values) helps with gradient flow.

### 2.6 Swish

$$\text{Swish}(z) = z \cdot \sigma(\beta z)$$

where $\sigma$ is the sigmoid function and $\beta$ is a learnable or fixed parameter.

**Connection to GELU**: When $\beta = 1$, Swish is very similar to GELU. In fact, $\text{Swish}(z) = z \cdot \sigma(z)$ while $\text{GELU}(z) = z \cdot \Phi(z)$. Since the sigmoid CDF and the Gaussian CDF are similar, Swish $\approx$ GELU.

**Discovery**: Ramachandran et al. (2017) found Swish via automated search over activation function space. It consistently outperformed ReLU across tasks.

### 2.7 SwiGLU (The Modern LLM Choice)

$$\text{SwiGLU}(\mathbf{x}, \mathbf{W}_1, \mathbf{V}, \mathbf{W}_2) = (\text{Swish}(\mathbf{x}\mathbf{W}_1) \otimes \mathbf{x}\mathbf{V})\mathbf{W}_2$$

where $\otimes$ denotes element-wise multiplication.

**GLU (Gated Linear Unit)** (Dauphin et al., 2017): The core idea is a **gating mechanism**:

$$\text{GLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \otimes \sigma(\mathbf{x}\mathbf{V} + \mathbf{c})$$

One linear projection provides the "content" and another provides the "gate" (via sigmoid). The gate controls how much of the content passes through. SwiGLU replaces the sigmoid gate with Swish activation.

**Why SwiGLU in Llama/PaLM/Gemini**: Shazeer (2020) showed that GLU variants outperform standard FFN activations in transformers. SwiGLU specifically gave the best results on language modeling benchmarks. Llama, PaLM, and most modern LLMs use SwiGLU in their FFN blocks.

**Parameter count implication**: SwiGLU requires **three** weight matrices ($\mathbf{W}_1, \mathbf{V}, \mathbf{W}_2$) instead of two ($\mathbf{W}_1, \mathbf{W}_2$) in the standard FFN. To keep the parameter count similar, the hidden dimension $d_{ff}$ is reduced from $4d_{model}$ to $\frac{8}{3}d_{model}$ (rounded to the nearest multiple of 256 for hardware efficiency). Llama uses $d_{ff} = \frac{8}{3} d_{model}$.

### 2.8 Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties**:
- Outputs sum to 1 -- forms a valid probability distribution
- Differentiable everywhere -- enables gradient-based training
- Amplifies differences: larger logits get disproportionately more probability mass

**Numerical stability trick**: Compute $\text{softmax}(z_i - \max(\mathbf{z}))$ to avoid overflow in $e^{z_i}$.

**Temperature scaling**: $\text{softmax}(z_i / T)$ where $T$ controls sharpness:
- $T \to 0$: approaches argmax (hard, deterministic)
- $T = 1$: standard softmax
- $T \to \infty$: approaches uniform distribution (maximum entropy)

This is exactly how temperature works in LLM generation -- it's applied to logits before sampling.

**Where softmax appears in transformers**:
1. **Attention weights**: $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ -- distributes attention over tokens
2. **Output layer**: $\text{softmax}(W_o h)$ -- produces next-token probability distribution
3. **Mixture of Experts routing**: softmax over expert selection logits

### 2.9 Summary: Which Activation Where?

| Component | Activation | Why |
|-----------|-----------|-----|
| Transformer FFN (modern) | SwiGLU | Best language modeling performance |
| Transformer FFN (BERT/GPT-2) | GELU | Smooth ReLU, good optimization |
| Attention weights | Softmax | Need probability distribution |
| Output logits | Softmax | Need token probabilities |
| LSTM gates | Sigmoid | Need values in (0, 1) for gating |
| LSTM cell update | Tanh | Need values in (-1, 1) |
| Vision models (ResNet) | ReLU | Fast, works well with BatchNorm |

---

## 3. Backpropagation

Backpropagation is the algorithm that makes deep learning possible. It's the efficient application of the chain rule to compute gradients of the loss with respect to every parameter in the network.

### 3.1 The Chain Rule

For a composition of functions $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

For multivariable functions, this generalizes to Jacobians:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

### 3.2 Computational Graphs

A neural network is a directed acyclic graph (DAG) of operations. Each node computes a function, and edges carry values (forward) and gradients (backward).

**Example**: For a simple 2-layer network with loss $\mathcal{L}$:

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
$$\mathbf{h}_1 = \sigma(\mathbf{z}_1)$$
$$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2$$
$$\mathcal{L} = \text{loss}(\mathbf{z}_2, \mathbf{y})$$

**Forward pass**: Compute values left to right, caching intermediate results.

**Backward pass**: Compute gradients right to left using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} = \frac{\partial \mathcal{L}}{\partial \text{loss}} \cdot \frac{\partial \text{loss}}{\partial \mathbf{z}_2}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_2} \cdot \mathbf{h}_1^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \mathbf{W}_2^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}_2}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} \odot \sigma'(\mathbf{z}_1)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} \cdot \mathbf{x}^T$$

where $\odot$ is element-wise multiplication.

### 3.3 Gradient Flow Through Deep Networks

In a network with $L$ layers, the gradient of the loss with respect to parameters in layer $l$ involves a product of Jacobians:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \cdot \prod_{k=l}^{L-1} \frac{\partial \mathbf{z}_{k+1}}{\partial \mathbf{z}_k} \cdot \frac{\partial \mathbf{z}_l}{\partial \mathbf{W}_l}$$

This product of Jacobians is where problems arise:

**Vanishing Gradients**: If $\|\frac{\partial \mathbf{z}_{k+1}}{\partial \mathbf{z}_k}\| < 1$ for most layers, the product shrinks exponentially with depth. With sigmoid activations (max derivative 0.25), gradients in early layers are negligible, so they barely learn.

**Exploding Gradients**: If $\|\frac{\partial \mathbf{z}_{k+1}}{\partial \mathbf{z}_k}\| > 1$ for most layers, the product grows exponentially. Gradients become NaN, training diverges.

**Solutions**:
1. **Better activations**: ReLU (gradient = 1 for positive inputs), GELU, SwiGLU
2. **Better initialization**: Xavier, He (keep variance stable across layers)
3. **Residual connections**: $\mathbf{h}_l = f(\mathbf{h}_{l-1}) + \mathbf{h}_{l-1}$ (gradient has a direct path through the "+" operation)
4. **Normalization**: Layer Norm / Batch Norm (controls the scale of activations)
5. **Gradient clipping**: Cap the gradient norm to prevent explosions

### 3.4 Backpropagation in Transformers

Transformers have a specific gradient flow pattern worth understanding:

**Through the residual stream**: Each transformer layer adds to the residual:

$$\mathbf{h}_l = \mathbf{h}_{l-1} + \text{Attention}(\mathbf{h}_{l-1}) + \text{FFN}(\mathbf{h}_{l-1} + \text{Attention}(\mathbf{h}_{l-1}))$$

The gradient flows directly through the addition, providing a "gradient highway" from the loss all the way back to the input embeddings. This is why transformers can be very deep (96+ layers) without vanishing gradients.

**Through attention**: The softmax in attention creates a bottleneck -- gradients must flow through:

$$\frac{\partial}{\partial Q}\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling by $\frac{1}{\sqrt{d_k}}$ prevents softmax saturation, which would kill gradients.

**Memory cost**: Backpropagation requires storing all intermediate activations from the forward pass. For a transformer with $L$ layers, sequence length $n$, and hidden size $d$, this is $O(L \cdot n \cdot d)$ memory. This is why training LLMs requires so much GPU memory, and why **gradient checkpointing** (recomputing activations during backward pass instead of storing them) is a standard technique.

---

## 4. Weight Initialization

Proper initialization is critical. Bad initialization means gradients vanish or explode from the very first step, and the network may never recover.

### 4.1 The Problem

Consider a layer $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ where $\mathbf{W} \in \mathbb{R}^{m \times n}$.

If weights are initialized too large: activations explode, gradients explode, loss = NaN.
If weights are initialized too small: activations shrink to zero, gradients vanish, nothing learns.
If weights are initialized identically (e.g., all zeros): all neurons compute the same thing. Gradient updates are identical. Neurons never differentiate. This is the **symmetry breaking** problem.

### 4.2 Xavier/Glorot Initialization

**Goal**: Keep the variance of activations (and gradients) roughly constant across layers.

**Derivation**: For a layer with $n_{in}$ input neurons and $n_{out}$ output neurons, if inputs have variance $\text{Var}(x)$ and weights have variance $\text{Var}(w)$:

$$\text{Var}(z_j) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_{ij} x_i\right) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

(Assuming $w$ and $x$ are independent with zero mean.)

To keep $\text{Var}(z) = \text{Var}(x)$, we need:

$$\text{Var}(w) = \frac{1}{n_{in}}$$

Similarly, for backward pass stability, we'd want $\text{Var}(w) = \frac{1}{n_{out}}$.

**Xavier compromise** (Glorot & Bengio, 2010):

$$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

**In practice** (uniform distribution):

$$w \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

**Or** (normal distribution):

$$w \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

**Assumption**: This derivation assumes **linear activations** or activations that are approximately linear near zero (tanh, sigmoid in their linear region). It does NOT account for ReLU, which zeroes out half the inputs.

### 4.3 He/Kaiming Initialization

**Problem**: ReLU sets half the outputs to zero, which halves the variance. Xavier init underestimates the needed weight variance for ReLU networks.

**He initialization** (He et al., 2015): Accounts for ReLU's halving effect:

$$\text{Var}(w) = \frac{2}{n_{in}}$$

**In practice**:

$$w \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

**For Leaky ReLU** with slope $\alpha$ for negative inputs:

$$\text{Var}(w) = \frac{2}{(1 + \alpha^2) n_{in}}$$

### 4.4 Initialization in Transformers

Modern transformers use specific initialization strategies:

**GPT-style**: Weights initialized from $\mathcal{N}(0, 0.02)$ with special scaling for residual layers. Specifically, weights in the output projection of attention and the second linear layer of FFN are scaled by $\frac{1}{\sqrt{2L}}$ where $L$ is the number of layers. This prevents the residual stream's variance from growing with depth.

**Why $\frac{1}{\sqrt{2L}}$**: Each transformer layer has two residual additions (attention + FFN). After $L$ layers, the residual stream has had $2L$ additions. If each addition contributes variance $\sigma^2$, the total variance is $2L\sigma^2$. Scaling by $\frac{1}{\sqrt{2L}}$ keeps the total variance $O(1)$.

**Practical note**: For very large models (100B+ parameters), initialization is so critical that a bad initialization can waste millions of dollars in failed training runs. This is why LLM training recipes are closely guarded.

---

## 5. Normalization Techniques

Normalization keeps activations in a "well-behaved" range during training, preventing both vanishing and exploding activations. The two major techniques are Batch Normalization and Layer Normalization, but they are used in very different contexts.

### 5.1 Batch Normalization (BatchNorm)

**Definition** (Ioffe & Szegedy, 2015): For a mini-batch $\mathcal{B} = \{x_1, ..., x_m\}$:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where:
- $\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i$ (batch mean)
- $\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2$ (batch variance)
- $\gamma, \beta$ are learnable scale and shift parameters
- $\epsilon$ is a small constant for numerical stability (typically $10^{-5}$)

**What BatchNorm normalizes**: Statistics are computed **across the batch dimension** for each feature/channel independently. For a batch of images with shape (B, C, H, W), it computes mean and variance over the (B, H, W) dimensions for each channel C separately.

**Why $\gamma$ and $\beta$?** Without them, normalization would force every layer's output to have zero mean and unit variance, which limits the representational power. The learnable $\gamma$ and $\beta$ allow the network to undo the normalization if needed -- it can learn $\gamma = \sigma$ and $\beta = \mu$ to recover the original distribution. So BatchNorm provides a beneficial inductive bias while preserving the model's expressiveness.

**Benefits**:
1. **Reduces internal covariate shift**: Each layer's input distribution stays stable
2. **Allows higher learning rates**: Activations don't explode/vanish
3. **Acts as regularization**: Mini-batch statistics add noise, similar to dropout

**Training vs inference**: During training, uses mini-batch statistics. During inference, uses running averages accumulated during training (exponential moving average of $\mu$ and $\sigma^2$).

### 5.2 Why BatchNorm Fails for Transformers

**Problem 1: Variable sequence lengths**. In NLP, sequences in a batch have different lengths (even with padding). Computing batch statistics across sequences of different lengths mixes meaningful tokens with padding tokens, producing unreliable statistics.

**Problem 2: Batch size dependency**. BatchNorm statistics become unstable for small batch sizes. LLM training often has small *effective* batch sizes per GPU due to memory constraints (even if total batch size is large across GPUs). With distributed training and gradient accumulation, the local batch on each GPU may be small.

**Problem 3: Autoregressive generation**. During inference, LLMs generate one token at a time (batch size = 1). BatchNorm needs a batch to compute statistics, making it fundamentally incompatible with single-sample inference.

**Problem 4: Sequence position coupling**. BatchNorm would normalize each position across all sequences in the batch, implicitly coupling the representations at the same position across different sequences. This is undesirable -- position 5 in "The cat sat on the mat" and position 5 in "Neural networks learn representations" have no meaningful relationship.

### 5.3 Layer Normalization (LayerNorm)

**Definition** (Ba et al., 2016): For a single sample's feature vector $\mathbf{x} \in \mathbb{R}^d$:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y_i = \gamma_i \hat{x}_i + \beta_i$$

where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ (mean across features)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$ (variance across features)
- $\gamma, \beta \in \mathbb{R}^d$ are learnable parameters

**What LayerNorm normalizes**: Statistics are computed **across the feature dimension** for each sample (and each position) independently. For a transformer hidden state of shape (B, T, d), it normalizes over the d dimension for each (batch, position) pair.

**Why LayerNorm works for transformers**:
1. **No batch dependency**: Statistics computed per-sample, so batch size doesn't matter
2. **Works at inference**: Single sample, single token -- no problem
3. **Handles variable lengths**: Each token normalized independently
4. **Position independence**: Each position normalized based on its own features

### 5.4 Pre-LN vs Post-LN

**Post-LN** (original transformer, "Attention Is All You Need"):

$$\mathbf{h} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

LayerNorm is applied **after** the residual addition.

**Pre-LN** (now standard in most LLMs):

$$\mathbf{h} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$$

LayerNorm is applied **before** the sublayer (attention or FFN), and the residual connection bypasses the normalization.

**Why Pre-LN won**:

1. **Gradient flow**: In Post-LN, the gradient must pass through LayerNorm at every layer. LayerNorm has a Jacobian that can dampen or amplify gradients. In Pre-LN, the residual connection provides a clean gradient path that bypasses LayerNorm entirely.

2. **Training stability**: Post-LN transformers are notoriously unstable to train without careful learning rate warmup. The gradient norms can vary dramatically across layers. Pre-LN produces much more uniform gradient norms.

3. **Warmup reduction**: Pre-LN transformers can often train with shorter warmup periods or even no warmup, because the gradient flow is inherently more stable.

4. **Empirical evidence**: Xiong et al. (2020) showed that Pre-LN allows stable training of transformers up to 24 layers without warmup, while Post-LN diverges without it.

**GPT-2/3, Llama, Mistral, PaLM** all use Pre-LN.

### 5.5 RMSNorm

**Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

**Simplification over LayerNorm**: RMSNorm drops the mean subtraction and the bias ($\beta$) term. It only re-scales by the root mean square of the features.

**Why it works**: The re-centering (mean subtraction) in LayerNorm is the expensive part and provides limited benefit. The re-scaling (variance normalization) is what matters for training stability. RMSNorm is ~10-15% faster than LayerNorm with negligible quality difference.

**Used by**: Llama (all versions), Mistral, Gemma, PaLM. It's becoming the de facto standard.

---

## 6. Dropout

### 6.1 Standard Dropout

**Definition** (Srivastava et al., 2014): During training, randomly set each neuron's output to zero with probability $p$:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i & \text{with probability } 1 - p \end{cases}$$

**Intuition**: Dropout prevents co-adaptation of neurons. Without dropout, neurons can develop complex co-dependencies -- neuron A only works if neuron B also fires in a specific pattern. Dropout forces each neuron to be independently useful because it can't rely on any particular partner being present.

**Ensemble interpretation**: A network with dropout is training an exponential number of "thinned" subnetworks simultaneously. Each training step samples a different subnetwork. At test time, you use the full network, which approximates the ensemble average of all subnetworks.

### 6.2 Inverted Dropout (The Standard Implementation)

**The scaling problem**: During training with dropout rate $p$, each neuron is active with probability $(1-p)$. The expected output is $(1-p) \cdot h_i$. At test time, all neurons are active, so the expected output is $h_i$. This mismatch means test-time outputs are $\frac{1}{1-p}$ times larger than training-time expected outputs.

**Naive approach**: Scale test-time outputs by $(1-p)$. But this requires modifying inference code.

**Inverted dropout**: Scale training-time outputs by $\frac{1}{1-p}$ instead:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ \frac{h_i}{1-p} & \text{with probability } 1-p \end{cases}$$

Now $E[\tilde{h}_i] = (1-p) \cdot \frac{h_i}{1-p} = h_i$, which matches the test-time output with no modification needed at inference. This is the standard implementation in PyTorch (`nn.Dropout`).

### 6.3 Dropout in Transformers

**Where dropout is applied in transformers**:
1. **Attention dropout**: Applied to attention weights after softmax (before multiplying by V)
2. **Residual dropout**: Applied to the output of each sublayer (attention, FFN) before adding to the residual
3. **Embedding dropout**: Sometimes applied to the sum of token + positional embeddings

**Typical dropout rates**: BERT used $p = 0.1$. GPT-2 also used $p = 0.1$.

**Modern trend**: Many large LLMs (Llama, PaLM, Chinchilla) use **no dropout at all**. Why?
1. **Sufficient data**: With trillions of training tokens, overfitting is not the primary concern -- underfitting is.
2. **Training efficiency**: Dropout effectively wastes compute -- you're zeroing out neurons you already computed.
3. **Scale as regularizer**: Very large models trained on very large datasets seem to generalize well without explicit regularization.
4. **Other regularizers**: Weight decay, data augmentation/mixing, and the stochastic nature of large-batch SGD provide implicit regularization.

### 6.4 DropPath (Stochastic Depth)

**Definition** (Huang et al., 2016): Instead of dropping individual neurons, drop entire layers (or residual branches):

$$\mathbf{h}_l = \begin{cases} \mathbf{h}_{l-1} & \text{with probability } p_l \\ \mathbf{h}_{l-1} + f_l(\mathbf{h}_{l-1}) & \text{with probability } 1 - p_l \end{cases}$$

**Used in**: Vision Transformers (ViT, DeiT), where it's more effective than standard dropout. The drop rate usually increases with depth (deeper layers are dropped more often).

---

## 7. Residual Connections

Residual connections are arguably the single most important architectural innovation for training deep networks. Without them, transformers with 96+ layers would be untrainable.

### 7.1 The Core Idea

**ResNet** (He et al., 2015): Instead of learning $\mathbf{h}_l = F(\mathbf{h}_{l-1})$ directly, learn the **residual**:

$$\mathbf{h}_l = F(\mathbf{h}_{l-1}) + \mathbf{h}_{l-1}$$

where $F$ is the layer's function (attention block, FFN, etc.).

**Intuition**: It's easier to learn $F(\mathbf{x}) = 0$ (identity mapping) than to learn $F(\mathbf{x}) = \mathbf{x}$ (identity function). If a layer isn't needed, the network can simply learn $F \approx 0$ and pass the input through unchanged. This means adding layers can never hurt -- at worst, they learn identity.

### 7.2 Why Residual Connections Fix Gradient Flow

**Without residuals**: Gradient is a product of layer Jacobians:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_l} = \prod_{k=l+1}^{L} \frac{\partial F_k}{\partial \mathbf{h}_{k-1}} \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}$$

This product can vanish or explode exponentially with $L - l$.

**With residuals**: Since $\mathbf{h}_l = F(\mathbf{h}_{l-1}) + \mathbf{h}_{l-1}$:

$$\frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} = \frac{\partial F}{\partial \mathbf{h}_{l-1}} + \mathbf{I}$$

The gradient from $\mathbf{h}_L$ to $\mathbf{h}_l$ becomes:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \cdot \prod_{k=l+1}^{L}\left(\mathbf{I} + \frac{\partial F_k}{\partial \mathbf{h}_{k-1}}\right)$$

Expanding this product, one term is always $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \cdot \mathbf{I}^{L-l} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}$. This means **the gradient from the loss always has a direct path to every layer**, regardless of depth. The gradient cannot vanish completely.

### 7.3 The Residual Stream View of Transformers

Modern interpretability research views the transformer as a **residual stream** that information flows through:

$$\mathbf{x}_0 = \text{Embedding}(\text{tokens})$$
$$\mathbf{x}_l = \mathbf{x}_{l-1} + \text{Attn}_l(\mathbf{x}_{l-1}) + \text{FFN}_l(\mathbf{x}_{l-1})$$
$$\text{output} = \text{Unembed}(\mathbf{x}_L)$$

Each attention head and FFN **reads from** and **writes to** the residual stream. The final prediction is made by reading from the residual stream at the last position. This view has been enormously productive for mechanistic interpretability:

- **Attention heads** read from the residual stream (via Q, K, V projections) and write back (via output projection)
- **FFN layers** read from the residual stream and write back
- All components communicate **only through the residual stream**

### 7.4 What Happens Without Residual Connections?

**Experiment**: Remove residual connections from a 96-layer transformer.

**Result**: The network is untrainable. Gradients vanish within ~10 layers. The first 86 layers receive essentially zero gradient and never learn. The effective depth drops to ~10 layers, defeating the purpose of depth.

Even with perfect initialization and normalization, residual connections are essential for depth. This has been empirically verified repeatedly -- there is no known alternative that works as well for very deep networks.

### 7.5 Residual Connection Variants

**Pre-activation residual** (He et al., 2016): Move normalization and activation before the weight layers:

$$\mathbf{h}_l = \mathbf{h}_{l-1} + F(\text{Norm}(\mathbf{h}_{l-1}))$$

This is equivalent to the Pre-LN transformer and gives even cleaner gradient flow.

**Weighted residuals**: Some architectures scale the residual addition:

$$\mathbf{h}_l = \alpha \cdot F(\mathbf{h}_{l-1}) + \mathbf{h}_{l-1}$$

DeepNet (Microsoft, 2022) uses $\alpha$ scaling based on depth to stabilize training of transformers with 1000+ layers.

---

## 8. Convolutional Neural Networks (Brief)

CNNs are not the focus of LLM interviews, but they appear in multimodal models and have historical importance. This section covers what you need for a GenAI Scientist interview.

### 8.1 Core Concepts

**Convolution**: Slide a filter (kernel) across the input, computing dot products at each position:

$$(f * g)[n] = \sum_{k} f[k] \cdot g[n-k]$$

In 2D (for images), a kernel $\mathbf{K} \in \mathbb{R}^{k \times k}$ slides over an input feature map $\mathbf{X} \in \mathbb{R}^{H \times W}$:

$$(\mathbf{X} * \mathbf{K})[i, j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} \mathbf{X}[i+m, j+n] \cdot \mathbf{K}[m, n]$$

**Key properties**:
1. **Translation equivariance**: If the input shifts, the output shifts identically. This is the right inductive bias for vision -- a cat is a cat regardless of position.
2. **Parameter sharing**: The same kernel is applied at every spatial position. A 3x3 kernel has only 9 parameters regardless of image size.
3. **Local connectivity**: Each output neuron depends only on a local patch of the input, unlike fully connected layers where every output depends on every input.

**Pooling**: Reduces spatial resolution. Max pooling takes the maximum value in each patch; average pooling takes the mean. Provides translation invariance (small shifts don't change the output) and reduces computation.

### 8.2 Key Architectures You Should Know

**ResNet** (He et al., 2015): Introduced residual connections (skip connections). Made it possible to train networks with 152+ layers. Key insight: deeper networks should perform at least as well as shallower ones (because extra layers can learn identity), but in practice they don't without residuals.

**Why ResNet matters for GenAI**: ResNet is the vision encoder backbone in many multimodal models. CLIP uses a modified ResNet (or ViT). ResNet's residual connections directly inspired their use in transformers.

### 8.3 Vision Transformers (ViT)

**Vision Transformer** (Dosovitskiy et al., 2020): Applies the transformer architecture to images:

1. Split image into patches (e.g., 16x16 pixels)
2. Flatten each patch into a vector
3. Project through a linear layer (patch embedding)
4. Add positional embeddings
5. Feed through a standard transformer encoder
6. Use [CLS] token output for classification

**Why ViT matters for GenAI**: ViT (and its variants like SigLIP, DINOv2) is the vision encoder in most modern multimodal LLMs (LLaVA, GPT-4V, Gemini). Understanding ViT means understanding how images are fed into LLMs.

### 8.4 CNN vs Transformer Trade-offs

| Aspect | CNN | Transformer |
|--------|-----|------------|
| Inductive bias | Local (convolutional) | Global (attention) |
| Data efficiency | Better with small data | Needs more data |
| Scalability | Saturates at scale | Keeps improving with scale |
| Sequence modeling | Not natural | Natural |
| Positional info | Built in (spatial structure) | Must be added explicitly |
| Compute | O(k^2 * d) per position | O(n * d) per position (attention) |

---

## 9. Loss Functions

Loss functions define what the network optimizes. Choosing the right loss function is not just a technical detail -- it fundamentally shapes what the model learns.

### 9.1 Cross-Entropy Loss

The most important loss function in deep learning.

**Binary Cross-Entropy** (for binary classification):

$$\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Categorical Cross-Entropy** (for multi-class classification):

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_{correct})$$

where $y_c$ is 1 for the correct class and 0 otherwise, and $\hat{y}_c = \text{softmax}(z)_c$.

**Why cross-entropy is the language model loss**: Language modeling is next-token prediction -- a classification problem over the vocabulary $V$. The loss per token is:

$$\mathcal{L}_t = -\log P(x_t \mid x_{<t})$$

The total loss over a sequence of $T$ tokens is:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

**Connection to MLE**: Minimizing cross-entropy is equivalent to maximizing the likelihood of the training data under the model. If the model assigns probability $P_\theta(x)$ to sequence $x$:

$$\arg\max_\theta \prod_{t} P_\theta(x_t \mid x_{<t}) = \arg\min_\theta -\sum_t \log P_\theta(x_t \mid x_{<t})$$

**Connection to KL divergence**: Cross-entropy between the true distribution $P$ and model distribution $Q$ is:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

Since $H(P)$ is constant (it's the entropy of the data), minimizing cross-entropy minimizes KL divergence between the true distribution and the model.

### 9.2 Mean Squared Error (MSE)

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Properties**:
- Penalizes large errors quadratically -- sensitive to outliers
- Gradient: $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$ -- gradient magnitude proportional to error size
- Corresponds to MLE under Gaussian noise assumption

**Where MSE appears in GenAI**:
1. **Diffusion models**: The denoising objective is essentially MSE between predicted noise and actual noise: $\mathcal{L} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$
2. **Regression heads**: When predicting continuous values (e.g., reward model scores)
3. **Knowledge distillation**: MSE on logits or hidden states between teacher and student

### 9.3 Contrastive Loss

**Purpose**: Learn representations where similar items are close and dissimilar items are far apart in embedding space.

**Pairwise contrastive loss** (Hadsell et al., 2006):

$$\mathcal{L} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

where $d = \|f(x_1) - f(x_2)\|$ is the distance between embeddings, $y = 1$ for similar pairs, and $m$ is a margin.

**InfoNCE / NT-Xent** (used in CLIP, SimCLR):

$$\mathcal{L}_i = -\log\frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i, z_j)/\tau)}$$

where:
- $z_i$ and $z_i^+$ are embeddings of a positive pair
- The denominator sums over all $N$ samples in the batch (including the positive)
- $\tau$ is a temperature parameter
- $\text{sim}$ is cosine similarity

**CLIP's contrastive loss**: For a batch of $N$ (image, text) pairs, CLIP maximizes the similarity of the $N$ matched pairs while minimizing the similarity of the $N^2 - N$ unmatched pairs. This creates a shared embedding space where images and their descriptions are close.

**Why contrastive loss matters for GenAI**:
1. **CLIP** -- the foundation of text-to-image models (Stable Diffusion) and multimodal LLMs
2. **Sentence-BERT** -- embedding models for RAG and semantic search
3. **DPO** -- can be viewed as a contrastive loss between preferred and dispreferred responses

### 9.4 Triplet Loss

$$\mathcal{L} = \max(0, d(a, p) - d(a, n) + m)$$

where:
- $a$ is the anchor sample
- $p$ is a positive (similar) sample
- $n$ is a negative (dissimilar) sample
- $m$ is a margin
- $d$ is a distance function (typically L2)

**Intuition**: Push the negative farther than the positive by at least margin $m$.

**Hard negative mining**: The most effective negatives are those that are close to the anchor but from a different class. Mining hard negatives is crucial for good performance -- easy negatives contribute zero gradient (already satisfy the margin).

### 9.5 Focal Loss

**Problem**: In tasks with extreme class imbalance (e.g., object detection where 99% of regions are background), cross-entropy is dominated by the easy, frequent class.

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $(1 - p_t)^\gamma$ down-weights the loss contribution from well-classified examples. With $\gamma = 0$, this is standard cross-entropy.

### 9.6 Label Smoothing

**Standard one-hot targets**: $y_{correct} = 1$, $y_{other} = 0$.

**Smoothed targets**: $y_{correct} = 1 - \epsilon + \frac{\epsilon}{K}$, $y_{other} = \frac{\epsilon}{K}$ where $K$ is the number of classes and $\epsilon$ is typically $0.1$.

**Why label smoothing helps**:
1. Prevents the model from becoming overconfident (logits don't need to be infinite)
2. Acts as regularization -- encourages the model to maintain some uncertainty
3. Improves calibration -- model's confidence better reflects actual accuracy

**Used in**: The original Transformer paper, T5, many LLM training recipes. GPT-3 training used label smoothing.

### 9.7 Loss Functions in Modern LLM Training

| Training Stage | Loss Function | Details |
|---------------|---------------|---------|
| Pretraining (CLM) | Cross-entropy | Next token prediction |
| Pretraining (MLM) | Cross-entropy | Masked token prediction |
| SFT | Cross-entropy | On instruction-response pairs |
| RLHF - Reward Model | Cross-entropy / ranking loss | On human preference pairs |
| RLHF - PPO | Policy gradient + KL penalty | $\max E[R] - \beta D_{KL}(\pi_\theta \| \pi_{ref})$ |
| DPO | Modified cross-entropy | $-\log\sigma(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})$ |
| Diffusion | MSE | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| CLIP | InfoNCE | Contrastive image-text alignment |
| Embedding models | Contrastive / Triplet | Semantic similarity learning |

---

## 10. Interview Questions & Answers

### Q1: Why do transformers use Layer Normalization instead of Batch Normalization?

**Answer**: There are four key reasons:

1. **Batch size independence**: LayerNorm computes statistics per-sample across features. BatchNorm computes statistics across the batch for each feature. In LLM training, effective batch sizes per GPU can be small, making BatchNorm statistics unreliable. At inference (especially autoregressive generation), batch size is often 1.

2. **Variable sequence lengths**: Sequences have different lengths. BatchNorm would mix statistics from meaningful tokens and padding tokens, or couple representations at the same position across different sequences.

3. **Autoregressive generation compatibility**: During generation, tokens are produced one at a time. BatchNorm requires a batch of samples to compute statistics, which is undefined for single-token generation.

4. **Empirical results**: LayerNorm consistently outperforms BatchNorm for transformer-based language models. Ba et al. (2016) showed LayerNorm works better for RNN/transformer architectures.

### Q2: Explain the vanishing gradient problem. How do residual connections solve it?

**Answer**: In a deep network, gradients flow backward through repeated multiplication by layer Jacobians. If each Jacobian has spectral norm < 1, the gradient shrinks exponentially with depth: $\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_l}\| \propto \prod_{k=l}^{L} \|J_k\|$. For sigmoid activations (max derivative 0.25), this shrinkage is severe.

Residual connections change the Jacobian of each layer from $J_k$ to $I + J_k$. The gradient from layer $L$ to layer $l$ becomes $\prod_{k}(I + J_k)$. When expanded, this product always includes a term $I^{L-l} = I$ (the product of identity matrices from the skip connections). This means the gradient always has a direct, unmultiplied path from the loss to every layer. Even if $J_k$ has small norms, the identity term ensures non-vanishing gradients.

This is why GPT-3 with 96 layers can train successfully -- the residual stream provides a gradient highway from the output to the earliest layers.

### Q3: Why GELU over ReLU in transformers? What about SwiGLU in Llama?

**Answer**: **GELU vs ReLU**: GELU ($z \cdot \Phi(z)$) is a smooth, differentiable approximation of ReLU with several advantages for transformer training:
- No dying neuron problem (GELU is never flat zero for negative inputs)
- Smooth gradient transitions near zero (no kink), which helps optimization with Adam
- Small negative outputs allow gradient flow through negative activations
- Empirically outperforms ReLU on language modeling benchmarks (confirmed in BERT, GPT-2, GPT-3)

**SwiGLU in Llama**: SwiGLU replaces the standard two-matrix FFN with a gated architecture using three matrices: $(\text{Swish}(xW_1) \otimes xV)W_2$. The gating mechanism allows the network to modulate information flow -- one projection produces "content" while another produces "gates" that control which content passes through. Shazeer (2020) showed GLU variants improve language modeling perplexity. The trade-off is 50% more parameters per FFN, compensated by reducing $d_{ff}$ from $4d_{model}$ to $\frac{8}{3}d_{model}$.

### Q4: What happens if you remove residual connections from a 96-layer transformer?

**Answer**: The network becomes effectively untrainable. Specifically:

1. **Gradient vanishing**: Without the skip connections, gradients must pass through 96 layers of attention and FFN Jacobians. Even with GELU and LayerNorm, the compounding effect means early layers (0-80) receive essentially zero gradient.

2. **Effective depth collapse**: Only the last ~10-15 layers learn meaningful representations. The first 80+ layers are wasted parameters that contribute nothing.

3. **Loss plateau**: Training loss barely decreases from its initial value, because the vast majority of parameters receive no useful learning signal.

4. **Initialization sensitivity**: Without residuals, the network is extremely sensitive to initialization. Small changes in init can mean the difference between gradients of $10^{-3}$ and $10^{-30}$ at early layers.

The fix is not just "use residuals" -- it's understanding that residual connections fundamentally change the optimization landscape. They convert a deep network from a serial chain (where gradient must survive L multiplications) into a parallel ensemble (where gradient has $2^L$ possible paths through the network).

### Q5: Derive the gradient of cross-entropy loss with respect to logits.

**Answer**: Given softmax probabilities $\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$ and cross-entropy loss $\mathcal{L} = -\log(\hat{y}_k)$ where $k$ is the correct class:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial (-\log \hat{y}_k)}{\partial z_i}$$

We need $\frac{\partial \hat{y}_k}{\partial z_i}$. The softmax Jacobian is:

$$\frac{\partial \hat{y}_k}{\partial z_i} = \begin{cases} \hat{y}_k(1 - \hat{y}_k) & \text{if } i = k \\ -\hat{y}_k \hat{y}_i & \text{if } i \neq k \end{cases} = \hat{y}_k(\delta_{ik} - \hat{y}_i)$$

Therefore:

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{1}{\hat{y}_k} \cdot \hat{y}_k(\delta_{ik} - \hat{y}_i) = \hat{y}_i - \delta_{ik}$$

In vector form: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$

This is remarkably simple: the gradient is just the predicted probability minus the true label. For the correct class, the gradient is $(\hat{y}_k - 1)$, pushing the logit up. For incorrect classes, the gradient is $\hat{y}_i$, pushing their logits down. This simplicity is one reason cross-entropy + softmax is so widely used.

### Q6: Compare Xavier and He initialization. When do you use each?

**Answer**:

**Xavier (Glorot) initialization**: $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$
- Derived assuming linear or near-linear activations (sigmoid, tanh in their linear region)
- Preserves variance in both forward and backward passes (a compromise between $\frac{1}{n_{in}}$ for forward and $\frac{1}{n_{out}}$ for backward)
- Use for: sigmoid, tanh, linear layers, attention projections

**He (Kaiming) initialization**: $\text{Var}(w) = \frac{2}{n_{in}}$
- Derived accounting for ReLU zeroing half the outputs (variance halved)
- Only considers forward pass variance preservation
- Use for: ReLU, Leaky ReLU, PReLU, and networks with ReLU variants

**In practice for transformers**: Neither is used directly. GPT-style models use $\mathcal{N}(0, 0.02)$ with depth-dependent scaling ($\frac{1}{\sqrt{2L}}$) for residual projections. The specific constant (0.02) was found empirically and works for the typical transformer hidden sizes (768-12288).

### Q7: What is inverted dropout? Why is it preferred?

**Answer**: Standard dropout zeroes neurons with probability $p$ during training. This changes the expected magnitude of layer outputs: training expects $(1-p) \cdot h$, but inference sees $h$. 

**Naive fix**: Scale at test time by $(1-p)$. Requires modifying inference code, error-prone.

**Inverted dropout**: Scale at training time by $\frac{1}{1-p}$: the surviving activations are divided by $(1-p)$, so the expected value equals $h$ during training. At inference, no scaling needed -- the model works identically with or without dropout disabled.

This is preferred because: (1) inference code is unchanged, (2) no risk of forgetting to scale at deployment, (3) the model can be exported as-is for production.

### Q8: Why does Pre-LN work better than Post-LN for training deep transformers?

**Answer**: In Post-LN ($\text{LN}(x + \text{Sublayer}(x))$), the gradient must pass through LayerNorm at every layer. LayerNorm's Jacobian depends on the input statistics, and these can vary unpredictably during early training, creating gradient instability. The gradient norms across layers can vary by orders of magnitude.

In Pre-LN ($x + \text{Sublayer}(\text{LN}(x))$), the residual connection provides a direct path that bypasses LayerNorm entirely. The gradient through the skip connection is exactly the identity: $\frac{\partial(x + f(\text{LN}(x)))}{\partial x} = I + \frac{\partial f}{\partial x}$. The identity component ensures stable gradient magnitude regardless of what happens inside the sublayer.

Xiong et al. (2020) proved that Pre-LN keeps gradient norms bounded at initialization, while Post-LN gradient norms grow with depth. Practically, Pre-LN eliminates the need for careful learning rate warmup and enables training of very deep transformers (100+ layers) that Post-LN cannot train stably.

### Q9: Explain backpropagation through the attention mechanism.

**Answer**: Attention computes $\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V = AV$ where $A$ is the attention matrix.

The backward pass must compute gradients w.r.t. $Q$, $K$, and $V$:

1. **Gradient w.r.t. V**: $\frac{\partial \mathcal{L}}{\partial V} = A^T \frac{\partial \mathcal{L}}{\partial \text{output}}$ -- straightforward linear gradient.

2. **Gradient w.r.t. A**: $\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial \text{output}} \cdot V^T$ -- again straightforward.

3. **Gradient through softmax**: This is where it gets interesting. For each row of the pre-softmax scores $S = \frac{QK^T}{\sqrt{d_k}}$, the softmax Jacobian is $\text{diag}(a) - aa^T$ where $a$ is the softmax output. This gradient can become very small when attention is "sharp" (one score much larger than others), potentially causing slow learning.

4. **Gradient w.r.t. Q and K**: After passing through softmax, $\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot \frac{K}{\sqrt{d_k}}$ and $\frac{\partial \mathcal{L}}{\partial K} = \frac{\partial \mathcal{L}}{\partial S}^T \cdot \frac{Q}{\sqrt{d_k}}$.

The scaling by $\frac{1}{\sqrt{d_k}}$ is critical: without it, the dot products $QK^T$ grow with $d_k$, pushing softmax into saturation where its gradient is nearly zero. This would effectively block gradient flow through attention.

### Q10: How does gradient checkpointing work and why is it necessary for LLM training?

**Answer**: Standard backpropagation stores all intermediate activations from the forward pass (needed to compute gradients during backward pass). For a transformer with $L$ layers, sequence length $n$, and hidden size $d$, this requires $O(L \cdot n \cdot d)$ memory -- often 10-100+ GB for large models.

**Gradient checkpointing** (Chen et al., 2016): Instead of storing all activations, only store activations at checkpointed layers (e.g., every $\sqrt{L}$ layers). During the backward pass, recompute the activations between checkpoints on-the-fly.

**Trade-off**: Memory reduced from $O(L)$ to $O(\sqrt{L})$ (with optimal checkpoint placement), at the cost of ~33% more compute (each activation is computed twice -- once in forward, once during backward recomputation).

**Why it's necessary**: Training a 70B parameter model with sequence length 4096 and hidden size 8192 would require storing activations for 80 layers, each with $4096 \times 8192$ tensors plus attention matrices ($4096 \times 4096$). Without checkpointing, this exceeds GPU memory. With checkpointing, the memory footprint becomes manageable across multiple GPUs.

All major LLM training frameworks (Megatron-LM, DeepSpeed, FSDP) use gradient checkpointing by default.

---

*Next topic: [Topic 5: Tokenization](../Phase_B_NLP_Core/05_Tokenization.md)*
