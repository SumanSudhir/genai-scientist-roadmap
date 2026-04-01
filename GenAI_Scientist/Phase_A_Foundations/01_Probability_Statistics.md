# Topic 1: Probability, Statistics & Information Theory

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 1 of 28
> **Scope**: Probability fundamentals, distributions, estimation, information theory, hypothesis testing, A/B testing
> **Why this matters**: Cross-entropy is your training loss. KL divergence is in RLHF, VAEs, and distillation. MLE is how every language model is trained. This is the mathematical language of modern AI.
> **Next**: [Topic 2: Linear Algebra & Optimization](02_Linear_Algebra_Optimization.md)

---

## Table of Contents

1. [Probability Fundamentals](#1-probability-fundamentals)
2. [Common Probability Distributions](#2-common-probability-distributions)
3. [Expectation, Variance & Covariance](#3-expectation-variance--covariance)
4. [Maximum Likelihood Estimation (MLE)](#4-maximum-likelihood-estimation-mle)
5. [Maximum A Posteriori (MAP)](#5-maximum-a-posteriori-map)
6. [Information Theory](#6-information-theory)
7. [Connecting Information Theory to ML](#7-connecting-information-theory-to-ml)
8. [Hypothesis Testing](#8-hypothesis-testing)
9. [A/B Testing for AI Systems](#9-ab-testing-for-ai-systems)
10. [Interview Questions & Answers](#10-interview-questions--answers)

---

## 1. Probability Fundamentals

### 1.1 Why Probability Is the Foundation of ML

Every machine learning model is, at its core, a probability machine. When GPT generates text, it's sampling from a probability distribution over the vocabulary at each step:

$$P(\text{next\_token} \mid \text{all\_previous\_tokens})$$

When BERT is trained with masked language modeling, it's maximizing the probability of the masked token given the surrounding context. When we train with cross-entropy loss, we're doing maximum likelihood estimation. When we do RLHF, the KL divergence penalty prevents the policy from drifting too far from the reference model.

Without a solid grasp of probability, everything downstream becomes pattern memorization instead of understanding.

### 1.2 Basic Probability Rules

**Sample Space ($\Omega$)**: The set of all possible outcomes.
- Rolling a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Next token prediction: $\Omega = \{\text{all tokens in vocabulary}\}$

**Event**: A subset of the sample space. "Rolling an even number" = $\{2, 4, 6\}$.

**Axioms of Probability** (Kolmogorov):
1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

**Addition Rule** (general):

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Conditional Probability**:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0$$

This is the foundation of language modeling. The probability of a word given all previous words is conditional probability.

**Independence**: Two events A and B are independent if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently: $P(A \mid B) = P(A)$. Knowing B happened tells you nothing about A.

**Conditional Independence**: A and B are conditionally independent given C if:

$$P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

This is critical in graphical models and Naive Bayes. Naive Bayes assumes features are conditionally independent given the class label -- an assumption that's almost always wrong but often works surprisingly well.

### 1.3 Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In ML terminology:

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}$$

Where:
- $P(\theta \mid \text{data})$ = Posterior: what we believe about the parameters after seeing data
- $P(\text{data} \mid \theta)$ = Likelihood: how probable the data is under these parameters
- $P(\theta)$ = Prior: what we believed before seeing data
- $P(\text{data})$ = Evidence (marginal likelihood): normalizing constant, often intractable

**Why Bayes matters for GenAI**:
- Language modeling is inherently Bayesian: given all previous tokens (data), what's the probability of the next token (parameter of interest)?
- MAP estimation adds a prior to MLE -- this is exactly what regularization does (L2 regularization = Gaussian prior on weights)
- Bayesian interpretations of in-context learning suggest LLMs perform implicit Bayesian inference

**Example: Spam Classification (Bayesian reasoning)**

You want $P(\text{spam} \mid \text{email\_text})$. Direct estimation is hard. Bayes helps:

$$P(\text{spam} \mid \text{text}) = \frac{P(\text{text} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{text})}$$

- $P(\text{text} \mid \text{spam})$: How likely this text is if it IS spam. Estimable from spam training data.
- $P(\text{spam})$: Prior probability of spam (e.g., 30% of all emails).
- $P(\text{text})$: Normalizing constant.

This is literally how Naive Bayes classifiers work.

### 1.4 The Chain Rule of Probability

$$P(x_1, x_2, \ldots, x_n) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_n \mid x_1, \ldots, x_{n-1})$$

Or more compactly:

$$P(x_1, \ldots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \ldots, x_{t-1})$$

**This is the autoregressive language modeling objective**. Every GPT-style model factorizes the joint probability of a sequence using this exact chain rule, predicting one token at a time, left to right.

When GPT generates "The cat sat on the mat", it computes:

$$P(\text{"The"}) \cdot P(\text{"cat"} \mid \text{"The"}) \cdot P(\text{"sat"} \mid \text{"The cat"}) \cdot P(\text{"on"} \mid \text{"The cat sat"}) \cdots$$

This is not an approximation -- it's an exact factorization of the joint probability via the chain rule.

### 1.5 Marginalization

To get the probability of a single variable from a joint distribution, sum (or integrate) over the others:

**Discrete**:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

**Continuous**:

$$p(x) = \int p(x, y) \, dy$$

**Where it shows up**: The evidence term $P(\text{data})$ in Bayes' theorem requires marginalizing over all possible parameter values, which is often intractable. This intractability is why we use techniques like variational inference (VAEs) and MCMC.

---

## 2. Common Probability Distributions

### 2.1 Discrete Distributions

**Bernoulli Distribution**: Single binary trial.

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

$$\text{Mean: } p, \quad \text{Variance: } p(1-p)$$

Used in: Binary classification outputs, dropout (each neuron is a Bernoulli trial).

**Categorical Distribution (Multinoulli)**: Single trial with K possible outcomes.

$$P(X = k) = p_k, \quad \text{where } \sum_{k=1}^{K} p_k = 1$$

This is a single roll of a K-sided die.

Used in: **Every language model output**. The softmax layer produces a categorical distribution over the vocabulary. When GPT outputs logits and applies softmax, the result is a categorical distribution over ~50K tokens.

**Binomial Distribution**: Number of successes in n independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

$$\text{Mean: } np, \quad \text{Variance: } np(1-p)$$

**Poisson Distribution**: Number of events in a fixed interval, when events occur at a constant average rate.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

$$\text{Mean: } \lambda, \quad \text{Variance: } \lambda$$

Used in: Modeling rare events, word frequency in some NLP contexts.

**Multinomial Distribution**: Extension of binomial to K categories over n trials.

$$P(X_1 = x_1, \ldots, X_K = x_K) = \frac{n!}{x_1! \cdots x_K!} \, p_1^{x_1} \cdots p_K^{x_K}$$

Used in: Bag-of-words models, topic models (LDA).

### 2.2 Continuous Distributions

**Gaussian (Normal) Distribution**: The most important distribution in ML.

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

$$\text{Mean: } \mu, \quad \text{Variance: } \sigma^2$$

Why it's everywhere:
- **Central Limit Theorem**: Sums of many independent random variables tend toward Gaussian, regardless of their individual distributions. This is why gradient noise is approximately Gaussian.
- **Weight initialization**: Xavier and He initialization draw weights from Gaussian distributions.
- **L2 regularization**: Equivalent to a Gaussian prior on weights (we'll prove this in Section 5).
- **Latent spaces**: VAEs assume Gaussian latent distributions.
- **Noise in diffusion models**: Forward process adds Gaussian noise.

**Multivariate Gaussian**:

$$p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

Where $\boldsymbol{\mu}$ is the mean vector and $\Sigma$ is the covariance matrix.

**Uniform Distribution**: All values equally likely.

$$p(x) = \frac{1}{b - a} \quad \text{for } a \leq x \leq b$$

$$\text{Mean: } \frac{a + b}{2}, \quad \text{Variance: } \frac{(b - a)^2}{12}$$

Used in: Random initialization of some parameters, random sampling strategies.

**Exponential Distribution**: Time between events in a Poisson process.

$$p(x) = \lambda e^{-\lambda x} \quad \text{for } x \geq 0$$

### 2.3 The Softmax Function as a Distribution

The softmax function transforms a vector of real numbers (logits) into a valid probability distribution:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Properties:
- Output is always positive (exponential ensures this)
- Outputs sum to 1 (normalization ensures this)
- Preserves ordering (larger logits -> larger probabilities)
- Temperature parameter controls sharpness (more on this in Topic 17: Decoding)

$$\text{softmax}(z_i / T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T \to 0$: argmax (greedy, one-hot)
- $T = 1$: standard softmax
- $T \to \infty$: uniform distribution

This is used in:
- **Every language model's output layer**: logits -> softmax -> categorical distribution over vocabulary
- **Attention mechanism**: the attention scores go through softmax to become attention weights
- **Classification**: final layer of any multi-class classifier

---

## 3. Expectation, Variance & Covariance

### 3.1 Expectation (Mean)

The expected value is the "average" outcome, weighted by probability.

**Discrete**:

$$E[X] = \sum_x x \cdot P(X = x)$$

**Continuous**:

$$E[X] = \int x \cdot p(x) \, dx$$

**Key properties** (linearity):

$$E[aX + b] = a \cdot E[X] + b$$

$$E[X + Y] = E[X] + E[Y] \quad \text{(always, even if X and Y are dependent!)}$$

**Where it shows up**: The training loss is an expectation. Cross-entropy loss over a dataset is:

$$\mathcal{L} = \mathbb{E}_{(x,y) \sim \text{data}} [-\log P(y \mid x; \theta)]$$

We approximate this expectation with the empirical average over a mini-batch.

### 3.2 Variance

Measures the spread/uncertainty of a random variable.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Key properties**:

$$\text{Var}(aX + b) = a^2 \cdot \text{Var}(X)$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2 \cdot \text{Cov}(X, Y)$$

If $X, Y$ independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

**Standard deviation**: $\sigma = \sqrt{\text{Var}(X)}$. Same units as $X$.

**Where it shows up**:
- **Bias-variance tradeoff**: Total error $= \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$
- **Weight initialization**: Xavier init sets $\text{Var}(\text{weights}) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$ to maintain variance across layers
- **Batch normalization**: Normalizes to unit variance to stabilize training
- **Gradient variance**: High variance in gradients -> noisy updates -> slow or unstable training

### 3.3 Covariance & Correlation

**Covariance**: Measures how two variables vary together.

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X] \cdot E[Y]$$

- $\text{Cov} > 0$: X and Y tend to increase together
- $\text{Cov} < 0$: When X increases, Y tends to decrease
- $\text{Cov} = 0$: No linear relationship (but could still be dependent!)

**Correlation**: Normalized covariance, between -1 and 1.

$$\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

**Covariance Matrix**: For a vector $\mathbf{X} = (X_1, \ldots, X_d)$:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

This is a $d \times d$ symmetric positive semi-definite matrix. PCA finds the eigenvectors of this matrix to identify the directions of maximum variance.

---

## 4. Maximum Likelihood Estimation (MLE)

### 4.1 The Core Idea

Given observed data $D = \{x_1, \ldots, x_n\}$, find the parameters $\theta$ that make the data most probable:

$$\theta_{MLE} = \arg\max_\theta P(D \mid \theta) = \arg\max_\theta \prod_{i=1}^{n} P(x_i \mid \theta) \quad \text{(assuming i.i.d.)}$$

Taking the log (monotonic, so doesn't change the argmax):

$$\theta_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i \mid \theta)$$

We maximize the log-likelihood, or equivalently, **minimize the negative log-likelihood (NLL)**:

$$\theta_{MLE} = \arg\min_\theta -\sum_{i=1}^{n} \log P(x_i \mid \theta)$$

### 4.2 MLE for a Gaussian

Given data $\{x_1, \ldots, x_n\}$ from a Gaussian:

$$\mathcal{L}(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \sum \frac{(x_i - \mu)^2}{2\sigma^2}$$

Taking derivatives and setting to zero:

$$\mu_{MLE} = \frac{1}{n} \sum x_i \quad \text{(the sample mean)}$$

$$\sigma^2_{MLE} = \frac{1}{n} \sum (x_i - \mu_{MLE})^2 \quad \text{(the sample variance -- note: biased estimator)}$$

### 4.3 MLE Is How Language Models Are Trained

A language model with parameters $\theta$ assigns probability $P(x_t \mid x_{<t}; \theta)$ to each token given its context. For a training corpus of sequences:

$$\theta_{MLE} = \arg\max_\theta \sum_{\text{sequences}} \sum_{t} \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

The negative of this is the **cross-entropy loss**:

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t}; \theta)$$

So when you hear "we train a language model with cross-entropy loss," you're hearing "we do MLE." They are the same thing.

### 4.4 Properties of MLE

**Consistency**: As $n \to \infty$, $\theta_{MLE} \to \theta_{\text{true}}$ (the true parameters). Good.

**Asymptotic efficiency**: Among consistent estimators, MLE achieves the lowest variance (Cramer-Rao bound) as $n \to \infty$.

**Can overfit**: With limited data, MLE can overfit. No regularization is built in. This motivates MAP estimation (Section 5).

**Invariance**: If $\theta_{MLE}$ is the MLE of $\theta$, then $f(\theta_{MLE})$ is the MLE of $f(\theta)$ for any function $f$.

---

## 5. Maximum A Posteriori (MAP)

### 5.1 From MLE to MAP

MLE only considers how well the parameters explain the data. MAP adds a prior belief about what the parameters should look like:

$$\theta_{MAP} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta P(D \mid \theta) \cdot P(\theta) \quad \text{(} P(D) \text{ is a constant w.r.t. } \theta \text{)}$$

In log form:

$$\theta_{MAP} = \arg\max_\theta \left[\log P(D \mid \theta) + \log P(\theta)\right] = \arg\max_\theta \left[\text{log-likelihood} + \text{log-prior}\right]$$

### 5.2 MAP = MLE + Regularization

This is one of the most important connections in all of ML.

**L2 Regularization = Gaussian Prior**:

If we set the prior $P(\theta) = \mathcal{N}(0, \sigma^2 I)$, then:

$$\log P(\theta) = -\frac{\|\theta\|^2}{2\sigma^2} + \text{constant}$$

So the MAP objective becomes:

$$\theta_{MAP} = \arg\max_\theta \left[\log P(D \mid \theta) - \lambda \|\theta\|^2\right]$$

where $\lambda = \frac{1}{2\sigma^2}$. This is exactly L2-regularized MLE (Ridge regression)!

A tight prior (small $\sigma^2$, large $\lambda$) means we strongly believe weights should be near zero -> heavy regularization.

**L1 Regularization = Laplace Prior**:

If we set the prior $P(\theta) = \text{Laplace}(0, b)$, then:

$$\log P(\theta) = -\frac{\|\theta\|_1}{b} + \text{constant}$$

The MAP objective becomes:

$$\theta_{MAP} = \arg\max_\theta \left[\log P(D \mid \theta) - \lambda \|\theta\|_1\right]$$

This is L1-regularized MLE (Lasso). The Laplace prior has sharp peaks at zero, which is why L1 produces sparse solutions -- it literally encodes the belief that most parameters should be exactly zero.

### 5.3 Why This Matters for GenAI

- **Weight decay in AdamW** is L2 regularization, which is a Gaussian prior on weights
- **Dropout** can be interpreted as approximate Bayesian inference
- **The KL term in RLHF**: $\beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ acts as a prior keeping the aligned model close to the reference -- this is structurally identical to MAP estimation where the prior is the reference policy

---

## 6. Information Theory

Information theory, developed by Claude Shannon, provides the mathematical language for quantifying information, uncertainty, and the difference between distributions. In GenAI, it's not abstract -- it's your loss function.

### 6.1 Self-Information (Surprisal)

How surprised are you when event $x$ occurs?

$$I(x) = -\log P(x)$$

- **Certain event** ($P(x) = 1$): $I(x) = 0$. No surprise.
- **Unlikely event** ($P(x) = 0.01$): $I(x) = -\log(0.01) = 6.64$ bits (using log base 2). Very surprising.
- **Impossible event** ($P(x) \to 0$): $I(x) \to \infty$. Infinitely surprising.

The log ensures that the information from two independent events is additive:

$$I(x, y) = I(x) + I(y) \quad \text{when } x, y \text{ are independent}$$

**In NLP**: When a language model assigns low probability to the actual next token, the surprisal is high. This is exactly what perplexity measures (more on this later).

### 6.2 Entropy

Entropy is the **expected surprisal** -- the average amount of information (or uncertainty) in a random variable:

$$H(X) = E[-\log P(X)] = -\sum_x P(x) \log P(x)$$

**Key properties**:
- **Non-negative**: $H(X) \geq 0$
- **Maximum at uniform distribution**: For $K$ outcomes, $H$ is maximized when $P(x) = 1/K$ for all $x$, giving $H = \log(K)$
- **Zero for deterministic variables**: If $P(x) = 1$ for some $x$, then $H = 0$
- **Units**: bits (log base 2) or nats (natural log)

**Intuition**: Entropy measures how unpredictable a random variable is.
- A fair coin has $H = 1$ bit (maximum uncertainty for 2 outcomes)
- A biased coin ($P(H) = 0.99$) has $H \approx 0.08$ bits (almost certain, low entropy)
- A uniform distribution over 50,000 tokens has $H = \log_2(50000) \approx 15.6$ bits

**In NLP**: The entropy of natural language (English) is estimated at ~1.0-1.5 bits per character. A perfect language model would achieve this entropy. Current models are approaching but haven't reached this limit.

### 6.3 Cross-Entropy

Cross-entropy measures the average number of bits needed to encode data from distribution $P$ using a code optimized for distribution $Q$:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Or in expectation notation:

$$H(P, Q) = \mathbb{E}_{x \sim P} [-\log Q(x)]$$

**Key relationship**:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

Since $D_{KL} \geq 0$, cross-entropy is always $\geq$ entropy: $H(P, Q) \geq H(P)$. Equality holds when $Q = P$.

**Why cross-entropy is the standard loss for classification and language models**:

In training, $P$ is the true distribution (one-hot for the correct token) and $Q$ is the model's predicted distribution. Minimizing cross-entropy $H(P, Q)$ is equivalent to minimizing KL divergence $D_{KL}(P \| Q)$ because $H(P)$ is a constant (doesn't depend on model parameters):

$$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta \left[H(P) + D_{KL}(P \| Q_\theta)\right] = \arg\min_\theta D_{KL}(P \| Q_\theta)$$

So minimizing cross-entropy = minimizing KL divergence from the true distribution = doing MLE.

**For a one-hot true distribution** (the correct token has $P = 1$, all others have $P = 0$):

$$H(P, Q) = -\log Q(\text{correct\_token})$$

This reduces to the negative log-likelihood of the correct token -- exactly the NLL loss used in language model training.

### 6.4 KL Divergence (Kullback-Leibler Divergence)

KL divergence measures how different distribution $Q$ is from distribution $P$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)] = H(P, Q) - H(P)$$

**Key properties**:

**Non-negative**: $D_{KL}(P \| Q) \geq 0$ (Gibbs' inequality). Equals zero iff $P = Q$.

**Asymmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general. This is NOT a distance metric!

The asymmetry matters enormously in practice:

**$D_{KL}(P \| Q)$ -- "Forward KL" or "M-projection"**:
- Computed by sampling from $P$ (the true distribution)
- Penalizes $Q$ heavily for assigning low probability where $P$ has high probability
- Tends to produce a $Q$ that is **mean-seeking**: $Q$ tries to cover all modes of $P$
- This is what we minimize in MLE / cross-entropy training

**$D_{KL}(Q \| P)$ -- "Reverse KL" or "I-projection"**:
- Computed by sampling from $Q$ (the model)
- Penalizes $Q$ for assigning high probability where $P$ has low probability
- Tends to produce a $Q$ that is **mode-seeking**: $Q$ locks onto one mode of $P$
- This is used in variational inference (VAEs) and some RL objectives

**Where KL divergence appears in GenAI**:

1. **Cross-entropy loss**: Minimizing $H(P, Q)$ = minimizing $D_{KL}(P \| Q)$ (since $H(P)$ is constant)

2. **RLHF objective**:

$$\max \mathbb{E}[R(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

   The KL term prevents the RLHF-trained policy from drifting too far from the SFT model. Without it, the model would overfit to the reward model.

3. **VAE loss (ELBO)**:

$$\mathcal{L} = \mathbb{E}[\log P(x \mid z)] - D_{KL}(Q(z \mid x) \| P(z))$$

   KL divergence regularizes the latent distribution to stay close to the prior.

4. **Knowledge distillation**:

$$\mathcal{L}_{\text{distill}} = D_{KL}(P_{\text{teacher}} \| P_{\text{student}})$$

   The student learns to match the teacher's output distribution.

### 6.5 Mutual Information

Mutual information measures how much knowing one variable tells you about another:

$$I(X; Y) = D_{KL}(P(X, Y) \| P(X) P(Y)) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$$

- $I(X; Y) = 0$ iff $X$ and $Y$ are independent
- $I(X; Y) \geq 0$ always
- Symmetric: $I(X; Y) = I(Y; X)$

**Intuition**: How much does knowing $Y$ reduce your uncertainty about $X$?

**Where it shows up**:
- Feature selection: Choose features with highest mutual information with the target
- Representation learning: Maximize mutual information between input and learned representation (e.g., contrastive learning objectives like InfoNCE in CLIP)
- Information bottleneck: Trade-off between compression and prediction

### 6.6 Jensen-Shannon Divergence (JSD)

A symmetric, bounded version of KL divergence:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$$

$$\text{where } M = \frac{1}{2}(P + Q)$$

Properties:
- Symmetric: $\text{JSD}(P \| Q) = \text{JSD}(Q \| P)$
- Bounded: $0 \leq \text{JSD} \leq \log(2)$ (using natural log) or $0 \leq \text{JSD} \leq 1$ (using log base 2)
- Its square root is a proper distance metric

Used in: The original GAN training objective is related to JSD between real and generated distributions.

---

## 7. Connecting Information Theory to ML

This section ties together the concepts above into a unified view of how they drive machine learning.

### 7.1 The Chain: MLE = Minimizing NLL = Minimizing Cross-Entropy = Minimizing KL Divergence

Let's trace this equivalence carefully.

**Start with MLE**:

$$\theta_{MLE} = \arg\max_\theta \sum_i \log P(x_i \mid \theta) = \arg\max_\theta \frac{1}{n} \sum_i \log P(x_i \mid \theta) = \arg\max_\theta \mathbb{E}_{x \sim P_{\text{data}}} [\log P_\theta(x)]$$

**Convert to minimization** (flip the sign):

$$= \arg\min_\theta \mathbb{E}_{x \sim P_{\text{data}}} [-\log P_\theta(x)] = \arg\min_\theta H(P_{\text{data}}, P_\theta)$$

**Expand cross-entropy**:

$$H(P_{\text{data}}, P_\theta) = H(P_{\text{data}}) + D_{KL}(P_{\text{data}} \| P_\theta)$$

Since $H(P_{\text{data}})$ doesn't depend on $\theta$:

$$\arg\min_\theta H(P_{\text{data}}, P_\theta) = \arg\min_\theta D_{KL}(P_{\text{data}} \| P_\theta)$$

**Summary**: MLE = minimize NLL = minimize cross-entropy = minimize KL divergence from data to model. Four ways of saying the same thing.

### 7.2 Perplexity

Perplexity is the go-to metric for evaluating language models. It's the exponential of the cross-entropy:

$$\text{PPL} = \exp\left(H(P_{\text{data}}, P_{\text{model}})\right) = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log P_{\text{model}}(x_t \mid x_{<t})\right)$$

**Intuition**: Perplexity represents the effective number of tokens the model is "confused" between at each step.

- PPL = 1: The model perfectly predicts every token (impossible in practice)
- PPL = 50: On average, the model is as uncertain as if choosing uniformly from 50 tokens
- PPL = V (vocab size): The model is no better than random guessing

**Lower perplexity = better model.** But perplexity alone doesn't measure factuality, helpfulness, or safety -- which is why we also need human evaluation and task-specific metrics.

### 7.3 The Information-Theoretic View of Attention

The attention mechanism can be viewed through information theory:

- Attention weights represent how much information each token contributes to the representation of the current token
- High attention weight = high mutual information between those positions
- The softmax in attention produces a valid probability distribution, and the weighted sum is an expectation under that distribution:

$$\text{Attention output} = \mathbb{E}_{j \sim \text{attention\_weights}} [V_j]$$

---

## 8. Hypothesis Testing

### 8.1 The Framework

Hypothesis testing is the formal framework for making data-driven decisions under uncertainty.

**Null Hypothesis ($H_0$)**: The default assumption (e.g., "the two models perform equally")
**Alternative Hypothesis ($H_1$)**: What we want to show (e.g., "model A is better than model B")

**Procedure**:
1. Assume $H_0$ is true
2. Compute a test statistic from the data
3. Calculate the p-value: the probability of observing a result at least as extreme as the data, assuming $H_0$ is true
4. If p-value $< \alpha$ (significance level, typically 0.05), reject $H_0$

### 8.2 Key Concepts

**p-value**: $P(\text{data this extreme or more} \mid H_0 \text{ is true})$. NOT the probability that $H_0$ is true.

Common misinterpretation: "p = 0.03 means there's a 3% chance the null is true." WRONG. It means: "If the null were true, there's a 3% chance of seeing data this extreme."

**Significance level ($\alpha$)**: The threshold for rejection, typically 0.05 or 0.01. This is the Type I error rate you're willing to accept.

**Type I Error (False Positive)**: Rejecting $H_0$ when it's actually true. Rate $= \alpha$.
**Type II Error (False Negative)**: Failing to reject $H_0$ when $H_1$ is true. Rate $= \beta$.

**Power** $= 1 - \beta$: The probability of correctly detecting a real effect. Higher power is better.

**Factors that increase power**:
- Larger sample size
- Larger effect size
- Higher significance level ($\alpha$)
- Lower variance

### 8.3 Common Tests

**t-test**: Compare means of two groups. Assumes roughly Gaussian data.
- **Paired t-test**: Before/after on the same subjects (e.g., same users with old vs new model)
- **Independent t-test**: Two separate groups

**Chi-squared test**: Test independence of categorical variables.

**Bootstrap test**: Resample data with replacement to estimate the distribution of a statistic. Non-parametric -- makes no distributional assumptions.

### 8.4 Multiple Testing Problem

If you test 20 hypotheses at $\alpha = 0.05$, you expect 1 false positive purely by chance.

**Corrections**:
- **Bonferroni**: Divide $\alpha$ by the number of tests. Conservative.
- **Benjamini-Hochberg (FDR control)**: Controls the false discovery rate. Less conservative, more commonly used in practice.

---

## 9. A/B Testing for AI Systems

### 9.1 Why A/B Testing Matters for GenAI

You've trained a new LLM variant or RAG system. Offline metrics (perplexity, BLEU, ROUGE) look better. Should you deploy it? You can't know for sure without an A/B test on real users.

A/B testing is the gold standard for measuring the causal effect of a change on real-world metrics.

### 9.2 Designing an A/B Test

**Step 1: Define metrics**
- **Primary metric** (guardrail): The ONE metric that determines success (e.g., user satisfaction, task completion rate)
- **Secondary metrics**: Additional signals (e.g., latency, cost per query, hallucination rate)
- **Guardrail metrics**: Must not degrade (e.g., safety violations, crash rate)

**Step 2: Determine sample size**

The required sample size depends on:
- **Minimum Detectable Effect (MDE)**: The smallest improvement you care about (e.g., 2% improvement in click-through rate)
- **Baseline metric value**: The current performance level
- **Variance**: Higher variance requires more samples
- **Significance level ($\alpha$)**: Usually 0.05
- **Power ($1 - \beta$)**: Usually 0.80 or 0.90

For a two-sample proportion test:

$$n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \left[p_1(1-p_1) + p_2(1-p_2)\right]}{(p_2 - p_1)^2}$$

**Step 3: Randomization**
- Randomly assign users (not requests) to control (A) and treatment (B)
- User-level randomization avoids contamination from the same user seeing both variants
- Stratify by important confounders (geography, device type)

**Step 4: Run the test**
- Run for a pre-determined duration (don't peek and stop early -- this inflates false positive rate)
- Account for novelty effects and day-of-week effects (run for at least 1-2 weeks)

**Step 5: Analyze results**
- Compute the test statistic and p-value
- Check for statistical significance AND practical significance
- A statistically significant 0.01% improvement may not be worth the engineering cost

### 9.3 Common Pitfalls

**Peeking problem**: Looking at results daily and stopping when significant inflates Type I error. Use sequential testing methods (e.g., always-valid p-values) if you need early stopping.

**Network effects**: In social platforms, user A's experience can affect user B's. Standard A/B tests assume independence.

**Simpson's paradox**: A trend that appears in subgroups can reverse when groups are combined. Always segment analysis by key dimensions.

### 9.4 A/B Testing for LLMs Specifically

**Challenges unique to LLMs**:
- **High variance in outputs**: The same query can get different quality answers. Need more samples.
- **Subjective quality**: "Better" is hard to define. Use human ratings or LLM-as-judge.
- **Interleaving**: Show both model outputs side-by-side and let users choose (like Chatbot Arena).
- **Long-term effects**: Users may adapt their prompting behavior over time.

**Metrics for LLM A/B tests**:
- Thumbs up/down rate
- Task completion rate
- Conversation length (shorter can be better if the answer is found faster)
- Retry rate (how often users rephrase and ask again)
- Latency (users abandon slow responses)

---

## 10. Interview Questions & Answers

### Q1: Why is cross-entropy the standard loss for language models? Derive it from MLE.

**Answer**: Language model training is maximum likelihood estimation. Given a training corpus, we want to find parameters $\theta$ that maximize the probability of the observed data:

$$\theta_{MLE} = \arg\max \prod_t P(x_t \mid x_{<t}; \theta)$$

Taking the log and converting to minimization:

$$\theta_{MLE} = \arg\min -\frac{1}{N} \sum_t \log P(x_t \mid x_{<t}; \theta)$$

This is exactly the cross-entropy between the true data distribution (which puts all mass on the actual next token) and the model's predicted distribution. So cross-entropy loss IS maximum likelihood -- they're the same objective expressed differently.

More formally, since the true distribution for each position is one-hot (the actual token has probability 1), the cross-entropy simplifies to:

$$H(P_{\text{true}}, P_{\text{model}}) = -\sum_v P_{\text{true}}(v) \log P_{\text{model}}(v) = -\log P_{\text{model}}(\text{correct\_token})$$

### Q2: What is KL divergence? Why is it asymmetric? When does this asymmetry matter?

**Answer**: KL divergence $D_{KL}(P \| Q)$ measures the extra bits needed to encode samples from $P$ using a code optimized for $Q$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

It's asymmetric because the weighting is different. $D_{KL}(P \| Q)$ weights by $P(x)$, while $D_{KL}(Q \| P)$ weights by $Q(x)$.

**When $P$ is a complex multimodal distribution and $Q$ is a simpler approximation**:

- **Forward KL** $D_{KL}(P \| Q)$: The model $Q$ is penalized for placing low probability where $P$ has high probability. $Q$ becomes **mean-seeking** -- it tries to cover all modes of $P$, which can lead to smearing probability mass over everything, including low-probability regions. This is what we minimize in standard ML training (cross-entropy = forward KL).

- **Reverse KL** $D_{KL}(Q \| P)$: The model $Q$ is penalized for placing high probability where $P$ has low probability. $Q$ becomes **mode-seeking** -- it locks onto the highest mode of $P$ and ignores the rest. This is used in variational inference (VAEs) and some RL formulations.

The asymmetry matters practically in RLHF: the KL penalty $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ uses the trained policy as $P$ and the reference as $Q$, meaning it penalizes the new policy for being too different from the reference in regions where the new policy places probability mass.

### Q3: Explain the relationship between entropy, cross-entropy, and KL divergence.

**Answer**: They form a clean hierarchy:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

$$\text{Cross-entropy} = \text{Entropy} + \text{KL Divergence}$$

- **Entropy $H(P)$**: The irreducible uncertainty in the data. The minimum average bits needed to encode samples from $P$ using the optimal code.
- **KL Divergence $D_{KL}(P \| Q)$**: The extra bits wasted because we're using $Q$'s code instead of $P$'s optimal code.
- **Cross-Entropy $H(P, Q)$**: The total average bits needed when encoding $P$ using $Q$'s code.

Since $D_{KL} \geq 0$, we always have $H(P, Q) \geq H(P)$. Cross-entropy equals entropy only when $Q = P$ (the model perfectly matches the data).

In language model training, $H(P)$ is the inherent entropy of language (approximately 1.0-1.5 bits per character for English). No model can beat this. The KL divergence $D_{KL}(P \| Q)$ represents how much our model still has to improve.

### Q4: How would you design an A/B test to compare two LLM variants?

**Answer**: I'd follow this framework:

**1. Define the question**: "Does LLM-v2 lead to higher user satisfaction than LLM-v1?"

**2. Choose metrics**:
- Primary: User satisfaction score (thumbs up rate or 5-point rating)
- Secondary: Task completion rate, average conversation length, retry rate
- Guardrails: Safety violation rate, latency (must not degrade)

**3. Calculate sample size**: Based on the baseline thumbs-up rate, the minimum detectable effect we care about (say, 2% absolute improvement), $\alpha = 0.05$, and power $= 0.80$. For typical LLM applications, this might require 5,000-50,000 conversations per variant due to high output variance.

**4. Randomize at the user level**: Hash user IDs to deterministically assign to control or treatment. This ensures the same user always sees the same variant, avoiding within-user contamination. Stratify by geography and user tenure.

**5. Run for 2+ weeks**: Accounts for day-of-week effects, novelty effects, and user adaptation to the new model's behavior.

**6. Analyze**: Compute p-value using a two-proportion z-test (or bootstrap for non-standard metrics). Check practical significance -- a statistically significant but tiny improvement may not justify deployment costs.

**7. LLM-specific considerations**: Also run LLM-as-judge evaluation on a sample of conversations. Check for regression on specific capability slices (e.g., coding, math, creative writing). Monitor hallucination rate using automated fact-checking on a subset.

### Q5: What is the difference between MLE and MAP? When does MAP reduce to MLE?

**Answer**: MLE maximizes the likelihood $P(\text{data} \mid \theta)$, finding parameters that make the data most probable. MAP maximizes the posterior $P(\theta \mid \text{data}) = P(\text{data} \mid \theta) \cdot P(\theta) / P(\text{data})$, which additionally incorporates a prior belief about the parameters.

The MAP objective is:

$$\theta_{MAP} = \arg\max \left[\log P(\text{data} \mid \theta) + \log P(\theta)\right]$$

The extra term $\log P(\theta)$ acts as a regularizer:
- Gaussian prior -> L2 regularization
- Laplace prior -> L1 regularization

MAP reduces to MLE when the prior is uniform (flat): $P(\theta) = \text{constant}$ for all $\theta$. In this case, $\log P(\theta)$ is a constant that doesn't affect the argmax, so the MAP solution equals the MLE solution. This makes intuitive sense: if you have no prior preference for any parameter values, you just go with whatever the data says.

### Q6: Explain the connection between L2 regularization and Bayesian priors.

**Answer**: L2 regularization adds a penalty $\lambda \|\theta\|^2$ to the loss function. From a Bayesian perspective, this is equivalent to placing a zero-mean Gaussian prior on the weights:

$$P(\theta) = \mathcal{N}(0, \sigma^2 I)$$

where $\lambda = \frac{1}{2\sigma^2}$.

The MAP objective becomes:

$$\arg\max \left[\log P(\text{data} \mid \theta) + \log P(\theta)\right] = \arg\max \left[\log P(\text{data} \mid \theta) - \frac{\|\theta\|^2}{2\sigma^2}\right] = \arg\min \left[-\log P(\text{data} \mid \theta) + \lambda \|\theta\|^2\right]$$

This is exactly the L2-regularized loss. A smaller $\sigma^2$ (tighter prior, stronger belief that weights should be near zero) corresponds to a larger $\lambda$ (stronger regularization).

This connection extends to modern deep learning: weight decay in AdamW is effectively imposing a Gaussian prior on the weights, biasing the model toward simpler solutions with smaller parameter magnitudes.

### Q7: You're told a model achieves perplexity of 15 on a held-out set. What does this mean? Is it good?

**Answer**: Perplexity of 15 means that on average, the model is as uncertain as if it were choosing uniformly among 15 tokens at each step. Mathematically:

$$\text{PPL} = \exp(\text{average cross-entropy}) = \exp\left(-\frac{1}{N} \sum \log P(\text{correct\_token})\right)$$

Whether 15 is good depends entirely on context:
- For a character-level model, PPL = 15 is excellent (out of ~100 characters)
- For a word-level model with 50K vocabulary, PPL = 15 is outstanding (GPT-2 achieved ~20-30 on common benchmarks)
- For a restricted domain with small vocabulary, PPL = 15 might be mediocre

Perplexity is useful for comparing models trained on the same data with the same tokenizer, but it has limitations: it doesn't measure factuality, coherence, helpfulness, or safety. A model could have low perplexity while being a fluent liar.

### Q8: What is mutual information? How is it used in representation learning?

**Answer**: Mutual information $I(X; Y)$ quantifies how much knowing one variable reduces uncertainty about another:

$$I(X; Y) = H(X) - H(X \mid Y) = D_{KL}(P(X,Y) \| P(X)P(Y))$$

It's zero when $X$ and $Y$ are independent, and equals $H(X)$ when $Y$ completely determines $X$.

In representation learning, we want to learn a representation $Z$ of input $X$ that captures the most useful information. The InfoMax principle says: maximize $I(X; Z)$.

This is the core idea behind contrastive learning methods like CLIP (used to align images and text):
- Positive pairs (matching image-text) should have high mutual information
- Negative pairs (mismatched image-text) should have low mutual information
- The InfoNCE loss is a lower bound on mutual information

In practice, mutual information is hard to compute exactly for high-dimensional data, so we optimize surrogate losses (like InfoNCE) that are tractable lower bounds.

### Q9: Why does softmax have a temperature parameter? What happens at extreme temperatures?

**Answer**: The temperature-scaled softmax is:

$$P(i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

Temperature controls the "sharpness" of the distribution:

- $T \to 0$: The distribution collapses to a one-hot vector on the argmax. This is greedy/deterministic. All probability mass goes to the highest logit.
- $T = 1$: Standard softmax. The distribution reflects the model's learned confidence.
- $T \to \infty$: The distribution approaches uniform. All tokens become equally likely, regardless of the logits.

Practical uses:
- **Decoding**: Lower temperature for factual tasks (deterministic), higher for creative tasks (diverse)
- **Knowledge distillation**: High temperature "softens" the teacher's outputs, revealing more information about the relative ranking of all classes, not just the top prediction. The student learns from these "dark knowledge" soft labels.
- **Contrastive learning**: Temperature scales the similarity scores, controlling how hard/soft the discrimination between positive and negative pairs is.

### Q10: What is the difference between the Bernoulli, Categorical, and Multinomial distributions? Where does each appear in NLP?

**Answer**:

- **Bernoulli**: A single trial with 2 outcomes (binary). $P(X = 1) = p$. Example: Binary sentiment classification (positive/negative). Each prediction is one Bernoulli trial.

- **Categorical**: A single trial with $K$ outcomes. $P(X = k) = p_k$, where probabilities sum to 1. Example: **Every single token prediction in a language model** is drawn from a categorical distribution over the vocabulary. The softmax output defines this categorical distribution.

- **Multinomial**: $N$ independent categorical trials, counting how many times each category occurs. Example: A bag-of-words representation counts how many times each word appears in a document -- this is a multinomial distribution.

The hierarchy: Bernoulli is Categorical with $K=2$. Categorical is Multinomial with $N=1$.

---

*Next: [Topic 2: Linear Algebra & Optimization](08_Linear_Algebra_Optimization.md)*
