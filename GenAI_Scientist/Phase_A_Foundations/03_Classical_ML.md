# Topic 3: Classical Machine Learning

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 3 of 28
> **Scope**: Linear/logistic regression, SVMs, trees, ensembles, clustering, dimensionality reduction, bias-variance, regularization, feature engineering, evaluation metrics
> **Why this matters**: These are the warm-up questions in every AI Scientist interview. Stumbling here ends the conversation before you ever reach transformers. Classical ML also builds the vocabulary (loss functions, regularization, overfitting) that carries into deep learning.
> **Previous**: [Topic 2: Linear Algebra & Optimization](02_Linear_Algebra_Optimization.md)
> **Next**: [Topic 4: Deep Learning Foundations](04_Deep_Learning_Foundations.md)

---

## Table of Contents

1. [Supervised Learning: Regression](#1-supervised-learning-regression)
2. [Supervised Learning: Classification](#2-supervised-learning-classification)
3. [Support Vector Machines](#3-support-vector-machines)
4. [Decision Trees](#4-decision-trees)
5. [Ensemble Methods](#5-ensemble-methods)
6. [Bias-Variance Tradeoff](#6-bias-variance-tradeoff)
7. [Regularization](#7-regularization)
8. [Unsupervised Learning: Clustering](#8-unsupervised-learning-clustering)
9. [Dimensionality Reduction](#9-dimensionality-reduction)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Overfitting: Detection & Prevention](#11-overfitting-detection--prevention)
12. [Interview Questions & Answers](#12-interview-questions--answers)

---

## 1. Supervised Learning: Regression

### 1.1 Linear Regression

**Goal**: Predict a continuous target $y$ from features $\mathbf{x}$.

**Model**:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

**Loss function** — Mean Squared Error (MSE):

$$L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

**Why MSE?** Under the assumption that errors are Gaussian ($y = \mathbf{w}^T \mathbf{x} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$), minimizing MSE is equivalent to maximizing the likelihood (MLE). This was derived in Topic 1.

**Closed-form solution** (Normal Equation):

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Where $\mathbf{X}$ is the $(n \times d)$ data matrix.

**When the closed form fails**: If $\mathbf{X}^T \mathbf{X}$ is singular (features are linearly dependent) or if $d$ is very large (inverting a $d \times d$ matrix is $O(d^3)$). In these cases, use gradient descent or regularized solutions (Ridge regression makes $\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}$ always invertible).

**Gradient descent solution**:

$$\frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$

$$\text{Update: } \mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}$$

**Assumptions of linear regression**:
1. Linearity: $y$ is a linear function of $\mathbf{x}$ (can be relaxed with feature engineering)
2. Independence: observations are independent
3. Homoscedasticity: constant error variance
4. Normality of errors: errors are Gaussian (for valid confidence intervals)

**Connection to deep learning**: A single linear layer `nn.Linear(d_in, d_out)` IS linear regression with $d_{out}$ outputs. The entire output head of a transformer (projecting from hidden dimension to vocabulary size) is a linear regression layer.

### 1.2 Polynomial Regression

Extend linear regression by adding polynomial features:

$$\mathbf{x} \rightarrow [x, x^2, x^3, \dots, x^p]$$

The model is still linear in the parameters ($\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$) even though it's nonlinear in $\mathbf{x}$. This is the simplest example of feature engineering expanding model capacity.

**Danger**: High-degree polynomials overfit dramatically. A degree-$n$ polynomial can perfectly fit $n+1$ points, regardless of the underlying pattern. This motivates regularization.

---

## 2. Supervised Learning: Classification

### 2.1 Logistic Regression

Despite its name, logistic regression is a **classification** algorithm.

**Model**: Apply the sigmoid function to a linear model to get probabilities:

$$P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

**The sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Range: $(0, 1)$ -- valid probability
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

**Loss function** — Binary Cross-Entropy (BCE):

$$L(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

where $p_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$

**Why this loss?** It's the negative log-likelihood of the Bernoulli distribution. Minimizing BCE = MLE for logistic regression. This is the same cross-entropy loss used in neural network classification and (extended to multi-class) in language models.

### 2.2 Deriving the Gradient for Logistic Regression

This derivation is frequently asked in interviews. Let's do it step by step.

For a single sample, the loss is:

$$l = -\left[ y \log(\sigma(z)) + (1-y) \log(1 - \sigma(z)) \right]$$

where $z = \mathbf{w}^T \mathbf{x} + b$

We need $\frac{\partial l}{\partial \mathbf{w}}$. Using the chain rule:

$$\frac{\partial l}{\partial \mathbf{w}} = \frac{\partial l}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$$

**Step 1**: Compute $\frac{\partial l}{\partial z}$.

$$\frac{\partial l}{\partial z} = -\left[ y \cdot \frac{\sigma'(z)}{\sigma(z)} + (1-y) \cdot \frac{-\sigma'(z)}{1-\sigma(z)} \right]$$

Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\frac{\partial l}{\partial z} = -[y(1 - \sigma(z)) + (1-y)(-\sigma(z))]$$
$$= -[y - y\sigma(z) - \sigma(z) + y\sigma(z)]$$
$$= -[y - \sigma(z)]$$
$$= \sigma(z) - y = p - y$$

This is remarkably clean: the gradient with respect to $z$ is simply (prediction - true label).

**Step 2**: Compute $\frac{\partial z}{\partial \mathbf{w}}$.

$$\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x} \quad \text{(since } z = \mathbf{w}^T \mathbf{x} + b\text{)}$$

**Step 3**: Combine.

$$\frac{\partial l}{\partial \mathbf{w}} = (p - y) \cdot \mathbf{x}$$

**The gradient update for a batch**:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) \mathbf{x}_i$$

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}$$

**Why this matters beyond logistic regression**: The gradient $(prediction - target) \cdot input$ pattern appears throughout deep learning. The output gradient of a softmax + cross-entropy layer in transformers has exactly this form.

### 2.3 Multi-Class Classification (Softmax Regression)

Extend to $K$ classes using softmax:

$$P(y=k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

**Loss** — Categorical Cross-Entropy:

$$L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik})$$

For one-hot targets, this simplifies to:

$$L = -\frac{1}{n} \sum_{i=1}^{n} \log(p_{i, \text{true\_class}})$$

This is identical to the language model training loss: the negative log-probability of the correct token.

### 2.4 Naive Bayes

**Approach**: Apply Bayes' theorem with the "naive" assumption of conditional independence of features given the class.

$$P(y | x_1, \dots, x_d) \propto P(y) \prod_{j=1}^{d} P(x_j | y)$$

**Types**:
- **Gaussian NB**: $P(x_j | y)$ is Gaussian (continuous features)
- **Multinomial NB**: $P(x_j | y)$ is Multinomial (word counts, TF-IDF)
- **Bernoulli NB**: $P(x_j | y)$ is Bernoulli (binary features)

**Strengths**: Fast, works well with high-dimensional sparse data (text classification), good baseline.

**Weakness**: The independence assumption is almost always violated. "New York" -- "New" and "York" are highly dependent given a location classification task.

**Connection to NLP**: Naive Bayes was the standard text classifier before deep learning. The independence assumption is what makes it "naive" -- and it's exactly the kind of assumption that contextual models (BERT, GPT) overcome by modeling token interactions.

### 2.5 K-Nearest Neighbors (KNN)

**Algorithm**: To classify a new point, find its $K$ nearest neighbors in the training set and take a majority vote.

**Hyperparameters**:
- **K**: Number of neighbors. Small $K$ = complex boundary (overfitting), large $K$ = smooth boundary (underfitting).
- **Distance metric**: Euclidean, Manhattan, cosine, etc.

**Properties**:
- Non-parametric: no training phase, all computation at inference time
- Suffers from curse of dimensionality: in high dimensions, all points become equidistant
- Memory-intensive: must store entire training set

**Connection to GenAI**: KNN over embeddings is essentially what retrieval in RAG does. Finding the $K$ most similar documents to a query embedding is KNN search. Vector databases (FAISS, Pinecone) are optimized KNN engines using approximate nearest neighbor algorithms.

---

## 3. Support Vector Machines

### 3.1 Linear SVM

**Goal**: Find the hyperplane that maximizes the margin between classes.

**Decision boundary**: $\mathbf{w}^T \mathbf{x} + b = 0$

**Margin**: The distance between the decision boundary and the nearest points from each class. Margin $= \frac{2}{\|\mathbf{w}\|}$.

**Optimization problem** (hard margin):

$$\min \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \; \forall \, i$$

**Intuition**: We want the largest possible margin (gap) between classes. The constraint ensures all points are correctly classified and at least margin-distance away from the boundary. Support vectors are the points that lie exactly on the margin boundary.

### 3.2 Soft Margin SVM

Real data is rarely linearly separable. Soft margin allows some misclassification:

$$\min \frac{1}{2} \|\mathbf{w}\|^2 + C \sum \xi_i \quad \text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$

- $\xi_i$ (slack variables): How much each point violates the margin
- $C$: Trade-off between large margin (small $\|\mathbf{w}\|$) and few violations (small $\xi$)
  - Large $C$: Narrow margin, fewer misclassifications (more complex, can overfit)
  - Small $C$: Wide margin, more misclassifications allowed (simpler, can underfit)

### 3.3 The Kernel Trick

**Problem**: Data that's not linearly separable in the original space might be separable in a higher-dimensional space.

**Naive approach**: Map $\mathbf{x} \rightarrow \boldsymbol{\phi}(\mathbf{x})$ in high dimensions, then find a linear separator. But computing $\boldsymbol{\phi}(\mathbf{x})$ explicitly is expensive.

**The kernel trick**: The SVM optimization and prediction only depend on dot products between data points. We can replace the dot product with a kernel function:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \boldsymbol{\phi}(\mathbf{x}_i)^T \boldsymbol{\phi}(\mathbf{x}_j)$$

We never need to compute $\boldsymbol{\phi}$ explicitly -- only the kernel function.

**Common kernels**:
- **Linear**: $K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{y}$
- **Polynomial**: $K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T \mathbf{y} + c)^d$
- **RBF (Gaussian)**: $K(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x}-\mathbf{y}\|^2}{2\sigma^2}\right)$
  - Implicitly maps to infinite-dimensional space
  - Most commonly used non-linear kernel

### 3.4 SVMs vs Neural Networks

| Aspect | SVMs | Neural Networks |
|--------|------|-----------------|
| Features | Need manual feature engineering | Learn features automatically |
| Data size | Good with small data | Need large data |
| Scalability | $O(n^2)$ to $O(n^3)$ training | Scales with SGD |
| Non-linearity | Via kernels | Via layers + activations |
| Interpretability | Support vectors are meaningful | Black box |

SVMs dominated NLP (text classification) until ~2015, when deep learning took over. For small datasets with good features, SVMs remain competitive.

---

## 4. Decision Trees

### 4.1 How Trees Work

A decision tree recursively splits the data on features to create regions that are as pure (homogeneous) as possible.

```
         [Is age > 30?]
        /              \
      Yes               No
       |                 |
  [Income > 50K?]   [Student?]
   /         \        /      \
  Buy     Don't    Buy    Don't
```

At each node, the algorithm selects the feature and threshold that best separates the classes.

### 4.2 Splitting Criteria

**For classification**:

**Gini Impurity**:

$$\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the fraction of class $k$ in set $S$

- Gini = 0: perfectly pure (all one class)
- Gini = 0.5: maximum impurity for binary classification (50/50 split)

**Information Gain** (based on entropy):

$$\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

$$\text{Information Gain} = \text{Entropy}(\text{parent}) - \sum \frac{|S_{\text{child}}|}{|S_{\text{parent}}|} \cdot \text{Entropy}(S_{\text{child}})$$

Choose the split that maximizes information gain (equivalently, maximizes the reduction in entropy).

**For regression**:

**Variance Reduction**: Split to minimize the variance of the target within each resulting node. Typically uses MSE.

### 4.3 Gini vs Entropy

In practice, they produce very similar trees. Gini is slightly faster to compute (no logarithm). Entropy can be slightly more balanced when classes are imbalanced. Most implementations default to Gini.

### 4.4 Tree Properties

**Strengths**:
- Interpretable (you can read the rules)
- Handles mixed feature types (categorical + numerical)
- No feature scaling needed
- Handles missing values naturally
- Captures non-linear relationships and interactions

**Weaknesses**:
- High variance: small data changes can produce very different trees
- Tend to overfit (keep splitting until pure leaves)
- Biased toward features with many categories
- Can't extrapolate beyond the training data range

**Controlling overfitting**:
- Max depth limit
- Minimum samples per leaf
- Minimum samples to split
- Maximum number of leaf nodes
- Pruning: grow full tree, then remove branches that don't improve validation performance

### 4.5 Decision Tree Split — Worked Example

10 patients, predicting disease (Y=1) or no disease (Y=0).

Feature: "Age > 40?" splits data:
  Age ≤ 40:  3 patients → [Y=0: 3, Y=1: 0]   pure leaf, no disease
  Age > 40:  7 patients → [Y=0: 1, Y=1: 6]   mostly disease

**Entropy before split** (5Y=0, 5Y=1 in parent):
  H(parent) = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1.0 bit (maximum uncertainty)

**Entropy after split**:
  H(Age≤40) = -1×log₂(1) = 0 bits (pure)
  H(Age>40) = -(1/7)×log₂(1/7) - (6/7)×log₂(6/7) ≈ 0.592 bits

**Weighted entropy** = (3/10)×0 + (7/10)×0.592 = 0.414 bits

**Information Gain** = H(parent) - H(after) = 1.0 - 0.414 = **0.586 bits**

A higher IG means this split is more informative. The tree chooses the feature with highest IG at each node.

Gini impurity is an alternative to entropy: G = 1 - Σpᵢ²
For the parent (0.5/0.5 split): G = 1 - (0.5² + 0.5²) = 0.5 (max impurity)
For Age≤40 leaf: G = 1 - (1² + 0²) = 0 (pure)

---

## 5. Ensemble Methods

The weaknesses of individual trees (high variance, overfitting) are overcome by combining many trees. Ensembles are among the most practically important ML methods.

### 5.1 Bagging (Bootstrap Aggregating)

**Core idea**: Train many models on different random subsets of the data, then average their predictions.

**Algorithm**:
1. Create $B$ bootstrap samples (sample $n$ points with replacement from the training data)
2. Train a separate model on each bootstrap sample
3. Predict by averaging (regression) or majority voting (classification)

**Why it works**: Each bootstrap model has high variance but low bias. Averaging reduces variance while preserving the low bias. Mathematically: $\text{Var}(\text{average of } B \text{ independent models}) = \text{Var}(\text{single model}) / B$.

### 5.2 Random Forests

Random Forest = Bagging + random feature subsets.

**Algorithm**:
1. Create $B$ bootstrap samples
2. For each tree, at each split: consider only a random subset of features (typically $\sqrt{d}$ for classification, $d/3$ for regression)
3. Predict by averaging/voting

**Why the feature randomness helps**: Without it, all trees would split on the same strong features and be highly correlated. Averaging correlated models doesn't reduce variance much. By forcing each tree to use different features, the trees become more diverse (decorrelated), and averaging is more effective.

**Hyperparameters**:
- **n_estimators** ($B$): Number of trees. More is generally better (diminishing returns). 100-500 typical.
- **max_features**: Number of features per split. $\sqrt{d}$ is a good default.
- **max_depth**: Tree depth. Unlimited is often fine (bagging handles overfitting).

**Feature importance**: Measure how much each feature reduces impurity across all trees. Random Forests give this for free.

### 5.3 Gradient Boosting

**Core idea**: Train trees sequentially, where each new tree corrects the errors of the ensemble so far.

**Algorithm**:
1. Start with a simple prediction (e.g., the mean)
2. Compute the residuals (errors) of the current ensemble
3. Train a new (small) tree to predict these residuals
4. Add the new tree's predictions (scaled by learning rate) to the ensemble
5. Repeat

**Formal view**: Gradient boosting performs gradient descent in function space. At each step, the new tree approximates the negative gradient of the loss function with respect to the current predictions.

```
F_0(x) = initial prediction (e.g., mean of y)
For m = 1, ..., M:
    r_i = -dL/dF(x_i)  evaluated at F = F_{m-1}     (pseudo-residuals)
    Train tree h_m to predict r_i
    F_m(x) = F_{m-1}(x) + eta * h_m(x)               (eta = learning rate)
```

For MSE loss, the pseudo-residuals are literally the residuals: $r_i = y_i - F_{m-1}(x_i)$.
For other losses (log-loss for classification), the pseudo-residuals are the gradient of that loss.

### 5.4 XGBoost (eXtreme Gradient Boosting)

XGBoost is the most widely used gradient boosting implementation. Key innovations:

**Regularized objective**:

$$L = \sum l(y_i, F(x_i)) + \sum \Omega(h_m)$$

$$\Omega(h) = \gamma T + \frac{1}{2} \lambda \|\mathbf{w}\|^2$$

Where $T$ is the number of leaves and $\mathbf{w}$ is the leaf weights. This penalizes complex trees.

**Second-order approximation**: Uses both gradient AND Hessian (second derivative) to find optimal splits, leading to faster convergence.

**Efficient implementation**:
- Histogram-based splitting (bin continuous features for faster split finding)
- Parallel tree construction
- Cache-aware access patterns
- Out-of-core computation for datasets that don't fit in memory

**Handling missing values**: XGBoost learns which direction to send missing values at each split.

### 5.5 LightGBM and CatBoost

**LightGBM** (Microsoft):
- Leaf-wise growth (vs level-wise): Grows the leaf with highest loss reduction, leading to deeper, more asymmetric trees
- GOSS (Gradient-based One-Side Sampling): Focus on data points with large gradients
- Faster than XGBoost on large datasets

**CatBoost** (Yandex):
- Native handling of categorical features (target encoding with ordered boosting to prevent leakage)
- Ordered boosting to reduce overfitting
- Good out-of-the-box performance

### 5.6 Bagging vs Boosting

```
                    Bagging (Random Forest)    Boosting (XGBoost)
Combine how?        Parallel, average          Sequential, additive
Reduces what?       Variance                   Bias (and variance)
Base learners       Deep, complex trees        Shallow, weak trees
Overfitting risk    Low                        Higher (needs tuning)
Training speed      Parallelizable             Sequential (slower)
Typical winner      Noisy data, outliers       Clean data, complex patterns
```

### 5.7 When to Use Tree Ensembles vs Neural Networks

**Tree ensembles (XGBoost/LightGBM) win when**:
- Tabular data with heterogeneous features
- Small to medium datasets (< 100K samples)
- Need fast training and interpretation
- Features have meaningful interactions that trees capture naturally

**Neural networks win when**:
- Unstructured data (text, images, audio)
- Very large datasets
- Transfer learning is applicable
- Sequential/spatial structure in the data

**Recent research**: Even for tabular data, well-tuned deep networks can match tree ensembles, but trees still win on ease of tuning. For most Kaggle-style tabular competitions, XGBoost/LightGBM remain dominant.

### 5.8 Gradient Boosting — Step-by-Step

Dataset: predict house price from size.
Initial prediction: mean(Y) = $200K for all houses.

**Iteration 1**:
  Residuals (actual - predicted):
    House A: $300K - $200K = +$100K  (underpredicted)
    House B: $150K - $200K = -$50K   (overpredicted)
    House C: $250K - $200K = +$50K   (underpredicted)
  
  Fit a weak learner (depth-1 tree) to the RESIDUALS:
    If size > 2000sqft: predict +$75K
    If size ≤ 2000sqft: predict -$25K
  
  Update predictions (learning rate η=0.1):
    House A (>2000sqft): $200K + 0.1×$75K = $207.5K
    House B (≤2000sqft): $200K + 0.1×(-$25K) = $197.5K
    House C (>2000sqft): $200K + 0.1×$75K = $207.5K

**Key insight**: Each tree corrects the errors of all previous trees. With 100 iterations, the model accumulates 100 small corrections. The small learning rate (0.1) prevents overfitting to any single tree's noise.

**XGBoost/LightGBM add**: Second-order gradients (Hessian) for smarter leaf value computation, regularization terms, and tree structure optimization.

---

## 6. Bias-Variance Tradeoff

This is the single most asked classical ML concept in interviews.

### 6.1 Decomposition

For any model, the expected prediction error can be decomposed as:

$$E\left[(y - \hat{f}(\mathbf{x}))^2\right] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

Where:
- $\text{Bias} = E[\hat{f}(\mathbf{x})] - f_{\text{true}}(\mathbf{x})$ (systematic error)
- $\text{Variance} = E\left[(\hat{f}(\mathbf{x}) - E[\hat{f}(\mathbf{x})])^2\right]$ (sensitivity to training data)
- $\text{Noise} = \text{Var}(\epsilon)$ (inherent randomness in data)

### 6.2 Intuition

**Bias**: How far the average prediction is from the truth. High bias means the model is too simple to capture the pattern. It systematically misses the target.

**Variance**: How much predictions change if you retrain with different data. High variance means the model is too sensitive to the training set. It fits noise.

**Noise**: Irreducible error from randomness in the data. No model can beat this.

### 6.3 The Tradeoff

Simple model (e.g., linear regression): **HIGH BIAS, LOW VARIANCE**
- Underfits: misses the true pattern
- Consistent: similar predictions regardless of training data

Complex model (e.g., deep tree, high-degree polynomial): **LOW BIAS, HIGH VARIANCE**
- Can fit any pattern (including noise)
- Unstable: different training sets -> very different predictions

**Optimal model complexity** minimizes the sum $\text{Bias}^2 + \text{Variance}$.

```
                Total Error
                    |
                    |   ________
     High Bias      |  /        \     High Variance
     (underfit)     | /    *     \    (overfit)
                    |/            \
                    +------+-------\-----> Model Complexity
                           |
                    Optimal point
```

### 6.4 Concrete Example

Suppose the true relationship is $y = \sin(x) + \text{noise}$.

**Linear model (high bias, low variance)**: Fits a straight line. Always misses the curvature (bias). But the line is stable -- different training samples give similar lines (low variance).

**Degree-20 polynomial (low bias, high variance)**: Can perfectly capture the sine curve. But with limited data, it also fits the noise, producing wild oscillations between data points. Different training samples produce very different curves (high variance).

**Degree-3 polynomial (balanced)**: Captures the curvature reasonably well without fitting noise. Sweet spot.

### 6.5 How Different Techniques Affect Bias and Variance

| Technique | Effect on Bias | Effect on Variance |
|-----------|---------------|-------------------|
| More training data | No change | Decreases |
| More features | Decreases | Increases |
| Regularization (L1/L2) | Increases slightly | Decreases |
| Bagging / Random Forest | No change | Decreases |
| Boosting | Decreases | Can increase |
| Dropout | Increases slightly | Decreases |
| Deeper network | Decreases | Increases |
| Early stopping | Increases slightly | Decreases |
| Data augmentation | May decrease | Decreases |

### 6.6 Bias-Variance in Deep Learning

Classical bias-variance theory suggests very complex models should overfit. But modern deep networks (with billions of parameters, far more than training points) generalize well. This is the "double descent" phenomenon:

```
Error
  |
  |  \         /\
  |   \       /  \_______________
  |    \     /
  |     \   /
  |      \_/
  +------+------+------+------> Number of Parameters
     Under-   Inter-   Over-
     param.   polation  param.
              threshold
```

Beyond the interpolation threshold (where the model can perfectly fit training data), increasing parameters FURTHER can actually improve generalization. This is why LLMs with 70B+ parameters trained on finite data don't catastrophically overfit -- they're in the "over-parameterized" regime where implicit regularization from SGD, architecture, and training procedures keeps generalization good.

### 6.7 Bias-Variance Visual

```
          High Bias        Low Bias
          (Underfitting)   (Good fit)

High      ┌──────────────┬──────────────┐
Variance  │  WORST:      │  OVERFIT:    │
          │  Simple model│  Wiggly model│
          │  wrong +     │  memorizes   │
          │  inconsistent│  training    │
          ├──────────────┼──────────────┤
Low       │  UNDERFIT:   │  BEST:       │
Variance  │  Simple model│  Complex     │
          │  consistently│  model, good │
          │  wrong       │  generalize  │
          └──────────────┴──────────────┘
```

Practical examples:
  High Bias:    Linear regression for non-linear data
  High Variance: Depth-10 decision tree on 100 samples (memorizes training)
  Good balance: Random forest, gradient boosting, regularized deep networks

The bias-variance tradeoff:
  Total Error = Bias² + Variance + Irreducible Noise
  
  Increasing model complexity:
    Bias ↓ (fits training data better)
    Variance ↑ (more sensitive to training data specifics)
  
  Sweet spot: where Bias² + Variance is minimized on validation data

---

## 7. Regularization

### 7.1 L2 Regularization (Ridge)

Add a penalty on the squared magnitude of weights:

$$L_{\text{ridge}} = L_{\text{data}} + \lambda \|\mathbf{w}\|_2^2 = L_{\text{data}} + \lambda \sum w_j^2$$

**Effect**: Shrinks all weights toward zero, but rarely makes them exactly zero. Smooth, proportional shrinkage.

**Bayesian interpretation**: Gaussian prior on weights (Topic 1, Section 5.2). A strong prior (large $\lambda$) pulls weights closer to zero.

**Geometric view**: The constraint region is a circle (sphere in higher dimensions). The optimal point is where the loss contours touch the circle -- since the circle is smooth, the touching point rarely lies exactly on an axis.

### 7.2 L1 Regularization (Lasso)

Add a penalty on the absolute magnitude of weights:

$$L_{\text{lasso}} = L_{\text{data}} + \lambda \|\mathbf{w}\|_1 = L_{\text{data}} + \lambda \sum |w_j|$$

**Effect**: Drives some weights to exactly zero, producing sparse models. Feature selection built-in.

**Bayesian interpretation**: Laplace prior on weights, which has a sharp peak at zero.

**Geometric view** — why L1 produces sparsity:

The constraint region is a diamond (cross-polytope in higher dimensions). The diamond has corners that lie on the axes. When the loss contours touch the diamond, they are most likely to touch at a corner (where some coordinates are exactly zero) rather than on a flat edge. This is because the corners are the "pointiest" parts of the diamond.

```
    L2 (circle)              L1 (diamond)

      ___                        *
    /     \                     /|\
   |       |     loss          / | \    loss
   |       |  contours  ->  *--+--*  contours
   |       |                 \ | /    touch at
    \_____/                   \|/     a corner
                               *      (sparse!)

    Touch on smooth             Touch at corner
    surface (dense)             (w_j = 0 for some j)
```

### 7.3 Elastic Net

Combine L1 and L2:

$$L = L_{\text{data}} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

Benefits of both: sparsity from L1 + stability from L2. Useful when features are correlated -- L1 alone arbitrarily picks one of correlated features, while Elastic Net keeps groups together.

### 7.4 Regularization in Deep Learning

- **Weight decay** (L2): Standard in AdamW, see Topic 2
- **Dropout**: Randomly zero neurons during training (see Topic 4)
- **Early stopping**: Stop training when validation loss starts increasing
- **Data augmentation**: Artificially increase training data diversity
- **Batch/Layer normalization**: Stabilizes training, has implicit regularization effect
- **Label smoothing**: Replace hard labels (0/1) with soft labels (0.1/0.9). Prevents the model from becoming over-confident. Used in transformer training.

---

## 8. Unsupervised Learning: Clustering

### 8.1 K-Means

**What it does**: Partitions data into $K$ clusters by iteratively assigning each point to its nearest centroid and recomputing centroids until convergence.

**Algorithm**: (1) Initialize $K$ centers (randomly or K-Means++). (2) Assign each point to nearest center. (3) Move each center to the mean of its cluster. (4) Repeat until stable.

**When to use**: Known number of clusters, roughly spherical clusters, fast approximate grouping.

**Key limitations**: $K$ must be chosen upfront; assumes equal-size spherical clusters; sensitive to outliers; converges to local optima (run multiple times).

**K-Means++ initialization**: Choose initial centers far apart — dramatically improves results over random init.

**Choosing K**: Elbow method (plot within-cluster variance vs K, pick the "elbow"), silhouette score.

### 8.2 DBSCAN (Density-Based Spatial Clustering)

**Core idea**: Clusters are dense regions separated by sparse regions.

**Parameters**:
- $\epsilon$: Maximum distance between two points to be considered neighbors
- **min_points**: Minimum number of points to form a dense region

**Point types**:
- **Core point**: Has $\geq$ min_points neighbors within $\epsilon$
- **Border point**: Within $\epsilon$ of a core point but not itself a core point
- **Noise**: Neither core nor border (outlier)

**Strengths over K-Means**:
- Discovers clusters of arbitrary shape
- Doesn't require specifying $K$
- Identifies outliers/noise naturally
- Robust to outliers

**Weaknesses**: Struggles with varying density, sensitive to $\epsilon$ choice, $O(n^2)$ without spatial indexing.

### 8.3 Hierarchical Clustering

Build a tree (dendrogram) of clusters by iteratively merging (agglomerative) or splitting (divisive).

**Agglomerative** (bottom-up):
1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains
4. Cut the dendrogram at the desired level to get $K$ clusters

**Linkage criteria** (how to measure distance between clusters):
- **Single**: min distance between any two points in different clusters (finds elongated clusters)
- **Complete**: max distance (finds compact clusters)
- **Average**: average distance
- **Ward's**: minimizes increase in total within-cluster variance (often best for balanced clusters)

---

## 9. Dimensionality Reduction

### 9.1 PCA (Covered in Topic 2)

Linear dimensionality reduction that preserves maximum variance. Projects onto top eigenvectors of covariance matrix. Fast, well-understood, but can only capture linear structure.

### 9.2 t-SNE (t-distributed Stochastic Neighbor Embedding)

**Goal**: Visualize high-dimensional data in 2D/3D while preserving local structure.

**How it works**:
1. In high-dimensional space, compute pairwise similarities using Gaussian distribution:

   $p_{ij}$ = similarity of points $i$ and $j$ (Gaussian kernel)

2. In low-dimensional space (2D), compute similarities using Student's t-distribution:

   $q_{ij}$ = similarity in the embedding space (t-distribution, heavier tails)

3. Minimize KL divergence between $P$ and $Q$:

$$L = \text{KL}(P \| Q) = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Why t-distribution in low dimensions?** The "crowding problem": when mapping from high to low dimensions, moderate distances in high-D all get crushed into a small range in low-D. The heavier tails of the t-distribution allow moderately distant points to be mapped further apart, preserving cluster separation.

**Properties**:
- Excellent for visualization
- Preserves local structure (nearby points stay nearby)
- Does NOT preserve global structure (distances between clusters are meaningless)
- Non-parametric: can't project new points without re-running
- Slow: $O(n^2)$ or $O(n \log n)$ with Barnes-Hut approximation

**Perplexity**: Key hyperparameter, roughly controls how many neighbors to consider. Typical values: 5-50.

### 9.3 UMAP (Uniform Manifold Approximation and Projection)

**Improvement over t-SNE**:
- Faster (handles millions of points)
- Better preserves global structure
- Can project new data points
- Has theoretical foundations in topological data analysis

**In practice**: UMAP has largely replaced t-SNE for embedding visualization. When you see plots of BERT embeddings or LLM latent spaces, they're usually UMAP.

### 9.4 When to Use Which

| Method | Speed | Global Structure | New Points | Use For |
|--------|-------|-----------------|------------|---------|
| PCA | Fast | Preserves | Yes | Preprocessing, denoising, initial reduction |
| t-SNE | Slow | No | No | 2D/3D visualization (< 50K points) |
| UMAP | Fast | Partially | Yes | Visualization, preprocessing (any scale) |

---

## 10. Evaluation Metrics

### 10.1 Classification Metrics

**Confusion Matrix**:
```
                    Predicted
                  Pos    Neg
Actual  Pos      TP      FN
        Neg      FP      TN
```

**Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$. Misleading with imbalanced classes (99% accuracy by predicting majority class).

**Precision**: $\frac{TP}{TP + FP}$. Of all predicted positives, how many are truly positive? High precision = few false alarms.

**Recall (Sensitivity)**: $\frac{TP}{TP + FN}$. Of all actual positives, how many did we catch? High recall = few missed positives.

**F1 Score**: Harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Balances precision and recall. Harmonic mean (not arithmetic) because it penalizes extreme imbalances -- a model with precision=1.0 and recall=0.01 gets $F_1 = 0.02$, not 0.505.

**When to prioritize which**:
- **High precision**: Spam filtering (don't delete real emails), content moderation (don't censor legitimate speech)
- **High recall**: Medical diagnosis (don't miss cancer), fraud detection (catch all fraud), safety filtering for LLMs (catch all harmful outputs)

### 10.2 Precision-Recall Trade-off

Most classifiers output a probability. The classification threshold determines the trade-off:
- **Low threshold** (e.g., 0.1): Predict positive more often -> high recall, low precision
- **High threshold** (e.g., 0.9): Predict positive rarely -> low recall, high precision

**PR Curve**: Plot precision vs recall at various thresholds.

**Average Precision (AP)**: Area under the PR curve. Summary metric for ranking quality.

### 10.3 ROC-AUC

**ROC Curve**: Plot True Positive Rate (recall) vs False Positive Rate ($\frac{FP}{FP + TN}$) at various thresholds.

**AUC**: Area under the ROC curve. Range $[0, 1]$.
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier (diagonal line)
- AUC < 0.5: Worse than random (flip predictions)

**Interpretation**: AUC is the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

**ROC-AUC vs PR-AUC**: With imbalanced data, ROC-AUC can be misleadingly optimistic (the large number of true negatives inflates TNR). PR-AUC is more informative for imbalanced problems because it focuses on the positive class.

### 10.4 Regression Metrics

**MSE** (Mean Squared Error): $\frac{1}{n} \sum (y - \hat{y})^2$. Penalizes large errors heavily (quadratic).

**RMSE** (Root MSE): $\sqrt{\text{MSE}}$. Same units as $y$, more interpretable.

**MAE** (Mean Absolute Error): $\frac{1}{n} \sum |y - \hat{y}|$. Robust to outliers (linear penalty).

**R-squared**: $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{total}}} = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$. Fraction of variance explained. $R^2 = 1$ is perfect, $R^2 = 0$ is as bad as predicting the mean.

### 10.5 Cross-Validation

Instead of a single train/test split, evaluate on multiple splits to get a more reliable estimate.

**K-Fold Cross-Validation**:
1. Split data into $K$ folds (typically $K = 5$ or $10$)
2. For each fold: train on $K-1$ folds, evaluate on the held-out fold
3. Average the $K$ evaluation scores

**Stratified K-Fold**: Ensures each fold has roughly the same class distribution. Essential for imbalanced data.

**Leave-One-Out (LOO)**: $K = n$. Train on $n-1$ points, test on 1. Unbiased but high variance and expensive.

**Why cross-validation matters**: A single train/test split can be lucky or unlucky. Cross-validation gives a more robust performance estimate and its variance tells you how stable the model is.

---

## 11. Overfitting: Detection & Prevention

### 11.1 What Is Overfitting?

The model learns the training data too well, including its noise and idiosyncrasies, and fails to generalize to new data.

**Signature**: Training loss continues decreasing, but validation loss starts increasing.

```
Loss
  |  \
  |   \    ____________  Validation loss (increases after a point)
  |    \  /
  |     \/
  |      \
  |       \____________  Training loss (keeps decreasing)
  +------+------+------> Epochs
         |
    Start of overfitting
```

### 11.2 Causes

1. **Model too complex** for the amount of data (too many parameters, too deep)
2. **Not enough training data** for the model's capacity
3. **Noisy data** that the model memorizes
4. **Training too long** without early stopping
5. **No regularization** applied

### 11.3 Prevention Techniques

**More data**: The most reliable cure. If you double your data, you can usually use a more complex model without overfitting. Data augmentation is a cheaper alternative.

**Regularization**: L1, L2, Elastic Net (covered in Section 7).

**Dropout**: Randomly zero out neurons during training. Forces the network to not rely on any single feature. Acts as approximate ensemble (see Topic 4).

**Early stopping**: Monitor validation loss and stop training when it hasn't improved for a patience period. Save the best checkpoint.

**Cross-validation**: Don't overfit to a single train/test split.

**Simplify the model**: Fewer parameters, shallower architecture, fewer features.

**Ensemble methods**: Bagging reduces variance (Random Forests). Even a simple average of multiple models generalizes better than any individual.

**Label smoothing**: Prevents the model from assigning 100% probability to the correct class, reducing overconfidence.

---

## 12. Interview Questions & Answers

### Q1: Derive the gradient update for logistic regression.

**Answer**: For logistic regression, the model predicts $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$ where $\sigma$ is the sigmoid function.

The binary cross-entropy loss for one sample is:

$$l = -\left[ y \log(\sigma(z)) + (1-y) \log(1 - \sigma(z)) \right] \quad \text{where } z = \mathbf{w}^T \mathbf{x}$$

Using the chain rule and the fact that $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial l}{\partial z} = \sigma(z) - y = p - y$$

$$\frac{\partial l}{\partial \mathbf{w}} = \frac{\partial l}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}} = (p - y) \cdot \mathbf{x}$$

For a batch of $n$ samples:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) \mathbf{x}_i$$

Update rule:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{1}{n} \sum (p_i - y_i) \mathbf{x}_i$$

The beauty of this result is that the gradient has the same form as linear regression: (prediction - target) * input. This carries through to neural networks: the gradient of softmax + cross-entropy is (softmax_output - one_hot_target), which is exactly what backpropagation computes at the output layer of every language model.

### Q2: Explain the bias-variance tradeoff with a concrete example. How does model complexity relate?

**Answer**: Consider predicting house prices from square footage. The true relationship is slightly curved (diminishing returns as houses get very large).

**Linear regression (simple model)**: Always fits a straight line. Every time we retrain on different data, we get a similar straight line (low variance). But the line systematically misses the curvature (high bias). Underfits.

**Degree-15 polynomial (complex model)**: Can capture any curve. But with 50 data points, it fits the noise -- a different 50 points gives a wildly different curve (high variance). The average prediction might be close to the truth (low bias), but any individual prediction can be way off. Overfits.

**Degree-3 polynomial (balanced)**: Captures the gentle curvature without fitting noise. Moderate bias, moderate variance. Best generalization.

The tradeoff: as model complexity increases, bias decreases (the model can represent more complex patterns) but variance increases (the model becomes more sensitive to training data specifics). Total error $= \text{bias}^2 + \text{variance}$, and the optimal complexity minimizes this sum.

The connection to model complexity is direct: more parameters = lower bias, higher variance. Regularization shifts this balance by constraining effective complexity. The notable exception is modern deep learning in the over-parameterized regime (double descent), where extremely high capacity combined with SGD's implicit regularization can achieve both low bias and low variance.

### Q3: Why does L1 produce sparse weights while L2 doesn't?

**Answer**: The geometric explanation is most intuitive.

Regularization constrains the weights to a region: $\|\mathbf{w}\|_1 \leq t$ for L1 (a diamond shape) or $\|\mathbf{w}\|_2 \leq t$ for L2 (a sphere shape).

The optimal weights are where the loss function's contour first touches the constraint region.

The L2 constraint is a smooth sphere with no corners. The loss contour can touch it at any angle, so the contact point typically has non-zero values in all dimensions.

The L1 constraint is a diamond with sharp corners that lie on the coordinate axes. At these corners, some weights are exactly zero. Because the loss contour approaches from an arbitrary direction, it's geometrically much more likely to first touch the diamond at a corner (where coordinates are zero) than on a flat edge.

From a Bayesian perspective: L1 corresponds to a Laplace prior, which has a sharp peak at zero and heavy tails. This prior says "most weights should be zero, but the ones that aren't can be large." L2 corresponds to a Gaussian prior, which has a smooth peak -- it says "weights should be small" but not exactly zero.

Practically: L1 performs automatic feature selection. If a feature's weight is driven to zero, that feature is eliminated. This makes L1 valuable for high-dimensional data with many irrelevant features.

### Q4: When would you use XGBoost over a neural network?

**Answer**: I'd use XGBoost when:

1. **Tabular data**: XGBoost handles heterogeneous features (mix of numerical, categorical, boolean) naturally without extensive preprocessing. Neural networks need careful encoding and normalization.

2. **Small to medium data** (1K - 100K samples): Trees ensemble methods are less prone to overfitting with limited data. Neural networks need more data to generalize.

3. **Interpretability needed**: Feature importance from XGBoost is directly actionable. Understanding what drives a neural network's predictions requires separate techniques (SHAP, LIME, attention analysis).

4. **Fast iteration**: XGBoost trains in minutes with minimal hyperparameter tuning. Neural networks require more careful architecture design, learning rate tuning, and training.

5. **Missing values**: XGBoost handles missing data natively by learning optimal split directions. Neural networks need imputation.

I'd switch to neural networks when:
- Working with text, images, audio, or sequences (where learned representations dominate)
- Having millions+ of training examples
- Transfer learning from a pretrained model is possible
- The data has spatial or sequential structure that architectures like CNNs or transformers exploit

The key insight: XGBoost is manual feature engineering + powerful modeling. Neural networks learn features automatically. When the raw features are already meaningful (tabular data), XGBoost shines. When features need to be learned from structure (language, vision), neural networks win.

### Q5: Explain Random Forest. Why is it better than a single decision tree?

**Answer**: A Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features considered at each split.

A single decision tree has high variance: small changes in the data can produce very different trees. It also tends to overfit -- an unrestricted tree will memorize the training data perfectly but generalize poorly.

Random Forest improves on this through two mechanisms:

**1. Bagging (bootstrap aggregating)**: Each tree is trained on a different random sample (with replacement) of the data. By averaging $B$ independent high-variance estimators, the variance of the ensemble is $\text{Var}_{\text{ensemble}} = \text{Var}_{\text{single}} / B$. More trees = lower variance.

**2. Feature randomization**: At each split, only a random subset of features (typically $\sqrt{d}$) is considered. Without this, all trees would split on the same strong features and be highly correlated. The variance of an average of correlated estimators is:

$$\text{Var}_{\text{ensemble}} = \rho \sigma^2 + \frac{(1 - \rho) \sigma^2}{B}$$

where $\rho$ is the correlation. Reducing correlation ($\rho$) via feature randomization makes averaging more effective.

The result: Random Forest has the same low bias as individual trees (deep trees can represent complex patterns) but much lower variance (bagging + decorrelation). It's essentially getting the best of both worlds -- the expressiveness of complex trees without their instability.

### Q6: How do you handle class imbalance? (e.g., 95% negative, 5% positive)

**Answer**: Class imbalance is common in real-world problems (fraud detection, rare disease diagnosis, harmful content detection in LLMs). Several strategies:

**Data-level approaches**:
- **Oversampling minority**: Duplicate or synthesize minority class examples (SMOTE creates synthetic examples by interpolating between existing minority points)
- **Undersampling majority**: Remove majority class examples (risk losing information)
- **Combined**: SMOTE + Tomek links (oversample minority + clean decision boundary)

**Algorithm-level approaches**:
- **Class weights**: Assign higher loss weight to minority class. In cross-entropy: $L = -[w_1 \cdot y \cdot \log(p) + w_0 \cdot (1-y) \cdot \log(1-p)]$. Setting $w_1 = n_{\text{neg}} / n_{\text{pos}}$ compensates for imbalance.
- **Focal loss**: Down-weight easy (well-classified) examples, focus on hard ones: $FL = -(1-p)^\gamma \log(p)$. Originally designed for object detection but applicable broadly.

**Evaluation-level approaches**:
- Don't use accuracy (misleading with imbalance)
- Use precision, recall, F1, PR-AUC
- Choose the metric that aligns with business cost (is a false positive or false negative more costly?)

**Threshold tuning**: The default 0.5 threshold is optimal only for balanced classes. With imbalance, tune the threshold based on the PR curve to match the desired precision-recall trade-off.

For LLM safety systems specifically: the cost of missing harmful content (false negative) is typically much higher than over-flagging (false positive). So optimize for high recall at an acceptable precision level.

### Q7: What is the difference between bagging and boosting?

**Answer**: Both are ensemble methods that combine many weak learners, but their philosophies are opposite:

**Bagging** (e.g., Random Forest):
- Trains models **in parallel** on different bootstrap samples
- Each model is independent -- could be trained on different machines
- Combines by **averaging** (regression) or **voting** (classification)
- Primarily **reduces variance** (since $\text{Var}(\text{average}) = \text{Var}(\text{individual}) / B$ for uncorrelated models)
- Uses **complex** base learners (deep, unpruned trees)
- Robust to overfitting: more trees never hurts

**Boosting** (e.g., XGBoost):
- Trains models **sequentially**, each correcting the previous ensemble's errors
- Each model depends on the previous -- must be sequential
- Combines by **weighted addition** (each model contributes proportionally to its accuracy)
- Primarily **reduces bias** (iteratively fitting residuals improves the approximation)
- Uses **simple** base learners (shallow trees, often depth 3-6)
- Can overfit if too many rounds or too high learning rate -- requires early stopping

The choice depends on the problem: if a simple model underfits (high bias), boosting will help. If a complex model overfits (high variance), bagging will help. In practice, gradient boosting (XGBoost, LightGBM) wins most tabular ML competitions because real-world data often has complex patterns that benefit from bias reduction.

### Q8: Explain the kernel trick in SVMs. Why is it useful?

**Answer**: Some data is not linearly separable in its original feature space. For example, two concentric circles of points from different classes -- no straight line separates them. But if we map to a higher dimension (e.g., adding a feature $r^2 = x_1^2 + x_2^2$), they become linearly separable in the new space.

The problem: explicitly computing the mapping $\boldsymbol{\phi}(\mathbf{x})$ can be extremely expensive (the RBF kernel maps to infinite dimensions).

The kernel trick exploits the fact that SVMs (in their dual form) only need dot products between data points, never the points themselves:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \boldsymbol{\phi}(\mathbf{x}_i)^T \boldsymbol{\phi}(\mathbf{x}_j)$$

If we can compute this dot product directly from the original features without ever computing $\boldsymbol{\phi}$, we get the benefit of the high-dimensional mapping for free. This works because certain kernel functions $K$ correspond to dot products in specific feature spaces.

The RBF kernel $K(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x}-\mathbf{y}\|^2}{2\sigma^2}\right)$ corresponds to an infinite-dimensional feature space, yet is a simple scalar computation in the original space. This lets SVMs find non-linear boundaries with the same computational complexity as linear SVMs.

### Q9: What is cross-validation and why is it important?

**Answer**: Cross-validation is a technique for evaluating model performance by training and testing on multiple different splits of the data.

In K-fold cross-validation: split the data into $K$ equal folds. For each fold, train on $K-1$ folds and evaluate on the held-out fold. Average the $K$ scores.

It's important for three reasons:

**1. Reliable performance estimate**: A single train/test split can be lucky (test set happens to be easy) or unlucky. K-fold gives $K$ estimates, whose mean is more reliable and whose variance tells you how stable the model is.

**2. Efficient use of limited data**: With a single 80/20 split, 20% of data is never used for training. With 5-fold CV, every sample is used for testing exactly once and for training 4 times. You get a reliable evaluation without "wasting" data.

**3. Hyperparameter selection**: Use CV to compare different hyperparameter settings (e.g., regularization strength, tree depth). Choose the setting with the best average CV score. This is more reliable than tuning on a single validation set.

**Critical caveat**: The test set must NEVER be used during model selection. Use CV on the training set for model selection, then evaluate the final model once on the held-out test set. Otherwise you're overfitting to the test set.

### Q10: Explain precision and recall. When would you optimize for each?

**Answer**: Both measure different aspects of a classifier's performance on the positive class.

**Precision** = $\frac{TP}{TP + FP}$: "Of everything I predicted positive, what fraction was actually positive?" High precision means few false alarms.

**Recall** = $\frac{TP}{TP + FN}$: "Of everything that was actually positive, what fraction did I catch?" High recall means few missed positives.

They trade off against each other. Lowering the classification threshold catches more positives (higher recall) but also includes more false positives (lower precision).

**Optimize for precision when** false positives are costly:
- Email spam filter: deleting a real email (FP) is worse than letting spam through (FN)
- Content recommendation: showing irrelevant content (FP) erodes user trust
- Autonomous driving alerts: too many false brake warnings cause driver alert fatigue

**Optimize for recall when** false negatives are costly:
- Medical screening: missing a cancer patient (FN) is worse than extra tests (FP)
- Safety filtering for LLMs: letting harmful content through (FN) is worse than over-filtering (FP)
- Fraud detection: missing fraud (FN) is financially costly

**F1 score** balances both and is useful when neither error type clearly dominates, or when you need a single summary metric.

In GenAI specifically: content safety classifiers optimize for very high recall (catch all harmful content) at the cost of some precision (occasional over-filtering). Retrieval in RAG systems optimize for high recall in the first stage (retrieve many candidates) then precision in the re-ranking stage (keep only the relevant ones).

---

*Next: [Topic 4: Deep Learning Foundations](04_Deep_Learning_Foundations.md)*
