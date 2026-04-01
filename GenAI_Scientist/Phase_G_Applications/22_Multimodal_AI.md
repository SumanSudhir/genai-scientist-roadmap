# Topic 22: Multimodal AI

> **Series**: Gen AI Scientist Interview Preparation
> **Topic**: 22 of 28
> **Scope**: CLIP (contrastive vision-language pretraining), LLaVA (visual instruction tuning), GPT-4V/Gemini (native multimodal), diffusion models (DDPM, DDIM, latent diffusion, Stable Diffusion, classifier-free guidance, ControlNet), video models (Sora), audio/speech (Whisper, TTS), multimodal embeddings and retrieval, cross-modal attention patterns
> **Why this matters**: The frontier of AI is multimodal — GPT-4o, Gemini, and Claude all process text, images, and audio. Understanding how modalities are aligned, how diffusion models generate images, and how vision-language models work is essential for AI Scientist interviews. Interviewers will ask you to explain CLIP's contrastive learning, derive the diffusion forward/reverse process, and design multimodal systems.
> **Previous**: [Topic 21: Agents & Tool Use](21_Agents_Tool_Use.md)
> **Next**: [Topic 23: Inference Optimization](23_Inference_Optimization.md)

---

## Table of Contents

1. [The Multimodal Vision — Why It Matters](#1-the-multimodal-vision--why-it-matters)
2. [CLIP — Contrastive Language-Image Pretraining](#2-clip--contrastive-language-image-pretraining)
3. [Vision Transformers (ViT) — Images as Token Sequences](#3-vision-transformers-vit--images-as-token-sequences)
4. [LLaVA & Visual Instruction Tuning](#4-llava--visual-instruction-tuning)
5. [GPT-4V, Gemini & Native Multimodal Models](#5-gpt-4v-gemini--native-multimodal-models)
6. [Diffusion Models — The Theory](#6-diffusion-models--the-theory)
7. [Latent Diffusion & Stable Diffusion](#7-latent-diffusion--stable-diffusion)
8. [Controlling Image Generation](#8-controlling-image-generation)
9. [Video Generation Models](#9-video-generation-models)
10. [Audio & Speech Models](#10-audio--speech-models)
11. [Multimodal Embeddings & Retrieval](#11-multimodal-embeddings--retrieval)
12. [Cross-Modal Attention Patterns](#12-cross-modal-attention-patterns)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. The Multimodal Vision — Why It Matters

### 1.1 From Text-Only to Any-to-Any

The evolution:

```
2017-2020: Text-only models (GPT, BERT)
2021:      CLIP aligns images + text
2022:      Stable Diffusion generates images from text
2023:      GPT-4V processes images + text together
2024:      GPT-4o processes text + images + audio natively
2025:      Sora generates video; models approach any-to-any
```

### 1.2 The Modality Landscape

| Modality | Input | Output | Key Models |
|----------|-------|--------|-----------|
| **Text** | ✓ | ✓ | GPT, Llama, Claude |
| **Image** | ✓ | ✓ | CLIP (understanding), Stable Diffusion (generation) |
| **Audio** | ✓ | ✓ | Whisper (ASR), VALL-E (TTS) |
| **Video** | ✓ | ✓ | VideoLLaVA (understanding), Sora (generation) |
| **Code** | ✓ | ✓ | Codex, CodeLlama |

### 1.3 Two Paradigms for Multimodality

**Paradigm 1: Separate encoders + alignment** (CLIP, LLaVA)
- Train separate encoders for each modality
- Learn to align their representation spaces
- More modular, easier to extend

**Paradigm 2: Native multimodal** (GPT-4o, Gemini)
- Single model processes all modalities natively
- Tokenize everything (images → visual tokens, audio → audio tokens)
- More integrated, potentially better cross-modal reasoning

---

## 2. CLIP — Contrastive Language-Image Pretraining

### 2.1 The Core Idea

CLIP (Radford et al., 2021) learns to align images and text in a **shared embedding space** using contrastive learning. An image of a cat and the text "a photo of a cat" should have similar embeddings; an image of a cat and "a photo of a car" should have different embeddings.

$$
f_{\text{image}}: \text{Image} \to \mathbb{R}^d, \quad f_{\text{text}}: \text{Text} \to \mathbb{R}^d
$$

$$
\text{sim}(\text{image}, \text{text}) = \cos(f_{\text{image}}(\text{img}), f_{\text{text}}(\text{txt}))
$$

### 2.2 Architecture

CLIP has two encoders trained jointly:

```
Image ──► Image Encoder (ViT or ResNet) ──► Image embedding ──┐
                                                                ├──► Cosine similarity
Text  ──► Text Encoder (Transformer)   ──► Text embedding  ──┘
```

- **Image encoder**: Vision Transformer (ViT-L/14) or ResNet-50
- **Text encoder**: 12-layer transformer (63M params)
- **Projection**: Both encoders project to a shared $d$-dimensional space (typically 512 or 768)

### 2.3 Contrastive Training (InfoNCE)

Given a batch of $N$ image-text pairs $\{(I_1, T_1), (I_2, T_2), \ldots, (I_N, T_N)\}$:

**Positive pairs**: $(I_i, T_i)$ — matching image and text
**Negative pairs**: $(I_i, T_j)$ for $i \neq j$ — non-matching

The loss (symmetric InfoNCE):

$$
\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
$$

$$
\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}
$$

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})
$$

where $\tau$ is a learned temperature parameter.

**Intuition**: For each image, maximize its similarity with the correct text and minimize similarity with all other texts in the batch. Vice versa for text.

### 2.4 The Similarity Matrix

For a batch of $N = 4$ pairs:

```
            T₁    T₂    T₃    T₄
      I₁  [ 0.9   0.1   0.2   0.1 ]   ← maximize diagonal
      I₂  [ 0.1   0.8   0.1   0.2 ]
      I₃  [ 0.2   0.1   0.9   0.1 ]
      I₄  [ 0.1   0.2   0.1   0.8 ]
                                         ← minimize off-diagonal
```

The diagonal should be high (matching pairs); off-diagonal should be low (non-matching). This is identical to a symmetric cross-entropy loss over the rows and columns.

### 2.5 Training Data

- **400 million** image-text pairs scraped from the internet (WebImageText dataset)
- Pairs are naturally occurring: images and their alt-text, captions, surrounding text
- No manual labeling — entirely self-supervised
- Huge diversity: photos, diagrams, memes, art, screenshots

### 2.6 Zero-Shot Classification

CLIP enables classification **without any training data** for the target task:

1. Create text prompts for each class: "a photo of a {class}"
2. Embed all class prompts: $\mathbf{t}_1, \mathbf{t}_2, \ldots, \mathbf{t}_K$
3. Embed the image: $\mathbf{v}$
4. Predict the class with highest similarity: $\hat{y} = \arg\max_k \cos(\mathbf{v}, \mathbf{t}_k)$

```
Image: [photo of a golden retriever]

Text prompts → embeddings:
  "a photo of a cat"     → t₁  (sim = 0.15)
  "a photo of a dog"     → t₂  (sim = 0.82)  ← highest
  "a photo of a car"     → t₃  (sim = 0.05)
  "a photo of a bird"    → t₄  (sim = 0.12)

Prediction: "dog"
```

**Remarkable result**: CLIP zero-shot matched a supervised ResNet-50 (trained on ImageNet's 1.28M labeled images) in top-1 accuracy — without seeing a single ImageNet training example.

### 2.7 Why CLIP Matters

1. **Foundation for multimodal AI**: CLIP's image encoder is used in Stable Diffusion, LLaVA, DALL-E, and many other systems
2. **Zero-shot transfer**: No task-specific training needed for classification
3. **Flexible prompting**: Change the class names or prompt templates to adapt to any task
4. **Robust representations**: CLIP embeddings transfer well across domains, styles, and distributions
5. **Enabled text-to-image generation**: CLIP's text encoder provides the semantic understanding that guides image generation in DALL-E and Stable Diffusion

---

## 3. Vision Transformers (ViT) — Images as Token Sequences

### 3.1 The Key Idea

ViT (Dosovitskiy et al., 2020) treats an image as a **sequence of patches**, just like a transformer treats text as a sequence of tokens:

$$
\text{Image} \to \text{Patches} \to \text{Linear projection} \to \text{Transformer encoder}
$$

### 3.2 How It Works

1. **Split image into patches**: A 224×224 image with patch size 16×16 → $14 \times 14 = 196$ patches
2. **Flatten each patch**: Each 16×16×3 patch → vector of dimension $16 \times 16 \times 3 = 768$
3. **Linear projection**: Project each patch vector to $d_{\text{model}}$
4. **Add position embeddings**: Learned 1D positional embeddings (the model learns 2D structure from data)
5. **Prepend [CLS] token**: A learnable class token, analogous to BERT's [CLS]
6. **Process with transformer encoder**: Standard self-attention layers
7. **Classification head**: MLP on the final [CLS] representation

```
Image (224×224):
┌──┬──┬──┬──┬──┬──┬──┐
│P1│P2│P3│P4│P5│P6│P7│   14 × 14 = 196 patches
├──┼──┼──┼──┼──┼──┼──┤   (each 16×16 pixels)
│P8│ ...                  │
   ...
└──┴──┴──┴──┴──┴──┴──┘

Patch sequence: [CLS] P₁ P₂ P₃ ... P₁₉₆
                  ↓    ↓   ↓   ↓       ↓
              Linear projection → d_model
                  +    +   +   +       +
              Position embeddings
                  ↓    ↓   ↓   ↓       ↓
              Transformer Encoder (L layers)
                  ↓
              [CLS] output → Classification head
```

### 3.3 ViT Variants

| Model | Patch Size | $d_{\text{model}}$ | Layers | Heads | Params |
|-------|-----------|-------------------|--------|-------|--------|
| ViT-B/16 | 16×16 | 768 | 12 | 12 | 86M |
| ViT-L/16 | 16×16 | 1024 | 24 | 16 | 307M |
| ViT-H/14 | 14×14 | 1280 | 32 | 16 | 632M |
| ViT-G/14 | 14×14 | 1664 | 48 | 16 | 1.8B |

### 3.4 Why ViT Matters for Multimodal AI

ViT provides a **universal architecture** for both text and images. Both are processed as token sequences through transformers. This architectural consistency enables:

- CLIP's image encoder is a ViT
- LLaVA connects a ViT to an LLM
- Multimodal models can share transformer layers across modalities

---

## 4. LLaVA & Visual Instruction Tuning

### 4.1 The Problem

CLIP aligns images and text, but it can't have a **conversation** about an image. You can't ask CLIP "What's happening in this image?" and get a natural language answer.

LLaVA (Liu et al., 2023) bridges this gap: connect a **vision encoder** to an **LLM** to create a visual chatbot.

### 4.2 Architecture

```
Image ──► Vision Encoder (CLIP ViT-L/14) ──► Visual features
                                                    │
                                            Projection Layer
                                            (Linear or MLP)
                                                    │
                                                    ▼
Text  ──► Text Tokenizer ──────────────────► Token embeddings
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │  LLM (Vicuna  │
                                            │  / Llama)     │
                                            │               │
                                            │ Visual tokens │
                                            │ + Text tokens │
                                            │ → Response    │
                                            └───────────────┘
```

**Three components**:

1. **Vision encoder**: Frozen CLIP ViT-L/14. Produces a grid of visual features (e.g., 576 tokens for 336×336 image with 14×14 patches)

2. **Projection layer**: A learned linear layer (or MLP) that maps visual features from the vision encoder's dimension to the LLM's embedding dimension:

$$
\mathbf{H}_v = \mathbf{W} \cdot f_{\text{ViT}}(\text{image}) + \mathbf{b} \quad \in \mathbb{R}^{N_v \times d_{\text{LLM}}}
$$

where $N_v$ is the number of visual tokens.

3. **LLM**: A pretrained language model (Vicuna, Llama) that processes the concatenation of visual tokens and text tokens.

### 4.3 Training Pipeline

LLaVA trains in two stages:

**Stage 1: Feature alignment** (pretrain the projection)
- Data: 558K image-text pairs from CC3M
- Only the projection layer is trained; ViT and LLM are frozen
- Objective: Align visual features with the LLM's embedding space
- Teaches the LLM to "understand" visual tokens

**Stage 2: Visual instruction tuning** (fine-tune LLM)
- Data: 158K visual instruction-following conversations (generated by GPT-4)
- Projection layer + LLM are both trained; ViT stays frozen
- Teaches the model to follow instructions about images

**The instruction data generation trick**: LLaVA's training data was created by:
1. Taking images with their COCO captions and bounding boxes
2. Asking GPT-4 (text-only) to generate conversations about the image *based on the captions*
3. Using these synthetic conversations as training data

This enabled creating high-quality visual instruction data without expensive human annotation.

### 4.4 LLaVA 1.5 & 1.6 Improvements

| Version | Vision Encoder | Resolution | LLM | Key Change |
|---------|---------------|-----------|-----|------------|
| LLaVA 1.0 | CLIP ViT-L/14 | 224×224 | Vicuna 7B/13B | Original |
| LLaVA 1.5 | CLIP ViT-L/14@336 | 336×336 | Vicuna 7B/13B | MLP projection, higher res |
| LLaVA 1.6 (NeXT) | CLIP ViT-L/14 | Dynamic (up to 672×672) | Mistral/Llama 3 | Dynamic resolution, multi-image |

### 4.5 The General VLM Architecture Pattern

LLaVA established a pattern that most Vision-Language Models (VLMs) follow:

$$
\text{Frozen Vision Encoder} + \text{Projection} + \text{LLM} = \text{VLM}
$$

| Model | Vision Encoder | Projection | LLM |
|-------|---------------|-----------|-----|
| LLaVA | CLIP ViT-L | Linear/MLP | Vicuna/Llama |
| InternVL | InternViT | QLLaMA | InternLM |
| Qwen-VL | ViT-G | Cross-attention resampler | Qwen |
| Idefics 2 | SigLIP | Perceiver resampler | Mistral |
| Phi-3 Vision | CLIP ViT | MLP | Phi-3 |

---

## 5. GPT-4V, Gemini & Native Multimodal Models

### 5.1 The Paradigm Shift

LLaVA-style models bolt a vision encoder onto an LLM. **Native multimodal models** process multiple modalities within a single architecture from the ground up.

### 5.2 GPT-4V (October 2023) / GPT-4o (May 2024)

**GPT-4V** (Vision): GPT-4 extended with image understanding
- Architecture details not published
- Likely uses a vision encoder feeding into the main transformer
- Capable of: image description, visual reasoning, OCR, chart reading, spatial understanding

**GPT-4o** (Omni): Natively multimodal — text, image, and audio in a single model
- Processes all modalities without separate encoders (reportedly)
- Real-time voice conversation with emotional expression
- End-to-end: audio in → audio out (no ASR/TTS pipeline)

### 5.3 Gemini (Google, December 2023)

Gemini was designed as **natively multimodal from the start**:

- Trained on multimodal data from the beginning (not retrofit)
- Processes text, images, audio, and video
- Gemini 1.5 Pro: 1M token context window (handling hours of video or thousands of pages)
- Architecture: Likely MoE-based with multimodal tokenization

**Key capabilities**:
- Video understanding (analyze entire movies)
- Multi-image reasoning (compare charts, track changes)
- Audio understanding (transcribe, analyze, respond to speech)
- Long context: Process a 10-hour video and answer questions about specific moments

### 5.4 Bolt-On vs Native Multimodal

| Aspect | Bolt-On (LLaVA) | Native (GPT-4o, Gemini) |
|--------|-----------------|------------------------|
| Architecture | Frozen vision encoder + projection + LLM | Single model, all modalities |
| Training | Two-stage (alignment then instruction tuning) | Joint training from scratch |
| Cross-modal reasoning | Limited by projection bottleneck | Deep integration across layers |
| Flexibility | Can swap encoders/LLMs | Monolithic |
| Cost to build | Low (leverage existing models) | Very high (train from scratch) |
| Quality | Good | Best |
| Open-source | Many available | Few (Gemma, Fuyu) |

---

## 6. Diffusion Models — The Theory

### 6.1 The Core Idea

Diffusion models generate images by learning to **reverse a noise-adding process**:

1. **Forward process**: Gradually add Gaussian noise to an image until it's pure noise
2. **Reverse process**: Learn to gradually remove noise, recovering the image

$$
\text{Clean image} \xrightarrow{\text{add noise (known)}} \text{Pure noise} \xrightarrow{\text{remove noise (learned)}} \text{Clean image}
$$

### 6.2 The Forward Process (Diffusion)

Starting with a clean image $\mathbf{x}_0$, add Gaussian noise over $T$ timesteps:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

where $\beta_t$ is the noise schedule (typically $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$).

**Key property**: We can jump directly to any timestep $t$ without iterating:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

This means:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

A weighted combination of the clean image and pure noise. At $t = 0$: mostly signal. At $t = T$: mostly noise.

```
t=0         t=T/4        t=T/2        t=3T/4       t=T
[Clean]     [Slightly    [Half         [Mostly      [Pure
 image]      noisy]       noisy]        noise]       noise]
████████    ████▓▓▓▓    ▓▓▓▓▓▓▓▓    ░░░░░░░░    ░░░░░░░░
████████    ▓▓████▓▓    ▓▓░▓▓▓▓░    ░░░░░░░░    ░░░░░░░░
████████    ████▓▓▓▓    ░▓▓▓░▓▓▓    ░░░░░░░░    ░░░░░░░░
```

### 6.3 The Reverse Process (Denoising)

The reverse process learns to undo each noise step:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})
$$

A neural network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ predicts the noise that was added at step $t$, and we use it to compute the mean:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

### 6.4 The Training Objective (Simplified)

Train the noise predictor $\boldsymbol{\epsilon}_\theta$ to predict the noise that was added:

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

**Training algorithm**:
1. Sample a clean image $\mathbf{x}_0$ from the dataset
2. Sample a random timestep $t \sim \text{Uniform}(1, T)$
3. Sample noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
4. Create noisy image: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$
5. Train: minimize $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$

### 6.5 Sampling (Generation)

To generate a new image:

1. Start with pure noise: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. For $t = T, T-1, \ldots, 1$:
   - Predict noise: $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - Compute $\mathbf{x}_{t-1}$ using the denoising formula
   - Add a small amount of noise (for stochasticity): $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta + \sigma_t \mathbf{z}$
3. Return $\mathbf{x}_0$ (the clean image)

This requires $T$ forward passes through the network (typically $T = 1000$). Slow but produces high-quality images.

### 6.6 The U-Net Architecture

The noise prediction network $\boldsymbol{\epsilon}_\theta$ is typically a **U-Net**: an encoder-decoder with skip connections.

```
Input (noisy image + timestep embedding)
  │
  ▼
[Encoder: Conv → Downsample → Conv → Downsample → ...]
  │                                                  │
  │              Skip connections                     │
  │                                                  │
  ▼                                                  │
[Bottleneck: Self-attention + Cross-attention]        │
  │                                                  │
  ▼                                                  │
[Decoder: Conv → Upsample → Conv → Upsample → ...]  │
  │              ◄── skip connections ────────────────┘
  ▼
Output (predicted noise, same size as input)
```

The U-Net includes:
- **Timestep conditioning**: Sinusoidal embedding of $t$ added to feature maps
- **Self-attention layers**: Allow global context (important for coherent images)
- **Cross-attention layers**: For text conditioning (text guides the generation)
- **Skip connections**: Preserve fine-grained spatial detail

### 6.7 DDIM — Faster Sampling

DDPM (Denoising Diffusion Probabilistic Models) requires $T = 1000$ steps. **DDIM** (Song et al., 2021) makes the process **deterministic** and allows **skipping steps**:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\boldsymbol{\epsilon}}}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted } \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \hat{\boldsymbol{\epsilon}}
$$

**Key insight**: DDIM defines a **non-Markovian** process (each step doesn't need the previous step's stochasticity). This allows taking larger steps: instead of $T = 1000$ steps, use 50-100 steps with similar quality.

| Method | Steps | Quality | Speed |
|--------|-------|---------|-------|
| DDPM | 1000 | Excellent | ~30 seconds |
| DDIM | 50-100 | Very good | ~2-5 seconds |
| DPM-Solver | 20-25 | Good | ~1 second |

---

## 7. Latent Diffusion & Stable Diffusion

### 7.1 The Problem with Pixel-Space Diffusion

Running diffusion in pixel space (e.g., 512×512×3 = 786K dimensions) is:
- **Computationally expensive**: The U-Net must process very high-dimensional tensors
- **Slow**: Many denoising steps on large tensors
- **Memory-intensive**: Storing intermediate activations for a 512×512 U-Net requires gigabytes

### 7.2 The Latent Diffusion Solution

**Latent Diffusion Models (LDM)** (Rombach et al., 2022) run the diffusion process in a **compressed latent space** instead of pixel space:

$$
\text{Image} \xrightarrow{\text{Encoder } \mathcal{E}} \text{Latent } \mathbf{z} \xrightarrow{\text{Diffusion}} \text{Denoised } \hat{\mathbf{z}} \xrightarrow{\text{Decoder } \mathcal{D}} \text{Image}
$$

**The autoencoder**:
- **Encoder** $\mathcal{E}$: Compresses 512×512×3 image → 64×64×4 latent (factor of 48× compression)
- **Decoder** $\mathcal{D}$: Reconstructs 64×64×4 latent → 512×512×3 image
- Trained separately using reconstruction + KL divergence loss (VAE)

**The diffusion model** operates in this 64×64×4 latent space:
- 48× fewer dimensions than pixel space
- Much faster training and inference
- Same image quality (the autoencoder preserves perceptual quality)

```
Pixel space:   512 × 512 × 3 = 786,432 dimensions
                        │
                   Encoder (↓8×)
                        │
Latent space:   64 × 64 × 4 = 16,384 dimensions (48× smaller)
                        │
              Diffusion process here
                        │
                   Decoder (↑8×)
                        │
Output:        512 × 512 × 3
```

### 7.3 Stable Diffusion Architecture

**Stable Diffusion** (Stability AI) is an open-source implementation of latent diffusion:

```
┌────────────────────────────────────────────────────────┐
│                   Stable Diffusion                      │
│                                                        │
│  Text prompt ──► CLIP Text Encoder ──► Text embeddings │
│                                            │           │
│                                     Cross-attention    │
│                                            │           │
│  Random noise ──► U-Net (in latent space) ──► Denoised │
│  (64×64×4)       (iterative denoising)       latent    │
│                                                │       │
│                                         VAE Decoder    │
│                                                │       │
│                                          Final image   │
│                                          (512×512)     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Components**:
1. **CLIP text encoder**: Converts text prompt → text embeddings (77 tokens × 768 dims)
2. **U-Net**: Predicts noise in latent space, conditioned on text via cross-attention
3. **VAE decoder**: Converts denoised latent → pixel image

### 7.4 Text Conditioning via Cross-Attention

The text guides image generation through **cross-attention** in the U-Net:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}
$$

where:
- $\mathbf{Q}$ = queries from the image features (spatial features of the noisy latent)
- $\mathbf{K}, \mathbf{V}$ = keys and values from the text embeddings

Each spatial location in the image "asks" the text for guidance on what to generate. The word "cat" in the prompt provides high attention to spatial regions where the cat should appear.

### 7.5 Stable Diffusion Versions

| Version | Date | Resolution | Key Innovation |
|---------|------|-----------|----------------|
| SD 1.5 | 2022 | 512×512 | Original latent diffusion |
| SD 2.1 | 2022 | 512-768 | OpenCLIP encoder, v-prediction |
| SDXL | 2023 | 1024×1024 | Two-stage (base + refiner), larger U-Net |
| SD 3 | 2024 | Variable | **DiT** (Diffusion Transformer) replaces U-Net |
| Flux | 2024 | Variable | Rectified flow, improved DiT |

### 7.6 Diffusion Transformers (DiT)

The latest trend replaces the U-Net with a **pure transformer** for the denoising network:

$$
\text{U-Net (convolutions + attention)} \longrightarrow \text{DiT (pure transformer)}
$$

**How DiT works**:
1. Patchify the latent: split 64×64×4 into patches → sequence of tokens
2. Add position embeddings and timestep/class conditioning
3. Process through transformer layers (self-attention + FFN)
4. Un-patchify back to latent dimensions

**Why DiT matters**: Transformers scale better than U-Nets. The same scaling laws that worked for LLMs (more parameters + more data = better) apply to DiT. SD 3, Flux, and Sora all use DiT architectures.

---

## 8. Controlling Image Generation

### 8.1 Classifier-Free Guidance (CFG)

The most important technique for improving text-to-image generation quality.

**Problem**: The model may generate images that match the text prompt weakly. How to strengthen the text conditioning?

**Idea**: During training, randomly drop the text conditioning (replace with null/empty embedding) with probability $p$ (typically 10%). This trains both a **conditional** model $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)$ and an **unconditional** model $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$ simultaneously.

At inference, extrapolate **away** from the unconditional prediction toward the conditional one:

$$
\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot \left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right)
$$

where $w$ is the **guidance scale** (typically 7-15).

**Intuition**: The difference $(\boldsymbol{\epsilon}_{\text{cond}} - \boldsymbol{\epsilon}_{\text{uncond}})$ represents the "direction toward the text." Multiplying by $w > 1$ pushes the generation more strongly in that direction.

```
w = 1:  Neutral (standard conditional generation)
w = 7:  Good balance (typical for Stable Diffusion)
w = 15: Very strong text adherence (may sacrifice diversity)
w = 30: Over-saturated, artifact-prone
```

**Cost**: CFG requires **two** forward passes per denoising step (conditional + unconditional), doubling inference time.

### 8.2 ControlNet

**Problem**: Text prompts give high-level guidance but limited spatial control. "A cat sitting on a mat" doesn't specify where the cat is.

**ControlNet** (Zhang et al., 2023) adds **spatial control** through additional conditioning inputs:

| Control Type | Input | Use Case |
|-------------|-------|----------|
| **Canny edge** | Edge map of desired layout | Preserve structure of a reference image |
| **Depth map** | Depth information | Control 3D composition |
| **Pose** | Human skeleton keypoints | Control human pose in generated image |
| **Segmentation** | Semantic segmentation map | Control which regions contain which objects |
| **Scribble** | Rough hand-drawn sketch | Generate from simple drawings |

**Architecture**: ControlNet creates a **trainable copy** of the U-Net encoder. The control input is processed by this copy, and its features are added to the main U-Net via zero convolutions (initialized to zero, gradually learning):

```
Control input ──► ControlNet (copy of U-Net encoder)
                        │
                   Zero convolution (learned, starts at 0)
                        │
                        ▼
Noisy latent ──► U-Net encoder ──► + ──► U-Net decoder ──► Denoised latent
```

### 8.3 IP-Adapter (Image Prompt Adapter)

**Problem**: Want to generate images in the *style* of a reference image, guided by text.

**IP-Adapter**: Feeds a reference image through CLIP's image encoder and injects those features via cross-attention (parallel to text cross-attention):

$$
\text{Output} = \text{softmax}\left(\frac{Q K_{\text{text}}^T}{\sqrt{d}}\right)V_{\text{text}} + \lambda \cdot \text{softmax}\left(\frac{Q K_{\text{image}}^T}{\sqrt{d}}\right)V_{\text{image}}
$$

This allows "style transfer" controlled by both a text prompt and a reference image.

### 8.4 Other Control Methods

| Method | Control | How It Works |
|--------|---------|-------------|
| **Textual Inversion** | New concepts | Learn a new embedding for a custom concept from 3-5 images |
| **DreamBooth** | Personalization | Fine-tune the entire model on your specific subject |
| **LoRA** | Style/concept | Low-rank adaptation of the U-Net or text encoder |
| **Inpainting** | Edit specific regions | Mask + regenerate portions of an image |
| **Img2Img** | Transform existing images | Start denoising from a partially noised version of a source image |

---

## 9. Video Generation Models

### 9.1 Sora (OpenAI, 2024)

Sora is the most prominent video generation model, generating up to 60 seconds of high-fidelity video from text prompts.

**Architecture** (based on public information):
- **Diffusion Transformer (DiT)** operating in a video latent space
- Video is encoded into **spacetime patches** — 3D patches that span spatial and temporal dimensions
- The transformer processes these patches as a sequence, similar to how ViT processes image patches

```
Video (T frames × H × W × 3)
       │
       ▼
Video Encoder (compress spatially + temporally)
       │
       ▼
Spacetime latent patches (sequence of tokens)
       │
       ▼
DiT (Diffusion Transformer)
  - Self-attention across all spacetime patches
  - Cross-attention with text conditioning
  - Iterative denoising
       │
       ▼
Video Decoder
       │
       ▼
Generated video
```

**Key properties**:
- Variable resolution and duration (not fixed to specific sizes)
- Temporal consistency: objects maintain identity across frames
- Physics understanding: basic physics simulation (water, reflections, motion)

### 9.2 Other Video Models

| Model | Approach | Duration | Key Feature |
|-------|---------|----------|-------------|
| **Sora** (OpenAI) | DiT, spacetime patches | Up to 60s | Highest quality, physical understanding |
| **Runway Gen-3** | Diffusion-based | ~10s | Production-ready, fast |
| **Pika** | Diffusion-based | ~4s | Easy to use, motion control |
| **Stable Video Diffusion** | Latent diffusion + temporal layers | ~4s | Open-source |
| **Kling** (Kuaishou) | DiT-like | ~10s | Strong motion, Chinese origin |

### 9.3 Challenges in Video Generation

1. **Temporal consistency**: Objects shouldn't change appearance between frames (identity preservation)
2. **Physics**: Water should flow, objects should fall, shadows should be consistent
3. **Compute**: A 10-second 24fps video at 1080p = 240 frames × 1920 × 1080 pixels — enormous latent space
4. **Long-range coherence**: Maintaining a narrative over 60 seconds requires understanding causality
5. **Evaluation**: No standard metrics for video quality (FVD is imperfect)

---

## 10. Audio & Speech Models

### 10.1 Whisper — Speech Recognition

**Whisper** (Radford et al., 2023) is OpenAI's speech recognition model. It's an **encoder-decoder transformer** trained on 680,000 hours of multilingual audio.

**Architecture**:

```
Audio waveform ──► Mel spectrogram ──► Encoder (Transformer) ──► Audio features
                                                                      │
                                                               Cross-attention
                                                                      │
                               Text output ◄── Decoder (Transformer) ──┘
```

1. **Audio preprocessing**: Convert waveform to 80-channel log-mel spectrogram
2. **Encoder**: Transformer with convolutional downsampling (2 conv layers → 2× time reduction)
3. **Decoder**: Autoregressive transformer that generates text tokens, cross-attending to encoder
4. **Multi-task**: Single model handles transcription, translation, language detection, timestamp prediction

**Whisper model sizes**:

| Model | Params | Layers | $d$ | Heads | WER (en) |
|-------|--------|--------|-----|-------|----------|
| Tiny | 39M | 4 enc + 4 dec | 384 | 6 | ~7.7% |
| Base | 74M | 6 + 6 | 512 | 8 | ~5.0% |
| Small | 244M | 12 + 12 | 768 | 12 | ~3.4% |
| Medium | 769M | 24 + 24 | 1024 | 16 | ~2.9% |
| Large-v3 | 1.5B | 32 + 32 | 1280 | 20 | ~2.0% |

**Why Whisper matters**: Robust across accents, noise levels, and languages. Open-source. Established encoder-decoder transformer as the standard ASR architecture.

### 10.2 Text-to-Speech (TTS)

Modern TTS generates natural-sounding speech from text:

| Model | Approach | Key Feature |
|-------|---------|-------------|
| **VALL-E** (Microsoft, 2023) | Neural codec LM | Clone any voice from 3-second sample |
| **Bark** (Suno) | GPT-like autoregressive | Music, sound effects, multilingual |
| **XTTS** (Coqui) | GPT-2 + DVAE | Open-source, cross-lingual voice cloning |
| **StyleTTS 2** | Diffusion-based | Studio-quality, style control |
| **Parler TTS** | Text-described style | "A female speaker with a warm tone..." |

**VALL-E architecture**:

$$
\text{Text} + \text{3s voice sample} \xrightarrow{\text{Neural codec LM}} \text{Audio codec tokens} \xrightarrow{\text{Codec decoder}} \text{Speech}
$$

1. Audio is tokenized using a neural codec (EnCodec) into discrete tokens
2. A transformer language model generates audio tokens conditioned on text + voice prompt
3. The codec decoder converts tokens back to audio waveform

This frames TTS as a **language modeling problem** — generate audio tokens the same way we generate text tokens.

### 10.3 Audio Understanding

| Model | Task | Approach |
|-------|------|---------|
| **AudioLM** (Google) | Audio continuation | Predict next audio tokens |
| **MusicLM** (Google) | Text-to-music | Hierarchical music generation |
| **Suno** | Text-to-song (vocals + music) | Multi-stage generation |
| **Qwen-Audio** | Audio understanding (QA, classification) | Audio encoder + LLM |

---

## 11. Multimodal Embeddings & Retrieval

### 11.1 Shared Embedding Spaces

CLIP demonstrated that different modalities can be embedded in a **shared space** where cross-modal similarity is meaningful:

$$
\text{sim}(\text{image of cat}, \text{"a photo of a cat"}) \gg \text{sim}(\text{image of cat}, \text{"a photo of a car"})
$$

### 11.2 Beyond CLIP: Multi-Modal Embedding Models

| Model | Modalities | Dimensions | Key Feature |
|-------|-----------|-----------|-------------|
| **CLIP** | Image + Text | 512/768 | Pioneer, foundation |
| **SigLIP** | Image + Text | 768-1152 | Sigmoid loss (no softmax), better at scale |
| **ImageBind** (Meta) | Image, Text, Audio, Depth, Thermal, IMU | 1024 | 6 modalities in one space |
| **CLAP** | Audio + Text | 512 | CLIP for audio |
| **ONE-PEACE** | Image, Audio, Text | 1536 | Unified architecture |

### 11.3 ImageBind — Binding All Modalities

**Key insight**: Use images as the "anchor" modality. If you align text→image and audio→image, then text and audio are implicitly aligned (through the shared image space):

$$
\text{Text} \xleftrightarrow{\text{CLIP}} \text{Image} \xleftrightarrow{\text{trained}} \text{Audio}
$$

$$
\implies \text{Text} \xleftrightarrow{\text{emergent}} \text{Audio}
$$

ImageBind trains separate encoders for each modality to align with images using contrastive learning. The result: any-to-any retrieval across 6 modalities without explicit pairing between non-image modalities.

### 11.4 Multimodal RAG

Extend RAG to handle images, tables, and diagrams:

```
Document (with images + text + tables)
       │
       ▼
Extract: text chunks + image descriptions + table summaries
       │
       ▼
Embed all with multimodal embedding model
       │
       ▼
Store in vector database
       │
       ▼
Query: "What does the revenue chart show?"
       │
       ▼
Retrieve: matching text chunks + relevant chart image
       │
       ▼
Send to multimodal LLM (GPT-4o) with image + text context
       │
       ▼
Answer grounded in the actual chart
```

---

## 12. Cross-Modal Attention Patterns

### 12.1 How Modalities Interact in Transformers

Different architectures use different attention patterns to combine modalities:

### 12.2 Pattern 1: Early Fusion

All modalities are tokenized and concatenated into a single sequence. Standard self-attention processes everything together.

$$
\text{Input} = [\text{text tokens}; \text{visual tokens}; \text{audio tokens}]
$$

```
[text₁] [text₂] [img₁] [img₂] [img₃] [audio₁] [audio₂]
   ↕       ↕       ↕       ↕       ↕       ↕        ↕
        Full self-attention across all tokens
```

**Used by**: GPT-4o (likely), Gemini, Fuyu

**Advantage**: Maximum cross-modal interaction — any token can attend to any other.
**Disadvantage**: Quadratic cost in total sequence length (all modalities combined).

### 12.3 Pattern 2: Cross-Attention Fusion

Separate encoders for each modality, connected via cross-attention:

```
Image ──► Vision Encoder ──► K, V (visual features)
                                    │
                              Cross-attention
                                    │
Text  ──► Text Encoder ──► Q ──────►+ ──► Fused representation
```

**Used by**: Stable Diffusion (text → image cross-attention), Whisper (audio → text cross-attention), Flamingo

**Advantage**: Modality-specific encoders can be optimized independently.
**Disadvantage**: Cross-modal interaction limited to cross-attention layers.

### 12.4 Pattern 3: Projection + Concatenation (Late Fusion)

Encode each modality independently, project to a shared dimension, concatenate, then process with a shared model:

$$
\mathbf{H} = [\text{Proj}_{\text{vis}}(f_{\text{vis}}(\text{img})); \text{Proj}_{\text{text}}(f_{\text{text}}(\text{txt}))]
$$

**Used by**: LLaVA (project visual features into LLM embedding space)

**Advantage**: Can leverage pretrained unimodal encoders. Simple.
**Disadvantage**: Visual and text tokens don't interact until the shared model.

### 12.5 Pattern 4: Perceiver / Resampler

Use a fixed number of **learnable query tokens** to attend to variable-length modality features:

$$
\text{Queries (learned, fixed count)} \xrightarrow{\text{cross-attend}} \text{Visual features (variable count)} \to \text{Fixed visual tokens}
$$

**Used by**: Flamingo (Perceiver Resampler), Qwen-VL, Idefics 2

**Advantage**: Controls the number of visual tokens fed to the LLM (e.g., compress 576 ViT tokens to 64 query tokens). Reduces compute.
**Disadvantage**: Information compression may lose detail.

### 12.6 Comparison

| Pattern | Cross-Modal Depth | Compute | Flexibility | Example |
|---------|------------------|---------|-------------|---------|
| Early Fusion | Deep (all layers) | High | Low (fixed) | GPT-4o |
| Cross-Attention | Medium (at cross-attn layers) | Medium | High | Stable Diffusion |
| Projection + Concat | Shallow (after projection) | Low | High | LLaVA |
| Perceiver | Controlled | Configurable | High | Flamingo |

---

## 13. Interview Questions & Answers

### Q1: How does CLIP align images and text? What is contrastive learning?

**A**: CLIP learns a shared embedding space where matching image-text pairs are close and non-matching pairs are far apart.

**Architecture**: Two encoders — a Vision Transformer for images and a Transformer for text — both projecting to the same $d$-dimensional space.

**Contrastive learning**: Given a batch of $N$ image-text pairs, create an $N \times N$ similarity matrix. The diagonal contains positive pairs (matching); off-diagonal contains negatives. The loss pushes diagonal similarities high and off-diagonal low:

$$
\mathcal{L}_i = -\log \frac{\exp(\cos(\mathbf{v}_i, \mathbf{t}_i) / \tau)}{\sum_{j=1}^{N} \exp(\cos(\mathbf{v}_i, \mathbf{t}_j) / \tau)}
$$

This is computed symmetrically (image→text and text→image) and averaged.

**Why it works**: With large batches ($N = 32768$ in CLIP), each positive pair is contrasted against thousands of negatives. The encoders are forced to learn representations that capture the *semantic content* shared between images and text, not superficial features.

**Key properties of the resulting space**:
- Zero-shot classification: Embed class descriptions as text, classify by nearest text embedding
- Cross-modal retrieval: Search images with text queries (or vice versa)
- Foundation for generation: CLIP's text encoder guides Stable Diffusion's image generation

---

### Q2: Explain the diffusion process. What are the forward and reverse steps?

**A**:

**Forward process** (fixed, no learning): Gradually add Gaussian noise to a clean image $\mathbf{x}_0$ over $T$ timesteps:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ decreases from ~1 to ~0 over $T$ steps. At $t = 0$: clean image. At $t = T$: pure Gaussian noise. The forward process is known analytically — no neural network needed.

**Reverse process** (learned): A neural network $\boldsymbol{\epsilon}_\theta$ learns to predict the noise that was added at each step. Starting from pure noise $\mathbf{x}_T$, iteratively denoise:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}
$$

**Training**: Sample random $(\mathbf{x}_0, t, \boldsymbol{\epsilon})$, create noisy $\mathbf{x}_t$, train $\boldsymbol{\epsilon}_\theta$ to predict $\boldsymbol{\epsilon}$:

$$
\mathcal{L} = \mathbb{E}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

**Key insight**: The model doesn't generate an image in one shot. It performs gradual, iterative refinement — each step making the image slightly less noisy. This iterative nature allows for extremely high-quality generation, because each step only needs to make a small correction.

---

### Q3: How does LLaVA connect a vision encoder to an LLM? What is visual instruction tuning?

**A**: LLaVA's architecture has three components:

1. **Frozen CLIP ViT**: Processes the image into a grid of visual feature vectors (e.g., 576 tokens for a 336×336 image)

2. **Learned projection layer**: A linear layer (or 2-layer MLP) that maps each visual feature from CLIP's dimension to the LLM's embedding dimension:

$$
\mathbf{h}_v^i = \text{MLP}(\mathbf{z}_v^i), \quad \mathbf{z}_v^i \in \mathbb{R}^{d_{\text{CLIP}}}, \quad \mathbf{h}_v^i \in \mathbb{R}^{d_{\text{LLM}}}
$$

3. **LLM**: Receives a sequence of [visual tokens; text tokens] and generates a response. The visual tokens are treated exactly like text tokens — the LLM's self-attention operates over both uniformly.

**Visual instruction tuning** is the fine-tuning process that teaches the model to follow visual instructions:

**Stage 1** (alignment): Train only the projection layer on image-caption pairs. This teaches the LLM to "read" visual tokens — mapping visual features into the LLM's semantic space.

**Stage 2** (instruction tuning): Fine-tune the projection + LLM on visual conversation data. Examples:
- "Describe this image in detail" → detailed description
- "What's happening in the top-right corner?" → spatial reasoning
- "Is this image appropriate for children?" → safety judgment

The training data for Stage 2 was generated synthetically using GPT-4 — given image captions and bounding boxes, GPT-4 generated realistic visual conversations. This avoided expensive human annotation.

---

### Q4: What is classifier-free guidance? Why does it improve image quality?

**A**: Classifier-free guidance (CFG) strengthens the influence of the text condition on generated images without requiring a separate classifier.

**How it works**:

During training, randomly replace the text condition with a null condition (empty string) with probability $p \approx 0.1$. This trains the model to generate both conditionally and unconditionally.

At inference, compute two noise predictions and extrapolate:

$$
\hat{\boldsymbol{\epsilon}} = \underbrace{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)}_{\text{unconditional}} + w \cdot \underbrace{\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right)}_{\text{direction toward text condition}}
$$

**Why it works**: The vector $(\boldsymbol{\epsilon}_{\text{cond}} - \boldsymbol{\epsilon}_{\text{uncond}})$ represents the "direction of the text prompt" in noise space. Multiplying by $w > 1$ amplifies this direction, making the generated image more strongly aligned with the text.

At $w = 1$: standard conditional generation.
At $w = 7$-$15$: images are sharper, more coherent, and better match the text.
At $w > 20$: over-saturation and artifacts (the text signal is amplified too much).

**The trade-off**: CFG improves **text-image alignment** and **image quality** at the cost of **diversity** — higher guidance makes all generations converge toward a "prototypical" interpretation of the text, reducing variation.

**Cost**: Requires two forward passes per denoising step (one conditional, one unconditional), doubling compute.

---

### Q5: Compare latent diffusion (Stable Diffusion) with pixel-space diffusion. Why work in latent space?

**A**: Pixel-space diffusion (DDPM) runs the denoising process on full-resolution images (e.g., 512×512×3 = 786K dimensions). Latent diffusion (LDM/Stable Diffusion) first compresses images to a latent space (e.g., 64×64×4 = 16K dimensions) and runs diffusion there.

**Why latent space is better**:

1. **48× dimensionality reduction**: From 786K to 16K. This makes the U-Net/DiT much smaller and faster.

2. **Perceptually equivalent**: The VAE autoencoder compresses the image while preserving perceptual quality. High-frequency details that don't affect perception (exact pixel values, noise patterns) are discarded — the diffusion model only needs to model the perceptually meaningful structure.

3. **Faster training**: Each denoising step operates on a 64×64 tensor instead of 512×512. Training is approximately 10× faster.

4. **Faster inference**: Each denoising step is 10× cheaper. Combined with fewer steps (DDIM), generation takes 2-5 seconds instead of 30+ seconds.

5. **Memory efficiency**: The U-Net for 64×64 latents fits comfortably on consumer GPUs (8GB). Pixel-space U-Nets for 512×512 require 40GB+.

**The cost of latent space**: The VAE introduces a small amount of reconstruction error. Fine details (e.g., text in images, small faces) can be slightly blurred by the compression. This is why Stable Diffusion historically struggled with readable text in images — the VAE couldn't perfectly reconstruct sharp text.

---

### Q6: How does Whisper work for speech recognition? Why is it an encoder-decoder?

**A**: Whisper is a transformer encoder-decoder trained on 680K hours of audio for speech recognition, translation, and language identification.

**Pipeline**:
1. Audio waveform → 80-channel log-mel spectrogram (30-second chunks)
2. Spectrogram → Encoder (transformer): produces audio feature representations
3. Audio features → Decoder (autoregressive transformer): generates text tokens via cross-attention

**Why encoder-decoder**:

Speech recognition is a **sequence-to-sequence** task where the input (audio) and output (text) are fundamentally different modalities with different lengths and structures:

- Audio: continuous signal, 16KHz sample rate → 480K samples for 30 seconds
- Text: discrete tokens, typically 10-100 tokens for 30 seconds of speech
- The ratio is ~5000:1 (audio samples to text tokens)

The encoder processes the full audio **bidirectionally** — understanding the complete utterance before generating any text. This is crucial because:
- Speech is ambiguous: "ice cream" vs "I scream" can only be resolved with full context
- Word boundaries aren't explicit in audio
- The decoder generates much shorter sequences, making autoregressive generation efficient

**Multi-task training**: Whisper uses special tokens to control behavior:
```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world
<|startoftranscript|><|fr|><|translate|><|notimestamps|> Bonjour le monde
<|startoftranscript|><|en|><|transcribe|><|0.00|> Hello<|0.52|> world<|1.04|>
```

This single model handles transcription, translation, language detection, and timestamp prediction.

---

### Q7: What is the DiT (Diffusion Transformer) architecture? Why is it replacing U-Nets?

**A**: DiT replaces the traditional U-Net denoising network with a **pure transformer**:

1. **Patchify**: Split the noisy latent (64×64×4) into patches (e.g., 2×2 → 32×32 = 1024 patches)
2. **Project + add position embeddings**: Each patch → token vector
3. **Condition**: Add timestep embedding and class/text conditioning (via adaptive layer norm or cross-attention)
4. **Process through transformer blocks**: Standard self-attention + FFN
5. **Un-patchify**: Reconstruct the latent-shaped output (predicted noise)

**Why DiT is replacing U-Nets**:

1. **Scaling laws**: Transformers have well-understood scaling laws. U-Nets don't — it's unclear how to scale a U-Net optimally. DiT follows the same "bigger model + more data = better" pattern as LLMs.

2. **Simplicity**: U-Nets have complex skip connections, multiple resolution stages, and convolutional blocks. DiT is a standard transformer — the same architecture used for LLMs. This simplifies engineering and enables code reuse.

3. **Architecture unification**: If both the language model and the image generation model are transformers, they can potentially share infrastructure, optimization techniques, and even parameters. This is the path toward unified multimodal models.

4. **Better quality at scale**: DiT-XL/2 (the large version) outperformed all U-Net-based diffusion models at the time of publication, with a cleaner FID (Fréchet Inception Distance) scaling curve.

**Who uses DiT**: SD 3, Flux, Sora, and most recent frontier image/video generation models.

---

### Q8: How would you build a multimodal RAG system that handles documents with images, tables, and text?

**A**: The challenge: traditional RAG only handles text. Documents often contain critical information in images (charts, diagrams) and tables that text retrieval misses.

**Indexing pipeline**:

1. **Document parsing**: Use document understanding models to extract:
   - Text blocks with positions
   - Images with captions and surrounding text
   - Tables converted to structured text (markdown or CSV)

2. **Multi-modal chunking**:
   - Text: Standard chunking (512 tokens, 50 overlap)
   - Images: Generate text descriptions using a VLM (GPT-4o or LLaVA). Store both the description (for retrieval) and the original image (for the LLM)
   - Tables: Convert to markdown + generate a natural language summary. Index both

3. **Embedding**: Use a text embedding model for all chunks (text, image descriptions, table summaries). Optionally use a multimodal embedding model (CLIP) for direct image embedding

4. **Storage**: Vector DB with metadata indicating chunk type (text/image/table) and source location

**Query pipeline**:

1. **Retrieve**: Hybrid search (dense + BM25) returning mixed-type chunks
2. **Re-rank**: Cross-encoder re-ranking, considering chunk type relevance
3. **Prompt assembly**: For image chunks, include both the description AND the original image in the prompt (using a multimodal LLM like GPT-4o)
4. **Generate**: Multimodal LLM processes text context + actual images → grounded answer

**Example**:

```
Query: "What trend does the Q3 revenue chart show?"

Retrieved:
  1. [IMAGE] Revenue chart from page 5 (description: "Bar chart showing quarterly
     revenue from Q1-Q4, with Q3 at $4.2M, highest of all quarters")
  2. [TEXT] "Q3 revenue reached $4.2M, a 15% YoY increase..."
  3. [TABLE] Revenue breakdown by segment (markdown table)

Prompt to GPT-4o:
  [System]: Answer based on the provided context including images
  [Image]: [actual chart image]
  [Text context]: [chunks 2 and 3]
  [User]: What trend does the Q3 revenue chart show?

Answer: "The chart shows a clear upward trend in quarterly revenue,
with Q3 at $4.2M being the highest quarter. The growth appears to
accelerate from Q2 to Q3..."
```

---

### Q9: What is ImageBind? How does it achieve cross-modal alignment without explicit pairing?

**A**: ImageBind aligns 6 modalities (image, text, audio, depth, thermal, IMU) in a **single shared embedding space**, using images as the "binding" modality.

**The key insight**: If modality A is aligned with images, and modality B is independently aligned with images, then A and B are **implicitly aligned** through the shared image space — even without explicit A-B training pairs.

$$
f_{\text{text}}(\text{"bird chirping"}) \approx f_{\text{image}}(\text{bird photo}) \approx f_{\text{audio}}(\text{bird sound})
$$

**Training**: For each non-image modality $M$, train a contrastive loss aligning $M$ with images:

$$
\mathcal{L}_M = -\sum_i \log \frac{\exp(\cos(\mathbf{v}_i^M, \mathbf{v}_i^{\text{img}}) / \tau)}{\sum_j \exp(\cos(\mathbf{v}_i^M, \mathbf{v}_j^{\text{img}}) / \tau)}
$$

The image encoder is CLIP's (pretrained, frozen). Only the new modality encoders are trained.

**Emergent cross-modal capabilities**: Without ever training on audio-text pairs, the model can:
- Retrieve images using audio queries (bird sound → bird photos)
- Retrieve text using depth maps
- Cross-modal arithmetic: audio(dog barking) + text("on the beach") → images of dogs on beaches

**Why images as the anchor**: Images are the most information-rich modality and have the largest number of naturally paired datasets (image-text: billions of pairs; image-audio: millions; image-depth: millions). Using images as the hub creates the most well-connected alignment space.

---

### Q10: Where is multimodal AI heading? What are the key open problems?

**A**:

**Current trajectory (2025-2026)**:

1. **Unified any-to-any models**: GPT-4o already processes text + image + audio in, text + audio out. The next step: models that natively generate images, audio, and video as first-class outputs — not through separate diffusion models, but within the same transformer. Google's Gemini and OpenAI are both pursuing this.

2. **Real-time multimodal agents**: Agents that see (screen, camera), hear (microphone), speak (TTS), and act (mouse/keyboard) in real-time. Anthropic's "computer use" and OpenAI's "operator" are early examples.

3. **Video as a first-class modality**: Sora showed that video generation is possible. The next challenges: interactive video (respond to user input), longer coherent videos (minutes, not seconds), and video understanding at the level of text understanding.

**Key open problems**:

1. **Hallucination in multimodal models**: VLMs hallucinate objects not in the image (e.g., "There's a cat on the table" when there's no cat). This is worse than text hallucination because users can see the image and easily detect errors.

2. **Fine-grained spatial reasoning**: Models struggle with "What's to the left of the red car?" or counting objects precisely. Spatial understanding remains weak.

3. **Temporal reasoning in video**: Understanding causality, predicting what happens next, and maintaining narrative coherence over minutes of video.

4. **Efficient multimodal processing**: A 10-second video at 24fps with a ViT encoder produces ~57,600 visual tokens. Processing this alongside text in a transformer is computationally prohibitive. Efficient tokenization and attention mechanisms for high-bandwidth modalities are needed.

5. **Evaluation**: We lack good benchmarks for multimodal understanding. Existing benchmarks (VQAv2, OK-VQA) are saturated. More challenging, real-world multimodal evaluation is needed.

6. **Multimodal alignment and safety**: Ensuring that image/audio generation doesn't produce harmful content. NSFW detection, deepfake prevention, and copyright respect across modalities.

---

*Multimodal AI is converging toward unified models that seamlessly combine text, images, audio, and video. Next: [Topic 23: Inference Optimization](23_Inference_Optimization.md) — making these massive models fast and cheap enough for production.*
