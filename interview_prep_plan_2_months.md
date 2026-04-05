# Senior AI Scientist Prep: 8-Week Sprint (4yr+ Exp)

This is a high-intensity plan for candidates targeting **Senior/L5+ roles** at Google, Microsoft, Amazon, Adobe, and Tier-1 AI startups. At 4 years of experience, you are expected to dominate **Machine Learning System Design**, **Distributed Training**, and **Inference Optimization**.

All topic numbers match the GenAI Scientist study files in `/GenAI_Scientist/` and the question bank in `AI_INTERVIEW_QUESTIONS.md`.

## The Strategy (32 hrs / week)
- **Core AI Deep-Dive (M-F, 4 hrs/day)**: Focus on *derivations*, *scaling*, and *mechanistic interpretability*.
- **System Design & Research (Sat-Sun, 6 hrs/day)**: Build architectural diagrams for MNC-scale problems and read current SOTA papers.

---

## Background Reading (Before Week 1 or Parallel)

These topics are foundational but rarely the focus of deep interview questions. Read the study files and skim the [E] and [M] interview questions. Skip [H] unless you feel weak.

| Topic | Study File | Priority |
|-------|-----------|----------|
| 03: Classical ML | `03_Classical_ML.md` | Medium — bias-variance, gradient boosting |
| 04: Deep Learning Foundations | `04_Deep_Learning_Foundations.md` | High — backprop, LayerNorm, residuals |
| 05: Tokenization | `05_Tokenization.md` | Medium — BPE, vocab size tradeoffs |
| 06: Word Embeddings | `06_Text_Preprocessing_Embeddings.md` | Low — know Word2Vec/GloVe basics |
| 07: RNNs & Sequence Models | `07_Sequence_Modeling.md` | Medium — LSTM gates, why transformers replaced RNNs |

---

## The Schedule

### Week 1: Math Foundations
*Goal: Master the foundations at an interviewer level. No surface-level answers.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon–Wed | Probability/Stats: MLE vs MAP, KL divergence, ELBO derivation | `01_Probability_Statistics.md` |
| Thu–Fri | Optimization: AdamW vs Adam, warmup, gradient clipping, SVD/LoRA connection | `02_Linear_Algebra_Optimization.md` |

- **Practice**: Solve `Topic 01 & 02 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key derivations**: Cross-entropy from MLE; bias correction in Adam; normal equation for ridge regression.
- **Sunday Case Study**: Design an A/B testing framework for LLM rankers at Google Search scale.

---

### Week 2: Transformer Mechanics & Positional Encoding
*Goal: Internalize the Transformer architecture and all its variants cold.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon–Tue | Scaled dot-product attention, multi-head attention, GQA/MQA tradeoffs | `08_Attention_Mechanisms.md` |
| Wed–Thu | Pre-LN vs Post-LN, SwiGLU FFN, parameter counting, KV cache | `09_Transformer_Architecture.md` |
| Fri | Sinusoidal PE → RoPE derivation → ALiBi → YaRN context extension | `10_Positional_Encodings.md` |

- **Practice**: Solve `Topic 08, 09 & 10 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key derivations**: RoPE rotation matrix; Flash Attention memory reduction; BERT-base parameter count.
- **Sunday Case Study**: Explain RoPE and how you'd extend a 4K-context model to 128K to an NVIDIA interviewer.

---

### Week 3: Language Models — BERT, GPT, T5
*Goal: Understand why we moved from BERT → GPT → Llama 3 and what was lost/gained.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon–Tue | BERT: MLM, NSP removal (RoBERTa), [CLS] head, DistilBERT | `11_BERT_Family.md` |
| Wed–Thu | GPT family: CLM, scaling laws (Kaplan → Chinchilla), emergent abilities, MoE (Mixtral) | `12_GPT_Open_Source_LLMs.md` |
| Fri | Encoder-Decoder: T5 span corruption, BART, cross-attention data flow | `13_Encoder_Decoder_Models.md` |

- **Practice**: Solve `Topic 11, 12 & 13 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key concepts**: MLM vs CLM training efficiency; Chinchilla-optimal N/D ratio; why decoder-only won.
- **Sunday Case Study**: Design a Mixture of Experts (MoE) router for a multi-tenant LLM API.

---

### Week 4: Pretraining at Scale — Distributed Training & Scaling Laws
*Goal: Be ready for the "How do 1,000 GPUs talk to each other?" questions.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon–Tue | ZeRO stages 1/2/3, FSDP, memory breakdown (params/grads/optimizer states) | `14_Pretraining.md` |
| Wed–Thu | Data/Tensor/Pipeline parallelism; bubble problem; MFU; BF16 vs FP16 | `14_Pretraining.md` |
| Fri | Scaling laws derivation; Chinchilla; data quality filtering (FineWeb) | `14_Pretraining.md` |

- **Practice**: Solve `Topic 14 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key numbers**: ZeRO-3 reduces memory 64× for 64 GPUs; 70B model needs ~140GB FP16; Chinchilla: N≈D/20.
- **Sunday Case Study**: Design the training pipeline for a 70B model on a cluster of 512 H100s. Address TP × PP × DP split, communication overhead, and checkpointing strategy.

---

### Week 5: Fine-Tuning, PEFT & Alignment
*Goal: Master LoRA, QLoRA, and the full RLHF → DPO pipeline.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon–Tue | LoRA derivation, rank selection, QLoRA (NF4, double quantization, paged AdamW) | `15_Fine_Tuning_PEFT.md` |
| Wed | DoRA, AdaLoRA, adapter layers, prefix tuning — PEFT comparison table | `15_Fine_Tuning_PEFT.md` |
| Thu–Fri | RLHF 3-stage pipeline → PPO clipping → KL penalty → DPO derivation → CAI | `16_RLHF_Alignment.md` |

- **Practice**: Solve `Topic 15 & 16 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key derivations**: LoRA W = W₀ + BA parameter count; DPO loss from RLHF objective; Bradley-Terry model.
- **Sunday Case Study**: Design a preference data collection and alignment pipeline for an Amazon customer support bot. Address reward hacking, sycophancy, and over-refusal.

---

### Week 6: Generation, Prompting & APIs
*Goal: Cover the full generation stack — from sampling to production API design.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon | Decoding: greedy, beam (length normalization), temperature, top-k, top-p, min-p, speculative decoding | `17_Decoding_Strategies.md` |
| Tue–Wed | Prompt engineering: CoT, self-consistency, ToT, ReAct, DSPy, structured output, injection defenses | `18_Prompt_Engineering.md` |
| Wed–Thu | ICL theory: Bayesian task inference, implicit gradient descent, induction heads | `18_Prompt_Engineering.md` |
| Fri | LLM APIs: function calling round-trip, parallel tools, cost estimation, semantic caching, model routing, LLM gateway | `19_LLM_APIs_Function_Calling.md` |

- **Practice**: Solve `Topic 17, 18 & 19 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key concepts**: Speculative decoding acceptance probability; prompt injection vs jailbreaking; ICL as Bayesian inference.
- **Sunday Case Study**: Design a cost-optimized LLM API gateway for a product processing 1M requests/day. Include model routing, caching layers, and fallback logic.

---

### Week 7: Applications — RAG, Agents, Multimodal & Inference
*Goal: Build systems that don't just work, but scale to MNC requirements.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon | RAG: chunking strategies, hybrid BM25+dense, RRF, cross-encoder reranking, RAGAS | `20_RAG.md` |
| Tue | Agents: ReAct loop, tool schemas, multi-agent systems, Reflexion, failure modes | `21_Agents_Tool_Use.md` |
| Wed | Multimodal: CLIP InfoNCE loss, ViT patch tokenization, LLaVA architecture, diffusion | `22_Multimodal_AI.md` |
| Thu | Inference optimization: KV cache memory, PagedAttention (vLLM), Flash Attention, INT8/INT4 quantization | `23_Inference_Optimization.md` |
| Fri | ML System Design framework: requirements, data, model, serving, monitoring | `24_ML_System_Design.md` |

- **Practice**: Solve `Topic 20, 21, 22, 23 & 24 [H]` questions in [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Key calculations**: KV cache memory for 7B model (MHA vs GQA); ViT 224×224 → 196 patches; RRF formula.
- **Sunday Case Study**: Design a production RAG system for Microsoft Office 365 (100B+ documents, sub-second latency). Cover ingestion, retrieval, generation, guardrails, evaluation, and monitoring.

---

### Week 8: Evaluation, Safety, Frontiers & Mock Interviews
*Goal: Polish, simulate the pressure, and cover the research frontier questions.*

| Day | Focus | Study File |
|-----|-------|-----------|
| Mon | Evaluation: BLEU/ROUGE/BERTScore, MMLU/HumanEval/HellaSwag, LLM-as-judge, contamination, RAGAS | `25_Evaluation_Benchmarks.md` |
| Tue | Safety & Ethics: hallucination detection, bias/fairness, prompt injection, DP, EU AI Act, red-teaming | `26_Safety_Ethics.md` |
| Wed | Research Frontiers: Mamba/SSMs, MoE (DeepSeek), test-time compute (o1/o3), long-context, synthetic data | `27_Research_Frontiers.md` |
| Thu | System Design mock: Search Ranking, Fraud Detection, or Content Moderation at scale | `24_ML_System_Design.md` |
| Fri | Pick any 10 `[H]` questions from [AI_INTERVIEW_QUESTIONS.md](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md) and answer them out loud / record yourself. |
| Sat–Sun | Full mock interviews. Behavioral: Amazon Leadership Principles / Googliness. |

- **Key topics for final polish**: Benchmark contamination; deceptive alignment; Mamba vs Transformer tradeoffs; test-time compute scaling law.
- **Behavioral prep**: Prepare 3 STAR stories for each of: impact at scale, handling ambiguity, technical disagreement, driving cross-functional alignment.

---

## Topic Coverage Map

| Week | Topics | Study Files |
|------|--------|-------------|
| Background | 03, 04, 05, 06, 07 | Classical ML, DL Foundations, Tokenization, Embeddings, RNNs |
| 1 | **01, 02** | Probability/Stats, Linear Algebra/Optimization |
| 2 | **08, 09, 10** | Attention, Transformer Architecture, Positional Encodings |
| 3 | **11, 12, 13** | BERT, GPT/LLMs, Encoder-Decoder |
| 4 | **14** | Pretraining (Scaling Laws, Distributed Training, ZeRO) |
| 5 | **15, 16** | Fine-Tuning/PEFT, Alignment (RLHF/DPO) |
| 6 | **17, 18, 19** | Decoding Strategies, Prompt Engineering + ICL, LLM APIs |
| 7 | **20, 21, 22, 23, 24** | RAG, Agents, Multimodal, Inference Optimization, ML System Design |
| 8 | **25, 26, 27** | Evaluation, Safety/Ethics, Research Frontiers |

---

## Key Derivations to Know Cold

These come up in MNC technical screens — practice until you can write them from memory.

| Derivation | Topic |
|-----------|-------|
| Cross-entropy loss = MLE under categorical distribution | 01 |
| KL divergence decomposition: $D_{KL}(P\|Q) = H(P,Q) - H(P)$ | 01 |
| AdamW bias correction terms | 02 |
| SVD → PCA → why LoRA works | 02 |
| Scaled dot-product attention formula + O(n²) complexity | 08 |
| RoPE: rotation matrix encodes relative position in dot product | 10 |
| BERT-base parameter count (≈110M) by component | 11 |
| Chinchilla: optimal N ≈ 20× tokens, derive from FLOPs = 6ND | 12, 14 |
| ZeRO-3 memory formula across $K$ GPUs | 14 |
| LoRA: $W = W_0 + BA$, parameter count = $2rd$ | 15 |
| DPO loss derivation from RLHF KL-constrained objective | 16 |
| Speculative decoding acceptance probability $\alpha = \min(1, p_{\text{target}}/p_{\text{draft}})$ | 17 |
| BLEU-N precision + brevity penalty calculation | 25 |

---

## Targeted Study Resources

- **Math/Core**: [Lilian Weng's Blog](https://lilianweng.github.io/posts/) — Essential for RLHF, Self-Attention, and Diffusion.
- **System Design**: [Chip Huyen's ML System Design](https://huyenchip.com/2022/01/02/real-time-machine-learning.html).
- **Implementation**: [Andrej Karpathy's makemore/GPT from scratch](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAJobLmo).
- **Recent Trends**: [Interconnects by Nathan Lambert](https://www.interconnects.ai/) — Alignment and frontier model analysis.
- **Interpretability**: [Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/) — For mechanistic interpretability questions.
