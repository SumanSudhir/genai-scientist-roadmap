# 2-Month AI Scientist Interview Prep: MNC India Focus

Tailored for **Data Scientists/AI Engineers (4+ yrs)** targeting **Top MNCs (Google, MSFT, Amazon, Adobe, NVIDIA)** and **Tier-1 AI Startups** in India.

## Key Focus Areas for MNC India Interviews
1. **DSA (Data Structures & Algorithms)**: Non-negotiable. 45-60 min rounds are standard.
2. **GenAI System Design**: Moving from "wrapper" apps to scalable, cost-optimized pipelines.
3. **Low-Level Mechanics**: Quantization, KV caching, and Flash Attention.
4. **Behavioral**: Company-specific principles (Amazon LP, Googliness).

---

## Weekly Hour Breakdown (Total: ~256 Hours)
- **Weekdays (Mon-Fri)**: 20 Hours (1-2 hrs DSA, 2-3 hrs Core AI)
- **Weekends (Sat-Sun)**: 12 Hours (System Design & Research Papers)

---

## The Schedule

### Week 1: DSA (Foundations) & ML Scaling (32 hrs)
- **DSA**: Arrays, Strings, Hashing, Two Pointers (LeetCode Top 150).
- **Core AI**: Review Prob/Stats from first principles (MLE/MAP). [Lilian Weng: KL Divergence](https://lilianweng.github.io/posts/2018-10-13-flow-models/#kullbackleibler-divergence).
- **Weekend**: Draw the Transformer architecture from memory. [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).

### Week 2: DSA (Advanced) & The Transformer Deep Dive (32 hrs)
- **DSA**: Linked Lists, Trees, Graphs (BFS/DFS).
- **Core AI**: Self-Attention vs Cross-Attention math. Pre-LN vs Post-LN stability. [Paper: Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- **Weekend**: Positional Encodings (RoPE is critical for Llama/Mistral). [RoPE Explained](https://blog.eleuther.ai/rotary-embeddings/).

### Week 3: DSA (Optimized) & LLM Architecture Evolution (32 hrs)
- **DSA**: Dynamic Programming (Classic 1D/2D) & Heaps.
- **Core AI**: BERT to Llama 3 evolution. MoE (Mixture of Experts) -- Mistral/Grok. [Interconnects: MoE](https://www.interconnects.ai/).
- **Weekend**: Scaling Laws (Chinchilla). How to size a model for a task.

### Week 4: Training & Distributed Systems (32 hrs)
- **DSA**: Stack, Queue, Monotonic Stack.
- **Core AI**: Data/Tensor/Pipeline Parallelism. ZeRO stages. [Hugging Face: Parallelism Guide](https://huggingface.co/docs/transformers/v4.18.0/en/parallelism).
- **Weekend**: Precision (BF16 is the standard). Why loss scaling is needed in FP16.

### Week 5: Fine-Tuning (PEFT) & MNC System Design (32 hrs)
- **DSA**: Tries, Segment Trees (if targeting research engineering).
- **System Design**: Designing a scalable RAG pipeline for 1M docs. [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval).
- **Core AI**: LoRA, QLoRA, DoRA. Derive the rank update. [Paper: LoRA](https://arxiv.org/abs/2106.09685).

### Week 6: Alignment (RLHF/DPO) & Production (32 hrs)
- **Core AI**: RLHF vs DPO. Why DPO simplifies training. [Paper: DPO](https://arxiv.org/abs/2305.18290).
- **System Design**: Content moderation system using LLMs (Standard MNC round).
- **Weekend**: Evaluation Benchmarks (MMLU, HumanEval). How to detect benchmark leakage.

### Week 7: Inference Optimization & Serving (32 hrs)
- **Core AI**: Paged Attention (vLLM), KV Caching, Flash Attention. [vLLM: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html).
- **System Design**: Cost-optimization: Model Routing (GPT-4 vs Llama 3) & Caching.
- **Weekend**: Quantization (GPTQ, AWQ, GGUF). NF4 intuition.

### Week 8: Behavioral & Mock Drills (32 hrs)
- **Behavioral**: Amazon Leadership Principles or Google Behavioral prep.
- **Mocks**: Case studies from [AI DS Interview Questions](file:///Users/ragaai_user/Desktop/NLP/AI_INTERVIEW_QUESTIONS.md).
- **Weekend**: Review top 5 AI research papers from 2024 (Search Arxiv Sanity).

---

## Must-Watch/Read Sites
1. **[NeetCode](https://neetcode.io/)** - Best for India MNC DSA prep.
2. **[Lilian Weng](https://lilianweng.github.io/posts/)** - Depth for research rounds.
3. **[Karpathy: Zero to Hero](https://www.youtube.com/@AndrejKarpathy)** - Mandatory for first-principles coding.
4. **[Chip Huyen's System Design](https://huyenchip.com/2022/01/02/real-time-machine-learning.html)** - Crucial for system rounds.
