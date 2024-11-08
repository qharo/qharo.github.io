---
title: "RF-Based Automatic Prompt Compression"
date: "2024-09-01"
summary: "Amadeus Research Team (ART)"
tags: ["NLP", "PyTorch", "Internship"]
type: "Experience"
weight: 10  # Controls the order (lower numbers appear first)
---
{{< lead >}}
While current state-of-the-art prompt compression methods rely on large neural networks or manual prompt engineering, this research proposes a lightweight alternative that leverages attention mechanisms inherent in transformer models. By extracting statistical features from attention vectors and training a Random Forest classifier, the approach achieves comparable performance to existing methods for a fraction of the inference cost and nearly no training cost. This makes the method particularly suitable for real-time context history compression in API-based LLM applications.
{{< /lead >}}

## Transformers
[Transformer architecture](https://arxiv.org/pdf/1706.03762) revolutionized natural language processing through its self-attention mechanism, which allows tokens to dynamically influence each other's representations. This attention mechanism computes relevance between queries and keys, applying the resulting weights to values, enabling the model to capture complex relationships and dependencies in text.

#### Increasing Computational Demand
This success has led to increasingly resource-intensive models, with sizes growing from BERT's 110M parameters to GPT-3's 175B and beyond. Context lengths have similarly expanded, from 512 tokens to over 1M tokens in models like Claude. This growth has significantly increased computational requirements, as attention mechanisms scale quadratically with context length.

#### API Alternative
While this trend has pushed most organizations toward API-based solutions as a more convenient and cost-effective alternative to in-house development, it introduces new challenges around token usage, blackbox functioning and computational overhead.




## Prompt Compression
Prompt compression techniques fall into two main categories:
- **Abstractive** compression generates new, condensed versions of prompts, potentially using tokens not present in the original input. While this approach offers greater flexibility and potential for creative compression, it tends to be computationally intensive and risks semantic drift from the original content.
- **Extractive** compression, conversely, selects and retains tokens from the original input. Though more constrained in its approach, it offers faster computation and better preservation of original meaning, making it particularly suitable for maintaining semantic fidelity in compressed prompts.

#### LLMLingua 1
[LLMLingua 1](https://aclanthology.org/2023.emnlp-main.825.pdf) introduced a two-step prompt compression framework based on perplexity optimization. The system first dynamically allocates compression ratios across prompt components (sentences, examples, etc.) based on user-defined targets. It then performs finer-grained optimization at the token level, leveraging the key insight that higher perplexity indicates more informative content. The framework utilized an Alpaca 7B model, requiring distribution alignment with target models, but demonstrated impressive results with up to 20x compression while maintaining prompt response quality.

#### LLMLingua 2
[LLMLingua 2](https://aclanthology.org/2024.findings-acl.57.pdf) addresses key limitations of its predecessor, particularly the reliance on a large aligned model and unidirectional attention constraints. It employs GPT-4 to create a high-quality token-labeled dataset, training a transformer encoder ([xlm-roberta](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) with approximately 355M parameters) for token classification. This approach achieved 3x-6x improvement in processing time while maintaining performance levels. The system has proven cost-effective, reducing both inference time and overall costs even when accounting for compression overhead, and has been successfully integrated into popular frameworks like LangChain and Prompt Flow.

## Framework

Transformer models derive their capabilities primarily from the attention mechanism, represented by the equation:
{{< katex >}}
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

We hypothesized that token importance could be quantified through this attention structure. Our analysis focused specifically on attention vectors before the dot product operation, along the non-softmax axis, as this represents how much attention each token receives.

The initial challenge lay in efficiently processing the high-dimensional attention map space {{< katex >}} \\(R^{L \times H \times N \times N }\\) where {{< katex >}} \\(L\\) is the number of layers, {{< katex >}} \\(H\\) is the number of heads and {{< katex >}} \\(N\\) is the number of tokens, which varies depending on the prompt.

We reframed the problem in 3 steps to effectively tackle it:
- **Layerwise Processing**: Considering one layer at a time, which reduced the target function to
{{< katex >}}
$$ f: \mathbb{R}^{H \times N \times N} \rightarrow [0, 1]^{N} $$

- **Individual Vector Analysis**: As the attention mechanism continuously enhances the attention vectors with contextual information, we hypothesized that each attention vector could be used individually
{{< katex >}}
$$ f: \mathbb{R}^{H \times N} \rightarrow [0, 1], \quad \forall i \in \{1, \ldots, N\} $$

- **Feature Extraction**: For each attention vector, we extract {{< katex >}} \\(f\\) statistical features that capture the key characteristics. This makes the input invariant to number fo tokens and finally leaves us with a problem framed as:
{{< katex >}}
$$ f: \mathbb{R}^{f \times N} \rightarrow [0, 1], \quad \forall i \in \{1, \ldots, N\} $$

This final formulation served as the basis for our classification model.

**Feature Engineering**
<img src="/images/experience/amadeus/FeatureEngineering.png" class="grid-w100" />

**Proposed Framework**
<img src="/images/experience/amadeus/FinalPipeline.png" class="grid-w100" />

### Experiments
Our framework required optimization of four key components:
1. Classification Model
    - After evaluating multiple approaches including Support Vector Machines (with RBF Kernel), Random Forest (100 trees), and a Neural Network (similar to LLMLingua 2's token classification layer), Random Forest performed best.

2. Feature Selection
    - We analyzed nine potential features: Mean, Standard Deviation, Median, Median Absolute Deviation (MAD), Self-Attention, Kurtosis, Skewness, Quartile Range Values and Entropy
    - Through Recursive Feature Elimination, we identified four critical features: MAD, Entropy, Standard Deviation and Self-Attention 

2. Attention Model
    - Our analysis revealed that fine-tuning played a less crucial role than initially anticipated, as [vanilla XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) showed performance comparable to [LLMLingua 2's fine-tuned model](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) through the first five layers. This revealed that raw attention maps inherently contain sufficient information for effective compression, regardless of model fine-tuning.
    - We ultimately selected the [GTE model](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) for its practical advantages: support for context lengths up to 8192 tokens and superior performance in our metrics.

4. Dataset
    - We utilized the [Microsoft MeetingBank-LLMCompressed dataset](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed), which provided high-quality token-level compression labels.

**Final Framework**

<img src="/images/experience/amadeus/Implementation.png" class="grid-w100" />

Our implemented solution processes input text through the GTE transformer's fifth layer, extracts our four identified features from the attention maps, and feeds these into a Random Forest classifier for token retention prediction. This streamlined pipeline offers an efficient balance between computational overhead and compression effectiveness.

## Results

### Performance

We evaluated our model using LongBench, a comprehensive benchmark for testing long-context understanding capabilities. The benchmark consists of:

- **Single-document QA**: Tasks requiring understanding of a single long document
- **Multi-document QA**: Tasks involving reasoning across multiple documents
- **Summarization**: Tasks focusing on condensing long documents 

Our Random Forest approach showed strong performance in single-document QA tasks, often outperforming LLMLingua 2, while remaining competitive in other categories. The benchmark tasks span from 2,000 to 18,000 tokens, providing a robust test of model capabilities across varying context lengths.

| Task | LLMLingua Performance | RF (ours) Performance | Task Type | Eval Metric | Average Length |
|------|----------------------|---------------------|------------|-----------------|----------------|
| Multifield QA | 33.72 | **34.48** | Single-doc QA | F1 | 4,559 |
| Qasper | 31.17 | **33.86** | Single-doc QA | F1 | 3,619 |
| Narrative QA | **16.34** | 15.34 | Single-doc QA | F1 | 18,409 |
| 2wikimqa | **35.83** | 34.71 | Multi-doc QA | F1 | 4,887 |
| Hotpotqa | **47.73** | 44.26 | Multi-doc QA | F1 | 9,151 |
| MuSiQue | **23.41** | 20.66 | Multi-doc QA | F1 | 11,214 |
| Multi News | 23.73 | **24.37** | Summarization | Rouge-L | 2,113 |
| Gov Report | **25.12** | 22.48 | Summarization | Rouge-L | 8,734 |
| QM Sum | **21.83** | 20.26 | Summarization | Rouge-L | 10,614 |


### Latency
We compared real-time processing speeds between LLMLingua 2 and the Random Forest method, tested on an Intel Xeon Platinum 8168 CPU. This significant speed difference demonstrates the Random Forest's superior efficiency for CPU-based deployments, making it particularly suitable for real-time applications where GPU access might be limited or cost-prohibitive.

| Number of Tokens | LLMLingua 2 (s) | RF (s) |
|-----------------|------------------|---------|
| 500 tokens | 7.74 | 1.45 |
| 2000 tokens | 8.89 | 3.58 |

Our approach offers a resource-efficient solution to prompt compression by using a simple Random Forest classifier instead of fine-tuning large language models. Compared to LLMLingua 2's 355M parameters and additional model layers, our method requires only 74M parameters by utilizing 5 transformer layers. This dramatically reduces both development and inference costs while extending context handling to 8192 tokens, effectively addressing the "lost in the middle" problem that affects many LLMs.