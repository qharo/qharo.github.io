---
title: "llama2.rs"
date: "2024-10-03"
summary: "A Rust implementation of Llama-2"
type: "Project"
tags: ["NLP", "Rust"]
thumbnailimage: "thumbnail.png"
# link: "https://github.com/yourusername/project"  # Optional
weight: 5  # Controls the order (lower numbers appear first)
---

{{< lead >}}
A Rust implementation of [llama2.c](https://github.com/karpathy/llama2.c), complete with a [WASM-backed GitHub Page](https://qharo.github.io/llama2.rs/) running the 15M param model in the browser.
{{< /lead >}}


## llama2.c
[llama2.c](https://github.com/karpathy/llama2.c) is Andrej  Karpathy's pure C implementation of the LLama-2 architecture. It contains code to for a "fullstack" training and inference of a Large Language Model, as well as model files (15M, 42M and 110M) trained on the [TinyStories](https://arxiv.org/pdf/2305.07759) dataset. 

This fairly straightforward implementation (~1000 lines) demonstrates that language models can achieve impressive results within a focused domain and removes nearly all abstractions. I have implemented the inference code, borrowing the model and tokenizer files and have added a WASM-interaction on top.

## Tokenizer
The inference process starts with tokenization, which converts text into numerical tokens. We implement this using a Byte-Pair tokenizer, which we load from the binary file.


This setup creates our vocabulary, which we then use for encoding and decoding using lookup. The encoding process first operates at the token level, then recursively merges tokens based on their score.

## Transformer

The Transformer processes input tokens by applying a series of weight matrices to transform token representations through multiple layers of attention and feedforward networks. The Transformer struct maintains both permanent weights and temporary state variables for processing.

#### 1. Token Embedding and Initial Processing

<img src="/images/projects/llama2.rs/TokenEmbedding.png" class="grid-w100" >

The vocabulary token ID is mapped onto a vector representation using the {{< katex >}} \\(token\\_embedding\\_table\\) matrix

#### 2. Position Encoding


<img src="/images/projects/llama2.rs/PositionalEncoding.png" class="grid-w100" >

We use RoPE encodings to embed the positional information of each particular token into that token's vector representation. We precalculate the cos and sin values for all positions.

#### 3. Multi-Head Attention

<img src="/images/projects/llama2.rs/Attention.png" class="grid-w100" >


After the position encoding, each layer processes the token representation through a multi-head attention mechanism:

First, the input is normalized using RMSNorm:
{{< katex >}} \\(\hat{x} = \text{RMSNorm}(x) \\cdot \text{rms\\_att\\_weight} \\)
The normalized input is projected into queries, keys, and values:
{{< katex >}} $$ \begin{aligned} Q &= \hat{x}W_q \end{aligned} $$
{{< katex >}} $$ \begin{aligned} K &= \hat{x}W_k \end{aligned} $$
{{< katex >}} $$ \begin{aligned} V &= \hat{x}W_v \end{aligned} $$

For each head _h_, attention scores are computed:
{{< katex >}} $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
The key-value attention mechanism uses grouped-query attention where {{< katex >}} \\( n\\_kv\\_heads \leq n\\_heads \\). The results are projected back to the model dimension:
{{< katex >}} \\(\text{output} = \text{Concat}(head_1, ..., head_h)W_o \\). Finally, a residual connection is added to _x_.

#### 4. Feed-Forward Network

<img src="/images/projects/llama2.rs/FFN2.png" class="grid-w100" >

We normalize the output of multi-head  attention {{< katex >}} \\( \hat{x} = \text{RMSNorm}(x) \cdot \text{rms\\_ffn\\_weight} \\). We then have two linear layers with {{< katex >}} \\( W_1 \\) and {{< katex >}} \\( W_3 \\). We then implement a SwiGLU, with {{< katex >}} \\( W_1 \\) going through a SiLU unit before a final {{< katex >}} \\( W_2 \\) layer.

#### 5. Output Processing
<img src="/images/projects/llama2.rs/Output.png" class="grid-w100" >

For our output, we simply perform one final RMS normalization with {{< katex >}} \\( \text{rms\\_final\\_weight} \\) and multiply with {{< katex >}} \\( W\_{cls} \\).

## Sampler
Once the logits for the next token is received, the sampler provides a method of making that decision.

The sampling process has three main strategies:

1. Greedy Sampling (temperature = 0)
This is a deterministic search method that selects only the most likely token by applying argmax. 


When temperature is not zero, the logits are scaled by the temperature value. A temperature above 1.0 makes the distribution more uniform, increasing randomness, while a temperature below 1.0 concentrates probability on higher likelihood tokens. Following temperature scaling, a softmax operation is applied to the logits. 

2. Multinomial Sampling (top-_p_)
Tokens are sampled according to their complete probability distribution.

3. Top-_p_ Sampling (0 < top-p < 1)
We retain only the highest-probability tokens whose cumulative probability sum reaches the specified 'p' threshold (for example, 0.9 for top-90%). This technique effectively eliminates the long tail of low-probability tokens, and sampling is then performed from this reduced set of tokens.


## Web Implementation
I had to implement a few changes to the original code to accomodate WASM-creation, namely the ```from_bytes()``` method and Transformer and Config wrappers. You can find the implementation [here](https://qharo.github.io/llama2.rs/).


{{< github repo="qharo/llama2.rs" >}}
