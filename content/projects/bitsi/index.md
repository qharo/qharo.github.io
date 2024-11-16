---
title: "Bitsi: An Optimized BitNet Implementation"
date: "2024-11-15"
summary: "BitNet b1.58 packing weights into 2 bits, enabling 16x model compression."
type: "Project"
tags: ["Quantization", "NLP"]
thumbnailimage: "thumbnail.png"
# link: "https://github.com/yourusername/project"  # Optional
weight: 20  # Controls the order (lower numbers appear first)
---

{{< lead >}}
Bitsi is an optimized inference implementation of [BitNet b1.58](https://arxiv.org/pdf/2402.17764) that uses quantization-aware training to constrain weights to three possible values: -1, 0, or 1. The implementation features a custom BitLinear layer that achieves a 16x reduction in model size while minimizing computational overhead.
{{< /lead >}}

## BitNet b1.58
BitNet b1.58 makes Large Language Models more efficient by training them from scratch to use only three values for weights: -1, 0, and 1, rather than the standard 32-bit floating point numbers. Unlike traditional post-training quantization (where a model is first trained with full precision and then compressed afterward), BitNet b1.58 trains a LLama model using quantization-aware training, meaning it learns to work with these limited values during the entire training process. The quantization process works by first scaling the weight matrix by its average absolute value, then rounding each value to the nearest of the three allowed values (-1, 0, 1). This is combined with 8-bit activations and a proposed architecture that replaces traditional matrix multiplications with simpler integer additions. 
<figure>
    <img src="/images/projects/bitsi/bitnet_arch.png" class="grid-w100" >
    <figcaption>Proposed modified MatMul replacement</figcaption>
</figure>

Quantization-aware training allows the model to adapt to these constraints during training, learning parameters that work well within these limitations, rather than trying to approximate full-precision weights after the fact. This results in better performance compared to post-training quantization approaches.

## Modified Forward Pass
The paper implemented their forward pass like so:

```python
# reduces weights to 3 values
def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)

def activation_quant(x, num_bits=8): ...

# original forward pass for QAT
def forward(self, input):
    
    quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
    quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

    out = nn.functional.linear(quant_input, quant_weight)
    if not self.bias is None:
        out += self.bias.view(1, -1).expand_as(out)

    return out
```

Which is obviously meant for quantization-aware training. During inference, while still functional, this method can be modified. I did it in the following manner:
1. The weight quantization function was modified to return both the scale factor and the raw quantized weights. Instead of dividing by the scale immediately, I store both the scale as a parameter and the unscaled quantized weights. These unscaled weights represent the true 2-bit values (-1, 0, 1) before scaling.

```python
# now returns weights and scale
def weight_quant(weight, num_bits=1):
    orig_dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1)
    return result.to(orig_dtype), s
```

2. Pre-quantizing the weight before inference allows us to skip this weight_quant function in the forward function. Using scale like below is equivalent to dividing the weight.

```python
# no weight quantization
def forward(self, input):
    
    quant_input = input + (activation_quant(input, self.input_bits) - input).detach()

    out = nn.functional.linear(quant_input, self.weight)
    if not self.bias is None:
        out += self.bias.view(1, -1).expand_as(out)

    return out / self.scale
```

## Weight Storage 
While the weights theoretically require only 1.58 bits per value (to represent -1, 0, 1), Python and PyTorch's byte-indexing results in each value consuming 8 bits, even for boolean values.

To achieve optimal memory usage, I implemented a custom C++ extension that packs the weights into int8 format. This packed representation is maintained during storage, with weights only being unpacked when needed during the forward pass. This approach achieves approximately 16x memory reduction compared to the original fp32 model while maintaining minimal performance overhead during inference.

## Further Improvements
While [BitNet.cpp](https://github.com/microsoft/BitNet) is the official reference implementation, research in this area is [evolving quickly](https://arxiv.org/abs/2411.04965v1). [PyTorch's ao](https://github.com/pytorch/ao) library offers promising tools for native Python implementations. Additionally, I plan to train a low-cost translation model using BitNet once GPU resources become available.

{{< github repo="qharo/bitsi" >}}