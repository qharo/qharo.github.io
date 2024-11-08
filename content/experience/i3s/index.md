---
title: "L1,∞ Convolutional-VAE Projection for DNA Encoding"
date: "2023-10-01"
summary: "Centre National de la Recherche Scientifique (CNRS)"
type: "Experience"
tags: ["CV", "PyTorch", "Internship"]
thumbnailimage: "thumbnail.png"
# link: "https://github.com/yourusername/project"  # Optional
weight: 20  # Controls the order (lower numbers appear first)
---
{{< lead >}}
We focused on two main objectives:
- Implementing L1,∞ projection in Convolutional Variational Autoencoders (CVAEs) to induce structural sparsity while preserving reconstruction quality
- Designing and implementing quaternary encoding schemes optimized for DNA sequence representation

We applied these innovations to the [Cool Chic](https://github.com/Orange-OpenSource/Cool-Chic) model, achieving significant network sparsification (~80% for CAE, ~50% for MDC) while maintaining competitive rate-distortion performance.
{{< /lead >}}

## L1,∞ Projection
[L1,∞ projection](https://webcms.i3s.unice.fr/Michel_Barlaud/sites/mbarlaud/files/2023-11/L1Infty_final.pdf) is a method for enforcing a structured budget constraint on neural network weights. 
\
\
Given a "budget" hyperparameter _C_, it first measures feature importance by finding the maximum absolute value within each column of the weight matrix. These column maximums are summed to quantify the total feature influence. When this sum exceeds our specified budget _C_, the projection algorithm activates, optimally redistributing the budget across features based on their relative importance:
- Important features: Scaled down proportionally, preserving signs
- Less important features: Completely zeroed out


This creates structured sparsity by removing entire features rather than individual weights, which differs from L1 regularization (random, scattered sparsity) and L1,1 projection (row-wise sample sparsity).



## Quaternary Encoding

DNA sequences consist of four nucleotide bases (Adenine, Thymine, Cytosine, Guanine), which naturally map to a base-4 numerical system (0, 1, 2, 3). While traditional neural networks typically operate in binary latent spaces, this biological constraint required us to develop a quaternary-formatted latent space. To achieve this, we implemented a modified Shannon-Fano encoding algorithm specifically optimized for base-4 representation. This adaptation necessitated changes to the entropy calculations, as they needed to account for four possible states rather than two.


## Results
### Decoder Projection

Our initial experiment focused on inducing sparsity in the decoder network of a fully connected autoencoder using L1,∞ projection. The study utilized metabolomic [data from urine samples of Non-Small Cell Lung Cancer (NSCLC) patients](https://pure.psu.edu/en/publications/noninvasive-urinary-metabolomic-profiling-identifies-diagnostic-a), comprising of 469 NSCLC patients (pre-treatment), 536 control patients. Each sample contained 2,944 metabolomic features. While implementing L1,∞ projection on the encoder network was relatively straightforward, the decoder presented unique challenges. The primary difficulty lay in determining the optimal direction of sparsity to achieve meaningful structural patterns without compromising the network's performance.

__Without Projection__
<img src="/images/experience/i3s/DecoderWithoutProjection.png" class="grid-w100" />

**With Projection**
<img src="/images/experience/i3s/DecoderWithProjection.png" class="grid-w100" />

### Convolutional VAE
Building on the success of the fully connected model, we extended our approach to image compression using a CVAE. This implementation aimed to achieve efficient image encoding and decoding while maintaining image quality.

**PSNR Comparison**
<img src="/images/experience/i3s/PSNRvsBitRate.png" class="grid-w100" />

We implemented a quaternary encoding scheme to compress the latent space representation. Performance was measured in nats (natural units of information).

**Comparing L1 Projection, Quaternary Projection**
<img src="/images/experience/i3s/PSNRvsNatsPixel.png" class="grid-w100" />


## Cool Chic
The final experiment focused on Multiple Description Coding (MDC) and Single Description Coding (SDC) using a sender-receiver architecture with the [CoolChic model](https://github.com/Orange-OpenSource/Cool-Chic).

**PSNR vs Sparsity**
<img src="/images/experience/i3s/PSNRvsSparsity.png" class="grid-w100" />

**Model Weight Comparison**
{{< gallery >}}
  <img src="/images/experience/i3s/WeightsWithProjection.png" class="grid-w50" />
  <img src="/images/experience/i3s/WeightsWithoutProjection.png" class="grid-w50" />
{{< /gallery >}}

When examining the network post-projection, we observed highly organized sparsity patterns emerging from what was previously a dense, interconnected structure. Despite this substantial reorganization, the network maintained its core functionality, successfully performing its intended tasks without significant performance degradation.



<!-- 
DNA has 4 bases (A, c,g and t) which numerically can be represented by 0, 1, 2 and 3. This necessitated our latent space to be quaternary formatted as opposed to the usual binary format used in computer storage. We implemented a modified version of SAhannon-Fano Encoding. This approach needed a careful consideration of entropy calculation and code assignment strategies for quaternary space.

Quaternary encoding represents a fundamental shift from traditional binary encoding by utilizing a base-4 numeric system. In our context, this system was specifically designed to represent DNA bases (A, C, G, T) as numerical values 0, 1, 2, and 3. This encoding scheme provides a more natural representation for DNA sequences and can potentially lead to more efficient compression when properly implemented in neural network architectures.
To effectively handle this quaternary data, we implemented a modif

ied version of Shannon-Fano coding. The process begins with a frequency analysis of the symbols in the input data, followed by sorting these symbols based on their frequencies. The algorithm then recursively divides the sorted list into sublists until each sublist contains only one symbol. While traditional Shannon-Fano coding typically assigns binary codes, we adapted the method to work with base-4 codes, creating a more appropriate encoding scheme for our DNA sequence representation. This modified approach required careful consideration of the entropy calculations and code assignment strategies to maintain efficiency while working in the quaternary space. -->

<!-- Given a budget _C_, it works in two steps:



First, it evaluates feature importance:

- Calculate the L∞ norm for each column (feature) in the weight matrix
- If this value exceeds _C_, performs projection by optimally allocating the budget:


Sum these maximums to get the total influence

If this sum exceeds hyperparameter _C_, projection is needed



Then, it optimally allocates the budget:



L1∞ projection combines the characteristics of L1 and L∞ norms to achieve structural sparsity in neural networks. The L∞ norm measures the maximum absolute value of any element in a vector, while L1 norm measures the sum of absolute values. When combined in L1∞ projection, we project weights onto an L1 ball while considering L∞ norm constraints. This creates a unique form of regularization that encourages both sparsity and structured weight patterns.
The projection operates by constraining weights according to the equation {y | ||y - x||₁ ≤ c}, where c is a constant that determines the radius of the projection ball. Unlike traditional L1 regularization or L1,1 projection, which tend to produce element-wise sparsity, L1∞ projection creates structural patterns in the weight matrices. This approach was particularly effective when applied to the decoder network, where it produced meaningful structural sparsity while preserving the network's performance characteristics. -->