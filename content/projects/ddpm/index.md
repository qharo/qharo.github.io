---
title: "Devanagiri Diffusion Model"
date: "2024-10-03"
summary: "A PyTorch implementation of DDPM trained to generate Devanagiri Script."
type: "Project"
tags: ["CV", "PyTorch"]
thumbnailimage: "thumbnail.png"
# link: "https://github.com/yourusername/project"  # Optional
weight: 20  # Controls the order (lower numbers appear first)
---

{{< lead >}}
A PyTorch implementation of a [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239) trained to generate [Devanagari](https://en.wikipedia.org/wiki/Devanagari) characters. It trains a 10 million parameter U-Net on 92,000 32x32 images.
{{< /lead >}}

{{< gallery >}}
  <img src="/images/projects/ddpm/x0_999.png" class="grid-w50" />
  <img src="/images/projects/ddpm/x0_249.png" class="grid-w50" />
  <img src="/images/projects/ddpm/x0_99.png" class="grid-w50" />
  <img src="/images/projects/ddpm/x0_0.png" class="grid-w50" />
{{< /gallery >}}

## Denoising Diffusion Probabilistic Models

DDPMs are generative machine learning architectures that learn to replicate complex data distributions, allowing them to create new samples that share characteristics with their training data. Think of them as learning the "recipe" for creating data from scratch.
The model operates through two complementary processes that form a Markov chain (where each state depends only on the immediately previous state):

- The **forward** diffusion process methodically corrupts the original data by adding small amounts of Gaussian noise over many timesteps (_T_). This follows a carefully designed schedule that gradually transforms meaningful data (like a clear image) into pure random noise. The analogy of ink diffusing in water is apt - just as a sharp droplet slowly spreads until it's uniformly distributed, the original data's structure is systematically dissolved into randomness.
- The **reverse** diffusion process is where the magic of generation happens. Starting from pure noise (timestep _T_), the model applies what it learned during training to iteratively "denoise" the data. At each step counting down from _T_ to 0, it:
    1. Identifies the noise component present in the current state
    2. Predicts how to partially remove it
    3. Produces a slightly cleaner version for the next step

This step-by-step reconstruction gradually reveals coherent patterns that match the statistical properties of the training data. It's like watching the ink diffusion process in reverse - random particles slowly coalescing into meaningful structure.
The model learns this denoising process through training on many examples, effectively discovering how to traverse the path from noise to realistic data samples.

## Dataset and Training
The model was trained using the [Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) from UCI Machine Learning Reposiroy. Training was performed on an NVIDIA RTX A4000 GPU (16GB VRAM) over 40 epochs.

For noise scheduling, I implemented the cosine variance schedule from [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672), which improves upon linear and squared schedules by providing smoother timestep transitions and enhanced sample quality and training stability.

## References
1. [Calvin Luo's](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) article beautifully explains the mathematics behind how diffusion models work. He provides an intuitive foundation, beginning with ELBO and tracing the evolution from variational autoencoders (VAEs) through Markovian hierarchical VAEs to modern diffusion models. 
2. The [Explaining AI's](https://www.youtube.com/@Explaining-AI) channel offers thorough coverage of both [theory](https://www.youtube.com/watch?v=H45lF4sUgiE) and [implementation](https://www.youtube.com/watch?v=vu6eKteJWew), with supporting code available in his [GitHub repository](https://github.com/explainingai-code/DDPM-Pytorch).
3. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672)
4. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

{{< github repo="qharo/Devanagiri-DDPM" >}}
