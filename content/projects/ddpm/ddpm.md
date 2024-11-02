---
title: "Devanagiri-DDPM"
date: "2023-01-01"
summary: "A PyTorch implementation of DDPM trained to generate Devanagiri Script."
type: "Project"
image: "img/DevanagiriDDPM.png"  # Optional
# link: "https://github.com/yourusername/project"  # Optional
weight: 1  # Controls the order (lower numbers appear first)
---

{{< lead >}}
A PyTorch implementation of a [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239) trained to generate [Devanagari](https://en.wikipedia.org/wiki/Devanagari) characters. It trains a 10 million parameter U-Net on 96,000 32x32 images.
{{< /lead >}}

{{< gallery >}}
  <img src="images/x0_0.png" class="grid-w33" />
{{< /gallery >}}

## Denoising Diffusion Probabilistic Models

DDPMs are generative machine learning architectures designed to learn and replicate the underlying probability distribution of a dataset, enabling the creation of new, similar data samples.

The training setup forms a Markov chain and operates in two distinct phases: \
The **forward** diffusion process systematically introduces Gaussian noise to the original data across a large number of timesteps (_T_), following a predetermined schedule. This gradually transforms the meaningful data into pure noise, analogous to how a drop of ink would slowly diffuse through water until completely distributed. 

The **reverse** diffusion process is where the actual generation occurs. The model, trained to identify and remove noise at each step, begins with pure noise and iteratively denoises it across timesteps _T_ through 0. At each step, it predicts and removes the noise component, gradually reconstructing a coherent sample that matches the learned data distribution.

The math behind how these models work is fascinating, very clearly explained by Calvin Luo and Explaining AI.



## Dataset and Training
The dataset I used is from UCI Machine Learning Repository. I trained the model on an RTX A4000 for 40 epochs. I've implemented the cosine scheduler from [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672)

## Results

