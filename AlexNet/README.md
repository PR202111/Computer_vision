# AlexNet – PyTorch Implementation

## Overview

This folder contains a PyTorch implementation of **AlexNet**, one of the earliest deep convolutional neural networks that demonstrated breakthrough performance on ImageNet (2012).

The objective is to understand early deep CNN design and convolutional feature hierarchies.

---

## Architecture Highlights

- 5 Convolutional Layers
- ReLU activations
- Max Pooling
- Dropout regularization
- 3 Fully Connected layers

---

## Key Concepts Explored

- Large receptive fields (11×11 initial convolution)
- Hierarchical feature extraction
- Effect of stride and padding on spatial dimensions
- Parameter-heavy fully connected layers

---

## Implementation Details

- Built using `torch.nn.Module`
- Modular layer definitions
- Custom forward pass
- Configurable number of output classes

---

## Observations

- ReLU significantly accelerates convergence
- Dropout reduces overfitting in fully connected layers
- Early layers capture low-level features (edges, textures)
