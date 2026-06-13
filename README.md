# Computer Vision Architectures – From Scratch Implementations

## Overview

This repository is a structured exploration of classical and modern computer vision architectures implemented in PyTorch.

The goal of this project is to deeply understand the evolution of convolutional and attention-based models — from AlexNet to Vision Transformers — by implementing them modularly and analyzing their architectural trade-offs.

---

## Architectures Implemented

- AlexNet  
- VGG  
- ResNet  
- DenseNet  
- GoogLeNet (Inception v1)  
- MobileNet  
- SqueezeNet  
- Vision Transformer (ViT)  

---

## Additional Modules

- Convolution operations (from scratch implementation)
- Regularization techniques
- Transfer learning strategies

## OCR & Text Detection modules

- DBNet (text detector) — implementation and pretrained weights are available under `DBNet/`. This repository includes a `dbnet_icdar_best.pth` checkpoint that the `OCR` pipeline uses for text localization.
- CRNN (text recognizer) — recognition model and training/inference notebooks are in `CRNN/` (see `CRNN/README.md`).
- OCR pipeline — end-to-end detection + recognition notebooks are in `OCR/` and demonstrate DBNet -> crop -> CRNN flow.

---

## Research Motivation

This repository was created to:

- Understand architectural innovations in deep learning
- Analyze parameter efficiency vs performance trade-offs
- Compare convolution-based and attention-based approaches
- Explore generalization techniques in deep neural networks

---

## Key Insights from Implementation

- Residual connections mitigate vanishing gradients
- Dense connectivity encourages feature reuse
- Depthwise separable convolutions reduce computational cost
- Transformers remove spatial locality bias but require larger datasets
- Regularization techniques significantly improve generalization

---

## Tech Stack

- Python
- PyTorch
- NumPy

---

## Future Work

- Hybrid CNN-Transformer architectures
- 3D CNNs for hyperspectral imagery
- Vision-Language Models (VLMs)
- Self-Supervised Pretraining methods

---

> This repository was created as a structured study of deep learning architectures to understand their mathematical foundations and architectural evolution.
