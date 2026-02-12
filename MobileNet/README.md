# MobileNet – Efficient CNN Architecture

## Overview

MobileNet uses depthwise separable convolutions to reduce computational complexity.

---

## Architecture Highlights

- Depthwise convolution
- Pointwise (1×1) convolution
- Lightweight architecture
- Designed for mobile and embedded systems

---

## Computational Advantage

Standard Convolution Cost:
Dk × Dk × M × N × Df × Df

Depthwise Separable Convolution:
(Dk × Dk × M × Df × Df) + (1 × 1 × M × N × Df × Df)

---

## Observations

- Significantly reduces parameters
- Suitable for low-resource environments
