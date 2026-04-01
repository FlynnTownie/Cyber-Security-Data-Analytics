# Adversarial Attacks & Defence on Neural Networks

> **Subject:** Data Analytics in Cyber Security (41180) — Assessment 2
> **Grade:** 28 / 30
> **Tools:** Python, PyTorch, torchvision, MNIST, CIFAR-10

---

## Overview

This project investigates adversarial machine learning — specifically how small, deliberately crafted perturbations to input data can cause a neural network classifier to misclassify images with high confidence. It then implements adversarial training as a defence mechanism and measures the resulting improvement in model robustness.

This is directly relevant to security contexts where ML models are used for detection tasks (e.g., malware classification, intrusion detection), because adversarial inputs represent a class of evasion attacks against those models.

---

## Background

Adversarial examples were first formally described by Goodfellow et al. (2014) in *Explaining and Harnessing Adversarial Examples* (arXiv:1412.6572). The core idea is that neural networks are sensitive to small input perturbations that are imperceptible to humans but cause confident misclassification.

The **Fast Gradient Sign Method (FGSM)** generates adversarial examples by computing the gradient of the loss with respect to the input, then nudging the input in the direction that increases loss:

```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

Where `ε` controls the perturbation magnitude. Higher `ε` causes more visible distortion but stronger attacks.

---

## Experiments

### Datasets

| Dataset | Classes | Image Size | Training Samples |
|---|---|---|---|
| MNIST | 10 (digits 0–9) | 28×28 greyscale | 60,000 |
| CIFAR-10 | 10 (objects) | 32×32 RGB | 50,000 |

### Models Implemented

**MLP Classifier (MNIST)** — 4-layer fully connected network:
```
Input (784) → FC(512, ReLU) → FC(128, ReLU) → FC(10)
```

**CNN Classifier (MNIST & CIFAR-10)**:
```
Conv(32, 3×3, ReLU) → MaxPool → Conv(64, 3×3, ReLU) → MaxPool → FC(128, ReLU) → FC(10)
```

### Attack Parameters

| Parameter | Value | Description |
|---|---|---|
| `ε` (epsilon) | 0.03 – 0.25 | Perturbation magnitude |
| Method | FGSM | Single-step gradient attack |
| Evaluation | Post-attack accuracy | Accuracy drop from clean baseline |

---

## Results

FGSM at `ε = 0.25` on MNIST caused a significant accuracy drop on the clean-trained MLP. After adversarial training (5 epochs, `ε = 0.2`), the model recovered meaningful robustness against FGSM examples at the same perturbation level.

On CIFAR-10, the more complex RGB input space made the model more susceptible to perturbation, illustrating that defence mechanisms must be tuned per dataset and architecture.

---

## Defence: Adversarial Training

Adversarial training augments the training process by including FGSM-generated examples in each training batch:

```python
def adversarial_train(model, loader, epsilon, epochs):
    for epoch in range(epochs):
        for images, labels in loader:
            adv_images = fgsm_attack(model, images, labels, epsilon)
            # Train on adversarial examples instead of clean inputs
            output = model(adv_images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
```

This is the most widely studied practical defence and forms the basis for certified robustness research.

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision matplotlib pandas
```

### Run the Notebook

```bash
jupyter notebook A2_Intrusion_Detection.ipynb
```

MNIST and CIFAR-10 datasets are downloaded automatically via `torchvision.datasets`. If using pre-saved `.pt` files, place them in a `Data/` subdirectory.

---

## Project Structure

```
├── A2_Intrusion_Detection.ipynb    # Main notebook
├── Dataset/
│   ├── MNIST.pth                   # Pre-saved MNIST model weights (optional)
│   └── CIFAR10.pth                 # Pre-saved CIFAR-10 model weights (optional)
├── Train-Test/
│   ├── train_mnist.pt              # Pre-saved MNIST train data (optional)
│   └── test_mnist.pt               # Pre-saved MNIST test data (optional)
└── A2-README.md
```

---

## References

- Goodfellow, I., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv:1412.6572. https://arxiv.org/abs/1412.6572
- PyTorch Documentation: https://pytorch.org/tutorials/
- Madry, A., et al. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR 2018.

---

## Key Concepts

**Evasion Attack** — an adversarial input crafted to bypass a trained model at inference time, without modifying the model itself. This is the primary threat model for ML-based security tools.

**Adversarial Robustness** — a model's ability to maintain correct predictions under adversarial perturbations. Measured by comparing clean accuracy to adversarial accuracy across epsilon values.

**Adversarial Training** — incorporating adversarial examples into the training loop as a data augmentation strategy. Currently the most effective practical defence against first-order attacks like FGSM and PGD.
