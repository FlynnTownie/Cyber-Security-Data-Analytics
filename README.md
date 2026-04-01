# Data Analytics in Cyber Security — ML Projects

> **Subject:** Data Analytics in Cyber Security (41180) — UTS Bachelor of Cyber Security
> **Tools:** Python, scikit-learn, TensorFlow/Keras, PyTorch, NLTK

---

## Overview

This repository contains three machine learning projects completed as part of the Data Analytics in Cyber Security subject at UTS. Each assessment applies a different class of ML/DL technique to a security-relevant problem — from text classification for phishing/spam detection, to adversarial robustness testing of neural network classifiers, to network intrusion detection using traffic flow features.

---

## Assessments

### [Assessment 1 — SMS Spam Detection](./A1%20Email%20Spam%20Filtering/A1-README.md)
> Grade: 27 / 30

Supervised text classification using TF-IDF vectorisation and four classifier architectures (SVC, Logistic Regression, sklearn MLP, LSTM with GloVe embeddings) applied to the UCI SMS Spam Collection dataset. Best result: SVC at **97.97% accuracy**.

Techniques: NLP pre-processing pipeline, TF-IDF, GloVe word embeddings, Keras LSTM, confusion matrix analysis.

---

### [Assessment 2 — Adversarial Attacks & Defence](./A2%20Intrusion%20Detection/A2-README.md)
> Grade: 28 / 30

Implementation and analysis of the Fast Gradient Sign Method (FGSM) adversarial attack against CNN and MLP classifiers trained on MNIST and CIFAR-10. Includes adversarial training as a defence mechanism and evaluation of model robustness across epsilon values.

Techniques: PyTorch, FGSM, adversarial training, MLP and CNN architectures, robustness benchmarking.

---

### Assessment 3 — Network Intrusion Detection System (NIDS)

Deep learning-based binary classification of network traffic to detect DDoS attacks. Used a labelled network traffic dataset with engineered flow-level features. Compared ML baselines against a deep learning model and evaluated using precision, recall, F1-score, and ROC-AUC.

Techniques: Feature engineering on network flow data, Random Forest baseline, deep neural network classifier, SMOTE class balancing, scikit-learn, Keras.

---

## Repository Structure

```
DACS-Cybersecurity-ML/
├── README.md
├── A1 Email Spam Filtering/
│   ├── A1-README.md
│   ├── A1_Email_Spam_Filtering.ipynb
│   ├── spam.csv
│   └── Assessment 1 DACS - Flynn Townsend.pdf
└── A2 Intrusion Detection/
    ├── A2-README.md
    ├── A2_Intrusion_Detection.ipynb
    ├── Assessment 2 Adversarial Attacks - DACS.pdf
    ├── Dataset/
    │   ├── CIFAR10.pth
    │   └── MNIST.pth
    └── Train-Test/
        ├── train_mnist.pt
        └── test_mnist.pt
```

---

## Skills Demonstrated

- Supervised ML classification with scikit-learn (SVM, LR, MLP)
- Deep learning with TensorFlow/Keras (LSTM, Embedding layers)
- Deep learning with PyTorch (CNN, adversarial training loop)
- NLP pre-processing: tokenisation, stopword removal, TF-IDF, GloVe embeddings
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrices
- Adversarial ML: FGSM attack implementation and robustness defence
- Network traffic analysis and intrusion detection modelling

---

## Setup

Each assessment has its own dependencies and setup instructions. See the individual README files linked above. A shared base environment can be set up with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras torch torchvision nltk textblob wordcloud
```
