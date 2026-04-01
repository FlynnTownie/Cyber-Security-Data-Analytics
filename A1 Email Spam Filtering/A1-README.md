# SMS Spam Detection — Machine Learning Classifiers

> **Subject:** Data Analytics in Cyber Security (41180) — Assessment 1
> **Grade:** 95%
> **Tools:** Python, scikit-learn, TensorFlow/Keras, NLTK, GloVe

---

## Overview

This project applies supervised machine learning to the classic SMS spam classification problem. The goal is to build and compare multiple classifiers capable of distinguishing spam messages from legitimate (ham) ones, using natural language processing (NLP) pipelines and both traditional ML and deep learning approaches.

The work demonstrates how text-based threat indicators — such as phishing messages — can be automatically detected using the same ML pipeline patterns used in security operations for alert classification and log analysis.

---

## Dataset

- **Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size:** 5,572 SMS messages
- **Class distribution:** 4,825 ham (86.6%) / 747 spam (13.4%)
- **Format:** CSV with label (`ham`/`spam`) and raw message text

---

## Approach

### Text Pre-processing Pipeline

1. Removed punctuation and stripped stopwords using NLTK
2. Applied TF-IDF vectorisation for traditional classifiers
3. Used tokenisation and sequence padding for the LSTM model
4. Integrated 100-dimensional [GloVe Twitter embeddings](https://nlp.stanford.edu/projects/glove/) (`glove.twitter.27B.100d`) for the deep learning model

### Models Implemented

| Model | Variant | Notes |
|---|---|---|
| Support Vector Classifier (SVC) | Linear kernel, γ=1.0 | Baseline strong performer |
| Logistic Regression | L1 penalty, liblinear solver, C=0.1 | Interpretable, fast |
| Multi-Layer Perceptron (MLP) | sklearn MLPClassifier | Adam solver, 2 hidden layers |
| LSTM (Deep Learning) | Bidirectional LSTM via Keras | GloVe embeddings, dropout regularisation |

---

## Results

| Model | Accuracy | Precision (weighted) | Recall (weighted) |
|---|---|---|---|
| SVC (linear) | 97.97% | — | — |
| Logistic Regression | 94.32% | 96.5% | 96.4% |
| MLP (sklearn) | 96.35% | 96.5% | 96.4% |
| LSTM (GloVe + Keras) | ~97%+ | — | — |

SVC achieved the best accuracy at 97.97%, correctly identifying 200 spam messages out of 232 in the test set while producing only 2 false positives. The LSTM with GloVe embeddings was trained as a deep learning baseline to compare neural sequence modelling against classical approaches.

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras nltk textblob wordcloud termcolor
```

You also need to download the GloVe embeddings:
- [GloVe Twitter 27B](https://nlp.stanford.edu/data/glove.twitter.27B.zip) — extract `glove.twitter.27B.100d.txt` to the project root

### NLTK Downloads

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')
```

### Run the Notebook

```bash
jupyter notebook A1_Email_Spam_Filtering.ipynb
```

Ensure `spam.csv` is in the same directory as the notebook.

---

## Project Structure

```
├── A1_Email_Spam_Filtering.ipynb   # Main notebook
├── spam.csv                        # Dataset
├── glove.twitter.27B.100d.txt      # GloVe embeddings (download separately)
└── A1-README.md
```

---

## Key Concepts

**TF-IDF Vectorisation** converts raw text into numerical feature vectors by weighting terms based on how often they appear in a document relative to the overall corpus. This is a standard technique in security for analysing log entries and alert messages.

**GloVe Embeddings** are pre-trained word vectors trained on 27 billion Twitter tokens. Using pre-trained embeddings allows the LSTM to leverage semantic relationships between words without requiring a large labelled corpus.

**LSTM (Long Short-Term Memory)** networks are a type of recurrent neural network designed to capture long-range dependencies in sequential data — making them suitable for text, network traffic sequences, and time-series security data.
