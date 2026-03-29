# Lab 10 – Sequential Data Generation using RNN, LSTM and Transformer

## Course

CSET419 – Introduction to Generative AI

---

## Objective

In this lab, we implemented generative models that work on sequential data (text).
The aim was to train models that can learn patterns from sentences and generate new text sequences.

---

## About the Dataset

The dataset consists of simple sentences related to AI, machine learning, and technology.
These sentences are used to train models to predict the next word in a sequence.

---

## What I Did

### Part 1 – LSTM Model

* Converted text into word-level tokens
* Created input-output sequence pairs
* Built an LSTM-based model
* Trained the model on the dataset
* Generated new text using a seed input

---

### Part 2 – Transformer Model

* Used the same dataset
* Built a simple Transformer encoder model
* Trained it on sequences
* Generated text using starting words

---

## Output

* Generated text sequences from LSTM
* Generated text sequences from Transformer
* Both models try to predict the next word based on previous words

---

## What I Learned

* How sequence generation works
* Difference between LSTM and Transformer
* How models learn patterns from text
* Basic idea of generative models for sequences

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Folder Structure

```
Lab10-Sequence-Gen/
│── main.py
│── requirements.txt
│── README.md
```

---

## Notes

* This is a simple implementation for learning
* Output may not be perfect due to small dataset
* Can be improved using larger datasets or pretrained models
