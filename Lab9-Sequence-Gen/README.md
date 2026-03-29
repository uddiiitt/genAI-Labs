# Lab 9 – Sequence Generation using RNN, LSTM and Transformer

## Course

CSET419 – Introduction to Generative AI

---

## Objective

In this lab, we work with **sequential data (text)** and try to build models that can learn patterns from it and generate new sequences.

We use two approaches:

* LSTM (a type of RNN)
* Transformer (more advanced model)

---

## About the Task

The dataset consists of simple sentences related to machine learning and AI.
The goal is to train a model so that, given a few words, it can predict and generate the next words.

Example:
Input → *"machine learning models"*
Output → *"machine learning models learn patterns from data"*

---

## What I Did

### Part 1 – LSTM Model

* Converted text into words (tokenization)
* Created input-output pairs (sequence → next word)
* Built an LSTM model using PyTorch
* Trained the model on the dataset
* Generated new sequences using a seed input

---

### Part 2 – Transformer Model

* Used the same dataset
* Built a simple Transformer encoder model
* Trained it on the sequences
* Generated text using a starting phrase

---

## Output

* The model generates new sentences based on input words
* LSTM gives simple sequential outputs
* Transformer produces slightly better structured results

---

## What I Learned

* How sequence data is handled in machine learning
* How models predict the next word in a sequence
* Basic working of LSTM and Transformer
* How generative models can create new text

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Folder Structure

```
Lab9-Sequence-Gen/
│── main.py
│── requirements.txt
│── README.md
```

---

## Notes

* This is a basic implementation for learning purposes
* The dataset is small, so outputs may not always be perfect
* Can be improved using larger datasets or pretrained models
