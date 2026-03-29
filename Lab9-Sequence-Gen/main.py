import torch
import torch.nn as nn
import numpy as np

# ==============================
# DATASET
# ==============================
text = """
machine learning models learn patterns from data
sequence models process data step by step
recurrent neural networks are designed for sequential tasks
rnn models maintain hidden states across time steps
long short term memory networks solve long dependency problems
lstm uses gates to control information flow
gru models simplify the lstm architecture
sequence prediction is useful in many applications
language modeling predicts the next word in a sentence
speech recognition processes audio sequences
time series forecasting predicts future values
music generation creates new melodies
generative models learn probability distributions
they generate new samples similar to training data
sequence generation is widely used in artificial intelligence
deep learning improves sequence modeling performance
"""

# ==============================
# PREPROCESSING
# ==============================
def preprocess(text):
    words = text.split()
    vocab = list(set(words))
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    return words, word_to_idx, idx_to_word, len(vocab)

words, word_to_idx, idx_to_word, vocab_size = preprocess(text)

# ==============================
# CREATE SEQUENCES
# ==============================
seq_length = 3

def create_sequences(words):
    X = []
    y = []
    
    for i in range(len(words) - seq_length):
        seq = words[i:i+seq_length]
        target = words[i+seq_length]
        
        X.append([word_to_idx[w] for w in seq])
        y.append(word_to_idx[target])
        
    return torch.tensor(X), torch.tensor(y)

X, y = create_sequences(words)

# ==============================
# LSTM MODEL
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(model, X, y, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        output = model(X)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==============================
# GENERATE TEXT
# ==============================
def generate_text(model, seed, length=10):
    model.eval()
    
    words = seed.split()
    
    for _ in range(length):
        seq = torch.tensor([[word_to_idx[w] for w in words[-seq_length:]]])
        with torch.no_grad():
            pred = model(seq)
        
        next_word = idx_to_word[pred.argmax().item()]
        words.append(next_word)
    
    return " ".join(words)

# ==============================
# TRANSFORMER MODEL (Simple)
# ==============================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=32, num_heads=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads),
            num_layers=2
        )
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Required shape
        out = self.transformer(x)
        out = out[-1]
        return self.fc(out)

# ==============================
# MAIN
# ==============================
def main():
    print("Training LSTM...")
    lstm = LSTMModel(vocab_size)
    train_model(lstm, X, y)
    
    print("\nGenerated (LSTM):")
    print(generate_text(lstm, "machine learning models"))
    
    print("\nTraining Transformer...")
    transformer = TransformerModel(vocab_size)
    train_model(transformer, X, y)
    
    print("\nGenerated (Transformer):")
    print(generate_text(transformer, "sequence models process"))

if __name__ == "__main__":
    main()
