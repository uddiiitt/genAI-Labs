import torch
import torch.nn as nn

# ==============================
# DATASET
# ==============================
text = """
artificial intelligence systems learn patterns from data
sequence models process information step by step
recurrent neural networks are useful for sequence prediction
lstm networks handle long term dependencies
deep learning models improve sequence learning
generative models create new samples from learned patterns
language models predict the next word in a sentence
sequence generation is used in chatbots and assistants
machine learning helps computers learn automatically
training data improves model accuracy
neural networks simulate human brain structures
optimization algorithms improve learning efficiency
technology is transforming modern education
online learning platforms use artificial intelligence
students benefit from intelligent tutoring systems
automation improves productivity and decision making
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
    X, y = [], []
    
    for i in range(len(words) - seq_length):
        X.append([word_to_idx[w] for w in words[i:i+seq_length]])
        y.append(word_to_idx[words[i+seq_length]])
    
    return torch.tensor(X), torch.tensor(y)

X, y = create_sequences(words)

# ==============================
# LSTM MODEL
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==============================
# TRANSFORMER MODEL
# ==============================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 2),
            num_layers=2
        )
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out[-1]
        return self.fc(out)

# ==============================
# TRAIN FUNCTION
# ==============================
def train(model, X, y, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        output = model(X)
        loss = loss_fn(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==============================
# GENERATE TEXT
# ==============================
def generate(model, seed, length=10):
    words = seed.split()
    
    for _ in range(length):
        seq = torch.tensor([[word_to_idx[w] for w in words[-3:]]])
        pred = model(seq)
        next_word = idx_to_word[pred.argmax().item()]
        words.append(next_word)
    
    return " ".join(words)

# ==============================
# MAIN
# ==============================
def main():
    print("Training LSTM...")
    lstm = LSTMModel(vocab_size)
    train(lstm, X, y)
    
    print("\nLSTM Output:")
    print(generate(lstm, "artificial intelligence systems"))
    
    print("\nTraining Transformer...")
    transformer = TransformerModel(vocab_size)
    train(transformer, X, y)
    
    print("\nTransformer Output:")
    print(generate(transformer, "sequence models process"))

if __name__ == "__main__":
    main()
