import torch
import torch.nn as nn

# Sample dataset
pairs = [
    ("hello", "hi how are you"),
    ("how are you", "i am fine"),
    ("what is your name", "i am a chatbot")
]

# Build vocabulary
words = set()
for p in pairs:
    words.update(p[0].split())
    words.update(p[1].split())

word_to_idx = {w: i for i, w in enumerate(words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# Convert sentence to tensor
def encode(sentence):
    return torch.tensor([word_to_idx[w] for w in sentence.split()])

# Model
class AttentionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.attn = nn.Linear(64, 1)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        # Attention weights
        weights = torch.softmax(self.attn(out), dim=1)
        context = (weights * out).sum(dim=1)

        return self.fc(context)

# Train
model = AttentionModel(len(words))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    total_loss = 0
    for inp, out in pairs:
        x = encode(inp).unsqueeze(0)
        y = encode(out)[0].unsqueeze(0)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test
def generate(text):
    x = encode(text).unsqueeze(0)
    pred = model(x)
    word = idx_to_word[pred.argmax().item()]
    return word

print("Input: how are you")
print("Output:", generate("how are you"))
