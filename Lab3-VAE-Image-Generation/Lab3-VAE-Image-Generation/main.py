import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# =============================
# USER INPUT
# =============================

dataset_choice = input("Dataset (mnist/fashion): ")
epochs = int(input("Epochs: "))
batch_size = int(input("Batch size: "))
latent_dim = int(input("Latent dimension: "))
learning_rate = float(input("Learning rate: "))


# =============================
# DEVICE
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# =============================
# DATASET
# =============================

transform = transforms.Compose([
    transforms.ToTensor()
])

if dataset_choice == "mnist":
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

elif dataset_choice == "fashion":
    dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)

else:
    raise ValueError("Invalid dataset")

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================
# VAE MODEL
# =============================

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)

        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):

        h = torch.relu(self.fc1(x))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        h = torch.relu(self.fc3(z))

        return torch.sigmoid(self.fc4(h))

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


model = VAE().to(device)


# =============================
# LOSS FUNCTION
# =============================

def loss_function(recon_x, x, mu, logvar):

    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =============================
# TRAINING
# =============================

loss_list = []

for epoch in range(epochs):

    total_loss = 0

    for images, _ in loader:

        images = images.view(-1, 784).to(device)

        recon, mu, logvar = model(images)

        loss = loss_function(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    loss_list.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


# =============================
# GENERATE NEW SAMPLES
# =============================

os.makedirs("generated_samples", exist_ok=True)

with torch.no_grad():

    z = torch.randn(25, latent_dim).to(device)

    samples = model.decode(z).cpu()

    samples = samples.view(25,1,28,28)

    from torchvision.utils import save_image

    save_image(samples, "generated_samples/generated.png", nrow=5)


# =============================
# LOSS CURVE
# =============================

plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
