import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib.pyplot as plt


# ==============================
# USER INPUT PARAMETERS
# ==============================

dataset_choice = input("Dataset (mnist/fashion): ")
epochs = int(input("Epochs: "))
batch_size = int(input("Batch size: "))
noise_dim = int(input("Noise dimension: "))
learning_rate = float(input("Learning rate: "))
save_interval = int(input("Save interval: "))


# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ==============================
# DATASET
# ==============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

if dataset_choice == "mnist":
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

elif dataset_choice == "fashion":
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

else:
    raise ValueError("Invalid dataset choice")


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ==============================
# GENERATOR
# ==============================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(noise_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):

        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)

        return img


# ==============================
# DISCRIMINATOR
# ==============================

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):

        return self.model(img)


# ==============================
# MODEL INIT
# ==============================

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()

opt_G = optim.Adam(G.parameters(), lr=learning_rate)
opt_D = optim.Adam(D.parameters(), lr=learning_rate)


# ==============================
# FOLDERS
# ==============================

os.makedirs("generated_samples", exist_ok=True)
os.makedirs("final_generated_images", exist_ok=True)


# ==============================
# TRAINING
# ==============================

for epoch in range(1, epochs + 1):

    for real_imgs, _ in loader:

        real_imgs = real_imgs.to(device)
        batch = real_imgs.size(0)

        real_labels = torch.ones(batch, 1).to(device)
        fake_labels = torch.zeros(batch, 1).to(device)

        # --------------------
        # Train Discriminator
        # --------------------

        noise = torch.randn(batch, noise_dim).to(device)
        fake_imgs = G(noise)

        real_loss = loss_fn(D(real_imgs), real_labels)
        fake_loss = loss_fn(D(fake_imgs.detach()), fake_labels)

        d_loss = real_loss + fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # --------------------
        # Train Generator
        # --------------------

        noise = torch.randn(batch, noise_dim).to(device)
        fake_imgs = G(noise)

        g_loss = loss_fn(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")

    # Save images
    if epoch % save_interval == 0:

        save_image(fake_imgs[:25], f"generated_samples/epoch_{epoch}.png", nrow=5, normalize=True)


# ==============================
# FINAL IMAGES
# ==============================

noise = torch.randn(100, noise_dim).to(device)
generated = G(noise)

for i in range(100):

    save_image(generated[i], f"final_generated_images/img_{i}.png", normalize=True)
