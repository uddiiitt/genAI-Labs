import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURATION
# ==============================
latent_dim = 100
image_size = 64
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)

# ==============================
# DATA LOADING
# ==============================
def load_data():
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


# ==============================
# GENERATOR MODEL (DCGAN)
# ==============================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output [-1, 1]
        )

    def forward(self, x):
        return self.model(x)


# ==============================
# LOAD GENERATOR
# ==============================
def load_generator():
    generator = Generator().to(device)

    # NOTE: In real lab, load pretrained weights
    # generator.load_state_dict(torch.load("generator.pth"))

    generator.eval()  # Freeze model
    return generator


# ==============================
# GENERATE IMAGES
# ==============================
def generate_images(generator, num_images=10):
    noise = torch.randn(num_images, latent_dim, 1, 1).to(device)

    with torch.no_grad():
        fake_images = generator(noise).cpu()

    save_images(fake_images, "outputs/generated.png")


# ==============================
# LATENT SPACE INTERPOLATION
# ==============================
def interpolate(generator, steps=10):
    z1 = torch.randn(1, latent_dim, 1, 1).to(device)
    z2 = torch.randn(1, latent_dim, 1, 1).to(device)

    interpolated = []

    for alpha in torch.linspace(0, 1, steps):
        z = alpha * z1 + (1 - alpha) * z2
        with torch.no_grad():
            img = generator(z).cpu()
        interpolated.append(img)

    interpolated = torch.cat(interpolated)
    save_images(interpolated, "outputs/interpolation.png")


# ==============================
# SAVE IMAGES
# ==============================
def save_images(images, filename):
    grid = torchvision.utils.make_grid(images, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(filename)
    plt.close()


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    print("Loading data...")
    load_data()  # Just for lab requirement

    print("Loading generator...")
    generator = load_generator()

    print("Generating images...")
    generate_images(generator)

    print("Interpolating latent space...")
    interpolate(generator)

    print("Done! Check outputs folder.")


if __name__ == "__main__":
    main()
