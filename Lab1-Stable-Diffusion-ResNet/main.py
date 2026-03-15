# ======================================================
# LAB 1
# Generate Dog Images using Stable Diffusion
# Train ResNet18 to classify them
# ======================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from diffusers import StableDiffusionPipeline

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ------------------------------------------------------
# 1. Select device (GPU if available)
# ------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------------------------------
# 2. List of Dog Breeds (40 classes)
# ------------------------------------------------------

DOG_BREEDS = [
    "Labrador Retriever","Golden Retriever","German Shepherd","French Bulldog",
    "Bulldog","Poodle (Standard)","Beagle","Rottweiler","Dachshund",
    "Yorkshire Terrier","Boxer","Doberman Pinscher","Siberian Husky",
    "Great Dane","Shih Tzu","Chihuahua","Pug","Border Collie",
    "Australian Shepherd","Belgian Malinois","Akita","Alaskan Malamute",
    "Samoyed","Bernese Mountain Dog","Saint Bernard","Newfoundland",
    "Cane Corso","Greyhound","Whippet","Bloodhound","Basset Hound",
    "Rhodesian Ridgeback","Weimaraner","Vizsla","Cocker Spaniel",
    "Cavalier King Charles Spaniel","Pomeranian","Chow Chow",
    "Shiba Inu","Basenji"
]


# ------------------------------------------------------
# 3. Load Stable Diffusion Model
# ------------------------------------------------------

def load_diffusion_model():

    model_name = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=True
    )

    pipe = pipe.to(device)

    return pipe


# ------------------------------------------------------
# 4. Generate Synthetic Images
# ------------------------------------------------------

def generate_images(pipe, breeds, folder, images_per_class=3):

    os.makedirs(folder, exist_ok=True)

    for breed in breeds:

        breed_folder = os.path.join(folder, breed.replace(" ", "_"))
        os.makedirs(breed_folder, exist_ok=True)

        for i in range(images_per_class):

            prompt = f"high quality photo of {breed}, ultra realistic, 4k"

            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]

            file_name = f"{breed.replace(' ','_')}_{i}.png"
            image_path = os.path.join(breed_folder, file_name)

            image.save(image_path)

            print("Saved:", image_path)


# ------------------------------------------------------
# 5. Load Dataset
# ------------------------------------------------------

def load_dataset(folder):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    dataset = datasets.ImageFolder(folder, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    return dataset, train_loader, val_loader


# ------------------------------------------------------
# 6. Build ResNet18 Model
# ------------------------------------------------------

def create_model(num_classes):

    model = models.resnet18(pretrained=True)

    # Replace last layer for 40 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


# ------------------------------------------------------
# 7. Train Model
# ------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=5):

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        accuracy = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")


# ------------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------------

def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, 1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# ------------------------------------------------------
# 9. Confusion Matrix
# ------------------------------------------------------

def show_confusion_matrix(model, loader):

    true_labels = []
    pred_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs,1)

            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())

    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


# ------------------------------------------------------
# 10. Main Pipeline
# ------------------------------------------------------

DATASET_FOLDER = "40_dog_dataset"

pipe = load_diffusion_model()

generate_images(pipe, DOG_BREEDS, DATASET_FOLDER, images_per_class=3)

dataset, train_loader, val_loader = load_dataset(DATASET_FOLDER)

model = create_model(len(dataset.classes))

train_model(model, train_loader, val_loader)

torch.save(model.state_dict(), "resnet18_dog_model.pth")

show_confusion_matrix(model, val_loader)
