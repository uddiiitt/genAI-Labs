# LAB 2: GAN Image Generation

## Objective
To implement and train a basic Generative Adversarial Network (GAN) that generates synthetic images similar to a given dataset.

## Overview
Generative Adversarial Networks (GANs) consist of two neural networks:

1. **Generator**
   - Generates fake images from random noise.

2. **Discriminator**
   - Distinguishes between real and fake images.

The two networks compete with each other during training, which gradually improves the quality of generated images.

## Dataset
One dataset is used:

- MNIST (handwritten digits)
OR
- Fashion-MNIST (clothing items)

Loaded using **Torchvision datasets**.

## Workflow

### 1. Generator
Takes random noise as input and generates synthetic images.

### 2. Discriminator
Classifies whether an image is **real or fake**.

### 3. Training
Both models are trained alternately:

- Discriminator learns to detect fake images.
- Generator learns to fool the discriminator.

### 4. Image Generation
Synthetic images are generated periodically and saved as image grids.

### 5. Label Prediction
Generated images are classified using a pre-trained classifier model.

## Outputs

The program generates:

- Training logs
- Periodic sample images
- Final generated images
- Label distribution of generated images

## Technologies Used

- PyTorch
- Torchvision
- GAN Architecture
- Matplotlib
