# LAB 3: Variational Autoencoder (VAE) for Image Generation

## Objective
The goal of this experiment is to implement a Variational Autoencoder (VAE) to learn latent representations of image data and generate new synthetic samples.

## Overview

A Variational Autoencoder is a generative model consisting of two main components:

### Encoder
The encoder compresses input images into a latent representation defined by:
- Mean (μ)
- Log variance (log σ²)

### Reparameterization Trick
Instead of sampling directly, the model samples from a standard normal distribution and transforms it using μ and σ.

### Decoder
The decoder reconstructs images from latent vectors.

## Dataset

One dataset is used:

- MNIST (handwritten digits)
or
- Fashion-MNIST (clothing images)

Loaded using Torchvision.

## Workflow

### 1. Dataset Preparation
- Load dataset
- Normalize images
- Split into training and testing sets

### 2. Build VAE Architecture
- Encoder network
- Latent mean and log variance
- Reparameterization trick
- Decoder network

### 3. Loss Function
The total loss consists of:

Reconstruction Loss  
Measures similarity between input and output images.

KL Divergence  
Encourages the latent distribution to follow a standard normal distribution.

### 4. Training
The VAE is trained for multiple epochs while monitoring loss.

### 5. Sample Generation
Random vectors are sampled from the latent space and passed through the decoder to generate new images.

### 6. Latent Space Visualization
The latent space can optionally be visualized in 2D.

## Outputs

The program produces:

- Trained VAE model
- Reconstructed images
- Generated samples
- Loss curves

## Technologies Used

- PyTorch
- Torchvision
- Matplotlib
