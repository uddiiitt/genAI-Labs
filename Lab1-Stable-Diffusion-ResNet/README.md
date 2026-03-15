# LAB 1: Stable Diffusion Dataset Generation & ResNet Classification

## Overview
This experiment explores the integration of Generative AI with classical computer vision techniques. 
The objective is to determine whether a classifier trained entirely on synthetic images can classify real-world objects.

## Workflow

### 1. Synthetic Data Generation
Images of 40 dog breeds are generated using Stable Diffusion (runwayml/stable-diffusion-v1-5).

Example Prompt:
"a high quality photo of a Golden Retriever, ultra realistic, cinematic lighting, 4k"

### 2. Dataset Structure
Images are automatically organized into folders by breed.

### 3. Model
ResNet18 pretrained on ImageNet is used for classification.

### 4. Training
Loss Function: CrossEntropyLoss  
Optimizer: Adam

### 5. Evaluation
Performance is evaluated using a confusion matrix.
