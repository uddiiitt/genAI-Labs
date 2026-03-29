#  Lab 8 – Artistic Image Generation using GANs

##  Course

**CSET419 – Introduction to Generative AI**

---

##  Objective

The goal of this lab is to generate artistic images using **Generative Adversarial Networks (GANs)** by exploring the **latent space** of the generator.

We will:

* Use a trained GAN model (DCGAN)
* Generate images using random latent vectors
* Perform interpolation between latent vectors
* Observe how image outputs change smoothly

---

##  Key Concepts

* GAN (Generator + Discriminator)
* Latent Space (random noise input)
* Image Generation
* Latent Vector Interpolation

---

## ⚙️ Workflow

### 1. Data Preparation

* Load dataset (CIFAR-10)
* Normalize images between **[-1, 1]**
* Define latent vector size

### 2. Load Trained GAN

* Use a **DCGAN Generator**
* Freeze weights (no training)
* Generate images from random noise

### 3. Latent Space Exploration

* Generate multiple images from random vectors
* Interpolate between two latent vectors
* Observe smooth transitions

### 4. Artistic Output Generation

* Create 5–10 generated samples
* Visualize outputs
* Compare patterns and diversity

---

##  Output

* Generated artistic images
* Interpolated image sequence
* Variation in outputs from different latent vectors

---

##  Learning Outcomes

After this lab, you will understand:

* How GANs generate images
* Role of latent space in creativity
* Difference between GAN architectures
* How AI can create artistic content

---

##  How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the code

```bash
python main.py
```

### 3. Output

* Images will be saved in:

```
outputs/
```

---

##  Project Structure

```
Lab8-GAN-Art/
│── main.py
│── requirements.txt
│── README.md
│── outputs/
```

---

##  Notes

* This implementation uses **DCGAN (Basic GAN)**
* You can extend it to:

  * StyleGAN
  * CycleGAN
  * BigGAN

---
