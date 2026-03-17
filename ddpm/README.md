# DDPM Implementation

This repository contains an implementation of Denoising Diffusion Probabilistic Models (DDPM). The codebase demonstrates how to progressively add noise to an image (forward process) and train a neural network to reverse the process to generate new images (reverse process).

## Project Structure

- `ddpm.ipynb`: The main Jupyter notebook containing the full pipeline—from dataset loading, forward diffusion and noise scheduling, to the UNet definition and training loop.

## Implementation Details

### Dataset

- **CIFAR-10**: The model is trained on the CIFAR-10 dataset.
- **Preprocessing**: Images are resized to `64x64`, horizontally flipped randomly for data augmentation, and scaled to a `[-1, 1]` pixel range.

### Forward Diffusion (Noise Scheduler)

- **Timesteps**: $T = 300$.
- **Beta Schedule**: A linear variance schedule is used, with $\beta$ linearly increasing from $10^{-4}$ to $0.02$.
- The forward process is computed in closed form using pre-calculated cumulative products of $\alpha = 1 - \beta$.

### Neural Network (Denoising Model)

- **Architecture**: A variant of the U-Net architecture (`SimpleUnet`). The downsampling path scales channels progressively `(64, 128, 256, 512, 1024)`, and the upsampling path reverses this while concatenating representations using skip connections.
- **Time Embeddings**: The temporal noise level $t$ is passed into the network via `SinusoidalPositionEmbeddings` (dim=32), projected through an MLP, and added at each U-Net block.

## References & Tutorials

**Tutorials:**

- [YouTube Tutorial](https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=970s)
- [Original Colab Notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=bpN_LKYwuLx0)

**Papers:**

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) (Ho et al.)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233) (Dhariwal & Nichol)
