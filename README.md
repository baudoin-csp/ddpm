# DDPM Implementation

This repository contains a Denoising Diffusion Probabilistic Models (DDPM) implementation trained on CIFAR-10. The notebook is still available for exploration, and the project now also includes a standalone Python script so you can run training directly in Google Colab with `python ddpm.py`.

## Project Structure

- `ddpm.ipynb`: Original notebook version.
- `ddpm.py`: Standalone training script converted from the notebook.
- `requirements.txt`: Dependencies installable with `pip`.

## Getting Started

### Local Installation

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run training:

   ```bash
   python ddpm.py --epochs 5 --sample-every 1
   ```

Generated samples and the final checkpoint are written to `outputs/` by default.

### Google Colab Workflow

Use Colab's default Python environment and install dependencies with `pip`.

1. Clone the repository:

   ```bash
   !git clone https://github.com/baudoin-csp/ddpm.git
   %cd ddpm
   ```

2. Install dependencies:

   ```bash
   !pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   !python ddpm.py --epochs 5 --sample-every 1 --output-dir outputs
   ```

4. Inspect the generated files:

   ```bash
   !ls outputs
   ```

If you want faster training in Colab, switch the runtime to GPU before running the script. The script automatically uses CUDA when it is available.

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
