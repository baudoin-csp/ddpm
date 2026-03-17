import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model on CIFAR-10.")
    parser.add_argument(
        "--data-dir", type=str, default=".", help="Dataset root directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for samples and checkpoints.",
    )
    parser.add_argument(
        "--img-size", type=int, default=64, help="Image size used during training."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument(
        "--timesteps", type=int, default=300, help="Number of diffusion steps."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Save a generated sample every N epochs.",
    )
    parser.add_argument(
        "--num-sample-frames",
        type=int,
        default=10,
        help="Number of denoising snapshots per saved sample.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Skip saving dataset and forward-diffusion preview images.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_dataset_preview(dataset, output_path, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    for index in range(min(num_samples, len(dataset))):
        image = (
            dataset[index][0]
            if isinstance(dataset[index], (tuple, list))
            else dataset[index]
        )
        plt.subplot(int(math.ceil(num_samples / cols)), cols, index + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def create_diffusion_schedule(timesteps, device):
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(x_0, t, schedule):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(
        schedule["sqrt_alphas_cumprod"], t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        schedule["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
    )
    noisy_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image, noise


def load_transformed_dataset(data_dir, img_size):
    data_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=data_transform
    )
    test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=data_transform
    )
    return torch.utils.data.ConcatDataset([train, test])


def tensor_to_image(image):
    if len(image.shape) == 4:
        image = image[0]
    image = image.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1) / 2).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def save_forward_diffusion_preview(
    dataloader, schedule, output_path, timesteps, num_images=10
):
    image_batch = next(iter(dataloader))[0]
    stepsize = max(timesteps // num_images, 1)

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    for plot_index, timestep_value in enumerate(range(0, timesteps, stepsize)):
        if plot_index == num_images:
            break
        t = torch.full(
            (image_batch.shape[0],),
            timestep_value,
            dtype=torch.long,
            device=image_batch.device,
        )
        noisy_image, _ = forward_diffusion_sample(image_batch, t, schedule)
        plt.subplot(1, num_images, plot_index + 1)
        plt.imshow(tensor_to_image(noisy_image))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ]
        )
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


def get_loss(model, x_0, t, schedule):
    x_noisy, noise = forward_diffusion_sample(x_0, t, schedule)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(model, x, t, schedule):
    betas_t = get_index_from_list(schedule["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        schedule["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(schedule["sqrt_recip_alphas"], t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(
        schedule["posterior_variance"], t, x.shape
    )

    if int(t[0].item()) == 0:
        return model_mean

    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def save_sample_plot(
    model, schedule, img_size, timesteps, device, output_path, num_images=10
):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    stepsize = max(timesteps // num_images, 1)
    snapshots = []

    for timestep_value in range(timesteps - 1, -1, -1):
        t = torch.full((1,), timestep_value, dtype=torch.long, device=device)
        img = sample_timestep(model, img, t, schedule)
        img = torch.clamp(img, -1.0, 1.0)
        if timestep_value % stepsize == 0 and len(snapshots) < num_images:
            snapshots.append((timestep_value_label(timestep_value), img.detach().cpu()))

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    for index, (label, snapshot) in enumerate(snapshots):
        plt.subplot(1, len(snapshots), index + 1)
        plt.imshow(tensor_to_image(snapshot))
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def timestep_value_label(timestep_value):
    return f"t={timestep_value}"


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    schedule = create_diffusion_schedule(args.timesteps, device)

    raw_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True
    )
    transformed_dataset = load_transformed_dataset(args.data_dir, args.img_size)
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if not args.skip_previews:
        save_dataset_preview(raw_dataset, output_dir / "dataset_samples.png")
        preview_loader = DataLoader(
            transformed_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        save_forward_diffusion_preview(
            preview_loader,
            create_diffusion_schedule(args.timesteps, torch.device("cpu")),
            output_dir / "forward_diffusion.png",
            args.timesteps,
            num_images=args.num_sample_frames,
        )

    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            images = batch[0].to(device)
            t = torch.randint(
                0, args.timesteps, (images.shape[0],), device=device
            ).long()
            loss = get_loss(model, images, t, schedule)
            loss.backward()
            optimizer.step()

            if epoch % args.sample_every == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} | loss {loss.item():.6f}")
                save_sample_plot(
                    model,
                    schedule,
                    args.img_size,
                    args.timesteps,
                    device,
                    output_dir / f"sample_epoch_{epoch:03d}.png",
                    num_images=args.num_sample_frames,
                )

    checkpoint_path = output_dir / "ddpm_cifar10.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
