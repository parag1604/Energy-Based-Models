"""Shared helpers: device selection, the MCMC sampler, and the Bitmoji data pipeline."""

import os
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Sampling from an EBM
# ---------------------------------------------------------------------------
def sample(model, x_init, steps=25, step_size=10, noise_std=0.01,
           grad_clamp=(-0.03, 0.03), x_clamp=(-3, 3)):
    """Short-run, Langevin-style MCMC on the energy E(x) = -model(x).

    Each step moves x downhill on the energy and adds a little Gaussian noise:
        x <- x - step_size * clamp(dE/dx) + noise_std * eps,   eps ~ N(0, I).
    (Exact Langevin dynamics would use noise_std = sqrt(2 * step_size); like most
    practical EBM code, we use much less noise and clip the gradient so that a few
    steps make steady progress without blowing up.)

    The parameters of the model are frozen while sampling: we only want gradients
    with respect to the input x.
    """
    x = x_init.clone().detach()
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x.requires_grad_(True)

    for _ in range(steps):
        energy = -model(x)
        grad, = torch.autograd.grad(energy.sum(), x)
        if grad_clamp is not None:
            grad = grad.clamp(*grad_clamp)
        with torch.no_grad():
            x -= grad * step_size + torch.randn_like(grad) * noise_std
            if x_clamp is not None:
                x.clamp_(*x_clamp)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)
    return x.detach()


def ebm_loss(real_scores, fake_scores, alpha):
    """Contrastive-divergence-style loss. The network outputs a score f(x) = -E(x).

    Minimising  f(fake).mean() - f(real).mean()  lowers the energy of real data and raises
    the energy of the model's own samples; this is a Monte-Carlo estimate of the gradient
    of the negative log-likelihood. The term alpha * (f^2) keeps the scores near zero, so
    the two energies cannot drift off to +/- infinity.
    """
    reg_loss = alpha * (real_scores ** 2 + fake_scores ** 2).mean()
    return fake_scores.mean() - real_scores.mean() + reg_loss


# ---------------------------------------------------------------------------
# Bitmoji super-resolution data
# ---------------------------------------------------------------------------
# Grayscale mean and std of the Bitmoji images (pixels in [0, 1]); compute them for
# your copy of the data with:  python utils.py --stats ./data/bitmoji
MEAN, STD = 0.8095, 0.3260


def denormalize(x):
    return (x * STD + MEAN).clamp(0, 1)


class SuperResolutionPairs(Dataset):
    """Returns (high_res, low_res) pairs built from one ImageFolder.

    high_res: the image resized to hr_size x hr_size.
    low_res : the same image downsampled to lr_size and upsampled back to hr_size,
              i.e. a blurry version at the same resolution as the target.
    """

    def __init__(self, root, hr_size=128, lr_size=32):
        self.images = datasets.ImageFolder(root=root, transform=transforms.Compose([
            transforms.Resize(hr_size),
            transforms.CenterCrop(hr_size),
            transforms.ToTensor(),
        ]))
        self.blur = transforms.Compose([
            transforms.Resize(lr_size, antialias=True),
            transforms.Resize(hr_size, antialias=True),
        ])
        self.normalize = transforms.Normalize((MEAN,), (STD,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        hr, _ = self.images[idx]
        lr = self.blur(hr)
        return self.normalize(hr), self.normalize(lr)


def compute_stats(path, max_images=50_000):
    """Grayscale mean / std of the pixels over (up to) max_images images."""
    dataset = datasets.ImageFolder(root=path, transform=transforms.ToTensor())
    n = min(len(dataset), max_images)
    mean, std = 0.0, 0.0
    for i in range(n):
        im = dataset[i][0]
        mean += (im.mean().item() - mean) / (i + 1)
        std += (im.std().item() - std) / (i + 1)
    return mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', type=str, required=True, help='ImageFolder root')
    args = parser.parse_args()
    print('mean = %.4f, std = %.4f' % compute_stats(args.stats))
