"""Shared helpers: the Langevin sampler, the EBM loss, the replay buffer, toy data and the
Bitmoji image pipeline.

Notation follows the lecture (https://parag1604.github.io/notes/Machine-Learning/Energy-Based-Models):
the networks output an energy E_theta(x) (low = likely), p_theta(x) = exp(-E_theta(x)) / Z_theta.
"""

import os
import math
import argparse
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Sampling: Langevin dynamics
# ---------------------------------------------------------------------------
def langevin(energy_fn, x_init, steps, step_size, noise_std, grad_clip=None, x_range=None):
    """Short-run Langevin dynamics on the energy  E(x) = energy_fn(x)  (one value per sample).

        x_{k+1} = x_k - step_size * grad_x E(x_k) + noise_std * eps_k,   eps_k ~ N(0, I).

    With step_size = eta / 2 and noise_std = sqrt(eta * tau) this is eq. (10) of the lecture
    run at temperature tau, i.e. it samples p(x) ~ exp(-E(x) / tau). The "exact" choice is
    tau = 1 (noise_std = sqrt(2 * step_size)). Image EBMs (Du & Mordatch 2019, Nijkamp et
    al. 2019) use much less noise than that, i.e. a temperature tau = noise_std^2 /
    (2 * step_size) far below 1, and they clip the gradient: the chain is then a learned,
    slightly noisy optimiser rather than an exact sampler.

    Only x receives gradients; the parameters of the energy are untouched (gradients are
    taken with torch.autograd.grad w.r.t. x only), and x is returned detached.
    """
    x = x_init.detach().clone()
    for _ in range(steps):
        x.requires_grad_(True)
        grad, = torch.autograd.grad(energy_fn(x).sum(), x)
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        with torch.no_grad():
            x = x - step_size * grad + noise_std * torch.randn_like(x)
            if x_range is not None:
                x = x.clamp(*x_range)
    return x.detach()


def ebm_loss(e_real, e_fake, alpha):
    """Monte-Carlo estimate of the negative log-likelihood gradient, eq. (9):

        L = E_data[E(x)] - E_model[E(x')] + alpha * E[E(x)^2 + E(x')^2].

    Its theta-gradient is E_data[grad E] - E_model[grad E] (+ the regulariser): energy down on
    the data, up on the samples. The fake samples are treated as constants (they are
    detached), which is the contrastive-divergence approximation. The alpha term keeps the
    energies near zero so that the two terms cannot drift off together to +/- infinity.
    Returns (loss, contrastive part) so the contrastive gap can be logged.
    """
    cd = e_real.mean() - e_fake.mean()
    reg = alpha * (e_real ** 2 + e_fake ** 2).mean()
    return cd + reg, cd


class ReplayBuffer:
    """Persistent Langevin chains (PCD-style), as in Du & Mordatch (2019).

    The buffer stores the end points of earlier chains. Each training step starts most new
    chains where old ones stopped, so the chains effectively run for many more steps than
    the K steps per iteration; a fraction `restart` of chains is restarted from fresh noise so
    that new modes can still be discovered.
    """

    def __init__(self, init_fn, size, restart=0.05):
        self.init_fn = init_fn          # init_fn(n) -> n fresh noise samples
        self.buffer = init_fn(size)
        self.restart = restart

    def sample(self, n):
        idx = torch.randint(len(self.buffer), (n,))
        x = self.buffer[idx].clone()
        fresh = torch.rand(n) < self.restart
        if fresh.any():
            x[fresh] = self.init_fn(int(fresh.sum())).to(x.device)
        return x, idx

    def update(self, x, idx):
        self.buffer[idx] = x.detach().to(self.buffer.device)


# ---------------------------------------------------------------------------
# Toy data: 8 Gaussians on a ring
# ---------------------------------------------------------------------------
def sample_ring(n, num_modes=8, radius=2.0, std=0.15):
    """n points from an equal-weight mixture of num_modes Gaussians on a circle."""
    k = torch.randint(num_modes, (n,))
    angle = 2 * math.pi * k / num_modes
    centers = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    return centers + std * torch.randn(n, 2)


def ring_log_density(x, num_modes=8, radius=2.0, std=0.15):
    """Exact log-density of the ring mixture (used only for plotting the ground truth)."""
    angle = 2 * math.pi * torch.arange(num_modes) / num_modes
    centers = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    d2 = ((x[:, None, :] - centers[None]) ** 2).sum(-1)
    return torch.logsumexp(-d2 / (2 * std ** 2), dim=1) - math.log(num_modes * 2 * math.pi * std ** 2)


# ---------------------------------------------------------------------------
# Bitmoji images
# ---------------------------------------------------------------------------
def _load_one(args):
    from PIL import Image
    path, size = args
    im = Image.open(path).convert('RGB')
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC, reducing_gap=None)
    return np.asarray(im, dtype=np.uint8)


def load_bitmojis(root, size, cache_dir='./data', workers=16, max_images=None):
    """All images of a flat folder of PNGs as a uint8 tensor (N, 3, size, size).

    The folder is read once (in parallel) and cached as data/bitmojis_<size>.npy, so later
    runs start instantly. Files are sorted by name, so the split below is deterministic.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'bitmojis_{size}.npy')
    if os.path.exists(cache):
        arr = np.load(cache)
    else:
        files = sorted(f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        with Pool(workers) as pool:
            arr = np.stack(pool.map(_load_one, [(os.path.join(root, f), size) for f in files],
                                    chunksize=256))
        np.save(cache, arr)
    if max_images is not None:
        arr = arr[:max_images]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def train_test_split(images, n_test=2000):
    """The last n_test images (by file name) are held out for evaluation."""
    return images[:-n_test], images[-n_test:]


def to_model(x_uint8):
    """uint8 [0, 255] -> float [-1, 1]."""
    return x_uint8.float() / 127.5 - 1.0


def to_image(x):
    """float [-1, 1] -> float [0, 1] for plotting / saving."""
    return ((x + 1) / 2).clamp(0, 1)


def downsample(x, size):
    """The degradation operator D of the super-resolution problem: antialiased bicubic."""
    return F.interpolate(x, size=(size, size), mode='bicubic', antialias=True, align_corners=False)


def upsample(x, size):
    """Bicubic upsampling U (the baseline super-resolver)."""
    return F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).clamp(-1, 1)


def psnr(x, y):
    """PSNR in dB per image, for images in [-1, 1] (peak-to-peak 2)."""
    mse = ((x - y) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(4.0 / mse)


def save_grid(x, path, nrow=8, padding=2):
    """Save a batch of [-1, 1] images as one PNG grid."""
    from torchvision.utils import save_image
    save_image(to_image(x), path, nrow=nrow, padding=padding, pad_value=1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-cache the Bitmoji folder at a resolution')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--size', type=int, default=32)
    args = parser.parse_args()
    imgs = load_bitmojis(args.data_dir, args.size)
    print(imgs.shape, imgs.dtype, 'mean pixel %.3f' % (imgs.float().mean() / 255))
