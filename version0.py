"""version0: an energy-based model on a 2-D toy distribution (no dataset needed).

What this file teaches:
  * an EBM defines p(x) = exp(-E(x)) / Z with a neural network for E. Here the network
    outputs a score f(x) = -E(x): high score = low energy = likely.
  * Z is intractable, but the gradient of the log-likelihood only needs samples from the
    model:  grad log p(x) = grad f(x) - E_{x' ~ p}[grad f(x')].
    So each training step
      1. draws "fake" samples x' with a few steps of Langevin-style MCMC
         (walk downhill on E, plus noise), started from uniform noise, and
      2. minimises  f(fake).mean() - f(real).mean() + alpha * (f(real)^2 + f(fake)^2).mean().
  * the data are 8 Gaussian blobs on a circle, generated on the fly.

The script writes results/toy_energy.png: the learned energy landscape with the data,
and fresh MCMC samples drawn from noise.

Run:  python version0.py --iters 4000
"""
import os
import math
import argparse

import torch
import matplotlib.pyplot as plt

from models import ToyNetwork
from utils import sample, ebm_loss


def sample_ring(n, num_modes=8, radius=2.0, std=0.15):
    """n points from a mixture of num_modes Gaussians placed evenly on a circle."""
    k = torch.randint(num_modes, (n,))
    angle = 2 * math.pi * k / num_modes
    centers = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    return centers + std * torch.randn(n, 2)


def mcmc(model, n, device, steps):
    x_init = torch.rand(n, 2, device=device) * 6 - 3  # uniform on [-3, 3]^2
    return sample(model, x_init, steps=steps, step_size=0.05, noise_std=0.05,
                  grad_clamp=(-2, 2), x_clamp=(-3, 3))


def plot(model, device, steps, path):
    xs = torch.linspace(-3, 3, 200)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing='xy'), dim=-1).view(-1, 2).to(device)
    with torch.no_grad():
        energy = -model(grid).view(200, 200).cpu()
    data = sample_ring(1000)
    samples = mcmc(model, 1000, device, steps).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    im = axes[0].contourf(xs, xs, energy, levels=30, cmap='viridis')
    fig.colorbar(im, ax=axes[0], label='energy E(x)')
    axes[0].scatter(data[:, 0], data[:, 1], s=2, c='white', alpha=0.5)
    axes[0].set_title('Learned energy (white: data)')
    axes[1].scatter(data[:, 0], data[:, 1], s=2, alpha=0.3, label='data')
    axes[1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.6, label='MCMC samples')
    axes[1].set_title(f'{steps} MCMC steps from uniform noise')
    axes[1].legend(loc='upper right')
    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='version0: 2-D toy EBM')
    parser.add_argument('--iters', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=60, help='MCMC steps per sample')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.1, help='score regulariser')
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # 2-D points are tiny tensors: the CPU is faster here than a GPU
    device = torch.device('cpu')
    torch.manual_seed(0)

    model = ToyNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.0, 0.999))

    for it in range(args.iters):
        real = sample_ring(args.batch_size).to(device)
        fake = mcmc(model, args.batch_size, device, args.steps)

        loss = ebm_loss(model(real), model(fake), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 200 == 0:
            print(f'Iter {it}, Loss: {loss.item():.4f}')

    plot(model, device, args.steps, f'{args.out_dir}/toy_energy.png')
    print(f'Saved {args.out_dir}/toy_energy.png')


if __name__ == '__main__':
    main()
