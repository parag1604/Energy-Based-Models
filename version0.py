"""version0: an energy-based model on a 2-D toy distribution (no dataset needed).

What this file teaches:
  * an EBM defines p_theta(x) = exp(-E_theta(x)) / Z_theta with a neural network E_theta.
  * Z_theta is intractable, but the gradient of the negative log-likelihood only needs samples
    from the model (lecture eq. 9):
        grad_theta NLL = E_data[grad_theta E(x)] - E_{x' ~ p_theta}[grad_theta E(x')].
  * the model samples x' come from SHORT-RUN Langevin dynamics (lecture eq. 10), started from
    uniform noise at every iteration:
        x_{k+1} = x_k - (eta/2) grad_x E(x_k) + sqrt(eta) eps_k.
    Treating x' as constants gives the contrastive-divergence-style loss
        L = mean E(real) - mean E(fake) + alpha * mean(E(real)^2 + E(fake)^2).
  * the data are 8 Gaussian blobs on a circle, generated on the fly.

The script writes results/version0_toy.png: the true density, the learned density
exp(-E_theta) (normalised numerically on the plotting grid, which is only possible in 2-D),
and fresh Langevin samples from noise.

Run:  python version0.py
"""
import os
import time
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ToyEnergy
from utils import langevin, ebm_loss, sample_ring, ring_log_density

LIM = 3.0


def uniform_noise(n):
    return (torch.rand(n, 2) * 2 - 1) * LIM


def grid_density(energy, n=200):
    xs = torch.linspace(-LIM, LIM, n)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing='xy'), dim=-1).view(-1, 2)
    with torch.no_grad():
        logp = -energy(grid)
    logp = logp - torch.logsumexp(logp, 0)   # normalise on the grid
    return xs, logp.view(n, n).exp()


def plot_toy(energy, short, long, short_title, long_title, path):
    xs, p_model = grid_density(energy)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing='xy'), dim=-1).view(-1, 2)
    p_true = ring_log_density(grid).view(len(xs), len(xs)).exp()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    axes = axes.ravel()
    axes[0].contourf(xs, xs, p_true, levels=30, cmap='viridis')
    axes[0].set_title('data density $p_{data}$')
    axes[1].contourf(xs, xs, p_model, levels=30, cmap='viridis')
    axes[1].set_title(r'learned $p_\theta \propto e^{-E_\theta(x)}$ (grid-normalised)')
    data = sample_ring(len(short))
    for ax, s, t in ((axes[2], short, short_title), (axes[3], long, long_title)):
        ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.25, c='tab:blue', label='data')
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.6, c='tab:orange', label='Langevin samples')
        ax.legend(loc='upper right', markerscale=4, fontsize=8)
        ax.set_title(t)
    for ax in axes:
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def mode_coverage(samples, radius=2.0, num_modes=8, tol=0.5):
    """Fraction of samples within tol of a mode, and the share of each mode (ideal: 1/8 each)."""
    import math
    angle = 2 * math.pi * torch.arange(num_modes) / num_modes
    centers = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    d = torch.cdist(samples, centers)
    near, which = d.min(1)
    ok = near < tol
    counts = torch.bincount(which[ok], minlength=num_modes).float()
    return ok.float().mean().item(), (counts / counts.sum().clamp(min=1)).tolist()


def density_mode_masses(energy, radius=2.0, num_modes=8):
    """Probability mass that the learned density exp(-E) (normalised on the grid) puts near each
    mode, i.e. the mode weights of the model itself (ideal: 1/8 = 0.125 each)."""
    import math
    xs, p = grid_density(energy)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing='xy'), dim=-1).view(-1, 2)
    angle = 2 * math.pi * torch.arange(num_modes) / num_modes
    centers = radius * torch.stack([angle.cos(), angle.sin()], dim=1)
    which = torch.cdist(grid, centers).argmin(1)
    return [p.view(-1)[which == k].sum().item() for k in range(num_modes)]


def report(energy, sampler, long_steps, out_path, label):
    """Short-run samples (the K steps used in training) and long-run samples (long_steps),
    their mode statistics, and the figure."""
    torch.manual_seed(1)
    short = langevin(energy, uniform_noise(2000), **sampler)
    long = langevin(energy, uniform_noise(2000), **{**sampler, 'steps': long_steps})
    for name, s in (('short-run', short), ('long-run', long)):
        frac, shares = mode_coverage(s)
        print(f'{name:9s} samples: {100 * frac:5.1f}% within 0.5 of a mode; '
              f'mode shares {[round(x, 3) for x in shares]}')
    masses = density_mode_masses(energy)
    tv = 0.5 * sum(abs(m / sum(masses) - 1 / len(masses)) for m in masses)
    print('learned density exp(-E), mass per mode', [round(m, 3) for m in masses],
          f'(total variation from the true 1/8 each: {tv:.2f})')
    plot_toy(energy, short, long, f'{label}: {sampler["steps"]} Langevin steps from noise',
             f'{long_steps} Langevin steps from noise', out_path)
    print('saved', out_path)


def main():
    parser = argparse.ArgumentParser(description='version0: 2-D toy EBM with short-run Langevin')
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=100, help='Langevin steps K per iteration')
    parser.add_argument('--eta', type=float, default=0.005, help='Langevin step size eta (eq. 10)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.01, help='energy-magnitude regulariser')
    parser.add_argument('--long-steps', type=int, default=2000,
                        help='steps for the long-run chains drawn after training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    energy = ToyEnergy()   # 2-D points are tiny: the CPU is fast enough
    optimizer = torch.optim.Adam(energy.parameters(), lr=args.lr, betas=(0.0, 0.999))
    # exact Langevin at temperature 1: step eta/2 on the gradient, noise sqrt(eta)
    sampler = dict(steps=args.steps, step_size=args.eta / 2, noise_std=args.eta ** 0.5,
                   x_range=(-LIM, LIM))

    t0 = time.time()
    for it in range(args.iters):
        real = sample_ring(args.batch_size)
        fake = langevin(energy, uniform_noise(args.batch_size), **sampler)
        loss, cd = ebm_loss(energy(real), energy(fake), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 250 == 0:
            print(f'iter {it:5d}  loss {loss.item():8.3f}  E(real)-E(fake) {cd.item():8.3f}')

    print(f'trained in {time.time() - t0:.0f}s')
    report(energy, sampler, args.long_steps, f'{args.out_dir}/version0_toy.png', 'short-run')

if __name__ == '__main__':
    main()
