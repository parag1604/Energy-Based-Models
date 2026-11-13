"""version4: diffusion recovery likelihood (Gao et al. 2021) on 32x32 Bitmoji faces.

NEW compared to version2: instead of ONE energy over clean images, whose sampler must travel
from pure noise to a face in a single short chain, learn a SEQUENCE of conditional energies,
one per noise level of a diffusion. Corrupt a face step by step,

    x_{t+1} = sqrt(1 - sigma_{t+1}^2) x_t + sigma_{t+1} eps,      t = 0, ..., T-1,

and learn to undo ONE step at a time. With y_t = sqrt(1 - sigma_{t+1}^2) x_t (x_t rescaled),
the "recovery" distribution of the less noisy image given the noisier one is

    p_theta(y_t | x_{t+1}) ~ exp( -E_theta(y_t, t) - ||x_{t+1} - y_t||^2 / (2 sigma_{t+1}^2) ).

The Gaussian term is known exactly; only E_theta is learned. Because it confines y_t to a ball
of radius ~sigma around x_{t+1}, every conditional is close to unimodal, so a short Langevin
chain with real noise (temperature 1) can sample it: each chain only has to travel a short way.
Training is the usual contrastive rule (eq. 9), applied to the conditional at a random level:
energy down on the true y_t, up on the Langevin sample. Generation starts from Gaussian noise
and runs the T conditional chains in turn, x_T -> x_{T-1} -> ... -> x_0.

All levels share one network (ImageEnergy with a learned per-level bias); the energy is
written E_theta(y, t) = f_theta(y, t) / s_t^2 with the Langevin step s_t = c * sigma_{t+1}, so the
gradient step (s^2 / 2) grad E = grad f / 2 has the same size at every level.

Outputs (results/):
  version4_samples.png    64 samples from Gaussian noise (T levels x K steps)
  version4_chain.png      8 chains: x_T, x_{T-1}, ..., x_0 from left to right

Run:  python version4.py --data-dir /path/to/bitmojis
"""
import os
import time
import math
import argparse

import torch

from models import ImageEnergy
from utils import (get_device, load_bitmojis, train_test_split, to_model, save_grid, pca_frechet,
                   roughness)


def noise_schedule(T, log10_snr_max=2.0, log10_snr_min=-2.0):
    """alpha_bar_t (fraction of signal variance left after t steps) for t = 0..T, with the
    signal-to-noise ratio alpha_bar / (1 - alpha_bar) log-spaced from 10^2 to 10^-2; and the
    per-step noise sigma_t^2 = 1 - alpha_bar_t / alpha_bar_{t-1}."""
    snr = 10 ** torch.linspace(log10_snr_max, log10_snr_min, T)
    alpha_bar = torch.cat([torch.ones(1), snr / (1 + snr)])
    sigma = (1 - alpha_bar[1:] / alpha_bar[:-1]).sqrt()          # sigma[t] = sigma_{t+1}
    return alpha_bar, sigma


class RecoveryEBM:
    """The T conditional energies and their Langevin samplers."""

    def __init__(self, net, T, K, c, amp):
        self.net, self.T, self.K, self.c, self.amp = net, T, K, c, amp
        self.alpha_bar, self.sigma = noise_schedule(T)

    def f(self, y, t):
        with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=self.amp):
            return self.net(y, t).float()

    def recover(self, x_next, t, noise=True):
        """K Langevin steps on p(y_t | x_{t+1}), started at y = x_{t+1}. t: (n,) level indices.
        y <- y - (s^2/2) grad_y [E(y,t) + ||x_{t+1} - y||^2 / (2 sigma^2)] + s * eps."""
        sig = self.sigma.to(x_next.device)[t][:, None, None, None]
        s = self.c * sig
        y = x_next.detach().clone()
        for _ in range(self.K):
            y.requires_grad_(True)
            grad_f, = torch.autograd.grad(self.f(y, t).sum(), y)
            with torch.no_grad():
                # (s^2/2) * grad(f / s^2) = grad f / 2;  (s^2/2) * (y - x) / sigma^2 = (c^2/2) (y - x)
                y = y - 0.5 * grad_f - 0.5 * self.c ** 2 * (y - x_next)
                if noise:
                    y = y + s * torch.randn_like(y)
        return y.detach()

    def training_pair(self, x0):
        """A random level per image; the true y_t (positive) and the noisier x_{t+1}."""
        n, dev = len(x0), x0.device
        t = torch.randint(self.T, (n,), device=dev)
        ab = self.alpha_bar.to(dev)[t][:, None, None, None]
        sig = self.sigma.to(dev)[t][:, None, None, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        y_pos = (1 - sig ** 2).sqrt() * x_t
        x_next = y_pos + sig * torch.randn_like(x0)
        return t, y_pos, x_next

    @torch.no_grad()
    def generate(self, n, size, device, snapshots=False):
        x = torch.randn(n, 3, size, size, device=device)
        snaps = [x]
        for t in reversed(range(self.T)):
            level = torch.full((n,), t, device=device, dtype=torch.long)
            with torch.enable_grad():
                y = self.recover(x, level, noise=t > 0)       # the last level ends noise-free
            x = y / (1 - self.sigma[t] ** 2).sqrt()
            snaps.append(x)
        x = x.clamp(-1, 1)
        return (x, snaps) if snapshots else x


def main():
    parser = argparse.ArgumentParser(description='version4: diffusion recovery likelihood EBM')
    parser.add_argument('--data-dir', type=str, default='./data/bitmojis', help='flat folder of PNGs')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--levels', type=int, default=6, help='T, the number of noise levels')
    parser.add_argument('--steps', type=int, default=30, help='K, Langevin steps per level')
    parser.add_argument('--c', type=float, default=0.2, help='Langevin step s_t = c * sigma_{t+1}')
    parser.add_argument('--iters', type=int, default=8000, help='the learning rate decays to 0 over these')
    parser.add_argument('--minutes', type=float, default=120, help='wall-clock training budget')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='energy-magnitude regulariser (0: here |f| sets the Langevin drift, see README)')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/version4.pt')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    device = get_device()
    torch.manual_seed(args.seed)

    train_imgs, test_imgs = train_test_split(load_bitmojis(args.data_dir, args.size))
    train_imgs = train_imgs.to(device)
    net = ImageEnergy(width=args.width, size=args.size, num_levels=args.levels).to(device)
    ebm = RecoveryEBM(net, args.levels, args.steps, args.c, amp=device.type == 'cuda')
    print('sigma_t:', ' '.join(f'{s:.3f}' for s in ebm.sigma.tolist()))

    if args.eval_only:
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # decay the learning rate to 0: our first run without decay or regulariser trained well
        # for 9,250 iterations and then blew up (energy gaps of 1e9 within 250 iterations)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters)
        t0, it = time.time(), 0
        while it < args.iters and time.time() - t0 < args.minutes * 60:
            x0 = to_model(train_imgs[torch.randint(len(train_imgs), (args.batch_size,), device=device)])
            t, y_pos, x_next = ebm.training_pair(x0)
            y_neg = ebm.recover(x_next, t)
            f_pos, f_neg = ebm.f(torch.cat([y_pos, y_neg]), torch.cat([t, t])).chunk(2)
            gap = f_pos.mean() - f_neg.mean()           # eq. 9 for the conditional (up to the s_t^2 scale)
            loss = gap + args.alpha * (f_pos ** 2 + f_neg ** 2).mean()   # keeps f near 0 (as in utils.ebm_loss)
            if not torch.isfinite(loss) or loss.abs() > 1e3:
                print(f'iter {it}: training diverged (loss {loss.item():.3g}); stopping, keep the last checkpoint')
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            schedule.step()
            if it % 250 == 0:
                print(f'iter {it:6d}  f(pos)-f(neg) {gap.item():8.4f}  mean |f| {f_pos.abs().mean().item():8.4f}  '
                      f'{(time.time() - t0) / 60:5.1f} min', flush=True)
            if it > 0 and it % 2000 == 0:
                save_grid(ebm.generate(64, args.size, device), f'{args.out_dir}/version4_progress.png')
                torch.save(net.state_dict(), args.checkpoint.replace('.pt', f'_iter{it}.pt'))
            it += 1
        print(f'trained {it} iterations in {(time.time() - t0) / 60:.1f} min')
        torch.save(net.state_dict(), args.checkpoint)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    torch.manual_seed(1)
    t0 = time.time()
    x, snaps = ebm.generate(64, args.size, device, snapshots=True)
    print(f'64 samples in {time.time() - t0:.1f} s ({args.levels} levels x {args.steps} steps)')
    save_grid(x, f'{args.out_dir}/version4_samples.png')
    save_grid(torch.stack([s[:8].clamp(-1, 1) for s in snaps], 1).flatten(0, 1),
              f'{args.out_dir}/version4_chain.png', nrow=len(snaps))
    # a crude quantitative check on 2,000 samples (see utils.pca_frechet)
    gen = torch.cat([ebm.generate(500, args.size, device) for _ in range(4)])
    test = to_model(test_imgs.to(device))
    ref = to_model(train_imgs[:20000])
    print(f'pixel-PCA Frechet distance to held-out faces: samples {pca_frechet(gen, test, ref):.1f}, '
          f'2,000 training faces {pca_frechet(ref[:2000], test, ref):.1f};  '
          f'roughness samples/data {roughness(gen) / roughness(test):.2f}')
    print('saved figures to', args.out_dir)


if __name__ == '__main__':
    main()
