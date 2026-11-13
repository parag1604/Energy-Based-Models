"""version2: an image EBM on Bitmoji faces that samples from noise, and what its energy can do.

NEW compared to version1:
  * a position-aware energy (models.ImageEnergy, laid out like a DCGAN discriminator): stride-2 convs and
    a final fully connected layer instead of global average pooling. version1's pooled network
    cannot easily tell where a feature is, and its chains from noise only reach blobs.
  * back to SHORT-RUN chains from noise (version0's recipe, Nijkamp et al. 2019): every
    iteration starts fresh chains from uniform noise and runs K = 100 Langevin steps. For images
    this is what we want: a model that turns noise into a face in K steps. (--buffer brings back
    version1's replay buffer for comparison; --pooled brings back version1's network.)
  * bfloat16 autocast to halve the cost of the Langevin steps (the bulk of the compute).
  * a second idea: an energy is a PRIOR you can reuse for other tasks without retraining.
      - inpainting: fix the observed pixels, run Langevin on the missing ones only;
      - denoising: start Langevin from the noisy image (at low temperature) so that it slides
        down to a nearby low-energy, i.e. clean-looking, image.

Outputs (results/):
  version2_samples.png     64 fresh chains from uniform noise, K = 100 steps (as in training)
  version2_chain.png       how 8 fresh chains evolve, left to right (beyond K too)
  version2_inpaint.png     rows: masked input | EBM inpainting | ground truth
  version2_denoise.png     rows: noisy input | EBM denoised | ground truth
  and a sample-quality score (utils.pca_frechet) and PSNRs for the restoration demos on 2000
  held-out images.

Run:  python version2.py --data-dir /path/to/bitmojis
"""
import os
import time
import argparse

import torch

from models import ConvEnergy, ImageEnergy
from utils import (langevin, ebm_loss, ReplayBuffer, get_device, load_bitmojis,
                   train_test_split, to_model, save_grid, psnr, roughness, pca_frechet)


def make_energy_fn(energy, amp):
    def energy_fn(x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            return energy(x).float()
    return energy_fn


def inpaint(energy_fn, x, mask, init, steps, sampler):
    """Langevin on the unknown pixels (mask == 0) only; the known pixels are reset after every
    10 steps. Unknown pixels start from `init` (here: uniform noise, as in training)."""
    z = torch.where(mask.bool(), x, init)
    for _ in range(steps // 10):
        z = langevin(energy_fn, z, **{**sampler, 'steps': 10})
        z = torch.where(mask.bool(), x, z)
    return z


def train(args, device, train_imgs, energy, energy_fn, sampler):
    optimizer = torch.optim.Adam(energy.parameters(), lr=args.lr, betas=(0.5, 0.999))
    noise = lambda n: torch.rand(n, 3, args.size, args.size, device=device) * 2 - 1
    buffer = ReplayBuffer(noise, size=args.buffer_size, restart=0.05) if args.buffer else None
    t0 = time.time()
    it = 0
    while it < args.iters and time.time() - t0 < args.minutes * 60:
        real = to_model(train_imgs[torch.randint(len(train_imgs), (args.batch_size,), device=device)])
        real = real + args.data_noise * torch.randn_like(real)
        if buffer is not None:                      # version1: persistent chains
            x0, idx = buffer.sample(args.batch_size)
            fake = langevin(energy_fn, x0, **sampler)
            buffer.update(fake, idx)
        else:                                       # short-run: fresh chains from noise
            fake = langevin(energy_fn, noise(args.batch_size), **sampler)
        loss, cd = ebm_loss(*energy_fn(torch.cat([real, fake])).chunk(2), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), args.clip)
        optimizer.step()
        if it % 250 == 0:
            # collapse detector: image EBMs can drift into saturated high-frequency noise. Compare
            # the pixel-to-pixel variation of the negatives with that of real images (about 1 = healthy)
            hf = roughness(buffer.buffer[:1000] if buffer is not None else fake) / roughness(real)
            print(f'iter {it:6d}  loss {loss.item():8.4f}  E(real)-E(fake) {cd.item():8.4f}  '
                  f'samples/data roughness {hf:5.2f}  {(time.time() - t0) / 60:5.1f} min', flush=True)
            save_grid(fake[:64], f'{args.out_dir}/version2_progress.png')
        if it > 0 and it % args.save_every == 0:
            torch.save(energy.state_dict(), args.checkpoint.replace('.pt', f'_iter{it}.pt'))
        it += 1
    print(f'trained {it} iterations in {(time.time() - t0) / 60:.1f} min')


def main():
    parser = argparse.ArgumentParser(description='version2: unconditional Bitmoji EBM')
    parser.add_argument('--data-dir', type=str, default='./data/bitmojis', help='flat folder of PNGs')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--minutes', type=float, default=45, help='wall-clock training budget')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--buffer', action='store_true', help="version1's replay buffer instead of short-run chains")
    parser.add_argument('--pooled', action='store_true', help="version1's network (global average pooling)")
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step-size', type=float, default=10.0)
    parser.add_argument('--noise', type=float, default=0.005)
    parser.add_argument('--grad-clip', type=float, default=0.01)
    parser.add_argument('--data-noise', type=float, default=0.005, help='noise added to training images')
    parser.add_argument('--save-every', type=int, default=1000, help='iterations between checkpoints')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.2, help='denoising demo: noise std ([-1, 1] scale)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/version2.pt')
    parser.add_argument('--eval-only', action='store_true', help='load --checkpoint, skip training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    device = get_device()
    torch.manual_seed(args.seed)

    train_imgs, test_imgs = train_test_split(load_bitmojis(args.data_dir, args.size))
    train_imgs = train_imgs.to(device)
    energy = (ConvEnergy(width=args.width) if args.pooled else
              ImageEnergy(width=args.width, size=args.size)).to(device).to(memory_format=torch.channels_last)
    energy_fn = make_energy_fn(energy, amp=not args.no_amp and device.type == 'cuda')
    sampler = dict(steps=args.steps, step_size=args.step_size, noise_std=args.noise,
                   grad_clip=args.grad_clip, x_range=(-1, 1))

    if args.eval_only:
        energy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        train(args, device, train_imgs, energy, energy_fn, sampler)
        torch.save(energy.state_dict(), args.checkpoint)
    energy.eval()
    for p in energy.parameters():
        p.requires_grad_(False)

    # fresh chains from noise, recording snapshots along the way (also beyond the K of training)
    torch.manual_seed(1)
    z = torch.rand(64, 3, args.size, args.size, device=device) * 2 - 1
    snaps, done = [z[:8]], 0
    for k in sorted({10, 20, 40, 60, 100, 200, 400, args.steps}):
        z = langevin(energy_fn, z, **{**sampler, 'steps': k - done})
        done = k
        snaps.append(z[:8])
        if k == args.steps:
            save_grid(z, f'{args.out_dir}/version2_samples.png')
    save_grid(torch.stack(snaps, 1).flatten(0, 1), f'{args.out_dir}/version2_chain.png', nrow=len(snaps))

    # a crude quantitative check on 2,000 fresh K-step samples (see utils.pca_frechet)
    noise = lambda n: torch.rand(n, 3, args.size, args.size, device=device) * 2 - 1
    gen = torch.cat([langevin(energy_fn, noise(500), **sampler) for _ in range(4)])
    test = to_model(test_imgs.to(device))
    ref = to_model(train_imgs[:20000])
    print(f'pixel-PCA Frechet distance to held-out faces: samples {pca_frechet(gen, test, ref):.1f}, '
          f'2,000 training faces {pca_frechet(ref[:2000], test, ref):.1f};  '
          f'roughness samples/data {roughness(gen) / roughness(test):.2f}')

    # restoration demos on held-out images
    x = to_model(test_imgs.to(device))
    mask = torch.ones_like(x[:, :1])
    mask[:, :, args.size // 2:, :] = 0                      # the lower half is missing
    # baseline: fill the hole with the average training image (per pixel)
    mean_img = to_model(train_imgs[:20000]).mean(0, keepdim=True)
    mean_fill = torch.where(mask.bool(), x, mean_img)
    rec, den, noisy = [], [], []
    for xb, mb in zip(x.split(500), mask.split(500)):
        rec.append(inpaint(energy_fn, xb, mb, torch.rand_like(xb) * 2 - 1, args.steps, sampler))
        nb = (xb + args.sigma * torch.randn_like(xb)).clamp(-1, 1)
        noisy.append(nb)
        # denoising: plain short-run descent from the noisy image (no added noise)
        den.append(langevin(energy_fn, nb, **{**sampler, 'steps': 20, 'noise_std': 0.0}))
    rec, den, noisy = torch.cat(rec), torch.cat(den), torch.cat(noisy)
    print(f'inpainting (lower half missing), PSNR of the whole image: mean-image fill '
          f'{psnr(mean_fill, x).mean():.2f} dB, EBM {psnr(rec, x).mean():.2f} dB')
    print(f'denoising sigma={args.sigma}: noisy {psnr(noisy, x).mean():.2f} dB -> EBM {psnr(den, x).mean():.2f} dB')
    n = 8
    save_grid(torch.cat([torch.where(mask[:n].bool(), x[:n], torch.zeros_like(x[:n])), rec[:n], x[:n]]),
              f'{args.out_dir}/version2_inpaint.png', nrow=n)
    save_grid(torch.cat([noisy[:n], den[:n], x[:n]]), f'{args.out_dir}/version2_denoise.png', nrow=n)
    print('saved figures to', args.out_dir)


if __name__ == '__main__':
    main()
