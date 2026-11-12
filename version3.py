"""version3: conditional super-resolution with an EBM, E_theta(x_hr | x_lr).

The idea is Parag's (the original course code, `main_SR.py`): treat super-resolution as
"walk downhill on an energy, starting from the blurry upsampled image". His model scored the
high-resolution image alone, E(x_hr); the low-resolution input entered only as the starting
point of the sampler, so nothing tied the result to the input. This version keeps the idea and
makes the model CONDITIONAL:

    p_theta(x_hr | x_lr) = exp(-E_theta(x_hr, x_lr)) / Z_theta(x_lr),

with x_lr (bicubic-upsampled) stacked with the candidate x_hr as the network input. The
training rule is eq. (9) with every expectation conditioned on x_lr:
    grad = E_data[grad E(x_hr | x_lr)] - E_{x' ~ p_theta(.|x_lr)}[grad E(x' | x_lr)],
and the negatives x' come from K short-run Langevin steps started at the bicubic upsample
U(x_lr). At test time the same K-step chain is the super-resolver.

A known degradation D (here: antialiased bicubic downsampling) also gives a likelihood, so we
can optionally add a data-consistency term at test time (Bayes: posterior = likelihood x prior):
    E_total(x) = E_theta(x | x_lr) + (lam / 2) * || D(x) - x_lr ||^2.

Evaluation on 2000 held-out faces: PSNR against the true high-resolution image, and the
consistency PSNR(D(x_hat), x_lr) (how well the output explains the input), for
bicubic, the consistency term alone (no EBM: classic back-projection), the EBM chain, the EBM chain + consistency term, and (--unconditional) Parag's
original unconditional formulation trained the same way.

Run:  python version3.py --data-dir /path/to/bitmojis                  # conditional (default)
      python version3.py --data-dir /path/to/bitmojis --unconditional  # the original idea
"""
import os
import time
import argparse

import torch

from models import ConditionalEnergy
from utils import (langevin, ebm_loss, get_device, load_bitmojis, train_test_split, to_model,
                   downsample, upsample, psnr, save_grid)


def main():
    parser = argparse.ArgumentParser(description='version3: conditional EBM super-resolution')
    parser.add_argument('--data-dir', type=str, default='./data/bitmojis', help='flat folder of PNGs')
    parser.add_argument('--hr', type=int, default=64, help='high-resolution size')
    parser.add_argument('--lr-size', type=int, default=16, help='low-resolution size')
    parser.add_argument('--unconditional', action='store_true',
                        help="Parag's original formulation: E(x_hr) only, x_lr is just the start")
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--minutes', type=float, default=45, help='wall-clock training budget')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--step-size', type=float, default=5.0)
    parser.add_argument('--noise', type=float, default=0.005)
    parser.add_argument('--grad-clip', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=50.0, help='weight of the consistency term at test time')
    parser.add_argument('--train-lam', type=float, default=0.0,
                        help='weight of the consistency term inside the training chains (0: EBM alone)')
    parser.add_argument('--tag', type=str, default=None, help='name for the outputs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    tag = args.tag or ('version3_uncond' if args.unconditional else 'version3')
    args.checkpoint = args.checkpoint or f'./checkpoints/{tag}.pt'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    device = get_device()
    torch.manual_seed(args.seed)
    amp = device.type == 'cuda'

    train_imgs, test_imgs = train_test_split(load_bitmojis(args.data_dir, args.hr))
    train_imgs = train_imgs.to(device)
    energy = ConditionalEnergy(width=args.width).to(device).to(memory_format=torch.channels_last)

    def energy_fn(x, cond):
        if args.unconditional:
            cond = torch.zeros_like(cond)   # the network never sees x_lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            return energy(x, cond).float()

    sampler = dict(steps=args.steps, step_size=args.step_size, noise_std=args.noise,
                   grad_clip=args.grad_clip, x_range=(-1, 1))

    def super_resolve(x_lr, lam=0.0, noise=True, use_energy=True):
        cond = upsample(x_lr, args.hr)
        def total(x):
            e = energy_fn(x, cond) if use_energy else torch.zeros(len(x), device=x.device)
            if lam > 0:
                e = e + 0.5 * lam * ((downsample(x, args.lr_size) - x_lr) ** 2).flatten(1).sum(1)
            return e
        return langevin(total, cond, **{**sampler, 'noise_std': args.noise if noise else 0.0})

    if args.eval_only:
        energy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        optimizer = torch.optim.Adam(energy.parameters(), lr=args.lr, betas=(0.0, 0.999))
        t0, it = time.time(), 0
        while it < args.iters and time.time() - t0 < args.minutes * 60:
            x_hr = to_model(train_imgs[torch.randint(len(train_imgs), (args.batch_size,), device=device)])
            x_lr = downsample(x_hr, args.lr_size)
            fake = super_resolve(x_lr, lam=args.train_lam)   # negatives: K Langevin steps from U(x_lr)
            cond = upsample(x_lr, args.hr)
            real = x_hr + 0.005 * torch.randn_like(x_hr)
            e = energy_fn(torch.cat([real, fake]), torch.cat([cond, cond]))
            loss, cd = ebm_loss(*e.chunk(2), args.alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy.parameters(), args.clip)
            optimizer.step()
            if it % 250 == 0:
                with torch.no_grad():
                    gain = (psnr(fake, x_hr) - psnr(cond, x_hr)).mean().item()
                print(f'iter {it:6d}  loss {loss.item():8.4f}  E(real)-E(fake) {cd.item():8.4f}  '
                      f'PSNR gain over bicubic {gain:+.2f} dB  {(time.time() - t0) / 60:5.1f} min', flush=True)
            it += 1
        print(f'trained {it} iterations in {(time.time() - t0) / 60:.1f} min')
        torch.save(energy.state_dict(), args.checkpoint)
    energy.eval()
    for p in energy.parameters():
        p.requires_grad_(False)

    # ---- evaluation on held-out faces -------------------------------------------------------
    torch.manual_seed(1)
    x_hr = to_model(test_imgs.to(device))
    x_lr = downsample(x_hr, args.lr_size)
    cons = f'EBM + consistency (lam={args.lam:g})'
    outs = {'bicubic': [], 'consistency only (no EBM)': [], 'EBM': [], 'EBM, no noise': [], cons: []}
    for lb in x_lr.split(250):
        outs['bicubic'].append(upsample(lb, args.hr))
        outs['EBM'].append(super_resolve(lb))
        outs['EBM, no noise'].append(super_resolve(lb, noise=False))
        outs['consistency only (no EBM)'].append(super_resolve(lb, lam=args.lam, use_energy=False))
        outs[cons].append(super_resolve(lb, lam=args.lam))
    def sharpness(v):
        """Mean absolute difference between neighbouring pixels: low = blurry."""
        return ((v[..., 1:, :] - v[..., :-1, :]).abs().mean() + (v[..., :, 1:] - v[..., :, :-1]).abs().mean()) / 2

    print(f'{"method":32s} {"PSNR vs HR":>11s} {"PSNR(D(x), x_lr)":>17s} {"sharpness":>10s}')
    for name, v in outs.items():
        v = torch.cat(v)
        outs[name] = v
        print(f'{name:32s} {psnr(v, x_hr).mean():8.2f} dB {psnr(downsample(v, args.lr_size), x_lr).mean():14.2f} dB'
              f' {sharpness(v):10.4f}')
    print(f'{"ground truth":32s} {"":11s} {"exact":>17s} {sharpness(x_hr):10.4f}')

    n = 8
    rows = [outs['bicubic'][:n], outs['EBM'][:n], outs[cons][:n], x_hr[:n]]
    save_grid(torch.cat(rows), f'{args.out_dir}/{tag}_sr.png', nrow=n)
    print('saved', f'{args.out_dir}/{tag}_sr.png  (rows: bicubic | EBM | EBM + consistency | ground truth)')


if __name__ == '__main__':
    main()
