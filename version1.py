"""version1: persistent chains (a replay buffer) plus the usual stabilisers.

NEW compared to version0:
  * a REPLAY BUFFER of Langevin chains (persistent contrastive divergence, PCD; Tieleman 2008,
    Du & Mordatch 2019). Instead of starting every chain from noise, 95% of the chains start
    where earlier chains stopped, and only 5% restart from noise. Each iteration still runs
    only K Langevin steps, but a chain that is re-used many times has effectively run for
    hundreds or thousands of steps, so its samples are much closer to p_theta itself.
  * stabilisers: the energy-magnitude regulariser alpha * E^2 (as in version0), clipping of the
    parameter-gradient norm, and (optionally) spectral normalisation of every layer.

Two modes:
  --data toy      the 8-Gaussian ring (CPU, ~3 min), compared with version0's short-run model;
  --data bitmoji  32x32 Bitmoji faces with a small ConvNet energy (a first, short image run;
                  version2 is the full image model).

Run:  python version1.py --data toy
      python version1.py --data bitmoji --data-dir /path/to/bitmojis --iters 4000
"""
import os
import time
import argparse

import torch

from models import ToyEnergy, ConvEnergy
from utils import (langevin, ebm_loss, ReplayBuffer, sample_ring, get_device,
                   load_bitmojis, train_test_split, to_model, save_grid)
from version0 import uniform_noise, report, LIM


def train_toy(args):
    torch.manual_seed(args.seed)
    energy = ToyEnergy(spectral_norm=args.spectral_norm)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3, betas=(0.0, 0.999))
    eta = 0.005
    sampler = dict(steps=args.steps, step_size=eta / 2, noise_std=eta ** 0.5, x_range=(-LIM, LIM))
    buffer = ReplayBuffer(uniform_noise, size=10000, restart=0.05)
    # decaying the learning rate to 0 lets the wells settle: with a constant rate, SGD noise keeps
    # changing their depths by a few nats, and the mode weights exp(-E) swing with it
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters)

    t0 = time.time()
    for it in range(args.iters):
        real = sample_ring(256)
        x0, idx = buffer.sample(256)
        fake = langevin(energy, x0, **sampler)
        buffer.update(fake, idx)
        loss, cd = ebm_loss(energy(real), energy(fake), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), args.clip)
        optimizer.step()
        schedule.step()
        if it % 250 == 0:
            print(f'iter {it:5d}  loss {loss.item():8.3f}  E(real)-E(fake) {cd.item():8.3f}')
    print(f'trained in {time.time() - t0:.0f}s')
    os.makedirs(args.out_dir, exist_ok=True)
    # NB: after training we sample from NOISE, not from the buffer, to compare with version0
    report(energy, {**sampler, 'steps': args.steps}, 2000, f'{args.out_dir}/version1_toy.png' if args.tag == '' else f'{args.out_dir}/version1_toy_{args.tag}.png',
           f'PCD (K={args.steps})')


def train_images(args):
    device = get_device()
    torch.manual_seed(args.seed)
    train, _ = train_test_split(load_bitmojis(args.data_dir, args.size))
    train = train.to(device)   # uint8 on the GPU: 127k x 3 x 32 x 32 = 0.4 GB
    print('training images', tuple(train.shape))

    energy = ConvEnergy(width=args.width, spectral_norm=args.spectral_norm).to(device)
    optimizer = torch.optim.Adam(energy.parameters(), lr=args.lr, betas=(0.0, 0.999))
    noise = lambda n: torch.rand(n, 3, args.size, args.size, device=device) * 2 - 1
    buffer = ReplayBuffer(noise, size=args.buffer_size, restart=0.05)
    sampler = dict(steps=args.steps, step_size=args.step_size, noise_std=args.noise,
                   grad_clip=args.grad_clip, x_range=(-1, 1))

    t0 = time.time()
    for it in range(args.iters):
        real = to_model(train[torch.randint(len(train), (args.batch_size,), device=device)])
        real = real + 0.005 * torch.randn_like(real)   # a little noise smooths p_data
        x0, idx = buffer.sample(args.batch_size)
        fake = langevin(energy, x0, **sampler)
        buffer.update(fake, idx)
        e = energy(torch.cat([real, fake]))
        loss, cd = ebm_loss(*e.chunk(2), args.alpha)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), args.clip)
        optimizer.step()
        if it % 200 == 0:
            print(f'iter {it:5d}  loss {loss.item():8.4f}  E(real)-E(fake) {cd.item():8.4f}  '
                  f'{time.time() - t0:6.0f}s', flush=True)
    print(f'trained in {(time.time() - t0) / 60:.1f} min')
    os.makedirs(args.out_dir, exist_ok=True)
    save_grid(buffer.buffer[torch.randperm(len(buffer.buffer))[:64]], f'{args.out_dir}/version1_buffer.png')
    fresh = langevin(energy, noise(64), **{**sampler, 'steps': args.steps * 4})
    save_grid(fresh, f'{args.out_dir}/version1_fresh.png')
    print('saved', f'{args.out_dir}/version1_buffer.png', f'{args.out_dir}/version1_fresh.png')


def main():
    parser = argparse.ArgumentParser(description='version1: EBM with a replay buffer (PCD)')
    parser.add_argument('--data', choices=['toy', 'bitmoji'], default='toy')
    parser.add_argument('--data-dir', type=str, default='./data/bitmojis', help='flat folder of PNGs')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None, help='Langevin steps K per iteration')
    parser.add_argument('--alpha', type=float, default=None, help='energy-magnitude regulariser')
    parser.add_argument('--clip', type=float, default=1.0, help='max norm of the parameter gradient')
    parser.add_argument('--spectral-norm', action='store_true')
    # image-only options
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--step-size', type=float, default=10.0)
    parser.add_argument('--noise', type=float, default=0.005)
    parser.add_argument('--grad-clip', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    if args.data == 'toy':
        args.iters, args.steps = args.iters or 3000, args.steps or 40
        args.alpha = 0.01 if args.alpha is None else args.alpha
        train_toy(args)
    else:
        args.iters, args.steps = args.iters or 4000, args.steps or 60
        args.alpha = 0.1 if args.alpha is None else args.alpha
        train_images(args)


if __name__ == '__main__':
    main()
