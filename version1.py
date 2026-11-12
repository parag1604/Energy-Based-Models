"""version1: super-resolution of Bitmoji faces with an energy-based model ("SR" = super-resolution).

NEW compared to version0:
  * the data are 3x128x128 images and the energy network is a CNN (models.Network);
  * the MCMC chains no longer start from noise. They start from a *blurry* image
    (the 128x128 face downsampled to 32x32 and upsampled back). A few steps of
    gradient descent on the energy then move it towards a nearby image that the model
    finds likely, i.e. a sharp face. That is super-resolution.

Note that the EBM itself is unconditional: E(x) only scores how much x looks like a
high-resolution Bitmoji. The low-resolution input enters only as the starting point of
the sampler; because the chains are short (25 steps, clipped gradients), the result
stays close to it. Training is the same contrastive recipe as version0, with the
"fakes" now being the super-resolved images:
    loss = f(SR).mean() - f(HR).mean() + alpha * (f(HR)^2 + f(SR)^2).mean().

Every --save-every batches the script writes results/<epoch>_<batch>.png with
low resolution | generated | high resolution columns.

Run:  python version1.py --data-dir ./data/bitmoji --epochs 10
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from models import Network
from utils import get_device, sample, ebm_loss, SuperResolutionPairs, denormalize


def save_comparison(lr_batch, sr_batch, hr_batch, path, n=5):
    n = min(n, len(hr_batch))
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n), squeeze=False)
    axes[0, 0].set_title('Low Resolution')
    axes[0, 1].set_title('Generated')
    axes[0, 2].set_title('High Resolution')
    for i in range(n):
        for j, img in enumerate((lr_batch[i], sr_batch[i], hr_batch[i])):
            axes[i, j].imshow(denormalize(img).permute(1, 2, 0).cpu().numpy())
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    model.train()
    num_batches = len(dataloader)
    for batch_idx, (hr_batch, lr_batch) in enumerate(dataloader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        # a little noise on the data smooths the data distribution, so the energy
        # cannot become infinitely sharp around the training images
        hr_batch = hr_batch + torch.randn_like(hr_batch) * 0.005
        lr_batch = lr_batch + torch.randn_like(lr_batch) * 0.005
        hr_batch, lr_batch = hr_batch.to(device), lr_batch.to(device)

        # negative samples: MCMC started from the blurry images
        sr_batch = sample(model, lr_batch, steps=args.steps)

        # one forward pass for both halves (keeps dropout / statistics consistent)
        hr_scores, sr_scores = model(torch.cat([hr_batch, sr_batch], dim=0)).chunk(2, dim=0)
        loss = ebm_loss(hr_scores, sr_scores, args.alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch + 1}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')

        if (batch_idx + 1) % args.save_every == 0:
            save_comparison(lr_batch, sr_batch, hr_batch,
                            f'{args.out_dir}/{epoch + 1}_{batch_idx + 1}.png')
            torch.save(model.state_dict(), f'models/model_{epoch + 1}.pt')
    model.eval()


def main():
    parser = argparse.ArgumentParser(description='version1: EBM super-resolution of Bitmoji faces')
    parser.add_argument('--data-dir', type=str, default='./data/bitmoji',
                        help='ImageFolder root: <data-dir>/<any subfolder>/<images>')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=3e-2, help='score regulariser')
    parser.add_argument('--steps', type=int, default=25, help='MCMC steps per sample')
    parser.add_argument('--save-every', type=int, default=50, help='batches between figures')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='stop each epoch after this many batches (quick runs)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dir', type=str, default='./results')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    device = get_device()

    dataset = SuperResolutionPairs(args.data_dir, hr_size=128, lr_size=32)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)

    model = Network().to(device)
    # beta1 = 0: no momentum, as is common for EBMs/GAN critics whose objective keeps moving
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.0, 0.999))

    for epoch in range(args.epochs):
        train_epoch(model, dataloader, optimizer, device, epoch, args)


if __name__ == "__main__":
    main()
