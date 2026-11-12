"""Energy networks. Every network returns one scalar energy E_theta(x) per input (low = likely)."""

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as _sn


def maybe_sn(layer, use_sn):
    return _sn(layer) if use_sn else layer


class ToyEnergy(nn.Module):
    """E_theta: R^2 -> R, a small MLP. SiLU (Swish) is smooth, so the energy landscape and its
    gradient (which drives Langevin dynamics) are smooth too."""

    def __init__(self, width=128, spectral_norm=False):
        super().__init__()
        self.net = nn.Sequential(
            maybe_sn(nn.Linear(2, width), spectral_norm), nn.SiLU(),
            maybe_sn(nn.Linear(width, width), spectral_norm), nn.SiLU(),
            maybe_sn(nn.Linear(width, width), spectral_norm), nn.SiLU(),
            maybe_sn(nn.Linear(width, 1), spectral_norm),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConvEnergy(nn.Module):
    """E_theta: images -> R. A plain ConvNet with SiLU activations and no normalisation layers
    (batch norm would make the energy of one image depend on the rest of the batch).

    Each stage halves the resolution; global average pooling + a linear layer give the scalar.
    in_ch = 3 for an unconditional model, 6 for the conditional super-resolution model
    (the candidate high-resolution image and the upsampled low-resolution input, stacked).
    """

    def __init__(self, in_ch=3, width=64, num_stages=3, spectral_norm=False):
        super().__init__()
        layers = [maybe_sn(nn.Conv2d(in_ch, width, 3, padding=1), spectral_norm), nn.SiLU()]
        ch = width
        for i in range(num_stages):
            out = width * 2 ** (i + 1) if i < 2 else ch
            layers += [maybe_sn(nn.Conv2d(ch, ch, 3, padding=1), spectral_norm), nn.SiLU(),
                       maybe_sn(nn.Conv2d(ch, out, 4, stride=2, padding=1), spectral_norm), nn.SiLU()]
            ch = out
        self.features = nn.Sequential(*layers)
        self.head = maybe_sn(nn.Linear(ch, 1), spectral_norm)

    def forward(self, x):
        h = self.features(x).mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


class ConditionalEnergy(nn.Module):
    """E_theta(x_hr | x_lr): the energy of a candidate high-resolution image given the
    low-resolution input. x_lr is upsampled (bicubic) to the output size and stacked with x_hr,
    so the network sees both at every pixel."""

    def __init__(self, width=64, num_stages=3, spectral_norm=False):
        super().__init__()
        self.net = ConvEnergy(in_ch=6, width=width, num_stages=num_stages, spectral_norm=spectral_norm)

    def forward(self, x_hr, x_lr_up):
        return self.net(torch.cat([x_hr, x_lr_up], dim=1))


if __name__ == '__main__':
    print(ToyEnergy()(torch.randn(5, 2)).shape)
    for s in (32, 64):
        m = ConvEnergy()
        print(s, m(torch.randn(2, 3, s, s)).shape, sum(p.numel() for p in m.parameters()))
    print(ConditionalEnergy()(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)).shape)
