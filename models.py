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


class ImageEnergy(nn.Module):
    """E_theta: images -> R for the image versions (2-4), laid out like a DCGAN discriminator.

    conv3x3 -> three stride-2 4x4 convs (width doubles each time) -> one final conv whose
    kernel covers the whole remaining (size/8 x size/8) map, i.e. a fully connected layer.
    The last layer matters: ConvEnergy above averages over positions, so its energy cannot
    easily tell WHERE a feature is (eyes above the mouth), and its fresh chains only reach
    face-coloured blobs. Here every position has its own weights. SiLU, no normalisation.

    in_ch = 3 for an unconditional energy, 6 for the conditional super-resolution energy.
    num_levels > 0 adds a learned per-level bias to every stage (version4: one energy per
    noise level of the diffusion, sharing all other weights).
    """

    def __init__(self, in_ch=3, width=64, size=32, num_levels=0):
        super().__init__()
        chans = [width, 2 * width, 4 * width, 8 * width]
        self.inp = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.down = nn.ModuleList(nn.Conv2d(a, b, 4, stride=2, padding=1) for a, b in zip(chans, chans[1:]))
        self.out = nn.Conv2d(chans[-1], 1, size // 8)
        self.act = nn.SiLU()
        self.level_bias = nn.ModuleList(nn.Embedding(num_levels, c) for c in chans) if num_levels else None

    def forward(self, x, level=None):
        h = self.inp(x)
        for i, layer in enumerate([None, *self.down]):
            if layer is not None:
                h = layer(h)
            if self.level_bias is not None:
                h = h + self.level_bias[i](level)[:, :, None, None]
            h = self.act(h)
        return self.out(h).flatten()


class ConditionalEnergy(nn.Module):
    """E_theta(x_hr | x_lr): the energy of a candidate high-resolution image given the
    low-resolution input. x_lr is upsampled (bicubic) to the output size and stacked with x_hr,
    so the network sees both at every pixel."""

    def __init__(self, width=64, size=64):
        super().__init__()
        self.net = ImageEnergy(in_ch=6, width=width, size=size)

    def forward(self, x_hr, x_lr_up):
        return self.net(torch.cat([x_hr, x_lr_up], dim=1))


if __name__ == '__main__':
    print(ToyEnergy()(torch.randn(5, 2)).shape)
    for s in (32, 64):
        m = ConvEnergy()
        print(s, m(torch.randn(2, 3, s, s)).shape, sum(p.numel() for p in m.parameters()))
    m = ImageEnergy()
    print('ImageEnergy', m(torch.randn(2, 3, 32, 32)).shape, sum(p.numel() for p in m.parameters()))
    m = ImageEnergy(num_levels=6)
    print('ImageEnergy, 6 levels', m(torch.randn(2, 3, 32, 32), torch.tensor([0, 5])).shape)
    print(ConditionalEnergy()(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)).shape)
