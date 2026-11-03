"""A compact residual U-Net that refines respiratory-state gridding."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, channels, states, width=16):
        super().__init__()
        self.encoders = nn.ModuleList([block(channels, width), block(width, 2 * width),
                                       block(2 * width, 4 * width)])
        self.center = block(4 * width, 8 * width)
        self.decoders = nn.ModuleList([block(12 * width, 4 * width), block(6 * width, 2 * width),
                                       block(3 * width, width)])
        self.output = nn.Conv2d(width, states, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, baseline):
        skips = []
        for encoder in self.encoders:
            image = encoder(image)
            skips.append(image)
            image = F.max_pool2d(image, 2)
        image = self.center(image)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = decoder(torch.cat((image, skip), dim=1))
        return (baseline + self.output(image)).clamp_min(0)
