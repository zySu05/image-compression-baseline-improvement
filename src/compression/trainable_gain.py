import torch
from torch import nn


class TrainableGain(nn.Module):
    """多档可学习通道增益及逆增益。"""

    def __init__(self, channels=192, levels=4):
        super().__init__()

        self.channels = channels
        self.levels = levels

        self.log_gain = nn.Parameter(
            torch.zeros(levels, channels)
        )

        self.log_inverse_gain = nn.Parameter(
            torch.zeros(levels, channels)
        )

    def _check_level(self, level):
        if not 0 <= level < self.levels:
            raise ValueError("码率档位超出范围")

    def forward(self, y, level):
        self._check_level(level)

        gain = torch.exp(
            self.log_gain[level]
        ).view(1, self.channels, 1, 1)

        return y * gain

    def inverse(self, y, level):
        self._check_level(level)

        inverse_gain = torch.exp(
            self.log_inverse_gain[level]
        ).view(1, self.channels, 1, 1)

        return y * inverse_gain
