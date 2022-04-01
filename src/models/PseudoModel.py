import torch
import torch.nn as nn


class PseudoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(2, 1)) / 2

    def forward(self, x):
        return x @ self.w
