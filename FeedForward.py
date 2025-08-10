import torch.nn as nn
import torch
from ActivationFunctions import GELU

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LinearLayers = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"] * 4),
            GELU(),
            nn.Linear(config["emb_dim"] * 4, config["emb_dim"]),
        )

    def forward(self, x):
        return self.LinearLayers(x)