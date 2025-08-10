import torch.nn as nn
import torch

torch.manual_seed(42)
class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config["emb_dim"]))
        self.shift = nn.Parameter(torch.zeros(config["emb_dim"]))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        variance = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift