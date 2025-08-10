import torch.nn as nn
import torch
torch.manual_seed(42)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))