import torch.nn as nn
import torch
from MultiHeadAttention import MultiHeadAttention
from LayerNorm import LayerNorm
from FeedForward import FeedForward

torch.manual_seed(42)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config["emb_dim"],
            config["emb_dim"],
            config["context_length"],
            config["dropout"],
            config["num_heads"],
            config["qkv_bias"]
        )
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + residual
        return x 