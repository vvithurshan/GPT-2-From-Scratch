import torch.nn as nn
import torch
from LayerNorm import LayerNorm
from TransformerBlock import TransformerBlock

torch.manual_seed(42)

class GPTMODEL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.normfinal = LayerNorm(config)
        self.linear_output = nn.Linear(config["emb_dim"], config["vocab_size"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

    def forward(self, token):
        batch, seq_length = token.shape
        token_embeds = self.token_embedding(token)
        position_embeds = self.position_embedding(torch.arange(seq_length, device = token.device))
        input_embeds = token_embeds + position_embeds
        x = self.dropout(input_embeds)
        x = self.transformer_blocks(x)
        x = self.normfinal(x)
        logits = self.linear_output(x)
        return logits
    
if __name__ == "__main__":
    from config import GPT_CONFIG_124M
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    input1 = "Hi how are you"
    input2 = "transformer is awesome"

    idx1 = tokenizer.encode(input1, allowed_special={"<|endoftext|>"})
    idx2 = tokenizer.encode(input2, allowed_special={"<|endoftext|>"})

    batch = [idx1, idx2]
    batch = torch.tensor(batch)

    model = GPTMODEL(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits.shape)