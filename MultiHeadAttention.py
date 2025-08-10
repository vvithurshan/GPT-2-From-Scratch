import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        assert context_length >= num_tokens, 'context_length must be greater than num_tokens'
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # (b, self.num_heads, num_tokens, self.head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores =queries @ keys.transpose(2, 3) # (b, self.num_heads, self.head_dim, num_tokens)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, float("-inf"))

        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim = -1)
        # print(attention_weights)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2) # (b,num n_tokens, self.num_heads, self.head_dim)
        context_vector = context_vector.contiguous()
        context_vector = context_vector.view(b, num_tokens, self.d_out)

        contextz_vector = self.W_out(context_vector)

        return context_vector
    
if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    input1 = "Hi how are you"
    input2 = "transformer is awesome"

    idx1 = tokenizer.encode(input1, allowed_special={"<|endoftext|>"})
    idx2 = tokenizer.encode(input2, allowed_special={"<|endoftext|>"})

    batch = [idx1, idx2]
    batch = torch.tensor(batch)
    print(batch.shape)

    embedding = torch.nn.Embedding(50257, 768)
    batch = embedding(batch)
    print(batch.shape)
    d_int = 768
    num_heads = 12
    context_length = 4
    dropout = 0.25
    d_out = 768
    mha = MultiHeadAttention(d_int, d_out, context_length, dropout, num_heads)
    context_vector = mha(batch)
    print(context_vector.shape)