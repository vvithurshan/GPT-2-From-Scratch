import torch
from GPTModel import GPTMODEL
from config import GPT_CONFIG_124M

def generateText(model, token, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        token = token[:, -context_size:]

        with torch.no_grad():
            logits = model(token)
        # focus only on the last output
        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim = -1)
        next_token = torch.argmax(probas, dim = -1, keepdim = True)

        token = torch.cat((token, next_token), dim = 1)
    return token


if __name__ == "__main__":
    import tiktoken
    model = GPTMODEL(GPT_CONFIG_124M)
        
    tokenizer = tiktoken.get_encoding("gpt2")
    input1 = "Hi how are you"
    input2 = "transformer is awesome"

    idx1 = tokenizer.encode(input1, allowed_special={"<|endoftext|>"})
    idx2 = tokenizer.encode(input2, allowed_special={"<|endoftext|>"})

    batch = [idx1, idx2]
    batch = torch.tensor(batch)
    print(f"Initial batch shape: {batch.shape}")

    output = generateText(model, batch, 100, GPT_CONFIG_124M["context_length"])
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print(f"Output-1: {tokenizer.decode(output[0].tolist())}")
    print()
    print(f"Output-2: {tokenizer.decode(output[1].tolist())}")
