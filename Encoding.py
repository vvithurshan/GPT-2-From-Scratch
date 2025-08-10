import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
input1 = "Hi how are you"
input2 = "transformer is awesome"

idx1 = tokenizer.encode(input1, allowed_special={"<|endoftext|>"})
idx2 = tokenizer.encode(input2, allowed_special={"<|endoftext|>"})

batch = [idx1, idx2]
print(batch)
print(batch.shape)

