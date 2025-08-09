from torch.utils.data import DataLoader
import tiktoken
from gpt_dataset import GPTDataset

def create_dataloader(data, batch_size = 4, max_length = 256, 
                      stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(data, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, 
                            batch_size = batch_size,
                            shuffle = shuffle,
                            drop_last = drop_last,
                            num_workers = num_workers)
    return dataloader