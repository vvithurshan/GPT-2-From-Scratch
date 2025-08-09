import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    """
    Custom PyTorch Dataset for preparing text data for a GPT-like model.

    This dataset takes a raw text string, tokenizes it, and creates
    input-target pairs for next-token prediction. It generates overlapping
    chunks of text to maximize the training data extracted from the source text.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt (str): The raw text data.
            tokenizer: The tokenizer instance (e.g., from tiktoken).
            max_length (int): The length of each input sequence (context window).
            stride (int): The step size to move between consecutive sequences.
                          A smaller stride creates more overlapping sequences.
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = self.tokenizer.encode(txt)

        # Create overlapping chunks for input and target sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Retrieves the input-target pair at the specified index.

        Returns:
            (torch.Tensor, torch.Tensor): A tuple containing the input IDs
                                          and the target IDs.
        """
        return self.input_ids[idx], self.target_ids[idx]