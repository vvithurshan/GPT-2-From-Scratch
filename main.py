from create_dataloader import create_dataloader
from load_data import load_data

text = load_data(path="Trials/discipline.txt")
dataloader = create_dataloader(text,
                               batch_size =4,
                               max_length = 256,
                               stride = 128,
                               )
dataiter = iter(dataloader) 
inputs, targets = next(dataiter)
print(inputs.shape)
print(targets.shape)