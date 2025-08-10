import torch
from create_dataloader import create_dataloader
from load_data import load_data
from config import GPT_CONFIG_124M
from GPTModel import GPTMODEL

text = load_data(path="Trials/discipline.txt")

# 1. Split the token list into training and validation sets
train_ratio = 0.8
train_size = int(train_ratio * len(text))
train_txt = text[:train_size]
val_txt = text[train_size:]

# 2. Create DataLoaders with the token lists
train_loader = create_dataloader(
    train_txt,
    batch_size = 2,
    max_length = GPT_CONFIG_124M['context_length'],
    stride = GPT_CONFIG_124M['context_length'],
    shuffle= False
)

val_loader = create_dataloader(
    val_txt,
    batch_size = 1,
    max_length = GPT_CONFIG_124M['context_length'],
    stride = GPT_CONFIG_124M['context_length'], 
    shuffle=False  
)

trainiter = iter(train_loader) 
train_inputs, train_targets = next(trainiter)
print(f"Train Loader Shape: \nInputs: {train_inputs.shape}\nTarget: {train_targets.shape}")

valiter = iter(val_loader) 
val_inputs, val_targets = next(valiter)
print(f"Validation Loader Shape: \nInputs: {val_inputs.shape}\nTarget: {val_targets.shape}")


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate model
def evaluate_model(model, train_loader, val_loader, device,):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,)
        val_loss = calc_loss_loader(val_loader, model, device,)
    model.train()
    return train_loss, val_loss

# Train model
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq,):
    train_losses, val_losses = [], []
    print("Training...")
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % eval_freq == 0:
            trai_loss, val_loss = evaluate_model(model, train_loader, val_loader, device)
            train_losses.append(trai_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {trai_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Training complete.")
    return train_losses, val_losses

model = GPTMODEL(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
num_epochs = 10
eval_freq = 1

train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq)

import matplotlib.pyplot as plt

plt.plot(list(range(1, num_epochs + 1)), train_losses, label='Training Loss')
plt.plot(list(range(1, num_epochs + 1)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()