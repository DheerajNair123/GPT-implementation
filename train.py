import torch
from torch.utils.data import Dataset, DataLoader
from model import GPT
from tokenizer import get_tokenizer
import torch.nn.functional as F
import os
import time
from tqdm import tqdm


# Hyperparameters
block_size = 128
batch_size = 64
epochs = 5
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer & dataset
tokenizer = get_tokenizer()
vocab_size = tokenizer.vocab_size

data = torch.load("data/tokens.pt", map_location="cpu")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + block_size]
        y = self.data[idx + 1:idx + block_size + 1]
        return x, y

train_loader = DataLoader(TextDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TextDataset(val_data), batch_size=batch_size)

# Initialize model
model = GPT(vocab_size=vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         x, y = [b.to(device) for b in batch]
#         logits, loss = model(x, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_train_loss = total_loss / len(train_loader)

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for batch in val_loader:
#             x, y = [b.to(device) for b in batch]
#             _, loss = model(x, y)
#             val_loss += loss.item()
#         avg_val_loss = val_loss / len(val_loader)

#     print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#     # Save checkpoint
#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save(model.state_dict(), f"checkpoints/gpt_epoch{epoch + 1}.pt")
total_start = time.time()

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_start = time.time()
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        x, y = [b.to(device) for b in batch]
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            x, y = [b.to(device) for b in batch]
            _, loss = model(x, y)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

    epoch_time = time.time() - epoch_start
    print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
    print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/gpt_epoch{epoch + 1}.pt")

total_time = time.time() - total_start
print(f"\n⏱️ Total training time: {total_time:.2f}s")
