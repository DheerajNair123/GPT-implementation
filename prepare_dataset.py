from tokenizer import get_tokenizer
from pathlib import Path
import torch

tokenizer = get_tokenizer()
data_path = Path("data/shakespeare.txt").read_text()

# Encode entire dataset
tokens = tokenizer.encode(data_path)
tokens_tensor = torch.tensor(tokens, dtype=torch.long)

# Save for training
torch.save(tokens_tensor, "data/tokens.pt")
print(f"Saved {len(tokens)} tokens.")
