import torch
from model import GPT
from tokenizer import get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.vocab_size
block_size = 128

# Load model & checkpoint
model = GPT(vocab_size=vocab_size, block_size=block_size).to(device)
checkpoint = torch.load("checkpoints/gpt_epoch5.pt", map_location=device)  # Adjust filename if needed
model.load_state_dict(checkpoint)
model.eval()

def generate_text(prompt, max_new_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_new_tokens):
        input_ids_cond = input_ids[:, -block_size:]
        logits = model(input_ids_cond)
        logits = logits[:, -1, :]  # Take last token's logits
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    output = tokenizer.decode(input_ids[0].tolist())
    return output

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated = generate_text(prompt, max_new_tokens=100)
    print("\nGenerated Text:\n", generated)
