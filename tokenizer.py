from transformers import GPT2TokenizerFast

def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    text = "To be, or not to be, that is the question."
    tokens = tokenizer.encode(text)
    print("Token IDs:", tokens)
    print("Decoded:", tokenizer.decode(tokens))
