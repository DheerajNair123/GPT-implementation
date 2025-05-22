# GPT Model Implementation

A minimal implementation of a GPT (Generative Pre-trained Transformer) model built with PyTorch. This project includes a complete pipeline for training a small-scale language model and generating text.

## Features

- **Custom GPT Architecture**: Implementation of transformer blocks with multi-head self-attention
- **Tokenization**: Uses GPT-2 tokenizer for text preprocessing
- **Training Pipeline**: Complete training loop with validation
- **Text Generation**: Autoregressive text generation with temperature sampling
- **Checkpoint Management**: Model saving and loading functionality

## Project Structure

```
├── config.py              # Configuration file (empty)
├── generate.py             # Text generation script
├── model.py                # GPT model architecture
├── prepare_dataset.py      # Dataset preprocessing
├── requirements.txt        # Python dependencies
├── tokenizer.py           # Tokenizer utilities
├── train.py               # Training script
├── data/
│   ├── shakespeare.txt    # Training data (not included)
│   └── tokens.pt          # Preprocessed tokens
└── checkpoints/           # Saved model checkpoints
    └── gpt_epoch*.pt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpt-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The GPT model consists of:
- **Embedding Layers**: Token and positional embeddings
- **Transformer Blocks**: Multi-head self-attention with feed-forward networks
- **Layer Normalization**: Pre-norm architecture
- **Causal Masking**: Ensures autoregressive generation

### Default Hyperparameters
- Embedding dimension: 128
- Number of attention heads: 4
- Number of layers: 2
- Block size (context length): 128
- Vocabulary size: Based on GPT-2 tokenizer

## Usage

### 1. Prepare Dataset

Place your training text file at `data/shakespeare.txt` (or modify the path in `prepare_dataset.py`), then run:

```bash
python prepare_dataset.py
```

This will tokenize the text and save it as `data/tokens.pt`.

### 2. Train the Model

```bash
python train.py
```

Training hyperparameters:
- Batch size: 64
- Learning rate: 3e-4
- Epochs: 5
- Train/validation split: 90/10

The script will save checkpoints after each epoch in the `checkpoints/` directory.

### 3. Generate Text

```bash
python generate.py
```

By default, this generates text starting with "Once upon a time". You can modify the prompt and generation parameters in the script.

### Customizing Generation

To generate text with a different prompt:

```python
from generate import generate_text

prompt = "To be or not to be"
generated_text = generate_text(prompt, max_new_tokens=50)
print(generated_text)
```

## Model Components

### SelfAttention
- Multi-head attention mechanism
- Causal masking for autoregressive generation
- Scaled dot-product attention

### TransformerBlock
- Pre-layer normalization
- Self-attention + residual connection
- Feed-forward network + residual connection

### GPT Model
- Token and positional embeddings
- Stack of transformer blocks
- Output projection to vocabulary

## Training Details

The training process:
1. Loads preprocessed tokens from `data/tokens.pt`
2. Creates sliding window dataset with context length of 128
3. Splits data into 90% training, 10% validation
4. Trains for 5 epochs with AdamW optimizer
5. Saves model checkpoints after each epoch

## Generation Process

Text generation uses:
- **Autoregressive sampling**: Generates one token at a time
- **Multinomial sampling**: Samples from probability distribution
- **Context window**: Maintains last 128 tokens for generation

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.40.0+

## Example Output

```
Generated Text:
Once upon a time, there was a young man who lived in a small village...
```

## Customization

### Modifying Model Architecture

Edit `model.py` to change:
- Model dimensions (`embed_dim`, `num_heads`, `num_layers`)
- Architecture components
- Attention mechanisms

### Training Parameters

Modify `train.py` for:
- Different learning rates
- Batch sizes
- Training epochs
- Optimization strategies

### Dataset

Replace `data/shakespeare.txt` with your own text data. The model works with any plain text file.

## Performance Notes

This is a minimal implementation designed for educational purposes. For production use, consider:
- Larger model dimensions
- More sophisticated attention mechanisms
- Better optimization techniques
- Distributed training for large datasets
