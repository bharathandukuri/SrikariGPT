import numpy as np

from tokenizer import BPETokenizerSimple, VOCAB_PATH, MERGES_PATH
from gpt import GPT

TRANSFORMER_BLOCKS = 4
EMBEDDING_DIMENSIONS = 256
ATTENTION_HEADS = 4
LAYERNORM_POS = "pre"


# 1. Load tokenizer
tokenizer = BPETokenizerSimple()
tokenizer.load_vocab_and_merges(VOCAB_PATH, MERGES_PATH)

VOCAB_SIZE = len(tokenizer.vocab)


# 2. Initialize GPT
gpt = GPT(
    vocab_size=VOCAB_SIZE,
    blocks=TRANSFORMER_BLOCKS,
    d_model=EMBEDDING_DIMENSIONS,
    n_heads=ATTENTION_HEADS,
    ln_position=LAYERNORM_POS,
)


# 3. Tokenize prompt
prompt = "Thou art a very earth. Villain, "

tokens = np.array([tokenizer.encode(prompt)], dtype=np.int64)


# 4. Create causal mask
B, T = tokens.shape

mask = np.triu(np.full((T, T), -np.inf), k=1)

mask = mask[np.newaxis, np.newaxis, :, :]


# 5. Forward pass
logits = gpt.forward(tokens, mask_attention=mask)


# 6. Inspect output
print("Input shape:", tokens.shape)
print("Logits shape:", logits.shape)

# Get predictions for the final position
next_token_logits = logits[:, -1, :]

next_token = np.argmax(next_token_logits, axis=-1)

print("Next token ID:", next_token[0])
print("Predicted token:", tokenizer.decode(next_token.tolist()))
