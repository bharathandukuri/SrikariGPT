import numpy as np
from tokenizer import Tokenizer
from gpt import GPT

TRANSFORMER_BLOCKS = 4
EMBEDDING_DIMENSIONS = 256
ATTENTION_HEADS = 4
LAYERNORM_POS = "pre"
VOCAB_PATH = ".\\models\\tokenizers\\shakespeare_all_10k_vocab.json"
MERGES_PATH = ".\\models\\tokenizers\\shakespeare_all_10k_merges.json"


# 1. Load tokenizer
tokenizer = Tokenizer.from_files(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    special_tokens=["<|endoftext|>"],
)

VOCAB_SIZE = len(tokenizer.vocab)


# 2. Initialize GPT
gpt = GPT(
    vocab_size=VOCAB_SIZE,
    blocks=TRANSFORMER_BLOCKS,
    d_model=EMBEDDING_DIMENSIONS,
    n_heads=ATTENTION_HEADS,
    ln_position=LAYERNORM_POS,
)


from generation import generate

prompt = "Thou art a very earth. Villain, "
print("Input prompt: ", prompt)
output = generate(model=gpt, tokenizer=tokenizer, prompt=prompt, max_new_tokens=30)

print("\nGenerated text:")
print(output)
