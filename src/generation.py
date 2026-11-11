import numpy as np


def generate(model, tokenizer, prompt, max_new_tokens=30):
    # 1. Encode prompt
    token_ids = tokenizer.encode(prompt)

    # Add batch dimension: (T) -> (1, T)
    inputs = np.array([token_ids], dtype=np.int64)

    # Get EOS token ID
    end_token_id = tokenizer.i_vocab.get(b"<|endoftext|>")

    # 2. Autoregressive generation
    for _ in range(max_new_tokens):

        B, T = inputs.shape

        # Create causal attention mask
        mask = np.triu(np.full((T, T), -np.inf), k=1)

        mask = mask[np.newaxis, np.newaxis, :, :]

        # Forward pass
        logits = model.forward(inputs, mask_attention=mask)

        # Extract logits of the last token
        next_token_logits = logits[:, -1, :]

        # Greedy decoding
        next_token = np.argmax(next_token_logits, axis=-1)

        # Append predicted token
        inputs = np.concatenate([inputs, next_token[:, np.newaxis]], axis=1)

        # Stop at EOS
        if end_token_id is not None and next_token[0] == end_token_id:
            break

    # 3. Decode generated sequence
    return tokenizer.decode(inputs[0].tolist())
