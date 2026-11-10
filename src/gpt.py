import numpy as np


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.2
        self.inputs = None

        # Gradients
        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]


class PositionalEncodings:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def forward(self, inputs):
        B, T, D = inputs.shape

        pos = np.arange(T)[:, np.newaxis]
        denominator = (np.arange(D) // 2 * 2) / D
        denominator = 1 / (10000**denominator)
        denominator = denominator[np.newaxis, :]

        inner = pos * denominator
        pe_sin = np.sin(inner)
        pe_cos = np.cos(inner)

        pe_sin[:, 1::2] = 0
        pe_cos[:, 0::2] = 0

        positional_encodings = pe_sin + pe_cos

        return inputs + positional_endcodings[np.newaxis, ...]


class Softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)
        exp_values = np.exp(inputs - max_logit)

        softmax_out = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        self.softmax_out = softmax_out
        return softmax_out


class CrossEntropyLossWithSoftmax:
    def __init__(self):
        self.lse = None
        self.logits_flat = None
        self.labels_flat = None

    def forward(self, logits, labels):
        assert len(logits.shape) == 3

        B, T, V = logits.shape

        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)

        max_float = np.max(logits_flat, axis=1, keepdims=True)
        diff = logits_flat - max_float

        lse = max_float + np.log(np.sum(np.exp(diff), axis=1, keepdims=True))
        correct_logits = logits_flat[np.arange(B * T), labels_flat]

        log_softmax = np.expand_dims(correct_logits, axis=-1) - lse

        self.lse = lse
        self.logits_flat = logits_flat
        self.labels_flat = labels_flat
        self.B = B
        self.T = T
        self.V = V

        return -np.sum(log_softmax) / (B * T)


class Linear:
    def __init__(self, input_units, output_units, include_bias=True):
        self.weights = np.random.randn(input_units, output_units) * 0.02
        if include_bias:
            self.bias = np.zeros((1, output_units))
            self.grad_bias = np.zeros(self.bias.shape)
        self.inputs = None
        self.include_bias = include_bias

    def forward(self, inputs):
        self.inputs = inputs
        if self.include_bias:
            return inputs @ self.weights + self.bias
        else:
            return inputs @ self.weights


class MultiHeadAttention:
    def __init__(self, dim_in, dim_out, n_heads):
        assert dim_out % n_heads == 0

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads

        self.query_linear = Linear(dim_in, dim_out, include_bias=False)
        self.key_linear = Linear(dim_in, dim_out, include_bias=False)
        self.value_linear = Linear(dim_in, dim_out, include_bias=False)

        self.softmax = Softmax()

        self.final_projection = Linear(dim_out, dim_out, include_bias=False)

        self.queries_matrix = None
        self.keys_matrix = None
        self.values_matrix = None
        self.attention_weights = None

    def forward(self, inputs, mask=None):
        B, T, D = inputs.shape

        # 1. Generate Q, K, V
        queries = self.query_linear.forward(inputs)
        keys = self.key_linear.forward(inputs)
        values = self.value_linear.forward(inputs)

        # 2. Split into multiple heads
        # (B, T, dim_out) -> (B, T, heads, head_dim)
        queries = queries.reshape(B, T, self.n_heads, self.head_dim)
        keys = keys.reshape(B, T, self.n_heads, self.head_dim)
        values = values.reshape(B, T, self.n_heads, self.head_dim)

        # 3. Rearrange dimensions
        # (B, T, heads, head_dim)
        # -> (B, heads, T, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # 4. Calculate attention scores
        attention_scores = queries @ keys.transpose(0, 1, 3, 2)

        # 5. Scale scores
        attention_scores /= np.sqrt(self.head_dim)

        # 6. Apply mask if provided
        if mask is not None:
            attention_scores += mask

        # 7. Apply softmax
        attention_weights = self.softmax.forward(attention_scores)

        # 8. Multiply attention weights by values
        final_vector = attention_weights @ values

        # 9. Combine all heads
        # (B, heads, T, head_dim)
        # -> (B, T, heads, head_dim)
        final_vector = final_vector.transpose(0, 2, 1, 3)

        # -> (B, T, dim_out)
        final_vector = final_vector.reshape(B, T, self.dim_out)

        # 10. Final linear projection
        final_vector = self.final_projection.forward(final_vector)

        # Cache intermediate values
        self.queries_matrix = queries
        self.keys_matrix = keys
        self.values_matrix = values
        self.attention_weights = attention_weights

        return final_vector


class LayerNormalization:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.gain = np.ones((1, input_dim))
        self.bias = np.zeros((1, input_dim))

        self.mean = None
        self.mu = None
        self.var = None
        self.std = None
        self.out = None

        self.grad_gain = np.zeros(self.gain.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, inputs):
        epsilon = 1e-5

        self.mean = (1 / self.input_dim) * np.sum(inputs, axis=-1, keepdims=True)
        self.mu = inputs - self.mean
        self.var = (1 / self.input_dim) * np.sum(
            np.square(self.mu), axis=-1, keepdims=True
        )
        self.std = np.sqrt(self.var + epsilon)
        self.out = self.gain * (self.mu / self.std) + self.bias

        return self.out


class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)


class FeedForwardNetwork:
    def __init__(self, input_units, hidden_units, output_units):
        self.linear1 = Linear(input_units, hidden_units)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_units, output_units)

    def forward(self, inputs):
        x = self.linear1.forward(inputs)
        x = self.activation.forward(x)
        x = self.linear2.forward(x)
        return x


class TransformerBlock:
    def __init__(self, d_model, n_heads, ln_pos="post", res_weight_init_scale=1):
        self.ln_pos = ln_pos
        self.multi_head_attention = MultiHeadAttention(d_model, d_model, n_heads)
        self.layernorm1 = LayerNormalization(d_model)

        self.ffn = FeedForwardNetwork(d_model, 4 * d_model, d_model)
        self.layernorm2 = LayerNormalization(d_model)

    def forward(self, inputs, mask_attention=None, padding_mask=None):
        if self.ln_pos == "post":
            out = inputs + self.multi_head_attention.forward(
                inputs, mask=mask_attention
            )
            out = self.layernorm1.forward(out)
            out = out + self.ffn.forward(out)
            out = self.layernorm2.forward(out)

        elif self.ln_pos == "pre":
            out = inputs + self.multi_head_attention.forward(
                self.layernorm1.forward(inputs), mask=mask_attention
            )
            out = out + self.ffn.forward(self.layernorm2.forward(out))

        return out


class GPT:
    def __init__(self, vocab_size, blocks, d_model, n_heads, ln_position="pre"):
        self.vocab_size = vocab_size

        # Decoder
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_dim=d_model)
        self.positional_encodings_decoder = PositionalEncodings(d_model)
        res_weight_init_scale = (2 * blocks) ** (-0.5)
        self.decoder_blocks = [
            TransformerBlock(d_model, n_heads, ln_position, res_weight_init_scale)
            for _ in range(blocks)
        ]

        self.final_layernorm = LayerNormalization(d_model)
        self.final_linear = Linear(d_model, self.vocab_size)
        self.loss_function = CrossEntropyLossWithSoftmax()

    def forward(self, input_tokens_decoder, mask_attention=None, padding_mask=None):
        # Decoder forward
        out_d = self.embedding_layer.forward(input_tokens_decoder)
        out_d = self.positional_encodings_decoder.forward(out_d)

        for decoder in self.decoder_blocks:
            out_d = decoder.forward(
                inputs=out_d, mask_attention=mask_attention, padding_mask=padding_mask
            )

        out = self.final_layernorm.forward(out_d)
        return self.final_linear.forward(out)
