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
    def __init__(self, dim_in, dim_out, include_bias=True):
        self.W = np.random.randn(dim_in, dim_out) * 0.02

        if include_bias:
            self.b = np.zeros(dim_out)
        else:
            self.b = None

    def forward(self, inputs):
        output = inputs @ self.W

        if self.b is not None:
            output = output + self.b

        return output


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
