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
        max_logit = np.max(inputs, axis=1, keepdims=True)

        softmax_out = np.exp(inputs - max_logit) / np.sum(
            np.exp(input - max_logit), axis=1, keepdims=true
        )
        self.softmax_out = softmax_out

        return softmax_out
