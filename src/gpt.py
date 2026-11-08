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


embedding = EmbeddingLayer(10, 5)
inputs = np.array([6, 9, 3, 2, 7])
print(embedding.forward(inputs))
