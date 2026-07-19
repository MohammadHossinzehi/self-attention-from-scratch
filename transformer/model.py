"""Assembles the layers in transformer.layers into a small encoder-only
transformer used for sequence-to-sequence tagging tasks (input and output
sequences have the same length, predicted in parallel rather than
autoregressively)."""
import numpy as np

from .layers import Embedding, Linear, TransformerEncoderBlock, sinusoidal_positional_encoding


class TinyTransformer:
    def __init__(self, vocab_size, dim=32, n_heads=4, n_layers=2,
                 hidden_dim=64, max_len=16, seed=0):
        rng = np.random.default_rng(seed)
        self.embed = Embedding(vocab_size, dim, rng)
        self.pos_enc = sinusoidal_positional_encoding(max_len, dim)
        self.blocks = [
            TransformerEncoderBlock(dim, n_heads, hidden_dim, rng)
            for _ in range(n_layers)
        ]
        self.out_proj = Linear(dim, vocab_size, rng)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed.forward(idx) + self.pos_enc[:T][None, :, :]
        for block in self.blocks:
            x = block.forward(x)
        return self.out_proj.forward(x)

    def backward(self, dlogits):
        dx = self.out_proj.backward(dlogits)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embed.backward(dx)

    def params_and_grads(self):
        pg = self.embed.params_and_grads() + self.out_proj.params_and_grads()
        for block in self.blocks:
            pg.extend(block.params_and_grads())
        return pg
