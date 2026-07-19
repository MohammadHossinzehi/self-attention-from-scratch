"""
Core building blocks of the transformer, implemented from scratch in NumPy.

Every layer exposes:
    forward(x)   -> y            (and caches whatever it needs for backward)
    backward(dy) -> dx           (accumulates dW/db-style gradients on self)
    params_and_grads()           -> list[(param_array, grad_array)]

There is no autograd anywhere in this file. Every backward() method is a
hand-derived chain-rule implementation. tests/test_gradients.py checks each
one against a numerical (finite-difference) gradient.
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sinusoidal_positional_encoding(max_len, dim):
    """Fixed (non-learned) sinusoidal position embeddings, as in
    'Attention Is All You Need'."""
    pe = np.zeros((max_len, dim))
    position = np.arange(max_len)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class Linear:
    """y = x @ W + b, applied to the last axis of x (any number of
    leading batch/sequence axes)."""

    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy):
        x = self._x
        flat_x = x.reshape(-1, x.shape[-1])
        flat_dy = dy.reshape(-1, dy.shape[-1])
        self.dW = flat_x.T @ flat_dy
        self.db = flat_dy.sum(axis=0)
        return dy @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class LayerNorm:
    """Normalizes over the last axis, then applies a learned scale/shift."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * std_inv
        out = self.gamma * xhat + self.beta
        self._cache = (xhat, std_inv)
        return out

    def backward(self, dy):
        xhat, std_inv = self._cache
        D = dy.shape[-1]
        self.dgamma = (dy * xhat).reshape(-1, D).sum(axis=0)
        self.dbeta = dy.reshape(-1, D).sum(axis=0)

        dxhat = dy * self.gamma
        dx = (1.0 / D) * std_inv * (
            D * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Embedding:
    def __init__(self, vocab_size, dim, rng):
        self.W = rng.normal(0, 0.02, size=(vocab_size, dim))
        self.dW = np.zeros_like(self.W)
        self._idx = None

    def forward(self, idx):
        self._idx = idx
        return self.W[idx]

    def backward(self, dy):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self._idx, dy)
        return None

    def params_and_grads(self):
        return [(self.W, self.dW)]


class MultiHeadSelfAttention:
    """Scaled dot-product self-attention, split across n_heads, with
    a from-scratch backward pass (no autograd)."""

    def __init__(self, dim, n_heads, rng):
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = Linear(dim, dim, rng)
        self.k_proj = Linear(dim, dim, rng)
        self.v_proj = Linear(dim, dim, rng)
        self.out_proj = Linear(dim, dim, rng)
        self._cache = None

    def _split_heads(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        return x.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x, mask=None):
        Q = self._split_heads(self.q_proj.forward(x))
        K = self._split_heads(self.k_proj.forward(x))
        V = self._split_heads(self.v_proj.forward(x))

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum('bhtd,bhsd->bhts', Q, K) * scale
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        attn = softmax(scores, axis=-1)
        out = np.einsum('bhts,bhsd->bhtd', attn, V)
        merged = self._merge_heads(out)
        y = self.out_proj.forward(merged)

        self._cache = (Q, K, V, attn, scale, mask)
        return y

    def backward(self, dy):
        Q, K, V, attn, scale, mask = self._cache
        dmerged = self.out_proj.backward(dy)
        dout = self._split_heads(dmerged)

        dattn = np.einsum('bhtd,bhsd->bhts', dout, V)
        dV = np.einsum('bhts,bhtd->bhsd', attn, dout)

        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        if mask is not None:
            dscores = np.where(mask, dscores, 0.0)

        dQ = np.einsum('bhts,bhsd->bhtd', dscores, K) * scale
        dK = np.einsum('bhts,bhtd->bhsd', dscores, Q) * scale

        dx_q = self.q_proj.backward(self._merge_heads(dQ))
        dx_k = self.k_proj.backward(self._merge_heads(dK))
        dx_v = self.v_proj.backward(self._merge_heads(dV))

        return dx_q + dx_k + dx_v

    def params_and_grads(self):
        pg = []
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            pg.extend(layer.params_and_grads())
        return pg


class FeedForward:
    """Two-layer position-wise MLP with a ReLU in between."""

    def __init__(self, dim, hidden_dim, rng):
        self.fc1 = Linear(dim, hidden_dim, rng)
        self.fc2 = Linear(hidden_dim, dim, rng)
        self._relu_mask = None

    def forward(self, x):
        h = self.fc1.forward(x)
        self._relu_mask = (h > 0)
        h = np.where(self._relu_mask, h, 0.0)
        return self.fc2.forward(h)

    def backward(self, dy):
        dh = self.fc2.backward(dy)
        dh = dh * self._relu_mask
        return self.fc1.backward(dh)

    def params_and_grads(self):
        return self.fc1.params_and_grads() + self.fc2.params_and_grads()


class TransformerEncoderBlock:
    """Post-norm encoder block: self-attention -> residual -> LayerNorm ->
    feed-forward -> residual -> LayerNorm (as in the original Transformer)."""

    def __init__(self, dim, n_heads, hidden_dim, rng):
        self.attn = MultiHeadSelfAttention(dim, n_heads, rng)
        self.ln1 = LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim, rng)
        self.ln2 = LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out = self.attn.forward(x, mask=mask)
        res1 = x + attn_out
        norm1 = self.ln1.forward(res1)
        ff_out = self.ff.forward(norm1)
        res2 = norm1 + ff_out
        return self.ln2.forward(res2)

    def backward(self, dy):
        dres2 = self.ln2.backward(dy)
        dnorm1 = dres2 + self.ff.backward(dres2)
        dres1 = self.ln1.backward(dnorm1)
        dx = dres1 + self.attn.backward(dres1)
        return dx

    def params_and_grads(self):
        return (self.attn.params_and_grads() + self.ln1.params_and_grads()
                + self.ff.params_and_grads() + self.ln2.params_and_grads())


class SoftmaxCrossEntropy:
    """Cross-entropy loss over per-position logits, averaged over all
    (batch, position) pairs."""

    def __init__(self):
        self._cache = None

    def forward(self, logits, targets, mask=None):
        probs = softmax(logits, axis=-1)
        B, T, V = logits.shape
        idx_b, idx_t = np.indices((B, T))
        logp = np.log(probs[idx_b, idx_t, targets] + 1e-12)
        if mask is None:
            mask = np.ones((B, T))
        loss = -(logp * mask).sum() / mask.sum()
        self._cache = (probs, targets, mask)
        return loss

    def backward(self):
        probs, targets, mask = self._cache
        B, T, V = probs.shape
        dlogits = probs.copy()
        idx_b, idx_t = np.indices((B, T))
        dlogits[idx_b, idx_t, targets] -= 1
        dlogits *= mask[..., None]
        dlogits /= mask.sum()
        return dlogits
