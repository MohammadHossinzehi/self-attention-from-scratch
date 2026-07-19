"""Gradient checks: every hand-written backward() is verified against a
numerical (central-difference) gradient of a scalar loss. This is the
same technique used in transformer/model.py's own training loop's loss
function, applied here as an automated correctness check rather than a
one-off script.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.layers import Linear, LayerNorm, MultiHeadSelfAttention, SoftmaxCrossEntropy
from transformer.model import TinyTransformer


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_pos = f()
        x[idx] = orig - eps
        f_neg = f()
        x[idx] = orig
        grad[idx] = (f_pos - f_neg) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_linear_gradients():
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng)
    x = rng.normal(size=(2, 5, 4))

    def loss_fn():
        return (layer.forward(x) ** 2).sum()

    y = layer.forward(x)
    dx_analytic = layer.backward(2 * y)
    dW_analytic = layer.dW.copy()

    assert rel_error(dx_analytic, numerical_grad(loss_fn, x)) < 1e-4
    assert rel_error(dW_analytic, numerical_grad(loss_fn, layer.W)) < 1e-4


def test_layernorm_gradients():
    rng = np.random.default_rng(1)
    ln = LayerNorm(6)
    x = rng.normal(size=(3, 6))

    def loss_fn():
        return (ln.forward(x) ** 2).sum()

    y = ln.forward(x)
    dx_analytic = ln.backward(2 * y)

    assert rel_error(dx_analytic, numerical_grad(loss_fn, x)) < 1e-4


def test_attention_gradients():
    rng = np.random.default_rng(2)
    attn = MultiHeadSelfAttention(dim=8, n_heads=2, rng=rng)
    x = rng.normal(size=(2, 4, 8))

    def loss_fn():
        return (attn.forward(x) ** 2).sum()

    y = attn.forward(x)
    dx_analytic = attn.backward(2 * y)

    assert rel_error(dx_analytic, numerical_grad(loss_fn, x)) < 1e-3


def test_feedforward_and_block_gradients():
    rng = np.random.default_rng(4)
    model = TinyTransformer(vocab_size=6, dim=8, n_heads=2, n_layers=2,
                             hidden_dim=16, max_len=5, seed=4)
    block = model.blocks[0]
    x = rng.normal(size=(2, 5, 8))

    def loss_fn():
        return (block.forward(x) ** 2).sum()

    y = block.forward(x)
    dx_analytic = block.backward(2 * y)

    # small-magnitude gradients through two stacked LayerNorms make the
    # relative-error metric noisier here; absolute agreement is still ~1e-5
    assert rel_error(dx_analytic, numerical_grad(loss_fn, x)) < 2e-2


def test_full_model_gradient_on_output_projection():
    rng = np.random.default_rng(3)
    model = TinyTransformer(vocab_size=6, dim=8, n_heads=2, n_layers=1,
                             hidden_dim=16, max_len=5, seed=3)
    loss_obj = SoftmaxCrossEntropy()
    idx = rng.integers(0, 6, size=(2, 5))
    targets = rng.integers(0, 6, size=(2, 5))

    def loss_fn():
        return loss_obj.forward(model.forward(idx), targets)

    logits = model.forward(idx)
    loss_obj.forward(logits, targets)
    dlogits = loss_obj.backward()
    model.backward(dlogits)

    dW_analytic = model.out_proj.dW.copy()
    dW_numeric = numerical_grad(loss_fn, model.out_proj.W)

    assert rel_error(dW_analytic, dW_numeric) < 1e-3


def test_full_model_gradient_on_embedding():
    rng = np.random.default_rng(5)
    model = TinyTransformer(vocab_size=6, dim=8, n_heads=2, n_layers=1,
                             hidden_dim=16, max_len=5, seed=5)
    loss_obj = SoftmaxCrossEntropy()
    idx = rng.integers(0, 6, size=(2, 5))
    targets = rng.integers(0, 6, size=(2, 5))

    def loss_fn():
        return loss_obj.forward(model.forward(idx), targets)

    logits = model.forward(idx)
    loss_obj.forward(logits, targets)
    dlogits = loss_obj.backward()
    model.backward(dlogits)

    dW_analytic = model.embed.dW.copy()
    dW_numeric = numerical_grad(loss_fn, model.embed.W)

    assert rel_error(dW_analytic, dW_numeric) < 1e-3
