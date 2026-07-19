"""Trains TinyTransformer on a synthetic sequence-reversal task.

The task: given a random sequence of digits, predict the same sequence
reversed. It is a deliberately simple probe for whether the self-attention
layers are actually learning to move information between positions (a
model with no attention, only per-position feed-forward layers, cannot
solve this task above chance, since reversing requires looking at other
positions).
"""
import numpy as np

from transformer.model import TinyTransformer
from transformer.optim import Adam
from transformer.layers import SoftmaxCrossEntropy


def make_batch(batch_size, seq_len, vocab_size, rng):
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len))
    y = x[:, ::-1].copy()
    return x, y


def accuracy(logits, targets):
    preds = logits.argmax(axis=-1)
    return (preds == targets).mean()


def main():
    rng = np.random.default_rng(42)
    vocab_size = 10
    seq_len = 6

    model = TinyTransformer(
        vocab_size=vocab_size, dim=32, n_heads=4, n_layers=2,
        hidden_dim=64, max_len=seq_len, seed=0,
    )
    opt = Adam(model.params_and_grads, lr=3e-3)
    loss_fn = SoftmaxCrossEntropy()

    for step in range(1, 2001):
        x, y = make_batch(64, seq_len, vocab_size, rng)
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        opt.step()

        if step % 100 == 0:
            acc = accuracy(logits, y)
            print(f"step {step:5d}  loss {loss:.4f}  train_acc {acc:.3f}")

    x_test, y_test = make_batch(256, seq_len, vocab_size, rng)
    logits = model.forward(x_test)
    acc = accuracy(logits, y_test)
    print(f"\nfinal held-out accuracy on sequence reversal: {acc:.3f}")

    example = x_test[0]
    pred = logits[0].argmax(axis=-1)
    print(f"example input:     {example.tolist()}")
    print(f"example predicted: {pred.tolist()}")
    print(f"example target:    {y_test[0].tolist()}")


if __name__ == "__main__":
    main()
