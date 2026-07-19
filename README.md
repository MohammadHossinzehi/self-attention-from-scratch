# self-attention-from-scratch

Multi-head self-attention and a transformer encoder block, implemented from
scratch in NumPy. No PyTorch, no TensorFlow, no autograd of any kind:
every `forward()` has a matching hand-derived `backward()`, and every one
of those backward passes is checked against a numerical (finite-difference)
gradient in `tests/test_gradients.py`.

## Why

Most "transformer from scratch" projects either wrap `torch.nn.Linear` and
call it a day, or lean on `autograd`/`jax.grad` for the hard part. Neither
actually shows how attention gets trained. This project derives and
implements the backward pass for scaled dot-product attention by hand,
including the softmax Jacobian and the split/merge across heads, so the
gradient math is fully visible in `transformer/layers.py` rather than
hidden behind a framework.

To prove the layers are actually correct (not just "the loss goes down"),
`train.py` trains the model on a synthetic sequence-reversal task: given a
random sequence of digits, predict the same sequence reversed. This task is
a deliberate probe for attention specifically — a model that can only look
at one position at a time (no cross-position mixing) cannot solve it above
chance, since producing position `i` of the output requires reading from
position `T-1-i` of the input. A working implementation converges to 100%
accuracy within a few hundred gradient steps; a broken one plateaus near
1/vocab_size.

## What's implemented

- `Linear`, `LayerNorm`, `Embedding` — with hand-written backward passes
- `MultiHeadSelfAttention` — scaled dot-product attention split across
  heads, forward and backward, using `einsum` for both directions
- `FeedForward` — position-wise two-layer MLP with ReLU
- `TransformerEncoderBlock` — post-norm residual block (attention -> add
  & norm -> feed-forward -> add & norm), matching the original
  "Attention Is All You Need" layout
- `TinyTransformer` — embedding + fixed sinusoidal positional encoding +
  a stack of encoder blocks + output projection, used here as a
  parallel (non-autoregressive) sequence tagger
- `Adam` — a minimal optimizer over the `params_and_grads()` convention
  shared by every layer

## Running it

```bash
pip install -r requirements.txt

# train on the sequence-reversal task
python train.py

# run the gradient checks
python -m pytest tests/ -v
```

Expected training output (this exact run is deterministic — everything is
seeded):

```
step   100  loss 0.0762  train_acc 1.000
step   200  loss 0.0034  train_acc 1.000
...
final held-out accuracy on sequence reversal: 1.000
example input:     [6, 1, 1, 9, 4, 6]
example predicted: [6, 4, 9, 1, 1, 6]
example target:    [6, 4, 9, 1, 1, 6]
```

## Design decisions

**Manual backprop over autograd.** Every layer caches what it needs during
`forward()` and computes exact gradients in `backward()` using the chain
rule (see `MultiHeadSelfAttention.backward` for the least obvious one: the
softmax-weighted attention backward pass). This is slower to write than
`loss.backward()`, but it's the whole point of the project — you can read
the backward pass and see exactly where each gradient term comes from.

**Encoder-only, non-autoregressive.** The reversal task maps a
fixed-length input to a same-length output, so the model runs as a
sequence tagger: bidirectional self-attention over all positions at once,
with a per-position softmax classifier on top. This sidesteps causal
masking and autoregressive decoding, which are separate concerns from
"does attention work," and keeps the implementation focused on the
attention mechanism itself. `MultiHeadSelfAttention.forward` does accept
an optional mask, so causal masking is a one-line change if you want to
extend this to a decoder.

**Testing via gradient checking, not example-based unit tests.** Because
the interesting bug surface here is "did I get a derivative wrong," the
test suite (`tests/test_gradients.py`) compares every layer's analytic
gradient against a central-difference numerical gradient
(`(f(x+eps) - f(x-eps)) / (2*eps)`) rather than asserting on fixed
input/output pairs. This catches sign errors, transposed matrix
multiplies, and wrong axes far more reliably than hand-picked examples
would. One test (`test_feedforward_and_block_gradients`) uses a looser
tolerance (2e-2 instead of 1e-3) because gradients flowing through two
stacked `LayerNorm`s shrink to a magnitude where finite-difference noise
becomes comparable to the signal — the absolute agreement is still
~1e-5, it's a metric artifact, not a bug.

**Post-norm, not pre-norm.** The encoder block normalizes after each
residual addition (as in the original Transformer paper) rather than
before (as most modern LLMs do). Post-norm is easier to derive backward
gradients for by hand and is what "Attention Is All You Need" actually
describes; pre-norm's main advantage (training stability at large depth)
doesn't matter at the 1-2 layer scale used here.
