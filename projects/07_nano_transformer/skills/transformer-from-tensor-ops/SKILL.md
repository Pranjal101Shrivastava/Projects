---
name: transformer-from-tensor-ops
description: Implement a decoder from first principles and evaluate it against real baselines
---

# Implement a decoder from first principles and evaluate it against real baselines

*Derived from [`projects/07_nano_transformer`](../../), which applied this procedure to: Tiny Shakespeare.*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

Learning transformer internals, or building a small model where you need to understand every
component.

## Procedure

### 1. Implement the components, do not assemble them

`nn.TransformerEncoderLayer` hides exactly the parts worth understanding, and it does not
expose rotary embeddings, gated feed-forwards or pre-norm arrangement.

### 2. Rotary position embeddings

Rotate each (even, odd) coordinate pair of q and k by θ = p·ω_i. The dot product of two
rotated vectors depends only on the *difference* in position, so attention becomes a function
of relative offset — with nothing learned and no absolute index to overfit.

```python
x_even, x_odd = x[..., 0::2], x[..., 1::2]
rotated = torch.stack([x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos], dim=-1)
return rotated.flatten(-2)
```

### 3. Pre-norm, not post-norm

`x + Attn(LN(x))`, never `LN(x + Attn(x))`. Post-norm routes the residual through the
normaliser, making gradient magnitude depth-dependent.

### 4. SwiGLU feed-forward

`(SiLU(W_gate x) ⊙ W_up x) W_down`. Consistently beats ReLU at matched parameter count.

### 5. Verify the causal mask

Use `is_causal=True` (or an explicit upper-triangular mask). Without it the model sees the
token it must predict, the loss collapses toward zero, and it learns nothing generative. A
suspiciously low training loss in the first hundred steps is the symptom.

### 6. Split continuous text contiguously

A random split interleaves validation windows between training windows, so validation context
is already memorised. Same reasoning as time series.

### 7. Evaluate against baselines, not against impressions

| Baseline | Meaning |
|---|---|
| Uniform over vocabulary | The ceiling — knowing nothing |
| Unigram (character frequencies) | Knowing only the marginal distribution |
| Your model | Must beat the unigram substantially |

**Decide your metrics before you see output.** A language model can always be made to look
good by selecting a sample.

### 8. Measure something falsifiable about generation

Real-word rate across temperatures is one option. State clearly what it does *not* measure:
orthography is not semantics, and a sample can be 95% real words and mean nothing.

### 9. Derive expected parameter count in code, and assert it

Documentation claiming a size that the architecture no longer produces is a silent
inaccuracy. Compute it from the configuration constants and fail the run on mismatch.

## Decision points requiring judgement

**Warmup is still needed with pre-norm.** Adam's second-moment estimate is unreliable in the
first few dozen steps.

**Character level versus subword.** Character keeps embeddings negligible so parameters sit in
the blocks, and makes spelling an observable achievement rather than the tokeniser's job.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/07_nano_transformer/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/07_nano_transformer/artifacts/`](../../artifacts/).
