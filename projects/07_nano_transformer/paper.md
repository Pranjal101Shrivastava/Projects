# Implementing and Evaluating a Character-Level Transformer Decoder

## 1. Scope

The objective is pedagogical: implement every component of a modern decoder from tensor
operations, train within a CPU budget, and evaluate honestly. The last constraint shapes the
work most, because a language model can always be made to look good by quoting its best
sample.

## 2. Architecture

### 2.1 Rotary position embeddings

Position is encoded by rotating each (even, odd) coordinate pair of the query and key vectors
by an angle θ_i = p · ω_i, where p is the token index and ω_i = base^(−2i/d).

For a dot product between a query at position p and a key at position q, the rotation
contributes a factor depending only on (p − q). Attention logits therefore become a function
of *relative* offset without any learned position parameters and without an absolute index the
model could overfit to.

### 2.2 SwiGLU feed-forward

    FFN(x) = (SiLU(W_gate x) ⊙ W_up x) W_down

A gated activation: one projection produces values, another a gate, and their elementwise
product passes through. Empirically outperforms ReLU and GELU at matched parameter count.

### 2.3 Pre-normalisation

Each block computes x + Attn(LN(x)) then x + FFN(LN(x)). The original post-norm arrangement
routes the residual through the normaliser, making gradient magnitude depend on depth and
requiring warmup for stability. Pre-norm leaves an unobstructed residual path.

### 2.4 Weight tying

The token embedding matrix serves as the output projection. On a 65-symbol
vocabulary this saves 12,480 parameters, which is
negligible. The justification is the inductive bias: the representation of a character on
input should be the vector scoring it on output.

### 2.5 Configuration

4 layers, 6 heads, d_model 192, d_ff
512, context 128, dropout 0.1. Total
1,785,408 parameters.

The expected parameter count is derived from the configuration constants in code and asserted
against the constructed model at runtime, so documentation cannot drift from the architecture.

## 3. Data and splitting

1,115,394 characters, 65 distinct symbols. Character-level
tokenisation was chosen over subword: it keeps embedding and output layers negligible so
nearly all parameters sit in the transformer blocks, and it makes spelling an observable
achievement rather than something the tokeniser handles invisibly.

Contiguous chronological split. A random split over a continuous text would interleave validation windows with training windows, so validation context would already have been memorised and perplexity would be optimistic.

## 4. Optimisation

AdamW (β = 0.9, 0.95; weight decay 0.1), linear warmup over 120 steps then cosine decay,
gradient clipping at global norm 1.0, batch size 48, 128-character context.

Warmup is retained despite pre-norm because Adam's second-moment estimate is unreliable in the
first few dozen steps; a full-magnitude update then can move parameters into a region from
which the model does not recover.

## 5. Evaluation

### 5.1 Perplexity against baselines

| Model | Perplexity |
|---|---:|
| Uniform over 65 symbols | 65.00 |
| Unigram (corpus character frequencies) | 27.462 |
| This model (held out) | **4.555** |

The two baselines bound what "learning something" must mean on this corpus. The model improves
6.0× over the frequency distribution.

### 5.2 Real-word rate

| Temperature | Words | Real-word rate | Mean line length |
|---:|---:|---:|---:|
| 0.5 | 81 | 97.5% | 32.2 |
| 0.8 | 75 | 94.7% | 35.3 |
| 1.0 | 73 | 82.2% | 32.2 |
| 1.3 | 70 | 78.6% | 32.3 |

This measures orthography, not semantics. It is reported precisely because it makes the
limitation explicit: high real-word rate is compatible with complete semantic emptiness.

### 5.3 Attention specialisation

Mean attention distance was probed per head on held-out text. 16 of
24 heads concentrate below 10 characters. Head roles are not assigned
by the architecture; the specialisation is learned.

### 5.4 Generalisation

Final train loss 1.3091, validation 1.5163,
gap 0.2072.

## 6. Limitations

- The model learns orthography and dramatic layout, not meaning. Generated text has correct-looking speaker labels and plausible English morphology while being semantically empty. Real-word rate measures the former and says nothing about the latter.
- Train and validation loss differ by 0.2072. That gap indicates the model has begun memorising and a larger corpus or stronger regularisation would be needed to train longer.
- Perplexity is measured on held-out Shakespeare. It says nothing about performance on any other kind of text.

*Generated at commit `1c3d162` · seed 42 · 2583.47s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
