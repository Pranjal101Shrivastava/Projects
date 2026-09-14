# Abstract — A Character-Level Transformer Decoder Trained on CPU

**Objective.** Implement each component of a contemporary transformer decoder directly
against tensor operations, train it within a CPU budget, and evaluate it against explicit
baselines rather than by qualitative sample inspection.

**Data.** 1,115,394 characters of public-domain Shakespeare,
65 distinct symbols, split contiguously 90/10. A random split over a
continuous text would interleave validation windows with training windows, making validation
context already memorised.

**Method.** A 4-layer pre-norm decoder with 6-head causal
self-attention, rotary position embeddings, SwiGLU gated feed-forward blocks and tied
input/output embeddings, totalling 1,785,408 parameters. Trained with AdamW under
linear warmup and cosine decay with gradient clipping, on CPU only.

**Results.** Held-out perplexity reached 4.555, against
27.46 for a unigram model and 65.0
for uniform guessing — a 6.0× improvement over the frequency
baseline. Generated text contained 94.7% real corpus words
at temperature 0.8, declining monotonically to
78.6% at temperature 1.3.
Attention probing found 16 of 24 heads specialised to
short-range context.

**Conclusion.** The model acquires orthography and structural form but not semantics. Reported
metrics are chosen to make that distinction measurable rather than relying on qualitative
inspection of selected samples.

**Keywords.** transformer, rotary position embedding, SwiGLU, language modelling, perplexity
