# I built a transformer from scratch on a CPU, and measured what it actually learned

*1,785,408 parameters, 43 minutes, no GPU*

Small language model demos follow a script. Train something on Shakespeare, generate a
paragraph, quote the paragraph. The reader is invited to be impressed by prose that sounds
vaguely Elizabethan.

The problem with that script is that you can always find a good paragraph. Generate twenty,
pick the best, and the model looks far better than it is.

So I decided up front what I would report, before I saw any output.

## Every component written by hand

No `nn.TransformerEncoderLayer`. The architecture is the subject, and a wrapper hides exactly
the parts worth understanding.

**Rotary position embeddings.** Instead of adding a position vector, you *rotate* the query
and key vectors by an angle proportional to position. The clever part is algebraic: when you
dot-product two rotated vectors, the rotations partly cancel and what survives depends only on
the *difference* in position. Relative position falls out of the geometry. Nothing is learned,
and there is no absolute index for the model to memorise.

**SwiGLU.** The feed-forward block computes two projections — one is values, one is a gate —
and multiplies them. It consistently beats ReLU at the same parameter count. Nobody has a fully
satisfying theory for why.

**Pre-norm.** Normalise *before* the attention block rather than after. Post-norm, the original
2017 arrangement, passes the residual stream through the normaliser, which makes gradients
depth-dependent and training fragile. Every modern decoder switched.

## What I promised to report

Two numbers, chosen before training.

**Perplexity against baselines.** A perplexity of 4.55 means
nothing on its own. So:

| | Perplexity |
|---|---:|
| Guessing uniformly among 65 characters | 65.0 |
| Knowing only how often each character appears | 27.46 |
| **The transformer** | **4.55** |

6.0× better than knowing the letter frequencies. That is a
real result, and it is bounded above by how much there is to learn from a million characters.

**Real-word rate.** What fraction of generated words are actual words from the corpus?

| Temperature | Real words |
|---:|---:|
| 0.5 | 97.5% |
| 0.8 | 94.7% |
| 1.0 | 82.2% |
| 1.3 | 78.6% |

At low temperature it spells almost perfectly. Crank the temperature and coherence degrades
predictably. That monotone decline is the temperature–quality trade-off, measured instead of
asserted.

## What the model did not learn

It learned **spelling**. It learned **dramatic layout** — speaker names in caps, a colon, a
line break, verse-length lines. It learned that `th` is common and `qx` is not.

It did not learn **meaning**. Not a little bit of meaning. None.

Real-word rate measures orthography and nothing else. You can score
97.5% real words and produce text that is completely
semantically empty — which is exactly what this model does.

That distinction disappears in write-ups that quote a sample and let the reader's pattern
matching do the rest. Shakespeare-shaped text activates the same recognition as Shakespeare.

## One thing I found that I did not expect

I probed each attention head to see how far back it typically looks.

**16 of 24 heads** concentrate on the previous ten characters —
they are doing letter-level work, bigrams and word boundaries. The rest reach across a line or
more, carrying longer structure.

Nothing in the code assigns those roles. All 24 heads are
architecturally identical and randomly initialised. The division of labour is learned, in
43 minutes, on a CPU.

## The split detail

The corpus is one continuous text. I split it contiguously — first 90% train, last 10%
validation.

Splitting *randomly* would have been a mistake. Random windows interleave: a validation window
sits between two training windows, so the model has already seen its immediate context. The
reported perplexity would be optimistic and nothing would indicate it.

Continuous text needs a continuous split, for the same reason time series do.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/transformer](https://pranjal101shrivastava.github.io/Projects/#/p/transformer)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
