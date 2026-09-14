# Video 08 · CRISP-DM Academy

**Estimated runtime:** ~6:19 at a measured pace (854 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/academy
**Project directory:** [`projects/08_crispdm_academy/`](../../projects/08_crispdm_academy/)

> **The one sentence this video has to land:** Gradient check agrees to 1.4e-08 — and the bias-variance curve refuses to be a U, with variance only 3.6% of test error at maximum capacity.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/academy**.
- Walk the modules in the order used below: backprop → bias-variance → gradient descent → sampling.
- The modules are interactive — move a slider on camera, it reads much better than a static chart.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the module index

Most statistics teaching material has a problem: the figures are drawn, not computed.

You've seen the bias-variance curve — the tidy U where training error falls forever and test
error turns back up. You've seen the central limit theorem demo where the sampling
distribution is beautifully normal by n equals thirty.

In this project I computed all of those from real data instead of drawing them. And several
of them came out **not** looking like the textbook.

Those are the interesting ones, so that's where I'll spend the time.

## [0:39] Gradient checking

> **On screen —** the backpropagation module

Start with the one that behaved exactly as promised, because it's also the most immediately
useful thing in here.

This module derives backpropagation by hand for a small network —
3 → 5 (tanh) → 1 (sigmoid), binary cross-entropy — trained on Titanic (64 real passengers, 3 standardised features).

Then it does the thing you should always do: it checks the analytic gradient against a
numerical one. Nudge a parameter up a little, nudge it down a little, see how much the loss
changed, divide. That's the derivative, approximately, with no calculus involved.

Agreement between the two: maximum relative error of
1.40e-08.

That's the answer you want. Correct gradients agree to around one part in ten million. If
you get an error near, say, zero point three, you have a bug — not noise, a bug.

If your network won't train, this is the first thing to run, and it takes about five lines.

There's also a nice detail in the derivation. Two terms in the chain multiply out to a very
clean expression — and that cancellation is *why* sigmoid is paired with cross-entropy. Use
squared error instead and the sigmoid derivative survives, and it goes to zero when the unit
saturates, which stalls learning. That's not a convention. It's arithmetic.

## [2:11] The curve that isn't a U

> **On screen —** the bias-variance module

Now the one that broke the textbook picture.

The bias-variance decomposition, computed on Melbourne daily temperature (600 real observations, day-of-year → temp) with
40 bootstrap resamples per model. Polynomial degree one through fifteen.

The textbook picture: bias falls with capacity, variance rises with capacity, and test error
is a U with the optimum in the middle.

What I got: test error is minimised at degree 5, and it does rise
afterwards — so far so good.

But **variance is only 3.6% of test error even at
degree fifteen**. Bias dominates the entire range. There's no regime here where variance is
the problem.

And training error is **not monotone** under bootstrap averaging. The textbook says it falls
forever as you add capacity. It doesn't, here.

Now — I want to be careful about what that means. It doesn't mean the decomposition is
wrong. It means this particular dataset and this particular model family don't produce the
canonical shape, and the canonical shape is a schematic, not a law.

Crucially, both of those departures are **detected in code**. The pipeline computes whether
training error is monotone and what share of error is variance, and reports the answer. It
doesn't assert the U and then draw one.

## [3:40] Learning rates, including one that diverges

> **On screen —** the gradient descent module

The gradient descent module, on Melbourne monthly mean temperature (real, standardised).

Most demonstrations show you a learning rate that's too small and one that's about right.
This one also shows you one that's genuinely broken.

At the smallest rate the trajectory crawls — it's still far from the optimum when it runs
out of steps. In the middle it converges cleanly. And at the largest rate each step
overshoots by more than it corrects, and the loss **diverges**.

Not "oscillates". Diverges — it grows without bound.

That word matters, and here's why I care. My first version of this module labelled that
trajectory "oscillates", because that's the word you expect. Then I checked the actual final
loss against the initial loss, and it had grown by a factor in the hundreds of thousands.
That's not oscillation.

So the classification is now computed from the final-versus-initial loss rather than
asserted from expectation. The label describes what the numbers did.

## [4:50] The central limit theorem, slower than advertised

> **On screen —** the sampling module

The sampling distribution module, and the rule of thumb everyone learns: n equals thirty and
you're approximately normal.

Drawing real samples from a genuinely skewed real distribution, the convergence is visibly
slower than that. At small sample sizes the shape is erratic — and I've flagged that
explicitly rather than smoothing the curve, because the erraticness is itself the lesson:
it's undersampling of a heavy tail, and it's exactly what makes small-sample inference on
skewed data unreliable.

"n equals thirty" is a rule of thumb about the *centre* of a distribution. It says very
little about the tails, and the tails are usually where the decisions get made.

## [5:37] Close

> **On screen —** the quiz sections across modules

6 modules, every figure computed from real data,
including the ones that came out inconvenient.

And a limitation I'll state plainly: one dataset per concept **illustrates** a phenomenon —
it doesn't establish one. Each module says which of those it's doing. And the quizzes check
recall, not judgement, which is the skill the material is actually arguing for. That's a
real gap and it's written down as one.

Next video: I build MinHash and locality-sensitive hashing from scratch — and then compute
the exact answer as well, so I can measure what the approximation actually cost.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
