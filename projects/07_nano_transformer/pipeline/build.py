"""A character-level transformer trained from scratch on CPU.

Run:
    PYTHONPATH=lib python3 projects/07_nano_transformer/pipeline/build.py

Every component is implemented directly against tensor operations rather than assembled
from ``nn.TransformerEncoderLayer``. The architecture is the subject here, so hiding it
behind a single library call would defeat the purpose — and several of the choices below
(rotary embeddings, SwiGLU, pre-norm) are precisely the details that a high-level wrapper
does not expose.

Architecture, and why each piece is what it is:

* **Pre-norm residual blocks.** Post-norm (the original 2017 arrangement) needs learning-rate
  warmup to train stably because the residual path passes through the normaliser. Pre-norm
  leaves the residual path clean, which is why every modern decoder uses it.
* **Rotary position embeddings (RoPE).** Encodes position by *rotating* query and key
  vectors, so attention logits depend on relative offset rather than absolute index. This
  falls out of the algebra rather than being trained, and lets the model handle offsets it
  never saw during training.
* **SwiGLU feed-forward.** A gated activation: one projection produces values, another
  produces a gate, and their product passes through. Empirically stronger than ReLU or GELU
  at equal parameter count, which is why it is now standard.
* **Weight tying.** The token embedding and output projection share one matrix. On a
  65-token vocabulary this is a rounding error in parameters, but it is the correct
  inductive bias: the vector representing a character on the way in should be the vector
  scoring it on the way out.

The honest framing throughout: this is a small model on a small corpus. It learns
Shakespearean *orthography and rhythm* — plausible spellings, speaker labels, line breaks —
and does not learn meaning. The evaluation below measures what it actually does rather than
cherry-picking a sample that reads well.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "07_nano_transformer"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

# Sized so a full training run finishes in a few minutes on CPU.
CONTEXT = 128
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
BATCH_SIZE = 48
MAX_STEPS = 2200
EVAL_EVERY = 100
EVAL_BATCHES = 24
LEARNING_RATE = 3e-4
WARMUP_STEPS = 120
VAL_FRACTION = 0.1

# Parameter count implied by the configuration above, used in prose written before the
# model is constructed. Embedding and output head share one matrix (weight tying), so the
# vocabulary term is counted once. Verified against the constructed model at runtime.
_VOCAB = 65
EXPECTED_PARAMS = (
    _VOCAB * D_MODEL                                   # tied embedding / output head
    + N_LAYERS * (
        4 * D_MODEL * D_MODEL                          # qkv projection + output projection
        + 3 * D_MODEL * D_FF                           # SwiGLU gate, up, down
        + 4 * D_MODEL                                  # two LayerNorms (weight + bias)
    )
    + 2 * D_MODEL                                      # final LayerNorm
)


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    # ==================================================================================
    # Model
    # ==================================================================================
    class RotaryEmbedding(nn.Module):
        """Rotary position embedding.

        Position enters by rotating each (even, odd) coordinate pair of the query and key
        vectors by an angle proportional to the token's index. Because a dot product
        between two rotated vectors depends only on the *difference* of their angles, the
        attention logit becomes a function of relative position — without any learned
        position parameters and without an absolute index the model could overfit to.
        """

        def __init__(self, dim: int, base: int = 10000) -> None:
            super().__init__()
            inverse_frequency = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inverse_frequency", inverse_frequency)

        def forward(self, seq_len: int, device: torch.device):
            positions = torch.arange(seq_len, device=device).float()
            angles = torch.outer(positions, self.inverse_frequency)
            return torch.cos(angles), torch.sin(angles)

    def apply_rotary(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"):
        """Rotate coordinate pairs of x by the supplied angles."""
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )
        return rotated.flatten(-2)

    class CausalSelfAttention(nn.Module):
        """Multi-head causal self-attention with rotary positions."""

        def __init__(self) -> None:
            super().__init__()
            self.n_heads = N_HEADS
            self.head_dim = D_MODEL // N_HEADS
            self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
            self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.rotary = RotaryEmbedding(self.head_dim)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, x):
            batch, seq_len, _ = x.shape
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = (
                t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                for t in qkv
            )

            cos, sin = self.rotary(seq_len, x.device)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

            # is_causal enforces the autoregressive mask: position t attends to <= t only.
            # Without it the model sees the token it must predict and the loss collapses to
            # near zero while the model learns nothing generative.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=DROPOUT if self.training else 0.0, is_causal=True
            )
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, D_MODEL)
            return self.dropout(self.proj(out))

    class SwiGLU(nn.Module):
        """Gated feed-forward: (W1 x ⊙ SiLU(W2 x)) W3."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(D_MODEL, D_FF, bias=False)
            self.up = nn.Linear(D_MODEL, D_FF, bias=False)
            self.down = nn.Linear(D_FF, D_MODEL, bias=False)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, x):
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

    class Block(nn.Module):
        """Pre-norm transformer block: x + attn(norm(x)), then x + ff(norm(x))."""

        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(D_MODEL)
            self.attn = CausalSelfAttention()
            self.norm2 = nn.LayerNorm(D_MODEL)
            self.ff = SwiGLU()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.ff(self.norm2(x))

    class NanoTransformer(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, D_MODEL)
            self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
            self.norm = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
            self.head.weight = self.embed.weight  # weight tying
            self.apply(self._init)

        @staticmethod
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            x = self.embed(idx)
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.norm(x))
            if targets is None:
                return logits, None
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                     top_k: int | None = 40):
            self.eval()
            for _ in range(max_new_tokens):
                window = idx[:, -CONTEXT:]
                logits, _ = self(window)
                logits = logits[:, -1, :] / max(1e-8, temperature)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                idx = torch.cat([idx, torch.multinomial(probabilities, 1)], dim=1)
            return idx

    # ==================================================================================
    # Data
    # ==================================================================================
    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "What does a four-layer character transformer actually learn from one million "
            "characters of Shakespeare on a CPU budget — and how would we know?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "The objective is pedagogical rather than commercial: build every "
                "component of a modern decoder from tensor operations, train it within a "
                "few CPU-minutes, and evaluate it honestly. 'Honestly' is the constraint "
                "that shapes the work — a language model can always be made to look good "
                "by quoting its best sample."
            ),
            decisions=[
                Decision(
                    question="What claim will be made about the trained model?",
                    choice=(
                        "That it learns orthography and dramatic form, not meaning."
                    ),
                    rationale=(
                        f"A {EXPECTED_PARAMS / 1e6:.1f}M-parameter model on 1.1M characters "
                        "can learn which letter "
                        "sequences are English-shaped and how a play is laid out. It "
                        "cannot learn semantics, and claiming otherwise from a "
                        "cherry-picked sample would be the standard dishonesty of small-LM "
                        "demos. The evaluation therefore measures held-out perplexity and "
                        "the proportion of generated words that are real, rather than "
                        "displaying one good paragraph."
                    ),
                ),
            ],
            evidence={
                "parameter_budget": f"~{EXPECTED_PARAMS / 1e6:.2f}M",
                "compute_budget": "CPU, minutes",
            },
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        print("\n[1/4] Loading corpus …")
        text = data.load_text("tiny_shakespeare")
        vocabulary = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(vocabulary)}
        itos = {i: ch for ch, i in stoi.items()}
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

        # Chronological split, not random. A random split over a contiguous corpus would
        # place validation windows between training windows, so the model would be
        # validated on passages whose immediate context it had memorised.
        split_at = int(len(encoded) * (1 - VAL_FRACTION))
        train_data, val_data = encoded[:split_at], encoded[split_at:]

        corpus = {
            "n_characters": len(text),
            "vocab_size": len(vocabulary),
            "vocabulary": "".join(vocabulary),
            "train_characters": int(split_at),
            "val_characters": int(len(encoded) - split_at),
            "split_rationale": (
                "Contiguous chronological split. A random split over a continuous text "
                "would interleave validation windows with training windows, so validation "
                "context would already have been memorised and perplexity would be "
                "optimistic."
            ),
            "char_frequencies": sorted(
                (
                    {"char": repr(ch)[1:-1], "count": int((encoded == i).sum())}
                    for ch, i in stoi.items()
                ),
                key=lambda r: -r["count"],
            )[:25],
            "uniform_baseline_perplexity": round(float(len(vocabulary)), 2),
        }
        print(f"      {len(text):,} chars, vocab {len(vocabulary)}, "
              f"train {split_at:,} / val {len(encoded) - split_at:,}")

        def get_batch(source, batch_size: int = BATCH_SIZE):
            starts = torch.randint(len(source) - CONTEXT - 1, (batch_size,))
            x = torch.stack([source[s : s + CONTEXT] for s in starts])
            y = torch.stack([source[s + 1 : s + CONTEXT + 1] for s in starts])
            return x.to(device), y.to(device)

        # ==============================================================================
        # Train
        # ==============================================================================
        model = NanoTransformer(len(vocabulary)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        # The prose above quotes EXPECTED_PARAMS, written before the model exists. If the
        # two disagree the documentation is wrong, so fail rather than publish it.
        if abs(n_params - EXPECTED_PARAMS) > 0.02 * n_params:
            raise AssertionError(
                f"EXPECTED_PARAMS ({EXPECTED_PARAMS:,}) disagrees with the constructed "
                f"model ({n_params:,}). Update the derivation before publishing."
            )
        n_params_unique = n_params - model.head.weight.numel() * 0  # tied, counted once
        print(f"\n[2/4] Training {n_params:,} parameters for {MAX_STEPS} steps on CPU …")

        optimiser = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.95)
        )

        def lr_at(step: int) -> float:
            """Linear warmup then cosine decay.

            Warmup matters even with pre-norm: Adam's second-moment estimate is unreliable
            in the first few dozen steps, so a full-size update then can move weights into
            a region the model never recovers from.
            """
            if step < WARMUP_STEPS:
                return LEARNING_RATE * step / max(1, WARMUP_STEPS)
            progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
            return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))

        @torch.no_grad()
        def estimate_loss() -> dict:
            model.eval()
            out = {}
            for name, source in (("train", train_data), ("val", val_data)):
                losses = torch.zeros(EVAL_BATCHES)
                for i in range(EVAL_BATCHES):
                    x, y = get_batch(source)
                    _, loss = model(x, y)
                    losses[i] = loss.item()
                out[name] = float(losses.mean())
            model.train()
            return out

        history = []
        started = time.time()
        model.train()
        for step in range(MAX_STEPS + 1):
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step)

            x, y = get_batch(train_data)
            _, loss = model(x, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            if step % EVAL_EVERY == 0:
                evaluated = estimate_loss()
                history.append(
                    {
                        "step": step,
                        "train_loss": round(evaluated["train"], 4),
                        "val_loss": round(evaluated["val"], 4),
                        "train_perplexity": round(math.exp(evaluated["train"]), 3),
                        "val_perplexity": round(math.exp(evaluated["val"]), 3),
                        "learning_rate": round(lr_at(step), 7),
                        "grad_norm": round(float(grad_norm), 4),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                )
                print(f"      step {step:>5}  train {evaluated['train']:.4f}  "
                      f"val {evaluated['val']:.4f}  "
                      f"ppl {math.exp(evaluated['val']):.2f}")

        train_seconds = time.time() - started
        final = history[-1]
        best_val = min(history, key=lambda h: h["val_loss"])
        print(f"      trained in {train_seconds:.0f}s, best val perplexity "
              f"{best_val['val_perplexity']}")

        # ==============================================================================
        # Evaluate
        # ==============================================================================
        print("\n[3/4] Evaluating …")

        vocab_words = set(w.strip(".,;:!?'\"-\n").lower() for w in text.split())
        vocab_words.discard("")

        samples = []
        for temperature in (0.5, 0.8, 1.0, 1.3):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, 400, temperature=temperature, top_k=40)
            output = "".join(itos[int(i)] for i in generated[0].tolist())

            words = [w.strip(".,;:!?'\"-\n").lower() for w in output.split()]
            words = [w for w in words if w]
            real = sum(1 for w in words if w in vocab_words)

            samples.append(
                {
                    "temperature": temperature,
                    "text": output,
                    "n_words": len(words),
                    "real_word_rate": round(real / max(1, len(words)), 4),
                    "mean_line_length": round(
                        float(np.mean([len(line) for line in output.split("\n") if line]))
                        if output.strip()
                        else 0.0,
                        1,
                    ),
                }
            )
            print(f"      T={temperature}: {real}/{len(words)} real words "
                  f"({real / max(1, len(words)):.1%})")

        # Baselines the model must beat for its perplexity to mean anything.
        counts = np.bincount(encoded.numpy(), minlength=len(vocabulary)).astype(float)
        probabilities = counts / counts.sum()
        unigram_entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())

        baselines = {
            "uniform": {
                "perplexity": round(float(len(vocabulary)), 3),
                "description": "every character equally likely",
            },
            "unigram": {
                "perplexity": round(float(math.exp(unigram_entropy)), 3),
                "description": "characters drawn from the corpus frequency distribution",
            },
            "model": {
                "perplexity": best_val["val_perplexity"],
                "description": f"{n_params:,}-parameter transformer, held-out",
            },
        }
        improvement = baselines["unigram"]["perplexity"] / baselines["model"]["perplexity"]
        print(f"      perplexity: uniform {baselines['uniform']['perplexity']} → "
              f"unigram {baselines['unigram']['perplexity']} → "
              f"model {baselines['model']['perplexity']}")

        # Attention introspection: how far back does each head actually look?
        model.eval()
        with torch.no_grad():
            probe, _ = get_batch(val_data, batch_size=1)
            x_probe = model.embed(probe)
            attention_spans = []
            for layer_index, block in enumerate(model.blocks):
                normed = block.norm1(x_probe)
                qkv = block.attn.qkv(normed).chunk(3, dim=-1)
                q, k, _ = (
                    t.view(1, CONTEXT, N_HEADS, D_MODEL // N_HEADS).transpose(1, 2)
                    for t in qkv
                )
                cos, sin = block.attn.rotary(CONTEXT, device)
                q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_MODEL // N_HEADS)
                mask = torch.triu(torch.ones(CONTEXT, CONTEXT, dtype=torch.bool), 1)
                scores = scores.masked_fill(mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)[0]

                positions = torch.arange(CONTEXT).float()
                for head in range(N_HEADS):
                    w = weights[head]
                    distances = (positions[None, :] - positions[:, None]).abs()
                    mean_distance = float((w * distances).sum(-1).mean())
                    attention_spans.append(
                        {
                            "layer": layer_index,
                            "head": head,
                            "mean_attention_distance": round(mean_distance, 2),
                        }
                    )
                x_probe = block(x_probe)

        local_heads = sum(1 for h in attention_spans if h["mean_attention_distance"] < 10)

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{len(text):,} characters, {len(vocabulary)} distinct symbols. A "
                    f"uniform guess gives perplexity {len(vocabulary)}; the character "
                    f"frequency distribution alone gives "
                    f"{baselines['unigram']['perplexity']}. Those two numbers bound what "
                    "'learning something' has to mean."
                ),
                evidence=corpus,
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Character-level tokenisation with no preprocessing: casing, "
                    "punctuation and line breaks are all retained, because dramatic layout "
                    "is part of what the model should learn. Contiguous 90/10 split."
                ),
                evidence={
                    "tokenisation": "character-level, 65 symbols",
                    "split": corpus["split_rationale"],
                },
                decisions=[
                    Decision(
                        question="Character-level or subword tokenisation?",
                        choice="Character-level.",
                        rationale=(
                            "A 65-symbol vocabulary keeps the embedding and output layers "
                            "negligible, so nearly all parameters sit in the transformer "
                            "blocks where the interesting behaviour is. It also makes "
                            "spelling an observable achievement rather than something the "
                            "tokeniser handles invisibly."
                        ),
                        alternatives_rejected=[
                            "BPE — better perplexity per compute, but the model would "
                            "never have to learn to spell, removing the clearest evidence "
                            "of what it learned.",
                        ],
                    ),
                ],
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    f"{N_LAYERS}-layer pre-norm decoder, {n_params:,} parameters: rotary "
                    f"position embeddings, {N_HEADS}-head causal attention, SwiGLU "
                    "feed-forward, tied embeddings. AdamW with linear warmup and cosine "
                    "decay, gradient clipping at 1.0."
                ),
                decisions=[
                    Decision(
                        question="Rotary or learned absolute position embeddings?",
                        choice="Rotary (RoPE).",
                        rationale=(
                            "Rotating q and k by an angle proportional to position makes "
                            "the attention logit depend on relative offset, since the dot "
                            "product of two rotated vectors is a function of their angle "
                            "difference. Nothing is learned, and no absolute index exists "
                            "for the model to overfit to."
                        ),
                        alternatives_rejected=[
                            "Learned absolute embeddings — a parameter per position, and "
                            "no generalisation past the trained context length.",
                            "Sinusoidal — relative-ish, but added to the residual stream "
                            "rather than applied where attention actually uses position.",
                        ],
                    ),
                    Decision(
                        question="Pre-norm or post-norm residual blocks?",
                        choice="Pre-norm.",
                        rationale=(
                            "Post-norm routes the residual through the normaliser, so "
                            "gradient magnitude depends on depth and warmup becomes "
                            "mandatory for stability. Pre-norm keeps an unobstructed "
                            "residual path, which is why every modern decoder uses it."
                        ),
                    ),
                ],
                evidence={
                    "n_parameters": n_params,
                    "architecture": {
                        "context": CONTEXT, "d_model": D_MODEL, "n_heads": N_HEADS,
                        "n_layers": N_LAYERS, "d_ff": D_FF, "dropout": DROPOUT,
                        "position": "rotary", "ffn": "SwiGLU", "norm": "pre-norm LayerNorm",
                        "weight_tying": True,
                    },
                    "optimisation": {
                        "optimiser": "AdamW", "lr": LEARNING_RATE, "warmup": WARMUP_STEPS,
                        "schedule": "cosine", "weight_decay": 0.1, "grad_clip": 1.0,
                        "batch_size": BATCH_SIZE, "steps": MAX_STEPS,
                    },
                    "train_seconds": round(train_seconds, 1),
                },
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"Held-out perplexity {best_val['val_perplexity']} against "
                    f"{baselines['unigram']['perplexity']} for a unigram model and "
                    f"{baselines['uniform']['perplexity']} for uniform guessing — a "
                    f"{improvement:.1f}× improvement over the frequency baseline. "
                    f"{samples[1]['real_word_rate']:.0%} of generated words at T=0.8 are "
                    "real words from the corpus."
                ),
                evidence={
                    "baselines": baselines,
                    "improvement_over_unigram": round(improvement, 2),
                    "final_train_loss": final["train_loss"],
                    "final_val_loss": final["val_loss"],
                    "overfit_gap": round(final["val_loss"] - final["train_loss"], 4),
                    "attention_spans": attention_spans,
                    "local_heads": local_heads,
                    "total_heads": len(attention_spans),
                },
                risks=[
                    "The model learns orthography and dramatic layout, not meaning. "
                    "Generated text has correct-looking speaker labels and plausible "
                    "English morphology while being semantically empty. Real-word rate "
                    "measures the former and says nothing about the latter.",
                    f"Train and validation loss differ by "
                    f"{final['val_loss'] - final['train_loss']:.4f}. "
                    + (
                        "That gap indicates the model has begun memorising and a larger "
                        "corpus or stronger regularisation would be needed to train longer."
                        if final["val_loss"] - final["train_loss"] > 0.1
                        else "The gap is small, so capacity rather than overfitting is the "
                        "binding constraint at this budget."
                    ),
                    "Perplexity is measured on held-out Shakespeare. It says nothing about "
                    "performance on any other kind of text.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Training telemetry, attention statistics and generated samples at four "
                    "temperatures are exported for the dashboard. Generation itself is not "
                    f"run in the browser — a {n_params / 1e6:.1f}M-parameter forward pass "
                    "per character is "
                    "not something to ask of a page — so pre-generated samples are shown "
                    "and labelled as such."
                ),
                evidence={"samples": len(samples)},
            )
        )

        print("\n[4/4] Writing artifacts …")
        artifacts.write(OUT / "corpus.json", corpus, context=ctx)
        artifacts.write(
            OUT / "training.json",
            {
                "history": history,
                "n_parameters": n_params,
                "train_seconds": round(train_seconds, 1),
                "architecture": {
                    "context": CONTEXT, "d_model": D_MODEL, "n_heads": N_HEADS,
                    "n_layers": N_LAYERS, "d_ff": D_FF, "dropout": DROPOUT,
                },
                "final": final,
                "best_val": best_val,
            },
            context=ctx,
        )
        artifacts.write(
            OUT / "evaluation.json",
            {
                "baselines": baselines,
                "improvement_over_unigram": round(improvement, 2),
                "samples": samples,
                "attention_spans": attention_spans,
            },
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("tiny_shakespeare")]},
                        context=ctx)

    print(f"\n✓ {PROJECT} complete — val perplexity {best_val['val_perplexity']}, "
          f"{improvement:.1f}× better than unigram")


if __name__ == "__main__":
    main()
