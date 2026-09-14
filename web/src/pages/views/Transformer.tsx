import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, Heatmap, LineChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Training = {
  history: {
    step: number; train_loss: number; val_loss: number;
    train_perplexity: number; val_perplexity: number;
    learning_rate: number; grad_norm: number; elapsed_seconds: number;
  }[];
  n_parameters: number; train_seconds: number;
  architecture: { context: number; d_model: number; n_heads: number; n_layers: number; d_ff: number; dropout: number };
  final: { train_loss: number; val_loss: number; val_perplexity: number };
  best_val: { step: number; val_loss: number; val_perplexity: number };
};

type Evaluation = {
  baselines: Record<string, { perplexity: number; description: string }>;
  improvement_over_unigram: number;
  samples: { temperature: number; text: string; n_words: number; real_word_rate: number; mean_line_length: number }[];
  attention_spans: { layer: number; head: number; mean_attention_distance: number }[];
};

type Corpus = {
  n_characters: number; vocab_size: number; vocabulary: string;
  train_characters: number; val_characters: number; split_rationale: string;
  char_frequencies: { char: string; count: number }[];
  uniform_baseline_perplexity: number;
};

export default function Transformer() {
  const state = useArtifacts<{ training: Training; evaluation: Evaluation; corpus: Corpus }>(
    "07_nano_transformer", ["training", "evaluation", "corpus"]
  );
  return (
    <Resolved state={state} what="transformer artifacts">
      {(d) => <Body training={d.training} evaluation={d.evaluation} corpus={d.corpus} />}
    </Resolved>
  );
}

function Body({ training, evaluation, corpus }: { training: Training; evaluation: Evaluation; corpus: Corpus }) {
  const [temperature, setTemperature] = useState(1);
  const sample = evaluation.samples[temperature];
  const arch = training.architecture;
  const nHeads = arch.n_heads;
  const nLayers = arch.n_layers;

  const overfitGap = training.final.val_loss - training.final.train_loss;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Parameters" value={training.n_parameters.toLocaleString()}
              sub={`${nLayers} layers × ${nHeads} heads`} />
        <Stat label="Held-out perplexity" value={fmt(training.best_val.val_perplexity, 2)}
              tone="accent" sub={`at step ${training.best_val.step}`} />
        <Stat label="vs unigram baseline" value={`${fmt(evaluation.improvement_over_unigram, 1)}×`}
              tone="good" sub={`unigram ${fmt(evaluation.baselines.unigram.perplexity, 2)}`} />
        <Stat label="CPU training time" value={`${Math.round(training.train_seconds)}s`}
              sub="no GPU used" />
      </div>

      <Callout kind="info" title="What this model actually learned">
        A {(training.n_parameters / 1e6).toFixed(2)}M-parameter model on 1.1M characters
        learns <strong>orthography and dramatic form</strong> — which letter sequences are
        English-shaped, how speaker labels and line breaks are laid out. It does not learn
        meaning. The evaluation below measures held-out perplexity against real baselines and
        the proportion of generated words that are genuine, rather than displaying one
        cherry-picked paragraph that happens to read well.
      </Callout>

      <Section title="Perplexity against baselines" note="what 'learning something' has to beat">
        <div className="grid grid-3">
          {Object.entries(evaluation.baselines).map(([key, b]) => (
            <div className="card" key={key}>
              <div className="stat-label">{key}</div>
              <div className={`stat-value${key === "model" ? " accent" : ""}`}>
                {fmt(b.perplexity, 2)}
              </div>
              <div className="stat-sub">{b.description}</div>
            </div>
          ))}
        </div>
        <p className="small dim" style={{ marginTop: 12 }}>
          A uniform guess over {corpus.vocab_size} symbols gives perplexity{" "}
          {fmt(evaluation.baselines.uniform.perplexity, 0)}. Knowing only the character
          frequency distribution gives {fmt(evaluation.baselines.unigram.perplexity, 2)}.
          Those two numbers bound what the transformer had to beat for its result to mean
          anything — and it reaches {fmt(evaluation.baselines.model.perplexity, 2)}.
        </p>
      </Section>

      <Section title="Training dynamics">
        <div className="grid grid-2">
          <div className="card">
            <h3>Loss</h3>
            <LineChart
              height={250}
              xLabel="step"
              yLabel="cross-entropy"
              series={[
                { name: "train", points: training.history.map((h) => ({ x: h.step, y: h.train_loss })) },
                { name: "validation", points: training.history.map((h) => ({ x: h.step, y: h.val_loss })), color: seriesColor(2) },
              ]}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Final gap {fmt(overfitGap, 4)}.{" "}
              {overfitGap > 0.15
                ? "The model has begun memorising; a larger corpus or stronger regularisation would be needed to train longer."
                : "Small, so capacity rather than overfitting is the binding constraint at this budget."}
            </p>
          </div>
          <div className="card">
            <h3>Learning rate and gradient norm</h3>
            <LineChart
              height={250}
              xLabel="step"
              series={[
                {
                  name: "learning rate (×10⁴)",
                  points: training.history.map((h) => ({ x: h.step, y: h.learning_rate * 1e4 })),
                },
                {
                  name: "gradient norm", color: seriesColor(3),
                  points: training.history.map((h) => ({ x: h.step, y: h.grad_norm })),
                },
              ]}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Linear warmup then cosine decay. Warmup matters even with pre-norm blocks:
              Adam's second-moment estimate is unreliable in the first few dozen steps, and a
              full-size update then can move weights somewhere the model never recovers from.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Generated samples" note="pre-generated at four temperatures">
        <div className="card">
          <div className="controls">
            <div className="control">
              <span className="control-label">Sampling temperature</span>
              <div className="row">
                {evaluation.samples.map((s, i) => (
                  <button key={s.temperature} className={`chip${temperature === i ? " active" : ""}`}
                          onClick={() => setTemperature(i)}>
                    T = {s.temperature}
                  </button>
                ))}
              </div>
            </div>
            <div className="control">
              <div className="control-label">Real-word rate</div>
              <div className="stat-value accent" style={{ fontSize: "1.5rem" }}>
                {(sample.real_word_rate * 100).toFixed(1)}%
              </div>
              <div className="stat-sub">{sample.n_words} words generated</div>
            </div>
          </div>

          <pre className="sample">{sample.text}</pre>

          <p className="tiny dim" style={{ marginTop: 10 }}>
            Low temperature concentrates probability on the likeliest next character,
            producing repetitive but well-spelled text; high temperature flattens the
            distribution, producing more variety and more non-words. The real-word rate
            quantifies that trade-off instead of leaving it to impression. It measures
            orthography only — a sample can be 90% real words and still mean nothing.
          </p>
        </div>
      </Section>

      <Section title="Architecture" note="every component written against tensor operations">
        <div className="grid grid-2">
          <div className="card">
            <h3>Configuration</h3>
            <MetricTable
              rows={[
                { k: "Layers", v: String(arch.n_layers) },
                { k: "Attention heads", v: String(arch.n_heads) },
                { k: "Model dimension", v: String(arch.d_model) },
                { k: "Feed-forward dimension", v: String(arch.d_ff) },
                { k: "Context length", v: String(arch.context) },
                { k: "Dropout", v: String(arch.dropout) },
                { k: "Position encoding", v: "Rotary (RoPE)" },
                { k: "Feed-forward", v: "SwiGLU (gated)" },
                { k: "Normalisation", v: "Pre-norm LayerNorm" },
                { k: "Embeddings", v: "Tied input/output" },
              ]}
              columns={[{ key: "k", label: "" }, { key: "v", label: "Value", num: true }]}
            />
          </div>
          <div className="card">
            <h3>Why these choices</h3>
            <div className="stack small">
              <div>
                <strong className="accent-text">Rotary position embeddings.</strong>{" "}
                <span className="dim">
                  Position enters by rotating query and key vectors by an angle proportional
                  to index. Because the dot product of two rotated vectors depends only on
                  their angle difference, the attention logit becomes a function of relative
                  offset — nothing learned, and no absolute index to overfit to.
                </span>
              </div>
              <div>
                <strong className="accent-text">Pre-norm blocks.</strong>{" "}
                <span className="dim">
                  Post-norm routes the residual through the normaliser, making gradient
                  magnitude depth-dependent and warmup mandatory. Pre-norm leaves the
                  residual path clean, which is why every modern decoder uses it.
                </span>
              </div>
              <div>
                <strong className="accent-text">SwiGLU feed-forward.</strong>{" "}
                <span className="dim">
                  One projection produces values, another a gate, and their product passes
                  through. Empirically stronger than ReLU or GELU at equal parameter count.
                </span>
              </div>
              <div>
                <strong className="accent-text">Weight tying.</strong>{" "}
                <span className="dim">
                  The vector representing a character on the way in is the vector scoring it
                  on the way out. On a {corpus.vocab_size}-symbol vocabulary the parameter
                  saving is negligible; the inductive bias is the point.
                </span>
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Section title="What the attention heads look at"
               note="mean attention distance, probed on held-out text">
        <div className="card">
          <Heatmap
            rows={nLayers}
            cols={nHeads}
            height={200}
            rowLabels={Array.from({ length: nLayers }, (_, i) => `Layer ${i}`)}
            colLabels={Array.from({ length: nHeads }, (_, i) => `H${i}`)}
            valueLabel="mean distance (chars)"
            cells={evaluation.attention_spans.map((a) => ({
              row: a.layer, col: a.head, value: a.mean_attention_distance,
            }))}
          />
          <p className="tiny dim" style={{ marginTop: 10 }}>
            Each cell is how far back, on average, that head's attention mass falls. Heads
            with short spans are doing local work — character bigrams, word boundaries — while
            long-span heads carry context across a line or more. The specialisation emerges
            from training; nothing in the architecture assigns roles to heads.
          </p>
        </div>
      </Section>

      <Section title="Corpus">
        <div className="grid grid-2">
          <div className="card">
            <h3>Character frequencies</h3>
            <BarChart
              horizontal
              height={280}
              data={corpus.char_frequencies.slice(0, 16).map((c) => ({
                label: c.char === " " ? "␣ (space)" : c.char === "\\n" ? "⏎ (newline)" : c.char,
                value: c.count,
              }))}
              valueFormat={(v) => v.toLocaleString()}
            />
          </div>
          <div className="card">
            <h3>Split</h3>
            <div className="grid grid-2">
              <Stat label="Training characters" value={corpus.train_characters.toLocaleString()} />
              <Stat label="Validation characters" value={corpus.val_characters.toLocaleString()} />
            </div>
            <Callout kind="info" title="Contiguous, not random">
              {corpus.split_rationale}
            </Callout>
            <p className="tiny faint mono" style={{ wordBreak: "break-all", marginTop: 10 }}>
              vocabulary ({corpus.vocab_size}): {corpus.vocabulary.replace(/\n/g, "⏎")}
            </p>
          </div>
        </div>
      </Section>

      <Caveat>
        The model learns orthography and dramatic layout, not meaning — generated text has
        plausible speaker labels and English morphology while being semantically empty, and
        the real-word rate measures only the former. Perplexity is also measured on held-out
        Shakespeare and says nothing about performance on any other kind of text. Generation
        runs pre-computed rather than live: a {(training.n_parameters / 1e6).toFixed(1)}M-parameter
        forward pass per character is not something to ask of a browser tab.
      </Caveat>
    </>
  );
}
