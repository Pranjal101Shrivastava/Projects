/**
 * SVG chart primitives.
 *
 * Written directly rather than pulled from a charting library for three reasons that
 * actually matter here: the published page must load no external scripts, the artifacts
 * arrive pre-binned so the "data wrangling" a chart library provides is unnecessary, and
 * the charts need to inherit the theme tokens so they work in both light and dark without
 * a second palette definition.
 *
 * Every component is responsive: an explicit viewBox with preserveAspectRatio scales the
 * drawing to its container, so no resize observers or layout measurement is needed.
 */

import { useCallback, useMemo, useState } from "react";

const PALETTE = [
  "var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)",
  "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)",
];

export const seriesColor = (i: number) => PALETTE[i % PALETTE.length];

type Margin = { top: number; right: number; bottom: number; left: number };
const DEFAULT_MARGIN: Margin = { top: 12, right: 16, bottom: 34, left: 50 };

/** Produce ~`count` round tick values spanning [min, max]. */
function ticks(min: number, max: number, count = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const span = max - min;
  const rawStep = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalised = rawStep / magnitude;
  const step =
    (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

/** Compact number formatting that stays readable at axis-label size. */
export function fmt(v: number, digits = 2): string {
  if (!isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (abs >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (abs >= 1e4) return (v / 1e3).toFixed(0) + "k";
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(Math.min(digits, 2));
  if (abs === 0) return "0";
  if (abs < 0.001) return v.toExponential(1);
  return v.toFixed(digits + 1);
}

/* ------------------------------------------------------------------------ */
/* Tooltip                                                                    */
/* ------------------------------------------------------------------------ */

function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const show = useCallback((e: React.MouseEvent, text: string) => {
    setTip({ x: e.clientX + 12, y: e.clientY + 12, text });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const node = tip ? (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
      {tip.text}
    </div>
  ) : null;
  return { show, hide, node };
}

/* ------------------------------------------------------------------------ */
/* Line chart                                                                 */
/* ------------------------------------------------------------------------ */

export type Series = {
  name: string;
  points: { x: number; y: number }[];
  color?: string;
  dashed?: boolean;
  width?: number;
};

export function LineChart({
  series,
  height = 240,
  xLabel,
  yLabel,
  yMin,
  yMax,
  xMin,
  xMax,
  markers,
  logY = false,
  showLegend = true,
  areaUnder = false,
}: {
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  yMin?: number;
  yMax?: number;
  xMin?: number;
  xMax?: number;
  markers?: { x?: number; y?: number; label: string; color?: string }[];
  logY?: boolean;
  showLegend?: boolean;
  areaUnder?: boolean;
}) {
  const W = 760;
  const H = height;
  const m = DEFAULT_MARGIN;
  const { show, hide, node } = useTooltip();

  const all = series.flatMap((s) => s.points).filter((p) => isFinite(p.x) && isFinite(p.y));
  if (all.length === 0) return <div className="faint small">No data</div>;

  const x0 = xMin ?? Math.min(...all.map((p) => p.x));
  const x1 = xMax ?? Math.max(...all.map((p) => p.x));
  const rawY0 = yMin ?? Math.min(...all.map((p) => p.y));
  const rawY1 = yMax ?? Math.max(...all.map((p) => p.y));
  // Pad the y-range slightly so lines never sit exactly on the frame.
  const pad = (rawY1 - rawY0) * 0.06 || 1;
  const y0 = yMin ?? (logY ? Math.max(1e-6, rawY0) : rawY0 - pad);
  const y1 = yMax ?? rawY1 + pad;

  const sx = (v: number) => m.left + ((v - x0) / (x1 - x0 || 1)) * (W - m.left - m.right);
  const sy = (v: number) => {
    if (logY) {
      const lv = Math.log10(Math.max(1e-9, v));
      const l0 = Math.log10(Math.max(1e-9, y0));
      const l1 = Math.log10(Math.max(1e-9, y1));
      return H - m.bottom - ((lv - l0) / (l1 - l0 || 1)) * (H - m.top - m.bottom);
    }
    return H - m.bottom - ((v - y0) / (y1 - y0 || 1)) * (H - m.top - m.bottom);
  };

  const xTicks = ticks(x0, x1, 6);
  const yTicks = logY
    ? [y0, Math.sqrt(y0 * y1), y1].filter((v) => isFinite(v))
    : ticks(y0, y1, 5);

  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           style={{ height }} role="img">
        {yTicks.map((t, i) => (
          <g key={`y${i}`}>
            <line className="grid-line" x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} />
            <text x={m.left - 7} y={sy(t)} textAnchor="end" dominantBaseline="middle">
              {fmt(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text key={`x${i}`} x={sx(t)} y={H - m.bottom + 15} textAnchor="middle">
            {fmt(t)}
          </text>
        ))}
        <line className="axis-line" x1={m.left} x2={W - m.right} y1={H - m.bottom} y2={H - m.bottom} />
        <line className="axis-line" x1={m.left} x2={m.left} y1={m.top} y2={H - m.bottom} />

        {markers?.map((mk, i) =>
          mk.y !== undefined ? (
            <g key={`mk${i}`}>
              <line x1={m.left} x2={W - m.right} y1={sy(mk.y)} y2={sy(mk.y)}
                    stroke={mk.color ?? "var(--text-faint)"} strokeDasharray="4 4" strokeWidth={1} />
              <text x={W - m.right - 3} y={sy(mk.y) - 4} textAnchor="end"
                    fill={mk.color ?? "var(--text-faint)"} fontSize={10}>
                {mk.label}
              </text>
            </g>
          ) : mk.x !== undefined ? (
            <g key={`mk${i}`}>
              <line x1={sx(mk.x)} x2={sx(mk.x)} y1={m.top} y2={H - m.bottom}
                    stroke={mk.color ?? "var(--text-faint)"} strokeDasharray="4 4" strokeWidth={1} />
              <text x={sx(mk.x) + 4} y={m.top + 10} fill={mk.color ?? "var(--text-faint)"} fontSize={10}>
                {mk.label}
              </text>
            </g>
          ) : null
        )}

        {series.map((s, si) => {
          const pts = s.points.filter((p) => isFinite(p.x) && isFinite(p.y));
          if (!pts.length) return null;
          const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join(" ");
          const colour = s.color ?? seriesColor(si);
          return (
            <g key={s.name}>
              {areaUnder && (
                <path
                  d={`${d} L${sx(pts[pts.length - 1].x).toFixed(2)},${H - m.bottom} L${sx(pts[0].x).toFixed(2)},${H - m.bottom} Z`}
                  fill={colour}
                  opacity={0.1}
                />
              )}
              <path d={d} fill="none" stroke={colour}
                    strokeWidth={s.width ?? 2}
                    strokeDasharray={s.dashed ? "5 4" : undefined}
                    strokeLinejoin="round" strokeLinecap="round" />
              {pts.length <= 40 &&
                pts.map((p, i) => (
                  <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} fill={colour}
                          onMouseMove={(e) => show(e, `${s.name}\n${fmt(p.x)} → ${fmt(p.y, 3)}`)}
                          onMouseLeave={hide} style={{ cursor: "crosshair" }} />
                ))}
            </g>
          );
        })}

        {yLabel && (
          <text className="axis-title" transform={`rotate(-90) translate(${-(H / 2)} 12)`} textAnchor="middle">
            {yLabel}
          </text>
        )}
        {xLabel && (
          <text className="axis-title" x={(W + m.left) / 2} y={H - 2} textAnchor="middle">
            {xLabel}
          </text>
        )}
      </svg>
      {showLegend && series.length > 1 && (
        <div className="legend">
          {series.map((s, i) => (
            <span className="legend-item" key={s.name}>
              <span className="legend-swatch" style={{ background: s.color ?? seriesColor(i) }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
      {node}
    </>
  );
}

/* ------------------------------------------------------------------------ */
/* Bar chart                                                                  */
/* ------------------------------------------------------------------------ */

export function BarChart({
  data,
  height = 240,
  horizontal = false,
  xLabel,
  yLabel,
  colorBy,
  valueFormat = (v: number) => fmt(v, 3),
}: {
  data: { label: string; value: number; color?: string }[];
  height?: number;
  horizontal?: boolean;
  xLabel?: string;
  yLabel?: string;
  colorBy?: (d: { label: string; value: number }, i: number) => string;
  valueFormat?: (v: number) => string;
}) {
  const W = 760;
  const H = height;
  const { show, hide, node } = useTooltip();
  if (!data.length) return <div className="faint small">No data</div>;

  const maxValue = Math.max(...data.map((d) => Math.abs(d.value)), 1e-9);

  if (horizontal) {
    const m = { top: 6, right: 56, bottom: 24, left: 148 };
    const rowHeight = (H - m.top - m.bottom) / data.length;
    return (
      <>
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
             style={{ height }} role="img">
          {data.map((d, i) => {
            const w = (Math.abs(d.value) / maxValue) * (W - m.left - m.right);
            const y = m.top + i * rowHeight;
            const colour = d.color ?? colorBy?.(d, i) ?? "var(--c1)";
            return (
              <g key={d.label}
                 onMouseMove={(e) => show(e, `${d.label}\n${valueFormat(d.value)}`)}
                 onMouseLeave={hide}>
                <text x={m.left - 8} y={y + rowHeight / 2} textAnchor="end" dominantBaseline="middle">
                  {d.label.length > 22 ? d.label.slice(0, 21) + "…" : d.label}
                </text>
                <rect x={m.left} y={y + rowHeight * 0.16} width={Math.max(1, w)}
                      height={rowHeight * 0.68} fill={colour} rx={2} />
                <text x={m.left + w + 6} y={y + rowHeight / 2} dominantBaseline="middle" fontSize={10}>
                  {valueFormat(d.value)}
                </text>
              </g>
            );
          })}
          <line className="axis-line" x1={m.left} x2={m.left} y1={m.top} y2={H - m.bottom} />
        </svg>
        {node}
      </>
    );
  }

  const m = { top: 12, right: 12, bottom: 46, left: 50 };
  const colWidth = (W - m.left - m.right) / data.length;
  const yTicks = ticks(0, maxValue, 4);
  const sy = (v: number) => H - m.bottom - (v / maxValue) * (H - m.top - m.bottom);
  const skip = Math.ceil(data.length / 26);

  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           style={{ height }} role="img">
        {yTicks.map((t, i) => (
          <g key={i}>
            <line className="grid-line" x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} />
            <text x={m.left - 7} y={sy(t)} textAnchor="end" dominantBaseline="middle">{fmt(t)}</text>
          </g>
        ))}
        {data.map((d, i) => {
          const h = (Math.abs(d.value) / maxValue) * (H - m.top - m.bottom);
          const colour = d.color ?? colorBy?.(d, i) ?? "var(--c1)";
          return (
            <g key={i} onMouseMove={(e) => show(e, `${d.label}\n${valueFormat(d.value)}`)}
               onMouseLeave={hide}>
              <rect x={m.left + i * colWidth + colWidth * 0.12} y={sy(Math.abs(d.value))}
                    width={colWidth * 0.76} height={Math.max(1, h)} fill={colour} rx={2}
                    style={{ cursor: "crosshair" }} />
              {i % skip === 0 && (
                <text x={m.left + i * colWidth + colWidth / 2} y={H - m.bottom + 14}
                      textAnchor="middle">
                  {d.label.length > 8 ? d.label.slice(0, 7) + "…" : d.label}
                </text>
              )}
            </g>
          );
        })}
        <line className="axis-line" x1={m.left} x2={W - m.right} y1={H - m.bottom} y2={H - m.bottom} />
        {yLabel && (
          <text className="axis-title" transform={`rotate(-90) translate(${-(H / 2)} 12)`} textAnchor="middle">
            {yLabel}
          </text>
        )}
        {xLabel && <text className="axis-title" x={W / 2} y={H - 4} textAnchor="middle">{xLabel}</text>}
      </svg>
      {node}
    </>
  );
}

/* ------------------------------------------------------------------------ */
/* Scatter                                                                    */
/* ------------------------------------------------------------------------ */

export function ScatterChart({
  points,
  height = 320,
  xLabel,
  yLabel,
  radius = 2.4,
  opacity = 0.62,
  categories,
}: {
  points: { x: number; y: number; c?: number; label?: string }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  radius?: number;
  opacity?: number;
  categories?: string[];
}) {
  const W = 760;
  const H = height;
  const m = DEFAULT_MARGIN;
  const { show, hide, node } = useTooltip();
  if (!points.length) return <div className="faint small">No data</div>;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padX = (x1 - x0) * 0.04 || 1;
  const padY = (y1 - y0) * 0.04 || 1;

  const sx = (v: number) => m.left + ((v - x0 + padX) / (x1 - x0 + 2 * padX || 1)) * (W - m.left - m.right);
  const sy = (v: number) => H - m.bottom - ((v - y0 + padY) / (y1 - y0 + 2 * padY || 1)) * (H - m.top - m.bottom);

  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           style={{ height }} role="img">
        {ticks(y0, y1, 5).map((t, i) => (
          <g key={i}>
            <line className="grid-line" x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} />
            <text x={m.left - 7} y={sy(t)} textAnchor="end" dominantBaseline="middle">{fmt(t)}</text>
          </g>
        ))}
        {ticks(x0, x1, 6).map((t, i) => (
          <text key={i} x={sx(t)} y={H - m.bottom + 15} textAnchor="middle">{fmt(t)}</text>
        ))}
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={radius}
                  fill={seriesColor(p.c ?? 0)} opacity={opacity}
                  onMouseMove={(e) =>
                    show(e, p.label ?? `${fmt(p.x, 3)}, ${fmt(p.y, 3)}`)
                  }
                  onMouseLeave={hide} />
        ))}
        <line className="axis-line" x1={m.left} x2={W - m.right} y1={H - m.bottom} y2={H - m.bottom} />
        <line className="axis-line" x1={m.left} x2={m.left} y1={m.top} y2={H - m.bottom} />
        {yLabel && (
          <text className="axis-title" transform={`rotate(-90) translate(${-(H / 2)} 12)`} textAnchor="middle">
            {yLabel}
          </text>
        )}
        {xLabel && <text className="axis-title" x={W / 2} y={H - 2} textAnchor="middle">{xLabel}</text>}
      </svg>
      {categories && (
        <div className="legend">
          {categories.map((c, i) => (
            <span className="legend-item" key={c}>
              <span className="legend-swatch" style={{ background: seriesColor(i) }} />
              {c}
            </span>
          ))}
        </div>
      )}
      {node}
    </>
  );
}

/* ------------------------------------------------------------------------ */
/* Heatmap                                                                    */
/* ------------------------------------------------------------------------ */

export function Heatmap({
  cells,
  rows,
  cols,
  height = 260,
  rowLabels,
  colLabels,
  valueLabel = "value",
}: {
  cells: { row: number; col: number; value: number }[];
  rows: number;
  cols: number;
  height?: number;
  rowLabels?: string[];
  colLabels?: string[];
  valueLabel?: string;
}) {
  const W = 760;
  const H = height;
  const m = { top: 8, right: 12, bottom: 30, left: 52 };
  const { show, hide, node } = useTooltip();

  const values = cells.map((c) => c.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const cw = (W - m.left - m.right) / cols;
  const ch = (H - m.top - m.bottom) / rows;

  // Sequential ramp built in HSL so it interpolates smoothly and reads in both themes.
  const colourFor = (v: number) => {
    const t = (v - lo) / (hi - lo || 1);
    const light = 16 + t * 46;
    const sat = 34 + t * 52;
    const hue = 190 - t * 30;
    return `hsl(${hue} ${sat}% ${light}%)`;
  };

  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
           style={{ height }} role="img">
        {cells.map((c, i) => (
          <rect key={i} x={m.left + c.col * cw} y={m.top + c.row * ch}
                width={cw - 1} height={ch - 1} fill={colourFor(c.value)} rx={1.5}
                onMouseMove={(e) =>
                  show(e,
                    `${rowLabels?.[c.row] ?? `row ${c.row}`} · ${colLabels?.[c.col] ?? `col ${c.col}`}\n${valueLabel}: ${fmt(c.value, 2)}`)
                }
                onMouseLeave={hide} style={{ cursor: "crosshair" }} />
        ))}
        {rowLabels?.map((label, r) => (
          <text key={r} x={m.left - 7} y={m.top + r * ch + ch / 2} textAnchor="end"
                dominantBaseline="middle">
            {label}
          </text>
        ))}
        {colLabels?.map((label, c) =>
          c % Math.ceil(cols / 24) === 0 ? (
            <text key={c} x={m.left + c * cw + cw / 2} y={H - m.bottom + 14} textAnchor="middle">
              {label}
            </text>
          ) : null
        )}
      </svg>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colourFor(lo) }} /> {fmt(lo)}
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colourFor((lo + hi) / 2) }} /> {fmt((lo + hi) / 2)}
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colourFor(hi) }} /> {fmt(hi)}
        </span>
      </div>
      {node}
    </>
  );
}

/* ------------------------------------------------------------------------ */
/* Histogram (from the pre-binned artifact shape)                             */
/* ------------------------------------------------------------------------ */

export type HistogramData = {
  bin_centers: number[];
  counts: number[];
  mean?: number;
  median?: number;
  n?: number;
};

export function Histogram({
  data,
  height = 220,
  xLabel,
  color = "var(--c1)",
  showMean = true,
}: {
  data: HistogramData;
  height?: number;
  xLabel?: string;
  color?: string;
  showMean?: boolean;
}) {
  const points = useMemo(
    () => data.bin_centers.map((c, i) => ({ x: c, y: data.counts[i] })),
    [data]
  );
  return (
    <LineChart
      series={[{ name: "count", points, color }]}
      height={height}
      xLabel={xLabel}
      yLabel="count"
      areaUnder
      showLegend={false}
      markers={
        showMean && data.mean !== undefined
          ? [{ x: data.mean, label: `mean ${fmt(data.mean)}`, color: "var(--warn)" }]
          : undefined
      }
    />
  );
}

/* ------------------------------------------------------------------------ */
/* Confusion matrix                                                           */
/* ------------------------------------------------------------------------ */

export function ConfusionMatrix({
  tp, fp, fn, tn, positiveLabel = "Positive", negativeLabel = "Negative",
}: {
  tp: number; fp: number; fn: number; tn: number;
  positiveLabel?: string; negativeLabel?: string;
}) {
  const cell = (value: number, kind: "tp" | "fp" | "fn" | "tn") => {
    const good = kind === "tp" || kind === "tn";
    return (
      <td className="num" style={{
        background: good ? "rgba(52,211,153,0.10)" : "rgba(248,113,113,0.10)",
        fontWeight: 600,
      }}>
        {value.toLocaleString()}
        <div className="tiny faint" style={{ fontWeight: 400 }}>{kind.toUpperCase()}</div>
      </td>
    );
  };
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th></th>
            <th className="num">Predicted {negativeLabel}</th>
            <th className="num">Predicted {positiveLabel}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Actual {negativeLabel}</th>
            {cell(tn, "tn")}
            {cell(fp, "fp")}
          </tr>
          <tr>
            <th>Actual {positiveLabel}</th>
            {cell(fn, "fn")}
            {cell(tp, "tp")}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
