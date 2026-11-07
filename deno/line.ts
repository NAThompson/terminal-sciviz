/**
 * 2D line view: series of (x, y) drawn by the WebGPU curve renderer (`NurbsGpuRenderer` of
 * lib/gpu_line.ts, as plot1d.ts): linear series as degree-1 B-splines, PCHIP (`getPchipSlopes`)
 * and not-a-knot splines (`getSplineSlopes`) as exact cubic Bézier NURBS, all in axis space (log,
 * symlog), with axes and ticks of lib/scales.ts, envelopes, reference lines and bands,
 * unplottable marks, the legend in the corner `getLegendCorner` picks, and a hover read-out.
 * Same CLI and serve protocol as plot1d.ts.
 *
 *   deno run -A line.ts png CONTROL.json -o out.png
 *   deno run -A line.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 *
 * Each series is drawn opaque (its curves by the renderers into one scratch layer, so their
 * chords don't add up), then composited into the frame's curve layer in the series' order, with
 * its opacity, and for dashed ones through its own mask: vertical strips along its length on
 * screen (6 px on, 4 off; exact for runs monotonic in x). Non-finite points split series, as do
 * repeated abscissas; cubics are also split where the abscissas reverse (the slope helpers give
 * slope 0 there), descending runs reversed.
 */

import { type AxisSpec, axisTicks, axisTransform, formatNumber } from "./lib/scales.ts";
import { getKnotSpans, getNurbsError, getSegmentCounts, type KnotSpan, type NurbsCurve } from "./lib/nurbs.ts";
import { COPY_DST, makeReadBuffer, requestGpuDevice, rowBytes, STORAGE, TextureTarget } from "./lib/gpu.ts";
import { NurbsGpuRenderer } from "./lib/gpu_line.ts";
import { getLegendCorner, type LegendObstacle } from "./lib/legend.ts";
import { LayerCompositor } from "./lib/layers.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba } from "./lib/overlay.ts";
import { getHermiteExtent, getPchipSlopes, getSplineSlopes, hermiteAt, hermiteNurbs, linspace } from "./lib/pchip.ts";
import { ChordRenderer } from "./lib/chords.ts";
import { type FnSpec, resolveFn } from "./lib/generators.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";
import { boxStats, closestSpacing, ecdf, histogram, linreg, qq, sortedFinite } from "./lib/stats.ts";
import { type ColorBar, type Columns, type Compiled, compileMarks, type Mark, type Prims, type Scales } from "./lib/marks.ts";
import { colormapRgb } from "./lib/colormaps.ts";

// GitHub Dark (terminal theme), dataviz palette
export const PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const INK = "#c9d1d9", INK2 = "#8b949e", GRID = "#262c36";
const SURFACE = "#101216", ZERO = "#484f58", HOVER_RING = "#e6edf3";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];
const Y_MARGIN = 0.08;
const DASH = 6, GAP = 4; // CSS px
const BAND_OPACITY = 0.2; // of reference bands
const MARK_SIZE = 4; // half width of an unplottable mark
const ENVELOPE_WIDTH = 2;

export interface SeriesSpec {
  x?: number[] | null; // default: indices
  y?: number[];
  file?: string; // instead of x, y: a .json {x?, y} (or [y...]), or a 1-2 column .csv/.tsv (y, or x y); read once per mtime
  sample?: { fn: FnSpec; n: number }; // x, y = n even samples of fn over its domain
  fn?: FnSpec; // a generator (lib/generators.ts), sampled (f64) over the visible window
  label?: string;
  color?: string;
  width?: number; // CSS px
  dash?: boolean;
  markers?: boolean | "only";
  bars?: boolean | number; // a histogram: a bar per point from 0 (log y: the bottom), this wide in x (true: the closest spacing)
  opacity?: number; // < 1: in the faded layer
  interpolation?: "linear" | "spline" | "pchip" | "step"; // step: post-step (flat to the next x, then up or down)
  envelope?: { type: "cummin" | "cummax"; mode?: "add" | "replace"; reverse?: boolean; color?: string };
  legend?: boolean;
  extent?: boolean; // false: drawn, but sets neither the default x window nor the y range (reference curves)
  // Geoms. A bound of a fill: a constant, values at x (interpolated as the series), or a generator
  // (evaluated in f64 per pixel column); missing: the series' own y (or fn). No y: only the fill.
  fill?: { lo?: FillBound; hi?: FillBound; opacity?: number };
  intervals?: { lo: number[]; hi: number[]; cap?: number; label?: string }; // vertical ranges at x (error bars, whiskers); cap: CSS px wide
  box?: { q1: number[]; q3: number[]; width?: number }; // boxes q1..q3 at x, `width` in x (default 0.6), y: the middle line
  // Stats: raw `samples` (or `file`'s y; linreg: x, y or file) expanded into the series above (`expandStat`)
  samples?: number[];
  stat?: StatSpec["type"] | StatSpec;
}

type FillBound = number | number[] | FnSpec;

export interface StatSpec {
  type: "ecdf" | "qq" | "box" | "hist" | "linreg";
  band?: number; // ecdf: α of the DKW band; linreg: α of the t band (default 0.05)
  reference?: boolean; // qq: the line through the quartiles (default true)
  position?: number; // box: x (default: its index among box series)
  width?: number; // box: in x (default 0.6)
  bins?: number; // hist: count, or
  binWidth?: number; // (default: Freedman–Diaconis)
  density?: boolean; // hist: count / (n width)
  kind?: "mean" | "prediction"; // linreg band (default mean)
}

export interface OverlaySpec {
  kind: "hline" | "vline" | "band";
  axis?: "x" | "y"; // bands
  values: number[];
  label?: string;
  color?: string;
  width?: number;
  dash?: boolean;
}

export const DEFAULTS = {
  width: 1200,
  height: 750,
  title: "PCHIP · bumps · n = 32",
  subtitle: "",
  xlabel: "",
  ylabel: "",
  x: {} as AxisSpec,
  y: {} as AxisSpec,
  series: [
    { fn: "bumps", label: "f", color: INK2, width: 1.2, opacity: 0.55 },
    { sample: { fn: "bumps", n: 32 }, interpolation: "pchip", markers: true, label: "PCHIP", width: 2.5 },
  ] as SeriesSpec[],
  overlays: [] as OverlaySpec[],
  marks: [] as Mark[], // drawn after the series (lib/marks.ts)
  color: {} as NonNullable<Scales["color"]>, // marks' colour scale
  r: {} as NonNullable<Scales["r"]>, // marks' radius scale
  legend: true,
  zeroLine: true,
  nanColor: "#e66767",
  x0: null as number | null, // values (not axis space); null: all data
  x1: null as number | null,
  ymin: null as number | null, // null: fit what is visible, 8% margin (in axis space)
  ymax: null as number | null,
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

// --- Series ------------------------------------------------------------------------------------

interface Piece {
  xw: Float64Array; // axis space (runs: increasing)
  yw: Float64Array;
  slopes?: Float64Array; // cubic Hermite (PCHIP, spline) in axis space
  step?: boolean; // post-step: y[i] on [x[i], x[i + 1])
}

/** A fill bound, resolved: a constant, runs, or f64 f (all axis space out). */
type Bound = { c: number } | { runs: Piece[] } | { f: (x: number) => number; domain: [number, number] };

interface Series {
  label: string;
  color: string;
  width: number;
  dash: boolean;
  markers: boolean | "only";
  bars: number; // width in x values; 0: none
  opacity: number;
  legend: boolean;
  extent: boolean;
  x: Float64Array; // values
  y: Float64Array;
  runs: Piece[]; // increasing in x: hover, extents, dashes
  curves: NurbsCurve[]; // drawn
  fn?: (x: number) => number; // f64, for the read-out
  gen?: ReturnType<typeof resolveFn>; // sampled over the window (`ensureDense`)
  domain?: [number, number];
  fill?: { lo: Bound; hi: Bound; opacity: number };
  intervals?: { lo: Float64Array; hi: Float64Array; cap: number; label: string };
  box?: { q1: Float64Array; q3: Float64Array; width: number };
  stats?: Record<string, number>; // of a stat, in the state
  // Marks' primitives (lib/marks.ts): per-point colours, alphas, dot radii; rects, texts, rules,
  // fills between functions of y; hover rows per point
  noLine?: boolean;
  base?: boolean;
  colors?: string[];
  alphas?: number[];
  sizes?: number[];
  rects?: { x1: Float64Array; x2: Float64Array; y1: Float64Array; y2: Float64Array; inset: [number, number]; radius: number; end: "x" | "y" | null };
  texts?: Prims["texts"];
  rules?: { axis: "x" | "y"; at: Float64Array; lo: Float64Array; hi: Float64Array };
  fillY?: { lo: Piece[]; hi: Piece[]; opacity: number };
  tips?: Prims["tips"];
}

const glyphWidth = (sr: Series) => sr.bars || sr.box?.width || 0; // in x: hovered under the cursor, its edges in the extent
const bandOnly = (sr: Series) => Boolean(sr.fill) && !sr.fn && !sr.y.some(Number.isFinite);

/** Envelope of a series: running min / max skipping NaN, carried over them. */
function envelope(y: ArrayLike<number>, type: "cummin" | "cummax", reverse: boolean): Float64Array {
  const pick = type === "cummin" ? Math.min : Math.max;
  const env = new Float64Array(y.length);
  let current = NaN;
  for (let k = 0; k < y.length; k += 1) {
    const i = reverse ? y.length - 1 - k : k;
    const val = y[i];
    if (!Number.isNaN(val)) {
      current = Number.isNaN(current) ? val : pick(current, val);
    }
    env[i] = current;
  }
  return env;
}

const envelopeLabel = (type: string, reverse: boolean) => `running ${type === "cummin" ? "min" : "max"}${reverse ? " (from the end)" : ""}`;

/** Runs of consecutive points finite in axis space (non-finite points split curves), with their
 * slopes (over the whole run: 0 where the abscissas reverse). */
function pieces(x: Float64Array, y: Float64Array, fx: (v: number) => number, fy: (v: number) => number, interp: string): Piece[] {
  const out: Piece[] = [];
  let xs: number[] = [];
  let ys: number[] = [];
  const flush = () => {
    if (xs.length >= 1) {
      const piece: Piece = { xw: Float64Array.from(xs), yw: Float64Array.from(ys) };
      if (xs.length >= 2 && interp === "pchip") piece.slopes = getPchipSlopes(piece.xw, piece.yw);
      if (xs.length >= 2 && interp === "spline") piece.slopes = getSplineSlopes(piece.xw, piece.yw);
      out.push(piece);
    }
    xs = [];
    ys = [];
  };
  for (let i = 0; i < x.length; i += 1) {
    const a = fx(x[i]);
    const b = fy(y[i]);
    if (Number.isFinite(a) && Number.isFinite(b)) {
      xs.push(a);
      ys.push(b);
    } else {
      flush();
    }
  }
  flush();
  return out;
}

/** Sub-runs of a piece strictly monotonic in x, increasing (descending ones reversed): split at
 * repeated abscissas, and where they reverse (the point shared by both sides). */
function monotoneRuns(p: Piece): Piece[] {
  const n = p.xw.length;
  const out: Piece[] = [];
  const emit = (a: number, b: number) => {
    if (b <= a) return;
    let xw = p.xw.slice(a, b + 1), yw = p.yw.slice(a, b + 1), slopes = p.slopes?.slice(a, b + 1);
    if (xw[xw.length - 1] < xw[0]) {
      xw = xw.reverse();
      yw = yw.reverse();
      slopes = slopes?.reverse(); // dy/dx: unchanged by the reversal
    }
    out.push({ xw, yw, slopes, step: p.step });
  };
  let start = 0;
  let dir = 0;
  for (let i = 1; i < n; i += 1) {
    const d = Math.sign(p.xw[i] - p.xw[i - 1]);
    if (d === 0) { // repeated abscissa: split
      emit(start, i - 1);
      start = i;
      dir = 0;
    } else {
      if (dir !== 0 && d !== dir) { // reversal: both sides share point i - 1
        emit(start, i - 1);
        start = i - 1;
      }
      dir = d;
    }
  }
  emit(start, n - 1);
  return out;
}

/** What a series draws and evaluates: increasing runs, and its NURBS (a polyline through each
 * finite piece as it is, or the exact cubic of each increasing run). */
function geometry(x: Float64Array, y: Float64Array, fx: (v: number) => number, fy: (v: number) => number, interp: string): { runs: Piece[]; curves: NurbsCurve[] } {
  const ps = pieces(x, y, fx, fy, interp);
  if (interp === "step") ps.forEach((p) => (p.step = true));
  const runs = ps.flatMap(monotoneRuns);
  const curves = interp === "linear" || interp === "step"
    ? ps.filter((p) => p.xw.length >= 2).map((p) => pieceNurbs(p.step ? stepPolyline(p) : p))
    : runs.map(pieceNurbs);
  return { runs, curves };
}

/** Numbers as f64, null (JSON's NaN) as NaN: a gap. */
const toF64 = (a: ArrayLike<unknown>) => Float64Array.from(a, (u) => (u === null ? NaN : Number(u)));

function resolveSeries(ctl: Control): Series[] {
  const series = ctl.series as Compiled[];
  const fx = axisTransform(ctl.x).fwd;
  const fy = axisTransform(ctl.y).fwd;
  const out: Series[] = [];
  let auto = 0, boxes = 0;
  const manyStats = series.filter((sp) => sp.stat).length > 1;
  const nextColor = () => PALETTE[auto++ % PALETTE.length];
  for (const given of series) {
    const color = given.color ?? nextColor();
    for (const spec of expandStat(given, color, statOf(given)?.type === "box" ? boxes++ : 0, manyStats) as Compiled[]) {
      const base = {
        label: spec.label ?? "",
        color,
        width: spec.width ?? 1.5,
        dash: Boolean(spec.dash),
        markers: spec.markers ?? false,
        bars: 0,
        opacity: spec.opacity ?? 1,
        legend: spec.legend ?? Boolean(spec.label),
        extent: spec.extent ?? true,
        stats: spec.stats,
      };
      const interp = spec.interpolation ?? "linear";
      if (spec.fn) {
        const fn = resolveFn(spec.fn);
        const fill = spec.fill && resolveFill(spec.fill, { f: fn.f, domain: fn.domain }, new Float64Array(0), fx, fy, interp);
        out.push({ ...base, x: new Float64Array(0), y: new Float64Array(0), runs: [], curves: [], fn: fn.f, gen: fn, domain: fn.domain, fill });
        continue;
      }
      let x: Float64Array, y: Float64Array;
      if (spec.sample) {
        const fn = resolveFn(spec.sample.fn);
        x = linspace(fn.domain[0], fn.domain[1], Math.max(2, Math.trunc(spec.sample.n)));
        y = fn.sample(x);
      } else if (spec.file) {
        ({ x, y } = seriesFile(spec.file));
      } else {
        const n = spec.y?.length ?? spec.x?.length ?? 0;
        y = spec.y ? toF64(spec.y) : new Float64Array(n).fill(NaN); // no y: geoms only
        x = spec.x ? toF64(spec.x) : Float64Array.from(y, (_, i) => i);
      }
      const env = spec.envelope;
      if (env && (env.mode ?? "add") === "replace") {
        y = envelope(y, env.type, Boolean(env.reverse));
      }
      const geo = geometry(x, y, fx, fy, interp);
      const bars = spec.bars ? (typeof spec.bars === "number" ? spec.bars : closestSpacing(x)) : 0;
      const arr = (v: number[] | undefined, what: string) => {
        if (!Array.isArray(v) || v.length !== x.length) throw new Error(`${what}: want ${x.length} numbers (one per x)`);
        return toF64(v);
      };
      const iv = spec.intervals;
      const f64 = toF64;
      const { rects, rules, fillY } = spec;
      out.push({
        ...base, bars, x, y, runs: geo.runs, curves: base.markers === "only" || bars || spec.box || spec.noLine ? [] : geo.curves,
        base: spec.base, colors: spec.colors, alphas: spec.alphas, sizes: spec.sizes, texts: spec.texts, tips: spec.tips,
        rects: rects && { ...rects, x1: f64(rects.x1), x2: f64(rects.x2), y1: f64(rects.y1), y2: f64(rects.y2) },
        rules: rules && { axis: rules.axis, at: f64(rules.at), lo: f64(rules.lo), hi: f64(rules.hi) },
        fillY: fillY && { // runs in (y, x) axis space
          lo: geometry(f64(fillY.y), f64(fillY.lo), fy, fx, interp).runs, hi: geometry(f64(fillY.y), f64(fillY.hi), fy, fx, interp).runs, opacity: fillY.opacity },
        fill: spec.fill && resolveFill(spec.fill, { runs: geo.runs }, x, fx, fy, interp),
        intervals: iv && { lo: arr(iv.lo, "intervals.lo"), hi: arr(iv.hi, "intervals.hi"), cap: iv.cap ?? 6, label: iv.label ?? "interval" },
        box: spec.box && { q1: arr(spec.box.q1, "box.q1"), q3: arr(spec.box.q3, "box.q3"), width: spec.box.width ?? 0.6 },
      });
      if (env && (env.mode ?? "add") === "add") {
        const ey = envelope(y, env.type, Boolean(env.reverse));
        const label = ctl.series.length > 1 && spec.label ? `${spec.label}, ${envelopeLabel(env.type, Boolean(env.reverse))}` : envelopeLabel(env.type, Boolean(env.reverse));
        const ecolor = env.color ?? (ctl.series.length > 1 ? color : PALETTE.find((c) => c !== color)!);
        out.push({
          ...base, label, color: ecolor, width: ENVELOPE_WIDTH, dash: false, markers: false, opacity: 1, legend: true, stats: undefined,
          x, y: ey, ...geometry(x, ey, fx, fy, "linear"),
        });
      }
    }
  }
  return out;
}

/** A fill's bounds: `self` where one is missing (the series' runs, or its fn). */
function resolveFill(spec: NonNullable<SeriesSpec["fill"]>, self: Bound, x: Float64Array, fx: (v: number) => number, fy: (v: number) => number, interp: string) {
  const bound = (b: FillBound | undefined): Bound => {
    if (b === undefined) return self;
    if (typeof b === "number") return { c: fy(b) };
    if (Array.isArray(b)) {
      if (b.length !== x.length) throw new Error(`fill: a bound of ${b.length} values for ${x.length} x`);
      return { runs: geometry(x, toF64(b), fx, fy, interp).runs };
    }
    const fn = resolveFn(b);
    return { f: fn.f, domain: fn.domain };
  };
  return { lo: bound(spec.lo), hi: bound(spec.hi), opacity: spec.opacity ?? BAND_OPACITY };
}

const statOf = (sp: SeriesSpec): StatSpec | undefined => (sp.stat === undefined ? undefined : typeof sp.stat === "string" ? { type: sp.stat } : sp.stat);
const percent = (alpha: number) => `${+(100 * (1 - alpha)).toFixed(3)}%`;

/** The plain series a stat series stands for (others: itself). They share its colour, label and
 * style; extras are unlabelled or labelled after it (`many`: prefixed with its label). */
function expandStat(spec: SeriesSpec, color: string, boxIndex: number, many: boolean): (SeriesSpec & { stats?: Record<string, number> })[] {
  const st = statOf(spec);
  if (!st) return [spec];
  const { label = "", width, dash, opacity, legend } = spec;
  const style = { label, color, width, dash, opacity, legend };
  const pre = many && label ? `${label}, ` : "";
  if (st.type === "linreg") {
    const src = spec.file ? seriesFile(spec.file) : { y: spec.y ?? [], x: spec.x ?? (spec.y ?? []).map((_, i) => i) };
    const alpha = st.band ?? 0.05;
    const r = linreg(src.x, src.y, alpha);
    const pad = 0.05 * (r.hi - r.lo);
    const domain: [number, number] = [r.lo - pad, r.hi + pad];
    const params = { a: r.a, b: r.b, t: r.t, s: r.s, n: r.n, m: r.mx, sxx: r.sxx, k: st.kind === "prediction" ? 1 : 0 };
    const half = "t*s*sqrt(k + 1/n + (x - m)^2/sxx)";
    const f4 = (v: number) => formatNumber(v, 4);
    return [
      { ...style, label: label || "data", x: Array.from(src.x), y: Array.from(src.y), markers: spec.markers ?? "only", stats: { n: r.n, a: r.a, b: r.b, r2: r.r2, s: r.s, t: r.t } },
      { color, fn: { expr: "a + b*x", params: { a: r.a, b: r.b }, domain }, width: 2, label: `${pre}fit ŷ = ${f4(r.a)} ${r.b < 0 ? "−" : "+"} ${f4(Math.abs(r.b))}𝑥` },
      { color, fill: { lo: { expr: `a + b*x - ${half}`, params, domain }, hi: { expr: `a + b*x + ${half}`, params, domain } }, label: `${pre}${percent(alpha)} ${st.kind === "prediction" ? "prediction" : "confidence"} band` },
    ];
  }
  const s = sortedFinite(spec.samples ?? (spec.file ? seriesFile(spec.file).y : spec.y ?? []));
  const n = s.length;
  if (!n) throw new Error(`stat ${st.type}: no finite samples`);
  switch (st.type) {
    case "ecdf": { // padded at both ends: 0 before the first sample, 1 after the last
      const e = ecdf(s, st.band);
      const pad = 0.03 * (s[n - 1] - s[0]) || 0.5;
      const x = [s[0] - pad, ...e.x, s[n - 1] + pad], y = [0, ...e.y, 1];
      const out: (SeriesSpec & { stats?: Record<string, number> })[] = [{ ...style, x, y, interpolation: "step", width: width ?? 2, stats: { n, eps: e.eps } }];
      if (st.band) out.push({ color, x, interpolation: "step", fill: { lo: y.map((v) => Math.max(0, v - e.eps)), hi: y.map((v) => Math.min(1, v + e.eps)) }, label: `${pre}${percent(st.band)} DKW band`, legend });
      return out;
    }
    case "qq": {
      const r = qq(s);
      const out: (SeriesSpec & { stats?: Record<string, number> })[] = [{ ...style, x: r.x, y: r.y, markers: spec.markers ?? "only", stats: { n, slope: r.slope, intercept: r.intercept } }];
      // a function, so it runs on past the samples as the view zooms out (|z| ≤ 8: Φ⁻¹ of 1e-15)
      const ref = { expr: "a + b*x", params: { a: r.intercept, b: r.slope }, domain: [-8, 8] as [number, number] };
      if (st.reference !== false) out.push({ color, fn: ref, extent: false, dash: true, width: 1.2, label: `${label || "sample"}, quartile line`, legend: false });
      return out;
    }
    case "box": {
      const b = boxStats(s);
      const pos = st.position ?? boxIndex;
      return [
        { ...style, x: [pos], y: [b.median], box: { q1: [b.q1], q3: [b.q3], width: st.width ?? 0.6 }, intervals: { lo: [b.lo], hi: [b.hi], cap: 18, label: "whiskers" },
          stats: { n, q1: b.q1, median: b.median, q3: b.q3, lo: b.lo, hi: b.hi, outliers: b.outliers.length } },
        { color, x: b.outliers.map(() => pos), y: b.outliers, markers: "only", label: `${label || "box"}, outlier`, legend: false },
      ];
    }
    case "hist": {
      const h = histogram(s, st.bins, st.binWidth);
      return [{ ...style, x: h.x, y: st.density ? h.counts.map((c) => c / (n * h.width)) : h.counts, bars: h.width, stats: { n, binWidth: h.width, bins: h.x.length } }];
    }
  }
  throw new Error(`stat must be one of ecdf, qq, box, hist, linreg (or {type, ...options})`);
}

const fileMtime = (path: string) => { try { return Deno.statSync(path).mtime?.getTime() ?? 0; } catch { return -1; } };
const fileCache = new Map<string, { mtime: number; value: unknown }>();

/** `parse` of a data file's text, once per path and mtime (`what`: "series", "mark", for errors). */
function cachedFile<T>(what: string, path: string, parse: (text: string) => T): T {
  const mtime = fileMtime(path), key = `${what}\0${path}`;
  const hit = fileCache.get(key);
  if (hit && hit.mtime === mtime) return hit.value as T;
  let text: string;
  try { text = Deno.readTextFileSync(path); } catch (err) { throw new Error(`${what} file ${path}: ${(err as Error).message}`); }
  const value = parse(text);
  if (fileCache.size > 64) fileCache.clear();
  fileCache.set(key, { mtime, value });
  return value;
}

/** Rows of cells of a CSV / TSV text; blank lines and lines starting with # skipped. Separated at
 * tabs (split as is), else commas, else semicolons (RFC 4180: "quoted" cells may hold the
 * separator, line breaks and "" for a quote), else runs of spaces. Unquoted cells trimmed. */
export function parseDelimited(text: string): string[][] {
  const first = text.split(/\r?\n/, 64).find((l) => l.trim() && !l.startsWith("#")) ?? "";
  // a semicolon wins over a comma: `1,5;2,25` is a European CSV, its commas decimal points
  const bare = first.replace(/"(?:[^"]|"")*"/g, ""); // separators inside quoted cells don't count
  const sep = bare.includes("\t") ? "\t" : bare.includes(";") ? ";" : bare.includes(",") ? "," : null;
  if (sep === null || sep === "\t") {
    return text.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("#")).map((l) => (sep ? l.split(sep).map((c) => c.trim()) : l.trim().split(/\s+/)));
  }
  const rows: string[][] = [];
  const n = text.length;
  let i = 0;
  const eol = (c: string) => c === "\n" || c === "\r";
  while (i < n) {
    if (text[i] === "#") { // a comment line
      while (i < n && text[i] !== "\n") i += 1;
      i += 1;
      continue;
    }
    const row: string[] = [];
    for (;;) {
      let j = i;
      while (text[j] === " " || text[j] === "\t") j += 1;
      if (text[j] === '"') { // quoted: up to the closing quote ("" a quote), then to the separator
        let cell = "";
        for (i = j + 1;;) {
          const q = text.indexOf('"', i);
          if (q < 0) throw new Error(`unterminated quoted cell (opened at character ${j + 1})`);
          cell += text.slice(i, q);
          i = q + 1;
          if (text[i] !== '"') break;
          cell += '"';
          i += 1;
        }
        while (i < n && text[i] !== sep && !eol(text[i])) i += 1;
        row.push(cell);
      } else {
        while (j < n && text[j] !== sep && !eol(text[j])) j += 1;
        row.push(text.slice(i, j).trim());
        i = j;
      }
      if (text[i] === sep) {
        i += 1;
        continue;
      }
      if (text[i] === "\r") i += 1;
      if (text[i] === "\n") i += 1;
      break;
    }
    if (row.length > 1 || row[0]) rows.push(sep === ";" ? row.map((c) => (/^[-+]?\d*,\d+(e[-+]?\d+)?$/i.test(c) ? c.replace(",", ".") : c)) : row);
  }
  return rows;
}

/** A series' data file (`SeriesSpec.file`), parsed once per path and mtime. */
function seriesFile(path: string): { x: Float64Array; y: Float64Array } {
  return cachedFile("series", path, (text) => {
    const num = (v: unknown) => (v === null || v === "" ? NaN : Number(String(v).replace(/^([-+]?)inf$/i, "$1Infinity")));
    let xs: unknown[] | undefined, ys: unknown[];
    if (/\.json$/i.test(path)) {
      const o = JSON.parse(text);
      [xs, ys] = Array.isArray(o) ? [undefined, o] : [o?.x ?? undefined, o?.y];
      if (!Array.isArray(ys) || (xs && !Array.isArray(xs))) throw new Error(`series file ${path}: want {x?: [...], y: [...]} or [y...]`);
    } else { // .csv / .tsv (`parseDelimited`): a header line is skipped
      let rows: string[][];
      try { rows = parseDelimited(text); } catch (err) { throw new Error(`series file ${path}: ${(err as Error).message}`); }
      if (rows.length && rows[0].some((c) => Number.isNaN(num(c)) && !/^nan$/i.test(c))) rows.shift();
      const cols = rows[0]?.length ?? 1;
      if (cols > 2) throw new Error(`series file ${path}: ${cols} columns; give 1 (y) or 2 (x, y)`);
      [xs, ys] = cols === 2 ? [rows.map((r) => r[0]), rows.map((r) => r[1])] : [undefined, rows.map((r) => r[0])];
    }
    if (xs && xs.length !== ys.length) throw new Error(`series file ${path}: ${xs.length} x but ${ys.length} y`);
    const y = Float64Array.from(ys, num);
    return { x: xs ? Float64Array.from(xs, num) : Float64Array.from(y, (_, i) => i), y };
  });
}

/** A mark's data file (`Mark.file`), parsed once per path and mtime: JSON columns {name: [...]}
 * or rows [{...}], or a CSV/TSV with a header row (`parseDelimited`; numbers where they parse,
 * "" a gap). */
function tableFile(path: string): Columns {
  return cachedFile("mark", path, (text) => {
    if (/\.json$/i.test(path)) {
      const data = JSON.parse(text);
      if (!data || typeof data !== "object") throw new Error(`mark file ${path}: want columns {name: [...]} or rows [{...}]`);
      return data as Columns;
    }
    let head: string[] | undefined, rows: string[][];
    try { [head, ...rows] = parseDelimited(text); } catch (err) { throw new Error(`mark file ${path}: ${(err as Error).message}`); }
    if (!head) throw new Error(`mark file ${path}: empty`);
    const cell = (c: string | undefined) => (c === undefined || c === "" ? null : Number.isNaN(Number(c)) && !/^[-+]?(inf|nan)$/i.test(c) ? c : Number(c.replace(/^([-+]?)inf$/i, "$1Infinity")));
    return Object.fromEntries(head.map((h, j) => [h, rows.map((r) => cell(r[j]))])) as Columns;
  });
}

/** The polyline of a step piece: (x[i], y[i]) then (x[i + 1], y[i]). */
function stepPolyline(p: Piece): Piece {
  const n = p.xw.length;
  const xw = new Float64Array(2 * n - 1), yw = new Float64Array(2 * n - 1);
  for (let i = 0; i < n; i += 1) {
    [xw[2 * i], yw[2 * i]] = [p.xw[i], p.yw[i]];
    if (i + 1 < n) [xw[2 * i + 1], yw[2 * i + 1]] = [p.xw[i + 1], p.yw[i]];
  }
  return { xw, yw };
}

/** The NURBS of a piece: degree 1 through its points (parameter: index), or exact cubic. */
function pieceNurbs(p: Piece): NurbsCurve {
  if (p.slopes) {
    return hermiteNurbs(p.xw, p.yw, p.slopes);
  }
  const n = p.xw.length;
  return {
    degree: 1,
    dims: 2,
    points: Array.from(p.xw).flatMap((x, i) => [x, p.yw[i]]),
    weights: Array(n).fill(1),
    knots: [0, ...Array.from({ length: n }, (_, i) => i), n - 1],
  };
}

/** y of a piece at axis-space x (null outside it). */
function pieceAt(p: Piece, t: number): number | null {
  const n = p.xw.length;
  if (n < 2 || !(t >= p.xw[0] && t <= p.xw[n - 1])) {
    return n === 1 && t === p.xw[0] ? p.yw[0] : null;
  }
  if (p.slopes) {
    return hermiteAt(p.xw, p.yw, p.slopes, t);
  }
  if (p.step && t >= p.xw[n - 1]) return p.yw[n - 1];
  let lo = 0, hi = n - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (p.xw[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  if (p.step) return p.yw[lo];
  const f = (t - p.xw[lo]) / (p.xw[lo + 1] - p.xw[lo]);
  return p.yw[lo] + f * (p.yw[lo + 1] - p.yw[lo]);
}

/** A fill bound at axis-space x (null where it is undefined). */
function boundAt(b: Bound, t: number, tx: (w: number) => number, fy: (v: number) => number): number | null {
  if ("c" in b) return b.c;
  if ("f" in b) {
    const xv = tx(t);
    return xv >= b.domain[0] && xv <= b.domain[1] ? fy(b.f(xv)) : null;
  }
  for (const p of b.runs) {
    const v = pieceAt(p, t);
    if (v !== null) return v;
  }
  return null;
}

// --- Curves --------------------------------------------------------------------------------------

/** A curve drawn by `NurbsGpuRenderer` into a shared layer. */
class CurveLayer {
  readonly renderer: NurbsGpuRenderer;
  readonly spans: KnotSpan[];
  readonly origin: [number, number];

  constructor(private readonly device: GPUDevice, canvas: TextureTarget, readonly curve: NurbsCurve) {
    const error = getNurbsError(curve);
    if (error) {
      throw new Error(error);
    }
    const p = curve.points;
    let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 2) {
      x0 = Math.min(x0, p[i]);
      x1 = Math.max(x1, p[i]);
      y0 = Math.min(y0, p[i + 1]);
      y1 = Math.max(y1, p[i + 1]);
    }
    this.origin = [(x0 + x1) / 2, (y0 + y1) / 2]; // precision in f32
    this.spans = getKnotSpans(curve);
    this.renderer = new NurbsGpuRenderer(device, canvas, curve, this.spans, this.origin);
  }

  draw(view: View, width: number, color: Rgba): void {
    const ppu: [number, number] = [view.sx, view.sy];
    this.renderer.setSegmentCounts(getSegmentCounts(this.curve, this.spans, ppu));
    this.renderer.draw({ scale: ppu, offset: view.offset(this.origin), size: [view.w, view.h], width, color }, true);
  }

  dispose(): void {
    this.renderer.dispose();
  }
}

interface View {
  x0: number; // axis space
  x1: number;
  y0: number;
  y1: number;
  plot: [number, number, number, number];
  scale: number;
  w: number;
  h: number;
  sx: number;
  sy: number;
  offset(origin: [number, number]): [number, number];
  px(x: number): number;
  py(y: number): number;
}

interface Hover {
  pixel: [number, number];
  x: number; // value
  series?: { label: string; color: string; value: number; pixel: [number, number]; text?: string }; // text: marks (rows: their channels)
  rows: [string, string, string][]; // more: the hovered sample's box, interval; bands at x
}

// --- Formatting ----------------------------------------------------------------------------------

const fmt = (v: number, digits = 5) => formatNumber(v, digits);

function valueText(spec: AxisSpec, v: number): string {
  if (spec.ticks === "exp2" || spec.ticks === "exp10") {
    const base = spec.ticks === "exp2" ? 2 : 10;
    return `${fmt(base ** v, 4)} (${base === 2 ? "log₂" : "log₁₀"} ${fmt(v, 4)})`;
  }
  return fmt(v);
}

// --- Plot ----------------------------------------------------------------------------------------

/** Dense samples of a generator, computed on the CPU (f64) into `run` (y range, dashes, legend)
 * and uploaded to `buffer` (drawn from there); kept while the window stays within them at about
 * the same zoom (panning doesn't resample). */
interface Dense {
  a: number; // values
  b: number;
  step: number; // axis space per sample
  run: Piece;
  buffer: GPUBuffer;
  count: number;
  origin: [number, number]; // of the samples in `buffer` (axis space)
}

const denseStep = (w: number, h: number, x0: number, x1: number) => {
  const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
  return (x1 - x0) / (3 * (w - Math.round(24 * s) - 66 * s)); // about 3 per pixel of the plot
};

export class LinePlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly frameReads: GPUBuffer[] = [];
  private readonly scratch: TextureTarget; // one series at a time, opaque
  private readonly layer: TextureTarget; // all series, composited in order
  private readonly compositor: LayerCompositor;
  private dataKey?: string;
  private dataGen = 0; // bumped when the data changes: in geomKey (not the data key itself, which can be long)
  private geomKey?: string;
  private chromeKey?: string;
  series: Series[] = [];
  private curves: { series: number; layer: CurveLayer }[] = [];
  private dense = new Map<number, Dense>();
  private readonly chords: ChordRenderer;
  private extent: [number, number] = [0, 1]; // axis-space x of all data
  view!: View;
  private ticks!: [ReturnType<typeof axisTicks>, ReturnType<typeof axisTicks>];
  private chrome?: Quads; // below the curves
  private top?: Quads; // above them: references, marks, markers, legend
  private marksKey?: string; // of the last compile, good or bad
  private marked?: ReturnType<typeof compileMarks>; // the last good compile
  private marksError?: unknown; // of the compile of marksKey, if it failed
  private colorbar?: ColorBar;

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.overlay = new OverlayRenderer(device, font);
    this.scratch = new TextureTarget(device, FRAME_FORMAT);
    this.layer = new TextureTarget(device, FRAME_FORMAT);
    this.compositor = new LayerCompositor(device);
    this.chords = new ChordRenderer(device);
  }

  static async create(): Promise<LinePlot> {
    const device = await requestGpuDevice();
    if (!device) {
      throw new Error("No WebGPU device");
    }
    return new LinePlot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  /** Series and their renderers: built and validated first, then swapped in (a bad control
   * leaves the last good plot as it was). */
  setData(ctl: Control, marksKey = ""): void {
    // the control's own series (marks: by their key, not their compiled series), a rewritten file: new data
    const own = ctl.series.slice(0, ctl.series.length - (marksKey ? this.marked!.series.length : 0));
    const key = JSON.stringify([own, ctl.x, ctl.y, own.map((sp) => (sp.file ? fileMtime(sp.file) : 0))]) + marksKey;
    if (key === this.dataKey) {
      return;
    }
    const curves: { series: number; layer: CurveLayer }[] = [];
    let series: Series[];
    try {
      series = resolveSeries(ctl);
      series.forEach((s, i) => {
        for (const c of s.curves) curves.push({ series: i, layer: new CurveLayer(this.device, this.scratch, c) });
      });
    } catch (err) {
      for (const c of curves) c.layer.dispose();
      throw err;
    }
    let [lo, hi] = [Infinity, -Infinity];
    const fx = axisTransform(ctl.x).fwd;
    for (const s of series) {
      if (!s.extent) continue;
      for (const v of s.x) {
        const gw = glyphWidth(s);
        for (const e of gw ? [v - gw / 2, v + gw / 2] : [v]) { // bars, boxes: their edges
          const w = fx(e);
          if (Number.isFinite(w)) {
            lo = Math.min(lo, w);
            hi = Math.max(hi, w);
          }
        }
      }
      for (const d of [s.domain, ...[s.fill?.lo, s.fill?.hi].map((b) => (b && "f" in b ? b.domain : undefined))]) {
        if (!d) continue;
        lo = Math.min(lo, fx(d[0]));
        hi = Math.max(hi, fx(d[1]));
      }
      const ws = [s.rects?.x1, s.rects?.x2, s.rules?.axis === "x" ? s.rules.at : undefined].flatMap((a) => (a ? Array.from(a, fx) : []));
      for (const p of [...(s.fillY?.lo ?? []), ...(s.fillY?.hi ?? [])]) ws.push(...p.yw); // axis space already
      for (const w of ws) {
        if (Number.isFinite(w)) [lo, hi] = [Math.min(lo, w), Math.max(hi, w)];
      }
    }
    if (series.some((s) => s.extent && s.rects?.end === "x") && hi > lo) hi += 0.05 * (hi - lo); // room past horizontal bars (their labels)
    if (series.some((s) => s.extent && s.sizes) && hi > lo) [lo, hi] = [lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo)]; // and around sized dots
    const cats = ctl.x.categories?.length;
    if (cats) [lo, hi] = [Math.min(lo, -0.5), Math.max(hi, cats - 0.5)]; // whole slots
    for (const c of this.curves) c.layer.dispose();
    for (const d of this.dense.values()) d.buffer.destroy();
    this.dense.clear();
    this.series = series;
    this.curves = curves;
    this.extent = lo < hi ? [lo, hi] : lo === hi ? [lo - 0.5, lo + 0.5] : [0, 1];
    this.dataKey = key;
    this.dataGen += 1;
    this.geomKey = this.chromeKey = undefined;
  }

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 19 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 13 * s),
    };
  }

  /** The window's x range (axis space) and its transforms. */
  private window(ctl: Control) {
    const tx = axisTransform(ctl.x);
    const ty = axisTransform(ctl.y);
    const x0 = ctl.x0 !== null && Number.isFinite(tx.fwd(ctl.x0)) ? tx.fwd(ctl.x0) : this.extent[0];
    const x1 = ctl.x1 !== null && Number.isFinite(tx.fwd(ctl.x1)) ? tx.fwd(ctl.x1) : this.extent[1];
    return { tx, ty, x0, x1 };
  }

  /** The dense samples of series `si` if they cover [a, b] (values) at about `step`. */
  private denseFor(si: number, a: number, b: number, step: number): Dense | null {
    const d = this.dense.get(si);
    return d && a >= d.a && b <= d.b && d.step <= step * 1.5 && d.step >= step / 1.5 ? d : null;
  }

  /** (Re)samples the generators whose dense samples don't cover the window: over three window
   * widths, about 3 per pixel, `x, y, 0, x` per sample relative to their middle (lib/chords.ts). */
  private ensureDense(ctl: Control): void {
    const { tx, x0, x1 } = this.window(ctl);
    if (!(x1 > x0)) return;
    const ty = axisTransform(ctl.y).fwd;
    const step = denseStep(Math.trunc(ctl.width), Math.trunc(ctl.height), x0, x1);
    this.series.forEach((sr, si) => {
      if (!sr.gen || !sr.domain) return;
      const a = Math.max(tx.inv(x0), sr.domain[0]);
      const b = Math.min(tx.inv(x1), sr.domain[1]);
      if (!(b > a) || this.denseFor(si, a, b, step)) return;
      const [d0, d1] = sr.domain;
      const ca = Math.max(tx.inv(x0 - (x1 - x0)), d0);
      const cb = Math.min(tx.inv(x1 + (x1 - x0)), d1);
      const wa = tx.fwd(ca), wb = tx.fwd(cb);
      if (!(wb > wa)) return;
      const m = Math.trunc(Math.min(Math.max((wb - wa) / step, 256), 65536)), dw = (wb - wa) / (m - 1);
      const xw = new Float64Array(m), xs = new Float64Array(m);
      for (let i = 0; i < m; i += 1) xs[i] = tx.inv((xw[i] = wa + i * dw));
      const yw = sr.gen.sample(xs);
      const origin: [number, number] = [wa + (dw * (m - 1)) / 2, 0];
      const data = new Float32Array(4 * m);
      const f32 = (v: number) => (Math.abs(v) > 3e38 && Number.isFinite(v) ? Math.sign(v) * 3e38 : v); // stays finite in f32
      for (let i = 0; i < m; i += 1) [data[4 * i], data[4 * i + 1]] = [xw[i] - origin[0], f32((yw[i] = ty(yw[i])))];
      let buffer = this.dense.get(si)?.buffer;
      if (!buffer || buffer.size < data.byteLength) {
        buffer?.destroy();
        buffer = this.device.createBuffer({ size: 2 ** Math.ceil(Math.log2(data.byteLength)), usage: STORAGE | COPY_DST });
      }
      this.device.queue.writeBuffer(buffer, 0, data);
      this.dense.set(si, { a: ca, b: cb, step: dw, run: { xw, yw }, buffer, count: m, origin });
    });
  }

  setWindow(ctl: Control): void {
    const w = Math.trunc(ctl.width);
    const h = Math.trunc(ctl.height);
    const { tx, ty, x0, x1 } = this.window(ctl);
    if (!(x1 > x0)) {
      throw new Error("x1 must be above x0");
    }
    // Geometry: the view and the curve layer; chrome: grid, text, references, legend
    const geomKey = JSON.stringify([this.dataGen, w, h, x0, x1, ctl.ymin, ctl.ymax, ctl.overlays, Boolean(ctl.ylabel), Boolean(ctl.xlabel)]);
    const chromeKey = JSON.stringify([geomKey, ctl.legend, ctl.title, ctl.subtitle, ctl.xlabel, ctl.ylabel, ctl.zeroLine, ctl.nanColor]);
    if (geomKey !== this.geomKey) {
      this.setGeometry(ctl, w, h, x0, x1, tx, ty);
      this.geomKey = geomKey;
    }
    if (chromeKey !== this.chromeKey) {
      this.overlay.setSize(w, h);
      [this.chrome, this.top] = this.makeChrome(ctl, ...this.ticks);
      this.chromeKey = chromeKey;
    }
  }

  private setGeometry(ctl: Control, w: number, h: number, x0: number, x1: number, tx: ReturnType<typeof axisTransform>, ty: ReturnType<typeof axisTransform>): void {
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    const F = this.fonts(s);
    const T = Math.round((ctl.ylabel ? 74 : 56) * s);
    const B = h - Math.round((ctl.xlabel ? 62 : 40) * s);
    const R = w - Math.round(24 * s);

    // Dense samples of generators over (at least) the window (`ensureDense`)
    const step = denseStep(w, h, x0, x1);
    const dense = this.series.map((sr, si) => {
      if (!sr.gen || !sr.domain) return null;
      const a = Math.max(tx.inv(x0), sr.domain[0]);
      const b = Math.min(tx.inv(x1), sr.domain[1]);
      return b > a ? this.denseFor(si, a, b, step) : null;
    });
    const runsOf = (si: number) => (dense[si] ? [dense[si]!.run] : this.series[si].runs);

    // y range: visible points (and the cubics' extrema), references on y
    let [lo, hi] = [Infinity, -Infinity];
    const take = (v: number) => {
      if (Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    };
    this.series.forEach((sr, si) => {
      if (!sr.extent) return;
      for (const p of runsOf(si)) {
        const n = p.xw.length;
        for (let i = 0; i < n; i += 1) {
          if (p.xw[i] >= x0 && p.xw[i] <= x1) take(p.yw[i]);
          if (p.slopes && i < n - 1 && p.xw[i + 1] >= x0 && p.xw[i] <= x1) {
            const d = p.xw[i + 1] - p.xw[i];
            const [a, b] = getHermiteExtent(p.yw[i], p.yw[i + 1], p.slopes[i] * d, p.slopes[i + 1] * d);
            take(a);
            take(b);
          }
        }
        for (const t of [x0, x1]) take(pieceAt(p, t) ?? NaN);
      }
    });
    this.series.forEach((sr) => { // markers (isolated points make no runs), geoms: fills (sampled), intervals, boxes
      if (!sr.extent) return;
      if (sr.markers) sr.x.forEach((xv, i) => { if (tx.fwd(xv) >= x0 && tx.fwd(xv) <= x1) take(ty.fwd(sr.y[i])); });
      if (sr.fill) {
        for (let k = 0; k <= 256; k += 1) {
          const t = x0 + ((x1 - x0) * k) / 256;
          const [a, b] = [boundAt(sr.fill.lo, t, tx.inv, ty.fwd), boundAt(sr.fill.hi, t, tx.inv, ty.fwd)];
          if (a !== null && b !== null) [a, b].forEach(take);
        }
      }
      if (sr.rects) { // marks: rects over the window, horizontal rules, fills of y
        const rc = sr.rects;
        rc.y1.forEach((_, i) => {
          const [a, b] = [tx.fwd(rc.x1[i]), tx.fwd(rc.x2[i])];
          if (Math.max(a, b) >= x0 && Math.min(a, b) <= x1) [rc.y1[i], rc.y2[i]].forEach((u) => take(ty.fwd(u)));
        });
      }
      if (sr.rules?.axis === "y") sr.rules.at.forEach((u) => take(ty.fwd(u)));
      for (const p of sr.fillY ? [...sr.fillY.lo, ...sr.fillY.hi] : []) p.xw.forEach((w, i) => p.yw[i] >= x0 && p.yw[i] <= x1 && take(w));
      for (const [lo_, hi_] of [[sr.intervals?.lo, sr.intervals?.hi], [sr.box?.q1, sr.box?.q3]]) {
        if (!lo_ || !hi_) continue;
        sr.x.forEach((xv, i) => {
          const w = tx.fwd(xv);
          if (w >= x0 - glyphWidth(sr) / 2 && w <= x1 + glyphWidth(sr) / 2) [lo_[i], hi_[i]].forEach((u) => take(ty.fwd(u)));
        });
      }
    });
    for (const o of ctl.overlays) {
      if (o.kind === "hline" || (o.kind === "band" && o.axis === "y")) o.values.forEach((v) => take(ty.fwd(v)));
    }
    if (!(lo <= hi)) [lo, hi] = [-1, 1];
    const pad = hi > lo ? (hi - lo) * Y_MARGIN : 0.5;
    const base = this.series.some((sr) => sr.bars || sr.base) ? ty.fwd(0) : NaN; // bars stand on 0, no margin below it
    let y0 = ctl.ymin === null && Number.isFinite(base) && base <= lo ? base : ctl.ymin !== null && Number.isFinite(ty.fwd(ctl.ymin)) ? ty.fwd(ctl.ymin) : lo - pad;
    let y1 = ctl.ymax !== null && Number.isFinite(ty.fwd(ctl.ymax)) ? ty.fwd(ctl.ymax) : hi + pad;
    if (!(y1 > y0)) [y0, y1] = [y0 - 0.5, y0 + 0.5];
    const ycats = ctl.y.categories?.length;
    if (ycats && ctl.ymin === null && ctl.ymax === null) [y0, y1] = [-0.5, ycats - 0.5]; // whole slots

    // Ticks, then the left margin from the widest label
    const yticks = axisTicks(ctl.y, ty.inv(y0), ty.inv(y1), (B - T) / s);
    const yw = Math.max(0, ...yticks.map((t) => this.font.measure(F.tick, t.label)));
    const L = Math.max(Math.round(yw + 22 * s), Math.round(50 * s));
    const xticks = axisTicks({ ...ctl.x, index: ctl.x.index ?? ctl.series.every((sp) => statOf(sp)?.type === "box" || (!sp.stat && !sp.x && !sp.sample && !sp.fn && !sp.file)) }, tx.inv(x0), tx.inv(x1), (R - L) / s);
    this.ticks = [xticks, yticks];
    const plot: [number, number, number, number] = [L, T, R, B];
    const sx = (R - L) / (x1 - x0);
    const sy = (B - T) / (y1 - y0);
    this.view = {
      x0, x1, y0, y1, plot, scale: s, w, h, sx, sy,
      offset: (o) => [L + (o[0] - x0) * sx - w / 2, h / 2 - T - (y1 - o[1]) * sy],
      px: (x) => L + (x - x0) * sx,
      py: (y) => T + (y1 - y) * sy,
    };

    // Curve layer: each series drawn opaque into the scratch layer, then composited in order
    // with its opacity, through its dash mask
    this.layer.clear(w, h);
    this.series.forEach((sr, si) => {
      const d = dense[si];
      const layers = this.curves.filter((c) => c.series === si).map((c) => c.layer);
      if (!d && !layers.length) return;
      this.scratch.clear(w, h);
      if (d) {
        this.chords.draw(this.scratch.currentTexture(), d.buffer, d.count, {
          scale: [this.view.sx, this.view.sy], offset: this.view.offset(d.origin), size: [w, h], width: sr.width * s, color: premul(sr.color),
        });
      }
      for (const l of layers) l.draw(this.view, sr.width * s, premul(sr.color));
      const rects: [number, number, number, number][] = sr.dash
        ? this.dashStrips(runsOf(si)).map(([l, r]) => [l, T, r, B + 1])
        : [[L, T, R + 1, B + 1]];
      this.compositor.composite(this.scratch.currentTexture(), this.layer.currentTexture(), rects, sr.opacity);
    });
  }

  /** Pixel x intervals of the dashes of a series, along its length on screen. */
  private dashStrips(runs: Piece[]): [number, number][] {
    const v = this.view;
    const [L, , R] = v.plot;
    const period = (DASH + GAP) * v.scale;
    const on = DASH * v.scale;
    const strips: [number, number][] = [];
    {
      for (const p of runs) {
        // Polyline on screen (cubics sampled every ~2 px)
        const pts: [number, number][] = [];
        for (let i = 0; i < p.xw.length; i += 1) {
          if (p.slopes && i > 0) {
            const steps = Math.max(2, Math.ceil(((p.xw[i] - p.xw[i - 1]) * v.sx) / 2));
            for (let k = 1; k < steps; k += 1) {
              const t = p.xw[i - 1] + ((p.xw[i] - p.xw[i - 1]) * k) / steps;
              pts.push([v.px(t), v.py(hermiteAt(p.xw, p.yw, p.slopes, t))]);
            }
          }
          if (p.step && i > 0) pts.push([v.px(p.xw[i]), v.py(p.yw[i - 1])]);
          pts.push([v.px(p.xw[i]), v.py(p.yw[i])]);
        }
        let dist = 0;
        for (let i = 1; i < pts.length; i += 1) {
          const [ax, ay] = pts[i - 1];
          const [bx, by] = pts[i];
          const len = Math.hypot(bx - ax, by - ay);
          if (!(len > 0)) continue;
          // pieces of [dist, dist + len] where (dist mod period) < on
          let s0 = dist;
          const end = dist + len;
          while (s0 < end) {
            const k = Math.floor(s0 / period);
            const isOn = s0 - k * period < on;
            let s1 = Math.min(end, k * period + (isOn ? on : period));
            if (!(s1 > s0)) s1 = Math.min(end, (k + 1) * period + (isOn ? 0 : on)); // rounding at a boundary
            if (!(s1 > s0)) break;
            if (isOn) {
              const xa = ax + ((bx - ax) * (s0 - dist)) / len;
              const xb = ax + ((bx - ax) * (s1 - dist)) / len;
              const [l, r] = [Math.max(Math.min(xa, xb), L), Math.min(Math.max(xa, xb), R)];
              const last = strips[strips.length - 1];
              if (last && Math.abs(last[1] - l) < 1e-6) last[1] = r;
              else if (r > l) strips.push([l, r]);
            }
            s0 = s1;
          }
          dist = end;
        }
      }
    }
    return strips;
  }

  /** Grid, ticks, labels, title and bands (below the curves); references, unplottable marks,
   * markers and the legend (above). */
  private makeChrome(ctl: Control, xticks: ReturnType<typeof axisTicks>, yticks: ReturnType<typeof axisTicks>): [Quads, Quads] {
    const q = new Quads(this.font);
    const top = new Quads(this.font);
    const v = this.view;
    const s = v.scale;
    const [L, T, R, B] = v.plot;
    const F = this.fonts(s);
    const tx = axisTransform(ctl.x);
    const ty = axisTransform(ctl.y);

    for (const t of xticks) {
      const X = Math.round(v.px(t.world));
      if (X >= L && X <= R) {
        q.rect(X, T, 1, B - T + 1, premul(GRID, t.label ? 1 : 0.5));
        if (t.label) q.text(t.label, X + 0.5, B + Math.round(8 * s), F.tick, premul(INK2), "mt");
      }
    }
    for (const t of yticks) {
      const Y = Math.round(v.py(t.world));
      if (Y >= T && Y <= B) {
        q.rect(L, Y, R - L + 1, 1, premul(GRID, t.label ? 1 : 0.5));
        if (t.label) q.text(t.label, L - Math.round(10 * s), Y + 0.5, F.tick, premul(INK2), "rm");
      }
    }
    if (ctl.zeroLine && (ctl.y.scale ?? "linear") !== "log" && !ctl.y.ticks?.startsWith("exp") && v.y0 < 0 && v.y1 > 0) {
      q.rect(L, Math.round(v.py(0)), R - L + 1, 1, premul(ZERO));
    }
    // Title row, axis labels
    const ty0 = Math.round(28 * s);
    const tw = ctl.title ? q.text(ctl.title, L, ty0, F.title, premul(INK), "lm") : 0;
    if (ctl.subtitle) q.text(ctl.subtitle, L + tw + Math.round(16 * s), ty0, F.sub, premul(INK2), "lm");
    if (ctl.ylabel) q.text(ctl.ylabel, L, T - Math.round(12 * s), F.sub, premul(INK2), "lb");
    if (ctl.xlabel) q.text(ctl.xlabel, (L + R) / 2, B + Math.round(30 * s), F.sub, premul(INK2), "mt");

    // References: bands under the curves, lines over them
    const refColor = (o: OverlaySpec) => o.color ?? "#666666";
    for (const o of ctl.overlays) {
      const axis = o.kind === "band" ? (o.axis ?? "x") : o.kind === "hline" ? "y" : "x";
      const pos = o.values.map((val) => (axis === "x" ? v.px(tx.fwd(val)) : v.py(ty.fwd(val))));
      if (!pos.every(Number.isFinite)) continue;
      if (o.kind === "band") {
        const [a, b] = [Math.min(pos[0], pos[1]), Math.max(pos[0], pos[1])];
        const c = premul(refColor(o), BAND_OPACITY);
        if (axis === "x") {
          const [l, r] = [Math.max(a, L), Math.min(b, R)];
          if (r > l) q.rect(Math.round(l), T, Math.round(r) - Math.round(l), B - T + 1, c);
        } else {
          const [t0, b0] = [Math.max(a, T), Math.min(b, B)];
          if (b0 > t0) q.rect(L, Math.round(t0), R - L + 1, Math.round(b0) - Math.round(t0), c);
        }
        continue;
      }
      const lw = Math.max(1, Math.round((o.width ?? 1.5) * s));
      const c = premul(refColor(o));
      const p = Math.round(pos[0] - lw / 2);
      const span = axis === "x" ? [T, B + 1] : [L, R + 1];
      if (axis === "x" ? p < L || p > R : p < T || p > B) continue;
      const step = o.dash ? Math.round((DASH + GAP) * s) : span[1] - span[0];
      const on = o.dash ? Math.round(DASH * s) : step;
      for (let a = span[0]; a < span[1]; a += step) {
        const len = Math.min(on, span[1] - a);
        if (axis === "x") top.rect(p, a, lw, len, c);
        else top.rect(a, p, len, lw, c);
      }
    }

    // Fills, below the curves: a column per pixel between the bounds there (runs of equal columns merged)
    for (const sr of this.series) {
      if (!sr.fill) continue;
      const c = premul(sr.color, sr.fill.opacity);
      let run: [number, number, number, number] | null = null; // x, top, width, height
      const flush = () => { if (run) q.rect(...run, c); run = null; };
      for (let X = Math.ceil(L); X <= R; X += 1) {
        const t = v.x0 + (X + 0.5 - L) / v.sx;
        const a = boundAt(sr.fill.lo, t, tx.inv, ty.fwd), b = boundAt(sr.fill.hi, t, tx.inv, ty.fwd);
        const [ya, yb] = [a, b].map((u) => (u === null || Number.isNaN(u) ? NaN : Math.min(Math.max(v.py(u), T), B + 1)));
        if (!(Math.abs(ya - yb) > 0)) { flush(); continue; }
        const [top0, h0] = [Math.min(ya, yb), Math.abs(ya - yb)];
        if (run && Math.abs(run[1] - top0) < 1e-3 && Math.abs(run[3] - h0) < 1e-3) run[2] += 1;
        else { flush(); run = [X, top0, 1, h0]; }
      }
      flush();
    }

    // Fills between functions of y (violins): a row per pixel
    for (const sr of this.series) {
      if (!sr.fillY) continue;
      const c = premul(sr.color, sr.fillY.opacity);
      for (let Y = Math.ceil(T); Y <= B; Y += 1) {
        const t = v.y1 - (Y + 0.5 - T) / v.sy;
        const [a, b] = [sr.fillY.lo, sr.fillY.hi].map((runs) => boundAt({ runs }, t, ty.inv, tx.fwd));
        if (a === null || b === null) continue;
        const [xa, xb] = [a, b].map((u) => Math.min(Math.max(v.px(u), L), R + 1)).sort((p, r) => p - r);
        if (xb - xa > 0) q.rect(xa, Y, xb - xa, 1, c);
      }
    }

    // Bars, below the curves: 2 px apart, rounded at the data end
    for (const sr of this.series) {
      if (!sr.bars) continue;
      const c = premul(sr.color, sr.opacity);
      const yb = Math.min(Math.max(Number.isFinite(ty.fwd(0)) ? v.py(ty.fwd(0)) : B + 1, T), B + 1);
      for (let i = 0; i < sr.x.length; i += 1) {
        const yv = ty.fwd(sr.y[i]);
        if (!Number.isFinite(yv)) continue;
        const l = Math.max(v.px(tx.fwd(sr.x[i] - sr.bars / 2)) + s, L);
        const r = Math.min(v.px(tx.fwd(sr.x[i] + sr.bars / 2)) - s, R + 1);
        const yt = Math.min(Math.max(v.py(yv), T), B + 1);
        if (!(r - l >= 1) || Math.abs(yb - yt) < 0.5) continue;
        const [top0, h0] = [Math.min(yt, yb), Math.abs(yb - yt)];
        const rad = sr.opacity < 1 ? 0 : Math.min(4 * s, (r - l) / 2, h0 / 2); // (overlapping translucent rects would blend twice)
        q.rect(l, top0, r - l, h0, c, { radius: rad });
        if (rad) q.rect(l, yt < yb ? top0 + h0 / 2 : top0, r - l, h0 / 2, c); // square at the baseline
      }
    }

    // Rects of marks (bars, cells), below the curves: inset, bars rounded at the data end
    for (const sr of this.series) {
      const rc = sr.rects;
      if (!rc) continue;
      const clamp = (u: number, a: number, b: number) => (Number.isNaN(u) ? NaN : Math.min(Math.max(u, a), b)); // ±∞ (log 0): the edge
      for (let i = 0; i < rc.x1.length; i += 1) {
        const [xa, xb] = [rc.x1[i], rc.x2[i]].map((u) => clamp(v.px(tx.fwd(u)), L, R + 1));
        const [ya, yb] = [rc.y1[i], rc.y2[i]].map((u) => clamp(v.py(ty.fwd(u)), T, B + 1));
        if (![xa, xb, ya, yb].every(Number.isFinite)) continue;
        const [l, r] = [Math.min(xa, xb) + rc.inset[0] * s, Math.max(xa, xb) - rc.inset[0] * s].map(Math.round); // whole pixels: the base patch meets exactly
        const [t0, b0] = [Math.min(ya, yb) + rc.inset[1] * s, Math.max(ya, yb) - rc.inset[1] * s].map(Math.round);
        const [w0, h0] = [r - l, b0 - t0];
        if (!(w0 >= 1) || !(h0 >= 0.5)) continue;
        const alpha = sr.alphas?.[i] ?? 1;
        const c = premul(sr.colors?.[i] ?? sr.color, alpha);
        const rad = rc.end && alpha === 1 ? Math.min(rc.radius * s, w0 / 2, h0 / 2) : 0;
        q.rect(l, t0, w0, h0, c, { radius: rad });
        if (rad && rc.end === "y") q.rect(l, yb < ya ? t0 + h0 / 2 : t0, w0, h0 / 2, c); // square at the base
        if (rad && rc.end === "x") q.rect(xb > xa ? l : l + w0 / 2, t0, w0 / 2, h0, c);
      }
    }

    // Unplottable values: triangles pointing out of the bottom edge (top: +∞)
    const size = MARK_SIZE * s;
    const nan = premul(ctl.nanColor);
    for (const sr of this.series) {
      const cols = new Map<string, [number, boolean]>();
      for (let i = 0; i < sr.y.length; i += 1) {
        const val = sr.y[i];
        if (Number.isNaN(val) || Number.isFinite(ty.fwd(val))) continue;
        const X = v.px(tx.fwd(sr.x[i]));
        if (X >= L - size && X <= R + size) cols.set(`${Math.round(X)}${val > 0}`, [X, val > 0]);
      }
      for (const [X, isTop] of cols.values()) {
        this.triangle(top, X, isTop ? T : B, isTop, size + 1, premul(sr.color));
        this.triangle(top, X, isTop ? T + 1 : B - 1, isTop, size - 0.5, nan);
      }
    }

    // Rules of marks: full span where an end is NaN
    for (const sr of this.series) {
      const ru = sr.rules;
      if (!ru) continue;
      const lw = Math.max(1, Math.round(sr.width * s));
      const c = premul(sr.color, sr.opacity);
      for (let i = 0; i < ru.at.length; i += 1) {
        const horiz = ru.axis === "y";
        const p = horiz ? v.py(ty.fwd(ru.at[i])) : v.px(tx.fwd(ru.at[i]));
        if (!(horiz ? p >= T && p <= B : p >= L && p <= R)) continue;
        const [e0, e1] = horiz ? [L, R + 1] : [T, B + 1];
        const end = (u: number) => (Number.isNaN(u) ? NaN : horiz ? v.px(tx.fwd(u)) : v.py(ty.fwd(u)));
        const ab = [end(ru.lo[i]), end(ru.hi[i])];
        const [a, b] = [Number.isNaN(ab[0]) ? e0 : ab[0], Number.isNaN(ab[1]) ? e1 : ab[1]].map((u) => Math.min(Math.max(u, e0), e1)).sort((m, n) => m - n);
        const step = sr.dash ? Math.round((DASH + GAP) * s) : b - a, on = sr.dash ? Math.round(DASH * s) : step;
        for (let u = a; u < b; u += step) {
          if (horiz) top.rect(u, Math.round(p - lw / 2), Math.min(on, b - u), lw, c);
          else top.rect(Math.round(p - lw / 2), u, lw, Math.min(on, b - u), c);
        }
      }
    }

    // Intervals (with caps), then boxes over them: filled opaque (the surface mixed in), median line at y
    for (const sr of this.series) {
      const iv = sr.intervals;
      if (!iv) continue;
      const lw = Math.max(1, Math.round(sr.width * s));
      const c = premul(sr.color, sr.opacity);
      for (let i = 0; i < sr.x.length; i += 1) {
        const X = v.px(tx.fwd(sr.x[i]));
        const [ya, yb] = [iv.lo[i], iv.hi[i]].map((u) => v.py(ty.fwd(u)));
        if (!(X >= L && X <= R) || !Number.isFinite(ya) || !Number.isFinite(yb)) continue;
        const [y0, y1] = [Math.max(Math.min(ya, yb), T), Math.min(Math.max(ya, yb), B + 1)];
        if (y1 > y0) top.rect(Math.round(X - lw / 2), y0, lw, y1 - y0, c);
        const cw = iv.cap * s;
        for (const Y of [ya, yb]) if (cw > 0 && Y >= T && Y <= B + 1) top.rect(Math.round(X - cw / 2), Math.round(Y - lw / 2), Math.round(cw), lw, c);
      }
    }
    const surface = premul(SURFACE);
    for (const sr of this.series) {
      const bx = sr.box;
      if (!bx) continue;
      const fill = premul(sr.color, 0.28).map((u, j) => u + surface[j] * 0.72) as Rgba;
      const lw = Math.max(1, Math.round(1.5 * s));
      for (let i = 0; i < sr.x.length; i += 1) {
        const [l, r] = [v.px(tx.fwd(sr.x[i] - bx.width / 2)), v.px(tx.fwd(sr.x[i] + bx.width / 2))];
        const [ya, yb] = [bx.q1[i], bx.q3[i]].map((u) => Math.min(Math.max(v.py(ty.fwd(u)), T), B + 1));
        if (!(r > L && l < R) || !(Number.isFinite(ya) && Number.isFinite(yb))) continue;
        const [l0, r0] = [Math.max(l, L), Math.min(r, R + 1)];
        top.rect(l0, Math.min(ya, yb), r0 - l0, Math.max(Math.abs(ya - yb), lw), fill, { radius: Math.round(3 * s), border: lw, borderColor: premul(sr.color) });
        const Y = v.py(ty.fwd(sr.y[i]));
        const mw = Math.max(2, Math.round(2.5 * s));
        if (Y >= T && Y <= B) top.rect(l0, Math.round(Y - mw / 2), r0 - l0, mw, premul(INK));
      }
    }

    // Markers
    for (const sr of this.series) {
      if (!sr.markers) continue;
      const r = (sr.markers === "only" ? 3.5 : 4.5) * s;
      for (let i = 0; i < sr.x.length; i += 1) {
        const X = v.px(tx.fwd(sr.x[i]));
        const Y = v.py(ty.fwd(sr.y[i]));
        if (!(X >= L && X <= R && Y >= T && Y <= B)) continue;
        if (sr.sizes || sr.colors || sr.alphas) top.dot(X, Y, (sr.sizes?.[i] ?? 3.5) * s, s, premul(sr.colors?.[i] ?? sr.color, sr.alphas?.[i] ?? 1), premul(SURFACE));
        else top.dot(X, Y, r, 1.5 * s, premul(sr.color), premul(SURFACE));
      }
    }

    // Texts of marks
    for (const sr of this.series) {
      const tt = sr.texts;
      if (!tt) continue;
      for (let i = 0; i < sr.x.length; i += 1) {
        const X = v.px(tx.fwd(sr.x[i])) + tt.dx * s, Y = v.py(ty.fwd(sr.y[i])) + tt.dy * s;
        if (X >= L && X <= R && Y >= T && Y <= B && tt.text[i]) top.text(tt.text[i], X, Y, F.label, premul(sr.colors?.[i] ?? sr.color, sr.alphas?.[i] ?? 1), tt.anchor);
      }
    }

    if (ctl.legend) this.legend(top, ctl);
    if (this.colorbar) this.colorBar(q, this.colorbar);
    return [q, top];
  }

  /** Filled triangle, apex on the edge at (x, edge), pointing out, `half` wide each side. */
  private triangle(q: Quads, x: number, edge: number, up: boolean, half: number, color: Rgba): void {
    const height = Math.round(2 * half);
    for (let r = 0; r < height; r += 1) {
      const half_w = half * ((r + 0.5) / height);
      const y = up ? edge + r : edge - r - 1;
      q.rect(x - half_w, y, 2 * half_w, 1, color);
    }
  }

  private legend(q: Quads, ctl: Control): void {
    const v = this.view;
    const s = v.scale;
    const [L, T, R, B] = v.plot;
    const F = this.fonts(s).label;
    type Item = { text: string; color: string; kind: "line" | "dash" | "band" | "marker" | "bar"; width: number; opacity: number };
    const items: Item[] = [];
    for (const sr of this.series) {
      if (sr.legend && sr.label && !items.some((it) => it.text === sr.label && it.color === sr.color)) { // marks: a category once
        const kind = bandOnly(sr) || sr.fillY ? "band" : sr.bars || sr.box || sr.rects ? "bar" : sr.markers === "only" ? "marker" : sr.dash ? "dash" : "line";
        items.push({ text: sr.label, color: sr.color, kind, width: sr.width, opacity: kind === "band" ? Math.min(1, 2 * (sr.fill ?? sr.fillY)!.opacity) : sr.opacity });
      }
    }
    for (const o of ctl.overlays) {
      if (o.label) {
        items.push({ text: o.label, color: o.color ?? "#666666", kind: o.kind === "band" ? "band" : o.dash ? "dash" : "line", width: o.width ?? 1.5, opacity: o.kind === "band" ? BAND_OPACITY * 2 : 1 });
      }
    }
    if (!items.length) return;
    const pad = Math.round(8 * s);
    const sw = Math.round(22 * s);
    const gap = Math.round(8 * s);
    const lh = F.ascent + F.descent + Math.round(4 * s);
    const bw = pad * 2 + sw + gap + Math.max(...items.map((it) => this.font.measure(F, it.text)));
    const bh = pad * 2 + lh * items.length - Math.round(4 * s);
    const margin = Math.round(10 * s);
    const [pw, ph] = [R - L, B - T];

    // Corner covering the fewest points, segments and references
    const tx = axisTransform(ctl.x).fwd;
    const ty = axisTransform(ctl.y).fwd;
    const xs: number[] = [];
    const ys: number[] = [];
    this.series.forEach((sr, si) => {
      const d = sr.gen ? this.dense.get(si) : undefined;
      if (d) { // axis space already; ~1000 of them
        const stride = Math.max(1, Math.floor(d.count / 1000));
        for (let i = 0; i < d.count; i += stride) {
          xs.push(d.run.xw[i]);
          ys.push(d.run.yw[i]);
        }
      }
      for (let i = 0; i < sr.x.length; i += 1) {
        xs.push(tx(sr.x[i]));
        ys.push(ty(sr.y[i]));
      }
      xs.push(NaN);
      ys.push(NaN);
      const segs = (a?: Float64Array, b?: Float64Array) => a && b && sr.x.forEach((xv, i) => { xs.push(tx(xv), tx(xv), NaN); ys.push(ty(a[i]), ty(b[i]), NaN); });
      segs(sr.intervals?.lo, sr.intervals?.hi);
      segs(sr.box?.q1, sr.box?.q3);
      for (const b of sr.fill ? [sr.fill.lo, sr.fill.hi] : []) { // its edges, sampled
        for (let k = 0; k <= 128; k += 1) {
          const t = v.x0 + ((v.x1 - v.x0) * k) / 128;
          xs.push(t);
          ys.push(boundAt(b, t, axisTransform(ctl.x).inv, ty) ?? NaN);
        }
        xs.push(NaN);
        ys.push(NaN);
      }
      const rc = sr.rects; // marks: rect outlines, fill-of-y edges
      rc?.x1.forEach((_, i) => {
        xs.push(...[rc.x1[i], rc.x2[i], rc.x2[i], rc.x1[i], rc.x1[i]].map(tx), NaN);
        ys.push(...[rc.y1[i], rc.y1[i], rc.y2[i], rc.y2[i], rc.y1[i]].map(ty), NaN);
      });
      for (const p of sr.fillY ? [...sr.fillY.lo, ...sr.fillY.hi] : []) {
        xs.push(...p.yw, NaN);
        ys.push(...p.xw, NaN);
      }
    });
    const relX = (w: number) => (w - v.x0) / (v.x1 - v.x0);
    const relY = (w: number) => (w - v.y0) / (v.y1 - v.y0);
    const obstacles = ctl.overlays.map((o) => {
      const axis = o.kind === "band" ? (o.axis ?? "x") : o.kind === "hline" ? "y" : "x";
      const f = axis === "x" ? tx : ty;
      return { axis, range: [f(o.values[0]), f(o.values[o.values.length - 1])] as [number, number] };
    });
    const corner = getLegendCorner(xs, [ys], relX, relY, { width: bw / pw, height: bh / ph, marginX: margin / pw, marginY: margin / ph },
      obstacles as LegendObstacle[]);
    const x = corner.endsWith("right") ? R - margin - bw : L + margin;
    const y = corner.startsWith("top") ? T + margin : B - margin - bh;
    q.rect(x, y, bw, bh, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    items.forEach((it, i) => {
      const cy = y + pad + i * lh + (F.ascent + F.descent) / 2;
      const sx0 = x + pad;
      const lw = Math.max(1, Math.round(it.width * s));
      if (it.kind === "band") {
        q.rect(sx0, Math.round(cy - 5 * s), sw, Math.round(10 * s), premul(it.color, it.opacity));
      } else if (it.kind === "bar") {
        q.rect(sx0 + Math.round(4 * s), Math.round(cy - 6 * s), sw - Math.round(8 * s), Math.round(12 * s), premul(it.color, it.opacity), { radius: Math.round(2 * s) });
      } else if (it.kind === "marker") {
        q.dot(sx0 + sw / 2, cy, 4 * s, 1.5 * s, premul(it.color), premul(SURFACE));
      } else if (it.kind === "dash") {
        const d = Math.round(DASH * s * 0.8), g = Math.round(GAP * s * 0.8);
        for (let a = 0; a < sw; a += d + g) q.rect(sx0 + a, Math.round(cy - lw / 2), Math.min(d, sw - a), lw, premul(it.color));
      } else {
        q.rect(sx0, Math.round(cy - lw / 2), sw, lw, premul(it.color, it.opacity));
      }
      q.text(it.text, sx0 + sw + gap, cy, F, premul(INK), "lm");
    });
  }

  /** The colour bar of a continuous fill, right-aligned in the title row (clear of the data):
   * its label, the low end, the strip, the high end. */
  private colorBar(q: Quads, cb: ColorBar): void {
    const v = this.view;
    const s = v.scale;
    const F = this.fonts(s).label;
    const [, , R] = v.plot;
    const [w, h, gap, cy] = [Math.round(140 * s), Math.round(10 * s), Math.round(8 * s), Math.round(28 * s)];
    const hiX = R - this.font.measure(F, fmt(cb.domain[1], 4));
    q.text(fmt(cb.domain[1], 4), R, cy, F, premul(INK), "rm");
    const x0 = Math.round(hiX - gap - w);
    const n = 64;
    for (let k = 0; k < n; k += 1) {
      const rgb = colormapRgb(cb.scheme, (k + 0.5) / n, cb.invert);
      q.rect(x0 + Math.round((w * k) / n), Math.round(cy - h / 2), Math.round((w * (k + 1)) / n) - Math.round((w * k) / n), h, [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1]);
    }
    const loX = x0 - gap;
    q.text(fmt(cb.domain[0], 4), loX, cy, F, premul(INK), "rm");
    q.text(cb.label, loX - this.font.measure(F, fmt(cb.domain[0], 4)) - Math.round(14 * s), cy, F, premul(INK2), "rm");
  }

  // Hover ----------------------------------------------------------------------------------------

  private hover(ctl: Control): Hover | null {
    const hv = ctl.hover;
    const v = this.view;
    const [L, T, R, B] = v.plot;
    if (!hv) return null;
    const [hx, hy] = [Number(hv[0]), Number(hv[1])];
    if (!(hx >= L && hx < R && hy >= 0 && hy < v.h)) return null;
    const cx = Math.floor(hx) + 0.5;
    const t = v.x0 + ((cx - L) / (R - L)) * (v.x1 - v.x0);
    const tx = axisTransform(ctl.x);
    const ty = axisTransform(ctl.y);
    const info: Hover = { pixel: [hx, hy], x: tx.inv(t), rows: [] };
    const range = (a: number, b: number) => `${valueText(ctl.y, a)} … ${valueText(ctl.y, b)}`;
    let best = Infinity;
    this.series.forEach((sr, si) => {
      if (sr.tips) { // marks: the rect under the cursor (bars: in its band), else the nearest point (lines: in x)
        const rc = sr.rects, lineLike = sr.curves.length > 0;
        let [k, dk, at] = [-1, Infinity, [0, 0] as [number, number]];
        for (let i = 0; i < sr.x.length; i += 1) {
          let d: number, p: [number, number];
          if (rc) {
            const [xa, xb] = [rc.x1[i], rc.x2[i]].map((u) => v.px(tx.fwd(u))), [ya, yb] = [rc.y1[i], rc.y2[i]].map((u) => v.py(ty.fwd(u)));
            const inX = hx >= Math.min(xa, xb) && hx <= Math.max(xa, xb), inY = hy >= Math.min(ya, yb) && hy <= Math.max(ya, yb);
            if (rc.end === "y" ? !inX : rc.end === "x" ? !inY : !(inX && inY)) continue;
            p = rc.end === "y" ? [(xa + xb) / 2, yb] : rc.end === "x" ? [xb, (ya + yb) / 2] : [(xa + xb) / 2, (ya + yb) / 2];
            d = inX && inY ? 0 : rc.end === "y" ? Math.abs(yb - hy) : Math.abs(xb - hx);
          } else {
            p = [v.px(tx.fwd(sr.x[i])), v.py(ty.fwd(sr.y[i]))];
            if (!(p[0] >= L && p[0] <= R && p[1] >= T && p[1] <= B)) continue;
            d = lineLike ? Math.abs(p[0] - hx) : Math.hypot(p[0] - hx, p[1] - hy);
          }
          if (d < dk) [k, dk, at] = [i, d, p];
        }
        const d = (lineLike ? Math.abs(at[1] - hy) : dk) - si * 1e-9;
        if (k < 0 || !(d < best)) return;
        best = d;
        info.series = { label: sr.label, color: sr.colors?.[k] ?? sr.color, value: sr.y[k], pixel: [Math.min(Math.max(at[0], L), R), Math.min(Math.max(at[1], T), B)], text: "" };
        info.rows = sr.tips[k].map(([key, val]) => [key, val, INK]);
        return;
      }
      let yw: number | null = null;
      let k = -1; // the hovered sample, if one
      if (sr.fn && sr.domain) {
        const xv = tx.inv(t);
        if (xv >= sr.domain[0] && xv <= sr.domain[1]) yw = ty.fwd(sr.fn(xv));
      } else {
        // nearest to the cursor of the runs' values at t (x need not be monotonic)
        for (const p of sr.runs) {
          const v = pieceAt(p, t);
          if (v !== null && (yw === null || Math.abs(this.view.py(v) - hy) < Math.abs(this.view.py(yw) - hy))) yw = v;
        }
        const gw = glyphWidth(sr);
        if (sr.markers === "only" || gw || sr.intervals) { // nearest sample (bars, boxes: the one under the cursor)
          for (let i = 0; i < sr.x.length; i += 1) if (k < 0 || Math.abs(tx.fwd(sr.x[i]) - t) < Math.abs(tx.fwd(sr.x[k]) - t)) k = i;
          if (k >= 0 && gw && Math.abs(sr.x[k] - tx.inv(t)) > gw / 2) k = -1;
          yw = k >= 0 ? ty.fwd(sr.y[k]) : null;
        }
      }
      if (yw === null || !Number.isFinite(yw)) return;
      const py = v.py(yw);
      const d = Math.abs(py - hy) - (si * 1e-9); // later series win ties (drawn on top)
      if (d < best && py >= T - 1 && py <= B + 1) {
        best = d;
        info.series = { label: sr.label || `series ${si}`, color: sr.color, value: ty.inv(yw), pixel: [cx, py] };
        info.rows = [];
        if (k >= 0 && sr.box) info.rows.push(["q₁ … q₃", range(sr.box.q1[k], sr.box.q3[k]), sr.color]);
        if (k >= 0 && sr.intervals) info.rows.push([sr.intervals.label, range(sr.intervals.lo[k], sr.intervals.hi[k]), sr.color]);
      }
    });
    for (const sr of this.series) { // bands at x
      if (!bandOnly(sr)) continue;
      const [a, b] = [boundAt(sr.fill!.lo, t, tx.inv, ty.fwd), boundAt(sr.fill!.hi, t, tx.inv, ty.fwd)];
      if (a !== null && b !== null && Number.isFinite(a) && Number.isFinite(b)) info.rows.push([sr.label || "band", range(ty.inv(a), ty.inv(b)), sr.color]);
    }
    return info;
  }

  private drawHover(q: Quads, info: Hover, ctl: Control): void {
    const v = this.view;
    const s = v.scale;
    const [L, T, R, B] = v.plot;
    q.rect(Math.floor(info.pixel[0]), T, 1, B - T + 1, premul(INK2, 0.6), { clip: true });
    if (info.series) {
      const cy = Math.floor(info.series.pixel[1]);
      if (cy >= T && cy <= B) q.rect(L, cy, R - L + 1, 1, premul(INK2, 0.35), { clip: true });
    }
    q.append(this.top!);
    if (info.series) {
      q.dot(info.series.pixel[0], info.series.pixel[1], 6 * s, 2 * s, premul(info.series.color), premul(HOVER_RING), true);
    }
    const font = this.fonts(s).label;
    const cat = ctl.x.categories?.[Math.round(info.x)];
    const xText = cat !== undefined && Math.abs(info.x - Math.round(info.x)) <= 0.5 ? String(cat) : ctl.x.ticks === "pi" ? `${fmt(info.x / Math.PI, 4)}π` : fmt(info.x, 6);
    const rows: [string, string, string][] = [[ctl.xlabel || "x", xText, INK]];
    if (info.series && (info.series.label || info.series.text === undefined)) rows.push([info.series.label, info.series.text ?? valueText(ctl.y, info.series.value), info.series.color]);
    rows.push(...info.rows);
    const gapRow = Math.round(3 * s);
    const lh = font.ascent + font.descent + gapRow;
    const kw = Math.max(...rows.map(([k]) => this.font.measure(font, k))) + Math.round(12 * s);
    const vw = Math.max(...rows.map(([, val]) => this.font.measure(font, val)));
    const pad = Math.round(8 * s);
    const lw = Math.ceil(pad * 2 + kw + vw);
    const lhTotal = pad * 2 + lh * rows.length - gapRow;
    const [hx, hy] = info.pixel;
    const gap = Math.round(14 * s);
    let x = hx + gap + lw <= R ? hx + gap : hx - gap - lw;
    let y = hy + gap + lhTotal <= B ? hy + gap : hy - gap - lhTotal;
    x = Math.trunc(Math.min(Math.max(x, 0), v.w - lw));
    y = Math.trunc(Math.min(Math.max(y, 0), v.h - lhTotal));
    q.rect(x, y, lw, lhTotal, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    rows.forEach(([k, val, color], i) => {
      const ry = y + pad + i * lh;
      q.text(k, x + pad, ry, font, premul(INK2), "lt");
      q.text(val, x + lw - pad, ry, font, premul(color), "rt");
    });
  }

  // Frame ----------------------------------------------------------------------------------------

  /** The control with its marks compiled into series after its own, with the categories they
   * imply (recompiled when the marks, scales or their files change). */
  private withMarks(ctl: Control): Control {
    const marks = Array.isArray(ctl.marks) ? ctl.marks : [];
    if (!marks.length) {
      this.colorbar = undefined;
      return ctl;
    }
    const key = JSON.stringify([marks, ctl.x, ctl.y, ctl.color, ctl.r, marks.map((m) => (m?.file ? fileMtime(m.file) : 0))]);
    if (key !== this.marksKey) { // a failing key is remembered too: not recompiled (its files re-read) each frame
      this.marksKey = key;
      this.marksError = undefined;
      try {
        this.marked = compileMarks(marks, { x: ctl.x, y: ctl.y }, { color: ctl.color, r: ctl.r }, PALETTE, INK, tableFile);
      } catch (err) {
        this.marksError = err;
      }
    }
    if (this.marksError) throw this.marksError; // the frame is refused: the last good one (and its colour bar) stands
    const c = this.marked!;
    this.colorbar = c.colorbar;
    return { ...ctl, series: [...ctl.series, ...c.series], x: c.x, y: c.y };
  }

  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    ctl = this.withMarks(ctl);
    this.setData(ctl, Array.isArray(ctl.marks) && ctl.marks.length ? this.marksKey : "");
    this.ensureDense(ctl);
    this.setWindow(ctl);
    const v = this.view;
    const { w, h } = v;
    const info = this.hover(ctl);

    const q = new Quads(this.font);
    q.append(this.chrome!);
    q.layer(0, 1, v.plot);
    if (info) {
      this.drawHover(q, info, ctl);
    } else {
      q.append(this.top!);
    }

    const encoder = device.createCommandEncoder();
    this.overlay.encode(encoder, q, [this.layer.currentTexture()], v.plot);
    const pitch = rowBytes(w);
    if (this.frameReads[slot]?.size !== pitch * h) {
      this.frameReads[slot]?.destroy();
      this.frameReads[slot] = makeReadBuffer(device, pitch * h);
    }
    const frameRead = this.frameReads[slot];
    encoder.copyTextureToBuffer({ texture: this.overlay.out! }, { buffer: frameRead, bytesPerRow: pitch, rowsPerImage: h }, [w, h]);
    device.queue.submit([encoder.finish()]);
    return {
      done: frameRead.mapAsync(0x01),
      read: (alloc) => {
        const src = new Uint8Array(frameRead.getMappedRange());
        const rgba = alloc(w * h * 4);
        if (pitch === w * 4) {
          rgba.set(src.subarray(0, w * h * 4));
        } else {
          for (let y = 0; y < h; y += 1) rgba.set(src.subarray(y * pitch, y * pitch + w * 4), y * w * 4);
        }
        frameRead.unmap();
        return { width: w, height: h, rgba, hover: info };
      },
    };
  }

  state(ctl: Control, generation: number, hover: unknown, ms: number): unknown {
    const v = this.view;
    const tx = axisTransform(ctl.x);
    const ty = axisTransform(ctl.y);
    return {
      generation,
      width: v.w,
      height: v.h,
      x0: tx.inv(v.x0),
      x1: tx.inv(v.x1),
      ymin: ty.inv(v.y0),
      ymax: ty.inv(v.y1),
      domain: [tx.inv(this.extent[0]), tx.inv(this.extent[1])],
      plotRect: [...v.plot],
      series: this.series.map((sr) => sr.label),
      ...(this.series.some((sr) => sr.stats) ? { stats: this.series.filter((sr) => sr.stats).map((sr) => ({ label: sr.label, ...sr.stats })) } : {}),
      hover,
      renderMs: Math.round(ms * 1000) / 1000,
    };
  }
}

if (import.meta.main) {
  await main<Control>({
    name: "line.ts",
    defaults: DEFAULTS as Control,
    create: LinePlot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
  });
}
