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
import { makeReadBuffer, requestGpuDevice, rowBytes, TextureTarget } from "./lib/gpu.ts";
import { NurbsGpuRenderer } from "./lib/gpu_line.ts";
import { getLegendCorner, type LegendObstacle } from "./lib/legend.ts";
import { LayerCompositor } from "./lib/layers.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba } from "./lib/overlay.ts";
import { getHermiteExtent, getPchipSlopes, getSplineSlopes, hermiteAt, hermiteNurbs, linspace } from "./lib/pchip.ts";
import { ChordRenderer } from "./lib/chords.ts";
import { fnSampler, type FnSpec, resolveFn } from "./lib/generators.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";

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
  sample?: { fn: FnSpec; n: number }; // x, y = n even samples of fn over its domain (y from the GPU)
  fn?: FnSpec; // a generator (lib/generators.ts), sampled on the GPU over the visible window
  label?: string;
  color?: string;
  width?: number; // CSS px
  dash?: boolean;
  markers?: boolean | "only";
  opacity?: number; // < 1: in the faded layer
  interpolation?: "linear" | "spline" | "pchip";
  envelope?: { type: "cummin" | "cummax"; mode?: "add" | "replace"; reverse?: boolean; color?: string };
  legend?: boolean;
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
}

interface Series {
  label: string;
  color: string;
  width: number;
  dash: boolean;
  markers: boolean | "only";
  opacity: number;
  legend: boolean;
  x: Float64Array; // values
  y: Float64Array;
  runs: Piece[]; // increasing in x: hover, extents, dashes
  curves: NurbsCurve[]; // drawn
  fn?: (x: number) => number; // f64, for the read-out
  gen?: ReturnType<typeof resolveFn>; // sampled on the GPU
  domain?: [number, number];
}

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
    out.push({ xw, yw, slopes });
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
  const runs = ps.flatMap(monotoneRuns);
  const curves = interp === "linear"
    ? ps.filter((p) => p.xw.length >= 2).map(pieceNurbs)
    : runs.map(pieceNurbs);
  return { runs, curves };
}

/** `fn` at `n` even samples of its domain: x (f64) and y (the GPU's, prepared by `LinePlot.prepare`). */
type Samples = (fn: FnSpec, n: number) => { x: Float64Array; y: Float64Array };

function resolveSeries(ctl: Control, samples: Samples): Series[] {
  const fx = axisTransform(ctl.x).fwd;
  const fy = axisTransform(ctl.y).fwd;
  const out: Series[] = [];
  let auto = 0;
  const nextColor = () => PALETTE[auto++ % PALETTE.length];
  for (const spec of ctl.series) {
    const color = spec.color ?? nextColor();
    const base = {
      label: spec.label ?? "",
      color,
      width: spec.width ?? 1.5,
      dash: Boolean(spec.dash),
      markers: spec.markers ?? false,
      opacity: spec.opacity ?? 1,
      legend: spec.legend ?? Boolean(spec.label),
    };
    if (spec.fn) {
      const fn = resolveFn(spec.fn);
      out.push({ ...base, x: new Float64Array(0), y: new Float64Array(0), runs: [], curves: [], fn: fn.f, gen: fn, domain: fn.domain });
      continue;
    }
    let x: Float64Array, y: Float64Array;
    if (spec.sample) {
      ({ x, y } = samples(spec.sample.fn, spec.sample.n));
    } else {
      y = Float64Array.from(spec.y ?? [], Number);
      x = spec.x ? Float64Array.from(spec.x, Number) : Float64Array.from(y, (_, i) => i);
    }
    const interp = spec.interpolation ?? "linear";
    const env = spec.envelope;
    if (env && (env.mode ?? "add") === "replace") {
      y = envelope(y, env.type, Boolean(env.reverse));
    }
    const geo = geometry(x, y, fx, fy, interp);
    out.push({ ...base, x, y, runs: geo.runs, curves: base.markers === "only" ? [] : geo.curves });
    if (env && (env.mode ?? "add") === "add") {
      const ey = envelope(y, env.type, Boolean(env.reverse));
      const label = ctl.series.length > 1 && spec.label ? `${spec.label}, ${envelopeLabel(env.type, Boolean(env.reverse))}` : envelopeLabel(env.type, Boolean(env.reverse));
      const ecolor = env.color ?? (ctl.series.length > 1 ? color : PALETTE.find((c) => c !== color)!);
      out.push({
        ...base, label, color: ecolor, width: ENVELOPE_WIDTH, dash: false, markers: false, opacity: 1, legend: true,
        x, y: ey, ...geometry(x, ey, fx, fy, "linear"),
      });
    }
  }
  return out;
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
  let lo = 0, hi = n - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (p.xw[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  const f = (t - p.xw[lo]) / (p.xw[lo + 1] - p.xw[lo]);
  return p.yw[lo] + f * (p.yw[lo + 1] - p.yw[lo]);
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
  series?: { label: string; color: string; value: number; pixel: [number, number] };
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

/** Dense samples of a generator, computed on the GPU into `buffer` (drawn from there), read back
 * into `run` (y range, dashes, legend); kept while the window stays within them at about the
 * same zoom (panning doesn't resample). */
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

const sampleKey = (fn: FnSpec, n: number) => JSON.stringify([fn, n]);

export class LinePlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly frameReads: GPUBuffer[] = [];
  private readonly scratch: TextureTarget; // one series at a time, opaque
  private readonly layer: TextureTarget; // all series, composited in order
  private readonly compositor: LayerCompositor;
  private dataKey?: string;
  private geomKey?: string;
  private chromeKey?: string;
  series: Series[] = [];
  private curves: { series: number; layer: CurveLayer }[] = [];
  private dense = new Map<number, Dense>();
  private readonly chords: ChordRenderer;
  private sampleCache = new Map<string, { x: Float64Array; y: Float64Array }>();
  private extent: [number, number] = [0, 1]; // axis-space x of all data
  view!: View;
  private ticks!: [ReturnType<typeof axisTicks>, ReturnType<typeof axisTicks>];
  private chrome?: Quads; // below the curves
  private top?: Quads; // above them: references, marks, markers, legend

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
  setData(ctl: Control): void {
    const key = JSON.stringify([ctl.series, ctl.x, ctl.y]);
    if (key === this.dataKey) {
      return;
    }
    const curves: { series: number; layer: CurveLayer }[] = [];
    let series: Series[];
    try {
      series = resolveSeries(ctl, (fn, n) => this.sampleCache.get(sampleKey(fn, n))!);
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
      for (const v of s.x) {
        const w = fx(v);
        if (Number.isFinite(w)) {
          lo = Math.min(lo, w);
          hi = Math.max(hi, w);
        }
      }
      if (s.domain) {
        lo = Math.min(lo, fx(s.domain[0]));
        hi = Math.max(hi, fx(s.domain[1]));
      }
    }
    for (const c of this.curves) c.layer.dispose();
    for (const d of this.dense.values()) d.buffer.destroy();
    this.dense.clear();
    this.series = series;
    this.curves = curves;
    this.extent = lo < hi ? [lo, hi] : lo === hi ? [lo - 0.5, lo + 0.5] : [0, 1];
    this.dataKey = key;
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

  /** `sample` specs' values, from the GPU (once per spec). */
  private async prepare(ctl: Control): Promise<void> {
    for (const spec of ctl.series) {
      if (!spec.sample) continue;
      const key = sampleKey(spec.sample.fn, spec.sample.n);
      if (this.sampleCache.has(key)) continue;
      const fn = resolveFn(spec.sample.fn);
      const n = Math.max(2, Math.trunc(spec.sample.n));
      const { y } = await fnSampler(this.device).values(fn, fn.domain[0], fn.domain[1], n);
      if (this.sampleCache.size > 64) this.sampleCache.clear();
      this.sampleCache.set(key, { x: linspace(fn.domain[0], fn.domain[1], n), y: Float64Array.from(y) });
    }
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

  /** (Re)samples, on the GPU and in one readback, the generators whose dense samples don't cover
   * the window: over three window widths, about 3 per pixel. */
  private async ensureDense(ctl: Control): Promise<void> {
    const { tx, x0, x1 } = this.window(ctl);
    if (!(x1 > x0)) return;
    const step = denseStep(Math.trunc(ctl.width), Math.trunc(ctl.height), x0, x1);
    const todo: { si: number; ca: number; cb: number; wa: number; m: number; dw: number }[] = [];
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
      const m = Math.trunc(Math.min(Math.max((wb - wa) / step, 256), 65536));
      todo.push({ si, ca, cb, wa, m, dw: (wb - wa) / (m - 1) });
    });
    if (!todo.length) return;
    const sampler = fnSampler(this.device);
    const outs = todo.map(({ si, m }) => {
      const old = this.dense.get(si);
      if (old && old.buffer.size >= m * 16) return old.buffer;
      old?.buffer.destroy();
      this.dense.delete(si);
      return sampler.buffer(m);
    });
    const jobs = todo.map(({ si, wa, m, dw }) => ({
      fn: this.series[si].gen!, w0: wa, dw, count: m, origin: [wa + (dw * (m - 1)) / 2, 0] as [number, number], x: ctl.x, y: ctl.y,
    }));
    const ys = await sampler.graphs(jobs, outs);
    todo.forEach(({ si, ca, cb, wa, m, dw }, k) => {
      const run: Piece = { xw: Float64Array.from({ length: m }, (_, i) => wa + i * dw), yw: Float64Array.from(ys[k].y) };
      this.dense.set(si, { a: ca, b: cb, step: dw, run, buffer: outs[k], count: m, origin: jobs[k].origin });
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
    const geomKey = JSON.stringify([this.dataKey, w, h, x0, x1, ctl.ymin, ctl.ymax, ctl.overlays, Boolean(ctl.ylabel), Boolean(ctl.xlabel)]);
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
    this.series.forEach((_, si) => {
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
    for (const o of ctl.overlays) {
      if (o.kind === "hline" || (o.kind === "band" && o.axis === "y")) o.values.forEach((v) => take(ty.fwd(v)));
    }
    if (!(lo <= hi)) [lo, hi] = [-1, 1];
    const pad = hi > lo ? (hi - lo) * Y_MARGIN : 0.5;
    let y0 = ctl.ymin !== null && Number.isFinite(ty.fwd(ctl.ymin)) ? ty.fwd(ctl.ymin) : lo - pad;
    let y1 = ctl.ymax !== null && Number.isFinite(ty.fwd(ctl.ymax)) ? ty.fwd(ctl.ymax) : hi + pad;
    if (!(y1 > y0)) [y0, y1] = [y0 - 0.5, y0 + 0.5];

    // Ticks, then the left margin from the widest label
    const yticks = axisTicks(ctl.y, ty.inv(y0), ty.inv(y1), (B - T) / s);
    const yw = Math.max(0, ...yticks.map((t) => this.font.measure(F.tick, t.label)));
    const L = Math.max(Math.round(yw + 22 * s), Math.round(50 * s));
    const xticks = axisTicks({ ...ctl.x, index: ctl.x.index ?? ctl.series.every((sp) => !sp.x && !sp.sample && !sp.fn) }, tx.inv(x0), tx.inv(x1), (R - L) / s);
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

    // Markers
    for (const sr of this.series) {
      if (!sr.markers) continue;
      const r = (sr.markers === "only" ? 3.5 : 4.5) * s;
      for (let i = 0; i < sr.x.length; i += 1) {
        const X = v.px(tx.fwd(sr.x[i]));
        const Y = v.py(ty.fwd(sr.y[i]));
        if (X >= L && X <= R && Y >= T && Y <= B) top.dot(X, Y, r, 1.5 * s, premul(sr.color), premul(SURFACE));
      }
    }

    if (ctl.legend) this.legend(top, ctl);
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
    type Item = { text: string; color: string; kind: "line" | "dash" | "band" | "marker"; width: number; opacity: number };
    const items: Item[] = [];
    for (const sr of this.series) {
      if (sr.legend && sr.label) {
        items.push({ text: sr.label, color: sr.color, kind: sr.markers === "only" ? "marker" : sr.dash ? "dash" : "line", width: sr.width, opacity: sr.opacity });
      }
    }
    for (const o of ctl.overlays) {
      if (o.label) {
        items.push({ text: o.label, color: o.color ?? "#666666", kind: o.kind === "band" ? "band" : o.dash ? "dash" : "line", width: o.width ?? 1.5, opacity: 1 });
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
        q.rect(sx0, Math.round(cy - 5 * s), sw, Math.round(10 * s), premul(it.color, BAND_OPACITY * 2));
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
    const info: Hover = { pixel: [hx, hy], x: tx.inv(t) };
    let best = Infinity;
    this.series.forEach((sr, si) => {
      let yw: number | null = null;
      if (sr.fn && sr.domain) {
        const xv = tx.inv(t);
        if (xv >= sr.domain[0] && xv <= sr.domain[1]) yw = ty.fwd(sr.fn(xv));
      } else {
        // nearest to the cursor of the runs' values at t (x need not be monotonic)
        for (const p of sr.runs) {
          const v = pieceAt(p, t);
          if (v !== null && (yw === null || Math.abs(this.view.py(v) - hy) < Math.abs(this.view.py(yw) - hy))) yw = v;
        }
        if (sr.markers === "only") { // nearest sample
          let k = -1;
          for (let i = 0; i < sr.x.length; i += 1) if (k < 0 || Math.abs(tx.fwd(sr.x[i]) - t) < Math.abs(tx.fwd(sr.x[k]) - t)) k = i;
          if (k >= 0) yw = ty.fwd(sr.y[k]);
        }
      }
      if (yw === null || !Number.isFinite(yw)) return;
      const py = v.py(yw);
      const d = Math.abs(py - hy) - (si * 1e-9); // later series win ties (drawn on top)
      if (d < best && py >= T - 1 && py <= B + 1) {
        best = d;
        info.series = { label: sr.label || `series ${si}`, color: sr.color, value: ty.inv(yw), pixel: [cx, py] };
      }
    });
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
    const rows: [string, string, string][] = [[ctl.xlabel || "x", ctl.x.ticks === "pi" ? `${fmt(info.x / Math.PI, 4)}π` : fmt(info.x, 6), INK]];
    if (info.series) rows.push([info.series.label, valueText(ctl.y, info.series.value), info.series.color]);
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

  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    await this.prepare(ctl);
    this.setData(ctl);
    await this.ensureDense(ctl);
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
