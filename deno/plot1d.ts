/**
 * Interactive 1D plot of the PCHIP interpolant of a sampled function, as an exact cubic NURBS
 * curve evaluated and drawn by the WebGPU curve renderer (`NurbsGpuRenderer` of lib/gpu_line.ts,
 * with `NurbsCurveSampler` of lib/gpu.ts and `getSegmentCounts` of lib/nurbs.ts), in Deno. Same
 * CLI and serve protocol as surface.ts (cf. lib/serve.ts).
 *
 *   deno run -A plot1d.ts png CONTROL.json -o out.png
 *   deno run -A plot1d.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 *   deno run -A plot1d.ts samples CONTROL.json OUT.json   # GPU samples of the curves (checks)
 *
 * Each curve (PCHIP, the not-a-knot cubic spline, and the true function as a dense degree-1
 * NURBS) is drawn by its own renderer into its own f16 texture, redrawn only when
 * the window changes; the axes, text (bitmap font atlas), dots, hover read-out and those layers
 * are then composited by lib/overlay.ts.
 */

import { evaluateNurbs, getKnotSpans, getNurbsError, getSegmentCounts, type KnotSpan, type NurbsCurve } from "./lib/nurbs.ts";
import { makeReadBuffer, readBuffer, requestGpuDevice, rowBytes, TextureTarget } from "./lib/gpu.ts";
import { NurbsGpuRenderer } from "./lib/gpu_line.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba } from "./lib/overlay.ts";
import { FUNCTIONS, getPchipSlopes, hermiteAt, hermiteNurbs, linspace, notAKnotSlopes } from "./lib/pchip.ts";
import { loadControl, main, type Pending, renderOnce, type Renderer as ServeRenderer } from "./lib/serve.ts";

// GitHub Dark (terminal theme)
const BLUE = "#56B4E9", ORANGE = "#E69F00", INK = "#c9d1d9", INK2 = "#8b949e", GRID = "#262c36";
const SURFACE = "#101216", ZERO = "#484f58", HOVER_RING = "#e6edf3";
const TRUE_ALPHA = 0.55;
const PCHIP_WIDTH = 2.5, TRUE_WIDTH = 1.2, CUBIC_WIDTH = 1.5;
const DOT_RADIUS = 5.5, DOT_RING = 1.5; // 8 px of fill inside a 1.5 px ring
const Y_MARGIN = 0.08;
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];

export const DEFAULTS = {
  width: 1200,
  height: 750,
  function: "bumps",
  n: 32,
  x0: null as number | null, // null: full domain
  x1: null as number | null,
  ymin: null as number | null, // null: fit what is visible, 8% margin
  ymax: null as number | null,
  showTrue: true,
  showSamples: true,
  compare: "none" as "none" | "cubic",
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

// --- Formatting ------------------------------------------------------------------------------

/** Ticks at 1, 2, 2.5 or 5 × 10^k, about `target` of them, and their decimals. */
function niceTicks(lo: number, hi: number, target: number): [number[], number] {
  const span = hi - lo;
  if (!(span > 0)) {
    return [[], 0];
  }
  const raw = span / Math.max(target, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  let m = 10;
  for (const c of [1, 2, 2.5, 5, 10]) {
    if (c * mag >= raw * 0.999) {
      m = c;
      break;
    }
  }
  const step = m * mag;
  const ticks: number[] = [];
  for (let i = Math.ceil(lo / step - 1e-9); i <= Math.floor(hi / step + 1e-9); i += 1) {
    ticks.push(i * step);
  }
  return [ticks, Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (m === 2.5 ? 1 : 0))];
}

const minus = (s: string) => s.replaceAll("-", "−");

function fmtTick(v: number, decimals: number): string {
  let s = v.toFixed(decimals);
  if (Number(s) === 0) {
    s = s.replace(/^-/, "");
  }
  return minus(s);
}

const stripZeros = (s: string) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);

/** Python's `f"{v:.{p}g}"` and `f"{v:.{d}e}"`. */
function fmtE(v: number, d: number, strip = false): string {
  const [mant, exp] = v.toExponential(d).split("e");
  const e = Number(exp);
  return `${strip ? stripZeros(mant) : mant}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
}

function fmtG(v: number, p: number): string {
  const e = Number(v.toExponential(p - 1).split("e")[1]);
  return e < -4 || e >= p ? fmtE(v, p - 1, true) : stripZeros(v.toFixed(Math.max(0, p - 1 - e)));
}

function fmtNum(v: number, digits = 5): string {
  if (v === 0) {
    return "0";
  }
  return minus(Math.abs(v) >= 1e-4 && Math.abs(v) < 1e6 ? fmtG(v, digits) : fmtE(v, digits - 2));
}

// --- Curves ----------------------------------------------------------------------------------

/** A curve drawn by `NurbsGpuRenderer` into its own (f16, premultiplied) texture. */
class CurveLayer {
  readonly canvas: TextureTarget;
  readonly renderer: NurbsGpuRenderer;
  readonly spans: KnotSpan[];
  readonly origin: [number, number];

  constructor(private readonly device: GPUDevice, readonly curve: NurbsCurve) {
    const error = getNurbsError(curve);
    if (error) {
      throw new Error(error);
    }
    const p = curve.points;
    const xs = Array.from({ length: p.length / 2 }, (_, i) => p[i * 2]);
    const ys = Array.from({ length: p.length / 2 }, (_, i) => p[i * 2 + 1]);
    // Centre of the control points, to keep precision in f32
    this.origin = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
    this.spans = getKnotSpans(curve);
    this.canvas = new TextureTarget(device, FRAME_FORMAT);
    this.renderer = new NurbsGpuRenderer(device, this.canvas, curve, this.spans, this.origin);
  }

  /** Chords from the zoom, then the draw (submitted right away). */
  draw(view: Window, width: number, color: Rgba): void {
    this.canvas.width = view.w;
    this.canvas.height = view.h;
    const ppu: [number, number] = [view.sx, view.sy];
    this.renderer.setSegmentCounts(getSegmentCounts(this.curve, this.spans, ppu));
    this.renderer.draw({ scale: ppu, offset: view.offset(this.origin), size: [view.w, view.h], width, color });
  }

  /** GPU samples (x, y, u per sample, data coordinates; u of `getGpuParams`), for checks. */
  async samples(): Promise<number[][]> {
    const { sampler } = this.renderer;
    const data = new Float32Array(await readBuffer(this.device, sampler.samples!, sampler.sampleCount * 16));
    return Array.from({ length: sampler.sampleCount }, (_, i) => [data[i * 4] + this.origin[0], data[i * 4 + 1] + this.origin[1], data[i * 4 + 3]]);
  }

  dispose(): void {
    this.renderer.dispose();
  }
}

interface Window {
  x0: number;
  x1: number;
  ymin: number;
  ymax: number;
  maxError: number | null;
  plot: [number, number, number, number]; // l, t, r, b in pixels
  scale: number;
  w: number;
  h: number;
  sx: number; // pixels per unit
  sy: number;
  compare: boolean;
  showTrue: boolean;
  showSamples: boolean;
  offset(origin: [number, number]): [number, number]; // pixels from the centre (y up) of `origin`
  px(x: number): number;
  py(y: number): number;
}

interface Hover {
  pixel: [number, number];
  t: number;
  pchip: number | null;
  f: number | null;
  error: number | null;
  curvePixel?: [number, number];
}

// --- Plot ------------------------------------------------------------------------------------

export class Plot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly frameReads: GPUBuffer[] = []; // per readback slot (frames in flight)
  private dataKey?: string;
  private windowKey?: string;
  private name = "";
  private f!: (x: number) => number;
  domain: [number, number] = [0, 1];
  n = 0;
  xs = new Float64Array(0);
  ys = new Float64Array(0);
  slopes = new Float64Array(0);
  private cubicSlopes?: Float64Array;
  layers: { pchip?: CurveLayer; cubic?: CurveLayer; true?: CurveLayer } = {};
  win!: Window;
  private chrome?: Quads;

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.overlay = new OverlayRenderer(device, font);
  }

  static async create(): Promise<Plot> {
    const device = await requestGpuDevice(); // the shared device
    if (!device) {
      throw new Error("No WebGPU device");
    }
    return new Plot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  // Data: samples, interpolants and their NURBS --------------------------------------------

  setData(ctl: Control): void {
    const key = JSON.stringify([ctl.function, Math.trunc(ctl.n)]);
    if (key === this.dataKey) {
      return;
    }
    const fn = FUNCTIONS[ctl.function];
    if (!fn) {
      throw new Error(`function must be one of ${Object.keys(FUNCTIONS)}`);
    }
    const n = Math.max(2, Math.trunc(ctl.n));
    this.name = ctl.function;
    this.f = fn.f;
    this.domain = fn.domain;
    this.n = n;
    this.xs = linspace(fn.domain[0], fn.domain[1], n);
    this.ys = this.xs.map(fn.f);
    this.slopes = getPchipSlopes(this.xs, this.ys); // Fritsch–Carlson, as scipy
    this.cubicSlopes = n >= 4 ? notAKnotSlopes(this.xs, this.ys) : undefined;
    this.layers.pchip?.dispose();
    this.layers.cubic?.dispose();
    this.layers.pchip = new CurveLayer(this.device, hermiteNurbs(this.xs, this.ys, this.slopes));
    this.layers.cubic = this.cubicSlopes ? new CurveLayer(this.device, hermiteNurbs(this.xs, this.ys, this.cubicSlopes)) : undefined;
    this.dataKey = key;
    this.windowKey = undefined;
  }

  pchip(t: number): number {
    return hermiteAt(this.xs, this.ys, this.slopes, t);
  }

  // Window: view, dense true function, layers, chrome --------------------------------------

  private layout(w: number, h: number): [number, [number, number, number, number]] {
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    return [s, [Math.round(66 * s), Math.round(54 * s), w - Math.round(22 * s), h - Math.round(40 * s)]];
  }

  setWindow(ctl: Control): void {
    const w = Math.trunc(ctl.width);
    const h = Math.trunc(ctl.height);
    const [a, b] = this.domain;
    const x0 = ctl.x0 ?? a;
    const x1 = ctl.x1 ?? b;
    if (!(x1 > x0)) {
      throw new Error("x1 must be above x0");
    }
    const compare = ctl.compare === "cubic" && this.layers.cubic !== undefined;
    const key = JSON.stringify([this.dataKey, w, h, x0, x1, ctl.ymin, ctl.ymax, Boolean(ctl.showTrue), Boolean(ctl.showSamples), compare]);
    if (key === this.windowKey) {
      return;
    }
    const [scale, plot] = this.layout(w, h);
    const [L, T, R, B] = plot;
    const pw = R - L;
    const ph = B - T;

    // Visible part of the domain, sampled densely (true function, fit, max error)
    const v0 = Math.max(x0, a);
    const v1 = Math.min(x1, b);
    let xd = new Float64Array(0);
    let fd = xd;
    let maxError: number | null = null;
    let lo = Infinity;
    let hi = -Infinity;
    if (v1 > v0) {
      const m = Math.trunc(Math.min(Math.max((3 * pw * (v1 - v0)) / (x1 - x0), 256), 16384));
      xd = linspace(v0, v1, m);
      fd = xd.map(this.f);
      maxError = 0;
      for (let i = 0; i < m; i += 1) {
        const p = this.pchip(xd[i]);
        maxError = Math.max(maxError, Math.abs(p - fd[i]));
        lo = Math.min(lo, p);
        hi = Math.max(hi, p);
        if (ctl.showTrue) {
          lo = Math.min(lo, fd[i]);
          hi = Math.max(hi, fd[i]);
        }
        if (compare) {
          const c = hermiteAt(this.xs, this.ys, this.cubicSlopes!, xd[i]);
          lo = Math.min(lo, c);
          hi = Math.max(hi, c);
        }
      }
    }
    if (!(lo <= hi)) {
      [lo, hi] = [-1, 1];
    }
    const pad = hi > lo ? (hi - lo) * Y_MARGIN : 0.5;
    const ymin = ctl.ymin ?? lo - pad;
    let ymax = ctl.ymax ?? hi + pad;
    if (!(ymax > ymin)) {
      ymax = ymin + 1;
    }

    const sx = pw / (x1 - x0);
    const sy = ph / (ymax - ymin);
    this.win = {
      x0, x1, ymin, ymax, maxError, plot, scale, w, h, sx, sy, compare,
      showTrue: Boolean(ctl.showTrue),
      showSamples: Boolean(ctl.showSamples),
      offset: (o) => [L + (o[0] - x0) * sx - w / 2, h / 2 - T - (ymax - o[1]) * sy],
      px: (x) => L + (x - x0) * sx,
      py: (y) => T + (ymax - y) * sy,
    };

    // Layers: PCHIP, cubic spline, and the true function as a degree-1 NURBS
    this.layers.true?.dispose();
    this.layers.true = undefined;
    if (this.win.showTrue && xd.length > 1) {
      const trueCurve: NurbsCurve = {
        degree: 1,
        dims: 2,
        points: Array.from(xd).flatMap((x, i) => [x, fd[i]]),
        weights: Array(xd.length).fill(1),
        knots: [xd[0], ...xd, xd[xd.length - 1]],
      };
      this.layers.true = new CurveLayer(this.device, trueCurve);
      // Opaque in its layer, translucent when composited: overlapping chords don't add up
      this.layers.true.draw(this.win, TRUE_WIDTH * scale, premul(INK2));
    }
    if (compare) {
      this.layers.cubic!.draw(this.win, CUBIC_WIDTH * scale, premul(ORANGE));
    }
    this.layers.pchip!.draw(this.win, PCHIP_WIDTH * scale, premul(BLUE));

    this.overlay.setSize(w, h);
    this.chrome = this.makeChrome();
    this.windowKey = key;
  }

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 19 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 13 * s),
    };
  }

  /** Grid, zero line, tick labels, title, error and legend. */
  private makeChrome(): Quads {
    const q = new Quads(this.font);
    const win = this.win;
    const s = win.scale;
    const [L, T, R, B] = win.plot;
    const F = this.fonts(s);
    const [xt, xdec] = niceTicks(win.x0, win.x1, (R - L) / (110 * s));
    const [yt, ydec] = niceTicks(win.ymin, win.ymax, (B - T) / (64 * s));
    for (const x of xt) {
      const X = Math.round(win.px(x));
      if (X >= L && X <= R) {
        q.rect(X, T, 1, B - T + 1, premul(GRID));
        q.text(fmtTick(x, xdec), X + 0.5, B + Math.round(8 * s), F.tick, premul(INK2), "mt");
      }
    }
    for (const y of yt) {
      const Y = Math.round(win.py(y));
      if (Y >= T && Y <= B) {
        q.rect(L, Y, R - L + 1, 1, premul(GRID));
        q.text(fmtTick(y, ydec), L - Math.round(10 * s), Y + 0.5, F.tick, premul(INK2), "rm");
      }
    }
    if (win.ymin < 0 && win.ymax > 0) {
      q.rect(L, Math.round(win.py(0)), R - L + 1, 1, premul(ZERO));
    }

    // Title, error, legend (top row)
    const ty = Math.round(T / 2) - Math.round(2 * s);
    const title = `PCHIP · ${this.name} · n = ${this.n}`;
    const tw = q.text(title, L, ty, F.title, premul(INK), "lm");
    if (win.maxError !== null) {
      q.text(`max |PCHIP − f| = ${fmtNum(win.maxError, 3)}`, L + tw + Math.round(18 * s), ty, F.sub, premul(INK2), "lm");
    }
    const entries: [string, Rgba, number][] = [["PCHIP", premul(BLUE), 3]];
    if (win.showTrue) entries.push(["f", premul(INK2, TRUE_ALPHA), 2]);
    if (win.compare) entries.push(["cubic spline", premul(ORANGE), 2]);
    let x = R;
    for (const [text, color, lw] of entries.reverse()) {
      x -= q.text(text, x, ty, F.sub, premul(INK), "rm") + Math.round(8 * s);
      const sw = Math.round(18 * s);
      q.rect(x - sw, ty - Math.floor(lw / 2), sw, lw, color);
      x -= sw + Math.round(20 * s);
    }
    return q;
  }

  // Hover ------------------------------------------------------------------------------------

  private hover(ctl: Control): Hover | null {
    const hv = ctl.hover;
    const win = this.win;
    const [L, T, R, B] = win.plot;
    if (!hv) {
      return null;
    }
    const [hx, hy] = [Number(hv[0]), Number(hv[1])];
    if (!(hx >= L && hx < R && hy >= 0 && hy < win.h)) {
      return null;
    }
    const cx = Math.floor(hx) + 0.5; // pixel centre: a crisp 1 px crosshair
    const t = win.x0 + ((cx - L) / (R - L)) * (win.x1 - win.x0);
    const [a, b] = this.domain;
    if (!(t >= a && t <= b)) {
      return { pixel: [hx, hy], t, pchip: null, f: null, error: null };
    }
    // PCHIP(t) by `evaluateNurbs` on the curve (f64): x(u) = u, so y(t)
    const p = evaluateNurbs(this.layers.pchip!.curve, t)[1];
    const fv = this.f(t);
    const cy = T + ((win.ymax - p) / (win.ymax - win.ymin)) * (B - T);
    return { pixel: [hx, hy], t, pchip: p, f: fv, error: p - fv, curvePixel: [cx, cy] };
  }

  /** Crosshair, dot on the curve, and the read-out next to the cursor, flipped away from the
   * right and bottom edges. */
  private drawHover(q: Quads, info: Hover): void {
    const win = this.win;
    const s = win.scale;
    const [L, T, R, B] = win.plot;
    const line = premul(INK2, 0.6);
    q.rect(Math.floor(info.pixel[0]), T, 1, B - T + 1, line, { clip: true });
    if (info.curvePixel) {
      const cy = Math.floor(info.curvePixel[1]);
      if (cy >= T && cy <= B) {
        q.rect(L, cy, R - L + 1, 1, premul(INK2, 0.35), { clip: true });
      }
    }
    this.drawDots(q);
    if (info.curvePixel) {
      q.dot(info.curvePixel[0], info.curvePixel[1], 6 * s, 2 * s, premul(BLUE), premul(HOVER_RING), true);
    }

    const font = this.fonts(s).label;
    const rows: [string, string, string][] = [["t", fmtNum(info.t, 6), INK]];
    if (info.pchip !== null) {
      rows.push(["PCHIP", fmtNum(info.pchip), BLUE], ["f", fmtNum(info.f!), INK], ["err", fmtNum(info.error!, 3), info.error ? ORANGE : INK]);
    }
    const gapRow = Math.round(3 * s);
    const lh = font.ascent + font.descent + gapRow;
    const kw = Math.max(...rows.map(([k]) => this.font.measure(font, k))) + Math.round(10 * s);
    const vw = Math.max(...rows.map(([, v]) => this.font.measure(font, v)));
    const pad = Math.round(8 * s);
    const lw = Math.ceil(pad * 2 + kw + vw);
    const lhTotal = pad * 2 + lh * rows.length - gapRow;
    const [hx, hy] = info.pixel;
    const gap = Math.round(14 * s);
    let x = hx + gap + lw <= R ? hx + gap : hx - gap - lw;
    let y = hy + gap + lhTotal <= B ? hy + gap : hy - gap - lhTotal;
    x = Math.trunc(Math.min(Math.max(x, 0), win.w - lw));
    y = Math.trunc(Math.min(Math.max(y, 0), win.h - lhTotal));
    q.rect(x, y, lw, lhTotal, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    rows.forEach(([k, v, color], i) => {
      const ry = y + pad + i * lh;
      q.text(k, x + pad, ry, font, premul(INK2), "lt");
      q.text(v, x + lw - pad, ry, font, premul(color), "rt");
    });
  }

  private drawDots(q: Quads): void {
    const win = this.win;
    if (!win.showSamples) {
      return;
    }
    const r = DOT_RADIUS * win.scale;
    for (let i = 0; i < this.n; i += 1) {
      const [x, y] = [this.xs[i], this.ys[i]];
      if (x >= win.x0 && x <= win.x1 && y >= win.ymin && y <= win.ymax) {
        q.dot(win.px(x), win.py(y), r, DOT_RING * win.scale, premul(BLUE), premul(SURFACE));
      }
    }
  }

  // Frame ------------------------------------------------------------------------------------

  /** Submits the frame, its readback into `slot` and its `mapAsync` (not awaited). */
  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    this.setData(ctl);
    this.setWindow(ctl);
    const win = this.win;
    const { w, h } = win;
    const info = this.hover(ctl);

    const q = new Quads(this.font);
    q.append(this.chrome!);
    if (win.showTrue && this.layers.true) q.layer(0, TRUE_ALPHA, win.plot);
    if (win.compare) q.layer(1, 1, win.plot);
    q.layer(2, 1, win.plot);
    if (info) {
      this.drawHover(q, info);
    } else {
      this.drawDots(q);
    }

    const encoder = device.createCommandEncoder();
    const layers = [this.layers.true, win.compare ? this.layers.cubic : undefined, this.layers.pchip];
    this.overlay.encode(encoder, q, layers.map((l) => l?.canvas.currentTexture()), win.plot);
    const pitch = rowBytes(w);
    if (this.frameReads[slot]?.size !== pitch * h) { // free: its frame was read
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
          for (let y = 0; y < h; y += 1) {
            rgba.set(src.subarray(y * pitch, y * pitch + w * 4), y * w * 4);
          }
        }
        frameRead.unmap();
        return { width: w, height: h, rgba, hover: info };
      },
    };
  }

  state(ctl: Control, generation: number, hover: unknown, ms: number): unknown {
    const win = this.win;
    return {
      generation,
      width: win.w,
      height: win.h,
      function: ctl.function,
      n: this.n,
      domain: [...this.domain],
      x0: win.x0,
      x1: win.x1,
      ymin: win.ymin,
      ymax: win.ymax,
      plotRect: [...win.plot], // l, t, r, b in pixels
      maxError: win.maxError,
      compare: win.compare ? "cubic" : "none",
      hover,
      renderMs: Math.round(ms * 1000) / 1000,
    };
  }
}

/** `samples CONTROL OUT.json`: nodes, slopes and the GPU samples of each curve (for checks). */
async function cmdSamples(args: string[]): Promise<void> {
  const [control, out] = args;
  const plot = await Plot.create();
  const ctl = loadControl(control, DEFAULTS as Control);
  await renderOnce(plot, { ...ctl, compare: "cubic" });
  const result: Record<string, unknown> = { xs: [...plot.xs], ys: [...plot.ys], slopes: [...plot.slopes], domain: plot.domain };
  for (const name of ["pchip", "cubic"] as const) {
    const layer = plot.layers[name];
    if (layer) {
      result[name] = await layer.samples();
    }
  }
  Deno.writeTextFileSync(out, JSON.stringify(result));
}

if (import.meta.main) {
  await main<Control>({
    name: "plot1d.ts",
    defaults: DEFAULTS as Control,
    create: Plot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
    commands: { samples: cmdSamples },
  });
}
