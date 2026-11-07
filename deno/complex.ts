/**
 * Complex heatmap (domain colouring) of f: C -> C, phase as colour and amplitude as opacity, on
 * the GPU per pixel, in Deno. Same CLI and serve protocol as surface.ts / plot1d.ts (cf.
 * lib/serve.ts).
 *
 *   deno run -A complex.ts png CONTROL.json -o out.png
 *   deno run -A complex.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 *
 * Colouring:
 * - colour: arg f in its fixed domain [-π, π], linear, looked up (nearest) in the 256 sRGB
 *   samples of the phase colour map (`phaseColorMap`, default `Twilight`);
 * - alpha: |f| normalised in the amplitude domain, logarithmically for `scale: "log"`:
 *   clamp((ln|f| − ln min) / (ln max − ln min), 0, 1), 0 where |f| = 0, the domain being
 *   [smallest positive |f|, largest |f|] of the values, here of the `n` × `n` grid of f
 *   (`linspace` of the domain, endpoints included).
 * The heatmap is straight-alpha over the (transparent) background, so amplitude fades it into it.
 *
 * `source: "gpu"` evaluates f in WGSL at each pixel (sharp at any zoom, f32), `"grid"` samples
 * the precomputed grid (nearest, cells centred on the grid points, as heatmap texels).
 * Axes, colour wheel (angle: arg f, radius: |f| in its domain) and the hover read-out (z, |f|,
 * arg f, in f64) are drawn by lib/overlay.ts. `colouring: "domain"` is classic domain
 * colouring (hue = arg, brightness rings on log2 |f|), `"phase"` colour only.
 */

import { makeReadBuffer, requestGpuDevice, rowBytes } from "./lib/gpu.ts";
import { colormapLut, colormapName } from "./lib/colormaps.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba } from "./lib/overlay.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";
import type { NurbsSurface } from "./lib/nurbs.ts";

// GitHub Dark (terminal theme)
const INK = "#c9d1d9", INK2 = "#8b949e", GRID = "#262c36", HOVER_RING = "#e6edf3";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];
const CMAP_SIZE = 256; // samples of a colour map

// --- Functions -------------------------------------------------------------------------------

type C = [number, number];
const add = (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]];
const sub = (a: C, b: C): C => [a[0] - b[0], a[1] - b[1]];
const mul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const div = (a: C, b: C): C => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};
const cexp = (a: C): C => [Math.exp(a[0]) * Math.cos(a[1]), Math.exp(a[0]) * Math.sin(a[1])];

interface ComplexFunction {
  id: number; // `fn` of the shader
  f: (z: C) => C; // f64, same formula as the WGSL
  domain: [number, number, number, number]; // re0, re1, im0, im1
  title: string; // with the atlas's glyphs
  fancy?: string; // with ² etc., if the atlas has them
}

/** The functions, in TS (f64: hover, grid, amplitude domain) and WGSL (`evaluate`). */
export const FUNCTIONS: Record<string, ComplexFunction> = {
  // the gallery's complex/rational
  rational: {
    id: 0,
    f: (z) => {
      const z2 = mul(z, z);
      const w = sub(z, [2, 1]);
      return div(mul(sub(z2, [1, 0]), mul(w, w)), add(z2, [2, 2]));
    },
    domain: [-2.5, 2.5, -2.5, 2.5],
    title: "(z^2 − 1)(z − 2 − i)^2 / (z^2 + 2 + 2i)",
    fancy: "(𝑧² − 1)(𝑧 − 2 − 𝑖)²/(𝑧² + 2 + 2𝑖)",
  },
  // the gallery's complex/cubic_surface
  "z^3-1": {
    id: 1,
    f: (z) => sub(mul(mul(z, z), z), [1, 0]),
    domain: [-2, 2, -2, 2],
    title: "z^3 − 1",
    fancy: "𝑧³ − 1",
  },
  sin: {
    id: 2,
    f: ([x, y]) => [Math.sin(x) * Math.cosh(y), Math.cos(x) * Math.sinh(y)],
    domain: [-2 * Math.PI, 2 * Math.PI, -2, 2],
    title: "sin z",
    fancy: "sin 𝑧",
  },
  // exp(2z) − z³, a classic domain-colouring example
  "exp-cubic": {
    id: 3,
    f: (z) => sub(cexp(add(z, z)), mul(mul(z, z), z)),
    domain: [-2, 2, -2, 2],
    title: "exp(2z) − z^3",
    fancy: "exp(2𝑧) − 𝑧³",
  },
};

const EVALUATE_WGSL = /* wgsl */ `
  fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
  fn cdiv(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
  fn cexp(a: vec2f) -> vec2f { return exp(a.x) * vec2f(cos(a.y), sin(a.y)); }

  fn evaluate(fn_id: u32, z: vec2f) -> vec2f {
    switch (fn_id) {
      case 1u: { return cmul(cmul(z, z), z) - vec2f(1.0, 0.0); }
      case 2u: { return vec2f(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y)); }
      case 3u: { return cexp(2.0 * z) - cmul(cmul(z, z), z); }
      default: {
        let z2 = cmul(z, z);
        let w = z - vec2f(2.0, 1.0);
        return cdiv(cmul(z2 - vec2f(1.0, 0.0), cmul(w, w)), z2 + vec2f(2.0, 2.0));
      }
    }
  }
`;

/** `n` points from `a` to `b` (numpy's `linspace`). */
function linspace(a: number, b: number, n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + (i * (b - a)) / (n - 1)));
}

/** f on the `n` × `n` grid of its domain (rows: ℑz, columns: ℜz), rounded to complex64 (f32
 * parts, as the GPU grid holds them), interleaved re, im. */
export function sampleGrid(fn: ComplexFunction, n: number, domain = fn.domain): Float64Array {
  const re = linspace(domain[0], domain[1], n);
  const im = linspace(domain[2], domain[3], n);
  const out = new Float64Array(n * n * 2);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const [u, v] = fn.f([re[j], im[i]]);
      out[(i * n + j) * 2] = Math.fround(u);
      out[(i * n + j) * 2 + 1] = Math.fround(v);
    }
  }
  return out;
}

/** Amplitude domain: bounds of |f|, made valid for the scale (log: [smallest strictly positive,
 * max]). */
function amplitudeDomain(grid: Float64Array, log: boolean): [number, number] {
  let min = Infinity, max = -Infinity, pos = Infinity;
  for (let i = 0; i < grid.length; i += 2) {
    const a = Math.hypot(grid[i], grid[i + 1]);
    if (!Number.isFinite(a)) continue;
    min = Math.min(min, a);
    max = Math.max(max, a);
    if (a > 0) pos = Math.min(pos, a);
  }
  if (log) {
    return pos <= max ? [pos, max] : [1, 10];
  }
  return min < max ? [min, max] : [min, min + 1];
}

/**
 * Degree-1 B-spline surface of the `n` × `n` grid of `function` over `domain` (cf. surface.ts):
 * control points (ℜz, ℑz, height), uniform clamped knots, weights 1, field arg f. Height: |f|
 * normalised by a log scale (`zScale`), clamped in its domain [smallest positive |f|, max |f|],
 * times `zStretch` (the box height relative to its footprint, here the longer side of
 * `domain`). Zeros of f (log undefined) and anything below the domain lie on the floor: height 0.
 */
export function complexGridSurface(spec: { complex?: string; n?: number; domain?: number[]; zStretch?: number; zScale?: string }): NurbsSurface {
  const fn = FUNCTIONS[spec.complex ?? "z^3-1"];
  if (!fn) {
    throw new Error(`complex must be one of ${Object.keys(FUNCTIONS)}`);
  }
  const n = Math.max(2, Math.trunc(spec.n ?? 200));
  const d = (spec.domain ?? fn.domain) as [number, number, number, number];
  const log = (spec.zScale ?? "log") === "log";
  const zStretch = spec.zStretch ?? 0.5;
  const grid = sampleGrid(fn, n, d);
  const [lo, hi] = amplitudeDomain(grid, log);
  const scale = (a: number) => (log ? (a > 0 ? Math.log(a) : -Infinity) : a);
  const [s0, s1] = [scale(lo), scale(hi)];
  const box = Math.max(d[1] - d[0], d[3] - d[2]) * zStretch;
  const re = linspace(d[0], d[1], n);
  const im = linspace(d[2], d[3], n);
  const points: number[] = new Array(n * n * 3);
  const field: number[] = new Array(n * n);
  // u along ℜz (rows of control points), v along ℑz
  for (let j = 0; j < n; j += 1) {
    for (let i = 0; i < n; i += 1) {
      const [u, v] = [grid[(i * n + j) * 2], grid[(i * n + j) * 2 + 1]];
      const t = Math.min(Math.max((scale(Math.hypot(u, v)) - s0) / (s1 - s0), 0), 1); // NaN-safe: -Inf → 0
      const k = j * n + i;
      points[k * 3] = re[j];
      points[k * 3 + 1] = im[i];
      points[k * 3 + 2] = (Number.isFinite(t) ? t : 0) * box;
      field[k] = Math.atan2(v, u);
    }
  }
  const knots = [0, ...linspace(0, 1, n), 1];
  return { degrees: [1, 1], shape: [n, n], points, weights: new Array(n * n).fill(1), knots: [knots, knots], fields: [field] };
}

// --- Formatting ------------------------------------------------------------------------------

/** Ticks at 1, 2, 2.5 or 5 × 10^k, about `target` of them, and their decimals (as plot1d.ts). */
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

function fmtNum(v: number, digits = 5): string {
  if (v === 0) return "0";
  if (!Number.isFinite(v)) return String(v);
  if (Math.abs(v) >= 1e-4 && Math.abs(v) < 1e6) {
    const s = v.toPrecision(digits);
    return minus(s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  }
  const [mant, exp] = v.toExponential(digits - 2).split("e");
  const e = Number(exp);
  return minus(`${mant}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`);
}

// --- Shader ----------------------------------------------------------------------------------

const DOMAIN_WGSL = /* wgsl */ `
  struct Params {
    plot: vec4f, // left, top, right, bottom (px)
    window: vec4f, // x0, x1, y0, y1 (data)
    wheel: vec4f, // centre x, y, radius (px), drawn
    amp: vec4f, // amplitude domain (scaled: ln for log), log, alpha by amplitude
    grid: vec4f, // grid domain (centres of the first and last cells): re0, re1, im0, im1
    mode: vec4u, // function, source (0 gpu, 1 grid), samples per side, colouring (0 phase-amplitude, 1 phase, 2 domain)
    gridSize: vec4u, // n
  }
  @group(0) @binding(0) var<uniform> P: Params;
  @group(0) @binding(1) var<storage, read> lut: array<vec4f>; // 256 sRGB colours
  @group(0) @binding(2) var grid: texture_2d<f32>; // rg32float: f on the grid

  ${EVALUATE_WGSL}

  const PI = 3.14159265358979;

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    var corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(corners[v], 0.0, 1.0);
  }

  fn phaseColor(phase: f32) -> vec3f { // HeatmapMaterial: texture2D(colorMap, normalized), nearest
    let t = clamp((phase + PI) / (2.0 * PI), 0.0, 1.0);
    return lut[min(u32(t * 256.0), 255u)].rgb;
  }

  /** Domain colouring: hue = arg f, brightness rings on log2 |f| */
  fn domainColor(f: vec2f) -> vec3f {
    let hue = fract(atan2(f.y, f.x) / (2.0 * PI));
    let s = sin(2.0 * PI * log2(length(f) + 1e-12));
    let lum = 0.55 + 0.25 * s * s;
    let k = (vec3f(5.0, 3.0, 1.0) + hue * 6.0) % vec3f(6.0);
    return clamp(lum - 0.45 * clamp(min(k, 4.0 - k), vec3f(0.0), vec3f(1.0)), vec3f(0.0), vec3f(1.0));
  }

  /** Premultiplied colour of the value f. */
  fn shade(f: vec2f) -> vec4f {
    let amplitude = length(f);
    if (!(amplitude == amplitude)) { return vec4f(0.0); } // NaN
    if (P.mode.w == 2u) { return vec4f(domainColor(f), 1.0); }
    let rgb = phaseColor(atan2(f.y, f.x));
    var a = 1.0;
    if (P.mode.w == 0u) { // HeatmapMaterial alpha
      if (P.amp.z > 0.5 && amplitude <= 0.0) {
        a = 0.0;
      } else {
        let v = select(amplitude, log(amplitude), P.amp.z > 0.5);
        a = clamp((v - P.amp.x) / (P.amp.y - P.amp.x), 0.0, 1.0);
      }
    }
    return vec4f(rgb * a, a);
  }

  fn valueAt(z: vec2f) -> vec2f {
    if (P.mode.y == 0u) {
      return evaluate(P.mode.x, z);
    }
    let n = i32(P.gridSize.x);
    let step = vec2f(P.grid.y - P.grid.x, P.grid.w - P.grid.z) / f32(n - 1);
    let ij = vec2i(floor((z - P.grid.xz) / step + 0.5)); // nearest cell centre
    if (any(ij < vec2i(0)) || any(ij >= vec2i(n))) {
      return vec2f(bitcast<f32>(0x7fc00000u)); // outside: NaN, transparent
    }
    return textureLoad(grid, ij, 0).rg;
  }

  @fragment
  fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let p = pos.xy; // pixel centre
    let m = max(P.mode.z, 1u);
    let d = 1.0 / f32(m);
    if (p.x >= P.plot.x && p.x < P.plot.z && p.y >= P.plot.y && p.y < P.plot.w) {
      let scale = vec2f((P.window.y - P.window.x) / (P.plot.z - P.plot.x), (P.window.w - P.window.z) / (P.plot.w - P.plot.y));
      var c = vec4f(0.0);
      for (var i = 0u; i < m; i += 1u) {
        for (var j = 0u; j < m; j += 1u) {
          let q = floor(p) + (vec2f(f32(i), f32(j)) + 0.5) * d;
          let z = vec2f(P.window.x + (q.x - P.plot.x) * scale.x, P.window.w - (q.y - P.plot.y) * scale.y);
          c += shade(valueAt(z));
        }
      }
      return c / f32(m * m);
    }
    let r = P.wheel.z;
    if (P.wheel.w > 0.5 && length(p - P.wheel.xy) < r + 1.0) { // legend: angle arg f, radius |f|
      var c = vec4f(0.0);
      for (var i = 0u; i < 4u; i += 1u) {
        for (var j = 0u; j < 4u; j += 1u) {
          let q = floor(p) + (vec2f(f32(i), f32(j)) + 0.5) * 0.25 - P.wheel.xy;
          let rho = length(q) / r;
          if (rho <= 1.0) {
            let lnA = mix(P.amp.x, P.amp.y, rho);
            let amplitude = select(lnA, exp(lnA), P.amp.z > 0.5);
            let phase = atan2(-q.y, q.x);
            c += shade(amplitude * vec2f(cos(phase), sin(phase)));
          }
        }
      }
      return c / 16.0;
    }
    return vec4f(0.0);
  }
`;

// --- Control ---------------------------------------------------------------------------------

export const DEFAULTS = {
  width: 1200,
  height: 750,
  function: "rational",
  x0: null as number | null, // view window (data); null: the function's domain
  x1: null as number | null,
  y0: null as number | null,
  y1: null as number | null,
  source: "gpu" as "gpu" | "grid", // f per pixel in WGSL, or the n × n grid (nearest)
  n: 400, // side of the sampled grid: amplitude domain, and `source: "grid"`
  colouring: "phase-amplitude" as "phase-amplitude" | "phase" | "domain",
  phaseColorMap: "Twilight", // cyclic
  invertPhaseColorMap: false,
  scale: "log" as "log" | "linear", // of the amplitude
  amplitudeDomain: null as [number, number] | null, // default: of the grid
  keepRatio: true, // equal aspect
  antialias: 2, // samples per side per pixel (source gpu)
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

interface View {
  w: number;
  h: number;
  scale: number;
  plot: [number, number, number, number];
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  wheel: [number, number, number];
  fn: ComplexFunction;
  name: string;
  amp: [number, number]; // amplitude domain
  log: boolean;
}

/** The colour map's 256 sRGB samples, 0..1 (RGBA). */
function colorMapLut(name: string, invert: boolean): Float32Array<ArrayBuffer> {
  const key = colormapName(name);
  if (!key) {
    throw new Error(`Unknown colormap ${name}`);
  }
  const bytes = colormapLut(key, invert);
  return Float32Array.from(bytes, (c) => c / 255);
}

interface Hover {
  pixel: [number, number];
  z: [number, number];
  f: [number, number] | null;
  amplitude: number | null;
  phase: number | null;
}

export class ComplexPlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly pipeline: GPURenderPipeline;
  private readonly params: GPUBuffer;
  private readonly lut: GPUBuffer;
  private layer?: GPUTexture;
  private gridTexture: GPUTexture;
  private group?: GPUBindGroup;
  private readonly frameReads: GPUBuffer[] = [];
  private lutKey?: string;
  private gridKey?: string;
  private viewKey?: string;
  private chrome?: Quads;
  grid: Float64Array = new Float64Array(0);
  gridN = 0;
  private gridAmp: [number, number] = [1, 10];
  view!: View;

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.overlay = new OverlayRenderer(device, font);
    const module = device.createShaderModule({ code: DOMAIN_WGSL });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module },
      fragment: { module, targets: [{ format: FRAME_FORMAT }] },
    });
    this.params = device.createBuffer({ size: 112, usage: 0x40 | 0x08 });
    this.lut = device.createBuffer({ size: CMAP_SIZE * 16, usage: 0x80 | 0x08 });
    this.gridTexture = this.makeGridTexture(1, new Float32Array(2));
  }

  static async create(): Promise<ComplexPlot> {
    const device = await requestGpuDevice(); // the shared device
    if (!device) {
      throw new Error("No WebGPU device");
    }
    return new ComplexPlot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  private makeGridTexture(n: number, data: Float32Array): GPUTexture {
    const t = this.device.createTexture({ size: [n, n], format: "rg32float", usage: 0x04 | 0x02 });
    this.device.queue.writeTexture({ texture: t }, data, { bytesPerRow: n * 8 }, [n, n]);
    return t;
  }

  /** Text if the atlas has all its glyphs, else the fallback. */
  private glyphs(font: FontSize, text: string | undefined, fallback: string): string {
    return text && [...text].every((ch) => ch in font.glyphs) ? text : fallback;
  }

  private setData(ctl: Control): void {
    const fn = FUNCTIONS[ctl.function];
    if (!fn) {
      throw new Error(`function must be one of ${Object.keys(FUNCTIONS)}`);
    }
    const n = Math.min(Math.max(2, Math.trunc(Number(ctl.n))), 4096);
    const key = JSON.stringify([ctl.function, n, ctl.scale]);
    if (key !== this.gridKey) {
      this.grid = sampleGrid(fn, n);
      this.gridN = n;
      this.gridAmp = amplitudeDomain(this.grid, ctl.scale === "log");
      this.gridTexture.destroy();
      this.gridTexture = this.makeGridTexture(n, Float32Array.from(this.grid));
      this.group = undefined;
      this.gridKey = key;
    }
    const lutKey = JSON.stringify([ctl.phaseColorMap, Boolean(ctl.invertPhaseColorMap)]);
    if (lutKey !== this.lutKey) {
      this.device.queue.writeBuffer(this.lut, 0, colorMapLut(String(ctl.phaseColorMap), Boolean(ctl.invertPhaseColorMap)));
      this.lutKey = lutKey;
    }
  }

  private setView(ctl: Control): void {
    const w = Math.trunc(ctl.width);
    const h = Math.trunc(ctl.height);
    const fn = FUNCTIONS[ctl.function];
    const [a, b, c, d] = fn.domain;
    const x0 = ctl.x0 ?? a, x1 = ctl.x1 ?? b, y0 = ctl.y0 ?? c, y1 = ctl.y1 ?? d;
    if (!(x1 > x0) || !(y1 > y0)) {
      throw new Error("x1 must be above x0 and y1 above y0");
    }
    const key = JSON.stringify([this.gridKey, w, h, x0, x1, y0, y1, ctl.keepRatio, ctl.amplitudeDomain, ctl.colouring]);
    if (key === this.viewKey) {
      return;
    }
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    const panel = Math.round(190 * s);
    const L = Math.round(70 * s), T = Math.round(54 * s);
    let R = w - panel - Math.round(28 * s), B = h - Math.round(58 * s);
    if (ctl.keepRatio) { // equal aspect, in the top-left of the space
      const aspect = (x1 - x0) / (y1 - y0);
      if ((R - L) / (B - T) > aspect) {
        R = L + Math.round((B - T) * aspect);
      } else {
        B = T + Math.round((R - L) / aspect);
      }
    }
    const radius = Math.round(Math.min(64 * s, (panel - 60 * s) / 2));
    const wheel: [number, number, number] = [R + Math.round(28 * s) + panel / 2, T + Math.round(44 * s) + radius, radius];
    const log = ctl.scale === "log";
    const amp = (ctl.amplitudeDomain as [number, number] | null) ?? this.gridAmp;
    this.view = { w, h, scale: s, plot: [L, T, R, B], x0, x1, y0, y1, wheel, fn, name: ctl.function, amp, log };
    this.overlay.setSize(w, h);
    if (this.layer?.width !== w || this.layer.height !== h) {
      this.layer?.destroy();
      this.layer = this.device.createTexture({ size: [w, h], format: FRAME_FORMAT, usage: 0x10 | 0x04 });
    }
    this.chrome = this.makeChrome(String(ctl.colouring));
    this.viewKey = key;
  }

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 18 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 13 * s),
    };
  }

  /** Axes, ticks, labels, title, and the wheel's labels. */
  private makeChrome(colouring: string): Quads {
    const q = new Quads(this.font);
    const v = this.view;
    const s = v.scale;
    const [L, T, R, B] = v.plot;
    const F = this.fonts(s);
    const tick = Math.round(5 * s);
    const axis = premul(INK2, 0.7);
    const px = (x: number) => L + ((x - v.x0) / (v.x1 - v.x0)) * (R - L);
    const py = (y: number) => T + ((v.y1 - y) / (v.y1 - v.y0)) * (B - T);
    const [xt, xdec] = niceTicks(v.x0, v.x1, (R - L) / (90 * s));
    const [yt, ydec] = niceTicks(v.y0, v.y1, (B - T) / (60 * s));
    q.rect(L - 1, T - 1, R - L + 2, B - T + 2, [0, 0, 0, 0], { border: 1, borderColor: premul(GRID) });
    q.rect(L - 1, B, R - L + 2, 1, axis);
    q.rect(L - 1, T - 1, 1, B - T + 2, axis);
    for (const x of xt) {
      const X = Math.round(px(x));
      if (X >= L && X <= R) {
        q.rect(X, B + 1, 1, tick, axis);
        q.text(fmtTick(x, xdec), X + 0.5, B + tick + Math.round(4 * s), F.tick, premul(INK2), "mt");
      }
    }
    for (const y of yt) {
      const Y = Math.round(py(y));
      if (Y >= T && Y <= B) {
        q.rect(L - 1 - tick, Y, tick, 1, axis);
        q.text(fmtTick(y, ydec), L - tick - Math.round(6 * s), Y + 0.5, F.tick, premul(INK2), "rm");
      }
    }
    const re = this.glyphs(F.sub, "ℜ𝑧", "Re z"), im = this.glyphs(F.sub, "ℑ𝑧", "Im z");
    q.text(re, (L + R) / 2, v.h - Math.round(10 * s), F.sub, premul(INK), "mb");
    q.text(im, L - tick - Math.round(6 * s), T - Math.round(10 * s), F.sub, premul(INK), "rb");

    // Title: `${title} (phase & amplitude)`
    const labels: Record<string, string> = { "phase-amplitude": "phase & amplitude", phase: "phase", domain: "domain colouring" };
    const title = `${this.glyphs(F.title, v.fn.fancy, v.fn.title)} (${labels[colouring] ?? colouring})`;
    q.text(title, L, Math.round(T / 2) - Math.round(4 * s), F.title, premul(INK), "lm");

    // Wheel: arg f around, |f| along the radius
    const [cx, cy, r] = v.wheel;
    const pi = this.glyphs(F.tick, "π", "pi");
    const g = Math.round(6 * s);
    q.text(this.glyphs(F.sub, "arg 𝑓", "arg f"), cx, T - Math.round(2 * s), F.sub, premul(INK), "mt");
    q.text("0", cx + r + g, cy + 0.5, F.tick, premul(INK2), "lm");
    q.text(`${pi}/2`, cx, cy - r - g, F.tick, premul(INK2), "mb");
    q.text(`±${pi}`, cx - r - g, cy + 0.5, F.tick, premul(INK2), "rm");
    q.text(`−${pi}/2`, cx, cy + r + g, F.tick, premul(INK2), "mt");
    let y = cy + r + Math.round(34 * s);
    const lh = F.tick.ascent + F.tick.descent + Math.round(4 * s);
    const rows: [string, string][] = colouring === "phase"
      ? [["opaque: arg 𝑓 only", ""]]
      : [
        [`|𝑓|, ${v.log ? "log" : "linear"}, centre → rim`, ""],
        ["centre", fmtNum(v.amp[0], 3)],
        ["rim", fmtNum(v.amp[1], 3)],
      ];
    const x0 = cx - Math.round(78 * s), x1 = cx + Math.round(78 * s);
    for (const [k, val] of rows) {
      if (val) {
        q.text(k, x0, y, F.tick, premul(INK2), "lt");
        q.text(val, x1, y, F.tick, premul(INK), "rt");
      } else {
        q.text(k, cx, y, F.tick, premul(INK2), "mt");
      }
      y += lh;
    }
    return q;
  }

  // Hover ------------------------------------------------------------------------------------

  /** z under the pixel centre, and f there: f64 (gpu), or the nearest grid value (grid). */
  private hover(ctl: Control): Hover | null {
    const hv = ctl.hover;
    const v = this.view;
    const [L, T, R, B] = v.plot;
    if (!hv) return null;
    const [hx, hy] = [Number(hv[0]), Number(hv[1])];
    if (!(hx >= L && hx < R && hy >= T && hy < B)) return null;
    const cx = Math.floor(hx) + 0.5, cy = Math.floor(hy) + 0.5;
    const z: C = [v.x0 + ((cx - L) / (R - L)) * (v.x1 - v.x0), v.y1 - ((cy - T) / (B - T)) * (v.y1 - v.y0)];
    let f: C | null = null;
    if (ctl.source === "grid") {
      const n = this.gridN;
      const [a, b, c, d] = v.fn.domain;
      const j = Math.floor(((z[0] - a) / (b - a)) * (n - 1) + 0.5);
      const i = Math.floor(((z[1] - c) / (d - c)) * (n - 1) + 0.5);
      if (i >= 0 && i < n && j >= 0 && j < n) {
        f = [this.grid[(i * n + j) * 2], this.grid[(i * n + j) * 2 + 1]];
      }
    } else {
      f = v.fn.f(z);
    }
    return {
      pixel: [hx, hy],
      z,
      f,
      amplitude: f ? Math.hypot(f[0], f[1]) : null,
      phase: f ? Math.atan2(f[1], f[0]) : null,
    };
  }

  private drawHover(q: Quads, info: Hover): void {
    const v = this.view;
    const s = v.scale;
    const [L, T, R, B] = v.plot;
    const line = premul(HOVER_RING, 0.45);
    const X = Math.floor(info.pixel[0]), Y = Math.floor(info.pixel[1]);
    q.rect(X, T, 1, B - T, line);
    q.rect(L, Y, R - L, 1, line);
    q.dot(X + 0.5, Y + 0.5, 4 * s, 1.5 * s, [0, 0, 0, 0], premul(HOVER_RING));
    if (info.phase !== null) { // the same direction on the wheel
      const [cx, cy, r] = v.wheel;
      const lnA = v.log ? Math.log(info.amplitude!) : info.amplitude!;
      const [l0, l1] = v.log ? v.amp.map(Math.log) : v.amp;
      const rho = Math.min(Math.max((lnA - l0) / (l1 - l0), 0), 1);
      q.dot(cx + r * rho * Math.cos(info.phase), cy - r * rho * Math.sin(info.phase), 3.5 * s, 1.5 * s, [0, 0, 0, 0], premul(HOVER_RING));
    }

    const font = this.fonts(s).label;
    const pi = this.glyphs(font, "π", "pi");
    const im = (x: number) => `${x < 0 ? "−" : "+"} ${fmtNum(Math.abs(x), 5)}𝑖`;
    const rows: [string, string][] = [["𝑧", `${fmtNum(info.z[0], 5)} ${im(info.z[1])}`]];
    if (info.f) {
      rows.push(
        ["𝑓", `${fmtNum(info.f[0], 4)} ${im(info.f[1]).replace(/(\d)(\.\d{3})\d+/, "$1$2")}`],
        ["|𝑓|", fmtNum(info.amplitude!, 5)],
        ["arg 𝑓", `${fmtNum(info.phase!, 4)} (${fmtNum(info.phase! / Math.PI, 3)}${pi})`],
      );
    } else {
      rows.push(["𝑓", "outside the grid"]);
    }
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
    rows.forEach(([k, val], i) => {
      const ry = y + pad + i * lh;
      q.text(k, x + pad, ry, font, premul(INK2), "lt");
      q.text(val, x + lw - pad, ry, font, premul(INK), "rt");
    });
  }

  // Frame ------------------------------------------------------------------------------------

  private writeParams(ctl: Control): void {
    const v = this.view;
    const buf = new ArrayBuffer(112);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    const scaled = v.log ? v.amp.map(Math.log) : v.amp;
    f.set(v.plot, 0);
    f.set([v.x0, v.x1, v.y0, v.y1], 4);
    f.set([...v.wheel, 1], 8);
    f.set([scaled[0], scaled[1] > scaled[0] ? scaled[1] : scaled[0] + 1, v.log ? 1 : 0, 0], 12);
    f.set(v.fn.domain, 16);
    const colouring = { "phase-amplitude": 0, phase: 1, domain: 2 }[String(ctl.colouring)];
    if (colouring === undefined) {
      throw new Error("colouring must be phase-amplitude, phase or domain");
    }
    const aa = ctl.source === "grid" ? 1 : Math.min(Math.max(1, Math.trunc(Number(ctl.antialias) || 1)), 4);
    u.set([v.fn.id, ctl.source === "grid" ? 1 : 0, aa, colouring], 20);
    u.set([this.gridN, 0, 0, 0], 24);
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Submits the frame, its readback into `slot` and its `mapAsync` (not awaited). */
  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    if (ctl.source !== "gpu" && ctl.source !== "grid") {
      throw new Error("source must be gpu or grid");
    }
    this.setData(ctl);
    this.setView(ctl);
    this.writeParams(ctl);
    const { w, h } = this.view;
    const info = this.hover(ctl);

    if (!this.group) {
      this.group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: this.lut } },
          { binding: 2, resource: this.gridTexture.createView() },
        ],
      });
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.layer!.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.draw(3);
    pass.end();

    const q = new Quads(this.font);
    q.layer(0, 1, [0, 0, w, h]);
    q.append(this.chrome!);
    if (info) this.drawHover(q, info);
    this.overlay.encode(encoder, q, [this.layer], [0, 0, w, h]);

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
    const v = this.view;
    return {
      generation,
      width: v.w,
      height: v.h,
      function: v.name,
      domain: [...v.fn.domain],
      x0: v.x0,
      x1: v.x1,
      y0: v.y0,
      y1: v.y1,
      plotRect: [...v.plot], // l, t, r, b in pixels
      wheel: [...v.wheel], // centre x, y, radius in pixels
      source: ctl.source,
      colouring: ctl.colouring,
      phaseColorMap: ctl.phaseColorMap,
      scale: v.log ? "log" : "linear",
      amplitudeDomain: [...v.amp],
      grid: this.gridN,
      hover,
      renderMs: Math.round(ms * 1000) / 1000,
    };
  }
}

if (import.meta.main) {
  await main<Control>({
    name: "complex.ts",
    defaults: DEFAULTS as Control,
    create: ComplexPlot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
  });
}
