/**
 * 2D and 3D NURBS / B-spline curves drawn by the WebGPU curve renderers: `NurbsGpuRenderer`
 * (lib/gpu_line.ts) in the plane, with equal aspect, and `NurbsCurve3DGpuRenderer`
 * (lib/gpu_curve3d.ts) in space, coloured along `u` through a colour LUT (`getLinearColorLut`),
 * with an orbit camera fitted to the curve and centred on both axes (lib/framing.ts) and the
 * segment counts of the projected control points (`projectControlPoints`). Polylines are
 * degree-1 B-splines.
 *
 * Colour by value: the 3D renderer colours by `u` through a 256-entry LUT; a value along the
 * curve (curvature, phase...) is the LUT composed with the value at each entry's `u` (the curve
 * can be parameterised so that the value varies evenly in `u`, e.g. by phase). `scale: "root"`
 * colours a (non-cyclic) value by t^(1/rootN) of its normalised value, with a slider for n.
 *
 *   deno run -A curve.ts png CONTROL.json -o out.png
 *   deno run -A curve.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 */

import { axisTicks, colorBarTicks, createScale, formatNumber, snapRootN } from "./lib/scales.ts";
import { cameraBasis, fitAngles, fitView, screenRect, type Vec3 } from "./lib/framing.ts";
import { lookAtView, mul4, perspectiveProjection } from "./lib/mat4.ts";
import { colormapName, colormapRgb, getLinearColorLut } from "./lib/colormaps.ts";
import {
  evaluateNurbs,
  getGpuParams,
  getKnotSpans,
  getNurbsError,
  getSegmentCounts,
  type KnotSpan,
  type NurbsCurve,
  projectControlPoints,
} from "./lib/nurbs.ts";
import { makeReadBuffer, requestGpuDevice, rowBytes, TextureTarget } from "./lib/gpu.ts";
import { NurbsGpuRenderer } from "./lib/gpu_line.ts";
import { DRAW_WGSL as CURVE3D_WGSL, NurbsCurve3DGpuRenderer } from "./lib/gpu_curve3d.ts";
import { DEPTH_FORMAT, getTargets, type RenderTargets, SAMPLE_COUNT } from "./lib/gpu_surface.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba, rootSlider, type RootSlider, rootSliderHeight } from "./lib/overlay.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";
import { CURVES, curvePoints } from "./lib/generators.ts";

const BLUE = "#3987e5", ORANGE = "#d95926", MAGENTA = "#d55181";
const INK = "#c9d1d9", INK2 = "#8b949e", GRID = "#262c36", ZERO = "#484f58", SURFACE = "#101216", HOVER_RING = "#e6edf3";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];
const FOV = 40;
const FIT_MARGIN = 1.06;
const MARGIN = 0.06; // 2D: around the curve and its control points
const LUT_SIZE = 256; // of the renderers' colour maps
const POLYGON_OPACITY = 0.6;

export interface CurveSpec {
  degree: number;
  points: number[] | number[][];
  weights?: number[];
  knots: number[];
  dims?: number;
}

export const DEFAULTS = {
  width: 1200,
  height: 750,
  title: "trefoil_3d",
  subtitle: "",
  curve: null as CurveSpec | null,
  polyline: null as number[][] | null, // degree-1 B-spline through these points
  params: null as number[] | null, // of the polyline's points (default: their indices)
  controlPolygon: false,
  color: BLUE, // 2D
  lineWidth: 2, // CSS px (2D and 3D: on screen, whatever the depth)
  lineCap: "square" as "square" | "round", // 3D: round joins and caps (each chord a capsule), for thick lines
  // 3D colour: by `u`, or by values at the polyline's points (cyclic: wrapped into the domain, e.g. phase)
  colorBy: "u" as "u" | { values: number[]; label?: string; domain?: [number, number]; ticks?: "pi"; cyclic?: boolean } | null,
  aspect: null as [number, number, number] | null, // 3D: world scale of each axis (shown values unscaled)
  colormap: "Viridis",
  invertColorMap: false, // 3D: the colour map (and bar) reversed
  scale: "linear" as "linear" | "root", // 3D colour by (non-cyclic) values: root colours by t^(1/rootN), t normalised
  rootN: 3, // 0.1..10, step 0.05
  colorLabel: "𝑢",
  paramLabel: null as string | null, // hover read-out name of the parameter (default 𝑢)
  axisLabels: null as string[] | null, // hover read-out names of x, y (z)
  azimuth: 35,
  elevation: 25,
  zoom: 1,
  x0: null as number | null, // 2D window (equal aspect kept: the larger of the two spans wins)
  x1: null as number | null,
  ymin: null as number | null,
  ymax: null as number | null,
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

/** An ODE trajectory, integrated here (classic fourth-order Runge-Kutta, the first
 * `transient` steps dropped) so controls stay small: `{system: "lorenz", n, dt, sigma, rho, beta, transient,
 * colorBy: "time" | "speed"}`. Expanded into polyline/params/colorBy before rendering. */
export type OdeSpec = {
  system: "lorenz"; n?: number; dt?: number; sigma?: number; rho?: number; beta?: number;
  transient?: number; colorBy?: "time" | "speed";
};
const odeCache = new Map<string, Pick<Control, "polyline" | "params" | "colorBy">>();
export function expandOde(ctl: Control): Control {
  const ode = ctl.ode as OdeSpec | undefined;
  if (!ode) return ctl;
  if (ode.system !== "lorenz") throw new Error(`Unknown ode system ${ode.system}`);
  const key = JSON.stringify(ode);
  let out = odeCache.get(key);
  if (!out) {
    const { n = 40_000, dt = 0.004, sigma = 10, rho = 28, beta = 8 / 3, transient = 1000, colorBy = "time" } = ode;
    if (!(n >= 2 && n <= 1_000_000) || !(dt > 0)) throw new Error("ode: need 2 <= n <= 1e6 and dt > 0");
    const f = (x: number, y: number, z: number) => [sigma * (y - x), x * (rho - z) - y, x * y - beta * z];
    let [x, y, z] = [1, 1, 1];
    const pts: number[][] = [], t: number[] = [], speed: number[] = [];
    for (let i = 0; i < n + transient; i++) {
      const a = f(x, y, z);
      const b = f(x + dt / 2 * a[0], y + dt / 2 * a[1], z + dt / 2 * a[2]);
      const c = f(x + dt / 2 * b[0], y + dt / 2 * b[1], z + dt / 2 * b[2]);
      const d = f(x + dt * c[0], y + dt * c[1], z + dt * c[2]);
      x += dt / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]);
      y += dt / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1]);
      z += dt / 6 * (a[2] + 2 * b[2] + 2 * c[2] + d[2]);
      if (i >= transient) {
        pts.push([x, y, z]);
        t.push((i - transient) * dt);
        const v = f(x, y, z);
        speed.push(Math.hypot(v[0], v[1], v[2]));
      }
    }
    out = {
      polyline: pts,
      params: t,
      colorBy: colorBy === "speed" ? { values: speed, label: "speed |𝐯|" } : { values: t, label: "time 𝑡" },
    };
    if (odeCache.size > 8) odeCache.clear();
    odeCache.set(key, out);
  }
  return { ...ctl, ...out };
}

/** A parametric curve of lib/generators.ts, evaluated here (f64) so controls stay small:
 * `{name: "trefoil" | "chirp", n, t: [t0, t1], ...its parameters}`. Expanded into polyline,
 * params (t) and the colour values (curvature, phase), `colorBy`'s other fields (label, domain,
 * ticks, cyclic) kept from the control. */
export type GenSpec = { name: string; n?: number; t?: [number, number] } & Record<string, unknown>;
const genCache = new Map<string, Pick<Control, "polyline" | "params"> & { values: number[] }>();
export function expandGen(ctl: Control): Control {
  const gen = ctl.gen as GenSpec | undefined;
  if (!gen) return ctl;
  const c = CURVES[gen.name];
  if (!c) throw new Error(`gen must be one of ${Object.keys(CURVES)}`);
  const key = JSON.stringify(gen);
  let out = genCache.get(key);
  if (!out) {
    const n = gen.n ?? 1000;
    const t = (gen.t ?? c.t).map(Number) as [number, number];
    if (!(n >= 2 && n <= 1_000_000) || !(t[1] > t[0])) throw new Error("gen: need 2 <= n <= 1e6 and t increasing");
    const args = c.args.map((a, i) => (gen[a] === undefined ? c.defaults[i] : Number(gen[a])));
    const pts = curvePoints(gen.name, n, t, args);
    out = {
      polyline: Array.from({ length: n }, (_, i) => [pts[i * 4], pts[i * 4 + 1], pts[i * 4 + 2]]),
      params: Array.from({ length: n }, (_, i) => t[0] + ((t[1] - t[0]) * i) / (n - 1)),
      values: Array.from({ length: n }, (_, i) => pts[i * 4 + 3]),
    };
    if (genCache.size > 8) genCache.clear();
    genCache.set(key, out);
  }
  const cb = ctl.colorBy;
  return { ...ctl, polyline: out.polyline, params: out.params, colorBy: cb && cb !== "u" ? { ...cb, values: out.values } : cb };
}

/** The default: a trefoil knot, (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), as a periodic cubic
 * B-spline on 60 points (the gallery's curves/trefoil_3d). */
export function trefoilSpec(): CurveSpec {
  const pts: number[][] = [];
  for (let i = 0; i < 60; i += 1) {
    const t = (2 * Math.PI * i) / 60;
    pts.push([Math.sin(t) + 2 * Math.sin(2 * t), Math.cos(t) - 2 * Math.cos(2 * t), -Math.sin(3 * t)]);
  }
  return periodic(pts, 3);
}

/** Closed B-spline of `pts`: the first `degree` points wrapped, uniform knots. */
export function periodic(pts: number[][], degree: number): CurveSpec {
  const all = [...pts, ...pts.slice(0, degree)];
  return { degree, points: all, knots: Array.from({ length: all.length + degree + 1 }, (_, i) => i) };
}

function toNurbs(ctl: Control): NurbsCurve {
  const a = ctl.aspect;
  const scaled = (pts: number[], dims: number) => (a && dims === 3 ? pts.map((v, i) => v * a[i % 3]) : pts);
  if (ctl.polyline) {
    const n = ctl.polyline.length;
    const dims = ctl.polyline[0].length;
    const u = ctl.params ?? Array.from({ length: n }, (_, i) => i);
    return { degree: 1, dims, points: scaled(ctl.polyline.flat(), dims), weights: Array(n).fill(1), knots: [u[0], ...u, u[n - 1]] };
  }
  const c = ctl.curve ?? trefoilSpec();
  const nested = Array.isArray(c.points[0]);
  const dims = c.dims ?? (nested ? (c.points[0] as number[]).length : 2);
  const points = nested ? (c.points as number[][]).flat() : (c.points as number[]);
  return { degree: c.degree, dims, points: scaled(points, dims), weights: c.weights ?? Array(points.length / dims).fill(1), knots: c.knots };
}

/** Dense CPU samples (`evaluateNurbs`): u and the points (dims each). */
function sampleCurve(curve: NurbsCurve, spans: KnotSpan[], total = 4000): { u: number[]; p: number[][] } {
  const len = spans.reduce((a, s) => a + (s.end - s.start), 0);
  const u: number[] = [];
  const p: number[][] = [];
  for (const s of spans) {
    const steps = Math.max(curve.degree === 1 ? 1 : 4, Math.ceil((total * (s.end - s.start)) / len));
    for (let k = 0; k < steps; k += 1) {
      const t = s.start + ((s.end - s.start) * k) / steps;
      u.push(t);
      p.push(evaluateNurbs(curve, t, s.index));
    }
  }
  const last = spans[spans.length - 1];
  u.push(last.end);
  p.push(evaluateNurbs(curve, last.end, last.index));
  return { u, p };
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/* Appended to the 3D curve draw shader (as surface.ts does for picking): the colour of a cyclic value, wrapped into the colour map's domain in the fragment
 * shader. The curve's parameter is the (unwrapped) value itself, `uRange` its domain. */
const WRAP_WGSL = /* wgsl */ `
  @fragment
  fn fragmentWrapped(in: Varyings) -> Fragment {
    let mask = capsuleMask(in);
    let r = (in.u - view.uRange.x) / (view.uRange.y - view.uRange.x);
    return Fragment(vec4f(toSrgb(getColor(r - floor(r))), 1.0), mask);
  }
`;

/**
 * A 3D polyline coloured by a cyclic value (e.g. phase), exactly: split into runs along which
 * the value increases (descending runs reversed), each a degree-1 B-spline whose knots are the
 * values, evaluated by its own `NurbsCurve3DGpuRenderer` (sampler, view and LUT buffers), all
 * drawn in one pass (shared depth and multisampling) with its draw shader plus `WRAP_WGSL`.
 */
class WrappedCurve3D {
  readonly runs: { curve: NurbsCurve; renderer: NurbsCurve3DGpuRenderer; group?: GPUBindGroup }[] = [];
  private pipeline!: GPURenderPipeline;
  private targets?: RenderTargets;

  constructor(private readonly device: GPUDevice, canvas: TextureTarget, points: number[][], values: number[], lut: Float32Array) {
    try {
      for (const [a, b] of monotoneRanges(values)) {
        const idx = Array.from({ length: b - a + 1 }, (_, i) => a + i);
        if (values[b] < values[a]) idx.reverse();
        const v = idx.map((i) => values[i]);
        const curve: NurbsCurve = { degree: 1, dims: 3, points: idx.flatMap((i) => points[i]), weights: Array(idx.length).fill(1), knots: [v[0], ...v, v[v.length - 1]] };
        if (getNurbsError(curve) || !(v[v.length - 1] > v[0])) continue;
        const renderer = new NurbsCurve3DGpuRenderer(device, canvas, curve, getKnotSpans(curve));
        renderer.setColorLut(lut);
        this.runs.push({ curve, renderer });
      }
      const module = device.createShaderModule({ code: CURVE3D_WGSL + WRAP_WGSL });
      this.pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "fragmentWrapped", targets: [{ format: FRAME_FORMAT }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less-equal" },
        multisample: { count: SAMPLE_COUNT },
      });
    } catch (err) {
      this.dispose();
      throw err;
    }
  }

  /** Draws all runs into `canvas` (cleared) with the view of the renderer's `draw`, `domain` the
   * colour map's (one period). */
  draw(canvas: TextureTarget, viewProjection: number[], near: number, size: [number, number], width: number, domain: [number, number], round = false): void {
    const { device } = this;
    this.targets = getTargets(device, FRAME_FORMAT, size, this.targets);
    for (const run of this.runs) {
      const { points, weights } = projectControlPoints(run.curve, viewProjection, size, near);
      const counts = getSegmentCounts({ ...run.curve, dims: 2, points: [...points], weights: [...weights] }, getKnotSpans(run.curve), [1, 1]);
      run.renderer.setSegmentCounts(counts);
      device.queue.writeBuffer(run.renderer.view, 0, Float32Array.from([
        ...viewProjection, size[0] / 2, size[1] / 2, ...getGpuParams(run.curve, domain), width, round ? 1 : 0, 0, 0,
      ])); // as `NurbsCurve3DGpuRenderer.draw`
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.targets.color.createView(),
        resolveTarget: canvas.currentTexture().createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: "clear",
        storeOp: "discard",
      }],
      depthStencilAttachment: { view: this.targets.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setPipeline(this.pipeline);
    for (const run of this.runs) {
      const { sampler, view, lut } = run.renderer;
      if (!sampler.samples || sampler.sampleCount < 2) continue;
      const group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [view, sampler.samples, lut].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      pass.setBindGroup(0, group);
      pass.draw(6, sampler.sampleCount - 1);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    for (const r of this.runs) r.renderer.dispose();
    this.runs.length = 0;
    this.targets?.color.destroy();
    this.targets?.depth.destroy();
  }
}

/** Index ranges [a, b] over which `values` are monotonic (neighbouring ranges share a point). */
function monotoneRanges(values: number[]): [number, number][] {
  const out: [number, number][] = [];
  let start = 0;
  let dir = 0;
  for (let i = 1; i < values.length; i += 1) {
    const d = Math.sign(values[i] - values[i - 1]);
    if (d !== 0 && dir !== 0 && d !== dir) {
      out.push([start, i - 1]);
      start = i - 1;
    }
    if (d !== 0) dir = d;
  }
  out.push([start, values.length - 1]);
  return out.filter(([a, b]) => b > a);
}

interface View2D {
  plot: [number, number, number, number];
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  k: number; // pixels per unit, both axes
  w: number;
  h: number;
  s: number;
  px(x: number): number;
  py(y: number): number;
}

export class CurvePlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly frameReads: GPUBuffer[] = [];
  private dataKey?: string;
  private frameKey?: string;
  curve!: NurbsCurve;
  private spans!: KnotSpan[];
  private samples!: { u: number[]; p: number[][] };
  private polygon?: NurbsCurve;
  // 2D
  private shared: TextureTarget[];
  private r2?: NurbsGpuRenderer;
  private p2?: NurbsGpuRenderer;
  private origin: [number, number] = [0, 0];
  view2?: View2D;
  // 3D
  private canvas3: TextureTarget[];
  private r3?: NurbsCurve3DGpuRenderer;
  private p3?: NurbsCurve3DGpuRenderer;
  private wrapped?: WrappedCurve3D; // cyclic colour
  private uRange: [number, number] = [0, 1];
  private barDomain: [number, number] = [0, 1];
  private rootN: number | null = null; // root colour scale's n (3D colour by values)
  private slider: RootSlider | null = null; // as drawn
  private values?: (u: number) => number;
  private fitPoints = new Float64Array(0);
  viewProjection: number[] = [];
  content: [number, number, number, number] | null = null; // the 3D curve's pixels (state)
  private plot3: [number, number, number, number] = [0, 0, 1, 1];
  private chrome?: Quads;
  private distance = 1;
  private target: Vec3 = [0, 0, 0];

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.overlay = new OverlayRenderer(device, font);
    this.shared = [0, 1].map(() => new TextureTarget(device, FRAME_FORMAT));
    this.canvas3 = [0, 1].map(() => new TextureTarget(device, FRAME_FORMAT));
  }

  static async create(): Promise<CurvePlot> {
    const device = await requestGpuDevice();
    if (!device) throw new Error("No WebGPU device");
    return new CurvePlot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  get is3D(): boolean {
    return this.curve.dims === 3;
  }

  /** Builds the curve and its renderers, validated, then swaps them in (a bad control leaves
   * the last good plot as it was). */
  private setData(ctl: Control): void {
    const cb = ctl.colorBy;
    const root = ctl.scale === "root" && typeof cb === "object" && cb && !cb.cyclic ? snapRootN(ctl.rootN) : null;
    const key = JSON.stringify([ctl.curve, ctl.polyline, ctl.params, ctl.colorBy, ctl.colormap, ctl.invertColorMap, ctl.controlPolygon, ctl.aspect, root]);
    if (key === this.dataKey) return;
    const built: { r2?: NurbsGpuRenderer; p2?: NurbsGpuRenderer; r3?: NurbsCurve3DGpuRenderer; p3?: NurbsCurve3DGpuRenderer; wrapped?: WrappedCurve3D } = {};
    try {
      const curve = toNurbs(ctl);
      const error = getNurbsError(curve);
      if (error) throw new Error(error);
      if (curve.dims !== 2 && curve.dims !== 3) throw new Error("curves must be 2D or 3D");
      const spans = getKnotSpans(curve);
      const samples = sampleCurve(curve, spans);
      const n = curve.weights.length;
      const polygon: NurbsCurve | undefined = ctl.controlPolygon
        ? { degree: 1, dims: curve.dims, points: curve.points, weights: Array(n).fill(1), knots: [0, ...Array.from({ length: n }, (_, i) => i), n - 1] }
        : undefined;
      const d = this.device;
      let origin: [number, number] = [0, 0];
      let uRange: [number, number] = [0, 1];
      let fitPoints = new Float64Array(0);
      let color: { base: Float32Array; lut: Float32Array; values?: (u: number) => number; barDomain: [number, number] } | undefined;
      if (curve.dims === 2) {
        const p = curve.points;
        const xs = p.filter((_, i) => i % 2 === 0);
        const ys = p.filter((_, i) => i % 2 === 1);
        origin = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
        const make = (c: NurbsCurve, canvas: TextureTarget) => new NurbsGpuRenderer(d, canvas, c, getKnotSpans(c), origin);
        built.r2 = make(curve, this.shared[1]);
        built.p2 = polygon && make(polygon, this.shared[0]);
      } else {
        const make = (c: NurbsCurve, canvas: TextureTarget) => new NurbsCurve3DGpuRenderer(d, canvas, c, getKnotSpans(c));
        const { degree, knots } = curve;
        uRange = [knots[degree], knots[knots.length - degree - 1]];
        color = this.colorLut(ctl, uRange, root);
        if (typeof cb === "object" && cb?.cyclic && ctl.polyline) {
          const pts = toNurbs({ ...ctl, params: null }).points;
          const points = Array.from({ length: pts.length / 3 }, (_, i) => [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]]);
          built.wrapped = new WrappedCurve3D(d, this.canvas3[1], points, cb.values, color.base);
        } else {
          built.r3 = make(curve, this.canvas3[1]);
          built.r3.setColorLut(color.lut);
        }
        built.p3 = polygon && make(polygon, this.canvas3[0]);
        if (built.p3) {
          const [r, g, b] = [0, 2, 4].map((i) => srgbToLinear(parseInt(MAGENTA.slice(1 + i, 3 + i), 16) / 255));
          built.p3.setColorLut(Float32Array.from({ length: LUT_SIZE * 3 }, (_, i) => [r, g, b][i % 3]));
        }
        // Fit points: the evaluated curve (and polygon), grown about their centre
        const pts = [...samples.p.flat(), ...(polygon ? Array.from(curve.points) : [])];
        const centre = [0, 1, 2].map((k) => {
          let lo = Infinity, hi = -Infinity;
          for (let i = k; i < pts.length; i += 3) {
            lo = Math.min(lo, pts[i]);
            hi = Math.max(hi, pts[i]);
          }
          return (lo + hi) / 2;
        });
        fitPoints = Float64Array.from(pts, (v, i) => centre[i % 3] + (v - centre[i % 3]) * FIT_MARGIN);
      }
      // Swap
      this.dispose();
      Object.assign(this, built);
      this.curve = curve;
      this.spans = spans;
      this.samples = samples;
      this.polygon = polygon;
      this.origin = origin;
      this.uRange = uRange;
      this.fitPoints = fitPoints;
      this.values = color?.values;
      this.barDomain = color?.barDomain ?? uRange;
      this.rootN = curve.dims === 3 ? root : null;
      this.dataKey = key;
      this.frameKey = undefined;
    } catch (err) {
      built.r2?.dispose();
      built.p2?.dispose();
      built.r3?.dispose();
      built.p3?.dispose();
      built.wrapped?.dispose();
      throw err;
    }
  }

  /** LUT of the colour map (`base`), and the one by `u` (`lut`: the map, or composed
   * with a value along the curve, by the root scale if `root`); the value at `u` and the colour
   * bar's domain. */
  private colorLut(ctl: Control, uRange: [number, number], root: number | null): { base: Float32Array; lut: Float32Array; values?: (u: number) => number; barDomain: [number, number] } {
    const name = colormapName(String(ctl.colormap)) ?? "Viridis";
    const base = getLinearColorLut(name, Boolean(ctl.invertColorMap));
    const cb = ctl.colorBy;
    if (!cb || cb === "u") {
      return { base, lut: base, barDomain: [...uRange] };
    }
    // value at u: linear in the polyline's parameters (or the indices of the control points)
    const vals = cb.values;
    const params = ctl.params ?? Array.from({ length: vals.length }, (_, i) => i);
    const at = (u: number) => {
      let lo = 0, hi = vals.length - 2;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (params[mid] <= u) lo = mid;
        else hi = mid - 1;
      }
      const span = params[lo + 1] - params[lo];
      const f = span > 0 ? Math.min(Math.max((u - params[lo]) / span, 0), 1) : 0;
      return vals[lo] + f * (vals[lo + 1] - vals[lo]);
    };
    const values = cb.cyclic && cb.domain
      ? (u: number) => { const [a, b] = cb.domain!; const v = at(u); return v - Math.floor((v - a) / (b - a)) * (b - a); }
      : at;
    const barDomain: [number, number] = cb.domain ?? [Math.min(...vals), Math.max(...vals)];
    const [d0, d1] = barDomain;
    const rootScale = root === null ? null : createScale(["root", root], { domain: barDomain, clamp: true });
    const lut = new Float32Array(LUT_SIZE * 3);
    for (let i = 0; i < LUT_SIZE; i += 1) {
      const u = uRange[0] + ((uRange[1] - uRange[0]) * i) / (LUT_SIZE - 1);
      const r = (at(u) - d0) / (d1 - d0 || 1);
      const t = (rootScale ? rootScale(at(u)) : cb.cyclic ? r - Math.floor(r) : Math.min(Math.max(r, 0), 1)) * (LUT_SIZE - 1);
      const k = Math.min(Math.floor(t), LUT_SIZE - 2);
      const f = t - k;
      for (let c = 0; c < 3; c += 1) lut[i * 3 + c] = base[k * 3 + c] * (1 - f) + base[(k + 1) * 3 + c] * f; // as COLOR_WGSL
    }
    return { base, lut, values, barDomain };
  }

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 19 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 13 * s),
    };
  }

  // 2D ---------------------------------------------------------------------------------------------

  private frame2D(ctl: Control, w: number, h: number, s: number): void {
    const [L, T, R, B] = [Math.round(66 * s), Math.round(56 * s), w - Math.round(24 * s), h - Math.round(40 * s)];
    let [x0, x1, y0, y1] = [Infinity, -Infinity, Infinity, -Infinity];
    const take = (x: number, y: number) => {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    };
    for (const p of this.samples.p) take(p[0], p[1]);
    if (this.polygon) for (let i = 0; i < this.curve.points.length; i += 2) take(this.curve.points[i], this.curve.points[i + 1]);
    const m = MARGIN * Math.max(x1 - x0, y1 - y0);
    [x0, x1, y0, y1] = [ctl.x0 ?? x0 - m, ctl.x1 ?? x1 + m, ctl.ymin ?? y0 - m, ctl.ymax ?? y1 + m];
    const k = Math.min((R - L) / (x1 - x0), (B - T) / (y1 - y0)); // equal aspect: centre the other axis
    const [cx, cy] = [(x0 + x1) / 2, (y0 + y1) / 2];
    [x0, x1] = [cx - (R - L) / k / 2, cx + (R - L) / k / 2];
    [y0, y1] = [cy - (B - T) / k / 2, cy + (B - T) / k / 2];
    const v: View2D = {
      plot: [L, T, R, B], x0, x1, y0, y1, k, w, h, s,
      px: (x) => L + (x - x0) * k,
      py: (y) => T + (y1 - y) * k,
    };
    this.view2 = v;
    const view = {
      scale: [k, k] as [number, number],
      offset: [L + (this.origin[0] - x0) * k - w / 2, h / 2 - T - (y1 - this.origin[1]) * k] as [number, number],
      size: [w, h] as [number, number],
    };
    for (const c of this.shared) c.clear(w, h);
    const draw = (r: NurbsGpuRenderer, c: NurbsCurve, width: number, color: Rgba) => {
      r.setSegmentCounts(getSegmentCounts(c, getKnotSpans(c), [k, k]));
      r.draw({ ...view, width, color }, true);
    };
    if (this.p2) draw(this.p2, this.polygon!, 1.5 * s, premul(ORANGE));
    draw(this.r2!, this.curve, ctl.lineWidth * s, premul(ctl.color));

    // Chrome: grid, ticks, zero lines, title, control points
    const q = new Quads(this.font);
    const F = this.fonts(s);
    for (const t of axisTicks({}, x0, x1, (R - L) / s)) {
      const X = Math.round(v.px(t.value));
      if (X < L || X > R) continue;
      q.rect(X, T, 1, B - T + 1, premul(t.value === 0 ? ZERO : GRID));
      q.text(t.label, X + 0.5, B + Math.round(8 * s), F.tick, premul(INK2), "mt");
    }
    for (const t of axisTicks({}, y0, y1, (B - T) / s)) {
      const Y = Math.round(v.py(t.value));
      if (Y < T || Y > B) continue;
      q.rect(L, Y, R - L + 1, 1, premul(t.value === 0 ? ZERO : GRID));
      q.text(t.label, L - Math.round(10 * s), Y + 0.5, F.tick, premul(INK2), "rm");
    }
    this.titleRow(q, ctl, L, s);
    this.chrome = q;
  }

  // 3D ---------------------------------------------------------------------------------------------

  private frame3D(ctl: Control, w: number, h: number, s: number): void {
    const bar = ctl.colorBy ? Math.round(110 * s) : 0;
    const [L, T, R, B] = [Math.round(16 * s), Math.round(56 * s), w - Math.round(16 * s) - bar, h - Math.round(16 * s)];
    this.plot3 = [L, T, R, B];
    const [pw, ph] = [R - L, B - T];
    const fit = fitView(this.fitPoints, ...fitAngles(ctl), FOV, pw / ph) ?? [[0, 0, 0], 3];
    this.target = fit[0];
    this.distance = fit[1] / (Number(ctl.zoom) || 1);
    const [f] = cameraBasis(ctl.azimuth, ctl.elevation);
    const eye = [0, 1, 2].map((i) => this.target[i] + this.distance * f[i]) as Vec3;
    const near = this.distance / 100;
    const projection = perspectiveProjection(FOV, pw / ph, near, this.distance * 100);
    // Into the plot rectangle of the frame (NDC scale and shift after the projection)
    const toRect = [pw / w, 0, 0, 0, 0, ph / h, 0, 0, 0, 0, 1, 0, (L + R) / w - 1, 1 - (T + B) / h, 0, 1];
    const vp = mul4(toRect, mul4(projection, lookAtView(eye, this.target, [0, 0, 1])));
    this.viewProjection = vp;
    this.content = screenRect(this.fitPoints, vp, w, h);
    const round = ctl.lineCap === "round";
    const draw = (r: NurbsCurve3DGpuRenderer, c: NurbsCurve, canvas: TextureTarget, width: number, round = false) => {
      canvas.width = w;
      canvas.height = h;
      const { points, weights } = projectControlPoints(c, vp, [w, h], near);
      const counts = getSegmentCounts({ ...c, dims: 2, points: [...points], weights: [...weights] }, getKnotSpans(c), [1, 1]);
      r.setSegmentCounts(counts);
      r.draw({ viewProjection: this.viewProjection, uRange: this.uRange, width, round, size: [w, h] });
    };
    if (this.p3) draw(this.p3, this.polygon!, this.canvas3[0], 1.5 * s);
    if (this.wrapped) {
      this.canvas3[1].width = w;
      this.canvas3[1].height = h;
      this.wrapped.draw(this.canvas3[1], this.viewProjection, near, [w, h], ctl.lineWidth * s, this.barDomain, round);
    } else {
      draw(this.r3!, this.curve, this.canvas3[1], ctl.lineWidth * s, round);
    }

    const q = new Quads(this.font);
    this.titleRow(q, ctl, L + Math.round(50 * s), s);
    // Root scale: its slider under the colour bar (raised to make room)
    const sliderTop = B - Math.round(4 * s) - rootSliderHeight(this.fonts(s).label, s);
    const barBottom = this.rootN === null ? B - Math.round(30 * s) : sliderTop - Math.round(16 * s);
    if (ctl.colorBy) this.colorBar(q, ctl, R + Math.round(30 * s), T + Math.round(20 * s), barBottom, s);
    this.slider = this.rootN === null ? null : rootSlider(q, this.fonts(s).label, R + Math.round(20 * s), sliderTop, Math.round(84 * s), this.rootN, s);
    this.chrome = q;
  }

  private colorBar(q: Quads, ctl: Control, x: number, top: number, bottom: number, s: number): void {
    const name = colormapName(String(ctl.colormap)) ?? "Viridis";
    const bw = Math.round(14 * s);
    const F = this.fonts(s);
    for (let y = top; y < bottom; y += 1) {
      const [r, g, b] = colormapRgb(name, 1 - (y + 0.5 - top) / (bottom - top), Boolean(ctl.invertColorMap));
      q.rect(x, y, bw, 1, [r / 255, g / 255, b / 255, 1]);
    }
    q.rect(x, top, bw, bottom - top, [0, 0, 0, 0], { border: 1, borderColor: LABEL_BORDER });
    const [d0, d1] = this.barDomain;
    const cb = ctl.colorBy;
    const root = this.rootN === null ? null : createScale(["root", this.rootN], { domain: [d0, d1], range: [bottom, top] });
    const ticks = root
      ? colorBarTicks(["root", this.rootN!], [d0, d1], (bottom - top) / s)
      : axisTicks({ ticks: typeof cb === "object" && cb ? cb.ticks : undefined }, d0, d1, (bottom - top) / s);
    for (const t of ticks) {
      const Y = Math.round(root ? root(t.value) : bottom - ((t.value - d0) / (d1 - d0)) * (bottom - top));
      q.rect(x + bw, Y, Math.round(4 * s), 1, premul(INK2));
      q.text(t.label, x + bw + Math.round(7 * s), Y + 0.5, F.tick, premul(INK2), "lm");
    }
    const label = typeof cb === "object" && cb?.label ? cb.label : ctl.colorLabel;
    q.text(label, x + bw / 2, top - Math.round(10 * s), F.sub, premul(INK), "mb");
  }

  private titleRow(q: Quads, ctl: Control, x: number, s: number): void {
    const F = this.fonts(s);
    const y = Math.round(28 * s);
    const tw = ctl.title ? q.text(ctl.title, x, y, F.title, premul(INK), "lm") : 0;
    if (ctl.subtitle) q.text(ctl.subtitle, x + tw + Math.round(16 * s), y, F.sub, premul(INK2), "lm");
  }

  // Hover ------------------------------------------------------------------------------------------

  /** Screen position of a point (2D: the view; 3D: the view-projection). */
  private project(p: number[]): [number, number] | null {
    if (!this.is3D) {
      return [this.view2!.px(p[0]), this.view2!.py(p[1])];
    }
    const e = this.viewProjection;
    const cw = e[3] * p[0] + e[7] * p[1] + e[11] * p[2] + e[15];
    if (!(cw > 0)) return null;
    const x = (e[0] * p[0] + e[4] * p[1] + e[8] * p[2] + e[12]) / cw;
    const y = (e[1] * p[0] + e[5] * p[1] + e[9] * p[2] + e[13]) / cw;
    return [((x + 1) / 2) * this.size[0], ((1 - y) / 2) * this.size[1]];
  }

  private size: [number, number] = [1, 1];

  private hover(ctl: Control): { pixel: [number, number]; u: number; point: number[]; at: [number, number] } | null {
    if (!ctl.hover) return null;
    const [hx, hy] = ctl.hover.map(Number);
    let best = (24 * this.scale) ** 2;
    let found: { pixel: [number, number]; u: number; point: number[]; at: [number, number] } | null = null;
    this.samples.p.forEach((p, i) => {
      const sp = this.project(p);
      if (!sp) return;
      const d = (sp[0] - hx) ** 2 + (sp[1] - hy) ** 2;
      if (d < best) {
        best = d;
        found = { pixel: [hx, hy], u: this.samples.u[i], point: p, at: sp };
      }
    });
    return found;
  }

  private scale = 1;

  private drawHover(q: Quads, ctl: Control, info: NonNullable<ReturnType<CurvePlot["hover"]>>): void {
    const s = this.scale;
    const color = this.is3D ? INK : ctl.color;
    q.dot(info.at[0], info.at[1], 6 * s, 2 * s, premul(color), premul(HOVER_RING));
    const font = this.fonts(s).label;
    const names = ["x", "y", "z"];
    const rows: [string, string, string][] = [[String(ctl.paramLabel ?? "𝑢"), formatNumber(info.u, 5), INK]];
    const a = this.is3D && ctl.aspect ? ctl.aspect : [1, 1, 1];
    const axisNames = (ctl.axisLabels as string[] | undefined) ?? names;
    info.point.forEach((v, i) => rows.push([axisNames[i] ?? names[i], formatNumber(v / a[i], 5), INK]));
    if (this.values) {
      const cb = ctl.colorBy as { label?: string };
      rows.push([cb.label ?? "value", formatNumber(this.values(info.u), 5), INK]);
    }
    if (this.slider) rows.push(["ⁿ√ n", this.slider.value.toFixed(2), INK]);
    const gapRow = Math.round(3 * s);
    const lh = font.ascent + font.descent + gapRow;
    const kw = Math.max(...rows.map(([k]) => this.font.measure(font, k))) + Math.round(12 * s);
    const vw = Math.max(...rows.map(([, v]) => this.font.measure(font, v)));
    const pad = Math.round(8 * s);
    const lw = Math.ceil(pad * 2 + kw + vw);
    const lhTotal = pad * 2 + lh * rows.length - gapRow;
    const [hx, hy] = info.pixel;
    const gap = Math.round(14 * s);
    const [w, h] = this.size;
    let x = hx + gap + lw <= w ? hx + gap : hx - gap - lw;
    let y = hy + gap + lhTotal <= h ? hy + gap : hy - gap - lhTotal;
    x = Math.trunc(Math.min(Math.max(x, 0), w - lw));
    y = Math.trunc(Math.min(Math.max(y, 0), h - lhTotal));
    q.rect(x, y, lw, lhTotal, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    rows.forEach(([k, v, c], i) => {
      const ry = y + pad + i * lh;
      q.text(k, x + pad, ry, font, premul(INK2), "lt");
      q.text(v, x + lw - pad, ry, font, premul(c), "rt");
    });
  }

  // Frame ------------------------------------------------------------------------------------------

  async submit(ctl: Control, slot: number): Promise<Pending> {
    ctl = expandOde(expandGen(ctl));
    const { device } = this;
    this.setData(ctl);
    const w = Math.trunc(ctl.width);
    const h = Math.trunc(ctl.height);
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    this.scale = s;
    this.size = [w, h];
    const key = JSON.stringify([this.dataKey, w, h, ctl.azimuth, ctl.elevation, ctl.zoom, ctl.x0, ctl.x1, ctl.ymin, ctl.ymax,
      ctl.title, ctl.subtitle, ctl.color, ctl.lineWidth, ctl.lineCap, ctl.colorLabel]);
    if (key !== this.frameKey) {
      if (this.is3D) this.frame3D(ctl, w, h, s);
      else this.frame2D(ctl, w, h, s);
      this.overlay.setSize(w, h);
      this.frameKey = key;
    }
    const clip: [number, number, number, number] = this.is3D ? [0, 0, w, h] : this.view2!.plot;
    const q = new Quads(this.font);
    q.append(this.chrome!);
    if (this.polygon) q.layer(0, POLYGON_OPACITY, clip);
    q.layer(1, 1, clip);
    if (!this.is3D && this.polygon) {
      const v = this.view2!;
      const p = this.curve.points;
      for (let i = 0; i < p.length; i += 2) {
        q.dot(v.px(p[i]), v.py(p[i + 1]), 4 * s, 1.5 * s, premul(ORANGE), premul(SURFACE), true);
      }
    }
    const info = this.hover(ctl);
    if (info) this.drawHover(q, ctl, info);

    const encoder = device.createCommandEncoder();
    const layers = this.is3D ? this.canvas3.map((c) => c.currentTexture()) : this.shared.map((c) => c.currentTexture());
    this.overlay.encode(encoder, q, layers, clip);
    const pitch = rowBytes(w);
    if (this.frameReads[slot]?.size !== pitch * h) {
      this.frameReads[slot]?.destroy();
      this.frameReads[slot] = makeReadBuffer(device, pitch * h);
    }
    const frameRead = this.frameReads[slot];
    encoder.copyTextureToBuffer({ texture: this.overlay.out! }, { buffer: frameRead, bytesPerRow: pitch, rowsPerImage: h }, [w, h]);
    device.queue.submit([encoder.finish()]);
    const hover = info && { u: (info as { u: number }).u, point: (info as { point: number[] }).point };
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
        return { width: w, height: h, rgba, hover };
      },
    };
  }

  state(ctl: Control, generation: number, hover: unknown, ms: number): unknown {
    const base = { generation, width: this.size[0], height: this.size[1], dims: this.curve.dims, hover, renderMs: Math.round(ms * 1000) / 1000 };
    if (this.is3D) {
      const root = this.slider ? { rootN: this.slider.value, rootSlider: this.slider } : {};
      return { ...base, azimuth: ctl.azimuth, elevation: ctl.elevation, distance: this.distance, target: this.target, plotRect: this.plot3, content: this.content, colorDomain: this.barDomain, ...root };
    }
    const v = this.view2!;
    return { ...base, x0: v.x0, x1: v.x1, ymin: v.y0, ymax: v.y1, plotRect: v.plot };
  }

  private dispose(): void {
    this.r2?.dispose();
    this.p2?.dispose();
    this.r3?.dispose();
    this.p3?.dispose();
    this.wrapped?.dispose();
    this.r2 = this.p2 = this.r3 = this.p3 = this.wrapped = undefined;
  }
}

if (import.meta.main) {
  await main<Control>({
    name: "curve.ts",
    defaults: DEFAULTS as Control,
    create: CurvePlot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
  });
}
