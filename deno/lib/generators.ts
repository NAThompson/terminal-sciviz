/**
 * Generator specs: pure functions of x (and parametric curves of t) named in a control instead of
 * their samples, e.g. `{name: "besselJ", order: 3}` or `"bumps"`, evaluated in f32 by a WGSL
 * compute shader, straight into the buffers the curves are drawn from: `x, y, 0, u` per sample,
 * relative to an origin, as lib/chords.ts draws them (line.ts), or `x, y, z, value` per point
 * (curve.ts). Each also
 * has its f64 CPU form (hover read-outs, and the reference the GPU is checked against:
 * tools/check_generators.ts).
 */

import { besselJ } from "./specfun.ts";
import { FUNCTIONS } from "./pchip.ts";

export type FnSpec = string | ({ name: string; domain?: [number, number] } & Record<string, unknown>);

interface Generator {
  id: number; // case of `evaluate` in the shader
  domain: [number, number];
  args: string[]; // names of the spec's numbers, in .x, .y, .z, .w of `args`
  defaults: number[];
  f: (x: number, a: number[]) => number; // f64
}

const sincCpu = (u: number) => (u === 0 ? 1 : Math.sin(u) / u);
const pure = (name: string) => FUNCTIONS[name].f;

export const GENERATORS: Record<string, Generator> = {
  bumps: { id: 0, domain: FUNCTIONS.bumps.domain, args: [], defaults: [], f: pure("bumps") },
  blocks: { id: 1, domain: FUNCTIONS.blocks.domain, args: [], defaults: [], f: pure("blocks") },
  runge: { id: 2, domain: FUNCTIONS.runge.domain, args: [], defaults: [], f: pure("runge") },
  step: { id: 3, domain: FUNCTIONS.step.domain, args: [], defaults: [], f: pure("step") },
  /** J_n(x), integer n >= 0 */
  besselJ: { id: 4, domain: [0, 30], args: ["order"], defaults: [0], f: (x, [n]) => besselJ(n, x) },
  /** sin(k x) / (k x) */
  sinc: { id: 5, domain: [-2 * Math.PI, 2 * Math.PI], args: ["k"], defaults: [Math.PI], f: (x, [k]) => sincCpu(k * x) },
  /** |H(f)| in dB of a second-order low-pass, r = f / f0: 20 log10(1 / |1 − r² + 2iζr|) */
  bode: {
    id: 6, domain: [1, 1e5], args: ["f0", "zeta"], defaults: [2e4 / (2 * Math.PI), 0.1],
    f: (x, [f0, z]) => {
      const r = x / f0;
      return 20 * Math.log10(1 / Math.hypot(1 - r * r, 2 * z * r));
    },
  },
  /** tight-binding band b: 2 cos k + 4b − 0.4 b cos 2k */
  dispersion: { id: 7, domain: [-Math.PI, Math.PI], args: ["band"], defaults: [0], f: (k, [b]) => 2 * Math.cos(k) + 4 * b - 0.4 * b * Math.cos(2 * k) },
  /** exp(−x/τ) cos(ωx) */
  dampedCos: { id: 8, domain: [0, 30], args: ["tau", "omega"], defaults: [1, 0], f: (x, [tau, w]) => Math.exp(-x / tau) * Math.cos(w * x) },
};

export interface ResolvedFn {
  key: string;
  id: number;
  args: number[]; // 4
  domain: [number, number];
  f: (x: number) => number; // f64
}

export function resolveFn(spec: FnSpec): ResolvedFn {
  const s = typeof spec === "string" ? { name: spec } : spec;
  const g = GENERATORS[s.name];
  if (!g) throw new Error(`fn must be one of ${Object.keys(GENERATORS)} (or {name, ...})`);
  const args = g.args.map((a, i) => (s[a] === undefined ? g.defaults[i] : Number(s[a])));
  if (!args.every(Number.isFinite)) throw new Error(`fn ${s.name}: ${g.args.join(", ")} must be numbers`);
  if (s.name === "besselJ" && !(Number.isInteger(args[0]) && args[0] >= 0 && args[0] <= 100)) throw new Error("besselJ: order must be an integer in [0, 100]");
  const domain = (s.domain ?? g.domain).map(Number) as [number, number];
  if (!(domain[1] > domain[0])) throw new Error(`fn ${s.name}: domain must increase`);
  return { key: JSON.stringify([s.name, args, domain]), id: g.id, args: [...args, 0, 0, 0, 0].slice(0, 4), domain, f: (x) => g.f(x, args) };
}

/** Parametric curves of t: `{name: "trefoil", n, t: [t0, t1]}` (point, curvature κ) or
 * `{name: "chirp", n, t, rate, width}` ((t, ℜ, ℑ) of exp(i rate t²) exp(−t²/width), phase). */
export const CURVES: Record<string, { id: number; t: [number, number]; args: string[]; defaults: number[] }> = {
  trefoil: { id: 0, t: [0, 2 * Math.PI], args: [], defaults: [] },
  chirp: { id: 1, t: [-6, 6], args: ["rate", "width"], defaults: [1.5, 12] },
};

const AXIS = { linear: 0, log: 1, symlog: 2 } as const;
type Scale = keyof typeof AXIS;

const WGSL = /* wgsl */ `
  struct Params {
    rel0: f32, // first sample's axis-space x, relative to origin.x
    dw: f32, // axis-space x between samples
    origin: vec2f, // of the written samples (axis space)
    count: u32,
    func: u32,
    xScale: u32, // 0 linear, 1 log10, 2 symlog
    yScale: u32,
    xc: f32, // symlog constants
    yc: f32,
    w0: f32, // first sample's axis-space x (absolute)
    mode: u32, // 0: graph y = f(x) of \`fn\`; 1: parametric curve \`fn\`
    args: vec4f,
  }

  @group(0) @binding(0) var<uniform> p: Params;
  @group(0) @binding(1) var<storage, read_write> out: array<vec4f>;

  const PI = 3.14159265358979;
  const DJ_T = array(0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81);
  const BUMPS_H = array(4.0, 5.0, 3.0, 4.0, 5.0, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2);
  const BUMPS_W = array(0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005);
  const BLOCKS_H = array(4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2);

  fn log1p_(u: f32) -> f32 {
    let w = 1.0 + u;
    return select(log(w) * u / (w - 1.0), u, w == 1.0);
  }
  fn expm1_(u: f32) -> f32 {
    return select(exp(u) - 1.0, u + 0.5 * u * u, abs(u) < 1e-4);
  }
  fn fwd(v: f32, scale: u32, c: f32) -> f32 {
    switch (scale) {
      case 1u: { return select(select(bitcast<f32>(0x7fc00000u), bitcast<f32>(0xff800000u), v == 0.0), log2(v) * 0.301029995663981, v > 0.0); }
      case 2u: { return sign(v) * log1p_(abs(v) / c); }
      default: { return v; }
    }
  }
  fn inv(w: f32, scale: u32, c: f32) -> f32 {
    switch (scale) {
      case 1u: { return exp2(w * 3.32192809488736); }
      case 2u: { return sign(w) * expm1_(abs(w)) * c; }
      default: { return w; }
    }
  }

  // J_n(x) by Miller's backward recurrence from an even order m well above n and |x|, normalised
  // by J_0 + 2 (J_2 + J_4 + ...) = 1, rescaled on the way down to stay within f32 (Numerical
  // Recipes' bessj, used for every x: stable downward, so also above n)
  fn besselJ(n: i32, x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-20) { return select(0.0, 1.0, n == 0); }
    let top = f32(max(n, i32(ceil(ax)))) + 2.0;
    let m = 2 * ((i32(top + sqrt(160.0 * top)) + 1) / 2);
    let tox = 2.0 / ax;
    var bjp = 0.0;
    var bj = 1.0;
    var sum = 0.0;
    var ans = 0.0;
    var even = false; // order j - 1 after the step at j is even <=> j odd
    for (var j = m; j > 0; j--) {
      let bjm = f32(j) * tox * bj - bjp;
      bjp = bj;
      bj = bjm;
      if (abs(bj) > 1e10) {
        bj *= 1e-10;
        bjp *= 1e-10;
        ans *= 1e-10;
        sum *= 1e-10;
      }
      if (even) { sum += bj; }
      even = !even;
      if (j == n) { ans = bjp; }
    }
    if (n == 0) { ans = bj; }
    sum = 2.0 * sum - bj;
    let r = ans / sum;
    return select(r, -r, x < 0.0 && (n & 1) == 1);
  }

  // (sin x, cos x) to ~1 ulp: the built-ins are fast approximations here (Metal: ~5e-7 absolute
  // on [-π, π], more outside). Cody-Waite reduction by π/2 in three parts (fma: exact products),
  // Cephes' sinf / cosf polynomials on [-π/4, π/4]
  fn sincos_(x: f32) -> vec2f {
    let q = round(x * 0.636619772367581);
    var r = fma(-q, 1.5703125, x);
    r = fma(-q, 4.837512969970703e-4, r);
    r = fma(-q, 7.549789954891882e-8, r);
    let r2 = r * r;
    let s = fma(r * r2, -1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4), r);
    let c = fma(r2 * r2, 4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5), 1.0 - 0.5 * r2);
    switch (i32(q) & 3) {
      case 0: { return vec2f(s, c); }
      case 1: { return vec2f(c, -s); }
      case 2: { return vec2f(-s, -c); }
      default: { return vec2f(-c, s); }
    }
  }
  fn sin_(x: f32) -> f32 { return sincos_(x).x; }
  fn cos_(x: f32) -> f32 { return sincos_(x).y; }

  fn tanh_(u: f32) -> f32 {
    let e = exp(-2.0 * abs(clamp(u, -20.0, 20.0)));
    return sign(u) * (1.0 - e) / (1.0 + e);
  }

  fn evaluate(id: u32, x: f32, a: vec4f) -> f32 {
    switch (id) {
      case 0u: { // bumps
        var s = 0.0;
        for (var j = 0; j < 11; j++) {
          let u = (x - DJ_T[j]) / BUMPS_W[j];
          let u2 = u * u;
          s += BUMPS_H[j] / (1.0 + u2 * u2);
        }
        return s;
      }
      case 1u: { // blocks
        var s = 0.0;
        for (var j = 0; j < 11; j++) { s += BLOCKS_H[j] * (1.0 + sign(x - DJ_T[j])) / 2.0; }
        return s;
      }
      case 2u: { return 1.0 / (1.0 + 25.0 * x * x); }
      case 3u: { return (1.0 + tanh_(x / 0.05)) / 2.0; }
      case 4u: { return besselJ(i32(a.x), x); }
      case 5u: {
        let u = a.x * x;
        let u2 = u * u; // near 0: Taylor (sin's absolute error over a small u would show)
        return select(sin_(u) / u, 1.0 - u2 / 6.0 * (1.0 - u2 / 20.0 * (1.0 - u2 / 42.0 * (1.0 - u2 / 72.0))), abs(u) < 0.5);
      }
      case 6u: {
        let r = x / a.x;
        return -10.0 * log2((1.0 - r * r) * (1.0 - r * r) + 4.0 * a.y * a.y * r * r) * 0.301029995663981;
      }
      case 7u: { return 2.0 * cos_(x) + 4.0 * a.x - 0.4 * a.x * cos_(2.0 * x); }
      default: { // the product ωx rounded (up to 39 rad here) would cost ~4e-6: corrected
        let wx = a.y * x;
        let cs = sincos_(wx);
        return exp(-x / a.x) * (cs.y - cs.x * fma(a.y, x, -wx));
      }
    }
  }

  fn curve(id: u32, t: f32, a: vec4f) -> vec4f {
    switch (id) {
      case 0u: { // trefoil (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), curvature |r′ × r″| / |r′|³;
        // multiples of t from sin t, cos t (no f32 rounding of 2t, 3t)
        let s1 = sin_(t);
        let c1 = cos_(t);
        let s2 = 2.0 * s1 * c1;
        let c2 = 1.0 - 2.0 * s1 * s1;
        let s3 = s1 * (3.0 - 4.0 * s1 * s1);
        let c3 = c1 * (4.0 * c1 * c1 - 3.0);
        let d1 = vec3f(c1 + 4.0 * c2, -s1 + 4.0 * s2, -3.0 * c3);
        let d2 = vec3f(-s1 - 8.0 * s2, -c1 + 8.0 * c2, 9.0 * s3);
        let l = length(d1);
        return vec4f(s1 + 2.0 * s2, c1 - 2.0 * c2, -s3, length(cross(d1, d2)) / (l * l * l));
      }
      default: { // chirp: (t, ℜ, ℑ) of exp(i a t²) exp(−t²/b), phase a t² (reduced mod 2π from
        // t² exact in two f32s: up to 54 rad, its f32 rounding alone would be ~4e-6)
        let hi = t * t;
        let lo = fma(t, t, -hi);
        let ph = a.x * hi;
        let k = round(ph / 6.28318530717959);
        let r = fma(-k, 6.28318548202515, ph) + (fma(a.x, hi, -ph) + a.x * lo + k * 1.74845553e-7);
        let e = exp(-hi / a.y);
        return vec4f(t, e * cos_(r), e * sin_(r), ph);
      }
    }
  }

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) { return; }
    let fi = f32(i);
    if (p.mode == 1u) {
      out[i] = curve(p.func, p.w0 + fi * p.dw, p.args);
      return;
    }
    let x = inv(p.w0 + fi * p.dw, p.xScale, p.xc);
    let y = fwd(evaluate(p.func, x, p.args), p.yScale, p.yc);
    out[i] = vec4f(p.rel0 + fi * p.dw, y - p.origin.y, 0.0, x); // .w (u): unused by the 2D draw
  }
`;

export interface Axis { scale?: Scale | string; threshold?: number }

/** One graph to sample: `count` samples from axis-space x `w0` every `dw`, written relative to
 * `origin` (axis space). */
export interface GraphJob { fn: ResolvedFn; w0: number; dw: number; count: number; origin: [number, number]; x: Axis; y: Axis }

const STORAGE = 0x80, COPY_SRC = 0x04, COPY_DST = 0x08, UNIFORM = 0x40, MAP_READ = 0x01;

export class FnSampler {
  private readonly pipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL }) } });
  }

  /** A buffer for `count` samples (vec4f), drawn by lib/chords.ts and copied from. */
  buffer(count: number): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, 2 ** Math.ceil(Math.log2(count * 16))), usage: STORAGE | COPY_SRC | COPY_DST });
  }

  private dispatch(pass: GPUComputePassEncoder, out: GPUBuffer, params: ArrayBuffer, count: number, temps: GPUBuffer[]): void {
    const uniform = this.device.createBuffer({ size: params.byteLength, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(uniform, 0, params);
    temps.push(uniform);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: out, size: count * 16 } }],
    }));
    pass.dispatchWorkgroups(Math.ceil(count / 64));
  }

  /** Samples the graphs into `outs` (lib/chords.ts' layout), and reads back their axis-space y
   * and the x (value) each was evaluated at. */
  async graphs(jobs: GraphJob[], outs: GPUBuffer[]): Promise<{ x: Float32Array; y: Float32Array }[]> {
    const { device } = this;
    const temps: GPUBuffer[] = [];
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    jobs.forEach((j, k) => {
      const params = new ArrayBuffer(64);
      const f = new Float32Array(params), u = new Uint32Array(params);
      f.set([j.w0 - j.origin[0], j.dw, j.origin[0], j.origin[1]]);
      u.set([j.count, j.fn.id, AXIS[(j.x.scale ?? "linear") as Scale] ?? 0, AXIS[(j.y.scale ?? "linear") as Scale] ?? 0], 4);
      f.set([j.x.threshold ?? 1, j.y.threshold ?? 1, j.w0], 8);
      u[11] = 0;
      f.set(j.fn.args, 12);
      this.dispatch(pass, outs[k], params, j.count, temps);
    });
    pass.end();
    const total = jobs.reduce((a, j) => a + j.count * 16, 0);
    const read = device.createBuffer({ size: Math.max(16, total), usage: MAP_READ | COPY_DST });
    let offset = 0;
    jobs.forEach((j, k) => {
      encoder.copyBufferToBuffer(outs[k], 0, read, offset, j.count * 16);
      offset += j.count * 16;
    });
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(MAP_READ);
    const all = new Float32Array(read.getMappedRange().slice(0));
    read.destroy();
    for (const t of temps) t.destroy();
    offset = 0;
    return jobs.map((j) => {
      const x = new Float32Array(j.count), y = new Float32Array(j.count);
      for (let i = 0; i < j.count; i += 1) {
        y[i] = all[(offset + i) * 4 + 1] + j.origin[1];
        x[i] = all[(offset + i) * 4 + 3];
      }
      offset += j.count;
      return { x, y };
    });
  }

  /** f at `n` even samples of [a, b] (linear axes), in f32, and those x (f32). */
  async values(fn: ResolvedFn, a: number, b: number, n: number): Promise<{ x: Float32Array; y: Float32Array }> {
    const out = this.buffer(n);
    try {
      return (await this.graphs([{ fn, w0: a, dw: (b - a) / Math.max(n - 1, 1), count: n, origin: [0, 0], x: {}, y: {} }], [out]))[0];
    } finally {
      out.destroy();
    }
  }

  /** Points `x, y, z, value` of a parametric curve at `n` even values of t over [t0, t1]. */
  async curve(name: string, n: number, t: [number, number], args: number[]): Promise<Float32Array> {
    const c = CURVES[name];
    const { device } = this;
    const out = this.buffer(n);
    const read = device.createBuffer({ size: n * 16, usage: MAP_READ | COPY_DST });
    const temps: GPUBuffer[] = [];
    const params = new ArrayBuffer(64);
    const f = new Float32Array(params), u = new Uint32Array(params);
    f[1] = (t[1] - t[0]) / Math.max(n - 1, 1);
    u.set([n, c.id], 4);
    f[10] = t[0];
    u[11] = 1;
    f.set([...args, 0, 0, 0, 0].slice(0, 4), 12);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.dispatch(pass, out, params, n, temps);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, read, 0, n * 16);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(MAP_READ);
    const pts = new Float32Array(read.getMappedRange().slice(0));
    read.destroy();
    out.destroy();
    for (const b of temps) b.destroy();
    return pts;
  }
}

const samplers = new WeakMap<GPUDevice, FnSampler>();
export function fnSampler(device: GPUDevice): FnSampler {
  let s = samplers.get(device);
  if (!s) samplers.set(device, (s = new FnSampler(device)));
  return s;
}
