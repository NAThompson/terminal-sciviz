/**
 * The f32 GPU generators (lib/generators.ts) against their f64 CPU forms, over the ranges the
 * gallery plots, 100 001 samples each, the CPU at the abscissas the GPU used (read back: its f32
 * x = w0 + i dw is up to an ulp off the f64 one, which shifts nothing on screen but would
 * dominate the comparison where f is steep): max |error| / max |f|, and max relative error where
 * |f| >= 5% of max |f| (not near a zero: f32 errors are absolute, ~1e-6 max |f|; blocks: not within 1e-6 of a jump).
 *
 *   deno run -A tools/check_generators.ts      # exits 1 if a relative error is above 1e-5
 */

import { CURVES, fnSampler, type FnSpec, resolveFn } from "../lib/generators.ts";

const device = await (await navigator.gpu.requestAdapter())?.requestDevice();
if (!device) throw new Error("No WebGPU device");
const sampler = fnSampler(device);
const N = 100_001;
const TOL = 1e-5;
let bad = 0;

const JUMPS = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81];

function report(name: string, ref: (i: number) => number, got: (i: number) => number, n: number, skip = (_i: number) => false): void {
  let scale = 0;
  for (let i = 0; i < n; i += 1) scale = Math.max(scale, Math.abs(ref(i)));
  let abs = 0, rel = 0;
  for (let i = 0; i < n; i += 1) {
    if (skip(i)) continue;
    const r = ref(i), e = Math.abs(got(i) - r);
    abs = Math.max(abs, e);
    if (Math.abs(r) >= 0.05 * scale) rel = Math.max(rel, e / Math.abs(r));
  }
  if (rel > TOL) bad += 1;
  console.log(`${name.padEnd(40)} max|err|/max|f| ${(abs / scale).toExponential(2)}  max rel ${rel.toExponential(2)}${rel > TOL ? "  ABOVE 1e-5" : ""}`);
}

const specs: FnSpec[] = [
  "bumps", "blocks", "runge", "step",
  ...Array.from({ length: 6 }, (_, n) => ({ name: "besselJ", order: n, domain: [0, 30] as [number, number] })),
  { name: "besselJ", order: 0, domain: [0, 100] }, { name: "besselJ", order: 20, domain: [0, 60] },
  { name: "sinc", k: 1.5 },
  ...Array.from({ length: 5 }, (_, b) => ({ name: "dispersion", band: b })),
  { name: "dampedCos", tau: 14, omega: 1.3 }, { name: "dampedCos", tau: 14 },
];
for (const spec of specs) {
  const fn = resolveFn(spec);
  const [a, b] = fn.domain;
  const { x, y } = await sampler.values(fn, a, b, N);
  const skip = fn.key.startsWith('["blocks"') ? (i: number) => JUMPS.some((t) => Math.abs(x[i] - t) < 1e-6) : undefined;
  report(fn.key, (i) => fn.f(x[i]), (i) => y[i], N, skip);
}

// Bode on its log x axis (as line.ts samples it): x = 10^w, w even in [0, 5]
{
  const fn = resolveFn({ name: "bode" });
  const [w0, w1] = [0, 5];
  const out = sampler.buffer(N);
  const [{ x, y }] = await sampler.graphs([{ fn, w0, dw: (w1 - w0) / (N - 1), count: N, origin: [2.5, 0], x: { scale: "log" }, y: {} }], [out]);
  out.destroy();
  report(`${fn.key} (log x)`, (i) => fn.f(x[i]), (i) => y[i], N);
}

// Parametric curves: each coordinate and the value
for (const [name, n] of [["trefoil", 1500], ["chirp", 1200]] as const) {
  const c = CURVES[name];
  const pts = await sampler.curve(name, N, c.t, c.defaults);
  const f32 = Math.fround, dt = f32((c.t[1] - c.t[0]) / (N - 1));
  // t as the shader computes it (chirp returns it: x = t), else w0 + f32(i) dt in f32
  const t = (i: number) => (name === "chirp" ? pts[i * 4] : f32(f32(c.t[0]) + f32(i * dt)));
  const cpu = (i: number): number[] => {
    const s = t(i);
    if (name === "trefoil") {
      const d1 = [Math.cos(s) + 4 * Math.cos(2 * s), -Math.sin(s) + 4 * Math.sin(2 * s), -3 * Math.cos(3 * s)];
      const d2 = [-Math.sin(s) - 8 * Math.sin(2 * s), -Math.cos(s) + 8 * Math.cos(2 * s), 9 * Math.sin(3 * s)];
      const cr = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
      return [Math.sin(s) + 2 * Math.sin(2 * s), Math.cos(s) - 2 * Math.cos(2 * s), -Math.sin(3 * s), Math.hypot(...cr) / Math.hypot(...d1) ** 3];
    }
    const e = Math.exp(-(s * s) / 12);
    return [s, e * Math.cos(1.5 * s * s), e * Math.sin(1.5 * s * s), 1.5 * s * s];
  };
  ["x", "y", "z", "value"].forEach((k, j) => report(`${name}.${k} (gallery n = ${n})`, (i) => cpu(i)[j], (i) => pts[i * 4 + j], N));
}

console.log(bad ? `${bad} above ${TOL}` : `all within ${TOL} (relative, away from zeros)`);
Deno.exit(bad ? 1 : 0);
