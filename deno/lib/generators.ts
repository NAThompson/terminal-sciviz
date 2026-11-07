/**
 * Generator specs: pure functions of x named in a control instead of their samples, e.g.
 * `{name: "besselJ", order: 3}`, `"bumps"` or `{expr, params}` (an expression of x: lib/expr.ts),
 * and curve.ts's parametric curves of t. All are f64 functions on the CPU, the special functions
 * Boost.Math's (lib/boost.ts); line.ts samples them over the visible window and uploads the
 * samples for lib/chords.ts to draw. tools/check_generators.ts checks them against direct Boost
 * calls and published values.
 */

import { bm } from "./boost.ts";
import { FUNCTIONS } from "./pchip.ts";
import { compileExpr } from "./expr.ts";

export type FnSpec =
  | string
  | ({ name: string; domain?: [number, number] } & Record<string, unknown>)
  | { expr: string; params?: Record<string, number>; domain?: [number, number] };

interface Generator {
  domain: [number, number];
  args: string[]; // names of the spec's numbers
  defaults: number[];
  f: (x: number, a: number[]) => number;
  many?: (xs: Float64Array, a: number[]) => Float64Array; // f over an array (a Boost `_array` twin)
}

const pure = (name: string) => FUNCTIONS[name].f;

export const GENERATORS: Record<string, Generator> = {
  bumps: { domain: FUNCTIONS.bumps.domain, args: [], defaults: [], f: pure("bumps") },
  blocks: { domain: FUNCTIONS.blocks.domain, args: [], defaults: [], f: pure("blocks") },
  runge: { domain: FUNCTIONS.runge.domain, args: [], defaults: [], f: pure("runge") },
  step: { domain: FUNCTIONS.step.domain, args: [], defaults: [], f: pure("step") },
  /** J_ν(x) (x < 0: integer ν only) */
  besselJ: { domain: [0, 30], args: ["order"], defaults: [0], f: (x, [n]) => bm.cyl_bessel_j(n, x), many: (xs, [n]) => bm.cyl_bessel_j_array(n, xs) },
  /** sin(k x) / (k x) */
  sinc: { domain: [-2 * Math.PI, 2 * Math.PI], args: ["k"], defaults: [Math.PI], f: (x, [k]) => bm.sinc_pi(k * x) },
  /** |H(f)| in dB of a second-order low-pass, r = f / f0: 20 log10(1 / |1 − r² + 2iζr|) */
  bode: {
    domain: [1, 1e5], args: ["f0", "zeta"], defaults: [2e4 / (2 * Math.PI), 0.1],
    f: (x, [f0, z]) => {
      const r = x / f0;
      return 20 * Math.log10(1 / Math.hypot(1 - r * r, 2 * z * r));
    },
  },
  /** tight-binding band b: 2 cos k + 4b − 0.4 b cos 2k */
  dispersion: { domain: [-Math.PI, Math.PI], args: ["band"], defaults: [0], f: (k, [b]) => 2 * Math.cos(k) + 4 * b - 0.4 * b * Math.cos(2 * k) },
  /** exp(−x/τ) cos(ωx) */
  dampedCos: { domain: [0, 30], args: ["tau", "omega"], defaults: [1, 0], f: (x, [tau, w]) => Math.exp(-x / tau) * Math.cos(w * x) },
  /** scale · λˣ e^(−λ) / Γ(x + 1): the Poisson pmf (times a count), smooth between the integers */
  // λ = 0: all the mass at 0 (λ⁰ = 1, not exp(0 · log 0)); λ < 0: no distribution (NaN, a gap)
  poisson: {
    domain: [0, 20], args: ["lambda", "scale"], defaults: [4, 1],
    f: (x, [l, n]) => (l > 0 ? n * Math.exp(x * Math.log(l) - l - bm.lgamma(x + 1)) : l === 0 ? (x === 0 ? n : 0) : NaN),
  },
};

export interface ResolvedFn {
  key: string;
  domain: [number, number];
  f: (x: number) => number;
  sample: (xs: Float64Array) => Float64Array; // f at each x
}

/** `{expr, params?, domain?}`: an expression of x and the params (lib/expr.ts); domain [-5, 5]
 * by default. */
function resolveExpr(s: { expr: unknown; params?: unknown; domain?: unknown }): ResolvedFn {
  const given = (s.params ?? {}) as Record<string, unknown>;
  if (typeof given !== "object" || Array.isArray(given)) throw new Error("fn expr: params must be an object of numbers, e.g. {a: 2, mu: 1}");
  const c = compileExpr(s.expr as string, Object.keys(given));
  const params = c.names.map((n) => Number(given[n]));
  c.names.forEach((n, i) => {
    if (!Number.isFinite(params[i])) throw new Error(`fn expr: param ${n} must be a finite number, got ${JSON.stringify(given[n])}`);
  });
  const domain = ((s.domain ?? [-5, 5]) as number[]).map(Number) as [number, number];
  if (!(domain.length === 2 && domain[1] > domain[0])) throw new Error("fn expr: domain must be [x0, x1], x0 < x1");
  const f = (x: number) => c.f(x, params);
  return { key: JSON.stringify(["expr", s.expr, given, domain]), domain, f, sample: (xs) => xs.map(f) };
}

export function resolveFn(spec: FnSpec): ResolvedFn {
  const s = typeof spec === "string" ? { name: spec } : spec;
  if ("expr" in s) return resolveExpr(s as { expr: unknown });
  const g = Object.hasOwn(GENERATORS, s.name) ? GENERATORS[s.name] : undefined;
  if (!g) throw new Error(`fn must be one of ${Object.keys(GENERATORS)} (or {name, ...}, or {expr: "exp(-x^2/2)", params?, domain?})`);
  const args = g.args.map((a, i) => (s[a] === undefined ? g.defaults[i] : Number(s[a])));
  if (!args.every(Number.isFinite)) throw new Error(`fn ${s.name}: ${g.args.join(", ")} must be numbers`);
  if (s.name === "poisson" && args[0] < 0) throw new Error(`fn poisson: lambda must be >= 0, got ${args[0]}`);
  const domain = (s.domain ?? g.domain).map(Number) as [number, number];
  if (!(domain[1] > domain[0])) throw new Error(`fn ${s.name}: domain must increase`);
  const f = (x: number) => g.f(x, args);
  return { key: JSON.stringify([s.name, args, domain]), domain, f, sample: (xs) => (g.many ? g.many(xs, args) : xs.map(f)) };
}

/** Parametric curves of t: `{name: "trefoil", n, t: [t0, t1]}` (point, curvature κ) or
 * `{name: "chirp", n, t, rate, width}` ((t, ℜ, ℑ) of exp(i rate t²) exp(−t²/width), phase). */
export const CURVES: Record<string, { t: [number, number]; args: string[]; defaults: number[]; f: (t: number, a: number[]) => number[] }> = {
  trefoil: { // (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), κ = |r′ × r″| / |r′|³
    t: [0, 2 * Math.PI], args: [], defaults: [],
    f: (t) => {
      const [s1, c1, s2, c2, s3, c3] = [Math.sin(t), Math.cos(t), Math.sin(2 * t), Math.cos(2 * t), Math.sin(3 * t), Math.cos(3 * t)];
      const d1 = [c1 + 4 * c2, -s1 + 4 * s2, -3 * c3], d2 = [-s1 - 8 * s2, -c1 + 8 * c2, 9 * s3];
      const cross = Math.hypot(d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]);
      return [s1 + 2 * s2, c1 - 2 * c2, -s3, cross / Math.hypot(...d1) ** 3];
    },
  },
  chirp: {
    t: [-6, 6], args: ["rate", "width"], defaults: [1.5, 12],
    f: (t, [a, b]) => {
      const e = Math.exp(-t * t / b), ph = a * t * t;
      return [t, e * Math.cos(ph), e * Math.sin(ph), ph];
    },
  },
};

/** Points `x, y, z, value` of a parametric curve at `n` even values of t over [t0, t1]. */
export function curvePoints(name: string, n: number, t: [number, number], args: number[]): Float64Array {
  const out = new Float64Array(4 * n);
  for (let i = 0; i < n; i += 1) out.set(CURVES[name].f(t[0] + ((t[1] - t[0]) * i) / Math.max(n - 1, 1), args), 4 * i);
  return out;
}
