/**
 * Our f64 curve functions (lib/generators.ts, lib/expr.ts: Boost.Math's special functions behind
 * them) against direct Boost calls and published values: each spec's samples (`sample`, the path
 * line.ts draws, vectorized where it can be) against its reference at 10 001 points of the domain,
 * the expression functions one by one, the parametric curves' values against finite differences
 * of their points. Relative error where |f| >= 5% of max |f| (away from zeros).
 *
 *   deno run -A tools/check_generators.ts      # exits 1 if an error is above its tolerance
 */

import { bm, normalQuantile, tQuantile } from "../lib/boost.ts";
import { CURVES, curvePoints, type FnSpec, resolveFn } from "../lib/generators.ts";
import { compileExpr } from "../lib/expr.ts";
import { FUNCTIONS } from "../lib/pchip.ts";

const N = 10_001;
let bad = 0;

function report(name: string, ref: (x: number) => number, got: ArrayLike<number>, xs: Float64Array, tol = 1e-13): void {
  const r = Array.from(xs, ref);
  const scale = Math.max(...r.map(Math.abs).filter(Number.isFinite));
  let rel = 0;
  r.forEach((v, i) => {
    if (Number.isNaN(v) !== Number.isNaN(got[i])) rel = Infinity;
    else if (Math.abs(v) >= 0.05 * scale && Number.isFinite(v)) rel = Math.max(rel, Math.abs(got[i] - v) / Math.abs(v));
  });
  if (!(rel <= tol)) bad += 1;
  console.log(`${name.padEnd(48)} max rel ${rel.toExponential(2)}${rel <= tol ? "" : `  ABOVE ${tol}`}`);
}
const grid = ([a, b]: [number, number], n = N) => Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

// 1. Specs, as line.ts samples them, against their formulas written here with direct Boost calls
const lam = 4.3;
const specs: [FnSpec, (x: number) => number][] = [
  ...["bumps", "blocks", "runge", "step"].map((n): [FnSpec, (x: number) => number] => [n, FUNCTIONS[n].f]),
  ...[0, 1, 2, 5, 20].map((n): [FnSpec, (x: number) => number] => [{ name: "besselJ", order: n, domain: [0, 60] }, (x) => bm.cyl_bessel_j(n, x)]),
  [{ name: "besselJ", order: 2.5, domain: [0, 30] }, (x) => bm.cyl_bessel_j(2.5, x)],
  [{ name: "sinc", k: 1.5 }, (x) => (x === 0 ? 1 : Math.sin(1.5 * x) / (1.5 * x))],
  [{ name: "poisson", lambda: lam, scale: 2000, domain: [0, 16] }, (x) => 2000 * bm.gamma_p_derivative(x + 1, lam)], // λ^x e^−λ / Γ(x+1)
  [{ name: "dampedCos", tau: 14, omega: 1.3 }, (x) => Math.exp(-x / 14) * Math.cos(1.3 * x)],
  [{ expr: "a*exp(x*log(lam) - lam - lgamma(x+1))", params: { a: 2000, lam }, domain: [0, 16] }, (x) => 2000 * bm.gamma_p_derivative(x + 1, lam)],
  [{ expr: "exp(-(x-mu)^2/(2*s^2))/(s*sqrt(2*pi))", params: { mu: 1, s: 0.7 }, domain: [-3, 5] }, (x) => pdf(new bm.normal_distribution(1, 0.7), x)],
  [{ expr: "0.5*(1 + erf((x - mu)/(s*sqrt(2))))", params: { mu: 1, s: 0.7 }, domain: [-1, 5] }, (x) => cdf(new bm.normal_distribution(1, 0.7), x)],
  [{ expr: "sinh(x)*cos(3*x)/(1 + x**2)", domain: [-4, 4] }, (x) => Math.sinh(x) * Math.cos(3 * x) / (1 + x * x)],
];
function pdf(d: InstanceType<typeof bm.normal_distribution>, x: number) { try { return d.pdf(x); } finally { d.delete(); } }
function cdf(d: InstanceType<typeof bm.normal_distribution>, x: number) { try { return d.cdf(x); } finally { d.delete(); } }
for (const [spec, ref] of specs) {
  const fn = resolveFn(spec), xs = grid(fn.domain);
  report(fn.key, ref, fn.sample(xs), xs, 1e-12);
}

// 2. Each expression function against its Boost (or Math) call
const one: [string, [number, number], (x: number) => number][] = [
  ["erf(x)", [-4, 4], bm.erf], ["erfc(x)", [-2, 8], bm.erfc], ["lgamma(x)", [-5.5, 30], bm.lgamma],
  ["tgamma(x)", [-3.5, 6], (x) => bm.tgamma(x)], ["digamma(x)", [-2.5, 20], bm.digamma], ["zeta(x)", [-10, 10], (x) => bm.zeta(x)],
  ["expint(x)", [-5, 5], (x) => bm.expint(x)], ["besselJ(3, x)", [-30, 30], (x) => bm.cyl_bessel_j(3, x)],
  ["besselY(1, x)", [0.1, 30], (x) => bm.cyl_neumann(1, x)], ["besselI(0, x)", [-5, 5], (x) => bm.cyl_bessel_i(0, x)],
  ["besselK(0.5, x)", [0.05, 10], (x) => bm.cyl_bessel_k(0.5, x)], ["airyAi(x)", [-15, 5], bm.airy_ai],
  ["lambertW(x)", [-1 / Math.E, 10], bm.lambert_w0], ["ellipK(x)", [-0.999, 0.999], (k) => bm.ellint_1(k)], ["sinc(x)", [-20, 20], bm.sinc_pi],
];
for (const [src, domain, ref] of one) {
  const fn = resolveFn({ expr: src, domain }), xs = grid(domain, 2001);
  report(`${src} = Boost's`, ref, fn.sample(xs), xs, 0);
}

// 3. Published values (A&S, DLMF; quantiles: R's qnorm, qt)
const at = (src: string, x: number) => compileExpr(src).f(x, []);
const published: [string, number, number][] = [
  ["J₀(1)", at("besselJ(0, x)", 1), 0.7651976865579666], ["Y₀(1)", at("besselY(0, x)", 1), 0.08825696421567696],
  ["I₀(1)", at("besselI(0, x)", 1), 1.266065877752008], ["K₀(1)", at("besselK(0, x)", 1), 0.4210244382407083],
  ["Ai(0)", at("airyAi(x)", 0), 0.3550280538878172], ["Γ(½)", at("tgamma(x)", 0.5), Math.sqrt(Math.PI)],
  ["ln |Γ(−5/2)|", at("lgamma(x)", -2.5), Math.log(8 * Math.sqrt(Math.PI) / 15)], ["ψ(1) = −γ", at("digamma(x)", 1), -0.5772156649015329],
  ["ζ(2)", at("zeta(x)", 2), Math.PI ** 2 / 6], ["Ei(1)", at("expint(x)", 1), 1.895117816355937], ["W₀(1) = Ω", at("lambertW(x)", 1), 0.5671432904097838],
  ["K(1/√2)", at("ellipK(x)", Math.SQRT1_2), 1.854074677301372], ["erf(1)", at("erf(x)", 1), 0.8427007929497149], ["sinc(π/2)", at("sinc(x)", Math.PI / 2), 2 / Math.PI],
  ["Φ⁻¹(0.975)", normalQuantile(0.975), 1.959963984540054], ["Φ⁻¹(1e-10)", normalQuantile(1e-10), -6.361340902404056],
  ["t₁₀(0.975)", tQuantile(0.975, 10), 2.228138851986274], ["t₁(0.975)", tQuantile(0.975, 1), 12.70620473617471],
];
for (const [name, got, want] of published) {
  const rel = Math.abs(got - want) / Math.abs(want);
  if (!(rel <= 1e-14)) bad += 1;
  console.log(`${name.padEnd(48)} rel ${rel.toExponential(2)}${rel <= 1e-14 ? "" : "  ABOVE 1e-14"}`);
}

// 4. Parametric curves: trefoil's curvature against central differences of its points (h = 1e-3,
// error ~h²), chirp's |ℜ + iℑ| = exp(−t²/width) and phase
{
  const h = 1e-3, ts = grid(CURVES.trefoil.t, 1500);
  const p = (t: number) => curvePoints("trefoil", 1, [t, t + 1], []);
  const kappa = (t: number) => {
    const [a, b, c] = [p(t - h), p(t), p(t + h)];
    const d1 = [0, 1, 2].map((j) => (c[j] - a[j]) / (2 * h)), d2 = [0, 1, 2].map((j) => (c[j] - 2 * b[j] + a[j]) / (h * h));
    return Math.hypot(d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]) / Math.hypot(...d1) ** 3;
  };
  report("trefoil κ vs differences", kappa, Array.from(ts, (t) => p(t)[3]), ts, 1e-5);
  const [rate, width] = CURVES.chirp.defaults, pts = curvePoints("chirp", 1200, CURVES.chirp.t, [rate, width]), tc = grid(CURVES.chirp.t, 1200);
  report("chirp |ℜ + iℑ|", (t) => Math.exp(-t * t / width), Array.from(tc, (_, i) => Math.hypot(pts[4 * i + 1], pts[4 * i + 2])), tc);
  report("chirp phase", (t) => rate * t * t, Array.from(tc, (_, i) => pts[4 * i + 3]), tc);
}

console.log(bad ? `${bad} failed` : "all passed");
Deno.exit(bad ? 1 : 0);
