/**
 * Summaries of raw samples for line.ts's stat layer (`SeriesSpec.stat`), in f64 on the CPU: the
 * ECDF and its DKW band, normal Q–Q points, box-plot numbers (quartiles: type 7, R's default),
 * histogram bins, the least-squares line with its t bands, and for lib/marks.ts's transforms the
 * Gaussian KDE (Silverman's bandwidth) and rolling reductions. Each returns plain arrays; line.ts
 * turns them into ordinary series (steps, markers, fills, intervals, boxes, bars, expr fns). The
 * normal and Student-t quantiles are Boost.Math's (lib/boost.ts). Quartiles stay type 7 here:
 * Boost's interquartile_range takes medians of the halves, another definition.
 */

import { normalQuantile, normalQuantiles, tQuantile } from "./boost.ts";

/** The finite samples, ascending. */
export const sortedFinite = (v: ArrayLike<number>) => Float64Array.from(Array.from(v).filter(Number.isFinite)).sort();

/** [min, max] of the finite values ([Infinity, -Infinity] if none): a loop, not a spread (which
 * overflows the stack on large columns). */
export function finiteRange(v: Iterable<number>): [number, number] {
  let [lo, hi] = [Infinity, -Infinity];
  for (const u of v) if (Number.isFinite(u)) [lo, hi] = [Math.min(lo, u), Math.max(hi, u)];
  return [lo, hi];
}

/** The smallest gap between distinct finite values (1 for fewer than two). */
export function closestSpacing(x: ArrayLike<number>): number {
  const xs = Array.from(x).filter(Number.isFinite).sort((a, b) => a - b);
  let d = Infinity;
  for (let i = 1; i < xs.length; i += 1) if (xs[i] > xs[i - 1]) d = Math.min(d, xs[i] - xs[i - 1]);
  return Number.isFinite(d) ? d : 1;
}

/** Quantile of sorted samples, type 7: linear between order statistics at h = (n − 1) p. */
export function quantile(s: Float64Array, p: number): number {
  const h = (s.length - 1) * p, i = Math.floor(h);
  return i + 1 < s.length ? s[i] + (h - i) * (s[i + 1] - s[i]) : s[s.length - 1];
}

/** ECDF F(x) = #{samples ≤ x}/n at the distinct values (ties: one jump), and the DKW half-width
 * ε = √(ln(2/α)/(2n)): P(sup |F − F_n| > ε) ≤ α. */
export function ecdf(s: Float64Array, alpha?: number): { x: number[]; y: number[]; eps: number } {
  const x: number[] = [], y: number[] = [];
  for (let i = 0; i < s.length; i += 1) {
    if (i + 1 < s.length && s[i + 1] === s[i]) continue;
    x.push(s[i]);
    y.push((i + 1) / s.length);
  }
  return { x, y, eps: alpha ? Math.sqrt(Math.log(2 / alpha) / (2 * s.length)) : 0 };
}

/** Normal Q–Q: Φ⁻¹((i − ½)/n) against the order statistics, and the line through the quartiles
 * (q₁ at z₁ = Φ⁻¹(¼), q₃ at z₃). */
export function qq(s: Float64Array): { x: number[]; y: number[]; slope: number; intercept: number } {
  const n = s.length;
  const x = Array.from(normalQuantiles(Array.from(s, (_, i) => (i + 0.5) / n)));
  const z3 = normalQuantile(0.75), q1 = quantile(s, 0.25), q3 = quantile(s, 0.75);
  const slope = (q3 - q1) / (2 * z3);
  return { x, y: Array.from(s), slope, intercept: q1 + z3 * slope };
}

/** Box-plot numbers: quartiles (type 7), whiskers to the most extreme samples within 1.5 IQR of
 * the box, the samples beyond them. */
export function boxStats(s: Float64Array) {
  const q1 = quantile(s, 0.25), median = quantile(s, 0.5), q3 = quantile(s, 0.75);
  const [a, b] = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)];
  const inside = s.filter((v) => v >= a && v <= b);
  return { n: s.length, q1, median, q3, lo: inside[0] ?? q1, hi: inside[inside.length - 1] ?? q3, outliers: Array.from(s.filter((v) => v < a || v > b)) };
}

/** Histogram: `bins` equal bins over [min, max], or bins `width` wide on multiples of it;
 * default width Freedman–Diaconis 2 IQR n^(−1/3) (Sturges' count if the IQR is 0). The last edge
 * is inclusive. */
export function histogram(s: Float64Array, bins?: number, width?: number): { x: number[]; counts: number[]; width: number } {
  const [lo, hi] = [s[0], s[s.length - 1]];
  if (!(hi > lo)) return { x: [lo], counts: [s.length], width: 1 };
  let w = width && width > 0 ? width : bins && bins >= 1 ? (hi - lo) / Math.round(bins) : 2 * (quantile(s, 0.75) - quantile(s, 0.25)) / Math.cbrt(s.length);
  if (!(w > 0)) w = (hi - lo) / (Math.ceil(Math.log2(s.length)) + 1);
  w = Math.max(w, (hi - lo) / 1000);
  const start = width && width > 0 ? Math.floor(lo / w) * w : lo;
  const m = Math.max(1, Math.ceil((hi - start) / w - 1e-9));
  const counts = new Array(m).fill(0);
  for (const v of s) counts[Math.min(Math.floor((v - start) / w), m - 1)] += 1;
  return { x: counts.map((_, i) => start + (i + 0.5) * w), counts, width: w };
}

/** Least squares ŷ = a + b x, with the residual s = √(SSE/(n − 2)), r², and the half-width
 * t_{n−2, 1−α/2} s √(k + 1/n + (x − x̄)²/Sxx) of the band (k = 0: mean response, 1: prediction). */
export function linreg(x: ArrayLike<number>, y: ArrayLike<number>, alpha = 0.05) {
  const idx = Array.from({ length: Math.min(x.length, y.length) }, (_, i) => i).filter((i) => Number.isFinite(x[i]) && Number.isFinite(y[i]));
  const n = idx.length;
  if (n < 3) throw new Error(`stat linreg: ${n} finite (x, y) points; need 3`);
  const mx = idx.reduce((t, i) => t + x[i], 0) / n, my = idx.reduce((t, i) => t + y[i], 0) / n;
  let sxx = 0, sxy = 0, syy = 0, sse = 0;
  for (const i of idx) {
    sxx += (x[i] - mx) ** 2;
    sxy += (x[i] - mx) * (y[i] - my);
    syy += (y[i] - my) ** 2;
  }
  if (!(sxx > 0)) throw new Error("stat linreg: all x equal");
  const b = sxy / sxx, a = my - b * mx;
  for (const i of idx) sse += (y[i] - my - b * (x[i] - mx)) ** 2; // the residuals themselves: Syy − b Sxy cancels
  return { n, a, b, mx, sxx, s: Math.sqrt(sse / (n - 2)), r2: syy > 0 ? 1 - sse / syy : 1, t: tQuantile(1 - alpha / 2, n - 2), lo: x[idx.reduce((m, i) => (x[i] < x[m] ? i : m), idx[0])], hi: x[idx.reduce((m, i) => (x[i] > x[m] ? i : m), idx[0])] };
}

/** Silverman's rule of thumb h = 0.9 min(σ, IQR/1.34) n^(−1/5) of sorted samples (σ alone if
 * the IQR is 0; 1 if both are). */
export function silverman(s: Float64Array): number {
  const n = s.length;
  const mean = s.reduce((a, v) => a + v, 0) / n;
  const sd = Math.sqrt(s.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(n - 1, 1));
  const iqr = (quantile(s, 0.75) - quantile(s, 0.25)) / 1.34;
  const spread = iqr > 0 ? Math.min(sd, iqr) : sd;
  return spread > 0 ? 0.9 * spread * n ** -0.2 : 1;
}

/** Gaussian kernel density of sorted samples at `m` even points over [min − cut h, max + cut h]
 * (h: `bandwidth`, default Silverman's); samples beyond 6 h of a point are skipped. */
export function kde(s: Float64Array, bandwidth?: number, m = 200, cut = 3): { x: number[]; y: number[]; bandwidth: number } {
  const h = bandwidth && bandwidth > 0 ? bandwidth : silverman(s);
  const [a, b] = [s[0] - cut * h, s[s.length - 1] + cut * h];
  const x: number[] = [], y: number[] = [];
  const norm = 1 / (s.length * h * Math.sqrt(2 * Math.PI));
  let lo = 0;
  for (let k = 0; k < m; k += 1) {
    const t = a + ((b - a) * k) / (m - 1);
    while (lo < s.length && s[lo] < t - 6 * h) lo += 1;
    let sum = 0;
    for (let i = lo; i < s.length && s[i] <= t + 6 * h; i += 1) sum += Math.exp(-0.5 * ((t - s[i]) / h) ** 2);
    x.push(t);
    y.push(sum * norm);
  }
  return { x, y, bandwidth: h };
}

/** A rolling reduction over `k` points centred on each (fewer at the ends; NaN skipped). */
export function rolling(y: ArrayLike<number>, k: number, reduce: "mean" | "median" | "min" | "max" | "sum" = "mean"): number[] {
  const half = Math.max(0, Math.floor((Math.max(1, Math.round(k)) - 1) / 2)), extra = Math.max(1, Math.round(k)) % 2 ? 0 : 1;
  return Array.from({ length: y.length }, (_, i) => {
    const w = Float64Array.from(Array.from({ length: Math.min(y.length, i + half + extra + 1) - Math.max(0, i - half) }, (_, j) => y[Math.max(0, i - half) + j])).filter(Number.isFinite);
    if (!w.length) return NaN;
    if (reduce === "median") return quantile(w.sort(), 0.5);
    if (reduce === "min" || reduce === "max") return finiteRange(w)[reduce === "min" ? 0 : 1];
    const sum = w.reduce((a, v) => a + v, 0);
    return reduce === "sum" ? sum : sum / w.length;
  });
}
