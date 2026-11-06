/**
 * Sample functions, PCHIP slopes (as `scipy.interpolate.PchipInterpolator`), not-a-knot cubic
 * spline slopes (as `scipy.interpolate.CubicSpline`), and the exact cubic NURBS of a C1
 * piecewise Hermite cubic.
 *
 * Hermite -> NURBS: on [x_i, x_i+1] (h = x_i+1 - x_i) the cubic with slopes d_i, d_i+1 is the
 * Bézier curve (x_i, y_i), (x_i + h/3, y_i + h d_i/3), (x_i+1 - h/3, y_i+1 - h d_i+1/3),
 * (x_i+1, y_i+1) over u in [x_i, x_i+1]. Its x(u) = u exactly (evenly spaced control abscissae),
 * so the curve is the graph of the interpolant. Knots: the ends with multiplicity 4, interior
 * ones with multiplicity 3 (Bézier pieces joined C0 by the knots, C1 by the slopes), weights 1.
 */

import type { Domain, NumArray, NurbsCurve } from "./nurbs.ts";

// --- Functions -----------------------------------------------------------------------------

// Donoho & Johnstone (1994)
const DJ_T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81];
const BUMPS_H = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2];
const BUMPS_W = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005];
const BLOCKS_H = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2];

/** Σ h_j K((t − t_j) / w_j), K(t) = (1 + |t|⁴)⁻¹ */
function bumps(x: number): number {
  let s = 0;
  for (let j = 0; j < DJ_T.length; j += 1) {
    const u = (x - DJ_T[j]) / BUMPS_W[j];
    s += BUMPS_H[j] / (1 + u ** 4);
  }
  return s;
}

/** Σ h_j (1 + sgn(t − t_j)) / 2 */
function blocks(x: number): number {
  let s = 0;
  for (let j = 0; j < DJ_T.length; j += 1) {
    s += (BLOCKS_H[j] * (1 + Math.sign(x - DJ_T[j]))) / 2;
  }
  return s;
}

const runge = (x: number) => 1 / (1 + 25 * x * x);
const step = (x: number) => (1 + Math.tanh(x / 0.05)) / 2; // smoothed step

export const FUNCTIONS: Record<string, { f: (x: number) => number; domain: [number, number] }> = {
  bumps: { f: bumps, domain: [0, 1] },
  runge: { f: runge, domain: [-1, 1] },
  step: { f: step, domain: [-1, 1] },
  blocks: { f: blocks, domain: [0, 1] },
};

/** `np.linspace(a, b, n)` */
export function linspace(a: number, b: number, n: number): Float64Array<ArrayBuffer> {
  const out = new Float64Array(n);
  const stepSize = (b - a) / (n - 1);
  for (let i = 0; i < n; i += 1) {
    out[i] = i * stepSize + a;
  }
  if (n > 1) {
    out[n - 1] = b;
  }
  return out;
}

// --- Slopes --------------------------------------------------------------------------------

/** Fritsch-Carlson slopes with Fritsch-Butland's weighted harmonic mean and the 3-point
 * one-sided ends, as `PchipInterpolator._find_derivatives` (same operations, same order). */
export function pchipSlopes(x: ArrayLike<number>, y: ArrayLike<number>): Float64Array<ArrayBuffer> {
  const n = x.length;
  const h = Float64Array.from({ length: n - 1 }, (_, i) => x[i + 1] - x[i]);
  const m = Float64Array.from({ length: n - 1 }, (_, i) => (y[i + 1] - y[i]) / h[i]);
  const d = new Float64Array(n);
  if (n === 2) {
    d.fill(m[0]);
    return d;
  }
  for (let k = 1; k < n - 1; k += 1) {
    const [m0, m1] = [m[k - 1], m[k]];
    const flat = Math.sign(m1) !== Math.sign(m0) || m0 === 0 || m1 === 0;
    const w1 = 2 * h[k] + h[k - 1];
    const w2 = h[k] + 2 * h[k - 1];
    d[k] = flat ? 0 : 1 / ((w1 / m0 + w2 / m1) / (w1 + w2));
  }
  d[0] = edge(h[0], h[1], m[0], m[1]);
  d[n - 1] = edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
  return d;
}

function edge(h0: number, h1: number, m0: number, m1: number): number {
  const d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
  if (Math.sign(m0) !== Math.sign(m1) && Math.abs(d) > 3 * Math.abs(m0)) {
    return 3 * m0;
  }
  return Math.sign(d) !== Math.sign(m0) ? 0 : d;
}

/** Slopes at the knots of the not-a-knot cubic spline (`CubicSpline` default, n >= 4): its
 * tridiagonal system in the slopes, solved by Thomas' algorithm. */
export function notAKnotSlopes(x: ArrayLike<number>, y: ArrayLike<number>): Float64Array<ArrayBuffer> {
  const n = x.length;
  const dx = Float64Array.from({ length: n - 1 }, (_, i) => x[i + 1] - x[i]);
  const slope = Float64Array.from({ length: n - 1 }, (_, i) => (y[i + 1] - y[i]) / dx[i]);
  const lower = new Float64Array(n);
  const diag = new Float64Array(n);
  const upper = new Float64Array(n);
  const b = new Float64Array(n);
  for (let i = 1; i < n - 1; i += 1) {
    lower[i] = dx[i];
    diag[i] = 2 * (dx[i - 1] + dx[i]);
    upper[i] = dx[i - 1];
    b[i] = 3 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
  }
  let dd = x[2] - x[0];
  diag[0] = dx[1];
  upper[0] = dd;
  b[0] = ((dx[0] + 2 * dd) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]) / dd;
  dd = x[n - 1] - x[n - 3];
  diag[n - 1] = dx[n - 2];
  lower[n - 1] = dd;
  b[n - 1] = (dx[n - 2] ** 2 * slope[n - 3] + (2 * dd + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / dd;

  for (let i = 1; i < n; i += 1) { // forward elimination
    const w = lower[i] / diag[i - 1];
    diag[i] -= w * upper[i - 1];
    b[i] -= w * b[i - 1];
  }
  const s = new Float64Array(n);
  s[n - 1] = b[n - 1] / diag[n - 1];
  for (let i = n - 2; i >= 0; i -= 1) {
    s[i] = (b[i] - upper[i] * s[i + 1]) / diag[i];
  }
  return s;
}

// --- Hermite cubics -------------------------------------------------------------------------

/** Exact cubic NURBS (dims 2, weights 1) of the C1 piecewise Hermite cubic (x, y, d). */
export function hermiteNurbs(x: ArrayLike<number>, y: ArrayLike<number>, d: ArrayLike<number>): NurbsCurve {
  const n = x.length;
  for (let i = 1; i < n; i += 1) {
    if (!(x[i] > x[i - 1])) { // knots must increase: callers split runs (cf. line.ts `monotoneRuns`)
      throw new Error(`hermiteNurbs: abscissas must increase strictly (x[${i - 1}] = ${x[i - 1]}, x[${i}] = ${x[i]})`);
    }
  }
  const points: number[] = [x[0], y[0]];
  const knots: number[] = [x[0], x[0], x[0], x[0]];
  for (let i = 0; i < n - 1; i += 1) {
    const h = x[i + 1] - x[i];
    points.push(x[i] + h / 3, y[i] + (h * d[i]) / 3, x[i + 1] - h / 3, y[i + 1] - (h * d[i + 1]) / 3, x[i + 1], y[i + 1]);
    knots.push(...(i < n - 2 ? [x[i + 1], x[i + 1], x[i + 1]] : [x[n - 1], x[n - 1], x[n - 1], x[n - 1]]));
  }
  return { degree: 3, dims: 2, points, weights: Array(points.length / 2).fill(1), knots };
}

/** The piecewise Hermite cubic (x, y, d) at t (f64; clamped to the ends' cubics outside). */
export function hermiteAt(x: ArrayLike<number>, y: ArrayLike<number>, d: ArrayLike<number>, t: number): number {
  const n = x.length;
  let lo = 0;
  let hi = n - 2;
  while (lo < hi) { // last interval starting at or before t
    const mid = (lo + hi + 1) >> 1;
    if (x[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  const h = x[lo + 1] - x[lo];
  const s = (t - x[lo]) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * y[lo] + (s3 - 2 * s2 + s) * h * d[lo] + (-2 * s3 + 3 * s2) * y[lo + 1] +
    (s3 - s2) * h * d[lo + 1];
}

// --- Slopes and extents of piecewise cubic Hermite curves (line plots) ----------------------


/**
 * Slopes of the shape-preserving PCHIP interpolant (Fritsch–Carlson), as in Moler,
 * _Numerical Computing with MATLAB_, §3.4.
 * https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/interp.pdf
 * Non-finite points split the curve into independent pieces.
 */
export function getPchipSlopes(xs: NumArray, ys: NumArray): Float64Array<ArrayBuffer> {
  function width(i: number): number {
    return xs[i + 1] - xs[i];
  }

  function secant(i: number): number {
    return i >= 0 && i < xs.length - 1
      ? (ys[i + 1] - ys[i]) / width(i)
      : Number.NaN;
  }

  function getSlope(i: number): number {
    const left = secant(i - 1);
    const right = secant(i);

    if (Number.isFinite(left) && Number.isFinite(right)) {
      if (left * right <= 0 || width(i - 1) * width(i) <= 0) {
        return 0; // local extremum, or abscissas reverse direction
      }
      const w1 = 2 * width(i) + width(i - 1);
      const w2 = width(i) + 2 * width(i - 1);
      return (w1 + w2) / (w1 / left + w2 / right);
    }

    if (Number.isFinite(right)) {
      return getPchipEndSlope(width(i), width(i + 1), right, secant(i + 1));
    }

    if (Number.isFinite(left)) {
      return getPchipEndSlope(width(i - 1), width(i - 2), left, secant(i - 2));
    }

    return 0;
  }

  const slopes = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    slopes[i] = getSlope(i);
  }
  return slopes;
}

// One-sided three-point estimate, clamped so the end interval cannot overshoot
function getPchipEndSlope(
  h1: number,
  h2: number,
  delta1: number,
  delta2: number,
): number {
  if (!Number.isFinite(delta2) || h1 * h2 <= 0) {
    return delta1;
  }

  const slope = ((2 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2);

  if (Math.sign(slope) !== Math.sign(delta1)) {
    return 0;
  }
  if (
    Math.sign(delta1) !== Math.sign(delta2) &&
    Math.abs(slope) > Math.abs(3 * delta1)
  ) {
    return 3 * delta1;
  }
  return slope;
}

/**
 * Slopes of the C² interpolating cubic spline with not-a-knot end conditions, like
 * MATLAB's `spline` (Moler, _Numerical Computing with MATLAB_, §3.5): through three
 * points, the parabola; through two, the line. Non-finite points and repeated abscissas
 * split the curve into independent splines; where abscissas reverse direction, the
 * splines on either side meet with slope 0, as with PCHIP.
 */
export function getSplineSlopes(xs: NumArray, ys: NumArray): Float64Array<ArrayBuffer> {
  const slopes = new Float64Array(xs.length);
  let start = 0;
  let clampStart = false;

  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) {
      start = i + 1;
      clampStart = false;
      continue;
    }

    const width = xs[i + 1] - xs[i];
    const isJoined =
      Number.isFinite(width) && width !== 0 && Number.isFinite(ys[i + 1]);
    const isReversal =
      isJoined &&
      i > start &&
      Math.sign(width) !== Math.sign(xs[i] - xs[i - 1]);

    if (isJoined && !isReversal) {
      continue;
    }

    slopes.set(
      getSplineRunSlopes(xs, ys, [start, i], [clampStart, isReversal]),
      start,
    );
    start = isReversal ? i : i + 1;
    clampStart = isReversal;
  }

  return slopes;
}

// Solves the tridiagonal system for the slopes of points `first` to `last` (Thomas algorithm)
function getSplineRunSlopes(
  xs: NumArray,
  ys: NumArray,
  [first, last]: [number, number],
  [clampFirst, clampLast]: [boolean, boolean], // slope 0 at that end
): Float64Array {
  function h(k: number): number {
    return xs[first + k + 1] - xs[first + k];
  }
  function delta(k: number): number {
    return (ys[first + k + 1] - ys[first + k]) / h(k);
  }

  const n = last - first + 1;
  if (n < 2) {
    return new Float64Array(n); // isolated point: slope 0
  }

  if (n === 2) {
    // Line, or parabola with a flat end
    return Float64Array.of(
      clampFirst ? 0 : clampLast ? 2 * delta(0) : delta(0),
      clampLast ? 0 : clampFirst ? 2 * delta(0) : delta(0),
    );
  }

  if (n === 3 && !clampFirst && !clampLast) {
    // Parabola, from its second divided difference
    const curvature = (delta(1) - delta(0)) / (h(0) + h(1));
    return Float64Array.of(
      delta(0) - curvature * h(0),
      delta(0) + curvature * h(0),
      delta(1) + curvature * h(1),
    );
  }

  // Sub-diagonal, diagonal, super-diagonal and right-hand side, as in Moler's `splinetx`
  const sub = new Float64Array(n);
  const diag = new Float64Array(n);
  const sup = new Float64Array(n);
  const rhs = new Float64Array(n);

  for (let k = 1; k < n - 1; k++) {
    sub[k] = h(k);
    diag[k] = 2 * (h(k - 1) + h(k));
    sup[k] = h(k - 1);
    rhs[k] = 3 * (h(k) * delta(k - 1) + h(k - 1) * delta(k));
  }

  // Not-a-knot: third derivative is continuous at the second and second-to-last points
  const [h0, h1, hm, hl] = [h(0), h(1), h(n - 3), h(n - 2)];
  diag[0] = clampFirst ? 1 : h1;
  sup[0] = clampFirst ? 0 : h0 + h1;
  rhs[0] = clampFirst
    ? 0
    : ((h0 + 2 * (h0 + h1)) * h1 * delta(0) + h0 ** 2 * delta(1)) / (h0 + h1);
  sub[n - 1] = clampLast ? 0 : hm + hl;
  diag[n - 1] = clampLast ? 1 : hm;
  rhs[n - 1] = clampLast
    ? 0
    : (hl ** 2 * delta(n - 3) + (2 * (hm + hl) + hl) * hm * delta(n - 2)) /
      (hm + hl);

  for (let k = 1; k < n; k++) {
    const factor = sub[k] / diag[k - 1];
    diag[k] -= factor * sup[k - 1];
    rhs[k] -= factor * rhs[k - 1];
  }

  const slopes = new Float64Array(n);
  slopes[n - 1] = rhs[n - 1] / diag[n - 1];
  for (let k = n - 2; k >= 0; k--) {
    slopes[k] = (rhs[k] - sup[k] * slopes[k + 1]) / diag[k];
  }
  return slopes;
}

/**
 * Lowest and highest values of the cubic Hermite from `y0` to `y1` with derivatives
 * `d0` and `d1` with respect to `t` in [0, 1], i.e. slopes times the interval's width.
 */
export function getHermiteExtent(
  y0: number,
  y1: number,
  d0: number,
  d1: number,
): Domain {
  const b = 3 * (y1 - y0) - 2 * d0 - d1;
  const c = d0 + d1 - 2 * (y1 - y0);
  const extent: Domain = [Math.min(y0, y1), Math.max(y0, y1)];

  // Derivative `3c t² + 2b t + d0` vanishes at (stable even for `c = 0`)
  const disc = b ** 2 - 3 * c * d0;
  const q = -(b + (b < 0 ? -1 : 1) * Math.sqrt(disc));
  const roots = disc >= 0 ? [q / (3 * c), d0 / q] : [];

  for (const t of roots) {
    if (t > 0 && t < 1) {
      const y = getHermiteY(y0, y1, d0, d1, t);
      extent[0] = Math.min(extent[0], y);
      extent[1] = Math.max(extent[1], y);
    }
  }
  return extent;
}

/* Value at `t` in [0, 1] of the cubic Hermite of `getHermiteExtent` */
export function getHermiteY(
  y0: number,
  y1: number,
  d0: number,
  d1: number,
  t: number,
): number {
  const b = 3 * (y1 - y0) - 2 * d0 - d1;
  const c = d0 + d1 - 2 * (y1 - y0);
  return y0 + t * (d0 + t * (b + t * c));
}
