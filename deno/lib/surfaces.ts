/** Sample surfaces (torus, sphere, `wave`, `wave-test`, `dini`, `breather`), as `NurbsSurface`s. Data only. */

import { findSpan, getBasisFunctions, type NurbsSurface } from "./nurbs.ts";

const S = Math.sqrt(0.5);
const CIRCLE = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]];
const CIRCLE_WEIGHTS = CIRCLE.map((_, i) => (i % 2 ? S : 1));
const CIRCLE_KNOTS = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1];

/** Surface of revolution: profile (r, z) revolved around z by a rational quadratic circle (u
 * around, v along). */
function revolve(profile: number[][], weights: number[], knots: number[]): NurbsSurface {
  return {
    degrees: [2, 2],
    shape: [CIRCLE.length, profile.length],
    points: CIRCLE.flatMap(([cx, cy]) => profile.flatMap(([r, z]) => [r * cx, r * cy, z])),
    weights: CIRCLE_WEIGHTS.flatMap((wu) => weights.map((wv) => wu * wv)),
    knots: [CIRCLE_KNOTS, knots],
  };
}

/** `TORUS` (radii 2 and 0.5, exact, rational), with a field winding around it: cos(3θ) + sin(φ). */
function torus(): NurbsSurface {
  const s = revolve(CIRCLE.map(([x, y]) => [2 + x / 2, y / 2]), CIRCLE_WEIGHTS, CIRCLE_KNOTS);
  s.fields = [CIRCLE.flatMap(([cx, cy]) => CIRCLE.map(([px, py]) => Math.cos(3 * Math.atan2(cy, cx)) + Math.sin(Math.atan2(py, px))))];
  return s;
}

/** `SPHERE` (poles: degenerate normals, nudged), with field x y. */
function sphere(): NurbsSurface {
  const s = revolve([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1]], [1, S, 1, S, 1], [0, 0, 0, 0.5, 0.5, 1, 1, 1]);
  const p = s.points as number[];
  s.fields = [Array.from({ length: p.length / 3 }, (_, i) => p[i * 3] * p[i * 3 + 1])];
  return s;
}

/** Test wave: bicubic B-spline with non-uniform knots. */
function waveTest(): NurbsSurface {
  const pts: number[] = [];
  for (let i = 0; i < 30; i += 1) {
    const [r, c] = [Math.floor(i / 6), i % 6];
    pts.push(r, c * 0.8, Math.sin(r + c) * (1 + c / 5));
  }
  return {
    degrees: [3, 3],
    shape: [5, 6],
    points: pts,
    weights: Array(30).fill(1),
    knots: [[0, 0, 0, 0, 0.3, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1.5, 4, 4, 4, 4]],
    fields: [pts.filter((_, i) => i % 3 === 0)],
  };
}

/** Rational bicubic wavy patch on [-1, 1]² (10 × 10, non-uniform knots, weights 0.5-2), with a
 * field unrelated to height: a Gaussian bump plus a ripple along x. */
function wave(): NurbsSurface {
  const n = 10;
  const xs = Array.from({ length: n }, (_, i) => (i === n - 1 ? 1 : i * (2 / (n - 1)) - 1)); // np.linspace
  const pts: number[] = [];
  const wts: number[] = [];
  const fld: number[] = [];
  for (const x of xs) {
    for (const y of xs) {
      pts.push(x, y, 0.28 * Math.sin(3.2 * x + 0.6) * Math.cos(2.4 * y) + 0.12 * x * y);
      wts.push(1.0 + 0.9 * Math.exp(-8 * ((x - 0.3) ** 2 + (y + 0.35) ** 2)) - 0.4 * Number(Math.abs(x) > 0.8));
      fld.push(Math.exp(-3 * ((x + 0.4) ** 2 + (y - 0.3) ** 2)) + 0.25 * Math.sin(6 * x));
    }
  }
  const knots = [0, 0, 0, 0, 0.12, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1, 1, 1]; // non-uniform: 7 spans
  return { degrees: [3, 3], shape: [n, n], points: pts, weights: wts, knots: [knots, knots], fields: [fld] };
}

/** Global cubic interpolation along one direction (Piegl & Tiller A9.1): `n` samples at uniform
 * parameters, knots by averaging; the collocation matrix is banded and totally positive, so it is
 * factored once without pivoting and `solve` turns samples into control values in place. */
function cubicInterpolator(n: number) {
  const p = 3, band = 2 * p;
  const t = Array.from({ length: n }, (_, i) => i / (n - 1));
  const inner = Array.from({ length: n - p - 1 }, (_, j) => (t[j + 1] + t[j + 2] + t[j + 3]) / 3);
  const knots = [0, 0, 0, 0, ...inner, 1, 1, 1, 1];
  const kv = { degree: p, knots };
  const A = Array.from({ length: n }, () => new Float64Array(n));
  t.forEach((u, i) => {
    const k = findSpan(kv, u);
    getBasisFunctions(kv, k, u).forEach((b, r) => (A[i][k - p + r] = b));
  });
  for (let c = 0; c < n; c += 1) { // LU, in place (L below the diagonal, unit diagonal implied)
    for (let r = c + 1; r < Math.min(n, c + band + 1); r += 1) {
      const f = A[r][c] / A[c][c];
      if (!f) continue;
      A[r][c] = f;
      for (let k = c + 1; k < Math.min(n, c + band + 1); k += 1) A[r][k] -= f * A[c][k];
    }
  }
  const solve = (b: Float64Array) => {
    for (let r = 0; r < n; r += 1) for (let c = Math.max(0, r - band); c < r; c += 1) b[r] -= A[r][c] * b[c];
    for (let r = n - 1; r >= 0; r -= 1) {
      for (let c = r + 1; c < Math.min(n, r + band + 1); c += 1) b[r] -= A[r][c] * b[c];
      b[r] /= A[r][r];
    }
  };
  return { knots, solve };
}

/** A bicubic B-spline through `nu × nv` samples of `f(s, t)` (s, t in [0, 1]) -> [x, y, z, field]:
 * it passes exactly through every sample, smooth (C²) in between. */
function interpolated(nu: number, nv: number, f: (s: number, t: number) => number[]): NurbsSurface {
  const U = cubicInterpolator(nu), V = cubicInterpolator(nv);
  const ch = Array.from({ length: 4 }, () => new Float64Array(nu * nv)); // x, y, z, field; [i * nv + j]
  for (let i = 0; i < nu; i += 1) {
    for (let j = 0; j < nv; j += 1) f(i / (nu - 1), j / (nv - 1)).forEach((x, c) => (ch[c][i * nv + j] = x));
  }
  const row = new Float64Array(nv), col = new Float64Array(nu);
  for (const a of ch) {
    for (let i = 0; i < nu; i += 1) { row.set(a.subarray(i * nv, (i + 1) * nv)); V.solve(row); a.set(row, i * nv) }
    for (let j = 0; j < nv; j += 1) {
      for (let i = 0; i < nu; i += 1) col[i] = a[i * nv + j];
      U.solve(col);
      for (let i = 0; i < nu; i += 1) a[i * nv + j] = col[i];
    }
  }
  const points = Array.from({ length: nu * nv * 3 }, (_, k) => ch[k % 3][Math.floor(k / 3)]);
  return { degrees: [3, 3], shape: [nu, nv], points, weights: Array(nu * nv).fill(1), knots: [U.knots, V.knots], fields: [Array.from(ch[3])] };
}

/** Dini's surface (a = 1, b = 0.2: two turns of a twisted pseudosphere, curvature -1/(a² + b²)
 * everywhere), with field the height along its axis. */
function dini(): NurbsSurface {
  const a = 1, b = 0.2;
  return interpolated(96, 48, (s, t) => {
    const u = 4 * Math.PI * s, v = 0.06 + (2 - 0.06) * t;
    const z = a * (Math.cos(v) + Math.log(Math.tan(v / 2))) + b * u;
    return [a * Math.cos(u) * Math.sin(v), a * Math.sin(u) * Math.sin(v), z, z];
  });
}

/** Breather surface of the sine-Gordon equation (a = 0.4; pseudospherical: curvature -1, cusps
 * along its edges), with field the breather angle 4 atan((a/w) sin(w v) / cosh(a u)) it is built from. */
function breather(): NurbsSurface {
  const a = 0.4, w = Math.sqrt(1 - a * a);
  return interpolated(140, 400, (s, t) => {
    const u = -14 + 28 * s, v = -37.4 + 74.8 * t;
    const ch = Math.cosh(a * u), d = a * ((w * ch) ** 2 + (a * Math.sin(w * v)) ** 2);
    return [
      -u + (2 * w * w * ch * Math.sinh(a * u)) / d,
      (2 * w * ch * (-w * Math.cos(v) * Math.cos(w * v) - Math.sin(v) * Math.sin(w * v))) / d,
      (2 * w * ch * (-w * Math.sin(v) * Math.cos(w * v) + Math.cos(v) * Math.sin(w * v))) / d,
      4 * Math.atan(((a / w) * Math.sin(w * v)) / ch),
    ];
  });
}

export const SURFACES: Record<string, () => NurbsSurface> = {
  wave,
  torus,
  sphere,
  "wave-test": waveTest,
  dini,
  breather,
};
