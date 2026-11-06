/** Sample surfaces (torus, sphere, `wave`, `wave-test`), as `NurbsSurface`s. Data only. */

import type { NurbsSurface } from "./nurbs.ts";

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

export const SURFACES: Record<string, () => NurbsSurface> = {
  wave,
  torus,
  sphere,
  "wave-test": waveTest,
};
