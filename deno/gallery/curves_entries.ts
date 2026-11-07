/**
 * The "curves" section: 2D and 3D NURBS curves (circle, ellipse, heart, rose, spirograph, a
 * kinked B-spline, a helix, trefoil knots, the Lorenz attractor), plus complex curves (chirp, Cornu spiral; section
 * "complex"), computed here and drawn by curve.ts.
 */

import type { Entry } from "./types.ts";
import { periodic } from "../curve.ts";
import { fresnel, linspace } from "../lib/specfun.ts";

const PI = Math.PI;
const S2 = Math.sqrt(0.5);
const CIRC = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]];
const CIRC_W = [1, S2, 1, S2, 1, S2, 1, S2, 1];
const CIRC_K = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4];

function nurbs(name: string, description: string, curve: Record<string, unknown>, extra: Record<string, unknown> = {}): Entry {
  const is3D = Array.isArray((curve.points as number[][])[0]) && (curve.points as number[][])[0].length === 3;
  return {
    id: `curves/${name}`,
    section: "curves",
    title: name,
    description,
    renderer: "curve.ts",
    control: { title: name, curve, controlPolygon: true, ...extra },
    interaction: is3D ? "orbit" : "pan-zoom-xy",
  };
}

const tt40 = linspace(0, 2 * PI, 40, false);
const heart = tt40.map((t) => [16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)]);
const tt60 = linspace(0, 2 * PI, 60, false);
const rose = tt60.map((t) => [Math.cos(4 * t) * Math.cos(t), Math.cos(4 * t) * Math.sin(t)]);
const tt180 = linspace(0, 6 * PI, 180, false);
const [Rr, rr, dd] = [5, 3, 5];
const spiro = tt180.map((t) => [(Rr - rr) * Math.cos(t) + dd * Math.cos(((Rr - rr) / rr) * t), (Rr - rr) * Math.sin(t) - dd * Math.sin(((Rr - rr) / rr) * t)]);
const trefoil = tt60.map((t) => [Math.sin(t) + 2 * Math.sin(2 * t), Math.cos(t) - 2 * Math.cos(2 * t), -Math.sin(3 * t)]);
const th50 = linspace(0, 6 * PI, 50);
const helix = th50.map((t) => [Math.cos(t), Math.sin(t), t / (2 * PI)]);

const nurbsEntries: Entry[] = [
  nurbs("circle", "exact unit circle (rational quadratic, weights √½)", { degree: 2, points: CIRC, weights: CIRC_W, knots: CIRC_K }),
  nurbs("ellipse", "rational quadratic, the circle's control points scaled by (3, 1)", { degree: 2, points: CIRC.map(([x, y]) => [3 * x, y]), weights: CIRC_W, knots: CIRC_K }),
  nurbs("heart", "periodic: first 𝑝 points wrapped, uniform knots", periodic(heart, 3) as unknown as Record<string, unknown>),
  nurbs("rose", "periodic cubic through a four-petal rose's samples", periodic(rose, 3) as unknown as Record<string, unknown>),
  nurbs("spirograph", "periodic cubic, 180 control points of a hypotrochoid", periodic(spiro, 3) as unknown as Record<string, unknown>),
  nurbs("kink_bspline", "cubic B-spline, a knot of multiplicity 3 = 𝑝: kink through a control point", {
    degree: 3,
    points: [[0, 0], [1, 2], [2, -1], [3, 2], [4, 0], [5, 2], [6, 1], [7, 0]],
    knots: [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3],
  }),
  nurbs("helix_3d", "cubic B-spline helix (3 turns, clamped), colored by 𝑢", {
    degree: 3,
    points: helix,
    knots: [0, 0, 0, ...Array.from({ length: 48 }, (_, i) => i), 47, 47, 47],
  }),
  nurbs("trefoil_3d", "periodic 3D curve, colored by 𝑢", periodic(trefoil, 3) as unknown as Record<string, unknown>),
];

// three_d/curves/trefoil_curvature: (tx, ty, tz) with curvature κ = |r′ × r″| / |r′|³, 1500
// points, evaluated (f64) by curve.ts from the `gen` spec
const trefoilCurvature: Entry = {
  id: "curves/trefoil_curvature",
  section: "curves",
  title: "trefoil_curvature",
  description: "CF trajectory (tx, ty, tz) colored by curvature 𝜅 (Plasma)",
  renderer: "curve.ts",
  control: {
    title: "curvature 𝜅 (m⁻¹)",
    gen: { name: "trefoil", n: 1500, t: [0, 2 * PI] },
    paramLabel: "𝑡",
    colorBy: { label: "𝜅 (m⁻¹)" },
    colormap: "Plasma",
    lineWidth: 3,
  },
  interaction: "orbit",
};

// basics/complex: chirp (t, ℜ, ℑ) colored by phase (1200 points from the `gen` spec, f64),
// Cornu spiral C + iS in the plane (cumulative quadrature: on the CPU)
const chirp: Entry = {
  id: "complex/chirp",
  section: "complex",
  title: "chirp",
  description: "complex curve (𝑡, ℜ, ℑ): a helix tightening as the frequency rises, colored by phase",
  renderer: "curve.ts",
  control: {
    title: "Chirp pulse exp(1.5𝑖𝑡²) exp(−𝑡²/12)",
    gen: { name: "chirp", n: 1200, t: [-6, 6], rate: 1.5, width: 12 }, // u = t (read-out)
    // the colour: the unwrapped phase, wrapped per fragment (curve.ts)
    colorBy: { label: "phase (rad)", domain: [-PI, PI], ticks: "pi", cyclic: true },
    colormap: "Twilight",
    aspect: [0.4, 1, 1],
    axisLabels: ["𝑡", "ℜ", "ℑ"],
    paramLabel: "𝑡",
    azimuth: 30,
    elevation: 20,
    lineWidth: 2.5,
  },
  interaction: "orbit",
};

const tf = linspace(-8, 8, 2000);
const [fc, fs] = fresnel(tf);
const cornu: Entry = {
  id: "complex/cornu_spiral",
  section: "complex",
  title: "cornu_spiral",
  description: "complex curve: Fresnel integrals 𝐶(𝑡) + 𝑖𝑆(𝑡) drawn in the complex plane",
  renderer: "curve.ts",
  control: {
    title: "Cornu spiral 𝐶(𝑡) + 𝑖𝑆(𝑡)",
    polyline: Array.from(tf, (_, i) => [fc[i], fs[i]]),
    params: tf,
    paramLabel: "𝑡",
    axisLabels: ["ℜ", "ℑ"],
    lineWidth: 2,
  },
  interaction: "pan-zoom-xy",
};

// points/lorenz: the Lorenz attractor (σ = 10, ρ = 28, β = 8/3), RK4 with dt = 0.004, 40 000 steps,
// coloured by time with Plasma; integrated by curve.ts from the `ode` spec
const lorenz: Entry = {
  id: "curves/lorenz",
  section: "curves",
  title: "lorenz",
  description: "Lorenz attractor (𝜎 = 10, 𝜌 = 28, 𝛽 = 8/3): RK4, 40 000 steps of 𝑑𝑡 = 0.004, coloured by time",
  renderer: "curve.ts",
  control: {
    title: "Lorenz attractor (𝜎 = 10, 𝜌 = 28, 𝛽 = 8/3)",
    ode: { system: "lorenz", n: 40_000, dt: 0.004, colorBy: "time" },
    paramLabel: "𝑡",
    axisLabels: ["𝑥", "𝑦", "𝑧"],
    colormap: "Plasma",
    lineWidth: 1.5,
    azimuth: 35,
    elevation: 20,
  },
  interaction: "orbit",
};

// The same coloured by speed through the root scale with n = 0.5 (t², squared): fast segments stand out
const lorenzRoot: Entry = {
  ...lorenz,
  id: "curves/lorenz_root",
  description: "Lorenz attractor coloured by speed, root color scale with n = 0.5 (squares): the fast segments stand out",
  control: { ...lorenz.control, ode: { system: "lorenz", n: 40_000, dt: 0.004, colorBy: "speed" }, scale: "root", rootN: 0.5 },
};

// The same with a slider for ρ (the mod writes `sliderValues.rho` as it's dragged, curve.ts
// re-integrates): below ρ = 1 the origin attracts; up to ρ ≈ 24.7 one of C± does; chaos past it
const lorenzRho: Entry = {
  ...lorenz,
  id: "curves/lorenz_rho",
  title: "lorenz_rho",
  description: "Lorenz attractor with a slider for 𝜌: from a stable fixed point (𝜌 < 24.7) to chaos",
  control: { ...lorenz.control, title: "Lorenz attractor (𝜎 = 10, 𝛽 = 8/3)", ode: { ...lorenz.control.ode as object, rho: 15 }, sliders: { rho: [0, 50] } },
};

// three.js's webgl_lines_fat: a 3D Hilbert curve (as GeometryUtils.hilbert3D: corners of a cube
// visited recursively, each sub-cube's corners permuted) through which a smooth curve is drawn
// thick, with a hue gradient. Here the curve is exact: three's Catmull-Rom spline through the 64
// points (one iteration; 512 with two) as a NURBS, on screen 6 px wide with round joins
function hilbert3D(center: number[], size: number, iterations: number, v = [0, 1, 2, 3, 4, 5, 6, 7]): number[][] {
  const h = size / 2;
  const [cx, cy, cz] = center;
  const corners = [
    [cx - h, cy + h, cz - h], [cx - h, cy + h, cz + h], [cx - h, cy - h, cz + h], [cx - h, cy - h, cz - h],
    [cx + h, cy - h, cz - h], [cx + h, cy - h, cz + h], [cx + h, cy + h, cz + h], [cx + h, cy + h, cz - h],
  ];
  const vec = v.map((i) => corners[i]);
  if (iterations <= 0) return vec;
  const [v0, v1, v2, v3, v4, v5, v6, v7] = v;
  const orders = [
    [v0, v3, v4, v7, v6, v5, v2, v1], [v0, v7, v6, v1, v2, v5, v4, v3], [v0, v7, v6, v1, v2, v5, v4, v3], [v2, v3, v0, v1, v6, v7, v4, v5],
    [v2, v3, v0, v1, v6, v7, v4, v5], [v4, v3, v2, v5, v6, v1, v0, v7], [v4, v3, v2, v5, v6, v1, v0, v7], [v6, v5, v2, v1, v0, v3, v4, v7],
  ];
  return vec.flatMap((c, k) => hilbert3D(c, h, iterations - 1, orders[k]));
}

/** Uniform Catmull-Rom spline through `pts` (ends extended by reflection, as three's
 * CatmullRomCurve3; centripetal = uniform for equal steps) as a C¹ cubic B-spline: segment i the
 * Bézier p_i, p_i + (p_{i+1} − p_{i−1})/6, p_{i+1} − (p_{i+2} − p_i)/6, p_{i+1}; knots of
 * multiplicity 3 at the points (u = their index). */
function catmullRom(pts: number[][]): { degree: number; points: number[][]; knots: number[] } {
  const n = pts.length;
  const at = (i: number) => pts[i] ?? (i < 0 ? pts[0].map((x, d) => 2 * x - pts[1][d]) : pts[n - 1].map((x, d) => 2 * x - pts[n - 2][d]));
  const points = [pts[0]];
  for (let i = 0; i < n - 1; i += 1) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    points.push(p1.map((x, d) => x + (p2[d] - p0[d]) / 6), p2.map((x, d) => x - (p3[d] - p1[d]) / 6), p2);
  }
  const knots = [0, 0, 0, 0, ...Array.from({ length: n - 2 }, (_, i) => [i + 1, i + 1, i + 1]).flat(), n - 1, n - 1, n - 1, n - 1];
  return { degree: 3, points, knots };
}

const hilbertFat: Entry = {
  id: "curves/hilbert_fat",
  section: "curves",
  title: "hilbert_fat",
  description: "fat line: a 3D Hilbert curve (64 points) through a Catmull-Rom spline (exact, a C¹ cubic B-spline), 6 px wide with round joins, Sinebow hue along 𝑢",
  renderer: "curve.ts",
  control: {
    title: "3D Hilbert curve, Catmull-Rom spline",
    curve: catmullRom(hilbert3D([0, 0, 0], 20, 1)),
    colormap: "Sinebow",
    lineWidth: 6,
    lineCap: "round",
    azimuth: 30,
    elevation: 25,
  },
  interaction: "orbit",
};

export const entries: Entry[] = [...nurbsEntries, trefoilCurvature, lorenz, lorenzRoot, lorenzRho, hilbertFat, chirp, cornu];
