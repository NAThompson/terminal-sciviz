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
// points, evaluated on the GPU by curve.ts from the `gen` spec
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

// basics/complex: chirp (t, ℜ, ℑ) colored by phase (1200 points from the `gen` spec, on the GPU),
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

export const entries: Entry[] = [...nurbsEntries, trefoilCurvature, lorenz, lorenzRoot, chirp, cornu];
