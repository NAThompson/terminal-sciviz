/**
 * The "scenes" section: 3D marks drawn by scene.ts: a convex hull over random points (three.js's
 * webgl_geometry_convex), a point cloud coloured by a number, a categorical 3D scatter. The points
 * come from scene.ts's seeded `gen` specs, so the controls stay small.
 */

import type { Entry } from "./types.ts";

const ball = { name: "ball", n: 100, seed: 11 }; // uniform in a ball of radius 10

const scene = (name: string, title: string, description: string, control: Record<string, unknown>): Entry => ({
  id: `scenes/${name}`,
  section: "scenes",
  title,
  description,
  renderer: "scene.ts",
  control: { title, ...control },
  interaction: "orbit",
});

export const entries: Entry[] = [
  scene("convex_hull", "convex hull", "100 random points in a ball and their convex hull (quickhull, f64): translucent, the points behind it dimmed", {
    subtitle: "100 points in a ball",
    marks: [
      { type: "dot", gen: ball, transform: { hull: true } },
      { type: "dot", gen: ball, fill: "#E69F00", r: 4 },
    ],
  }),
  scene("torus_cloud", "noisy torus", "20 000 points near a torus as shaded sphere impostors, coloured by their angle (cyclic Twilight map, colour bar)", {
    subtitle: "20 000 points",
    marks: [{ type: "dot", gen: { name: "torus", n: 20000, seed: 2 }, fill: "angle", r: 1.6 }],
    color: { scheme: "twilight", label: "angle (°)" },
  }),
  scene("clusters", "three clusters", "a categorical 3D scatter: three Gaussian clusters, Okabe–Ito colours and a legend", {
    marks: [{ type: "dot", gen: { name: "clusters", n: 900, seed: 5 }, fill: "cluster", r: 3 }],
    elevation: 20,
  }),
];
