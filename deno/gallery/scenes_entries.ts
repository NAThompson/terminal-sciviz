/**
 * The "scenes" section: 3D marks drawn by scene.ts: a convex hull over random points (three.js's
 * webgl_geometry_convex), a point cloud coloured by a number, a categorical 3D scatter, a
 * volume with a surface and points in it. The points come from scene.ts's seeded `gen` specs, so the controls stay small.
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
  scene("volume_marks", "field, surface, points", "a ray-marched field (a Gaussian) with an opaque patch through it and points: each ray ends at the patch or a point, the points inside tinted by the field in front of them", {
    subtitle: "one volume mark, composited by depth",
    marks: [
      { type: "volume", gen: { name: "gaussian", n: 48 }, extent: [[-10, 10], [-10, 10], [-10, 10]], opacity: 0.12, label: "field" },
      { type: "surface", nurbs: { degrees: [1, 1], shape: [2, 2], points: [-10, -10, 0, -10, 10, 0, 10, -10, 0, 10, 10, 0], weights: [1, 1, 1, 1], knots: [[0, 0, 1, 1], [0, 0, 1, 1]] }, fill: "#8b949e", resolution: 1 },
      { type: "dot", gen: ball, fill: "#E69F00", r: 4 },
    ],
    color: { scheme: "magma" },
  }),
];
