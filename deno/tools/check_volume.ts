/**
 * Checks of volume.ts against references:
 *
 * 1. GPU generators against f64 CPU references written here (hydrogen orbitals from the
 *    associated Legendre and generalised Laguerre polynomials and real spherical harmonics, the
 *    bands, the breathing Gaussian) at random voxels:
 *    max |gpu − ref| / max |ref|, and the median relative error of voxels above 1% of the max.
 * 2. GPU marching cubes against the CPU reference (lib/marching_cubes.ts) at the same levels, on
 *    the same (read-back) f32 voxels: triangle counts and surface areas (index space).
 *
 *   deno run -A tools/check_volume.ts [--voxels 300]
 */

import { marchingCubes } from "../lib/marching_cubes.ts";
import { axisValues, DEFAULTS, type VolumeMeta, VolumePlot, type VolumeSpec, volumeMeta } from "../volume.ts";
import { renderOnce } from "../lib/serve.ts";

const fact = (k: number) => { let f = 1; for (let i = 2; i <= k; i += 1) f *= i; return f; };
function assocLegendre(l: number, m: number, x: number): number {
  let pmm = 1;
  const somx2 = Math.sqrt(Math.max(1 - x * x, 0));
  let f = 1;
  for (let i = 0; i < m; i += 1) { pmm *= f * somx2; f += 2; }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  for (let ll = m + 2; ll <= l; ll += 1) [pmm, pmmp1] = [pmmp1, (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)];
  return pmmp1;
}
function genLaguerre(n: number, a: number, x: number): number {
  let [l0, l1] = [1, 1 + a - x];
  if (n === 0) return l0;
  for (let k = 1; k < n; k += 1) [l0, l1] = [l1, ((2 * k + 1 + a - x) * l1 - (k + a) * l0) / (k + 1)];
  return l1;
}
function hydrogen(n: number, l: number, m: number, x: number, y: number, z: number): number {
  const r = Math.hypot(x, y, z);
  const theta = Math.acos(r > 0 ? z / r : 1);
  const phi = Math.atan2(y, x);
  const rho = 2 * r / n;
  const norm = Math.sqrt((2 / n) ** 3 * fact(n - l - 1) / (2 * n * fact(n + l)));
  const radial = norm * Math.exp(-rho / 2) * rho ** l * genLaguerre(n - l - 1, 2 * l + 1, rho);
  const am = Math.abs(m);
  const ynorm = Math.sqrt((2 * l + 1) / (4 * Math.PI) * fact(l - am) / fact(l + am));
  const p = assocLegendre(l, am, Math.cos(theta));
  const ang = m === 0 ? ynorm * p : Math.SQRT2 * ynorm * p * (m > 0 ? Math.cos(m * phi) : Math.sin(-m * phi));
  return radial * ang;
}

function reference(spec: VolumeSpec, meta: VolumeMeta, slice: number, i: number, j: number, k: number): number {
  const [xs, ys, zs] = meta.axes.map(axisValues);
  const [x, y, z] = [xs[i], ys[j], zs[k]];
  const s = spec as unknown as Record<string, number>;
  switch (spec.dataset) {
    case "hydrogen": {
      const psi = hydrogen(s.n ?? 3, s.l ?? 2, s.m ?? 1, x, y, z);
      const q = (spec as { quantity?: string }).quantity ?? "psi";
      return q === "psi" ? psi : q === "density" ? psi * psi + 1e-14 : Math.log10(psi * psi + 1e-14);
    }
    case "hydrogen_spherical": {
      const psi = hydrogen(s.n ?? 3, s.l ?? 1, s.m ?? 0, x * Math.sin(y) * Math.cos(z), x * Math.sin(y) * Math.sin(z), x * Math.cos(y));
      return psi * psi;
    }
    case "breathing": {
      const t = slice / meta.slices;
      return Math.exp(-(x * x + y * y + z * z) / (0.1 + 0.08 * Math.sin(2 * Math.PI * t))) * Math.cos(6 * x);
    }
    case "tight_binding":
      return -2 * (Math.cos(x) + Math.cos(y) + Math.cos(z));
    default: {
      const [cx, cy, cz] = [Math.cos(x), Math.cos(y), Math.cos(z)];
      return 1.5 - 2 * (cx + cy + cz) + 1.2 * (cx * cy + cy * cz + cz * cx);
    }
  }
}

function area(pos: ArrayLike<number>, idx?: ArrayLike<number>): number {
  let a = 0;
  const n = idx ? idx.length / 3 : pos.length / 9;
  for (let t = 0; t < n; t += 1) {
    const v = [0, 1, 2].map((c) => (idx ? idx[t * 3 + c] : t * 3 + c));
    const p = v.map((q) => [pos[q * 3], pos[q * 3 + 1], pos[q * 3 + 2]]);
    const u = [0, 1, 2].map((c) => p[1][c] - p[0][c]), w = [0, 1, 2].map((c) => p[2][c] - p[0][c]);
    a += Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
  }
  return a;
}

const nVox = Number(Deno.args[Deno.args.indexOf("--voxels") + 1]) || 300;
const plot = await VolumePlot.create();
const specs: [VolumeSpec, number][] = [
  [{ dataset: "hydrogen", n: 3, l: 2, m: 1, quantity: "psi" }, 0],
  [{ dataset: "hydrogen", n: 3, l: 2, m: 1, quantity: "density" }, 0],
  [{ dataset: "hydrogen", n: 3, l: 2, m: 1, quantity: "log10_density" }, 0],
  [{ dataset: "hydrogen", n: 4, l: 3, m: -2, quantity: "psi", size: 48 }, 0],
  [{ dataset: "hydrogen_spherical", n: 3, l: 1, m: 0 }, 0],
  [{ dataset: "breathing" }, 3],
  [{ dataset: "tight_binding" }, 0],
  [{ dataset: "fermi" }, 0],
];
let seed = 12345;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
console.log("1. generators (GPU f32 vs CPU f64)");
for (const [spec, slice] of specs) {
  await renderOnce(plot, { ...DEFAULTS, volume: spec, slice, width: 64, height: 48 } as never);
  const { values, meta } = plot.volumeData!;
  const [nz, ny, nx] = meta.shape;
  let max = 0, maxErr = 0;
  const rels: number[] = [];
  const pts: [number, number, number, number, number][] = [];
  for (let s = 0; s < nVox; s += 1) {
    const [i, j, k] = [Math.floor(rand() * nx), Math.floor(rand() * ny), Math.floor(rand() * nz)];
    const ref = reference(spec, meta, slice, i, j, k);
    const gpu = values[(k * ny + j) * nx + i];
    pts.push([i, j, k, ref, gpu]);
    max = Math.max(max, Math.abs(ref));
    maxErr = Math.max(maxErr, Math.abs(gpu - ref));
  }
  for (const [, , , ref, gpu] of pts) if (Math.abs(ref) > 0.01 * max) rels.push(Math.abs(gpu - ref) / Math.abs(ref));
  rels.sort((a, b) => a - b);
  console.log(`  ${JSON.stringify(spec).padEnd(78)} slice ${slice}  max|err|/max|ref| ${(maxErr / max).toExponential(2)}  median rel ${rels[rels.length >> 1].toExponential(2)}  p99 rel ${rels[Math.floor(rels.length * 0.99)].toExponential(2)} (${rels.length}/${nVox} voxels)`);
}

console.log("2. isosurfaces (GPU marching cubes vs CPU reference, same f32 voxels)");
const cases: [string, VolumeSpec, number[] | null][] = [
  ["orbital_iso", { dataset: "hydrogen", n: 3, l: 2, m: 1, quantity: "psi" }, [-0.01, 0.01]],
  ["fermi_surface", { dataset: "fermi", size: 64 }, null],
];
for (const [name, spec, levels] of cases) {
  await renderOnce(plot, { ...DEFAULTS, volume: spec, isoLevels: levels, width: 64, height: 48 } as never);
  const { values, meta } = plot.volumeData!;
  const stats = plot.isoStats;
  for (let i = 0; i < stats.length; i += 1) {
    const gpu = await plot.isoPositions(i);
    const t0 = performance.now();
    const cpu = marchingCubes(values, meta.shape, stats[i].level);
    const cpuMs = performance.now() - t0;
    const [ga, ca] = [area(gpu), area(cpu.positions, cpu.indices)];
    console.log(`  ${name} level ${stats[i].level}: triangles GPU ${stats[i].triangles}, CPU ${cpu.indices.length / 3} (diff ${stats[i].triangles - cpu.indices.length / 3}); area GPU ${ga.toFixed(3)}, CPU ${ca.toFixed(3)} (rel diff ${((ga - ca) / ca).toExponential(2)}); CPU vertices ${cpu.positions.length / 3} (welded); GPU ${stats[i].ms} ms, CPU ${cpuMs.toFixed(1)} ms`);
  }
}
Deno.exit(0);
