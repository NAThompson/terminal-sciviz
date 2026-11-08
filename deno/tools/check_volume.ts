/**
 * Checks of volume.ts against references:
 *
 * 1. GPU generators against f64 CPU references written here (hydrogen orbitals from the
 *    associated Legendre and generalised Laguerre polynomials and real spherical harmonics, the
 *    bands, the breathing Gaussian) at random voxels:
 *    max |gpu − ref| / max |ref|, and the median relative error of voxels above 1% of the max.
 * 2. GPU marching cubes against the CPU reference (lib/marching_cubes.ts) at the same levels, on
 *    the same (read-back) f32 voxels: triangle counts and surface areas (index space); and over
 *    the active bricks only against every cell: the same triangles.
 * 3. Files: .npy headers, non-finite values, slices read on demand (lib/stack.ts) against the
 *    whole file, short reads, a changed file read again, the halved preview, NaN validity.
 *
 *   deno run -A tools/check_volume.ts [--voxels 300]
 */

import { marchingCubes } from "../lib/marching_cubes.ts";
import { axisValues, DEFAULTS, readStack, type VolumeMeta, VolumePlot, type VolumeSpec, volumeMeta } from "../volume.ts";
import { downsample, prepareSlice, readAt, readSlice, StackReader } from "../lib/stack.ts";
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
let badMc = 0;
const cases: [string, VolumeSpec, number[] | null][] = [
  ["orbital_iso", { dataset: "hydrogen", n: 3, l: 2, m: 1, quantity: "psi" }, [-0.01, 0.01]],
  ["fermi_surface", { dataset: "fermi", size: 64 }, null],
  ["file_slices (NaN outside the ball)", { dataset: "file", path: "assets/ball_modes.npy" }, [0.3]],
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
  // every cell (fullMarch) against the active bricks': the same triangles, in another order
  const tris = async () => {
    const out: string[] = [];
    for (let i = 0; i < stats.length; i += 1) {
      const p = await plot.isoPositions(i);
      for (let t = 0; t < p.length; t += 9) out.push(Array.from(p.subarray(t, t + 9)).join(","));
    }
    return out.sort().join(";");
  };
  const bricked = await tris();
  plot.fullMarch = true;
  await renderOnce(plot, { ...DEFAULTS, volume: spec, isoLevels: levels, width: 64, height: 48 } as never);
  const full = await tris();
  plot.fullMarch = false;
  await renderOnce(plot, { ...DEFAULTS, volume: spec, isoLevels: levels, width: 64, height: 48 } as never); // the bricked meshes current again
  if (bricked !== full) badMc += 1;
  console.log(`  ${bricked === full ? "ok  " : "FAIL"} ${name}: active bricks' triangles = every cell's (${full.split(";").length} triangles)`);
}

console.log("3. files: .npy headers, non-finite values, slices on demand, slice values, NaN validity");
let bad = badMc;
const ok = (name: string, f: () => unknown) => {
  let r: unknown;
  try { r = f(); } catch (err) { r = `threw ${(err as Error).message}`; }
  if (r !== true) bad += 1;
  console.log(`  ${r === true ? "ok  " : "FAIL"} ${name}${r === true ? "" : `: ${r}`}`);
};
const dir = Deno.makeTempDirSync();
const npy = (name: string, header: string, values: ArrayLike<number>, wide = false) => {
  const h = header + " ".repeat(63 - ((10 + header.length) % 64)) + "\n";
  const b = new Uint8Array(10 + h.length + values.length * (wide ? 8 : 4));
  b.set([0x93, 78, 85, 77, 80, 89, 1, 0, h.length & 255, h.length >> 8]);
  b.set(new TextEncoder().encode(h), 10);
  new (wide ? Float64Array : Float32Array)(b.buffer, 10 + h.length).set(values);
  Deno.writeFileSync(`${dir}/${name}`, b);
  return { dataset: "file" as const, path: `${dir}/${name}` };
};
const eight = [1, 2, 3, 4, 5, 6, 7, 8];
const throwsRe = (f: () => unknown, re: RegExp) => { try { f(); } catch (e) { return re.test((e as Error).message) || (e as Error).message; } return "did not throw"; };
ok("double-quoted header", () => readSlice(readStack(npy("dq.npy", `{"descr": "<f4", "fortran_order": False, "shape": (2, 2, 2)}`, eight)), 0)[7] === 8);
ok("mixed quotes, fortran_order True rejected", () => throwsRe(() => readStack(npy("fq.npy", `{'descr': '<f4', "fortran_order": True, 'shape': (2, 2, 2), }`, eight)), /fortran_order/));
ok("1-tuple shape (5,): not 3D", () => throwsRe(() => readStack(npy("one.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (5,), }`, [1, 2, 3, 4, 5])), /shape must be 3 or 4/));
ok("a missing key", () => throwsRe(() => readStack(npy("mk.npy", `{'descr': '<f4', 'shape': (2, 2, 2), }`, eight)), /without descr, fortran_order and shape/));
const prepared = (spec: { path: string }, k: number) => { const st = readStack({ dataset: "file", ...spec }); return prepareSlice(readSlice(st, k), st, k); };
ok("±∞ rejected at its index", () => throwsRe(() => prepared(npy("inf.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }`, [1, 2, 3, Infinity, 5, 6, 7, 8]), 0), /flat index 3/));
ok("±∞ in slice 1 of a stack: its flat index in the file", () => throwsRe(() => prepared(npy("inf4.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2, 2), }`, [...eight, 1, 2, -Infinity, 4, 5, 6, 7, 8]), 1), /value -Infinity at flat index 10/));
ok("f64 beyond f32 rejected with its value", () => throwsRe(() => prepared(npy("big64.npy", `{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2, 2), }`, [1, 2, 3, 4, 1e300, 6, 7, 8], true), 0), /value 1e\+300 at flat index 4/));
// Slices read on demand = the whole file read at once (f32 and f64, NaN kept)
const ref = Array.from({ length: 5 * 4 * 6 * 8 }, (_, i) => (i % 7 === 3 ? NaN : Math.sin(i) * 1e3));
for (const wide of [false, true]) {
  const spec = npy(`stack${wide ? 64 : 32}.npy`, `{'descr': '<f${wide ? 8 : 4}', 'fortran_order': False, 'shape': (5, 4, 6, 8), }`, ref, wide);
  const st = readStack(spec);
  const bytes = Deno.readFileSync(spec.path);
  const whole = Float32Array.from(new (wide ? Float64Array : Float32Array)(bytes.buffer.slice(st.offset)));
  ok(`slices on demand = the whole file (${wide ? "f64" : "f32"}, header ${st.offset} bytes)`, () => {
    for (let k = 0; k < 5; k += 1) {
      const s = readSlice(st, k), w = whole.subarray(k * 192, (k + 1) * 192);
      if (!s.every((v, i) => Object.is(v, w[i]))) return `slice ${k} differs`;
    }
    return true;
  });
}
ok("short reads continued; reading past the end throws", () => {
  const data = Uint8Array.from({ length: 100 }, (_, i) => i);
  let at = 0;
  const file = { seekSync: (o: number) => (at = o), readSync: (b: Uint8Array) => { const n = Math.min(7, b.length, data.length - at); if (n <= 0) return null; b.set(data.subarray(at, at + n)); at += n; return n; } };
  const out = new Uint8Array(30);
  readAt(file as unknown as Deno.FsFile, 13, out);
  if (!out.every((v, i) => v === 13 + i)) return `got ${out}`;
  return throwsRe(() => readAt(file as unknown as Deno.FsFile, 90, new Uint8Array(20)), /end of file at byte 100/);
});
ok("a truncated file: its size checked", () => {
  const spec = npy("trunc.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }`, eight);
  Deno.truncateSync(spec.path, Deno.statSync(spec.path).size - 4);
  return throwsRe(() => readStack(spec), /28 bytes of data/);
});
{ // a changed file (same size, its mtime moved): read again, through the renderer (the worker)
  const spec = npy("change.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2, 2), }`, [...eight, ...eight.map((v) => v * 10)]);
  const shown = async () => { await renderOnce(plot, { ...DEFAULTS, volume: spec, slice: 1, width: 64, height: 48 } as never); return plot.volumeData!.values[7]; };
  const before = await shown();
  npy("change.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2, 2), }`, [...eight, ...eight.map((v) => v * 100)]);
  Deno.utimeSync(spec.path, new Date(), new Date(Date.now() + 5000));
  const after = await shown();
  ok("a changed file read again (its slices and domain)", () => (before === 80 && after === 800) || `slice 1 [7]: ${before}, then ${after}`);
}
{ // one file read two ways (raw f64, then the same bytes as f32): jobs and domains by layout
  const path = `${dir}/two.bin`;
  Deno.writeFileSync(path, new Uint8Array(Float64Array.from(eight).buffer));
  const f64 = { dataset: "file" as const, path, shape: [2, 2, 2], dtype: "f64" as const };
  const f32 = { dataset: "file" as const, path, shape: [2, 2, 4], dtype: "f32" as const };
  const reader = new StackReader();
  const pre = reader.slice(readStack(f32), 0, { prefetch: true }); // running when the f64 request comes
  const got = await reader.slice(readStack(f64), 0);
  await pre.catch(() => {});
  ok("a pending prefetch of another layout isn't shared", () => (got.values.length === 8 && got.values[7] === 8) || `got ${got.values}`);
  const domain = async (volume: object) => ((await renderOnce(plot, { ...DEFAULTS, volume, width: 64, height: 48 } as never)).state as { dataDomain: number[] }).dataDomain;
  const [d64, d32] = [await domain(f64), await domain(f32)];
  ok("the whole-stack domain by layout (f64 [1, 8], the same bytes as f32 their own)", () => (d64.join() === "1,8" && d32.join() !== "1,8" && d32[0] === 0) || `f64 ${d64}, f32 ${d32}`);
}
{ // the renderer's slice (worker, NaN restored for volumeData) = the file's; the halved preview
  const spec = npy("nan4.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (5, 4, 6, 8), }`, ref);
  await renderOnce(plot, { ...DEFAULTS, volume: spec, slice: 3, width: 64, height: 48 } as never);
  const st = readStack(spec);
  ok("renderer's slice 3 = the file's (NaN at the same voxels)", () => plot.volumeData!.values.every((v, i) => Object.is(v, readSlice(st, 3)[i])) || "differs");
  ok("preview: means of the valid voxels of each 2³, valid with at least 4", () => {
    const src = readSlice(st, 2), d = downsample(readSlice(st, 2), st, 2);
    for (let o = 0; o < 2 * 3 * 4; o += 1) {
      const [x, y, z] = [o % 4, Math.floor(o / 4) % 3, Math.floor(o / 12)];
      const vals = [0, 1].flatMap((dz) => [0, 1].flatMap((dy) => [0, 1].map((dx) => src[((2 * z + dz) * 6 + 2 * y + dy) * 8 + 2 * x + dx]))).filter((v) => v === v);
      const valid = vals.length >= 4;
      if (valid !== (!d.valid || d.valid[o] === 255)) return `validity at ${o}`;
      if (valid && Math.abs(d.values[o] - vals.reduce((a, b) => a + b, 0) / vals.length) > 1e-3) return `value at ${o}`;
    }
    return true;
  });
}
const big = npy("big.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 2, 2, 2), }`, Array.from({ length: 24 }, (_, i) => i));
await renderOnce(plot, { ...DEFAULTS, volume: { ...big, axes: [{ lo: 0, hi: 1 }, { lo: 0, hi: 1 }, { lo: 0, hi: 1 }, { lo: 1e12, hi: 1e12 + 2 }] }, slice: 2, width: 64, height: 48 } as never);
ok("slice values at full precision (1e12 … 1e12 + 2): slice 2 shown", () => plot.volumeData!.values[0] === 16 || `first value ${plot.volumeData!.values[0]}`);
// NaN validity: the CPU hover rule (≥ 1 − 1e-3 of the trilinear weight on valid voxels) on a NaN corner
const half = npy("nan.npy", `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }`, [NaN, 3e38, 3e38, 3e38, 3e38, 3e38, 3e38, 3e38]);
const st = await renderOnce(plot, { ...DEFAULTS, volume: half, width: 64, height: 48 } as never);
ok("NaN next to 3e38: domain over the valid voxels only", () => JSON.stringify((st.state as { dataDomain: number[] }).dataDomain) === JSON.stringify([Math.fround(3e38), Math.fround(3e38)]) || JSON.stringify((st.state as { dataDomain: number[] }).dataDomain));
Deno.removeSync(dir, { recursive: true });
Deno.exit(bad ? 1 : 0);
