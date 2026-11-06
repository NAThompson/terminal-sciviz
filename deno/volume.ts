/**
 * Volume renderer (WebGPU):
 *
 * - data generated on the GPU from a small spec (`volume: {dataset: "hydrogen", n, l, m, ...}`,
 *   or `breathing`, `tight_binding`, `fermi`), by a compute shader writing straight into an r32float 3D
 *   texture (filtered: `float32-filterable`), cached by spec and slice;
 * - a single-pass ray marcher: DVR / MIP / Iso modes, opacity ramp above a threshold (of the
 *   distance from the centre value for diverging colouring, so values near it stay clear) with
 *   sampling-rate correction, colours over [value at threshold, max], gradient shading, jitter,
 *   non-uniform axes (lookup tables of voxel edges and centres), rays stopped at opaque
 *   isosurfaces (a depth pre-pass); colour maps sampled as a nearest 256-texel texture;
 * - isosurfaces by marching cubes in compute shaders (count, prefix sum, emit), with the case
 *   triangulations of lib/marching_cubes.ts (face walks, asymptotic decider, centroid fans), its
 *   snapping and degenerate-triangle rule, normals from central differences; lit as a standard
 *   PBR material (roughness 0.55, metalness 0.05) under a headlight (2.4) and ambient light (0.35),
 *   one solid colour per level (or the level through the colour map);
 * - the box (white edges at 0.2), ticks and titles on its edges when the axes are labelled (label
 *   collisions by priority), the colour bar with its bounds, title, hover read-out.
 *
 * World: z up (orbit camera of curve.ts, lib/framing.ts); `up: "y"` (default) maps the box's y to
 * z, so the default azimuth/elevation looks from (1.4, 1, 1.75) in box space.
 *
 *   deno run -A volume.ts png CONTROL.json -o out.png
 *   deno run -A volume.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 */

import { apply, cameraWorld, invert, type Mat4, mul, perspective, rows, scaling } from "./lib/mat4.ts";
import { colormapLut, colormapName, colormapRgb } from "./lib/colormaps.ts";
import {
  colorBarTicks,
  createScale,
  DEFAULT_DOMAIN,
  type Domain,
  formatBound,
  formatTick,
  getDomain,
  getRealValue,
  getSafeDomain,
  getShaderScale,
  getVisDomain,
  type ScaleType,
  scaleDomain,
  snapRootN,
  ticks as linearTicks,
  type ValueTransform,
  type VisScale,
} from "./lib/scales.ts";
import {
  getAxisDomain,
  getAxisLookup,
  getBoxSize,
  getBoxSizeFromAxes,
  getStepCount,
  getThresholdDomain,
  getThresholdUniforms,
  normalizeBoxSize,
  type Quality,
  SAMPLES_PER_VOXEL,
} from "./lib/volume_utils.ts";
import { CENTROID_BASE, getCaseTriangles } from "./lib/marching_cubes.ts";
import { cameraBasis, fitView, type Vec3 } from "./lib/framing.ts";
import { makeReadBuffer, rowBytes } from "./lib/gpu.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba, rootSlider, type RootSlider, rootSliderHeight } from "./lib/overlay.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";

const INK = "#c9d1d9", INK2 = "#8b949e", HOVER_RING = "#e6edf3";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];
const FOV = 45; // VolumeControls
const FIT_MARGIN = 1.08;
const SAMPLES = 4;
const DEPTH = "depth24plus";
const ISO_COLORS = ["#4cc9f0", "#f72585", "#90be6d", "#f9c74f", "#b388ff"]; // one per level, cycled

const DIVERGING = new Set(["BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral"]);

// --- Data: generator specs ---------------------------------------------------------------------

export type VolumeSpec =
  | { dataset: "hydrogen"; n?: number; l?: number; m?: number; size?: number; extent?: number; quantity?: "psi" | "density" | "log10_density" }
  | { dataset: "hydrogen_spherical"; n?: number; l?: number; m?: number; nr?: number; ntheta?: number; nphi?: number; rmin?: number; rmax?: number }
  | { dataset: "breathing"; size?: number; frames?: number }
  | { dataset: "tight_binding"; size?: number }
  | { dataset: "fermi"; size?: number };

interface AxisMeta { lo: number; hi: number; n: number; geometric?: boolean; label?: string; unit?: string }

/** What a spec makes: shape `[z, y, x]`, the axes (x, y, z), leading slices, and the
 * generator's uniforms. All small: the voxels themselves are only ever made on the GPU. */
export interface VolumeMeta {
  kind: number;
  shape: [number, number, number];
  slices: number; // leading dimension (4D stacks), 1 otherwise
  axes: [AxisMeta, AxisMeta, AxisMeta]; // x, y, z
  sameUnit: boolean; // all three axes in one unit: box sized by them (`withPhysicalAspect`)
  params: number[]; // n, l, m, quantity
  norm: [number, number]; // hydrogen radial and angular normalisation (f64 on the CPU)
}

const factorial = (k: number) => { let f = 1; for (let i = 2; i <= k; i += 1) f *= i; return f; };
const linspaceAxis = (lo: number, hi: number, n: number, label?: string, unit?: string): AxisMeta => ({ lo, hi, n, label, unit });

export function volumeMeta(spec: VolumeSpec): VolumeMeta {
  const hyd = (n: number, l: number, m: number): [number, number] => {
    if (!(Number.isInteger(n) && Number.isInteger(l) && Number.isInteger(m) && n >= 1 && l >= 0 && l < n && Math.abs(m) <= l)) {
      throw new Error(`hydrogen: need integers n ≥ 1, 0 ≤ l < n, |m| ≤ l (got ${n}, ${l}, ${m})`);
    }
    return [
      Math.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))),
      Math.sqrt((2 * l + 1) / (4 * Math.PI) * factorial(l - Math.abs(m)) / factorial(l + Math.abs(m))),
    ];
  };
  const size = (s: number | undefined, d: number) => {
    const v = s ?? d;
    if (!(Number.isInteger(v) && v >= 2 && v <= 256)) throw new Error(`volume size must be an integer in [2, 256] (got ${v})`);
    return v;
  };
  switch (spec.dataset) {
    case "hydrogen": {
      const { n = 3, l = 2, m = 1, extent = 20, quantity = "psi" } = spec;
      const s = size(spec.size, 56);
      const q = ["psi", "density", "log10_density"].indexOf(quantity);
      if (q < 0) throw new Error(`hydrogen: unknown quantity ${quantity}`);
      const ax = (name: string) => linspaceAxis(-extent, extent, s, name, "a₀");
      return { kind: 0, shape: [s, s, s], slices: 1, axes: [ax("x"), ax("y"), ax("z")], sameUnit: true, params: [n, l, m, q], norm: hyd(n, l, m) };
    }
    case "hydrogen_spherical": {
      const { n = 3, l = 1, m = 0, rmin = 0.05, rmax = 20 } = spec;
      const nr = size(spec.nr, 40), nt = size(spec.ntheta, 32), np = size(spec.nphi, 32);
      return {
        kind: 1, shape: [np, nt, nr], slices: 1, sameUnit: false, params: [n, l, m, 3], norm: hyd(n, l, m),
        axes: [{ lo: rmin, hi: rmax, n: nr, geometric: true, label: "𝑟", unit: "a₀" }, linspaceAxis(0, Math.PI, nt, "𝜃", "rad"), linspaceAxis(-Math.PI, Math.PI, np, "𝜑", "rad")],
      };
    }
    case "breathing": {
      const s = size(spec.size, 32);
      const frames = spec.frames ?? 8;
      if (!(Number.isInteger(frames) && frames >= 1 && frames <= 1024)) throw new Error("breathing: frames must be in [1, 1024]");
      const ax = linspaceAxis(-1, 1, s);
      return { kind: 2, shape: [s, s, s], slices: frames, axes: [ax, ax, ax], sameUnit: false, params: [0, 0, 0, 0], norm: [0, 0] };
    }
    case "tight_binding": {
      const s = size(spec.size, 40);
      const ax = (name: string) => linspaceAxis(-Math.PI, Math.PI, s, name, "1/a");
      return { kind: 3, shape: [s, s, s], slices: 1, axes: [ax("kx"), ax("ky"), ax("kz")], sameUnit: true, params: [0, 0, 0, 0], norm: [0, 0] };
    }
    case "fermi": {
      const s = size(spec.size, 64);
      const ax = (name: string) => linspaceAxis(-Math.PI, Math.PI, s, name, "1/a");
      return { kind: 4, shape: [s, s, s], slices: 1, axes: [ax("𝑘ₓ"), ax("𝑘𝑦"), ax("𝑘𝑧")], sameUnit: true, params: [0, 0, 0, 0], norm: [0, 0] };
    }
    default:
      throw new Error(`Unknown volume dataset ${(spec as { dataset: unknown }).dataset}`);
  }
}

/** Coordinates of an axis (numpy's linspace / geomspace, f64). */
export function axisValues(a: AxisMeta): Float64Array {
  return Float64Array.from({ length: a.n }, (_, i) => {
    const f = a.n > 1 ? i / (a.n - 1) : 0;
    return a.geometric ? 10 ** (Math.log10(a.lo) + f * (Math.log10(a.hi) - Math.log10(a.lo))) : a.lo + f * (a.hi - a.lo);
  });
}

const GEN_WGSL = /* wgsl */ `
  struct G {
    kind: u32, n: i32, l: i32, m: i32,
    quantity: u32, slice: u32, slices: u32, pad: u32,
    dims: vec4u,
    lo: vec4f,
    hi: vec4f,
    geo: vec4u,
    norm: vec4f,
  }
  @group(0) @binding(0) var<uniform> g: G;
  @group(0) @binding(1) var out: texture_storage_3d<r32float, write>;

  const PI = 3.14159265358979;

  fn coord(axis: u32, i: u32) -> f32 {
    let f = f32(i) / f32(max(g.dims[axis] - 1u, 1u));
    if (g.geo[axis] != 0u) {
      return exp2(log2(g.lo[axis]) + f * (log2(g.hi[axis]) - log2(g.lo[axis])));
    }
    return g.lo[axis] + f * (g.hi[axis] - g.lo[axis]);
  }

  // Generalised Laguerre L_n^a(x), associated Legendre P_l^m (m >= 0, no Condon-Shortley phase), and
  // the real spherical harmonics built from them (f64 CPU references: tools/check_volume.ts)
  fn genLaguerre(n: i32, a: f32, x: f32) -> f32 {
    var l0 = 1.0;
    var l1 = 1.0 + a - x;
    if (n == 0) { return l0; }
    for (var k = 1; k < n; k++) {
      let next = ((f32(2 * k + 1) + a - x) * l1 - (f32(k) + a) * l0) / f32(k + 1);
      l0 = l1;
      l1 = next;
    }
    return l1;
  }

  fn assocLegendre(l: i32, m: i32, x: f32) -> f32 {
    var pmm = 1.0;
    let somx2 = sqrt(max(1.0 - x * x, 0.0));
    var fact = 1.0;
    for (var i = 0; i < m; i++) {
      pmm = pmm * fact * somx2;
      fact += 2.0;
    }
    if (l == m) { return pmm; }
    var pmmp1 = x * f32(2 * m + 1) * pmm;
    if (l == m + 1) { return pmmp1; }
    for (var ll = m + 2; ll <= l; ll++) {
      let next = (x * f32(2 * ll - 1) * pmmp1 - f32(ll + m - 1) * pmm) / f32(ll - m);
      pmm = pmmp1;
      pmmp1 = next;
    }
    return pmmp1;
  }

  fn hydrogen(x: f32, y: f32, z: f32) -> f32 {
    let r = sqrt(x * x + y * y + z * z);
    let cosTheta = select(1.0, z / r, r > 0.0);
    let phi = atan2(y, x);
    let rho = 2.0 * r / f32(g.n);
    var rhoL = 1.0;
    for (var i = 0; i < g.l; i++) { rhoL *= rho; }
    let radial = g.norm.x * exp(-rho / 2.0) * rhoL * genLaguerre(g.n - g.l - 1, f32(2 * g.l + 1), rho);
    let p = assocLegendre(g.l, abs(g.m), cosTheta);
    var ang = g.norm.y * p;
    if (g.m > 0) { ang *= sqrt(2.0) * cos(f32(g.m) * phi); }
    if (g.m < 0) { ang *= sqrt(2.0) * sin(f32(-g.m) * phi); }
    return radial * ang;
  }

  @compute @workgroup_size(4, 4, 4)
  fn generate(@builtin(global_invocation_id) id: vec3u) {
    if (any(id >= g.dims.xyz)) { return; }
    let x = coord(0u, id.x);
    let y = coord(1u, id.y);
    let z = coord(2u, id.z);
    var v = 0.0;
    switch (g.kind) {
      case 0u: { // Cartesian orbital: psi, density (+1e-14), log10 density
        let psi = hydrogen(x, y, z);
        if (g.quantity == 0u) { v = psi; }
        else if (g.quantity == 1u) { v = psi * psi + 1e-14; }
        else { v = log2(psi * psi + 1e-14) * 0.30102999566398120; }
      }
      case 1u: { // (phi, theta, r) grid: |psi|^2
        let r = x;
        let theta = y;
        let ph = z;
        let psi = hydrogen(r * sin(theta) * cos(ph), r * sin(theta) * sin(ph), r * cos(theta));
        v = psi * psi;
      }
      case 2u: { // breathing Gaussian at t = slice / slices
        let t = f32(g.slice) / f32(g.slices);
        v = exp(-(x * x + y * y + z * z) / (0.1 + 0.08 * sin(2.0 * PI * t))) * cos(6.0 * x);
      }
      case 3u: { // nearest-neighbour tight-binding band
        v = -2.0 * (cos(x) + cos(y) + cos(z));
      }
      default: { // with next-nearest-neighbour hopping (Fermi surface)
        let cx = cos(x);
        let cy = cos(y);
        let cz = cos(z);
        v = 1.5 - 2.0 * (cx + cy + cz) + 1.2 * (cx * cy + cy * cz + cz * cx);
      }
    }
    textureStore(out, id, vec4f(v, 0.0, 0.0, 0.0));
  }
`;

// --- Marching cubes (compute) ------------------------------------------------------------------

/* Corner n of a cell at (n & 1, (n >> 1) & 1, (n >> 2) & 1); edges and faces as lib/marching_cubes.ts
 * (`CUBE_EDGES`, `CUBE_FACES`), whose triangulations (`getCaseTriangles`) fill the tables. */
const MC_WGSL = /* wgsl */ `
  struct M {
    dims: vec4u, // nx, ny, nz, scale (0 linear, 1 log, 2 symlog, 3 sqrt)
    level: f32, symlogConstant: f32, cells: u32, pad: u32,
  }
  @group(0) @binding(0) var<uniform> u: M;
  @group(0) @binding(1) var data: texture_3d<f32>;
  @group(0) @binding(2) var<storage, read> cases: array<vec4u>; // tri offset, count, loop offset, loop lengths
  @group(0) @binding(3) var<storage, read> tris: array<u32>; // 3 x 4 bits
  @group(0) @binding(4) var<storage, read> loops: array<u32>;
  @group(0) @binding(5) var<storage, read> centres: array<f32>; // relative voxel centres: x, y then z
  @group(0) @binding(6) var<storage, read_write> counts: array<u32>; // per cell, then offsets
  @group(0) @binding(7) var<storage, read_write> verts: array<vec4f>; // position, normal per vertex

  const EDGES = array<vec2u, 12>(
    vec2u(0, 1), vec2u(2, 3), vec2u(4, 5), vec2u(6, 7),
    vec2u(0, 2), vec2u(1, 3), vec2u(4, 6), vec2u(5, 7),
    vec2u(0, 4), vec2u(1, 5), vec2u(2, 6), vec2u(3, 7));
  const FACES = array<vec4u, 6>(
    vec4u(0, 4, 6, 2), vec4u(1, 3, 7, 5), vec4u(0, 1, 5, 4),
    vec4u(2, 6, 7, 3), vec4u(0, 2, 3, 1), vec4u(4, 5, 7, 6));
  const SNAP = 1e-6;

  fn corner(n: u32) -> vec3u { return vec3u(n & 1u, (n >> 1u) & 1u, (n >> 2u) & 1u); }

  fn isValid(v: f32) -> bool {
    let finite = v == v && abs(v) <= 3.4e38;
    switch (u.dims.w) {
      case 1u: { return finite && v > 0.0; }
      case 3u: { return finite && v >= 0.0; }
      default: { return finite; }
    }
  }

  fn tf(v: f32) -> f32 {
    switch (u.dims.w) {
      case 1u: { return log2(v) * 0.30102999566398120; }
      case 2u: { return sign(v) * log(1.0 + abs(v) / u.symlogConstant); }
      case 3u: { return sqrt(v); }
      default: { return v; }
    }
  }

  fn value(p: vec3u) -> f32 { return textureLoad(data, p, 0).r; }

  // Gradient component: central differences, one-sided at borders or next to invalid values
  fn partial(p: vec3u, axis: u32) -> f32 {
    let n = u.dims[axis];
    let centre = value(p);
    var lo = p;
    var hi = p;
    if (p[axis] > 0u) { lo[axis] -= 1u; }
    if (p[axis] < n - 1u) { hi[axis] += 1u; }
    var vLo = value(lo);
    var vHi = value(hi);
    if (!isValid(vLo)) { lo = p; vLo = centre; }
    if (!isValid(vHi)) { hi = p; vHi = centre; }
    if (hi[axis] == lo[axis]) { return 0.0; }
    return (vHi - vLo) / f32(hi[axis] - lo[axis]);
  }
  fn gradientAt(p: vec3u) -> vec3f { return vec3f(partial(p, 0u), partial(p, 1u), partial(p, 2u)); }
  fn unit(v: vec3f) -> vec3f { let l = length(v); return select(vec3f(0.0, 0.0, 1.0), v / l, l > 0.0); }

  // Relative position (0 to 1) in the box of an index coordinate, and the slope there
  fn rel(f: f32, axis: u32) -> vec2f {
    let n = u.dims[axis];
    var base = 0u;
    if (axis > 0u) { base += u.dims.x; }
    if (axis > 1u) { base += u.dims.y; }
    let i0 = min(u32(max(floor(f), 0.0)), n - 2u);
    let c0 = centres[base + i0];
    let c1 = centres[base + i0 + 1u];
    return vec2f(mix(c0, c1, f - f32(i0)), c1 - c0);
  }

  var<private> vpos: array<vec3f, 14>;
  var<private> vnrm: array<vec3f, 14>;
  var<private> vid: array<u32, 14>;
  var<private> vdone: array<bool, 14>;

  // Vertex on cell edge e: crossing in the scale's space, snapped to grid points within 1e-6
  fn edgeVertex(cell: vec3u, e: u32, cv: array<f32, 8>) {
    if (vdone[e]) { return; }
    vdone[e] = true;
    let ca = EDGES[e].x;
    let cb = EDGES[e].y;
    let sa = tf(cv[ca]);
    let sb = tf(cv[cb]);
    var t = (tf(u.level) - sa) / (sb - sa);
    t = select(0.5, clamp(t, 0.0, 1.0), t == t && abs(t) <= 3.4e38);
    let pa = cell + corner(ca);
    let pb = cell + corner(cb);
    if (t <= SNAP || t >= 1.0 - SNAP) {
      let c = select(ca, cb, t >= 1.0 - SNAP);
      let p = cell + corner(c);
      vpos[e] = vec3f(p);
      vnrm[e] = unit(gradientAt(p));
      vid[e] = 16u + c;
      return;
    }
    vpos[e] = mix(vec3f(pa), vec3f(pb), t);
    vnrm[e] = unit(mix(gradientAt(pa), gradientAt(pb), t));
    vid[e] = e;
  }

  fn writeVertex(o: u32, k: u32) {
    let p = vpos[k];
    let x = rel(p.x, 0u);
    let y = rel(p.y, 1u);
    let z = rel(p.z, 2u);
    verts[2u * o] = vec4f(x.x, y.x, z.x, 1.0);
    verts[2u * o + 1u] = vec4f(vnrm[k] / vec3f(x.y, y.y, z.y), 0.0); // cf. remapIsosurface
  }

  fn march(index: u32, emit: bool) {
    if (index >= u.cells) { return; }
    let cx = u.dims.x - 1u;
    let cy = u.dims.y - 1u;
    let cell = vec3u(index % cx, (index / cx) % cy, index / (cx * cy));
    var cv: array<f32, 8>;
    var caseIndex = 0u;
    for (var n = 0u; n < 8u; n++) {
      let v = value(cell + corner(n));
      if (!isValid(v)) { if (!emit) { counts[index] = 0u; } return; }
      cv[n] = v;
      if (v < u.level) { caseIndex |= 1u << n; }
    }
    if (caseIndex == 0u || caseIndex == 255u) { if (!emit) { counts[index] = 0u; } return; }
    var connected = 0u;
    for (var f = 0u; f < 6u; f++) {
      let c = FACES[f];
      let i0 = (caseIndex >> c.x) & 1u;
      let i1 = (caseIndex >> c.y) & 1u;
      let i2 = (caseIndex >> c.z) & 1u;
      let i3 = (caseIndex >> c.w) & 1u;
      if (i0 == i2 && i1 == i3 && i0 != i1) {
        let a = cv[c.x]; let b = cv[c.y]; let cc = cv[c.z]; let d = cv[c.w];
        let denom = a + cc - b - d;
        if (denom != 0.0 && (a * cc - b * d) / denom < u.level) { connected |= 1u << f; }
      }
    }
    let info = cases[caseIndex | (connected << 8u)];
    for (var k = 0u; k < 14u; k++) { vdone[k] = false; }
    // Edge vertices, then the centroids of long loops
    for (var t = 0u; t < info.y; t++) {
      let tri = tris[info.x + t];
      for (var j = 0u; j < 3u; j++) {
        let e = (tri >> (4u * j)) & 15u;
        if (e < 12u) { edgeVertex(cell, e, cv); }
      }
    }
    var lo = info.z;
    for (var c = 0u; c < 2u; c++) {
      let len = (info.w >> (8u * c)) & 255u;
      if (len == 0u) { continue; }
      var p = vec3f(0.0);
      var nr = vec3f(0.0);
      for (var i = 0u; i < len; i++) {
        let e = loops[lo + i];
        edgeVertex(cell, e, cv);
        p += vpos[e];
        nr += vnrm[e];
      }
      vpos[12u + c] = p / f32(len);
      let l = length(nr);
      vnrm[12u + c] = select(nr, nr / l, l > 0.0);
      vid[12u + c] = 12u + c;
      lo += len;
    }
    var o = select(0u, counts[index], emit);
    var kept = 0u;
    for (var t = 0u; t < info.y; t++) {
      let tri = tris[info.x + t];
      let a = tri & 15u;
      let b = (tri >> 4u) & 15u;
      let c = (tri >> 8u) & 15u;
      if (vid[a] == vid[b] || vid[b] == vid[c] || vid[a] == vid[c]) { continue; } // collapsed by snapping
      kept += 1u;
      if (emit) {
        writeVertex(3u * o, a);
        writeVertex(3u * o + 1u, b);
        writeVertex(3u * o + 2u, c);
        o += 1u;
      }
    }
    if (!emit) { counts[index] = kept; }
  }

  @compute @workgroup_size(64)
  fn count(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) nw: vec3u) {
    march(id.x + id.y * nw.x * 64u, false);
  }

  @compute @workgroup_size(64)
  fn emit(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) nw: vec3u) {
    march(id.x + id.y * nw.x * 64u, true);
  }
`;

/* Exclusive prefix sum, 256 per workgroup (Hillis-Steele in workgroup memory), block sums out */
const SCAN_WGSL = /* wgsl */ `
  struct S { n: u32, pad0: u32, pad1: u32, pad2: u32 }
  @group(0) @binding(0) var<uniform> s: S;
  @group(0) @binding(1) var<storage, read_write> data: array<u32>;
  @group(0) @binding(2) var<storage, read_write> sums: array<u32>;
  var<workgroup> tmp: array<u32, 256>;

  @compute @workgroup_size(256)
  fn scan(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let block = wg.x + wg.y * nw.x;
    let i = block * 256u + li;
    let v = select(0u, data[min(i, s.n - 1u)], i < s.n);
    tmp[li] = v;
    workgroupBarrier();
    for (var d = 1u; d < 256u; d <<= 1u) {
      var add = 0u;
      if (li >= d) { add = tmp[li - d]; }
      workgroupBarrier();
      tmp[li] += add;
      workgroupBarrier();
    }
    if (i < s.n) { data[i] = tmp[li] - v; }
    if (li == 255u && block * 256u < s.n) { sums[block] = tmp[255]; }
  }

  @compute @workgroup_size(256)
  fn add(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nw: vec3u) {
    let block = wg.x + wg.y * nw.x;
    let i = block * 256u + li;
    if (i < s.n) { data[i] += sums[block]; }
  }
`;

/** The triangulation of every case (bits 0-7: corners below the level) and connected
 * ambiguous faces (bits 8-13), packed for the shader. */
function caseTables(): { cases: Uint32Array; tris: Uint32Array; loops: Uint32Array } {
  const cases = new Uint32Array(256 * 64 * 4);
  const tris: number[] = [];
  const loops: number[] = [];
  for (let key = 0; key < 256 * 64; key += 1) {
    const { triangles, centroidLoops } = getCaseTriangles(key & 255, key >> 8);
    cases[key * 4] = tris.length;
    cases[key * 4 + 1] = triangles.length / 3;
    cases[key * 4 + 2] = loops.length;
    cases[key * 4 + 3] = (centroidLoops[0]?.length ?? 0) | ((centroidLoops[1]?.length ?? 0) << 8);
    for (let t = 0; t < triangles.length; t += 3) {
      const e = [0, 1, 2].map((j) => (triangles[t + j] >= CENTROID_BASE ? 12 + triangles[t + j] - CENTROID_BASE : triangles[t + j]));
      tris.push(e[0] | (e[1] << 4) | (e[2] << 8));
    }
    for (const loop of centroidLoops) loops.push(...loop);
  }
  return { cases, tris: Uint32Array.from(tris), loops: Uint32Array.from(loops.length ? loops : [0]) };
}

// --- Drawing ------------------------------------------------------------------------------------

const COMMON_WGSL = /* wgsl */ `
  fn toSrgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
  }
`;

/* Ray marcher */
const VOLUME_WGSL = /* wgsl */ `
  struct U {
    model: mat4x4f, // box (-0.5..0.5) to clip
    depthToBox: mat4x4f, // NDC + depth to box
    camLocal: vec4f, // camera in box coordinates
    dims: vec4f, // xyz, w: samplesPerVoxel
    stepDims: vec4f, // xyz, w: opacity
    p0: vec4f, // domainMin, oneOverRange, gammaExponent, symlogConstant
    p1: vec4f, // normRevertFactor, valueOffset, threshold, oneOverRamp
    p2: vec4f, // colorStart, oneOverColorSpan, hasSurfaceDepth, mode (0 MIP, 1 DVR, 2 Iso)
    p3: vec4f, // scale (0 linear, 1 log, 2 symlog, 3 sqrt), lookup offsets x, y, z (-1: none)
    size: vec4f, // target width, height; diverging: 1 / largest distance from the centre (0: not), centre
    depthScale: vec4f, // surface depth texels per target pixel (xy): 1, or about 2 at half resolution; z: 1 for forward-difference gradients
  }
  @group(0) @binding(0) var<uniform> u: U;
  @group(0) @binding(1) var volume: texture_3d<f32>;
  @group(0) @binding(2) var volumeSampler: sampler;
  @group(0) @binding(3) var<storage, read> colorMap: array<u32>; // 256 RGBA8 (sRGB)
  @group(0) @binding(4) var<storage, read> lookups: array<f32>;
  @group(0) @binding(5) var surfaceDepth: texture_depth_2d;

  const oneOverLog10 = 0.43429448190325176;
  const MAX_STEPS = 4096;
  // Specialised per pipeline (as #defines): non-uniform axes, mode (0 MIP, 1 DVR, 2 Iso), scale
  // (0 linear, 1 log, 2 symlog, 3 sqrt)
  const LOOKUP_X = $LOOKUP_X;
  const LOOKUP_Y = $LOOKUP_Y;
  const LOOKUP_Z = $LOOKUP_Z;
  const MODE = $MODEu;
  const SCALE = $SCALEu;

  struct V { @builtin(position) position: vec4f, @location(0) local: vec3f }

  // Back faces of the box (BackSide): rays are cast even from inside it
  const CUBE = array<vec3f, 8>(
    vec3f(-0.5, -0.5, -0.5), vec3f(0.5, -0.5, -0.5), vec3f(-0.5, 0.5, -0.5), vec3f(0.5, 0.5, -0.5),
    vec3f(-0.5, -0.5, 0.5), vec3f(0.5, -0.5, 0.5), vec3f(-0.5, 0.5, 0.5), vec3f(0.5, 0.5, 0.5));
  const CUBE_TRIS = array<u32, 36>(
    0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, // -z, +z
    0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, // -y, +y
    0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5); // -x, +x

  @vertex
  fn vertexMain(@builtin(vertex_index) i: u32) -> V {
    let p = CUBE[CUBE_TRIS[i]];
    var out: V;
    out.position = u.model * vec4f(p, 1.0);
    out.local = p;
    return out;
  }

  fn isnan(v: f32) -> bool { return !(v == v); }

  fn isSupported(value: f32) -> bool {
    if (SCALE == 1u) { return value > 0.0; }
    if (SCALE == 3u) { return value >= 0.0; }
    return true;
  }

  fn log1p(x: f32) -> f32 { return select(log(1.0 + x), x * (1.0 - x * (0.5 - x * (1.0 / 3.0 - x * 0.25))), x < 1e-2); }

  fn scale(value: f32) -> f32 {
    switch (SCALE) {
      case 1u: { return log(value) * oneOverLog10; }
      case 2u: { return sign(value) * log1p(abs(value) / u.p0.w) * oneOverLog10; }
      case 3u: { return sqrt(value); }
      default: { return value; }
    }
  }

  fn lut(t: f32) -> vec3f { // nearest, clamped (DataTexture defaults)
    let c = unpack4x8unorm(colorMap[u32(clamp(floor(t * 256.0), 0.0, 255.0))]);
    return c.rgb;
  }

  fn thresholdColor(position: f32) -> vec3f {
    let t = pow(clamp((position - u.p2.x) * u.p2.y, 0.0, 1.0), u.p0.z);
    return lut(t);
  }

  fn knot(offset: i32, k: i32) -> f32 { return lookups[offset + k]; }

  fn lookUp(offset: i32, voxels: f32, pos: f32) -> f32 {
    let count = 2 * i32(voxels) + 1;
    var lo = 0;
    var hi = count - 2;
    for (var k = 0; k < 24; k++) {
      if (lo >= hi) { break; }
      let mid = (lo + hi + 1) / 2;
      if (knot(offset, mid) <= pos) { lo = mid; } else { hi = mid - 1; }
    }
    let start = knot(offset, lo);
    let end = knot(offset, lo + 1);
    let t = clamp((pos - start) / max(end - start, 1e-30), 0.0, 1.0);
    return (f32(lo) + t) / f32(count - 1);
  }

  fn toTexCoords(q: vec3f) -> vec3f {
    var p = q;
    if (LOOKUP_X) { p.x = lookUp(i32(u.p3.y), u.dims.x, p.x); }
    if (LOOKUP_Y) { p.y = lookUp(i32(u.p3.z), u.dims.y, p.y); }
    if (LOOKUP_Z) { p.z = lookUp(i32(u.p3.w), u.dims.z, p.z); }
    return p;
  }

  fn sampleRaw(p: vec3f) -> vec2f {
    let value = textureSampleLevel(volume, volumeSampler, toTexCoords(p), 0.0).r;
    return select(vec2f(0.0), vec2f(value, 1.0), !isnan(value));
  }

  fn linearPosition(raw: f32) -> f32 {
    let value = raw * u.p1.x + u.p1.y;
    if (isnan(value) || !isSupported(value)) { return -1.0; }
    let scaledValue = scale(value);
    if (u.p0.y == 0.0) { return select(0.0, 1.0, scaledValue >= u.p0.x); }
    return clamp(u.p0.y * (scaledValue - u.p0.x), 0.0, 1.0);
  }

  fn normalizeValue(raw: f32) -> f32 {
    let position = linearPosition(raw);
    return select(pow(position, u.p0.z), -1.0, position < 0.0);
  }

  fn sampleVolume(p: vec3f) -> f32 {
    let s = sampleRaw(p);
    return select(normalizeValue(s.x), -1.0, s.y < 0.5);
  }

  fn density(p: vec3f) -> f32 {
    let s = sampleRaw(p);
    return max(normalizeValue(s.x), 0.0) * smoothstep(0.25, 0.75, s.y);
  }

  fn gradient(p: vec3f) -> vec3f {
    let d = 1.0 / u.dims.xyz;
    return 0.5 * vec3f(
      density(p + vec3f(d.x, 0.0, 0.0)) - density(p - vec3f(d.x, 0.0, 0.0)),
      density(p + vec3f(0.0, d.y, 0.0)) - density(p - vec3f(0.0, d.y, 0.0)),
      density(p + vec3f(0.0, 0.0, d.z)) - density(p - vec3f(0.0, 0.0, d.z)));
  }

  /* Forward differences from the density \`d0\` at \`p\`: half the samples of \`gradient\` (drags) */
  fn forwardGradient(p: vec3f, d0: f32) -> vec3f {
    let d = 1.0 / u.dims.xyz;
    return vec3f(
      density(p + vec3f(d.x, 0.0, 0.0)) - d0,
      density(p + vec3f(0.0, d.y, 0.0)) - d0,
      density(p + vec3f(0.0, 0.0, d.z)) - d0);
  }

  fn shade(color: vec3f, grad: vec3f, viewDir: vec3f, minGradient: f32) -> vec3f {
    let len = length(grad);
    if (len < 1e-6) { return color; }
    let nDotV = abs(dot(grad / len, viewDir));
    let lit = color * (0.3 + 0.8 * nDotV) + 0.3 * pow(nDotV, 48.0);
    return mix(color, lit, smoothstep(0.0, minGradient, len));
  }

  fn intersectBox(origin: vec3f, dir: vec3f) -> vec2f {
    let invDir = 1.0 / dir;
    let t0 = -origin * invDir;
    let t1 = (1.0 - origin) * invDir;
    let tMin = min(t0, t1);
    let tMax = max(t0, t1);
    return vec2f(max(max(tMin.x, tMin.y), tMin.z), min(min(tMax.x, tMax.y), tMax.z));
  }

  @fragment
  fn fragmentMain(in: V) -> @location(0) vec4f {
    let origin = u.camLocal.xyz + 0.5;
    let dir = normalize(in.local - u.camLocal.xyz);
    var bounds = intersectBox(origin, dir);
    bounds.x = max(bounds.x, 0.0);

    if (u.p2.z > 0.5) { // stop rays at the nearest opaque surface
      let depth = textureLoad(surfaceDepth, vec2i(in.position.xy * u.depthScale.xy), 0);
      if (depth < 1.0) {
        let uv = in.position.xy / u.size.xy;
        let hit = u.depthToBox * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let hitPoint = hit.xyz / hit.w + 0.5;
        bounds.y = min(bounds.y, dot(hitPoint - origin, dir));
      }
    }
    if (bounds.x >= bounds.y) { discard; }

    let voxelDir = dir * u.dims.xyz;
    let viewDir = -normalize(voxelDir);
    let rayLength = bounds.y - bounds.x;
    var dt = 1.0 / (length(dir * u.stepDims.xyz) * u.dims.w);
    var steps = i32(ceil(rayLength / dt));
    if (steps > MAX_STEPS) {
      steps = MAX_STEPS;
      dt = rayLength / f32(MAX_STEPS);
    }
    let jitter = fract(52.9829189 * fract(dot(in.position.xy, vec2f(0.06711056, 0.00583715))));
    var p = origin + dir * (bounds.x + jitter * dt);
    let delta = dir * dt;
    let threshold = u.p1.z;
    let mode = MODE;

    if (mode == 0u) { // MIP
      var maxValue = -1.0;
      for (var i = 0; i < steps; i++) {
        maxValue = max(maxValue, sampleVolume(p));
        if (maxValue >= 1.0) { break; }
        p += delta;
      }
      if (maxValue < threshold) { discard; }
      return vec4f(thresholdColor(pow(maxValue, 1.0 / u.p0.z)), 1.0);
    }
    if (mode == 1u) { // DVR
      let alphaExponent = dt * length(voxelDir);
      var acc = vec4f(0.0);
      let diverging = u.size.z > 0.0;
      for (var i = 0; i < steps; i++) {
        let s = sampleRaw(p);
        let covered = s.y;
        let position = select(-1.0, linearPosition(s.x), covered > 0.0);
        let value = select(pow(position, u.p0.z), -1.0, position < 0.0);
        // Opacity grows with the value, or with its distance from the centre (diverging)
        let ramp = select(value, select(-1.0, abs(value - u.size.w) * u.size.z, value >= 0.0), diverging);
        if (ramp > threshold) {
          let voxelAlpha = clamp(u.stepDims.w * (ramp - threshold) * u.p1.w, 0.0, 0.999);
          let alpha = 1.0 - pow(1.0 - voxelAlpha, alphaExponent * covered);
          var color = thresholdColor(position);
          if (voxelAlpha > 0.002) {
            var grad: vec3f;
            if (u.depthScale.z > 0.5) {
              grad = forwardGradient(p, max(value, 0.0) * smoothstep(0.25, 0.75, covered));
            } else {
              grad = gradient(p);
            }
            color = shade(color, grad, viewDir, 0.01);
          }
          acc += (1.0 - acc.a) * vec4f(color * alpha, alpha);
          if (acc.a > 0.98) { break; }
        }
        p += delta;
      }
      if (acc.a < 0.002) { discard; }
      return acc;
    }
    var prev = p; // Iso
    for (var i = 0; i < steps; i++) {
      if (sampleVolume(p) > threshold) {
        var lo = prev;
        var hi = p;
        for (var j = 0; j < 6; j++) {
          let mid = 0.5 * (lo + hi);
          if (sampleVolume(mid) > threshold) { hi = mid; } else { lo = mid; }
        }
        return vec4f(shade(lut(threshold), gradient(hi), viewDir, 1e-6), 1.0);
      }
      prev = p;
      p += delta;
    }
    discard;
  }
`;

/* Isosurface meshes: a physically based material (roughness 0.55, metalness 0.05, both sides)
 * under the headlight (2.4) and ambient light (0.35), GGX specular without multiscattering */
const ISO_WGSL = /* wgsl */ `
  struct I {
    clip: mat4x4f, // box to clip
    model: mat4x4f, // box to world
    eye: vec4f, // world
    light: vec4f, // direction to the light, world
    boxSize: vec4f,
    color: vec4f, // linear
  }
  @group(0) @binding(0) var<uniform> u: I;
  @group(0) @binding(1) var<storage, read> verts: array<vec4f>;

  struct V { @builtin(position) position: vec4f, @location(0) world: vec3f, @location(1) normal: vec3f }

  @vertex
  fn vertexMain(@builtin(vertex_index) i: u32) -> V {
    let local = verts[2u * i].xyz - 0.5;
    var out: V;
    out.position = u.clip * vec4f(local, 1.0);
    out.world = (u.model * vec4f(local, 1.0)).xyz;
    out.normal = (u.model * vec4f(verts[2u * i + 1u].xyz / u.boxSize.xyz, 0.0)).xyz; // model is rotation x scale: scale out again
    return out;
  }

  @vertex
  fn vertexDepth(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return u.clip * vec4f(verts[2u * i].xyz - 0.5, 1.0);
  }

  const PI = 3.141592653589793;
  fn F_Schlick(f0: vec3f, f90: f32, dotVH: f32) -> vec3f {
    let fresnel = exp2((-5.55473 * dotVH - 6.98316) * dotVH);
    return f0 * (1.0 - fresnel) + f90 * fresnel;
  }

  @fragment
  fn fragmentMain(in: V) -> @location(0) vec4f {
    let viewDir = normalize(u.eye.xyz - in.world);
    var n = normalize(in.normal);
    let dxy = max(abs(dpdx(n)), abs(dpdy(n)));
    n = select(n, -n, dot(n, viewDir) < 0.0); // two-sided
    let base = u.color.rgb;
    let metalness = 0.05;
    let diffuse = base * (1.0 - metalness);
    let f0 = mix(vec3f(0.04), base, metalness);
    let roughness = min(max(0.55, 0.0525) + max(max(dxy.x, dxy.y), dxy.z), 1.0);
    let alpha = roughness * roughness;
    let l = u.light.xyz;
    let h = normalize(l + viewDir);
    let dotNL = clamp(dot(n, l), 0.0, 1.0);
    let dotNV = clamp(dot(n, viewDir), 0.0, 1.0);
    let dotNH = clamp(dot(n, h), 0.0, 1.0);
    let dotVH = clamp(dot(viewDir, h), 0.0, 1.0);
    let a2 = alpha * alpha;
    let gv = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);
    let gl = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
    let V = 0.5 / max(gv + gl, 1e-6);
    let denom = dotNH * dotNH * (a2 - 1.0) + 1.0;
    let D = a2 / (PI * denom * denom);
    let specular = F_Schlick(f0, 1.0, dotVH) * V * D;
    let irradiance = dotNL * 2.4;
    let color = irradiance * (diffuse / PI + specular) + 0.35 * diffuse / PI;
    return vec4f(toSrgb(color), 1.0);
  }
`;

/* Segments of the box and axes, screen-space quads of a width in pixels */
const LINES_WGSL = /* wgsl */ `
  struct L { clip: mat4x4f, size: vec4f }
  struct Seg { a: vec4f, b: vec4f, color: vec4f, width: vec4f }
  @group(0) @binding(0) var<uniform> u: L;
  @group(0) @binding(1) var<storage, read> segs: array<Seg>;
  struct V { @builtin(position) position: vec4f, @location(0) @interpolate(flat) color: vec4f }

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> V {
    let s = segs[i];
    let ca = u.clip * vec4f(s.a.xyz, 1.0);
    let cb = u.clip * vec4f(s.b.xyz, 1.0);
    let pa = ca.xy / ca.w * u.size.xy * 0.5;
    let pb = cb.xy / cb.w * u.size.xy * 0.5;
    var d = pb - pa;
    let len = length(d);
    d = select(vec2f(1.0, 0.0), d / len, len > 1e-6);
    let nrm = vec2f(-d.y, d.x) * s.width.x * 0.5;
    var corners = array(vec2f(0, -1), vec2f(1, -1), vec2f(0, 1), vec2f(0, 1), vec2f(1, -1), vec2f(1, 1));
    let c = corners[v];
    let clip = select(ca, cb, c.x > 0.5);
    let offset = nrm * c.y / (u.size.xy * 0.5) * clip.w;
    var out: V;
    out.position = vec4f(clip.xy + offset, clip.z, clip.w);
    out.color = s.color;
    return out;
  }

  @fragment
  fn fragmentMain(in: V) -> @location(0) vec4f { return in.color; }
`;

// --- Control ------------------------------------------------------------------------------------

export const DEFAULTS = {
  width: 1200,
  height: 750,
  volume: { dataset: "hydrogen", n: 3, l: 2, m: 1 } as VolumeSpec,
  slice: 0, // leading dimension (4D)
  title: null as string | null, // default: generator's
  subtitle: "",
  colormap: "Inferno" as string, // or "auto": RdBu (symmetric) if the domain spans `center`, else Viridis
  invertColormap: false,
  scale: "linear" as "linear" | "log" | "symlog" | "sqrt" | "root",
  rootN: 3, // root scale: colours (and the DVR transfer) by t^(1/n) of the normalised value, n in 0.1..10 (step 0.05)
  symlogThreshold: null as number | null,
  valueTransform: null as "exp10" | "exp2" | null,
  domain: null as [number | null, number | null] | null,
  symmetric: false,
  center: null as number | null,
  mode: "DVR" as "DVR" | "MIP" | "Iso",
  opacity: 0.25,
  threshold: 0.1,
  quality: "medium" as "low" | "medium" | "high",
  interacting: false, // while the camera moves: half the samples, half resolution, cheaper shading
  showVolume: true,
  isoLevels: null as number[] | null, // default: generator's (Fermi level)
  isoColors: "levels" as "levels" | "colormap", // a colour per level, or the level through the colour map
  showBox: true,
  showAxes: true,
  valueLabel: null as string | null, // hover read-out name of the value
  up: "y" as "y" | "z", // box axis drawn up
  azimuth: 38.66, // from (1.4, 1, 1.75) in box space, y up
  elevation: 24.04,
  zoom: 1,
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;


interface Data {
  key: string;
  meta: VolumeMeta;
  slice: number;
  texture: GPUTexture;
  values: Float32Array; // read back: domain, hover, Fermi level
  dataDomain: [number, number];
  fermi?: number;
}

interface IsoMesh { level: number; key: string; buffer?: GPUBuffer; triangles: number; ms: number }

interface Label { text: string; at: Vec3; outward: Vec3; priority: number; axis: boolean }

/** Pixel bounds [l, t, r, b] of the box (-0.5..0.5) on screen, grown by `pad`, within the
 * frame (all of it if the box reaches behind the camera). */
function screenRect(clip: Mat4, w: number, h: number, pad: number): [number, number, number, number] {
  let [l, t, r, b] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < 8; i += 1) {
    const p = [i & 1, (i >> 1) & 1, (i >> 2) & 1].map((c) => c - 0.5);
    const cw = clip[3] * p[0] + clip[7] * p[1] + clip[11] * p[2] + clip[15];
    if (!(cw > 0)) return [0, 0, w, h];
    const x = ((clip[0] * p[0] + clip[4] * p[1] + clip[8] * p[2] + clip[12]) / cw + 1) / 2 * w;
    const y = (1 - (clip[1] * p[0] + clip[5] * p[1] + clip[9] * p[2] + clip[13]) / cw) / 2 * h;
    [l, t, r, b] = [Math.min(l, x), Math.min(t, y), Math.max(r, x), Math.max(b, y)];
  }
  return [Math.max(Math.floor(l) - pad, 0), Math.max(Math.floor(t) - pad, 0), Math.min(Math.ceil(r) + pad, w), Math.min(Math.ceil(b) + pad, h)];
}

/** Size of the half-resolution volume layer. */
const halfSize = (w: number, h: number): [number, number] => [Math.ceil(w / 2), Math.ceil(h / 2)];

const MAX_DISPATCH = 65535;
const dispatch2D = (groups: number): [number, number] => {
  const x = Math.min(groups, MAX_DISPATCH);
  return [Math.max(x, 1), Math.max(1, Math.ceil(groups / MAX_DISPATCH))];
};

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const hexLinear = (hex: string): [number, number, number] =>
  [0, 2, 4].map((i) => srgbToLinear(parseInt(hex.slice(1 + i, 3 + i), 16) / 255)) as [number, number, number];


/** Box axis → world: up "y" maps (x, y, z) to (x, −z, y) (the box's y up is world z). */
function boxToWorld(up: string): Mat4 {
  return up === "z" ? scaling(1, 1, 1) : rows(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
}

export class VolumePlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly frameReads: GPUBuffer[] = [];
  private readonly genPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly emitPipeline: GPUComputePipeline;
  private readonly scanPipeline: GPUComputePipeline;
  private readonly addPipeline: GPUComputePipeline;
  private readonly volumePipelines = new Map<string, GPURenderPipeline>();
  private readonly blendOver: GPUBlendState;
  private readonly isoPipeline: GPURenderPipeline;
  private readonly isoDepthPipeline: GPURenderPipeline;
  private readonly linePipelines: { depth: GPURenderPipeline; over: GPURenderPipeline };
  private readonly tables: { cases: GPUBuffer; tris: GPUBuffer; loops: GPUBuffer };
  private readonly sampler: GPUSampler;
  private readonly volumeUniforms: GPUBuffer;
  private readonly lineUniforms: GPUBuffer;
  private isoUniforms: GPUBuffer[] = [];
  private data?: Data;
  private readonly dataCache = new Map<string, Promise<Data>>();
  /** Data domain over every slice of a spec (and scale), apart from the slice textures. */
  private readonly domainCache = new Map<string, Promise<[number, number] | null>>();
  private meshes: IsoMesh[] = [];
  private colorKey?: string;
  private colorBuffer?: GPUBuffer;
  private lookupBuffer?: GPUBuffer;
  private lookupKey?: string;
  private targets?: { color: GPUTexture; depth: GPUTexture; isoDepth: GPUTexture; layer: GPUTexture; edges: GPUTexture; half: GPUTexture; w: number; h: number };
  private segBuffer?: GPUBuffer;
  private size: [number, number] = [1, 1];
  private scale = 1;
  private slider: RootSlider | null = null; // of the root scale, as drawn
  // Of the last frame, for the hover read-out and state.json
  private view!: {
    vpWorld: Mat4; boxToWorld: Mat4; boxSize: Vec3; camBox: Vec3; dirOf: (x: number, y: number) => Vec3;
    target: Vec3; distance: number; plot: [number, number, number, number];
  };
  private colors!: ReturnType<VolumePlot["colorState"]>;
  private hoverInfo: unknown = null;
  readonly float32Filterable: boolean;

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.float32Filterable = device.features.has("float32-filterable");
    if (!this.float32Filterable) throw new Error("volume.ts needs the float32-filterable WebGPU feature");
    this.overlay = new OverlayRenderer(device, font);
    const mod = (code: string) => device.createShaderModule({ code });
    const gen = mod(GEN_WGSL), mc = mod(MC_WGSL), scan = mod(SCAN_WGSL);
    this.genPipeline = device.createComputePipeline({ layout: "auto", compute: { module: gen, entryPoint: "generate" } });
    this.countPipeline = device.createComputePipeline({ layout: "auto", compute: { module: mc, entryPoint: "count" } });
    this.emitPipeline = device.createComputePipeline({ layout: "auto", compute: { module: mc, entryPoint: "emit" } });
    this.scanPipeline = device.createComputePipeline({ layout: "auto", compute: { module: scan, entryPoint: "scan" } });
    this.addPipeline = device.createComputePipeline({ layout: "auto", compute: { module: scan, entryPoint: "add" } });
    const over: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    this.blendOver = over;
    const iso = mod(COMMON_WGSL + ISO_WGSL);
    const isoLayout = device.createPipelineLayout({
      bindGroupLayouts: [device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        ],
      })],
    });
    this.isoPipeline = device.createRenderPipeline({
      layout: isoLayout,
      vertex: { module: iso, entryPoint: "vertexMain" },
      fragment: { module: iso, entryPoint: "fragmentMain", targets: [{ format: FRAME_FORMAT }] },
      depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: "less" },
      multisample: { count: SAMPLES },
    });
    this.isoDepthPipeline = device.createRenderPipeline({
      layout: isoLayout,
      vertex: { module: iso, entryPoint: "vertexDepth" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });
    const lines = mod(LINES_WGSL);
    const lineDesc = (depthCompare?: GPUCompareFunction): GPURenderPipelineDescriptor => ({
      layout: "auto",
      vertex: { module: lines, entryPoint: "vertexMain" },
      fragment: { module: lines, entryPoint: "fragmentMain", targets: [{ format: FRAME_FORMAT, blend: over }] },
      depthStencil: depthCompare && { format: DEPTH, depthWriteEnabled: false, depthCompare },
      multisample: { count: SAMPLES },
    });
    this.linePipelines = { depth: device.createRenderPipeline(lineDesc("less-equal")), over: device.createRenderPipeline(lineDesc()) };
    const t = caseTables();
    const upload = (arr: Uint32Array) => {
      const b = device.createBuffer({ size: Math.max(arr.byteLength, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, arr);
      return b;
    };
    this.tables = { cases: upload(t.cases), tris: upload(t.tris), loops: upload(t.loops) };
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.volumeUniforms = device.createBuffer({ size: 68 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lineUniforms = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  static async create(): Promise<VolumePlot> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");
    const features: GPUFeatureName[] = adapter.features.has("float32-filterable") ? ["float32-filterable"] : [];
    const device = await adapter.requestDevice({ requiredFeatures: features });
    device.addEventListener("uncapturederror", (e) => console.error(`volume.ts: WebGPU: ${(e as GPUUncapturedErrorEvent).error.message}`));
    return new VolumePlot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  /** The ray marcher, specialised (compiled) per mode, scale and non-uniform axes
   * (single-sampled: drawn over the resolved surfaces). */
  private volumePipeline(lookups: boolean[], mode: number, scale: number): GPURenderPipeline {
    const key = `${lookups}|${mode}|${scale}`;
    let p = this.volumePipelines.get(key);
    if (!p) {
      const code = VOLUME_WGSL.replace("$LOOKUP_X", String(lookups[0])).replace("$LOOKUP_Y", String(lookups[1]))
        .replace("$LOOKUP_Z", String(lookups[2])).replace("$MODE", String(mode)).replace("$SCALE", String(scale));
      const module = this.device.createShaderModule({ code });
      p = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "fragmentMain", targets: [{ format: FRAME_FORMAT, blend: this.blendOver }] },
        primitive: { cullMode: "front" },
      });
      this.volumePipelines.set(key, p);
    }
    return p;
  }

  // Data -------------------------------------------------------------------------------------------

  /** Generates a slice into a fresh texture (compute), and reads it back. */
  private async generate(meta: VolumeMeta, slice: number): Promise<{ texture: GPUTexture; values: Float32Array }> {
    const { device } = this;
    const [nz, ny, nx] = meta.shape;
    const texture = device.createTexture({
      size: [nx, ny, nz],
      dimension: "3d",
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const u = new ArrayBuffer(112);
    const i32 = new Int32Array(u), u32 = new Uint32Array(u), f32 = new Float32Array(u);
    u32[0] = meta.kind;
    [i32[1], i32[2], i32[3]] = meta.params;
    u32[4] = meta.params[3];
    u32[5] = slice;
    u32[6] = meta.slices;
    u32.set([nx, ny, nz, 0], 8);
    f32.set(meta.axes.map((a) => a.lo), 12);
    f32.set(meta.axes.map((a) => a.hi), 16);
    u32.set(meta.axes.map((a) => (a.geometric ? 1 : 0)), 20);
    f32.set(meta.norm, 24);
    const ub = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, u);
    const group = device.createBindGroup({
      layout: this.genPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: texture.createView() }],
    });
    const pitch = rowBytes(nx);
    const read = makeReadBuffer(device, pitch * ny * nz);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.genPipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: pitch, rowsPerImage: ny }, [nx, ny, nz]);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(read.getMappedRange());
    const values = new Float32Array(nx * ny * nz);
    const rowF = pitch / 4;
    for (let r = 0; r < ny * nz; r += 1) values.set(src.subarray(r * rowF, r * rowF + nx), r * nx);
    read.unmap();
    read.destroy();
    ub.destroy();
    return { texture, values };
  }

  /** The volume of the spec and slice (cached by spec): the data domain over every slice
   * (`useStableDomain` during playback), and the Fermi level (median, half filling). */
  private loadData(spec: VolumeSpec, slice: number, scaleType: ScaleType, ignore?: (v: number) => boolean): Promise<Data> {
    const key = JSON.stringify([spec, slice, scaleType, !!ignore]);
    let p = this.dataCache.get(key);
    if (!p) {
      p = (async () => {
        const meta = volumeMeta(spec);
        const s = Math.min(Math.max(Math.trunc(slice), 0), meta.slices - 1);
        const { texture, values } = await this.generate(meta, s);
        const domain = await this.stableDomain(spec, meta, scaleType, ignore, s, values);
        let fermi: number | undefined;
        if (meta.kind === 4) {
          const sorted = Float64Array.from(values).sort();
          const n = sorted.length;
          fermi = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2; // np.median
        }
        return { key, meta, slice: s, texture, values, dataDomain: domain ?? [...DEFAULT_DOMAIN], fermi } as Data;
      })();
      p.catch(() => this.dataCache.delete(key));
      if (this.dataCache.size >= 24) {
        const [oldest] = this.dataCache.keys();
        this.dataCache.get(oldest)?.then((d) => d !== this.data && d.texture.destroy());
        this.dataCache.delete(oldest);
      }
      this.dataCache.set(key, p);
    }
    return p;
  }

  /** Data domain over every slice (cached by spec and scale, once per stack: playback requests
   * each slice in turn); slice `s`'s `values` are at hand, the others are generated for it. */
  private stableDomain(spec: VolumeSpec, meta: VolumeMeta, scaleType: ScaleType, ignore: ((v: number) => boolean) | undefined, s: number, values: Float32Array): Promise<[number, number] | null> {
    const key = JSON.stringify([spec, scaleType, !!ignore]);
    let p = this.domainCache.get(key);
    if (!p) {
      p = (async () => {
        let lo = Infinity, hi = -Infinity;
        for (let k = 0; k < meta.slices; k += 1) {
          let vals = values;
          if (k !== s) { // other slices: domain only
            const other = await this.generate(meta, k);
            other.texture.destroy();
            vals = other.values;
          }
          const d = getDomain(vals, scaleType, ignore);
          if (d) {
            lo = Math.min(lo, d[0]);
            hi = Math.max(hi, d[1]);
          }
        }
        return Number.isFinite(lo) ? [lo, hi] as [number, number] : null;
      })();
      p.catch(() => this.domainCache.delete(key));
      if (this.domainCache.size >= 24) this.domainCache.delete(this.domainCache.keys().next().value!);
      this.domainCache.set(key, p);
    }
    return p;
  }

  // Isosurfaces ------------------------------------------------------------------------------------

  /** Marching cubes on the GPU: count per cell, exclusive prefix sum, emit. */
  private async extract(data: Data, level: number, vis: VisScale, centres: Float32Array): Promise<IsoMesh> {
    const t0 = performance.now();
    const { device } = this;
    const [nz, ny, nx] = data.meta.shape;
    const cells = (nx - 1) * (ny - 1) * (nz - 1);
    const kind = Array.isArray(vis) ? (vis[0] === "symlog" ? 2 : 0) : vis === "log" ? 1 : vis === "sqrt" ? 3 : vis === "symlog" ? 2 : 0;
    const key = `${data.key}|${level}|${JSON.stringify(vis)}`;
    if (kind === 1 && !(level > 0) || kind === 3 && !(level >= 0) || !Number.isFinite(level)) {
      return { level, key, triangles: 0, ms: 0 };
    }
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const u = new ArrayBuffer(32);
    new Uint32Array(u).set([nx, ny, nz, kind]);
    new Float32Array(u).set([level, Array.isArray(vis) ? vis[1] : 1], 4);
    new Uint32Array(u)[6] = cells;
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, u);
    const counts = device.createBuffer({ size: cells * 4, usage });
    const centreBuf = device.createBuffer({ size: centres.byteLength, usage });
    device.queue.writeBuffer(centreBuf, 0, centres);
    const dummy = device.createBuffer({ size: 32, usage });
    const group = (pipeline: GPUComputePipeline, verts: GPUBuffer) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ub } },
        { binding: 1, resource: data.texture.createView() },
        { binding: 2, resource: { buffer: this.tables.cases } },
        { binding: 3, resource: { buffer: this.tables.tris } },
        { binding: 4, resource: { buffer: this.tables.loops } },
        { binding: 5, resource: { buffer: centreBuf } },
        { binding: 6, resource: { buffer: counts } },
        { binding: 7, resource: { buffer: verts } },
      ],
    });
    const temps: GPUBuffer[] = [ub, centreBuf, dummy];
    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.countPipeline);
    pass.setBindGroup(0, group(this.countPipeline, dummy));
    pass.dispatchWorkgroups(...dispatch2D(Math.ceil(cells / 64)));
    pass.end();
    // Prefix sum, level by level (256 per block); the last level's single sum is the total
    const levels: { data: GPUBuffer; sums: GPUBuffer; n: number; blocks: number; ub: GPUBuffer }[] = [];
    let buf = counts;
    let n = cells;
    for (;;) {
      const blocks = Math.ceil(n / 256);
      const sums = device.createBuffer({ size: Math.max(blocks, 4) * 4, usage });
      const sb = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(sb, 0, Uint32Array.of(n, 0, 0, 0));
      temps.push(sums, sb);
      levels.push({ data: buf, sums, n, blocks, ub: sb });
      if (blocks === 1) break;
      buf = sums;
      n = blocks;
    }
    const scanGroup = (pipeline: GPUComputePipeline, l: (typeof levels)[number]) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: l.ub } }, { binding: 1, resource: { buffer: l.data } }, { binding: 2, resource: { buffer: l.sums } }],
    });
    for (const l of levels) {
      const p = encoder.beginComputePass();
      p.setPipeline(this.scanPipeline);
      p.setBindGroup(0, scanGroup(this.scanPipeline, l));
      p.dispatchWorkgroups(...dispatch2D(l.blocks));
      p.end();
    }
    for (let i = levels.length - 2; i >= 0; i -= 1) {
      const p = encoder.beginComputePass();
      p.setPipeline(this.addPipeline);
      p.setBindGroup(0, scanGroup(this.addPipeline, levels[i]));
      p.dispatchWorkgroups(...dispatch2D(levels[i].blocks));
      p.end();
    }
    const total = makeReadBuffer(device, 4);
    encoder.copyBufferToBuffer(levels[levels.length - 1].sums, 0, total, 0, 4);
    device.queue.submit([encoder.finish()]);
    await total.mapAsync(GPUMapMode.READ);
    const triangles = new Uint32Array(total.getMappedRange())[0];
    total.unmap();
    total.destroy();
    let vertexBuffer: GPUBuffer | undefined;
    if (triangles > 0) {
      vertexBuffer = device.createBuffer({ size: triangles * 3 * 32, usage });
      encoder = device.createCommandEncoder();
      pass = encoder.beginComputePass();
      pass.setPipeline(this.emitPipeline);
      pass.setBindGroup(0, group(this.emitPipeline, vertexBuffer));
      pass.dispatchWorkgroups(...dispatch2D(Math.ceil(cells / 64)));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    counts.destroy();
    for (const b of temps) b.destroy();
    return { level, key, buffer: vertexBuffer, triangles, ms: performance.now() - t0 };
  }

  /** Vertices (index space, x y z) of an extracted level, for checks against `marchingCubes` (lib/marching_cubes.ts). */
  async isoPositions(i: number): Promise<Float32Array> {
    const m = this.meshes[i];
    if (!m?.buffer) return new Float32Array(0);
    const read = makeReadBuffer(this.device, m.buffer.size);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(m.buffer, 0, read, 0, m.buffer.size);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    const [nz, ny, nx] = this.data!.meta.shape;
    const dims = [nx, ny, nz];
    const out = new Float32Array(m.triangles * 9);
    for (let k = 0; k < m.triangles * 3; k += 1) for (let a = 0; a < 3; a += 1) out[k * 3 + a] = v[k * 8 + a] * dims[a] - 0.5; // uniform axes
    return out;
  }

  /** The voxels of the current slice as read back from the GPU, and its spec's metadata. */
  get volumeData(): { values: Float32Array; meta: VolumeMeta; fermi?: number } | undefined {
    return this.data && { values: this.data.values, meta: this.data.meta, fermi: this.data.fermi };
  }

  get isoStats(): { level: number; triangles: number; ms: number }[] {
    return this.meshes.map(({ level, triangles, ms }) => ({ level, triangles, ms: Math.round(ms * 10) / 10 }));
  }

  // Colour ---------------------------------------------------------------------------------------

  /** Colour scale, domains (data, shown, colour bar) and the ray marcher's colour uniforms. */
  private colorState(ctl: Control, data: Data) {
    const scaleName = ctl.scale;
    const configScale: ScaleType = scaleName === "sqrt" || scaleName === "root" ? "gamma" : scaleName === "log" ? "log" : scaleName === "symlog" ? "symlog" : "linear";
    const vis: VisScale = scaleName === "sqrt"
      ? ["gamma", 0.5] // square root as the nth root of 2
      : scaleName === "root" ? ["root", snapRootN(ctl.rootN)]
      : scaleName === "symlog" && ctl.symlogThreshold ? ["symlog", ctl.symlogThreshold] : configScale;
    const dataDomain = data.dataDomain;
    // Fixed ends, else symmetric about `center` (`symmetric`, or colormap "auto" when the data cross it)
    const center = ctl.center ?? (data.fermi !== undefined ? Math.round(data.fermi * 1000) / 1000 : 0);
    const fixed = ctl.domain?.some((v) => v !== null && Number.isFinite(v)) ? ctl.domain : null;
    const auto = String(ctl.colormap).toLowerCase() === "auto";
    const [fmin = dataDomain[0], fmax = dataDomain[1]] = (fixed ?? []).map((v) => (v === null || !Number.isFinite(v) ? undefined : v));
    const crosses = ctl.symmetric || (fmin < center && fmax > center);
    const symmetric = ctl.symmetric || (auto && crosses && !fixed);
    const half = Math.max(Math.abs(dataDomain[0] - center), Math.abs(dataDomain[1] - center));
    const custom: [number | null, number | null] = fixed
      ? fixed.map((v) => (v === null || !Number.isFinite(v) ? null : v)) as [number | null, number | null]
      : symmetric && Number.isFinite(half) ? [center - (half || 1), center + (half || 1)] : [null, null];
    const colormap = auto ? (crosses ? "RdBu" : "Viridis") : colormapName(String(ctl.colormap)) ?? "Inferno";
    const visDomain = getVisDomain(custom, dataDomain);
    const domain = getSafeDomain(visDomain, dataDomain, configScale);
    const showVolume = ctl.showVolume !== false;
    const threshold = Number(ctl.threshold);
    const [shaderScale, gammaExponent, symlogConstant] = getShaderScale(vis);
    // Diverging colouring (a domain symmetric about `center`, or a diverging map spanning it): the
    // DVR opacity ramp is over the distance from the centre, the colours over the whole map
    const centre = createScale(vis, { domain, range: [0, 1], clamp: true })(center);
    const diverging = (symmetric || (DIVERGING.has(colormap) && domain[0] < center && center < domain[1])) &&
      ctl.mode === "DVR" && Number.isFinite(centre);
    const barDomain = showVolume && ctl.mode !== "Iso" && !diverging ? getThresholdDomain(domain, vis, threshold) : domain;
    const [min, max] = scaleDomain(domain, shaderScale, symlogConstant);
    const inv = Math.fround(1 / (max - min));
    const thresholdU = getThresholdUniforms(threshold, gammaExponent);
    return {
      vis, configScale, colormap, domain: domain as Domain, barDomain: barDomain as Domain,
      shaderScale, gammaExponent, symlogConstant, domainMin: min, oneOverRange: Number.isFinite(inv) ? inv : 0,
      thresholdU: diverging ? { ...thresholdU, colorStart: 0, oneOverColorSpan: 1 } : thresholdU,
      diverging: diverging ? { centre, scale: 1 / Math.max(centre, 1 - centre) } : null,
    };
  }

  private colorMapBuffer(name: string, invert: boolean): GPUBuffer {
    const key = `${name}|${invert}`;
    if (key !== this.colorKey) {
      const bytes = colormapLut(name, invert);
      this.colorBuffer?.destroy();
      this.colorBuffer = this.device.createBuffer({ size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.colorBuffer, 0, bytes);
      this.colorKey = key;
    }
    return this.colorBuffer!;
  }

  // Frame ------------------------------------------------------------------------------------------

  private ensureTargets(w: number, h: number) {
    if (this.targets?.w === w && this.targets.h === h) return this.targets;
    for (const t of Object.values(this.targets ?? {})) if (t instanceof GPUTexture) t.destroy();
    const d = this.device;
    const tex = (format: GPUTextureFormat, usage: number, sampleCount = 1, size = [w, h]) => d.createTexture({ size, format, usage, sampleCount });
    this.targets = {
      color: tex(FRAME_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT, SAMPLES),
      depth: tex(DEPTH, GPUTextureUsage.RENDER_ATTACHMENT, SAMPLES),
      isoDepth: tex("depth32float", GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING),
      layer: tex(FRAME_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING),
      edges: tex(FRAME_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING),
      half: tex(FRAME_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING, 1, halfSize(w, h)),
      w, h,
    };
    return this.targets;
  }

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label" | "axisTick" | "axisLabel", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 19 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 13 * s),
      axisTick: this.font.get("regular", 11 * s),
      axisLabel: this.font.get("bold", 13 * s),
    };
  }

  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    const w = Math.trunc(ctl.width), h = Math.trunc(ctl.height);
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    this.scale = s;
    this.size = [w, h];
    const preScale: ScaleType = ctl.scale === "log" ? "log" : ctl.scale === "symlog" ? "symlog" : ctl.scale === "sqrt" || ctl.scale === "root" ? "gamma" : "linear";
    const ignore = ctl.valueTransform ? (v: number) => !Number.isFinite(v) : undefined; // logarithms: ±∞ ignored
    const data = await this.loadData(ctl.volume, Number(ctl.slice) || 0, preScale, ignore);
    this.data = data;
    const meta = data.meta;
    const [nz, ny, nx] = meta.shape;
    const colors = this.colorState(ctl, data);
    this.colors = colors;

    // Axes (x, y, z): values, lookups of non-uniform axes, box size
    const values = meta.axes.map(axisValues);
    const counts = [nx, ny, nz];
    const lookupsByAxis = values.map((v, i) => getAxisLookup(v, counts[i]));
    const boxSize = (meta.sameUnit ? normalizeBoxSize(getBoxSizeFromAxes(meta.shape, values) ?? getBoxSize(meta.shape)) : getBoxSize(meta.shape)) as Vec3;
    const lookupKey = JSON.stringify(meta.axes);
    if (lookupKey !== this.lookupKey) {
      const flat: number[] = [];
      for (const t of lookupsByAxis) if (t) flat.push(...t);
      this.lookupBuffer?.destroy();
      this.lookupBuffer = device.createBuffer({ size: Math.max(flat.length * 4, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(this.lookupBuffer, 0, Float32Array.from(flat.length ? flat : [0]));
      this.lookupKey = lookupKey;
    }
    const lookupOffsets: number[] = [];
    let off = 0;
    for (const t of lookupsByAxis) {
      lookupOffsets.push(t ? off : -1);
      off += t?.length ?? 0;
    }

    // Isosurfaces: (re)extracted when the data, a level or the scale changes
    const levels = (ctl.isoLevels ?? (data.fermi !== undefined ? [Math.round(data.fermi * 1000) / 1000] : [])).map(Number);
    const wanted = levels.map((level) => `${data.key}|${level}|${JSON.stringify(colors.vis)}`);
    if (wanted.join("\n") !== this.meshes.map((m) => m.key).join("\n")) {
      const centres = Float32Array.from(lookupsByAxis.flatMap((t, a) =>
        Array.from({ length: counts[a] }, (_, i) => (t ? t[2 * i + 1] : (i + 0.5) / counts[a]))));
      const keep = new Map(this.meshes.map((m) => [m.key, m]));
      const next: IsoMesh[] = [];
      for (let i = 0; i < levels.length; i += 1) {
        next.push(keep.get(wanted[i]) ?? await this.extract(data, levels[i], colors.vis, centres));
        keep.delete(wanted[i]);
      }
      for (const m of keep.values()) m.buffer?.destroy();
      this.meshes = next;
    }

    // Camera: fitted to the box corners, as curve.ts
    const bar = Math.round(150 * s);
    const [L, T, R, B] = [Math.round(16 * s), Math.round(56 * s), w - Math.round(16 * s) - bar, h - Math.round(16 * s)];
    const [pw, ph] = [R - L, B - T];
    const b2w = boxToWorld(String(ctl.up));
    const corners: number[] = [];
    const fitPoint = (x: number, y: number, z: number) => {
      const p = apply(b2w, [x, y, z]);
      corners.push(p[0] * FIT_MARGIN, p[1] * FIT_MARGIN, p[2] * FIT_MARGIN);
    };
    const [hw, hh, hd] = boxSize.map((v) => v / 2);
    for (let c = 0; c < 8; c += 1) fitPoint((c & 1 ? 1 : -1) * hw, (c & 2 ? 1 : -1) * hh, (c & 4 ? 1 : -1) * hd);
    if (ctl.showBox && ctl.showAxes && meta.axes.some((a) => a.label)) { // axis titles, on either side
      for (const sgn of [-1, 1]) {
        fitPoint(0, sgn * (hh + 0.24), -hd);
        fitPoint(sgn * (hw + 0.24), 0, -hd);
        for (const sy of [-1, 1]) fitPoint(sgn * hw, sy * hh, hd + 0.1);
      }
    }
    const fit = fitView(corners, ctl.azimuth, ctl.elevation, FOV, pw / ph) ?? [[0, 0, 0] as Vec3, 3];
    const target = fit[0];
    const distance = fit[1] / (Number(ctl.zoom) || 1);
    const [f] = cameraBasis(ctl.azimuth, ctl.elevation);
    const eye = [0, 1, 2].map((i) => target[i] + distance * f[i]) as Vec3;
    const camWorld = cameraWorld(eye, target, [0, 0, 1]);
    const proj = perspective(FOV, pw / ph, distance / 100, distance * 100);
    const toRect = rows(pw / w, 0, 0, (L + R) / w - 1, 0, ph / h, 0, 1 - (T + B) / h, 0, 0, 1, 0, 0, 0, 0, 1);
    const zFix = rows(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1); // GL clip z to WebGPU's
    const vpWorld = mul(zFix, toRect, proj, invert(camWorld));
    const model = mul(b2w, scaling(...boxSize));
    const clip = mul(vpWorld, model);
    const camBoxV = apply(invert(model), eye);
    const camBox = { x: camBoxV[0], y: camBoxV[1], z: camBoxV[2] };
    const invClip = invert(clip);
    this.view = {
      vpWorld, boxToWorld: b2w, boxSize, camBox: [camBox.x, camBox.y, camBox.z], target, distance, plot: [L, T, R, B],
      dirOf: (x, y) => {
        const p = apply(invClip, [(x / w) * 2 - 1, 1 - (y / h) * 2, 0.5]);
        const d = [p[0] - camBox.x, p[1] - camBox.y, p[2] - camBox.z];
        const n = Math.hypot(...d);
        return d.map((v) => v / n) as Vec3;
      },
    };

    // Uniforms
    const samples = SAMPLES_PER_VOXEL[ctl.quality as Quality] ?? 1;
    const steps = [getStepCount(lookupsByAxis[0], nx), getStepCount(lookupsByAxis[1], ny), getStepCount(lookupsByAxis[2], nz)];
    const opaqueIso = this.meshes.some((m) => m.triangles > 0);
    const modeIndex = ctl.mode === "MIP" ? 0 : ctl.mode === "Iso" ? 2 : 1;
    const scaleIndex = colors.shaderScale === "log" ? 1 : colors.shaderScale === "symlog" ? 2 : colors.shaderScale === "sqrt" ? 3 : 0;
    // While the camera moves, rays are marched at half resolution (and half the samples, shaded
    // by forward differences) into a layer of their own, upsampled as the overlay composites it:
    // the ray march is most of the frame's cost
    const half = ctl.interacting && ctl.showVolume !== false;
    const [vw, vh] = half ? halfSize(w, h) : [w, h];
    const vu = new Float32Array(68);
    vu.set(clip, 0);
    vu.set(invClip, 16);
    vu.set([camBox.x, camBox.y, camBox.z, 1], 32);
    vu.set([nx, ny, nz, ctl.interacting ? samples / 2 : samples], 36);
    vu.set([...steps, Number(ctl.opacity)], 40);
    vu.set([colors.domainMin, colors.oneOverRange, colors.gammaExponent, colors.symlogConstant], 44);
    vu.set([1, 0, Number(ctl.threshold), colors.thresholdU.oneOverRamp], 48);
    vu.set([colors.thresholdU.colorStart, colors.thresholdU.oneOverColorSpan, opaqueIso ? 1 : 0, modeIndex], 52);
    vu.set([scaleIndex, ...lookupOffsets], 56);
    vu.set([vw, vh, colors.diverging?.scale ?? 0, colors.diverging?.centre ?? 0], 60);
    vu.set([w / vw, h / vh, half ? 1 : 0, 0], 64);
    device.queue.writeBuffer(this.volumeUniforms, 0, vu);
    const tg = this.ensureTargets(w, h);

    // Isosurface uniforms: headlight above left of the camera (camera space), towards the origin
    const lightP = apply(camWorld, [-0.6, 0.8, 1]);
    const lightN = Math.hypot(...lightP);
    const light = { x: lightP[0] / lightN, y: lightP[1] / lightN, z: lightP[2] / lightN };
    while (this.isoUniforms.length < this.meshes.length) {
      this.isoUniforms.push(device.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    this.meshes.forEach((m, i) => {
      const color = ctl.isoColors === "colormap"
        ? (() => {
          const t = createScale(colors.vis, { domain: colors.domain, range: [0, 1], clamp: true })(m.level);
          return colormapRgb(colors.colormap, t, !!ctl.invertColormap).map((c) => srgbToLinear(c / 255));
        })()
        : hexLinear(ISO_COLORS[i % ISO_COLORS.length]);
      const iu = new Float32Array(48);
      iu.set(clip, 0);
      iu.set(model, 16);
      iu.set([...eye, 1], 32);
      iu.set([light.x, light.y, light.z, 0], 36);
      iu.set([...boxSize, 0], 40);
      iu.set([...color, 1], 44);
      device.queue.writeBuffer(this.isoUniforms[i], 0, iu);
    });

    // Box edges and axes (segments in world)
    const q = new Quads(this.font);
    const segs: number[] = [];
    const W = (p: Vec3): Vec3 => apply(b2w, p);
    const seg = (a: Vec3, b: Vec3, color: Rgba, width: number) => segs.push(...W(a), 1, ...W(b), 1, ...color, width, 0, 0, 0);
    const [bw, bh, bd] = boxSize;
    const overCount = { depth: 0 };
    const labels: Label[] = [];
    if (ctl.showBox && ctl.showAxes && meta.axes.some((a) => a.label)) {
      this.axes(meta, values, counts, boxSize, camBox, seg, labels, s);
    }
    overCount.depth = segs.length / 16;
    if (ctl.showBox) {
      for (let a = 0; a < 3; a += 1) {
        for (let k = 0; k < 4; k += 1) {
          const p: number[] = [0, 0, 0], q2: number[] = [0, 0, 0];
          const [u1, u2] = [0, 1, 2].filter((x) => x !== a);
          p[a] = -0.5; q2[a] = 0.5;
          p[u1] = q2[u1] = (k & 1) - 0.5;
          p[u2] = q2[u2] = ((k >> 1) & 1) - 0.5;
          seg(p.map((v, i) => v * boxSize[i]) as Vec3, q2.map((v, i) => v * boxSize[i]) as Vec3, [0.2, 0.2, 0.2, 0.2], Math.max(1, s));
        }
      }
    }
    void bw; void bh; void bd;
    const segData = Float32Array.from(segs.length ? segs : new Array(16).fill(0));
    if (!this.segBuffer || this.segBuffer.size < segData.byteLength) {
      this.segBuffer?.destroy();
      this.segBuffer = device.createBuffer({ size: Math.max(2 ** Math.ceil(Math.log2(segData.byteLength)), 1024), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(this.segBuffer, 0, segData);
    const lu = new Float32Array(20);
    lu.set(vpWorld, 0);
    lu.set([w, h, 0, 0], 16);
    device.queue.writeBuffer(this.lineUniforms, 0, lu);

    // Passes: isosurface depth (rays stop there), then surfaces, axes, volume, box edges
    const encoder = device.createCommandEncoder();
    const isoGroups = this.meshes.map((m, i) => m.buffer && device.createBindGroup({
      layout: this.isoPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.isoUniforms[i] } }, { binding: 1, resource: { buffer: m.buffer } }],
    }));
    const dp = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: { view: tg.isoDepth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    dp.setPipeline(this.isoDepthPipeline);
    this.meshes.forEach((m, i) => {
      if (!isoGroups[i]) return;
      dp.setBindGroup(0, isoGroups[i]!);
      dp.draw(m.triangles * 3);
    });
    dp.end();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: tg.color.createView(), resolveTarget: tg.layer.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "discard" }],
      depthStencilAttachment: { view: tg.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setPipeline(this.isoPipeline);
    this.meshes.forEach((m, i) => {
      if (!isoGroups[i]) return;
      pass.setBindGroup(0, isoGroups[i]!);
      pass.draw(m.triangles * 3);
    });
    const lineGroup = (p: GPURenderPipeline) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.lineUniforms } }, { binding: 1, resource: { buffer: this.segBuffer! } }],
    });
    const total = segs.length / 16;
    if (overCount.depth > 0) {
      pass.setPipeline(this.linePipelines.depth);
      pass.setBindGroup(0, lineGroup(this.linePipelines.depth));
      pass.draw(6, overCount.depth);
    }
    pass.end();
    // The volume, over the resolved surfaces (rays stop at their depth), or into the half
    // resolution layer
    const vp = encoder.beginRenderPass({
      colorAttachments: [half
        ? { view: tg.half.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }
        : { view: tg.layer.createView(), loadOp: "load", storeOp: "store" }],
    });
    if (ctl.showVolume !== false) {
      const volumePipeline = this.volumePipeline(lookupsByAxis.map((t) => !!t), modeIndex, scaleIndex);
      vp.setPipeline(volumePipeline);
      vp.setBindGroup(0, device.createBindGroup({
        layout: volumePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.volumeUniforms } },
          { binding: 1, resource: data.texture.createView() },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.colorMapBuffer(colors.colormap, !!ctl.invertColormap) } },
          { binding: 4, resource: { buffer: this.lookupBuffer! } },
          { binding: 5, resource: tg.isoDepth.createView() },
        ],
      }));
      vp.draw(36);
    }
    vp.end();
    // Box edges on top (no depth test), in a layer of their own
    const ep = encoder.beginRenderPass({
      colorAttachments: [{ view: tg.color.createView(), resolveTarget: tg.edges.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "discard" }],
    });
    if (total > overCount.depth) {
      ep.setPipeline(this.linePipelines.over);
      ep.setBindGroup(0, lineGroup(this.linePipelines.over));
      ep.draw(6, total - overCount.depth, 0, overCount.depth);
    }
    ep.end();

    // Overlay: the 3D layer, axis labels, title, colour bar, hover
    // The volume (half resolution) and box edges layers are clear outside the box on screen:
    // composited only within its bounds, which saves full-frame passes
    const boxRect = screenRect(clip, w, h, Math.ceil(4 * s));
    const layers = half ? [tg.layer, tg.half, tg.edges] : [tg.layer, tg.edges];
    layers.forEach((_, i) => q.layer(i, 1, i === 0 ? [0, 0, w, h] : boxRect));
    this.drawLabels(q, labels, s);
    const F = this.fonts(s);
    const title = ctl.title ?? (data.fermi !== undefined ? `Fermi surface 𝐸(𝐤) = ${data.fermi.toFixed(2)} eV (half filling)` : "");
    const tw = title ? q.text(title, L + Math.round(50 * s), Math.round(28 * s), F.title, premul(INK), "lm") : 0;
    if (ctl.subtitle) q.text(String(ctl.subtitle), L + Math.round(50 * s) + tw + Math.round(16 * s), Math.round(28 * s), F.sub, premul(INK2), "lm");
    // Root scale: its slider under the colour bar (raised to make room)
    const root = Array.isArray(colors.vis) && colors.vis[0] === "root" ? colors.vis[1] : null;
    const sliderTop = B - Math.round(10 * s) - rootSliderHeight(F.label, s);
    const barBottom = root === null ? B - Math.round(34 * s) : sliderTop - Math.round(34 * s);
    this.colorBar(q, ctl, colors, R + Math.round(36 * s), T + Math.round(34 * s), barBottom, s);
    this.slider = root === null ? null : rootSlider(q, F.label, R + Math.round(36 * s), sliderTop, Math.round(100 * s), root, s);
    this.hoverInfo = null;
    if (ctl.hover) {
      const info = this.pick(ctl, Number(ctl.hover[0]), Number(ctl.hover[1]), values, counts);
      if (info) {
        this.drawHover(q, ctl, info);
        this.hoverInfo = info;
      }
    }
    this.overlay.setSize(w, h);
    this.overlay.encode(encoder, q, layers, [0, 0, w, h]);
    const pitch = rowBytes(w);
    if (this.frameReads[slot]?.size !== pitch * h) {
      this.frameReads[slot]?.destroy();
      this.frameReads[slot] = makeReadBuffer(device, pitch * h);
    }
    const frameRead = this.frameReads[slot];
    encoder.copyTextureToBuffer({ texture: this.overlay.out! }, { buffer: frameRead, bytesPerRow: pitch, rowsPerImage: h }, [w, h]);
    device.queue.submit([encoder.finish()]);
    const hover = this.hoverInfo;
    return {
      done: frameRead.mapAsync(0x01),
      read: (alloc) => {
        const src = new Uint8Array(frameRead.getMappedRange());
        const rgba = alloc(w * h * 4);
        if (pitch === w * 4) {
          rgba.set(src.subarray(0, w * h * 4));
        } else {
          for (let y = 0; y < h; y += 1) rgba.set(src.subarray(y * pitch, y * pitch + w * 4), y * w * 4);
        }
        frameRead.unmap();
        return { width: w, height: h, rgba, hover };
      },
    };
  }

  /** Box axes: outline of the far face along z, an edge along z, ticks and labels
   * on the edges nearest the camera, in the axes' coordinates (box space). */
  private axes(meta: VolumeMeta, values: Float64Array[], counts: number[], boxSize: Vec3, camBox: { x: number; y: number }, seg: (a: Vec3, b: Vec3, c: Rgba, w: number) => void, labels: Label[], s: number): void {
    const [w, h, d] = boxSize;
    const z0 = -d / 2;
    const sx = camBox.x * w >= 0 ? 1 : -1; // camera side, in box space
    const sy = camBox.y * h >= 0 ? 1 : -1;
    const tickColor = premul(INK2);
    const lw = Math.max(1, s);
    const configs = [0, 1, 2].map((a) => ({ domain: getAxisDomain(values[a], counts[a]), label: meta.axes[a].label && (meta.axes[a].unit ? `${meta.axes[a].label} (${meta.axes[a].unit})` : meta.axes[a].label) }));
    const map = (a: number, v: number) => {
      const [lo, hi] = configs[a].domain;
      return ((v - lo) / (hi - lo) - 0.5) * boxSize[a];
    };
    const xNear = (sx * w) / 2, yNear = (sy * h) / 2, yFar = -yNear;
    seg([-w / 2, -h / 2, z0], [w / 2, -h / 2, z0], tickColor, lw);
    seg([w / 2, -h / 2, z0], [w / 2, h / 2, z0], tickColor, lw);
    seg([w / 2, h / 2, z0], [-w / 2, h / 2, z0], tickColor, lw);
    seg([-w / 2, h / 2, z0], [-w / 2, -h / 2, z0], tickColor, lw);
    seg([xNear, yFar, z0], [xNear, yFar, z0 + d], tickColor, lw);
    const TICK = 0.025, OFF = 0.04, AXIS_OFF = 0.16, Z_OFF = 0.05;
    for (const v of linearTicks(...configs[0].domain, Math.max(Math.round(w * 6), 2))) {
      const x = map(0, v);
      seg([x, yNear, z0], [x, yNear + sy * TICK, z0], tickColor, lw);
      labels.push({ text: formatTick(v), at: [x, yNear + sy * OFF, z0], outward: [0, sy, 0], priority: 2, axis: false });
    }
    for (const v of linearTicks(...configs[1].domain, Math.max(Math.round(h * 6), 2))) {
      const y = map(1, v);
      seg([xNear, y, z0], [xNear + sx * TICK, y, z0], tickColor, lw);
      labels.push({ text: formatTick(v), at: [xNear + sx * OFF, y, z0], outward: [sx, 0, 0], priority: 2, axis: false });
    }
    const zCount = Math.max(Math.round(d * 8), 2);
    for (const v of linearTicks(...configs[2].domain, zCount)) {
      const z = map(2, v);
      seg([xNear, yFar, z], [xNear, yFar - sy * TICK, z], tickColor, lw);
      labels.push({ text: formatTick(v), at: [xNear + sx * Z_OFF, yFar - sy * Z_OFF, z], outward: [sx, -sy, 0], priority: 1, axis: false });
    }
    labels.push(
      { text: configs[0].label ?? "𝑥", at: [0, yNear + sy * AXIS_OFF, z0], outward: [0, sy, 0], priority: 3, axis: true },
      { text: configs[1].label ?? "𝑦", at: [xNear + sx * AXIS_OFF, 0, z0], outward: [sx, 0, 0], priority: 3, axis: true },
    );
    if (configs[2].label) labels.push({ text: configs[2].label, at: [xNear, yFar, z0 + d + OFF], outward: [0, 0, 1], priority: 3, axis: true });
  }

  /** Screen position (px) of a box-space point, or null behind the camera. */
  private project(p: Vec3): [number, number] | null {
    const [vx, vy, vz] = apply(this.view.boxToWorld, p);
    const e = this.view.vpWorld;
    const cw = e[3] * vx + e[7] * vy + e[11] * vz + e[15];
    if (!(cw > 0)) return null;
    const x = (e[0] * vx + e[4] * vy + e[8] * vz + e[12]) / cw;
    const y = (e[1] * vx + e[5] * vy + e[9] * vz + e[13]) / cw;
    return [((x + 1) / 2) * this.size[0], ((1 - y) / 2) * this.size[1]];
  }

  /** Labels anchored on the side towards the box, hidden where they overlap one of higher priority. */
  private drawLabels(q: Quads, labels: Label[], s: number): void {
    const F = this.fonts(s);
    const pad = 2 * s;
    const shown: [number, number, number, number][] = [];
    for (const l of [...labels].sort((a, b) => b.priority - a.priority)) {
      const at = this.project(l.at);
      const to = this.project(l.at.map((v, i) => v + l.outward[i] * 0.1) as Vec3);
      if (!at || !to) continue;
      const font = l.axis ? F.axisLabel : F.axisTick;
      const text = l.text;
      const tw = this.font.measure(font, text) + 2 * pad;
      const th = font.ascent + font.descent + 2 * pad;
      const dx = to[0] - at[0], dy = -(to[1] - at[1]);
      const len = Math.hypot(dx, dy);
      const [cx, cy] = len > 0 ? [0.5 - (0.5 * dx) / len, 0.5 - (0.5 * dy) / len] : [0.5, 0.5];
      const left = at[0] - cx * tw, top = at[1] - (1 - cy) * th;
      const rect: [number, number, number, number] = [left, top, left + tw, top + th];
      if (shown.some((o) => rect[0] < o[2] + pad && o[0] < rect[2] + pad && rect[1] < o[3] + pad && o[1] < rect[3] + pad)) continue;
      shown.push(rect);
      q.text(text, left + tw / 2, top + th / 2, font, premul(l.axis ? INK : INK2), "mm");
    }
  }

  /** Colour bar (vertical, with bounds) of the colour domain: the map evenly along the bar, ticks
   * by the scale (lib/scales.ts). */
  private colorBar(q: Quads, ctl: Control, colors: VolumePlot["colors"], x: number, top: number, bottom: number, s: number): void {
    const F = this.fonts(s);
    const bw = Math.round(14 * s);
    for (let y = top; y < bottom; y += 1) {
      const [r, g, b] = colormapRgb(colors.colormap, 1 - (y + 0.5 - top) / (bottom - top), !!ctl.invertColormap);
      q.rect(x, y, bw, 1, [r / 255, g / 255, b / 255, 1]);
    }
    q.rect(x, top, bw, bottom - top, [0, 0, 0, 0], { border: 1, borderColor: LABEL_BORDER });
    const domain = colors.barDomain;
    const vt = (ctl.valueTransform ?? undefined) as ValueTransform | undefined;
    const axisScale = createScale(colors.vis, { domain, range: [bottom, top] });
    for (const t of colorBarTicks(colors.vis, domain, (bottom - top) / s, vt)) {
      const Y = Math.round(axisScale(t.value));
      if (!(Y >= top - 0.5 && Y <= bottom + 0.5)) continue;
      q.rect(x + bw, Y, Math.round(4 * s), 1, premul(INK2));
      const label = t.label;
      if (label) q.text(label, x + bw + Math.round(7 * s), Y + 0.5, F.tick, premul(INK2), "lm");
    }
    const [lo, hi] = domain[0] === domain[1] ? [NaN, NaN] : vt ? domain.map((v) => getRealValue(v, vt)) : domain;
    const bound = (v: number, inf: string) => (Number.isNaN(v) ? inf : formatBound(v));
    q.text(bound(hi, "+∞"), x + bw / 2, top - Math.round(8 * s), F.label, premul(INK), "mb");
    q.text(bound(lo, "−∞"), x + bw / 2, bottom + Math.round(8 * s), F.label, premul(INK), "mt");
  }

  // Hover ------------------------------------------------------------------------------------------

  /** CPU read-out along the hover ray (the read-back voxels, trilinear as the texture): the first
   * sample the ray marcher shows (normalized value above the threshold), or the first crossing of
   * an iso level if nearer (an opaque surface stops the ray there). */
  private pick(ctl: Control, hx: number, hy: number, values: Float64Array[], counts: number[]) {
    const data = this.data!;
    const [L, T, R, B] = this.view.plot;
    if (hx < L || hx > R || hy < T || hy > B) return null;
    const [nx, ny, nz] = counts;
    const c = this.colors;
    const lookups = values.map((v, i) => getAxisLookup(v, counts[i]));
    const toTex = (p: number, a: number) => {
      const t = lookups[a];
      if (!t) return p;
      let lo = 0, hi = t.length - 2;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (t[mid] <= p) lo = mid; else hi = mid - 1;
      }
      const f = Math.min(Math.max((p - t[lo]) / Math.max(t[lo + 1] - t[lo], 1e-30), 0), 1);
      return (lo + f) / (t.length - 1);
    };
    const v = data.values;
    const at = (i: number, j: number, k: number) => v[(Math.min(Math.max(k, 0), nz - 1) * ny + Math.min(Math.max(j, 0), ny - 1)) * nx + Math.min(Math.max(i, 0), nx - 1)];
    const sample = (p: Vec3): { value: number; tex: Vec3 } => {
      const tex = [toTex(p[0], 0), toTex(p[1], 1), toTex(p[2], 2)] as Vec3;
      const g = tex.map((t, a) => t * counts[a] - 0.5);
      const i0 = g.map(Math.floor);
      const fr = g.map((x, a) => x - i0[a]);
      let val = 0;
      for (let n = 0; n < 8; n += 1) {
        const o = [n & 1, (n >> 1) & 1, (n >> 2) & 1];
        const wgt = o.reduce((acc, b, a) => acc * (b ? fr[a] : 1 - fr[a]), 1);
        val += wgt * at(i0[0] + o[0], i0[1] + o[1], i0[2] + o[2]);
      }
      return { value: val, tex };
    };
    const [shaderScale, gamma, symC] = [c.shaderScale, c.gammaExponent, c.symlogConstant];
    const scaled = (x: number) => shaderScale === "log" ? Math.log10(x) : shaderScale === "symlog" ? Math.sign(x) * Math.log1p(Math.abs(x) / symC) / Math.LN10 : shaderScale === "sqrt" ? Math.sqrt(x) : x;
    const supported = (x: number) => shaderScale === "log" ? x > 0 : shaderScale === "sqrt" ? x >= 0 : Number.isFinite(x);
    const normalized = (x: number) => {
      if (!supported(x)) return -1;
      const pos = c.oneOverRange === 0 ? (scaled(x) >= c.domainMin ? 1 : 0) : Math.min(Math.max(c.oneOverRange * (scaled(x) - c.domainMin), 0), 1);
      return pos ** gamma;
    };
    const origin = this.view.camBox.map((x) => x + 0.5) as Vec3;
    const dir = this.view.dirOf(hx, hy);
    let t0 = 0, t1 = Infinity;
    for (let a = 0; a < 3; a += 1) {
      const inv = 1 / dir[a];
      const ta = -origin[a] * inv, tb = (1 - origin[a]) * inv;
      t0 = Math.max(t0, Math.min(ta, tb));
      t1 = Math.min(t1, Math.max(ta, tb));
    }
    if (!(t0 < t1)) return null;
    const stepDims = [getStepCount(lookups[0], nx), getStepCount(lookups[1], ny), getStepCount(lookups[2], nz)];
    const dt = 1 / (Math.hypot(dir[0] * stepDims[0], dir[1] * stepDims[1], dir[2] * stepDims[2]) * 2);
    const levels = this.meshes.filter((m) => m.triangles > 0).map((m) => m.level);
    const showVolume = ctl.showVolume !== false;
    const threshold = Number(ctl.threshold);
    let prev: { value: number; p: Vec3 } | null = null;
    for (let t = t0; t <= t1; t += dt) {
      const p = origin.map((o, a) => o + dir[a] * t) as Vec3;
      const sm = sample(p);
      if (prev) {
        const crossed = levels.find((lv) => (prev!.value - lv) * (sm.value - lv) <= 0 && prev!.value !== sm.value);
        if (crossed !== undefined) {
          let [a, b] = [prev.p, p];
          for (let it = 0; it < 20; it += 1) {
            const mid = a.map((x, k) => (x + b[k]) / 2) as Vec3;
            if ((sample(a).value - crossed) * (sample(mid).value - crossed) <= 0) b = mid; else a = mid;
          }
          return this.hoverResult(ctl, hx, hy, b, sample(b), values, counts, crossed);
        }
      }
      if (showVolume && normalized(sm.value) > threshold) return this.hoverResult(ctl, hx, hy, p, sm, values, counts);
      prev = { value: sm.value, p };
    }
    return null;
  }

  private hoverResult(ctl: Control, hx: number, hy: number, p: Vec3, sm: { value: number; tex: Vec3 }, values: Float64Array[], counts: number[], level?: number) {
    const coords = sm.tex.map((t, a) => {
      const g = Math.min(Math.max(t * counts[a] - 0.5, 0), counts[a] - 1);
      const i = Math.min(Math.floor(g), counts[a] - 2);
      return values[a][i] + (g - i) * (values[a][i + 1] - values[a][i]);
    });
    const at = this.project(p.map((x, a) => (x - 0.5) * this.view.boxSize[a]) as Vec3) ?? [hx, hy];
    const value = ctl.valueTransform ? getRealValue(sm.value, ctl.valueTransform) : sm.value;
    return { pixel: [hx, hy] as [number, number], at, coords, value, raw: sm.value, isosurface: level ?? null, slice: this.data!.slice };
  }

  private drawHover(q: Quads, ctl: Control, info: NonNullable<ReturnType<VolumePlot["pick"]>>): void {
    const s = this.scale;
    q.dot(info.at[0], info.at[1], 6 * s, 2 * s, premul(INK), premul(HOVER_RING));
    const font = this.fonts(s).label;
    const meta = this.data!.meta;
    const names = ["𝑥", "𝑦", "𝑧"];
    const rows: [string, string][] = meta.axes.map((a, i) => [a.label ?? names[i], formatNumber(info.coords[i])]);
    if (meta.slices > 1) rows.unshift(["slice", String(info.slice)]);
    rows.push([ctl.valueLabel ?? "value", formatNumber(info.value)]);
    if (info.isosurface !== null) rows.push(["isosurface", formatNumber(info.isosurface)]);
    if (this.slider) rows.push(["ⁿ√ n", this.slider.value.toFixed(2)]);
    const gapRow = Math.round(3 * s);
    const lh = font.ascent + font.descent + gapRow;
    const kw = Math.max(...rows.map(([k]) => this.font.measure(font, k))) + Math.round(12 * s);
    const vw = Math.max(...rows.map(([, v]) => this.font.measure(font, v)));
    const pad = Math.round(8 * s);
    const lw = Math.ceil(pad * 2 + kw + vw);
    const lhTotal = pad * 2 + lh * rows.length - gapRow;
    const [hx, hy] = info.pixel;
    const gap = Math.round(14 * s);
    const [w, h] = this.size;
    let x = hx + gap + lw <= w ? hx + gap : hx - gap - lw;
    let y = hy + gap + lhTotal <= h ? hy + gap : hy - gap - lhTotal;
    x = Math.trunc(Math.min(Math.max(x, 0), w - lw));
    y = Math.trunc(Math.min(Math.max(y, 0), h - lhTotal));
    q.rect(x, y, lw, lhTotal, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    rows.forEach(([k, v], i) => {
      const ry = y + pad + i * lh;
      q.text(k, x + pad, ry, font, premul(INK2), "lt");
      q.text(v, x + lw - pad, ry, font, premul(INK), "rt");
    });
  }

  state(ctl: Control, generation: number, hover: unknown, ms: number): unknown {
    const c = this.colors;
    const meta = this.data!.meta;
    return {
      generation, width: this.size[0], height: this.size[1], hover, renderMs: Math.round(ms * 1000) / 1000,
      azimuth: ctl.azimuth, elevation: ctl.elevation, distance: this.view.distance, target: this.view.target, plotRect: this.view.plot,
      shape: meta.shape, slice: this.data!.slice, slices: meta.slices, boxSize: this.view.boxSize,
      dataDomain: this.data!.dataDomain, domain: c.domain, colorDomain: c.barDomain, colormap: c.colormap, scale: c.vis,
      ...(this.slider ? { rootN: this.slider.value, rootSlider: this.slider } : {}),
      isosurfaces: this.isoStats,
      fermiLevel: this.data!.fermi ?? null,
    };
  }
}

function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [m, e] = v.toExponential(3).split("e");
    const sup = (n: string) => n.replace(/[-+\d]/g, (ch) => ({ "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" } as Record<string, string>)[ch]);
    return `${Number(m)}×10${sup(e)}`;
  }
  return String(Number(v.toPrecision(5)));
}

if (import.meta.main) {
  await main<Control>({
    name: "volume.ts",
    defaults: DEFAULTS as Control,
    create: VolumePlot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
  });
}
