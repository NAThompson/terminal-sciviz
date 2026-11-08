/**
 * 3D marks: an orbitable scene of the model's own data, drawn from marks like line.ts's
 * (lib/marks.ts), in one multisampled, depth-tested WebGPU pass:
 *
 * - dot: x, y, z (columns or constants); shaded spheres, as instanced impostors (a quad per point,
 *   the ray–sphere hit in the fragment shader, its depth written, coverage per sample).
 * - line: polylines through x, y, z per `group` (default: a categorical fill), fat lines of
 *   constant screen width (as lib/gpu_curve3d.ts draws chords).
 * - mesh: triangles, `{vertices, faces}` or columns x, y, z with `faces` (row indices); flat or
 *   smooth shading, `opacity` (translucent meshes drawn after everything opaque, back faces then
 *   front faces, so what is behind them shows through dimmed), optional `edges`.
 * - transform `hull: true` (dot or mesh): the 3D convex hull of the points (lib/hull.ts), as a mesh.
 *
 * fill: a column (strings: the Okabe–Ito palette, a legend entry each; numbers: a colour map with a
 * colour bar) or a #rrggbb colour; r: CSS px, or a column (sqrt scale, top-level `r`). Axes: the
 * three back panes of the data's box with grid lines, tick labels on the edges facing the camera
 * (as matplotlib's 3D axes), equal aspect unless `aspect: "auto"` (each axis normalised) or
 * [ax, ay, az]. Hover: the nearest point under the pointer (the frontmost one it covers) and its
 * row. Orbit camera, held framing while `interacting`, as curve.ts.
 *
 *   deno run -A scene.ts png CONTROL.json -o out.png
 *   deno run -A scene.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 */

import { axisTicks, formatNumber } from "./lib/scales.ts";
import { cameraBasis, fitAngles, fitView, screenRect, type Vec3 } from "./lib/framing.ts";
import { lookAtView, mul4, perspectiveProjection } from "./lib/mat4.ts";
import { colormapName, colormapRgb, srgbToLinear } from "./lib/colormaps.ts";
import { createBuffer, makeReadBuffer, requestGpuDevice, rowBytes, STORAGE, TextureTarget, UNIFORM } from "./lib/gpu.ts";
import { DEPTH_FORMAT, getTargets, type RenderTargets, SAMPLE_COUNT } from "./lib/gpu_surface.ts";
import { BitmapFont, type FontSize, FRAME_FORMAT, OverlayRenderer, premul, Quads, type Rgba } from "./lib/overlay.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";
import { type Columns, sqrtRadius } from "./lib/marks.ts";
import { INK, INK2, PALETTE, tableFile } from "./line.ts";
import { convexHull } from "./lib/hull.ts";
import { normalRng } from "./lib/specfun.ts";

const GRID = "#262c36", EDGE = "#30363d", HOVER_RING = "#e6edf3";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];
const FOV = 40;
const FIT_MARGIN = 1.04;
const PAD = 0.03; // of each axis' range, around the data
const MAX_ROWS = 1_000_000;

type Value = number | string | null;
type Channel = string | number;

export interface SceneMark {
  type: "dot" | "line" | "mesh";
  data?: Columns | Record<string, Value>[];
  file?: string; // CSV/TSV with a header row, or JSON columns / rows
  gen?: { name: "ball" | "torus" | "clusters"; n?: number; seed?: number; k?: number; noise?: number };
  vertices?: number[][]; // mesh
  faces?: number[][]; // mesh: row (vertex) indices, counter-clockwise seen from outside
  x?: Channel;
  y?: Channel;
  z?: Channel;
  fill?: Channel;
  stroke?: Channel; // the same colour channel as fill
  r?: Channel;
  group?: string; // line: a column splitting it into polylines (default: a categorical fill)
  opacity?: number;
  width?: number; // line, edges: CSS px
  shading?: "flat" | "smooth";
  edges?: boolean | string;
  transform?: { hull?: boolean };
  label?: string;
  legend?: boolean;
}

export const DEFAULTS = {
  width: 1200,
  height: 750,
  title: "",
  subtitle: "",
  marks: [] as SceneMark[],
  xlabel: null as string | null, // default: the first mark's x column
  ylabel: null as string | null,
  zlabel: null as string | null,
  aspect: "equal" as "equal" | "auto" | [number, number, number],
  color: {} as { scheme?: string; domain?: Value[]; label?: string; invert?: boolean },
  r: {} as { range?: [number, number]; domain?: [number, number] },
  legend: true,
  axes: true,
  azimuth: 35,
  elevation: 25,
  zoom: 1,
  interacting: false, // while the mod orbits: the framing held
  hover: null as [number, number] | null,
  snapshot: null as { path: string; rev: number } | string | null,
  frameFile: false,
  quit: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

// --- Shaders ---------------------------------------------------------------------------------------

const COMMON_WGSL = /* wgsl */ `
  struct View { view: mat4x4f, proj: mat4x4f, size: vec2f, pxScale: f32, css: f32 } // css: device px per CSS px
  @group(0) @binding(0) var<uniform> u: View;

  fn toSrgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
  }

  // Key light from the upper left, a dim fill from the lower right, a highlight: view space
  fn shade(n: vec3f, v: vec3f, base: vec3f) -> vec3f {
    let key = normalize(vec3f(-0.45, 0.6, 0.65));
    let fill = normalize(vec3f(0.6, -0.35, 0.7));
    let spec = pow(max(dot(n, normalize(key + v)), 0.0), 40.0);
    return base * (0.3 + 0.62 * max(dot(n, key), 0.0) + 0.2 * max(dot(n, fill), 0.0)) + vec3f(0.18 * spec);
  }

  fn premultiplied(c: vec3f, a: f32) -> vec4f { return vec4f(toSrgb(c) * a, a); }
`;

const DOT_WGSL = COMMON_WGSL + /* wgsl */ `
  @group(1) @binding(0) var<storage, read> dots: array<vec4f>; // world xyz, radius (CSS px)
  @group(1) @binding(1) var<storage, read> colors: array<vec4f>; // linear RGB, alpha

  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) q: vec3f, // view space, on the quad through the centre
    @location(1) @interpolate(flat) centre: vec3f, // view space
    @location(2) @interpolate(flat) radius: vec2f, // view space, device px
    @location(3) @interpolate(flat) screen: vec2f, // centre, device px (y down)
    @location(4) @interpolate(flat) color: vec4f,
  }
  struct Fragment { @location(0) color: vec4f, @builtin(frag_depth) depth: f32, @builtin(sample_mask) mask: u32 }

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Varyings {
    var corners = array(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));
    let d = vec4f(dots[i].xyz, dots[i].w * u.css);
    let c = (u.view * vec4f(d.xyz, 1.0)).xyz;
    let rw = d.w * -c.z / u.pxScale;
    let q = c + vec3f(corners[v] * rw * (d.w + 1.5) / d.w * 1.04, 0.0); // a pixel and the perspective silhouette beyond r
    var out: Varyings;
    out.position = u.proj * vec4f(q, 1.0);
    out.position.z = 0.5 * (out.position.z + out.position.w);
    let cc = u.proj * vec4f(c, 1.0);
    out.screen = (cc.xy / cc.w * vec2f(0.5, -0.5) + 0.5) * u.size;
    out.q = q;
    out.centre = c;
    out.radius = vec2f(rw, d.w);
    out.color = colors[i];
    if (c.z >= 0.0 || colors[i].a <= 0.0) {
      out.position = vec4f(0.0, 0.0, 2.0, 1.0); // behind the camera, or hidden: clipped
    }
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> Fragment {
    // coverage of the projected disc per sample (the standard 4× pattern)
    var offsets = array(vec2f(0.375, 0.125), vec2f(0.875, 0.375), vec2f(0.125, 0.625), vec2f(0.625, 0.875));
    var mask = 0u;
    for (var k = 0u; k < 4u; k++) {
      if (length(floor(in.position.xy) + offsets[k] - in.screen) <= in.radius.y) {
        mask |= 1u << k;
      }
    }
    if (mask == 0u) {
      discard;
    }
    // the ray through the pixel centre and the sphere (tangent at the silhouette's edge)
    let dir = normalize(in.q);
    let b = dot(dir, in.centre);
    let disc = max(b * b - dot(in.centre, in.centre) + in.radius.x * in.radius.x, 0.0);
    let hit = dir * (b - sqrt(disc));
    let n = normalize(hit - in.centre);
    let clip = u.proj * vec4f(hit, 1.0);
    return Fragment(premultiplied(shade(n, -dir, in.color.rgb), in.color.a), 0.5 * (clip.z / clip.w + 1.0), mask);
  }
`;

const MESH_WGSL = COMMON_WGSL + /* wgsl */ `
  @group(1) @binding(0) var<storage, read> corners: array<vec4f>; // per corner: xyz, flat; normal; linear RGB, alpha

  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) vpos: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) @interpolate(flat) flatShade: f32,
  }

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32) -> Varyings {
    let p = corners[3u * v];
    let vp = u.view * vec4f(p.xyz, 1.0);
    var out: Varyings;
    out.position = u.proj * vp;
    out.position.z = 0.5 * (out.position.z + out.position.w);
    out.vpos = vp.xyz;
    out.normal = (u.view * vec4f(corners[3u * v + 1u].xyz, 0.0)).xyz;
    out.color = corners[3u * v + 2u];
    out.flatShade = p.w;
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    let face = cross(dpdx(in.vpos), dpdy(in.vpos));
    var n = select(in.normal, face, in.flatShade > 0.5);
    n = n / max(length(n), 1e-20);
    let v = -normalize(in.vpos);
    if (dot(n, v) < 0.0) {
      n = -n; // two-sided
    }
    return premultiplied(shade(n, v, in.color.rgb), in.color.a);
  }
`;

const LINE_WGSL = COMMON_WGSL + /* wgsl */ `
  @group(1) @binding(0) var<storage, read> segs: array<vec4f>; // per segment: a (xyz, width CSS px), b (xyz, depth bias), colour a, colour b

  struct Varyings { @builtin(position) position: vec4f, @location(0) color: vec4f }

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Varyings {
    let A = segs[4u * i];
    let B = segs[4u * i + 1u];
    let toward = 1.0 - B.w; // pulled towards the eye: edges over the faces they bound
    let a = u.proj * vec4f((u.view * vec4f(A.xyz, 1.0)).xyz * toward, 1.0);
    let b = u.proj * vec4f((u.view * vec4f(B.xyz, 1.0)).xyz * toward, 1.0);
    let half = 0.5 * u.size;
    let sa = a.xy / a.w * half;
    let sb = b.xy / b.w * half;
    let len = length(sb - sa);
    let dir = select(vec2f(1.0, 0.0), (sb - sa) / len, len > 1e-6);
    let normal = vec2f(-dir.y, dir.x);
    let hw = 0.5 * A.w * u.css;
    var ends = array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    var sides = array(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let isEnd = ends[v] > 0.5;
    let clip = select(a, b, isEnd);
    let screen = select(sa - dir * hw, sb + dir * hw, isEnd) + normal * hw * sides[v];
    var out: Varyings;
    out.position = vec4f(screen / half * clip.w, 0.5 * (clip.z + clip.w), clip.w);
    out.color = select(segs[4u * i + 2u], segs[4u * i + 3u], isEnd);
    if (a.w <= 0.0 || b.w <= 0.0) {
      out.position = vec4f(0.0, 0.0, 2.0, 1.0);
    }
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    return premultiplied(in.color.rgb, in.color.a);
  }
`;

// --- Data -------------------------------------------------------------------------------------------

const num = (v: Value | undefined): number => (v === null || v === undefined || v === "" ? NaN : Number(v));

/** A mark's data as columns (rows of objects transposed), all of one length. */
function toColumns(d: Columns | Record<string, Value>[], where: string): Columns {
  const cols: Columns = Object.create(null);
  if (Array.isArray(d)) {
    const keys = [...new Set(d.flatMap((row) => Object.keys(row ?? {})))];
    for (const k of keys) cols[k] = d.map((row) => row?.[k] ?? null);
  } else {
    for (const [k, v] of Object.entries(d)) {
      if (!Array.isArray(v)) throw new Error(`${where}: data.${k} is not an array (give columns {name: [...]} or rows [{...}])`);
      cols[k] = v;
    }
  }
  const lens = new Set(Object.values(cols).map((c) => c.length));
  if (lens.size > 1) throw new Error(`${where}: data columns of different lengths (${[...lens].join(", ")})`);
  return cols;
}

/** Generated data (the gallery's, so controls stay small), seeded: `ball` (x, y, z: uniform in a
 * ball of radius 10), `torus` (x, y, z, angle: points
 * near a torus, R = 3, r = 1, `noise` (0.08) normal off it), `clusters` (x, y, z, cluster: `k` (3)
 * Gaussian blobs of n / k points each). */
function generate(g: NonNullable<SceneMark["gen"]>, where: string): Columns {
  const n = Math.trunc(g.n ?? 2000);
  if (!(n >= 1 && n <= MAX_ROWS)) throw new Error(`${where}: gen.n must be in [1, ${MAX_ROWS}]`);
  const rnd = normalRng(g.seed ?? 1);
  const x: number[] = [], y: number[] = [], z: number[] = [];
  if (g.name === "ball") {
    while (x.length < n) {
      const p = [rnd(), rnd(), rnd()], len = Math.hypot(...p) || 1;
      const r = 10 * Math.cbrt(Math.exp(-(rnd() ** 2 + rnd() ** 2) / 2)); // exp(−χ²₂ / 2) is uniform
      [x[x.length], y[y.length], z[z.length]] = p.map((v) => (v / len) * r);
    }
    return { x, y, z };
  }
  if (g.name === "torus") {
    const angle: number[] = [];
    const s = g.noise ?? 0.08;
    for (let i = 0; i < n; i += 1) {
      const u = Math.atan2(rnd(), rnd()), v = Math.atan2(rnd(), rnd());
      const rr = 1 + s * rnd(), ring = 3 + rr * Math.cos(v);
      x.push(ring * Math.cos(u));
      y.push(ring * Math.sin(u));
      z.push(rr * Math.sin(v) + s * rnd());
      angle.push((u * 180) / Math.PI);
    }
    return { x, y, z, angle };
  }
  if (g.name === "clusters") {
    const k = Math.max(1, Math.min(7, Math.trunc(g.k ?? 3)));
    const cluster: string[] = [];
    const names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"];
    const centres = Array.from({ length: k }, (_, j) => [3 * Math.cos((2 * Math.PI * j) / k), 3 * Math.sin((2 * Math.PI * j) / k), (j % 2 ? 1 : -1) * 1.2]);
    for (let i = 0; i < n; i += 1) {
      const j = i % k, c = centres[j], sd = 0.6 + 0.25 * j;
      x.push(c[0] + sd * rnd());
      y.push(c[1] + sd * rnd());
      z.push(c[2] + sd * 0.7 * rnd());
      cluster.push(names[j]);
    }
    return { x, y, z, cluster };
  }
  throw new Error(`${where}: gen.name must be ball, torus or clusters`);
}

/** #rgb or #rrggbb as linear RGB, or null if `v` is not a colour. */
function hexColor(v: string): [number, number, number] | null {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(v.trim());
  if (!m) return null;
  const h = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1];
  return [0, 2, 4].map((i) => srgbToLinear(parseInt(h.slice(i, i + 2), 16) / 255)) as [number, number, number];
}

/** A mark resolved: positions (data space), colours, and what hover reports. */
interface Built {
  index: number;
  type: "dot" | "line" | "mesh";
  label?: string;
  cols: Columns; // for hover: every column of a row
  pos: Float64Array; // data space, xyz per row
  fillCol?: Value[];
  fillName?: string;
  constColor?: [number, number, number];
  color: Float32Array; // linear RGBA per row
  radius?: Float32Array; // dot: CSS px per row
  opacity: number;
  faces?: number[][]; // mesh
  vertices?: number[]; // mesh: the rows its faces use (hover)
  flat: boolean;
  edges?: [number, number, number] | null;
  width: number;
  groups?: number[][]; // line: rows of each polyline, in order
  groupKeys?: Value[];
}

function channel(m: SceneMark, k: "x" | "y" | "z", cols: Columns, n: number, where: string): Float64Array {
  const v = m[k] ?? k;
  if (typeof v === "number") return new Float64Array(n).fill(v);
  const col = cols[v];
  if (!col) throw new Error(`${where}: ${k}: "${v}" is not a column (columns: ${Object.keys(cols).join(", ") || "none"})`);
  return Float64Array.from(col, num);
}

function resolveMark(m: SceneMark, index: number): Built {
  const where = `marks[${index}] (${m.type})`;
  if (!["dot", "line", "mesh"].includes(m.type)) throw new Error(`${where}: type must be dot, line or mesh`);
  let cols: Columns;
  if (m.gen) cols = generate(m.gen, where);
  else if (m.file) cols = toColumns(tableFile(m.file), where);
  else if (m.data) cols = toColumns(m.data, where);
  else if (m.vertices) {
    if (!Array.isArray(m.vertices) || m.vertices.some((p) => !Array.isArray(p) || p.length !== 3)) throw new Error(`${where}: vertices must be [[x, y, z], ...]`);
    cols = { x: m.vertices.map((p) => p[0]), y: m.vertices.map((p) => p[1]), z: m.vertices.map((p) => p[2]) };
  } else throw new Error(`${where}: give data (columns or rows), file, or vertices`);
  const n = Object.values(cols)[0]?.length ?? 0;
  if (n > MAX_ROWS) throw new Error(`${where}: ${n} rows; at most ${MAX_ROWS}`);
  const [X, Y, Z] = (["x", "y", "z"] as const).map((k) => channel(m, k, cols, n, where));
  const pos = new Float64Array(3 * n);
  for (let i = 0; i < n; i += 1) [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]] = [X[i], Y[i], Z[i]];
  const b: Built = {
    index, type: m.type, label: m.label, cols, pos, color: new Float32Array(4 * n), opacity: Math.min(Math.max(m.opacity ?? 1, 0), 1),
    flat: (m.shading ?? "flat") === "flat", width: m.width ?? (m.type === "line" ? 2 : 1), edges: null,
  };
  const fill = m.fill ?? m.stroke;
  if (typeof fill === "string" && Object.hasOwn(cols, fill)) [b.fillCol, b.fillName] = [cols[fill], fill];
  else if (typeof fill === "string") {
    b.constColor = hexColor(fill) ?? undefined;
    if (!b.constColor) throw new Error(`${where}: fill "${fill}" is neither a column nor a #rrggbb colour`);
  } else if (fill !== undefined) throw new Error(`${where}: fill must be a column name or a #rrggbb colour`);
  if (m.edges) {
    b.edges = typeof m.edges === "string" ? hexColor(m.edges) : hexColor(EDGE_DEFAULT);
    if (!b.edges) throw new Error(`${where}: edges must be true or a #rrggbb colour`);
  }
  if (m.transform?.hull) {
    if (m.type === "line") throw new Error(`${where}: hull applies to dot or mesh marks`);
    const finite = Array.from({ length: n }, (_, i) => i).filter((i) => Number.isFinite(pos[3 * i]) && Number.isFinite(pos[3 * i + 1]) && Number.isFinite(pos[3 * i + 2]));
    let hull: ReturnType<typeof convexHull>;
    try {
      hull = convexHull(finite.flatMap((i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]));
    } catch (err) {
      throw new Error(`${where}: ${(err as Error).message}`);
    }
    b.type = "mesh";
    b.faces = hull.faces.map((f) => f.map((k) => finite[k]));
    if (m.opacity === undefined) b.opacity = 0.35;
  } else if (m.transform && Object.keys(m.transform).length) {
    throw new Error(`${where}: unknown transform ${Object.keys(m.transform).join(", ")} (known: hull)`);
  }
  if (b.type === "mesh" && !b.faces) {
    if (!Array.isArray(m.faces)) throw new Error(`${where}: a mesh needs faces [[i, j, k], ...] (or transform: {hull: true})`);
    b.faces = m.faces.map((f, k) => {
      if (!Array.isArray(f) || f.length !== 3 || f.some((i) => !Number.isInteger(i) || i < 0 || i >= n)) {
        throw new Error(`${where}: faces[${k}] must be 3 row indices in [0, ${n - 1}]`);
      }
      return f;
    });
  }
  if (b.faces) b.vertices = [...new Set(b.faces.flat())];
  if (b.type === "dot") {
    const r = m.r ?? 3;
    if (typeof r === "number") b.radius = new Float32Array(n).fill(r);
    else if (!Object.hasOwn(cols, r)) throw new Error(`${where}: r: "${r}" is not a column`);
  }
  if (b.type === "line") {
    const g = m.group ?? (b.fillCol?.some((v) => typeof v === "string") ? b.fillName : undefined);
    if (g !== undefined && !Object.hasOwn(cols, g)) throw new Error(`${where}: group: "${g}" is not a column`);
    const key = g === undefined ? undefined : cols[g];
    const groups = new Map<Value, number[]>();
    for (let i = 0; i < n; i += 1) {
      const k = key ? key[i] : null;
      (groups.get(k) ?? groups.set(k, []).get(k)!).push(i);
    }
    [b.groupKeys, b.groups] = [[...groups.keys()], [...groups.values()]];
  }
  return b;
}
const EDGE_DEFAULT = "#c9d1d9";

// --- Renderer -----------------------------------------------------------------------------------------

interface ColorBar { label: string; scheme: string; invert: boolean; domain: [number, number] }
interface Box { lo: Vec3; hi: Vec3; centre: Vec3; scale: Vec3 }

/** Opaque geometry, and the translucent kept on the CPU too: sorted back to front as the view changes. */
interface Gpu {
  dots: { buf: GPUBuffer; colors: GPUBuffer; count: number; group?: GPUBindGroup };
  mesh: { buf: GPUBuffer; count: number; group?: GPUBindGroup };
  segs: { buf: GPUBuffer; count: number; group?: GPUBindGroup };
  blend: Blend;
}
interface Blend {
  dots: Float32Array; // 8 per dot: xyz, r, rgba
  tris: Float32Array; // 36 per triangle: 3 corners as the mesh buffer's
  segs: Float32Array; // 16 per segment, as the line buffer's
  bufs: GPUBuffer[]; // dot positions, dot colours, triangles, segments (sorted)
  groups?: GPUBindGroup[];
  runs: [0 | 1 | 2, number, number][]; // kind (dot, triangle, segment), first, count: in draw order
  key?: string; // the view they were sorted for
}

export class ScenePlot implements ServeRenderer<Control> {
  private readonly overlay: OverlayRenderer;
  private readonly canvas: TextureTarget;
  private readonly frameReads: GPUBuffer[] = [];
  private readonly uniform: GPUBuffer;
  private readonly pipes: Record<"line" | "lineBlend" | "dot" | "dotBlend" | "mesh" | "meshBlend", GPURenderPipeline>;
  private targets?: RenderTargets;
  private dataKey?: string;
  private built: Built[] = [];
  private gpu?: Gpu;
  private axes?: { buf: GPUBuffer; count: number; group: GPUBindGroup };
  private box: Box = { lo: [-1, -1, -1], hi: [1, 1, 1], centre: [0, 0, 0], scale: [1, 1, 1] };
  private cats: Value[] = [];
  private bar?: ColorBar;
  private heldFit?: [Vec3, number];
  private vp: number[] = [];
  private viewM: number[] = []; // world to view
  private proj: number[] = [];
  private size: [number, number] = [1, 1];
  private scale = 1;
  private plot: [number, number, number, number] = [0, 0, 1, 1];
  private content: [number, number, number, number] | null = null;
  private distance = 1;
  private target: Vec3 = [0, 0, 0];
  private hoverInfo: { at: [number, number]; pixel: [number, number]; rows: [string, string][]; color: Rgba; state: unknown } | null = null;

  constructor(private readonly device: GPUDevice, private readonly font: BitmapFont) {
    this.overlay = new OverlayRenderer(device, font);
    this.canvas = new TextureTarget(device, FRAME_FORMAT);
    this.uniform = createBuffer(device, 144, UNIFORM);
    const blend: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    const make = (code: string, opts: { blend?: boolean } = {}) => {
      const module = device.createShaderModule({ code });
      return device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: { module, entryPoint: "fragmentMain", targets: [{ format: FRAME_FORMAT, ...(opts.blend ? { blend } : {}) }] },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: !opts.blend, depthCompare: "less-equal" },
        multisample: { count: SAMPLE_COUNT },
      });
    };
    this.pipes = {
      line: make(LINE_WGSL),
      dot: make(DOT_WGSL),
      dotBlend: make(DOT_WGSL, { blend: true }),
      mesh: make(MESH_WGSL),
      meshBlend: make(MESH_WGSL, { blend: true }),
      lineBlend: make(LINE_WGSL, { blend: true }),
    };
  }

  static async create(): Promise<ScenePlot> {
    const device = await requestGpuDevice();
    if (!device) throw new Error("No WebGPU device");
    return new ScenePlot(device, await BitmapFont.load(new URL("./assets/", import.meta.url)));
  }

  // Data ---------------------------------------------------------------------------------------------

  private setData(ctl: Control): void {
    const marks = ctl.marks ?? [];
    if (!Array.isArray(marks)) throw new Error("marks must be an array");
    const mtimes = marks.map((m) => { try { return m?.file ? Deno.statSync(m.file).mtime?.getTime() : 0; } catch { return -1; } });
    const key = JSON.stringify([marks, mtimes, ctl.aspect, ctl.color, ctl.r]);
    if (key === this.dataKey) return;
    const built = marks.map((m, i) => resolveMark(m, i));
    // box: every finite point, padded; aspect: equal, auto (each axis normalised) or given
    const lo: Vec3 = [Infinity, Infinity, Infinity], hi: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const b of built) {
      for (let i = 0; i < b.pos.length; i += 3) {
        if (!(Number.isFinite(b.pos[i]) && Number.isFinite(b.pos[i + 1]) && Number.isFinite(b.pos[i + 2]))) continue;
        for (let k = 0; k < 3; k += 1) [lo[k], hi[k]] = [Math.min(lo[k], b.pos[i + k]), Math.max(hi[k], b.pos[i + k])];
      }
    }
    for (let k = 0; k < 3; k += 1) {
      if (!(hi[k] >= lo[k])) [lo[k], hi[k]] = [-1, 1];
      const half = hi[k] > lo[k] ? (hi[k] - lo[k]) * PAD : Math.max(Math.abs(lo[k]) * 0.05, 0.5);
      [lo[k], hi[k]] = [lo[k] - half, hi[k] + half];
    }
    const span = [0, 1, 2].map((k) => hi[k] - lo[k]);
    const a = ctl.aspect;
    let scale: Vec3 = [1, 1, 1];
    if (a === "auto") scale = span.map((s) => Math.max(...span) / s) as Vec3;
    else if (Array.isArray(a)) scale = [0, 1, 2].map((k) => Number(a[k]) || 1) as Vec3;
    else if (a !== "equal" && a !== undefined && a !== null) throw new Error(`aspect must be "equal", "auto" or [ax, ay, az]`);
    const box: Box = { lo, hi, centre: [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2) as Vec3, scale };
    const { cats, bar } = this.colorize(built, ctl);
    const gpu = this.upload(built, box);
    // swap
    this.disposeData();
    Object.assign(this, { built, box, cats, bar, gpu, dataKey: key });
  }

  /** Colours of every row: categorical (shared palette order), numeric (one colour map and bar), constant, default. */
  private colorize(built: Built[], ctl: Control): { cats: Value[]; bar?: ColorBar } {
    const scale = ctl.color ?? {};
    const isCat = (b: Built) => Boolean(b.fillCol?.some((v) => typeof v === "string"));
    let cats: Value[] = [];
    if (scale.domain && built.some(isCat)) cats = [...scale.domain];
    for (const b of built) if (isCat(b)) for (const v of b.fillCol!) if (v !== null && !cats.includes(v)) cats.push(v);
    if (cats.length > PALETTE.length) {
      throw new Error(`fill: ${cats.length} categories (${cats.slice(0, 12).join(", ")}${cats.length > 12 ? ", ..." : ""}); the palette has ${PALETTE.length}: fold the rest into "other", or colour by a number`);
    }
    const numeric = built.filter((b) => b.fillCol && !isCat(b));
    let bar: ColorBar | undefined;
    if (numeric.length) {
      let [d0, d1] = [Infinity, -Infinity];
      for (const b of numeric) for (const v of b.fillCol!) if (Number.isFinite(num(v))) [d0, d1] = [Math.min(d0, num(v)), Math.max(d1, num(v))];
      const given = !isCat(numeric[0]) && scale.domain?.length === 2 ? scale.domain.map(Number) : null;
      if (given && given.every(Number.isFinite)) [d0, d1] = given;
      if (!(d1 >= d0)) [d0, d1] = [0, 1];
      if (d1 === d0) [d0, d1] = [d0 - 0.5, d1 + 0.5];
      bar = { label: scale.label ?? numeric[0].fillName!, scheme: colormapName(String(scale.scheme ?? "viridis")) ?? "Viridis", invert: Boolean(scale.invert), domain: [d0, d1] };
    }
    const palette = PALETTE.map((h) => hexColor(h)!);
    const nan = hexColor("#6e7681")!;
    const rRange = ctl.r?.range ?? [2, 12];
    for (const b of built) {
      const n = b.pos.length / 3;
      const alpha = b.opacity;
      const cat = isCat(b);
      for (let i = 0; i < n; i += 1) {
        let c: [number, number, number];
        if (b.fillCol && cat) c = palette[cats.indexOf(b.fillCol[i])] ?? nan;
        else if (b.fillCol) {
          const v = num(b.fillCol[i]);
          if (!Number.isFinite(v)) c = nan;
          else {
            const t = (v - bar!.domain[0]) / (bar!.domain[1] - bar!.domain[0]);
            c = colormapRgb(bar!.scheme, t, bar!.invert).map((x) => srgbToLinear(x / 255)) as [number, number, number];
          }
        } else c = b.constColor ?? (cats.length ? hexColor(INK)! : palette[0]);
        b.color.set([c[0], c[1], c[2], alpha], 4 * i);
      }
      if (b.type === "dot" && !b.radius) { // a column: sqrt scale
        const mark = (ctl.marks as SceneMark[])[b.index];
        const col = b.cols[mark.r as string].map(num);
        const domain: [number, number] = [ctl.r?.domain?.[0] ?? 0, ctl.r?.domain?.[1] ?? col.reduce((m, v) => (v > m ? v : m), 0)];
        b.radius = Float32Array.from(col, (v) => (Number.isFinite(v) ? sqrtRadius(v, domain, rRange) : 0));
      }
    }
    return { cats, bar };
  }

  /** World position of a data point: about the box centre, times the aspect. */
  private static world(box: Box, p: ArrayLike<number>, i: number): Vec3 {
    return [0, 1, 2].map((k) => (p[3 * i + k] - box.centre[k]) * box.scale[k]) as Vec3;
  }

  private upload(built: Built[], box: Box): Gpu {
    const d = this.device;
    const dotPos: number[] = [], dotCol: number[] = [], meshData: number[] = [], segs: number[] = []; // opaque
    const tDots: number[] = [], tTris: number[] = [], tSegs: number[] = []; // translucent
    const seg = (a: Vec3, b: Vec3, ca: ArrayLike<number>, cb: ArrayLike<number>, width: number, bias: number) =>
      (ca[3] < 1 || cb[3] < 1 ? tSegs : segs).push(...a, width, ...b, bias, ca[0], ca[1], ca[2], ca[3], cb[0], cb[1], cb[2], cb[3]);
    const ok = (p: Vec3) => p.every(Number.isFinite);
    for (const b of built) {
      const n = b.pos.length / 3;
      const W = (i: number) => ScenePlot.world(box, b.pos, i);
      if (b.type === "dot") {
        for (let i = 0; i < n; i += 1) {
          const p = W(i);
          if (!ok(p) || !(b.radius![i] > 0) || !(b.color[4 * i + 3] > 0)) continue;
          const col = [b.color[4 * i], b.color[4 * i + 1], b.color[4 * i + 2], b.color[4 * i + 3]];
          if (b.opacity < 1) tDots.push(...p, b.radius![i], ...col);
          else dotPos.push(...p, b.radius![i]), dotCol.push(...col);
        }
      } else if (b.type === "line") {
        for (const g of b.groups!) {
          for (let j = 0; j + 1 < g.length; j += 1) {
            const [p, q] = [W(g[j]), W(g[j + 1])];
            if (ok(p) && ok(q)) seg(p, q, b.color.subarray(4 * g[j]), b.color.subarray(4 * g[j + 1]), b.width, 0);
          }
        }
      } else {
        const tris = b.opacity < 1 ? tTris : meshData;
        // smooth: area-weighted vertex normals
        const normals = new Float64Array(3 * n);
        const faceN = (f: number[]) => {
          const [p, q, r] = f.map(W);
          const u = [q[0] - p[0], q[1] - p[1], q[2] - p[2]], v = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
          return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        };
        const faces = b.faces!.filter((f) => f.every((i) => ok(W(i)))); // drawable: finite corners
        if (!b.flat) for (const f of faces) { const fn = faceN(f); for (const i of f) for (let c = 0; c < 3; c += 1) normals[3 * i + c] += fn[c]; }
        const edges = new Set<string>();
        for (const f of faces) {
          const ps = f.map(W);
          if (b.opacity > 0) f.forEach((i, c) => tris.push(...ps[c], b.flat ? 1 : 0, normals[3 * i], normals[3 * i + 1], normals[3 * i + 2], 0, b.color[4 * i], b.color[4 * i + 1], b.color[4 * i + 2], b.color[4 * i + 3]));
          if (b.edges) {
            for (let c = 0; c < 3; c += 1) {
              const [i, j] = [f[c], f[(c + 1) % 3]];
              const e = i < j ? `${i},${j}` : `${j},${i}`;
              if (edges.has(e)) continue;
              edges.add(e);
              const col = [...b.edges, 1];
              seg(ps[c], ps[(c + 1) % 3], col, col, b.width, 2e-4);
            }
          }
        }
      }
    }
    const storage = (data: number[]) => createBuffer(d, data.length ? Float32Array.from(data) : 16, STORAGE); // (writing nothing panics)
    const blend: Blend = {
      dots: Float32Array.from(tDots), tris: Float32Array.from(tTris), segs: Float32Array.from(tSegs), runs: [],
      bufs: [4, 4, 36, 16].map((k, j) => createBuffer(d, Math.max(4 * k * [tDots.length / 8, tDots.length / 8, tTris.length / 36, tSegs.length / 16][j], 16), STORAGE)),
    };
    return {
      dots: { buf: storage(dotPos), colors: storage(dotCol), count: dotPos.length / 4 },
      mesh: { buf: storage(meshData), count: meshData.length / 12 },
      segs: { buf: storage(segs), count: segs.length / 16 },
      blend,
    };
  }

  // Frame ---------------------------------------------------------------------------------------------

  private fonts(s: number): Record<"tick" | "title" | "sub" | "label", FontSize> {
    return {
      tick: this.font.get("regular", 13 * s),
      title: this.font.get("bold", 19 * s),
      sub: this.font.get("regular", 14 * s),
      label: this.font.get("regular", 14 * s),
    };
  }

  /** Legend rows: categories, then labelled marks of one colour. */
  private legendRows(ctl: Control): [string, string][] {
    if (ctl.legend === false) return [];
    const rows: [string, string][] = this.cats.map((c, i) => [String(c), PALETTE[i]]);
    for (const b of this.built) {
      const m = (ctl.marks as SceneMark[])[b.index];
      if (!b.label || m.legend === false || (b.fillCol && !b.constColor)) continue;
      const fill = m.fill ?? m.stroke;
      rows.push([b.label, typeof fill === "string" && fill.startsWith("#") ? fill : this.cats.length ? INK : PALETTE[0]]);
    }
    return rows;
  }

  private project(p: Vec3): [number, number, number] | null {
    const e = this.vp;
    const w = e[3] * p[0] + e[7] * p[1] + e[11] * p[2] + e[15];
    if (!(w > 1e-9)) return null;
    const x = (e[0] * p[0] + e[4] * p[1] + e[8] * p[2] + e[12]) / w;
    const y = (e[1] * p[0] + e[5] * p[1] + e[9] * p[2] + e[13]) / w;
    return [((x + 1) / 2) * this.size[0], ((1 - y) / 2) * this.size[1], w];
  }

  /** The camera, the axes' segments and labels, the chrome. */
  private frame(ctl: Control, q: Quads, w: number, h: number, s: number): number[] {
    const F = this.fonts(s);
    const legend = this.legendRows(ctl);
    const legendW = legend.length ? Math.max(...legend.map(([t]) => this.font.measure(F.label, t))) + Math.round(34 * s) : 0;
    const right = Math.max(this.bar ? Math.round(110 * s) : 0, legendW ? legendW + Math.round(16 * s) : 0);
    const [L, T, R, B] = [Math.round(56 * s), Math.round(64 * s), w - Math.round(56 * s) - right, h - Math.round(44 * s)];
    this.plot = [L, T, R, B];
    const [pw, ph] = [Math.max(R - L, 16), Math.max(B - T, 16)];
    const { lo, hi, centre, scale } = this.box;
    const W = (v: number[]): Vec3 => [0, 1, 2].map((k) => (v[k] - centre[k]) * scale[k]) as Vec3;
    const corners: number[] = [];
    for (let c = 0; c < 8; c += 1) corners.push(...W([c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]]).map((v) => v * FIT_MARGIN));
    const fit = (ctl.interacting && this.heldFit) || (fitView(corners, ...fitAngles(ctl), FOV, pw / ph) ?? [[0, 0, 0], 3]);
    this.heldFit = fit;
    this.target = fit[0];
    this.distance = fit[1] / (Number(ctl.zoom) || 1);
    const [f] = cameraBasis(ctl.azimuth, ctl.elevation);
    const eye = [0, 1, 2].map((i) => this.target[i] + this.distance * f[i]) as Vec3;
    const near = this.distance / 100;
    const projection = perspectiveProjection(FOV, pw / ph, near, this.distance * 100);
    const toRect = [pw / w, 0, 0, 0, 0, ph / h, 0, 0, 0, 0, 1, 0, (L + R) / w - 1, 1 - (T + B) / h, 0, 1];
    const proj = mul4(toRect, projection);
    const view = lookAtView(eye, this.target, [0, 0, 1]);
    this.vp = mul4(proj, view);
    [this.viewM, this.proj] = [view, proj];
    this.content = screenRect(corners, this.vp, w, h);
    this.device.queue.writeBuffer(this.uniform, 0, Float32Array.from([...view, ...proj, w, h, (proj[5] * h) / 2, s]));

    // Axes: the back panes (grid lines at the ticks), tick labels on the edges facing the camera
    const segs: number[] = [];
    if (ctl.axes !== false) {
      const back = [0, 1, 2].map((k) => (f[k] * scale[k] > 0 ? lo[k] : hi[k])); // the far side of each axis
      const front = [0, 1, 2].map((k) => (back[k] === lo[k] ? hi[k] : lo[k]));
      const lin = (hex: string) => [...hexColor(hex)!, 1];
      const line = (a: number[], b: number[], hex: string, width: number) => segs.push(...W(a), width, ...W(b), 0, ...lin(hex), ...lin(hex));
      // edges of interest and their ticks: x and y along the floor's front edges, z up a side
      const edge = (k: number, at: number[]): { a: number[]; b: number[]; k: number } => {
        const a = [...at], b = [...at];
        [a[k], b[k]] = [lo[k], hi[k]];
        return { a, b, k };
      };
      const sx = this.project(W([front[0], back[1], back[2]]));
      const sy = this.project(W([back[0], front[1], back[2]]));
      const zAt = sx && sy && sx[0] < sy[0] ? [front[0], back[1], 0] : [back[0], front[1], 0];
      const tickEdges = [edge(0, [0, front[1], back[2]]), edge(1, [front[0], 0, back[2]]), edge(2, zAt)];
      const ticks = tickEdges.map(({ a, b, k }) => {
        const [pa, pb] = [this.project(W(a)), this.project(W(b))];
        const len = pa && pb ? Math.hypot(pb[0] - pa[0], pb[1] - pa[1]) : 0;
        return axisTicks({}, lo[k], hi[k], Math.max(len / s, 1) * 0.8);
      });
      for (let k = 0; k < 3; k += 1) { // the pane perpendicular to k, at its back
        const [i, j] = [(k + 1) % 3, (k + 2) % 3];
        const P = (vi: number, vj: number) => { const p = [0, 0, 0]; [p[k], p[i], p[j]] = [back[k], vi, vj]; return p; };
        const inner = (t: { value: number }, a: number) => Math.min(t.value - lo[a], hi[a] - t.value) > (hi[a] - lo[a]) * PAD * 1.25; // not doubling an edge (ticks at the data's extent, a pad inside)
        for (const t of ticks[i]) if (inner(t, i)) line(P(t.value, lo[j]), P(t.value, hi[j]), GRID, 1);
        for (const t of ticks[j]) if (inner(t, j)) line(P(lo[i], t.value), P(hi[i], t.value), GRID, 1);
        line(P(lo[i], lo[j]), P(hi[i], lo[j]), EDGE, 1.25);
        line(P(hi[i], lo[j]), P(hi[i], hi[j]), EDGE, 1.25);
        line(P(hi[i], hi[j]), P(lo[i], hi[j]), EDGE, 1.25);
        line(P(lo[i], hi[j]), P(lo[i], lo[j]), EDGE, 1.25);
      }
      const mid = this.project(W(centre));
      const placed: number[][] = [];
      const sep = 6 * s; // between labels: those at a shared corner of two edges kept apart
      const free = (r: number[]) => !placed.some((o) => r[0] < o[2] + sep && r[2] + sep > o[0] && r[1] < o[3] + sep && r[3] + sep > o[1]);
      const names = [ctl.xlabel, ctl.ylabel, ctl.zlabel].map((v, k) => v ?? this.defaultLabel(ctl, k));
      const fh = F.tick.ascent + F.tick.descent;
      [2, 0, 1].forEach((k) => {
        const { a, b } = tickEdges[k];
        const [pa, pb] = [this.project(W(a)), this.project(W(b))];
        if (!pa || !pb || !mid) return;
        // outward: from the box's centre, across the edge on screen
        let e = [pb[0] - pa[0], pb[1] - pa[1]];
        const el = Math.hypot(e[0], e[1]) || 1;
        e = [e[0] / el, e[1] / el];
        const m = [(pa[0] + pb[0]) / 2 - mid[0], (pa[1] + pb[1]) / 2 - mid[1]];
        let nrm = [m[0] - (m[0] * e[0] + m[1] * e[1]) * e[0], m[1] - (m[0] * e[0] + m[1] * e[1]) * e[1]];
        const nl = Math.hypot(nrm[0], nrm[1]);
        nrm = nl > 1e-6 ? [nrm[0] / nl, nrm[1] / nl] : [-e[1], e[0]];
        const gap = 10 * s;
        let reach = 0;
        for (const t of ticks[k]) {
          const p = [...a];
          p[k] = t.value;
          const sp = this.project(W(p));
          if (!sp || !t.label) continue;
          const tw = this.font.measure(F.tick, t.label);
          const along = Math.abs(nrm[0]) * tw / 2 + Math.abs(nrm[1]) * fh / 2;
          const [cx, cy] = [sp[0] + nrm[0] * (gap + along), sp[1] + nrm[1] * (gap + along)];
          const r = [cx - tw / 2 - 2 * s, cy - fh / 2, cx + tw / 2 + 2 * s, cy + fh / 2];
          if (!free(r)) continue;
          placed.push(r);
          reach = Math.max(reach, gap + 2 * along);
          q.text(t.label, cx, cy, F.tick, premul(INK2), "mm");
        }
        if (!names[k]) return;
        const lw = this.font.measure(F.label, names[k]!);
        const lh = F.label.ascent + F.label.descent;
        const off = reach + 8 * s + Math.abs(nrm[0]) * lw / 2 + Math.abs(nrm[1]) * lh / 2;
        const [cx, cy] = [(pa[0] + pb[0]) / 2 + nrm[0] * off, (pa[1] + pb[1]) / 2 + nrm[1] * off];
        placed.push([cx - lw / 2, cy - lh / 2, cx + lw / 2, cy + lh / 2]);
        q.text(names[k]!, cx, cy, F.label, premul(INK), "mm");
      });
    }

    // Title, legend, colour bar
    const ty = Math.round(28 * s);
    const tw = ctl.title ? q.text(ctl.title, Math.round(16 * s), ty, F.title, premul(INK), "lm") : 0;
    if (ctl.subtitle) q.text(ctl.subtitle, Math.round(16 * s) + tw + (tw ? Math.round(16 * s) : 0), ty, F.sub, premul(INK2), "lm");
    let top = T;
    if (legend.length) {
      const x = w - Math.round(16 * s) - legendW;
      const lh = Math.round(22 * s);
      legend.forEach(([t, hex], i) => {
        const y = top + i * lh + lh / 2;
        q.dot(x + Math.round(8 * s), y, 5 * s, 0, premul(hex), premul(hex));
        q.text(t, x + Math.round(22 * s), y, F.label, premul(INK), "lm");
      });
      top += legend.length * lh + Math.round(30 * s);
    }
    if (this.bar) this.colorBar(q, w - right + Math.round(30 * s), top + Math.round(20 * s), B - Math.round(10 * s), s);
    return segs;
  }

  private defaultLabel(ctl: Control, k: number): string {
    const ch = (["x", "y", "z"] as const)[k];
    const m = (ctl.marks as SceneMark[]).find((m) => typeof (m[ch] ?? ch) === "string" && !m.vertices);
    return m ? String(m[ch] ?? ch) : ch;
  }

  private colorBar(q: Quads, x: number, top: number, bottom: number, s: number): void {
    const bar = this.bar!;
    const bw = Math.round(14 * s);
    const F = this.fonts(s);
    if (bottom - top < 40 * s) return;
    for (let y = top; y < bottom; y += 1) {
      const [r, g, b] = colormapRgb(bar.scheme, 1 - (y + 0.5 - top) / (bottom - top), bar.invert);
      q.rect(x, y, bw, 1, [r / 255, g / 255, b / 255, 1]);
    }
    q.rect(x, top, bw, bottom - top, [0, 0, 0, 0], { border: 1, borderColor: LABEL_BORDER });
    const [d0, d1] = bar.domain;
    for (const t of axisTicks({}, d0, d1, (bottom - top) / s)) {
      const Y = Math.round(bottom - ((t.value - d0) / (d1 - d0)) * (bottom - top));
      q.rect(x + bw, Y, Math.round(4 * s), 1, premul(INK2));
      q.text(t.label, x + bw + Math.round(7 * s), Y + 0.5, F.tick, premul(INK2), "lm");
    }
    q.text(bar.label, x + bw / 2, top - Math.round(10 * s), F.sub, premul(INK), "mb");
  }

  // Hover ------------------------------------------------------------------------------------------

  /** The point under the pointer: the frontmost dot whose disc covers it, else the nearest point
   * (dots, line and mesh vertices) within reach. */
  private hover(ctl: Control): void {
    this.hoverInfo = null;
    if (!ctl.hover) return;
    const [hx, hy] = ctl.hover.map(Number);
    const s = this.scale;
    if (!(hx >= 0 && hy >= 0 && hx < this.size[0] && hy < this.size[1])) return;
    // the ray through the pointer, in view space (as the dot shader's), and its pixels per view unit
    const P = this.proj, V = this.viewM;
    const ray = [((2 * hx) / this.size[0] - 1 + P[8] - P[12]) / P[0], (1 - (2 * hy) / this.size[1] + P[9] - P[13]) / P[5], -1];
    const rl = Math.hypot(ray[0], ray[1], ray[2]);
    const pxScale = (P[5] * this.size[1]) / 2;
    let cover: { b: Built; i: number; p: [number, number, number] } | null = null;
    let near: { b: Built; i: number; p: [number, number, number]; d: number } | null = null;
    const reach = 14 * s;
    const e = this.vp, [w, h] = this.size;
    for (const b of this.built) {
      const n = b.vertices?.length ?? b.pos.length / 3;
      const { centre: c, scale: k } = this.box;
      for (let j = 0; j < n; j += 1) {
        const i = b.vertices ? b.vertices[j] : j;
        const X = (b.pos[3 * i] - c[0]) * k[0], Y = (b.pos[3 * i + 1] - c[1]) * k[1], Z = (b.pos[3 * i + 2] - c[2]) * k[2];
        if (!(b.color[4 * i + 3] > 0) || (b.radius && !(b.radius[i] > 0))) continue; // not drawn
        const cw = e[3] * X + e[7] * Y + e[11] * Z + e[15];
        const cz = (e[2] * X + e[6] * Y + e[10] * Z + e[14]) / cw;
        if (!(cw > 1e-9) || !(cz >= -1 && cz <= 1)) continue; // behind, or past the near / far planes
        const px = ((e[0] * X + e[4] * Y + e[8] * Z + e[12]) / cw + 1) / 2 * w;
        const py = (1 - (e[1] * X + e[5] * Y + e[9] * Z + e[13]) / cw) / 2 * h;
        const d = Math.hypot(px - hx, py - hy);
        if (!(d <= reach) && !(b.radius && d <= b.radius[i] * s)) continue;
        if (b.radius && d <= b.radius[i] * s + 1) { // under the pointer: where the ray meets its sphere
          const cv = [0, 1, 2].map((r) => V[r] * X + V[4 + r] * Y + V[8 + r] * Z + V[12 + r]);
          const rw = (b.radius[i] * s * -cv[2]) / pxScale;
          const t0 = (ray[0] * cv[0] + ray[1] * cv[1] + ray[2] * cv[2]) / rl;
          const t = t0 - Math.sqrt(Math.max(t0 * t0 - (cv[0] ** 2 + cv[1] ** 2 + cv[2] ** 2) + rw * rw, 0));
          if (!cover || t < cover.p[2]) cover = { b, i, p: [px, py, t] };
        }
        if (!near || d < near.d) near = { b, i, p: [px, py, cw], d };
      }
    }
    const hit = cover ?? near;
    if (!hit) return;
    const { b, i } = hit;
    const rows: [string, string][] = [];
    const values: Record<string, Value> = {};
    const names = Object.keys(b.cols);
    for (const name of names.slice(0, 12)) {
      const v = b.cols[name][i];
      values[name] = v;
      rows.push([name, typeof v === "number" ? formatNumber(v, 5) : String(v ?? "–")]);
    }
    const c = b.color.subarray(4 * i, 4 * i + 3);
    const toS = (x: number) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
    const color: Rgba = [toS(c[0]), toS(c[1]), toS(c[2]), 1];
    this.hoverInfo = { at: [hit.p[0], hit.p[1]], pixel: [hx, hy], rows, color, state: { mark: b.index, type: b.type, row: i, ...(b.label ? { label: b.label } : {}), values } };
  }

  private drawHover(q: Quads): void {
    const info = this.hoverInfo!;
    const s = this.scale;
    q.dot(info.at[0], info.at[1], 6 * s, 2 * s, info.color, premul(HOVER_RING));
    const font = this.fonts(s).tick;
    const rows = info.rows;
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

  // Draw -------------------------------------------------------------------------------------------

  private group(pipe: GPURenderPipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(1), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  private draw(axes: number[], w: number, h: number): void {
    const { device, gpu, pipes } = this;
    this.canvas.width = w;
    this.canvas.height = h;
    this.targets = getTargets(device, FRAME_FORMAT, [w, h], this.targets);
    if (axes.length && (!this.axes || this.axes.buf.size < axes.length * 4)) {
      this.axes?.buf.destroy();
      const buf = createBuffer(device, axes.length * 4 * 2, STORAGE);
      this.axes = { buf, count: 0, group: this.group(pipes.line, [buf]) };
    }
    if (this.axes) {
      if (axes.length) device.queue.writeBuffer(this.axes.buf, 0, Float32Array.from(axes));
      this.axes.count = axes.length / 16;
    }
    this.sortBlend();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.targets.color.createView(), resolveTarget: this.canvas.currentTexture().createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "discard" }],
      depthStencilAttachment: { view: this.targets.depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    const view = device.createBindGroup({ layout: pipes.line.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
    const viewOf = (p: GPURenderPipeline) => (p === pipes.line ? view : device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] }));
    const run = (pipe: GPURenderPipeline, group: GPUBindGroup | undefined, vertices: number, instances: number) => {
      if (!group || !vertices || !instances) return;
      pass.setPipeline(pipe);
      pass.setBindGroup(0, viewOf(pipe));
      pass.setBindGroup(1, group);
      pass.draw(vertices, instances);
    };
    // opaque: axes, lines and edges, dots, meshes; then translucent: back faces, dots, front faces
    if (this.axes) run(pipes.line, this.axes.group, 6, this.axes.count);
    const { dots, mesh, segs, blend } = gpu!;
    if (segs.count) run(pipes.line, segs.group ??= this.group(pipes.line, [segs.buf]), 6, segs.count);
    if (dots.count) run(pipes.dot, dots.group ??= this.group(pipes.dot, [dots.buf, dots.colors]), 6, dots.count);
    if (mesh.count) run(pipes.mesh, mesh.group ??= this.group(pipes.mesh, [mesh.buf]), mesh.count, 1);
    // translucent: back to front, in runs of one kind
    if (blend.runs.length) {
      const [dp, dc, tb, sb] = blend.bufs;
      const g = blend.groups ??= [this.group(pipes.dotBlend, [dp, dc]), this.group(pipes.meshBlend, [tb]), this.group(pipes.lineBlend, [sb])];
      const kinds = [pipes.dotBlend, pipes.meshBlend, pipes.lineBlend];
      for (const [kind, first, count] of blend.runs) {
        pass.setPipeline(kinds[kind]);
        pass.setBindGroup(0, viewOf(kinds[kind]));
        pass.setBindGroup(1, g[kind]);
        if (kind === 1) pass.draw(3 * count, 1, 3 * first, 0);
        else pass.draw(6, count, 0, first);
      }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /** The translucent dots, triangles and segments sorted back to front (by view depth of the
   * centre, centroid, midpoint), written to their buffers, and the runs of one kind to draw. */
  private sortBlend(): void {
    const b = this.gpu!.blend;
    const key = this.viewM.join();
    if (b.key === key) return;
    b.key = key;
    const [nd, nt, ns] = [b.dots.length / 8, b.tris.length / 36, b.segs.length / 16];
    const n = nd + nt + ns;
    if (!n) return;
    const v = this.viewM;
    const z = (x: number, y: number, w: number) => v[2] * x + v[6] * y + v[10] * w + v[14]; // view z: more negative, farther
    const depth = new Float64Array(n);
    for (let i = 0; i < nd; i += 1) depth[i] = z(b.dots[8 * i], b.dots[8 * i + 1], b.dots[8 * i + 2]);
    for (let i = 0; i < nt; i += 1) {
      const o = 36 * i;
      depth[nd + i] = z(b.tris[o] + b.tris[o + 12] + b.tris[o + 24], b.tris[o + 1] + b.tris[o + 13] + b.tris[o + 25], b.tris[o + 2] + b.tris[o + 14] + b.tris[o + 26]) / 3 + (v[14] * 2) / 3;
    }
    for (let i = 0; i < ns; i += 1) {
      const o = 16 * i;
      depth[nd + nt + i] = (z(b.segs[o], b.segs[o + 1], b.segs[o + 2]) + z(b.segs[o + 4], b.segs[o + 5], b.segs[o + 6])) / 2;
    }
    // sorted natively: depth quantised to 2²⁰ levels, the index in the low 21 bits (n ≤ 2²¹, exact in f64)
    let [lo, hi] = [Infinity, -Infinity];
    for (const d of depth) [lo, hi] = [Math.min(lo, d), Math.max(hi, d)];
    const order = n <= 0x200000
      ? depth.map((d, i) => Math.floor(((d - lo) / (hi - lo || 1)) * 0xfffff) * 0x200000 + i).sort().map((k) => k % 0x200000)
      : Uint32Array.from({ length: n }, (_, i) => i).sort((p, q) => depth[p] - depth[q]);
    const pos = new Float32Array(4 * nd), col = new Float32Array(4 * nd), tris = new Float32Array(36 * nt), segs = new Float32Array(16 * ns);
    const at = [0, 0, 0];
    b.runs = [];
    for (const i of order) {
      const kind = i < nd ? 0 : i < nd + nt ? 1 : 2;
      const k = at[kind]++;
      if (kind === 0) {
        pos.set(b.dots.subarray(8 * i, 8 * i + 4), 4 * k);
        col.set(b.dots.subarray(8 * i + 4, 8 * i + 8), 4 * k);
      } else if (kind === 1) tris.set(b.tris.subarray(36 * (i - nd), 36 * (i - nd + 1)), 36 * k);
      else segs.set(b.segs.subarray(16 * (i - nd - nt), 16 * (i - nd - nt + 1)), 16 * k);
      const last = b.runs[b.runs.length - 1];
      if (last?.[0] === kind) last[2] += 1;
      else b.runs.push([kind, k, 1]);
    }
    const q = this.device.queue;
    [pos, col, tris, segs].forEach((data, j) => data.length && q.writeBuffer(b.bufs[j], 0, data));
  }

  async submit(ctl: Control, slot: number): Promise<Pending> {
    const { device } = this;
    this.setData(ctl);
    const w = Math.trunc(ctl.width);
    const h = Math.trunc(ctl.height);
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    this.scale = s;
    this.size = [w, h];
    const q = new Quads(this.font);
    q.layer(0, 1, [0, 0, w, h]); // the scene under the chrome (quads draw in order)
    const axes = this.frame(ctl, q, w, h, s);
    this.draw(axes, w, h);
    this.overlay.setSize(w, h);
    this.hover(ctl);
    if (this.hoverInfo) this.drawHover(q);
    const encoder = device.createCommandEncoder();
    this.overlay.encode(encoder, q, [this.canvas.currentTexture()], [0, 0, w, h]);
    const pitch = rowBytes(w);
    if (this.frameReads[slot]?.size !== pitch * h) {
      this.frameReads[slot]?.destroy();
      this.frameReads[slot] = makeReadBuffer(device, pitch * h);
    }
    const frameRead = this.frameReads[slot];
    encoder.copyTextureToBuffer({ texture: this.overlay.out! }, { buffer: frameRead, bytesPerRow: pitch, rowsPerImage: h }, [w, h]);
    device.queue.submit([encoder.finish()]);
    const hover = this.hoverInfo?.state ?? null;
    return {
      done: frameRead.mapAsync(0x01),
      read: (alloc) => {
        const src = new Uint8Array(frameRead.getMappedRange());
        const rgba = alloc(w * h * 4);
        if (pitch === w * 4) rgba.set(src.subarray(0, w * h * 4));
        else for (let y = 0; y < h; y += 1) rgba.set(src.subarray(y * pitch, y * pitch + w * 4), y * w * 4);
        frameRead.unmap();
        return { width: w, height: h, rgba, hover };
      },
    };
  }

  state(ctl: Control, generation: number, hover: unknown, ms: number): unknown {
    const counts = this.built.map((b) => ({ type: b.type, rows: b.pos.length / 3, ...(b.faces ? { faces: b.faces.length } : {}) }));
    return {
      generation, width: this.size[0], height: this.size[1], hover, renderMs: Math.round(ms * 1000) / 1000,
      azimuth: ctl.azimuth, elevation: ctl.elevation, distance: this.distance, target: this.target,
      plotRect: this.plot, content: this.content, marks: counts,
      box: { lo: this.box.lo, hi: this.box.hi }, ...(this.bar ? { colorDomain: this.bar.domain } : {}), ...(this.cats.length ? { categories: this.cats } : {}),
    };
  }

  private disposeData(): void {
    const g = this.gpu;
    if (!g) return;
    for (const buf of [g.dots.buf, g.dots.colors, g.mesh.buf, g.segs.buf, ...g.blend.bufs]) buf.destroy();
    this.gpu = undefined;
  }
}

if (import.meta.main) {
  await main<Control>({
    name: "scene.ts",
    defaults: DEFAULTS as Control,
    create: ScenePlot.create,
    onPng(_ctl, _frame, state) {
      console.log(JSON.stringify(state));
    },
  });
}
