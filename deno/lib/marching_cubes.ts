// Marching cubes triangulations and a CPU reference. Tables come from face walks (every patch
// consistent with its neighbours), ambiguous faces are resolved by the asymptotic decider, long
// loops are fanned around a centroid; crossings within 1e-6 of a grid point snap to it.
// volume.ts runs the same scheme in a compute shader; `marchingCubes` here is its check.

/* Corner n of a cube is at (n & 1, (n >> 1) & 1, (n >> 2) & 1) */
export const CUBE_EDGES: [number, number][] = [
  [0, 1], [2, 3], [4, 5], [6, 7], // along x
  [0, 2], [1, 3], [4, 6], [5, 7], // along y
  [0, 4], [1, 5], [2, 6], [3, 7], // along z
];

/* Corners of each face, counter-clockwise seen from outside */
export const CUBE_FACES: [number, number, number, number][] = [
  [0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6],
];

const EDGE_LOOKUP = new Int8Array(64).fill(-1);
CUBE_EDGES.forEach(([a, b], i) => {
  EDGE_LOOKUP[a * 8 + b] = i;
  EDGE_LOOKUP[b * 8 + a] = i;
});

const AMBIGUOUS_FACES: number[][] = Array.from({ length: 256 }, (_, c) =>
  CUBE_FACES.flatMap((corners, f) => {
    const [i0, i1, i2, i3] = corners.map((k) => (c >> k) & 1);
    return i0 === i2 && i1 === i3 && i0 !== i1 ? [f] : [];
  }));

const MIN_CENTROID_LOOP = 6;
/** Triangle vertices >= CENTROID_BASE are centroids of `centroidLoops[i - CENTROID_BASE]`. */
export const CENTROID_BASE = 12;

export interface CaseTriangulation { triangles: Int8Array; centroidLoops: Int8Array[] }
const cache = new Map<number, CaseTriangulation>();

/** Triangles (cube edge triplets, or centroids) of a case: bit n of `caseIndex` set when corner n
 * is below the level; bit f of `connectedFaces` when the inside corners of ambiguous face f connect. */
export function getCaseTriangles(caseIndex: number, connectedFaces = 0): CaseTriangulation {
  const key = caseIndex | (connectedFaces << 8);
  const hit = cache.get(key);
  if (hit) return hit;
  const nextEdge = new Int8Array(12).fill(-1);
  const inside = (c: number) => ((caseIndex >> c) & 1) === 1;
  CUBE_FACES.forEach((corners, f) => {
    const crossings: { edge: number; entering: boolean }[] = [];
    for (let k = 0; k < 4; k += 1) {
      const a = corners[k], b = corners[(k + 1) % 4];
      if (inside(a) !== inside(b)) crossings.push({ edge: EDGE_LOOKUP[a * 8 + b], entering: inside(b) });
    }
    const connected = ((connectedFaces >> f) & 1) === 1;
    const n = crossings.length;
    crossings.forEach(({ edge, entering }, i) => {
      if (entering) nextEdge[edge] = (connected ? crossings[(i - 1 + n) % n] : crossings[(i + 1) % n]).edge;
    });
  });
  const triangles: number[] = [];
  const centroidLoops: Int8Array[] = [];
  const visited = new Uint8Array(12);
  for (let start = 0; start < 12; start += 1) {
    if (nextEdge[start] < 0 || visited[start]) continue;
    const loop: number[] = [];
    for (let e = start; !visited[e]; e = nextEdge[e]) {
      visited[e] = 1;
      loop.push(e);
    }
    if (loop.length < MIN_CENTROID_LOOP) {
      for (let i = 1; i < loop.length - 1; i += 1) triangles.push(loop[0], loop[i], loop[i + 1]);
    } else {
      const centroid = CENTROID_BASE + centroidLoops.length;
      centroidLoops.push(Int8Array.from(loop));
      loop.forEach((e, i) => triangles.push(centroid, e, loop[(i + 1) % loop.length]));
    }
  }
  const out = { triangles: Int8Array.from(triangles), centroidLoops };
  cache.set(key, out);
  return out;
}

/** Scale transforms placing vertices along edges (in the space values are colour-mapped in). */
export function scaleTransform(scale: string | [string, number] | undefined): { fn: (v: number) => number; isValid: (v: number) => boolean } {
  if (Array.isArray(scale) && scale[0] === "symlog") {
    const c = scale[1];
    return { fn: (v) => Math.sign(v) * Math.log1p(Math.abs(v) / c), isValid: Number.isFinite };
  }
  const type = Array.isArray(scale) ? "linear" : scale; // gamma: after normalization
  if (type === "log") return { fn: Math.log10, isValid: (v) => Number.isFinite(v) && v > 0 };
  if (type === "sqrt") return { fn: Math.sqrt, isValid: (v) => Number.isFinite(v) && v >= 0 };
  if (type === "symlog") return { fn: (v) => Math.sign(v) * Math.log1p(Math.abs(v)), isValid: Number.isFinite };
  return { fn: (v) => v, isValid: Number.isFinite };
}

const SNAP = 1e-6;

/** Welded isosurface of `values` (`[z][y][x]`) at `level`: positions (x, y, z index space) and
 * triangle indices (full resolution). */
export function marchingCubes(values: ArrayLike<number>, shape: [number, number, number], level: number, scale?: string | [string, number]): { positions: Float32Array; indices: Uint32Array } {
  const [nz, ny, nx] = shape;
  const { fn, isValid } = scaleTransform(scale);
  const positions: number[] = [];
  const indices: number[] = [];
  if (nx < 2 || ny < 2 || nz < 2 || !isValid(level)) return { positions: new Float32Array(0), indices: new Uint32Array(0) };
  const at = (i: number, j: number, k: number) => values[(k * ny + j) * nx + i];
  const cornerVertex = new Map<number, number>();
  const edgeVertex = new Map<string, number>();
  const corner = (i: number, j: number, k: number) => {
    const key = (k * ny + j) * nx + i;
    let v = cornerVertex.get(key);
    if (v === undefined) {
      v = positions.length / 3;
      positions.push(i, j, k);
      cornerVertex.set(key, v);
    }
    return v;
  };
  const lv = fn(level);
  const cv = new Float64Array(8);
  const verts = new Int32Array(CENTROID_BASE + 2);
  for (let k = 0; k < nz - 1; k += 1) {
    for (let j = 0; j < ny - 1; j += 1) {
      for (let i = 0; i < nx - 1; i += 1) {
        let caseIndex = 0, invalid = false;
        for (let n = 0; n < 8; n += 1) {
          const v = at(i + (n & 1), j + ((n >> 1) & 1), k + ((n >> 2) & 1));
          if (!isValid(v)) { invalid = true; break; }
          cv[n] = v;
          if (v < level) caseIndex |= 1 << n;
        }
        if (invalid || caseIndex === 0 || caseIndex === 255) continue;
        let connected = 0;
        for (const f of AMBIGUOUS_FACES[caseIndex]) {
          const [a, b, c, d] = CUBE_FACES[f].map((q) => cv[q]);
          const denom = a + c - b - d;
          if (denom !== 0 && (a * c - b * d) / denom < level) connected |= 1 << f;
        }
        const { triangles, centroidLoops } = getCaseTriangles(caseIndex, connected);
        verts.fill(-1);
        for (const e of triangles) {
          if (e >= CENTROID_BASE || verts[e] >= 0) continue;
          const [ca, cb] = CUBE_EDGES[e];
          const pa = [i + (ca & 1), j + ((ca >> 1) & 1), k + ((ca >> 2) & 1)];
          const pb = [i + (cb & 1), j + ((cb >> 1) & 1), k + ((cb >> 2) & 1)];
          const key = `${pa}|${pb}`;
          let v = edgeVertex.get(key);
          if (v === undefined) {
            const sa = fn(cv[ca]), sb = fn(cv[cb]);
            let t = (lv - sa) / (sb - sa);
            t = Number.isFinite(t) ? Math.min(Math.max(t, 0), 1) : 0.5;
            if (t <= SNAP) v = corner(pa[0], pa[1], pa[2]);
            else if (t >= 1 - SNAP) v = corner(pb[0], pb[1], pb[2]);
            else {
              v = positions.length / 3;
              positions.push(...pa.map((x, a) => x + t * (pb[a] - x)));
            }
            edgeVertex.set(key, v);
          }
          verts[e] = v;
        }
        centroidLoops.forEach((loop, c) => {
          const p = [0, 0, 0];
          for (const e of loop) for (let a = 0; a < 3; a += 1) p[a] += positions[verts[e] * 3 + a];
          verts[CENTROID_BASE + c] = positions.length / 3;
          positions.push(...p.map((x) => x / loop.length));
        });
        for (let t = 0; t < triangles.length; t += 3) {
          const [a, b, c] = [verts[triangles[t]], verts[triangles[t + 1]], verts[triangles[t + 2]]];
          if (a !== b && b !== c && a !== c) indices.push(a, b, c);
        }
      }
    }
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}
