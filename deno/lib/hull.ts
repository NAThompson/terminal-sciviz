/**
 * 3D convex hull in f64 (quickhull: an initial tetrahedron, then the furthest outside point of a
 * face at a time, its visible faces replaced by a cone to the horizon). Robust to duplicates and
 * coplanar points: a point counts as outside a face only beyond `eps` (relative to the points'
 * extent), so points on or near the hull's faces are absorbed, never made vertices of
 * slivers. Coordinates are taken about their bounding-box centre for precision.
 */

export interface Hull {
  /** Triangles as indices into the input points, counter-clockwise seen from outside. */
  faces: [number, number, number][];
  /** The input indices that are hull vertices, ascending. */
  vertices: number[];
}

interface Face {
  v: [number, number, number];
  n: [number, number, number]; // unit outward normal
  d: number; // plane: n·p = d
  outside: number[];
  alive: boolean;
}

/** Hull of `points` (flat xyz, or [x, y, z] rows); throws for fewer than 4 non-coplanar points. */
export function convexHull(points: ArrayLike<number> | number[][]): Hull {
  const flat = Array.isArray(points) && Array.isArray(points[0]) ? (points as number[][]).flat() : Array.from(points as ArrayLike<number>);
  const n = Math.floor(flat.length / 3);
  // finite points only, about the centre of their box
  const idx: number[] = [];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i += 1) {
    const p = [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]];
    if (!p.every(Number.isFinite)) continue;
    idx.push(i);
    for (let k = 0; k < 3; k += 1) [lo[k], hi[k]] = [Math.min(lo[k], p[k]), Math.max(hi[k], p[k])];
  }
  if (idx.length < 4) throw new Error(`hull: need at least 4 non-coplanar points (got ${idx.length} finite)`);
  const c = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const P = new Float64Array(n * 3);
  for (const i of idx) for (let k = 0; k < 3; k += 1) P[3 * i + k] = flat[3 * i + k] - c[k];
  const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  // round-off of a plane distance: of the centred coordinates, and of the centring itself (|c|)
  const eps = Math.max(1e-10 * size, 64 * Number.EPSILON * size, 4 * Number.EPSILON * Math.max(...c.map(Math.abs)));
  if (!(size > 0)) throw new Error("hull: need at least 4 non-coplanar points (all points coincide)");

  const at = (i: number): [number, number, number] => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  const sub = (a: number[], b: number[]): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a: number[], b: number[]): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2]);

  // Initial simplex: the most distant pair of axis extremes, then furthest from their line, then from their plane
  const ext: number[] = [];
  for (let k = 0; k < 3; k += 1) {
    let a = idx[0], b = idx[0];
    for (const i of idx) {
      if (P[3 * i + k] < P[3 * a + k]) a = i;
      if (P[3 * i + k] > P[3 * b + k]) b = i;
    }
    ext.push(a, b);
  }
  let [i0, i1] = [ext[0], ext[1]];
  let best = -1;
  for (const a of ext) for (const b of ext) {
    const d = norm(sub(at(a), at(b)));
    if (d > best) [best, i0, i1] = [d, a, b];
  }
  const dir = sub(at(i1), at(i0));
  let i2 = -1;
  best = eps * norm(dir);
  for (const i of idx) {
    const d = norm(cross(dir, sub(at(i), at(i0))));
    if (d > best) [best, i2] = [d, i];
  }
  if (i2 < 0) throw new Error("hull: need at least 4 non-coplanar points (all points are collinear)");
  const n0 = cross(dir, sub(at(i2), at(i0)));
  const u0 = n0.map((v) => v / norm(n0));
  let i3 = -1;
  best = eps;
  for (const i of idx) {
    const d = Math.abs(dot(u0, sub(at(i), at(i0))));
    if (d > best) [best, i3] = [d, i];
  }
  if (i3 < 0) throw new Error("hull: need at least 4 non-coplanar points (all points are coplanar)");

  const faces: Face[] = [];
  const edges = new Map<string, number>(); // directed edge "a,b" -> face holding it (counter-clockwise)
  const key = (a: number, b: number) => `${a},${b}`;
  const makeFace = (a: number, b: number, cc: number): number => {
    const nrm = cross(sub(at(b), at(a)), sub(at(cc), at(a)));
    const len = norm(nrm) || 1;
    const u: [number, number, number] = [nrm[0] / len, nrm[1] / len, nrm[2] / len];
    const f = faces.length;
    faces.push({ v: [a, b, cc], n: u, d: dot(u, at(a)), outside: [], alive: true });
    edges.set(key(a, b), f);
    edges.set(key(b, cc), f);
    edges.set(key(cc, a), f);
    return f;
  };
  const dist = (f: Face, i: number) => P[3 * i] * f.n[0] + P[3 * i + 1] * f.n[1] + P[3 * i + 2] * f.n[2] - f.d;

  // the tetrahedron, oriented outward (i3 below the face i0 i1 i2, or the face flipped)
  const below = dot(u0, sub(at(i3), at(i0))) < 0;
  const [a, b, cc] = below ? [i0, i1, i2] : [i0, i2, i1];
  const first = [makeFace(a, b, cc), makeFace(a, i3, b), makeFace(b, i3, cc), makeFace(cc, i3, a)];
  const assign = (pts: Iterable<number>, candidates: number[]) => {
    for (const i of pts) {
      for (const f of candidates) {
        if (dist(faces[f], i) > eps) {
          faces[f].outside.push(i);
          break;
        }
      }
    }
  };
  const seed = new Set([i0, i1, i2, i3]);
  assign(idx.filter((i) => !seed.has(i)), first);

  const stack = [...first];
  let guard = 0;
  while (stack.length) {
    const fi = stack.pop()!;
    const face = faces[fi];
    if (!face.alive || !face.outside.length) continue;
    if (++guard > 4 * n + 64) throw new Error("hull: did not converge");
    // the furthest outside point of this face
    let p = face.outside[0], far = -Infinity;
    for (const i of face.outside) {
      const d = dist(face, i);
      if (d > far) [far, p] = [d, i];
    }
    // faces visible from p (flood fill across edges), and the horizon: their edges whose other face isn't
    const visible = new Set<number>([fi]);
    const queue = [fi];
    const horizon: [number, number][] = [];
    while (queue.length) {
      const f = faces[queue.shift()!];
      for (let e = 0; e < 3; e += 1) {
        const [u, v] = [f.v[e], f.v[(e + 1) % 3]];
        const g = edges.get(key(v, u))!;
        if (visible.has(g)) continue;
        if (dist(faces[g], p) > eps) {
          visible.add(g);
          queue.push(g);
        }
      }
    }
    for (const f of visible) {
      const fv = faces[f].v;
      for (let e = 0; e < 3; e += 1) {
        const [u, v] = [fv[e], fv[(e + 1) % 3]];
        if (!visible.has(edges.get(key(v, u))!)) horizon.push([u, v]);
      }
    }
    const orphans: number[] = [];
    for (const f of visible) {
      const F = faces[f];
      F.alive = false;
      for (let e = 0; e < 3; e += 1) {
        const k = key(F.v[e], F.v[(e + 1) % 3]);
        if (edges.get(k) === f) edges.delete(k);
      }
      for (const i of F.outside) if (i !== p) orphans.push(i);
      F.outside = [];
    }
    const cone = horizon.map(([u, v]) => makeFace(u, v, p));
    assign(orphans, cone);
    stack.push(...cone);
  }

  const out = faces.filter((f) => f.alive).map((f) => f.v);
  return { faces: out, vertices: [...new Set(out.flat())].sort((x, y) => x - y) };
}
