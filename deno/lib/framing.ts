/** Camera framing: the closest camera along a view direction that keeps
 * given points in the frustum, centred on both screen axes. */

export type Vec3 = [number, number, number];

export const MAX_ELEVATION = 89.9; // degrees; z stays "up" in the image

/** The pixel rectangle [x0, y0, x1, y1] (y down) that `points` (flat world xyz) cover on screen
 * under `viewProjection` (column-major), clamped to the frame; null if none is in front. The
 * client makes only this part of a view interactive, so presses on the empty frame around an
 * object fall through to the transcript (text selection). */
export function screenRect(points: ArrayLike<number>, viewProjection: ArrayLike<number>, width: number, height: number): [number, number, number, number] | null {
  const e = viewProjection;
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < points.length; i += 3) {
    const [x, y, z] = [points[i], points[i + 1], points[i + 2]];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (!(w > 1e-9)) continue;
    const px = (((e[0] * x + e[4] * y + e[8] * z + e[12]) / w + 1) / 2) * width;
    const py = ((1 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w) / 2) * height;
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  if (!(x1 >= x0)) return null;
  const c = (v: number, hi: number) => Math.round(Math.min(hi, Math.max(0, v)));
  return [c(x0, width), c(y0, height), c(x1, width), c(y1, height)];
}

/** The view angles the camera is fitted from: `fitAt` [azimuth, elevation] when set (the client
 * pins it while orbiting, so the pivot and distance stay put), else the view's own. */
export function fitAngles(ctl: { azimuth: number; elevation: number; fitAt?: unknown }): [number, number] {
  const f = ctl.fitAt;
  return Array.isArray(f) && f.length === 2 && f.every((x) => Number.isFinite(x)) ? [f[0], f[1]] : [ctl.azimuth, ctl.elevation];
}

const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const rad = (deg: number) => (deg * Math.PI) / 180;

/** (f, x, y): f from the target towards the camera, x/y the screen's right/up axes, as
 * a look-at camera builds them with z up. The one place the view direction is derived. */
export function cameraBasis(azimuth: number, elevation: number): [Vec3, Vec3, Vec3] {
  const az = rad(azimuth);
  const el = rad(Math.max(-MAX_ELEVATION, Math.min(MAX_ELEVATION, elevation)));
  const f: Vec3 = [Math.cos(el) * Math.sin(az), -Math.cos(el) * Math.cos(az), Math.sin(el)];
  let x = cross([0, 0, 1], f);
  const n = Math.hypot(...x);
  x = [x[0] / n, x[1] / n, x[2] / n];
  return [f, x, cross(f, x)];
}

/**
 * Closest camera looking along the view direction that keeps all `points` (flat xyz) in the
 * frustum, centred on both axes: [target, distance], or null for no valid points.
 *
 * With the eye at ex x + ey y + ez f, a point p is in the frustum iff
 * |p.x - ex| <= th (ez - p.f) and |p.y - ey| <= tv (ez - p.f). Horizontally that is
 * ez >= A - ex / th and ez >= B + ex / th with A = max(p.f + p.x / th) and
 * B = max(p.f - p.x / th), so the closest eye is at ez = (A + B) / 2; same vertically.
 * The binding axis is then centred exactly; the other is centred by bisection.
 */
export function fitView(
  points: ArrayLike<number>,
  azimuth: number,
  elevation: number,
  fov: number,
  aspect: number,
): [Vec3, number] | null {
  const count = Math.floor(points.length / 3);
  if (count === 0) {
    return null;
  }
  const [f, x, y] = cameraBasis(azimuth, elevation);
  const tv = Math.tan(rad(fov) / 2);
  const th = tv * aspect;
  const pf = new Float64Array(count);
  const px = new Float64Array(count);
  const py = new Float64Array(count);
  let a = -Infinity, b = -Infinity, c = -Infinity, d = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const p: Vec3 = [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]];
    pf[i] = dot(p, f);
    px[i] = dot(p, x);
    py[i] = dot(p, y);
    a = Math.max(a, pf[i] + px[i] / th);
    b = Math.max(b, pf[i] - px[i] / th);
    c = Math.max(c, pf[i] + py[i] / tv);
    d = Math.max(d, pf[i] - py[i] / tv);
  }
  const ez = Math.max((a + b) / 2, (c + d) / 2);
  if (!(ez > 0) || !Number.isFinite(ez)) {
    return null; // e.g. a single point
  }
  const depth = pf.map((v) => Math.max(ez - v, 1e-9));
  const ex = centre(px, depth, th, th * (a - ez), th * (ez - b));
  const ey = centre(py, depth, tv, tv * (c - ez), tv * (ez - d));
  // target on the view axis, level with the origin along f
  return [[0, 1, 2].map((k) => ex * x[k] + ey * y[k]) as Vec3, ez];
}

/** Offset e in [lo, hi] at which the projections (q - e) / (t depth) span [-s, s]: their
 * max + min decreases strictly with e, so bisect. On the binding axis lo == hi. */
function centre(q: Float64Array, depth: Float64Array, t: number, lo: number, hi: number): number {
  for (let it = 0; it < 60; it += 1) {
    if (hi - lo <= 1e-12) {
      break;
    }
    const e = (lo + hi) / 2;
    let max = -Infinity;
    let min = Infinity;
    for (let i = 0; i < q.length; i += 1) {
      const ndc = (q[i] - e) / (t * depth[i]);
      max = Math.max(max, ndc);
      min = Math.min(min, ndc);
    }
    if (max + min > 0) {
      lo = e;
    } else {
      hi = e;
    }
  }
  return (lo + hi) / 2;
}
