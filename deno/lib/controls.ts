/**
 * Bounds on the controls of every renderer, checked before anything is allocated on the GPU
 * (lib/serve.ts calls `validateControl` ahead of each `submit`): sizes and sample counts must be
 * finite integers in range, other numbers finite. A key means the same wherever it is used (`n`:
 * plot1d's samples, complex.ts's grid); absent and null values are left to the renderer's
 * defaults.
 */

export class ControlError extends Error {
  override name = "ControlError";
}

type Check = (v: unknown) => string | null; // what is wrong, or null

const int = (lo: number, hi: number): Check => (v) =>
  typeof v === "number" && Number.isInteger(v) && v >= lo && v <= hi ? null : `must be an integer in [${lo}, ${hi}]`;
const num = (lo = -Infinity, hi = Infinity): Check => (v) =>
  typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi
    ? null
    : `must be a finite number${lo > -Infinity || hi < Infinity ? ` in [${lo}, ${hi}]` : ""}`;
const list = (max: number, item?: Check): Check => (v) => {
  if (!Array.isArray(v)) return "must be an array";
  if (v.length > max) return `must have at most ${max} items (has ${v.length})`;
  if (item) {
    for (let i = 0; i < v.length; i += 1) {
      const e = item(v[i]);
      if (e) return `[${i}] ${e}`;
    }
  }
  return null;
};

export const MAX_SIZE = 4096; // frame width and height, px
const MAX_POINTS = 1_000_000;

/** Paths (`a.b`, `a[].b` for each item of an array) and their checks. */
const RULES: [string, Check][] = [
  ["width", int(1, MAX_SIZE)],
  ["height", int(1, MAX_SIZE)],
  ["n", int(2, 4096)], // plot1d samples, complex.ts grid side
  ["resolution", int(1, 1024)], // surface.ts steps per direction
  ["antialias", int(1, 4)],
  ["slice", int(0, 1023)],
  ["field", int(0, 64)],
  ["rev", num()],
  ["hover", list(2, num())],
  ...["azimuth", "elevation", "distance", "x0", "x1", "y0", "y1", "ymin", "ymax", "threshold", "symlogThreshold", "center", "rootN"]
    .map((k): [string, Check] => [k, num()]),
  ["zoom", num(1e-3, 1e3)],
  ["fov", num(1, 179)],
  ["opacity", num(0, 1)],
  ["lineWidth", num(0, 64)],
  ["target", list(3, num())],
  ["domain", list(2)],
  ["aspect", (v) => (v === "equal" || v === "auto" ? null : typeof v === "string" ? 'must be "equal", "auto" or [ax, ay, az]' : list(3, num(1e-6, 1e6))(v))], // scene.ts: also "equal" | "auto"
  ["isoLevels", list(16, num())],
  // line.ts series: sampled generators and data
  ["series", list(64)],
  ["series[].sample.n", int(2, 65536)],
  ["series[].x", list(MAX_POINTS)],
  ["series[].y", list(MAX_POINTS)],
  ["series[].width", num(0, 64)],
  ["series[].opacity", num(0, 1)],
  ["overlays", list(256)],
  // curve.ts: data, generated curves, integrated ODEs
  ["polyline", list(MAX_POINTS, list(3, num()))],
  ["params", list(MAX_POINTS, num())],
  ["curve.degree", int(1, 16)],
  ["curve.points", list(3 * MAX_POINTS)],
  ["curve.knots", list(MAX_POINTS + 32, num())],
  ["colorBy.values", list(MAX_POINTS, num())],
  ["gen.n", int(2, MAX_POINTS)],
  ["ode.n", int(2, MAX_POINTS)],
  ["ode.dt", num(1e-9, 1)],
  ["ode.transient", int(0, MAX_POINTS)],
  // volume.ts: at most 256³ voxels per slice
  ...["size", "nr", "ntheta", "nphi"].map((k): [string, Check] => [`volume.${k}`, int(2, 256)]),
  ["volume.frames", int(1, 1024)],
  ["volume.n", int(1, 30)],
  ["volume.l", int(0, 29)],
  ["volume.m", int(-29, 29)],
  ["volume.extent", num(1e-6, 1e6)],
];

/** Values at `path` in `ctl` (absent and null ones skipped), with their concrete paths. */
function* at(value: unknown, parts: string[], path: string): Generator<[string, unknown]> {
  if (value === undefined || value === null) return;
  if (!parts.length) {
    yield [path, value];
    return;
  }
  const [head, ...rest] = parts;
  const each = head.endsWith("[]");
  const key = each ? head.slice(0, -2) : head;
  if (typeof value !== "object" || Array.isArray(value)) return;
  const child = (value as Record<string, unknown>)[key];
  const name = path ? `${path}.${key}` : key;
  if (each) {
    if (Array.isArray(child)) for (let i = 0; i < child.length; i += 1) yield* at(child[i], rest, `${name}[${i}]`);
  } else {
    yield* at(child, rest, name);
  }
}

/** Throws a `ControlError` naming the first value out of bounds. */
export function validateControl(ctl: Record<string, unknown>): void {
  for (const [path, check] of RULES) {
    for (const [where, v] of at(ctl, path.split("."), "")) {
      const error = check(v);
      if (error) throw new ControlError(`${where} ${error} (got ${(typeof v === "number" ? String(v) : String(JSON.stringify(v))).slice(0, 40)})`);
    }
  }
}
