/** 4×4 matrices, column-major (element [c * 4 + r]), with OpenGL-style projections (clip z in
 * [-w, w]), as WebGL engines build them. */

export type Mat4 = Float64Array;
export type V3 = [number, number, number];

/** From rows, as written on paper. */
export function rows(...r: number[]): Mat4 {
  const m = new Float64Array(16);
  for (let i = 0; i < 4; i += 1) for (let j = 0; j < 4; j += 1) m[j * 4 + i] = r[i * 4 + j];
  return m;
}

export const identity = (): Mat4 => rows(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
export const scaling = (x: number, y: number, z: number): Mat4 => rows(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);

export function mul(...ms: Mat4[]): Mat4 {
  return ms.reduce((a, b) => {
    const o = new Float64Array(16);
    for (let c = 0; c < 4; c += 1) {
      for (let r = 0; r < 4; r += 1) {
        let s = 0;
        for (let k = 0; k < 4; k += 1) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
    }
    return o;
  });
}

export function invert(m: Mat4): Mat4 {
  // Gauss-Jordan with partial pivoting, on rows
  const a = Array.from({ length: 4 }, (_, r) => [0, 1, 2, 3].map((c) => m[c * 4 + r]).concat([0, 1, 2, 3].map((c) => (c === r ? 1 : 0))));
  for (let c = 0; c < 4; c += 1) {
    let p = c;
    for (let r = c + 1; r < 4; r += 1) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    const d = a[c][c];
    for (let k = 0; k < 8; k += 1) a[c][k] /= d;
    for (let r = 0; r < 4; r += 1) {
      if (r === c) continue;
      const f = a[r][c];
      for (let k = 0; k < 8; k += 1) a[r][k] -= f * a[c][k];
    }
  }
  return rows(...a.flatMap((row) => row.slice(4)));
}

/** Point transform with the perspective divide. */
export function apply(m: Mat4, p: V3): V3 {
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  return [0, 1, 2].map((r) => (m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]) / w) as V3;
}

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: V3): V3 => { const n = Math.hypot(...a); return [a[0] / n, a[1] / n, a[2] / n]; };

/** Camera-to-world matrix of a camera at `eye` looking at `target` (`up` kept up). */
export function cameraWorld(eye: V3, target: V3, up: V3): Mat4 {
  const z = unit(sub(eye, target));
  const x = unit(cross(up, z));
  const y = cross(z, x);
  return rows(x[0], y[0], z[0], eye[0], x[1], y[1], z[1], eye[1], x[2], y[2], z[2], eye[2], 0, 0, 0, 1);
}

/** Perspective projection (vertical `fov` in degrees). */
export function perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan((fov * Math.PI) / 360);
  return rows(f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near), 0, 0, -1, 0);
}

// --- Perspective camera ------------------------------------------------------------------------
// Matrices of a camera at `eye` looking at `target` built step by step as WebGL engines do: the
// look-at rotation as a unit quaternion, the camera-to-world matrix composed from it (its columns
// renormalised if they don't come out of unit length), then inverted in closed form. Every step
// is kept as is so that the matrices (and the frames drawn with them) are reproducible to the bit.

type Quat = [number, number, number, number]; // x, y, z, w

const scaled = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const norm = (a: V3) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
const normalized = (a: V3): V3 => scaled(a, 1 / (norm(a) || 1));

/** Unit quaternion of the rotation in the upper 3×3 of `m` (column-major). */
function rotationQuat(m: ArrayLike<number>): Quat {
  const [m11, m12, m13] = [m[0], m[4], m[8]];
  const [m21, m22, m23] = [m[1], m[5], m[9]];
  const [m31, m32, m33] = [m[2], m[6], m[10]];
  const trace = m11 + m22 + m33;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    return [(m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s];
  }
  if (m11 > m22 && m11 > m33) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m22 - m33);
    return [0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s];
  }
  if (m22 > m33) {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m11 - m33);
    return [(m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s];
  }
  const s = 2.0 * Math.sqrt(1.0 + m33 - m11 - m22);
  return [(m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s];
}

/** Rotation `q` then translation to `p`, column-major. */
function rigid(p: V3, [x, y, z, w]: Quat): number[] {
  const [x2, y2, z2] = [x + x, y + y, z + z];
  const [xx, xy, xz] = [x * x2, x * y2, x * z2];
  const [yy, yz, zz] = [y * y2, y * z2, z * z2];
  const [wx, wy, wz] = [w * x2, w * y2, w * z2];
  return [
    1 - (yy + zz), xy + wz, xz - wy, 0,
    xy - wz, 1 - (xx + zz), yz + wx, 0,
    xz + wy, yz - wx, 1 - (xx + yy), 0,
    p[0], p[1], p[2], 1,
  ];
}

/** Inverse of a 4×4 matrix by cofactors (zero if singular), column-major. */
export function inverse4(te: ArrayLike<number>): number[] {
  const [n11, n21, n31, n41, n12, n22, n32, n42, n13, n23, n33, n43, n14, n24, n34, n44] = Array.from(te);
  const t1 = n11 * n22 - n21 * n12, t2 = n11 * n32 - n31 * n12, t3 = n11 * n42 - n41 * n12;
  const t4 = n21 * n32 - n31 * n22, t5 = n21 * n42 - n41 * n22, t6 = n31 * n42 - n41 * n32;
  const t7 = n13 * n24 - n23 * n14, t8 = n13 * n34 - n33 * n14, t9 = n13 * n44 - n43 * n14;
  const t10 = n23 * n34 - n33 * n24, t11 = n23 * n44 - n43 * n24, t12 = n33 * n44 - n43 * n34;
  const det = t1 * t12 - t2 * t11 + t3 * t10 + t4 * t9 - t5 * t8 + t6 * t7;
  if (det === 0) return Array(16).fill(0);
  const k = 1 / det;
  return [
    (n22 * t12 - n32 * t11 + n42 * t10) * k, (n31 * t11 - n21 * t12 - n41 * t10) * k,
    (n24 * t6 - n34 * t5 + n44 * t4) * k, (n33 * t5 - n23 * t6 - n43 * t4) * k,
    (n32 * t9 - n12 * t12 - n42 * t8) * k, (n11 * t12 - n31 * t9 + n41 * t8) * k,
    (n34 * t3 - n14 * t6 - n44 * t2) * k, (n13 * t6 - n33 * t3 + n43 * t2) * k,
    (n12 * t11 - n22 * t9 + n42 * t7) * k, (n21 * t9 - n11 * t11 - n41 * t7) * k,
    (n14 * t5 - n24 * t3 + n44 * t1) * k, (n23 * t3 - n13 * t5 - n43 * t1) * k,
    (n22 * t8 - n12 * t10 - n32 * t7) * k, (n11 * t10 - n21 * t8 + n31 * t7) * k,
    (n24 * t2 - n14 * t4 - n34 * t1) * k, (n13 * t4 - n23 * t2 + n33 * t1) * k,
  ];
}

/** World-to-camera (view) matrix of a camera at `eye` looking at `target`, `up` kept up. */
export function lookAtView(eye: V3, target: V3, up: V3): number[] {
  let z = sub(eye, target);
  if (norm(z) === 0) z = [z[0], z[1], 1];
  z = normalized(z);
  let x = cross(up, z);
  if (norm(x) === 0) { // up along the view direction
    z = normalized(Math.abs(up[2]) === 1 ? [z[0] + 0.0001, z[1], z[2]] : [z[0], z[1], z[2] + 0.0001]);
    x = cross(up, z);
  }
  x = normalized(x);
  const y = cross(z, x);
  const rotation = [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, 0, 0, 0, 1];
  let world = rigid(eye, rotationQuat(rotation));
  const lengths = [0, 4, 8].map((c) => norm([world[c], world[c + 1], world[c + 2]]));
  if (lengths.some((l) => l !== 1)) { // renormalised columns, as a pure rotation
    const unitCols = world.map((v, i) => (i < 12 && i % 4 < 3 ? v * (1 / lengths[i >> 2]) : v));
    world = rigid(eye, rotationQuat(unitCols));
  }
  return inverse4(world);
}

/** Perspective projection (vertical `fov` in degrees, clip z in [-w, w]), column-major. */
export function perspectiveProjection(fov: number, aspect: number, near: number, far: number): number[] {
  const top = near * Math.tan((Math.PI / 180) * 0.5 * fov);
  const height = 2 * top;
  const width = aspect * height;
  const left = -0.5 * width;
  const [right, bottom] = [left + width, top - height];
  const x = (2 * near) / (right - left);
  const y = (2 * near) / (top - bottom);
  const a = (right + left) / (right - left);
  const b = (top + bottom) / (top - bottom);
  const c = -(far + near) / (far - near);
  const d = (-2 * far * near) / (far - near);
  return [x, 0, 0, 0, 0, y, 0, 0, a, b, c, -1, 0, 0, d, 0];
}

/** Product `a × b` of two 4×4 matrices, column-major. */
export function mul4(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const out = Array<number>(16);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}
