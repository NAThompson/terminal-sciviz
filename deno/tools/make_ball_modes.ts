/**
 * The gallery's file-backed 4D stack (assets/ball_modes.npy): two modes of a vibrating ball,
 * u = j₀(πr) cos 2πt + 0.8 j₁(4.4934 r) cos θ sin 2πt (Dirichlet at r = 1), on a 24³ grid over
 * [−1, 1]³ (NaN outside the ball) at t = 0, 0.1, …, 1: [11, 24, 24, 24] little-endian f32, C order.
 *
 *   deno run -A tools/make_ball_modes.ts
 */

const N = 24, T = 11;
const j0 = (x: number) => (x < 1e-6 ? 1 : Math.sin(x) / x);
const j1 = (x: number) => (x < 1e-4 ? x / 3 : Math.sin(x) / (x * x) - Math.cos(x) / x);
const data = new Float32Array(T * N * N * N);
for (let s = 0; s < T; s += 1) {
  const t = s / (T - 1);
  for (let k = 0; k < N; k += 1) for (let j = 0; j < N; j += 1) for (let i = 0; i < N; i += 1) {
    const [x, y, z] = [i, j, k].map((q) => -1 + (2 * q) / (N - 1));
    const r = Math.hypot(x, y, z);
    const u = r > 1 ? NaN : j0(Math.PI * r) * Math.cos(2 * Math.PI * t) + 0.8 * (r > 0 ? j1(4.4934 * r) * (z / r) : 0) * Math.sin(2 * Math.PI * t) / 0.4362;
    data[((s * N + k) * N + j) * N + i] = u;
  }
}
// .npy version 1.0: magic, version, little-endian u16 header length, the header dict padded to 64 bytes
let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${T}, ${N}, ${N}, ${N}), }`;
header += " ".repeat(63 - ((10 + header.length) % 64)) + "\n";
const out = new Uint8Array(10 + header.length + data.byteLength);
out.set([0x93, ...new TextEncoder().encode("NUMPY"), 1, 0, header.length & 255, header.length >> 8]);
out.set(new TextEncoder().encode(header), 10);
out.set(new Uint8Array(data.buffer), 10 + header.length);
const path = new URL("../assets/ball_modes.npy", import.meta.url).pathname;
Deno.writeFileSync(path, out);
console.log(`${path}: ${out.length} bytes`);
