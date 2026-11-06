/** Special functions for the gallery's data (Bessel J_n, Fresnel integrals, ...), computed by
 * simple quadrature in f64, and a seeded normal generator for its noisy series. */

/** `bessel_j`: J_n(x) = 1/2π ∫ cos(nτ − x sin τ) dτ, trapezoid over a period (256 points). */
export function besselJ(n: number, x: number): number {
  let s = 0;
  for (let k = 0; k < 256; k += 1) {
    const tau = (2 * Math.PI * k) / 256;
    s += Math.cos(n * tau - x * Math.sin(tau));
  }
  return s / 256;
}

/** `fresnel`: C(t) + i S(t) = ∫₀ᵗ exp(iπs²/2) ds by cumulative trapezoid (200 001 points on
 * [0, max |t|]), interpolated linearly. Returns [re, im] arrays. */
export function fresnel(t: ArrayLike<number>): [Float64Array, Float64Array] {
  let tmax = 0;
  for (let i = 0; i < t.length; i += 1) tmax = Math.max(tmax, Math.abs(t[i]));
  const m = 200_001;
  const ds = tmax / (m - 1);
  const cre = new Float64Array(m);
  const cim = new Float64Array(m);
  let pr = 1, pi = 0;
  for (let k = 1; k < m; k += 1) {
    const s = k * ds;
    const a = (Math.PI * s * s) / 2;
    const fr = Math.cos(a), fi = Math.sin(a);
    cre[k] = cre[k - 1] + ((fr + pr) / 2) * ds;
    cim[k] = cim[k - 1] + ((fi + pi) / 2) * ds;
    pr = fr;
    pi = fi;
  }
  const re = new Float64Array(t.length);
  const im = new Float64Array(t.length);
  for (let i = 0; i < t.length; i += 1) {
    const u = Math.abs(t[i]) / ds;
    const k = Math.min(Math.floor(u), m - 2);
    const f = u - k;
    const sg = Math.sign(t[i]);
    re[i] = sg * (cre[k] + f * (cre[k + 1] - cre[k]));
    im[i] = sg * (cim[k] + f * (cim[k + 1] - cim[k]));
  }
  return [re, im];
}

/** Standard normal deviates from a seeded mulberry32 stream (Box–Muller). Not numpy's PCG64:
 * same distribution, other draws. */
export function normalRng(seed: number): () => number {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const r = Math.sqrt(-2 * Math.log(uniform()));
    const th = 2 * Math.PI * uniform();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

export function linspace(a: number, b: number, n: number, endpoint = true): number[] {
  const d = endpoint ? n - 1 : n;
  return Array.from({ length: n }, (_, i) => (endpoint && i === n - 1 ? b : a + ((b - a) * i) / d));
}
