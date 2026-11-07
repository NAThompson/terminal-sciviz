/** Data helpers of the gallery: a seeded normal generator for its noisy series, and `linspace`. */

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
