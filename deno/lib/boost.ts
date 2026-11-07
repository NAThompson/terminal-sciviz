/**
 * Boost.Math in f64 (deno/vendor/boost-math-wasm, its WebAssembly build), instantiated once per
 * process on first import (top-level await, ~40 ms). Special functions, function curves and
 * distribution quantiles are computed with it on the CPU; only their samples reach the GPU.
 * Bad arguments give NaN (poles, out-of-domain orders), never a throw.
 */

// @ts-types="../vendor/boost-math-wasm/boost_math.d.ts"
import createModule from "../vendor/boost-math-wasm/boost_math.mjs";
import type { Distribution } from "../vendor/boost-math-wasm/boost_math.d.ts";

export const bm = await createModule();

/** `f` of the distribution `d`, freed after (embind objects live in the wasm heap). */
function using<T>(d: Distribution, f: (d: Distribution) => T): T {
  try {
    return f(d);
  } finally {
    d.delete();
  }
}

/** Φ⁻¹ at each p. */
export const normalQuantiles = (ps: ArrayLike<number>) => using(new bm.normal_distribution(0, 1), (d) => d.quantile_array(ps));
export const normalQuantile = (p: number) => normalQuantiles([p])[0];

/** The Fresnel integrals C(t) + i S(t) = ∫₀ᵗ exp(iπs²/2) ds (normalised, DLMF 7.2) at each t: [C, S]. */
export function fresnel(t: ArrayLike<number>): [Float64Array, Float64Array] {
  const cs = bm.fresnel_cis_array(t); // interleaved C0, S0, C1, S1, ...
  const c = new Float64Array(t.length), s = new Float64Array(t.length);
  for (let i = 0; i < t.length; i += 1) [c[i], s[i]] = [cs[2 * i], cs[2 * i + 1]];
  return [c, s];
}

/** Student's t quantile with `df` degrees of freedom. */
export const tQuantile = (p: number, df: number) => using(new bm.students_t_distribution(df), (d) => d.quantile(p));
