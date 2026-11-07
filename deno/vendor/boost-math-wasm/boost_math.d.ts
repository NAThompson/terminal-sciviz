// Copyright 2026 Nick Thompson
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
//
// Types for boost_math.mjs, the Emscripten build of wasm/bindings.cpp. Everything is double precision.
//
// Memory: every array returned is a new Float64Array owned by JavaScript, so there is nothing to free. Objects
// made with `new` (distributions, fft, real_fft) live in the wasm heap: call .delete() when done with them.
//
// Errors: nothing throws for bad values. Domain errors give NaN, overflow gives ±Infinity, and a distribution
// constructed with invalid parameters gives NaN from every method. Bad sizes, shapes and axes throw a RangeError:
// the fft and real_fft constructors and transforms, the *_axis functions and the statistics say which below.

/** Input arrays: Float64Array, Float32Array, number[] or any array-like of numbers. They are copied, never kept. */
export type Samples = ArrayLike<number>;
export type WaveletFamily = "daubechies" | "symlet";
export type ThresholdRule = "wiener" | "universal_soft" | "sure" | "eb_posterior_median";

/** An object in the wasm heap, which must be freed with delete(). */
export interface Handle {
  delete(): void;
  isDeleted(): boolean;
  clone(): this;
}

/** A probability distribution. Each scalar method f(x) has a twin f_array(xs) that maps it over an array. */
export interface Distribution extends Handle {
  pdf(x: number): number;
  pdf_array(xs: Samples): Float64Array;
  cdf(x: number): number;
  cdf_array(xs: Samples): Float64Array;
  /** 1 - cdf(x), without cancellation. */
  ccdf(x: number): number;
  ccdf_array(xs: Samples): Float64Array;
  quantile(p: number): number;
  quantile_array(ps: Samples): Float64Array;
  mean(): number;
  variance(): number;
  /** NaN if Boost has no entropy for this distribution. */
  entropy(): number;
  /** [lower, upper]. */
  support(): [number, number];
}

/** Complex data are interleaved: [re0, im0, re1, im1, ...]. Transforms are unnormalized, X[k] = sum_j x[j] e^(-2 pi i jk/n). */
export interface FFT extends Handle {
  /** z holds 2n numbers; throws a RangeError otherwise. */
  transform(z: Samples): Float64Array;
  /** The inverse, including the 1/n: inverse_transform(transform(z)) = z. */
  inverse_transform(z: Samples): Float64Array;
}

export interface RealFFT extends Handle {
  /** x holds n numbers (else RangeError); returns X[0], ..., X[floor(n/2)] interleaved, 2 (floor(n/2) + 1) numbers. */
  transform(x: Samples): Float64Array;
}

/** A test statistic and its two-sided p-value. */
export interface TestResult {
  statistic: number;
  p_value: number;
}

export interface BoostMath {
  // Special functions. Most f(..., x) have a twin f_array(..., xs) that maps the last argument over an array.
  // Integer orders n, l, m that are not integers, or out of range, give NaN.
  tgamma(z: number): number;
  tgamma_array(zs: Samples): Float64Array;
  tgamma(a: number, x: number): number;
  tgamma_array(a: number, xs: Samples): Float64Array;
  lgamma(z: number): number;
  lgamma_array(zs: Samples): Float64Array;
  digamma(x: number): number;
  digamma_array(xs: Samples): Float64Array;
  trigamma(x: number): number;
  trigamma_array(xs: Samples): Float64Array;
  polygamma(n: number, x: number): number;
  polygamma_array(n: number, xs: Samples): Float64Array;
  tgamma1pm1(z: number): number;
  tgamma1pm1_array(zs: Samples): Float64Array;
  tgamma_lower(a: number, x: number): number;
  tgamma_lower_array(a: number, xs: Samples): Float64Array;
  tgamma_ratio(a: number, b: number): number;
  tgamma_ratio_array(a: number, bs: Samples): Float64Array;
  tgamma_delta_ratio(a: number, delta: number): number;
  tgamma_delta_ratio_array(a: number, deltas: Samples): Float64Array;
  gamma_p(a: number, x: number): number;
  gamma_p_array(a: number, xs: Samples): Float64Array;
  gamma_q(a: number, x: number): number;
  gamma_q_array(a: number, xs: Samples): Float64Array;
  gamma_p_inv(a: number, p: number): number;
  gamma_p_inv_array(a: number, ps: Samples): Float64Array;
  gamma_q_inv(a: number, q: number): number;
  gamma_q_inv_array(a: number, qs: Samples): Float64Array;
  gamma_p_derivative(a: number, x: number): number;
  gamma_p_derivative_array(a: number, xs: Samples): Float64Array;
  beta(a: number, b: number): number;
  beta_array(a: number, bs: Samples): Float64Array;
  beta(a: number, b: number, x: number): number;
  beta_array(a: number, b: number, xs: Samples): Float64Array;
  betac(a: number, b: number, x: number): number;
  betac_array(a: number, b: number, xs: Samples): Float64Array;
  ibeta(a: number, b: number, x: number): number;
  ibeta_array(a: number, b: number, xs: Samples): Float64Array;
  ibetac(a: number, b: number, x: number): number;
  ibetac_array(a: number, b: number, xs: Samples): Float64Array;
  ibeta_inv(a: number, b: number, p: number): number;
  ibeta_inv_array(a: number, b: number, ps: Samples): Float64Array;
  ibetac_inv(a: number, b: number, q: number): number;
  ibetac_inv_array(a: number, b: number, qs: Samples): Float64Array;
  ibeta_derivative(a: number, b: number, x: number): number;
  ibeta_derivative_array(a: number, b: number, xs: Samples): Float64Array;
  erf(x: number): number;
  erf_array(xs: Samples): Float64Array;
  erfc(x: number): number;
  erfc_array(xs: Samples): Float64Array;
  erf_inv(p: number): number;
  erf_inv_array(ps: Samples): Float64Array;
  erfc_inv(q: number): number;
  erfc_inv_array(qs: Samples): Float64Array;
  expint(x: number): number;
  expint_array(xs: Samples): Float64Array;
  expint(n: number, x: number): number;
  expint_array(n: number, xs: Samples): Float64Array;
  zeta(s: number): number;
  zeta_array(ss: Samples): Float64Array;
  cyl_bessel_j(v: number, x: number): number;
  cyl_bessel_j_array(v: number, xs: Samples): Float64Array;
  cyl_neumann(v: number, x: number): number;
  cyl_neumann_array(v: number, xs: Samples): Float64Array;
  cyl_bessel_i(v: number, x: number): number;
  cyl_bessel_i_array(v: number, xs: Samples): Float64Array;
  cyl_bessel_k(v: number, x: number): number;
  cyl_bessel_k_array(v: number, xs: Samples): Float64Array;
  sph_bessel(n: number, x: number): number;
  sph_bessel_array(n: number, xs: Samples): Float64Array;
  sph_neumann(n: number, x: number): number;
  sph_neumann_array(n: number, xs: Samples): Float64Array;
  airy_ai(x: number): number;
  airy_ai_array(xs: Samples): Float64Array;
  airy_bi(x: number): number;
  airy_bi_array(xs: Samples): Float64Array;
  airy_ai_prime(x: number): number;
  airy_ai_prime_array(xs: Samples): Float64Array;
  airy_bi_prime(x: number): number;
  airy_bi_prime_array(xs: Samples): Float64Array;
  ellint_1(k: number): number;
  ellint_1_array(ks: Samples): Float64Array;
  ellint_1(k: number, phi: number): number;
  ellint_1_array(k: number, phis: Samples): Float64Array;
  ellint_2(k: number): number;
  ellint_2_array(ks: Samples): Float64Array;
  ellint_2(k: number, phi: number): number;
  ellint_2_array(k: number, phis: Samples): Float64Array;
  ellint_3(k: number, n: number): number;
  ellint_3_array(k: number, ns: Samples): Float64Array;
  ellint_3(k: number, n: number, phi: number): number;
  ellint_3_array(k: number, n: number, phis: Samples): Float64Array;
  lambert_w0(z: number): number;
  lambert_w0_array(zs: Samples): Float64Array;
  lambert_wm1(z: number): number;
  lambert_wm1_array(zs: Samples): Float64Array;
  owens_t(h: number, a: number): number;
  owens_t_array(h: number, as: Samples): Float64Array;
  legendre_p(l: number, x: number): number;
  legendre_p_array(l: number, xs: Samples): Float64Array;
  laguerre(n: number, x: number): number;
  laguerre_array(n: number, xs: Samples): Float64Array;
  hermite(n: number, x: number): number;
  hermite_array(n: number, xs: Samples): Float64Array;
  chebyshev_t(n: number, x: number): number;
  chebyshev_t_array(n: number, xs: Samples): Float64Array;
  log1p(x: number): number;
  log1p_array(xs: Samples): Float64Array;
  expm1(x: number): number;
  expm1_array(xs: Samples): Float64Array;
  cbrt(x: number): number;
  cbrt_array(xs: Samples): Float64Array;
  sqrt1pm1(x: number): number;
  sqrt1pm1_array(xs: Samples): Float64Array;
  powm1(x: number, y: number): number;
  powm1_array(x: number, ys: Samples): Float64Array;
  sinc_pi(x: number): number;
  sinc_pi_array(xs: Samples): Float64Array;
  sin_pi(x: number): number;
  sin_pi_array(xs: Samples): Float64Array;
  cos_pi(x: number): number;
  cos_pi_array(xs: Samples): Float64Array;
  hypergeometric_0F1(b: number, z: number): number;
  hypergeometric_0F1_array(b: number, zs: Samples): Float64Array;
  hypergeometric_1F1(a: number, b: number, z: number): number;
  hypergeometric_1F1_array(a: number, b: number, zs: Samples): Float64Array;
  hypergeometric_2F0(a1: number, a2: number, z: number): number;
  hypergeometric_2F0_array(a1: number, a2: number, zs: Samples): Float64Array;
  jacobi_sn(k: number, u: number): number;
  jacobi_sn_array(k: number, us: Samples): Float64Array;
  jacobi_cn(k: number, u: number): number;
  jacobi_cn_array(k: number, us: Samples): Float64Array;
  jacobi_dn(k: number, u: number): number;
  jacobi_dn_array(k: number, us: Samples): Float64Array;
  jacobi_zeta(k: number, phi: number): number;
  jacobi_zeta_array(k: number, phis: Samples): Float64Array;
  heuman_lambda(k: number, phi: number): number;
  heuman_lambda_array(k: number, phis: Samples): Float64Array;
  ellint_d(k: number): number;
  ellint_d_array(ks: Samples): Float64Array;
  ellint_d(k: number, phi: number): number;
  ellint_d_array(k: number, phis: Samples): Float64Array;
  /** DLMF's spheroidal eigenvalue λ_n^m(c²) (prolate) or λ_n^m(-c²) (oblate); NaN for invalid orders. */
  prolate_spheroidal_eigenvalue(n: number, m: number, c: number): number;
  prolate_spheroidal_eigenvalue_array(n: number, m: number, cs: Samples): Float64Array;
  oblate_spheroidal_eigenvalue(n: number, m: number, c: number): number;
  oblate_spheroidal_eigenvalue_array(n: number, m: number, cs: Samples): Float64Array;
  prolate_spheroidal_angular(n: number, m: number, c: number, eta: number): number;
  prolate_spheroidal_angular_prime(n: number, m: number, c: number, eta: number): number;
  prolate_spheroidal_angular_array(n: number, m: number, c: number, etas: Samples): Float64Array;
  prolate_spheroidal_angular_prime_array(n: number, m: number, c: number, etas: Samples): Float64Array;
  oblate_spheroidal_angular(n: number, m: number, c: number, eta: number): number;
  oblate_spheroidal_angular_prime(n: number, m: number, c: number, eta: number): number;
  oblate_spheroidal_angular_array(n: number, m: number, c: number, etas: Samples): Float64Array;
  oblate_spheroidal_angular_prime_array(n: number, m: number, c: number, etas: Samples): Float64Array;
  prolate_spheroidal_radial_1(n: number, m: number, c: number, xi: number): number;
  prolate_spheroidal_radial_1_prime(n: number, m: number, c: number, xi: number): number;
  prolate_spheroidal_radial_1_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  prolate_spheroidal_radial_1_prime_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  prolate_spheroidal_radial_2(n: number, m: number, c: number, xi: number): number;
  prolate_spheroidal_radial_2_prime(n: number, m: number, c: number, xi: number): number;
  prolate_spheroidal_radial_2_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  prolate_spheroidal_radial_2_prime_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  oblate_spheroidal_radial_1(n: number, m: number, c: number, xi: number): number;
  oblate_spheroidal_radial_1_prime(n: number, m: number, c: number, xi: number): number;
  oblate_spheroidal_radial_1_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  oblate_spheroidal_radial_1_prime_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  oblate_spheroidal_radial_2(n: number, m: number, c: number, xi: number): number;
  oblate_spheroidal_radial_2_prime(n: number, m: number, c: number, xi: number): number;
  oblate_spheroidal_radial_2_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  oblate_spheroidal_radial_2_prime_array(n: number, m: number, c: number, xis: Samples): Float64Array;
  /** Struve H_v(x) and modified Struve L_v(x), with the derivatives in x; x < 0 only for integer v (NaN otherwise). */
  struve_h(v: number, x: number): number;
  struve_h_prime(v: number, x: number): number;
  struve_h_array(v: number, xs: Samples): Float64Array;
  struve_h_prime_array(v: number, xs: Samples): Float64Array;
  struve_l(v: number, x: number): number;
  struve_l_prime(v: number, x: number): number;
  struve_l_array(v: number, xs: Samples): Float64Array;
  struve_l_prime_array(v: number, xs: Samples): Float64Array;
  /** Fresnel integrals C(x) = ∫₀ˣ cos(πt²/2) dt and S(x) = ∫₀ˣ sin(πt²/2) dt (the normalized convention of DLMF 7.2, mpmath, SciPy). */
  fresnel_c(x: number): number;
  fresnel_c_array(xs: Samples): Float64Array;
  fresnel_s(x: number): number;
  fresnel_s_array(xs: Samples): Float64Array;
  /** C(x) + i S(x) for the cost of one of them: [C, S]. */
  fresnel_cis(x: number): Float64Array;
  /** Interleaved: [C(x0), S(x0), C(x1), S(x1), ...], 2 xs.length numbers. */
  fresnel_cis_array(xs: Samples): Float64Array;
  jacobi_theta1(x: number, q: number): number;
  jacobi_theta1_array(x: number, qs: Samples): Float64Array;
  jacobi_theta2(x: number, q: number): number;
  jacobi_theta2_array(x: number, qs: Samples): Float64Array;
  jacobi_theta3(x: number, q: number): number;
  jacobi_theta3_array(x: number, qs: Samples): Float64Array;
  jacobi_theta4(x: number, q: number): number;
  jacobi_theta4_array(x: number, qs: Samples): Float64Array;
  ellint_rf(x: number, y: number, z: number): number;
  ellint_rf_array(x: number, y: number, zs: Samples): Float64Array;
  ellint_rd(x: number, y: number, z: number): number;
  ellint_rd_array(x: number, y: number, zs: Samples): Float64Array;
  ellint_rc(x: number, y: number): number;
  ellint_rc_array(x: number, ys: Samples): Float64Array;
  ellint_rg(x: number, y: number, z: number): number;
  ellint_rg_array(x: number, y: number, zs: Samples): Float64Array;
  cyl_bessel_j_prime(v: number, x: number): number;
  cyl_bessel_j_prime_array(v: number, xs: Samples): Float64Array;
  cyl_neumann_prime(v: number, x: number): number;
  cyl_neumann_prime_array(v: number, xs: Samples): Float64Array;
  cyl_bessel_i_prime(v: number, x: number): number;
  cyl_bessel_i_prime_array(v: number, xs: Samples): Float64Array;
  cyl_bessel_k_prime(v: number, x: number): number;
  cyl_bessel_k_prime_array(v: number, xs: Samples): Float64Array;
  cyl_bessel_j_zero(v: number, m: number): number;
  cyl_neumann_zero(v: number, m: number): number;
  airy_ai_zero(m: number): number;
  airy_bi_zero(m: number): number;
  gegenbauer(n: number, lambda: number, x: number): number;
  spherical_harmonic_r(n: number, m: number, theta: number, phi: number): number;
  spherical_harmonic_i(n: number, m: number, theta: number, phi: number): number;
  // Distributions: new bm.normal_distribution(0, 1). The hypergeometric parameters are unsigned integers.
  normal_distribution: new (mean: number, sd: number) => Distribution;
  students_t_distribution: new (df: number) => Distribution;
  chi_squared_distribution: new (df: number) => Distribution;
  beta_distribution: new (alpha: number, beta: number) => Distribution;
  gamma_distribution: new (shape: number, scale: number) => Distribution;
  exponential_distribution: new (lambda: number) => Distribution;
  lognormal_distribution: new (location: number, scale: number) => Distribution;
  weibull_distribution: new (shape: number, scale: number) => Distribution;
  cauchy_distribution: new (location: number, scale: number) => Distribution;
  fisher_f_distribution: new (df1: number, df2: number) => Distribution;
  uniform_distribution: new (lower: number, upper: number) => Distribution;
  laplace_distribution: new (location: number, scale: number) => Distribution;
  logistic_distribution: new (location: number, scale: number) => Distribution;
  non_central_chi_squared_distribution: new (df: number, lambda: number) => Distribution;
  non_central_t_distribution: new (df: number, delta: number) => Distribution;
  binomial_distribution: new (n: number, p: number) => Distribution;
  poisson_distribution: new (mean: number) => Distribution;
  negative_binomial_distribution: new (r: number, p: number) => Distribution;
  geometric_distribution: new (p: number) => Distribution;
  arcsine_distribution: new (x_min: number, x_max: number) => Distribution;
  bernoulli_distribution: new (p: number) => Distribution;
  extreme_value_distribution: new (location: number, scale: number) => Distribution;
  generalized_extreme_value_distribution: new (location: number, scale: number, shape: number) => Distribution;
  holtsmark_distribution: new (location: number, scale: number) => Distribution;
  hypergeometric_distribution: new (r: number, n: number, N: number) => Distribution;
  inverse_chi_squared_distribution: new (df: number, scale: number) => Distribution;
  inverse_gamma_distribution: new (shape: number, scale: number) => Distribution;
  inverse_gaussian_distribution: new (mean: number, scale: number) => Distribution;
  kolmogorov_smirnov_distribution: new (n: number) => Distribution;
  landau_distribution: new (location: number, scale: number) => Distribution;
  mapairy_distribution: new (location: number, scale: number) => Distribution;
  non_central_beta_distribution: new (alpha: number, beta: number, lambda: number) => Distribution;
  non_central_f_distribution: new (df1: number, df2: number, lambda: number) => Distribution;
  pareto_distribution: new (scale: number, shape: number) => Distribution;
  rayleigh_distribution: new (sigma: number) => Distribution;
  saspoint5_distribution: new (location: number, scale: number) => Distribution;
  skew_normal_distribution: new (location: number, scale: number, shape: number) => Distribution;
  triangular_distribution: new (lower: number, mode: number, upper: number) => Distribution;
  von_mises_distribution: new (mean: number, concentration: number) => Distribution;

  /** A transform of length n, an integer in [1, 2^24] (else RangeError). Any n: radix 2, mixed radix or Bluestein. */
  fft: new (n: number) => FFT;
  real_fft: new (n: number) => RealFFT;

  // Transforms along one axis of an N-dimensional array, stored C-order (row major) as a flat array with the given
  // shape. The axis may be negative, counting from the end as in NumPy. Elements of complex arrays are (re, im)
  // pairs, so a complex array of shape s holds 2 product(s) numbers, and its last (fastest) index is the pair.
  // RangeError unless the shape holds positive integers, the axis is an integer in [-ndim, ndim), the length along
  // the axis is at most 2^24, and the data has product(shape) numbers (2 product(shape) if complex).
  /** Complex in, complex out, same shape. */
  fft_axis(z: Samples, shape: Samples, axis: number): Float64Array;
  /** Complex in, complex out, same shape; includes the 1/n. */
  inverse_fft_axis(z: Samples, shape: Samples, axis: number): Float64Array;
  /** Real in, complex out: shape[axis] = n becomes floor(n/2) + 1. */
  real_fft_axis(x: Samples, shape: Samples, axis: number): Float64Array;
  /**
   * The one-sided power spectrum of real data, P[k] for k = 0, ..., floor(n/2): P[k] = c |X[k]|^2 / n^2, where X is
   * the unnormalized DFT, c = 2 except c = 1 at k = 0 and at k = n/2 for even n. So sum_k P[k] = mean(x^2), and
   * A cos(2 pi k j / n) has P[k] = A^2/2 (0 < k < n/2). Any length n; no window, no detrending. Real out, with
   * shape[axis] = n becoming floor(n/2) + 1.
   */
  power_spectrum_axis(x: Samples, shape: Samples, axis: number): Float64Array;
  /** power_spectrum_axis(x, [x.length], 0). */
  power_spectrum(x: Samples): Float64Array;

  // Multivariate root finding. f(x) returns [F, J], F(x) an array of n and J(x) the Jacobian, J[i][j] = dF_i/dx_j, as n
  // arrays of n or n*n numbers row major; Halley also needs d2(v) = F''(x)[v, v], the second derivative of
  // t -> F(x + t v) at t = 0. digits (default 53) is the accuracy wanted in bits; max_iterations defaults to 1000.
  // The root is all NaN on failure (e.g. a local minimum of |F| that is not a root); iterations == max_iterations
  // means it did not converge. A wrong shape from f, or an exception f throws, propagates as a RangeError / that exception.
  multivariate_newton(f: (x: Float64Array) => [Samples, Samples | Samples[], ...unknown[]], x0: Samples, digits?: number,
                      max_iterations?: number): { root: Float64Array; iterations: number };
  multivariate_halley(f: (x: Float64Array) => [Samples, Samples | Samples[], (v: Float64Array) => Samples], x0: Samples,
                      digits?: number, max_iterations?: number): { root: Float64Array; iterations: number };

  // Trend filtering: argmin_x 1/2 sum (y - x)^2 + lambda sum |D^(k+1) x|, piecewise constant (k = 0), linear (1)
  // or quadratic (2). Bad k or lambda, or fewer than k + 2 samples, give all NaN.
  trend_filter(y: Samples, k: number, lambda: number): Float64Array;
  /** k and lambda chosen by even/odd cross-validation; needs at least 4 samples. */
  trend_filter(y: Samples): Float64Array;
  /** The wavelet and threshold rule chosen by even/odd cross-validation (length >= 14). */
  wavelet_denoise(x: Samples): Float64Array;
  /** Shrinks the coefficients of x in the wavelet family p (p = 2..10; length >= 4p - 1). sigma: the noise level, NaN (the default) to estimate it. */
  wavelet_denoise(x: Samples, family: WaveletFamily, p: number, rule?: ThresholdRule, sigma?: number): Float64Array;
  /** Coefficients of x, coarsest first: [a_L | d_L | ... | d_1]; levels < 0 (the default) takes as many as the length allows. */
  wavelet_transform(x: Samples, family: WaveletFamily, p: number, levels?: number): Float64Array;
  inverse_wavelet_transform(c: Samples, family: WaveletFamily, p: number, levels?: number): Float64Array;
  /** The better of wavelet_denoise(x) and trend_filter(x) by the same cross-validation score. */
  denoise(x: Samples): Float64Array;

  // Statistics of the values in an array. Too few values, or arrays of different lengths, throw a RangeError; the
  // count needed is given for each. Any NaN in the data makes the result NaN.
  /** n >= 1. */
  mean(x: Samples): number;
  /** The population variance, sum (x - mean)^2 / n; n >= 1. */
  variance(x: Samples): number;
  /** sum (x - mean)^2 / (n - 1); n >= 2. */
  sample_variance(x: Samples): number;
  /** n >= 2. */
  mean_and_sample_variance(x: Samples): { mean: number; sample_variance: number };
  /** The population skewness, 0 for constant data; n >= 1. */
  skewness(x: Samples): number;
  /** The population kurtosis (3 for a normal distribution), 0 for constant data; n >= 1. */
  kurtosis(x: Samples): number;
  /** kurtosis - 3; n >= 1. */
  excess_kurtosis(x: Samples): number;
  /** n >= 1; the input is not modified. */
  median(x: Samples): number;
  /** The median of |x - median(x)|, unscaled; n >= 1. */
  median_absolute_deviation(x: Samples): number;
  /** Q3 - Q1, the quartiles being medians of the lower and upper halves; n >= 3. */
  interquartile_range(x: Samples): number;
  /** For nonnegative data; 0 if all are zero. n >= 1. */
  gini_coefficient(x: Samples): number;
  /** n gini_coefficient / (n - 1); n >= 2. */
  sample_gini_coefficient(x: Samples): number;
  /** (sqrt(n) - ||x||_1 / ||x||_2) / (sqrt(n) - 1), in [0, 1]: 0 when all |x_i| are equal, 1 for one nonzero; n >= 2. */
  hoyer_sparsity(x: Samples): number;
  /** The population covariance, sum (x - mean x)(y - mean y) / n; same lengths, n >= 1. */
  covariance(x: Samples, y: Samples): number;
  /** Pearson's r; same lengths, n >= 1. NaN if x or y is constant. */
  correlation_coefficient(x: Samples, y: Samples): number;
  /** The least-squares line y = intercept + slope x and its R^2; same lengths, n >= 2, and x not constant. */
  simple_ordinary_least_squares(x: Samples, y: Samples): { intercept: number; slope: number; r_squared: number };
  /** Student's t test that the mean of x is mu; n >= 2. */
  one_sample_t_test(x: Samples, mu: number): TestResult;
  /** As one_sample_t_test, but with the normal distribution; n >= 2. */
  one_sample_z_test(x: Samples, mu: number): TestResult;
  /** That x and y have the same mean: Welch's test if one standard deviation is over twice the other; n >= 2 each. */
  two_sample_t_test(x: Samples, y: Samples): TestResult;
  /** With the normal distribution; n >= 2 each. */
  two_sample_z_test(x: Samples, y: Samples): TestResult;
  /** That the mean of x - y is 0; same lengths, n >= 2. */
  paired_samples_t_test(x: Samples, y: Samples): TestResult;
  /** The Wald–Wolfowitz runs test of randomness; values equal to the threshold are skipped. n >= 2. */
  runs_above_and_below_threshold(x: Samples, threshold: number): TestResult;
  /** runs_above_and_below_threshold(x, median(x)); n >= 2. */
  runs_above_and_below_median(x: Samples): TestResult;
  /**
   * The Anderson–Darling statistic A^2 for normality, with the mean and standard deviation estimated from x, which
   * needn't be sorted. There is no p-value. n >= 2.
   */
  anderson_darling_normality_statistic(x: Samples): number;
  /**
   * The Ljung–Box test for autocorrelation over the given number of lags, an integer in [1, n), or -1 for
   * ceil(log(n)); the chi-squared distribution has lags - fit_dof degrees of freedom. RangeError if lags or fit_dof
   * isn't such an integer; n >= 2.
   */
  ljung_box(x: Samples, lags: number, fit_dof: number): TestResult;
}

/** Loads and instantiates the module. Options are Emscripten's (locateFile, wasmBinary, ...). */
export default function createModule(options?: Record<string, unknown>): Promise<BoostMath>;
