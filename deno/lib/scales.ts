// Scales, ticks, number formats and domains of every axis and colour bar (no dependencies):
// linear, log, symlog, gamma and root (colours) scales; nice 1-2-5 ticks, log ticks, integer
// ticks, ticks at multiples of π and at every k-th decade of symlog scales; numbers plainly or with
// superscript powers of 10; the world transform of line plot axes (curves are drawn in axis space).

export type ScaleType = "linear" | "log" | "symlog" | "sqrt" | "gamma";
export type VisScale = ScaleType | ["symlog" | "gamma" | "root", number]; // symlog constant, gamma exponent, root n
export type Domain = [number, number];
export type ValueTransform = "exp10" | "exp2";

export const DEFAULT_DOMAIN: Domain = [0.1, 1];

/** n of the root colour scale: its range, grid and default (n = 1 linear, 2 square root, 0.5 squares). */
export const ROOT_N = { min: 0.1, max: 10, step: 0.05, default: 3 };

/** `n` on the grid of ROOT_N, within its range (the default if not a finite number). */
export function snapRootN(n: unknown): number {
  const v = typeof n === "number" && Number.isFinite(n) ? n : ROOT_N.default;
  const k = Math.round(Math.min(Math.max(v, ROOT_N.min), ROOT_N.max) / ROOT_N.step);
  return Number((k * ROOT_N.step).toFixed(2));
}

const VALID_MIN: Record<ScaleType, number> = { linear: -Infinity, log: Number.MIN_VALUE, symlog: -Infinity, sqrt: 0, gamma: -Infinity };

export interface Scale {
  (v: number): number;
  invert(y: number): number;
  ticks(count?: number): number[];
}

/** Scale of a type (symlog with its constant), or the gamma scale: the value normalised over the
 * domain, to the power `exponent` (sqrt: 0.5; root n: 1 / n). Ticks: nice ticks, log ticks on log
 * scales, `rootTicks` on root scales. */
export function createScale(type: VisScale, opts: { domain: Domain; range?: Domain; clamp?: boolean }): Scale {
  const [kind, param] = Array.isArray(type) ? type : [type, undefined];
  const [d0, d1] = opts.domain, [r0, r1] = opts.range ?? [0, 1], clamp = !!opts.clamp;
  const cl01 = (t: number) => (clamp ? Math.min(Math.max(t, 0), 1) : t);
  if (kind === "gamma" || kind === "sqrt" || kind === "root") {
    const e = kind === "sqrt" ? 0.5 : kind === "root" ? 1 / (param ?? 1) : param ?? 1;
    const norm = (x: number, a: number, b: number) => (b !== a ? (x - a) / (b - a) : 0.5);
    const s = ((v: number) => r0 + (r1 - r0) * Math.max(cl01(norm(v, d0, d1)), 0) ** e) as Scale;
    s.invert = (y) => d0 + (d1 - d0) * Math.max(cl01(norm(y, r0, r1)), 0) ** (1 / e);
    s.ticks = (n = 10) => (kind === "root" ? rootTicks(d0, d1, param ?? 1, n) : ticks(d0, d1, n));
    return s;
  }
  const c = param ?? 1;
  const [fwd, inv]: [(x: number) => number, (x: number) => number] =
    kind === "log" ? [Math.log10, (x) => 10 ** x]
    : kind === "symlog" ? [(x) => Math.sign(x) * Math.log1p(Math.abs(x) / c), (x) => Math.sign(x) * Math.expm1(Math.abs(x)) * c]
    : [(x) => x, (x) => x];
  const [t0, t1] = [fwd(d0), fwd(d1)];
  const s = ((v: number) => r0 + (r1 - r0) * cl01((fwd(v) - t0) / (t1 - t0))) as Scale;
  s.invert = (y) => inv(t0 + (t1 - t0) * cl01((y - r0) / (r1 - r0)));
  s.ticks = (n = 10) => (kind === "log" ? logTicks(d0, d1, n) : ticks(d0, d1, n));
  return s;
}

// --- Ticks -------------------------------------------------------------------------------------

/** About one tick per 85 CSS px of an axis `length` px long: 3 to 10. */
export function tickCount(length: number): number {
  const t = (Math.min(Math.max(length, 300), 900) - 300) / 600;
  return Math.round(3 * (1 - t) + 10 * t);
}

/**
 * Ticks i × step for about `count` steps over [lo, hi] (lo < hi): the step is 1, 2, 5 or 10 times a
 * power of 10, the one nearest (geometrically) the raw step. Below 1 the step is held as its
 * inverse `1 / step` (an integer), so that ticks are exact decimals (i / 20, not i × 0.05).
 * Returns [first i, last i, step] with the step negated when inverted.
 */
function tickIndices(lo: number, hi: number, count: number): [number, number, number] {
  const raw = (hi - lo) / Math.max(0, count);
  const power = Math.floor(Math.log10(raw));
  const mantissa = raw / Math.pow(10, power);
  const factor = mantissa >= Math.sqrt(50) ? 10 : mantissa >= Math.sqrt(10) ? 5 : mantissa >= Math.sqrt(2) ? 2 : 1;
  let first: number, last: number, step: number;
  if (power < 0) {
    const inverse = Math.pow(10, -power) / factor;
    first = Math.round(lo * inverse);
    last = Math.round(hi * inverse);
    if (first / inverse < lo) first += 1;
    if (last / inverse > hi) last -= 1;
    step = -inverse;
  } else {
    step = Math.pow(10, power) * factor;
    first = Math.round(lo / step);
    last = Math.round(hi / step);
    if (first * step < lo) first += 1;
    if (last * step > hi) last -= 1;
  }
  if (last < first && 0.5 <= count && count < 2) return tickIndices(lo, hi, count * 2);
  return [first, last, step];
}

/** Nice ticks from `start` to `stop` (either way round), about `count` of them. */
export function ticks(start: number, stop: number, count: number): number[] {
  if (!(count > 0)) return [];
  if (start === stop) return [start];
  const reverse = stop < start;
  const [first, last, step] = reverse ? tickIndices(stop, start, count) : tickIndices(start, stop, count);
  if (!(last >= first)) return [];
  const at = (i: number) => (step < 0 ? i / -step : i * step);
  return Array.from({ length: last - first + 1 }, (_, k) => at(reverse ? last - k : first + k));
}

/** Step of the nice ticks of `ticks` (negative if `stop < start`). */
export function tickStep(start: number, stop: number, count: number): number {
  const reverse = stop < start;
  const step = (reverse ? tickIndices(stop, start, count) : tickIndices(start, stop, count))[2];
  return (reverse ? -1 : 1) * (step < 0 ? 1 / -step : step);
}

const pow10 = (e: number) => (Number.isFinite(e) ? Number(`1e${e}`) : e < 0 ? 0 : e);

/** Ticks of a log scale: 1..9 × 10ⁿ if the domain spans fewer decades than `count` (nice ticks
 * if that gives fewer than `count / 2`), else nice powers of 10. Negative domains mirrored. */
export function logTicks(d0: number, d1: number, count: number): number[] {
  const negative = d0 < 0;
  const log = negative ? (x: number) => -Math.log10(-x) : Math.log10;
  const pow = negative ? (x: number) => -pow10(-x) : pow10;
  const reverse = d1 < d0;
  const [u, v] = reverse ? [d1, d0] : [d0, d1];
  let [i, j] = [log(u), log(v)];
  let out: number[] = [];
  if (j - i < count) {
    [i, j] = [Math.floor(i), Math.ceil(j)];
    for (; i <= j; i += 1) {
      for (let k = u > 0 ? 1 : 9; u > 0 ? k < 10 : k >= 1; k += u > 0 ? 1 : -1) {
        const t = (u > 0 ? i < 0 : i > 0) ? k / pow(-i) : k * pow(i);
        if (t < u) continue;
        if (t > v) break;
        out.push(t);
      }
    }
    if (out.length * 2 < count) out = ticks(u, v, count);
  } else {
    out = ticks(i, j, Math.min(j - i, count)).map(pow);
  }
  return reverse ? out.reverse() : out;
}

/** Ticks of a root scale (t ↦ t^(1/n) over [d0, d1]): the nice ticks for `count`, then finer ones
 * (steps down to 1/1024 of theirs) where the scale stretches their step to 0.7 nice steps, each
 * kept at least that far from the kept ones on the scale: fills the stretched end of the bar, thins
 * the squeezed one; at n = 1, the nice ticks. */
export function rootTicks(d0: number, d1: number, n: number, count: number): number[] {
  if (!(count > 0) || d0 === d1) return ticks(d0, d1, count);
  const at = (v: number) => Math.min(Math.max((v - d0) / (d1 - d0), 0), 1) ** (1 / n);
  const gap = 0.7 * Math.abs(tickStep(d0, d1, count) / (d1 - d0));
  const kept: number[] = [];
  for (let k = 1; k <= 1024; k *= 2) {
    const step = Math.abs(tickStep(d0, d1, count * k));
    for (const v of ticks(d0, d1, count * k)) {
      const stretched = k === 1 || Math.max(Math.abs(at(v) - at(v - step)), Math.abs(at(v + step) - at(v))) >= gap;
      if (stretched && kept.every((u) => Math.abs(at(u) - at(v)) >= gap)) kept.push(v);
    }
  }
  return kept.sort((a, b) => (d0 < d1 ? a - b : b - a));
}

/** Integer ticks over the domain: steps of nice ticks, at least 1. */
export function integerTicks(domain: Domain, count: number): number[] {
  const [min, max] = [Math.min(...domain), Math.max(...domain)];
  const [intMin, intMax] = [Math.ceil(min), Math.floor(max)];
  const optimal = Math.min(intMax - intMin + 1, count);
  if (optimal === 0) return [];
  const step = Math.max(tickStep(intMin, intMax, optimal), 1);
  const [start, stop] = [Math.ceil(min / step), Math.floor(max / step)];
  return Array.from({ length: stop - start + 1 }, (_, i) => (start + i) * step);
}

const PI_FRACTIONS = [1 / 12, 1 / 6, 1 / 4, 1 / 2]; // finest steps, in units of π

/**
 * Ticks at multiples of π for about `count` ticks, like nice ticks: the step is the closest
 * (geometrically) of π/12, π/6, π/4, π/2, π, 2π, 4π... Zoomed in beyond π/12, plain nice ticks.
 */
export function piTicks(domain: Domain, count: number): number[] {
  const min = Math.min(...domain);
  const max = Math.max(...domain);
  const rawStep = (max - min) / Math.PI / Math.max(count, 1);
  if (!(rawStep > 0) || !Number.isFinite(rawStep)) {
    return [];
  }
  if (rawStep < PI_FRACTIONS[0] / Math.SQRT2) {
    return ticks(min, max, count);
  }
  const pow2 = 2 ** Math.max(Math.round(Math.log2(rawStep)), 0);
  const step = [...PI_FRACTIONS, pow2].reduce((best, candidate) =>
    Math.abs(Math.log(candidate / rawStep)) < Math.abs(Math.log(best / rawStep)) ? candidate : best
  );
  const eps = 1e-9;
  const start = Math.ceil(min / Math.PI / step - eps);
  const stop = Math.floor(max / Math.PI / step + eps);
  return Array.from({ length: Math.max(stop + 1 - start, 0) }, (_, i) => (start + i) * step * Math.PI);
}

/**
 * Ticks of a symlog scale of given constant: 0 and every k-th power of 10 from the linear band
 * out to each end of the domain, counted from the decade nearest the end (plain nice ticks would
 * all crowd at the ends of the axis when the constant is small). About `count` ticks in all,
 * shared by the two signs in proportion to their decades (at least one each). Nice ticks within
 * the linear band.
 */
export function symlogTicks(domain: Domain, constant: number, count: number): number[] {
  const min = Math.min(...domain);
  const max = Math.max(...domain);
  const linearTicks = () => ticks(min, max, count);

  // Exponents of the powers of 10 in [lo, hi] (0 <= lo), outside the linear band
  function decades(lo: number, hi: number): number[] {
    const from = Math.ceil(Math.log10(Math.max(lo, constant)));
    const to = Math.floor(Math.log10(hi));
    return to >= from ? Array.from({ length: to + 1 - from }, (_, i) => from + i) : [];
  }

  const pos = max > 0 ? decades(Math.max(min, 0), max) : [];
  const neg = min < 0 ? decades(Math.max(-max, 0), -min) : [];
  const decadeCount = pos.length + neg.length;
  if (decadeCount < 2) {
    return linearTicks();
  }

  const hasZero = min <= 0 && max >= 0;
  const budget = Math.max(count - (hasZero ? 1 : 0), 1);

  // Not within a quarter of a nominal tick spacing of 0 (about 1.5 label heights: labels would
  // overlap), in the space of the scale
  const scaled = (val: number) => Math.log1p(Math.abs(val) / constant);
  const minGap = hasZero ? (scaled(min) + scaled(max)) / Math.max(count, 1) / 4 : 0;

  // Every k-th decade down from the one nearest the end of the domain
  function pick(exps: number[]): number[] {
    if (exps.length === 0) {
      return [];
    }
    const share = Math.max(Math.round((budget * exps.length) / decadeCount), 1);
    const stride = Math.ceil(exps.length / share);
    const top = exps.length - 1;
    return exps
      .filter((_, i) => (top - i) % stride === 0)
      .map((exp) => Number(`1e${exp}`))
      .filter((val) => scaled(val) >= minGap);
  }

  const out = [...pick(neg).map((val) => -val).reverse(), ...(hasZero ? [0] : []), ...pick(pos)];
  return out.length >= 2 ? out : linearTicks();
}

const exponent = (x: number) => Math.floor(Math.log10(x) + 1e-9); // as in `toExponential`

/** Significant digits for adjacent ticks to differ once formatted (at least 3: e.g. 0.9994,
 * 0.9995, 0.9996 rather than 0.999, 1, 1). */
export function tickPrecision(values: number[]): number {
  let precision = 3;
  for (let i = 1; i < values.length; i += 1) {
    const [prev, val] = [values[i - 1], values[i]];
    const step = Number(Math.abs(val - prev).toPrecision(12)); // e.g. 0.9997 - 0.9996
    if (step > 0 && Number.isFinite(step)) {
      const max = Math.max(Math.abs(prev), Math.abs(val));
      precision = Math.max(precision, exponent(max) - exponent(step) + 1);
    }
  }
  return Math.min(precision, 21); // max of `toPrecision`
}

// --- Numbers -----------------------------------------------------------------------------------

const SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** Power with a superscript exponent (`10⁻⁴`, `2¹⁰`). */
export const formatPower = (base: number, exp: number) =>
  `${base}${exp < 0 ? "⁻" : ""}${String(Math.abs(exp)).replace(/\d/g, (d) => SUPERSCRIPTS[Number(d)])}`;

/** Whether a number shows plainly (cf. `formatNumber`), rather than with a power of 10. */
export function isPlainNumber(v: number, precision = 4): boolean {
  if (!Number.isFinite(v) || v === 0) return true;
  const exp = Number(v.toExponential(precision - 1).split("e")[1]); // rounded: 9.9999e-4 => 1.000e-3
  return exp >= -3 && exp < Math.max(5, precision);
}

/** `precision` significant digits: plain from 10⁻³ to 10⁵, else mantissa × power of 10 (always
 * with `powers`, so that ticks read alike); true minus signs, never e-notation. */
export function formatNumber(v: number, precision = 4, powers = false): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "NaN" : v > 0 ? "∞" : "−∞";
  if (v === 0) return "0";
  const sign = v < 0 ? "−" : "";
  const a = Math.abs(v);
  if (!powers && isPlainNumber(v, precision)) return `${sign}${Number(a.toPrecision(precision))}`;
  const [m, e] = a.toExponential(precision - 1).split("e").map(Number);
  return m === 1 ? `${sign}${formatPower(10, e)}` : `${sign}${m}×${formatPower(10, e)}`;
}

export const formatBound = (v: number) => formatNumber(v, 4);
export const formatTick = (v: number, precision = 3, powers = false) => formatNumber(v, precision, powers);
export const getRealValue = (v: number, t: ValueTransform) => (t === "exp2" ? 2 : 10) ** v;

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

/** Tick label of a multiple of π/12 (`π/4`, `3π/2`, `−π`...); other values as by `formatTick`. */
export function formatPiTick(val: number): string {
  const units = (val / Math.PI) * 12;
  const twelfths = Math.round(units);
  if ((twelfths === 0 && units !== 0) || Math.abs(units - twelfths) > 1e-9 * Math.abs(units)) {
    return formatTick(val); // e.g. plain ticks zoomed in near 0
  }
  if (twelfths === 0) return "0";
  const divisor = gcd(Math.abs(twelfths), 12);
  const num = Math.abs(twelfths) / divisor;
  const den = 12 / divisor;
  return `${twelfths < 0 ? "−" : ""}${num === 1 ? "" : num}π${den === 1 ? "" : `/${den}`}`;
}

/** Tick label of a logarithm: a power at integers (`10⁻³`), else the real value. */
export function formatPowerTick(exp: number, transform: ValueTransform, precision?: number): string {
  return Number.isInteger(exp) ? formatPower(transform === "exp2" ? 2 : 10, exp) : formatTick(getRealValue(exp, transform), precision);
}

// --- Domains -----------------------------------------------------------------------------------

/** Extent of the finite values the scale can show (log: positive ones; sqrt: non-negative). */
export function getDomain(values: ArrayLike<number>, type: ScaleType, ignore?: (v: number) => boolean): Domain | undefined {
  const lo = VALID_MIN[type];
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v) || v < lo || ignore?.(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return Number.isFinite(min) ? [min, max] : undefined;
}

export function getVisDomain(custom: [number | null, number | null], data: Domain): Domain {
  return [custom[0] ?? data[0], custom[1] ?? data[1]];
}

/** A domain the scale can show: ends it can't take come from the data, inverted ends give the
 * data's domain, a constant one is widened (a decade on log scales, 5% otherwise). */
export function getSafeDomain(domain: Domain, fallback: Domain, type: ScaleType): Domain {
  if (domain[0] > domain[1]) return fallback;
  const lo = VALID_MIN[type];
  let [min, max] = [domain[0] >= lo ? domain[0] : fallback[0], domain[1] >= lo ? domain[1] : fallback[1]];
  if (min > max) min = max;
  if (min === max && Number.isFinite(min)) {
    if (type === "log") return [min / Math.sqrt(10), min * Math.sqrt(10)];
    const d = min === 0 ? 1 : Math.abs(min) * 0.05;
    return [Math.max(min - d, lo), max + d];
  }
  return [min, max];
}

/** Scale type for the shader, gamma exponent and symlog constant. */
export function getShaderScale(vis: VisScale): [ScaleType, number, number] {
  if (!Array.isArray(vis)) return [vis, 1, 1];
  return vis[0] === "symlog" ? ["symlog", 1, vis[1]] : ["gamma", vis[0] === "root" ? 1 / vis[1] : vis[1], 1];
}

/** The domain's ends in the space the shader maps linearly (log10, symlog in decades, sqrt). */
export function scaleDomain(domain: Domain, type: ScaleType, constant = 1): Domain {
  const f = (v: number) =>
    type === "log" ? Math.log10(v)
    : type === "symlog" ? Math.sign(v) * Math.log1p(Math.abs(v) / constant) * Math.LOG10E
    : type === "sqrt" ? Math.sqrt(v)
    : v;
  return [f(domain[0]), f(domain[1])];
}

// --- Colour bars -------------------------------------------------------------------------------

/** Labelled subset of log ticks: k × 10^e for k up to a limit that keeps about `count` labels. */
function logLabelled(domain: Domain, count: number): (v: number) => boolean {
  const k = Math.max(1, (10 * count) / logTicks(domain[0], domain[1], 10).length);
  return (d) => {
    let m = d / 10 ** Math.round(Math.log10(d));
    if (m * 10 < 9.5) m *= 10;
    return m <= k;
  };
}

/** Ticks and labels of a colour bar `length` CSS px long over `domain`. Logarithms (a value
 * transform) are labelled as powers where they span two decades; log scales label a
 * subset of their ticks (k × 10ⁿ up to a k that keeps about `count` labels). */
export function colorBarTicks(vis: VisScale, domain: Domain, length: number, transform?: ValueTransform): { value: number; label: string }[] {
  const n = tickCount(length);
  const kind = Array.isArray(vis) ? vis[0] : vis;
  if (transform && kind === "linear") {
    const powers = Math.abs(domain[1] - domain[0]) >= 2;
    const lo = Math.ceil(Math.min(...domain)), hi = Math.floor(Math.max(...domain));
    const values = powers
      ? ticks(lo, hi, Math.min(n, hi - lo + 1)).filter(Number.isInteger)
      : ticks(domain[0], domain[1], n);
    return values.map((v) => ({
      value: v,
      label: powers ? formatPower(transform === "exp2" ? 2 : 10, v) : formatTick(getRealValue(v, transform)),
    }));
  }
  const values = Array.isArray(vis) && vis[0] === "symlog" ? symlogTicks(domain, vis[1], n) : createScale(vis, { domain }).ticks(n);
  if (kind === "log") {
    const labelled = logLabelled(domain, n);
    const powers = values.filter(labelled).some((v) => !isPlainNumber(v, 3));
    return values.map((v) => ({ value: v, label: labelled(v) ? formatTick(v, 3, powers) : "" }));
  }
  const precision = tickPrecision(values);
  const powers = values.some((v) => !isPlainNumber(v, precision));
  return values.map((v) => ({ value: v, label: formatTick(v, precision, powers) }));
}

// --- Line plot axes ----------------------------------------------------------------------------

export type AxisScale = "linear" | "log" | "symlog";
export type AxisTicks = "pi" | ValueTransform | undefined;

export interface AxisSpec {
  scale?: AxisScale;
  threshold?: number; // symlog constant
  ticks?: AxisTicks; // π multiples, or the values are logarithms (value transform)
  index?: boolean; // integer ticks (an index axis)
}

export interface Tick {
  value: number;
  world: number;
  label: string; // '' for unlabelled (log minor) ticks
}

/** World (axis-space) coordinate of a value, and back. Non-finite where it can't be shown. */
export function axisTransform(spec: AxisSpec): { fwd: (v: number) => number; inv: (w: number) => number } {
  const c = spec.threshold ?? 1;
  switch (spec.scale ?? "linear") {
    case "log":
      return { fwd: (v) => (v > 0 ? Math.log10(v) : v === 0 ? -Infinity : NaN), inv: (w) => 10 ** w };
    case "symlog":
      return { fwd: (v) => Math.sign(v) * Math.log1p(Math.abs(v) / c), inv: (w) => Math.sign(w) * Math.expm1(Math.abs(w)) * c };
    default:
      return { fwd: (v) => v, inv: (w) => w };
  }
}

/** Formatter of the default ticks of an axis: on log axes too long for every tick to be labelled,
 * only powers of 10 (all with powers if one needs it). */
function axisFormatter(domain: Domain, length: number, scale: AxisScale): (v: number, precision?: number, powers?: boolean) => string {
  if (scale !== "log") return formatTick;
  const threshold = 0.8 * (1 - (length - 300) / 200) + 1.4 * ((length - 300) / 200); // decades
  const [min, max] = domain.map(Math.abs).sort((a, b) => a - b);
  if (max / min < 10 ** threshold || Math.ceil(Math.log10(min)) > Math.log10(max)) {
    return formatTick; // all fit, or no power of 10 to keep (e.g. short axis)
  }
  const decades = [Math.ceil(Math.log10(min)), Math.floor(Math.log10(max))];
  const powers = decades.some((exp) => !isPlainNumber(10 ** exp));
  return (v, precision) => {
    const logged = Math.log10(Math.abs(v));
    return logged === Math.floor(logged) ? formatTick(v, precision, powers) : "";
  };
}

/**
 * Ticks of an axis of `cssLength` CSS pixels over the values [lo, hi]: at multiples of π or at
 * powers (logarithms, on linear axes), integer ticks (index axes), every k-th decade (symlog with
 * a threshold), else the scale's ticks; labels with the digits that tell adjacent ticks apart,
 * all with powers of 10 if one needs it.
 */
export function axisTicks(spec: AxisSpec, lo: number, hi: number, cssLength: number): Tick[] {
  const scale = spec.scale ?? "linear";
  const domain: Domain = [lo, hi];
  const count = tickCount(cssLength);
  const transform = spec.ticks === "pi" ? undefined : spec.ticks;
  let values: number[];
  let format: (v: number, precision?: number, powers?: boolean) => string;
  if (spec.ticks && scale === "linear") {
    if (spec.ticks === "pi") {
      values = piTicks(domain, count);
      format = formatPiTick;
    } else {
      const powers = Math.abs(hi - lo) >= 2;
      values = powers ? integerTicks(domain, count) : ticks(lo, hi, count);
      const t = spec.ticks;
      format = powers ? (v, p) => formatPowerTick(v, t, p) : (v, p) => formatTick(getRealValue(v, t), p); // `1`, not `10⁰`
    }
  } else {
    values = spec.index
      ? integerTicks(domain, count)
      : scale === "symlog" && spec.threshold
      ? symlogTicks(domain, spec.threshold, count)
      : scale === "log"
      ? logTicks(lo, hi, count)
      : ticks(lo, hi, count);
    const formatter = axisFormatter(domain, cssLength, scale);
    format = transform ? (v, p) => formatter(v, p) && formatPowerTick(v, transform) : formatter;
  }
  const precision = tickPrecision(values);
  const powers = values.some((v) => !isPlainNumber(v, precision));
  const { fwd } = axisTransform(spec);
  return values
    .filter((v) => v >= Math.min(lo, hi) - 1e-12 * Math.abs(hi - lo) && v <= Math.max(lo, hi) + 1e-12 * Math.abs(hi - lo))
    .map((value) => ({ value, world: fwd(value), label: format(value, precision, powers) }));
}
