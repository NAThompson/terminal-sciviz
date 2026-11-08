/**
 * Marks for line.ts: a small grammar of graphics in the spirit of Observable Plot. A mark draws
 * columns of data (inline columns or rows, a CSV/TSV/JSON file, or an `fn`) through channels
 * (x, y, x1, x2, y1, y2, z, fill/stroke, r, opacity, text) after optional transforms (bin, group,
 * window, density, then stack; or a stat: ecdf, qq, linreg, boxplot). `compileMarks` turns them
 * into the renderer's own series (`Compiled`: a SeriesSpec plus per-point primitives), so marks
 * and series share one drawing, hover, legend, extent and y-range path.
 * Scales: string x / y become categories (x left to right, y top to bottom, in order of first
 * appearance); fill strings map to the palette in a fixed order (never cycled), fill numbers to a
 * colormap; r is a sqrt scale. z (default: a categorical fill) splits a mark into series.
 * `slice` (a numeric column) splits a mark's rows by its value before the transforms: each value's
 * series are tagged with it (`Compiled.slice`), and the renderer draws those of the slider's value.
 */

import type { SeriesSpec } from "../line.ts";
import type { FnSpec } from "./generators.ts";
import type { Anchor } from "./overlay.ts";
import { type AxisSpec, formatNumber } from "./scales.ts";
import { colormapName, colormapRgb } from "./colormaps.ts";
import { closestSpacing, finiteRange, histogram, kde, rolling, sortedFinite } from "./stats.ts";

type Value = number | string | null;
export type Columns = Record<string, Value[]>;
type ChannelValue = string | number;
type Opt<T> = T | true;

export interface Mark {
  type: "line" | "dot" | "bar" | "barX" | "area" | "rect" | "rule" | "text" | "interval" | "box";
  data?: Columns | Record<string, Value>[];
  file?: string; // CSV/TSV with a header row, or JSON columns / rows
  fn?: FnSpec; // line, area: a function of x, sampled over the window (f64)
  x?: ChannelValue;
  y?: ChannelValue;
  x1?: ChannelValue;
  x2?: ChannelValue;
  y1?: ChannelValue;
  y2?: ChannelValue;
  z?: ChannelValue;
  fill?: ChannelValue;
  stroke?: ChannelValue; // the same colour channel as fill
  r?: ChannelValue;
  opacity?: ChannelValue;
  text?: ChannelValue;
  transform?: Transform;
  label?: string;
  legend?: boolean;
  extent?: boolean;
  width?: number;
  dash?: boolean;
  interpolation?: SeriesSpec["interpolation"];
  cap?: number; // interval: CSS px
  inset?: number; // bar, rect: CSS px off each side
  anchor?: Anchor; // text
  dx?: number; // text: CSS px
  dy?: number;
  slice?: string; // a numeric column: only the rows at the slider of that name's value are drawn
}

type BinOpt = { thresholds?: number; width?: number; reduce?: "count" | "density" | "proportion" };
type GroupOpt = { reduce?: "count" | "sum" | "mean"; sort?: "asc" | "desc" };
type WindowOpt = { k: number; reduce?: "mean" | "median" | "min" | "max" | "sum" };
type DensityOpt = { bandwidth?: number; width?: number };
type StackOpt = { offset?: "none" | "expand" | "center" };

export interface Transform {
  bin?: Opt<BinOpt>;
  group?: Opt<GroupOpt>;
  window?: WindowOpt;
  density?: Opt<DensityOpt>;
  stack?: Opt<StackOpt>;
  ecdf?: Opt<{ band?: number }>;
  qq?: Opt<{ reference?: boolean }>;
  linreg?: Opt<{ band?: number; kind?: "mean" | "prediction" }>;
  boxplot?: Opt<{ width?: number }>;
}

export interface Scales {
  color?: { scheme?: string; domain?: Value[]; label?: string; invert?: boolean };
  r?: { range?: [number, number]; domain?: [number, number] };
}

/** Per-point primitives the renderer draws beyond SeriesSpec's (values, not axis space). */
export interface Prims {
  noLine?: boolean;
  base?: boolean; // stands on y = 0: no margin below it (bars, areas from 0)
  colors?: string[];
  alphas?: number[];
  sizes?: number[]; // dot radii, CSS px
  rects?: { x1: number[]; x2: number[]; y1: number[]; y2: number[]; inset: [number, number]; radius: number; end: "x" | "y" | null };
  texts?: { text: string[]; anchor: Anchor; dx: number; dy: number };
  rules?: { axis: "x" | "y"; at: number[]; lo: number[]; hi: number[] }; // NaN ends: the plot's edges
  fillY?: { y: number[]; lo: number[]; hi: number[]; opacity: number }; // x between lo and hi as functions of y
  tips?: [string, string][][]; // hover rows of each point
}
export type Compiled = SeriesSpec & Prims & { stats?: Record<string, number>; slice?: { name: string; value: number } };

export interface ColorBar { label: string; scheme: string; invert: boolean; domain: [number, number] }

type Ch = "x" | "y" | "x1" | "x2" | "y1" | "y2" | "z" | "fill" | "r" | "opacity" | "text" | "xo1" | "xo2" | "v";
const CHANNELS: Ch[] = ["x", "y", "x1", "x2", "y1", "y2", "z", "fill", "r", "opacity", "text"];
const STATS = ["ecdf", "qq", "linreg", "boxplot"] as const;

/** A mark's rows as channels: columns (`ch`), where each came from (`names`, for hover), constants. */
interface Stage {
  m: Mark;
  where: string;
  n: number;
  ch: Partial<Record<Ch, Value[]>>;
  names: Partial<Record<Ch, string>>;
  color?: string; // a constant fill / stroke
  text?: string; // a constant text
  pending: Partial<Record<Ch, string>>; // names of a transform's outputs (e.g. fill: "y")
  stacked?: boolean;
  plain?: boolean; // z splits series only: too many groups for the palette, one colour
  slice?: number; // the value of the mark's slice column its rows have
}

const num = (v: Value | undefined): number => (v === null || v === undefined || v === "" ? NaN : Number(v));
const isCat = (col?: Value[]) => Boolean(col?.some((v) => typeof v === "string"));
const opt = <T extends object>(o: Opt<T> | undefined): T | undefined => (o === undefined ? undefined : o === true ? ({} as T) : o);
const f4 = (v: Value) => (typeof v === "number" ? formatNumber(v, 4) : String(v ?? ""));

/** A mark's data as columns (rows of objects transposed), all of one length. */
function columns(m: Mark, where: string, table: (path: string) => Columns): Columns | undefined {
  const d = m.file ? table(m.file) : m.data;
  if (d === undefined) return undefined;
  const cols: Columns = Object.create(null); // names are data: "constructor" is a column, not Object's
  if (Array.isArray(d)) {
    const keys = [...new Set(d.flatMap((row) => Object.keys(row ?? {})))];
    for (const k of keys) cols[k] = d.map((row) => row?.[k] ?? null);
  } else {
    for (const [k, v] of Object.entries(d)) {
      if (!Array.isArray(v)) throw new Error(`${where}: data.${k} is not an array (give columns {name: [...]} or rows [{...}])`);
      cols[k] = v;
    }
  }
  const lens = new Set(Object.values(cols).map((c) => c.length));
  if (lens.size > 1) throw new Error(`${where}: data columns of different lengths (${[...lens].join(", ")})`);
  return cols;
}

/** Channels of a mark: a column by name, a number constant (positions expanded to columns), a
 * CSS colour (fill/stroke), a text constant, or a name a transform will output. */
function resolve(m: Mark, where: string, cols: Columns | undefined): Stage {
  let n = cols ? Object.values(cols)[0]?.length ?? 0 : 0;
  const st: Stage = { m, where, n, ch: {}, names: {}, pending: {} };
  const consts: Partial<Record<Ch, number>> = {};
  for (const k of CHANNELS) {
    const v = k === "fill" ? m.fill ?? m.stroke : m[k as keyof Mark] as ChannelValue | undefined;
    if (v === undefined) continue;
    if (typeof v === "string" && cols && Object.hasOwn(cols, v)) [st.ch[k], st.names[k]] = [cols[v], v];
    else if (typeof v === "number") consts[k] = v;
    else if (m.transform && (CHANNELS as string[]).includes(v)) st.pending[k] = v; // a transform's output, e.g. text: "y"
    else if (k === "fill") st.color = v;
    else if (k === "text") st.text = v;
    else st.pending[k] = v;
  }
  if (!cols && Object.keys(consts).length) n = st.n = 1; // a rule at a constant
  for (const [k, v] of Object.entries(consts) as [Ch, number][]) st.ch[k] = new Array(n).fill(v);
  if (!st.ch.z && isCat(st.ch.fill)) [st.ch.z, st.names.z] = [st.ch.fill, st.names.fill]; // categorical colour groups
  return st;
}

/** Whether fill is the z column (by source name: `take` and the transforms copy arrays). */
const fillIsZ = (st: Stage) => st.names.fill !== undefined && st.names.fill === st.names.z;

/** The rows `idx` of every channel. */
function take(st: Stage, idx: number[]): void {
  for (const k of Object.keys(st.ch) as Ch[]) st.ch[k] = idx.map((i) => st.ch[k]![i]);
  st.n = idx.length;
}

/** A mark's stage split by its slice column, one stage per value (ascending); the values join `slices`. */
function sliceStages(st: Stage, cols: Columns | undefined, slices: Record<string, number[]>): Stage[] {
  const name = st.m.slice!;
  const col = typeof name === "string" && cols && Object.hasOwn(cols, name) ? cols[name] : undefined;
  if (!col) throw new Error(`${st.where}: slice: "${name}" is not a column${cols ? ` (columns: ${Object.keys(cols).join(", ")})` : " (fn marks have none)"}`);
  if (isCat(col)) throw new Error(`${st.where}: slice column "${name}" must be numeric`);
  const by = new Map<number, number[]>();
  col.forEach((v, i) => {
    const x = num(v);
    if (Number.isFinite(x)) (by.get(x) ?? by.set(x, []).get(x)!).push(i);
  });
  const values = [...by.keys()].sort((a, b) => a - b);
  slices[name] = [...new Set([...(slices[name] ?? []), ...values])].sort((a, b) => a - b);
  return values.map((v) => {
    const part: Stage = { ...st, ch: { ...st.ch }, names: { ...st.names }, pending: { ...st.pending }, slice: v };
    take(part, by.get(v)!);
    return part;
  });
}

/** Row indices per z value, in order of first appearance (one group without z). */
function groupsOf(st: Stage): [Value | undefined, number[]][] {
  const z = st.ch.z;
  if (!z) return [[undefined, Array.from({ length: st.n }, (_, i) => i)]];
  const out = new Map<Value, number[]>();
  z.forEach((v, i) => (out.get(v) ?? out.set(v, []).get(v)!).push(i));
  return [...out];
}

const need = (st: Stage, k: Ch, what: string) => {
  const c = st.ch[k];
  if (!c) throw new Error(`${st.where}: ${what} needs ${k} (a column name${st.pending[k] ? `; "${st.pending[k]}" is not a column` : ""})`);
  return c;
};

/** Replaces the rows with new ones (aggregating transforms: other channels don't carry over). */
function rebuild(st: Stage, ch: Stage["ch"], names: Stage["names"]): void {
  const keep = { z: st.names.z, fill: fillIsZ(st) ? st.names.fill : undefined };
  st.ch = ch;
  st.names = { ...names, z: keep.z };
  if (keep.fill && ch.z) [st.ch.fill, st.names.fill] = [ch.z, keep.fill];
  st.n = Object.values(ch)[0]?.length ?? 0;
}

// --- Transforms ----------------------------------------------------------------------------------

/** The band (category) axis and the value axis: bar x, barX y; others y if only y is categorical. */
function bandAxes(st: Stage): ["x", "y"] | ["y", "x"] {
  const { type } = st.m;
  const yBand = type === "barX" || (type !== "bar" && isCat(st.ch.y) && !isCat(st.ch.x));
  return yBand ? ["y", "x"] : ["x", "y"];
}

/** bin: x into equal bins (shared by the groups) → x (centre), x1, x2, y (count, density or proportion). */
function bin(st: Stage, o: BinOpt): void {
  const xs = need(st, "x", "bin").map(num);
  const s = sortedFinite(xs);
  if (!s.length) throw new Error(`${st.where}: bin: no finite x`);
  const h = histogram(s, o.thresholds, o.width);
  const [w, m, start] = [h.width, h.x.length, h.x[0] - h.width / 2];
  const reduce = o.reduce ?? "count";
  const ch: Stage["ch"] = { x: [], x1: [], x2: [], y: [], z: [] };
  for (const [key, idx] of groupsOf(st)) {
    const counts = new Array(m).fill(0);
    let total = 0;
    for (const i of idx) {
      if (!Number.isFinite(xs[i])) continue;
      counts[Math.min(Math.max(Math.floor((xs[i] - start) / w), 0), m - 1)] += 1;
      total += 1;
    }
    counts.forEach((c, b) => {
      ch.x!.push(h.x[b]);
      ch.x1!.push(start + b * w);
      ch.x2!.push(start + (b + 1) * w);
      ch.y!.push(reduce === "density" ? c / (total * w) : reduce === "proportion" ? c / total : c);
      ch.z!.push(key ?? null);
    });
  }
  if (!st.ch.z) delete ch.z;
  rebuild(st, ch, { x: st.names.x, y: reduce });
}

/** group: rows by the band (x; barX: y) and z → the value channel reduced (count, sum, mean),
 * bands optionally sorted by their total. */
function group(st: Stage, o: GroupOpt): void {
  const [b, v] = bandAxes(st);
  const band = need(st, b, "group");
  const reduce = o.reduce ?? "count";
  const vals = reduce === "count" ? undefined : need(st, v, `group (${reduce})`).map(num);
  const cells = new Map<string, { band: Value; z: Value; vs: number[] }>();
  band.forEach((bv, i) => {
    const z = st.ch.z?.[i] ?? null;
    const key = JSON.stringify([bv, z]);
    (cells.get(key) ?? cells.set(key, { band: bv, z, vs: [] }).get(key)!).vs.push(vals ? vals[i] : 1);
  });
  const out = [...cells.values()].map((c) => {
    const f = c.vs.filter(Number.isFinite);
    const sum = f.reduce((a, u) => a + u, 0);
    return { ...c, value: reduce === "count" ? c.vs.length : reduce === "sum" ? sum : f.length ? sum / f.length : NaN };
  });
  if (o.sort) {
    const totals = new Map<Value, number>();
    for (const c of out) totals.set(c.band, (totals.get(c.band) ?? 0) + c.value);
    const sign = o.sort === "desc" ? -1 : 1;
    out.sort((p, q) => sign * (totals.get(p.band)! - totals.get(q.band)!));
  }
  const ch: Stage["ch"] = { [b]: out.map((c) => c.band), [v]: out.map((c) => c.value) };
  if (st.ch.z) ch.z = out.map((c) => c.z);
  rebuild(st, ch, { [b]: st.names[b], [v]: reduce === "count" ? "count" : `${reduce} of ${st.names[v] ?? v}` });
}

/** window: per group, rows sorted by x and y replaced by its rolling reduction over k points. */
function windowed(st: Stage, o: WindowOpt): void {
  const xs = need(st, "x", "window").map(num);
  need(st, "y", "window");
  const order = groupsOf(st).flatMap(([, idx]) => idx.sort((a, b) => xs[a] - xs[b]));
  const starts = groupsOf(st).map(([, idx]) => idx.length);
  take(st, order);
  const ys = st.ch.y!.map(num);
  const out: number[] = [];
  let at = 0;
  for (const len of starts) {
    out.push(...rolling(ys.slice(at, at + len), o.k, o.reduce));
    at += len;
  }
  st.ch.y = out;
  st.names.y = `${o.reduce ?? "mean"} of ${o.k}${st.names.y ? ` (${st.names.y})` : ""}`;
}

/** density: a Gaussian KDE per group of x → x, y; with categorical x and numeric y (violins),
 * of y per category → y and half-widths xo1, xo2 around it (`width` of a slot at the densest). */
function density(st: Stage, o: DensityOpt): void {
  const violin = isCat(st.ch.x) && st.ch.y;
  const vals = (violin ? st.ch.y! : need(st, "x", "density")).map(num);
  const cells = new Map<string, { x: Value; z: Value; idx: number[] }>();
  for (let i = 0; i < st.n; i += 1) {
    const [x, z] = [violin ? st.ch.x![i] : null, st.ch.z?.[i] ?? null];
    const key = JSON.stringify([x, z]);
    (cells.get(key) ?? cells.set(key, { x, z, idx: [] }).get(key)!).idx.push(i);
  }
  const ks = [...cells.values()].map((c) => {
    const s = sortedFinite(c.idx.map((i) => vals[i]));
    if (!s.length) throw new Error(`${st.where}: density: a group without finite values`);
    return { ...c, k: kde(s, o.bandwidth) };
  });
  const ch: Stage["ch"] = { x: [], y: [], z: [] };
  if (violin) {
    const max = finiteRange(ks.flatMap((c) => c.k.y))[1];
    const half = (o.width ?? 0.9) / 2;
    [ch.xo1, ch.xo2] = [[], []];
    for (const c of ks) {
      c.k.x.forEach((t, j) => {
        ch.x!.push(c.x);
        ch.y!.push(t);
        ch.xo1!.push(-half * c.k.y[j] / max);
        ch.xo2!.push(half * c.k.y[j] / max);
        ch.z!.push(c.z);
      });
    }
  } else {
    for (const c of ks) c.k.x.forEach((t, j) => (ch.x!.push(t), ch.y!.push(c.k.y[j]), ch.z!.push(c.z)));
  }
  if (!st.ch.z) delete ch.z;
  rebuild(st, ch, violin ? { x: st.names.x, y: st.names.y } : { x: st.names.x, y: "density" });
}

/** stack: the value channel (y; barX: x) stacked across z at each band position, in z order →
 * y1, y2 (and y = y2 for lines and dots); negatives stack down; offset expand (to 0..1) or center. */
function stack(st: Stage, o: StackOpt): void {
  const [b, v] = bandAxes(st);
  const band = need(st, b, "stack"), vals = need(st, v, "stack").map(num);
  const [lo, hi] = [new Array(st.n).fill(NaN), new Array(st.n).fill(NaN)];
  const pos = new Map<Value, number>(), neg = new Map<Value, number>();
  for (const [, idx] of groupsOf(st)) {
    for (const i of idx) {
      if (!Number.isFinite(vals[i])) continue;
      const sums = vals[i] < 0 ? neg : pos;
      const base = sums.get(band[i]) ?? 0;
      [lo[i], hi[i]] = [base, base + vals[i]];
      sums.set(band[i], base + vals[i]);
    }
  }
  if (o.offset === "expand" || o.offset === "center") {
    for (let i = 0; i < st.n; i += 1) {
      const [p, q] = [pos.get(band[i]) ?? 0, neg.get(band[i]) ?? 0];
      const [scale, shift] = o.offset === "expand" ? [1 / (p - q || 1), -q] : [1, -(p + q) / 2];
      [lo[i], hi[i]] = [(lo[i] + shift) * scale, (hi[i] + shift) * scale];
    }
  }
  st.ch.v = st.ch[v];
  st.names.v = st.names[v];
  [st.ch[`${v}1` as Ch], st.ch[`${v}2` as Ch], st.ch[v]] = [lo, hi, hi];
  st.stacked = true;
}

/** The transforms, in a fixed order: bin | group | window | density, then stack. */
function transform(st: Stage): void {
  const t = st.m.transform ?? {};
  const known = ["bin", "group", "window", "density", "stack", ...STATS];
  const bad = Object.keys(t).filter((k) => !known.includes(k));
  if (bad.length) throw new Error(`${st.where}: unknown transform ${bad.join(", ")} (known: ${known.join(", ")})`);
  if (t.bin) bin(st, opt(t.bin)!);
  if (t.group) group(st, opt(t.group)!);
  if (t.window) windowed(st, t.window);
  if (t.density) density(st, opt(t.density)!);
  if (t.stack) stack(st, opt(t.stack)!);
  for (const [k, name] of Object.entries(st.pending) as [Ch, string][]) { // e.g. fill: "y" after a bin
    const from = (["x", "y", "x1", "x2", "y1", "y2"] as Ch[]).find((c) => c === name && st.ch[c]);
    if (from) [st.ch[k], st.names[k]] = [st.ch[from], st.names[from] ?? from];
    else throw new Error(`${st.where}: ${k}: "${name}" is neither a column nor a transform output`);
  }
}

// --- Compile -------------------------------------------------------------------------------------

/** Whether a category of `band` holds rows of more than one z group (then they dodge; fill: the
 * category itself doesn't). */
function shared(st: Stage, band: Ch): boolean {
  const zs = new Map<Value, Value>();
  return Boolean(st.ch.z) && st.ch[band]!.some((b, i) => (zs.has(b) ? zs.get(b) !== st.ch.z![i] : (zs.set(b, st.ch.z![i]), false)));
}

interface Ctx {
  palette: string[];
  ink: string; // text and the default colour beside categorical colours
  xCats: string[];
  yCats: string[];
  colorCats: Value[];
  color: { scheme: string; invert: boolean; domain: [number, number] } | null;
  r: { range: [number, number]; domain: [number, number] };
}

/** Radius of `v` on a sqrt (area) scale from `domain` onto `range` (clamped; a collapsed domain: its low end). */
export function sqrtRadius(v: number, [d0, d1]: [number, number], [r0, r1]: [number, number]): number {
  return r0 + (r1 - r0) * Math.sqrt(Math.min(Math.max((v - d0) / (d1 - d0 || 1), 0), 1));
}

/** Marks compiled: their series (after `series`), the categories string x / y imply, the
 * colour bar of a continuous fill. */
export function compileMarks(marks: Mark[], axes: { x: AxisSpec; y: AxisSpec }, scales: Scales, palette: string[], ink: string, table: (path: string) => Columns): { series: Compiled[]; x: AxisSpec; y: AxisSpec; colorbar?: ColorBar; colors: Value[]; slices: Record<string, number[]> } {
  const slices: Record<string, number[]> = Object.create(null);
  const stages = marks.flatMap((m, i) => {
    const where = `marks[${i}] (${m?.type})`;
    if (!m || !["line", "dot", "bar", "barX", "area", "rect", "rule", "text", "interval", "box"].includes(m.type)) {
      throw new Error(`${where}: type must be one of line, dot, bar, barX, area, rect, rule, text, interval, box`);
    }
    if (m.fn && !["line", "area"].includes(m.type)) throw new Error(`${where}: fn works for line and area marks`);
    const cols = m.fn ? undefined : columns(m, where, table);
    const st = resolve(m, where, cols);
    const parts = m.slice === undefined ? [st] : sliceStages(st, cols, slices);
    parts.forEach(transform);
    return parts;
  });
  const seen = (cats: string[], col?: Value[]) => col?.forEach((v) => typeof v === "string" && !cats.includes(v) && cats.push(v));
  const xCats = [...(axes.x.categories ?? [])], yCats: string[] = [];
  for (const st of stages) seen(xCats, st.ch.x);
  for (const st of stages) seen(yCats, st.ch.y);
  const yAxis = axes.y.categories?.length ? [...axes.y.categories] : [...yCats].reverse(); // auto: the first at the top
  for (const c of yCats) if (!yAxis.includes(c)) yAxis.unshift(c);

  // Colour categories: z values (a categorical fill is z). An explicit categorical fill must fit
  // the palette; z alone colours its groups while all fit, else they share one colour.
  const cats = (strict: boolean) => {
    const out: Value[] = [...(scales.color?.domain ?? [])];
    for (const st of stages) if (!st.color && (!strict || fillIsZ(st))) st.ch.z?.forEach((v) => !out.includes(v) && out.push(v));
    return out;
  };
  let colorCats = cats(false);
  if (colorCats.length > palette.length) {
    colorCats = cats(true);
    for (const st of stages) st.plain = Boolean(st.ch.z) && !st.color && !fillIsZ(st);
  }
  if (colorCats.length > palette.length) {
    throw new Error(`color: ${colorCats.length} categories (${colorCats.slice(0, 12).map(String).join(", ")}${colorCats.length > 12 ? ", ..." : ""}); the palette has ${palette.length}: fold the rest into "other", or colour by a number`);
  }
  const numericFill = stages.filter((st) => st.ch.fill && !isCat(st.ch.fill));
  const fills = numericFill.flatMap((st) => st.ch.fill!.map(num));
  const given = scales.color?.domain?.length === 2 && scales.color.domain.every((v) => typeof v === "number") ? (scales.color.domain as [number, number]) : null;
  const scheme = colormapName(scales.color?.scheme ?? "viridis");
  if (scales.color?.scheme && !scheme) throw new Error(`color.scheme ${scales.color.scheme}: not a colormap`);
  const color = fills.some(Number.isFinite) ? { scheme: scheme!, invert: Boolean(scales.color?.invert), domain: given ?? finiteRange(fills) } : null;
  const rs = stages.flatMap((st) => (st.names.r ? st.ch.r!.map(num) : []));
  const ctx: Ctx = {
    palette, ink, xCats, yCats: yAxis, colorCats, color,
    r: { range: scales.r?.range ?? [2, 12], domain: [scales.r?.domain?.[0] ?? 0, scales.r?.domain?.[1] ?? Math.max(0, finiteRange(rs)[1])] },
  };
  return {
    series: stages.flatMap((st) => {
      const out = compileStage(st, ctx);
      if (st.slice === undefined) return out;
      const slice = { name: st.m.slice!, value: st.slice };
      return out.map((c) => ({ ...c, slice, tips: c.tips?.map((rows) => [...rows, [slice.name, f4(slice.value)] as [string, string]]) }));
    }),
    x: xCats.length ? { ...axes.x, categories: xCats } : axes.x,
    y: yCats.length ? { ...axes.y, categories: yAxis } : axes.y,
    colorbar: color ? { label: scales.color?.label ?? numericFill[0].names.fill ?? "", ...color } : undefined,
    colors: colorCats,
    slices,
  };
}

function hex(rgb: [number, number, number]): string {
  return `#${rgb.map((c) => Math.round(c).toString(16).padStart(2, "0")).join("")}`;
}

function compileStage(st: Stage, ctx: Ctx): Compiled[] {
  const m = st.m;
  const t = m.transform ?? {};
  const base = { width: m.width, dash: m.dash, extent: m.extent };
  if (m.fn) {
    const color = st.color ?? ctx.palette[0];
    const style = { ...base, label: m.label, color, legend: m.legend, opacity: typeof m.opacity === "number" ? m.opacity : undefined };
    return [m.type === "area" ? { ...style, fill: { lo: 0, hi: m.fn, opacity: style.opacity ?? 0.3 }, opacity: undefined } : { ...style, fn: m.fn }];
  }
  const ch = st.ch;
  const xCat = isCat(ch.x), yCat = isCat(ch.y);
  const px = (v: Value) => (typeof v === "string" ? ctx.xCats.indexOf(v) : num(v));
  const py = (v: Value) => (typeof v === "string" ? ctx.yCats.indexOf(v) : num(v));
  const col = (k: Ch, idx: number[], pos = k[0] === "x" ? px : k[0] === "y" ? py : num) => (ch[k] ? idx.map((i) => pos(ch[k]![i])) : undefined);
  const groups = groupsOf(st);
  const defaultColor = ctx.colorCats.length ? ctx.ink : ctx.palette[0];
  const statKind = STATS.find((k) => t[k]) ?? (m.type === "box" ? "boxplot" : undefined);
  const out: Compiled[] = [];
  groups.forEach(([key, idx], g) => {
    const cat = key !== undefined && !st.color && !st.plain ? ctx.colorCats.indexOf(key) : -1;
    const color = cat >= 0 ? ctx.palette[cat] : st.color ?? (m.type === "text" ? ctx.ink : defaultColor);
    const label = key !== undefined && key !== null && !st.plain ? String(key) : m.label ?? "";
    const style: Compiled = { ...base, label, color, legend: (m.legend ?? (m.type !== "text" && Boolean(label))) && !(st.plain && g > 0) };
    if (typeof m.opacity === "number") style.opacity = m.opacity;
    if (statKind) return out.push(...statSpecs(st, statKind, style, idx, px));

    // Per point: colour (numeric fill), alpha (opacity column), radius (r), hover rows
    const prims: Prims = {};
    if (ctx.color && ch.fill && !isCat(ch.fill)) {
      const [a, b] = ctx.color.domain;
      prims.colors = idx.map((i) => hex(colormapRgb(ctx.color!.scheme, (num(ch.fill![i]) - a) / (b - a || 1), ctx.color!.invert)));
    }
    if (st.names.opacity) prims.alphas = idx.map((i) => Math.min(Math.max(num(ch.opacity![i]), 0), 1));
    else if (typeof m.opacity === "number" && ["dot", "text", "bar", "barX", "rect"].includes(m.type)) prims.alphas = idx.map(() => m.opacity as number);
    if (ch.r) {
      prims.sizes = idx.map((i) => (st.names.r ? sqrtRadius(num(ch.r![i]), ctx.r.domain, ctx.r.range) : num(ch.r![i])));
    }
    prims.tips = idx.map((i) => tipRows(st, i));

    // Dodge: bars, texts, intervals of several groups side by side within a category's slot
    const band = bandAxes(st)[0];
    const dodged = ["bar", "barX", "text", "interval"].includes(m.type) && (band === "x" ? xCat : yCat) && !st.stacked && !ch[`${band}1` as Ch] && shared(st, band);
    const order = groups.map(([k]) => k).sort((p, q) => ctx.colorCats.indexOf(p ?? null) - ctx.colorCats.indexOf(q ?? null));
    const sub = dodged ? 0.8 / groups.length : 0.8;
    const shift = dodged ? -0.4 + (order.indexOf(key) + 0.5) * sub : 0;
    const X = col("x", idx)?.map((v) => v + (band === "x" ? shift : 0)), Y = col("y", idx)?.map((v) => v + (band === "y" ? shift : 0));

    switch (m.type) {
      case "line":
        return out.push({ ...style, ...prims, x: need(st, "x", "line") && X, y: need(st, "y", "line") && Y, interpolation: m.interpolation });
      case "dot": {
        need(st, "x", "dot");
        need(st, "y", "dot");
        const order = idx.map((_, j) => j);
        if (prims.sizes) order.sort((a, b) => prims.sizes![b] - prims.sizes![a]); // big ones first: small ones stay visible
        const pick = <T>(a?: T[]) => a && order.map((j) => a[j]);
        return out.push({ ...style, x: pick(X), y: pick(Y), markers: "only", colors: pick(prims.colors), alphas: pick(prims.alphas), sizes: pick(prims.sizes), tips: pick(prims.tips) });
      }
      case "bar":
      case "barX":
      case "rect": {
        const horiz = m.type === "barX";
        const bandRange = (k: "x" | "y", vs: number[] | undefined, w: number) => {
          const [a, b] = [col(`${k}1`, idx), col(`${k}2`, idx)];
          if (a && b) return [a, b];
          if (!vs) throw new Error(`${st.where}: needs ${k} (or ${k}1 and ${k}2)`);
          const half = ((k === "x" ? xCat : yCat) ? (m.type === "rect" ? 1 : sub) : closestSpacing(vs)) * w / 2;
          return [vs.map((v) => v - half), vs.map((v) => v + half)];
        };
        const valueRange = (k: "x" | "y", vs: number[] | undefined) => {
          const [a, b] = [col(`${k}1`, idx), col(`${k}2`, idx)];
          if (a && b) return [a, b];
          if (!vs) throw new Error(`${st.where}: needs ${k} (or ${k}1 and ${k}2)`);
          return [vs.map(() => 0), vs];
        };
        const [x1, x2] = m.type === "rect" || !horiz ? bandRange("x", X, 1) : valueRange("x", X);
        const [y1, y2] = m.type === "rect" || horiz ? bandRange("y", Y, 1) : valueRange("y", Y);
        const mid = (a: number[], b: number[]) => a.map((v, j) => (v + b[j]) / 2);
        const inset = m.inset ?? (m.type === "rect" ? 0 : 1);
        return out.push({
          ...style, ...prims, noLine: true, base: !horiz && m.type === "bar",
          x: horiz ? x2 : mid(x1, x2), y: horiz ? mid(y1, y2) : m.type === "rect" ? mid(y1, y2) : y2,
          rects: { x1, x2, y1, y2, inset: m.type === "rect" ? [inset, inset] : horiz ? [0, inset] : [inset, 0], radius: m.type === "rect" || st.stacked ? 0 : 4, end: m.type === "rect" ? null : horiz ? "x" : "y" },
        });
      }
      case "area": {
        const opacity = typeof m.opacity === "number" ? m.opacity : st.stacked ? 0.75 : 0.3;
        if ((ch.x1 || ch.xo1) && !ch.y1) { // as functions of y (violins: one per x category)
          const ys = need(st, "y", "area") && Y!;
          const lo = ch.x1 ? col("x1", idx)! : col("xo1", idx, num)!.map((o, j) => o + X![j]);
          const hi = ch.x2 ? col("x2", idx)! : col("xo2", idx, num)!.map((o, j) => o + X![j]);
          const byX = new Map<Value | undefined, number[]>();
          idx.forEach((i, j) => {
            const c = ch.xo1 ? ch.x![i] : undefined;
            (byX.get(c) ?? byX.set(c, []).get(c)!).push(j);
          });
          return [...byX.values()].forEach((js, n) => {
            const ord = js.sort((a, b) => ys[a] - ys[b]);
            out.push({ ...style, legend: style.legend && n === 0, fillY: { y: ord.map((j) => ys[j]), lo: ord.map((j) => lo[j]), hi: ord.map((j) => hi[j]), opacity } });
          });
        }
        const xs = need(st, "x", "area") && X!;
        const ord = xs.map((_, j) => j).sort((a, b) => xs[a] - xs[b]);
        const lo = col("y1", idx) ?? xs.map(() => 0), hi = col("y2", idx) ?? (need(st, "y", "area") && Y!);
        const pick = (a: number[]) => ord.map((j) => a[j]);
        return out.push({ ...style, x: pick(xs), fill: { lo: pick(lo), hi: pick(hi), opacity }, interpolation: m.interpolation, base: lo.some((v) => v === 0) });
      }
      case "rule": {
        const axis = ch.y && !ch.x ? "y" : ch.x && !ch.y ? "x" : null;
        if (!axis) throw new Error(`${st.where}: give y (horizontal rules) or x (vertical), not both`);
        const at = (axis === "y" ? Y : X)!;
        const ends = (k: Ch) => col(k, idx) ?? at.map(() => NaN);
        return out.push({ ...style, rules: { axis, at, lo: ends(axis === "y" ? "x1" : "y1"), hi: ends(axis === "y" ? "x2" : "y2") } });
      }
      case "text": {
        const text = ch.text ? idx.map((i) => f4(ch.text![i])) : idx.map(() => st.text ?? "");
        return out.push({ ...style, ...prims, x: need(st, "x", "text") && X, y: need(st, "y", "text") && Y, noLine: true, texts: { text, anchor: m.anchor ?? "mm", dx: m.dx ?? 0, dy: m.dy ?? 0 } });
      }
      case "interval": {
        const [lo, hi] = [col("y1", idx), col("y2", idx)];
        if (!lo || !hi || !X) throw new Error(`${st.where}: needs x, y1 and y2`);
        return out.push({ ...style, ...prims, x: X, y: Y ?? lo.map((v, j) => (v + hi[j]) / 2), noLine: true, intervals: { lo, hi, cap: m.cap ?? 6, label: st.names.y1 && st.names.y2 ? `${st.names.y1} … ${st.names.y2}` : "interval" } });
      }
    }
  });
  return out;
}

/** The stat series (line.ts `expandStat`) of a group: ecdf of x, qq of y (or x), linreg of x, y,
 * boxplot of y per x. */
function statSpecs(st: Stage, kind: typeof STATS[number], style: Compiled, idx: number[], px: (v: Value) => number): Compiled[] {
  const t = st.m.transform ?? {};
  const vals = (k: Ch, what: string, rows = idx) => rows.map((i) => num(need(st, k, what)[i]));
  switch (kind) {
    case "ecdf":
      return [{ ...style, samples: vals("x", "ecdf"), stat: { type: "ecdf", band: opt(t.ecdf)?.band } }];
    case "qq":
      return [{ ...style, samples: vals(st.ch.y ? "y" : "x", "qq"), stat: { type: "qq", reference: opt(t.qq)?.reference } }];
    case "linreg":
      return [{ ...style, x: vals("x", "linreg"), y: vals("y", "linreg"), stat: { type: "linreg", ...opt(t.linreg) } }];
    case "boxplot": {
      const width = opt(t.boxplot)?.width;
      if (!st.ch.x) return [{ ...style, samples: vals("y", "boxplot"), stat: { type: "box", position: 0, width } }];
      const by = new Map<Value, number[]>();
      for (const i of idx) (by.get(st.ch.x[i]) ?? by.set(st.ch.x[i], []).get(st.ch.x[i])!).push(i);
      return [...by].map(([x, rows]) => ({ ...style, label: style.label || String(x), legend: style.legend && Boolean(style.label), samples: vals("y", "boxplot", rows), stat: { type: "box" as const, position: px(x), width } }));
    }
  }
}

/** Hover rows of row i: its channels by source column name (ranges for bins and stacks). */
function tipRows(st: Stage, i: number): [string, string][] {
  const { ch, names } = st;
  const rows: [string, string][] = [];
  const add = (k: Ch, k1: Ch, k2: Ch, fallback: string) => {
    if (ch[k1] && ch[k2] && !st.stacked && (k1 === "x1" || !ch[k])) rows.push([names[k] ?? fallback, `${f4(ch[k1]![i])} … ${f4(ch[k2]![i])}`]);
    else if (ch[k] && (names[k] || isCat(ch[k]))) rows.push([names[k] ?? fallback, f4((st.stacked && ch.v && k === bandAxes(st)[1] ? ch.v : ch[k])![i])]);
  };
  add("x", "x1", "x2", "x");
  add("y", "y1", "y2", "y");
  for (const k of ["fill", "r", "opacity", "text"] as Ch[]) if (ch[k] && names[k] && !(k === "fill" && fillIsZ(st))) rows.push([names[k]!, f4(ch[k]![i])]);
  return rows;
}
