// Volume geometry and threshold helpers: non-uniform axes, box sizes, ray-march step counts.

import { createScale, type Domain, getShaderScale, type VisScale } from "./scales.ts";

export type Quality = "low" | "medium" | "high";
export const SAMPLES_PER_VOXEL: Record<Quality, number> = { low: 0.5, medium: 1, high: 2 }; // halved while interacting

const UNIFORMITY_TOLERANCE = 1e-2;
const FLOAT_PRECISION = 2 ** -20;
const MAX_STEP_DENSITY = 4;

export function isNonUniform(values: ArrayLike<number>): boolean {
  const n = values.length;
  if (n < 3) return false;
  const meanStep = (values[n - 1] - values[0]) / (n - 1);
  if (meanStep === 0 || !Number.isFinite(meanStep)) return false;
  const tolerance = Math.max(UNIFORMITY_TOLERANCE * Math.abs(meanStep), FLOAT_PRECISION * Math.max(Math.abs(values[0]), Math.abs(values[n - 1])));
  let uniform = true;
  for (let i = 1; i < n; i += 1) {
    const step = values[i] - values[i - 1];
    if (!(step / meanStep > 0)) return false;
    if (Math.abs(step - meanStep) > tolerance) uniform = false;
  }
  return !uniform;
}

export const isNonUniformAxis = (values: ArrayLike<number> | undefined, count: number): values is ArrayLike<number> =>
  values !== undefined && values.length >= count && isNonUniform(Array.prototype.slice.call(values, 0, count + 1));

function pixelEdges(values: ArrayLike<number> | undefined, count: number): number[] {
  if (!values) return Array.from({ length: count + 1 }, (_, i) => i);
  if (values.length === count) return [...Array.from(values), values[count - 1] + (values[1] - values[0])];
  return Array.from(values).slice(0, count + 1);
}

/** Voxel edges: non-uniform values are voxel centres (edges at mid-points). */
export function getCellEdges(values: ArrayLike<number> | undefined, count: number): number[] {
  if (values?.length !== count || !isNonUniformAxis(values, count)) return pixelEdges(values, count);
  return [
    values[0] - (values[1] - values[0]) / 2,
    ...Array.from({ length: count - 1 }, (_, i) => (values[i] + values[i + 1]) / 2),
    values[count - 1] + (values[count - 1] - values[count - 2]) / 2,
  ];
}

/** Relative positions (0 to 1) of voxel edges and centres, interleaved; undefined if uniform. */
export function getAxisLookup(values: ArrayLike<number> | undefined, count: number): Float32Array | undefined {
  if (!isNonUniformAxis(values, count)) return undefined;
  const edges = getCellEdges(values, count);
  const centres = values.length === count ? values : undefined;
  const [first, last] = [edges[0], edges[count]];
  return Float32Array.from({ length: 2 * count + 1 }, (_, i) => {
    const k = Math.floor(i / 2);
    const v = i % 2 === 0 ? edges[k] : (centres?.[k] ?? (edges[k] + edges[k + 1]) / 2);
    return (v - first) / (last - first);
  });
}

/** Coordinates at the two faces of the box along an axis (evenly spaced values are voxel starts). */
export function getAxisDomain(values: ArrayLike<number> | undefined, count: number): Domain {
  const edges = getCellEdges(values, count);
  const starts = !values || (values.length === count && !isNonUniformAxis(values, count));
  const shift = starts ? (edges[1] - edges[0]) / 2 : 0;
  return [edges[0] - shift, edges[count] - shift];
}

/** Box of the shape `[z, y, x]`, largest side 1. */
export function getBoxSize(shape: number[]): [number, number, number] {
  const [d, r, c] = shape;
  const m = Math.max(d, r, c);
  return [c / m, r / m, d / m];
}

export function normalizeBoxSize(size: [number, number, number]): [number, number, number] {
  const m = Math.max(...size);
  return m > 0 && Number.isFinite(m) ? [size[0] / m, size[1] / m, size[2] / m] : [1, 1, 1];
}

/** Physical size along x, y, z from the axes' extents (voxel edges), if all are usable. */
export function getBoxSizeFromAxes(shape: number[], axes: (ArrayLike<number> | undefined)[]): [number, number, number] | undefined {
  const [d, r, c] = shape;
  const sizes = [c, r, d];
  const ext = axes.map((a, i) => {
    const n = sizes[i];
    if (!a || (a.length !== n + 1 && (a.length !== n || n < 2))) return undefined;
    const e = getCellEdges(a, n);
    const x = Math.abs(e[n] - e[0]);
    return Number.isFinite(x) && x > 0 ? x : undefined;
  });
  return ext.every((x) => x !== undefined) ? (ext as [number, number, number]) : undefined;
}

/** Voxels along an axis for the step size: enough for the narrowest voxel, up to 4× more. */
export function getStepCount(knots: Float32Array | undefined, count: number): number {
  if (!knots) return count;
  let minWidth = 1;
  for (let k = 2; k < knots.length; k += 2) minWidth = Math.min(minWidth, knots[k] - knots[k - 2]);
  return Math.min(Math.max(count, 1 / minWidth), MAX_STEP_DENSITY * count);
}

/** Opacity ramp above the threshold, and the colours from its linear position. */
export function getThresholdUniforms(threshold: number, gamma: number): { oneOverRamp: number; colorStart: number; oneOverColorSpan: number } {
  const start = threshold ** (1 / gamma);
  const usable = start < 1;
  return { oneOverRamp: usable ? 1 / (1 - threshold) : 1, colorStart: usable ? start : 0, oneOverColorSpan: usable ? 1 / (1 - start) : 1 };
}

/** Colour domain of MIP and DVR: [value at threshold, max]. */
export function getThresholdDomain(domain: Domain, vis: VisScale, threshold: number): Domain {
  const [, gamma] = getShaderScale(vis);
  if (getThresholdUniforms(threshold, gamma).colorStart <= 0) return domain;
  return [createScale(vis, { domain, range: [0, 1], clamp: true }).invert(threshold), domain[1]];
}
