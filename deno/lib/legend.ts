/* Corner of a line plot for its legend: the one covering the fewest points, segments and
 * reference lines or bands. */

import type { NumArray } from "./nurbs.ts";

/* A reference line (`range` of one value) or band, along the axis of its values */
export interface LegendObstacle {
  axis: "x" | "y";
  range: [number, number];
}


const LEGEND_CORNERS = ["top-right", "top-left", "bottom-right", "bottom-left"];
const CORNER_SIZE = 0.35; // fraction of each axis, without a legend size

/* Size of a legend and its margin from the edges, as fractions of the canvas */
export interface LegendBox {
  width: number;
  height: number;
  marginX: number;
  marginY: number;
}

const CORNER_BOX = {
  width: CORNER_SIZE,
  height: CORNER_SIZE,
  marginX: 0,
  marginY: 0,
};

interface Rect {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

function getCornerRect(corner: string, box: LegendBox): Rect {
  const { width, height, marginX, marginY } = box;
  const isRight = corner.endsWith("right");
  const isTop = corner.startsWith("top");
  const x0 = isRight ? 1 - marginX - width : marginX;
  const y0 = isTop ? 1 - marginY - height : marginY;
  return { x0, x1: x0 + width, y0, y1: y0 + height };
}

/* Whether segment (x0, y0)-(x1, y1) crosses a rectangle (Liang–Barsky clipping) */
function crossesRect(
  [x0, y0]: [number, number],
  [x1, y1]: [number, number],
  rect: Rect,
): boolean {
  let [tMin, tMax] = [0, 1];
  for (const [p, q] of [
    [x0 - x1, x0 - rect.x0],
    [x1 - x0, rect.x1 - x0],
    [y0 - y1, y0 - rect.y0],
    [y1 - y0, rect.y1 - y0],
  ]) {
    if (p === 0) {
      if (q < 0) {
        return false; // parallel to this edge, outside
      }
    } else if (p < 0) {
      tMin = Math.max(tMin, q / p);
    } else {
      tMax = Math.min(tMax, q / p);
    }
  }
  return tMin <= tMax;
}

/* How much a rectangle covers a reference line (1) or band (1 to 5, by overlap), else 0 */
function getObstacleCover(
  { axis, range: ends }: LegendObstacle,
  rect: Rect,
  toRelative: (val: number) => number,
): number {
  const [lo, hi] = ends.map(toRelative).sort((a, b) => a - b);
  const [r0, r1] = axis === "x" ? [rect.x0, rect.x1] : [rect.y0, rect.y1];
  const overlap = Math.min(hi, r1) - Math.max(lo, r0); // NaN if not shown (e.g. on a log axis)
  return overlap >= 0 ? 1 + (4 * overlap) / (r1 - r0) : 0; // line: overlap of 0
}

/* Corner of the plot whose legend box covers the fewest curve points and segments, and reference
 * lines and bands, which weigh more (top right if several) */
export function getLegendCorner(
  abscissas: NumArray,
  curves: NumArray[],
  toRelativeX: (val: number) => number, // to [0, 1]
  toRelativeY: (val: number) => number,
  box: LegendBox = CORNER_BOX,
  obstacles: LegendObstacle[] = [],
): string {
  const rects = LEGEND_CORNERS.map((corner) => getCornerRect(corner, box));
  const counts = [0, 0, 0, 0];
  const xs = Array.from(abscissas, toRelativeX);

  // A covered reference line weighs as a tenth of the curve points (at least 10); a band more
  const weight = Math.max(10, (abscissas.length * curves.length) / 10);
  for (const obstacle of obstacles) {
    const toRelative = obstacle.axis === "x" ? toRelativeX : toRelativeY;
    for (const [k, rect] of rects.entries()) {
      counts[k] += weight * getObstacleCover(obstacle, rect, toRelative);
    }
  }

  for (const ordinates of curves) {
    let prev: [number, number] | undefined;
    for (const [i, val] of ordinates.entries()) {
      const pt: [number, number] = [xs[i], toRelativeY(val)];
      if (!Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) {
        prev = undefined; // gap in the curve
        continue;
      }
      for (const [k, rect] of rects.entries()) {
        const isIn =
          pt[0] >= rect.x0 &&
          pt[0] <= rect.x1 &&
          pt[1] >= rect.y0 &&
          pt[1] <= rect.y1;
        counts[k] +=
          (isIn ? 1 : 0) + (prev && crossesRect(prev, pt, rect) ? 1 : 0);
      }
      prev = pt;
    }
  }
  return LEGEND_CORNERS[counts.indexOf(Math.min(...counts))];
}
