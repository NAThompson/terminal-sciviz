/**
 * Regression checks of line.ts's data path on the CPU (lib/stats.ts, lib/marks.ts, lib/expr.ts,
 * the CSV reader), and two of the renderer's (extent: false, a failing marks
 * compile), each a case once reported:
 *
 *   deno run -A tools/check_marks.ts      # exits 1 if one fails
 */

import { closestSpacing, finiteRange, linreg } from "../lib/stats.ts";
import { compileMarks, type Mark } from "../lib/marks.ts";
import { compileExpr } from "../lib/expr.ts";
import { resolveFn } from "../lib/generators.ts";
import { type Control, DEFAULTS, INK, LinePlot, PALETTE, parseDelimited } from "../line.ts";
import { type Control as SceneControl, DEFAULTS as SCENE_DEFAULTS, ScenePlot } from "../scene.ts";
import { resolveSliders } from "../lib/sliders.ts";

let bad = 0;
function check(name: string, f: () => unknown): void {
  let ok: unknown;
  try { ok = f(); } catch (err) { ok = `threw ${(err as Error).message}`; }
  if (ok !== true) bad += 1;
  console.log(`${ok === true ? "ok  " : "FAIL"} ${name}${ok === true ? "" : `: ${ok}`}`);
}
const throws = (f: () => unknown, re: RegExp) => {
  try { f(); } catch (err) { return re.test((err as Error).message) || `wrong error: ${(err as Error).message}`; }
  return "did not throw";
};
const axes = { x: {}, y: {} };
const compile = (marks: Mark[]) => compileMarks(marks, axes, {}, PALETTE, INK, () => ({}));

// 1: min / max over large columns (a spread overflows the stack)
const N = 200_000;
const big = { x: Array.from({ length: N }, (_, i) => i), y: Array.from({ length: N }, (_, i) => Math.sin(i)), g: Array.from({ length: N }, (_, i) => "ab"[i % 2]) };
check("finiteRange of 200k", () => finiteRange(big.y)[1] <= 1 && finiteRange([NaN, 3, -1])[0] === -1);
check("dot, numeric fill and r over 200k rows", () => compile([{ type: "dot", data: big, x: "x", y: "y", fill: "y", r: "x" }]).colorbar!.domain[1] <= 1);
check("violin density over 200k rows", () => compile([{ type: "area", data: big, x: "g", y: "y", transform: { density: true } }]).series.map((s) => s.fillY!.y.length).join() === "200,200");

// 2: bar width with fewer than two distinct positions
check("closestSpacing of one position is 1", () => closestSpacing([3, 3]) === 1 && closestSpacing([]) === 1);
check("a bar at a single numeric x has finite width", () => {
  const r = compile([{ type: "bar", data: { x: [2], y: [5] }, x: "x", y: "y" }]).series[0].rects!;
  return r.x2[0] - r.x1[0] === 1 || `width ${r.x2[0] - r.x1[0]}`; // was Infinity
});

// 3: linreg's residual s without cancellation (residuals 0, 1, −1, 0 on a 1e9 slope)
check("linreg s on a steep line", () => {
  const x = [0, 1, 2, 3];
  const s = linreg(x, [0, 1000000001, 1999999999, 3000000000]).s, ref = linreg(x, [0, 1, -1, 0]).s;
  return Math.abs(s - ref) < 1e-6 * ref || `s = ${s}, want ${ref}`;
});

// 7: names of Object.prototype are not functions, constants or columns
check("param named constructor", () => compileExpr("constructor*x", ["constructor"]).f(3, [2]) === 6 && resolveFn({ expr: "constructor*x", params: { constructor: 2 } }).f(3) === 6);
check("toString is not a function", () => throws(() => compileExpr("toString(x)"), /unknown function "toString"/));
check("generator named constructor", () => throws(() => resolveFn({ name: "constructor" }), /fn must be one of/));
check("x: constructor, no such column", () => throws(() => compile([{ type: "dot", data: { a: [1] }, x: "constructor", y: "a" }]), /x: "constructor" is neither a column nor a transform output/));
check("fill: constructor, no such column: as any other name (a colour)", () => {
  const as = (fill: string) => compile([{ type: "bar", data: { a: [1, 2] }, x: "a", fill, transform: { bin: true } }]).series.map((s) => s.color);
  return JSON.stringify(as("constructor")) === JSON.stringify(as("toString").map(() => "constructor")) || JSON.stringify(as("constructor"));
});
check("a column named constructor", () => compile([{ type: "dot", data: { constructor: [1, 2], y: [3, 4] }, x: "constructor", y: "y" }]).series[0].x![1] === 2);

// 8: ln |Γ| below 0 (Boost's; Γ(−5/2) = −8√π/15; the poles NaN, a gap: +∞ in our old lgamma,
// whose −2.5 was off by 2e-11)
const lgamma = (x: number) => compileExpr("lgamma(x)").f(x, []);
check("lgamma reflected", () => Math.abs(lgamma(-2.5) - Math.log(8 * Math.sqrt(Math.PI) / 15)) < 1e-15 && Number.isNaN(lgamma(-3)) && Number.isFinite(lgamma(-1e8 + 0.5)) && Number.isNaN(lgamma(-1e17)));
check("poisson at x = −1e17 returns", () => typeof resolveFn({ name: "poisson", domain: [-1e8, 20] }).f(-1e17) === "number");

// 11: deep expressions fail at parse time, positioned
check("a 1000-term sum: a positioned parse error", () => throws(() => compileExpr(Array(1000).fill("x").join("+")), /nested more than 128 deep .* at character \d+/));
check("x^x^... 1000 deep: a positioned parse error", () => throws(() => compileExpr(Array(1000).fill("x").join("^")), /nested more than 128 deep .* at character \d+/));
check("a 100-term sum compiles", () => compileExpr(Array(100).fill("x").join("+")).f(1, []) === 100);
const parens = (n: number) => "(".repeat(n) + "x" + ")".repeat(n);
check("65 nested parentheses compile (one limit: 128 levels)", () => compileExpr(parens(65)).f(2, []) === 2 && compileExpr(parens(1000)).f(2, []) === 2);
const sins = (n: number) => "sin(".repeat(n) + "x" + ")".repeat(n);
check("128 nested calls compile, 129 fail with the one message", () =>
  Number.isFinite(compileExpr(sins(128)).f(1, [])) && throws(() => compileExpr(sins(129)), /nested more than 128 deep \(each operator/));

// 12: violins without a categorical fill (or with z ≠ x): one per category, not one zigzag
const trial = { g: [] as string[], v: [] as number[], arm: [] as string[] };
["a", "b", "c"].forEach((g, i) => { for (let j = 0; j < 40; j += 1) (trial.g.push(g), trial.v.push(i + Math.sin(j)), trial.arm.push("pq"[j % 2])); });
check("violin without fill: a polygon per category, each in its slot", () => {
  const s = compile([{ type: "area", data: trial, x: "g", y: "v", transform: { density: true } }]).series;
  return (s.length === 3 && s.every((c, k) => c.fillY!.lo.every((lo, j) => lo <= k && c.fillY!.hi[j] >= k && c.fillY!.hi[j] - lo < 1))) || `${s.length} series`;
});
check("violin with z ≠ x: a polygon per category and z", () => compile([{ type: "area", data: trial, x: "g", y: "v", z: "arm", transform: { density: true } }]).series.length === 6);

// 13: z alone never runs out of palette; a categorical fill still does
const ids = { t: [] as number[], y: [] as number[], id: [] as string[] };
for (let k = 0; k < 9; k += 1) for (let t = 0; t < 3; t += 1) (ids.t.push(t), ids.y.push(k + t), ids.id.push(`id${k}`));
check("line with z over 9 ids: one colour, one legend entry at most", () => {
  const s = compile([{ type: "line", data: ids, x: "t", y: "y", z: "id" }]).series;
  return (s.length === 9 && new Set(s.map((c) => c.color)).size === 1 && s[0].color === PALETTE[0] && s.every((c) => !c.legend)) || JSON.stringify(s.map((c) => [c.color, c.legend]));
});
check("line with z over 3 ids: palette colours", () => compile([{ type: "line", data: { t: ids.t.slice(0, 9), y: ids.y.slice(0, 9), id: ids.id.slice(0, 9) }, x: "t", y: "y", z: "id" }]).series.map((c) => c.color).join() === PALETTE.slice(0, 3).join());
check("fill over 9 categories still throws", () => throws(() => compile([{ type: "line", data: ids, x: "t", y: "y", fill: "id" }]), new RegExp(`the palette has ${PALETTE.length}`)));

// 14: window keeps fill as z (hover shows the category once; a later transform keeps the fill)
const win = { x: [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], y: [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], g: [..."aaaaaabbbbbb"] };
check("window k = 5 with fill: the category once in the tip", () => {
  const tip = compile([{ type: "line", data: win, x: "x", y: "y", fill: "g", transform: { window: { k: 5 } } }]).series[0].tips![0];
  return tip.filter(([, v]) => v === "a").length <= 1 && !tip.some(([k]) => k === "g") || JSON.stringify(tip);
});
check("window then density with fill: the fill kept (still a fill: 9 categories throw)", () => {
  const s = compile([{ type: "area", data: win, x: "x", y: "y", fill: "g", transform: { window: { k: 5 }, density: true } }]).series;
  return (s.length === 2 && s[0].color === PALETTE[0] && s[1].color === PALETTE[1] &&
    throws(() => compile([{ type: "area", data: ids, x: "t", y: "y", fill: "id", transform: { window: { k: 2 }, density: true } }]), new RegExp(`the palette has ${PALETTE.length}`))) || JSON.stringify(s.map((c) => c.color));
});

// 6: CSV (RFC 4180) and TSV
check("CSV quoted cells", () => {
  const rows = parseDelimited('# comment\n"name, full",value\n"a ""b""",1\r\n"two\nlines", 2\n\nc,\n');
  return JSON.stringify(rows) === JSON.stringify([["name, full", "value"], ['a "b"', "1"], ["two\nlines", "2"], ["c", ""]]) || JSON.stringify(rows);
});
check("TSV split as is", () => JSON.stringify(parseDelimited('a\tb\n"x\t1\n')) === JSON.stringify([["a", "b"], ['"x', "1"]]));
check("whitespace separated", () => JSON.stringify(parseDelimited("1 2\n 3   4 \n")) === JSON.stringify([["1", "2"], ["3", "4"]]));
check("unterminated quote", () => throws(() => parseDelimited('a,"b\n'), /unterminated quoted cell/));

// The renderer: 9 (extent: false markers set no y range), 10 (a failing marks compile)
const plot = await LinePlot.create();
const frame = async (over: Partial<Control>) => {
  const ctl = { ...structuredClone(DEFAULTS), width: 400, height: 300, ...over } as Control;
  const p = await plot.submit(ctl, 0);
  await p.done;
  p.read((n) => new Uint8Array(n));
  return plot.state(ctl, 0, null, 0) as { ymax: number };
};
{
  const s = await frame({ series: [{ y: [0, 1, 0.5] }, { x: [1], y: [1000], markers: "only", extent: false }, { x: [0, 2], y: [0, 0], intervals: { lo: [-500, -500], hi: [500, 500] }, extent: false }] });
  check("extent: false markers and intervals leave the y range", () => s.ymax < 2 || `ymax ${s.ymax}`);
}
{
  const good: Mark[] = [{ type: "dot", data: { x: [1, 2], y: [1, 2], v: [0, 1] }, x: "x", y: "y", fill: "v" }];
  await frame({ series: [], marks: good });
  const p = plot as unknown as { colorbar?: unknown; marked?: unknown };
  const [bar, marked] = [p.colorbar, p.marked];
  const errs: unknown[] = [];
  for (let k = 0; k < 2; k += 1) await frame({ series: [], marks: [{ type: "dot", data: { a: [1] }, x: "nope", y: "a" }] }).catch((e) => errs.push(e));
  check("a failing marks compile: reported, not recompiled, the last good kept", () =>
    (errs.length === 2 && errs[0] === errs[1] && p.colorbar === bar && p.marked === marked) || `errors ${errs.length}, same ${errs[0] === errs[1]}`);
  await frame({ series: [], marks: good });
  check("then a good compile again", () => p.colorbar !== undefined);
}

// 11: slices: a mark's rows split by a column before its transforms, every slice compiled once
{
  const data = { x: [1, 2, 3, 1, 2, 3, 1], t: [0, 0, 0, 0.5, 0.5, 0.5, 1], v: [1, 2, 3, 10, 20, 30, 5] };
  const c = compile([{ type: "dot", data, x: "x", y: "v", fill: "v", slice: "t" }, { type: "line", data: { x: [0, 3], y: [0, 0] }, x: "x", y: "y" }]);
  check("slice: one series per value, tagged, the rest untagged", () =>
    JSON.stringify(c.series.map((s) => s.slice?.value ?? null)) === "[0,0.5,1,null]" && JSON.stringify(c.slices) === '{"t":[0,0.5,1]}' || JSON.stringify(c.series.map((s) => s.slice)));
  check("slice: the colour domain over every slice", () => JSON.stringify(c.colorbar!.domain) === "[1,30]" || JSON.stringify(c.colorbar!.domain));
  check("slice: rows of the value, its row in the hover", () => c.series[1].y!.join() === "10,20,30" && c.series[1].tips![0].at(-1)!.join() === "t,0.5");
  const binned = compile([{ type: "bar", data, x: "x", transform: { bin: { thresholds: 3 } }, slice: "t" }]);
  check("slice: transforms per slice (bin counts)", () => binned.series.map((s) => s.y!.reduce((a, b) => a + b, 0)).join() === "3,3,1" || binned.series.map((s) => s.y).join("|"));
  check("slice: not a column", () => throws(() => compile([{ type: "dot", data, x: "x", y: "v", slice: "time" }]), /slice: "time" is not a column/));
  check("slice: categorical column", () => throws(() => compile([{ type: "dot", data: { x: [1], y: [1], g: ["a"] }, x: "x", y: "y", slice: "g" }]), /must be numeric/));
  const [d] = resolveSliders({ t: {} }, { t: 0.7 }, [], { t: [0, 0.5, 1] });
  check("slice slider: snaps to the nearest value, the values reported", () => (d.value === 0.5 && d.min === 0 && d.max === 1 && d.values!.length === 3) || JSON.stringify(d));
  check("slice slider: [min, max] keeps the values within", () => resolveSliders({ t: [0.2, 1] }, null, [], { t: [0, 0.5, 1] })[0].values!.join() === "0.5,1");
  check("slider of no parameter or slice", () => throws(() => resolveSliders({ q: [0, 1] }, null, [], { t: [0] }), /no parameter "q".*slice columns: t/));
}
{
  const data = { x: [0, 1, 2, 0, 1, 2], y: [0, 1, 0, 5, 9, 5], t: [0, 0, 0, 1, 1, 1] };
  const marks: Mark[] = [{ type: "line", data, x: "x", y: "y", slice: "t" }];
  const s0 = await frame({ series: [], marks }) as { ymax: number; sliders?: { name: string; value: number; values: number[] }[] };
  check("chart: a slider added for the slice column, at its first value", () => (s0.sliders?.[0]?.name === "t" && s0.sliders[0].value === 0 && s0.ymax < 2) || JSON.stringify(s0));
  const held = await frame({ series: [], marks, sliderValues: { t: 1 }, interacting: true });
  check("chart: dragging (interacting) holds the y range", () => held.ymax === s0.ymax || `ymax ${held.ymax}`);
  const t0 = performance.now();
  for (let k = 0; k < 20; k += 1) await frame({ series: [], marks, sliderValues: { t: k % 2 }, interacting: true });
  console.log(`     chart slice switch (frame, 400×300): ${((performance.now() - t0) / 20).toFixed(1)} ms`);
  const released = await frame({ series: [], marks, sliderValues: { t: 1 } });
  check("chart: released, refitted to the slice", () => released.ymax > 9 || `ymax ${released.ymax}`);
}
{
  const scene = await ScenePlot.create();
  const sframe = async (over: Partial<SceneControl>) => {
    const ctl = { ...structuredClone(SCENE_DEFAULTS), width: 400, height: 300, ...over } as SceneControl;
    const p = await scene.submit(ctl, 0);
    await p.done;
    p.read((n) => new Uint8Array(n));
    return scene as unknown as { gpu: { dots: { count: number }; mesh: { count: number }; segs: { count: number } }; state: ScenePlot["state"] };
  };
  const data = { x: [0, 1, 0, 0, 1, 0, 9], y: [0, 0, 1, 0, 0, 1, 9], z: [0, 0, 0, 1, 1, 1, 9], t: [0, 0, 0, 1, 1, 1, 2] };
  const marks = [
    { type: "dot", data, slice: "t" },
    { type: "mesh", data, faces: [[0, 1, 2], [3, 4, 5], [2, 3, 4]], slice: "t" },
    { type: "line", data, slice: "t" },
    { type: "dot", data: { x: [5], y: [5], z: [5] } },
  ] as SceneControl["marks"];
  const g = await sframe({ marks, sliderValues: { t: 1 } });
  check("scene: dots, mesh faces, line segments of the slice (faces across slices dropped)", () =>
    (g.gpu.dots.count === 4 && g.gpu.mesh.count === 3 && g.gpu.segs.count === 2) || JSON.stringify([g.gpu.dots.count, g.gpu.mesh.count, g.gpu.segs.count]));
  const st = g.state({ ...SCENE_DEFAULTS, marks } as SceneControl, 0, null, 0) as { sliders: { name: string; value: number }[]; box: { hi: number[] } };
  check("scene: the slider, the box over every slice", () => (st.sliders[0].name === "t" && st.sliders[0].value === 1 && st.box.hi[0] > 9) || JSON.stringify(st));
  const alt = await sframe({ marks: [{ type: "line", data: { x: [0, 1, 2, 3], y: [0, 0, 0, 0], z: [0, 1, 0, 1], t: [0, 1, 0, 1] }, slice: "t" }] as SceneControl["marks"], sliderValues: { t: 1 } });
  check("scene: a line of alternating slices joins the slice's rows", () => alt.gpu.segs.count === 1 || `segments ${alt.gpu.segs.count}`);
  const cube = (o: number) => [0, 1, 2, 3, 4, 5, 6, 7].map((c) => [o + (c & 1), (c >> 1) & 1, (c >> 2) & 1]);
  const pts = [...cube(0), ...cube(10)];
  const hulls = await sframe({ marks: [{ type: "dot", data: { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2]), t: pts.map((_, i) => (i < 8 ? 0 : 1)) }, transform: { hull: true }, slice: "t" }] as SceneControl["marks"], sliderValues: { t: 1 } });
  check("scene: a hull per slice (a cube's 12 triangles, not the hull of both)", () => hulls.gpu.mesh.count === 0 && (hulls as unknown as { gpu: { blend: { tris: Float32Array } } }).gpu.blend.tris.length / 36 === 12 || `opaque ${hulls.gpu.mesh.count}, translucent ${(hulls as unknown as { gpu: { blend: { tris: Float32Array } } }).gpu.blend.tris.length / 36}`);
}

console.log(bad ? `${bad} failed` : "all passed");
Deno.exit(bad ? 1 : 0);
