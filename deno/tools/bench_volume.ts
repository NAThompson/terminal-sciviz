/**
 * Slice-slider latency of volume.ts on a file-backed 4D stack, in process (as the gallery): open
 * (first frame at slice 16), stepping 16 → 24 as a drag does (`sliderValues`, `interacting`), the
 * settled frame on release (isosurfaces extracted), a warm frame, and a PNG at slice 16. Steps
 * one at a time (latency: each frame waited for), and as `serve` pipelines them (3 in flight:
 * time per step), through slices not read yet (cold) and read (warm).
 *
 *   deno run -A tools/bench_volume.ts FILE.npy [-o out.png] [--from 16] [--to 24]
 */

import { renderOnce } from "../lib/serve.ts";
import { encodePng } from "../lib/png.ts";
import { DEFAULTS, VolumePlot } from "../volume.ts";

const args = Deno.args;
const opt = (f: string, d: string) => (args.includes(f) ? args[args.indexOf(f) + 1] : d);
const path = args.find((a, i) => !a.startsWith("-") && !["-o", "--from", "--to"].includes(args[i - 1]));
if (!path) throw new Error("usage: bench_volume.ts FILE.npy [-o out.png]");
const from = Number(opt("--from", "16")), to = Number(opt("--to", "24"));
const out = opt("-o", "");
const R2 = (k: number) => (1.95 * k) / 39; // the r₂ axis value of slice k (40 slices over [0, 1.95])

const base = {
  ...DEFAULTS,
  volume: {
    dataset: "file", path,
    axes: [{ lo: -2, hi: 2, label: "x", unit: "a₀" }, { lo: -2, hi: 2, label: "y", unit: "a₀" }, { lo: -2, hi: 2, label: "z → e₂", unit: "a₀" }, { lo: 0, hi: 1.95, label: "r₂", unit: "a₀" }],
  },
  up: "z", sliders: { slice: { label: "r₂" } }, slice: 16, colormap: "viridis", domain: [0.418, 2.435], scale: "root", rootN: 2.35,
  mode: "DVR", threshold: 0.5, opacity: 0.04, isoLevels: [2.8], isoColors: "levels", quality: "high", elevation: 10, azimuth: 40, zoom: 1.3,
} as never as typeof DEFAULTS;

const t0 = performance.now();
const plot = await VolumePlot.create();
const createMs = performance.now() - t0;
const frame = async (ctl: Partial<typeof DEFAULTS>) => {
  const t = performance.now();
  const r = await renderOnce(plot, { ...base, ...ctl } as never);
  return { ms: performance.now() - t, ...r, st: r.state as Record<string, any> };
};
const ms = (v: number) => v.toFixed(1);

const open = await frame({});
if (out) Deno.writeFileSync(out, await encodePng(open.frame.rgba, open.frame.width, open.frame.height));
const steps: number[] = [];
for (let k = from + 1; k <= to; k += 1) steps.push((await frame({ slice: from, sliderValues: { slice: R2(k) }, interacting: true })).ms);
const back: number[] = [];
for (let k = to - 1; k >= from; k -= 1) back.push((await frame({ slice: from, sliderValues: { slice: R2(k) }, interacting: true })).ms);
// as `serve` does: up to 3 frames in flight, the next control submitted meanwhile (throughput)
const piped = async (ks: number[]) => {
  const t = performance.now(), flights: Promise<unknown>[] = [];
  for (const [i, k] of ks.entries()) {
    const ctl = { ...base, slice: from, sliderValues: { slice: R2(k) }, interacting: true } as never;
    const p = await plot.submit(ctl, i % 3);
    flights.push(p.done.then(() => p.read((n) => new Uint8Array(n))));
    if (flights.length >= 3) await flights.shift();
  }
  await Promise.all(flights);
  return (performance.now() - t) / ks.length;
};
const pipedCold = await piped(Array.from({ length: to - from }, (_, i) => to + 1 + i).filter((k) => k < 40));
const pipedWarm = await piped(Array.from({ length: to - from + 1 }, (_, i) => from + i));
const settled = await frame({ slice: from, sliderValues: { slice: R2(to) }, interacting: false });
const warm = await frame({ slice: from, sliderValues: { slice: R2(to) }, interacting: false });
const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
const iso = (st: Record<string, any>) => (st.isosurfaces ?? []).map((m: { ms: number; triangles: number }) => `${m.ms} ms/${m.triangles} tris`).join(", ");
console.log(JSON.stringify({
  file: path.split("/").pop(), shape: open.st.shape, createMs: ms(createMs), openMs: ms(open.ms), openFile: open.st.file, openIso: iso(open.st),
  stepMs: { median: ms(med(steps)), max: ms(Math.max(...steps)), all: steps.map(ms) },
  backMs: { median: ms(med(back)), max: ms(Math.max(...back)) }, pipedColdMs: ms(pipedCold), pipedWarmMs: ms(pipedWarm),
  settledMs: ms(settled.ms), settledIso: iso(settled.st), warmMs: ms(warm.ms), slice: settled.st.slice,
}));
Deno.exit(0);
