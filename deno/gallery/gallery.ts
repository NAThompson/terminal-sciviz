/**
 * Gallery harness: renders every entry headlessly, in process (each renderer module imported once,
 * its class's `create()` called once and reused), with labelled contact sheets per section, and
 * compares against golden images.
 *
 *   deno run -A gallery/gallery.ts render [--section S] [--out DIR] [--sheets DIR] [--size WxH] [--cols 2] [--thumb 800]
 *   deno run -A gallery/gallery.ts check  [--section S] [--size WxH] [--update]
 *   deno run -A gallery/gallery.ts manifest
 *
 * render: DIR/<id>.png per entry (default DIR: gallery/out) and the contact sheet of each section,
 * SHEETS/contact_<section>.png (default SHEETS: docs/ at the repository root, shown by the
 * README): a grid of `--cols` thumbnails `--thumb` px wide, titled by id, with the wrapped
 * description. Every entry module is required (a broken one fails the run, as does a failed
 * entry).
 * check: re-renders and compares with gallery/golden/<id>.png: max and mean absolute channel
 * difference (straight-alpha RGBA8), pass at max <= 3 and mean <= 0.01; --update rewrites them.
 * Also fails if gallery/index.json is stale (--update rewrites it) or any text drew a glyph
 * the font atlas lacks.
 * manifest: writes gallery/index.json, every entry as pretty-printed JSON (committed: the Claude
 * Code plugin reads the entry list from it, with no build step).
 */

import { encodePng, decodePng } from "../lib/png.ts";
import { BitmapFont, type FontSize, missingGlyphs } from "../lib/overlay.ts";
import { type Control, renderOnce, type Renderer } from "../lib/serve.ts";
import type { Entry, Section } from "./types.ts";

const HERE = new URL("./", import.meta.url);
const ROOT = new URL("../", import.meta.url);
const INDEX = new URL("index.json", HERE);
const DOCS = new URL("../../docs/", import.meta.url);
const SECTIONS: Section[] = ["lines", "curves", "complex", "volumes"];
const INK = [0xc9, 0xd1, 0xd9], INK2 = [0x8b, 0x94, 0x9e];
const MAX_TOL = 3, MEAN_TOL = 0.01;

/** Entries of every section. Each module is required: a missing or broken one fails the run. */
async function loadEntries(): Promise<Entry[]> {
  const files = ["lines_entries.ts", "curves_entries.ts", "complex_entries.ts", "volumes_entries.ts"];
  const out: Entry[] = [];
  for (const f of files) {
    const mod = await import(new URL(f, HERE).href).catch((err) => {
      throw new Error(`gallery/${f}: ${err instanceof Error ? err.message : err}`);
    });
    if (!Array.isArray(mod.entries) || !mod.entries.length) {
      throw new Error(`gallery/${f}: no entries exported`);
    }
    out.push(...(mod.entries as Entry[]));
  }
  const ids = out.map((e) => e.id);
  const dup = ids.find((id, i) => ids.indexOf(id) !== i);
  if (dup) throw new Error(`duplicate entry id ${dup}`);
  return out;
}

// --- Renderers: one instance per script, in process ----------------------------------------------

type Factory = { defaults: Control; renderer: Renderer<Control> };
const factories = new Map<string, Promise<Factory>>();

function factory(script: string): Promise<Factory> {
  let f = factories.get(script);
  if (!f) {
    f = (async () => {
      const mod = await import(new URL(script, ROOT).href);
      const cls = Object.values(mod).find((v) => typeof v === "function" && typeof (v as { create?: unknown }).create === "function") as
        | { create(): Promise<Renderer<Control>> }
        | undefined;
      if (!cls || !mod.DEFAULTS) throw new Error(`${script}: no exported DEFAULTS and class with static create()`);
      return { defaults: mod.DEFAULTS as Control, renderer: await cls.create() };
    })();
    factories.set(script, f);
  }
  return f;
}

async function render(entry: Entry, width: number, height: number): Promise<{ rgba: Uint8Array; width: number; height: number; ms: number }> {
  const { defaults, renderer } = await factory(entry.renderer);
  const ctl = { ...defaults, ...entry.control, width, height, hover: null, snapshot: null, rev: 0 } as Control;
  const t0 = performance.now();
  const { frame } = await renderOnce(renderer, ctl);
  return { rgba: frame.rgba, width: frame.width, height: frame.height, ms: performance.now() - t0 };
}

// --- Contact sheets ------------------------------------------------------------------------------

/** Area-average downscale by an integer factor (premultiplied, back to straight alpha). */
function downscale(rgba: Uint8Array, w: number, h: number, f: number): { rgba: Uint8Array; w: number; h: number } {
  const tw = Math.floor(w / f), th = Math.floor(h / f);
  const out = new Uint8Array(tw * th * 4);
  for (let y = 0; y < th; y += 1) {
    for (let x = 0; x < tw; x += 1) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let j = 0; j < f; j += 1) {
        let i = ((y * f + j) * w + x * f) * 4;
        for (let k = 0; k < f; k += 1, i += 4) {
          const al = rgba[i + 3];
          r += rgba[i] * al;
          g += rgba[i + 1] * al;
          b += rgba[i + 2] * al;
          a += al;
        }
      }
      const o = (y * tw + x) * 4;
      if (a > 0) {
        out[o] = Math.round(r / a);
        out[o + 1] = Math.round(g / a);
        out[o + 2] = Math.round(b / a);
        out[o + 3] = Math.round(a / (f * f));
      }
    }
  }
  return { rgba: out, w: tw, h: th };
}

/** Text from the font atlas, straight alpha over a transparent sheet (max coverage). */
function drawText(img: Uint8Array, W: number, font: BitmapFont, size: FontSize, text: string, x: number, top: number, color: number[], maxWidth: number): void {
  let s = text;
  if (font.measure(size, s) > maxWidth) {
    while (s.length > 1 && font.measure(size, s + "…") > maxWidth) s = [...s].slice(0, -1).join("");
    s += "…";
  }
  font.raster(size, s, x, Math.round(top + size.ascent), (px, py, cov) => {
    const o = (py * W + px) * 4;
    const c = Math.round(cov * 255);
    if (px < 0 || px >= W || o < 0 || o >= img.length || c < img[o + 3]) return;
    img[o] = color[0];
    img[o + 1] = color[1];
    img[o + 2] = color[2];
    img[o + 3] = c;
  });
}

/** Words of `text` in lines no wider than `width`. */
function wrap(font: BitmapFont, size: FontSize, text: string, width: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (line && font.measure(size, next) > width) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

interface Thumb { entry: Entry; rgba: Uint8Array; w: number; h: number }

/** Thumbnail of a frame `thumbWidth` wide (integer area-average factor). */
function thumbnail(entry: Entry, rgba: Uint8Array, width: number, height: number, thumbWidth: number): Thumb {
  const t = downscale(rgba, width, height, Math.max(1, Math.round(width / thumbWidth)));
  return { entry, rgba: t.rgba, w: t.w, h: t.h };
}

async function contactSheet(section: string, items: Thumb[], cols: number, path: string): Promise<void> {
  const font = await BitmapFont.load(new URL("assets/", ROOT));
  const title = font.get("bold", 17), desc = font.get("regular", 14), head = font.get("bold", 26);
  const tw = Math.max(...items.map((t) => t.w)), th = Math.max(...items.map((t) => t.h));
  cols = Math.min(cols, items.length);
  const gap = 32, pad = 28;
  const titleH = Math.round(title.ascent + title.descent + 6); // whole pixels: offsets into the sheet
  const lineH = Math.round(desc.ascent + desc.descent + 3);
  const descs = items.map((t) => wrap(font, desc, t.entry.description, tw));
  const rows = Math.ceil(items.length / cols);
  // Per grid row: text height of its tallest description
  const textH = Array.from({ length: rows }, (_, r) =>
    titleH + Math.max(...descs.slice(r * cols, r * cols + cols).map((d) => d.length)) * lineH + 8);
  const headH = Math.round(head.ascent + head.descent + 20);
  const W = pad * 2 + cols * tw + (cols - 1) * gap;
  const H = pad * 2 + headH + textH.reduce((a, b) => a + b, 0) + rows * th + (rows - 1) * gap;
  const img = new Uint8Array(W * H * 4);
  drawText(img, W, font, head, `${section} · ${items.length} entries`, pad, pad, INK, W - 2 * pad);
  let rowTop = pad + headH;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const n = r * cols + c;
      if (n >= items.length) break;
      const { entry, rgba, w, h } = items[n];
      const x = pad + c * (tw + gap);
      drawText(img, W, font, title, entry.id, x, rowTop, INK, tw);
      descs[n].forEach((line, i) => drawText(img, W, font, desc, line, x, rowTop + titleH + i * lineH, INK2, tw));
      const top = rowTop + textH[r];
      for (let j = 0; j < h; j += 1) {
        img.set(rgba.subarray(j * w * 4, (j + 1) * w * 4), ((top + j) * W + x) * 4);
      }
      // frame of the cell (faint), so empty space reads as the plot's extent
      const edge = (o: number) => {
        if (img[o + 3] === 0) img.set([0x30, 0x36, 0x3d, 255], o);
      };
      for (let i = 0; i < w; i += 1) for (const yy of [top, top + h - 1]) edge((yy * W + x + i) * 4);
      for (let j = 0; j < h; j += 1) for (const xx of [x, x + w - 1]) edge(((top + j) * W + xx) * 4);
    }
    rowTop += textH[r] + th + gap;
  }
  Deno.writeFileSync(path, await encodePng(img, W, H));
}

// --- Commands ------------------------------------------------------------------------------------

function parseArgs(args: string[]) {
  const opt = (name: string) => {
    const i = args.indexOf(name);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const [w, h] = (opt("--size") ?? "1600x1000").split("x").map(Number);
  return {
    section: opt("--section"),
    out: opt("--out"),
    sheets: opt("--sheets"),
    width: w,
    height: h,
    update: args.includes("--update"),
    cols: Number(opt("--cols") ?? 2), // contact sheets: columns of thumbnails
    thumb: Number(opt("--thumb") ?? 800), // and their width
  };
}

function mkdirFor(path: string): void {
  Deno.mkdirSync(path.slice(0, path.lastIndexOf("/")), { recursive: true });
}

async function cmdRender(args: string[]): Promise<void> {
  const o = parseArgs(args);
  const out = o.out ?? new URL("out", HERE).pathname;
  const t0 = performance.now();
  const entries = (await loadEntries()).filter((e) => !o.section || e.section === o.section);
  const bySection = new Map<string, Thumb[]>(); // thumbnails only, not the frames
  let failed = 0;
  for (const entry of entries) {
    try {
      const r = await render(entry, o.width, o.height);
      const path = `${out}/${entry.id}.png`;
      mkdirFor(path);
      Deno.writeFileSync(path, await encodePng(r.rgba, r.width, r.height));
      console.log(`${entry.id.padEnd(34)} ${r.ms.toFixed(0).padStart(5)} ms  ${path}`);
      bySection.set(entry.section, [...(bySection.get(entry.section) ?? []), thumbnail(entry, r.rgba, r.width, r.height, o.thumb)]);
    } catch (err) {
      failed += 1;
      console.error(`${entry.id}: FAILED ${err instanceof Error ? err.stack : err}`);
    }
  }
  for (const section of SECTIONS) {
    const items = bySection.get(section);
    if (items?.length) {
      const path = `${o.sheets ?? DOCS.pathname.replace(/\/$/, "")}/contact_${section}.png`;
      mkdirFor(path);
      await contactSheet(section, items, o.cols, path);
      console.log(`contact sheet ${path}`);
    }
  }
  console.log(`${entries.length - failed}/${entries.length} entries in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  if (missingGlyphs.size) console.log(`FAIL: drawn as "?" (not in the font atlas): ${[...missingGlyphs].join(" ")}`);
  if (failed || missingGlyphs.size) Deno.exit(1);
}

async function cmdCheck(args: string[]): Promise<void> {
  const o = parseArgs(args);
  const golden = new URL("golden", HERE).pathname;
  const t0 = performance.now();
  const entries = (await loadEntries()).filter((e) => !o.section || e.section === o.section);
  let failed = 0;
  for (const entry of entries) {
    const path = `${golden}/${entry.id}.png`;
    let line = entry.id.padEnd(34);
    try {
      const r = await render(entry, o.width, o.height);
      if (o.update) {
        mkdirFor(path);
        Deno.writeFileSync(path, await encodePng(r.rgba, r.width, r.height));
        console.log(`${line} updated`);
        continue;
      }
      let g: { width: number; height: number; rgba: Uint8Array };
      try {
        g = await decodePng(Deno.readFileSync(path));
      } catch {
        failed += 1;
        console.log(`${line} FAIL (no golden: run with --update)`);
        continue;
      }
      if (g.width !== r.width || g.height !== r.height) {
        failed += 1;
        console.log(`${line} FAIL (size ${r.width}x${r.height}, golden ${g.width}x${g.height})`);
        continue;
      }
      let max = 0, sum = 0;
      for (let i = 0; i < r.rgba.length; i += 1) {
        const d = Math.abs(r.rgba[i] - g.rgba[i]);
        sum += d;
        if (d > max) max = d;
      }
      const mean = sum / r.rgba.length;
      const ok = max <= MAX_TOL && mean <= MEAN_TOL;
      if (!ok) failed += 1;
      line += `max ${String(max).padStart(3)}  mean ${mean.toFixed(4)}  ${ok ? "pass" : "FAIL"}`;
      console.log(line);
    } catch (err) {
      failed += 1;
      console.log(`${line} FAIL ${err instanceof Error ? err.message : err}`);
    }
  }
  console.log(`${entries.length - failed}/${entries.length} ${o.update ? "updated" : "pass"} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  if (o.update) {
    await writeIndex();
  } else if (!o.section && (await Deno.readTextFile(INDEX).catch(() => "")) !== await indexText()) {
    failed += 1;
    console.log("FAIL: gallery/index.json is stale (run `deno task manifest`)");
  }
  if (missingGlyphs.size) {
    failed += 1;
    console.log(`FAIL: drawn as "?" (not in the font atlas): ${[...missingGlyphs].join(" ")}`);
  }
  if (failed) Deno.exit(1);
}

/** gallery/index.json: every entry, in order, pretty-printed (deterministic). */
async function indexText(): Promise<string> {
  return JSON.stringify(await loadEntries(), null, 2) + "\n";
}

async function writeIndex(): Promise<void> {
  await Deno.writeTextFile(INDEX, await indexText());
  console.log(INDEX.pathname);
}

if (import.meta.main) {
  const [cmd, ...rest] = Deno.args;
  if (cmd === "render") await cmdRender(rest);
  else if (cmd === "check") await cmdCheck(rest);
  else if (cmd === "manifest") await writeIndex();
  else {
    console.error("usage: gallery.ts render [--section S] [--out DIR] [--sheets DIR] [--size WxH] [--cols N] [--thumb PX] | check [--section S] [--size WxH] [--update] | manifest");
    Deno.exit(2);
  }
  Deno.exit(0); // the spinner-less renderers keep no handles, but the GPU device does
}
