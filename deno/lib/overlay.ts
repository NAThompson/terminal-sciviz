/**
 * Everything of plot1d.ts that isn't a curve, in one instanced draw (one quad per item, blended
 * premultiplied `over` in order): rectangles (grid, swatches, rounded label box with border),
 * glyphs of the signed-distance-field font atlas (any size: the field sampled bilinearly, its edge
 * antialiased over the screen pixel `fwidth` gives), dots with a ring, and the curve
 * layers drawn by `NurbsGpuRenderer` (sampled 1:1, or upsampled bilinearly if smaller than the
 * frame, times an opacity, clipped to the
 * plot). Then `resolve` turns the premultiplied f16 frame into straight-alpha RGBA8.
 */

import { decodePng } from "./png.ts";
import { ROOT_N } from "./scales.ts";

const quadWgsl = (spread: number) => /* wgsl */ `
  struct Globals { size: vec2f, pad: vec2f, clip: vec4f } // clip: left, top, right, bottom (px)
  struct Quad {
    rect: vec4f, // x, y, w, h in pixels from the top left
    uv: vec4f, // glyph: atlas rectangle (texels); layer: index in .z
    color: vec4f, // premultiplied
    color2: vec4f, // premultiplied: border, ring
    params: vec4f, // kind, radius (or opacity of a layer), border/ring width, clipped to plot
  }
  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) index: u32,
    @location(1) texel: vec2f, // glyph: atlas texels
  }

  @group(0) @binding(0) var<uniform> g: Globals;
  @group(0) @binding(1) var<storage, read> quads: array<Quad>;
  @group(0) @binding(2) var atlas: texture_2d<f32>;
  @group(0) @binding(3) var layer0: texture_2d<f32>;
  @group(0) @binding(4) var layer1: texture_2d<f32>;
  @group(0) @binding(5) var layer2: texture_2d<f32>;
  @group(0) @binding(6) var linear: sampler;
  const SUBSAMPLE = ${SUBSAMPLE};
  const SPREAD = ${spread.toFixed(1)}; // atlas texels of distance either side of the outline

  // Coverage of a pixel \`d\` px inside the outline (negative: outside), \`k\` atlas texels per
  // px: a ramp one pixel wide, steeper for small text (\`glyphCoverage\` below)
  fn glyphCoverage(d: f32, k: f32) -> f32 {
    let sharp = 1.0 + ${SHARPEN} * clamp((k - 1.0) / 3.0, 0.0, 1.0);
    return clamp(sharp * d + 0.5, 0.0, 1.0);
  }

  // A layer's colour at pixel centre \`p\`: its texel, or bilinear if it is smaller than the frame
  fn layerColor(t: texture_2d<f32>, p: vec2f) -> vec4f {
    if (all(vec2f(textureDimensions(t)) == g.size)) {
      return textureLoad(t, vec2i(p), 0);
    }
    return textureSampleLevel(t, linear, p / g.size, 0.0);
  }

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Varyings {
    var corners = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
    let q = quads[i];
    let p = (q.rect.xy + corners[v] * q.rect.zw) / g.size;
    var out: Varyings;
    out.position = vec4f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    out.index = i;
    out.texel = q.uv.xy + corners[v] * q.uv.zw;
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    let q = quads[in.index];
    let p = in.position.xy; // pixel centre
    let texelsPerPx = fwidth(in.texel); // before any branch: screen-space derivatives
    if (q.params.w > 0.5 && (p.x < g.clip.x || p.x > g.clip.z || p.y < g.clip.y || p.y > g.clip.w)) {
      discard;
    }
    let kind = u32(q.params.x);
    switch (kind) {
      case 1u: { // glyph
        // minified (thin strokes thinner than the field's sampling): 2×2 samples per pixel
        let k = max(0.5 * (texelsPerPx.x + texelsPerPx.y), 1e-6);
        let n = select(1, 2, k > 2.0);
        var cover = 0.0;
        for (var j = 0; j < n; j++) {
          for (var i = 0; i < n; i++) {
            let o = select(vec2f(0.0), (vec2f(f32(i), f32(j)) - 0.5) * SUBSAMPLE * texelsPerPx, n == 2);
            let v = textureSampleLevel(atlas, linear, (in.texel + o) / vec2f(textureDimensions(atlas)), 0.0).r;
            cover += glyphCoverage((v * 255.0 - 128.0) / 127.0 * SPREAD / k, k); // px inside
          }
        }
        return q.color * (cover / f32(n * n));
      }
      case 2u: { // dot: fill inside a ring
        let d = length(p - (q.rect.xy + 0.5 * q.rect.zw));
        let outer = clamp(q.params.y + 0.5 - d, 0.0, 1.0);
        let inner = clamp(q.params.y - q.params.z + 0.5 - d, 0.0, 1.0);
        return q.color * inner + q.color2 * (outer - inner);
      }
      case 3u: { // layer
        var c: vec4f;
        switch (u32(q.uv.z)) {
          case 0u: { c = layerColor(layer0, p); }
          case 1u: { c = layerColor(layer1, p); }
          default: { c = layerColor(layer2, p); }
        }
        return c * q.params.y;
      }
      default: { // rectangle, rounded, with a border inside
        let half = 0.5 * q.rect.zw;
        let r = q.params.y;
        let e = abs(p - (q.rect.xy + half)) - half + r;
        let d = length(max(e, vec2f(0.0))) + min(max(e.x, e.y), 0.0) - r;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let inner = select(cover, clamp(0.5 - d - q.params.z, 0.0, 1.0), q.params.z > 0.0);
        return q.color * inner + q.color2 * (cover - inner);
      }
    }
  }
`;

const RESOLVE_WGSL = /* wgsl */ `
  @group(0) @binding(0) var frame: texture_2d<f32>;

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    var corners = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(corners[v], 0.0, 1.0);
  }

  @fragment
  fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let c = textureLoad(frame, vec2i(pos.xy), 0);
    return select(vec4f(min(c.rgb / c.a, vec3f(1.0)), c.a), vec4f(0.0), c.a <= 0.0);
  }
`;

export type Rgba = [number, number, number, number]; // premultiplied, 0..1

/** `#rrggbb` at `alpha`, premultiplied. */
export function premul(hex: string, alpha = 1): Rgba {
  const h = hex.replace(/^#/, "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  return [r * alpha, g * alpha, b * alpha, alpha];
}

// --- Font ------------------------------------------------------------------------------------

/** A glyph of the atlas at its base size: advance, and the field's rectangle in the atlas
 * (texels = base px) with its offset from the pen on the baseline. */
interface Glyph { advance: number; x?: number; y?: number; w?: number; h?: number; bx?: number; by?: number }
/** A face at `size` px: `scale` screen px per atlas texel; metrics in px, glyphs at the base size. */
export interface FontSize { size: number; scale: number; ascent: number; descent: number; glyphs: Record<string, Glyph> }
interface FaceFile { size: number; ascent: number; descent: number; glyphs: Record<string, Glyph>; small?: Record<string, Glyph> }
interface FontFile { width: number; height: number; sdf: { spread: number }; smallPx?: number; faces: Record<string, FaceFile> }
export type Face = "regular" | "bold";

/** The signed-distance-field font of tools/font_atlas.py: one base size per face, drawn at any. */
export class BitmapFont {
  private readonly sizes = new Map<string, FontSize>();

  private constructor(private readonly file: FontFile, readonly field: Uint8Array<ArrayBuffer>) {}

  static async load(dir: URL): Promise<BitmapFont> {
    const file = JSON.parse(await Deno.readTextFile(new URL("font.json", dir))) as FontFile;
    const { rgba } = await decodePng(await Deno.readFile(new URL("font.png", dir)));
    return new BitmapFont(file, Uint8Array.from({ length: rgba.length / 4 }, (_, i) => rgba[i * 4]));
  }

  get width(): number {
    return this.file.width;
  }

  get height(): number {
    return this.file.height;
  }

  /** Atlas texels of distance encoded either side of an outline. */
  get spread(): number {
    return this.file.sdf.spread;
  }

  /** `face` at exactly `px` pixels per em. */
  get(face: Face, px: number): FontSize {
    const key = `${face}${px}`;
    let font = this.sizes.get(key);
    if (!font) {
      const f = this.file.faces[face];
      const scale = px / f.size;
      // Below `smallPx`, subscripts and superscripts drawn larger, with open counters
      const small = f.small && px < (this.file.smallPx ?? 0);
      font = { size: px, scale, ascent: f.ascent * scale, descent: f.descent * scale, glyphs: small ? { ...f.glyphs, ...f.small } : f.glyphs };
      if (this.sizes.size > 256) this.sizes.clear();
      this.sizes.set(key, font);
    }
    return font;
  }

  measure(font: FontSize, text: string): number {
    let w = 0;
    for (const ch of text) {
      w += glyphOf(font, ch).advance;
    }
    return w * font.scale;
  }

  /** CPU rendering of `text` with its pen from (x, baseline), as the quad shader does it:
   * `plot(px, py, coverage)` for each pixel a glyph covers (0 < coverage <= 1, in order). */
  raster(font: FontSize, text: string, x: number, baseline: number, plot: (px: number, py: number, coverage: number) => void): void {
    const k = 1 / font.scale; // texels per px
    const W = this.width, H = this.height, field = this.field, spread = this.spread;
    const at = (u: number, v: number) => field[Math.min(Math.max(v, 0), H - 1) * W + Math.min(Math.max(u, 0), W - 1)];
    for (const p of glyphQuads(font, text)) {
      const [gx, gy, gw, gh] = [x + p.rect[0], baseline + p.rect[1], p.rect[2], p.rect[3]];
      for (let py = Math.floor(gy); py < gy + gh; py += 1) {
        for (let px = Math.floor(gx); px < gx + gw; px += 1) {
          const cu = p.uv[0] + (px + 0.5 - gx) * k - 0.5, cv = p.uv[1] + (py + 0.5 - gy) * k - 0.5;
          if (cu < p.uv[0] - 0.5 || cv < p.uv[1] - 0.5 || cu > p.uv[0] + p.uv[2] - 0.5 || cv > p.uv[1] + p.uv[3] - 0.5) continue;
          const n = k > 2 ? 2 : 1;
          let c = 0;
          for (let j = 0; j < n; j += 1) {
            for (let i = 0; i < n; i += 1) {
              const tu = n === 2 ? cu + (i - 0.5) * SUBSAMPLE * k : cu, tv = n === 2 ? cv + (j - 0.5) * SUBSAMPLE * k : cv;
              const u0 = Math.floor(tu), v0 = Math.floor(tv), fu = tu - u0, fv = tv - v0;
              const val = (1 - fv) * ((1 - fu) * at(u0, v0) + fu * at(u0 + 1, v0)) + fv * ((1 - fu) * at(u0, v0 + 1) + fu * at(u0 + 1, v0 + 1));
              c += glyphCoverage(((val - 128) / 127) * spread / k, k);
            }
          }
          c /= n * n;
          if (c > 0) plot(px, py, c);
        }
      }
    }
  }
}

/** Steepening of the edge ramp of small text: hinted bitmaps (the atlas this replaced) put most
 * ink in full pixels, so an unhinted field's 1 px ramp reads blurrier below ~16 px. Up to
 * 1.5× steeper as the atlas is minified 4× (12 px from 48), same weight. */
/** Pitch (px) of the 2×2 samples of minified glyphs. */
const SUBSAMPLE = 0.3;
const SHARPEN = 1.0;

/** As the shader's `glyphCoverage`. */
function glyphCoverage(d: number, k: number): number {
  const sharp = 1 + SHARPEN * Math.min(Math.max((k - 1) / 3, 0), 1);
  return Math.min(Math.max(sharp * d + 0.5, 0), 1);
}

/** Characters not in the atlas, drawn as "?" (each logged once): the gallery fails on any. */
export const missingGlyphs = new Set<string>();

function glyphOf(font: FontSize, ch: string): Glyph {
  const g = font.glyphs[ch];
  if (g) return g;
  if (!missingGlyphs.has(ch)) {
    missingGlyphs.add(ch);
    console.error(`overlay: no glyph for ${JSON.stringify(ch)} (U+${ch.codePointAt(0)!.toString(16).toUpperCase()}), drawn as "?"`);
  }
  return font.glyphs["?"];
}

/** The quads of `text`'s glyphs with the pen at 0 on the baseline: screen rectangle (px) and atlas rectangle (texels). */
function glyphQuads(font: FontSize, text: string): { rect: number[]; uv: number[] }[] {
  const out: { rect: number[]; uv: number[] }[] = [];
  const k = font.scale;
  let pen = 0;
  for (const ch of text) {
    const g = glyphOf(font, ch);
    if (g.w) {
      const x = Math.round(pen); // whole pixels: crisp stems, every instance of a glyph alike
      out.push({ rect: [x + g.bx! * k, g.by! * k, g.w * k, g.h! * k], uv: [g.x!, g.y!, g.w, g.h!] });
    }
    pen += g.advance * k;
  }
  return out;
}

// --- Quads -----------------------------------------------------------------------------------

const QUAD_FLOATS = 20;
export type Anchor = `${"l" | "m" | "r"}${"t" | "m" | "s" | "b"}`;

export class Quads {
  private data = new Float32Array(QUAD_FLOATS * 64); // grown by doubling; `used` floats
  private used = 0;

  constructor(private readonly font: BitmapFont) {}

  get count(): number {
    return this.used / QUAD_FLOATS;
  }

  private reserve(floats: number): void {
    if (this.used + floats > this.data.length) {
      let size = this.data.length * 2;
      while (size < this.used + floats) size *= 2;
      const grown = new Float32Array(size);
      grown.set(this.data.subarray(0, this.used));
      this.data = grown;
    }
  }

  append(other: Quads): void {
    this.reserve(other.used);
    this.data.set(other.data.subarray(0, other.used), this.used);
    this.used += other.used;
  }

  /** The quads (a view of the buffer, valid until the next change). */
  array(): Float32Array<ArrayBuffer> {
    return this.data.subarray(0, this.used) as Float32Array<ArrayBuffer>;
  }

  private push(rect: number[], uv: number[], color: Rgba, color2: Rgba, params: number[]): void {
    this.reserve(QUAD_FLOATS);
    const d = this.data;
    let o = this.used;
    for (const list of [rect, uv, color, color2, params]) {
      for (let i = 0; i < 4; i += 1) d[o++] = list[i];
    }
    this.used = o;
  }

  rect(x: number, y: number, w: number, h: number, color: Rgba, opts: { radius?: number; border?: number; borderColor?: Rgba; clip?: boolean } = {}): void {
    this.push([x, y, w, h], [0, 0, 0, 0], color, opts.borderColor ?? [0, 0, 0, 0], [0, opts.radius ?? 0, opts.border ?? 0, opts.clip ? 1 : 0]);
  }

  /** Dot of outer radius `radius` around (cx, cy), `ring` wide. */
  dot(cx: number, cy: number, radius: number, ring: number, fill: Rgba, ringColor: Rgba, clip = false): void {
    const r = radius + 1;
    this.push([cx - r, cy - r, 2 * r, 2 * r], [0, 0, 0, 0], fill, ringColor, [2, radius, ring, clip ? 1 : 0]);
  }

  /** Layer `index` over the plot rectangle (clipped), times `opacity`. */
  layer(index: number, opacity: number, rect: [number, number, number, number]): void {
    const [l, t, r, b] = rect;
    this.push([l, t, r - l, b - t], [0, 0, index, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, opacity, 0, 1]);
  }

  /** Text from `anchor` at (x, y), its pen and baseline at whole pixels: horizontal left/middle/right, vertical top
   * (ascender), middle (between ascender and descender), baseline or bottom (descender).
   * Returns its width. */
  text(text: string, x: number, y: number, font: FontSize, color: Rgba, anchor: Anchor = "ls"): number {
    const width = this.font.measure(font, text);
    const pen = Math.round(anchor[0] === "l" ? x : anchor[0] === "m" ? x - width / 2 : x - width);
    const v = anchor[1];
    const base = Math.round(
      v === "t" ? y + font.ascent : v === "m" ? y + (font.ascent - font.descent) / 2 : v === "b" ? y - font.descent : y,
    );
    for (const g of glyphQuads(font, text)) {
      this.push([pen + g.rect[0], base + g.rect[1], g.rect[2], g.rect[3]], g.uv, color, [0, 0, 0, 0], [1, 0, 0, 0]);
    }
    return width;
  }
}

// --- Root scale slider -----------------------------------------------------------------------

/** state.json's `rootSlider`: the track's hit area (frame px, generous vertically), its range and
 * value; x maps linearly from `rect[0]` (min) to `rect[2]` (max). */
export interface RootSlider { rect: [number, number, number, number]; min: number; max: number; step: number; value: number }

/** Height (px) `rootSlider` draws at scale `s`. */
export const rootSliderHeight = (font: FontSize, s: number) => Math.ceil(font.ascent + font.descent) + Math.round(14 * s);

/** Slider of the root colour scale's `n` (in the theme's ink: track #8b949e, thumb #c9d1d9): its
 * label `ⁿ√  n = 3.00` from (x, top), the track `width` px long below it. */
export function rootSlider(q: Quads, font: FontSize, x: number, top: number, width: number, n: number, s: number): RootSlider {
  const { min, max, step } = ROOT_N;
  x = Math.round(x);
  q.text(`ⁿ√  n = ${n.toFixed(2)}`, x, top, font, premul("#c9d1d9"), "lt");
  const y = Math.round(top + font.ascent + font.descent + 8 * s); // track centre
  const th = Math.max(Math.round(2 * s), 2);
  q.rect(x, y - th / 2, width, th, premul("#8b949e"), { radius: th / 2 });
  q.dot(x + ((n - min) / (max - min)) * width, y, 5 * s, 1.5 * s, premul("#c9d1d9"), premul("#101216"));
  const reach = Math.round(12 * s);
  return { rect: [x, y - reach, x + Math.round(width), y + reach], min, max, step, value: n };
}

// --- Pipelines -------------------------------------------------------------------------------

const RENDER_ATTACHMENT = 0x10;
const TEXTURE_BINDING = 0x04;
const COPY_DST = 0x02;
const COPY_SRC = 0x01;
export const FRAME_FORMAT = "rgba16float"; // no 8-bit rounding of faint edges before un-premultiplying

export class OverlayRenderer {
  private readonly quadPipeline: GPURenderPipeline;
  private readonly resolvePipeline: GPURenderPipeline;
  private readonly globals: GPUBuffer;
  private quads?: GPUBuffer;
  private readonly atlas: GPUTexture;
  private readonly sampler: GPUSampler;
  private readonly blank: GPUTexture;
  frame?: GPUTexture; // premultiplied
  out?: GPUTexture; // straight alpha RGBA8
  private resolveGroup?: GPUBindGroup;
  private group?: { key: unknown[]; group: GPUBindGroup };

  constructor(private readonly device: GPUDevice, font: BitmapFont) {
    const over: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    const quad = device.createShaderModule({ code: quadWgsl(font.spread) });
    this.quadPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: quad },
      fragment: { module: quad, targets: [{ format: FRAME_FORMAT, blend: over }] },
    });
    const resolve = device.createShaderModule({ code: RESOLVE_WGSL });
    this.resolvePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: resolve },
      fragment: { module: resolve, targets: [{ format: "rgba8unorm" }] },
    });
    this.globals = device.createBuffer({ size: 32, usage: 0x40 | 0x08 });
    this.atlas = device.createTexture({ size: [font.width, font.height], format: "r8unorm", usage: TEXTURE_BINDING | COPY_DST });
    device.queue.writeTexture({ texture: this.atlas }, font.field, { bytesPerRow: font.width }, [font.width, font.height]);
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.blank = device.createTexture({ size: [1, 1], format: FRAME_FORMAT, usage: TEXTURE_BINDING });
  }

  setSize(w: number, h: number): void {
    if (this.frame?.width === w && this.frame.height === h) {
      return;
    }
    this.frame?.destroy();
    this.out?.destroy();
    this.frame = this.device.createTexture({ size: [w, h], format: FRAME_FORMAT, usage: RENDER_ATTACHMENT | TEXTURE_BINDING });
    this.out = this.device.createTexture({ size: [w, h], format: "rgba8unorm", usage: RENDER_ATTACHMENT | COPY_SRC });
    this.resolveGroup = this.device.createBindGroup({
      layout: this.resolvePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: this.frame.createView() }],
    });
  }

  /** Draws `quads` into the frame, then resolves it into `out` (in `encoder`). */
  encode(encoder: GPUCommandEncoder, quads: Quads, layers: (GPUTexture | undefined)[], clip: number[]): void {
    const { device } = this;
    const w = this.frame!.width;
    const h = this.frame!.height;
    device.queue.writeBuffer(this.globals, 0, Float32Array.from([w, h, 0, 0, ...clip]));
    const data = quads.array();
    if (!this.quads || this.quads.size < data.byteLength) {
      this.quads?.destroy();
      this.quads = device.createBuffer({ size: Math.max(2 ** Math.ceil(Math.log2(data.byteLength || 16)), 1024), usage: 0x80 | 0x08 });
      this.group = undefined;
    }
    device.queue.writeBuffer(this.quads, 0, data);
    const textures = [0, 1, 2].map((i) => layers[i] ?? this.blank);
    const key = [this.quads, ...textures];
    if (!this.group || this.group.key.some((k, i) => k !== key[i])) {
      this.group = {
        key,
        group: device.createBindGroup({
          layout: this.quadPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.globals } },
            { binding: 1, resource: { buffer: this.quads } },
            { binding: 2, resource: this.atlas.createView() },
            ...textures.map((t, i) => ({ binding: 3 + i, resource: t.createView() })),
            { binding: 6, resource: this.sampler },
          ],
        }),
      };
    }

    let pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.frame!.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
    });
    if (quads.count > 0) {
      pass.setPipeline(this.quadPipeline);
      pass.setBindGroup(0, this.group.group);
      pass.draw(6, quads.count);
    }
    pass.end();

    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.out!.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.resolvePipeline);
    pass.setBindGroup(0, this.resolveGroup!);
    pass.draw(3);
    pass.end();
  }
}
