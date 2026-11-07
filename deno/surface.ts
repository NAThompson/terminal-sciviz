/**
 * Headless renderer of NURBS surfaces (`NurbsSurfaceGpuRenderer` of lib/gpu_surface.ts), in Deno.
 * CLI, control file and serve protocol (shared-memory frames) of lib/serve.ts, with a
 * state.json per frame.
 *
 *   deno run -A surface.ts png CONTROL.json -o out.png
 *   deno run -A surface.ts serve CONTROL.json OUTDIR [--ignore-stdin] [--spin 0.5] [--inflight 3]
 *
 * The renderer evaluates (compute shader), colours (`getLinearColorLut`, `getColorByDomain` on
 * values read back with `readValues`) and draws; here: the sample counts (`resolution` steps per
 * direction, shared among knot spans), framing, the camera, readback, and hover picking: a
 * second fragment entry point appended to the draw shader. `scale: "root"` colours by t^(1/rootN)
 * of the normalised value t (heights unchanged), with a colour bar and a slider for n drawn over
 * the frame by lib/overlay.ts.
 */

import { cameraBasis, fitAngles, fitView, screenRect, type Vec3 } from "./lib/framing.ts";
import { lookAtView, mul4, perspectiveProjection } from "./lib/mat4.ts";
import { getLinearColorLut, colormapName, colormapRgb } from "./lib/colormaps.ts";
import { colorBarTicks, createScale, getShaderScale, snapRootN, type VisScale } from "./lib/scales.ts";
import { BitmapFont, OverlayRenderer, premul, Quads, type Rgba, rootSlider, type RootSlider, rootSliderHeight } from "./lib/overlay.ts";
import {
  evaluateNurbsSurface,
  evaluateSurfaceAnalysis,
  getColorByDomain,
  getKnotSpans,
  getKnotVectors,
  getNurbsSurfaceError,
  type KnotSpan,
  NurbsColorBy,
  type NurbsSurface,
} from "./lib/nurbs.ts";
import { DRAW_WGSL, NurbsSurfaceGpuRenderer, type SurfaceTransform } from "./lib/gpu_surface.ts";
import {
  COPY_SRC as COPY_SRC_BUFFER,
  createBuffer,
  makeReadBuffer,
  readBuffer,
  requestGpuDevice,
  STORAGE,
  TextureTarget,
  UNIFORM,
} from "./lib/gpu.ts";
import { main, type Pending, type Renderer as ServeRenderer } from "./lib/serve.ts";
import { SURFACES } from "./lib/surfaces.ts";
import { complexGridSurface } from "./complex.ts";

/** Named surfaces, plus grids of complex functions as degree-1 B-splines (complex.ts): by name
 * (`"complex-cubic"`: |z³ − 1| as a height field, coloured by phase) or `{complex: "z^3-1", n, domain,
 * zStretch, zScale}`; anything else is an inline `NurbsSurface`. */
function namedSurface(spec: unknown): NurbsSurface | undefined {
  if (typeof spec === "string") {
    return spec === "complex-cubic" ? complexGridSurface({ complex: "z^3-1" }) : SURFACES[spec]?.();
  }
  return spec && typeof spec === "object" && "complex" in spec ? complexGridSurface(spec as { complex: string }) : spec as NurbsSurface;
}

// Appended to the draw shader: knot coordinates (span + fraction), value and world position
// under a pixel, written to two colour targets
const PICK_WGSL = /* wgsl */ `
  struct Pick { @location(0) knotValue: vec4f, @location(1) world: vec4f }

  @fragment
  fn pickMain(in: Varyings) -> Pick {
    return Pick(vec4f(in.knot, in.value), vec4f(in.world, 1.0));
  }
`;

// Lights and camera
const AMBIENT_SHARE = 0.55;
const LIGHT_POSITION: Vec3 = (([x, y, z]) => {
  const k = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
  return [x * k, y * k, z * k] as Vec3;
})([-0.2, -0.3, 1.0]);
const LIGHT: [number, number, number, number] = [...LIGHT_POSITION, AMBIENT_SHARE];
const KNOT_LINE_WIDTH = 1.5;
const FOV = 40;
const FIT_MARGIN = 1.08;
const MAX_GPU_DEGREE = 7;
const DEPTH_FORMAT = "depth24plus";
const VERTEX_FLOATS = 12;

const RENDER_ATTACHMENT = 0x10;
const COPY_SRC = 0x01;
const INK = "#c9d1d9", INK2 = "#8b949e";
const LABEL_FILL: Rgba = [20 / 255 * (240 / 255), 24 / 255 * (240 / 255), 30 / 255 * (240 / 255), 240 / 255];
const LABEL_BORDER: Rgba = [48 / 255, 54 / 255, 61 / 255, 1];

export const DEFAULTS = {
  width: 960,
  height: 600,
  azimuth: 35.0,
  elevation: 25.0,
  distance: null as number | null, // null: fit the evaluated surface, centred (framing.ts)
  target: null as Vec3 | null,
  zoom: 1.0, // with distance null: fitted distance / zoom
  fov: undefined as number | undefined, // of the projection, degrees (default FOV, which framing also uses)
  snapshot: null as { path: string; rev: number } | string | null, // serve: PNG of a frame (lib/serve.ts)
  colormap: "viridis",
  invertColorMap: false,
  background: "transparent" as string | number[], // or "#rrggbb", [r, g, b] or [r, g, b, a] (sRGB 0..1)
  resolution: 128,
  hover: null as [number, number] | null,
  frameFile: false, // serve: also write OUTDIR/frame.rgba
  quit: false,
  surface: "wave" as string | NurbsSurface, // wave | torus | sphere | wave-test | complex-cubic, {complex}, or inline
  colorBy: "field",
  field: 0,
  domain: null as [number, number] | null, // of the colour map; default: `getColorByDomain`
  scale: "linear" as "linear" | "root", // root: colours by t^(1/rootN) of the normalised value t (not cyclic), heights unchanged
  rootN: 3, // 0.1..10, step 0.05
  knotLines: false,
  /* Colour domain is a period: values wrap around the colour map instead of being clamped. Give a
   * full period as `domain` (e.g. [-π, π] of a phase) and a cyclic map (e.g. Twilight; others show
   * a hard seam). Seamless where neighbouring samples differ by less than half a period; inside
   * subdivided knot spans a short sweep can remain along a branch cut */
  cyclic: false,
};
export type Control = typeof DEFAULTS & Record<string, unknown>;

/** `"transparent"`, `"#rrggbb"`, or sRGB 0..1 `[r, g, b]` / `[r, g, b, a]`. */
function parseBackground(bg: unknown): [number, number, number, number] {
  if (bg === null || bg === undefined || bg === "transparent") {
    return [0, 0, 0, 0];
  }
  if (typeof bg === "string") {
    const h = bg.replace(/^#/, "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).concat(1) as [number, number, number, number];
  }
  const c = (bg as number[]).map(Number);
  return [c[0], c[1], c[2], c.length > 3 ? c[3] : 1];
}

/** Steps per knot span: `resolution` steps along each direction, shared by span length, with
 * round-half-to-even (as numpy rounds). */
function stepCounts(spans: KnotSpan[], resolution: number): Uint32Array {
  const lengths = spans.map(({ start, end }) => end - start);
  const sum = lengths.reduce((a, b) => a + b, 0);
  return Uint32Array.from(lengths, (len) => {
    const x = (resolution * len) / sum;
    const r = Math.round(x);
    const even = Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
    return Math.max(1, even);
  });
}

/** Straight alpha from premultiplied 8-bit values: `rgb * float32(255 / max(a, 1)) + 0.5`,
 * truncated, all in f32 (bit-exact with the same formula in numpy): a 64 KiB table, looked up
 * on the GPU by `STRAIGHT_WGSL`. */
const UNPREMULTIPLY = (() => {
  const f = Math.fround;
  const table = new Uint8Array(256 * 256);
  for (let a = 1; a < 256; a += 1) {
    const k = f(255 / a);
    for (let c = 0; c < 256; c += 1) {
      table[a * 256 + c] = Math.min(Math.max(Math.trunc(f(f(c * k) + 0.5)), 0), 255);
    }
  }
  return table;
})();

/* Readback glue: the drawn canvas texture (premultiplied) to tightly packed straight-alpha RGBA8
 * (`UNPREMULTIPLY`), or over an opaque background, so the CPU only copies the frame. */
const STRAIGHT_WGSL = /* wgsl */ `
  struct Params { size: vec2u, overBackground: u32, pad: u32, background: vec4f } // 0..255
  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var frame: texture_2d<f32>;
  @group(0) @binding(2) var<storage, read> table: array<u32>; // bytes, a × 256 + c
  @group(0) @binding(3) var<storage, read_write> pixels: array<u32>;

  fn straight(a: u32, c: u32) -> u32 {
    let i = a * 256u + c;
    return (table[i / 4u] >> ((i % 4u) * 8u)) & 255u;
  }

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.size.x || id.y >= params.size.y) {
      return;
    }
    let p = vec4u(round(textureLoad(frame, id.xy, 0) * 255.0)); // RGBA whatever the format
    var c = vec4u(straight(p.a, p.r), straight(p.a, p.g), straight(p.a, p.b), p.a);
    if (params.overBackground == 1u) {
      c = vec4u(floor(vec4f(p) + (1.0 - f32(p.a) / 255.0) * params.background + 0.5));
    }
    pixels[id.y * params.size.x + id.x] = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
  }
`;

export class SurfaceRenderer implements ServeRenderer<Control> {
  private gpu?: NurbsSurfaceGpuRenderer;
  private readonly canvas: TextureTarget;
  private pickPipeline?: GPURenderPipeline;
  private pickGroup?: { vertices: GPUBuffer; group: GPUBindGroup };
  private readonly pickReads: GPUBuffer[] = []; // per readback slot (frames in flight)
  private pickTargets?: GPUTexture[];
  private readonly frameReads: GPUBuffer[] = [];
  private pixels?: GPUBuffer;
  private readonly straightPipeline: GPUComputePipeline;
  private readonly straightParams: GPUBuffer;
  private readonly straightTable: GPUBuffer;
  private straightGroup?: { texture: GPUTexture; group: GPUBindGroup };
  private surfaceSpec?: string;
  private surfaceKey?: string;
  private lutKey?: string;
  private fitKey?: string;
  private fit: [Vec3, number] = [[0, 0, 0], 3];

  surface!: NurbsSurface;
  private transform!: SurfaceTransform;
  private spans!: [KnotSpan[], KnotSpan[]];
  colorBy!: NurbsColorBy;
  private field = 0;
  grid: [number, number] = [0, 0];
  dataDomain: [number, number] = [-0.5, 0.5];
  private fitPoints = new Float64Array(0); // valid evaluated vertices (world xyz), grown by FIT_MARGIN
  private overlay?: { font: BitmapFont; renderer: OverlayRenderer }; // root scale only
  private slider: RootSlider | null = null; // root scale's, as drawn

  constructor(private readonly device: GPUDevice) {
    this.canvas = new TextureTarget(device, "bgra8unorm");
    this.straightPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: STRAIGHT_WGSL }) },
    });
    this.straightParams = createBuffer(device, 32, UNIFORM);
    this.straightTable = createBuffer(device, new Uint32Array(UNPREMULTIPLY.buffer), STORAGE);
  }

  static async create(): Promise<SurfaceRenderer> {
    const device = await requestGpuDevice(); // the shared device
    if (!device) {
      throw new Error("No WebGPU device");
    }
    return new SurfaceRenderer(device);
  }

  private get internals(): NurbsSurfaceGpuRenderer {
    return this.gpu!;
  }

  // Surface: renderer of the patch, and its evaluation (only when these change) ---------

  async setSurface(ctl: Control): Promise<boolean> {
    const key = JSON.stringify([ctl.surface, ctl.colorBy, ctl.field, ctl.resolution]);
    if (key === this.surfaceKey) {
      return false;
    }
    const spec = JSON.stringify(ctl.surface);
    if (!this.gpu || spec !== this.surfaceSpec) {
      this.upload(namedSurface(ctl.surface), ctl.surface);
      this.surfaceSpec = spec;
    }
    await this.evaluate(ctl);
    this.surfaceKey = key;
    return true;
  }

  private upload(surface: NurbsSurface | undefined, name: unknown): void {
    if (!surface) {
      throw new Error(`surface must be one of ${Object.keys(SURFACES)},complex-cubic or a NurbsSurface (got ${name})`);
    }
    const error = getNurbsSurfaceError(surface);
    if (error) {
      throw new Error(error);
    }
    if (Math.max(...surface.degrees) > MAX_GPU_DEGREE) {
      throw new Error(`Degree above the maximum of ${MAX_GPU_DEGREE} for WebGPU rendering`);
    }

    // World box: the control points bound the patch, centred on the origin
    const p = surface.points;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]; // no spread: large grids
    for (let i = 0; i < p.length; i += 1) {
      lo[i % 3] = Math.min(lo[i % 3], p[i]);
      hi[i % 3] = Math.max(hi[i % 3], p[i]);
    }
    const extents = lo.map((l, d) => hi[d] - l);
    const longest = Math.max(...extents) || 1;
    this.transform = {
      origin: lo.map((l, d) => (l + hi[d]) / 2) as Vec3,
      scale: extents.map((e) => (e > 0 ? e / longest / e : 0)) as Vec3,
      offset: [0, 0, 0],
    };

    const vectors = getKnotVectors(surface);
    this.spans = [getKnotSpans(vectors[0]), getKnotSpans(vectors[1])];
    this.gpu?.dispose();
    this.gpu = new NurbsSurfaceGpuRenderer(this.device, this.canvas, [surface], [this.spans], this.transform);
    this.surface = surface;
    this.lutKey = undefined;
    this.pickGroup = undefined;

    const module = this.device.createShaderModule({ code: DRAW_WGSL + PICK_WGSL });
    this.pickPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vertexMain" },
      fragment: { module, entryPoint: "pickMain", targets: [{ format: "rgba32float" }, { format: "rgba32float" }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  /** `setColorBy` + `setSampleCounts`, then the evaluated vertices (framing) and values
   * (`readValues`, for the colour domain) read back. */
  private async evaluate(ctl: Control): Promise<void> {
    const gpu = this.gpu!;
    const modes = Object.values(NurbsColorBy) as string[];
    if (!modes.includes(ctl.colorBy)) {
      throw new Error(`colorBy must be one of ${modes}`);
    }
    const hasFields = (this.surface.fields?.length ?? 0) > 0;
    let colorBy = ctl.colorBy as NurbsColorBy;
    if (colorBy.startsWith("field") && !hasFields) {
      colorBy = NurbsColorBy.Height; // no fields to colour by
    }
    this.colorBy = colorBy;
    this.field = hasFields ? Math.trunc(Number(ctl.field)) : 0;
    const res = Math.max(1, Math.trunc(Number(ctl.resolution)));
    const counts: [Uint32Array, Uint32Array] = [stepCounts(this.spans[0], res), stepCounts(this.spans[1], res)];
    this.grid = [counts[0].reduce((a, n) => a + n, 1), counts[1].reduce((a, n) => a + n, 1)];
    const total = this.grid[0] * this.grid[1];

    gpu.setColorBy(colorBy, this.field);
    gpu.setSampleCounts([counts]);
    const [vertexData, { values, areas }] = await Promise.all([
      readBuffer(this.device, this.internals.patches[0].vertices!, total * VERTEX_FLOATS * 4),
      gpu.readValues([counts]),
    ]);
    const vertices = new Float32Array(vertexData);
    const positions: number[] = [];
    for (let i = 0; i < total; i += 1) {
      if (vertices[i * VERTEX_FLOATS + 9] > 0.5) {
        positions.push(vertices[i * VERTEX_FLOATS], vertices[i * VERTEX_FLOATS + 1], vertices[i * VERTEX_FLOATS + 2]);
      }
    }
    // Grown about their centre by the framing margin (the fit only depends on angles then)
    const n = positions.length / 3;
    const centre = [0, 1, 2].map((d) => {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i += 1) {
        lo = Math.min(lo, positions[i * 3 + d]);
        hi = Math.max(hi, positions[i * 3 + d]);
      }
      return n ? (lo + hi) / 2 : 0;
    });
    this.fitPoints = Float64Array.from(positions, (v, i) => centre[i % 3] + (v - centre[i % 3]) * FIT_MARGIN);
    this.dataDomain = getColorByDomain(values, colorBy, areas);
  }

  // Camera ------------------------------------------------------------------------------------

  /** Missing `distance`/`target` from fitting the evaluated surface in the view, centred. */
  async resolveCamera(ctl: Control): Promise<Control> {
    if (ctl.distance !== null && ctl.target !== null) {
      return ctl;
    }
    await this.setSurface(ctl);
    const aspect = ctl.width / ctl.height;
    const key = JSON.stringify([this.surfaceKey, ...fitAngles(ctl), aspect]);
    if (key !== this.fitKey) {
      this.fit = fitView(this.fitPoints, ...fitAngles(ctl), FOV, aspect) ?? [[0, 0, 0], 3]; // no valid vertex
      this.fitKey = key;
    }
    const [target, distance] = this.fit;
    return {
      ...ctl,
      distance: ctl.distance ?? distance / (Number(ctl.zoom) || 1),
      target: ctl.target ?? [...target],
    };
  }

  /** Perspective camera with z up (view-projection, eye). */
  static camera(ctl: Control): { viewProjection: number[]; eye: Vec3 } {
    const [f] = cameraBasis(ctl.azimuth, ctl.elevation);
    const dist = Number(ctl.distance);
    const target = ctl.target as Vec3;
    const eye: Vec3 = [0, 1, 2].map((k) => target[k] + dist * f[k]) as Vec3;
    const projection = perspectiveProjection(ctl.fov ?? FOV, ctl.width / ctl.height, dist / 100, dist * 100);
    return { viewProjection: mul4(projection, lookAtView(eye, target, [0, 0, 1])), eye };
  }

  // Frames ------------------------------------------------------------------------------------

  private setTargets(w: number, h: number): void {
    if (this.canvas.width === w && this.canvas.height === h && this.pixels) {
      return;
    }
    this.canvas.width = w;
    this.canvas.height = h;
    this.pixels?.destroy();
    this.pixels = createBuffer(this.device, w * h * 4, STORAGE | COPY_SRC_BUFFER);
    this.pickTargets?.forEach((t) => t.destroy());
    const make = (format: GPUTextureFormat, usage: number) => this.device.createTexture({ size: [w, h], format, usage });
    this.pickTargets = [
      make("rgba32float", RENDER_ATTACHMENT | COPY_SRC),
      make("rgba32float", RENDER_ATTACHMENT | COPY_SRC),
      make(DEPTH_FORMAT, RENDER_ATTACHMENT),
    ];
  }

  /** Submits the frame, its readback into `slot` and their `mapAsync` (not awaited). */
  async submit(input: Control, slot: number): Promise<Pending> {
    const { device } = this;
    await this.setSurface(input);
    const gpu = this.gpu!;
    const lutKey = JSON.stringify([String(input.colormap).toLowerCase(), Boolean(input.invertColorMap)]);
    if (lutKey !== this.lutKey) { // `getLinearColorLut` + `writeColorLut`
      const name = colormapName(String(input.colormap));
      if (!name) {
        throw new Error(`Unknown colormap ${input.colormap}`);
      }
      gpu.setColorLut(getLinearColorLut(name, Boolean(input.invertColorMap)));
      this.lutKey = lutKey;
    }

    const w = Math.trunc(input.width);
    const h = Math.trunc(input.height);
    this.setTargets(w, h);
    const ctl = await this.resolveCamera(input);
    const { viewProjection, eye } = SurfaceRenderer.camera(ctl);
    const vis: VisScale | null = ctl.scale === "root" && !ctl.cyclic ? ["root", snapRootN(ctl.rootN)] : null;
    if (vis && !this.overlay) {
      const font = await BitmapFont.load(new URL("./assets/", import.meta.url));
      this.overlay = { font, renderer: new OverlayRenderer(device, font) };
    }
    gpu.draw({ // writes the view and submits the draw right away
      viewProjection,
      camera: eye,
      light: LIGHT,
      valueRange: ctl.domain ?? this.dataDomain,
      knotLines: ctl.knotLines ? KNOT_LINE_WIDTH : 0,
      cyclic: Boolean(ctl.cyclic),
      colorExponent: vis ? getShaderScale(vis)[1] : 0,
      size: [w, h],
    });

    if (this.frameReads[slot]?.size !== w * h * 4) { // free: its frame was read
      this.frameReads[slot]?.destroy();
      this.frameReads[slot] = makeReadBuffer(device, w * h * 4);
    }
    const frameRead = this.frameReads[slot];
    const pickRead = (this.pickReads[slot] ??= makeReadBuffer(device, 512));
    const encoder = device.createCommandEncoder();
    let frame = this.canvas.currentTexture();
    this.slider = null;
    if (vis) { // the colour bar and slider over the frame
      const { renderer } = this.overlay!;
      renderer.setSize(w, h);
      renderer.encode(encoder, this.rootLegend(ctl, vis, w, h), [frame], [0, 0, w, h]);
      frame = renderer.frame!;
    }
    this.encodeStraight(encoder, frame, w, h, parseBackground(ctl.background), frameRead);
    let pixel: [number, number] | null = null;
    if (ctl.hover) {
      const [px, py] = [Math.floor(ctl.hover[0]), Math.floor(ctl.hover[1])];
      if (px >= 0 && px < w && py >= 0 && py < h) {
        pixel = [px, py];
        this.encodePick(encoder, px, py, pickRead);
      }
    }
    device.queue.submit([encoder.finish()]);

    const reads = [frameRead.mapAsync(0x01)];
    if (pixel) {
      reads.push(pickRead.mapAsync(0x01));
    }
    const pick = this.pickContext(ctl.cyclic ? ctl.domain ?? this.dataDomain : null); // of this frame's surface, whatever is submitted next
    return {
      done: Promise.all(reads),
      read: (alloc) => {
        const rgba = alloc(w * h * 4);
        rgba.set(new Uint8Array(frameRead.getMappedRange()));
        frameRead.unmap();
        const hover = pixel ? this.readPick(pixel, pickRead, pick) : null;
        return { width: w, height: h, rgba, hover };
      },
    };
  }

  /** Root scale: a horizontal colour bar (its ticks by the scale, labelled with values) and the
   * slider for n under it, on a panel at the bottom right. */
  private rootLegend(ctl: Control, vis: VisScale, w: number, h: number): Quads {
    const { font } = this.overlay!;
    const s = Math.min(Math.max(Math.min(w / 1200, h / 750), 0.6), 2.5);
    const label = font.get("regular", 13 * s);
    const [pad, bw, bh] = [Math.round(12 * s), Math.round(240 * s), Math.round(10 * s)];
    const lh = Math.ceil(label.ascent + label.descent);
    const panelH = pad + lh + Math.round(6 * s) + bh + Math.round(6 * s) + lh + Math.round(12 * s) + rootSliderHeight(label, s) + pad;
    const [px, py] = [w - Math.round(16 * s) - bw - 2 * pad, h - Math.round(16 * s) - panelH];
    const q = new Quads(font);
    q.layer(0, 1, [0, 0, w, h]); // the surface
    q.rect(px, py, bw + 2 * pad, panelH, LABEL_FILL, { radius: Math.round(5 * s), border: 1, borderColor: LABEL_BORDER });
    const [x, top] = [px + pad, py + pad + lh + Math.round(6 * s)];
    q.text(this.colorBy, x, py + pad, label, premul(INK), "lt");
    const name = colormapName(String(ctl.colormap)) ?? "Viridis";
    for (let i = 0; i < bw; i += 1) {
      const [r, g, b] = colormapRgb(name, (i + 0.5) / bw, Boolean(ctl.invertColorMap));
      q.rect(x + i, top, 1, bh, [r / 255, g / 255, b / 255, 1]);
    }
    q.rect(x, top, bw, bh, [0, 0, 0, 0], { border: 1, borderColor: LABEL_BORDER });
    const domain = ctl.domain ?? this.dataDomain;
    const at = createScale(vis, { domain, range: [x, x + bw] });
    for (const t of colorBarTicks(vis, domain, bw / s)) {
      const X = Math.round(at(t.value));
      q.rect(X, top + bh, 1, Math.round(4 * s), premul(INK2));
      q.text(t.label, X + 0.5, top + bh + Math.round(6 * s), label, premul(INK2), "mt");
    }
    const sliderTop = top + bh + Math.round(6 * s) + lh + Math.round(12 * s);
    this.slider = rootSlider(q, label, x, sliderTop, bw, (vis as ["root", number])[1], s);
    return q;
  }

  /** The drawn frame, straight alpha (or over the background), into `frameRead`. */
  private encodeStraight(encoder: GPUCommandEncoder, texture: GPUTexture, w: number, h: number, bg: number[], frameRead: GPUBuffer): void {
    if (this.straightGroup?.texture !== texture) {
      this.straightGroup = {
        texture,
        group: this.device.createBindGroup({
          layout: this.straightPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.straightParams } },
            { binding: 1, resource: texture.createView() },
            { binding: 2, resource: { buffer: this.straightTable } },
            { binding: 3, resource: { buffer: this.pixels! } },
          ],
        }),
      };
    }
    const params = new ArrayBuffer(32);
    new Uint32Array(params).set([w, h, bg[3] === 0 ? 0 : 1, 0]);
    new Float32Array(params).set(bg.map((c) => c * 255), 4);
    this.device.queue.writeBuffer(this.straightParams, 0, params);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.straightPipeline);
    pass.setBindGroup(0, this.straightGroup.group);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    encoder.copyBufferToBuffer(this.pixels!, 0, frameRead, 0, w * h * 4);
  }

  private encodePick(encoder: GPUCommandEncoder, px: number, py: number, pickRead: GPUBuffer): void {
    const [knotValue, world, depth] = this.pickTargets!;
    const patch = this.internals.patches[0];
    const vertices = patch.vertices!;
    if (this.pickGroup?.vertices !== vertices) {
      this.pickGroup = {
        vertices,
        group: this.device.createBindGroup({
          layout: this.pickPipeline!.getBindGroupLayout(0),
          entries: [patch.params, this.internals.view, vertices].map((buffer, binding) => ({ binding, resource: { buffer } })),
        }),
      };
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [knotValue, world].map((t) => ({
        view: t.createView(),
        clearValue: [-1, -1, 0, 0],
        loadOp: "clear" as const,
        storeOp: "store" as const,
      })),
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setPipeline(this.pickPipeline!);
    pass.setBindGroup(0, this.pickGroup.group);
    pass.setScissorRect(px, py, 1, 1); // only the hovered pixel is shaded
    pass.draw(patch.cellCount * 6);
    pass.end();
    [knotValue, world].forEach((texture, i) =>
      encoder.copyTextureToBuffer(
        { texture, origin: [px, py] },
        { buffer: pickRead, offset: i * 256, bytesPerRow: 256, rowsPerImage: 1 },
        [1, 1],
      )
    );
  }

  private pickContext(period: [number, number] | null) {
    const { spans, surface, field, colorBy, transform } = this;
    return { spans, surface, field, colorBy, transform, shift: this.internals.shift, period };
  }

  private readPick(pixel: [number, number], pickRead: GPUBuffer, at: ReturnType<SurfaceRenderer["pickContext"]>): unknown {
    const data = new Float32Array(pickRead.getMappedRange().slice(0));
    pickRead.unmap();
    const knotValue = data.subarray(0, 4);
    const world = data.subarray(64, 68);
    if (world[3] < 0.5) {
      return null; // background
    }
    // Knot coordinates (span index + fraction) to parameters of the original knots
    const [u, v] = [0, 1].map((d) => {
      const spans = at.spans[d];
      const s = Math.min(Math.max(Math.floor(knotValue[d]), 0), spans.length - 1);
      const frac = Math.min(Math.max(knotValue[d] - s, 0), 1);
      const { start, end } = spans[s];
      return start + frac * (end - start);
    });
    const ref = evaluateNurbsSurface(at.surface, u, v); // CPU, f64
    const fields = at.surface.fields ?? [];
    const field = fields.length
      ? evaluateSurfaceAnalysis(at.surface, u, v, fields[at.field])[NurbsColorBy.Field]
      : null;
    const { origin, scale, offset } = at.transform;
    const shift = at.shift;
    const [lo, hi] = at.period ?? [0, 0]; // cyclic: unwrapped by the vertex shader, back into a period
    const wrap = (x: number) => (hi !== lo ? x - (hi - lo) * Math.floor((x - lo) / (hi - lo)) : x);
    return {
      pixel,
      u,
      v,
      xyz: ref.point, // CPU (f64) at (u, v)
      xyzMesh: [0, 1, 2].map((d) => (world[d] - offset[d]) / (scale[d] > 0 ? scale[d] : 1) + origin[d]), // GPU
      normal: ref.normal,
      field,
      colorBy: at.colorBy,
      value: knotValue[3] > 0.5 ? wrap(knotValue[2] + shift) : null, // coloured value
    };
  }

  // Outputs -----------------------------------------------------------------------------------

  state(input: Control, generation: number, hover: unknown, ms: number): unknown {
    const ctl = { ...input, ...this.resolvedCamera(input) };
    const { eye, viewProjection } = SurfaceRenderer.camera(ctl);
    return {
      generation,
      content: screenRect(this.fitPoints, viewProjection, Math.trunc(ctl.width), Math.trunc(ctl.height)), // the surface's pixels
      width: Math.trunc(ctl.width),
      height: Math.trunc(ctl.height),
      camera: {
        azimuth: ctl.azimuth,
        elevation: ctl.elevation,
        distance: ctl.distance,
        target: [...(ctl.target as Vec3)],
        position: eye,
        fov: FOV,
      },
      surface: typeof ctl.surface === "string" ? ctl.surface : "inline",
      colorBy: this.colorBy,
      colormap: ctl.colormap,
      valueRange: [...(ctl.domain ?? this.dataDomain)],
      ...(this.slider ? { rootN: this.slider.value, rootSlider: this.slider } : {}),
      grid: [...this.grid],
      hover,
      renderMs: Math.round(ms * 1000) / 1000,
    };
  }

  /** `resolveCamera` once the surface is set (synchronous: the fit is cached by then). */
  private resolvedCamera(ctl: Control): { distance: number; target: Vec3 } {
    const [target, distance] = this.fit;
    return {
      distance: ctl.distance ?? distance / (Number(ctl.zoom) || 1),
      target: ctl.target ?? [...target],
    };
  }

  dispose(): void {
    this.gpu?.dispose();
  }
}

if (import.meta.main) {
  await main<Control>({
    name: "surface.ts",
    defaults: DEFAULTS as Control,
    create: SurfaceRenderer.create,
    onPng(ctl, frame) {
      if (frame.hover !== null || ctl.hover !== null) {
        console.log(JSON.stringify({ hover: frame.hover }));
      }
    },
  });
}
