import { getGpuParams, type KnotSpan, type NurbsCurve } from "./nurbs.ts";
import { createBuffer, NurbsCurveSampler, type TextureTarget, UNIFORM } from "./gpu.ts";
import {
  COLOR_WGSL,
  createLutBuffer,
  DEPTH_FORMAT,
  getTargets,
  type RenderTargets,
  SAMPLE_COUNT,
  writeColorLut,
} from "./gpu_surface.ts";

/*
 * Each chord between consecutive samples as a quad of constant width on screen, between the
 * projections of its ends (with their depths), coloured by `u` with the colour map. Square: the
 * quads overlap by half the width past each end (covering the joins). Round (\`view.round\`): each
 * chord a capsule, a pixel wider, its coverage per multisample (\`capsuleMask\`): round joins and
 * caps, antialiased as the edges are.
 */
export const DRAW_WGSL = /* wgsl */ `
  struct View {
    viewProjection: mat4x4f, // WebGL clip space
    halfSize: vec2f, // half size of canvas in device pixels
    uRange: vec2f, // parameters at ends of colour map
    width: f32, // in device pixels
    round: f32, // 1: round joins and caps
  }

  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) u: f32,
    @location(1) @interpolate(linear) local: vec2f, // device pixels along the chord from its start, across it
    @location(2) @interpolate(flat) len: f32, // of the chord, device pixels
  }

  struct Fragment { @location(0) color: vec4f, @builtin(sample_mask) mask: u32 }

  @group(0) @binding(0) var<uniform> view: View;
  @group(0) @binding(1) var<storage, read> samples: array<vec4f>; // x, y, z, u
  @group(0) @binding(2) var<storage, read> lut: array<vec4f>; // linear RGB

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) chord: u32) -> Varyings {
    let a = view.viewProjection * vec4f(samples[chord].xyz, 1.0);
    let b = view.viewProjection * vec4f(samples[chord + 1u].xyz, 1.0);
    let sa = a.xy / a.w * view.halfSize; // device pixels from centre of canvas
    let sb = b.xy / b.w * view.halfSize;
    let len = length(sb - sa);
    let dir = select(vec2f(1.0, 0.0), (sb - sa) / len, len > 1e-6);
    let normal = vec2f(-dir.y, dir.x);
    let half = 0.5 * view.width + select(0.0, 1.0, view.round > 0.5); // round: room for the coverage

    // Two triangles; ends overlap the next chords' to cover the joins
    var ends = array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    var sides = array(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let isEnd = ends[v] > 0.5;
    let clip = select(a, b, isEnd);
    let screen = select(sa - dir * half, sb + dir * half, isEnd) + normal * half * sides[v];

    var out: Varyings;
    out.local = vec2f(select(-half, len + half, isEnd), half * sides[v]);
    out.len = len;
    out.position = vec4f(screen / view.halfSize * clip.w, 0.5 * (clip.z + clip.w), clip.w);
    out.u = select(samples[chord].w, samples[chord + 1u].w, isEnd);
    if (a.w <= 0.0 || b.w <= 0.0) {
      out.position = vec4f(0.0, 0.0, 2.0, 1.0); // behind the camera: clipped
    }
    return out;
  }

  ${COLOR_WGSL}

  /* Samples (of the standard 4× pattern) within half the width of the chord, if round; all otherwise.
   * Screen y down, as the framebuffer's. */
  fn capsuleMask(in: Varyings) -> u32 {
    let dx = dpdx(in.local);
    let dy = dpdy(in.local);
    if (view.round < 0.5) {
      return 0xffffffffu;
    }
    var offsets = array(vec2f(0.375, 0.125), vec2f(0.875, 0.375), vec2f(0.125, 0.625), vec2f(0.625, 0.875));
    let half = 0.5 * view.width;
    var mask = 0u;
    for (var i = 0u; i < 4u; i++) {
      let o = offsets[i] - 0.5;
      let p = in.local + dx * o.x + dy * o.y;
      if (length(vec2f(p.x - clamp(p.x, 0.0, in.len), p.y)) <= half) {
        mask |= 1u << i;
      }
    }
    return mask;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> Fragment {
    let mask = capsuleMask(in);
    let range = view.uRange.y - view.uRange.x;
    let t = select(0.5, (in.u - view.uRange.x) / range, range > 0.0);
    return Fragment(vec4f(toSrgb(getColor(t)), 1.0), mask);
  }
`;

export interface Curve3DGpuView {
  viewProjection: number[]; // column-major, as `Matrix4.elements`
  uRange: [number, number];
  width: number; // in device pixels
  round?: boolean; // round joins and caps (default: square, overlapping chords)
  size: [number, number]; // of the target in device pixels
}

/*
 * Draws a 3D NURBS curve (in world coordinates) into a texture target: the chords between the
 * samples of `NurbsCurveSampler`, evaluated again as the camera moves closer or away, with a
 * depth buffer and multisampling.
 */
export class NurbsCurve3DGpuRenderer {
  readonly sampler: NurbsCurveSampler;
  private readonly drawPipeline: GPURenderPipeline;
  readonly view: GPUBuffer;
  readonly lut: GPUBuffer;
  private drawBindGroup?: { group: GPUBindGroup; samples: GPUBuffer };
  private targets?: RenderTargets;

  public constructor(
    private readonly device: GPUDevice,
    private readonly target: TextureTarget,
    private readonly curve: NurbsCurve,
    spans: KnotSpan[],
  ) {
    const drawModule = device.createShaderModule({ code: DRAW_WGSL });
    this.drawPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: drawModule },
      fragment: { module: drawModule, targets: [{ format: target.format }] },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
      multisample: { count: SAMPLE_COUNT },
    });

    this.sampler = new NurbsCurveSampler(device, curve, spans, [0, 0, 0]);
    this.view = createBuffer(device, 96, UNIFORM);
    this.lut = createLutBuffer(device);
  }

  public setColorLut(lut: Float32Array): void {
    writeColorLut(this.device, this.lut, lut);
  }

  public setSegmentCounts(counts: Uint32Array): void {
    this.sampler.setSegmentCounts(counts);
  }

  public draw(view: Curve3DGpuView): void {
    const { device } = this;
    const { viewProjection, uRange, width, round, size } = view;

    device.queue.writeBuffer(
      this.view,
      0,
      Float32Array.from([
        ...viewProjection,
        size[0] / 2,
        size[1] / 2,
        ...getGpuParams(this.curve, uRange), // as `u` of samples
        width,
        round ? 1 : 0,
        0,
        0,
      ]),
    );
    this.targets = getTargets(device, this.target.format, size, this.targets);

    this.redraw();
  }

  /* Draws again with the last view */
  public redraw(): void {
    const { device, sampler, targets } = this;
    if (!targets) {
      return;
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targets.color.createView(),
          resolveTarget: this.target.currentTexture().createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "discard",
        },
      ],
      depthStencilAttachment: {
        view: targets.depth.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });
    const { samples, sampleCount } = sampler;
    if (samples && sampleCount > 1) {
      if (this.drawBindGroup?.samples !== samples) {
        this.drawBindGroup = {
          samples,
          group: device.createBindGroup({
            layout: this.drawPipeline.getBindGroupLayout(0),
            entries: [this.view, samples, this.lut].map((buffer, binding) => ({
              binding,
              resource: { buffer },
            })),
          }),
        };
      }
      pass.setPipeline(this.drawPipeline);
      pass.setBindGroup(0, this.drawBindGroup.group);
      pass.draw(6, sampleCount - 1);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  public dispose(): void {
    this.sampler.dispose();
    this.view.destroy();
    this.lut.destroy();
    this.targets?.color.destroy();
    this.targets?.depth.destroy();
  }
}
