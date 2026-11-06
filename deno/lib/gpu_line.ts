import { type KnotSpan, type NurbsCurve } from "./nurbs.ts";
import { createBuffer, NurbsCurveSampler, type TextureTarget, UNIFORM } from "./gpu.ts";

/* Each chord between consecutive samples as a quad of constant width on screen */
const DRAW_WGSL = /* wgsl */ `
  struct View {
    scale: vec2f, // device pixels per unit
    offset: vec2f, // device pixels from centre of canvas (y up) of origin of samples
    halfSize: vec2f, // half size of canvas in device pixels
    width: f32, // in device pixels
    color: vec4f, // premultiplied
  }

  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) across: f32, // device pixels from centre line
  }

  @group(0) @binding(0) var<uniform> view: View;
  @group(0) @binding(1) var<storage, read> samples: array<vec4f>;

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) chord: u32) -> Varyings {
    let a = samples[chord].xy * view.scale + view.offset;
    let b = samples[chord + 1u].xy * view.scale + view.offset;
    let len = length(b - a);
    let dir = select(vec2f(1.0, 0.0), (b - a) / len, len > 1e-6);
    let normal = vec2f(-dir.y, dir.x);
    let half = 0.5 * view.width + 1.0; // one more pixel to antialias edges

    // Two triangles; ends overlap the next chords' to cover the joins
    var ends = array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    var sides = array(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let along = mix(a - dir * half, b + dir * half, ends[v]);

    var out: Varyings;
    out.position = vec4f((along + normal * half * sides[v]) / view.halfSize, 0.0, 1.0);
    out.across = half * sides[v];
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    return view.color * clamp(0.5 * view.width + 0.5 - abs(in.across), 0.0, 1.0);
  }
`;
export interface GpuView {
  scale: [number, number];
  offset: [number, number];
  size: [number, number]; // of the target in device pixels
  width: number;
  color: [number, number, number, number]; // premultiplied
}

/*
 * Draws a 2D NURBS curve into a texture target: the chords between the samples of
 * `NurbsCurveSampler`, evaluated again on zoom. Several renderers can draw into one target in
 * turn (`load`: blend over what is there rather than clear it).
 */
export class NurbsGpuRenderer {
  readonly sampler: NurbsCurveSampler;
  private readonly drawPipeline: GPURenderPipeline;
  private readonly view: GPUBuffer;
  private drawBindGroup?: { group: GPUBindGroup; samples: GPUBuffer };

  public constructor(
    private readonly device: GPUDevice,
    private readonly target: TextureTarget,
    curve: NurbsCurve,
    spans: KnotSpan[],
    origin: [number, number],
  ) {
    const drawModule = device.createShaderModule({ code: DRAW_WGSL });
    this.drawPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: drawModule },
      fragment: {
        module: drawModule,
        targets: [
          {
            format: target.format,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
    });

    this.sampler = new NurbsCurveSampler(device, curve, spans, origin);
    this.view = createBuffer(device, 48, UNIFORM);
  }

  public setSegmentCounts(counts: Uint32Array): void {
    this.sampler.setSegmentCounts(counts);
  }

  public draw(view: GpuView, load = false): void {
    const { device, sampler } = this;
    const { scale, offset, size, width, color } = view;

    device.queue.writeBuffer(
      this.view,
      0,
      Float32Array.from([
        ...scale,
        ...offset,
        size[0] / 2,
        size[1] / 2,
        width,
        0,
        ...color,
      ]),
    );

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.target.currentTexture().createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: load ? "load" : "clear",
          storeOp: "store",
        },
      ],
    });
    const { samples, sampleCount } = sampler;
    if (samples && sampleCount > 1) {
      if (this.drawBindGroup?.samples !== samples) {
        this.drawBindGroup = {
          samples,
          group: device.createBindGroup({
            layout: this.drawPipeline.getBindGroupLayout(0),
            entries: [this.view, samples].map((buffer, binding) => ({
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
  }
}
