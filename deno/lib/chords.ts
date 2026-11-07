/**
 * Polyline of samples already on the GPU (`x, y, z, u` per sample, relative to an origin, as
 * line.ts uploads them), drawn as chords of constant width on screen into a texture,
 * blended `over` what is there (premultiplied). The draw shader is that of the NURBS line
 * renderer (`NurbsGpuRenderer`, lib/gpu_line.ts), unchanged, so these match its curves.
 */

import { FRAME_FORMAT } from "./overlay.ts";

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
    let pa = samples[chord].xy;
    let pb = samples[chord + 1u].xy;
    // a NaN or infinite end (a pole, outside the domain): no chord, a gap; huge finite ends are
    // clamped on screen, so the chord leaves the plot steeply instead of overflowing
    let finite = all(abs(vec4f(pa, pb)) < vec4f(3.4e38));
    let a = clamp(pa * view.scale + view.offset, vec2f(-1e7), vec2f(1e7));
    let b = clamp(pb * view.scale + view.offset, vec2f(-1e7), vec2f(1e7));
    let len = length(b - a);
    let dir = select(vec2f(1.0, 0.0), (b - a) / len, len > 1e-6);
    let normal = vec2f(-dir.y, dir.x);
    let half = 0.5 * view.width + 1.0; // one more pixel to antialias edges

    // Two triangles; ends overlap the next chords' to cover the joins
    var ends = array(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    var sides = array(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);
    let along = mix(a - dir * half, b + dir * half, ends[v]);

    var out: Varyings;
    out.position = select(vec4f(2.0, 2.0, 0.0, 1.0), vec4f((along + normal * half * sides[v]) / view.halfSize, 0.0, 1.0), finite);
    out.across = half * sides[v];
    return out;
  }

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    return view.color * clamp(0.5 * view.width + 0.5 - abs(in.across), 0.0, 1.0);
  }
`;

export interface ChordView {
  scale: [number, number];
  offset: [number, number];
  size: [number, number]; // of the target in device pixels
  width: number;
  color: [number, number, number, number]; // premultiplied
}

export class ChordRenderer {
  private readonly pipeline: GPURenderPipeline;
  private readonly view: GPUBuffer;
  private readonly groups = new WeakMap<GPUBuffer, GPUBindGroup>();

  constructor(private readonly device: GPUDevice) {
    const module = device.createShaderModule({ code: DRAW_WGSL });
    const over: GPUBlendComponent = { srcFactor: "one", dstFactor: "one-minus-src-alpha" };
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module },
      fragment: { module, targets: [{ format: FRAME_FORMAT, blend: { color: over, alpha: over } }] },
    });
    this.view = device.createBuffer({ size: 48, usage: 0x40 | 0x08 });
  }

  /** Draws the `count` samples of `samples` into `target`, over what is there. */
  draw(target: GPUTexture, samples: GPUBuffer, count: number, v: ChordView): void {
    const { device } = this;
    if (count < 2) return;
    device.queue.writeBuffer(this.view, 0, Float32Array.from([...v.scale, ...v.offset, v.size[0] / 2, v.size[1] / 2, v.width, 0, ...v.color]));
    let group = this.groups.get(samples);
    if (!group) {
      group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [this.view, samples].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.groups.set(samples, group);
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "load", storeOp: "store" }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(6, count - 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  dispose(): void {
    this.view.destroy();
  }
}
