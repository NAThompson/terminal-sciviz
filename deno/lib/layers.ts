/**
 * Composites the layers of line.ts: each series' curves are drawn into a scratch layer, then
 * composited over the frame's curve layer within rectangles (dash strips), times an opacity.
 */

import { FRAME_FORMAT } from "./overlay.ts";

const COMPOSITE_WGSL = /* wgsl */ `
  struct Globals { size: vec2f, opacity: f32, pad: f32 }
  @group(0) @binding(0) var<uniform> g: Globals;
  @group(0) @binding(1) var<storage, read> rects: array<vec4f>; // left, top, right, bottom (px)
  @group(0) @binding(2) var src: texture_2d<f32>;

  @vertex
  fn vertexMain(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4f {
    var corners = array(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
    let r = rects[i];
    let p = (mix(r.xy, r.zw, corners[v])) / g.size;
    return vec4f(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
  }

  @fragment
  fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    return textureLoad(src, vec2i(pos.xy), 0) * g.opacity;
  }
`;

/**
 * Composites one layer texture (premultiplied) `over` another within rectangles, times an
 * opacity: each series of line.ts is drawn opaque into a scratch layer, then composited into the
 * frame's curve layer in draw order, with its own opacity and its own dash mask (strips).
 */
export class LayerCompositor {
  private readonly pipeline: GPURenderPipeline;
  private readonly globals: GPUBuffer;
  private rects?: GPUBuffer;
  private group?: { key: unknown[]; group: GPUBindGroup };

  constructor(private readonly device: GPUDevice) {
    const module = device.createShaderModule({ code: COMPOSITE_WGSL });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module },
      fragment: {
        module,
        targets: [{
          format: FRAME_FORMAT,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
    });
    this.globals = device.createBuffer({ size: 16, usage: 0x40 | 0x08 });
  }

  /** Submits `src` over `dst` within `rects` ([l, t, r, b] in pixels), times `opacity`. */
  composite(src: GPUTexture, dst: GPUTexture, rects: [number, number, number, number][], opacity: number): void {
    if (!rects.length) return;
    const { device } = this;
    const data = Float32Array.from(rects.flat());
    if (!this.rects || this.rects.size < data.byteLength) {
      this.rects?.destroy();
      this.rects = device.createBuffer({ size: Math.max(2 ** Math.ceil(Math.log2(data.byteLength)), 256), usage: 0x80 | 0x08 });
    }
    // Queue order: these writes land before this submit, after the previous one
    device.queue.writeBuffer(this.globals, 0, Float32Array.of(dst.width, dst.height, opacity, 0));
    device.queue.writeBuffer(this.rects, 0, data);
    const key = [src, this.rects];
    if (!this.group || this.group.key.some((k, i) => k !== key[i])) {
      this.group = {
        key,
        group: device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.globals } },
            { binding: 1, resource: { buffer: this.rects } },
            { binding: 2, resource: src.createView() },
          ],
        }),
      };
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: dst.createView(), loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group.group);
    pass.draw(6, rects.length);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}
