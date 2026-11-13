import { type Domain, getKnotVectors, getGpuParams, type KnotSpan, type NurbsSurface, COLOR_BY_MODES, DEGENERATE_METRIC, NurbsColorBy } from "./nurbs.ts";
import { createBuffer, MAX_GPU_DEGREE, SAFE_DIVIDE_WGSL, STORAGE, type TextureTarget, UNIFORM } from "./gpu.ts";

type Vector3Tuple = [number, number, number];

export const NAN_COLOR: Vector3Tuple = [0.2, 0.2, 0.2]; // linear RGB of invalid values

/* Extent of the finite values of one or more arrays; `[-0.5, 0.5]` if there is none */
export function getExtentDomain(...arrays: ArrayLike<number>[]): Domain {
  let min = Infinity;
  let max = -Infinity;
  for (const arr of arrays) {
    for (let i = 0; i < arr.length; i += 1) {
      const val = arr[i];
      if (Number.isFinite(val)) {
        min = Math.min(min, val);
        max = Math.max(max, val);
      }
    }
  }
  return min > max ? [-0.5, 0.5] : [min, max];
}

const WORKGROUP_SIZE = 64;
export const SAMPLE_COUNT = 4; // multisampling, as WebGL's antialiasing
export const DEPTH_FORMAT = "depth24plus";
const LUT_SIZE = 256;

/* Linear RGB of the colour map `lut` at `t` in [0, 1], and its conversion to sRGB */
export const COLOR_WGSL = /* wgsl */ `
  fn getColor(t: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0) * f32(${LUT_SIZE - 1});
    let i = u32(floor(x));
    return mix(lut[i], lut[min(i + 1u, ${LUT_SIZE - 1}u)], fract(x)).rgb;
  }

  fn toSrgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308));
  }
`;

/*
 * Position and normal of the surface at each vertex of the grid (A2.2/A2.3 + A3.5 + A4.4 of
 * "The NURBS Book"), like `evaluateNurbsSurface`, and the value it is coloured by, like
 * `evaluateSurfaceAnalysis` (up to second derivatives, in data coordinates relative to an origin;
 * positions and normals are then scaled and offset to world coordinates). `.w` of position and
 * normal holds the position of the vertex in "knot coordinates", whole at knots: span index plus
 * fraction of the span. Values are flagged invalid where the metric is degenerate (WGSL has no
 * reliable `NaN`).
 */
const EVALUATE_WGSL = /* wgsl */ `
  const ORDER = ${MAX_GPU_DEGREE + 1};
  const NUDGE = 1e-3; // fraction of span to find normals where they vanish (f32)
  const DEGENERATE_METRIC = ${DEGENERATE_METRIC};

  // Colour modes (cf. \`COLOR_BY_MODES\`)
  const HEIGHT = 0u;
  const FIELD = 1u;
  const FIELD_GRADIENT = 2u;
  const HEIGHT_GRADIENT = 3u;
  const GAUSSIAN_CURVATURE = 4u;
  const MEAN_CURVATURE = 5u;

  struct Params {
    degrees: vec2u,
    countV: u32, // control points along v
    sizeV: u32, // vertices along v
    spanCounts: vec2u,
    total: u32, // vertices
    knotsV: u32, // offset of knots along v
    mode: u32,
    fieldOffset: u32, // of the field's control values
    scale: vec4f, // world = data relative to origin × scale + offset
    offset: vec4f,
  }
  struct Span { knot: u32, first: u32, count: u32 } // \`count\` steps from vertex \`first\`
  struct Vertex { position: vec4f, normal: vec4f, value: vec4f } // value, 1 if valid, area
  struct Basis { n: array<f32, ORDER>, d: array<f32, ORDER>, d2: array<f32, ORDER> }
  struct Point { position: vec3f, normal: vec3f, value: vec2f, area: f32 } // per unit u × v

  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<storage, read> points: array<vec4f>; // x w, y w, z w, w
  @group(0) @binding(2) var<storage, read> knots: array<f32>; // along u, then along v
  @group(0) @binding(3) var<storage, read> spans: array<Span>; // along u, then along v
  @group(0) @binding(4) var<storage, read_write> vertices: array<Vertex>;
  @group(0) @binding(5) var<storage, read> fields: array<f32>; // control values, field after field

  ${SAFE_DIVIDE_WGSL}

  /* Last span from \`first\` (of \`count\`) starting at or before step \`i\` */
  fn findSpan(first: u32, count: u32, i: u32) -> u32 {
    var low = first;
    var high = first + count - 1u;
    while (low < high) {
      let mid = (low + high + 1u) / 2u;
      if (spans[mid].first <= i) { low = mid; } else { high = mid - 1u; }
    }
    return low;
  }

  /* Derivatives of the basis functions of degree \`q\` of span \`k\` from \`low\`, those of degree
   * \`q - 1\` or their derivatives (cf. \`deriveBasis\`) */
  fn derive(offset: u32, q: u32, k: u32, low: array<f32, ORDER>) -> array<f32, ORDER> {
    var d: array<f32, ORDER>;
    for (var r = 0u; r <= q; r++) {
      var der = 0.0;
      if (r > 0u) {
        der += safeDivide(low[r - 1u], knots[offset + k + r] - knots[offset + k + r - q]);
      }
      if (r < q) {
        der -= safeDivide(low[r], knots[offset + k + r + 1u] - knots[offset + k + r + 1u - q]);
      }
      d[r] = f32(q) * der;
    }
    return d;
  }

  /* Basis functions of span \`k\` at \`t\` and their first and second derivatives, from those of
   * degrees \`p - 1\` and \`p - 2\` */
  fn getBasis(offset: u32, p: u32, k: u32, t: f32) -> Basis {
    var b: Basis;
    var lower: array<f32, ORDER>;
    var lower2: array<f32, ORDER>;
    var left: array<f32, ORDER>;
    var right: array<f32, ORDER>;
    b.n[0] = 1.0;
    for (var j = 1u; j <= p; j++) {
      if (j == p) { lower = b.n; }
      if (j + 1u == p) { lower2 = b.n; }
      left[j] = t - knots[offset + k + 1u - j];
      right[j] = knots[offset + k + j] - t;
      var saved = 0.0;
      for (var r = 0u; r < j; r++) {
        let temp = safeDivide(b.n[r], right[r + 1u] + left[j - r]);
        b.n[r] = saved + right[r + 1u] * temp;
        saved = left[j - r] * temp;
      }
      b.n[j] = saved;
    }
    b.d = derive(offset, p, k, lower);
    if (p > 1u) {
      b.d2 = derive(offset, p, k, derive(offset, p - 1u, k, lower2));
    }
    return b;
  }

  fn evaluate(ku: u32, kv: u32, u: f32, v: f32) -> Point {
    let p = params.degrees;
    let bu = getBasis(0u, p.x, ku, u);
    let bv = getBasis(params.knotsV, p.y, kv, v);
    let hasField = params.mode == FIELD || params.mode == FIELD_GRADIENT;

    // Homogeneous point and its derivatives; field times weight and its first derivatives
    var a = vec4f(0.0);
    var au = vec4f(0.0);
    var av = vec4f(0.0);
    var auu = vec4f(0.0);
    var auv = vec4f(0.0);
    var avv = vec4f(0.0);
    var fw = vec3f(0.0);
    for (var r = 0u; r <= p.x; r++) {
      for (var c = 0u; c <= p.y; c++) {
        let i = (ku - p.x + r) * params.countV + kv - p.y + c;
        let point = points[i];
        a += bu.n[r] * bv.n[c] * point;
        au += bu.d[r] * bv.n[c] * point;
        av += bu.n[r] * bv.d[c] * point;
        auu += bu.d2[r] * bv.n[c] * point;
        auv += bu.d[r] * bv.d[c] * point;
        avv += bu.n[r] * bv.d2[c] * point;
        if (hasField) {
          let n = vec3f(bu.n[r] * bv.n[c], bu.d[r] * bv.n[c], bu.n[r] * bv.d[c]);
          fw += n * fields[params.fieldOffset + i] * point.w;
        }
      }
    }
    let w = a.w;
    let s = a.xyz / w;
    let su = (au.xyz - au.w * s) / w;
    let sv = (av.xyz - av.w * s) / w;
    let suu = (auu.xyz - 2.0 * au.w * su - auu.w * s) / w;
    let svv = (avv.xyz - 2.0 * av.w * sv - avv.w * s) / w;
    let suv = (auv.xyz - au.w * sv - av.w * su - auv.w * s) / w;

    // Drawn in world coordinates
    let scale = params.scale.xyz;
    let du = su * scale;
    let dv = sv * scale;
    let normal = cross(du, dv);
    let longest = max(dot(du, du), dot(dv, dv));
    let isDegenerate = dot(normal, normal) <= 1e-8 * longest * longest; // e.g. du = 0 at a pole

    // Fundamental forms in data coordinates
    let e = dot(su, su);
    let f = dot(su, sv);
    let g = dot(sv, sv);
    let n = cross(su, sv);
    let det = dot(n, n); // EG − F²
    let isFlat = det <= DEGENERATE_METRIC * max(e, g) * max(e, g);
    let unit = n / sqrt(select(det, 1.0, isFlat));
    let l = dot(suu, unit);
    let m = dot(suv, unit);
    let nn = dot(svv, unit);

    let fieldValue = fw.x / w;
    var grad = vec2f(su.z, sv.z);
    if (params.mode == FIELD_GRADIENT) {
      grad = (fw.yz - vec2f(au.w, av.w) * fieldValue) / w;
    }

    var value = vec2f(s.z, 1.0);
    switch (params.mode) {
      case FIELD: { value.x = fieldValue; }
      case FIELD_GRADIENT, HEIGHT_GRADIENT: {
        let sq = g * grad.x * grad.x - 2.0 * f * grad.x * grad.y + e * grad.y * grad.y;
        value = vec2f(sqrt(max(sq, 0.0) / det), select(1.0, 0.0, isFlat));
      }
      case GAUSSIAN_CURVATURE: {
        value = vec2f((l * nn - m * m) / det, select(1.0, 0.0, isFlat));
      }
      case MEAN_CURVATURE: {
        value = vec2f((e * nn - 2.0 * f * m + g * l) / (2.0 * det), select(1.0, 0.0, isFlat));
      }
      default: {}
    }
    if (value.y == 0.0) {
      value.x = 0.0;
    }

    return Point(
      s * scale + params.offset.xyz,
      select(normalize(normal), vec3f(0.0), isDegenerate),
      value,
      sqrt(det),
    );
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= params.total) {
      return;
    }

    let steps = vec2u(i / params.sizeV, i % params.sizeV);
    let su = findSpan(0u, params.spanCounts.x, steps.x);
    let sv = findSpan(params.spanCounts.x, params.spanCounts.y, steps.y);
    let ku = spans[su].knot;
    let kv = spans[sv].knot;
    let fu = f32(steps.x - spans[su].first) / f32(spans[su].count);
    let fv = f32(steps.y - spans[sv].first) / f32(spans[sv].count);
    let u0 = knots[ku];
    let u1 = knots[ku + 1u];
    let v0 = knots[params.knotsV + kv];
    let v1 = knots[params.knotsV + kv + 1u];

    var point = evaluate(ku, kv, mix(u0, u1, fu), mix(v0, v1, fv));
    if (all(point.normal == vec3f(0.0))) { // e.g. a pole or a collapsed edge: from just inside
      var nudge = NUDGE;
      var inside = evaluate(ku, kv, mix(u0, u1, mix(fu, 0.5, nudge)), mix(v0, v1, mix(fv, 0.5, nudge)));
      point.normal = inside.normal;
      // Curvature, gradient: undefined at the point itself, and its metric still flat (f32) as
      // close as the normal is found if the collapse is of second order
      for (var k = 0; k < 3 && point.value.y == 0.0; k++) { // 1e-3, 1e-2, 1e-1 of the span
        if (k > 0) {
          nudge *= 10.0;
          inside = evaluate(ku, kv, mix(u0, u1, mix(fu, 0.5, nudge)), mix(v0, v1, mix(fv, 0.5, nudge)));
        }
        point.value = inside.value;
      }
    }

    vertices[i] = Vertex(
      vec4f(point.position, f32(su) + fu),
      vec4f(point.normal, f32(sv - params.spanCounts.x) + fv),
      vec4f(point.value, point.area * (u1 - u0) * (v1 - v0) / f32(spans[su].count * spans[sv].count), 0.0),
    );
  }
`;

/*
 * Two triangles per cell of the grid, Lambert-lit (both sides) by an ambient and a directional
 * light, coloured by value with the colour map, or
 * with the NaN colour where the value is invalid.
 */
export const DRAW_WGSL = /* wgsl */ `
  const KNOT_COLOR = vec3f(0.0);

  struct Params { degrees: vec2u, countV: u32, sizeV: u32 }
  struct View {
    viewProjection: mat4x4f, // WebGL clip space
    camera: vec3f, // position
    colorExponent: f32, // > 0: colours by t^colorExponent of the normalised value t (root scale), not cyclic
    light: vec4f, // direction to light, share of ambient light
    valueRange: vec2f, // values at ends of colour map
    knotLines: f32, // width in device pixels, 0 to hide
    cyclic: f32, // 1 if values wrap around the colour map (e.g. phase), its range being a period
    nanColor: vec4f, // linear RGB
  }
  struct Vertex { position: vec4f, normal: vec4f, value: vec4f }

  struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) knot: vec2f,
    @location(3) value: vec2f,
  }

  @group(0) @binding(0) var<uniform> params: Params;
  @group(0) @binding(1) var<uniform> view: View;
  @group(0) @binding(2) var<storage, read> vertices: array<Vertex>;
  @group(0) @binding(3) var<storage, read> lut: array<vec4f>; // linear RGB

  @vertex
  fn vertexMain(@builtin(vertex_index) index: u32) -> Varyings {
    let cell = index / 6u;
    let cols = params.sizeV - 1u;
    var du = array(0u, 1u, 0u, 0u, 1u, 1u);
    var dv = array(0u, 0u, 1u, 1u, 0u, 1u);
    let corner = index % 6u;
    let vertex = vertices[(cell / cols + du[corner]) * params.sizeV + cell % cols + dv[corner]];

    var clip = view.viewProjection * vec4f(vertex.position.xyz, 1.0);
    clip.z = 0.5 * (clip.z + clip.w); // to WebGPU's [0, 1]

    var out: Varyings;
    out.position = clip;
    out.world = vertex.position.xyz;
    out.normal = vertex.normal.xyz;
    out.knot = vec2f(vertex.position.w, vertex.normal.w);
    out.value = vertex.value.xy;

    // Within half a period (the colour domain) of the first valid vertex of the triangle (invalid
    // ones at it), so as not to interpolate across a wrap; ambiguous where values of a triangle
    // span half a period or more (e.g. at phase singularities); within subdivided knot spans, the
    // field itself interpolates control values across a wrap (short sweeps along branch cuts)
    let range = view.valueRange.y - view.valueRange.x;
    if (view.cyclic > 0.5 && range != 0.0) {
      var anchor = out.value.x; // as it is if all are invalid
      for (var c = corner - corner % 3u; c < corner - corner % 3u + 3u; c++) {
        let value = vertices[(cell / cols + du[c]) * params.sizeV + cell % cols + dv[c]].value;
        if (value.y > 0.5) {
          anchor = value.x;
          break;
        }
      }
      let wrapped = out.value.x - range * round((out.value.x - anchor) / range);
      out.value.x = select(anchor, wrapped, out.value.y > 0.5);
    }
    return out;
  }

  ${COLOR_WGSL}

  @fragment
  fn fragmentMain(in: Varyings) -> @location(0) vec4f {
    let knotWidth = fwidth(in.knot);

    // Facing the camera (both sides are lit)
    let n = normalize(in.normal);
    let normal = select(n, -n, dot(n, view.camera - in.world) < 0.0);

    let range = view.valueRange.y - view.valueRange.x;
    var t = select(0.5, (in.value.x - view.valueRange.x) / range, range != 0.0);
    t = select(t, fract(t), view.cyclic > 0.5); // a period of the colour map, if cyclic
    if (view.colorExponent > 0.0 && view.cyclic < 0.5) {
      t = pow(clamp(t, 0.0, 1.0), view.colorExponent);
    }
    let albedo = select(getColor(t), view.nanColor.rgb, in.value.y < 0.5);
    let ambient = view.light.w;
    var color = albedo * (ambient + (1.0 - ambient) * max(dot(normal, view.light.xyz), 0.0));

    // Iso-parameter lines at the knots (whole knot coordinates)
    if (view.knotLines > 0.0) {
      let d = abs(in.knot - round(in.knot)) / max(knotWidth, vec2f(1e-6));
      let coverage = clamp(0.5 * view.knotLines + 0.5 - min(d.x, d.y), 0.0, 1.0);
      color = mix(color, KNOT_COLOR, 0.85 * coverage);
    }

    return vec4f(toSrgb(color), 1.0);
  }
`;

const PARAMS_SIZE = 80;
const VERTEX_SIZE = 48; // bytes
const MAP_READ = 0x01;

export interface SurfaceGpuView {
  viewProjection: number[]; // column-major, as `Matrix4.elements`
  camera: [number, number, number];
  light: [number, number, number, number]; // unit direction, share of ambient light
  valueRange: Domain; // values at the ends of the colour map
  knotLines: number; // width in device pixels, 0 to hide
  cyclic?: boolean; // whether values wrap around the colour map, its range being a period
  colorExponent?: number; // > 0: colours by t^colorExponent of the normalised value t (root n: 1 / n)
  size: [number, number]; // of the target in device pixels
}

/* World coordinates of data coordinates `p`: `(p - origin) × scale + offset`, per axis */
export interface SurfaceTransform {
  origin: Vector3Tuple;
  scale: Vector3Tuple;
  offset: Vector3Tuple;
}

/* GPU buffers of a surface patch, and the steps of its knot spans */
export interface Patch {
  spans: [KnotSpan[], KnotSpan[]];
  paramData: ArrayBuffer; // of `params`, but for `sizeV` and `total`
  params: GPUBuffer;
  points: GPUBuffer;
  knots: GPUBuffer;
  fields: GPUBuffer;
  spanTable: GPUBuffer;
  vertices?: GPUBuffer;
  drawBindGroup?: GPUBindGroup;
  cellCount: number;
  counts?: [Uint32Array, Uint32Array];
}

/* Where a patch is evaluated: buffers of the patch, or temporary ones to read values back */
interface EvaluationTarget {
  params: GPUBuffer;
  spanTable: GPUBuffer;
  vertices: GPUBuffer;
}

/*
 * Draws NURBS surfaces (patches, in data coordinates, cf. `SurfaceTransform`) into a texture target:
 * a compute shader evaluates positions, normals and values on a grid of each patch whenever the
 * number of steps of its knot spans changes (i.e. as the camera moves closer or away) or what
 * the surface is coloured by changes, and a render pass draws the grids with a shared depth
 * buffer. Values can also be read back, on a given grid (e.g. for the colour domain).
 */
export class NurbsSurfaceGpuRenderer {
  private readonly evaluatePipeline: GPUComputePipeline;
  private readonly drawPipeline: GPURenderPipeline;
  readonly patches: Patch[];
  readonly view: GPUBuffer;
  private readonly lut: GPUBuffer;
  private readonly origin: Vector3Tuple;
  private readonly fieldShifts: number[]; // subtracted from control values, for `f32`
  shift = 0; // subtracted from values on the GPU
  private targets?: RenderTargets;

  public constructor(
    private readonly device: GPUDevice,
    private readonly target: TextureTarget,
    surfaces: NurbsSurface[],
    spans: [KnotSpan[], KnotSpan[]][], // of each surface
    transform: SurfaceTransform,
  ) {
    this.evaluatePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: EVALUATE_WGSL }) },
    });

    const drawModule = device.createShaderModule({ code: DRAW_WGSL });
    this.drawPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: drawModule },
      fragment: { module: drawModule, targets: [{ format: target.format }] },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: { count: SAMPLE_COUNT },
    });

    const { origin, scale, offset } = transform;
    this.origin = origin;
    this.fieldShifts = (surfaces[0].fields ?? []).map((_, f) => {
      const [min, max] = getExtentDomain(
        ...surfaces.map(({ fields }) => fields?.[f] ?? []),
      );
      return (min + max) / 2;
    });

    this.patches = surfaces.map((surface, p) => {
      const { degrees, shape, points, weights, knots, fields = [] } = surface;
      const homogeneous = new Float32Array(weights.length * 4);
      weights.forEach((w, i) => {
        homogeneous.set(
          [
            (points[i * 3] - origin[0]) * w,
            (points[i * 3 + 1] - origin[1]) * w,
            (points[i * 3 + 2] - origin[2]) * w,
            w,
          ],
          i * 4,
        );
      });
      const fieldValues = new Float32Array(
        Math.max(fields.length, 1) * weights.length,
      );
      fields.forEach((values, f) => {
        fieldValues.set(
          Float32Array.from(values, (val) => val - this.fieldShifts[f]),
          f * weights.length,
        );
      });

      const [spansU, spansV] = spans[p];
      const paramData = new ArrayBuffer(PARAMS_SIZE);
      new Uint32Array(paramData).set([
        ...degrees,
        shape[1],
        0,
        spansU.length,
        spansV.length,
        0,
        knots[0].length,
      ]);
      new Float32Array(paramData).set([...scale, 0, ...offset, 0], 12);
      const params = createBuffer(device, PARAMS_SIZE, UNIFORM);
      device.queue.writeBuffer(params, 0, paramData);

      return {
        spans: spans[p],
        paramData,
        params,
        points: createBuffer(device, homogeneous, STORAGE),
        knots: createBuffer(
          device,
          Float32Array.from(
            getKnotVectors(surface).flatMap((vec) => [...getGpuParams(vec)]),
          ),
          STORAGE,
        ),
        fields: createBuffer(device, fieldValues, STORAGE),
        spanTable: createBuffer(
          device,
          (spansU.length + spansV.length) * 12,
          STORAGE,
        ),
        cellCount: 0,
      };
    });
    this.view = createBuffer(device, 128, UNIFORM);
    this.lut = createLutBuffer(device);
  }

  public setColorLut(lut: Float32Array): void {
    writeColorLut(this.device, this.lut, lut);
  }

  /* Values of the next evaluation: what to colour by, and which field (index in `fields`) */
  public setColorBy(colorBy: NurbsColorBy, field = 0): void {
    const isField =
      colorBy === NurbsColorBy.Field || colorBy === NurbsColorBy.FieldGradient;
    this.shift =
      colorBy === NurbsColorBy.Height
        ? this.origin[2]
        : colorBy === NurbsColorBy.Field
          ? this.fieldShifts[field]
          : 0;

    for (const patch of this.patches) {
      const pointCount = patch.points.size / 16;
      const data = Uint32Array.of(
        COLOR_BY_MODES[colorBy],
        isField ? field * pointCount : 0,
      );
      new Uint32Array(patch.paramData).set(data, 8);
      this.device.queue.writeBuffer(patch.params, 32, data);
      patch.counts = undefined; // evaluate again
    }
  }

  /* Evaluates each patch again if its spans need other numbers of steps */
  public setSampleCounts(counts: [Uint32Array, Uint32Array][]): void {
    const { device } = this;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.evaluatePipeline);

    for (const [p, patch] of this.patches.entries()) {
      const { counts: prev } = patch;
      const patchCounts = counts[p];
      if (
        prev &&
        patchCounts.every((arr, dir) => arr.every((n, i) => n === prev[dir][i]))
      ) {
        continue;
      }
      patch.counts = patchCounts;

      const [sizeU, sizeV] = getGridSize(patchCounts);
      const total = sizeU * sizeV;
      patch.cellCount = (sizeU - 1) * (sizeV - 1);

      if (!patch.vertices || patch.vertices.size < total * VERTEX_SIZE) {
        patch.vertices?.destroy();
        patch.vertices = createBuffer(
          device,
          2 ** Math.ceil(Math.log2(total * VERTEX_SIZE)), // room to move closer
          STORAGE,
        );
        patch.drawBindGroup = device.createBindGroup({
          layout: this.drawPipeline.getBindGroupLayout(0),
          entries: [patch.params, this.view, patch.vertices, this.lut].map(
            (buffer, binding) => ({ binding, resource: { buffer } }),
          ),
        });
      }

      this.encodeEvaluation(pass, patch, patchCounts, {
        params: patch.params,
        spanTable: patch.spanTable,
        vertices: patch.vertices,
      });
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /* Values (`NaN` where invalid) on grids of given steps per knot span, patch after patch, and
   * the area each vertex stands for (in data units) */
  public async readValues(
    counts: [Uint32Array, Uint32Array][],
  ): Promise<{ values: Float64Array; areas: Float64Array }> {
    const { device } = this;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.evaluatePipeline);

    const targets = this.patches.map((patch, p) => {
      const [sizeU, sizeV] = getGridSize(counts[p]);
      const target: EvaluationTarget = {
        params: createBuffer(device, PARAMS_SIZE, UNIFORM),
        spanTable: createBuffer(device, patch.spanTable.size, STORAGE),
        vertices: createBuffer(
          device,
          sizeU * sizeV * VERTEX_SIZE,
          STORAGE, // copied from (cf. `createBuffer`)
        ),
      };
      this.encodeEvaluation(pass, patch, counts[p], target);
      return target;
    });
    pass.end();

    const total = targets.reduce((acc, { vertices }) => acc + vertices.size, 0);
    const readBuffer = device.createBuffer({
      size: total,
      usage: MAP_READ | 0x08, // `COPY_DST`
    });
    let offset = 0;
    for (const { vertices } of targets) {
      encoder.copyBufferToBuffer(
        vertices,
        0,
        readBuffer,
        offset,
        vertices.size,
      );
      offset += vertices.size;
    }
    device.queue.submit([encoder.finish()]);

    try {
      await readBuffer.mapAsync(MAP_READ);
      const data = new Float32Array(readBuffer.getMappedRange());
      const stride = VERTEX_SIZE / 4;
      const count = total / VERTEX_SIZE;
      return {
        values: Float64Array.from({ length: count }, (_, k) =>
          data[k * stride + 9] > 0.5
            ? data[k * stride + 8] + this.shift
            : Number.NaN,
        ),
        areas: Float64Array.from(
          { length: count },
          (_, k) => data[k * stride + 10],
        ),
      };
    } finally {
      readBuffer.destroy();
      for (const { params, spanTable, vertices } of targets) {
        params.destroy();
        spanTable.destroy();
        vertices.destroy();
      }
    }
  }

  public draw(view: SurfaceGpuView): void {
    const { device } = this;
    const { viewProjection, camera, light, valueRange, knotLines, cyclic, colorExponent, size } = view;

    device.queue.writeBuffer(
      this.view,
      0,
      Float32Array.from([
        ...viewProjection,
        ...camera,
        colorExponent ?? 0,
        ...light,
        valueRange[0] - this.shift,
        valueRange[1] - this.shift,
        knotLines,
        cyclic ? 1 : 0,
        ...NAN_COLOR,
        1,
      ]),
    );

    this.targets = getTargets(device, this.target.format, size, this.targets);
    this.redraw();
  }

  /* Draws again with the last view */
  public redraw(): void {
    const { device, targets } = this;
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
    pass.setPipeline(this.drawPipeline);
    for (const { drawBindGroup, cellCount } of this.patches) {
      if (drawBindGroup && cellCount > 0) {
        pass.setBindGroup(0, drawBindGroup);
        pass.draw(cellCount * 6);
      }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  public dispose(): void {
    for (const buffer of [
      this.view,
      this.lut,
      ...this.patches.flatMap((patch) => [
        patch.params,
        patch.points,
        patch.knots,
        patch.fields,
        patch.spanTable,
        patch.vertices,
      ]),
    ]) {
      buffer?.destroy();
    }
    this.targets?.color.destroy();
    this.targets?.depth.destroy();
  }

  /* Evaluates a patch on a grid of given steps per knot span, into `target` */
  private encodeEvaluation(
    pass: GPUComputePassEncoder,
    patch: Patch,
    counts: [Uint32Array, Uint32Array],
    target: EvaluationTarget,
  ): void {
    const { device } = this;
    const { spans } = patch;
    const [sizeU, sizeV] = getGridSize(counts);
    const total = sizeU * sizeV;

    const table = new Uint32Array((spans[0].length + spans[1].length) * 3);
    let offset = 0;
    for (const dir of [0, 1]) {
      let first = 0;
      for (const [s, { index }] of spans[dir].entries()) {
        table.set([index, first, counts[dir][s]], offset);
        offset += 3;
        first += counts[dir][s];
      }
    }

    const paramData = new Uint32Array(patch.paramData);
    paramData[3] = sizeV;
    paramData[6] = total;
    device.queue.writeBuffer(target.spanTable, 0, table);
    device.queue.writeBuffer(target.params, 0, paramData);

    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.evaluatePipeline.getBindGroupLayout(0),
        entries: [
          target.params,
          patch.points,
          patch.knots,
          target.spanTable,
          target.vertices,
          patch.fields,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(total / WORKGROUP_SIZE));
  }
}

/* Vertices along `u` and `v` of a grid of given steps per knot span */
function getGridSize(counts: [Uint32Array, Uint32Array]): [number, number] {
  return [
    counts[0].reduce((acc, n) => acc + n, 1),
    counts[1].reduce((acc, n) => acc + n, 1),
  ];
}

export function createLutBuffer(device: GPUDevice): GPUBuffer {
  return createBuffer(device, LUT_SIZE * 16, STORAGE);
}

/* Colour map, `LUT_SIZE × 3` linear RGB values (cf. `getLinearColorLut`) */
export function writeColorLut(
  device: GPUDevice,
  buffer: GPUBuffer,
  lut: Float32Array,
): void {
  const padded = new Float32Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i += 1) {
    padded.set(lut.subarray(i * 3, i * 3 + 3), i * 4);
  }
  device.queue.writeBuffer(buffer, 0, padded);
}

export interface RenderTargets {
  color: GPUTexture; // multisampled
  depth: GPUTexture;
}

/* Multisampled colour and depth textures of the target's size, new if `targets` aren't */
export function getTargets(
  device: GPUDevice,
  format: GPUTextureFormat,
  size: [number, number],
  targets?: RenderTargets,
): RenderTargets {
  if (targets?.color.width === size[0] && targets.color.height === size[1]) {
    return targets;
  }
  targets?.color.destroy();
  targets?.depth.destroy();
  const desc = { size, sampleCount: SAMPLE_COUNT, usage: 0x14 }; // `RENDER_ATTACHMENT | TEXTURE_BINDING` (scene.ts's volume reads the depth)
  return {
    color: device.createTexture({ ...desc, format }),
    depth: device.createTexture({ ...desc, format: DEPTH_FORMAT }),
  };
}
