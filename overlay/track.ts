// Native overlay that places itself exactly over a live sciviz view while the pointer is on it.
//   deno run -A --unstable-webgpu --unstable-ffi overlay/track.ts [SECONDS] [--events PATH]
// Calibration: the mod writes each pointer event's position *in the view's cells* to
// overlay.json in its session's state folder; we pair it with the global mouse position (points, top-left origin) and fit
// mouse = origin + cell * size per axis. Shown while hovering/dragging, hidden 1.5 s after leaving.
// Draws a cell grid, the view's border and a crosshair at the pointer, to check alignment.

const SECONDS = Number(Deno.args.find((a) => /^\d+(\.\d+)?$/.test(a)) ?? 120);
/** The pointer-event file: --events PATH, else the newest session's (~/.cache/terminal-sciviz/sessions/<id>/overlay.json). */
function eventsFile(): string {
  const i = Deno.args.indexOf("--events");
  if (i >= 0 && Deno.args[i + 1]) return Deno.args[i + 1];
  const root = `${Deno.env.get("HOME")}/.cache/terminal-sciviz/sessions`;
  let best = "", when = 0;
  try {
    for (const d of Deno.readDirSync(root)) {
      const mtime = Deno.statSync(`${root}/${d.name}`).mtime?.getTime() ?? 0;
      if (d.isDirectory && mtime > when) [best, when] = [`${root}/${d.name}/overlay.json`, mtime];
    }
  } catch { /* no sessions yet */ }
  return best;
}
const EVENTS = eventsFile();
const SCALE = 2; // Retina backing scale

// ---- Objective-C runtime / Cocoa ----
const RECT = { struct: ["f64", "f64", "f64", "f64"] } as const;
const POINT = { struct: ["f64", "f64"] } as const;
const objc = Deno.dlopen("/usr/lib/libobjc.A.dylib", {
  objc_getClass: { parameters: ["buffer"], result: "pointer" },
  sel_registerName: { parameters: ["buffer"], result: "pointer" },
  send: { name: "objc_msgSend", parameters: ["pointer", "pointer"], result: "pointer" },
  sendP: { name: "objc_msgSend", parameters: ["pointer", "pointer", "pointer"], result: "pointer" },
  sendI: { name: "objc_msgSend", parameters: ["pointer", "pointer", "i64"], result: "pointer" },
  sendB: { name: "objc_msgSend", parameters: ["pointer", "pointer", "u8"], result: "pointer" },
  sendD: { name: "objc_msgSend", parameters: ["pointer", "pointer", "f64"], result: "pointer" },
  sendInitRect: { name: "objc_msgSend", parameters: ["pointer", "pointer", RECT, "u64", "u64", "u8"], result: "pointer" },
  sendRectB: { name: "objc_msgSend", parameters: ["pointer", "pointer", RECT, "u8"], result: "void" },
  sendEventArgs: { name: "objc_msgSend", parameters: ["pointer", "pointer", "u64", "pointer", "pointer", "u8"], result: "pointer" },
  sendRectRet: { name: "objc_msgSend", parameters: ["pointer", "pointer"], result: RECT },
});
Deno.dlopen("/System/Library/Frameworks/AppKit.framework/AppKit", {});
Deno.dlopen("/System/Library/Frameworks/QuartzCore.framework/QuartzCore", {});
const cg = Deno.dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", {
  CGColorSpaceCreateWithName: { parameters: ["pointer"], result: "pointer" },
  CGEventCreate: { parameters: ["pointer"], result: "pointer" },
  CGEventGetLocation: { parameters: ["pointer"], result: POINT },
  CFRelease: { parameters: ["pointer"], result: "void" },
});
const S = objc.symbols;
const cstr = (s: string) => new TextEncoder().encode(s + "\0");
const cls = (n: string) => S.objc_getClass(cstr(n));
const sel = (n: string) => S.sel_registerName(cstr(n));
const nsstr = (s: string) => S.sendP(cls("NSString"), sel("stringWithUTF8String:"), Deno.UnsafePointer.of(cstr(s)));
const name = (o: Deno.PointerValue) =>
  Deno.UnsafePointerView.getCString(S.send(S.send(S.send(o, sel("class")), sel("description")), sel("UTF8String"))!);

/** Global mouse position in points, origin at the top-left of the main screen. */
function mouse(): [number, number] {
  const ev = cg.symbols.CGEventCreate(null);
  const p = new Float64Array(cg.symbols.CGEventGetLocation(ev).buffer);
  cg.symbols.CFRelease(ev);
  return [p[0], p[1]];
}

const app = S.send(cls("NSApplication"), sel("sharedApplication"));
S.sendI(app, sel("setActivationPolicy:"), 1n); // accessory: no Dock icon, never steals focus
S.send(app, sel("finishLaunching"));
const screenH = new Float64Array(S.sendRectRet(S.send(cls("NSScreen"), sel("mainScreen")), sel("frame")).buffer)[3];
const cocoaRect = (x: number, y: number, w: number, h: number) => new Float64Array([x, screenH - y - h, w, h]);

let rect = { x: 100, y: 100, w: 400, h: 300 }; // points, top-left origin
const win = S.sendInitRect(S.send(cls("NSWindow"), sel("alloc")), sel("initWithContentRect:styleMask:backing:defer:"),
  cocoaRect(rect.x, rect.y, rect.w, rect.h), 0n, 2n, 0);
S.sendB(win, sel("setOpaque:"), 0);
S.sendP(win, sel("setBackgroundColor:"), S.send(cls("NSColor"), sel("clearColor")));
S.sendB(win, sel("setHasShadow:"), 0);
S.sendB(win, sel("setIgnoresMouseEvents:"), 1); // the terminal keeps the pointer: the mod's input layer drives
S.sendI(win, sel("setLevel:"), 3n);
const view = S.send(win, sel("contentView"));
S.sendB(view, sel("setWantsLayer:"), 1);

// ---- WebGPU into the window ----
const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
const surface = new Deno.UnsafeWindowSurface({ system: "cocoa", windowHandle: win, displayHandle: view,
  width: rect.w * SCALE, height: rect.h * SCALE });
const ctx = surface.getContext("webgpu") as GPUCanvasContext;
const format = navigator.gpu.getPreferredCanvasFormat();
const srgb = cg.symbols.CGColorSpaceCreateWithName(nsstr("kCGColorSpaceSRGB"));

/** wgpu's surface is opaque and nests its CAMetalLayer under the view's: make it see-through, sRGB, Retina. */
function tagLayer() {
  const subs = S.send(S.send(view, sel("layer")), sel("sublayers"));
  const n = subs ? Number(Deno.UnsafePointer.value(S.send(subs, sel("count")))) : 0;
  for (let i = 0; i < n; i++) {
    const l = S.sendI(subs, sel("objectAtIndex:"), BigInt(i));
    if (!name(l).endsWith("MetalLayer")) continue;
    S.sendB(l, sel("setOpaque:"), 0);
    S.sendP(l, sel("setColorspace:"), srgb);
    S.sendD(l, sel("setContentsScale:"), SCALE);
  }
}
let configured = { w: 0, h: 0 };
function configure() {
  const w = Math.round(rect.w * SCALE), h = Math.round(rect.h * SCALE);
  ctx.configure({ device, format, alphaMode: "opaque", width: w, height: h } as GPUCanvasConfiguration);
  tagLayer();
  configured = { w, h };
}

const module = device.createShaderModule({ code: `
  struct U { size: vec2f, cells: vec2f, ptr: vec2f, drag: f32, _p: f32 }
  @group(0) @binding(0) var<uniform> u: U;
  @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(p[i], 0, 1);
  }
  @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let cell = pos.xy / u.size * u.cells;            // position in cells
    let px = u.cells / u.size;                        // cells per pixel
    let f = fract(cell);
    let grid = f.x < px.x || f.y < px.y;              // 1-px cell grid
    let edge = cell.x < 2.0 * px.x || cell.y < 2.0 * px.y || cell.x > u.cells.x - 2.0 * px.x || cell.y > u.cells.y - 2.0 * px.y;
    let d = abs(cell - u.ptr) / px;                   // pixels from the pointer
    let cross = (d.x < 1.5 || d.y < 1.5) && length(d) > 6.0;
    let blue = vec3f(0.22, 0.53, 0.90);
    if (edge) { return vec4f(0.92, 0.41, 0.20, 1.0); }               // orange border
    if (cross) { return vec4f(select(blue, vec3f(0.99, 0.91, 0.14), u.drag > 0.5), 1.0); }
    if (grid) { return vec4f(0.55, 0.58, 0.62, 1.0) * 0.45; }        // grey grid, premultiplied
    return vec4f(0.0);
  }` });
const pipeline = device.createRenderPipeline({ layout: "auto",
  vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [{ format }] } });
const ubuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] });

// ---- calibration: mouse = origin + cell * size, per axis, least squares over recent samples ----
type Sample = { cx: number; cy: number; mx: number; my: number };
let samples: Sample[] = [];
type Cal = { ox: number; oy: number; cw: number; ch: number };
let cal = null as Cal | null; // assigned in recalibrate(): no narrowing to null
function fit1(c: number[], m: number[]): [number, number] | null {
  const n = c.length, mc = c.reduce((a, b) => a + b) / n, mm = m.reduce((a, b) => a + b) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (c[i] - mc) ** 2; sxy += (c[i] - mc) * (m[i] - mm); }
  if (sxx < 4) return null; // need a spread of at least a couple of cells
  const k = sxy / sxx;
  return [mm - k * mc, k];
}
function recalibrate() {
  const X = fit1(samples.map((s) => s.cx), samples.map((s) => s.mx));
  const Y = fit1(samples.map((s) => s.cy), samples.map((s) => s.my));
  const prev = cal;
  const cw = X?.[1] ?? prev?.cw, ch = Y?.[1] ?? prev?.ch;
  if (!cw || !ch) return;
  // origin from the newest sample, with the fitted (or previous) cell size: follows scrolls/moves
  const s = samples[samples.length - 1];
  cal = { ox: s.mx - s.cx * cw, oy: s.my - s.cy * ch, cw, ch };
}

// ---- loop ----
const mode = nsstr("kCFRunLoopDefaultMode");
const workspace = S.send(cls("NSWorkspace"), sel("sharedWorkspace"));
const frontIsGhostty = () => {
  const a = S.send(workspace, sel("frontmostApplication"));
  const id = a && S.send(a, sel("bundleIdentifier"));
  const c = id && S.send(id, sel("UTF8String"));
  return c ? Deno.UnsafePointerView.getCString(c) === "com.mitchellh.ghostty" : false;
};
let lastSeq = -1, lastEvent = 0, ev: any = null, shown = false, frames = 0;
const start = performance.now();
while ((performance.now() - start) / 1000 < SECONDS) {
  for (;;) {
    const e = S.sendEventArgs(app, sel("nextEventMatchingMask:untilDate:inMode:dequeue:"), ~0n, null, mode, 1);
    if (!e) break;
    S.sendP(app, sel("sendEvent:"), e);
  }
  try {
    const next = JSON.parse(Deno.readTextFileSync(EVENTS));
    if (next.seq !== lastSeq) {
      lastSeq = next.seq; ev = next; lastEvent = performance.now();
      if (ev.type !== "leave") {
        const [mx, my] = mouse();
        samples.push({ cx: ev.x, cy: ev.y, mx, my });
        samples = samples.slice(-40);
        recalibrate();
      }
    }
  } catch { /* no events yet, or mid-write */ }

  const active = !!ev && ev.type !== "leave" && (ev.drag || performance.now() - lastEvent < 1500);
  const want = !!cal && active && frontIsGhostty();
  if (want && cal && ev) {
    const r = { x: cal.ox, y: cal.oy, w: ev.cols * cal.cw, h: ev.rows * cal.ch };
    if (Math.abs(r.x - rect.x) + Math.abs(r.y - rect.y) + Math.abs(r.w - rect.w) + Math.abs(r.h - rect.h) > 0.25) {
      rect = r;
      S.sendRectB(win, sel("setFrame:display:"), cocoaRect(r.x, r.y, r.w, r.h), 1);
    }
    if (!shown) { S.sendP(win, sel("orderFrontRegardless"), null); shown = true; }
    if (configured.w !== Math.round(rect.w * SCALE) || configured.h !== Math.round(rect.h * SCALE)) configure();
    // live pointer position in cells, from the global mouse (no wait for the next event)
    const [mx, my] = mouse();
    device.queue.writeBuffer(ubuf, 0, new Float32Array([configured.w, configured.h, ev.cols, ev.rows,
      (mx - cal.ox) / cal.cw, (my - cal.oy) / cal.ch, ev.drag ? 1 : 0, 0]));
    let tex: GPUTexture | null = null;
    try { tex = ctx.getCurrentTexture(); } catch { configure(); }
    if (tex) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
      device.queue.submit([enc.finish()]);
      surface.present();
      frames++;
    }
  } else if (shown) {
    S.sendP(win, sel("orderOut:"), null); shown = false;
  }
  await new Promise((r) => setTimeout(r, shown ? 1 : 16));
}
console.log(`frames ${frames}; calibration ${JSON.stringify(cal)}`);
