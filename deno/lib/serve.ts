/**
 * Command line and `serve` protocol shared by every renderer (surface.ts, plot1d.ts, line.ts,
 * curve.ts, complex.ts, volume.ts):
 *
 *   png CONTROL.json -o out.png
 *   serve CONTROL.json OUTDIR [--ignore-stdin] [--spin SECONDS] [--inflight K]
 *
 * `serve` prints `ready` once the renderer is created: its device, font and fixed pipelines
 * (those that depend on the data are built with the first frame, before it), then renders whenever CONTROL.json changes (kqueue on macOS, else stat polling every 4 ms;
 * busy-polls for `--spin` seconds after each frame). Up to K frames (3) are in flight: each is
 * submitted with its readback and `mapAsync` started, and the next control is taken meanwhile
 * (the latest one: changes in between are coalesced, never queued). Frames are published in
 * order, each as a fresh POSIX shared-memory object `/scv-<pid>-<generation>` (RGBA8, straight
 * alpha, top row first), then OUTDIR/state.json is written atomically and
 *
 *   frame <generation> <width> <height> <shm name or -> <rev>
 *
 * printed, `rev` being the control's `rev` (integer, increasing; 0 if absent) this frame
 * rendered. A control whose `rev` is below the last applied one is stale: ignored. Requests,
 * each answered once, by the first frame whose rev is at or after theirs:
 *
 * - `"snapshot": {"path": P, "rev": R}` (or `"snapshot": P`, R = 0): writes the PNG at P
 *   (atomically: a temporary file renamed), then prints `snapshot <R> <P>`, once per distinct
 *   (P, R); if writing fails, logs it and prints `snapshotfailed <R> <P>` instead (and keeps
 *   serving). Pending until then, whatever arrives.
 * - `"frameFileOnce": R`: writes OUTDIR/frame.rgba (atomically), then prints `framefile <R>`.
 *   (`"frameFile": true` still writes it for every frame, before the `frame` line.)
 *
 * A control out of bounds (lib/controls.ts: sizes up to 4096 px, sample counts, volume sizes up
 * to 256³, non-finite numbers, ...) or that the renderer rejects is not rendered: one line
 *
 *   controlerror <rev> <message>
 *
 * on stderr (the message on one line), and the last good frame stands; serving goes on.
 *
 * The reader (the terminal) unlinks each object once read; here, objects older than 2 s that
 * still exist are unlinked, at most 32 are left outstanding (the oldest is unlinked, and
 * logged), and all are unlinked on exit: `"quit": true`, stdin closing (unless
 * `--ignore-stdin`), SIGINT/SIGTERM, or an error.
 */

import { validateControl } from "./controls.ts";
import { missingSinceFrame } from "./overlay.ts";
import { encodePng } from "./png.ts";
import { ControlWatcher, readStamped, setInteractiveQos, ShmFrame, shmExists, shmUnlink, statStamp } from "./sys.ts";

export type Control = Record<string, unknown>;

export interface Frame {
  width: number;
  height: number;
  rgba: Uint8Array; // straight alpha, top row first
  hover: unknown;
}

/** A submitted frame: `done` settles once its readback is mapped, then `read` copies it into
 * `alloc(width × height × 4)` and frees its readback slot. */
export interface Pending {
  done: Promise<unknown>;
  read(alloc: (bytes: number) => Uint8Array): Frame;
}

export interface Renderer<C extends Control> {
  /** Submits a frame of `ctl`, read back into `slot` (< K, free: its last frame was read). */
  submit(ctl: C, slot: number): Promise<Pending>;
  /** state.json of the frame last submitted (called right after `submit`) */
  state(ctl: C, generation: number, hover: unknown, ms: number): unknown;
}

export interface Tool<C extends Control> {
  name: string;
  defaults: C;
  create(): Promise<Renderer<C>>;
  onPng?(ctl: C, frame: Frame, state: unknown): void;
  commands?: Record<string, (args: string[]) => Promise<void>>;
}

const SHM_MAX_AGE_MS = 2000;
const SHM_MAX_OUTSTANDING = 32;
const YIELD_MS = 100; // while busy-polling: let signals in this often

/**
 * The busy-spin trick, extended: Deno's `mapAsync` sleeps 10 ms on the main thread before it
 * resolves (ext/webgpu/buffer.rs), long enough for the clocks to drop mid-frame. A worker spins
 * while a frame is rendered and for `--spin` seconds after it, and sleeps otherwise.
 */
class Spinner {
  private readonly state = new Int32Array(new SharedArrayBuffer(4));
  private readonly worker = new Worker(new URL("./spinner_worker.ts", import.meta.url), { type: "module" });

  constructor() {
    this.worker.postMessage(this.state.buffer);
  }

  set(on: boolean): void {
    if (Atomics.load(this.state, 0) !== Number(on)) {
      Atomics.store(this.state, 0, Number(on));
      Atomics.notify(this.state, 0);
    }
  }

  stop(): void {
    Atomics.store(this.state, 0, 2);
    Atomics.notify(this.state, 0);
  }
}

export function loadControl<C extends Control>(path: string, defaults: C): C {
  return parseControl(Deno.readTextFileSync(path), defaults);
}

const parseControl = <C extends Control>(text: string, defaults: C): C => ({ ...defaults, ...JSON.parse(text || "{}") });

export function writeAtomic(path: string, data: Uint8Array): void {
  const slash = path.lastIndexOf("/");
  const tmp = `${path.slice(0, slash + 1)}.${path.slice(slash + 1)}.${Deno.pid}.tmp`;
  Deno.writeFileSync(tmp, data);
  try {
    Deno.renameSync(tmp, path);
  } catch (err) {
    try {
      Deno.removeSync(tmp);
    } catch { /* already gone */ }
    throw err;
  }
}

const encoder = new TextEncoder();
const stdout = (line: string) => Deno.stdout.writeSync(encoder.encode(line + "\n"));
const stderr = (line: string) => Deno.stderr.writeSync(encoder.encode(line + "\n"));

function usage(name: string): never {
  stderr(`usage: deno run -A ${name} png CONTROL.json [-o out.png]\n` +
    `       deno run -A ${name} serve CONTROL.json OUTDIR [--ignore-stdin] [--spin SECONDS] [--inflight K]`);
  Deno.exit(2);
}

export async function main<C extends Control>(tool: Tool<C>, args = Deno.args): Promise<void> {
  const [cmd, ...rest] = args;
  if (cmd === "png") {
    const control = rest.find((a, i) => !a.startsWith("-") && rest[i - 1] !== "-o" && rest[i - 1] !== "--output");
    const oi = rest.findIndex((a) => a === "-o" || a === "--output");
    const output = oi >= 0 ? rest[oi + 1] : "out.png";
    if (!control) usage(tool.name);
    await cmdPng(tool, control, output);
  } else if (cmd === "serve") {
    const pos = rest.filter((a, i) => !a.startsWith("--") && rest[i - 1] !== "--spin" && rest[i - 1] !== "--inflight");
    const opt = (flag: string, value: number) => (rest.includes(flag) ? Number(rest[rest.indexOf(flag) + 1]) : value);
    const spin = opt("--spin", 0.5);
    const inflight = opt("--inflight", 3);
    if (pos.length < 2 || !Number.isFinite(spin) || !(Number.isInteger(inflight) && inflight >= 1)) usage(tool.name);
    await cmdServe(tool, pos[0], pos[1], rest.includes("--ignore-stdin"), spin, inflight);
  } else if (cmd && tool.commands?.[cmd]) {
    await tool.commands[cmd](rest);
  } else {
    usage(tool.name);
  }
}

/** One frame, waited for: the frame and its state.json (generation 1). */
export async function renderOnce<C extends Control>(
  renderer: Renderer<C>,
  ctl: C,
  alloc = (n: number) => new Uint8Array(n),
): Promise<{ frame: Frame; state: unknown }> {
  const start = performance.now();
  validateControl(ctl);
  const pending = await renderer.submit(ctl, 0);
  const glyphs = takeMissing();
  const state = renderer.state(ctl, 1, null, 0);
  await pending.done;
  const frame = pending.read(alloc);
  return { frame, state: withFrame(state, 1, frame.hover, performance.now() - start, glyphs) };
}

/** Characters the font atlas lacked (drawn as "?") since the last call: taken right after a
 * submit, they belong to that frame (several can be in flight), whose state reports them. */
function takeMissing(): string {
  const s = [...missingSinceFrame].join("");
  missingSinceFrame.clear();
  return s;
}

function withFrame(state: unknown, generation: number, hover: unknown, ms: number, glyphs = ""): unknown {
  const missing = glyphs ? { missingGlyphs: glyphs } : {};
  return { ...(state as object), ...missing, generation, hover, renderMs: Math.round(ms * 1000) / 1000 };
}

async function cmdPng<C extends Control>(tool: Tool<C>, control: string, output: string): Promise<void> {
  const ctl = loadControl(control, tool.defaults);
  const renderer = await tool.create();
  const { frame, state } = await renderOnce(renderer, ctl);
  Deno.writeFileSync(output, await encodePng(frame.rgba, frame.width, frame.height));
  stdout(`wrote ${output} (${frame.width}x${frame.height})`);
  tool.onPng?.(ctl, frame, state);
}


interface Snapshot {
  path: string;
  rev: number;
}

/** `{path, rev}`, or a plain path (rev 0), or null. */
function snapshotOf(value: unknown): Snapshot | null {
  if (typeof value === "string") {
    return value ? { path: value, rev: 0 } : null;
  }
  const { path, rev } = (value ?? {}) as Partial<Snapshot>;
  return typeof path === "string" && path ? { path, rev: Number(rev) || 0 } : null;
}

const revOf = (ctl: Control): number | null => (Number.isInteger(ctl.rev) ? Number(ctl.rev) : null);

interface Flight<C> {
  ctl: C;
  rev: number;
  start: number;
  pending: Pending;
  state: unknown;
  glyphs: string; // characters the font lacked, met while submitting this frame
  ready: Promise<void>; // `pending.done` settled (never rejects)
  settled: boolean;
  error?: unknown;
}

async function cmdServe<C extends Control>(
  tool: Tool<C>,
  control: string,
  outdir: string,
  ignoreStdin: boolean,
  spin: number,
  inflight: number,
): Promise<void> {
  setInteractiveQos();
  Deno.mkdirSync(outdir, { recursive: true });
  const renderer = await tool.create();
  stdout("ready"); // device, font and fixed pipelines initialised
  const watcher = new ControlWatcher(control, !ignoreStdin);
  const spinner = spin > 0 ? new Spinner() : undefined;
  const report = (err: unknown) => stderr(`${tool.name}: ${err instanceof Error ? `${err.name}: ${err.message}` : err}`);
  const controlError = (rev: number, err: unknown) =>
    stderr(`controlerror ${rev} ${(err instanceof Error ? err.message : String(err)).replace(/\s+/g, " ").trim()}`);
  let stopped = false;
  const stop = () => (stopped = true);
  for (const sig of ["SIGINT", "SIGTERM"] as const) {
    Deno.addSignalListener(sig, stop);
  }

  // Shared-memory objects not known to be consumed, oldest first
  const outstanding: { name: string; time: number }[] = [];
  const sweep = (now: number) => {
    while (outstanding.length && now - outstanding[0].time > SHM_MAX_AGE_MS) {
      shmUnlink(outstanding.shift()!.name); // ENOENT if consumed since
    }
    if (outstanding.length > SHM_MAX_OUTSTANDING) { // forget those the reader unlinked
      outstanding.splice(0, outstanding.length, ...outstanding.filter(({ name }) => shmExists(name)));
    }
    while (outstanding.length > SHM_MAX_OUTSTANDING) {
      const { name, time } = outstanding.shift()!;
      if (shmUnlink(name)) {
        stderr(`${tool.name}: unlinked unread ${name} (${(now - time).toFixed(0)} ms old, over ${SHM_MAX_OUTSTANDING} outstanding)`);
      }
    }
  };

  // Requests: answered once, by the first frame at or after their rev
  const snapshots = new Map<string, Snapshot>();
  const snapshotsDone = new Set<string>();
  const frameFiles = new Set<number>();
  const frameFilesDone = new Set<number>();

  let generation = 0;
  const publish = async (flight: Flight<C>) => {
    if (flight.error !== undefined) {
      report(flight.error);
      return;
    }
    const name = `/scv-${Deno.pid}-${generation + 1}`; // at most 30 characters on macOS
    let shm: ShmFrame | undefined;
    let frame: Frame;
    try {
      frame = flight.pending.read((bytes) => (shm = new ShmFrame(name, bytes)).bytes);
    } catch (err) {
      report(err);
      if (shm) {
        shm.close();
        shmUnlink(name);
      }
      return;
    }
    const now = performance.now();
    generation += 1;
    try {
      const { ctl, rev } = flight;
      if (ctl.frameFile) {
        writeAtomic(`${outdir}/frame.rgba`, frame.rgba);
      }
      const state = withFrame(flight.state, generation, frame.hover, now - flight.start, flight.glyphs);
      writeAtomic(`${outdir}/state.json`, encoder.encode(JSON.stringify(state)));
      outstanding.push({ name, time: now });
      sweep(now);
      stdout(`frame ${generation} ${frame.width} ${frame.height} ${shm ? name : "-"} ${rev}`);
      const due = [...frameFiles].filter((r) => r <= rev);
      if (due.length) {
        writeAtomic(`${outdir}/frame.rgba`, frame.rgba);
        for (const r of due) {
          frameFiles.delete(r);
          frameFilesDone.add(r);
          stdout(`framefile ${r}`);
        }
      }
      for (const [key, snap] of snapshots) {
        if (snap.rev <= rev) {
          snapshots.delete(key);
          snapshotsDone.add(key); // answered once, written or failed
          try {
            writeAtomic(snap.path, await encodePng(frame.rgba, frame.width, frame.height));
            stdout(`snapshot ${snap.rev} ${snap.path}`);
          } catch (err) { // e.g. a missing directory: report it, keep serving
            stderr(`${tool.name}: snapshot rev ${snap.rev} to ${snap.path} failed: ${err instanceof Error ? err.message : err}`);
            stdout(`snapshotfailed ${snap.rev} ${snap.path}`);
          }
        }
      }
    } finally {
      shm?.close(); // `frame.rgba` maps it: only now
    }
    watcher.stayAwake(spin);
    spinUntil = performance.now() + spin * 1000;
    if (Deno.env.get("RENDER_DEBUG")) {
      stderr(`frame ${generation} rev ${flight.rev}: ${(now - flight.start).toFixed(2)} ms`);
    }
  };

  const flights: Flight<C>[] = []; // submitted, in order
  let submitted = 0;
  let last: string | null = null;
  let lastRev = -Infinity;
  let warned = false;
  let spinUntil = 0;
  let yielded = performance.now();
  try {
    while (!stopped && !watcher.stdinClosed) {
      while (flights[0]?.settled) {
        await publish(flights.shift()!);
      }
      sweep(performance.now());
      const stamp = statStamp(control);
      if (stamp !== null && stamp !== last && flights.length < inflight) {
        let ctl: C | null = null;
        let read: string = stamp;
        try {
          const { text, stamp } = readStamped(control); // the stamp of what is read
          read = stamp;
          ctl = parseControl(text, tool.defaults);
          warned = false;
        } catch (err) { // e.g. caught mid-write: try again
          if (!warned) {
            stderr(`${tool.name}: bad control file: ${err}`);
            warned = true;
          }
        }
        if (!ctl) {
          await (flights.length ? Promise.race([flights[0].ready, watcher.waitAsync(null)]) : wait(null));
          continue;
        }
        last = read;
        const rev = revOf(ctl);
        if (rev !== null && rev < lastRev) {
          continue; // stale
        }
        lastRev = rev ?? lastRev;
        if (ctl.quit) {
          break;
        }
        try {
          validateControl(ctl);
        } catch (err) { // out of bounds: nothing allocated, the last good frame stands
          controlError(rev ?? 0, err);
          continue;
        }
        const snap = snapshotOf(ctl.snapshot);
        const key = snap && `${snap.rev}\0${snap.path}`;
        if (snap && !snapshotsDone.has(key!)) {
          snapshots.set(key!, snap);
        }
        const once = ctl.frameFileOnce;
        if (typeof once === "number" && Number.isFinite(once) && !frameFilesDone.has(once)) {
          frameFiles.add(once);
        }

        spinner?.set(true);
        const start = performance.now();
        let pending: Pending;
        try {
          pending = await renderer.submit(ctl, submitted % inflight);
        } catch (err) { // bad parameters: report, keep serving
          controlError(rev ?? 0, err);
          continue;
        }
        const flight: Flight<C> = {
          ctl,
          rev: rev ?? 0,
          start,
          pending,
          glyphs: takeMissing(),
          state: renderer.state(ctl, 0, null, 0),
          settled: false,
          ready: undefined!,
        };
        flight.ready = pending.done.then(
          () => void (flight.settled = true),
          (err) => void Object.assign(flight, { settled: true, error: err ?? "readback failed" }),
        );
        flights.push(flight);
        submitted += 1;
        continue;
      }
      if (flights.length) { // the event loop resolves them: wait there
        await Promise.race(flights.length < inflight ? [flights[0].ready, watcher.waitAsync(stamp)] : [flights[0].ready]);
        continue;
      }
      await wait(stamp);
    }
  } finally {
    spinner?.stop();
    for (const { name } of outstanding) {
      shmUnlink(name);
    }
    for (const sig of ["SIGINT", "SIGTERM"] as const) {
      Deno.removeSignalListener(sig, stop);
    }
  }

  /** Nothing in flight: blocks in `watcher.wait` (keeps the loop synchronous), yielding to the
   * event loop when idle (and now and then while busy-polling) to let signals in. */
  async function wait(stamp: string | null): Promise<void> {
    const pending = watcher.settled();
    if (pending) {
      await pending;
      return;
    }
    const now = performance.now();
    if (now >= spinUntil) {
      spinner?.set(false);
    }
    if (now >= spinUntil || now - yielded > YIELD_MS) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      yielded = performance.now();
    }
    watcher.wait(stamp);
  }
}
