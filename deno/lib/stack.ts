/**
 * Slices of a file-backed volume stack (volume.ts), read on demand: slice k at its offset (short
 * reads continued; the file's version checked on its descriptor before and after), ±∞ rejected,
 * NaN uploaded as 0 with a validity mask, the value ranges (colour domains). Read and prepared off
 * the main thread by a worker (this module, loaded as one): the latest prefetch replaces an older
 * one still waiting, requests go first.
 */

export interface StackFile {
  path: string;
  version: string; // fileVersion of what was read
  offset: number; // of the first value
  dtype: "f32" | "f64";
  shape: [number, number, number]; // z, y, x
  slices: number;
}

/** A slice, ready to upload: NaN as 0 in `values`, invalid in `valid` (absent: all valid). */
export interface SliceData {
  values: Float32Array;
  valid?: Uint8Array; // 255 valid, 0 NaN
  stats: Float64Array; // min, max of the finite values, of those > 0, of those ≥ 0 (+∞, −∞: none)
  ms: number;
}

/** Everything a read depends on: the file, its version and how its bytes are laid out. */
export const stackKey = (st: StackFile) => `${st.path}|${st.version}|${st.dtype}|${st.offset}|${st.shape}|${st.slices}`;
export const fileVersion = (info: Deno.FileInfo) => `${info.mtime?.getTime() ?? 0}:${info.size}:${info.ino ?? 0}`;

/** `bytes.length` bytes at `offset`, short reads continued; the end of the file throws. */
export function readAt(file: Deno.FsFile, offset: number, bytes: Uint8Array): void {
  file.seekSync(offset, Deno.SeekMode.Start);
  for (let got = 0; got < bytes.length;) {
    const n = file.readSync(bytes.subarray(got));
    if (!n) throw new Error(`unexpected end of file at byte ${offset + got}`);
    got += n;
  }
}

/** Slice `k` as f32 (NaN kept; ±∞ rejected by `prepareSlice`), from `file` (else opened for it),
 * which must still be `st.version`. */
export function readSlice(st: StackFile, k: number, file?: Deno.FsFile): Float32Array {
  const f = file ?? Deno.openSync(st.path);
  try {
    const same = () => {
      if (fileVersion(f.statSync()) !== st.version) throw new Error(`file volume ${st.path} changed while reading`);
    };
    same();
    const [nz, ny, nx] = st.shape, n = nx * ny * nz, size = st.dtype === "f64" ? 8 : 4;
    const buf = new ArrayBuffer(n * size);
    readAt(f, st.offset + k * n * size, new Uint8Array(buf));
    same();
    if (size === 4) return new Float32Array(buf);
    const wide = new Float64Array(buf), values = Float32Array.from(wide);
    for (let i = 0; i < n; i += 1) if (Math.abs(values[i]) === Infinity) throw notFinite(st, k * n + i, wide[i]);
    return values;
  } finally {
    if (!file) f.close();
  }
}

const notFinite = (st: StackFile, i: number, v: number) =>
  new Error(`file volume ${st.path}: value ${v} at flat index ${i} is not a finite f32 (|v| ≤ 3.4e38); use NaN for missing values`);

/** Value ranges (`SliceData.stats`) from the min and max of the valid values; those over the
 * positive and non-negative ones from `values` when they differ (another pass). */
function ranges(lo: number, hi: number, values: Float32Array, valid?: Uint8Array): Float64Array {
  let plo = lo, zlo = lo;
  if (lo <= 0 && hi >= 0) {
    [plo, zlo] = [Infinity, Infinity];
    for (let i = 0; i < values.length; i += 1) {
      const v = values[i];
      if (valid && !valid[i]) continue;
      if (v >= 0 && v < zlo) zlo = v; // NaN compares false
      if (v > 0 && v < plo) plo = v;
    }
  }
  return Float64Array.of(lo, hi, hi > 0 ? plo : Infinity, hi > 0 ? hi : -Infinity, hi >= 0 ? zlo : Infinity, hi >= 0 ? hi : -Infinity);
}

/** Prepares slice `k` read by `readSlice`, in place: NaN to 0 and invalid, value ranges; ±∞ (or
 * beyond f32) rejected, NaN allowed (transparent). */
export function prepareSlice(values: Float32Array, st: StackFile, k: number, t0 = performance.now()): SliceData {
  const valid = new Uint8Array(values.length);
  let lo = Infinity, hi = -Infinity, invalid = 0;
  for (let i = 0; i < values.length; i += 1) { // branchless but for the range: the fastest found
    const v = values[i];
    const ok = v === v;
    valid[i] = ok ? 255 : 0;
    values[i] = ok ? v : 0;
    invalid += ok ? 0 : 1;
    if (ok) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (lo === -Infinity || hi === Infinity) {
    const i = values.findIndex((v) => Math.abs(v) === Infinity);
    throw notFinite(st, k * values.length + i, values[i]);
  }
  const stats = ranges(lo, hi, values, invalid ? valid : undefined);
  return { values, valid: invalid ? valid : undefined, stats, ms: performance.now() - t0 };
}

/** Slice `k` read by `readSlice`, halved along each axis (even sizes): each value the mean of
 * its 8 voxels' valid ones, valid if at least 4 are, as `prepareSlice` otherwise (value ranges of
 * the voxels themselves). A cheaper stand-in while dragging through the slices. */
export function downsample(src: Float32Array, st: StackFile, k: number, t0 = performance.now()): SliceData {
  const [nz, ny, nx] = st.shape;
  const [mz, my, mx] = [nz >> 1, ny >> 1, nx >> 1];
  const values = new Float32Array(mx * my * mz);
  const offsets = Int32Array.from([0, 1, nx, nx + 1].flatMap((o) => [o, o + nx * ny]));
  let valid: Uint8Array | undefined;
  let lo = Infinity, hi = -Infinity;
  for (let z = 0, o = 0; z < mz; z += 1) {
    for (let y = 0; y < my; y += 1) {
      for (let x = 0, i0 = (2 * z * ny + 2 * y) * nx; x < mx; x += 1, o += 1, i0 += 2) {
        let sum = 0, c = 0;
        for (let j = 0; j < 8; j += 1) {
          const v = src[i0 + offsets[j]];
          if (v !== v) continue;
          sum += v;
          c += 1;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (c >= 4) values[o] = sum / c;
        else {
          valid ??= new Uint8Array(values.length).fill(255);
          valid[o] = 0;
        }
      }
    }
  }
  if (lo === -Infinity || hi === Infinity) {
    const i = src.findIndex((v) => Math.abs(v) === Infinity);
    throw notFinite(st, k * src.length + i, src[i]);
  }
  return { values, valid, stats: ranges(lo, hi, src), ms: performance.now() - t0 };
}

/** Value ranges of the whole stack (`SliceData.stats`), one slice at a time. */
export function stackStats(st: StackFile, file?: Deno.FsFile): Float64Array {
  const all = Float64Array.of(Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity);
  for (let k = 0; k < st.slices; k += 1) {
    const { stats } = prepareSlice(readSlice(st, k, file), st, k);
    for (let i = 0; i < 6; i += 2) [all[i], all[i + 1]] = [Math.min(all[i], stats[i]), Math.max(all[i + 1], stats[i + 1])];
  }
  return all;
}

type Job = { op: "slice" | "preview" | "stats"; st: StackFile; k: number; key: string; promise: Promise<never>; resolve: (v: never) => void; reject: (e: Error) => void };

/** The worker's client: one job at a time, requests (in order) before the latest prefetch; a
 * prefetch still waiting is dropped by a newer one (rejected), or becomes a request when one asks
 * for its slice (a job is shared by everyone asking for it). The worker is ended when idle a while
 * (it would keep the process up). */
export class StackReader {
  private worker?: Worker;
  private running?: Job;
  private queue: Job[] = [];
  private prefetchJob?: Job;
  private idle?: ReturnType<typeof setTimeout>;

  /** Slice `k` (`preview`: `downsample`d). */
  slice(st: StackFile, k: number, { prefetch = false, preview = false } = {}): Promise<SliceData> {
    return this.push(preview ? "preview" : "slice", st, k, prefetch);
  }

  /** A prefetch of slice `k` still waiting becomes a request. */
  promote(st: StackFile, k: number, preview = false): void {
    const op = preview ? "preview" : "slice";
    if (this.prefetchJob?.key === `${op}|${stackKey(st)}|${k}`) void this.push(op, st, k, false);
  }

  stats(st: StackFile): Promise<Float64Array> {
    return this.push("stats", st, -1, false);
  }

  private push<T>(op: Job["op"], st: StackFile, k: number, prefetch: boolean): Promise<T> {
    const key = `${op}|${stackKey(st)}|${k}`;
    const known = [this.running, ...this.queue].find((j) => j?.key === key);
    if (known) return known.promise as Promise<T>;
    if (this.prefetchJob?.key === key) {
      const job = this.prefetchJob;
      if (!prefetch) {
        this.prefetchJob = undefined;
        this.queue.push(job);
      }
      return job.promise as Promise<T>;
    }
    const job = { op, st, k, key } as Job;
    job.promise = new Promise<never>((resolve, reject) => Object.assign(job, { resolve, reject }));
    if (prefetch) {
      this.prefetchJob?.reject(new Error("prefetch superseded"));
      this.prefetchJob = job;
    } else this.queue.push(job);
    this.next();
    return job.promise as Promise<T>;
  }

  private next(): void {
    if (this.running) return;
    const job = this.queue.shift() ?? this.prefetchJob;
    if (job === this.prefetchJob) this.prefetchJob = undefined;
    if (!job) {
      this.idle = setTimeout(() => { this.worker?.terminate(); this.worker = undefined; }, 1000);
      return;
    }
    clearTimeout(this.idle);
    this.running = job;
    const w = this.worker ??= new Worker(import.meta.url, { type: "module" });
    w.onmessage = (e: MessageEvent<{ result?: unknown; error?: string }>) => {
      this.running = undefined;
      if (e.data.error !== undefined) job.reject(new Error(e.data.error));
      else job.resolve(e.data.result as never);
      this.next();
    };
    w.onerror = (e) => {
      e.preventDefault();
      this.worker?.terminate();
      this.worker = undefined;
      this.running = undefined;
      job.reject(new Error(`stack reader: ${e.message}`));
      this.next();
    };
    w.postMessage({ op: job.op, st: job.st, k: job.k });
  }
}

// The worker: keeps the file open while its version holds
if ("WorkerGlobalScope" in globalThis) {
  const scope = globalThis as unknown as { onmessage: unknown; postMessage(m: unknown, transfer?: Transferable[]): void };
  let open: { path: string; version: string; file: Deno.FsFile } | undefined;
  scope.onmessage = (e: MessageEvent<{ op: Job["op"]; st: StackFile; k: number }>) => {
    const { op, st, k } = e.data;
    try {
      if (open && (open.path !== st.path || open.version !== st.version)) {
        open.file.close();
        open = undefined;
      }
      open ??= { path: st.path, version: st.version, file: Deno.openSync(st.path) };
      if (op === "stats") {
        scope.postMessage({ result: stackStats(st, open.file) });
        return;
      }
      const t0 = performance.now();
      const d = (op === "preview" ? downsample : prepareSlice)(readSlice(st, k, open.file), st, k, t0);
      scope.postMessage({ result: d }, [d.values.buffer, ...(d.valid ? [d.valid.buffer] : [])]);
    } catch (err) {
      open?.file.close();
      open = undefined;
      scope.postMessage({ error: err instanceof Error ? err.message : String(err) });
    }
  };
}
