/**
 * libc through Deno's FFI: kqueue, POSIX shared memory and thread QoS, which Deno's own APIs
 * don't offer.
 *
 * macOS (arm64 and x86-64) is what is tested. Linux is gated in but untested: stat polling
 * instead of kqueue, `shm_open` from libc (glibc ≥ 2.34) or librt, no QoS.
 */

const IS_MAC = Deno.build.os === "darwin";
const IS_LINUX = Deno.build.os === "linux";
if (!IS_MAC && !IS_LINUX) {
  throw new Error(`unsupported OS ${Deno.build.os} (macOS or Linux)`);
}

// `shm_open(name, oflag, ...)` is variadic: on Apple arm64 (only there) variadic arguments go
// on the stack, so `mode` is passed as a 9th argument, after 6 dummies that fill the integer
// registers. Elsewhere (x86-64 SysV, Linux AAPCS64) it is the 3rd register argument as usual.
const STACK_VARARGS = IS_MAC && Deno.build.arch === "aarch64";
const SHM_OPEN: { parameters: Deno.NativeType[]; result: "i32"; optional: true } = STACK_VARARGS
  ? { parameters: ["buffer", "i32", "i64", "i64", "i64", "i64", "i64", "i64", "i64"], result: "i32", optional: true }
  : { parameters: ["buffer", "i32", "i64"], result: "i32", optional: true };
const SHM = { shm_open: SHM_OPEN, shm_unlink: { parameters: ["buffer"], result: "i32", optional: true } } as const;
const KEVENT = { parameters: ["i32", "buffer", "i32", "buffer", "i32", "buffer"], result: "i32" } as const;

const libc = Deno.dlopen(IS_MAC ? "/usr/lib/libSystem.B.dylib" : "libc.so.6", {
  open: { parameters: ["buffer", "i32"], result: "i32" }, // no O_CREAT: `mode` unused
  close: { parameters: ["i32"], result: "i32" },
  read: { parameters: ["i32", "buffer", "usize"], result: "isize" },
  poll: { parameters: ["buffer", "u32", "i32"], result: "i32" },
  ...SHM,
  ftruncate: { parameters: ["i32", "i64"], result: "i32" },
  mmap: { parameters: ["pointer", "usize", "i32", "i32", "i32", "i64"], result: "pointer" },
  munmap: { parameters: ["pointer", "usize"], result: "i32" },
} as const);
const sym = libc.symbols;
// glibc < 2.34 has shm_open in librt only
const shm = sym.shm_open ? sym : Deno.dlopen("librt.so.1", SHM).symbols;
const mac = IS_MAC
  ? Deno.dlopen("/usr/lib/libSystem.B.dylib", {
    kqueue: { parameters: [], result: "i32" },
    kevent: KEVENT,
    keventAsync: { name: "kevent", ...KEVENT, nonblocking: true }, // on a worker thread
    pthread_set_qos_class_self_np: { parameters: ["u32", "i32"], result: "i32" },
  } as const).symbols
  : null;

const cstr = (s: string) => new TextEncoder().encode(s + "\0");
/** kqueue for `ControlWatcher` (macOS), unless SCIVIZ_STAT_POLL=1 forces the stat polling used
 * elsewhere (Linux), e.g. to test that path on macOS. */
const kq = mac && Deno.env.get("SCIVIZ_STAT_POLL") !== "1" ? mac : null;

// --- stat ------------------------------------------------------------------------------------

/** `(mtime, ctime, size, inode)` of `path` as a string key (cf. `os.stat`), or null if missing.
 * Times are only milliseconds: atomic replacements still change the inode. */
export function statStamp(path: string): string | null {
  try {
    return stampOf(Deno.statSync(path));
  } catch {
    return null;
  }
}

export function stampOf(s: Deno.FileInfo): string {
  return `${s.mtime?.getTime()}.${s.ctime?.getTime()}:${s.size}:${s.ino}`;
}

/** Contents of `path` and their stamp, both of the same open file (it may be replaced
 * between a `statStamp` and a read). */
export function readStamped(path: string): { text: string; stamp: string } {
  const file = Deno.openSync(path);
  try {
    const info = file.statSync();
    const buf = new Uint8Array(info.size + 1);
    let n = 0;
    for (let k; n < buf.length && (k = file.readSync(buf.subarray(n))) !== null;) {
      n += k;
    }
    return { text: new TextDecoder().decode(buf.subarray(0, n)), stamp: stampOf(info) };
  } finally {
    file.close();
  }
}

function inodeOf(stamp: string | null): string | null {
  return stamp === null ? null : stamp.slice(stamp.lastIndexOf(":") + 1);
}

// --- Waiting for the control file or stdin -----------------------------------------------------

const EVFILT_READ = -1;
const EVFILT_VNODE = -4;
const EV_ADD = 0x1;
const EV_CLEAR = 0x20;
const NOTES = 0x1 | 0x2 | 0x4 | 0x8 | 0x20; // DELETE, WRITE, EXTEND, ATTRIB, RENAME
const O_EVTONLY = 0x8000;
const KEVENT_SIZE = 32;
const POLLIN = 0x1;

function kevent(ident: number, filter: number, flags: number, fflags: number): Uint8Array {
  const ev = new Uint8Array(KEVENT_SIZE);
  const v = new DataView(ev.buffer);
  v.setBigUint64(0, BigInt(ident), true);
  v.setInt16(8, filter, true);
  v.setUint16(10, flags, true);
  v.setUint32(12, fflags, true);
  return ev;
}

/**
 * Wakes up as soon as the control file changes or stdin is readable, else after `POLL_MS`:
 * callers still compare stamps. macOS: kqueue on the file and on its directory (which sees
 * atomic replacements) and stdin. Linux: only poll(2) on stdin, the stamps are polled.
 * `wait` blocks (keeps the loop synchronous); `waitAsync` runs the same wait on a worker
 * thread (while frames are in flight, whose promises need the event loop). One at a time.
 */
export class ControlWatcher {
  static readonly POLL_MS = 4;
  private readonly kq: number = -1;
  private fileFd = -1;
  private inode: string | null = null;
  private spinUntil = 0;
  private readonly events = new Uint8Array(KEVENT_SIZE * 4);
  private readonly zero = new Uint8Array(16);
  private readonly poll = new Uint8Array(16);
  private readonly readBuf = new Uint8Array(65536);
  public stdinClosed = false;
  private pollStdin: boolean; // kqueue can't watch it (e.g. /dev/null, a regular file): poll(2)
  private readonly pollFd = new Uint8Array(8); // struct pollfd { fd 0, POLLIN }
  private pending: Promise<void> | null = null;

  constructor(private readonly path: string, private readonly watchStdin: boolean) {
    new DataView(this.pollFd.buffer).setInt16(4, POLLIN, true);
    this.pollStdin = watchStdin;
    if (kq) {
      this.kq = kq.kqueue();
      new DataView(this.poll.buffer).setBigInt64(8, BigInt(ControlWatcher.POLL_MS * 1e6), true);
      const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) || "/" : ".";
      this.watch(sym.open(cstr(dir), O_EVTONLY));
      if (watchStdin) {
        this.pollStdin = kq.kevent(this.kq, kevent(0, EVFILT_READ, EV_ADD, 0), 1, null, 0, null) < 0;
      }
    }
  }

  private watch(fd: number): void {
    if (fd >= 0) {
      kq!.kevent(this.kq, kevent(fd, EVFILT_VNODE, EV_ADD | EV_CLEAR, NOTES), 1, null, 0, null);
    }
  }

  /** macOS: watches the file anew once replaced. */
  private rewatch(stamp: string | null): void {
    const inode = inodeOf(stamp);
    if (kq && inode !== this.inode) {
      if (this.fileFd >= 0) {
        sym.close(this.fileFd); // also removes its kevent
      }
      this.fileFd = sym.open(cstr(this.path), O_EVTONLY);
      this.watch(this.fileFd);
      this.inode = inode;
    }
  }

  /** `n` events in `events` (macOS), then stdin: reads what is there, EOF closes. */
  private handle(n: number): void {
    const v = new DataView(this.events.buffer);
    let readable = this.watchStdin && this.pollStdin && sym.poll(this.pollFd, 1, 0) > 0;
    for (let i = 0; i < n; i += 1) {
      readable ||= v.getBigUint64(i * KEVENT_SIZE, true) === 0n && v.getInt16(i * KEVENT_SIZE + 8, true) === EVFILT_READ;
    }
    if (readable && Number(sym.read(0, this.readBuf, BigInt(this.readBuf.length))) <= 0) {
      this.stdinClosed = true; // EOF (or error): the client is gone
    }
  }

  /** Blocks until something happens or `POLL_MS` (not at all while spinning). */
  wait(stamp: string | null): void {
    if (this.pending) {
      throw new Error("ControlWatcher: wait() while waitAsync() is pending");
    }
    this.rewatch(stamp);
    const spinning = performance.now() < this.spinUntil; // busy: keep the core's clock up
    if (kq) {
      this.handle(kq.kevent(this.kq, null, 0, this.events, 4, spinning ? this.zero : this.poll));
    } else {
      this.pollStdinFor(spinning ? 0 : ControlWatcher.POLL_MS);
    }
  }

  /** Stat polling (Linux): sleeps up to `ms` in poll(2) on stdin (or on nothing), reads what is
   * there, EOF closes. Unguarded: also the tail of a pending `waitAsync`. */
  private pollStdinFor(ms: number): void {
    const nfds = this.watchStdin ? 1 : 0;
    const ready = sym.poll(this.pollFd, nfds, ms) > 0;
    if (ready && Number(sym.read(0, this.readBuf, BigInt(this.readBuf.length))) <= 0) {
      this.stdinClosed = true;
    }
  }

  /** `wait` without blocking the event loop (at most `POLL_MS`); the pending one if any. */
  waitAsync(stamp: string | null): Promise<void> {
    if (!this.pending) {
      this.rewatch(stamp);
      const done = kq
        ? kq.keventAsync(this.kq, null, 0, this.events, 4, this.poll).then((n) => this.handle(n))
        : new Promise<void>((resolve) => setTimeout(resolve, ControlWatcher.POLL_MS)).then(() => this.pollStdinFor(0));
      this.pending = done.finally(() => (this.pending = null));
    }
    return this.pending;
  }

  /** Once no `waitAsync` is pending (before `wait`). */
  settled(): Promise<void> | null {
    return this.pending;
  }

  stayAwake(seconds: number): void {
    this.spinUntil = performance.now() + seconds * 1000;
  }
}

/** macOS: run this (mostly idle) thread as user-interactive, so it isn't parked on the
 * efficiency cores at low clocks between frames (QOS_CLASS_USER_INTERACTIVE). */
export function setInteractiveQos(): void {
  mac?.pthread_set_qos_class_self_np(0x21, 0);
}

// --- POSIX shared memory -----------------------------------------------------------------------

const O_RDONLY = 0x0;
const O_RDWR = 0x2;
const O_CREAT = IS_MAC ? 0x200 : 0o100;
const O_EXCL = IS_MAC ? 0x800 : 0o200;
const PROT_RW = 0x3;
const MAP_SHARED = 0x1;

function shmOpen(name: string, flags: number): number {
  const args = STACK_VARARGS ? [0n, 0n, 0n, 0n, 0n, 0n, 0o600n] : [0o600n];
  // deno-lint-ignore no-explicit-any
  return (shm.shm_open as any)(cstr(name), flags, ...args);
}

/** A fresh shared-memory object of `size` bytes, mapped: write into `bytes`, then `close()`
 * (unmaps; the object stays until `shm_unlink`). */
export class ShmFrame {
  readonly bytes: Uint8Array;
  private ptr: Deno.PointerValue;

  constructor(readonly name: string, readonly size: number) {
    shmUnlink(name); // stale one of a previous process with this pid
    const fd = shmOpen(name, O_RDWR | O_CREAT | O_EXCL);
    if (fd < 0) {
      throw new Error(`shm_open ${name} failed`);
    }
    try {
      if (sym.ftruncate(fd, BigInt(size)) !== 0) {
        throw new Error(`ftruncate ${name} failed`);
      }
      this.ptr = sym.mmap(null, BigInt(size), PROT_RW, MAP_SHARED, fd, 0n);
      if (this.ptr === null || Deno.UnsafePointer.value(this.ptr) === 0xffffffffffffffffn) {
        throw new Error(`mmap ${name} failed`);
      }
    } catch (err) {
      sym.close(fd);
      shmUnlink(name);
      throw err;
    }
    sym.close(fd); // the mapping keeps it
    this.bytes = new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(this.ptr!, size));
  }

  close(): void {
    if (this.ptr) {
      sym.munmap(this.ptr, BigInt(this.size));
      this.ptr = null;
    }
  }
}

export function shmUnlink(name: string): boolean {
  return shm.shm_unlink!(cstr(name)) === 0;
}

/** Whether the object `name` exists (opened read-only, closed again). */
export function shmExists(name: string): boolean {
  const fd = shmOpen(name, O_RDONLY);
  if (fd >= 0) {
    sym.close(fd);
  }
  return fd >= 0;
}
