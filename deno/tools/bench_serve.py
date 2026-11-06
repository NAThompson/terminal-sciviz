# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""`serve` of surface.ts / plot1d.ts / volume.ts (BENCH_CTL: JSON merged into each control), driven as the terminal does (cf. lib/serve.ts): writes
CONTROL.json atomically with an increasing `rev`, reads `frame` lines on a thread, maps each
frame's shm object and unlinks it.

    uv run deno/tools/bench_serve.py latency    surface|plot1d|line|curve [--size=1032x645] [--inflight=3] [--n=60] [--hz=20]
    uv run deno/tools/bench_serve.py throughput surface|plot1d|line|curve [--size=...] [--inflight=3] [--hz=120] [--seconds=5]
    uv run deno/tools/bench_serve.py matrix     [--runs=3] [--sizes=2048x1280,1032x645] [--ks=1,2,3,4]
    uv run deno/tools/bench_serve.py check      surface|plot1d|line|curve|complex|volume  # protocol and cleanup checks
    uv run deno/tools/bench_serve.py coldstart  [surface|plot1d|...] [--runs=3]   # spawn -> `ready`, -> first frame

latency: write → its frame line (matched by rev), at a gentle rate. throughput: distinct frames
per second delivered, revs skipped (coalesced), and each frame's input age (frame line − write
of the rev it rendered). `--orig=DIR` runs another copy of deno/ (no revs: matched in order).
"""

import _posixshmem
import json
import mmap
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DENO = Path(__file__).resolve().parents[1]


def shm_exists(name):
    try:
        fd = _posixshmem.shm_open(name, os.O_RDONLY)
    except FileNotFoundError:
        return False
    os.close(fd)
    return True


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


class Server:
    def __init__(self, tool, w, h, inflight=3, spin=None, root=DENO, consume=True, stdin=True):
        self.tool, self.w, self.h = tool, w, h
        self.tmp = Path(tempfile.mkdtemp())
        self.control, self.outdir = self.tmp / "control.json", self.tmp / "out"
        self.rev = 0
        self.writes = {}  # rev -> time
        self.lines = []  # (time, fields)
        self.errs = []  # (time, stderr line)
        self.cond = threading.Condition()
        self.consume = consume  # map + unlink each frame's object, as the terminal
        self.keep = set()  # names not to consume (left for the server)
        self.data = {}  # generation -> frame bytes, the last few
        self.names = []
        self.write({})
        self.t_spawn = time.perf_counter()
        cmd = ["deno", "run", "-A", str(root / f"{tool}.ts"), "serve", str(self.control), str(self.outdir)]
        if inflight is not None:
            cmd += ["--inflight", str(inflight)]
        if spin is not None:
            cmd += ["--spin", str(spin)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=root)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        threading.Thread(target=self.read_err, daemon=True).start()
        self.wait_lines(lambda ls: any(f[0] == "frame" for _, f in ls), 30)

    def ctl(self, i):
        if self.tool in ("surface", "curve", "volume"):
            return {"azimuth": 35 + 0.7 * i, "elevation": 25 + (i % 5)}
        return {"x0": 0.1 + 0.0005 * i, "x1": 0.6 + 0.0005 * i}

    def write(self, extra, rev=None):
        self.rev = self.rev + 1 if rev is None else rev
        base = json.loads(os.environ.get("BENCH_CTL", "{}"))  # e.g. a gallery entry's control
        ctl = {"width": self.w, "height": self.h, "rev": self.rev, **base, **self.ctl(self.rev), **extra}
        t = self.control.with_suffix(".tmp")
        t.write_text(json.dumps(ctl))
        os.replace(t, self.control)
        self.writes[self.rev] = time.perf_counter()
        return self.rev

    def read(self):
        for line in self.proc.stdout:
            t = time.perf_counter()
            f = line.split()
            if f and f[0] == "frame" and len(f) > 4 and f[4] != "-":
                self.names.append(f[4])
                if self.consume and f[4] not in self.keep:
                    try:  # gone if the server exited (or it is over 2 s old)
                        fd = _posixshmem.shm_open(f[4], os.O_RDONLY)
                        with mmap.mmap(fd, int(f[2]) * int(f[3]) * 4, prot=mmap.PROT_READ) as m:
                            self.data[int(f[1])] = bytes(m)
                            self.data.pop(int(f[1]) - 8, None)
                        os.close(fd)
                        _posixshmem.shm_unlink(f[4])
                    except FileNotFoundError:
                        pass
            with self.cond:
                self.lines.append((t, f))
                self.cond.notify_all()

    def read_err(self):
        for line in self.proc.stderr:
            sys.stderr.write(line)
            with self.cond:
                self.errs.append((time.perf_counter(), line.rstrip("\n")))
                self.cond.notify_all()

    def wait_errs(self, pred, timeout=10):
        with self.cond:
            if not self.cond.wait_for(lambda: pred(self.errs), timeout):
                raise TimeoutError(f"{self.tool}: timed out; last stderr {self.errs[-3:]}")

    def wait_lines(self, pred, timeout=10):
        with self.cond:
            if not self.cond.wait_for(lambda: pred(self.lines), timeout):
                raise TimeoutError(f"{self.tool}: timed out; last lines {self.lines[-3:]}")

    def frames(self):
        return [(t, f) for t, f in self.lines if f[0] == "frame"]

    def stop(self, how="quit"):
        if how == "quit":
            self.write({"quit": True})
        elif how == "sigterm":
            self.proc.send_signal(signal.SIGTERM)
        elif how == "sigint":
            self.proc.send_signal(signal.SIGINT)
        elif how == "eof":
            self.proc.stdin.close()
        code = self.proc.wait(timeout=10)
        self.reader.join(timeout=5)
        return code


def latency(tool, w, h, inflight, n=60, hz=20.0, root=DENO, revs=True):
    s = Server(tool, w, h, inflight, root=root)
    lat = []
    for _ in range(n):
        before = len(s.frames())
        time.sleep(1 / hz)
        rev = s.write({})
        if revs:
            s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= rev for _, f in ls[before:]))
            t = next(t for t, f in s.frames()[before:] if int(f[5]) >= rev)
        else:
            s.wait_lines(lambda ls: sum(f[0] == "frame" for _, f in ls) > before)
            t = s.frames()[before][0]
        lat.append((t - s.writes[rev]) * 1000)
    s.stop()
    return {"median": statistics.median(lat), "p95": pct(lat, 95)}


def throughput(tool, w, h, inflight, hz=120.0, seconds=5.0):
    s = Server(tool, w, h, inflight)
    time.sleep(0.3)
    start_line = len(s.lines)
    first = s.rev + 1
    period = 1 / hz
    t0 = time.perf_counter()
    k = 0
    while True:
        due = t0 + k * period
        if due - t0 >= seconds:
            break
        while (now := time.perf_counter()) < due:
            if due - now > 0.002:
                time.sleep(due - now - 0.0015)
        s.write({})
        k += 1
    last = s.rev
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= last for _, f in ls), 5)
    t_end = s.writes[last]
    frames = [(t, int(f[5])) for t, f in s.lines[start_line:] if f[0] == "frame" and int(f[5]) >= first]
    rendered = {r for _, r in frames}
    in_window = [x for x in frames if x[0] <= t_end + 0.001]
    ages = [(t - s.writes[r]) * 1000 for t, r in frames]
    s.stop()
    return {
        "fps": len(in_window) / (t_end - s.writes[first]),
        "frames": len(frames),
        "written": last - first + 1,
        "skipped": (last - first + 1) - len(rendered),
        "repeats": len(frames) - len(rendered),  # a rev rendered twice (the stamp changed again)
        "age_median": statistics.median(ages),
        "age_p95": pct(ages, 95),
    }


def matrix(opts):
    runs = int(opts.get("runs", 3))
    sizes = [tuple(map(int, s.split("x"))) for s in opts.get("sizes", "2048x1280,1032x645").split(",")]
    ks = [int(k) for k in opts.get("ks", "1,2,3,4").split(",")]
    tools = opts.get("tools", "surface,plot1d").split(",")
    hz = float(opts.get("hz", 120))
    out = Path(opts.get("out", "/dev/null"))
    results = []
    print(f"load: {os.getloadavg()}", flush=True)
    for tool in tools:
        for w, h in sizes:
            for run in range(runs):
                for k in ks:  # back to back, interleaved across runs
                    lat = latency(tool, w, h, k, n=int(opts.get("n", 60)))
                    thr = throughput(tool, w, h, k, hz=hz)
                    r = {"tool": tool, "size": f"{w}x{h}", "k": k, "run": run, **{f"lat_{a}": b for a, b in lat.items()}, **thr,
                         "load": os.getloadavg()[0]}
                    results.append(r)
                    print(json.dumps(r), flush=True)
                    with out.open("a") as fh:
                        fh.write(json.dumps(r) + "\n")
    print("\nmedians of runs:")
    print(f"{'tool':8} {'size':10} {'K':>2} {'lat med':>8} {'lat p95':>8} | {'fps':>6} {'skipped':>8} {'age med':>8} {'age p95':>8}")
    for tool in tools:
        for w, h in sizes:
            for k in ks:
                rs = [r for r in results if r["tool"] == tool and r["size"] == f"{w}x{h}" and r["k"] == k]
                m = lambda key: statistics.median(r[key] for r in rs)
                print(f"{tool:8} {w}x{h:<5} {k:>2} {m('lat_median'):8.2f} {m('lat_p95'):8.2f} | {m('fps'):6.1f} "
                      f"{m('skipped'):5.0f}/{rs[0]['written']:<3} {m('age_median'):8.2f} {m('age_p95'):8.2f}")


def check(tool):
    w, h = 640, 400
    ok = True

    def expect(cond, what):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")

    print(f"{tool}: snapshots")
    s = Server(tool, w, h, 3)
    snap = s.tmp / "snap.png"
    # Request a snapshot at a future rev: no ack for the frames before it
    r1 = s.write({"snapshot": {"path": str(snap), "rev": s.rev + 3}})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= r1 for _, f in ls))
    expect(not any(f[0] == "snapshot" for _, f in s.lines), "no ack for frames before the requested rev")
    r2 = s.write({"snapshot": {"path": str(snap), "rev": r1 + 2}})
    s.write({"snapshot": {"path": str(snap), "rev": r1 + 2}})  # rev r1+2: due now
    s.wait_lines(lambda ls: any(f[0] == "snapshot" for _, f in ls))
    for _ in range(3):
        s.write({"snapshot": {"path": str(snap), "rev": r1 + 2}})  # same request again: no new ack
    s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= s.rev for _, f in ls))
    acks = [(i, f) for i, (_, f) in enumerate(s.lines) if f[0] == "snapshot"]
    expect(len(acks) == 1 and acks[0][1] == ["snapshot", str(r1 + 2), str(snap)], f"exactly one ack for (path, {r1 + 2}): {acks}")
    i = acks[0][0]
    prev = [f for _, f in s.lines[:i] if f[0] == "frame"][-1]
    expect(int(prev[5]) >= r1 + 2, f"acked after a frame at or after its rev (frame rev {prev[5]})")
    expect(snap.exists() and snap.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "PNG written")
    # Old plain-string form: rev 0, the next frame answers it
    old = s.tmp / "old.png"
    s.write({"snapshot": str(old)})
    s.wait_lines(lambda ls: any(f[:2] == ["snapshot", "0"] for _, f in ls))
    s.write({"snapshot": str(old)})
    s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= s.rev for _, f in ls))
    expect(sum(f[:2] == ["snapshot", "0"] for _, f in s.lines) == 1 and old.exists(), "plain-string snapshot: one ack, rev 0")
    # A snapshot pending while frames of older revs arrive stays pending
    far = s.tmp / "far.png"
    target = s.rev + 6
    s.write({"snapshot": {"path": str(far), "rev": target}})
    for _ in range(3):
        time.sleep(0.03)
        s.write({})  # older than target, no snapshot field: must not clear the request
    time.sleep(0.2)
    expect(not any(f[:2] == ["snapshot", str(target)] for _, f in s.lines), "no ack while frames are older")
    s.write({}, rev=target)
    s.wait_lines(lambda ls: any(f[:2] == ["snapshot", str(target)] for _, f in ls))
    expect(far.exists(), "pending snapshot acked once its rev is rendered")

    print(f"{tool}: ready, controlerror")
    expect(s.lines and s.lines[0][1] == ["ready"], f"`ready` before the first frame: {s.lines[0][1] if s.lines else None}")
    good = s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= good for _, f in ls))
    n = len(s.frames())
    for bad in [{"width": 100000}, {"height": 0}, {"width": 640.5}, {"zoom": "x"}]:
        rev = s.write(bad)
        s.wait_errs(lambda es: any(e.startswith(f"controlerror {rev} ") for _, e in es), 5)
    err = [e for _, e in s.errs if e.startswith("controlerror")]
    expect(len(err) == 4 and all(len(e.split(" ", 2)) == 3 for e in err), f"controlerror <rev> <message>: {err[0] if err else None!r}")
    expect(len(s.frames()) == n, "no frame for the bad controls")
    rev = s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) == rev for _, f in ls))
    expect(True, "serving goes on after them")

    print(f"{tool}: frameFileOnce")
    ff = s.outdir / "frame.rgba"
    if ff.exists():
        ff.unlink()
    target = s.rev + 2
    s.write({"frameFileOnce": target})
    time.sleep(0.15)
    expect(not ff.exists() and not any(f[0] == "framefile" for _, f in s.lines), "nothing before the rev")
    s.write({"frameFileOnce": target})
    s.wait_lines(lambda ls: any(f == ["framefile", str(target)] for _, f in ls))
    frame = [f for _, f in s.lines if f[0] == "frame" and int(f[5]) >= target][0]
    data = ff.read_bytes()
    expect(len(data) == w * h * 4, "frame.rgba written, w × h × 4 bytes")
    expect(s.data.get(int(frame[1])) == data, f"frame.rgba == shm frame of the first frame at or after the rev (gen {frame[1]})")
    ff.unlink()
    s.write({"frameFileOnce": target})
    s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= s.rev for _, f in ls))
    expect(not ff.exists() and sum(f[0] == "framefile" for _, f in s.lines) == 1, "once only (not again for the same value)")
    expect(not any(f[0] == "frame" and len(f) != 6 for _, f in s.lines), "frame lines: frame gen w h name rev")
    del frame, data
    revs = [int(f[5]) for _, f in s.frames()]
    expect(revs == sorted(revs), "frames in rev order")

    print(f"{tool}: stale and torn controls")
    n = len(s.frames())
    s.write({}, rev=s.rev - 5)  # stale: ignored
    time.sleep(0.2)
    expect(len(s.frames()) == n, "stale rev ignored")
    s.rev += 5
    s.control.write_text('{"width": ')  # torn (in place)
    time.sleep(0.1)
    rev = s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) == rev for _, f in ls))
    expect(True, "recovered from a torn control")
    s.stop()

    for how in ["quit", "sigterm", "sigint", "eof"]:
        s = Server(tool, w, h, 3, consume=False)
        for _ in range(5):
            time.sleep(0.03)
            s.write({})
        s.wait_lines(lambda ls: sum(f[0] == "frame" for _, f in ls) >= 4, 5)
        before = sum(shm_exists(nm) for nm in s.names)
        code = s.stop(how)
        after = [nm for nm in s.names if shm_exists(nm)]
        expect(before > 0 and not after, f"{how}: {before} objects before exit, {len(after)} after (exit code {code})")

    print(f"{tool}: shm cap and ageing")
    s = Server(tool, 320, 200, 3, consume=False)
    for _ in range(40):
        time.sleep(0.01)
        s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= s.rev for _, f in ls))
    alive = [nm for nm in s.names if shm_exists(nm)]
    expect(len(alive) <= 32, f"at most 32 unread objects outstanding: {len(alive)} of {len(s.names)}")
    time.sleep(2.6)
    s.write({})
    s.wait_lines(lambda ls: any(f[0] == "frame" and int(f[5]) >= s.rev for _, f in ls))
    alive = [nm for nm in s.names[:-1] if shm_exists(nm)]
    expect(not alive, f"objects older than 2 s unlinked: {len(alive)} left")
    s.stop()
    print("all ok" if ok else "FAILURES")
    return ok


def coldstart(tool, runs=3, w=1032, h=645):
    """Seconds from spawn to `ready` and to the first frame (median of `runs`)."""
    ready, first = [], []
    for _ in range(runs):
        s = Server(tool, w, h, 3)
        ready.append(next(t for t, f in s.lines if f == ["ready"]) - s.t_spawn)
        first.append(s.frames()[0][0] - s.t_spawn)
        s.stop()
    return {"tool": tool, "ready_s": round(statistics.median(ready), 3), "first_frame_s": round(statistics.median(first), 3),
            "runs": [round(x, 3) for x in first]}


def main():
    cmd = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    w, h = map(int, opts.get("size", "1032x645").split("x"))
    k = int(opts.get("inflight", 3))
    if cmd == "latency":
        root = Path(opts["orig"]) if "orig" in opts else DENO
        print(latency(args[0], w, h, None if "orig" in opts else k, int(opts.get("n", 60)), float(opts.get("hz", 20)),
                      root=root, revs="orig" not in opts))
    elif cmd == "throughput":
        print(throughput(args[0], w, h, k, float(opts.get("hz", 120)), float(opts.get("seconds", 5))))
    elif cmd == "matrix":
        matrix(opts)
    elif cmd == "check":
        sys.exit(0 if all([check(t) for t in (args or ["surface", "plot1d"])]) else 1)
    elif cmd == "coldstart":
        for t in args or ["surface", "plot1d", "line", "curve", "complex", "volume"]:
            print(json.dumps(coldstart(t, int(opts.get("runs", 3)))), flush=True)


if __name__ == "__main__":
    main()
