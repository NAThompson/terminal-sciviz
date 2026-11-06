/**
 * Checks the stat-polling `ControlWatcher` (lib/sys.ts), the path Linux takes, forced on any OS
 * with SCIVIZ_STAT_POLL=1: `waitAsync` settles (it used to call the guarded `wait` while itself
 * pending, and reject), the guard still holds, and `serve` keeps rendering control changes made
 * while frames are in flight (the `waitAsync` of lib/serve.ts's loop).
 *
 *   deno run -A tools/check_watcher.ts
 */

Deno.env.set("SCIVIZ_STAT_POLL", "1"); // before lib/sys.ts is evaluated
const { ControlWatcher, statStamp } = await import("../lib/sys.ts");

let failed = 0;
const expect = (ok: boolean, what: string) => {
  if (!ok) failed += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
};

const dir = Deno.makeTempDirSync();
const control = `${dir}/control.json`;
const write = (ctl: object) => {
  Deno.writeTextFileSync(`${control}.tmp`, JSON.stringify(ctl));
  Deno.renameSync(`${control}.tmp`, control);
};

console.log("ControlWatcher (stat polling)");
write({ rev: 1 });
const watcher = new ControlWatcher(control, false);
const stamp = statStamp(control);
let error: unknown;
const t0 = performance.now();
for (let i = 0; i < 5; i += 1) {
  await watcher.waitAsync(stamp).catch((err) => (error ??= err));
}
expect(error === undefined, `waitAsync settles (${error ?? "no error"})`);
expect(performance.now() - t0 < 500, `5 waits in ${(performance.now() - t0).toFixed(1)} ms`);
const pending = watcher.waitAsync(stamp);
let threw = false;
try {
  watcher.wait(stamp);
} catch {
  threw = true;
}
expect(threw, "wait() while waitAsync() is pending still throws");
await pending;
watcher.wait(stamp);
expect(watcher.settled() === null, "wait() once settled");

console.log("serve (stat polling, frames in flight)");
write({ width: 320, height: 200, rev: 1 });
const proc = new Deno.Command(Deno.execPath(), {
  args: ["run", "-A", new URL("../plot1d.ts", import.meta.url).pathname, "serve", control, `${dir}/out`, "--ignore-stdin"],
  env: { SCIVIZ_STAT_POLL: "1" },
  stdout: "piped",
  stderr: "piped",
}).spawn();
const lines: string[] = [];
const reader = (async () => {
  let buf = "";
  for await (const chunk of proc.stdout.pipeThrough(new TextDecoderStream())) {
    buf += chunk;
    const parts = buf.split("\n");
    buf = parts.pop()!;
    lines.push(...parts);
  }
})();
const stderr = new Response(proc.stderr).text();
const until = async (pred: () => boolean, ms: number) => {
  const end = performance.now() + ms;
  while (!pred() && performance.now() < end) await new Promise((r) => setTimeout(r, 5));
  return pred();
};
const frameRev = (rev: number) => lines.some((l) => l.startsWith("frame ") && Number(l.split(" ")[5]) >= rev);
expect(await until(() => frameRev(1), 30_000), "first frame");
let rev = 1;
for (let i = 0; i < 40; i += 1) { // faster than frames complete: some arrive while others are in flight
  rev += 1;
  write({ width: 320, height: 200, rev, x0: 0.1 + 0.001 * i, x1: 0.6 });
  await new Promise((r) => setTimeout(r, 2));
}
expect(await until(() => frameRev(rev), 10_000), `the last control (rev ${rev}) rendered`);
write({ rev: rev + 1, quit: true });
const { code } = await proc.status;
await reader;
const err = await stderr;
expect(code === 0, `exit code ${code}`);
expect(!/pending|Error/.test(err), `no errors on stderr${err ? `: ${err.trim()}` : ""}`);
Deno.removeSync(dir, { recursive: true });
console.log(failed ? `${failed} FAILED` : "all ok");
Deno.exit(failed ? 1 : 0);
