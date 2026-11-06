/// <reference lib="deno.worker" />
/** Busy-spins while `state[0]` is 1 (keeps the CPU cluster's clocks up while the main thread
 * waits on the GPU, cf. lib/serve.ts), parks while it is 0, exits at 2. */
import { setInteractiveQos } from "./sys.ts";

setInteractiveQos();
self.onmessage = (event: MessageEvent<SharedArrayBuffer>) => {
  const state = new Int32Array(event.data);
  for (;;) {
    const s = Atomics.load(state, 0);
    if (s === 2) {
      self.close();
      return;
    }
    if (s === 0) {
      Atomics.wait(state, 0, 0);
    }
  }
};
