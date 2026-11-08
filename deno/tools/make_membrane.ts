/**
 * The gallery's sliced heatmap data (assets/membrane.json, columns x, y, t, u): two modes of a square
 * membrane, u = sin πx sin 2πy cos 2πt + ½ sin 2πx sin πy sin 2πt, on a 16 × 12 grid of cells at
 * t = 0, 0.1, …, 0.9.
 *
 *   deno run -A tools/make_membrane.ts
 */

const m = { x: [] as number[], y: [] as number[], t: [] as number[], u: [] as number[] };
for (let k = 0; k < 10; k += 1) {
  for (let j = 0; j < 12; j += 1) {
    for (let i = 0; i < 16; i += 1) {
      const [x, y, t] = [(i + 0.5) / 16, (j + 0.5) / 12, k / 10];
      const u = Math.sin(Math.PI * x) * Math.sin(2 * Math.PI * y) * Math.cos(2 * Math.PI * t) + 0.5 * Math.sin(2 * Math.PI * x) * Math.sin(Math.PI * y) * Math.sin(2 * Math.PI * t);
      m.x.push(i);
      m.y.push(j);
      m.t.push(t);
      m.u.push(Math.round(u * 1000) / 1000);
    }
  }
}
const path = new URL("../assets/membrane.json", import.meta.url).pathname;
Deno.writeTextFileSync(path, JSON.stringify(m) + "\n");
console.log(`${path}: ${Deno.statSync(path).size} bytes`);
