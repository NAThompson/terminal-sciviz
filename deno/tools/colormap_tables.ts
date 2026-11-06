/**
 * Generates the colour map tables of lib/colormaps.ts: 256 sRGB samples (rrggbb) of each map at
 * i / 255, from d3's colour scheme data (d3-scale-chromatic), plus matplotlib's cyclic twilight
 * (16 stops, closed B-spline) and a fully saturated HSL hue wheel. Single-hue maps run dark to
 * light, like viridis.
 *
 * A one-off generator, not a dependency of anything: the d3 packages are fetched from npm when it
 * runs (dynamic imports, outside every module graph). The tables it writes are committed.
 *
 *   deno run -A tools/colormap_tables.ts           # compare with lib/colormaps.ts
 *   deno run -A tools/colormap_tables.ts --write   # rewrite its tables
 */

const npm = (name: string) => import(["npm", name].join(":"));
const chroma = await npm("d3-scale-chromatic@3.1.0");
const { rgb, hsl } = await npm("d3-color@3.1.0");
const { interpolateRgbBasisClosed } = await npm("d3-interpolate@3.0.1");

type Interp = (t: number) => string;
const scheme = (name: string): Interp => chroma[`interpolate${name}`];
const reversed = (f: Interp): Interp => (t) => f(1 - t);

const TWILIGHT_STOPS = [
  "#e2d9e2", "#c4ced4", "#95b5c7", "#7297c1", "#6276ba", "#5e51ad", "#592a8f", "#45135c",
  "#2f1436", "#4a1342", "#741e4f", "#983550", "#b25652", "#c27c63", "#cca389", "#d8c7be",
];
const twilight: Interp = interpolateRgbBasisClosed(TWILIGHT_STOPS);

const MAPS: Record<string, Interp> = {
  // Single-hue, dark to light
  ...Object.fromEntries(["Blues", "Greens", "Greys", "Oranges", "Purples", "Reds"].map((n) => [n, reversed(scheme(n))])),
  // Multi-hue
  ...Object.fromEntries(
    ["Turbo", "Viridis", "Inferno", "Magma", "Plasma", "Cividis", "Warm", "Cool"].map((n) => [n, scheme(n)]),
  ),
  Cubehelix: scheme("CubehelixDefault"),
  ...Object.fromEntries(
    ["BuGn", "BuPu", "GnBu", "OrRd", "PuBuGn", "PuBu", "PuRd", "RdPu", "YlGnBu", "YlGn", "YlOrBr", "YlOrRd"].map((n) => [n, scheme(n)]),
  ),
  // Cyclic
  Twilight: (t) => twilight(t - Math.floor(t)),
  Rainbow: scheme("Rainbow"),
  Sinebow: scheme("Sinebow"),
  HSL: (t) => hsl(t * 360, 1, 0.5).formatRgb(),
  // Diverging
  ...Object.fromEntries(
    ["BrBG", "PRGn", "PiYG", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral"].map((n) => [n, scheme(n)]),
  ),
};

const hex2 = (c: number) => c.toString(16).padStart(2, "0");
const tables = Object.entries(MAPS).map(([name, f]) => {
  let hex = "";
  for (let i = 0; i < 256; i += 1) {
    const { r, g, b } = rgb(f(i / 255));
    hex += hex2(r) + hex2(g) + hex2(b);
  }
  return [name, hex] as const;
});

const file = new URL("../lib/colormaps.ts", import.meta.url);
const source = Deno.readTextFileSync(file);
const start = source.indexOf("const TABLES: Record<string, string> = {\n");
const end = source.indexOf("\n};\n", start);
if (start < 0 || end < 0) throw new Error("TABLES not found in lib/colormaps.ts");
const block = `const TABLES: Record<string, string> = {\n${tables.map(([n, h]) => `  ${n}: "${h}",`).join("\n")}`;

if (Deno.args.includes("--write")) {
  Deno.writeTextFileSync(file, source.slice(0, start) + block + source.slice(end));
  console.log(`wrote ${tables.length} tables`);
} else {
  const current = new Map([...source.slice(start, end).matchAll(/^ {2}(\w+): "([0-9a-f]+)",$/gm)].map((m) => [m[1], m[2]]));
  let bad = 0;
  for (const [name, hex] of tables) {
    if (current.get(name) !== hex) {
      bad += 1;
      console.log(`${name}: ${current.has(name) ? "differs" : "missing"}`);
    }
  }
  for (const name of current.keys()) if (!(name in MAPS)) console.log(`${name}: extra`);
  console.log(bad ? `${bad} of ${tables.length} tables differ` : `all ${tables.length} tables identical`);
  if (bad) Deno.exit(1);
}
