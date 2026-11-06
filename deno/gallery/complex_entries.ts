// The "complex" section: complex functions as phase/amplitude heatmaps and as a height-field
// surface, computed in Deno (complex.ts, surface.ts).

import type { Entry } from "./types.ts";

export const entries: Entry[] = [
  {
    id: "complex/rational",
    section: "complex",
    title: "(z² − 1)(z − 2 − i)²/(z² + 2 + 2i)",
    description: "phase and amplitude: hue arg f (Twilight), opacity log |f|",
    renderer: "complex.ts",
    control: { function: "rational", colouring: "phase-amplitude", scale: "log", phaseColorMap: "Twilight" },
    interaction: "pan-zoom-xy",
  },
  {
    id: "complex/cubic_surface",
    section: "complex",
    title: "z³ − 1",
    description: "surface: height log |f| (z stretched by 0.5), colour arg f (Twilight)",
    renderer: "surface.ts",
    control: {
      surface: "complex-cubic",
      colorBy: "field",
      colormap: "Twilight",
      domain: [-Math.PI, Math.PI],
      cyclic: true,
    },
    interaction: "orbit",
  },
];
