// The gallery's shared contract. Entries are plain data; each names the renderer script (a
// sibling of surface.ts / plot1d.ts speaking lib/serve.ts's protocol) and the control that
// renders it. The gallery harness renders every entry headlessly to PNG, builds contact sheets,
// and compares against golden images; the Claude Code mod opens any entry as a live view.

export type Section = "lines" | "curves" | "complex" | "volumes" | "scenes" | "surfaces";

export interface Entry {
  /** Stable id, also the golden image's file name: e.g. "lines/residual", "complex/rational". */
  id: string;
  section: Section;
  /** Short title for the contact sheet. */
  title: string;
  /** One line: what it shows / which feature it exercises. */
  description: string;
  /** Renderer script under deno/, e.g. "line.ts", "curve.ts", "complex.ts", "surface.ts", "plot1d.ts". */
  renderer: string;
  /** The renderer's control (without width/height/rev/hover/snapshot, which the harness sets). */
  control: Record<string, unknown>;
  /** Interaction the live view supports: "pan-zoom-x" (2D lines), "orbit" (3D), "pan-zoom-xy". */
  interaction: "pan-zoom-x" | "pan-zoom-xy" | "orbit";
  /** 4D stacks: the live view steps the control's `slice` through `slices` at `fps`. */
  playback?: { fps: number; slices: number };
}
