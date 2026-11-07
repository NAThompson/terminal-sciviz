/**
 * Parameter sliders: the control's `sliders: {name: [min, max] | {min, max, step?, label?}}` names
 * parameters of the plot (an fn's expr params or a named generator's arguments, an ODE's
 * coefficients); `sliderValues: {name: v}` (written by the mod as the user drags) overrides them.
 * Drawn into the frame (overlay.ts `slider`) and reported in state.json `sliders` with their hit
 * rects, so the mod can hit-test them.
 */

import { type FontSize, type Quads, rootSliderHeight, slider } from "./overlay.ts";
import { GENERATORS } from "./generators.ts";

export interface SliderDef { name: string; label: string; min: number; max: number; step: number; value: number }
/** As drawn: the track's hit area (frame px), x linear from rect[0] (min) to rect[2] (max). */
export interface Slider extends SliderDef { rect: [number, number, number, number] }
/** Where a parameter lives in a (copied) control: `obj[key]`, its current value. */
export interface Param { name: string; obj: Record<string, unknown>; key: string; value: number }

const GREEK: Record<string, string> = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", kappa: "κ", lambda: "λ",
  lam: "λ", mu: "μ", nu: "ν", xi: "ξ", rho: "ρ", sigma: "σ", tau: "τ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω",
};

/** An fn spec as an object: a generator's name alone (`"dampedCos"`) as `{name}`, so its args can be set. */
export const fnObject = (spec: unknown) => (typeof spec === "string" ? { name: spec } : spec);

/** The parameters of an fn spec (lib/generators.ts): an expr's params, a named generator's args. */
export function fnParams(spec: unknown): Param[] {
  if (!spec || typeof spec !== "object" || Array.isArray(spec)) return [];
  const s = spec as Record<string, unknown>;
  if ("expr" in s) {
    const p = s.params as Record<string, unknown> | undefined;
    return p && typeof p === "object" ? Object.keys(p).map((k) => ({ name: k, obj: p, key: k, value: Number(p[k]) })) : [];
  }
  const g = typeof s.name === "string" && Object.hasOwn(GENERATORS, s.name) ? GENERATORS[s.name] : undefined;
  return g ? g.args.map((a, i) => ({ name: a, obj: s, key: a, value: s[a] === undefined ? g.defaults[i] : Number(s[a]) })) : [];
}

/** The sliders of `sliders` over `params` (all of a name set to its value: `values`' if given,
 * else the parameter's own, clamped to the range). Throws on a bad spec or an unknown name. */
export function resolveSliders(sliders: unknown, values: unknown, params: Param[]): SliderDef[] {
  if (sliders === undefined || sliders === null) return [];
  if (typeof sliders !== "object" || Array.isArray(sliders)) throw new Error("sliders: {name: [min, max] | {min, max, step?, label?}}");
  const given = (values && typeof values === "object" ? values : {}) as Record<string, unknown>;
  return Object.entries(sliders as Record<string, unknown>).map(([name, spec]) => {
    const o = (Array.isArray(spec) ? { min: spec[0], max: spec[1] } : spec ?? {}) as Record<string, unknown>;
    const [min, max] = [Number(o.min), Number(o.max)];
    if (!(Number.isFinite(min) && Number.isFinite(max) && Number.isFinite(max - min) && max > min)) throw new Error(`sliders.${name}: need finite [min, max] with min < max (and max − min finite)`);
    const step = o.step === undefined ? 10 ** Math.floor(Math.log10((max - min) / 100)) : Number(o.step);
    if (!(Number.isFinite(step) && step > 0)) throw new Error(`sliders.${name}: step must be a finite number > 0${o.step === undefined ? " (the range is too small for one)" : `, got ${JSON.stringify(o.step)}`}`);
    const at = params.filter((p) => p.name === name);
    if (!at.length) {
      const names = [...new Set(params.map((p) => p.name))];
      throw new Error(`sliders: no parameter "${name}" in this plot (${names.length ? `its parameters: ${names.join(", ")}` : "it has none"})`);
    }
    const v = Number(given[name] ?? at[0].value);
    const value = Math.min(max, Math.max(min, Number.isFinite(v) ? v : min));
    for (const p of at) p.obj[p.key] = value;
    return { name, label: typeof o.label === "string" ? o.label : GREEK[name] ?? name, min, max, step, value };
  });
}

/** Height (px) of a slider at scale `s`. */
export const sliderHeight = rootSliderHeight;

/** `defs` drawn from (x, top), each `width` px long, `gap` apart in a row or a column; `hot`: the one under the pointer. */
export function drawSliders(q: Quads, font: FontSize, defs: SliderDef[], x: number, top: number, width: number, gap: number, s: number, column = false, hot?: unknown): Slider[] {
  return defs.map((d, i) => {
    const digits = Math.min(6, Math.max(0, Math.ceil(-Math.log10(d.step) - 1e-9)));
    const [sx, sy] = column ? [x, top + i * (sliderHeight(font, s) + gap)] : [x + i * (width + gap), top];
    const text = `${d.label} = ${d.value.toFixed(digits)}`;
    const tx = Math.min(sx, sx + width - q.measure(font, text)); // a long label ends with the track
    const rect = slider(q, font, sx, sy, width, text, (d.value - d.min) / (d.max - d.min), s, tx, d.name === hot);
    return { ...d, rect };
  });
}
