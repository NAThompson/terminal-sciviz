/**
 * The "lines" section: Bessel functions J_n and line plots exercising scales, markers, errors,
 * NaN gaps, envelopes and reference lines, plus the PCHIP demos of plot1d.ts, all drawn by
 * line.ts. Colours come from the theme's palette.
 * Pure functions of x are generator specs (`fn`, `sample.fn`: lib/generators.ts), sampled on the
 * GPU by line.ts; noise (residual, measured, score_runs) comes from a seeded generator.
 */

import type { Entry } from "./types.ts";
import { linspace, normalRng } from "../lib/specfun.ts";

const BLUE = "#3987e5", ORANGE = "#d95926", AQUA = "#199e70", MAGENTA = "#d55181", RED = "#e66767", INK2 = "#8b949e";
const PI = Math.PI;

const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);

function line(id: string, title: string, description: string, control: Record<string, unknown>): Entry {
  return { id: `lines/${id}`, section: "lines", title, description, renderer: "line.ts", control: { title, ...control }, interaction: "pan-zoom-x" };
}

// --- basics/bessel_Jn ------------------------------------------------------------------------------

const bessel = line("bessel_Jn", "Bessel functions 𝐽₀(𝑥) … 𝐽₅(𝑥)", "orders 0 to 5, one curve per order, evaluated on the GPU", {
  xlabel: "𝑥",
  series: range(0, 6).map((n) => ({ fn: { name: "besselJ", order: n, domain: [0, 30] }, label: `𝐽${"₀₁₂₃₄₅"[n]}` })),
});

// --- runge_samples: linear / spline / pchip --------------------------------------------------------

const rungeSamples = (["linear", "spline", "pchip"] as const).map((interp) =>
  line(`runge_samples_${interp}`, `Runge 1/(1 + 25𝑥²) · ${interp === "pchip" ? "PCHIP" : interp}`,
    `15 coarse samples, interpolation ${interp} (the line toolbar's Interpolation); true function faded`, {
      xlabel: "𝑥",
      series: [
        { fn: "runge", label: "1/(1 + 25𝑥²)", color: INK2, width: 1.2, opacity: 0.55 },
        { sample: { fn: "runge", n: 15 }, label: `samples, ${interp}`, interpolation: interp, markers: true, width: 2 },
      ],
    })
);

// --- residual ------------------------------------------------------------------------------------

const rng = normalRng(11);
const it = range(1, 301);
const resid = it.map((k) => 10 ** (-k / 40) * Math.exp(0.9 * rng()) * (1 + 50 * Math.exp(-k / 15)));
const residual = line("residual", "residual ‖𝐫‖", "log y, running-min envelope (add), reference lines/bands with legend", {
  xlabel: "iteration",
  y: { scale: "log" },
  series: [{ x: it, y: resid, label: "residual", color: BLUE, width: 1.2, envelope: { type: "cummin", mode: "add" } }],
  overlays: [ // warmup, target, tolerance, restart
    { kind: "band", axis: "x", values: [1, 40], label: "warm-up", color: MAGENTA, width: 1.5, dash: true },
    { kind: "band", axis: "y", values: [1e-7, 1e-6], label: "target", color: AQUA },
    { kind: "hline", values: [1e-5], label: "tolerance 10⁻⁵", color: RED, dash: true },
    { kind: "vline", values: [150], label: "restart", color: INK2 },
  ],
});

// --- score_runs ----------------------------------------------------------------------------------

const scoreRuns = line("score_runs", "score", "suffix-max envelope (cummax, reverse) replacing the four runs", {
  xlabel: "iteration",
  series: range(0, 4).map((i) => {
    let acc = 0;
    const y = it.map(() => (acc += 0.02 * (i + 1) + rng()));
    return { x: it, y, label: `run ${i}`, envelope: { type: "cummax", mode: "replace", reverse: true } };
  }),
});

// --- measured, fit, envelope_bound ---------------------------------------------------------------

const tt = linspace(0, 30, 400);
const tau = 14; // slice τ = 14 s of (τ, t) = ({4, 6, 9, 14}, t)
const clean = tt.map((t) => Math.exp(-t / tau) * Math.cos(1.3 * t));
const measured = line("measured", "damped oscillator", "overlaid series: fit (same scales, sliced along 𝜏 = 14 s) and a 1D dashed bound", {
  xlabel: "𝑡 (s)",
  series: [
    { x: tt, y: clean.map((c) => c + 0.05 * rng()), label: "measured", color: ORANGE },
    { fn: { name: "dampedCos", tau, omega: 1.3, domain: [0, 30] }, label: "fit exp(−𝑡/𝜏) cos 𝜔𝑡", color: RED, width: 2 },
    { fn: { name: "dampedCos", tau: 14, domain: [0, 30] }, label: "bound exp(−𝑡/14)", color: INK2, dash: true },
  ],
});

// --- value transforms, log zero marks, symlog, π ticks, Bode, dispersion -----------------------------

const n64 = range(1, 65);
const log2Error = line("log2_error", "log₂ error", "values are log₂ of the error: y ticks shown as 2ᵏ", {
  y: { ticks: "exp2" },
  series: [{ y: n64.map((n) => -n * 0.9 + 3 * Math.sin(n / 3)), label: "log₂ error", legend: false }],
});

const zeros = linspace(0, 12, 200).map((s, i) => Math.abs(Math.sin(s)) ** 3 * 10 ** (-6 * i / 199));
for (let i = 0; i < 200; i += 25) zeros[i] = 0;
zeros[37] = Infinity;
const logZeroMarks = line("log_zero_marks", "intensity", "log y: values ≤ 0 and +∞ marked at the plot edges", {
  y: { scale: "log" },
  nanColor: RED,
  series: [{ y: zeros, label: "intensity", legend: false }],
});

const symlogLine = line("symlog_line", "signed amplitude", "values ±10⁻¹⁶ to ±1, symlog y with threshold 10⁻¹²", {
  y: { scale: "symlog", threshold: 1e-12 },
  series: [{ y: linspace(0, 9, 400).map((s, i) => Math.sign(Math.sin(s)) * 10 ** (-16 + (16 * i) / 399)), label: "signed amplitude", legend: false }],
});

const sincPi = line("sinc_pi_ticks", "sin(1.5𝜑)/(1.5𝜑)", "x ticks at multiples of π", {
  xlabel: "𝜑 (rad)",
  x: { ticks: "pi" },
  series: [{ fn: { name: "sinc", k: 1.5, domain: [-2 * PI, 2 * PI] }, label: "sin(1.5𝜑)/(1.5𝜑)", legend: false }],
});

// second-order low-pass, r = 2πf / 2·10⁴, damping 0.1: 20 log₁₀ 1/|1 − r² + 0.2𝑖r|
const bode = line("bode_magnitude", "|𝐻(𝑓)| (dB)", "log frequency axis", {
  xlabel: "f (Hz)",
  x: { scale: "log" },
  series: [{ fn: { name: "bode", f0: 2e4 / (2 * PI), zeta: 0.1, domain: [1, 1e5] }, label: "|𝐻(𝑓)|", legend: false }],
});

const dispersion = line("dispersion", "band energy 𝐸(𝑘) (eV)", "(k, band) array: dimension 0 (𝑘) on x (π ticks), one curve per band", {
  xlabel: "𝑘 (1/a)",
  x: { ticks: "pi" },
  series: range(0, 5).map((b) => ({ fn: { name: "dispersion", band: b }, label: `band ${b}` })),
});

// --- PCHIP demos of plot1d.ts --------------------------------------------------------------------

const pchip = (fn: string, n: number) =>
  line(`pchip_${fn}`, `PCHIP · ${fn} · n = ${n}`, `${n} samples of ${fn}, PCHIP (Fritsch-Carlson slopes) over the true function`, {
    series: [
      { fn, label: "f", color: INK2, width: 1.2, opacity: 0.55 },
      { sample: { fn, n }, interpolation: "pchip", markers: true, label: "PCHIP", width: 2.5 },
    ],
  });

export const entries: Entry[] = [
  bessel,
  ...rungeSamples,
  residual,
  measured,
  log2Error,
  logZeroMarks,
  symlogLine,
  sincPi,
  bode,
  dispersion,
  scoreRuns,
  pchip("bumps", 32),
  pchip("runge", 12),
  pchip("blocks", 32),
];
