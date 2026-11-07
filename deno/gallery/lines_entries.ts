/**
 * The "lines" section: Bessel functions J_n and line plots exercising scales, markers, errors,
 * NaN gaps, envelopes and reference lines, plus the PCHIP demos of plot1d.ts, all drawn by
 * line.ts. Colours come from the theme's palette.
 * Pure functions of x are generator specs (`fn`, `sample.fn`: lib/generators.ts), sampled in f64
 * by line.ts; noise (residual, measured, score_runs) comes from a seeded generator.
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

const bessel = line("bessel_Jn", "Bessel functions 𝐽₀(𝑥) … 𝐽₅(𝑥)", "orders 0 to 5, one curve per order, evaluated by Boost.Math", {
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

// --- poisson_hist ---------------------------------------------------------------------------------

// 2000 counts of a Poisson(4.3) process (Knuth: multiply seeded uniforms until below e^(−λ)),
// the histogram as bars, and the maximum-likelihood fit λ̂ = mean over it
let seed = 7;
const uniform = () => { // mulberry32
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const counts = range(0, 2000).map(() => {
  let [k, p] = [0, uniform()];
  while (p > Math.exp(-4.3)) [k, p] = [k + 1, p * uniform()];
  return k;
});
const kmax = Math.max(...counts);
const hist = range(0, kmax + 1).map((k) => counts.filter((c) => c === k).length);
const lam = counts.reduce((a, b) => a + b, 0) / counts.length;
const expected = (k: number) => counts.length * Math.exp(k * Math.log(lam) - lam - range(1, k + 1).reduce((a, j) => a + Math.log(j), 0));
const bins = range(0, kmax + 1).filter((k) => expected(k) >= 5);
const chi2 = bins.reduce((a, k) => a + (hist[k] - expected(k)) ** 2 / expected(k), 0) / (bins.length - 2);
const poissonHist = line("poisson_hist", "Poisson counts · 𝑁 = 2000", "histogram bars with the maximum-likelihood Poisson fit, its pmf smooth between the integers (𝜆ˣ e^−𝜆 / Γ(𝑥 + 1))", {
  subtitle: `maximum likelihood 𝜆̂ = ${lam.toFixed(3)}, 𝜒²/𝜈 = ${chi2.toFixed(2)}`,
  xlabel: "count 𝑘",
  ylabel: "events",
  series: [
    { x: range(0, kmax + 1), y: hist, bars: true, label: "observed", color: BLUE },
    { fn: { name: "poisson", lambda: lam, scale: counts.length, domain: [0, kmax + 0.5] }, label: `Poisson fit, 𝜆̂ = ${lam.toFixed(2)}`, color: ORANGE, width: 2 },
    { x: range(0, kmax + 1), y: range(0, kmax + 1).map(expected), markers: "only", color: ORANGE, legend: false },
  ],
});

// --- expr_normal_approx: expressions of x (lib/expr.ts) -----------------------------------------

// the Poisson pmf smooth between the integers against its normal approximation N(λ, λ) (and that
// CDF, scaled by 0.2), for λ = 4.3
const lamParams = { lam: 4.3 };
const exprNormal = line("expr_normal_approx", "Poisson(λ) and its normal approximation", "fn as expressions of 𝑥 with params: lgamma, erf (Boost.Math) and a Gaussian", {
  subtitle: "λ = 4.3, normal with μ = σ² = λ",
  xlabel: "𝑥",
  series: [
    { fn: { expr: "exp(x*log(lam) - lam - lgamma(x + 1))", params: lamParams, domain: [0, 14] }, label: "λ^𝑥 e^−λ / Γ(𝑥 + 1)", color: BLUE, width: 2 },
    { fn: { expr: "exp(-(x - lam)^2 / (2*lam)) / sqrt(2*pi*lam)", params: lamParams, domain: [0, 14] }, label: "normal density", color: ORANGE, dash: true },
    { fn: { expr: "0.1*(1 + erf((x - lam) / sqrt(2*lam)))", params: lamParams, domain: [0, 14] }, label: "normal CDF / 5", color: AQUA },
  ],
});

// the same damped cosine as a named generator, its envelope ±exp(−x/τ) as expressions: sliders for τ
// and ω set the parameter of that name in every fn (the mod writes `sliderValues` as they're dragged)
const envelope = { tau: 6 };
const exprSliders = line("expr_sliders", "Damped cosine exp(−𝑡/τ) cos ω𝑡", "sliders for fn parameters: τ in a named generator and two exprs, ω in the generator", {
  xlabel: "𝑡",
  series: [
    { fn: { name: "dampedCos", tau: 6, omega: 1.2, domain: [0, 30] }, label: "exp(−𝑡/τ) cos ω𝑡", color: BLUE, width: 2 },
    { fn: { expr: "exp(-x/tau)", params: envelope, domain: [0, 30] }, label: "±exp(−𝑡/τ)", color: INK2, dash: true },
    { fn: { expr: "-exp(-x/tau)", params: envelope, domain: [0, 30] }, color: INK2, dash: true },
  ],
  sliders: { tau: [0.5, 20], omega: [0, 3] },
});

// --- distributions: stats of raw samples (line.ts `stat`, lib/stats.ts) ---------------------------

// 300 log-normal(0, ½) draws: their ECDF with the 95% DKW band, over the true CDF Φ(ln 𝑥 / σ)
const lnRng = normalRng(23);
const logNormal = range(0, 300).map(() => Math.exp(0.5 * lnRng()));
const ecdfBand = line("ecdf_band", "ECDF · log-normal, 𝑛 = 300", "stat ecdf of raw samples: a step function with its DKW band (fill between, stepped), over the true CDF as an expr", {
  subtitle: "band ±ε, ε = √(ln(2/α)/2𝑛), α = 0.05",
  xlabel: "𝑥",
  ylabel: "𝐹(𝑥)",
  series: [
    { samples: logNormal, stat: { type: "ecdf", band: 0.05 }, label: "ECDF", color: BLUE },
    { fn: { expr: "0.5*(1 + erf(log(x)/(s*sqrt(2))))", params: { s: 0.5 }, domain: [0.05, 5] }, label: "true CDF Φ(ln 𝑥 / σ)", color: ORANGE, dash: true },
  ],
});

// normal and Student-t₃ draws (𝑍 / √(χ²₃/3)), 200 each: the t's heavy tails bend away from the line
const qqRng = normalRng(5);
const normal200 = range(0, 200).map(() => qqRng());
const t3 = range(0, 200).map(() => qqRng() / Math.sqrt((qqRng() ** 2 + qqRng() ** 2 + qqRng() ** 2) / 3));
const qqNormal = line("qq_normal", "normal Q–Q plot", "stat qq: Φ⁻¹((𝑖 − ½)/𝑛) against the sorted samples, each with its line through the quartiles", {
  subtitle: "𝑛 = 200 each",
  xlabel: "normal quantile",
  ylabel: "sample quantile",
  series: [
    { samples: normal200, stat: "qq", label: "normal" },
    { samples: t3, stat: "qq", label: "Student 𝑡₃" },
  ],
});

// four groups, a few outliers planted: box plots on a category axis
const boxRng = normalRng(31);
const groups = [["control", 10, 1.5], ["dose A", 12, 2], ["dose B", 15, 2.5], ["dose C", 14, 1.2]] as const;
const boxGroups = line("box_groups", "response by group", "stat box per group on a category axis: quartiles (type 7), whiskers to 1.5 IQR, outliers as markers", {
  subtitle: "𝑛 = 60 per group",
  ylabel: "response",
  x: { categories: groups.map((g) => g[0]) },
  legend: false,
  series: groups.map(([name, mu, sd], i) => ({
    samples: [...range(0, 58).map(() => mu + sd * boxRng()), mu + sd * (3.2 + i * 0.4), mu - sd * (3 + (i % 2))],
    stat: "box",
    label: name,
  })),
});

// 40 noisy points on a line: least squares with its 95% mean-response band (hyperbolic, flaring at the ends)
const lrRng = normalRng(17);
const lrX = range(0, 40).map((i) => 0.25 * i + 0.4 * lrRng());
const lrY = lrX.map((x) => 1.5 + 0.8 * x + 1.2 * lrRng());
const linregBand = line("linreg_band", "least squares with a confidence band", "stat linreg: the fit and the t band edges as expr fns (resampled on zoom), the band a fill between them", {
  subtitle: "band ± 𝑡ₙ₋₂ 𝑠 √(1/𝑛 + (𝑥 − 𝑥̄)²/𝑆ₓₓ), 95%",
  xlabel: "𝑥",
  ylabel: "𝑦",
  series: [{ x: lrX, y: lrY, stat: { type: "linreg", band: 0.05 }, label: "data" }],
});

// --- marks: channels, scales and transforms composed (line.ts `marks`, lib/marks.ts) ---------------

const mark = (id: string, title: string, description: string, control: Record<string, unknown>) => line(id, title, description, { series: [], ...control });
const uniformRng = (seed: number) => () => { // mulberry32
  seed = (seed + 0x6d2b79f5) >>> 0;
  let t = Math.imul(seed ^ (seed >>> 15), seed | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// synthetic countries: GDP per head (log x), life expectancy rising with its log, population as area
const bRng = normalRng(41), bU = uniformRng(4141);
const regions = [["Africa", 2.5, 9], ["Americas", 3.9, 7], ["Asia", 3.6, 9], ["Europe", 4.5, 8], ["Oceania", 4.3, 4]] as const;
const countries = regions.flatMap(([region, lg, n]) => range(0, n).map(() => {
  const gdp = 10 ** (lg + 0.35 * bRng());
  return { region, gdp: +(gdp / 1000).toPrecision(3), life: +(38 + 8.5 * Math.log10(gdp) + 2.5 * bRng()).toFixed(1), pop: Math.round(10 ** (6.3 + 1.1 * bU() ** 2 * 1.6)) };
}));
const bubble = mark("marks_bubble", "life expectancy against income", "dot marks: x, y, r (sqrt area scale) and a categorical fill (palette legend), on a log x axis", {
  subtitle: "synthetic countries · area ∝ population",
  xlabel: "GDP per head (k$, log)",
  ylabel: "life expectancy (years)",
  x: { scale: "log" },
  r: { range: [3, 22] },
  marks: [{ type: "dot", data: countries, x: "gdp", y: "life", r: "pop", fill: "region", opacity: 0.8 }],
});

// four sectors over 25 years, stacked: areas with lines on their tops
const sRng = normalRng(43);
const sectors = [["services", 40, 1.2], ["industry", 30, 0.3], ["agriculture", 18, -0.25], ["energy", 8, 0.35]] as const;
const output = { year: [] as number[], value: [] as number[], sector: [] as string[] };
for (const [name, v0, slope] of sectors) {
  let v: number = v0;
  for (const year of range(2000, 2025)) {
    v = Math.max(1, v + slope + 0.8 * sRng());
    output.year.push(year);
    output.value.push(+v.toFixed(2));
    output.sector.push(name);
  }
}
const stackedArea = mark("marks_stacked_area", "output by sector", "area + stack by z (fill: sector), with line marks of the same stack on the tops", {
  subtitle: "synthetic · stacked",
  xlabel: "year",
  ylabel: "output (bn)",
  marks: [
    { type: "area", data: output, x: "year", y: "value", fill: "sector", transform: { stack: true } },
    { type: "line", data: output, x: "year", y: "value", stroke: "sector", width: 1.5, transform: { stack: true } },
  ],
});

// a bimodal sample: its histogram (bin → bar), frequency polygon (bin → line) and KDE (density → line)
const mRng = normalRng(47), mU = uniformRng(4747);
const bimodal = { v: range(0, 600).map(() => (mU() < 0.35 ? -1.2 + 0.6 * mRng() : 1.4 + 0.9 * mRng())) };
const binLine = mark("marks_bin_line", "one sample, three summaries", "bin → bar and bin → line over the same samples, density (Gaussian KDE, Silverman bandwidth) → line, and the true density as an expr fn", {
  subtitle: "𝑛 = 600 from 0.35 𝒩(−1.2, 0.6²) + 0.65 𝒩(1.4, 0.9²)",
  xlabel: "𝑣",
  ylabel: "density",
  marks: [
    { type: "bar", data: bimodal, x: "v", fill: "#3987e5", opacity: 0.55, transform: { bin: { width: 0.4, reduce: "density" } }, label: "histogram" },
    { type: "line", data: bimodal, x: "v", stroke: "#d95926", width: 2, transform: { bin: { width: 0.4, reduce: "density" } }, label: "frequency polygon" },
    { type: "line", data: bimodal, x: "v", stroke: "#199e70", width: 2.5, transform: { density: true }, label: "KDE" },
    { type: "line", fn: { expr: "0.35*exp(-(x+1.2)^2/0.72)/(0.6*sqrt(2*pi)) + 0.65*exp(-(x-1.4)^2/1.62)/(0.9*sqrt(2*pi))", domain: [-4, 5] }, stroke: "#8b949e", dash: true, label: "true density", extent: false },
  ],
});

// a noisy daily series: raw dots, rolling mean and median (window), and its mean as a rule
const wRng = normalRng(53);
const daily = { day: range(0, 365), temp: range(0, 365).map((d) => +(11 - 9 * Math.cos((2 * PI * (d - 15)) / 365) + 2.6 * wRng() + (wRng() > 2.2 ? 6 : 0)).toFixed(2)) };
const meanTemp = daily.temp.reduce((a, b) => a + b, 0) / daily.temp.length;
const rollingEntry = mark("marks_rolling", "daily temperature", "dot marks of raw data, window transforms (rolling mean, k = 15; median, k = 31) as lines, a rule at the mean", {
  subtitle: "synthetic year",
  xlabel: "day",
  ylabel: "°C",
  marks: [
    { type: "dot", data: daily, x: "day", y: "temp", fill: "#8b949e", r: 2.2, opacity: 0.6, label: "daily" },
    { type: "line", data: daily, x: "day", y: "temp", stroke: "#d95926", width: 2.5, transform: { window: { k: 15 } }, label: "15-day mean" },
    { type: "line", data: daily, x: "day", y: "temp", stroke: "#3987e5", width: 2, dash: true, transform: { window: { k: 31, reduce: "median" } }, label: "31-day median" },
    { type: "rule", y: +meanTemp.toFixed(2), stroke: "#199e70", dash: true, label: `mean ${meanTemp.toFixed(1)} °C` },
  ],
});

// four groups, one bimodal: violins (density per category, mirrored) with narrow box plots inside
const vRng = normalRng(59), vU = uniformRng(5959);
const vGroups = ["placebo", "low", "medium", "high"];
const trial = { group: [] as string[], response: [] as number[] };
vGroups.forEach((g, i) => range(0, 120).forEach(() => {
  trial.group.push(g);
  trial.response.push(+(i === 2 && vU() < 0.4 ? 9 + 0.8 * vRng() : 4 + 1.4 * i + (0.8 + 0.25 * i) * vRng()).toFixed(3));
}));
const violin = mark("marks_violin", "response by dose", "density transform per category as mirrored areas (violins), with the boxplot transform over them", {
  subtitle: "𝑛 = 120 per group · synthetic",
  ylabel: "response",
  legend: false,
  marks: [
    { type: "area", data: trial, x: "group", y: "response", fill: "group", opacity: 0.45, transform: { density: true } },
    { type: "box", data: trial, x: "group", y: "response", fill: "group", transform: { boxplot: { width: 0.12 } } },
  ],
});

// quarterly sales of three products: bars dodged by fill, error bars (interval) dodged alike
const gRng = normalRng(61);
const sales = { quarter: [] as string[], product: [] as string[], sales: [] as number[], lo: [] as number[], hi: [] as number[] };
["Q1", "Q2", "Q3", "Q4"].forEach((q, i) => ["basic", "plus", "pro"].forEach((p, j) => {
  const v = 20 + 6 * i + [8, 3, -4][j] * (i % 2 ? 1 : 0.5) + 3 * gRng();
  const e = 1.5 + 1.5 * Math.abs(gRng());
  sales.quarter.push(q);
  sales.product.push(p);
  sales.sales.push(+v.toFixed(1));
  sales.lo.push(+(v - e).toFixed(1));
  sales.hi.push(+(v + e).toFixed(1));
}));
const bars = mark("marks_bars", "sales by quarter", "bar marks on a categorical x, dodged by a categorical fill, with interval marks (error bars) dodged the same way", {
  subtitle: "synthetic · ± 1 s.e.",
  ylabel: "sales (k units)",
  marks: [
    { type: "bar", data: sales, x: "quarter", y: "sales", fill: "product" },
    { type: "interval", data: sales, x: "quarter", y1: "lo", y2: "hi", z: "product", stroke: "#c9d1d9", width: 1.5, legend: false },
  ],
});

// survey answers as raw rows: group (count, sorted) → horizontal bars, labelled with text marks
const aU = uniformRng(67);
const langs = [["Python", 0.31], ["JavaScript", 0.22], ["C++", 0.13], ["Rust", 0.1], ["Julia", 0.08], ["Go", 0.07], ["Fortran", 0.05], ["other", 0.04]] as const;
const answers = range(0, 400).map(() => {
  let u = aU();
  return { language: (langs.find(([, p]) => (u -= p) < 0) ?? langs[langs.length - 1])[0] };
});
const barx = mark("marks_barx", "favourite language", "barX with the group transform (count, sorted descending) on raw rows; text marks of the same transform label the bars", {
  subtitle: "𝑛 = 400 synthetic answers",
  xlabel: "answers",
  legend: false,
  marks: [
    { type: "barX", data: answers, y: "language", fill: "#199e70", transform: { group: { sort: "desc" } } },
    { type: "text", data: answers, y: "language", x: "x", text: "x", anchor: "lm", dx: 6, transform: { group: { sort: "desc" } } },
  ],
});

// activity by weekday and hour: rect cells coloured by a continuous fill (colour bar), values as text
const hRng = normalRng(71);
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const activity = days.flatMap((day, d) => range(6, 23).map((hour) => ({
  day, hour,
  load: Math.max(0, Math.round((d < 5 ? 60 * Math.exp(-((hour - 10.5) ** 2) / 6) + 45 * Math.exp(-((hour - 15) ** 2) / 5) : 35 * Math.exp(-((hour - 13) ** 2) / 12)) + 6 + 4 * hRng())),
})));
const heatmap = mark("marks_heatmap", "load by weekday and hour", "rect marks on a numeric x and categorical y, a continuous fill through a colormap (colour bar), text marks of the values", {
  subtitle: "synthetic · requests / min",
  xlabel: "hour",
  color: { scheme: "magma", label: "requests / min" },
  marks: [
    { type: "rect", data: activity, x: "hour", y: "day", fill: "load", inset: 1 },
    // text colour by the cell's lightness: two text marks over the rows split at 45
    { type: "text", data: activity.filter((a) => a.load <= 45), x: "hour", y: "day", text: "load", fill: "#e6edf3" },
    { type: "text", data: activity.filter((a) => a.load > 45), x: "hour", y: "day", text: "load", fill: "#101216" },
  ],
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
  poissonHist,
  exprNormal,
  exprSliders,
  ecdfBand,
  qqNormal,
  boxGroups,
  linregBand,
  bubble,
  stackedArea,
  binLine,
  rollingEntry,
  violin,
  bars,
  barx,
  heatmap,
  pchip("bumps", 32),
  pchip("runge", 12),
  pchip("blocks", 32),
];
