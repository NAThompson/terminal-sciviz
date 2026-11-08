// The "volumes" section: hydrogen orbitals, a 4D stack and tight-binding bands, generated on the GPU by
// volume.ts from small specs, with the datasets' display settings as controls: iso levels,
// log / symlog / sqrt / root scales, value transform, a non-uniform axis, a 4D stack (`slice`, stepped
// by the live view), a 3D signal and a Fermi surface; and a 4D stack read from a file (a .npy in
// assets/, made by tools/make_ball_modes.ts) with a slice slider.

import type { Entry } from "./types.ts";

const orbital = { dataset: "hydrogen", n: 3, l: 2, m: 1 };

export const entries: Entry[] = [
  {
    id: "volumes/orbital_iso",
    section: "volumes",
    title: "𝜓₃,₂,₁",
    description: "isosurfaces at ±0.01 over the orbital, diverging RdBu map on a symmetric domain",
    renderer: "volume.ts",
    control: { volume: { ...orbital, quantity: "psi" }, title: "𝜓₃,₂,₁", colormap: "RdBu", symmetric: true, isoLevels: [-0.01, 0.01], valueLabel: "𝜓" },
    interaction: "orbit",
  },
  {
    id: "volumes/density_log",
    section: "volumes",
    title: "|𝜓₃,₂,₁|²",
    description: "log color scale, domain from 10⁻⁹ to the maximum",
    renderer: "volume.ts",
    control: { volume: { ...orbital, quantity: "density" }, title: "|𝜓₃,₂,₁|²", scale: "log", domain: [1e-9, null], valueLabel: "|𝜓|²" },
    interaction: "orbit",
  },
  {
    id: "volumes/orbital_symlog",
    section: "volumes",
    title: "𝜓₃,₂,₁",
    description: "symlog with threshold 10⁻⁵, automatic map: diverging as the data change sign",
    renderer: "volume.ts",
    control: { volume: { ...orbital, quantity: "psi" }, title: "𝜓₃,₂,₁", scale: "symlog", symlogThreshold: 1e-5, colormap: "auto", valueLabel: "𝜓" },
    interaction: "orbit",
  },
  {
    id: "volumes/density_root",
    section: "volumes",
    title: "|𝜓₃,₂,₁|²",
    description: "root color scale, n = 3 (cube root of the normalised value): the faint outer lobes become visible",
    renderer: "volume.ts",
    control: { volume: { ...orbital, quantity: "density" }, title: "|𝜓₃,₂,₁|²", scale: "root", rootN: 3, valueLabel: "|𝜓|²" },
    interaction: "orbit",
  },
  {
    id: "volumes/density_log10",
    section: "volumes",
    title: "log₁₀ |𝜓₃,₂,₁|²",
    description: "values are log₁₀ of the density: color bar ticks as powers of 10",
    renderer: "volume.ts",
    control: { volume: { ...orbital, quantity: "log10_density" }, title: "log₁₀ |𝜓₃,₂,₁|²", valueTransform: "exp10", valueLabel: "|𝜓|²" },
    interaction: "orbit",
  },
  {
    id: "volumes/radial_nonuniform",
    section: "volumes",
    title: "|𝜓₃,₁,₀|²(𝜑, 𝜃, 𝑟)",
    description: "non-uniform (log-spaced) 𝑟 axis, sqrt scale",
    renderer: "volume.ts",
    control: { volume: { dataset: "hydrogen_spherical", n: 3, l: 1, m: 0 }, title: "|𝜓₃,₁,₀|²(𝜑, 𝜃, 𝑟)", scale: "sqrt", valueLabel: "|𝜓|²" },
    interaction: "orbit",
  },
  {
    id: "volumes/breathing_stack",
    section: "volumes",
    title: "breathing Gaussian",
    description: "4D stack (t, z, y, x): the leading dimension is sliced, played back at 4 frames/s",
    renderer: "volume.ts",
    control: { volume: { dataset: "breathing", size: 32, frames: 8 }, slice: 0, title: "breathing Gaussian" },
    interaction: "orbit",
    playback: { fps: 4, slices: 8 },
  },
  {
    id: "volumes/tight_binding",
    section: "volumes",
    title: "Tight-binding band 𝐸(𝐤)",
    description: "Nearest-neighbour tight-binding band E(k) = −2(cos kx + cos ky + cos kz), as a volume",
    renderer: "volume.ts",
    control: { volume: { dataset: "tight_binding", size: 40 }, title: "Tight-binding band 𝐸(𝐤)", valueLabel: "𝐸 (eV)" },
    interaction: "orbit",
  },
  {
    id: "volumes/fermi_surface",
    section: "volumes",
    title: "Fermi surface",
    description: "E(k) = E_F of a simple cubic tight-binding band (t = 1, t' = 0.3 eV) at half filling, as an isosurface over the band volume",
    renderer: "volume.ts",
    control: { volume: { dataset: "fermi", size: 64 }, colormap: "Twilight", valueLabel: "𝐸 (eV)", scale: "root", rootN: 3 }, // iso level, centre, title: E_F (the median)
    interaction: "orbit",
  },
  {
    id: "volumes/file_slices",
    section: "volumes",
    title: "vibrating ball",
    description: "a 4D stack from a .npy file (11 × 24³, NaN outside the ball: transparent), an in-frame slider over 𝑡",
    renderer: "volume.ts",
    control: {
      volume: {
        dataset: "file", path: "assets/ball_modes.npy",
        axes: [{ lo: -1, hi: 1, label: "𝑥" }, { lo: -1, hi: 1, label: "𝑦" }, { lo: -1, hi: 1, label: "𝑧" }, { lo: 0, hi: 1, label: "𝑡" }],
      },
      slice: 2, title: "vibrating ball", subtitle: "𝑗₀(𝜋𝑟) cos 2𝜋𝑡 + 𝑗₁ cos 𝜃 sin 2𝜋𝑡", colormap: "RdBu", symmetric: true, valueLabel: "𝑢", up: "z",
    },
    interaction: "orbit",
  },
];
