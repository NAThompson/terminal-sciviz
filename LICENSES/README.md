# Third-party licences

terminal-sciviz itself is MIT-licensed (see [LICENSE](LICENSE)). It bundles four kinds of
third-party material, which keep their own licences:

1. **Colour-map tables**: the 256-sample sRGB tables in
   [`deno/lib/colormaps.ts`](deno/lib/colormaps.ts), generated once by
   [`deno/tools/colormap_tables.ts`](deno/tools/colormap_tables.ts) and committed.
2. **Fonts**: the signed-distance-field atlas [`deno/assets/font.png`](deno/assets/font.png) /
   [`font.json`](deno/assets/font.json), built by [`deno/tools/font_atlas.py`](deno/tools/font_atlas.py)
   from JetBrains Mono and STIX Two Math, whose font files ship in
   [`deno/assets/fonts/`](../deno/assets/fonts/).
3. **The Utah teapot**: [`deno/assets/teapot.json`](../deno/assets/teapot.json), see
   [Teapot](#teapot).
4. **Boost.Math**: the WebAssembly build in
   [`deno/vendor/boost-math-wasm/`](../deno/vendor/boost-math-wasm/), see [Boost.Math](#boostmath).

Boost.Math is the only third-party code that runs at run time. The generator fetches the npm packages
`d3-scale-chromatic@3.1.0`, `d3-color@3.1.0` and `d3-interpolate@3.0.1` when it is run; only the
colour values it computes with them are committed.


## Colour maps

### Provenance of each table

"Via" is the `d3-scale-chromatic@3.1.0` function the table was sampled from. Every table is 256
samples at t = i / 255; the single-hue ColorBrewer maps are reversed (t → 1 − t) so they run dark
to light.

| Table | Original design | Via | Licence of the data |
|---|---|---|---|
| Blues | ColorBrewer, Cynthia A. Brewer (Brewer, Harrower, Penn State) | `interpolateBlues`, reversed | Apache-2.0 (ColorBrewer); ISC (d3-scale-chromatic) |
| Greens | ColorBrewer | `interpolateGreens`, reversed | Apache-2.0; ISC |
| Greys | ColorBrewer | `interpolateGreys`, reversed | Apache-2.0; ISC |
| Oranges | ColorBrewer | `interpolateOranges`, reversed | Apache-2.0; ISC |
| Purples | ColorBrewer | `interpolatePurples`, reversed | Apache-2.0; ISC |
| Reds | ColorBrewer | `interpolateReds`, reversed | Apache-2.0; ISC |
| Turbo | Anton Mikhailov, Google LLC (2019) | `interpolateTurbo`: Mike Bostock's fifth-order polynomial approximation of Google's 256-entry table | Apache-2.0 (Google's table); ISC (d3's approximation) |
| Viridis | Stéfan van der Walt, Nathaniel J. Smith and Eric Firing, for matplotlib (BIDS colormap project) | `interpolateViridis` (embeds the 256-entry table) | CC0 1.0; ISC |
| Inferno | Stéfan van der Walt and Nathaniel J. Smith (BIDS colormap project) | `interpolateInferno` | CC0 1.0; ISC |
| Magma | Stéfan van der Walt and Nathaniel J. Smith (BIDS colormap project) | `interpolateMagma` | CC0 1.0; ISC |
| Plasma | Stéfan van der Walt and Nathaniel J. Smith (BIDS colormap project) | `interpolatePlasma` | CC0 1.0; ISC |
| Cividis | Jamie R. Nuñez, Christopher R. Anderton and Ryan S. Renslow, PNNL ([PLOS ONE 13(7): e0199239, 2018](https://doi.org/10.1371/journal.pone.0199239)) | `interpolateCividis`: Mike Bostock's fifth-order polynomial approximation of the cividis table | ISC (d3's approximation); the reference table is under the Battelle BSD-style licence below |
| Warm | d3: a 180° hue rotation of Matteo Niccoli's perceptual rainbow, in cubehelix space | `interpolateWarm` | ISC |
| Cool | d3: after Matteo Niccoli's perceptual rainbow, in cubehelix space | `interpolateCool` | ISC |
| Cubehelix | Dave Green's cubehelix scheme ([Bull. Astr. Soc. India 39, 289, 2011](https://arxiv.org/abs/1108.5083)), default parameters as set by d3 | `interpolateCubehelixDefault` | ISC (computed from the published formula; no data copied) |
| BuGn | ColorBrewer | `interpolateBuGn` | Apache-2.0; ISC |
| BuPu | ColorBrewer | `interpolateBuPu` | Apache-2.0; ISC |
| GnBu | ColorBrewer | `interpolateGnBu` | Apache-2.0; ISC |
| OrRd | ColorBrewer | `interpolateOrRd` | Apache-2.0; ISC |
| PuBuGn | ColorBrewer | `interpolatePuBuGn` | Apache-2.0; ISC |
| PuBu | ColorBrewer | `interpolatePuBu` | Apache-2.0; ISC |
| PuRd | ColorBrewer | `interpolatePuRd` | Apache-2.0; ISC |
| RdPu | ColorBrewer | `interpolateRdPu` | Apache-2.0; ISC |
| YlGnBu | ColorBrewer | `interpolateYlGnBu` | Apache-2.0; ISC |
| YlGn | ColorBrewer | `interpolateYlGn` | Apache-2.0; ISC |
| YlOrBr | ColorBrewer | `interpolateYlOrBr` | Apache-2.0; ISC |
| YlOrRd | ColorBrewer | `interpolateYlOrRd` | Apache-2.0; ISC |
| Twilight | Bastian Bechtold, contributed to matplotlib as `twilight` (matplotlib 3.0) | not d3: 16 samples of matplotlib's table, joined by d3-interpolate's `interpolateRgbBasisClosed` (see [Twilight](#twilight)) | matplotlib License |
| Rainbow | d3 (Mike Bostock): Warm on [0, 0.5] then Cool on [0.5, 1], the cyclic "less-angry rainbow" | `interpolateRainbow` | ISC |
| Sinebow | Jim Bumgardner and Charlie Loyd | `interpolateSinebow` (computed from the sine formula) | ISC |
| HSL | generated locally: `hsl(360·t, 100 %, 50 %)` converted to sRGB | none (d3-color's HSL→RGB conversion only) | no third-party data; MIT, as the rest of this project |
| BrBG | ColorBrewer | `interpolateBrBG` | Apache-2.0; ISC |
| PRGn | ColorBrewer | `interpolatePRGn` | Apache-2.0; ISC |
| PiYG | ColorBrewer | `interpolatePiYG` | Apache-2.0; ISC |
| PuOr | ColorBrewer | `interpolatePuOr` | Apache-2.0; ISC |
| RdBu | ColorBrewer | `interpolateRdBu` | Apache-2.0; ISC |
| RdGy | ColorBrewer | `interpolateRdGy` | Apache-2.0; ISC |
| RdYlBu | ColorBrewer | `interpolateRdYlBu` | Apache-2.0; ISC |
| RdYlGn | ColorBrewer | `interpolateRdYlGn` | Apache-2.0; ISC |
| Spectral | ColorBrewer | `interpolateSpectral` | Apache-2.0; ISC |

Sources: the d3-scale-chromatic documentation and source at v3.1.0 (credits for Turbo, Viridis,
Inferno, Magma, Plasma, Cividis, Warm, Cool, Cubehelix, Rainbow and Sinebow; ColorBrewer for the
rest), the BIDS colormap repository (CC0 dedication), Google's Turbo look-up table (Apache-2.0
header), PNNL's cmaputil repository (cividis), and matplotlib (twilight).

### How the tables were made

- **ColorBrewer maps.** ColorBrewer publishes discrete schemes only; d3-scale-chromatic
  interpolates them with uniform B-splines, and we sample that interpolation.
- **Turbo and Cividis.** d3's versions are not the original look-up tables but fifth-order
  polynomial fits to them, published by Mike Bostock in d3-scale-chromatic 1.5.0 (2019). The
  Turbo fit is d3's own, not the GLSL approximation by Ruofei Du that accompanies Google's table.
  Our tables are samples of these polynomials, so they differ slightly from the originals.
- **Viridis, Inferno, Magma, Plasma.** d3 embeds the 256-entry tables as hex strings and picks the
  nearest entry; our samples reproduce them.
- **Warm, Cool, Rainbow, Cubehelix, Sinebow, HSL.** Computed from formulas (cubehelix
  interpolation, sine curves, HSL conversion); no colour data is copied.

The d3 packages' code is not shipped; only the values they computed are. Their ISC notices are
kept below.

### Twilight

`Twilight` is a closed uniform B-spline (d3-interpolate's `interpolateRgbBasisClosed`) through 16
stops taken from matplotlib's `twilight` colormap. The stops are the 510-entry table
`_twilight_data` in matplotlib's `lib/matplotlib/_cm_listed.py` at indices
⌊k · 510 / 16⌋, k = 0 … 15 (that is, matplotlib's `twilight(k / 16)`), rounded to 8-bit hex; all
16 match exactly.

matplotlib's twilight was designed by Bastian Bechtold with the viscm tool and contributed to
matplotlib in pull request [#6254](https://github.com/matplotlib/matplotlib/pull/6254), released
in matplotlib 3.0. His original repository, <https://github.com/bastibe/twilight>, carries no
licence file; the colour data we use is the version distributed by matplotlib, under the
matplotlib License (reproduced [below](matplotlib.txt)).

As clause 3 of that licence asks, a summary of our changes: we took 16 evenly spaced entries of
matplotlib's 510-entry `twilight` table, rounded them to 8-bit sRGB, joined them with a periodic
uniform cubic B-spline in RGB, and resampled that at 256 points. Nothing else from matplotlib is
used.

## Fonts

| Font | Files | Copyright | Licence |
|---|---|---|---|
| JetBrains Mono Regular and Bold | [`deno/assets/fonts/JetBrainsMono/`](../deno/assets/fonts/JetBrainsMono/) | Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) | SIL Open Font License 1.1: [`deno/assets/fonts/JetBrainsMono/OFL.txt`](../deno/assets/fonts/JetBrainsMono/OFL.txt) |
| STIX Two Math | [`deno/assets/fonts/STIXTwoMath/STIXTwoMath-Regular.otf`](../deno/assets/fonts/STIXTwoMath/STIXTwoMath-Regular.otf) | Copyright 2001-2021 The STIX Fonts Project Authors (https://github.com/stipub/stixfonts), published by the STI Pub companies | SIL Open Font License 1.1: [`deno/assets/fonts/STIXTwoMath/OFL.txt`](../deno/assets/fonts/STIXTwoMath/OFL.txt) |

JetBrains Mono is by JetBrains; STIX Two by the STI Pub companies, through the STIX (Scientific and
Technical Information Exchange) project. The font files are redistributed unmodified with their licences. The SDF
atlas (`deno/assets/font.png`, `deno/assets/font.json`) is rendered from glyphs of these fonts and
is likewise covered by the SIL Open Font License 1.1; it is not a font and uses no Reserved Font
Name.

## Teapot

`deno/assets/teapot.json` holds Martin Newell's Utah teapot (University of Utah, 1975): 32
bicubic Bézier patches over 290 control points, in the version with Jim Blinn's vertical scaling
(heights divided by about 1.3), z up. The numbers were copied from the `teapotPatches` and
`teapotVertices` arrays of three.js's `examples/jsm/geometries/TeapotGeometry.js` (Eric Haines's
conversion of his SPD data), with the patch indices kept 0-based as there and the flat arrays
grouped per patch and per vertex. three.js is under the MIT licence ([below](threejs-MIT.txt), Copyright © 2010-2024
three.js authors); the teapot data itself has been freely redistributed since its publication and
carries no licence of its own. `deno/lib/surfaces.ts` reads it as 32 degree-3 × 3 NURBS patches.

## Boost.Math

`deno/vendor/boost-math-wasm/` is boost-math-wasm 0.5.0, an unofficial Emscripten build of
[Boost.Math](https://www.boost.org/libs/math) from the fork <https://github.com/NAThompson/math>
(release `boost-math-v0.5.0`), redistributed unmodified; its README there gives the source URL
and SHA-256. `deno/lib/boost.ts` loads it to evaluate special functions, function curves and
distribution quantiles in double precision. Boost.Math is under the Boost Software License 1.0,
the Emscripten runtime and the musl libc parts compiled in under their MIT-style licences; all are
in [`deno/vendor/boost-math-wasm/LICENSE`](../deno/vendor/boost-math-wasm/LICENSE).

## Licence texts

- [d3-scale-chromatic (ISC) and ColorBrewer (Apache-2.0)](d3-scale-chromatic-ISC-and-ColorBrewer.txt)
- [d3-color (ISC)](d3-color-ISC.txt)
- [d3-interpolate (ISC)](d3-interpolate-ISC.txt)
- [matplotlib (Twilight)](matplotlib.txt)
- [Cividis reference implementation (Battelle)](cividis-Battelle.txt)
- [Viridis, Magma, Inferno, Plasma (CC0)](BIDS-colormaps-CC0.txt)
- [Turbo (Apache-2.0)](turbo-Google.txt)
- [Apache License 2.0](Apache-2.0.txt)
- [three.js (MIT; teapot data)](threejs-MIT.txt)
- [Boost.Math (BSL-1.0), Emscripten, musl](../deno/vendor/boost-math-wasm/LICENSE)
