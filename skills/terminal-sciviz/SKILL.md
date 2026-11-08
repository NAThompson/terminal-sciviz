---
name: terminal-sciviz
description: How to present graphics to the user inside the terminal chat with the sciviz mod — static images (mcp__sciviz__show), live interactive WebGPU views (mcp__sciviz__chart, mcp__sciviz__scene, mcp__sciviz__volume, mcp__sciviz__gallery_open, mcp__sciviz__view, mcp__sciviz__plot), and how to make plots that look right there. Use whenever a picture, plot, comparison, 3D view or interactive exploration would help the user, before writing any plotting code, and whenever you're tempted to describe something visual in words.
---

# Presenting graphics in the terminal

The sciviz mod draws real pixels inline in the chat, between your text, through the terminal's
graphics protocol (kitty graphics; Ghostty and kitty support it). You decide to show a graphic and
the user sees it — no commands for them to run. **Prefer showing over describing**: when a choice
has a visible effect, render the alternatives side by side and show them before asking.

## Which tool

| You have / want | Use |
|---|---|
| A PNG you made (matplotlib, a renderer, a screenshot) | `mcp__sciviz__show` with the absolute `path` (and a one-line `caption`) |
| A live chart of **your own** data or functions (scatter/bubble, lines, bars, areas, histograms, heatmaps, distributions, fits) | `mcp__sciviz__chart` with `marks`: each a type (`line`, `dot`, `bar`, `barX`, `area`, `rect`, `rule`, `text`, `interval`, `box`) over `data` columns (or a CSV/JSON `file` with a header, or a function `fn`), channels naming columns (`x`, `y`, `fill`, `r`, `z`, ...) and `transform`s (below). A later call without `marks`/`series` adjusts the same chart (window, title, scales). `series` is the older one-line-per-entry shorthand |
| A live 3D scene of **your own** data: 3D scatter, point clouds, 3D polylines, triangle meshes, convex hulls | `mcp__sciviz__scene` with `marks` (`dot`, `line`, `mesh`) over `data` columns or a `file`, channels `x`, `y`, `z`, `fill`, `r` like `chart` (below) |
| A live 3D volume of **your own** array (a 3D field, or a 4D stack over time / a parameter) | `mcp__sciviz__volume` with `volume: {dataset: "file", path}`: a `.npy` (or raw little-endian `.f32` with `shape`), `axes` [x, y, z, slice?] `{lo, hi, label, unit}`; NaN = transparent; a 4D array gets a slice slider. Colour/opacity/iso fields like the volume gallery entries |
| An interactive view of a gallery entry (lines, curves, complex, volumes, scenes...) | `mcp__sciviz__gallery_open` with its `id` (fuzzy names work: `blocks`, `lorenz`); `control` overrides fields (window `x0`/`x1`, camera `azimuth`/`elevation`/`zoom`, `colormap`) |
| A live 3D surface: built-in (`torus`, `sphere`, `wave`, `dini`, `breather`, `teapot`) or **your own** | `mcp__sciviz__view` with `surface`, or with `nurbs` (one patch or an array of patches; see below) |
| A live PCHIP interpolation plot | `mcp__sciviz__plot` |

Live views render on the GPU (WebGPU) and stream frames into the chat. **The newest view is live;
earlier ones freeze** on their last settled frame, so the conversation keeps a visual history.
Every live-view tool returns a `snapshot` PNG path: **Read it** to see exactly what the user sees
before you talk about it, and the renderer `state` (window, value ranges, camera, hover).

**Write figures into the explanation, like a paper.** The call itself isn't shown (only the
figure), so the text around it must carry the argument: lead in with what to look at ("the counts
sit just under the fit at k = 3–4, and the normal approximation leans right"), call the tool, then
continue from what the snapshot shows. Don't announce the tool ("let me render…", "here is the
chart") or restate its parameters; put the figure where it supports the sentence, more than one
per answer if the argument has steps.

In 3D views only the object itself takes the mouse; the user can select text in the empty space
around it. Tell the user how to drive a live view the first time: drag (pan in 2D, orbit in 3D),
shift-drag or `+`/`-` (zoom), hover (values drawn next to the cursor), click the view then
`c`/`C` (cycle colour maps within their group), `i` (invert), `r` (reset). Their one slash command:
`/sciviz` shows an overview, `/sciviz <section>` a contact sheet (lines, curves, complex, volumes,
scenes, surfaces), `/sciviz <entry>` (e.g. `lorenz`, `teapot`) opens a live view.

## Where to compute the data

Compute as close to the renderer as you can, in this order:

1. **By the renderer.** Gallery entries and generator specs (`fn: {name: "besselJ", order: 3}`,
   ODEs, orbitals, complex functions) are evaluated by the renderer itself: function curves in f64
   on the CPU (special functions by Boost.Math), re-sampled live over the visible window as the user
   pans and zooms; volumes and surfaces on the GPU. Reach for an existing entry (overriding its
   `control`), or a `chart` line mark with `fn`, before anything else. Any closed-form f(x) works as
   `fn: {expr: "a*exp(x*log(lam) - lam - lgamma(x+1))", params: {a: 2000, lam: 4.3}, domain: [0, 16]}`
   (a fit over a histogram, a density, a model curve); parse errors come back in the
   tool result. Titles and labels take Unicode math (𝑥, 𝜆̂, 𝑥̄, ℝⁿ, 10⁻⁶); `state.missingGlyphs`
   lists any character the font lacks.
2. **Exactly, as a NURBS.** Many surfaces have an exact rational form, and you can pass it to `view` as
   `nurbs`. A polynomial graph z = f(x, y) of degree (p, q) is one Bézier patch: x and y are
   linear in the parameters, and z's control values are f's Bernstein coefficients (solve the
   (p+1)-point collocation in each direction). Scalar fields go in `fields` in the same basis, e.g.
   det ∇²f to show a saddle, coloured with a diverging map (`rdbu`, `invertColorMap: true`, symmetric
   `domain`). Revolved and conic shapes are exact with rational weights. The renderer evaluates it,
   normals and curvature included, on the GPU.
3. **Only for data that exists as measurements:** compute it elsewhere and `chart` it: small
   arrays inline, anything bigger (past 64 kB of JSON) written to a `.json`/`.csv`/`.tsv` file
   given as the mark's (or series') `file` (the renderer reads it once, and again when it changes). Or plot a
   PNG and `show` it. Don't sample a mathematical function in Python and ship its values as JSON
   when tier 1 or 2 can draw it: that path can't re-sample when the user zooms, and the arrays
   flood the control.

**Sliders for parameters.** To let the user explore a parameter, add `sliders: {name: [min, max]}`
(or `{min, max, step?, label?}`) to `chart` (it sets the fn param of that name, an expr's `params`
or a named fn's `order`/`lambda`/`tau`/`omega`/`zeta`, in every series and mark) or to a
`gallery_open` control (curve ODEs: `curves/lorenz` with `{sliders: {rho: [0, 50]}}`, also `sigma`,
`beta`). They're drawn in the frame; dragging re-renders live. `state.sliders` reports the values the
user set (call `chart` again without `marks` to read them); unknown names come back as an error.
**Sliders over data**: give a `chart` or `scene` mark `slice: "t"` (a numeric column) and only the
rows at the `t` slider's value are drawn, snapping to the column's values (a slider is added; set
`sliders: {t: {label, value}}` to name or start it). Put every time step in one mark: a heatmap over
time is `{type: "rect", data, x, y, fill: "u", slice: "t"}`. Volumes slice their leading axis the same way.

**Charts are combinations of marks, channels and transforms**, not special options. Strings in
`x`/`y` become categories; a string `fill` colours by category (Okabe–Ito palette legend, at most 7; it also
groups rows like `z`), a numeric `fill` through a colormap (`color: {scheme, domain}`, colour bar);
`r` is a sqrt (area) scale. E.g. bubble `{type: "dot", data, x: "gdp", y: "life", r: "pop", fill: "continent"}`;
stacked area `{type: "area", data, x: "year", y: "v", fill: "sector", transform: {stack: true}}`
(add the same with `type: "line"` for edges); grouped bars `{type: "bar", x: "quarter", y: "sales", fill: "product"}`
(stack: true stacks them); horizontal counts of raw rows `{type: "barX", y: "lang", transform: {group: {sort: "desc"}}}`
with `{type: "text", y: "lang", x: "x", text: "x", anchor: "lm", dx: 6, transform: {group: {sort: "desc"}}}`;
heatmap `{type: "rect", x: "hour", y: "day", fill: "load"}`; a reference `{type: "rule", y: 0}`.
Five or more labelled lines also get their names at their right ends (`endLabels: false|true` overrides).

**3D data: `scene`.** Its marks work like `chart`'s, with a `z` channel: `{type: "dot", data, x, y, z,
fill: "species"}` (categories, legend) or `fill: "t"` (colour map and bar), `r` px or a column; dots
are shaded spheres, fast to 10⁵ points (give a `file` past 64 kB). `line` draws polylines in row
order, one per `group` (default: a categorical fill). `mesh` takes `vertices` + `faces` (or x/y/z
columns + `faces`), `shading: "smooth"`, `edges: true`, `opacity` (translucent meshes let what's
behind them show, dimmed). `transform: {hull: true}` on a dot mark draws its convex hull; add the
points again as a second dot mark to see them inside. Equal aspect by default; `aspect: "auto"`
when the axes have unrelated units. Hover reports the point's row. Use `chart` for anything 2D.

**Distributions and fits: pass raw data with a transform**, not bins or quantiles you computed:
`bin` (`width`/`thresholds`, `reduce: count|density|proportion`) under a `bar` (histogram) or a
`line` (frequency polygon); `density` (Gaussian KDE, Silverman bandwidth) under a `line`, or with a
categorical x and numeric y under an `area` (violins), with `{type: "box", x, y}` over it;
`window: {k, reduce}` for rolling means/medians; `ecdf: {band: 0.05}` (step ECDF + DKW band), `qq`,
`linreg: {kind: "mean"|"prediction"}` (fit and t band as expr fns). `state.stats` returns the
numbers (n, quartiles, a, b, r², ...). Hover shows the hovered item's channels by column name.

## Making images that look right in the terminal

- **Transparent background.** Images are composited over the terminal's own background, so never
  bake one in: matplotlib `fig.savefig(path, transparent=True)` (and `fig.patch.set_alpha(0)`,
  `ax.patch.set_alpha(0)`). A white or grey box around a plot is the most common mistake.
- **Ink from the user's theme.** Read it: `ghostty +show-config | grep -E '^(theme|background|foreground) '`
  (kitty: `kitty @ get-colors` with remote control enabled, else `foreground`/`background` in
  `~/.config/kitty/kitty.conf`). Use the foreground for labels and a brighter shade for titles; grid lines barely above
  the background. Pick series colours that read on that background.
- **Full width, scrolling.** `show` fills the chat width at the image's aspect ratio; tall images
  run past the screen and scroll with the text. So make figures **wide** (≈ 16:10 or wider) unless
  they really are tall, and don't shrink them by passing `columns`/`rows` unless asked.
- **Native resolution.** Text is sharpest when the image's pixel width matches the box it's shown
  in. Cell size in device pixels: read `TIOCGWINSZ` on the session's tty (`ws_xpixel / cols`,
  `ws_ypixel / rows`, minus window padding; e.g. 22 × 48 px per cell on a Retina display at font
  size 18). A figure ≈ `(cols − 6) × cell width` pixels wide shows 1:1. Otherwise render at ≥ 2×
  the logical size (`dpi` 200+).
- **Look before you show.** Read the PNG yourself first; fix overlapping labels, clipped legends,
  unreadable ticks. Then show it.
- **Comparisons:** put alternatives in one labelled grid (current vs candidates, the key numbers in
  each label) rather than several separate images.

## Don'ts

- Don't print kitty/sixel escape sequences from the Bash tool: tool output isn't passed to the
  terminal raw, so nothing renders. Use `show`.
- Don't put big data arrays into a live view's control (`gallery_open` overrides, `chart` inline
  `data`): the control is rewritten on every pointer event. Give `chart` a `file` instead, and
  don't borrow a gallery entry for ad-hoc data. Entries carry small generator specs (e.g. an ODE or a function name) that
  the renderer evaluates on the GPU. A `nurbs` surface should be small exact patches (one, or an
  array of up to 256, e.g. Bézier patches) or a coarse control net (under 64 kB), not a dense sample grid.
  3D curves: `lineWidth` is in screen pixels at any depth; `lineCap: "round"` gives round joins and caps
  for thick lines (see `curves/hilbert_fat`).
- Don't open a new live view for every small change: one view the user drives is better than five
  frozen ones. Change parameters with the tool's `control` only when the user asks or you're
  showing something specific.

## Where things live

The renderers are Deno + WebGPU programs in the plugin's `deno/` directory; the gallery's entries are in
`deno/gallery/*_entries.ts`, `deno task gallery` renders contact sheets, `deno task gallery-check`
diffs every entry against golden images. Each renderer speaks a small control-file protocol
(`deno/lib/serve.ts`): the mod writes `control.json` (with an increasing `rev`), the renderer prints
`frame <gen> <w> <h> <shm> <rev>` and `snapshot <rev> <path>`.
