---
name: terminal-graphics
description: How to present graphics to the user inside the terminal chat with the sciviz mod — static images (mcp__sciviz__show), live interactive WebGPU views (mcp__sciviz__gallery_open, mcp__sciviz__view, mcp__sciviz__plot), and how to make plots that look right there. Use whenever a picture, plot, comparison, 3D view or interactive exploration would help the user, before writing any plotting code, and whenever you're tempted to describe something visual in words.
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
| An interactive view of a gallery entry (lines, curves, complex, volumes...) | `mcp__sciviz__gallery_open` with its `id` (fuzzy names work: `blocks`, `lorenz`); `control` overrides fields (window `x0`/`x1`, camera `azimuth`/`elevation`/`zoom`, `colormap`) |
| A live 3D surface: built-in (`torus`, `sphere`, `wave`, `dini`, `breather`) or **your own** | `mcp__sciviz__view` with `surface`, or with `nurbs` (see below) |
| A live PCHIP interpolation plot | `mcp__sciviz__plot` |

Live views render on the GPU (WebGPU) and stream frames into the chat. **The newest view is live;
earlier ones freeze** on their last settled frame, so the conversation keeps a visual history.
Every live-view tool returns a `snapshot` PNG path: **Read it** to see exactly what the user sees
before you talk about it, and the renderer `state` (window, value ranges, camera, hover).

In 3D views only the object itself takes the mouse; the user can select text in the empty space
around it. Tell the user how to drive a live view the first time: drag (pan in 2D, orbit in 3D),
shift-drag or `+`/`-` (zoom), hover (values drawn next to the cursor), click the view then
`c`/`C` (cycle colour maps within their group), `i` (invert), `r` (reset). Their one slash command:
`/sciviz [section]` shows a contact sheet, `/sciviz <entry>` (e.g. `lorenz`) opens a live view.

## Where to compute the data

Compute as close to the renderer as you can, in this order:

1. **On the GPU, by the renderer.** Gallery entries and generator specs (`fn: {name: "besselJ", order: 3}`,
   ODEs, orbitals, complex functions) are evaluated in compute shaders. The data never leaves the
   GPU and is re-sampled live as the user pans and zooms. Reach for an existing entry and override
   its `control` before anything else.
2. **Exactly, as a NURBS.** Many surfaces have an exact rational form, and you can pass it to `view` as
   `nurbs`. A polynomial graph z = f(x, y) of degree (p, q) is one Bézier patch: x and y are
   linear in the parameters, and z's control values are f's Bernstein coefficients (solve the
   (p+1)-point collocation in each direction). Scalar fields go in `fields` in the same basis, e.g.
   det ∇²f to show a saddle, coloured with a diverging map (`rdbu`, `invertColorMap: true`, symmetric
   `domain`). Revolved and conic shapes are exact with rational weights. The renderer evaluates it,
   normals and curvature included, on the GPU.
3. **Only for data that exists as measurements:** compute it elsewhere and send arrays, or plot a
   PNG and `show` it. Don't sample a mathematical function in Python and ship its values as JSON
   when tier 1 or 2 can draw it: that path can't re-sample when the user zooms, and the arrays
   flood the control.

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
- Don't put big data arrays into a live view's `control` overrides: the control is rewritten on
  every pointer event. Entries carry small generator specs (e.g. an ODE or a function name) that
  the renderer evaluates on the GPU. A `nurbs` surface should be a small exact patch or a coarse
  control net (under 64 kB), not a dense sample grid.
- Don't open a new live view for every small change: one view the user drives is better than five
  frozen ones. Change parameters with the tool's `control` only when the user asks or you're
  showing something specific.

## Where things live

The renderers are Deno + WebGPU programs in the plugin's `deno/` directory; the gallery's entries are in
`deno/gallery/*_entries.ts`, `deno task gallery` renders contact sheets, `deno task gallery-check`
diffs every entry against golden images. Each renderer speaks a small control-file protocol
(`deno/lib/serve.ts`): the mod writes `control.json` (with an increasing `rev`), the renderer prints
`frame <gen> <w> <h> <shm> <rev>` and `snapshot <rev> <path>`.
