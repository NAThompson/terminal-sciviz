import type { Register } from 'claude-code'

const TOOL = 'mcp__sciviz__show'
type Shown = {
  path: string
  columns?: number // requested; absent: fit the width of the transcript
  rows?: number
  width: number // PNG pixels
  height: number
  caption: string
  generation: number
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
function decodeBase64(text: string): Uint8Array {
  const clean = text.replace(/[^A-Za-z0-9+/]/g, '')
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4))
  let bits = 0, acc = 0, n = 0
  for (const ch of clean) {
    acc = (acc << 6) | B64.indexOf(ch)
    bits += 6
    if (bits >= 8) { bits -= 8; out[n++] = (acc >> bits) & 255 }
  }
  return out.subarray(0, n)
}

// Ghostty's cell, in device pixels (font-size 18, Retina): TIOCGWINSZ gives 136 x 35 cells in 3010 x 1706 px
// including padding, i.e. integer 22 x 48 px cells. Frames rendered at exactly cols x 22 by rows x 48 are
// shown 1:1, with no resampling (which softens thin text).
const CELL_PX_W = 22, CELL_PX_H = 48
const CELL_ASPECT = CELL_PX_H / CELL_PX_W // cell height / width
const RESULT_INDENT = 6 // columns the transcript indents a tool result by

/** Cells for an image: the requested size, else as wide as the transcript at the image's aspect
 * ratio. Tall images run past the screen and scroll with the chat (up to the 255-row limit),
 * rather than shrinking to a narrow strip. */
function cellSize(s: Shown, viewport?: { columns: number; rows: number }) {
  const clamp = (n: number) => Math.max(1, Math.min(255, Math.round(n)))
  const ratio = s.height / s.width / CELL_ASPECT // rows per column
  if (s.columns && s.rows) return { columns: clamp(s.columns), rows: clamp(s.rows) }
  if (s.columns) return { columns: clamp(s.columns), rows: clamp(s.columns * ratio) }
  if (s.rows) return { columns: clamp(s.rows / ratio), rows: clamp(s.rows) }
  let columns = Math.max(10, (viewport?.columns ?? 100) - RESULT_INDENT)
  if (columns * ratio > 255) columns = 255 / ratio // the Image element's row limit
  return { columns: clamp(columns), rows: clamp(columns * ratio) }
}


// ---- live views, inline in the transcript, each kind drawn by its own GPU renderer process ----
// surface: deno/surface.ts, pchip: deno/plot1d.ts; gallery entries: deno/{line,curve,complex,volume,scene}.ts (all WebGPU).
// Protocol: control.json carries an increasing `rev`; frame lines are `frame <gen> <w> <h> <shm|-> <rev>`;
// a snapshot request {path, rev} is acknowledged by `snapshot <rev> <path>` once a frame at/after rev
// is saved. Writes are coalesced (the renderer retries a torn file and ignores stale revs).
// The newest view (a tool result or a /sciviz output) is live: frames are blitted into it and a
// transparent Client over it takes the pointer and keys. Older views freeze on the PNG of their
// last settled frame (`${VIEWS}/<id>.png`).
const VIEW_TOOL = 'mcp__sciviz__view'
const PLOT_TOOL = 'mcp__sciviz__plot'
const OPEN_TOOL = 'mcp__sciviz__gallery_open'
const CHART_TOOL = 'mcp__sciviz__chart'
const CHART_KEYS = ['title', 'subtitle', 'xlabel', 'ylabel', 'x', 'y', 'series', 'marks', 'color', 'r', 'overlays', 'legend', 'endLabels', 'x0', 'x1', 'ymin', 'ymax', 'sliders']
const SCENE_TOOL = 'mcp__sciviz__scene'
const SCENE_KEYS = ['title', 'subtitle', 'marks', 'xlabel', 'ylabel', 'zlabel', 'aspect', 'color', 'r', 'legend', 'axes', 'azimuth', 'elevation', 'zoom', 'sliders']
const VOLUME_TOOL = 'mcp__sciviz__volume'
const VOLUME_KEYS = ['volume', 'slice', 'sliders', 'title', 'subtitle', 'colormap', 'invertColormap', 'scale', 'rootN', 'symlogThreshold', 'domain', 'symmetric', 'center',
  'mode', 'opacity', 'threshold', 'quality', 'isoLevels', 'isoColors', 'showVolume', 'showBox', 'showAxes', 'valueLabel', 'up', 'azimuth', 'elevation', 'zoom']
// Resolved at session start (resolvePaths): ROOT holds deno/ (the plugin's own folder once installed,
// or the projectDir option); RUN is per-user state (control files, frames, snapshots).
let ROOT = ''
let RUN = ''
let VIEWS = ''
let ACTIVE = '' // touched while the session is used (the 30-day prune) // the session's frozen views (shared by its processes); RUN: this process's controls and frames
let DENO = '' // absolute path, or '' when none was found
const MAX_FRAME_PX = 4096 // ImageSource limit for shm/file frames
const DRAG_SCALE = 0.55 // resolution while dragging (faster frames); full Retina again on release
const VIEW_ASPECT = 16 / 10 // pixels, width / height
const SURFACES = ['wave', 'torus', 'sphere', 'dini', 'breather', 'teapot', 'wave-test']
const PUBLIC_SURFACES = SURFACES.filter((x) => !x.endsWith('-test'))
const COLORMAPS = ['viridis', 'magma', 'inferno', 'plasma']
const COLOR_BY = ['field', 'height', 'gaussian-curvature', 'mean-curvature']
const FUNCTIONS = ['bumps', 'runge', 'step', 'blocks']
// Colour map groups: `c` cycles within the current one (cyclic maps stay cyclic, for phase)
const CYCLICAL_MAPS = ['Twilight', 'Rainbow', 'Sinebow', 'HSL']
const SEQUENTIAL_MAPS = ['Viridis', 'Inferno', 'Magma', 'Plasma', 'Cividis', 'Turbo', 'Warm', 'Cool', 'Cubehelix', 'Greys', 'Blues', 'YlGnBu', 'YlOrRd']
const DIVERGING_MAPS = ['RdBu', 'BrBG', 'PRGn', 'PiYG', 'PuOr', 'RdGy', 'RdYlBu', 'Spectral']
/** The next colour map after `cur` in its group (case-insensitive), going back with `step = -1`. */
function nextColorMap(cur: string, step: number): string {
  const lc = String(cur ?? '').toLowerCase()
  const group = [CYCLICAL_MAPS, DIVERGING_MAPS, SEQUENTIAL_MAPS].find((g) => g.some((m) => m.toLowerCase() === lc)) ?? SEQUENTIAL_MAPS
  const i = group.findIndex((m) => m.toLowerCase() === lc)
  const next = group[(Math.max(i, 0) + step + group.length) % group.length] ?? cur
  return cur && cur === lc ? next.toLowerCase() : next
}
const BLANK = 'AAAAAA==' // 1x1 transparent RGBA
const HOME = { azimuth: 35, elevation: 25, zoom: 1 }

type Ctl = Record<string, any>
type Interaction = 'orbit' | 'pan-zoom-x' | 'pan-zoom-xy'
/** A kind is one renderer process: the legacy demos ('surface', 'pchip') and one per gallery renderer script. */
type Kind = string
type Entry = { id: string; section: string; title: string; description: string; renderer: string; control: Ctl;
  interaction: Interaction; playback?: { fps: number; slices: number } }
let GALLERY = ''
/** The last live view's size (cells and frame px), so a tool call's first snapshot after a reload is full size. */
const SIZE_FILE = () => `${denoSearch.home}/.cache/terminal-sciviz/view-size.json`
const SECTIONS = ['lines', 'curves', 'complex', 'volumes', 'scenes', 'surfaces']

const KINDS: Record<Kind, { script: string; ctl: Ctl }> = {
  surface: { script: 'surface.ts', ctl: { surface: 'wave', ...HOME, colormap: 'viridis', colorBy: 'field', knotLines: false } },
  pchip: { script: 'plot1d.ts', ctl: { function: 'bumps', n: 32, x0: null, x1: null, ymin: null, ymax: null,
    showTrue: true, showSamples: true, compare: 'none' } },
}
const BASE = { rev: 0, width: 960, height: 600, hover: null, snapshot: null }
for (const k of Object.values(KINDS)) Object.assign(k.ctl, BASE)
/** The gallery's kind for a renderer script (one process per script). */
function galleryKind(script: string): Kind {
  const kind = `g-${script.replace(/\.ts$/, '')}`
  KINDS[kind] ??= { script, ctl: { ...BASE } }
  return kind
}

const dir = (kind: Kind) => `${RUN}/${kind}`
const control = (kind: Kind) => `${dir(kind)}/control.json`
const running: Record<string, boolean> = {}
const frames: Record<string, { gen: number; w: number; h: number; rev: number }> = {}
const writing: Record<string, { busy: boolean; again: boolean }> = {}
const revs: Record<string, number> = {} // per kind: the last rev written (controls may be replaced, revs never go back)
const saved: Record<string, number> = {} // snapshot path -> rev acknowledged by the renderer
const kindOf: Record<string, Kind> = {} // view id -> kind
/** view id -> how it's driven, and its starting control (for reset) */
const viewInfo: Record<string, { interaction: Interaction; home: Ctl; entry?: Entry }> = {}
/** Command opens, by a token printed in the command's output text: binds exactly that output row. */
const opens: Record<string, { kind: Kind; interaction: Interaction; home: Ctl; entry?: Entry; bound?: string }> = {}
let latestOpen = '' // the newest open's token: its row becomes live when it renders
let openSeq = 0
const view = { live: '', liveMinRev: 0, intended: '', cols: 80, rows: 25, pxW: 960, pxH: 600, input: null as null | { cols: number; rows: number; rect: number[]; extra?: number[][] },
  drag: null as null | { x: number; y: number; ctl: Ctl; win: Win | null; zooming: boolean } }

const cycle = (list: string[], cur: string) => list[(list.indexOf(cur) + 1) % list.length]
const viewPng = (id: string) => `${VIEWS}/${id.replace(/[^A-Za-z0-9_-]/g, '_')}.png`
const liveKind = (): Kind | undefined => kindOf[view.live]
const legacyInteraction = (kind: Kind): Interaction => (kind === 'surface' || kind === 'g-scene' || kind === 'g-volume' ? 'orbit' : 'pan-zoom-x')

let manifest: Entry[] | null = null
/** The gallery's entries (gallery.ts manifest), cached; `fresh` re-reads it (new entries since load). */
async function entries($: any, fresh = false): Promise<Entry[]> {
  if (manifest && !fresh) return manifest
  try { manifest = JSON.parse(await $.fs.read(`${ROOT}/deno/gallery/index.json`)) as Entry[] } catch { return [] } // not cached: retry
  return manifest
}

/** Write the kind's control with a new rev. Coalesced: while a write is in flight, later calls
 * only mark it dirty and the latest state is written once it lands. */
async function writeControl($: any, kind: Kind) {
  revs[kind] = Math.max(revs[kind] ?? 0, KINDS[kind].ctl.rev ?? 0) + 1
  KINDS[kind].ctl.rev = revs[kind]
  const w = (writing[kind] ??= { busy: false, again: false })
  if (w.busy) { w.again = true; return }
  w.busy = true
  try {
    // the current control object each time: an open may have replaced it meanwhile
    do { w.again = false; await $.fs.write(control(kind), JSON.stringify(KINDS[kind].ctl)) } while (w.again)
  } finally {
    w.busy = false
  }
}
const nextRev = (kind: Kind) => Math.max(revs[kind] ?? 0, KINDS[kind].ctl.rev ?? 0) + 1

/** Save the live view's frame for the next rev as its PNG (what it freezes on, what Claude reads). */
let touchedAt = 0
async function settle($: any, id = view.live) {
  const kind = kindOf[id]
  if (!kind) return
  if (ACTIVE && Date.now() - touchedAt > 3600e3) { touchedAt = Date.now(); void $.process.run(['/usr/bin/touch', ACTIVE]).catch(() => {}) } // in use: not pruned
  const ctl = KINDS[kind].ctl
  ctl.snapshot = { path: viewPng(id), rev: nextRev(kind) } // the rev writeControl is about to assign
  await writeControl($, kind)
}

/** No status line: the user's screen is space-constrained; readouts belong in the view. */
function status($: any) {
  $.ui.status(undefined)
}

const ready: Record<string, boolean> = {} // kind -> printed `ready` (device and pipelines up)
const codeAt: Record<string, string> = {} // kind -> fingerprint of the renderer code it started with

/** A fingerprint of the renderers' code: a running renderer older than it is restarted. */
async function codeFingerprint($: any): Promise<string> {
  if (!ROOT) return ''
  const r = await $.process.run(['/bin/sh', '-c', 'cat deno/*.ts deno/lib/*.ts 2>/dev/null | cksum'], { cwd: ROOT }).catch(() => null)
  return r?.stdout?.trim() ?? ''
}

/** Restart the kind's renderer if the code changed since it started (e.g. while developing). */
async function refreshRenderer($: any, kind: Kind) {
  if (procs[kind]?.state !== 'running') return
  const now = await codeFingerprint($)
  if (now && codeAt[kind] && now !== codeAt[kind]) {
    await stopRenderer($, kind)
    await startRenderer($, kind) // waits for the old one to exit
  }
}
const lastError: Record<string, string> = {} // kind -> its last error (stderr, controlerror, exit)

/** Renderer processes: one per kind. A stopping one is awaited before the kind starts again. */
const procs: Record<string, { state: 'running' | 'stopping'; done: Promise<void> }> = {}

async function startRenderer($: any, kind: Kind) {
  if (setupError) { lastError[kind] = setupError; return }
  const p = procs[kind]
  if (p?.state === 'running') return
  if (p?.state === 'stopping') {
    // Wait for it to exit; if it's still alive after 3 s its quit was superseded (a newer control
    // replaced it before the coalesced write landed), so it never saw it: reuse it instead.
    const exited = await Promise.race([p.done.then(() => true), $.clock.sleep(3000).then(() => false)])
    if (!exited && procs[kind] === p) {
      p.state = 'running'
      KINDS[kind].ctl.quit = false
      await writeControl($, kind)
      return
    }
  }
  if (procs[kind]?.state === 'running') return // another caller restarted it meanwhile
  let resolveDone!: () => void
  const done = new Promise<void>((r) => (resolveDone = r))
  procs[kind] = { state: 'running', done }
  running[kind] = true
  void codeFingerprint($).then((f) => { codeAt[kind] = f })
  ready[kind] = false
  delete lastError[kind]
  void (async () => {
    try {
      await $.process.run(['/bin/mkdir', '-p', dir(kind), VIEWS])
      KINDS[kind].ctl.quit = false
      await writeControl($, kind)
      const argv = [DENO, 'run', '-A', '--unstable-webgpu', '--unstable-ffi', KINDS[kind].script, 'serve',
        control(kind), dir(kind), '--ignore-stdin']
      let buf = ''
      for await (const { stream, text } of $.process.spawn({ argv, cwd: `${ROOT}/deno` })) {
        if (stream === 'stderr') {
          $.ui.log(`${kind}: ${text}`, { to: 'debug' })
          const ce = /controlerror (\d+) (.+)/.exec(text)
          if (ce) lastError[kind] = ce[2].trim()
          else if (/\berror\b/i.test(text)) lastError[kind] = text.trim().split('\n').slice(-3).join(' ').slice(0, 400)
          continue
        }
        buf += text
        let nl
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl)
          buf = buf.slice(nl + 1)
          if (line.trim() === 'ready') { ready[kind] = true; continue }
          const m = /^frame (\d+) (\d+) (\d+)(?: (\S+))?(?: (\d+))?/.exec(line)
          if (m) { await onFrame($, kind, +m[1], +m[2], +m[3], m[4] === '-' ? undefined : m[4], +(m[5] ?? 0)); continue }
          const snap = /^snapshot(failed)? (\d+) (.+)$/.exec(line)
          if (snap) { onSnapshot(kind, +snap[2], snap[3], !!snap[1]); if (!view.drag && kind === liveKind()) void refreshInputRect($, kind) }
        }
      }
    } catch (err) {
      lastError[kind] = `renderer failed to start: ${err instanceof Error ? err.message : String(err)}`
      $.ui.log(`${kind}: ${lastError[kind]}`, { to: 'debug' })
    } finally {
      running[kind] = false
      if (procs[kind]?.done === done) delete procs[kind]
      resolveDone()
    }
  })()
}

/** Ask a renderer to exit, through the ordered writer (a later control can't be overtaken by it). */
async function stopRenderer($: any, kind: Kind) {
  const p = procs[kind]
  if (p?.state !== 'running') return
  p.state = 'stopping'
  KINDS[kind].ctl.quit = true // stays set until startRenderer clears it: the coalesced writer may land later
  await writeControl($, kind).catch(() => {})
}

const failed: Record<string, number> = {} // snapshot path -> rev the renderer could not save
// ---- terminals that can't show pictures (Terminal.app, iTerm2: it takes byte images only, and only
// fullscreen; plain xterm; tmux without passthrough):
// every Image draws its alt there, so the alt says why and what works; tool results carry a
// `warning`, so the model explains in words rather than pointing at a picture nobody sees ----
const NO_IMAGES = 'sciviz: this terminal can\'t show its images: it needs Ghostty or kitty (Terminal.app and iTerm2 aren\'t supported)'
const TMUX_FIX = 'sciviz: tmux is holding back the images: set -g allow-passthrough on and set -g focus-events on'
const images = { blocked: '', termProgram: '' } // blocked: why a blit was refused (set once seen)
function imageWarning(): string {
  if (images.blocked) return images.blocked
  return images.termProgram === 'Apple_Terminal' || images.termProgram === 'iTerm.app' ? NO_IMAGES : ''
}
const withWarning = (o: Record<string, unknown>) => (imageWarning() ? { ...o, warning: `${imageWarning()}; the user sees no picture: describe it in words` } : o)

function onSnapshot(kind: Kind, rev: number, path: string, isFailure = false) {
  if (isFailure) failed[path] = Math.max(failed[path] ?? 0, rev)
  else saved[path] = Math.max(saved[path] ?? 0, rev)
  const ctl = KINDS[kind].ctl
  if (ctl.snapshot && ctl.snapshot.path === path && rev >= ctl.snapshot.rev) ctl.snapshot = null // done; nothing to rewrite
}

async function onFrame($: any, kind: Kind, gen: number, w: number, h: number, shm: string | undefined, rev: number) {
  frames[kind] = { gen, w, h, rev }
  if (view.live && kindOf[view.live] === kind && rev >= view.liveMinRev) {
    const source = shm
      ? { shm, format: 'rgba', width: w, height: h } // shared memory: no disk, read once by the terminal
      : { file: `${dir(kind)}/frame.rgba`, format: 'rgba', width: w, height: h, generation: gen }
    const r = await $.ui.blit({ requestId: view.live, key: 'view', columns: view.cols, rows: view.rows, source }).catch(() => null)
    if (r?.fix === 'tmux-passthrough') images.blocked = TMUX_FIX
    else if (r?.deny && /alt/i.test(r.deny)) images.blocked = NO_IMAGES // the Image draws its alt: no pictures here
  }
}

// ---- playback of 4D stacks (entry.playback: step the control's `slice` at fps) ----
const play = { token: 0, paused: false }
async function startPlayback($: any, id: string) {
  const pb = viewInfo[id]?.entry?.playback, kind = kindOf[id]
  const token = ++play.token
  if (!pb || !kind || !(pb.fps > 0) || !(pb.slices > 1)) return
  play.paused = false
  while (token === play.token && view.live === id) {
    await $.clock.sleep(1000 / pb.fps)
    if (token !== play.token || view.live !== id) break
    if (play.paused || view.drag) continue
    const ctl = KINDS[kind].ctl
    ctl.slice = ((Number(ctl.slice) || 0) + 1) % pb.slices
    await writeControl($, kind)
  }
}

/** Make `id` the live view: it only takes frames from now on, and other gallery renderers are stopped. */
async function makeLive($: any, id: string) {
  const kind = kindOf[id]
  view.live = id
  view.liveMinRev = nextRev(kind)
  view.input = null // the whole view takes input until its renderer says where the object is
  void startPlayback($, id) // a no-op unless the entry plays (and it stops any other)
  for (const k of Object.keys(KINDS)) if (k !== kind && k.startsWith('g-')) void stopRenderer($, k)
}

/** Cells of an inline view: always the transcript's full width; VIEW_ASPECT tall unless that
 * exceeds the screen, in which case the view gets wider than VIEW_ASPECT instead of narrower
 * (a live view should fit on screen while it's being driven). */
function inlineSize(viewport?: { columns: number; rows: number }) {
  const cols = Math.min(255, Math.max(20, (viewport?.columns ?? 100) - RESULT_INDENT))
  const maxRows = Math.max(8, (viewport?.rows ?? 40) - 8)
  return { cols, rows: Math.min(255, maxRows, Math.round(cols / VIEW_ASPECT / CELL_ASPECT)) }
}

/** The tree of the view `id`: live (frames + input layer) if it's the newest, else its PNG. */
async function drawView($: any, e: any, id: string, kind: Kind) {
  const { Box, Image, Client, Text } = $.ui.resolve(e) as any
  kindOf[id] ??= kind
  viewInfo[id] ??= { interaction: legacyInteraction(kind), home: { ...KINDS[kind].ctl } }
  if (view.intended && id === view.intended) { view.intended = ''; await makeLive($, id); void settle($, id) } // the row just opened
  const { cols, rows } = inlineSize(e.viewport)
  // a terminal known not to show pictures: one line, not a box of empty cells (the renderer still
  // runs, so the model's snapshots and read-outs work)
  if (imageWarning() && !setupError && !lastError[kind]) return <Text dimColor>{imageWarning()}</Text>
  if (id !== view.live) {
    const png = viewPng(id)
    return (await $.fs.exists(png))
      ? <Image source={{ file: png, format: 'png' }} columns={cols} rows={rows} alt={imageWarning() || NO_IMAGES} />
      : <Text dimColor>sciviz view (no frame saved)</Text>
  }
  Object.assign(view, { cols, rows })
  const ctl = KINDS[kind].ctl
  // exactly the box's device pixels, so Ghostty shows the frame 1:1 (scaled down only past MAX_FRAME_PX)
  const fit = Math.min(1, MAX_FRAME_PX / (cols * CELL_PX_W), MAX_FRAME_PX / (rows * CELL_PX_H))
  const pxW = Math.round(cols * CELL_PX_W * fit), pxH = Math.round(rows * CELL_PX_H * fit)
  if (pxW !== view.pxW || pxH !== view.pxH) void $.fs.write(SIZE_FILE(), JSON.stringify({ cols, rows, pxW, pxH })).catch(() => {})
  Object.assign(view, { pxW, pxH })
  if (!view.drag && (pxW !== ctl.width || pxH !== ctl.height)) { Object.assign(ctl, { width: pxW, height: pxH }); if (running[kind]) void writeControl($, kind) }
  void startRenderer($, kind)
  // A shm frame is read once, so a (re)mounted view starts from its saved PNG and asks for a fresh frame
  const png = viewPng(id)
  const havePng = await $.fs.exists(png)
  const problem = setupError || (!havePng && !running[kind] ? lastError[kind] : '')
  if (problem) return <Text color="red">sciviz: {problem}</Text>
  const source = havePng ? { file: png, format: 'png', generation: saved[png] ?? 0 } : { rgba: BLANK, width: 1, height: 1 }
  if (running[kind]) void writeControl($, kind)
  // input only over the object (3D views): presses on the empty frame around it select transcript text
  const inRect = view.input && view.input.cols === cols && view.input.rows === rows ? view.input.rect : [0, 0, cols, rows]
  // a line of the controls the user has likely forgotten; none: no line, no space
  void $.store.get('hints').catch(() => null) // subscribes this draw: a practised() write redraws it
  const hints = imageWarning() ? '' : hintLine(kind, await hintMemory($)) // none over an alt message
  return (
    <Box width={cols} height={rows}>
      <Image key="view" source={source} columns={cols} rows={rows} alt={setupError || lastError[kind] ? `sciviz: ${setupError || lastError[kind]}` : imageWarning() || NO_IMAGES} />
      <Box position="absolute" top={inRect[1]} left={inRect[0]}>
        <Client key="input" module="./input.tsx" width={inRect[2]} height={inRect[3]} />
      </Box>
      {(view.input && view.input.cols === cols && view.input.rows === rows ? view.input.extra ?? [] : []).map((r) => (
        <Box key={`in-${r.join(',')}`} position="absolute" top={r[1]} left={r[0]}>
          <Client key={`input-${r.join(',')}`} module="./input.tsx" props={{ off: [r[0], r[1]] }} width={r[2]} height={r[3]} />
        </Box>
      ))}
      {hints ? ( // in the view's top row, right-aligned (by the title): no row of its own below it
        <Box position="absolute" top={0} left={Math.max(0, cols - hints.length - 1)}>
          <Text dimColor>{hints}</Text>
        </Box>
      ) : null}
    </Box>
  )
}

// ---- 2D windows: pan/zoom in axis space (log axes in log10), mapped through the plot rectangle ----
type Tx = { fwd: (v: number) => number; inv: (w: number) => number }
type Win = { x: [number, number]; y: [number, number] | null; tx: Tx; plot: [number, number, number, number]; w: number; h: number }
/** The axis transform the renderers use (deno/lib/scales.ts axisTransform): pan/zoom happen in its space. */
function axisTx(spec: any): Tx {
  const c = Number(spec?.threshold ?? 1) || 1
  if (spec?.scale === 'log') return { fwd: (v) => Math.log10(v), inv: (w) => 10 ** w }
  if (spec?.scale === 'symlog') return { fwd: (v) => Math.sign(v) * Math.log1p(Math.abs(v) / c), inv: (w) => Math.sign(w) * Math.expm1(Math.abs(w)) * c }
  return { fwd: (v) => v, inv: (w) => w }
}
const yKeys = (kind: Kind) => (KINDS[kind].script === 'complex.ts' ? ['y0', 'y1'] : ['ymin', 'ymax'])

/** The view's current window as the renderer resolved it (state.json), in axis space. */
async function readWindow($: any, kind: Kind): Promise<Win | null> {
  try {
    const st = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`))
    const [ya, yb] = yKeys(kind).map((k) => st[k])
    const tx = axisTx(KINDS[kind].ctl.x)
    if (typeof st.x0 !== 'number' || typeof st.x1 !== 'number') return null
    return {
      x: [tx.fwd(st.x0), tx.fwd(st.x1)],
      y: typeof ya === 'number' && typeof yb === 'number' ? [ya, yb] : null,
      tx,
      plot: st.plotRect ?? [0, 0, st.width, st.height],
      w: st.width,
      h: st.height,
    }
  } catch {
    return null
  }
}
function writeWindow(kind: Kind, win: Win, x: [number, number], y: [number, number] | null) {
  const ctl = KINDS[kind].ctl
  Object.assign(ctl, { x0: win.tx.inv(x[0]), x1: win.tx.inv(x[1]) })
  if (y) { const [a, b] = yKeys(kind); Object.assign(ctl, { [a]: y[0], [b]: y[1] }) }
}
/** Zoom an axis range by f about axis value `at`. */
const zoomRange = ([a, b]: [number, number], at: number, f: number): [number, number] => [at - (at - a) * f, at + (b - at) * f]

// ---- control hints, spaced like a forgetting curve: retention e^(-t/S) since a control's last use;
// a hint shows once it falls below 1/2 (never used: shown), and each use spaced a minute or more
// from the last doubles S (10 min, 20, 40, ... a month). Kept across sessions in $.store ----
type Memory = { last: number; s: number } // ms since the epoch; strength, ms
const HINT_S0 = 10 * 60e3, HINT_SMAX = 30 * 86400e3, HINT_SPACING = 60e3
let hintLoad: Promise<Record<string, Memory>> | null = null // one load, shared: concurrent first uses edit one object
const hintMemory = ($: any): Promise<Record<string, Memory>> =>
  (hintLoad ??= ($.store.get('hints') as Promise<Record<string, Memory> | undefined>).then((m) => m ?? {}, () => ({})))
let hintWrite = false
async function practised($: any, id: string) {
  const m = await hintMemory($), now = Date.now(), cur = m[id] // loaded first: never overwrite saved strengths with {}
  if (cur && now - cur.last < HINT_SPACING) cur.last = now // massed practice: no stronger, but remembered
  else m[id] = cur ? { last: now, s: Math.min(HINT_SMAX, cur.s * 2) } : { last: now, s: HINT_S0 }
  if (hintWrite) return
  hintWrite = true // one write per 2 s at most (it redraws the live view, whose draw read the store)
  await $.clock.sleep(cur ? 2000 : 0)
  hintWrite = false
  await $.store.set('hints', m).catch(() => {})
}
const hintDue = (m: Memory | undefined, now: number) => !m || Math.exp(-(now - m.last) / m.s) < 0.5
const KEY_HINT: Record<string, string> = { c: 'colors', C: 'colors', i: 'invert', r: 'reset', '+': 'zoom', '-': 'zoom', '=': 'zoom', s: 'surface', b: 'colorBy', p: 'play', ' ': 'play' }
/** The due hints of the live view, as one line ('' when none is due). */
function hintLine(kind: Kind, mem: Record<string, Memory>): string {
  const ctl = KINDS[kind]?.ctl ?? {}, info = viewInfo[view.live]
  const orbit = (info?.interaction ?? legacyInteraction(kind)) === 'orbit'
  const maps = ('colormap' in ctl || 'phaseColorMap' in ctl) && kind !== 'pchip'
  const all: [string, string][] = [
    ['drag', orbit ? 'drag orbit' : 'drag pan'], ['zoom', 'shift-drag or +/- zoom'],
    ...(maps ? [['colors', 'c colours'], ['invert', 'i invert']] as [string, string][] : []),
    ...(kind === 'surface' ? [['surface', 's surface'], ['colorBy', 'b colour by']] as [string, string][] : []),
    ...(info?.entry?.playback ? [['play', 'space pause']] as [string, string][] : []),
    ['reset', 'r reset'],
  ]
  const now = Date.now()
  return all.filter(([id]) => hintDue(mem[id], now)).map(([, text]) => text).join(' · ')
}

async function onKey($: any, kind: Kind, k: string) {
  if (KEY_HINT[k]) void practised($, KEY_HINT[k])
  const ctl = KINDS[kind].ctl
  const info = viewInfo[view.live]
  const interaction = info?.interaction ?? legacyInteraction(kind)
  if (k === 'r' && info) { // back to how the view opened
    for (const key of ['azimuth', 'elevation', 'zoom', 'fitAt', 'x0', 'x1', 'y0', 'y1', 'ymin', 'ymax'])
      if (key in info.home) ctl[key] = info.home[key]
      else delete ctl[key] // back to the renderer's default
    return settle($)
  }
  // 4D stacks: p / space pause and resume playback; , and . step a slice
  const pb = info?.entry?.playback
  if (pb && (k === 'p' || k === ' ')) { play.paused = !play.paused; if (play.paused) await settle($); return }
  if (pb && (k === ',' || k === '.')) {
    play.paused = true
    ctl.slice = (((Number(ctl.slice) || 0) + (k === '.' ? 1 : -1)) % pb.slices + pb.slices) % pb.slices
    return settle($)
  }
  // any view with a colour map: c / C cycle it within its group, i inverts it
  const mapKey = 'phaseColorMap' in ctl ? 'phaseColorMap' : 'colormap' in ctl ? 'colormap' : ''
  const invertKey = mapKey === 'phaseColorMap' ? 'invertPhaseColorMap' : 'invertColorMap'
  if (mapKey && (k === 'c' || k === 'C') && kind !== 'pchip') ctl[mapKey] = nextColorMap(ctl[mapKey], k === 'C' ? -1 : 1)
  else if (mapKey && k === 'i') ctl[invertKey] = !ctl[invertKey]
  // n-th-root colour scale (t -> t^(1/n), colours only): n toggles it, [ and ] step n on its grid
  if (mapKey && k === 'n' && kind !== 'pchip') {
    if (ctl.scale === 'root') { ctl.scale = ctl.scaleBeforeRoot ?? 'linear'; delete ctl.scaleBeforeRoot }
    else { ctl.scaleBeforeRoot = ctl.scale ?? 'linear'; ctl.scale = 'root'; ctl.rootN ??= 3 }
    return settle($)
  }
  if (mapKey && (k === '[' || k === ']') && ctl.scale === 'root') {
    ctl.rootN = snapRootN((Number(ctl.rootN) || 3) * (k === ']' ? 1.12 : 1 / 1.12))
    return settle($)
  }
  if (kind === 'surface') {
    if (k === 's') ctl.surface = cycle(SURFACES, ctl.surface)
    else if (k === 'b') ctl.colorBy = cycle(COLOR_BY, ctl.colorBy)
    else if (k === 'k') ctl.knotLines = !ctl.knotLines
  } else if (kind === 'pchip') {
    if (k === 'f') Object.assign(ctl, { function: cycle(FUNCTIONS, ctl.function), x0: null, x1: null })
    else if (k === 'n') ctl.n = ctl.n >= 256 ? 8 : ctl.n * 2
    else if (k === 'c') ctl.compare = ctl.compare === 'cubic' ? 'none' : 'cubic'
    else if (k === 't') ctl.showTrue = !ctl.showTrue
  }
  const zoomIn = k === '+' || k === '=', zoomOut = k === '-' || k === '_'
  if (zoomIn || zoomOut) {
    if (interaction === 'orbit') ctl.zoom = Math.max(0.2, Math.min(30, (ctl.zoom ?? 1) * (zoomIn ? 1.2 : 1 / 1.2)))
    else {
      const win = await readWindow($, kind)
      if (!win) return
      const f = zoomIn ? 1 / 1.5 : 1.5, mid = (r: [number, number]) => (r[0] + r[1]) / 2
      writeWindow(kind, win, zoomRange(win.x, mid(win.x), f),
        interaction === 'pan-zoom-xy' && win.y ? zoomRange(win.y, mid(win.y), f) : null)
    }
  }
  await settle($)
}

/** Camera fields the control may leave to the renderer's defaults, taken from its state.json. */
async function resolveCamera($: any, kind: Kind) {
  const ctl = KINDS[kind].ctl
  if (typeof ctl.azimuth === 'number' && typeof ctl.elevation === 'number' && typeof ctl.zoom === 'number') return
  let st: any = {}
  try { st = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`)) } catch { /* defaults */ }
  const cam = st.camera ?? st
  if (typeof ctl.azimuth !== 'number') ctl.azimuth = typeof cam.azimuth === 'number' ? cam.azimuth : HOME.azimuth
  if (typeof ctl.elevation !== 'number') ctl.elevation = typeof cam.elevation === 'number' ? cam.elevation : HOME.elevation
  if (typeof ctl.zoom !== 'number') ctl.zoom = 1
}

/** `v` on a slider's grid (min + k step), rounded to the step's precision, then within its range;
 * a data slice's slider (`values`: distinct, sorted): the nearest value. */
const snapTo = (v: number, { min, max, step, values }: { min: number; max: number; step: number; values?: number[] }) => {
  if (Array.isArray(values) && values.length) return values.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a))
  const digits = Math.min(100, Math.max(0, Math.ceil(-Math.log10(step)) + 2))
  return Math.min(max, Math.max(min, Number((min + Math.round((v - min) / step) * step).toFixed(digits))))
}
/** The root degree on its slider's grid: 0.1 to 10 in steps of 0.05. */
const snapRootN = (n: number) => snapTo(n, { min: 0.1, max: 10, step: 0.05 })

/** control.sliderHot for a slider of stateSliders: its name, or the root slider's sentinel (overlay.ts ROOT_HOT). */
const hotName = (sl: any) => (sl.kind === 'root' ? 'ⁿ√' : sl.name)

/** The renderer's in-frame sliders as drawn (state.json: `sliders` of parameters, `rootSlider` of
 * the root colour scale, `kind: 'root'`): track rects in frame px, ranges. */
function stateSliders(st: any): any[] {
  const all = [...(Array.isArray(st?.sliders) ? st.sliders.map((sl: any) => ({ ...sl, kind: 'param' })) : []),
    ...(st?.rootSlider ? [{ ...st.rootSlider, kind: 'root' }] : [])]
  return all.filter((sl) => Array.isArray(sl?.rect)).map((sl) => ({ ...sl, w: st.width, h: st.height }))
}

/** After a settled frame of a 3D view: the cells its object covers (state.json `content`, plus the
 * root slider and a cell of slack), so the input region can shrink to them. Redraws when it moved. */
async function refreshInputRect($: any, kind: Kind) {
  let st: any
  try { st = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`)) } catch { return }
  const { cols, rows } = view
  let rect: number[] | null = null
  const extra: number[][] = []
  const content = st?.content ?? st?.plotRect // 3D: the object's box; 2D: the plot area (pan, zoom, hover)
  if (Array.isArray(content) && st.width > 0 && st.height > 0) {
    const boxes = [content, ...stateSliders(st).map((sl) => sl.rect)]
    const [x0, y0, x1, y1] = boxes.reduce((a: number[], b: number[]) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])])
    const cx0 = Math.max(0, Math.floor((x0 / st.width) * cols) - 1), cy0 = Math.max(0, Math.floor((y0 / st.height) * rows) - 1)
    // no slack below: what follows the view on screen can start above its laid-out bottom rows
    // a row clear of the view's bottom too: on screen the view can end above its laid-out rows,
    // and what follows it (text to select) is drawn there
    const cx1 = Math.min(cols, Math.ceil((x1 / st.width) * cols) + 1), cy1 = Math.min(rows - 2, Math.floor((y1 / st.height) * rows))
    if (cx1 > cx0 && cy1 > cy0) rect = [cx0, cy0, cx1 - cx0, cy1 - cy0]
    // a slider in those bottom rows (the volume's root scale under its colour bar): an input strip
    // of its own, just its columns, so the rest of the rows stays selectable
    if (rect) for (const sl of stateSliders(st)) {
      const sy1 = Math.min(rows, Math.ceil((sl.rect[3] / st.height) * rows))
      if (sy1 <= cy1) continue
      const sx0 = Math.max(0, Math.floor((sl.rect[0] / st.width) * cols) - 1), sx1 = Math.min(cols, Math.ceil((sl.rect[2] / st.width) * cols) + 1)
      extra.push([sx0, cy1, sx1 - sx0, sy1 - cy1])
    }
  }
  const key = (i: typeof view.input) => JSON.stringify(i ? [i.rect, i.extra] : null)
  const was = key(view.input)
  view.input = rect ? { cols, rows, rect, extra } : null
  if (key(view.input) !== was) $.ui.invalidate('ui.render')
}

let overlaySeq = 0
const overlayOut = { busy: false, next: '' }
/** Latest-only, serialized: an older position never lands after a newer one. */
async function writeOverlay($: any, text: string) {
  overlayOut.next = text
  if (overlayOut.busy) return
  overlayOut.busy = true
  try {
    while (overlayOut.next) { const t = overlayOut.next; overlayOut.next = ''; await $.fs.write(`${RUN}/overlay.json`, t) }
  } catch { /* overlay is optional */ } finally {
    overlayOut.busy = false
  }
}
async function onPointer($: any, kind: Kind, ev: any) {
  const ctl = KINDS[kind].ctl
  const interaction = viewInfo[view.live]?.interaction ?? legacyInteraction(kind)
  // the image's own size in cells: the input region can be clipped by the screen edge, the image's coordinates are not
  const cols = view.cols || ev.columns, rows = view.rows || ev.rows
  // region-relative -> view cells: the input region may be only the object's part of the view
  const off = Array.isArray(ev.off) ? ev.off : view.input && view.input.cols === cols && view.input.rows === rows ? view.input.rect : [0, 0]
  const x = (ev.fine?.x ?? ev.x + 0.5) + off[0], y = (ev.fine?.y ?? ev.y + 0.5) + off[1]
  // a move with no button held while a drag is open: its `up` was lost (outside the view, a remount).
  // End the drag, or every hover would keep orbiting the camera and re-rendering.
  if (view.drag && ev.type === 'move' && !ev.button) ev = { ...ev, type: 'up' }
  // a held drag (slider or orbit) owns the pointer until `up`: crossing the region's edge doesn't end it
  if (view.drag && ev.type === 'leave') return
  // For the native overlay (overlay/*.ts): where the pointer is in the view, in cells; the
  // overlay pairs it with the global mouse position to place itself exactly over the view.
  void writeOverlay($, JSON.stringify({ seq: ++overlaySeq, type: ev.type, x, y, cols, rows,
    fine: !!ev.fine, drag: ev.type === 'down' || (!!view.drag && ev.type !== 'up'), kind, id: view.live }))
  const sliderAt = async () => {
    // sliders: given, the root scale's, a mark's slice column's or a file volume's slice (added by the renderer)
    const sliced = ctl.volume?.dataset === 'file' || (Array.isArray(ctl.marks) && ctl.marks.some((m: any) => m?.slice !== undefined))
    if (!ctl.sliders && !sliced && (Array.isArray(ctl.scale) ? ctl.scale[0] : ctl.scale) !== 'root') return null
    let st: any = null
    try { st = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`)) } catch { return null }
    // hit test in cells, at least a whole cell beyond the track on every side: the slider is under
    // a cell tall on screen and terminal pointer positions are coarse; the nearest track wins
    let best: any = null
    for (const sl of stateSliders(st)) {
      const [x0, y0, x1, y1] = sl.rect
      const [tx0, tx1, ty0, ty1] = [(x0 / sl.w) * cols, (x1 / sl.w) * cols, (y0 / sl.h) * rows, (y1 / sl.h) * rows]
      if (x < tx0 - 1 || x > tx1 + 1 || y < ty0 - 1 || y > ty1 + 1) continue
      const d = Math.abs(y - (ty0 + ty1) / 2) + Math.max(0, tx0 - x, x - tx1) // in cells, from the track
      if (!best || d < best.d) best = { sl, d, px: Math.min(x1, Math.max(x0, (x / cols) * sl.w)) }
    }
    return best
  }
  const setFromSlider = (sl: any, px: number) => {
    const [x0, , x1] = sl.rect
    const v = sl.min + ((px - x0) / Math.max(1, x1 - x0)) * (sl.max - sl.min)
    ctl.sliderHot = hotName(sl)
    if (sl.kind === 'root') ctl.rootN = snapTo(v, sl)
    else ctl.sliderValues = { ...ctl.sliderValues, [sl.name]: snapTo(v, sl) }
  }
  if (ev.type === 'down') {
    const hit = await sliderAt()
    if (hit) {
      view.drag = { x, y, ctl: { ...ctl }, win: null, zooming: false, slider: hit.sl } as any
      ctl.interacting = true // curve.ts holds its framing while a slider moves; volume.ts renders cheaper frames, isosurfaces on release
      setFromSlider(hit.sl, hit.px)
      await writeControl($, kind)
      return
    }
  }
  if (ev.type === 'move' && (view.drag as any)?.slider) {
    const sl = (view.drag as any).slider
    setFromSlider(sl, Math.min(sl.rect[2], Math.max(sl.rect[0], (x / cols) * sl.w)))
    await writeControl($, kind)
    return
  }
  if (ev.type === 'down') {
    const win = interaction === 'orbit' ? null : await readWindow($, kind)
    if (interaction === 'orbit') {
      await resolveCamera($, kind)
      // orbit about a fixed pivot: keep the framing of the angle the first drag started from
      // (re-fitting at every angle slides the pivot and pumps the distance); r or a tool call clears it
      ctl.fitAt ??= [ctl.azimuth, ctl.elevation]
    }
    view.drag = { x, y, ctl: { ...ctl }, win, zooming: !!ev.shift }
    void practised($, ev.shift ? 'zoom' : 'drag')
    ctl.hover = null
    // lower resolution while orbiting (3D frames cost most); 2D keeps its layout so pan/zoom stay anchored
    // while orbiting: fewer pixels, and renderers that support it (volume.ts) cut their ray-march samples
    if (interaction === 'orbit') Object.assign(ctl, { width: Math.round(view.pxW * DRAG_SCALE), height: Math.round(view.pxH * DRAG_SCALE), interacting: true })
  } else if (ev.type === 'up') {
    const slid = !!(view.drag as any)?.slider
    view.drag = null
    if (interaction === 'orbit') Object.assign(ctl, { width: view.pxW, height: view.pxH, interacting: false })
    if (slid) ctl.interacting = false
    await settle($)
    return
  } else if (ev.type === 'move' && view.drag) {
    const d = view.drag
    if (interaction === 'orbit') {
      if (d.zooming) ctl.zoom = Math.max(0.2, Math.min(30, (d.ctl.zoom ?? 1) * Math.pow(2, -(y - d.y) / (rows / 2))))
      else {
        ctl.azimuth = d.ctl.azimuth - (x - d.x) * (360 / cols)
        ctl.elevation = Math.max(-89, Math.min(89, d.ctl.elevation + (y - d.y) * (180 / rows)))
      }
    } else if (d.win) {
      const win = d.win
      // cells -> fraction of the plot area (the frame also holds axes, title, legend)
      const plotW = (win.plot[2] - win.plot[0]) / win.w, plotH = (win.plot[3] - win.plot[1]) / win.h
      const fx = (x - d.x) / cols / plotW, fy = (y - d.y) / rows / plotH
      const at = (r: [number, number], frac: number) => r[0] + frac * (r[1] - r[0])
      const gx = ((d.x / cols) * win.w - win.plot[0]) / (win.plot[2] - win.plot[0]) // grab point in the plot, 0..1
      const gy = 1 - ((d.y / rows) * win.h - win.plot[1]) / (win.plot[3] - win.plot[1])
      const xy = interaction === 'pan-zoom-xy' && win.y
      if (d.zooming) { // about the point grabbed; drag down = zoom out
        const f = Math.pow(2, (y - d.y) / (rows / 2))
        writeWindow(kind, win, zoomRange(win.x, at(win.x, gx), f), xy ? zoomRange(win.y!, at(win.y!, gy), f) : null)
      } else {
        const dx = fx * (win.x[1] - win.x[0])
        const dy = xy ? fy * (win.y![1] - win.y![0]) : 0
        writeWindow(kind, win, [win.x[0] - dx, win.x[1] - dx], xy ? [win.y![0] + dy, win.y![1] + dy] : null)
      }
    }
  } else if (ev.type === 'move') {
    const hit = await sliderAt() // over a slider: its knob grows, no value readout
    const hot = hit ? hotName(hit.sl) : null
    const hover = hit ? null : [Math.floor((x / cols) * ctl.width), Math.floor((y / rows) * ctl.height)]
    if (hit && hot === ctl.sliderHot && ctl.hover === null) return // nothing changed: no frame
    Object.assign(ctl, { sliderHot: hot, hover })
  } else if (ev.type === 'leave') {
    Object.assign(ctl, { sliderHot: null, hover: null })
  }
  await writeControl($, kind)
}

/** A tool call that shows a live view: apply `input` to the kind's control, make this call's
 * result row the live view, wait for its frame and answer what Claude needs to see it. */
const openLocks: Record<string, Promise<unknown>> = {}
/** One open per kind at a time, each through its snapshot acknowledgement. */
function showView($: any, e: any, kind: Kind, keys: string[], replace?: Ctl, entry?: Entry) {
  const run = (openLocks[kind] ?? Promise.resolve()).then(() => showViewNow($, e, kind, keys, replace, entry))
  openLocks[kind] = run.catch(() => {})
  return run
}
async function showViewNow($: any, e: any, kind: Kind, keys: string[], replace?: Ctl, entry?: Entry) {
  const input = e as Ctl
  await checkDeno($)
  if (setupError) return { result: JSON.stringify({ snapshot: null, error: setupError }) }
  if (replace) KINDS[kind].ctl = { ...replace, ...BASE, rev: KINDS[kind].ctl.rev, width: view.pxW, height: view.pxH }
  const ctl = KINDS[kind].ctl
  Object.assign(ctl, { width: view.pxW, height: view.pxH }) // the size the row will show it at, not a default
  for (const k of keys) if (input[k] !== undefined) ctl[k] = input[k]
  if (['azimuth', 'elevation', 'zoom', 'surface'].some((k) => keys.includes(k) && input[k] !== undefined)) delete ctl.fitAt // a new view: fit it afresh
  const id = String(e.tool_use_id)
  kindOf[id] = kind
  viewInfo[id] = { interaction: entry?.interaction ?? legacyInteraction(kind), home: { ...ctl }, entry }
  latestOpen = ''; view.intended = '' // supersedes any command open whose row hasn't rendered yet
  await makeLive($, id) // this call's result row becomes the live view (the previous one keeps its last saved PNG)
  await refreshRenderer($, kind)
  await startRenderer($, kind)
  await settle($, id) // render with these parameters and save that very frame as the view's PNG
  const png = viewPng(id), want = ctl.snapshot?.rev ?? ctl.rev
  // a cold renderer compiles shaders (and generates data) first: allow 60 s until it prints `ready`
  const t0 = Date.now()
  const deadline = () => (ready[kind] ? 10_000 : 60_000)
  while ((saved[png] ?? -1) < want && (failed[png] ?? -1) < want && Date.now() - t0 < deadline() && !setupError && running[kind]) await $.clock.sleep(50)
  const done = (saved[png] ?? -1) >= want
  const problem = setupError || (done ? '' : (failed[png] ?? -1) >= want ? `snapshot could not be saved` : lastError[kind] || (running[kind] ? 'timed out waiting for the frame' : 'renderer exited'))
  let state: unknown = null
  try { state = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`)) } catch { /* none yet */ }
  const { snapshot: _, hover: __, rev: ___, ...shown } = ctl
  // data arrays (series samples, control points) would flood the model's context: summarise them
  const brief = JSON.parse(JSON.stringify(shown, (_k, v) => (Array.isArray(v) && v.length > 16 ? `[${v.length} values]` : v)))
  const briefState = state && JSON.parse(JSON.stringify(state, (_k, v) => (Array.isArray(v) && v.length > 16 ? `[${v.length} values]` : v)))
  return { result: JSON.stringify(withWarning({ snapshot: done ? png : null, ...(problem ? { error: problem } : {}), entry: entry?.id, control: brief, state: briefState })) }
}

/** /sciviz <surface>: the output row about to be drawn becomes the live view. */
async function openView($: any, e: any, kind: Kind, field: string, options: string[]) {
  const arg = String(e.args ?? '').trim()
  await checkDeno($)
  if (setupError) return { text: `sciviz: ${setupError}` }
  if (options.includes(arg)) KINDS[kind].ctl[field] = arg
  delete KINDS[kind].ctl.fitAt
  const tok = `v${++openSeq}`
  opens[tok] = { kind, interaction: legacyInteraction(kind), home: { ...KINDS[kind].ctl } }
  latestOpen = tok
  await refreshRenderer($, kind)
  await startRenderer($, kind)
  await writeControl($, kind)
  status($)
  return { text: `sciviz ${kind} #${tok}` }
}

/** Gallery entries matching what someone typed: the exact id, else ids ending in it, else ids
 * whose name contains the typed name (within the typed section, if one was given). */
function findEntries(all: Entry[], typed: string): Entry[] {
  const t = typed.trim().toLowerCase()
  if (!t) return []
  const exact = all.filter((x) => x.id.toLowerCase() === t)
  if (exact.length) return exact
  const suffix = all.filter((x) => x.id.toLowerCase().endsWith(`/${t}`))
  if (suffix.length) return suffix
  const [section, name] = t.includes('/') ? t.split('/', 2) : ['', t]
  return all.filter((x) => (!section || x.section === section) && x.id.split('/')[1].toLowerCase().includes(name))
}

/** /sciviz <entry id>: the gallery entry as a live view in the output row. */
async function openEntry($: any, e: any) {
  const id = String(e.args ?? '').trim()
  if (SECTIONS.includes(id)) // a section on its own: list it
    return { text: `${id}: ${(await entries($)).filter((x) => x.section === id).map((x) => x.id.split('/')[1]).join('  ')}` }
  let found = findEntries(await entries($), id)
  if (!found.length) found = findEntries(await entries($, true), id) // maybe added since the mod loaded
  if (found.length !== 1) {
    return { text: found.length
      ? `sciviz: "${id}" matches ${found.map((x) => x.id).join(', ')}; which one?`
      : `sciviz: no gallery entry "${id}". /sciviz [${SECTIONS.join('|')}] shows them.` }
  }
  await checkDeno($)
  if (setupError) return { text: `sciviz: ${setupError}` }
  const entry = found[0]
  const kind = galleryKind(entry.renderer)
  KINDS[kind].ctl = { ...entry.control, ...BASE, rev: KINDS[kind].ctl.rev, width: view.pxW, height: view.pxH }
  const tok = `v${++openSeq}`
  opens[tok] = { kind, interaction: entry.interaction, home: { ...KINDS[kind].ctl }, entry }
  latestOpen = tok
  await refreshRenderer($, kind)
  await startRenderer($, kind)
  await writeControl($, kind)
  status($)
  return { text: `sciviz ${entry.id} #${tok}` }
}

/** Where the renderers, the deno binary and the state live; tried in order, first that exists. */
let setupError = '' // shown in views and tool results when the renderers can't run
const NO_DENO = 'Deno is not installed: install Deno 2.x (`brew install deno`, or see https://deno.com) or set the plugin\'s deno option'

const denoSearch = { home: '', wanted: '' }
let installing: Promise<void> | null = null

/** The deno binary: the option, else PATH, else the usual install locations; '' when none exists. */
async function findDeno($: any) {
  const { home, wanted } = denoSearch
  if (wanted) return wanted.startsWith('/') && (await $.fs.exists(wanted)) ? wanted : ''
  const onPath = (await $.process.run(['/bin/sh', '-c', 'command -v deno || true'])).stdout.trim()
  for (const d of [onPath, `${home}/.deno/bin/deno`, '/opt/homebrew/bin/deno', '/usr/local/bin/deno', '/usr/bin/deno'])
    if (d.startsWith('/') && (await $.fs.exists(d))) return d
  return ''
}

/** Before each view: is deno (still) there? Find it again if it moved; else install it or say why not. */
async function checkDeno($: any) {
  if (installing || (DENO.startsWith('/') && (await $.fs.exists(DENO)))) return
  DENO = await findDeno($)
  if (DENO) { if (ROOT) setupError = ''; return } // with ROOT found, the only setup error is deno's
  setupError = denoSearch.wanted ? `deno not found at ${denoSearch.wanted} (the plugin's deno option)` : NO_DENO
  if (!denoSearch.wanted) installing = installDeno($, denoSearch.home).finally(() => { installing = null })
}

/** No deno anywhere: install it with Homebrew when that is there (in the background; views say so meanwhile). */
async function installDeno($: any, home: string) {
  const brews = ['/opt/homebrew/bin/brew', '/usr/local/bin/brew', '/home/linuxbrew/.linuxbrew/bin/brew', `${home}/.linuxbrew/bin/brew`]
  let brew = ''
  for (const b of brews) if (await $.fs.exists(b)) { brew = b; break }
  if (!brew) return
  setupError = 'installing Deno with Homebrew (`brew install deno`)…'
  $.ui.toast('sciviz: Deno not found, installing it with Homebrew…')
  try {
    const r = await $.process.run([brew, 'install', 'deno'], { env: { NONINTERACTIVE: '1', HOMEBREW_NO_AUTO_UPDATE: '1' }, timeoutMs: 600_000 })
    const deno = brew.replace(/brew$/, 'deno')
    if (r.exitCode === 0 && (await $.fs.exists(deno))) {
      DENO = deno
      setupError = ''
      $.ui.toast('sciviz: Deno installed; views are ready')
      return
    }
    setupError = `\`brew install deno\` failed (exit ${r.exitCode}): ${String(r.stderr ?? '').trim().split('\n').slice(-2).join(' ')}. ${NO_DENO}`
  } catch (err) {
    setupError = `\`brew install deno\` failed (${(err as Error)?.message ?? err}). ${NO_DENO}`
  }
  $.ui.toast(`sciviz: ${setupError}`)
}

async function resolvePaths($: any, options: Record<string, unknown>) {
  const sh = async (cmd: string) => (await $.process.run(['/bin/sh', '-c', cmd])).stdout.trim()
  const home = await sh('printf %s "$HOME"')
  const expand = (p: string) => (p === '~' || p.startsWith('~/') ? home + p.slice(1) : p)
  // the plugin's folder (its root, or the folder above .claude-plugin/), the option, or a checkout in ~
  const root = String($.plugin.root ?? '')
  const roots = [String(options.projectDir ?? ''), root, root.replace(/\/\.claude-plugin\/?$/, ''), `${root}/..`,
    `${home}/terminal-sciviz`].filter(Boolean).map(expand)
  ROOT = ''
  for (const r of roots) if (await $.fs.exists(`${r}/deno/surface.ts`)) { ROOT = r; break }
  denoSearch.home = home
  denoSearch.wanted = expand(String(options.deno ?? ''))
  setupError = !ROOT ? `renderers not found (looked in ${roots.join(', ')}); set the plugin's projectDir option` : ''
  if (ROOT) await checkDeno($)
  if (setupError) $.ui.log(`sciviz: ${setupError}`, { to: 'debug' })
  // per-session state: two Claude sessions never share controls or frames. Named by the session's
  // id, so a reload or a resume finds the frozen views' PNGs it saved before
  let session = ''
  try { const id: unknown = await $.session.id(); if (typeof id === 'string') session = id.replace(/[^A-Za-z0-9_-]/g, '') } catch { /* older host */ }
  const proc = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  // the views' PNGs (named by tool call id) are the session's; controls and frames this process's
  // alone (two terminals resuming one conversation must not share them)
  const sessions = `${home}/.cache/terminal-sciviz/sessions`
  VIEWS = `${sessions}/${session || proc}/views`
  RUN = `${sessions}/${session || proc}/run-${proc}`
  // sessions inactive for 30 days go, frozen views with them: `.active` is touched at start and
  // as views are saved (a directory's own mtime doesn't change as files inside it are rewritten)
  ACTIVE = `${sessions}/${session || proc}/.active`
  void $.process.run(['/bin/sh', '-c', 'mkdir -p "$1" && touch "$2" && for d in "$3"/*/; do a="$d.active"; [ -e "$a" ] || a="$d"; [ -n "$(find "$a" -prune -mtime +30)" ] && rm -rf "$d"; done; true',
    'sh', `${sessions}/${session || proc}`, ACTIVE, sessions]).catch(() => {})
  GALLERY = `${ROOT}/docs`
  try { images.termProgram = ((await $.process.run(['/usr/bin/printenv', 'TERM_PROGRAM']))?.stdout ?? '').trim() } catch { /* unknown */ }
  try { Object.assign(view, JSON.parse(await $.fs.read(SIZE_FILE()))) } catch { /* first run: defaults until a view draws */ }
}

export const register: Register = (on, options) => {
  // one command: /sciviz shows the overview, /sciviz <section> a contact sheet, /sciviz <entry> or <surface> a live view
  on('command.run', { command: 'sciviz' }, async ($, e) => {
    const arg = String((e as any).args ?? '').trim()
    if (SURFACES.includes(arg)) return openView($, e, 'surface', 'surface', SURFACES)
    if (arg && !SECTIONS.includes(arg)) return openEntry($, e)
    if (!arg) return { text: SECTIONS.map((x) => `/sciviz ${x}`).join('  ') }
    const ids = arg === 'surfaces' ? PUBLIC_SURFACES : (await entries($)).filter((x) => x.section === arg).map((x) => x.id)
    return { text: `${arg}: ${ids.join('  ')}` }
  })

  on('ui.render', { component: 'CommandOutput' }, async ($, e, next) => {
    const command = (e.props as any).command
    if (e.surface !== 'terminal') return next(e)
    if (command === 'sciviz' && ['', ...SECTIONS].includes(String((e.props as any).args ?? '').trim())) { // the overview or section's contact sheet, then its commands or ids
      const { Box, Image, Text } = $.ui.resolve(e) as any
      const section = String((e.props as any).args ?? '').trim() || 'overview'
      const png = `${ROOT}/docs/contact_${section}.png`
      if (!(await $.fs.exists(png))) return next(e)
      const cols = Math.max(20, (e.viewport?.columns ?? 100) - RESULT_INDENT)
      const { base64 } = await $.fs.read(png, { as: 'bytes' })
      const hd = new DataView(decodeBase64(base64.slice(0, 32)).buffer)
      const rows = Math.min(255, Math.round((cols * hd.getUint32(20)) / hd.getUint32(16) / CELL_ASPECT))
      if (imageWarning()) return <Text dimColor>{imageWarning()}  ·  {String((e.props as any).text ?? '')}</Text>
      return (
        <Box flexDirection="column">
          <Image source={{ file: png, format: 'png' }} columns={cols} rows={rows} alt={imageWarning() || NO_IMAGES} />
          <Text dimColor>{String((e.props as any).text ?? '')}  ·  /sciviz {'<id>'}</Text>
        </Box>
      )
    }
    if (command !== 'sciviz') return next(e)
    const id = String(e.requestId)
    const tok = /#(v\d+)\b/.exec(String((e.props as any).text ?? ''))?.[1]
    if (!tok && !kindOf[id]) return next(e) // a message (setup error, "which one?"), not a view: show its text
    const open = tok ? opens[tok] : undefined
    if (open && !open.bound && !kindOf[id]) { // the row this very command produced: bind it once
      open.bound = id
      kindOf[id] = open.kind
      viewInfo[id] = { interaction: open.interaction, home: open.home, entry: open.entry }
      if (tok === latestOpen) { latestOpen = ''; view.intended = id }
    }
    const kind = kindOf[id]
    if (!kind) return next(e)
    return drawView($, e, id, kind)
  })

  on('ui.message', async ($, e, next) => {
    const kind = liveKind()
    const data = e.data as any
    if (e.requestId !== view.live || !kind) return next(e)
    for (const ev of data?.type === 'batch' && Array.isArray(data.events) ? data.events : [data]) {
      if (ev?.type === 'key') await onKey($, kind, String(ev.key))
      else if (ev?.type) await onPointer($, kind, ev)
    }
    return {}
  })

  on('tool.call', { tool: VIEW_TOOL }, async ($, e) => {
    const input = e as any
    if (input.nurbs !== undefined) { // a surface of Claude's own: it replaces the named one
      const size = JSON.stringify(input.nurbs).length
      if (size > 64_000) return { deny: `nurbs is ${size} bytes; keep it under 64 kB (exact low-degree patches, or a coarse control net)` }
      if (Array.isArray(input.nurbs) && (input.nurbs.length === 0 || input.nurbs.length > 256)) return { deny: 'nurbs: give 1 to 256 patches' }
    }
    const e2 = input.nurbs !== undefined ? { ...input, surface: input.nurbs } : { ...input }
    // a named surface (or a new nurbs) starts with the renderer's own colour range unless one is given
    if (input.surface !== undefined || input.nurbs !== undefined) e2.domain ??= null
    return showView($, e2, 'surface', ['surface', 'colorBy', 'colormap', 'invertColorMap', 'domain', 'azimuth', 'elevation', 'zoom', 'knotLines'])
  })

  on('tool.call', { tool: OPEN_TOOL }, async ($, e) => {
    let all = await entries($)
    const want = String((e as any).id ?? '')
    let found = findEntries(all, want)
    if (!found.length) found = findEntries((all = await entries($, true)), want) // maybe added since the mod loaded
    if (found.length !== 1) return { deny: `"${want}" matches ${found.length ? found.map((x) => x.id).join(', ') : 'no entry'}; ids: ${all.map((x) => x.id).join(', ')}` }
    const entry = found[0]
    const overrides = ((e as any).control ?? {}) as Ctl
    const size = JSON.stringify(overrides).length
    if (size > 8000) return { deny: `control overrides are ${size} bytes; keep them small (parameters, not data: entries generate their data on the GPU)` }
    return showView($, { ...e, ...overrides }, galleryKind(entry.renderer), Object.keys(overrides), entry.control, entry)
  })

  on('tool.call', { tool: CHART_TOOL }, async ($, e) => {
    const input = e as any
    // data goes in files past 64 kB (the control is rewritten on every pointer event); line.ts reads them once
    const size = JSON.stringify([input.series ?? null, input.marks ?? null, input.overlays ?? null]).length
    if (size > 64_000) return { deny: `series/marks/overlays are ${size} bytes inline; keep them under 64 kB: write the data to a .json/.csv/.tsv file and give each series or mark \`file\` (absolute path) instead of inline data` }
    for (const s of [...(Array.isArray(input.series) ? input.series : []), ...(Array.isArray(input.marks) ? input.marks : [])]) {
      const f = s?.file
      if (f === undefined) continue
      if (typeof f !== 'string' || !f.startsWith('/') || !/\.(json|csv|tsv)$/i.test(f)) return { deny: `file must be an absolute path to a .json, .csv or .tsv file: ${f}` }
      if (!(await $.fs.exists(f))) return { deny: `no such file: ${f}` }
    }
    // a new `series` or `marks` starts a new chart (fields not given: the renderer's defaults, but no
    // demo series); without either, the fields given change the current line chart (this tool's or a
    // gallery lines/* entry's) and the rest stay
    const kind = galleryKind('line.ts')
    const fresh = input.series !== undefined || input.marks !== undefined
    if (!fresh && !Array.isArray(KINDS[kind]?.ctl.series)) return { deny: 'no chart open yet: pass `marks` (or `series`)' }
    return showView($, input, kind, CHART_KEYS, fresh ? { title: '', series: [] } : undefined)
  })

  on('tool.call', { tool: SCENE_TOOL }, async ($, e) => {
    const input = e as any
    const size = JSON.stringify(input.marks ?? null).length // as chart: past 64 kB, data goes in files
    if (size > 64_000) return { deny: `marks are ${size} bytes inline; keep them under 64 kB: write the data to a .json/.csv/.tsv file and give each mark \`file\` (absolute path) instead of inline data (nurbs: exact low-degree patches, or a coarse control net)` }
    for (const m of Array.isArray(input.marks) ? input.marks : []) {
      const f = m?.file
      if (f === undefined) continue
      const ext = m.type === 'volume' ? ['npy', 'f32', 'bin'] : ['json', 'csv', 'tsv']
      if (typeof f !== 'string' || !f.startsWith('/') || !ext.includes(f.split('.').pop()!.toLowerCase())) return { deny: `file must be an absolute path to a .${ext[0]}, .${ext[1]} or .${ext[2]} file: ${f}` }
      if (!(await $.fs.exists(f))) return { deny: `no such file: ${f}` }
    }
    // new `marks`: a new scene (fields not given: defaults); without, the fields given change the current one
    const kind = galleryKind('scene.ts')
    if (input.marks === undefined && !Array.isArray(KINDS[kind]?.ctl.marks)) return { deny: 'no scene open yet: pass `marks`' }
    return showView($, input, kind, SCENE_KEYS, input.marks !== undefined ? { title: '' } : undefined)
  })

  on('tool.call', { tool: VOLUME_TOOL }, async ($, e) => {
    const input = e as any
    const v = input.volume
    if (v !== undefined) {
      if (!v || typeof v !== 'object' || typeof v.dataset !== 'string') return { deny: 'volume: {dataset: "file", path, shape?, dtype?, axes?} or a generator {dataset: "hydrogen" | "hydrogen_spherical" | "breathing" | "tight_binding" | "fermi", ...}' }
      if (v.dataset === 'file') {
        if (typeof v.path !== 'string' || !v.path.startsWith('/')) return { deny: `volume.path must be an absolute path (.npy, or raw .f32/.bin with shape): ${v.path}` }
        if (!(await $.fs.exists(v.path))) return { deny: `no such file: ${v.path}` }
      }
    }
    // a new `volume`: a new view (fields not given: the renderer's defaults); without, the fields given change the current one
    const kind = galleryKind('volume.ts')
    if (v === undefined && !KINDS[kind]?.ctl.volume) return { deny: 'no volume open yet: pass `volume`' }
    const sv = KINDS[kind]?.ctl.sliderValues
    if (v === undefined && input.slice !== undefined && sv) delete sv.slice // a slice given: over the slider's
    return showView($, input, kind, VOLUME_KEYS, v !== undefined ? {} : undefined)
  })

  on('tool.call', { tool: PLOT_TOOL }, async ($, e) =>
    showView($, e, 'pchip', ['function', 'n', 'x0', 'x1', 'ymin', 'ymax', 'showTrue', 'showSamples', 'compare']))

  on('session.end', async ($, e, next) => {
    for (const k of Object.keys(KINDS)) await stopRenderer($, k).catch(() => {})
    // controls, frames, logs go; views/ stays: a resumed conversation still draws its frozen views
    if (RUN.includes('/.cache/terminal-sciviz/sessions/')) await $.process.run(['/bin/rm', '-rf', RUN]).catch(() => {}) // VIEWS stays: a resume redraws them
    return next(e)
  })

  on('session.start', async ($, e, next) => {
    await resolvePaths($, options ?? {})
    await $.command.register({ name: 'sciviz', description: `Inline WebGPU views: /sciviz (overview), /sciviz ${SECTIONS.join('|')} (contact sheet), /sciviz <entry> (live view, e.g. lorenz, chirp, blocks), /sciviz ${PUBLIC_SURFACES.join('|')} (NURBS surface)` })
    const ids = (await entries($)).map((x) => x.id)
    await $.tool.register({
      name: 'gallery_open',
      description:
        'Show the user a gallery entry (lines, curves, complex plots, volumes; rendered with WebGPU) ' +
        'as a live, interactive view inline in the chat, right here between your text. 2D views pan/zoom (drag, shift-drag, +/-), 3D views orbit; ' +
        'hover shows values; r resets. Optional `control` overrides fields of the entry\'s control (e.g. x0/x1 window, azimuth). ' +
        'Returns the renderer state and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          id: ids.length ? { type: 'string', enum: ids } : { type: 'string' },
          control: { type: 'object', description: 'overrides of the entry control; sliders: {name: [min, max]} adds in-frame sliders for parameters (e.g. curves/lorenz: {sliders: {rho: [0, 50]}})' },
        },
        required: ['id'],
      },
      isDeferred: false,
    })
    const axis = {
      type: 'object',
      properties: {
        scale: { type: 'string', enum: ['linear', 'log', 'symlog'] },
        threshold: { type: 'number', description: 'symlog: linear within ±threshold (default 1)' },
        ticks: { type: 'string', enum: ['pi', 'exp2', 'exp10'], description: 'pi: at multiples of π; exp2/exp10: the values are log2/log10 of the data, ticks read 2^v / 10^v' },
        index: { type: 'boolean', description: 'integer ticks (default: when no series gives x)' },
        categories: { type: 'array', items: { type: 'string' }, description: 'ticks at 0, 1, ... labelled with these (box groups, grouped bars)' },
      },
    }
    await $.tool.register({
      name: 'chart',
      description:
        'Show the user a live, interactive chart of your own data inline in the chat, right here between your text ' +
        '(WebGPU; they drag to pan x, shift-drag/+/- to zoom, hover for values, r resets). The newest view is live; earlier ones freeze. ' +
        'Build charts from `marks` (a grammar like Observable Plot): each mark is a type (line, dot, bar, barX, area, rect, rule, text, ' +
        'interval, box) over `data` columns, its channels naming columns, and optional transforms; bubble, stacked area, histogram, ' +
        'frequency polygon, rolling mean, violin, grouped/stacked/horizontal bars and heatmaps are combinations. Examples: ' +
        'bubble {type:"dot", data, x:"gdp", y:"life", r:"pop", fill:"continent"}; frequency polygon {type:"line", data:{v:[...]}, x:"v", transform:{bin:{width:0.5}}}; ' +
        'violin {type:"area", data, x:"group", y:"value", fill:"group", transform:{density:true}} plus {type:"box", data, x:"group", y:"value"}; ' +
        'stacked bars {type:"bar", data, x:"year", y:"sales", fill:"region", transform:{stack:true}} (without stack: grouped side by side). ' +
        'A call with `marks` or `series` opens a new chart (fields not given: defaults); a call without either changes only the fields given ' +
        '(window, title, axis scale, overlays...) on the current chart and keeps the rest. `series` is the older shorthand (one line per entry, ' +
        'with bolted-on bars/fill/stat options), drawn under the marks. Each series is one of: x/y inline ' +
        '(series + overlays under 64 kB of JSON in all), `file` (more data), `fn` (a function of x evaluated in f64, re-sampled as they zoom: ' +
        'prefer it for mathematical functions) or `sample`. Histograms: `bars: true` on bin centres x and counts y. null y break lines. ' +
        'Distributions and fits: give raw `samples` (or x/y for linreg) with a `stat`, not precomputed quantiles; it expands into ' +
        'ordinary geoms (step lines, fills, intervals, boxes, bars, fns) and state.stats returns its numbers. ' +
        'Text may use Unicode math (𝑥 𝜆̂ 𝑥̄ ℝⁿ 10⁻⁶ ≤ ∇ ⟨⟩); characters the font lacks are listed in state.missingGlyphs. ' +
        'Waits for the frame and returns the control, renderer state (window, y range) and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          subtitle: { type: 'string', description: 'dimmer text after the title' },
          xlabel: { type: 'string' },
          ylabel: { type: 'string' },
          x: { ...axis, description: 'x axis' },
          y: { ...axis, description: 'y axis' },
          marks: {
            type: 'array',
            description: 'drawn in order after any series (fills and bars under lines, dots, rules and text). Strings in x or y make categories ' +
              '(x left to right, y top to bottom, in order of appearance; bars and rects take a slot each, 0.8 wide); fill strings make a palette legend ' +
              '(at most 7 categories, Okabe–Ito, in order of appearance), fill numbers a colormap with a colour bar (`color`)',
            items: {
              type: 'object',
              properties: {
                type: { type: 'string', enum: ['line', 'dot', 'bar', 'barX', 'area', 'rect', 'rule', 'text', 'interval', 'box'], description:
                  'line: x, y (or fn); dot: x, y (r, fill); bar: vertical, band x (category, or x1/x2 from bin), 0..y (or y1..y2 from stack); barX: horizontal, band y, 0..x; ' +
                  'area: x, 0..y or y1..y2 (or fn: 0..f); with x1, x2 and y (density of a categorical x): a violin, x1..x2 as functions of y; rect: x1..x2 × y1..y2, or a band of a category ' +
                  '/ the unit cell around a number on each axis (heatmaps); rule: y alone (horizontal lines across, x1..x2 to shorten) or x alone (vertical); ' +
                  'text: x, y, text; interval: x, y1..y2 (error bars, cap); box: box plots of y per x (boxplot transform)' },
                data: { description: 'columns {name: [...], ...} (one length) or rows [{name: value, ...}, ...]; numbers, strings (categories), null (a gap)' },
                file: { type: 'string', description: 'instead of data: absolute path to a .csv/.tsv with a header row (columns by name) or .json (columns or rows); re-read when it changes' },
                fn: { description: 'line, area: instead of data, a function of x evaluated in f64 and re-sampled on zoom (the series `fn`: a name or {expr, params, domain})' },
                x: { type: ['string', 'number'], description: 'channel: a column name, a number constant; or after a transform the name of its output ("x", "y", ...)' },
                y: { type: ['string', 'number'] }, x1: { type: ['string', 'number'] }, x2: { type: ['string', 'number'] }, y1: { type: ['string', 'number'] }, y2: { type: ['string', 'number'] },
                z: { type: ['string', 'number'], description: 'a column grouping rows into separate series (stack order, dodge, lines per group); default: a categorical fill' },
                fill: { type: ['string', 'number'], description: 'colour: a column (strings: categories, a legend entry each; numbers: the colormap) or a CSS colour; stroke is the same channel' },
                stroke: { type: ['string', 'number'] },
                r: { type: ['string', 'number'], description: 'dot radius: a column (sqrt scale, area ∝ value, top-level `r`) or CSS px' },
                opacity: { type: ['string', 'number'], description: '0..1, a column or constant (areas: fill opacity, default 0.3, stacked 0.75)' },
                text: { type: ['string', 'number'], description: 'text: a column, a transform output (e.g. "y": a bin count), or a constant' },
                transform: {
                  type: 'object',
                  description: 'applied in this order: bin {width? | thresholds?, reduce?: count|density|proportion} (x → x1, x2, x centre, y; bins shared by the z groups); ' +
                    'group {reduce?: count|sum|mean, sort?: asc|desc} (rows by the band category, x (barX: y), and z → y (barX: x)); window {k, reduce?: mean|median|min|max|sum} ' +
                    '(rolling over k points centred, by x per group); density {bandwidth? (default Silverman), width? (violin slot fraction, 0.9)} (Gaussian KDE of x → x, y; ' +
                    'of y per x category → violins); then stack {offset?: none|expand|center} (y across z at each x → y1, y2; barX: x). Or one stat: ecdf {band?: α}, ' +
                    'qq {reference?}, linreg {band?: α, kind?: mean|prediction} on x, y, boxplot {width?} of y per x. `true` takes the defaults, e.g. {bin: true, stack: true}',
                },
                label: { type: 'string', description: 'legend entry of a mark without categorical fill' },
                legend: { type: 'boolean' },
                extent: { type: 'boolean', description: 'false: drawn, but sets neither the x window nor the y range' },
                width: { type: 'number', description: 'line width, CSS px' },
                dash: { type: 'boolean' },
                interpolation: { type: 'string', enum: ['linear', 'pchip', 'spline', 'step'] },
                cap: { type: 'number', description: 'interval: cap width, CSS px (6)' },
                inset: { type: 'number', description: 'bar, rect: CSS px off each side (bars 1, rects 0)' },
                anchor: { type: 'string', description: 'text: horizontal l|m|r then vertical t|m|b, e.g. "lm" (default "mm")' },
                dx: { type: 'number', description: 'text: offset, CSS px' },
                dy: { type: 'number' },
                slice: { type: 'string', description: 'a numeric column: only the rows where it equals the slider of that name are drawn (a slider is added if not given; e.g. a heatmap over time)' },
              },
              required: ['type'],
            },
          },
          color: { type: 'object', description: 'marks\' colour scale: {scheme?: a colormap (viridis, magma, turbo, rdbu, ...) for numeric fill, domain?: [lo, hi] or the categories in order, label?: colour bar title, invert?}' },
          r: { type: 'object', description: 'marks\' radius scale: {range?: [2, 12] CSS px, domain?: [0, max]}' },
          series: {
            type: 'array',
            description: 'the older shorthand: drawn in order (later on top), under the marks',
            items: {
              type: 'object',
              properties: {
                x: { type: 'array', items: { type: 'number' }, description: 'default: 0, 1, 2, ...' },
                y: { type: 'array', items: { type: ['number', 'null'] }, description: 'null: a gap' },
                file: { type: 'string', description: 'instead of x/y: absolute path to a .json ({x?, y} or [y, ...]) or a .csv/.tsv of 1 (y) or 2 (x, y) columns (a header line and # comments are skipped); read by the renderer, again when the file changes' },
                fn: {
                  description: 'instead of x/y: a function of x evaluated in f64 (special functions: Boost.Math) over the visible window: a name or {name, ...params, domain?: [a, b]}. ' +
                    'Names (params): bumps, blocks, runge, step, besselJ (order ν), sinc (k: sin(kx)/(kx)), bode (f0, zeta), dispersion (band), ' +
                    'dampedCos (tau, omega: exp(-x/tau) cos(omega x)), poisson (lambda, scale: scale λ^x e^-λ / Γ(x+1)). ' +
                    'Or any function of x as an expression: {expr: "a*exp(-(x-m)^2/(2*s^2))", params: {a: 1, m: 0, s: 1}, domain: [-4, 4]}: ' +
                    '+ - * / ^ (or **), comparisons (1 or 0), pi, e, sin cos tan asin acos atan atan2 sinh cosh tanh exp log log2 log10 sqrt cbrt ' +
                    'abs sign floor ceil min max pow hypot erf erfc lgamma (ln|Γ|) tgamma digamma zeta expint (Ei) besselJ besselY besselI besselK (ν, x) ' +
                    'airyAi lambertW (W₀) ellipK (modulus k) sinc (sin x / x); poles and domain errors give NaN (a gap)',
                },
                sample: { type: 'object', description: 'instead of x/y: {fn, n}: n even samples of fn over its domain (e.g. to show an interpolation)' },
                label: { type: 'string', description: 'legend entry (unlabelled series stay out of the legend)' },
                color: { type: 'string', description: 'CSS hex (default: the next palette colour)' },
                width: { type: 'number', description: 'line width, CSS px (default 1.5)' },
                dash: { type: 'boolean' },
                markers: { enum: [true, false, 'only'], description: 'true: dots at the points; "only": dots and no line' },
                bars: { type: ['boolean', 'number'], description: 'a bar per point, from 0: true = as wide as the closest x spacing, a number = that width in x units' },
                opacity: { type: 'number', description: '0..1 (default 1)' },
                interpolation: { type: 'string', enum: ['linear', 'pchip', 'spline', 'step'], description: 'between points (default linear); pchip: monotone cubic; spline: not-a-knot cubic; step: flat to the next x, then jump' },
                envelope: { type: 'object', description: '{type: "cummin"|"cummax", mode?: "add" (an extra series, default)|"replace", reverse?: boolean, color?}: running min/max' },
                legend: { type: 'boolean', description: 'default: whether it has a label' },
                fill: {
                  type: 'object',
                  description: '{lo?, hi?, opacity? (0.2)}: shade between two bounds, below the lines; each a number (e.g. 0: area under), an array (one per x, ' +
                    'interpolated like the series) or an fn spec; a missing one is the series\' own y / fn. Without y: only the band (confidence band, its own legend swatch)',
                },
                intervals: { type: 'object', description: '{lo: [...], hi: [...], cap?: px (6), label?}: a vertical range at each x (error bars), in the series colour' },
                box: { type: 'object', description: '{q1: [...], q3: [...], width?: x units (0.6)}: a box q1..q3 at each x, y = its median line' },
                samples: { type: 'array', items: { type: 'number' }, description: 'raw samples for `stat` (or `file`\'s y)' },
                stat: {
                  description: 'a name or {type, ...options}: "ecdf" (band?: α, e.g. 0.05: DKW band), "qq" (normal Q–Q; reference?: false drops the quartile line), ' +
                    '"box" (position?: x, default its index among box series; width?), "hist" (bins? | binWidth?, default Freedman–Diaconis; density?), ' +
                    '"linreg" on x/y (band?: α, default 0.05; kind?: "mean" | "prediction"): data markers, fit ŷ = a + bx and its t band',
                },
              },
            },
          },
          overlays: {
            type: 'array',
            description: 'reference lines and bands',
            items: {
              type: 'object',
              properties: {
                kind: { type: 'string', enum: ['hline', 'vline', 'band'] },
                axis: { type: 'string', enum: ['x', 'y'], description: 'band: the axis of its values (default x)' },
                values: { type: 'array', items: { type: 'number' }, description: 'hline/vline: [v]; band: [lo, hi]' },
                label: { type: 'string' },
                color: { type: 'string', description: 'CSS hex (default #666666)' },
                width: { type: 'number' },
                dash: { type: 'boolean' },
              },
              required: ['kind', 'values'],
            },
          },
          legend: { type: 'boolean', description: 'default true (in the emptiest corner)' },
          endLabels: { type: 'boolean', description: 'each labelled line\'s name at its right end (last visible point), in the right margin; default: on with 5 or more labelled lines' },
          sliders: { type: 'object', description: '{name: [min, max] | {min, max, step?, label?}}: in-frame sliders the user drags, each setting the fn param (expr params, or a named fn\'s order, lambda, tau, omega, zeta...) of that name in every series/mark; its value starts at the param\'s. A mark `slice` column\'s slider: {label?, value?} or [min, max], snapping to the column\'s values. state.sliders reports the values the user set' },
          x0: { type: ['number', 'null'], description: 'visible x window start, data units; null: all data' },
          x1: { type: ['number', 'null'] },
          ymin: { type: ['number', 'null'], description: 'null: fit what is visible' },
          ymax: { type: ['number', 'null'] },
        },
      },
      isDeferred: false,
    })
    await $.tool.register({
      name: 'scene',
      description:
        'Show the user a live 3D scene of your own data inline in the chat, right here between your text (WebGPU; they drag to orbit, ' +
        'shift-drag/+/- to zoom, hover a point for its row, r resets). Built from `marks` like `chart`: dot (3D scatter, point clouds: ' +
        'shaded spheres, 10⁵ points stay fast), line (3D polylines), mesh (triangles), each over `data` columns or a `file`, surface (NURBS / Bézier patches, exact), volume (a ray-marched 3D field around them, one per scene). ' +
        'Examples: scatter {type:"dot", data, x:"a", y:"b", z:"c", fill:"species"}; point cloud coloured by a value {type:"dot", file, fill:"t", r:1.5}; ' +
        'convex hull over points {type:"dot", data, transform:{hull:true}} plus the points themselves as a second dot mark; ' +
        'surface mesh {type:"mesh", vertices:[[x,y,z],...], faces:[[i,j,k],...], shading:"smooth"}; ' +
        'exact patch under points on one colour scale {type:"surface", nurbs:{degrees:[2,2], shape:[3,3], points, weights, knots, fields:[f]}, fill:"field"} plus {type:"dot", data, fill:"f"}; ' +
        'a field with a surface and points in it {type:"volume", file:"/abs/g.npy", extent:[[0,2],[0,2],[0,2]], opacity:0.05} plus an opaque surface or mesh and dots: rays end at them, so points inside are tinted by the field in front. ' +
        'Axes: a box with ticks on the edges facing the camera, equal aspect unless `aspect`. A call with `marks` opens a new scene; ' +
        'without, it changes only the fields given (camera, title, aspect...). Waits for the frame and returns the control, renderer state ' +
        '(camera, box, hover row) and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          subtitle: { type: 'string', description: 'dimmer text after the title' },
          marks: {
            type: 'array',
            description: 'drawn together, depth-tested; translucent ones (opacity < 1) after the opaque, so what is behind them shows through dimmed; a volume after the opaque (which cut its rays) and under the translucent (drawn over it, untinted)',
            items: {
              type: 'object',
              properties: {
                type: { type: 'string', enum: ['dot', 'line', 'mesh', 'surface', 'volume'], description: 'dot: x, y, z (fill, r); line: x, y, z in row order, one polyline per `group`; mesh: vertices + faces, or x, y, z columns + faces (row indices); surface: `nurbs`, drawn as a smooth mesh (fill, opacity, edges, shading; no data, no slice); volume: a 3D array `file` over `extent`, ray-marched on the colour scale of the numeric fills (`color`; `domain`, `threshold`, `opacity`; no slice)' },
                data: { description: 'columns {name: [...], ...} (one length) or rows [{name: value, ...}, ...]; null: a gap' },
                file: { type: 'string', description: 'instead of data: absolute path to a .csv/.tsv with a header row or .json (columns or rows); for more than 64 kB; volume: a .npy (3D, little-endian f4/f8, C order) or raw .f32 / .bin with `shape`; NaN is clear' },
                shape: { type: 'array', items: { type: 'integer' }, description: 'volume, raw file: [nz, ny, nx] (x fastest)' },
                extent: { type: 'array', items: { type: 'array', items: { type: 'number' } }, description: 'volume: [[x0, x1], [y0, y1], [z0, z1]], the positions of its first and last samples (default: their indices)' },
                domain: { type: 'array', items: { type: 'number' }, description: 'volume: [lo, hi] of the colour scale (default: its range with the numeric fills\'; `color.domain` overrides)' },
                threshold: { type: 'number', description: 'volume: clear below this fraction of the colour domain (0.1)' },
                x: { type: ['string', 'number'], description: 'a column name or a constant (default: the column "x")' },
                y: { type: ['string', 'number'] }, z: { type: ['string', 'number'] },
                fill: { type: 'string', description: 'colour: a column (strings: categories, Okabe–Ito, a legend entry each, at most 7; numbers: the colormap and a colour bar) or #rrggbb; surface: "field" (the patch\'s first field, on the colour scale of the numeric fills) or #rrggbb; stroke is the same channel' },
                stroke: { type: 'string' },
                r: { type: ['string', 'number'], description: 'dot radius: CSS px (3) or a column (sqrt scale, top-level `r`)' },
                group: { type: 'string', description: 'line: a column splitting rows into polylines (default: a categorical fill)' },
                opacity: { type: 'number', description: '0..1 (hull: 0.35); volume: opacity at the top of the domain per 1/56 of its largest side (0.25; lower to see deep into a dense field)' },
                width: { type: 'number', description: 'line and edge width, CSS px (line 2, edges 1)' },
                vertices: { type: 'array', items: { type: 'array', items: { type: 'number' } }, description: 'mesh: [[x, y, z], ...]' },
                faces: { type: 'array', items: { type: 'array', items: { type: 'integer' } }, description: 'mesh: [[i, j, k], ...] vertex/row indices, counter-clockwise seen from outside' },
                nurbs: {
                  type: ['object', 'array'],
                  items: { type: 'object' },
                  description: 'surface: a patch as `view`\'s nurbs, {degrees: [p, q], shape: [nu, nv], points: flat xyz (nu*nv*3, u outer), weights: nu*nv, ' +
                    'knots: [u knots (nu+p+1), v knots (nv+q+1)], fields?: [nu*nv control values]}, or an array of them (1 to 256); ' +
                    'an edge collapsed to a point makes a triangle',
                },
                resolution: { type: 'integer', description: 'surface: steps along u and along v, shared among the knot spans (32)' },
                shading: { type: 'string', enum: ['flat', 'smooth'], description: 'mesh (default flat; smooth: averaged vertex normals); surface (default smooth: its exact normals)' },
                edges: { type: ['boolean', 'string'], description: 'mesh: draw its edges (true, or a #rrggbb colour)' },
                transform: { type: 'object', description: '{hull: true}: the 3D convex hull of the points as a mesh (≥ 4 non-coplanar points)' },
                label: { type: 'string', description: 'legend entry of a mark without categorical fill' },
                legend: { type: 'boolean' },
                slice: { type: 'string', description: 'a numeric column: only the rows where it equals the slider of that name are drawn (mesh: faces wholly in the slice); a slider is added if not given' },
              },
              required: ['type'],
            },
          },
          sliders: { type: 'object', description: '{column: {label?, value?} | [min, max]}: in-frame sliders over marks\' `slice` columns, snapping to their values' },
          xlabel: { type: 'string', description: 'default: the x column\'s name' },
          ylabel: { type: 'string' },
          zlabel: { type: 'string' },
          aspect: { type: ['string', 'array'], items: { type: 'number' }, description: '"equal" (default: one unit the same length on every axis), "auto" (each axis normalised to a cube) or [ax, ay, az]' },
          color: { type: 'object', description: '{scheme?: a colormap (viridis, magma, turbo, twilight, rdbu, ...) for numeric fill, domain?: [lo, hi] or the categories in order, label?: colour bar title, invert?}' },
          r: { type: 'object', description: '{range?: [2, 12] CSS px, domain?: [0, max]}' },
          legend: { type: 'boolean' },
          axes: { type: 'boolean', description: 'false: no box, grid or ticks' },
          azimuth: { type: 'number', description: 'degrees (35)' },
          elevation: { type: 'number', description: 'degrees, -89..89 (25)' },
          zoom: { type: 'number', description: 'relative to the auto-fit framing (1)' },
        },
      },
      isDeferred: false,
    })
    await $.tool.register({
      name: 'volume',
      description:
        'Show the user a live 3D volume of your own data inline in the chat, right here between your text (WebGPU ray marcher; they drag to orbit, ' +
        'shift-drag/+/- to zoom, hover for the value, r resets). `volume: {dataset: "file", path}` reads a .npy (shape and dtype from its header) or raw ' +
        'little-endian .f32/.bin (give shape); a 4D array [slices, z, y, x] gets an in-frame slider over its leading axis. NaN voxels are transparent ' +
        '(also skipped by the colour domain and isosurfaces). Write the array to a file, don\'t inline it. A call with `volume` opens a new view ' +
        '(fields not given: defaults); without, it changes only the fields given. Waits for the frame and returns the control, renderer state ' +
        '(data domain, slice, isosurface triangles, hover) and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          volume: {
            type: 'object',
            description: '{dataset: "file", path: absolute .npy | .f32 | .bin, shape?: [nz, ny, nx] | [ns, nz, ny, nx] (raw only; C order, x fastest), dtype?: "f32" | "f64" (raw, default f32), ' +
              'axes?: [x, y, z, slice?] each {lo, hi, label?, unit?} (default: indices; one unit on x, y, z: box to scale), nan?: "transparent"}; ' +
              'or a generator {dataset: "hydrogen", n, l, m} (also hydrogen_spherical, breathing, tight_binding, fermi)',
          },
          slice: { type: 'integer', description: 'index of the leading dimension shown (the user can drag the slider)' },
          sliders: { type: ['object', 'boolean'], description: '{slice: {label?}}: the slice slider (default for 4D files; false: none)' },
          title: { type: 'string' },
          subtitle: { type: 'string' },
          colormap: { type: 'string', description: 'viridis, inferno (default), magma, rdbu, twilight..., or "auto" (RdBu when the data cross `center`)' },
          invertColormap: { type: 'boolean' },
          scale: { type: 'string', enum: ['linear', 'log', 'symlog', 'sqrt', 'root'] },
          rootN: { type: 'number', description: 'root scale: t^(1/n), 0.1..10 (3)' },
          symlogThreshold: { type: 'number' },
          domain: { type: ['array', 'null'], items: { type: ['number', 'null'] }, description: '[lo, hi] colour domain, either null: the data\'s (over every slice)' },
          symmetric: { type: 'boolean', description: 'domain symmetric about `center` (0): diverging data' },
          center: { type: 'number' },
          mode: { type: 'string', enum: ['DVR', 'MIP', 'Iso'], description: 'DVR (default): emission-absorption; MIP: maximum; Iso: first crossing of threshold' },
          opacity: { type: 'number', description: 'DVR opacity scale (0.25), per 1/56 of the box: alike at any resolution' },
          threshold: { type: 'number', description: 'normalised value (0..1) below which voxels are clear (0.1)' },
          quality: { type: 'string', enum: ['low', 'medium', 'high'] },
          isoLevels: { type: ['array', 'null'], items: { type: 'number' }, description: 'opaque isosurfaces (marching cubes) at these data values' },
          isoColors: { type: 'string', enum: ['levels', 'colormap'] },
          showVolume: { type: 'boolean', description: 'false: isosurfaces only' },
          showBox: { type: 'boolean' },
          showAxes: { type: 'boolean' },
          valueLabel: { type: 'string', description: 'hover name of the value' },
          up: { type: 'string', enum: ['y', 'z'], description: 'box axis drawn up (y)' },
          azimuth: { type: 'number', description: 'degrees' },
          elevation: { type: 'number', description: 'degrees, -89..89' },
          zoom: { type: 'number', description: 'relative to the auto-fit framing (1)' },
        },
      },
      isDeferred: false,
    })
    await $.tool.register({
      name: 'plot',
      description:
        'Show the user a live, interactive PCHIP interpolation plot inline in the chat, right here between your text ' +
        '(WebGPU renderer, PCHIP drawn as an exact NURBS curve; they drag to pan, shift-drag/+/- to zoom, hover for t, PCHIP(t), f(t)). ' +
        'The newest view is live; earlier ones freeze. Sets any given parameters (others keep their current values; x0/x1 null = full domain), ' +
        'waits for the frame, and returns the control, renderer state (window, max error, hover) and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          function: { type: 'string', enum: ['bumps', 'runge', 'step', 'blocks'] },
          n: { type: 'number', description: 'uniform samples (knots of the interpolant)' },
          x0: { type: ['number', 'null'], description: 'view window start, function units' },
          x1: { type: ['number', 'null'], description: 'view window end' },
          ymin: { type: ['number', 'null'] },
          ymax: { type: ['number', 'null'] },
          showTrue: { type: 'boolean', description: 'draw the true function' },
          showSamples: { type: 'boolean' },
          compare: { type: 'string', enum: ['none', 'cubic'], description: 'overlay a not-a-knot cubic spline' },
        },
      },
      isDeferred: false,
    })
    await $.tool.register({
      name: 'view',
      description:
        'Show the user a live, interactive 3D NURBS surface view inline in the chat, right here between your text ' +
        '(WebGPU renderer evaluating the NURBS surface in a compute shader; they drag to orbit, shift-drag/+/- to zoom, hover for values). ' +
        'The newest view is live; earlier ones freeze. Sets any given parameters (others keep their current values), waits for the frame, ' +
        'and returns the control, renderer state (camera, value range) and a PNG snapshot path you can Read to see what the user sees.',
      inputSchema: {
        type: 'object',
        properties: {
          surface: { type: 'string', enum: SURFACES, description: 'a built-in surface; dini and breather: pseudospherical (curvature -1), coloured by height / the sine-Gordon angle; teapot: Newell\'s 32 bicubic Bézier patches, exact (by height; mean-curvature with rdbu reads well)' },
          nurbs: {
            type: ['object', 'array'],
            items: { type: 'object' },
            description: 'a surface of your own instead of `surface`: {degrees: [p, q], shape: [nu, nv], points: flat xyz (nu*nv*3, u outer), ' +
              'weights: nu*nv, knots: [u knots (nu+p+1), v knots (nv+q+1)], fields?: [nu*nv control values]}. A polynomial graph z = f(x, y) ' +
              'is exactly a Bezier patch (Bernstein coefficients); colour by `field` for a scalar on it (e.g. det of the Hessian), ' +
              'or by gaussian-/mean-curvature. Or an array of such patches drawn together (a model made of Bézier patches; hover reports `patch`)',
          },
          colorBy: { type: 'string', enum: ['field', 'height', 'field-gradient', 'height-gradient', 'gaussian-curvature', 'mean-curvature'] },
          colormap: { type: 'string', enum: [...SEQUENTIAL_MAPS, ...DIVERGING_MAPS, ...CYCLICAL_MAPS].map((m) => m.toLowerCase()), description: 'diverging maps (rdbu, puor, ...) for signed values, with a symmetric `domain`' },
          invertColorMap: { type: 'boolean' },
          domain: { type: ['array', 'null'], items: { type: 'number' }, description: '[lo, hi] of the colour map; null: the renderer\'s own (2nd-98th percentile)' },
          azimuth: { type: 'number', description: 'degrees' },
          elevation: { type: 'number', description: 'degrees, -89..89' },
          zoom: { type: 'number', description: 'relative to the auto-fit framing (1)' },
          knotLines: { type: 'boolean' },
        },
      },
      isDeferred: false,
    })
    await $.tool.register({
      name: 'show',
      description:
        'Show a PNG image to the user inline in the chat (real pixels in their terminal). ' +
        'Use whenever a picture would help: plots, heatmaps, renders. By default it fills the transcript width at the image\'s own aspect ratio; ' +
        'pass columns/rows only to make it smaller.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Absolute path to a PNG file' },
          columns: { type: 'number', description: 'Width in terminal cells (default: full width)' },
          rows: { type: 'number', description: 'Height in terminal cells (default: from aspect ratio)' },
          caption: { type: 'string', description: 'Short description of the image' },
        },
        required: ['path'],
      },
      isDeferred: false,
    })
    return next(e)
  })

  on('tool.call', { tool: TOOL }, async ($, e) => {
    const input = e as unknown as { path?: string; columns?: number; rows?: number; caption?: string }
    const path = input.path ?? ''
    if (!path.startsWith('/') || !path.toLowerCase().endsWith('.png')) {
      return { deny: 'path must be an absolute path to a .png file' }
    }
    if (!(await $.fs.exists(path))) return { deny: `no such file: ${path}` }
    const { base64 } = await $.fs.read(path, { as: 'bytes' })
    const header = decodeBase64(base64.slice(0, 32)) // signature + IHDR: 24 bytes
    const png = new DataView(header.buffer)
    if (header.length < 24 || png.getUint32(12) !== 0x49484452) return { deny: `not a PNG: ${path}` } // 'IHDR'
    const shown: Shown = {
      path,
      columns: input.columns,
      rows: input.rows,
      width: png.getUint32(16),
      height: png.getUint32(20),
      caption: input.caption ?? path.split('/').pop() ?? 'image',
      generation: Date.now(),
    }
    return { result: JSON.stringify(withWarning({ ...shown })) }
  })

  // A view's call row (`sciviz - chart (MCP)(title: ..., series: [...])`) is code in the middle of
  // the explanation: the figure stands alone between the text. ctrl+o (expanded) and errors show it.
  on('ui.render', { component: 'ToolUse' }, async ($, e, next) => {
    const views = [TOOL, VIEW_TOOL, PLOT_TOOL, OPEN_TOOL, CHART_TOOL, SCENE_TOOL, VOLUME_TOOL]
    if (!views.includes(e.props.tool) || e.props.isExpanded || e.props.isErrored || e.surface !== 'terminal') return next(e)
    const { Box } = $.ui.resolve(e)
    return <Box />
  })

  on('ui.render', { component: 'ToolResult' }, async ($, e, next) => {
    if ([VIEW_TOOL, PLOT_TOOL, OPEN_TOOL, CHART_TOOL, SCENE_TOOL, VOLUME_TOOL].includes(e.props.tool) && !e.props.isErrored && e.surface === 'terminal') {
      // after a reload kindOf is empty: these rows are all frozen PNGs then, but they need their kind
      let kind = kindOf[e.props.tool_use_id] ?? (e.props.tool === VIEW_TOOL ? 'surface' : e.props.tool === PLOT_TOOL ? 'pchip' : e.props.tool === CHART_TOOL ? galleryKind('line.ts') : e.props.tool === SCENE_TOOL ? galleryKind('scene.ts') : e.props.tool === VOLUME_TOOL ? galleryKind('volume.ts') : undefined)
      if (!kind && e.props.tool === OPEN_TOOL) {
        const out = e.props.output as any
        const id = (out?.entry ?? JSON.stringify(out).match(/\\?"entry\\?":\\?"([^"\\]+)/)?.[1]) as string | undefined
        const entry = id ? (await entries($)).find((x) => x.id === id) : undefined
        if (entry) kind = galleryKind(entry.renderer)
      }
      if (kind) return drawView($, e, e.props.tool_use_id, kind)
    }
    if (e.props.tool !== TOOL || e.props.isErrored || e.surface !== 'terminal') return next(e)
    let s: Shown | undefined
    try {
      const o = e.props.output
      s = (typeof o === 'string' ? JSON.parse(o) : Array.isArray(o) ? JSON.parse(o[0]?.text ?? '') : o) as Shown
    } catch {
      return next(e)
    }
    if (!s?.path) return next(e)
    const { Box, Image, Text } = $.ui.resolve(e)
    if (imageWarning()) return <Text dimColor>{imageWarning()}  ·  {s.caption}</Text>
    const size = cellSize(s, e.viewport)
    return (
      <Box flexDirection="column">
        <Image
          source={{ file: s.path, format: 'png', generation: s.generation }}
          columns={size.columns}
          rows={size.rows}
          alt={`${imageWarning() || NO_IMAGES} (${s.caption})`}
        />
        <Text dimColor>{s.caption}</Text>
      </Box>
    )
  })
}
