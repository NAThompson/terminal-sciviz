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
// surface: deno/surface.ts, pchip: deno/plot1d.ts; gallery entries: deno/{line,curve,complex,volume}.ts (all WebGPU).
// Protocol: control.json carries an increasing `rev`; frame lines are `frame <gen> <w> <h> <shm|-> <rev>`;
// a snapshot request {path, rev} is acknowledged by `snapshot <rev> <path>` once a frame at/after rev
// is saved. Writes are coalesced (the renderer retries a torn file and ignores stale revs).
// The newest view (a tool result or a /sciviz-* output) is live: frames are blitted into it and a
// transparent Client over it takes the pointer and keys. Older views freeze on the PNG of their
// last settled frame (`${RUN}/views/<id>.png`).
const VIEW_TOOL = 'mcp__sciviz__view'
const PLOT_TOOL = 'mcp__sciviz__plot'
const OPEN_TOOL = 'mcp__sciviz__gallery_open'
// Resolved at session start (resolvePaths): ROOT holds deno/ (the plugin's own folder once installed,
// or the projectDir option); RUN is per-user state (control files, frames, snapshots).
let ROOT = ''
let RUN = ''
let DENO = '' // absolute path, or '' when none was found
const MAX_FRAME_PX = 4096 // ImageSource limit for shm/file frames
const DRAG_SCALE = 0.55 // resolution while dragging (faster frames); full Retina again on release
const VIEW_ASPECT = 16 / 10 // pixels, width / height
const SURFACES = ['wave', 'torus', 'sphere', 'wave-test']
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
  return group[(Math.max(i, 0) + step + group.length) % group.length]
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
const SECTIONS = ['lines', 'curves', 'complex', 'volumes']

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
const view = { live: '', liveMinRev: 0, intended: '', cols: 80, rows: 25, pxW: 960, pxH: 600,
  drag: null as null | { x: number; y: number; ctl: Ctl; win: Win | null; zooming: boolean } }

const cycle = (list: string[], cur: string) => list[(list.indexOf(cur) + 1) % list.length]
const viewPng = (id: string) => `${RUN}/views/${id.replace(/[^A-Za-z0-9_-]/g, '_')}.png`
const liveKind = (): Kind | undefined => kindOf[view.live]
const legacyInteraction = (kind: Kind): Interaction => (kind === 'surface' ? 'orbit' : 'pan-zoom-x')

let manifest: Entry[] | null = null
/** The gallery's entries (gallery.ts manifest), cached; `fresh` re-reads it (new entries since load). */
async function entries($: any, fresh = false): Promise<Entry[]> {
  if (manifest && !fresh) return manifest
  try { manifest = JSON.parse(await $.fs.read(`${ROOT}/deno/gallery/index.json`)) as Entry[] } catch { manifest = [] }
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
async function settle($: any, id = view.live) {
  const kind = kindOf[id]
  if (!kind) return
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
      await $.process.run(['/bin/mkdir', '-p', dir(kind), `${RUN}/views`])
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
          if (snap) onSnapshot(kind, +snap[2], snap[3], !!snap[1])
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
    await $.ui.blit({ requestId: view.live, key: 'view', columns: view.cols, rows: view.rows, source }).catch(() => {})
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
  if (id !== view.live) {
    const png = viewPng(id)
    return (await $.fs.exists(png))
      ? <Image source={{ file: png, format: 'png' }} columns={cols} rows={rows} alt="sciviz view (frozen)" />
      : <Text dimColor>sciviz view (no frame saved)</Text>
  }
  Object.assign(view, { cols, rows })
  const ctl = KINDS[kind].ctl
  // exactly the box's device pixels, so Ghostty shows the frame 1:1 (scaled down only past MAX_FRAME_PX)
  const fit = Math.min(1, MAX_FRAME_PX / (cols * CELL_PX_W), MAX_FRAME_PX / (rows * CELL_PX_H))
  const pxW = Math.round(cols * CELL_PX_W * fit), pxH = Math.round(rows * CELL_PX_H * fit)
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
  return (
    <Box width={cols} height={rows}>
      <Image key="view" source={source} columns={cols} rows={rows} alt={setupError || lastError[kind] ? `sciviz: ${setupError || lastError[kind]}` : 'sciviz: starting the GPU renderer…'} />
      <Box position="absolute" top={0} left={0}>
        <Client key="input" module="./input.tsx" width={cols} height={rows} />
      </Box>
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

async function onKey($: any, kind: Kind, k: string) {
  const ctl = KINDS[kind].ctl
  const info = viewInfo[view.live]
  const interaction = info?.interaction ?? legacyInteraction(kind)
  if (k === 'r' && info) { // back to how the view opened
    for (const key of ['azimuth', 'elevation', 'zoom', 'x0', 'x1', 'y0', 'y1', 'ymin', 'ymax'])
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

/** The root degree on its slider's grid: 0.1 to 10 in steps of 0.05. */
const snapRootN = (n: number) => Math.min(10, Math.max(0.1, Math.round(n / 0.05) * 0.05))

/** The renderer's in-frame slider (state.json rootSlider: track rect in frame px, range), if any. */
async function rootSlider($: any, kind: Kind) {
  try {
    const st = JSON.parse(await $.fs.read(`${dir(kind)}/state.json`))
    const r = st.rootSlider
    return r && Array.isArray(r.rect) ? { ...r, w: st.width, h: st.height } : null
  } catch { return null }
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
  const cols = ev.columns || view.cols, rows = ev.rows || view.rows
  const x = ev.fine?.x ?? ev.x + 0.5, y = ev.fine?.y ?? ev.y + 0.5
  // a move with no button held while a drag is open: its `up` was lost (outside the view, a remount).
  // End the drag, or every hover would keep orbiting the camera and re-rendering.
  if (view.drag && ev.type === 'move' && !ev.button) ev = { ...ev, type: 'up' }
  // For the native overlay (overlay/*.ts): where the pointer is in the view, in cells; the
  // overlay pairs it with the global mouse position to place itself exactly over the view.
  void writeOverlay($, JSON.stringify({ seq: ++overlaySeq, type: ev.type, x, y, cols, rows,
    fine: !!ev.fine, drag: ev.type === 'down' || (!!view.drag && ev.type !== 'up' && ev.type !== 'leave'), kind, id: view.live }))
  const sliderAt = async () => {
    const sl = ctl.scale === 'root' ? await rootSlider($, kind) : null
    if (!sl) return null
    // hit test in cells, at least a whole cell beyond the track on every side: the slider is under
    // a cell tall on screen and terminal pointer positions are coarse
    const [x0, y0, x1, y1] = sl.rect
    const cx0 = (x0 / sl.w) * cols - 1, cx1 = (x1 / sl.w) * cols + 1
    const cy0 = (y0 / sl.h) * rows - 1, cy1 = (y1 / sl.h) * rows + 1
    if (x < cx0 || x > cx1 || y < cy0 || y > cy1) return null
    return { sl, px: Math.min(x1, Math.max(x0, (x / cols) * sl.w)) }
  }
  const setFromSlider = (sl: any, px: number) => {
    const [x0, , x1] = sl.rect
    ctl.rootN = snapRootN(sl.min + ((px - x0) / Math.max(1, x1 - x0)) * (sl.max - sl.min))
  }
  if (ev.type === 'down') {
    const hit = await sliderAt()
    if (hit) { view.drag = { x, y, ctl: { ...ctl }, win: null, zooming: false, slider: hit.sl } as any; setFromSlider(hit.sl, hit.px); await writeControl($, kind); return }
  }
  if (ev.type === 'move' && (view.drag as any)?.slider) {
    const sl = (view.drag as any).slider
    setFromSlider(sl, Math.min(sl.rect[2], Math.max(sl.rect[0], (x / cols) * sl.w)))
    await writeControl($, kind)
    return
  }
  if (ev.type === 'down') {
    const win = interaction === 'orbit' ? null : await readWindow($, kind)
    if (interaction === 'orbit') await resolveCamera($, kind)
    view.drag = { x, y, ctl: { ...ctl }, win, zooming: !!ev.shift }
    ctl.hover = null
    // lower resolution while orbiting (3D frames cost most); 2D keeps its layout so pan/zoom stay anchored
    // while orbiting: fewer pixels, and renderers that support it (volume.ts) cut their ray-march samples
    if (interaction === 'orbit') Object.assign(ctl, { width: Math.round(view.pxW * DRAG_SCALE), height: Math.round(view.pxH * DRAG_SCALE), interacting: true })
  } else if (ev.type === 'up' || (ev.type === 'leave' && view.drag)) {
    view.drag = null
    if (interaction === 'orbit') Object.assign(ctl, { width: view.pxW, height: view.pxH, interacting: false })
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
    ctl.hover = [Math.floor((x / cols) * ctl.width), Math.floor((y / rows) * ctl.height)]
  } else if (ev.type === 'leave') {
    ctl.hover = null
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
  for (const k of keys) if (input[k] !== undefined) ctl[k] = input[k]
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
  return { result: JSON.stringify({ snapshot: done ? png : null, ...(problem ? { error: problem } : {}), entry: entry?.id, control: brief, state: briefState }) }
}

/** /sciviz-view and /sciviz-plot: the output row about to be drawn becomes the live view. */
async function openView($: any, e: any, kind: Kind, field: string, options: string[]) {
  const arg = String(e.args ?? '').trim()
  await checkDeno($)
  if (setupError) return { text: `sciviz: ${setupError}` }
  if (options.includes(arg)) KINDS[kind].ctl[field] = arg
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

/** /sciviz-open <entry id>: the gallery entry as a live view in the output row. */
async function openEntry($: any, e: any) {
  const id = String(e.args ?? '').trim()
  if (SECTIONS.includes(id)) // a section on its own: list it
    return { text: `${id}: ${(await entries($)).filter((x) => x.section === id).map((x) => x.id.split('/')[1]).join('  ')}` }
  let found = findEntries(await entries($), id)
  if (!found.length) found = findEntries(await entries($, true), id) // maybe added since the mod loaded
  if (found.length !== 1) {
    return { text: found.length
      ? `sciviz: "${id}" matches ${found.map((x) => x.id).join(', ')}; which one?`
      : `sciviz: no gallery entry "${id}". /sciviz-gallery [${SECTIONS.join('|')}] lists them.` }
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
  // per-session state: two Claude sessions never share controls or frames
  const session = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  RUN = `${home}/.cache/terminal-sciviz/sessions/${session}`
  GALLERY = `${ROOT}/docs`
}

export const register: Register = (on, options) => {
  on('command.run', { command: 'sciviz-view' }, ($, e) => {
    const arg = String((e as any).args ?? '').trim()
    return arg && !SURFACES.includes(arg) ? openEntry($, e) : openView($, e, 'surface', 'surface', SURFACES)
  })
  on('command.run', { command: 'sciviz-plot' }, ($, e) => openView($, e, 'pchip', 'function', FUNCTIONS))
  on('command.run', { command: 'sciviz-open' }, ($, e) => openEntry($, e))
  on('command.run', { command: 'sciviz-gallery' }, async ($, e) => {
    const section = String((e as any).args ?? '').trim() || 'lines'
    if (!SECTIONS.includes(section)) return openEntry($, e) // an entry id: open it, as /sciviz-open does
    const ids = (await entries($)).filter((x) => x.section === section).map((x) => x.id)
    return { text: `${section}: ${ids.join('  ')}` }
  })

  on('ui.render', { component: 'CommandOutput' }, async ($, e, next) => {
    const command = (e.props as any).command
    if (e.surface !== 'terminal') return next(e)
    if (command === 'sciviz-gallery' && SECTIONS.includes(String((e.props as any).args ?? '').trim() || 'lines')) { // the section's contact sheet, then its entry ids
      const { Box, Image, Text } = $.ui.resolve(e) as any
      const section = String((e.props as any).args ?? '').trim() || 'lines'
      const png = `${ROOT}/docs/contact_${section}.png`
      if (!SECTIONS.includes(section) || !(await $.fs.exists(png))) return next(e)
      const cols = Math.max(20, (e.viewport?.columns ?? 100) - RESULT_INDENT)
      const { base64 } = await $.fs.read(png, { as: 'bytes' })
      const hd = new DataView(decodeBase64(base64.slice(0, 32)).buffer)
      const rows = Math.min(255, Math.round((cols * hd.getUint32(20)) / hd.getUint32(16) / CELL_ASPECT))
      return (
        <Box flexDirection="column">
          <Image source={{ file: png, format: 'png' }} columns={cols} rows={rows} alt={`sciviz gallery: ${section}`} />
          <Text dimColor>{String((e.props as any).text ?? '')}  ·  /sciviz-open {'<id>'}</Text>
        </Box>
      )
    }
    if (command !== 'sciviz-view' && command !== 'sciviz-plot' && command !== 'sciviz-open' && command !== 'sciviz-gallery') return next(e)
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
    const kind = kindOf[id] ?? (command === 'sciviz-view' ? 'surface' : command === 'sciviz-plot' ? 'pchip' : undefined)
    if (!kind) return next(e)
    return drawView($, e, id, kind)
  })

  on('ui.message', async ($, e, next) => {
    const kind = liveKind()
    if (e.requestId !== view.live || !kind) return next(e)
    const data = e.data as any
    for (const ev of data?.type === 'batch' && Array.isArray(data.events) ? data.events : [data]) {
      if (ev?.type === 'key') await onKey($, kind, String(ev.key))
      else if (ev?.type) await onPointer($, kind, ev)
    }
    return {}
  })

  on('tool.call', { tool: VIEW_TOOL }, async ($, e) =>
    showView($, e, 'surface', ['surface', 'colorBy', 'colormap', 'azimuth', 'elevation', 'zoom', 'knotLines']))

  on('tool.call', { tool: OPEN_TOOL }, async ($, e) => {
    const all = await entries($)
    const want = String((e as any).id ?? '')
    const found = findEntries(all, want)
    if (found.length !== 1) return { deny: `"${want}" matches ${found.length ? found.map((x) => x.id).join(', ') : 'no entry'}; ids: ${all.map((x) => x.id).join(', ')}` }
    const entry = found[0]
    const overrides = ((e as any).control ?? {}) as Ctl
    const size = JSON.stringify(overrides).length
    if (size > 8000) return { deny: `control overrides are ${size} bytes; keep them small (parameters, not data: entries generate their data on the GPU)` }
    return showView($, { ...e, ...overrides }, galleryKind(entry.renderer), Object.keys(overrides), entry.control, entry)
  })

  on('tool.call', { tool: PLOT_TOOL }, async ($, e) =>
    showView($, e, 'pchip', ['function', 'n', 'x0', 'x1', 'ymin', 'ymax', 'showTrue', 'showSamples', 'compare']))

  on('session.end', async ($, e, next) => {
    for (const k of Object.keys(KINDS)) await stopRenderer($, k).catch(() => {})
    if (RUN.includes('/.cache/terminal-sciviz/sessions/')) await $.process.run(['/bin/rm', '-rf', RUN]).catch(() => {})
    return next(e)
  })

  on('session.start', async ($, e, next) => {
    await resolvePaths($, options ?? {})
    await $.command.register({ name: 'sciviz-view', description: 'Live view inline: /sciviz-view [wave|torus|sphere|wave-test] or a gallery entry (e.g. lines/pchip_blocks, or just blocks)' })
    await $.command.register({ name: 'sciviz-plot', description: 'Live GPU PCHIP plot inline: /sciviz-plot [bumps|runge|step|blocks]' })
    await $.command.register({ name: 'sciviz-gallery', description: 'Gallery contact sheet: /sciviz-gallery [lines|curves|complex|volumes], or an entry id to open it' })
    await $.command.register({ name: 'sciviz-open', description: 'Open a gallery entry as a live view: /sciviz-open lines/residual' })
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
          control: { type: 'object', description: 'overrides of the entry control' },
        },
        required: ['id'],
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
          surface: { type: 'string', enum: ['wave', 'torus', 'sphere', 'wave-test'] },
          colorBy: { type: 'string', enum: ['field', 'height', 'field-gradient', 'height-gradient', 'gaussian-curvature', 'mean-curvature'] },
          colormap: { type: 'string', enum: ['viridis', 'magma', 'inferno', 'plasma'] },
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
    return { result: JSON.stringify(shown) }
  })

  on('ui.render', { component: 'ToolResult' }, async ($, e, next) => {
    if ((e.props.tool === VIEW_TOOL || e.props.tool === PLOT_TOOL || e.props.tool === OPEN_TOOL) && !e.props.isErrored && e.surface === 'terminal') {
      const kind = kindOf[e.props.tool_use_id] ?? (e.props.tool === VIEW_TOOL ? 'surface' : e.props.tool === PLOT_TOOL ? 'pchip' : undefined)
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
    const size = cellSize(s, e.viewport)
    return (
      <Box flexDirection="column">
        <Image
          source={{ file: s.path, format: 'png', generation: s.generation }}
          columns={size.columns}
          rows={size.rows}
          alt={s.caption}
        />
        <Text dimColor>{s.caption}</Text>
      </Box>
    )
  })
}
