// Transparent input layer over the view: forwards pointer and key events to the hooks module.
// `post` keeps only the latest undelivered message per frame, so events are queued and sent as one
// batch per frame (consecutive moves coalesced), never dropping a down/up/key.
export default function Input(props: any, s: any) {
  if (s.state === undefined) {
    let queue: any[] = []
    const push = (ev: any) => {
      const last = queue[queue.length - 1]
      if (ev.type === 'move' && last?.type === 'move' && !!last.button === !!ev.button) queue[queue.length - 1] = ev
      else queue.push(ev)
    }
    // a strip of its own (a slider below the main region) says where it sits in the view
    const off = Array.isArray(props?.off) ? props.off : undefined
    s.onPointer((ev: any) => push({ ...ev, columns: s.columns, rows: s.rows, off }))
    s.onKey((ev: any) => push({ type: 'key', ...ev }))
    s.every(16, () => {
      if (!queue.length) return
      s.post({ type: 'batch', events: queue })
      queue = []
    })
    s.setState(1)
  }
  return s.elements.Box({})
}
