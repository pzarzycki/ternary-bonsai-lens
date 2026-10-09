import { useEffect, useMemo, useRef, useState } from 'react'
import { showTok, type Readout, type Tok, type VStats } from './api'

const fmt = (x: number, d = 3) => (Math.abs(x) >= 1000 ? x.toExponential(2) : x.toFixed(d))

type RGB = [number, number, number]
const hex = (h: string): RGB => {
  const n = parseInt(h.trim().replace('#', ''), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** Heatmap colors from the theme tokens (--heat-neg / --heat-mid / --heat-pos). */
function useHeatColors(): { neg: RGB; mid: RGB; pos: RGB } {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme)
  useEffect(() => {
    const el = document.documentElement
    const obs = new MutationObserver(() => setTheme(el.dataset.theme))
    obs.observe(el, { attributes: true, attributeFilter: ['data-theme'] })
    return () => obs.disconnect()
  }, [])
  return useMemo(() => {
    const css = getComputedStyle(document.documentElement)
    const get = (k: string, fb: string) => hex(css.getPropertyValue(k) || fb)
    return { neg: get('--heat-neg', '#3b82f6'), mid: get('--heat-mid', '#ffffff'), pos: get('--heat-pos', '#f43f5e') }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme])
}

/** Diverging neg–mid–pos, symmetric around 0, clipped at ±scale. The midpoint is the card color. */
function diverge(v: number, scale: number, c: { neg: RGB; mid: RGB; pos: RGB }): RGB {
  const t = Math.max(-1, Math.min(1, v / scale))
  const a = Math.abs(t)
  const end = t >= 0 ? c.pos : c.neg
  return [0, 1, 2].map((k) => Math.round(c.mid[k] + (end[k] - c.mid[k]) * a)) as RGB
}

export function Heatmap({ values, cols = 80 }: { values: number[]; cols?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<number | null>(null)
  const rows = Math.ceil(values.length / cols)
  const colors = useHeatColors()
  // Robust scale: 99th percentile of |v| so a few massive outlier dims don't wash everything out.
  const scale = useMemo(() => {
    const s = values.map(Math.abs).sort((a, b) => a - b)
    return s[Math.floor(s.length * 0.99)] || 1
  }, [values])

  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.width = cols
    c.height = rows
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    values.forEach((v, i) => {
      const [r, g, b] = diverge(v, scale, colors)
      img.data.set([r, g, b, 255], i * 4)
    })
    ctx.putImageData(img, 0, 0)
  }, [values, cols, rows, scale, colors])

  return (
    <div className="heatmap">
      <canvas
        ref={ref}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const x = Math.floor(((e.clientX - r.left) / r.width) * cols)
          const y = Math.floor(((e.clientY - r.top) / r.height) * rows)
          const i = y * cols + x
          setHover(i < values.length ? i : null)
        }}
        onMouseLeave={() => setHover(null)}
      />
      <div className="heatmap-caption">
        <span>
          {cols}×{rows} grid, row-major · color clip ±{fmt(scale)} (p99 |v|)
        </span>
        <span className="mono">{hover !== null ? `dim ${hover} = ${fmt(values[hover], 4)}` : 'hover for value'}</span>
      </div>
    </div>
  )
}

export function Stats({ s }: { s: VStats }) {
  return (
    <div className="stats">
      {(
        [
          ['‖v‖₂', s.l2],
          ['mean', s.mean],
          ['std', s.std],
          ['min', s.min],
          ['max', s.max],
        ] as const
      ).map(([k, v]) => (
        <div key={k} className="stat">
          <span className="stat-k">{k}</span>
          <span className="stat-v mono">{fmt(v)}</span>
        </div>
      ))}
      <div className="stat">
        <span className="stat-k">argmax |v|</span>
        <span className="stat-v mono">dim {s.argmax_abs}</span>
      </div>
    </div>
  )
}

export function TopDims({ values, n = 12 }: { values: number[]; n?: number }) {
  const top = useMemo(
    () =>
      values
        .map((v, i) => [i, v] as const)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, n),
    [values, n],
  )
  const m = Math.abs(top[0]?.[1] ?? 1)
  return (
    <div className="topdims">
      {top.map(([i, v]) => (
        <div key={i} className="topdim">
          <span className="mono dim-i">{i}</span>
          <div className="dim-bar-wrap">
            <div
              className={`dim-bar ${v >= 0 ? 'pos' : 'neg'}`}
              style={{ width: `${(Math.abs(v) / m) * 100}%` }}
            />
          </div>
          <span className="mono dim-v">{fmt(v)}</span>
        </div>
      ))}
    </div>
  )
}

export function Histogram({ counts, edges, marks }: { counts: number[]; edges: number[]; marks: { x: number; label: string }[] }) {
  const W = 360,
    H = 90
  const lo = edges[0],
    hi = edges[edges.length - 1]
  const logc = counts.map((c) => Math.log10(c + 1))
  const max = Math.max(...logc)
  const bw = W / counts.length
  const X = (x: number) => ((x - lo) / (hi - lo)) * W
  return (
    <svg viewBox={`0 0 ${W} ${H + 16}`} className="hist">
      {logc.map((c, i) => (
        <rect key={i} x={i * bw} y={H - (c / max) * H} width={bw - 0.5} height={(c / max) * H} className="hist-bar" />
      ))}
      {marks.map((m) => (
        <g key={m.label}>
          <line x1={X(m.x)} x2={X(m.x)} y1={0} y2={H} className="hist-mark" />
          <text x={Math.min(X(m.x) + 3, W - 40)} y={10} className="hist-label">
            {m.label}
          </text>
        </g>
      ))}
      <text x={0} y={H + 13} className="hist-axis">
        {fmt(lo, 1)}
      </text>
      <text x={W} y={H + 13} textAnchor="end" className="hist-axis">
        {fmt(hi, 1)}
      </text>
    </svg>
  )
}

export function TopK({
  r,
  chosenId,
  onPick,
}: {
  r: Readout
  chosenId: number
  onPick?: (t: Tok) => void
}) {
  const pmax = r.top_probs[0] || 1
  return (
    <table className="topk">
      <thead>
        <tr>
          <th>#</th>
          <th>token</th>
          <th>id</th>
          <th className="num">logit</th>
          <th className="num">p</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {r.top.map((t, i) => (
          <tr
            key={t.id}
            className={`${t.id === chosenId ? 'chosen' : ''} ${onPick ? 'pickable' : ''}`}
            onClick={onPick ? () => onPick(t) : undefined}
            title={onPick ? 'Use this token instead' : `raw piece: ${t.piece}`}
          >
            <td className="mono dim">{i + 1}</td>
            <td>
              <span className="chip mono">{showTok(t.text)}</span>
            </td>
            <td className="mono dim">{t.id}</td>
            <td className="mono num">{r.top_logits[i].toFixed(2)}</td>
            <td className="mono num">{(r.top_probs[i] * 100).toFixed(2)}%</td>
            <td className="pbar-cell">
              <div className="pbar" style={{ width: `${(r.top_probs[i] / pmax) * 100}%` }} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function cosine(a: number[], b: number[]) {
  let d = 0,
    na = 0,
    nb = 0
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return d / Math.sqrt(na * nb)
}
