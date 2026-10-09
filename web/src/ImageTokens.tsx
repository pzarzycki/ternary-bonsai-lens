import { useEffect, useMemo, useRef, useState } from 'react'
import { showTok } from './api'

type ImgTok = {
  seq: number
  row: number
  col: number
  mrope: [number, number, number]
  norm: number
  in_top: [number, number][]
  in_entropy: number
  out_top?: [number, number][]
  out_entropy?: number
  out_norm?: number
}

type ImgInfo = {
  index: number
  grid_thw: [number, number, number]
  patch: number
  merge: number
  rows: number
  cols: number
  width: number
  height: number
  n_tokens: number
  n_patches: number
  patch_dim: number
  span: [number, number]
  text_embed_norm: number
  has_final: boolean
  vocab: Record<string, string>
  tokens: ImgTok[]
}

type Overlay = 'none' | 'norm' | 'in_entropy' | 'out_p' | 'out_entropy'

const OVERLAYS: { key: Overlay; label: string; needsFinal?: boolean }[] = [
  { key: 'none', label: 'grid only' },
  { key: 'norm', label: '‖embedding‖' },
  { key: 'in_entropy', label: 'input-lens entropy' },
  { key: 'out_p', label: 'final-layer top-1 p', needsFinal: true },
  { key: 'out_entropy', label: 'final-layer entropy', needsFinal: true },
]

function metric(t: ImgTok, o: Overlay): number {
  switch (o) {
    case 'norm':
      return t.norm
    case 'in_entropy':
      return t.in_entropy
    case 'out_p':
      return t.out_top?.[0][1] ?? 0
    case 'out_entropy':
      return t.out_entropy ?? 0
    default:
      return 0
  }
}

export default function ImageTokens({ n, onClose }: { n: number; onClose: () => void }) {
  const [info, setInfo] = useState<ImgInfo | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [overlay, setOverlay] = useState<Overlay>('none')
  const [sel, setSel] = useState(0)
  const [pinned, setPinned] = useState(false)
  const img = useRef<HTMLImageElement>(null)
  const crop = useRef<HTMLCanvasElement>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    fetch(`/api/image/${n}/info`)
      .then(async (r) => (r.ok ? r.json() : Promise.reject(new Error((await r.json()).detail))))
      .then(setInfo)
      .catch((e) => setErr(String(e.message ?? e)))
  }, [n])

  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const range = useMemo(() => {
    if (!info || overlay === 'none') return [0, 1]
    const v = info.tokens.map((t) => metric(t, overlay))
    return [Math.min(...v), Math.max(...v)]
  }, [info, overlay])

  const t = info?.tokens[sel]
  const cell = info ? info.patch * info.merge : 32 // pixels covered by one token

  // Zoomed crop of the selected token's 2×2 patch block.
  useEffect(() => {
    const c = crop.current
    if (!c || !info || !t || !loaded || !img.current) return
    const ctx = c.getContext('2d')!
    ctx.imageSmoothingEnabled = false
    ctx.clearRect(0, 0, c.width, c.height)
    ctx.drawImage(img.current, t.col * cell, t.row * cell, cell, cell, 0, 0, c.width, c.height)
    ctx.strokeStyle = 'rgba(255,255,255,.8)'
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(c.width / 2, 0)
    ctx.lineTo(c.width / 2, c.height)
    ctx.moveTo(0, c.height / 2)
    ctx.lineTo(c.width, c.height / 2)
    ctx.stroke()
  }, [info, t, loaded, cell])

  const word = (id: number) => showTok(info?.vocab[String(id)] ?? `#${id}`)

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Image {n + 1}: how it became tokens</h3>
          <button className="link" onClick={onClose}>
            ✕ close
          </button>
        </div>
        {err && <div className="err">{err}</div>}
        {!info && !err && <div className="dim">Running the vision tower and lm_head over every image token…</div>}
        {info && (
          <>
            <ol className="pipeline">
              <li>
                <b>Resize</b> to {info.width}×{info.height} px (both sides multiples of {cell}).
              </li>
              <li>
                <b>Patchify</b> into {info.grid_thw[1]}×{info.grid_thw[2]} = {info.n_patches} patches of {info.patch}×{info.patch}{' '}
                px. Each patch is 3 ch × 2 frames × {info.patch}² ={' '}
                {info.patch_dim} values; a still image is duplicated across the 2 frames.
              </li>
              <li>
                <b>Vision tower</b> (27-block ViT). Its merger joins each {info.merge}×{info.merge} block of patches and projects it to
                5120 dims, giving{' '}
                <b>
                  {info.rows}×{info.cols} = {info.n_tokens} tokens
                </b>
                .
              </li>
              <li>
                <b>Scatter</b> into the text sequence. Positions {info.span[0] + 1}–{info.span[1] - 1} hold the id{' '}
                <code>&lt;|image_pad|&gt;</code>. That id only marks a slot: the model replaces its text embedding with the
                vision vector.
              </li>
              <li>
                <b>M-RoPE</b> gives each image token a 3D position (t, h, w), so {info.n_tokens} tokens span only{' '}
                {Math.max(info.rows, info.cols)} position steps. Text after the image continues from the max + 1.
              </li>
            </ol>

            <div className="img-layout">
              <div className="img-left">
                <div className="seg small overlay-seg">
                  {OVERLAYS.map((o) => (
                    <button
                      key={o.key}
                      disabled={o.needsFinal && !info.has_final}
                      className={overlay === o.key ? 'on' : ''}
                      onClick={() => setOverlay(o.key)}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
                <div className="img-stage" style={{ aspectRatio: `${info.width} / ${info.height}` }}>
                  <img ref={img} src={`/api/image/${n}.png`} alt="" onLoad={() => setLoaded(true)} />
                  <div
                    className="img-grid"
                    style={{ gridTemplateColumns: `repeat(${info.cols}, 1fr)`, gridTemplateRows: `repeat(${info.rows}, 1fr)` }}
                  >
                    {info.tokens.map((tk, j) => {
                      const v = overlay === 'none' ? 0 : (metric(tk, overlay) - range[0]) / (range[1] - range[0] || 1)
                      return (
                        <div
                          key={j}
                          className={`img-cell ${j === sel ? 'sel' : ''}`}
                          style={overlay === 'none' ? undefined : { background: `rgba(255, 90, 40, ${0.08 + v * 0.72})` }}
                          onMouseEnter={() => !pinned && setSel(j)}
                          onClick={() => {
                            setSel(j)
                            setPinned((p) => !(p && j === sel))
                          }}
                        />
                      )
                    })}
                  </div>
                </div>
                <div className="dim small-note">
                  Hover over a cell to inspect its token, or click to pin it. Each cell is one token covering {cell}×{cell} px.
                  {overlay !== 'none' && ` Overlay range ${range[0].toFixed(3)} … ${range[1].toFixed(3)}.`}
                </div>
              </div>

              {t && (
                <div className="img-right">
                  <div className="row between">
                    <label className="lbl">
                      Token {sel + 1} / {info.n_tokens} {pinned && <span className="badge pin">pinned</span>}
                    </label>
                  </div>
                  <div className="tok-detail">
                    <canvas ref={crop} width={128} height={128} className="crop" />
                    <div className="kv2">
                      <span>sequence pos</span>
                      <b className="mono">{t.seq}</b>
                      <span>grid (row, col)</span>
                      <b className="mono">
                        ({t.row}, {t.col})
                      </b>
                      <span>pixels</span>
                      <b className="mono">
                        x {t.col * cell}–{(t.col + 1) * cell}, y {t.row * cell}–{(t.row + 1) * cell}
                      </b>
                      <span>M-RoPE (t, h, w)</span>
                      <b className="mono">({t.mrope.join(', ')})</b>
                      <span>‖vision emb‖</span>
                      <b className="mono">
                        {t.norm.toFixed(1)} <span className="dim">({(t.norm / info.text_embed_norm).toFixed(0)}× avg text emb)</span>
                      </b>
                      {t.out_norm !== undefined && (
                        <>
                          <span>‖final hidden‖</span>
                          <b className="mono">{t.out_norm.toFixed(1)}</b>
                        </>
                      )}
                    </div>
                  </div>

                  <LensList
                    title="Input lens: lm_head(norm(vision embedding))"
                    hint={`entropy ${t.in_entropy.toFixed(2)} nats. A heuristic for which vocabulary direction the raw image vector points towards.`}
                    top={t.in_top}
                    word={word}
                  />
                  {t.out_top ? (
                    <LensList
                      title="Final layer at this position: lm_head(h)"
                      hint={`entropy ${t.out_entropy!.toFixed(2)} nats. What the model would predict as the next token after this image token.`}
                      top={t.out_top}
                      word={word}
                    />
                  ) : (
                    <div className="dim small-note">Press Step once (the prefill) to also get final-layer readouts per image token.</div>
                  )}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function LensList({
  title,
  hint,
  top,
  word,
}: {
  title: string
  hint: string
  top: [number, number][]
  word: (id: number) => string
}) {
  const m = top[0]?.[1] || 1
  return (
    <div className="lens-list">
      <label className="lbl sm">{title}</label>
      {top.map(([id, p]) => (
        <div key={id} className="lens-row">
          <span className="chip mono">{word(id)}</span>
          <div className="dim-bar-wrap">
            <div className="dim-bar pos" style={{ width: `${(p / m) * 100}%` }} />
          </div>
          <span className="mono dim-v">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
      <div className="dim small-note">{hint}</div>
    </div>
  )
}
