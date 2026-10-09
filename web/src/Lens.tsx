import { useCallback, useEffect, useState } from 'react'
import { api, showTok, type Info, type StepResult, type Tok } from './api'
import ImageTokens from './ImageTokens'
import { cosine, Heatmap, Histogram, Stats, TopDims, TopK } from './viz'

type Vec = 'post' | 'pre'

export default function Lens({ info }: { info: Info }) {
  const [prompt, setPrompt] = useState('What is the capital of France? Answer in one word.')
  const [mode, setMode] = useState<'chat' | 'raw'>('chat')
  const [thinking, setThinking] = useState(false)
  const [effort, setEffort] = useState<'xhigh' | 'medium' | 'low'>('medium')
  const [system, setSystem] = useState('')
  const [image, setImage] = useState<string | null>(null)
  const [strategy, setStrategy] = useState<'greedy' | 'sample'>('greedy')
  const [temperature, setTemperature] = useState(1.0)

  const [promptToks, setPromptToks] = useState<Tok[] | null>(null)
  const [rendered, setRendered] = useState('')
  const [steps, setSteps] = useState<StepResult[]>([])
  const [sel, setSel] = useState<number | null>(null)
  const [vec, setVec] = useState<Vec>('post')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [showRendered, setShowRendered] = useState(false)
  const [imgOpen, setImgOpen] = useState<number | null>(null)

  const ended = promptToks !== null && steps.length > 0 && steps[steps.length - 1].is_eos

  const guard = async (fn: () => Promise<void>) => {
    setBusy(true)
    setErr(null)
    try {
      await fn()
    } catch (e) {
      setErr(String((e as Error).message ?? e))
    } finally {
      setBusy(false)
    }
  }

  const reset = () =>
    guard(async () => {
      const r = await api.reset({
        prompt,
        mode,
        thinking,
        reasoning_effort: effort,
        system: system || undefined,
        image_b64: image ?? undefined,
      })
      setPromptToks(r.tokens)
      setRendered(r.rendered)
      setSteps([])
      setSel(null)
    })

  const step = useCallback(
    () =>
      guard(async () => {
        if (!promptToks) {
          const r = await api.reset({
        prompt,
        mode,
        thinking,
        reasoning_effort: effort,
        system: system || undefined,
        image_b64: image ?? undefined,
      })
          setPromptToks(r.tokens)
          setRendered(r.rendered)
          setSteps([])
        }
        const s = await api.step({ strategy, temperature, top_p: 0.95, top_k: 20, k: 50 })
        setSteps((prev) => {
          setSel(prev.length)
          return [...prev, s]
        })
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [promptToks, prompt, mode, thinking, effort, system, image, strategy, temperature],
  )

  const pick = (t: Tok) =>
    guard(async () => {
      const r = await api.replace(t.id)
      setSteps((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        const i = last.readout.top.findIndex((x) => x.id === t.id)
        next[next.length - 1] = {
          ...last,
          chosen: { ...r.chosen, logit: last.readout.top_logits[i] },
          is_eos: r.is_eos,
        }
        return next
      })
    })

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !busy && info.ready) {
        e.preventDefault()
        step()
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [step, busy, info.ready, ended])

  const dirty = () => setPromptToks(null)
  const cur = sel !== null ? steps[sel] : undefined
  const prev = sel !== null && sel > 0 ? steps[sel - 1] : undefined
  const isLatest = sel !== null && sel === steps.length - 1
  const v = cur ? (vec === 'post' ? cur.readout.post_norm : cur.readout.pre_norm) : null
  const vs = cur ? (vec === 'post' ? cur.readout.post_stats : cur.readout.pre_stats) : null
  const cos =
    cur && prev
      ? cosine(
          vec === 'post' ? cur.readout.post_norm : cur.readout.pre_norm,
          vec === 'post' ? prev.readout.post_norm : prev.readout.pre_norm,
        )
      : null

  return (
      <main>
        <section className="left">
          <div className="card">
            <label className="lbl">Prompt</label>
            <textarea
              value={prompt}
              rows={5}
              onChange={(e) => {
                setPrompt(e.target.value)
                dirty()
              }}
            />
            <div className="row">
              <div className="seg">
                {(['chat', 'raw'] as const).map((m) => (
                  <button
                    key={m}
                    className={mode === m ? 'on' : ''}
                    onClick={() => {
                      setMode(m)
                      dirty()
                    }}
                  >
                    {m === 'chat' ? 'Chat template' : 'Raw text'}
                  </button>
                ))}
              </div>
              {mode === 'chat' && (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={thinking}
                    onChange={(e) => {
                      setThinking(e.target.checked)
                      dirty()
                    }}
                  />
                  thinking
                </label>
              )}
              {mode === 'chat' && thinking && (
                <label className="check" title="Chat-template reasoning_effort. xhigh/low inject a system instruction; medium injects none.">
                  effort
                  <select
                    value={effort}
                    onChange={(e) => {
                      setEffort(e.target.value as typeof effort)
                      dirty()
                    }}
                  >
                    <option value="medium">medium (no system text)</option>
                    <option value="low">low</option>
                    <option value="xhigh">xhigh (template default)</option>
                  </select>
                </label>
              )}
            </div>
            {mode === 'chat' && (
              <input
                className="sys"
                placeholder="System prompt (optional)"
                value={system}
                onChange={(e) => {
                  setSystem(e.target.value)
                  dirty()
                }}
              />
            )}
            <div className="row">
              <label className="file">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (!f) return
                    const rd = new FileReader()
                    rd.onload = () => {
                      setImage(rd.result as string)
                      dirty()
                    }
                    rd.readAsDataURL(f)
                  }}
                />
                {image ? 'Change image' : '+ Image'}
              </label>
              {image && (
                <>
                  <img src={image} className="thumb" alt="" />
                  <button
                    className="link"
                    onClick={() => {
                      setImage(null)
                      dirty()
                    }}
                  >
                    remove
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="card">
            <label className="lbl">Decoding</label>
            <div className="row">
              <div className="seg">
                {(['greedy', 'sample'] as const).map((s) => (
                  <button key={s} className={strategy === s ? 'on' : ''} onClick={() => setStrategy(s)}>
                    {s}
                  </button>
                ))}
              </div>
              {strategy === 'sample' && (
                <label className="temp">
                  T
                  <input
                    type="number"
                    step={0.1}
                    min={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value) || 1)}
                  />
                  <span className="dim">top-p .95 · top-k 20</span>
                </label>
              )}
            </div>
            <div className="row actions">
              <button className="primary" disabled={!info.ready || busy || ended} onClick={step}>
                {busy ? 'Running…' : 'Step ▸ 1 token'}
              </button>
              <button disabled={!info.ready || busy} onClick={reset}>
                Reset
              </button>
              <span className="dim kbd">⌘/Ctrl + Enter</span>
            </div>
            {ended && (
              <div className="dim ended">
                Generation ended at EOS. Pick another candidate in the latest step to continue, or edit the prompt / Reset.
              </div>
            )}
            {err && <div className="err">{err}</div>}
          </div>

          <div className="card grow">
            <div className="row between">
              <label className="lbl">
                Sequence{' '}
                {promptToks && (
                  <span className="dim">
                    · {promptToks.length} prompt + {steps.length} generated
                  </span>
                )}
              </label>
              {rendered && (
                <button className="link" onClick={() => setShowRendered((x) => !x)}>
                  {showRendered ? 'tokens' : 'rendered text'}
                </button>
              )}
            </div>
            {!promptToks && <div className="dim empty">Press Step to tokenize the prompt and generate the first token.</div>}
            {promptToks && showRendered && <pre className="rendered">{rendered}</pre>}
            {promptToks && !showRendered && (
              <div className="tokens">
                {groupVision(promptToks).map((g) =>
                  g.kind === 'image' ? (
                    <span
                      key={`img${g.start}`}
                      className="chip image-chip"
                      onClick={info.image_token_readouts === false ? undefined : () => setImgOpen(g.n)}
                      aria-disabled={info.image_token_readouts === false}
                      title={info.image_token_readouts === false ? 'Native image prompt: final-position readouts are available; individual image-token inspection requires direct CUDA or MLX.' : `positions ${g.start}–${g.end}: <|vision_start|> + ${g.pads} × <|image_pad|> + <|vision_end|>. Click to inspect.`}
                    >
                      🖼 image {g.n + 1}{info.image_token_readouts === false ? '' : ` · ${g.pads} tokens`}
                    </span>
                  ) : (
                    <span key={`p${g.i}`} className="chip prompt mono" title={`#${g.i} id ${g.t.id}`}>
                      {showTok(g.t.text)}
                    </span>
                  ),
                )}
                {steps.map((s, i) => (
                  <span
                    key={`g${i}`}
                    className={`chip gen mono ${sel === i ? 'sel' : ''} ${s.is_eos ? 'eos' : ''}`}
                    onClick={() => setSel(i)}
                    title={`step ${i + 1} · pos ${s.position} · id ${s.chosen.id}`}
                  >
                    {showTok(s.chosen.text)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="right">
          {!cur && (
            <div className="card placeholder">
              <h3>Readouts appear here</h3>
              <p>
                Each <b>Step</b> runs exactly one forward pass. On the first step that is the prompt prefill; after
                that, it's the one token chosen in the previous step. The panels show what the model computed at the{' '}
                <b>last position</b>:
              </p>
              <ul>
                <li>
                  <b>After unembedding</b>: logits = lm_head(h), over {info.vocab_size?.toLocaleString() ?? '248k'} tokens
                </li>
                <li>
                  <b>Before unembedding</b>: h, the {info.hidden_size ?? 5120}-dim output of the final RMSNorm. You can
                  switch to the pre-norm residual stream.
                </li>
              </ul>
            </div>
          )}
          {cur && (
            <>
              <div className="card summary">
                <div>
                  <div className="lbl">Step {sel! + 1}</div>
                  <div className="big">
                    <span className="chip gen mono big-chip">{showTok(cur.chosen.text)}</span>
                    <span className="dim mono">id {cur.chosen.id}</span>
                    {cur.is_eos && <span className="badge">EOS</span>}
                  </div>
                </div>
                <div className="kv">
                  <span>position</span>
                  <b className="mono">{cur.position}</b>
                  <span>fed tokens</span>
                  <b className="mono">{cur.fed_tokens}</b>
                  <span>forward</span>
                  <b className="mono">{cur.forward_ms.toFixed(0)} ms</b>
                  <span>entropy</span>
                  <b className="mono">{cur.readout.entropy.toFixed(3)} nats</b>
                  <span>chosen logit</span>
                  <b className="mono">{cur.chosen.logit.toFixed(2)}</b>
                </div>
                <a className="link" href="/api/last.npz" title="Full logits + pre/post vectors of the latest step">
                  ⬇ npz (latest)
                </a>
              </div>

              <div className="grid2">
                <div className="card">
                  <div className="row between">
                    <label className="lbl">After unembedding · top-50 logits</label>
                    {isLatest && <span className="dim">click a row to choose it instead</span>}
                  </div>
                  <TopK r={cur.readout} chosenId={cur.chosen.id} onPick={isLatest && !busy ? pick : undefined} />
                </div>

                <div className="col">
                  <div className="card">
                    <label className="lbl">Logit distribution · all {info.vocab_size?.toLocaleString()} (log count)</label>
                    <Histogram
                      counts={cur.readout.logit_hist.counts}
                      edges={cur.readout.logit_hist.edges}
                      marks={[{ x: cur.chosen.logit, label: 'chosen' }]}
                    />
                    <div className="stats">
                      {Object.entries(cur.readout.logit_stats).map(([k, x]) => (
                        <div key={k} className="stat">
                          <span className="stat-k">{k}</span>
                          <span className="stat-v mono">{x.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    <div className="row between">
                      <label className="lbl">Before unembedding · h ∈ ℝ^{v!.length}</label>
                      <div className="seg small">
                        <button className={vec === 'post' ? 'on' : ''} onClick={() => setVec('post')}>
                          post-norm (→ lm_head)
                        </button>
                        <button className={vec === 'pre' ? 'on' : ''} onClick={() => setVec('pre')}>
                          pre-norm residual
                        </button>
                      </div>
                    </div>
                    <Stats s={vs!} />
                    {cos !== null && (
                      <div className="dim cos">
                        cosine to previous step: <b className="mono">{cos.toFixed(4)}</b>
                      </div>
                    )}
                    <Heatmap values={v!} />
                    <label className="lbl sm">Largest |dims|</label>
                    <TopDims values={v!} />
                  </div>
                </div>
              </div>
            </>
          )}
        </section>
        {imgOpen !== null && <ImageTokens n={imgOpen} onClose={() => setImgOpen(null)} />}
      </main>
  )
}

const VISION_START = '<|vision_start|>'
const VISION_END = '<|vision_end|>'

type Group = { kind: 'tok'; i: number; t: Tok } | { kind: 'image'; n: number; start: number; end: number; pads: number }

/** Collapse each <|vision_start|> … <|vision_end|> span into one group. */
function groupVision(toks: Tok[]): Group[] {
  const out: Group[] = []
  let n = 0
  for (let i = 0; i < toks.length; i++) {
    if (toks[i].text === VISION_START) {
      let j = i + 1
      while (j < toks.length && toks[j].text !== VISION_END) j++
      out.push({ kind: 'image', n: n++, start: i, end: j, pads: j - i - 1 })
      i = j
    } else out.push({ kind: 'tok', i, t: toks[i] })
  }
  return out
}
