import { useEffect, useState } from 'react'
import { chatApi, type ChatEvent, type Info } from './api'
import { Markdown } from './Markdown'
import { useAutosize } from './autosize'
import { useFollowScroll, useStickyScroll } from './stickyScroll'

type Stats = Extract<ChatEvent, { type: 'done' }>
type Msg =
  | { role: 'user'; text: string; image?: string }
  | {
      role: 'assistant'
      raw: string
      thinking: boolean
      streaming: boolean
      stats?: Stats
      error?: string
      thinkStart?: number // ms timestamps for the reasoning phase
      thinkEnd?: number
      thinkTokens?: number // stream deltas received while inside <think> (≈ tokens)
    }

/** With thinking on, the prompt opens <think>, so the reply starts as reasoning until </think>. */
function split(m: Extract<Msg, { role: 'assistant' }>) {
  if (!m.thinking) return { reasoning: '', answer: m.raw }
  const i = m.raw.indexOf('</think>')
  if (i < 0) return { reasoning: m.raw, answer: '', inReasoning: true }
  return { reasoning: m.raw.slice(0, i).trim(), answer: m.raw.slice(i + 8).trimStart() }
}

/** Collapsible reasoning block: open and highlighted while thinking, collapsed after </think>. */
export function Reasoning({ text, live, streaming, secs, tokens }: {
  text: string
  live: boolean
  streaming: boolean
  secs: number
  tokens: number
}) {
  // Follow the live reasoning stream unless the reader scrolled up inside it.
  const body = useStickyScroll<HTMLDivElement>([text], live)
  return (
    <details className={`reasoning ${live ? 'live' : ''}`} open={live}>
      <summary>
        <span className="reasoning-title">{live ? (streaming ? 'Thinking' : 'Thinking (stopped)') : 'Thought'}</span>
        <span className="reasoning-meta mono">
          {secs.toFixed(1)}s · {tokens} tokens
        </span>
        {live && streaming && <span className="dots" />}
      </summary>
      <div className="reasoning-body" ref={body}>
        {text}
      </div>
    </details>
  )
}

function Assistant({ m, onContinue }: { m: Extract<Msg, { role: 'assistant' }>; onContinue?: () => void }) {
  const { reasoning, answer, inReasoning } = split(m)
  const secs = m.thinkStart ? ((m.thinkEnd ?? Date.now()) - m.thinkStart) / 1000 : 0
  return (
    <div className="msg assistant">
      {m.thinking && (reasoning || inReasoning) && (
        <Reasoning text={reasoning} live={!!inReasoning} streaming={m.streaming} secs={secs} tokens={m.thinkTokens ?? 0} />
      )}
      {answer && <Markdown text={answer} />}
      {m.streaming && !inReasoning && <span className="cursor" />}
      {m.error && <div className="err">{m.error}</div>}
      {m.stats && (
        <div className="msg-stats mono">
          {m.stats.tokens} tok · {m.stats.tok_per_s.toFixed(1)} tok/s · prefill {m.stats.prefill_s.toFixed(2)}s · ctx{' '}
          {m.stats.context}
          {m.stats.stop === 'length' && <span className="warn"> · stopped at the reply limit</span>}
          {m.stats.stop === 'cancelled' && <span className="warn"> · stopped</span>}
        </div>
      )}
      {onContinue && (
        <button className="continue-btn" onClick={onContinue}>
          Continue generating ▸
        </button>
      )}
    </div>
  )
}

export default function Chat({ info }: { info: Info }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [image, setImage] = useState<string | null>(null)
  const [system, setSystem] = useState('')
  const [thinking, setThinking] = useState(true)
  const [effort, setEffort] = useState<'xhigh' | 'medium' | 'low'>('medium')
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(32768)
  useEffect(() => {
    const limit = info.context_limit
    if (limit) setMaxTokens((current) => current > limit ? Math.max(2048, 2 ** Math.floor(Math.log2(limit / 2))) : current)
  }, [info.context_limit])
  const [streaming, setStreaming] = useState(false)
  const [fresh, setFresh] = useState(true) // server-side conversation must be (re)created before next send
  const [err, setErr] = useState<string | null>(null)
  const chatScroll = useFollowScroll<HTMLDivElement>([msgs])
  const inputRef = useAutosize(input)

  const patchLast = (f: (m: Extract<Msg, { role: 'assistant' }>) => Partial<Extract<Msg, { role: 'assistant' }>>) =>
    setMsgs((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      if (last?.role === 'assistant') next[next.length - 1] = { ...last, ...f(last) }
      return next
    })

  /** Stream handler; when resuming, keep the message's reasoning timer and token count. */
  const onEvent = (resuming: boolean) => (e: ChatEvent) => {
    if (e.type === 'start') {
      if (!resuming) patchLast(() => ({ thinking: e.thinking, thinkStart: Date.now(), thinkTokens: 0 }))
    } else if (e.type === 'delta')
      patchLast((m) => {
        const raw = m.raw + e.text
        if (!m.thinking || m.thinkEnd) return { raw }
        const ended = raw.includes('</think>')
        return { raw, thinkTokens: (m.thinkTokens ?? 0) + (ended ? 0 : 1), thinkEnd: ended ? Date.now() : undefined }
      })
    else if (e.type === 'done') patchLast(() => ({ stats: e }))
    else if (e.type === 'error') patchLast(() => ({ error: e.message }))
  }

  const resume = async () => {
    if (streaming || !info.ready) return
    setErr(null)
    setStreaming(true)
    chatScroll.jumpToEnd()
    patchLast(() => ({ streaming: true, stats: undefined, error: undefined }))
    try {
      await chatApi.resume({ max_tokens: maxTokens }, onEvent(true))
    } catch (e) {
      patchLast(() => ({ error: String((e as Error).message ?? e) }))
    } finally {
      patchLast(() => ({ streaming: false }))
      setStreaming(false)
    }
  }

  const send = async () => {
    const text = input.trim()
    if (!text || streaming || !info.ready) return
    setErr(null)
    setStreaming(true)
    chatScroll.jumpToEnd()
    const img = image
    setInput('')
    setImage(null)
    setMsgs((p) => [
      ...p,
      { role: 'user', text, image: img ?? undefined },
      { role: 'assistant', raw: '', thinking, streaming: true },
    ])
    try {
      if (fresh) {
        await chatApi.create({ system: system || undefined, thinking, reasoning_effort: effort, temperature })
        setFresh(false)
      }
      await chatApi.send(
        { text, image_b64: img ?? undefined, thinking, reasoning_effort: effort, temperature, max_tokens: maxTokens },
        onEvent(false),
      )
    } catch (e) {
      patchLast(() => ({ error: String((e as Error).message ?? e) }))
    } finally {
      patchLast(() => ({ streaming: false }))
      setStreaming(false)
    }
  }

  const newChat = async () => {
    if (streaming) await chatApi.stop()
    setMsgs([])
    setFresh(true)
    setErr(null)
  }

  return (
    <div className="chat">
      <div className="chat-settings">
        <input
          className="sys"
          placeholder="System prompt (applies to a new chat)"
          value={system}
          onChange={(e) => {
            setSystem(e.target.value)
            if (!msgs.length) setFresh(true)
          }}
        />
        <label className="check">
          <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
          thinking
        </label>
        {thinking && (
          <label className="check">
            effort
            <select value={effort} onChange={(e) => setEffort(e.target.value as typeof effort)}>
              <option value="medium">medium</option>
              <option value="low">low</option>
              <option value="xhigh">xhigh</option>
            </select>
          </label>
        )}
        <label className="check" title="Output cap per reply (thinking + answer). A reply cut off here can be continued.">
          max reply
          <select value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))}>
            {[2048, 4096, 8192, 16384, 32768, 65536].filter((n) => n <= (info.context_limit ?? Infinity)).map((n) => (
              <option key={n} value={n}>
                {n >= 1024 ? `${n / 1024}k` : n} tokens
              </option>
            ))}
          </select>
        </label>
        <label className="temp">
          T
          <input
            type="number"
            step={0.1}
            min={0}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
          />
        </label>
        <button onClick={newChat} disabled={!msgs.length && fresh}>
          New chat
        </button>
      </div>

      <div
        className="chat-scroll"
        ref={chatScroll.ref}
      >
        <div className="chat-col">
          {!msgs.length && (
            <div className="chat-empty">
              <div className="logo big-logo">◆</div>
              <div>Chat with Ternary Bonsai 2 27B. It runs locally on {info.backend === 'llama' ? 'llama.cpp + CUDA' : info.backend === 'cuda' ? 'CUDA' : 'MLX'}.</div>
              <div className="dim">Markdown, code and images are supported. Enter sends, Shift+Enter adds a new line.</div>
            </div>
          )}
          {msgs.map((m, i) =>
            m.role === 'user' ? (
              <div key={i} className="msg user">
                {m.image && <img src={m.image} className="msg-img" alt="" />}
                <div className="bubble">{m.text}</div>
              </div>
            ) : (
              <Assistant
                key={i}
                m={m}
                onContinue={i === msgs.length - 1 && !streaming && m.stats?.resumable ? resume : undefined}
              />
            ),
          )}
        </div>
      </div>

      <div className="composer-wrap">
          {!chatScroll.atEnd && (
            <button className="jump-latest" onClick={chatScroll.jumpToEnd} title="Scroll to the latest message and follow it">
              ↓ Latest
            </button>
          )}
        <div className="composer">
          {image && (
            <div className="composer-img">
              <img src={image} alt="" />
              <button className="link" onClick={() => setImage(null)}>
                ✕
              </button>
            </div>
          )}
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            placeholder={info.ready ? 'Message Bonsai…' : 'Waiting for the model to load…'}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                send()
              }
            }}
          />
          <div className="composer-actions">
            <label className="file icon" title="Attach image">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (!f) return
                  const rd = new FileReader()
                  rd.onload = () => setImage(rd.result as string)
                  rd.readAsDataURL(f)
                  e.target.value = ''
                }}
              />
              ＋
            </label>
            {streaming ? (
              <button className="primary" onClick={() => chatApi.stop()}>
                ■ Stop
              </button>
            ) : (
              <button className="primary" disabled={!input.trim() || !info.ready} onClick={send}>
                Send
              </button>
            )}
          </div>
        </div>
        {err && <div className="err">{err}</div>}
      </div>
    </div>
  )
}
