import hljs from 'highlight.js/lib/common'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { codeApi, type CodeEvent, type CodeStatus, type Info } from './api'
import { Reasoning } from './Chat'
import { Markdown } from './Markdown'
import { useAutosize } from './autosize'
import { useFollowScroll, useStickyScroll } from './stickyScroll'

type Result = { ok: boolean; output: string; ms: number }
type Part =
  | { t: 'think'; text: string; start: number; end?: number; toks: number }
  | { t: 'text'; text: string }
  | { t: 'call'; raw: string; done: boolean; id?: string; name?: string | null; args?: Record<string, unknown>; result?: Result }
type Turn = {
  kind: 'turn'
  parts: Part[]
  streaming: boolean
  stats?: { tokens: number; tok_per_s: number; prefill_s: number; context: number }
}
type Item =
  | { kind: 'user'; text: string; queued?: boolean }
  | { kind: 'hook'; event: string; name: string; ok: boolean; blocked: boolean; output: string; image: string | null; n: number }
  | Turn
  | { kind: 'note'; text: string; warn?: boolean }
  | { kind: 'reminder'; text: string }
  | { kind: 'check'; ok: boolean; output: string }
  | { kind: 'browser'; ok: boolean; errors: string[]; failed: string[]; empty?: boolean; canvases?: number; shot: number }
type CallPart = Extract<Part, { t: 'call' }>
type Pane = 'preview' | 'files' | 'plan' | 'terminal' | 'history' | 'trace'
type Checkpoint = { id: string; time: number; message: string }

const STOP_NOTES: Record<string, string> = {
  cancelled: 'Stopped.',
  steps: 'Stopped: reached the step limit for one message. Send "continue" to keep going.',
  context: 'Stopped: the context is full. Reset the session to start over.',
  length: 'Stopped: one model turn hit its token limit.',
}

/** Workspace-relative path (the model sometimes uses /workspace/… or ./…). */
const rel = (p: string) => p.trim().replace(/^\/workspace\/?/, '').replace(/^\.\//, '')
/** Project files only: "follow agent" doesn't open the agent's reference reads (skills, the global AGENTS.md). */
const inWorkspace = (p: string) => !!rel(p) && !rel(p).startsWith('/') && !/(^|\/)SKILL\.md$/.test(rel(p))

/** Parse a (possibly still streaming) <tool_call> body: function name and parameters so far. */
function parseLive(raw: string): { name?: string; params: Record<string, string>; open?: string } {
  const name = /<function=([^>\n]+)>/.exec(raw)?.[1]?.trim()
  const params: Record<string, string> = {}
  let open: string | undefined
  const re = /<parameter=([^>\n]+)>\n?([\s\S]*?)(\n?<\/parameter>|$)/g
  let m
  while ((m = re.exec(raw))) {
    params[m[1].trim()] = m[2]
    open = m[3] ? undefined : m[1].trim()
    if (!m[0]) break
  }
  return { name, params, open }
}

function callInfo(p: CallPart) {
  const live = parseLive(p.raw)
  const name = p.name ?? live.name
  const args = (p.args as Record<string, string> | undefined) ?? live.params
  return { name, args }
}

function summary(name: string | undefined | null, a: Record<string, unknown>): { icon: string; text: string } {
  const s = (k: string) => String(a[k] ?? '')
  switch (name) {
    case 'run_shell_command':
      return { icon: '$', text: s('command') }
    case 'write_file':
      return { icon: '✎', text: `${rel(s('file_path'))}${a.content ? ` · ${String(a.content).split('\n').length} lines` : ''}` }
    case 'edit':
      return { icon: '✎', text: `${rel(s('file_path'))} (edit)` }
    case 'read_file':
      return { icon: '📄', text: rel(s('file_path')) + (a.offset ? ` @${s('offset')}` : '') }
    case 'glob':
      return { icon: '▤', text: `glob ${s('pattern')}` }
    case 'grep_search':
      return { icon: '⌕', text: `grep ${s('pattern')} ${s('path')}` }
    case 'read_process':
      return { icon: '≡', text: `read ${s('process_id')}` }
    case 'kill_process':
      return { icon: '■', text: `kill ${s('process_id')}` }
    case 'update_plan':
      return { icon: '☑', text: 'update plan' }
    default:
      return { icon: '⚙', text: name ?? '…' }
  }
}

const LANG: Record<string, string> = {
  ts: 'typescript', tsx: 'typescript', js: 'javascript', jsx: 'javascript', mjs: 'javascript', cjs: 'javascript',
  json: 'json', css: 'css', scss: 'scss', html: 'xml', svg: 'xml', xml: 'xml', md: 'markdown', py: 'python',
  sh: 'bash', yml: 'yaml', yaml: 'yaml', toml: 'ini',
}

/** Highlighted file view. While the agent is writing the file (`live`), it follows the end like a terminal. */
function CodeView({ path, text, live }: { path: string; text: string; live?: boolean }) {
  const scroller = useStickyScroll<HTMLDivElement>([text], !!live)
  const html = useMemo(() => {
    const lang = LANG[path.split('.').pop()?.toLowerCase() ?? '']
    try {
      return lang ? hljs.highlight(text, { language: lang, ignoreIllegals: true }).value : hljs.highlightAuto(text).value
    } catch {
      return text.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!)
    }
  }, [path, text])
  const lines = text.split('\n').length
  return (
    <div className="code-view" ref={scroller}>
      <pre className="gutter mono">{Array.from({ length: lines }, (_, i) => i + 1).join('\n')}</pre>
      <pre className="hljs mono">
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  )
}

function ToolCard({ p, active }: { p: CallPart; active: boolean }) {
  const { name, args } = callInfo(p)
  const { icon, text } = summary(name, args)
  const status = p.result ? (p.result.ok ? 'ok' : 'fail') : active ? 'run' : 'idle'
  return (
    <details className={`tool-card ${status}`}>
      <summary>
        <span className="tool-icon mono">{icon}</span>
        <span className="tool-text mono">{text}</span>
        <span className="tool-status">
          {status === 'run' && <span className="spinner" />}
          {status === 'ok' && '✓'}
          {status === 'fail' && '✗'}
          {p.result && <span className="dim mono"> {(p.result.ms / 1000).toFixed(1)}s</span>}
        </span>
      </summary>
      {name === 'write_file' && args.content !== undefined && (
        <pre className="tool-out mono">{String(args.content)}</pre>
      )}
      {name === 'edit' && (
        <pre className="tool-out mono">
          <span className="del">{String(args.old_string ?? '')}</span>
          {'\n'}
          <span className="add">{String(args.new_string ?? '')}</span>
        </pre>
      )}
      {name === 'update_plan' && <div className="tool-out"><Markdown text={String(args.plan ?? '')} /></div>}
      {p.result && name !== 'update_plan' && <pre className={`tool-out mono ${p.result.ok ? '' : 'fail'}`}>{p.result.output}</pre>}
      <details className="tool-raw">
        <summary className="dim">raw tool call (as generated)</summary>
        <pre className="tool-out mono">{'<tool_call>' + p.raw + (p.done ? '</tool_call>' : '')}</pre>
      </details>
    </details>
  )
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <div className="msg assistant code-turn">
      {turn.parts.map((p, i) => {
        if (p.t === 'think')
          return (p.text.trim() || !p.end) ? (
            <Reasoning
              key={i}
              text={p.text.trim()}
              live={!p.end}
              streaming={turn.streaming}
              secs={((p.end ?? Date.now()) - p.start) / 1000}
              tokens={p.toks}
            />
          ) : null
        if (p.t === 'text') return p.text.trim() ? <Markdown key={i} text={p.text} /> : null
        return <ToolCard key={i} p={p} active={turn.streaming || !p.result} />
      })}
      {turn.stats && (
        <div className="msg-stats mono">
          {turn.stats.tokens} tok · {turn.stats.tok_per_s.toFixed(1)} tok/s · prefill {turn.stats.prefill_s.toFixed(2)}s · ctx{' '}
          {turn.stats.context.toLocaleString()}
        </div>
      )}
    </div>
  )
}

function FileTree({ files, sel, onSel, live }: { files: string[]; sel: string | null; onSel: (f: string) => void; live?: string }) {
  const rows: { key: string; depth: number; label: string; file?: string }[] = []
  const seen = new Set<string>()
  for (const f of files) {
    const parts = f.split('/')
    for (let d = 0; d < parts.length - 1; d++) {
      const dir = parts.slice(0, d + 1).join('/')
      if (!seen.has(dir)) {
        seen.add(dir)
        rows.push({ key: dir + '/', depth: d, label: parts[d] + '/' })
      }
    }
    rows.push({ key: f, depth: parts.length - 1, label: parts[parts.length - 1], file: f })
  }
  return (
    <div className="file-tree mono">
      {!files.length && <div className="dim pad">No files yet.</div>}
      {rows.map((r) => (
        <div
          key={r.key}
          className={`tree-row ${r.file ? 'file' : 'dir'} ${r.file && r.file === sel ? 'sel' : ''}`}
          style={{ paddingLeft: 10 + r.depth * 14 }}
          onClick={() => r.file && onSel(r.file)}
        >
          {r.label}
          {r.file && r.file === live && <span className="badge live-badge">writing</span>}
        </div>
      ))}
    </div>
  )
}

export default function Code({ info, active }: { info: Info; active: boolean }) {
  const [status, setStatus] = useState<CodeStatus | null>(null)
  const [startLog, setStartLog] = useState<string[]>([])
  const [starting, setStarting] = useState(false)
  const [startErr, setStartErr] = useState<string | null>(null)
  const [items, setItems] = useState<Item[]>([])
  const [input, setInput] = useState('')
  const [running, setRunning] = useState(false)
  const [thinking, setThinking] = useState(true)
  const [effort, setEffort] = useState<'xhigh' | 'medium' | 'low'>('medium')
  const [temperature, setTemperature] = useState(0.6)
  const [context, setContext] = useState(0)
  const [plan, setPlan] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [sel, setSel] = useState<string | null>(null)
  const [fileText, setFileText] = useState<{ path: string; text: string; binary: boolean } | null>(null)
  const [filesVersion, setFilesVersion] = useState(0)
  const [pane, setPane] = useState<Pane>('preview')
  const [follow, setFollow] = useState(true)
  const [previewKey, setPreviewKey] = useState(0)
  const [previewUp, setPreviewUp] = useState(false)
  const [leftPct, setLeftPct] = useState(58)
  const [trace, setTrace] = useState<{ text: string; tokens: number } | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [lastShot, setLastShot] = useState(0)
  const [procs, setProcs] = useState<Record<string, { command: string; status: string }>>({})
  const [dragging, setDragging] = useState(false)
  const split = useRef<HTMLDivElement>(null)
  const autoStarted = useRef(false)
  const traceRef = useStickyScroll<HTMLDivElement>([trace?.text, pane === 'trace'])
  const buildLogRef = useStickyScroll<HTMLPreElement>([startLog])
  const followRef = useRef(follow)
  useEffect(() => {
    followRef.current = follow
  }, [follow])
  const eventRef = useRef<(e: CodeEvent) => void>(() => {})

  const refresh = useCallback(async () => {
    try {
      const s = await codeApi.status()
      setStatus(s)
      setPreviewUp(s.preview)
      return s
    } catch {
      return null
    }
  }, [])

  const start = useCallback(async () => {
    setStarting(true)
    setStartErr(null)
    setStartLog([])
    try {
      await codeApi.start((e) => {
        if (e.type === 'log') setStartLog((l) => [...l.slice(-400), e.line])
        else if (e.type === 'error') setStartErr(e.message)
      })
    } catch (e) {
      setStartErr(String((e as Error).message ?? e))
    } finally {
      setStarting(false)
      const s = await refresh()
      if (s?.running) codeApi.tree().then((t) => setFiles(t.files)).catch(() => {})
    }
  }, [refresh])

  // Poll the sandbox while the tab is visible; start it the first time the tab opens.
  useEffect(() => {
    if (!active) return
    let t: number
    const poll = async () => {
      const s = await refresh()
      if (s && s.docker && !s.running && !s.starting && !autoStarted.current) {
        autoStarted.current = true
        start()
      } else if (s?.running && !autoStarted.current) {
        autoStarted.current = true
        codeApi.tree().then((r) => setFiles(r.files)).catch(() => {})
        codeApi.checkpoints().then((r) => setCheckpoints(r.checkpoints)).catch(() => {})
        if (s.plan) setPlan(s.plan)
        setContext(s.context)
      }
      t = window.setTimeout(poll, 3000)
    }
    poll()
    return () => clearTimeout(t)
  }, [active, refresh, start])

  const chatScroll = useFollowScroll<HTMLDivElement>([items])

  const inputRef = useAutosize(input)

  // Selected file content (refetched whenever the agent changed files).
  useEffect(() => {
    if (!sel) return
    let ok = true
    codeApi
      .file(sel)
      .then((f) => ok && setFileText({ path: f.path, text: f.text, binary: f.binary }))
      .catch(() => ok && setFileText(null))
    return () => {
      ok = false
    }
  }, [sel, filesVersion])

  // Raw context (what the model actually sees), refreshed after every model turn while the pane is open.
  useEffect(() => {
    if (pane !== 'trace' || !active) return
    codeApi.transcript().then(setTrace).catch(() => {})
  }, [pane, active, context, items.length])

  // The file currently being written by a streaming write_file call (the "canvas").
  const liveFile = useMemo(() => {
    const last = items[items.length - 1]
    if (!last || last.kind !== 'turn' || !last.streaming) return null
    const p = last.parts[last.parts.length - 1]
    if (!p || p.t !== 'call' || p.done) return null
    const { name, params } = parseLive(p.raw)
    const fp = params.file_path ?? params.path
    if (name !== 'write_file' || !fp || params.content === undefined) return null
    return { path: rel(fp), content: params.content }
  }, [items])

  useEffect(() => {
    if (liveFile && follow && inWorkspace(liveFile.path)) {
      setPane('files')
      setSel(liveFile.path)
    }
  }, [liveFile?.path, follow]) // eslint-disable-line react-hooks/exhaustive-deps

  const patchTurn = (f: (t: Turn) => Turn) =>
    setItems((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        const it = prev[i]
        if (it.kind === 'turn') {
          const next = [...prev]
          next[i] = f(it)
          return next
        }
      }
      return prev
    })

  const patchLastPart = (t: Turn, f: (p: Part | undefined) => Part | null): Turn => {
    const parts = [...t.parts]
    const r = f(parts[parts.length - 1])
    if (r) parts[parts.length - 1] = r
    return { ...t, parts }
  }

  const clearSession = () => {
    setItems([])
    setPlan('')
    setFiles([])
    setSel(null)
    setFileText(null)
    setContext(0)
    setRunning(false)
    setCheckpoints([])
    setProcs({})
  }

  const onEvent = (e: CodeEvent) => {
    const now = e.t ?? Date.now()
    const follow = followRef.current
    switch (e.type) {
      case 'reset':
        clearSession()
        break
      case 'user':
        chatScroll.jumpToEnd()
        if (!e.queued) setRunning(true)
        setItems((p) => [...p, { kind: 'user', text: e.text, queued: e.queued }])
        break
      case 'hook':
        setItems((p) => [...p, { kind: 'hook', event: e.event, name: e.name, ok: e.ok, blocked: e.blocked, output: e.output, image: e.image ?? null, n: Date.now() }])
        break
      case 'process':
        setProcs((m) => ({ ...m, [e.id]: { command: e.command, status: e.status } }))
        break
      case 'turn_start':
        setItems((p) => [
          ...p,
          { kind: 'turn', streaming: true, parts: e.thinking ? [{ t: 'think', text: '', start: now, toks: 0 }] : [] },
        ])
        break
      case 'delta':
        patchTurn((t) => {
          const last = t.parts[t.parts.length - 1]
          if (e.kind === 'think' && last?.t === 'think')
            return patchLastPart(t, () => ({ ...last, text: last.text + e.text, toks: last.toks + 1 }))
          if (e.kind === 'call' && last?.t === 'call') return patchLastPart(t, () => ({ ...last, raw: last.raw + e.text }))
          if (e.kind === 'text') {
            if (last?.t === 'text') return patchLastPart(t, () => ({ ...last, text: last.text + e.text }))
            return { ...t, parts: [...t.parts, { t: 'text', text: e.text }] }
          }
          return t
        })
        break
      case 'think_end':
        patchTurn((t) => ({ ...t, parts: t.parts.map((p) => (p.t === 'think' && !p.end ? { ...p, end: now } : p)) }))
        break
      case 'call_start':
        patchTurn((t) => ({ ...t, parts: [...t.parts, { t: 'call', raw: '', done: false }] }))
        break
      case 'call_end':
        patchTurn((t) => patchLastPart(t, (p) => (p?.t === 'call' ? { ...p, done: true } : null)))
        break
      case 'turn_end':
        setContext(e.context)
        patchTurn((t) => ({
          ...t,
          streaming: false,
          stats: e,
          parts: t.parts.map((p) => (p.t === 'think' && !p.end ? { ...p, end: now } : p.t === 'call' ? { ...p, done: true } : p)),
        }))
        break
      case 'tool_call':
        patchTurn((t) => {
          const i = t.parts.findIndex((p) => p.t === 'call' && !p.id)
          if (i < 0) return t
          const parts = [...t.parts]
          parts[i] = { ...(parts[i] as CallPart), id: e.id, name: e.name, args: e.args }
          return { ...t, parts }
        })
        if (follow) {
          if (e.name === 'run_shell_command' || e.name === 'read_process') setPane('terminal')
          else if (e.name === 'update_plan') setPane('plan')
          else if ((e.name === 'edit' || e.name === 'read_file') && inWorkspace(String(e.args.file_path ?? e.args.path ?? ''))) {
            setPane('files')
            setSel(rel(String(e.args.file_path ?? e.args.path ?? '')))
          }
        }
        break
      case 'tool_result':
        patchTurn((t) => ({
          ...t,
          parts: t.parts.map((p) => (p.t === 'call' && p.id === e.id ? { ...p, result: { ok: e.ok, output: e.output, ms: e.ms } } : p)),
        }))
        break
      case 'files':
        setFiles(e.tree)
        setFilesVersion((v) => v + 1)
        break
      case 'plan':
        setPlan(e.md)
        break
      case 'preview':
        setPreviewUp(e.up)
        if (e.up) {
          setPreviewKey((k) => k + 1)
          if (follow) setPane('preview')
        }
        break
      case 'browser':
        setLastShot(e.shot)
        setItems((p) => [...p, { kind: 'browser', ok: e.ok, errors: e.errors, failed: e.failed, empty: e.empty, canvases: e.canvases, shot: e.shot }])
        break
      case 'checkpoint':
        setCheckpoints((c) => [{ id: e.id, time: Math.round(now / 1000), message: e.message }, ...c.filter((x) => x.id !== e.id)])
        break
      case 'note':
        setItems((p) => [...p, { kind: 'note', text: e.text }])
        break
      case 'introspection':
        setItems((p) => [...p, { kind: 'note', warn: !e.assessment.toLowerCase().startsWith('progressing'),
          text: `Progress check: ${e.assessment || '?'}. ${e.root_cause ? 'Root cause: ' + e.root_cause.split('\n')[0] + '. ' : ''}${e.new_approach ? 'Next: ' + e.new_approach.split('\n')[0] : ''}` }])
        break
      case 'reminder':
        setItems((p) => [...p, { kind: 'reminder', text: e.text }])
        break
      case 'check':
        setItems((p) => [...p, { kind: 'check', ok: e.ok, output: e.output }])
        break
      case 'compact':
        setContext(e.after)
        setItems((p) => [
          ...p,
          {
            kind: 'note',
            text: `Context compacted: ${(e.before / 1000).toFixed(1)}k → ${(e.after / 1000).toFixed(1)}k tokens (old tool outputs, file bodies and reasoning masked).`,
          },
        ])
        break
      case 'done':
        setContext(e.context)
        setRunning(false)
        patchTurn((t) => (t.streaming ? { ...t, streaming: false } : t))
        setFilesVersion((v) => v + 1)
        if (STOP_NOTES[e.stop]) setItems((p) => [...p, { kind: 'note', text: STOP_NOTES[e.stop], warn: e.stop !== 'cancelled' }])
        break
      case 'error':
        setItems((p) => [...p, { kind: 'note', text: e.message, warn: true }])
        break
    }
  }

  useEffect(() => {
    eventRef.current = onEvent
  })

  // Subscribe to the session's event log: replays everything on (re)connect, then follows it live.
  // The agent runs server-side, so a dropped connection or a page reload loses nothing.
  useEffect(() => {
    const ctl = new AbortController()
    let epoch = -1
    let next = 0
    ;(async () => {
      while (!ctl.signal.aborted) {
        try {
          await codeApi.events(
            epoch,
            next,
            (e) => {
              if (e.type === 'reset') {
                epoch = e.epoch
                next = 0
              } else if (e.seq !== undefined) next = e.seq + 1
              eventRef.current(e)
            },
            ctl.signal,
          )
        } catch {
          /* disconnected: retry */
        }
        await new Promise((r) => setTimeout(r, 1000))
      }
    })()
    return () => ctl.abort()
  }, [])

  const send = async () => {
    const text = input.trim()
    if (!text || !info.ready || !status?.running) return
    setInput('')
    try {
      await codeApi.send({ text, thinking, reasoning_effort: effort, temperature })
    } catch (e) {
      setInput(text)
      setItems((p) => [...p, { kind: 'note', text: String((e as Error).message ?? e), warn: true }])
    }
  }

  const reset = async () => {
    if (!confirm('Reset the coding session? The container is recreated: all files and the conversation are lost.')) return
    try {
      await codeApi.reset()
    } catch (e) {
      alert(String((e as Error).message ?? e))
      return
    }
    clearSession()
    setPreviewUp(false)
    refresh()
  }

  // Draggable divider between workspace and chat.
  useEffect(() => {
    if (!dragging) return
    const move = (ev: MouseEvent) => {
      const r = split.current?.getBoundingClientRect()
      if (r) setLeftPct(Math.min(80, Math.max(25, ((ev.clientX - r.left) / r.width) * 100)))
    }
    const up = () => setDragging(false)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
  }, [dragging])

  const terminal = useMemo(() => {
    const out: CallPart[] = []
    for (const it of items)
      if (it.kind === 'turn')
        for (const p of it.parts)
          if (p.t === 'call' && ['run_shell_command', 'read_process', 'kill_process'].includes(callInfo(p).name ?? '')) out.push(p)
    return out
  }, [items])
  const terminalRef = useStickyScroll<HTMLDivElement>([terminal, procs, pane === 'terminal'])

  const maxCtx = status?.max_context ?? 56000
  const sandboxReady = !!status?.running
  const shownFile =
    liveFile && sel === liveFile.path ? { path: liveFile.path, text: liveFile.content, binary: false } : fileText?.path === sel ? fileText : null

  return (
    <div className={`code ${dragging ? 'dragging' : ''}`} ref={split}>
      {/* ------------------------------------------------ workspace */}
      <section className="code-left" style={{ width: `${leftPct}%` }}>
        <div className="pane-tabs">
          {(['preview', 'files', 'plan', 'terminal', 'history', 'trace'] as Pane[]).map((p) => (
            <button key={p} className={pane === p ? 'on' : ''} onClick={() => setPane(p)}>
              {{ preview: 'Preview', files: 'Files', plan: 'Plan', terminal: 'Terminal', history: 'History', trace: 'Raw context' }[p]}
              {p === 'preview' && <span className={`led ${previewUp ? 'up' : ''}`} />}
              {p === 'files' && files.length > 0 && <span className="count">{files.length}</span>}
              {p === 'terminal' && terminal.length > 0 && <span className="count">{terminal.length}</span>}
            </button>
          ))}
          <label className="check follow" title="Switch panes to follow what the agent is doing">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
            follow agent
          </label>
        </div>

        {!sandboxReady ? (
          <div className="pane-body sandbox-setup">
            {status && !status.docker ? (
              <div className="err">Docker is not running. Start Docker Desktop, then reload this tab.</div>
            ) : (
              <>
                <div>
                  {starting || status?.starting
                    ? status?.image
                      ? 'Starting the sandbox container…'
                      : 'Building the sandbox image (first run only: Node 22, uv, a React template)…'
                    : 'The sandbox container is not running.'}
                </div>
                {!starting && (
                  <button className="primary" onClick={start}>
                    Start sandbox
                  </button>
                )}
                {startErr && <div className="err">{startErr}</div>}
                {startLog.length > 0 && <pre className="build-log mono" ref={buildLogRef}>{startLog.slice(-200).join('\n')}</pre>}
              </>
            )}
          </div>
        ) : (
          <>
            <div className="pane-body" hidden={pane !== 'preview'}>
              {previewUp ? (
                <div className="preview">
                  <div className="preview-bar">
                    <button className="link" title="Reload" onClick={() => setPreviewKey((k) => k + 1)}>
                      ↻
                    </button>
                    <span className="mono dim url">{status?.preview_url}</span>
                    <a href={status?.preview_url} target="_blank" rel="noreferrer" className="link" title="Open in a new tab">
                      ↗
                    </a>
                  </div>
                  <iframe key={previewKey} src={status?.preview_url} title="preview" />
                </div>
              ) : (
                <div className="empty-pane dim">
                  No dev server is running yet. Ask the agent to build something, e.g. "Build a todo app with React".
                </div>
              )}
            </div>
            <div className="pane-body files-pane" hidden={pane !== 'files'}>
              <FileTree files={files} sel={sel} onSel={setSel} live={liveFile?.path} />
              <div className="file-view">
                {shownFile ? (
                  <>
                    <div className="file-head mono">
                      {shownFile.path}
                      {liveFile?.path === shownFile.path && <span className="badge live-badge">agent is writing…</span>}
                    </div>
                    {shownFile.binary ? (
                      <div className="dim pad">Binary file.</div>
                    ) : (
                      // keyed by path: each file opens at the top with fresh follow state
                      <CodeView
                        key={shownFile.path}
                        path={shownFile.path}
                        text={shownFile.text}
                        live={liveFile?.path === shownFile.path}
                      />
                    )}
                  </>
                ) : (
                  <div className="empty-pane dim">{sel ? 'Loading…' : 'Select a file.'}</div>
                )}
              </div>
            </div>
            <div className="pane-body plan-pane" hidden={pane !== 'plan'}>
              {plan ? <Markdown text={plan} /> : <div className="empty-pane dim">The agent hasn't made a plan yet.</div>}
            </div>
            <div className="pane-body history-pane" hidden={pane !== 'history'}>
              {!checkpoints.length && <div className="dim">No checkpoints yet. One is saved before each message and after every step that changes files.</div>}
              {checkpoints.map((c, i) => (
                <div key={c.id} className="ckpt">
                  <span className="mono dim">{new Date(c.time * 1000).toLocaleTimeString()}</span>
                  <span className="mono ckpt-id">{c.id}</span>
                  <span className="ckpt-msg">{c.message}</span>
                  {i > 0 && (
                    <button
                      disabled={running}
                      onClick={async () => {
                        if (!confirm(`Restore the workspace to checkpoint ${c.id}?\n"${c.message}"`)) return
                        try {
                          await codeApi.restore(c.id)
                          setFilesVersion((v) => v + 1)
                          setPreviewKey((k) => k + 1)
                        } catch (e) {
                          alert(String((e as Error).message ?? e))
                        }
                      }}
                    >
                      Restore
                    </button>
                  )}
                </div>
              ))}
            </div>
            <div className="pane-body trace-pane mono" hidden={pane !== 'trace'} ref={traceRef}>
              {trace?.text ? (
                <>
                  <div className="dim trace-head">
                    {trace.tokens.toLocaleString()} tokens: the agent's whole context exactly as the model sees it (system prompt, tool
                    schemas, reasoning, tool calls and tool responses).
                  </div>
                  <pre>{trace.text}</pre>
                </>
              ) : (
                <div className="dim">No context yet.</div>
              )}
            </div>
            <div className="pane-body terminal-pane mono" hidden={pane !== 'terminal'} ref={terminalRef}>
              {Object.keys(procs).length > 0 && (
                <div className="proc-list">
                  {Object.entries(procs).map(([id, p]) => (
                    <span key={id} className={`proc ${p.status.startsWith('running') ? 'live' : ''}`} title={p.command}>
                      {id} · {p.command.slice(0, 40)} · {p.status}
                    </span>
                  ))}
                </div>
              )}
              {!terminal.length && <div className="dim">No commands yet.</div>}
              {terminal.map((p, i) => {
                const { name, args } = callInfo(p)
                const cmd = name === 'run_shell_command' ? args.command : `[${name === 'read_process' ? 'read' : 'kill'} ${args.process_id}]`
                return (
                  <div key={i} className="term-entry">
                    <div className="term-cmd">
                      <span className="prompt">$</span> {String(cmd ?? '')}
                      {!p.result && <span className="spinner" />}
                    </div>
                    {p.result && <pre className={p.result.ok ? '' : 'fail'}>{p.result.output}</pre>}
                  </div>
                )
              })}
            </div>
          </>
        )}
      </section>

      <div className="divider" onMouseDown={() => setDragging(true)} />

      {/* ------------------------------------------------ agent chat */}
      <section className="code-right">
        <div className="code-toolbar">
          <label className="check">
            <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
            thinking
          </label>
          {thinking && (
            <select value={effort} onChange={(e) => setEffort(e.target.value as typeof effort)} title="reasoning effort">
              <option value="medium">medium</option>
              <option value="low">low</option>
              <option value="xhigh">xhigh</option>
            </select>
          )}
          <label className="temp">
            T
            <input type="number" step={0.1} min={0} value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)} />
          </label>
          <div className="ctx-meter" title={`context ${context.toLocaleString()} / ${maxCtx.toLocaleString()} tokens`}>
            <div className="ctx-fill" style={{ width: `${Math.min(100, (context / maxCtx) * 100)}%` }} />
            <span className="mono">
              {(context / 1000).toFixed(1)}k / {Math.round(maxCtx / 1000)}k
            </span>
          </div>
          <a className="link" href="/api/code/archive" title="Download the workspace (.tar.gz, without node_modules)">
            ⤓
          </a>
          <button onClick={reset} disabled={!sandboxReady} title="Recreate the container and start over">
            Reset
          </button>
        </div>

        <div
          className="chat-scroll"
          ref={chatScroll.ref}
        >
          <div className="chat-col code-col">
            {!items.length && (
              <div className="chat-empty">
                <div className="logo big-logo">◆</div>
                <div>A coding agent running on Ternary Bonsai 2 27B.</div>
                <div className="dim">
                  It works in a Docker sandbox (Node, npm, uv) and can run any command. Web apps it starts show up in Preview.
                </div>
              </div>
            )}
            {items.map((it, i) =>
              it.kind === 'user' ? (
                <div key={i} className="msg user">
                  <div className="bubble">{it.text}</div>
                  {it.queued && <span className="dim small-note">queued: the agent sees it at its next step</span>}
                </div>
              ) : it.kind === 'hook' ? (
                <details key={i} className={`tool-card ${it.ok ? 'ok' : 'fail'}`} open={!it.ok && it.event === 'Stop'}>
                  <summary>
                    <span className="tool-icon mono">{it.event === 'Stop' ? '⏹' : '✓'}</span>
                    <span className="tool-text">
                      {it.event === 'Stop' ? 'Finish check' : 'Check'} <b>{it.name}</b>: {it.ok ? 'ok' : it.blocked ? 'failed (sent to the agent)' : 'hook error'}
                    </span>
                  </summary>
                  <div className="tool-out">
                    {it.image && it.ok !== undefined && (
                      <img className="shot" src={`/api/code/image?path=${encodeURIComponent(it.image)}&n=${it.n}`} alt="" onError={(ev) => ((ev.target as HTMLImageElement).style.display = 'none')} />
                    )}
                    <pre className={`mono ${it.ok ? '' : 'fail'}`}>{it.output || '(no output)'}</pre>
                  </div>
                </details>
              ) : it.kind === 'browser' ? (
                <details key={i} className={`tool-card ${it.ok ? 'ok' : 'fail'}`}>
                  <summary>
                    <span className="tool-icon mono">◉</span>
                    <span className="tool-text">
                      Headless browser check: {it.ok ? 'renders cleanly' : `${it.errors.length + it.failed.length + (it.empty ? 1 : 0)} problem(s)`}
                      {it.canvases ? ` · ${it.canvases} canvas` : ''}
                    </span>
                  </summary>
                  <div className="tool-out">
                    {it.shot === lastShot ? (
                      <img className="shot" src={`/api/code/screenshot?n=${it.shot}`} alt="headless screenshot" />
                    ) : (
                      <div className="dim">(older screenshot replaced by a newer check)</div>
                    )}
                    {(it.errors.length > 0 || it.failed.length > 0 || it.empty) && (
                      <pre className="mono fail">{[...it.errors, ...it.failed, ...(it.empty ? ['blank page'] : [])].join('\n')}</pre>
                    )}
                  </div>
                </details>
              ) : it.kind === 'check' ? (
                <details key={i} className={`tool-card ${it.ok ? 'ok' : 'fail'}`} open={!it.ok}>
                  <summary>
                    <span className="tool-icon mono">✔</span>
                    <span className="tool-text">
                      Automatic check before finishing: {it.ok ? 'build is clean' : 'errors found, sent back to the agent'}
                    </span>
                  </summary>
                  <pre className={`tool-out mono ${it.ok ? '' : 'fail'}`}>{it.output}</pre>
                </details>
              ) : it.kind === 'reminder' ? (
                <details key={i} className="tool-card harness">
                  <summary>
                    <span className="tool-icon mono">↳</span>
                    <span className="tool-text dim">harness note to the model</span>
                  </summary>
                  <pre className="tool-out mono">{it.text.replace(/^<system-reminder>\n?|\n?<\/system-reminder>$/g, '')}</pre>
                </details>
              ) : it.kind === 'note' ? (
                <div key={i} className={`code-note ${it.warn ? 'warn' : ''}`}>
                  {it.text}
                </div>
              ) : (
                <TurnView key={i} turn={it} />
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
            <textarea
              ref={inputRef}
              rows={2}
              value={input}
              placeholder={
                !info.ready ? 'Waiting for the model to load…' : !sandboxReady ? 'Waiting for the sandbox…' : 'Describe what to build or change…'
              }
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <div className="composer-actions">
              <span className="dim small-note">{running ? 'Agent is working…' : 'Enter sends · Shift+Enter new line'}</span>
              <div className="row-gap">
                {running && (
                  <button onClick={() => codeApi.stop()} title="Stop the agent">
                    ■ Stop
                  </button>
                )}
                <button className="primary" disabled={!input.trim() || !info.ready || !sandboxReady} onClick={send}
                  title={running ? 'Queue a message: the agent reads it at its next step' : 'Send'}>
                  {running ? 'Queue' : 'Send'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
