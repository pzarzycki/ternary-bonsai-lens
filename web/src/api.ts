export type Tok = { id: number; piece: string; text: string }

export type VStats = { l2: number; mean: number; std: number; min: number; max: number; argmax_abs: number }

export type Readout = {
  top: Tok[]
  top_logits: number[]
  top_probs: number[]
  entropy: number
  logit_stats: { mean: number; std: number; min: number; max: number }
  logit_hist: { counts: number[]; edges: number[] }
  pre_norm: number[]
  post_norm: number[]
  pre_stats: VStats
  post_stats: VStats
}

export type StepResult = {
  step: number
  position: number
  fed_tokens: number
  forward_ms: number
  chosen: Tok & { logit: number }
  is_eos: boolean
  readout: Readout
}

export type ResetResult = { rendered: string; tokens: Tok[]; prompt_len: number }

export type Info = {
  ready: boolean
  backend?: string
  lens_readouts?: boolean
  image_token_readouts?: boolean
  context_limit?: number
  error?: string | null
  hidden_size?: number
  vocab_size?: number
  num_layers?: number
  load_seconds?: number
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) {
    let msg = `${r.status}`
    try {
      msg = (await r.json()).detail ?? msg
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  return r.json()
}

export const api = {
  info: () => call<Info>('/api/info'),
  reset: (b: {
    prompt: string
    mode: string
    thinking: boolean
    reasoning_effort: string
    system?: string
    image_b64?: string
  }) =>
    call<ResetResult>('/api/reset', b),
  step: (b: { strategy: string; temperature: number; top_p: number; top_k: number; k: number }) =>
    call<StepResult>('/api/step', b),
  replace: (token_id: number) => call<{ chosen: Tok; is_eos: boolean }>('/api/replace', { token_id }),
}

/** Make whitespace / control chars visible in token chips. */
export function showTok(t: string): string {
  return t.replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/ /g, '·') || '∅'
}

export type ChatEvent =
  | { type: 'start'; thinking: boolean }
  | { type: 'delta'; text: string }
  | {
      type: 'done'
      stop: 'eos' | 'length' | 'cancelled'
      resumable?: boolean
      tokens: number
      prefill_s: number
      tok_per_s: number
      prompt_tokens: number
      context: number
    }
  | { type: 'error'; message: string }

/** POST and consume a text/event-stream reply, calling onEvent per `data:` event. */
export function streamPost<E>(path: string, body: unknown, onEvent: (e: E) => void, signal?: AbortSignal) {
  return streamFetch(
    path,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal },
    onEvent,
  )
}

/** Fetch `path` and consume its text/event-stream body, calling onEvent per `data:` event. */
export async function streamFetch<E>(path: string, init: RequestInit, onEvent: (e: E) => void) {
  const r = await fetch(path, init)
  if (!r.ok || !r.body) {
    let msg = `${r.status}`
    try {
      msg = (await r.json()).detail ?? msg
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += value
    let i
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i)
      buf = buf.slice(i + 2)
      for (const line of frame.split('\n')) if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)))
    }
  }
}

export const chatApi = {
  create: (b: { system?: string; thinking: boolean; reasoning_effort: string; temperature: number; max_tokens?: number }) =>
    call<{ ok: boolean }>('/api/chat/new', b),
  stop: () => call<{ ok: boolean }>('/api/chat/stop', {}),
  send: (
    b: { text: string; image_b64?: string; thinking: boolean; reasoning_effort: string; temperature: number; max_tokens: number },
    onEvent: (e: ChatEvent) => void,
    signal?: AbortSignal,
  ) => streamPost('/api/chat/send', b, onEvent, signal),
  /** Resume the last reply after it hit max_tokens or was stopped; same events as send. */
  resume: (b: { max_tokens: number }, onEvent: (e: ChatEvent) => void) => streamPost('/api/chat/continue', b, onEvent),
}

// ------------------------------------------------------------------ coding agent
export type CodeStatus = {
  docker: boolean
  image: boolean
  running: boolean
  starting: boolean
  preview: boolean
  preview_url: string
  busy: boolean
  context: number
  max_context: number
  plan: string
}

export type CodeEvent = (
  | { type: 'user'; text: string; queued?: boolean }
  | { type: 'hook'; event: string; name: string; ok: boolean; blocked: boolean; output: string; ms: number; image?: string | null }
  | { type: 'process'; id: string; command: string; status: string }
  | { type: 'reset'; epoch: number }
  | { type: 'turn_start'; step: number; thinking: boolean }
  | { type: 'delta'; kind: 'think' | 'text' | 'call'; text: string }
  | { type: 'think_end' }
  | { type: 'call_start'; index: number }
  | { type: 'call_end'; index: number }
  | { type: 'turn_end'; context: number; tokens: number; tok_per_s: number; prefill_s: number; stop: string }
  | { type: 'tool_call'; id: string; name: string | null; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; ok: boolean; output: string; ms: number }
  | { type: 'files'; tree: string[] }
  | { type: 'plan'; md: string }
  | { type: 'preview'; up: boolean }
  | { type: 'check'; ok: boolean; output: string }
  | { type: 'browser'; ok: boolean; errors: string[]; failed: string[]; empty?: boolean; canvases?: number; shot: number }
  | { type: 'checkpoint'; id: string; message: string }
  | { type: 'note'; text: string }
  | { type: 'reminder'; id: string; text: string }
  | { type: 'introspection'; assessment: string; evidence: string; root_cause: string; new_approach: string; need_user: string }
  | { type: 'compact'; before: number; after: number }
  | { type: 'done'; stop: string; steps: number; tool_calls: number; context: number }
  | { type: 'error'; message: string }
) & { seq?: number; t?: number }

export type StartEvent = { type: 'log'; line: string } | { type: 'ready' } | { type: 'error'; message: string }

export const codeApi = {
  status: () => call<CodeStatus>('/api/code/status'),
  start: (onEvent: (e: StartEvent) => void) => streamPost('/api/code/start', {}, onEvent),
  send: (b: { text: string; thinking: boolean; reasoning_effort: string; temperature: number }) =>
    call<{ ok: boolean; queued?: boolean }>('/api/code/send', b),
  /** Replay the session's event log from `since`, then follow it live (until aborted or disconnected). */
  events: (epoch: number, since: number, onEvent: (e: CodeEvent) => void, signal: AbortSignal) =>
    streamFetch(`/api/code/events?epoch=${epoch}&since=${since}`, { signal }, onEvent),
  stop: () => call<{ ok: boolean }>('/api/code/stop', {}),
  reset: () => call<{ ok: boolean }>('/api/code/reset', {}),
  tree: () => call<{ files: string[] }>('/api/code/tree'),
  checkpoints: () => call<{ checkpoints: { id: string; time: number; message: string }[] }>('/api/code/checkpoints'),
  restore: (id: string) => call<{ ok: boolean }>('/api/code/restore', { id }),
  transcript: () => call<{ text: string; tokens: number }>('/api/code/transcript'),
  file: (path: string) =>
    call<{ path: string; binary: boolean; size: number; text: string }>(`/api/code/file?path=${encodeURIComponent(path)}`),
}
