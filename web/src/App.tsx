import { useEffect, useState } from 'react'
import { api, type Info } from './api'
import Chat from './Chat'
import Code from './Code'
import Lens from './Lens'
import { applyTheme, loadTheme, type Theme } from './theme'

type Tab = 'lens' | 'chat' | 'code'
const fromHash = (): Tab => (location.hash === '#chat' ? 'chat' : location.hash === '#code' ? 'code' : 'lens')

export default function App() {
  const [info, setInfo] = useState<Info>({ ready: false })
  const [tab, setTab] = useState<Tab>(fromHash)
  const [theme, setTheme] = useState<Theme>(loadTheme)
  useEffect(() => applyTheme(theme), [theme])

  useEffect(() => {
    let t: number
    const poll = async () => {
      try {
        const i = await api.info()
        setInfo(i)
        if (i.ready && i.lens_readouts === false && !location.hash) { setTab("chat"); history.replaceState(null, "", "#chat") }
        if (!i.ready && !i.error) t = window.setTimeout(poll, 2000)
      } catch {
        setInfo({ ready: false, error: 'server unreachable — is `uv run server.py` running?' })
        t = window.setTimeout(poll, 3000)
      }
    }
    poll()
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    const h = () => setTab(fromHash())
    window.addEventListener('hashchange', h)
    return () => window.removeEventListener('hashchange', h)
  }, [])

  const go = (t: Tab) => {
    setTab(t)
    history.replaceState(null, '', `#${t}`)
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">◆</span> Bonsai Lens
          <span className="sub">Ternary Bonsai 2 27B</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'lens' ? 'on' : ''} onClick={() => go('lens')}>
            Lens · single step
          </button>
          <button className={tab === 'chat' ? 'on' : ''} onClick={() => go('chat')}>
            Chat
          </button>
          <button className={tab === 'code' ? 'on' : ''} onClick={() => go('code')}>
            Code
          </button>
        </nav>
        <div className="header-end">
          <div className={`status ${info.ready ? 'ok' : info.error ? 'bad' : 'wait'}`}>
            {info.ready
              ? `ready · ${info.backend ?? "mlx"} · d=${info.hidden_size} · V=${info.vocab_size?.toLocaleString()} · ${info.num_layers} layers`
              : info.error ?? 'loading model…'}
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? (
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
              </svg>
            )}
          </button>
        </div>
      </header>
      {/* All views stay mounted so switching tabs keeps their state. */}
      <div hidden={tab !== 'lens'}>
        <>{info.lens_readouts === false ? <p style={{ padding: 24 }}>This server does not expose layer readouts. Build the Prism readout bridge to use Lens with the fast CUDA engine.</p> : <Lens info={info} />}</>
      </div>
      <div hidden={tab !== 'chat'} className="chat-host">
        <Chat info={info} />
      </div>
      <div hidden={tab !== 'code'} className="chat-host">
        <Code info={info} active={tab === 'code'} />
      </div>
    </div>
  )
}
