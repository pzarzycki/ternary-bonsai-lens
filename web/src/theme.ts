import hljsDark from 'highlight.js/styles/github-dark.css?inline'
import hljsLight from 'highlight.js/styles/github.css?inline'

export type Theme = 'light' | 'dark'

const KEY = 'bonsai-theme'

export function loadTheme(): Theme {
  try {
    return localStorage.getItem(KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

/** Set data-theme on <html>, swap the highlight.js stylesheet, and remember the choice. */
export function applyTheme(t: Theme) {
  document.documentElement.dataset.theme = t
  let style = document.getElementById('hljs-theme') as HTMLStyleElement | null
  if (!style) {
    style = document.createElement('style')
    style.id = 'hljs-theme'
    document.head.appendChild(style)
  }
  style.textContent = t === 'dark' ? hljsDark : hljsLight
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* storage blocked */
  }
}
