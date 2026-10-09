// Headless browser check of a running web page (a Bonsai Code hook).
//   node /opt/bonsai/tools/check-page.mjs <url> [screenshot.png] [waitMs] [--json]
// Hook contract: exit 0 = renders cleanly, exit 2 = problems (one per line on stdout).
// Checks: uncaught errors, console.error, failed requests, the Vite error overlay, a blank #root,
// canvases left at the 300x150 default, class names with no CSS rule. Saves a screenshot.
import { mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname } from 'node:path'
import { chromium } from 'playwright-core'

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'))
const asJson = process.argv.includes('--json')
const url = args[0] ?? 'http://localhost:5173/'
const shot = args[1] ?? '/tmp/bonsai/preview.png'
const wait = Number(args[2] ?? 2500)
mkdirSync(dirname(shot), { recursive: true })

const errors = []
const failed = []
const browser = await chromium.launch({
  // SwiftShader gives headless Chromium a software WebGL implementation.
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
})
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  page.on('pageerror', (e) => errors.push(`uncaught: ${e.message}${e.stack ? '\n' + e.stack.split('\n').filter((l) => l.includes('/src/')).slice(0, 4).join('\n') : ''}`))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`)
  })
  page.on('requestfailed', (r) => failed.push(`${r.failure()?.errorText ?? 'failed'} ${r.url()}`))
  page.on('response', (r) => {
    if (r.status() >= 400) failed.push(`HTTP ${r.status()} ${r.url()}`)
  })
  await page.goto(url, { waitUntil: 'load', timeout: 20000 })
  await page.waitForTimeout(wait)
  const info = await page.evaluate((sourceWords) => {
    const overlay = document.querySelector('vite-error-overlay')
    const c = document.createElement('canvas')
    return {
      title: document.title,
      text: document.body?.innerText?.slice(0, 600) ?? '',
      overlay: overlay?.shadowRoot?.querySelector('.message')?.textContent ?? null,
      webgl: !!(c.getContext('webgl2') || c.getContext('webgl')),
      canvases: document.querySelectorAll('canvas').length,
      // Blank: a framework mount point that stayed empty, or a page with no visible text and no canvas.
      empty: (() => {
        const root = document.getElementById('root') || document.getElementById('app')
        if (root) return root.children.length === 0
        return !(document.body?.innerText || '').trim() && !document.querySelector('canvas, img, svg, video')
      })(),
      layout: layoutProblems(),
    }

    // Deterministic visual checks (a page can render with zero errors and still look broken).
    function layoutProblems() {
      const out = []
      for (const cv of document.querySelectorAll('canvas')) {
        const r = cv.getBoundingClientRect()
        if (r.width <= 310 || r.height <= 160)
          out.push(`a <canvas> is only ${Math.round(r.width)}x${Math.round(r.height)} px (the browser default is 300x150): ` +
            `give its parent an explicit size, e.g. position: fixed; inset: 0, or width: 100vw; height: 100vh`)
      }
      // Class names used in the DOM that no CSS rule mentions: usually a mismatch between the
      // classNames in the components and the selectors in the CSS files.
      const selectors = []
      const walk = (rules) => {
        for (const r of rules) {
          if (r.selectorText) selectors.push(r.selectorText)
          if (r.cssRules) walk(r.cssRules)
        }
      }
      for (const sh of document.styleSheets) {
        try { walk(sh.cssRules) } catch { /* cross-origin sheet */ }
      }
      const css = selectors.join('\n')
      const used = new Set()
      for (const el of document.querySelectorAll('[class]'))
        for (const c of (el.getAttribute('class') || '').split(/\s+/)) if (c) used.add(c)
      const esc = (c) => c.replace(/[^a-zA-Z0-9_-]/g, (m) => '\\' + m)
      // Only classes the project's own sources mention: libraries (icons, widgets) add their own.
      const own = new Set(sourceWords)
      const missing = [...used].filter((c) => own.has(c) && !css.includes('.' + esc(c)))
      if (missing.length >= 3)
        out.push(`${missing.length} CSS classes used in the page have no CSS rule (unstyled): ${missing.slice(0, 20).join(', ')}` +
          `. Make the classNames in the components match the selectors in your CSS files (or write the missing rules).`)
      return out
    }
  }, sourceWords())
  if (info.overlay) errors.unshift(`vite error overlay: ${info.overlay}`)
  errors.push(...info.layout.map((l) => `layout: ${l}`))
  errors.splice(0, errors.length, ...errors.map((e) => e.replace(/\(?http:\/\/localhost:\d+/g, '(').replace(/\?t=\d+/g, '')))
  await page.screenshot({ path: shot })
  const uniq = [...new Set(errors)] // React StrictMode runs effects twice in dev
  const problems = [...uniq, ...[...new Set(failed)].filter((f) => !f.includes('/__bonsai/')).map((f) => `request failed: ${f}`)]
  if (info.empty) problems.push('the page is blank: nothing rendered into #root')
  report(problems, { ...info, screenshot: shot })
} catch (e) {
  report([...errors, `could not load ${url}: ${e.message.split('\n')[0]} (is the dev server running?)`], {})
} finally {
  await browser.close()
}

/** Words in the project's source files (cwd), to tell its own class names from libraries'. */
function sourceWords() {
  const words = new Set()
  const walk = (dir, depth) => {
    if (depth > 8) return
    for (const name of readdirSync(dir)) {
      if (name.startsWith('.') || ['node_modules', 'dist', 'build'].includes(name)) continue
      const p = `${dir}/${name}`
      let st
      try { st = statSync(p) } catch { continue }
      if (st.isDirectory()) walk(p, depth + 1)
      else if (/\.(tsx?|jsx?|vue|svelte|html)$/.test(name) && st.size < 500000)
        for (const w of readFileSync(p, 'utf8').match(/[A-Za-z_][\w-]*/g) ?? []) words.add(w)
    }
  }
  try { walk(process.cwd(), 0) } catch { /* not a project dir */ }
  return [...words]
}

function report(problems, info) {
  if (asJson) console.log(JSON.stringify({ ok: problems.length === 0, problems, ...info }))
  else if (problems.length) console.log(problems.join('\n'))
  else console.log(`renders cleanly${info.canvases ? ` (${info.canvases} canvas)` : ''}: ${(info.title || '').slice(0, 60)}`)
  process.exitCode = problems.length ? 2 : 0
}
