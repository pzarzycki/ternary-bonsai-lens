// Deterministic import checker (used by the Bonsai Code harness after each edit; no model call).
//   cd /workspace && node /opt/bonsai/tools/check-imports.mjs src/App.tsx [more files…]
// Hook contract: exit 0 = clean, exit 2 = problems (printed one per line).
// Reports: missing relative files, packages that aren't installed, and named imports a package
// doesn't export (with the closest real names, e.g. a hallucinated lucide icon).
import { existsSync, readFileSync, statSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const cwd = process.cwd()
const req = createRequire(resolve(cwd, 'package.json'))
const EXT = ['', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.json', '/index.ts', '/index.tsx', '/index.js']
const IMPORT = /(?:^|\n)\s*(import|export)\s+(type\s+)?([\s\S]*?)\s*from\s*['"]([^'"]+)['"]|(?:^|\n)\s*import\s*['"]([^'"]+)['"]/g
const problems = []
const exportCache = new Map()

function lev(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)])
  for (let j = 1; j <= b.length; j++) d[0][j] = j
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
  return d[a.length][b.length]
}

async function exportsOf(spec) {
  if (exportCache.has(spec)) return exportCache.get(spec)
  let names = null
  try {
    names = new Set(Object.keys(await import(pathToFileURL(req.resolve(spec)).href)))
  } catch {
    /* cannot load in Node (needs a DOM, etc.): skip the named-export check */
  }
  exportCache.set(spec, names)
  return names
}

for (const file of process.argv.slice(2)) {
  if (!/\.(tsx?|jsx?|mjs)$/.test(file) || !existsSync(file)) continue
  const src = readFileSync(file, 'utf8')
  for (const m of src.matchAll(IMPORT)) {
    const typeOnly = !!m[2]
    const clause = m[3] ?? ''
    const spec = m[4] ?? m[5]
    if (spec.startsWith('/')) continue // served from the web root (e.g. public/), not the file system
    if (spec.startsWith('.')) {
      const base = resolve(dirname(file), spec.split('?')[0])
      if (!EXT.some((e) => existsSync(base + e) && (e !== '' || statSync(base).isFile())))
        problems.push(`${file}: imports "${spec}", but that file does not exist`)
      continue
    }
    if (spec.startsWith('virtual:') || spec.startsWith('node:')) continue
    const pkg = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0]
    if (!existsSync(resolve(cwd, 'node_modules', pkg))) {
      problems.push(`${file}: package "${pkg}" is not installed (run: npm install ${pkg})`)
      continue
    }
    if (typeOnly || !clause.includes('{') || /\.(css|svg|png|jpe?g|json)$/.test(spec)) continue
    const names = await exportsOf(spec)
    if (!names) continue
    const named = clause
      .slice(clause.indexOf('{') + 1, clause.lastIndexOf('}'))
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s && !s.startsWith('type '))
      .map((s) => s.split(/\s+as\s+/)[0].trim())
    for (const n of named) {
      if (names.has(n)) continue
      const close = [...names].map((x) => [lev(n.toLowerCase(), x.toLowerCase()), x]).sort((a, b) => a[0] - b[0]).slice(0, 3)
      problems.push(`${file}: "${spec}" has no export "${n}". Closest: ${close.map((c) => c[1]).join(', ')}`)
    }
  }
}
console.log(problems.length ? problems.join('\n') : 'OK')
process.exit(problems.length ? 2 : 0)
