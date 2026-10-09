/**
 * Prepare model output for remark-math, which only knows $…$ (inline) and $$ on their own lines (block).
 * - \( … \) -> $…$  and  \[ … \] -> block $$
 * - a line that is only $$ … $$ (as models usually write display math) -> block $$, keeping indentation
 * - a lone $ before a digit ("$5 and $10") is treated as currency and escaped
 * Fenced and inline code are left untouched.
 */
export function normalizeMath(src: string): string {
  return src
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g)
    .map((part, i) => (i % 2 ? part : fixMath(part)))
    .join('')
}

function fixMath(s: string): string {
  return s
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m: string) => `$${m.trim()}$`)
    .replace(/(^|[^$\\\w])\$(?=\d[\d.,]*(?:\s|$|[.,;:!?)]))/gm, '$1\\$$')
    .replace(/^([ \t]*)\$\$([^\n]+?)\$\$[ \t]*$/gm, (_, ind: string, m: string) => `${ind}$$\n${ind}${m.trim()}\n${ind}$$`)
}
