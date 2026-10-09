import { memo, useRef, useState, type ComponentProps, type ReactElement } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import { normalizeMath } from './math'

function CopyIcon({ done }: { done: boolean }) {
  return done ? (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
      <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden>
      <rect x="5" y="5" width="8.5" height="8.5" rx="1.6" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3.5 10.5h-.5A1.5 1.5 0 0 1 1.5 9V3A1.5 1.5 0 0 1 3 1.5h6A1.5 1.5 0 0 1 10.5 3v.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  )
}

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    ta.remove()
  }
}

function CodeBlock(props: ComponentProps<'pre'>) {
  const ref = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)
  const child = props.children as ReactElement<{ className?: string }> | undefined
  const lang = /language-([\w+#-]+)/.exec(child?.props?.className ?? '')?.[1]
  return (
    <div className="codeblock">
      <div className="codeblock-head">
        <span className="codeblock-lang">{lang ?? 'text'}</span>
        <button
          className="codeblock-copy"
          title="Copy to clipboard"
          onClick={async () => {
            await copyText(ref.current?.textContent ?? '')
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
          }}
        >
          <CopyIcon done={copied} />
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre ref={ref}>{props.children}</pre>
    </div>
  )
}

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        // KaTeX first: math arrives as <code class="language-math">, which highlight.js must not touch.
        rehypePlugins={[
          [rehypeKatex, { throwOnError: false, strict: 'ignore' }],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
        components={{ pre: CodeBlock }}
      >
        {normalizeMath(text)}
      </ReactMarkdown>
    </div>
  )
})
