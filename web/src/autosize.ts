import { useLayoutEffect, useRef } from 'react'

/** Size a textarea to its content (up to `max` px, then it scrolls). Driven by the value, so clearing it
 * after a send shrinks the box back instead of leaving it at the height of the last long message. */
export function useAutosize(value: string, max = 220) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, max) + 'px'
    el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden'
  }, [value, max])
  return ref
}
