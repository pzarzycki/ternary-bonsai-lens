import { useCallback, useEffect, useLayoutEffect, useRef, useState, type DependencyList } from 'react'

/**
 * Keep a scrollable element pinned to its bottom while its content grows, like a terminal.
 *
 * Following is driven by what the user does, not by geometry sampled in scroll events: our own jumps
 * to the bottom also fire scroll events, and if more content lands before such an event is handled the
 * element looks "scrolled away" and following would silently stop. So:
 *  - scrolling up (wheel/touch/keys/scrollbar, i.e. scrollTop decreasing) releases it;
 *  - reaching the bottom (within `slack` px) re-engages it;
 *  - while following, any growth re-pins: `deps` changes and a ResizeObserver on the content
 *    (images loading, <details> opening, streamed text) both trigger it.
 * `enabled: false` never auto-scrolls.
 */
export function useFollowScroll<T extends HTMLElement>(deps: DependencyList, enabled = true, slack = 40) {
  const ref = useRef<T>(null)
  const following = useRef(true)
  const [atEnd, setAtEnd] = useState(true)
  const enabledRef = useRef(enabled)
  enabledRef.current = enabled

  const set = useCallback((v: boolean) => {
    if (following.current !== v) {
      following.current = v
      setAtEnd(v)
    }
  }, [])

  const pin = useCallback(() => {
    const el = ref.current
    if (el && enabledRef.current && following.current && el.clientHeight) el.scrollTop = el.scrollHeight
  }, [])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    let lastTop = el.scrollTop
    const onScroll = () => {
      if (!el.clientHeight) return // hidden (display: none) pane: keep the last state
      const dist = el.scrollHeight - el.scrollTop - el.clientHeight
      if (dist <= slack) set(true)
      else if (el.scrollTop < lastTop - 1) set(false) // moved up: only the user does that
      lastTop = el.scrollTop
    }
    // A wheel/touch gesture upward releases immediately, even before the scroll event (or when an inner
    // scroller consumes it), so the next streamed token can't yank the view back down.
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) set(false)
    }
    let touchY = 0
    const onTouchStart = (e: TouchEvent) => (touchY = e.touches[0]?.clientY ?? 0)
    const onTouchMove = (e: TouchEvent) => {
      if ((e.touches[0]?.clientY ?? 0) > touchY + 4) set(false)
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('touchstart', onTouchStart, { passive: true })
    el.addEventListener('touchmove', onTouchMove, { passive: true })
    const ro = new ResizeObserver(pin)
    ro.observe(el)
    const watchChildren = () => {
      for (const c of Array.from(el.children)) ro.observe(c)
    }
    watchChildren()
    const mo = new MutationObserver(watchChildren)
    mo.observe(el, { childList: true })
    return () => {
      el.removeEventListener('scroll', onScroll)
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchstart', onTouchStart)
      el.removeEventListener('touchmove', onTouchMove)
      ro.disconnect()
      mo.disconnect()
    }
  }, [slack, set, pin])

  useLayoutEffect(pin, [...deps, enabled]) // eslint-disable-line react-hooks/exhaustive-deps

  /** Follow again and jump to the bottom (e.g. when the user sends a message). */
  const jumpToEnd = useCallback(() => {
    set(true)
    pin()
  }, [set, pin])

  return { ref, atEnd, jumpToEnd }
}

/** useFollowScroll for callers that only need the ref. */
export function useStickyScroll<T extends HTMLElement>(deps: DependencyList, enabled = true, slack = 40) {
  return useFollowScroll<T>(deps, enabled, slack).ref
}
