---
name: web-design
description: Use when building or restyling any user interface: palette and tokens, typography, spacing, overlay panels, states, responsiveness.
---

# Web design rules

- Palette: 3–5 colours, defined once as CSS variables or a Tailwind theme and used everywhere. Avoid raw
  one-off colours.
- At most 2 font families. Build a clear hierarchy from size, weight and letter-spacing, e.g. a small
  uppercase label above a large value.
- Use generous, consistent spacing on a 4/8 px scale, and align things to a grid.
- Make it complete: realistic content, plus hover, focus and active states and empty states. It must be
  responsive down to about 375 px wide.
- Overlays on visual content (maps, 3D, video):
  - make the container `position: fixed; inset: 0` and set `pointer-events: none` on it;
  - set `pointer-events: auto` on the panels;
  - give panels a glass look for dark scenes: `background: rgba(15,15,30,.55); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.12); border-radius: 16px`.
- Use the same class names in components and CSS. Every `className` needs a matching rule, or Tailwind
  utilities.
- Tokens example: `:root { --bg: #0b1020; --surface: rgba(15,15,30,.55); --text: #e8ecf6; --accent: #f4a261 }`,
  then use `var(--accent)` everywhere. Use either Tailwind or plain CSS per component, not both.
- Keep focus rings visible (`:focus-visible { outline: 2px solid var(--accent) }`).
- No external images or fonts unless asked. Use CSS gradients, inline SVG, lucide icons or 3D.
