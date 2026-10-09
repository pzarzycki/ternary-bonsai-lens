---
name: live-dashboard
description: Use for a web page that shows live data or progress (training metrics, job status, monitoring): FastAPI JSON + HTML page with polling charts, robust states, verified in a real browser.
---

# Live dashboards

## Shape
- A FastAPI app with `GET /api/<data>` returning JSON (read the source file on every request; small and
  stateless) and `GET /` returning one self-contained HTML page with inline CSS and JS.
- Run `uv run uvicorn app:app --host 0.0.0.0 --port 8000` as its own command. It keeps running in the
  background, and the preview shows it.
- No CDNs or external assets. Draw charts yourself: an inline `<svg>` polyline with axes, or `<canvas>`.
  About 40 lines of JS covers a line chart with min/max scaling, a log-scale option and tooltips via `title`.

## The page script (where dashboards usually break)
- Put the `<script>` at the end of `<body>` (or use `DOMContentLoaded`). Look every element up by an id
  that exists in the HTML. A missing id gives "Cannot set properties of null" and the page hangs on
  "loading…".
- Poll with `setInterval(refresh, 5000)` and `fetch('/api/…')`. Use `try/catch` and show "server
  unreachable" instead of freezing. Keep polling after errors.
- Handle each state visibly:
  - no data yet ("waiting for the first metrics");
  - running (the latest values, the elapsed time);
  - finished (a final summary);
  - error.
- Use relative URLs (`/api/metrics`), never `http://localhost:8000/...`, because the user opens the page
  through a different port.

## Verify in a browser, not with curl
- `curl` shows the HTML but doesn't run its JavaScript. Use `check-page http://localhost:8000/`, which
  reports JS errors, failed requests and a blank page, and makes it pass (exit 0) before you tick the
  criterion.
- Typical criteria for the plan (copy the form exactly):
  ```
  - [ ] dashboard renders without JS errors — verify: `check-page http://localhost:8000/`
  - [ ] metrics API returns data — verify: `curl -sf localhost:8000/api/metrics | uv run python -c "import json,sys; assert len(json.load(sys.stdin)) > 0"`
  ```
