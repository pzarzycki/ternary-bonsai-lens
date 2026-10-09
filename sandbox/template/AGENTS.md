# React + TypeScript + Vite project
- Entry: src/main.tsx mounts src/App.tsx; global styles in src/index.css. Set the page <title> in index.html.
- src/App.tsx starts as a blank placeholder; src/index.css enables Tailwind and makes html/body/#root full height.
- Installed: tailwindcss v4 (Vite plugin configured; add `@import "tailwindcss";` at the top of src/index.css),
  lucide-react (real icon names only), three + @react-three/fiber + @react-three/drei (read the
  react-three-fiber skill before 3D work).
- Other packages: `npm install <pkg>` BEFORE importing them; prefer what is installed.
- Components in src/components/, one per file, under ~200 lines.
- Never recreate package.json, tsconfig*.json, index.html or vite.config.ts.
- Dev server: `npm run dev` (port 5173). The harness starts it automatically; don't start another one.
- Checks after edits: imports (missing files/packages/exports), typecheck (tsc), and the page check (headless
  browser: runtime errors, blank page, unsized canvas, unstyled classes, plus a screenshot).
