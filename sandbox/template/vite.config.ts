import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Listen on all interfaces so the sandbox can relay the dev server to the user's preview.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: '0.0.0.0', port: 5173, strictPort: true, allowedHosts: true },
})
