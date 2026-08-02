import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Frontend/backend are separated. In development, Vite serves the SPA on
// :5173 and proxies /api to the FastAPI backend (default 127.0.0.1:8000).
// In production, `npm run build` emits web/frontend/dist which the FastAPI
// app serves when LIVE_DOC_SERVE_FRONTEND=1.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
