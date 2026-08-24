import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
// Dev: proxy the two Gradio backends under distinct prefixes so the browser stays same-origin.
// /g1 -> :7860 (Narrative→Scenario), /g2 -> :7861 (Scenario→Narrative). Path is rewritten
// to strip the prefix so each Gradio app sees its own root. In prod, co-serve same-origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/g1': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
        rewrite: (requestPath) => requestPath.replace(/^\/g1/, ''),
      },
      '/g2': {
        target: 'http://127.0.0.1:7861',
        changeOrigin: true,
        rewrite: (requestPath) => requestPath.replace(/^\/g2/, ''),
      },
    },
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
