import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Sub-path when served from GitHub Pages (e.g. /PBL-5/); "/" otherwise.
  base: process.env.VITE_BASE || '/',
})
