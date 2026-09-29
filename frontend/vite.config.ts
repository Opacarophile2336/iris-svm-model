import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 2336,
    proxy: {
      '/auth': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/ml': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/prediction': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/image': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/datasets': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/dl': { target: 'http://127.0.0.1:2006', changeOrigin: true },
      '/assets': { target: 'http://127.0.0.1:2006', changeOrigin: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 1500,
  },
})
