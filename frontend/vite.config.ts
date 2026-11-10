import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development /api is proxied to the local FastAPI server (uvicorn on :8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: {
    chunkSizeWarningLimit: 5000,
    rollupOptions: { output: { manualChunks: (id) => (id.includes('plotly') ? 'plotly' : id.includes('three') || id.includes('@react-three') ? 'three' : undefined) } },
  },
})
