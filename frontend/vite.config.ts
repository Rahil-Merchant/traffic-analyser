// project/frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // This is the new proxy configuration.
    proxy: {
      // Any request starting with /api will be forwarded to the backend.
      '/api': {
        target: 'http://127.0.0.1:7862', // The address of your FastAPI backend
        changeOrigin: true,
      },
    },
  },
})