import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    headers: {
      // Required for SharedArrayBuffer (WASM threading) and memory measurement API
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
});
