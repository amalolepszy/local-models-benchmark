import { defineConfig, type Plugin } from 'vite';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve, extname, normalize, sep } from 'node:path';

const COOP_COEP_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

/**
 * Serves the cloned Speedometer repo (vendor/Speedometer/) at /speedometer/
 * so it shares an origin with the rest of the app. That lets the
 * speedometer_under_load.py driver inject inference code into the
 * Speedometer page using the existing src/benchmarks/* modules without
 * fighting CORS or losing crossOriginIsolated (needed for SharedArrayBuffer
 * → multi-threaded WASM).
 */
function serveSpeedometer(): Plugin {
  const root = resolve(__dirname, 'vendor/Speedometer');
  const rootWithSep = root + sep;
  return {
    name: 'serve-speedometer',
    configureServer(server) {
      server.middlewares.use('/speedometer', (req, res, next) => {
        const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0]!);
        const requested = urlPath === '/' || urlPath === '' ? '/index.html' : urlPath;
        const candidate = normalize(resolve(root, '.' + requested));

        // Path-traversal guard.
        if (candidate !== root && !candidate.startsWith(rootWithSep)) {
          return next();
        }

        let filePath = candidate;
        if (existsSync(filePath) && statSync(filePath).isDirectory()) {
          filePath = resolve(filePath, 'index.html');
        }
        if (!existsSync(filePath)) return next();

        const mime = MIME[extname(filePath).toLowerCase()] ?? 'application/octet-stream';
        res.setHeader('Content-Type', mime);
        for (const [k, v] of Object.entries(COOP_COEP_HEADERS)) res.setHeader(k, v);
        createReadStream(filePath).pipe(res);
      });
    },
  };
}

export default defineConfig({
  server: {
    port: 5173,
    headers: {
      // Required for SharedArrayBuffer (WASM threading) and memory measurement API
      ...COOP_COEP_HEADERS,
    },
    fs: {
      allow: ['.', './vendor'],
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
  plugins: [serveSpeedometer()],
});
