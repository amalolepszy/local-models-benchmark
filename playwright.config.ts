import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 5 * 60_000, // 5 min per test — model downloads can be slow
  retries: 0,
  workers: 1, // sequential — memory measurements need isolation
  use: {
    baseURL: 'http://localhost:5173',
    browserName: 'chromium',
    launchOptions: {
      executablePath: 'D:/chr-build/chromium/src/out/Release/chrome.exe',
      args: [
        '--enable-precise-memory-info',
        '--js-flags=--expose-gc',
        '--enable-blink-features=TFLiteNativeInference',
      ],
    },
  },
  webServer: {
    command: 'npm run dev',
    port: 5173,
    reuseExistingServer: true,
  },
});
