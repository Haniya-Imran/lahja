import { defineConfig } from '@playwright/test'

// Runs against a built site + API: `npm run build && npx vite preview --port 4173` and uvicorn on :8000.
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: process.env.BASE || 'http://localhost:4173',
    viewport: { width: 1366, height: 900 },
    acceptDownloads: true,
    launchOptions: process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  },
})
