import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './browser_tests',
  use: { baseURL: 'http://127.0.0.1:7860', viewport: {width: 1280, height: 900} },
  webServer: {
    command: 'python server.py', url: 'http://127.0.0.1:7860/api/health', timeout: 30000,
    env: { STUDIO_START_WORKERS: '0', STUDIO_ROOT: '/tmp/modellab-browser-tests', STUDIO_PASSWORD: 'browser-test-password', STUDIO_COOKIE_SECURE: '0' }
  }
});
