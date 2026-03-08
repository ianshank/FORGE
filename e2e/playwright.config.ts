import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "**/*.spec.ts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["html", { open: "never" }]],
  timeout: 30_000,

  use: {
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "dashboard",
      testDir: "./dashboard",
      use: {
        ...devices["Desktop Chrome"],
        baseURL: "http://localhost:5173",
      },
    },
    {
      name: "demo-ui",
      testDir: "./demo-ui",
      use: {
        ...devices["Desktop Chrome"],
        baseURL: "http://localhost:8000",
      },
    },
  ],

  webServer: [
    {
      command: "npm run dev",
      cwd: "../dashboard",
      port: 5173,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command:
        "python -m uvicorn demo_ui.backend.main:app --host 127.0.0.1 --port 8000",
      cwd: "..",
      port: 8000,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
