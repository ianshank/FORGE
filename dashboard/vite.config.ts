/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PROXY_TARGET =
  process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8080";
const WS_PROXY_TARGET =
  process.env.VITE_WS_PROXY_TARGET ?? "ws://localhost:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": API_PROXY_TARGET,
      "/ws": {
        target: WS_PROXY_TARGET,
        ws: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    // Vitest owns src/*.test|spec; Playwright (e2e/) is a separate runner.
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["e2e/**", "node_modules", "dist"],
    coverage: {
      provider: "v8",
      reporter: ["text", "text-summary", "html"],
      // Only the application source is graded — exclude entry/bootstrap,
      // type-only modules, the test harness, and generated assets.
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/main.tsx",
        "src/test/**",
        "src/**/*.d.ts",
        "src/types/**",
        "src/**/__tests__/**",
      ],
      thresholds: {
        statements: 85,
        branches: 85,
        functions: 85,
        lines: 85,
      },
    },
  },
});
