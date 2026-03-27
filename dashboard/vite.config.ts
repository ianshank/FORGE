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
    setupFiles: [],
  },
});
