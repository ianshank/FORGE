import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, getConfig, _resetConfigCache } from "../config/environment";

describe("DashboardConfig", () => {
  it("getConfig returns a config object with all required fields", () => {
    _resetConfigCache();
    const config = getConfig();
    expect(config.wsUrl).toBeDefined();
    expect(config.apiBaseUrl).toBeDefined();
    expect(config.demoApiBaseUrl).toBeDefined();
    expect(config.metricsPollingInterval).toBeGreaterThan(0);
    expect(config.maxTraceEntries).toBeGreaterThan(0);
    expect(typeof config.showGridLines).toBe("boolean");
    expect(config.cellSize).toBeGreaterThan(0);
    expect(config.maxReconnectAttempts).toBeGreaterThan(0);
  });

  it("getConfig returns the same cached instance on subsequent calls", () => {
    _resetConfigCache();
    const a = getConfig();
    const b = getConfig();
    expect(a).toBe(b);
  });

  it("_resetConfigCache clears the cache", () => {
    const a = getConfig();
    _resetConfigCache();
    const b = getConfig();
    expect(a).not.toBe(b);
  });

  it("DEFAULT_CONFIG has valid default values", () => {
    expect(DEFAULT_CONFIG.metricsPollingInterval).toBeGreaterThanOrEqual(500);
    expect(DEFAULT_CONFIG.cellSize).toBeGreaterThanOrEqual(2);
    expect(DEFAULT_CONFIG.cellSize).toBeLessThanOrEqual(64);
  });

  it("history polling config has valid clamped defaults", () => {
    expect(DEFAULT_CONFIG.trainingHistoryInterval).toBeGreaterThanOrEqual(500);
    expect(DEFAULT_CONFIG.runsInterval).toBeGreaterThanOrEqual(500);
    expect(DEFAULT_CONFIG.historyLimit).toBeGreaterThanOrEqual(1);
    expect(DEFAULT_CONFIG.historyLimit).toBeLessThanOrEqual(10000);
  });

  it("default wsUrl points to localhost", () => {
    expect(DEFAULT_CONFIG.wsUrl).toContain("localhost");
  });

  it("default apiBaseUrl points to localhost", () => {
    expect(DEFAULT_CONFIG.apiBaseUrl).toContain("localhost");
  });

  it("default demoApiBaseUrl points to localhost", () => {
    expect(DEFAULT_CONFIG.demoApiBaseUrl).toContain("localhost");
  });
});
