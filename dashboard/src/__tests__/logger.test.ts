import { describe, expect, it, vi } from "vitest";
import { createLogger } from "../utils/logger";

describe("createLogger", () => {
  it("creates a logger with all log methods", () => {
    const log = createLogger("test");
    expect(typeof log.debug).toBe("function");
    expect(typeof log.info).toBe("function");
    expect(typeof log.warn).toBe("function");
    expect(typeof log.error).toBe("function");
  });

  it("includes module name in log output", () => {
    const spy = vi.spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("MyComponent");
    log.info("hello");
    expect(spy).toHaveBeenCalledWith("[MyComponent]", "hello");
    spy.mockRestore();
  });

  it("passes extra arguments through", () => {
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const log = createLogger("test");
    log.warn("msg", { key: "val" }, 42);
    expect(spy).toHaveBeenCalledWith("[test]", "msg", { key: "val" }, 42);
    spy.mockRestore();
  });

  it("error method uses console.error", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const log = createLogger("err-test");
    log.error("fail");
    expect(spy).toHaveBeenCalledWith("[err-test]", "fail");
    spy.mockRestore();
  });
});
