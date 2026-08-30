import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

/**
 * End-to-end coverage for the in-browser WASM demo.
 *
 * Everything here runs against the real deploy artifact. There are no mocks:
 * the page loads `pkg/forge_wasm.js`, instantiates the module, and every
 * assertion below is downstream of real `forge-core` simulation.
 */

/** Wait for `main()` in app.js to finish wiring the page up. */
async function waitForReady(page: Page): Promise<void> {
  await expect(page.locator("#status")).toHaveText("ready", { timeout: 30_000 });
}

/** Read the tick counter as a number. */
async function tick(page: Page): Promise<number> {
  return Number(await page.locator("#tick").innerText());
}

test.describe("WASM demo", () => {
  test("loads the module and renders a grid", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    // `setStatus("ready")` is only reached after `await init()`, the
    // ForgeWasmEnv constructor, readActionCount() and resetEpisode() have all
    // succeeded -- so this is the broadest single signal that the wasm
    // boundary works. The grid assertions confirm real render output rather
    // than the "loading…" placeholder.
    const grid = await page.locator("#grid").innerText();
    expect(grid).not.toBe("loading…");
    expect(grid).toContain("A"); // at least one agent
    expect(grid.split("\n").length).toBeGreaterThan(1);
  });

  test("shows the action-space size the module reports", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    const shown = Number(await page.locator("#actions").innerText());
    expect(Number.isInteger(shown)).toBe(true);
    expect(shown).toBeGreaterThan(0);

    // Cross-check against the module rather than pinning a literal: the count
    // is config-derived, and hardcoding it here would bake a value the config
    // is allowed to change (CHARTER Invariant 5).
    const reported = await page.evaluate(async () => {
      // @ts-expect-error -- generated wasm-pack bundle; built with
      // --no-typescript, so it ships no .d.ts for TS to resolve.
      const mod = await import("/pkg/forge_wasm.js");
      await mod.default();
      const env = new mod.ForgeWasmEnv("");
      return JSON.parse(env.action_space_json()).n as number;
    });
    expect(shown).toBe(reported);
  });

  test("Step advances the tick and reports a reward", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);
    expect(await tick(page)).toBe(0);

    await page.locator("#step").click();
    await expect(page.locator("#tick")).toHaveText("1");
    await page.locator("#step").click();
    await expect(page.locator("#tick")).toHaveText("2");

    // Deliberately not asserting the grid changed: a Noop action legitimately
    // leaves the world identical, and the demo picks actions at random.
    const reward = await page.locator("#reward").innerText();
    expect(reward).not.toBe("—");
    expect(Number.isNaN(Number(reward))).toBe(false);
  });

  test("Play advances the simulation and Pause stops it", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    await page.locator("#play").click();
    await expect(page.locator("#status")).toHaveText("playing");
    await expect(page.locator("#play")).toBeDisabled();
    await expect(page.locator("#pause")).toBeEnabled();

    // The window is many multiples of app.js's PLAY_INTERVAL_MS. That constant
    // is deliberately not imported: coupling the timing assertions to it is
    // how this kind of test turns flaky.
    await expect.poll(() => tick(page), { timeout: 5_000 }).toBeGreaterThan(2);

    await page.locator("#pause").click();
    await expect(page.locator("#status")).toHaveText("paused");
    const stopped = await tick(page);
    await page.waitForTimeout(1_000);
    expect(await tick(page)).toBe(stopped);
  });

  test("the same seed reproduces the same world", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    // Reset with the box empty -> a random seed, which the page reports.
    await page.locator("#reset").click();
    const seed = await page.locator("#seed-used").innerText();
    expect(seed).toMatch(/^\d+$/);
    const first = await page.locator("#grid").innerText();

    // Same seed, same world. Only the post-reset grid is compared: the demo
    // steps with Math.random(), so comparing after N steps would be flaky.
    // Action-sequence determinism is covered by the wasm-runtime test in
    // crates/forge-wasm/tests/wasm_bindings.rs.
    await page.locator("#seed").fill(seed);
    await page.locator("#reset").click();
    await expect(page.locator("#seed-used")).toHaveText(seed);
    expect(await page.locator("#grid").innerText()).toBe(first);

    // And across a fresh module instantiation, which is the claim the demo
    // actually makes to a visitor.
    await page.reload();
    await waitForReady(page);
    await page.locator("#seed").fill(seed);
    await page.locator("#reset").click();
    await expect(page.locator("#seed-used")).toHaveText(seed);
    expect(await page.locator("#grid").innerText()).toBe(first);
  });

  test("reset takes a BigInt seed and rejects a Number", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    // The regression pin for the defect this suite exists to catch. It is
    // unreachable from Rust: `Option<u64>` is a plain u64 there, and the
    // coercion lives only in the wasm-bindgen-generated JS glue, which
    // forwards the value untouched to an i64 wasm parameter.
    const result = await page.evaluate(async () => {
      // @ts-expect-error -- generated wasm-pack bundle; no .d.ts (--no-typescript).
      const mod = await import("/pkg/forge_wasm.js");
      await mod.default();
      const env = new mod.ForgeWasmEnv("");

      let bigintOk = false;
      try {
        env.reset(42n);
        bigintOk = true;
      } catch {
        bigintOk = false;
      }

      let numberThrew = false;
      try {
        env.reset(42);
      } catch {
        numberThrew = true;
      }

      return { bigintOk, numberThrew };
    });

    expect(result.bigintOk).toBe(true);
    // Asserting only that it throws. The message is engine-specific and
    // differs between a --dev build (wasm-bindgen's own bigint assert) and the
    // --release build CI and the deploy workflows produce, where that assert is
    // compiled out and the JS API's own TypeError surfaces instead.
    expect(result.numberThrew).toBe(true);
  });

  test("an invalid config throws a readable Error, not a wasm trap", async ({ page }) => {
    await page.goto("/");
    await waitForReady(page);

    const message = await page.evaluate(async () => {
      // @ts-expect-error -- generated wasm-pack bundle; no .d.ts (--no-typescript).
      const mod = await import("/pkg/forge_wasm.js");
      await mod.default();
      try {
        new mod.ForgeWasmEnv("{ not json");
        return null;
      } catch (err) {
        return String((err as Error).message ?? err);
      }
    });

    expect(message).not.toBeNull();
    expect(message).toContain("ForgeConfig");
    // The pre-fix behaviour: a panic crossing the boundary as an opaque trap.
    expect(message).not.toContain("unreachable");
  });

  test("no console or page errors through load, reset and stepping", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg: ConsoleMessage) => {
      if (msg.type() === "error" && !msg.text().includes("favicon")) {
        errors.push(msg.text());
      }
    });
    page.on("pageerror", (err: Error) => errors.push(err.message));

    await page.goto("/");
    await waitForReady(page);
    await page.locator("#reset").click();
    for (let i = 0; i < 5; i += 1) {
      await page.locator("#step").click();
    }

    // A Rust panic crossing the wasm boundary arrives as a pageerror, so this
    // is what makes console_error_panic_hook worth having.
    expect(errors).toEqual([]);
  });
});
