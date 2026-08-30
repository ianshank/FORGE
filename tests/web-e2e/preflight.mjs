// Ensure web/pkg/ holds a usable wasm bundle before the E2E server starts.
//
// This runs as the first half of Playwright's `webServer.command` rather than
// as a `globalSetup`: Playwright starts webServer as a *plugin*, and plugin
// setup is ordered ahead of global setups, so a globalSetup check would only
// fire after the server had already come up on an artifact-less web/ and the
// suite had spent its webServer timeout. Failing here fails immediately, with
// a message that says what to run.
import { access } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const GLUE = fileURLToPath(new URL("../../web/pkg/forge_wasm.js", import.meta.url));
const WASM = fileURLToPath(new URL("../../web/pkg/forge_wasm_bg.wasm", import.meta.url));

// Set when the bundle is already built and you want a fast edit/run loop, or on
// a host that cannot reach the binaryen release wasm-opt downloads. CI never
// sets it -- it builds the bundle in its own step and leaves this to rebuild.
const skipBuild = process.env.WEB_E2E_SKIP_BUILD === "1";

async function present() {
  try {
    await Promise.all([access(GLUE), access(WASM)]);
    return true;
  } catch {
    return false;
  }
}

if (!skipBuild) {
  execFileSync(`${REPO_ROOT}scripts/build_wasm_demo.sh`, { stdio: "inherit" });
}

if (!(await present())) {
  console.error(
    [
      "E2E preflight failed: web/pkg/ has no wasm bundle.",
      "",
      "  Build it with:  make wasm      (or scripts/build_wasm_demo.sh)",
      "",
      "The suite drives the real deploy artifact -- web/index.html + web/app.js",
      "+ the wasm-pack output -- so there is nothing meaningful to test without it.",
    ].join("\n"),
  );
  process.exit(1);
}
