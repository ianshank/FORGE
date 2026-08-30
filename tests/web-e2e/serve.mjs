// Static file server for the WASM demo's E2E run.
//
// Serves web/ byte-for-byte -- no transform, no bundling -- so the browser sees
// exactly the files gh-pages.yml uploads and hf-space.yml mirrors.
//
// Node built-ins only, deliberately. `vite preview` would mean adding a bundler
// to a project that has none, and the MIME table below is the whole reason a
// generic static server is risky here: wasm-pack's `--target web` glue calls
// WebAssembly.instantiateStreaming, which requires `application/wasm`. On any
// other content type it falls back to arrayBuffer()+instantiate and emits a
// console.warn -- not an error -- so a wrong MIME type would silently degrade
// the demo while the "no console errors" spec stayed green.
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { join, normalize, extname, sep } from "node:path";

const ROOT = fileURLToPath(new URL("../../web", import.meta.url));
const DEFAULT_PORT = 4174; // dashboard/ uses 4173; keep them distinct.
const PORT = Number(process.env.WEB_E2E_PORT ?? DEFAULT_PORT);
const HOST = process.env.WEB_E2E_HOST ?? "127.0.0.1";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".wasm": "application/wasm",
};

const server = createServer(async (req, res) => {
  let file;
  try {
    const url = new URL(req.url ?? "/", `http://${HOST}:${PORT}`);
    const pathname = decodeURIComponent(url.pathname);
    file = join(ROOT, normalize(pathname));
  } catch {
    res.writeHead(400).end();
    return;
  }

  // Containment check: normalize() collapses `..`, but a crafted path could
  // still resolve outside ROOT. Compare against ROOT + separator so a sibling
  // directory sharing the prefix (web-e2e vs web) cannot slip through.
  if (file !== ROOT && !file.startsWith(ROOT + sep)) {
    res.writeHead(403).end();
    return;
  }

  try {
    const info = await stat(file);
    if (info.isDirectory()) file = join(file, "index.html");
    await stat(file);
  } catch {
    res.writeHead(404).end();
    return;
  }

  res.writeHead(200, {
    "content-type": MIME[extname(file)] ?? "application/octet-stream",
    // The demo is rebuilt for every run; a cached .wasm would mask a bad build.
    "cache-control": "no-store",
  });
  createReadStream(file).pipe(res);
});

server.listen(PORT, HOST, () => {
  console.log(`serving ${ROOT} on http://${HOST}:${PORT}`);
});
