#!/usr/bin/env bash
# Install the pinned wasm-pack release to /usr/local/bin.
#
# WASM_PACK_VERSION is supplied by the caller -- a step-level `env:` in
# gh-pages.yml (the canonical pin) and hf-space.yml, and a workflow-level `env:`
# in ci.yml. Deliberately NOT defaulted here, so this script never becomes a
# fourth copy of the version literal that could drift (CHARTER Invariant 5).
# scripts/check_pinned_config_consistency.py cross-checks every copy.
#
# Why a shell install rather than jetli/wasm-pack-action@v0.4.0: that action is
# node16, with no newer tag and no node24 support anywhere including its master
# branch -- there is no version bump that fixes it. A plain shell step has no
# Node runtime at all, so it is permanently immune to that class of deprecation.
set -euo pipefail

: "${WASM_PACK_VERSION:?WASM_PACK_VERSION must be set by the caller}"

asset="wasm-pack-v${WASM_PACK_VERSION}-x86_64-unknown-linux-musl"

# -L is load-bearing: rustwasm/wasm-pack now redirects to wasm-bindgen/wasm-pack,
# so dropping it turns the download into a 404. -f makes a moved/missing release
# fail loudly here instead of downstream in `tar` on the error body.
curl -sSLf "https://github.com/rustwasm/wasm-pack/releases/download/v${WASM_PACK_VERSION}/${asset}.tar.gz" \
  | tar xz -C /tmp

# sudo: GitHub-hosted runners execute steps as a non-root user, and
# /usr/local/bin is root-owned.
sudo install -m 755 "/tmp/${asset}/wasm-pack" /usr/local/bin/wasm-pack
wasm-pack --version
