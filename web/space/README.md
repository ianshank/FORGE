---
title: FORGE Live Demo
emoji: ⚒️
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Deterministic RL gridworld simulator running in your browser
tags:
  - reinforcement-learning
  - simulation
  - rust
  - webassembly
  - gridworld
---

## FORGE — Live In-Browser Demo

[FORGE](https://github.com/ianshank/FORGE) (Fast Open-source Runtime for
Generalist Environments) is a high-performance simulation platform for
training and evaluating AI agents, built in Rust with Python, WebAssembly,
and REST surfaces. This Space runs the **full simulation client-side** via
WebAssembly — there is no server behind this page.

## What you're looking at

A procedurally generated gridworld with resources, crafting, combat, and a
day/night cycle, rendered as ASCII. The agent takes random actions from the
environment's action space; the same engine runs 130K+ steps/second from
Python (see `benchmarks/baselines/cloud_agent/pyo3_step.json`) and byte-identically reproduces any episode from its seed.

## Controls

- **Reset** — start a new episode (fresh procedural world)
- **Step** — advance the simulation one tick
- **▶ Play / ⏸ Pause** — auto-step the environment

## Under the hood

- `crates/forge-wasm` exposes the deterministic `forge-core` engine through
  `wasm-bindgen` (`ForgeWasmEnv`: `reset` / `step` / `render_ascii`).
- Fixed-point arithmetic (`fixed`) + `rand_pcg` RNG make every episode
  reproducible from its seed — in the browser, from Python, or natively.
- The Python surface implements Gymnasium / PettingZoo APIs; see the
  [repository](https://github.com/ianshank/FORGE) for training pipelines,
  the MCTS/MuZero stack, and the live-Minecraft self-improvement loop.

This Space is synced automatically from the GitHub repository by the
`hf-space.yml` workflow.
