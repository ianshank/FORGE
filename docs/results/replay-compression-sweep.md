# Replay Compression Sweep Benchmark Results

**Status:** Completed | **Obs Dim:** 920 | **Action Space:** 12 | **Steps per Episode:** 100

We ran a performance and size benchmarking sweep comparing plain JSON vs. Gzip compression (Fastest, Default, and Best levels) on a realistic v0.5 Minecraft self-play episode trajectory.

---

## Benchmark Results

Below is the side-by-side comparison of the compression variants. Measurements were captured on the local host running the unoptimized debug target (optimized release builds will have significantly lower write/read latencies).

| Variant | Size (bytes) | Size Reduction | Write Latency (ms) | Read Latency (ms) |
|---|---|---|---|---|
| **None** (Plain JSON) | 496,857 | 1.0x (Baseline) | 12.783 ms | 16.737 ms |
| **Fastest** (Gzip Lvl 1) | 5,494 | **90.4x reduction** | 14.265 ms | 17.205 ms |
| **Default** (Gzip Lvl 6) | 2,307 | **215.3x reduction** | 35.599 ms | 18.415 ms |
| **Best** (Gzip Lvl 9) | 2,307 | **215.3x reduction** | 18.476 ms | 18.184 ms |

> [!NOTE]
> - Gzip compression provides a monumental **90x to 215x storage footprint reduction** for self-play trajectories.
> - The trade-off in latency is negligible: even in an unoptimized debug build, compressing a 100-step trajectory adds less than 25 ms of write overhead, while decompression overhead is under 2 ms.
> - For long-running, multi-thousand-episode self-play iterations, enabling **Gzip (Default or Fastest)** is a critical storage safeguard, keeping the directory size within megabytes instead of gigabytes.

---

## Recommendation & Code Integration

We recommend using **`Gzip` with `NamedGzipLevel::Default` (or Fastest)** for live self-play training pipelines. The hot-reload runner and training pipeline natively support gzipped `.json.gz` files:
- The Rust runner's `TrajectoryWriter` is equipped with `.with_compression(TrajectoryCompression::Gzip, ...)` to write gzipped files directly.
- The Python `TrajectoryReader` automatically detects gzipped inputs by inspecting the file suffix and decompresses them using standard `gzip` library utilities, matching the Rust reader's custom `.json.gz` auto-detection logic.

The benchmarking test has been fully integrated into the test suite at [runner.rs](file:///c:/Users/iansh/OneDrive/Documents/FORGE/crates/forge-mc-runner/src/runner.rs#L1095) to prevent future regression and ensure cross-crate compression compatibility.
