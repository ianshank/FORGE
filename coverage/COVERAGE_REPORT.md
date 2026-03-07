# FORGE Test Coverage Report

**Generated:** 2026-03-07
**Tool:** cargo-tarpaulin v0.35.2
**Overall Coverage:** 91.13% (1809/1985 lines)

## Per-Crate Coverage

### forge-types (Shared Types & Config)
| File | Covered | Total | % |
|------|---------|-------|---|
| action.rs | 38 | 38 | 100.0% |
| config.rs | 6 | 6 | 100.0% |
| entity.rs | 52 | 53 | 98.1% |
| grid.rs | 66 | 71 | 93.0% |
| observation.rs | 31 | 31 | 100.0% |
| resource.rs | 72 | 72 | 100.0% |
| task.rs | 4 | 4 | 100.0% |
| validation.rs | 44 | 54 | 81.5% |
| **Crate Total** | **313** | **329** | **95.1%** |

### forge-core (Simulation Engine)
| File | Covered | Total | % |
|------|---------|-------|---|
| combat.rs | 42 | 45 | 93.3% |
| communication.rs | 30 | 30 | 100.0% |
| crafting.rs | 27 | 30 | 90.0% |
| day_night.rs | 21 | 21 | 100.0% |
| physics.rs | 90 | 93 | 96.8% |
| resource.rs | 36 | 38 | 94.7% |
| rng.rs | 31 | 33 | 93.9% |
| systems.rs | 85 | 87 | 97.7% |
| visibility.rs | 60 | 61 | 98.4% |
| world.rs | 179 | 182 | 98.4% |
| **Crate Total** | **601** | **620** | **96.9%** |

### forge-worldgen (Procedural Generation)
| File | Covered | Total | % |
|------|---------|-------|---|
| biome.rs | 24 | 24 | 100.0% |
| entities.rs | 29 | 29 | 100.0% |
| lib.rs | 23 | 24 | 95.8% |
| noise.rs | 66 | 69 | 95.7% |
| objects.rs | 26 | 26 | 100.0% |
| resources.rs | 22 | 22 | 100.0% |
| terrain.rs | 33 | 39 | 84.6% |
| **Crate Total** | **223** | **233** | **95.7%** |

### forge-task (Task DSL & Curriculum)
| File | Covered | Total | % |
|------|---------|-------|---|
| composer.rs | 39 | 43 | 90.7% |
| curriculum.rs | 59 | 63 | 93.7% |
| difficulty.rs | 34 | 35 | 97.1% |
| evaluator.rs | 31 | 31 | 100.0% |
| generator.rs | 161 | 189 | 85.2% |
| predicate.rs | 105 | 133 | 78.9% |
| **Crate Total** | **429** | **494** | **86.8%** |

### forge-agent (MCTS & Baselines)
| File | Covered | Total | % |
|------|---------|-------|---|
| baselines.rs | 54 | 63 | 85.7% |
| forward_model.rs | 12 | 12 | 100.0% |
| mcts/policy.rs | 16 | 16 | 100.0% |
| mcts/search.rs | 43 | 52 | 82.7% |
| mcts/tree.rs | 65 | 65 | 100.0% |
| **Crate Total** | **190** | **208** | **91.3%** |

### forge-wasm (WebAssembly)
| File | Covered | Total | % |
|------|---------|-------|---|
| lib.rs | 53 | 53 | 100.0% |
| **Crate Total** | **53** | **53** | **100.0%** |

## Excluded Crates

- **forge-python**: PyO3 bindings require Python runtime; excluded from tarpaulin coverage
- **forge-bench**: Benchmark crate; no business logic to cover

## Gap Analysis

All crates exceed the 80% coverage target:

| Crate | Coverage | Status |
|-------|----------|--------|
| forge-types | 95.1% | PASS |
| forge-core | 96.9% | PASS |
| forge-worldgen | 95.7% | PASS |
| forge-task | 86.8% | PASS |
| forge-agent | 91.3% | PASS |
| forge-wasm | 100.0% | PASS |
| **Overall** | **91.13%** | **PASS** |

### Areas Below 90% (Improvement Opportunities)

1. **forge-task/predicate.rs** (78.9%): Some predicate evaluation branches untested
2. **forge-task/generator.rs** (85.2%): Edge cases in task generation
3. **forge-agent/baselines.rs** (85.7%): Some baseline agent paths untested
4. **forge-agent/mcts/search.rs** (82.7%): Deep MCTS search paths
5. **forge-types/validation.rs** (81.5%): Some validation boundary conditions
6. **forge-worldgen/terrain.rs** (84.6%): Terrain generation edge cases

### Summary

The FORGE codebase achieves **91.13% overall line coverage**, significantly exceeding
the 80% target. All 6 measured crates individually exceed 80%. The strongest coverage
is in forge-wasm (100%) and forge-core (96.9%), reflecting the thorough testing of the
core simulation engine. The lowest individual file coverage is forge-task/predicate.rs
at 78.9%, which contains complex branching logic for 10 different predicate types.
