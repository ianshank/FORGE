//! Entity/agent spawn point selection.
//!
//! [`SpawnPlacer`] finds valid spawn locations on walkable tiles that are not
//! already occupied by resources or objects.  Spawn points are spread out to
//! avoid clustering.

use forge_types::grid::{Grid, Position};
use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;
use tracing::instrument;

/// Minimum Manhattan distance between spawn points.
///
/// This prevents agents from spawning on top of each other and gives them
/// some room to manoeuvre at the start of an episode.
const MIN_SPAWN_DISTANCE: u32 = 3;

/// Finds valid spawn locations for agents on the grid.
#[derive(Debug)]
pub struct SpawnPlacer;

impl SpawnPlacer {
    /// Finds `num_agents` spawn points on walkable, unoccupied tiles.
    ///
    /// The algorithm:
    /// 1. Collect all walkable tiles that have no resource or object.
    /// 2. Shuffle them deterministically with `rng`.
    /// 3. Greedily pick tiles that are at least [`MIN_SPAWN_DISTANCE`] apart.
    /// 4. If not enough spread-out tiles exist, relax the distance constraint
    ///    and pick from the remaining candidates.
    ///
    /// Returns up to `num_agents` positions.  If fewer walkable tiles exist
    /// than requested agents, as many as possible are returned.
    #[instrument(skip_all)]
    pub fn find_spawn_points(grid: &Grid, num_agents: u32, rng: &mut Pcg64Mcg) -> Vec<Position> {
        let num = num_agents as usize;
        tracing::trace!(num_agents, "finding spawn points");

        // Collect candidate tiles
        let mut candidates: Vec<Position> = Vec::new();
        for y in 0..grid.height {
            for x in 0..grid.width {
                if let Some(tile) = grid.get(x, y) {
                    if tile.terrain.is_walkable()
                        && tile.resource_id.is_none()
                        && tile.object_id.is_none()
                        && tile.agent_id.is_none()
                    {
                        candidates.push(Position::new(x, y));
                    }
                }
            }
        }

        tracing::trace!(candidates = candidates.len(), "walkable candidates found");

        // Shuffle for randomness
        candidates.shuffle(rng);

        // Greedy selection with distance constraint
        let mut selected: Vec<Position> = Vec::with_capacity(num);
        let mut remaining: Vec<Position> = Vec::new();

        for pos in candidates {
            if selected.len() >= num {
                break;
            }

            let far_enough = selected
                .iter()
                .all(|s| pos.manhattan_distance(s) >= MIN_SPAWN_DISTANCE);

            if far_enough {
                selected.push(pos);
            } else {
                remaining.push(pos);
            }
        }

        // If we still need more, relax the distance constraint
        if selected.len() < num {
            for pos in remaining {
                if selected.len() >= num {
                    break;
                }
                selected.push(pos);
            }
        }

        tracing::trace!(count = selected.len(), "spawn points selected");
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainGenerator;
    use forge_types::config::WorldConfig;
    use rand::SeedableRng;

    fn make_grid(seed: u64) -> (Grid, WorldConfig) {
        let config = WorldConfig {
            width: 32,
            height: 32,
            seed,
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);
        (grid, config)
    }

    #[test]
    fn test_find_spawn_points() {
        let (grid, _config) = make_grid(42);
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let spawns = SpawnPlacer::find_spawn_points(&grid, 4, &mut rng);
        assert_eq!(spawns.len(), 4, "Should find 4 spawn points");
    }

    #[test]
    fn test_spawns_on_walkable_tiles() {
        let (grid, _config) = make_grid(55);
        let mut rng = Pcg64Mcg::seed_from_u64(55);
        let spawns = SpawnPlacer::find_spawn_points(&grid, 8, &mut rng);

        for pos in &spawns {
            let tile = grid.get_pos(pos).unwrap();
            assert!(
                tile.terrain.is_walkable(),
                "Spawn at ({}, {}) is on non-walkable {:?}",
                pos.x,
                pos.y,
                tile.terrain
            );
        }
    }

    #[test]
    fn test_determinism() {
        let (grid, _config) = make_grid(99);

        let mut rng1 = Pcg64Mcg::seed_from_u64(99);
        let spawns1 = SpawnPlacer::find_spawn_points(&grid, 4, &mut rng1);

        let mut rng2 = Pcg64Mcg::seed_from_u64(99);
        let spawns2 = SpawnPlacer::find_spawn_points(&grid, 4, &mut rng2);

        assert_eq!(spawns1, spawns2);
    }

    #[test]
    fn test_spread_out() {
        let (grid, _config) = make_grid(77);
        let mut rng = Pcg64Mcg::seed_from_u64(77);
        let spawns = SpawnPlacer::find_spawn_points(&grid, 4, &mut rng);

        // At least the first few should respect minimum distance
        // (unless the grid is very small / mostly non-walkable)
        if spawns.len() >= 2 {
            let mut some_distant = false;
            for i in 0..spawns.len() {
                for j in (i + 1)..spawns.len() {
                    if spawns[i].manhattan_distance(&spawns[j]) >= MIN_SPAWN_DISTANCE {
                        some_distant = true;
                    }
                }
            }
            assert!(some_distant, "Spawn points should be spread apart");
        }
    }

    #[test]
    fn test_handles_more_agents_than_tiles() {
        // Tiny grid
        let config = WorldConfig {
            width: 8,
            height: 8,
            seed: 42,
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, config.seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let mut rng = Pcg64Mcg::seed_from_u64(42);
        // Request way more agents than tiles
        let spawns = SpawnPlacer::find_spawn_points(&grid, 1000, &mut rng);
        // Should return as many as possible without panicking
        assert!(spawns.len() <= 64);
    }

    #[test]
    fn test_spawn_more_agents_than_tiles() {
        // Create the smallest possible grid (2x2 = 4 tiles)
        let config = WorldConfig {
            width: 2,
            height: 2,
            seed: 42,
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, config.seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let mut rng = Pcg64Mcg::seed_from_u64(42);
        // Request far more agents than total tiles
        let spawns = SpawnPlacer::find_spawn_points(&grid, 100, &mut rng);

        // Should not panic and should return at most the number of walkable tiles
        let walkable_count = (0..grid.height)
            .flat_map(|y| (0..grid.width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                grid.get(x, y)
                    .map(|t| {
                        t.terrain.is_walkable()
                            && t.resource_id.is_none()
                            && t.object_id.is_none()
                            && t.agent_id.is_none()
                    })
                    .unwrap_or(false)
            })
            .count();
        assert!(
            spawns.len() <= walkable_count,
            "spawn count {} should not exceed walkable tiles {}",
            spawns.len(),
            walkable_count
        );
        // Also verify all positions are unique
        let unique: std::collections::HashSet<(u16, u16)> =
            spawns.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(
            unique.len(),
            spawns.len(),
            "spawn positions should be unique"
        );
    }

    #[test]
    fn test_unique_positions() {
        let (grid, _config) = make_grid(88);
        let mut rng = Pcg64Mcg::seed_from_u64(88);
        let spawns = SpawnPlacer::find_spawn_points(&grid, 8, &mut rng);

        let unique: std::collections::HashSet<(u16, u16)> =
            spawns.iter().map(|p| (p.x, p.y)).collect();
        assert_eq!(
            unique.len(),
            spawns.len(),
            "All spawn positions should be unique"
        );
    }
}
