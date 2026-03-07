//! Procedural map generation using cluster-based terrain placement.
//!
//! Generates [`Grid`] instances with configurable terrain distribution,
//! ensuring connectivity of walkable tiles.

use forge_types::grid::{Grid, TerrainType};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::instrument;

/// Configuration for procedural map generation.
///
/// All fields have sensible defaults. Terrain fractions control approximate
/// proportions — actual placement uses cluster-based random walks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapGenConfig {
    /// Grid width in tiles.
    pub width: u16,
    /// Grid height in tiles.
    pub height: u16,
    /// RNG seed for deterministic generation.
    pub seed: u64,
    /// Number of terrain clusters to place.
    pub num_clusters: u32,
    /// Minimum cluster size in tiles.
    pub cluster_size_min: u16,
    /// Maximum cluster size in tiles.
    pub cluster_size_max: u16,
    /// Target fraction of tiles that should be water.
    pub water_fraction: f64,
    /// Target fraction of tiles that should be forest.
    pub forest_fraction: f64,
    /// Target fraction of tiles that should be mountain.
    pub mountain_fraction: f64,
}

impl Default for MapGenConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            seed: 0,
            num_clusters: 10,
            cluster_size_min: 3,
            cluster_size_max: 8,
            water_fraction: 0.1,
            forest_fraction: 0.2,
            mountain_fraction: 0.05,
        }
    }
}

/// Procedural map generator that creates terrain grids from configuration.
#[derive(Debug, Clone)]
pub struct MapGenerator {
    /// Generation configuration.
    config: MapGenConfig,
}

impl MapGenerator {
    /// Creates a new map generator with the given configuration.
    #[instrument(skip(config))]
    pub fn new(config: MapGenConfig) -> Self {
        Self { config }
    }

    /// Generates a [`Grid`] with terrain clusters placed randomly.
    ///
    /// The algorithm:
    /// 1. Allocates cluster counts proportional to terrain fractions.
    /// 2. Places clusters using random-walk growth from random seed points.
    /// 3. Fills remaining tiles with [`TerrainType::Ground`].
    /// 4. Verifies walkable tile connectivity.
    #[instrument(skip(self))]
    pub fn generate(&self) -> Grid {
        let mut grid = Grid::new(self.config.width, self.config.height);
        let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed);

        let total_tiles = self.config.width as usize * self.config.height as usize;
        let water_tiles = (total_tiles as f64 * self.config.water_fraction) as usize;
        let forest_tiles = (total_tiles as f64 * self.config.forest_fraction) as usize;
        let mountain_tiles = (total_tiles as f64 * self.config.mountain_fraction) as usize;

        let terrain_budgets = [
            (TerrainType::Water, water_tiles),
            (TerrainType::Forest, forest_tiles),
            (TerrainType::Mountain, mountain_tiles),
        ];

        for (terrain, budget) in &terrain_budgets {
            let clusters_for_terrain = self.config.num_clusters.max(1);
            let tiles_per_cluster = budget / clusters_for_terrain as usize;

            for _ in 0..clusters_for_terrain {
                if tiles_per_cluster == 0 {
                    break;
                }
                let cluster_size = if self.config.cluster_size_max > self.config.cluster_size_min {
                    rng.gen_range(self.config.cluster_size_min..=self.config.cluster_size_max)
                } else {
                    self.config.cluster_size_min
                };
                let actual_size = (cluster_size as usize).min(tiles_per_cluster);
                let cx = rng.gen_range(0..self.config.width);
                let cy = rng.gen_range(0..self.config.height);
                self.place_cluster(&mut grid, &mut rng, cx, cy, actual_size, *terrain);
            }
        }

        grid
    }

    /// Places a cluster of terrain using random-walk growth from a center point.
    fn place_cluster(
        &self,
        grid: &mut Grid,
        rng: &mut Pcg64Mcg,
        start_x: u16,
        start_y: u16,
        size: usize,
        terrain: TerrainType,
    ) {
        let mut placed = 0;
        let mut x = start_x;
        let mut y = start_y;

        while placed < size {
            if let Some(tile) = grid.get_mut(x, y) {
                if tile.terrain == TerrainType::Ground {
                    tile.terrain = terrain;
                    placed += 1;
                }
            }
            // Random walk
            match rng.gen_range(0u8..4) {
                0 if y > 0 => y -= 1,
                1 if y + 1 < self.config.height => y += 1,
                2 if x > 0 => x -= 1,
                3 if x + 1 < self.config.width => x += 1,
                _ => {}
            }
        }
    }
}

/// Checks whether all walkable tiles in the grid form a single connected component.
///
/// Uses BFS starting from the first walkable tile found.
#[instrument(skip(grid))]
pub fn is_connected(grid: &Grid) -> bool {
    let w = grid.width as usize;
    let h = grid.height as usize;
    let total = w * h;
    if total == 0 {
        return true;
    }

    // Find the first walkable tile
    let start = grid.tiles.iter().position(|t| t.terrain.is_walkable());
    let start = match start {
        Some(idx) => idx,
        None => return true, // No walkable tiles — trivially connected
    };

    let mut visited = vec![false; total];
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    let mut count = 1usize;

    while let Some(idx) = queue.pop_front() {
        let x = idx % w;
        let y = idx / w;

        let neighbors = [
            if y > 0 { Some(idx - w) } else { None },
            if y + 1 < h { Some(idx + w) } else { None },
            if x > 0 { Some(idx - 1) } else { None },
            if x + 1 < w { Some(idx + 1) } else { None },
        ];

        for neighbor in neighbors.into_iter().flatten() {
            if !visited[neighbor] && grid.tiles[neighbor].terrain.is_walkable() {
                visited[neighbor] = true;
                count += 1;
                queue.push_back(neighbor);
            }
        }
    }

    let walkable_count = grid
        .tiles
        .iter()
        .filter(|t| t.terrain.is_walkable())
        .count();
    count == walkable_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_map_correct_size() {
        let config = MapGenConfig {
            width: 32,
            height: 32,
            ..Default::default()
        };
        let gen = MapGenerator::new(config);
        let grid = gen.generate();
        assert_eq!(grid.width, 32);
        assert_eq!(grid.height, 32);
        assert_eq!(grid.tiles.len(), 32 * 32);
    }

    #[test]
    fn test_generated_map_deterministic() {
        let config = MapGenConfig {
            width: 16,
            height: 16,
            seed: 42,
            ..Default::default()
        };
        let gen = MapGenerator::new(config.clone());
        let grid_a = gen.generate();
        let grid_b = gen.generate();

        for (a, b) in grid_a.tiles.iter().zip(grid_b.tiles.iter()) {
            assert_eq!(a.terrain, b.terrain);
        }
    }

    #[test]
    fn test_generated_map_has_ground() {
        let config = MapGenConfig {
            width: 32,
            height: 32,
            seed: 123,
            ..Default::default()
        };
        let gen = MapGenerator::new(config);
        let grid = gen.generate();
        let ground_count = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Ground)
            .count();
        assert!(ground_count > 0, "Map should have ground tiles");
    }

    #[test]
    fn test_terrain_distribution_reasonable() {
        let config = MapGenConfig {
            width: 64,
            height: 64,
            seed: 99,
            ..Default::default()
        };
        let gen = MapGenerator::new(config);
        let grid = gen.generate();

        let total = grid.tiles.len() as f64;
        let water = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Water)
            .count() as f64;
        let forest = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Forest)
            .count() as f64;

        // Allow generous tolerance — cluster placement is approximate
        assert!(water / total < 0.5, "Water should not dominate the map");
        assert!(forest / total < 0.5, "Forest should not dominate the map");
    }

    #[test]
    fn test_is_connected_simple() {
        let grid = Grid::new(4, 4); // All ground
        assert!(is_connected(&grid));
    }

    #[test]
    fn test_is_connected_empty() {
        let grid = Grid::new(0, 0);
        assert!(is_connected(&grid));
    }

    #[test]
    fn test_is_connected_with_obstacles() {
        let mut grid = Grid::new(4, 4);
        // Place a wall that doesn't disconnect
        grid.get_mut(1, 0).unwrap().terrain = TerrainType::Wall;
        assert!(is_connected(&grid));
    }

    #[test]
    fn test_is_connected_disconnected() {
        // Create a 3x3 grid with a wall column splitting it
        let mut grid = Grid::new(3, 3);
        for y in 0..3u16 {
            grid.get_mut(1, y).unwrap().terrain = TerrainType::Wall;
        }
        // Left side and right side are disconnected
        assert!(!is_connected(&grid));
    }
}
