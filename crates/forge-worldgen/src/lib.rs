//! # forge-worldgen
//!
//! Procedural world generation for the FORGE platform.
//! Includes Perlin noise terrain, biome rules, resource distribution,
//! object placement, and entity spawn selection.
//!
//! ## Usage
//!
//! ```rust
//! use forge_types::config::WorldConfig;
//! use forge_worldgen::WorldGenerator;
//!
//! let config = WorldConfig::default();
//! let generator = WorldGenerator::new(&config);
//!
//! use rand::SeedableRng;
//! use rand_pcg::Pcg64Mcg;
//! let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
//! let (grid, resources, objects, spawn_points) = generator.generate(&mut rng);
//! ```

pub mod biome;
pub mod entities;
pub mod noise;
pub mod objects;
pub mod resources;
pub mod terrain;

use forge_types::config::WorldConfig;
use forge_types::entity::Object;
use forge_types::grid::{Grid, Position};
use forge_types::resource::ResourceNode;
use rand_pcg::Pcg64Mcg;
use tracing::instrument;

use crate::entities::SpawnPlacer;
use crate::objects::ObjectPlacer;
use crate::resources::ResourcePlacer;
use crate::terrain::TerrainGenerator;

/// Top-level world generator that orchestrates terrain, resource, object,
/// and spawn-point generation.
///
/// All generation is deterministic: given the same [`WorldConfig`] (which
/// contains the seed), the output is identical across runs.
#[derive(Debug, Clone)]
pub struct WorldGenerator {
    /// World configuration.
    config: WorldConfig,
    /// Random seed (copied from config for convenience).
    seed: u64,
}

impl WorldGenerator {
    /// Creates a new world generator from the given configuration.
    ///
    /// The seed is taken from `config.seed`.
    #[instrument(skip_all)]
    pub fn new(config: &WorldConfig) -> Self {
        tracing::trace!(
            seed = config.seed,
            width = config.width,
            height = config.height,
            "created WorldGenerator"
        );
        Self {
            config: config.clone(),
            seed: config.seed,
        }
    }

    /// Generates a complete world: terrain grid, resource nodes, objects,
    /// and agent spawn points.
    ///
    /// * `rng` -- a seeded `Pcg64Mcg` RNG.  The caller is responsible for
    ///   creating this from a deterministic seed if reproducibility is desired.
    ///
    /// Returns `(grid, resources, objects, spawn_points)`.
    #[instrument(skip_all)]
    pub fn generate(
        &self,
        rng: &mut Pcg64Mcg,
    ) -> (Grid, Vec<ResourceNode>, Vec<Object>, Vec<Position>) {
        tracing::trace!(seed = self.seed, "WorldGenerator::generate starting");

        // 1. Generate terrain
        let terrain_gen = TerrainGenerator::new(&self.config, self.seed);
        let mut grid = Grid::new(self.config.width, self.config.height);
        terrain_gen.generate(&mut grid);

        // 2. Place resources
        let resources = ResourcePlacer::place_resources(&grid, &self.config, rng);

        // Register resource IDs on grid tiles
        for node in &resources {
            if let Some(tile) = grid.get_mut(node.position.x, node.position.y) {
                tile.resource_id = Some(node.id);
            }
        }

        // 3. Place objects
        let objects = ObjectPlacer::place_objects(&grid, &self.config, rng);

        // Register object IDs on grid tiles
        for obj in &objects {
            if let Some(tile) = grid.get_mut(obj.position.x, obj.position.y) {
                tile.object_id = Some(obj.id);
            }
        }

        // 4. Find spawn points (needs the grid with resources/objects registered
        //    so we avoid spawning on occupied tiles)
        let num_agents = 1u32; // default; callers can request more via find_spawn_points directly
        let spawn_points = SpawnPlacer::find_spawn_points(&grid, num_agents, rng);

        tracing::trace!(
            terrain_tiles = grid.len(),
            resources = resources.len(),
            objects = objects.len(),
            spawns = spawn_points.len(),
            "WorldGenerator::generate complete"
        );

        (grid, resources, objects, spawn_points)
    }

    /// Returns the seed used for this generator.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_world_generator_basic() {
        let config = WorldConfig {
            width: 32,
            height: 32,
            ..WorldConfig::default()
        };
        let gen = WorldGenerator::new(&config);
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
        let (grid, resources, objects, spawns) = gen.generate(&mut rng);

        assert_eq!(grid.width, 32);
        assert_eq!(grid.height, 32);
        assert_eq!(grid.len(), 32 * 32);
        assert!(!resources.is_empty());
        // Objects may or may not be placed depending on density + terrain
        let _ = objects;
        assert!(!spawns.is_empty());
    }

    #[test]
    fn test_full_determinism() {
        let config = WorldConfig {
            width: 32,
            height: 32,
            seed: 12345,
            ..WorldConfig::default()
        };

        // First generation
        let gen1 = WorldGenerator::new(&config);
        let mut rng1 = Pcg64Mcg::seed_from_u64(config.seed);
        let (grid1, resources1, objects1, spawns1) = gen1.generate(&mut rng1);

        // Second generation with same seed
        let gen2 = WorldGenerator::new(&config);
        let mut rng2 = Pcg64Mcg::seed_from_u64(config.seed);
        let (grid2, resources2, objects2, spawns2) = gen2.generate(&mut rng2);

        // Grids must be identical
        assert_eq!(grid1.width, grid2.width);
        assert_eq!(grid1.height, grid2.height);
        for i in 0..grid1.tiles.len() {
            assert_eq!(
                grid1.tiles[i].terrain, grid2.tiles[i].terrain,
                "terrain mismatch at tile {i}"
            );
            assert_eq!(
                grid1.tiles[i].elevation, grid2.tiles[i].elevation,
                "elevation mismatch at tile {i}"
            );
            assert_eq!(
                grid1.tiles[i].resource_id, grid2.tiles[i].resource_id,
                "resource_id mismatch at tile {i}"
            );
            assert_eq!(
                grid1.tiles[i].object_id, grid2.tiles[i].object_id,
                "object_id mismatch at tile {i}"
            );
        }

        // Resources must be identical
        assert_eq!(resources1.len(), resources2.len());
        for (a, b) in resources1.iter().zip(resources2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.position, b.position);
            assert_eq!(a.resource_type, b.resource_type);
            assert_eq!(a.quantity, b.quantity);
            assert_eq!(a.max_quantity, b.max_quantity);
        }

        // Objects must be identical
        assert_eq!(objects1.len(), objects2.len());
        for (a, b) in objects1.iter().zip(objects2.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.position, b.position);
            assert_eq!(a.object_type, b.object_type);
            assert_eq!(a.state, b.state);
        }

        // Spawn points must be identical
        assert_eq!(spawns1, spawns2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let config1 = WorldConfig {
            width: 32,
            height: 32,
            seed: 1,
            ..WorldConfig::default()
        };
        let config2 = WorldConfig {
            width: 32,
            height: 32,
            seed: 2,
            ..WorldConfig::default()
        };

        let gen1 = WorldGenerator::new(&config1);
        let mut rng1 = Pcg64Mcg::seed_from_u64(config1.seed);
        let (grid1, _, _, _) = gen1.generate(&mut rng1);

        let gen2 = WorldGenerator::new(&config2);
        let mut rng2 = Pcg64Mcg::seed_from_u64(config2.seed);
        let (grid2, _, _, _) = gen2.generate(&mut rng2);

        let mut differences = 0u32;
        for i in 0..grid1.tiles.len() {
            if grid1.tiles[i].terrain != grid2.tiles[i].terrain {
                differences += 1;
            }
        }
        assert!(
            differences > 0,
            "Different seeds should produce different worlds"
        );
    }

    #[test]
    fn test_resources_registered_on_grid() {
        let config = WorldConfig {
            width: 32,
            height: 32,
            seed: 42,
            ..WorldConfig::default()
        };
        let gen = WorldGenerator::new(&config);
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
        let (grid, resources, _, _) = gen.generate(&mut rng);

        for node in &resources {
            let tile = grid.get_pos(&node.position).unwrap();
            assert_eq!(
                tile.resource_id,
                Some(node.id),
                "Resource {} at ({}, {}) not registered on grid",
                node.id,
                node.position.x,
                node.position.y
            );
        }
    }

    #[test]
    fn test_objects_registered_on_grid() {
        let config = WorldConfig {
            width: 32,
            height: 32,
            seed: 42,
            ..WorldConfig::default()
        };
        let gen = WorldGenerator::new(&config);
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
        let (grid, _, objects, _) = gen.generate(&mut rng);

        for obj in &objects {
            let tile = grid.get_pos(&obj.position).unwrap();
            assert_eq!(
                tile.object_id,
                Some(obj.id),
                "Object {} at ({}, {}) not registered on grid",
                obj.id,
                obj.position.x,
                obj.position.y
            );
        }
    }

    #[test]
    fn test_seed_accessor() {
        let config = WorldConfig {
            seed: 999,
            ..WorldConfig::default()
        };
        let gen = WorldGenerator::new(&config);
        assert_eq!(gen.seed(), 999);
    }
}
