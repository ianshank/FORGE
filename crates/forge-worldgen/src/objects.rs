//! Object placement for interactive world objects.
//!
//! [`ObjectPlacer`] distributes interactive objects (boulders, doors, switches,
//! crafting stations, etc.) across the generated grid.  Placement is governed
//! by terrain compatibility and the configured entity budget.

use forge_types::config::WorldConfig;
use forge_types::entity::{Object, ObjectState, ObjectType};
use forge_types::grid::{Grid, Position, TerrainType};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

/// Fixed-point 1.0 representation (16 fractional bits).
const FP_ONE: i32 = 65536;

/// Describes a single object placement rule.
#[derive(Debug, Clone)]
struct ObjectRule {
    /// The type of object to place.
    object_type: ObjectType,
    /// Terrain types where this object is allowed.
    allowed_terrain: &'static [TerrainType],
    /// Base probability of placing this object on an eligible tile.
    base_probability: f64,
    /// Mass in fixed-point (65536 = 1.0).
    mass: i32,
    /// Durability in fixed-point.
    durability: i32,
    /// Initial state.
    initial_state: ObjectState,
}

/// Placement rules for all object types.
const OBJECT_RULES: &[ObjectRule] = &[
    ObjectRule {
        object_type: ObjectType::Boulder,
        allowed_terrain: &[TerrainType::Ground, TerrainType::Mountain],
        base_probability: 0.01,
        mass: FP_ONE * 10,
        durability: FP_ONE * 20,
        initial_state: ObjectState::Active,
    },
    ObjectRule {
        object_type: ObjectType::CraftingStation,
        allowed_terrain: &[TerrainType::Ground],
        base_probability: 0.003,
        mass: FP_ONE * 5,
        durability: FP_ONE * 50,
        initial_state: ObjectState::Active,
    },
    ObjectRule {
        object_type: ObjectType::Container,
        allowed_terrain: &[TerrainType::Ground, TerrainType::Sand],
        base_probability: 0.005,
        mass: FP_ONE * 2,
        durability: FP_ONE * 10,
        initial_state: ObjectState::Closed,
    },
    ObjectRule {
        object_type: ObjectType::Torch,
        allowed_terrain: &[TerrainType::Ground, TerrainType::Forest],
        base_probability: 0.004,
        mass: FP_ONE,
        durability: FP_ONE * 5,
        initial_state: ObjectState::Active,
    },
];

/// Places interactive objects on the world grid.
#[derive(Debug)]
pub struct ObjectPlacer;

impl ObjectPlacer {
    /// Places objects throughout the grid.
    ///
    /// * `grid` -- the generated terrain grid (read-only).
    /// * `config` -- world configuration (provides entity budget via `max_entities`).
    /// * `rng` -- seeded RNG for deterministic placement.
    ///
    /// Returns a vector of [`Object`]s.  The number of placed objects is
    /// capped at `config.max_entities / 2` so that there is room for agents
    /// and other dynamic entities.
    pub fn place_objects(grid: &Grid, config: &WorldConfig, rng: &mut Pcg64Mcg) -> Vec<Object> {
        // Reserve at most half the entity budget for objects.
        let max_objects = (config.max_entities / 2) as usize;
        let density_factor = config.resource_density as f64;
        let mut objects: Vec<Object> = Vec::new();

        tracing::trace!(max_objects, density_factor, "placing objects");

        for y in 0..grid.height {
            for x in 0..grid.width {
                if objects.len() >= max_objects {
                    break;
                }

                let tile = match grid.get(x, y) {
                    Some(t) => t,
                    None => continue,
                };

                // Skip tiles already occupied by a resource
                if tile.resource_id.is_some() {
                    continue;
                }

                for rule in OBJECT_RULES {
                    if !rule.allowed_terrain.contains(&tile.terrain) {
                        continue;
                    }

                    let probability = rule.base_probability * density_factor;
                    let roll: f64 = rng.gen();
                    if roll >= probability {
                        continue;
                    }

                    let id = objects.len() as u32;
                    objects.push(Object {
                        id,
                        position: Position::new(x, y),
                        object_type: rule.object_type,
                        mass: rule.mass,
                        durability: rule.durability,
                        state: rule.initial_state,
                    });

                    // Only one object per tile
                    break;
                }
            }

            if objects.len() >= max_objects {
                break;
            }
        }

        tracing::trace!(count = objects.len(), "object placement complete");
        objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainGenerator;
    use rand::SeedableRng;

    fn make_grid_and_config(seed: u64) -> (Grid, WorldConfig) {
        let config = WorldConfig {
            width: 32,
            height: 32,
            seed,
            resource_density: 0.8,
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);
        (grid, config)
    }

    #[test]
    fn test_objects_placed() {
        let config = WorldConfig {
            width: 64,
            height: 64,
            seed: 42,
            resource_density: 1.0,
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, config.seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let objects = ObjectPlacer::place_objects(&grid, &config, &mut rng);
        // With full density on a 64x64 grid, we expect some objects
        assert!(!objects.is_empty(), "Should place at least some objects");
    }

    #[test]
    fn test_determinism() {
        let (grid, config) = make_grid_and_config(99);

        let mut rng1 = Pcg64Mcg::seed_from_u64(99);
        let objects1 = ObjectPlacer::place_objects(&grid, &config, &mut rng1);

        let mut rng2 = Pcg64Mcg::seed_from_u64(99);
        let objects2 = ObjectPlacer::place_objects(&grid, &config, &mut rng2);

        assert_eq!(objects1.len(), objects2.len());
        for (a, b) in objects1.iter().zip(objects2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.object_type, b.object_type);
            assert_eq!(a.state, b.state);
        }
    }

    #[test]
    fn test_terrain_compatibility() {
        let (grid, config) = make_grid_and_config(55);
        let mut rng = Pcg64Mcg::seed_from_u64(55);
        let objects = ObjectPlacer::place_objects(&grid, &config, &mut rng);

        for obj in &objects {
            let tile = grid.get_pos(&obj.position).unwrap();
            let matching_rule = OBJECT_RULES
                .iter()
                .find(|r| r.object_type == obj.object_type)
                .expect("object type should have a rule");
            assert!(
                matching_rule.allowed_terrain.contains(&tile.terrain),
                "Object {:?} placed on {:?}, expected one of {:?}",
                obj.object_type,
                tile.terrain,
                matching_rule.allowed_terrain,
            );
        }
    }

    #[test]
    fn test_respects_entity_budget() {
        let mut config = WorldConfig {
            width: 64,
            height: 64,
            max_entities: 10,
            resource_density: 1.0,
            ..WorldConfig::default()
        };
        config.seed = 42;
        let gen = TerrainGenerator::new(&config, config.seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let objects = ObjectPlacer::place_objects(&grid, &config, &mut rng);

        let max_allowed = (config.max_entities / 2) as usize;
        assert!(
            objects.len() <= max_allowed,
            "Objects ({}) should not exceed budget ({max_allowed})",
            objects.len()
        );
    }

    #[test]
    fn test_unique_ids() {
        let (grid, config) = make_grid_and_config(88);
        let mut rng = Pcg64Mcg::seed_from_u64(88);
        let objects = ObjectPlacer::place_objects(&grid, &config, &mut rng);

        let ids: std::collections::HashSet<u32> = objects.iter().map(|o| o.id).collect();
        assert_eq!(ids.len(), objects.len(), "All object IDs should be unique");
    }
}
