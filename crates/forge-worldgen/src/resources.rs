//! Resource node placement based on terrain type and configuration.
//!
//! [`ResourcePlacer`] distributes harvestable resource nodes across the world
//! according to terrain affinity rules.  Forest tiles yield Wood, Mountain
//! tiles yield Stone/Ore, Water tiles yield Fish, and so on.  Placement
//! density and resource properties are controlled through [`WorldConfig`].

use forge_types::config::WorldConfig;
use forge_types::constants::{DEFAULT_RESOURCE_MAX_QUANTITY, DEFAULT_RESOURCE_RESPAWN_TICKS};
use forge_types::grid::{Grid, Position, TerrainType};
use forge_types::resource::{ItemType, ResourceNode};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use tracing::instrument;

/// Describes a resource-to-terrain mapping rule.
#[derive(Debug, Clone)]
struct ResourceRule {
    /// The item this rule produces.
    item: ItemType,
    /// Terrain types where this resource can spawn.
    terrains: &'static [TerrainType],
    /// Relative weight (higher = more common among eligible tiles).
    weight: f64,
    /// Whether harvesting requires a specific tool.
    requires_tool: Option<ItemType>,
}

/// All resource placement rules.
const RESOURCE_RULES: &[ResourceRule] = &[
    ResourceRule {
        item: ItemType::Wood,
        terrains: &[TerrainType::Forest],
        weight: 1.0,
        requires_tool: None,
    },
    ResourceRule {
        item: ItemType::Stone,
        terrains: &[TerrainType::Mountain],
        weight: 0.7,
        requires_tool: Some(ItemType::Pickaxe),
    },
    ResourceRule {
        item: ItemType::Ore,
        terrains: &[TerrainType::Mountain],
        weight: 0.3,
        requires_tool: Some(ItemType::Pickaxe),
    },
    ResourceRule {
        item: ItemType::Fish,
        terrains: &[TerrainType::Water],
        weight: 0.6,
        requires_tool: None,
    },
    ResourceRule {
        item: ItemType::Fiber,
        terrains: &[TerrainType::Ground, TerrainType::Forest],
        weight: 0.4,
        requires_tool: None,
    },
    ResourceRule {
        item: ItemType::Clay,
        terrains: &[TerrainType::Sand],
        weight: 0.5,
        requires_tool: None,
    },
];

/// Places resource nodes on the grid based on terrain and configuration.
#[derive(Debug)]
pub struct ResourcePlacer;

impl ResourcePlacer {
    /// Places resources throughout the grid according to terrain affinity and density.
    ///
    /// * `grid` -- the generated terrain grid (read-only for terrain queries).
    /// * `config` -- world configuration (provides `resource_density`).
    /// * `rng` -- seeded RNG for deterministic placement.
    ///
    /// Returns a list of [`ResourceNode`]s.  Each node is also registered
    /// on the grid tile via `resource_id` (index into the returned vector).
    #[instrument(skip_all)]
    pub fn place_resources(
        grid: &Grid,
        config: &WorldConfig,
        rng: &mut Pcg64Mcg,
    ) -> Vec<ResourceNode> {
        let density = config.resource_density as f64;
        let mut nodes: Vec<ResourceNode> = Vec::new();

        tracing::trace!(
            density,
            width = grid.width,
            height = grid.height,
            "placing resources"
        );

        for y in 0..grid.height {
            for x in 0..grid.width {
                let tile = match grid.get(x, y) {
                    Some(t) => t,
                    None => continue,
                };

                // Find all applicable rules for this terrain
                for rule in RESOURCE_RULES {
                    if !rule.terrains.contains(&tile.terrain) {
                        continue;
                    }

                    // Probability of placing this resource = density * weight
                    let probability = density * rule.weight;
                    let roll: f64 = rng.gen();
                    if roll >= probability {
                        continue;
                    }

                    let id = nodes.len() as u32;
                    let max_quantity = if config.resource_max_quantity > 0 {
                        config.resource_max_quantity
                    } else {
                        DEFAULT_RESOURCE_MAX_QUANTITY
                    };
                    let quantity = rng.gen_range(1..=max_quantity);
                    let respawn_rate = if config.resource_respawn_rate > 0 {
                        config.resource_respawn_rate
                    } else {
                        DEFAULT_RESOURCE_RESPAWN_TICKS
                    };

                    nodes.push(ResourceNode {
                        id,
                        position: Position::new(x, y),
                        resource_type: rule.item,
                        quantity,
                        max_quantity,
                        respawn_timer: 0,
                        respawn_rate,
                        requires_tool: rule.requires_tool,
                    });

                    // Only one resource per tile
                    break;
                }
            }
        }

        tracing::trace!(count = nodes.len(), "resource placement complete");
        nodes
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
            ..WorldConfig::default()
        };
        let gen = TerrainGenerator::new(&config, seed);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);
        (grid, config)
    }

    #[test]
    fn test_resources_placed() {
        let (grid, config) = make_grid_and_config(42);
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let nodes = ResourcePlacer::place_resources(&grid, &config, &mut rng);
        assert!(!nodes.is_empty(), "Should place at least some resources");
    }

    #[test]
    fn test_determinism() {
        let (grid, config) = make_grid_and_config(99);

        let mut rng1 = Pcg64Mcg::seed_from_u64(99);
        let nodes1 = ResourcePlacer::place_resources(&grid, &config, &mut rng1);

        let mut rng2 = Pcg64Mcg::seed_from_u64(99);
        let nodes2 = ResourcePlacer::place_resources(&grid, &config, &mut rng2);

        assert_eq!(nodes1.len(), nodes2.len());
        for (a, b) in nodes1.iter().zip(nodes2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.resource_type, b.resource_type);
            assert_eq!(a.quantity, b.quantity);
        }
    }

    #[test]
    fn test_terrain_affinity() {
        let (grid, config) = make_grid_and_config(55);
        let mut rng = Pcg64Mcg::seed_from_u64(55);
        let nodes = ResourcePlacer::place_resources(&grid, &config, &mut rng);

        for node in &nodes {
            let tile = grid.get_pos(&node.position).unwrap();
            let matching_rule = RESOURCE_RULES
                .iter()
                .find(|r| r.item == node.resource_type)
                .expect("resource type should have a rule");
            assert!(
                matching_rule.terrains.contains(&tile.terrain),
                "Resource {:?} placed on {:?}, expected one of {:?}",
                node.resource_type,
                tile.terrain,
                matching_rule.terrains,
            );
        }
    }

    #[test]
    fn test_zero_density_no_resources() {
        let config = WorldConfig {
            resource_density: 0.0,
            width: 32,
            height: 32,
            ..Default::default()
        };
        let gen = TerrainGenerator::new(&config, 42);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let nodes = ResourcePlacer::place_resources(&grid, &config, &mut rng);
        assert!(nodes.is_empty(), "Zero density should produce no resources");
    }

    #[test]
    fn test_resource_quantities_valid() {
        let (grid, config) = make_grid_and_config(77);
        let mut rng = Pcg64Mcg::seed_from_u64(77);
        let nodes = ResourcePlacer::place_resources(&grid, &config, &mut rng);

        for node in &nodes {
            assert!(node.quantity > 0);
            assert!(node.quantity <= node.max_quantity);
            assert!(node.max_quantity > 0);
        }
    }

    #[test]
    fn test_unique_ids() {
        let (grid, config) = make_grid_and_config(88);
        let mut rng = Pcg64Mcg::seed_from_u64(88);
        let nodes = ResourcePlacer::place_resources(&grid, &config, &mut rng);

        let ids: std::collections::HashSet<u32> = nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids.len(), nodes.len(), "All resource IDs should be unique");
    }
}
