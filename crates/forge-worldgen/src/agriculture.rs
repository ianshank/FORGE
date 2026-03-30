//! Agricultural world generation: terrain conversion, crop state initialization,
//! and soil sensor node placement.
//!
//! All generation is deterministic given the same seed and configuration.

use forge_types::agriculture::{CropState, SoilSensorNode};
use forge_types::config::AgriConfig;
use forge_types::grid::{Grid, Position, TerrainType};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use tracing::instrument;

/// Converts a fraction of walkable `Ground` tiles to `Cropland` and `Pasture`
/// based on `AgriConfig` density parameters.
///
/// Tiles with lower elevation are more likely to become cropland (simulating
/// fertile lowlands). The conversion is deterministic given the same RNG state.
#[instrument(skip_all)]
pub fn convert_terrain(grid: &mut Grid, config: &AgriConfig, rng: &mut Pcg64Mcg) {
    let width = grid.width;
    let height = grid.height;

    for y in 0..height {
        for x in 0..width {
            let tile = match grid.get(x, y) {
                Some(t) => t,
                None => continue,
            };

            // Only convert Ground tiles that aren't occupied
            if tile.terrain != TerrainType::Ground
                || tile.agent_id.is_some()
                || tile.object_id.is_some()
                || tile.resource_id.is_some()
            {
                continue;
            }

            let roll: f32 = rng.gen();

            // Lower elevation tiles are more suitable for crops
            let elevation_factor = 1.0 - (tile.elevation as f32 / 255.0) * 0.5;

            if roll < config.cropland_density * elevation_factor {
                if let Some(tile_mut) = grid.get_mut(x, y) {
                    tile_mut.terrain = TerrainType::Cropland;
                }
            } else if roll < (config.cropland_density + config.pasture_density) * elevation_factor {
                if let Some(tile_mut) = grid.get_mut(x, y) {
                    tile_mut.terrain = TerrainType::Pasture;
                }
            }
        }
    }
}

/// Generates initial `CropState` for every tile in the grid.
///
/// Only `Cropland` and `Orchard` tiles get meaningful crop state;
/// other tiles receive a default (zero-growth) state that is never
/// ticked by the simulation.
#[instrument(skip_all)]
pub fn generate_crop_states(grid: &Grid, config: &AgriConfig) -> Vec<CropState> {
    let total = grid.tiles.len();
    let mut states = Vec::with_capacity(total);

    for tile in &grid.tiles {
        match tile.terrain {
            TerrainType::Cropland | TerrainType::Orchard => {
                states.push(CropState::with_health(config.initial_crop_health));
            }
            _ => {
                states.push(CropState::default());
            }
        }
    }

    states
}

/// Spawns ground-deployed IoT soil sensor nodes on `Cropland` tiles.
///
/// Nodes are distributed as evenly as possible across the cropland area.
/// Returns up to `config.num_soil_nodes` nodes.
#[instrument(skip_all)]
pub fn spawn_soil_nodes(
    grid: &Grid,
    config: &AgriConfig,
    rng: &mut Pcg64Mcg,
) -> Vec<SoilSensorNode> {
    if config.num_soil_nodes == 0 {
        return Vec::new();
    }

    // Collect all cropland positions
    let mut cropland_positions: Vec<Position> = Vec::new();
    let width = grid.width;
    for (i, tile) in grid.tiles.iter().enumerate() {
        if tile.terrain == TerrainType::Cropland {
            let x = (i % width as usize) as u16;
            let y = (i / width as usize) as u16;
            cropland_positions.push(Position::new(x, y));
        }
    }

    if cropland_positions.is_empty() {
        return Vec::new();
    }

    // Shuffle and pick up to num_soil_nodes positions
    let count = (config.num_soil_nodes as usize).min(cropland_positions.len());

    // Fisher-Yates partial shuffle
    for i in 0..count {
        let j = rng.gen_range(i..cropland_positions.len());
        cropland_positions.swap(i, j);
    }

    cropland_positions
        .into_iter()
        .take(count)
        .enumerate()
        .map(|(id, pos)| SoilSensorNode::new(id as u32, pos))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_test_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    #[test]
    fn test_convert_terrain_creates_cropland() {
        let mut grid = make_test_grid(32, 32);
        let config = AgriConfig {
            enabled: true,
            cropland_density: 0.3,
            pasture_density: 0.1,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        convert_terrain(&mut grid, &config, &mut rng);

        let cropland_count = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Cropland)
            .count();
        let pasture_count = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Pasture)
            .count();

        assert!(cropland_count > 0, "should have created some cropland");
        assert!(pasture_count > 0, "should have created some pasture");
    }

    #[test]
    fn test_convert_terrain_deterministic() {
        let config = AgriConfig {
            enabled: true,
            cropland_density: 0.3,
            pasture_density: 0.1,
            ..AgriConfig::default()
        };

        let mut grid1 = make_test_grid(16, 16);
        let mut rng1 = Pcg64Mcg::seed_from_u64(123);
        convert_terrain(&mut grid1, &config, &mut rng1);

        let mut grid2 = make_test_grid(16, 16);
        let mut rng2 = Pcg64Mcg::seed_from_u64(123);
        convert_terrain(&mut grid2, &config, &mut rng2);

        for (a, b) in grid1.tiles.iter().zip(grid2.tiles.iter()) {
            assert_eq!(a.terrain, b.terrain, "terrain must be deterministic");
        }
    }

    #[test]
    fn test_convert_terrain_zero_density_no_change() {
        let mut grid = make_test_grid(16, 16);
        let config = AgriConfig {
            enabled: true,
            cropland_density: 0.0,
            pasture_density: 0.0,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        convert_terrain(&mut grid, &config, &mut rng);

        let cropland = grid
            .tiles
            .iter()
            .filter(|t| t.terrain == TerrainType::Cropland)
            .count();
        assert_eq!(cropland, 0, "zero density should not create cropland");
    }

    #[test]
    fn test_generate_crop_states_size() {
        let grid = make_test_grid(8, 8);
        let config = AgriConfig::default();
        let states = generate_crop_states(&grid, &config);
        assert_eq!(states.len(), grid.tiles.len());
    }

    #[test]
    fn test_generate_crop_states_for_cropland() {
        let mut grid = make_test_grid(4, 4);
        // Set a tile to Cropland
        grid.get_mut(1, 1).unwrap().terrain = TerrainType::Cropland;

        let config = AgriConfig::default();
        let states = generate_crop_states(&grid, &config);

        let idx = 1 + 1 * 4; // x=1, y=1 in a 4-wide grid
        assert_eq!(
            states[idx].health, config.initial_crop_health,
            "cropland tiles should use initial_crop_health"
        );
    }

    #[test]
    fn test_spawn_soil_nodes_count() {
        let mut grid = make_test_grid(16, 16);
        // Create some cropland
        for i in 0..50 {
            grid.tiles[i].terrain = TerrainType::Cropland;
        }

        let config = AgriConfig {
            enabled: true,
            num_soil_nodes: 10,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        let nodes = spawn_soil_nodes(&grid, &config, &mut rng);
        assert_eq!(nodes.len(), 10);

        // All nodes should be on cropland tiles
        for node in &nodes {
            let tile = grid.get_pos(&node.position).unwrap();
            assert_eq!(tile.terrain, TerrainType::Cropland);
        }
    }

    #[test]
    fn test_spawn_soil_nodes_capped_by_available() {
        let mut grid = make_test_grid(4, 4);
        // Only 3 cropland tiles
        grid.tiles[0].terrain = TerrainType::Cropland;
        grid.tiles[1].terrain = TerrainType::Cropland;
        grid.tiles[2].terrain = TerrainType::Cropland;

        let config = AgriConfig {
            enabled: true,
            num_soil_nodes: 100,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        let nodes = spawn_soil_nodes(&grid, &config, &mut rng);
        assert_eq!(nodes.len(), 3, "should be capped by available cropland");
    }

    #[test]
    fn test_spawn_soil_nodes_zero() {
        let grid = make_test_grid(4, 4);
        let config = AgriConfig {
            enabled: true,
            num_soil_nodes: 0,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        let nodes = spawn_soil_nodes(&grid, &config, &mut rng);
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_spawn_soil_nodes_unique_ids() {
        let mut grid = make_test_grid(8, 8);
        for i in 0..20 {
            grid.tiles[i].terrain = TerrainType::Cropland;
        }

        let config = AgriConfig {
            enabled: true,
            num_soil_nodes: 10,
            ..AgriConfig::default()
        };
        let mut rng = Pcg64Mcg::seed_from_u64(42);

        let nodes = spawn_soil_nodes(&grid, &config, &mut rng);
        let ids: Vec<u32> = nodes.iter().map(|n| n.id).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "sensor node IDs must be unique");
            }
        }
    }
}
