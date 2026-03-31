//! Agricultural simulation systems for drone-based precision farming.
//!
//! These systems process crop growth, disease spread, spraying, scanning,
//! soil relay, and report generation. All systems are opt-in via `AgriConfig`
//! and use fixed-point arithmetic for deterministic simulation.

use forge_types::agriculture::{CropScanResult, CropState, SoilReading, SoilSensorNode};
use forge_types::config::AgriConfig;
use forge_types::constants::FIXED_POINT_ONE;
use forge_types::entity::{Agent, AgentMorphology};
use forge_types::grid::{Grid, TerrainType};
use forge_types::resource::ItemType;
use forge_types::Action;
use tracing::{instrument, trace};

/// Advances crop growth, drains moisture/nutrients, and spreads disease.
///
/// Only processes tiles with `Cropland` or `Orchard` terrain. Growth stage
/// advances when accumulated growth exceeds a stage threshold. Disease
/// spreads from infected tiles to adjacent cropland with configurable
/// probability.
#[instrument(skip_all)]
pub fn process_crop_growth(
    crop_states: &mut [CropState],
    grid: &Grid,
    config: &AgriConfig,
    tick: u64,
    disease_spread_candidates: &mut Vec<(u16, u16)>,
) {
    let width = grid.width as usize;
    let height = grid.height as usize;

    disease_spread_candidates.clear();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let tile = match grid.get(x as u16, y as u16) {
                Some(t) => t,
                None => continue,
            };

            // Only process crop tiles
            if tile.terrain != TerrainType::Cropland && tile.terrain != TerrainType::Orchard {
                continue;
            }

            let crop = &mut crop_states[idx];

            // Drain moisture and nutrients
            crop.moisture = (crop.moisture - config.moisture_drain_rate).max(0);
            crop.nutrients = (crop.nutrients - config.nutrient_drain_rate).max(0);

            // Health degrades when moisture or nutrients are low (below 30%)
            let low_threshold = FIXED_POINT_ONE * 3 / 10;
            if crop.moisture < low_threshold || crop.nutrients < low_threshold {
                crop.health = (crop.health - config.crop_growth_rate / 2).max(0);
            }

            // Disease degrades health
            if crop.disease_level > 0 {
                let damage = (crop.disease_level as i64 * config.crop_growth_rate as i64
                    / FIXED_POINT_ONE as i64) as i32;
                crop.health = (crop.health - damage).max(0);

                // Natural disease decay (if sprayed, decay faster)
                let decay = if crop.sprayed {
                    config.disease_decay_rate * 2
                } else {
                    config.disease_decay_rate
                };
                crop.disease_level = (crop.disease_level - decay).max(0);

                // Collect candidates for disease spread
                disease_spread_candidates.push((x as u16, y as u16));
            }

            // Spray flag decays each tick
            if crop.sprayed && tick % 10 == 0 {
                crop.sprayed = false;
            }

            // Growth advancement (only if healthy enough)
            let health_threshold = FIXED_POINT_ONE * 3 / 10;
            if crop.health > health_threshold
                && crop.growth_stage < config.max_growth_stages
                && crop.moisture > low_threshold
                && crop.nutrients > low_threshold
            {
                // Use tick modulo for growth rate
                let growth_interval =
                    (FIXED_POINT_ONE as u64 / config.crop_growth_rate.max(1) as u64).max(1);
                if tick % growth_interval == 0 {
                    crop.growth_stage += 1;
                }
            }
        }
    }

    // Disease spread: each diseased tile can infect cardinal neighbors
    // We use a simple seeded iteration to maintain determinism
    for &(x, y) in disease_spread_candidates.iter() {
        let neighbors = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for &(nx, ny) in &neighbors {
            if nx >= grid.width || ny >= grid.height {
                continue;
            }
            let n_idx = ny as usize * width + nx as usize;
            if n_idx >= crop_states.len() {
                continue;
            }
            let neighbor_tile = match grid.get(nx, ny) {
                Some(t) => t,
                None => continue,
            };
            if neighbor_tile.terrain != TerrainType::Cropland
                && neighbor_tile.terrain != TerrainType::Orchard
            {
                continue;
            }
            // Spread probability: disease_spread_rate / FIXED_POINT_ONE
            // Use deterministic check based on tick + position
            let spread_hash = tick
                .wrapping_mul(31)
                .wrapping_add(nx as u64 * 17 + ny as u64 * 13);
            let threshold = (spread_hash % FIXED_POINT_ONE as u64) as i32;
            if threshold < config.disease_spread_rate && !crop_states[n_idx].sprayed {
                crop_states[n_idx].disease_level = (crop_states[n_idx].disease_level
                    + config.disease_spread_rate)
                    .min(FIXED_POINT_ONE);
            }
        }
    }
}

/// Processes spray actions for agents carrying pesticide.
///
/// Aerial agents that are airborne and carry `Pesticide` in the specified
/// inventory slot can spray, reducing disease in a radius around their position.
#[instrument(skip_all)]
pub fn process_spraying(
    agents: &mut [Agent],
    crop_states: &mut [CropState],
    grid: &Grid,
    actions: &[Action],
    config: &AgriConfig,
) {
    let width = grid.width as i32;
    let height = grid.height as i32;
    let grid_width = grid.width as usize;

    for (i, action) in actions.iter().enumerate() {
        let slot = match action {
            Action::Spray(s) => *s,
            _ => continue,
        };

        let agent = &agents[i];
        if !agent.alive || agent.morphology != AgentMorphology::Aerial || agent.altitude == 0 {
            continue;
        }

        // Check if agent has Pesticide in the given slot
        let has_pesticide = agent
            .inventory
            .get_slot(slot as usize)
            .is_some_and(|stack| stack.item_type == ItemType::Pesticide);
        if !has_pesticide {
            continue;
        }

        let cx = agent.position.x as i32;
        let cy = agent.position.y as i32;
        let r = config.spray_radius as i32;

        // Consume 1 pesticide
        agents[i].inventory.remove_item(ItemType::Pesticide, 1);

        // Deduct battery cost
        agents[i].battery = (agents[i].battery - config.spray_battery_cost).max(0);

        for dy in -r..=r {
            for dx in -r..=r {
                let tx = cx + dx;
                let ty = cy + dy;
                if tx < 0 || tx >= width || ty < 0 || ty >= height {
                    continue;
                }

                let idx = ty as usize * grid_width + tx as usize;
                if idx >= crop_states.len() {
                    continue;
                }

                let tile = match grid.get(tx as u16, ty as u16) {
                    Some(t) => t,
                    None => continue,
                };
                if tile.terrain != TerrainType::Cropland && tile.terrain != TerrainType::Orchard {
                    continue;
                }

                crop_states[idx].disease_level =
                    (crop_states[idx].disease_level - config.spray_efficacy).max(0);
                crop_states[idx].sprayed = true;
            }
        }
    }
}

/// Processes multispectral NDVI scans for aerial agents.
///
/// Computes NDVI for crop tiles within scan radius, records results,
/// and marks tiles as surveyed.
#[instrument(skip_all)]
pub fn process_multispectral_scan(
    agents: &mut [Agent],
    crop_states: &mut [CropState],
    grid: &Grid,
    actions: &[Action],
    config: &AgriConfig,
    tick: u64,
    scan_results: &mut Vec<CropScanResult>,
) {
    scan_results.clear();

    for (i, action) in actions.iter().enumerate() {
        if *action != Action::ScanMultispectral {
            continue;
        }

        if !agents[i].alive
            || agents[i].morphology != AgentMorphology::Aerial
            || agents[i].altitude == 0
        {
            continue;
        }

        let cx = agents[i].position.x as i32;
        let cy = agents[i].position.y as i32;
        let agent_id = agents[i].id;

        // Deduct battery
        agents[i].battery = (agents[i].battery - config.scan_battery_cost).max(0);

        let r = config.ndvi_scan_radius as i32;
        let width = grid.width as i32;
        let height = grid.height as i32;
        let grid_width = grid.width as usize;

        for dy in -r..=r {
            for dx in -r..=r {
                let tx = cx + dx;
                let ty = cy + dy;
                if tx < 0 || tx >= width || ty < 0 || ty >= height {
                    continue;
                }

                let tile = match grid.get(tx as u16, ty as u16) {
                    Some(t) => t,
                    None => continue,
                };
                if tile.terrain != TerrainType::Cropland && tile.terrain != TerrainType::Orchard {
                    continue;
                }

                let idx = ty as usize * grid_width + tx as usize;
                if idx >= crop_states.len() {
                    continue;
                }

                let crop = &mut crop_states[idx];
                let ndvi = crop.compute_ndvi();
                let ndvi_f32 = ndvi as f32 / FIXED_POINT_ONE as f32;

                scan_results.push(CropScanResult {
                    position: (tx as u16, ty as u16),
                    ndvi: ndvi_f32,
                    thermal: 0.0, // multispectral scan doesn't include thermal
                    disease_flag: crop.disease_level > FIXED_POINT_ONE / 10,
                });

                // Mark as surveyed
                crop.surveyed_tick = tick;
            }
        }

        // Track disease detections per agent
        let disease_count = scan_results.iter().filter(|r| r.disease_flag).count() as u16;
        trace!(
            agent_id,
            scanned_tiles = scan_results.len(),
            disease_count,
            "multispectral scan complete"
        );
    }
}

/// Processes thermal scans for irrigation stress mapping.
///
/// Computes canopy temperature proxy (based on moisture) for crop tiles
/// within thermal scan radius.
#[instrument(skip_all)]
pub fn process_thermal_scan(
    agents: &mut [Agent],
    crop_states: &[CropState],
    grid: &Grid,
    actions: &[Action],
    config: &AgriConfig,
    scan_results: &mut Vec<CropScanResult>,
) {
    // Thermal results are appended to scan_results (shared buffer)
    for (i, action) in actions.iter().enumerate() {
        if *action != Action::ScanThermal {
            continue;
        }

        if !agents[i].alive
            || agents[i].morphology != AgentMorphology::Aerial
            || agents[i].altitude == 0
        {
            continue;
        }

        let cx = agents[i].position.x as i32;
        let cy = agents[i].position.y as i32;

        agents[i].battery = (agents[i].battery - config.scan_battery_cost).max(0);
        let r = config.thermal_scan_radius as i32;
        let width = grid.width as i32;
        let height = grid.height as i32;
        let grid_width = grid.width as usize;

        for dy in -r..=r {
            for dx in -r..=r {
                let tx = cx + dx;
                let ty = cy + dy;
                if tx < 0 || tx >= width || ty < 0 || ty >= height {
                    continue;
                }

                let tile = match grid.get(tx as u16, ty as u16) {
                    Some(t) => t,
                    None => continue,
                };
                if tile.terrain != TerrainType::Cropland && tile.terrain != TerrainType::Orchard {
                    continue;
                }

                let idx = ty as usize * grid_width + tx as usize;
                if idx >= crop_states.len() {
                    continue;
                }

                let crop = &crop_states[idx];
                // CWSI proxy: low moisture = high canopy temperature = stress
                let thermal = 1.0 - (crop.moisture as f32 / FIXED_POINT_ONE as f32).clamp(0.0, 1.0);

                scan_results.push(CropScanResult {
                    position: (tx as u16, ty as u16),
                    ndvi: 0.0, // thermal scan doesn't compute NDVI
                    thermal,
                    disease_flag: false,
                });
            }
        }
    }
}

/// Processes soil sensor data relay from nearby ground nodes.
///
/// Agents within `soil_relay_range` of a soil sensor node can collect
/// its data, marking the node as collected.
#[instrument(skip_all)]
pub fn process_soil_relay(
    agents: &[Agent],
    soil_nodes: &mut [SoilSensorNode],
    actions: &[Action],
    config: &AgriConfig,
    tick: u64,
    soil_readings: &mut Vec<SoilReading>,
) {
    soil_readings.clear();

    for (i, action) in actions.iter().enumerate() {
        if *action != Action::RelaySoilData {
            continue;
        }

        let agent = &agents[i];
        if !agent.alive {
            continue;
        }

        let range = config.soil_relay_range as i32;

        for node in soil_nodes.iter_mut() {
            let dx = (agent.position.x as i32 - node.position.x as i32).abs();
            let dy = (agent.position.y as i32 - node.position.y as i32).abs();

            if dx <= range && dy <= range {
                node.collected = true;
                node.last_read_tick = tick;

                soil_readings.push(SoilReading {
                    node_id: node.id,
                    npk: [
                        node.npk[0] as f32 / FIXED_POINT_ONE as f32,
                        node.npk[1] as f32 / FIXED_POINT_ONE as f32,
                        node.npk[2] as f32 / FIXED_POINT_ONE as f32,
                    ],
                    ph: node.ph as f32 / FIXED_POINT_ONE as f32,
                    moisture: node.moisture as f32 / FIXED_POINT_ONE as f32,
                });
            }
        }
    }
}

/// Processes report generation actions.
///
/// Agents with available scan data can generate a field report,
/// setting `report_ready` on their observation.
#[instrument(skip_all)]
pub fn process_report_generation(
    agents: &mut [Agent],
    actions: &[Action],
    config: &AgriConfig,
    report_flags: &mut Vec<bool>,
) {
    report_flags.clear();
    report_flags.resize(agents.len(), false);

    for (i, action) in actions.iter().enumerate() {
        if *action != Action::GenerateReport {
            continue;
        }

        if !agents[i].alive {
            continue;
        }

        let agent_id = agents[i].id;

        // Deduct battery
        agents[i].battery = (agents[i].battery - config.report_generation_cost).max(0);
        report_flags[i] = true;

        trace!(agent_id, "field report generated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::grid::Position;

    fn make_agri_config() -> AgriConfig {
        AgriConfig {
            enabled: true,
            ..AgriConfig::default()
        }
    }

    fn make_aerial_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        let mut agent = Agent::new(id, Position::new(x, y), &config);
        agent.morphology = AgentMorphology::Aerial;
        agent.capabilities.can_fly = true;
        agent.altitude = 3;
        agent.battery = FIXED_POINT_ONE;
        agent
    }

    fn make_cropland_grid(width: u16, height: u16) -> Grid {
        let mut grid = Grid::new(width, height);
        for tile in grid.tiles.iter_mut() {
            tile.terrain = TerrainType::Cropland;
        }
        grid
    }

    fn make_crop_states(count: usize) -> Vec<CropState> {
        vec![CropState::default(); count]
    }

    #[test]
    fn test_crop_growth_advances() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);
        let mut candidates = Vec::new();

        // Run enough ticks for growth
        for tick in 0..100 {
            process_crop_growth(&mut states, &grid, &config, tick, &mut candidates);
        }

        // At least some crops should have advanced
        let grown = states.iter().filter(|s| s.growth_stage > 0).count();
        assert!(grown > 0, "crops should advance growth after 100 ticks");
    }

    #[test]
    fn test_crop_moisture_drains() {
        let grid = make_cropland_grid(2, 2);
        let config = make_agri_config();
        let mut states = make_crop_states(4);
        let initial_moisture = states[0].moisture;
        let mut candidates = Vec::new();

        process_crop_growth(&mut states, &grid, &config, 0, &mut candidates);

        assert!(
            states[0].moisture < initial_moisture,
            "moisture should drain each tick"
        );
    }

    #[test]
    fn test_crop_moisture_clamps_at_zero() {
        let grid = make_cropland_grid(1, 1);
        let mut config = make_agri_config();
        config.moisture_drain_rate = FIXED_POINT_ONE * 2; // Very high drain
        let mut states = make_crop_states(1);
        let mut candidates = Vec::new();

        process_crop_growth(&mut states, &grid, &config, 0, &mut candidates);

        assert_eq!(states[0].moisture, 0, "moisture must not go below 0");
    }

    #[test]
    fn test_disease_spread() {
        let grid = make_cropland_grid(3, 3);
        let mut config = make_agri_config();
        config.disease_spread_rate = FIXED_POINT_ONE; // 100% spread rate
        let mut states = make_crop_states(9);
        states[4].disease_level = FIXED_POINT_ONE / 2; // Center tile diseased
        let mut candidates = Vec::new();

        // Run several ticks
        for tick in 0..10 {
            process_crop_growth(&mut states, &grid, &config, tick, &mut candidates);
        }

        // At least some neighbors should be infected
        let infected = states.iter().filter(|s| s.disease_level > 0).count();
        assert!(
            infected > 1,
            "disease should spread to neighbors, but only {} infected",
            infected
        );
    }

    #[test]
    fn test_spraying_reduces_disease() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);
        // Set some disease
        for state in states.iter_mut() {
            state.disease_level = FIXED_POINT_ONE / 2;
        }

        let mut agent = make_aerial_agent(0, 2, 2);
        // Give agent a pesticide
        agent.inventory.add_item(ItemType::Pesticide, 5);

        let mut agents = vec![agent];
        let actions = vec![Action::Spray(0)];

        process_spraying(&mut agents, &mut states, &grid, &actions, &config);

        // Tiles near agent should have reduced disease
        let center_idx = 2 + 2 * 4;
        assert!(
            states[center_idx].disease_level < FIXED_POINT_ONE / 2,
            "spraying should reduce disease"
        );
        assert!(states[center_idx].sprayed, "sprayed flag should be set");
    }

    #[test]
    fn test_spraying_consumes_pesticide() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);

        let mut agent = make_aerial_agent(0, 2, 2);
        agent.inventory.add_item(ItemType::Pesticide, 3);

        let mut agents = vec![agent];
        let actions = vec![Action::Spray(0)];

        process_spraying(&mut agents, &mut states, &grid, &actions, &config);

        let remaining = agents[0].inventory.get_slot(0).map_or(0, |s| s.count);
        assert_eq!(remaining, 2, "should consume 1 pesticide");
    }

    #[test]
    fn test_spraying_ground_agent_ignored() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);
        for state in states.iter_mut() {
            state.disease_level = FIXED_POINT_ONE / 2;
        }

        let agent_config = AgentConfig::default();
        let mut agent = Agent::new(0, Position::new(2, 2), &agent_config);
        agent.inventory.add_item(ItemType::Pesticide, 5);

        let mut agents = vec![agent];
        let actions = vec![Action::Spray(0)];

        process_spraying(&mut agents, &mut states, &grid, &actions, &config);

        // Disease should be unchanged (ground agent can't spray)
        assert_eq!(
            states[2 + 2 * 4].disease_level,
            FIXED_POINT_ONE / 2,
            "ground agent spray should be ignored"
        );
    }

    #[test]
    fn test_multispectral_scan_records_results() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);
        states[5].disease_level = FIXED_POINT_ONE / 2; // One diseased tile

        let mut agents = vec![make_aerial_agent(0, 2, 2)];
        let actions = vec![Action::ScanMultispectral];
        let mut scan_results = Vec::new();

        process_multispectral_scan(
            &mut agents,
            &mut states,
            &grid,
            &actions,
            &config,
            100,
            &mut scan_results,
        );

        assert!(!scan_results.is_empty(), "should produce scan results");
    }

    #[test]
    fn test_multispectral_scan_marks_surveyed() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);

        let mut agents = vec![make_aerial_agent(0, 2, 2)];
        let actions = vec![Action::ScanMultispectral];
        let mut scan_results = Vec::new();

        process_multispectral_scan(
            &mut agents,
            &mut states,
            &grid,
            &actions,
            &config,
            42,
            &mut scan_results,
        );

        // Tiles within scan radius should be marked as surveyed
        let surveyed = states.iter().filter(|s| s.surveyed_tick == 42).count();
        assert!(surveyed > 0, "scanned tiles should be marked surveyed");
    }

    #[test]
    fn test_thermal_scan_computes_cwsi() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let states = make_crop_states(16);

        let mut agents = vec![make_aerial_agent(0, 2, 2)];
        let actions = vec![Action::ScanThermal];
        let mut scan_results = Vec::new();

        process_thermal_scan(
            &mut agents,
            &states,
            &grid,
            &actions,
            &config,
            &mut scan_results,
        );

        assert!(!scan_results.is_empty(), "should produce thermal results");
        // Default moisture is FIXED_POINT_ONE (1.0), so thermal should be ~0.0
        for result in &scan_results {
            assert!(
                result.thermal < 0.1,
                "full moisture should give low thermal stress, got {}",
                result.thermal
            );
        }
    }

    #[test]
    fn test_thermal_scan_high_stress() {
        let grid = make_cropland_grid(2, 2);
        let config = make_agri_config();
        let mut states = make_crop_states(4);
        // Drain all moisture
        for s in states.iter_mut() {
            s.moisture = 0;
        }

        let mut agents = vec![make_aerial_agent(0, 0, 0)];
        let actions = vec![Action::ScanThermal];
        let mut scan_results = Vec::new();

        process_thermal_scan(
            &mut agents,
            &states,
            &grid,
            &actions,
            &config,
            &mut scan_results,
        );

        for result in &scan_results {
            assert!(
                result.thermal > 0.9,
                "zero moisture should give high thermal stress, got {}",
                result.thermal
            );
        }
    }

    #[test]
    fn test_soil_relay_collects_from_nearby_nodes() {
        let config = make_agri_config();
        let agent = make_aerial_agent(0, 5, 5);
        let agents = vec![agent];
        let actions = vec![Action::RelaySoilData];

        let mut nodes = vec![
            SoilSensorNode::new(0, Position::new(5, 5)),
            SoilSensorNode::new(1, Position::new(6, 5)),
            SoilSensorNode::new(2, Position::new(100, 100)), // Far away
        ];
        let mut readings = Vec::new();

        process_soil_relay(&agents, &mut nodes, &actions, &config, 10, &mut readings);

        assert_eq!(readings.len(), 2, "should collect from 2 nearby nodes");
        assert!(nodes[0].collected, "node 0 should be marked collected");
        assert!(nodes[1].collected, "node 1 should be marked collected");
        assert!(
            !nodes[2].collected,
            "node 2 should not be collected (too far)"
        );
    }

    #[test]
    fn test_report_generation() {
        let config = make_agri_config();
        let mut agents = vec![make_aerial_agent(0, 5, 5)];
        let initial_battery = agents[0].battery;
        let actions = vec![Action::GenerateReport];
        let mut report_flags = Vec::new();

        process_report_generation(&mut agents, &actions, &config, &mut report_flags);

        assert!(report_flags[0], "report should be flagged as ready");
        assert!(
            agents[0].battery < initial_battery,
            "report should cost battery"
        );
    }

    #[test]
    fn test_spraying_deducts_battery() {
        let grid = make_cropland_grid(4, 4);
        let config = make_agri_config();
        let mut states = make_crop_states(16);

        let mut agent = make_aerial_agent(0, 2, 2);
        agent.inventory.add_item(ItemType::Pesticide, 5);
        let initial_battery = agent.battery;

        let mut agents = vec![agent];
        let actions = vec![Action::Spray(0)];

        process_spraying(&mut agents, &mut states, &grid, &actions, &config);

        assert!(
            agents[0].battery < initial_battery,
            "spraying should deduct battery"
        );
    }
}
