//! Sensor and observation model for agent perception.
//!
//! Provides configurable sensor types (Visual, Acoustic, Radar) that produce
//! [`ObservationMask`] results describing which tiles and entities an agent
//! can currently perceive. Sensors respect line-of-sight, range limits, and
//! can be jammed to suppress all detections.

use std::collections::HashSet;

use forge_types::entity::Agent;
use forge_types::grid::Grid;
use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// The kind of sensor an agent is using.
///
/// Different sensor types have different detection characteristics:
/// Visual is blocked by terrain, while Acoustic and Radar penetrate it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SensorType {
    /// Line-of-sight visual detection. Blocked by vision-blocking terrain.
    #[default]
    Visual,
    /// Sound-based detection. Penetrates vision-blocking terrain.
    Acoustic,
    /// Radar-based detection. Penetrates vision-blocking terrain.
    Radar,
}

/// Configuration for an agent's sensor system.
///
/// All detection parameters are specified here — no hard-coded values.
/// Use `Default` for reasonable starting values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SensorConfig {
    /// Maximum detection range in tiles (Euclidean distance).
    pub range: f64,
    /// Standard deviation of Gaussian noise applied to position readings.
    ///
    /// This does not affect *whether* an entity is detected, only the
    /// positional accuracy reported downstream.
    pub noise_sigma: f64,
    /// Whether the sensor is currently jammed (suppresses all output).
    pub jammed: bool,
    /// The type of sensor being used.
    pub sensor_type: SensorType,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            range: 10.0,
            noise_sigma: 0.0,
            jammed: false,
            sensor_type: SensorType::default(),
        }
    }
}

/// The result of a sensor observation: which tiles and entities were detected.
///
/// Two masks can be merged (unioned) to combine observations from multiple
/// sensors or multiple agents.
#[derive(Debug, Clone)]
pub struct ObservationMask {
    /// Set of `(x, y)` tile coordinates currently visible to the sensor.
    pub visible_cells: HashSet<(u16, u16)>,
    /// Set of entity IDs detected by the sensor.
    pub detected_entities: HashSet<u32>,
}

impl ObservationMask {
    /// Creates an empty observation mask with no visible cells or detected entities.
    pub fn new() -> Self {
        Self {
            visible_cells: HashSet::new(),
            detected_entities: HashSet::new(),
        }
    }

    /// Returns `true` if the tile at `(x, y)` is visible in this mask.
    pub fn is_visible(&self, x: u16, y: u16) -> bool {
        self.visible_cells.contains(&(x, y))
    }

    /// Returns `true` if the entity with the given ID has been detected.
    pub fn is_detected(&self, entity_id: u32) -> bool {
        self.detected_entities.contains(&entity_id)
    }

    /// Merges another observation mask into this one (set union).
    ///
    /// After merging, this mask contains all visible cells and detected
    /// entities from both masks.
    pub fn merge(&mut self, other: &ObservationMask) {
        self.visible_cells
            .extend(other.visible_cells.iter().copied());
        self.detected_entities
            .extend(other.detected_entities.iter().copied());
    }
}

impl Default for ObservationMask {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes a sensor observation for an agent, returning which tiles and
/// entities it can detect.
///
/// The observation follows these rules:
/// - If the sensor is **jammed**, an empty mask is returned immediately.
/// - Tiles within the configured `range` are tested for line-of-sight using
///   Bresenham ray casting. Vision-blocking terrain stops the ray.
/// - Other agents whose positions fall within visible tiles are added to the
///   detected entities set (only alive agents are detected).
/// - `noise_sigma` does **not** affect detection; it indicates positional
///   noise for downstream consumers.
#[instrument(skip_all, fields(agent_id = agent.id, range = config.range, jammed = config.jammed))]
pub fn compute_observation(
    agent: &Agent,
    agents: &[Agent],
    grid: &Grid,
    config: &SensorConfig,
) -> ObservationMask {
    if config.jammed {
        trace!("sensor jammed, returning empty mask");
        return ObservationMask::new();
    }

    let mut mask = ObservationMask::new();

    let ax = agent.position.x as i32;
    let ay = agent.position.y as i32;
    let range_i = config.range.ceil() as i32;
    let range_sq = config.range * config.range;

    // Scan tiles within the bounding box of the sensor range.
    for dy in -range_i..=range_i {
        for dx in -range_i..=range_i {
            // Euclidean distance check.
            let dist_sq = (dx * dx + dy * dy) as f64;
            if dist_sq > range_sq {
                continue;
            }

            let tx = ax + dx;
            let ty = ay + dy;

            // Bounds check.
            if tx < 0 || ty < 0 || tx >= grid.width as i32 || ty >= grid.height as i32 {
                continue;
            }

            // Visual sensors use Bresenham line-of-sight; Acoustic and Radar
            // penetrate terrain and only require range.
            let visible = match config.sensor_type {
                SensorType::Visual => has_line_of_sight(grid, ax, ay, tx, ty),
                SensorType::Acoustic | SensorType::Radar => true,
            };

            if visible {
                mask.visible_cells.insert((tx as u16, ty as u16));
            }
        }
    }

    // Detect other alive agents whose positions fall in visible cells.
    for other in agents {
        if other.id == agent.id || !other.alive {
            continue;
        }
        let pos = (other.position.x, other.position.y);
        if mask.visible_cells.contains(&pos) {
            mask.detected_entities.insert(other.id);
        }
    }

    trace!(
        visible_cells = mask.visible_cells.len(),
        detected_entities = mask.detected_entities.len(),
        "observation computed"
    );

    mask
}

/// Bresenham line-of-sight check between two grid positions.
///
/// Intermediate tiles that block vision cause the function to return `false`.
/// The start and end tiles themselves are not considered blocking.
fn has_line_of_sight(grid: &Grid, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    if x0 == x1 && y0 == y1 {
        return true;
    }

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut cx = x0;
    let mut cy = y0;

    loop {
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            cx += sx;
        }
        if e2 < dx {
            err += dx;
            cy += sy;
        }

        if cx == x1 && cy == y1 {
            return true;
        }

        if cx >= 0 && cy >= 0 && cx < grid.width as i32 && cy < grid.height as i32 {
            if let Some(tile) = grid.get(cx as u16, cy as u16) {
                if tile.terrain.blocks_vision() {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::entity::Agent;
    use forge_types::grid::{Grid, Position, TerrainType};

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    // ---- Jammed sensor ----

    #[test]
    fn test_jammed_sensor_returns_empty() {
        let grid = make_grid(16, 16);
        let agent = make_agent(0, 5, 5);
        let config = SensorConfig {
            jammed: true,
            ..Default::default()
        };

        let mask = compute_observation(&agent, std::slice::from_ref(&agent), &grid, &config);

        assert!(
            mask.visible_cells.is_empty(),
            "jammed sensor should see no tiles"
        );
        assert!(
            mask.detected_entities.is_empty(),
            "jammed sensor should detect no entities"
        );
    }

    // ---- Default config detects nearby agents ----

    #[test]
    fn test_default_config_detects_nearby_agent() {
        let grid = make_grid(32, 32);
        let observer = make_agent(0, 10, 10);
        let target = make_agent(1, 12, 10);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig::default();

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(mask.is_visible(12, 10), "nearby tile should be visible");
        assert!(mask.is_detected(1), "nearby agent should be detected");
    }

    #[test]
    fn test_agent_sees_own_tile() {
        let grid = make_grid(16, 16);
        let agent = make_agent(0, 5, 5);
        let config = SensorConfig::default();

        let mask = compute_observation(&agent, std::slice::from_ref(&agent), &grid, &config);

        assert!(mask.is_visible(5, 5), "agent should see its own tile");
    }

    #[test]
    fn test_self_not_in_detected_entities() {
        let grid = make_grid(16, 16);
        let agent = make_agent(0, 5, 5);
        let config = SensorConfig::default();

        let mask = compute_observation(&agent, std::slice::from_ref(&agent), &grid, &config);

        assert!(!mask.is_detected(0), "agent should not detect itself");
    }

    // ---- Range limiting ----

    #[test]
    fn test_range_limits_visibility() {
        let grid = make_grid(32, 32);
        let agent = make_agent(0, 10, 10);
        let config = SensorConfig {
            range: 3.0,
            ..Default::default()
        };

        let mask = compute_observation(&agent, std::slice::from_ref(&agent), &grid, &config);

        // Tile within range should be visible.
        assert!(
            mask.is_visible(13, 10),
            "tile at distance 3 should be visible"
        );

        // Tile beyond range should not be visible.
        assert!(
            !mask.is_visible(15, 10),
            "tile at distance 5 should not be visible"
        );
    }

    #[test]
    fn test_range_limits_entity_detection() {
        let grid = make_grid(32, 32);
        let observer = make_agent(0, 10, 10);
        let near = make_agent(1, 12, 10); // distance 2
        let far = make_agent(2, 25, 10); // distance 15
        let agents = vec![observer.clone(), near, far];
        let config = SensorConfig {
            range: 5.0,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(mask.is_detected(1), "near agent should be detected");
        assert!(!mask.is_detected(2), "far agent should not be detected");
    }

    // ---- Merge ----

    #[test]
    fn test_merge_combines_visible_cells() {
        let mut mask_a = ObservationMask::new();
        mask_a.visible_cells.insert((1, 1));
        mask_a.visible_cells.insert((2, 2));

        let mut mask_b = ObservationMask::new();
        mask_b.visible_cells.insert((2, 2));
        mask_b.visible_cells.insert((3, 3));

        mask_a.merge(&mask_b);

        assert!(mask_a.is_visible(1, 1));
        assert!(mask_a.is_visible(2, 2));
        assert!(mask_a.is_visible(3, 3));
        assert_eq!(
            mask_a.visible_cells.len(),
            3,
            "union should have 3 unique cells"
        );
    }

    #[test]
    fn test_merge_combines_detected_entities() {
        let mut mask_a = ObservationMask::new();
        mask_a.detected_entities.insert(1);

        let mut mask_b = ObservationMask::new();
        mask_b.detected_entities.insert(2);
        mask_b.detected_entities.insert(3);

        mask_a.merge(&mask_b);

        assert!(mask_a.is_detected(1));
        assert!(mask_a.is_detected(2));
        assert!(mask_a.is_detected(3));
        assert_eq!(mask_a.detected_entities.len(), 3);
    }

    #[test]
    fn test_merge_with_empty_mask() {
        let mut mask_a = ObservationMask::new();
        mask_a.visible_cells.insert((5, 5));
        mask_a.detected_entities.insert(42);

        let mask_b = ObservationMask::new();

        mask_a.merge(&mask_b);

        assert_eq!(mask_a.visible_cells.len(), 1);
        assert_eq!(mask_a.detected_entities.len(), 1);
    }

    // ---- SensorType variants ----

    #[test]
    fn test_sensor_type_default_is_visual() {
        assert_eq!(SensorType::default(), SensorType::Visual);
    }

    #[test]
    fn test_sensor_type_variants_distinct() {
        let variants = [SensorType::Visual, SensorType::Acoustic, SensorType::Radar];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_acoustic_sensor_computes_observation() {
        let grid = make_grid(16, 16);
        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 7, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig {
            sensor_type: SensorType::Acoustic,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            mask.is_detected(1),
            "acoustic sensor should detect nearby agent"
        );
    }

    #[test]
    fn test_radar_sensor_computes_observation() {
        let grid = make_grid(16, 16);
        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 7, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig {
            sensor_type: SensorType::Radar,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            mask.is_detected(1),
            "radar sensor should detect nearby agent"
        );
    }

    #[test]
    fn test_acoustic_sensor_sees_through_walls() {
        let mut grid = make_grid(16, 16);
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 9, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig {
            sensor_type: SensorType::Acoustic,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            mask.is_visible(9, 5),
            "acoustic sensor should see through walls"
        );
        assert!(
            mask.is_detected(1),
            "acoustic sensor should detect agent behind wall"
        );
    }

    #[test]
    fn test_radar_sensor_sees_through_walls() {
        let mut grid = make_grid(16, 16);
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 9, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig {
            sensor_type: SensorType::Radar,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            mask.is_visible(9, 5),
            "radar sensor should see through walls"
        );
        assert!(
            mask.is_detected(1),
            "radar sensor should detect agent behind wall"
        );
    }

    // ---- Entity detection edge cases ----

    #[test]
    fn test_dead_agent_not_detected() {
        let grid = make_grid(16, 16);
        let observer = make_agent(0, 5, 5);
        let mut target = make_agent(1, 6, 5);
        target.alive = false;
        let agents = vec![observer.clone(), target];
        let config = SensorConfig::default();

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(!mask.is_detected(1), "dead agent should not be detected");
    }

    #[test]
    fn test_wall_blocks_sensor_detection() {
        let mut grid = make_grid(16, 16);
        // Place a wall between observer at (5,5) and target at (9,5).
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 9, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig::default();

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            !mask.is_visible(9, 5),
            "tile behind wall should not be visible"
        );
        assert!(
            !mask.is_detected(1),
            "agent behind wall should not be detected"
        );
    }

    #[test]
    fn test_multiple_entities_detected() {
        let grid = make_grid(32, 32);
        let observer = make_agent(0, 10, 10);
        let a1 = make_agent(1, 11, 10);
        let a2 = make_agent(2, 10, 11);
        let a3 = make_agent(3, 12, 12);
        let agents = vec![observer.clone(), a1, a2, a3];
        let config = SensorConfig::default();

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(mask.is_detected(1));
        assert!(mask.is_detected(2));
        assert!(mask.is_detected(3));
    }

    // ---- SensorConfig default ----

    #[test]
    fn test_sensor_config_default_values() {
        let config = SensorConfig::default();
        assert!((config.range - 10.0).abs() < f64::EPSILON);
        assert!((config.noise_sigma - 0.0).abs() < f64::EPSILON);
        assert!(!config.jammed);
        assert_eq!(config.sensor_type, SensorType::Visual);
    }

    // ---- ObservationMask default ----

    #[test]
    fn test_observation_mask_default_is_empty() {
        let mask = ObservationMask::default();
        assert!(mask.visible_cells.is_empty());
        assert!(mask.detected_entities.is_empty());
    }

    // ---- Noise sigma does not prevent detection ----

    #[test]
    fn test_noise_sigma_does_not_prevent_detection() {
        let grid = make_grid(16, 16);
        let observer = make_agent(0, 5, 5);
        let target = make_agent(1, 7, 5);
        let agents = vec![observer.clone(), target];
        let config = SensorConfig {
            noise_sigma: 5.0,
            ..Default::default()
        };

        let mask = compute_observation(&observer, &agents, &grid, &config);

        assert!(
            mask.is_detected(1),
            "noise_sigma should not prevent detection"
        );
    }
}
