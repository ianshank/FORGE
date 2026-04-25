//! Agricultural simulation types for drone-based precision farming.
//!
//! These types model crop fields, soil sensors, and the data collected
//! by agricultural drone missions. All numerical state uses fixed-point
//! i32 (16.16 format) for deterministic simulation.

use serde::{Deserialize, Serialize};

use crate::constants::FIXED_POINT_ONE;
use crate::grid::Position;

/// Per-tile agricultural state. Only populated for `Cropland` / `Orchard` tiles.
///
/// All fixed-point fields use i32 with 16 fractional bits, clamped to `[0, FIXED_POINT_ONE]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CropState {
    /// Growth stage: 0 = bare soil, `max_growth_stages` = harvestable.
    pub growth_stage: u8,
    /// Crop health. Fixed-point `[0, 1.0]`. Below 0.3 the crop is failing.
    pub health: i32,
    /// Disease severity. Fixed-point `[0, 1.0]`. 0 = healthy, 1.0 = fully diseased.
    pub disease_level: i32,
    /// Soil moisture. Fixed-point `[0, 1.0]`. Drains each tick, replenished by irrigation.
    pub moisture: i32,
    /// Soil nutrient level. Fixed-point `[0, 1.0]`. Drains each tick, replenished by fertilizer.
    pub nutrients: i32,
    /// Whether this tile was recently sprayed with pesticide.
    pub sprayed: bool,
    /// Tick when this tile was last surveyed via multispectral scan. 0 = never.
    pub surveyed_tick: u64,
}

impl Default for CropState {
    fn default() -> Self {
        Self {
            growth_stage: 0,
            health: FIXED_POINT_ONE,
            disease_level: 0,
            moisture: FIXED_POINT_ONE,
            nutrients: FIXED_POINT_ONE,
            sprayed: false,
            surveyed_tick: 0,
        }
    }
}

impl CropState {
    /// Creates a new crop state with configurable initial health.
    pub fn with_health(initial_health: i32) -> Self {
        Self {
            health: initial_health.clamp(0, FIXED_POINT_ONE),
            ..Self::default()
        }
    }

    /// Whether this crop is diseased (disease_level > 0).
    pub fn is_diseased(&self) -> bool {
        self.disease_level > 0
    }

    /// Whether this crop has been surveyed.
    pub fn is_surveyed(&self) -> bool {
        self.surveyed_tick > 0
    }

    /// Computes a simulated NDVI value in fixed-point.
    ///
    /// NDVI = (health - disease_level) / (health + disease_level + epsilon).
    /// Returns a value in `[-FIXED_POINT_ONE, FIXED_POINT_ONE]`.
    pub fn compute_ndvi(&self) -> i32 {
        let epsilon = 1; // prevent division by zero
        let numerator = self.health - self.disease_level;
        let denominator = self.health + self.disease_level + epsilon;
        if denominator == 0 {
            return 0;
        }
        // Fixed-point division: (numerator * FIXED_POINT_ONE) / denominator
        ((numerator as i64 * FIXED_POINT_ONE as i64) / denominator as i64) as i32
    }
}

/// A ground-deployed IoT soil sensor node.
///
/// These are stationary sensors placed across cropland that drones
/// collect data from as aerial data mules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoilSensorNode {
    /// Unique sensor node identifier.
    pub id: u32,
    /// Grid position of the sensor.
    pub position: Position,
    /// Nitrogen, Phosphorus, Potassium readings (fixed-point each).
    pub npk: [i32; 3],
    /// Soil pH reading (fixed-point, ~7.0 = neutral = 7 * FIXED_POINT_ONE).
    pub ph: i32,
    /// Soil moisture reading (fixed-point `[0, 1.0]`).
    pub moisture: i32,
    /// Last tick this node's data was collected by a drone. 0 = never.
    pub last_read_tick: u64,
    /// Whether data has been collected this episode.
    pub collected: bool,
}

impl SoilSensorNode {
    /// Creates a new soil sensor node at the given position with default readings.
    pub fn new(id: u32, position: Position) -> Self {
        Self {
            id,
            position,
            npk: [FIXED_POINT_ONE / 2; 3], // mid-range defaults
            ph: FIXED_POINT_ONE * 7,       // neutral pH
            moisture: FIXED_POINT_ONE / 2, // mid moisture
            last_read_tick: 0,
            collected: false,
        }
    }
}

/// Result of a multispectral or thermal scan of a crop tile.
///
/// Produced by `ScanMultispectral` and `ScanThermal` drone actions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CropScanResult {
    /// Tile position that was scanned.
    pub position: (u16, u16),
    /// Computed NDVI value (normalized, -1.0 to 1.0 as f32).
    pub ndvi: f32,
    /// Normalized canopy temperature (0.0 = cool/healthy, 1.0 = hot/stressed).
    pub thermal: f32,
    /// Whether disease was flagged on this tile.
    pub disease_flag: bool,
}

/// Data collected from a soil sensor node relay.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SoilReading {
    /// The sensor node that provided this reading.
    pub node_id: u32,
    /// Normalized N, P, K values (0.0 to 1.0 each).
    pub npk: [f32; 3],
    /// Normalized pH (0.0 to 14.0 scale).
    pub ph: f32,
    /// Normalized soil moisture (0.0 to 1.0).
    pub moisture: f32,
}

/// Pre-allocated scratch buffers for agricultural systems.
///
/// Avoids heap allocation on the hot path (`WorldState::step`).
#[derive(Debug, Clone, Default)]
pub struct AgriScratch {
    /// Buffer for crop scan results produced this tick.
    pub scan_results: Vec<CropScanResult>,
    /// Buffer for soil readings collected this tick.
    pub soil_readings: Vec<SoilReading>,
    /// Temporary buffer for disease spread candidates.
    pub disease_spread_candidates: Vec<(u16, u16)>,
    /// Per-agent flag buffer used by `process_report_generation` to mark
    /// agents that produced a field report this tick. Sized to `agents.len()`
    /// at the top of the agri pipeline and reused across ticks.
    pub report_flags: Vec<bool>,
}

impl AgriScratch {
    /// Creates a new scratch buffer with pre-allocated capacity.
    pub fn with_capacity(scan_cap: usize, soil_cap: usize) -> Self {
        Self {
            scan_results: Vec::with_capacity(scan_cap),
            soil_readings: Vec::with_capacity(soil_cap),
            disease_spread_candidates: Vec::with_capacity(scan_cap),
            report_flags: Vec::new(),
        }
    }

    /// Clears all buffers without deallocating.
    pub fn clear(&mut self) {
        self.scan_results.clear();
        self.soil_readings.clear();
        self.disease_spread_candidates.clear();
        self.report_flags.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_state_default() {
        let crop = CropState::default();
        assert_eq!(crop.growth_stage, 0);
        assert_eq!(crop.health, FIXED_POINT_ONE);
        assert_eq!(crop.disease_level, 0);
        assert_eq!(crop.moisture, FIXED_POINT_ONE);
        assert_eq!(crop.nutrients, FIXED_POINT_ONE);
        assert!(!crop.sprayed);
        assert_eq!(crop.surveyed_tick, 0);
    }

    #[test]
    fn test_crop_state_with_health() {
        let crop = CropState::with_health(FIXED_POINT_ONE / 2);
        assert_eq!(crop.health, FIXED_POINT_ONE / 2);
        assert_eq!(crop.disease_level, 0);
    }

    #[test]
    fn test_crop_state_with_health_clamped() {
        let crop = CropState::with_health(FIXED_POINT_ONE * 2);
        assert_eq!(crop.health, FIXED_POINT_ONE);
        let crop = CropState::with_health(-100);
        assert_eq!(crop.health, 0);
    }

    #[test]
    fn test_crop_is_diseased() {
        let mut crop = CropState::default();
        assert!(!crop.is_diseased());
        crop.disease_level = 100;
        assert!(crop.is_diseased());
    }

    #[test]
    fn test_crop_is_surveyed() {
        let mut crop = CropState::default();
        assert!(!crop.is_surveyed());
        crop.surveyed_tick = 42;
        assert!(crop.is_surveyed());
    }

    #[test]
    fn test_crop_ndvi_healthy() {
        let crop = CropState::default(); // health=1.0, disease=0
        let ndvi = crop.compute_ndvi();
        // Should be close to FIXED_POINT_ONE (1.0)
        assert!(
            ndvi > FIXED_POINT_ONE - 100,
            "healthy crop NDVI should be near 1.0, got {ndvi}"
        );
    }

    #[test]
    fn test_crop_ndvi_diseased() {
        let crop = CropState {
            health: FIXED_POINT_ONE / 2,
            disease_level: FIXED_POINT_ONE / 2,
            ..Default::default()
        };
        let ndvi = crop.compute_ndvi();
        // health == disease, so NDVI should be near 0
        assert!(
            ndvi.abs() < 100,
            "equal health/disease NDVI should be near 0, got {ndvi}"
        );
    }

    #[test]
    fn test_crop_ndvi_fully_diseased() {
        let crop = CropState {
            health: 0,
            disease_level: FIXED_POINT_ONE,
            ..Default::default()
        };
        let ndvi = crop.compute_ndvi();
        // Should be close to -FIXED_POINT_ONE (-1.0)
        assert!(
            ndvi < -FIXED_POINT_ONE + 100,
            "fully diseased NDVI should be near -1.0, got {ndvi}"
        );
    }

    #[test]
    fn test_soil_sensor_node_new() {
        let node = SoilSensorNode::new(42, Position::new(10, 20));
        assert_eq!(node.id, 42);
        assert_eq!(node.position, Position::new(10, 20));
        assert!(!node.collected);
        assert_eq!(node.last_read_tick, 0);
        assert_eq!(node.npk[0], FIXED_POINT_ONE / 2);
    }

    #[test]
    fn test_agri_scratch_clear() {
        let mut scratch = AgriScratch::with_capacity(16, 8);
        scratch.scan_results.push(CropScanResult {
            position: (0, 0),
            ndvi: 0.5,
            thermal: 0.3,
            disease_flag: false,
        });
        scratch.soil_readings.push(SoilReading {
            node_id: 0,
            npk: [0.5, 0.5, 0.5],
            ph: 7.0,
            moisture: 0.5,
        });
        assert!(!scratch.scan_results.is_empty());
        scratch.clear();
        assert!(scratch.scan_results.is_empty());
        assert!(scratch.soil_readings.is_empty());
    }

    #[test]
    fn test_crop_state_serde_roundtrip() {
        let crop = CropState {
            growth_stage: 3,
            health: FIXED_POINT_ONE / 2,
            disease_level: FIXED_POINT_ONE / 4,
            moisture: FIXED_POINT_ONE * 3 / 4,
            nutrients: FIXED_POINT_ONE / 3,
            sprayed: true,
            surveyed_tick: 100,
        };
        let json = serde_json::to_string(&crop).unwrap();
        let deser: CropState = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.growth_stage, 3);
        assert_eq!(deser.health, crop.health);
        assert_eq!(deser.disease_level, crop.disease_level);
        assert!(deser.sprayed);
    }

    #[test]
    fn test_soil_sensor_serde_roundtrip() {
        let node = SoilSensorNode::new(1, Position::new(5, 5));
        let json = serde_json::to_string(&node).unwrap();
        let deser: SoilSensorNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, 1);
        assert_eq!(deser.position, Position::new(5, 5));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// NDVI always stays within [-FIXED_POINT_ONE, FIXED_POINT_ONE].
            #[test]
            fn ndvi_bounded(
                health in 0i32..=FIXED_POINT_ONE,
                disease in 0i32..=FIXED_POINT_ONE,
            ) {
                let crop = CropState {
                    health,
                    disease_level: disease,
                    ..Default::default()
                };
                let ndvi = crop.compute_ndvi();
                prop_assert!(ndvi >= -FIXED_POINT_ONE, "NDVI {} below -1.0", ndvi);
                prop_assert!(ndvi <= FIXED_POINT_ONE, "NDVI {} above 1.0", ndvi);
            }

            /// CropState with_health always clamps to valid range.
            #[test]
            fn crop_health_clamped(health in -100_000i32..200_000) {
                let crop = CropState::with_health(health);
                prop_assert!(crop.health >= 0);
                prop_assert!(crop.health <= FIXED_POINT_ONE);
            }
        }
    }
}
