//! Observation space adapter: FORGE grid observations → flat state vectors.
//!
//! FORGE produces ego-centric grid views (11x11x7 by default) plus scalar
//! fields. MangoMAS encoders expect flat state vectors. This adapter
//! flattens and normalizes FORGE observations into a format suitable
//! for MangoMAS world model (RSSM) and policy network inputs.

use forge_types::observation::Observation;
use tracing::instrument;

use crate::config::{
    ObservationAdapterConfig, Platform, DRONE_FEATURES, GRID_SUMMARY_FEATURES, INVENTORY_FEATURES,
    SCALAR_FEATURES,
};
use crate::error::MangoMasResult;

/// Trait for adapting FORGE observations to MangoMAS-compatible state vectors.
pub trait ObservationAdapter: Send + Sync {
    /// Converts a FORGE observation to a flat f32 vector.
    fn adapt(&self, obs: &Observation) -> MangoMasResult<Vec<f32>>;

    /// Returns the dimensionality of the output vector.
    fn output_dim(&self) -> usize;
}

/// Flattens FORGE observations into a compact state vector.
///
/// The output vector contains:
/// - Grid summary: resource/agent/object counts from the visible grid
/// - Scalar fields: health, stamina, position (normalized), day_phase
/// - Drone fields (if platform is Drone): altitude, battery, morphology, heading
/// - Inventory summary: total item count, distinct types
pub struct FlatStateAdapter {
    config: ObservationAdapterConfig,
}

impl FlatStateAdapter {
    /// Creates a new flat state adapter.
    #[instrument(skip_all)]
    pub fn new(config: ObservationAdapterConfig) -> Self {
        Self { config }
    }

    /// Extract summary statistics from the ego-centric grid observation.
    fn grid_summary(obs: &Observation) -> Vec<f32> {
        let mut agent_count = 0u32;
        let mut object_count = 0u32;
        let mut resource_count = 0u32;
        let mut terrain_counts = [0u32; 8]; // Up to 8 terrain types

        for tile in &obs.grid_view {
            if tile.has_agent {
                agent_count += 1;
            }
            if tile.has_object {
                object_count += 1;
            }
            if tile.has_resource {
                resource_count += 1;
            }
            let terrain_idx = (tile.terrain as usize).min(terrain_counts.len() - 1);
            terrain_counts[terrain_idx] += 1;
        }

        let total_tiles = obs.grid_view.len().max(1) as f32;
        let mut features = vec![
            agent_count as f32 / total_tiles,
            object_count as f32 / total_tiles,
            resource_count as f32 / total_tiles,
        ];
        for &count in &terrain_counts {
            features.push(count as f32 / total_tiles);
        }
        features
    }

    /// Summarize inventory contents as a normalized feature vector.
    fn inventory_summary(&self, obs: &Observation) -> Vec<f32> {
        let total_items: u32 = obs
            .inventory
            .slots
            .iter()
            .map(|&(_, count)| count as u32)
            .sum();
        let occupied_slots = obs
            .inventory
            .slots
            .iter()
            .filter(|&&(_, count)| count > 0)
            .count();
        let capacity = obs.inventory.slots.len().max(1) as f32;

        vec![
            total_items as f32 / self.config.inventory_norm,
            occupied_slots as f32 / capacity,
        ]
    }
}

impl ObservationAdapter for FlatStateAdapter {
    #[instrument(skip_all)]
    fn adapt(&self, obs: &Observation) -> MangoMasResult<Vec<f32>> {
        let mut state = Vec::with_capacity(self.output_dim());

        // Grid summary (11 features: 3 counts + 8 terrain proportions)
        state.extend(Self::grid_summary(obs));

        // Scalar fields (normalized)
        state.push(obs.health);
        state.push(obs.stamina);
        state.push(obs.position.0 as f32 / self.config.max_world_dim);
        state.push(obs.position.1 as f32 / self.config.max_world_dim);
        state.push(obs.day_phase as f32 / self.config.max_day_phase);

        // Inventory summary
        state.extend(self.inventory_summary(obs));

        // Drone fields (always included for drone platform)
        if self.config.platform == Platform::Drone {
            state.push(obs.altitude as f32 / self.config.max_altitude);
            state.push(obs.battery);
            state.push(obs.morphology as f32 / self.config.max_morphology);
            state.push(obs.heading as f32 / self.config.max_heading);
        }

        // Raw grid tiles (optional, for RSSM encoder input)
        if self.config.include_raw_grid {
            for tile in &obs.grid_view {
                state.push(tile.terrain as f32 / self.config.max_terrain);
                state.push(if tile.has_agent { 1.0 } else { 0.0 });
                state.push(if tile.has_object { 1.0 } else { 0.0 });
                state.push(if tile.has_resource { 1.0 } else { 0.0 });
                state.push(tile.elevation as f32 / self.config.max_elevation);
                state.push(tile.object_type as f32 / self.config.max_type_id);
                state.push(tile.resource_type as f32 / self.config.max_type_id);
            }
        }

        Ok(state)
    }

    fn output_dim(&self) -> usize {
        let drone = if self.config.platform == Platform::Drone {
            DRONE_FEATURES
        } else {
            0
        };
        let raw_grid = if self.config.include_raw_grid {
            self.config.grid_summary_dim
        } else {
            0
        };

        GRID_SUMMARY_FEATURES + SCALAR_FEATURES + INVENTORY_FEATURES + drone + raw_grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::observation::{InventoryObservation, TileObservation};

    fn make_obs() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default(); 9],
            view_width: 3,
            view_height: 3,
            inventory: InventoryObservation {
                slots: vec![(0, 5), (1, 3), (255, 0)],
            },
            health: 0.8,
            stamina: 0.6,
            position: (32, 48),
            messages: vec![],
            day_phase: 1,
            task_progress: vec![],
            altitude: 3,
            battery: 0.75,
            morphology: 2,
            heading: 1,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    #[test]
    fn test_flat_state_adapter_car() {
        let adapter = FlatStateAdapter::new(ObservationAdapterConfig {
            platform: Platform::Car,
            ..ObservationAdapterConfig::default()
        });
        let obs = make_obs();
        let state = adapter.adapt(&obs).unwrap();
        assert_eq!(state.len(), adapter.output_dim());
        // No drone fields for car
        assert_eq!(adapter.output_dim(), 11 + 5 + 2);
    }

    #[test]
    fn test_flat_state_adapter_drone() {
        let adapter = FlatStateAdapter::new(ObservationAdapterConfig {
            platform: Platform::Drone,
            ..ObservationAdapterConfig::default()
        });
        let obs = make_obs();
        let state = adapter.adapt(&obs).unwrap();
        assert_eq!(state.len(), adapter.output_dim());
        // Drone adds 4 fields
        assert_eq!(adapter.output_dim(), 11 + 5 + 2 + 4);
    }

    #[test]
    fn test_health_stamina_in_output() {
        let adapter = FlatStateAdapter::new(ObservationAdapterConfig {
            platform: Platform::Car,
            ..ObservationAdapterConfig::default()
        });
        let obs = make_obs();
        let state = adapter.adapt(&obs).unwrap();
        // Grid summary is 11 elements, then health at index 11
        assert!((state[11] - 0.8).abs() < 1e-6);
        assert!((state[12] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_output_values_normalized() {
        let adapter = FlatStateAdapter::new(ObservationAdapterConfig {
            platform: Platform::Drone,
            ..ObservationAdapterConfig::default()
        });
        let obs = make_obs();
        let state = adapter.adapt(&obs).unwrap();
        for (i, &val) in state.iter().enumerate() {
            assert!(
                (0.0..=1.1).contains(&val),
                "value at index {} is {} (out of normalized range)",
                i,
                val
            );
        }
    }

    #[test]
    fn test_empty_grid_view() {
        let adapter = FlatStateAdapter::new(ObservationAdapterConfig {
            platform: Platform::Car,
            ..ObservationAdapterConfig::default()
        });
        let mut obs = make_obs();
        obs.grid_view = vec![];
        let state = adapter.adapt(&obs).unwrap();
        assert_eq!(state.len(), adapter.output_dim());
    }
}
