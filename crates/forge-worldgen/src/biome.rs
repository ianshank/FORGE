//! Biome classification rules for mapping elevation and moisture to terrain types.
//!
//! The [`BiomeClassifier`] converts continuous elevation and moisture values
//! (both in \[0.0, 1.0\]) into discrete [`TerrainType`] variants using
//! configurable thresholds derived from [`WorldConfig`].

use forge_types::config::WorldConfig;
use forge_types::grid::TerrainType;

/// Configurable thresholds that drive biome classification.
///
/// All thresholds are in the normalised \[0.0, 1.0\] range.
#[derive(Debug, Clone)]
pub struct BiomeThresholds {
    /// Elevation at or below which the tile is Water.
    pub water_level: f64,
    /// Elevation at or above which the tile is Mountain.
    pub mountain_level: f64,
    /// Elevation threshold for Sand (beach strip just above water).
    pub sand_level: f64,
    /// Moisture threshold above which a low-mid elevation tile becomes Forest.
    pub forest_moisture: f64,
    /// Moisture threshold below which a tile at mid elevation becomes Sand
    /// (desert-like areas).
    pub desert_moisture: f64,
}

impl BiomeThresholds {
    /// Derives thresholds from the [`WorldConfig`].
    ///
    /// The `biome_scale` field is repurposed as a general biome-intensity
    /// knob.  A higher `biome_scale` pushes the water level down and the
    /// mountain level higher, producing more varied terrain.
    pub fn from_config(config: &WorldConfig) -> Self {
        let scale = config.biome_scale as f64;

        // Sensible defaults that respond to biome_scale.
        // biome_scale of 0.1 (default) gives water=0.35, mountain=0.72
        let water_level = (0.35 - scale * 0.3).clamp(0.10, 0.50);
        let mountain_level = (0.72 + scale * 0.3).clamp(0.60, 0.90);
        let sand_level = water_level + 0.05;
        let forest_moisture = 0.45;
        let desert_moisture = 0.25;

        Self {
            water_level,
            mountain_level,
            sand_level,
            forest_moisture,
            desert_moisture,
        }
    }
}

/// Classifies (elevation, moisture) pairs into terrain types.
#[derive(Debug, Clone)]
pub struct BiomeClassifier {
    /// The thresholds driving classification.
    pub thresholds: BiomeThresholds,
}

impl BiomeClassifier {
    /// Creates a new classifier from a [`WorldConfig`].
    pub fn new(config: &WorldConfig) -> Self {
        let thresholds = BiomeThresholds::from_config(config);
        tracing::trace!(?thresholds, "created BiomeClassifier");
        Self { thresholds }
    }

    /// Creates a classifier with explicit thresholds.
    pub fn with_thresholds(thresholds: BiomeThresholds) -> Self {
        Self { thresholds }
    }

    /// Maps an (elevation, moisture) pair to a [`TerrainType`].
    ///
    /// Both `elevation` and `moisture` are expected to be in \[0.0, 1.0\].
    pub fn classify(&self, elevation: f64, moisture: f64) -> TerrainType {
        let t = &self.thresholds;

        if elevation <= t.water_level {
            TerrainType::Water
        } else if elevation <= t.sand_level {
            TerrainType::Sand
        } else if elevation >= t.mountain_level {
            TerrainType::Mountain
        } else if moisture >= t.forest_moisture {
            TerrainType::Forest
        } else if moisture <= t.desert_moisture {
            TerrainType::Sand
        } else {
            TerrainType::Ground
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_classifier() -> BiomeClassifier {
        BiomeClassifier::new(&WorldConfig::default())
    }

    #[test]
    fn test_deep_water() {
        let c = default_classifier();
        assert_eq!(c.classify(0.0, 0.5), TerrainType::Water);
        assert_eq!(c.classify(0.1, 0.5), TerrainType::Water);
    }

    #[test]
    fn test_mountain() {
        let c = default_classifier();
        assert_eq!(c.classify(1.0, 0.5), TerrainType::Mountain);
        assert_eq!(c.classify(0.95, 0.5), TerrainType::Mountain);
    }

    #[test]
    fn test_sand_beach() {
        let c = default_classifier();
        // Just above water level should be sand
        let elev = c.thresholds.water_level + 0.01;
        assert_eq!(c.classify(elev, 0.5), TerrainType::Sand);
    }

    #[test]
    fn test_forest_high_moisture() {
        let c = default_classifier();
        // Mid elevation, high moisture => Forest
        let elev = (c.thresholds.sand_level + c.thresholds.mountain_level) / 2.0;
        assert_eq!(c.classify(elev, 0.9), TerrainType::Forest);
    }

    #[test]
    fn test_ground_mid_moisture() {
        let c = default_classifier();
        let elev = (c.thresholds.sand_level + c.thresholds.mountain_level) / 2.0;
        assert_eq!(c.classify(elev, 0.35), TerrainType::Ground);
    }

    #[test]
    fn test_desert_low_moisture() {
        let c = default_classifier();
        let elev = (c.thresholds.sand_level + c.thresholds.mountain_level) / 2.0;
        assert_eq!(c.classify(elev, 0.1), TerrainType::Sand);
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = BiomeThresholds {
            water_level: 0.2,
            mountain_level: 0.8,
            sand_level: 0.25,
            forest_moisture: 0.5,
            desert_moisture: 0.2,
        };
        let c = BiomeClassifier::with_thresholds(thresholds);
        assert_eq!(c.classify(0.1, 0.5), TerrainType::Water);
        assert_eq!(c.classify(0.9, 0.5), TerrainType::Mountain);
        assert_eq!(c.classify(0.5, 0.7), TerrainType::Forest);
    }

    #[test]
    fn test_boundary_water_sand() {
        let c = default_classifier();
        // Exactly at water_level should be Water (<=)
        assert_eq!(
            c.classify(c.thresholds.water_level, 0.5),
            TerrainType::Water
        );
        // Just above should be Sand
        assert_eq!(
            c.classify(c.thresholds.water_level + 0.001, 0.5),
            TerrainType::Sand
        );
    }
}
