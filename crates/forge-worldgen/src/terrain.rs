//! Terrain generation combining Perlin noise with biome classification.
//!
//! [`TerrainGenerator`] produces a fully populated [`Grid`] by sampling
//! two independent noise layers (elevation and moisture), normalising them
//! to \[0, 1\], and feeding the results through a [`BiomeClassifier`].

use forge_types::config::WorldConfig;
use forge_types::constants;
use forge_types::grid::Grid;
use tracing::instrument;

use crate::biome::BiomeClassifier;
use crate::noise::PerlinNoise;

/// Offset applied to the moisture noise seed so that the elevation and
/// moisture layers are statistically independent.
const MOISTURE_SEED_OFFSET: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Generates terrain on a [`Grid`] using Perlin noise and biome rules.
#[derive(Debug, Clone)]
pub struct TerrainGenerator {
    /// Perlin noise for the elevation layer.
    elevation_noise: PerlinNoise,
    /// Perlin noise for the moisture layer.
    moisture_noise: PerlinNoise,
    /// Biome classifier that maps (elevation, moisture) to terrain.
    classifier: BiomeClassifier,
    /// Scale factor controlling terrain feature size.
    biome_scale: f64,
    /// Number of octaves for fractal noise.
    octaves: u32,
    /// Persistence (amplitude decay per octave).
    persistence: f64,
}

impl TerrainGenerator {
    /// Creates a new terrain generator from a config and seed.
    ///
    /// The `biome_scale` field from `config` controls feature frequency:
    /// lower values produce larger biome patches, higher values produce
    /// smaller, more detailed features.
    #[instrument(skip_all)]
    pub fn new(config: &WorldConfig, seed: u64) -> Self {
        let elevation_noise = PerlinNoise::new(seed);
        let moisture_noise = PerlinNoise::new(seed.wrapping_add(MOISTURE_SEED_OFFSET));
        let classifier = BiomeClassifier::new(config);
        let biome_scale = config.biome_scale as f64;

        // Derive octave count from biome_scale; persistence is a fixed constant.
        // A scale of 0.1 (default) yields 4 octaves.
        let octaves = ((biome_scale * constants::TERRAIN_NOISE_OCTAVES_MULTIPLIER).clamp(
            constants::TERRAIN_NOISE_OCTAVES_MIN as f64,
            constants::TERRAIN_NOISE_OCTAVES_MAX as f64,
        )) as u32;
        let persistence = constants::TERRAIN_NOISE_PERSISTENCE;

        tracing::trace!(
            seed,
            biome_scale,
            octaves,
            persistence,
            "created TerrainGenerator"
        );

        Self {
            elevation_noise,
            moisture_noise,
            classifier,
            biome_scale,
            octaves,
            persistence,
        }
    }

    /// Fills the given grid with terrain types derived from noise + biomes.
    ///
    /// Every tile's `terrain` and `elevation` fields are set.
    #[instrument(skip_all)]
    pub fn generate(&self, grid: &mut Grid) {
        let width = grid.width;
        let height = grid.height;

        tracing::trace!(width, height, "generating terrain");

        for y in 0..height {
            for x in 0..width {
                let nx = x as f64 * self.biome_scale;
                let ny = y as f64 * self.biome_scale;

                // Sample noise and normalise from [-1, 1] to [0, 1]
                let raw_elev =
                    self.elevation_noise
                        .octave_noise_2d(nx, ny, self.octaves, self.persistence);
                let raw_moist =
                    self.moisture_noise
                        .octave_noise_2d(nx, ny, self.octaves, self.persistence);

                let elevation = ((raw_elev + 1.0) / 2.0).clamp(0.0, 1.0);
                let moisture = ((raw_moist + 1.0) / 2.0).clamp(0.0, 1.0);

                let terrain = self.classifier.classify(elevation, moisture);

                if let Some(tile) = grid.get_mut(x, y) {
                    tile.terrain = terrain;
                    // Store elevation as u8 (0-255)
                    tile.elevation = (elevation * 255.0) as u8;
                }
            }
        }

        tracing::trace!("terrain generation complete");
    }

    /// Returns the raw elevation value at the given tile coordinate.
    ///
    /// Useful for downstream systems that need the continuous value.
    #[instrument(skip_all)]
    pub fn elevation_at(&self, x: u16, y: u16) -> f64 {
        let nx = x as f64 * self.biome_scale;
        let ny = y as f64 * self.biome_scale;
        let raw = self
            .elevation_noise
            .octave_noise_2d(nx, ny, self.octaves, self.persistence);
        ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Returns the raw moisture value at the given tile coordinate.
    #[instrument(skip_all)]
    pub fn moisture_at(&self, x: u16, y: u16) -> f64 {
        let nx = x as f64 * self.biome_scale;
        let ny = y as f64 * self.biome_scale;
        let raw = self
            .moisture_noise
            .octave_noise_2d(nx, ny, self.octaves, self.persistence);
        ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::grid::TerrainType;

    fn default_config() -> WorldConfig {
        WorldConfig::default()
    }

    #[test]
    fn test_generate_fills_grid() {
        let config = default_config();
        let gen = TerrainGenerator::new(&config, 42);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        // Every tile should have been touched -- at least some non-Ground terrain
        let mut terrain_counts = std::collections::HashMap::new();
        for tile in &grid.tiles {
            *terrain_counts.entry(tile.terrain).or_insert(0u32) += 1;
        }
        // We expect at least 2 different terrain types in any reasonable world
        assert!(
            terrain_counts.len() >= 2,
            "Expected multiple terrain types, got {terrain_counts:?}"
        );
    }

    #[test]
    fn test_determinism() {
        let config = default_config();
        let seed = 12345;

        let gen1 = TerrainGenerator::new(&config, seed);
        let mut grid1 = Grid::new(config.width, config.height);
        gen1.generate(&mut grid1);

        let gen2 = TerrainGenerator::new(&config, seed);
        let mut grid2 = Grid::new(config.width, config.height);
        gen2.generate(&mut grid2);

        for i in 0..grid1.tiles.len() {
            assert_eq!(
                grid1.tiles[i].terrain, grid2.tiles[i].terrain,
                "Terrain mismatch at tile {i}"
            );
            assert_eq!(
                grid1.tiles[i].elevation, grid2.tiles[i].elevation,
                "Elevation mismatch at tile {i}"
            );
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let config = default_config();

        let gen1 = TerrainGenerator::new(&config, 1);
        let mut grid1 = Grid::new(config.width, config.height);
        gen1.generate(&mut grid1);

        let gen2 = TerrainGenerator::new(&config, 2);
        let mut grid2 = Grid::new(config.width, config.height);
        gen2.generate(&mut grid2);

        let mut differences = 0u32;
        for i in 0..grid1.tiles.len() {
            if grid1.tiles[i].terrain != grid2.tiles[i].terrain {
                differences += 1;
            }
        }
        assert!(
            differences > 0,
            "Different seeds should produce different terrain"
        );
    }

    #[test]
    fn test_elevation_at_consistency() {
        let config = default_config();
        let gen = TerrainGenerator::new(&config, 99);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        // elevation_at should match the stored tile elevation (within u8 quantisation)
        for y in 0..config.height.min(10) {
            for x in 0..config.width.min(10) {
                let continuous = gen.elevation_at(x, y);
                let stored = grid.get(x, y).unwrap().elevation;
                let expected = (continuous * 255.0) as u8;
                assert_eq!(stored, expected, "Elevation mismatch at ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_moisture_at_values_in_range() {
        let config = default_config();
        let gen = TerrainGenerator::new(&config, 42);

        for y in 0..config.height.min(20) {
            for x in 0..config.width.min(20) {
                let moisture = gen.moisture_at(x, y);
                assert!(
                    (0.0..=1.0).contains(&moisture),
                    "Moisture out of range at ({x}, {y}): {moisture}"
                );
            }
        }
    }

    #[test]
    fn test_moisture_at_determinism() {
        let config = default_config();
        let gen1 = TerrainGenerator::new(&config, 42);
        let gen2 = TerrainGenerator::new(&config, 42);

        for y in 0..config.height.min(10) {
            for x in 0..config.width.min(10) {
                let m1 = gen1.moisture_at(x, y);
                let m2 = gen2.moisture_at(x, y);
                assert_eq!(m1, m2, "Moisture not deterministic at ({x}, {y})");
            }
        }
    }

    #[test]
    fn test_all_tiles_have_valid_terrain() {
        let config = default_config();
        let gen = TerrainGenerator::new(&config, 777);
        let mut grid = Grid::new(config.width, config.height);
        gen.generate(&mut grid);

        let valid = [
            TerrainType::Ground,
            TerrainType::Water,
            TerrainType::Sand,
            TerrainType::Forest,
            TerrainType::Mountain,
        ];
        for (i, tile) in grid.tiles.iter().enumerate() {
            assert!(
                valid.contains(&tile.terrain),
                "Unexpected terrain {:?} at tile {i}",
                tile.terrain
            );
        }
    }
}
