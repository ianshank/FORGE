//! Deterministic Perlin noise implementation for terrain generation.
//!
//! Provides 2D Perlin noise with multi-octave fractal brownian motion.
//! No external noise crate dependency -- fully self-contained and deterministic
//! for any given seed.

use tracing::instrument;

/// A deterministic 2D Perlin noise generator.
///
/// Given the same seed, `sample_2d` and `octave_noise_2d` will always
/// return identical results for the same coordinates.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    /// Permutation table (doubled to avoid index wrapping).
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Creates a new Perlin noise generator with the given seed.
    ///
    /// The seed deterministically initialises the internal permutation table
    /// via a Fisher-Yates shuffle driven by a simple splitmix64 PRNG.
    #[instrument(skip_all)]
    pub fn new(seed: u64) -> Self {
        let mut perm_base: [u8; 256] = [0; 256];
        for i in 0..256u16 {
            perm_base[i as usize] = i as u8;
        }

        // Fisher-Yates shuffle with splitmix64
        let mut state = seed;
        for i in (1..256usize).rev() {
            state = splitmix64(state);
            let j = (state >> 32) as usize % (i + 1);
            perm_base.swap(i, j);
        }

        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = perm_base[i & 255];
        }

        tracing::trace!(seed, "created PerlinNoise generator");
        Self { perm }
    }

    /// Samples 2D Perlin noise at the given coordinates.
    ///
    /// Returns a value in the range \[-1.0, 1.0\].
    #[instrument(skip(self), level = "trace")]
    pub fn sample_2d(&self, x: f64, y: f64) -> f64 {
        // Determine grid cell coordinates
        let xi = fast_floor(x);
        let yi = fast_floor(y);

        // Relative position within the cell [0, 1)
        let xf = x - xi as f64;
        let yf = y - yi as f64;

        // Wrap grid coordinates to 0..255
        let xi = (xi & 255) as usize;
        let yi = (yi & 255) as usize;

        // Fade curves for interpolation
        let u = fade(xf);
        let v = fade(yf);

        // Hash corners
        let aa = self.perm[self.perm[xi] as usize + yi] as usize;
        let ab = self.perm[self.perm[xi] as usize + yi + 1] as usize;
        let ba = self.perm[self.perm[xi + 1] as usize + yi] as usize;
        let bb = self.perm[self.perm[xi + 1] as usize + yi + 1] as usize;

        // Gradient dot products at each corner
        let g_aa = grad2d(aa, xf, yf);
        let g_ba = grad2d(ba, xf - 1.0, yf);
        let g_ab = grad2d(ab, xf, yf - 1.0);
        let g_bb = grad2d(bb, xf - 1.0, yf - 1.0);

        // Bilinear interpolation
        let x1 = lerp(u, g_aa, g_ba);
        let x2 = lerp(u, g_ab, g_bb);
        lerp(v, x1, x2)
    }

    /// Computes multi-octave (fractal Brownian motion) 2D Perlin noise.
    ///
    /// Each successive octave doubles the frequency and reduces amplitude
    /// by the `persistence` factor.  The result is normalised so that it
    /// stays approximately within \[-1.0, 1.0\].
    ///
    /// * `octaves` -- number of noise layers (typically 1..8).
    /// * `persistence` -- amplitude decay per octave (typically 0.4..0.7).
    #[instrument(skip_all)]
    pub fn octave_noise_2d(&self, x: f64, y: f64, octaves: u32, persistence: f64) -> f64 {
        let mut total = 0.0_f64;
        let mut frequency = 1.0_f64;
        let mut amplitude = 1.0_f64;
        let mut max_value = 0.0_f64;

        for _ in 0..octaves {
            total += self.sample_2d(x * frequency, y * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= persistence;
            frequency *= 2.0;
        }

        if max_value > 0.0 {
            total / max_value
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Splitmix64 PRNG -- produces a deterministic sequence from a state.
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fast floor that handles negative values correctly.
///
/// For values outside `i32` range the result is clamped to `i32::MIN` / `i32::MAX`.
#[inline]
fn fast_floor(x: f64) -> i32 {
    let floored = x.floor();
    // Clamp to i32 range to prevent overflow on extreme inputs.
    if floored <= i32::MIN as f64 {
        i32::MIN
    } else if floored >= i32::MAX as f64 {
        i32::MAX
    } else {
        floored as i32
    }
}

/// Fade curve (6t^5 - 15t^4 + 10t^3) for smooth interpolation.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation.
#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

/// 2D gradient function -- picks one of 4 gradient directions based on hash.
#[inline]
fn grad2d(hash: usize, x: f64, y: f64) -> f64 {
    match hash & 3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let n1 = PerlinNoise::new(42);
        let n2 = PerlinNoise::new(42);

        for i in 0..50 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            assert_eq!(
                n1.sample_2d(x, y),
                n2.sample_2d(x, y),
                "Mismatch at ({x}, {y})"
            );
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let n1 = PerlinNoise::new(1);
        let n2 = PerlinNoise::new(2);
        // It is *theoretically* possible for two seeds to agree on a single
        // sample, but astronomically unlikely over many samples.
        let mut any_different = false;
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = i as f64 * 0.2;
            if (n1.sample_2d(x, y) - n2.sample_2d(x, y)).abs() > 1e-10 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Two different seeds should produce different noise"
        );
    }

    #[test]
    fn test_range() {
        let noise = PerlinNoise::new(123);
        for i in 0..1000 {
            let x = (i as f64) * 0.13 - 25.0;
            let y = (i as f64) * 0.17 - 30.0;
            let v = noise.sample_2d(x, y);
            assert!(
                (-1.5..=1.5).contains(&v),
                "Noise value {v} at ({x}, {y}) out of expected range"
            );
        }
    }

    #[test]
    fn test_octave_noise_determinism() {
        let n1 = PerlinNoise::new(99);
        let n2 = PerlinNoise::new(99);

        for i in 0..50 {
            let x = i as f64 * 0.23;
            let y = i as f64 * 0.41;
            assert_eq!(
                n1.octave_noise_2d(x, y, 4, 0.5),
                n2.octave_noise_2d(x, y, 4, 0.5),
            );
        }
    }

    #[test]
    fn test_octave_noise_range() {
        let noise = PerlinNoise::new(77);
        for i in 0..500 {
            let x = (i as f64) * 0.07 - 10.0;
            let y = (i as f64) * 0.11 - 15.0;
            let v = noise.octave_noise_2d(x, y, 6, 0.5);
            assert!(
                (-1.5..=1.5).contains(&v),
                "Octave noise value {v} at ({x}, {y}) out of expected range"
            );
        }
    }

    #[test]
    fn test_single_octave_equals_sample() {
        let noise = PerlinNoise::new(55);
        let x = std::f64::consts::PI;
        let y = std::f64::consts::E;
        let single = noise.octave_noise_2d(x, y, 1, 0.5);
        let direct = noise.sample_2d(x, y);
        assert!(
            (single - direct).abs() < 1e-12,
            "Single octave should equal direct sample"
        );
    }

    #[test]
    fn test_perlin_large_coordinates() {
        let noise = PerlinNoise::new(42);
        // Large coordinates within the safe i32 range (the internal fast_floor
        // casts to i32, so coordinates must stay within ~2.1e9).
        // Test with large but valid coordinates to ensure no panics or NaN.
        let coords = [1e3, -1e3, 1e5, -1e5, 5e5, -5e5, 1e6, -1e6];
        for &x in &coords {
            for &y in &coords {
                let v = noise.sample_2d(x, y);
                assert!(
                    v.is_finite(),
                    "sample_2d({x}, {y}) produced non-finite: {v}"
                );
            }
        }
        // Also test octave_noise_2d at large coordinates
        for &x in &coords {
            let v = noise.octave_noise_2d(x, x, 4, 0.5);
            assert!(
                v.is_finite(),
                "octave_noise_2d({x}, {x}) produced non-finite: {v}"
            );
        }
    }

    #[test]
    fn test_negative_coordinates() {
        let noise = PerlinNoise::new(10);
        // Should not panic or produce NaN
        let v = noise.sample_2d(-5.5, -3.3);
        assert!(v.is_finite());
    }

    // ---- Proptest: noise invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Perlin noise always returns values in [-1, 1].
            #[test]
            fn sample_2d_bounded(
                seed in 0u64..10_000,
                x in -500.0f64..500.0,
                y in -500.0f64..500.0,
            ) {
                let noise = PerlinNoise::new(seed);
                let v = noise.sample_2d(x, y);
                prop_assert!(v.is_finite());
                prop_assert!(v >= -1.0 && v <= 1.0,
                    "sample_2d({}, {}) = {} out of [-1, 1]", x, y, v);
            }

            /// Same seed produces same noise value (determinism).
            #[test]
            fn noise_determinism(
                seed in 0u64..10_000,
                x in -100.0f64..100.0,
                y in -100.0f64..100.0,
            ) {
                let n1 = PerlinNoise::new(seed);
                let n2 = PerlinNoise::new(seed);
                prop_assert_eq!(n1.sample_2d(x, y), n2.sample_2d(x, y));
            }

            /// Octave noise is finite for any octave count.
            #[test]
            fn octave_noise_finite(
                seed in 0u64..10_000,
                x in -100.0f64..100.0,
                y in -100.0f64..100.0,
                octaves in 1u32..8,
            ) {
                let noise = PerlinNoise::new(seed);
                let v = noise.octave_noise_2d(x, y, octaves, 0.5);
                prop_assert!(v.is_finite());
            }
        }
    }
}
