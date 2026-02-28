//! Deterministic PRNG wrapper for the FORGE simulation.
//!
//! Uses PCG (Permuted Congruential Generator) for fast, deterministic,
//! seedable random number generation. The RNG state is part of WorldState
//! to ensure full reproducibility.

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Deterministic random number generator for FORGE simulations.
///
/// Wraps PCG64 for fast, reproducible random number generation.
/// The state can be serialized for MCTS state snapshots.
#[derive(Debug, Clone)]
pub struct ForgeRng {
    inner: Pcg64Mcg,
    /// The original seed, stored for debugging and replay.
    seed: u64,
    /// Number of values generated (for debugging).
    generation_count: u64,
}

/// Serializable RNG state for snapshots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RngState {
    /// The seed used to initialize the RNG.
    pub seed: u64,
    /// Number of values generated.
    pub generation_count: u64,
}

impl ForgeRng {
    /// Creates a new deterministic RNG from a seed.
    #[instrument(skip_all)]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Pcg64Mcg::seed_from_u64(seed),
            seed,
            generation_count: 0,
        }
    }

    /// Returns the original seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns a random u32.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.generation_count += 1;
        self.inner.gen()
    }

    /// Returns a random u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.generation_count += 1;
        self.inner.gen()
    }

    /// Returns a random f32 in [0.0, 1.0).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.generation_count += 1;
        self.inner.gen()
    }

    /// Returns a random value in [0, max) (exclusive).
    #[inline]
    pub fn next_range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.generation_count += 1;
        self.inner.gen_range(0..max)
    }

    /// Returns a random boolean with given probability of true.
    #[inline]
    pub fn next_bool(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }

    /// Saves the current RNG state for serialization.
    #[instrument(skip_all)]
    pub fn save_state(&self) -> RngState {
        RngState {
            seed: self.seed,
            generation_count: self.generation_count,
        }
    }

    /// Restores RNG from a saved state.
    #[instrument(skip_all)]
    pub fn from_state(state: &RngState) -> Self {
        // Recreate by replaying from seed
        // This ensures exact state reproduction
        let mut rng = Self::new(state.seed);
        for _ in 0..state.generation_count {
            let _ = rng.next_u64();
        }
        rng
    }

    /// Creates a derived RNG with a new seed mixed from the current state.
    /// Useful for generating sub-seeds for world generation layers.
    #[instrument(skip_all)]
    pub fn derive(&mut self, domain: u64) -> ForgeRng {
        let derived_seed = self.next_u64() ^ domain;
        ForgeRng::new(derived_seed)
    }

    /// Returns the number of values generated.
    pub fn generation_count(&self) -> u64 {
        self.generation_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let mut rng1 = ForgeRng::new(42);
        let mut rng2 = ForgeRng::new(42);

        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = ForgeRng::new(42);
        let mut rng2 = ForgeRng::new(43);

        // Overwhelmingly likely to produce different sequences
        let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_range() {
        let mut rng = ForgeRng::new(42);
        for _ in 0..1000 {
            let val = rng.next_range(10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_range_zero() {
        let mut rng = ForgeRng::new(42);
        assert_eq!(rng.next_range(0), 0);
    }

    #[test]
    fn test_f32_range() {
        let mut rng = ForgeRng::new(42);
        for _ in 0..1000 {
            let val = rng.next_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_state_save_restore() {
        let mut rng = ForgeRng::new(42);
        // Advance the RNG
        for _ in 0..100 {
            rng.next_u64();
        }
        let state = rng.save_state();

        // Create a new RNG from the saved state
        let mut restored = ForgeRng::from_state(&state);

        // They should produce the same sequence going forward
        for _ in 0..100 {
            assert_eq!(rng.next_u64(), restored.next_u64());
        }
    }

    #[test]
    fn test_derive() {
        let mut rng1 = ForgeRng::new(42);
        let mut rng2 = ForgeRng::new(42);

        let derived1 = rng1.derive(1);
        let derived2 = rng2.derive(1);

        // Derived RNGs with same parent state and domain should be identical
        assert_eq!(derived1.seed(), derived2.seed());
    }

    #[test]
    fn test_generation_count() {
        let mut rng = ForgeRng::new(42);
        assert_eq!(rng.generation_count(), 0);
        rng.next_u32();
        assert_eq!(rng.generation_count(), 1);
        rng.next_f32();
        assert_eq!(rng.generation_count(), 2);
    }
}
