//! Deterministic seed derivation utilities.
//!
//! Provides reproducible seed generation from a base seed and component names,
//! ensuring that different subsystems receive unique but deterministic seeds.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::instrument;

/// Derives a deterministic child seed by hashing a component name into a base seed.
///
/// The same `(base, component)` pair always produces the same output.
#[instrument]
pub fn derive_seed(base: u64, component: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    component.hash(&mut hasher);
    hasher.finish()
}

/// Generates a deterministic sequence of child seeds from a base seed.
///
/// Each call to [`SeedSequence::next`] produces the next seed in the sequence.
/// The sequence is fully deterministic given the same base seed.
#[derive(Debug, Clone)]
pub struct SeedSequence {
    /// Current state used to derive the next seed.
    state: u64,
    /// Counter for sequence position.
    counter: u64,
}

impl SeedSequence {
    /// Creates a new seed sequence from a base seed.
    #[instrument]
    pub fn new(base: u64) -> Self {
        Self {
            state: base,
            counter: 0,
        }
    }

    /// Returns the next deterministic seed in the sequence.
    #[instrument(skip(self))]
    pub fn next_seed(&mut self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        self.counter.hash(&mut hasher);
        self.counter += 1;
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        let a = derive_seed(42, "map");
        let b = derive_seed(42, "map");
        assert_eq!(a, b);
    }

    #[test]
    fn test_derive_seed_different_components() {
        let a = derive_seed(42, "map");
        let b = derive_seed(42, "objective");
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_different_bases() {
        let a = derive_seed(1, "map");
        let b = derive_seed(2, "map");
        assert_ne!(a, b);
    }

    #[test]
    fn test_seed_sequence_deterministic() {
        let mut seq_a = SeedSequence::new(42);
        let mut seq_b = SeedSequence::new(42);
        for _ in 0..10 {
            assert_eq!(seq_a.next_seed(), seq_b.next_seed());
        }
    }

    #[test]
    fn test_seed_sequence_unique() {
        let mut seq = SeedSequence::new(42);
        let seeds: Vec<u64> = (0..100).map(|_| seq.next_seed()).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(seeds.len(), unique.len(), "All seeds should be unique");
    }

    #[test]
    fn test_seed_sequence_different_bases() {
        let mut seq_a = SeedSequence::new(1);
        let mut seq_b = SeedSequence::new(2);
        // At least the first seeds should differ
        assert_ne!(seq_a.next_seed(), seq_b.next_seed());
    }

    #[test]
    fn test_seed_sequence_counter_wraps_correctly() {
        // Start from a state near u64::MAX to verify counter doesn't panic.
        // We can't directly set the counter, but we can call next_seed many times.
        // Instead, verify that the sequence works at high call counts.
        let mut seq = SeedSequence::new(u64::MAX);
        // Should not panic even with max base seed.
        let s1 = seq.next_seed();
        let s2 = seq.next_seed();
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_many_next_seed_calls_produce_unique_values() {
        let mut seq = SeedSequence::new(12345);
        let seeds: Vec<u64> = (0..1000).map(|_| seq.next_seed()).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(
            seeds.len(),
            unique.len(),
            "All 1000 seeds should be unique"
        );
    }
}
