//! Semantic memory: persistent facts and concepts learned by an agent.
//!
//! Semantic memories represent declarative knowledge — things the agent
//! has learned about the world, other agents, or itself. Each fact has
//! a confidence score and decays over time unless reinforced.

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// A single semantic fact stored by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique key identifying this fact (e.g., "agent_3.preferred_resource").
    pub key: String,
    /// The fact value (e.g., "wood").
    pub value: String,
    /// Confidence in this fact (0.0 = uncertain, 1.0 = certain).
    pub confidence: f32,
    /// Tick when this fact was first learned.
    pub source_tick: u64,
    /// Current memory strength (decays over time, reinforced on access).
    pub strength: f32,
    /// Number of times this fact has been reinforced.
    pub reinforcement_count: u32,
}

impl SemanticFact {
    /// Creates a new semantic fact with full strength.
    #[instrument(skip_all)]
    pub fn new(key: String, value: String, confidence: f32, source_tick: u64) -> Self {
        Self {
            key,
            value,
            confidence,
            source_tick,
            strength: 1.0,
            reinforcement_count: 0,
        }
    }

    /// Applies decay to the memory strength.
    #[instrument(skip_all)]
    pub fn decay(&mut self, rate: f32) {
        self.strength = (self.strength - rate).max(0.0);
        trace!(key = %self.key, strength = self.strength, "semantic fact decayed");
    }

    /// Reinforces the memory, increasing its strength by the given increment.
    #[instrument(skip_all)]
    pub fn reinforce(&mut self, increment: f32) {
        self.strength = (self.strength + increment).min(1.0);
        self.reinforcement_count += 1;
        trace!(key = %self.key, strength = self.strength, "semantic fact reinforced");
    }
}

/// Bounded collection of semantic facts with LRU-style eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    /// Stored facts.
    facts: Vec<SemanticFact>,
    /// Maximum capacity.
    capacity: usize,
}

impl SemanticMemory {
    /// Creates a new semantic memory with the given capacity.
    #[instrument(skip_all)]
    pub fn new(capacity: usize) -> Self {
        Self {
            facts: Vec::new(),
            capacity,
        }
    }

    /// Stores a fact, replacing an existing fact with the same key
    /// or evicting the weakest fact if at capacity.
    #[instrument(skip_all)]
    pub fn store(&mut self, fact: SemanticFact) {
        // Replace existing fact with same key
        if let Some(pos) = self.facts.iter().position(|f| f.key == fact.key) {
            self.facts[pos] = fact;
            return;
        }

        if self.facts.len() >= self.capacity {
            // Evict the weakest fact
            if let Some(min_idx) = self
                .facts
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|(i, _)| i)
            {
                self.facts.swap_remove(min_idx);
            }
        }

        self.facts.push(fact);
    }

    /// Retrieves a fact by key.
    pub fn get(&self, key: &str) -> Option<&SemanticFact> {
        self.facts.iter().find(|f| f.key == key)
    }

    /// Returns all facts with strength above the given threshold.
    #[instrument(skip_all)]
    pub fn query_by_strength(&self, min_strength: f32) -> Vec<&SemanticFact> {
        self.facts
            .iter()
            .filter(|f| f.strength >= min_strength)
            .collect()
    }

    /// Returns facts whose key starts with the given prefix, up to `limit`.
    ///
    /// Unlike chaining `query_by_strength` + filter, this avoids allocating
    /// an intermediate `Vec` for the full fact set.
    #[instrument(skip_all)]
    pub fn query_by_prefix(&self, prefix: &str, limit: usize) -> Vec<&SemanticFact> {
        self.facts
            .iter()
            .filter(|f| f.key.starts_with(prefix))
            .take(limit)
            .collect()
    }

    /// Applies decay to all facts and prunes those below the threshold.
    #[instrument(skip_all)]
    pub fn tick_decay(&mut self, rate: f32, min_strength: f32) {
        for fact in &mut self.facts {
            fact.decay(rate);
        }
        self.facts.retain(|f| f.strength >= min_strength);
    }

    /// Returns the number of stored facts.
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Returns `true` if no facts are stored.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = SemanticMemory::new(100);
        let fact = SemanticFact::new("agent_1.role".into(), "gatherer".into(), 0.9, 10);
        mem.store(fact);
        assert_eq!(mem.len(), 1);
        let retrieved = mem.get("agent_1.role").unwrap();
        assert_eq!(retrieved.value, "gatherer");
    }

    #[test]
    fn test_replace_existing_key() {
        let mut mem = SemanticMemory::new(100);
        mem.store(SemanticFact::new("k".into(), "v1".into(), 0.5, 1));
        mem.store(SemanticFact::new("k".into(), "v2".into(), 0.8, 2));
        assert_eq!(mem.len(), 1);
        assert_eq!(mem.get("k").unwrap().value, "v2");
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut mem = SemanticMemory::new(2);
        let mut f1 = SemanticFact::new("a".into(), "1".into(), 1.0, 1);
        f1.strength = 0.1;
        mem.store(f1);
        mem.store(SemanticFact::new("b".into(), "2".into(), 1.0, 2));
        // At capacity, adding a third should evict the weakest (a)
        mem.store(SemanticFact::new("c".into(), "3".into(), 1.0, 3));
        assert_eq!(mem.len(), 2);
        assert!(mem.get("a").is_none());
    }

    #[test]
    fn test_decay_and_prune() {
        let mut mem = SemanticMemory::new(100);
        let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 1);
        fact.strength = 0.05;
        mem.store(fact);
        mem.tick_decay(0.01, 0.05);
        // 0.05 - 0.01 = 0.04 < 0.05 threshold → pruned
        assert!(mem.is_empty());
    }

    #[test]
    fn test_reinforce() {
        let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 1);
        fact.strength = 0.5;
        fact.reinforce(0.1);
        assert_eq!(fact.strength, 0.6);
        assert_eq!(fact.reinforcement_count, 1);
    }

    #[test]
    fn test_reinforce_capped_at_one() {
        let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 1);
        fact.strength = 0.95;
        fact.reinforce(0.2);
        assert_eq!(fact.strength, 1.0);
    }

    #[test]
    fn test_decay_capped_at_zero() {
        let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 1);
        fact.strength = 0.02;
        fact.decay(0.1);
        assert_eq!(fact.strength, 0.0);
    }

    #[test]
    fn test_query_by_strength() {
        let mut mem = SemanticMemory::new(100);
        let mut f1 = SemanticFact::new("a".into(), "1".into(), 1.0, 1);
        f1.strength = 0.3;
        mem.store(f1);
        let mut f2 = SemanticFact::new("b".into(), "2".into(), 1.0, 2);
        f2.strength = 0.7;
        mem.store(f2);
        assert_eq!(mem.query_by_strength(0.5).len(), 1);
        assert_eq!(mem.query_by_strength(0.0).len(), 2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn capacity_never_exceeded(
            cap in 1_usize..50,
            num_inserts in 0_usize..200
        ) {
            let mut mem = SemanticMemory::new(cap);
            for i in 0..num_inserts {
                mem.store(SemanticFact::new(
                    format!("key_{i}"),
                    format!("val_{i}"),
                    1.0,
                    i as u64,
                ));
            }
            prop_assert!(mem.len() <= cap);
        }

        #[test]
        fn decay_monotonically_decreases_strength(
            initial in 0.0_f32..=1.0,
            rate in 0.0_f32..=0.5
        ) {
            let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 0);
            fact.strength = initial;
            fact.decay(rate);
            prop_assert!(fact.strength <= initial);
            prop_assert!(fact.strength >= 0.0);
        }

        #[test]
        fn reinforce_bounded_to_one(
            initial in 0.0_f32..=1.0,
            increment in 0.0_f32..=1.0
        ) {
            let mut fact = SemanticFact::new("k".into(), "v".into(), 1.0, 0);
            fact.strength = initial;
            fact.reinforce(increment);
            prop_assert!(fact.strength >= initial);
            prop_assert!(fact.strength <= 1.0);
        }
    }
}
