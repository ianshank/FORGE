//! Preference memory: learned action tendencies and value weights.
//!
//! Preferences capture what an agent tends to choose in specific contexts,
//! enabling stable personality and consistent decision-making across episodes.

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// A learned preference for a specific context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    /// Context key (e.g., "combat.low_health", "social.greeting").
    pub context_key: String,
    /// Weighted action preferences: (action_id, weight).
    pub action_weights: Vec<(u32, f32)>,
    /// Number of times this preference has been updated.
    pub update_count: u32,
    /// Memory strength.
    pub strength: f32,
}

impl Preference {
    /// Creates a new preference with default strength.
    #[instrument(skip_all)]
    pub fn new(context_key: String) -> Self {
        Self {
            context_key,
            action_weights: Vec::new(),
            update_count: 0,
            strength: 1.0,
        }
    }

    /// Updates the weight for a specific action using exponential moving average.
    ///
    /// The `strength_increment` parameter controls how much the memory
    /// is reinforced on each update (typically from config).
    #[instrument(skip_all)]
    pub fn update_action(
        &mut self,
        action_id: u32,
        reward: f32,
        learning_rate: f32,
        strength_increment: f32,
    ) {
        if let Some(entry) = self
            .action_weights
            .iter_mut()
            .find(|(id, _)| *id == action_id)
        {
            entry.1 = entry.1 * (1.0 - learning_rate) + reward * learning_rate;
        } else {
            self.action_weights.push((action_id, reward));
        }
        self.update_count += 1;
        self.strength = (self.strength + strength_increment).min(1.0);
        trace!(
            context = %self.context_key,
            action_id,
            update_count = self.update_count,
            "preference updated"
        );
    }

    /// Returns the preferred action (highest weight).
    #[instrument(skip_all)]
    pub fn preferred_action(&self) -> Option<u32> {
        self.action_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(id, _)| *id)
    }

    /// Applies decay to this preference's strength.
    #[instrument(skip_all)]
    pub fn decay(&mut self, rate: f32) {
        self.strength = (self.strength - rate).max(0.0);
    }
}

/// Bounded collection of preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceMemory {
    /// Stored preferences.
    preferences: Vec<Preference>,
    /// Maximum capacity.
    capacity: usize,
}

impl PreferenceMemory {
    /// Creates a new preference memory with the given capacity.
    #[instrument(skip_all)]
    pub fn new(capacity: usize) -> Self {
        Self {
            preferences: Vec::new(),
            capacity,
        }
    }

    /// Gets or creates a preference for the given context key.
    #[instrument(skip_all)]
    pub fn get_or_create(&mut self, context_key: &str) -> &mut Preference {
        let pos = self
            .preferences
            .iter()
            .position(|p| p.context_key == context_key);
        if let Some(idx) = pos {
            return &mut self.preferences[idx];
        }

        if self.preferences.len() >= self.capacity {
            if let Some(min_idx) = self
                .preferences
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|(i, _)| i)
            {
                self.preferences.swap_remove(min_idx);
            }
        }

        self.preferences
            .push(Preference::new(context_key.to_string()));
        let last = self.preferences.len() - 1;
        &mut self.preferences[last]
    }

    /// Retrieves a preference by context key.
    #[instrument(skip_all)]
    pub fn get(&self, context_key: &str) -> Option<&Preference> {
        self.preferences
            .iter()
            .find(|p| p.context_key == context_key)
    }

    /// Applies decay to all preferences and prunes those below the threshold.
    #[instrument(skip_all)]
    pub fn tick_decay(&mut self, rate: f32, min_strength: f32) {
        for pref in &mut self.preferences {
            pref.decay(rate);
        }
        self.preferences.retain(|p| p.strength >= min_strength);
    }

    /// Returns the number of stored preferences.
    pub fn len(&self) -> usize {
        self.preferences.len()
    }

    /// Returns `true` if no preferences are stored.
    pub fn is_empty(&self) -> bool {
        self.preferences.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preference_update() {
        let mut pref = Preference::new("combat".into());
        pref.update_action(1, 1.0, 0.5, 0.05);
        pref.update_action(2, 0.5, 0.5, 0.05);
        assert_eq!(pref.preferred_action(), Some(1));
        assert_eq!(pref.update_count, 2);
    }

    #[test]
    fn test_preference_memory() {
        let mut mem = PreferenceMemory::new(100);
        let pref = mem.get_or_create("combat.low_health");
        pref.update_action(3, 1.0, 0.5, 0.05);
        assert_eq!(mem.len(), 1);
        assert_eq!(
            mem.get("combat.low_health").unwrap().preferred_action(),
            Some(3)
        );
    }

    #[test]
    fn test_preference_eviction() {
        let mut mem = PreferenceMemory::new(2);
        {
            let p = mem.get_or_create("a");
            p.strength = 0.1;
        }
        mem.get_or_create("b");
        mem.get_or_create("c"); // should evict "a"
        assert_eq!(mem.len(), 2);
        assert!(mem.get("a").is_none());
    }

    #[test]
    fn test_strength_increment_bounded() {
        let mut pref = Preference::new("ctx".into());
        pref.strength = 0.98;
        pref.update_action(1, 1.0, 0.5, 0.05);
        assert!(pref.strength <= 1.0);
    }

    #[test]
    fn test_update_nonexistent_action_creates_it() {
        let mut pref = Preference::new("combat".into());
        assert!(pref.action_weights.is_empty());

        // First insertion: action does not exist yet, should be created with
        // weight equal to the reward value.
        pref.update_action(42, 0.8, 0.5, 0.05);
        assert_eq!(pref.action_weights.len(), 1);
        assert_eq!(pref.action_weights[0], (42, 0.8));
        assert_eq!(pref.update_count, 1);

        // Second call: now the action exists, EMA applies.
        pref.update_action(42, 0.2, 0.5, 0.05);
        assert_eq!(pref.action_weights.len(), 1);
        // EMA: 0.8 * 0.5 + 0.2 * 0.5 = 0.5
        let w = pref.action_weights[0].1;
        assert!((w - 0.5).abs() < 1e-5);
        assert_eq!(pref.update_count, 2);
    }

    #[test]
    fn test_preferred_action_single_entry() {
        let mut pref = Preference::new("ctx".into());
        pref.update_action(7, 0.3, 0.5, 0.05);
        assert_eq!(pref.preferred_action(), Some(7));
    }

    #[test]
    fn test_preferred_action_all_equal_weights() {
        let mut pref = Preference::new("ctx".into());
        // Insert three actions all with the same reward (and they are new,
        // so each gets the raw reward as its weight).
        pref.action_weights.push((1, 0.5));
        pref.action_weights.push((2, 0.5));
        pref.action_weights.push((3, 0.5));

        // preferred_action uses max_by which returns the last max for equal
        // elements; we just verify it returns *some* valid action.
        let preferred = pref.preferred_action().unwrap();
        assert!([1, 2, 3].contains(&preferred));
    }

    #[test]
    fn test_preferred_action_empty() {
        let pref = Preference::new("ctx".into());
        assert_eq!(pref.preferred_action(), None);
    }

    #[test]
    fn test_get_or_create_eviction_at_capacity() {
        let mut mem = PreferenceMemory::new(2);

        // Fill to capacity with differing strengths.
        {
            let p1 = mem.get_or_create("alpha");
            p1.strength = 0.2;
        }
        {
            let p2 = mem.get_or_create("beta");
            p2.strength = 0.8;
        }
        assert_eq!(mem.len(), 2);

        // Creating a third should evict the weakest ("alpha").
        mem.get_or_create("gamma");
        assert_eq!(mem.len(), 2);
        assert!(mem.get("alpha").is_none(), "weakest preference should be evicted");
        assert!(mem.get("beta").is_some());
        assert!(mem.get("gamma").is_some());

        // Accessing an existing key at capacity should NOT evict.
        mem.get_or_create("beta");
        assert_eq!(mem.len(), 2);
        assert!(mem.get("beta").is_some());
        assert!(mem.get("gamma").is_some());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn capacity_never_exceeded(
            cap in 1_usize..30,
            num_inserts in 0_usize..100
        ) {
            let mut mem = PreferenceMemory::new(cap);
            for i in 0..num_inserts {
                mem.get_or_create(&format!("ctx_{i}"));
            }
            prop_assert!(mem.len() <= cap);
        }

        #[test]
        fn ema_stays_bounded(
            initial in -10.0_f32..10.0,
            reward in -10.0_f32..10.0,
            lr in 0.0_f32..=1.0,
            increment in 0.0_f32..=0.5
        ) {
            let mut pref = Preference::new("test".into());
            pref.action_weights.push((0, initial));
            pref.update_action(0, reward, lr, increment);
            // EMA is a convex combination when lr in [0,1]:
            // new = old * (1-lr) + reward * lr
            let weight = pref.action_weights[0].1;
            let expected_min = initial.min(reward);
            let expected_max = initial.max(reward);
            // Allow small floating-point tolerance
            prop_assert!(weight >= expected_min - 0.01);
            prop_assert!(weight <= expected_max + 0.01);
        }

        #[test]
        fn strength_always_in_0_1(
            initial in 0.0_f32..=1.0,
            increment in 0.0_f32..=1.0
        ) {
            let mut pref = Preference::new("test".into());
            pref.strength = initial;
            pref.update_action(0, 1.0, 0.5, increment);
            prop_assert!(pref.strength >= 0.0);
            prop_assert!(pref.strength <= 1.0);
        }
    }
}
