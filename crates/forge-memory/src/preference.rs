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
    pub fn new(context_key: String) -> Self {
        Self {
            context_key,
            action_weights: Vec::new(),
            update_count: 0,
            strength: 1.0,
        }
    }

    /// Updates the weight for a specific action using exponential moving average.
    #[instrument(skip_all)]
    pub fn update_action(&mut self, action_id: u32, reward: f32, learning_rate: f32) {
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
        self.strength = (self.strength + 0.05).min(1.0);
        trace!(
            context = %self.context_key,
            action_id,
            update_count = self.update_count,
            "preference updated"
        );
    }

    /// Returns the preferred action (highest weight).
    pub fn preferred_action(&self) -> Option<u32> {
        self.action_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(id, _)| *id)
    }

    /// Applies decay to this preference's strength.
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
        pref.update_action(1, 1.0, 0.5);
        pref.update_action(2, 0.5, 0.5);
        assert_eq!(pref.preferred_action(), Some(1));
        assert_eq!(pref.update_count, 2);
    }

    #[test]
    fn test_preference_memory() {
        let mut mem = PreferenceMemory::new(100);
        let pref = mem.get_or_create("combat.low_health");
        pref.update_action(3, 1.0, 0.5);
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
}
