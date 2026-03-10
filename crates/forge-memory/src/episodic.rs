//! Episodic memory: records of past events indexed by time, agents, and location.
//!
//! Episodic memories capture "what happened when" — sequences of events
//! from specific time periods, involving specific agents, at specific locations.
//! They enable agents to learn from past experience and recognize patterns.

use serde::{Deserialize, Serialize};
use tracing::{instrument, trace};

/// Outcome classification for an episode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EpisodeOutcome {
    /// The agent achieved its goal.
    Success,
    /// The agent failed to achieve its goal.
    Failure,
    /// The episode ended without a clear outcome.
    Neutral,
    /// The episode is still in progress.
    InProgress,
}

/// A recorded episode of agent experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Tick range this episode covers (start, end).
    pub tick_range: (u64, u64),
    /// IDs of agents involved in this episode.
    pub agent_ids: Vec<u32>,
    /// Primary location where the episode took place.
    pub location: (u16, u16),
    /// Summary of key events (action descriptions, not full SimulationEvents).
    pub event_summaries: Vec<String>,
    /// Outcome of the episode.
    pub outcome: EpisodeOutcome,
    /// Total reward accumulated during this episode.
    pub reward: f32,
    /// Tags for categorization (e.g., "combat", "crafting", "social").
    pub tags: Vec<String>,
    /// Memory strength (decays over time).
    pub strength: f32,
}

impl Episode {
    /// Creates a new episode record.
    #[instrument(skip_all)]
    pub fn new(
        tick_range: (u64, u64),
        agent_ids: Vec<u32>,
        location: (u16, u16),
        outcome: EpisodeOutcome,
        reward: f32,
    ) -> Self {
        Self {
            tick_range,
            agent_ids,
            location,
            event_summaries: Vec::new(),
            outcome,
            reward,
            tags: Vec::new(),
            strength: 1.0,
        }
    }

    /// Applies decay to this episode's strength.
    pub fn decay(&mut self, rate: f32) {
        self.strength = (self.strength - rate).max(0.0);
    }
}

/// Bounded collection of episodic memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Stored episodes, ordered by insertion time.
    episodes: Vec<Episode>,
    /// Maximum capacity.
    capacity: usize,
}

impl EpisodicMemory {
    /// Creates a new episodic memory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            episodes: Vec::new(),
            capacity,
        }
    }

    /// Stores an episode, evicting the weakest if at capacity.
    #[instrument(skip_all)]
    pub fn store(&mut self, episode: Episode) {
        if self.episodes.len() >= self.capacity {
            if let Some(min_idx) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
                .map(|(i, _)| i)
            {
                self.episodes.swap_remove(min_idx);
            }
        }
        self.episodes.push(episode);
    }

    /// Queries episodes involving a specific agent.
    #[instrument(skip_all)]
    pub fn query_by_agent(&self, agent_id: u32) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| e.agent_ids.contains(&agent_id))
            .collect()
    }

    /// Queries episodes with a specific tag.
    #[instrument(skip_all)]
    pub fn query_by_tag(&self, tag: &str) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Queries episodes near a location (Manhattan distance).
    #[instrument(skip_all)]
    pub fn query_by_location(&self, x: u16, y: u16, radius: u16) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| {
                let dx = (e.location.0 as i32 - x as i32).unsigned_abs() as u16;
                let dy = (e.location.1 as i32 - y as i32).unsigned_abs() as u16;
                dx + dy <= radius
            })
            .collect()
    }

    /// Returns the most recent episodes, up to `n`.
    pub fn recent(&self, n: usize) -> Vec<&Episode> {
        self.episodes.iter().rev().take(n).collect()
    }

    /// Applies decay to all episodes and prunes those below the threshold.
    #[instrument(skip_all)]
    pub fn tick_decay(&mut self, rate: f32, min_strength: f32) {
        for ep in &mut self.episodes {
            ep.decay(rate);
        }
        self.episodes.retain(|e| e.strength >= min_strength);
        trace!(remaining = self.episodes.len(), "episodic memory decayed");
    }

    /// Returns the number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Returns `true` if no episodes are stored.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(tick: u64, agents: Vec<u32>, tag: &str) -> Episode {
        let mut ep = Episode::new(
            (tick, tick + 10),
            agents,
            (5, 5),
            EpisodeOutcome::Success,
            1.0,
        );
        ep.tags.push(tag.to_string());
        ep
    }

    #[test]
    fn test_store_and_query() {
        let mut mem = EpisodicMemory::new(100);
        mem.store(make_episode(0, vec![1, 2], "combat"));
        mem.store(make_episode(10, vec![1, 3], "crafting"));
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.query_by_agent(1).len(), 2);
        assert_eq!(mem.query_by_agent(2).len(), 1);
        assert_eq!(mem.query_by_tag("combat").len(), 1);
    }

    #[test]
    fn test_location_query() {
        let mut mem = EpisodicMemory::new(100);
        let mut ep = make_episode(0, vec![1], "nav");
        ep.location = (10, 10);
        mem.store(ep);
        assert_eq!(mem.query_by_location(10, 10, 0).len(), 1);
        assert_eq!(mem.query_by_location(12, 10, 3).len(), 1);
        assert_eq!(mem.query_by_location(20, 20, 5).len(), 0);
    }

    #[test]
    fn test_eviction() {
        let mut mem = EpisodicMemory::new(2);
        let mut ep1 = make_episode(0, vec![1], "a");
        ep1.strength = 0.1;
        mem.store(ep1);
        mem.store(make_episode(10, vec![1], "b"));
        mem.store(make_episode(20, vec![1], "c"));
        assert_eq!(mem.len(), 2);
    }

    #[test]
    fn test_recent() {
        let mut mem = EpisodicMemory::new(100);
        mem.store(make_episode(0, vec![1], "a"));
        mem.store(make_episode(10, vec![1], "b"));
        mem.store(make_episode(20, vec![1], "c"));
        let recent = mem.recent(2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_decay_prunes_weak() {
        let mut mem = EpisodicMemory::new(100);
        let mut ep = make_episode(0, vec![1], "a");
        ep.strength = 0.05;
        mem.store(ep);
        mem.tick_decay(0.01, 0.05);
        assert!(mem.is_empty());
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
            let mut mem = EpisodicMemory::new(cap);
            for i in 0..num_inserts {
                mem.store(Episode::new(
                    (i as u64, i as u64 + 10),
                    vec![0],
                    (0, 0),
                    EpisodeOutcome::Neutral,
                    0.0,
                ));
            }
            prop_assert!(mem.len() <= cap);
        }

        #[test]
        fn query_by_agent_returns_subset(
            num_episodes in 1_usize..50,
            query_agent in 0_u32..10
        ) {
            let mut mem = EpisodicMemory::new(100);
            for i in 0..num_episodes {
                let agents = vec![(i as u32) % 10];
                mem.store(Episode::new(
                    (i as u64, i as u64 + 10),
                    agents,
                    (0, 0),
                    EpisodeOutcome::Neutral,
                    0.0,
                ));
            }
            let results = mem.query_by_agent(query_agent);
            for ep in results {
                prop_assert!(ep.agent_ids.contains(&query_agent));
            }
        }
    }
}
