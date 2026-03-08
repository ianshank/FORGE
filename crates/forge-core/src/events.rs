//! Event system for simulation events.
//!
//! Provides [`SimulationEvent`] variants for all observable simulation events
//! and an [`EventLog`] buffer for collecting and querying them. Events are
//! consumed by the dashboard, replay recorder, and analytics pipelines.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use tracing::instrument;

/// Default maximum number of events retained in an [`EventLog`].
const DEFAULT_MAX_EVENTS: usize = 10_000;

/// A discrete event emitted by the simulation engine during a tick.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SimulationEvent {
    /// Two agents engaged in combat.
    Combat {
        /// Identifier of the attacking agent.
        attacker_id: u32,
        /// Identifier of the defending agent.
        defender_id: u32,
        /// Damage dealt (positive = damage, negative = heal).
        damage: i32,
        /// World position where the combat occurred.
        position: (u16, u16),
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent picked up a resource.
    ResourcePickup {
        /// Identifier of the agent.
        agent_id: u32,
        /// Type discriminant for the resource.
        resource_type: u8,
        /// World position of the pickup.
        position: (u16, u16),
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent dropped a resource.
    ResourceDrop {
        /// Identifier of the agent.
        agent_id: u32,
        /// Type discriminant for the resource.
        resource_type: u8,
        /// World position of the drop.
        position: (u16, u16),
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent broadcast a communication token.
    MessageSent {
        /// Identifier of the sending agent.
        sender_id: u32,
        /// The token that was broadcast.
        token: u16,
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent died.
    AgentDeath {
        /// Identifier of the agent that died.
        agent_id: u32,
        /// World position where the death occurred.
        position: (u16, u16),
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent completed a crafting recipe.
    CraftComplete {
        /// Identifier of the crafting agent.
        agent_id: u32,
        /// Recipe that was completed.
        recipe_id: u16,
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent completed a task and received a reward.
    TaskComplete {
        /// Identifier of the agent.
        agent_id: u32,
        /// Identifier of the completed task.
        task_id: String,
        /// Reward granted for task completion.
        reward: f32,
        /// Simulation tick when the event occurred.
        tick: u64,
    },
    /// An agent declared an intent (theory-of-mind signal).
    IntentDeclared {
        /// Identifier of the agent.
        agent_id: u32,
        /// Human-readable label for the declared intent.
        intent_label: String,
        /// Confidence value in `[0.0, 1.0]`.
        confidence: f64,
        /// Simulation tick when the event occurred.
        tick: u64,
    },
}

/// A bounded, append-only log of [`SimulationEvent`]s.
///
/// When the number of stored events exceeds [`max_events`](EventLog::max_events),
/// the oldest events are evicted to make room.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventLog {
    /// The stored events, ordered by insertion time.
    events: VecDeque<SimulationEvent>,
    /// Maximum number of events to retain before eviction.
    max_events: usize,
}

impl Default for EventLog {
    fn default() -> Self {
        Self {
            events: VecDeque::new(),
            max_events: DEFAULT_MAX_EVENTS,
        }
    }
}

impl EventLog {
    /// Creates a new [`EventLog`] with the specified capacity limit.
    #[instrument(level = "debug")]
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::new(),
            max_events,
        }
    }

    /// Appends an event to the log, evicting the oldest event if the log
    /// has reached its maximum capacity.
    #[instrument(level = "trace", skip(self))]
    pub fn push(&mut self, event: SimulationEvent) {
        if self.max_events > 0 && self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Returns references to all events that occurred at the given tick.
    #[instrument(level = "debug", skip(self))]
    pub fn events_at_tick(&self, tick: u64) -> Vec<&SimulationEvent> {
        self.events
            .iter()
            .filter(|e| event_tick(e) == tick)
            .collect()
    }

    /// Removes all events from the log.
    #[instrument(level = "debug", skip(self))]
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Returns the number of events currently stored.
    #[instrument(level = "trace", skip(self))]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the log contains no events.
    #[instrument(level = "trace", skip(self))]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Extracts the tick from any [`SimulationEvent`] variant.
fn event_tick(event: &SimulationEvent) -> u64 {
    match event {
        SimulationEvent::Combat { tick, .. }
        | SimulationEvent::ResourcePickup { tick, .. }
        | SimulationEvent::ResourceDrop { tick, .. }
        | SimulationEvent::MessageSent { tick, .. }
        | SimulationEvent::AgentDeath { tick, .. }
        | SimulationEvent::CraftComplete { tick, .. }
        | SimulationEvent::TaskComplete { tick, .. }
        | SimulationEvent::IntentDeclared { tick, .. } => *tick,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn combat_event(tick: u64) -> SimulationEvent {
        SimulationEvent::Combat {
            attacker_id: 1,
            defender_id: 2,
            damage: 10,
            position: (5, 5),
            tick,
        }
    }

    fn pickup_event(tick: u64) -> SimulationEvent {
        SimulationEvent::ResourcePickup {
            agent_id: 1,
            resource_type: 0,
            position: (3, 4),
            tick,
        }
    }

    #[test]
    fn push_and_len() {
        let mut log = EventLog::new(100);
        assert!(log.is_empty());
        log.push(combat_event(1));
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn evicts_oldest_when_over_max() {
        let mut log = EventLog::new(3);
        log.push(combat_event(1));
        log.push(combat_event(2));
        log.push(combat_event(3));
        assert_eq!(log.len(), 3);

        // Pushing a 4th event should evict tick-1
        log.push(combat_event(4));
        assert_eq!(log.len(), 3);

        let ticks: Vec<u64> = log.events.iter().map(event_tick).collect();
        assert_eq!(ticks, vec![2, 3, 4]);
    }

    #[test]
    fn events_at_tick_filters_correctly() {
        let mut log = EventLog::new(100);
        log.push(combat_event(1));
        log.push(pickup_event(1));
        log.push(combat_event(2));
        log.push(pickup_event(3));

        let tick_1 = log.events_at_tick(1);
        assert_eq!(tick_1.len(), 2);

        let tick_2 = log.events_at_tick(2);
        assert_eq!(tick_2.len(), 1);

        let tick_99 = log.events_at_tick(99);
        assert!(tick_99.is_empty());
    }

    #[test]
    fn clear_empties_log() {
        let mut log = EventLog::new(100);
        log.push(combat_event(1));
        log.push(combat_event(2));
        assert_eq!(log.len(), 2);

        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn empty_state() {
        let log = EventLog::default();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.events_at_tick(0).is_empty());
    }

    #[test]
    fn serialization_roundtrip() {
        let mut log = EventLog::new(100);
        log.push(SimulationEvent::TaskComplete {
            agent_id: 7,
            task_id: "gather_wood".to_string(),
            reward: 1.5,
            tick: 42,
        });
        log.push(SimulationEvent::IntentDeclared {
            agent_id: 3,
            intent_label: "explore".to_string(),
            confidence: 0.95,
            tick: 42,
        });

        let json = serde_json::to_string(&log).expect("serialize");
        let restored: EventLog = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.len(), log.len());
        assert_eq!(restored.events, log.events);
    }

    #[test]
    fn default_max_events() {
        let log = EventLog::default();
        assert_eq!(log.max_events, DEFAULT_MAX_EVENTS);
    }
}
