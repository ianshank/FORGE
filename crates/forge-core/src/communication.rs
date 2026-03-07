//! Communication system for multi-agent messaging.
//!
//! Agents can broadcast discrete communication tokens to nearby agents.
//! Tokens are drawn from a fixed vocabulary and delivered to recipients'
//! communication buffers based on proximity.

use forge_types::config::AgentConfig;
use forge_types::entity::Agent;
use forge_types::Action;
use tracing::{instrument, trace};

/// Processes Communicate actions: broadcasts tokens to agents within comm_radius.
///
/// When an agent performs `Communicate(token)`:
/// - Validates token is within vocabulary size
/// - Checks the sending agent has the `can_communicate` capability
/// - Finds all other alive agents within comm_radius (manhattan distance)
/// - Adds the token to each recipient's comm_buffer (evicting oldest if full)
///
/// If comm_radius is 0, the message is broadcast to ALL alive agents.
#[instrument(skip_all)]
pub fn process_communication(agents: &mut [Agent], actions: &[Action], config: &AgentConfig) {
    // Collect messages first to avoid borrow conflicts.
    // Each entry: (sender_index, token)
    let mut messages: Vec<(usize, u16)> = Vec::new();

    for (i, action) in actions.iter().enumerate() {
        if i >= agents.len() {
            continue;
        }

        let token = match action {
            Action::Communicate(t) => *t,
            _ => continue,
        };

        let agent = &agents[i];

        // Dead agents cannot send
        if !agent.alive {
            trace!(agent_id = agent.id, "dead agent cannot communicate");
            continue;
        }

        // Check capability
        if !agent.capabilities.can_communicate {
            trace!(agent_id = agent.id, "agent lacks communicate capability");
            continue;
        }

        // Validate token is within vocabulary
        if token >= config.comm_vocab_size {
            trace!(
                agent_id = agent.id,
                token = token,
                vocab_size = config.comm_vocab_size,
                "token exceeds vocabulary size"
            );
            continue;
        }

        messages.push((i, token));
    }

    // Deliver each message to agents within range
    for (sender_idx, token) in messages {
        let sender_pos = agents[sender_idx].position;
        let sender_id = agents[sender_idx].id;
        let comm_radius = config.comm_radius;
        let buffer_size = config.comm_buffer_size as usize;

        trace!(
            agent_id = sender_id,
            token = token,
            comm_radius = comm_radius,
            "broadcasting message"
        );

        for (j, recipient) in agents.iter_mut().enumerate() {
            if j == sender_idx {
                continue;
            }

            if !recipient.alive {
                continue;
            }

            // Check distance: comm_radius == 0 means global broadcast
            if comm_radius > 0 {
                let dist = sender_pos.manhattan_distance(&recipient.position);
                if dist > comm_radius as u32 {
                    continue;
                }
            }

            // Deliver: evict oldest if buffer is full
            if buffer_size > 0 && recipient.comm_buffer.len() >= buffer_size {
                recipient.comm_buffer.remove(0);
            }
            recipient.comm_buffer.push(token);

            trace!(
                sender_id = sender_id,
                recipient_id = recipient.id,
                token = token,
                "message delivered"
            );
        }
    }
}

/// Configuration for enhanced communication channels with bandwidth and latency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommChannelConfig {
    /// Maximum messages that can be sent per tick across all agents.
    pub max_messages_per_tick: u32,
    /// Latency in ticks: message arrives at `sent_tick + latency`.
    pub latency_ticks: u64,
    /// Whether any comm jamming zones are active.
    pub jamming_enabled: bool,
}

impl Default for CommChannelConfig {
    fn default() -> Self {
        Self {
            max_messages_per_tick: 64,
            latency_ticks: 0,
            jamming_enabled: false,
        }
    }
}

/// A pending message waiting for delivery after latency.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingMessage {
    /// Sender agent ID.
    pub sender_id: u32,
    /// Communication token.
    pub token: u16,
    /// Tick when the message should be delivered.
    pub delivery_tick: u64,
    /// Recipient agent index.
    pub recipient_idx: usize,
}

/// Enhanced communication channel with bandwidth limits and latency modeling.
#[derive(Debug, Clone, Default)]
pub struct CommChannel {
    /// Channel configuration.
    pub config: CommChannelConfig,
    /// Queue of messages waiting for delivery.
    pub pending: Vec<PendingMessage>,
    /// Positions of jammed zones: (center_x, center_y, radius).
    pub jammed_zones: Vec<(u16, u16, u16)>,
}

impl CommChannel {
    /// Creates a new comm channel with the given config.
    pub fn new(config: CommChannelConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            jammed_zones: Vec::new(),
        }
    }

    /// Returns `true` if the position is inside a jammed zone.
    pub fn is_jammed(&self, x: u16, y: u16) -> bool {
        if !self.config.jamming_enabled {
            return false;
        }
        let pos = forge_types::grid::Position::new(x, y);
        self.jammed_zones.iter().any(|&(cx, cy, r)| {
            let center = forge_types::grid::Position::new(cx, cy);
            pos.manhattan_distance(&center) <= r as u32
        })
    }

    /// Queues a message for delivery after the configured latency.
    pub fn queue_message(
        &mut self,
        sender_id: u32,
        token: u16,
        recipient_idx: usize,
        current_tick: u64,
    ) {
        let delivery_tick = current_tick + self.config.latency_ticks;
        self.pending.push(PendingMessage {
            sender_id,
            token,
            delivery_tick,
            recipient_idx,
        });
    }

    /// Delivers all messages whose `delivery_tick` has arrived.
    ///
    /// Returns the delivered messages and removes them from the queue.
    #[instrument(skip_all)]
    pub fn deliver_pending(&mut self, current_tick: u64, agents: &mut [Agent], buffer_size: usize) {
        let (delivered, remaining): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|msg| msg.delivery_tick <= current_tick);
        self.pending = remaining;

        for msg in delivered {
            if msg.recipient_idx < agents.len() {
                let recipient = &mut agents[msg.recipient_idx];
                if recipient.alive {
                    if buffer_size > 0 && recipient.comm_buffer.len() >= buffer_size {
                        recipient.comm_buffer.remove(0);
                    }
                    recipient.comm_buffer.push(msg.token);
                    trace!(
                        sender_id = msg.sender_id,
                        recipient_id = recipient.id,
                        token = msg.token,
                        "delayed message delivered"
                    );
                }
            }
        }
    }

    /// Returns the number of pending messages.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::config::AgentConfig;
    use forge_types::entity::Agent;
    use forge_types::grid::Position;

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_config() -> AgentConfig {
        AgentConfig::default()
    }

    #[test]
    fn test_broadcast_message() {
        let mut agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        let actions = vec![Action::Communicate(3), Action::Noop];
        let config = make_config();

        process_communication(&mut agents, &actions, &config);

        assert!(
            agents[0].comm_buffer.is_empty(),
            "sender should not receive own message"
        );
        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 3);
    }

    #[test]
    fn test_comm_radius_filtering() {
        let mut config = make_config();
        config.comm_radius = 3;

        let mut agents = vec![
            make_agent(0, 5, 5),   // sender
            make_agent(1, 6, 5),   // distance 1 — in range
            make_agent(2, 20, 20), // distance 30 — out of range
        ];
        let actions = vec![Action::Communicate(1), Action::Noop, Action::Noop];

        process_communication(&mut agents, &actions, &config);

        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert!(agents[2].comm_buffer.is_empty());
    }

    #[test]
    fn test_comm_buffer_overflow() {
        let mut config = make_config();
        config.comm_buffer_size = 3;
        config.comm_radius = 0; // global

        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];

        // Send 5 messages — buffer should only keep the latest 3
        for token in 0u16..5 {
            let actions = vec![Action::Communicate(token), Action::Noop];
            process_communication(&mut agents, &actions, &config);
        }

        assert_eq!(agents[1].comm_buffer.len(), 3);
        // Oldest messages (0 and 1) should have been evicted
        assert_eq!(agents[1].comm_buffer[0], 2);
        assert_eq!(agents[1].comm_buffer[1], 3);
        assert_eq!(agents[1].comm_buffer[2], 4);
    }

    #[test]
    fn test_dead_agent_no_send() {
        let mut agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        agents[0].alive = false;
        let actions = vec![Action::Communicate(1), Action::Noop];
        let config = make_config();

        process_communication(&mut agents, &actions, &config);

        assert!(agents[1].comm_buffer.is_empty());
    }

    #[test]
    fn test_dead_agent_no_receive() {
        let mut agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        agents[1].alive = false;
        let actions = vec![Action::Communicate(1), Action::Noop];
        let config = make_config();

        process_communication(&mut agents, &actions, &config);

        assert!(agents[1].comm_buffer.is_empty());
    }

    #[test]
    fn test_global_broadcast() {
        let mut config = make_config();
        config.comm_radius = 0; // global

        let mut agents = vec![
            make_agent(0, 0, 0),
            make_agent(1, 100, 100),
            make_agent(2, 200, 200),
        ];
        let actions = vec![Action::Communicate(7), Action::Noop, Action::Noop];

        process_communication(&mut agents, &actions, &config);

        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 7);
        assert_eq!(agents[2].comm_buffer.len(), 1);
        assert_eq!(agents[2].comm_buffer[0], 7);
    }

    #[test]
    fn test_vocab_validation() {
        let mut config = make_config();
        config.comm_vocab_size = 10;

        let mut agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        // Token 10 is out of vocabulary (valid: 0..9)
        let actions = vec![Action::Communicate(10), Action::Noop];

        process_communication(&mut agents, &actions, &config);

        assert!(agents[1].comm_buffer.is_empty());
    }

    #[test]
    fn test_can_communicate_flag() {
        let mut agents = vec![make_agent(0, 5, 5), make_agent(1, 6, 5)];
        agents[0].capabilities.can_communicate = false;
        let actions = vec![Action::Communicate(1), Action::Noop];
        let config = make_config();

        process_communication(&mut agents, &actions, &config);

        assert!(agents[1].comm_buffer.is_empty());
    }

    #[test]
    fn test_multiple_messages() {
        let mut config = make_config();
        config.comm_radius = 0; // global

        let mut agents = vec![
            make_agent(0, 0, 0),
            make_agent(1, 1, 0),
            make_agent(2, 2, 0),
        ];
        // Agents 0 and 2 both send
        let actions = vec![Action::Communicate(5), Action::Noop, Action::Communicate(9)];

        process_communication(&mut agents, &actions, &config);

        // Agent 0 should receive from agent 2 only
        assert_eq!(agents[0].comm_buffer.len(), 1);
        assert_eq!(agents[0].comm_buffer[0], 9);

        // Agent 1 should receive from both 0 and 2
        assert_eq!(agents[1].comm_buffer.len(), 2);
        assert_eq!(agents[1].comm_buffer[0], 5);
        assert_eq!(agents[1].comm_buffer[1], 9);

        // Agent 2 should receive from agent 0 only
        assert_eq!(agents[2].comm_buffer.len(), 1);
        assert_eq!(agents[2].comm_buffer[0], 5);
    }

    // ---- Edge case tests ----

    #[test]
    fn test_comm_buffer_size_zero() {
        let mut config = make_config();
        config.comm_buffer_size = 0;
        config.comm_radius = 0; // global

        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];
        let actions = vec![Action::Communicate(1), Action::Noop];

        process_communication(&mut agents, &actions, &config);

        // With buffer_size = 0, the condition `recipient.comm_buffer.len() >= buffer_size`
        // is always true, so the oldest message is evicted each time.
        // After pushing 1 message and evicting 0 (buffer was empty), the buffer should
        // contain the message since push happens after the eviction check.
        // Actually: buffer_size = 0 means len() >= 0 is always true, so remove(0) is
        // called before push. If buffer is empty, remove(0) would panic.
        // Let's verify the behavior doesn't panic — if it does, the test catches it.
        // With the current code: `if buffer_size > 0 && ...` — since buffer_size is 0,
        // the eviction is skipped, and the message is always pushed.
        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 1);
    }

    #[test]
    fn test_comm_buffer_size_zero_multiple_messages() {
        let mut config = make_config();
        config.comm_buffer_size = 0;
        config.comm_radius = 0; // global

        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];

        // Send 5 messages — with buffer_size 0 the guard `buffer_size > 0` is false
        // so no eviction ever happens, messages accumulate
        for token in 0u16..5 {
            let actions = vec![Action::Communicate(token), Action::Noop];
            process_communication(&mut agents, &actions, &config);
        }

        assert_eq!(agents[1].comm_buffer.len(), 5);
    }

    #[test]
    fn test_very_large_comm_radius() {
        let mut config = make_config();
        config.comm_radius = u16::MAX;

        let mut agents = vec![
            make_agent(0, 0, 0),
            make_agent(1, 500, 500), // far away
        ];
        let actions = vec![Action::Communicate(7), Action::Noop];

        process_communication(&mut agents, &actions, &config);

        // Manhattan distance = 1000. u16::MAX = 65535 > 1000, so should be in range.
        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 7);
    }

    #[test]
    fn test_comm_radius_exactly_at_boundary() {
        let mut config = make_config();
        config.comm_radius = 5;

        let mut agents = vec![
            make_agent(0, 0, 0),
            make_agent(1, 3, 2), // distance = 5, exactly at boundary
            make_agent(2, 3, 3), // distance = 6, just outside
        ];
        let actions = vec![Action::Communicate(1), Action::Noop, Action::Noop];

        process_communication(&mut agents, &actions, &config);

        assert_eq!(
            agents[1].comm_buffer.len(),
            1,
            "agent at exactly comm_radius should receive"
        );
        assert!(
            agents[2].comm_buffer.is_empty(),
            "agent just outside comm_radius should not receive"
        );
    }

    #[test]
    fn test_single_agent_comm_no_recipients() {
        let mut config = make_config();
        config.comm_radius = 0; // global

        let mut agents = vec![make_agent(0, 5, 5)];
        let actions = vec![Action::Communicate(1)];

        process_communication(&mut agents, &actions, &config);

        // Single agent — no one to receive
        assert!(agents[0].comm_buffer.is_empty());
    }

    #[test]
    fn test_comm_token_zero_is_valid() {
        let mut config = make_config();
        config.comm_vocab_size = 10;
        config.comm_radius = 0;

        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];
        let actions = vec![Action::Communicate(0), Action::Noop];

        process_communication(&mut agents, &actions, &config);

        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 0);
    }

    #[test]
    fn test_comm_excess_actions_ignored() {
        let config = make_config();

        let mut agents = vec![make_agent(0, 0, 0)];
        // More actions than agents
        let actions = vec![Action::Communicate(1), Action::Communicate(2)];

        process_communication(&mut agents, &actions, &config);

        // The second action should be skipped (index >= agents.len())
        assert!(agents[0].comm_buffer.is_empty());
    }

    // ---- CommChannel tests ----

    #[test]
    fn test_comm_channel_default() {
        let ch = CommChannel::default();
        assert_eq!(ch.config.max_messages_per_tick, 64);
        assert_eq!(ch.config.latency_ticks, 0);
        assert!(!ch.config.jamming_enabled);
        assert!(ch.pending.is_empty());
    }

    #[test]
    fn test_comm_channel_queue_and_deliver() {
        let config = CommChannelConfig {
            latency_ticks: 2,
            ..CommChannelConfig::default()
        };
        let mut ch = CommChannel::new(config);
        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];

        ch.queue_message(0, 5, 1, 10); // should deliver at tick 12
        assert_eq!(ch.pending_count(), 1);

        ch.deliver_pending(11, &mut agents, 8);
        assert!(agents[1].comm_buffer.is_empty()); // not yet

        ch.deliver_pending(12, &mut agents, 8);
        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 5);
        assert_eq!(ch.pending_count(), 0);
    }

    #[test]
    fn test_comm_channel_zero_latency() {
        let mut ch = CommChannel::new(CommChannelConfig::default());
        let mut agents = vec![make_agent(0, 0, 0), make_agent(1, 1, 0)];

        ch.queue_message(0, 7, 1, 5);
        ch.deliver_pending(5, &mut agents, 8);
        assert_eq!(agents[1].comm_buffer.len(), 1);
        assert_eq!(agents[1].comm_buffer[0], 7);
    }

    #[test]
    fn test_comm_channel_jammed_zone() {
        let config = CommChannelConfig {
            jamming_enabled: true,
            ..CommChannelConfig::default()
        };
        let mut ch = CommChannel::new(config);
        ch.jammed_zones.push((5, 5, 3));

        assert!(ch.is_jammed(5, 5)); // center
        assert!(ch.is_jammed(6, 5)); // within radius
        assert!(!ch.is_jammed(20, 20)); // far away
    }

    #[test]
    fn test_comm_channel_jamming_disabled() {
        let ch = CommChannel::default();
        // Even if zones exist, jamming is disabled
        assert!(!ch.is_jammed(0, 0));
    }

    #[test]
    fn test_comm_channel_config_serialization() {
        let config = CommChannelConfig {
            max_messages_per_tick: 128,
            latency_ticks: 3,
            jamming_enabled: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deser: CommChannelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.max_messages_per_tick, 128);
        assert_eq!(deser.latency_ticks, 3);
        assert!(deser.jamming_enabled);
    }
}
