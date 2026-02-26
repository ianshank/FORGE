//! Communication system for multi-agent messaging.
//!
//! Agents can broadcast discrete communication tokens to nearby agents.
//! Tokens are drawn from a fixed vocabulary and delivered to recipients'
//! communication buffers based on proximity.

use forge_types::config::AgentConfig;
use forge_types::entity::Agent;
use forge_types::Action;
use tracing::trace;

/// Processes Communicate actions: broadcasts tokens to agents within comm_radius.
///
/// When an agent performs `Communicate(token)`:
/// - Validates token is within vocabulary size
/// - Checks the sending agent has the `can_communicate` capability
/// - Finds all other alive agents within comm_radius (manhattan distance)
/// - Adds the token to each recipient's comm_buffer (evicting oldest if full)
///
/// If comm_radius is 0, the message is broadcast to ALL alive agents.
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
}
