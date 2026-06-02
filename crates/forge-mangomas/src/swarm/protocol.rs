//! Swarm protocol trait for multi-drone coordination.
//!
//! Defines the interface that a swarm coordination strategy must implement.
//! This is a Phase 6 stub — implementations will be added when
//! multi-agent coordination is developed.

use forge_core::WorldState;
use forge_types::action::Action;
use forge_types::observation::Observation;
use tracing::instrument;

/// Trait for multi-agent swarm coordination protocols.
///
/// Implementations receive observations from all swarm members and
/// produce coordinated actions. The protocol manages inter-agent
/// communication and cooperative planning.
pub trait SwarmProtocol: Send + Sync {
    /// Returns the name of the coordination strategy.
    fn name(&self) -> &str;

    /// Produces coordinated actions for all agents given their observations.
    ///
    /// # Arguments
    /// * `observations` - Per-agent observations from the environment.
    /// * `comm_tokens` - Communication tokens received from other agents.
    ///
    /// # Returns
    /// Per-agent actions to execute.
    fn coordinate(&self, observations: &[Observation], comm_tokens: &[Vec<u16>]) -> Vec<Action>;

    /// Produces coordinated actions given access to the full simulation state.
    ///
    /// This is an **additive, non-breaking** extension: the default delegates to
    /// the world-less [`Self::coordinate`], so existing implementations need no
    /// changes. Stateful protocols (e.g. cooperative MCTS, which needs to
    /// simulate) override this to use the [`WorldState`] directly.
    ///
    /// # Arguments
    /// * `world` - The shared simulation state for all agents.
    /// * `observations` - Per-agent observations from the environment.
    /// * `comm_tokens` - Communication tokens received from other agents.
    fn coordinate_stateful(
        &self,
        world: &WorldState,
        observations: &[Observation],
        comm_tokens: &[Vec<u16>],
    ) -> Vec<Action> {
        let _ = world;
        self.coordinate(observations, comm_tokens)
    }

    /// Returns the number of agents in the swarm.
    fn swarm_size(&self) -> usize;
}

/// Stub implementation: each agent acts independently (no coordination).
pub struct IndependentProtocol {
    swarm_size: usize,
}

impl IndependentProtocol {
    /// Create a new independent (no-coordination) protocol.
    pub fn new(swarm_size: usize) -> Self {
        Self { swarm_size }
    }
}

impl SwarmProtocol for IndependentProtocol {
    fn name(&self) -> &str {
        "independent"
    }

    #[instrument(skip(self, observations, _comm_tokens))]
    fn coordinate(&self, observations: &[Observation], _comm_tokens: &[Vec<u16>]) -> Vec<Action> {
        // Stub: all agents do nothing
        vec![Action::Noop; observations.len()]
    }

    fn swarm_size(&self) -> usize {
        self.swarm_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_types::observation::{InventoryObservation, TileObservation};

    fn make_obs() -> Observation {
        Observation {
            grid_view: vec![TileObservation::default()],
            view_width: 1,
            view_height: 1,
            inventory: InventoryObservation { slots: vec![] },
            health: 1.0,
            stamina: 1.0,
            position: (0, 0),
            messages: vec![],
            day_phase: 0,
            task_progress: vec![],
            altitude: 0,
            battery: 1.0,
            morphology: 0,
            heading: 0,
            crop_scan_results: vec![],
            soil_readings: vec![],
            disease_detections: 0,
            report_ready: false,
        }
    }

    #[test]
    fn test_independent_protocol() {
        let protocol = IndependentProtocol::new(3);
        assert_eq!(protocol.name(), "independent");
        assert_eq!(protocol.swarm_size(), 3);

        let obs = vec![make_obs(); 3];
        let comm = vec![vec![]; 3];
        let actions = protocol.coordinate(&obs, &comm);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0], Action::Noop);
    }
}
