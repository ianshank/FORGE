//! Agent observation generation and buffer filling.

use forge_civ::grid_topology::{GridTopology, GridTopologyKind};
use forge_types::constants::{OBS_EMPTY_SLOT_ITEM, OBS_NO_OBJECT, OBS_NO_RESOURCE};
use forge_types::entity::Agent;
use forge_types::observation::{Observation, TileObservation};
use tracing::instrument;

use super::WorldState;

impl WorldState {
    /// Generates an observation for a single agent.
    ///
    /// Convenience wrapper around [`Self::fill_observation`] that allocates
    /// a fresh [`Observation`]. Use `fill_observation` directly to avoid
    /// the allocation when filling a reusable buffer.
    pub fn generate_observation(&self, agent: &Agent) -> Observation {
        let mut obs = Observation::default();
        self.fill_observation(agent, &mut obs);
        obs
    }

    /// Fills an existing [`Observation`] in place from the current state.
    ///
    /// The output's inner `Vec`s (`grid_view`, `inventory.slots`, `messages`,
    /// `task_progress`, `crop_scan_results`, `soil_readings`) are `clear`ed
    /// and re-extended rather than reallocated. After a single warm call,
    /// repeated invocations on the same `out` buffer perform no heap
    /// allocations — this is what keeps `step_into` zero-alloc.
    #[instrument(skip_all)]
    pub fn fill_observation(&self, agent: &Agent, out: &mut Observation) {
        let vr = agent.vision_radius as i32;
        let view_side = (2 * vr + 1) as u16;
        let cells = (view_side as usize) * (view_side as usize);
        let is_hex = matches!(self.topology, GridTopologyKind::Hex(_));

        out.grid_view.clear();
        out.grid_view.reserve(cells);
        for dy in -vr..=vr {
            for dx in -vr..=vr {
                let wx = agent.position.x as i32 + dx;
                let wy = agent.position.y as i32 + dy;

                let in_bounds = wx >= 0
                    && wx < self.grid.width as i32
                    && wy >= 0
                    && wy < self.grid.height as i32;

                // On hex grids, positions inside the bounding box but outside the
                // axial-radius view are treated as out of bounds (shown as walls).
                let in_disk = if in_bounds && is_hex {
                    self.topology.distance(
                        agent.position,
                        forge_types::grid::Position::new(wx as u16, wy as u16),
                    ) <= u32::from(agent.vision_radius)
                } else {
                    in_bounds
                };

                if in_disk {
                    let tile = self
                        .grid
                        .get(wx as u16, wy as u16)
                        .expect("invariant: in_disk implies in_bounds");
                    out.grid_view.push(TileObservation {
                        terrain: tile.terrain as u8,
                        elevation: tile.elevation,
                        has_agent: tile.agent_id.is_some(),
                        has_object: tile.object_id.is_some(),
                        has_resource: tile.resource_id.is_some(),
                        object_type: tile.object_id.map_or(OBS_NO_OBJECT, |_| 0),
                        resource_type: tile.resource_id.map_or(OBS_NO_RESOURCE, |_| 0),
                    });
                } else {
                    // Out of bounds — show as wall
                    out.grid_view.push(TileObservation {
                        terrain: forge_types::TerrainType::Wall as u8,
                        elevation: 0,
                        has_agent: false,
                        has_object: false,
                        has_resource: false,
                        object_type: OBS_NO_OBJECT,
                        resource_type: OBS_NO_RESOURCE,
                    });
                }
            }
        }

        out.view_width = view_side;
        out.view_height = view_side;

        out.inventory.slots.clear();
        out.inventory
            .slots
            .extend(agent.inventory.slots.iter().map(|slot| match slot {
                Some(stack) => (stack.item_type as u8, stack.count),
                None => (OBS_EMPTY_SLOT_ITEM, 0),
            }));

        let max_health = self.config.agents.max_health as f32;
        let max_stamina = self.config.agents.max_stamina as f32;

        out.health = if max_health > 0.0 {
            agent.health as f32 / max_health
        } else {
            0.0
        };
        out.stamina = if max_stamina > 0.0 {
            agent.stamina as f32 / max_stamina
        } else {
            0.0
        };
        out.position = (agent.position.x, agent.position.y);

        out.messages.clear();
        out.messages.extend_from_slice(&agent.comm_buffer);

        out.day_phase = self.day_phase;

        out.task_progress.clear();
        out.task_progress.extend(
            self.tasks
                .iter()
                .map(|t| t.progress.first().copied().unwrap_or(0.0)),
        );

        out.altitude = agent.altitude;
        out.battery = if self.config.drone.enabled
            && agent.morphology == forge_types::entity::AgentMorphology::Aerial
        {
            let max = self.config.drone.max_battery as f32;
            if max > 0.0 {
                (agent.battery as f32 / max).clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        out.morphology = agent.morphology as u8;
        out.heading = agent.heading as u8;
        out.crop_scan_results.clear();
        out.soil_readings.clear();
        out.disease_detections = 0;
        out.report_ready = false;
    }
}
