//! Visibility system: fog-of-war and line-of-sight.
//!
//! Updates tile visibility states based on agent positions and vision radii.
//! Uses the grid topology abstraction for neighbor/LoS queries so that both
//! square and hex grids are supported.

use forge_civ::grid_topology::{GridTopology, GridTopologyKind};
use forge_types::entity::Agent;
use forge_types::grid::{Grid, Position, VisibilityState};
use tracing::{instrument, trace};

use crate::day_night;

/// Updates visibility state for all tiles based on agent positions and vision radii.
///
/// Process:
/// 1. Mark all currently Visible tiles as Explored (they were seen before)
/// 2. For each alive agent, compute effective vision radius (base * day/night modifier)
/// 3. Iterate the bounded vision area for the active topology and use
///    topology.line_of_sight() to check for terrain blockers
/// 4. If the ray reaches the tile without hitting vision-blocking terrain, mark it Visible
///
/// Vision is blocked by tiles where `terrain.blocks_vision()` returns true.
/// The `day_phase` parameter controls vision range: day = full, dawn/dusk = 75%, night = 50%.
#[instrument(skip_all)]
pub fn update_visibility(
    agents: &[Agent],
    grid: &mut Grid,
    day_phase: u8,
    topology: &GridTopologyKind,
) {
    // Step 1: Demote all Visible tiles to Explored
    for tile in grid.tiles.iter_mut() {
        if tile.visibility == VisibilityState::Visible {
            tile.visibility = VisibilityState::Explored;
        }
    }

    // Step 2: For each alive agent, reveal tiles within effective vision radius with line-of-sight
    let modifier = day_night::vision_modifier(day_phase);

    for agent in agents {
        if !agent.alive {
            continue;
        }

        // Apply day/night vision modifier to base vision radius
        let effective_radius = (agent.vision_radius as f32 * modifier).round() as u16;

        trace!(
            agent_id = agent.id,
            x = agent.position.x,
            y = agent.position.y,
            base_radius = agent.vision_radius,
            effective_radius = effective_radius,
            day_phase = day_phase,
            "updating visibility"
        );

        let ax = agent.position.x as i32;
        let ay = agent.position.y as i32;
        let radius = effective_radius as i32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let wx = ax + dx;
                let wy = ay + dy;

                if wx < 0 || wy < 0 || wx >= grid.width as i32 || wy >= grid.height as i32 {
                    continue;
                }

                let candidate = Position::new(wx as u16, wy as u16);
                let in_range = match topology {
                    GridTopologyKind::Square(_) => dx * dx + dy * dy <= radius * radius,
                    GridTopologyKind::Hex(_) => {
                        topology.distance(agent.position, candidate) <= u32::from(effective_radius)
                    }
                };

                if in_range && has_line_of_sight_topo(grid, agent.position, candidate, topology) {
                    if let Some(tile) = grid.get_mut(candidate.x, candidate.y) {
                        tile.visibility = VisibilityState::Visible;
                    }
                }
            }
        }
    }
}

/// Public accessor for line-of-sight queries from other modules.
///
/// Returns `true` if there is a clear line of sight from `(x0, y0)` to
/// `(x1, y1)` on the given grid using the topology's line algorithm.
#[instrument(skip_all)]
pub fn check_line_of_sight(
    grid: &Grid,
    from: Position,
    to: Position,
    topology: &GridTopologyKind,
) -> bool {
    has_line_of_sight_topo(grid, from, to, topology)
}

/// Checks line of sight between two positions using the topology's line algorithm.
///
/// Only intermediate tiles are checked for vision blockers — the start and end
/// tiles themselves are not considered blocking. This means an agent can always
/// see its own tile, and can see a wall tile (but not through it).
fn has_line_of_sight_topo(
    grid: &Grid,
    from: Position,
    to: Position,
    topology: &GridTopologyKind,
) -> bool {
    // Same tile — always visible
    if from == to {
        return true;
    }

    let cells = topology.line_of_sight(from, to);

    // Skip first (from) and last (to) — only check intermediate cells
    for cell in cells.iter().skip(1) {
        if *cell == to {
            // Reached the target — line of sight is clear
            return true;
        }
        if let Some(tile) = grid.get(cell.x, cell.y) {
            if tile.terrain.blocks_vision() {
                return false;
            }
        }
    }

    true
}

/// Generates an ego-centric visibility mask for an agent.
///
/// Returns a flat `Vec<bool>` of `(2*radius+1)^2` elements, row-major.
/// Each element is `true` if the corresponding tile is currently Visible,
/// `false` otherwise (Hidden, Explored, or out of bounds).
#[instrument(skip_all)]
pub fn visibility_mask(agent: &Agent, grid: &Grid) -> Vec<bool> {
    let vr = agent.vision_radius as i32;
    let side = (2 * vr + 1) as usize;
    let mut mask = vec![false; side * side];

    let ax = agent.position.x as i32;
    let ay = agent.position.y as i32;

    for dy in -vr..=vr {
        for dx in -vr..=vr {
            let wx = ax + dx;
            let wy = ay + dy;

            let idx = ((dy + vr) as usize) * side + (dx + vr) as usize;

            if wx >= 0 && wy >= 0 && wx < grid.width as i32 && wy < grid.height as i32 {
                if let Some(tile) = grid.get(wx as u16, wy as u16) {
                    mask[idx] = tile.visibility == VisibilityState::Visible;
                }
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_civ::grid_topology::GridTopologyKind;
    use forge_civ::{HexTopology, SquareTopology};
    use forge_types::config::AgentConfig;
    use forge_types::constants::{DAY_PHASE_DAWN, DAY_PHASE_DAY, DAY_PHASE_NIGHT};
    use forge_types::entity::Agent;
    use forge_types::grid::{Grid, Position, TerrainType, VisibilityState};

    fn topo() -> GridTopologyKind {
        GridTopologyKind::Square(SquareTopology)
    }

    fn hex_topo() -> GridTopologyKind {
        GridTopologyKind::Hex(HexTopology)
    }

    fn make_agent(id: u32, x: u16, y: u16) -> Agent {
        let config = AgentConfig::default();
        Agent::new(id, Position::new(x, y), &config)
    }

    fn make_grid(width: u16, height: u16) -> Grid {
        Grid::new(width, height)
    }

    #[test]
    fn test_agent_sees_own_tile() {
        let mut grid = make_grid(16, 16);
        let agent = make_agent(0, 5, 5);

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        let tile = grid.get(5, 5).unwrap();
        assert_eq!(tile.visibility, VisibilityState::Visible);
    }

    #[test]
    fn test_vision_radius_limit() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // Tile within radius should be visible
        assert_eq!(
            grid.get(16, 15).unwrap().visibility,
            VisibilityState::Visible
        );

        // Tile well beyond radius should be hidden
        assert_eq!(
            grid.get(25, 25).unwrap().visibility,
            VisibilityState::Hidden
        );
    }

    #[test]
    fn test_hex_visibility_uses_hex_distance_not_square_bbox() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 1;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &hex_topo());

        assert_eq!(grid.get(6, 5).unwrap().visibility, VisibilityState::Visible);
        assert_eq!(
            grid.get(4, 4).unwrap().visibility,
            VisibilityState::Hidden,
            "square-corner cells should stay hidden on hex grids at radius 1"
        );
    }

    #[test]
    fn test_wall_blocks_vision() {
        let mut grid = make_grid(16, 16);
        // Place a wall between agent and target
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 5;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // The wall tile itself should be visible (we can see walls)
        assert_eq!(grid.get(7, 5).unwrap().visibility, VisibilityState::Visible);

        // Tile behind the wall should remain hidden (blocked line of sight)
        assert_eq!(grid.get(9, 5).unwrap().visibility, VisibilityState::Hidden);
    }

    #[test]
    fn test_explored_tiles_remain() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 2;

        // First pass: agent at (5,5) reveals nearby tiles
        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Visible);

        // Move agent away
        agent.position = Position::new(12, 12);

        // Second pass: old tiles should become Explored, not Hidden
        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        assert_eq!(
            grid.get(5, 5).unwrap().visibility,
            VisibilityState::Explored
        );
        assert_eq!(
            grid.get(12, 12).unwrap().visibility,
            VisibilityState::Visible
        );
    }

    #[test]
    fn test_hidden_tiles_default() {
        let grid = make_grid(8, 8);
        for tile in &grid.tiles {
            assert_eq!(tile.visibility, VisibilityState::Hidden);
        }
    }

    #[test]
    fn test_multiple_agents_visibility() {
        let mut grid = make_grid(32, 32);
        let mut a1 = make_agent(0, 5, 5);
        a1.vision_radius = 2;
        let mut a2 = make_agent(1, 20, 20);
        a2.vision_radius = 2;

        update_visibility(&[a1, a2], &mut grid, DAY_PHASE_DAY, &topo());

        // Both agents' own tiles should be visible
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Visible);
        assert_eq!(
            grid.get(20, 20).unwrap().visibility,
            VisibilityState::Visible
        );

        // A tile far from both should remain hidden
        assert_eq!(
            grid.get(15, 15).unwrap().visibility,
            VisibilityState::Hidden
        );
    }

    #[test]
    fn test_dead_agent_no_vision() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.alive = false;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // Dead agent should not reveal any tiles
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Hidden);
    }

    #[test]
    fn test_visibility_mask_shape() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());

        let mask = visibility_mask(&agent, &grid);
        let side = 2 * 3 + 1;
        assert_eq!(mask.len(), side * side);

        // Center of mask (agent's tile) should be true
        let center = 3 * side + 3;
        assert!(mask[center]);
    }

    #[test]
    fn test_line_of_sight_clear() {
        let grid = make_grid(16, 16);
        // All ground tiles — should have clear line of sight
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(8, 7),
            &topo()
        ));
    }

    #[test]
    fn test_line_of_sight_blocked() {
        let mut grid = make_grid(16, 16);
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        // Line from (5,5) to (9,5) passes through the wall at (7,5)
        assert!(!check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(9, 5),
            &topo()
        ));
    }

    // ---- Edge case tests ----

    #[test]
    fn test_vision_radius_zero() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 0;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());

        // Agent should see only its own tile
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Visible);

        // Adjacent tiles should remain hidden
        assert_eq!(grid.get(5, 4).unwrap().visibility, VisibilityState::Hidden);
        assert_eq!(grid.get(5, 6).unwrap().visibility, VisibilityState::Hidden);
        assert_eq!(grid.get(4, 5).unwrap().visibility, VisibilityState::Hidden);
        assert_eq!(grid.get(6, 5).unwrap().visibility, VisibilityState::Hidden);

        // Visibility mask for radius 0 should be a single element
        let mask = visibility_mask(&agent, &grid);
        assert_eq!(mask.len(), 1);
        assert!(mask[0]);
    }

    #[test]
    fn test_agent_at_origin_visibility() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 0, 0);
        agent.vision_radius = 3;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());

        // Own tile should be visible
        assert_eq!(grid.get(0, 0).unwrap().visibility, VisibilityState::Visible);

        // Some tiles within radius that are in bounds should be visible
        assert_eq!(grid.get(1, 0).unwrap().visibility, VisibilityState::Visible);
        assert_eq!(grid.get(0, 1).unwrap().visibility, VisibilityState::Visible);
        assert_eq!(grid.get(2, 2).unwrap().visibility, VisibilityState::Visible);

        // Tiles far away should be hidden
        assert_eq!(
            grid.get(10, 10).unwrap().visibility,
            VisibilityState::Hidden
        );

        // Mask should still be correct shape
        let mask = visibility_mask(&agent, &grid);
        let side = (2 * 3 + 1) as usize;
        assert_eq!(mask.len(), side * side);
        // Center of mask (which represents agent pos) should be visible
        let center = 3 * side + 3;
        assert!(mask[center]);
    }

    #[test]
    fn test_diagonal_line_of_sight() {
        let grid = make_grid(16, 16);

        // Diagonal line of sight across open terrain should be clear
        assert!(check_line_of_sight(
            &grid,
            Position::new(3, 3),
            Position::new(7, 7),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(7, 7),
            Position::new(3, 3),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(3, 7),
            Position::new(7, 3),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(7, 3),
            Position::new(3, 7),
            &topo()
        ));
    }

    #[test]
    fn test_diagonal_line_of_sight_blocked() {
        let mut grid = make_grid(16, 16);
        // Place a wall on the diagonal path from (3,3) to (7,7)
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Wall;

        assert!(!check_line_of_sight(
            &grid,
            Position::new(3, 3),
            Position::new(7, 7),
            &topo()
        ));
    }

    #[test]
    fn test_line_of_sight_same_tile() {
        let grid = make_grid(16, 16);
        // Same tile should always be visible
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(5, 5),
            &topo()
        ));
    }

    #[test]
    fn test_line_of_sight_adjacent() {
        let grid = make_grid(16, 16);
        // Adjacent tiles should have clear LOS on open terrain
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(5, 6),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(5, 4),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(6, 5),
            &topo()
        ));
        assert!(check_line_of_sight(
            &grid,
            Position::new(5, 5),
            Position::new(4, 5),
            &topo()
        ));
    }

    #[test]
    fn test_visibility_uses_euclidean_radius() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // (15 + 3, 15 + 3) => dx=3, dy=3 => dist^2 = 18 > 9 = radius^2
        // So corners of the bounding box should NOT be visible
        assert_eq!(
            grid.get(18, 18).unwrap().visibility,
            VisibilityState::Hidden,
            "corner of bounding box should be outside Euclidean radius"
        );

        // (15 + 3, 15) => dx=3, dy=0 => dist^2 = 9 <= 9 => visible
        assert_eq!(
            grid.get(18, 15).unwrap().visibility,
            VisibilityState::Visible,
            "tile at exact radius along axis should be visible"
        );
    }

    #[test]
    fn test_visibility_mask_at_corner() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 0, 0);
        agent.vision_radius = 2;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());

        let mask = visibility_mask(&agent, &grid);
        let side = (2 * 2 + 1) as usize; // 5
        assert_eq!(mask.len(), side * side);

        // Center (agent's own tile) should be true
        let center = 2 * side + 2;
        assert!(mask[center]);

        // Top-left corner of mask (offset -2,-2 from agent at 0,0 => world (-2,-2)) is OOB
        assert!(!mask[0], "out-of-bounds tile should be false in mask");
    }

    #[test]
    fn test_mountain_blocks_vision() {
        let mut grid = make_grid(16, 16);
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Mountain;

        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 5;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // Mountain itself should be visible
        assert_eq!(grid.get(7, 5).unwrap().visibility, VisibilityState::Visible);

        // Tile behind the mountain should be hidden
        assert_eq!(grid.get(9, 5).unwrap().visibility, VisibilityState::Hidden);
    }

    // ---- Day/night vision modifier tests ----

    #[test]
    fn test_night_reduces_vision_radius() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 6; // night modifier 0.5 → effective radius 3

        update_visibility(&[agent], &mut grid, DAY_PHASE_NIGHT, &topo());

        // Tile at distance 3 along axis should be visible (effective radius = 3)
        assert_eq!(
            grid.get(18, 15).unwrap().visibility,
            VisibilityState::Visible,
            "tile at effective night radius should be visible"
        );

        // Tile at distance 5 along axis should be hidden (beyond effective radius)
        assert_eq!(
            grid.get(20, 15).unwrap().visibility,
            VisibilityState::Hidden,
            "tile beyond effective night radius should be hidden"
        );
    }

    #[test]
    fn test_dawn_reduces_vision_radius() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 4; // dawn modifier 0.75 → effective radius 3

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAWN, &topo());

        // Tile at distance 3 along axis should be visible
        assert_eq!(
            grid.get(18, 15).unwrap().visibility,
            VisibilityState::Visible,
            "tile at effective dawn radius should be visible"
        );

        // Tile at distance 4 along axis should be hidden
        assert_eq!(
            grid.get(19, 15).unwrap().visibility,
            VisibilityState::Hidden,
            "tile beyond effective dawn radius should be hidden"
        );
    }

    #[test]
    fn test_day_full_vision_radius() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 4;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY, &topo());

        // Tile at full radius should be visible during day
        assert_eq!(
            grid.get(19, 15).unwrap().visibility,
            VisibilityState::Visible,
            "tile at full day radius should be visible"
        );
    }

    #[test]
    fn test_night_vision_radius_zero_still_sees_own_tile() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 1; // night modifier 0.5 → rounds to 1, still sees adjacent

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_NIGHT, &topo());

        // Agent should always see own tile regardless of phase
        assert_eq!(
            grid.get(5, 5).unwrap().visibility,
            VisibilityState::Visible,
            "agent should always see own tile at night"
        );
    }

    #[test]
    fn test_fog_of_war_persists_with_day_night_transition() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 6;

        // Day pass: full radius 6, tiles up to distance 6 are visible
        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY, &topo());
        assert_eq!(
            grid.get(21, 15).unwrap().visibility,
            VisibilityState::Visible
        );

        // Night pass: effective radius 3, far tiles become Explored
        update_visibility(&[agent], &mut grid, DAY_PHASE_NIGHT, &topo());
        assert_eq!(
            grid.get(21, 15).unwrap().visibility,
            VisibilityState::Explored,
            "tiles beyond night radius should become Explored, not Hidden"
        );
    }

    // ---- Proptest: visibility invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// An agent always sees its own tile.
            #[test]
            fn agent_sees_own_tile(
                x in 1u16..30,
                y in 1u16..30,
                phase in 0u8..4,
            ) {
                let mut grid = make_grid(32, 32);
                let agent = make_agent(0, x, y);
                update_visibility(&[agent], &mut grid, phase, &topo());
                prop_assert_eq!(
                    grid.get(x, y).unwrap().visibility,
                    VisibilityState::Visible,
                );
            }

            /// Visibility mask has correct shape for given vision radius.
            #[test]
            fn visibility_mask_shape(
                vr in 0u8..6,
                x in 6u16..26,
                y in 6u16..26,
            ) {
                let grid = make_grid(32, 32);
                let mut agent = make_agent(0, x, y);
                agent.vision_radius = vr;
                let mask = visibility_mask(&agent, &grid);
                let side = 2 * vr as usize + 1;
                prop_assert_eq!(mask.len(), side * side);
            }
        }
    }
}
