//! Visibility system: fog-of-war and line-of-sight.
//!
//! Updates tile visibility states based on agent positions and vision radii,
//! using Bresenham-based ray casting for line-of-sight checks.

use forge_types::entity::Agent;
use forge_types::grid::{Grid, VisibilityState};
use tracing::{instrument, trace};

use crate::day_night;

/// Updates visibility state for all tiles based on agent positions and vision radii.
///
/// Process:
/// 1. Mark all currently Visible tiles as Explored (they were seen before)
/// 2. For each alive agent, compute effective vision radius (base * day/night modifier)
/// 3. Cast rays to all tiles within the effective radius
/// 4. If the ray reaches the tile without hitting vision-blocking terrain, mark it Visible
///
/// Vision is blocked by tiles where `terrain.blocks_vision()` returns true.
/// The `day_phase` parameter controls vision range: day = full, dawn/dusk = 75%, night = 50%.
#[instrument(skip_all)]
pub fn update_visibility(agents: &[Agent], grid: &mut Grid, day_phase: u8) {
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

        let ax = agent.position.x as i32;
        let ay = agent.position.y as i32;
        // Apply day/night vision modifier to base vision radius
        let effective_radius = (agent.vision_radius as f32 * modifier).round() as i32;
        let vr = effective_radius.max(0);

        trace!(
            agent_id = agent.id,
            x = ax,
            y = ay,
            base_radius = agent.vision_radius,
            effective_radius = vr,
            day_phase = day_phase,
            "updating visibility"
        );

        for dy in -vr..=vr {
            for dx in -vr..=vr {
                let tx = ax + dx;
                let ty = ay + dy;

                // Check within vision radius (Chebyshev / square area — we use
                // the radius as a bounding box, but also check Euclidean distance
                // squared to get a circular field of view)
                if dx * dx + dy * dy > vr * vr {
                    continue;
                }

                // Check bounds
                if tx < 0 || ty < 0 || tx >= grid.width as i32 || ty >= grid.height as i32 {
                    continue;
                }

                // Check line of sight from agent to target tile
                if has_line_of_sight(grid, ax, ay, tx, ty) {
                    let tile = grid.get_mut(tx as u16, ty as u16).unwrap();
                    tile.visibility = VisibilityState::Visible;
                }
            }
        }
    }
}

/// Checks line of sight between two positions using Bresenham's line algorithm.
///
/// Only intermediate tiles are checked for vision blockers — the start and end
/// tiles themselves are not considered blocking. This means an agent can always
/// see its own tile, and can see a wall tile (but not through it).
fn has_line_of_sight(grid: &Grid, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    // Same tile — always visible
    if x0 == x1 && y0 == y1 {
        return true;
    }

    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut cx = x0;
    let mut cy = y0;

    loop {
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            cx += sx;
        }
        if e2 < dx {
            err += dx;
            cy += sy;
        }

        // If we've reached the target, line of sight is clear
        if cx == x1 && cy == y1 {
            return true;
        }

        // Check if intermediate tile blocks vision
        if cx >= 0 && cy >= 0 && cx < grid.width as i32 && cy < grid.height as i32 {
            if let Some(tile) = grid.get(cx as u16, cy as u16) {
                if tile.terrain.blocks_vision() {
                    return false;
                }
            }
        } else {
            // Out of bounds — treat as blocked
            return false;
        }
    }
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
    use forge_types::config::AgentConfig;
    use forge_types::constants::{DAY_PHASE_DAWN, DAY_PHASE_DAY, DAY_PHASE_NIGHT};
    use forge_types::entity::Agent;
    use forge_types::grid::{Grid, Position, TerrainType, VisibilityState};

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

        let tile = grid.get(5, 5).unwrap();
        assert_eq!(tile.visibility, VisibilityState::Visible);
    }

    #[test]
    fn test_vision_radius_limit() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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
    fn test_wall_blocks_vision() {
        let mut grid = make_grid(16, 16);
        // Place a wall between agent and target
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 5;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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
        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Visible);

        // Move agent away
        agent.position = Position::new(12, 12);

        // Second pass: old tiles should become Explored, not Hidden
        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[a1, a2], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

        // Dead agent should not reveal any tiles
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Hidden);
    }

    #[test]
    fn test_visibility_mask_shape() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);

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
        assert!(has_line_of_sight(&grid, 5, 5, 8, 7));
    }

    #[test]
    fn test_line_of_sight_blocked() {
        let mut grid = make_grid(16, 16);
        grid.get_mut(7, 5).unwrap().terrain = TerrainType::Wall;

        // Line from (5,5) to (9,5) passes through the wall at (7,5)
        assert!(!has_line_of_sight(&grid, 5, 5, 9, 5));
    }

    // ---- Edge case tests ----

    #[test]
    fn test_vision_radius_zero() {
        let mut grid = make_grid(16, 16);
        let mut agent = make_agent(0, 5, 5);
        agent.vision_radius = 0;

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);

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
        assert!(has_line_of_sight(&grid, 3, 3, 7, 7));
        assert!(has_line_of_sight(&grid, 7, 7, 3, 3));
        assert!(has_line_of_sight(&grid, 3, 7, 7, 3));
        assert!(has_line_of_sight(&grid, 7, 3, 3, 7));
    }

    #[test]
    fn test_diagonal_line_of_sight_blocked() {
        let mut grid = make_grid(16, 16);
        // Place a wall on the diagonal path from (3,3) to (7,7)
        grid.get_mut(5, 5).unwrap().terrain = TerrainType::Wall;

        assert!(!has_line_of_sight(&grid, 3, 3, 7, 7));
    }

    #[test]
    fn test_line_of_sight_same_tile() {
        let grid = make_grid(16, 16);
        // Same tile should always be visible
        assert!(has_line_of_sight(&grid, 5, 5, 5, 5));
    }

    #[test]
    fn test_line_of_sight_adjacent() {
        let grid = make_grid(16, 16);
        // Adjacent tiles should have clear LOS on open terrain
        assert!(has_line_of_sight(&grid, 5, 5, 5, 6));
        assert!(has_line_of_sight(&grid, 5, 5, 5, 4));
        assert!(has_line_of_sight(&grid, 5, 5, 6, 5));
        assert!(has_line_of_sight(&grid, 5, 5, 4, 5));
    }

    #[test]
    fn test_visibility_uses_euclidean_radius() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_NIGHT);

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAWN);

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

        update_visibility(&[agent], &mut grid, DAY_PHASE_DAY);

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

        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_NIGHT);

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
        update_visibility(&[agent.clone()], &mut grid, DAY_PHASE_DAY);
        assert_eq!(
            grid.get(21, 15).unwrap().visibility,
            VisibilityState::Visible
        );

        // Night pass: effective radius 3, far tiles become Explored
        update_visibility(&[agent], &mut grid, DAY_PHASE_NIGHT);
        assert_eq!(
            grid.get(21, 15).unwrap().visibility,
            VisibilityState::Explored,
            "tiles beyond night radius should become Explored, not Hidden"
        );
    }
}
