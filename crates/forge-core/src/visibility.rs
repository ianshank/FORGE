//! Visibility system: fog-of-war and line-of-sight.
//!
//! Updates tile visibility states based on agent positions and vision radii,
//! using Bresenham-based ray casting for line-of-sight checks.

use forge_types::entity::Agent;
use forge_types::grid::{Grid, VisibilityState};
use tracing::trace;

/// Updates visibility state for all tiles based on agent positions and vision radii.
///
/// Process:
/// 1. Mark all currently Visible tiles as Explored (they were seen before)
/// 2. For each alive agent, cast rays to all tiles within vision_radius
/// 3. If the ray reaches the tile without hitting vision-blocking terrain, mark it Visible
///
/// Vision is blocked by tiles where `terrain.blocks_vision()` returns true.
pub fn update_visibility(agents: &[Agent], grid: &mut Grid) {
    // Step 1: Demote all Visible tiles to Explored
    for tile in grid.tiles.iter_mut() {
        if tile.visibility == VisibilityState::Visible {
            tile.visibility = VisibilityState::Explored;
        }
    }

    // Step 2: For each alive agent, reveal tiles within vision radius with line-of-sight
    for agent in agents {
        if !agent.alive {
            continue;
        }

        let ax = agent.position.x as i32;
        let ay = agent.position.y as i32;
        let vr = agent.vision_radius as i32;

        trace!(
            agent_id = agent.id,
            x = ax,
            y = ay,
            vision_radius = vr,
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

        update_visibility(&[agent], &mut grid);

        let tile = grid.get(5, 5).unwrap();
        assert_eq!(tile.visibility, VisibilityState::Visible);
    }

    #[test]
    fn test_vision_radius_limit() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent], &mut grid);

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

        update_visibility(&[agent], &mut grid);

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
        update_visibility(&[agent.clone()], &mut grid);
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Visible);

        // Move agent away
        agent.position = Position::new(12, 12);

        // Second pass: old tiles should become Explored, not Hidden
        update_visibility(&[agent], &mut grid);

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

        update_visibility(&[a1, a2], &mut grid);

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

        update_visibility(&[agent], &mut grid);

        // Dead agent should not reveal any tiles
        assert_eq!(grid.get(5, 5).unwrap().visibility, VisibilityState::Hidden);
    }

    #[test]
    fn test_visibility_mask_shape() {
        let mut grid = make_grid(32, 32);
        let mut agent = make_agent(0, 15, 15);
        agent.vision_radius = 3;

        update_visibility(&[agent.clone()], &mut grid);

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
}
