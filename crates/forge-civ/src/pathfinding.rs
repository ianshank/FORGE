//! A* pathfinding over any [`GridTopology`].
//!
//! Works on both square and hex grids. Uses terrain movement costs for
//! weighted edges. Returns `None` for unreachable destinations.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use forge_types::grid::{Grid, Position};
use smallvec::SmallVec;

use crate::grid_topology::GridTopology;

/// A node in the A* open set.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AStarNode {
    pos: Position,
    /// f = g + h (estimated total cost).
    f_cost: u32,
    /// g = cost from start to this node.
    g_cost: u32,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering so lower cost has higher priority
        other
            .f_cost
            .cmp(&self.f_cost)
            .then_with(|| other.g_cost.cmp(&self.g_cost))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Finds the shortest path from `start` to `goal` using A*.
///
/// Returns the path as a sequence of positions (including start and goal),
/// or `None` if no path exists.
///
/// Movement costs come from `TerrainType::movement_cost()` on each tile.
/// Non-walkable terrain is treated as impassable.
pub fn astar<T: GridTopology>(
    topology: &T,
    grid: &Grid,
    start: Position,
    goal: Position,
) -> Option<Vec<Position>> {
    let w = grid.width;
    let h = grid.height;
    let size = w as usize * h as usize;

    if start == goal {
        return Some(vec![start]);
    }

    // Check goal is walkable
    let tile = grid.get_pos(&goal)?;
    if !tile.terrain.is_walkable() {
        return None;
    }

    // g_cost[idx] = best known cost from start to that cell, u32::MAX = unvisited
    let mut g_costs = vec![u32::MAX; size];
    // came_from[idx] = previous position in optimal path
    let mut came_from: Vec<Option<Position>> = vec![None; size];

    let start_idx = start.y as usize * w as usize + start.x as usize;
    g_costs[start_idx] = 0;

    let mut open = BinaryHeap::new();
    open.push(AStarNode {
        pos: start,
        f_cost: topology.distance(start, goal),
        g_cost: 0,
    });

    while let Some(current) = open.pop() {
        if current.pos == goal {
            // Reconstruct path
            return Some(reconstruct_path(&came_from, w, start, goal));
        }

        let curr_idx = current.pos.y as usize * w as usize + current.pos.x as usize;
        if current.g_cost > g_costs[curr_idx] {
            // Stale entry
            continue;
        }

        let neighbors: SmallVec<[Position; 6]> = topology.neighbors(current.pos, w, h);
        for neighbor in neighbors {
            let tile = match grid.get_pos(&neighbor) {
                Some(t) => t,
                None => continue,
            };

            if !tile.terrain.is_walkable() {
                continue;
            }

            // Use terrain movement cost (fixed-point → approximate u32 step cost)
            let move_cost = (tile.terrain.movement_cost() as u32).clamp(1, 1_000_000);
            let tentative_g = current.g_cost.saturating_add(move_cost);

            let n_idx = neighbor.y as usize * w as usize + neighbor.x as usize;
            if tentative_g < g_costs[n_idx] {
                g_costs[n_idx] = tentative_g;
                came_from[n_idx] = Some(current.pos);
                let h = topology.distance(neighbor, goal);
                open.push(AStarNode {
                    pos: neighbor,
                    f_cost: tentative_g.saturating_add(h),
                    g_cost: tentative_g,
                });
            }
        }
    }

    None // No path found
}

/// Reconstructs the path from `came_from` map.
fn reconstruct_path(
    came_from: &[Option<Position>],
    width: u16,
    start: Position,
    goal: Position,
) -> Vec<Position> {
    let mut path = Vec::new();
    let mut current = goal;
    loop {
        path.push(current);
        if current == start {
            break;
        }
        let idx = current.y as usize * width as usize + current.x as usize;
        match came_from[idx] {
            Some(prev) => current = prev,
            None => break, // shouldn't happen if path exists
        }
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HexTopology, SquareTopology};
    use forge_types::grid::TerrainType;

    fn make_grid(w: u16, h: u16) -> Grid {
        Grid::new(w, h)
    }

    fn make_grid_with_wall(w: u16, h: u16, wall_positions: &[(u16, u16)]) -> Grid {
        let mut grid = Grid::new(w, h);
        for &(x, y) in wall_positions {
            grid.get_mut(x, y).unwrap().terrain = TerrainType::Wall;
        }
        grid
    }

    // ---- Square grid pathfinding tests ----

    #[test]
    fn test_square_path_same_tile() {
        let grid = make_grid(8, 8);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(3, 3),
            Position::new(3, 3),
        );
        assert_eq!(path, Some(vec![Position::new(3, 3)]));
    }

    #[test]
    fn test_square_path_adjacent() {
        let grid = make_grid(8, 8);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(3, 3),
            Position::new(3, 4),
        );
        let path = path.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], Position::new(3, 3));
        assert_eq!(path[1], Position::new(3, 4));
    }

    #[test]
    fn test_square_path_straight_line() {
        let grid = make_grid(16, 16);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(2, 5),
            Position::new(7, 5),
        );
        let path = path.unwrap();
        // Should go in a straight horizontal line (5 steps)
        assert_eq!(path.len(), 6); // 6 positions including start and end
        assert_eq!(path[0], Position::new(2, 5));
        assert_eq!(path[5], Position::new(7, 5));
    }

    #[test]
    fn test_square_path_around_wall() {
        // Wall at (4,3), (4,4), (4,5) — path from (3,4) to (5,4) must go around
        let grid = make_grid_with_wall(8, 8, &[(4, 3), (4, 4), (4, 5)]);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(3, 4),
            Position::new(5, 4),
        );
        let path = path.unwrap();
        assert!(path.len() > 2); // must detour
        assert_eq!(*path.first().unwrap(), Position::new(3, 4));
        assert_eq!(*path.last().unwrap(), Position::new(5, 4));
        // No position in path should be a wall
        for pos in &path {
            assert_ne!(
                grid.get_pos(pos).unwrap().terrain,
                TerrainType::Wall,
                "path goes through wall at {:?}",
                pos
            );
        }
    }

    #[test]
    fn test_square_path_unreachable() {
        // Surround goal with walls
        let grid = make_grid_with_wall(8, 8, &[(4, 3), (4, 5), (3, 4), (5, 4)]);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(0, 0),
            Position::new(4, 4),
        );
        assert!(path.is_none());
    }

    #[test]
    fn test_square_goal_is_wall() {
        let grid = make_grid_with_wall(8, 8, &[(5, 5)]);
        let path = astar(
            &SquareTopology,
            &grid,
            Position::new(3, 3),
            Position::new(5, 5),
        );
        assert!(path.is_none());
    }

    // ---- Hex grid pathfinding tests ----

    #[test]
    fn test_hex_path_same_tile() {
        let grid = make_grid(8, 8);
        let path = astar(
            &HexTopology,
            &grid,
            Position::new(3, 3),
            Position::new(3, 3),
        );
        assert_eq!(path, Some(vec![Position::new(3, 3)]));
    }

    #[test]
    fn test_hex_path_adjacent() {
        let grid = make_grid(16, 16);
        let start = Position::new(5, 5);
        let neighbors = HexTopology.neighbors(start, 16, 16);
        let goal = neighbors[0];

        let path = astar(&HexTopology, &grid, start, goal).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], start);
        assert_eq!(path[1], goal);
    }

    #[test]
    fn test_hex_path_around_wall() {
        let grid = make_grid_with_wall(16, 16, &[(5, 5), (5, 6), (6, 5)]);
        let path = astar(
            &HexTopology,
            &grid,
            Position::new(4, 5),
            Position::new(7, 5),
        );
        let path = path.unwrap();
        assert!(path.len() > 2);
        for pos in &path {
            assert_ne!(
                grid.get_pos(pos).unwrap().terrain,
                TerrainType::Wall,
                "path goes through wall at {:?}",
                pos
            );
        }
    }

    #[test]
    fn test_hex_path_unreachable() {
        // Surround (5,5) with walls on all 6 hex neighbors
        let center = Position::new(5, 5);
        let walls: Vec<(u16, u16)> = HexTopology
            .neighbors(center, 16, 16)
            .iter()
            .map(|p| (p.x, p.y))
            .collect();
        let grid = make_grid_with_wall(16, 16, &walls);
        let path = astar(
            &HexTopology,
            &grid,
            Position::new(0, 0),
            Position::new(5, 5),
        );
        assert!(path.is_none());
    }
}
