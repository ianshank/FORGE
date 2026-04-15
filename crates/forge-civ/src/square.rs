//! Square grid topology (4-neighbor, cardinal directions).
//!
//! Extracts the existing square-grid logic from `Position::offset()` and
//! `Direction::all()` into a [`GridTopology`] implementation.

use forge_types::grid::{Direction, Position};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::grid_topology::GridTopology;

/// 4-neighbor square grid topology.
///
/// Directions map to the existing [`Direction`] enum (Up=0, Down=1, Left=2, Right=3).
/// Distance is Chebyshev (max of |dx|, |dy|).
/// Line-of-sight uses Bresenham's line algorithm.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SquareTopology;

impl GridTopology for SquareTopology {
    #[inline]
    fn num_directions(&self) -> u8 {
        4
    }

    #[inline]
    fn neighbor(&self, pos: Position, dir: u8, w: u16, h: u16) -> Option<Position> {
        let direction = Direction::from_index(dir)?;
        pos.offset(direction, w, h)
    }

    fn neighbors(&self, pos: Position, w: u16, h: u16) -> SmallVec<[Position; 6]> {
        let mut result = SmallVec::new();
        for dir in Direction::all() {
            if let Some(n) = pos.offset(dir, w, h) {
                result.push(n);
            }
        }
        result
    }

    #[inline]
    fn distance(&self, a: Position, b: Position) -> u32 {
        let dx = (a.x as i32 - b.x as i32).unsigned_abs();
        let dy = (a.y as i32 - b.y as i32).unsigned_abs();
        dx.max(dy)
    }

    fn line_of_sight(&self, from: Position, to: Position) -> SmallVec<[Position; 16]> {
        let mut cells = SmallVec::new();

        let x0 = from.x as i32;
        let y0 = from.y as i32;
        let x1 = to.x as i32;
        let y1 = to.y as i32;

        // Bresenham's line algorithm
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut cx = x0;
        let mut cy = y0;

        loop {
            cells.push(Position::new(cx as u16, cy as u16));

            if cx == x1 && cy == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                cx += sx;
            }
            if e2 < dx {
                err += dx;
                cy += sy;
            }
        }

        cells
    }

    fn disk(&self, center: Position, radius: u16, w: u16, h: u16) -> Vec<Position> {
        let r = radius as i32;
        let cx = center.x as i32;
        let cy = center.y as i32;
        let r_sq = r * r;

        let mut result = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                // Euclidean circle (matches existing visibility behavior)
                if dx * dx + dy * dy > r_sq {
                    continue;
                }
                let tx = cx + dx;
                let ty = cy + dy;
                if tx >= 0 && ty >= 0 && tx < w as i32 && ty < h as i32 {
                    result.push(Position::new(tx as u16, ty as u16));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOPO: SquareTopology = SquareTopology;

    #[test]
    fn test_num_directions() {
        assert_eq!(TOPO.num_directions(), 4);
    }

    #[test]
    fn test_neighbor_center() {
        let pos = Position::new(5, 5);
        // Up (dir=0)
        assert_eq!(TOPO.neighbor(pos, 0, 16, 16), Some(Position::new(5, 4)));
        // Down (dir=1)
        assert_eq!(TOPO.neighbor(pos, 1, 16, 16), Some(Position::new(5, 6)));
        // Left (dir=2)
        assert_eq!(TOPO.neighbor(pos, 2, 16, 16), Some(Position::new(4, 5)));
        // Right (dir=3)
        assert_eq!(TOPO.neighbor(pos, 3, 16, 16), Some(Position::new(6, 5)));
    }

    #[test]
    fn test_neighbor_at_edge() {
        // Top edge: Up is out of bounds
        assert_eq!(TOPO.neighbor(Position::new(5, 0), 0, 16, 16), None);
        // Bottom edge: Down is out of bounds
        assert_eq!(TOPO.neighbor(Position::new(5, 15), 1, 16, 16), None);
        // Left edge: Left is out of bounds
        assert_eq!(TOPO.neighbor(Position::new(0, 5), 2, 16, 16), None);
        // Right edge: Right is out of bounds
        assert_eq!(TOPO.neighbor(Position::new(15, 5), 3, 16, 16), None);
    }

    #[test]
    fn test_neighbor_invalid_dir() {
        assert_eq!(TOPO.neighbor(Position::new(5, 5), 4, 16, 16), None);
        assert_eq!(TOPO.neighbor(Position::new(5, 5), 255, 16, 16), None);
    }

    #[test]
    fn test_neighbors_count() {
        // Interior: 4 neighbors
        assert_eq!(TOPO.neighbors(Position::new(5, 5), 16, 16).len(), 4);
        // Corner: 2 neighbors
        assert_eq!(TOPO.neighbors(Position::new(0, 0), 16, 16).len(), 2);
        // Edge: 3 neighbors
        assert_eq!(TOPO.neighbors(Position::new(5, 0), 16, 16).len(), 3);
    }

    #[test]
    fn test_distance_symmetry() {
        let a = Position::new(3, 7);
        let b = Position::new(8, 2);
        assert_eq!(TOPO.distance(a, b), TOPO.distance(b, a));
    }

    #[test]
    fn test_distance_values() {
        // Same position
        assert_eq!(TOPO.distance(Position::new(5, 5), Position::new(5, 5)), 0);
        // Adjacent
        assert_eq!(TOPO.distance(Position::new(5, 5), Position::new(5, 6)), 1);
        // Chebyshev: max(|5-8|, |5-9|) = max(3, 4) = 4
        assert_eq!(TOPO.distance(Position::new(5, 5), Position::new(8, 9)), 4);
    }

    #[test]
    fn test_line_of_sight_horizontal() {
        let cells = TOPO.line_of_sight(Position::new(2, 5), Position::new(6, 5));
        assert_eq!(cells.len(), 5);
        assert_eq!(cells[0], Position::new(2, 5));
        assert_eq!(cells[4], Position::new(6, 5));
    }

    #[test]
    fn test_line_of_sight_same_tile() {
        let cells = TOPO.line_of_sight(Position::new(3, 3), Position::new(3, 3));
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], Position::new(3, 3));
    }

    #[test]
    fn test_disk_at_center() {
        let tiles = TOPO.disk(Position::new(5, 5), 1, 16, 16);
        // Radius 1 Euclidean disk: only the 4 cardinal + center (not corners, since 1+1 > 1)
        assert_eq!(tiles.len(), 5); // center + 4 cardinal
        assert!(tiles.contains(&Position::new(5, 5)));
        assert!(tiles.contains(&Position::new(5, 4)));
        assert!(tiles.contains(&Position::new(5, 6)));
        assert!(tiles.contains(&Position::new(4, 5)));
        assert!(tiles.contains(&Position::new(6, 5)));
    }

    #[test]
    fn test_disk_at_corner() {
        let tiles = TOPO.disk(Position::new(0, 0), 2, 16, 16);
        // All tiles should be in bounds
        for t in &tiles {
            assert!(t.x < 16 && t.y < 16);
        }
        assert!(tiles.contains(&Position::new(0, 0)));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Distance is always symmetric.
            #[test]
            fn distance_symmetric(
                ax in 0u16..100, ay in 0u16..100,
                bx in 0u16..100, by in 0u16..100,
            ) {
                let a = Position::new(ax, ay);
                let b = Position::new(bx, by);
                prop_assert_eq!(TOPO.distance(a, b), TOPO.distance(b, a));
            }

            /// Neighbor count is at most 4.
            #[test]
            fn neighbor_count_at_most_4(x in 0u16..32, y in 0u16..32) {
                let n = TOPO.neighbors(Position::new(x, y), 32, 32).len();
                prop_assert!(n <= 4);
            }

            /// Distance to self is zero.
            #[test]
            fn distance_to_self_zero(x in 0u16..1000, y in 0u16..1000) {
                let p = Position::new(x, y);
                prop_assert_eq!(TOPO.distance(p, p), 0);
            }
        }
    }
}
