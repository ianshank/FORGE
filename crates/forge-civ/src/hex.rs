//! Hexagonal grid topology (odd-r offset layout, 6 neighbors).
//!
//! Uses **offset coordinates** (col, row) for flat-array storage compatibility
//! with existing `Grid`. Internally converts to **cube coordinates** for
//! distance and line-of-sight calculations.
//!
//! # Coordinate Systems
//!
//! - **Offset (odd-r)**: `(x=col, y=row)`. Odd rows are shifted right by half a hex.
//! - **Cube**: `(q, r, s)` with constraint `q + r + s = 0`. Used for math.
//!
//! Reference: <https://www.redblobgames.com/grids/hexagons/>

use forge_types::grid::Position;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::grid_topology::GridTopology;
use crate::hex_direction::HexDirection;

/// 6-neighbor hex grid topology using odd-r offset coordinates.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct HexTopology;

/// Cube coordinates for hex math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CubeCoord {
    q: i32,
    r: i32,
    s: i32,
}

impl CubeCoord {
    #[inline]
    fn new(q: i32, r: i32, s: i32) -> Self {
        debug_assert_eq!(q + r + s, 0, "cube constraint violated");
        Self { q, r, s }
    }
}

/// Converts offset (odd-r) coordinates to cube coordinates.
#[inline]
fn offset_to_cube(x: i32, y: i32) -> CubeCoord {
    let q = x - (y - (y & 1)) / 2;
    let r = y;
    let s = -q - r;
    CubeCoord::new(q, r, s)
}

/// Converts cube coordinates to offset (odd-r) coordinates.
#[inline]
fn cube_to_offset(cube: CubeCoord) -> (i32, i32) {
    let col = cube.q + (cube.r - (cube.r & 1)) / 2;
    let row = cube.r;
    (col, row)
}

/// Rounds fractional cube coordinates to the nearest cube hex.
#[inline]
fn cube_round(fq: f64, fr: f64, fs: f64) -> CubeCoord {
    let mut rq = fq.round() as i32;
    let mut rr = fr.round() as i32;
    let mut rs = fs.round() as i32;

    let q_diff = (rq as f64 - fq).abs();
    let r_diff = (rr as f64 - fr).abs();
    let s_diff = (rs as f64 - fs).abs();

    if q_diff > r_diff && q_diff > s_diff {
        rq = -rr - rs;
    } else if r_diff > s_diff {
        rr = -rq - rs;
    } else {
        rs = -rq - rr;
    }

    CubeCoord::new(rq, rr, rs)
}

/// Linear interpolation between two cube coordinates.
fn cube_linedraw(a: CubeCoord, b: CubeCoord) -> SmallVec<[CubeCoord; 16]> {
    let n = cube_distance(a, b) as usize;
    let mut results = SmallVec::with_capacity(n + 1);

    if n == 0 {
        results.push(a);
        return results;
    }

    let n_f = n as f64;
    for i in 0..=n {
        let t = i as f64 / n_f;
        let fq = a.q as f64 + (b.q - a.q) as f64 * t;
        let fr = a.r as f64 + (b.r - a.r) as f64 * t;
        let fs = a.s as f64 + (b.s - a.s) as f64 * t;
        results.push(cube_round(fq, fr, fs));
    }

    results
}

/// Cube coordinate distance.
#[inline]
fn cube_distance(a: CubeCoord, b: CubeCoord) -> u32 {
    ((a.q - b.q).abs() + (a.r - b.r).abs() + (a.s - b.s).abs()) as u32 / 2
}

impl GridTopology for HexTopology {
    #[inline]
    fn num_directions(&self) -> u8 {
        6
    }

    #[inline]
    fn neighbor(&self, pos: Position, dir: u8, w: u16, h: u16) -> Option<Position> {
        let hex_dir = HexDirection::from_index(dir)?;
        let odd_row = pos.y & 1 == 1;
        let (dx, dy) = hex_dir.offset_for_parity(odd_row);
        let nx = pos.x as i32 + dx;
        let ny = pos.y as i32 + dy;
        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
            Some(Position::new(nx as u16, ny as u16))
        } else {
            None
        }
    }

    fn neighbors(&self, pos: Position, w: u16, h: u16) -> SmallVec<[Position; 6]> {
        let mut result = SmallVec::new();
        let odd_row = pos.y & 1 == 1;
        for &dir in &HexDirection::ALL {
            let (dx, dy) = dir.offset_for_parity(odd_row);
            let nx = pos.x as i32 + dx;
            let ny = pos.y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                result.push(Position::new(nx as u16, ny as u16));
            }
        }
        result
    }

    #[inline]
    fn distance(&self, a: Position, b: Position) -> u32 {
        let ca = offset_to_cube(a.x as i32, a.y as i32);
        let cb = offset_to_cube(b.x as i32, b.y as i32);
        cube_distance(ca, cb)
    }

    fn line_of_sight(&self, from: Position, to: Position) -> SmallVec<[Position; 16]> {
        let ca = offset_to_cube(from.x as i32, from.y as i32);
        let cb = offset_to_cube(to.x as i32, to.y as i32);
        let cube_cells = cube_linedraw(ca, cb);

        cube_cells
            .into_iter()
            .map(|c| {
                let (col, row) = cube_to_offset(c);
                Position::new(col as u16, row as u16)
            })
            .collect()
    }

    fn disk(&self, center: Position, radius: u16, w: u16, h: u16) -> Vec<Position> {
        let cc = offset_to_cube(center.x as i32, center.y as i32);
        let r = radius as i32;
        let mut result = Vec::new();

        for dq in -r..=r {
            let r1 = (-r).max(-dq - r);
            let r2 = r.min(-dq + r);
            for dr in r1..=r2 {
                let ds = -dq - dr;
                let cube = CubeCoord::new(cc.q + dq, cc.r + dr, cc.s + ds);
                let (col, row) = cube_to_offset(cube);
                if col >= 0 && row >= 0 && col < w as i32 && row < h as i32 {
                    result.push(Position::new(col as u16, row as u16));
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOPO: HexTopology = HexTopology;

    // ---- Cube coordinate conversion tests ----

    #[test]
    fn test_offset_to_cube_origin() {
        let c = offset_to_cube(0, 0);
        assert_eq!(c.q + c.r + c.s, 0);
        assert_eq!(c, CubeCoord::new(0, 0, 0));
    }

    #[test]
    fn test_offset_cube_roundtrip() {
        for y in 0..8 {
            for x in 0..8 {
                let cube = offset_to_cube(x, y);
                assert_eq!(cube.q + cube.r + cube.s, 0, "cube constraint at ({x}, {y})");
                let (ox, oy) = cube_to_offset(cube);
                assert_eq!((ox, oy), (x, y), "roundtrip failed at ({x}, {y})");
            }
        }
    }

    // ---- Neighbor tests ----

    #[test]
    fn test_num_directions() {
        assert_eq!(TOPO.num_directions(), 6);
    }

    #[test]
    fn test_neighbors_interior_count() {
        let n = TOPO.neighbors(Position::new(5, 5), 16, 16);
        assert_eq!(n.len(), 6, "interior hex should have 6 neighbors");
    }

    #[test]
    fn test_neighbors_corner_bounded() {
        let n = TOPO.neighbors(Position::new(0, 0), 16, 16);
        assert!(n.len() <= 6);
        assert!(n.len() >= 2); // corner has at least 2 in-bounds neighbors
    }

    #[test]
    fn test_neighbor_parity_even_row() {
        // Row 4 (even): NE should go to (col, row-1) = (5, 3) -> offset (0, -1) for even
        let pos = Position::new(5, 4);
        let ne = TOPO.neighbor(pos, 0, 16, 16); // NE
        assert_eq!(ne, Some(Position::new(5, 3)));
    }

    #[test]
    fn test_neighbor_parity_odd_row() {
        // Row 5 (odd): NE should go to (col+1, row-1) = (6, 4)
        let pos = Position::new(5, 5);
        let ne = TOPO.neighbor(pos, 0, 16, 16); // NE
        assert_eq!(ne, Some(Position::new(6, 4)));
    }

    #[test]
    fn test_neighbor_east() {
        // East is always (col+1, row) regardless of parity
        assert_eq!(
            TOPO.neighbor(Position::new(5, 4), 1, 16, 16),
            Some(Position::new(6, 4))
        );
        assert_eq!(
            TOPO.neighbor(Position::new(5, 5), 1, 16, 16),
            Some(Position::new(6, 5))
        );
    }

    #[test]
    fn test_neighbor_west() {
        // West is always (col-1, row) regardless of parity
        assert_eq!(
            TOPO.neighbor(Position::new(5, 4), 4, 16, 16),
            Some(Position::new(4, 4))
        );
        assert_eq!(
            TOPO.neighbor(Position::new(5, 5), 4, 16, 16),
            Some(Position::new(4, 5))
        );
    }

    #[test]
    fn test_neighbor_invalid_dir() {
        assert_eq!(TOPO.neighbor(Position::new(5, 5), 6, 16, 16), None);
    }

    // ---- Distance tests ----

    #[test]
    fn test_distance_self() {
        assert_eq!(TOPO.distance(Position::new(5, 5), Position::new(5, 5)), 0);
    }

    #[test]
    fn test_distance_symmetric() {
        let a = Position::new(3, 7);
        let b = Position::new(8, 2);
        assert_eq!(TOPO.distance(a, b), TOPO.distance(b, a));
    }

    #[test]
    fn test_distance_adjacent() {
        // All 6 neighbors of (5,5) should be at distance 1
        let pos = Position::new(5, 5);
        for n in TOPO.neighbors(pos, 32, 32) {
            assert_eq!(
                TOPO.distance(pos, n),
                1,
                "neighbor {:?} should be distance 1",
                n
            );
        }
    }

    #[test]
    fn test_distance_two_steps() {
        // Two east moves from (0,0): (0,0) → (1,0) → (2,0)
        assert_eq!(TOPO.distance(Position::new(0, 0), Position::new(2, 0)), 2);
    }

    // ---- Line-of-sight tests ----

    #[test]
    fn test_los_same_tile() {
        let cells = TOPO.line_of_sight(Position::new(3, 3), Position::new(3, 3));
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], Position::new(3, 3));
    }

    #[test]
    fn test_los_endpoints_included() {
        let from = Position::new(2, 2);
        let to = Position::new(5, 5);
        let cells = TOPO.line_of_sight(from, to);
        assert_eq!(*cells.first().unwrap(), from);
        assert_eq!(*cells.last().unwrap(), to);
    }

    #[test]
    fn test_los_length_matches_distance() {
        let from = Position::new(3, 3);
        let to = Position::new(6, 6);
        let cells = TOPO.line_of_sight(from, to);
        let dist = TOPO.distance(from, to);
        assert_eq!(cells.len(), dist as usize + 1);
    }

    // ---- Disk tests ----

    #[test]
    fn test_disk_radius_0() {
        let tiles = TOPO.disk(Position::new(5, 5), 0, 16, 16);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], Position::new(5, 5));
    }

    #[test]
    fn test_disk_radius_1() {
        let tiles = TOPO.disk(Position::new(5, 5), 1, 16, 16);
        assert_eq!(tiles.len(), 7); // center + 6 neighbors
        assert!(tiles.contains(&Position::new(5, 5)));
    }

    #[test]
    fn test_disk_radius_2() {
        let tiles = TOPO.disk(Position::new(8, 8), 2, 32, 32);
        // Hex ring 0 = 1, ring 1 = 6, ring 2 = 12 → total = 19
        assert_eq!(tiles.len(), 19);
    }

    #[test]
    fn test_disk_bounds_clipping() {
        // At corner (0,0), many hex positions are out of bounds
        let tiles = TOPO.disk(Position::new(0, 0), 2, 16, 16);
        for t in &tiles {
            assert!(t.x < 16 && t.y < 16, "tile {:?} out of bounds", t);
        }
        assert!(tiles.len() < 19); // fewer than full ring due to clipping
    }

    // ---- Property tests ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Distance is always symmetric.
            #[test]
            fn distance_symmetric(
                ax in 0u16..50, ay in 0u16..50,
                bx in 0u16..50, by in 0u16..50,
            ) {
                let a = Position::new(ax, ay);
                let b = Position::new(bx, by);
                prop_assert_eq!(TOPO.distance(a, b), TOPO.distance(b, a));
            }

            /// Distance to self is zero.
            #[test]
            fn distance_to_self_zero(x in 0u16..100, y in 0u16..100) {
                let p = Position::new(x, y);
                prop_assert_eq!(TOPO.distance(p, p), 0);
            }

            /// Neighbor count is at most 6.
            #[test]
            fn neighbor_count_at_most_6(x in 0u16..32, y in 0u16..32) {
                let n = TOPO.neighbors(Position::new(x, y), 32, 32).len();
                prop_assert!(n <= 6);
            }

            /// All neighbors are at distance 1.
            #[test]
            fn neighbors_at_distance_1(x in 1u16..30, y in 1u16..30) {
                let pos = Position::new(x, y);
                for n in TOPO.neighbors(pos, 32, 32) {
                    prop_assert_eq!(TOPO.distance(pos, n), 1);
                }
            }

            /// Cube coordinate roundtrip is lossless.
            #[test]
            fn cube_roundtrip(x in 0i32..100, y in 0i32..100) {
                let cube = offset_to_cube(x, y);
                let (ox, oy) = cube_to_offset(cube);
                prop_assert_eq!((ox, oy), (x, y));
            }

            /// Triangle inequality holds for hex distance.
            #[test]
            fn triangle_inequality(
                ax in 0u16..30, ay in 0u16..30,
                bx in 0u16..30, by in 0u16..30,
                cx in 0u16..30, cy in 0u16..30,
            ) {
                let a = Position::new(ax, ay);
                let b = Position::new(bx, by);
                let c = Position::new(cx, cy);
                let ab = TOPO.distance(a, b);
                let bc = TOPO.distance(b, c);
                let ac = TOPO.distance(a, c);
                prop_assert!(ac <= ab + bc, "triangle inequality: {} <= {} + {}", ac, ab, bc);
            }

            /// Disk at radius r contains exactly the right number of tiles when not clipped.
            #[test]
            fn disk_center_count(r in 0u16..5) {
                // Place center far from edges so no clipping
                let tiles = TOPO.disk(Position::new(20, 20), r, 64, 64);
                // Hex disk formula: 1 + 3*r*(r+1)
                let expected = 1 + 3 * (r as usize) * (r as usize + 1);
                prop_assert_eq!(tiles.len(), expected, "disk radius {}", r);
            }
        }
    }
}
