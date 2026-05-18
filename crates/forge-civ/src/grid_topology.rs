//! Grid topology trait: the abstraction over square vs hex grids.
//!
//! All grid-shape-specific operations go through this trait so that the
//! simulation engine can work on either topology without branching in
//! hot-path code.

use forge_types::grid::Position;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Abstraction over grid topology (square, hex, etc.).
///
/// Implementations provide neighbor iteration, distance calculation,
/// line-of-sight, and area queries. The trait is object-safe but we
/// prefer enum dispatch via [`GridTopologyKind`] for inlining.
pub trait GridTopology {
    /// Number of possible movement directions (4 for square, 6 for hex).
    fn num_directions(&self) -> u8;

    /// Returns the neighbor of `pos` in direction `dir` (0-indexed),
    /// or `None` if out of bounds.
    fn neighbor(&self, pos: Position, dir: u8, w: u16, h: u16) -> Option<Position>;

    /// Returns all in-bounds neighbors of `pos`.
    fn neighbors(&self, pos: Position, w: u16, h: u16) -> SmallVec<[Position; 6]>;

    /// Grid distance between two positions.
    ///
    /// For square grids this is Chebyshev distance; for hex grids
    /// this is the hex (cube-coordinate) distance.
    fn distance(&self, a: Position, b: Position) -> u32;

    /// Returns all grid cells along a line from `from` to `to` (inclusive of
    /// both endpoints).
    ///
    /// For square grids this uses Bresenham's algorithm; for hex grids
    /// this uses cube-coordinate linear interpolation.
    fn line_of_sight(&self, from: Position, to: Position) -> SmallVec<[Position; 16]>;

    /// Returns all in-bounds positions within `radius` of `center`
    /// (inclusive of `center` itself).
    fn disk(&self, center: Position, radius: u16, w: u16, h: u16) -> Vec<Position>;
}

/// Enum dispatch wrapper for [`GridTopology`] implementations.
///
/// Uses enum dispatch instead of `dyn GridTopology` so the compiler
/// can inline topology methods in the hot path (`WorldState::step()`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GridTopologyKind {
    /// Standard 4-neighbor square grid.
    Square(crate::SquareTopology),
    /// 6-neighbor hexagonal grid (odd-r offset).
    Hex(crate::HexTopology),
}

impl GridTopology for GridTopologyKind {
    #[inline]
    fn num_directions(&self) -> u8 {
        match self {
            Self::Square(t) => t.num_directions(),
            Self::Hex(t) => t.num_directions(),
        }
    }

    #[inline]
    fn neighbor(&self, pos: Position, dir: u8, w: u16, h: u16) -> Option<Position> {
        match self {
            Self::Square(t) => t.neighbor(pos, dir, w, h),
            Self::Hex(t) => t.neighbor(pos, dir, w, h),
        }
    }

    #[inline]
    fn neighbors(&self, pos: Position, w: u16, h: u16) -> SmallVec<[Position; 6]> {
        match self {
            Self::Square(t) => t.neighbors(pos, w, h),
            Self::Hex(t) => t.neighbors(pos, w, h),
        }
    }

    #[inline]
    fn distance(&self, a: Position, b: Position) -> u32 {
        match self {
            Self::Square(t) => t.distance(a, b),
            Self::Hex(t) => t.distance(a, b),
        }
    }

    #[inline]
    fn line_of_sight(&self, from: Position, to: Position) -> SmallVec<[Position; 16]> {
        match self {
            Self::Square(t) => t.line_of_sight(from, to),
            Self::Hex(t) => t.line_of_sight(from, to),
        }
    }

    #[inline]
    fn disk(&self, center: Position, radius: u16, w: u16, h: u16) -> Vec<Position> {
        match self {
            Self::Square(t) => t.disk(center, radius, w, h),
            Self::Hex(t) => t.disk(center, radius, w, h),
        }
    }
}

impl Default for GridTopologyKind {
    fn default() -> Self {
        Self::Square(crate::SquareTopology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HexTopology, SquareTopology};

    /// Helper: square-grid `GridTopologyKind`.
    fn square_kind() -> GridTopologyKind {
        GridTopologyKind::Square(SquareTopology)
    }

    /// Helper: hex-grid `GridTopologyKind`.
    fn hex_kind() -> GridTopologyKind {
        GridTopologyKind::Hex(HexTopology)
    }

    /// `Default` impl returns the square variant — used by many configs.
    #[test]
    fn default_is_square_variant() {
        let k = GridTopologyKind::default();
        assert!(matches!(k, GridTopologyKind::Square(_)));
    }

    /// Square dispatch advertises 4 directions; hex dispatches 6.
    #[test]
    fn num_directions_dispatches_to_variant() {
        let sq = square_kind();
        let hx = hex_kind();
        assert_eq!(sq.num_directions(), SquareTopology.num_directions());
        assert_eq!(hx.num_directions(), HexTopology.num_directions());
        assert_eq!(sq.num_directions(), 4);
        assert_eq!(hx.num_directions(), 6);
    }

    /// `neighbor()` through the enum agrees with the concrete variant.
    /// Asserts on both variants so the Hex dispatch arm gets exercised.
    #[test]
    fn neighbor_dispatch_matches_concrete_for_both_variants() {
        let pos = Position::new(2, 2);
        let (w, h) = (8u16, 8u16);
        for dir in 0..6u8 {
            let sq_via_enum = square_kind().neighbor(pos, dir % 4, w, h);
            let sq_direct = SquareTopology.neighbor(pos, dir % 4, w, h);
            assert_eq!(sq_via_enum, sq_direct, "square dir={dir}");

            let hx_via_enum = hex_kind().neighbor(pos, dir, w, h);
            let hx_direct = HexTopology.neighbor(pos, dir, w, h);
            assert_eq!(hx_via_enum, hx_direct, "hex dir={dir}");
        }
    }

    /// `neighbors()` enum dispatch returns the same set as the concrete impl
    /// for both variants.
    #[test]
    fn neighbors_dispatch_matches_concrete_for_both_variants() {
        let pos = Position::new(4, 4);
        let (w, h) = (10u16, 10u16);
        let sq_via_enum = square_kind().neighbors(pos, w, h);
        let sq_direct = SquareTopology.neighbors(pos, w, h);
        assert_eq!(sq_via_enum.len(), sq_direct.len());
        for n in sq_direct.iter() {
            assert!(sq_via_enum.contains(n));
        }

        let hx_via_enum = hex_kind().neighbors(pos, w, h);
        let hx_direct = HexTopology.neighbors(pos, w, h);
        assert_eq!(hx_via_enum.len(), hx_direct.len());
        for n in hx_direct.iter() {
            assert!(hx_via_enum.contains(n));
        }
    }

    /// `distance()` dispatches correctly for both variants.
    #[test]
    fn distance_dispatch_matches_concrete_for_both_variants() {
        let a = Position::new(0, 0);
        let b = Position::new(3, 4);
        assert_eq!(square_kind().distance(a, b), SquareTopology.distance(a, b));
        assert_eq!(hex_kind().distance(a, b), HexTopology.distance(a, b));
    }

    /// `line_of_sight()` dispatches correctly for both variants.
    #[test]
    fn line_of_sight_dispatch_matches_concrete_for_both_variants() {
        let from = Position::new(0, 0);
        let to = Position::new(2, 2);
        let sq = square_kind().line_of_sight(from, to);
        let sq_direct = SquareTopology.line_of_sight(from, to);
        assert_eq!(sq.len(), sq_direct.len());

        let hx = hex_kind().line_of_sight(from, to);
        let hx_direct = HexTopology.line_of_sight(from, to);
        assert_eq!(hx.len(), hx_direct.len());
    }

    /// `disk()` dispatches correctly for both variants.
    #[test]
    fn disk_dispatch_matches_concrete_for_both_variants() {
        let center = Position::new(5, 5);
        let radius = 2u16;
        let (w, h) = (16u16, 16u16);
        let sq = square_kind().disk(center, radius, w, h);
        let sq_direct = SquareTopology.disk(center, radius, w, h);
        assert_eq!(sq.len(), sq_direct.len());

        let hx = hex_kind().disk(center, radius, w, h);
        let hx_direct = HexTopology.disk(center, radius, w, h);
        assert_eq!(hx.len(), hx_direct.len());
    }

    /// Round-trip serialization preserves the variant for both arms.
    #[test]
    fn roundtrip_serde_preserves_variant() {
        for kind in [square_kind(), hex_kind()] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: GridTopologyKind = serde_json::from_str(&json).unwrap();
            match (kind, back) {
                (GridTopologyKind::Square(_), GridTopologyKind::Square(_)) => {}
                (GridTopologyKind::Hex(_), GridTopologyKind::Hex(_)) => {}
                other => panic!("variant changed after roundtrip: {other:?}"),
            }
        }
    }
}
