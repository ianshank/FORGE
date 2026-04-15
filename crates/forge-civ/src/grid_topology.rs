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
