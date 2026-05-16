//! Grid, tile, and coordinate types for the FORGE world representation.
//!
//! The grid is stored as a flat row-major array for cache-friendly access.
//! All coordinate types are compact for minimal memory footprint.

use serde::{Deserialize, Serialize};

/// A 2D position on the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    /// Horizontal coordinate (column).
    pub x: u16,
    /// Vertical coordinate (row).
    pub y: u16,
}

impl Position {
    /// Creates a new position.
    pub fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }

    /// Computes Manhattan distance to another position.
    pub fn manhattan_distance(&self, other: &Position) -> u32 {
        let dx = (self.x as i32 - other.x as i32).unsigned_abs();
        let dy = (self.y as i32 - other.y as i32).unsigned_abs();
        dx + dy
    }

    /// Returns the position offset by a direction, if within bounds.
    pub fn offset(&self, dir: Direction, width: u16, height: u16) -> Option<Position> {
        match dir {
            Direction::Up if self.y > 0 => Some(Position::new(self.x, self.y - 1)),
            Direction::Down if self.y + 1 < height => Some(Position::new(self.x, self.y + 1)),
            Direction::Left if self.x > 0 => Some(Position::new(self.x - 1, self.y)),
            Direction::Right if self.x + 1 < width => Some(Position::new(self.x + 1, self.y)),
            _ => None,
        }
    }
}

/// Cardinal directions for movement and facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Direction {
    /// Upward (decreasing y).
    #[default]
    Up = 0,
    /// Downward (increasing y).
    Down = 1,
    /// Leftward (decreasing x).
    Left = 2,
    /// Rightward (increasing x).
    Right = 3,
}

impl Direction {
    /// Returns all four cardinal directions.
    pub fn all() -> [Direction; 4] {
        [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
    }

    /// Returns the opposite direction.
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    /// Converts from a u8 index (0-3).
    pub fn from_index(index: u8) -> Option<Direction> {
        match index {
            0 => Some(Direction::Up),
            1 => Some(Direction::Down),
            2 => Some(Direction::Left),
            3 => Some(Direction::Right),
            _ => None,
        }
    }
}

/// The six directions on a hexagonal grid.
///
/// Layout assumes **odd-r offset** (flat-top hexes with odd rows shifted right).
///
/// ```text
///     NW  NE
///   W  ·  E
///     SW  SE
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum HexDirection {
    /// Northeast (upper-right).
    NE = 0,
    /// East (right).
    E = 1,
    /// Southeast (lower-right).
    SE = 2,
    /// Southwest (lower-left).
    SW = 3,
    /// West (left).
    W = 4,
    /// Northwest (upper-left).
    NW = 5,
}

impl HexDirection {
    /// All six hex directions in canonical order.
    pub const ALL: [HexDirection; 6] = [
        HexDirection::NE,
        HexDirection::E,
        HexDirection::SE,
        HexDirection::SW,
        HexDirection::W,
        HexDirection::NW,
    ];

    /// Returns the (dx, dy) offset for this direction on an **even** row.
    ///
    /// Odd-r layout: even rows (y % 2 == 0) have one offset table,
    /// odd rows (y % 2 == 1) have another.
    #[inline]
    pub fn offset_even_row(self) -> (i32, i32) {
        match self {
            HexDirection::NE => (0, -1),
            HexDirection::E => (1, 0),
            HexDirection::SE => (0, 1),
            HexDirection::SW => (-1, 1),
            HexDirection::W => (-1, 0),
            HexDirection::NW => (-1, -1),
        }
    }

    /// Returns the (dx, dy) offset for this direction on an **odd** row.
    #[inline]
    pub fn offset_odd_row(self) -> (i32, i32) {
        match self {
            HexDirection::NE => (1, -1),
            HexDirection::E => (1, 0),
            HexDirection::SE => (1, 1),
            HexDirection::SW => (0, 1),
            HexDirection::W => (-1, 0),
            HexDirection::NW => (0, -1),
        }
    }

    /// Returns the (dx, dy) offset for the given row parity.
    #[inline]
    pub fn offset_for_parity(self, odd_row: bool) -> (i32, i32) {
        if odd_row {
            self.offset_odd_row()
        } else {
            self.offset_even_row()
        }
    }

    /// Constructs from a `u8` index (0–5).
    #[inline]
    pub fn from_index(index: u8) -> Option<HexDirection> {
        match index {
            0 => Some(HexDirection::NE),
            1 => Some(HexDirection::E),
            2 => Some(HexDirection::SE),
            3 => Some(HexDirection::SW),
            4 => Some(HexDirection::W),
            5 => Some(HexDirection::NW),
            _ => None,
        }
    }

    /// Returns the opposite direction.
    #[inline]
    pub fn opposite(self) -> HexDirection {
        match self {
            HexDirection::NE => HexDirection::SW,
            HexDirection::E => HexDirection::W,
            HexDirection::SE => HexDirection::NW,
            HexDirection::SW => HexDirection::NE,
            HexDirection::W => HexDirection::E,
            HexDirection::NW => HexDirection::SE,
        }
    }
}

/// Terrain types that make up the world grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[repr(u8)]
pub enum TerrainType {
    /// Normal walkable ground.
    Ground = 0,
    /// Impassable water terrain.
    Water = 1,
    /// Impassable wall (blocks vision).
    Wall = 2,
    /// Damaging lava terrain.
    Lava = 3,
    /// Slippery ice (reduced stamina cost).
    Ice = 4,
    /// Sandy terrain (increased stamina cost).
    Sand = 5,
    /// Dense forest (high stamina cost).
    Forest = 6,
    /// Impassable mountain (blocks vision).
    Mountain = 7,
    /// Agricultural cropland (slightly slower ground movement).
    Cropland = 8,
    /// Open pasture / grazing land (normal speed).
    Pasture = 9,
    /// Orchard with tree rows (slow, like forest).
    Orchard = 10,
}

impl TerrainType {
    /// Whether agents can walk on this terrain.
    pub fn is_walkable(&self) -> bool {
        matches!(
            self,
            TerrainType::Ground
                | TerrainType::Ice
                | TerrainType::Sand
                | TerrainType::Forest
                | TerrainType::Cropland
                | TerrainType::Pasture
                | TerrainType::Orchard
        )
    }

    /// Movement cost multiplier (fixed-point). Higher = more stamina drain.
    /// `FIXED_POINT_ONE` (65536) = 1.0x cost (normal), 2*FIXED_POINT_ONE = 2.0x cost, etc.
    pub fn movement_cost(&self) -> i32 {
        use crate::constants;
        match self {
            TerrainType::Ground => constants::TERRAIN_COST_GROUND,
            TerrainType::Ice => constants::TERRAIN_COST_ICE,
            TerrainType::Sand => constants::TERRAIN_COST_SAND,
            TerrainType::Forest => constants::TERRAIN_COST_FOREST,
            TerrainType::Cropland => constants::TERRAIN_COST_CROPLAND,
            TerrainType::Pasture => constants::TERRAIN_COST_PASTURE,
            TerrainType::Orchard => constants::TERRAIN_COST_ORCHARD,
            // Non-walkable terrains return max cost
            _ => i32::MAX,
        }
    }

    /// Whether this terrain blocks line-of-sight.
    pub fn blocks_vision(&self) -> bool {
        matches!(self, TerrainType::Wall | TerrainType::Mountain)
    }

    /// Converts from a u8 discriminant.
    pub fn from_u8(value: u8) -> Option<TerrainType> {
        match value {
            0 => Some(TerrainType::Ground),
            1 => Some(TerrainType::Water),
            2 => Some(TerrainType::Wall),
            3 => Some(TerrainType::Lava),
            4 => Some(TerrainType::Ice),
            5 => Some(TerrainType::Sand),
            6 => Some(TerrainType::Forest),
            7 => Some(TerrainType::Mountain),
            8 => Some(TerrainType::Cropland),
            9 => Some(TerrainType::Pasture),
            10 => Some(TerrainType::Orchard),
            _ => None,
        }
    }
}

/// Visibility state for fog-of-war.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[repr(u8)]
pub enum VisibilityState {
    /// Never been seen.
    Hidden = 0,
    /// Previously seen but not currently visible.
    Explored = 1,
    /// Currently visible to at least one agent.
    Visible = 2,
}

/// Configurable terrain properties, complementing the static [`TerrainType`] methods.
///
/// These are driven by config and can be tuned per scenario.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerrainProperties {
    /// Movement cost multiplier (1.0 = normal).
    pub movement_cost: f64,
    /// Concealment bonus applied to agents on this terrain (0.0-1.0).
    pub concealment_bonus: f64,
    /// Whether the terrain blocks line-of-sight.
    pub blocks_los: bool,
    /// Whether agents can traverse this terrain.
    pub passable: bool,
    /// Defensive bonus for agents standing here (0.0-1.0).
    pub defense_bonus: f64,
}

impl Default for TerrainProperties {
    fn default() -> Self {
        Self {
            movement_cost: 1.0,
            concealment_bonus: 0.0,
            blocks_los: false,
            passable: true,
            defense_bonus: 0.0,
        }
    }
}

impl TerrainProperties {
    /// Returns default properties for a given terrain type.
    pub fn for_terrain(terrain: TerrainType) -> Self {
        match terrain {
            TerrainType::Ground => Self::default(),
            TerrainType::Water => Self {
                movement_cost: f64::MAX,
                passable: false,
                ..Self::default()
            },
            TerrainType::Wall => Self {
                movement_cost: f64::MAX,
                blocks_los: true,
                passable: false,
                defense_bonus: 0.5,
                ..Self::default()
            },
            TerrainType::Lava => Self {
                movement_cost: f64::MAX,
                passable: false,
                ..Self::default()
            },
            TerrainType::Ice => Self {
                movement_cost: 0.5,
                ..Self::default()
            },
            TerrainType::Sand => Self {
                movement_cost: 1.5,
                ..Self::default()
            },
            TerrainType::Forest => Self {
                movement_cost: 2.0,
                concealment_bonus: 0.3,
                ..Self::default()
            },
            TerrainType::Mountain => Self {
                movement_cost: f64::MAX,
                blocks_los: true,
                passable: false,
                defense_bonus: 0.3,
                ..Self::default()
            },
            TerrainType::Cropland => Self {
                movement_cost: 1.2,
                ..Self::default()
            },
            TerrainType::Pasture => Self::default(),
            TerrainType::Orchard => Self {
                movement_cost: 1.5,
                concealment_bonus: 0.2,
                ..Self::default()
            },
        }
    }
}

/// A single tile in the world grid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tile {
    /// The terrain type of this tile.
    pub terrain: TerrainType,
    /// Elevation (0-255). Affects visibility and terrain generation.
    pub elevation: u8,
    /// Agent occupying this tile, if any.
    pub agent_id: Option<u32>,
    /// Object on this tile, if any.
    pub object_id: Option<u32>,
    /// Resource node at this tile, if any.
    pub resource_id: Option<u32>,
    /// Fog-of-war visibility state.
    pub visibility: VisibilityState,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            terrain: TerrainType::Ground,
            elevation: 0,
            agent_id: None,
            object_id: None,
            resource_id: None,
            visibility: VisibilityState::Hidden,
        }
    }
}

/// The 2D grid representing the world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    /// Width of the grid in tiles.
    pub width: u16,
    /// Height of the grid in tiles.
    pub height: u16,
    /// Flat row-major array of tiles. Index = y * width + x.
    pub tiles: Vec<Tile>,
}

impl Grid {
    /// Creates a new grid filled with the default tile.
    pub fn new(width: u16, height: u16) -> Self {
        let size = width as usize * height as usize;
        Self {
            width,
            height,
            tiles: vec![Tile::default(); size],
        }
    }

    /// Returns the tile at (x, y), or None if out of bounds.
    #[inline]
    pub fn get(&self, x: u16, y: u16) -> Option<&Tile> {
        if x < self.width && y < self.height {
            Some(&self.tiles[y as usize * self.width as usize + x as usize])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the tile at (x, y), or None if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, x: u16, y: u16) -> Option<&mut Tile> {
        if x < self.width && y < self.height {
            let idx = y as usize * self.width as usize + x as usize;
            Some(&mut self.tiles[idx])
        } else {
            None
        }
    }

    /// Returns the tile at a Position.
    #[inline]
    pub fn get_pos(&self, pos: &Position) -> Option<&Tile> {
        self.get(pos.x, pos.y)
    }

    /// Returns a mutable reference to the tile at a Position.
    #[inline]
    pub fn get_pos_mut(&mut self, pos: &Position) -> Option<&mut Tile> {
        self.get_mut(pos.x, pos.y)
    }

    /// Checks if a position is within grid bounds.
    #[inline]
    pub fn in_bounds(&self, x: u16, y: u16) -> bool {
        x < self.width && y < self.height
    }

    /// Total number of tiles.
    #[inline]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the grid is empty (zero dimensions).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_manhattan_distance() {
        let a = Position::new(0, 0);
        let b = Position::new(3, 4);
        assert_eq!(a.manhattan_distance(&b), 7);
        assert_eq!(b.manhattan_distance(&a), 7);
    }

    #[test]
    fn test_position_offset_in_bounds() {
        let pos = Position::new(5, 5);
        let up = pos.offset(Direction::Up, 10, 10);
        assert_eq!(up, Some(Position::new(5, 4)));
    }

    #[test]
    fn test_position_offset_out_of_bounds() {
        let pos = Position::new(0, 0);
        assert_eq!(pos.offset(Direction::Up, 10, 10), None);
        assert_eq!(pos.offset(Direction::Left, 10, 10), None);
    }

    #[test]
    fn test_position_offset_at_boundary() {
        let pos = Position::new(9, 9);
        assert_eq!(pos.offset(Direction::Down, 10, 10), None);
        assert_eq!(pos.offset(Direction::Right, 10, 10), None);
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::Up.opposite(), Direction::Down);
        assert_eq!(Direction::Left.opposite(), Direction::Right);
    }

    #[test]
    fn test_direction_from_index() {
        assert_eq!(Direction::from_index(0), Some(Direction::Up));
        assert_eq!(Direction::from_index(4), None);
    }

    #[test]
    fn test_terrain_walkable() {
        assert!(TerrainType::Ground.is_walkable());
        assert!(TerrainType::Forest.is_walkable());
        assert!(TerrainType::Cropland.is_walkable());
        assert!(TerrainType::Pasture.is_walkable());
        assert!(TerrainType::Orchard.is_walkable());
        assert!(!TerrainType::Water.is_walkable());
        assert!(!TerrainType::Wall.is_walkable());
        assert!(!TerrainType::Lava.is_walkable());
    }

    #[test]
    fn test_terrain_blocks_vision() {
        assert!(TerrainType::Wall.blocks_vision());
        assert!(TerrainType::Mountain.blocks_vision());
        assert!(!TerrainType::Ground.blocks_vision());
        assert!(!TerrainType::Forest.blocks_vision());
    }

    #[test]
    fn test_grid_creation() {
        let grid = Grid::new(16, 16);
        assert_eq!(grid.len(), 256);
        assert_eq!(grid.width, 16);
        assert_eq!(grid.height, 16);
    }

    #[test]
    fn test_grid_get_and_set() {
        let mut grid = Grid::new(8, 8);
        assert!(grid.get(0, 0).is_some());
        assert!(grid.get(7, 7).is_some());
        assert!(grid.get(8, 0).is_none());

        if let Some(tile) = grid.get_mut(3, 3) {
            tile.terrain = TerrainType::Water;
        }
        assert_eq!(grid.get(3, 3).unwrap().terrain, TerrainType::Water);
    }

    #[test]
    fn test_grid_position_access() {
        let grid = Grid::new(16, 16);
        let pos = Position::new(5, 5);
        assert!(grid.get_pos(&pos).is_some());

        let oob = Position::new(20, 20);
        assert!(grid.get_pos(&oob).is_none());
    }

    #[test]
    fn test_tile_default() {
        let tile = Tile::default();
        assert_eq!(tile.terrain, TerrainType::Ground);
        assert_eq!(tile.elevation, 0);
        assert!(tile.agent_id.is_none());
        assert!(tile.object_id.is_none());
        assert!(tile.resource_id.is_none());
    }

    #[test]
    fn test_terrain_from_u8() {
        for i in 0..11u8 {
            assert!(
                TerrainType::from_u8(i).is_some(),
                "terrain {} should exist",
                i
            );
        }
        assert!(TerrainType::from_u8(11).is_none());
    }

    #[test]
    fn test_grid_is_empty() {
        // A 1x1 grid has one tile and is therefore not empty.
        let grid = Grid::new(1, 1);
        assert!(!grid.is_empty());
        assert_eq!(grid.len(), 1);

        // A 0x0 grid (zero dimensions) is empty.
        let empty_grid = Grid::new(0, 0);
        assert!(empty_grid.is_empty());
        assert_eq!(empty_grid.len(), 0);
    }

    #[test]
    fn test_terrain_movement_cost() {
        use crate::constants;
        // Walkable terrains have specific costs.
        assert_eq!(
            TerrainType::Ground.movement_cost(),
            constants::TERRAIN_COST_GROUND
        );
        assert_eq!(
            TerrainType::Ice.movement_cost(),
            constants::TERRAIN_COST_ICE
        );
        assert_eq!(
            TerrainType::Sand.movement_cost(),
            constants::TERRAIN_COST_SAND
        );
        assert_eq!(
            TerrainType::Forest.movement_cost(),
            constants::TERRAIN_COST_FOREST
        );

        // Agricultural terrains have specific costs.
        assert_eq!(
            TerrainType::Cropland.movement_cost(),
            constants::TERRAIN_COST_CROPLAND
        );
        assert_eq!(
            TerrainType::Pasture.movement_cost(),
            constants::TERRAIN_COST_PASTURE
        );
        assert_eq!(
            TerrainType::Orchard.movement_cost(),
            constants::TERRAIN_COST_ORCHARD
        );

        // Non-walkable terrains return i32::MAX.
        assert_eq!(TerrainType::Water.movement_cost(), i32::MAX);
        assert_eq!(TerrainType::Wall.movement_cost(), i32::MAX);
        assert_eq!(TerrainType::Lava.movement_cost(), i32::MAX);
        assert_eq!(TerrainType::Mountain.movement_cost(), i32::MAX);
    }

    #[test]
    fn test_direction_from_index_all() {
        // All four valid indices.
        assert_eq!(Direction::from_index(0), Some(Direction::Up));
        assert_eq!(Direction::from_index(1), Some(Direction::Down));
        assert_eq!(Direction::from_index(2), Some(Direction::Left));
        assert_eq!(Direction::from_index(3), Some(Direction::Right));

        // Invalid index returns None.
        assert_eq!(Direction::from_index(4), None);
        assert_eq!(Direction::from_index(255), None);
    }

    // ---- Proptest: grid invariants ----

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_direction() -> impl Strategy<Value = Direction> {
            prop_oneof![
                Just(Direction::Up),
                Just(Direction::Down),
                Just(Direction::Left),
                Just(Direction::Right),
            ]
        }

        proptest! {
            /// Manhattan distance is symmetric.
            #[test]
            fn manhattan_symmetric(
                ax in 0u16..256,
                ay in 0u16..256,
                bx in 0u16..256,
                by in 0u16..256,
            ) {
                let a = Position::new(ax, ay);
                let b = Position::new(bx, by);
                prop_assert_eq!(a.manhattan_distance(&b), b.manhattan_distance(&a));
            }

            /// Manhattan distance is non-negative and satisfies triangle inequality with origin.
            #[test]
            fn manhattan_triangle(
                ax in 0u16..256,
                ay in 0u16..256,
                bx in 0u16..256,
                by in 0u16..256,
            ) {
                let a = Position::new(ax, ay);
                let b = Position::new(bx, by);
                let origin = Position::new(0, 0);
                let ab = a.manhattan_distance(&b);
                let ao = a.manhattan_distance(&origin);
                let ob = origin.manhattan_distance(&b);
                prop_assert!(ab <= ao + ob);
            }

            /// Offset produces in-bounds result or None.
            #[test]
            fn offset_in_bounds(
                x in 0u16..64,
                y in 0u16..64,
                dir in arb_direction(),
            ) {
                let pos = Position::new(x, y);
                if let Some(new_pos) = pos.offset(dir, 64, 64) {
                    prop_assert!(new_pos.x < 64);
                    prop_assert!(new_pos.y < 64);
                }
            }

            /// Grid get is in-bounds for valid coordinates.
            #[test]
            fn grid_get_valid(
                w in 1u16..32,
                h in 1u16..32,
                x in 0u16..32,
                y in 0u16..32,
            ) {
                let grid = Grid::new(w, h);
                if x < w && y < h {
                    prop_assert!(grid.get(x, y).is_some());
                } else {
                    prop_assert!(grid.get(x, y).is_none());
                }
            }

            /// TerrainType from_u8 roundtrips for valid values.
            #[test]
            fn terrain_from_u8_valid(i in 0u8..11) {
                let terrain = TerrainType::from_u8(i).unwrap();
                prop_assert_eq!(terrain as u8, i);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Coverage gap fills (Direction::all, HexDirection, TerrainProperties,
    // Grid::get_mut / get_pos_mut / in_bounds).
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_direction_all_lists_every_variant() {
        let all = Direction::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&Direction::Up));
        assert!(all.contains(&Direction::Down));
        assert!(all.contains(&Direction::Left));
        assert!(all.contains(&Direction::Right));
    }

    #[test]
    fn test_direction_opposite_round_trip_all_variants() {
        for dir in Direction::all() {
            assert_eq!(dir.opposite().opposite(), dir, "opposite is involution");
        }
        assert_eq!(Direction::Down.opposite(), Direction::Up);
        assert_eq!(Direction::Right.opposite(), Direction::Left);
    }

    #[test]
    fn test_hex_direction_constants_match_repr() {
        assert_eq!(HexDirection::ALL.len(), 6);
        for (idx, dir) in HexDirection::ALL.iter().enumerate() {
            assert_eq!(*dir as u8, idx as u8, "index {idx} aligns with repr");
        }
    }

    #[test]
    fn test_hex_direction_offset_even_row_each_variant() {
        assert_eq!(HexDirection::NE.offset_even_row(), (0, -1));
        assert_eq!(HexDirection::E.offset_even_row(), (1, 0));
        assert_eq!(HexDirection::SE.offset_even_row(), (0, 1));
        assert_eq!(HexDirection::SW.offset_even_row(), (-1, 1));
        assert_eq!(HexDirection::W.offset_even_row(), (-1, 0));
        assert_eq!(HexDirection::NW.offset_even_row(), (-1, -1));
    }

    #[test]
    fn test_hex_direction_offset_odd_row_each_variant() {
        assert_eq!(HexDirection::NE.offset_odd_row(), (1, -1));
        assert_eq!(HexDirection::E.offset_odd_row(), (1, 0));
        assert_eq!(HexDirection::SE.offset_odd_row(), (1, 1));
        assert_eq!(HexDirection::SW.offset_odd_row(), (0, 1));
        assert_eq!(HexDirection::W.offset_odd_row(), (-1, 0));
        assert_eq!(HexDirection::NW.offset_odd_row(), (0, -1));
    }

    #[test]
    fn test_hex_direction_offset_for_parity_dispatches() {
        for dir in HexDirection::ALL {
            assert_eq!(dir.offset_for_parity(false), dir.offset_even_row());
            assert_eq!(dir.offset_for_parity(true), dir.offset_odd_row());
        }
    }

    #[test]
    fn test_hex_direction_from_index_full_range() {
        for (idx, expected) in HexDirection::ALL.iter().enumerate() {
            assert_eq!(HexDirection::from_index(idx as u8), Some(*expected));
        }
        assert_eq!(HexDirection::from_index(6), None);
        assert_eq!(HexDirection::from_index(u8::MAX), None);
    }

    #[test]
    fn test_hex_direction_opposite_is_involution() {
        for dir in HexDirection::ALL {
            assert_eq!(dir.opposite().opposite(), dir);
        }
        assert_eq!(HexDirection::NE.opposite(), HexDirection::SW);
        assert_eq!(HexDirection::E.opposite(), HexDirection::W);
        assert_eq!(HexDirection::SE.opposite(), HexDirection::NW);
    }

    #[test]
    fn test_terrain_properties_default_is_walkable_flat() {
        let p = TerrainProperties::default();
        assert!(p.passable);
        assert!(!p.blocks_los);
        assert_eq!(p.movement_cost, 1.0);
        assert_eq!(p.concealment_bonus, 0.0);
        assert_eq!(p.defense_bonus, 0.0);
    }

    #[test]
    fn test_terrain_properties_for_each_terrain_variant() {
        // Walk every variant — exercise every match arm in for_terrain.
        for variant_idx in 0..11u8 {
            let terrain = TerrainType::from_u8(variant_idx)
                .unwrap_or_else(|| panic!("missing variant for {variant_idx}"));
            let props = TerrainProperties::for_terrain(terrain);
            // Sanity: blocking terrain implies impassable.
            if props.blocks_los {
                assert!(!props.passable, "{terrain:?} blocks LOS but is passable");
            }
            // Sanity: f64::MAX movement cost ↔ impassable.
            if props.movement_cost == f64::MAX {
                assert!(
                    !props.passable,
                    "{terrain:?} has MAX cost yet claims passable"
                );
            }
        }
    }

    #[test]
    fn test_terrain_properties_for_terrain_specific_values() {
        let ground = TerrainProperties::for_terrain(TerrainType::Ground);
        assert!(ground.passable && !ground.blocks_los);

        let wall = TerrainProperties::for_terrain(TerrainType::Wall);
        assert!(wall.blocks_los && !wall.passable);
        assert!(wall.defense_bonus > 0.0);

        let forest = TerrainProperties::for_terrain(TerrainType::Forest);
        assert!(forest.passable);
        assert!(forest.concealment_bonus > 0.0);

        let ice = TerrainProperties::for_terrain(TerrainType::Ice);
        assert!(ice.movement_cost < 1.0); // slippery

        let mountain = TerrainProperties::for_terrain(TerrainType::Mountain);
        assert!(mountain.blocks_los && !mountain.passable);

        let pasture = TerrainProperties::for_terrain(TerrainType::Pasture);
        assert_eq!(pasture.movement_cost, 1.0);
    }

    #[test]
    fn test_grid_get_mut_returns_mutable_reference() {
        let mut grid = Grid::new(4, 4);
        {
            let tile = grid.get_mut(2, 2).expect("in-bounds");
            tile.terrain = TerrainType::Lava;
            tile.elevation = 99;
        }
        let t = grid.get(2, 2).unwrap();
        assert_eq!(t.terrain, TerrainType::Lava);
        assert_eq!(t.elevation, 99);
    }

    #[test]
    fn test_grid_get_mut_oob_returns_none() {
        let mut grid = Grid::new(4, 4);
        assert!(grid.get_mut(4, 0).is_none());
        assert!(grid.get_mut(0, 4).is_none());
        assert!(grid.get_mut(99, 99).is_none());
    }

    #[test]
    fn test_grid_get_pos_mut_round_trip() {
        let mut grid = Grid::new(8, 8);
        let pos = Position::new(3, 5);
        grid.get_pos_mut(&pos).unwrap().terrain = TerrainType::Sand;
        assert_eq!(grid.get_pos(&pos).unwrap().terrain, TerrainType::Sand);

        let bad = Position::new(100, 100);
        assert!(grid.get_pos_mut(&bad).is_none());
    }

    #[test]
    fn test_grid_in_bounds_matches_get() {
        let grid = Grid::new(4, 4);
        for x in 0..6u16 {
            for y in 0..6u16 {
                assert_eq!(
                    grid.in_bounds(x, y),
                    grid.get(x, y).is_some(),
                    "in_bounds disagrees with get for ({x},{y})"
                );
            }
        }
    }
}
