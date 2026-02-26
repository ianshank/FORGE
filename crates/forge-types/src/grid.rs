//! Grid, tile, and coordinate types for the FORGE world representation.
//!
//! The grid is stored as a flat row-major array for cache-friendly access.
//! All coordinate types are compact for minimal memory footprint.

use serde::{Deserialize, Serialize};

/// A 2D position on the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub x: u16,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
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

/// Terrain types that make up the world grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
#[repr(u8)]
pub enum TerrainType {
    Ground = 0,
    Water = 1,
    Wall = 2,
    Lava = 3,
    Ice = 4,
    Sand = 5,
    Forest = 6,
    Mountain = 7,
}

impl TerrainType {
    /// Whether agents can walk on this terrain.
    pub fn is_walkable(&self) -> bool {
        matches!(
            self,
            TerrainType::Ground | TerrainType::Ice | TerrainType::Sand | TerrainType::Forest
        )
    }

    /// Movement cost multiplier (fixed-point). Higher = more stamina drain.
    /// 65536 = 1.0x cost (normal), 131072 = 2.0x cost, etc.
    pub fn movement_cost(&self) -> i32 {
        match self {
            TerrainType::Ground => 65536,  // 1.0x
            TerrainType::Ice => 32768,     // 0.5x (slippery, less stamina)
            TerrainType::Sand => 98304,    // 1.5x
            TerrainType::Forest => 131072, // 2.0x
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
        for i in 0..8u8 {
            assert!(TerrainType::from_u8(i).is_some());
        }
        assert!(TerrainType::from_u8(8).is_none());
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
        // Walkable terrains have specific costs.
        assert_eq!(TerrainType::Ground.movement_cost(), 65536); // 1.0x
        assert_eq!(TerrainType::Ice.movement_cost(), 32768); // 0.5x
        assert_eq!(TerrainType::Sand.movement_cost(), 98304); // 1.5x
        assert_eq!(TerrainType::Forest.movement_cost(), 131072); // 2.0x

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
}
