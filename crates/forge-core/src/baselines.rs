//! Deterministic heuristic policies used as graded baselines.
//!
//! These are not learned agents. Coverage uses a boustrophedon (lawnmower)
//! sweep so SAC / random can be compared against a known geometric policy.

use forge_types::grid::{Direction, Position};
use forge_types::Action;

/// Boustrophedon tile order covering the interior after `margin`.
pub fn boustrophedon_tiles(width: u16, height: u16, margin: u16) -> Vec<Position> {
    let x0 = margin;
    let y0 = margin;
    let x1 = width.saturating_sub(margin);
    let y1 = height.saturating_sub(margin);
    if x0 >= x1 || y0 >= y1 {
        return Vec::new();
    }
    let mut tiles = Vec::new();
    let mut row_even = true;
    for y in y0..y1 {
        if row_even {
            for x in x0..x1 {
                tiles.push(Position::new(x, y));
            }
        } else {
            for x in (x0..x1).rev() {
                tiles.push(Position::new(x, y));
            }
        }
        row_even = !row_even;
    }
    tiles
}

/// Cardinal steps from `from` to `to` (Manhattan, no diagonal).
pub fn manhattan_steps(from: Position, to: Position) -> Vec<Action> {
    let mut actions = Vec::new();
    let mut x = from.x;
    let mut y = from.y;
    while x < to.x {
        actions.push(Action::Move(Direction::Right));
        x += 1;
    }
    while x > to.x {
        actions.push(Action::Move(Direction::Left));
        x -= 1;
    }
    while y < to.y {
        actions.push(Action::Move(Direction::Down));
        y += 1;
    }
    while y > to.y {
        actions.push(Action::Move(Direction::Up));
        y -= 1;
    }
    actions
}

/// Take off, sweep every interior tile with a multispectral scan, return home, land.
pub fn lawnmower_coverage_actions(
    width: u16,
    height: u16,
    home: Position,
    margin: u16,
) -> Vec<Action> {
    let mut actions = vec![Action::TakeOff];
    let mut cursor = home;
    for tile in boustrophedon_tiles(width, height, margin) {
        actions.extend(manhattan_steps(cursor, tile));
        actions.push(Action::ScanMultispectral);
        cursor = tile;
    }
    actions.extend(manhattan_steps(cursor, home));
    actions.push(Action::Land);
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boustrophedon_covers_every_tile_once() {
        let tiles = boustrophedon_tiles(4, 3, 0);
        assert_eq!(tiles.len(), 12);
        let unique: std::collections::HashSet<_> = tiles.iter().copied().collect();
        assert_eq!(unique.len(), 12);
        assert_eq!(tiles[0], Position::new(0, 0));
        assert_eq!(tiles[3], Position::new(3, 0));
        assert_eq!(tiles[4], Position::new(3, 1));
    }

    #[test]
    fn lawnmower_starts_airborne_and_ends_landed() {
        let actions = lawnmower_coverage_actions(2, 2, Position::new(0, 0), 0);
        assert_eq!(actions.first(), Some(&Action::TakeOff));
        assert_eq!(actions.last(), Some(&Action::Land));
        assert!(actions
            .iter()
            .any(|a| matches!(a, Action::ScanMultispectral)));
    }
}
