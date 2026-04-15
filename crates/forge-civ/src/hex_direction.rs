//! Re-exports [`HexDirection`] from `forge-types` and provides hex-grid-specific tests.
//!
//! The canonical definition lives in `forge_types::grid::HexDirection` so that
//! the `Action` enum can reference it without a circular dependency.

pub use forge_types::grid::HexDirection;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_directions_count() {
        assert_eq!(HexDirection::ALL.len(), 6);
    }

    #[test]
    fn test_from_index_roundtrip() {
        for (i, &dir) in HexDirection::ALL.iter().enumerate() {
            assert_eq!(HexDirection::from_index(i as u8), Some(dir));
            assert_eq!(dir as u8, i as u8);
        }
    }

    #[test]
    fn test_from_index_invalid() {
        assert_eq!(HexDirection::from_index(6), None);
        assert_eq!(HexDirection::from_index(255), None);
    }

    #[test]
    fn test_opposite_involution() {
        for dir in HexDirection::ALL {
            assert_eq!(dir.opposite().opposite(), dir);
        }
    }

    #[test]
    fn test_opposite_not_self() {
        for dir in HexDirection::ALL {
            assert_ne!(dir.opposite(), dir);
        }
    }

    #[test]
    fn test_even_row_offsets_unique() {
        let offsets: Vec<_> = HexDirection::ALL
            .iter()
            .map(|d| d.offset_even_row())
            .collect();
        for (i, a) in offsets.iter().enumerate() {
            for (j, b) in offsets.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "directions {} and {} have same even-row offset", i, j);
                }
            }
        }
    }

    #[test]
    fn test_odd_row_offsets_unique() {
        let offsets: Vec<_> = HexDirection::ALL
            .iter()
            .map(|d| d.offset_odd_row())
            .collect();
        for (i, a) in offsets.iter().enumerate() {
            for (j, b) in offsets.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "directions {} and {} have same odd-row offset", i, j);
                }
            }
        }
    }

    #[test]
    fn test_east_same_both_parities() {
        // East is always (1, 0) regardless of parity
        assert_eq!(HexDirection::E.offset_even_row(), (1, 0));
        assert_eq!(HexDirection::E.offset_odd_row(), (1, 0));
    }

    #[test]
    fn test_west_same_both_parities() {
        // West is always (-1, 0) regardless of parity
        assert_eq!(HexDirection::W.offset_even_row(), (-1, 0));
        assert_eq!(HexDirection::W.offset_odd_row(), (-1, 0));
    }
}
