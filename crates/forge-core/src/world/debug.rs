//! ASCII rendering and terminal visualization for WorldState.

use tracing::instrument;

use super::WorldState;

impl WorldState {
    /// Returns an ASCII debug representation of the world.
    #[instrument(skip_all)]
    pub fn to_debug_grid(&self) -> String {
        let mut result =
            String::with_capacity((self.grid.width as usize + 1) * self.grid.height as usize);

        for y in 0..self.grid.height {
            for x in 0..self.grid.width {
                let tile = self
                    .grid
                    .get(x, y)
                    .expect("invariant: x and y iterate within grid dimensions");
                let ch = if tile.agent_id.is_some() {
                    'A'
                } else if tile.object_id.is_some() {
                    'O'
                } else if tile.resource_id.is_some() {
                    'R'
                } else {
                    match tile.terrain {
                        forge_types::TerrainType::Ground => '.',
                        forge_types::TerrainType::Water => '~',
                        forge_types::TerrainType::Wall => '#',
                        forge_types::TerrainType::Lava => 'L',
                        forge_types::TerrainType::Ice => 'I',
                        forge_types::TerrainType::Sand => 'S',
                        forge_types::TerrainType::Forest => 'T',
                        forge_types::TerrainType::Mountain => 'M',
                        _ => '?',
                    }
                };
                result.push(ch);
            }
            result.push('\n');
        }

        result
    }
}
