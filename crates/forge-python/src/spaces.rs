//! Gymnasium-compatible space description helpers.
//!
//! Creates Python dicts that describe observation and action spaces
//! in a format compatible with Gymnasium's space API.

use forge_types::config::ForgeConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tracing::instrument;

/// Builds a Gymnasium-compatible observation space description as a Python dict.
///
/// Returns a nested dict describing shape, dtype, and bounds for each observation component:
/// grid_view, inventory, health, stamina, position, messages, and day_phase.
#[instrument(skip_all)]
pub fn observation_space(py: Python<'_>, config: &ForgeConfig) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    let vr = config.agents.default_vision_radius as usize;
    let view_side = 2 * vr + 1;

    // Grid view space
    let grid_dict = PyDict::new_bound(py);
    grid_dict.set_item(
        "shape",
        (
            view_side,
            view_side,
            forge_types::constants::OBS_FEATURES_PER_TILE,
        ),
    )?;
    grid_dict.set_item("low", 0u8)?;
    grid_dict.set_item("high", u8::MAX)?;
    grid_dict.set_item("dtype", "uint8")?;
    dict.set_item("grid_view", grid_dict)?;

    // Inventory space
    let inv_dict = PyDict::new_bound(py);
    let capacity = config.agents.default_carry_capacity as usize;
    inv_dict.set_item("shape", (capacity, 2))?;
    inv_dict.set_item("low", 0u16)?;
    inv_dict.set_item("high", u16::MAX)?;
    inv_dict.set_item("dtype", "uint16")?;
    dict.set_item("inventory", inv_dict)?;

    // Health space
    let health_dict = PyDict::new_bound(py);
    health_dict.set_item("low", 0.0f32)?;
    health_dict.set_item("high", 1.0f32)?;
    health_dict.set_item("dtype", "float32")?;
    dict.set_item("health", health_dict)?;

    // Stamina space
    let stamina_dict = PyDict::new_bound(py);
    stamina_dict.set_item("low", 0.0f32)?;
    stamina_dict.set_item("high", 1.0f32)?;
    stamina_dict.set_item("dtype", "float32")?;
    dict.set_item("stamina", stamina_dict)?;

    // Position space
    let pos_dict = PyDict::new_bound(py);
    pos_dict.set_item("shape", (2,))?;
    pos_dict.set_item("low", 0u16)?;
    pos_dict.set_item("high", 65535u16)?;
    pos_dict.set_item("dtype", "uint16")?;
    dict.set_item("position", pos_dict)?;

    // Messages space
    let msg_dict = PyDict::new_bound(py);
    let buffer_size = config.agents.comm_buffer_size as usize;
    msg_dict.set_item("shape", (buffer_size,))?;
    msg_dict.set_item("low", 0u16)?;
    msg_dict.set_item("high", config.agents.comm_vocab_size)?;
    msg_dict.set_item("dtype", "uint16")?;
    dict.set_item("messages", msg_dict)?;

    // Day phase space
    let day_dict = PyDict::new_bound(py);
    day_dict.set_item("low", 0u8)?;
    day_dict.set_item("high", forge_types::constants::NUM_DAY_PHASES - 1)?;
    day_dict.set_item("dtype", "uint8")?;
    dict.set_item("day_phase", day_dict)?;

    // Flat dimension values for Python wrapper convenience
    dict.set_item("grid_view_height", view_side)?;
    dict.set_item("grid_view_width", view_side)?;
    dict.set_item(
        "grid_view_channels",
        forge_types::constants::OBS_FEATURES_PER_TILE,
    )?;
    dict.set_item("inventory_capacity", capacity)?;

    Ok(dict.unbind().into())
}

/// Builds a Gymnasium-compatible discrete action space description as a Python dict.
///
/// Returns a dict with "type" ("Discrete") and "n" (total actions including drone if enabled).
#[instrument(skip_all)]
pub fn action_space(py: Python<'_>, config: &ForgeConfig) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    let n = forge_types::Action::space_size(config.agents.comm_vocab_size, config.drone.enabled);
    dict.set_item("type", "Discrete")?;
    dict.set_item("n", n)?;
    Ok(dict.unbind().into())
}
