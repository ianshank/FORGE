//! Main ForgeEnv pyclass — Gymnasium-compatible environment wrapper.
//!
//! Wraps the Rust WorldState to provide a Python-friendly interface
//! with numpy array observations and standard Gymnasium step/reset API.

// PyO3 proc-macro generates .into() calls that trigger this lint
#![allow(clippy::useless_conversion)]

use forge_core::WorldState;
use forge_types::config::ForgeConfig;
use forge_types::observation::{Observation, StepInfo};
use forge_types::Action;
use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The main FORGE environment, compatible with the Gymnasium API.
///
/// Wraps the Rust simulation engine and provides observations as
/// numpy arrays for efficient integration with Python ML frameworks.
#[pyclass]
pub struct ForgeEnv {
    state: WorldState,
    config: ForgeConfig,
}

#[pymethods]
impl ForgeEnv {
    /// Creates a new ForgeEnv.
    ///
    /// Args:
    ///     config: Optional Python dict with configuration overrides.
    ///             If None, uses default configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let forge_config = match config {
            Some(dict) => crate::config::config_from_dict(py, dict)?,
            None => ForgeConfig::default(),
        };
        let state = WorldState::new(forge_config.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            state,
            config: forge_config,
        })
    }

    /// Resets the environment to an initial state.
    ///
    /// Args:
    ///     seed: Optional random seed for deterministic reset.
    ///     options: Optional dict of reset options (currently unused).
    ///
    /// Returns:
    ///     Tuple of (observation_dict, info_dict)
    #[pyo3(signature = (seed=None, options=None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        seed: Option<u64>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let _ = options;
        let result = self.state.reset(seed);
        let obs = self.obs_to_dict(py, &result.observations[0])?;
        let info = self.info_to_dict(py, &result.info)?;
        Ok((obs, info).to_object(py))
    }

    /// Advances the environment by one step with the given action.
    ///
    /// Args:
    ///     action: Discrete action index (u32).
    ///
    /// Returns:
    ///     Tuple of (observation_dict, reward, terminated, truncated, info_dict)
    fn step(&mut self, py: Python<'_>, action: u32) -> PyResult<PyObject> {
        let comm_vocab_size = self.config.agents.comm_vocab_size;
        let action = Action::from_discrete(action, comm_vocab_size, self.config.drone.enabled)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid action: {action}"))
            })?;

        // Release GIL during computation
        let result = py.allow_threads(|| self.state.step(&[action]));

        let obs = self.obs_to_dict(py, &result.observations[0])?;
        let reward = result.rewards[0];
        let info = self.info_to_dict(py, &result.info)?;

        Ok((obs, reward, result.terminated, result.truncated, info).to_object(py))
    }

    /// Returns a description of the observation space.
    #[getter]
    fn observation_space(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::spaces::observation_space(py, &self.config)
    }

    /// Returns a description of the action space.
    #[getter]
    fn action_space(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::spaces::action_space(py, &self.config)
    }

    /// Returns an ASCII rendering of the current world state.
    fn render(&self) -> String {
        self.state.to_debug_grid()
    }

    /// Closes the environment. No-op for this implementation.
    fn close(&self) -> PyResult<()> {
        Ok(())
    }

    /// Returns the unwrapped environment (self, since there is no wrapper).
    #[getter]
    fn unwrapped(&self) -> PyResult<()> {
        Ok(())
    }
}

impl ForgeEnv {
    /// Converts an Observation to a Python dict with numpy arrays.
    ///
    /// The dict contains:
    /// - "grid_view": numpy array shape (view_height, view_width, 7) with tile features
    /// - "inventory": numpy array shape (capacity, 2) with (item_type, count)
    /// - "health": float
    /// - "stamina": float
    /// - "position": tuple (x, y)
    /// - "messages": list of ints
    /// - "day_phase": int
    fn obs_to_dict<'py>(&self, py: Python<'py>, obs: &Observation) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        // Grid view: (view_height, view_width, 7) as u8
        let h = obs.view_height as usize;
        let w = obs.view_width as usize;
        let mut grid_data = Vec::with_capacity(h * w * 7);
        for tile in &obs.grid_view {
            grid_data.push(tile.terrain);
            grid_data.push(tile.has_agent as u8);
            grid_data.push(tile.has_object as u8);
            grid_data.push(tile.has_resource as u8);
            grid_data.push(tile.elevation);
            grid_data.push(tile.object_type);
            grid_data.push(tile.resource_type);
        }
        let grid_array = Array3::from_shape_vec((h, w, 7), grid_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create grid array: {e}"
            ))
        })?;
        let py_grid = PyArray3::from_owned_array_bound(py, grid_array);
        dict.set_item("grid_view", py_grid)?;

        // Inventory: (capacity, 2) as u16
        let capacity = obs.inventory.slots.len();
        let mut inv_data = Vec::with_capacity(capacity * 2);
        for &(item_type, count) in &obs.inventory.slots {
            inv_data.push(item_type as u16);
            inv_data.push(count);
        }
        let inv_array = Array2::from_shape_vec((capacity, 2), inv_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create inventory array: {e}"
            ))
        })?;
        let py_inv = PyArray2::from_owned_array_bound(py, inv_array);
        dict.set_item("inventory", py_inv)?;

        // Scalar values
        dict.set_item("health", obs.health)?;
        dict.set_item("stamina", obs.stamina)?;
        dict.set_item("position", (obs.position.0, obs.position.1))?;

        // Messages: list of ints
        let msg_vec: Vec<i64> = obs.messages.iter().map(|&m| m as i64).collect();
        let messages = PyList::new_bound(py, &msg_vec);
        dict.set_item("messages", messages)?;

        // Day phase
        dict.set_item("day_phase", obs.day_phase)?;

        Ok(dict)
    }

    /// Converts StepInfo to a Python dict.
    fn info_to_dict<'py>(&self, py: Python<'py>, info: &StepInfo) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("tick", info.tick)?;
        dict.set_item("agents_alive", &info.agents_alive)?;
        dict.set_item("total_resources", info.total_resources)?;
        dict.set_item("day_phase", info.day_phase)?;

        let task_lists: Vec<Vec<i64>> = info
            .tasks_completed
            .iter()
            .map(|tasks| tasks.iter().map(|&t| t as i64).collect())
            .collect();
        let tasks_completed = PyList::new_bound(py, &task_lists);
        dict.set_item("tasks_completed", tasks_completed)?;

        Ok(dict)
    }
}
