//! Configuration conversion from Python dicts to Rust ForgeConfig.
//!
//! Uses serde_json as an intermediate format: Python dict -> JSON string -> ForgeConfig.

use forge_types::config::ForgeConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Converts a Python dictionary to a ForgeConfig by serializing through JSON.
pub fn config_from_dict(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<ForgeConfig> {
    // Convert Python dict -> JSON string -> ForgeConfig
    let json_module = py.import_bound("json")?;
    let json_str: String = json_module.call_method1("dumps", (dict,))?.extract()?;
    let config: ForgeConfig = serde_json::from_str(&json_str).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid config: {e}"))
    })?;
    Ok(config)
}
