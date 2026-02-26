//! # forge-python
//!
//! Python bindings for the FORGE simulation platform via PyO3.
//!
//! Provides `ForgeEnv`, a Gymnasium-compatible environment class
//! that wraps the Rust simulation engine for use from Python.

use pyo3::prelude::*;

mod config;
mod env;
mod spaces;

/// The `forge_env` Python module.
///
/// Exposes the ForgeEnv class for creating and interacting with
/// FORGE simulation environments from Python.
#[pymodule]
fn forge_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<env::ForgeEnv>()?;
    Ok(())
}
