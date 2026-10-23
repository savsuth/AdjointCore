//! PyO3 bridge — this is what exposes the Monte Carlo pricer to Python as
//! `adjoint_core.mc_pricer`.
//!
//! This is the only place in the whole crate that knows Python exists.
//! Everything coming in is a plain `f64`, `usize`, or list of floats, and
//! everything going out is a `dict` — the `Dual` tape and its arena never
//! cross the FFI boundary at all.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::payoff::EuropeanCall;
use crate::pricer::{monte_carlo_greeks_bucketed, monte_carlo_price};

/// Prices a European Call by Monte Carlo Euler-Maruyama simulation and hands
/// back `{"price", "delta", "vega", "rho"}`.
///
/// Every call seeds its own RNG from OS entropy, so you'll get different
/// results run to run — that's normal for a production pricer. The actual
/// correctness checking happens in the Rust test suite, against a seeded
/// `StdRng`; this binding's only job is moving numbers across the FFI
/// boundary.
#[pyfunction]
fn mc_pricer(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: usize,
) -> HashMap<String, f64> {
    let mut rng = StdRng::from_entropy();
    let result = monte_carlo_price(spot, strike, rate, vol, time, num_paths as u64, &mut rng);
    let greeks = result.greeks;

    HashMap::from([
        ("price".to_string(), greeks.price),
        ("delta".to_string(), greeks.delta),
        ("vega".to_string(), greeks.vega),
        ("rho".to_string(), greeks.rho),
    ])
}

/// Prices a European Call under a piecewise-constant vol curve and hands
/// back `{"price", "delta", "rho", "vegas"}`, where `vegas` is a list with
/// one sensitivity per bucket of `vols`. The buckets split [0, time] into
/// `len(vols)` equal pieces; all of them come from one adjoint sweep per path.
///
/// Seeded from OS entropy, like `mc_pricer`. An empty `vols` raises
/// `ValueError` here rather than reaching the pricer's assertion, which
/// would abort the interpreter (the release profile uses `panic = "abort"`).
#[pyfunction]
fn mc_pricer_buckets<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vols: Vec<f64>,
    time: f64,
    num_paths: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if vols.is_empty() {
        return Err(PyValueError::new_err("vols needs at least one bucket"));
    }
    let mut rng = StdRng::from_entropy();
    let result = monte_carlo_greeks_bucketed(
        &EuropeanCall { strike },
        spot,
        rate,
        &vols,
        time,
        num_paths as u64,
        &mut rng,
    );
    let greeks = result.greeks;

    let out = PyDict::new(py);
    out.set_item("price", greeks.price)?;
    out.set_item("delta", greeks.delta)?;
    out.set_item("rho", greeks.rho)?;
    out.set_item("vegas", greeks.vegas)?;
    Ok(out)
}

#[pymodule]
fn adjoint_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mc_pricer, m)?)?;
    m.add_function(wrap_pyfunction!(mc_pricer_buckets, m)?)?;
    Ok(())
}
