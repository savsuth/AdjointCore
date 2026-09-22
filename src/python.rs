//! PyO3 bridge — this is what exposes the Monte Carlo pricer to Python as
//! `adjoint_core.mc_pricer`.
//!
//! This is the only place in the whole crate that knows Python exists.
//! Everything coming in is a plain `f64` or `usize`, and everything going
//! out is a `HashMap` — the `Dual` tape and its arena never cross the FFI
//! boundary at all.

use std::collections::HashMap;

use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::pricer::monte_carlo_price;

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

#[pymodule]
fn adjoint_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mc_pricer, m)?)?;
    Ok(())
}
