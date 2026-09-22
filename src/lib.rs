//! Reverse-mode Adjoint Algorithmic Differentiation for derivative pricing.

pub mod dual;
pub mod ops;
pub mod pricer;
pub mod special;
pub mod tape;

// The PyO3 bridge (`#[pymodule] fn adjoint_core`) — we keep it out of the
// public Rust API. All that actually needs to be reachable is the
// `PyInit_adjoint_core` symbol it generates, and that works fine no matter
// how it's nested or what its visibility is.
mod python;

pub use dual::{Dual, Gradient};
pub use pricer::{Greeks, MonteCarloResult, black_scholes, monte_carlo_price};
