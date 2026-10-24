//! Reverse-mode Adjoint Algorithmic Differentiation for derivative pricing.

pub mod bump;
pub mod dual;
pub mod ops;
pub mod payoff;
pub mod pricer;
pub mod special;
pub mod tape;
pub mod variance;

// The PyO3 bridge (`#[pymodule] fn adjoint_core`) — we keep it out of the
// public Rust API. All that actually needs to be reachable is the
// `PyInit_adjoint_core` symbol it generates, and that works fine no matter
// how it's nested or what its visibility is.
mod python;

pub use bump::bump_and_reval;
pub use dual::{Dual, Gradient};
pub use payoff::{
    AsianCall, DigitalCall, EuropeanCall, EuropeanPut, GeometricAsianCall, Payoff, Real,
    UpAndOutCall,
};
pub use pricer::{
    BucketedGreeks, BucketedResult, Greeks, Market, MonteCarloResult, PathChunks, black_scholes,
    black_scholes_digital, black_scholes_put, monte_carlo_greeks, monte_carlo_greeks_bucketed,
    monte_carlo_greeks_parallel, monte_carlo_price, monte_carlo_value, monte_carlo_value_bucketed,
};
pub use variance::{VarianceReduction, geometric_asian_call, monte_carlo_greeks_reduced};
