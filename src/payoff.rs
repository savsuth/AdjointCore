//! Payoffs the Monte Carlo pricer is generic over, and the scalar type they
//! are written in.
//!
//! A payoff is written once against [`Real`] and then runs two ways: on
//! `Dual`, where every step lands on the tape and one backward sweep gives
//! the pathwise Greeks, and on plain `f64`, where nothing is recorded. The
//! `f64` run is what bump-and-reval uses, so the finite-difference oracle and
//! the adjoint pricer share the exact same path arithmetic.
//!
//! Pathwise differentiation needs the payoff to be continuous in the path
//! (Lipschitz is enough). A call, a put and an arithmetic Asian all qualify;
//! a digital or a barrier does not, and is deliberately not offered here.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::dual::Dual;

/// The arithmetic a payoff and the path simulation need, implemented for
/// `Dual` (recorded on the tape) and for `f64` (not recorded).
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    /// max(x, 0), with the subgradient at 0 taken as 0 (see `Dual::max_zero`).
    fn max_zero(self) -> Self;
}

impl Real for f64 {
    fn exp(self) -> f64 {
        f64::exp(self)
    }

    fn ln(self) -> f64 {
        f64::ln(self)
    }

    fn max_zero(self) -> f64 {
        if self > 0.0 { self } else { 0.0 }
    }
}

impl Real for Dual {
    fn exp(self) -> Dual {
        Dual::exp(self)
    }

    fn ln(self) -> Dual {
        Dual::ln(self)
    }

    fn max_zero(self) -> Dual {
        Dual::max_zero(self)
    }
}

/// An undiscounted payoff observed on equally spaced fixing dates
/// T/n, 2T/n, ..., T, where n = [`Payoff::fixings`].
pub trait Payoff {
    /// How many fixing dates the payoff observes.
    fn fixings(&self) -> usize;

    /// The payoff given the spot at each fixing date, in date order.
    fn value<T: Real>(&self, path: &[T]) -> T;

    /// A control variate for this payoff: the geometric-average Asian call
    /// on the same fixings, whose expectation has a closed form. `None` when
    /// it would not track the payoff (a put, say).
    fn geometric_control(&self) -> Option<GeometricAsianCall> {
        None
    }
}

/// max(S_T - K, 0).
#[derive(Clone, Copy, Debug)]
pub struct EuropeanCall {
    pub strike: f64,
}

/// max(K - S_T, 0).
#[derive(Clone, Copy, Debug)]
pub struct EuropeanPut {
    pub strike: f64,
}

/// max(A - K, 0), where A is the arithmetic average of the spot over
/// `fixings` equally spaced dates ending at T (the spot today is not one of
/// them).
#[derive(Clone, Copy, Debug)]
pub struct AsianCall {
    pub strike: f64,
    pub fixings: usize,
}

/// max(G - K, 0), where G is the geometric average of the spot over
/// `fixings` equally spaced dates ending at T. Its log is normal under
/// Black-Scholes, which is what gives it a closed form
/// ([`crate::geometric_asian_call`]) and makes it the control variate for
/// the arithmetic Asian.
#[derive(Clone, Copy, Debug)]
pub struct GeometricAsianCall {
    pub strike: f64,
    pub fixings: usize,
}

impl Payoff for EuropeanCall {
    fn fixings(&self) -> usize {
        1
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        (path[path.len() - 1] - self.strike).max_zero()
    }

    /// With one fixing the geometric average is the terminal spot, so the
    /// control is the call itself.
    fn geometric_control(&self) -> Option<GeometricAsianCall> {
        Some(GeometricAsianCall {
            strike: self.strike,
            fixings: 1,
        })
    }
}

impl Payoff for EuropeanPut {
    fn fixings(&self) -> usize {
        1
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        (-(path[path.len() - 1] - self.strike)).max_zero()
    }
}

impl Payoff for AsianCall {
    fn fixings(&self) -> usize {
        self.fixings
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        let sum = path[1..].iter().fold(path[0], |acc, &s| acc + s);
        (sum / path.len() as f64 - self.strike).max_zero()
    }

    fn geometric_control(&self) -> Option<GeometricAsianCall> {
        Some(GeometricAsianCall {
            strike: self.strike,
            fixings: self.fixings,
        })
    }
}

impl Payoff for GeometricAsianCall {
    fn fixings(&self) -> usize {
        self.fixings
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        let sum_ln = path[1..].iter().fold(path[0].ln(), |acc, &s| acc + s.ln());
        ((sum_ln / path.len() as f64).exp() - self.strike).max_zero()
    }
}
