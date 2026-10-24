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
//! a digital or a barrier does not. Those two are offered with their jump
//! smoothed into a call spread of width h, which makes them Lipschitz: the
//! pathwise Greeks are then unbiased for the smoothed payoff, and biased for
//! the true one by the smoothing alone. With h = 0 they are the exact
//! payoffs, whose prices are unbiased and whose pathwise Greeks miss the jump.

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
    /// 1{x > 0}, with derivative 0 (see `Dual::step`).
    fn step(self) -> Self;
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

    fn step(self) -> f64 {
        if self > 0.0 { 1.0 } else { 0.0 }
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

    fn step(self) -> Dual {
        Dual::step(self)
    }
}

/// The indicator 1{x > 0}, smoothed into a centered call spread of width h:
/// ((x + h/2)+ - (x - h/2)+) / h, a ramp from 0 at x = -h/2 to 1 at x = h/2.
/// With h = 0 it is the exact step.
///
/// The ramp is the indicator averaged over shifts in [-h/2, h/2]. That
/// average is symmetric, so the first-order term of the bias cancels and the
/// bias it adds to a price or Greek is O(h^2), where a one-sided spread
/// would add O(h). Inside the band the pathwise derivative is 1/h, so the
/// variance of a pathwise Greek grows like 1/h as h shrinks.
fn smoothed_step<T: Real>(x: T, width: f64) -> T {
    if width == 0.0 {
        x.step()
    } else {
        ((x + 0.5 * width).max_zero() - (x - 0.5 * width).max_zero()) / width
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

/// A cash-or-nothing call paying 1{S_T > K}, with the indicator smoothed
/// into a call spread of width `smoothing` around the strike (0 = exact).
///
/// The smoothed payoff is (C(K - h/2) - C(K + h/2)) / h in calls, so its
/// expectation has a closed form and the smoothing bias can be measured
/// exactly against [`crate::black_scholes_digital`].
#[derive(Clone, Copy, Debug)]
pub struct DigitalCall {
    pub strike: f64,
    pub smoothing: f64,
}

/// A discretely monitored up-and-out call: max(S_T - K, 0) if the spot is
/// below the barrier at every one of `fixings` equally spaced dates ending
/// at T, else 0. Each fixing's survival indicator 1{S_i < B} is smoothed
/// into a call spread of width `smoothing` around the barrier (0 = exact).
#[derive(Clone, Copy, Debug)]
pub struct UpAndOutCall {
    pub strike: f64,
    pub barrier: f64,
    pub fixings: usize,
    pub smoothing: f64,
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

impl Payoff for DigitalCall {
    fn fixings(&self) -> usize {
        1
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        smoothed_step(path[path.len() - 1] - self.strike, self.smoothing)
    }
}

impl Payoff for UpAndOutCall {
    fn fixings(&self) -> usize {
        self.fixings
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        let call = (path[path.len() - 1] - self.strike).max_zero();
        path.iter().fold(call, |acc, &s| {
            acc * smoothed_step(-(s - self.barrier), self.smoothing)
        })
    }
}
