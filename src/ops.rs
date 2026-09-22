//! Mathematical operations on `Dual`.
//!
//! Each operation here just works out its value in plain `f64` and records
//! the local partial derivative of that one step on the tape. It's the
//! backward sweep that chains all those partials together — none of these
//! operations knows or cares what expression it's sitting inside of.

use std::ops::{Add, Div, Mul, Sub};

use crate::dual::Dual;
use crate::special;

impl Dual {
    /// Standard normal cumulative distribution, N(x).
    ///
    /// The derivative here is just phi(x), the normal density — and it's the
    /// exact analytical value, not a finite-difference approximation.
    pub fn norm_cdf(self) -> Dual {
        Dual::unary(
            special::norm_cdf(self.value),
            self,
            special::norm_pdf(self.value),
        )
    }

    /// e^x. Its own derivative is itself, so we just reuse the value we
    /// already computed as the partial.
    pub fn exp(self) -> Dual {
        let value = self.value.exp();
        Dual::unary(value, self, value)
    }

    /// sqrt(x), with derivative 1 / (2 * sqrt(x)).
    pub fn sqrt(self) -> Dual {
        let value = self.value.sqrt();
        Dual::unary(value, self, 0.5 / value)
    }

    /// ln(x), with derivative 1/x.
    pub fn ln(self) -> Dual {
        Dual::unary(self.value.ln(), self, 1.0 / self.value)
    }

    /// max(x, 0) — the kink in an option's payoff.
    ///
    /// Away from the kink it's simple: derivative is 1 where x > 0, 0 where
    /// x < 0. Right at x = 0 we just take the subgradient as 0, which is the
    /// standard convention — under a continuous terminal spot distribution,
    /// landing exactly on the kink has probability zero anyway, so it never
    /// actually affects a Monte Carlo pathwise estimator.
    pub fn max_zero(self) -> Dual {
        if self.value > 0.0 {
            Dual::unary(self.value, self, 1.0)
        } else {
            Dual::unary(0.0, self, 0.0)
        }
    }
}

impl Add<Dual> for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual::binary(self.value + rhs.value, self, 1.0, rhs, 1.0)
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, rhs: f64) -> Dual {
        Dual::unary(self.value + rhs, self, 1.0)
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual::unary(self + rhs.value, rhs, 1.0)
    }
}

impl Sub<Dual> for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual::binary(self.value - rhs.value, self, 1.0, rhs, -1.0)
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, rhs: f64) -> Dual {
        Dual::unary(self.value - rhs, self, 1.0)
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual::unary(self - rhs.value, rhs, -1.0)
    }
}

impl Mul<Dual> for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual::binary(self.value * rhs.value, self, rhs.value, rhs, self.value)
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, rhs: f64) -> Dual {
        Dual::unary(self.value * rhs, self, rhs)
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual::unary(self * rhs.value, rhs, self)
    }
}

impl Div<Dual> for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual::binary(
            self.value / rhs.value,
            self,
            1.0 / rhs.value,
            rhs,
            -self.value / (rhs.value * rhs.value),
        )
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, rhs: f64) -> Dual {
        Dual::unary(self.value / rhs, self, 1.0 / rhs)
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual::unary(self / rhs.value, rhs, -self / (rhs.value * rhs.value))
    }
}
