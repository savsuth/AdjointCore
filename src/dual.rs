//! The `Dual` number: a value paired with its spot on the adjoint tape.

use crate::tape;

/// A quantity we're tracking the sensitivity of.
///
/// `Dual` is just a 16-byte `Copy` handle — it doesn't own any derivative
/// data itself, that all lives on the tape. So passing one through a pricing
/// routine costs no more than passing a pair of machine words around.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub value: f64,
    pub(crate) index: u32,
}

impl Dual {
    /// Register an independent variable — a market input like Spot,
    /// Volatility, or the risk-free Rate — that we want a sensitivity back
    /// for.
    pub fn var(value: f64) -> Self {
        Dual {
            value,
            index: tape::push_leaf(),
        }
    }

    /// Record a one-input operation.
    ///
    /// `partial` is d(result)/d(input), evaluated at the input's current value.
    pub fn unary(value: f64, input: Dual, partial: f64) -> Self {
        Dual {
            value,
            index: tape::push_unary(input.index, partial),
        }
    }

    /// Record a two-input operation.
    ///
    /// `d_left` and `d_right` are the partial derivatives of the result with
    /// respect to each input, evaluated at their current values.
    pub fn binary(value: f64, left: Dual, d_left: f64, right: Dual, d_right: f64) -> Self {
        Dual {
            value,
            index: tape::push_binary(left.index, d_left, right.index, d_right),
        }
    }

    /// Walk this value's adjoint back through every operation that produced
    /// it, which gives you the sensitivity to every input, all at once.
    pub fn backward(&self) -> Gradient {
        tape::run_backward(self.index);
        Gradient(())
    }
}

/// A handle onto the adjoints the most recent backward sweep left behind.
///
/// We hand back this handle instead of a copied buffer so the sweep stays
/// allocation-free. It's reading straight from the thread's live adjoint
/// store, though, so calling `backward` again or `tape::reset` will make it
/// stale.
pub struct Gradient(());

impl Gradient {
    /// How sensitive the swept output is to `input`.
    pub fn wrt(&self, input: Dual) -> f64 {
        tape::adjoint_of(input.index)
    }
}
