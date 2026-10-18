//! The `Dual` number: a value paired with its spot on the adjoint tape.

use crate::tape;

/// A quantity we're tracking the sensitivity of.
///
/// `Dual` is just a 16-byte `Copy` handle — it doesn't own any derivative
/// data itself, that all lives on the tape. So passing one through a pricing
/// routine costs no more than passing a pair of machine words around.
///
/// A handle is only valid on the tape recording it came from: the same
/// thread, and no `tape::reset` in between. Feeding a stale one to an
/// operation, `backward`, or `Gradient::wrt` panics rather than reading the
/// unrelated node that now sits at its index.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub value: f64,
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl Dual {
    /// Register an independent variable — a market input like Spot,
    /// Volatility, or the risk-free Rate — that we want a sensitivity back
    /// for.
    pub fn var(value: f64) -> Self {
        let (index, generation) = tape::push_leaf();
        Dual {
            value,
            index,
            generation,
        }
    }

    /// Record a one-input operation.
    ///
    /// `partial` is d(result)/d(input), evaluated at the input's current value.
    pub fn unary(value: f64, input: Dual, partial: f64) -> Self {
        let (index, generation) = tape::push_unary(input.slot(), partial);
        Dual {
            value,
            index,
            generation,
        }
    }

    /// Record a two-input operation.
    ///
    /// `d_left` and `d_right` are the partial derivatives of the result with
    /// respect to each input, evaluated at their current values.
    pub fn binary(value: f64, left: Dual, d_left: f64, right: Dual, d_right: f64) -> Self {
        let (index, generation) = tape::push_binary(left.slot(), d_left, right.slot(), d_right);
        Dual {
            value,
            index,
            generation,
        }
    }

    /// Walk this value's adjoint back through every operation that produced
    /// it, which gives you the sensitivity to every input, all at once.
    pub fn backward(&self) -> Gradient {
        Gradient {
            sweep: tape::run_backward(self.slot()),
        }
    }

    fn slot(self) -> (u32, u32) {
        (self.index, self.generation)
    }
}

/// A handle onto the adjoints the most recent backward sweep left behind.
///
/// We hand back this handle instead of a copied buffer so the sweep stays
/// allocation-free. It's reading straight from the thread's live adjoint
/// store, though, so it's only good until the next `backward` or
/// `tape::reset` on that thread. After that, or on another thread, `wrt`
/// panics instead of returning another sweep's numbers. Read every
/// derivative you need before sweeping again.
pub struct Gradient {
    sweep: (u32, u32),
}

impl Gradient {
    /// How sensitive the swept output is to `input`.
    ///
    /// Panics if this gradient is stale, or if `input` is from a different
    /// tape recording (see [`Dual`]).
    pub fn wrt(&self, input: Dual) -> f64 {
        tape::adjoint_of(self.sweep, input.slot())
    }
}
