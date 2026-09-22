//! Phase 1: making sure the tape records operations correctly and that the
//! backward sweep actually gets back the sensitivities we'd expect from
//! doing the calculus by hand.

use adjoint_core::{Dual, tape};

const TOL: f64 = 1e-12;

/// z = x * y + exp(x), differentiated in one reverse sweep. We should get
/// dz/dx = y + exp(x) and dz/dy = x back.
#[test]
fn records_product_plus_exponential() {
    tape::reset();

    let x = Dual::var(2.0);
    let y = Dual::var(3.0);

    let xy = Dual::binary(x.value * y.value, x, y.value, y, x.value);

    let exp_x = x.value.exp();
    let e = Dual::unary(exp_x, x, exp_x);

    let z = Dual::binary(xy.value + e.value, xy, 1.0, e, 1.0);

    let grad = z.backward();

    assert!(
        (z.value - (6.0 + 2f64.exp())).abs() < TOL,
        "value: {}",
        z.value
    );
    assert!(
        (grad.wrt(x) - (3.0 + 2f64.exp())).abs() < TOL,
        "dz/dx: {}",
        grad.wrt(x)
    );
    assert!((grad.wrt(y) - 2.0).abs() < TOL, "dz/dy: {}", grad.wrt(y));
}

/// If a node feeds the output twice, both contributions need to get summed.
/// Here z = x * x, so we should end up with dz/dx = 2x.
#[test]
fn accumulates_adjoint_for_repeated_input() {
    tape::reset();

    let x = Dual::var(7.0);
    let z = Dual::binary(x.value * x.value, x, x.value, x, x.value);

    let grad = z.backward();

    assert!((grad.wrt(x) - 14.0).abs() < TOL, "dz/dx: {}", grad.wrt(x));
}

/// An input that doesn't actually feed the output should carry zero
/// sensitivity, even though it's sitting on the same tape.
#[test]
fn unrelated_input_has_zero_sensitivity() {
    tape::reset();

    let x = Dual::var(2.0);
    let unused = Dual::var(5.0);
    let z = Dual::unary(x.value * x.value, x, 2.0 * x.value);

    let grad = z.backward();

    assert!((grad.wrt(x) - 4.0).abs() < TOL);
    assert!(grad.wrt(unused).abs() < TOL);
}

#[test]
fn reset_clears_the_tape_for_the_next_path() {
    tape::reset();

    let x = Dual::var(1.0);
    let _ = Dual::unary(x.value.exp(), x, x.value.exp());
    assert_eq!(tape::len(), 2);

    tape::reset();
    assert_eq!(tape::len(), 0);

    let fresh = Dual::var(1.0);
    assert_eq!(tape::len(), 1);
    assert!((fresh.backward().wrt(fresh) - 1.0).abs() < TOL);
}
