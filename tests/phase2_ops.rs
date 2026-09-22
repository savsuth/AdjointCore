//! Phase 2 (everything else): the std::ops overloads and the exp/sqrt/ln
//! math functions, checked against what calculus says they should be.
//!
//! Every partial in here is a closed-form fact, not an approximation of
//! anything — so we can afford a tight tolerance (1e-12), much tighter than
//! the looser bound we needed for the NormCDF kernel.

use adjoint_core::{Dual, tape};

const TOL: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err < TOL,
        "{what}: got {actual}, expected {expected}, error {err:e}"
    );
}

// ---- Add ----

#[test]
fn add_dual_dual() {
    tape::reset();
    let x = Dual::var(2.0);
    let y = Dual::var(3.0);
    let z = x + y;
    let grad = z.backward();

    assert_close(z.value, 5.0, "value");
    assert_close(grad.wrt(x), 1.0, "dz/dx");
    assert_close(grad.wrt(y), 1.0, "dz/dy");
}

#[test]
fn add_dual_scalar() {
    tape::reset();
    let x = Dual::var(2.0);
    let z = x + 10.0;
    let grad = z.backward();

    assert_close(z.value, 12.0, "value");
    assert_close(grad.wrt(x), 1.0, "dz/dx");
}

#[test]
fn add_scalar_dual() {
    tape::reset();
    let x = Dual::var(2.0);
    let z = 10.0 + x;
    let grad = z.backward();

    assert_close(z.value, 12.0, "value");
    assert_close(grad.wrt(x), 1.0, "dz/dx");
}

// ---- Sub ----

#[test]
fn sub_dual_dual() {
    tape::reset();
    let x = Dual::var(5.0);
    let y = Dual::var(3.0);
    let z = x - y;
    let grad = z.backward();

    assert_close(z.value, 2.0, "value");
    assert_close(grad.wrt(x), 1.0, "dz/dx");
    assert_close(grad.wrt(y), -1.0, "dz/dy");
}

#[test]
fn sub_dual_scalar() {
    tape::reset();
    let x = Dual::var(5.0);
    let z = x - 3.0;
    let grad = z.backward();

    assert_close(z.value, 2.0, "value");
    assert_close(grad.wrt(x), 1.0, "dz/dx");
}

#[test]
fn sub_scalar_dual() {
    tape::reset();
    let x = Dual::var(3.0);
    let z = 5.0 - x;
    let grad = z.backward();

    assert_close(z.value, 2.0, "value");
    assert_close(grad.wrt(x), -1.0, "dz/dx");
}

// ---- Mul ----

#[test]
fn mul_dual_dual() {
    tape::reset();
    let x = Dual::var(4.0);
    let y = Dual::var(7.0);
    let z = x * y;
    let grad = z.backward();

    assert_close(z.value, 28.0, "value");
    assert_close(grad.wrt(x), 7.0, "dz/dx");
    assert_close(grad.wrt(y), 4.0, "dz/dy");
}

#[test]
fn mul_dual_scalar() {
    tape::reset();
    let x = Dual::var(4.0);
    let z = x * 2.5;
    let grad = z.backward();

    assert_close(z.value, 10.0, "value");
    assert_close(grad.wrt(x), 2.5, "dz/dx");
}

#[test]
fn mul_scalar_dual() {
    tape::reset();
    let x = Dual::var(4.0);
    let z = 2.5 * x;
    let grad = z.backward();

    assert_close(z.value, 10.0, "value");
    assert_close(grad.wrt(x), 2.5, "dz/dx");
}

// ---- Div ----

#[test]
fn div_dual_dual() {
    tape::reset();
    let x = Dual::var(10.0);
    let y = Dual::var(4.0);
    let z = x / y;
    let grad = z.backward();

    // dz/dx = 1/y, dz/dy = -x/y^2
    assert_close(z.value, 2.5, "value");
    assert_close(grad.wrt(x), 0.25, "dz/dx");
    assert_close(grad.wrt(y), -10.0 / 16.0, "dz/dy");
}

#[test]
fn div_dual_scalar() {
    tape::reset();
    let x = Dual::var(10.0);
    let z = x / 4.0;
    let grad = z.backward();

    assert_close(z.value, 2.5, "value");
    assert_close(grad.wrt(x), 0.25, "dz/dx");
}

#[test]
fn div_scalar_dual() {
    tape::reset();
    let x = Dual::var(4.0);
    let z = 10.0 / x;
    let grad = z.backward();

    // z = c/x, dz/dx = -c/x^2
    assert_close(z.value, 2.5, "value");
    assert_close(grad.wrt(x), -10.0 / 16.0, "dz/dx");
}

// ---- exp ----

#[test]
fn exp_forward_and_backward() {
    tape::reset();
    let x = Dual::var(1.5);
    let z = x.exp();
    let grad = z.backward();

    let expected = 1.5f64.exp();
    assert_close(z.value, expected, "value");
    // d/dx exp(x) = exp(x)
    assert_close(grad.wrt(x), expected, "dz/dx");
}

// ---- sqrt ----

#[test]
fn sqrt_forward_and_backward() {
    tape::reset();
    let x = Dual::var(16.0);
    let z = x.sqrt();
    let grad = z.backward();

    assert_close(z.value, 4.0, "value");
    // d/dx sqrt(x) = 1/(2*sqrt(x))
    assert_close(grad.wrt(x), 1.0 / 8.0, "dz/dx");
}

// ---- ln ----

#[test]
fn ln_forward_and_backward() {
    tape::reset();
    let x = Dual::var(std::f64::consts::E);
    let z = x.ln();
    let grad = z.backward();

    assert_close(z.value, 1.0, "value");
    // d/dx ln(x) = 1/x
    assert_close(grad.wrt(x), 1.0 / std::f64::consts::E, "dz/dx");
}

// ---- Composition: a mock Black-Scholes-shaped expression ----

/// A d1-shaped expression: (ln(s/k) + (r + 0.5*vol*vol)*t) / (vol*sqrt(t)).
/// This puts every operator and function to work together in one graph, and
/// checks each partial against a central difference — we re-run the whole
/// expression at perturbed inputs to get that. It's really an end-to-end
/// check that composing things through the tape actually matches composing
/// them in plain f64 calculus.
#[test]
fn composed_expression_matches_central_difference() {
    fn d1(s: f64, k: f64, r: f64, vol: f64, t: f64) -> f64 {
        (s / k).ln() + (r + 0.5 * vol * vol) * t
    }
    fn d1_over_vol_sqrt_t(s: f64, k: f64, r: f64, vol: f64, t: f64) -> f64 {
        d1(s, k, r, vol, t) / (vol * t.sqrt())
    }

    let (s0, k0, r0, vol0, t0) = (100.0, 95.0, 0.03, 0.2, 1.0);

    tape::reset();
    let s = Dual::var(s0);
    let k = Dual::var(k0);
    let r = Dual::var(r0);
    let vol = Dual::var(vol0);
    let t = Dual::var(t0);

    let numerator = (s / k).ln() + (r + 0.5 * vol * vol) * t;
    let denominator = vol * t.sqrt();
    let z = numerator / denominator;
    let grad = z.backward();

    let expected_value = d1_over_vol_sqrt_t(s0, k0, r0, vol0, t0);
    assert_close(z.value, expected_value, "composed value");

    type Bump = Box<dyn Fn(f64) -> f64>;

    const H: f64 = 1e-6;
    let cases: [(f64, Bump); 5] = [
        (
            grad.wrt(s),
            Box::new(move |h| d1_over_vol_sqrt_t(s0 + h, k0, r0, vol0, t0)),
        ),
        (
            grad.wrt(k),
            Box::new(move |h| d1_over_vol_sqrt_t(s0, k0 + h, r0, vol0, t0)),
        ),
        (
            grad.wrt(r),
            Box::new(move |h| d1_over_vol_sqrt_t(s0, k0, r0 + h, vol0, t0)),
        ),
        (
            grad.wrt(vol),
            Box::new(move |h| d1_over_vol_sqrt_t(s0, k0, r0, vol0 + h, t0)),
        ),
        (
            grad.wrt(t),
            Box::new(move |h| d1_over_vol_sqrt_t(s0, k0, r0, vol0, t0 + h)),
        ),
    ];

    for (adjoint, bump) in cases {
        let numerical = (bump(H) - bump(-H)) / (2.0 * H);
        let err = (adjoint - numerical).abs();
        assert!(
            err < 1e-6,
            "adjoint {adjoint} vs central difference {numerical}, error {err:e}"
        );
    }
}

/// If you use the same Dual twice in a chain of operations, the adjoints
/// need to accumulate, not just get overwritten. Here z = (x + 1) * (x - 1),
/// which is x^2 - 1, so we should get dz/dx = 2x.
#[test]
fn repeated_variable_accumulates_through_multiple_ops() {
    tape::reset();
    let x = Dual::var(6.0);
    let z = (x + 1.0) * (x - 1.0);
    let grad = z.backward();

    assert_close(z.value, 35.0, "value");
    assert_close(grad.wrt(x), 12.0, "dz/dx");
}
