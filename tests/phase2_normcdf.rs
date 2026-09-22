//! Phase 2 (first operation): checking that the normal CDF kernel really
//! does hold double precision, and that its tape node differentiates
//! correctly to the normal density.
//!
//! We generated the reference values ourselves, independently, in 60-digit
//! arithmetic with mpmath and rounded them to f64 — so this is checking the
//! approximation against actual true values, not just against some other
//! double-precision library that could be wrong in the same way ours is.

use adjoint_core::special::{norm_cdf, norm_pdf};
use adjoint_core::{Dual, tape};

/// (x, N(x)) in 60-digit arithmetic, rounded to f64.
const CDF_REFERENCE: &[(f64, f64)] = &[
    (-40.0, 0.0),
    (-37.5, 4.605353009581955e-308),
    (-20.0, 2.7536241186062337e-89),
    (-10.0, 7.619853024160525e-24),
    (-8.0, 6.220960574271784e-16),
    (-7.5, 3.1908916729108963e-14),
    (-7.0710678118654755, 7.68729897214016e-13),
    (-7.0, 1.279812543885835e-12),
    (-5.0, 2.866515718791939e-07),
    (-3.0, 0.0013498980316300946),
    (-2.5, 0.006209665325776135),
    // These straddle the series/continued-fraction split at 1.5, which is
    // where the dense sweep found the worst case. Without them the test
    // never actually exercises its own tolerance.
    (-2.0, 0.02275013194817921),
    (-1.959963984540054, 0.025000000000000012),
    (-1.75, 0.04005915686381709),
    (-1.575, 0.05762822227615315),
    (-1.5, 0.06680720126885807),
    (-1.45, 0.07352925960964836),
    (-1.35, 0.08850799143740203),
    (-1.25, 0.10564977366685525),
    (-1.0, 0.15865525393145705),
    (-0.5, 0.3085375387259869),
    (-0.25, 0.4012936743170763),
    (-0.1, 0.460172162722971),
    (-1e-08, 0.4999999960105772),
    (0.0, 0.5),
    (1e-08, 0.5000000039894228),
    (0.1, 0.539827837277029),
    (0.25, 0.5987063256829237),
    (0.5, 0.6914624612740131),
    (1.0, 0.8413447460685429),
    (1.25, 0.8943502263331448),
    (1.35, 0.9114920085625979),
    (1.45, 0.9264707403903516),
    (1.5, 0.9331927987311419),
    (1.575, 0.9423717777238468),
    (1.75, 0.9599408431361829),
    (1.959963984540054, 0.975),
    (2.0, 0.9772498680518208),
    (2.5, 0.9937903346742238),
    (3.0, 0.9986501019683699),
    (5.0, 0.9999997133484281),
    (7.0, 0.9999999999987201),
    (7.0710678118654755, 0.9999999999992313),
    (7.5, 0.9999999999999681),
    (8.0, 0.9999999999999993),
    (10.0, 1.0),
    (20.0, 1.0),
    (37.5, 1.0),
    (40.0, 1.0),
];

/// (x, phi(x)) in 60-digit arithmetic, rounded to f64.
const PDF_REFERENCE: &[(f64, f64)] = &[
    (-7.0710678118654755, 5.540487995575823e-12),
    (-3.0, 0.0044318484119380075),
    (-1.959963984540054, 0.05844506980503538),
    (-1.0, 0.24197072451914334),
    (-0.5, 0.35206532676429947),
    (0.0, 0.3989422804014327),
    (0.5, 0.35206532676429947),
    (1.0, 0.24197072451914334),
    (1.959963984540054, 0.05844506980503538),
    (3.0, 0.0044318484119380075),
    (7.0710678118654755, 5.540487995575823e-12),
];

/// Below -37.5 the true result is subnormal, so relative error doesn't mean
/// anything down there anyway — the kernel just flushes to zero on purpose.
const FLUSH_LIMIT: f64 = 37.5;

/// We swept the shipped kernel against the 60-digit references every 0.005
/// and found a worst case of 2.6e-15, near x = -2.0. This tolerance gives
/// about 4x margin over that measured worst case, and it's still nearly six
/// orders of magnitude tighter than the Hart/West approximation we replaced.
const CDF_TOL: f64 = 1e-14;

fn relative_error(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

#[test]
fn norm_cdf_holds_double_precision() {
    let mut worst = 0.0f64;
    let mut worst_at = f64::NAN;

    for &(x, expected) in CDF_REFERENCE {
        if x < -FLUSH_LIMIT {
            let err = (norm_cdf(x) - expected).abs();
            assert!(err < 1e-300, "N({x}) absolute error {err:e}");
            continue;
        }

        let err = relative_error(norm_cdf(x), expected);
        if err > worst {
            worst = err;
            worst_at = x;
        }
        assert!(
            err < CDF_TOL,
            "N({x}) = {} vs {expected}, relative error {err:e}",
            norm_cdf(x)
        );
    }

    eprintln!("norm_cdf worst relative error {worst:e} at x = {worst_at}");
}

#[test]
fn norm_pdf_holds_double_precision() {
    for &(x, expected) in PDF_REFERENCE {
        let err = relative_error(norm_pdf(x), expected);
        assert!(
            err < 1e-15,
            "phi({x}) = {} vs {expected}, relative error {err:e}",
            norm_pdf(x)
        );
    }
}

/// The lower tail needs to keep *relative* accuracy, not just absolute.
/// N(-8) is 6.2e-16 — if we formed that as 1 - N(8), we'd be left with
/// nothing but rounding noise, and Phase 3 depends on deep-tail Greeks
/// actually meaning something.
#[test]
fn lower_tail_keeps_relative_accuracy() {
    for &(x, expected) in CDF_REFERENCE {
        if !(-FLUSH_LIMIT..=-5.0).contains(&x) {
            continue;
        }
        let err = relative_error(norm_cdf(x), expected);
        assert!(err < CDF_TOL, "N({x}) relative error {err:e}");
    }
}

/// N(x) + N(-x) should come out to 1, give or take rounding.
#[test]
fn reflection_symmetry_holds() {
    for &(x, _) in CDF_REFERENCE {
        if x.abs() > FLUSH_LIMIT {
            continue;
        }
        let sum = norm_cdf(x) + norm_cdf(-x);
        assert!((sum - 1.0).abs() < 1e-15, "N({x}) + N(-{x}) = {sum}");
    }
}

/// Sweeping back through a single norm_cdf node should get us the density.
#[test]
fn norm_cdf_backward_sweep_yields_density() {
    for &(x, expected_density) in PDF_REFERENCE {
        tape::reset();

        let v = Dual::var(x);
        let n = v.norm_cdf();
        let grad = n.backward();

        assert!(
            (n.value - norm_cdf(x)).abs() < 1e-16,
            "value mismatch at {x}"
        );
        let err = relative_error(grad.wrt(v), expected_density);
        assert!(
            err < 1e-15,
            "dN/dx at {x} = {} vs {expected_density}, relative error {err:e}",
            grad.wrt(v)
        );
    }
}

/// The chain rule needs to compose across tape nodes properly — for
/// z = N(a*x), we should get dz/dx = a * phi(a*x).
#[test]
fn norm_cdf_chains_through_an_inner_operation() {
    const A: f64 = 3.0;
    const X: f64 = 0.4;

    tape::reset();

    let x = Dual::var(X);
    let scaled = Dual::unary(A * X, x, A);
    let z = scaled.norm_cdf();
    let grad = z.backward();

    let expected = A * norm_pdf(A * X);
    let err = relative_error(grad.wrt(x), expected);

    assert!((z.value - norm_cdf(A * X)).abs() < 1e-16);
    assert!(
        err < 1e-15,
        "dz/dx = {} vs {expected}, relative error {err:e}",
        grad.wrt(x)
    );
}

/// Since the density is literally the derivative, a central difference of
/// the CDF should agree with the swept adjoint, at least to
/// finite-difference accuracy. This is the test that would catch a partial
/// that's wrong in a way that's consistent across both the kernel and the
/// tape — nothing else here would notice that.
#[test]
fn swept_adjoint_agrees_with_central_difference() {
    const H: f64 = 1e-5;

    for x in [-2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5] {
        tape::reset();

        let v = Dual::var(x);
        let grad = v.norm_cdf().backward();

        let numerical = (norm_cdf(x + H) - norm_cdf(x - H)) / (2.0 * H);
        let err = relative_error(grad.wrt(v), numerical);

        assert!(
            err < 1e-9,
            "at x = {x}: adjoint {} vs central difference {numerical}, relative error {err:e}",
            grad.wrt(v)
        );
    }
}
