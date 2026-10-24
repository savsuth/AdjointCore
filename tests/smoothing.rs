//! Smoothed digital and barrier payoffs.
//!
//! A digital call and an up-and-out call jump, so their pathwise derivative
//! misses the jump: on the exact indicator the pathwise Delta and Vega of the
//! digital are zero on every path. Replacing each indicator with a centered
//! call spread of width h makes the payoff Lipschitz, so the pathwise
//! estimator becomes unbiased for the smoothed payoff, and biased for the
//! true one by the smoothing alone.
//!
//! References, and what each one is a reference for:
//! - Smoothed digital: the call spread (C(K - h/2) - C(K + h/2)) / h from the
//!   Black-Scholes call, exact for the smoothed payoff's price and Greeks.
//! - Digital: the Black-Scholes cash-or-nothing closed form, exact for the
//!   true price and Greeks.
//! - Smoothed barrier: central-difference bump-and-reval of the same smoothed
//!   payoff on common random numbers, a check that the adjoint Greeks are the
//!   derivatives of the smoothed price.

use adjoint_core::{
    DigitalCall, Dual, EuropeanCall, Greeks, MonteCarloResult, Payoff, Real, UpAndOutCall,
    black_scholes, black_scholes_digital, bump_and_reval, monte_carlo_greeks,
    monte_carlo_greeks_bucketed,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 42;
const NUM_PATHS: u64 = 100_000;
const SIGMA: f64 = 5.0;

fn assert_within_mc_tolerance(name: &str, mc: f64, reference: f64, std_error: f64) {
    let error = (mc - reference).abs();
    let tolerance = SIGMA * std_error;
    assert!(
        error < tolerance,
        "{name}: MC = {mc}, reference = {reference}, error = {error:e} \
         exceeds {SIGMA} standard errors ({tolerance:e}, std_error = {std_error:e})"
    );
}

fn greeks_of<P: Payoff>(payoff: &P, spot: f64, rate: f64, vol: f64, time: f64) -> MonteCarloResult {
    let mut rng = StdRng::seed_from_u64(SEED);
    monte_carlo_greeks(payoff, spot, rate, vol, time, NUM_PATHS, &mut rng)
}

/// Price and Greeks of the call spread (C(K - h/2) - C(K + h/2)) / h: the
/// expectation of the smoothed digital, exactly.
fn call_spread(spot: f64, strike: f64, width: f64, rate: f64, vol: f64, time: f64) -> Greeks {
    let low = black_scholes(spot, strike - 0.5 * width, rate, vol, time);
    let high = black_scholes(spot, strike + 0.5 * width, rate, vol, time);
    Greeks {
        price: (low.price - high.price) / width,
        delta: (low.delta - high.delta) / width,
        vega: (low.vega - high.vega) / width,
        rho: (low.rho - high.rho) / width,
    }
}

// ---- Payoff values on a fixed path ----

#[test]
fn smoothed_payoffs_evaluate_on_a_plain_path() {
    let exact = DigitalCall {
        strike: 100.0,
        smoothing: 0.0,
    };
    let smooth = DigitalCall {
        strike: 100.0,
        smoothing: 2.0,
    };
    assert_eq!(exact.fixings(), 1);
    assert_eq!(exact.value(&[100.5]), 1.0);
    assert_eq!(exact.value(&[99.5]), 0.0);
    // The ramp runs linearly from 0 at K - h/2 to 1 at K + h/2.
    assert_eq!(smooth.value(&[98.0]), 0.0);
    assert_eq!(smooth.value(&[100.0]), 0.5);
    assert_eq!(smooth.value(&[100.5]), 0.75);
    assert_eq!(smooth.value(&[102.0]), 1.0);

    let exact_barrier = UpAndOutCall {
        strike: 100.0,
        barrier: 120.0,
        fixings: 4,
        smoothing: 0.0,
    };
    let smooth_barrier = UpAndOutCall {
        smoothing: 2.0,
        ..exact_barrier
    };
    assert_eq!(exact_barrier.fixings(), 4);
    assert_eq!(exact_barrier.value(&[110.0, 115.0, 119.0, 118.0]), 18.0);
    // Knocked out at the second fixing, though the spot ends below the barrier.
    assert_eq!(exact_barrier.value(&[110.0, 121.0, 119.0, 118.0]), 0.0);
    assert_eq!(exact_barrier.value(&[90.0, 95.0, 100.0, 99.0]), 0.0);
    // Inside the band B -+ h/2 a fixing keeps part of the payoff: 0.75 at 119.5.
    assert_eq!(smooth_barrier.value(&[110.0, 119.5, 115.0, 118.0]), 13.5);
    assert_eq!(smooth_barrier.value(&[110.0, 121.0, 115.0, 118.0]), 0.0);
    assert_eq!(smooth_barrier.value(&[110.0, 115.0, 118.0, 118.0]), 18.0);
}

/// The step 1{x > 0} has derivative 0 wherever it has one, and that is what
/// the tape records: a pathwise sweep sees nothing of the jump.
#[test]
fn step_records_a_zero_partial() {
    assert_eq!(Real::step(0.3_f64), 1.0);
    assert_eq!(Real::step(-0.3_f64), 0.0);

    let x = Dual::var(0.3);
    let y = (x * 2.0).step() * 5.0 + x;
    assert_eq!(y.value, 5.3);
    assert_eq!(y.backward().wrt(x), 1.0);

    let x = Dual::var(-0.3);
    let y = x.step();
    assert_eq!(y.value, 0.0);
    assert_eq!(y.backward().wrt(x), 0.0);
}

// ---- Digital call ----

/// The digital is minus the strike derivative of the call: D = -dC/dK, and
/// so is each of its Greeks. A central difference in strike of the existing
/// Black-Scholes call checks the closed form to O(eps^2): measured, the gap
/// falls 100x from eps = 1e-2 to 1e-3 (worst 1.0e-7 to 1.0e-9, vega at
/// S = 80), so what is left at 1e-3 is truncation, not the closed form.
#[test]
fn digital_closed_form_is_the_strike_derivative_of_the_call() {
    let (rate, vol, time) = (0.05, 0.2, 1.0);
    let eps = 1e-3;
    for (spot, strike) in [(100.0, 100.0), (80.0, 100.0), (125.0, 100.0)] {
        let digital = black_scholes_digital(spot, strike, rate, vol, time);
        let reference = call_spread(spot, strike, 2.0 * eps, rate, vol, time);
        for (name, d, r) in [
            ("price", digital.price, reference.price),
            ("delta", digital.delta, reference.delta),
            ("vega", digital.vega, reference.vega),
            ("rho", digital.rho, reference.rho),
        ] {
            assert!(
                (d - r).abs() < 1e-8,
                "S = {spot}: digital {name} = {d}, -dC/dK = {r}"
            );
        }
    }
}

/// On the smoothed digital the pathwise estimator is unbiased for the call
/// spread it is: price and every Greek within 5 standard errors of the
/// closed-form spread, at every strike and width.
#[test]
fn smoothed_digital_is_unbiased_for_its_call_spread() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);
    for strike in [90.0, 100.0, 110.0] {
        for width in [1.0, 5.0] {
            let digital = DigitalCall {
                strike,
                smoothing: width,
            };
            let result = greeks_of(&digital, spot, rate, vol, time);
            let (mc, se) = (result.greeks, result.std_error);
            let target = call_spread(spot, strike, width, rate, vol, time);
            let name = |q: &str| format!("K = {strike}, h = {width}: {q}");

            assert_within_mc_tolerance(&name("price"), mc.price, target.price, se.price);
            assert_within_mc_tolerance(&name("delta"), mc.delta, target.delta, se.delta);
            assert_within_mc_tolerance(&name("vega"), mc.vega, target.vega, se.vega);
            assert_within_mc_tolerance(&name("rho"), mc.rho, target.rho, se.rho);
        }
    }
}

/// The centered spread averages the digital over strikes K -+ h/2, so the
/// first-order term cancels and the smoothing bias is h^2 / 24 times the
/// second strike derivative: halving the width divides it by 4. A one-sided
/// spread would only halve it.
#[test]
fn smoothing_bias_of_the_digital_is_second_order_in_the_width() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let digital = black_scholes_digital(spot, strike, rate, vol, time);
    let bias = |width: f64| {
        let s = call_spread(spot, strike, width, rate, vol, time);
        [
            s.price - digital.price,
            s.delta - digital.delta,
            s.vega - digital.vega,
            s.rho - digital.rho,
        ]
    };
    let (wide, narrow) = (bias(2.0), bias(1.0));
    for (i, name) in ["price", "delta", "vega", "rho"].iter().enumerate() {
        let (w, n) = (wide[i], narrow[i]);
        let ratio = w / n;
        assert!(
            (3.8..4.2).contains(&ratio),
            "{name}: bias(h = 2) / bias(h = 1) = {ratio} ({w:e} / {n:e})"
        );
    }
}

/// On the exact indicator a small bump never moves a path across the
/// strike, so each path's pathwise Delta and Vega are exactly zero: the mean
/// and the standard error are both 0.0, which means no path had a nonzero
/// one. Rho sees only the discount factor, -T times the price, path by
/// path. The price itself is still unbiased.
#[test]
fn unsmoothed_digital_pathwise_delta_and_vega_are_zero_on_every_path() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let exact = DigitalCall {
        strike,
        smoothing: 0.0,
    };
    let result = greeks_of(&exact, spot, rate, vol, time);
    let (mc, se) = (result.greeks, result.std_error);
    let truth = black_scholes_digital(spot, strike, rate, vol, time);

    assert_within_mc_tolerance("price", mc.price, truth.price, se.price);
    assert_eq!((mc.delta, se.delta), (0.0, 0.0), "delta");
    assert_eq!((mc.vega, se.vega), (0.0, 0.0), "vega");
    assert!(
        (mc.rho + time * mc.price).abs() < 1e-12 * mc.price,
        "rho = {} should be -T * price = {}",
        mc.rho,
        -time * mc.price
    );
    // While the true Delta is far from zero.
    assert!(truth.delta > 0.01, "true delta = {}", truth.delta);
}

// ---- Up-and-out call ----

/// With the barrier out of reach and no smoothing, every fixing's survival
/// weight is exactly 1 with a zero partial, so the up-and-out call is the
/// European call bit for bit.
#[test]
fn unreachable_unsmoothed_barrier_is_the_european_call() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let barrier = UpAndOutCall {
        strike,
        barrier: 1e12,
        fixings: 1,
        smoothing: 0.0,
    };
    let b = greeks_of(&barrier, spot, rate, vol, time);
    let e = greeks_of(&EuropeanCall { strike }, spot, rate, vol, time);
    for (name, x, y) in [
        ("price", b.greeks.price, e.greeks.price),
        ("delta", b.greeks.delta, e.greeks.delta),
        ("vega", b.greeks.vega, e.greeks.vega),
        ("rho", b.greeks.rho, e.greeks.rho),
        ("price se", b.std_error.price, e.std_error.price),
        ("delta se", b.std_error.delta, e.std_error.delta),
    ] {
        assert_eq!(x, y, "{name}");
    }
}

/// The smoothed up-and-out call is Lipschitz in the path, so its adjoint
/// Greeks are the derivatives of its Monte Carlo price: each one matches
/// central-difference bump-and-reval of the same smoothed payoff on common
/// random numbers, bucket vegas included, within 5 standard errors.
#[test]
fn smoothed_barrier_greeks_match_bump_and_reval_of_the_smoothed_payoff() {
    let (spot, rate, time) = (100.0, 0.05, 1.0);
    let vols = [0.18, 0.22, 0.25];
    for width in [1.0, 5.0] {
        let barrier = UpAndOutCall {
            strike: 100.0,
            barrier: 120.0,
            fixings: 12,
            smoothing: width,
        };
        let mut rng = StdRng::seed_from_u64(SEED);
        let adjoint =
            monte_carlo_greeks_bucketed(&barrier, spot, rate, &vols, time, NUM_PATHS, &mut rng);
        let bumped = bump_and_reval(&barrier, spot, rate, &vols, time, NUM_PATHS, SEED);
        let (aad, se) = (&adjoint.greeks, &adjoint.std_error);
        let name = |q: &str| format!("h = {width}: {q}");

        assert_eq!(bumped.price, aad.price, "{}", name("price"));
        assert_within_mc_tolerance(&name("delta"), aad.delta, bumped.delta, se.delta);
        assert_within_mc_tolerance(&name("rho"), aad.rho, bumped.rho, se.rho);
        for j in 0..vols.len() {
            assert_within_mc_tolerance(
                &name(&format!("bucket {j} vega")),
                aad.vegas[j],
                bumped.vegas[j],
                se.vegas[j],
            );
        }
    }
}
