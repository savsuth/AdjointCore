//! The pricer is generic over the payoff: a European call, a European put,
//! and an arithmetic-average Asian call all run through the same path
//! simulation and the same adjoint sweep.
//!
//! The put has a closed form, so it gets the same check the call does
//! (Black-Scholes within 5 standard errors) plus put-call parity on common
//! paths. The Asian call has no closed form, so each pathwise Greek is checked
//! against central-difference bump-and-reval on common random numbers, using
//! the tape-free `f64` pricer as the oracle.

use adjoint_core::{
    AsianCall, EuropeanCall, EuropeanPut, MonteCarloResult, Payoff, black_scholes,
    black_scholes_put, monte_carlo_greeks, monte_carlo_value,
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

fn value_of<P: Payoff>(payoff: &P, spot: f64, rate: f64, vol: f64, time: f64) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED);
    monte_carlo_value(payoff, spot, rate, vol, time, NUM_PATHS, &mut rng)
}

// ---- Payoff values on a fixed path ----

#[test]
fn payoffs_evaluate_on_a_plain_path() {
    let call = EuropeanCall { strike: 100.0 };
    let put = EuropeanPut { strike: 100.0 };
    let asian = AsianCall {
        strike: 100.0,
        fixings: 4,
    };

    assert_eq!(call.fixings(), 1);
    assert_eq!(put.fixings(), 1);
    assert_eq!(asian.fixings(), 4);

    assert_eq!(call.value(&[110.0]), 10.0);
    assert_eq!(call.value(&[90.0]), 0.0);
    assert_eq!(put.value(&[90.0]), 10.0);
    assert_eq!(put.value(&[110.0]), 0.0);
    // Average of the four fixings is 105.
    assert_eq!(asian.value(&[90.0, 100.0, 110.0, 120.0]), 5.0);
    assert_eq!(asian.value(&[80.0, 90.0, 100.0, 110.0]), 0.0);
}

// ---- European put ----

/// Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=0.1,
/// vol=0.2, T=0.5 gives a put price of 0.81.
#[test]
fn black_scholes_put_matches_textbook_value() {
    let g = black_scholes_put(42.0, 40.0, 0.1, 0.2, 0.5);
    assert!((g.price - 0.81).abs() < 0.01, "price = {}", g.price);
}

/// The two closed forms have to satisfy parity exactly: C - P = S - K e^{-rT},
/// Delta_C - Delta_P = 1, equal Vega, Rho_C - Rho_P = K T e^{-rT}.
#[test]
fn black_scholes_call_and_put_satisfy_parity() {
    let (spot, strike, rate, vol, time) = (100.0, 95.0, 0.05, 0.25, 0.75);
    let c = black_scholes(spot, strike, rate, vol, time);
    let p = black_scholes_put(spot, strike, rate, vol, time);
    let discounted_strike = strike * (-rate * time).exp();

    assert!((c.price - p.price - (spot - discounted_strike)).abs() < 1e-12);
    assert!((c.delta - p.delta - 1.0).abs() < 1e-12);
    assert!((c.vega - p.vega).abs() < 1e-12);
    assert!((c.rho - p.rho - time * discounted_strike).abs() < 1e-12);
}

#[test]
fn monte_carlo_put_matches_black_scholes() {
    let (rate, vol, time) = (0.05, 0.25, 0.75);

    for (spot, strike) in [(100.0, 100.0), (70.0, 100.0), (130.0, 100.0)] {
        let analytical = black_scholes_put(spot, strike, rate, vol, time);
        let result = greeks_of(&EuropeanPut { strike }, spot, rate, vol, time);
        let mc = result.greeks;
        let se = result.std_error;

        assert_within_mc_tolerance("put price", mc.price, analytical.price, se.price);
        assert_within_mc_tolerance("put delta", mc.delta, analytical.delta, se.delta);
        assert_within_mc_tolerance("put vega", mc.vega, analytical.vega, se.vega);
        assert_within_mc_tolerance("put rho", mc.rho, analytical.rho, se.rho);
    }
}

/// Put-call parity on common paths. Path by path, call minus put is the
/// discounted forward disc * (S_T - K), so each difference estimator has a
/// closed-form variance and gets its own exact standard error:
///
/// - price: disc * (S_T - K), variance S^2 (e^{vol^2 T} - 1)
/// - Delta: disc * S_T / S, mean 1, variance e^{vol^2 T} - 1
/// - Vega: disc * S_T * (sqrt(T) z - vol T), mean 0, variance
///   S^2 T e^{vol^2 T} (1 + vol^2 T)
/// - Rho: K T disc on every path, so the difference is exact, not statistical.
#[test]
fn monte_carlo_put_call_parity_on_common_paths() {
    let (spot, strike, rate, vol, time) = (100.0, 95.0, 0.05, 0.25, 0.75);
    let call = greeks_of(&EuropeanCall { strike }, spot, rate, vol, time).greeks;
    let put = greeks_of(&EuropeanPut { strike }, spot, rate, vol, time).greeks;

    let n = NUM_PATHS as f64;
    let total_var = vol * vol * time;
    let discounted_strike = strike * (-rate * time).exp();

    let price_se = spot * ((total_var.exp() - 1.0) / n).sqrt();
    let delta_se = ((total_var.exp() - 1.0) / n).sqrt();
    let vega_se = spot * (time * total_var.exp() * (1.0 + total_var) / n).sqrt();

    assert_within_mc_tolerance(
        "C - P",
        call.price - put.price,
        spot - discounted_strike,
        price_se,
    );
    assert_within_mc_tolerance("Delta_C - Delta_P", call.delta - put.delta, 1.0, delta_se);
    assert_within_mc_tolerance("Vega_C - Vega_P", call.vega - put.vega, 0.0, vega_se);

    let rho_gap = call.rho - put.rho - time * discounted_strike;
    assert!(
        rho_gap.abs() < 1e-9 * time * discounted_strike,
        "Rho_C - Rho_P should be K T e^(-rT) on every path; off by {rho_gap:e}"
    );
}

// ---- Arithmetic Asian call ----

/// With one fixing at T, the Asian call is the European call, path for path.
#[test]
fn asian_with_one_fixing_is_the_european_call() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let asian = greeks_of(&AsianCall { strike, fixings: 1 }, spot, rate, vol, time);
    let european = greeks_of(&EuropeanCall { strike }, spot, rate, vol, time);

    for (name, a, e) in [
        ("price", asian.greeks.price, european.greeks.price),
        ("delta", asian.greeks.delta, european.greeks.delta),
        ("vega", asian.greeks.vega, european.greeks.vega),
        ("rho", asian.greeks.rho, european.greeks.rho),
        ("price se", asian.std_error.price, european.std_error.price),
        ("vega se", asian.std_error.vega, european.std_error.vega),
    ] {
        assert_eq!(a, e, "{name}");
    }
}

/// The tape-free pricer has to see exactly the paths the adjoint pricer sees,
/// or it is no use as a common-random-numbers oracle.
#[test]
fn tape_free_pricer_reproduces_the_adjoint_price_exactly() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);
    let asian = AsianCall {
        strike: 100.0,
        fixings: 12,
    };

    let adjoint = greeks_of(&asian, spot, rate, vol, time).greeks.price;
    let plain = value_of(&asian, spot, rate, vol, time);
    assert_eq!(adjoint, plain);
}

/// Each pathwise Greek of a 12-fixing Asian call against central-difference
/// bump-and-reval on the same seed. Common random numbers make the finite
/// difference path-by-path, so the two estimators differ only by O(h^2)
/// truncation and the few paths whose average crosses the strike inside the
/// bump; the tolerance is still the pathwise estimator's own standard error.
#[test]
fn asian_greeks_match_central_difference_bump_and_reval() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);

    for strike in [90.0, 100.0, 110.0] {
        let asian = AsianCall {
            strike,
            fixings: 12,
        };
        let result = greeks_of(&asian, spot, rate, vol, time);
        let (mc, se) = (result.greeks, result.std_error);

        let h_spot = 1e-4 * spot;
        let h_vol = 1e-4;
        let h_rate = 1e-4;

        let delta_fd = (value_of(&asian, spot + h_spot, rate, vol, time)
            - value_of(&asian, spot - h_spot, rate, vol, time))
            / (2.0 * h_spot);
        let vega_fd = (value_of(&asian, spot, rate, vol + h_vol, time)
            - value_of(&asian, spot, rate, vol - h_vol, time))
            / (2.0 * h_vol);
        let rho_fd = (value_of(&asian, spot, rate + h_rate, vol, time)
            - value_of(&asian, spot, rate - h_rate, vol, time))
            / (2.0 * h_rate);

        assert_within_mc_tolerance("asian delta", mc.delta, delta_fd, se.delta);
        assert_within_mc_tolerance("asian vega", mc.vega, vega_fd, se.vega);
        assert_within_mc_tolerance("asian rho", mc.rho, rho_fd, se.rho);
    }
}
