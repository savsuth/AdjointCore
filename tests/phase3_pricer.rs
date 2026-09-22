//! Phase 3: the Monte Carlo engine's price and pathwise Greeks need to agree
//! with closed-form Black-Scholes.
//!
//! # A note on the tolerance
//!
//! Price/Delta/Vega/Rho are independent pathwise estimators, and they don't
//! have anywhere near the same variance — Delta's estimator is roughly 25x
//! tighter than Vega's here. So a single fixed absolute bound would be wrong
//! for most of them. That's exactly why `monte_carlo_price` reports each
//! quantity's own standard error: the test just asserts agreement to some
//! multiple of that error, which is really what "standard MC variance
//! tolerance" is supposed to mean in the first place.
//!
//! We landed on `SIGMA` = 5 by checking the normalized error
//! (|MC - analytical| / std_error) across five different seeds — it came out
//! somewhere in [0.04, 2.22] every time. At 5 sigma, the false-positive rate
//! works out to roughly 1-in-a-million per quantity, while a genuinely wrong
//! chain rule would blow way past dozens of sigma instead of sitting
//! anywhere near 1.

use adjoint_core::{black_scholes, monte_carlo_price};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SEED: u64 = 42;
const NUM_PATHS: u64 = 100_000;
const SIGMA: f64 = 5.0;

fn assert_within_mc_tolerance(name: &str, mc: f64, analytical: f64, std_error: f64) {
    let error = (mc - analytical).abs();
    let tolerance = SIGMA * std_error;
    assert!(
        error < tolerance,
        "{name}: MC = {mc}, analytical = {analytical}, error = {error:e} \
         exceeds {SIGMA} standard errors ({tolerance:e}, std_error = {std_error:e})"
    );
}

#[test]
fn monte_carlo_matches_black_scholes_atm() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);

    let analytical = black_scholes(spot, strike, rate, vol, time);

    let mut rng = StdRng::seed_from_u64(SEED);
    let result = monte_carlo_price(spot, strike, rate, vol, time, NUM_PATHS, &mut rng);
    let mc = result.greeks;
    let se = result.std_error;

    assert_within_mc_tolerance("price", mc.price, analytical.price, se.price);
    assert_within_mc_tolerance("delta", mc.delta, analytical.delta, se.delta);
    assert_within_mc_tolerance("vega", mc.vega, analytical.vega, se.vega);
    assert_within_mc_tolerance("rho", mc.rho, analytical.rho, se.rho);
}

/// Same check, but away from the money in both directions — the ATM case
/// only exercises the payoff kink on roughly half its paths, so this covers
/// the kink firing rarely (deep OTM) and almost always (deep ITM) instead.
#[test]
fn monte_carlo_matches_black_scholes_away_from_the_money() {
    let (rate, vol, time) = (0.05, 0.25, 0.75);

    for (spot, strike) in [(70.0, 100.0), (130.0, 100.0)] {
        let analytical = black_scholes(spot, strike, rate, vol, time);

        let mut rng = StdRng::seed_from_u64(SEED);
        let result = monte_carlo_price(spot, strike, rate, vol, time, NUM_PATHS, &mut rng);
        let mc = result.greeks;
        let se = result.std_error;

        assert_within_mc_tolerance("price", mc.price, analytical.price, se.price);
        assert_within_mc_tolerance("delta", mc.delta, analytical.delta, se.delta);
        assert_within_mc_tolerance("vega", mc.vega, analytical.vega, se.vega);
        assert_within_mc_tolerance("rho", mc.rho, analytical.rho, se.rho);
    }
}

/// The analytical Black-Scholes formula on its own, checked against a
/// textbook value (Hull) — no Monte Carlo engine involved at all here.
#[test]
fn black_scholes_matches_textbook_values() {
    // From Hull's "Options, Futures, and Other Derivatives": with S=42,
    // K=40, r=0.1, vol=0.2, T=0.5, the call price should come out to 4.76.
    let g = black_scholes(42.0, 40.0, 0.1, 0.2, 0.5);
    assert!((g.price - 4.76).abs() < 0.01, "price = {}", g.price);
}

/// Two independent Monte Carlo runs, from two different seeds, should agree
/// with each other within their combined sampling error. This is what would
/// catch a seed-specific fluke that `monte_carlo_matches_black_scholes_atm`
/// alone could easily miss.
#[test]
fn independent_seeds_agree_with_each_other() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);

    let mut rng_a = StdRng::seed_from_u64(SEED);
    let a = monte_carlo_price(spot, strike, rate, vol, time, NUM_PATHS, &mut rng_a);

    let mut rng_b = StdRng::seed_from_u64(SEED + 1);
    let b = monte_carlo_price(spot, strike, rate, vol, time, NUM_PATHS, &mut rng_b);

    let combined_se = (a.std_error.price.powi(2) + b.std_error.price.powi(2)).sqrt();
    assert_within_mc_tolerance(
        "price (cross-seed)",
        a.greeks.price,
        b.greeks.price,
        combined_se,
    );
}
