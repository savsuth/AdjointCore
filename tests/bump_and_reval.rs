//! Bump-and-reval: central differences of the tape-free pricer, every
//! revaluation re-seeded so all of them see the same paths (common random
//! numbers). It is the baseline the adjoint pricer is benchmarked against and
//! an oracle for it, so it has to agree with the adjoint Greeks on the same
//! seed, and its price has to be the adjoint price exactly.

use adjoint_core::{
    AsianCall, BucketedGreeks, EuropeanCall, Payoff, bump_and_reval, monte_carlo_greeks_bucketed,
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
        "{name}: AAD = {mc}, bump-and-reval = {reference}, error = {error:e} \
         exceeds {SIGMA} standard errors ({tolerance:e}, std_error = {std_error:e})"
    );
}

fn check_against_adjoint<P: Payoff>(payoff: &P, spot: f64, rate: f64, vols: &[f64], time: f64) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let adjoint = monte_carlo_greeks_bucketed(payoff, spot, rate, vols, time, NUM_PATHS, &mut rng);
    let bumped: BucketedGreeks = bump_and_reval(payoff, spot, rate, vols, time, NUM_PATHS, SEED);
    let (aad, se) = (&adjoint.greeks, &adjoint.std_error);

    assert_eq!(bumped.price, aad.price, "price");
    assert_eq!(bumped.vegas.len(), vols.len());
    assert_within_mc_tolerance("delta", aad.delta, bumped.delta, se.delta);
    assert_within_mc_tolerance("rho", aad.rho, bumped.rho, se.rho);
    for j in 0..vols.len() {
        assert_within_mc_tolerance(
            &format!("bucket {j} vega"),
            aad.vegas[j],
            bumped.vegas[j],
            se.vegas[j],
        );
    }
}

#[test]
fn bump_and_reval_matches_the_adjoint_greeks_for_a_european_call() {
    check_against_adjoint(
        &EuropeanCall { strike: 100.0 },
        100.0,
        0.05,
        &[0.15, 0.25, 0.20, 0.30],
        1.0,
    );
}

#[test]
fn bump_and_reval_matches_the_adjoint_greeks_for_an_asian_call() {
    check_against_adjoint(
        &AsianCall {
            strike: 105.0,
            fixings: 12,
        },
        100.0,
        0.03,
        &[0.18, 0.22, 0.20, 0.25, 0.15],
        2.0,
    );
}

/// The same seed gives the same Greeks: nothing in bump-and-reval draws from
/// a source other than the seed.
#[test]
fn bump_and_reval_is_deterministic_for_a_seed() {
    let asian = AsianCall {
        strike: 100.0,
        fixings: 4,
    };
    let vols = [0.2, 0.3];
    let a = bump_and_reval(&asian, 100.0, 0.05, &vols, 1.0, 1_000, SEED);
    let b = bump_and_reval(&asian, 100.0, 0.05, &vols, 1.0, 1_000, SEED);
    assert_eq!(a.price, b.price);
    assert_eq!(a.delta, b.delta);
    assert_eq!(a.rho, b.rho);
    assert_eq!(a.vegas, b.vegas);
}
