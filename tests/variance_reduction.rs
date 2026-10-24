//! Variance reduction: antithetic pairs and a geometric-average control
//! variate, applied to the price and to every pathwise Greek.
//!
//! - Antithetic: each path is paired with its mirror (every normal negated),
//!   and the pair's average is one sample. `num_paths` counts simulated
//!   paths, so the comparison with the plain estimator is at equal path
//!   count.
//! - Control variate: the geometric-average Asian call on the same path,
//!   with coefficient 1. Its expectation has a closed form under the
//!   piecewise-constant vol curve, and that closed form is recorded on the
//!   tape so its exact Greeks are added back as well. With a fixed
//!   coefficient the controlled estimator stays unbiased for the price and
//!   every Greek.

use adjoint_core::{
    AsianCall, EuropeanCall, GeometricAsianCall, Market, VarianceReduction, black_scholes,
    geometric_asian_call, monte_carlo_greeks_bucketed, monte_carlo_greeks_reduced,
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

fn assert_close(name: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "{name}: got {actual}, expected {expected}"
    );
}

const PLAIN: VarianceReduction = VarianceReduction {
    antithetic: false,
    control_variate: false,
};
const ANTITHETIC: VarianceReduction = VarianceReduction {
    antithetic: true,
    control_variate: false,
};
const CONTROL: VarianceReduction = VarianceReduction {
    antithetic: false,
    control_variate: true,
};
const BOTH: VarianceReduction = VarianceReduction {
    antithetic: true,
    control_variate: true,
};

/// With nothing switched on, the reduced pricer is the plain pricer.
#[test]
fn no_reduction_is_the_plain_pricer_exactly() {
    let asian = AsianCall {
        strike: 100.0,
        fixings: 12,
    };
    let vols = [0.18, 0.22, 0.20, 0.25, 0.15];
    let market = Market {
        spot: 100.0,
        rate: 0.05,
        vols: &vols,
        time: 1.0,
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    let plain = monte_carlo_greeks_bucketed(&asian, 100.0, 0.05, &vols, 1.0, 10_000, &mut rng);
    let mut rng = StdRng::seed_from_u64(SEED);
    let reduced = monte_carlo_greeks_reduced(&asian, &market, 10_000, &mut rng, PLAIN);
    assert_eq!(reduced, plain);
}

/// With one fixing the geometric average is the terminal spot, so the
/// closed form has to be Black-Scholes at the integrated-variance vol, with
/// each bucket vega the chain rule through it.
#[test]
fn geometric_closed_form_with_one_fixing_is_black_scholes() {
    let (spot, strike, rate, time) = (100.0, 95.0, 0.05, 1.5);
    let vols = [0.15, 0.25, 0.20, 0.30];
    let market = Market {
        spot,
        rate,
        vols: &vols,
        time,
    };

    let bucket = time / vols.len() as f64;
    let variance: f64 = vols.iter().map(|v| v * v * bucket).sum();
    let vol_eff = (variance / time).sqrt();
    let bs = black_scholes(spot, strike, rate, vol_eff, time);

    let g = geometric_asian_call(&GeometricAsianCall { strike, fixings: 1 }, &market);
    assert_close("price", g.price, bs.price);
    assert_close("delta", g.delta, bs.delta);
    assert_close("rho", g.rho, bs.rho);
    for (j, &v) in vols.iter().enumerate() {
        assert_close(
            &format!("bucket {j} vega"),
            g.vegas[j],
            bs.vega * v * bucket / (vol_eff * time),
        );
    }
}

/// The closed form for 12 fixings against a plain Monte Carlo of the
/// geometric payoff, flat and on a curve whose edges fall between fixings.
#[test]
fn geometric_closed_form_matches_monte_carlo() {
    let geometric = GeometricAsianCall {
        strike: 100.0,
        fixings: 12,
    };

    for vols in [vec![0.2], vec![0.18, 0.22, 0.20, 0.25, 0.15]] {
        let market = Market {
            spot: 100.0,
            rate: 0.05,
            vols: &vols,
            time: 1.0,
        };
        let exact = geometric_asian_call(&geometric, &market);

        let mut rng = StdRng::seed_from_u64(SEED);
        let mc =
            monte_carlo_greeks_bucketed(&geometric, 100.0, 0.05, &vols, 1.0, NUM_PATHS, &mut rng);
        let (g, se) = (&mc.greeks, &mc.std_error);

        assert_within_mc_tolerance("price", g.price, exact.price, se.price);
        assert_within_mc_tolerance("delta", g.delta, exact.delta, se.delta);
        assert_within_mc_tolerance("rho", g.rho, exact.rho, se.rho);
        for j in 0..vols.len() {
            assert_within_mc_tolerance(
                &format!("bucket {j} vega"),
                g.vegas[j],
                exact.vegas[j],
                se.vegas[j],
            );
        }
    }
}

/// For a European call the geometric control with one fixing is the call
/// itself, so payoff minus control is zero on every path and the controlled
/// estimator is the closed form, with no sampling error left.
#[test]
fn control_variate_on_a_european_call_is_exact() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let market = Market {
        spot,
        rate,
        vols: &[vol],
        time,
    };
    let bs = black_scholes(spot, strike, rate, vol, time);

    let mut rng = StdRng::seed_from_u64(SEED);
    let r = monte_carlo_greeks_reduced(&EuropeanCall { strike }, &market, 1_000, &mut rng, CONTROL);

    assert_close("price", r.greeks.price, bs.price);
    assert_close("delta", r.greeks.delta, bs.delta);
    assert_close("rho", r.greeks.rho, bs.rho);
    assert_close("vega", r.greeks.vegas[0], bs.vega);
    assert!(r.std_error.price < 1e-10, "price SE {}", r.std_error.price);
    assert!(
        r.std_error.vegas[0] < 1e-10,
        "vega SE {}",
        r.std_error.vegas[0]
    );
}

#[test]
fn antithetic_european_matches_black_scholes() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let market = Market {
        spot,
        rate,
        vols: &[vol],
        time,
    };
    let bs = black_scholes(spot, strike, rate, vol, time);

    let mut rng = StdRng::seed_from_u64(SEED);
    let r = monte_carlo_greeks_reduced(
        &EuropeanCall { strike },
        &market,
        NUM_PATHS,
        &mut rng,
        ANTITHETIC,
    );
    let (g, se) = (&r.greeks, &r.std_error);

    assert_within_mc_tolerance("price", g.price, bs.price, se.price);
    assert_within_mc_tolerance("delta", g.delta, bs.delta, se.delta);
    assert_within_mc_tolerance("rho", g.rho, bs.rho, se.rho);
    assert_within_mc_tolerance("vega", g.vegas[0], bs.vega, se.vegas[0]);
}

/// Every reduction is unbiased: on an independent seed it agrees with the
/// plain estimator within their combined standard error, for the price and
/// every Greek. Each one must also shrink the price's standard error at the
/// same number of simulated paths, which is the point of using it.
#[test]
fn reduced_asian_estimates_agree_with_plain_and_shrink_the_price_error() {
    let asian = AsianCall {
        strike: 100.0,
        fixings: 12,
    };
    let vols = [0.18, 0.22, 0.20, 0.25, 0.15];
    let market = Market {
        spot: 100.0,
        rate: 0.05,
        vols: &vols,
        time: 1.0,
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    let plain = monte_carlo_greeks_reduced(&asian, &market, NUM_PATHS, &mut rng, PLAIN);

    for (name, reduction) in [
        ("antithetic", ANTITHETIC),
        ("control", CONTROL),
        ("both", BOTH),
    ] {
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        let reduced = monte_carlo_greeks_reduced(&asian, &market, NUM_PATHS, &mut rng, reduction);
        let (a, sa) = (&reduced.greeks, &reduced.std_error);
        let (b, sb) = (&plain.greeks, &plain.std_error);
        let combined = |x: f64, y: f64| (x * x + y * y).sqrt();

        assert_within_mc_tolerance(
            &format!("{name} price"),
            a.price,
            b.price,
            combined(sa.price, sb.price),
        );
        assert_within_mc_tolerance(
            &format!("{name} delta"),
            a.delta,
            b.delta,
            combined(sa.delta, sb.delta),
        );
        assert_within_mc_tolerance(
            &format!("{name} rho"),
            a.rho,
            b.rho,
            combined(sa.rho, sb.rho),
        );
        for j in 0..vols.len() {
            assert_within_mc_tolerance(
                &format!("{name} bucket {j} vega"),
                a.vegas[j],
                b.vegas[j],
                combined(sa.vegas[j], sb.vegas[j]),
            );
        }
        assert!(
            sa.price < sb.price,
            "{name}: price SE {} not below plain {}",
            sa.price,
            sb.price
        );
    }
}

#[test]
#[should_panic(expected = "even")]
fn antithetic_needs_an_even_path_count() {
    let market = Market {
        spot: 100.0,
        rate: 0.05,
        vols: &[0.2],
        time: 1.0,
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    monte_carlo_greeks_reduced(
        &EuropeanCall { strike: 100.0 },
        &market,
        1_001,
        &mut rng,
        ANTITHETIC,
    );
}
