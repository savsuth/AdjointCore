//! A piecewise-constant volatility curve: N equal buckets over [0, T], each
//! bucket's volatility its own tape input, so one backward sweep per path
//! returns the price, Delta, Rho, and all N bucket vegas.
//!
//! References:
//! - N = 1 is the flat-vol pricer, exactly.
//! - Equal bucket vols: the bucket vegas sum to the flat Vega.
//! - A European call depends on the curve only through the integrated
//!   variance V = sum(vol_j^2) T / N, so Black-Scholes at vol_eff = sqrt(V / T)
//!   gives the price, Delta, and Rho, and each bucket vega is
//!   Vega_BS * vol_j (T / N) / (vol_eff T).
//! - Each bucket vega against central-difference bump-and-reval on common
//!   random numbers, including an Asian whose fixings fall between bucket
//!   edges.

use adjoint_core::{
    AsianCall, EuropeanCall, Greeks, Payoff, black_scholes, monte_carlo_greeks,
    monte_carlo_greeks_bucketed, monte_carlo_value_bucketed,
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

/// Black-Scholes for a call under a piecewise-constant vol curve with equal
/// buckets: the flat formula at the integrated-variance vol, plus the chain
/// rule through vol_eff for each bucket vega.
fn black_scholes_buckets(
    spot: f64,
    strike: f64,
    rate: f64,
    vols: &[f64],
    time: f64,
) -> (Greeks, Vec<f64>) {
    let bucket = time / vols.len() as f64;
    let variance: f64 = vols.iter().map(|v| v * v * bucket).sum();
    let vol_eff = (variance / time).sqrt();
    let flat = black_scholes(spot, strike, rate, vol_eff, time);
    let vegas = vols
        .iter()
        .map(|v| flat.vega * v * bucket / (vol_eff * time))
        .collect();
    (flat, vegas)
}

/// With one bucket the curve is the flat vol, and the pricer takes the
/// identical path arithmetic, so every number matches bit for bit.
#[test]
fn one_bucket_reproduces_the_flat_pricer_exactly() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);

    fn check<P: Payoff>(payoff: &P, spot: f64, rate: f64, vol: f64, time: f64) {
        let mut rng = StdRng::seed_from_u64(SEED);
        let flat = monte_carlo_greeks(payoff, spot, rate, vol, time, NUM_PATHS, &mut rng);
        let mut rng = StdRng::seed_from_u64(SEED);
        let curve =
            monte_carlo_greeks_bucketed(payoff, spot, rate, &[vol], time, NUM_PATHS, &mut rng);

        assert_eq!(curve.greeks.vegas.len(), 1);
        assert_eq!(curve.greeks.price, flat.greeks.price);
        assert_eq!(curve.greeks.delta, flat.greeks.delta);
        assert_eq!(curve.greeks.rho, flat.greeks.rho);
        assert_eq!(curve.greeks.vegas[0], flat.greeks.vega);
        assert_eq!(curve.std_error.price, flat.std_error.price);
        assert_eq!(curve.std_error.vegas[0], flat.std_error.vega);
    }

    check(&EuropeanCall { strike: 100.0 }, spot, rate, vol, time);
    check(
        &AsianCall {
            strike: 100.0,
            fixings: 12,
        },
        spot,
        rate,
        vol,
        time,
    );
}

/// An N-fixing Asian on N equal buckets steps on exactly the flat pricer's
/// grid, so both see the same paths. By the chain rule, the bucket vegas
/// then sum to the flat Vega on every path, and so do their means.
#[test]
fn equal_bucket_vegas_sum_to_the_flat_vega_on_common_paths() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);
    let buckets = 12;
    let asian = AsianCall {
        strike: 100.0,
        fixings: buckets,
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    let flat = monte_carlo_greeks(&asian, spot, rate, vol, time, NUM_PATHS, &mut rng);
    let mut rng = StdRng::seed_from_u64(SEED);
    let curve = monte_carlo_greeks_bucketed(
        &asian,
        spot,
        rate,
        &vec![vol; buckets],
        time,
        NUM_PATHS,
        &mut rng,
    );

    let sum: f64 = curve.greeks.vegas.iter().sum();
    let gap = sum - flat.greeks.vega;
    assert!(
        gap.abs() < 1e-10 * flat.greeks.vega,
        "sum of bucket vegas {sum} vs flat Vega {}, off by {gap:e}",
        flat.greeks.vega
    );
    assert!((curve.greeks.price - flat.greeks.price).abs() < 1e-12 * flat.greeks.price);
}

/// The European call on ten equal buckets against the flat closed form. The
/// curve's paths take ten steps, so this is a statistical check; the sum of
/// the bucket standard errors bounds the standard error of the sum from
/// above, so the tolerance can only be too loose, never too tight.
#[test]
fn equal_bucket_vegas_sum_to_the_black_scholes_vega() {
    let (spot, strike, rate, vol, time) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let analytical = black_scholes(spot, strike, rate, vol, time);

    let mut rng = StdRng::seed_from_u64(SEED);
    let curve = monte_carlo_greeks_bucketed(
        &EuropeanCall { strike },
        spot,
        rate,
        &[vol; 10],
        time,
        NUM_PATHS,
        &mut rng,
    );

    let sum: f64 = curve.greeks.vegas.iter().sum();
    let se_bound: f64 = curve.std_error.vegas.iter().sum();
    assert_within_mc_tolerance("sum of bucket vegas", sum, analytical.vega, se_bound);
    assert_within_mc_tolerance(
        "price",
        curve.greeks.price,
        analytical.price,
        curve.std_error.price,
    );
}

#[test]
fn european_on_a_vol_curve_matches_integrated_variance_black_scholes() {
    let (rate, time) = (0.05, 1.0);
    let vols = [0.15, 0.25, 0.20, 0.30];

    for (spot, strike) in [(100.0, 100.0), (80.0, 100.0), (120.0, 100.0)] {
        let (analytical, analytical_vegas) = black_scholes_buckets(spot, strike, rate, &vols, time);

        let mut rng = StdRng::seed_from_u64(SEED);
        let curve = monte_carlo_greeks_bucketed(
            &EuropeanCall { strike },
            spot,
            rate,
            &vols,
            time,
            NUM_PATHS,
            &mut rng,
        );
        let (mc, se) = (&curve.greeks, &curve.std_error);

        assert_within_mc_tolerance("price", mc.price, analytical.price, se.price);
        assert_within_mc_tolerance("delta", mc.delta, analytical.delta, se.delta);
        assert_within_mc_tolerance("rho", mc.rho, analytical.rho, se.rho);
        assert_eq!(mc.vegas.len(), vols.len());
        for (j, (&v, &reference)) in mc.vegas.iter().zip(&analytical_vegas).enumerate() {
            assert_within_mc_tolerance(&format!("bucket {j} vega"), v, reference, se.vegas[j]);
        }
    }
}

/// Each bucket vega against central-difference bump-and-reval of that one
/// bucket's vol on the same seed. The 12-fixing Asian on 5 buckets puts
/// fixings between bucket edges, so this also exercises the merged time grid.
#[test]
fn bucket_vegas_match_central_difference_bump_and_reval() {
    let (spot, rate, time) = (100.0, 0.05, 1.0);
    let h = 1e-4;

    fn value<P: Payoff>(payoff: &P, spot: f64, rate: f64, vols: &[f64], time: f64) -> f64 {
        let mut rng = StdRng::seed_from_u64(SEED);
        monte_carlo_value_bucketed(payoff, spot, rate, vols, time, NUM_PATHS, &mut rng)
    }

    fn check<P: Payoff>(payoff: &P, spot: f64, rate: f64, vols: &[f64], time: f64, h: f64) {
        let mut rng = StdRng::seed_from_u64(SEED);
        let curve =
            monte_carlo_greeks_bucketed(payoff, spot, rate, vols, time, NUM_PATHS, &mut rng);

        // The tape-free revaluation must see exactly the adjoint pricer's paths.
        assert_eq!(value(payoff, spot, rate, vols, time), curve.greeks.price);

        for j in 0..vols.len() {
            let mut up = vols.to_vec();
            let mut down = vols.to_vec();
            up[j] += h;
            down[j] -= h;
            let fd = (value(payoff, spot, rate, &up, time)
                - value(payoff, spot, rate, &down, time))
                / (2.0 * h);
            assert_within_mc_tolerance(
                &format!("bucket {j} vega"),
                curve.greeks.vegas[j],
                fd,
                curve.std_error.vegas[j],
            );
        }
    }

    check(
        &EuropeanCall { strike: 100.0 },
        spot,
        rate,
        &[0.15, 0.25, 0.20, 0.30],
        time,
        h,
    );
    check(
        &AsianCall {
            strike: 100.0,
            fixings: 12,
        },
        spot,
        rate,
        &[0.18, 0.22, 0.20, 0.25, 0.15],
        time,
        h,
    );
}

/// A 12-fixing Asian on 5 equal buckets steps on a merged grid whose
/// fixings fall between bucket edges, so it shares no paths with the flat
/// pricer. The extra grid points must not change the distribution of the
/// fixings, so on independent seeds both prices and the flat Vega must agree
/// within their combined error. The bucket-vega standard errors are summed,
/// which bounds the standard error of their sum from above.
#[test]
fn asian_on_a_merged_grid_matches_the_flat_asian() {
    let (spot, rate, vol, time) = (100.0, 0.05, 0.2, 1.0);
    let asian = AsianCall {
        strike: 100.0,
        fixings: 12,
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    let flat = monte_carlo_greeks(&asian, spot, rate, vol, time, NUM_PATHS, &mut rng);
    let mut rng = StdRng::seed_from_u64(SEED + 1);
    let curve =
        monte_carlo_greeks_bucketed(&asian, spot, rate, &[vol; 5], time, NUM_PATHS, &mut rng);

    let price_se = (flat.std_error.price.powi(2) + curve.std_error.price.powi(2)).sqrt();
    assert_within_mc_tolerance("price", curve.greeks.price, flat.greeks.price, price_se);

    let delta_se = (flat.std_error.delta.powi(2) + curve.std_error.delta.powi(2)).sqrt();
    assert_within_mc_tolerance("delta", curve.greeks.delta, flat.greeks.delta, delta_se);

    let sum: f64 = curve.greeks.vegas.iter().sum();
    let vega_se = flat.std_error.vega + curve.std_error.vegas.iter().sum::<f64>();
    assert_within_mc_tolerance("sum of bucket vegas", sum, flat.greeks.vega, vega_se);
}
