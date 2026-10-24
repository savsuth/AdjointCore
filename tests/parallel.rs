//! Paths split into fixed-size chunks and priced in parallel with rayon.
//!
//! Each chunk has its own generator keyed by (seed, chunk index) and its own
//! thread-local tape on whichever worker runs it. The per-chunk statistics
//! are merged in chunk order, so a seed and a chunk size fix the result
//! completely: the thread count must not change a single bit of it.

use adjoint_core::{
    AsianCall, EuropeanCall, PathChunks, black_scholes, monte_carlo_greeks_parallel,
};
use rayon::ThreadPoolBuilder;

const SEED: u64 = 42;
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

/// Runs `op` on a fresh rayon pool of exactly `threads` workers.
fn on_threads<T: Send>(threads: usize, op: impl FnOnce() -> T + Send) -> T {
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    assert_eq!(pool.current_num_threads(), threads);
    pool.install(op)
}

#[test]
fn identical_results_on_one_and_many_threads() {
    let asian = AsianCall {
        strike: 100.0,
        fixings: 12,
    };
    let vols = [0.18, 0.22, 0.20, 0.25, 0.15];
    // 20,500 paths in chunks of 1,000: twenty full chunks and a partial one.
    let price = || {
        monte_carlo_greeks_parallel(
            &asian,
            100.0,
            0.05,
            &vols,
            1.0,
            20_500,
            PathChunks {
                seed: SEED,
                chunk_size: 1_000,
            },
        )
    };
    let one = on_threads(1, price);

    for threads in [2, 4, 8] {
        let many = on_threads(threads, price);
        assert_eq!(many, one, "{threads} threads");
    }
}

/// The parallel estimator is still the right one: a European call on a vol
/// curve against Black-Scholes at the integrated-variance vol, with a path
/// count that leaves a partial last chunk.
#[test]
fn parallel_european_on_a_vol_curve_matches_black_scholes() {
    let (spot, strike, rate, time) = (100.0, 100.0, 0.05, 1.0);
    let vols = [0.15, 0.25, 0.20, 0.30];

    let bucket = time / vols.len() as f64;
    let variance: f64 = vols.iter().map(|v| v * v * bucket).sum();
    let vol_eff = (variance / time).sqrt();
    let analytical = black_scholes(spot, strike, rate, vol_eff, time);

    let call = EuropeanCall { strike };
    let result = on_threads(4, || {
        monte_carlo_greeks_parallel(
            &call,
            spot,
            rate,
            &vols,
            time,
            100_003,
            PathChunks {
                seed: SEED,
                chunk_size: 1_000,
            },
        )
    });
    let (mc, se) = (&result.greeks, &result.std_error);

    assert_within_mc_tolerance("price", mc.price, analytical.price, se.price);
    assert_within_mc_tolerance("delta", mc.delta, analytical.delta, se.delta);
    assert_within_mc_tolerance("rho", mc.rho, analytical.rho, se.rho);
    for (j, &v) in vols.iter().enumerate() {
        let reference = analytical.vega * v * bucket / (vol_eff * time);
        assert_within_mc_tolerance(
            &format!("bucket {j} vega"),
            mc.vegas[j],
            reference,
            se.vegas[j],
        );
    }
}

/// The standard error has to come out of the merged accumulators at the
/// size the full path count implies: two runs with different chunk sizes
/// estimate the same quantity on different paths, so their price standard
/// errors agree to within sampling noise (here 5%), and both prices agree
/// within their combined error.
#[test]
fn chunk_size_changes_the_paths_but_not_the_estimate() {
    let call = EuropeanCall { strike: 100.0 };
    let vols = [0.2];
    let price = |chunk_size| {
        on_threads(4, || {
            monte_carlo_greeks_parallel(
                &call,
                100.0,
                0.05,
                &vols,
                1.0,
                100_000,
                PathChunks {
                    seed: SEED,
                    chunk_size,
                },
            )
        })
    };
    let (a, b) = (price(1_000), price(7_919));

    assert_ne!(
        a.greeks.price, b.greeks.price,
        "different chunk sizes, same paths?"
    );
    let ratio = a.std_error.price / b.std_error.price;
    assert!((ratio - 1.0).abs() < 0.05, "price SE ratio {ratio}");
    let combined = (a.std_error.price.powi(2) + b.std_error.price.powi(2)).sqrt();
    assert_within_mc_tolerance("price", a.greeks.price, b.greeks.price, combined);
}
