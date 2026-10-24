//! Variance reduction benchmark: how much each reduction shrinks the
//! variance of the price and of every Greek, and what it costs in time.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example variance_reduction
//! ```
//!
//! Setup: a 12-fixing arithmetic Asian call, S = 100, r = 5%, T = 1, on a
//! 5-bucket vol curve, at strikes 90, 100 and 110. Every method simulates the
//! same 100,000 paths' worth of normals from seed 42 (antithetic: 50,000
//! pairs). For each quantity the variance reduction factor is
//! (plain SE / method SE)^2, the number of times more plain paths it would
//! take to reach the same standard error. For the vegas the smallest and
//! largest factor across the five buckets are shown. Time is the median of
//! `RUNS` runs after one warm-up. Efficiency is the price's factor times
//! (plain time / method time): the gain per unit of compute.

use std::hint::black_box;
use std::time::{Duration, Instant};

use adjoint_core::{
    AsianCall, BucketedResult, Market, VarianceReduction, monte_carlo_greeks_reduced,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const VOLS: [f64; 5] = [0.18, 0.22, 0.20, 0.25, 0.15];
const NUM_PATHS: u64 = 100_000;
const SEED: u64 = 42;
const RUNS: usize = 5;

fn run(
    asian: &AsianCall,
    market: &Market,
    reduction: VarianceReduction,
) -> (Duration, BucketedResult) {
    let price = || {
        let mut rng = StdRng::seed_from_u64(SEED);
        monte_carlo_greeks_reduced(asian, market, NUM_PATHS, &mut rng, reduction)
    };
    black_box(price());
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            black_box(price());
            start.elapsed()
        })
        .collect();
    times.sort();
    (times[RUNS / 2], price())
}

fn main() {
    let market = Market {
        spot: 100.0,
        rate: 0.05,
        vols: &VOLS,
        time: 1.0,
    };
    let methods = [
        ("plain", false, false),
        ("antithetic", true, false),
        ("control", false, true),
        ("both", true, true),
    ];

    println!(
        "Asian call, 12 fixings, 5-bucket vol curve, {NUM_PATHS} simulated paths, seed {SEED}, \
         median of {RUNS} runs"
    );
    println!();
    println!(
        "| K | method | price | price SE | VR price | VR delta | VR rho | VR vegas [min, max] | ms | efficiency |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");

    for strike in [90.0, 100.0, 110.0] {
        let asian = AsianCall {
            strike,
            fixings: 12,
        };
        let mut plain: Option<(Duration, BucketedResult)> = None;

        for (name, antithetic, control_variate) in methods {
            let reduction = VarianceReduction {
                antithetic,
                control_variate,
            };
            let (time, result) = run(&asian, &market, reduction);
            let (base_time, base) = plain.get_or_insert_with(|| (time, result.clone()));
            let (b, r) = (&base.std_error, &result.std_error);
            let vr = |plain_se: f64, se: f64| (plain_se / se).powi(2);

            let vega_vr: Vec<f64> = b
                .vegas
                .iter()
                .zip(&r.vegas)
                .map(|(&p, &m)| vr(p, m))
                .collect();
            let vega_min = vega_vr.iter().cloned().fold(f64::INFINITY, f64::min);
            let vega_max = vega_vr.iter().cloned().fold(0.0, f64::max);
            let ms = time.as_secs_f64() * 1e3;
            let efficiency = vr(b.price, r.price) * base_time.as_secs_f64() / time.as_secs_f64();

            println!(
                "| {strike} | {name} | {:.4} | {:.5} | {:.1} | {:.1} | {:.1} | [{vega_min:.1}, {vega_max:.1}] \
                 | {ms:.1} | {efficiency:.1} |",
                result.greeks.price,
                r.price,
                vr(b.price, r.price),
                vr(b.delta, r.delta),
                vr(b.rho, r.rho),
            );
        }
    }
}
