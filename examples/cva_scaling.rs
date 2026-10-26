//! CVA scaling benchmark: adjoint sensitivities of netting-set CVA against
//! bump-and-reval, as the number of vol buckets N and of netting sets K
//! grows.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example cva_scaling
//! ```
//!
//! Setup, identical for both methods: S = 100, r = 5%, a flat 20% vol split
//! into N buckets over 5 years, 20 quarterly exposure dates, recovery 40%,
//! `NUM_PATHS` paths, seed 42. Each netting set holds 10 European options
//! (calls and puts, strikes 80 to 120, maturities 1 to 5 years, long and
//! short) and its own hazard rate between 1% and 3%; set s is the same in
//! every book, so a larger book adds sets without changing the others.
//!
//! Each method returns, for every set, its CVA, Delta, Rho, N bucket vegas
//! and K hazard-rate sensitivities. The adjoint pricer records each path once
//! and sweeps it K times; bump-and-reval revalues the whole book
//! 2(N + K) + 5 times. Two sweeps: N with K = 10 sets, then K with N = 10
//! buckets. For every point one warm-up round is discarded, then `RUNS`
//! rounds each time a plain valuation, the adjoint pricer and bump-and-reval
//! back to back. Reported: median and [min, max] wall time, the ratios, and
//! the largest gap between an adjoint sensitivity and its bump-and-reval
//! value in units of the adjoint standard error (sensitivities that are
//! exactly zero, another set's hazard rate, must be exactly zero in both).
//!
//! The 20 exposure dates merge with the N bucket edges: the grid has 20
//! steps for N = 1, 5 and 10, 40 for N = 25 and 60 for N = 50.

use std::hint::black_box;
use std::time::{Duration, Instant};

use adjoint_core::{
    CvaGreeks, CvaResult, Market, NettingSet, OptionKind, Trade, bump_and_reval_cva, cva_value,
    monte_carlo_cva,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const VOL: f64 = 0.2;
const TIME: f64 = 5.0;
const EXPOSURE_DATES: usize = 20;
const TRADES_PER_SET: usize = 10;
const NUM_PATHS: u64 = 2_000;
const SEED: u64 = 42;
const RUNS: usize = 5;
const BUCKETS: [usize; 5] = [1, 5, 10, 25, 50];
const SETS: [usize; 5] = [1, 2, 5, 10, 20];

/// Netting set s: ten trades whose kind, strike, maturity and direction
/// cycle with s and the trade index, and a hazard rate from 1% to 3%.
fn netting_set(s: usize) -> NettingSet {
    let trades = (0..TRADES_PER_SET)
        .map(|t| Trade {
            kind: if (s + t).is_multiple_of(2) {
                OptionKind::Call
            } else {
                OptionKind::Put
            },
            strike: 80.0 + 5.0 * ((7 * s + 3 * t) % 9) as f64,
            maturity: 1.0 + ((s + t) % 5) as f64,
            notional: if (s + 2 * t).is_multiple_of(3) { -1.0 } else { 1.0 },
        })
        .collect();
    NettingSet {
        trades,
        hazard_rate: 0.01 + 0.002 * (s % 11) as f64,
        recovery: 0.4,
    }
}

fn time<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let start = Instant::now();
    let out = black_box(f());
    (start.elapsed(), out)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Median, min, and max of a set of timings, in milliseconds.
fn summary(mut samples: Vec<Duration>) -> (f64, f64, f64) {
    samples.sort();
    (
        ms(samples[samples.len() / 2]),
        ms(samples[0]),
        ms(samples[samples.len() - 1]),
    )
}

/// Largest |adjoint - bump| / SE over every set's sensitivities. A
/// sensitivity with standard error 0 must agree exactly.
fn max_gap(adjoint: &CvaResult, bumped: &[CvaGreeks]) -> f64 {
    let gap = |a: f64, b: f64, se: f64| {
        if se > 0.0 {
            ((a - b) / se).abs()
        } else if a == b {
            0.0
        } else {
            f64::INFINITY
        }
    };
    let mut worst: f64 = 0.0;
    for (set, fd) in adjoint.sets.iter().zip(bumped) {
        let (g, se) = (&set.greeks, &set.std_error);
        worst = worst.max(gap(g.delta, fd.delta, se.delta));
        worst = worst.max(gap(g.rho, fd.rho, se.rho));
        for k in 0..g.vegas.len() {
            worst = worst.max(gap(g.vegas[k], fd.vegas[k], se.vegas[k]));
        }
        for l in 0..g.hazards.len() {
            worst = worst.max(gap(g.hazards[l], fd.hazards[l], se.hazards[l]));
        }
    }
    worst
}

/// One point of a sweep: K sets, N buckets.
fn point(label: usize, sets: usize, buckets: usize) {
    let book: Vec<NettingSet> = (0..sets).map(netting_set).collect();
    let vols = vec![VOL; buckets];
    let market = Market {
        spot: SPOT,
        rate: RATE,
        vols: &vols,
        time: TIME,
    };
    let valuation = || {
        let mut rng = StdRng::seed_from_u64(SEED);
        cva_value(&book, &market, EXPOSURE_DATES, NUM_PATHS, &mut rng)
    };
    let adjoint = || {
        let mut rng = StdRng::seed_from_u64(SEED);
        monte_carlo_cva(&book, &market, EXPOSURE_DATES, NUM_PATHS, &mut rng)
    };
    let bump = || bump_and_reval_cva(&book, &market, EXPOSURE_DATES, NUM_PATHS, SEED);

    // Warm-up round, discarded.
    black_box(valuation());
    black_box(adjoint());
    black_box(bump());

    let (mut t_val, mut t_aad, mut t_bump) = (Vec::new(), Vec::new(), Vec::new());
    let mut last = None;
    for _ in 0..RUNS {
        t_val.push(time(valuation).0);
        let (d, aad) = time(adjoint);
        t_aad.push(d);
        let (d, bumped) = time(bump);
        t_bump.push(d);
        last = Some((aad, bumped));
    }
    let (aad, bumped) = last.expect("RUNS > 0");
    let gap = max_gap(&aad, &bumped);

    let (val_med, _, _) = summary(t_val);
    let (aad_med, aad_min, aad_max) = summary(t_aad);
    let (bump_med, bump_min, bump_max) = summary(t_bump);
    println!(
        "| {label} | {} x {sets} | {} | {val_med:.0} | {aad_med:.0} [{aad_min:.0}, {aad_max:.0}] \
         | {bump_med:.0} [{bump_min:.0}, {bump_max:.0}] | {:.3} | {:.2} | {:.1} | {:.4} | {gap:.3} |",
        buckets + 2 + sets,
        2 * (buckets + sets) + 5,
        aad_med / bump_med,
        aad_med / val_med,
        bump_med / val_med,
        aad.total.cva,
    );
}

fn header(swept: &str) {
    println!(
        "| {swept} | sens. per set x sets | valuations | valuation ms | AAD ms (median [min, max]) \
         | bump ms (median [min, max]) | AAD / bump | AAD / valuation | bump / valuation \
         | total CVA | max gap (SE) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
}

fn main() {
    println!(
        "Netting-set CVA, {TRADES_PER_SET} trades per set, {EXPOSURE_DATES} exposure dates over \
         {TIME} years, {NUM_PATHS} paths, seed {SEED}, {RUNS} timed runs per method after 1 warm-up"
    );
    println!();
    println!("Vol buckets N, with K = 10 netting sets:");
    println!();
    header("N");
    for n in BUCKETS {
        point(n, 10, n);
    }
    println!();
    println!("Netting sets K, with N = 10 vol buckets:");
    println!();
    header("K");
    for k in SETS {
        point(k, k, 10);
    }
}
