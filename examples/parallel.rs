//! Parallel scaling benchmark: the adjoint pricer on 1 to 10 rayon threads.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example parallel
//! ```
//!
//! Setup: an arithmetic Asian call with 100 fixings over one year on a
//! 100-bucket vol curve (102 sensitivities), 100,000 paths in chunks of
//! 1,000, seed 42. For each thread count, one warm-up run is discarded and
//! `RUNS` runs are timed on a dedicated pool of that size. The serial pricer
//! (no chunking, no pool) is timed the same way as the reference. Every
//! parallel run is also checked to be bit-identical to the one-thread run.

use std::hint::black_box;
use std::time::{Duration, Instant};

use adjoint_core::{
    AsianCall, PathChunks, monte_carlo_greeks_bucketed, monte_carlo_greeks_parallel,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::ThreadPoolBuilder;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const VOL: f64 = 0.2;
const TIME: f64 = 1.0;
const FIXINGS: usize = 100;
const BUCKETS: usize = 100;
const NUM_PATHS: u64 = 100_000;
const CHUNKS: PathChunks = PathChunks {
    seed: 42,
    chunk_size: 1_000,
};
const RUNS: usize = 5;
const THREADS: [usize; 5] = [1, 2, 4, 8, 10];

/// Median, min, and max of a set of timings, in milliseconds.
fn summary(mut samples: Vec<Duration>) -> (f64, f64, f64) {
    samples.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    (
        ms(samples[samples.len() / 2]),
        ms(samples[0]),
        ms(samples[samples.len() - 1]),
    )
}

fn main() {
    let payoff = AsianCall {
        strike: 100.0,
        fixings: FIXINGS,
    };
    let vols = vec![VOL; BUCKETS];

    println!(
        "Asian call, {FIXINGS} fixings, {BUCKETS} vol buckets ({} sensitivities), \
         {NUM_PATHS} paths in chunks of {}, {RUNS} timed runs after 1 warm-up",
        BUCKETS + 2,
        CHUNKS.chunk_size
    );
    println!();

    let serial = || {
        let mut rng = StdRng::seed_from_u64(CHUNKS.seed);
        monte_carlo_greeks_bucketed(&payoff, SPOT, RATE, &vols, TIME, NUM_PATHS, &mut rng)
    };
    black_box(serial());
    let serial_times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            black_box(serial());
            start.elapsed()
        })
        .collect();
    let (serial_med, serial_min, serial_max) = summary(serial_times);
    println!("serial pricer (no chunks): {serial_med:.1} ms [{serial_min:.1}, {serial_max:.1}]");
    println!();
    println!(
        "| threads | AAD ms (median [min, max]) | speedup vs 1 thread | identical to 1 thread |"
    );
    println!("|---|---|---|---|");

    let mut reference = None;
    let mut one_thread_median = None;
    for &threads in &THREADS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let run = || {
            pool.install(|| {
                monte_carlo_greeks_parallel(&payoff, SPOT, RATE, &vols, TIME, NUM_PATHS, CHUNKS)
            })
        };

        let result = run();
        let reference = reference.get_or_insert_with(|| result.clone());
        let identical = result == *reference;

        let times: Vec<Duration> = (0..RUNS)
            .map(|_| {
                let start = Instant::now();
                black_box(run());
                start.elapsed()
            })
            .collect();
        let (med, min, max) = summary(times);
        let base = *one_thread_median.get_or_insert(med);
        println!(
            "| {threads} | {med:.1} [{min:.1}, {max:.1}] | {:.2} | {identical} |",
            base / med
        );
    }
}
