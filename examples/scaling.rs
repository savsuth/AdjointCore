//! Scaling benchmark: adjoint (AAD) Greeks against bump-and-reval as the
//! number of vol buckets N grows.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example scaling
//! ```
//!
//! Setup, identical for both methods: an arithmetic Asian call with 100
//! fixings over one year, the same path count and seed, and a flat 20% vol
//! split into N buckets. Every N in the sweep divides 100, so the simulation
//! grid is the same 100 steps for every N. The cost of one valuation does
//! not move with N, and any growth in a column comes from the method itself.
//!
//! Each method returns N + 2 sensitivities (Delta, Rho, N bucket vegas).
//! For every N, one warm-up round is discarded, then `RUNS` rounds are timed.
//! Each round times a plain valuation, the adjoint pricer, and bump-and-reval
//! back to back, so a slow stretch of the machine hits all three alike.
//! Reported: median and [min, max] wall time per method, and the ratios. The
//! last column checks both methods computed the same thing: the largest gap
//! between an adjoint Greek and its bump-and-reval value, in units of the
//! adjoint estimator's standard error. The tape column is measured, not
//! derived: a probe payoff reads `tape::len()` on one adjoint path at the
//! moment the payoff is evaluated (path simulation plus the payoff's own
//! operations; the three discounting nodes come after it).

use std::cell::Cell;
use std::hint::black_box;
use std::time::{Duration, Instant};

use adjoint_core::{
    AsianCall, Payoff, Real, bump_and_reval, monte_carlo_greeks_bucketed,
    monte_carlo_value_bucketed, tape,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const VOL: f64 = 0.2;
const TIME: f64 = 1.0;
const FIXINGS: usize = 100;
const NUM_PATHS: u64 = 10_000;
const SEED: u64 = 42;
const RUNS: usize = 7;
const BUCKETS: [usize; 7] = [1, 2, 5, 10, 25, 50, 100];

/// Wraps a payoff and records the tape length when it is evaluated.
struct TapeProbe<P> {
    inner: P,
    nodes: Cell<usize>,
}

impl<P: Payoff> Payoff for TapeProbe<P> {
    fn fixings(&self) -> usize {
        self.inner.fixings()
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        let value = self.inner.value(path);
        self.nodes.set(tape::len());
        value
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

fn main() {
    let payoff = AsianCall {
        strike: 100.0,
        fixings: FIXINGS,
    };

    println!(
        "Asian call, {FIXINGS} fixings, T = {TIME}, {NUM_PATHS} paths, seed {SEED}, \
         {RUNS} timed runs per method after 1 warm-up"
    );
    println!();
    println!(
        "| N | sens. | valuation ms | AAD ms (median [min, max]) | bump ms (median [min, max]) \
         | AAD / bump | AAD / valuation | bump / valuation | tape nodes / path | max gap (SE) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");

    for &n in &BUCKETS {
        let vols = vec![VOL; n];
        let valuation = || {
            let mut rng = StdRng::seed_from_u64(SEED);
            monte_carlo_value_bucketed(&payoff, SPOT, RATE, &vols, TIME, NUM_PATHS, &mut rng)
        };
        let adjoint = || {
            let mut rng = StdRng::seed_from_u64(SEED);
            monte_carlo_greeks_bucketed(&payoff, SPOT, RATE, &vols, TIME, NUM_PATHS, &mut rng)
        };
        let bump = || bump_and_reval(&payoff, SPOT, RATE, &vols, TIME, NUM_PATHS, SEED);

        let probe = TapeProbe {
            inner: payoff,
            nodes: Cell::new(0),
        };
        let mut rng = StdRng::seed_from_u64(SEED);
        monte_carlo_greeks_bucketed(&probe, SPOT, RATE, &vols, TIME, 1, &mut rng);
        let tape_nodes = probe.nodes.get();

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
        let (g, se) = (&aad.greeks, &aad.std_error);
        let mut gap = ((g.delta - bumped.delta) / se.delta).abs();
        gap = gap.max(((g.rho - bumped.rho) / se.rho).abs());
        for j in 0..n {
            gap = gap.max(((g.vegas[j] - bumped.vegas[j]) / se.vegas[j]).abs());
        }

        let (val_med, _, _) = summary(t_val);
        let (aad_med, aad_min, aad_max) = summary(t_aad);
        let (bump_med, bump_min, bump_max) = summary(t_bump);
        println!(
            "| {n} | {} | {val_med:.1} | {aad_med:.1} [{aad_min:.1}, {aad_max:.1}] \
             | {bump_med:.1} [{bump_min:.1}, {bump_max:.1}] | {:.3} | {:.2} | {:.1} | {tape_nodes} | {gap:.3} |",
            n + 2,
            aad_med / bump_med,
            aad_med / val_med,
            bump_med / val_med,
        );
    }
}
