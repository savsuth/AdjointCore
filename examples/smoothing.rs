//! Smoothing bias: how far the pathwise Greeks of a smoothed digital and a
//! smoothed up-and-out call sit from the true Greeks, and how their
//! standard errors grow as the smoothing width h shrinks.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example smoothing
//! ```
//!
//! Setup: S = 100, K = 100, r = 5%, flat vol 20%, T = 1. Every indicator is
//! smoothed into a centered call spread of width h; h = 0 is the exact
//! payoff. A bias is the Monte Carlo estimate minus the reference, and z is
//! that bias over its standard error.
//!
//! Digital call (cash-or-nothing, pays 1): `DIGITAL_PATHS` paths, seed 42.
//! The reference is the Black-Scholes closed form, exact, so the standard
//! error is the estimator's alone. The smoothed payoff's own expectation is
//! the call spread (C(K - h/2) - C(K + h/2)) / h, also closed form, which
//! splits the measured bias into smoothing bias (exact) and noise.
//!
//! Up-and-out call, barrier 120 on 12 monthly fixings: `BARRIER_PATHS`
//! paths, seed 42 (parallel pricer, chunks of 10,000). Neither reference is
//! closed form:
//! - price: the unsmoothed Monte Carlo price. The price bias is measured on
//!   common paths as the mean of the per-path difference, smoothed payoff
//!   minus exact payoff, over `REFERENCE_PATHS` paths (seed 7), so its
//!   standard error is that of the difference alone;
//! - Greeks: central-difference bump-and-reval of the unsmoothed payoff on
//!   common random numbers (`bump_and_reval`), unbiased for the true Greeks
//!   up to O(bump^2), averaged over `BUMP_SEEDS` independent seeds of
//!   `BUMP_PATHS` paths, with the standard error taken across seeds. A
//!   Greek bias's standard error combines the estimator's and this
//!   reference's, which are independent (different seeds).
//!
//! Within a table every row reuses the same paths, so the rows' noise is
//! correlated: compare a column down the rows for its trend, not row by row
//! for noise.

use adjoint_core::{
    DigitalCall, Greeks, PathChunks, Payoff, Real, UpAndOutCall, black_scholes,
    black_scholes_digital, bump_and_reval, monte_carlo_greeks, monte_carlo_greeks_parallel,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const RATE: f64 = 0.05;
const VOL: f64 = 0.2;
const TIME: f64 = 1.0;
const SEED: u64 = 42;

const DIGITAL_PATHS: u64 = 1_000_000;
const DIGITAL_WIDTHS: [f64; 7] = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];

const BARRIER: f64 = 120.0;
const FIXINGS: usize = 12;
const BARRIER_PATHS: u64 = 1_000_000;
const BARRIER_CHUNKS: PathChunks = PathChunks {
    seed: SEED,
    chunk_size: 10_000,
};
const BARRIER_WIDTHS: [f64; 6] = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
const REFERENCE_PATHS: u64 = 20_000_000;
const REFERENCE_CHUNKS: PathChunks = PathChunks {
    seed: 7,
    chunk_size: 10_000,
};
const BUMP_SEEDS: u64 = 128;
const BUMP_PATHS: u64 = 1_000_000;

/// "bias (z)", with z left out when the standard error is zero.
fn biased(bias: f64, se: f64, digits: usize) -> String {
    if se > 0.0 {
        format!("{bias:+.digits$} ({:+.1})", bias / se)
    } else {
        format!("{bias:+.digits$} (-)")
    }
}

fn digital() {
    let truth = black_scholes_digital(SPOT, STRIKE, RATE, VOL, TIME);
    println!(
        "Digital call, {DIGITAL_PATHS} paths, seed {SEED}. Black-Scholes: price {:.5}, \
         Delta {:.5}, Vega {:.4}, Rho {:.4}",
        truth.price, truth.delta, truth.vega, truth.rho
    );
    println!();
    println!(
        "| h | price bias (z) | Delta | Delta bias (z) | Delta SE | Delta exact bias | Delta RMSE \
         | Vega bias (z) | Vega SE | Rho bias (z) | gap to spread (max abs z) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let mut unsmoothed = None;
    for width in DIGITAL_WIDTHS {
        let mut rng = StdRng::seed_from_u64(SEED);
        let payoff = DigitalCall {
            strike: STRIKE,
            smoothing: width,
        };
        let result = monte_carlo_greeks(&payoff, SPOT, RATE, VOL, TIME, DIGITAL_PATHS, &mut rng);
        let (mc, se) = (result.greeks, result.std_error);

        // What the pathwise Delta estimates, exactly: the call spread's Delta,
        // or at h = 0 zero, since it is zero on every path.
        let (expected_delta, gap) = if width > 0.0 {
            let target = call_spread(width);
            let z = |m: f64, t: f64, s: f64| ((m - t) / s).abs();
            let worst = [
                z(mc.price, target.price, se.price),
                z(mc.delta, target.delta, se.delta),
                z(mc.vega, target.vega, se.vega),
                z(mc.rho, target.rho, se.rho),
            ]
            .into_iter()
            .fold(0.0, f64::max);
            (target.delta, format!("{worst:.2}"))
        } else {
            unsmoothed = Some(result);
            (0.0, "-".to_string())
        };
        let exact_bias = expected_delta - truth.delta;
        let rmse = (exact_bias * exact_bias + se.delta * se.delta).sqrt();

        println!(
            "| {width} | {} | {:.5} | {} | {:.5} | {exact_bias:+.1e} | {rmse:.5} | {} | {:.4} \
             | {} | {gap} |",
            biased(mc.price - truth.price, se.price, 5),
            mc.delta,
            biased(mc.delta - truth.delta, se.delta, 5),
            se.delta,
            biased(mc.vega - truth.vega, se.vega, 4),
            se.vega,
            biased(mc.rho - truth.rho, se.rho, 4),
        );
    }

    // Mean 0 and standard error 0 can only happen if every path's value is 0.
    if let Some(r) = unsmoothed {
        println!();
        println!(
            "Unsmoothed digital: pathwise Delta mean {:e}, SE {:e}; Vega mean {:e}, SE {:e}; \
             Rho + T * price = {:e}",
            r.greeks.delta,
            r.std_error.delta,
            r.greeks.vega,
            r.std_error.vega,
            r.greeks.rho + TIME * r.greeks.price
        );
    }
    println!();
}

/// The call spread (C(K - h/2) - C(K + h/2)) / h: the smoothed digital's
/// price and Greeks, exactly.
fn call_spread(width: f64) -> Greeks {
    let low = black_scholes(SPOT, STRIKE - 0.5 * width, RATE, VOL, TIME);
    let high = black_scholes(SPOT, STRIKE + 0.5 * width, RATE, VOL, TIME);
    Greeks {
        price: (low.price - high.price) / width,
        delta: (low.delta - high.delta) / width,
        vega: (low.vega - high.vega) / width,
        rho: (low.rho - high.rho) / width,
    }
}

/// Smoothed payoff minus exact payoff on the same path: its Monte Carlo
/// mean is the smoothing bias of the price, with its own standard error.
struct SmoothingGap {
    width: f64,
}

impl Payoff for SmoothingGap {
    fn fixings(&self) -> usize {
        FIXINGS
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        barrier_payoff(self.width).value(path) - barrier_payoff(0.0).value(path)
    }
}

fn barrier_payoff(width: f64) -> UpAndOutCall {
    UpAndOutCall {
        strike: STRIKE,
        barrier: BARRIER,
        fixings: FIXINGS,
        smoothing: width,
    }
}

/// Mean and standard error of each Greek across independent seeds of
/// common-random-number bump-and-reval on the exact barrier.
fn bumped_reference() -> (Greeks, Greeks) {
    let exact = barrier_payoff(0.0);
    let mut runs: Vec<(u64, Greeks)> = (0..BUMP_SEEDS)
        .into_par_iter()
        .map(|b| {
            let g = bump_and_reval(&exact, SPOT, RATE, &[VOL], TIME, BUMP_PATHS, 1_000 + b);
            let greeks = Greeks {
                price: g.price,
                delta: g.delta,
                vega: g.vegas[0],
                rho: g.rho,
            };
            (b, greeks)
        })
        .collect();
    // Sum in seed order whatever order the workers finished in.
    runs.sort_unstable_by_key(|&(b, _)| b);
    let runs: Vec<Greeks> = runs.into_iter().map(|(_, g)| g).collect();
    let n = runs.len() as f64;
    let stat = |f: fn(&Greeks) -> f64| {
        let mean = runs.iter().map(f).sum::<f64>() / n;
        let var = runs.iter().map(|g| (f(g) - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    };
    let (price, price_se) = stat(|g| g.price);
    let (delta, delta_se) = stat(|g| g.delta);
    let (vega, vega_se) = stat(|g| g.vega);
    let (rho, rho_se) = stat(|g| g.rho);
    (
        Greeks {
            price,
            delta,
            vega,
            rho,
        },
        Greeks {
            price: price_se,
            delta: delta_se,
            vega: vega_se,
            rho: rho_se,
        },
    )
}

fn barrier() {
    let exact = monte_carlo_greeks_parallel(
        &barrier_payoff(0.0),
        SPOT,
        RATE,
        &[VOL],
        TIME,
        REFERENCE_PATHS,
        REFERENCE_CHUNKS,
    );
    let (bumped, bumped_se) = bumped_reference();

    println!(
        "Up-and-out call, barrier {BARRIER}, {FIXINGS} fixings, {BARRIER_PATHS} paths, seed {SEED}."
    );
    println!(
        "Unsmoothed MC price ({REFERENCE_PATHS} paths, seed {}): {:.5} (SE {:.5}). \
         Greek reference (CRN bump-and-reval of the unsmoothed payoff, {BUMP_SEEDS} seeds x \
         {BUMP_PATHS} paths): Delta {:.5} (SE {:.5}), Vega {:.4} (SE {:.4}), Rho {:.4} (SE {:.4})",
        REFERENCE_CHUNKS.seed,
        exact.greeks.price,
        exact.std_error.price,
        bumped.delta,
        bumped_se.delta,
        bumped.vega,
        bumped_se.vega,
        bumped.rho,
        bumped_se.rho
    );
    println!();
    println!(
        "| h | price smoothing bias (z) | Delta | Delta bias (z) | Delta SE | Vega | Vega bias (z) \
         | Vega SE | Rho | Rho bias (z) | Rho SE |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let combined = |a: f64, b: f64| (a * a + b * b).sqrt();
    for width in BARRIER_WIDTHS {
        let gap = monte_carlo_greeks_parallel(
            &SmoothingGap { width },
            SPOT,
            RATE,
            &[VOL],
            TIME,
            REFERENCE_PATHS,
            REFERENCE_CHUNKS,
        );
        let price_bias = if width > 0.0 {
            biased(gap.greeks.price, gap.std_error.price, 5)
        } else {
            "0 (exact)".to_string()
        };
        let result = monte_carlo_greeks_parallel(
            &barrier_payoff(width),
            SPOT,
            RATE,
            &[VOL],
            TIME,
            BARRIER_PATHS,
            BARRIER_CHUNKS,
        );
        let (mc, se) = (&result.greeks, &result.std_error);
        println!(
            "| {width} | {price_bias} | {:.4} | {} | {:.4} | {:.3} | {} | {:.3} | {:.3} | {} | {:.3} |",
            mc.delta,
            biased(
                mc.delta - bumped.delta,
                combined(se.delta, bumped_se.delta),
                4
            ),
            se.delta,
            mc.vegas[0],
            biased(
                mc.vegas[0] - bumped.vega,
                combined(se.vegas[0], bumped_se.vega),
                3
            ),
            se.vegas[0],
            mc.rho,
            biased(mc.rho - bumped.rho, combined(se.rho, bumped_se.rho), 3),
            se.rho,
        );
    }
}

fn main() {
    digital();
    barrier();
}
