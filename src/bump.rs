//! Bump-and-reval: Greeks by central finite differences of the tape-free
//! pricer, with no adjoint machinery involved anywhere.
//!
//! This is how sensitivities are computed without AAD, so it is the baseline
//! the adjoint pricer is benchmarked against, and an independent oracle for
//! it. Every revaluation re-seeds its own `StdRng` from the same seed, so all
//! of them see the same paths (common random numbers). The differences are
//! then taken path by path, and the noise of two independent Monte Carlo runs
//! never enters the quotient.
//!
//! The cost is what the benchmark measures: one base valuation plus two
//! revaluations per input, 2N + 5 full valuations for N vol buckets.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::payoff::Payoff;
use crate::pricer::{BucketedGreeks, monte_carlo_value_bucketed};

/// Spot bump, relative to spot.
const SPOT_BUMP: f64 = 1e-4;
/// Rate bump, absolute.
const RATE_BUMP: f64 = 1e-4;
/// Bucket vol bump, absolute.
const VOL_BUMP: f64 = 1e-4;

/// Price, Delta, Rho, and one vega per vol bucket by central differences,
/// each revaluation on `seed`. Same arguments and meaning as
/// [`crate::monte_carlo_greeks_bucketed`], which on the same seed returns
/// exactly this price.
pub fn bump_and_reval<P: Payoff>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vols: &[f64],
    time: f64,
    num_paths: u64,
    seed: u64,
) -> BucketedGreeks {
    let value = |spot: f64, rate: f64, vols: &[f64]| {
        let mut rng = StdRng::seed_from_u64(seed);
        monte_carlo_value_bucketed(payoff, spot, rate, vols, time, num_paths, &mut rng)
    };

    let h_spot = SPOT_BUMP * spot;
    let delta =
        (value(spot + h_spot, rate, vols) - value(spot - h_spot, rate, vols)) / (2.0 * h_spot);
    let rho = (value(spot, rate + RATE_BUMP, vols) - value(spot, rate - RATE_BUMP, vols))
        / (2.0 * RATE_BUMP);

    let mut bumped = vols.to_vec();
    let vegas = (0..vols.len())
        .map(|j| {
            bumped[j] = vols[j] + VOL_BUMP;
            let up = value(spot, rate, &bumped);
            bumped[j] = vols[j] - VOL_BUMP;
            let down = value(spot, rate, &bumped);
            bumped[j] = vols[j];
            (up - down) / (2.0 * VOL_BUMP)
        })
        .collect();

    BucketedGreeks {
        price: value(spot, rate, vols),
        delta,
        rho,
        vegas,
    }
}
