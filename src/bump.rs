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
//! revaluations per input, 2N + 5 full valuations for N vol buckets (and
//! 2(N + K) + 5 for the CVA of K netting sets, one hazard rate each).

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::payoff::Payoff;
use crate::pricer::{BucketedGreeks, Market, monte_carlo_value_bucketed};
use crate::xva::{CvaGreeks, NettingSet, cva_value};

/// Spot bump, relative to spot.
const SPOT_BUMP: f64 = 1e-4;
/// Rate bump, absolute.
const RATE_BUMP: f64 = 1e-4;
/// Bucket vol bump, absolute.
const VOL_BUMP: f64 = 1e-4;
/// Hazard rate bump, absolute.
const HAZARD_BUMP: f64 = 1e-4;

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

/// Each netting set's CVA, Delta, Rho, bucket vegas and hazard-rate
/// sensitivities by central differences of the tape-free CVA, every
/// revaluation on `seed`. Same arguments and meaning as
/// [`crate::monte_carlo_cva`], which on the same seed returns exactly these
/// CVAs. One revaluation prices every set, so this costs 2(N + K) + 5
/// valuations for N vol buckets and K netting sets.
pub fn bump_and_reval_cva(
    sets: &[NettingSet],
    market: &Market,
    exposure_dates: usize,
    num_paths: u64,
    seed: u64,
) -> Vec<CvaGreeks> {
    let value = |sets: &[NettingSet], market: &Market| {
        let mut rng = StdRng::seed_from_u64(seed);
        cva_value(sets, market, exposure_dates, num_paths, &mut rng)
    };
    let difference = |up: Vec<f64>, down: Vec<f64>, h: f64| -> Vec<f64> {
        up.iter()
            .zip(&down)
            .map(|(u, d)| (u - d) / (2.0 * h))
            .collect()
    };

    let h_spot = SPOT_BUMP * market.spot;
    let at_spot = |spot: f64| value(sets, &Market { spot, ..*market });
    let delta = difference(
        at_spot(market.spot + h_spot),
        at_spot(market.spot - h_spot),
        h_spot,
    );
    let at_rate = |rate: f64| value(sets, &Market { rate, ..*market });
    let rho = difference(
        at_rate(market.rate + RATE_BUMP),
        at_rate(market.rate - RATE_BUMP),
        RATE_BUMP,
    );

    let mut bumped = market.vols.to_vec();
    let vegas: Vec<Vec<f64>> = (0..market.vols.len())
        .map(|j| {
            bumped[j] = market.vols[j] + VOL_BUMP;
            let up = value(
                sets,
                &Market {
                    vols: &bumped,
                    ..*market
                },
            );
            bumped[j] = market.vols[j] - VOL_BUMP;
            let down = value(
                sets,
                &Market {
                    vols: &bumped,
                    ..*market
                },
            );
            bumped[j] = market.vols[j];
            difference(up, down, VOL_BUMP)
        })
        .collect();

    let mut bumped_sets = sets.to_vec();
    let hazards: Vec<Vec<f64>> = (0..sets.len())
        .map(|l| {
            bumped_sets[l].hazard_rate = sets[l].hazard_rate + HAZARD_BUMP;
            let up = value(&bumped_sets, market);
            bumped_sets[l].hazard_rate = sets[l].hazard_rate - HAZARD_BUMP;
            let down = value(&bumped_sets, market);
            bumped_sets[l].hazard_rate = sets[l].hazard_rate;
            difference(up, down, HAZARD_BUMP)
        })
        .collect();

    value(sets, market)
        .into_iter()
        .enumerate()
        .map(|(i, cva)| CvaGreeks {
            cva,
            delta: delta[i],
            rho: rho[i],
            vegas: vegas.iter().map(|v| v[i]).collect(),
            hazards: hazards.iter().map(|h| h[i]).collect(),
        })
        .collect()
}
