//! Option pricing — analytical Black-Scholes baselines to check against, and
//! a Monte Carlo simulator, generic over the payoff, that gets to the same
//! Greeks a completely different way: pathwise adjoint differentiation
//! instead of a closed-form formula.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::dual::Dual;
use crate::payoff::{EuropeanCall, Payoff, Real};
use crate::special::{norm_cdf, norm_pdf};
use crate::tape;

/// Price and sensitivities of an option.
#[derive(Clone, Copy, Debug)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// A Monte Carlo estimate, with each quantity's own standard error attached.
/// We do this because Price/Delta/Vega/Rho are independent pathwise
/// estimators with pretty different variances — Delta's estimator is way
/// tighter than Vega's, for instance — so a single fixed tolerance across all
/// four would end up wrong for most of them.
#[derive(Clone, Copy, Debug)]
pub struct MonteCarloResult {
    pub greeks: Greeks,
    pub std_error: Greeks,
}

/// Closed-form Black-Scholes price and Greeks of a European Call — the ground truth we check
/// the Monte Carlo engine against. This isn't part of the AAD pipeline at
/// all; everything here is just a plain `f64`.
pub fn black_scholes(spot: f64, strike: f64, rate: f64, vol: f64, time: f64) -> Greeks {
    let sqrt_t = time.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * time) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let discount = (-rate * time).exp();

    Greeks {
        price: spot * norm_cdf(d1) - strike * discount * norm_cdf(d2),
        delta: norm_cdf(d1),
        vega: spot * sqrt_t * norm_pdf(d1),
        rho: strike * time * discount * norm_cdf(d2),
    }
}

/// Closed-form Black-Scholes price and Greeks of a European Put, the
/// reference for the Monte Carlo put. Plain `f64`, like [`black_scholes`].
pub fn black_scholes_put(spot: f64, strike: f64, rate: f64, vol: f64, time: f64) -> Greeks {
    let sqrt_t = time.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * time) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let discount = (-rate * time).exp();

    Greeks {
        price: strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1),
        delta: norm_cdf(d1) - 1.0,
        vega: spot * sqrt_t * norm_pdf(d1),
        rho: -strike * time * discount * norm_cdf(-d2),
    }
}

/// Streaming mean and variance, using Welford's algorithm — this lets us
/// report a Monte Carlo estimator's standard error without ever storing a
/// single path's sample. Just O(1) state and no heap allocation, which fits
/// right in with the rest of the pricer's zero-allocation hot loop.
#[derive(Clone, Copy, Default)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Fold in another accumulator's samples, as if they had been fed to
    /// `update` here. This is the pairwise combination of Chan, Golub and
    /// LeVeque: the means combine weighted by count, and the squared
    /// deviations pick up a correction for the gap between the two means.
    fn merge(&mut self, other: &RunningStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let (n_a, n_b) = (self.count as f64, other.count as f64);
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * n_b / n;
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.count += other.count;
    }

    /// Standard error of the mean — just sqrt(sample variance / n).
    fn standard_error(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let sample_variance = self.m2 / (self.count as f64 - 1.0);
        (sample_variance / self.count as f64).sqrt()
    }
}

/// Price and sensitivities of an option under a piecewise-constant vol
/// curve: one vega per bucket, in bucket order.
#[derive(Clone, Debug, PartialEq)]
pub struct BucketedGreeks {
    pub price: f64,
    pub delta: f64,
    pub rho: f64,
    pub vegas: Vec<f64>,
}

/// A Monte Carlo estimate on a vol curve, with each quantity's own standard
/// error attached, as in [`MonteCarloResult`].
#[derive(Clone, Debug, PartialEq)]
pub struct BucketedResult {
    pub greeks: BucketedGreeks,
    pub std_error: BucketedGreeks,
}

/// One step of the simulation grid. Every step lies inside a single vol
/// bucket, so the exact log-normal transition applies with that bucket's vol.
struct Step {
    bucket: usize,
    dt: f64,
    sqrt_dt: f64,
    /// Whether the step ends on one of the payoff's fixing dates.
    fixing: bool,
}

/// The simulation grid out to maturity, which is also the discounting date.
struct TimeGrid {
    steps: Vec<Step>,
    maturity: f64,
}

/// The payoff's fixing dates merged with the vol buckets' edges.
///
/// Fixings sit at i T / n and bucket edges at j T / N, so in units of
/// T / (n N) both are integers (i N and j n) and merge exactly, with no
/// floating-point comparison of times. With one bucket this is the uniform
/// fixing grid, and dt = T / n computed exactly as the flat pricer did.
fn time_grid(fixings: usize, buckets: usize, time: f64) -> TimeGrid {
    assert!(fixings > 0, "a payoff needs at least one fixing date");
    assert!(buckets > 0, "a vol curve needs at least one bucket");
    let units = fixings * buckets;

    let mut grid = Vec::with_capacity(fixings + buckets);
    let (mut i, mut j, mut prev) = (1, 1, 0);
    while i <= fixings {
        let fixing_at = i * buckets;
        let edge_at = j * fixings;
        let next = fixing_at.min(edge_at);
        let dt = time * (next - prev) as f64 / units as f64;
        grid.push(Step {
            bucket: prev / fixings,
            dt,
            sqrt_dt: dt.sqrt(),
            fixing: next == fixing_at,
        });
        if next == fixing_at {
            i += 1;
        }
        if next == edge_at {
            j += 1;
        }
        prev = next;
    }
    TimeGrid {
        steps: grid,
        maturity: time,
    }
}

/// One path's discounted payoff, written once for both scalar types.
///
/// The spot moves along `grid` by the exact log-normal step
/// S * exp((r - vol^2 / 2) dt + vol sqrt(dt) z), which is Euler-Maruyama on
/// ln S and has no discretization bias for GBM with piecewise-constant vol.
/// Each step draws one normal from `rng` in date order, so two runs on the
/// same seed see the same paths whatever scalar type they are in. `path` is
/// a buffer reused across paths; after the first path it never reallocates.
///
/// The drift and vol sqrt(dt) terms are recorded once per run of steps with
/// the same bucket and dt, not once per step, which keeps the tape short.
fn discounted_payoff<T: Real, P: Payoff, R: Rng>(
    payoff: &P,
    grid: &TimeGrid,
    spot: T,
    rate: T,
    vols: &[T],
    path: &mut Vec<T>,
    rng: &mut R,
) -> T {
    path.clear();
    let mut s = spot;
    let mut segment: Option<(usize, f64, T, T)> = None;
    for step in &grid.steps {
        let (drift, vol_sqrt_dt) = match segment {
            Some((bucket, dt, drift, vol_sqrt_dt)) if bucket == step.bucket && dt == step.dt => {
                (drift, vol_sqrt_dt)
            }
            _ => {
                let vol = vols[step.bucket];
                let drift = (rate - vol * vol * 0.5) * step.dt;
                let vol_sqrt_dt = vol * step.sqrt_dt;
                segment = Some((step.bucket, step.dt, drift, vol_sqrt_dt));
                (drift, vol_sqrt_dt)
            }
        };
        let z: f64 = StandardNormal.sample(rng);
        s = s * (drift + vol_sqrt_dt * z).exp();
        if step.fixing {
            path.push(s);
        }
    }

    (rate * -grid.maturity).exp() * payoff.value(path)
}

/// Monte Carlo price and Greeks of `payoff` under a piecewise-constant vol
/// curve, differentiated with one reverse-mode adjoint sweep per path. That's
/// the pathwise approach, ∇E[V] = E[∇V], which is valid because every payoff
/// offered here is continuous in the path.
///
/// `vols` holds one volatility per bucket, the buckets splitting [0, T] into
/// `vols.len()` equal pieces. Spot, Rate, and every bucket vol are tape
/// variables, so Delta, Rho, and all the bucket vegas fall out of the same
/// sweep that gives us the price: one sweep per path however many buckets
/// there are. Time and the payoff's own terms (strike, fixings) stay plain
/// constants.
///
/// The tape gets reset after every single path (see [`tape::reset`]), and
/// because its high-water-mark capacity carries over between resets, every
/// path after the first one doesn't allocate anything at all during the
/// forward pass.
pub fn monte_carlo_greeks_bucketed<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vols: &[f64],
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> BucketedResult {
    let grid = time_grid(payoff.fixings(), vols.len(), time);
    accumulate_greeks(payoff, &grid, spot, rate, vols, num_paths, rng).into_result()
}

/// What fixes the paths of a parallel run: the seed, and how many paths go
/// in each chunk. It stands where the generator stands in the serial API.
#[derive(Clone, Copy, Debug)]
pub struct PathChunks {
    pub seed: u64,
    pub chunk_size: u64,
}

/// The same estimate as [`monte_carlo_greeks_bucketed`], with the paths
/// split into chunks of `chunks.chunk_size` and the chunks priced in
/// parallel on the current rayon thread pool.
///
/// Chunk `c` draws from its own generator, keyed by (`seed`, `c`), and runs
/// on whichever worker picks it up, recording on that worker's thread-local
/// tape. The per-chunk accumulators are then merged in chunk order. Which
/// paths exist and the order their statistics combine in depend only on
/// the seed, the chunk size, and `num_paths`, so the result is bit-identical at
/// any thread count. A different `chunk_size` gives different paths, and so
/// a different (equally valid) estimate.
pub fn monte_carlo_greeks_parallel<P: Payoff + Sync>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vols: &[f64],
    time: f64,
    num_paths: u64,
    chunks: PathChunks,
) -> BucketedResult {
    let PathChunks { seed, chunk_size } = chunks;
    assert!(chunk_size > 0, "chunk_size must be at least one path");
    let grid = time_grid(payoff.fixings(), vols.len(), time);
    let num_chunks = num_paths.div_ceil(chunk_size);

    let mut priced: Vec<(u64, GreeksStats)> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk| {
            let paths = chunk_size.min(num_paths - chunk * chunk_size);
            let mut rng = chunk_rng(seed, chunk);
            let stats = accumulate_greeks(payoff, &grid, spot, rate, vols, paths, &mut rng);
            (chunk, stats)
        })
        .collect();
    // Merge in chunk order whatever order the workers finished in.
    priced.sort_unstable_by_key(|&(chunk, _)| chunk);

    let mut total = GreeksStats::new(vols.len());
    for (_, stats) in &priced {
        total.merge(stats);
    }
    total.into_result()
}

/// The generator for one chunk of paths: ChaCha12 (`StdRng`) keyed by the
/// seed and the chunk index. Distinct (seed, chunk) pairs are distinct keys,
/// so no two chunks share a stream.
fn chunk_rng(seed: u64, chunk: u64) -> StdRng {
    let mut key = [0u8; 32];
    key[..8].copy_from_slice(&seed.to_le_bytes());
    key[8..16].copy_from_slice(&chunk.to_le_bytes());
    StdRng::from_seed(key)
}

/// Running statistics of every output the adjoint pricer reports.
#[derive(Clone)]
struct GreeksStats {
    price: RunningStats,
    delta: RunningStats,
    rho: RunningStats,
    vegas: Vec<RunningStats>,
}

impl GreeksStats {
    fn new(buckets: usize) -> Self {
        GreeksStats {
            price: RunningStats::default(),
            delta: RunningStats::default(),
            rho: RunningStats::default(),
            vegas: vec![RunningStats::default(); buckets],
        }
    }

    fn merge(&mut self, other: &GreeksStats) {
        self.price.merge(&other.price);
        self.delta.merge(&other.delta);
        self.rho.merge(&other.rho);
        for (mine, theirs) in self.vegas.iter_mut().zip(&other.vegas) {
            mine.merge(theirs);
        }
    }

    fn into_result(self) -> BucketedResult {
        BucketedResult {
            greeks: BucketedGreeks {
                price: self.price.mean,
                delta: self.delta.mean,
                rho: self.rho.mean,
                vegas: self.vegas.iter().map(|v| v.mean).collect(),
            },
            std_error: BucketedGreeks {
                price: self.price.standard_error(),
                delta: self.delta.standard_error(),
                rho: self.rho.standard_error(),
                vegas: self.vegas.iter().map(|v| v.standard_error()).collect(),
            },
        }
    }
}

/// Price `num_paths` paths with one adjoint sweep each, on this thread's
/// tape, and return the running statistics of every output.
fn accumulate_greeks<P: Payoff, R: Rng>(
    payoff: &P,
    grid: &TimeGrid,
    spot: f64,
    rate: f64,
    vols: &[f64],
    num_paths: u64,
    rng: &mut R,
) -> GreeksStats {
    let mut stats = GreeksStats::new(vols.len());
    let mut vol_vars = Vec::with_capacity(vols.len());
    let mut path = Vec::with_capacity(payoff.fixings());

    for _ in 0..num_paths {
        let s = Dual::var(spot);
        let r = Dual::var(rate);
        vol_vars.clear();
        vol_vars.extend(vols.iter().map(|&v| Dual::var(v)));

        let discounted = discounted_payoff(payoff, grid, s, r, &vol_vars, &mut path, rng);
        let grad = discounted.backward();

        stats.price.update(discounted.value);
        stats.delta.update(grad.wrt(s));
        stats.rho.update(grad.wrt(r));
        for (vega, &vol) in stats.vegas.iter_mut().zip(&vol_vars) {
            vega.update(grad.wrt(vol));
        }

        tape::reset();
    }

    stats
}

/// Monte Carlo price of `payoff` under a piecewise-constant vol curve with
/// nothing recorded on the tape.
///
/// Same paths as [`monte_carlo_greeks_bucketed`] for the same seed, and the
/// same accumulator, so its price matches that one exactly. This is the
/// revaluation that bump-and-reval repeats once per bumped input.
pub fn monte_carlo_value_bucketed<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vols: &[f64],
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> f64 {
    let grid = time_grid(payoff.fixings(), vols.len(), time);
    let mut price_stats = RunningStats::default();
    let mut path = Vec::with_capacity(payoff.fixings());

    for _ in 0..num_paths {
        price_stats.update(discounted_payoff(
            payoff, &grid, spot, rate, vols, &mut path, rng,
        ));
    }

    price_stats.mean
}

/// Monte Carlo price and Greeks of `payoff` under a flat vol: the one-bucket
/// curve of [`monte_carlo_greeks_bucketed`].
pub fn monte_carlo_greeks<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> MonteCarloResult {
    let curve = monte_carlo_greeks_bucketed(payoff, spot, rate, &[vol], time, num_paths, rng);
    let flat = |g: &BucketedGreeks| Greeks {
        price: g.price,
        delta: g.delta,
        vega: g.vegas[0],
        rho: g.rho,
    };
    MonteCarloResult {
        greeks: flat(&curve.greeks),
        std_error: flat(&curve.std_error),
    }
}

/// Monte Carlo price of `payoff` under a flat vol with nothing recorded on
/// the tape; see [`monte_carlo_value_bucketed`].
pub fn monte_carlo_value<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> f64 {
    monte_carlo_value_bucketed(payoff, spot, rate, &[vol], time, num_paths, rng)
}

/// Monte Carlo price and Greeks for a European Call; see
/// [`monte_carlo_greeks`].
pub fn monte_carlo_price<R: Rng>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> MonteCarloResult {
    monte_carlo_greeks(
        &EuropeanCall { strike },
        spot,
        rate,
        vol,
        time,
        num_paths,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Merging two accumulators has to give what one accumulator fed every
    /// sample would, including when either side is empty.
    #[test]
    fn merged_running_stats_match_one_sequential_pass() {
        let samples: Vec<f64> = (0..1_000)
            .map(|i| ((i * 7919) % 1_000) as f64 * 0.37 - 50.0)
            .collect();

        let mut sequential = RunningStats::default();
        samples.iter().for_each(|&x| sequential.update(x));

        for split in [0, 1, 333, 999, 1_000] {
            let (mut a, mut b) = (RunningStats::default(), RunningStats::default());
            samples[..split].iter().for_each(|&x| a.update(x));
            samples[split..].iter().for_each(|&x| b.update(x));
            a.merge(&b);

            assert_eq!(a.count, sequential.count, "split {split}");
            assert!(
                (a.mean - sequential.mean).abs() < 1e-12,
                "split {split} mean"
            );
            assert!(
                (a.m2 - sequential.m2).abs() < 1e-9 * sequential.m2,
                "split {split} m2: {} vs {}",
                a.m2,
                sequential.m2
            );
        }
    }
}
