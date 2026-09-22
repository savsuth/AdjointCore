//! European Call pricing — an analytical Black-Scholes baseline to check
//! against, and a Monte Carlo Euler-Maruyama simulator that gets to the same
//! Greeks a completely different way: pathwise adjoint differentiation
//! instead of a closed-form formula.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::dual::Dual;
use crate::special::{norm_cdf, norm_pdf};
use crate::tape;

/// Price and sensitivities of a European Call.
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

/// Closed-form Black-Scholes price and Greeks — the ground truth we check
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

/// Streaming mean and variance, using Welford's algorithm — this lets us
/// report a Monte Carlo estimator's standard error without ever storing a
/// single path's sample. Just O(1) state and no heap allocation, which fits
/// right in with the rest of the pricer's zero-allocation hot loop.
#[derive(Default)]
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

    /// Standard error of the mean — just sqrt(sample variance / n).
    fn standard_error(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let sample_variance = self.m2 / (self.count as f64 - 1.0);
        (sample_variance / self.count as f64).sqrt()
    }
}

/// Monte Carlo price and Greeks for a European Call, simulated with
/// Euler-Maruyama and differentiated with one reverse-mode adjoint sweep per
/// path — that's the pathwise approach, ∇E[V] = E[∇V].
///
/// Spot, Rate, and Vol are all tape variables, so Delta/Rho/Vega just fall
/// out of the same sweep that gives us the price. Strike and Time stay plain
/// constants, since we're not asking for a Greek against either of those.
///
/// The tape gets reset after every single path (see [`tape::reset`]), and
/// because its high-water-mark capacity carries over between resets, every
/// path after the first one doesn't allocate anything at all during the
/// forward pass.
pub fn monte_carlo_price<R: Rng>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> MonteCarloResult {
    let sqrt_t = time.sqrt();

    let mut price_stats = RunningStats::default();
    let mut delta_stats = RunningStats::default();
    let mut vega_stats = RunningStats::default();
    let mut rho_stats = RunningStats::default();

    for _ in 0..num_paths {
        let s = Dual::var(spot);
        let r = Dual::var(rate);
        let vol_d = Dual::var(vol);

        let z: f64 = StandardNormal.sample(rng);

        let drift = (r - 0.5 * vol_d * vol_d) * time;
        let diffusion = vol_d * sqrt_t * z;
        let terminal_spot = s * (drift + diffusion).exp();

        let discount = (0.0 - r * time).exp();
        let payoff = (terminal_spot - strike).max_zero();
        let discounted_payoff = discount * payoff;

        let grad = discounted_payoff.backward();

        price_stats.update(discounted_payoff.value);
        delta_stats.update(grad.wrt(s));
        vega_stats.update(grad.wrt(vol_d));
        rho_stats.update(grad.wrt(r));

        tape::reset();
    }

    MonteCarloResult {
        greeks: Greeks {
            price: price_stats.mean,
            delta: delta_stats.mean,
            vega: vega_stats.mean,
            rho: rho_stats.mean,
        },
        std_error: Greeks {
            price: price_stats.standard_error(),
            delta: delta_stats.standard_error(),
            vega: vega_stats.standard_error(),
            rho: rho_stats.standard_error(),
        },
    }
}
