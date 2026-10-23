//! Option pricing — analytical Black-Scholes baselines to check against, and
//! a Monte Carlo simulator, generic over the payoff, that gets to the same
//! Greeks a completely different way: pathwise adjoint differentiation
//! instead of a closed-form formula.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

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

/// One path's discounted payoff, written once for both scalar types.
///
/// The spot moves between equally spaced fixing dates by the exact
/// log-normal step S * exp((r - vol^2 / 2) dt + vol sqrt(dt) z), which is
/// Euler-Maruyama on ln S and has no discretization bias for GBM. Each step
/// draws one normal from `rng` in date order, so two runs on the same seed
/// see the same paths whatever scalar type they are in. `path` is a buffer
/// reused across paths; after the first path it never reallocates.
fn discounted_payoff<T: Real, P: Payoff, R: Rng>(
    payoff: &P,
    spot: T,
    rate: T,
    vol: T,
    time: f64,
    path: &mut Vec<T>,
    rng: &mut R,
) -> T {
    let steps = payoff.fixings();
    let dt = time / steps as f64;
    let drift = (rate - vol * vol * 0.5) * dt;
    let vol_sqrt_dt = vol * dt.sqrt();

    path.clear();
    let mut s = spot;
    for _ in 0..steps {
        let z: f64 = StandardNormal.sample(rng);
        s = s * (drift + vol_sqrt_dt * z).exp();
        path.push(s);
    }

    (rate * -time).exp() * payoff.value(path)
}

/// Monte Carlo price and Greeks of `payoff`, differentiated with one
/// reverse-mode adjoint sweep per path. That's the pathwise approach,
/// ∇E[V] = E[∇V], which is valid because every payoff offered here is
/// continuous in the path.
///
/// Spot, Rate, and Vol are all tape variables, so Delta/Rho/Vega just fall
/// out of the same sweep that gives us the price. Time and the payoff's own
/// terms (strike, fixings) stay plain constants.
///
/// The tape gets reset after every single path (see [`tape::reset`]), and
/// because its high-water-mark capacity carries over between resets, every
/// path after the first one doesn't allocate anything at all during the
/// forward pass.
pub fn monte_carlo_greeks<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> MonteCarloResult {
    let mut price_stats = RunningStats::default();
    let mut delta_stats = RunningStats::default();
    let mut vega_stats = RunningStats::default();
    let mut rho_stats = RunningStats::default();
    let mut path = Vec::with_capacity(payoff.fixings());

    for _ in 0..num_paths {
        let s = Dual::var(spot);
        let r = Dual::var(rate);
        let vol_d = Dual::var(vol);

        let discounted = discounted_payoff(payoff, s, r, vol_d, time, &mut path, rng);
        let grad = discounted.backward();

        price_stats.update(discounted.value);
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

/// Monte Carlo price of `payoff` with nothing recorded on the tape.
///
/// Same paths as [`monte_carlo_greeks`] for the same seed, and the same
/// accumulator, so its price matches that one exactly. This is the
/// revaluation that bump-and-reval repeats once per bumped input.
pub fn monte_carlo_value<P: Payoff, R: Rng>(
    payoff: &P,
    spot: f64,
    rate: f64,
    vol: f64,
    time: f64,
    num_paths: u64,
    rng: &mut R,
) -> f64 {
    let mut price_stats = RunningStats::default();
    let mut path = Vec::with_capacity(payoff.fixings());

    for _ in 0..num_paths {
        price_stats.update(discounted_payoff(
            payoff, spot, rate, vol, time, &mut path, rng,
        ));
    }

    price_stats.mean
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
