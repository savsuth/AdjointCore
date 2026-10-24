//! Variance reduction for the price and every pathwise Greek: antithetic
//! pairs and a geometric-average control variate.
//!
//! The control variate is the geometric-average Asian call on the same path
//! (Kemna and Vorst), used with coefficient 1. Its expectation has a closed
//! form under Black-Scholes with a piecewise-constant vol curve, because the
//! log of a geometric average of log-normal spots is normal. That closed form
//! is written in `Dual` arithmetic and recorded on the tape once, so a single
//! backward sweep gives its exact Greeks as well as its price. The controlled
//! estimator samples payoff minus control along the paths and adds back the
//! control's exact price and Greeks. A fixed coefficient keeps it unbiased
//! for the price and every Greek; an estimated optimal coefficient would buy
//! a little more reduction at the cost of an O(1/n) bias.

use rand::Rng;

use crate::dual::Dual;
use crate::payoff::{GeometricAsianCall, Payoff, Real};
use crate::pricer::{
    BucketedGreeks, BucketedResult, Market, TimeGrid, accumulate_greeks, time_grid,
};
use crate::tape;

/// Which variance reductions a run uses. The default is neither.
#[derive(Clone, Copy, Debug, Default)]
pub struct VarianceReduction {
    /// Pair every path with its mirror (every normal negated) and use the
    /// pair's average as one sample.
    pub antithetic: bool,
    /// Subtract the geometric-average Asian call on the same path and add
    /// back its closed-form price and Greeks.
    pub control_variate: bool,
}

/// Monte Carlo price and Greeks of `payoff` with the chosen variance
/// reductions, on the vol curve in `market`.
///
/// `num_paths` counts simulated paths whatever the reduction, so results
/// compare at equal simulation effort; with `antithetic` it must be even and
/// gives `num_paths / 2` independent pairs. With neither reduction this is
/// exactly [`crate::monte_carlo_greeks_bucketed`]. With `control_variate`
/// the payoff has to offer a geometric control (see
/// [`Payoff::geometric_control`]).
pub fn monte_carlo_greeks_reduced<P: Payoff, R: Rng>(
    payoff: &P,
    market: &Market,
    num_paths: u64,
    rng: &mut R,
    reduction: VarianceReduction,
) -> BucketedResult {
    let samples = if reduction.antithetic {
        assert!(
            num_paths.is_multiple_of(2),
            "antithetic sampling needs an even number of paths, got {num_paths}"
        );
        num_paths / 2
    } else {
        num_paths
    };
    let grid = time_grid(payoff.fixings(), market.vols.len(), market.time);

    if !reduction.control_variate {
        return accumulate_greeks(payoff, &grid, market, samples, rng, reduction.antithetic)
            .into_result();
    }

    let control = payoff
        .geometric_control()
        .expect("control_variate needs a payoff that offers a geometric control");
    assert_eq!(
        control.fixings,
        payoff.fixings(),
        "control must share the payoff's fixings"
    );
    let residual = MinusControl { payoff, control };
    let mut result =
        accumulate_greeks(&residual, &grid, market, samples, rng, reduction.antithetic)
            .into_result();

    let exact = geometric_asian_call(&control, market);
    let g = &mut result.greeks;
    g.price += exact.price;
    g.delta += exact.delta;
    g.rho += exact.rho;
    for (vega, exact_vega) in g.vegas.iter_mut().zip(&exact.vegas) {
        *vega += exact_vega;
    }
    result
}

/// Closed-form price and Greeks of a geometric-average Asian call under
/// Black-Scholes with the vol curve in `market`. One recording of the
/// closed form on this thread's tape and one backward sweep give the price,
/// Delta, Rho, and every bucket vega exactly.
pub fn geometric_asian_call(control: &GeometricAsianCall, market: &Market) -> BucketedGreeks {
    let grid = time_grid(control.fixings, market.vols.len(), market.time);
    let s = Dual::var(market.spot);
    let r = Dual::var(market.rate);
    let vols: Vec<Dual> = market.vols.iter().map(|&v| Dual::var(v)).collect();

    let price = geometric_expectation(control, &grid, s, r, &vols);
    let grad = price.backward();
    let greeks = BucketedGreeks {
        price: price.value,
        delta: grad.wrt(s),
        rho: grad.wrt(r),
        vegas: vols.iter().map(|&v| grad.wrt(v)).collect(),
    };
    tape::reset();
    greeks
}

/// Discounted E[max(G - K, 0)].
///
/// With V(t) the integrated variance up to t, ln S(t) is ln S0 + r t - V(t)/2
/// plus a Brownian term of variance V(t), and two fixings' Brownian terms
/// have covariance V(min(t_i, t_k)). So with n fixings, ln G is normal, and
/// the call on a log-normal gives:
///
/// ```text
/// m  = ln S0 + (r sum(t_i) - sum(V(t_i)) / 2) / n
/// v  = sum_i sum_k V(min(t_i, t_k)) / n^2
///    = sum_i (2 (n - i) + 1) V(t_i) / n^2,   i = 1..n
/// d1 = (m - ln K + v) / sqrt(v),   d2 = d1 - sqrt(v)
/// price = e^{-rT} (e^{m + v/2} N(d1) - K N(d2))
/// ```
///
/// V(t) accumulates along the merged grid, whose steps each sit inside one
/// vol bucket.
fn geometric_expectation(
    control: &GeometricAsianCall,
    grid: &TimeGrid,
    spot: Dual,
    rate: Dual,
    vols: &[Dual],
) -> Dual {
    let n = control.fixings as f64;
    let add = |sum: Option<Dual>, x: Dual| Some(sum.map_or(x, |s| s + x));

    let mut integrated: Option<Dual> = None;
    let mut sum_v: Option<Dual> = None;
    let mut weighted_v: Option<Dual> = None;
    let mut sum_t = 0.0;
    let mut i = 0;
    for step in &grid.steps {
        let vol = vols[step.bucket];
        integrated = add(integrated, vol * vol * step.dt);
        if step.fixing {
            i += 1;
            let v = integrated.expect("a fixing ends a step, so variance has accrued");
            sum_t += grid.maturity * i as f64 / n;
            sum_v = add(sum_v, v);
            weighted_v = add(weighted_v, v * (2.0 * (n - i as f64) + 1.0));
        }
    }
    let (sum_v, weighted_v) = (
        sum_v.expect("at least one fixing"),
        weighted_v.expect("at least one fixing"),
    );

    let mean = spot.ln() + (rate * sum_t - sum_v * 0.5) / n;
    let variance = weighted_v / (n * n);
    let sd = variance.sqrt();
    let d1 = (mean - control.strike.ln() + variance) / sd;
    let d2 = d1 - sd;
    let forward = (mean + variance * 0.5).exp();
    (rate * -grid.maturity).exp() * (forward * d1.norm_cdf() - d2.norm_cdf() * control.strike)
}

/// The payoff minus its control, path by path: what the controlled
/// estimator actually samples.
struct MinusControl<'a, P> {
    payoff: &'a P,
    control: GeometricAsianCall,
}

impl<P: Payoff> Payoff for MinusControl<'_, P> {
    fn fixings(&self) -> usize {
        self.payoff.fixings()
    }

    fn value<T: Real>(&self, path: &[T]) -> T {
        self.payoff.value(path) - self.control.value(path)
    }
}
